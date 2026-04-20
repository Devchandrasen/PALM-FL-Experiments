from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dp import StatsDPMechanism
from .latent_stats import LatentStatsPackage, sample_from_stats_package
from .models import PALMFLModel, covariance_regularizer
from .utils import bytes_from_tensors, count_parameters


@dataclass
class ClientConfigView:
    num_classes: int
    latent_dim: int
    local_epochs: int
    real_finetune_epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    prototype_loss_weight: float
    prototype_ce_loss_weight: float
    prototype_ce_temperature: float
    covariance_loss_weight: float
    stat_head_loss_weight: float
    label_smoothing: float
    latent_mixup_alpha: float
    normalize_latent: bool
    stats_max_batches: int | None
    enable_prototype_alignment: bool
    upload_stats: bool
    stat_head_steps: int
    stat_batch_size: int
    stat_prior_mix: float
    stat_missing_class_boost: float
    stat_inverse_hist_power: float
    stat_class_focus_mix: float
    summary_mode: str


class Client:
    def __init__(
        self,
        client_id: int,
        model: PALMFLModel,
        train_loader: DataLoader,
        client_meta: dict,
        dp_mechanism: StatsDPMechanism,
        cfg: Dict,
        device: torch.device,
    ) -> None:
        self.client_id = int(client_id)
        self.model = model
        self.train_loader = train_loader
        self.client_meta = client_meta
        self.dp_mechanism = dp_mechanism
        self.runtime_device = device

        training_cfg = cfg["training"]
        model_cfg = cfg["model"]
        system_cfg = cfg["system"]
        algorithm_cfg = cfg.get("algorithm", {})
        self.cfg = ClientConfigView(
            num_classes=int(client_meta["num_classes"]),
            latent_dim=int(model_cfg["latent_dim"]),
            local_epochs=int(training_cfg.get("local_epochs", 1)),
            real_finetune_epochs=int(training_cfg.get("real_finetune_epochs", 0)),
            lr=float(training_cfg.get("lr", 1e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
            grad_clip=float(training_cfg.get("grad_clip", 5.0)),
            prototype_loss_weight=float(training_cfg.get("prototype_loss_weight", 0.0)),
            prototype_ce_loss_weight=float(training_cfg.get("prototype_ce_loss_weight", 0.0)),
            prototype_ce_temperature=float(training_cfg.get("prototype_ce_temperature", 0.2)),
            covariance_loss_weight=float(training_cfg.get("covariance_loss_weight", 0.0)),
            stat_head_loss_weight=float(training_cfg.get("stat_head_loss_weight", 1.0)),
            label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
            latent_mixup_alpha=float(training_cfg.get("latent_mixup_alpha", 0.0)),
            normalize_latent=bool(training_cfg.get("normalize_latent", False)),
            stats_max_batches=training_cfg.get("stats_max_batches", None),
            enable_prototype_alignment=bool(training_cfg.get("enable_prototype_alignment", False)),
            upload_stats=bool(training_cfg.get("upload_stats", True)),
            stat_head_steps=int(training_cfg.get("stat_head_steps", 0)),
            stat_batch_size=int(system_cfg.get("stat_batch_size", system_cfg.get("synthetic_batch_size", 128))),
            stat_prior_mix=float(training_cfg.get("stat_prior_mix", 0.0)),
            stat_missing_class_boost=float(training_cfg.get("stat_missing_class_boost", 0.0)),
            stat_inverse_hist_power=float(training_cfg.get("stat_inverse_hist_power", 0.0)),
            stat_class_focus_mix=float(training_cfg.get("stat_class_focus_mix", 0.0)),
            summary_mode=str(algorithm_cfg.get("summary_mode", "full")).lower(),
        )

        self.optimizer_all = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.optimizer_head = torch.optim.Adam(
            self.model.head.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
        self.local_label_hist = torch.tensor(client_meta["label_hist"], dtype=torch.float32)

    @property
    def num_parameters(self) -> int:
        return count_parameters(self.model)

    @property
    def num_samples(self) -> int:
        return int(self.client_meta["num_samples"])

    @property
    def unique_labels(self) -> int:
        return int(self.client_meta["unique_labels"])

    def estimate_local_steps(self) -> int:
        return int(len(self.train_loader) * (self.cfg.local_epochs + self.cfg.real_finetune_epochs) + self.cfg.stat_head_steps)

    def _stat_class_probs(self, device: torch.device, package: LatentStatsPackage | None = None) -> torch.Tensor:
        hist = self.local_label_hist.to(device)
        if hist.sum() <= 0:
            local = torch.ones_like(hist) / hist.numel()
        else:
            local = hist / hist.sum().clamp_min(1e-12)

        if package is None or self.cfg.stat_prior_mix <= 0:
            base = local
        else:
            global_priors = package.class_priors.to(device)
            if global_priors.sum() <= 0:
                global_priors = torch.ones_like(global_priors) / global_priors.numel()
            else:
                global_priors = global_priors / global_priors.sum().clamp_min(1e-12)

            if self.cfg.stat_missing_class_boost > 0:
                missing_mask = (hist <= 0).float()
                boost = 1.0 + self.cfg.stat_missing_class_boost * missing_mask
                global_priors = global_priors * boost
                global_priors = global_priors / global_priors.sum().clamp_min(1e-12)

            mix = float(self.cfg.stat_prior_mix)
            base = (1.0 - mix) * local + mix * global_priors

        if self.cfg.stat_inverse_hist_power > 0 or self.cfg.stat_class_focus_mix > 0:
            inv = (hist + 1.0).pow(-float(self.cfg.stat_inverse_hist_power))
            if self.cfg.stat_missing_class_boost > 0:
                inv = inv * (1.0 + self.cfg.stat_missing_class_boost * (hist <= 0).float())
            if inv.sum() > 0:
                inv = inv / inv.sum().clamp_min(1e-12)
                focus_mix = float(self.cfg.stat_class_focus_mix)
                base = (1.0 - focus_mix) * base + focus_mix * inv

        return base / base.sum().clamp_min(1e-12)

    def _latent_mixup(self, z: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        targets = torch.zeros(y.size(0), self.cfg.num_classes, device=y.device)
        targets.scatter_(1, y.view(-1, 1), 1.0)
        if self.cfg.label_smoothing > 0:
            targets = targets * (1.0 - self.cfg.label_smoothing) + self.cfg.label_smoothing / self.cfg.num_classes
        if self.cfg.latent_mixup_alpha <= 0:
            return z, targets
        beta = torch.distributions.Beta(self.cfg.latent_mixup_alpha, self.cfg.latent_mixup_alpha)
        lam = beta.sample((z.size(0),)).to(z.device)
        perm = torch.randperm(z.size(0), device=z.device)
        lam_z = lam.view(-1, 1)
        z_mix = lam_z * z + (1.0 - lam_z) * z[perm]
        y_mix = lam_z * targets + (1.0 - lam_z) * targets[perm]
        return z_mix, y_mix

    def _train_stat_head(self, package: LatentStatsPackage, stat_head_steps: int) -> dict:
        if stat_head_steps <= 0:
            return {
                "stat_head_loss": 0.0,
                "stat_head_accuracy": 0.0,
            }
        self.model.train()
        losses = []
        total = 0
        correct = 0
        class_probs = self._stat_class_probs(self.runtime_device, package)
        for _ in range(int(stat_head_steps)):
            z, y = sample_from_stats_package(
                package=package,
                batch_size=self.cfg.stat_batch_size,
                num_classes=self.cfg.num_classes,
                device=self.runtime_device,
                class_probs=class_probs,
            )
            z_mix, soft_targets = self._latent_mixup(z, y)
            self.optimizer_head.zero_grad(set_to_none=True)
            logits = self.model.classify_latent(z_mix)
            if self.cfg.latent_mixup_alpha > 0:
                per = -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
                loss = self.cfg.stat_head_loss_weight * per.mean()
            else:
                loss = self.cfg.stat_head_loss_weight * F.cross_entropy(
                    logits,
                    y,
                    label_smoothing=self.cfg.label_smoothing,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), self.cfg.grad_clip)
            self.optimizer_head.step()
            losses.append(float(loss.item()))
            preds = logits.argmax(dim=-1)
            total += y.numel()
            correct += int((preds == y).sum().item())
        return {
            "stat_head_loss": float(sum(losses) / max(len(losses), 1)),
            "stat_head_accuracy": float(correct / max(total, 1)),
        }

    def _prototype_loss(self, z: torch.Tensor, y: torch.Tensor, package: LatentStatsPackage | None) -> torch.Tensor:
        if package is None or (not self.cfg.enable_prototype_alignment) or self.cfg.prototype_loss_weight <= 0:
            return z.new_tensor(0.0)
        prototypes = package.prototypes.to(z.device)
        mask = package.mask.to(z.device)[y]
        if not mask.any():
            return z.new_tensor(0.0)
        z_sel = F.normalize(z[mask], dim=-1)
        proto_sel = F.normalize(prototypes[y[mask]], dim=-1)
        return (z_sel - proto_sel).pow(2).sum(dim=-1).mean()

    def _prototype_ce_loss(self, z: torch.Tensor, y: torch.Tensor, package: LatentStatsPackage | None) -> torch.Tensor:
        if package is None or (not self.cfg.enable_prototype_alignment) or self.cfg.prototype_ce_loss_weight <= 0:
            return z.new_tensor(0.0)
        prototypes = package.prototypes.to(z.device)
        class_mask = package.mask.to(z.device)
        sample_mask = class_mask[y]
        if not sample_mask.any() or int(class_mask.sum().item()) < 2:
            return z.new_tensor(0.0)

        z_norm = F.normalize(z[sample_mask], dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        temperature = max(float(self.cfg.prototype_ce_temperature), 1e-4)
        logits = (z_norm @ proto_norm.T) / temperature
        logits[:, ~class_mask] = -1e9
        return F.cross_entropy(logits, y[sample_mask])

    def _train_real(self, package: LatentStatsPackage | None, epochs: int) -> dict:
        self.model.train()
        losses = []
        correct = 0
        total = 0
        ce_losses = []
        proto_losses = []
        proto_ce_losses = []
        cov_losses = []
        for _ in range(int(epochs)):
            for x, y in self.train_loader:
                x = x.to(self.runtime_device, non_blocking=True)
                y = y.to(self.runtime_device, non_blocking=True)
                self.optimizer_all.zero_grad(set_to_none=True)
                logits, z = self.model(x, return_latent=True, normalize_latent=self.cfg.normalize_latent)
                ce = self.ce_loss(logits, y)
                proto = self._prototype_loss(z, y, package)
                proto_ce = self._prototype_ce_loss(z, y, package)
                cov = covariance_regularizer(z)
                loss = (
                    ce
                    + self.cfg.prototype_loss_weight * proto
                    + self.cfg.prototype_ce_loss_weight * proto_ce
                    + self.cfg.covariance_loss_weight * cov
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer_all.step()

                losses.append(float(loss.item()))
                ce_losses.append(float(ce.item()))
                proto_losses.append(float(proto.item()))
                proto_ce_losses.append(float(proto_ce.item()))
                cov_losses.append(float(cov.item()))
                preds = logits.argmax(dim=-1)
                total += y.numel()
                correct += int((preds == y).sum().item())
        return {
            "avg_loss": float(sum(losses) / max(len(losses), 1)),
            "real_ce_loss": float(sum(ce_losses) / max(len(ce_losses), 1)),
            "real_proto_loss": float(sum(proto_losses) / max(len(proto_losses), 1)),
            "real_proto_ce_loss": float(sum(proto_ce_losses) / max(len(proto_ce_losses), 1)),
            "real_cov_loss": float(sum(cov_losses) / max(len(cov_losses), 1)),
            "train_accuracy": float(correct / max(total, 1)),
        }

    @torch.no_grad()
    def compute_private_stats(self) -> dict:
        if not self.cfg.upload_stats:
            return {
                "client_id": self.client_id,
                "upload_bytes": 0,
            }
        self.model.eval()
        self.model.to(self.runtime_device)
        counts = torch.zeros(self.cfg.num_classes, dtype=torch.float32, device=self.runtime_device)
        sums = torch.zeros(self.cfg.num_classes, self.cfg.latent_dim, dtype=torch.float32, device=self.runtime_device)
        sum_sq = torch.zeros(self.cfg.num_classes, self.cfg.latent_dim, dtype=torch.float32, device=self.runtime_device)

        for batch_idx, (x, y) in enumerate(self.train_loader):
            if self.cfg.stats_max_batches is not None and batch_idx >= int(self.cfg.stats_max_batches):
                break
            x = x.to(self.runtime_device, non_blocking=True)
            y = y.to(self.runtime_device, non_blocking=True)
            z = self.model.encode_latent(x, normalize=self.cfg.normalize_latent)
            for cls in y.unique():
                cls_int = int(cls.item())
                mask = y == cls
                z_cls = z[mask]
                counts[cls_int] += z_cls.size(0)
                sums[cls_int] += z_cls.sum(dim=0)
                sum_sq[cls_int] += z_cls.pow(2).sum(dim=0)

        counts_safe = counts.clamp_min(1.0).unsqueeze(-1)
        means = sums / counts_safe
        variances = sum_sq / counts_safe - means.pow(2)
        variances = torch.clamp(variances, min=1e-6)

        private = self.dp_mechanism.privatize(
            means=means.detach().cpu(),
            variances=variances.detach().cpu(),
            counts=counts.detach().cpu(),
        )
        summary_mode = str(self.cfg.summary_mode).lower()
        if summary_mode == "full":
            upload_bytes = bytes_from_tensors(private["means"], private["variances"], private["counts"], private["mask"])
            return {
                "client_id": self.client_id,
                "summary_mode": summary_mode,
                "means": private["means"],
                "variances": private["variances"],
                "counts": private["counts"],
                "mask": private["mask"],
                "upload_bytes": upload_bytes,
            }
        if summary_mode == "histogram_only":
            upload_bytes = bytes_from_tensors(private["counts"], private["mask"])
            return {
                "client_id": self.client_id,
                "summary_mode": summary_mode,
                "counts": private["counts"],
                "mask": private["mask"],
                "upload_bytes": upload_bytes,
            }
        if summary_mode == "mean_only":
            availability = (counts.detach().cpu() > 0).float()
            private = self.dp_mechanism.privatize(
                means=means.detach().cpu(),
                variances=torch.ones_like(variances.detach().cpu()),
                counts=availability,
            )
            upload_bytes = bytes_from_tensors(private["means"], private["counts"], private["mask"])
            return {
                "client_id": self.client_id,
                "summary_mode": summary_mode,
                "means": private["means"],
                "counts": private["counts"],
                "mask": private["mask"],
                "upload_bytes": upload_bytes,
            }
        raise ValueError(f"Unknown algorithm.summary_mode: {summary_mode}")

    def local_update(
        self,
        round_idx: int,
        package: LatentStatsPackage | None,
        stat_head_steps: int = 0,
    ) -> dict:
        self.model.to(self.runtime_device)
        start = time.perf_counter()

        stat_head_log = {
            "stat_head_loss": 0.0,
            "stat_head_accuracy": 0.0,
        }
        if package is not None and stat_head_steps > 0:
            stat_head_log = self._train_stat_head(package, stat_head_steps=stat_head_steps)

        real_log = self._train_real(package=package, epochs=self.cfg.local_epochs)
        if self.cfg.real_finetune_epochs > 0:
            finetune_log = self._train_real(package=None, epochs=self.cfg.real_finetune_epochs)
            real_log["avg_loss"] = finetune_log["avg_loss"]
            real_log["train_accuracy"] = finetune_log["train_accuracy"]

        stats_payload = self.compute_private_stats()
        duration = time.perf_counter() - start
        payload = {
            **stats_payload,
            **real_log,
            **stat_head_log,
            "client_id": self.client_id,
            "arch_name": self.model.arch_name,
            "num_parameters": self.num_parameters,
            "num_samples": self.num_samples,
            "duration_s": duration,
            "round_idx": round_idx,
            "stat_head_steps_used": int(stat_head_steps),
        }
        self.model.to(torch.device("cpu"))
        return payload
