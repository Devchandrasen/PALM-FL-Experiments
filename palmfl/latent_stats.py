from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from .utils import bytes_from_tensors


@dataclass
class LatentStatsPackage:
    prototypes: torch.Tensor
    variances: torch.Tensor
    counts: torch.Tensor
    mask: torch.Tensor
    class_priors: torch.Tensor
    temperature: float = 1.0
    package_mode: str = "full"

    def num_bytes(self) -> int:
        if self.package_mode == "counts_only":
            return bytes_from_tensors(
                self.counts,
                self.mask,
                self.class_priors,
            )
        return bytes_from_tensors(
            self.prototypes,
            self.variances,
            self.counts,
            self.mask,
            self.class_priors,
        )


class PrivateLatentStatsServer:
    """Aggregates differentially private class-wise latent statistics.

    Unlike the v7 generator path, this module does not learn a separate generator.
    It simply maintains global class prototypes and diagonal variances in the shared
    latent space.
    """

    def __init__(
        self,
        num_classes: int,
        latent_dim: int,
        min_variance: float = 0.05,
        ema_momentum: float = 0.7,
    ) -> None:
        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)
        self.min_variance = float(min_variance)
        self.ema_momentum = float(ema_momentum)

        self.prototypes = torch.zeros(self.num_classes, self.latent_dim, dtype=torch.float32)
        self.variances = torch.ones(self.num_classes, self.latent_dim, dtype=torch.float32)
        self.counts = torch.zeros(self.num_classes, dtype=torch.float32)
        self.mask = torch.zeros(self.num_classes, dtype=torch.bool)
        self.ready = False

    def update(self, payloads: Sequence[dict]) -> None:
        if not payloads:
            return

        new_means = torch.zeros_like(self.prototypes)
        new_vars = torch.zeros_like(self.variances)
        new_counts = torch.zeros_like(self.counts)
        mean_observed = torch.zeros_like(self.mask)
        count_observed = torch.zeros_like(self.mask)

        for cls in range(self.num_classes):
            cls_counts: list[float] = []
            cls_means: list[torch.Tensor] = []
            cls_vars: list[torch.Tensor] = []
            cls_count_only: list[float] = []
            for payload in payloads:
                if "counts" not in payload:
                    continue
                if "mask" in payload and not bool(payload["mask"][cls].item()):
                    continue
                count = float(payload["counts"][cls].item())
                if count <= 0.0:
                    continue
                cls_count_only.append(count)
                if "means" not in payload:
                    continue
                cls_counts.append(count)
                cls_means.append(payload["means"][cls])
                if "variances" in payload:
                    cls_vars.append(payload["variances"][cls])
                else:
                    cls_vars.append(torch.full_like(payload["means"][cls], self.min_variance))

            if cls_counts:
                counts_t = torch.tensor(cls_counts, dtype=torch.float32)
                means_t = torch.stack(cls_means, dim=0)
                vars_t = torch.stack(cls_vars, dim=0)
                total = counts_t.sum().clamp_min(1e-6)
                mean = (counts_t[:, None] * means_t).sum(dim=0) / total
                second = (counts_t[:, None] * (vars_t + means_t.pow(2))).sum(dim=0) / total
                var = torch.clamp(second - mean.pow(2), min=self.min_variance)
                new_means[cls] = mean
                new_vars[cls] = var
                new_counts[cls] = total
                mean_observed[cls] = True
                count_observed[cls] = True
            elif cls_count_only:
                new_counts[cls] = torch.tensor(cls_count_only, dtype=torch.float32).sum()
                count_observed[cls] = True

        observed = new_counts > 0
        if self.ready:
            m = self.ema_momentum
            self.prototypes[mean_observed] = m * self.prototypes[mean_observed] + (1.0 - m) * new_means[mean_observed]
            self.variances[mean_observed] = m * self.variances[mean_observed] + (1.0 - m) * new_vars[mean_observed]
            self.counts[count_observed] = m * self.counts[count_observed] + (1.0 - m) * new_counts[count_observed]
        else:
            self.prototypes[mean_observed] = new_means[mean_observed]
            self.variances[mean_observed] = new_vars[mean_observed]
            self.counts[count_observed] = new_counts[count_observed]

        self.counts = self.counts.clamp_min(0.0)
        self.variances = torch.clamp(self.variances, min=self.min_variance)
        self.mask = self.counts > 0
        self.ready = bool(self.mask.any())

    def class_coverage(self) -> float:
        if not self.ready:
            return 0.0
        return float(self.mask.float().mean().item())

    def class_deficit(self) -> torch.Tensor:
        if not self.ready or self.counts.sum() <= 0:
            return torch.ones(self.num_classes, dtype=torch.float32) / max(self.num_classes, 1)
        priors = self.counts / self.counts.sum().clamp_min(1e-12)
        target = torch.ones_like(priors) / max(self.num_classes, 1)
        deficit = torch.clamp(target - priors, min=0.0)
        if deficit.sum() <= 0:
            return target
        return deficit / deficit.sum().clamp_min(1e-12)

    def export_package(self, temperature: float = 1.0, package_mode: str = "full") -> LatentStatsPackage | None:
        if not self.ready:
            return None
        package_mode = str(package_mode).lower()
        if package_mode not in {"full", "counts_only"}:
            raise ValueError(f"Unknown stats package mode: {package_mode}")
        priors = self.counts.clone()
        if priors.sum() <= 0:
            priors = torch.ones_like(priors) / priors.numel()
        else:
            priors = priors / priors.sum().clamp_min(1e-12)
        return LatentStatsPackage(
            prototypes=self.prototypes.clone(),
            variances=self.variances.clone(),
            counts=self.counts.clone(),
            mask=self.mask.clone(),
            class_priors=priors,
            temperature=float(temperature),
            package_mode=package_mode,
        )

    def state_dict(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "latent_dim": self.latent_dim,
            "min_variance": self.min_variance,
            "ema_momentum": self.ema_momentum,
            "prototypes": self.prototypes,
            "variances": self.variances,
            "counts": self.counts,
            "mask": self.mask,
            "ready": self.ready,
        }

    def load_state_dict(self, state: dict) -> None:
        self.prototypes = state["prototypes"]
        self.variances = state["variances"]
        self.counts = state["counts"]
        self.mask = state["mask"]
        self.ready = bool(state["ready"])


@torch.no_grad()
def sample_from_stats_package(
    package: LatentStatsPackage,
    batch_size: int,
    num_classes: int,
    device: torch.device,
    class_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if class_probs is None:
        class_probs = package.class_priors.to(device)
    else:
        class_probs = class_probs.to(device)
    if class_probs.sum() <= 0:
        class_probs = torch.ones(num_classes, device=device) / num_classes
    else:
        class_probs = class_probs / class_probs.sum().clamp_min(1e-12)

    y = torch.multinomial(class_probs, num_samples=batch_size, replacement=True)
    if package.package_mode == "counts_only":
        mu = torch.zeros(batch_size, package.prototypes.size(1), device=device)
        var = torch.ones_like(mu)
    else:
        mu = package.prototypes.to(device)[y]
        var = torch.clamp(package.variances.to(device)[y], min=1e-6)
    z = mu + torch.randn_like(mu) * torch.sqrt(var * float(package.temperature))
    return z, y
