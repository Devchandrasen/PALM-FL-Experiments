from __future__ import annotations

import math
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .client import Client
from .dp import GaussianZCDPAccountant
from .latent_stats import PrivateLatentStatsServer
from .metrics import evaluate_model, summarize_metric_list
from .scheduler import MobileAwareScheduler
from .utils import append_jsonl, ensure_dir, save_json


class PALMFLServer:
    def __init__(
        self,
        cfg: Dict,
        clients: Dict[int, Client],
        test_loader,
        experiment_dir: str | Path,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.clients = clients
        self.test_loader = test_loader
        self.device = device
        self.experiment_dir = ensure_dir(experiment_dir)
        self.checkpoint_dir = ensure_dir(self.experiment_dir / "checkpoints")
        self.metrics_path = self.experiment_dir / "metrics.jsonl"
        self.rng = random.Random(int(cfg.get("seed", 0)))

        num_classes = int(next(iter(self.clients.values())).cfg.num_classes)
        latent_dim = int(cfg["model"]["latent_dim"])
        training_cfg = cfg["training"]
        system_cfg = cfg["system"]

        self.stats_server = PrivateLatentStatsServer(
            num_classes=num_classes,
            latent_dim=latent_dim,
            min_variance=float(training_cfg.get("min_variance", cfg.get("stats_server", {}).get("min_variance", 0.05))),
            ema_momentum=float(cfg.get("stats_server", {}).get("ema_momentum", 0.7)),
        )
        summary_mode = str(cfg.get("algorithm", {}).get("summary_mode", "full")).lower()
        releases_default = {"histogram_only": 1, "mean_only": 2}.get(summary_mode, 3)
        self.dp_accountant = GaussianZCDPAccountant(
            noise_multiplier=float(cfg["dp"].get("noise_multiplier", 1.0)),
            delta=float(cfg["dp"].get("delta", 1e-5)),
            releases_per_round=int(cfg["dp"].get("releases_per_round", releases_default)),
            enabled=bool(cfg["dp"].get("enable", True)),
        )
        self.scheduler = MobileAwareScheduler(
            num_clients=int(system_cfg["num_clients"]),
            num_classes=num_classes,
            scheduler_cfg=cfg["scheduler"],
            seed=int(cfg.get("seed", 0)),
        )
        for cid, client in self.clients.items():
            self.scheduler.register_client(
                client_id=cid,
                num_samples=client.num_samples,
                unique_labels=client.unique_labels,
                model_num_params=client.num_parameters,
            )

        self.history: List[dict] = []

    def _variant_flags(self) -> dict:
        algorithm_cfg = self.cfg.get("algorithm", {})
        variant = str(algorithm_cfg.get("variant", "stats_transfer")).lower()
        summary_mode = str(algorithm_cfg.get("summary_mode", "full")).lower()
        transfer_package_mode = str(algorithm_cfg.get("transfer_package_mode", "full")).lower()
        flags = {
            "variant": variant,
            "summary_mode": summary_mode,
            "transfer_package_mode": transfer_package_mode,
            "upload_stats": True,
            "download_package": True,
            "enable_prototype_alignment": bool(self.cfg["training"].get("enable_prototype_alignment", False)),
            "enable_stat_head": bool(self.cfg["training"].get("enable_stat_head", False)),
        }
        if variant == "local_only":
            flags.update(upload_stats=False, download_package=False, enable_prototype_alignment=False, enable_stat_head=False)
        elif variant == "stats_upload_only":
            flags.update(upload_stats=True, download_package=False, enable_prototype_alignment=False, enable_stat_head=False)
        elif variant == "stats_transfer":
            pass
        elif variant == "histogram_only":
            flags.update(upload_stats=True, download_package=False, enable_prototype_alignment=False, enable_stat_head=False)
            flags["summary_mode"] = "histogram_only"
        elif variant == "mean_only":
            flags.update(upload_stats=True, download_package=False, enable_prototype_alignment=False, enable_stat_head=False)
            flags["summary_mode"] = "mean_only"
        elif variant == "count_only_transfer":
            flags.update(upload_stats=True, download_package=True, enable_prototype_alignment=False, enable_stat_head=False)
            flags["summary_mode"] = "histogram_only"
            flags["transfer_package_mode"] = "counts_only"
        else:
            raise ValueError(f"Unknown algorithm.variant: {variant}")
        if flags["summary_mode"] not in {"full", "histogram_only", "mean_only"}:
            raise ValueError(f"Unknown algorithm.summary_mode: {flags['summary_mode']}")
        if flags["transfer_package_mode"] not in {"full", "counts_only"}:
            raise ValueError(f"Unknown algorithm.transfer_package_mode: {flags['transfer_package_mode']}")
        if flags["transfer_package_mode"] == "counts_only":
            flags.update(enable_prototype_alignment=False, enable_stat_head=False)
        return flags

    def _estimated_upload_mb(self, flags: dict) -> float:
        if not flags["upload_stats"]:
            return 0.0
        client = next(iter(self.clients.values()))
        num_classes = client.cfg.num_classes
        latent_dim = client.cfg.latent_dim
        summary_mode = flags["summary_mode"]
        if summary_mode == "full":
            bytes_est = num_classes * latent_dim * 4 * 2 + num_classes * 4 + num_classes
        elif summary_mode == "histogram_only":
            bytes_est = num_classes * 4 + num_classes
        elif summary_mode == "mean_only":
            bytes_est = num_classes * latent_dim * 4 + num_classes * 4 + num_classes
        else:
            raise ValueError(f"Unknown summary mode: {summary_mode}")
        return bytes_est / (1024**2)

    def _estimated_download_mb(self, download_package: bool, flags: dict) -> float:
        if not download_package:
            return 0.0
        client = next(iter(self.clients.values()))
        num_classes = client.cfg.num_classes
        latent_dim = client.cfg.latent_dim
        if flags["transfer_package_mode"] == "counts_only":
            bytes_est = num_classes * 4 + num_classes + num_classes * 4
        else:
            bytes_est = num_classes * latent_dim * 4 * 2
            bytes_est += num_classes * 4
            bytes_est += num_classes
            bytes_est += num_classes * 4
        return bytes_est / (1024**2)

    def _max_clients_per_round(self) -> int:
        system_cfg = self.cfg["system"]
        return max(1, int(math.ceil(float(system_cfg.get("participation_rate", 0.2)) * int(system_cfg["num_clients"]))))

    def _mean_local_steps(self, head_steps: int = 0) -> int:
        steps = [client.estimate_local_steps() + max(0, head_steps - client.cfg.stat_head_steps) for client in self.clients.values()]
        return max(1, int(round(sum(steps) / len(steps))))

    def _current_head_steps(self, round_idx: int, warmup_rounds: int, package_available: bool) -> int:
        training_cfg = self.cfg["training"]
        if not package_available or not bool(training_cfg.get("enable_stat_head", False)):
            return 0
        target_steps = int(training_cfg.get("stat_head_steps", 0))
        if target_steps <= 0:
            return 0
        min_steps = max(0, int(training_cfg.get("stat_head_min_steps", 1)))
        min_steps = min(min_steps, target_steps)
        ramp_rounds = max(0, int(training_cfg.get("stat_head_ramp_rounds", 0)))
        if ramp_rounds <= 0:
            return target_steps
        post_warmup_idx = max(0, round_idx - warmup_rounds)
        frac = min(1.0, float(post_warmup_idx + 1) / float(ramp_rounds))
        steps = min_steps + frac * max(0, target_steps - min_steps)
        return int(max(0, min(target_steps, round(steps))))

    def _select_clients(
        self,
        round_idx: int,
        upload_mb: float,
        download_mb: float,
        local_steps: int,
        class_deficit: np.ndarray | None = None,
    ) -> tuple[list[int], dict]:
        policy = str(self.cfg["scheduler"].get("policy", "mobile")).lower()
        max_clients = self._max_clients_per_round()
        if policy == "mobile":
            return self.scheduler.select_clients(
                round_idx=round_idx,
                max_clients=max_clients,
                upload_mb=upload_mb,
                download_mb=download_mb,
                local_steps=local_steps,
                class_deficit=class_deficit,
            )
        client_ids = sorted(self.clients.keys())
        if policy == "all":
            selected = client_ids
        elif policy == "random":
            selected = self.rng.sample(client_ids, k=min(max_clients, len(client_ids)))
            selected.sort()
        else:
            raise ValueError(f"Unknown scheduler.policy: {policy}")

        info = {}
        for cid in selected:
            costs = self.scheduler.estimate_costs(cid, upload_mb=upload_mb, download_mb=download_mb, local_steps=local_steps)
            costs["utility"] = 0.0
            info[cid] = costs
        return selected, info

    def evaluate(
        self,
        round_idx: int,
        max_batches: int | None = None,
        return_details: bool = False,
    ) -> dict | tuple[dict, list[dict]]:
        eval_cfg = self.cfg["system"]
        eval_num_clients = int(eval_cfg.get("eval_num_clients", -1))
        normalize_latent = bool(self.cfg["training"].get("normalize_latent", False))
        num_classes = int(next(iter(self.clients.values())).cfg.num_classes)
        if max_batches is None:
            max_batches = int(eval_cfg.get("eval_max_batches", 0)) or None

        all_client_ids = sorted(self.clients.keys())
        if eval_num_clients <= 0 or eval_num_clients >= len(all_client_ids):
            eval_ids = all_client_ids
        else:
            rng = np.random.default_rng(int(self.cfg.get("seed", 0)) + round_idx)
            eval_ids = sorted(rng.choice(all_client_ids, size=eval_num_clients, replace=False).tolist())

        metrics = []
        details = []
        for cid in eval_ids:
            client = self.clients[cid]
            client.model.to(self.device)
            result = evaluate_model(
                model=client.model,
                loader=self.test_loader,
                device=self.device,
                num_classes=num_classes,
                normalize_latent=normalize_latent,
                max_batches=max_batches,
            )
            client.model.to(torch.device("cpu"))
            metrics.append(result)
            details.append(
                {
                    "client_id": int(cid),
                    "arch_name": str(client.model.arch_name),
                    "num_samples": int(client.num_samples),
                    "unique_labels": int(client.unique_labels),
                    **{k: float(v) for k, v in result.items()},
                }
            )
        summary = summarize_metric_list(metrics)
        if return_details:
            return summary, details
        return summary

    def _save_checkpoint(self, round_idx: int) -> None:
        ckpt = {
            "round_idx": round_idx,
            "config": self.cfg,
            "stats_server_state": self.stats_server.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "dp_accountant": {
                "participations": self.dp_accountant.participations,
                "noise_multiplier": self.dp_accountant.noise_multiplier,
                "delta": self.dp_accountant.delta,
                "enabled": self.dp_accountant.enabled,
            },
            "client_states": {cid: client.model.state_dict() for cid, client in self.clients.items()},
            "history": self.history,
        }
        path = self.checkpoint_dir / f"round_{round_idx:04d}.pt"
        torch.save(ckpt, path)

    def run(self) -> dict:
        system_cfg = self.cfg["system"]
        training_cfg = self.cfg["training"]
        flags = self._variant_flags()
        rounds = int(system_cfg["rounds"])
        warmup_rounds = int(system_cfg.get("warmup_rounds", 0))
        eval_every = max(1, int(system_cfg.get("eval_every", 5)))
        eval_initial = bool(system_cfg.get("eval_initial", True))
        eval_max_batches = int(system_cfg.get("eval_max_batches", 0)) or None
        final_eval_max_batches = int(system_cfg.get("final_eval_max_batches", system_cfg.get("eval_max_batches", 0))) or None
        save_ckpt = bool(self.cfg.get("logging", {}).get("save_checkpoints", True))
        ckpt_every = int(self.cfg.get("logging", {}).get("checkpoint_every", 10))
        min_coverage = float(self.cfg.get("stats_server", {}).get("min_class_coverage_for_transfer", 0.0))
        package_temperature = float(self.cfg.get("stats_server", {}).get("temperature", 1.0))

        print("=" * 88)
        print("Starting PALM-FL v8 pivot training")
        print(f"Variant: {flags['variant']} | Rounds: {rounds} | Warmup rounds: {warmup_rounds} | Clients: {len(self.clients)}")
        print(f"Summary mode: {flags['summary_mode']} | Transfer package: {flags['transfer_package_mode']}")
        print(f"Scheduler policy: {self.cfg['scheduler'].get('policy', 'mobile')}")
        print("=" * 88)

        for round_idx in range(rounds):
            wall_start = time.perf_counter()
            coverage = self.stats_server.class_coverage()
            package_available = (
                flags["download_package"]
                and (round_idx >= warmup_rounds)
                and self.stats_server.ready
                and (coverage >= min_coverage)
            )
            stat_head_steps = self._current_head_steps(round_idx, warmup_rounds, package_available)
            upload_mb_est = self._estimated_upload_mb(flags)
            download_mb_est = self._estimated_download_mb(package_available, flags)
            deficit_weight = float(self.cfg["scheduler"].get("utility_weights", {}).get("deficit", 0.0))
            class_deficit = None
            if flags["upload_stats"] and deficit_weight > 0 and self.stats_server.ready:
                class_deficit = self.stats_server.class_deficit().detach().cpu().numpy()

            selected_ids, scheduler_info = self._select_clients(
                round_idx=round_idx,
                upload_mb=upload_mb_est,
                download_mb=download_mb_est,
                local_steps=self._mean_local_steps(stat_head_steps),
                class_deficit=class_deficit,
            )

            package = (
                self.stats_server.export_package(
                    temperature=package_temperature,
                    package_mode=flags["transfer_package_mode"],
                )
                if package_available
                else None
            )
            round_upload_bytes = 0
            round_download_bytes = 0
            local_wall_times = []
            updates: Dict[int, dict] = {}

            for cid in selected_ids:
                client = self.clients[cid]
                client.cfg.enable_prototype_alignment = flags["enable_prototype_alignment"]
                client.cfg.upload_stats = flags["upload_stats"]
                client.cfg.summary_mode = flags["summary_mode"]
                update = client.local_update(
                    round_idx=round_idx,
                    package=package,
                    stat_head_steps=stat_head_steps if flags["enable_stat_head"] else 0,
                )
                update.update(scheduler_info.get(cid, {}))
                updates[cid] = update
                round_upload_bytes += int(update.get("upload_bytes", 0))
                if package is not None:
                    round_download_bytes += package.num_bytes()
                local_wall_times.append(float(update["duration_s"]))
                if flags["upload_stats"]:
                    self.dp_accountant.step(cid)

            if flags["upload_stats"]:
                self.stats_server.update(list(updates.values()))
            self.scheduler.update_after_round(round_idx=round_idx, updates=updates)

            if flags["upload_stats"]:
                privacy = self.dp_accountant.summary()
            else:
                privacy = {"max_epsilon": 0.0, "mean_epsilon": 0.0, "num_participating_clients": 0}

            record = {
                "round_idx": round_idx,
                "variant": flags["variant"],
                "summary_mode": flags["summary_mode"],
                "transfer_package_mode": flags["transfer_package_mode"],
                "selected_clients": selected_ids,
                "num_selected_clients": len(selected_ids),
                "round_upload_mb": round_upload_bytes / (1024**2),
                "round_download_mb": round_download_bytes / (1024**2),
                "predicted_round_time_s": max((updates[cid].get("predicted_time_s", 0.0) for cid in updates), default=0.0),
                "predicted_round_energy_j": sum(updates[cid].get("predicted_energy_j", 0.0) for cid in updates),
                "mean_client_wall_time_s": statistics.mean(local_wall_times) if local_wall_times else 0.0,
                "package_ready": self.stats_server.ready,
                "package_class_coverage": coverage,
                "stats_upload_enabled": flags["upload_stats"],
                "stats_transfer_enabled": package_available,
                "prototype_transfer_enabled": bool(flags["enable_prototype_alignment"] and package_available),
                "stat_head_transfer_enabled": bool(flags["enable_stat_head"] and package_available and stat_head_steps > 0),
                "stat_head_steps_used": int(stat_head_steps if flags["enable_stat_head"] else 0),
                "deficit_scheduler_enabled": bool(class_deficit is not None),
                "dp_enabled": bool(self.dp_accountant.enabled),
                "accounting_releases_per_round": int(self.dp_accountant.releases_per_round),
                **privacy,
            }
            if updates:
                record["mean_train_accuracy"] = float(sum(u["train_accuracy"] for u in updates.values()) / len(updates))
                record["mean_train_loss"] = float(sum(u["avg_loss"] for u in updates.values()) / len(updates))
                record["mean_stat_head_accuracy"] = float(sum(u.get("stat_head_accuracy", 0.0) for u in updates.values()) / len(updates))
                record["mean_stat_head_loss"] = float(sum(u.get("stat_head_loss", 0.0) for u in updates.values()) / len(updates))
                record["mean_real_proto_loss"] = float(sum(u.get("real_proto_loss", 0.0) for u in updates.values()) / len(updates))
                record["mean_real_proto_ce_loss"] = float(sum(u.get("real_proto_ce_loss", 0.0) for u in updates.values()) / len(updates))
                record["mean_real_cov_loss"] = float(sum(u.get("real_cov_loss", 0.0) for u in updates.values()) / len(updates))

            eval_ran = False
            should_eval = (round_idx == rounds - 1) or (
                round_idx % eval_every == 0 and (eval_initial or round_idx > 0)
            )
            if should_eval:
                current_eval_max_batches = final_eval_max_batches if round_idx == rounds - 1 else eval_max_batches
                if round_idx == rounds - 1 and bool(system_cfg.get("save_final_client_metrics", True)):
                    eval_metrics, eval_details = self.evaluate(
                        round_idx,
                        max_batches=current_eval_max_batches,
                        return_details=True,
                    )
                    save_json(self.experiment_dir / "final_client_metrics.json", eval_details)
                else:
                    eval_metrics = self.evaluate(round_idx, max_batches=current_eval_max_batches)
                record.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                record["eval_max_batches"] = 0 if current_eval_max_batches is None else int(current_eval_max_batches)
                record["eval_is_final"] = bool(round_idx == rounds - 1)
                eval_ran = True
            record["eval_ran"] = eval_ran
            record["wall_clock_s"] = time.perf_counter() - wall_start
            self.history.append(record)
            append_jsonl(self.metrics_path, record)

            eval_str = f"{record['eval_accuracy']:.4f}" if record.get("eval_ran", False) else "skip"
            msg = (
                f"[Round {round_idx:03d}] variant={flags['variant']} clients={len(selected_ids)} "
                f"train_acc={record.get('mean_train_accuracy', 0.0):.4f} "
                f"eval_acc={eval_str} "
                f"coverage={record['package_class_coverage']:.2f} "
                f"head_steps={record['stat_head_steps_used']} "
                f"upload={record['round_upload_mb']:.3f}MB "
                f"download={record['round_download_mb']:.3f}MB "
                f"eps_max={record['max_epsilon']:.3f}"
            )
            print(msg)

            if save_ckpt and ((round_idx + 1) % ckpt_every == 0 or round_idx == rounds - 1):
                self._save_checkpoint(round_idx)

        eval_rows = [row for row in self.history if row.get("eval_ran", False)]
        final_eval_record = next((row for row in reversed(self.history) if row.get("eval_ran", False)), {})
        summary = {
            "best_eval_accuracy": max((row.get("eval_accuracy", 0.0) for row in eval_rows), default=0.0),
            "best_eval_macro_f1": max((row.get("eval_macro_f1", 0.0) for row in eval_rows), default=0.0),
            "final_eval_accuracy": float(final_eval_record.get("eval_accuracy", 0.0)),
            "final_eval_macro_f1": float(final_eval_record.get("eval_macro_f1", 0.0)),
            "final_eval_max_batches": int(final_eval_record.get("eval_max_batches", 0)),
            "final_record": self.history[-1] if self.history else {},
        }
        save_json(self.experiment_dir / "summary.json", summary)
        return summary
