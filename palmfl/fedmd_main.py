from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from .client import Client
from .data import build_client_loaders, build_public_proxy_loader
from .dp import StatsDPMechanism
from .main import assign_architectures
from .metrics import evaluate_model, summarize_metric_list
from .models import PALMFLModel, available_architectures, soft_cross_entropy
from .scheduler import MobileAwareScheduler
from .utils import append_jsonl, bytes_from_tensors, ensure_dir, load_config, pretty_num_params, resolve_device, save_json, save_yaml, set_seed, timestamp


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FedMD-style public-proxy distillation baseline for heterogeneous PALM-FL clients")
    parser.add_argument("--config", type=str, required=True, help="Path to a PALM-FL YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in key=value form. Can be passed multiple times.",
    )
    return parser


def _select_clients(
    cfg: Dict,
    scheduler: MobileAwareScheduler,
    rng: random.Random,
    round_idx: int,
    upload_mb: float,
    download_mb: float,
    local_steps: int,
) -> tuple[list[int], dict[int, dict]]:
    system_cfg = cfg["system"]
    policy = str(cfg["scheduler"].get("policy", "mobile")).lower()
    max_clients = max(1, int(math.ceil(float(system_cfg.get("participation_rate", 0.2)) * int(system_cfg["num_clients"]))))
    if policy == "mobile":
        return scheduler.select_clients(
            round_idx=round_idx,
            max_clients=max_clients,
            upload_mb=upload_mb,
            download_mb=download_mb,
            local_steps=local_steps,
        )

    client_ids = list(range(int(system_cfg["num_clients"])))
    if policy == "all":
        selected = sorted(client_ids)
    elif policy == "random":
        selected = sorted(rng.sample(client_ids, k=min(max_clients, len(client_ids))))
    else:
        raise ValueError(f"Unknown scheduler.policy: {policy}")

    info = {}
    for cid in selected:
        costs = scheduler.estimate_costs(cid, upload_mb=upload_mb, download_mb=download_mb, local_steps=local_steps)
        costs["utility"] = 0.0
        info[cid] = costs
    return selected, info


@torch.no_grad()
def _proxy_probs(
    client: Client,
    proxy_loader,
    device: torch.device,
    temperature: float,
    normalize_latent: bool,
) -> torch.Tensor:
    client.model.eval()
    client.model.to(device)
    outputs = []
    for x, _ in proxy_loader:
        x = x.to(device, non_blocking=True)
        logits = client.model(x, normalize_latent=normalize_latent)
        outputs.append(F.softmax(logits / temperature, dim=-1).detach().cpu())
    client.model.to(torch.device("cpu"))
    return torch.cat(outputs, dim=0)


def _distill_client(
    client: Client,
    proxy_loader,
    consensus_probs: torch.Tensor,
    device: torch.device,
    epochs: int,
    temperature: float,
    normalize_latent: bool,
    grad_clip: float,
) -> dict:
    if epochs <= 0:
        return {"distill_loss": 0.0}
    client.model.train()
    client.model.to(device)
    losses = []
    for _ in range(epochs):
        offset = 0
        for x, _ in proxy_loader:
            batch_size = x.size(0)
            target = consensus_probs[offset : offset + batch_size].to(device, non_blocking=True)
            offset += batch_size
            x = x.to(device, non_blocking=True)
            client.optimizer_all.zero_grad(set_to_none=True)
            logits = client.model(x, normalize_latent=normalize_latent) / temperature
            loss = (temperature**2) * soft_cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(client.model.parameters(), grad_clip)
            client.optimizer_all.step()
            losses.append(float(loss.item()))
    client.model.to(torch.device("cpu"))
    return {"distill_loss": float(sum(losses) / max(len(losses), 1))}


def _evaluate_clients(
    clients: dict[int, Client],
    test_loader,
    device: torch.device,
    num_classes: int,
    normalize_latent: bool,
    max_batches: int | None,
    return_details: bool = False,
) -> dict | tuple[dict, list[dict]]:
    metrics = []
    details = []
    for cid in sorted(clients):
        client = clients[cid]
        client.model.to(device)
        result = evaluate_model(
            model=client.model,
            loader=test_loader,
            device=device,
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


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    cfg.setdefault("algorithm", {})["variant"] = "fedmd"
    cfg["training"]["upload_stats"] = False
    cfg["training"]["enable_prototype_alignment"] = False
    cfg["training"]["enable_stat_head"] = False
    cfg["dataset"]["public_proxy_size"] = int(cfg["dataset"].get("public_proxy_size", cfg.get("baseline", {}).get("proxy_size", 512)))
    cfg["dataset"]["public_proxy_seed"] = int(cfg["dataset"].get("public_proxy_seed", cfg.get("seed", 0)))

    num_threads = int(cfg["system"].get("num_threads", 1))
    torch.set_num_threads(max(1, num_threads))
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(num_threads, 4)))
        except RuntimeError:
            pass

    device = resolve_device(str(cfg["system"].get("device", "auto")))
    cfg["system"]["resolved_device"] = str(device)
    set_seed(int(cfg.get("seed", 0)), device=device)
    rng = random.Random(int(cfg.get("seed", 0)))

    experiment_name = cfg.get("experiment_name", Path(args.config).stem)
    output_root = ensure_dir(cfg.get("output_root", "./outputs"))
    exp_dir = ensure_dir(output_root / f"{experiment_name}_{timestamp()}")
    checkpoint_dir = ensure_dir(exp_dir / "checkpoints")
    metrics_path = exp_dir / "metrics.jsonl"
    save_yaml(exp_dir / "config_resolved.yaml", cfg)

    train_loaders, test_loader, client_meta, meta = build_client_loaders(cfg)
    proxy_loader, _ = build_public_proxy_loader(cfg)
    proxy_size = len(proxy_loader.dataset)
    proxy_payload_bytes = proxy_size * meta.num_classes * 4
    proxy_payload_mb = proxy_payload_bytes / (1024**2)

    archs = list(cfg["model"]["architectures"])
    for arch in archs:
        if arch not in available_architectures():
            raise ValueError(f"Unknown architecture {arch}. Available: {list(available_architectures())}")
    assigned_archs = assign_architectures(
        num_clients=int(cfg["system"]["num_clients"]),
        architectures=archs,
        seed=int(cfg["model"].get("arch_seed", cfg.get("seed", 0))),
    )

    dp_mechanism = StatsDPMechanism(enabled=False, clip_norm=1.0, count_clip=1.0, noise_multiplier=0.0)
    clients: dict[int, Client] = {}
    arch_counter = Counter()
    for client_id, loader in train_loaders.items():
        arch_name = assigned_archs[client_id]
        arch_counter[arch_name] += 1
        cmeta = dict(client_meta[client_id])
        cmeta["num_classes"] = meta.num_classes
        model = PALMFLModel(
            arch_name=arch_name,
            in_channels=meta.channels,
            latent_dim=int(cfg["model"]["latent_dim"]),
            num_classes=meta.num_classes,
            adapter_hidden_dim=int(cfg["model"].get("adapter_hidden_dim", 128)),
            head_hidden_dim=int(cfg["model"].get("head_hidden_dim", 0)),
            dropout=float(cfg["model"].get("dropout", 0.0)),
        )
        client = Client(
            client_id=client_id,
            model=model,
            train_loader=loader,
            client_meta=cmeta,
            dp_mechanism=dp_mechanism,
            cfg=cfg,
            device=device,
        )
        client.cfg.upload_stats = False
        clients[client_id] = client

    scheduler = MobileAwareScheduler(
        num_clients=int(cfg["system"]["num_clients"]),
        num_classes=meta.num_classes,
        scheduler_cfg=cfg["scheduler"],
        seed=int(cfg.get("seed", 0)),
    )
    for cid, client in clients.items():
        scheduler.register_client(
            client_id=cid,
            num_samples=client.num_samples,
            unique_labels=client.unique_labels,
            model_num_params=client.num_parameters,
        )

    local_epochs = int(cfg["training"].get("local_epochs", 1))
    finetune_epochs = int(cfg["training"].get("real_finetune_epochs", 0))
    distill_epochs = int(cfg.get("baseline", {}).get("distill_epochs", 1))
    temperature = float(cfg.get("baseline", {}).get("distill_temperature", 2.0))
    normalize_latent = bool(cfg["training"].get("normalize_latent", False))
    grad_clip = float(cfg["training"].get("grad_clip", 5.0))
    mean_steps = max(
        1,
        int(
            round(
                sum(len(c.train_loader) * (local_epochs + finetune_epochs) for c in clients.values()) / len(clients)
            )
        )
        + distill_epochs * len(proxy_loader),
    )
    rounds = int(cfg["system"]["rounds"])
    eval_every = max(1, int(cfg["system"].get("eval_every", 5)))
    eval_initial = bool(cfg["system"].get("eval_initial", True))
    eval_max_batches = int(cfg["system"].get("eval_max_batches", 0)) or None
    final_eval_max_batches = int(cfg["system"].get("final_eval_max_batches", cfg["system"].get("eval_max_batches", 0))) or None
    save_ckpt = bool(cfg.get("logging", {}).get("save_checkpoints", True))
    ckpt_every = int(cfg.get("logging", {}).get("checkpoint_every", 10))

    print("=" * 88)
    print("FedMD-style heterogeneous distillation baseline configuration")
    print(f"Experiment dir: {exp_dir}")
    print(f"Dataset: {meta.name} | Proxy samples: {proxy_size} | Clients: {len(clients)}")
    print(f"Runtime device: {device} | proxy-logit payload={proxy_payload_mb:.4f} MB/client")
    for arch_name, count in sorted(arch_counter.items()):
        first_client = next(client for client in clients.values() if client.model.arch_name == arch_name)
        print(f"  - {arch_name:14s}: {count:3d} clients | params={pretty_num_params(first_client.num_parameters)}")
    print("=" * 88)

    history = []
    for round_idx in range(rounds):
        wall_start = time.perf_counter()
        selected_ids, scheduler_info = _select_clients(
            cfg=cfg,
            scheduler=scheduler,
            rng=rng,
            round_idx=round_idx,
            upload_mb=proxy_payload_mb,
            download_mb=proxy_payload_mb,
            local_steps=mean_steps,
        )

        local_wall_times = []
        updates = {}
        proxy_outputs = []
        weights = []
        for cid in selected_ids:
            client = clients[cid]
            start = time.perf_counter()
            client.model.to(device)
            log = client._train_real(package=None, epochs=local_epochs)
            if finetune_epochs > 0:
                log = client._train_real(package=None, epochs=finetune_epochs)
            probs = _proxy_probs(client, proxy_loader, device, temperature, normalize_latent)
            duration = time.perf_counter() - start
            proxy_outputs.append(probs)
            weights.append(float(client.num_samples))
            update = {
                **log,
                "client_id": cid,
                "round_idx": round_idx,
                "duration_s": duration,
                "upload_bytes": proxy_payload_bytes,
                "download_bytes": proxy_payload_bytes,
            }
            update.update(scheduler_info.get(cid, {}))
            updates[cid] = update
            local_wall_times.append(duration)

        weight_sum = max(sum(weights), 1e-12)
        consensus = sum(probs * (weight / weight_sum) for probs, weight in zip(proxy_outputs, weights))
        consensus = consensus / consensus.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        for cid in selected_ids:
            distill_log = _distill_client(
                clients[cid],
                proxy_loader,
                consensus,
                device,
                distill_epochs,
                temperature,
                normalize_latent,
                grad_clip,
            )
            updates[cid].update(distill_log)

        scheduler.update_after_round(round_idx=round_idx, updates=updates)
        record = {
            "round_idx": round_idx,
            "variant": "fedmd",
            "selected_clients": selected_ids,
            "num_selected_clients": len(selected_ids),
            "round_upload_mb": proxy_payload_mb * len(selected_ids),
            "round_download_mb": proxy_payload_mb * len(selected_ids),
            "predicted_round_time_s": max((updates[cid].get("predicted_time_s", 0.0) for cid in updates), default=0.0),
            "predicted_round_energy_j": sum(updates[cid].get("predicted_energy_j", 0.0) for cid in updates),
            "mean_client_wall_time_s": statistics.mean(local_wall_times) if local_wall_times else 0.0,
            "max_epsilon": 0.0,
            "mean_epsilon": 0.0,
            "num_participating_clients": 0,
            "mean_train_accuracy": float(sum(u.get("train_accuracy", 0.0) for u in updates.values()) / max(len(updates), 1)),
            "mean_train_loss": float(sum(u.get("avg_loss", 0.0) for u in updates.values()) / max(len(updates), 1)),
            "mean_distill_loss": float(sum(u.get("distill_loss", 0.0) for u in updates.values()) / max(len(updates), 1)),
        }

        eval_ran = False
        should_eval = (round_idx == rounds - 1) or (
            round_idx % eval_every == 0 and (eval_initial or round_idx > 0)
        )
        if should_eval:
            current_eval_max_batches = final_eval_max_batches if round_idx == rounds - 1 else eval_max_batches
            if round_idx == rounds - 1 and bool(cfg["system"].get("save_final_client_metrics", True)):
                eval_metrics, eval_details = _evaluate_clients(
                    clients,
                    test_loader,
                    device,
                    meta.num_classes,
                    normalize_latent,
                    current_eval_max_batches,
                    return_details=True,
                )
                save_json(exp_dir / "final_client_metrics.json", eval_details)
            else:
                eval_metrics = _evaluate_clients(
                    clients,
                    test_loader,
                    device,
                    meta.num_classes,
                    normalize_latent,
                    current_eval_max_batches,
                )
            record.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            record["eval_max_batches"] = 0 if current_eval_max_batches is None else int(current_eval_max_batches)
            record["eval_is_final"] = bool(round_idx == rounds - 1)
            eval_ran = True
        record["eval_ran"] = eval_ran
        record["wall_clock_s"] = time.perf_counter() - wall_start
        history.append(record)
        append_jsonl(metrics_path, record)

        eval_str = f"{record['eval_accuracy']:.4f}" if eval_ran else "skip"
        print(
            f"[Round {round_idx:03d}] variant=fedmd clients={len(selected_ids)} "
            f"train_acc={record['mean_train_accuracy']:.4f} eval_acc={eval_str} "
            f"upload={record['round_upload_mb']:.4f}MB download={record['round_download_mb']:.4f}MB"
        )

        if save_ckpt and ((round_idx + 1) % ckpt_every == 0 or round_idx == rounds - 1):
            torch.save(
                {
                    "round_idx": round_idx,
                    "config": cfg,
                    "scheduler_state": scheduler.state_dict(),
                    "history": history,
                    "client_states": {cid: client.model.state_dict() for cid, client in clients.items()},
                },
                checkpoint_dir / f"round_{round_idx:04d}.pt",
            )

    eval_rows = [row for row in history if row.get("eval_ran", False)]
    final_eval_record = next((row for row in reversed(history) if row.get("eval_ran", False)), {})
    summary = {
        "best_eval_accuracy": max((row.get("eval_accuracy", 0.0) for row in eval_rows), default=0.0),
        "best_eval_macro_f1": max((row.get("eval_macro_f1", 0.0) for row in eval_rows), default=0.0),
        "final_eval_accuracy": float(final_eval_record.get("eval_accuracy", 0.0)),
        "final_eval_macro_f1": float(final_eval_record.get("eval_macro_f1", 0.0)),
        "final_eval_max_batches": int(final_eval_record.get("eval_max_batches", 0)),
        "final_record": history[-1] if history else {},
    }
    save_json(exp_dir / "summary.json", summary)
    print("=" * 88)
    print("FedMD-style baseline complete")
    print(summary)
    print("=" * 88)


if __name__ == "__main__":
    main()
