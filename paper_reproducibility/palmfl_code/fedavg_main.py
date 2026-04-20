from __future__ import annotations

import argparse
import copy
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Dict

import torch

from .client import Client
from .data import build_client_loaders
from .dp import StatsDPMechanism
from .metrics import evaluate_model
from .models import PALMFLModel, available_architectures
from .scheduler import MobileAwareScheduler
from .utils import (
    append_jsonl,
    bytes_from_tensors,
    ensure_dir,
    load_config,
    pretty_num_params,
    resolve_device,
    save_json,
    save_yaml,
    set_seed,
    timestamp,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Homogeneous FedAvg baseline for PALM-FL experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to a PALM-FL YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in key=value form. Can be passed multiple times.",
    )
    return parser


def _state_num_bytes(state: dict[str, torch.Tensor]) -> int:
    return bytes_from_tensors(*(tensor for tensor in state.values() if torch.is_tensor(tensor)))


def _average_state_dicts(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("Cannot average an empty state list")
    weight_sum = max(float(sum(weights)), 1e-12)
    averaged: dict[str, torch.Tensor] = {}
    for key in states[0]:
        first = states[0][key]
        if not torch.is_tensor(first):
            averaged[key] = copy.deepcopy(first)
            continue
        if not first.is_floating_point():
            averaged[key] = first.clone()
            continue
        acc = torch.zeros_like(first, dtype=torch.float32)
        for state, weight in zip(states, weights):
            acc += state[key].detach().float() * float(weight / weight_sum)
        averaged[key] = acc.to(dtype=first.dtype)
    return averaged


def _reset_client_optimizers(client: Client) -> None:
    client.optimizer_all = torch.optim.Adam(
        client.model.parameters(),
        lr=client.cfg.lr,
        weight_decay=client.cfg.weight_decay,
    )
    client.optimizer_head = torch.optim.Adam(
        client.model.head.parameters(),
        lr=client.cfg.lr,
        weight_decay=client.cfg.weight_decay,
    )


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


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    cfg.setdefault("algorithm", {})["variant"] = "fedavg"
    cfg["training"]["upload_stats"] = False
    cfg["training"]["enable_prototype_alignment"] = False
    cfg["training"]["enable_stat_head"] = False

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
    baseline_cfg = cfg.get("baseline", {})
    architectures = cfg["model"].get("architectures", ["small_cnn"])
    arch_name = str(baseline_cfg.get("architecture", architectures[0] if isinstance(architectures, list) else architectures))
    if arch_name not in available_architectures():
        raise ValueError(f"Unknown FedAvg architecture {arch_name}. Available: {list(available_architectures())}")

    def new_model() -> PALMFLModel:
        return PALMFLModel(
            arch_name=arch_name,
            in_channels=meta.channels,
            latent_dim=int(cfg["model"]["latent_dim"]),
            num_classes=meta.num_classes,
            adapter_hidden_dim=int(cfg["model"].get("adapter_hidden_dim", 128)),
            head_hidden_dim=int(cfg["model"].get("head_hidden_dim", 0)),
            dropout=float(cfg["model"].get("dropout", 0.0)),
        )

    global_model = new_model()
    global_state = copy.deepcopy(global_model.state_dict())
    model_bytes = _state_num_bytes(global_state)
    model_mb = model_bytes / (1024**2)

    dp_mechanism = StatsDPMechanism(enabled=False, clip_norm=1.0, count_clip=1.0, noise_multiplier=0.0)
    clients: dict[int, Client] = {}
    for client_id, loader in train_loaders.items():
        cmeta = dict(client_meta[client_id])
        cmeta["num_classes"] = meta.num_classes
        client = Client(
            client_id=client_id,
            model=new_model(),
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
    mean_steps = max(1, int(round(sum(len(c.train_loader) * (local_epochs + finetune_epochs) for c in clients.values()) / len(clients))))
    rounds = int(cfg["system"]["rounds"])
    eval_every = max(1, int(cfg["system"].get("eval_every", 5)))
    eval_initial = bool(cfg["system"].get("eval_initial", True))
    eval_max_batches = int(cfg["system"].get("eval_max_batches", 0)) or None
    final_eval_max_batches = int(cfg["system"].get("final_eval_max_batches", cfg["system"].get("eval_max_batches", 0))) or None
    save_ckpt = bool(cfg.get("logging", {}).get("save_checkpoints", True))
    ckpt_every = int(cfg.get("logging", {}).get("checkpoint_every", 10))
    normalize_latent = bool(cfg["training"].get("normalize_latent", False))

    print("=" * 88)
    print("Homogeneous FedAvg baseline configuration")
    print(f"Experiment dir: {exp_dir}")
    print(f"Dataset: {meta.name} | Architecture: {arch_name} | Clients: {len(clients)}")
    print(f"Runtime device: {device} | model payload={model_mb:.3f} MB")
    print("=" * 88)

    history = []
    for round_idx in range(rounds):
        wall_start = time.perf_counter()
        selected_ids, scheduler_info = _select_clients(
            cfg=cfg,
            scheduler=scheduler,
            rng=rng,
            round_idx=round_idx,
            upload_mb=model_mb,
            download_mb=model_mb,
            local_steps=mean_steps,
        )

        local_states = []
        local_weights = []
        local_wall_times = []
        updates = {}
        for cid in selected_ids:
            client = clients[cid]
            client.model.load_state_dict(global_state)
            _reset_client_optimizers(client)
            client.model.to(device)
            start = time.perf_counter()
            log = client._train_real(package=None, epochs=local_epochs)
            if finetune_epochs > 0:
                log = client._train_real(package=None, epochs=finetune_epochs)
            duration = time.perf_counter() - start
            client.model.to(torch.device("cpu"))
            update = {
                **log,
                "client_id": cid,
                "round_idx": round_idx,
                "duration_s": duration,
                "upload_bytes": model_bytes,
                "download_bytes": model_bytes,
            }
            update.update(scheduler_info.get(cid, {}))
            updates[cid] = update
            local_states.append(copy.deepcopy(client.model.state_dict()))
            local_weights.append(float(client.num_samples))
            local_wall_times.append(duration)

        global_state = _average_state_dicts(local_states, local_weights)
        global_model.load_state_dict(global_state)
        scheduler.update_after_round(round_idx=round_idx, updates=updates)

        record = {
            "round_idx": round_idx,
            "variant": "fedavg",
            "architecture": arch_name,
            "selected_clients": selected_ids,
            "num_selected_clients": len(selected_ids),
            "round_upload_mb": model_mb * len(selected_ids),
            "round_download_mb": model_mb * len(selected_ids),
            "predicted_round_time_s": max((updates[cid].get("predicted_time_s", 0.0) for cid in updates), default=0.0),
            "predicted_round_energy_j": sum(updates[cid].get("predicted_energy_j", 0.0) for cid in updates),
            "mean_client_wall_time_s": statistics.mean(local_wall_times) if local_wall_times else 0.0,
            "max_epsilon": 0.0,
            "mean_epsilon": 0.0,
            "num_participating_clients": 0,
            "mean_train_accuracy": float(sum(u.get("train_accuracy", 0.0) for u in updates.values()) / max(len(updates), 1)),
            "mean_train_loss": float(sum(u.get("avg_loss", 0.0) for u in updates.values()) / max(len(updates), 1)),
        }

        eval_ran = False
        should_eval = (round_idx == rounds - 1) or (
            round_idx % eval_every == 0 and (eval_initial or round_idx > 0)
        )
        if should_eval:
            current_eval_max_batches = final_eval_max_batches if round_idx == rounds - 1 else eval_max_batches
            global_model.to(device)
            eval_metrics = evaluate_model(
                model=global_model,
                loader=test_loader,
                device=device,
                num_classes=meta.num_classes,
                normalize_latent=normalize_latent,
                max_batches=current_eval_max_batches,
            )
            global_model.to(torch.device("cpu"))
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
            f"[Round {round_idx:03d}] variant=fedavg arch={arch_name} clients={len(selected_ids)} "
            f"train_acc={record['mean_train_accuracy']:.4f} eval_acc={eval_str} "
            f"upload={record['round_upload_mb']:.3f}MB download={record['round_download_mb']:.3f}MB"
        )

        if save_ckpt and ((round_idx + 1) % ckpt_every == 0 or round_idx == rounds - 1):
            torch.save(
                {
                    "round_idx": round_idx,
                    "config": cfg,
                    "global_model": global_state,
                    "scheduler_state": scheduler.state_dict(),
                    "history": history,
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
    print("FedAvg baseline complete")
    print(summary)
    print("=" * 88)


if __name__ == "__main__":
    main()
