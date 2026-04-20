from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from .client import Client
from .data import build_client_loaders
from .dp import StatsDPMechanism
from .models import PALMFLModel, available_architectures
from .server import PALMFLServer
from .utils import ensure_dir, load_config, pretty_num_params, resolve_device, save_yaml, set_seed, timestamp


def assign_architectures(num_clients: int, architectures: list[str], seed: int) -> list[str]:
    if not architectures:
        raise ValueError("Config must provide at least one architecture")
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_clients, generator=rng).tolist()
    assigned = [architectures[i % len(architectures)] for i in range(num_clients)]
    shuffled = [None] * num_clients
    for src_idx, dst_idx in enumerate(order):
        shuffled[dst_idx] = assigned[src_idx]
    return [str(x) for x in shuffled]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PALM-FL v8 pivot prototype")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in key=value form. Can be passed multiple times.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
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

    experiment_name = cfg.get("experiment_name", Path(args.config).stem)
    output_root = ensure_dir(cfg.get("output_root", "./outputs"))
    exp_dir = ensure_dir(output_root / f"{experiment_name}_{timestamp()}")
    save_yaml(exp_dir / "config_resolved.yaml", cfg)

    train_loaders, test_loader, client_meta, meta = build_client_loaders(cfg)
    archs = list(cfg["model"]["architectures"])
    supported_archs = available_architectures()
    for arch in archs:
        if arch not in supported_archs:
            raise ValueError(f"Unknown architecture {arch}. Available: {list(supported_archs)}")

    assigned_archs = assign_architectures(
        num_clients=int(cfg["system"]["num_clients"]),
        architectures=archs,
        seed=int(cfg["model"].get("arch_seed", cfg.get("seed", 0))),
    )

    dp_cfg = cfg["dp"]
    summary_mode = str(cfg.get("algorithm", {}).get("summary_mode", "full")).lower()
    releases_default = {"histogram_only": 1, "mean_only": 2}.get(summary_mode, 3)
    dp_mechanism = StatsDPMechanism(
        clip_norm=float(dp_cfg.get("clip_norm", 4.0)),
        count_clip=float(dp_cfg.get("count_clip", 200.0)),
        noise_multiplier=float(dp_cfg.get("noise_multiplier", 1.0)),
        enabled=bool(dp_cfg.get("enable", True)),
        releases_per_round=int(dp_cfg.get("releases_per_round", releases_default)),
        count_threshold=float(dp_cfg.get("count_threshold", 1.0)),
    )

    clients = {}
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
        clients[client_id] = Client(
            client_id=client_id,
            model=model,
            train_loader=loader,
            client_meta=cmeta,
            dp_mechanism=dp_mechanism,
            cfg=cfg,
            device=device,
        )

    print("=" * 88)
    print("PALM-FL v8 pivot configuration")
    print(f"Experiment dir: {exp_dir}")
    print(f"Dataset: {meta.name} | Classes: {meta.num_classes} | Input channels: {meta.channels}")
    print(f"Runtime device: {device}")
    print(f"Variant: {cfg.get('algorithm', {}).get('variant', 'stats_transfer')}")
    print(
        "Summary mode: "
        f"{cfg.get('algorithm', {}).get('summary_mode', 'full')} | "
        f"Transfer package: {cfg.get('algorithm', {}).get('transfer_package_mode', 'full')}"
    )
    for arch_name, count in sorted(arch_counter.items()):
        first_client = next(client for client in clients.values() if client.model.arch_name == arch_name)
        print(f"  - {arch_name:14s}: {count:3d} clients | params={pretty_num_params(first_client.num_parameters)}")
    print("=" * 88)

    server = PALMFLServer(
        cfg=cfg,
        clients=clients,
        test_loader=test_loader,
        experiment_dir=exp_dir,
        device=device,
    )
    summary = server.run()

    print("=" * 88)
    print("Training complete")
    print(summary)
    print("=" * 88)


if __name__ == "__main__":
    main()
