from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as stats
from pathlib import Path
from typing import Any

import yaml


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def nested_get(data: dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def infer_mode(cfg: dict[str, Any], summary: dict[str, Any]) -> str:
    record = summary.get("final_record", {})
    variant = str(record.get("variant") or nested_get(cfg, "algorithm.variant", "")).lower()
    summary_mode = str(record.get("summary_mode") or nested_get(cfg, "algorithm.summary_mode", "full")).lower()
    transfer_package_mode = str(
        record.get("transfer_package_mode") or nested_get(cfg, "algorithm.transfer_package_mode", "full")
    ).lower()
    dp_enabled = bool(record.get("dp_enabled", nested_get(cfg, "dp.enable", True)))
    noise_multiplier = scalar(nested_get(cfg, "dp.noise_multiplier", 0.0))
    no_noise_suffix = "-no-noise" if (variant in {"stats_upload_only", "stats_transfer"} and not dp_enabled and noise_multiplier <= 0) else ""
    if variant == "fedavg":
        arch = str(record.get("architecture") or nested_get(cfg, "baseline.architecture", "homogeneous"))
        return f"FedAvg-{arch}"
    if variant == "fedmd":
        return "FedMD-proxy"
    if variant == "local_only":
        return "Local-only"
    if variant in {"histogram_only"} or (variant == "stats_upload_only" and summary_mode == "histogram_only"):
        return "Histogram-only"
    if variant in {"mean_only"} or (variant == "stats_upload_only" and summary_mode == "mean_only"):
        return "Mean-only"
    if variant == "stats_upload_only":
        return f"Stats-only{no_noise_suffix}"
    if variant == "count_only_transfer" or (variant == "stats_transfer" and transfer_package_mode == "counts_only"):
        return "Count-only-transfer"
    if variant == "stats_transfer":
        stat_head = bool(nested_get(cfg, "training.enable_stat_head", False))
        proto_ce = float(nested_get(cfg, "training.prototype_ce_loss_weight", 0.0) or 0.0)
        if not stat_head and proto_ce <= 0:
            return "FedProto-style"
        return f"PALM-transfer{no_noise_suffix}"
    return variant or "unknown"


def infer_split_protocol(exp_name: str, seed: int, partition_seed: int, arch_seed: int) -> str:
    name = exp_name.lower()
    if "fixedsplit" in name:
        return "fixed"
    if "indsplit" in name or (partition_seed == seed and arch_seed == seed):
        return "independent"
    return "controlled"


def scalar(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(outputs_root.glob("*/summary.json")):
        exp_dir = summary_path.parent
        config_path = exp_dir / "config_resolved.yaml"
        if not config_path.exists():
            continue
        summary = read_json(summary_path)
        cfg = read_yaml(config_path)
        final = summary.get("final_record", {})
        dataset = str(nested_get(cfg, "dataset.name", "unknown")).lower()
        scheduler = str(nested_get(cfg, "scheduler.policy", "unknown")).lower()
        seed = int(nested_get(cfg, "seed", -1))
        partition_seed = nested_get(cfg, "dataset.partition_seed", seed)
        arch_seed = nested_get(cfg, "model.arch_seed", seed)
        partition_seed_int = int(partition_seed) if partition_seed is not None else seed
        arch_seed_int = int(arch_seed) if arch_seed is not None else seed
        row = {
            "experiment_dir": str(exp_dir),
            "experiment_name": exp_dir.name,
            "dataset": dataset,
            "mode": infer_mode(cfg, summary),
            "scheduler": scheduler,
            "scheduler_ablation": str(nested_get(cfg, "scheduler.ablation", "")),
            "seed": seed,
            "partition_seed": partition_seed_int,
            "arch_seed": arch_seed_int,
            "split_protocol": infer_split_protocol(exp_dir.name, seed, partition_seed_int, arch_seed_int),
            "rounds": int(nested_get(cfg, "system.rounds", 0)),
            "participation_rate": scalar(nested_get(cfg, "system.participation_rate", 0.0)),
            "noise_multiplier": scalar(nested_get(cfg, "dp.noise_multiplier", 0.0)),
            "dp_enabled": str(bool(nested_get(cfg, "dp.enable", True))).lower(),
            "summary_mode": str(nested_get(cfg, "algorithm.summary_mode", "full")),
            "transfer_package_mode": str(nested_get(cfg, "algorithm.transfer_package_mode", "full")),
            "accounting_releases_per_round": int(final.get("accounting_releases_per_round", nested_get(cfg, "dp.releases_per_round", 0)) or 0),
            "profile_csv": str(nested_get(cfg, "scheduler.profile_csv", "")),
            "final_accuracy": scalar(summary.get("final_eval_accuracy", final.get("eval_accuracy", summary.get("best_eval_accuracy")))),
            "final_macro_f1": scalar(summary.get("final_eval_macro_f1", final.get("eval_macro_f1", summary.get("best_eval_macro_f1")))),
            "uplink_mb": scalar(final.get("round_upload_mb")),
            "downlink_mb": scalar(final.get("round_download_mb")),
            "time_s": scalar(final.get("predicted_round_time_s")),
            "energy_j": scalar(final.get("predicted_round_energy_j")),
            "epsilon_max": scalar(final.get("max_epsilon")),
            "epsilon_mean": scalar(final.get("mean_epsilon")),
            "num_selected_clients": int(final.get("num_selected_clients", 0) or 0),
        }
        rows.append(row)
    return rows


def collect_fairness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        path = Path(str(row["experiment_dir"])) / "final_client_metrics.json"
        if not path.exists():
            continue
        details = read_json(path)
        if not isinstance(details, list):
            continue
        for item in details:
            out.append(
                {
                    "experiment_dir": row["experiment_dir"],
                    "dataset": row["dataset"],
                    "mode": row["mode"],
                    "scheduler": row["scheduler"],
                    "scheduler_ablation": row["scheduler_ablation"],
                    "split_protocol": row["split_protocol"],
                    "seed": row["seed"],
                    "partition_seed": row["partition_seed"],
                    "arch_seed": row["arch_seed"],
                    "noise_multiplier": row["noise_multiplier"],
                    "dp_enabled": row["dp_enabled"],
                    "rounds": row["rounds"],
                    "client_id": int(item.get("client_id", -1)),
                    "arch_name": str(item.get("arch_name", "unknown")),
                    "num_samples": int(item.get("num_samples", 0) or 0),
                    "unique_labels": int(item.get("unique_labels", 0) or 0),
                    "accuracy": scalar(item.get("accuracy")),
                    "macro_f1": scalar(item.get("macro_f1")),
                    "loss": scalar(item.get("loss")),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def grouped(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["mode"],
            row["scheduler"],
            row["scheduler_ablation"],
            row["split_protocol"],
            row["noise_multiplier"],
            row["dp_enabled"],
            row["rounds"],
        )
        groups.setdefault(key, []).append(row)

    out = []
    metrics = ["final_accuracy", "final_macro_f1", "uplink_mb", "downlink_mb", "time_s", "energy_j", "epsilon_max"]
    for key, members in sorted(groups.items()):
        item = {
            "dataset": key[0],
            "mode": key[1],
            "scheduler": key[2],
            "scheduler_ablation": key[3],
            "split_protocol": key[4],
            "noise_multiplier": key[5],
            "dp_enabled": key[6],
            "rounds": key[7],
            "n": len(members),
            "seeds": ",".join(str(m["seed"]) for m in sorted(members, key=lambda r: r["seed"])),
            "partition_seeds": ",".join(str(m["partition_seed"]) for m in sorted(members, key=lambda r: r["seed"])),
            "arch_seeds": ",".join(str(m["arch_seed"]) for m in sorted(members, key=lambda r: r["seed"])),
        }
        for metric in metrics:
            values = [float(m[metric]) for m in members]
            item[f"{metric}_mean"] = stats.mean(values)
            item[f"{metric}_std"] = stats.stdev(values) if len(values) > 1 else 0.0
        out.append(item)
    return out


def grouped_fairness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["mode"],
            row["scheduler"],
            row["scheduler_ablation"],
            row["split_protocol"],
            row["noise_multiplier"],
            row["dp_enabled"],
            row["rounds"],
            row["arch_name"],
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        item = {
            "dataset": key[0],
            "mode": key[1],
            "scheduler": key[2],
            "scheduler_ablation": key[3],
            "split_protocol": key[4],
            "noise_multiplier": key[5],
            "dp_enabled": key[6],
            "rounds": key[7],
            "arch_name": key[8],
            "n_clients": len(members),
            "seeds": ",".join(str(m["seed"]) for m in sorted(members, key=lambda r: (r["seed"], r["client_id"]))),
        }
        for metric in ["accuracy", "macro_f1", "loss", "num_samples", "unique_labels"]:
            values = [float(m[metric]) for m in members]
            item[f"{metric}_mean"] = stats.mean(values)
            item[f"{metric}_std"] = stats.stdev(values) if len(values) > 1 else 0.0
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PALM-FL and baseline summary.json artifacts")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--outdir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    rows = collect(args.outputs)
    write_csv(args.outdir / "all_results.csv", rows)
    group_rows = grouped(rows)
    write_csv(args.outdir / "grouped_results.csv", group_rows)
    fairness_rows = collect_fairness(rows)
    write_csv(args.outdir / "client_fairness.csv", fairness_rows)
    write_csv(args.outdir / "architecture_fairness.csv", grouped_fairness(fairness_rows))
    print(f"Wrote {len(rows)} runs to {args.outdir / 'all_results.csv'}")
    print(f"Wrote {len(group_rows)} grouped rows to {args.outdir / 'grouped_results.csv'}")
    print(f"Wrote {len(fairness_rows)} client fairness rows to {args.outdir / 'client_fairness.csv'}")


if __name__ == "__main__":
    main()
