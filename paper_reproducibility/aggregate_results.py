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
    if variant == "fedavg":
        arch = str(record.get("architecture") or nested_get(cfg, "baseline.architecture", "homogeneous"))
        return f"FedAvg-{arch}"
    if variant == "local_only":
        return "Local-only"
    if variant == "stats_upload_only":
        return "Stats-only"
    if variant == "stats_transfer":
        stat_head = bool(nested_get(cfg, "training.enable_stat_head", False))
        proto_ce = float(nested_get(cfg, "training.prototype_ce_loss_weight", 0.0) or 0.0)
        if not stat_head and proto_ce <= 0:
            return "FedProto-style"
        return "PALM-transfer"
    return variant or "unknown"


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
        row = {
            "experiment_dir": str(exp_dir),
            "experiment_name": exp_dir.name,
            "dataset": dataset,
            "mode": infer_mode(cfg, summary),
            "scheduler": scheduler,
            "seed": seed,
            "partition_seed": int(partition_seed) if partition_seed is not None else seed,
            "arch_seed": int(arch_seed) if arch_seed is not None else seed,
            "rounds": int(nested_get(cfg, "system.rounds", 0)),
            "participation_rate": scalar(nested_get(cfg, "system.participation_rate", 0.0)),
            "noise_multiplier": scalar(nested_get(cfg, "dp.noise_multiplier", 0.0)),
            "fixed_split": bool(partition_seed is not None or arch_seed is not None),
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
            row["partition_seed"],
            row["arch_seed"],
            row["noise_multiplier"],
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
            "partition_seed": key[3],
            "arch_seed": key[4],
            "noise_multiplier": key[5],
            "rounds": key[6],
            "n": len(members),
            "seeds": ",".join(str(m["seed"]) for m in sorted(members, key=lambda r: r["seed"])),
        }
        for metric in metrics:
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
    print(f"Wrote {len(rows)} runs to {args.outdir / 'all_results.csv'}")
    print(f"Wrote {len(group_rows)} grouped rows to {args.outdir / 'grouped_results.csv'}")


if __name__ == "__main__":
    main()
