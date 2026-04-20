from __future__ import annotations

import argparse
import csv
import statistics as stats
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def grouped(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["mode"],
            row["scheduler"],
            row.get("scheduler_ablation", ""),
            row["split_protocol"],
            row["noise_multiplier"],
            row.get("dp_enabled", "true"),
            row["rounds"],
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    metrics = ["final_accuracy", "final_macro_f1", "uplink_mb", "downlink_mb", "time_s", "energy_j", "epsilon_max"]
    for key, members in sorted(groups.items()):
        members = sorted(members, key=lambda r: (int(r["seed"]), r["experiment_name"]))
        item: dict[str, Any] = {
            "dataset": key[0],
            "mode": key[1],
            "scheduler": key[2],
            "scheduler_ablation": key[3],
            "split_protocol": key[4],
            "noise_multiplier": key[5],
            "dp_enabled": key[6],
            "rounds": key[7],
            "n": len(members),
            "seeds": ",".join(m["seed"] for m in members),
            "experiment_names": "|".join(m["experiment_name"] for m in members),
        }
        for metric in metrics:
            values = [f(m, metric) for m in members]
            item[f"{metric}_mean"] = stats.mean(values)
            item[f"{metric}_std"] = stats.stdev(values) if len(values) > 1 else 0.0
        out.append(item)
    return out


def grouped_fairness(rows: list[dict[str, str]], keep_dirs: set[str]) -> list[dict[str, Any]]:
    rows = [r for r in rows if r["experiment_dir"] in keep_dirs]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["mode"],
            row["scheduler"],
            row.get("scheduler_ablation", ""),
            row["split_protocol"],
            row["noise_multiplier"],
            row.get("dp_enabled", "true"),
            row["rounds"],
            row["arch_name"],
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        item: dict[str, Any] = {
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
            "seeds": ",".join(sorted({m["seed"] for m in members})),
        }
        for metric in ["accuracy", "macro_f1", "loss", "num_samples", "unique_labels"]:
            values = [f(m, metric) for m in members]
            item[f"{metric}_mean"] = stats.mean(values)
            item[f"{metric}_std"] = stats.stdev(values) if len(values) > 1 else 0.0
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate only the revised TMC trace/independent-split runs.")
    parser.add_argument("--all-results", type=Path, default=Path("analysis/all_results.csv"))
    parser.add_argument("--client-fairness", type=Path, default=Path("analysis/client_fairness.csv"))
    parser.add_argument("--out-all", type=Path, default=Path("analysis/tmc_trace_all_results.csv"))
    parser.add_argument("--out-grouped", type=Path, default=Path("analysis/tmc_trace_grouped_results.csv"))
    parser.add_argument("--out-fairness", type=Path, default=Path("analysis/tmc_trace_architecture_fairness.csv"))
    args = parser.parse_args()

    rows = [
        r
        for r in read_rows(args.all_results)
        if r.get("dataset") in {"mnist", "cifar10"}
        and "_indsplit_trace_" in r.get("experiment_name", "")
        and f(r, "final_accuracy") > 0
    ]
    write_rows(args.out_all, rows)
    write_rows(args.out_grouped, grouped(rows))
    keep_dirs = {r["experiment_dir"] for r in rows}
    fairness_rows = read_rows(args.client_fairness) if args.client_fairness.exists() and args.client_fairness.stat().st_size else []
    write_rows(args.out_fairness, grouped_fairness(fairness_rows, keep_dirs))
    print(f"Wrote {len(rows)} curated runs to {args.out_all}")
    print(f"Wrote {len(grouped(rows))} grouped rows to {args.out_grouped}")


if __name__ == "__main__":
    main()
