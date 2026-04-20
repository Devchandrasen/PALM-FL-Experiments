#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def require(path: Path) -> None:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(ROOT)}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_required_files() -> None:
    required = [
        "palmfl/main.py",
        "palmfl/client.py",
        "palmfl/server.py",
        "palmfl/scheduler.py",
        "palmfl/dp.py",
        "palmfl/fedavg_main.py",
        "palmfl/fedmd_main.py",
        "configs/palmfl_fake_smoke.yaml",
        "configs/palmfl_mnist_stats_upload_only.yaml",
        "configs/palmfl_cifar10_stats_transfer_mobile.yaml",
        "scripts/run_revised_experiments.sh",
        "scripts/run_ablation_experiments.sh",
        "scripts/aggregate_results.py",
        "scripts/curate_trace_results.py",
        "scripts/plot_experiment_figures.py",
        "data/mobilebandwidth/real_mobile_profiles.csv",
        "analysis/all_results.csv",
        "analysis/trace_all_results.csv",
        "analysis/trace_grouped_results.csv",
        "analysis/trace_architecture_fairness.csv",
    ]
    for rel in required:
        require(ROOT / rel)


def check_docs_are_experiment_focused() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "README_REPRODUCIBILITY.md",
        ROOT / "PROJECT_IMPLEMENTATION.md",
        ROOT / "RUNNER_README.md",
    ]
    banned: list[str] = []
    for path in docs:
        require(path)
        text = read_text(path).lower()
        for phrase in banned:
            if phrase in text:
                fail(f"disallowed phrase found in {path.relative_to(ROOT)}: {phrase}")


def check_curated_results() -> None:
    grouped_path = ROOT / "analysis" / "trace_grouped_results.csv"
    all_path = ROOT / "analysis" / "trace_all_results.csv"
    grouped = read_rows(grouped_path)
    all_rows = read_rows(all_path)
    if not grouped:
        fail("analysis/trace_grouped_results.csv is empty")
    if not all_rows:
        fail("analysis/trace_all_results.csv is empty")
    datasets = {row.get("dataset") for row in grouped}
    missing = {"mnist", "cifar10"} - datasets
    if missing:
        fail(f"missing dataset(s) in grouped trace results: {', '.join(sorted(missing))}")


def main() -> int:
    check_required_files()
    check_docs_are_experiment_focused()
    check_curated_results()
    print("PASS: experiment repository consistency checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
