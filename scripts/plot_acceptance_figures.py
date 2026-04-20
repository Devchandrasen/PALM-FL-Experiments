from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "Local-only": "#4c78a8",
    "Stats-only": "#72b7b2",
    "PALM-transfer": "#f58518",
    "FedProto-style": "#54a24b",
    "FedMD-proxy": "#b279a2",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
        return value if math.isfinite(value) else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def label_for(row: dict[str, str]) -> str:
    mode = row.get("mode", "")
    if mode.startswith("FedAvg"):
        return "FedAvg"
    return mode


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_split(rows: list[dict[str, str]], split_protocol: str) -> list[dict[str, str]]:
    if not split_protocol:
        return rows
    filtered = [r for r in rows if r.get("split_protocol") == split_protocol]
    return filtered if filtered else rows


def plot_accuracy_energy(rows: list[dict[str, str]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 3.6))
    for row in rows:
        label = label_for(row)
        color = COLORS.get(row.get("mode", ""), "#b279a2")
        marker = "s" if row.get("dataset") == "cifar10" else "o"
        ax.scatter(f(row, "energy_j"), 100.0 * f(row, "final_accuracy"), c=color, marker=marker, s=46, alpha=0.85)
        if row.get("mode") in {"PALM-transfer", "Stats-only", "Local-only", "FedMD-proxy"}:
            txt = f"{row.get('dataset','')[:3].upper()} {label.replace('-transfer','')[:5]}"
            ax.annotate(txt, (f(row, "energy_j"), 100.0 * f(row, "final_accuracy")), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Predicted round energy (J)")
    ax.set_ylabel("Final accuracy (%)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    handles = []
    labels = []
    for mode, color in COLORS.items():
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=color))
        labels.append(mode)
    handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color="#555555"))
    labels.append("MNIST")
    handles.append(plt.Line2D([0], [0], marker="s", linestyle="", color="#555555"))
    labels.append("CIFAR-10")
    ax.legend(handles, labels, ncol=3, fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_comm_privacy(rows: list[dict[str, str]], out: Path) -> None:
    rows = [r for r in rows if r.get("mode") in {"Local-only", "Stats-only", "PALM-transfer"}]
    labels = [f"{r['dataset'].upper()}\\n{r['mode']}\\n{r['scheduler']}" for r in rows]
    x = list(range(len(rows)))
    uplink = [f(r, "uplink_mb") for r in rows]
    downlink = [f(r, "downlink_mb") for r in rows]
    eps = [f(r, "epsilon_max") for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    ax1.bar(x, uplink, color="#4c78a8", label="uplink")
    ax1.bar(x, downlink, bottom=uplink, color="#f58518", label="downlink")
    ax1.set_ylabel("Payload (MB)")
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    ax2.bar(x, eps, color="#72b7b2")
    ax2.set_ylabel(r"$\epsilon_{\max}$")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
    ax2.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(rows: list[dict[str, str]], out: Path) -> None:
    selected_modes = {"Local-only", "Stats-only", "PALM-transfer", "FedMD-proxy"}
    rows = [r for r in rows if r.get("mode") in selected_modes and r.get("dataset") in {"mnist", "cifar10"}]
    groups: dict[tuple[str, str, str], dict[int, list[float]]] = {}
    for row in rows:
        metrics_path = Path(row["experiment_dir"]) / "metrics.jsonl"
        for rec in read_jsonl(metrics_path):
            if not rec.get("eval_ran", False):
                continue
            if "eval_accuracy" not in rec:
                continue
            key = (row["dataset"], row["mode"], row["scheduler"])
            groups.setdefault(key, {}).setdefault(int(rec["round_idx"]), []).append(100.0 * float(rec["eval_accuracy"]))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, dataset in zip(axes, ["mnist", "cifar10"]):
        for (ds, mode, scheduler), by_round in sorted(groups.items()):
            if ds != dataset:
                continue
            rounds = sorted(by_round)
            means = [sum(by_round[r]) / len(by_round[r]) for r in rounds]
            label = f"{mode.replace('PALM-', 'PALM ')}-{scheduler[:1].upper()}"
            ax.plot(rounds, means, marker="o", markersize=2.5, linewidth=1.2, color=COLORS.get(mode, "#555555"), label=label)
        ax.set_title(dataset.upper())
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axes[0].legend(fontsize=6, frameon=False, loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_fairness(path: Path, out: Path, split_protocol: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    rows = read_rows(path)
    if split_protocol:
        filtered = [r for r in rows if r.get("split_protocol") == split_protocol]
        rows = filtered if filtered else rows
    rows = [
        r
        for r in rows
        if r.get("mode") == "PALM-transfer"
        and r.get("scheduler") == "random"
        and r.get("dataset") in {"mnist", "cifar10"}
    ]
    if not rows:
        return
    archs = sorted({r["arch_name"] for r in rows})
    datasets = ["mnist", "cifar10"]
    width = 0.35
    x = list(range(len(archs)))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for offset, dataset in [(-width / 2, "mnist"), (width / 2, "cifar10")]:
        vals = []
        for arch in archs:
            match = next((r for r in rows if r["dataset"] == dataset and r["arch_name"] == arch), None)
            vals.append(100.0 * f(match or {}, "accuracy_mean"))
        ax.bar([i + offset for i in x], vals, width=width, label=dataset.upper())
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("tiny_", "t.").replace("_", " ") for a in archs], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Per-architecture accuracy (%)")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create acceptance-readiness figures from aggregated PALM-FL results")
    parser.add_argument("--results", type=Path, default=Path("analysis/all_results.csv"))
    parser.add_argument("--fairness", type=Path, default=Path("analysis/architecture_fairness.csv"))
    parser.add_argument("--figdir", type=Path, default=Path("/home/dr-chandrasen-pandey/PALM_review/figures"))
    parser.add_argument("--split-protocol", default="independent")
    args = parser.parse_args()
    rows = filter_split(read_rows(args.results), args.split_protocol)
    completed = [r for r in rows if f(r, "final_accuracy") > 0 and r.get("dataset") in {"mnist", "cifar10"}]
    plot_learning_curves(completed, args.figdir / "fig2_learning_curves.pdf")
    plot_accuracy_energy(completed, args.figdir / "fig5_accuracy_energy_frontier.pdf")
    plot_comm_privacy(completed, args.figdir / "fig6_comm_privacy_bars.pdf")
    plot_architecture_fairness(args.fairness, args.figdir / "fig7_architecture_fairness.pdf", args.split_protocol)
    print(f"Wrote figures under {args.figdir}")


if __name__ == "__main__":
    main()
