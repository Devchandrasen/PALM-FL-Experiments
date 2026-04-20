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
    "FedAvg": "#5f5f5f",
}

NUMERIC_FIELDS = (
    "final_accuracy",
    "final_macro_f1",
    "uplink_mb",
    "downlink_mb",
    "time_s",
    "energy_j",
    "epsilon_max",
)


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


def color_for(row: dict[str, str]) -> str:
    return COLORS.get(label_for(row), COLORS.get(row.get("mode", ""), "#8c8c8c"))


def short_dataset(row: dict[str, str]) -> str:
    return "CI" if row.get("dataset") == "cifar10" else "MN"


def short_mode(row: dict[str, str]) -> str:
    mode = row.get("mode", "")
    scheduler = row.get("scheduler", "")
    if mode.startswith("FedAvg"):
        return "FedAvg"
    if mode == "PALM-transfer":
        return "Tr-R" if scheduler == "random" else "Tr-M"
    if mode == "Stats-only":
        return "Stats"
    if mode == "Local-only":
        return "Local"
    if mode == "FedProto-style":
        return "FedProto"
    if mode == "FedMD-proxy":
        return "FedMD"
    return mode


def mean_by_operating_point(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(
            (
                row.get("dataset", ""),
                row.get("mode", ""),
                row.get("scheduler", ""),
                row.get("scheduler_ablation", ""),
            ),
            [],
        ).append(row)

    out: list[dict[str, str]] = []
    dataset_order = {"mnist": 0, "cifar10": 1}
    mode_order = {
        "Local-only": 0,
        "Stats-only": 1,
        "PALM-transfer": 2,
        "FedProto-style": 3,
        "FedMD-proxy": 4,
        "FedAvg-small_cnn": 5,
    }
    for (dataset, mode, scheduler, scheduler_ablation), members in sorted(
        groups.items(),
        key=lambda item: (dataset_order.get(item[0][0], 99), mode_order.get(item[0][1], 99), item[0][2]),
    ):
        item: dict[str, str] = {
            "dataset": dataset,
            "mode": mode,
            "scheduler": scheduler,
            "scheduler_ablation": scheduler_ablation,
            "n": str(len(members)),
        }
        for field in NUMERIC_FIELDS:
            item[field] = str(sum(f(member, field) for member in members) / max(len(members), 1))
        out.append(item)
    return out


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
    selected_modes = {"Local-only", "Stats-only", "PALM-transfer", "FedProto-style", "FedMD-proxy", "FedAvg-small_cnn"}
    rows = [
        r
        for r in mean_by_operating_point(rows)
        if r.get("mode") in selected_modes and not r.get("scheduler_ablation")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.35), sharey=False)
    annotation_offsets = {
        ("mnist", "Local-only", "mobile"): (4, -8),
        ("mnist", "Stats-only", "mobile"): (4, 6),
        ("mnist", "PALM-transfer", "mobile"): (4, -13),
        ("mnist", "PALM-transfer", "random"): (4, 8),
        ("mnist", "FedProto-style", "mobile"): (4, 7),
        ("mnist", "FedMD-proxy", "mobile"): (4, -10),
        ("mnist", "FedAvg-small_cnn", "mobile"): (4, -12),
        ("cifar10", "Local-only", "mobile"): (4, -12),
        ("cifar10", "Stats-only", "mobile"): (4, 8),
        ("cifar10", "PALM-transfer", "mobile"): (-42, -12),
        ("cifar10", "PALM-transfer", "random"): (4, 8),
        ("cifar10", "FedAvg-small_cnn", "mobile"): (4, -12),
    }
    for ax, dataset, title in zip(axes, ["mnist", "cifar10"], ["MNIST", "CIFAR-10"]):
        for row in [r for r in rows if r.get("dataset") == dataset]:
            color = color_for(row)
            is_fedavg = label_for(row) == "FedAvg"
            marker = "*" if is_fedavg else "o"
            size = 86 if is_fedavg else 44
            ax.scatter(
                f(row, "energy_j"),
                100.0 * f(row, "final_accuracy"),
                c=color,
                marker=marker,
                s=size,
                alpha=0.9,
                edgecolors="#333333" if is_fedavg else "none",
                linewidths=0.5,
            )
            offset = annotation_offsets.get((row.get("dataset", ""), row.get("mode", ""), row.get("scheduler", "")), (4, 4))
            ax.annotate(
                short_mode(row),
                (f(row, "energy_j"), 100.0 * f(row, "final_accuracy")),
                fontsize=6.5,
                xytext=offset,
                textcoords="offset points",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.35},
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Predicted round energy (J)")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.margins(y=0.14, x=0.08)
    axes[0].set_ylabel("Final accuracy (%)")
    handles = []
    labels = []
    present = {label_for(row) for row in rows}
    for mode, color in COLORS.items():
        if mode not in present:
            continue
        marker = "*" if mode == "FedAvg" else "o"
        handles.append(plt.Line2D([0], [0], marker=marker, linestyle="", color=color))
        labels.append(mode)
    fig.legend(handles, labels, ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.52, 1.02))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_comm_privacy(rows: list[dict[str, str]], out: Path) -> None:
    rows = [
        r
        for r in mean_by_operating_point(rows)
        if r.get("mode") in {"Local-only", "Stats-only", "PALM-transfer"} and not r.get("scheduler_ablation")
    ]
    labels = [short_mode(r) for r in rows]
    x = list(range(len(rows)))
    uplink = [f(r, "uplink_mb") for r in rows]
    downlink = [f(r, "downlink_mb") for r in rows]
    eps = [f(r, "epsilon_max") for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.15), sharex=True)
    ax1.bar(x, uplink, color="#4c78a8", label="uplink")
    ax1.bar(x, downlink, bottom=uplink, color="#f58518", label="downlink")
    ax1.set_ylabel("Payload (MB)")
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    ax2.bar(x, eps, color="#72b7b2")
    ax2.set_title("Client-summary privacy accounting", fontsize=8, pad=2)
    ax2.set_ylabel(r"$\varepsilon_{\max}$")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax2.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    for ax in (ax1, ax2):
        mnist_idxs = [i for i, r in enumerate(rows) if r.get("dataset") == "mnist"]
        cifar_idxs = [i for i, r in enumerate(rows) if r.get("dataset") == "cifar10"]
        if mnist_idxs and cifar_idxs:
            ax.axvline((max(mnist_idxs) + min(cifar_idxs)) / 2.0, color="#999999", linewidth=0.5, alpha=0.7)
    for dataset, name in [("mnist", "MNIST"), ("cifar10", "CIFAR-10")]:
        idxs = [i for i, r in enumerate(rows) if r.get("dataset") == dataset]
        if idxs:
            ax2.text((min(idxs) + max(idxs)) / 2.0, -0.34, name, ha="center", va="top", fontsize=7, transform=ax2.get_xaxis_transform())
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(rows: list[dict[str, str]], out: Path) -> None:
    selected_modes = {"Local-only", "Stats-only", "PALM-transfer", "FedMD-proxy"}
    rows = [
        r
        for r in rows
        if r.get("mode") in selected_modes
        and r.get("dataset") in {"mnist", "cifar10"}
        and not r.get("scheduler_ablation")
    ]
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
        ax.set_title("CIFAR-10" if dataset == "cifar10" else dataset.upper())
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
