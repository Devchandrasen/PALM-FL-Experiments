#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
GROUPED = ROOT / "reproducibility" / "tmc_trace_grouped_results.csv"
FAIRNESS = ROOT / "reproducibility" / "tmc_trace_architecture_fairness.csv"
OUT = ROOT / "generated_tables"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
        return value if math.isfinite(value) else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def fmt_pct(row: dict[str, str], key: str, std_key: str, n_key: str = "n") -> str:
    mean = 100.0 * f(row, key)
    std = 100.0 * f(row, std_key)
    n = int(float(row.get(n_key, "1") or 1))
    return f"{mean:.2f}±{std:.2f}" if n > 1 else f"{mean:.2f}"


def fmt_float(row: dict[str, str], key: str, std_key: str | None = None, digits: int = 4) -> str:
    mean = f(row, key)
    if std_key and int(float(row.get("n", "1") or 1)) > 1:
        std = f(row, std_key)
        return f"{mean:.{digits}f}±{std:.{digits}f}"
    return f"{mean:.{digits}f}"


def eps_status(row: dict[str, str]) -> str:
    mode = row.get("mode", "")
    if mode in {"Local-only", "FedAvg-small_cnn", "FedMD-proxy"}:
        return "not DP-accounted"
    if row.get("dp_enabled", "true") == "false" or mode.endswith("-no-noise"):
        return "non-private diagnostic"
    return f"{f(row, 'epsilon_max_mean'):.2f}"


def mode_label(row: dict[str, str]) -> str:
    mode = row.get("mode", "")
    scheduler = row.get("scheduler", "")
    if mode == "PALM-transfer":
        return "stats-transfer-R" if scheduler == "random" else "stats-transfer-M"
    if mode == "PALM-transfer-no-noise":
        return "transfer no-noise"
    if mode == "Stats-only-no-noise":
        return "stats-only no-noise"
    if mode == "Count-only-transfer":
        return "count-only transfer"
    if mode == "Histogram-only":
        return "histogram-only"
    if mode == "Mean-only":
        return "mean-only"
    if mode == "FedAvg-small_cnn":
        return "FedAvg"
    if mode == "FedMD-proxy":
        return "FedMD-style"
    return mode


def select(
    rows: list[dict[str, str]],
    dataset: str,
    mode: str,
    scheduler: str,
    scheduler_ablation: str | None = "",
) -> dict[str, str] | None:
    return next(
        (
            r
            for r in rows
            if r.get("dataset") == dataset
            and r.get("mode") == mode
            and r.get("scheduler") == scheduler
            and (scheduler_ablation is None or r.get("scheduler_ablation", "") == scheduler_ablation)
        ),
        None,
    )


def result_row(row: dict[str, str], interpretation: str = "") -> dict[str, str]:
    return {
        "Dataset": "CIFAR-10" if row["dataset"] == "cifar10" else "MNIST",
        "Mode": mode_label(row),
        "Scheduler": (
            f"{row.get('scheduler', '')}/{row.get('scheduler_ablation', '')}"
            if row.get("scheduler_ablation")
            else row.get("scheduler", "")
        ),
        "Accuracy": fmt_pct(row, "final_accuracy_mean", "final_accuracy_std"),
        "Macro-F1": fmt_float(row, "final_macro_f1_mean", "final_macro_f1_std", 4),
        "Upload MB": fmt_float(row, "uplink_mb_mean", None, 4),
        "Download MB": fmt_float(row, "downlink_mb_mean", None, 4),
        "Predicted time s": fmt_float(row, "time_s_mean", None, 2),
        "Predicted energy J": fmt_float(row, "energy_j_mean", None, 2),
        "Epsilon/accounting": eps_status(row),
        "Interpretation": interpretation,
    }


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "±": r"$\pm$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex(path: Path, rows: list[dict[str, str]], caption: str, label: str) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    align = "l" * len(columns)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
    ]
    use_resize = len(columns) > 8
    if use_resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend([
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in columns) + r" \\",
        r"\midrule",
    ])
    for row in rows:
        lines.append(" & ".join(latex_escape(row[c]) for c in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if use_resize:
        lines.append(r"}")
    lines.extend([r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_operating(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for dataset in ["mnist", "cifar10"]:
        for mode, scheduler, interpretation in [
            ("Local-only", "mobile", "local client training reference"),
            ("Stats-only", "mobile", "zero-downlink clipped summary scheduling"),
            ("PALM-transfer", "mobile", "masked transfer with mobile-aware selection"),
            ("PALM-transfer", "random", "masked transfer with broader random coverage"),
        ]:
            row = select(rows, dataset, mode, scheduler)
            if row:
                out.append(result_row(row, interpretation))
    return out


def make_baselines(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for dataset, mode, scheduler, interpretation in [
        ("mnist", "FedAvg-small_cnn", "mobile", "homogeneous full-model reference"),
        ("mnist", "FedProto-style", "mobile", "heterogeneous prototype-style reference"),
        ("mnist", "FedMD-proxy", "mobile", "heterogeneous proxy-logit reference"),
        ("cifar10", "FedAvg-small_cnn", "mobile", "homogeneous full-model case-study reference"),
    ]:
        row = select(rows, dataset, mode, scheduler)
        if row:
            out.append(result_row(row, interpretation))
    return out


def make_privacy(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    modes = [
        ("Local-only", "mobile", "none", "--", "--", "not DP-accounted", "no summary-release privacy accounting"),
        ("Stats-only", "mobile", "counts, means, variances", "S_mu=S_v=4; S_n=200", "0.5", "client-summary", "high epsilon; not sample-level DP"),
        ("PALM-transfer", "mobile", "counts, means, variances; prototype package", "S_mu=S_v=4; S_n=200", "0.5", "client-summary", "high epsilon; not sample-level DP"),
        ("PALM-transfer", "random", "counts, means, variances; prototype package", "S_mu=S_v=4; S_n=200", "0.5", "client-summary", "high epsilon; not sample-level DP"),
        ("Histogram-only", "mobile", "counts and class mask", "S_n=200", "0.5", "client-summary", "counts-only release; not sample-level DP"),
        ("Mean-only", "mobile", "latent means and noisy availability", "S_mu=4; S_a=1", "0.5", "client-summary", "mean-release diagnostic; not sample-level DP"),
        ("FedAvg-small_cnn", "mobile", "model weights", "--", "--", "not DP-accounted", "not private by this accountant"),
        ("FedMD-proxy", "mobile", "proxy logits", "--", "--", "not DP-accounted", "not private by this accountant"),
    ]
    out = []
    for dataset in ["mnist", "cifar10"]:
        for mode, scheduler, released, clipping, sigma, unit, interp in modes:
            row = select(rows, dataset, mode, scheduler)
            if not row:
                continue
            out.append(
                {
                    "Dataset": "CIFAR-10" if dataset == "cifar10" else "MNIST",
                    "Mode": mode_label(row),
                    "Released objects": released,
                    "Clipping threshold": clipping,
                    "Noise multiplier": sigma,
                    "Delta": "1e-5" if sigma != "--" else "--",
                    "Accounting unit": unit,
                    "Epsilon max": eps_status(row),
                    "Interpretation": interp,
                }
            )
    return out


def missing_ablation_row(dataset: str, mode: str, purpose: str) -> dict[str, str]:
    return {
        "Dataset": dataset,
        "Mode": mode,
        "Scheduler": "--",
        "Accuracy": "--",
        "Macro-F1": "--",
        "Upload MB": "--",
        "Download MB": "--",
        "Predicted time s": "--",
        "Predicted energy J": "--",
        "Epsilon/accounting": "not completed",
        "Interpretation": purpose,
    }


def make_ablation_status(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    specs = [
        ("Histogram-only", "mobile", "", "counts-only scheduler control"),
        ("Histogram-only", "random", "", "counts-only random-participation control"),
        ("Mean-only", "mobile", "", "latent means with noisy availability"),
        ("Mean-only", "random", "", "latent means under random participation"),
        ("Stats-only-no-noise", "mobile", "", "non-private summary diagnostic"),
        ("PALM-transfer-no-noise", "mobile", "", "non-private transfer diagnostic"),
        ("Count-only-transfer", "mobile", "", "transfer package without prototypes"),
        ("Stats-only", "mobile", "no_deficit", "full-summary scheduler without released-count deficit"),
        ("PALM-transfer", "mobile", "no_deficit", "mobile transfer without released-count deficit"),
        ("PALM-transfer", "random", "no_deficit", "random transfer without released-count deficit"),
    ]
    for dataset in ["mnist", "cifar10"]:
        seed_note = "three MNIST seeds" if dataset == "mnist" else "three CIFAR-10 seeds"
        for mode, scheduler, ablation, interpretation in specs:
            row = select(rows, dataset, mode, scheduler, ablation)
            if row:
                out.append(result_row(row, f"{interpretation}; {seed_note}"))
            else:
                dataset_label = "CIFAR-10" if dataset == "cifar10" else "MNIST"
                out.append(
                    missing_ablation_row(
                        dataset_label,
                        mode_label(mode, scheduler, ablation),
                        f"{interpretation}; not completed",
                    )
                )
    return out


def make_fairness(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if row.get("mode") != "PALM-transfer" or row.get("scheduler") != "random":
            continue
        out.append(
            {
                "Dataset": "CIFAR-10" if row["dataset"] == "cifar10" else "MNIST",
                "Architecture": row["arch_name"],
                "Mean accuracy": f"{100.0 * f(row, 'accuracy_mean'):.2f}",
                "Std accuracy": f"{100.0 * f(row, 'accuracy_std'):.2f}",
                "Mean macro-F1": f"{f(row, 'macro_f1_mean'):.4f}",
                "Std macro-F1": f"{f(row, 'macro_f1_std'):.4f}",
                "Client count": row.get("n_clients", ""),
                "Seeds": row.get("seeds", ""),
            }
        )
    return out


def emit(name: str, rows: list[dict[str, str]], caption: str, label: str) -> None:
    write_csv(OUT / f"{name}.csv", rows)
    write_latex(OUT / f"{name}.tex", rows, caption, label)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    grouped = read_rows(GROUPED)
    fairness = read_rows(FAIRNESS)
    emit("table_operating_modes", make_operating(grouped), "PALM-FL operating modes from completed trace-derived artifacts.", "tab:generated_operating")
    emit("table_baselines", make_baselines(grouped), "Baseline references from completed trace-derived artifacts.", "tab:generated_baselines")
    emit("table_privacy_accounting", make_privacy(grouped), "Privacy accounting parameters and status.", "tab:generated_privacy")
    emit("table_ablation_status", make_ablation_status(grouped), "Ablation study isolating count, latent, noise, transfer, and scheduler effects. Completed MNIST and CIFAR-10 ablation rows use seeds 42, 43, and 44.", "tab:generated_ablation_status")
    emit("table_architecture_fairness", make_fairness(fairness), "Architecture-level fairness summary for PALM random transfer.", "tab:generated_fairness")
    print(f"Wrote generated tables under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
