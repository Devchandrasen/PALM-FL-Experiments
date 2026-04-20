#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT if (ROOT / "main.tex").exists() else ROOT / "paper"
REPRO_ROOT = ROOT / "reproducibility"
if not REPRO_ROOT.exists():
    REPRO_ROOT = ROOT / "paper_reproducibility"


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def require(path: Path) -> None:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(ROOT)}")


def pdf_text() -> str:
    require(PAPER_ROOT / "main.pdf")
    try:
        proc = subprocess.run(
            ["pdftotext", str(PAPER_ROOT / "main.pdf"), "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        fail(f"could not extract text from main.pdf: {exc}")
    return proc.stdout


def check_no_placeholders(tex: str, text: str) -> None:
    banned = [
        "insert URL here",
        "anonymized artifact repository URL should be inserted",
        "earlier draft",
        "major-revision",
        "guaranteed-acceptance",
        "outperforms FedAvg",
        "strong differential privacy",
        "real mobile energy measurement",
    ]
    haystack = (tex + "\n" + text).lower()
    for phrase in banned:
        if phrase.lower() in haystack:
            fail(f"banned placeholder/claim found: {phrase}")


def check_references(text: str) -> None:
    for marker in ["Figure ??", "Table ??", "Section ??", "??"]:
        if marker in text:
            fail(f"unresolved reference marker found: {marker}")


def check_no_misleading_epsilon(tex: str) -> None:
    labels = ("FedAvg", "FedMD-style", r"\localonly", "local-only")
    for line in tex.splitlines():
        if not any(label in line for label in labels):
            continue
        if "&" not in line or r"\\" not in line:
            continue
        last_cell = line.rsplit("&", 1)[-1].replace(r"\\", "").strip()
        if last_cell == "0.00":
            fail(f"misleading epsilon zero in final table cell: {line.strip()}")


def check_generated_tables() -> None:
    expected = [
        "table_operating_modes",
        "table_baselines",
        "table_privacy_accounting",
        "table_ablation_status",
        "table_architecture_fairness",
    ]
    for stem in expected:
        require(PAPER_ROOT / "generated_tables" / f"{stem}.csv")
        require(PAPER_ROOT / "generated_tables" / f"{stem}.tex")


def check_curated_results() -> None:
    path = REPRO_ROOT / "tmc_trace_grouped_results.csv"
    require(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail("curated grouped results are empty")
    c10 = [r for r in rows if r.get("dataset") == "cifar10" and r.get("split_protocol") == "independent"]
    complete_cifar_multiseed = any(int(float(r.get("n", "0") or 0)) >= 3 for r in c10)
    manuscript = read_text(PAPER_ROOT / "main.tex").lower()
    if not complete_cifar_multiseed and "case study" not in manuscript:
        fail("CIFAR-10 is not multi-seed but manuscript does not label it as a case study")


def check_log() -> None:
    log = PAPER_ROOT / "main.log"
    if not log.exists():
        print("WARN: main.log not found; skipping LaTeX log check")
        return
    content = read_text(log)
    bad = ["LaTeX Warning", "Undefined", "Overfull", "! LaTeX Error", "! Package"]
    for marker in bad:
        if marker in content:
            fail(f"LaTeX issue found in main.log: {marker}")


def main() -> int:
    tex = read_text(PAPER_ROOT / "main.tex")
    text = pdf_text()
    check_no_placeholders(tex, text)
    check_references(text)
    check_no_misleading_epsilon(tex)
    check_generated_tables()
    check_curated_results()
    check_log()
    print("PASS: result/package consistency checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
