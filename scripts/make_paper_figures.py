#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_ROOT = Path(os.environ.get("PALM_RUNNER_ROOT", "/home/dr-chandrasen-pandey/Desktop/My Experiments/palmfl_v8_pivot/palmfl_v8_pivot"))


def main() -> int:
    plot_script = ROOT / "reproducibility" / "scripts" / "plot_acceptance_figures.py"
    results = ROOT / "reproducibility" / "tmc_trace_all_results.csv"
    fairness = ROOT / "reproducibility" / "tmc_trace_architecture_fairness.csv"
    figdir = ROOT / "figures"
    cmd = [
        sys.executable,
        str(plot_script),
        "--results",
        str(results),
        "--fairness",
        str(fairness),
        "--split-protocol",
        "independent",
        "--figdir",
        str(figdir),
    ]
    subprocess.run(cmd, cwd=RUNNER_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
