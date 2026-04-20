# PALM-FL Reproducibility

This package contains the manuscript, curated result files, figure/table scripts, and links to the full local experiment runner used for the current PALM-FL revision.

The public artifact repository is available at:

```text
https://github.com/Devchandrasen/PALM-FL-TMC-Artifact
```

## Layout

| Path | Purpose |
|---|---|
| `main.tex`, `main.pdf` | Manuscript source and compiled PDF. |
| `figures/` | Figure PDFs used by the manuscript. |
| `reproducibility/tmc_trace_all_results.csv` | Curated completed trace-derived run rows. |
| `reproducibility/tmc_trace_grouped_results.csv` | Grouped mean/std rows used for tables. |
| `reproducibility/tmc_trace_architecture_fairness.csv` | Per-architecture fairness aggregation. |
| `reproducibility/mobilebandwidth/real_mobile_profiles.csv` | Deterministic ten-client trace-derived bandwidth profiles. |
| `reproducibility/scripts/` | Existing aggregation, trace-profile, curation, and figure scripts. |
| `scripts/` | TMC package wrappers for reruns, table generation, figure generation, and consistency checks. |
| `generated_tables/` | Generated CSV/LaTeX tables from current artifacts. |

The full runnable PALM codebase used by the existing run scripts is currently located at:

```bash
/home/dr-chandrasen-pandey/Desktop/My Experiments/palmfl_v8_pivot/palmfl_v8_pivot
```

Set `PALM_RUNNER_ROOT` if your checkout is elsewhere.

## Environment

The existing scripts assume a Python environment with PyTorch, torchvision, numpy, pandas-compatible CSV handling, PyYAML, matplotlib, and a LaTeX toolchain. The local run scripts default to:

```bash
PY=/home/dr-chandrasen-pandey/anaconda3/envs/palmfl310/bin/python
```

Override it when needed:

```bash
export PY=/path/to/python
```

## Dataset Download

MNIST and CIFAR-10 are loaded by the PALM runner. If the datasets are not already present, the runner downloads them through torchvision according to the dataset configuration files under the runner's `configs/` directory.

## Trace Profile Construction

From the full runner root:

```bash
cd "$PALM_RUNNER_ROOT"
"${PY:-python}" scripts/build_real_mobile_profiles.py --num-clients 10
```

The manuscript package stores the curated profile at:

```bash
reproducibility/mobilebandwidth/real_mobile_profiles.csv
```

## Current Completed Runs

The current manuscript uses completed artifacts summarized in:

```bash
reproducibility/tmc_trace_all_results.csv
reproducibility/tmc_trace_grouped_results.csv
reproducibility/tmc_trace_architecture_fairness.csv
```

MNIST operating-mode, baseline, and ablation rows use seeds 42, 43, and 44 under independent partition and architecture seeds. CIFAR-10 ablation rows also use seeds 42, 43, and 44. The primary CIFAR-10 operating-mode and baseline rows remain a seed-42 trace-calibrated case study in the curated independent-split results.

## Rerun Commands

Run the current MNIST campaign:

```bash
cd /home/dr-chandrasen-pandey/PALM_review
PALM_RUNNER_ROOT="/home/dr-chandrasen-pandey/Desktop/My Experiments/palmfl_v8_pivot/palmfl_v8_pivot" \
  bash scripts/run_all_mnist.sh
```

Run the current CIFAR campaign. This may take several hours:

```bash
cd /home/dr-chandrasen-pandey/PALM_review
PALM_RUNNER_ROOT="/home/dr-chandrasen-pandey/Desktop/My Experiments/palmfl_v8_pivot/palmfl_v8_pivot" \
  bash scripts/run_all_cifar.sh
```

Run the ablation campaign from the full runner. The current curated package includes completed MNIST and CIFAR-10 seeds 42, 43, and 44 ablation artifacts:

```bash
cd /home/dr-chandrasen-pandey/PALM_review
PALM_RUNNER_ROOT="/home/dr-chandrasen-pandey/Desktop/My Experiments/palmfl_v8_pivot/palmfl_v8_pivot" \
  SEEDS="42 43 44" DATASETS="mnist cifar10" PY=python3 \
  bash scripts/run_ablation_campaign.sh
```

## Table and Figure Generation

Generate CSV/LaTeX tables from current artifacts:

```bash
cd /home/dr-chandrasen-pandey/PALM_review
python3 scripts/make_tables.py
```

Regenerate manuscript figures:

```bash
cd /home/dr-chandrasen-pandey/PALM_review
PALM_RUNNER_ROOT="/home/dr-chandrasen-pandey/Desktop/My Experiments/palmfl_v8_pivot/palmfl_v8_pivot" \
  python3 scripts/make_figures.py
```

Compile the manuscript:

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Run consistency checks:

```bash
python3 scripts/check_results_consistency.py
```

## Expected Outputs

| Command | Expected outputs |
|---|---|
| `scripts/make_tables.py` | `generated_tables/*.csv`, `generated_tables/*.tex` |
| `scripts/make_figures.py` | `figures/fig2_learning_curves.pdf`, `figures/fig5_accuracy_energy_frontier.pdf`, `figures/fig6_comm_privacy_bars.pdf`, `figures/fig7_architecture_fairness.pdf` |
| `pdflatex` | `main.pdf` |
| `scripts/check_results_consistency.py` | Exit code 0 only when current manuscript/package checks pass |

## Hardware and Runtime Notes

The completed trace-derived MNIST operating-mode campaign and MNIST/CIFAR-10 ablation campaigns were run locally before this package update. The ablation diagnostics can run on CPU, but the local `palmfl310` environment was used for CUDA runs on the visible GTX 1050. The full CIFAR main operating-mode/baseline campaign is substantially more expensive on this setup and is incomplete in the curated artifacts beyond the seed-42 case-study rows. Real-device energy calibration has not been run.

## Incomplete Items

The following TMC-readiness items are documented but not completed in current artifacts:

- multi-seed independent CIFAR-10 main operating-mode/baseline campaign,
- real-device calibration or cost-model sensitivity campaign,
- stronger hetero-FL baselines such as HeteroFL/FjORD/FedDF/FedGH/GEFL.
