# PALM-FL Experiment Reproducibility

This repository is an experiment-only package. It contains runnable code, configs, curated CSV outputs, and scripts for reproducing or extending the PALM-FL experiment matrix.

Repository:

```text
https://github.com/Devchandrasen/PALM-FL-Experiments
```

## Environment

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA runs, use a PyTorch build compatible with your local driver. Every script accepts `PY=/path/to/python` so an existing environment can be used directly.

## Dataset Download

MNIST and CIFAR-10 are loaded through torchvision. If the datasets are missing locally, the loader downloads them according to the configuration file being run. Dataset caches are excluded from git.

## Mobile Trace Profiles

The packaged mobile profile file is:

```text
data/mobilebandwidth/real_mobile_profiles.csv
```

Regenerate it with:

```bash
python3 scripts/build_real_mobile_profiles.py --num-clients 10
```

## Smoke Test

```bash
python -m palmfl.main --config configs/palmfl_fake_smoke.yaml
```

## Full Experiment Sweeps

Revised operating modes and reference baselines:

```bash
PY=python3 DEVICE=cuda bash scripts/run_revised_experiments.sh
```

Ablation sweep:

```bash
SEEDS="42 43 44" DATASETS="mnist cifar10" DEVICE=cuda PY=python3 bash scripts/run_ablation_experiments.sh
```

The sweeps can be expensive. For quick validation, set `DEVICE=cpu` and reduce `SEEDS` or `DATASETS`.

## Aggregation

Raw run folders are expected under `outputs/`. Aggregate them with:

```bash
python3 scripts/aggregate_results.py
python3 scripts/curate_trace_results.py
```

Expected curated outputs:

```text
analysis/trace_all_results.csv
analysis/trace_grouped_results.csv
analysis/trace_architecture_fairness.csv
```

## Diagnostic Figures

Diagnostic plots can be generated from the curated CSVs:

```bash
python3 scripts/plot_experiment_figures.py \
  --results analysis/trace_all_results.csv \
  --fairness analysis/trace_architecture_fairness.csv \
  --figdir analysis/figures
```

Generated plots are written under `analysis/figures/`.

## Consistency Check

```bash
python3 scripts/check_results_consistency.py
```

The checker verifies that required code, config, and result files exist, curated CSVs are non-empty, and repository documentation remains experiment-focused.

## Current Packaged Results

The committed CSVs include completed MNIST and CIFAR-10 experiment artifacts from the current implementation snapshot, including ablation rows for seeds 42, 43, and 44 where present. Some larger runs remain incomplete and should be regenerated before making stronger claims:

- full multi-seed CIFAR-10 main operating-mode and baseline sweep,
- real-device latency or energy calibration,
- additional heterogeneous FL baselines beyond the packaged references.
