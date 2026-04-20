# PALM-FL Experiments

This repository contains the PALM-FL experiment implementation, configuration files, analysis CSVs, and rerun utilities.

Repository:

https://github.com/Devchandrasen/PALM-FL-Experiments

## Scope

PALM-FL is a heterogeneous federated learning experiment framework. Each client keeps a local model architecture and communicates compact class-wise summaries through a shared latent adapter dimension. The implementation supports:

- local-only training,
- stats-only client-summary upload,
- masked stats-transfer,
- resource-aware and random client scheduling,
- clipped-and-noised summary release accounting,
- FedAvg and FedMD-style reference runs,
- histogram-only, mean-only, no-noise, count-only-transfer, and no-deficit ablations.

This repository is for running and analyzing experiments only.

## Repository Layout

```text
palmfl/                  Core PALM-FL implementation
configs/                 Experiment configuration files
scripts/                 Experiment, aggregation, curation, plotting, and consistency scripts
analysis/                Curated result CSVs from completed runs
data/mobilebandwidth/    Trace-derived mobile bandwidth profiles
docs/                    Experiment audit and change log
requirements.txt         Python dependencies
README_REPRODUCIBILITY.md
PROJECT_IMPLEMENTATION.md
RUNNER_README.md
```

Dataset caches and raw output directories are not committed. MNIST and CIFAR-10 are downloaded by torchvision when required, and new run artifacts are written under `outputs/`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m palmfl.main --config configs/palmfl_fake_smoke.yaml
```

The smoke run checks the main PALM-FL code path without executing the full experiment matrix.

## Main Commands

Run the revised operating-mode and baseline sweep:

```bash
PY=python3 DEVICE=cuda bash scripts/run_revised_experiments.sh
```

Run the ablation sweep:

```bash
SEEDS="42 43 44" DATASETS="mnist cifar10" DEVICE=cuda PY=python3 bash scripts/run_ablation_experiments.sh
```

Aggregate and curate existing outputs:

```bash
python3 scripts/aggregate_results.py
python3 scripts/curate_trace_results.py
```

Generate diagnostic plots from curated CSVs:

```bash
python3 scripts/plot_experiment_figures.py \
  --results analysis/trace_all_results.csv \
  --fairness analysis/trace_architecture_fairness.csv \
  --figdir analysis/figures
```

Validate the repository package:

```bash
python3 scripts/check_results_consistency.py
```

## Packaged Result Files

```text
analysis/all_results.csv
analysis/grouped_results.csv
analysis/client_fairness.csv
analysis/architecture_fairness.csv
analysis/mobile_trace_summary.csv
analysis/trace_all_results.csv
analysis/trace_grouped_results.csv
analysis/trace_architecture_fairness.csv
```

These CSVs are included so experiment summaries can be inspected without rerunning the full training matrix.

## Notes

- Privacy values are client-summary accounting values, not sample-level end-to-end DP certificates.
- Time and energy values are trace-calibrated estimates from the experiment cost model, not measured phone energy.
- Full CIFAR-10 multi-seed main sweeps and real-device calibration remain open experiment extensions.
