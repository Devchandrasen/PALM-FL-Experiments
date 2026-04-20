# PALM-FL Project Implementation

This document describes the experiment implementation packaged in this repository.

## 1. Core Modules

```text
palmfl/client.py        Client training, evaluation, and summary extraction
palmfl/data.py          Dataset loading, client partitioning, and non-IID splits
palmfl/dp.py            Clipping, Gaussian release accounting, and epsilon conversion
palmfl/latent_stats.py  Class-wise latent count, mean, and variance summaries
palmfl/server.py        Server memory, summary update, and optional transfer objects
palmfl/scheduler.py     Resource-aware and random participation scheduling
palmfl/models.py        Heterogeneous local architectures and adapters
palmfl/main.py          PALM-FL experiment entry point
palmfl/fedavg_main.py   Homogeneous FedAvg reference entry point
palmfl/fedmd_main.py    FedMD-style reference entry point
palmfl/metrics.py       Accuracy, macro-F1, fairness, payload, and cost metrics
```

## 2. Experiment Modes

### Local-only

Clients train and evaluate local models without server coordination.

### Stats-only

Clients upload clipped-and-noised class-wise summaries. The server updates summary memory and schedules later participants. No model weights or prototypes are sent back to clients.

### Masked stats-transfer

The server uses the summary memory to form masked class-wise transfer objects. Clients use the transfer package for lightweight prototype/stat-head regularization.

### Histogram-only ablation

Only noisy class counts are released. Latent means and variances are disabled.

### Mean-only ablation

Latent means are released with count-driven deficit scheduling disabled or reduced to noisy availability.

### No-noise diagnostic

Gaussian noise is disabled. This is a non-private utility diagnostic.

### Count-only transfer

Clients receive count or class-availability information without prototype vectors.

### No-deficit scheduler

The scheduler class-deficit weight is set to zero while the remaining scheduling terms stay active.

## 3. Scheduler and Cost Model

The scheduler can combine utility, staleness, bandwidth, battery, upload/download payload, predicted time, and predicted energy terms. The packaged mobile profile file is:

```text
data/mobilebandwidth/real_mobile_profiles.csv
```

Bandwidth profiles are trace-derived. Compute throughput, energy, battery, and recharge behavior are model parameters unless a separate real-device calibration is added.

## 4. Privacy Accounting

The accounting unit is the clipped client-summary object. The implementation tracks released summary objects, clipping thresholds, noise multiplier, participation count, delta, and maximum client-summary epsilon across clients.

Methods without the same summary-release mechanism are marked as not DP-accounted in result interpretation. They should not be treated as having epsilon equal to zero.

## 5. Baselines

The repository includes:

- homogeneous FedAvg reference runs through `palmfl/fedavg_main.py`,
- FedMD-style proxy-logit reference runs through `palmfl/fedmd_main.py`,
- PALM-FL low-payload modes and ablations through `palmfl/main.py`.

## 6. Result Artifacts

Curated result CSVs are stored under `analysis/`:

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

Diagnostic plots can be regenerated into `analysis/figures/`.

## 7. Main Commands

Smoke test:

```bash
python -m palmfl.main --config configs/palmfl_fake_smoke.yaml
```

Revised experiment sweep:

```bash
PY=python3 DEVICE=cuda bash scripts/run_revised_experiments.sh
```

Ablation sweep:

```bash
SEEDS="42 43 44" DATASETS="mnist cifar10" DEVICE=cuda PY=python3 bash scripts/run_ablation_experiments.sh
```

Aggregation and curation:

```bash
python3 scripts/aggregate_results.py
python3 scripts/curate_trace_results.py
```

Diagnostic plots:

```bash
python3 scripts/plot_experiment_figures.py \
  --results analysis/trace_all_results.csv \
  --fairness analysis/trace_architecture_fairness.csv \
  --figdir analysis/figures
```

Consistency check:

```bash
python3 scripts/check_results_consistency.py
```

## 8. Excluded Files

The repository intentionally excludes downloaded dataset caches, large raw output directories, local virtual environments, and private credentials.
