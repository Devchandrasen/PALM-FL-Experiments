# PALM-FL Project Implementation

This document describes the implementation packaged for the PALM-FL TMC revision artifact.

## 1. Implementation Overview

PALM-FL implements a heterogeneous federated learning workflow in which each client keeps its own model architecture and exposes compact class-wise summaries through a shared latent adapter dimension. The server does not aggregate full client models in the PALM-FL operating modes. Instead, it maintains a summary memory and uses that memory for scheduling and, optionally, masked stats-transfer.

The core implementation is organized as follows:

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

## 2. PALM-FL Operating Modes

The packaged experiment scripts support the following modes.

### Local-only

Clients train and evaluate local models with no server coordination. This is the baseline for determining whether PALM-FL preserves or improves local utility.

### Stats-only PALM-FL

Clients upload clipped-and-noised class-wise summaries. The server updates memory and schedules future participants. No models, gradients, logits, prototypes, or latent means are sent back to clients.

Properties:

- Zero PALM-FL downlink payload.
- Architecture-agnostic server path, assuming all clients expose the agreed latent dimension.
- Useful primarily as a low-payload scheduling and coordination path.

### Masked stats-transfer PALM-FL

The server uses its memory to form masked class-wise transfer objects for participating clients. This path can improve accuracy when client participation provides adequate class coverage, but it is scheduler-sensitive and not a replacement for full-model FL.

### Histogram-only ablation

Only noisy class counts are released. Latent means and variances are disabled. This isolates whether class-count balancing explains stats-only behavior.

### Latent-mean-only ablation

No class-deficit scheduler signal is used. The mode tests whether latent summaries alone contribute when count-driven deficit scheduling is removed.

### No-noise diagnostic

Gaussian noise is disabled. This is a non-private diagnostic for measuring the utility cost of the noise mechanism.

### Count-only transfer

Clients receive class-prior or availability information without prototype vectors. This tests whether transfer gains require latent prototype vectors.

### No-deficit scheduler

The scheduler class-deficit weight is set to zero while other scheduling terms are kept. This tests whether observed gains are driven mainly by the deficit term.

## 3. Scheduler and Mobile Cost Model

The scheduler combines utility and mobile-system terms, including resource budget, bandwidth-derived upload/download cost, battery state, staleness, and recent performance. The packaged trace profiles are derived from public mobile bandwidth measurements. Compute throughput, battery, recharge, and energy terms remain modeled unless real-device calibration is added.

The paper reports these values as trace-calibrated operating estimates, not measured phone energy.

## 4. Privacy Accounting

The privacy unit is the clipped client-summary object. The accountant reports participation-counted Gaussian release values converted through a zCDP-style expression.

The implementation tracks:

- Released summary type.
- Clipping threshold.
- Noise multiplier.
- Participation count.
- Delta.
- Maximum client-summary epsilon across clients.

The packaged paper and tables use `N/A` or `not DP-accounted` for methods that do not implement the same summary-release mechanism. They do not report epsilon as zero for non-DP baselines.

## 5. Baseline Separation

The artifact separates baseline categories:

- Homogeneous full-model references: FedAvg.
- Heterogeneous/proxy/prototype references: FedMD-style and FedProto-style references where available.
- PALM-FL low-payload operating modes: stats-only, transfer, scheduler variants, and ablations.

This separation prevents homogeneous full-model sharing from being interpreted as a like-for-like competitor to PALM-FL's architecture-flexible low-payload path.

## 6. Result Artifacts

Curated result CSVs are packaged in `analysis/` and mirrored in `paper_reproducibility/`. Generated LaTeX tables and figures are packaged under `paper/generated_tables/` and `paper/figures/`.

Important files:

```text
analysis/tmc_trace_all_results.csv
analysis/tmc_trace_grouped_results.csv
analysis/tmc_trace_architecture_fairness.csv
analysis/grouped_results.csv
paper/generated_tables/*.tex
paper/figures/*.pdf
paper/figures/*.png
```

The script `scripts/check_results_consistency.py` checks that tables, figures, payload values, and privacy values are generated from the current result artifacts rather than hand-entered.

## 7. Reproducibility Commands

Smoke test:

```bash
python -m palmfl.main --config configs/palmfl_fake_smoke.yaml
```

Full revised experiments:

```bash
bash scripts/run_tmc_revised_experiments.sh
```

Ablation experiments:

```bash
SEEDS="42 43 44" DATASETS="mnist cifar10" DEVICE=cuda PY=python scripts/run_tmc_ablation_experiments.sh
```

Tables, figures, and checks:

```bash
python scripts/make_paper_tables.py
python scripts/make_paper_figures.py
python scripts/check_results_consistency.py
```

Paper build:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## 8. What Is Not Included

The artifact intentionally excludes:

- Downloaded MNIST/CIFAR-10 dataset caches.
- Large raw experiment output directories.
- Any fabricated real-device energy measurements.
- Any private GitHub tokens or credentials.

The scripts regenerate datasets and outputs as needed.

## 9. Current Research Status

The implementation and paper package are ready for public review and reproducibility inspection. The scientific claims remain modest: PALM-FL is presented as a low-payload heterogeneous coordination framework with summary-level accounting and trace-calibrated cost estimates. It is not claimed to be a strong sample-level DP method, a measured phone-energy study, or a universal accuracy replacement for homogeneous FedAvg.
