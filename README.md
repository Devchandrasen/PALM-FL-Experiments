# PALM-FL TMC Artifact

This repository packages the PALM-FL implementation, curated result artifacts, paper source, generated figures, generated tables, and reproducibility scripts for:

**PALM-FL: Low-Payload Client-Summary Coordination and Resource-Aware Scheduling for Heterogeneous Mobile Federated Learning**

Public repository:

https://github.com/Devchandrasen/PALM-FL-TMC-Artifact

## Scope

PALM-FL is a low-payload, architecture-flexible coordination framework for heterogeneous mobile federated learning. The implementation supports clipped-and-noised client-summary release, server-side summary memory, resource-aware client scheduling, and optional masked stats-transfer.

The manuscript and artifact are intentionally conservative:

- Stats-only PALM-FL is a zero-downlink scheduling path, not full-model aggregation.
- Privacy values are client-summary accounting values, not sample-level end-to-end DP certificates.
- Time and energy values are trace-calibrated estimates unless a real-device calibration is explicitly supplied.
- Homogeneous FedAvg is reported as a full-model reference, not a like-for-like heterogeneous low-payload competitor.

## Repository Layout

```text
palmfl/                  Core PALM-FL implementation
configs/                 Experiment configuration files
scripts/                 Experiment, aggregation, table, figure, and consistency scripts
analysis/                Curated CSV result artifacts used by the paper
data/mobilebandwidth/    Public trace-derived mobile bandwidth profiles
paper/                   IEEE manuscript source, compiled PDF, figures, and generated tables
paper_reproducibility/   Self-contained paper reproduction bundle
docs/                    TMC readiness audit and revision changelog
requirements.txt         Python dependency list
README_REPRODUCIBILITY.md
PROJECT_IMPLEMENTATION.md
```

Large dataset caches and raw experiment output directories are not included. MNIST/CIFAR-10 are downloaded by the training scripts when needed, and new raw outputs are regenerated under `outputs/`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m palmfl.main --config configs/palmfl_fake_smoke.yaml
```

The smoke run verifies that the code path is functional without running the full experimental matrix.

## Reproducing Experiments

Main revised experiment sweep:

```bash
bash scripts/run_tmc_revised_experiments.sh
```

Ablation sweep:

```bash
SEEDS="42 43 44" DATASETS="mnist cifar10" DEVICE=cuda PY=python scripts/run_tmc_ablation_experiments.sh
```

The full sweeps are compute-intensive. Use `DEVICE=cpu` only for small smoke runs or debugging.

## Regenerating Tables and Figures

```bash
python scripts/make_paper_tables.py
python scripts/make_paper_figures.py
python scripts/check_results_consistency.py
```

The consistency checker validates that generated tables and figures are derived from the packaged CSV artifacts and recomputes the key privacy/payload checks.

## Building the Paper

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The current compiled manuscript is included at `paper/main.pdf`.

## Artifact Status

Completed and packaged:

- PALM-FL stats-only and masked stats-transfer implementation.
- Histogram-only, latent-mean-only, no-noise, count-only-transfer, and no-deficit ablation support.
- Curated MNIST and CIFAR-10 result CSVs used by the paper.
- Publication-quality regenerated figures and LaTeX tables.
- Reproducibility README, TMC readiness audit, and revision changelog.
- Public GitHub repository link embedded in the manuscript.

Remaining research limitations are documented in `docs/CHANGELOG_TMC_REVISION.md` and the manuscript limitations section. The artifact does not claim strong sample-level DP, measured phone energy, or universal superiority over homogeneous full-model FL.
