# TMC Readiness Audit

Audit date: 2026-04-19

This audit covers the packaged manuscript directory at `/home/dr-chandrasen-pandey/PALM_review` and the full experiment runner referenced by the package at `/home/dr-chandrasen-pandey/Desktop/My Experiments/palmfl_v8_pivot/palmfl_v8_pivot`.

## Repository Inventory

| Item | Status | Location | Notes |
|---|---:|---|---|
| Main LaTeX file | Exists | `main.tex` | Current manuscript source. |
| Bibliography | Inline | `main.tex` | References are in `thebibliography`, not a separate `.bib` file. |
| Compiled PDF | Exists | `main.pdf` | Rebuilt from current source. |
| Figure assets | Exists | `figures/*.pdf` | Figures 1, 2, 5, 6, 7 are current; older unused figures 3/4 remain. |
| Curated result CSVs | Exists | `reproducibility/tmc_trace_*.csv` | Current manuscript tables use these curated trace-derived rows. |
| Aggregation scripts | Exists | `reproducibility/scripts/aggregate_results.py`, `curate_tmc_results.py` | Build `all_results`, grouped rows, and curated TMC rows from run artifacts. |
| Figure-generation script | Exists | `reproducibility/scripts/plot_acceptance_figures.py` | Regenerates learning, communication/privacy, frontier, and fairness figures. |
| Trace-profile builder | Exists | `reproducibility/scripts/build_real_mobile_profiles.py` | Builds deterministic bandwidth-profile CSV from public mobile bandwidth data. |
| Experiment launch script | Exists | `reproducibility/scripts/run_tmc_revised_experiments.sh`, `reproducibility/scripts/run_tmc_ablation_experiments.sh` | Main campaign and ablation campaign scripts are packaged. |
| Full PALM codebase | Exists outside package | `/home/dr-chandrasen-pandey/Desktop/My Experiments/palmfl_v8_pivot/palmfl_v8_pivot/palmfl` | Contains `client.py`, `dp.py`, `latent_stats.py`, `scheduler.py`, `server.py`, baselines, metrics, and models. |
| Packaged PALM code snapshot | Exists | `reproducibility/palmfl_code` | Updated with the current runner modules used for the MNIST three-seed ablation artifacts. |
| Privacy-accounting code | Exists | `reproducibility/palmfl_code/dp.py` and full runner | Accountant supports release-count differences for full, histogram-only, and mean-only summaries. |
| Scheduler code | Exists | Full runner and partial snapshot | Scheduler cost model is model-based except bandwidth profile. |
| Aggregation / latent-memory code | Exists | `reproducibility/palmfl_code/latent_stats.py`, `server.py` and full runner | Supports full summaries, histogram-only count memory, mean-only summaries, and counts-only package export. |
| Baseline code | Exists in full runner | `fedavg_main.py`, `fedmd_main.py`, FedProto-style via PALM config overrides | HeteroFL/FjORD/FedDF/FedGH/GEFL wrappers are not implemented. |
| Artifact documentation | Added | `README_REPRODUCIBILITY.md` | Describes setup, commands, expected outputs, and limitations. |
| Consistency checker | Added | `scripts/check_results_consistency.py` | Checks tables/PDF/source for common misleading states. |

## Required TMC Items

| Requirement | Status | Evidence / Gap |
|---|---:|---|
| Modest title/abstract/conclusion | Complete | Current manuscript frames PALM-FL as low-payload client-summary coordination. |
| No misleading epsilon zeros | Complete | Non-DP-accounted rows use `--` in manuscript tables. |
| Cleaned communication/privacy figure | Complete | `fig6_comm_privacy_bars.pdf` has grouped dataset labels and clear summary-epsilon axis. |
| Cleaned accuracy-energy frontier | Complete | `fig5_accuracy_energy_frontier.pdf` separates MNIST and CIFAR-10 panels and marks FedAvg as a homogeneous reference. |
| Privacy accounting table | Complete | Added manuscript table and generated CSV/LaTeX table. |
| Artifact/reproducibility package | Partial | Documentation, scripts, and current PALM modules are packaged; full external packaging still needs environment-lock and dataset/cache instructions for review upload. |
| Histogram-only ablation | Complete | Completed MNIST and CIFAR-10 mobile/random artifacts over seeds 42, 43, and 44. |
| Latent-means-only ablation | Complete | Completed MNIST and CIFAR-10 mobile/random artifacts over seeds 42, 43, and 44. |
| No-deficit scheduler ablation | Complete | Completed MNIST and CIFAR-10 stats-only, mobile-transfer, and random-transfer artifacts over seeds 42, 43, and 44. |
| No-noise diagnostic | Complete | Completed MNIST and CIFAR-10 stats-only and transfer diagnostics over seeds 42, 43, and 44; these are non-private diagnostics. |
| Count-only transfer ablation | Complete | Completed MNIST and CIFAR-10 mobile artifacts over seeds 42, 43, and 44. |
| Multi-seed CIFAR-10 main trace campaign | Partial | Seed-42 independent main operating-mode and FedAvg rows are complete; seeds 43/44 main rows are not complete in curated trace artifacts. |
| Stronger hetero-FL baselines | Partial | FedProto-style and FedMD-style exist; HeteroFL/FjORD/FedDF/FedGH/GEFL wrappers are not implemented. |
| Real-device calibration | Not completed | Current mobile results are trace-calibrated operating estimates only. |
| Cost-model sensitivity study | Not completed | No completed sensitivity artifacts found. |

## Current Readiness Judgment

The package is still a major-revision / promising-prototype state, not TMC-ready. The new MNIST and CIFAR-10 ablations over seeds 42, 43, and 44 improve isolation of counts, means, noise, transfer, and scheduler effects, but they also show that counts, participation coverage, and scheduler choice explain much of the observed behavior. The project still does not provide sample-level DP, measured device energy, or multi-seed CIFAR-10 main operating-mode/baseline stability.
