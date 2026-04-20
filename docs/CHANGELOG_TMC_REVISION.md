# TMC Revision Changelog

## Completed Manuscript Changes

- Reframed the paper as a low-payload client-summary coordination framework rather than a strong accuracy replacement for full-model FL.
- Changed the title to `PALM-FL: Low-Payload Client-Summary Coordination and Resource-Aware Scheduling for Heterogeneous Mobile Federated Learning`.
- Rewrote abstract, introduction, discussion, limitations, and conclusion to avoid unsupported claims about strong privacy, strong FL gain, measured energy, or FedAvg outperformance.
- Replaced confusing client-set notation with `\mathcal{K}=\{1,\ldots,K_{\mathrm{cl}}\}` while reserving `C` for number of classes.
- Clarified that the server is architecture-agnostic but still assumes a shared latent dimension through private adapters.
- Clarified that stats-only uses the memory for scheduling rather than model aggregation and that the current stats-only path only partially isolates latent means from count-driven scheduling.
- Replaced the missing-ablation table with a generated ablation table populated from completed MNIST seeds 42, 43, and 44 artifacts and explicit CIFAR-10 incomplete rows.
- Added privacy-accounting wording and a privacy-parameter table. Non-DP-accounted rows are marked with `--`, not `0.00`.
- Clarified that time and energy are trace-calibrated operating estimates, not phone power-meter measurements.

## Completed Figure/Table Changes

- Regenerated the communication/privacy figure with grouped MNIST/CIFAR-10 labels and a clear `\varepsilon_{\max}` privacy axis.
- Regenerated the accuracy-energy frontier as separate MNIST and CIFAR-10 panels with FedAvg marked as a homogeneous full-model reference.
- Generated CSV and LaTeX table files under `generated_tables/` from current curated artifacts.
- Added a consistency checker to catch unresolved references, placeholder text, misleading epsilon zeros, and missing generated tables.

## Completed Documentation and Scripts

- Added `TMC_READINESS_AUDIT.md`.
- Added `README_REPRODUCIBILITY.md`.
- Added `scripts/run_all_mnist.sh`.
- Added `scripts/run_all_cifar.sh`.
- Added `scripts/make_tables.py`.
- Added `scripts/make_figures.py`.
- Added `scripts/check_results_consistency.py`.
- Added and packaged `reproducibility/scripts/run_tmc_ablation_experiments.sh`.
- Updated the packaged PALM code snapshot with the current runner modules, including `client.py`, `dp.py`, `latent_stats.py`, `metrics.py`, `models.py`, and `utils.py`.

## Completed Experiments

- MNIST independent trace-derived campaign for local-only, stats-only, stats-transfer mobile, stats-transfer random, FedAvg, FedProto-style, and FedMD-style references over seeds 42, 43, and 44.
- CIFAR-10 independent trace-derived seed-42 case-study rows for local-only, stats-only, stats-transfer mobile, stats-transfer random, and FedAvg.
- Trace-derived bandwidth-profile construction from public mobile bandwidth measurements.
- MNIST histogram-only ablation with mobile and random scheduling over seeds 42, 43, and 44.
- MNIST latent-mean-only ablation with mobile and random scheduling over seeds 42, 43, and 44.
- MNIST no-noise stats-only and transfer diagnostics over seeds 42, 43, and 44, marked as non-private diagnostics.
- MNIST count-only transfer ablation over seeds 42, 43, and 44.
- MNIST no-deficit scheduler ablations for stats-only, mobile transfer, and random transfer over seeds 42, 43, and 44.
- CIFAR-10 histogram-only ablation with mobile and random scheduling over seeds 42, 43, and 44.
- CIFAR-10 latent-mean-only ablation with mobile and random scheduling over seeds 42, 43, and 44.
- CIFAR-10 no-noise stats-only and transfer diagnostics over seeds 42, 43, and 44, marked as non-private diagnostics.
- CIFAR-10 count-only transfer ablation over seeds 42, 43, and 44.
- CIFAR-10 no-deficit scheduler ablations for stats-only, mobile transfer, and random transfer over seeds 42, 43, and 44.

## Failed or Incomplete Experiments

- Multi-seed independent CIFAR-10 main operating-mode/baseline campaign: not completed in curated artifacts.
- Real-device time/energy calibration: not completed.
- Cost-model sensitivity study: not completed.
- HeteroFL/FjORD/FedDF/FedGH/GEFL empirical baselines: not implemented in current artifacts.

## Changed Claims

- The core claim is now that PALM-FL is an architecture-flexible, low-payload coordination mechanism.
- The stats-only claim is now that it preserves local-training utility with zero downlink and small uplink, not that it provides a strong FL improvement.
- The transfer claim is now scheduler-sensitive and framed as suggestive under broad participation coverage.
- The privacy claim is limited to clipped client-summary release with participation-counted zCDP-style accounting.
- The mobile-systems claim is limited to trace-derived bandwidth plus modeled compute/energy estimates.

## Removed Claims

- Removed or avoided “strong differential privacy” language.
- Removed or avoided any implication that FedAvg is outperformed.
- Removed or avoided any claim of measured phone energy.
- Removed or avoided any claim that latent means are proven to be the source of gains; the new MNIST and CIFAR-10 ablations show that count-only and random-participation controls remain competitive.
- Removed placeholder artifact URL language.

## New Limitations

- The value of latent means remains only partially isolated because the new ablations show competitive count-only and random-participation controls.
- CIFAR-10 main operating-mode and baseline comparisons remain a trace-calibrated case study until independent multi-seed main runs finish.
- Current privacy values are high and are not sample-level DP certificates.
- Current energy/time values are not measured on phones or edge devices.
- The package-level code snapshot now contains the current Python modules, but a formal review artifact still needs environment locking and complete external packaging.

## Remaining TMC Review Risks

- Reviewers may reject because the new ablations make the latent-moment contribution more modest than the original framing.
- Reviewers may reject for single-seed CIFAR-10 main operating-mode and baseline evidence.
- Reviewers may reject for lack of real-device calibration.
- Reviewers may reject for missing stronger heterogeneous-FL baselines.
- Reviewers may view high epsilon values as weak privacy despite improved wording.
