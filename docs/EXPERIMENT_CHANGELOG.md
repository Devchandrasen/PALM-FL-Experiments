# Experiment Repository Changelog

## Experiment-Only Repository Cleanup

- Removed non-experiment source bundles, generated document assets, and document-generation instructions.
- Removed duplicate reproduction bundle.
- Renamed run scripts to experiment-oriented names.
- Renamed curated trace CSVs to experiment-focused names.
- Rewrote repository documentation around code, configs, runs, and analysis CSVs.
- Updated consistency checks so they validate the experiment package.

## Implemented Experiment Features

- Local-only, stats-only, and masked stats-transfer modes.
- Histogram-only, mean-only, no-noise, count-only-transfer, and no-deficit ablations.
- Resource-aware and random scheduling.
- Summary-level clipping/noising and participation-counted accounting.
- FedAvg and FedMD-style reference entry points.
- Trace-derived bandwidth profile support.

## Remaining Experiment Work

- Complete larger CIFAR-10 multi-seed main sweeps where needed.
- Add real-device calibration if deployment-level energy claims are required.
- Add stronger heterogeneous FL baselines if direct comparison is required.
- Add leakage diagnostics if privacy attack resistance needs to be evaluated.
