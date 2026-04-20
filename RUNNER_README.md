# PALM-FL Experiment Runner Notes

This implementation evaluates a low-payload heterogeneous federated learning workflow based on local client encoders, shared latent adapter dimensions, clipped-and-noised class-wise summaries, optional masked stats-transfer, and mobile-aware scheduling.

## Main Variants

### local-only

No uploads, no downloads, and no shared server package.

### stats-upload-only

Clients upload clipped-and-noised latent statistics. The server updates summary memory and schedules future clients. Clients do not consume a transfer package.

### stats-transfer

Clients upload clipped-and-noised latent statistics and consume a compact masked transfer package derived from server memory.

## Scheduler Policies

Set `scheduler.policy` to:

- `mobile` for the mobile-aware selector,
- `random` for a participation ablation,
- `all` to train every client each round.

## Important Paths

```text
palmfl/       Python implementation
configs/      YAML experiment configs
scripts/      run, aggregate, curate, and plot utilities
analysis/     committed CSV summaries
outputs/      regenerated raw run artifacts, ignored by git
```

## Install

```bash
pip install -r requirements.txt
```

## Smoke Test

```bash
python -m palmfl.main --config configs/palmfl_fake_smoke.yaml
```

## Recommended Runs

```bash
PY=python3 DEVICE=cuda bash scripts/run_revised_experiments.sh
SEEDS="42 43 44" DATASETS="mnist cifar10" DEVICE=cuda PY=python3 bash scripts/run_ablation_experiments.sh
```

## Reading Results

For local-only runs:

- `stats_upload_enabled=False`
- `stats_transfer_enabled=False`
- `round_upload_mb=0.000`
- `round_download_mb=0.000`

For stats-upload-only runs:

- `stats_upload_enabled=True`
- `stats_transfer_enabled=False`
- `round_upload_mb>0`
- `round_download_mb=0.000`

For stats-transfer runs:

- `stats_upload_enabled=True`
- `stats_transfer_enabled=True`
- `prototype_transfer_enabled=True` when enabled in the config,
- `stat_head_transfer_enabled=True` when enabled in the config,
- small nonzero download after the warm-up phase.
