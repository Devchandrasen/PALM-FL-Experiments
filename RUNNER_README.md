# PALM-FL v8 pivot prototype

This build is the **v8 pivot** after the matched v6/v7 ablations.

The generator branch is no longer the center of the method. Your latest results showed:

- on **MNIST**, gated transfer was only marginally above no-transfer
- on **CIFAR-10**, no-transfer remained better than gated transfer while also using **0 MB** download per round
- the current utility runs still had unresolved privacy accounting (`eps_max=inf` when DP was disabled / utility-only)

So this package pivots the codebase toward the stronger follow-up direction:

- **private local encoders + private latent adapters**
- **differentially private class-wise latent statistics**
- **global latent prototype / variance aggregation without a learned generator**
- **mobile-aware client scheduling** under upload / download / energy / time budgets
- clean baselines for
  - **local_only**
  - **stats_upload_only**
  - **stats_transfer**

## What changed from v7

The v7 path used a clustered latent mixture "generator" plus gated synthetic transfer. In v8, the main path no longer trains or uses a separate latent generator. Instead, the server only aggregates **DP class-wise latent means / variances / counts** and can send back a compact latent-stats package.

Clients can use that package in two lightweight ways:

1. **Prototype alignment on real batches**
2. **Stat-head transfer**: cheap head-only updates from latent samples drawn from aggregated class Gaussians
3. **Prototype-contrastive alignment**: real-batch latents are trained to score their own global class prototype above other available class prototypes

This makes the main transfer path much closer to the paper direction we converged on after the ablations.

## Latest research upgrade

This copy adds a compact **masked prototype-contrastive transfer** path:

- DP stats payloads now carry a configurable `dp.count_threshold` mask.
- The server ignores class statistics whose payload mask is false, avoiding prototype pollution from absent/noisy classes.
- Transfer configs add `training.prototype_ce_loss_weight` and `training.prototype_ce_temperature`.
- The real-batch objective now combines local cross-entropy, own-prototype alignment, prototype-contrastive classification over available global prototypes, and covariance regularization.
- Logs include `mean_real_proto_ce_loss` for ablation tracking.

This is still a lightweight stats-only method: no client model weights are uploaded and no neural generator is trained.

## Main experiment variants

### 1) local_only
No uploads. No downloads. No shared package. This is the clean communication-free baseline.

### 2) stats_upload_only
Clients upload DP latent statistics, but they do not consume any shared package. This isolates the upload/privacy overhead.

### 3) stats_transfer
Clients upload DP latent statistics and consume a compact global latent-stats package. This is the main v8 pivot method.

## Scheduler policies

Set `scheduler.policy` to:

- `mobile` for the mobile-aware selector
- `random` for a scheduler ablation
- `all` to train every client each round

## Important files

```text
palmfl_v8_pivot/
├── README.md
├── requirements.txt
├── run_smoke.sh
├── run_mnist_local_only.sh
├── run_mnist_stats_upload_only.sh
├── run_mnist_stats_transfer_mobile.sh
├── run_mnist_stats_transfer_random.sh
├── run_cifar10_local_only.sh
├── run_cifar10_stats_upload_only.sh
├── run_cifar10_stats_transfer_mobile.sh
├── run_cifar10_stats_transfer_random.sh
├── configs/
│   ├── palmfl_fake_smoke.yaml
│   ├── palmfl_mnist_local_only.yaml
│   ├── palmfl_mnist_stats_upload_only.yaml
│   ├── palmfl_mnist_stats_transfer_mobile.yaml
│   ├── palmfl_mnist_stats_transfer_random.yaml
│   ├── palmfl_cifar10_local_only.yaml
│   ├── palmfl_cifar10_stats_upload_only.yaml
│   ├── palmfl_cifar10_stats_transfer_mobile.yaml
│   └── palmfl_cifar10_stats_transfer_random.yaml
└── palmfl/
    ├── client.py
    ├── data.py
    ├── dp.py
    ├── latent_stats.py
    ├── main.py
    ├── metrics.py
    ├── models.py
    ├── scheduler.py
    ├── server.py
    └── utils.py
```

## Install

```bash
cd palmfl_v8_pivot
pip install -r requirements.txt
```

## Smoke test

```bash
bash run_smoke.sh
```

## Recommended first runs

MNIST:

```bash
bash run_mnist_local_only.sh
bash run_mnist_stats_upload_only.sh
bash run_mnist_stats_transfer_mobile.sh
bash run_mnist_stats_transfer_random.sh
```

CIFAR-10:

```bash
bash run_cifar10_local_only.sh
bash run_cifar10_stats_upload_only.sh
bash run_cifar10_stats_transfer_mobile.sh
bash run_cifar10_stats_transfer_random.sh
```

## How to read the results

For **local_only**, you should see:

- `stats_upload_enabled=False`
- `stats_transfer_enabled=False`
- `round_upload_mb=0.000`
- `round_download_mb=0.000`

For **stats_upload_only**, you should see:

- `stats_upload_enabled=True`
- `stats_transfer_enabled=False`
- `round_upload_mb>0`
- `round_download_mb=0.000`

For **stats_transfer**, you should see after warm-up:

- `stats_upload_enabled=True`
- `stats_transfer_enabled=True`
- `prototype_transfer_enabled=True` if enabled in config
- `stat_head_transfer_enabled=True` if enabled in config
- small but nonzero download per round

## Current intent of v8

This package is meant to answer the next paper question after the generator ablations:

> Is the stronger TMC follow-up really the private latent adapter + compact stats transfer + mobile scheduler path?

That is what this build is designed to test.
