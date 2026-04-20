#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/dr-chandrasen-pandey/anaconda3/envs/palmfl310/bin/python}
PLOT_PY=${PLOT_PY:-python}

COMMON=(
  --override system.device=cuda
  --override dataset.partition_seed=42
  --override model.arch_seed=42
  --override system.eval_initial=false
  --override system.eval_every=5
  --override system.eval_max_batches=5
  --override system.final_eval_max_batches=0
  --override system.num_threads=4
  --override logging.save_checkpoints=false
)

run_palm() {
  local cfg="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  echo "===== PALM ${name} seed=${seed} $(date '+%F %T') ====="
  "$PY" -u -m palmfl.main --config "$cfg" \
    --override "experiment_name=${name}_seed${seed}_fixedsplit_cuda" \
    --override "seed=${seed}" \
    "${COMMON[@]}" "$@"
}

run_fedavg() {
  local cfg="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  local arch="$1"; shift
  echo "===== FEDAVG ${name} seed=${seed} arch=${arch} $(date '+%F %T') ====="
  "$PY" -u -m palmfl.fedavg_main --config "$cfg" \
    --override "experiment_name=${name}_${arch}_seed${seed}_fixedsplit_cuda" \
    --override "seed=${seed}" \
    --override "baseline.architecture=${arch}" \
    "${COMMON[@]}" "$@"
}

# Missing CIFAR transfer fixed-split rows for the current paper.
for seed in 43 44; do
  run_palm configs/palmfl_cifar10_stats_transfer_mobile.yaml palmfl_v8_cifar10_stats_transfer_mobile_sigma05 "$seed" --override dp.noise_multiplier=0.5
  run_palm configs/palmfl_cifar10_stats_transfer_random.yaml palmfl_v8_cifar10_stats_transfer_random_sigma05 "$seed" --override dp.noise_multiplier=0.5
done

# Homogeneous FedAvg reference baseline.
for seed in 42 43 44; do
  run_fedavg configs/palmfl_mnist_local_only.yaml fedavg_mnist "$seed" small_cnn
  run_fedavg configs/palmfl_cifar10_local_only.yaml fedavg_cifar10 "$seed" small_cnn
done

"$PY" scripts/aggregate_results.py
"$PLOT_PY" scripts/plot_acceptance_figures.py
