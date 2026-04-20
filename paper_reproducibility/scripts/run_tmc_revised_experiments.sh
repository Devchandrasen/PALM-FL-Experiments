#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/dr-chandrasen-pandey/anaconda3/envs/palmfl310/bin/python}
PLOT_PY=${PLOT_PY:-/home/dr-chandrasen-pandey/anaconda3/envs/palmfl310/bin/python}
PROFILE_CSV=${PROFILE_CSV:-data/mobilebandwidth/real_mobile_profiles.csv}

"$PY" scripts/build_real_mobile_profiles.py --num-clients 10

COMMON=(
  --override system.device=cuda
  --override scheduler.profile_csv="${PROFILE_CSV}"
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
    --override "experiment_name=${name}_seed${seed}_indsplit_trace" \
    --override "seed=${seed}" \
    --override "dataset.partition_seed=${seed}" \
    --override "model.arch_seed=${seed}" \
    "${COMMON[@]}" "$@"
}

run_fedavg() {
  local cfg="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  local arch="$1"; shift
  echo "===== FEDAVG ${name} seed=${seed} arch=${arch} $(date '+%F %T') ====="
  "$PY" -u -m palmfl.fedavg_main --config "$cfg" \
    --override "experiment_name=${name}_${arch}_seed${seed}_indsplit_trace" \
    --override "seed=${seed}" \
    --override "dataset.partition_seed=${seed}" \
    --override "model.arch_seed=${seed}" \
    --override "baseline.architecture=${arch}" \
    "${COMMON[@]}" "$@"
}

run_fedmd() {
  local cfg="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  local proxy_size="$1"; shift
  echo "===== FEDMD ${name} seed=${seed} proxy=${proxy_size} $(date '+%F %T') ====="
  "$PY" -u -m palmfl.fedmd_main --config "$cfg" \
    --override "experiment_name=${name}_seed${seed}_indsplit_trace" \
    --override "seed=${seed}" \
    --override "dataset.partition_seed=${seed}" \
    --override "model.arch_seed=${seed}" \
    --override "dataset.public_proxy_seed=${seed}" \
    --override "dataset.public_proxy_size=${proxy_size}" \
    --override baseline.distill_epochs=1 \
    --override baseline.distill_temperature=2.0 \
    "${COMMON[@]}" "$@"
}

run_fedproto() {
  local cfg="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  echo "===== FEDPROTO-STYLE ${name} seed=${seed} $(date '+%F %T') ====="
  run_palm "$cfg" "$name" "$seed" \
    --override training.enable_stat_head=false \
    --override training.stat_head_steps=0 \
    --override training.stat_head_min_steps=0 \
    --override training.stat_head_ramp_rounds=0 \
    --override training.prototype_ce_loss_weight=0.0 \
    --override training.latent_mixup_alpha=0.0 \
    --override dp.noise_multiplier=0.5 "$@"
}

for seed in 42 43 44; do
  run_palm configs/palmfl_mnist_local_only.yaml palmfl_v8_mnist_local_only "$seed"
  run_palm configs/palmfl_mnist_stats_upload_only.yaml palmfl_v8_mnist_stats_upload_only "$seed" --override dp.noise_multiplier=0.5
  run_palm configs/palmfl_mnist_stats_transfer_mobile.yaml palmfl_v8_mnist_stats_transfer_mobile_sigma05 "$seed" --override dp.noise_multiplier=0.5
  run_palm configs/palmfl_mnist_stats_transfer_random.yaml palmfl_v8_mnist_stats_transfer_random_sigma05 "$seed" --override dp.noise_multiplier=0.5
  run_fedavg configs/palmfl_mnist_local_only.yaml fedavg_mnist "$seed" small_cnn
  run_fedproto configs/palmfl_mnist_stats_transfer_mobile.yaml fedproto_mnist_mobile_sigma05 "$seed"
  run_fedmd configs/palmfl_mnist_local_only.yaml fedmd_mnist_proxy "$seed" 512
done

for seed in 42 43 44; do
  run_palm configs/palmfl_cifar10_local_only.yaml palmfl_v8_cifar10_local_only "$seed"
  run_palm configs/palmfl_cifar10_stats_upload_only.yaml palmfl_v8_cifar10_stats_upload_only "$seed" --override dp.noise_multiplier=0.5
  run_palm configs/palmfl_cifar10_stats_transfer_mobile.yaml palmfl_v8_cifar10_stats_transfer_mobile_sigma05 "$seed" --override dp.noise_multiplier=0.5
  run_palm configs/palmfl_cifar10_stats_transfer_random.yaml palmfl_v8_cifar10_stats_transfer_random_sigma05 "$seed" --override dp.noise_multiplier=0.5
  run_fedavg configs/palmfl_cifar10_local_only.yaml fedavg_cifar10 "$seed" small_cnn
  run_fedproto configs/palmfl_cifar10_stats_transfer_mobile.yaml fedproto_cifar10_mobile_sigma05 "$seed"
  run_fedmd configs/palmfl_cifar10_local_only.yaml fedmd_cifar10_proxy "$seed" 1024
done

"$PY" scripts/aggregate_results.py
"$PLOT_PY" scripts/plot_acceptance_figures.py --split-protocol independent
