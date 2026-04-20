#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
PROFILE_CSV=${PROFILE_CSV:-data/mobilebandwidth/real_mobile_profiles.csv}
SEEDS=${SEEDS:-"42 43 44"}
DATASETS=${DATASETS:-"mnist cifar10"}
DEVICE=${DEVICE:-cpu}

"$PY" scripts/build_real_mobile_profiles.py --num-clients 10

COMMON=(
  --override system.device="${DEVICE}"
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
  echo "===== PALM ablation ${name} seed=${seed} $(date '+%F %T') ====="
  "$PY" -u -m palmfl.main --config "$cfg" \
    --override "experiment_name=${name}_seed${seed}_indsplit_trace" \
    --override "seed=${seed}" \
    --override "dataset.partition_seed=${seed}" \
    --override "model.arch_seed=${seed}" \
    "${COMMON[@]}" "$@"
}

for dataset in ${DATASETS}; do
  stats_cfg="configs/palmfl_${dataset}_stats_upload_only.yaml"
  transfer_mobile_cfg="configs/palmfl_${dataset}_stats_transfer_mobile.yaml"
  transfer_random_cfg="configs/palmfl_${dataset}_stats_transfer_random.yaml"

  for seed in ${SEEDS}; do
    run_palm "$stats_cfg" "palmfl_v8_${dataset}_histogram_only_mobile_sigma05" "$seed" \
      --override algorithm.summary_mode=histogram_only \
      --override dp.noise_multiplier=0.5 \
      --override dp.releases_per_round=1 \
      --override scheduler.utility_weights.deficit=1.0

    run_palm "$stats_cfg" "palmfl_v8_${dataset}_histogram_only_random_sigma05" "$seed" \
      --override algorithm.summary_mode=histogram_only \
      --override scheduler.policy=random \
      --override dp.noise_multiplier=0.5 \
      --override dp.releases_per_round=1

    run_palm "$stats_cfg" "palmfl_v8_${dataset}_mean_only_mobile_sigma05" "$seed" \
      --override algorithm.summary_mode=mean_only \
      --override dp.noise_multiplier=0.5 \
      --override dp.count_clip=1.0 \
      --override dp.count_threshold=0.5 \
      --override dp.releases_per_round=2

    run_palm "$stats_cfg" "palmfl_v8_${dataset}_mean_only_random_sigma05" "$seed" \
      --override algorithm.summary_mode=mean_only \
      --override scheduler.policy=random \
      --override dp.noise_multiplier=0.5 \
      --override dp.count_clip=1.0 \
      --override dp.count_threshold=0.5 \
      --override dp.releases_per_round=2

    run_palm "$stats_cfg" "palmfl_v8_${dataset}_stats_upload_only_no_noise" "$seed" \
      --override dp.enable=false \
      --override dp.noise_multiplier=0.0

    run_palm "$transfer_mobile_cfg" "palmfl_v8_${dataset}_stats_transfer_mobile_no_noise" "$seed" \
      --override dp.enable=false \
      --override dp.noise_multiplier=0.0

    run_palm "$transfer_mobile_cfg" "palmfl_v8_${dataset}_count_only_transfer_mobile_sigma05" "$seed" \
      --override algorithm.variant=count_only_transfer \
      --override algorithm.summary_mode=histogram_only \
      --override algorithm.transfer_package_mode=counts_only \
      --override dp.noise_multiplier=0.5 \
      --override dp.releases_per_round=1

    run_palm "$stats_cfg" "palmfl_v8_${dataset}_stats_upload_only_no_deficit_sigma05" "$seed" \
      --override scheduler.ablation=no_deficit \
      --override scheduler.utility_weights.deficit=0.0 \
      --override dp.noise_multiplier=0.5

    run_palm "$transfer_mobile_cfg" "palmfl_v8_${dataset}_stats_transfer_mobile_no_deficit_sigma05" "$seed" \
      --override scheduler.ablation=no_deficit \
      --override scheduler.utility_weights.deficit=0.0 \
      --override dp.noise_multiplier=0.5

    run_palm "$transfer_random_cfg" "palmfl_v8_${dataset}_stats_transfer_random_no_deficit_sigma05" "$seed" \
      --override scheduler.ablation=no_deficit \
      --override scheduler.utility_weights.deficit=0.0 \
      --override dp.noise_multiplier=0.5
  done
done

"$PY" scripts/aggregate_results.py
"$PY" scripts/curate_tmc_results.py
