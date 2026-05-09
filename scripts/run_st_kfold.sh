#!/bin/bash
# Run ST pipeline (WindowedSpatioTemporalGATNet) with 5-fold and 10-fold cross-validation.
#
# Applies May-3 Optuna lr_cosine search best (study st_hbo_mt4_ep100_tr25_kf5_lr_cosine,
# Trial #36, 95 complete + 5 pruned; 5-fold mean F1 = 0.7693 on HBO mt4):
#   - lr=3.04e-4
#   - scheduler=cosine_annealing, T_max=150 (= --epochs), eta_min=1e-5
#
# Architecture (held fixed during the search; identical to current
# src/core_st/experiment_config.yaml):
#   - GATv2: n_layers=2, n_filters=80, n_heads=2, fc_size=256, dropout=0.3
#   - use_residual=false, use_norm=true (batch)
#   - Temporal: window_size=16, window_stride=8, temporal_hidden=192, temporal_layers=1
#
# Supersedes the Apr-29 Trial #137 search (n_layers=3, ws=48, lr=1.375e-4) and
# any earlier ST tunings. Search type was lr_cosine (lr + scheduler params only),
# so the architecture is whatever the YAML pinned on 2026-05-03 (== current YAML).
#
# Sweeps: max_trials (2, 4) × k_folds (5, 10) × signal_type (hbo, hbr, hbt) = 12 runs total.
# Dataset: processed-new-mc (GNG task; dataset class appends task_type internally).
#
# Results land at:
#   research/experiments/<DATE>/st-kfold/<K>-fold/<DATE>/<exp_name>/
#
# Usage:
#   bash scripts/run_st_kfold.sh                                # full sweep: mt={2,4} × k={5,10} × signals={hbo,hbr,hbt}
#   bash scripts/run_st_kfold.sh hbt                            # hbt only (still mt={2,4} × k={5,10})
#   bash scripts/run_st_kfold.sh hbo hbr                        # two signal types
#   bash scripts/run_st_kfold.sh --max_trials=2                 # only mt2 (single value)
#   bash scripts/run_st_kfold.sh --max_trials=2,4 hbt           # explicit mt sweep on hbt
#   bash scripts/run_st_kfold.sh --checkpoint_metric=loss       # checkpoint on lowest val loss
#   bash scripts/run_st_kfold.sh --scheduler=cosine_warmup      # override default cosine_annealing
#   bash scripts/run_st_kfold.sh --eta_min=0.0                  # override default eta_min=1e-5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_ROOT/src/core_st/experiment_config.yaml"
DATE_TAG="$(date +%Y%m%d)"
BASE_SAVE="$REPO_ROOT/research/experiments/$DATE_TAG/st-kfold"
DATA_DIR="$REPO_ROOT/data/processed-new-mc"
SPLITS_JSON="$REPO_ROOT/data/splits/kfold_splits_processed_new_mc.json"

# May-3 Optuna lr_cosine best (Trial #36)
LR=3.04e-4
EPOCHS=150              # also serves as T_max for CosineAnnealingLR
SCHEDULER=cosine_annealing
ETA_MIN=1e-5
BATCH_SIZE=8
PATIENCE=30
SEED=42
NUM_WORKERS=4

CHECKPOINT_METRIC="f1"
MAX_TRIALS_LIST=(2 4)        # default sweep — overridden by --max_trials=N or --max_trials=N,M
DATA_TYPES=()
for arg in "$@"; do
    case "$arg" in
        --checkpoint_metric=*) CHECKPOINT_METRIC="${arg#--checkpoint_metric=}" ;;
        --scheduler=*) SCHEDULER="${arg#--scheduler=}" ;;
        --eta_min=*) ETA_MIN="${arg#--eta_min=}" ;;
        --max_trials=*) IFS=',' read -ra MAX_TRIALS_LIST <<< "${arg#--max_trials=}" ;;
        hbo|hbr|hbt) DATA_TYPES+=("$arg") ;;
        --*) echo "WARNING: unrecognised argument '$arg' — ignored. Did you mean --checkpoint_metric=, --scheduler=, --eta_min=, or --max_trials=?" >&2 ;;
    esac
done
if [ "${#DATA_TYPES[@]}" -eq 0 ]; then
    DATA_TYPES=(hbo hbr hbt)
fi

EXTRA_ARGS=(--scheduler "$SCHEDULER" --eta_min "$ETA_MIN")

cd "$REPO_ROOT"

for MAX_TRIALS in "${MAX_TRIALS_LIST[@]}"; do
    for K_FOLDS in 5 10; do
        for DATA_TYPE in "${DATA_TYPES[@]}"; do

            echo ""
            echo "=========================================="
            echo "max_trials: $MAX_TRIALS | k_folds: $K_FOLDS | Signal: $DATA_TYPE | epochs: $EPOCHS | lr: $LR | scheduler: $SCHEDULER | eta_min: $ETA_MIN"
            echo "=========================================="
            python -m src.core_st.main \
                --config "$CONFIG" \
                --data_dir "$DATA_DIR" \
                --save_dir "$BASE_SAVE/${K_FOLDS}-fold" \
                --splits_json "$SPLITS_JSON" \
                --task GNG \
                --data_type "$DATA_TYPE" \
                --validation kfold \
                --k_folds "$K_FOLDS" \
                --max_trials "$MAX_TRIALS" \
                --epochs "$EPOCHS" \
                --lr "$LR" \
                --batch_size "$BATCH_SIZE" \
                --patience "$PATIENCE" \
                --checkpoint_metric "$CHECKPOINT_METRIC" \
                --seed "$SEED" \
                --num_workers "$NUM_WORKERS" \
                "${EXTRA_ARGS[@]}"

        done
    done
done

echo ""
echo "All experiments complete. Results in: $BASE_SAVE"
