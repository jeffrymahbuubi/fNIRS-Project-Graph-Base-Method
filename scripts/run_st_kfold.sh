#!/bin/bash
# Run ST pipeline (WindowedSpatioTemporalGATNet) with 5-fold and 10-fold cross-validation.
#
# Applies Optuna HBT search findings (200 trials, 2026-04-29, Trial #137 best):
#   - n_layers=3, n_filters=96, n_heads=6, fc_size=256, dropout=0.4
#   - use_residual=false, use_norm=true (batch), window_size=48, window_stride=16
#   - temporal_hidden=128, temporal_layers=3
#   - lr=1.375e-4  (Optuna best-trial LR; lower than core default 1e-3)
#   - All hyperparams come from src/core_st/experiment_config.yaml
#
# NOTE: Optuna was tuned on HBT only. HBO/HBR runs use the same config as a
# reasonable starting point but have not been independently tuned.
#
# Sweeps: k_folds (5, 10) × signal_type (hbo, hbr, hbt) = 6 runs total.
# Dataset: processed-new (GNG task; dataset class appends task_type internally).
#
# Results land at:
#   research/experiments/<DATE>/st-kfold/<K>-fold/<DATE>/<exp_name>/
#
# Usage:
#   bash scripts/run_st_kfold.sh               # all signal types, 5-fold and 10-fold
#   bash scripts/run_st_kfold.sh hbt           # hbt only
#   bash scripts/run_st_kfold.sh hbo hbr       # two signal types
#   bash scripts/run_st_kfold.sh --checkpoint_metric=loss    # checkpoint on lowest val loss
#   bash scripts/run_st_kfold.sh hbo --checkpoint_metric=loss
#   bash scripts/run_st_kfold.sh --scheduler=cosine_warmup   # override YAML scheduler
#   bash scripts/run_st_kfold.sh --scheduler=reduce_on_plateau hbo
#   bash scripts/run_st_kfold.sh --max_trials=4              # override YAML max_trials
#   bash scripts/run_st_kfold.sh --max_trials=4 hbo --scheduler=cosine_annealing

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_ROOT/src/core_st/experiment_config.yaml"
DATE_TAG="$(date +%Y%m%d)"
BASE_SAVE="$REPO_ROOT/research/experiments/$DATE_TAG/st-kfold"
DATA_DIR="$REPO_ROOT/data/processed-new-mc"
SPLITS_JSON="$REPO_ROOT/data/splits/kfold_splits_processed_new_mc.json"

# Optuna best-trial fixed settings
LR=3.04e-4
EPOCHS=150
BATCH_SIZE=8
PATIENCE=30
SEED=42
NUM_WORKERS=4

CHECKPOINT_METRIC="f1"
SCHEDULER=""
MAX_TRIALS=""
DATA_TYPES=()
for arg in "$@"; do
    case "$arg" in
        --checkpoint_metric=*) CHECKPOINT_METRIC="${arg#--checkpoint_metric=}" ;;
        --scheduler=*) SCHEDULER="${arg#--scheduler=}" ;;
        --max_trials=*) MAX_TRIALS="${arg#--max_trials=}" ;;
        hbo|hbr|hbt) DATA_TYPES+=("$arg") ;;
        --*) echo "WARNING: unrecognised argument '$arg' — ignored. Did you mean --checkpoint_metric=, --scheduler=, or --max_trials=?" >&2 ;;
    esac
done
if [ "${#DATA_TYPES[@]}" -eq 0 ]; then
    DATA_TYPES=(hbo hbr hbt)
fi

EXTRA_ARGS=()
[ -n "$SCHEDULER" ] && EXTRA_ARGS+=(--scheduler "$SCHEDULER")
[ -n "$MAX_TRIALS" ] && EXTRA_ARGS+=(--max_trials "$MAX_TRIALS")
SCHED_LABEL="${SCHEDULER:-yaml-default}"
MT_LABEL="${MAX_TRIALS:-yaml-default}"

cd "$REPO_ROOT"

for K_FOLDS in 5 10; do
    for DATA_TYPE in "${DATA_TYPES[@]}"; do

        echo ""
        echo "=========================================="
        echo "k_folds: $K_FOLDS | Signal: $DATA_TYPE | epochs: $EPOCHS | lr: $LR | scheduler: $SCHED_LABEL | max_trials: $MT_LABEL"
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

echo ""
echo "All experiments complete. Results in: $BASE_SAVE"
