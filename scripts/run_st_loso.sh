#!/bin/bash
# Run ST pipeline (WindowedSpatioTemporalGATNet) with Leave-One-Subject-Out (LOSO) validation.
#
# Applies Optuna HBT search findings (200 trials, 2026-04-29, Trial #137 best):
#   - n_layers=3, n_filters=96, n_heads=6, fc_size=256, dropout=0.4
#   - use_residual=false, use_norm=true (batch), window_size=48, window_stride=16
#   - temporal_hidden=128, temporal_layers=3
#   - lr=1.375e-4  (Optuna best-trial LR)
#   - All hyperparams come from src/core_st/experiment_config.yaml
#
# NOTE: Optuna was tuned on HBT only. HBO/HBR runs use the same config as a
# reasonable starting point but have not been independently tuned.
#
# Sweeps: signal_type (hbo, hbr, hbt) = 3 runs total.
# Dataset: processed-new (GNG task; dataset class appends task_type internally).
#
# Results land at:
#   research/experiments/<DATE>/st-loso/<DATE>/<exp_name>/
#
# Usage:
#   bash scripts/run_st_loso.sh               # all signal types
#   bash scripts/run_st_loso.sh hbt           # hbt only
#   bash scripts/run_st_loso.sh hbo hbr       # two signal types
#   bash scripts/run_st_loso.sh --checkpoint_metric=loss     # checkpoint on lowest val loss
#   bash scripts/run_st_loso.sh hbo --checkpoint_metric=loss

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_ROOT/src/core_st/experiment_config.yaml"
DATE_TAG="$(date +%Y%m%d)"
BASE_SAVE="$REPO_ROOT/research/experiments/$DATE_TAG/st-loso"
DATA_DIR="$REPO_ROOT/data/processed-new-mc"

# Optuna best-trial fixed settings
LR=0.000635
EPOCHS=150
BATCH_SIZE=8
PATIENCE=30
SEED=42
NUM_WORKERS=4

CHECKPOINT_METRIC="f1"
DATA_TYPES=()
for arg in "$@"; do
    case "$arg" in
        --checkpoint_metric=*) CHECKPOINT_METRIC="${arg#--checkpoint_metric=}" ;;
        hbo|hbr|hbt) DATA_TYPES+=("$arg") ;;
        --*) echo "WARNING: unrecognised argument '$arg' — ignored. Did you mean --checkpoint_metric=?" >&2 ;;
    esac
done
if [ "${#DATA_TYPES[@]}" -eq 0 ]; then
    DATA_TYPES=(hbo hbr hbt)
fi

cd "$REPO_ROOT"

for DATA_TYPE in "${DATA_TYPES[@]}"; do

    echo ""
    echo "=========================================="
    echo "LOSO | Signal: $DATA_TYPE | epochs: $EPOCHS | lr: $LR"
    echo "=========================================="
    python -m src.core_st.main \
        --config "$CONFIG" \
        --data_dir "$DATA_DIR" \
        --save_dir "$BASE_SAVE" \
        --task GNG \
        --data_type "$DATA_TYPE" \
        --validation loso \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE" \
        --patience "$PATIENCE" \
        --checkpoint_metric "$CHECKPOINT_METRIC" \
        --seed "$SEED" \
        --num_workers "$NUM_WORKERS"

done

echo ""
echo "All experiments complete. Results in: $BASE_SAVE"
