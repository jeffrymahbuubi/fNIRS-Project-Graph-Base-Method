#!/bin/bash
# Run CORAL-validated GNN experiments with 5-fold and 10-fold cross-validation.
#
# Applies CORAL optimization findings (106 evals, 2026-04-29):
#   - Signed correlation edges (already in dataset.py — no change needed here)
#   - epochs=150  (confirmed +0.036 holdout F1 vs epochs=100)
#   - max_trials=2  (CORAL winning config for GNG/hbo; applied to hbr/hbt as best guess)
#   - No augmentation, Adam lr=1e-3, batch_size=8, patience=9999 (locked by CORAL)
#   - All other hyperparams come from src/core/experiment_config.yaml
#
# Sweeps: k_folds (5, 10) × signal_type (hbo, hbr, hbt) = 6 runs total.
# Dataset: processed-new only (CORAL was validated on this dataset).
#
# Results land at:
#   research/experiments/<DATE>/coral-kfold/<K>-fold/with-additional-data/<DATE>/<exp_name>/
#
# Usage:
#   bash scripts/run_coral_kfold.sh               # all signal types, both fold configs
#   bash scripts/run_coral_kfold.sh hbo           # hbo only
#   bash scripts/run_coral_kfold.sh hbo hbr       # two signal types

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_ROOT/src/core/experiment_config.yaml"
DATE_TAG="$(date +%Y%m%d)"
BASE_SAVE="$REPO_ROOT/research/experiments/$DATE_TAG/coral-kfold"
DATA_NEW="$REPO_ROOT/data/processed-new"
SPLITS_NEW="$REPO_ROOT/data/splits/kfold_splits_processed_new.json"

# CORAL-locked hyperparameters
MAX_TRIALS=2
EPOCHS=150
LR=1e-3
BATCH_SIZE=8
PATIENCE=9999
SEED=42

CHECKPOINT_METRIC="f1"
DATA_TYPES=()
for arg in "$@"; do
    case "$arg" in
        --checkpoint_metric=*) CHECKPOINT_METRIC="${arg#--checkpoint_metric=}" ;;
        hbo|hbr|hbt) DATA_TYPES+=("$arg") ;;
    esac
done
if [ "${#DATA_TYPES[@]}" -eq 0 ]; then
    DATA_TYPES=(hbo hbr hbt)
fi

cd "$REPO_ROOT"

for K_FOLDS in 5 10; do
    for DATA_TYPE in "${DATA_TYPES[@]}"; do

        echo ""
        echo "=========================================="
        echo "k_folds: $K_FOLDS | Signal: $DATA_TYPE | max_trials: $MAX_TRIALS | epochs: $EPOCHS"
        echo "=========================================="
        python -m src.core.main \
            --config "$CONFIG" \
            --data_dir "$DATA_NEW" \
            --save_dir "$BASE_SAVE/${K_FOLDS}-fold/with-additional-data" \
            --splits_json "$SPLITS_NEW" \
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
            --seed "$SEED"

    done
done

echo ""
echo "All experiments complete. Results in: $BASE_SAVE"
