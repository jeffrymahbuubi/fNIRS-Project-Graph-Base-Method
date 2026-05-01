#!/bin/bash
# Run baseline GNN (src/core, GATv2 / GINE-GAT) with 5-fold and 10-fold cross-validation.
#
# Uses the best Optuna-found hyperparameters from src/core/experiment_config.yaml:
#   - n_layers=2, n_filters=[112,32], n_heads=[6,4], fc_size=96, dropout=0.4
#   - use_gine_first_layer=true, use_norm=true (batch), use_residual=true
#   - max_trials=4, directed=true, self_loops=true, corr_threshold=0.1
#   - All model/graph hyperparams sourced from YAML; do NOT override max_trials here.
#   - epochs=150 (CORAL-confirmed improvement over 100), lr=1e-3, batch_size=8
#
# Sweeps: k_folds (5, 10) × signal_type (hbo, hbr, hbt) = 6 runs total.
# Dataset: processed-new (GNG task; uses pre-defined subject-level splits).
#
# Results land at:
#   research/experiments/<DATE>/kfold/<K>-fold/<DATE>/<exp_name>/
#
# Usage:
#   bash scripts/run_kfold.sh                              # all signal types, 5-fold and 10-fold
#   bash scripts/run_kfold.sh hbt                         # hbt only
#   bash scripts/run_kfold.sh hbo hbr                     # two signal types
#   bash scripts/run_kfold.sh --checkpoint_metric=loss    # checkpoint on lowest val loss
#   bash scripts/run_kfold.sh hbo --checkpoint_metric=loss

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_ROOT/src/core/experiment_config.yaml"
DATE_TAG="$(date +%Y%m%d)"
BASE_SAVE="$REPO_ROOT/research/experiments/$DATE_TAG/kfold"
DATA_DIR="$REPO_ROOT/data/processed-new-mc"
SPLITS_JSON="$REPO_ROOT/data/splits/kfold_splits_processed_new_mc.json"

# Training settings (max_trials and model/graph params come from experiment_config.yaml)
EPOCHS=150
LR=6.79e-03
BATCH_SIZE=8
PATIENCE=9999
SEED=42
NUM_WORKERS=0

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

for K_FOLDS in 5 10; do
    for DATA_TYPE in "${DATA_TYPES[@]}"; do

        echo ""
        echo "=========================================="
        echo "k_folds: $K_FOLDS | Signal: $DATA_TYPE | epochs: $EPOCHS | lr: $LR | ckpt: $CHECKPOINT_METRIC"
        echo "=========================================="
        python -m src.core.main \
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
            --num_workers "$NUM_WORKERS"

    done
done

echo ""
echo "All experiments complete. Results in: $BASE_SAVE"
