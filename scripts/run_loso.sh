#!/bin/bash
# Run baseline GNN (src/core, GATv2 / GINE-GAT) with Leave-One-Subject-Out (LOSO) validation.
#
# Uses the best Optuna-found hyperparameters from src/core/experiment_config.yaml:
#   - n_layers=2, n_filters=[112,32], n_heads=[6,4], fc_size=96, dropout=0.4
#   - use_gine_first_layer=true, use_norm=true (batch), use_residual=true
#   - max_trials=4, directed=true, self_loops=true, corr_threshold=0.1
#   - All model/graph hyperparams sourced from YAML; do NOT override max_trials here.
#   - epochs=150 (CORAL-confirmed improvement over 100), lr=1e-3, batch_size=8
#
# Sweeps: signal_type (hbo, hbr, hbt) = 3 runs total.
# Dataset: processed-new (GNG task; subject-level leave-one-out).
#
# Results land at:
#   research/experiments/<DATE>/loso/<DATE>/<exp_name>/
#
# Usage:
#   bash scripts/run_loso.sh                              # all signal types
#   bash scripts/run_loso.sh hbt                         # hbt only
#   bash scripts/run_loso.sh hbo hbr                     # two signal types
#   bash scripts/run_loso.sh --checkpoint_metric=loss    # checkpoint on lowest val loss
#   bash scripts/run_loso.sh hbo --checkpoint_metric=loss

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_ROOT/src/core/experiment_config.yaml"
DATE_TAG="$(date +%Y%m%d)"
BASE_SAVE="$REPO_ROOT/research/experiments/$DATE_TAG/loso"
DATA_DIR="$REPO_ROOT/data/processed-new"

# Training settings (max_trials and model/graph params come from experiment_config.yaml)
EPOCHS=150
LR=0.036652
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
    esac
done
if [ "${#DATA_TYPES[@]}" -eq 0 ]; then
    DATA_TYPES=(hbo hbr hbt)
fi

cd "$REPO_ROOT"

for DATA_TYPE in "${DATA_TYPES[@]}"; do

    echo ""
    echo "=========================================="
    echo "LOSO | Signal: $DATA_TYPE | epochs: $EPOCHS | lr: $LR | ckpt: $CHECKPOINT_METRIC"
    echo "=========================================="
    python -m src.core.main \
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
