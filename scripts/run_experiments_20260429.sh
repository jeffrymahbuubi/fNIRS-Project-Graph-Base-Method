#!/bin/bash
# Run 5-fold GNN experiments for 20260429.
#
# Sweeps: max_trials (2, 4) × signal types (hbo, hbr, hbt) × datasets (old, new)
# = 12 runs total.
#
# Hyperparameters come from src/core/experiment_config.yaml.
# Fold assignments are loaded from the pre-defined splits JSON via --splits_json,
# ensuring exact reproducibility regardless of subject ordering.
#
# Results land at:
#   research/experiments/20260429/{with,without}-additional-data/20260429/<exp_name>/
#
# Usage:
#   bash scripts/run_experiments_20260429.sh               # all signal types
#   bash scripts/run_experiments_20260429.sh hbo           # single signal type
#   bash scripts/run_experiments_20260429.sh hbo hbr       # two signal types

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_ROOT/src/core/experiment_config.yaml"
BASE_SAVE="$REPO_ROOT/research/experiments/20260429"
DATA_OLD="$REPO_ROOT/data/processed-old"
DATA_NEW="$REPO_ROOT/data/processed-new"
SPLITS_OLD="$REPO_ROOT/data/splits/kfold_splits_processed_old.json"
SPLITS_NEW="$REPO_ROOT/data/splits/kfold_splits_processed_new.json"

if [ "$#" -gt 0 ]; then
    DATA_TYPES=("$@")
else
    DATA_TYPES=(hbt)
fi

cd "$REPO_ROOT"

for MAX_TRIALS in 2 4; do
    for DATA_TYPE in "${DATA_TYPES[@]}"; do

        # echo ""
        # echo "=========================================="
        # echo "Dataset: OLD | Signal: $DATA_TYPE | max_trials: $MAX_TRIALS"
        # echo "=========================================="
        # python -m src.core.main \
        #     --config "$CONFIG" \
        #     --data_dir "$DATA_OLD" \
        #     --save_dir "$BASE_SAVE/without-additional-data" \
        #     --splits_json "$SPLITS_OLD" \
        #     --task GNG \
        #     --data_type "$DATA_TYPE" \
        #     --validation kfold \
        #     --k_folds 5 \
        #     --max_trials "$MAX_TRIALS" \
        #     --epochs 100 \
        #     --batch_size 8 \
        #     --lr 1e-3 \
        #     --patience 9999 \
        #     --seed 42

        echo ""
        echo "=========================================="
        echo "Dataset: NEW | Signal: $DATA_TYPE | max_trials: $MAX_TRIALS"
        echo "=========================================="
        python -m src.core.main \
            --config "$CONFIG" \
            --data_dir "$DATA_NEW" \
            --save_dir "$BASE_SAVE/loso/with-additional-data" \
            --splits_json "$SPLITS_NEW" \
            --task GNG \
            --data_type "$DATA_TYPE" \
            --validation loso \
            --k_folds 10 \
            --max_trials "$MAX_TRIALS" \
            --epochs 100 \
            --batch_size 8 \
            --lr 1e-3 \
            --patience 9999 \
            --seed 42 \
            --num_workers 4


    done
done

echo ""
echo "All experiments complete. Results in: $BASE_SAVE"
