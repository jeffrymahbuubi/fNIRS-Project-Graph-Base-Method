#!/bin/bash
# Distributed Optuna ST search across N GPUs on a single machine (vast.ai).
#
# Each GPU runs as a separate process (CUDA_VISIBLE_DEVICES=N), all sharing
# a JournalFile on the local filesystem. This avoids SQLite's single-writer
# limit and Optuna's GIL-limited n_jobs threading — no DB server required.
#
# Prerequisites (run once on the instance):
#   pip install -r requirements.txt
#   # No MySQL/PostgreSQL needed — JournalFileBackend uses a shared log file.
#
# Usage:
#   bash scripts/run_optuna_multigpu_st.sh               # auto-detect GPUs
#   bash scripts/run_optuna_multigpu_st.sh 4             # use 4 GPUs
#   bash scripts/run_optuna_multigpu_st.sh 4 hbo         # 4 GPUs, HBO signal
#   bash scripts/run_optuna_multigpu_st.sh 4 hbo kfold   # 4 GPUs, HBO, kfold

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Config — edit these to match your run
# ---------------------------------------------------------------------------
N_GPUS="${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}"
DATA_TYPE="${2:-hbo}"
EVAL_STRATEGY="${3:-kfold}"
INNER_FOLDS=5
N_TRIALS=150
N_EPOCHS=100
MAX_TRIALS=4
EARLY_STOP_PATIENCE=15
NUM_WORKERS=8   # DataLoader workers per process (safe default for unknown CPU count)
DATA_DIR="$REPO_ROOT/data/processed-new-mc"
SPLITS_JSON="$REPO_ROOT/data/splits/kfold_splits_processed_new_mc.json"
BASE_DIR="$REPO_ROOT/research/experiments/multigpu/$(date +%Y%m%d)"

# JournalFile path — shared by all GPU processes on this machine
JOURNAL_FILE="$BASE_DIR/optuna_journal.log"
STORAGE_URL="journal:$JOURNAL_FILE"

LOG_DIR="$BASE_DIR/worker_logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ "$N_GPUS" -lt 1 ]; then
    echo "ERROR: No GPUs detected. Check nvidia-smi." >&2
    exit 1
fi

if [ "$EVAL_STRATEGY" = "kfold" ] && [ ! -f "$SPLITS_JSON" ]; then
    echo "ERROR: splits_json not found: $SPLITS_JSON" >&2
    exit 1
fi

echo "=============================================="
echo "Multi-GPU Optuna ST Search"
echo "  GPUs         : $N_GPUS"
echo "  Signal       : $DATA_TYPE"
echo "  Strategy     : $EVAL_STRATEGY (inner_folds=$INNER_FOLDS)"
echo "  Trials       : $N_TRIALS  Epochs: $N_EPOCHS"
echo "  Storage      : $JOURNAL_FILE (JournalFi
le)"
echo "  Logs         : $LOG_DIR"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Build common args
# ---------------------------------------------------------------------------
COMMON_ARGS=(
    --data_dir "$DATA_DIR"
    --data_type "$DATA_TYPE"
    --max_trials "$MAX_TRIALS"
    --n_trials "$N_TRIALS"
    --n_epochs "$N_EPOCHS"
    --num_workers "$NUM_WORKERS"
    --eval_strategy "$EVAL_STRATEGY"
    --early_stop_patience "$EARLY_STOP_PATIENCE"
    --update_interval 10
    --storage_url "$STORAGE_URL"
)

if [ "$EVAL_STRATEGY" = "kfold" ]; then
    COMMON_ARGS+=(--inner_folds "$INNER_FOLDS" --splits_json "$SPLITS_JSON")
fi

# ---------------------------------------------------------------------------
# Launch one worker per GPU
# ---------------------------------------------------------------------------
PIDS=()
for GPU_ID in $(seq 0 $((N_GPUS - 1))); do
    LOG_FILE="$LOG_DIR/gpu${GPU_ID}.log"
    echo "Starting worker GPU $GPU_ID → $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.core_st.optuna_search \
        "${COMMON_ARGS[@]}" \
        --base_dir "$BASE_DIR/gpu${GPU_ID}" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All $N_GPUS workers launched. PIDs: ${PIDS[*]}"
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/gpu0.log"
echo "  watch -n 5 'grep Progress $LOG_DIR/*.log | tail -$N_GPUS'"
echo ""

# ---------------------------------------------------------------------------
# Wait for all workers and report
# ---------------------------------------------------------------------------
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "GPU $i worker completed successfully."
    else
        echo "GPU $i worker FAILED (PID $PID). Check $LOG_DIR/gpu${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================="
if [ "$FAILED" -eq 0 ]; then
    echo "All $N_GPUS workers finished. Study complete."
else
    echo "WARNING: $FAILED worker(s) failed."
fi
echo "=============================================="
echo ""

STUDY_NAME="st_${DATA_TYPE}_mt${MAX_TRIALS}_ep${N_EPOCHS}_tr${N_TRIALS}_kf${INNER_FOLDS}"
echo "Inspect results:"
cat <<PYEOF
  python -c "
  import optuna
  from optuna.storages import JournalStorage
  try:
      from optuna.storages import JournalFileBackend
  except ImportError:
      from optuna.storages import JournalFileStorage as JournalFileBackend
  study = optuna.load_study(
      study_name='${STUDY_NAME}',
      storage=JournalStorage(JournalFileBackend('${JOURNAL_FILE}'))
  )
  complete = [t for t in study.trials if t.state.name == 'COMPLETE']
  print('Complete:', len(complete))
  print('Best F1 :', study.best_value)
  print('Best params:', study.best_params)
  "
PYEOF
