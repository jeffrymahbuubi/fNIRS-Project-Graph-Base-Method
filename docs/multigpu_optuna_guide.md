# Multi-GPU Optuna Hyperparameter Search (Single Machine)

This guide covers running parallel Optuna optimization across multiple GPUs on a single machine (e.g. vast.ai 4× RTX 4060 Ti).

---

## Why Multiple Processes, Not `n_jobs > 1`

Optuna's `n_jobs > 1` uses Python **threads** inside one process. Threads share a CUDA context, so all trials land on GPU 0 regardless of `n_jobs`. The other GPUs sit idle.

The correct approach: **one process per GPU**, each with `CUDA_VISIBLE_DEVICES=N`. Every process thinks it has exactly one GPU. All processes share a single study via a common storage file.

```
[JournalFile: /tmp/optuna_st_hbo.log]   ← shared study state
        ↕            ↕            ↕            ↕
  [Process 0]  [Process 1]  [Process 2]  [Process 3]
  CUDA=0       CUDA=1       CUDA=2       CUDA=3
  1 trial/time  1 trial/time  1 trial/time  1 trial/time
```

---

## Prerequisites

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# No database server required — JournalFileBackend uses a plain log file
```

---

## Option A — Use the Launch Script (Recommended)

`scripts/run_optuna_multigpu_st.sh` handles everything: GPU detection, log directory creation, process launching, and result reporting.

```bash
# Auto-detect all GPUs, HBO signal, 3-fold kfold (defaults)
bash scripts/run_optuna_multigpu_st.sh

# Explicit: 4 GPUs, HBO signal, kfold strategy
bash scripts/run_optuna_multigpu_st.sh 4 hbo kfold

# 4 GPUs, HBR signal, kfold
bash scripts/run_optuna_multigpu_st.sh 4 hbr kfold

# 4 GPUs, HBO, holdout (faster but less stable)
bash scripts/run_optuna_multigpu_st.sh 4 hbo holdout
```

Logs are written to `research/experiments/multigpu/<YYYYMMDD>/worker_logs/gpu{N}.log`.

---

## Option B — Manual Commands (4 terminals or backgrounded)

### ST Pipeline (`src/core_st`)

```bash
JOURNAL="/tmp/optuna_st_hbo.log"
SPLITS="data/splits/kfold_splits_processed_new_mc.json"

for GPU in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU python -m src.core_st.optuna_search \
        --data_dir data/processed-new-mc \
        --data_type hbo \
        --max_trials 4 \
        --n_trials 200 \
        --n_epochs 100 \
        --num_workers 8 \
        --eval_strategy kfold \
        --inner_folds 3 \
        --splits_json "$SPLITS" \
        --update_interval 10 \
        --early_stop_patience 15 \
        --storage_url "journal:$JOURNAL" \
        > /tmp/st_gpu${GPU}.log 2>&1 &
done
wait
echo "All ST workers done."
```

### Core (Baseline) Pipeline (`src/core`)

```bash
JOURNAL="/tmp/optuna_core_hbo.log"
SPLITS="data/splits/kfold_splits_processed_new_mc.json"

for GPU in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU python -m src.core.optuna_search \
        --data_dir data/processed-new-mc \
        --data_type hbo \
        --max_trials 4 \
        --n_trials 300 \
        --n_epochs 100 \
        --num_workers 8 \
        --eval_strategy kfold \
        --inner_folds 3 \
        --splits_json "$SPLITS" \
        --update_interval 10 \
        --early_stop_patience 15 \
        --storage_url "journal:$JOURNAL" \
        > /tmp/core_gpu${GPU}.log 2>&1 &
done
wait
echo "All core workers done."
```

---

## Monitoring Progress

```bash
# Follow one worker's output
tail -f /tmp/st_gpu0.log

# Watch all workers simultaneously (updates every 5 seconds)
watch -n 5 'grep "Progress:" /tmp/st_gpu*.log | tail -4'

# Check how many trials are complete
python -c "
import optuna
from optuna.storages import JournalStorage, JournalFileBackend
study = optuna.load_study(
    study_name='st_hbo_mt4_ep100_tr200_kf3',
    storage=JournalStorage(JournalFileBackend('/tmp/optuna_st_hbo.log'))
)
complete = [t for t in study.trials if t.state.name == 'COMPLETE']
pruned  = [t for t in study.trials if t.state.name == 'PRUNED']
print(f'Complete : {len(complete)}')
print(f'Pruned   : {len(pruned)}')
print(f'Best F1  : {study.best_value:.4f}  (Trial #{study.best_trial.number})')
print(f'Best params: {study.best_params}')
"
```

---

## Storage Backend Options

| `--storage_url` value | Backend | Use case |
|---|---|---|
| *(omitted)* | SQLite `{save_dir}/optuna_study.db` | Single process only |
| `journal:/path/to/file.log` | JournalFileBackend | **Single machine, multi-process** |
| `mysql+pymysql://user:pass@host/db` | MySQL | Multi-machine |
| `postgresql://user:pass@host/db` | PostgreSQL | Multi-machine |

> **Note:** SQLite breaks with concurrent writes from multiple processes. Always use `journal:` or an RDB backend when running multiple workers.

---

## Recommended Configuration

| Parameter | Value | Reason |
|---|---|---|
| `--num_workers` | `8` | AMD EPYC 7K62: 96 threads ÷ 4 GPU processes = 24/process; 8 is safe and sufficient for dataset size (248 graphs) |
| `--eval_strategy` | `kfold` | More stable objective signal than holdout for ST's 14-param space |
| `--inner_folds` | `3` | Best balance: ~40% cheaper than 5-fold with acceptable robustness |
| `--n_trials` | `200` | TPE converges well by 200 trials for 14 params |
| `--early_stop_patience` | `15` | Recommended for 3-fold kfold strategy |
| `--n_jobs` | `1` | Leave at 1 — parallelism comes from multiple processes, not threads |

---

## Speed Estimates (4× RTX 4060 Ti, 3-fold × 200 trials)

| Setup | Est. wall time |
|---|---|
| 1 GPU (local machine) | ~6–7 h |
| 4 GPUs (vast.ai) | ~1.5–2 h |

---

## Inspecting Results After Completion

```python
import optuna
from optuna.storages import JournalStorage, JournalFileBackend
import pandas as pd

study = optuna.load_study(
    study_name="st_hbo_mt4_ep100_tr200_kf3",
    storage=JournalStorage(JournalFileBackend("/tmp/optuna_st_hbo.log"))
)

# Summary
print(f"Best F1     : {study.best_value:.4f}")
print(f"Best trial  : #{study.best_trial.number}")
print(f"Best params : {study.best_params}")

# Full results as DataFrame
df = study.trials_dataframe()
df_complete = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False)
print(df_complete[["number", "value", "params_learning_rate", "params_n_layers"]].head(10))
```
