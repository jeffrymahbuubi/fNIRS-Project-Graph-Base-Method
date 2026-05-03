# Implementation Spec — CORAL fnirs_st Best Configuration → src/core_st/

**Status:** Ready to implement  
**Source:** CORAL study `fnirs-st-temporal-search` (2026-05-02, 131 attempts)  
**Full results:** `coral/fnirs_st/RESULTS_REPORT.md`

---

## Scheduler State (Corrected Understanding)

Before implementing, the current scheduler state across the codebase is:

| File | Scheduler | Notes |
|------|-----------|-------|
| `src/core_st/main.py` (lines 229–230) | `CosineWarmupScheduler` | All past training runs used this |
| `src/core_st/optuna_search.py` (objectives) | `ReduceLROnPlateau` | Optuna search used this; lr=0.000635 was found here |
| `coral/fnirs_st/seed/solution.py` (CORAL grader) | `ReduceLROnPlateau` | Baseline 0.7432 was produced under this |

**Implication:** lr=0.000635 was optimized under `ReduceLROnPlateau`. After switching to `CosineAnnealingLR`, the optimal LR may shift — the LR-only re-search (Change 2 below) addresses this.

---

## Scope

Two changes. Subject-level hard voting is **out of scope** (future work).

| # | Change | File(s) | Priority |
|---|--------|---------|----------|
| 1 | Add `--scheduler` toggle; default to `CosineAnnealingLR` | `main.py`, `config.py`, `training.py` | High |
| 2 | Add dedicated LR search for `CosineAnnealingLR` | `optuna_search.py` | High |
| — | Subject-level hard voting | `training.py`, `dataset.py` | Future work |

---

## Change 1 — Scheduler Toggle in `src/core_st/`

### Goal
Allow user to select scheduler via `--scheduler` CLI flag. Default switches from `CosineWarmupScheduler` to `CosineAnnealingLR`.

### Files affected
- `src/core_st/config.py` — add `scheduler` field
- `src/core_st/main.py` — replace hardcoded scheduler block with dispatch
- `src/core_st/training.py` — handle `ReduceLROnPlateau` needing `step(metric)`

### 1a. `src/core_st/config.py`

Add `scheduler` to `ExperimentConfig`:

```python
scheduler: str = "cosine_annealing"  # options: cosine_annealing | cosine_warmup | reduce_on_plateau
```

### 1b. `src/core_st/main.py`

Add CLI argument to `build_parser()`:
```python
p.add_argument(
    "--scheduler",
    type=str,
    default="cosine_annealing",
    choices=["cosine_annealing", "cosine_warmup", "reduce_on_plateau"],
    help="LR scheduler (default: cosine_annealing; CORAL-validated best for GRU)",
)
```

Wire it through `_args_to_config()`:
```python
scheduler=pick(args.scheduler, "scheduler", "cosine_annealing"),
```

Replace the hardcoded scheduler block (lines 229–230) in `main()`:
```python
# --- Scheduler dispatch ---
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
if cfg.scheduler == "cosine_annealing":
    scheduler_class = CosineAnnealingLR
    scheduler_params = {"T_max": cfg.epochs, "eta_min": 0}
elif cfg.scheduler == "cosine_warmup":
    scheduler_class = CosineWarmupScheduler
    scheduler_params = {"warmup": 5, "max_iters": cfg.epochs}
else:  # reduce_on_plateau
    scheduler_class = ReduceLROnPlateau
    scheduler_params = {"mode": "max", "factor": 0.5, "patience": 5}
```

### 1c. `src/core_st/training.py` — `_run_fold()` scheduler.step() fix

**Current issue:** `_run_fold` calls `scheduler.step()` with no arguments (line 286). This is correct for `CosineAnnealingLR` and `CosineWarmupScheduler` but **will raise a TypeError** for `ReduceLROnPlateau` which requires `scheduler.step(metric)`.

**Fix:** pass the val metric only when the scheduler is plateau-based:

```python
# Replace the current unconditional scheduler.step() call:
if scheduler is not None:
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(monitor_val)
    else:
        scheduler.step()
```

### Verification for Change 1

```bash
# Confirm cosine_annealing runs without error (default)
python -m src.core_st.main \
    --data_dir data/processed-new-mc/GNG \
    --validation kfold --k_folds 5 \
    --data_type hbo --max_trials 4 \
    --epochs 5 --patience 3 --batch_size 8 \
    --splits_json data/splits/kfold_splits_processed_new_mc.json

# Confirm reduce_on_plateau still works
python -m src.core_st.main \
    --data_dir data/processed-new-mc/GNG \
    --validation kfold --k_folds 5 \
    --scheduler reduce_on_plateau \
    --epochs 5 --patience 3 --batch_size 8 \
    --splits_json data/splits/kfold_splits_processed_new_mc.json
```

Expected: both complete without error. cosine_annealing F1 should trend higher than reduce_on_plateau over the same epochs.

---

## Change 2 — Dedicated LR Search for CosineAnnealingLR in `src/core_st/optuna_search.py`

### Goal
A "small" Optuna search that locks all architecture params to CORAL-winning values and searches only over `learning_rate` (and optionally `T_max`) under `CosineAnnealingLR`. Selectable via `--search_type lr_cosine`.

### Context
The existing `design_search_space_st()` searches 13+ hyperparameters. The CORAL study established that GRU with the Optuna-best spatial config is already optimal. The only unknown is: **what is the best starting LR under CosineAnnealingLR?** The current lr=0.000635 was found under `ReduceLROnPlateau` and may not be optimal for the new scheduler.

### Fixed values for "small" search (CORAL-validated best)

| Parameter | Value | Source |
|-----------|-------|--------|
| `temporal_type` | `gru` | CORAL winner |
| `temporal_hidden` | `192` | CORAL winner |
| `temporal_layers` | `1` | CORAL winner |
| `n_layers` | `2` | Optuna-locked |
| `n_filters` | `80` | Optuna-locked |
| `heads` | `2` | Optuna-locked |
| `window_size` | `16` | mt4 config |
| `window_stride` | `8` | mt4 config |
| `dropout` | `0.30` | CORAL winner |
| `fc_size` | `256` | CORAL winner |
| `use_residual` | `False` | Optuna-locked |
| `use_norm` | `True` | Optuna-locked |
| `norm_type` | `batch` | Optuna-locked |

### Search space (only these are suggested)

| Parameter | Range | Note |
|-----------|-------|------|
| `learning_rate` | [1e-5, 1e-1] log-uniform | Primary target |
| `T_max` | {50, 100, 150, 200} | Optional; default 100 = epochs |
| `eta_min` | {0.0, 1e-6, 1e-5} | Optional; default 0 |

### New function: `design_search_space_lr_cosine_st()`

```python
def design_search_space_lr_cosine_st(trial: optuna.Trial) -> Dict:
    """Small search: only LR/scheduler params; all architecture locked to CORAL best."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    T_max = trial.suggest_categorical("T_max", [50, 100, 150, 200])
    eta_min = trial.suggest_categorical("eta_min", [0.0, 1e-6, 1e-5])
    return {
        # Architecture: locked to CORAL best
        "window_size": 16, "window_stride": 8,
        "n_layers": 2, "n_filters": 80, "heads": 2,
        "temporal_hidden": 192, "temporal_layers": 1,
        "dropout": 0.30, "fc_size": 256,
        "use_residual": False, "use_norm": True, "norm_type": "batch",
        "focal_alpha": None, "focal_gamma": None,
        # Searched
        "learning_rate": learning_rate,
        "T_max": T_max,
        "eta_min": eta_min,
    }
```

### New objective: `objective_lr_cosine_st()` and `objective_kfold_lr_cosine_st()`

Mirror `objective_st()` and `objective_kfold_st()` exactly, except:
- Call `design_search_space_lr_cosine_st(trial)` instead of `design_search_space_st(trial)`
- Use `CosineAnnealingLR(T_max=hparams["T_max"], eta_min=hparams["eta_min"])` instead of `ReduceLROnPlateau`
- `_train_single_run_st` must call `scheduler.step()` (no args) — already correct for CosineAnnealingLR

Note: `_train_single_run_st` currently calls `scheduler.step(vl_f1)` (line 154). For `CosineAnnealingLR` this is wrong — it must be `scheduler.step()` with no args. The lr_cosine objectives need a separate training helper or a guard inside `_train_single_run_st`.

**Fix option (preferred):** add a `plateau_scheduler: bool = False` parameter to `_train_single_run_st`, and conditionally pass the metric:
```python
scheduler.step(vl_f1) if plateau_scheduler else scheduler.step()
```

### New CLI arg

```python
p.add_argument(
    "--search_type",
    default="full",
    choices=["full", "lr_cosine"],
    help="full: original 13-param search; lr_cosine: LR-only search under CosineAnnealingLR with locked CORAL best architecture",
)
```

### `run_optuna_st()` dispatch

```python
if search_type == "lr_cosine":
    obj_fn = lambda trial: objective_lr_cosine_st(...)   # or kfold variant
else:
    obj_fn = lambda trial: objective_st(...)              # existing full search
```

### Verification for Change 2

```bash
# Small search (lr_cosine) — should only log learning_rate, T_max, eta_min in trial params
python -m src.core_st.optuna_search \
    --data_dir data/processed-new-mc/GNG \
    --data_type hbo --max_trials 4 \
    --n_trials 20 --n_epochs 30 \
    --search_type lr_cosine \
    --eval_strategy kfold \
    --inner_folds 5 \
    --splits_json data/splits/kfold_splits_processed_new_mc.json

# Confirm trial params contain ONLY lr/T_max/eta_min (not n_layers, window_size, etc.)
```

---

## Out of Scope — Subject-Level Hard Voting (Future Work)

**What it is:** Group 4 trials per subject → majority class wins as subject-level prediction. CORAL showed +3pp mean F1 over trial-level (0.8721 vs 0.8323 for soft voting).

**Why deferred:** Requires `subject_id` to be passed through `evaluate()` and returned alongside `all_preds` / `all_labels`. `data.subject_id` is already populated in the dataset (line 123 in `dataset.py`), so no schema change needed — it's purely a training.py change. Deferred because early stopping behavior, tie-breaking rules, and dual metric reporting (trial-level + subject-level) need separate alignment.

**Where it goes when ready:** `src/core_st/training.py` — new `_subject_level_vote()` called after `_collect_fold_results()`.

---

## Implementation Order

```
1. Change 1a — config.py (add scheduler field)                 [~5 min]
2. Change 1b — main.py (add CLI arg + dispatch)                [~10 min]
3. Change 1c — training.py (_run_fold scheduler.step fix)      [~5 min]
4. Verify Change 1 (two smoke-test runs)
5. Change 2 — optuna_search.py (small search)                  [~30 min]
6. Verify Change 2 (20-trial smoke test)
```

---

## Expected Outcomes After Implementation

| Scenario | Expected F1 (trial-level) |
|----------|--------------------------|
| Full kfold, cosine_annealing, lr=0.000635 | ~0.80–0.87 (CORAL-validated range) |
| Full kfold, cosine_annealing, lr from lr_cosine search | ≥ 0.8721 possible with optimal LR |
| Full kfold, cosine_warmup (current default) | ~0.7792 (prior baseline) |
| Full kfold, reduce_on_plateau | ~0.7432 (CORAL grader baseline) |
