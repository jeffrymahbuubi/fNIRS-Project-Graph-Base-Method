# `src/core_st` — Windowed Spatial-Temporal fNIRS Pipeline

Spatial-temporal GNN pipeline for fNIRS anxiety classification.
A windowed variant of `src/core/`: the model applies shared GATv2 layers per time window,
then uses a GRU + additive attention to aggregate temporal context.

---

## Module Overview

| File | Responsibility |
|---|---|
| `models.py` | `WindowedSpatioTemporalGATNet` — GATv2 (per window) → GRU → classifier |
| `dataset.py` | `fNIRSGraphDataset` — loads `.npy` trials, builds graphs with per-channel z-scored raw time series (`Data.x = [23, T]`) |
| `transforms.py` | `StandardizeGraphFeatures` (edge_attr only); `DropoutEdgeAugmentation`; `MaskFeatureAugmentation` |
| `training.py` | `train_epoch`, `evaluate`, `EarlyStopping`, `FocalLoss`, `CosineWarmupScheduler`; full training runners (`perform_holdout_training`, `perform_kfold_training`, `perform_loso_training`) |
| `config.py` | `ExperimentConfig` dataclass; `load_config` / `save_config` (YAML) |
| `utils.py` | `set_seed`, `get_experiment_dir`, `pearson_correlation_matrix`, `coherence_matrix` |
| `optuna_search.py` | Hyperparameter search: `design_search_space_st`, `objective_st`, `run_optuna_st` |
| `main.py` | CLI entry point — parses args, builds model, dispatches to training runners |
| `experiment_config.yaml` | Default YAML config (overridden by CLI flags) |

---

## Architecture — `WindowedSpatioTemporalGATNet`

```
Data.x [B×C, T]  (C=23 channels, T=326 timepoints, z-scored per-channel per-trial)
        │
        ▼
x.unfold(dim=1, size=W, step=S)  →  [B×C, K, W]    K windows of size W
        │
        ▼
_compute_window_stats()          →  [B×C, K, 6]    6 stats per window per node
        │                                           (mean, min, max, var, skew, kurtosis)
        ▼  (for each window k, shared weights)
GATv2Conv × n_layers             →  [B×C, n_filters×heads]
global_mean_pool                 →  [B, n_filters×heads]
Linear (pre_gru)                 →  [B, temporal_hidden]
        │
        ▼  (stack K windows)
GRU (temporal_layers)            →  [B, K, temporal_hidden]
Additive attention               →  [B, temporal_hidden]   (weighted sum over K)
        │
        ▼
Linear (pre_cls) → Linear (classifier)  →  [B, 2]
```

**Key invariant:** `in_channels=6` is always fixed — `_compute_window_stats()` always emits 6 features
regardless of `window_size`. The windowing and stat extraction happen inside `forward()`.

---

## Data Directory Structure

```
data/processed-new/
└── GNG/
    ├── anxiety/
    │   └── <subject_id>/
    │       └── hbo/          # or hbr / hbt
    │           ├── 0.npy     # shape: [23, 326]
    │           ├── 1.npy
    │           └── ...
    └── healthy/
        └── ...
```

`data_dir` passed to the CLI or dataset should point to `data/processed-new`
(not `data/processed-new/GNG`). The dataset appends `task_type` internally.

---

## Running Training

### Via CLI

```bash
# Holdout training — HbO, GNG task, 100 epochs
python -m src.core_st.main \
    --data_dir data/processed-new \
    --task_type GNG \
    --data_type hbo \
    --validation holdout \
    --epochs 100 \
    --lr 1e-3 \
    --n_layers 2 \
    --n_filters 64 \
    --n_heads 4 \
    --temporal_hidden 64 \
    --window_size 32 \
    --window_stride 16 \
    --save_dir experiments/st_run

# K-fold training with pre-defined splits
python -m src.core_st.main \
    --data_dir data/processed-new \
    --validation kfold \
    --k_folds 5 \
    --splits_json data/splits/gng_splits.json \
    --epochs 100 \
    --save_dir experiments/st_kfold
```

### Via YAML config

```bash
# Edit src/core_st/experiment_config.yaml, then:
python -m src.core_st.main \
    --config src/core_st/experiment_config.yaml \
    --data_dir data/processed-new
```

CLI flags override YAML values.

---

## Hyperparameter Search with Optuna

`optuna_search.py` finds the best combination of model and training parameters.
Results are persisted to SQLite (`{save_dir}/optuna_study.db`) and can be resumed.

### Run from CLI

```bash
# Basic run — HbO, 500 trials, 100 epochs each
python -m src.core_st.optuna_search \
    --data_dir data/processed-new \
    --data_type hbo \
    --n_trials 500 \
    --n_epochs 100 \
    --base_dir experiments/optuna_st

# With focal-loss params in the search space
python -m src.core_st.optuna_search \
    --data_dir data/processed-new \
    --data_type hbo \
    --n_trials 500 \
    --n_epochs 100 \
    --use_fl \
    --base_dir experiments/optuna_st_fl

# Limit trials per subject (faster, for quick exploration)
python -m src.core_st.optuna_search \
    --data_dir data/processed-new \
    --max_trials 2 \
    --n_trials 100 \
    --n_epochs 50
```

### Run from Python

```python
from src.core_st.optuna_search import run_optuna_st

study = run_optuna_st(
    data_dir="data/processed-new",
    data_type="hbo",
    n_trials=500,
    n_epochs=100,
    base_dir="experiments/optuna_st",
    seed=42,
)
print(study.best_params)
```

### Search Space Summary

| Group | Parameter | Range |
|---|---|---|
| Windowing | `window_size` | {16, 32, 48, 64} — step 16 (1.6 s at 10 Hz) |
| Windowing | `window_stride` | {8, 16, 24, 32} — step 8; clamped to ≤ `window_size` |
| Spatial | `n_layers` | {1, 2, 3} |
| Spatial | `n_filters` | {16 … 128} — step 16 |
| Spatial | `heads` | {2, 4, 6, 8} |
| Temporal | `temporal_hidden` | {32 … 256} — step 32 |
| Temporal | `temporal_layers` | {1, 2, 3} |
| Regularization | `dropout` | {0.1 … 0.5} — step 0.1 |
| Regularization | `use_residual` | {True, False} |
| Regularization | `use_norm` | {True, False} |
| Regularization | `norm_type` | {batch, layer} — conditional on `use_norm=True` |
| Classifier | `fc_size` | {32 … 256} — step 32 |
| Optimizer | `learning_rate` | log-uniform [1e-5, 1e-1] |

`in_channels=6` is **not** a search parameter — it is always fixed.

### Analyzing Results

```python
import optuna

study = optuna.load_study(
    study_name="st_hbo_mt2_ep100_tr500",
    storage="sqlite:///experiments/optuna_st/20260429/st_hbo_.../optuna_study.db",
)

# Top-5 trials by F1
trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
for t in trials[:5]:
    print(f"Trial #{t.number}  F1={t.value:.4f}  params={t.params}")

# Optuna visualizations
import optuna.visualization as vis
vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
vis.plot_slice(study).show()
```

---

## Explainability

After training, call `model.explain(data, device)` to retrieve attention weights:

```python
result = model.explain(data=dataset[0], device=device)

# Temporal attention — which windows were most influential
print(result["temporal_attention"].shape)   # [K]

# Spatial attention per window per GATv2 layer
# result["spatial_attention"][k][l].shape = [E, heads]
print(result["n_windows"])                  # K
```

---

## Key Differences from `src/core/` (baseline)

| | `src/core/` | `src/core_st/` |
|---|---|---|
| `Data.x` | `[23, 6]` statistical features | `[23, 326]` raw z-scored time series |
| Model | `FlexibleGATNet` (static) | `WindowedSpatioTemporalGATNet` (windowed) |
| `n_filters` / `heads` | per-layer lists | single int (shared across layers) |
| Temporal module | none | GRU + additive attention over K windows |
| `compute_stats()` | x + edge stats | edge stats only (`mean_ea`, `std_ea`) |
| `StandardizeGraphFeatures` | normalizes x and edge_attr | normalizes edge_attr only |
| `use_gine_first_layer` | present | removed |
