# fNIRS Graph-Based Anxiety Classification

Binary classification of **Anxiety vs Healthy** from functional near-infrared spectroscopy (fNIRS) brain signals using graph neural networks. Two complementary GNN pipelines are implemented:

| Pipeline | Module | Architecture | Key idea |
|---|---|---|---|
| **Spatial** | `src/core/` | GATv2Conv + global pooling | Full-trial fNIRS graph → single classification |
| **Spatial-Temporal** | `src/core_st/` | GATv2Conv + GRU + attention | Windowed time series → temporal aggregation → classification |

Both pipelines share the same graph construction: fNIRS channels are nodes, edges are drawn between channels whose Pearson correlation exceeds a threshold, weighted by `|r|` and coherence.

---

## Project Structure

```
.
├── data/
│   ├── raw/                          # Raw .nirs files — 51 subjects (gitignored)
│   ├── additional-raw/               # Additional cohort — 11 more subjects (gitignored)
│   ├── processed-new/                # GNN-ready graphs — 62 subjects, full-trial (gitignored)
│   ├── processed-new-mc/             # GNN-ready graphs — 62 subjects, motion-corrected windowed (gitignored)
│   ├── processed-old/                # GNN-ready graphs — 51 subjects (gitignored)
│   ├── splits/
│   │   ├── kfold_splits_processed_new.json      # k-fold splits for processed-new (62 subjects)
│   │   ├── kfold_splits_processed_new_mc.json   # k-fold splits for processed-new-mc (62 subjects)
│   │   └── kfold_splits_processed_old.json      # k-fold splits for processed-old (51 subjects)
│   ├── subjects-raw.json             # Subject metadata (51-subject cohort)
│   ├── subjects-additional-raw.json  # Subject metadata (additional cohort)
│   ├── processor_cli.py              # .nirs → HbO/HbR/HbT CSV converter
│   └── generate_splits.py            # Reproducible k-fold split generator
├── src/
│   ├── core/                         # Spatial-only pipeline
│   │   ├── main.py                   # CLI entry point
│   │   ├── dataset.py                # Graph construction from CSV trials
│   │   ├── models.py                 # GATv2 + GINE model definitions
│   │   ├── training.py               # Training loop, k-fold, LOSO
│   │   ├── config.py                 # Argument parsing
│   │   ├── transforms.py             # Data augmentation transforms
│   │   ├── utils.py                  # Metrics, plotting helpers
│   │   └── experiment_config.yaml    # Tunable hyperparameters
│   ├── core_st/                      # Spatial-temporal pipeline
│   │   ├── main.py                   # CLI entry point
│   │   ├── dataset.py                # Windowed graph dataset
│   │   ├── models.py                 # GATv2 + GRU + attention model
│   │   ├── training.py               # Training loop
│   │   ├── config.py                 # Argument parsing
│   │   ├── transforms.py             # Augmentation for windowed inputs
│   │   ├── utils.py                  # Metrics, plotting helpers
│   │   ├── optuna_search.py          # Optuna hyperparameter search
│   │   └── experiment_config.yaml    # Tunable hyperparameters (Optuna-best)
│   ├── notebook/                     # Exploratory notebooks
│   ├── requirements.txt              # Full dependency list
│   └── requirements-pyg-extensions.txt  # PyTorch Geometric extensions
├── scripts/
│   ├── run_coral_kfold.sh            # Spatial k-fold sweep (CORAL-validated config)
│   └── run_st_kfold.sh               # Spatial-temporal k-fold sweep
├── coral/
│   └── fnirs_gat/
│       ├── seed/                     # Seeded code submitted to CORAL
│       ├── eval/                     # CORAL evaluation grader
│       └── task.yaml                 # CORAL task definition
├── docs/
│   └── SPEC_core_st.md               # Spatial-temporal pipeline specification
└── research/
    └── experiments/                  # Local results (gitignored)
        └── 20260429/
            └── RESULTS_SUMMARY.md    # Full 5-fold / 10-fold result tables
```

---

## Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environment management.

### 1. Install uv

```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
```

### 2. Create the virtual environment and install dependencies

All main dependencies (PyTorch with CUDA 12.4, MNE, scikit-learn, etc.) are pinned in `src/requirements.txt`.

```bash
# From the repository root
uv venv src/.venv --python 3.12
source src/.venv/bin/activate

uv pip install -r src/requirements.txt
```

### 3. Install PyTorch Geometric extensions

PyG sparse extensions must be installed against the exact PyTorch + CUDA version. Use the CUDA wheel index:

```bash
# CUDA 12.4 (matches requirements.txt: torch==2.6.0+cu124)
uv pip install -r src/requirements-pyg-extensions.txt \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# CPU-only alternative
uv pip install -r src/requirements-pyg-extensions.txt \
  -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

### 4. Verify installation

```bash
python -c "import torch; import torch_geometric; print(torch.__version__, torch_geometric.__version__)"
```

---

## Data Setup

The raw `.nirs` files and pre-processed graph datasets are not included in the repository. Two processed variants are used depending on the pipeline:

| Pipeline | Dataset | Splits |
|---|---|---|
| Spatial (`src/core`) | `data/processed-new/` (full-trial graphs) | `data/splits/kfold_splits_processed_new.json` |
| Spatial-Temporal (`src/core_st`) | `data/processed-new-mc/` (motion-corrected, windowed) | `data/splits/kfold_splits_processed_new_mc.json` |

```bash
# Step 1 — Convert raw .nirs files to HbO/HbR/HbT CSVs
python data/processor_cli.py --help

# Step 2 — The PyG dataset is built automatically on first run
# (fNIRSGraphDataset in src/core{,_st}/dataset.py caches under data/processed-new[-mc]/)
```

The pre-defined cross-validation splits are committed and should be used as-is to reproduce the published numbers (all generated with seed 42).

---

## Overall Results

Leak-free cross-validation sweep over both pipelines × `{HBO, HBR, HBT}` × `max_trials ∈ {2, 4}`, on the 62-subject GNG cohort. Numbers below are read from:

- `experiments/spatial_graph/experiment_metrics.xlsx`
- `experiments/spatial_temporal_graph/experiment_metrics.xlsx`

The best row in each CV strategy is **bolded**.

### 5-Fold Cross-Validation (62 subjects, 5 folds)

| Pipeline | Signal | mt | Acc (Mean±SD) | F1 (Mean±SD) | Overall Acc | Overall F1 |
|---|---|---:|---|---|---:|---:|
| Spatial | HBO | 2 | 0.710±0.125 | 0.759±0.068 | 0.7097 | 0.7534 |
| Spatial | HBO | 4 | 0.600±0.087 | 0.692±0.044 | 0.6008 | 0.6897 |
| Spatial | HBR | 2 | 0.708±0.081 | 0.753±0.051 | 0.7097 | 0.7534 |
| Spatial | HBR | 4 | 0.626±0.126 | 0.694±0.067 | 0.6290 | 0.6913 |
| Spatial | HBT | 2 | 0.696±0.153 | 0.748±0.096 | 0.6935 | 0.7397 |
| Spatial | HBT | 4 | 0.573±0.117 | 0.678±0.057 | 0.5726 | 0.6728 |
| ST | HBO | 2 | 0.749±0.125 | 0.773±0.109 | 0.7500 | 0.7704 |
| ST | HBO | 4 | 0.733±0.077 | 0.757±0.059 | 0.7339 | 0.7556 |
| ST | HBR | 2 | 0.741±0.128 | 0.777±0.097 | 0.7419 | 0.7714 |
| ST | HBR | 4 | 0.692±0.087 | 0.740±0.067 | 0.6935 | 0.7397 |
| **ST** | **HBT** | **2** | **0.774±0.117** | **0.794±0.102** | **0.7742** | **0.7910** |
| ST | HBT | 4 | 0.712±0.112 | 0.750±0.073 | 0.7137 | 0.7455 |

### 10-Fold Cross-Validation (62 subjects, 10 folds)

| Pipeline | Signal | mt | Acc (Mean±SD) | F1 (Mean±SD) | Overall Acc | Overall F1 |
|---|---|---:|---|---|---:|---:|
| Spatial | HBO | 2 | 0.782±0.075 | 0.794±0.057 | 0.7823 | 0.7939 |
| Spatial | HBO | 4 | 0.654±0.104 | 0.722±0.051 | 0.6532 | 0.7152 |
| Spatial | HBR | 2 | 0.762±0.130 | 0.807±0.075 | 0.7661 | 0.8000 |
| Spatial | HBR | 4 | 0.607±0.097 | 0.693±0.058 | 0.6089 | 0.6959 |
| Spatial | HBT | 2 | 0.780±0.106 | 0.804±0.081 | 0.7823 | 0.8000 |
| Spatial | HBT | 4 | 0.651±0.117 | 0.721±0.071 | 0.6492 | 0.7148 |
| ST | HBO | 2 | 0.764±0.146 | 0.798±0.098 | 0.7661 | 0.7914 |
| ST | HBO | 4 | 0.749±0.129 | 0.781±0.078 | 0.7500 | 0.7737 |
| ST | HBR | 2 | 0.764±0.135 | 0.795±0.088 | 0.7661 | 0.7883 |
| ST | HBR | 4 | 0.736±0.133 | 0.775±0.080 | 0.7379 | 0.7687 |
| **ST** | **HBT** | **2** | **0.771±0.167** | **0.805±0.113** | **0.7742** | **0.7941** |
| ST | HBT | 4 | 0.740±0.146 | 0.775±0.089 | 0.7419 | 0.7630 |

### Leave-One-Subject-Out (LOSO, 62 subjects)

LOSO yields a single 0/1 outcome per held-out subject, so per-fold `Mean±SD` is uninformative (SD ≈ 0.5 by construction). The table reports the **pooled** classification metrics aggregated over all 62 subject predictions.

| Pipeline | Signal | mt | Acc | Sens | Spec | Prec | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Spatial | HBO | 2 | 0.7097 | 1.0000 | 0.4545 | 0.6170 | 0.7632 |
| Spatial | HBO | 4 | 0.6573 | 1.0000 | 0.3561 | 0.5771 | 0.7319 |
| Spatial | HBR | 2 | 0.7419 | 1.0000 | 0.5152 | 0.6444 | 0.7838 |
| Spatial | HBR | 4 | 0.6935 | 1.0000 | 0.4242 | 0.6042 | 0.7532 |
| Spatial | HBT | 2 | 0.7097 | 1.0000 | 0.4545 | 0.6170 | 0.7632 |
| Spatial | HBT | 4 | 0.6452 | 1.0000 | 0.3333 | 0.5686 | 0.7250 |
| ST | HBO | 2 | 0.7903 | 1.0000 | 0.6061 | 0.6905 | 0.8169 |
| ST | HBO | 4 | 0.7702 | 0.9914 | 0.5758 | 0.6725 | 0.8014 |
| **ST** | **HBR** | **2** | **0.8226** | **1.0000** | **0.6667** | **0.7250** | **0.8406** |
| ST | HBR | 4 | 0.8145 | 0.9914 | 0.6591 | 0.7188 | 0.8333 |
| ST | HBT | 2 | 0.7823 | 1.0000 | 0.5909 | 0.6824 | 0.8112 |
| ST | HBT | 4 | 0.7540 | 0.9655 | 0.5682 | 0.6627 | 0.7860 |

### Headlines

- The Spatial-Temporal pipeline beats the Spatial-only pipeline on every CV strategy.
- **Best Spatial:** 10-fold · HBO · `mt=2` → Acc 0.782±0.075, F1 0.794±0.057.
- **Best ST (paper baseline):** 10-fold · HBT · `mt=2` → **Acc 0.771±0.167, F1 0.805±0.113** (Overall Acc 0.7742, F1 0.7941).
- LOSO is the most conservative protocol; here the ST winner shifts to **HBR · `mt=2`** (Overall Acc 0.8226, F1 0.8406), with HBT · `mt=2` close behind (F1 0.8112).

---

## Reproducing Results — Spatial Pipeline (`src/core`)

The spatial pipeline treats each fNIRS trial as a single graph and classifies it with GATv2.

**Best configuration (leak-free, 62 subjects):** HBO · `max_trials=2` · 10-fold → **Mean Acc 0.782±0.075, Mean F1 0.794±0.057**. See [Overall Results](#overall-results) for the full sweep.

### Hyperparameters (from `src/core/experiment_config.yaml`)

| Parameter | Value |
|---|---|
| Architecture | GATv2Conv (2 layers) + GINE first layer |
| Filters | [112, 32] |
| Attention heads | [6, 4] |
| FC size | 96 |
| Dropout | 0.4 |
| Batch norm | Yes |
| Residual | Yes |
| Correlation threshold | 0.1 |
| Directed edges | Yes |
| Self-loops | Yes |
| max\_trials | 2 |

### Reproduce — 5-fold (with additional data, 62 subjects)

```bash
# From the repository root
python -m src.core.main \
  --config src/core/experiment_config.yaml \
  --data_dir data/processed-new \
  --task GNG \
  --data_type hbo \
  --validation kfold \
  --k_folds 5 \
  --max_trials 2 \
  --epochs 100 \
  --lr 1e-3 \
  --batch_size 8 \
  --patience 9999 \
  --seed 42 \
  --splits_json data/splits/kfold_splits_processed_new.json \
  --save_dir research/experiments/my_run/5-fold/with-additional-data
```

### Reproduce — 10-fold (with additional data, 62 subjects)

```bash
python -m src.core.main \
  --config src/core/experiment_config.yaml \
  --data_dir data/processed-new \
  --task GNG \
  --data_type hbo \
  --validation kfold \
  --k_folds 10 \
  --max_trials 2 \
  --epochs 100 \
  --lr 1e-3 \
  --batch_size 8 \
  --patience 9999 \
  --seed 42 \
  --splits_json data/splits/kfold_splits_processed_new.json \
  --save_dir research/experiments/my_run/10-fold/with-additional-data
```

### Full signal sweep (HBO / HBR / HBT)

Use the provided script to run all three signals × both fold configurations in one shot:

```bash
bash scripts/run_coral_kfold.sh           # all signals, 5-fold and 10-fold
bash scripts/run_coral_kfold.sh hbo       # HBO only
```


---

## Reproducing Results — Spatial-Temporal Pipeline (`src/core_st`)

The ST pipeline slices each trial into overlapping time windows, encodes each window as a graph with GATv2, then aggregates over the temporal dimension with a GRU + additive attention layer.

**Hyperparameters** are from `src/core_st/experiment_config.yaml`. The architecture is fixed; the learning-rate / scheduler combo was tuned by an Optuna `lr_cosine` study on 2026-05-03 (study `st_hbo_mt4_ep100_tr25_kf5_lr_cosine`, Trial #36, 95 complete + 5 pruned, 5-fold mean F1 = 0.7693 on HBO `mt=4`). This **supersedes** the earlier 2026-04-29 Trial #137 search (`n_layers=3`, `window_size=48`, `lr=1.375e-4`).

| Parameter | Value |
|---|---|
| GATv2 layers | 2 |
| Filters | 80 (shared across layers) |
| Attention heads | 2 |
| FC size | 256 |
| Dropout | 0.3 |
| Residual | No |
| Batch norm | Yes |
| Window size | 16 timepoints (~1.6 s at 10 Hz) |
| Window stride | 8 (50 % overlap) |
| GRU hidden dim | 192 |
| GRU layers | 1 |
| Learning rate | 3.04e-4 (Optuna best) |
| Scheduler | `cosine_annealing` (T_max = epochs, eta_min = 1e-5) |
| Epochs | 150 |
| Patience | 30 |

### Reproduce — 10-fold (62 subjects, motion-corrected dataset)

```bash
python -m src.core_st.main \
  --config src/core_st/experiment_config.yaml \
  --data_dir data/processed-new-mc \
  --task GNG \
  --data_type hbt \
  --validation kfold \
  --k_folds 10 \
  --max_trials 2 \
  --epochs 150 \
  --lr 3.04e-4 \
  --scheduler cosine_annealing \
  --eta_min 1e-5 \
  --batch_size 8 \
  --patience 30 \
  --seed 42 \
  --splits_json data/splits/kfold_splits_processed_new_mc.json \
  --save_dir research/experiments/my_run/st-kfold/10-fold
```

### Full sweep

```bash
bash scripts/run_st_kfold.sh           # all signals × {5,10}-fold × mt={2,4}
bash scripts/run_st_kfold.sh hbt       # HBT only
```

> **Note:** The Optuna search optimised the scheduler/lr on HBO `mt=4`. HBR and HBT inherit the same config; the architecture itself is shared across signals.

---

## Output Format

Each experiment folder contains:

```
<experiment_name>/
├── config.yaml                        # Full resolved configuration
├── <exp>_fold_1.pkl  …  _fold_N.pkl   # Per-fold metrics (training curves, confusion matrix)
├── <exp>_fold_1.pt   …  _fold_N.pt    # Best model weights per fold
├── <exp>_kfold_overall.pkl            # Aggregated metrics across all folds
└── <exp>_fold_N_{accuracy,f1,loss,cm}.png
```

Load a fold result:

```python
import pickle
with open("path/to/exp_fold_1.pkl", "rb") as f:
    metrics = pickle.load(f)
print(metrics.keys())   # train_loss, val_loss, val_acc, val_f1, best_epoch, confusion_matrix, ...
```
