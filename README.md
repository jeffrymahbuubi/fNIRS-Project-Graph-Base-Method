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
│   ├── processed-new/                # GNN-ready graphs — 62 subjects (gitignored)
│   ├── processed-old/                # GNN-ready graphs — 51 subjects (gitignored)
│   ├── splits/
│   │   ├── kfold_splits_processed_new.json   # Pre-defined k-fold splits (62 subjects)
│   │   └── kfold_splits_processed_old.json   # Pre-defined k-fold splits (51 subjects)
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

The raw `.nirs` files and pre-processed graph datasets are not included in the repository. To reproduce results you need `data/processed-new` (62 subjects, GNG task).

```bash
# Step 1 — Convert raw .nirs files to HbO/HbR/HbT CSVs
python data/processor_cli.py --help

# Step 2 — The PyG dataset is built automatically on first run
# (fNIRSGraphDataset in src/core/dataset.py caches to data/processed-new/)
```

The pre-defined cross-validation splits are committed and should be used as-is to reproduce the published numbers:

```
data/splits/kfold_splits_processed_new.json   # 62 subjects, seed 42
data/splits/kfold_splits_processed_old.json   # 51 subjects, seed 42
```

---

## Reproducing Results — Spatial Pipeline (`src/core`)

The spatial pipeline treats each fNIRS trial as a single graph and classifies it with GATv2.

**Best configuration (2026-04-29):** HBO · max\_trials=2 · 5-fold · with additional data → **Mean Acc 0.8237, Mean F1 0.8065**

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

**Hyperparameters** are from `src/core_st/experiment_config.yaml`, tuned by Optuna (200 trials on HBT, 2026-04-29, Trial #137):

| Parameter | Value |
|---|---|
| GATv2 layers | 3 |
| Filters | 96 (shared across layers) |
| Attention heads | 6 |
| FC size | 256 |
| Residual | No |
| Batch norm | Yes |
| Window size | 48 timepoints (~4.8 s at 10 Hz) |
| Window stride | 16 (~33 % overlap) |
| GRU hidden dim | 128 |
| GRU layers | 3 |
| Learning rate | 1.375e-4 (Optuna best) |

### Reproduce — 10-fold (with additional data, 62 subjects)

```bash
python -m src.core_st.main \
  --config src/core_st/experiment_config.yaml \
  --data_dir data/processed-new \
  --task GNG \
  --data_type hbt \
  --validation kfold \
  --k_folds 10 \
  --max_trials 4 \
  --epochs 100 \
  --lr 1.375e-4 \
  --batch_size 8 \
  --patience 9999 \
  --seed 42 \
  --splits_json data/splits/kfold_splits_processed_new.json \
  --save_dir research/experiments/my_run/st-kfold/10-fold
```

### Full sweep

```bash
bash scripts/run_st_kfold.sh           # all signals, 10-fold
bash scripts/run_st_kfold.sh hbt       # HBT only
```

> **Note:** Optuna was tuned on HBT. HBO and HBR runs use the same config as a starting point but have not been independently optimised.

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
