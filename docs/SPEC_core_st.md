# SPEC: Windowed Spatial-Temporal GNN for fNIRS (`src/core_st/`)

**Status:** Approved  
**Date:** 2026-04-29  
**Author:** Aunuun Jeffry Mahbuubi

---

## 1. Overview

Extend the existing fNIRS GNN pipeline with a spatial-temporal paradigm that captures temporal dynamics of fNIRS signals (currently discarded by the baseline). The new module lives in `src/core_st/` — a copy of `src/core/` with targeted modifications. The baseline in `src/core/` is untouched.

**Motivation:** `src/core/models.py:FlexibleGATNet` operates on 6 static statistical features per channel. All 326 timepoints of hemodynamic evolution are collapsed before the model sees any data. The spatial-temporal model processes the raw time series through windowed GATv2 + GRU with additive attention, enabling both temporal and spatial explainability.

---

## 2. Scope

### In Scope
- Copy `src/core/` → `src/core_st/`
- Modify `dataset.py`, `models.py`, `config.py`, `transforms.py`, `main.py`, `__init__.py`, `experiment_config.yaml` within `src/core_st/`
- Node features: 6 statistical features computed per time window (Option A)
- Temporal module: GRU with additive attention
- Spatial module: GATv2Conv (same as baseline)
- XAI: separate `model.explain(data)` method

### Out of Scope (Future Work)
- Option B: raw window signal as node features
- Transformer-based temporal encoder
- Dynamic edge construction per window
- Changes to `src/core/` (baseline)
- Changes to `training.py` (no modification needed)
- Changes to `utils.py` (no modification needed)

---

## 3. Module Structure

```
src/core_st/
├── __init__.py            ← updated exports
├── config.py              ← ExperimentConfig + new ST fields
├── dataset.py             ← fNIRSGraphDataset (x = [C, T] raw, z-scored)
├── experiment_config.yaml ← includes new ST params
├── main.py                ← updated model instantiation + new CLI args
├── models.py              ← WindowedSpatioTemporalGATNet only
├── training.py            ← copied unchanged
├── transforms.py          ← StandardizeGraphFeatures skips x, keeps edge_attr
└── utils.py               ← copied unchanged
```

---

## 4. Data Representation

### 4.1 Node Features (changed)

| | `src/core` | `src/core_st` |
|---|---|---|
| `Data.x` shape | `[23, 6]` (stats of full trial) | `[23, 326]` (raw time series, z-scored) |
| Computed in | `_build_graph()` via `compute_statistical_features()` | `_build_graph()` via per-channel z-score |

**Per-channel z-score in `_build_graph()`:**
```python
mu = trial.mean(axis=1, keepdims=True)   # [C, 1]
sigma = trial.std(axis=1, keepdims=True).clip(min=1e-8)
x = (trial - mu) / sigma                 # [C, T] — stored as Data.x
```
This normalization is self-contained per trial — no cross-sample statistics needed, eliminating any leakage risk from normalization.

### 4.2 Edge Features (unchanged)

`Data.edge_index` and `Data.edge_attr` are constructed identically to `src/core/`: Pearson correlation + coherence, filtered by `corr_threshold`. Static edges from the full trial.

### 4.3 Data Leakage Guarantee

- Subject-level splitting in `get_holdout_loaders` / `get_kfold_loaders_from_json` assigns all trials of a subject atomically to train or val (unchanged behavior).
- Windowing occurs inside `model.forward()` on each `Data.x = [C, T]` independently.
- A window is bounded within its parent trial → bounded within its parent subject → bounded within its set.
- Per-channel z-score is computed per-trial in `_build_graph()`, requiring no statistics from other samples.

---

## 5. Model: `WindowedSpatioTemporalGATNet`

### 5.1 Architecture

```
Input: batch.x [B*C, T],  batch.edge_index,  batch.edge_attr,  batch.batch

Step 1 — Windowing (inside forward):
  x.unfold(dim=1, size=W, step=stride) → [B*C, K, W]
  compute_window_stats([B*C, K, W])    → [B*C, K, 6]
  where K = floor((T - W) / stride) + 1

Step 2 — Spatial encoding per window (loop over K):
  for k in range(K):
      x_k = window_stats[:, k, :]          # [B*C, 6]
      x_k = GATv2Conv layers (shared weights across K)
      h_k = global_mean_pool(x_k, batch)   # [B, spatial_hidden]
      attn_coeffs_k stored for explain()
  → window_embeddings: [B, K, spatial_hidden]

Step 3 — Temporal encoding:
  gru_out, _ = GRU(window_embeddings)      # [B, K, gru_hidden]
  e = tanh(W_1 · gru_out)                 # additive attention energy
  alpha = softmax(W_2 · e, dim=1)         # [B, K, 1] ← stored for explain()
  context = sum(alpha * gru_out, dim=1)   # [B, gru_hidden]

Step 4 — Classification:
  logits = Linear(Dropout(context))        # [B, 2]
  return logits
```

### 5.2 Windowing Parameters

| Parameter | Default | Rationale |
|---|---|---|
| `window_size` | 32 | ~3.2s at 10Hz — matches hemodynamic response timescale |
| `window_stride` | 16 | 50% overlap — K ≈ 19 windows from 326 timepoints |
| K (computed) | `floor((326 - 32) / 16) + 1 = 19` | 19 temporal snapshots |

### 5.3 Constructor Signature

```python
class WindowedSpatioTemporalGATNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 23,       # number of fNIRS channels (nodes)
        in_channels: int = 6,       # node features per window (stats)
        edge_dim: int = 2,          # edge feature dim (corr + coherence)
        window_size: int = 32,
        window_stride: int = 16,
        n_layers: int = 2,          # GATv2 layers
        n_filters: int = 64,        # GATv2 hidden dim
        heads: int = 4,             # GATv2 attention heads
        temporal_hidden: int = 64,  # GRU hidden dim
        temporal_layers: int = 1,   # GRU layers
        fc_size: int = 64,          # classifier hidden dim
        dropout: float = 0.5,
        n_classes: int = 2,
        use_residual: bool = True,
        use_norm: bool = False,
        norm_type: str = "batch",
    )
```

### 5.4 XAI: `explain()` Method

```python
def explain(self, data: Data, device: torch.device) -> dict:
    """
    Returns attention weights for post-hoc XAI analysis.
    Call after training — do not call during training loop.

    Returns:
        {
          "temporal_attention": Tensor [K]       # importance weight per time window
          "spatial_attention":  List[Tensor[E]]  # GATv2 edge attn coeffs per window k
          "window_size":        int
          "window_stride":      int
          "n_windows":          int (K)
        }
    """
```

The `explain()` method runs a single-sample forward pass in `eval()` mode with `torch.no_grad()` and returns stored attention tensors. It is not invoked during `train_epoch()` or `evaluate()` in `training.py`.

---

## 6. Config Changes (`config.py`)

New fields added to `ExperimentConfig`:

```python
# Spatial-Temporal (new in core_st)
window_size: int = 32
window_stride: int = 16
temporal_hidden: int = 64
temporal_layers: int = 1
```

Field **removed** from `src/core_st/config.py` (not applicable to new model):
- `use_gine_first_layer`

All other fields retained for compatibility (training, augmentation, validation options unchanged).

---

## 7. Transform Changes (`transforms.py`)

`StandardizeGraphFeatures.forward()` modified:

```python
def forward(self, data: Data) -> Data:
    data = data.clone()
    # x is [C, T] — already z-scored per-channel per-trial in _build_graph()
    # No x normalization applied here.
    if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
        data.edge_attr = (data.edge_attr - self.mean_ea) / (self.std_ea + self.eps)
    return data
```

`compute_stats()` in `dataset.py` returns only `mean_ea` / `std_ea` (not `mean_x` / `std_x`).

Augmentation transforms retained:
- `DropoutEdgeAugmentation` — unchanged
- `MaskFeatureAugmentation` — operates on `[C, T]`; semantics shift to masking timepoint values
- `RandomWalkPEAugmentation` — unchanged

---

## 8. CLI Changes (`main.py`)

New arguments:

```
--window_size       int   default=32
--window_stride     int   default=16
--temporal_hidden   int   default=64
--temporal_layers   int   default=1
```

Experiment name prefix updated to `ST_GATv2_` for result directory identification.

Model instantiation updated from `FlexibleGATNet` → `WindowedSpatioTemporalGATNet`.

---

## 9. What Does NOT Change

| Component | Reason |
|---|---|
| `training.py` | `model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)` signature identical |
| `utils.py` | Pearson correlation + coherence still used for edge construction |
| Subject-level split logic | All loader functions copied unchanged |
| `src/core/` | Baseline completely untouched |

---

## 10. Future Work (Explicitly Deferred)

- **Option B**: Use raw window signal `[C, W]` as node features (no stat compression)
- **Transformer temporal encoder**: Multi-head self-attention over windows instead of GRU
- **Dynamic edges**: Recompute correlation edges per window (rather than static from full trial)
- **Multi-signal fusion**: Combine HbO and HbR in a single graph
