# SPEC — CORAL Temporal Module Search for fNIRS ST-GNN

**Date:** 2026-05-02  
**Status:** Draft — awaiting implementation  
**Predecessor:** CORAL `coral/fnirs_gat/` (spatial-only GNN, 106 evals, best holdout F1=0.8696 / 5-fold F1=0.7792)

---

## 1. Objective

Determine whether an alternative temporal module can outperform the current GRU in the ST-GNN (`WindowedSpatioTemporalGATNet`) for fNIRS anxiety vs healthy classification.

The spatial encoder (GATv2), training hyperparameters, and windowing configuration are locked at their Optuna-validated best values. Only the temporal module architecture (and its specific hyperparameters) is subject to exploration.

**Hypothesis:** GRU was never compared against alternatives during hyperparameter search. LSTM, BiGRU, a Transformer encoder, or a TCN may extract richer sequential patterns from the K=16 window embeddings.

---

## 2. Baseline

| Metric | Value |
|---|---|
| Model | ST-GNN with GRU (Optuna Trial #91) |
| Validation | 5-fold CV, HBO, mt4 (window_size=16, window_stride=8) |
| Mean Acc | 0.7577 |
| **Mean F1** | **0.7792** ← CORAL target to beat |
| Mean Sens | 0.8967 |
| Mean Spec | 0.6452 |
| Data | `data/processed-new-mc`, 62 subjects (29 anxiety / 33 healthy) |
| Splits | `data/splits/kfold_splits_processed_new_mc.json` → key `kfold_5` |

---

## 3. Fixed Parameters (agents must NOT change these)

### 3.1 Spatial Encoder — Optuna best (Trial #91)

| Param | Value |
|---|---|
| `n_layers` | 2 |
| `n_filters` | 80 |
| `heads` | 2 |
| `use_residual` | False |
| `use_norm` | True |
| `norm_type` | `"batch"` |

### 3.2 Windowing — mt4 config

| Param | Value |
|---|---|
| `window_size` | 16 |
| `window_stride` | 8 |

### 3.3 Dataset

| Param | Value |
|---|---|
| `data_type` | `"hbo"` |
| `max_trials` | 4 (mt4) |
| `task_type` | `"GNG"` |
| `directed` | True |
| `self_loops` | True |
| `corr_threshold` | 0.1 |

### 3.4 Training loop

| Param | Value |
|---|---|
| `epochs` | 100 |
| `patience` | 20 |
| `batch_size` | 8 |
| `optimizer` | Adam |
| `scheduler` | ReduceLROnPlateau (mode=max, factor=0.5, patience=3) |

*Note: `lr` floats — see search space below.*

---

## 4. Search Space (agents explore these)

### 4.1 Primary — Temporal module type

| Param | Type | Values |
|---|---|---|
| `temporal_type` | categorical | `"gru"` (baseline), `"lstm"`, `"bigru"`, `"transformer"`, `"tcn"` |

### 4.2 Universal temporal hyperparameters

| Param | Type | Range |
|---|---|---|
| `temporal_hidden` | int | [32, 256, step=32] |
| `temporal_layers` | int | [1, 3] |

### 4.3 Module-specific hyperparameters (conditional on temporal_type)

| Module | Additional params |
|---|---|
| `transformer` | `transformer_heads` ∈ {2, 4, 8}; `ffn_ratio` ∈ {2, 4} |
| `tcn` | `tcn_kernel_size` ∈ {3, 5, 7}; `tcn_dilation_base` ∈ {1, 2} |
| `gru`, `lstm`, `bigru` | none (temporal_hidden and temporal_layers suffice) |

### 4.4 Shared tunable (may adjust per temporal type)

| Param | Type | Range |
|---|---|---|
| `dropout` | float | [0.1, 0.5, step=0.1] |
| `fc_size` | int | [32, 256, step=32] |
| `learning_rate` | float log | [1e-5, 1e-1] |

---

## 5. Temporal Module Specifications

All five modules receive the same input sequence: `seq ∈ [B, K, temporal_hidden]`, where K ≈ 16 windows (given window_size=16, window_stride=8, T≈326 time steps → K = (326−16)//8 + 1 = 39 windows). Output must be a context vector `[B, temporal_hidden]` before the classifier head.

### 5.1 GRU (baseline — existing implementation)

```
GRU(input_size=temporal_hidden, hidden_size=temporal_hidden, num_layers=temporal_layers)
→ gru_out [B, K, temporal_hidden]
→ additive attention (attn_v + attn_u) → context [B, temporal_hidden]
```

### 5.2 LSTM

```
LSTM(input_size=temporal_hidden, hidden_size=temporal_hidden, num_layers=temporal_layers)
→ lstm_out [B, K, temporal_hidden]
→ additive attention → context [B, temporal_hidden]
```
Same interface as GRU; add cell state. Expected to be a marginal improvement or parity.

### 5.3 BiGRU (Bidirectional GRU)

```
GRU(input_size=temporal_hidden, hidden_size=temporal_hidden//2, bidirectional=True, ...)
→ bigru_out [B, K, temporal_hidden]  (concat fwd+bwd = temporal_hidden)
→ additive attention → context [B, temporal_hidden]
```
*Implementation note:* `hidden_size` must be `temporal_hidden // 2` so the concatenated output stays at `temporal_hidden`. If `temporal_hidden` is odd, clamp to nearest even.

### 5.4 Transformer Encoder

```
PositionalEncoding(K, temporal_hidden)  # sinusoidal, learned, or no encoding
→ TransformerEncoder(
    d_model=temporal_hidden,
    nhead=transformer_heads,
    num_encoder_layers=temporal_layers,
    dim_feedforward=temporal_hidden * ffn_ratio,
    dropout=dropout,
    batch_first=True
  )
→ transformer_out [B, K, temporal_hidden]
→ mean pool (or CLS token) → context [B, temporal_hidden]
```
*Implementation notes:*
- `temporal_hidden` must be divisible by `transformer_heads`. Enforce by: if `temporal_hidden % transformer_heads != 0`, round `temporal_hidden` up to next multiple.
- For small K (≈ 39), positional encoding is important — start with sinusoidal, then optionally try learned.
- The fNIRS window sequence is short (K ≈ 39) — Transformer should work well without causal masking.

### 5.5 TCN (Temporal Convolutional Network)

```
# Stack of 1-D dilated causal convolutions:
for i in range(temporal_layers):
    dilation = tcn_dilation_base ** i
    Conv1d(
        in_channels=temporal_hidden,
        out_channels=temporal_hidden,
        kernel_size=tcn_kernel_size,
        dilation=dilation,
        padding=(tcn_kernel_size-1)*dilation  # causal padding
    )
    → BatchNorm1d + ReLU + Dropout
    → residual connection (identity if same dim)
→ tcn_out [B, temporal_hidden, K]
→ transpose → [B, K, temporal_hidden]
→ last time step (tcn_out[:, -1, :]) OR global mean → context [B, temporal_hidden]
```
*Implementation notes:*
- Input to Conv1d must be `[B, temporal_hidden, K]` (channels-first). Transpose `seq` before the first conv.
- Causal padding removes future leakage: `left_pad = (kernel_size - 1) * dilation`, `right_pad = 0`.
- Use last time step for the context vector (causal) OR global mean (pooling).

---

## 6. Validation Protocol

```
run(data_dir, splits_json) → float (mean 5-fold F1)

1. Load fNIRSGraphDataset(data_dir, task_type="GNG", data_type="hbo", max_trials=4, ...)
2. compute_stats() for feature normalization
3. get_kfold_loaders_from_json(dataset, splits_json=splits_json, n_splits=5, ...)
4. For each fold: fresh model → train 100 epochs, patience=20 → evaluate → collect F1
5. Return mean(fold_f1_scores)
```

**Grader direction:** maximize  
**Grader timeout:** 600s  
**Metric:** mean 5-fold validation F1

---

## 7. Success Criteria

| Level | Criterion |
|---|---|
| **Improvement** | Mean 5-fold F1 > 0.7792 on HBO mt4 |
| **Meaningful improvement** | ΔF1 > +2pp (> 0.7992) |
| **Strong improvement** | ΔF1 > +5pp (> 0.8292) |
| **Secondary validation** | Improvement also holds for HBT (cross-signal) |

A result is considered validated only if the improvement holds with at least 2 different `temporal_hidden` values (ruling out lucky single-config finds).

---

## 8. CORAL File Structure

```
coral/fnirs_st/
├── task.yaml                  ← CORAL task definition
└── seed/
    ├── solution.py            ← run(data_dir, splits_json) -> float
    └── core/
        ├── __init__.py
        ├── dataset.py         ← copy of src/core_st/dataset.py (no changes needed)
        ├── models.py          ← modified: TemporalModule factory + all 5 implementations
        ├── training.py        ← copy of src/core_st/training.py (no changes needed)
        ├── transforms.py      ← copy of src/core_st/transforms.py (no changes needed)
        └── utils.py           ← copy of src/core_st/utils.py (no changes needed)
```

*Notes:*
- `core/` is a CORAL-local copy — agents may freely modify it without affecting `src/core_st/`
- `solution.py` is the only entry point the grader calls
- `models.py` is the primary file agents will modify — it must support all 5 temporal_type values

---

## 9. task.yaml Design

```yaml
task:
  name: fnirs-st-temporal-search
  description: |
    Find the best temporal module for WindowedSpatioTemporalGATNet on fNIRS
    anxiety vs healthy classification (GNG task, HBO signal).

    The spatial encoder is LOCKED — do NOT modify these params:
      n_layers=2, n_filters=80, heads=2
      use_residual=False, use_norm=True, norm_type="batch"

    Windowing is LOCKED at mt4:
      window_size=16, window_stride=8

    CURRENT BASELINE: 5-fold CV mean F1 = 0.7792 (GRU, Optuna Trial #91)

    PRIMARY SEARCH TARGET:
      temporal_type in {gru, lstm, bigru, transformer, tcn}

    SECONDARY (tune per temporal_type):
      temporal_hidden [32..256], temporal_layers [1..3]
      Transformer: transformer_heads {2,4,8}, ffn_ratio {2,4}
      TCN: tcn_kernel_size {3,5,7}, tcn_dilation_base {1,2}
      dropout, fc_size, learning_rate

    File layout:
      solution.py    — defines run(data_dir, splits_json) -> float (5-fold mean F1)
      core/models.py — TemporalModule factory: gru, lstm, bigru, transformer, tcn
      core/dataset.py, training.py, transforms.py, utils.py — do not modify

  tips: |
    - Metric is mean 5-fold validation F1 (higher is better)
    - Baseline F1 = 0.7792 with GRU — beat this to claim improvement
    - Spatial encoder is fixed — do not change n_layers/n_filters/heads
    - Start each new temporal_type with temporal_hidden=192 (Optuna best for GRU)
    - Transformer: temporal_hidden MUST be divisible by transformer_heads
    - BiGRU: use hidden_size=temporal_hidden//2 so concat output = temporal_hidden
    - TCN: input to Conv1d is channels-first [B, C, K]; transpose seq before first conv
    - lr=0.000635 was tuned for GRU; Transformer often needs lower lr (1e-4 to 5e-4)
    - TCN often needs higher lr than RNNs (1e-3 to 5e-3)
    - After finding the best temporal_type, verify improvement holds with 2+ different
      temporal_hidden values before reporting as a validated finding
    - Commit after each eval that beats the baseline F1

grader:
  timeout: 600
  direction: maximize
  args:
    data_dir: "/home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/data/processed-new-mc"
    splits_json: "/home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/data/splits/kfold_splits_processed_new_mc.json"
    python_executable: "/home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/src/.venv/bin/python"
    program_file: "solution.py"

agents:
  count: 3
  runtime: claude_code
  model: sonnet
  max_turns: 100
  warmstart:
    enabled: false

workspace:
  results_dir: "/home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/coral/fnirs_st/results"
  repo_path: "/home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/coral/fnirs_st/seed"

run:
  verbose: false
  ui: false
  session: tmux
```

---

## 10. Agent Exploration Strategy (recommended order)

Agents should explore in roughly this order, though they are free to deviate based on intermediate results:

1. **GRU baseline verification** — confirm `solution.py` returns ~0.7792 before any modifications
2. **LSTM** — closest to GRU; establish whether cell state adds value (expect +0 to +2pp)
3. **Transformer** — highest potential for parallel attention on K=39 windows; most tuning needed
4. **BiGRU** — bidirectionality may help since fNIRS temporal patterns are not strictly causal
5. **TCN** — lightweight, parallelizable; test with kernel_size=5 first
6. **Winner re-tuning** — take the best-performing type and sweep `temporal_hidden` + `temporal_layers`

---

## 11. Open Questions (resolve after CORAL completes)

1. Does the best temporal module also improve HBR and HBT signals, or is the gain HBO-specific?
2. Does the improvement scale with longer windows (mt2 has fewer temporal steps K ≈ 19 vs mt4 K ≈ 39)?
3. Is the additive attention readout still optimal, or does a different pooling (mean, last, CLS) work better with non-GRU modules?
4. If Transformer wins: is positional encoding necessary (ablation: no PE vs sinusoidal vs learned)?
