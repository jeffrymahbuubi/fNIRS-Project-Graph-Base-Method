# PAPER_ARCH_TABLES — Architecture and training-hyperparameter tables

> Drop-in tables for §2.3.3 of `PAPER_OUTLINE.md`. **Verified against the
> running code** (`src/core/models.py:FlexibleGATNet`,
> `src/core_st/models.py:WindowedSpatioTemporalGATNet`) and the paper-headline
> `config.yaml` files. Lift verbatim into the manuscript.
>
> **Headline-config provenance**
> - SG: `experiments/spatial_graph/5-fold/GATv2_GNG_hbo_kfold_mt4_noaug_20260507/config.yaml`
> - ST: `experiments/spatial_temporal_graph/5-fold/ST_GATv2_GNG_hbo_kfold_mt4_noaug_20260509/config.yaml`
>
> Both configs are reused (without architectural changes) across all
> chromophores (HbO/HbR/HbT) and all CV regimes (5-fold / 10-fold / LOSO);
> only the dataset and split file differ.

---

## Table A — Spatial Graph (SG): `FlexibleGATNet`

**Convention.** N = batch nodes (= B × 23). [shape] columns are post-layer.
Equation IDs reference `PAPER_MATH.md`. "—" = not applicable / Identity.

| # | Stage | Operation | Configuration | Output shape | Activation / Reg |
|---|---|---|---|---|---|
| 1 | Input | Graph $G=(V,E)$ | $\|V\|=23$, $\mathbf{x}_i \in \mathbb{R}^6$ (E5), $\mathbf{e}_{ij}\in\mathbb{R}^2$ (E4); directed; self-loops on | $\mathbf{X}\in\mathbb{R}^{N\times 6}$, $\mathbf{e}_{\text{attr}}\in\mathbb{R}^{\|E\|\times 2}$ | per-fold standardised (E10) |
| 2 | Spatial-1 | `GATv2Conv` (E7) | $d_{\text{out}}=112$, heads $H=8$, `concat=True`, `edge_dim=2`, `dropout=0.3` | $\mathbb{R}^{N\times 896}$ | residual `Linear(6→896)`, ELU, Dropout 0.3 |
| 3 | Spatial-2 | `GATv2Conv` (E7) | $d_{\text{out}}=80$, $H=6$, `concat=True`, `edge_dim=2`, `dropout=0.3` | $\mathbb{R}^{N\times 480}$ | residual `Linear(896→480)`, ELU, Dropout 0.3 |
| 4 | Pre-pool | `Linear` (per-node) | $480 \to 224$ | $\mathbb{R}^{N\times 224}$ | ELU |
| 5 | Pool | `global_mean_pool` over 23 nodes | reduces along node axis per graph | $\mathbb{R}^{B\times 224}$ | Dropout 0.3 |
| 6 | Classifier | `Linear` | $224 \to 2$ | $\mathbb{R}^{B\times 2}$ logits | softmax → CE loss (E9) |

**Subtleties to preserve when redrawing in PowerPoint:**
- **Per-node `Linear(480, 224)` is applied *before* global-mean-pool**, not after. Most pipeline diagrams (including our Fig 2 inspiration figure) draw it the other way — that is wrong. Source: `src/core/models.py:101–102` (`F.elu(self.pre_pool(x))` → `global_mean_pool(x, batch)`).
- `BatchNorm` is **disabled** for SG (`use_norm=False` in the headline config). The Identity placeholder is intentional.
- Two distinct residual projections (rows 2, 3) are used because $d_{\text{in}}\ne d_{\text{out}}$ at both layers; both are bias-free `Linear` mappings (`src/core/models.py:76–80`).
- A `GINEConv` first-layer variant exists in the code (`use_gine_first_layer=True`) but is **not used** in the headline runs — only ablation.

**Parameter count (SG headline):**
- GATv2Conv-1: ≈ 14.4k  ·  Residual Linear(6→896): 5,376  ·  GATv2Conv-2: ≈ 0.6M  ·  Residual Linear(896→480): 0.43M  ·  Pre-pool Linear(480→224): 107.7k  ·  Classifier Linear(224→2): 450  ·  **Total ≈ 1.16 M trainable parameters.**

---

## Table B — Spatio-Temporal Graph (ST): `WindowedSpatioTemporalGATNet`

**Convention.** Symbols as Table A; $K=\lfloor(T-W)/S\rfloor+1$ is the number of windows. **For the paper-headline config: $T=326$, $W=16$, $S=8$ → $K=39$.**

| # | Stage | Operation | Configuration | Output shape | Activation / Reg |
|---|---|---|---|---|---|
| 1 | Input | Per-trial z-scored time series | $\mathbf{X}^{\text{ST}}\in\mathbb{R}^{N\times T}$ (E6); $\mathbf{e}_{\text{attr}}$ per-fold standardised (E10) | $\mathbb{R}^{N\times 326}$, $\mathbb{R}^{\|E\|\times 2}$ | — |
| 2 | Windowing | `tensor.unfold(time, W, S)` | $W=16$, $S=8$ | $\mathbb{R}^{N\times K\times 16}$ | — |
| 3 | Window-stats | `_compute_window_stats` (E6) | per-(node, window) → 6 stats | $\mathbb{R}^{N\times K\times 6}$ | `nan_to_num(0)` for degenerate windows |
| 4 | Spatial-1 ⊛ | `GATv2Conv` shared over $k=1..K$ (E7) | $d_{\text{out}}=80$, $H=2$, `concat=True`, `edge_dim=2`, `dropout=0.3` | $\mathbb{R}^{N\times K\times 160}$ | `BatchNorm(160)`, ELU, Dropout; **no residual** |
| 5 | Spatial-2 ⊛ | `GATv2Conv` shared (E7) | $d_{\text{out}}=80$, $H=2$, `concat=True` | $\mathbb{R}^{N\times K\times 160}$ | `BatchNorm(160)`, ELU, Dropout; **no residual** |
| 6 | Per-window pool | `global_mean_pool` over 23 nodes (per window) | reduces along node axis | $\mathbb{R}^{B\times K\times 160}$ | — |
| 7 | Pre-GRU | `Linear` | $160 \to 192$ | $\mathbb{R}^{B\times K\times 192}$ | ELU |
| 8 | Temporal | `GRU` (E8) | $d_h=192$, layers=1, `batch_first=True` | $\mathbb{R}^{B\times K\times 192}$ | (no inter-layer dropout — single layer) |
| 9 | Temporal attention | additive over $K$ windows (E8) | $\mathbf{W}_v$, $\mathbf{u}_v$ over $\mathbb{R}^{192}$ | $\mathbb{R}^{B\times 192}$ | softmax over $K$ |
| 10 | Pre-classifier | `Linear` | $192 \to 256$ | $\mathbb{R}^{B\times 256}$ | Dropout 0.3 (before this), ELU (after) |
| 11 | Classifier | `Linear` | $256 \to 2$ | $\mathbb{R}^{B\times 2}$ logits | softmax → CE loss (E9) |

**⊛** indicates **shared weights across all $K=39$ windows** — this is the central architectural choice that makes the model parameter-efficient compared to a 39-stack of independent GATv2 networks.

**Subtleties to preserve when redrawing:**
- The **window-stat module is inside the model's `forward()`**, not part of the dataset preprocessing. This means $\texttt{Data.x}$ stored on disk is the raw `[23, 326]` z-scored time series — the `[23, K, 6]` representation only exists at inference/training time.
- `in_channels=6` is **fixed**; it is *not* an Optuna search parameter (`src/core_st/optuna_search.py`). Window size and stride change `K`, but the per-window feature dimension is invariant.
- `use_residual=False` and `use_norm=True` (`norm_type=batch`) for ST headline — opposite of SG headline.
- Temporal attention weights $\boldsymbol{\alpha}\in\mathbb{R}^{K}$ are exposed by `model.explain()` and are the **primary native-attention XAI path** (cf. `docs/SPEC_xai_graph.md` §3.2 rev. 6).

**Parameter count (ST headline):**
- GATv2Conv-1 (in=6 → 80×2 heads): ≈ 1.4k  ·  GATv2Conv-2 (in=160 → 80×2): ≈ 26k  ·  BatchNorm×2: 640  ·  Pre-GRU Linear(160→192): 30.9k  ·  GRU(192→192): 222.7k  ·  Attention $\mathbf{W}_v(192,192)$ + $\mathbf{u}_v(192,1)$: 37.2k  ·  Pre-cls Linear(192→256): 49.4k  ·  Classifier Linear(256→2): 514  ·  **Total ≈ 0.37 M trainable parameters** (≈ 3× smaller than SG despite the temporal axis, because spatial GATv2 weights are shared across $K=39$ windows).

---

## Table C — Training hyperparameters (paper-headline runs)

Single table for both architectures; differences highlighted.

| Hyperparameter | SG (`FlexibleGATNet`) | ST (`WindowedSpatioTemporalGATNet`) | Source |
|---|---|---|---|
| Optimizer | Adam | Adam | code default |
| Initial learning rate | **6.79 × 10⁻³** | **3.04 × 10⁻⁴** | Optuna best (P0.8 / `result_report.md`) |
| LR scheduler | `cosine_warmup` (linear warmup → cosine decay) | `CosineAnnealingLR`, $T_{\max}=150$, $\eta_{\min}=10^{-5}$ | config.yaml |
| Batch size | 8 | 8 | config.yaml |
| Epochs | 150 | 150 | config.yaml |
| Early-stopping patience | 9999 (effectively off — full epoch run) | 30 | config.yaml |
| Loss | Cross-entropy (E9), no class weights, no focal | same | config.yaml |
| Random seed | 42 | 42 | config.yaml |
| `dropout` | 0.3 | 0.3 | config.yaml |
| `use_residual` | True | False | config.yaml |
| `use_norm` | False | True (`batch`) | config.yaml |
| Augmentation | none (`augment=false`, `_noaug_*` runs) | none | config.yaml |
| Per-fold standardisation (E10) | x and edge_attr | edge_attr only (x is pre-z-scored, E6) | `dataset.py:139` |
| Trial cap per subject (`max_trials`) | 4 | 4 | config.yaml |
| Pre-processing pipeline | `processed-new-mc` (Wavelet+CBSI+bandpass) | `processed-new-mc` | config.yaml |
| Splits | `data/splits/kfold_splits_processed_new_mc.json` (kfold); LOSO derived in-code | same | config.yaml |

**Two seeds-of-confusion to flag for the SPEC plan:**
1. SG `patience=9999` was a deliberate decision after the leak-free patience study (`research/experiments/20260506/leak-free-patience-9999/`). Do NOT replace with a smaller patience without re-running the study.
2. SG `lr=6.79e-3` is **two orders of magnitude larger** than ST `lr=3.04e-4`. This is not a typo — it is what each Optuna study independently converged to and reflects the differing model size + scheduler (cosine_warmup vs CosineAnnealingLR).

---

## Table D — Search-space summary (Optuna; for §2.3.4)

This table summarises the *search space*, not the *best values* — the latter live in §2.3.4.3 / Table C above.

| Group | Parameter | SG search range | ST search range |
|---|---|---|---|
| Spatial | `n_layers` | {1, 2, 3} | {1, 2, 3} |
| Spatial | `n_filters` per layer | {16…128} step 16 (per-layer list) | {16…128} step 16 (single int, shared across layers) |
| Spatial | `heads` per layer | {2, 4, 6, 8} (per-layer list) | {2, 4, 6, 8} (single int) |
| Regularisation | `dropout` | {0.1…0.5} step 0.1 | {0.1…0.5} step 0.1 |
| Regularisation | `use_residual` | {True, False} | {True, False} |
| Regularisation | `use_norm`, `norm_type` | {True, False} × {batch, layer} | same |
| Classifier | `fc_size` | {32…256} step 32 | {32…256} step 32 |
| Optimizer | `learning_rate` | log-uniform [1e-5, 1e-1] | log-uniform [1e-5, 1e-1] |
| Windowing | `window_size` | n/a (no temporal axis) | {16, 32, 48, 64} step 16 |
| Windowing | `window_stride` | n/a | {8, 16, 24, 32} step 8; clamped ≤ `window_size` |
| Temporal | `temporal_hidden` | n/a | {32…256} step 32 |
| Temporal | `temporal_layers` | n/a | {1, 2, 3} |
| Scheduler (focused sub-study) | `T_max`, `eta_min` | n/a (cosine_warmup fixed) | {50, 100, 150, 200}, {0.0, 1e-6, 1e-5} |

**Source:** `src/core/optuna_search.py` (SG full sweep) and `src/core_st/optuna_search.py` (ST full sweep) + `src/core_st/optuna_search_lr_cosine.py` (focused LR/cosine sub-study that produced the 20260503 best-trial used in headline config).

---

## What the SPEC plan should do with this file

1. **Tables A and B** become **Methods Tables 2 and 3** in the manuscript (positioned in §2.3.3).
2. **Table C** becomes **Methods Table 4** (positioned at the end of §2.3.3 or beginning of §2.3.5).
3. **Table D** is a candidate for the **Supplementary Materials** rather than the main text (search-space documentation is rarely required in IEEE TBME / Frontiers main text but is mandatory for reproducibility supplementary).
4. The "Subtleties to preserve" boxes under each table should be flagged in the SPEC plan as **review-defensive caveats**: they are the exact details that a careful reviewer will catch if the figure is sloppy. They should appear as foot-of-table notes in the published manuscript.
