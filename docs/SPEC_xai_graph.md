# SPEC: Graph Explainability for fNIRS GNN (`src/core/` + `src/core_st/`)

**Status:** Draft (rev. 4)
**Date:** 2026-05-08
**Author:** Aunuun Jeffry Mahbuubi
**Targets:** `src/core/FlexibleGATNet` (Spatial Graph) and `src/core_st/WindowedSpatioTemporalGATNet` (Spatial-Temporal)
**Execution model:** **Jupyter notebook driven**. Reusable building blocks live in `src/xai/` and are imported by notebooks under `src/notebook/xai/`. No CLI entry point — all runs are notebook cells (mirrors how `src/notebook/4_non_recurrent_gnn.ipynb` and `src/notebook/statistical-analysis/*.ipynb` already work in this project).
**PyG version pin:** `torch_geometric==2.7.0` (verified 2026-05-07 against `src/requirements.txt` runtime). All API claims in §5, §6, §15 are valid for this version.

**Changelog**
- **rev. 4 (2026-05-08):** Realigned the data-path model with the actual checkpoint inventory. Findings during Phase A recon: (a) all SG runs (kfold + LOSO) and all ST runs share `data/processed-new-mc`; rev. 1–3's claim that SG used `processed-new` was wrong for current checkpoints. (b) SG kfold + ST kfold share `data/splits/kfold_splits_processed_new_mc.json`; LOSO has `splits_json: null` because LOSO splits are derived in-code from dataset subject IDs (`dataset.py:get_loso_splits`). (c) One in-scope SG checkpoint (HbO mt4 LOSO, 20260506) was trained on a cloud instance whose `config.yaml` records `/root/remote-training-setup/data/processed-new-mc`. SPEC now specifies an **auto-rebase rule** in §10.4: `data_dir` is read from the checkpoint's own `config.yaml` and rebased to `<project_root>/data/<basename>` if the literal path is unreachable, with the rewrite logged in `run.json`. (d) §9.2/§9.3 cell patterns updated: `data_dir` and `splits_json` are no longer required `XAIRunConfig` fields — they're derived from the checkpoint and made overridable. (e) The "SG-vs-ST silent leakage" risk in old §10.4 is removed; SG and ST share data and (for kfold) splits, so the only validation needed is the rebase + the standard fold-leak-free `compute_stats(train_indices)` pattern. C1 acceptance criterion clarified: target is the final-scalar `f1_score` field of `*_fold_F.pkl` (not `max(val_f1)`), to within ±0.005.
- rev. 3 (2026-05-07): Added §15 PyG API Reference appendix. APIs cross-checked against installed `torch_geometric==2.7.0` via `mcp__context7__*` and `inspect.signature`. Notable upgrades discovered during the doc pass: (a) PyG ships a built-in `AttentionExplainer` that auto-aggregates GAT/GATv2 attention across layers/heads — added as a candidate cross-check for ST §6.4 and as a fast alternative for SG §5.3; (b) `Explainer.__call__` signature confirmed as `(x, edge_index, *, target=None, index=None, **kwargs)` — `edge_attr` and `batch` go through `**kwargs`; (c) `ThresholdConfig(threshold_type='topk', value=K)` can replace manual top-K post-processing on edge masks. No §1–§14 semantics changed; §5.3, §6.4, §11 references the new appendix.
- rev. 2 (2026-05-07): Notebook-driven execution; population-level aggregation; correctly-classified-only filter; SG-vs-ST diff figure; HbR/HbT and Brodmann mapping deferred.
- rev. 1 (2026-05-07): Initial feasibility verdict.

---

## 1. Overview & Feasibility Verdict

**Verdict: Feasible for both architectures.** Both models expose the canonical PyG forward signature `forward(x, edge_index, edge_attr, batch)` required by `torch_geometric.explain.Explainer`, and both training pipelines persist bare `state_dict` checkpoints that can be reloaded directly. In addition, `WindowedSpatioTemporalGATNet` already ships a built-in `model.explain()` that exposes per-layer GATv2 edge-attention and per-window temporal attention — no extra training is needed for ST attention-based XAI.

**Research goal driving this SPEC:**
1. Identify which fNIRS channels (of the 23-channel prefrontal montage) contribute most to the Healthy-vs-GAD prediction.
2. Identify which channel-to-channel connections (23 × 23) carry the strongest discriminative signal.
3. Map the resulting maps onto the prefrontal montage (5 × 7 grid) so they can be cross-referenced with the statistical-analysis findings (`src/notebook/statistical-analysis/02_brain_activation`, `03_hb_type_comparison`) and with the GAD-PFC literature.

**Two architectures, complementary explanations:**

| Aspect | `src/core/` (Spatial Graph, SG) | `src/core_st/` (Spatial-Temporal, ST) |
|---|---|---|
| `Data.x` | `[23, 6]` static channel statistics | `[23, T≈326]` raw z-scored time series |
| Forward path | one GATv2 stack on full-trial features | window→stats→GATv2 (shared)→GRU→additive attention |
| Native introspection | none | `model.explain()` returns spatial + temporal attention |
| GNNExplainer node mask shape | `[23, 6]` (channel × feature) | `[23, T]` (channel × time-step) — too high-dim to be paper-interpretable directly |
| Edge mask | `[E]` per pair-direction | `[E]` per pair-direction (one mask per call) |

ST provides **both** temporal localisation (which seconds matter) and spatial graph attention (which channels exchange information). SG cannot localise time; it answers only "which channel × which statistic" and "which connection".

---

## 2. Scope

### 2.1 In Scope (per user clarification, 2026-05-07)
- **Chromophore:** HbO only (the primary chromophore per project memory).
- **Architectures:** both `src/core/` SG and `src/core_st/` ST.
- **CV regimes:** all of `kfold-5`, `kfold-10`, `loso`.
  - SG checkpoints live under `research/experiments/20260506/leak-free-patience-9999/spatial-graph/{kfold/5-fold,kfold/10-fold,loso}/...`
  - ST checkpoints live under `research/experiments/20260501/spatial_temporal_graph/{5-fold,10-fold,loso}/...`
- **Trial counts:** mt=2 and mt=4.
- **Deliverable level:** **population-level aggregation** — one ranked channel list and one 23×23 channel-pair matrix per architecture × CV regime × mt.

### 2.2 Out of Scope
- **HbR / HbT XAI — explicit future work.** Rerun the validated HbO notebook with `hb='hbr'|'hbt'` only after the HbO building blocks have been validated and accepted (per user feedback 2026-05-07).
- **MNE / AtlasViewer / Brodmann mapping — explicit future work.** Map the montage to MNI coordinates and label Brodmann areas only after the importance-map pipeline is fully working (per user feedback 2026-05-07; was §14.5).
- LOSO subject-level case studies (defer; a per-subject debugging notebook can reuse the same explainer code).
- Re-training any model — XAI consumes only existing checkpoints.
- Explanations for the GINEConv first-layer variants of SG (only used in early experiments); current 20260506 leak-free runs are pure GATv2.
- **No CLI / `__main__` / argparse plumbing in this iteration.** All execution is via Jupyter notebooks. A CLI wrapper can be added later if needed for batch reproducibility.

---

## 3. Architectural Compatibility Analysis

### 3.1 SG — `FlexibleGATNet` (`src/core/models.py`)
- `forward(x, edge_index, edge_attr, batch) → logits[B, 2]` ✅ matches `Explainer` signature.
- Uses `GATv2Conv(edge_dim=2)` and `GINEConv(edge_dim=2)` (optional). PyG `GNNExplainer` propagates an `edge_mask` automatically through any `MessagePassing` layer that supports edge weights, which both layer types do.
- **Wrapper required:** `Explainer` with `model_config.return_type='probs'` expects probabilities. We wrap as `nn.Module` returning `softmax(logits, dim=-1)` and use `mode='multiclass_classification'` — cleaner than the notebook §6 sigmoid-on-2-logits pattern, which is mathematically inconsistent for a 2-output softmax head.

### 3.2 ST — `WindowedSpatioTemporalGATNet` (`src/core_st/models.py`)
- `forward(x, edge_index, edge_attr, batch) → logits[B, 2]` ✅ also matches.
- **Internal windowing complicates raw GNNExplainer:** `Data.x = [23, T]` — GNNExplainer with `node_mask_type='attributes'` would learn a `[23, T]` mask. Each timestep is treated as an independent feature. For T=326, this is ~7,500 dims of importance — interpretable as a per-channel time-localised heatmap, but heavy and harder to defend in a paper.
- **Native attention is the better primary path.** `model.explain(data, device)` (already implemented) returns:
  - `temporal_attention: Tensor[K]` — softmax weight per window.
  - `spatial_attention: List[List[Tensor[E, heads]]]` — GATv2 attention per window k per layer l.
  - `n_windows: int (K)`, `window_size`, `window_stride`.
- **Recommended ST pipeline:** native attention as primary; optionally a supplementary GNNExplainer pass with `node_mask_type='object'` (one scalar per channel) and `edge_mask_type='object'` to cross-check spatial attention rankings without interpreting [N, T] masks.

### 3.3 Checkpoint format
- `training.py` saves bare state-dict via `torch.save(model.state_dict(), …)`.
- File naming (verified from listings):
  - SG kfold: `GATv2_GNG_hbo_kfold_mt{N}_noaug_{DATE}_fold_{F}.pt`
  - SG LOSO: `GATv2_GNG_hbo_loso_mt{N}_noaug_{DATE}_subj_{SID}.pt`
  - ST kfold: `ST_GATv2_GNG_hbo_kfold_mt{N}_noaug_{DATE}_fold_{F}.pt`
  - ST LOSO: `ST_GATv2_GNG_hbo_loso_mt{N}_noaug_{DATE}_subj_{SID}.pt`
- Each experiment directory ships a `config.yaml` containing the full hyperparameter set needed to reconstruct the model class — no separate hyperparameter store required.

---

## 4. Module Layout

XAI is delivered as **two layers**:

1. **Reusable Python module `src/xai/`** — pure-Python building blocks that can be imported from any notebook or future CLI. This is the codebase the notebooks consume.
2. **Notebook driver `src/notebook/xai/`** — one notebook per experimental dimension; cells call into `src/xai.*` and produce the figures/tables for the paper.

```
src/xai/                              ← REUSABLE BUILDING BLOCKS (importable)
├── __init__.py                       # re-export top-level API: load_checkpoint, explain_sg_graph, explain_st_graph, aggregate_population, plot_montage, plot_pair_matrix, ...
├── channels.py                       # CHANNEL_NAMES (23) + GRID_POS (5×7) — single source of truth
├── checkpoints.py                    # discover_checkpoints(experiment_root), load_checkpoint(info) → (model, dataset, val_indices, val_transform, fold_train_indices)
├── sg_explainer.py                   # explain_sg_graph(model, data) → SGTrialExplanation;  explain_sg_subject(...) → SubjectExplanation
├── st_explainer.py                   # explain_st_graph(model, data) → STTrialExplanation (native attention);  explain_st_with_gnnexplainer(...) — supplementary
├── aggregate.py                      # aggregate_population(per_subject_explanations, only_correct=True) → PopulationResult
├── visualize.py                      # plot_montage_channel_importance, plot_pair_matrix, plot_top_pairs_chord, plot_temporal_attention, plot_sg_vs_st_scatter, plot_pair_matrix_diff
├── io.py                             # save_population_result(result, out_dir) — writes the CSVs / NPY / run.json from §7.3
└── config.py                         # XAIRunConfig dataclass — passed in directly from notebook (no YAML required)
```

```
src/notebook/xai/                     ← NOTEBOOK DRIVERS (visualisation + reporting)
├── 00_setup_and_smoke.ipynb          # env check, single-checkpoint smoke test (one fold, one trial), confirms blocks import cleanly
├── 01_sg_population.ipynb            # SG × HbO × {kfold-5, kfold-10, loso} × {mt2, mt4} — 6 runs, all figures + CSVs
├── 02_st_population.ipynb            # ST × HbO × {kfold-5, kfold-10, loso} × {mt2, mt4} — 6 runs, all figures + CSVs (includes temporal attention)
├── 03_cross_arch_comparison.ipynb    # SG vs ST: ρ, Jaccard, scatter, pair-matrix diff
└── README.md                         # how to run the notebooks (kernel, paths, runtime expectations)
```

The notebooks are **thin** — they handle paths/parameters, call `src/xai.*`, and lay out figures. All numerical work and per-trial logic lives inside `src/xai/`. This keeps the building blocks unit-testable and reusable while the notebooks remain the human-facing artifact for the paper.

The canonical channel list and grid layout (already used in `src/notebook/statistical-analysis/04_severity_correlation/`):
```python
CHANNEL_NAMES = [
    'S1_D1','S1_D3','S2_D2','S2_D1','S2_D5','S3_D1','S3_D3','S3_D4','S3_D6',
    'S4_D4','S4_D5','S4_D7','S5_D2','S5_D5','S5_D8','S6_D3','S6_D6',
    'S7_D4','S7_D6','S7_D7','S8_D5','S8_D7','S8_D8',
]
GRID_POS = [(0,2),(1,1),(0,4),(0,3),(1,4),(1,2),(2,1),(2,2),(3,1),
            (2,3),(2,4),(3,4),(1,5),(2,5),(3,6),(3,0),(4,1),
            (3,2),(4,2),(4,3),(3,5),(4,4),(4,5)]
GRID_SHAPE = (5, 7)
```
**Reuse exactly this list** so XAI heatmaps line up channel-for-channel with the statistical-analysis figures.

Output directory:
```
research/xai/
├── sg/
│   ├── kfold-5/mt2/{node_importance.csv, edge_importance.csv, channel_pair_matrix.npy, run.json, plots/...}
│   ├── kfold-10/mt2/...
│   ├── loso/mt2/...
│   └── kfold-5/mt4/...   (etc.)
└── st/
    ├── kfold-5/mt2/{spatial_attention.csv, temporal_attention.csv, channel_pair_matrix.npy, run.json, plots/...}
    └── ...
```

---

## 5. SG Explainer Spec — `src/xai/sg_explainer.py`

### 5.1 Inputs
- `checkpoint_path: Path` to one `*.pt` state-dict file.
- `config: dict` parsed from the experiment's `config.yaml` (gives `n_layers`, `n_filters`, `heads`, `fc_size`, `dropout`, `use_residual`, `use_norm`, `norm_type`, `use_gine_first_layer`).
- `dataset_indices: List[int]` — the *validation* indices for the fold/subject this checkpoint was trained for. Read from the existing splits JSON (`data/splits/kfold_splits_processed_*.json`) for kfold or LOSO is single-subject.

### 5.2 Pipeline (per checkpoint × per validation graph)
1. Reconstruct `FlexibleGATNet` from `config.yaml`.
2. Load `state_dict` (`strict=True`).
3. Apply the **same val transform** the trained model expects — `StandardizeGraphFeatures` with **fold-leak-free** stats. Call `dataset.compute_stats(train_indices)` for the matching fold (we hold the splits JSON), not the global stats.
   - This is non-negotiable: explanations on raw, un-normalised inputs will be incoherent w.r.t. what the model actually saw at validation time.
4. Wrap the model:
   ```python
   class ProbWrapper(nn.Module):
       def __init__(self, m): super().__init__(); self.m = m
       def forward(self, x, edge_index, edge_attr, batch=None):
           if batch is None:
               batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
           return F.softmax(self.m(x, edge_index, edge_attr, batch), dim=-1)
   ```
5. Configure the explainer (canonical PyG pattern, fixed for our 2-class softmax model):
   ```python
   explainer = Explainer(
       model=ProbWrapper(model).eval(),
       algorithm=GNNExplainer(epochs=200, lr=0.01),
       explanation_type='model',
       node_mask_type='attributes',
       edge_mask_type='object',
       model_config=dict(
           mode='multiclass_classification',
           task_level='graph',
           return_type='probs',
       ),
   )
   ```
6. For each validation graph in this fold/subject:
   ```python
   explanation = explainer(
       x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
       target=data.y.unsqueeze(0) if data.y.dim() == 0 else data.y,
   )
   node_mask = explanation.node_mask    # [23, 6]
   edge_mask = explanation.edge_mask    # [E]
   ```
7. **Per-trial reductions:**
   - `channel_importance[c] = |node_mask[c, :]|.sum()` → length 23
   - `feature_importance[f] = |node_mask[:, f]|.sum()` → length 6 (mean/min/max/skew/kurt/var)
   - `pair_importance[c_i, c_j] += edge_mask[e]` for each edge `e` with endpoints `(c_i, c_j)`. Sum then symmetrise (`M = (M + M.T) / 2`) — the directed-graph builds duplicate the same correlation in both directions, so this is loss-free.

### 5.3 Optional sanity-check explainer
- A second pass with `CaptumExplainer('IntegratedGradients')` (PyG wraps Captum) on the same wrapper. Cross-check the top-K channel rankings with Spearman ρ. Disagreement > 0.5 ρ between methods is a flag, not a failure — report both.
- **rev. 3 addition (see §15.4 / §15.8):** add a third pass with `AttentionExplainer(reduce='mean')` for an essentially-free third estimator that uses the trained GATv2 attention directly. C4 (§11) is upgraded to require ρ ≥ 0.4 between any two of the three rankings.

---

## 6. ST Explainer Spec — `src/xai/st_explainer.py`

### 6.1 Primary path: native attention (no extra training)
1. Reconstruct `WindowedSpatioTemporalGATNet` from `config.yaml` (note: ST uses scalar `n_filters`, `heads`, `temporal_hidden`, `temporal_layers`, `window_size`, `window_stride`; **not** per-layer lists like SG).
2. Load `state_dict`.
3. Apply the same val transform with fold-leak-free edge stats (ST's `compute_stats` returns only edge stats — node features are per-trial z-scored, no leakage).
4. For each validation graph:
   ```python
   out = model.explain(data.to(device), device=device)
   # out['temporal_attention']: [K]
   # out['spatial_attention'][k][l]: [E, heads]
   ```
5. **Per-trial reductions:**
   - **Temporal:** keep `α_k` directly; convert window index → seconds via `(k * window_stride / fs, (k * window_stride + window_size) / fs)`.
   - **Spatial per-pair:**
     ```python
     # data.edge_index: [2, E];  edge endpoints (c_i, c_j)
     # For each window k, average heads, average layers (or take the last layer):
     attn_kl = spatial_attention[k][l].mean(dim=-1)     # [E]
     attn_per_window = mean_layers(attn_kl)             # [E]
     # Weight windows by temporal attention so the trial-level summary
     # aligns with what the model "looked at":
     pair_E = sum_k(α_k * attn_per_window_k)            # [E]
     pair_matrix[c_i, c_j] += pair_E[e]                 # 23×23
     ```
   - **Channel importance** = row-sum of the symmetrised pair matrix (a node's total attended in/out-flow).

### 6.2 Aggregating the heads
Default: arithmetic mean across heads. Alternative: max across heads (highlights the most-attending head). Configurable via `--head-reduce {mean,max}`; report both in the run JSON.

### 6.3 Aggregating the layers
Two sensible choices:
- **`last`**: only layer L−1's attention. Closer to the "decision-relevant" attention.
- **`mean`**: average of all layers. Smoother, less noisy.
Default: `mean`. Configurable.

### 6.4 Optional supplementary path: GNNExplainer with object masks
To get a method-independent cross-check on which channels matter, do **not** ask GNNExplainer for `[23, T]` attribute masks. Instead:

> **rev. 3 note (see §15.4):** PyG's `AttentionExplainer` was evaluated as a candidate replacement for the per-window aggregation in §6.1 and rejected — it returns a single fused `[E]` mask with no per-window decomposition and no temporal `α_k`, which discards the temporal localisation that motivates ST. It remains useful as an additional aggregate cross-check (`AttentionExplainer(reduce='mean')`) alongside the GNNExplainer object-mask pass below.

```python
explainer = Explainer(
    model=ProbWrapper(model).eval(),
    algorithm=GNNExplainer(epochs=200, lr=0.01),
    explanation_type='model',
    node_mask_type='object',     # [N] scalar per node
    edge_mask_type='object',     # [E] scalar per edge
    model_config=dict(mode='multiclass_classification', task_level='graph', return_type='probs'),
)
```
Then compare top-K channels between native attention and GNNExplainer. Treat agreement (Spearman ρ ≥ 0.5) as supporting evidence; disagreement as a finding to discuss.

---

## 7. Aggregation Pipeline — `src/xai/aggregate.py`

### 7.1 Hierarchy
```
trial-level   ← raw output of one explainer call
  ↓ mean over correctly-classified trials of one subject (per checkpoint)
subject-level  (62 subjects in dataset)
  ↓ mean over subjects within one fold/CV-arm
fold-level
  ↓ mean over folds (kfold-5/10) OR over LOSO subjects
regime-level   ← final population-level output for the paper
```

### 7.2 Why mean correctly-classified trials only
A misclassified trial's saliency tells us about how the model failed, not about the discriminative biology. Following common XAI practice we restrict aggregation to predictions where `argmax(softmax(logits)) == y`. Reported alongside: %-correct included, raw-vs-correct ranking Spearman ρ (sanity check).

### 7.3 Outputs per (architecture, regime, mt)
- `node_importance.csv`: 23 rows × {`channel`, `mean`, `std`, `n_trials`, `n_subjects`, `rank`}
- `edge_importance.csv`: top-K rows × {`channel_i`, `channel_j`, `mean`, `std`, `n_trials`, `rank`}
- `channel_pair_matrix.npy`: 23×23 float32, symmetrised, mean across population
- `temporal_attention.csv` (ST only): K rows × {`window_idx`, `t_start_s`, `t_end_s`, `mean`, `std`}
- `feature_importance.csv` (SG only): 6 rows × {`feature` ∈ {mean,min,max,skew,kurt,var}, `mean`, `std`}
- `run.json`: full provenance — checkpoint paths, config hashes, seed, `explainer_epochs`, `head_reduce`, `layer_reduce`, %-correct included, scipy/torch/PyG versions.

### 7.4 Cross-architecture comparison
After SG and ST runs both produce 23-vectors of channel importance and 23×23 pair matrices:
- Spearman ρ between SG and ST channel rankings.
- Top-5 / top-10 set overlap (Jaccard).
- **Pair-matrix diff** (per user feedback 2026-05-07): `M_diff = z(M_sg) − z(M_st)` where each matrix is z-scored over its non-diagonal entries. Highlights edges that one architecture sees as discriminative and the other does not.
- All artefacts saved to `research/xai/cross_arch_comparison.{csv,md,npy}`.

---

## 8. Visualisation — `src/xai/visualize.py`

Three figures per (arch, regime, mt) plus one cross-arch figure:

1. **`fig_montage_channel_importance.png`** — 5×7 grid heatmap with the 23 channels at `GRID_POS`. Cell colour = z-score of channel importance vs the 23-channel mean. Channel labels overlaid. Same layout used in `02_brain_activation/REPORT.md`, ensuring direct visual comparability with the MWU activation maps.

2. **`fig_pair_matrix.png`** — 23×23 symmetric heatmap of `channel_pair_matrix.npy`, with diagonal blanked (no self-loops). Channels ordered by row-sum (so the visually dense block is the high-importance subgraph).

3. **`fig_top_pairs_chord.png`** *(optional)* — chord-style diagram of the top-20 edges using channel positions on the 5×7 grid, line width ∝ importance. Only produced if matplotlib + the chord-diagram helper is available.

4. **(ST only) `fig_temporal_attention.png`** — line plot of `α_k` over time-in-seconds, with healthy and GAD overlaid (mean ± 95% CI).

5. **(cross-arch) `fig_sg_vs_st_scatter.png`** — scatter of SG vs ST channel importance ranks, points labelled with channel names; Spearman ρ in legend.

6. **(cross-arch) `fig_pair_matrix_diff.png`** — 23×23 diverging-colour heatmap of `M_diff = z(M_sg) − z(M_st)`. Red cells: SG-only edges; blue cells: ST-only edges; near-zero cells: agreement. Channels share row/column ordering with `fig_pair_matrix.png` from each architecture for visual line-up.

All figures use `dpi=300`, vector-friendly fonts, `.svg` mirror saved alongside `.png` for paper inclusion.

---

## 9. Notebook Execution Model

XAI runs are driven from Jupyter notebooks under `src/notebook/xai/` (per user feedback 2026-05-07). Each notebook imports from `src/xai/*` — no command-line flags, no argparse, no shell wrappers. This keeps figure exploration, parameter tuning, and provenance recording tightly co-located with the visual outputs the paper needs.

### 9.1 Notebook responsibilities
- Resolve absolute paths to checkpoints, splits JSON, data directory.
- Build an `XAIRunConfig` dataclass instance.
- Call `src.xai.run(config)` (or its lower-level building blocks).
- Render figures inline with `matplotlib`; mirror to `.png` + `.svg` in the output directory.
- Print top-K tables in cells for visual review before saving CSVs.

### 9.2 Canonical SG cell pattern (in `01_sg_population.ipynb`)
```python
from src.xai import XAIRunConfig, run_sg, plot_montage_channel_importance, plot_pair_matrix

# data_dir + splits_json are read from each checkpoint's own config.yaml (rev. 4).
# Pass them here only as overrides — e.g., when the recorded path was a cloud-only
# path that auto-rebase can't resolve.
cfg = XAIRunConfig(
    arch="sg",
    hb="hbo",
    regime="kfold-5",
    mt=2,
    experiment_root="research/experiments/20260506/leak-free-patience-9999/spatial-graph",
    gnn_explainer_epochs=200,
    gnn_explainer_lr=0.01,
    include_misclassified=False,
    output_dir="research/xai/sg/kfold-5/mt2",
    seed=42,
    device="cuda:0",
    # data_dir_override=None,     # default: derive from config.yaml + rebase
    # splits_json_override=None,  # default: derive from config.yaml (None for LOSO)
)

result = run_sg(cfg)         # → PopulationResult dataclass
plot_montage_channel_importance(result, save_dir=cfg.output_dir)
plot_pair_matrix(result, save_dir=cfg.output_dir)
result.to_csv(cfg.output_dir)  # writes node_importance.csv, edge_importance.csv, channel_pair_matrix.npy, run.json
```

### 9.3 Canonical ST cell pattern (in `02_st_population.ipynb`)
```python
from src.xai import XAIRunConfig, run_st, plot_montage_channel_importance, plot_pair_matrix, plot_temporal_attention

# Same as §9.2: data_dir + splits_json are read from each checkpoint's own config.yaml.
cfg = XAIRunConfig(
    arch="st",
    hb="hbo",
    regime="kfold-5",
    mt=2,
    experiment_root="research/experiments/20260501/spatial_temporal_graph/5-fold",
    head_reduce="mean",
    layer_reduce="mean",
    run_supplementary_gnnexplainer=False,
    include_misclassified=False,
    output_dir="research/xai/st/kfold-5/mt2",
    seed=42,
    device="cuda:0",
)

result = run_st(cfg)
plot_montage_channel_importance(result, save_dir=cfg.output_dir)
plot_pair_matrix(result, save_dir=cfg.output_dir)
plot_temporal_attention(result, save_dir=cfg.output_dir)
result.to_csv(cfg.output_dir)
```

### 9.4 Cross-architecture cell pattern (in `03_cross_arch_comparison.ipynb`)
```python
from src.xai import compare_architectures, plot_sg_vs_st_scatter, plot_pair_matrix_diff

sg_result = PopulationResult.from_csv("research/xai/sg/kfold-5/mt2")
st_result = PopulationResult.from_csv("research/xai/st/kfold-5/mt2")

cmp = compare_architectures(sg_result, st_result)
# cmp: spearman_rho, jaccard_top5, jaccard_top10, pair_matrix_diff (23×23)

plot_sg_vs_st_scatter(cmp, save_dir="research/xai/cross_arch")
plot_pair_matrix_diff(cmp, save_dir="research/xai/cross_arch")
cmp.to_files("research/xai/cross_arch")
```

### 9.5 Fan-out
Each population notebook contains 6 sequential cells (one per CV-regime × mt) plus a summary table cell. No bash wrapper script. If batch reproducibility is needed later, `nbconvert --execute` can run any of the four notebooks headlessly.

---

## 10. Data-Leakage & Correctness Guarantees

These match the project's existing leak-free guarantees (cf. `SPEC_core_st.md` §4.3 and project memory `project_st_vs_sg_validation.md`). XAI must not silently re-introduce leakage:

1. **Per-fold normalisation reused.** For SG, each checkpoint's val transform must use `dataset.compute_stats(train_indices_for_that_fold)`. For LOSO checkpoints, that's `train_indices = all_indices \ subject_indices`. For ST, only edge stats are normalised — already leak-free by construction.
2. **Validation indices only.** Explanations are computed on the *validation* graphs of each fold/subject. We never explain on the training set with the same checkpoint — it would inflate confidence in spurious channels the model overfit to.
3. **No transform-time augmentation.** XAI uses the val transform path (no `EdgeDropout`, no `FeatureMask`, no `RWPE` randomness if disabled — just `StandardizeGraphFeatures`).
4. **Data-path resolution (rev. 4 rewrite).** `data_dir` and `splits_json` are **read from each checkpoint's own `config.yaml`** — the SPEC does not hardcode them. As of 2026-05-08 all in-scope SG and ST checkpoints share `data/processed-new-mc`; SG kfold + ST kfold share `data/splits/kfold_splits_processed_new_mc.json`; LOSO has `splits_json: null` and derives splits in-code from dataset subject IDs (`src/core/dataset.py:get_loso_splits` / `src/core_st/dataset.py:get_loso_splits`).
   **Auto-rebase rule for non-portable paths:**
   ```python
   def _resolve_data_dir(raw: str, project_root: Path) -> Path:
       p = Path(raw)
       if p.exists():                                         # local path — use as-is
           return p
       if str(p).startswith('/root/remote-training-setup/'):  # known cloud prefix
           candidate = project_root / str(p)[len('/root/remote-training-setup/'):]
           if candidate.exists():
               return candidate
       candidate = project_root / 'data' / p.name              # fallback by basename
       if candidate.exists():
           return candidate
       raise FileNotFoundError(f"data_dir from config.yaml not resolvable: {raw}")
   ```
   Every rebase event is appended to `run.json` under `path_rebases: [{checkpoint, raw_path, resolved_path}, ...]`. Override fields `XAIRunConfig.data_dir_override` / `splits_json_override` exist for the rare case the rebase logic can't recover the right path.
5. **Deterministic seeds.** GNNExplainer optimisation is stochastic (random mask init). Fix `torch.manual_seed(seed)` and `numpy.random.seed(seed)` per-trial for reproducibility; `seed` is a field on `XAIRunConfig` (default `42`).
6. **Correctly-classified-only aggregation (confirmed by user 2026-05-07).** §7.2 already filters to `argmax(softmax(logits)) == y`. The notebook prints the per-fold inclusion rate so the reader can audit how many trials were dropped.

---

## 11. Validation & Acceptance Criteria

The XAI pipeline is accepted when **all** of:

- [ ] **C1 — Reload sanity.** Each loaded SG/ST checkpoint reproduces the validation F1/accuracy reported in its `*.pkl` results file to within ±0.005.
- [ ] **C2 — Stability across folds.** Spearman ρ between channel-importance ranks of any two folds within the same regime ≥ 0.4 (population aggregation requires non-trivial agreement; ρ < 0.4 in all pairs would mean the rankings are noise).
- [ ] **C3 — Stability across mt.** ρ(mt=2 ranking, mt=4 ranking) ≥ 0.5 for both SG and ST.
- [ ] **C4 — Method cross-check (SG).** ρ(GNNExplainer top-10 channels, IntegratedGradients top-10 channels) ≥ 0.4. Disagreement triggers a documented investigation, not auto-rejection. *(rev. 3, see §15.4 / §15.8: extended to a 3-way comparison by also running `AttentionExplainer`. The criterion becomes ρ ≥ 0.4 between any two of the three rankings.)*
- [ ] **C5 — ST attention vs ST GNNExplainer cross-check.** ρ(native attention top-10, GNNExplainer object-mask top-10) ≥ 0.4.
- [ ] **C6 — Biological prior plausibility (advisory, not blocking).** At least 2 of {S1_D1, S5_D5, S3_D3, S2_D1, S4_D5, S4_D7} appear in the top-10 for each architecture × regime. These are the channels surfaced by the parallel statistical analyses (`02_brain_activation` MWU + `06_canonical_beta` β-tests). Disagreement is a finding worth discussing in the paper, not a bug.
- [ ] **C7 — Output format.** All CSVs / NPYs match §7.3; `run.json` includes `git_sha`, package versions, full CLI args.

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| GNNExplainer instability on small validation graphs (E ≈ 200) | Run with `epochs=200`, `lr=0.01`; report top-K stability (Jaccard) across 3 random seeds in the run JSON. |
| ST `[23, T]` masks too high-dim to interpret | Choose native attention as primary; relegate GNNExplainer to `node_mask_type='object'` only. |
| Fold-stat mismatch silently changes inputs | CLI hard-fails if `--data-dir` or `--splits-json` disagrees with the loaded `config.yaml`. |
| 3-class softmax bug from notebook §6 (sigmoid on 2-logits) | Use `softmax + multiclass_classification` wrapper consistently. SPEC §3.1. |
| Long LOSO runtime (62 subjects × ~20–40 trials each × GNNExplainer 200 epochs) | Parallelise across subjects (one worker per checkpoint); skip GNNExplainer on LOSO if total wall-time > 12 h — keep ST native attention which is essentially free at LOSO scale. |
| Conflating mt=2 and mt=4 datasets | `XAIRunConfig.mt` is required and validated against the loaded `config.yaml`. The dataset is instantiated with the checkpoint's `max_trials`, not a notebook default. |
| Notebook drift (cells edited ad-hoc, lose provenance) | All numerical work goes through `src/xai/*` building blocks. Notebook cells contain only paths, config, plotting calls, and prose. `run.json` records `git_sha`, package versions, and the full `XAIRunConfig` so any cell's output can be reproduced from CSVs alone. |

---

## 13. Implementation Order (when SPEC is approved)

**Phase A — building blocks (`src/xai/`)**
1. `src/xai/channels.py` — copy CHANNEL_NAMES + GRID_POS verbatim from `04_severity_correlation`.
2. `src/xai/config.py` — `XAIRunConfig` dataclass.
3. `src/xai/checkpoints.py` — `discover_checkpoints(experiment_root) → List[CheckpointInfo]` and `load_checkpoint(info, cfg) → (model, dataset, val_indices, val_transform, train_indices)`. Validates §10.4.
4. `src/xai/sg_explainer.py` — implement §5; expose `run_sg(cfg) → PopulationResult`.
5. `src/xai/st_explainer.py` — implement §6 (native attention first; supplementary GNNExplainer behind a config flag); expose `run_st(cfg) → PopulationResult`.
6. `src/xai/aggregate.py` — implement §7; defines `PopulationResult` dataclass with `to_csv` / `from_csv`.
7. `src/xai/visualize.py` — implement §8.
8. `src/xai/io.py` — write `run.json` (git_sha, versions, config dump).
9. `src/xai/__init__.py` — re-export the top-level API used by notebooks.
10. Unit tests under `tests/xai/`:
    - reload reproduces F1 (C1)
    - aggregator handles ragged subject trial counts
    - pair-matrix symmetrisation
    - leak-free transform contract

**Phase B — notebooks (`src/notebook/xai/`)**
11. `00_setup_and_smoke.ipynb` — env/import check, single SG fold + single ST fold smoke run, visual confirmation.
12. `01_sg_population.ipynb` — 6 cells (kfold-5/10/loso × mt2/mt4), each producing CSVs + figures.
13. `02_st_population.ipynb` — 6 cells, same fan-out, plus temporal-attention figure.
14. `03_cross_arch_comparison.ipynb` — load saved CSVs, run `compare_architectures`, render scatter + pair-matrix diff.
15. `src/notebook/xai/README.md` — how to run, kernel requirements, expected runtime per regime, where outputs land.

**Phase C — validation**
16. Run §11 acceptance criteria C1–C7 from inside the notebooks; copy the validation tables into the README.

**Phase D — future work (not in this iteration, per user feedback)**
- HbR / HbT extension: rerun the validated notebooks with `hb='hbr'` and `hb='hbt'`.
- MNE / AtlasViewer Brodmann labelling pass — separate downstream SPEC.

---

## 14. Open Questions — Resolutions (2026-05-07)

| # | Question | Resolution |
|---|---|---|
| 1 | Execution model: CLI vs notebook? | **Notebook-driven.** Building blocks live in `src/xai/`, notebooks in `src/notebook/xai/`. No CLI in this iteration. |
| 2 | Output directory location? | **`research/xai/...`** (top-level), since outputs span multiple experiment dates and both architectures. Mirrors the existing `research/experiments/...` sibling. |
| 3 | Misclassified-trial policy? | **Correctly-classified trials only** (§7.2). Notebook prints inclusion rate per fold for auditability. |
| 4 | Cross-architecture pair-matrix diff figure? | **Yes.** Added to §7.4 and §8 as `fig_pair_matrix_diff.png` with `M_diff = z(M_sg) − z(M_st)`. |
| 5 | HbR / HbT extension trigger? | **Future work.** Triggered only after HbO building blocks are validated. Same notebooks rerun with `hb='hbr'|'hbt'`. |
| 6 | MNE / AtlasViewer Brodmann mapping? | **Future work, separate downstream SPEC.** Begins after this XAI pipeline is fully working and accepted. |

---

## 15. PyG API Reference (appendix, pinned to `torch_geometric==2.7.0`)

This appendix is the API contract for everything the explainer building blocks in `src/xai/` will compile against. Every signature here was verified on 2026-05-07 by combining `mcp__context7__*` lookups against the canonical PyG docs (`/pyg-team/pytorch_geometric`) with `inspect.signature` on the locally installed `torch_geometric==2.7.0` and `inspect.getsource` of the algorithms' `supports()` methods. Treat this as the source of truth; the prose in §5/§6 is a project-specific reading of this contract.

### 15.1 The `Explainer` interface (`torch_geometric.explain.Explainer`)

```python
Explainer(
    model: torch.nn.Module,
    algorithm: ExplainerAlgorithm,                              # see §15.3
    explanation_type: Union[ExplanationType, str],              # 'model' | 'phenomenon'
    model_config: Union[ModelConfig, Dict[str, Any]],           # see §15.2
    node_mask_type: Optional[Union[MaskType, str]] = None,      # 'object' | 'common_attributes' | 'attributes'
    edge_mask_type: Optional[Union[MaskType, str]] = None,      # 'object' (currently the only edge variant)
    threshold_config: Optional[ThresholdConfig] = None,         # see §15.2
)
```

Calling the explainer:

```python
Explainer.__call__(
    x: Tensor | Dict[str, Tensor],
    edge_index: Tensor | Dict[Tuple[str, str, str], Tensor],
    *,
    target: Optional[Tensor] = None,                            # required if explanation_type='phenomenon'
    index: Optional[int | Tensor] = None,                       # node-level subject; None for graph-level
    **kwargs                                                    # ← edge_attr, batch, etc. go HERE
) -> Explanation | HeteroExplanation
```

**Project-relevant facts**
- `edge_attr` and `batch` are **not** positional or named parameters — they ride in `**kwargs`. Our wrappers in §5.2 / §6.1 thread them through unchanged.
- `target` is required when `explanation_type='phenomenon'` (we always use `'model'`, so it's optional but still useful for symmetry).
- `index=None` is correct for graph-level classification; passing an integer interprets the call as node-level explanation.

### 15.2 Config dataclasses

```python
ModelConfig(
    mode: ModelMode | str,                # 'binary_classification' | 'multiclass_classification' | 'regression'
    task_level: ModelTaskLevel | str,     # 'node' | 'edge' | 'graph'
    return_type: Optional[ModelReturnType | str] = None   # 'raw' | 'probs' | 'log_probs'
)

ExplainerConfig(
    explanation_type: ExplanationType | str,   # 'model' | 'phenomenon'
    node_mask_type: Optional[MaskType | str],  # 'object' | 'common_attributes' | 'attributes' | None
    edge_mask_type: Optional[MaskType | str],  # 'object' | None
)

ThresholdConfig(
    threshold_type: ThresholdType | str,   # 'hard' | 'topk' | 'topk_hard'
    value: float | int                     # float for 'hard'; int K for 'topk' / 'topk_hard'
)
```

**`MaskType` semantics (from PyG 2.7.0 source):**
| Value | Mask shape on `Data.x[N, F]` | Meaning |
|---|---|---|
| `'object'` | `[N]` | One scalar per node — coarse "is this node important?" |
| `'common_attributes'` | `[F]` | One scalar per feature, shared across nodes — "is this feature column important?" |
| `'attributes'` | `[N, F]` | Per-node-per-feature — finest-grained, what GNNExplainer learns by default |

For SG (§5.2) we use `'attributes'` (`[23, 6]`) so we can split into channel-importance (row-sum) and feature-importance (col-sum). For ST cross-check (§6.4) we use `'object'` (`[23]`) to avoid the `[23, 326]` blow-up.

**`ThresholdConfig` is the canonical replacement for our manual top-K post-processing on `edge_mask`.** Passing `threshold_config=dict(threshold_type='topk', value=20)` makes the returned `Explanation.edge_mask` already contain only the top-20 edges (with the rest zeroed). We **do not** wire this into the trial-level explainer — population-level aggregation needs the full-precision mask — but it is the right choice for the chord-diagram visualisation (§8 fig 3) so the diagram is computed from masks generated with the same `threshold_config` used at plot time.

### 15.3 Algorithm matrix (PyG 2.7.0 ships exactly these in `torch_geometric.explain.algorithm`)

| Algorithm | Constructor | `node_mask_type` allowed | `edge_mask_type` allowed | `explanation_type` | `task_level` | Pre-train? | Used here? |
|---|---|---|---|---|---|---|---|
| `GNNExplainer` | `(epochs=100, lr=0.01, **kwargs)` | object / common_attributes / attributes | object | model + phenomenon | node + edge + graph | per-call optimisation only | **Yes** — SG primary (§5.2) and ST supplementary (§6.4) |
| `CaptumExplainer` | `(attribution_method, **kwargs)` | None or `'attributes'` only | object | model + phenomenon | node + edge + graph | no | **Yes** — SG sanity check (§5.3) with `'IntegratedGradients'` |
| `AttentionExplainer` | `(reduce='max')` | **None only** (no node mask) | object (auto-extracted) | **model only** | node + edge + graph | no | **Yes** — added as a fast cross-check (see §15.4) |
| `PGExplainer` | `(epochs, lr=0.003, **kwargs)` | **None only** | object | **phenomenon only** | node + graph | **YES** (parametric — needs separate training loop) | **No** — see §15.5 |
| `GraphMaskExplainer` | `(num_layers, epochs=100, lr=0.01, penalty_scaling=5, ...)` | object / attributes | object | model + phenomenon | node + edge + graph | per-call optimisation only | No — too expensive vs. GNNExplainer for our scale |
| `DummyExplainer` | `(*args, **kwargs)` | any | any | any | any | n/a | Tests only (random masks) |

**Key constraints (from each algorithm's `supports()` source):**
- `CaptumExplainer.supports()` rejects `node_mask_type ∈ {'object', 'common_attributes'}` — only `None` or `'attributes'`. Compatible with our SG plan (we already use `'attributes'`).
- `AttentionExplainer.supports()` rejects any `node_mask_type` and any `explanation_type='phenomenon'`. It is **edge-only** + **model-explanation only**.
- `PGExplainer.supports()` rejects any `node_mask_type` and rejects `explanation_type='model'`. It is **edge-only** + **phenomenon-explanation only**.

### 15.4 `AttentionExplainer` — newly adopted (rev. 3)

```python
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import AttentionExplainer

explainer = Explainer(
    model=ProbWrapper(model).eval(),
    algorithm=AttentionExplainer(reduce='mean'),     # 'mean' | 'max' | 'min'
    explanation_type='model',
    edge_mask_type='object',                         # AttentionExplainer rejects node masks
    model_config=dict(mode='multiclass_classification', task_level='graph', return_type='probs'),
)
explanation = explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, target=data.y)
edge_mask = explanation.edge_mask                    # [E]
```

`AttentionExplainer` walks the model, **automatically** finds every `GATConv` / `GATv2Conv` / `TransformerConv` it contains, harvests their attention coefficients (`alpha`) via PyG's `MessagePassing` hooks, and reduces across **layers and heads** with the `reduce` argument (default `'max'`). For our use cases:

- **SG (§5.3 cross-check):** drop-in alternative to GNNExplainer that requires no per-trial optimisation — explanation is essentially free at inference time. Run with `reduce='mean'` for our primary cross-check (matches our SPEC §5.3 head/layer aggregation default) and `reduce='max'` as a secondary view (highlights the most-attending head). Add to the §11 cross-check matrix as a third independent estimator (alongside GNNExplainer and CaptumExplainer-IG).
- **ST (§6.4 supplementary):** use only as an *aggregate-across-windows* sanity check — `AttentionExplainer` has no notion of the windowed forward pass, so the attention it harvests is a single fused view across all K windows. Native `model.explain()` (§6.1) remains primary because it preserves per-window granularity and the temporal `α_k` weights.

**Why not promote it to ST primary:** `AttentionExplainer` returns a single `[E]` mask per call, with no per-window decomposition and no temporal attention. Replacing `model.explain()` with it would discard the temporal localisation that motivates picking ST in the first place.

### 15.5 `PGExplainer` — considered, not adopted

`PGExplainer` is a parametric edge explainer that trains a small auxiliary network to produce edge masks; once trained, inference is fast. PyG's tutorial demonstrates it for graph regression. We **deliberately exclude** it from the v1 pipeline:

1. It only supports `explanation_type='phenomenon'` (the explanation tracks the *true label* `y`, not the model's prediction). For our research question — *what does the model use to discriminate Healthy vs GAD?* — `'model'` explanations are the correct framing.
2. It requires an additional per-fold training loop (`explainer.algorithm.train(epoch, ..., target=batch.target)` for ~30 epochs). For our matrix of 2 archs × 3 regimes × 2 mt × {5, 10, 62} folds = up to ~140 explainer-trainings. Cost is high; benefit over GNNExplainer at our edge count (E ≈ 200 per trial) is unclear from the literature.
3. It does not produce node masks, so it cannot answer "which channel matters" — only "which connection matters". For population-level channel ranking we'd still need a node-mask method.

If a reviewer specifically asks for a PGExplainer comparison, the simplest extension is a future-work notebook `04_pgexplainer_followup.ipynb` that consumes the same checkpoints but trains PGExplainer per fold and outputs only the 23×23 pair matrix.

### 15.6 `Explanation` object — methods we will use

`Explanation` extends `torch_geometric.data.Data`, so all the standard `Data` methods are available. The XAI-specific bits we'll touch:

- `explanation.node_mask` — `[N, F]` for `'attributes'`, `[N]` for `'object'`, `[F]` for `'common_attributes'`.
- `explanation.edge_mask` — `[E]`.
- `explanation.available_explanations` — list of mask names actually populated (depends on which mask types we asked for).
- `explanation.threshold(...)` — applies a `ThresholdConfig` post-hoc; equivalent to passing `threshold_config` into the `Explainer`.
- `explanation.get_explanation_subgraph()` / `get_complement_subgraph()` — restricts/inverts the graph by the masks. We don't use these for population aggregation but may use them in `00_setup_and_smoke.ipynb` for visual sanity.
- `explanation.visualize_feature_importance(path, top_k=K)` — bar-chart of per-feature importance. Used by `01_sg_population.ipynb` for the SG `[23, 6]` mask.
- `explanation.visualize_graph(path)` — renders the masked computation subgraph as a PDF. Useful for spot-checks; not part of the population pipeline because our 23-channel grid montage (§8 fig 1) is more readable than PyG's auto-layout.

### 15.7 `CaptumExplainer` — supported attribution methods

`CaptumExplainer.SUPPORTED_METHODS` (verified from source in PyG 2.7.0):

```python
['IntegratedGradients', 'Saliency', 'InputXGradient',
 'Deconvolution', 'ShapleyValueSampling', 'GuidedBackprop']
```

We use `'IntegratedGradients'` per §5.3. `'Saliency'` is a faster fallback if IG is too slow at LOSO scale (see §12 Risks row 5); `'ShapleyValueSampling'` is the gold-standard but is `O(n_features²)` per trial and infeasible at our scale. The other three are not commonly used for GNN explainability and we ignore them.

### 15.8 Where this changes the implementation plan

The newly verified API has **two** practical deltas vs §5/§6 as written in rev. 2:

1. **§5.3 SG sanity check is now a 3-way comparison** (was 2-way): GNNExplainer + CaptumExplainer-IG + AttentionExplainer. Acceptance criterion C4 (SPEC §11) becomes: "ρ between any two of the three rankings ≥ 0.4". `src/xai/sg_explainer.py` adds an `attention_explainer` code path behind a `XAIRunConfig.run_attention_cross_check: bool = True` flag.
2. **§8 fig 3 (top-pairs chord)** can use `threshold_config=dict(threshold_type='topk', value=20)` instead of manual `np.argsort()` slicing of the population pair matrix. Cosmetic, but cleaner provenance — the threshold config gets dumped into `run.json`.

No other section of the SPEC needs to change. §5.2 (SG primary), §6.1 (ST primary, native `model.explain()`), §6.4 (ST supplementary, GNNExplainer with `node_mask_type='object'`), §7 (aggregation), and §10 (leak-free guarantees) all remain valid.

---

## 16. References

- PyG 2.7.0 explainability docs (canonical): https://pytorch-geometric.readthedocs.io/en/2.7.0/modules/explain.html
- PyG 2.7.0 explain tutorial: https://pytorch-geometric.readthedocs.io/en/2.7.0/tutorial/explain.html
- Captum (attribution methods): https://captum.ai/api/
- Original GNNExplainer paper: Ying et al., "GNNExplainer: Generating Explanations for Graph Neural Networks", NeurIPS 2019.
- Original PGExplainer paper: Luo et al., "Parameterized Explainer for Graph Neural Network", NeurIPS 2020.
- Original AttentionExplainer-style approach (GAT attention as explanation): Veličković et al., "Graph Attention Networks", ICLR 2018, and Brody et al., "How Attentive are Graph Attention Networks?" (GATv2), ICLR 2022.
