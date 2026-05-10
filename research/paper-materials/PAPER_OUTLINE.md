# PAPER_OUTLINE — Graph-Based fNIRS Classification of Healthy Controls vs. Generalised Anxiety Disorder

> **Purpose.** Annotated version of the rough outline by the author. Each leaf
> carries a **status badge** and a **source pointer** so the downstream
> `PAPER_SPEC_PLAN.md` can lift numbers, figures, and citations directly without
> re-deriving them.
>
> **Source of truth for ML results:** `experiments/` (the user-validated final
> run set of 2026-05-09 for ST and 2026-05-06/07 for SG). The
> `research/experiments/2026-04-28 … 2026-05-09` directories are kept for
> hyperparameter-search and ablation provenance only.
>
> **Date:** 2026-05-10 · **Author:** Aunuun Jeffry Mahbuubi
> **Repo root:** `/home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method`

---

## Status legend

| Badge | Meaning |
|---|---|
| **[DONE]** | Numerical result already exists; pointer below. Paper task = lift, narrate, format. |
| **[WRITE]** | No new analysis needed; prose must be drafted from named source(s). |
| **[TODO]** | New analysis or new content (math derivation, figure, table) that has not been produced yet. |
| **[NEEDS-DATA]** | Blocked: a dataset, label, or pre-processing step is missing or incomplete. |
| **[FUTURE]** | Explicitly out-of-scope for this paper (record as Future Work). |

A leaf may carry multiple badges (`[DONE+WRITE]` is common). When a number is
known, it is quoted inline so reviewers of this outline can sanity-check it
without re-running anything.

---

## Cross-cutting decisions (locked in before drafting begins)

| Topic | Decision | Source |
|---|---|---|
| Primary chromophore for stats | **HbO** (replaced HbT in original draft) | `src/notebook/statistical-analysis/02_brain_activation/REPORT.md §0` |
| Primary chromophore for ML | Report **HbO + HbR + HbT** in the main results table; HbR is the LOSO winner (F1 = 0.8406) | `experiments/spatial_temporal_graph/experiment_metrics.xlsx` LOSO sheet |
| Primary architecture for paper headline | **ST-GATv2** (`WindowedSpatioTemporalGATNet`) | LOSO F1 ranges 0.79–0.84 vs SG 0.73–0.78 |
| Primary CV regime in headline | **5-fold + 10-fold + LOSO** (all three) | User outline §2.3.5; experiments/ has all three |
| Subject-level vs trial-level eval | **Trial-level metrics** (the codebase default — no soft/hard voting). Subject-level voting is **future work**. | User outline §2.3.5.2; `src/core_st/training.py` |
| Splits | Deterministic JSON fixed for kfold; LOSO derived in code | `data/splits/kfold_splits_processed_new_mc.json`; `dataset.py:get_loso_splits` |
| Pre-processing pipeline | Wavelet MC + CBSI + bandpass 0.01–0.5 Hz (`processed-new-mc`) | `data/DATA_QUALITY_REPORT.md §9` |
| Augmentation | **None** for paper headline (`_noaug_*` runs); aug variants are ablation only | All `experiments/*/...noaug_*` directories |
| XAI scope | SG = HbO only; ST = HbO/HbR/HbT (rev. 6); native attention is the primary ST channel | `docs/SPEC_xai_graph.md` rev. 6 |
| Atlas registration | Brodmann via fsaverage midpoint projection (in scope); MCX Monte Carlo deferred | `docs/SPEC_xai_graph.md §16` |
| Cohort exclusion | Keep all 62 subjects in main table; pre-specified sensitivity drops (AH024, AH029, LA063, demographics-missing 11) are **future-work** appendix material | `data/DATA_QUALITY_REPORT.md §6`; `FUTURE_ANALYSES.md §1.3` |
| Age confound | Acknowledged limitation; ANCOVA done in `05_age_adjusted` for §02 metric | `02_brain_activation/REPORT.md §6` cross-references §05 |
| **Target venue** (locked 2026-05-11) | **IEEE TNSRE** — 8-page limit, IEEE numeric citations, supplementary supported | User decision P0.1 |
| **Chromophore main vs supp** (locked 2026-05-11) | **HbO + HbR in main; HbT supplementary** | User decision P0.2 |
| **SG comparator placement** (locked 2026-05-11) | **Side-by-side ST + SG in main results table** (NOT supplementary) | User decision P0.3 |
| **Citation style** (locked 2026-05-11) | **IEEE numeric** `[1], [2], ...` — applies to `references/refs.bib` and all drafts | User decision P0.4 |
| **mt2 + mt4 in main table** (locked 2026-05-11) | **Both mt2 and mt4 in main results table** (24 rows: HbO+HbR × {5-fold, 10-fold, LOSO} × {ST, SG} × {mt2, mt4}). Tight for IEEE TNSRE 8-page budget — SPEC plan must carefully ration figure count and prose. | User decision (P0.3 follow-up) |

---

# 1. Introduction

## 1.1 Clinical and disease explanation
- **[WRITE]** GAD definition, DSM-5 criteria, lifetime prevalence (~5–6 %), clinical impact, comorbidity profile.
- **Sources required (none in repo):** DSM-5; Bandelow & Michaelis 2015; Kessler 2012 prevalence study.
- **Length target:** ≈ 1 paragraph (150–200 words).

## 1.2 Traditional anxiety assessment methodology
- **[WRITE]** HAMA, STAI-S, STAI-T as gold-standard scales; clinician interview + self-report constraints; subjectivity / state-vs-trait distinction; need for objective biomarkers.
- **Tie-in to this paper's data:** STAI-T is administered to every subject in the cohort (HC mean 33.9 ± 9.6, GAD 54.2 ± 11.6, *d* = 1.92, *p* = 7.07 × 10⁻¹⁰). HAMA is GAD-only (mean 20.8 ± 9.1, n = 28). → `01_demographic/REPORT.md §4.5–§4.6`.
- **Length:** ≈ ½–1 paragraph.

## 1.3 fNIRS compared to other modalities
- **[WRITE]** Comparative table fNIRS vs fMRI / EEG / MEG / PET on:
  spatial resolution, temporal resolution, motion tolerance, ecological
  validity, cost, prefrontal access. Standard reference: Pinti et al. 2020,
  Yücel et al. 2017 (review).
- **Length:** ≈ 1 paragraph + optional comparison table.

## 1.4 Literature review of fNIRS in anxiety disorders
- **[WRITE+TODO]** No bibliography in the repo yet. Survey the prefrontal-fNIRS GAD/SAD literature: Yang 2020, Bauernfeind 2014, Tupak 2014, Husain 2020, Yeung 2020, Pereira 2018. Group the studies along: prefrontal hypoactivation findings, GNG/Stroop paradigms, GNN/CNN-on-fNIRS results.
- **Action item for spec plan:** assemble a dedicated bibliography file (`research/paper-materials/refs.bib`).
- **Length:** ≈ 1.5–2 paragraphs; include 1 summary table of "studies that classified anxiety vs healthy with fNIRS, sample size, accuracy".

## 1.5 Research motivation & significance
### 1.5.1 Graph modality strength to preserve neurophysiology
- **[WRITE]** Argument: fNIRS measures multi-channel time-series whose **spatial topology and pairwise coupling** carry diagnostic information that a flat CNN/MLP discards. A graph encodes channels as nodes and inter-channel coupling as edges, which preserves both. Cite: Velickovic 2018 (GAT), Wu 2021 (GNN review), Saeidi 2022 (graph fNIRS), Mehmood 2024.
- **Length:** ≈ ½ paragraph.

### 1.5.2 AI explainability
- **[WRITE]** Black-box CNN/RNN classifiers cannot give clinicians per-channel or per-region attribution. Graph attention gives both **node** and **edge** importance natively. We use this to bridge the model's prediction back to the prefrontal hemodynamic literature.
- **Tie-in:** Section 3.3 reports channel-level and channel-pair attention; results are then cross-referenced to §3.1 statistical channels (S5_D5, S2_D1, S3_D3, S1_D1, S4_D5, S4_D7).
- **Length:** ≈ ½ paragraph.

### 1.5.3 Contributions (NEW subsection — recommended addition)
- **[TODO]** Itemise 3–4 contributions:
  1. End-to-end leakage-free graph dataset construction (per-fold standardisation).
  2. Two complementary architectures: SG (static, channel-statistic features) and ST (windowed, raw time-series features) on the same prefrontal montage.
  3. Nested-CV Optuna hyperparameter search; reproducible deterministic-split protocol.
  4. Native-attention XAI mapped to Brodmann regions, validated against a parallel statistical-analysis arm (§02 / §06).
- **Length:** bullet list, ½ page.

---

# 2. Materials and Methods

## 2.1 Dataset characteristics

### 2.1.1 Demographic dataset overview
- **[DONE+WRITE]** Numbers ready in `01_demographic/REPORT.md`. Final cohort = **HC 33 / GAD 29 = 62 subjects**.
  - Age: HC 73.0 ± 5.6 yr · GAD 51.1 ± 14.7 yr · Welch *t* = 6.08 · *p* = 6.5 × 10⁻⁶ (large confound — must be acknowledged, see §5.1).
  - Demographics-eligible subset: HC 33, GAD 18 (11 GAD subjects lack demographics — AA089, AA090, AA092, AA093, AA094, AA097, AA098, AA099, LA091, LA095, LA096).
- **Figures:** `01_demographic/fig_age_comparison.png`.
- **Per-subject CSV:** `01_demographic/cohort_per_subject.csv`.

### 2.1.2 Gender composition
- **[DONE+WRITE]** HC: 23 F / 10 M (70 % F) · GAD: 14 F / 4 M (78 % F) · χ²(1) = 0.084 · *p* = 0.772 (n.s.).
- **Figure:** `01_demographic/fig_sex_distribution.png`.

### 2.1.3 Participant characteristics
- **[DONE+WRITE]** Education χ²(3) = 0.78, *p* = 0.855 (n.s.).
- **[DONE+WRITE]** STAI-S/STAI-T separation — Cohen's *d* = 1.34 / 1.92.
- **[DONE+WRITE]** HAMA distribution within GAD (mild 10 / moderate 8 / severe 10).
- **Figures:** `01_demographic/fig_education.png`, `fig_stai_comparison.png`, `fig_hama_gad.png`.
- **Subject-level ground truth file:** `data/subjects_ground_truth.xlsx`.

### 2.1.4 Data-quality summary (NEW — recommended addition)
- **[DONE+WRITE]** Full quality scan in `data/DATA_QUALITY_REPORT.md`. Headline numbers to lift:
  - Pipeline switch: `processed-new` → `processed-new-mc` (Wavelet + CBSI motion correction).
  - Spike-ratio reduction 23 % overall, 31 % in the anxiety group (§9.2).
  - HbO/HbR correlation forced to −1.00 by CBSI (§9.5 → motivates HbT for ML).
  - 2 anxiety subjects (EA012, EA016) recovered by the new pipeline (n=64 GNG epochs in old → 256 in new).
- **Action:** Reproduce key rows of §9.2 as a Methods supplementary table.

## 2.2 Experimental paradigms

### 2.2.1 Sensor–detector placement (prefrontal cortex)
- **[TODO]** Need a figure: 8 sources × 8 detectors → 23 channels. Source/detector positions on the international 10-20 reference using the `data/brainproducts-RNP-BA-128-custom.elc` montage (provides head-coordinates + LPA/Nz/RPA fiducials).
- **Action:** generate the prefrontal layout schematic from `src/xai/atlas.py` (already loads ELC).
- **Reference figure available:** `assets/electrode_*.png` (per memory `project_dataset_state.md`).

### 2.2.2 Channel selection & arrangement on standard optode layout
- **[DONE+WRITE]** Use canonical 23-channel list (single source of truth for SG/ST/XAI/stats):
  ```python
  CHANNEL_NAMES = ['S1_D1','S1_D3','S2_D2','S2_D1','S2_D5','S3_D1','S3_D3',
                   'S3_D4','S3_D6','S4_D4','S4_D5','S4_D7','S5_D2','S5_D5',
                   'S5_D8','S6_D3','S6_D6','S7_D4','S7_D6','S7_D7','S8_D5',
                   'S8_D7','S8_D8']
  GRID_POS = [(0,2),(1,1),(0,4),(0,3),(1,4),(1,2),(2,1),(2,2),(3,1),
              (2,3),(2,4),(3,4),(1,5),(2,5),(3,6),(3,0),(4,1),
              (3,2),(4,2),(4,3),(3,5),(4,4),(4,5)]
  GRID_SHAPE = (5, 7)
  ```
  Source: `docs/SPEC_xai_graph.md §4` and `src/xai/channels.py`.
- **[TODO]** Brodmann area assignment per channel — pull from `research/xai/atlas/channel_to_brodmann.csv` after `04_atlas_registration.ipynb` runs.

### 2.2.3 Dataset collection protocols
#### 2.2.3.1 Go/No-Go neurocognitive protocol
- **[WRITE]** Block design: 4 task blocks × ~32 s @ 10.17 Hz sampling. Acquisition device, NIRS wavelengths, distance source–detector — pull from the lab's existing acquisition documentation.
- **[NEEDS-DATA]** Per-trial Go-vs-No-Go event timing **does not exist** in the project's `.tri`/processed files (only block-level markers). Document this explicitly as a constraint that prevents a true GLM/HRF first-level analysis with a No-Go − Go contrast (`06_glm_hrf/REPORT.md §2`).

## 2.3 Overall workflow pipeline

### 2.3.1 Signal-processing techniques
#### 2.3.1.1 Raw dataset state
- **[DONE+WRITE]** Spike density, HbO/HbR Pearson r, signal scale per group → `data/DATA_QUALITY_REPORT.md §1, §5`.
- **Headline:** Mean HbO/HbR r = +0.21 in raw pipeline (motion + Mayer waves dominating).

#### 2.3.1.2 Why motion correction & noise removal were needed
- **[DONE+WRITE]** Argument:
  - Positive HbO/HbR correlation is unphysical; indicates motion + Mayer-wave dominance.
  - High-spike subjects (HIGH-RISK list, 11 subjects with >3000 spikes) over-represent the anxiety group.
- **Source:** `data/DATA_QUALITY_REPORT.md §2, §7`.

#### 2.3.1.3 State of dataset after correction
- **[DONE+WRITE]** Wavelet (IQR=1.5) + CBSI + bandpass 0.01–0.5 Hz pipeline:
  - Spike ratio ↓ 23 % (31 % in anxiety).
  - r(HbO,HbR) → −1 by construction (CBSI).
  - Motivates **HbT** as the primary ML signal (HbO + HbR; not constrained to −1).
- **Source:** `data/DATA_QUALITY_REPORT.md §9.2, §9.5`.
- **Pipeline schematic asset:** `assets/SG-vs-ST_*.png` (per memory) + Homer3 block-diagram figure to draft.

### 2.3.2 Graph dataset creation

#### 2.3.2.1 1-D signal → graph dataset (shared scaffolding)
- **[TODO+WRITE]** Section needs explicit math. Concretely:
  - Notation: 23 nodes V, edge set E ⊆ V×V, node features X ∈ ℝ^{23 × F}, edge features A ∈ ℝ^{|E| × 2}.
  - Edge construction (shared by SG and ST, source `src/core/dataset.py`, `src/core_st/dataset.py`):
    - Compute Pearson correlation matrix C ∈ ℝ^{23×23} on the raw trial.
    - Compute spectral coherence matrix Coh ∈ ℝ^{23×23} via Welch with `seg_length = N/3`.
    - Edge i→j exists iff |C_ij| ≥ τ_corr (τ = 0.1 default; configurable).
    - Each edge carries a 2-D feature `(Coh_ij, |C_ij|)`.
    - `directed=True` (both i→j and j→i are stored as separate directed edges) and self-loops added when `self_loops=True` (default).
  - **[TODO]** Need formula box + Algorithm 1 pseudocode block.

##### 2.3.2.1.1 Static graph (`src/core/`)
- **Node features** [DONE+WRITE]: x_i = (μ_i, min_i, max_i, skew_i, kurt_i, var_i) ∈ ℝ⁶ over the full 326-sample trial, leak-free fold-wise standardised by `StandardizeGraphFeatures`.
- **Edge features**: identical 2-D coh/|corr| as above.
- **[TODO]** Mathematical definitions for skewness `m₃/σ³` and kurtosis `m₄/σ⁴` (Pearson form, no Fisher correction — verify against `src/core/utils.py:47–50`). Also provide `pearson_correlation_matrix(·)` and Welch-coherence formulas explicitly.
- **Schematic:** `assets/SG-vs-ST_schematic.png` per recent commit `ca6104b`.

##### 2.3.2.1.2 Static-temporal graph (`src/core_st/`)
- **Windowing** [DONE+WRITE]: window size W ∈ {16, 32, 48, 64} samples, stride S ∈ {8, 16, 24, 32}. Best paper config = W=16, S=8 (`config.yaml` from `experiments/spatial_temporal_graph/5-fold/...20260509`). Number of windows K = ⌊(T − W)/S⌋ + 1.
- **Node features** [DONE+WRITE]: input `Data.x = [23, T=326]` is per-channel z-scored per-trial; the model **internally** unfolds it to `[23, K, W]` and emits per-window stats `[23, K, 6] = (mean, min, max, var, skew, kurt)` — **deterministically computed inside `forward()`**, no external feature engineering. Source: `src/core_st/models.py:_compute_window_stats()`. Critical invariant: `in_channels=6` is fixed and is **not** a hyperparameter.
- **Edge features**: same 2-D coh/|corr| as SG, computed once on the full trial (not per-window).
- **[TODO]** Formula box + diagram of the `[23, T] → [23, K, 6]` transform (mirrors `src/core_st/README.md` ASCII diagram lines 27–47).

##### 2.3.2.1.3 Leak-free statistical metrics (the user-flagged invariant)
- **[DONE+WRITE]** `compute_stats(train_indices)` is called per fold/per LOSO holdout to produce `(mean_x, std_x, mean_ea, std_ea)` from the **train graphs only**, then `StandardizeGraphFeatures` applies that to both train and val. This guarantees no validation-set statistics leak into normalisation.
- **Differences SG vs ST in this respect:**
  - SG: standardises both `x` (node features) and `edge_attr`.
  - ST: standardises only `edge_attr` (node `x` is already per-trial z-scored at dataset build time).
- **Source:** `src/core/dataset.py:139` (SG), `src/core_st/dataset.py:139` (ST), `src/core_st/transforms.py`.
- **Action item for spec plan:** add a small Methods box titled "Leakage controls" explicitly listing this fold-wise normalisation contract.

### 2.3.3 Graph-based methodology

#### 2.3.3.1 Spatial Static Graph (SG) architecture — `FlexibleGATNet`
- **Architecture table** [TODO]:
  - Input `Data.x = [23, 6]`, `edge_attr = [|E|, 2]`.
  - GATv2Conv stack with `n_layers=2`, **per-layer** `n_filters=[112, 80]`, `n_heads=[8, 6]`, `edge_dim=2`.
  - `dropout=0.3`, `use_residual=True`, `use_norm=False`, `use_gine_first_layer=False`.
  - Global mean pool → FC(`fc_size=224`) → linear → 2 logits.
  - Source: `experiments/spatial_graph/5-fold/GATv2_GNG_hbo_kfold_mt4_noaug_20260507/config.yaml`.
- **[TODO]** Render this as a 5–7 row table + one schematic figure.
- **Training parameters** [DONE+WRITE]:
  | Param | Value | Source |
  |---|---|---|
  | Epochs | 150 | config.yaml |
  | LR | 6.79 × 10⁻³ | (Optuna `core_hbo_mt4_ep100_tr600_kf5` best) |
  | Scheduler | CosineWarmupScheduler | config.yaml |
  | Batch size | 8 | config.yaml |
  | Early stopping | patience = 9999 (effectively off — full-epoch run) | config.yaml |
  | Random seed | 42 | config.yaml |
  | Optimizer | Adam | `src/core/training.py` |
  | Loss | CrossEntropyLoss (no class weights, no focal loss in headline runs) | config.yaml |
- **Validation note:** patience=9999 was the deliberate choice after the leak-free patience study (see `research/experiments/20260506/leak-free-patience-9999/`).

#### 2.3.3.2 Spatial-Temporal Graph (ST) architecture — `WindowedSpatioTemporalGATNet`
- **Architecture table** [TODO]:
  - Input `Data.x = [23, T=326]`, `edge_attr = [|E|, 2]`.
  - Windowing W=16, S=8 → K = ⌊(326−16)/8⌋ + 1 = 39 windows; per-window stats `[23, K, 6]`.
  - Spatial encoder: GATv2Conv stack (shared across windows) with `n_layers=2`, `n_filters=80` (single int — shared across layers, **unlike SG**), `n_heads=2`, `dropout=0.3`, `use_residual=False`, `use_norm=True`, `norm_type=batch`, `edge_dim=2`.
  - Per-window pool → Linear(pre_gru) → `temporal_hidden=192` → GRU(`temporal_layers=1`) over K windows.
  - Additive attention over K → Linear(`fc_size=256`) → linear → 2 logits.
  - Source: `experiments/spatial_temporal_graph/5-fold/ST_GATv2_GNG_hbo_kfold_mt4_noaug_20260509/config.yaml`.
- **[TODO]** Render as a table + the ASCII pipeline diagram from `src/core_st/README.md` lines 27–47 redrawn as a publication figure.
- **Training parameters** [DONE+WRITE]:
  | Param | Value | Source |
  |---|---|---|
  | Epochs | 150 | config.yaml |
  | LR | 3.04 × 10⁻⁴ | (Optuna `st_hbo_mt4_ep100_tr25_kf5_lr_cosine` Trial #36 best) |
  | Scheduler | CosineAnnealingLR (T_max=150, eta_min=1 × 10⁻⁵) | config.yaml |
  | Batch size | 8 | config.yaml |
  | Early stopping | patience = 30 (paper baseline) | config.yaml |
  | Random seed | 42 | config.yaml |
  | Optimizer | Adam | `src/core_st/training.py` |
  | Loss | CrossEntropyLoss (FocalLoss available but `use_focal_loss=False`) | config.yaml |
- **Provenance pointer:** Optuna result `research/experiments/20260503/optuna_search_nested_validation_st_cosine_annealing/result_report.md` — explains why this LR / scheduler choice is justified by 100 trials × 5-fold nested CV.

#### 2.3.3.3 Architectural diff table (SG vs ST) — RECOMMENDED ADDITION
- **[TODO]** Lift the table from `src/core_st/README.md` "Key Differences" section (lines 228–238). Communicates the design choice cleanly to reviewers.

### 2.3.4 Optuna hyperparameter optimisation

#### 2.3.4.1 Introduction & motivation
- **[WRITE]** Argue: the GAT search space is large (per-layer filter counts, head counts, scheduler families, lr decades, dropout) and grid search is intractable. TPE is sample-efficient.

#### 2.3.4.2 Nested validation (k=5 inner CV per trial)
- **[DONE+WRITE]** Setup: each trial trains the model on k=5 inner folds of the **deterministic split JSON** (`data/splits/kfold_splits_processed_new_mc.json`), reports mean validation F1. The Optuna study optimises **mean 5-fold val F1**.
- **Compute:** 4 GPUs × shared `optuna_journal.log` storage backend, JournalStorage allows multi-process resume.

#### 2.3.4.3 Best configs that drove the paper headline runs
- **[DONE+WRITE]** Two concrete results to cite:
  1. **ST** — `st_hbo_mt4_ep100_tr25_kf5_lr_cosine` (100 completed trials, 5 pruned). Best **Trial #36**: lr = 3.04 × 10⁻⁴, T_max = 150, eta_min = 1 × 10⁻⁵, mean F1 = **0.7693**, per-fold F1 [0.786, 0.764, 0.880, 0.727, 0.690]. Source: `research/experiments/20260503/optuna_search_nested_validation_st_cosine_annealing/result_report.md`.
  2. **SG** — `core_hbo_mt4_ep100_tr600_kf5` (600 trials, full search-space across n_filters/n_heads per layer). Source: `research/experiments/20260430/optuna_search_nested_validation/core_hbo_mt4_ep100_tr600_kf5/optuna_study.db`. **[TODO]** Extract best-trial summary identical in format to the ST report (no `result_report.md` exists yet — create one).
- **[TODO]** A small Methods table with "Search space → Best value → CI/marginal effect".

### 2.3.5 Evaluation framework

#### 2.3.5.1 Performance metrics
- **[WRITE]** Define each metric as used in `src/core_st/training.py` and the Excel sheets:
  - **Accuracy**, **Sensitivity (Recall = TPR for the positive=GAD class)**, **Specificity (TNR for the negative=HC class)**, **Precision**, **F1**.
  - **Per-fold mean ± SD** vs **overall pooled** (overall = computed on the concatenated true/pred labels across all folds — `kfold_overall.pkl`).
  - All metrics two-class binary; positive class = GAD (label=0 anxiety per `kfold_splits…json`).
- **Action:** add a Methods box defining "trial-level" vs "subject-level (future)" evaluation.

#### 2.3.5.2 Validation strategy — emphasise subject-level partitioning
- **[WRITE]** Critical leak-prevention argument: **all 4 trials of a subject go to the same fold**. The split JSON (`kfold_splits_processed_new_mc.json`) lists subjects (not trial indices), preventing any within-subject leakage. Trial-level metrics (no soft/hard voting) are reported as the conservative (lower) bound.
- **Action:** keep the trial-vs-subject-voting distinction explicit; voting analysis is Future Work.

#### 2.3.5.3 K-Fold cross-validation
##### 2.3.5.3.1 5-fold subject-level CV
- **[DONE+WRITE]** Splits in `data/splits/kfold_splits_processed_new_mc.json["kfold_5"]` (62 subjects, ~12–13 per fold).
##### 2.3.5.3.2 10-fold subject-level CV
- **[DONE+WRITE]** Splits in same file, key `kfold_10` (62 subjects, 6–7 per fold).

#### 2.3.5.4 Leave-One-Subject-Out (LOSO)
- **[DONE+WRITE]** Constructed on the fly in `src/core_st/dataset.py:get_loso_splits` (no JSON). 62 folds; each fold trains on 61 subjects and evaluates the 4 held-out trials of the 62nd. Mean ± SD over 62 subjects, plus overall pooled F1 reported.
- **Caveat to flag in the paper:** LOSO `Sens Mean ± SD` columns in the Excel summary show 0.468 ± 0.503 for *every* row — this is because each LOSO fold has a single subject (1 class), so sens/spec/prec is degenerate per fold. **Always quote `Overall Sens / Spec` (computed on concatenated predictions) for LOSO**, not per-fold mean ± SD.

#### 2.3.5.5 Note: deterministic subject-fold JSON
- **[DONE+WRITE]** Reproducibility statement: random_state=42 across all configs; splits file checked into `data/splits/`; metadata block in the JSON documents `generated_at = 2026-04-30`, `seed = 42`, `total_subjects = 62`, class distribution {anxiety: 29, healthy: 33}.

---

# 3. Results

> **Source-of-truth files for the headline numbers**
> - SG metrics: `experiments/spatial_graph/experiment_metrics.xlsx` (sheets: 5-Fold Summary, 10-Fold Summary, LOSO Summary).
> - ST metrics: `experiments/spatial_temporal_graph/experiment_metrics.xlsx` (same sheet structure).
> - Per-fold pickles: `experiments/<arch>/<regime>/<run_name>/<run_name>_fold_<F>.pkl` (numpy arrays + lists of true/pred labels).
> - Best-config copies of `config.yaml` are co-located in every run folder.

## 3.1 Statistical analysis

### 3.1.1 Demographics
- **[DONE+WRITE]** Already covered in §2.1; do not duplicate the table — instead refer back. Optionally show one summary figure here as an at-a-glance.

### 3.1.2 Ground-truth labels (HAMA, STAI-S, STAI-T)
- **[DONE+WRITE]** Source: `01_demographic/REPORT.md §4.5–§4.6` and `01_demographic/cohort_per_subject.csv`.
- Headline: STAI-S Cohen's *d* = 1.34, STAI-T *d* = 1.92 (validates clinical separation of the two groups). HAMA distribution within GAD: 10 mild / 8 moderate / 10 severe (LA063 excluded — administrative zero).
- **Figure to lift:** `01_demographic/fig_stai_comparison.png`.

### 3.1.3 HC vs GAD per-channel HbO activation (primary chromophore)
- **[DONE+WRITE]** Source: `02_brain_activation/REPORT.md`. Headline:
  - 10 / 23 channels uncorrected sig; **4 / 23 FDR-significant**: **S5_D5, S2_D1, S3_D3, S1_D1** (all HC > GAD; |Cohen's *d*| 0.77–0.92).
  - Whole-cortex grand-mean HbO STD: HC 0.903 ± 0.031 vs GAD 0.886 ± 0.028 · *U* = 643 · *p* = 0.021.
  - Channels cluster on the superior-medial quadrant of the 5×7 grid.
- **Figures to lift:** `fig_topo_activation.png`, `fig_topo_effect_size.png`, `fig_gng_sig_channels.png`, `fig_task_grand_mean.png`.
- **CSV:** `02_brain_activation/results_brain_activation_stats.csv`.

### 3.1.4 GLM/HRF analysis (combine with 3.1.3 if correlated)
- **[DONE+WRITE]** Source: `06_glm_hrf/REPORT.md`. Block-level analog of a first-level GLM (event-level GLM not feasible — see §2.2.3.1). Headline:
  - **2 / 23 FDR-significant canonical-HRF β channels: S4_D5, S4_D7** (Cohen's *d* −0.89, −0.72).
  - Direction: **HC β > GAD β** at every uncorrected-significant channel.
  - 13 significant clusters across 9 channels in two windows: **early (0–7 s)** and **late (21–32 s)**, with the late window dominant (HC ramps up across the 32-s task block; GAD does not sustain).
  - §02 STD vs §06 canonical-β rankings: Spearman ρ = +0.19 — **the two methods tag complementary channels**. Combined "biologically-relevant channel set" (both methods, raw α=0.05) = **{S5_D5, S2_D1, S3_D3, S1_D1, S4_D5, S4_D7}** — this is the C6 acceptance set in `docs/SPEC_xai_graph.md §11`.
- **Decision:** Report §3.1.3 and §3.1.4 as **complementary**, not redundant — the combined channel set above is the biological prior the XAI section will cross-validate.
- **Figures to lift:** `06_glm_hrf/fig_evoked_top_channels.png`, `fig_topo_compare_metrics.png`.

### 3.1.5 (RECOMMENDED ADDITION) Demographic age confound acknowledgement
- **[DONE+WRITE]** Pull §6 caveat from `02_brain_activation/REPORT.md`: top-4 FDR channels remain raw-significant after age ANCOVA on the n=51 demographics-eligible subset (S5_D5: β = −0.048, *p* = 0.0023, η²ₚ = 0.18). Source: `05_age_adjusted/` notebooks.
- Position: end of §3.1, two sentences. Required for reviewer defensibility.

## 3.2 Experimental results — graph models

> **Coverage:** 6 chromophore × max-trial cells × 3 CV regimes × 2 architectures = **36 cells**. Excel sheets contain all 36 rows (HBO/HBR/HBT × mt2/mt4 × {5-fold, 10-fold, LOSO} × {SG, ST}).
> Reporting plan: one **Main Results table** in the paper covering ST in full (18 rows). SG is reported as the comparator in a side-by-side mini-table or in supplementary.

### 3.2.1 Headline numbers from `experiments/spatial_temporal_graph/experiment_metrics.xlsx`

#### 3.2.1.1 ST 5-fold (mean ± SD across 5 folds)
| Hb | mt | Acc | Sens | Spec | Prec | F1 | Overall F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HbO | 2 | 0.749±0.125 | 0.900±0.137 | 0.631±0.213 | 0.694±0.149 | 0.773±0.109 | 0.7704 |
| HbO | 4 | 0.733±0.077 | 0.880±0.052 | 0.608±0.151 | 0.672±0.102 | 0.757±0.059 | 0.7556 |
| HbR | 2 | 0.741±0.128 | 0.930±0.071 | 0.581±0.252 | 0.684±0.163 | 0.777±0.097 | 0.7714 |
| HbR | 4 | 0.692±0.087 | 0.932±0.055 | 0.486±0.124 | 0.617±0.079 | 0.740±0.067 | 0.7397 |
| **HbT** | **2** | **0.774±0.117** | 0.913±0.102 | 0.660±0.201 | 0.714±0.146 | **0.794±0.102** | **0.7910** |
| HbT | 4 | 0.712±0.112 | 0.900±0.081 | 0.558±0.240 | 0.661±0.140 | 0.750±0.073 | 0.7455 |

*HbT mt2 wins in 5-fold (F1 = 0.794).*

#### 3.2.1.2 ST 10-fold
- HbT mt2 again wins (F1 = 0.805 ± 0.113; Overall F1 = 0.7941). HbO mt2 = 0.798 (close second). Full row table to be lifted from sheet `10-Fold Summary`.

#### 3.2.1.3 ST LOSO (62 subjects)
| Hb | mt | Overall Acc | Overall Sens | Overall Spec | Overall Prec | **Overall F1** |
|---|---:|---:|---:|---:|---:|---:|
| HbO | 2 | 0.7903 | 1.0000 | 0.6061 | 0.6905 | 0.8169 |
| HbO | 4 | 0.7702 | 0.9914 | 0.5758 | 0.6725 | 0.8014 |
| **HbR** | **2** | **0.8226** | **1.0000** | **0.6667** | **0.7250** | **0.8406** |
| HbR | 4 | 0.8145 | 0.9914 | 0.6591 | 0.7188 | 0.8333 |
| HbT | 2 | 0.7823 | 1.0000 | 0.5909 | 0.6824 | 0.8112 |
| HbT | 4 | 0.7540 | 0.9655 | 0.5682 | 0.6627 | 0.7860 |

*LOSO winner: **HbR mt2, F1 = 0.8406***. (Note: per-fold mean ± SD columns in this sheet are degenerate for LOSO — see §2.3.5.4 caveat. Always quote `Overall *` for LOSO.)

### 3.2.2 SG comparator results (`experiments/spatial_graph/experiment_metrics.xlsx`)
- **[DONE+WRITE]** Best SG cell:
  - 5-fold: HbO mt2 F1 = 0.759 ± 0.068 (Overall 0.7534). SG 5-fold weakest = HbT mt4 F1 = 0.678.
  - 10-fold: HbR mt2 F1 = 0.807 ± 0.075 (Overall 0.8000) — note this beats some ST 10-fold cells; SG is still competitive at 10-fold but *consistently* loses at LOSO.
  - LOSO: HbR mt2 F1 = 0.7838 (Overall) — vs ST HbR mt2 = 0.8406. ST wins LOSO by **+5.7 pp**.

### 3.2.3 Best-architecture / best-cell narration
- **[WRITE]** Headline: **ST > SG**, with the gap most pronounced at LOSO (the strictest leakage-control regime). The single best paper number is **ST × HbR × mt2 × LOSO = F1 0.8406**.
- **Tie-in to memory:** project memory `project_st_vs_sg_validation.md` records this finding (ST wins; LOSO mt2 leak Δ ≈ −30 pp historically — current 20260509 sweep is leak-free).
- **Action items:**
  1. Decide whether to put the full 36-row table in main text or split (recommend: ST 18 rows main, SG 18 rows supplementary, plus a comparison summary figure in main).
  2. **[TODO]** Compute and report a paired statistical test (DeLong on overall ROC, or paired Wilcoxon on per-fold F1) to back the "ST beats SG" claim.
  3. **[TODO]** Confusion-matrix figure for the headline cell (ST × HbR × mt2 × LOSO) — pickled in `experiments/spatial_temporal_graph/loso/ST_GATv2_GNG_hbr_loso_mt2_noaug_20260509/...kfold_overall.pkl`.

### 3.2.4 (RECOMMENDED ADDITION) Training-curve figure for the headline cell
- **[TODO]** Render F1/loss/accuracy curves from per-fold PNGs already produced (`*_fold_<F>_f1.png`, `_loss.png`, `_accuracy.png` co-located with the pickles) → 5×3 grid figure → 1 supplementary panel.

## 3.3 AI explainability (single best-performance model)

### 3.3.1 Decision: which model + which signal as the XAI base
- **[DECISION + DONE]** ST (winner of §3.2) × **HbR mt2 LOSO** = paper-headline XAI subject. SPEC §2.1 (rev. 6) covers HbO/HbR/HbT for ST; HbR is the headline.
- **Output paths (after `02_st_population.ipynb`):**
  - Native-attention CSVs: `research/xai/st/hbr/{kfold-5, kfold-10, loso}/mt2/native/{node_importance.csv, edge_importance.csv, channel_pair_matrix.npy, temporal_attention.csv, ...}`
  - GNN-explainer cross-check CSVs: same path with `supplementary/`.
  - Plots: `fig_montage_channel_importance.png`, `fig_pair_matrix.png`, `fig_temporal_attention.png`.

### 3.3.2 Channel importance (node-level)
- **[DONE+WRITE]** Methodology to describe (already implemented in `src/xai/st_explainer.py`):
  - Native: aggregate spatial GATv2 attention across (windows × layers × heads), pool to per-node importance, normalise, average over correctly-classified validation trials.
  - Supplementary: PyG `GNNExplainer` with `node_mask_type='object'` for cross-check.
- **[DONE+WRITE]** Acceptance criterion C6 (SPEC §11): **≥ 2 of {S1_D1, S5_D5, S3_D3, S2_D1, S4_D5, S4_D7} appear in top-10**. This is the biological prior derived from §3.1.3 + §3.1.4.
- **Figure plan:** 5×7 grid topomap of per-channel importance with the C6 channels marked.

### 3.3.3 Channel-to-channel importance (edge-level)
- **[DONE+WRITE]** 23 × 23 channel-pair matrix from edge attention (native) and edge-mask (supplementary). Already a SPEC deliverable: `channel_pair_matrix.npy`. Methodology: edges aggregated across windows × layers, symmetrised since SG/ST treat directed = True.
- **[DONE+WRITE]** Top-K pair list to enumerate (K = 10) and a chord/heatmap figure.

### 3.3.4 Temporal attention (ST native — UNIQUE TO ST)
- **[DONE+WRITE]** `temporal_attention.csv` per cell = `[K = 39]` softmax weights over windows. Plot weight vs window index; expected: emphasis on **late** windows (~21–32 s) consistent with the §3.1.4 cluster-permutation finding.
- **Action item:** This is a strong figure; ensure it lands in the main text.

### 3.3.5 Brodmann-region cross-reference
- **[DONE+WRITE]** Region-level re-aggregation produces `region_importance.csv` and `region_pair_matrix.npy` per cell (output of `04_atlas_registration.ipynb`). Sanity check C8: S2_D1 → BA10 probability ≥ 0.5.
- **Expected narration:** the highest-importance regions cluster in **medial PFC (BA 10) / dlPFC (BA 9, 46)**; cross-reference Yeung 2020, Tupak 2014 GAD-PFC findings.

### 3.3.6 Convergence narrative
- **[DONE+WRITE]** The hypothesis the user wants to validate ("explainability matches the actual GAD region/channels") is testable in **two** quantitative checks:
  1. **C6 (channel level):** ≥ 2 / 6 statistical channels in top-10 attention (expected to pass — already passing in pilot smoke runs per `00_setup_and_smoke.ipynb`).
  2. **Spearman ρ between attention rank and |Cohen's d| rank** across 23 channels, separately for HbO and HbR.
- **[TODO]** Add a small "concordance" sub-figure (saliency vs |d| topomap triptych) — same idea as `FUTURE_ANALYSES.md §1.4`, but already feasible from the existing artefacts.

---

# 4. Discussion

## 4.1 Discussion plan
- **[WRITE — POPULATE LAST]** As the user noted, this section is derived from §3 and only finalised after the Results are locked. Skeleton bullets for now:
  1. Summary of the headline finding (ST × HbR × mt2 × LOSO = 0.84 F1) and what it implies for fNIRS-based GAD biomarker development.
  2. Why ST > SG on LOSO specifically: temporal dynamics in the late HRF window (§3.1.4) are present in raw [23, T] time series but discarded by SG's STD-collapse.
  3. Convergence of three independent evidence streams: (a) per-channel HbO statistics §3.1.3, (b) canonical-HRF β maps §3.1.4, (c) GNN attention §3.3 — all point to a connected superior-medial PFC cluster (S5_D5, S2_D1, S3_D3, S1_D1, S4_D5, S4_D7) ↔ BA 10 / 9 / 46. This is the strongest defence against "the model could be picking up artifact".
  4. Comparison to existing fNIRS-anxiety classifiers (literature review §1.4) — accuracy is competitive but the **per-region attribution** is the differentiator.
  5. Practical implications: graph-attention XAI is clinically deployable (no extra forward pass cost beyond the model itself).

## 4.2 (RECOMMENDED ADDITION) Threats to validity
- **[WRITE]** Briefly rebut age confound, demographics-missing-11 confound, no per-stimulus event timing, single-site cohort (n=62) — each rebuttal points to a Future Work item in §5.

---

# 5. Limitations and Future Work

## 5.1 Age distribution of the dataset (already in user outline)
- **[DONE+WRITE]** Quote `01_demographic/REPORT.md §6` and `02_brain_activation/REPORT.md §6` and `FUTURE_ANALYSES.md §1.1`: HC mean age 22 yr older than GAD; ANCOVA reduces but does not eliminate the §02 effect; dedicated age-matched-subsample analysis would be the gold standard.

## 5.2 (RECOMMENDED ADDITIONS — pulled from `FUTURE_ANALYSES.md`)
- **[WRITE]** Each one a 1-sentence acknowledgement, not a deep dive:
  1. **Per-stimulus event timing** missing → no true Go-vs-No-Go GLM contrast (§2.2.3.1). Future: re-collect with E-Prime / PsychoPy logs to enable `mne_nirs.first_level.run_glm()`.
  2. **LOSO sensitivity cohorts** (drop AH024, AH029, LA063, demographics-missing-11) not run as headline — `FUTURE_ANALYSES.md §1.3`.
  3. **Permutation classifier null + post-hoc power** — `FUTURE_ANALYSES.md §2.7`.
  4. **Mixed-effects on trial-level data** — `FUTURE_ANALYSES.md §2.1`.
  5. **TFCE / cluster-permutation correction** to replace BH-FDR — `FUTURE_ANALYSES.md §2.2`.
  6. **FC-based GNN edges** (currently spatial proximity only) — `FUTURE_ANALYSES.md §2.3`.
  7. **Channel-ablation studies** (necessity / sufficiency) — `FUTURE_ANALYSES.md §2.4`.
  8. **Subject-level voting** (soft / hard) — currently trial-level only.
  9. **MCX Monte Carlo atlas mapping** to replace midpoint projection — `docs/SPEC_xai_graph.md §16.10`.
  10. **External cohort generalisation probe** — `FUTURE_ANALYSES.md §3.4`.
  11. **fNIRS-BIDS QC reporting** — `FUTURE_ANALYSES.md §2.5`.
  12. **Cross-cohort replication on `processed-old`** — `FUTURE_ANALYSES.md §3.2`.

---

# 6. Conclusions
- **[WRITE — POPULATE LAST]** 1 paragraph. Skeleton:
  1. Restate the contribution: a graph-attention pipeline that classifies HC-vs-GAD on prefrontal fNIRS during a Go/No-Go task at LOSO F1 = 0.84 (ST × HbR × mt2).
  2. Restate the convergence: GNN attention agrees with classical statistical activation maps (3+ independent streams) on the same connected superior-medial PFC cluster (BA 10/9/46).
  3. Briefly point to the most important future work (event-timing GLM + external cohort).

---

# Appendices / Supplementary (recommended)

## A. Full per-(arch × Hb × mt × CV) results table
- **[DONE+WRITE]** Render all 36 rows from the two Excel files as a single supplementary table (`SI_Table_S1.csv`).

## B. Optuna search reports
- **[DONE+TODO]** ST: lift `research/experiments/20260503/.../result_report.md` directly. SG: **needs** an analogous report from `research/experiments/20260430/.../optuna_study.db` — currently has only `progress.log` and `optuna_detailed.log`. **[TODO]** Write `research/experiments/20260430/.../result_report.md`.

## C. Data-quality report excerpts
- **[DONE]** Lift §9.2, §9.3, §9.5, §9.6 of `data/DATA_QUALITY_REPORT.md`.

## D. Per-fold F1 / loss / accuracy curves for the headline ST cell
- **[DONE]** All PNGs already in `experiments/spatial_temporal_graph/loso/ST_GATv2_GNG_hbr_loso_mt2_noaug_20260509/*.png`. Bundle into one figure.

## E. XAI saliency vs statistical-d concordance figure
- **[TODO]** Single triptych: |d|, ST attention (HbR), ST attention (HbO) — same 5×7 grid layout — supports §3.3.6.

---

# Open questions for the spec-plan author

These need answers before `PAPER_SPEC_PLAN.md` can lock down section lengths, figure counts, and final venue.

1. **Target venue.** IEEE TBME/TNSRE? Frontiers in Neuroscience / NeuroImage Reports? MICCAI/EMBC conference? — drives word count, figure budget, supplementary policy.
2. **HbO vs HbR vs HbT in main vs supplementary.** Recommended: HbO (matches statistical analysis) + HbR (model winner) in main; HbT in supplementary. Confirm.
3. **SG in main vs supplementary.** Recommended: SG comparator → supplementary table; ST → main. Confirm.
4. **Atlas figure ownership.** §3.3.5 needs the fsaverage 3-D surface render — pull from `04_atlas_registration.ipynb` or commission a redrawn vector version.
5. **Statistical concordance figure (§3.3.6 / Appendix E).** Confirm whether this becomes a main-text panel or supplementary.
6. **Bibliography.** Confirm citation style (APA / IEEE / Vancouver) so the bibliography file is initialised correctly.
7. **Co-authorship / acknowledgements.** Out of scope for the technical outline but flag for the spec plan.

---

# Quick provenance pointers (for grep-friendly access)

| Asset | Path |
|---|---|
| Final SG metrics | `experiments/spatial_graph/experiment_metrics.xlsx` |
| Final ST metrics | `experiments/spatial_temporal_graph/experiment_metrics.xlsx` |
| Headline ST cell (LOSO HbR mt2) | `experiments/spatial_temporal_graph/loso/ST_GATv2_GNG_hbr_loso_mt2_noaug_20260509/` |
| Headline SG cell (LOSO HbR mt2) | `experiments/spatial_graph/loso/GATv2_GNG_hbr_loso_mt2_noaug_20260507/` |
| Best ST hyperparameters report | `research/experiments/20260503/optuna_search_nested_validation_st_cosine_annealing/result_report.md` |
| Best SG hyperparameters DB | `research/experiments/20260430/optuna_search_nested_validation/core_hbo_mt4_ep100_tr600_kf5/optuna_study.db` |
| Subject-fold splits JSON | `data/splits/kfold_splits_processed_new_mc.json` |
| Per-subject ground truth | `data/subjects_ground_truth.xlsx` (also `01_demographic/cohort_per_subject.csv`) |
| Data quality report | `data/DATA_QUALITY_REPORT.md` |
| Statistical analysis suite | `src/notebook/statistical-analysis/{01_demographic, 02_brain_activation, 03_hb_type_comparison, 04_severity_correlation, 05_age_adjusted, 06_glm_hrf}/REPORT.md` |
| Future analyses roadmap | `src/notebook/statistical-analysis/FUTURE_ANALYSES.md` |
| XAI SPEC (rev. 6) | `docs/SPEC_xai_graph.md` |
| XAI driver notebooks | `src/notebook/xai/{00..04}*.ipynb` |
| XAI building blocks | `src/xai/{channels, checkpoints, sg_explainer, st_explainer, aggregate, visualize, atlas, io, config}.py` |
| Core SG code | `src/core/{models, dataset, training, transforms, optuna_search}.py` |
| Core ST code | `src/core_st/{models, dataset, training, transforms, optuna_search}.py` |
| Core ST README | `src/core_st/README.md` |
| Optode geometry | `data/brainproducts-RNP-BA-128-custom.elc` |
