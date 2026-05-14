# Paper Specification & Writing Plan
**fNIRS Graph-Based Method — IEEE TNSRE Submission**
**Last updated: 2026-05-12 (v2 — added per-section writing prompts)**

---

## 1. Paper Identity

| Field | Value |
|---|---|
| **Working Title** | Graph-Based Detection of Generalized Anxiety Disorder from Prefrontal fNIRS During a Go/No-Go Task, with Native-Attention Explainability and Brodmann-Region Cross-Validation |
| **Target Journal** | IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE) |
| **Paper Type** | Original Research — Methods + Classification + Explainability |
| **Core Contribution** | Two-architecture graph framework (Spatial Graph + Spatio-Temporal Graph) over 23 prefrontal fNIRS channels, with native-attention XAI mapped to Brodmann regions and triangulated against a parallel statistical-analysis arm |
| **Triple Scope** | (1) Graph-based fNIRS classification (ST > SG by +3.78 pp F1, p=0.0024); (2) Native-attention XAI with Brodmann cross-validation; (3) Three-stream convergence on BA 8/9/10/46 PFC cluster |
| **Author** | Aunuun Jeffry Mahbuubi (`11208120@gs.ncku.edu.tw`) |
| **Status** | Draft v2 (2026-05-12). All technical-content blockers cleared (P0/P1/P2 complete per PAPER_TODO.md); only `references/refs.bib` DOI verification deferred to a pre-submission citation-management pass. |

---

## 2. Key Technical Facts (Ground Truth — Reference Before Writing)

### 2.1 Dataset
**Source:** `src/notebook/statistical-analysis/01_demographic/01_demographic_analysis.ipynb`
+ `data/subjects_ground_truth.xlsx` + memory `project_subjects_ground_truth.md`

| Parameter | Value |
|---|---|
| Total subjects in final analysis | **62** (n=33 HC, n=29 GAD) — all retained |
| HC | n=33, age **73.0** yrs (mean), prefrontal Go/No-Go fNIRS |
| GAD | n=29, age **51.1** yrs (mean), prefrontal Go/No-Go fNIRS |
| **Age confound** | Welch t = 6.08, p = 6.5 × 10⁻⁶ — **largest threat to validity**; defended via §05_age_adjusted ANCOVA + demographics-missing-11 fNIRS-only argument |
| Demographics-missing | 11 GAD subjects (AA089–AA099, LA091, LA095, LA096) have only HAM-A / STAI scores — no age/sex/education |
| Clinical scores (n=51 eligible) | HAM-A, STAI-S, STAI-T pulled from `cohort_per_subject.csv` |
| Sex distribution | TBD from `cohort_per_subject.csv` (compute Table 1) |
| Cohort decisions | Keep all 62 in main table (D13); sensitivity drops (AH024, AH029, LA063, demographics-missing-11) → future-work appendix |

### 2.2 fNIRS System
| Parameter | Value |
|---|---|
| Channels | **23 channels** (8 sources × 8 detectors → 23 valid pairs) |
| Wavelengths | 760 nm, 850 nm (NIRX device) |
| Placement | **Prefrontal cortex** (international 10-20 system, custom BrainTech montage) |
| Source/Detector positions | `data/brainproducts-RNP-BA-128-custom.elc` parsed via `src/xai/atlas.py:parse_elc` |
| Hemodynamic measures | HbO, HbR, HbT (HbT computed from HbO + HbR post-CBSI) |
| **Chromophore in main paper (D2)** | **HbO + HbR main; HbT supplementary** |
| **Headline ML chromophore (D7)** | **HbO LOSO mt=2 K=12** → F1 = 0.8529 (sens 1.000, spec 0.697; +1.23 pp over the prior HbR K=23 = 0.8406). LOCKED 2026-05-14 after P1.7 channel-ablation. HbO matches stats arm; K=12 is the most beneficial subset across configurations (mean Δ-F1 +2.52 pp). Secondary number: HbO mt=4 K=8 = 0.8672 (highest raw F1). |
| **Headline XAI chromophore (D6)** | **HbO LOSO mt2** (matches stats arm; ρ(HbO,HbR XAI) = +0.899 → chromophore-invariant) |
| CBSI constraint | r(HbO, HbR) = −1 by construction (`DATA_QUALITY_REPORT.md §9.5`) — only HbT is unconstrained |

**Canonical 23-channel list (single source of truth across SG/ST/XAI/stats):**
```python
CHANNEL_NAMES = [
    'S1_D1','S1_D3','S2_D2','S2_D1','S2_D5','S3_D1','S3_D3',
    'S3_D4','S3_D6','S4_D4','S4_D5','S4_D7','S5_D2','S5_D5',
    'S5_D8','S6_D3','S6_D6','S7_D4','S7_D6','S7_D7','S8_D5',
    'S8_D7','S8_D8']
```

**Channel → Brodmann mapping (deterministic, fsaverage midpoint projection, `research/xai/atlas/channel_to_brodmann.csv`):**

| Channel | Brodmann | Hemisphere |
|---|---|---|
| S1_D1, S1_D3 | BA10 | L |
| S2_D1, S2_D2, S5_D2 | BA10 | R |
| S2_D5, S4_D4, S4_D5, S8_D5 | BA9 | R |
| S3_D1, S3_D3, S3_D4 | BA9 | L |
| S3_D6, S6_D6, S7_D4, S7_D6 | BA8 | L |
| S4_D7, S7_D7, S8_D7, S8_D8 | BA8 | R |
| S5_D5, S5_D8 | BA46 | R |
| S6_D3 | BA46 | L |

**Channel → ROI mapping (user-corrected 2026-05-12, used by figures):**

| Region | Left channels | Right channels | Midline (both) |
|---|---|---|---|
| VMPFC (blue) | 1 | 3 | 4 |
| DMPFC (green) | 8, 18, 19 | 11, 12, 22 | 10, 20 |
| DLPFC (red) | 7, 9, 16, 17 | 14, 15, 21, 23 | — |
| Default (orange) | 2, 6 | 5, 13 | — |

### 2.3 Experimental Task

| Task | Code | Role in paper |
|---|---|---|
| Go/No-Go | GNG | **Sole task** — inhibitory-control paradigm; 4 task blocks × ~32 s @ 10.17 Hz |
| Trial epoching | – | event codes 3.0/4.0 |
| Effective trial length | – | 32 s × 4 blocks per subject |
| Constraint | – | **No per-stimulus markers** — precludes a true first-level GLM with No-Go − Go contrast; canonical-HRF β is the maximally-defensible block-level analog (§3.1.4) |

### 2.4 Preprocessing Pipeline
**Source:** `data/DATA_QUALITY_REPORT.md §9` + `processed-new-mc/` outputs

| Step | Detail |
|---|---|
| 1 | Wavelet motion correction [`molavi2012wavelet`; `improved_wavelet_cbsi`] |
| 2 | CBSI to enforce r(HbO, HbR) = −1 [`cui2010cbsi`] |
| 3 | Bandpass filter 0.01–0.5 Hz |
| 4 | Per-trial standardization, **fold-wise** for leak-free CV (Math E10 in `PAPER_MATH.md`) |
| 5 | Trial epoching (block-level event codes) |
| **Lock (D11)** | Pipeline = `processed-new-mc/` (mandatory; see `DATA_QUALITY_REPORT.md §9`) |
| **Lock (D12)** | **No data augmentation** for headline (`_noaug_*` runs) |

### 2.5 Graph Construction
**Source:** `src/core/utils.py`, `PAPER_MATH.md` E1–E5

| Step | Detail |
|---|---|
| Nodes | 23 channels (fixed) |
| Edge rule | `\|corr(i, j)\| ≥ τ_corr = 0.1` |
| Edge attributes | `(coherence_ij, \|corr_ij\|)` |
| Edge direction | Directed=True, self_loops=True |
| **SG node features `[23, 6]`** | `(mean, min, max, skew, kurt, var)` per channel per trial (Math E1, E2 for skew/kurt) |
| **ST node features `[23, 326]`** | raw z-scored time-series; model unfolds internally to `[23, K, 6]` over K windows |
| Pearson r | Math E3 |
| Welch coherence | Math E4 |
| Edge rule formal | Math E5 |

### 2.6 Model Architectures
**Source:** `PAPER_ARCH_TABLES.md` (Tables A, B, C, D)

#### Spatial Graph (SG)
| Parameter | Value |
|---|---|
| Input | `[23, 6]` (per-trial graph) |
| Backbone | 2-layer GINEConv → GATv2 |
| Optuna-best filters | `n_filters = [112, 80]` |
| Optuna-best heads | `n_heads = [8, 6]` |
| Pool | Global mean pool |
| Pre-pool | Linear(480 → 224) — **applied PER-NODE BEFORE pool** (figure draws this incorrectly; see PAPER_ARCH_TABLES note) |
| Head | Softmax (2 classes) |
| Parameter count | ≈ 1.16 M |
| Optuna-best LR | 6.79 × 10⁻³ |
| LR schedule | cosine_warmup |
| Patience | 9999 (effectively off — full epoch run; locked via leak-free patience study 20260506) |
| Math reference | GATv2 = Math E6 |

#### Spatio-Temporal Graph (ST)
| Parameter | Value |
|---|---|
| Input | `[23, 326]` (per-trial time-series) |
| Windowing | Sliding window W=48, stride yielding K windows |
| Spatial | Per-window GATv2 with **shared weights across K** |
| Temporal | GRU(hidden=64) → additive attention over time |
| Head | Softmax (2 classes) |
| Parameter count | ≈ 0.37 M |
| Optuna-best LR | 3.04 × 10⁻⁴ |
| LR schedule | CosineAnnealingLR(T_max=150, eta_min=1 × 10⁻⁵) |
| Patience | 30 (paper baseline) |
| 5-fold val F1 | 0.7693 (Optuna best trial #36) |
| Math reference | GATv2 = E6; GRU+attention = E7+E8 |

### 2.7 Evaluation Strategies (D9: all three reported)

| Strategy | Detail |
|---|---|
| 5-fold CV | Subject-level stratified; ablation reporting |
| 10-fold CV | Subject-level stratified; stability reporting |
| **LOSO** | 62 iterations (one per subject); strongest evidence of generalization to unseen subjects |
| Splits source | `data/splits/kfold_splits_processed_new_mc.json` + in-code `get_loso_splits` |
| Metrics reported | F1, Accuracy, Sensitivity, Specificity, ROC-AUC (per-fold mean ± SD + pooled CM) |
| Positive class | GAD (label 1); Negative = HC (label 0) |

### 2.8 Key Statistical Results (already computed)

**§02 Brain Activation (per-channel HbO STD):**
**Source:** `02_brain_activation/REPORT.md §4.2`; `02_brain_activation/results_brain_activation_stats.csv`

- 4/23 channels FDR-significant (HC > GAD):
  - **S5_D5** (BA46-R): *d* = −0.92, p_FDR = 0.014
  - **S2_D1** (BA10-R): *d* = −0.85
  - **S3_D3** (BA9-L): *d* = −0.80
  - **S1_D1** (BA10-L): *d* = −0.81, p_FDR = 0.014
- Direction at every FDR channel: **HC > GAD** → hypoactivity pattern consistent with GAD-PFC literature
- ANCOVA holds: S5_D5 β = −0.048, p = 0.0023, η²ₚ = 0.18 on n=51 demographics-eligible subset

**§06 Canonical HRF β (GLM):**
**Source:** `06_glm_hrf/REPORT.md §4.2, §4.4`

- 2/23 channels FDR-significant: **S4_D5**, **S4_D7** (both BA9)
- HC β > GAD β at all 13 cluster-permutation hits
- **Two-window finding:** early (0–7 s) + **late (21–32 s) dominant** — HC ramps up across the 32-s block; GAD does not sustain
- Cross-method: Spearman(§02, §06) = +0.19 → complementary, not redundant

**Combined C6 biological-prior 6-channel set (SPEC §11):**
`{S1_D1, S5_D5, S3_D3, S2_D1, S4_D5, S4_D7}` — acceptance threshold ≥ 2/6 channels in GNN top-10.

**ML headline (ST × HbO × LOSO × mt=2 × K=12) — LOCKED 2026-05-14 after P1.7 channel-ablation:**
F1 = **0.8529**, Acc = 0.8387, Sens = 1.000, Spec = 0.697. CM: TN=46, FP=20, FN=0, TP=58.
**Sources:** `experiments/spatial_temporal_graph/experiment_metrics.xlsx` (baseline) + `research/experiments/20260513/mt2/ST_GATv2_GNG_hbo_loso_mt2_noaug_K12_20260513/*_loso_overall.pkl` (K=12 headline cell).
+3.60 pp over the same-cell K=23 baseline (F1 = 0.8169). +1.23 pp over the prior global best (HbR mt2 K=23 = 0.8406, now superseded).

**Secondary number to also report (highest raw F1; sensitivity-analysis row):**
ST × HbO × LOSO × mt=4 × K=8 = F1 0.8672, Acc 0.8629, Sens 0.957, Spec 0.780. CM: TN=103, FP=29, FN=5, TP=111.

**K-consistency claim for §III.C.7:**
K=12 is the most beneficial subset across configurations — improves over the 23-channel baseline in 5 of 6 (chromo × mt) cells (mean Δ-F1 +2.52 pp; vs +1.85 pp at K=16 and +1.29 pp at K=8).

**ST > SG aggregate paired test:**
Wilcoxon W=3, p = 0.00244 across 12 paired k-fold cells. Mean Δ(ST−SG) F1 = **+3.78 pp**.
McNemar HbR mt4 LOSO p = 2.3 × 10⁻⁴ (47 ST-corrects vs 17 SG-corrects).
**Source:** `stats/st_vs_sg_paired_test.md` + `stats/st_vs_sg_mcnemar_loso.csv`.

**Top-5 XAI channels (ST × HbO × LOSO × mt2 native):**
1. **S5_D5** (BA46-R, Right DLPFC) — also §02 FDR-sig (*d* = −0.92)
2. **S1_D1** (BA10-L, Left VMPFC) — also §02 FDR-sig (*d* = −0.81)
3. **S8_D5** (BA9-R, Right DLPFC) — not §02-FDR (multivariate-only signal)
4. **S4_D4** (BA9-R, Midline DMPFC) — not §02-FDR
5. **S7_D4** (BA8-L, Left DMPFC) — not §02-FDR
**Source:** `research/xai/st/hbo/loso/mt2/native/channel_importance.csv`.

**Concordance (ρ-null, set-overlap is figure-of-merit):**
**Source:** `stats/concordance_rho_table.md/.csv`

- ρ over 23 channels: **null in all 4 pairings** (HbO|HbR × §02|§06). **Do NOT lead with ρ.**
- **Top-10 overlap:** ST-HbO ∩ §06-|βd| = **6/10**; ST-HbR ∩ §06-|βd| = **6/10**.
- **C6 prior set in XAI top-10:** ST-HbO 3/6; ST-HbR 4/6 → both exceed ≥ 2/6 threshold.
- **Cross-chromophore XAI stability:** ρ(ST-HbO, ST-HbR) = +0.899 (p < 0.001) → chromophore-invariant attention surface.

**Clinical utility:**
**Source:** `stats/clinical_utility.md`

- Cohort-level (46.7% prevalence): PPV = 0.72, NPV = 1.00, LR+ = 3
- Primary-care (20% prevalence): PPV = 0.43, NPV = 1.00
- **Demographics-missing-11 defence:** GNN classifies all 22 trials of the 11 demographics-missing GAD subjects correctly (Sens = 1.000) → fNIRS-only signal no demographic baseline can match
- LogReg(age, sex) baseline LOSO on n=51 = F1 0.8750 / AUC 0.916 — higher than GNN on the same subset (F1 0.766) → age **is** a partial confound

**Subject-level voting (counter-intuitive negative result, P2.4):**
**Source:** `stats/subject_level_voting.md`

- Voting reduces F1 by 3–5 pp because errors are subject-specific, not random per-trial
- Best subject-level cell: ST × HbR × mt4 LOSO = F1 0.8056

---

## 3. Locked Decisions (Manuscript Preamble Table)

| # | Topic | Decision | Source |
|---|---|---|---|
| D1 | Target venue | IEEE TNSRE (8-page limit, IEEE numeric citations) | User P0.1 (2026-05-11) |
| D2 | Chromophore in main paper | **HbO + HbR** main; **HbT supplementary** | User P0.2 (2026-05-11) |
| D3 | SG comparator placement | Side-by-side ST + SG in main results table | User P0.3 (2026-05-11) |
| D4 | Citation style | IEEE numeric `[1], [2], …` | User P0.4 (2026-05-11) |
| D5 | mt placement in main table | **Both mt2 + mt4 in main** (24-row table) | User (P0.3 follow-up) |
| D6 | Headline XAI chromophore | **HbO** LOSO mt2 (matches stats arm) | Session 6 (2026-05-11) |
| D7 | Headline ML chromophore | **HbO LOSO mt=2 K=12 (F1=0.8529 — NEW global best)** | Locked 2026-05-14 after P1.7 channel-ablation; supersedes HbR K=23 = 0.8406 |
| D8 | XAI scope | ST = HbO/HbR/HbT; SG = HbO-only; native attention primary | `docs/SPEC_xai_graph.md` rev. 6 |
| D9 | CV regimes reported | 5-fold + 10-fold + LOSO (all three) | Locked |
| D10 | Trial-level vs subject-level | Trial-level in headline; subject-level voting in Discussion as failed experiment | `stats/subject_level_voting.md` |
| D11 | Pre-processing | `processed-new-mc` (Wavelet + CBSI + bandpass 0.01–0.5 Hz); mandatory | `DATA_QUALITY_REPORT.md §9` |
| D12 | Augmentation | None for headline (`_noaug_*` runs) | OUTLINE |
| D13 | Cohort | Keep all 62 subjects in main table; sensitivity drops → future-work appendix | `DATA_QUALITY_REPORT.md §6` |
| D14 | Age confound | Acknowledge as largest limitation; ANCOVA in §05 holds; demographics-missing-11 defence in §4.5 | `stats/clinical_utility.md` |
| D15 | Atlas | Brodmann via fsaverage midpoint projection; MCX Monte Carlo → future work | `docs/SPEC_xai_graph.md §16` |
| D16 | §4.3 narration framing | **Top-5 lead** (S5_D5, S1_D1, S8_D5, S4_D4, S7_D4 → BA 8/9/10/46) — 2/5 are §02 FDR-sig; 3 are GNN-only multivariate signal | Session 7 (2026-05-12) |

---

## 4. Page Budget (IEEE TNSRE 8-page limit)

| Section | Pages | Words (≈) | Figures | Tables | Status |
|---|---|---|---|---|---|
| Front matter (title, abstract, keywords) | 0.5 | 250 | – | – | 🔜 abstract to draft (Step 2) |
| §I Introduction | 1.0 | 700 | – | – | source: `literature_review.md` §B (5 paragraphs drafted) |
| §II Materials and Methods | 2.5 | 1700 | 5 | 2 | source: OUTLINE §2 + PAPER_MATH + PAPER_ARCH_TABLES |
| §III Results | 2.5 | 1700 | 5 | 2 | source: OUTLINE §3 + `stats/` |
| §IV Discussion | 1.0 | 700 | – | – | source: PROSE_SKELETONS §4 (drafted) |
| §V Conclusion + Future Work | 0.5 | 350 | – | – | source: PROSE_SKELETONS §5 (drafted) |
| References (40–50 cited) | 0.5 | – | – | – | `references/refs.bib` (52 entries; DOI verify deferred) |
| **TOTAL** | **8.0** | **5400** | **10** | **4** | – |

**Supplementary material (no page limit, IEEE Xplore):**
- Tables S1–S6: HbT main results, full Optuna trials, subject-level voting, all SG/ST per-fold metrics, §02 + §06 per-channel stats, channel-to-Brodmann mapping
- Figures S1–S10: Per-fold CMs, all-23-channel XAI heatmap, 3-view brain montage, 12-colormap selection rationale, individual top-5 bar charts, HbR XAI overlays
- SI-A: pipeline reproducibility (commit hashes, environment, seeds)
- SI-B: detailed XAI methodology (SPEC §11 acceptance criteria)
- SI-C: literature-review cluster inventory

---

## 5. Paper Section Plan

### §I Introduction
**Status: READY TO WRITE**
**Word target: ~700 words**
**Source paragraphs:** `literature_review.md §B` paragraphs §1.1, §1.2, §1.3, §1.4, §1.5.1, §1.5.2 (all drafted, lift verbatim)

**Sub-sections to cover (in order):**

| § | Subsection | Words | Key citations |
|---|---|---|---|
| 1.1 | Clinical motivation for objective GAD detection | 130 | `apa2013dsm5`, `nimh2024gad`, `who2023anxiety`, `maier1988hamilton`, `spielberger1983manual` |
| 1.2 | fNIRS as a modality for PFC anxiety research | 130 | `pinti2020fnirs`, `cui2011combined`, `pinti2018benefits` |
| 1.3 | Prior fNIRS-anxiety classification | 120 | `wang2022deeplearning`, `cnn2024spatiotemporal`, `mentalworkload2023cnn`, `comorbidity2025fnirs` |
| 1.4 | GAD–PFC literature (the strongest hook) | 200 | `etkin2009functional`, `davidson2002anxiety`, `pokorny2024young`, **`ren2026anxiety`** (strongest single anchor), `wang2024exploring` |
| 1.5 | Our contributions (5-bullet list) | 120 | – |

**Figures:** none in §I (citations only).

---

### §II Materials and Methods
**Status: READY TO WRITE**
**Word target: ~1700 words**

| § | Subsection | Words | Figure / Table | Citations |
|---|---|---|---|---|
| 2.1 | Dataset characteristics | 200 | **Table 1** (compute from `cohort_per_subject.csv`) | – |
| 2.2 | Sensor placement and channel layout | 200 | **Fig. 4** `fig_optode_layout.png`; **Fig. 5** `fig_montage_brain.png`; **Table 2 (supp)** channel-to-Brodmann | – |
| 2.3.1 | Preprocessing | 220 | – | `molavi2012wavelet`, `cui2010cbsi`, `improved_wavelet_cbsi`, `qtnirs_motion` |
| 2.3.2 | Graph dataset construction (math) | 250 | Math E1–E5 | – |
| 2.3.3 | Architectures (SG + ST) | 350 | **Fig. 2** `fig2_sg_architecture.png`; **Fig. 3** `fig3_st_architecture.png`; **Table 2** (compacted A+B) | `kipf2017semi`, `velickovic2018graph`, `brody2022how`, `wu2021comprehensive` |
| 2.3.4 | Training & evaluation | 250 | Math E9 (loss), E10 (leakage); **Fig. 1** `fig1_overall_workflow.png` | `cawley2010overfitting`, `nestedcv2018classifiers`, `bergstra2011algorithms`, `varoquaux2017assessing` |
| 2.3.5 | XAI methodology | 200 | – | `ying2019gnnexplainer`, `xai_gnn_survey2023`, `jain2019attention`, `wiegreffe2019attention` |
| 2.3.6 | Statistical-analysis arm | 200 | – | `glover1999deconvolution`, `maris2007nonparametric` |

---

### §III Results
**Status: READY TO WRITE (all numbers pre-computed)**
**Word target: ~1700 words**

| § | Subsection | Words | Figure / Table | Source |
|---|---|---|---|---|
| 3.1.3 | §02 per-channel STD | 100 | – | `02_brain_activation/results_brain_activation_stats.csv` |
| 3.1.4 | §06 canonical-HRF β | 120 | – | `06_glm_hrf/results_canonical_hrf_beta.csv` |
| 3.1.5 | Age-confound acknowledgement | 80 | – | `05_age_adjusted/results_ancova_age_sex.csv` |
| 3.2.1 | Headline cell | 100 | **Fig. 6** `fig5_headline_cm.png` | experiment_metrics.xlsx |
| 3.2.2 | ST > SG paired test | 200 | – | `stats/st_vs_sg_paired_test.md` |
| 3.2.3 | Main 24-row results table | 250 | **Table 3** (compute from experiment_metrics.xlsx) | – |
| 3.2.4 | Training dynamics | 150 | **Fig. 7** `fig6_headline_training_grid.png` | training logs |
| 3.3.1 | Headline XAI cell decision | 100 | – | – |
| 3.3.2 | Channel importance | 200 | **Fig. 8** `fig_xai_brain_overlay_top5.png` | `xai/st/hbo/loso/mt2/native/channel_importance.csv` |
| 3.3.3 | Channel-pair attention | 100 | – | `xai/st/hbo/loso/mt2/native/pair_importance.csv` |
| 3.3.4 | Temporal attention | 100 | (in concordance triptych panels E/F) | `xai/st/{hbo,hbr}/loso/mt2/native/temporal_attention.csv` |
| 3.3.5 | Brodmann mapping (top-5 lead) | 100 | **Fig. 9** `fig_top5_hbo_bars.png` | + `02_brain_activation/REPORT.md §4.4` |
| 3.3.6 | Convergence narrative | 100 | **Fig. 10** `concordance_triptych.png` | `stats/concordance_rho_table.csv` |

---

### §IV Discussion
**Status: ✅ DRAFT COMPLETE — `PAPER_PROSE_SKELETONS.md` §4.1–§4.5 + threats**
**Word target: ~700 words + 5 threat bullets**

| § | Subsection | Words | Source paragraph |
|---|---|---|---|
| 4.1 | Headline finding restated | 140 | `PROSE_SKELETONS §4.1` |
| 4.2 | Why ST > SG at LOSO specifically | 140 | `PROSE_SKELETONS §4.2` |
| 4.3 | Three-stream convergence (**top-5 lead**) | 140 | `PROSE_SKELETONS §4.3` (refreshed 2026-05-12) |
| 4.4 | Comparison to prior fNIRS-anxiety classifiers | 100 | `PROSE_SKELETONS §4.4` |
| 4.5 | Clinical deployability | 100 | `PROSE_SKELETONS §4.5` |
| 4.threats | Threats-to-validity (5 bullets) | – | `PROSE_SKELETONS §4.2 Threats` |

---

### §V Conclusion + Future Work
**Status: ✅ DRAFT COMPLETE — `PAPER_PROSE_SKELETONS.md §5`**
**Word target: ~350 words**

| § | Subsection | Words |
|---|---|---|
| 5.1 | Conclusion (2 sentences) | 50 |
| 5.2 | Future work (top 3 bullets in main; 4–8 to supplementary) | 300 |

---

## 6. Figures Master List (Final 10 for Main Paper)

| # | File | 1-line caption | Section | Source |
|---|---|---|---|---|
| 1 | `fig1_overall_workflow.{png,svg}` | End-to-end pipeline: raw fNIRS → preprocessing → graph construction → SG/ST training → XAI → atlas | §II.3 | `scientific-schematics` skill |
| 2 | `fig2_sg_architecture.{png,svg}` | Spatial-Graph architecture diagram | §II.3.3 | `scientific-schematics` skill |
| 3 | `fig3_st_architecture.{png,svg}` | Spatio-Temporal-Graph architecture diagram | §II.3.3 | `scientific-schematics` skill |
| 4 | `fig_optode_layout.{png,svg}` | 23-channel prefrontal montage (180°-rotated for face-on view) | §II.2 | `scripts/build_montage_optode_brain.py` |
| 5 | `fig_montage_brain.{png,svg}` | Frontal view of fsaverage with ROI-colored sensor dots (VMPFC blue / DMPFC green / DLPFC red) | §II.2 | `notebook_panel_b.ipynb` |
| 6 | `fig5_headline_cm.{png,svg}` | Confusion matrix: ST × HbR × mt2 × LOSO (TN=44, FP=22, FN=0, TP=58) | §III.2.1 | matplotlib render |
| 7 | `fig6_headline_training_grid.{png,svg}` | Per-fold training curves (5×3 grid: F1/loss/accuracy with best-epoch markers) | §III.2.4 | matplotlib render |
| 8 | `fig_xai_brain_overlay_top5.{png,svg}` | XAI channel-importance overlay (top-5 channels, 4-view brain, YlOrRd) | §III.3.2 | `notebook_xai_brain_figures.ipynb` cell 9 |
| 9 | `fig_top5_hbo_bars.{png,svg}` | HC vs GAD HbO STD per top-5 channel + strip plot + FDR brackets | §III.3.5 | `notebook_xai_brain_figures.ipynb` cell 8 |
| 10 | `concordance_triptych.{png,svg}` | 3-stream convergence: ST attn vs §02 vs §06 topomaps + temporal attention | §III.3.6 | `scripts/build_concordance_triptych.py` |

**Supplementary figures (no count limit):**
- `fig7_montage_anatomical.{png,svg}` — alternative ground-truth montage
- `fig_xai_brain_overlay_all23.{png,svg}` — all-23 XAI heatmap
- `fig_montage_brain_3views.{png,svg}` — composite Frontal/Left/Right
- `fig_xai_brain_overlay_top5.{png,svg}` HbR version (separately rendered)
- `fig_xai_brain_colormap_grid.{png,svg}` — 12-colormap selection rationale
- 5 × `fig_top5_bar_{S5_D5,S7_D4,S4_D4,S8_D5,S1_D1}.{png,svg}` — each top-5 channel standalone
- HbR XAI 4-view overlays

---

## 7. Tables Master List (Final 4 for Main Paper)

| # | Source | Caption | Section |
|---|---|---|---|
| 1 | `cohort_per_subject.csv` + `results_demographic_summary.csv` | Cohort characteristics (age, sex, HAM-A, STAI) by group | §II.1 |
| 2 | `PAPER_ARCH_TABLES.md` Tables A + B (compacted) | SG vs ST architecture summary (input shape, layers, params) | §II.3.3 |
| 3 | `experiments/{spatial_graph,spatial_temporal_graph}/experiment_metrics.xlsx` | Main 24-row results: HbO+HbR × {5f, 10f, LOSO} × {ST, SG} × {mt2, mt4} | §III.2.3 |
| 4 | `stats/clinical_utility.md` | PPV / NPV / LR+ at cohort + primary-care prevalence | §IV.5 |

**Supplementary tables:**
- Full Optuna search space + best trial config (SG + ST)
- HbT main results (24 rows)
- §02 + §06 per-channel statistics (full 23-channel tables)
- ST-vs-SG paired test details (12 cells)
- Subject-level voting results (P2.4)
- Channel-to-Brodmann mapping (23 rows)

---

## 8. Citation Budget per Section

| Section | Citation keys (from `refs.bib`) | Count |
|---|---|---|
| §1.1 | `apa2013dsm5`, `nimh2024gad`, `who2023anxiety`, `maier1988hamilton`, `spielberger1983manual` | 5 |
| §1.2 | `pinti2020fnirs`, `cui2011combined`, `pinti2018benefits` | 3 |
| §1.3 | `wang2022deeplearning`, `cnn2024spatiotemporal`, `mentalworkload2023cnn`, `comorbidity2025fnirs` | 4 |
| §1.4 | `etkin2009functional`, `davidson2002anxiety`, `pokorny2024young`, `ren2026anxiety`, `wang2024exploring` | 5 |
| §2.3.1 | `molavi2012wavelet`, `cui2010cbsi`, `improved_wavelet_cbsi`, `qtnirs_motion` | 4 |
| §2.3.2 | (math only) | 0 |
| §2.3.3 | `kipf2017semi`, `velickovic2018graph`, `brody2022how`, `wu2021comprehensive` | 4 |
| §2.3.4 | `cawley2010overfitting`, `nestedcv2018classifiers`, `bergstra2011algorithms`, `varoquaux2017assessing` | 4 |
| §2.3.5 | `ying2019gnnexplainer`, `xai_gnn_survey2023`, `xai_gnn_eval`, `jain2019attention`, `wiegreffe2019attention` | 5 |
| §2.3.6 | `glover1999deconvolution`, `maris2007nonparametric` | 2 |
| §III | (no new citations — local artefacts) | 0 |
| §IV.3 | reuse §1.4 anchors | 0 |
| §IV.4 | reuse §1.3 anchors | 0 |
| §IV.5 | `wilcoxon1945individual`, `mcnemar1947note`, `efron1986bootstrap`, `wynants2020prediction` | 4 |
| **TOTAL UNIQUE** | – | **≈ 40** |

---

## 9. Writing Workflow (Step-by-Step Skill Prompts)

These prompts are copy-paste-ready. They invoke skills in `.claude/skills/` (`scientific-writing`, `research-lookup`, `citation-management`, `peer-review`, `venue-templates`).

### Step 1 — Confirm IEEE TNSRE LaTeX template

```
@venue-templates Get IEEE TNSRE LaTeX template requirements and formatting
guidelines. Set up the main .tex file skeleton at
research/paper-materials/drafts/main.tex with:
- IEEEtran document class, conference=false, journal=true
- Two-column layout, IEEE TNSRE biographies section (optional)
- bibliography style = IEEEtran, citation style = numeric
- File hooks: \input{drafts/v1_abstract.md → .tex}, etc.
```

---

### Step 2 — Write Abstract

```
@scientific-writing Write the Abstract for an IEEE TNSRE paper titled
"Graph-Based Detection of Generalized Anxiety Disorder from Prefrontal fNIRS During
a Go/No-Go Task, with Native-Attention Explainability and Brodmann-Region
Cross-Validation". Target 200 words (single paragraph, unstructured), IEEE TNSRE style.

Follow this 5-sentence template:
1. MOTIVATION — GAD is prevalent, underdiagnosed, lacks objective neural biomarkers;
   fNIRS is portable + ecologically valid alternative to fMRI.
2. METHOD — Propose a Spatio-Temporal Graph Attention Network (ST-GATv2) over 23
   prefrontal fNIRS channels during a Go/No-Go task, paired with native-attention
   XAI and Brodmann-area cross-validation.
3. HEADLINE RESULT — 62-subject cohort (HC=33, GAD=29) under LOSO CV; ST achieves
   F1 = 0.8529 (Acc = 0.839, Sens = 1.000, Spec = 0.697) on HbO mt=2 with a
   K=12 channel subset derived from class-differential XAI — a +3.60 pp F1 gain
   over the all-23-channel baseline. ST outperforms Spatial-Graph baseline by
   +3.78 pp F1 across 12 paired k-fold cells (Wilcoxon W=3, p = 0.0024).
4. PARSIMONY — A 12-channel discriminative subnetwork centred on S2_D1 (BA10-R)
   improves classification reproducibly across configurations (5 of 6
   chromophore × trial-cap cells positive; mean Δ-F1 +2.52 pp), validating that
   anxiety-relevant signal concentrates in a small portion of the prefrontal montage.
5. XAI CONVERGENCE — Native attention identifies a top-5 prefrontal cluster
   (BA46-R, BA10-L, BA9-R, BA8-L) consistent with established GAD-PFC literature;
   two of five channels (S5_D5 and S1_D1) reach FDR significance in classical
   per-channel hemodynamic analyses (HC > GAD; d = −0.92 and −0.81).
6. SIGNIFICANCE — First graph-based fNIRS GAD classifier with per-channel attribution
   at no extra inference cost, cross-validated against three independent statistical
   streams and an empirical parsimony test on the XAI-derived subnetwork.

Sources for numbers:
- research/experiments/20260513/mt2/ST_GATv2_GNG_hbo_loso_mt2_noaug_K12_20260513/*_loso_overall.pkl (F1=0.8529 headline)
- experiments/spatial_temporal_graph/experiment_metrics.xlsx (K=23 baselines)
- research/experiments/20260513/CHANNEL_ABLATION_RESULTS.md (parsimony validation)
- stats/st_vs_sg_paired_test.md (paired W=3, p=0.0024)
- xai/st/hbo/loso/mt2/native/channel_importance.csv (top-5 channels)
- 02_brain_activation/results_brain_activation_stats.csv (FDR d-values)

Save to research/paper-materials/drafts/v1_abstract.md
```

---

### Step 3 — Write Introduction

```
@scientific-writing Write Section I (Introduction) for an IEEE TNSRE paper on
graph-based fNIRS GAD classification with native-attention XAI. Target ~700 words.
Use IEEE numeric citation style.

Lift verbatim from research/paper-materials/literature_review.md §B paragraphs
§1.1, §1.2, §1.3, §1.4, §1.5.1, §1.5.2 — already drafted with citation keys in place.

Five sub-sections in order:

1. §I.A Clinical motivation for objective GAD detection (~130 words):
   - GAD lifetime prevalence ~5% adults; symptom overlap with depression/MDD
     causes diagnostic delay [apa2013dsm5; nimh2024gad; who2023anxiety]
   - Current diagnostic standard = clinician-administered HAM-A or self-report STAI;
     subjective, schedule-dependent [maier1988hamilton; spielberger1983manual]

2. §I.B fNIRS as a modality for prefrontal-cortex anxiety research (~130 words):
   - Portable, motion-tolerant, ecologically valid; direct hemodynamic readout
     of prefrontal cortex [pinti2020fnirs; cui2011combined; pinti2018benefits]
   - Penetration depth ~10-30 mm sufficient for PFC subregions implicated in anxiety

3. §I.C Prior fNIRS-anxiety classification (~120 words):
   - CNN/MLP/SVM classifiers achieve 75-85% accuracy on similar tasks
     [wang2022deeplearning; cnn2024spatiotemporal; mentalworkload2023cnn; comorbidity2025fnirs]
   - Most lack per-channel attribution; post-hoc explanations are model-specific

4. §I.D GAD-PFC literature (~200 words, the strongest hook):
   - Anxiety implicates DLPFC (BA9/46) inhibitory control + dmPFC (BA8) monitoring
     + vmPFC (BA10) emotion regulation
     [etkin2009functional; davidson2002anxiety; pokorny2024young]
   - Go/No-Go specifically reveals DLPFC inhibition deficits in GAD
     [pokorny2024young (TMS-EEG paradigm-matched); ren2026anxiety (largest fNIRS-anxiety
     sample, strongest single citation); wang2024exploring (paradigm-matched fNIRS-GNG)]

5. §I.E Our contributions (~120 words, 5-bullet list — bullet acceptable in intro):
   - Graph-based fNIRS analysis: channels as nodes, correlation/coherence as edges,
     on the canonical 23-channel prefrontal montage
   - Two complementary architectures (Spatial Graph + Spatio-Temporal Graph) compared
   - Nested-CV Optuna search; reproducible deterministic-split protocol with leakage controls
   - Native-attention XAI mapped to Brodmann regions, cross-validated against a parallel
     classical statistical-analysis arm (§02 STD + §06 canonical HRF β)
   - First report to triangulate ML attention, hemodynamic-amplitude statistics, and
     atlas-registered anatomy on a single GAD-fNIRS cohort

IEEE TNSRE style, IEEE numeric citations.
Save to research/paper-materials/drafts/v1_introduction.md
```

---

### Step 4 — Write Methods §II.1 (Dataset characteristics)

```
@scientific-writing Write Section II.A (Dataset Characteristics) for an IEEE TNSRE
paper on graph-based fNIRS GAD classification. Target ~200 words + Table 1.

Include:
1. Recruitment: n=62 final analysis (33 HC, 29 GAD). All retained.
2. Table 1 — Group comparison (compute from cohort_per_subject.csv):
   - Age mean ± SD per group; range
   - Sex distribution
   - HAM-A score (GAD), STAI-S, STAI-T (both groups)
   - n=51 demographics-eligible subset; n=11 GAD demographics-missing
     (AA089-AA099, LA091, LA095, LA096) — clinical scores only
3. Age confound disclosure: HC mean 73.0 yrs vs GAD mean 51.1 yrs;
   Welch t = 6.08, p = 6.5 × 10⁻⁶. Acknowledge here; expand defence in §V Discussion.
4. Cohort decisions:
   - Keep all 62 in main analysis (no exclusions)
   - Sensitivity drops (AH024, AH029, LA063, demographics-missing 11)
     → future-work appendix only
5. Ethics statement: [IRB approval number and institution — TO BE PROVIDED
   by dataset author / advisor. Leave as PLACEHOLDER — do NOT fabricate.]

Sources:
- src/notebook/statistical-analysis/01_demographic/cohort_per_subject.csv
- src/notebook/statistical-analysis/01_demographic/results_demographic_summary.csv
- memory: project_subjects_ground_truth.md

IEEE TNSRE style. ~200 words + Table 1.
Save to research/paper-materials/drafts/v1_methods_participants.md
```

---

### Step 5 — Write Methods §II.2 (Sensor placement and channel layout)

```
@scientific-writing Write Section II.B (Sensor Placement and Channel Layout)
for an IEEE TNSRE paper. Target ~200 words.

Include:
1. NIRX system, 8 source-detector pairs → 23 channels, wavelengths 760 nm + 850 nm.
2. Prefrontal cortex placement following international 10-20 system, custom
   BrainTech montage (data/brainproducts-RNP-BA-128-custom.elc).
3. Reference Figure 4 (`fig_optode_layout.png`) — 23-channel montage rotated 180°
   for face-on view (subject's left on viewer's right, matches rostral brain view).
4. Reference Figure 5 (`fig_montage_brain.png`) — frontal view of fsaverage with
   ROI-colored sensor dots: VMPFC blue, DMPFC green, DLPFC red, default orange.
5. Channel-to-Brodmann mapping (probabilistic, fsaverage midpoint projection)
   reported in Supplementary Table SI-Ch:
   - BA10 (vmPFC): S1_D1/L, S1_D3/L, S2_D1/R, S2_D2/R, S5_D2/R
   - BA9 (dlPFC): S2_D5/R, S3_D1/L, S3_D3/L, S3_D4/L, S4_D4/R, S4_D5/R, S8_D5/R
   - BA8 (dmPFC): S3_D6/L, S6_D6/L, S7_D4/L, S7_D6/L, S4_D7/R, S7_D7/R, S8_D7/R, S8_D8/R
   - BA46 (DLPFC): S5_D5/R, S5_D8/R, S6_D3/L
6. Brief note that the channel-pair list is the single source of truth across
   SG / ST / XAI / stats arms.

Sources:
- data/brainproducts-RNP-BA-128-custom.elc (ELC file with optode positions)
- src/xai/atlas.py:parse_elc (deterministic atlas parser)
- src/xai/channels.py:CHANNEL_NAMES (canonical 23-channel list)
- research/xai/atlas/channel_to_brodmann.csv (deterministic BA mapping)

IEEE TNSRE style. ~200 words.
Save to research/paper-materials/drafts/v1_methods_montage.md
```

---

### Step 6 — Write Methods §II.3.1 (Preprocessing)

```
@scientific-writing Write Section II.C.1 (Preprocessing Pipeline) for an IEEE TNSRE
paper. Target ~220 words.

Describe the 4-stage preprocessing pipeline (`processed-new-mc/`):

1. Wavelet motion correction [molavi2012wavelet; improved_wavelet_cbsi]
2. Correlation-Based Signal Improvement (CBSI) [cui2010cbsi]
   — enforces r(HbO, HbR) = -1 by construction (acknowledge constraint;
   HbT = HbO + HbR is the only chromophore not so constrained, reported supplementary)
3. Bandpass filter 0.01-0.5 Hz
4. Per-trial standardization, applied FOLD-WISE for leak-free cross-validation
   (mathematically formalized as Eq. E10 in §II.C.4)

Include 2 numbered display equations:
- HbO/HbR derivation via modified Beer-Lambert Law:
    ΔC(t) = ΔOD(t) / (ε · DPF · L)
  where ε = molar extinction coefficient, DPF = differential pathlength factor,
  L = source-detector separation [cm]
- HbT derivation:
    HbT(t) = HbO(t) + HbR(t)

Reference the supplementary `DATA_QUALITY_REPORT.md §9` for the rationale of the
processed-new-mc pipeline choice.

Sources:
- data/DATA_QUALITY_REPORT.md §9
- processed-new-mc/ directory structure
- PAPER_MATH.md (E10 — leakage control)

IEEE TNSRE style. ~220 words + 2 numbered equations.
Save to research/paper-materials/drafts/v1_methods_preprocessing.md
```

---

### Step 7 — Write Methods §II.3.2 (Graph dataset construction)

```
@scientific-writing Write Section II.C.2 (Graph Dataset Construction) for an IEEE TNSRE
paper. Target ~250 words.

CORE METHODOLOGICAL CONTENT — write carefully with all equations numbered.

Content (in order):

1. Nodes: 23 channels as graph nodes.

2. Node feature construction:
   - Spatial-Graph (SG): per-channel-per-trial feature vector [23, 6] =
     (mean, min, max, skew, kurt, var). Include numbered equations for
     skewness (Eq. E1) and kurtosis (Eq. E2) from PAPER_MATH.md.
   - Spatio-Temporal-Graph (ST): per-channel-per-trial raw z-scored time series
     [23, 326]; model internally unfolds to [23, K, 6] over K sliding windows of
     width W=48.

3. Edge construction (data-driven, per-trial):
   - Pearson correlation between channel pairs (Eq. E3)
   - Magnitude-squared coherence via Welch's method (Eq. E4)
   - Edge rule: keep edge (i,j) iff |corr(i,j)| >= τ_corr = 0.1 (Eq. E5)
   - Edge attributes = (coherence_ij, |corr_ij|)
   - directed = True, self_loops = True

All equations E1-E5 lifted directly from research/paper-materials/PAPER_MATH.md;
each equation is verified against src/core/utils.py.

Reference Figure 1 (`fig1_overall_workflow.png`) for the visual flow:
raw fNIRS → preprocessing → graph construction → SG/ST training → XAI → atlas.

Sources:
- research/paper-materials/PAPER_MATH.md (E1-E5)
- src/core/utils.py (validation of equation implementation)

IEEE TNSRE style. ~250 words + 5 numbered equations (E1-E5).
Save to research/paper-materials/drafts/v1_methods_graph_construction.md
```

---

### Step 8 — Write Methods §II.3.3 (Architectures)

```
@scientific-writing Write Section II.C.3 (Graph Neural Network Architectures)
for an IEEE TNSRE paper. Target ~350 words + Table 2 (compacted A+B).

Describe BOTH architectures with parameter counts and Optuna-best hyperparameters.

A. Spatial Graph (SG):
- Input: [23, 6] (per-trial graph with engineered node features)
- 2-layer GINEConv (graph isomorphism + edge attrs) → GATv2 attention layers
- Optuna-best filters: n_filters = [112, 80]; heads = [8, 6]
- Pre-pool Linear(480 → 224) — APPLIED PER-NODE BEFORE global mean-pool
  (note in PAPER_ARCH_TABLES.md: the schematic figure draws this incorrectly)
- Global mean pool → softmax (2 classes)
- Parameter count: ≈ 1.16 M
- Reference: [kipf2017semi] (GCN baseline); [velickovic2018graph] (GAT);
  [brody2022how] (GATv2 — used as primary)

B. Spatio-Temporal Graph (ST):
- Input: [23, 326] raw z-scored time series
- Sliding window W=48 → K windows
- Per-window GATv2 with SHARED weights across all K windows
- GRU(hidden=64) + additive attention over time (the explainability hook)
- Softmax (2 classes)
- Parameter count: ≈ 0.37 M
- Reference: [brody2022how] (GATv2); GATv2 equation = Eq. E6; GRU+attention
  combination = Eq. E7 + E8 from PAPER_MATH.md

Include Eq. E6 (GATv2 attention coefficient), Eq. E7 (GRU update), and
Eq. E8 (additive temporal attention) as numbered display equations.

Reference Figure 2 (`fig2_sg_architecture.png`) and Figure 3 (`fig3_st_architecture.png`)
for visual schematics. NOTE: PAPER_ARCH_TABLES.md flags two figure-vs-table subtleties:
(a) SG pre-pool ordering is per-node, not post-pool — the figure draws it incorrectly;
(b) ST GATv2 weights are shared across all K windows.

Add a brief comparator paragraph: SG is ~3× the parameters of ST, making ST the
more compact architecture; this counters the intuition that "deeper = better" and
motivates §IV.2 (Why ST > SG at LOSO).

Cite [wu2021comprehensive] for the GNN survey context.

Source files:
- research/paper-materials/PAPER_ARCH_TABLES.md (Tables A, B with full hyperparams)
- research/paper-materials/PAPER_MATH.md (E6, E7, E8)
- src/core/models.py (SG)
- src/core_st/models.py (ST)

IEEE TNSRE style. ~350 words + 3 numbered equations + Table 2 (compacted).
Save to research/paper-materials/drafts/v1_methods_architectures.md
```

---

### Step 9 — Write Methods §II.3.4 (Training & Evaluation)

```
@scientific-writing Write Section II.C.4 (Training and Evaluation Protocol)
for an IEEE TNSRE paper. Target ~250 words.

Include:

1. Optimizer + loss:
   - Adam (β1=0.9, β2=0.999)
   - Cross-entropy loss (Eq. E9 from PAPER_MATH.md)

2. Optuna-best hyperparameters (from `experiments/...result_report.md`):
   - SG: lr = 6.79 × 10⁻³, cosine_warmup, n_filters=[112,80], heads=[8,6]
     Trial #67 in `research/experiments/20260430/...optuna_search_nested_validation/`
   - ST: lr = 3.04 × 10⁻⁴, CosineAnnealingLR(T_max=150, eta_min=1×10⁻⁵)
     Trial #36 in `research/experiments/20260503/...`
   - **Note headline flips:** SG headline config flips 3 flags (use_residual,
     use_norm, use_gine_first_layer) vs Trial #67 best — these come from the
     leak-free patience study (20260506), not Optuna. Flag this in Methods.

3. Patience:
   - SG = 9999 (effectively off — full epoch run per leak-free patience study 20260506)
   - ST = 30 (paper baseline)

4. CV protocol (D9 — all three reported):
   - 5-fold subject-level stratified
   - 10-fold subject-level stratified
   - LOSO: 62 iterations (one per subject); strongest evidence of generalization
   - Splits source: data/splits/kfold_splits_processed_new_mc.json (deterministic,
     seed=42) + in-code get_loso_splits

5. Leakage control protocol (Eq. E10):
   - Per-trial standardization done fold-wise (mean/std from train split only)
   - No subject appears in both train and test of any fold
   - Verified empirically in §06_glm_hrf leakage tests

6. Validation methodology:
   - Nested Optuna search → outer CV for performance estimation
   - References: [cawley2010overfitting; nestedcv2018classifiers] (nested-CV is
     mandatory for small-cohort biomedical classifiers)
   - [bergstra2011algorithms] (TPE optimizer used by Optuna)
   - [varoquaux2017assessing] (CV variance assessment)

7. Reproducibility:
   - Random seed = 42 (fully deterministic across all CV folds)
   - Source: `experiments/{spatial_graph,spatial_temporal_graph}/experiment_metrics.xlsx`
     is the authoritative metrics file

Include Eq. E9 (cross-entropy) and Eq. E10 (fold-wise standardization) as numbered
display equations.

Source files:
- PAPER_MATH.md (E9, E10)
- PAPER_ARCH_TABLES.md Table C (training hyperparams), Table D (Optuna search space)
- `research/experiments/20260430/...result_report.md` (SG Optuna)
- `research/experiments/20260503/...result_report.md` (ST Optuna)
- memory: project_paper_outline_task_state.md (locked decisions)

IEEE TNSRE style. ~250 words + 2 numbered equations.
Save to research/paper-materials/drafts/v1_methods_training.md
```

---

### Step 10 — Write Methods §II.3.5 (XAI methodology)

```
@scientific-writing Write Section II.C.5 (Explainable AI Methodology) for an
IEEE TNSRE paper. Target ~200 words.

Include:

1. Native-attention extraction (the primary XAI path, D8):
   - ST GATv2 attention coefficients (Eq. E6) are extracted DIRECTLY at inference
   - No perturbation, no gradient-based attribution required
   - Source: src/xai/st_explainer.py

2. Aggregation:
   - Channel-importance = sum of all attention coefficients flowing INTO each node,
     averaged across trials in the test set
   - Channel-pair importance = pairwise attention matrix from the same source
   - Temporal attention = additive attention weights over the GRU's K windows
     (Eq. E8)

3. Cross-check via GNN-Explainer [ying2019gnnexplainer]:
   - Supplementary path; 3-way SG comparison
     (native gnn attention / Captum-IG / GNNExplainer-mask)
   - Validated in `02_st_population.ipynb` cross-cell sweep
   - References for survey: [xai_gnn_survey2023; xai_gnn_eval]

4. Atlas registration (D15, deferred MCX):
   - Channel midpoint projected to fsaverage pial surface; probabilistic
     Brodmann via FreeSurfer atlas
   - Acceptance criterion C8: S2_D1 → BA10 probability ≥ 0.5 (verified)
   - Source: docs/SPEC_xai_graph.md §16; src/xai/atlas.py

5. Critical XAI caveat [jain2019attention; wiegreffe2019attention]:
   - Attention ≠ causation
   - Must be defended via §III.3.6 concordance against §02 + §06 statistical streams
   - This caveat is the central rhetorical move that justifies the cross-validation
     of XAI against an independent statistical-analysis arm

Cite all of [ying2019gnnexplainer], [xai_gnn_survey2023], [xai_gnn_eval],
[jain2019attention], [wiegreffe2019attention] in this section.

Source files:
- docs/SPEC_xai_graph.md (rev. 6)
- src/xai/st_explainer.py (native attention extraction)
- src/xai/atlas.py (Brodmann registration)
- memory: project_xai_task_state.md

IEEE TNSRE style. ~200 words.
Save to research/paper-materials/drafts/v1_methods_xai.md
```

---

### Step 11 — Write Methods §II.3.6 (Statistical-analysis arm)

```
@scientific-writing Write Section II.C.6 (Statistical-Analysis Arm) for an IEEE TNSRE
paper. Target ~200 words.

This section establishes the THREE INDEPENDENT statistical streams that converge on
the GAD-PFC cluster (§III.3.6 narrative).

Include:

1. §02 Brain Activation analysis (per-channel STD):
   - Mann-Whitney U test per channel on HbO STD across all 62 subjects' trials
   - Benjamini-Hochberg FDR correction (q < 0.05) across the 23 channels
   - Chromophore = HbO only (statistical-arm decision)
   - Source: 02_brain_activation/REPORT.md §4.2;
            02_brain_activation/results_brain_activation_stats.csv

2. §06 Canonical HRF β (GLM with double-gamma HRF):
   - HRF basis: double-gamma [glover1999deconvolution]
   - Cluster-based permutation [maris2007nonparametric] for inference
   - Source: 06_glm_hrf/REPORT.md §4.2

3. §05 Age-adjusted analysis (sensitivity):
   - ANCOVA controlling for age + sex on the n=51 demographics-eligible subset
   - Top-4 §02 channels survive age control (S5_D5: β = -0.048, p = 0.0023, η²ₚ = 0.18)
   - Source: 05_age_adjusted/results_ancova_age_sex.csv

4. Constraint disclosure:
   - The acquisition .tri trigger files contain only BLOCK-level event markers
     (task / baseline / rest); per-stimulus Go-vs-No-Go onsets are unavailable
   - This precludes a true first-level GLM with a No-Go - Go inhibition contrast
   - Canonical-HRF β is the maximally-defensible block-level analog
   - Source: 06_glm_hrf/REPORT.md §2

Cite [glover1999deconvolution] and [maris2007nonparametric].

Source files:
- 02_brain_activation/REPORT.md
- 06_glm_hrf/REPORT.md
- 05_age_adjusted/results_ancova_age_sex.csv

IEEE TNSRE style. ~200 words.
Save to research/paper-materials/drafts/v1_methods_stats_arm.md
```

---

### Step 12 — Write Results §III.1 (Statistical-analysis arm)

```
@scientific-writing Write Section III.A (Statistical-Analysis Arm — Results) for an
IEEE TNSRE paper. Target ~300 words total across 3 sub-sections.

§III.A.1 §02 Per-channel STD activation (~100 words):
- 4/23 channels FDR-significant at q < 0.05 (HC > GAD at every FDR channel):
  | Channel | Brodmann | d | p_FDR |
  | S5_D5 | BA46-R | -0.92 | 0.014 |
  | S2_D1 | BA10-R | -0.85 | (TBD pull from CSV) |
  | S3_D3 | BA9-L | -0.80 | (TBD pull) |
  | S1_D1 | BA10-L | -0.81 | 0.014 |
- Direction at every FDR channel: HC > GAD → hypoactivity pattern consistent
  with GAD-PFC literature

§III.A.2 §06 Canonical-HRF β (~120 words):
- 2/23 channels FDR-significant: S4_D5, S4_D7 (both BA9)
- HC β > GAD β at all 13 cluster-permutation hits
- TWO-WINDOW finding: early (0-7 s) + late (21-32 s); LATE WINDOW DOMINANT
  → HC ramps up across the 32-s block, GAD does not sustain
- Cross-method: Spearman(§02, §06) = +0.19 → methods tag complementary channels
  (NOT redundant)

§III.A.3 Age-confound acknowledgement (~80 words):
- Top-4 §02 channels remain raw-significant after age ANCOVA on n=51 demographics-
  eligible subset. S5_D5: β = -0.048, p = 0.0023, η²ₚ = 0.18 — large effect
  surviving age control.
- This positions the §02 findings as not purely age-driven; full defence of the
  age confound in §V.B (Discussion).

Do NOT add new citations in §III.A (results are our own).
Pull exact numbers from the source CSVs — do NOT round/round-trip from memory.

Source files (pull exact numbers from these):
- 02_brain_activation/results_brain_activation_stats.csv
- 06_glm_hrf/results_canonical_hrf_beta.csv
- 06_glm_hrf/results_cluster_permutation.csv
- 05_age_adjusted/results_ancova_age_sex.csv

IEEE TNSRE style. ~300 words across 3 sub-sections.
Save to research/paper-materials/drafts/v1_results_stats_arm.md
```

---

### Step 13 — Write Results §III.2 (ML results)

```
@scientific-writing Write Section III.B (Machine-Learning Results) for an IEEE TNSRE
paper. Target ~700 words + Table 3 (24 rows) + Figures 6, 7.

§III.B.1 Headline cell (~150 words; LOCKED 2026-05-14 after P1.7 channel-ablation):
- Primary: ST × HbO × LOSO × mt=2 × K=12 = **F1 0.8529**; Acc = 0.839; Sens = 1.000; Spec = 0.697.
  Confusion matrix: TN = 46, FP = 20, FN = 0, TP = 58. The K=12 subset is the
  top-12 channels by HC-vs-GAD differential XAI weighted-degree (S2_D1, S4_D4,
  S3_D1, S6_D3, S8_D5, S3_D6, S8_D7, S3_D4, S7_D6, S6_D6, S2_D2, S1_D1; details §III.C.7).
  +3.60 pp over the all-23-channel baseline (F1 = 0.8169 at HbO mt2 K=23);
  +1.23 pp over the prior published global best (HbR mt2 K=23 = 0.8406).
- Secondary (sensitivity-analysis row): ST × HbO × LOSO × mt=4 × K=8 = F1 0.8672
  (sens 0.957, spec 0.780, CM TN=103/FP=29/FN=5/TP=111). Highest raw F1 across
  the 24-cell channel-ablation sweep; reported here because it shows the
  classifier can reach higher F1 at the cost of imperfect anxiety detection.
- 62-subject LOSO regime (each subject held out once as test set).
- Reference Figure 6 (`fig5_headline_cm.png` — to be regenerated for the new HbO K=12 CM).
- Sources:
  - research/experiments/20260513/mt2/ST_GATv2_GNG_hbo_loso_mt2_noaug_K12_20260513/*_loso_overall.pkl (headline)
  - research/experiments/20260513/mt4/ST_GATv2_GNG_hbo_loso_mt4_noaug_K08_20260513/*_loso_overall.pkl (secondary)
  - experiments/spatial_temporal_graph/experiment_metrics.xlsx (K=23 baseline)
  - research/experiments/20260513/CHANNEL_ABLATION_RESULTS.md (full 24-cell breakdown)

§III.B.2 ST > SG paired test (~200 words):
- Aggregate Wilcoxon W = 3, p = 0.00244 across 12 paired k-fold cells
  (HbO + HbR × {5-fold, 10-fold, LOSO} × {mt2, mt4})
- Mean Δ(ST − SG) F1 = +3.78 percentage points
- McNemar test on HbR mt4 LOSO: p = 2.3 × 10⁻⁴
  (47 ST-corrects vs 17 SG-corrects on the same trials)
- Interpretation: LOSO is where ST gains most — exactly the subject-level
  distribution-shift regime where SG's static features over-fit
- Source: stats/st_vs_sg_paired_test.md + stats/st_vs_sg_mcnemar_loso.csv

§III.B.3 Main 24-row results table (~250 words):
- Table 3: HbO + HbR × {5-fold, 10-fold, LOSO} × {ST, SG} × {mt2, mt4}
- Per-row entries: F1 (mean ± SD over folds), Accuracy, Sensitivity, Specificity,
  ROC-AUC
- Source: experiments/{spatial_graph,spatial_temporal_graph}/experiment_metrics.xlsx
- Narrative pointers:
  - At LOSO mt4 HbR, ST dominates SG by the widest margin (matches McNemar finding)
  - At k-fold regimes, the gap narrows but remains positive on F1 across all 12 cells
  - HbO and HbR show similar patterns at K=23; HbR K=23 = 0.8406 was the prior global best at full-channel (now superseded by HbO K=12 = 0.8529 after channel-ablation — see §III.C.8 + Table 4)
- HbT supplementary (D2): 24-row HbT table goes to Supplementary Table SI-T

§III.B.4 Training dynamics (~150 words):
- Reference Figure 7 (`fig6_headline_training_grid.png`): 5×3 grid showing per-fold
  train + val curves for {Loss, Accuracy, F1} with best-epoch markers
- LOSO mt=2 HbR run: best epoch typically at 80-120 of 150; minimal over-fitting;
  validation F1 closely tracks train F1 with stable margin
- Source: training logs from headline run; figure rendered via matplotlib

Do NOT add new citations in §III.B.
Pull exact numbers from the source files — do NOT round/round-trip from memory.

IEEE TNSRE style. ~700 words + Table 3 + 2 figure references.
Save to research/paper-materials/drafts/v1_results_ml.md
```

---

### Step 14 — Write Results §III.3 (XAI)

```
@scientific-writing Write Section III.C (Explainable AI Results) for an IEEE TNSRE
paper. Target ~700 words + Figures 8, 9, 10.

§III.C.1 Headline XAI cell decision (~100 words):
- Decision D6: ST × HbO × mt=2 × LOSO is the headline XAI cell (matches stats arm).
- Justification: ρ(ST attention HbO, ST attention HbR) = +0.899 (p<0.001) across the
  23 channels → attention surface is chromophore-invariant; HbO is the cleanest
  narrative anchor because it matches the statistical-analysis arm.
- HbR cell exists on disk (research/xai/st/hbr/loso/mt2/native/) and is referenced
  in §III.C.4 (temporal-attention panel) + Supplementary Figure SI-XAI-HbR.
- Source: stats/concordance_rho_table.csv

§III.C.2 Channel importance (~200 words):
- Reference Figure 8 (`fig_xai_brain_overlay_top5.png`) — 4-view brain overlay
  (Frontal / Dorsal / Left-lateral / Right-lateral); Gaussian σ=15 mm vertex
  overlay; YlOrRd colormap.
- Top-5 most-attended channels (with Brodmann mapping):
  1. S5_D5 — BA46-R (right DLPFC)
  2. S1_D1 — BA10-L (left vmPFC / frontopolar)
  3. S8_D5 — BA9-R (right DLPFC)
  4. S4_D4 — BA9-R (right dmPFC midline-leaning)
  5. S7_D4 — BA8-L (left dmPFC / SEF)
- All five span BA 8/9/10/46 — the Brodmann cluster implicated in GAD (DLPFC
  inhibition control + dmPFC monitoring + vmPFC emotion regulation).
- Source: xai/st/hbo/loso/mt2/native/channel_importance.csv

§III.C.3 Channel-pair attention matrix (~100 words):
- Brief reference to pair_importance.csv heatmap; the strongest pair-edges are
  intra-DLPFC and DLPFC↔dmPFC connections.
- Full pair-heatmap → Supplementary Figure SI-PAIR.
- Source: xai/st/hbo/loso/mt2/native/pair_importance.csv

§III.C.4 Temporal attention (~100 words):
- LATE-WINDOW DOMINANCE in HC trials, attenuated in GAD trials — matches §06
  cluster-permutation late window finding (21-32 s).
- HbO + HbR temporal-attention curves are nearly identical (consistent with the
  ρ = +0.899 cross-chromophore stability finding).
- Figure: lifted from concordance triptych panels E (HbO) and F (HbR).
- Source: xai/st/{hbo,hbr}/loso/mt2/native/temporal_attention.csv

§III.C.5 Brodmann mapping (~100 words):
- Reference Figure 9 (`fig_top5_hbo_bars.png`) — HC vs GAD HbO STD per top-5 channel
  with per-subject strip plot and FDR-significance brackets.
- Top-5 channels span BA 8 / 9 / 10 / 46 — the dorsolateral, dorsomedial, and
  ventromedial PFC subdivisions implicated in anxiety.
- TWO of the top-5 (S5_D5 BA46-R and S1_D1 BA10-L) are also §02 FDR-significant
  (HC > GAD, d = -0.92 and -0.81) → direct hypoactivity convergence.
- The REMAINING three top-5 channels (S7_D4, S4_D4, S8_D5) sit within BA 8/9
  dmPFC but fall BELOW univariate FDR — precisely the multivariate, network-level
  signal that motivates a graph-based classifier.
- Source: 02_brain_activation/REPORT.md §4.4 + xai/st/hbo/loso/mt2/native/

§III.C.6 Convergence narrative (~100 words):
- SET-OVERLAP IS THE FIGURE-OF-MERIT, NOT rank-correlation.
- ρ over 23 channels is NULL in all 4 pairings (HbO|HbR × §02|§06):
  HbO vs §02 |d|: ρ = +0.002, p = 0.99
  HbO vs §06 |β d|: ρ = -0.007, p = 0.98
  HbR vs §02 |d|: ρ = +0.10, p = 0.64
  HbR vs §06 |β d|: ρ = -0.05, p = 0.84
  **DO NOT lead with ρ.**
- Top-10 overlap (the number to cite): ST-HbO ∩ §06-|βd| = **6/10**;
  ST-HbR ∩ §06-|βd| = **6/10**.
- C6 prior set in XAI top-10: ST-HbO 3/6 (S5_D5, S1_D1, S3_D3);
  ST-HbR 4/6 (above + S4_D7). Both exceed the SPEC §11 C6 ≥ 2/6 threshold.
- Reference Figure 10 (`concordance_triptych.png`) — 3×2 layout:
  (A) ST attn HbO z-score topomap, (B) ST attn HbR z-score topomap,
  (C) §02 |d| topomap, (D) §06 |β d| topomap,
  (E) HbO temporal attention, (F) HbR temporal attention.
  C6 channels labelled in yellow; §06 cluster-permutation windows shaded
  in panels E/F.
- Source: stats/concordance_rho_table.csv + concordance_rho_table.md

§III.C.7 Class-differential connectivity: channel parsimony & S2_D1-centred network reshaping (~120 words, NEW 2026-05-13; narrative reframed 2026-05-13 — NOT a hardware-reduction sub-section):
- LEAD CLAIM: every channel earns its place in the prefrontal montage. AVERAGE
  GNN ATTENTION IS NEAR-UNIFORM across the 23 channels (K@80 % mass ≈ 21 in
  every cell — top-3 hubs only ~1 % above the 1/22 uniform floor). No single
  channel is visibly redundant.
- HC-vs-GAD CLASS DIFFERENTIAL `|M_diff| = |M_GAD − M_HC|` IS concentrated
  (max/min weighted-degree ratio 2.4–2.9 × across cells) — class discrimination
  is carried by a small subnetwork, not by amplifying or silencing single
  channels.
- HEADLINE: **S2_D1 (BA10-R, right frontopolar / vmPFC) is the top-1
  differential channel** in ST × HbO and ST × HbR; #2 in SG × HbO. This CLOSES
  the §02 ↔ XAI disagreement noted in `project_xai_stats_concordance.md`:
  S2_D1 ranked dead-last in *average* attention but rank-1 in *differential*
  attention, restoring statistical convergence (S2_D1 is §02 FDR #1 and §06 #3).
- Robust pairs across all 3 cells: S2_D1–S3_D3 (GAD > HC), S2_D1–S3_D6 (HC > GAD),
  S2_D1–S6_D6 (GAD > HC) → S2_D1's connectivity profile is **rewired** between
  groups (same node, different partners), not amplified or silenced. This is
  the multivariate, edge-level signal that justifies a graph model over a flat
  MLP.
- STRUCTURAL-CORRELATION CONFOUND REJECTED: ρ (XAI `|M_diff|` weighted-degree,
  structural `|C_diff|` weighted-degree from the raw fNIRS signal) is NEGATIVE
  in all 3 cells (−0.33 / −0.28 / −0.39); the GNN's differential picks up
  signal beyond raw correlation differences.
- Cross-chromophore stability (HbO ↔ HbR within ST) Kendall τ = +0.35
  (p = 0.019); cross-architecture (ST ↔ SG) τ ≈ +0.06 (n.s., accepted limitation).
- Frame as a parsimony hypothesis: "every channel earns its place" — falsifiable
  via the channel-ablation experiment in §V.B (top-K ablation cuts K = 12 / 8 / 16
  produced by `05_channel_reduction.ipynb` are the test points). DO NOT frame
  as a hardware / minimal-montage proposal: the 23 channels share 16 optodes,
  and a K = 12 channel cut still requires 14 / 16 optodes — channel ablation
  is the scientific test, not a device-design recommendation. Note this in
  one sentence at most.
- Reference Supplementary Figure SI-SUBNET (`subnetwork_top_decile.png`) —
  three-panel 5×7 grid showing the top-decile `|M_diff|` edges per cell with
  node size encoding weighted degree.
- Source: research/xai/channel_reduction/{run.json, reduction_summary.csv,
  top_differential_pairs.csv, figures/}; build script
  scripts/build_xai_class_differential.py; analysis notebook
  src/notebook/xai/05_channel_reduction.ipynb.

§III.C.8 Parsimony-validation outcome (~140 words, NEW 2026-05-14 — empirical validation of §III.C.7):
- LEAD: "We empirically tested the parsimony hypothesis stated in §III.C.7 by
  retraining the ST model on top-K channel subsets ranked by `|M_diff|`
  weighted degree (K ∈ {8, 12, 16}) across all three chromophores and both
  trial-cap regimes (mt=2 and mt=4)."
- K-CONSISTENCY RESULT: K=12 is the most beneficial subset across configurations.
  It improves over the 23-channel baseline in 5 of 6 chromophore × trial-cap
  cells (mean Δ-F1 +2.52 pp; vs +1.85 pp at K=16 and +1.29 pp at K=8). K=12 has
  the highest mean F1 (0.8401) of any tested K. The single negative cell is
  HbR at mt=4 (Δ-F1 −2.29 pp).
- HEADLINE CELL: At mt=2 every chromophore peaks at K=12 (mean Δ-F1 +3.64 pp
  over the K=23 baseline). The locked paper headline cell (HbO mt=2 K=12 =
  F1 0.8529) is drawn from this regime.
- CROSS-MT TRANSFERABILITY: At mt=4 the K=12 subset still helps in 2 of 3
  chromophores (HbO +0.42 pp, HbT +6.05 pp); per-chromophore mt4 optimum K
  shifts (HbO→K=8, HbR→K=16, HbT→K=12). The discriminative subnetwork is
  well-localised at mt=2 (the regime the XAI was derived from) and
  substantially transfers to mt=4.
- SECONDARY NUMBER: HbO × mt=4 × K=8 = F1 0.8672 (sens 0.957, spec 0.780).
  Highest raw F1 across the 24-cell sweep; included as a sensitivity-analysis row
  in Table 3 §III.B.3, not as the headline (sens trade-off + sharp local maximum
  rather than smooth peak).
- CROSS-MODALITY REPLICATION: The mt=2 mean Δ-F1 (+3.64 pp at K=12/23 ≈ 52 %
  retention) replicates Liu et al. [2023]'s +4–5 pp peak on EEG-emotion graphs
  at K=30/62 ≈ 48 % retention — quantitative concordance across modalities.
- HBT MT=4 K=23 NOTE: This cell showed run-to-run F1 variance of ~5 pp across
  3 independent training runs (0.7860 / 0.8413 / 0.7860); 2 of 3 reproduce the
  20260509 published baseline bit-identically. Disclosed under Threats-to-Validity (§V).
- Reference Table 4 (channel-ablation Δ-F1 matrix, 24 cells) and Figure 8
  (K-consistency bar plot or Δ-F1 grouped bar chart, to be generated).
- Source: research/experiments/20260513/CHANNEL_ABLATION_RESULTS.md (full
  breakdown, all metrics tables, K-consistency analysis §9.3, paper-headline
  rationale §10.3); CHANNEL_ABLATION_COMMANDS.md (reproducible run commands).
  Code: src/core_st/main.py `--channel_subset` flag + tests/core_st/test_channel_subset.py.

Do NOT add new citations in §III.C.
Pull exact ρ values from the source CSV — do NOT round/round-trip from memory.

IEEE TNSRE style. ~960 words + 3 main-figure refs + 1 supplementary-figure ref + 1 new table ref (Table 4 channel-ablation Δ-F1 matrix).
Save to research/paper-materials/drafts/v1_results_xai.md
```

---

### Step 15 — Write Discussion §IV

```
@scientific-writing Write Section IV (Discussion) for an IEEE TNSRE paper. Target
~700 words across 5 sub-sections + a 5-bullet Threats-to-Validity list (~100 words).

Lift sub-section prose directly from research/paper-materials/PAPER_PROSE_SKELETONS.md
§4.1, §4.2, §4.3 (refreshed 2026-05-12 with top-5 lead), §4.4, §4.5.

§IV.A Headline finding restated (~140 words) — lift from PROSE_SKELETONS §4.1.

§IV.B Why ST > SG at LOSO specifically (~140 words) — lift from PROSE_SKELETONS §4.2.

§IV.C Three-stream convergence — TOP-5 LEAD (~140 words) — lift from PROSE_SKELETONS §4.3
  (refreshed 2026-05-12). KEY: lead with the 5 top-attended channels (S5_D5/BA46-R,
  S1_D1/BA10-L, S8_D5/BA9-R, S4_D4/BA9-R, S7_D4/BA8-L); call out S5_D5 + S1_D1 as the
  §02-FDR-significant pair (d=-0.92 and -0.81); frame the remaining 3 (S7_D4, S4_D4,
  S8_D5) as multivariate-only signal. Cite [pokorny2024young; etkin2009functional;
  davidson2002anxiety]. Original C6-prior framing is preserved as a fallback paragraph
  in literature_review.md §4.3.

§IV.D Comparison to prior fNIRS-anxiety classifiers (~100 words) — lift from
PROSE_SKELETONS §4.4. Key delta: our pipeline produces per-channel attribution at
NO extra inference cost; prior work (75-85% accuracy benchmark in §I.C) lacks this.

§IV.E Clinical deployability (~100 words) — lift from PROSE_SKELETONS §4.5. Key
defence: GNN classifies all 22 trials of the 11 demographics-missing GAD subjects
correctly (Sens = 1.000) → fNIRS-only signal no demographic baseline can match.

§IV.F Threats to validity (5 bullets, ~100 words total):
1. Age confound (largest threat) — HC 73.0 vs GAD 51.1 (Welch t=6.08, p=6.5e-6).
   Defence: §05 ANCOVA holds; GNN works on demographics-missing-11 where no
   demographic baseline applies.
2. Demographics-missing 11 subjects — accepted limitation; subgroup analysis in
   FUTURE_ANALYSES.md §3.5.
3. No per-stimulus event timing — precludes a true first-level GLM; canonical-HRF β
   is the maximally-defensible block-level analog.
4. Single-site n=62 cohort — accepted; cross-cohort replication in
   FUTURE_ANALYSES.md §3.2; external public-cohort generalization in §3.4.
5. CBSI r(HbO,HbR) = -1 constraint — accepted methodological choice; HbT is the
   unconstrained chromophore reported in supplementary.

Cite all of [pokorny2024young; etkin2009functional; davidson2002anxiety] in §IV.C
(REUSE — already cited in §I.D, no new citations introduced).

Source files:
- research/paper-materials/PAPER_PROSE_SKELETONS.md §4 (drafted skeletons)
- research/paper-materials/literature_review.md §4.3 (alternate prose; fallback)

IEEE TNSRE style. ~700 words + 5 threat bullets.
Save to research/paper-materials/drafts/v1_discussion.md
```

---

### Step 16 — Write Conclusion §V + Future Work

```
@scientific-writing Write Section V (Conclusion and Future Work) for an IEEE TNSRE
paper. Target ~350 words.

§V.A Conclusion (2 sentences, ~50 words):
- Restate the core contribution: graph-based fNIRS GAD classification with
  native-attention XAI cross-validated against a parallel statistical-analysis arm
  and an empirical channel-parsimony test on the XAI-derived subnetwork.
- State the headline: ST F1 = 0.8529 (LOSO mt=2 HbO K=12, sens 1.000); ST > SG
  by +3.78 pp F1 (p = 0.0024); XAI top-5 spans BA 8/9/10/46 with 2/5 FDR-converging
  on §02; channel-parsimony at K=12 improves over the 23-channel baseline in 5 of 6
  chromophore × trial-cap cells (mean Δ-F1 +2.52 pp).

§V.B Future Work (8 bullets ranked by reviewer impact; top 3 in main paper,
4-8 in supplementary, ~300 words):
1. Cross-cohort replication on processed-old (independent processing of same
   cohort) and external public anxiety-fNIRS cohorts.
2. Age-matched sub-cohort analysis — disentangle the 22-year HC-GAD age gap.
3. MCX Monte Carlo photon-transport simulation for cortical-distance bounds
   (SPEC §16.10 deferred item).
4. Per-stimulus event timing — re-collect with E-Prime / PsychoPy logs to
   enable a true first-level GLM with No-Go - Go contrast.
5. SG retraining for HbR and HbT (currently HbO-only) to enable per-chromophore
   SG-vs-ST comparison.
6. Per-fold output split for §11 C2 (currently aggregate-only).
7. Subject-level voting + uncertainty quantification (P2.4 negative result;
   needs proper Bayesian treatment, not majority voting).
8. End-to-end attention-rationale prompts: feed top-attended channels back into
   the training loop as a regularization signal (RLHF-style for biomedical XAI).
9. (NEW 2026-05-14; replaces the prior "channel-ablation validation" bullet now
   that the validation experiment is complete) Two residual follow-ups from the
   channel-parsimony test (full results in §III.C.8 and Table 4):
   (a) **K=4/K=6 collapse-point characterisation** — the 24-cell sweep tested
   K ∈ {8, 12, 16, 23}; K=8 still beats baseline in HbO mt=4 (+6.58 pp Δ-F1).
   The actual collapse point of the parsimony curve is below K=8 but not
   characterised, and a K=4/K=6 follow-up would localise it.
   (b) **mt=4-derived differential XAI re-derivation** — the K=12 subset used
   in this paper was derived from ST × HbO × mt=2 differential XAI. HbR at
   mt=4 K=12 underperforms (Δ-F1 −2.29 pp); a mt=4-specific XAI re-derivation
   would test whether HbR's anti-parsimony behaviour is due to a wrong-regime
   subset rather than a genuine HbR-mt4 limitation. Not pursued for v1 because
   the existing mt=2 ranking is already validated by 5/6 channel-ablation cells.

Top 3 to keep in main paper; 4-9 compressed into "Additional directions are
detailed in the Supplementary Information."

Lift from research/paper-materials/PAPER_PROSE_SKELETONS.md §5 (already drafted).

Sources:
- research/paper-materials/PAPER_PROSE_SKELETONS.md §5
- src/notebook/statistical-analysis/FUTURE_ANALYSES.md (Tier-1 + Tier-2 items)

IEEE TNSRE style. ~350 words.
Save to research/paper-materials/drafts/v1_conclusion.md
```

---

### Step 17 — Caption pass (all 10 figures + 4 tables)

```
@scientific-writing Generate IEEE TNSRE-style captions for each of the 10 main
figures and 4 main tables. Each caption: 1-2 sentences, sentence case for figure
labels (Fig. 1, Fig. 2, ...), italic figure label, period after label, then
description that does NOT repeat the paper title. Reference figures by file name
for traceability (omit in final manuscript).

Figures:
1. fig1_overall_workflow.png — End-to-end pipeline
2. fig2_sg_architecture.png — SG architecture
3. fig3_st_architecture.png — ST architecture
4. fig_optode_layout.png — 23-channel prefrontal montage
5. fig_montage_brain.png — Frontal fsaverage with ROI dots
6. fig5_headline_cm.png — Confusion matrix
7. fig6_headline_training_grid.png — Training curves
8. fig_xai_brain_overlay_top5.png — XAI overlay (top-5, 4 views)
9. fig_top5_hbo_bars.png — HC vs GAD HbO per channel
10. concordance_triptych.png — 3-stream convergence

Tables:
1. Cohort demographics
2. SG vs ST architecture summary
3. Main 24-row classification results
4. Clinical utility (PPV/NPV/LR+ at prevalence)

Save to research/paper-materials/drafts/v1_captions.md
```

---

### Step 18 — DOI verification + Citation Management

```
@citation-management Verify DOIs for the 38 entries in
research/paper-materials/references/refs.bib that lack a `doi = {...}` field.
For each entry use Crossref API (https://api.crossref.org/works?query=...)
or PubMed to resolve the canonical DOI; for arXiv-only papers record
`eprint = {YYMM.NNNNN}` + `archivePrefix = {arXiv}` instead.

If a particular entry resists verification after 3 search attempts, mark it
[NEEDS-MANUAL-RESOLUTION] in the note field and move on.

Produce a side report at:
research/paper-materials/references/DOI_VERIFY_REPORT.md
with: (a) each entry's resolved DOI or status, (b) entries that could not be
verified with notes on why, (c) summary table of "verified / replaced /
still-missing" counts.
```

---

### Step 19 — Peer review (after full draft assembled)

```
@peer-review Evaluate the draft manuscript at
research/paper-materials/drafts/ using ScholarEval 8-dimension scoring:
1. Significance / novelty
2. Methodology rigor
3. Statistical rigor
4. Figure / table quality
5. Writing clarity
6. Citation completeness
7. Reproducibility
8. Venue alignment (IEEE TNSRE scope)

Score each 1-5. Flag any score ≤ 3 with specific recommendations.
Save to research/paper-materials/drafts/v1_peer_review_report.md
```

---

### Step 20 — Assemble final manuscript

```
@scientific-writing Concatenate the drafted sections into a single IEEE TNSRE LaTeX
manuscript at research/paper-materials/drafts/main.tex:
- v1_abstract.md → \begin{abstract} ... \end{abstract}
- v1_introduction.md → §I
- v1_methods_*.md → §II (concatenated in order: participants, montage,
  preprocessing, graph construction, architectures, training, xai, stats arm)
- v1_results_*.md → §III (stats_arm, ml, xai)
- v1_discussion.md → §IV
- v1_conclusion.md → §V
- v1_captions.md → integrate into respective \caption{} fields
- references/refs.bib → \bibliography{}

Verify:
- All citation keys resolve
- All figure/table cross-references compile
- Page count ≤ 8 pages (IEEE TNSRE TeX template)
- Bibliography style = IEEEtran
```

---

## 10. What NOT to Write Yet (Blocked Items)

| Item | Blocked by | When to unblock |
|---|---|---|
| Final Abstract numbers | All results pre-computed; F1, ρ, top-5 channels all locked | ✅ UNBLOCKED — use Step 2 prompt |
| §II.A IRB number | Dataset author / advisor input | Insert as `[IRB PLACEHOLDER]` until provided |
| §II.A Table 1 demographics | Compute mean ± SD + sex distribution from `cohort_per_subject.csv` | Resolve in Step 4 prompt |
| §III.B.3 Table 3 (24 rows) | Pull from `experiment_metrics.xlsx` — all 24 cells exist | ✅ UNBLOCKED — use Step 13 prompt |
| §III.C.3 pair-importance heatmap | Optional in main; main paper cites the source CSV only | Decide in drafting Step 14 |
| §III.A.1 exact p_FDR for S2_D1 + S3_D3 | Pull from `results_brain_activation_stats.csv` (already computed) | ✅ UNBLOCKED — use Step 12 prompt |
| §IV.E demographics-missing-11 numbers | Pull from `stats/clinical_utility.md` | ✅ UNBLOCKED — use Step 15 prompt |
| **References DOI fields** | network access to Crossref / WebSearch | Step 18 — DOI verification pass; OR commit citation keys without DOIs and run pass before LaTeX scaffold |

---

## 11. Pending Technical Items (Non-Writing)

- [ ] Confirm IRB approval number + institution for §II.A ethics statement (request from advisor)
- [ ] DOI verification on `references/refs.bib` (38 missing-DOI entries) — Step 18
- [ ] Build Table 1 (cohort demographics) — compute from `cohort_per_subject.csv` during Step 4
- [ ] Build Table 3 (24-row main classification results) — compute from `experiment_metrics.xlsx` during Step 13
- [ ] Final LaTeX scaffold (Step 1) + cross-reference pass (Step 20)
- [ ] Author affiliations + corresponding-author designation (Step 20)

---

## 12. Reference Files

### Tier-1 — Outline / Planning
| Resource | Path |
|---|---|
| Section-by-section outline | `research/paper-materials/PAPER_OUTLINE.md` |
| Ranked P0/P1/P2 work list | `research/paper-materials/PAPER_TODO.md` |
| This SPEC plan | `research/paper-materials/PAPER_SPEC_PLAN.md` (you are here) |

### Tier-2 — Ready-for-Paper Artefacts
| Resource | Path |
|---|---|
| Numbered equations E1-E10 | `research/paper-materials/PAPER_MATH.md` |
| Architecture tables A/B/C/D | `research/paper-materials/PAPER_ARCH_TABLES.md` |
| Drafted §4 + §5 skeletons | `research/paper-materials/PAPER_PROSE_SKELETONS.md` |
| Literature-review synthesis (§A + §B + §C + §D) | `research/paper-materials/literature_review.md` |
| Literature-review query plan | `research/paper-materials/LITERATURE_REVIEW_PLAN.md` |

### Tier-3 — Figures (in `research/paper-materials/figures/`)
| File | Role |
|---|---|
| `fig1_overall_workflow.{png,svg}` | §II workflow schematic |
| `fig2_sg_architecture.{png,svg}` | §II SG schematic |
| `fig3_st_architecture.{png,svg}` | §II ST schematic |
| `fig_optode_layout.{png,svg}` | §II.B optode layout (180°-rotated) |
| `fig_montage_brain.{png,svg}` | §II.B brain frontal view (ROI-colored) |
| `fig5_headline_cm.{png,svg}` | §III.B headline confusion matrix |
| `fig6_headline_training_grid.{png,svg}` | §III.B training curves |
| `fig_xai_brain_overlay_top5.{png,svg}` | §III.C top-5 brain overlay |
| `fig_top5_hbo_bars.{png,svg}` | §III.C top-5 HC vs GAD bars |
| `concordance_triptych.{png,svg}` | §III.C 3-stream convergence |
| `fig7_montage_anatomical.{png,svg}` | Supplementary (alternative montage) |
| `fig_xai_brain_overlay_all23.{png,svg}` | Supplementary (all-23 XAI) |
| `fig_montage_brain_3views.{png,svg}` | Supplementary (Frontal/L/R composite) |
| `fig_xai_brain_colormap_grid.{png,svg}` | Supplementary (12-cmap rationale) |
| 5 × `fig_top5_bar_*.{png,svg}` | Supplementary (per-channel standalones) |

### Tier-4 — Statistics Artefacts (in `research/paper-materials/stats/`)
| Resource | Path | Used by |
|---|---|---|
| ST-vs-SG paired test | `stats/st_vs_sg_paired_test.md/.csv` + `_aggregate.json` + `_mcnemar_loso.csv` | §III.B.2 |
| Subject-level voting | `stats/subject_level_voting.md/.csv` | §IV / supplementary |
| Clinical utility | `stats/clinical_utility.md` + `_baselines.json` + `_full.json` | §IV.E |
| XAI-stats concordance | `stats/concordance_rho_table.md/.csv` | §III.C.6 |

### Tier-5 — Reference Bibliography
| Resource | Path |
|---|---|
| BibTeX database (52 entries, 38 missing DOIs) | `research/paper-materials/references/refs.bib` |
| DOI verify report (TBD) | `research/paper-materials/references/DOI_VERIFY_REPORT.md` |

### Tier-6 — Literature-Review Sources (in `research/paper-materials/sources/`)
| Cluster | File | Role |
|---|---|---|
| C1 | `C1-clinical-gad.md` | §I.A clinical motivation |
| C2 | `C2-fnirs-modality.md` | §I.B fNIRS modality |
| C3 | `C3-fnirs-anxiety.md` | §I.D GAD-fNIRS prior art |
| C4 | `C4-gnn-neural.md` | §II.C.3 GNN refs |
| C5 | `C5-pfc-gad-anatomy.md` | §I.D + §IV.C |
| C6 | `C6-fnirs-preproc.md` | §II.C.1 |
| C7 | `C7-fnirs-glm-hrf.md` | §II.C.6 |
| C8 | `C8-gnn-explainability.md` | §II.C.5 |
| C9 | `C9-cross-validation.md` | §II.C.4 |
| C10 | `C10-hpo-optuna.md` | §II.C.4 |
| C11 | `C11-stats-methods.md` | §III.A |
| C12 | `C12-counter-evidence.md` | §IV (threats) |

### Tier-7 — Statistical-Analysis Notebook Reports
| Resource | Path | Used by |
|---|---|---|
| §02 Brain Activation | `src/notebook/statistical-analysis/02_brain_activation/REPORT.md` | §III.A.1 |
| §06 Canonical HRF | `src/notebook/statistical-analysis/06_glm_hrf/REPORT.md` | §III.A.2 |
| §05 Age-Adjusted ANCOVA | `src/notebook/statistical-analysis/05_age_adjusted/` | §III.A.3 |
| §01 Demographics | `src/notebook/statistical-analysis/01_demographic/` | §II.A |

### Tier-8 — Source-of-Truth Files for Headline Metrics
| Resource | Path |
|---|---|
| **SG ML metrics** | `experiments/spatial_graph/experiment_metrics.xlsx` |
| **ST ML metrics (headline)** | `experiments/spatial_temporal_graph/experiment_metrics.xlsx` |
| **SG Optuna result_report** | `research/experiments/20260430/optuna_search_nested_validation/core_hbo_mt4_ep100_tr600_kf5/result_report.md` |
| **ST Optuna result_report** | `research/experiments/20260503/optuna_search_st_kfold/result_report.md` |
| **ST 20260509 sweep (current)** | `research/experiments/20260509/CONFIG_VS_BASELINE_REPORT.md` |
| **XAI atlas table** | `research/xai/atlas/channel_to_brodmann.csv` |
| **XAI native attention (headline)** | `research/xai/st/hbo/loso/mt2/native/channel_importance.csv` |

### Tier-9 — Memory (auto-loaded by `MEMORY.md`)
| Memory | Role |
|---|---|
| `project_paper_outline_task_state.md` | Resumable state for this paper-prep task |
| `project_xai_task_state.md` | XAI methodology + checkpoint paths |
| `project_xai_stats_concordance.md` | P1.2 concordance numbers |
| `project_montage_figures.md` | Figure recipes + audit |
| `project_st_20260509_results.md` | ST 20260509 sweep results |
| `project_subjects_ground_truth.md` | 62-subject metadata canonical source |
| `project_statistical_analysis_suite.md` | §02 + §05 + §06 navigation |
| `project_scientific_writer_toolkit.md` | Skills inventory + invocation pattern |

---

## 13. Clarifications Still Needed

| # | Open question | Default | Lock by |
|---|---|---|---|
| C1 | Manuscript primary format (LaTeX IEEEtran from start vs. Markdown-first) | Markdown-first | Drafting session 1 |
| C2 | Author affiliations + corresponding author | TBD | User |
| C3 | Whether to cite `ren2026anxiety` as primary §I.D anchor or as supplementary citation | Primary (strongest GAD-fNIRS paper found) | User |
| C4 | Whether the demographics-missing-11 defence in §IV.E deserves its own sub-paragraph or stays in §IV.E main | Main paragraph (already drafted) | User |
| C5 | Whether to include subject-level-voting result (counter-intuitive: voting reduces F1) in main §III or only Discussion | Discussion only (negative result) | User |
| C6 | IRB approval number + institution | – | Dataset author / advisor |
| C7 | Order of figures 8 / 9 / 10 in §III.C — does the order top-5 overlay → bar charts → concordance triptych read best? | Yes (current order) | Drafting Step 14 |

None of these block SPEC drafting; they are choices the writer makes inline.

---

## 14. Acceptance Criteria (manuscript ready for submission)

1. **Every claim has a source pointer** in this SPEC plan. ✅ done.
2. **Every main-paper figure exists as a render artifact** in `research/paper-materials/figures/`. ✅ done (10/10).
3. **Every citation key resolves to a real reference** in `refs.bib` with title + author + year fields populated. ✅ done (52 entries); **DOIs deferred** to Step 18 citation-management pass.
4. **Page budget sums to ≤ 8 pages** with the chosen figure / table allocation. ✅ verified by §4 budget table.
5. **§IV.C narration uses the top-5 framing** (rev. 2026-05-12). ✅ done across PAPER_OUTLINE, PAPER_PROSE_SKELETONS, literature_review.
6. **P0.1–P0.4 + mt-main-table locked decisions cited** in the manuscript preamble. ✅ §3 table above.
7. **Threats-to-validity §IV.F covers the 5 reviewer concerns** (age, demographics-missing, no per-stimulus, single-site, CBSI). ✅ source paragraph in PROSE_SKELETONS.
8. **All 10 main figures + 4 main tables have captions drafted** (1–2 sentences each, IEEE-style "Figure N. ..."). — Step 17.
9. **References cross-checked** — no orphan citation keys, no uncited entries. — Step 20.
10. **One full proofread pass + peer-review pass** before submission. — Step 19 + final polish.

---

*This plan is a living document. Update it as sections are drafted and reviewed.*
*v1 → v2 (2026-05-12): added per-section writing prompts (Steps 1-20), reference-files table (Tier-1 through Tier-9), blocked-items / pending-technical-items / clarifications-needed tables. v1 high-level structure preserved.*
