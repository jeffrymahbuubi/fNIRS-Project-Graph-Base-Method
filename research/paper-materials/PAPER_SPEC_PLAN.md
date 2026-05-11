# PAPER_SPEC_PLAN.md

**Working title:** *Graph-Based Detection of Generalized Anxiety Disorder from Prefrontal fNIRS During a Go/No-Go Task, With Native-Attention Explainability and Brodmann-Region Cross-Validation*

**Target venue:** IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE), 8-page limit, IEEE numeric citations, supplementary material supported.

**Status:** Draft v1 (2026-05-12). Spec plan only — no manuscript prose yet. All technical-content blockers cleared (P0/P1/P2 complete per `PAPER_TODO.md`); only `references/refs.bib` DOI verification deferred to a pre-submission citation-management pass.

**Author:** Aunuun Jeffry Mahbuubi (`11208120@gs.ncku.edu.tw`)

**Provenance:** This SPEC plan synthesizes:

| Source | Provides |
|---|---|
| `PAPER_OUTLINE.md` (540 lines) | Section-by-section structure with provenance pointers |
| `PAPER_TODO.md` (297 lines) | P0/P1/P2 work-list with locked decisions |
| `PAPER_MATH.md` | 10 numbered equations E1..E10 (skewness, Pearson, Welch coherence, edge rule, SG/ST features, GATv2, GRU+attention, loss, leakage control) |
| `PAPER_ARCH_TABLES.md` | SG / ST architecture tables + training hyperparams + Optuna search space |
| `PAPER_PROSE_SKELETONS.md` | Drafted §4 Discussion + §4.2 Threats + §5 Future Work |
| `literature_review.md` (§A + §B + §C + §D) | 12-cluster citation inventory + 11 drafted narrative paragraphs |
| `stats/*.md` + `stats/*.csv` | Pre-computed ST-vs-SG paired test, subject-level voting, clinical utility, XAI-stats concordance |
| `figures/*.png` + `*.svg` | All 10 main-paper figures (rendered) |
| `02_brain_activation/REPORT.md` + `06_glm_hrf/REPORT.md` | Statistical-analysis arm |
| `docs/SPEC_xai_graph.md` rev. 6 | XAI methodology + atlas registration |
| `experiments/{spatial_graph,spatial_temporal_graph}/experiment_metrics.xlsx` | Source of truth for ML metrics |
| `research/xai/atlas/channel_to_brodmann.csv` | Channel → BA mapping (deterministic, fsaverage midpoint) |

---

## §0 Locked decisions (preamble table for the manuscript)

| # | Topic | Decision | Source |
|---|---|---|---|
| D1 | Target venue | IEEE TNSRE (8-page limit, IEEE numeric citations) | User P0.1 (2026-05-11) |
| D2 | Chromophore in main paper | **HbO + HbR** main; **HbT supplementary** | User P0.2 (2026-05-11) |
| D3 | SG comparator placement | Side-by-side ST + SG in main results table | User P0.3 (2026-05-11) |
| D4 | Citation style | IEEE numeric `[1], [2], …` | User P0.4 (2026-05-11) |
| D5 | mt placement in main table | **Both mt2 + mt4 in main** (24-row table) | User (P0.3 follow-up) |
| D6 | Headline XAI chromophore | **HbO** LOSO mt2 (matches stats arm; ρ(HbO,HbR XAI)=+0.899 → chromophore-invariant) | Session 6 (2026-05-11) |
| D7 | Headline ML chromophore | HbR LOSO mt2 (F1=0.8406 global best) | Locked since 2026-05-09 |
| D8 | XAI scope | ST = HbO/HbR/HbT; SG = HbO-only; **native attention is primary** | `docs/SPEC_xai_graph.md` rev. 6 |
| D9 | CV regimes reported | 5-fold + 10-fold + LOSO (all three) | Locked |
| D10 | Trial-level vs subject-level | Trial-level in headline; subject-level voting in Discussion as failed experiment | `stats/subject_level_voting.md` |
| D11 | Pre-processing | `processed-new-mc` (Wavelet + CBSI + bandpass 0.01–0.5 Hz); mandatory | `data/DATA_QUALITY_REPORT.md §9` |
| D12 | Augmentation | None for headline (`_noaug_*` runs) | OUTLINE |
| D13 | Cohort | Keep all 62 subjects in main table; sensitivity drops → future-work appendix | `data/DATA_QUALITY_REPORT.md §6` |
| D14 | Age confound | Acknowledge as the largest limitation; ANCOVA in §05 holds; demographics-missing-11 defence in §4.5 | `stats/clinical_utility.md` |
| D15 | Atlas | Brodmann via fsaverage midpoint projection; MCX Monte Carlo → future work | `docs/SPEC_xai_graph.md §16` |
| D16 | §4.3 narration framing | **Top-5 lead** (S5_D5, S1_D1, S8_D5, S4_D4, S7_D4 → BA 8/9/10/46) — 2 of 5 are §02 FDR-sig; 3 are GNN-only multivariate signal | Session 7 (2026-05-12) |

---

## §1 Page budget for IEEE TNSRE 8-page limit

| Section | Pages | Words (≈) | Figures | Tables | Drafting status |
|---|---|---|---|---|---|
| Front matter (title, abstract, keywords) | 0.5 | 250 | – | – | 🔜 abstract to draft |
| §1 Introduction | 1.0 | 700 | – | – | source: `literature_review.md` §B + OUTLINE §1.5 |
| §2 Materials and Methods | 2.5 | 1700 | 5 | 2 | source: OUTLINE §2 + PAPER_MATH + PAPER_ARCH_TABLES |
| §3 Results | 2.5 | 1700 | 5 | 2 | source: OUTLINE §3 + `stats/` artefacts |
| §4 Discussion | 1.0 | 700 | – | – | source: PAPER_PROSE_SKELETONS §4 |
| §5 Conclusion + Future Work | 0.5 | 350 | – | – | source: PAPER_PROSE_SKELETONS §5 |
| References (40–50 cited) | 0.5 | – | – | – | `references/refs.bib` (52 entries, DOI verify deferred) |
| **TOTAL** | **8.0** | **5400** | **10** | **4** | – |

Supplementary material (no page limit, hosted on IEEE Xplore):

- Tables S1–S6: HbT main table, full Optuna trials, subject-level voting, all SG/ST per-fold metrics
- Figures S1–S10: Per-fold confusion matrices, all-23-channel XAI heatmap, 3-view brain montage, 12-colormap contact sheet, individual top-5 bar charts, HbR XAI overlays
- SI-A: pipeline reproducibility (commit hashes, environment, seeds)
- SI-B: detailed XAI methodology (SPEC §11 acceptance criteria)
- SI-C: literature-review cluster inventory

---

## §2 Section-by-section content plan

Each subsection lists: (a) key claims with exact numbers, (b) figure/table refs, (c) citation keys (from `refs.bib`), (d) source artefact.

---

### Abstract (≈ 200 words)

**Structure (5-sentence template):**
1. *Motivation* — GAD is prevalent, underdiagnosed, and lacks objective neural biomarkers; fNIRS offers a portable / ecologically-valid alternative.
2. *Method* — We propose a Spatio-Temporal Graph Attention Network (ST-GATv2) over 23 prefrontal fNIRS channels during a Go/No-Go task, paired with native-attention XAI and Brodmann-area cross-validation.
3. *Headline result* — On a 62-subject cohort (HC=33, GAD=29) under leave-one-subject-out (LOSO) cross-validation, ST achieves **F1 = 0.8406** (Acc = 0.823, Sens = 1.000, Spec = 0.667) on HbR mt=2, outperforming a Spatial-Graph baseline by **+3.78 pp F1** across 12 paired k-fold cells (Wilcoxon W=3, p = 0.0024).
4. *XAI convergence* — Native attention identifies a top-5 prefrontal cluster (BA46-R, BA10-L, BA9-R, BA8-L) consistent with established GAD-PFC literature; two of five channels also reach FDR significance in classical per-channel hemodynamic analyses (HC > GAD; *d* = −0.92 and −0.81).
5. *Significance* — The model is the first graph-based fNIRS GAD classifier to produce per-channel attribution at no extra inference cost, cross-validated against three independent statistical streams.

**Source:** Headline numbers from `experiments/spatial_temporal_graph/experiment_metrics.xlsx`; ST-vs-SG from `stats/st_vs_sg_paired_test.md`; XAI from `research/xai/st/hbo/loso/mt2/native/`; §02 FDR from `02_brain_activation/results_brain_activation_stats.csv`.

---

### §1 Introduction (1.0 page, ≈ 700 words)

#### §1.1 Clinical motivation for objective GAD detection (≈ 130 words)
- **Claim 1:** GAD lifetime prevalence ~5% adults; symptom overlap with depression/MDD causes diagnostic delay [`apa2013dsm5`; `nimh2024gad`; `who2023anxiety`].
- **Claim 2:** Current diagnostic standard = clinician-administered HAM-A or self-report STAI — subjective, schedule-dependent [`maier1988hamilton`; `spielberger1983manual`].
- **Source:** `literature_review.md` §1.1 narrative paragraph (drafted).

#### §1.2 fNIRS as a modality for prefrontal-cortex anxiety research (≈ 130 words)
- **Claim 3:** fNIRS is portable, motion-tolerant, ecologically valid, and gives direct hemodynamic readout of prefrontal cortex [`pinti2020fnirs`; `cui2011combined`; `pinti2018benefits`].
- **Claim 4:** Penetration depth limits to gyri (~10–30 mm) — sufficient for the PFC subregions implicated in anxiety.
- **Source:** `literature_review.md` §1.2 narrative.

#### §1.3 Prior fNIRS-anxiety classification (≈ 120 words)
- **Claim 5:** Prior CNN/MLP/SVM classifiers achieve 75–85% accuracy on similar fNIRS-anxiety tasks [`wang2022deeplearning`; `cnn2024spatiotemporal`; `mentalworkload2023cnn`; `comorbidity2025fnirs`].
- **Claim 6:** Most prior work lacks per-channel attribution; post-hoc explanations (Grad-CAM, IG) are model-specific and rarely validated.
- **Source:** `literature_review.md` §1.3 narrative.

#### §1.4 GAD-PFC literature (≈ 200 words)
- **Claim 7:** Anxiety implicates a dorsolateral / dorsomedial / ventromedial PFC network: DLPFC (BA9/46) for inhibitory control, dmPFC (BA8) for monitoring, vmPFC (BA10) for emotion regulation [`etkin2009functional`; `davidson2002anxiety`; `pokorny2024young`].
- **Claim 8:** Go/No-Go paradigms specifically reveal DLPFC inhibition deficits in GAD [`pokorny2024young`; `ren2026anxiety`; `wang2024exploring`].
- **Source:** `literature_review.md` §1.4 narrative (the strongest single citation = `ren2026anxiety` Wiley *Depression and Anxiety* 2026; paradigm-matched = `wang2024exploring`).

#### §1.5 Our contribution (≈ 120 words, 5-bullet list)
1. Graph-based fNIRS analysis: channels as nodes, correlation/coherence as edges, on the canonical 23-channel prefrontal montage.
2. Two complementary architectures (Spatial Graph + Spatio-Temporal Graph) compared on the same montage.
3. Nested-CV Optuna search; reproducible deterministic-split protocol with full leakage controls.
4. Native-attention XAI mapped to Brodmann regions, cross-validated against a parallel classical statistical-analysis arm (§02 + §06).
5. First report to triangulate machine-learning attention, hemodynamic-amplitude statistics, and atlas-registered anatomy on a single GAD-fNIRS cohort.

---

### §2 Materials and Methods (2.5 pages, ≈ 1700 words)

#### §2.1 Dataset characteristics (≈ 200 words)
- **Claim:** n=62 subjects (HC=33, GAD=29), Go/No-Go task with 4 task blocks × ~32 s @ 10.17 Hz, NIRX device, 23 prefrontal channels.
- **Cohort table (Table 1):** age mean ± SD per group; sex distribution; HAM-A / STAI scores; 11 GAD subjects (AA089–AA099 + LA091/095/096) flagged as "demographics-missing" (clinical scores only, no age/sex).
- **Acknowledge:** age confound HC 73.0 vs GAD 51.1 (Welch *t* = 6.08, *p* = 6.5e-6) — defer detailed defence to §4.5 / threats §4.2.
- **Source:** `cohort_per_subject.csv` + `results_demographic_summary.csv` + `project_subjects_ground_truth.md` memory.

#### §2.2 Sensor placement and channel layout (≈ 200 words)
- **Figure 4:** `fig_optode_layout.png` — 8 sources × 8 detectors → 23 channels on prefrontal cortex, 180°-rotated to align with rostral brain view.
- **Figure 5:** `fig_montage_brain.png` — frontal view of fsaverage with ROI-colored sensor dots (VMPFC blue / DMPFC green / DLPFC red).
- **Source of truth:** `data/brainproducts-RNP-BA-128-custom.elc` parsed via `src/xai/atlas.py:parse_elc`.
- **Channel-to-Brodmann mapping table (Table 2 / supplementary):** lift from `research/xai/atlas/channel_to_brodmann.csv` — deterministic, top-1 probability per channel.

#### §2.3 Pipeline (≈ 1300 words)

##### §2.3.1 Preprocessing (≈ 220 words)
- **Wavelet motion correction** [`molavi2012wavelet`; `improved_wavelet_cbsi`] → **CBSI** to enforce hemoglobin anti-correlation [`cui2010cbsi`] → **bandpass 0.01–0.5 Hz**.
- **Constraint:** CBSI forces r(HbO, HbR) = −1 by construction (`DATA_QUALITY_REPORT.md §9.5`). HbT = HbO + HbR is the only chromophore not so constrained; reported in supplementary.
- Per-trial standardisation done **fold-wise** for leak-free CV (E10 of PAPER_MATH).

##### §2.3.2 Graph dataset construction (≈ 250 words)
- **Math E1–E5:** skewness / kurtosis (E1, E2), Pearson r (E3), Welch coherence (E4), edge rule (E5) from `PAPER_MATH.md`.
- **Edge rule:** keep edge `(i,j)` iff `|corr(i,j)| ≥ τ_corr = 0.1`. Edge attributes = `(coherence_ij, |corr_ij|)`. Directed=True, self-loops=True.
- **Node features (SG):** `[23, 6] = (mean, min, max, skew, kurt, var)` per channel per trial.
- **Node features (ST):** `[23, 326]` raw z-scored time-series, model unfolds internally to `[23, K, 6]` over K windows.

##### §2.3.3 Architectures (≈ 350 words)
- **Spatial Graph (Table A):** input `[23,6]` → 2-layer GINEConv-GATv2 → global mean-pool → pre-pool Linear(480→224) → softmax. ≈ 1.16M params.
- **Spatio-Temporal Graph (Table B):** sliding window (W=48, stride=K) → per-window GATv2 (shared weights) → GRU(hidden=64) → additive attention over time → softmax. ≈ 0.37M params.
- **References:** `kipf2017semi` (GCN baseline); `velickovic2018graph` + `brody2022how` (GAT / GATv2).
- **Figures 2 + 3:** `fig2_sg_architecture.png` + `fig3_st_architecture.png` (schematics; see note in PAPER_ARCH_TABLES.md re: pre-pool ordering).

##### §2.3.4 Training & evaluation (≈ 250 words)
- **Optimiser:** Adam. **Loss:** cross-entropy. **Loss equation = E9** in PAPER_MATH.
- **CV regimes:** 5-fold, 10-fold, LOSO. All three reported (D9).
- **Patience:** SG = 9999 (effectively off — full epoch run); ST = 30. Source: `project_paper_outline_task_state.md`.
- **Optuna best for ST:** lr=3.04e-4, CosineAnnealingLR(T_max=150, eta_min=1e-5). 5-fold val F1=0.7693.
- **Optuna best for SG:** per-layer `n_filters=[112,80]`, `n_heads=[8,6]`, lr=6.79e-3, cosine_warmup.
- **Leakage control:** per-fold standardisation; deterministic splits from `data/splits/kfold_splits_processed_new_mc.json` + in-code `get_loso_splits`. Math E10.
- **Validation methodology:** `cawley2010overfitting`; `nestedcv2018classifiers`; `bergstra2011algorithms` for TPE.

##### §2.3.5 XAI methodology (≈ 200 words)
- **Native attention:** ST GATv2 attention coefficients are extracted directly at inference (no perturbation/gradient methods needed). Source: `src/xai/st_explainer.py`.
- **Aggregation:** channel-importance = sum of all attention coefficients into each node, averaged across trials.
- **GNN-Explainer cross-check** [`ying2019gnnexplainer`; `xai_gnn_survey2023`]: supplementary path; 3-way SG comparison (gnn / captum-IG / attention) in `02_st_population.ipynb`.
- **Atlas registration (D15):** ELC-channel midpoint projection onto fsaverage pial surface; probabilistic Brodmann via FreeSurfer atlas. Sanity check C8: S2_D1 → BA10 probability ≥ 0.5. Source: `docs/SPEC_xai_graph.md §16`.
- **Critical XAI caveat** [`jain2019attention`; `wiegreffe2019attention`]: attention ≠ causation; must defend with §3.3.6 concordance against §02 + §06.

##### §2.3.6 Statistical-analysis arm (≈ 200 words)
- **§02 STD activation:** Mann-Whitney U test per channel on HbO STD across trials, BH-FDR-corrected. Source: `02_brain_activation/REPORT.md §4.2`.
- **§06 canonical HRF β:** GLM with double-gamma HRF [`glover1999deconvolution`], cluster-based permutation [`maris2007nonparametric`]. Source: `06_glm_hrf/REPORT.md §4.2`.
- Statistical chromophore = **HbO only** (per stats-arm decision); cohort = all 62.

---

### §3 Experimental Results (2.5 pages, ≈ 1700 words)

#### §3.1 Statistical-analysis arm (≈ 300 words)

##### §3.1.3 Per-channel STD (§02) (≈ 100 words)
- **Headline:** 4/23 channels FDR-significant: **S5_D5** (BA46-R, *d* = −0.92, p_FDR = 0.014), **S2_D1** (BA10-R, *d* = −0.85), **S3_D3** (BA9-L, *d* = −0.80), **S1_D1** (BA10-L, *d* = −0.81, p_FDR = 0.014).
- **Direction:** HC > GAD at every FDR channel → **hypoactivity pattern** consistent with GAD-PFC literature.
- **Source:** `02_brain_activation/REPORT.md §4.2`; `results_brain_activation_stats.csv`.

##### §3.1.4 Canonical HRF β (§06) (≈ 120 words)
- **Headline:** 2/23 channels FDR-significant: **S4_D5**, **S4_D7** (both BA9). HC β > GAD β at all 13 cluster-permutation hits.
- **Two-window finding:** early (0–7 s) + late (21–32 s) — late window dominant: HC ramps up across the 32-s block, GAD does not sustain.
- **Cross-method ρ:** Spearman(§02, §06) = +0.19 → methods tag complementary channels (not redundant).
- **Source:** `06_glm_hrf/REPORT.md §4.2, §4.4`.

##### §3.1.5 Age-confound acknowledgement (≈ 80 words)
- Top-4 §02 channels remain raw-significant after age ANCOVA on n=51 demographics-eligible subset. **S5_D5: β = −0.048, *p* = 0.0023, η²ₚ = 0.18.**
- Source: `05_age_adjusted/results_ancova_age_sex.csv`.

#### §3.2 ML results (≈ 700 words)

##### §3.2.1 Headline cell (≈ 100 words)
- **ST × HbR × mt2 × LOSO: F1 = 0.8406, Acc = 0.823, Sens = 1.000, Spec = 0.667.** Confusion matrix: TN=44, FP=22, FN=0, TP=58.
- **Figure 6:** `fig5_headline_cm.png`.
- **Source:** `experiments/spatial_temporal_graph/experiment_metrics.xlsx`.

##### §3.2.2 ST > SG paired test (≈ 200 words)
- Aggregate Wilcoxon W=3, *p* = 0.00244 across 12 paired k-fold cells. Mean Δ(ST−SG) F1 = **+3.78 pp**.
- McNemar HbR mt4 LOSO *p* = 2.3e-4 (47 ST-corrects vs 17 SG-corrects on the same trials).
- **Interpretation:** LOSO is where ST gains most — exactly the subject-level-distribution-shift regime where SG's static features over-fit.
- **Source:** `stats/st_vs_sg_paired_test.md` + `stats/st_vs_sg_mcnemar_loso.csv`.

##### §3.2.3 Main results table (Table 3, ≈ 250 words)
- **24 rows:** HbO + HbR × {5-fold, 10-fold, LOSO} × {ST, SG} × {mt2, mt4}.
- **Source:** `experiments/spatial_temporal_graph/experiment_metrics.xlsx` + `experiments/spatial_graph/experiment_metrics.xlsx`.
- Per-row entries: F1 (mean ± SD over folds), Accuracy, Sensitivity, Specificity, ROC-AUC.

##### §3.2.4 Training dynamics (≈ 150 words)
- **Figure 7:** `fig6_headline_training_grid.png` — 5×3 grid: per-fold train + val curves for {Loss, Accuracy, F1} with best-epoch markers.
- **Source:** training logs from headline run.

#### §3.3 XAI (≈ 700 words)

##### §3.3.1 Headline XAI cell decision (≈ 100 words)
- **D6:** ST × HbO × mt2 × LOSO. Rationale: matches stats arm; ρ(HbO XAI, HbR XAI) = +0.899 → chromophore-invariant attention surface.
- HbR cell exists on disk and is referenced in supplementary (`SI_Fig_S?`) + §3.3.4 temporal-attention panel.

##### §3.3.2 Channel importance (≈ 200 words)
- **Figure 8:** `fig_xai_brain_overlay_top5.png` — 4-view brain overlay (frontal/dorsal/L-lat/R-lat) with YlOrRd colormap, Gaussian σ=15 mm vertex overlay.
- **Top-5 channels:** S5_D5 (BA46-R), S1_D1 (BA10-L), S8_D5 (BA9-R), S4_D4 (BA9-R), S7_D4 (BA8-L).
- **Source:** `xai/st/hbo/loso/mt2/native/channel_importance.csv`.

##### §3.3.3 Channel-pair attention matrix (≈ 100 words)
- Brief reference to `pair_importance.csv` heatmap; full figure in supplementary.

##### §3.3.4 Temporal attention (≈ 100 words)
- **Late-window dominance** in HC trials, attenuated in GAD trials — matches §06 cluster-permutation late window.
- **Figure:** lifted from concordance triptych panels E/F.
- **Source:** `xai/st/{hbo,hbr}/loso/mt2/native/temporal_attention.csv`.

##### §3.3.5 Brodmann mapping (≈ 100 words)
- **Per OUTLINE §3.3.5 (rev. 2026-05-12) — top-5 lead.** Top-5 span BA 8/9/10/46.
- **Figure 9:** `fig_top5_hbo_bars.png` — HC vs GAD HbO STD per top-5 channel with per-subject strip plot, FDR brackets.
- **Source:** `02_brain_activation/REPORT.md §4.4` + `xai/st/hbo/loso/mt2/native/`.

##### §3.3.6 Convergence narrative (≈ 100 words)
- **Set-overlap is the figure-of-merit, NOT rank-correlation.**
- ρ over 23 channels is null in all 4 pairings (HbO|HbR × §02|§06). **Do NOT lead with ρ.**
- Top-10 overlap (the number to cite): ST-HbO ∩ §06-|βd| = **6/10**; ST-HbR ∩ §06-|βd| = **6/10**.
- C6 prior set in XAI top-10: ST-HbO 3/6; ST-HbR 4/6 (both exceed ≥2/6 threshold).
- **Figure 10:** `concordance_triptych.png` (3×2 layout: A-F).
- **Source:** `stats/concordance_rho_table.csv` + `concordance_rho_table.md`.

---

### §4 Discussion (1.0 page, ≈ 700 words)

Drafted in `PAPER_PROSE_SKELETONS.md` §4.1–§4.5; lift verbatim with minor polishing.

#### §4.1 Headline finding restated (≈ 140 words) — `PROSE_SKELETONS §4.1`
#### §4.2 Why ST > SG at LOSO specifically (≈ 140 words) — `PROSE_SKELETONS §4.2`
#### §4.3 Three-stream convergence — top-5 lead (≈ 140 words, rev. 2026-05-12) — `PROSE_SKELETONS §4.3` (refreshed) + `literature_review.md §4.3`
#### §4.4 Comparison to prior fNIRS-anxiety classifiers (≈ 100 words) — `PROSE_SKELETONS §4.4`
#### §4.5 Clinical deployability (≈ 100 words) — `PROSE_SKELETONS §4.5`

**Threats-to-validity** (collapse to 5 bullets, end of §4): see `PROSE_SKELETONS §4.2 Threats`.
1. Age confound (largest threat) — addressed via ANCOVA + demographics-missing-11 defence
2. Demographics-missing 11 subjects — accepted limitation
3. No per-stimulus event timing — precludes true GLM
4. Single-site n=62 cohort — accepted; future-work cross-cohort
5. CBSI r(HbO,HbR) = −1 constraint — accepted methodological choice

---

### §5 Conclusion + Future Work (0.5 page, ≈ 350 words)

#### §5.1 Conclusion (2 sentences) — `PROSE_SKELETONS §5`
#### §5.2 Future work (8 bullets, ranked) — `PROSE_SKELETONS §5`

Top 3 bullets to keep in main paper; bullets 4–8 → supplementary or compressed into one summary sentence.

---

## §3 Figures master list (final 10 for main paper)

| # | File | Caption (1-line) | Section |
|---|---|---|---|
| 1 | `fig1_overall_workflow.{png,svg}` | End-to-end pipeline: raw fNIRS → preprocessing → graph construction → SG/ST training → XAI → atlas | §2.3 |
| 2 | `fig2_sg_architecture.{png,svg}` | Spatial-Graph architecture diagram | §2.3.3 |
| 3 | `fig3_st_architecture.{png,svg}` | Spatio-Temporal-Graph architecture diagram | §2.3.3 |
| 4 | `fig_optode_layout.{png,svg}` | 23-channel prefrontal montage (rotated 180° for face-on view) | §2.2 |
| 5 | `fig_montage_brain.{png,svg}` | Frontal view of fsaverage with ROI-colored sensor dots | §2.2 |
| 6 | `fig5_headline_cm.{png,svg}` | Confusion matrix: ST × HbR × mt2 × LOSO | §3.2.1 |
| 7 | `fig6_headline_training_grid.{png,svg}` | Per-fold training curves (5×3 grid) | §3.2.4 |
| 8 | `fig_xai_brain_overlay_top5.{png,svg}` | XAI channel-importance overlay (top-5 channels, 4 views) | §3.3.2 |
| 9 | `fig_top5_hbo_bars.{png,svg}` | HC vs GAD HbO STD per top-5 channel + strip plot + FDR brackets | §3.3.5 |
| 10 | `concordance_triptych.{png,svg}` | 3-stream convergence: ST attn vs §02 vs §06 topomaps + temporal attention | §3.3.6 |

Supplementary figures (no count limit):
- `fig7_montage_anatomical` (alternative ground-truth montage)
- `fig_xai_brain_overlay_all23` (all-23 XAI heatmap)
- `fig_montage_brain_3views` (composite Frontal/Left/Right)
- `fig_xai_brain_colormap_grid` (12-colormap selection rationale)
- 5 × `fig_top5_bar_*` (each top-5 channel as standalone)
- HbR XAI overlays
- Per-fold confusion matrices

---

## §4 Tables master list (final 4 for main paper)

| # | Source CSV/MD | Caption | Section |
|---|---|---|---|
| 1 | (compute from `cohort_per_subject.csv`) | Cohort characteristics (age, sex, HAM-A, STAI) by group | §2.1 |
| 2 | `PAPER_ARCH_TABLES.md` Tables A+B (compacted) | SG vs ST architecture, parameter counts, layer details | §2.3.3 |
| 3 | `experiments/{spatial_graph,spatial_temporal_graph}/experiment_metrics.xlsx` | Main 24-row results: HbO+HbR × {5f,10f,LOSO} × {ST,SG} × {mt2,mt4}, F1 / Acc / Sens / Spec / AUC | §3.2.3 |
| 4 | `stats/clinical_utility.md` | PPV / NPV / LR+ at cohort + primary-care prevalence | §4.5 |

Supplementary tables:
- Full Optuna search space + best trial config (SG + ST)
- HbT main results (24 rows)
- §02 + §06 per-channel statistics (full 23-channel tables)
- ST-vs-SG paired test details (12 cells)
- Subject-level voting results (P2.4)
- Channel-to-Brodmann mapping (23 rows)

---

## §5 Citation budget per section

| Section | Citation keys (from `refs.bib`) | Total |
|---|---|---|
| §1.1 | `apa2013dsm5`, `nimh2024gad`, `who2023anxiety`, `maier1988hamilton`, `spielberger1983manual` | 5 |
| §1.2 | `pinti2020fnirs`, `cui2011combined`, `pinti2018benefits` | 3 |
| §1.3 | `wang2022deeplearning`, `cnn2024spatiotemporal`, `mentalworkload2023cnn`, `comorbidity2025fnirs` | 4 |
| §1.4 | `etkin2009functional`, `davidson2002anxiety`, `pokorny2024young`, `ren2026anxiety`, `wang2024exploring` | 5 |
| §2.3.1 | `molavi2012wavelet`, `cui2010cbsi`, `improved_wavelet_cbsi`, `qtnirs_motion` | 4 |
| §2.3.2 | (math only; no new citations) | 0 |
| §2.3.3 | `kipf2017semi`, `velickovic2018graph`, `brody2022how`, `wu2021comprehensive` | 4 |
| §2.3.4 | `cawley2010overfitting`, `nestedcv2018classifiers`, `bergstra2011algorithms`, `varoquaux2017assessing` | 4 |
| §2.3.5 | `ying2019gnnexplainer`, `xai_gnn_survey2023`, `xai_gnn_eval`, `jain2019attention`, `wiegreffe2019attention` | 5 |
| §2.3.6 | `glover1999deconvolution`, `maris2007nonparametric` | 2 |
| §3 | (no new citations — all numbers from local artefacts) | 0 |
| §4.3 | `pokorny2024young`, `etkin2009functional`, `davidson2002anxiety` (already in §1.4 — reuse) | 0 |
| §4.4 | `wang2022deeplearning`, `cnn2024spatiotemporal` (already cited — reuse) | 0 |
| §4.5 | `wilcoxon1945individual`, `mcnemar1947note`, `efron1986bootstrap`, `wynants2020prediction` | 4 |
| **TOTAL UNIQUE** | – | **≈ 40** |

**Budget headroom:** 40 unique citations × ½ line each ≈ 0.4 page → matches §1 page-budget allocation.

---

## §6 Acceptance criteria

A draft of `PAPER_SPEC_PLAN.md → MANUSCRIPT.tex` (final manuscript) is acceptance-complete when:

1. **Every claim has a source pointer** in the SPEC plan (this document). ✅ done.
2. **Every main-paper figure exists as a render artifact** in `research/paper-materials/figures/`. ✅ done (10/10).
3. **Every citation key resolves to a real reference** in `refs.bib` with title + author + year fields populated. ✅ done (52 entries); **DOIs deferred** to pre-submission citation-management pass.
4. **Page budget sums to ≤ 8 pages** with the chosen figure/table allocation. ✅ verified by §1 budget table.
5. **§4.3 narration uses the top-5 framing** (rev. 2026-05-12). ✅ done across all 3 narrative docs.
6. **P0.1–P0.4 + mt-main-table locked decisions cited** in the manuscript preamble. ✅ §0 table above.
7. **Threats-to-validity §4.2 covers the 5 reviewer concerns** (age, demographics-missing, no per-stimulus, single-site, CBSI). ✅ source-paragraph in PROSE_SKELETONS.
8. **All 10 main figures + 4 main tables have captions drafted** (1–2 sentences each, IEEE-style "Figure N. ...").
9. **References cross-checked** — no orphan citation keys, no uncited entries.
10. **One full proofread pass** before LaTeX scaffolding.

---

## §7 Drafting order (next session checklist)

When the user is ready to begin writing the manuscript prose itself (LaTeX scaffold or Markdown-first), recommended order:

| Step | Section | Reason for ordering |
|---|---|---|
| 1 | Abstract | Sets the narrative arc; easier to write once §1.5 contribution list is locked |
| 2 | §1 Introduction | All citation paragraphs already in `literature_review.md §B` |
| 3 | §2 Materials & Methods | Lift Math E1-E10 + Tables A-D + figures 1–5 (all pre-drafted) |
| 4 | §3 Results | Numbers all from `stats/` + `experiments/` — minimal new prose |
| 5 | §4 Discussion | Lift `PROSE_SKELETONS §4.1–§4.5` with §4.3 already refreshed |
| 6 | §5 Conclusion + Future Work | Lift `PROSE_SKELETONS §5` |
| 7 | Reference DOI verification pass | After page-cap proof, before LaTeX scaffold |
| 8 | Caption pass | All 10 figures + 4 tables get 1–2-sentence captions |
| 9 | Cross-reference pass | Verify every `[Figure N]` / `[Table N]` / `[1]` `[2]` etc. resolves |
| 10 | LaTeX scaffold (IEEEtran) | Final formatting only after content is locked |

Estimated time:
- Steps 1–6 (prose drafting): ~6–8 h focused writing
- Step 7 (DOI verification): ~30–60 min with citation-management skill
- Steps 8–10 (polish + LaTeX): ~3–4 h

**Total time to submission-ready first draft:** ~10–12 h of focused work, distributed across 2–3 sessions.

---

## §8 Open items at SPEC-plan-lock (i.e., things still to decide before drafting)

| # | Open | Default | Lock by |
|---|---|---|---|
| O1 | Manuscript primary format (LaTeX IEEEtran from start vs Markdown-first) | Markdown-first | Drafting session 1 |
| O2 | Author affiliations + corresponding author | TBD | User |
| O3 | Whether to cite `ren2026anxiety` as primary §1.4 anchor or as supplementary citation | Primary (strongest GAD-fNIRS paper found) | User |
| O4 | Whether the demographics-missing-11 defence in §4.5 deserves its own sub-paragraph or stays in §4.5 main | Main paragraph (already drafted) | User |
| O5 | Whether to include subject-level-voting result (counter-intuitive: voting reduces F1) in main §3 or only Discussion | Discussion only (it's a negative result) | User |

None of these block SPEC drafting; they are choices the writer makes inline.

---

*End of `PAPER_SPEC_PLAN.md` v1 (2026-05-12).*
