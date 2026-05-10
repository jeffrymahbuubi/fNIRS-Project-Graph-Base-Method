# PAPER_TODO — Work to do before drafting `PAPER_SPEC_PLAN.md`

> **Goal of this file.** Enumerate every open piece of information / decision /
> analysis that the SPEC plan needs as input, ranked by how badly it blocks the
> SPEC plan from being **detailed and complete**. Once every **P0** item is
> resolved, the SPEC plan can be drafted in a single sitting.
>
> **Companion files**
> - `research/paper-materials/PAPER_OUTLINE.md` — annotated outline, source of every cross-reference below.
> - Auto-memory: `project_paper_outline_task_state.md` — resumable task state.
>
> **Date created:** 2026-05-10 · **Last updated:** 2026-05-11
>
> **Progress log (2026-05-11 session — user decisions resolved):**
> - ✅ **P0.1 venue** — **IEEE TNSRE** (Transactions on Neural Systems and Rehabilitation Engineering). 8-page limit, IEEE numeric citations, supplementary supported. Drives §3.2 main-table size budget.
> - ✅ **P0.2 chromophore split** — **HbO + HbR in main; HbT in supplementary**. Matches statistical-analysis arm (HbO) + LOSO winner (HbR). HbT (CBSI-derived, r(HbO,HbR)=−1) goes to SI_Table_S1.
> - ✅ **P0.3 SG comparator placement** — **Side-by-side ST + SG in main results table** (NOT supplementary). User chose symmetric presentation over supplementary-only. **Implication flagged:** main table size depends on mt2/mt4 inclusion (see open question below).
> - ✅ **P0.4 citation style** — **IEEE numeric** `[1], [2], …`. `research/paper-materials/references/refs.bib` keys stay; bibliography style switches to IEEEtran/ACM-numeric when LaTeX scaffold lands.
>
> **Resolved 2026-05-11 — mt main-table inclusion:**
> ✅ **Both mt2 and mt4 in main results table** (24 rows total: HbO+HbR × {5-fold, 10-fold, LOSO} × {ST, SG} × {mt2, mt4}). Full ablation in main. **Implication for SPEC plan:** IEEE TNSRE 8-page budget is tight — main table will consume ~1.0–1.5 pages. SPEC plan §3.2 must keep figure count lean (recommend: 1 confusion-matrix figure + 1 training-curve figure + 1 ST-vs-SG comparison figure = 3 main figures). XAI section §3.3 should aim for 2 main figures (montage importance + temporal attention).
>
> **Progress log (2026-05-10 session 2):**
> - ✅ **P0.5 PAPER_MATH.md** — written, verified against `src/core/utils.py` and `src/core_st/models.py`, 10 numbered equations E1..E10 with source line references and citation key placeholders. Saved to `research/paper-materials/PAPER_MATH.md`.
> - ✅ **P0.6 PAPER_ARCH_TABLES.md** — Tables A (SG), B (ST), C (training hyperparams), D (Optuna search space). All hyperparameters cross-checked against the headline `config.yaml` files. Two key subtleties documented: (a) SG `pre_pool` Linear(480,224) is applied per-node BEFORE pool, not after — the inspiration figure draws this incorrectly; (b) ST has shared GATv2 weights across K=39 windows. Param counts: SG ≈ 1.16M, ST ≈ 0.37M.
> - ✅ **P0.7 figures (inspiration / PowerPoint reference)** — generated via `scientific-schematics` skill (Nano Banana 2):
>   - `figures/fig1_overall_workflow.png` (graphical abstract; minor: EXPLAINABILITY arrow drawn twice, `ℝ` unicode slightly garbled)
>   - `figures/fig2_sg_architecture.png` (clean, all hyperparameters accurate; pre-pool ordering wrong — see PAPER_ARCH_TABLES.md note)
>   - `figures/fig3_st_architecture.png` (content accurate; box numbering shuffled — 5/6 swapped visually, but pipeline meaning unchanged)
>   - `figures/fig4_montage.png` (headcap + 5×7 grid; not pixel-accurate to channel coordinates — use as anatomical-feel inspiration only)
>   - All saved with `_v1.png` originals and `_review_log.json` (Gemini reviewer hit a model-ID 400 error — review skipped, image generation succeeded).
> - ✅ **P0.8 SG Optuna result_report.md** — written at `research/experiments/20260430/optuna_search_nested_validation/core_hbo_mt4_ep100_tr600_kf5/result_report.md`. 519 completed + 81 pruned trials. Best Trial #67: F1 0.7377, n_filters=[112,80], heads=[8,6], lr=6.79e-3. Important finding: headline config flips 3 flags (use_residual T/F, use_norm F/T, use_gine_first_layer F/T) relative to Trial #67 best — these flips trace to the leak-free patience study, not this sweep, and must be flagged in the Methods section.
> - ✅ **P0.11 headline figures** — `figures/fig5_headline_cm.png/.svg` (LOSO HbR mt2: TN=44, FP=22, FN=0, TP=58 → Acc 0.823, Sens 1.000, Spec 0.667, F1 0.8406 — matches Excel headline exactly) and `figures/fig6_headline_training_grid.png/.svg` (5×3 grid: per-fold train+val curves for {Loss, Accuracy, F1} with best-epoch markers).
> - ✅ **P1.1 paired ST-vs-SG test** — `research/paper-materials/stats/st_vs_sg_paired_test.md` (full report) + 3 CSV/JSON outputs. **Headline: aggregate Wilcoxon across 12 paired k-fold cells, W=3, p=0.00244, mean Δ(ST−SG) F1 = +3.78 pp.** McNemar at LOSO: HbR mt4 p=2.3e-4 (47 vs 17 discordant), HbO mt4 p=0.0023, HbT mt4 p=0.0040; mt2 cells trend ST but n=124 underpowered. 11/12 paired k-fold cells favour ST (only HbR mt2 10-fold favours SG by 1.2 pp).
> - ✅ **P1.4 anatomical montage figure** — `figures/fig7_montage_anatomical.{png,svg}` + `figures/fig7_montage_anatomical_data.csv` (23 rows of channel midpoints in head-coordinate mm). Two-panel: (A) anatomically-accurate top-down scalp view with all 8 sources, 8 detectors, 23 channel midpoints, head outline + Nz/LPA/RPA fiducials + nose triangle, all rendered from `data/brainproducts-RNP-BA-128-custom.elc`; (B) the 5×7 grid mapping with every channel labeled at its (row, col) per `src/xai/channels.py:GRID_POS`. **This is the ground-truth montage figure** — the AI-generated `fig4_montage.png` was illustrative only; use this one for the SPEC plan.
> - ✅ **P2.4 subject-level voting** — `stats/subject_level_voting.md` + `stats/subject_level_voting.csv`. Headline: subject-level majority voting **decreases** F1 by 3–5 pp across all 6 ST cells (HbR mt2 LOSO: 0.8406 → 0.7945, ΔF1 = −0.046). Sens stays at 1.000; Spec drops because errors are subject-specific not random per-trial. Best subject-level cell: ST × HbR × mt4 LOSO at F1 = 0.8056. Recommendation: trial-level F1 stays as the headline; subject-level reported as supplementary sensitivity analysis.
> - ✅ **P2.5 clinical utility metrics** — `stats/clinical_utility.md` + 2 JSON files. PPV/NPV at 4 prevalences (3 % → 0.085, 6 % → 0.161, 20 % → 0.429, cohort 46.7 % → 0.724); NPV = 1.000 throughout (model never misses a positive); LR⁺ = 3.00, LR⁻ = 0.00. Bootstrap 95 % CI: F1 [0.771, 0.902], Spec [0.551, 0.778]. **Critical age-confound finding:** LogReg(age, sex) LOSO baseline on n=51 subset achieves F1 = 0.8750 / AUC = 0.916 — *higher* than GNN restricted to same n=51 (F1 = 0.766). **Defence:** GNN classifies all 22 trials of the 11 demographics-missing GAD subjects (Sens = 1.000) — fNIRS-only signal that no demographic baseline can match. Both findings flagged for §3.2.4 / §5.1 of the manuscript.
> - ✅ **P2.1 + P2.2 + P2.3 prose skeletons** — `PAPER_PROSE_SKELETONS.md`. Five-bullet Discussion outline (§4.1–§4.5), five-bullet Threats-to-Validity (§4.2 of paper), eight-bullet ranked Future Work list. Every bullet ends with a `(source: …)` pointer to a project file.
> - ✅ **P1.3 literature review (research-lookup pass)** — `LITERATURE_REVIEW_PLAN.md` + `literature_review.md` + `references/refs.bib` + 12 `sources/C{1..12}-*.md` raw search outputs. **26 queries via parallel-cli search; ~50 papers identified across 12 thematic clusters.** Headline cluster C3 (fNIRS-anxiety classification) yielded 8+ papers including Ren 2026 Depression and Anxiety (large-sample fNIRS in anxiety), Wang 2024 (task-based Go/No-Go fNIRS), Frontiers Psychiatry 2026 (ML+fNIRS biomarkers), and a 2025 MDD/GAD comorbidity paper. Initial BibTeX skeleton (refs.bib) has 38 entries; **all are placeholders requiring DOI verification by `citation-management` skill before manuscript submission**. 6 canonical refs marked `[MANUAL]` (Etkin 2009, Cui 2010 CBSI, Glover 1999, Maris-Oostenveld 2007, Wu 2021 GNN survey, Kipf 2017 GCN) need title-only resolution since they were not surfaced by the keyword searches. 10 of 12 clusters fully meet acceptance criteria; C2 (fNIRS modality) and C4 (GNN-on-neural) are adequate with manual additions.
>
> **Convention.** Each item: `ID. Title — what / why / where it lands / effort` where:
> - **what** = the deliverable
> - **why** = why the SPEC plan needs this *now* (not after drafting)
> - **lands** = the file/section that will receive the output
> - **effort** = rough time estimate

---

## ⏸ POSTPONED — Notebook-execution items (resume after `02_st_population.ipynb` finishes)

> **Status (2026-05-10):** User is running `src/notebook/xai/02_st_population.ipynb`
> in a separate session — several hours expected. Three tasks block on its
> outputs and are explicitly **postponed** until those outputs land on disk.
>
> **DO NOT attempt to execute these in this session.** When the notebook
> finishes, the next session should:
>
> ### Resume protocol (run in order)
>
> 1. **Verify P0.9 outputs exist.** Check that the following directories are populated (CSV / NPY files non-empty):
>    ```
>    research/xai/st/{hbo,hbr,hbt}/{kfold-5,kfold-10,loso}/mt2/native/
>      ├── node_importance.csv
>      ├── edge_importance.csv
>      ├── channel_pair_matrix.npy
>      ├── temporal_attention.csv
>      ├── result_meta.json
>      ├── run.json
>      └── plots/fig_montage_channel_importance.{png,svg}
>            fig_pair_matrix.{png,svg}
>            fig_temporal_attention.{png,svg}
>    ```
>    Minimum needed for the paper headline = the **HbR LOSO mt2 native** cell. Other cells are nice-to-have for ablation tables.
>
> 2. **Run `04_atlas_registration.ipynb`** (P0.10) — produces:
>    ```
>    research/xai/atlas/channel_to_brodmann.csv          ← single source of truth
>    research/xai/atlas/channel_midpoints_mni.csv
>    research/xai/atlas/registration_run.json
>    ```
>    Plus per-cell BA-region re-aggregations under `research/xai/atlas/{sg,st}/...`. Validates SPEC §11 acceptance C8 (S2_D1 → BA10 probability ≥ 0.5).
>
> 3. **Then build P1.2 concordance figure** (saliency–statistics convergence). Combines P0.9 ST attention with `02_brain_activation/results_brain_activation_stats.csv` (Cohen d) and `06_glm_hrf/results_canonical_hrf_beta.csv` (canonical β). Outputs:
>    ```
>    research/paper-materials/figures/concordance_triptych.{png,svg}
>    research/paper-materials/stats/concordance_rho_table.csv
>    ```
>    Required Spearman ρ checks:
>    - ST attention (HbR) rank vs |Cohen d| (HbO §02) — expected ρ ≥ 0.4 to pass C6 acceptance.
>    - ST attention (HbR) rank vs |canonical β d| (HbO §06).
>    - Cross-chromophore: ST HbO attention vs ST HbR attention (within-architecture stability).
>
> 4. **Re-tick the acceptance gate** at the bottom of this file once 1–3 are done.
>
> ### What CAN still be done while waiting
>
> - P0.1–P0.4 user decisions (venue, chromophore split, SG main/supp, citation style) — pure decisions, no compute needed.
> - P1.1 paired ST-vs-SG test — done in this session (see Progress log).
> - P1.3 literature review draft — needs P0.4 (citation style) decision but `refs.bib` skeleton can begin.
> - P1.4 sensor-placement figure — replot via matplotlib + `data/brainproducts-RNP-BA-128-custom.elc` for anatomical accuracy (current Fig 4 is illustrative only).
> - P2.x polish items (threats-to-validity skeleton, future-work bullets, discussion outline).

---

## P0 — Blockers (SPEC plan cannot be detailed without these)

### Decisions (must come from the user; no analysis required)

#### P0.1. Pick target venue
- **what.** Choose one of: IEEE TBME / TNSRE / Frontiers in Neuroscience / NeuroImage Reports / MICCAI / EMBC / venue-agnostic.
- **why.** Drives word count, figure budget, supplementary policy, IMRAD-vs-clinical structure, math typesetting style, citation style, and reference cap. Almost every other P0 item compounds with this choice.
- **lands.** Top-of-file note in `PAPER_SPEC_PLAN.md`; affects every section's length budget.
- **effort.** 5 min (decision only).

#### P0.2. Lock chromophore main-vs-supplementary split
- **what.** Decide whether the main results table reports {HbO, HbR, HbT} or only the LOSO winner (HbR) + 1 comparator (HbO). Default recommendation in the outline: **HbO + HbR in main, HbT supplementary**.
- **why.** Determines whether §3.2 has a 6-row, 12-row, or 18-row main table.
- **lands.** §3.2 of `PAPER_SPEC_PLAN.md`.
- **effort.** 5 min.

#### P0.3. Lock SG comparator placement
- **what.** Decide whether SG numbers go in the main paper or only in supplementary. Default recommendation: **SG → supplementary; ST = main**.
- **why.** Determines whether §3.2 has a side-by-side comparison table or a comparison summary figure pointing to supplementary.
- **lands.** §3.2 of `PAPER_SPEC_PLAN.md`.
- **effort.** 5 min.

#### P0.4. Pick citation style + initialise `refs.bib`
- **what.** Pick one of {APA, IEEE, Vancouver, Nature, ACM} consistent with P0.1. Create an empty `research/paper-materials/refs.bib`.
- **why.** Every literature-review and discussion citation needs a stable key.
- **lands.** `research/paper-materials/refs.bib`.
- **effort.** 5 min for decision; population happens in P0.5.

### Mathematical formulae (currently absent from the codebase docs — must be derived/written)

#### P0.5. Write the maths box for §2.3.2 (graph dataset construction)
- **what.** Single Methods box (½–¾ page) containing all formulas the paper will reference:
  - **Skewness** `g₁ = m₃ / σ³` and **kurtosis** `g₂ = m₄ / σ⁴` (Pearson form, no Fisher correction). Verify against `src/core/utils.py:47–50`.
  - **Pearson correlation matrix** `C_ij = cov(x_i, x_j) / (σ_i σ_j)`.
  - **Welch coherence matrix** `Coh_ij(f) = |S_ij(f)|² / (S_ii(f) S_jj(f))` with `seg_length = N/3` (per `src/core/utils.py:100`).
  - **Edge-selection rule** `(i, j) ∈ E ⟺ |C_ij| ≥ τ_corr`, default τ = 0.1; edge feature `e_ij = (Coh_ij, |C_ij|)`.
  - **SG node feature** `x_i = (μ_i, min_i, max_i, g₁_i, g₂_i, σ²_i) ∈ ℝ⁶`.
  - **ST raw input + window-stat transform** `Data.x = z-score([23, 326])` → `unfold(W, S)` → `[23, K, W]` → window stats `[23, K, 6]`.
  - **GATv2 message passing** `α_ij = softmax_j(a_θᵀ · LeakyReLU(W [h_i ‖ h_j ‖ W_e e_ij]))`; cite Brody et al. 2022.
  - **GRU + additive attention** over K windows `g_k = GRU(h_k)`, `a_k = softmax_k(v_aᵀ · tanh(W_a g_k))`, `h̄ = Σ_k a_k g_k`.
  - **Loss** binary cross-entropy on softmax(logits[2]).
  - **Leakage control** `(μ_train, σ_train) := compute_stats(train_indices)` per fold; both train and val standardised by these.
- **why.** The user's outline §2.3.2 explicitly flagged math is missing. Without these expressions the SPEC plan cannot fix figure / equation numbering.
- **lands.** `research/paper-materials/PAPER_MATH.md` (new file). Each formula numbered (E1..E10) so the SPEC plan can cite them.
- **effort.** 2–3 h (the harder part is verifying every formula matches the actual code; do not paraphrase).

#### P0.6. Write architecture tables for SG and ST
- **what.** Two compact tables, ≤ 7 rows each, each row = one architectural component (input shape, layer, layer config, output shape, activation, normalisation). Lift from `src/core_st/README.md` lines 27–47 (ST) + analogous reading of `src/core/models.py:FlexibleGATNet` (SG). Include both as drop-in tables for §2.3.3.
- **why.** Outline §2.3.3.1 and §2.3.3.2 explicitly say "Table architecture configurations" — tables don't exist yet.
- **lands.** `research/paper-materials/PAPER_ARCH_TABLES.md` (new file). Becomes Table 2 + Table 3 in the paper.
- **effort.** 1.5 h.

#### P0.7. Render two architecture schematics (figures)
- **what.** Two flow diagrams (vector / SVG):
  - **SG.** `Trial[23, 326]` → per-channel stats `[23, 6]` + edge graph → GATv2 stack → mean pool → FC → 2 logits.
  - **ST.** `Trial[23, 326]` z-scored → unfold W=16, S=8 → `[23, 39, 16]` → window stats `[23, 39, 6]` → shared GATv2 per window → pool → GRU → additive attention → FC → 2 logits.
- **why.** Required as Figures 2 + 3 in §2.3.3; the SPEC plan needs to know they exist before assigning figure numbers.
- **lands.** `research/paper-materials/figures/SG_pipeline.svg`, `research/paper-materials/figures/ST_pipeline.svg`. (Existing `assets/SG-vs-ST_schematic.png` may be reusable as a starting point.)
- **effort.** 3–4 h (publication-quality vector).

### Missing analyses (small jobs that the SPEC plan needs to cite)

#### P0.8. Generate `result_report.md` for the SG Optuna sweep
- **what.** Produce an SG analog of the ST Optuna report. The ST report at `research/experiments/20260503/.../result_report.md` is the template — same headers and sections. Source DB: `research/experiments/20260430/optuna_search_nested_validation/core_hbo_mt4_ep100_tr600_kf5/optuna_study.db` (600 trials).
- **why.** §2.3.4.3 of the outline cites a "best-trial summary" for SG that does not exist on disk. The SPEC plan would otherwise have to insert "TBD" — defeats the purpose.
- **lands.** `research/experiments/20260430/optuna_search_nested_validation/core_hbo_mt4_ep100_tr600_kf5/result_report.md`.
- **effort.** 1.5 h (load study, top-15 + marginal-effects + recommended config, mirror ST template).

#### P0.9. Run the headline-cell XAI artefacts (ST × {HbO, HbR, HbT} × {kfold-5, kfold-10, LOSO} × mt2)
- **what.** Execute `src/notebook/xai/02_st_population.ipynb` for the **9 cells** that the paper will reference (3 chromophores × 3 regimes; mt2 only — paper headline). Default kwargs from SPEC §2.1 rev. 6 (chromophores=("hbo","hbr","hbt")).
- **why.** §3.3 of the paper depends on `node_importance.csv`, `edge_importance.csv`, `channel_pair_matrix.npy`, `temporal_attention.csv`, and the figures `fig_montage_channel_importance.png`, `fig_pair_matrix.png`, `fig_temporal_attention.png` for at least the LOSO HbR mt2 cell. The SPEC plan can't number figures it can't point at.
- **lands.** `research/xai/st/{hbo,hbr,hbt}/{kfold-5,kfold-10,loso}/mt2/native/...`.
- **effort.** 2–3 h GPU time (notebook is already wired; just run cells).

#### P0.10. Run atlas registration (`04_atlas_registration.ipynb`)
- **what.** Execute the notebook once to produce `research/xai/atlas/channel_to_brodmann.csv` and `channel_midpoints_mni.csv`. Then re-aggregate every P0.9 cell at the BA-region level (notebook auto-discovers).
- **why.** §3.3.5 of the paper claims Brodmann mapping; this needs the table to exist before the SPEC plan can describe the figure ("region-importance bar chart" and "BA × BA region-pair heatmap").
- **lands.** `research/xai/atlas/...` (per `docs/SPEC_xai_graph.md §16`).
- **effort.** 1–2 h (one-time fsaverage download + run).

#### P0.11. Render the headline-cell confusion matrix + training-curve grid
- **what.**
  - **Confusion matrix** for ST × HbR × mt2 × LOSO from `experiments/spatial_temporal_graph/loso/ST_GATv2_GNG_hbr_loso_mt2_noaug_20260509/...kfold_overall.pkl` → publication PNG/SVG.
  - **Training-curve grid** = 5 (folds) × 3 (F1/loss/accuracy) panel from the per-fold PNGs already in `experiments/spatial_temporal_graph/5-fold/ST_GATv2_GNG_hbr_kfold_mt2_noaug_20260509/*.png`.
- **why.** §3.2.3 and §3.2.4 of the outline reference these figures. Both files already have the raw artefacts; only assembly is needed.
- **lands.** `research/paper-materials/figures/headline_cm.svg`, `research/paper-materials/figures/headline_training_grid.svg`.
- **effort.** 30 min – 1 h.

---

## P1 — Must-haves before submission (SPEC plan can stub these but must allocate space)

### P1.1. Paired statistical test "ST > SG"
- **what.** Either DeLong's test on overall ROC curves of ST vs SG for the matched cell (HbR mt2 LOSO), or paired Wilcoxon on per-fold F1 for the matched cell (5-fold or 10-fold). Report *p*-value + effect size.
- **why.** Reviewer-level requirement to substantiate the "ST beats SG" claim. Outline §3.2.3 flags this as TODO.
- **lands.** `research/paper-materials/stats/st_vs_sg_paired_test.md` + a row in the §3.2 table.
- **effort.** 1 h.

### P1.2. Saliency–statistics concordance figure (XAI vs |Cohen's d|)
- **what.** 2-row × 2-col triptych of 5×7 grid topomaps: row 1 = ST attention HbO + HbR; row 2 = §02 |d| (HbO) + §06 canonical-β |d| (HbO). Plus a small Spearman ρ table comparing the four ranks.
- **why.** Outline §3.3.6 names this as the convergence check (FUTURE_ANALYSES.md §1.4 also). It is the single strongest "model uses biology" defence.
- **lands.** `research/paper-materials/figures/concordance_triptych.svg` + `concordance_rho_table.csv`.
- **effort.** 2–3 h (data already exists from P0.9 + §02 + §06).

### P1.3. Literature-review draft + bibliography population
- **what.** Survey ~12 representative papers on:
  - Prefrontal-fNIRS GAD/SAD findings (Yang 2020, Bauernfeind 2014, Tupak 2014, Husain 2020, Yeung 2020, Pereira 2018).
  - Graph methods on fNIRS / EEG (Velickovic 2018 GAT, Brody 2022 GATv2, Wu 2021 GNN review, Saeidi 2022, Mehmood 2024).
  - GAD-PFC anatomy refs (Etkin 2009, Mochcovitch 2014, Shin 2009).
- **why.** §1.4 is empty in the outline. Drafting it during the SPEC plan adds 4–6 h of unbudgeted work.
- **lands.** `research/paper-materials/refs.bib` + `research/paper-materials/literature_review.md` (1.5–2 page draft).
- **effort.** 4 h.

### P1.4. Sensor placement / 23-channel layout figure
- **what.** Paper-quality figure showing 8 sources × 8 detectors → 23 channels on the prefrontal scalp, with the 5×7 grid mapping shown side-by-side. Use `data/brainproducts-RNP-BA-128-custom.elc` geometry.
- **why.** §2.2.1 of the outline. Standard required figure for an fNIRS paper.
- **lands.** `research/paper-materials/figures/montage.svg`.
- **effort.** 2–3 h (`src/xai/atlas.py` already loads the ELC, so geometry is solved — pure rendering work).

### P1.5. Decide which atlas figure goes in §3.3.5
- **what.** Choose between `fig_montage_brodmann.png` (5×7 grid coloured by BA) vs `fig_surface_atlas.png` (3-D fsaverage scatter). Default recommendation: 5×7 grid in main, surface in supplementary.
- **why.** Open question 4 in the outline. Affects §3.3.5 layout.
- **lands.** §3.3.5 of `PAPER_SPEC_PLAN.md`.
- **effort.** 5 min (decision); files already exist after P0.10.

---

## P2 — Polish (not blocking; SPEC plan can defer)

### P2.1. Threats-to-validity skeleton in §4.2
- **what.** 3–5 bullets: age confound, demographics-missing-11, no per-stimulus event timing, single-site n=62, possible motion-correction residual.
- **lands.** §4.2 of `PAPER_SPEC_PLAN.md`.
- **effort.** 30 min.

### P2.2. §5 Future-work bullets
- **what.** Lift 8–12 bullets from `src/notebook/statistical-analysis/FUTURE_ANALYSES.md` into the §5 list of `PAPER_OUTLINE.md` (already partially populated). Keep each bullet to 1 sentence.
- **lands.** §5.2 of `PAPER_OUTLINE.md` / `PAPER_SPEC_PLAN.md`.
- **effort.** 30 min.

### P2.3. Discussion outline (§4)
- **what.** 5-bullet skeleton covering: headline finding, why ST > SG at LOSO specifically, three-stream convergence (§3.1.3 + §3.1.4 + §3.3), comparison to prior fNIRS-anxiety classifiers, deployability.
- **lands.** §4 of `PAPER_SPEC_PLAN.md` (full prose populated AFTER results are locked).
- **effort.** 1 h skeleton.

### P2.4. Subject-level voting analysis (Future Work add-on)
- **what.** Re-aggregate per-trial predictions to per-subject via majority vote and soft vote; report new F1. **Caveat:** with only 4 trials per subject, voting is statistically thin — describe as exploratory.
- **why.** Reviewers will ask. Best to address in Future Work, not main results.
- **lands.** §5 of paper.
- **effort.** 1 h (post-aggregation only — no retraining).

### P2.5. Clinical utility metrics (PPV/NPV at GAD prevalence + DeLong)
- **what.** Compute PPV/NPV at GAD prevalence 3–6 % (community) and ~20 % (primary care). DeLong test against HAMA-only baseline.
- **why.** `FUTURE_ANALYSES.md §1.5` — strongly recommended. Boosts clinical-reviewer reception.
- **lands.** §3.2 supplementary or §4 discussion.
- **effort.** 2 h.

---

## Suggested execution order (linear path through P0)

1. **P0.1, P0.2, P0.3, P0.4** (one sitting, ~30 min) — pure decisions; unblocks everything else.
2. **P0.5, P0.6, P0.7** (writing + drawing day, ~6–8 h) — math + tables + diagrams; can be done offline without GPU.
3. **P0.8** (1.5 h) — SG Optuna report; small Python session.
4. **P0.10** (1–2 h) — atlas registration; one-time prerequisite for P1.5.
5. **P0.9** (2–3 h GPU) — XAI sweep for the 9 cells.
6. **P0.11** (30–60 min) — confusion matrix + training-curve grid; once P0.9 has run.
7. **P1.1, P1.2, P1.3, P1.4, P1.5** in any order (~9 h total) — once P0 is closed.
8. **Begin drafting `PAPER_SPEC_PLAN.md`** — at this point every section can be specified with concrete numbers, equations, figure file paths, and reference keys.

---

## What "ready for SPEC plan" looks like (acceptance gate)

When all of the following are true, drafting `PAPER_SPEC_PLAN.md` becomes a writing task, not a research task:

- [ ] P0.1–P0.4 decisions written into the SPEC plan's preamble (one short table).
- [ ] `PAPER_MATH.md` exists with E1..E10 numbered formulas, each verified against the codebase.
- [ ] `PAPER_ARCH_TABLES.md` exists with the SG and ST tables.
- [ ] Two pipeline schematics (SG, ST) live in `research/paper-materials/figures/` as SVG.
- [ ] SG Optuna `result_report.md` exists alongside the ST one.
- [ ] At least the LOSO HbR mt2 ST XAI cell has populated `research/xai/st/hbr/loso/mt2/native/...` outputs (preferably all 9 cells).
- [ ] Atlas table exists (`research/xai/atlas/channel_to_brodmann.csv`).
- [ ] Headline confusion matrix + training-curve grid figures exist.

P1 items should be in flight but do not block the first SPEC plan draft — they can be added between SPEC v1 and SPEC v2.
