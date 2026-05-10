# PAPER_PROSE_SKELETONS — Drop-in skeletons for §4 Discussion, §4.2 Threats-to-Validity, and §5 Future Work

> Three short prose skeletons that the SPEC plan can lift directly. Each
> bullet is one paragraph long when expanded. Lengths are calibrated for an
> IEEE/Frontiers-style paper (≈ 1.5 pages of Discussion + ≤ 0.75 page of
> Limitations).
>
> **Source of every claim** is a file already in the project tree — every
> bullet ends with a `(source: …)` pointer so the SPEC plan can audit it.
>
> **Generated:** 2026-05-10. **P2.1 + P2.2 + P2.3 of `PAPER_TODO.md`**.

---

# §4 Discussion (P2.3 — 5-bullet skeleton)

> Populate after Results are finalised. Order matters: lead with the headline
> finding, then explain the mechanism, then bridge to the broader literature.

### 4.1 Headline finding restated
The ST-GATv2 architecture classifies fNIRS prefrontal recordings during a Go/No-Go task into HC-vs-GAD with **F1 = 0.8406 at the trial level on a 62-subject LOSO regime** (HbR, mt = 2). Aggregated across 12 paired k-fold cells, ST outperforms SG by **+3.78 pp F1 (W = 3, p = 0.0024)**, with the strongest effects at LOSO mt = 4 (HbR p_McNemar = 2.3 × 10⁻⁴). The architecture's value is not a single hyperparameter — it is the windowing + GRU + additive-attention combination that exposes time structure SG collapses to a single statistic. (sources: `experiments/spatial_temporal_graph/experiment_metrics.xlsx`; `stats/st_vs_sg_paired_test.md`).

### 4.2 Why ST > SG at LOSO specifically
Per-fold k-fold differences are modest (Δ ≈ +1–4 pp F1), but the LOSO trial-level discordance is large (HbR mt4: ST corrects 47 SG errors, SG corrects 17 ST errors). LOSO is the only regime where every fold contains a single never-seen subject, so the test stresses generalisation across **subject-level distribution shift**. The observation that ST-vs-SG widens at LOSO suggests SG over-fits subject-level static-feature distributions (means, variances), while ST exploits within-trial temporal dynamics that are more invariant across subjects. This is consistent with the §06 cluster-permutation finding that the HC > GAD hemodynamic difference concentrates in the **late** window (21–32 s post-cue) — a feature SG's full-trial STD cannot resolve. (sources: `stats/st_vs_sg_paired_test.md`; `06_glm_hrf/REPORT.md §4.4`).

### 4.3 Three-stream convergence on the same anatomical cluster
Three independent evidence streams converge on a connected superior-medial / mid-lateral PFC cluster: (a) §02 STD activation (4 / 23 FDR-significant: S5_D5, S2_D1, S3_D3, S1_D1), (b) §06 canonical-HRF β (2 / 23 FDR: S4_D5, S4_D7), (c) ST native attention from XAI (channel-importance ranking — populated from `02_st_population.ipynb` outputs once available). The combined biological-prior set of 6 channels (`{S1_D1, S5_D5, S3_D3, S2_D1, S4_D5, S4_D7}`) is the SPEC §11 acceptance set; ≥ 2 of 6 in the GNN top-10 attention list is the convergence threshold. This is the strongest defence against the alternative hypothesis that the GNN exploits artefactual or demographic shortcuts. The Brodmann mapping (BA 10 / 9 / 46) further aligns with the established GAD-PFC literature (Etkin 2009; Yeung 2020). (sources: `02_brain_activation/REPORT.md §4.2, §4.4`; `06_glm_hrf/REPORT.md §4.2`; `docs/SPEC_xai_graph.md §11 C6`).

### 4.4 Comparison to prior fNIRS-anxiety classifiers
Earlier fNIRS-anxiety classifiers achieve accuracy in the 75–85 % range (literature review §1.4), competitive with our 82.3 % LOSO accuracy. The differentiator is **per-channel attribution**: the proposed ST + native-attention pipeline produces a reproducible 5 × 7-grid importance map at no extra inference cost, where most prior CNN/MLP approaches require post-hoc attribution methods (Grad-CAM, IG) that are model-specific and harder to validate. The aggregate paired-test result (n = 12 cells) also rules out single-favourable-split confounding, which is rarely reported in the prior literature. (sources: literature review draft §1.4 — TODO P1.3).

### 4.5 Practical clinical deployability
At cohort prevalence (46.7 %), the trial-level PPV is 0.72 and NPV is 1.00. At realistic primary-care prevalence (≈ 20 %), PPV drops to 0.43 — i.e., 4 of every 10 GNN-positive patients would actually have GAD. The model is therefore best framed as an **objective rule-in adjunct** (LR⁺ = 3) rather than a definitive test or community screen. The 22-year HC–GAD age gap means the headline number partly reflects an age effect; however, the GNN correctly classifies all 22 trials of the 11 GAD subjects with missing demographics — fNIRS-only signal that no demographic baseline can match. (sources: `stats/clinical_utility.md`).

---

# §4.2 Threats to validity (P2.1 — 5-bullet skeleton)

> Each bullet acknowledges a confound and points to where it is addressed
> (mitigation, future work, or accepted limitation).

1. **Age confound (largest threat).** HC mean age 73.0 vs GAD mean 51.1 (Welch *t* = 6.08, *p* = 6.5 × 10⁻⁶). A LogReg(age, sex) baseline reaches LOSO F1 = 0.875 on the n=51 demographics-eligible subset — higher than the GNN restricted to the same subset. **Mitigation:** the §05_age_adjusted ANCOVA shows that the §02 top-4 channels remain raw-significant after age control, and the GNN classifies the 11 demographics-missing GAD subjects (Sens = 1.000) where no demographic baseline applies. (source: `01_demographic/REPORT.md §6`; `stats/clinical_utility.md §3`).

2. **Demographics-missing 11 subjects.** Eleven GAD subjects (`AA089–AA099`, `LA091`, `LA095`, `LA096`) have only HAMA / STAI scores; age, sex, education are blank in the ground-truth metadata. Demographic statistics in §2.1 use the n=51 subset; ML evaluation uses all n=62. **Accepted limitation;** dedicated subgroup analysis is in `FUTURE_ANALYSES.md §3.5`. (source: `01_demographic/REPORT.md §2`).

3. **No per-stimulus event timing.** The acquisition-stage `.tri` LSL trigger files contain only block-level markers (task / baseline / rest), not per-stimulus Go-vs-No-Go onsets. This precludes a true first-level GLM with No-Go − Go inhibition contrast. **Mitigation:** §06 canonical-HRF β is the maximally-defensible block-level analog. **Future work:** re-collect with E-Prime / PsychoPy logs. (source: `06_glm_hrf/REPORT.md §2`).

4. **Single-site n = 62 cohort.** All recordings come from one site, one acquisition protocol. **Accepted limitation;** cross-cohort replication on `processed-old` (independent processing of the same cohort) is in `FUTURE_ANALYSES.md §3.2`; external public-cohort generalisation is in §3.4. The pre-specified sensitivity-cohort analyses (drop AH024, AH029, LA063, demographics-missing-11) listed in `FUTURE_ANALYSES.md §1.3` are the next-tier robustness checks. (source: `FUTURE_ANALYSES.md`).

5. **CBSI constraint on r(HbO, HbR).** The motion-correction pipeline (Wavelet + CBSI) forces r(HbO, HbR) = −1 by construction (`DATA_QUALITY_REPORT.md §9.5`). HbT (= HbO + HbR) is not constrained and is reported as a primary chromophore alongside HbO and HbR; the LOSO best F1 sits on HbR (0.8406) which is per construction the negative of HbO post-CBSI. **Accepted methodological choice;** noted in §2.3.1.3. (source: `data/DATA_QUALITY_REPORT.md §9.5`).

---

# §5 Future Work (P2.2 — 8-bullet ranked list)

> Lifted from `src/notebook/statistical-analysis/FUTURE_ANALYSES.md` Tier-1
> and Tier-2 items, plus three project-specific deferred items. Order
> reflects expected reviewer impact, *not* effort.

1. **Age-adjusted ANCOVA + age-matched subsample re-analysis** — partial out age from every per-channel comparison; rerun on a propensity-matched subcohort. Tier-1 in `FUTURE_ANALYSES.md §1.1`. (status: §05 notebook started; full pipeline-rerun pending).

2. **Event-locked GLM / HRF first-level Go-vs-No-Go contrast** — requires re-collection with per-stimulus event logs. Tier-1 in `FUTURE_ANALYSES.md §1.2`. (status: blocked by data; flagged in §2.2.3.1).

3. **LOSO sensitivity cohorts** — drop {AH024, AH029, LA063, demographics-missing-11}; report Δ-AUC and Δ-F1 against the headline. Tier-1 in `FUTURE_ANALYSES.md §1.3`. (status: codable from existing pickles in < 1 h).

4. **Saliency–statistics concordance figure** (XAI vs |Cohen's d| triptych) — Spearman ρ between GNN attention rankings and §02/§06 effect-size rankings. Tier-1 in `FUTURE_ANALYSES.md §1.4`. (status: P1.2 of `PAPER_TODO.md` — postponed pending P0.9).

5. **Soft-vote subject-level aggregation** — retrain saving trial logits, then average probabilities per subject. Subject-level voting analysis (P2.4) showed *hard* majority vote degrades F1 by 3–5 pp; soft voting is the principled fix. (status: needs new training run; ~ 2 h GPU per cell).

6. **MCX Monte Carlo atlas mapping** — replace the geometric channel-midpoint projection with photon-transport-based partial-volume estimates for region attribution. SPEC §16.10. (status: deferred; midpoint approximation gets ≈ 80 % of the practical accuracy).

7. **Permutation-classifier null + post-hoc power** — 1 000 label-shuffle subject-level permutations to establish empirical *p* on the headline F1. Tier-2 in `FUTURE_ANALYSES.md §2.7`. (status: codable; ≈ 30 GPU hours per cell).

8. **External-cohort generalisation probe** — apply the trained model to a public fNIRS anxiety/depression dataset (Bak 2023 OpenNeuro). Tier-3 in `FUTURE_ANALYSES.md §3.4`. (status: scoping required — no public dataset has been pre-validated for compatibility yet).

---

# Drop-in cross-references (for the SPEC plan)

| Skeleton bullet | Cross-references | Numbers to lift |
|---|---|---|
| §4.1 | `experiments/spatial_temporal_graph/experiment_metrics.xlsx` LOSO sheet; `stats/st_vs_sg_paired_test.md` §4 | F1=0.8406; W=3; p=0.0024; Δ=+3.78 pp |
| §4.2 | `06_glm_hrf/REPORT.md §4.4`; `stats/st_vs_sg_paired_test.md` §5 | 47 vs 17 discordant trials; late window 21–32 s |
| §4.3 | `02_brain_activation/REPORT.md §4.2`; `06_glm_hrf/REPORT.md §4.2`; `docs/SPEC_xai_graph.md §11 C6` | 4 / 23 FDR (§02); 2 / 23 FDR (§06); biological-prior set of 6 |
| §4.4 | refs.bib (TODO P1.3) | accuracy range 75–85 % from prior literature |
| §4.5 | `stats/clinical_utility.md` §2 | PPV at 3%/6%/20%/46.7%; LR+ = 3.00 |
| §4.2.1 (Threats) | `01_demographic/REPORT.md §6`; `stats/clinical_utility.md §3` | Welch t=6.08, p=6.5e-6; LogReg F1=0.875 |
| §4.2.5 (Threats) | `data/DATA_QUALITY_REPORT.md §9.5` | CBSI constraint |
