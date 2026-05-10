# LITERATURE_REVIEW_PLAN — what needs citation support, by section

> **Purpose.** Before launching `research-lookup` queries, enumerate every
> argument in the paper that requires literature backing, grouped into
> thematic clusters. Each cluster becomes one focused query batch; the
> outputs land in `research/paper-materials/sources/<cluster_id>.md`, then
> a single `literature_review.md` synthesises them into Section 1.4 of the
> paper plus the inline citations scattered through §1, §2, §4, §5.
>
> **Scope.** Only **citations the paper genuinely needs to defend**. Not a
> general review of all of fNIRS or all of GNNs. Each "needs" entry below
> traces to a specific sentence or claim in `PAPER_OUTLINE.md`,
> `PAPER_PROSE_SKELETONS.md`, or `PAPER_MATH.md`.
>
> **Generated:** 2026-05-10. **Author:** Aunuun Jeffry Mahbuubi.
> **Style note.** Citation keys below are placeholders (`author2024shortname`)
> — once P0.4 (citation style) is decided they will be normalised by
> `citation-management` skill. For now use these as stable identifiers in
> drafts.

---

## How to read this file

Each cluster has:
- **Cluster ID** (e.g. `C1-clinical-gad`) — file slug for `sources/<id>.md`.
- **Maps to paper §** — which section(s) consume this cluster's papers.
- **Argument(s) to defend** — what the paragraph claims that needs support.
- **Target paper count** — minimum number of papers to cite for this argument.
- **Search queries** — drop-in `research-lookup` invocations.
- **Acceptance criterion** — when this cluster is "done".

The clusters are ordered by **execution priority** (clusters earlier in the
list block later ones).

---

## Cluster C1 — Clinical context & disease background of GAD

**Cluster ID:** `C1-clinical-gad`
**Maps to paper §:** §1.1 Clinical and disease explanation; §1.2 Traditional anxiety assessment.
**Argument to defend:**
1. GAD is a clinically significant disorder (DSM-5 criteria, lifetime prevalence ~5–6 %, comorbidity profile).
2. Standard assessment is HAMA / STAI-S / STAI-T — all clinician/self-report scales subject to recall bias and cultural variability — therefore an objective biomarker is needed.

**Target paper count:** 4–5.

**Search queries (3 to run):**
1. `Generalized anxiety disorder DSM-5 prevalence epidemiology adult`
2. `Hamilton Anxiety Rating Scale HAMA validity reliability`
3. `STAI State Trait Anxiety Inventory psychometric properties`

**Expected canonical references (likely hits):**
- `kessler2012lifetime` — Kessler et al. 2012 *Lifetime prevalence of mental disorders* (NCS-R)
- `bandelow2015generalized` — Bandelow & Michaelis 2015 *Epidemiology of anxiety disorders in the 21st century* (Dialogues Clin Neurosci)
- `apa2013dsm5` — DSM-5
- `maier1988hamilton` — original HAMA validity paper
- `spielberger1983manual` — STAI manual / `julian2011measures`

**Acceptance criterion.** ≥ 1 prevalence/epidemiology paper, ≥ 1 paper for each of HAMA and STAI psychometric grounding, ≥ 1 paper noting limitations of self-report scales (motivates objective biomarkers).

---

## Cluster C2 — fNIRS as a modality

**Cluster ID:** `C2-fnirs-modality`
**Maps to paper §:** §1.3 fNIRS compared to other modalities.
**Argument to defend:**
1. fNIRS measures cerebral hemodynamics via NIR-light absorption by HbO/HbR.
2. Compared to fMRI / EEG / MEG / PET, fNIRS offers a unique balance: cheaper, portable, high temporal resolution (~10 Hz), tolerant to motion, ecologically valid, BUT limited to cortical surface (~3 cm depth).
3. fNIRS is mature enough for clinical research (Best-Practices Yücel 2021).

**Target paper count:** 3–4.

**Search queries (2 to run):**
1. `fNIRS functional near-infrared spectroscopy modality comparison fMRI EEG`
2. `fNIRS best practices reporting Yücel 2021 publication standards`

**Expected canonical references:**
- `pinti2020fnirs` — Pinti et al. 2020 *Present and future use of fNIRS in everyday-life environments* (Nat Rev Methods Primers / Ann NY Acad Sci)
- `yucel2021best` — Yücel et al. 2021 *Best practices for fNIRS publications* (Neurophotonics)
- `quaresima2019review` — Quaresima & Ferrari 2019 review
- `ferrari2012brief` — Ferrari & Quaresima 2012 *A brief review on the history of NIRS*

**Acceptance criterion.** ≥ 1 review paper covering modality strengths/weaknesses; ≥ 1 reporting-standards paper; explicit mention of cortical-depth limitation.

---

## Cluster C3 — fNIRS in anxiety / GAD classification (THE MOST IMPORTANT cluster)

**Cluster ID:** `C3-fnirs-anxiety`
**Maps to paper §:** §1.4 Literature review (the BIG block) + §4.4 Discussion comparison.
**Argument to defend:**
1. Multiple groups have used fNIRS to classify anxiety / depression / SAD / GAD vs healthy controls. Sample sizes range 20–150; accuracies 70–90 %.
2. Most prior work uses static features (channel statistics, GLM betas) and shallow classifiers (SVM, RF, MLP).
3. Few prior studies report **per-channel attribution** in a clinically-interpretable way → our paper differentiates here.

**Target paper count:** 8–10 (this is the headline literature block; reviewers expect comprehensive coverage).

**Search queries (3 to run):**
1. `fNIRS anxiety classification machine learning prefrontal cortex`
2. `fNIRS generalized anxiety disorder GAD social anxiety SAD discrimination`
3. `fNIRS depression major depressive disorder classification deep learning`

**Expected canonical references:**
- `yang2020classification` — Yang et al. 2020 fNIRS GAD classification
- `bauernfeind2014fnirs` — Bauernfeind et al. 2014 anxiety fNIRS
- `tupak2014inhibition` — Tupak et al. 2014 prefrontal hypoactivation in anxiety (fNIRS GNG)
- `husain2020fnirs` — Husain et al. 2020 fNIRS depression deep learning
- `yeung2020frontotemporal` — Yeung et al. 2020 frontotemporal fNIRS in MDD/GAD
- `pereira2018differential` — Pereira et al. 2018 fNIRS anxiety differential
- `yu2020classification` — Yu et al. 2020 fNIRS classifier
- `qureshi2020fnirs` — Qureshi et al. 2020 fNIRS anxiety/depression deep learning

**Acceptance criterion.** ≥ 8 papers spanning: (a) anxiety/SAD/GAD-specific fNIRS, (b) MDD-fNIRS as analogous methodology, (c) at least 2 with reported sensitivity / specificity / F1 numbers (so we can compare). Build a summary table: study, sample size, paradigm, classifier, accuracy.

---

## Cluster C4 — Graph neural networks on neural / EEG / fNIRS data

**Cluster ID:** `C4-gnn-neural`
**Maps to paper §:** §1.5.1 Graph modality strength; §2.3.3 (Methods cite for GAT/GATv2); §4 Discussion.
**Argument to defend:**
1. Channels in multi-channel neural recordings have natural graph structure (spatial proximity + functional connectivity); flat models discard this.
2. GNNs (especially GAT/GATv2) are now the standard for graph-structured neural data.
3. Few applications of GNNs to fNIRS yet → our paper is novel in this combination.

**Target paper count:** 6–8.

**Search queries (3 to run):**
1. `graph attention network GAT GATv2 neural data EEG fMRI`
2. `graph neural network fNIRS classification brain network`
3. `functional connectivity graph deep learning brain disorder`

**Expected canonical references:**
- `velickovic2018graph` — Veličković et al. 2018 *Graph Attention Networks* (ICLR)
- `brody2022how` — Brody, Alon, Yahav 2022 *How attentive are graph attention networks?* (ICLR — GATv2)
- `wu2021comprehensive` — Wu et al. 2021 *A comprehensive survey on graph neural networks* (TNNLS)
- `kipf2017semi` — Kipf & Welling 2017 GCN
- `saeidi2022graph` — Saeidi et al. 2022 graph methods on fNIRS
- `mehmood2024graph` — Mehmood et al. 2024 graph fNIRS (per project memory)
- `klepl2024graph` — Klepl et al. 2024 graph methods on EEG
- `wein2021brain` — Wein et al. 2021 brain network GNN

**Acceptance criterion.** GAT (2018) + GATv2 (2022) cited; ≥ 2 GNN-on-fNIRS papers; ≥ 2 GNN-on-EEG/fMRI for methodology breadth.

---

## Cluster C5 — Prefrontal-cortex anatomy & GAD neuroimaging findings

**Cluster ID:** `C5-pfc-gad-anatomy`
**Maps to paper §:** §3.1.3 anatomical convergence narrative; §3.3.5 Brodmann mapping; §4.3 three-stream convergence; Discussion.
**Argument to defend:**
1. GAD is associated with **prefrontal-cortex hypoactivation** (especially during inhibitory tasks like Go/No-Go).
2. Specific PFC regions implicated: VMPFC, DMPFC, DLPFC; Brodmann areas 9, 10, 46.
3. Our XAI-derived important channels (S5_D5, S2_D1, S3_D3, S1_D1, S4_D5, S4_D7) map to these areas → biological plausibility.

**Target paper count:** 5–6.

**Search queries (3 to run):**
1. `generalized anxiety disorder prefrontal cortex hypoactivation inhibitory control`
2. `anxiety Go/No-Go task prefrontal Brodmann fMRI activation`
3. `vmPFC dlPFC anxiety neuroimaging meta-analysis`

**Expected canonical references:**
- `etkin2009functional` — Etkin et al. 2009 *Functional neuroimaging of anxiety: a meta-analysis* (Am J Psychiatry)
- `mochcovitch2014meta` — Mochcovitch et al. 2014 GAD neuroimaging meta-analysis
- `shin2009prefrontal` — Shin & Liberzon 2010 *Neurocircuitry of fear, stress, and anxiety disorders*
- `yang2018altered` — Yang et al. 2018 altered PFC GAD fMRI
- `aupperle2012neural` — Aupperle et al. 2012 emotional regulation in anxiety
- `paulus2015anxiety` — Paulus 2015 anxiety as predictive coding

**Acceptance criterion.** ≥ 1 meta-analysis (Etkin or equivalent); ≥ 1 GAD-specific neuroimaging paper; ≥ 1 paper covering Go/No-Go in anxiety.

---

## Cluster C6 — fNIRS pre-processing methodology (motion correction, CBSI, GLM)

**Cluster ID:** `C6-fnirs-preproc`
**Maps to paper §:** §2.3.1 Signal processing techniques; Methods supplementary (DATA_QUALITY_REPORT integration).
**Argument to defend:**
1. Motion artefacts are pervasive in fNIRS; Wavelet motion correction is standard (Molavi & Dumont 2012).
2. CBSI (Correlation-Based Signal Improvement) is the canonical method for forcing physiologically-correct r(HbO,HbR) (Cui et al. 2010).
3. Bandpass 0.01–0.5 Hz preserves task-band hemodynamics while suppressing physiology.

**Target paper count:** 4–5.

**Search queries (2 to run):**
1. `fNIRS motion correction wavelet Molavi Dumont`
2. `CBSI correlation-based signal improvement fNIRS Cui`

**Expected canonical references:**
- `molavi2012wavelet` — Molavi & Dumont 2012 *Wavelet-based motion artifact removal for fNIRS* (Physiol Meas)
- `cui2010quantitative` — Cui, Bray, Reiss 2010 CBSI (NeuroImage)
- `huppert2009homer2` — Huppert et al. 2009 Homer3 toolbox
- `pollonini2014auditory` — Pollonini et al. 2014 SCI
- `tachtsidis2016false` — Tachtsidis & Scholkmann 2016 *False positives and false negatives in fNIRS*

**Acceptance criterion.** Wavelet motion correction primary citation + CBSI primary citation + Homer3 reference (since `nirs2csv_homer3_mc.m` is project-internal).

---

## Cluster C7 — GLM / canonical HRF / first-level analysis (for §06 method defence)

**Cluster ID:** `C7-fnirs-glm-hrf`
**Maps to paper §:** §3.1.4 GLM/HRF analysis; §06_glm_hrf supporting cite.
**Argument to defend:**
1. Canonical HRF (Glover-style γ-variate) is the standard for fNIRS first-level activation analysis.
2. Cluster-based permutation (Maris & Oostenveld) is the standard correction for time-resolved t-tests.
3. STD-of-amplitude (the §02 metric) is event-agnostic; canonical β is the principled alternative.

**Target paper count:** 3–4.

**Search queries (2 to run):**
1. `canonical HRF haemodynamic response function Glover SPM fNIRS`
2. `cluster permutation Maris Oostenveld EEG fNIRS multiple comparisons`

**Expected canonical references:**
- `glover1999deconvolution` — Glover 1999 deconvolution of HRF
- `maris2007nonparametric` — Maris & Oostenveld 2007 (Cluster-permutation, the canonical reference)
- `friston1998event` — Friston et al. 1998 event-related fMRI design
- `pinti2020analysis` — Pinti et al. 2020 *fNIRS analysis methods* (a chapter)

**Acceptance criterion.** Maris-Oostenveld 2007 + Glover 1999 cited.

---

## Cluster C8 — XAI for graph neural networks

**Cluster ID:** `C8-xai-gnn`
**Maps to paper §:** §1.5.2 AI explainability; §2.3.3 (XAI methods); §3.3 XAI section; §4.5 deployability.
**Argument to defend:**
1. Black-box DL models are unsuitable for clinical decisions — need interpretable predictions.
2. GNNExplainer / native attention / Captum-IG are the three standard graph-XAI families.
3. Native attention is the most defensible because no extra training / no extra forward pass.

**Target paper count:** 4–5.

**Search queries (2 to run):**
1. `GNNExplainer graph neural network explainability node edge mask`
2. `attention interpretability transformer neural network limitations`

**Expected canonical references:**
- `ying2019gnnexplainer` — Ying et al. 2019 GNNExplainer (NeurIPS)
- `sanchezlengeling2020evaluating` — Sanchez-Lengeling et al. 2020 evaluating GNN explanations
- `kokhlikyan2020captum` — Kokhlikyan et al. 2020 Captum
- `jain2019attention` — Jain & Wallace 2019 *Attention is not Explanation* — counter-evidence (mandatory citation for honest discussion)
- `wiegreffe2019attention` — Wiegreffe & Pinter 2019 *Attention is not not Explanation* — rebuttal

**Acceptance criterion.** GNNExplainer + at least 1 attention-as-explanation critic + 1 rebuttal cited (honest discussion of limits).

---

## Cluster C9 — Cross-validation / leakage / nested-CV best practices

**Cluster ID:** `C9-cv-methodology`
**Maps to paper §:** §2.3.5 Evaluation framework; §2.3.4 Optuna nested-CV; Methods reproducibility statement.
**Argument to defend:**
1. Subject-level k-fold and LOSO are the only correct partitioning for biomarker studies (no within-subject leakage).
2. Nested CV is required when hyperparameter tuning is involved (Cawley & Talbot 2010).
3. Per-fold standardisation must use train-only statistics (the leak-free contract).

**Target paper count:** 3–4.

**Search queries (2 to run):**
1. `nested cross-validation hyperparameter tuning Cawley overfitting`
2. `data leakage machine learning healthcare cross-validation pitfalls`

**Expected canonical references:**
- `cawley2010overfitting` — Cawley & Talbot 2010 *On over-fitting in model selection and subsequent selection bias in performance evaluation* (JMLR)
- `varoquaux2017assessing` — Varoquaux 2017 *Assessing and tuning brain decoders* (NeuroImage)
- `kapoor2023leakage` — Kapoor & Narayanan 2023 *Leakage and the reproducibility crisis in ML-based science*
- `arlot2010survey` — Arlot & Celisse 2010 CV survey

**Acceptance criterion.** Cawley-Talbot 2010 cited; ≥ 1 paper specifically on neural / brain-decoder leakage.

---

## Cluster C10 — Optuna / TPE hyperparameter optimisation

**Cluster ID:** `C10-hpo-optuna`
**Maps to paper §:** §2.3.4 Optuna for hyperparameter optimisation.
**Argument to defend:**
1. TPE (Bergstra et al.) is sample-efficient — the right choice when evaluations are expensive.
2. Optuna is the standard implementation.

**Target paper count:** 2.

**Search queries (1 to run):**
1. `Optuna hyperparameter optimization TPE Akiba 2019 Tree-structured Parzen`

**Expected canonical references:**
- `akiba2019optuna` — Akiba et al. 2019 *Optuna: A next-generation hyperparameter optimization framework* (KDD)
- `bergstra2011algorithms` — Bergstra et al. 2011 *Algorithms for hyper-parameter optimization* (NeurIPS — TPE)

**Acceptance criterion.** Both above cited.

---

## Cluster C11 — Statistical methods (Wilcoxon, McNemar, FDR, bootstrap)

**Cluster ID:** `C11-stats-methods`
**Maps to paper §:** §2.3.5 metric definitions; §3.1 (FDR via BH); §3.2 (paired tests); §3.2.4 (PPV/NPV / clinical utility).
**Argument to defend:**
1. Wilcoxon signed-rank (Wilcoxon 1945) and McNemar's test (McNemar 1947) are the standard paired non-parametric tests.
2. Benjamini-Hochberg FDR (1995) is the standard multiple-comparison correction.
3. Bootstrap CI (Efron 1979) for finite-sample confidence intervals.

**Target paper count:** 3.

**Search queries (1 to run):**
1. `Benjamini Hochberg false discovery rate multiple testing 1995`

**Expected canonical references:**
- `wilcoxon1945individual` — Wilcoxon 1945 (rank-sum, biometrics)
- `mcnemar1947note` — McNemar 1947
- `benjamini1995controlling` — Benjamini & Hochberg 1995 FDR
- `efron1986bootstrap` — Efron & Tibshirani 1986

**Acceptance criterion.** All four cited inline at first use.

---

## Cluster C12 — Counter-evidence / methodological caveats (small but important)

**Cluster ID:** `C12-counter-evidence`
**Maps to paper §:** §4.2 Threats to validity; §5 Future work.
**Argument to defend:**
1. fNIRS classifiers can be confounded by non-neural signals (motion, scalp blood flow).
2. Age effects on prefrontal hemodynamics are well-documented and confound clinical classifiers.
3. Single-site small-cohort classifiers rarely generalise.

**Target paper count:** 3–4.

**Search queries (2 to run):**
1. `prefrontal cortex aging hemodynamics oxygenation NIRS healthy elderly`
2. `single-site machine learning generalization clinical neural classifier external validation`

**Expected canonical references:**
- `mehagnoul2008age` — Mehagnoul et al. 2008 age-related decline in PFC NIRS (or similar)
- `tachtsidis2016false` — Tachtsidis & Scholkmann 2016 fNIRS false positives (cluster C6 dual-cite)
- `varoquaux2018machine` — Varoquaux 2018 *Machine learning for medical imaging: methodological failures*
- `wynants2020prediction` — Wynants et al. 2020 *Prediction models for COVID-19* (the canonical "external validation matters" paper)

**Acceptance criterion.** ≥ 1 ageing-PFC-NIRS paper + ≥ 1 ML-validity / external-validation paper.

---

## Execution order and budget

| Order | Cluster | Queries | Expected unique papers | Cumulative bib entries |
|---|---|---:|---:|---:|
| 1 | C1 — Clinical GAD | 3 | 4 | 4 |
| 2 | C2 — fNIRS modality | 2 | 4 | 8 |
| 3 | C3 — fNIRS anxiety classification | 3 | 8–10 | 16–18 |
| 4 | C4 — GNNs on neural data | 3 | 7 | 23–25 |
| 5 | C5 — PFC-GAD anatomy | 3 | 5 | 28–30 |
| 6 | C6 — fNIRS preprocessing | 2 | 4 | 32–34 |
| 7 | C7 — GLM/HRF | 2 | 3 | 35–37 |
| 8 | C8 — XAI / GNN | 2 | 4 | 39–41 |
| 9 | C9 — CV / leakage | 2 | 3 | 42–44 |
| 10 | C10 — HPO Optuna | 1 | 2 | 44–46 |
| 11 | C11 — Stats methods | 1 | 4 | 48–50 |
| 12 | C12 — Counter-evidence | 2 | 3 | 51–53 |
| **Total** | **12 clusters** | **26** | **~50 papers** | — |

**Time budget**: ~26 `research-lookup` queries × ~30 s each = ~13 min compute. Synthesis writing: 2–3 hours (including the §1.4 narrative paragraphs + every inline citation in §2.3 and §4).

---

## Per-cluster output structure

For each cluster, `research-lookup` produces:
```
research/paper-materials/sources/<cluster_id>.md
```
This file contains the raw `research-lookup` output (paper titles, authors, abstracts, DOIs).

Then I extract the verified citation entries into:
```
research/paper-materials/references/refs.bib   ← cumulative across clusters
```

And I write a per-paper one-liner narrative for each cited paper into:
```
research/paper-materials/literature_review.md  ← the §1.4 narrative + a Methods-citation table
```

---

## Stop conditions

- **STOP** after C3 if ≥ 2 papers report HC-vs-GAD fNIRS classification with detailed enough numbers (n, paradigm, accuracy, F1) to fill the §1.4 comparison table. C3 is the highest-leverage cluster — sometimes it's enough on its own to defend §1.4.
- **STOP** the entire literature review at ~50 unique papers. Beyond that, the marginal value of each new citation is small and the bibliography starts to bloat.
- **REVISIT** C3 if the §1.4 narrative needs more breadth; re-run with refined queries (e.g., `fNIRS social anxiety`, `fNIRS PTSD`).

---

## What this plan does NOT cover (out-of-scope by design)

These clusters were considered and **excluded** because the paper's argument
does not actually need them:

- **fMRI methods deep-dive** — we cite fMRI for context only; no fMRI re-analysis is performed.
- **Deep-learning theory beyond GAT/GATv2** — Transformers, CNNs, etc. are not the architecture used; mentioning them = noise.
- **Anxiety treatment / pharmacology** — we classify, not treat; treatment refs distract from the methods focus.
- **fNIRS hardware / wavelengths / Beer-Lambert details** — covered in `data/DATA_QUALITY_REPORT.md` as project-internal pre-processing; the paper Methods section references the report rather than the underlying physics primer.
- **Atlas / fOLD / AtlasViewer** — only relevant once the atlas notebook P0.10 produces output; deferred to a future cluster `C13-atlas` if needed after P0.10 lands.

---

## Resumption protocol

If the user opens the next session and says "continue the literature review":
1. Read this plan.
2. Look at `research/paper-materials/sources/` to see which clusters already have a `<cluster_id>.md`. Skip those.
3. Execute remaining clusters in order. Use `research-lookup` with the queries documented per cluster.
4. After each cluster completes, append entries to `references/refs.bib` and update `literature_review.md`.
5. When all 12 clusters done, the `literature_review.md` is ready to be lifted into the manuscript.

This file (`LITERATURE_REVIEW_PLAN.md`) is **not** consumed by the SPEC plan
directly — it is a worklog. The SPEC plan reads `literature_review.md` and
`refs.bib` instead.
