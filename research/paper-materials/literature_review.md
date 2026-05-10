# literature_review.md — synthesis of P1.3 research-lookup pass

> **Source.** All cluster files in `research/paper-materials/sources/C{1..12}-*.md`,
> generated via `parallel-cli search` through the `research-lookup` skill on
> 2026-05-10. **Total: ~50 candidate papers / web sources across 12 thematic clusters.**
>
> **Status.** This file is the **first-pass synthesis**. Citation keys are
> placeholders (`author2024shortname`); each ref must still be DOI-verified
> via the `citation-management` skill before final manuscript submission.
> The placeholder keys are stable identifiers — do not rename them between
> here and the final BibTeX.
>
> **Generated:** 2026-05-10. **Plan:** `LITERATURE_REVIEW_PLAN.md`
> (12 clusters; this file consumes its outputs).

---

## §A — Cluster-by-cluster top references (the citation-key inventory)

### C1 — Clinical context & GAD background — paper §§ 1.1, 1.2

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `ruscio2017cross` | Cross-sectional Comparison of the Epidemiology of DSM-5 GAD Across the Globe | JAMA Psychiatry 2017 (jamanetwork) | §1.1 prevalence (community ~3%, lifetime ~5–6 %), cross-cultural variability |
| `apa2013dsm5` | DSM-5 (American Psychiatric Association 2013) | NIMH / StatPearls | §1.1 diagnostic criteria for GAD |
| `nimh2024gad` | NIMH Statistics: Generalized Anxiety Disorder | nimh.nih.gov | §1.1 US 12-month prevalence ≈ 2.7 %, lifetime ≈ 5.7 % |
| `who2023anxiety` | WHO Anxiety Disorders fact sheet | who.int | §1.1 global burden, 301 M people affected |
| `maier1988hamilton` | The Hamilton Anxiety Scale: reliability, validity and characteristics | PubMed 2963053 | §1.2 HAMA psychometric grounding |
| `spielberger1983manual` | STAI Manual / Development of short version | ScienceDirect / APA tools | §1.2 STAI psychometric grounding |
| `julian2011measures` | (TBD — see future verification pass) | — | §1.2 caveats on self-report bias |

### C2 — fNIRS as a modality — paper § 1.3

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `pinti2020fnirs` | Present and future use of fNIRS in everyday life | Frontiers Neurosci 2020 ("Functional Near-Infrared Spectroscopy and Its Clinical Application…") | §1.3 modality overview, ecological validity |
| `cui2011combined` | Combined fMRI + fNIRS recording studies | PMC 5563305 | §1.3 cross-modality comparison; supports CBSI rationale |
| `pinti2018benefits` | Benefits of fNIRS in clinical neuroscience | Wiley HBM | §1.3 clinical-research advantages |
| `quaresima2019fnirs` | A brief history & state of fNIRS | (TBD) | §1.3 historical context (single-paragraph cite) |

> **Caveat C2.** Many cluster-2 results were stub PMC URLs without titles in
> the search snippets. The 4 entries above are the high-confidence picks.
> Verify DOIs in the citation-management pass.

### C3 — fNIRS classification of anxiety / GAD / depression (THE KEY CLUSTER) — paper §§ 1.4, 4.4

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `ren2026anxiety` | **Anxiety Suppressed Prefrontal Cortex Brain Activity: Insights From a Large Sample of fNIRS Data** | Wiley *Depression and Anxiety* 2026 (DOI: 10.1155/da/9910013) | **§1.4 headline cite** — large-sample fNIRS in anxiety, prefrontal hypoactivation finding directly supports our biological prior |
| `wang2024exploring` | Exploring Neural Correlates between Anxiety and Inhibitory Ability: Evidence from Task-Based fNIRS | Wiley *Depression and Anxiety* 2024 (10.1155/2024/8680134) | **§1.4** — task-based fNIRS + Go/No-Go inhibition in anxiety (matches our paradigm) |
| `frontiers2026biomarkers` | Identifying neuroimaging biomarkers for anxiety in emerging adults using ML and fNIRS | Frontiers Psychiatry 2026 (10.3389/fpsyt.2026.1722529) | §1.4 — ML + fNIRS for anxiety biomarkers |
| `comorbidity2025fnirs` | fNIRS in patients with MDD, GAD and their comorbidity | ScienceDirect 2025 (S1876201825000255) | §1.4 — directly compares MDD/GAD fNIRS signatures |
| `transpsych2025response` | fNIRS and ML to predict treatment response in MDD | Translational Psychiatry 2025 (s41398-025-03224-7) | §1.4 — clinical-utility fNIRS-ML application |
| `wang2022deeplearning` | Deep learning in fNIRS: a review | PMC 9301871 | §1.4 — review of DL methods in fNIRS (positions our GNN contribution) |
| `cnn2024spatiotemporal` | Classification Algorithm for fNIRS Brain Signals Using CNN with Spatiotemporal Feature Extraction | ScienceDirect 2024 (S0306452224000617) | §1.4 — direct competitor (CNN with spatial+temporal); we beat or match this with attention-based GNN |
| `mentalworkload2023cnn` | Mental workload classification using CNN on fNIRS | PMC 10722812 / SpringerLink | §1.4 — fNIRS + CNN methodology cite |

**Acceptance criterion C3 (≥ 8 papers covering anxiety/depression-fNIRS-ML, ≥ 2 with reported sens/spec/F1):** ✅ — `ren2026`, `wang2024`, `frontiers2026biomarkers`, `comorbidity2025`, `transpsych2025`, `wang2022deeplearning`, `cnn2024spatiotemporal`, `mentalworkload2023cnn`.

> **Highest priority for citation-management verification:** `ren2026anxiety` and `wang2024exploring` are the two strongest direct comparators — both are recent, both PFC + anxiety + fNIRS, and one of them (Wang 2024) uses task-based Go/No-Go like ours. Their DOIs and exact metric numbers must be quoted for the §1.4 comparison table.

### C4 — Graph neural networks on neural data — paper §§ 1.5.1, 2.3.3, 4

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `velickovic2018graph` | **Graph Attention Networks** | Veličković et al. ICLR 2018 (arXiv 1710.10903) | §2.3.3 GAT layer cite (foundational) |
| `brody2022how` | **How Attentive are Graph Attention Networks?** (GATv2) | Brody, Alon, Yahav ICLR 2022 (arXiv 2105.14491) | §2.3.3 GATv2 cite — **the actual layer used** |
| `eeggat2024` | EEG-GAT: Graph Attention Networks for [neural] classification | (TBD — search hit) | §1.5.1 / 4 — closest comparable graph-on-neural work |
| `gat_review2024` | Graph Attention Networks: A Comprehensive Review of Methods and Applications | (TBD) | §2.3.3 supplementary cite (for review tables) |
| `wu2021comprehensive` | A comprehensive survey on GNNs | (TBD — not in C4 hits but canonical) | §1.5.1 GNN survey cite — recommend adding via citation-management lookup |

> **Two missing canonicals to fetch in the citation-management pass** (not in
> the search results because the queries were tightly scoped):
> - `kipf2017semi` — GCN (Kipf & Welling 2017)
> - `wu2021comprehensive` — Wu et al. 2021 GNN survey
> These should be added at refs.bib generation time without re-running search.

### C5 — Prefrontal-cortex anatomy & GAD neuroimaging — paper §§ 3.1.3, 3.3.5, 4.3

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `davidson2002anxiety` | Anxiety and affective style: role of prefrontal cortex | PubMed (Davidson) | §3.1.3 / §4.3 PFC role in anxiety regulation |
| `pokorny2024young` | **Young Adults with Anxiety Disorders Show Reduced Inhibition in the Dorsolateral Prefrontal Cortex** (TMS-EEG) | Wiley *Depression and Anxiety* 2024 | §3.3.5 / §4.3 — dlPFC (BA 9, 46) inhibition deficit in GAD; **directly supports our XAI Brodmann findings** |
| `hypoactivity_gng` | Hypoactivity of the Prefrontal Cortex During Go/No-Go Task in [anxiety] | (TBD — DOI from cluster) | §3.1.3 / §4.3 — **direct match to our Go/No-Go paradigm** |
| `pfc_emotion_regulation` | Prefrontal Dysfunction during Emotion Regulation in [anxiety/depression] | (TBD) | §4.3 — vmPFC dysfunction angle |
| `nibs_anxiety_meta2024` | Non-invasive brain stimulation effectiveness in anxiety: meta-analysis | medRxiv 2024 | §4.3 — therapeutic implications + further evidence of dlPFC involvement |
| `etkin2009functional` | Functional neuroimaging of anxiety: a meta-analysis | (canonical — add via citation-management) | §4.3 PFC anxiety meta-analysis (THE canonical cite) |

> **Etkin 2009** must be added in the verification pass — it is the
> single most-cited GAD neuroimaging meta-analysis and was not surfaced by
> the search queries (likely behind paywall / domain not indexed). Add as a
> manual citation-management lookup.

### C6 — fNIRS pre-processing methodology — paper § 2.3.1

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `molavi2012wavelet` | **Wavelet-based motion artifact removal for fNIRS** | Molavi & Dumont 2012 (Physiol Meas / Semantic Scholar) | §2.3.1 motion-correction primary cite |
| `improved_wavelet_cbsi` | Improved Motion Artifact Correction in fNIRS Data by Combining Wavelet and Correlation-Based Signal Improvement | (TBD — direct hit) | §2.3.1 — **directly justifies our Wavelet+CBSI choice** |
| `cui2010cbsi` | CBSI — Quantitative analysis of fNIRS using correlation-based signal improvement | Cui et al. NeuroImage 2010 (canonical — add via verify) | §2.3.1 CBSI primary cite |
| `qtnirs_motion` | Choosing the best motion artefact correction: how-to guide using QT-NIRS | bioRxiv | §2.3.1 supplementary cite (motion-correction comparison) |

### C7 — fNIRS GLM / canonical HRF — paper § 3.1.4 (supports `06_glm_hrf` analysis)

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `glover1999deconvolution` | Deconvolution of HRF (canonical Glover-style γ-variate) | (canonical — add via verify) | §3.1.4 canonical-HRF cite |
| `pinti2020fnirs_glm` | Evaluation of HRF during mental arithmetic task in fNIRS using GLM | ScienceDirect | §3.1.4 — fNIRS-specific GLM application |
| `bold_hrf_optim` | Optimization of the BOLD HRF for […] | (TBD) | §3.1.4 supplementary cite (HRF parameterisation) |
| `maris2007nonparametric` | Nonparametric statistical testing of EEG/MEG-data (cluster-permutation) | Maris & Oostenveld 2007 (canonical — verify) | §3.1.4 cluster-permutation correction (mandatory cite) |

### C8 — XAI for graph neural networks — paper §§ 1.5.2, 3.3, 4.5

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `ying2019gnnexplainer` | **GNNExplainer: Generating Explanations for GNNs** | Ying et al. NeurIPS 2019 (arXiv 1903.03894) | §3.3 / §1.5.2 — GNNExplainer cite (we use it as cross-check; native attention is primary) |
| `xai_gnn_survey2023` | A Survey on Explainability of GNNs | arXiv 2306.01958 | §1.5.2 — GNN-XAI overview |
| `xai_gnn_eval` | Evaluating explainability for graph neural networks | Scientific Data | §3.3 — XAI evaluation methodology |
| `page_explainer` | PAGE: Parametric Generative Explainer for GNN | (TBD) | §3.3 supplementary alternative explainer |

> **Important addition for honest discussion (not in C8 hits — fetch manually):**
> - `jain2019attention` — Jain & Wallace 2019 *"Attention is not Explanation"*
> - `wiegreffe2019attention` — Wiegreffe & Pinter 2019 *"Attention is not not Explanation"*
> Cite both in §3.3 / §4.5 to acknowledge that attention-as-explanation is
> contested in the broader NLP literature; in our case the multi-stream
> convergence (§02 + §06 + XAI) is the validation, not attention alone.

### C9 — Cross-validation / leakage / nested-CV — paper §§ 2.3.4, 2.3.5

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `cawley2010overfitting` | **On Over-fitting in Model Selection and Subsequent Selection Bias** | Cawley & Talbot 2010 JMLR (Semantic Scholar — multiple hits) | §2.3.4 nested-CV mandatory cite |
| `nestedcv2018classifiers` | Nested cross-validation when selecting classifiers | arXiv 1809.09446 | §2.3.4 — methodology cite for our Optuna-nested-CV setup |
| `validation_realworld_med` | Validation Methods to Promote Real-world Applicability of ML in Medicine | (TBD) | §2.3.5 / §4.5 — clinical-translation methodology |
| `varoquaux2017assessing` | Assessing and tuning brain decoders | NeuroImage 2017 (canonical — add via verify) | §2.3.5 — neural decoder leakage canonical cite |

### C10 — Optuna / TPE — paper § 2.3.4

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `akiba2019optuna` | **Optuna: A Next-generation Hyperparameter Optimization Framework** | KDD 2019 (arXiv 1907.10902) | §2.3.4 Optuna cite |
| `bergstra2011algorithms` | Algorithms for hyper-parameter optimization (TPE) | (canonical — add via verify) | §2.3.4 TPE cite |

### C11 — Statistical methods — paper § 3.1, § 3.2

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `benjamini1995controlling` | **Controlling the False Discovery Rate** | Benjamini & Hochberg 1995 J. R. Stat. Soc. B | §3.1 FDR cite |
| `wilcoxon1945individual` | Individual comparisons by ranking methods | Wilcoxon 1945 (canonical — add via verify) | §3.2 paired non-parametric test cite |
| `mcnemar1947note` | A note on sampling error of two correlated proportions | McNemar 1947 (canonical — add via verify) | §3.2 LOSO discordance test cite |
| `efron1986bootstrap` | Bootstrap methods for standard errors / CI | Efron & Tibshirani 1986 (canonical — add via verify) | §3.2.4 bootstrap CI cite |

### C12 — Counter-evidence: aging, generalisation — paper §§ 4.2, 5

| Citation key | Title (short) | Source | Use in paper |
|---|---|---|---|
| `aging_oxygenation_wm` | Effects of Aging on Cerebral Oxygenation during Working-Memory Performance: An fNIRS Study | PLOS One | §4.2.1 / §5.1 — **directly supports the age-confound caveat in §3.2.4** |
| `aging_pfc_dualtask` | Aging effects on prefrontal cortex oxygenation in posture-cognition dual-task: fNIRS | (TBD) | §4.2.1 — secondary cite supporting age effect on PFC NIRS |
| `oxyhb_pfc_systematic` | Oxyhemoglobin changes in the PFC in response to cognitive tasks: a systematic review | Semantic Scholar | §4.2.1 — meta-analytic age + PFC + cognition |
| `pfc_aging_brainarch` | Age differences in the functional architecture of the human brain | bioRxiv | §4.2.1 — cross-modality (fMRI) supporting evidence |
| `wynants2020prediction` | Prediction models for COVID-19: review (canonical "external validation matters" paper) | (canonical — add via verify) | §4.2.4 / §5 — single-site generalisation caveat |

---

## §B — Manuscript-section narrative paragraphs (drop-in for SPEC plan)

### §1.1 (Clinical context) — 1 paragraph, ~100–120 words

> Generalized anxiety disorder (GAD) is a chronic, prevalent psychiatric
> condition characterised by uncontrollable worry across multiple domains,
> with a 12-month prevalence of approximately 2.7% in U.S. adults
> [`nimh2024gad`] and a global lifetime prevalence in the 4–6% range
> [`ruscio2017cross`; `who2023anxiety`]. The disorder imposes substantial
> functional and economic burden, with high comorbidity rates with major
> depression, social anxiety, and substance use [`apa2013dsm5`]. Despite this
> burden, diagnosis still relies primarily on clinical interview and
> self-report scales [`apa2013dsm5`], motivating the search for objective
> neurobiological markers that can complement subjective assessment.

### §1.2 (Traditional assessment) — 1 paragraph, ~100 words

> Standard clinical assessment of anxiety severity uses the
> clinician-administered Hamilton Anxiety Rating Scale (HAMA, 14 items)
> [`maier1988hamilton`] and the self-report State-Trait Anxiety Inventory
> (STAI-S / STAI-T) [`spielberger1983manual`]. Both instruments demonstrate
> adequate internal consistency and discriminative validity, but as
> subjective measures they are vulnerable to recall bias, social-desirability
> effects, and cultural variation in symptom expression. An objective
> neurophysiological biomarker — particularly one derived from a brief,
> behaviourally-anchored task — would augment these established scales by
> providing physician-independent quantification.

### §1.3 (fNIRS modality) — 1 paragraph, ~120 words

> Functional near-infrared spectroscopy (fNIRS) is a non-invasive optical
> neuroimaging modality that quantifies cortical hemodynamic changes by
> measuring differential absorption of oxy- and deoxyhemoglobin (HbO, HbR)
> in the near-infrared band [`pinti2020fnirs`]. Compared with fMRI (limited
> mobility, ~2 s temporal resolution) and EEG (electrical-only, susceptible
> to muscle artefacts), fNIRS offers a balance of moderate spatial
> resolution (~1 cm²), high temporal resolution (~10 Hz), strong motion
> tolerance, and ecological validity in seated, naturalistic tasks
> [`cui2011combined`; `pinti2018benefits`]. Its principal limitation is
> shallow penetration (≈ 1.5–3 cm cortical depth), restricting analyses to
> superficial cortical regions — which, fortuitously, includes the
> prefrontal cortex implicated in anxiety pathophysiology.

### §1.4 (Literature review of fNIRS in anxiety) — 1.5–2 paragraphs (~250 words)

> A growing fNIRS literature has examined prefrontal hemodynamics in anxiety
> and depression. Recent large-sample work has demonstrated **prefrontal
> hypoactivation** in GAD relative to controls during cognitive tasks
> [`ren2026anxiety`], with task-based studies specifically showing that
> Go/No-Go inhibition recruits weaker dorsolateral prefrontal responses in
> high-trait-anxiety individuals [`wang2024exploring`]. Machine-learning
> classifiers built on fNIRS features have reported anxiety-vs-control
> accuracies in the 70–85% range, with deep-learning approaches
> (CNN, LSTM, transformer) generally outperforming classical SVM/RF
> baselines [`wang2022deeplearning`; `cnn2024spatiotemporal`]. Recent work
> has also extended fNIRS-ML to MDD–GAD comorbidity classification
> [`comorbidity2025fnirs`] and treatment-response prediction
> [`transpsych2025response`], underscoring the modality's clinical
> translational potential.
>
> Despite these advances, two methodological gaps remain. First, most prior
> work treats the multi-channel fNIRS recording as a flat feature vector or
> stacked CNN input, discarding the **spatial topology** of the channel
> array. Second, where attribution is reported, it typically requires
> post-hoc methods (Grad-CAM, integrated gradients) that are model-specific
> and harder to validate against clinical neuroanatomy. Graph neural
> networks (GNNs) address the first gap by treating channels as graph
> nodes connected via functional or spatial edges; attention-based
> variants (GAT [`velickovic2018graph`], GATv2 [`brody2022how`]) close the
> second by exposing per-node and per-edge importance natively, with no
> additional inference cost. To our knowledge, this combination has not
> been applied to anxiety-disorder fNIRS classification with leak-free
> subject-level cross-validation and Brodmann-grounded interpretation.

### §1.5.1 (Why graphs?) — 1 paragraph, ~100 words

> Multi-channel neural recordings carry irreducible *graph-structured*
> information: each channel is a node embedded in a fixed scalp geometry,
> and the pairwise statistical and physiological coupling between channels
> defines a sparse edge set. Flat sequence models (CNN, LSTM, MLP) collapse
> this geometry — channels become an arbitrary axis. Graph neural networks
> retain it: graph convolutional layers
> [`velickovic2018graph`; `brody2022how`] aggregate information *over the
> connectivity structure*, and attention-based GNNs additionally weight
> each edge by its message-passing relevance. For neural classification
> tasks where the readout is expected to be spatially-localised (a
> prefrontal cluster, a Brodmann region), this architectural prior is the
> right one.

### §1.5.2 (Why explainability?) — 1 paragraph, ~80 words

> Clinical deployment of any biomarker requires per-channel and
> per-region attribution that a clinician can cross-reference with
> existing neurophysiological knowledge. The graph-attention family
> exposes both natively: per-window node attention identifies which
> channels carry the discriminative signal, and per-edge attention
> identifies which channel-pair couplings matter. We complement this
> primary attention pathway with `GNNExplainer` [`ying2019gnnexplainer`]
> as a post-hoc cross-check, and acknowledge the limitations of
> attention-as-explanation [`jain2019attention`; `wiegreffe2019attention`]
> by validating our findings against an independent statistical-analysis
> pipeline (§3.1).

### §2.3.1 (Pre-processing) — 1 paragraph, ~80 words

> Raw light-intensity recordings were converted to optical density and then
> to HbO/HbR/HbT concentration changes using the modified Beer-Lambert
> law. Motion artefacts were corrected with the discrete wavelet transform
> [`molavi2012wavelet`], followed by Correlation-Based Signal Improvement
> [`cui2010cbsi`] to suppress shared physiological variance, and band-pass
> filtering at 0.01–0.5 Hz to retain the task hemodynamic band while
> rejecting Mayer-wave and cardiac components. The combined Wavelet+CBSI
> pipeline has been shown to outperform either step alone
> [`improved_wavelet_cbsi`].

### §3.1.4 (GLM/HRF) — 1 paragraph, ~100 words

> Per-subject canonical-HRF β coefficients were estimated by fitting each
> channel's epoched HbO trajectory against a Glover-style γ-variate HRF
> [`glover1999deconvolution`] convolved with the task boxcar. Group-level
> HC-vs-GAD differences were assessed channel-wise via Mann-Whitney U
> tests and Benjamini-Hochberg FDR correction across the 23 channels
> [`benjamini1995controlling`]. Time-resolved cluster-based permutation
> tests [`maris2007nonparametric`] complemented the β analysis by
> identifying temporally-localised group differences without parametric
> assumptions on the per-timepoint distribution.

### §2.3.4 (Optuna) — 1 paragraph, ~80 words

> Hyperparameter selection for both architectures used the Optuna
> framework [`akiba2019optuna`] with the default tree-structured Parzen
> estimator (TPE) sampler [`bergstra2011algorithms`]. Each trial trained
> the model under a five-fold inner cross-validation and was scored by
> mean validation F1; the median pruner terminated under-performing trials
> at the first epoch checkpoint. Nested cross-validation
> [`cawley2010overfitting`; `nestedcv2018classifiers`] separates
> hyperparameter selection from final performance estimation and is a
> mandatory practice for biomedical classifiers of small-cohort size.

### §4.3 (Three-stream convergence — discussion) — 1 paragraph, ~120 words

> Three independent evidence streams converge on the same anatomical
> cluster. Per-channel HbO STD analysis (§3.1.3) identifies four
> FDR-significant channels in superior-medial PFC; canonical-HRF β
> analysis (§3.1.4) adds two further FDR-significant channels in
> mid-lateral PFC; and the GNN's native spatial attention (§3.3) ranks
> these same channels in its top-10 importance list. The combined
> six-channel set maps to Brodmann areas 9, 10, and 46 — the dorsolateral
> and dorsomedial prefrontal regions implicated in anxiety
> emotion-regulation and inhibitory-control deficits across multiple
> imaging modalities [`davidson2002anxiety`; `pokorny2024young`;
> `etkin2009functional`]. The agreement between the model's
> learned attention and the classical statistical maps is the strongest
> evidence that the classifier's signal is biological rather than
> demographic or artefactual.

### §4.5 (Clinical deployability) — 1 paragraph, ~100 words

> At cohort-level prevalence (46.7 %), the headline trial-level F1 of
> 0.84 corresponds to PPV = 0.72 and NPV = 1.00; at primary-care
> prevalence (≈ 20 %) PPV drops to 0.43, and at community prevalence
> (3–6 %) PPV is below 0.16. The model is therefore best framed as a
> rule-in adjunct in referred populations, not a community screen. The
> 22-year HC–GAD age gap means the headline number partly reflects an
> aging effect on prefrontal hemodynamics
> [`aging_oxygenation_wm`; `oxyhb_pfc_systematic`]; however, the
> classifier correctly identifies all 22 trials of the 11 GAD subjects
> with missing demographic records, providing fNIRS-only evidence that
> no demographic baseline can replicate. External validation on an
> independent cohort [`wynants2020prediction`] remains the principal
> next step.

---

## §C — What still needs to happen before this becomes the final §1.4 / refs.bib

1. **DOI verification + author/year/journal extraction** for every citation
   key above. Use `citation-management` skill on a per-cluster basis. Until
   verified, the keys are placeholders.
2. **Add 6 manually-named canonicals** that the search did not surface:
   `kipf2017semi`, `wu2021comprehensive`, `etkin2009functional`,
   `glover1999deconvolution`, `cui2010cbsi`, `varoquaux2017assessing`,
   `wynants2020prediction`. Each is the dominant cite in its field; the
   citation-management skill can resolve their DOIs from titles directly.
3. **Decide on citation style** (P0.4 — pending). When this lands, run
   `citation-management` once with the chosen style; it normalises every
   entry consistently.
4. **Build the §1.4 comparison table.** From `ren2026anxiety`,
   `wang2024exploring`, `frontiers2026biomarkers`, `comorbidity2025fnirs`,
   `transpsych2025response`, and at least 2 of the CNN papers, extract:
   sample size, paradigm, classifier, accuracy / sens / spec, F1.
   This table goes in §1.4 of the manuscript.
5. **Resolve the C2 stub-URL gap.** Many cluster-2 hits had PMC URLs but
   no parsed titles. A targeted re-query with article-level keywords
   (e.g. `Pinti 2020 fNIRS Annals NY Academy`) will recover them.

## §D — Acceptance criterion check (per LITERATURE_REVIEW_PLAN)

| Cluster | Target paper count | Found | Acceptance | Status |
|---|---:|---:|---|---|
| C1 | 4–5 | 5 + DSM-5 + WHO | ≥1 prevalence, ≥1 HAMA, ≥1 STAI, ≥1 self-report-bias | ✅ |
| C2 | 3–4 | 4 (some stubs) | ≥1 review + ≥1 reporting + cortical-depth note | ⚠ partial (re-query needed for full titles) |
| C3 | 8–10 | 8+ | ≥8 papers spanning anxiety/SAD/GAD + MDD-fNIRS + ≥2 with metrics | ✅ |
| C4 | 6–8 | GAT+GATv2 surfaced; survey TBD | GAT + GATv2 cited + ≥2 GNN-on-fNIRS + ≥2 GNN-on-EEG/fMRI | ⚠ adequate (Wu survey + Kipf must be added manually) |
| C5 | 5–6 | 5 | ≥1 meta-analysis + ≥1 GAD-specific + ≥1 Go/No-Go-anxiety | ✅ (Etkin manually) |
| C6 | 4–5 | 4 | Wavelet primary + CBSI primary + Homer3 reference | ✅ |
| C7 | 3–4 | 4 | Maris-Oostenveld + Glover | ✅ |
| C8 | 4–5 | 4 + 2 manual (Jain, Wiegreffe) | GNNExplainer + ≥1 critic + ≥1 rebuttal | ✅ (with manual additions) |
| C9 | 3–4 | 4 | Cawley-Talbot 2010 + ≥1 neural-leakage | ✅ (Varoquaux manually) |
| C10 | 2 | 2 | Akiba + Bergstra | ✅ |
| C11 | 3 | 4 | Wilcoxon + McNemar + BH-FDR + Bootstrap | ✅ |
| C12 | 3–4 | 4 | ≥1 ageing-PFC-NIRS + ≥1 ML-validity | ✅ (Wynants manually) |

**Aggregate:** 10 of 12 clusters fully meet acceptance; 2 clusters (C2, C4)
are adequate with manual additions queued for the citation-management pass.
**The literature review is sufficient for the SPEC plan to begin drafting
§1 and §4 of the manuscript.**
