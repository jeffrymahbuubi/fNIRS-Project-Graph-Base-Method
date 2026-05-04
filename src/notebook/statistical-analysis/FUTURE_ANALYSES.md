# Future Statistical Analyses — Prioritized Roadmap

> Synthesis of 5 independent reviewer opinions (clinical neurophysiology,
> fNIRS methodology, biostatistics, ML interpretability, reproducibility).
> Each section below names the recommendation, its rationale, the agreement
> across reviewers, and a difficulty/impact verdict.
>
> Convention: **Tier 1 = rejection-grade if missing**, **Tier 2 = expected by
> any thorough reviewer**, **Tier 3 = polish that elevates a "good" paper to a
> "strong" one**.

## TL;DR — top 5 to do first

| # | Analysis | Reviewer agreement | Difficulty | Why first |
|---|----------|--------------------|------------|-----------|
| 1 | **Age-adjusted ANCOVA + age-matched subsample re-analysis** | Clinical, Biostat (both top pick) | Low–Med | The 22-yr HC–GAD age gap confounds *every* group result; without this the paper is rejectable on confound grounds alone |
| 2 | **GLM/HRF first-level + Go-vs-No-Go contrast** (replace STD-of-amplitude metric) | fNIRS (top pick) | Medium | STD across concatenated trials is event-agnostic; an fNIRS journal will reject any task-fNIRS paper without canonical-HRF β maps |
| 3 | **Saliency–statistics concordance figure** (GNN attention vs MWU \|d\| vs AUC) on the 5×7 grid | ML-interp (top pick), Clinical (MNI mapping) | Medium | Currently the paper has two parallel narratives (stats and models); one figure unifies them |
| 4 | **LOSO stability of channel rankings + sensitivity-cohort table** (drop AH029, AH024, LA063, demographics-missing 11) | Reproducibility (top pick), Biostat, Clinical | Low | Pre-empts every "you cherry-picked the cohort" critique |
| 5 | **Per-channel ROC-AUC univariate floor + linear logistic baseline** | ML-interp, Clinical | Low | Converts MWU/d into the same currency as the GNN; establishes the floor the GNN must beat |

The full roadmap below covers ~20 distinct recommendations organized into tiers.

---

## TIER 1 — Rejection-grade if missing

### 1.1 Age (and sex) confound adjustment
- **Why:** HC mean age 73 vs GAD mean age 51 (Welch *t* = 6.08, *p* ≈ 6.5 × 10⁻⁶). Prefrontal HbO/HbR amplitude declines with age; without partialling this out, the §02 channel differences and §04 severity correlations are confounded with normal aging.
- **Method:**
  - Per-channel `statsmodels.formula.api.ols("HbX ~ Group + Age + Sex", data=...)` (or `pingouin.ancova`); report group β, 95 % CI, partial η².
  - Sensitivity refit on the n=51 with complete demographics.
  - **Age-matched subsample**: propensity-matching (`psmpy`) or coarsened exact matching (`cem`); rerun every group test on the matched cohort; report Δ-AUC matched vs full.
- **Reviewer overlap:** Clinical (top pick), Biostat (top pick), Reproducibility (sensitivity-cohort echo).
- **Difficulty:** Low–Medium. **Impact:** **Critical**.

### 1.2 Event-locked GLM / HRF first-level analysis
- **Why:** Notebook 02's metric is STD across concatenated trials — a single scalar that ignores GNG event timing. An fNIRS reviewer will (correctly) reject this as the primary activation metric.
- **Method:**
  - `mne_nirs.statistics.run_glm()` with canonical SPM HRF + temporal derivative; AR-IRLS for autocorrelation.
  - Per-subject β maps for **Go**, **No-Go**, and the **No-Go − Go** contrast (the inhibition signal the task was designed to elicit).
  - Second-level GAD vs HC on β maps with permutation cluster correction (see §2.2).
  - Use β-contrast maps as alternative GNN node features and benchmark.
- **Reviewer overlap:** fNIRS (top pick).
- **Difficulty:** Medium. **Impact:** **Critical**.

### 1.3 LOSO stability + pre-specified sensitivity cohorts
- **Why:** With n = 62 and contested HC labels (AH029 self-reported MDD; AH024 ANS dysregulation), reviewers must be shown that the convergent severity channels (S3_D4, S4_D4) and the GNN AUC are not driven by 1–2 subjects.
- **Method:**
  - LOSO: drop one subject, recompute Cohen's *d* / Spearman ρ per channel, get a 62-fold distribution. Report Spearman ρ between LOSO ranking and full ranking, plus a per-channel **stability index** (fraction of folds where it stays in top-k). Tornado plot of S3_D4 / S4_D4 d.
  - **Sensitivity cohorts** (rerun full pipeline on each): **(a)** full N=62, **(b)** drop AH029+AH024, **(c)** additionally drop the 11 demographics-missing GAD. Tabulate Δ-AUC, Δ-FDR-survivors, Δ-d.
  - Add bootstrap 95 % CIs (`scipy.stats.bootstrap`, `BCa`, n=5000) on every Cohen's *d*, Spearman ρ, and AUC currently reported.
- **Reviewer overlap:** Reproducibility (top pick), Biostat (top pick), Clinical (subgroup).
- **Difficulty:** Low. **Impact:** **Critical**.

### 1.4 Saliency–statistics concordance (bridge stats ↔ GNN)
- **Why:** The paper currently has two disconnected halves (statistical maps and GNN performance). Convergence between them is the single strongest signal that the model is using *biology*, not artifacts.
- **Method:**
  - Extract **GAT attention** (`alpha` from `torch_geometric.nn.GATConv`) and **integrated gradients** (`captum.attr.IntegratedGradients`) per channel.
  - Topomap triptych on the 5×7 grid: |Cohen's d|, univariate AUC, GNN saliency.
  - Spearman ρ and Kendall τ between rankings; report per-Hb-type.
  - **Mismatch table**: channels where (stat-significant ∧ low-importance) and (stat-null ∧ high-importance) — discuss each (network effects, suppressors, possible confound).
- **Reviewer overlap:** ML-interp (top pick), Clinical (channels-to-anatomy mapping).
- **Difficulty:** Medium. **Impact:** **Critical**.

### 1.5 Diagnostic-utility metrics (clinical decision support)
- **Why:** A clinical reviewer cannot evaluate the paper without sensitivity / specificity / PPV / NPV at realistic GAD prevalence. "Accuracy = 82 %" is meaningless to a clinician.
- **Method:**
  - Per-subject ROC-AUC with bootstrapped 95 % CI; sensitivity / specificity at Youden's J; PPV/NPV at GAD prevalence (3–6 % community, ~20 % primary care).
  - **DeLong test** comparing GNN AUC vs HAMA-only baseline AUC.
  - **Decision-curve analysis** (`dcurves`) — net-benefit vs intervention threshold.
- **Reviewer overlap:** Clinical (Critical), ML-interp (Critical for the per-channel univariate floor).
- **Difficulty:** Low–Medium. **Impact:** **Critical**.

### 1.6 Batch / data-collection-wave equivalence
- **Why:** The cohort comes from at least 2 waves (`filenamelist_20240305` vs `nirs_metadata.csv` per `data/subjects_ground_truth.xlsx`). The 11 demographics-missing GAD subjects are entirely from the second wave. Reviewers will demand proof that "GAD vs HC" is not "wave A vs wave B".
- **Method:**
  - Code each subject by acquisition wave; run PERMANOVA on channel-wise HbO/HbR feature matrices with `wave` as factor.
  - Logistic regression of `wave ~ diagnosis` to test confounding.
  - Apply **ComBat harmonization** (`neuroCombat`, `neuroharmonize`); rerun group tests; report whether differences survive.
- **Reviewer overlap:** Reproducibility (Critical).
- **Difficulty:** Medium. **Impact:** **Critical**.

---

## TIER 2 — Expected by any thorough reviewer

### 2.1 Mixed-effects models on trial-level data
- **Why:** Aggregating 4 trials into 1 STD per subject discards within-subject variance; mixed effects (subject as random intercept) recovers ~3–4× effective df and matches how the GNN itself is trained (trial-wise).
- **Method:** `statsmodels.MixedLM` or `pymer4` / `lme4`: `HbT ~ Group*Channel + Age + (1|Subject)`; Satterthwaite df. For severity: `STAI ~ HbT + Age + (1|Subject)`.
- **Reviewer:** Biostat. **Difficulty:** Medium. **Impact:** High.

### 2.2 Permutation cluster correction over the 23-channel adjacency
- **Why:** BH-FDR assumes independence; adjacent channels in the 5×7 prefrontal grid are spatially correlated, making BH over-conservative and ignoring cluster-level signal.
- **Method:** `mne.stats.permutation_cluster_test` with adjacency from `mne.channels.find_ch_adjacency`; 10 000 permutations; TFCE preferred. Replaces or augments BH-FDR in §02 / §04.
- **Reviewer:** Biostat. **Difficulty:** Medium. **Impact:** High.

### 2.3 Functional connectivity as GNN edge weights
- **Why:** A "graph-based fNIRS method" paper that uses only spatial proximity for edges is graph-theoretically thin. FC-driven adjacency is the standard justification for choosing a GNN over a CNN.
- **Method:** Compute 23×23 connectivity matrices: Pearson, partial correlation, wavelet coherence (0.01–0.1 Hz task band, MNE-NIRS), optionally PLV. Use them as GNN adjacency (data-driven) instead of, or alongside, spatial 5×7 distance; benchmark against current model.
- **Reviewer:** fNIRS. **Difficulty:** Low–Medium. **Impact:** High.

### 2.4 Channel-ablation studies (necessity / sufficiency)
- **Why:** Causal-style test that the four uncorrected-significant channels are necessary and/or sufficient drivers of GNN performance.
- **Method:** Retrain (and inference-mask) GNN with the top-k MWU channels (S1_D1, S5_D5, S3_D3, S3_D1) zeroed; inverse — keep only those k. Repeat k = 1..6. Report Δ-AUC with bootstrap CIs.
- **Reviewer:** ML-interp. **Difficulty:** Medium. **Impact:** High.

### 2.5 fNIRS-BIDS / Best-Practices QC reporting
- **Why:** The Yücel 2021 *Best Practices for fNIRS Publication* checklist is now a de-facto reviewer rubric; without per-channel SCI / CV / motion-correction parameters, papers are routinely desk-rejected.
- **Method:**
  - **Scalp Coupling Index** (Pollonini): % channels rejected per subject.
  - **CV(%)** per channel, **PSP**.
  - Report motion-correction parameters used (already in `processor_cli.py`); list trials excluded per subject.
  - **HbO–HbR anti-correlation QC** (expect *r* < −0.3 for genuine neurovascular).
  - Document or implement systemic-physiology removal: short-channel regression (you have no short channels — flag this), CBSI, PCA / tPCA global-signal regression (`hmrR_PCAFilter`).
- **Reviewer:** fNIRS, Reproducibility. **Difficulty:** Low–Medium. **Impact:** High.

### 2.6 Within-GAD severity probing of GNN embeddings
- **Why:** Shows the GNN encodes a graded severity axis that matches Notebook 04, not just a binary boundary.
- **Method:** Extract penultimate GNN embeddings; Spearman between embedding dims (or a linear probe) and STAI-T / HAMA on GAD subjects only; cross-reference against the convergent S3_D4 / S4_D4 channels.
- **Reviewer:** ML-interp. **Difficulty:** Low–Medium. **Impact:** High.

### 2.7 Permutation classifier null + post-hoc sensitivity power
- **Why:** With n = 62, "82 % accuracy" is unfalsifiable without an empirical null. Reviewers also want post-hoc minimum-detectable-d.
- **Method:** 1 000 label-shuffle permutations (subject-level, not trial-level — to respect LOSO leakage rules); empirical *p*. Pair with `statsmodels.stats.power` reporting that with n=33/29, α=.05/23 BH-FDR, MDE ≈ |d| ≈ 0.95.
- **Reviewer:** Reproducibility, Biostat. **Difficulty:** Low. **Impact:** High.

---

## TIER 3 — Polish that strengthens the paper

### 3.1 Bayes factors / equivalence tests for null claims
- **Why:** "p > .05" ≠ no effect. The paper currently makes two implicit null claims that need formal null evidence: (a) HbT is *worse* than HbO/HbR (currently a Wilcoxon *rejection*, but BF01 supporting this would be cleaner), (b) "0 / 23 FDR-significant" implies absence of effect.
- **Method:** `pingouin.bayesfactor_ttest` (BF01); TOST equivalence (`statsmodels.stats.weightstats.tost_paired` / `tost_ind`) with SESOI |d| = 0.3.
- **Reviewer:** Biostat. **Difficulty:** Low. **Impact:** Medium–High.

### 3.2 Cross-cohort replication on `processed-old`
- **Why:** Project memory (`auto-memory:project_dataset_state.md`) records that `data/processed-old/` exists with an independently processed version of the same cohort. Internal replication on a parallel pipeline is the cheapest defense against pipeline-overfitting.
- **Method:** Freeze ST-GNN hyperparameters from `processed-new-mc`; retrain / evaluate on `processed-old` with the same k-fold CV; report accuracy, F1, and channel-attention overlap (Jaccard / rank-correlation of top-k channels).
- **Reviewer:** Reproducibility. **Difficulty:** Medium. **Impact:** High.

### 3.3 Channel-to-anatomy mapping (MNI / Brodmann)
- **Why:** "Channel 14 (S5_D5) is important" is not interpretable to a clinical reader. "Right dlPFC hypoactivation, consistent with Yeung et al. 2020" is.
- **Method:** Map montage to MNI space via fOLD or AtlasViewer; tabulate Brodmann areas; cross-reference convergent channels (S3_D4 / S4_D4 / S1_D1 / S5_D5) against published GAD-PFC literature with effect-size comparison.
- **Reviewer:** Clinical. **Difficulty:** Medium. **Impact:** High.

### 3.4 External cohort generalization probe
- **Why:** Single-site n = 62 classifiers virtually never generalize. A held-out probe — even a small one — substantially raises the paper's clinical credibility.
- **Method:** Apply trained model (with CORAL) to a public fNIRS anxiety/depression dataset (Bak 2023 OpenNeuro, or any internally available pilot data); report calibration plot and Brier score.
- **Reviewer:** Clinical. **Difficulty:** High. **Impact:** High.

### 3.5 Subgroup / comorbidity analyses
- **Why:** Pure GAD is rare in clinical practice; reviewers want to know whether the model collapses on comorbid mood / autonomic cases.
- **Method:** AUC by HAMA severity tertile (mild / moderate / severe); separate report on the 11 demographics-missing GAD; LOOCV-rerun excluding LA063 / AH029 / AH024.
- **Reviewer:** Clinical. **Difficulty:** Low. **Impact:** Medium–High.

### 3.6 Open-science release plan
- **Method:** Zenodo DOI for code + processed features; OSF preregistration of analysis plan (retrospective if necessary, clearly timestamped post-hoc); `requirements.txt` + Docker image.
- **Reviewer:** Reproducibility. **Difficulty:** Low. **Impact:** Medium.

---

## Cross-reviewer convergence map

| Recommendation | Clinical | fNIRS | Biostat | ML-interp | Reproducibility | Verdict |
|----------------|:-:|:-:|:-:|:-:|:-:|---------|
| Age adjustment / matched subsample | ★ | • | ★ | • | • | **Tier 1** (3 explicit, 5/5 endorse) |
| Event-locked GLM / HRF / Go vs No-Go | • | ★ | • | • | | **Tier 1** |
| LOSO stability + sensitivity cohorts | • | | ★ | • | ★ | **Tier 1** |
| Saliency–statistics concordance figure | • | | | ★ | | **Tier 1** |
| Diagnostic utility (PPV / NPV / DCA) | ★ | | | ★ | | **Tier 1** |
| Batch / wave equivalence test | | | | | ★ | **Tier 1** |
| Mixed-effects on trials | | | • | • | | Tier 2 |
| Permutation cluster correction (TFCE/NBS) | | | • | | | Tier 2 |
| FC-based GNN edges | | ★ | | | | Tier 2 |
| Channel ablation (necessity/sufficiency) | • | | | • | | Tier 2 |
| fNIRS-BIDS QC reporting (SCI, CV, motion) | | • | | | • | Tier 2 |
| Severity probing of GNN embeddings | • | | | • | | Tier 2 |
| Permutation classifier null + power | | | • | | • | Tier 2 |
| Bayes factors / TOST | | | • | | | Tier 3 |
| Cross-cohort replication on `processed-old` | | | | | • | Tier 3 |
| MNI / Brodmann anatomy mapping | • | • | | | | Tier 3 |
| External public-cohort probe | • | | | | | Tier 3 |
| Subgroup / comorbidity analyses | • | | | | • | Tier 3 |
| Open-science release plan | | | | | • | Tier 3 |

★ = top pick. • = secondary endorsement.

---

## Suggested next-notebook targets (concrete artefacts)

If we move forward, these would be the natural next notebooks under
`src/notebook/statistical-analysis/`:

| Folder | Notebook | Tier 1 contents |
|--------|----------|------------------|
| `05_age_adjusted/` | `05_ancova_age_adjusted.ipynb` | ANCOVA, age-matched subsample, refit §02 / §04 — supersedes raw MWU as primary inference |
| `06_glm_hrf/` | `06_glm_hrf_first_level.ipynb` | MNE-NIRS GLM, Go vs No-Go β-contrast, second-level group test, β-feature export for GNN |
| `07_robustness/` | `07_loso_sensitivity.ipynb` | LOSO d-stability, sensitivity cohorts (a/b/c), batch/wave PERMANOVA + ComBat |
| `08_saliency_bridge/` | `08_saliency_vs_stats.ipynb` | GAT attention extraction, IG, Spearman/Kendall vs |d|/AUC, mismatch table, ablation curves |
| `09_diagnostic_utility/` | `09_classifier_metrics.ipynb` | Per-channel AUC floor, logistic baseline, GNN ROC/PPV/NPV/DCA, DeLong |

These are all addressable with the existing `processed-new-mc` data and the
project's existing GNN / ST-GNN code.

---

## Single-sentence verdict

The four notebooks completed so far establish *that* there is a group
difference and a within-GAD severity gradient; the **age confound (Tier 1.1)**,
the **event-locked GLM (Tier 1.2)**, the **LOSO/sensitivity battery (Tier 1.3)**,
the **stats-↔-GNN bridge (Tier 1.4)**, and **clinical-utility metrics (Tier 1.5)**
are what will turn this into a defensible journal submission.
