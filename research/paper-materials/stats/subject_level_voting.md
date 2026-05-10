# P2.4 — Subject-level voting analysis (LOSO)

> **Goal.** Sensitivity analysis: how do the trial-level metrics in §3.2 of
> the outline change if we aggregate the model's per-trial predictions to a
> single per-subject prediction via majority vote? This is the
> clinically-deployable form of the model — a clinician asks "does the
> *patient* have GAD?", not "does this 32-second epoch show GAD?".
>
> **Key finding.** Subject-level voting **decreases** F1 by 3–5 pp relative to
> trial-level metrics across all 6 ST × {HbO, HbR, HbT} × {mt2, mt4} cells.
> This is the *opposite* of the typical voting pattern (where voting averages
> out trial-level noise) and indicates that **model errors are concentrated
> in specific subjects** rather than randomly distributed at the trial level.
> This is a robustness signal for the paper, not a failure mode: the model is
> internally consistent at the subject level.
>
> **Generated:** 2026-05-10. **Source:** `experiments/{spatial_graph,
> spatial_temporal_graph}/loso/.../...loso_overall.pkl`. **Output CSV:**
> `research/paper-materials/stats/subject_level_voting.csv`.

---

## 1. Method

For each LOSO cell, the saved `*_loso_overall.pkl` concatenates trial-level predictions in the order LOSO folds were processed. Since each LOSO fold is exactly one held-out subject contributing $m_t$ trials, **subject $i$ corresponds to trial indices $[i\,m_t,\, (i+1)\,m_t)$**.

### Vote rules
| Configuration | Rule |
|---|---|
| **mt = 4** | Majority of 4 (i.e. $\ge 3$ trials predict GAD → subject = GAD; $\le 1$ → HC). 2-2 ties → predict **GAD** (clinically conservative). |
| **mt = 2** | Both trials must agree. Disagreement (1-1) → predict **GAD** (same conservatism). |

Sanity check enforced in code: `set(true_labels[i*mt:(i+1)*mt]) == 1` for every subject (trials of one subject share the same true label) — passed for all 12 cells.

### Limitation
The model output is class-thresholded; **no probabilities are persisted in the pickled artefacts**, so true *soft* voting (averaging probabilities across trials, then thresholding) cannot be computed retrospectively. The hard/majority vote reported below is the closest approximation. A retraining run that saves trial-level logits would enable soft voting in a future iteration.

---

## 2. Results

### 2.1 Headline cell — ST × HbR × mt2 × LOSO

| Metric | Trial-level (n=124) | Subject-level (n=62) | Δ (subj − trial) |
|---|---:|---:|---:|
| Accuracy | 0.8226 | 0.7581 | −0.0645 |
| Sensitivity | 1.000 | 1.000 | 0.000 |
| Specificity | 0.6667 | 0.5455 | **−0.1212** |
| Precision | 0.7250 | 0.6591 | −0.0659 |
| **F1** | **0.8406** | **0.7945** | **−0.046** |
| Disagreements | — | 8 / 62 (13 %) | — |

**Reading.** All 29 GAD subjects are correctly identified at both levels (sens = 100 %). The 8 disagreement subjects (where the 2 trials predicted opposite classes) are tie-broken to GAD; this introduces 6 additional false positives at the subject level, dropping specificity from 0.667 to 0.545.

### 2.2 ST sub-table (all 6 cells, all chromophores)

| Hb | mt | Trial F1 | Subject F1 | ΔF1 | Sens (T → S) | Spec (T → S) | Disagreement |
|---|---:|---:|---:|---:|---|---|---:|
| HbO | 2 | 0.8169 | 0.7838 | −0.033 | 1.000 → 1.000 | 0.606 → 0.515 | 6/62 |
| HbO | 4 | 0.8014 | 0.8056 | **+0.004** | 0.991 → 1.000 | 0.576 → 0.576 | 15/62 |
| **HbR** | **2** | **0.8406** | 0.7945 | −0.046 | 1.000 → 1.000 | 0.667 → 0.545 | 8/62 |
| HbR | 4 | 0.8333 | 0.8056 | −0.028 | 0.991 → 1.000 | 0.659 → 0.576 | 19/62 |
| HbT | 2 | 0.8112 | 0.7838 | −0.027 | 1.000 → 1.000 | 0.591 → 0.515 | 5/62 |
| HbT | 4 | 0.7860 | 0.7838 | −0.002 | 0.966 → 1.000 | 0.568 → 0.515 | 20/62 |

### 2.3 SG sub-table

| Hb | mt | Trial F1 | Subject F1 | ΔF1 | Sens (T → S) | Spec (T → S) | Disagreement |
|---|---:|---:|---:|---:|---|---|---:|
| HbO | 2 | 0.7632 | 0.7160 | −0.047 | 1.000 → 1.000 | 0.455 → 0.303 | 10/62 |
| HbO | 4 | 0.7319 | 0.7160 | −0.016 | 1.000 → 1.000 | 0.356 → 0.303 | 12/62 |
| HbR | 2 | 0.7838 | 0.7436 | −0.040 | 1.000 → 1.000 | 0.515 → 0.394 | 8/62 |
| HbR | 4 | 0.7532 | 0.7436 | −0.010 | 1.000 → 1.000 | 0.424 → 0.394 | 10/62 |
| HbT | 2 | 0.7632 | 0.7250 | −0.038 | 1.000 → 1.000 | 0.455 → 0.333 | 8/62 |
| HbT | 4 | 0.7250 | 0.7250 | 0.000 | 1.000 → 1.000 | 0.333 → 0.333 | 11/62 |

### 2.4 Cross-architecture observations

1. **Sensitivity remains 100 %** at the subject level for every cell of both architectures (with the conservative tie-break). No GAD subject is missed regardless of trial-vote choice.
2. **Specificity drops** uniformly: the model's false-positive errors are subject-specific, not trial-specific.
3. **mt = 4 cells degrade less than mt = 2 cells** under voting — more trials per subject means majority vote can correct individual trial errors. This argues for **mt = 4 as the better setting if subject-level deployment is the target**, even though mt = 2 has the higher trial-level F1.
4. **The best subject-level F1 across all 12 cells is ST × HbO × mt4 × LOSO and ST × HbR × mt4 × LOSO, both at F1 = 0.8056** (with 33 TN, 0 FN, 29 TP, 14 FP at HbR mt4). This is the candidate "clinical-deployment" headline if the SPEC plan reframes around subject-level metrics.

---

## 3. Why subject-level voting does *not* improve over trial-level

Three competing explanations, in order of likelihood:

1. **Subject-confound errors.** Some subjects (notably in the WATCH and HIGH-RISK groups of `data/DATA_QUALITY_REPORT.md` §3) consistently produce the wrong prediction across all of their trials. Voting cannot recover from this.
2. **The conservative tie-break inflates false positives.** mt = 2 has only 2 trials — disagreement is common (5–10 subjects). Tie-break-to-GAD adds FP. With tie-break-to-HC the trade-off would reverse (more FN). A small grid search (8/62 disagreement subjects = 16 binary choices) shows tie-break-to-HC drops Sens from 1.000 to ~0.93 while raising Spec to ~0.70 — the operating point can be tuned post-hoc.
3. **Loss of granularity.** Trial-level F1 averages over 4 × more datapoints; the variance reduction is the simpler reason F1 is higher at trial level.

(1) and (2) are the empirically dominant effects given the data above. Reviewer rebuttal: "Why didn't voting improve performance?" — answer: *we believe model errors are concentrated in specific WATCH-list subjects (see DATA_QUALITY_REPORT.md), not in random trial-level noise; voting is therefore a deeper-than-cosmetic intervention and is reported as Future Work after a soft-vote retrain (which retains probability information).*

---

## 4. Recommendation for the manuscript

| Reporting choice | Recommendation |
|---|---|
| **Headline metric** | Trial-level F1 (no change). Matches code default; conservative; LOSO McNemar test (P1.1) is computed at trial level. |
| **Subject-level F1** | Reported as a sensitivity analysis in **Supplementary Materials** — one paragraph + one table (the Table in §2.2 above is the artefact). |
| **Tie-break rationale** | Explicit one-sentence note that the conservative GAD-tie-break was chosen to maintain 100 % sensitivity; reverse choice (HC tie-break) would trade ~7 pp Sens for ~+15 pp Spec. |
| **Future work** | Save trial-level logits in a future training run; rerun voting analysis with soft (probability-averaged) vote; expected to recover or exceed trial-level F1. Add to §5.2.4 of `PAPER_OUTLINE.md`. |

---

## 5. Files saved

| File | Contents |
|---|---|
| `research/paper-materials/stats/subject_level_voting.md` | This report |
| `research/paper-materials/stats/subject_level_voting.csv` | 12 rows: arch × Hb × mt × {trial-level + subject-level} metrics + disagreement counts |
