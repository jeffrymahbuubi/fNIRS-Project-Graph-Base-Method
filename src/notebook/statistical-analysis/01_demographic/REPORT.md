# 01 — Demographic & Anxiety-Scale Analysis (GNG Cohort)

> Paper-writing summary for the *Participants* section. Source notebook:
> `01_demographic_analysis.ipynb`. Ground-truth file: `data/subjects_ground_truth.xlsx`.
> Cohort defined by subject folders under `data/processed-new-mc/GNG/`.

## 1. Purpose

Describe the GNG-task participant cohort and quantify between-group differences
on demographic variables (age, sex, education, marital status, occupation) and
on standard anxiety scales (HAMA, STAI-S, STAI-T).

## 2. Cohort

| Group | n (full cohort) | n (demographics-eligible) | Notes |
|-------|----------------:|--------------------------:|-------|
| HC (Healthy Control)         | 33 | 33 | All have full demographics + scales |
| GAD (Generalized Anxiety)    | 29 | 18 | 11 GAD subjects flagged `demographics_missing=Y` |
| **Total**                    | **62** | **51** | |

**11 GAD subjects without demographics** (have HAMA / STAI scores only):
`AA089, AA090, AA092, AA093, AA094, AA097, AA098, AA099, LA091, LA095, LA096`.

**Cohort footnotes (carry over from prior cohort, still applicable):**
- `AH029` (HC) self-reported MDD on psychotherapy + medication.
- `AH024` (HC) self-reported autonomic-nervous-system dysregulation.
- `LA063` (GAD) HAMA not administered (`HAMA_sum=0`); retained in GAD via STAI validation.
- `EH017–EH028` in original Record-Invitation file correspond to `AH017–AH028`
  (data-collector naming error; already corrected in `subjects_ground_truth.xlsx`).

> **⚠ Caveat for the paper.** Demographic statistics in §4.1–§4.3 use the
> demographics-eligible subset (HC=33, GAD=18). Anxiety-scale statistics in
> §4.4–§4.5 use the full GNG cohort (HC=33, GAD=29). HAMA mean excludes LA063.

## 3. Methods

- **Continuous variables** (age, STAI-S, STAI-T): Welch's two-sample *t*-test,
  reported as mean ± SD, range, *t*, *p*. Cohen's *d* is computed for the STAI
  comparisons (pooled SD).
- **Categorical variables** (sex, education, marital status, occupation):
  Pearson χ² test on the contingency table; degrees of freedom and *p*
  reported. Empty rows are dropped before the test.
- **HAMA** is reported descriptively for the GAD group only (HC HAMA = 0 by
  definition); LA063 is excluded from HAMA mean/SD because the scale was not
  administered.
- All tests two-sided. Significance threshold α = 0.05.

## 4. Results

### 4.1 Age

| Group | n | Mean ± SD (years) | Range |
|-------|--:|------------------:|------:|
| HC    | 33 | 73.0 ± 5.6  | 65–85 |
| GAD   | 18 | 51.1 ± 14.7 | 29–75 |

Welch's *t* = 6.08, **p = 6.46 × 10⁻⁶** (very large group difference; large
imbalance must be acknowledged when interpreting downstream brain-activation
results — see §6).

→ Figure: `fig_age_comparison.png`

### 4.2 Sex

| Group | F | M | F% |
|-------|--:|--:|---:|
| HC    | 23 | 10 | 70% |
| GAD   | 14 | 4  | 78% |

χ²(1) = 0.084, *p* = 0.772 (ns). → `fig_sex_distribution.png`

### 4.3 Education

| Level                  | HC | GAD | HC% | GAD% |
|------------------------|---:|----:|----:|-----:|
| Elementary             | 1  | 0   | 3.0  | 0.0 |
| High School            | 8  | 5   | 24.2 | 29.4 |
| University / College   | 21 | 11  | 63.6 | 64.7 |
| Graduate or higher     | 3  | 1   | 9.1  | 5.9 |

χ²(3) = 0.78, *p* = 0.855 (ns). → `fig_education.png`

### 4.4 Marital Status & Occupation (descriptive only, large differences)

- **Marital status:** χ²(4) = 16.6, *p* = 0.0023. The HC group includes 11
  widowed and 5 divorced participants (none in the GAD demographics-eligible
  subset), reflecting the older HC age structure. GAD is dominated by *married*
  and *single, never married* statuses.
- **Occupation:** χ²(3) = 26.9, *p* < 0.0001. HC is dominated by *volunteer*
  (n=12) and *unemployed/homemaker* (n=20); GAD includes 10 *full-time
  employed* participants and no volunteers. Strongly confounded with age.

These two variables are **not** suitable as covariates without explicit
adjustment for age, given the demographic imbalance.

### 4.5 Anxiety Scales (full cohort)

| Scale         | HC mean ± SD (n=33) | GAD mean ± SD (n=29) | Statistic | *p* | Cohen's *d* |
|---------------|--------------------:|---------------------:|----------:|----:|------------:|
| **STAI-S**    | 29.5 ± 8.4 (range 20–46) | 43.2 ± 11.9 (range 22–68) | *t* = 5.15 | 4.45 × 10⁻⁶ | 1.34 |
| **STAI-T**    | 33.9 ± 9.6 (range 20–62) | 54.2 ± 11.6 (range 28–75) | *t* = 7.46 | 7.07 × 10⁻¹⁰ | 1.92 |

→ `fig_stai_comparison.png`

### 4.6 HAMA (GAD only, LA063 excluded)

n = 28, mean ± SD = 20.8 ± 9.1, range [7, 40], median = 20.

| Severity bucket | Threshold | n |
|-----------------|-----------|--:|
| Mild            | < 18      | 10 |
| Moderate        | 18 – 24   | 8  |
| Severe          | ≥ 25      | 10 |

→ `fig_hama_gad.png`

## 5. Discussion / Interpretation

- **Group separation on anxiety scales is very large** (Cohen's *d* of 1.34 for
  STAI-S and 1.92 for STAI-T), confirming the cohort is well separated on the
  symptom dimension that motivated recruitment.
- **Sex and education are well matched** between groups, supporting their use
  as covariates if required.
- **Age, marital status, and occupation are strongly imbalanced**: HC is
  predominantly older, retired/volunteer, widowed/married participants, while
  GAD is younger, working-age, and dominated by married or never-married
  participants. Any group-difference inferences on the fNIRS signal must
  acknowledge or adjust for the **age** confound in particular.

## 6. Caveats

1. **Missing-demographics subset.** 11 GAD subjects (`AA089–AA099`, `LA091`,
   `LA095`, `LA096`) lack age, sex, education, marital, occupation, diagnosis,
   treatment, and medication. They are included in HAMA / STAI statistics
   only. If those records are recovered later, this report (and §4.1–§4.4) must
   be regenerated.
2. **HAMA = 0 for `LA063`** is an *administrative zero*, not a clinical
   measurement. It is excluded from the HAMA mean/SD/distribution in §4.6
   but kept in the cohort for STAI analyses.
3. **Age confound.** The HC mean age is ≈ 22 years older than the GAD mean,
   which is large. Brain-activation analyses (§02, §03) compare groups on
   per-channel STD; future work should consider age regression or matched
   subsets.

## 7. Generated artefacts

| File | Type | What it contains |
|------|------|------------------|
| `01_demographic_analysis.ipynb` | Notebook | Full reproducible analysis (executed) |
| `results_demographic_summary.csv` | CSV | One-row-per-variable summary table for the paper |
| `cohort_per_subject.csv` | CSV | Per-subject ground-truth slice (n=62) for downstream notebooks |
| `fig_age_comparison.png` | Figure | Boxplot + histogram, age by group |
| `fig_sex_distribution.png` | Figure | Stacked-bar sex distribution |
| `fig_education.png` | Figure | Grouped-bar education levels |
| `fig_hama_gad.png` | Figure | HAMA distribution + severity buckets (GAD) |
| `fig_stai_comparison.png` | Figure | STAI-S and STAI-T boxplots + histograms |
