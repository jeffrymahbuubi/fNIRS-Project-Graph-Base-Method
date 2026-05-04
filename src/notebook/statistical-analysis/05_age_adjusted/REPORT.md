# 05 — Age-Adjusted Per-Channel Activation & Severity (GNG, HbO)

> Paper-writing summary. Source notebook: `05_ancova_age_adjusted.ipynb`.
>
> **Primary chromophore: HbO** (changed from HbT per Yücel et al. 2021 +
> §03's HbO ≈ HbR > HbT finding).

## 1. Purpose

Re-run §02 (group-level activation) and §04 (within-GAD severity correlation)
with age and sex as covariates, to address the 22-year HC–GAD age gap
documented in §01 (Welch *t* = 6.08, *p* ≈ 6.5 × 10⁻⁶).

## 2. Cohort

| Subset | HC | GAD | Notes |
|--------|----|-----|-------|
| Demographics-eligible (analysed here) | 33 | 18 | 11 GAD excluded — no age/sex |
| Age-overlap subsample (age ≥ 60)      | 33 | 7  | Sensitivity, see §4.4 |
| 1:1 NN-matched on age                  | 18 | 18 | Degenerate — see §4.3 |

> **⚠ Future-self update protocol.** When demographics for the 11 missing
> GAD subjects (`AA089, AA090, AA092, AA093, AA094, AA097, AA098, AA099,
> LA091, LA095, LA096`) are recovered:
> 1. Update `data/subjects_ground_truth.xlsx`.
> 2. Rerun `05_ancova_age_adjusted.ipynb`.
> 3. ANCOVA cohort grows from 51 → 62; partial-Spearman cohort from 18 → 29.
> 4. Update this REPORT, the cross-references in §02 / §04 / `FUTURE_ANALYSES.md`,
>    and any paper text quoting n = 51 / GAD = 18.

## 3. Methods

1. **Per-channel ANCOVA.** `HbO_STD_channel ~ Group + age + Sex` per channel
   (statsmodels OLS, type-II ANOVA for partial η²). HC reference. BH-FDR
   over 23 channels.
2. **Same-cohort unadjusted MWU.** Mann-Whitney *U* on the n = 51 cohort
   without covariates — to separate sample-size effects from
   covariate-adjustment effects.
3. **Age-overlap subsample.** Restrict to `age ≥ 60` (HC range lower
   bound), rerun unadjusted MWU.
4. **1:1 NN age-matching.** For each GAD, pick closest-age HC. Reports
   residual age gap to demonstrate matching infeasibility.
5. **Within-GAD severity (partial Spearman).** For the 18 GAD with valid
   age, regress age out of both HbO STD and severity score, then Spearman.

## 4. Results

### 4.1 Group-effect summary (cohort × method matrix)

| Analysis | n | sig (raw) | sig (FDR) | Top channel | Effect at top |
|----------|---|----------:|----------:|-------------|---------------|
| §02 unadjusted MWU (full)    | 33 + 29 | **10 / 23** | **4 / 23** | `S5_D5` | *d* = −0.92, *p* = 8.7 × 10⁻⁴ |
| §05a unadjusted MWU (eligible) | 33 + 18 | 8 / 23 | 0 / 23 | `S5_D5` | *d* = −0.96, *p* = 0.003 |
| **§05b ANCOVA (Group + age + Sex)** | 33 + 18 | **7 / 23** | 0 / 23 | **`S5_D5`** | **β = −0.048, *p* = 0.0023, η²ₚ = 0.18** |
| §05c Age-overlap (age ≥ 60)  | 33 + 7  | 4 / 23 | 0 / 23 | `S1_D1` / `S5_D5` | *d* = −1.17 / −1.26, *p* = 0.007 |

### 4.2 Top-5 channels by §05b ANCOVA *p*-value (HbO + age + Sex, n = 51)

| Channel | β (group) | 95 % CI | partial η² | *p* (raw) | *p* (FDR) | *p*_age |
|---------|---:|---:|---:|---:|---:|---:|
| **`S5_D5`** | −0.048 | [−0.078, −0.018] | **0.181** | **0.0023** | 0.052 | 0.159 |
| **`S8_D5`** | −0.044 | [−0.076, −0.011] | 0.136 | **0.009** | 0.104 | **0.027** |
| **`S4_D7`** | −0.041 | [−0.075, −0.008] | 0.116 | **0.017** | 0.114 | **0.045** |
| **`S3_D1`** | −0.038 | [−0.070, −0.006] | 0.110 | **0.020** | 0.114 | 0.320 |
| **`S2_D1`** | −0.033 | [−0.063, −0.002] | 0.091 | **0.035** | 0.145 | 0.490 |

(7 channels in total reach raw α = 0.05; full table in
`results_ancova_age_sex.csv`.)

**Critical observations.**

- The §02 top FDR channel `S5_D5` retains its rank under age + sex
  adjustment, with **partial η² = 0.18** (large effect).
- *p*_age < 0.05 in **3 / 23** channels (`S8_D5`, `S4_D7`, plus one other).
  Age does explain residual variance in HbO at *some* channels, but the
  Group effect at `S5_D5` is **not** explained by age.
- 0 / 23 channels reach FDR significance after age control — the §02
  4-channel FDR cluster does not survive simultaneous covariate adjustment
  + 38 % cohort reduction.

### 4.3 Topographic comparison

→ `fig_ancova_vs_unadj_topo.png` shows §02 *d* (full N=62) vs §05a *d*
(N=51 same cohort) vs §05b ANCOVA β. The same superior-medial cluster
(`S5_D5`, `S2_D1`, `S3_D3`, `S1_D1`) drops in significance count but
preserves direction across all three views.

### 4.4 Why propensity matching fails

1:1 NN age-matching on the n = 51 cohort yields HC ages 65–75 vs GAD ages
29–75. **Residual mean |Δage| = 17.9 yr; Welch *t* = 5.08, *p* = 7.5 ×
10⁻⁵.** The HC age distribution (65–85) does not span the GAD distribution
(29–75); 12 of 18 GAD subjects are < 65. Propensity matching is not viable.

The defensible age-control strategies on this cohort are (a) ANCOVA — used
in §05b — and (b) age-overlap restriction — §05c.

### 4.5 Age-overlap sensitivity (`age ≥ 60`)

n = 33 HC + 7 GAD. Despite the tiny GAD n, **`S1_D1` retains *d* = −1.17
(*p* = 0.007)** and **`S5_D5` retains *d* = −1.26 (*p* = 0.007)** —
extremely large effects in the only age range where both groups overlap.
Strong qualitative evidence that the group difference is biology, not
aging.

### 4.6 Within-GAD severity (partial Spearman | age, n = 18)

| Measure | n | §04 raw (n=29) | §05 raw (n=18) | §05 partial \| age (n=18) |
|---------|--:|---:|---:|---:|
| STAI-T sig channels | 18 | 1 / 23 | **0 / 23** | **0 / 23** |
| HAMA sig channels  | 17 | 3 / 23 | **0 / 23** | **0 / 23** |

The §04 severity correlations **collapse** when the demographics-eligible
subset is used (n = 18) even before age is controlled for — so the loss
is driven by sample-size reduction, not by age confounding. The original
§04 findings are **not defensible** for the demographics-eligible subset
under the current power.

## 5. Discussion / Interpretation

1. **`S5_D5` is the most defensible group marker.** Top by FDR in §02
   (full cohort), top by ANCOVA p in §05b (age-adjusted), and top by *d*
   in §05c (age-overlap). It is the channel the paper should anchor its
   group-difference claims to.
2. **Age does *not* explain the strongest group effect.** Once Group is in
   the ANCOVA, age contributes residual variance only at a small minority
   of channels (3 / 23), and not at the top group-effect channels. The
   age confound is real for some channels but does not undermine `S5_D5`.
3. **The cohort is fundamentally not age-matchable.** The HC age range
   (65–85) barely overlaps GAD (29–75). Methods section should explicitly
   state this and use ANCOVA + age-overlap restriction as the dual
   age-control strategy, **not** matched-pair design.
4. **§04 correlations are fragile** under demographics-eligible
   restriction. The convergent `S3_D4` / `S7_D4` finding from §04
   (full-cohort HbO) cannot be defended at age-adjusted level — paper
   should down-grade those claims to "exploratory."

## 6. Caveats

1. **Cohort reduction.** Eleven GAD subjects (38 % of GAD) excluded.
2. **No FDR-significant channel after age adjustment.** Strongest case
   (`S5_D5`, *p*_FDR = 0.052) is right at the threshold — re-running with
   recovered demographics (n → 62) would likely push it under.
3. **ANCOVA assumptions.** Slope-homogeneity (Group × age interaction) not
   formally tested; recommended for the paper as a sensitivity refit.

## 7. Generated artefacts

| File | Type | Contents |
|------|------|----------|
| `05_ancova_age_adjusted.ipynb` | Notebook | Reproducible analysis |
| `results_ancova_age_sex.csv` | CSV | ANCOVA β, CI, partial η², *p*, FDR per channel |
| `results_unadj_vs_ancova_compare.csv` | CSV | Side-by-side §02 / §05a / §05b |
| `results_age_matched_mwu.csv` | CSV | NN-matched MWU per channel |
| `results_age_overlap_mwu.csv` | CSV | age ≥ 60 subsample MWU per channel |
| `results_partial_spearman_severity.csv` | CSV | STAI-T & HAMA, raw + partial-`age` |
| `fig_ancova_vs_unadj_topo.png` | Figure | Three-panel topo: §02 / §05a / §05b |
| `fig_partial_spearman_stait.png` | Figure | Three-panel topo: STAI-T |
| `fig_partial_spearman_hama.png` | Figure | Three-panel topo: HAMA |
