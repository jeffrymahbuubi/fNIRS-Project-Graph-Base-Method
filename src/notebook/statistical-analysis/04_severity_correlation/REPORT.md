# 04 — Clinical-Severity Correlation (HbO Activation vs Anxiety Scores, GNG)

> Paper-writing summary. Source notebook: `04_severity_correlation.ipynb`.
> Data: `data/processed-new-mc/GNG/anxiety/`. Scores from
> `data/subjects_ground_truth.xlsx`.
>
> **Primary chromophore: HbO** (changed from HbT per Yücel et al. 2021 +
> §03's HbO ≈ HbR > HbT discriminability finding).

## 1. Purpose

Within the GAD group (n = 29), test whether per-channel HbO activation
amplitude on the GNG task tracks **clinical anxiety severity**, indexed by
two standard scales: STAI-T (trait anxiety) and HAMA (Hamilton Anxiety).

## 2. Cohort & Data

- **GAD subjects:** n = 29 (the same as in §02 / §03). All 29 have STAI-T;
  HAMA uses **n = 28** (LA063 excluded — `HAMA_sum = 0` is administrative).
- Severity scores (descriptive, GAD only):

  | Scale  |  n | Mean ± SD   | Range  |
  |--------|---:|------------:|-------:|
  | STAI-T | 29 | 54.2 ± 11.6 | 28–75 |
  | HAMA   | 28 | 20.8 ± 9.1  | 7–40  |

- Activation metric: same as §02 / §03 (per-channel STD across the
  concatenated 4-trial HbO time series, 1 304 samples per channel).

> **Note on demographics-missing subjects.** Not affected — the 11 GAD
> subjects without demographics still have HAMA / STAI scores. All 29 GAD
> participate.

## 3. Methods

- **Spearman rank correlation** between (subject HbO STD on channel *c*) and
  (subject severity score), independently for each of the 23 channels.
- **Multiple-comparison correction.** BH-FDR across the 23 channels
  (*q* = 0.05).

## 4. Results

### 4.1 Headline numbers

| Measure | n | Channels *p* < 0.05 (raw) | Channels *q* < 0.05 (FDR) | Direction |
|---------|--:|----:|----:|---|
| STAI-T  | 29 | 1 / 23 | 0 / 23 | Negative (higher anxiety → lower amplitude) |
| HAMA    | 28 | 3 / 23 | 0 / 23 | All negative |

### 4.2 Top channels by |Spearman *r*|

**STAI-T (n = 29)**

| Rank | Channel | Spearman *r* | *p* (raw) | *p* (FDR) |
|----:|---------|---:|---:|---:|
| 1 | **`S3_D4`** | **−0.44** | **0.016** | 0.378 |
| 2 | S5_D2 | −0.37 | 0.051 | 0.461 |
| 3 | S8_D5 | −0.31 | 0.104 | 0.461 |
| 4 | S3_D6 | −0.29 | 0.124 | 0.461 |
| 5 | S8_D8 | −0.28 | 0.136 | 0.461 |

**HAMA (n = 28, LA063 excluded)**

| Rank | Channel | Spearman *r* | *p* (raw) | *p* (FDR) |
|----:|---------|---:|---:|---:|
| 1 | **`S7_D4`** | **−0.51** | **0.006** | 0.138 |
| 2 | **`S3_D4`** | **−0.41** | **0.029** | 0.331 |
| 3 | **`S8_D7`** | **−0.38** | **0.043** | 0.331 |
| 4 | S8_D5 | −0.35 | 0.065 | 0.375 |
| 5 | S3_D6 | −0.32 | 0.093 | 0.386 |

→ Topographic maps: `fig_stait_correlation_topo.png`,
`fig_hama_correlation_topo.png`. |*r*| ranking bar chart:
`fig_severity_top_channels.png`. Scatter plots for the top channel per
measure: `fig_s7d6_severity_scatter.png` (filename retained for parity with
the reference notebook; actual top channels are **`S3_D4`** for STAI-T and
**`S7_D4`** for HAMA).

### 4.3 Cross-measure overlap

Channels appearing in the uncorrected-significant list for **both** measures:

| Channel | STAI-T *r* | HAMA *r* |
|---------|---:|---:|
| **`S3_D4`** | −0.44 | −0.41 |

`S3_D4` is the single channel with cross-measure replication, with both
correlations of similar magnitude (~−0.4) and direction. The HAMA-only top
channel `S7_D4` (*r* = −0.51) is the strongest single-channel association
of either measure.

### 4.4 Direction of effect

Every uncorrected-significant correlation is **negative**: higher anxiety
score → lower per-channel HbO STD amplitude — consistent with §02 (HC > GAD
on group-level activation) and with the prefrontal-hypoactivation
interpretation.

## 5. Discussion / Interpretation

1. **The severity gradient is concentrated at `S3_D4`** for STAI-T and
   peaks at `S7_D4` for HAMA. Both lie in the central/lateral region of
   the 5×7 prefrontal grid, distinct from but adjacent to the
   §02-FDR-significant cluster (`S5_D5`, `S2_D1`, `S3_D3`, `S1_D1`,
   superior-medial).
2. **Effect size moderate (|*r*| ≈ 0.4–0.5).** With n = 28–29, no channel
   survives 23-way FDR correction, but the cross-measure overlap at `S3_D4`
   is non-trivial evidence that the signal is real.
3. **HAMA produces stronger correlations than STAI-T** with HbO (top |*r*|
   = 0.51 vs 0.44), opposite to the HbT version where STAI-T was slightly
   stronger. HAMA is a clinician-rated measure (vs STAI-T self-report), so
   stronger HAMA-HbO correlation may reflect the clinician's exposure to
   behavioural cues that the prefrontal HbO amplitude actually tracks.
4. **Modelling implication.** A regression head predicting STAI-T or HAMA
   from per-channel HbO features is a reasonable next experiment,
   especially anchored to `S3_D4` (cross-measure marker) and `S7_D4` (top
   HAMA marker).

## 6. Caveats

1. **No FDR-significant channel**; all claims are uncorrected and
   exploratory.
2. Within-GAD analysis only (n = 29 / 28). No HC inferences here.
3. **§05 partial-Spearman with age control** dramatically reduces these
   findings: with the n = 18 GAD subset that has age available, **0 / 23
   channels** reach uncorrected significance for either STAI-T or HAMA.
   The §04 results in their original form **cannot be defended in a paper
   that also controls for age**. See §05 §4.5.
4. **`LA063` HAMA = 0** is administrative — explicitly excluded from HAMA
   correlations.

## 7. Generated artefacts

| File | Type | Contents |
|------|------|----------|
| `04_severity_correlation.ipynb` | Notebook | Reproducible analysis |
| `results_severity_correlation.csv` | CSV | 46 rows (2 measures × 23 channels) |
| `fig_severity_score_dist.png` | Figure | STAI-T and HAMA histograms (GAD) |
| `fig_stait_correlation_topo.png` | Figure | 5×7 topo of Spearman *r* (STAI-T) |
| `fig_hama_correlation_topo.png` | Figure | 5×7 topo of Spearman *r* (HAMA) |
| `fig_severity_top_channels.png` | Figure | Ranked &#124;r&#124; bar chart per measure |
| `fig_s7d6_severity_scatter.png` | Figure | Scatter plots — top |r| channel per measure (S3_D4 / S7_D4) |
