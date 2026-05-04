# 03 — HbO / HbR / HbT Comparison (GNG)

> Paper-writing summary for the *Hemoglobin-type selection* sub-section.
> Source notebook: `03_hb_type_comparison.ipynb`. Data:
> `data/processed-new-mc/GNG/{healthy,anxiety}/<subject>/{hbo,hbr,hbt}/`.

## 1. Purpose

Compare the three hemoglobin signal types — oxygenated (HbO), deoxygenated
(HbR), and total (HbT) — for their ability to discriminate HC from GAD on
the GNG task. Provide neurophysiological evidence to justify the choice (or
re-evaluation) of HbT as the primary biomarker used by the downstream graph
models.

## 2. Cohort & Data

- **HC** n = 33, **GAD** n = 29.
- Three (channels × time) tensors per subject — one for each Hb type — each
  with 4 trials of shape (23, 326). Folders are named `hbo/`, `hbr/`,
  `hbt/`.
- **Activation metric.** Per (subject × channel × Hb type): STD across the
  concatenated 4-trial time series (1 304 samples per channel).

## 3. Methods

- **Per-Hb-type analysis (per channel).** For each Hb type independently:
  Mann-Whitney *U* (HC vs GAD, two-sided) on each of the 23 channels,
  followed by BH-FDR correction across the 23 channels (*q* = 0.05).
  Cohen's *d* (pooled SD; positive = GAD > HC).
- **Cross-Hb comparison.** Per-channel |Cohen's *d*| values (one per
  channel, one per Hb type → 23 × 3 matrix) compared with:
  - **Friedman test** (k = 3 conditions, n = 23 channels, paired).
  - **Wilcoxon signed-rank** for each pairwise contrast (HbO–HbR, HbO–HbT,
    HbR–HbT).
- **Grand-mean per Hb type.** Mean STD across the 23 channels per subject;
  HC vs GAD compared with Mann-Whitney *U*.

## 4. Results

### 4.1 Per-channel HC-vs-GAD discriminability per Hb type

| Hb type | Channels *p*<.05 (raw) | Channels *q*<.05 (FDR) | Mean &#124;d&#124; | Max &#124;d&#124; |
|---------|----:|----:|----:|----:|
| **HbO** | 10 / 23 | 4 / 23 | 0.428 | 0.925 |
| **HbR** | 7 / 23  | 5 / 23 | 0.419 | 1.073 |
| **HbT** | 4 / 23  | 0 / 23 | 0.224 | 0.750 |

→ Topographic maps: `fig_hb_type_cohen_d.png`. Per-channel |*d*| bar chart:
`fig_hb_type_abs_d_bar.png`. Full table: `results_hb_type_comparison.csv`.

### 4.2 Top-3 channels by uncorrected *p*, per Hb type

**HbO**

| Channel | *r* of GAD vs HC means | Cohen's *d* | *p* (raw) | *p* (FDR) |
|---------|---:|---:|---:|---:|
| S5_D5 | 0.918 → 0.885 | −0.92 | **8.7 × 10⁻⁴** | **0.0139** |
| S2_D1 | 0.913 → 0.877 | −0.92 | **0.0013**     | **0.0139** |
| S3_D3 | 0.910 → 0.883 | −0.77 | **0.0023**     | **0.0139** |

**HbR**

| Channel | Mean HC → GAD | Cohen's *d* | *p* (raw) | *p* (FDR) |
|---------|--------------:|---:|---:|---:|
| S2_D1 | 0.789 → 0.684 | −1.07 | **2.7 × 10⁻⁴** | **0.0048** |
| S4_D5 | 0.791 → 0.671 | −0.96 | **4.2 × 10⁻⁴** | **0.0048** |
| S3_D4 | 0.742 → 0.655 | −0.79 | **0.0012**     | **0.0095** |

**HbT**

| Channel | Mean HC → GAD | Cohen's *d* | *p* (raw) | *p* (FDR) |
|---------|--------------:|---:|---:|---:|
| S1_D1 | 0.868 → 0.805 | −0.75 | 0.0042 | 0.096 |
| S5_D5 | 0.858 → 0.837 | −0.22 | 0.0098 | 0.113 |
| S3_D3 | 0.858 → 0.812 | −0.51 | 0.0223 | 0.154 |

### 4.3 Cross-Hb comparison on |Cohen's *d*|

- **Friedman χ²(2) = 6.87, *p* = 0.032.** The three Hb types do **not**
  produce equally large effects across the 23 channels.
- Pairwise (Wilcoxon signed-rank, paired across channels):

  | Pair | *W* | *p* | Mean &#124;d&#124; (a) | Mean &#124;d&#124; (b) | Verdict |
  |------|---:|---:|---:|---:|---|
  | HbO – HbR | 123 | 0.665 | 0.428 | 0.419 | HbO ≈ HbR |
  | HbO – HbT |  31 | **0.0006** | 0.428 | 0.224 | **HbO ≫ HbT** |
  | HbR – HbT |  56 | **0.0112** | 0.419 | 0.224 | **HbR > HbT** |

  **Ranking: HbO ≈ HbR > HbT.**

### 4.4 Grand-mean activation per Hb type

| Hb type | HC mean | GAD mean | Mann-Whitney *U* | *p* |
|---------|--------:|---------:|-----------------:|----:|
| HbO | 0.903 | 0.886 | 643 | **0.021 *** |
| HbR | 0.769 | 0.735 | 607 | 0.071 (ns) |
| HbT | 0.838 | 0.817 | 586 | 0.131 (ns) |

→ `fig_hb_type_grand_mean.png`.

### 4.5 Channel spotlight — `S2_D1`

`S2_D1` is the channel with the largest |*d*| in HbR.

| Hb | Cohen's *d* | *p* (raw) |
|----|---:|---:|
| HbO | −0.92 | 0.0013 (**) |
| HbR | −1.07 | 0.0003 (***) |
| HbT | −0.42 | 0.117 (ns) |

→ `fig_s7d6_hb_comparison.png` (filename retained for parity with the
reference notebook; the channel actually shown is `S2_D1` for this cohort).

## 5. Discussion / Interpretation

**Headline.** In this cohort and on the GNG task, **HbO and HbR are
substantially more discriminative between HC and GAD than HbT.** Mean |Cohen's
*d*| across the 23 channels is ≈ 0.42 for HbO and HbR vs ≈ 0.22 for HbT, and
this difference is paired-significant at *p* = 0.0006 (HbO vs HbT) and
*p* = 0.011 (HbR vs HbT). HbO and HbR are not distinguishable from each other
on this metric.

> **Decision (project-wide).** Based on this §03 result and the fNIRS
> best-practice literature (Yücel et al. 2021 *NeuroPhotonics*; Pinti et al.
> 2020 *NYAS*), **HbO has been promoted to the primary chromophore for §02,
> §04, §05, and §06**. HbT analyses have been replaced. HbR remains a
> secondary check that should be reported alongside in any final paper
> figure (canonical fNIRS quality-control: HbO ↑ paired with HbR ↓). After
> the HbO promotion, §02 reaches **4 / 23 FDR-significant channels** (vs
> 0 / 23 with HbT) and §06 canonical-HRF β reaches **2 / 23 FDR**.

**Interpretation note (important for the paper).** This result is **opposite
to the conclusion of the original reference cohort** (Wang et al. 2025-style
analysis), which justified HbT over HbO/HbR. The reversal can be due to:

1. **Cohort difference.** Our GAD group is younger and more demographically
   heterogeneous than the reference HC; subjects with `demographics_missing=Y`
   (n = 11 of 29 GAD) were a separate fNIRS data-collection wave.
2. **Different Hb dynamics.** HbO and HbR carry largely opposite signs in
   typical task-evoked responses and partial cancellation when summed into
   HbT can blunt group-level amplitude differences. This is consistent with
   the channel-level signs we observe (HbO ≈ HbR have similar |d|, HbT
   roughly half).
3. **Pre-processing.** `processed-new-mc` uses motion-correction parameters
   that may differ from those of the original published pipeline.

**Modelling implication.** The downstream graph-based models that ingest only
HbT may be **leaving information on the table**. A near-term experiment is
to repeat the model search with HbO or HbO+HbR features and compare against
the existing HbT baseline.

## 6. Caveats

1. The **Cohen's *d*** comparison tests difference in **|effect size|**
   across the 23 channels for the same group contrast — it does *not* test
   whether HbO/HbR are independently better predictors than HbT in a model;
   that requires a downstream classification benchmark.
2. **No correction across Hb types.** The 23-channel FDR is applied within
   each Hb type but not across the 3 × 23 = 69 simultaneous channel × Hb
   tests. Quoted FDR counts in §4.1 are within-Hb only.
3. **Spotlight channel filename.** `fig_s7d6_hb_comparison.png` retains the
   filename from the reference notebook for diff-friendliness; the channel
   actually shown is `S2_D1` (top |*d*| in HbR for this cohort).

## 7. Generated artefacts

| File | Type | What it contains |
|------|------|------------------|
| `03_hb_type_comparison.ipynb` | Notebook | Full reproducible analysis |
| `results_hb_type_comparison.csv` | CSV | 69 rows (3 Hb × 23 channels): *U*, *p*_raw, *p*_FDR, *d* |
| `fig_hb_type_abs_d_bar.png` | Figure | Per-channel &#124;d&#124; grouped by Hb type |
| `fig_hb_type_cohen_d.png` | Figure | Three topographic maps of Cohen's *d* (HbO, HbR, HbT) |
| `fig_hb_type_grand_mean.png` | Figure | Grand-mean STD per Hb type, HC vs GAD |
| `fig_s7d6_hb_comparison.png` | Figure | Spotlight box plots — channel `S2_D1`, three Hb types |
