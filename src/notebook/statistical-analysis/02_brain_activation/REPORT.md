# 02 — Per-Channel Brain Activation Analysis (HbO, GNG)

> Paper-writing summary. Source notebook: `02_brain_activation_anova.ipynb`.
> Data: `data/processed-new-mc/GNG/`.
>
> **Primary chromophore: HbO** (changed from HbT in the original draft per
> Yücel et al. 2021 *Best Practices for fNIRS Publications* and confirmed by
> §03's HbO ≈ HbR > HbT discriminability ranking on this cohort).

## 1. Purpose

Statistical comparison of per-channel HbO activation between HC and GAD on
the GNG (Go/No-Go) task. Provides neurophysiological evidence supporting the
use of GNG-derived HbO features in the downstream graph models.

## 2. Cohort & Data

- **HC** n = 33, **GAD** n = 29 (62 total). Defined by folders under
  `data/processed-new-mc/GNG/{healthy, anxiety}/`.
- 4 epoched HbO trials per subject of shape (23 channels × 326 time points,
  ≈ 32 s @ 10.17 Hz). 23-channel prefrontal montage on a 5×7 grid.

## 3. Methods

- **Activation metric.** Per subject and channel: trials concatenated along
  time (4 × 326 = 1 304 samples), then **STD across the concatenated time
  series** as a scalar amplitude per (subject × channel). For zero-mean,
  band-pass-filtered fNIRS data this STD is equivalent to RMS amplitude.
- **Normality.** Shapiro-Wilk on three representative channels (S1_D1,
  S4_D7, S8_D8); HbO is mostly normal at S4_D7 / S8_D8 but non-normal at
  S1_D1 (HC *p* = 0.004). Non-parametric test preferred for safety.
- **Group comparison.** Mann-Whitney *U* (two-sided), HC vs GAD, run
  independently on each of the 23 channels.
- **Multiple comparisons.** Benjamini-Hochberg FDR control at *q* = 0.05
  across the 23 channels.
- **Effect size.** Cohen's *d* (pooled SD); positive = GAD > HC.
- **Cross-channel summary.** Grand-mean per subject = mean STD across the
  23 channels; HC vs GAD compared with a single Mann-Whitney *U*.

## 4. Results

### 4.1 Headline numbers

- Channels with **uncorrected** *p* < 0.05: **10 / 23**.
- Channels surviving **FDR** (*q* < 0.05): **4 / 23**.
- Direction of every significant channel: **HC > GAD** (negative *d*; smaller
  prefrontal HbO amplitude in the GAD group).

### 4.2 FDR-significant channels (4)

| Channel | Mean HC ± SD | Mean GAD ± SD | Δ (GAD − HC) | Cohen's *d* | *U* | *p* (raw) | *p* (FDR) |
|---------|---:|---:|---:|---:|---:|---:|---:|
| **S5_D5** | 0.918 ± 0.032 | 0.885 ± 0.040 | −0.033 | **−0.92** | 715 | **8.7 × 10⁻⁴** | **0.014** |
| **S2_D1** | 0.913 ± 0.032 | 0.877 ± 0.046 | −0.036 | **−0.92** | 707 | **1.3 × 10⁻³** | **0.014** |
| **S3_D3** | 0.910 ± 0.036 | 0.883 ± 0.034 | −0.027 | **−0.77** | 695 | **2.3 × 10⁻³** | **0.014** |
| **S1_D1** | 0.908 ± 0.044 | 0.868 ± 0.054 | −0.040 | **−0.81** | 694 | **2.4 × 10⁻³** | **0.014** |

### 4.3 Top-10 channels by uncorrected *p* (the 10 surviving raw α = 0.05)

| Channel | Cohen's *d* | *U* | *p* (raw) | *p* (FDR) | FDR sig |
|---------|---:|---:|---:|---:|---|
| S5_D5 | −0.92 | 715 | 8.7 × 10⁻⁴ | 0.014 | ✓ |
| S2_D1 | −0.92 | 707 | 1.3 × 10⁻³ | 0.014 | ✓ |
| S3_D3 | −0.77 | 695 | 2.3 × 10⁻³ | 0.014 | ✓ |
| S1_D1 | −0.81 | 694 | 2.4 × 10⁻³ | 0.014 | ✓ |
| S3_D4 | −0.64 | 659 | 1.1 × 10⁻² | 0.051 |   |
| S4_D5 | −0.66 | 653 | 1.4 × 10⁻² | 0.054 |   |
| S2_D5 | −0.53 | 641 | 2.2 × 10⁻² | 0.064 |   |
| S2_D2 | −0.55 | 641 | 2.2 × 10⁻² | 0.064 |   |
| S3_D1 | −0.56 | 633 | 3.0 × 10⁻² | 0.076 |   |
| S5_D2 | −0.50 | 619 | 4.8 × 10⁻² | 0.111 |   |

Full per-channel table: `results_brain_activation_stats.csv`.

### 4.4 Anatomical pattern

The four FDR-significant channels — `S5_D5`, `S2_D1`, `S3_D3`, `S1_D1` —
cluster on the **superior-medial** quadrant of the prefrontal grid, with
the next-strongest tier (`S3_D4`, `S4_D5`, `S2_D2`, `S2_D5`, `S3_D1`,
`S5_D2`) extending laterally. All show **reduced HbO STD amplitude in GAD**.

→ Topographic maps: `fig_topo_activation.png` (group-mean STD and
GAD − HC contrast) and `fig_topo_effect_size.png` (Cohen's *d* per channel,
4 channels marked `#` for FDR significance).

### 4.5 Grand-mean activation across channels

| Group | n | Grand-mean STD ± SD |
|-------|--:|--------------------:|
| HC    | 33 | 0.903 ± 0.031 |
| GAD   | 29 | 0.886 ± 0.028 |

Mann-Whitney *U* = 643, *p* = 0.021 (*). → `fig_task_grand_mean.png`.

Unlike the HbT version, the **whole-cortex average HbO does discriminate
groups** at the uncorrected level — consistent with the higher signal-to-
noise ratio of HbO.

## 5. Discussion / Interpretation

1. **HbO yields a substantially stronger group separation than HbT.** Where
   HbT had 4 / 23 raw, 0 / 23 FDR-significant channels, HbO produces
   **10 / 23 raw, 4 / 23 FDR-significant**. The convergent finding from §03
   (HbO ≈ HbR > HbT) is now confirmed at the per-channel statistical level.
2. **Direction is consistent.** All significant channels show HC > GAD,
   indicating reduced prefrontal HbO amplitude in chronic anxiety. This is
   compatible with a "prefrontal hypoactivation" interpretation in GAD
   literature.
3. **Anatomical convergence.** The four FDR-significant channels are spatial
   neighbours on the 5×7 grid, suggesting a real neurophysiological
   cluster rather than scattered single-channel noise. A cluster-based
   correction (TFCE or NBS, currently in `FUTURE_ANALYSES.md` Tier 2)
   would likely strengthen this further.
4. **Implication for the modelling pipeline.** The downstream graph models
   should ingest **HbO as the primary node feature** (replacing the
   current HbT-only setup), with HbR as a secondary stream and HbT as
   supplementary.

## 6. Caveats

1. **Demographic imbalance.** The HC group is on average 22 years older than
   GAD (§01 §6). Prefrontal hemodynamic amplitude declines with age; the
   reduced GAD HbO reported here could partly reflect that confound. **§05
   (age-adjusted ANCOVA) addresses this directly: the top 4 FDR channels
   here remain raw-significant after age control on the n = 51 subset, with
   `S5_D5` retaining the largest effect (β = −0.048, *p* = 0.0023, η²ₚ =
   0.18).**
2. **Activation metric.** STD across concatenated trials is amplitude-only;
   it cannot separate increased noise from increased task-evoked response
   and ignores temporal structure. **§06 addresses this** with canonical-HRF
   β + peak amplitude + cluster-permutation analyses (also confirms HbO
   superiority over HbT, with 2 / 23 FDR-significant β channels).
3. **No per-stimulus event timing**, so this cannot be re-cast as a true
   first-level GLM with Go-vs-No-Go contrast.

## 7. Generated artefacts

| File | Type | What it contains |
|------|------|------------------|
| `02_brain_activation_anova.ipynb` | Notebook | Full reproducible analysis |
| `results_brain_activation_stats.csv` | CSV | All 23 channels: HC/GAD means, *U*, *p*_raw, *p*_FDR, *d* |
| `fig_topo_activation.png` | Figure | HC mean / GAD mean / Δ topographic maps |
| `fig_topo_effect_size.png` | Figure | Per-channel Cohen's *d* topographic map (4 FDR-sig marked `#`) |
| `fig_channel_task_heatmap.png` | Figure | Channels ranked by −log10(p) and by *d* |
| `fig_gng_sig_channels.png` | Figure | Bar plots for the 10 uncorrected-significant channels |
| `fig_task_grand_mean.png` | Figure | Grand-mean STD across channels, HC vs GAD |
