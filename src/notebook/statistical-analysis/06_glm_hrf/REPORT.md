# 06 — Block-Level HRF Analysis on Epoched HbO (GNG)

> Paper-writing summary. Source notebook: `06_glm_hrf_first_level.ipynb`.
> Data: `data/processed-new-mc/GNG/`.
>
> **Primary chromophore: HbO** (changed from HbT per Yücel et al. 2021).
> HbO is the standard chromophore for canonical-HRF analyses — HbT mixes
> partially-anti-correlated HbO and HbR signals and produces a less
> interpretable canonical-fit β.

## 1. Purpose

Add **temporally-aware** activation features (canonical HRF β, peak
amplitude, time-to-peak, AUC, mean amplitude, time-resolved
cluster-permutation) on top of the §02 STD-of-amplitude metric. The fNIRS-
methodology reviewer flagged STD as event-agnostic.

## 2. Scope & data caveat (must be acknowledged in the paper)

The reviewer's full first-level GLM with a **No-Go − Go inhibition contrast**
is not feasible because:

1. `data/processed-new-mc/` contains **already-epoched** trials (4 task-block
   trials × 326 samples ≈ 32 s @ 10.17 Hz per subject); no continuous
   time-series for `mne_nirs.first_level.run_glm()`.
2. Raw `.tri` LSL trigger files in `data/raw/` and `data/additional-raw/`
   contain only **block-level** markers (task / baseline / rest) — no
   per-stimulus Go-vs-No-Go onsets. Per-stimulus behavioural logs
   (E-Prime / PsychoPy CSV) are not in the project tree.

This notebook does the maximally-defensible block-level analog:
per-subject canonical-HRF β, peak / AUC / mean-window amplitude, and
time-resolved cluster-permutation between the two groups' evoked HbO
responses.

> **Future work.** A true first-level GLM with No-Go − Go contrast requires
> (a) continuous fNIRS time-courses and (b) per-stimulus behavioural logs.

## 3. Methods

- **Block-averaged response.** Per subject and channel, the 4 trials are
  averaged to one (23 × 326) HbO trajectory.
- **Canonical SPM HRF design.** Glover-style γ-variate (peak ≈ 6 s,
  undershoot ≈ 16 s, ratio = 6) convolved with a "task ON" boxcar across
  the 32-s epoch; design column z-scored. Per (subject × channel), β
  recovered by OLS of the trajectory on `[const, design]`.
- **Temporal features.** Within the canonical-HRF window 4–14 s:
  *peak amplitude* (max |HbO|), *time-to-peak*, *AUC* (trapezoidal),
  *mean amplitude*.
- **Group comparison.** Per channel, Mann-Whitney *U* (HC vs GAD) on each
  feature; BH-FDR across 23 channels.
- **Time-resolved cluster-permutation.** Per channel, Welch *t* at every
  time point; cluster-mass with parametric two-sided *t*₀.₀₅ threshold;
  1 000 sign-flip permutations; empirical cluster *p*.
- **Cross-method consistency.** Spearman ρ between channel-*d* vectors of
  §02 STD and the §06 features.

## 4. Results

### 4.1 Group-effect summary, per metric

| Metric | sig (raw) | sig (FDR) | mean &#124;d&#124; | Top channel |
|--------|----------:|----------:|---:|-------------|
| §02 STD of HbO (reference) | **10 / 23** | **4 / 23** | 0.428 | `S5_D5` (*d* = −0.92) |
| **§06 Canonical-HRF β** | **8 / 23** | **2 / 23** | **0.478** | **`S4_D5`** (*d* = −0.89) |
| §06 Peak amplitude (4–14 s) | 7 / 23 | 0 / 23 | 0.357 | `S4_D7` |
| §06 Time-to-peak | 5 / 23 | 0 / 23 | 0.249 | `S3_D4` |
| §06 AUC (4–14 s) | 4 / 23 | 0 / 23 | 0.315 | `S5_D8` |
| §06 Mean amplitude (4–14 s) | 4 / 23 | 0 / 23 | 0.316 | `S5_D8` |
| §06 Cluster-permutation (time) | **9 / 23 channels with sig clusters** | — | — | `S4_D5`, `S4_D7`, `S5_D5` (early + late) |

### 4.2 FDR-significant canonical-HRF β channels (2)

| Channel | mean β (HC) | mean β (GAD) | Cohen's *d* | *U* | *p* (raw) | *p* (FDR) |
|---------|---:|---:|---:|---:|---:|---:|
| **`S4_D5`** | +0.184 | −0.053 | **−0.89** | 685 | **0.0037** | **0.048** |
| **`S4_D7`** | +0.191 | −0.018 | **−0.72** | 682 | **0.0042** | **0.048** |

### 4.3 Top-8 channels by canonical-β *p*-value

| Rank | Channel | mean β (HC) | mean β (GAD) | Cohen's *d* | *p* (raw) | *p* (FDR) |
|----:|---------|---:|---:|---:|---:|---:|
| 1 | **`S4_D5`** | +0.184 | −0.053 | **−0.89** | **0.0037** | **0.048** |
| 2 | **`S4_D7`** | +0.191 | −0.018 | −0.72 | **0.0042** | **0.048** |
| 3 | `S2_D1` | +0.250 | −0.001 | −0.78 | 0.0074 | 0.056 |
| 4 | `S7_D7` | +0.227 | +0.077 | −0.57 | 0.012  | 0.066 |
| 5 | `S8_D5` | +0.234 | +0.042 | −0.60 | 0.017  | 0.066 |
| 6 | `S5_D5` | +0.213 | +0.025 | −0.67 | 0.017  | 0.066 |
| 7 | `S1_D1` | +0.215 | +0.020 | −0.70 | 0.023  | 0.075 |
| 8 | `S8_D7` | +0.219 | +0.052 | −0.63 | 0.026  | 0.075 |

Direction: **GAD has lower canonical-HRF β than HC** at every uncorrected-
significant channel — consistent with §02.

### 4.4 Time-resolved cluster-permutation findings

13 significant clusters across 9 channels in two distinct windows:

| Window | Channels (with cluster *p* < 0.05) | Interpretation |
|--------|--------------------|----------------|
| **Early (0–7 s)** | `S2_D1`, `S4_D5`, `S4_D7`, `S5_D5`, `S8_D7` | HC starts the trial below baseline and rises sharply; GAD starts above and is flat — group difference at *trial onset*. |
| **Late (21–32 s)** | `S2_D1`, `S2_D5`, `S4_D4`, `S4_D5`, `S4_D7`, `S5_D2`, `S5_D5`, `S7_D6` | HC continues rising into the response *tail*; GAD declines / undershoots. |

The dominance of the **late** window is the single most striking
observation visible in `fig_evoked_top_channels.png` — HC progressively
ramps up across the 32-s task block while GAD is flat or declining.

### 4.5 Cross-method ranking — STD ≠ canonical β

| Methods compared | Spearman ρ |
|------------------|---:|
| §02 STD vs §06 canonical-HRF β | **+0.19** (weak positive) |
| §02 STD vs §06 peak amp | +0.04 |
| §02 STD vs §06 AUC | −0.16 |
| §02 STD vs §06 mean amp | −0.15 |
| §06 canonical β vs §06 peak amp | −0.56 |

The §02 STD and §06 canonical-β methods are **largely uncorrelated** in
their per-channel rankings (ρ = 0.19), measuring different aspects of the
response: STD = amplitude variability across the whole 32-s epoch;
canonical β = HRF-shape match. **Both surface FDR-significant channels in
HbO** (4 for §02, 2 for §06), and the channel sets partially overlap
(`S5_D5` is FDR-sig in §02 and raw-sig in §06; `S4_D5` is FDR-sig in §06
and raw-sig in §02). They should be reported as **complementary**.

## 5. Discussion / Interpretation

1. **HbO + canonical HRF reaches FDR significance.** Combining HbO with a
   temporal-feature analysis produces the strongest HC-vs-GAD inference on
   this cohort: 2 / 23 FDR-significant channels (`S4_D5`, `S4_D7`) for the
   canonical β, and 4 / 23 for the §02 STD. This is qualitatively different
   from the original HbT setup (0 / 23 FDR for both metrics).
2. **Group difference is concentrated in the response *tail*.** 8 of 13
   sig clusters are in the 21–32 s window. HC subjects ramp HbO up across
   the task block; GAD subjects do not sustain. This is biologically
   meaningful and consistent with prefrontal-disengagement narratives in
   GAD.
3. **§02 STD and §06 β tag overlapping but distinct channel clusters.**
   The §02 FDR cluster (`S5_D5`, `S2_D1`, `S3_D3`, `S1_D1`,
   superior-medial) and the §06 FDR cluster (`S4_D5`, `S4_D7`,
   middle/lateral) are spatially adjacent on the 5×7 grid — together they
   span a connected ROI that any GNN node-attention map should be expected
   to weight. Combining the two methods in the paper makes a stronger
   anatomical claim than either alone.
4. **Modelling implication.** The temporal-feature β maps from §06 are
   strong candidates as **alternative GNN node features** (instead of the
   STD scalar currently used by the ST-GNN). A direct benchmark would be
   a nice add to the paper.

## 6. Caveats

1. **Not a true first-level GLM.** No per-stimulus event timing → no
   Go-vs-No-Go contrast. β reflects "how HRF-shaped is the average
   response", not parametric event-locked amplitude.
2. **Block-averaged trajectories** discard within-subject trial-to-trial
   variance; a follow-up with mixed-effects on trial-level data is in
   `FUTURE_ANALYSES.md` Tier 2.
3. **Canonical HRF assumption.** Glover-style; sensitivity to HRF
   parameterization not assessed.
4. **Demographic confound (age) not formally adjusted here.** §05 covers
   it for the §02 STD metric; an analogous ANCOVA on the canonical β is
   on the to-do list (§ Tier 2 in `FUTURE_ANALYSES.md`).

## 7. Generated artefacts

| File | Type | Contents |
|------|------|----------|
| `06_glm_hrf_first_level.ipynb` | Notebook | Reproducible analysis |
| `results_canonical_hrf_beta.csv` | CSV | 23 channels: HC/GAD β, *d*, *U*, *p*_raw, *p*_FDR |
| `results_temporal_features.csv` | CSV | 4 features × 23 channels |
| `results_cluster_permutation.csv` | CSV | Per-channel cluster windows + cluster *p* |
| `block_avg_hc.npy` | NumPy | (33, 23, 326) HC group block-averaged HbO trajectories |
| `block_avg_gad.npy` | NumPy | (29, 23, 326) GAD group block-averaged HbO trajectories |
| `t_axis.npy` | NumPy | (326,) time vector in seconds |
| `fig_hrf_design.png` | Figure | Canonical HRF + boxcar-convolved design |
| `fig_evoked_top_channels.png` | Figure | HC vs GAD evoked HbO (top 8 by canonical-β), with cluster shading |
| `fig_topo_compare_metrics.png` | Figure | 3-panel topo: STD *d* / canonical-β *d* / mean-amp *d* |
