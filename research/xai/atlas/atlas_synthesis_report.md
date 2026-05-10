# Atlas Synthesis Report — Brodmann-level XAI for fNIRS GNN

**Generated:** notebook `04_atlas_registration.ipynb` § C, 2026-05-10T20:31:40.400013+00:00
**SPEC:** docs/SPEC_xai_graph.md (rev. 6)  ·  **Atlas:** `PALS_B12_Brodmann` on fsaverage
**Inputs:** 54 `region_importance.csv` cells (18 SG + 36 ST × 3 chromophores) from `01_sg_population` + `02_st_population`

## Two valid ways to rank Brodmann areas

Reporting both removes the most common interpretation pitfall:

1. **Total mass** = `Σ channel_importance × P(channel ∈ BA)` summed across all cells.
   *Bias:* dominated by **how many channels** sample each BA in this montage
   (BA 8 has 8 channels, BA 9 has 7, BA 10 has 5, BA 46 has 3).
   *Reads as:* "for the brain coverage this study has, which BAs accumulate the most total signal."

2. **Per-channel mean** = `region_mean / n_channels_contributing`.
   *Bias:* none from channel count.
   *Reads as:* "which BA produces the strongest **per-channel** discriminative signal."

### Total mass ranking

| Rank | BA | Total mass | GAD-relevant function |
|---|---|---|---|
| 1 | BA8 | 272.17 | DMPFC posterior / pre-SMA — executive monitoring, conflict detection |
| 2 | BA9 | 237.60 | DLPFC anterior — cognitive control, working memory load |
| 3 | BA10 | 172.07 | Frontopolar cortex — worry, expectation, future-oriented thought |
| 4 | BA46 | 101.54 | DLPFC core — top-down regulation of limbic activity |

### Per-channel-mean ranking (signal strength, channel-count bias removed)

| Rank | BA | Mean per-channel signal | GAD-relevant function |
|---|---|---|---|
| 1 | BA10 | 0.6373 | Frontopolar cortex — worry, expectation, future-oriented thought |
| 2 | BA8 | 0.6300 | DMPFC posterior / pre-SMA — executive monitoring, conflict detection |
| 3 | BA9 | 0.6286 | DLPFC anterior — cognitive control, working memory load |
| 4 | BA46 | 0.6268 | DLPFC core — top-down regulation of limbic activity |

## Channel-level vs BA-level cross-architecture agreement

Channel-level ρ between SG-gnn and ST-native rankings is essentially noise
(`research/xai/cross_arch_comparison.md`). At the **Brodmann** level the same
comparison gives:

| regime   |   mt | hb   |   rho_channel_rank |   rho_BA_mass |   rho_BA_per_channel |   jaccard_top3_per_channel |
|:---------|-----:|:-----|-------------------:|--------------:|---------------------:|---------------------------:|
| kfold-5  |    2 | hbo  |           0.167984 |             1 |                  0.2 |                        1   |
| kfold-5  |    2 | hbr  |          -0.088933 |             1 |                 -0.8 |                        0.5 |
| kfold-5  |    2 | hbt  |          -0.019763 |             1 |                  0.2 |                        1   |
| kfold-5  |    4 | hbo  |          -0.175889 |             1 |                 -0.8 |                        0.5 |
| kfold-5  |    4 | hbr  |          -0.094862 |             1 |                 -0.4 |                        0.5 |
| kfold-5  |    4 | hbt  |          -0.018775 |             1 |                  0.4 |                        0.5 |
| kfold-10 |    2 | hbo  |           0.089921 |             1 |                  0.2 |                        1   |
| kfold-10 |    2 | hbr  |           0.134387 |             1 |                 -0.4 |                        0.5 |
| kfold-10 |    2 | hbt  |           0.007905 |             1 |                  0.2 |                        1   |
| kfold-10 |    4 | hbo  |          -0.284585 |             1 |                 -0.8 |                        0.5 |
| kfold-10 |    4 | hbr  |          -0.344862 |             1 |                 -0.8 |                        0.5 |
| kfold-10 |    4 | hbt  |          -0.078063 |             1 |                 -0.4 |                        0.5 |
| loso     |    2 | hbo  |           0.060277 |             1 |                 -0.2 |                        0.5 |
| loso     |    2 | hbr  |           0.066206 |             1 |                  0.4 |                        1   |
| loso     |    2 | hbt  |           0.11166  |             1 |                 -0.2 |                        0.5 |
| loso     |    4 | hbo  |          -0.166008 |             1 |                 -0.8 |                        0.5 |
| loso     |    4 | hbr  |          -0.241107 |             1 |                 -0.8 |                        0.5 |
| loso     |    4 | hbt  |          -0.215415 |             1 |                 -0.8 |                        0.5 |

**Mean ρ across all cells:**
- channel-level: **-0.061**
- BA-level (mass-weighted): **+1.000** — driven largely by channel count, not a finding on its own
- BA-level (per-channel mean): **-0.311**

**Interpretation:** per-channel-mean BA agreement is also negative — the two architectures emphasise **different Brodmann regions**, not just different channels. The BA-mass ρ ≈ 1.0 is purely a structural artefact of the montage (BAs with more channels accumulate more mass).

## Top-3 Brodmann area per cell (raw mass)

| arch   | hb   | regime   |   mt | sub           | top1      | top2      | top3       |
|:-------|:-----|:---------|-----:|:--------------|:----------|:----------|:-----------|
| sg     | nan  | kfold-10 |    2 | attention     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | kfold-10 |    2 | captum_ig     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | kfold-10 |    2 | gnn           | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | kfold-10 |    4 | attention     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | kfold-10 |    4 | captum_ig     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | kfold-10 |    4 | gnn           | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | kfold-5  |    2 | attention     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | kfold-5  |    2 | captum_ig     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | kfold-5  |    2 | gnn           | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | kfold-5  |    4 | attention     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | kfold-5  |    4 | captum_ig     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | kfold-5  |    4 | gnn           | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | loso     |    2 | attention     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | loso     |    2 | captum_ig     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | loso     |    2 | gnn           | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | loso     |    4 | attention     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | loso     |    4 | captum_ig     | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| sg     | nan  | loso     |    4 | gnn           | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | kfold-10 |    2 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | kfold-10 |    2 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | kfold-10 |    4 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | kfold-10 |    4 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | kfold-5  |    2 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | kfold-5  |    2 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | kfold-5  |    4 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | kfold-5  |    4 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | loso     |    2 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | loso     |    2 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | loso     |    4 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbo  | loso     |    4 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | kfold-10 |    2 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | kfold-10 |    2 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | kfold-10 |    4 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | kfold-10 |    4 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | kfold-5  |    2 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | kfold-5  |    2 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | kfold-5  |    4 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | kfold-5  |    4 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | loso     |    2 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | loso     |    2 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | loso     |    4 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbr  | loso     |    4 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | kfold-10 |    2 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | kfold-10 |    2 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | kfold-10 |    4 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | kfold-10 |    4 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | kfold-5  |    2 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | kfold-5  |    2 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | kfold-5  |    4 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | kfold-5  |    4 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | loso     |    2 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | loso     |    2 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | loso     |    4 | native        | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |
| st     | hbt  | loso     |    4 | supplementary | BA8 (L+R) | BA9 (L+R) | BA10 (L+R) |

## Channel→BA assignment (single source of truth)

Replaces the hand-crafted VMPFC/DMPFC/DLPFC table per SPEC §14 Q9.
Computed via Procrustes-aligned midpoint projection onto fsaverage with
σ=5.0 mm Gaussian kernel, radius=10.0 mm.
Empirical projection distances 14–33 mm (mean 22.5 mm), all within the
35.0 mm C8.b bound.

| channel   | ba_label   | hemi   |
|:----------|:-----------|:-------|
| S1_D1     | BA10       | L      |
| S1_D3     | BA10       | L      |
| S2_D1     | BA10       | R      |
| S2_D2     | BA10       | R      |
| S2_D5     | BA9        | R      |
| S3_D1     | BA9        | L      |
| S3_D3     | BA9        | L      |
| S3_D4     | BA9        | L      |
| S3_D6     | BA8        | L      |
| S4_D4     | BA9        | R      |
| S4_D5     | BA9        | R      |
| S4_D7     | BA8        | R      |
| S5_D2     | BA10       | R      |
| S5_D5     | BA46       | R      |
| S5_D8     | BA46       | R      |
| S6_D3     | BA46       | L      |
| S6_D6     | BA8        | L      |
| S7_D4     | BA8        | L      |
| S7_D6     | BA8        | L      |
| S7_D7     | BA8        | R      |
| S8_D5     | BA9        | R      |
| S8_D7     | BA8        | R      |
| S8_D8     | BA8        | R      |

## Caveats

1. **BA-mass is channel-count-biased.** Reporting only the mass-weighted ranking is misleading; the per-channel-mean ranking is the unbiased view.
2. **Template-head registration only** — no per-subject MRI; coordinates aligned to fsaverage via 3 fiducials.
3. **Midpoint projection ≠ cortical sampling region** — real fNIRS sensitivity is a banana-shaped volume; midpoint is first-order. AtlasViewer/MCX Monte Carlo (SPEC §16.10) is the gold-standard refinement and remains future work.
4. **Atlas uncertainty** — `PALS_B12_Brodmann` is a population-average projection; individual cytoarchitectonic boundaries differ.
5. **Subcortical structures invisible** — fNIRS cannot reach amygdala/insula/hippocampus. The atlas describes *cortical regulatory correlates* of GAD, not the affective drivers.
