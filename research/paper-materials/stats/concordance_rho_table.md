# Concordance — XAI saliency vs HbO statistical analysis

**Headline XAI cells (mt2, LOSO, native attention):**
- ST × HbO × LOSO × mt2 (n_trials = 98, 52 subj)
- ST × HbR × LOSO × mt2 (n_trials = 102, 55 subj)

**Statistical references (HbO):**
- §02 brain activation — per-channel STD, MWU, Cohen's d (`02_brain_activation/results_brain_activation_stats.csv`)
- §06 GLM/HRF — per-channel canonical-β, MWU, Cohen's d (`06_glm_hrf/results_canonical_hrf_beta.csv`)

## Spearman ρ over 23 channels

| Comparison | ρ | p | n |
|---|---:|---:|---:|
| ST attn HbO  vs §02 |d| (HbO STD) | +0.002 | 0.993 | 23 |
| ST attn HbO  vs §06 |β d| (HbO canon) | -0.007 | 0.975 | 23 |
| ST attn HbR  vs §02 |d| (HbO STD) | +0.103 | 0.641 | 23 |
| ST attn HbR  vs §06 |β d| (HbO canon) | -0.045 | 0.837 | 23 |
| ST attn HbO  vs ST attn HbR | +0.899 | 0.000 | 23 |
| §02 |d|     vs §06 |β d| | +0.202 | 0.356 | 23 |

## Top-10 inclusion vs C6 biological prior
C6 = {S1_D1, S5_D5, S3_D3, S2_D1, S4_D5, S4_D7} per `docs/SPEC_xai_graph.md §11`.

| Set | Count |
|---|---:|
| ST attn HbO top-10 ∩ C6 | 3 / 6 |
| ST attn HbR top-10 ∩ C6 | 4 / 6 |
| §02 |d| top-10 ∩ C6 | 5 / 6 |
| §06 |β d| top-10 ∩ C6 | 5 / 6 |

## Top-10 ∩ Top-10 cross-method overlap

| Comparison | Overlap |
|---|---:|
| ST attn HbO ∩ §02 |d|  (top-10) | 4 / 10 |
| ST attn HbO ∩ §06 |β d| (top-10) | 6 / 10 |
| ST attn HbR ∩ §02 |d|  (top-10) | 4 / 10 |
| ST attn HbR ∩ §06 |β d| (top-10) | 6 / 10 |

## Notes
- ρ is rank-based; computed via `scipy.stats.spearmanr` on the raw scalar vectors.
- §02 / §06 Cohen's d are signed (HC − GAD; HC > GAD ⇒ d < 0); XAI cannot recover sign, so we compare against |d|.
- C6 inclusion is by membership in the top-10 channel ranking, NOT rank-correlation.
- Generated: 2026-05-11T14:59:28
