## Table 4. Channel-ablation Δ-F1 vs K=23 baseline (ST × LOSO, 24 cells)

Each cell shows the held-out test F1 with the Δ vs the same (chromo × mt) K=23 baseline in parentheses.
Bold = best non-baseline K within row. ★ = highest mean Δ-F1 across configurations.
† HbT mt=4 K=23 uses the 2026-05-14 anomaly-check re-run (canonical 20260509 baseline reproduced bit-identically); the 20260513 in-batch value of 0.8413 was a non-deterministic outlier — see `CHANNEL_ABLATION_RESULTS.md §6.2`.

| Chromo | mt | K=23 (baseline) | K=16 | K=12 | K=8 |
|---|---|---:|---:|---:|---:|
| HBO | 2 | 0.8169 | 0.8143 (-0.26 pp) | **0.8529 (+3.60 pp)** | 0.8406 (+2.37 pp) |
| HBO | 4 | 0.8014 | 0.8339 (+3.26 pp) | 0.8056 (+0.42 pp) | **0.8672 (+6.58 pp)** |
| HBR | 2 | 0.8406 | 0.8507 (+1.02 pp) | **0.8593 (+1.87 pp)** | 0.8593 (+1.87 pp) |
| HBR | 4 | 0.8333 | **0.8444 (+1.11 pp)** | 0.8104 (-2.29 pp) | 0.7703 (-6.30 pp) |
| HBT | 2 | 0.8112 | 0.8286 (+1.74 pp) | **0.8657 (+5.45 pp)** | 0.8406 (+2.94 pp) |
| HBT | 4 | 0.7860 † | 0.8284 (+4.24 pp) | **0.8464 (+6.05 pp)** | 0.7887 (+0.28 pp) |
| **Mean Δ-F1** | | — | **+1.85 pp** | **+2.52 pp** ★ | **+1.29 pp** |
| **Sign-consistency** | | — | 5/6 | 5/6 | 5/6 |

**Reading the K-consistency claim.** K=12 has the highest mean Δ-F1 (+2.52 pp) of any tested K, and matches the other K values on sign-consistency (5/6 cells positive across the 3 chromophores × 2 trial-cap regimes). HbR mt=4 is the single anti-parsimony cell for K=12; all other (chromo × mt) configurations show positive Δ-F1 at K=12. The locked paper headline (ST × HbO × LOSO × mt=2 × K=12 = F1 0.8529) is the highest-F1 K=12 cell at the regime where the differential XAI was derived.

**Source:** `research/experiments/20260513/CHANNEL_ABLATION_RESULTS.md` (full breakdown).
