# Graph XAI notebooks

Driver notebooks for the Graph Explainability pipeline. Reusable building
blocks live in [`src/xai/`](../../xai/); these notebooks are thin
wrappers that handle paths, run the matrix sweeps, and produce the
SPEC §7.3 deliverables for the paper.

Reference: [`docs/SPEC_xai_graph.md`](../../../docs/SPEC_xai_graph.md) (rev. 5).

## Run order

| # | Notebook | Purpose | Typical runtime |
|---|---|---|---|
| 0 | [`00_setup_and_smoke.ipynb`](00_setup_and_smoke.ipynb) | Quickstart — imports, channel sanity, one fold of SG and ST end-to-end with all figures. | 2–5 min |
| 1 | [`01_sg_population.ipynb`](01_sg_population.ipynb) | SG full sweep: 3 regimes × 2 mt × 3 estimators = 18 PopulationResults. | 2–4 h total |
| 2 | [`02_st_population.ipynb`](02_st_population.ipynb) | ST full sweep: 3 regimes × 2 mt × {native, supplementary} = 12 PopulationResults. | 30–60 min |
| 3 | [`03_cross_arch_comparison.ipynb`](03_cross_arch_comparison.ipynb) | Loads saved CSVs from 01/02; computes SPEC §11 C3/C4/C5/C6 acceptance; renders cross-arch scatter + pair-matrix-diff. | 1–2 min |
| 4 | [`04_atlas_registration.ipynb`](04_atlas_registration.ipynb) *(rev. 5)* | Builds `channel_to_brodmann.csv` once via MNE/fsaverage Procrustes; re-aggregates every 01/02 cell at the Brodmann-region level; runs SPEC §11 C8 acceptance. **Replaces** the hand-crafted VMPFC/DMPFC/DLPFC table as the single source of truth for region attribution. | 1–2 min (after one-time fsaverage download) |

01 and 02 are independent — run them in parallel if you have the GPU
budget. 03 and 04 both consume the saved CSVs from 01/02; 04 only needs
01/02 outputs that already exist on disk (it autodiscovers cells), so it
can be re-run any time the channel-level XAI is updated.

## Kernel requirements

Use the project's existing Python environment. Confirmed working with:

```
python              3.12.12
torch               2.6.0+cu124
torch_geometric     2.7.0
captum              0.9.0
matplotlib, scipy, numpy, pandas (project versions)
```

Captum is required for the SG `captum_ig` cross-check estimator. Install
with `pip install captum` if missing.

## Path conventions

Each notebook autodetects `PROJECT_ROOT` from `Path(os.getcwd()).parents[2]`,
so notebooks must stay at `src/notebook/xai/<name>.ipynb`. The notebooks
read:

- `data/processed-new-mc/` — the dataset both SG and ST were trained on
  (rev. 4 unification; see SPEC §10.4).
- `data/splits/kfold_splits_processed_new_mc.json` — k-fold splits (LOSO
  splits are derived in-code, not from JSON).
- `research/experiments/20260506/leak-free-patience-9999/spatial-graph/{kfold,loso}/...`
  — SG checkpoints.
- `research/experiments/20260501/spatial_temporal_graph/{5-fold,10-fold,loso}/...`
  — ST checkpoints.

The HbO mt4 SG LOSO checkpoint records a `/root/remote-training-setup/`
path (cloud-trained); `src/xai/checkpoints.py:_resolve_data_dir`
auto-rebases it to `<project_root>/data/processed-new-mc/` and logs the
rewrite into `LoadedCheckpoint.path_rebases`.

## Output layout

```
research/xai/
├── _smoke/                           ← 00_setup_and_smoke (delete after)
│   ├── sg_fold1/  st_fold1/  cross_arch/
│
├── sg/                               ← 01_sg_population
│   └── {kfold-5, kfold-10, loso}/mt{2, 4}/{gnn, captum_ig, attention}/
│       ├── node_importance.csv
│       ├── edge_importance.csv
│       ├── channel_pair_matrix.npy + _std.npy
│       ├── feature_importance.csv         (gnn / captum_ig only)
│       ├── result_meta.json
│       ├── run.json
│       └── fig_montage_channel_importance.{png,svg}
│           fig_pair_matrix.{png,svg}
│
├── st/                               ← 02_st_population
│   └── {kfold-5, kfold-10, loso}/mt{2, 4}/{native, supplementary}/
│       ├── (same as SG, plus:)
│       ├── temporal_attention.csv          (native only)
│       └── fig_temporal_attention.{png,svg}  (native only)
│
├── cross_arch/                       ← 03_cross_arch_comparison
│   ├── {regime}_mt{N}/fig_sg_vs_st_scatter.{png,svg}
│   │                  fig_pair_matrix_diff.{png,svg}
│   ├── cross_arch_comparison.csv
│   ├── cross_arch_comparison.md
│   └── cross_arch_pair_diffs.npy + .keys.json
│
└── atlas/                            ← 04_atlas_registration  (rev. 5)
    ├── channel_to_brodmann.csv          ← single source of truth (long)
    ├── channel_midpoints_mni.csv        ← projection diagnostics
    ├── registration_run.json            ← ELC sha256, fsaverage version, fiducial RMSE, 4×4 trans
    ├── fig_montage_brodmann.{png,svg}   ← 5×7 grid coloured by primary BA
    ├── fig_surface_atlas.{png,svg}      ← 3D fsaverage scatter of 23 channels
    └── {sg, st}/{regime}/mt{N}/{sub}/
        ├── region_importance.csv        ← BA × hemi × {mean, n_channels_contrib, p_mass_total}
        ├── region_pair_matrix.npy       ← N_BA × N_BA symmetric
        ├── region_keys.csv              ← row/col ordering for the matrix
        ├── fig_region_bar.{png,svg}
        └── fig_region_pair_heatmap.{png,svg}
```

## SPEC §11 acceptance — where each criterion is evaluated

| ID | What | Threshold | Computed in |
|---|---|---|---|
| C1 | Reload reproduces stored predictions | F1 within ±0.005 | `tests/xai/test_phase_a_foundation.py` (CI) |
| C2 | Across-fold stability | ρ ≥ 0.4 | **Not yet computed** — `run_sg` / `run_st` aggregate folds; needs per-fold output split (future enhancement) |
| C3 | mt=2 vs mt=4 stability | ρ ≥ 0.5 | summary cells of 01, 02; aggregated in 03 |
| C4 | SG 3-way (gnn / captum_ig / attention) | ρ ≥ 0.4 between any two | summary cell of 01; aggregated in 03 |
| C5 | ST native vs supplementary | ρ ≥ 0.4 | summary cell of 02; aggregated in 03 |
| C6 | Biological-prior overlap (advisory) | ≥ 2 of `{S1_D1, S5_D5, S3_D3, S2_D1, S4_D5, S4_D7}` in top-10 | 03 |
| C8 | Atlas registration sanity *(rev. 5)* | parser / projection ≤ 35 mm / S2_D1 → BA10 ≥ 0.5 | summary cell of 04; also `tests/xai/test_atlas_registration.py` |

## Tweakables

- **SG estimator subset.** `01_sg_population.ipynb`'s `run_sg_cell(...)`
  takes an `estimators=` tuple. Shorten to e.g. `("gnn",)` to skip the
  cross-checks and run only the primary path.
- **ST supplementary path.** `run_st_cell(..., run_supplementary=False)`
  in `02_st_population.ipynb` to skip the GNN object-mask pass — the
  longest part of the ST sweep.
- **GNNExplainer epochs.** `gnn_epochs=200` is the SPEC default.
  `gnn_epochs=50` cuts runtime ≈ 4× with mildly noisier masks; use for
  quick re-runs while iterating.
- **Cross-arch primary.** `03_cross_arch_comparison.ipynb` exposes
  `SG_PRIMARY_ESTIMATOR` and `ST_PRIMARY_PATH` near the top — change to
  compare e.g. `attention` vs `supplementary` instead of `gnn` vs `native`.

## Future work

Triggered after the HbO pipeline is validated and accepted (rev. 4 §2.2):

- **HbR / HbT extension.** Rerun 01/02/03 with `cfg.hb='hbr'` and
  `cfg.hb='hbt'`. Same checkpoint roots; same notebook structure.
- **MNE / AtlasViewer Brodmann labelling.** Map the 5×7 montage to MNI
  coordinates and label Brodmann areas. Separate downstream SPEC.
- **C2 acceptance.** Add a per-fold output split to `run_sg` / `run_st`
  so the aggregator can compute across-fold ρ.
