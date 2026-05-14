#!/usr/bin/env python
"""Build class-differential XAI pair matrices for the channel-reduction notebook.

For each cell {st_hbo, st_hbr, sg_hbo} at LOSO mt2, re-runs `explain_checkpoint`
to obtain `List[TrialAttribution]`, filters to correctly-classified trials,
partitions by `true_label` (0=GAD, 1=HC), and produces per-class + differential
pair matrices using the same subject-equal-weighting as `aggregate_population`.

Outputs land in `research/xai/channel_reduction/{cell_id}/` and are consumed by
`src/notebook/xai/05_channel_reduction.ipynb`.

This script makes no changes to `src/xai/`; it composes the existing public API.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai import (
    CHANNEL_NAMES,
    XAIRunConfig,
    discover_checkpoints,
    load_checkpoint,
)
from src.xai.aggregate import TrialAttribution, aggregate_population
from src.xai.sg_explainer import explain_checkpoint as explain_sg_checkpoint
from src.xai.st_explainer import explain_checkpoint as explain_st_checkpoint


# Label convention from src/core/dataset.py + src/core_st/dataset.py:
#   `LABEL_MAP = {"healthy": 0, "anxiety": 1}` → label 0 = HC, label 1 = GAD.
# (Prior project-memory note claiming label=0=anxiety was incorrect.)
HC_LABEL = 0
GAD_LABEL = 1


CELLS = {
    "st_hbo": {
        "arch": "st",
        "hb": "hbo",
        "regime": "loso",
        "mt": 2,
        "experiment_root": str(PROJECT_ROOT / "research/experiments/20260509"),
    },
    "st_hbr": {
        "arch": "st",
        "hb": "hbr",
        "regime": "loso",
        "mt": 2,
        "experiment_root": str(PROJECT_ROOT / "research/experiments/20260509"),
    },
    "sg_hbo": {
        "arch": "sg",
        "hb": "hbo",
        "regime": "loso",
        "mt": 2,
        "experiment_root": str(
            PROJECT_ROOT
            / "research/experiments/20260506/leak-free-patience-9999/spatial-graph"
        ),
    },
}


def build_cfg(cell_id: str, out_dir: Path, device: str) -> XAIRunConfig:
    spec = CELLS[cell_id]
    return XAIRunConfig(
        arch=spec["arch"],
        hb=spec["hb"],
        regime=spec["regime"],
        mt=spec["mt"],
        experiment_root=spec["experiment_root"],
        output_dir=str(out_dir),
        device=device,
        include_misclassified=False,
        # SG estimator is fixed to 'gnn' to match existing on-disk run
        # (research/xai/sg/loso/mt2/gnn/run.json). ST native attention does
        # not consult `estimator`, so the value is ignored for ST cells.
        estimator="gnn",
        gnn_explainer_epochs=200,
        head_reduce="mean",
        layer_reduce="mean",
        experiment_subdir=None,  # LOSO under 20260509 uses default
    )


def collect_trials(
    cfg: XAIRunConfig,
    explain_fn: Callable,
) -> tuple[List[TrialAttribution], int]:
    """Mirror run_sg / run_st but stop before aggregation."""
    infos = discover_checkpoints(
        cfg.experiment_root,
        arch=cfg.arch,
        hb=cfg.hb,
        regime=cfg.regime,
        mt=cfg.mt,
        subdir_override=cfg.experiment_subdir,
    )
    if not infos:
        raise FileNotFoundError(
            f"no checkpoints under {cfg.experiment_root!r} matching "
            f"arch={cfg.arch} hb={cfg.hb} regime={cfg.regime} mt={cfg.mt}"
        )

    all_trials: List[TrialAttribution] = []
    for i, info in enumerate(infos, 1):
        loaded = load_checkpoint(info, cfg)
        trials = explain_fn(loaded, cfg)
        all_trials.extend(trials)
        if i % 10 == 0 or i == len(infos):
            n_inc = sum(1 for t in all_trials if t.included)
            print(
                f"  [{i:>3d}/{len(infos)}] {info.label}: "
                f"+{len(trials)} trials ({n_inc} included so far)",
                flush=True,
            )
    return all_trials, len(infos)


def channel_importance_from_pair_matrix(M: np.ndarray) -> np.ndarray:
    """Sum over rows of |M| with diagonal masked (undirected weighted degree).

    Note: We use |M| because for the differential map (GAD − HC) the row-sum of
    the signed matrix would cancel to zero (the matrix is mean-centred between
    classes). Channel importance is "how much does this channel's connectivity
    profile DIFFER between the two classes", which is the absolute-magnitude
    row-sum.
    """
    M2 = M.copy()
    np.fill_diagonal(M2, 0.0)
    return np.abs(M2).sum(axis=1)


def write_channel_csv(
    out_dir: Path,
    *,
    mean_hc: np.ndarray,
    mean_gad: np.ndarray,
    diff: np.ndarray,
    n_hc_trials: int,
    n_gad_trials: int,
    n_hc_subjects: int,
    n_gad_subjects: int,
) -> None:
    """Write per-channel summary CSVs.

    Three files for parity with the existing PopulationResult deliverables:
      - channel_importance_hc.csv  : per-channel mean from HC subset
      - channel_importance_gad.csv : per-channel mean from GAD subset
      - channel_importance_diff.csv: signed and |·| row-sum of GAD−HC pair matrix
    """
    # HC channel-importance (row-sum of HC pair matrix; absolute value of off-
    # diagonal entries → same convention as the existing node_importance.csv
    # in the canonical XAI outputs).
    hc_imp = channel_importance_from_pair_matrix(mean_hc)
    gad_imp = channel_importance_from_pair_matrix(mean_gad)

    pd.DataFrame({
        "channel": CHANNEL_NAMES,
        "mean": hc_imp,
        "n_trials": [n_hc_trials] * len(CHANNEL_NAMES),
        "n_subjects": [n_hc_subjects] * len(CHANNEL_NAMES),
        "rank": (-hc_imp).argsort().argsort() + 1,
    }).to_csv(out_dir / "channel_importance_hc.csv", index=False, float_format="%.8f")

    pd.DataFrame({
        "channel": CHANNEL_NAMES,
        "mean": gad_imp,
        "n_trials": [n_gad_trials] * len(CHANNEL_NAMES),
        "n_subjects": [n_gad_subjects] * len(CHANNEL_NAMES),
        "rank": (-gad_imp).argsort().argsort() + 1,
    }).to_csv(out_dir / "channel_importance_gad.csv", index=False, float_format="%.8f")

    # Differential channel importance — row-sum of |GAD − HC|.
    diff_abs_rowsum = channel_importance_from_pair_matrix(diff)
    # Also signed row-sum (with diag masked) — informative but small in magnitude
    # because |M| sums are conserved per class under softmax row-normalisation.
    diff_signed = diff.copy()
    np.fill_diagonal(diff_signed, 0.0)
    diff_signed_rowsum = diff_signed.sum(axis=1)

    pd.DataFrame({
        "channel": CHANNEL_NAMES,
        "mean_hc": hc_imp,
        "mean_gad": gad_imp,
        "diff_signed": diff_signed_rowsum,
        "diff_abs_mass": diff_abs_rowsum,
        "rank_abs": (-diff_abs_rowsum).argsort().argsort() + 1,
    }).to_csv(out_dir / "channel_importance_diff.csv", index=False, float_format="%.8f")


def run_one_cell(cell_id: str, device: str, out_root: Path) -> dict:
    print(f"\n=== {cell_id} ===", flush=True)
    out_dir = out_root / cell_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(cell_id, out_dir, device=device)
    explain_fn = explain_sg_checkpoint if cfg.arch == "sg" else explain_st_checkpoint

    t0 = datetime.now(timezone.utc)
    all_trials, n_ckpts = collect_trials(cfg, explain_fn)
    elapsed_explainer_s = (datetime.now(timezone.utc) - t0).total_seconds()

    included = [t for t in all_trials if t.included]
    hc = [t for t in included if t.true_label == HC_LABEL]
    gad = [t for t in included if t.true_label == GAD_LABEL]
    n_hc_subjects = len({t.subject_id for t in hc})
    n_gad_subjects = len({t.subject_id for t in gad})

    print(
        f"  → checkpoints={n_ckpts} trials_total={len(all_trials)} "
        f"included={len(included)} ({100 * len(included) / len(all_trials):.1f}%) "
        f"HC={len(hc)} GAD={len(gad)} explainer_s={elapsed_explainer_s:.1f}",
        flush=True,
    )

    if not hc:
        raise RuntimeError(f"{cell_id}: zero HC trials included")
    if not gad:
        raise RuntimeError(f"{cell_id}: zero GAD trials included")

    pop_hc = aggregate_population(
        hc, arch=cfg.arch, hb=cfg.hb, regime=cfg.regime, mt=cfg.mt, only_included=True
    )
    pop_gad = aggregate_population(
        gad, arch=cfg.arch, hb=cfg.hb, regime=cfg.regime, mt=cfg.mt, only_included=True
    )

    M_hc = pop_hc.pair_matrix.astype(np.float64)
    M_gad = pop_gad.pair_matrix.astype(np.float64)
    M_diff = M_gad - M_hc

    np.save(out_dir / "channel_pair_matrix_hc.npy", M_hc.astype(np.float32))
    np.save(out_dir / "channel_pair_matrix_gad.npy", M_gad.astype(np.float32))
    np.save(out_dir / "channel_pair_matrix_diff.npy", M_diff.astype(np.float32))
    # Across-subject std (signal-to-noise audit at the cell level).
    np.save(out_dir / "channel_pair_matrix_hc_std.npy", pop_hc.pair_matrix_std.astype(np.float32))
    np.save(out_dir / "channel_pair_matrix_gad_std.npy", pop_gad.pair_matrix_std.astype(np.float32))

    write_channel_csv(
        out_dir,
        mean_hc=M_hc,
        mean_gad=M_gad,
        diff=M_diff,
        n_hc_trials=pop_hc.n_trials,
        n_gad_trials=pop_gad.n_trials,
        n_hc_subjects=pop_hc.n_subjects,
        n_gad_subjects=pop_gad.n_subjects,
    )

    # Per-trial pair matrices + meta (for future re-aggregation, e.g., bootstrap).
    trial_meta = pd.DataFrame({
        "subject_id": [t.subject_id for t in included],
        "fold_or_subj_label": [t.fold_or_subj_label for t in included],
        "trial_idx_in_subject": [t.trial_idx_in_subject for t in included],
        "true_label": [t.true_label for t in included],
        "pred_label": [t.pred_label for t in included],
        "class": ["HC" if t.true_label == HC_LABEL else "GAD" for t in included],
    })
    trial_meta.to_csv(out_dir / "per_trial_meta.csv", index=False)
    per_trial_matrices = np.stack([t.pair_matrix.astype(np.float32) for t in included])
    np.save(out_dir / "per_trial_pair_matrices.npy", per_trial_matrices)

    # run.json — mirrors the convention used by run_sg / run_st outputs.
    meta = {
        "cell_id": cell_id,
        "arch": cfg.arch,
        "hb": cfg.hb,
        "regime": cfg.regime,
        "mt": cfg.mt,
        "estimator": "native_attention" if cfg.arch == "st" else cfg.estimator,
        "experiment_root": cfg.experiment_root,
        "output_dir": str(out_dir),
        "n_checkpoints": n_ckpts,
        "n_trials_total": len(all_trials),
        "n_trials_included": len(included),
        "included_pct": 100.0 * len(included) / len(all_trials),
        "n_hc_trials": pop_hc.n_trials,
        "n_gad_trials": pop_gad.n_trials,
        "n_hc_subjects": n_hc_subjects,
        "n_gad_subjects": n_gad_subjects,
        "label_convention": {"0": "GAD", "1": "HC"},
        "explainer_elapsed_s": elapsed_explainer_s,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gnn_explainer_epochs": cfg.gnn_explainer_epochs,
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(CELLS.keys()),
        choices=list(CELLS.keys()),
        help="Which cells to build (default: all three).",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument(
        "--out-root",
        default=str(PROJECT_ROOT / "research/xai/channel_reduction"),
        help="Output root directory.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for cell_id in args.cells:
        summary.append(run_one_cell(cell_id, device=args.device, out_root=out_root))

    (out_root / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Wrote {len(summary)} cells to {out_root}")


if __name__ == "__main__":
    main()
