"""Phase A foundation tests for `src/xai/` (SPEC §11 C1, plus cheap sanity).

Layout:
- test_imports / test_channel_layout / test_config_validation : cheap, no I/O.
- test_discover_*                                            : walk experiment dirs only.
- test_c1_sg_fold1_reproduces_pkl_predictions                : the SPEC §11 C1
  acceptance criterion. Reloads SG fold 1 (kfold-5 mt2 HbO), re-runs the val
  loader with the leak-free transform, and asserts the recomputed
  predictions exactly match what the .pkl stored at training time. F1 is
  then guaranteed to match the stored `f1_score` to floating-point precision.

Run only the cheap tests:
    pytest tests/xai/test_phase_a_foundation.py -k "not c1"
Run everything (needs trained checkpoints + dataset on disk):
    pytest tests/xai/test_phase_a_foundation.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SG_EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "research/experiments/20260506/leak-free-patience-9999/spatial-graph"
)
SG_DATA_DIR = PROJECT_ROOT / "data/processed-new-mc"
SG_SPLITS_JSON = PROJECT_ROOT / "data/splits/kfold_splits_processed_new_mc.json"


# ---------------------------------------------------------------------------
# Cheap tests (no model load, no dataset build)
# ---------------------------------------------------------------------------


def test_imports_clean() -> None:
    """The package surface imports without side-effects."""
    from src.xai import (  # noqa: F401
        CHANNEL_NAMES, GRID_POS, GRID_SHAPE, N_CH, CH_TO_IDX, IDX_TO_GRID,
        XAIRunConfig, CheckpointInfo, LoadedCheckpoint,
        discover_checkpoints, load_checkpoint,
    )


def test_channel_layout_invariants() -> None:
    from src.xai import CHANNEL_NAMES, GRID_POS, GRID_SHAPE, N_CH, CH_TO_IDX

    assert N_CH == 23
    assert len(CHANNEL_NAMES) == 23
    assert len(GRID_POS) == 23
    assert len(set(CHANNEL_NAMES)) == 23, "channel names must be unique"
    assert len(set(GRID_POS)) == 23, "grid positions must be unique"
    rmax = GRID_SHAPE[0] - 1
    cmax = GRID_SHAPE[1] - 1
    for r, c in GRID_POS:
        assert 0 <= r <= rmax and 0 <= c <= cmax, f"grid pos ({r},{c}) outside {GRID_SHAPE}"
    # CH_TO_IDX is the inverse of CHANNEL_NAMES.
    assert all(CH_TO_IDX[CHANNEL_NAMES[i]] == i for i in range(N_CH))


def test_config_validates_arch_hb_regime_mt() -> None:
    from src.xai import XAIRunConfig

    base = dict(
        arch="sg", hb="hbo", regime="kfold-5", mt=2,
        experiment_root="/tmp/exp", output_dir="/tmp/out",
    )
    XAIRunConfig(**base)  # baseline ok

    with pytest.raises(ValueError, match="arch"):
        XAIRunConfig(**{**base, "arch": "bogus"})
    with pytest.raises(ValueError, match="hb"):
        XAIRunConfig(**{**base, "hb": "hbX"})
    with pytest.raises(ValueError, match="regime"):
        XAIRunConfig(**{**base, "regime": "kfold-3"})
    with pytest.raises(ValueError, match="mt"):
        XAIRunConfig(**{**base, "mt": 3})
    with pytest.raises(ValueError, match="head_reduce"):
        XAIRunConfig(**{**base, "head_reduce": "median"})
    with pytest.raises(ValueError, match="layer_reduce"):
        XAIRunConfig(**{**base, "layer_reduce": "first"})
    with pytest.raises(ValueError, match="gnn_explainer_epochs"):
        XAIRunConfig(**{**base, "gnn_explainer_epochs": 0})


def test_resolve_data_dir_local_passthrough(tmp_path: Path) -> None:
    """Existing local path is returned unchanged with no rebase log."""
    from src.xai.checkpoints import _resolve_data_dir

    p, log = _resolve_data_dir(str(tmp_path), tmp_path)
    assert p == tmp_path
    assert log is None


def test_resolve_data_dir_cloud_prefix_rebases(tmp_path: Path) -> None:
    """SPEC §10.4 rebase rule: known cloud prefix → project_root + tail."""
    from src.xai.checkpoints import _resolve_data_dir

    project_root = tmp_path
    real = project_root / "data" / "processed-new-mc"
    real.mkdir(parents=True)
    raw = "/root/remote-training-setup/data/processed-new-mc"

    p, log = _resolve_data_dir(raw, project_root)
    assert p == real
    assert log is not None
    assert log["resolved"] == str(real)
    assert "cloud_prefix" in log["reason"]


def test_resolve_data_dir_basename_fallback(tmp_path: Path) -> None:
    """Unknown prefix but matching basename under project_root/data/ → recovered."""
    from src.xai.checkpoints import _resolve_data_dir

    project_root = tmp_path
    real = project_root / "data" / "processed-new-mc"
    real.mkdir(parents=True)
    raw = "/some/weird/place/processed-new-mc"

    p, log = _resolve_data_dir(raw, project_root)
    assert p == real
    assert log is not None
    assert log["reason"] == "basename_fallback"


def test_resolve_data_dir_unrecoverable_raises(tmp_path: Path) -> None:
    from src.xai.checkpoints import _resolve_data_dir

    with pytest.raises(FileNotFoundError):
        _resolve_data_dir("/no/such/place/processed-XYZ", tmp_path)


# ---------------------------------------------------------------------------
# Discovery — walks the experiment dir; needs trained checkpoints to be present
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SG_EXPERIMENT_ROOT.is_dir(),
    reason="SG experiment root not present (training not run on this machine)",
)
def test_discover_sg_kfold5_mt2_hbo_returns_5_folds() -> None:
    from src.xai.checkpoints import discover_checkpoints

    infos = discover_checkpoints(
        SG_EXPERIMENT_ROOT, arch="sg", hb="hbo", regime="kfold-5", mt=2,
    )
    folds = sorted(i.fold for i in infos)
    assert folds == [1, 2, 3, 4, 5], f"expected folds 1..5, got {folds}"
    for info in infos:
        assert info.subject is None
        assert info.config_path.is_file()
        assert info.ckpt_path.is_file()
        assert info.pkl_path.is_file()
        assert info.hb == "hbo" and info.mt == 2 and info.regime == "kfold-5"


@pytest.mark.skipif(
    not SG_EXPERIMENT_ROOT.is_dir(),
    reason="SG experiment root not present",
)
def test_discover_sg_loso_mt2_hbo_returns_subjects() -> None:
    from src.xai.checkpoints import discover_checkpoints

    infos = discover_checkpoints(
        SG_EXPERIMENT_ROOT, arch="sg", hb="hbo", regime="loso", mt=2,
    )
    # Project's reference cohort is 62 subjects; allow >= 30 to keep the test
    # robust to partial experiments while still catching wholesale failures.
    assert len(infos) >= 30, f"expected many LOSO subjects, got {len(infos)}"
    for info in infos:
        assert info.fold is None
        assert info.subject is not None
        assert info.regime == "loso"


# ---------------------------------------------------------------------------
# C1 — SPEC §11 acceptance criterion (slow: builds dataset + reloads model)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (SG_EXPERIMENT_ROOT.is_dir() and SG_DATA_DIR.is_dir() and SG_SPLITS_JSON.is_file()),
    reason="SG checkpoints / dataset / splits missing on this machine",
)
def test_c1_sg_fold1_reproduces_pkl_predictions() -> None:
    """SPEC §11 C1: reloaded model must reproduce the .pkl-stored predictions exactly,
    and the recomputed F1 must match the stored `f1_score` within ±0.005.

    Strategy: replay the val loader the same way training did
    (`shuffle=False`, batch_size from config, leak-free `compute_stats(train_indices)`
    transform), forward through the reloaded model, and compare label-by-label
    to the stored `pred_labels`. If those match exactly, F1 will too.
    """
    import numpy as np
    import torch
    from sklearn.metrics import f1_score as sk_f1
    from torch_geometric.loader import DataLoader

    from src.xai import XAIRunConfig, discover_checkpoints, load_checkpoint
    from src.core.dataset import SubsetWithTransform
    from torch.utils.data import Subset

    cfg = XAIRunConfig(
        arch="sg", hb="hbo", regime="kfold-5", mt=2,
        experiment_root=str(SG_EXPERIMENT_ROOT),
        output_dir=str(PROJECT_ROOT / "research/xai/_c1_smoke"),
        device=("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    infos = discover_checkpoints(
        SG_EXPERIMENT_ROOT, arch="sg", hb="hbo", regime="kfold-5", mt=2,
    )
    fold1 = next(i for i in infos if i.fold == 1)
    loaded = load_checkpoint(fold1, cfg)

    # Training used DataLoader(..., shuffle=False, batch_size=cfg.batch_size).
    # Match that exactly so prediction order aligns with stored pred_labels.
    batch_size = int(loaded.config["batch_size"])
    val_subset = Subset(loaded.dataset, loaded.val_indices)
    val_ds = SubsetWithTransform(val_subset, transform=loaded.val_transform)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device(cfg.device)
    pred_labels: List[int] = []
    true_labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = loaded.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds = logits.argmax(dim=-1).cpu().tolist()
            pred_labels.extend(preds)
            true_labels.extend(batch.y.cpu().tolist())

    stored_pred = np.asarray(loaded.stored_metrics["pred_labels"], dtype=int).tolist()
    stored_true = np.asarray(loaded.stored_metrics["true_labels"], dtype=int).tolist()
    stored_f1 = float(loaded.stored_metrics["f1_score"])

    assert true_labels == stored_true, (
        f"true label order drift — reload {true_labels[:10]}... vs stored {stored_true[:10]}..."
    )
    assert pred_labels == stored_pred, (
        f"prediction drift — reload {pred_labels[:10]}... vs stored {stored_pred[:10]}..."
    )

    recomputed_f1 = float(sk_f1(true_labels, pred_labels, pos_label=1, average="binary"))
    assert abs(recomputed_f1 - stored_f1) <= 0.005, (
        f"F1 drift {recomputed_f1:.6f} vs stored {stored_f1:.6f} (delta {recomputed_f1 - stored_f1:+.6f})"
    )

    # Audit: any path rebase that fired must be flagged in the loaded bundle.
    if "/root/remote-training-setup/" in str(loaded.config.get("data_dir", "")):
        assert any(r["field"] == "data_dir" for r in loaded.path_rebases)
