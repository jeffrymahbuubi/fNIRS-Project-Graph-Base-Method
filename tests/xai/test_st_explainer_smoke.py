"""Smoke test for the ST explainer pipeline (SPEC §6 + §7).

Mirrors the SG smoke test: one fold, shape assertions, CSV write/read
round-trip. ST native attention is non-stochastic (deterministic eval-mode
forward), so this also touches a few cheap numerical invariants — temporal
attention sums to ~1, channel importance equals row-sum of the
symmetrised pair matrix.

Run with:
    pytest tests/xai/test_st_explainer_smoke.py -v
Skipped automatically when ST checkpoints / dataset / splits are absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 20260509 layout: kfold lives under st-kfold/<n>-fold/<date>/, LOSO at <root>/loso/.
ST_EXPERIMENT_ROOT = PROJECT_ROOT / "research/experiments/20260509"
ST_KFOLD5_SUBDIR = "st-kfold/5-fold/20260509"
ST_DATA_DIR = PROJECT_ROOT / "data/processed-new-mc"
ST_SPLITS_JSON = PROJECT_ROOT / "data/splits/kfold_splits_processed_new_mc.json"


_ST_PRESENT = (
    (ST_EXPERIMENT_ROOT / ST_KFOLD5_SUBDIR).is_dir()
    and (ST_EXPERIMENT_ROOT / "loso").is_dir()
    and ST_DATA_DIR.is_dir()
    and ST_SPLITS_JSON.is_file()
)


@pytest.mark.skipif(not _ST_PRESENT, reason="ST checkpoints / dataset / splits missing on this machine")
def test_st_one_fold_produces_well_shaped_population_result(tmp_path: Path) -> None:
    """Fold 1 of ST kfold-5 mt2 hbo via native attention extraction.

    Asserts:
      - per-trial channel_importance / pair_matrix / temporal_attention shapes
      - temporal_attention is a softmax (sums to ~1.0)
      - channel_importance == row-sum of symmetric pair_matrix (per SPEC §6.1)
      - PopulationResult shapes; pair_matrix symmetric
      - to_csv writes the §7.3 ST deliverables (temporal_attention.csv yes,
        feature_importance.csv no)
      - from_csv round-trips
    """
    import torch
    from src.xai import (
        XAIRunConfig,
        aggregate_population,
        discover_checkpoints,
        load_checkpoint,
        explain_st_checkpoint,
        PopulationResult,
        N_CH,
    )

    cfg = XAIRunConfig(
        arch="st", hb="hbo", regime="kfold-5", mt=2,
        experiment_root=str(ST_EXPERIMENT_ROOT),
        output_dir=str(tmp_path),
        device=("cuda:0" if torch.cuda.is_available() else "cpu"),
        head_reduce="mean",
        layer_reduce="mean",
        seed=42,
        experiment_subdir=ST_KFOLD5_SUBDIR,
    )

    infos = discover_checkpoints(
        ST_EXPERIMENT_ROOT, arch="st", hb="hbo", regime="kfold-5", mt=2,
        subdir_override=ST_KFOLD5_SUBDIR,
    )
    fold1 = next(i for i in infos if i.fold == 1)
    loaded = load_checkpoint(fold1, cfg)

    trial_atts = explain_st_checkpoint(loaded, cfg)

    assert len(trial_atts) > 0, "fold 1 produced no trial attributions"

    K_first = trial_atts[0].temporal_attention.shape[0]
    for att in trial_atts:
        assert att.channel_importance.shape == (N_CH,)
        assert att.pair_matrix.shape == (N_CH, N_CH)
        assert att.feature_importance is None, "ST does not produce feature_importance"
        assert att.temporal_attention is not None
        assert att.temporal_attention.shape == (K_first,), (
            f"window count drifted across trials: {att.temporal_attention.shape}"
        )
        assert att.window_times is not None
        assert att.window_times.shape == (K_first, 2)

        # temporal_attention is a softmax over windows.
        np.testing.assert_allclose(att.temporal_attention.sum(), 1.0, atol=1e-4)

        # SG explainer symmetrises per-trial; ST does too.
        np.testing.assert_allclose(att.pair_matrix, att.pair_matrix.T, atol=1e-6)
        # SPEC §6.1: channel_importance is the row-sum of the symmetric pair matrix.
        np.testing.assert_allclose(
            att.channel_importance, att.pair_matrix.sum(axis=1), atol=1e-5,
        )

        # Window times monotonic and consistent with stride/size.
        starts = att.window_times[:, 0]
        ends = att.window_times[:, 1]
        assert (ends > starts).all()
        if K_first > 1:
            assert (np.diff(starts) > 0).all()

        assert att.included == (att.pred_label == att.true_label)

    if not any(att.included for att in trial_atts):
        pytest.skip("fold 1 had no correctly-classified val trials")

    result = aggregate_population(
        trial_atts,
        arch="st", hb="hbo", regime="kfold-5", mt=2,
        only_included=True,
    )

    assert result.channel_importance_mean.shape == (N_CH,)
    assert result.pair_matrix.shape == (N_CH, N_CH)
    np.testing.assert_allclose(result.pair_matrix, result.pair_matrix.T, atol=1e-6)
    assert result.feature_importance_mean is None
    assert result.temporal_attention_mean is not None
    assert result.temporal_attention_mean.shape == (K_first,)
    # The mean of softmaxes is itself a probability distribution (sums to 1).
    np.testing.assert_allclose(result.temporal_attention_mean.sum(), 1.0, atol=1e-4)
    assert result.window_times is not None
    assert result.window_times.shape == (K_first, 2)

    assert result.n_trials > 0
    assert result.n_subjects > 0
    assert 0 < result.included_pct <= 100.0

    # Persistence — ST writes temporal_attention.csv, NOT feature_importance.csv.
    result.to_csv(tmp_path)
    assert (tmp_path / "node_importance.csv").is_file()
    assert (tmp_path / "edge_importance.csv").is_file()
    assert (tmp_path / "channel_pair_matrix.npy").is_file()
    assert (tmp_path / "temporal_attention.csv").is_file()
    assert not (tmp_path / "feature_importance.csv").exists(), "feature CSV is SG-only"
    assert (tmp_path / "result_meta.json").is_file()

    reloaded = PopulationResult.from_csv(tmp_path)
    np.testing.assert_allclose(
        reloaded.channel_importance_mean, result.channel_importance_mean, rtol=1e-5,
    )
    np.testing.assert_allclose(reloaded.pair_matrix, result.pair_matrix, rtol=1e-5)
    np.testing.assert_allclose(
        reloaded.temporal_attention_mean, result.temporal_attention_mean, rtol=1e-5,
    )
    assert reloaded.feature_importance_mean is None
    assert reloaded.temporal_attention_mean is not None


@pytest.mark.skipif(not _ST_PRESENT, reason="ST checkpoints / dataset / splits missing on this machine")
def test_st_supplementary_gnn_object_one_fold(tmp_path: Path) -> None:
    """SPEC §6.4 supplementary path — GNNExplainer with object masks on ST.

    Asserts the [23] node_mask shape (not [23, 326]), pair_matrix symmetry,
    and that no temporal_attention / feature_importance leak through.
    """
    import torch
    from src.xai import (
        XAIRunConfig,
        aggregate_population,
        discover_checkpoints,
        load_checkpoint,
        explain_st_supplementary_checkpoint,
        N_CH,
    )

    cfg = XAIRunConfig(
        arch="st", hb="hbo", regime="kfold-5", mt=2,
        experiment_root=str(ST_EXPERIMENT_ROOT),
        output_dir=str(tmp_path),
        device=("cuda:0" if torch.cuda.is_available() else "cpu"),
        gnn_explainer_epochs=20,    # smoke speed
        gnn_explainer_lr=0.01,
        run_supplementary_gnnexplainer=True,
        seed=42,
        experiment_subdir=ST_KFOLD5_SUBDIR,
    )

    infos = discover_checkpoints(
        ST_EXPERIMENT_ROOT, arch="st", hb="hbo", regime="kfold-5", mt=2,
        subdir_override=ST_KFOLD5_SUBDIR,
    )
    fold1 = next(i for i in infos if i.fold == 1)
    loaded = load_checkpoint(fold1, cfg)

    trial_atts = explain_st_supplementary_checkpoint(loaded, cfg)
    assert len(trial_atts) > 0

    for att in trial_atts:
        assert att.channel_importance.shape == (N_CH,)
        assert att.pair_matrix.shape == (N_CH, N_CH)
        np.testing.assert_allclose(att.pair_matrix, att.pair_matrix.T, atol=1e-6)
        assert att.feature_importance is None, "object mask has no per-feature breakdown"
        assert att.temporal_attention is None, "supplementary path is ST-shape but channel-only"
        assert att.window_times is None
        assert (att.channel_importance >= 0).all()

    if not any(att.included for att in trial_atts):
        pytest.skip("fold 1 had no correctly-classified val trials")

    result = aggregate_population(
        trial_atts, arch="st", hb="hbo", regime="kfold-5", mt=2, only_included=True,
    )
    assert result.channel_importance_mean.shape == (N_CH,)
    assert result.feature_importance_mean is None
    assert result.temporal_attention_mean is None
