"""Smoke test for the SG explainer pipeline (SPEC §5 + §7).

Scope (per agreed pacing): one fold, fixed seed, shape assertions only.
No numerical reproducibility check — GNNExplainer's stochastic mask init
means re-runs differ in absolute values; SPEC §11 C2/C3/C6 are the
population-level acceptance criteria, evaluated in the notebooks (Phase B),
not here.

Run with:
    pytest tests/xai/test_sg_explainer_smoke.py -v
Skipped automatically when SG checkpoints / dataset / splits are absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SG_EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "research/experiments/20260506/leak-free-patience-9999/spatial-graph"
)
SG_DATA_DIR = PROJECT_ROOT / "data/processed-new-mc"
SG_SPLITS_JSON = PROJECT_ROOT / "data/splits/kfold_splits_processed_new_mc.json"


_SG_PRESENT = SG_EXPERIMENT_ROOT.is_dir() and SG_DATA_DIR.is_dir() and SG_SPLITS_JSON.is_file()


@pytest.mark.skipif(not _SG_PRESENT, reason="SG checkpoints / dataset / splits missing on this machine")
@pytest.mark.parametrize("estimator", ["gnn", "captum_ig", "attention"])
def test_sg_one_fold_produces_well_shaped_population_result(
    tmp_path: Path, estimator: str,
) -> None:
    """Fold 1 of SG kfold-5 mt2 hbo for each of the three SPEC §11 C4 estimators.

    'gnn' / 'captum_ig' produce node_mask → feature_importance is populated.
    'attention' produces edge_mask only → feature_importance is None and
    channel_importance is the row-sum of the symmetric pair matrix.
    """
    import torch
    from src.xai import (
        XAIRunConfig,
        aggregate_population,
        discover_checkpoints,
        load_checkpoint,
        explain_sg_checkpoint,
        PopulationResult,
        N_CH,
    )

    cfg = XAIRunConfig(
        arch="sg", hb="hbo", regime="kfold-5", mt=2,
        experiment_root=str(SG_EXPERIMENT_ROOT),
        output_dir=str(tmp_path),
        device=("cuda:0" if torch.cuda.is_available() else "cpu"),
        gnn_explainer_epochs=20,    # fast smoke; SPEC default is 200
        gnn_explainer_lr=0.01,
        estimator=estimator,
        seed=42,
    )

    infos = discover_checkpoints(
        SG_EXPERIMENT_ROOT, arch="sg", hb="hbo", regime="kfold-5", mt=2,
    )
    fold1 = next(i for i in infos if i.fold == 1)
    loaded = load_checkpoint(fold1, cfg)

    trial_atts = explain_sg_checkpoint(loaded, cfg)

    assert len(trial_atts) > 0, f"{estimator}: fold 1 produced no trial attributions"
    expects_feature = estimator in ("gnn", "captum_ig")
    for att in trial_atts:
        assert att.channel_importance.shape == (N_CH,), att.channel_importance.shape
        assert att.pair_matrix.shape == (N_CH, N_CH), att.pair_matrix.shape
        if expects_feature:
            assert att.feature_importance is not None
            assert att.feature_importance.shape == (6,), att.feature_importance.shape
        else:
            assert att.feature_importance is None
            # AttentionExplainer: channel_importance == row-sum of pair (per
            # _per_trial_reductions when node_mask is None).
            np.testing.assert_allclose(
                att.channel_importance, att.pair_matrix.sum(axis=1), atol=1e-5,
            )
        assert att.temporal_attention is None
        # All SG estimators symmetrise per-trial.
        np.testing.assert_allclose(att.pair_matrix, att.pair_matrix.T, atol=1e-6)
        # GNNExplainer/Captum return non-negative reductions; AttentionExplainer
        # returns softmax probabilities → non-negative.
        assert (att.channel_importance >= 0).all()
        assert att.included == (att.pred_label == att.true_label)

    if not any(att.included for att in trial_atts):
        pytest.skip(
            f"{estimator}: fold 1 had no correctly-classified val trials; aggregation needs at least one"
        )

    result = aggregate_population(
        trial_atts,
        arch="sg", hb="hbo", regime="kfold-5", mt=2,
        only_included=True,
    )

    assert result.channel_importance_mean.shape == (N_CH,)
    assert result.pair_matrix.shape == (N_CH, N_CH)
    np.testing.assert_allclose(result.pair_matrix, result.pair_matrix.T, atol=1e-6)
    if expects_feature:
        assert result.feature_importance_mean is not None
        assert result.feature_importance_mean.shape == (6,)
    else:
        assert result.feature_importance_mean is None
    assert result.temporal_attention_mean is None
    assert result.n_trials > 0
    assert result.n_subjects > 0

    # Persistence — SPEC §7.3 deliverables land on disk for the SG-shape case.
    result.to_csv(tmp_path)
    assert (tmp_path / "node_importance.csv").is_file()
    assert (tmp_path / "edge_importance.csv").is_file()
    assert (tmp_path / "channel_pair_matrix.npy").is_file()
    if expects_feature:
        assert (tmp_path / "feature_importance.csv").is_file()
    else:
        assert not (tmp_path / "feature_importance.csv").exists()
    assert not (tmp_path / "temporal_attention.csv").exists(), "temporal CSV is ST-only"
    assert (tmp_path / "result_meta.json").is_file()

    reloaded = PopulationResult.from_csv(tmp_path)
    np.testing.assert_allclose(
        reloaded.channel_importance_mean, result.channel_importance_mean, rtol=1e-5,
    )
    np.testing.assert_allclose(reloaded.pair_matrix, result.pair_matrix, rtol=1e-5)
    assert reloaded.n_trials == result.n_trials
    assert reloaded.n_subjects == result.n_subjects
    if expects_feature:
        assert reloaded.feature_importance_mean is not None
    else:
        assert reloaded.feature_importance_mean is None
