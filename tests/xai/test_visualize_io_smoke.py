"""Smoke tests for `src/xai/visualize.py` and `src/xai/io.py`.

Builds a synthetic PopulationResult (no model load), calls each plot
function, asserts both PNG and SVG files land at the expected paths.
Then writes run.json and round-trips it.

Fast — ~1s on this machine. Always runs (no checkpoint dependency).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _synthetic_population_result(arch: str, *, K: int = 20):
    from src.xai import PopulationResult, N_CH

    rng = np.random.default_rng(seed=42)
    ch_mean = rng.uniform(0.1, 1.0, N_CH)
    pair = rng.uniform(0.0, 1.0, (N_CH, N_CH))
    pair = (pair + pair.T) / 2.0
    np.fill_diagonal(pair, 0.0)

    feature_mean = rng.uniform(0.0, 1.0, 6) if arch == "sg" else None
    feature_std = (feature_mean * 0.1) if feature_mean is not None else None

    if arch == "st":
        temporal_mean = rng.dirichlet(np.ones(K))      # softmax → sums to 1.0
        temporal_std = rng.uniform(0.0, 0.05, K)
        window_times = np.array(
            [(k * 0.8, k * 0.8 + 1.6) for k in range(K)], dtype=np.float64,
        )
    else:
        temporal_mean = temporal_std = window_times = None

    return PopulationResult(
        arch=arch, hb="hbo", regime="kfold-5", mt=2,
        channel_importance_mean=ch_mean,
        channel_importance_std=ch_mean * 0.1,
        pair_matrix=pair,
        pair_matrix_std=pair * 0.1,
        feature_importance_mean=feature_mean,
        feature_importance_std=feature_std,
        temporal_attention_mean=temporal_mean,
        temporal_attention_std=temporal_std,
        window_times=window_times,
        n_trials=50,
        n_trials_total=60,
        n_subjects=10,
        included_pct=83.33,
        per_subject_trial_counts={f"S{i:02d}": 5 for i in range(10)},
        extras={"estimator": "synthetic", "n_checkpoints": 5},
    )


def _assert_png_svg(out_path: Path) -> None:
    assert out_path.suffix == ".png", out_path
    assert out_path.is_file(), f"missing PNG: {out_path}"
    assert out_path.with_suffix(".svg").is_file(), f"missing SVG: {out_path}"
    assert out_path.stat().st_size > 1024, f"PNG suspiciously small: {out_path}"


# ---------------------------------------------------------------------------
# visualize.py
# ---------------------------------------------------------------------------


def test_plot_montage_channel_importance_writes_png_and_svg(tmp_path: Path) -> None:
    from src.xai import plot_montage_channel_importance
    sg = _synthetic_population_result("sg")
    out = plot_montage_channel_importance(sg, tmp_path)
    _assert_png_svg(out)
    assert out.name == "fig_montage_channel_importance.png"


def test_plot_pair_matrix_writes_png_and_svg(tmp_path: Path) -> None:
    from src.xai import plot_pair_matrix
    sg = _synthetic_population_result("sg")
    out = plot_pair_matrix(sg, tmp_path)
    _assert_png_svg(out)
    assert out.name == "fig_pair_matrix.png"


def test_plot_temporal_attention_st_only(tmp_path: Path) -> None:
    from src.xai import plot_temporal_attention
    st = _synthetic_population_result("st", K=20)
    out = plot_temporal_attention(st, tmp_path)
    _assert_png_svg(out)
    assert out.name == "fig_temporal_attention.png"

    # SG result has no temporal_attention → should refuse.
    sg = _synthetic_population_result("sg")
    with pytest.raises(ValueError, match="temporal_attention"):
        plot_temporal_attention(sg, tmp_path)


def test_plot_sg_vs_st_scatter_writes_png_and_svg(tmp_path: Path) -> None:
    from src.xai import plot_sg_vs_st_scatter
    sg = _synthetic_population_result("sg")
    st = _synthetic_population_result("st")
    out = plot_sg_vs_st_scatter(sg, st, tmp_path)
    _assert_png_svg(out)
    assert out.name == "fig_sg_vs_st_scatter.png"


def test_plot_pair_matrix_diff_writes_png_and_svg(tmp_path: Path) -> None:
    from src.xai import plot_pair_matrix_diff
    sg = _synthetic_population_result("sg")
    st = _synthetic_population_result("st")
    out = plot_pair_matrix_diff(sg, st, tmp_path)
    _assert_png_svg(out)
    assert out.name == "fig_pair_matrix_diff.png"


# ---------------------------------------------------------------------------
# io.py
# ---------------------------------------------------------------------------


def test_write_run_json_round_trip(tmp_path: Path) -> None:
    from src.xai import XAIRunConfig, write_run_json, read_run_json
    sg = _synthetic_population_result("sg")
    cfg = XAIRunConfig(
        arch="sg", hb="hbo", regime="kfold-5", mt=2,
        experiment_root=str(tmp_path / "exp"),
        output_dir=str(tmp_path),
    )
    path = write_run_json(cfg, sg, tmp_path, extra={"called_from": "smoke_test"})
    assert path == tmp_path / "run.json"
    assert path.is_file()

    record = read_run_json(tmp_path)
    assert record["schema_version"] == 1
    assert "timestamp_utc" in record
    # Git probe is best-effort; the field exists even if SHA resolution failed.
    assert "git" in record and "sha" in record["git"]
    versions = record["versions"]
    for k in ("python", "torch", "torch_geometric", "numpy", "pandas"):
        assert k in versions, f"missing version key {k!r}"
    config = record["config"]
    assert config["arch"] == "sg"
    assert config["hb"] == "hbo"
    assert config["regime"] == "kfold-5"
    assert config["mt"] == 2
    result = record["result"]
    assert result["arch"] == "sg"
    assert result["n_trials"] == 50
    assert result["n_subjects"] == 10
    assert record["extras"]["estimator"] == "synthetic"
    assert record["extra"]["called_from"] == "smoke_test"


def test_write_run_json_default_out_dir_uses_cfg_output_dir(tmp_path: Path) -> None:
    from src.xai import XAIRunConfig, write_run_json
    sg = _synthetic_population_result("sg")
    cfg = XAIRunConfig(
        arch="sg", hb="hbo", regime="kfold-5", mt=2,
        experiment_root="/dev/null",
        output_dir=str(tmp_path / "explicit"),
    )
    path = write_run_json(cfg, sg)  # no out_dir → uses cfg.output_dir
    assert path == Path(cfg.output_dir) / "run.json"
    assert path.is_file()
