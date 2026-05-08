"""Tests for src/xai/atlas.py (SPEC §16, rev. 5).

Coverage:
* Pure-Python parser (no MNE) — C8.a + negative tests.
* MNE-dependent end-to-end run (cached fsaverage in $MNE_DATA) — C8.b, C8.c,
  Σ probability == 1, region aggregation conserves mass.

Heavy MNE work is amortised by ``module_mapping`` (one ``build_channel_to_brodmann``
call per test session).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.xai.atlas import (
    DEFAULT_PROJECTION_DISTANCE_BOUND_MM,
    aggregate_to_regions,
    build_channel_to_brodmann,
    compute_channel_midpoints,
    compute_sd_distances,
    parse_elc,
)
from src.xai.channels import CHANNEL_NAMES, N_CH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ELC_PATH = PROJECT_ROOT / "data" / "brainproducts-RNP-BA-128-custom.elc"


# =========================================================================== #
# C8.a — ELC parser (pure-Python, no MNE)                                     #
# =========================================================================== #

def test_parse_elc_returns_16_optodes_and_3_fiducials_C8a():
    elc = parse_elc(ELC_PATH)
    # 19 total positions = 3 fiducials + 16 optodes
    assert elc.positions_mm.shape == (19, 3)
    assert len(elc.labels) == 19
    assert elc.n_optodes == 16
    # Fiducials in the right order
    assert elc.labels[0] == "LPA"
    assert elc.labels[1] in ("Nz", "Nasion")
    assert elc.labels[2] == "RPA"
    # All 8 sources + 8 detectors present
    expected_optodes = {f"S{i}" for i in range(1, 9)} | {f"D{i}" for i in range(1, 9)}
    assert set(elc.labels[3:]) == expected_optodes
    # SHA stable
    assert len(elc.sha256) == 64
    assert all(c in "0123456789abcdef" for c in elc.sha256)


def test_parse_elc_rejects_missing_unit_position(tmp_path):
    bad = tmp_path / "bad.elc"
    bad.write_text(
        "# no UnitPosition\n"
        "NumberPositions=1\n"
        "Positions\n"
        "1.0 2.0 3.0\n"
        "Labels\n"
        "LPA\n"
    )
    with pytest.raises(ValueError, match="UnitPosition"):
        parse_elc(bad)


def test_parse_elc_rejects_wrong_fiducial_order(tmp_path):
    bad = tmp_path / "bad.elc"
    bad.write_text(
        "UnitPosition mm\n"
        "NumberPositions=3\n"
        "Positions\n"
        "1.0 2.0 3.0\n"
        "4.0 5.0 6.0\n"
        "7.0 8.0 9.0\n"
        "Labels\n"
        "RPA\n"        # wrong: must be LPA first
        "LPA\n"
        "Nasion\n"
    )
    with pytest.raises(ValueError, match="LPA"):
        parse_elc(bad)


# =========================================================================== #
# Channel midpoint sanity (pure-Python, used by C8.b downstream)              #
# =========================================================================== #

def test_compute_channel_midpoints_returns_all_23_channels():
    elc = parse_elc(ELC_PATH)
    mids = compute_channel_midpoints(elc)
    assert set(mids.keys()) == set(CHANNEL_NAMES)
    assert all(m.shape == (3,) for m in mids.values())


def test_channel_midpoints_in_prefrontal_region():
    elc = parse_elc(ELC_PATH)
    mids = compute_channel_midpoints(elc)
    arr = np.stack([mids[ch] for ch in CHANNEL_NAMES])
    # Y > 0 → anterior (prefrontal) for ASA head-CTF convention used in this ELC
    assert (arr[:, 1] > 0).all(), "every channel midpoint must be anterior"
    # Bilateral X coverage
    assert arr[:, 0].min() < -10 and arr[:, 0].max() > 10


def test_sd_distances_in_realistic_range():
    elc = parse_elc(ELC_PATH)
    sd = np.array(list(compute_sd_distances(elc).values()))
    # Realistic S–D for adult fNIRS: 25–45 mm; mean expected ~30–35 mm
    assert sd.min() > 20.0 and sd.max() < 50.0
    assert 28.0 < sd.mean() < 40.0


def test_S2_D1_is_at_midline_anterior():
    """S2_D1's midpoint is the canonical midline-anterior probe, used by C8.c."""
    elc = parse_elc(ELC_PATH)
    mid = compute_channel_midpoints(elc)["S2_D1"]
    assert abs(mid[0]) < 1.0, f"S2_D1 must be at midline, got X={mid[0]:.2f}"
    assert mid[1] > 80.0, f"S2_D1 must be anterior pole, got Y={mid[1]:.2f}"


# =========================================================================== #
# MNE-dependent: end-to-end registration + Brodmann query                     #
# =========================================================================== #

@pytest.fixture(scope="module")
def module_mapping(tmp_path_factory):
    """Run the full §16 pipeline once and reuse across tests."""
    pytest.importorskip("mne")
    pytest.importorskip("scipy.spatial")
    out = tmp_path_factory.mktemp("atlas_out")
    mapping = build_channel_to_brodmann(
        elc_path=ELC_PATH,
        output_dir=out,
        # subjects_dir=None → fetch_fsaverage uses the cached download.
    )
    return mapping, out


def test_registration_residual_is_low(module_mapping):
    mapping, _ = module_mapping
    # Three fiducials → exact rigid fit; residual should be sub-millimetre
    # for ASA-style head-CTF aligned to fsaverage.
    # ELC fiducial spacing differs slightly from fsaverage's (BrainProducts head
    # model vs population average); rigid (no-scale) Procrustes leaves a small
    # residual. Empirical 2026-05-08: 4.94 mm.
    assert mapping.registration.fiducial_residual_rmse_mm < 8.0, (
        f"fiducial-fit RMSE = {mapping.registration.fiducial_residual_rmse_mm:.2f} mm "
        "(expected < 8 mm)"
    )


def test_all_midpoints_within_25mm_of_cortex_C8b(module_mapping):
    mapping, _ = module_mapping
    diag = mapping.midpoints_mri_mm
    assert len(diag) == N_CH
    bound = DEFAULT_PROJECTION_DISTANCE_BOUND_MM
    over = diag[diag["projection_distance_mm"] > bound]
    assert over.empty, (
        f"channels with projection_distance_mm > {bound} mm:\n{over.to_string()}"
    )
    # Empirical 2026-05-08 against -custom.elc: mean 22.5 mm, max 32.7 mm.
    assert 15.0 < diag["projection_distance_mm"].mean() < 28.0, (
        f"mean projection_distance_mm = {diag['projection_distance_mm'].mean():.2f} "
        "(expected in [15, 28] mm)"
    )


def test_S2_D1_maps_to_BA10_C8c(module_mapping):
    mapping, _ = module_mapping
    ch2ba = mapping.channel_to_ba_long
    s2d1 = ch2ba[ch2ba["channel"] == "S2_D1"]
    ba10_total = s2d1[s2d1["ba_label"] == "Brodmann.10"]["probability"].sum()
    assert ba10_total >= 0.5, (
        f"S2_D1 → Brodmann.10 probability = {ba10_total:.3f}, expected ≥ 0.5\n"
        f"full distribution:\n{s2d1.to_string(index=False)}"
    )


def test_channel_probabilities_sum_to_one(module_mapping):
    mapping, _ = module_mapping
    sums = mapping.channel_to_ba_long.groupby("channel")["probability"].sum()
    assert set(sums.index) == set(CHANNEL_NAMES)
    np.testing.assert_allclose(sums.values, 1.0, atol=1e-6)


def test_outputs_persisted_with_correct_schema(module_mapping):
    _, out = module_mapping
    ch2ba = pd.read_csv(out / "channel_to_brodmann.csv")
    assert list(ch2ba.columns) == ["channel", "ba_label", "hemi", "probability"]
    diag = pd.read_csv(out / "channel_midpoints_mni.csv")
    assert {
        "channel", "x_mni_mm", "y_mni_mm", "z_mni_mm",
        "projected_vertex_id", "hemi", "projection_distance_mm", "sd_distance_mm",
    }.issubset(set(diag.columns))
    meta = json.loads((out / "registration_run.json").read_text())
    for required in (
        "elc_sha256", "fsaverage_dir", "mne_version", "atlas_parc",
        "sigma_mm", "radius_mm", "fiducial_residual_rmse_mm",
        "trans_head_to_mri_mm", "n_optodes", "n_channels",
    ):
        assert required in meta, f"registration_run.json missing field: {required!r}"
    assert meta["atlas_parc"] == "PALS_B12_Brodmann"
    assert meta["n_optodes"] == 16
    assert meta["n_channels"] == N_CH


# =========================================================================== #
# Region aggregation (§16.7) — conservation invariant                         #
# =========================================================================== #

def test_aggregate_to_regions_conserves_mass(module_mapping, tmp_path):
    """region_imp['mean'].sum() ≈ node_importance['mean'].sum() — mass is
    preserved by the probability split (Σ_BA P(ch∈BA) == 1 per channel)."""
    _, atlas_out = module_mapping
    ch2ba_csv = atlas_out / "channel_to_brodmann.csv"

    # Fabricate a deterministic node_importance.csv + 23×23 pair matrix.
    rng = np.random.default_rng(seed=0)
    node = pd.DataFrame({
        "channel": CHANNEL_NAMES,
        "mean": rng.uniform(0.1, 1.0, size=N_CH).astype(np.float64),
        "std": rng.uniform(0.0, 0.2, size=N_CH).astype(np.float64),
        "n_trials": np.full(N_CH, 88, dtype=np.int64),
        "n_subjects": np.full(N_CH, 49, dtype=np.int64),
        "rank": np.arange(1, N_CH + 1, dtype=np.int64),
    })
    pair = rng.uniform(0, 0.15, size=(N_CH, N_CH)).astype(np.float64)
    pair = (pair + pair.T) / 2.0
    np.fill_diagonal(pair, 0.0)

    cell_dir = tmp_path / "fake_cell"
    cell_dir.mkdir()
    node_csv = cell_dir / "node_importance.csv"
    pair_npy = cell_dir / "channel_pair_matrix.npy"
    node.to_csv(node_csv, index=False, float_format="%.8f")
    np.save(pair_npy, pair)

    out_dir = tmp_path / "region_out"
    region_imp, region_pair, region_keys = aggregate_to_regions(
        node_csv=node_csv, pair_npy=pair_npy,
        channel_to_brodmann_csv=ch2ba_csv, output_dir=out_dir,
    )

    # Mass conservation: sum of weighted region importance == sum of channel importance
    np.testing.assert_allclose(
        region_imp["mean"].sum(), node["mean"].sum(), rtol=1e-9,
        err_msg="region aggregation lost mass relative to channel-level total",
    )

    # region_pair shape and symmetry
    n_ba = len(region_keys)
    assert region_pair.shape == (n_ba, n_ba)
    np.testing.assert_allclose(region_pair, region_pair.T, atol=1e-12)

    # Outputs persisted
    assert (out_dir / "region_importance.csv").exists()
    assert (out_dir / "region_pair_matrix.npy").exists()
    assert (out_dir / "region_keys.csv").exists()


def test_aggregate_to_regions_rejects_wrong_pair_shape(module_mapping, tmp_path):
    _, atlas_out = module_mapping
    bad_pair = tmp_path / "bad_pair.npy"
    np.save(bad_pair, np.zeros((10, 10), dtype=np.float64))
    node_csv = tmp_path / "node.csv"
    pd.DataFrame({"channel": CHANNEL_NAMES, "mean": [0.0] * N_CH}).to_csv(
        node_csv, index=False
    )
    with pytest.raises(ValueError, match="pair_matrix shape"):
        aggregate_to_regions(
            node_csv=node_csv,
            pair_npy=bad_pair,
            channel_to_brodmann_csv=atlas_out / "channel_to_brodmann.csv",
        )
