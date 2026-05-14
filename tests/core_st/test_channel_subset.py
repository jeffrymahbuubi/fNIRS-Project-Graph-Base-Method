"""Unit tests for the `channel_subset` feature in src/core_st/.

Two contracts being verified:

1. **Bit-identity:** `channel_subset=None` and `channel_subset=list(range(23))` must
   produce identical graphs (x, edge_index, edge_attr) when fed the same trial.
   This is the safety net that future refactors must not break — if it fails, any
   K-subset run is no longer comparable to the baseline.

2. **Forward-pass shape correctness with K<23.** The model has no architectural
   dependence on n_channels, but we assert this concretely with a K=12 batch so a
   subtle regression (e.g. someone hard-codes 23 in models.py) is caught.

3. **CLI / YAML name-or-int parsing.** `_resolve_channel_subset` is the only entry
   point a user can hit; verify it tolerates the half-dozen sensible input forms.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.core_st.dataset import fNIRSGraphDataset
from src.core_st.main import _resolve_channel_subset
from src.core_st.models import WindowedSpatioTemporalGATNet


# ---------------------------------------------------------------------------
# Fixture: a synthetic 23-channel trial. Avoids touching the on-disk dataset.
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_trial() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((23, 326)).astype(np.float32)


def _bare_dataset(channel_subset):
    """Construct a fNIRSGraphDataset without running `_load` (avoids needing data on disk)."""
    ds = fNIRSGraphDataset.__new__(fNIRSGraphDataset)
    ds.root = ""
    ds.task_type = "GNG"
    ds.data_type = "hbo"
    ds.max_trials = None
    ds.directed = True
    ds.corr_threshold = 0.1
    ds.self_loops = True
    ds.channel_subset = list(channel_subset) if channel_subset is not None else None
    ds._graphs = []
    return ds


# ---------------------------------------------------------------------------
# 1. Bit-identity: None vs full-range
# ---------------------------------------------------------------------------


def test_none_equals_full_range_identity(synthetic_trial: np.ndarray) -> None:
    ds_none = _bare_dataset(None)
    ds_full = _bare_dataset(list(range(23)))

    g_none = ds_none._build_graph(synthetic_trial.copy(), fs=10.0)
    g_full = ds_full._build_graph(synthetic_trial.copy(), fs=10.0)

    assert torch.equal(g_none.x, g_full.x)
    assert torch.equal(g_none.edge_index, g_full.edge_index)
    assert torch.equal(g_none.edge_attr, g_full.edge_attr)


# ---------------------------------------------------------------------------
# 2. Subset shapes: K=12 graph has K nodes, edges remapped to 0..K-1
# ---------------------------------------------------------------------------


def test_subset_k12_shapes(synthetic_trial: np.ndarray) -> None:
    # K=12 primary from research/xai/channel_reduction/run.json
    subset = [3, 9, 5, 15, 20, 8, 21, 7, 18, 16, 2, 0]
    ds = _bare_dataset(subset)
    g = ds._build_graph(synthetic_trial.copy(), fs=10.0)

    assert g.x.shape == (12, 326)
    # All edge endpoints must be in [0, 12)
    assert g.edge_index.min().item() >= 0
    assert g.edge_index.max().item() < 12
    # edge_attr aligns with edge_index
    assert g.edge_attr.shape[0] == g.edge_index.shape[1]
    assert g.edge_attr.shape[1] == 2


def test_subset_k8_shapes(synthetic_trial: np.ndarray) -> None:
    subset = [3, 9, 5, 15, 20, 8, 21, 7]
    ds = _bare_dataset(subset)
    g = ds._build_graph(synthetic_trial.copy(), fs=10.0)
    assert g.x.shape == (8, 326)
    assert g.edge_index.max().item() < 8


def test_subset_slicing_matches_post_hoc_slice(synthetic_trial: np.ndarray) -> None:
    """Z-scoring K rows of trial must equal z-scoring the full trial then slicing
    the result. (Per-channel z-score is row-independent — this is the sanity guard.)"""
    subset = [3, 9, 5, 15, 20, 8, 21, 7, 18, 16, 2, 0]

    ds_subset = _bare_dataset(subset)
    g_subset = ds_subset._build_graph(synthetic_trial.copy(), fs=10.0)

    # Compute x_raw on full trial, then slice rows
    full = synthetic_trial.copy()
    mu = full.mean(axis=1, keepdims=True)
    sigma = full.std(axis=1, keepdims=True).clip(min=1e-8)
    x_full = ((full - mu) / sigma).astype(np.float32)
    x_sliced = torch.tensor(x_full[subset, :], dtype=torch.float)

    assert torch.allclose(g_subset.x, x_sliced)


# ---------------------------------------------------------------------------
# 3. Model forward pass works with K<23
# ---------------------------------------------------------------------------


def test_model_forward_k12() -> None:
    """End-to-end shape contract with K=12. Uses two graphs (batch_size=2) so
    BatchNorm has >1 sample per channel in eval mode."""
    K = 12
    T = 326
    model = WindowedSpatioTemporalGATNet(
        n_channels=K,
        in_channels=6,
        edge_dim=2,
        window_size=16,
        window_stride=8,
        n_layers=2,
        n_filters=80,
        heads=2,
        temporal_hidden=192,
        temporal_layers=1,
        fc_size=256,
        dropout=0.3,
        n_classes=2,
        use_residual=False,
        use_norm=True,
        norm_type="batch",
    ).eval()

    # Build 2 graphs each with K nodes, then PyG-batch them
    edge_index_single = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]],
        dtype=torch.long,
    )
    edge_attr_single = torch.randn(edge_index_single.shape[1], 2)
    g1 = Data(x=torch.randn(K, T), edge_index=edge_index_single, edge_attr=edge_attr_single)
    g2 = Data(x=torch.randn(K, T), edge_index=edge_index_single, edge_attr=edge_attr_single)

    from torch_geometric.loader import DataLoader

    loader = DataLoader([g1, g2], batch_size=2)
    batch = next(iter(loader))

    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# 4. _resolve_channel_subset name-or-int parsing
# ---------------------------------------------------------------------------


def test_resolve_none_and_all() -> None:
    assert _resolve_channel_subset(None) is None
    assert _resolve_channel_subset("") is None
    assert _resolve_channel_subset("all") is None
    assert _resolve_channel_subset("ALL") is None
    assert _resolve_channel_subset([]) is None


def test_resolve_csv_ints() -> None:
    assert _resolve_channel_subset("0,3,5,7") == [0, 3, 5, 7]
    assert _resolve_channel_subset(" 0 , 3 , 5 ") == [0, 3, 5]


def test_resolve_csv_names() -> None:
    # S1_D1=0, S2_D1=3, S3_D1=5 per src/xai/channels.py
    assert _resolve_channel_subset("S1_D1,S2_D1,S3_D1") == [0, 3, 5]


def test_resolve_yaml_list_of_ints() -> None:
    assert _resolve_channel_subset([0, 3, 5]) == [0, 3, 5]


def test_resolve_yaml_list_of_names() -> None:
    assert _resolve_channel_subset(["S1_D1", "S2_D1", "S3_D1"]) == [0, 3, 5]


def test_resolve_yaml_mixed() -> None:
    assert _resolve_channel_subset([0, "S2_D1", 5]) == [0, 3, 5]


def test_resolve_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _resolve_channel_subset("0,3,0")


def test_resolve_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="not an int"):
        _resolve_channel_subset("S9_D9")


def test_resolve_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _resolve_channel_subset("0,23")


def test_resolve_rejects_bool() -> None:
    with pytest.raises(ValueError, match="bool"):
        _resolve_channel_subset([True, 0])
