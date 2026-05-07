"""Aggregation pipeline (SPEC §7).

Pipeline:

    TrialAttribution (per validation graph, per checkpoint)
        ↓ filter to correctly-classified
        ↓ mean over a subject's trials
    SubjectAttribution
        ↓ mean over subjects
    PopulationResult     ← what gets written to CSV / NPY (SPEC §7.3)

`aggregate_population` is the top-level entry point. The two explainer modules
(`sg_explainer.py`, `st_explainer.py`) produce the per-trial list; everything
beyond that lives here so per-trial logic and aggregation logic don't tangle.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.xai.channels import CHANNEL_NAMES, N_CH


# ---------------------------------------------------------------------------
# Per-trial attribution (produced by an explainer)
# ---------------------------------------------------------------------------


@dataclass
class TrialAttribution:
    """One explainer output for one validation graph.

    Per-trial reductions (SPEC §5.2 step 7 / §6.1):
    - `channel_importance` : sum of |node_mask| over features (23,)
    - `feature_importance` : sum of |node_mask| over channels (6,) — SG only
    - `pair_matrix`         : symmetric 23×23 — sum of edge_mask aggregated by
                              endpoint pair, then averaged with its transpose
                              (the directed dataset duplicates each pair so
                              symmetrisation is loss-free)
    - `temporal_attention`  : per-window α_k (K,) — ST only
    - `window_times`        : per-window (t_start_s, t_end_s) (K, 2) — ST only
    """

    subject_id: str
    fold_or_subj_label: str           # 'fold-1' or 'subj-AA011'
    trial_idx_in_subject: int          # 0-based ordinal within subject's val trials
    true_label: int
    pred_label: int
    included: bool                     # passes correctly-classified filter

    channel_importance: np.ndarray     # (23,) float32, all >= 0
    pair_matrix: np.ndarray            # (23, 23) float32, symmetric

    feature_importance: Optional[np.ndarray] = None   # (6,) — SG only
    temporal_attention: Optional[np.ndarray] = None    # (K,) — ST only
    window_times: Optional[np.ndarray] = None          # (K, 2) — ST only

    def __post_init__(self) -> None:
        assert self.channel_importance.shape == (N_CH,), self.channel_importance.shape
        assert self.pair_matrix.shape == (N_CH, N_CH), self.pair_matrix.shape


# ---------------------------------------------------------------------------
# Population-level result (the SPEC §7.3 deliverable)
# ---------------------------------------------------------------------------


@dataclass
class PopulationResult:
    arch: str
    hb: str
    regime: str
    mt: int

    # Per-channel: 23-vector means / stds across the population.
    channel_importance_mean: np.ndarray              # (23,)
    channel_importance_std: np.ndarray               # (23,)

    # Per channel-pair: symmetric 23×23 mean across the population.
    pair_matrix: np.ndarray                          # (23, 23)
    pair_matrix_std: np.ndarray                      # (23, 23) — across-subject std

    # Counters / audit.
    n_trials: int                                    # included trials
    n_trials_total: int                              # included + excluded
    n_subjects: int
    included_pct: float
    per_subject_trial_counts: Dict[str, int]          # included only

    # SG-only.
    feature_importance_mean: Optional[np.ndarray] = None    # (6,)
    feature_importance_std: Optional[np.ndarray] = None     # (6,)

    # ST-only.
    temporal_attention_mean: Optional[np.ndarray] = None    # (K,)
    temporal_attention_std: Optional[np.ndarray] = None     # (K,)
    window_times: Optional[np.ndarray] = None               # (K, 2) seconds

    # Provenance / extras the writer carries through to run.json.
    extras: Dict[str, object] = field(default_factory=dict)

    # ---------------------------------------------------------------- io
    _NODE_CSV = "node_importance.csv"
    _EDGE_CSV = "edge_importance.csv"
    _MATRIX_NPY = "channel_pair_matrix.npy"
    _MATRIX_STD_NPY = "channel_pair_matrix_std.npy"
    _FEATURE_CSV = "feature_importance.csv"
    _TEMPORAL_CSV = "temporal_attention.csv"
    _META_JSON = "result_meta.json"

    def to_csv(self, out_dir: str | Path, *, top_k_edges: int = 50) -> None:
        """Persist the §7.3 deliverable to disk.

        `top_k_edges` controls how many edge rows go into edge_importance.csv;
        the full 23×23 matrix is always saved as channel_pair_matrix.npy.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # node_importance.csv
        per_subj = list(self.per_subject_trial_counts.values())
        # Per-channel n_trials is the same for every channel (same trials feed
        # every channel mean); keep n_trials as a column for downstream sanity.
        ranks = (-self.channel_importance_mean).argsort().argsort() + 1
        node_df = pd.DataFrame({
            "channel": CHANNEL_NAMES,
            "mean": self.channel_importance_mean,
            "std": self.channel_importance_std,
            "n_trials": [self.n_trials] * N_CH,
            "n_subjects": [self.n_subjects] * N_CH,
            "rank": ranks,
        })
        node_df.to_csv(out / self._NODE_CSV, index=False, float_format="%.6f")

        # edge_importance.csv — top-K off-diagonal pairs (after symmetrisation,
        # both (i, j) and (j, i) carry the same value; keep only i < j).
        rows: List[Tuple[str, str, float, float, int, int]] = []
        std = self.pair_matrix_std
        for i in range(N_CH):
            for j in range(i + 1, N_CH):
                rows.append((CHANNEL_NAMES[i], CHANNEL_NAMES[j],
                             float(self.pair_matrix[i, j]), float(std[i, j]),
                             self.n_trials, 0))
        rows.sort(key=lambda r: -r[2])
        rows = rows[:top_k_edges]
        edge_df = pd.DataFrame(rows, columns=[
            "channel_i", "channel_j", "mean", "std", "n_trials", "rank",
        ])
        edge_df["rank"] = np.arange(1, len(edge_df) + 1)
        edge_df.to_csv(out / self._EDGE_CSV, index=False, float_format="%.6f")

        # raw matrices
        np.save(out / self._MATRIX_NPY, self.pair_matrix.astype(np.float32))
        np.save(out / self._MATRIX_STD_NPY, self.pair_matrix_std.astype(np.float32))

        if self.feature_importance_mean is not None:
            assert self.feature_importance_std is not None
            feat_df = pd.DataFrame({
                "feature": ["mean", "min", "max", "skew", "kurt", "var"],
                "mean": self.feature_importance_mean,
                "std": self.feature_importance_std,
            })
            feat_df.to_csv(out / self._FEATURE_CSV, index=False, float_format="%.6f")

        if self.temporal_attention_mean is not None:
            assert self.temporal_attention_std is not None
            K = self.temporal_attention_mean.shape[0]
            wt = self.window_times
            t_start = wt[:, 0] if wt is not None else np.full(K, np.nan)
            t_end = wt[:, 1] if wt is not None else np.full(K, np.nan)
            tmp_df = pd.DataFrame({
                "window_idx": np.arange(K),
                "t_start_s": t_start,
                "t_end_s": t_end,
                "mean": self.temporal_attention_mean,
                "std": self.temporal_attention_std,
            })
            tmp_df.to_csv(out / self._TEMPORAL_CSV, index=False, float_format="%.6f")

        meta = {
            "arch": self.arch,
            "hb": self.hb,
            "regime": self.regime,
            "mt": self.mt,
            "n_trials": self.n_trials,
            "n_trials_total": self.n_trials_total,
            "n_subjects": self.n_subjects,
            "included_pct": self.included_pct,
            "per_subject_trial_counts": self.per_subject_trial_counts,
            "extras": self.extras,
        }
        (out / self._META_JSON).write_text(json.dumps(meta, indent=2))

    @classmethod
    def from_csv(cls, in_dir: str | Path) -> "PopulationResult":
        """Inverse of `to_csv`. Used by the cross-arch comparison notebook."""
        in_path = Path(in_dir)
        meta = json.loads((in_path / cls._META_JSON).read_text())

        node_df = pd.read_csv(in_path / cls._NODE_CSV)
        ch_mean = node_df.sort_values("channel", key=lambda s: s.map(CHANNEL_NAMES.index))
        # Restore CHANNEL_NAMES order regardless of how to_csv wrote.
        ch_mean = node_df.set_index("channel").reindex(CHANNEL_NAMES)
        channel_importance_mean = ch_mean["mean"].to_numpy(dtype=np.float64)
        channel_importance_std = ch_mean["std"].to_numpy(dtype=np.float64)

        pair_matrix = np.load(in_path / cls._MATRIX_NPY).astype(np.float64)
        pair_matrix_std = np.load(in_path / cls._MATRIX_STD_NPY).astype(np.float64)

        feature_mean = feature_std = None
        if (in_path / cls._FEATURE_CSV).exists():
            f = pd.read_csv(in_path / cls._FEATURE_CSV)
            feature_mean = f["mean"].to_numpy(dtype=np.float64)
            feature_std = f["std"].to_numpy(dtype=np.float64)

        temporal_mean = temporal_std = None
        window_times = None
        if (in_path / cls._TEMPORAL_CSV).exists():
            t = pd.read_csv(in_path / cls._TEMPORAL_CSV)
            temporal_mean = t["mean"].to_numpy(dtype=np.float64)
            temporal_std = t["std"].to_numpy(dtype=np.float64)
            window_times = t[["t_start_s", "t_end_s"]].to_numpy(dtype=np.float64)

        return cls(
            arch=meta["arch"],
            hb=meta["hb"],
            regime=meta["regime"],
            mt=int(meta["mt"]),
            channel_importance_mean=channel_importance_mean,
            channel_importance_std=channel_importance_std,
            pair_matrix=pair_matrix,
            pair_matrix_std=pair_matrix_std,
            n_trials=int(meta["n_trials"]),
            n_trials_total=int(meta["n_trials_total"]),
            n_subjects=int(meta["n_subjects"]),
            included_pct=float(meta["included_pct"]),
            per_subject_trial_counts=dict(meta["per_subject_trial_counts"]),
            feature_importance_mean=feature_mean,
            feature_importance_std=feature_std,
            temporal_attention_mean=temporal_mean,
            temporal_attention_std=temporal_std,
            window_times=window_times,
            extras=dict(meta.get("extras", {})),
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _stack_per_subject(
    trials: Sequence[TrialAttribution],
    *,
    only_included: bool,
) -> Dict[str, List[TrialAttribution]]:
    grouped: Dict[str, List[TrialAttribution]] = {}
    for t in trials:
        if only_included and not t.included:
            continue
        grouped.setdefault(t.subject_id, []).append(t)
    return grouped


def _subject_means(
    grouped: Dict[str, List[TrialAttribution]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray],
           Optional[Dict[str, np.ndarray]],
           Optional[Dict[str, np.ndarray]],
           Optional[np.ndarray]]:
    """Mean over a subject's trials → per-subject vectors / matrices."""
    ch: Dict[str, np.ndarray] = {}
    pair: Dict[str, np.ndarray] = {}
    feat: Dict[str, np.ndarray] = {}
    temp: Dict[str, np.ndarray] = {}
    window_times: Optional[np.ndarray] = None
    for sid, ts in grouped.items():
        ch[sid] = np.mean([t.channel_importance for t in ts], axis=0)
        pair[sid] = np.mean([t.pair_matrix for t in ts], axis=0)
        if ts[0].feature_importance is not None:
            feat[sid] = np.mean([t.feature_importance for t in ts], axis=0)  # type: ignore[arg-type]
        if ts[0].temporal_attention is not None:
            temp[sid] = np.mean([t.temporal_attention for t in ts], axis=0)  # type: ignore[arg-type]
            if window_times is None and ts[0].window_times is not None:
                window_times = ts[0].window_times
    return ch, pair, (feat or None), (temp or None), window_times


def aggregate_population(
    trials: Sequence[TrialAttribution],
    *,
    arch: str,
    hb: str,
    regime: str,
    mt: int,
    only_included: bool = True,
    extras: Optional[Dict[str, object]] = None,
) -> PopulationResult:
    """Aggregate per-trial attributions into a PopulationResult.

    Following SPEC §7.1 / §7.2: the population mean is taken across subjects
    (equal weight per subject regardless of trial count), itself a mean over
    each subject's correctly-classified trials. This avoids letting a subject
    with more retained trials dominate the population estimate.
    """
    if not trials:
        raise ValueError("aggregate_population: no trials")

    grouped = _stack_per_subject(trials, only_included=only_included)
    if not grouped:
        raise ValueError("aggregate_population: no included trials (all excluded by filter)")

    ch, pair, feat, temp, window_times = _subject_means(grouped)

    sids = sorted(ch.keys())
    ch_stack = np.stack([ch[s] for s in sids])                # (S, 23)
    pair_stack = np.stack([pair[s] for s in sids])             # (S, 23, 23)

    channel_importance_mean = ch_stack.mean(axis=0)
    channel_importance_std = ch_stack.std(axis=0, ddof=1) if len(sids) > 1 else np.zeros(N_CH)

    pair_mean = pair_stack.mean(axis=0)
    pair_mean = (pair_mean + pair_mean.T) / 2.0
    pair_std = pair_stack.std(axis=0, ddof=1) if len(sids) > 1 else np.zeros((N_CH, N_CH))

    feature_mean = feature_std = None
    if feat is not None:
        f_stack = np.stack([feat[s] for s in sids])
        feature_mean = f_stack.mean(axis=0)
        feature_std = f_stack.std(axis=0, ddof=1) if len(sids) > 1 else np.zeros_like(feature_mean)

    temporal_mean = temporal_std = None
    if temp is not None:
        t_stack = np.stack([temp[s] for s in sids])
        temporal_mean = t_stack.mean(axis=0)
        temporal_std = t_stack.std(axis=0, ddof=1) if len(sids) > 1 else np.zeros_like(temporal_mean)

    n_included = sum(len(ts) for ts in grouped.values())
    n_total = sum(1 for t in trials)
    included_pct = (100.0 * n_included / n_total) if n_total > 0 else 0.0
    per_subject_trial_counts = {s: len(grouped[s]) for s in sids}

    return PopulationResult(
        arch=arch,
        hb=hb,
        regime=regime,
        mt=mt,
        channel_importance_mean=channel_importance_mean.astype(np.float64),
        channel_importance_std=channel_importance_std.astype(np.float64),
        pair_matrix=pair_mean.astype(np.float64),
        pair_matrix_std=pair_std.astype(np.float64),
        feature_importance_mean=None if feature_mean is None else feature_mean.astype(np.float64),
        feature_importance_std=None if feature_std is None else feature_std.astype(np.float64),
        temporal_attention_mean=None if temporal_mean is None else temporal_mean.astype(np.float64),
        temporal_attention_std=None if temporal_std is None else temporal_std.astype(np.float64),
        window_times=window_times,
        n_trials=n_included,
        n_trials_total=n_total,
        n_subjects=len(sids),
        included_pct=included_pct,
        per_subject_trial_counts=per_subject_trial_counts,
        extras=extras or {},
    )
