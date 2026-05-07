"""Figure generation — SPEC §8.

Five figure functions, all writing PNG @ 300 dpi plus an SVG mirror per
SPEC §8 (vector-friendly for paper inclusion):

    plot_montage_channel_importance(result, save_dir)   -> fig 1 (5×7 montage)
    plot_pair_matrix(result, save_dir)                  -> fig 2 (23×23 heatmap)
    plot_temporal_attention(result, save_dir)           -> fig 4 (ST only)
    plot_sg_vs_st_scatter(sg, st, save_dir)             -> fig 5 (cross-arch)
    plot_pair_matrix_diff(sg, st, save_dir)             -> fig 6 (cross-arch)

The optional fig 3 (top-pairs chord diagram, SPEC §8 item 3) is intentionally
not implemented in v1 — it requires an external chord helper and the SPEC
itself flags it as optional.

Each function returns the absolute path of the saved `.png`. The `.svg`
sibling is always written next to it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # no interactive backend — these are file outputs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as _scipy_stats

from src.xai.aggregate import PopulationResult
from src.xai.channels import CHANNEL_NAMES, GRID_POS, GRID_SHAPE, N_CH


_RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _save_both(fig: plt.Figure, save_dir: Path, stem: str) -> Path:
    """Write `<stem>.png` (300 dpi) and `<stem>.svg` next to it; return the PNG path."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    png = save_dir / f"{stem}.png"
    svg = save_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# Single-result figures
# ---------------------------------------------------------------------------


def plot_montage_channel_importance(
    result: PopulationResult,
    save_dir: str | Path,
    *,
    stem: str = "fig_montage_channel_importance",
    title: str | None = None,
) -> Path:
    """Fig 1 — 5×7 grid heatmap of per-channel importance (z-scored).

    Layout matches `02_brain_activation/REPORT.md` so XAI maps line up
    visually with the statistical-analysis figures.
    """
    with plt.rc_context(_RC):
        ch = result.channel_importance_mean
        z = (ch - ch.mean()) / (ch.std(ddof=0) + 1e-12)

        grid = np.full(GRID_SHAPE, np.nan, dtype=np.float64)
        for i, (r, c) in enumerate(GRID_POS):
            grid[r, c] = z[i]

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 1e-6)
        im = ax.imshow(grid, cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                       aspect="auto")
        for i, (r, c) in enumerate(GRID_POS):
            zv = z[i]
            txt_color = "white" if abs(zv) > 0.6 * vmax else "black"
            ax.text(c, r - 0.22, CHANNEL_NAMES[i], ha="center", va="center",
                    fontsize=6.5, color=txt_color)
            ax.text(c, r + 0.20, f"{zv:+.2f}", ha="center", va="center",
                    fontsize=7.0, color=txt_color, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title or _default_title(result, "Channel importance (z-score over 23 channels)"))
        plt.colorbar(im, ax=ax, shrink=0.75, label="z-score")
        return _save_both(fig, Path(save_dir), stem)


def plot_pair_matrix(
    result: PopulationResult,
    save_dir: str | Path,
    *,
    stem: str = "fig_pair_matrix",
    title: str | None = None,
) -> Path:
    """Fig 2 — 23×23 symmetric heatmap; channels reordered by row-sum so the
    high-importance subgraph is the dense top-left block.
    """
    with plt.rc_context(_RC):
        M = result.pair_matrix.copy()
        np.fill_diagonal(M, np.nan)
        order = np.argsort(-np.nansum(M, axis=1))
        M_sorted = M[order][:, order]
        labels = [CHANNEL_NAMES[i] for i in order]

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        vmax = float(np.nanmax(M_sorted)) if np.isfinite(np.nanmax(M_sorted)) else 1.0
        im = ax.imshow(M_sorted, cmap="magma_r", vmin=0.0, vmax=vmax, aspect="equal")
        ax.set_xticks(np.arange(N_CH)); ax.set_yticks(np.arange(N_CH))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(title or _default_title(result, "Channel-pair importance (23×23, ordered by row-sum)"))
        plt.colorbar(im, ax=ax, shrink=0.85, label="pair importance")
        return _save_both(fig, Path(save_dir), stem)


def plot_temporal_attention(
    result: PopulationResult,
    save_dir: str | Path,
    *,
    stem: str = "fig_temporal_attention",
    title: str | None = None,
) -> Path:
    """Fig 4 (ST only) — α_k over time, mean ± 95% CI band.

    Raises ValueError if the result has no temporal attention (SG case).
    """
    if result.temporal_attention_mean is None:
        raise ValueError(
            "plot_temporal_attention requires an ST PopulationResult "
            "(temporal_attention_mean is None)"
        )

    with plt.rc_context(_RC):
        mean = result.temporal_attention_mean
        std = result.temporal_attention_std
        n = max(result.n_subjects, 1)
        # 95% CI from across-subject std (use 1.96·SE for normal approx).
        ci = 1.96 * (std if std is not None else np.zeros_like(mean)) / np.sqrt(n)

        if result.window_times is not None:
            t_centers = result.window_times.mean(axis=1)
            x_label = "Time (s)"
        else:
            t_centers = np.arange(mean.shape[0], dtype=np.float64)
            x_label = "Window index"

        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ax.plot(t_centers, mean, color="#5577CC", lw=1.6, label="α_k mean")
        ax.fill_between(t_centers, mean - ci, mean + ci, color="#5577CC", alpha=0.18,
                        label="95% CI")
        ax.axhline(1.0 / mean.shape[0], color="grey", lw=0.8, ls="--",
                   label=f"uniform (1/K={1.0 / mean.shape[0]:.3f})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Temporal attention α_k")
        ax.set_title(title or _default_title(result, "ST temporal attention"))
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        return _save_both(fig, Path(save_dir), stem)


# ---------------------------------------------------------------------------
# Cross-architecture figures
# ---------------------------------------------------------------------------


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = _scipy_stats.spearmanr(a, b)
    return float(rho)


def plot_sg_vs_st_scatter(
    sg: PopulationResult,
    st: PopulationResult,
    save_dir: str | Path,
    *,
    stem: str = "fig_sg_vs_st_scatter",
) -> Path:
    """Fig 5 — scatter of SG vs ST channel-importance ranks.

    Channels labelled with their montage names. Spearman ρ in the legend.
    """
    with plt.rc_context(_RC):
        x = sg.channel_importance_mean
        y = st.channel_importance_mean
        rho = _spearman_rho(x, y)

        rx = (-x).argsort().argsort() + 1   # rank 1 = largest
        ry = (-y).argsort().argsort() + 1

        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.scatter(rx, ry, s=42, color="#5577CC", alpha=0.85, edgecolor="white")
        for i, (xi, yi) in enumerate(zip(rx, ry)):
            ax.annotate(CHANNEL_NAMES[i], (xi, yi),
                        xytext=(4, 4), textcoords="offset points", fontsize=6.5)
        # Diagonal reference.
        lim = (0.5, N_CH + 0.5)
        ax.plot(lim, lim, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.invert_yaxis()  # rank 1 at top
        ax.invert_xaxis()  # rank 1 at left
        ax.set_xlabel("SG rank (1 = most important)")
        ax.set_ylabel("ST rank (1 = most important)")
        ax.set_title(f"Channel-importance ranks: SG vs ST  (Spearman ρ = {rho:+.3f})")
        return _save_both(fig, Path(save_dir), stem)


def plot_pair_matrix_diff(
    sg: PopulationResult,
    st: PopulationResult,
    save_dir: str | Path,
    *,
    stem: str = "fig_pair_matrix_diff",
) -> Path:
    """Fig 6 — z(M_sg) − z(M_st), 23×23 diverging heatmap (SPEC §7.4).

    Each architecture's pair matrix is z-scored over its non-diagonal entries
    so the diff is dimensionless. Red ⇒ SG-only edges, blue ⇒ ST-only edges,
    near-zero ⇒ agreement.
    """
    sg_z = _z_offdiag(sg.pair_matrix)
    st_z = _z_offdiag(st.pair_matrix)
    diff = sg_z - st_z
    np.fill_diagonal(diff, 0.0)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 1e-6)
        im = ax.imshow(diff, cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                       aspect="equal")
        ax.set_xticks(np.arange(N_CH)); ax.set_yticks(np.arange(N_CH))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=6)
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=6)
        ax.set_title("z(M_sg) − z(M_st)   (red = SG-only, blue = ST-only)")
        plt.colorbar(im, ax=ax, shrink=0.85, label="z-score difference")
        return _save_both(fig, Path(save_dir), stem)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _z_offdiag(M: np.ndarray) -> np.ndarray:
    """Z-score `M` using mean/std computed over off-diagonal entries only."""
    mask = ~np.eye(M.shape[0], dtype=bool)
    vals = M[mask]
    mu = float(vals.mean())
    sd = float(vals.std(ddof=0)) + 1e-12
    return (M - mu) / sd


def _default_title(result: PopulationResult, what: str) -> str:
    return (
        f"{what}\n"
        f"{result.arch.upper()} · {result.hb} · {result.regime} · mt={result.mt}  "
        f"(n_subjects={result.n_subjects}, n_trials={result.n_trials}, "
        f"included={result.included_pct:.1f}%)"
    )
