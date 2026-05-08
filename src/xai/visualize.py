"""Figure generation — SPEC §8 + §16.8 (rev. 5).

Channel-level (SPEC §8):

    plot_montage_channel_importance(result, save_dir)   -> fig 1 (5×7 montage)
    plot_pair_matrix(result, save_dir)                  -> fig 2 (23×23 heatmap)
    plot_temporal_attention(result, save_dir)           -> fig 4 (ST only)
    plot_sg_vs_st_scatter(sg, st, save_dir)             -> fig 5 (cross-arch)
    plot_pair_matrix_diff(sg, st, save_dir)             -> fig 6 (cross-arch)

Atlas / region-level (SPEC §16.8, rev. 5):

    plot_brodmann_montage(ch2ba, save_dir)              -> 5×7 montage by primary BA
    plot_brodmann_surface(diag, save_dir, subjects_dir) -> fsaverage 3D scatter
    plot_region_bar(region_imp, save_dir)               -> ranked BA bar chart
    plot_region_pair_heatmap(M, keys, save_dir)         -> N_BA × N_BA heatmap

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


# =========================================================================== #
# Atlas / region-level figures (SPEC §16.8, rev. 5)                           #
# =========================================================================== #

# Stable categorical palette for BA assignment. Only the 4 BAs that fall
# inside the 23-channel prefrontal montage need colours; anything else
# (e.g. an Unlabelled fallback) gets neutral grey.
_BA_COLOURS: dict[str, str] = {
    "Brodmann.10": "#4C72B0",   # frontopolar
    "Brodmann.9":  "#DD8452",   # DLPFC anterior
    "Brodmann.46": "#55A467",   # DLPFC core
    "Brodmann.8":  "#C44E52",   # DMPFC posterior / pre-SMA
    "Unlabelled":  "#9C9C9C",
}
_BA_COLOUR_FALLBACK = "#7F7F7F"


def _ba_colour(ba_label: str) -> str:
    return _BA_COLOURS.get(ba_label, _BA_COLOUR_FALLBACK)


def _primary_ba_per_channel(channel_to_brodmann_long) -> "list[tuple[str, str, float]]":
    """Reduce the long ``channel→BA`` table to one ``(ba_label, hemi, prob)``
    per channel — the row with the largest probability.

    Accepts either a DataFrame or a path/string pointing at a CSV in the
    long format produced by `atlas.build_channel_to_brodmann`.
    """
    import pandas as pd  # local import keeps the chunk above pandas-free
    df = channel_to_brodmann_long
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df)
    elif not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    out: list[tuple[str, str, float]] = []
    for ch in CHANNEL_NAMES:
        sub = df[df["channel"] == ch].sort_values("probability", ascending=False)
        if sub.empty:
            out.append(("Unlabelled", "U", 0.0))
        else:
            r = sub.iloc[0]
            out.append((str(r["ba_label"]), str(r["hemi"]), float(r["probability"])))
    return out


def plot_brodmann_montage(
    channel_to_brodmann_long,
    save_dir: str | Path,
    *,
    stem: str = "fig_montage_brodmann",
    title: str | None = None,
) -> Path:
    """5×7 grid coloured by the **dominant Brodmann area** per channel.

    Layout matches `plot_montage_channel_importance` so the two figures can be
    overlaid in paper figures. Each cell shows the channel name and the
    BA short code (e.g. "10", "46"), with the cell colour given by `_BA_COLOURS`.
    """
    primary = _primary_ba_per_channel(channel_to_brodmann_long)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        # Render a blank grid first.
        ax.set_xlim(-0.6, GRID_SHAPE[1] - 0.4)
        ax.set_ylim(GRID_SHAPE[0] - 0.4, -0.6)   # invert y so row 0 sits on top
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # One filled rectangle per channel.
        used_bas: dict[str, str] = {}
        for i, (r, c) in enumerate(GRID_POS):
            ba, hemi, p = primary[i]
            colour = _ba_colour(ba)
            used_bas[ba] = colour
            ax.add_patch(plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                        facecolor=colour, edgecolor="white",
                                        linewidth=1.2, alpha=0.92))
            short_ba = ba.replace("Brodmann.", "BA")
            ax.text(c, r - 0.20, CHANNEL_NAMES[i], ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")
            ax.text(c, r + 0.18, f"{short_ba}-{hemi}",
                    ha="center", va="center", fontsize=7.5, color="white")
        ax.set_title(title or "Channel → primary Brodmann area (PALS_B12)")

        # Legend — sorted by BA number.
        def _ba_sort_key(name: str) -> tuple[int, str]:
            try:
                return (int(name.replace("Brodmann.", "")), name)
            except ValueError:
                return (10**6, name)
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=col, edgecolor="white")
            for ba, col in sorted(used_bas.items(), key=lambda kv: _ba_sort_key(kv[0]))
        ]
        legend_labels = [
            ba.replace("Brodmann.", "BA ") if ba.startswith("Brodmann.") else ba
            for ba, _ in sorted(used_bas.items(), key=lambda kv: _ba_sort_key(kv[0]))
        ]
        ax.legend(legend_handles, legend_labels,
                  loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=8)
        return _save_both(fig, Path(save_dir), stem)


def plot_brodmann_surface(
    midpoints_mri_mm,
    save_dir: str | Path,
    subjects_dir: str | Path | None = None,
    subject: str = "fsaverage",
    *,
    stem: str = "fig_surface_atlas",
    title: str | None = None,
) -> Path:
    """3D scatter of the 23 channel midpoints over the fsaverage pial surface.

    Implementation: matplotlib 3D scatter (no PyVista dependency). The pial
    surface is rendered as a low-alpha point cloud subsampled for speed.
    Marker colour = primary BA (from `_BA_COLOURS`); marker size scales with
    inverse projection distance (closer-to-cortex channels are bigger).

    `midpoints_mri_mm` should be the DataFrame returned by
    `BrodmannMapping.midpoints_mri_mm`. If the optional ``primary_ba`` column
    is present, it is used directly; otherwise everything is rendered grey.
    """
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)
    df = midpoints_mri_mm
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Fetch fsaverage if not provided.
    if subjects_dir is None:
        import mne
        fs_root = mne.datasets.fetch_fsaverage(verbose="ERROR")
        subjects_dir = Path(fs_root).parent
    subjects_dir = Path(subjects_dir)

    import mne
    surf_dir = subjects_dir / subject / "surf"
    verts_lh, _ = mne.read_surface(str(surf_dir / "lh.pial"))
    verts_rh, _ = mne.read_surface(str(surf_dir / "rh.pial"))

    # Subsample for plotting speed (full pial = ~150k verts/hemi).
    rng = np.random.default_rng(seed=0)
    sub_lh = verts_lh[rng.choice(len(verts_lh), size=8000, replace=False)]
    sub_rh = verts_rh[rng.choice(len(verts_rh), size=8000, replace=False)]
    sub_all = np.concatenate([sub_lh, sub_rh], axis=0)

    primary_col = df.get("primary_ba")
    proj_dist = df.get("projection_distance_mm",
                        pd.Series([20.0] * len(df)))

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(8.0, 7.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(sub_all[:, 0], sub_all[:, 1], sub_all[:, 2],
                   c="lightgrey", s=0.25, alpha=0.18, linewidths=0)

        x = df["x_mni_mm"].to_numpy()
        y = df["y_mni_mm"].to_numpy()
        z = df["z_mni_mm"].to_numpy()
        # Marker size in [40, 120] inversely scaled by projection distance.
        d = proj_dist.to_numpy(dtype=float)
        d_norm = (d - d.min()) / max(float(np.ptp(d)), 1e-9)
        sizes = 120.0 - 80.0 * d_norm

        used_bas: dict[str, str] = {}
        if primary_col is not None:
            for xi, yi, zi, si, ba_i in zip(x, y, z, sizes, primary_col):
                col = _ba_colour(str(ba_i))
                used_bas[str(ba_i)] = col
                ax.scatter([xi], [yi], [zi], c=col, s=si, edgecolor="black", linewidths=0.6)
        else:
            ax.scatter(x, y, z, c="#5577CC", s=sizes, edgecolor="black", linewidths=0.6)

        # Annotate channel names — keep concise; offset slightly along +z.
        for ch, xi, yi, zi in zip(df["channel"].to_numpy(), x, y, z):
            ax.text(xi, yi, zi + 6.0, str(ch), fontsize=5.5,
                    ha="center", color="black")

        ax.set_xlabel("X (MNI, mm)")
        ax.set_ylabel("Y (MNI, mm)")
        ax.set_zlabel("Z (MNI, mm)")
        ax.set_title(title or "23 channel midpoints on fsaverage pial surface")
        # Anterior-superior view that shows the prefrontal montage best.
        ax.view_init(elev=30, azim=-95)

        if used_bas:
            handles = [plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=col, markersize=8,
                                   markeredgecolor="black", label=ba.replace(
                                       "Brodmann.", "BA "))
                       for ba, col in sorted(used_bas.items())]
            ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
        return _save_both(fig, Path(save_dir), stem)


def plot_region_bar(
    region_imp,
    save_dir: str | Path,
    *,
    stem: str = "fig_region_bar",
    title: str | None = None,
) -> Path:
    """Horizontal bar chart of region importance, panels separated by hemisphere.

    `region_imp` should be the DataFrame returned by `aggregate_to_regions`.
    Bars are colour-coded by BA via `_BA_COLOURS`. Panels: L (left), R (right),
    plus M / U if any rows fall there.
    """
    import pandas as pd
    df = region_imp
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if df.empty:
        raise ValueError("region_imp is empty")

    # Group by hemi so we can show side-by-side panels.
    hemis = sorted(df["hemi"].unique(), key=lambda h: ("L", "M", "R", "U").index(h)
                   if h in ("L", "M", "R", "U") else 99)

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(hemis),
                                  figsize=(3.6 * len(hemis), 0.45 * df.shape[0] + 1.5),
                                  sharex=True)
        if len(hemis) == 1:
            axes = [axes]

        vmax = float(df["mean"].max()) * 1.05 + 1e-9

        for ax, h in zip(axes, hemis):
            sub = df[df["hemi"] == h].sort_values("mean", ascending=True)
            colours = [_ba_colour(b) for b in sub["ba_label"]]
            y = np.arange(len(sub))
            ax.barh(y, sub["mean"], color=colours, edgecolor="white", linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels([
                b.replace("Brodmann.", "BA ") if b.startswith("Brodmann.") else b
                for b in sub["ba_label"]
            ])
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.text(row["mean"] + 0.01 * vmax, i,
                        f"  (n={int(row['n_channels_contrib'])})",
                        va="center", fontsize=7, color="grey")
            ax.set_xlim(0, vmax)
            ax.set_title(f"Hemisphere: {h}")
            ax.set_xlabel("Region importance (Σ ch_imp × P)")
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
        fig.suptitle(title or "Brodmann region importance", y=1.02)
        fig.tight_layout()
        return _save_both(fig, Path(save_dir), stem)


def plot_montage_with_atlas(
    result: PopulationResult,
    channel_to_brodmann_long,
    save_dir: str | Path,
    *,
    stem: str = "fig_montage_with_atlas",
    title: str | None = None,
) -> Path:
    """**Paper-ready combined figure.** 5×7 grid where:

    * **cell border colour** = primary Brodmann area (from the atlas mapping).
    * **cell fill colour** = channel-importance z-score (RdBu_r diverging).

    This is the figure that should appear in the paper because it lets a
    reader cross-read "which channels matter" against "which Brodmann area
    each channel sits in" without flipping between two separate figures.

    Parameters mirror `plot_montage_channel_importance` (channel-level) plus
    the long-format `channel_to_brodmann_long` table from
    `BrodmannMapping.channel_to_ba_long` or the on-disk
    `research/xai/atlas/channel_to_brodmann.csv`.
    """
    primary = _primary_ba_per_channel(channel_to_brodmann_long)

    with plt.rc_context(_RC):
        ch = result.channel_importance_mean
        z = (ch - ch.mean()) / (ch.std(ddof=0) + 1e-12)

        grid = np.full(GRID_SHAPE, np.nan, dtype=np.float64)
        for i, (r, c) in enumerate(GRID_POS):
            grid[r, c] = z[i]

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.set_xlim(-0.6, GRID_SHAPE[1] - 0.4)
        ax.set_ylim(GRID_SHAPE[0] - 0.4, -0.6)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 1e-6)
        cmap = plt.get_cmap("RdBu_r")
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

        used_bas: dict[str, str] = {}
        for i, (r, c) in enumerate(GRID_POS):
            ba, hemi, _ = primary[i]
            border = _ba_colour(ba)
            used_bas[ba] = border
            fill = cmap(norm(z[i]))
            ax.add_patch(plt.Rectangle(
                (c - 0.45, r - 0.45), 0.9, 0.9,
                facecolor=fill, edgecolor=border, linewidth=3.0, alpha=0.95))
            txt_color = "white" if abs(z[i]) > 0.6 * vmax else "black"
            short_ba = ba.replace("Brodmann.", "BA")
            ax.text(c, r - 0.22, CHANNEL_NAMES[i],
                    ha="center", va="center", fontsize=6.5,
                    color=txt_color, fontweight="bold")
            ax.text(c, r + 0.05, f"{z[i]:+.2f}",
                    ha="center", va="center", fontsize=7.5,
                    color=txt_color, fontweight="bold")
            ax.text(c, r + 0.27, f"{short_ba}-{hemi}",
                    ha="center", va="center", fontsize=6.5, color=txt_color)

        ax.set_title(title or _default_title(
            result, "Channel importance × Brodmann area (border = BA, fill = z-score)"))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.65, label="z-score (channel importance)",
                     location="right", pad=0.04)

        def _ba_sort_key(name: str) -> tuple[int, str]:
            try:
                return (int(name.replace("Brodmann.", "")), name)
            except ValueError:
                return (10**6, name)
        legend_handles = [
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor="white", markeredgecolor=col,
                       markersize=12, markeredgewidth=2.5)
            for ba, col in sorted(used_bas.items(), key=lambda kv: _ba_sort_key(kv[0]))
        ]
        legend_labels = [
            ba.replace("Brodmann.", "BA ") if ba.startswith("Brodmann.") else ba
            for ba, _ in sorted(used_bas.items(), key=lambda kv: _ba_sort_key(kv[0]))
        ]
        # Reserve bottom margin so the legend sits below the grid, not on top of it.
        fig.subplots_adjust(bottom=0.18)
        ax.legend(legend_handles, legend_labels,
                  loc="upper center", bbox_to_anchor=(0.5, -0.04),
                  frameon=False, fontsize=8, ncol=len(legend_handles),
                  title="Brodmann area (border colour)",
                  title_fontsize=8.5)
        return _save_both(fig, Path(save_dir), stem)


def plot_region_pair_heatmap(
    region_pair: np.ndarray,
    region_keys: list[tuple[str, str]],
    save_dir: str | Path,
    *,
    stem: str = "fig_region_pair_heatmap",
    title: str | None = None,
) -> Path:
    """N_BA × N_BA heatmap of `region_pair_matrix.npy`. Diagonal blanked.

    Rows/cols are taken in the order of `region_keys` (the natural order
    produced by `aggregate_to_regions`).
    """
    M = np.asarray(region_pair).copy().astype(np.float64)
    if M.shape[0] != M.shape[1] or M.shape[0] != len(region_keys):
        raise ValueError(
            f"region_pair shape {M.shape} does not match {len(region_keys)} keys"
        )
    np.fill_diagonal(M, np.nan)
    n = M.shape[0]
    labels = [
        f"{ba.replace('Brodmann.', 'BA ')}-{h}" if ba.startswith("Brodmann.") else f"{ba}-{h}"
        for ba, h in region_keys
    ]

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(0.50 * n + 4.0, 0.50 * n + 3.5))
        vmax = float(np.nanmax(M)) if np.isfinite(np.nanmax(M)) else 1.0
        im = ax.imshow(M, cmap="magma_r", vmin=0.0, vmax=vmax, aspect="equal")
        ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=60, fontsize=8, ha="right")
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title or "Region-pair importance (Brodmann × Brodmann)")
        plt.colorbar(im, ax=ax, shrink=0.85, label="region-pair importance")
        return _save_both(fig, Path(save_dir), stem)
