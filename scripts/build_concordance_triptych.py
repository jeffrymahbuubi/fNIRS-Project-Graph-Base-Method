"""Build P1.2 saliency-statistics concordance figure + Spearman rho table.

Inputs
------
- research/xai/st/{hbo,hbr}/loso/mt2/native/node_importance.csv
- research/xai/st/{hbo,hbr}/loso/mt2/native/temporal_attention.csv
- src/notebook/statistical-analysis/02_brain_activation/results_brain_activation_stats.csv
- src/notebook/statistical-analysis/06_glm_hrf/results_canonical_hrf_beta.csv

Outputs
-------
- research/paper-materials/figures/concordance_triptych.{png,svg}
- research/paper-materials/stats/concordance_rho_table.csv
- research/paper-materials/stats/concordance_rho_table.md

Layout (3 rows x 2 cols)
  Row A: ST attention topomap HbO | ST attention topomap HbR
  Row B: |Cohen d| topomap §02 HbO STD | |Cohen d| topomap §06 canonical-beta HbO
  Row C: temporal attention HbO alpha_k | temporal attention HbR alpha_k
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
XAI_ROOT = ROOT / "research" / "xai" / "st"
STATS_ROOT = ROOT / "src" / "notebook" / "statistical-analysis"
FIG_OUT = ROOT / "research" / "paper-materials" / "figures"
STAT_OUT = ROOT / "research" / "paper-materials" / "stats"

# Canonical channel order — verbatim from src/xai/channels.py.
CHANNEL_NAMES: list[str] = [
    "S1_D1", "S1_D3", "S2_D2", "S2_D1", "S2_D5", "S3_D1", "S3_D3", "S3_D4", "S3_D6",
    "S4_D4", "S4_D5", "S4_D7", "S5_D2", "S5_D5", "S5_D8", "S6_D3", "S6_D6",
    "S7_D4", "S7_D6", "S7_D7", "S8_D5", "S8_D7", "S8_D8",
]
GRID_POS: list[tuple[int, int]] = [
    (0, 2), (1, 1), (0, 4), (0, 3), (1, 4), (1, 2), (2, 1), (2, 2), (3, 1),
    (2, 3), (2, 4), (3, 4), (1, 5), (2, 5), (3, 6), (3, 0), (4, 1),
    (3, 2), (4, 2), (4, 3), (3, 5), (4, 4), (4, 5),
]
GRID_SHAPE = (5, 7)
N_CH = len(CHANNEL_NAMES)

# Biological prior set per docs/SPEC_xai_graph.md §11 C6.
C6_PRIOR: set[str] = {"S1_D1", "S5_D5", "S3_D3", "S2_D1", "S4_D5", "S4_D7"}


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_node_importance(hb: str) -> np.ndarray:
    df = pd.read_csv(XAI_ROOT / hb / "loso" / "mt2" / "native" / "node_importance.csv")
    df = df.set_index("channel").loc[CHANNEL_NAMES]
    return df["mean"].to_numpy(dtype=np.float64)


def load_temporal(hb: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(XAI_ROOT / hb / "loso" / "mt2" / "native" / "temporal_attention.csv")
    return (
        df["mean"].to_numpy(dtype=np.float64),
        df["std"].to_numpy(dtype=np.float64),
        df["t_start_s"].to_numpy(dtype=np.float64),
        df["t_end_s"].to_numpy(dtype=np.float64),
    )


def load_stats_d(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    df = df.set_index("channel").loc[CHANNEL_NAMES]
    return df["d"].to_numpy(dtype=np.float64)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)


def topomap(ax, values, *, signed: bool, title: str,
            highlight: set[str] | None = None, cbar_label: str = "z-score"):
    grid = np.full(GRID_SHAPE, np.nan, dtype=np.float64)
    for i, (r, c) in enumerate(GRID_POS):
        grid[r, c] = values[i]

    if signed:
        vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 1e-6)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        im = ax.imshow(grid, cmap="RdBu_r", norm=norm, aspect="equal")
    else:
        vmax = float(np.nanmax(grid)) if np.isfinite(np.nanmax(grid)) else 1.0
        im = ax.imshow(grid, cmap="magma_r", vmin=0.0, vmax=vmax, aspect="equal")

    for i, (r, c) in enumerate(GRID_POS):
        v = values[i]
        is_prior = highlight is not None and CHANNEL_NAMES[i] in highlight
        label_color = "#FFD700" if is_prior else "black"
        label_weight = "bold" if is_prior else "normal"
        ax.text(c, r - 0.24, CHANNEL_NAMES[i], ha="center", va="center",
                fontsize=5.8, color=label_color, fontweight=label_weight)
        ax.text(c, r + 0.22, f"{v:+.2f}" if signed else f"{v:.2f}",
                ha="center", va="center", fontsize=6.0, color="black")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.5)
    return im


def temporal_panel(ax, mean, std, t_start, t_end, *, n: int,
                   color: str, title: str):
    centers = 0.5 * (t_start + t_end)
    ci = 1.96 * std / np.sqrt(max(n, 1))
    ax.plot(centers, mean, color=color, lw=1.5, label="alpha_k mean")
    ax.fill_between(centers, mean - ci, mean + ci, color=color, alpha=0.22,
                    label="95% CI")
    uniform = 1.0 / mean.shape[0]
    ax.axhline(uniform, color="grey", lw=0.8, ls="--",
               label=f"uniform (1/K={uniform:.3f})")
    # §06 cluster-permutation windows: early 0-7 s, late 21-32 s.
    ax.axvspan(0.0, 7.0, color="#56AED1", alpha=0.10, label="§06 early cluster")
    ax.axvspan(21.0, 32.0, color="#FFB400", alpha=0.10, label="§06 late cluster")
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Temporal attention alpha_k", fontsize=8)
    ax.set_title(title, fontsize=9.5)
    ax.legend(loc="lower center", frameon=False, fontsize=6.5, ncols=2)
    ax.tick_params(axis="both", labelsize=7)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    print("=== Loading inputs ===")
    attn_hbo = load_node_importance("hbo")
    attn_hbr = load_node_importance("hbr")
    print(f"  ST attn HbO  channel-mean range = [{attn_hbo.min():.4f}, {attn_hbo.max():.4f}]")
    print(f"  ST attn HbR  channel-mean range = [{attn_hbr.min():.4f}, {attn_hbr.max():.4f}]")

    temp_hbo = load_temporal("hbo")
    temp_hbr = load_temporal("hbr")
    print(f"  Temporal HbO alpha_k range = [{temp_hbo[0].min():.5f}, {temp_hbo[0].max():.5f}]")
    print(f"  Temporal HbR alpha_k range = [{temp_hbr[0].min():.5f}, {temp_hbr[0].max():.5f}]")
    print(f"  Uniform reference = {1.0 / temp_hbo[0].shape[0]:.5f}")

    d_02_signed = load_stats_d(STATS_ROOT / "02_brain_activation" / "results_brain_activation_stats.csv")
    d_06_signed = load_stats_d(STATS_ROOT / "06_glm_hrf" / "results_canonical_hrf_beta.csv")
    abs_d_02 = np.abs(d_02_signed)
    abs_d_06 = np.abs(d_06_signed)
    print(f"  §02 |d| range = [{abs_d_02.min():.3f}, {abs_d_02.max():.3f}]")
    print(f"  §06 |β d| range = [{abs_d_06.min():.3f}, {abs_d_06.max():.3f}]")

    # ---- Spearman ρ table ----
    pairs = [
        ("ST attn HbO  vs §02 |d| (HbO STD)",      attn_hbo, abs_d_02),
        ("ST attn HbO  vs §06 |β d| (HbO canon)",  attn_hbo, abs_d_06),
        ("ST attn HbR  vs §02 |d| (HbO STD)",      attn_hbr, abs_d_02),
        ("ST attn HbR  vs §06 |β d| (HbO canon)",  attn_hbr, abs_d_06),
        ("ST attn HbO  vs ST attn HbR",            attn_hbo, attn_hbr),
        ("§02 |d|     vs §06 |β d|",               abs_d_02, abs_d_06),
    ]
    rho_rows = []
    for name, x, y in pairs:
        rho, p = spearmanr(x, y)
        rho_rows.append({
            "comparison": name,
            "spearman_rho": float(rho),
            "p_value": float(p),
            "n_channels": N_CH,
        })
    rho_df = pd.DataFrame(rho_rows)

    # ---- top-10 inclusion counts ----
    def top_k_set(v: np.ndarray, k: int = 10) -> set[str]:
        order = np.argsort(-v)[:k]
        return {CHANNEL_NAMES[i] for i in order}

    c6_counts = {
        "ST attn HbO top-10 ∩ C6":  len(top_k_set(attn_hbo) & C6_PRIOR),
        "ST attn HbR top-10 ∩ C6":  len(top_k_set(attn_hbr) & C6_PRIOR),
        "§02 |d| top-10 ∩ C6":      len(top_k_set(abs_d_02) & C6_PRIOR),
        "§06 |β d| top-10 ∩ C6":    len(top_k_set(abs_d_06) & C6_PRIOR),
    }
    cross_overlap = {
        "ST attn HbO ∩ §02 |d|  (top-10)":  len(top_k_set(attn_hbo) & top_k_set(abs_d_02)),
        "ST attn HbO ∩ §06 |β d| (top-10)": len(top_k_set(attn_hbo) & top_k_set(abs_d_06)),
        "ST attn HbR ∩ §02 |d|  (top-10)":  len(top_k_set(attn_hbr) & top_k_set(abs_d_02)),
        "ST attn HbR ∩ §06 |β d| (top-10)": len(top_k_set(attn_hbr) & top_k_set(abs_d_06)),
    }

    # ---- write CSV + MD ----
    STAT_OUT.mkdir(parents=True, exist_ok=True)
    rho_df.to_csv(STAT_OUT / "concordance_rho_table.csv",
                  index=False, float_format="%.6f")

    md_lines: list[str] = [
        "# Concordance — XAI saliency vs HbO statistical analysis",
        "",
        "**Headline XAI cells (mt2, LOSO, native attention):**",
        "- ST × HbO × LOSO × mt2 (n_trials = 98, 52 subj)",
        "- ST × HbR × LOSO × mt2 (n_trials = 102, 55 subj)",
        "",
        "**Statistical references (HbO):**",
        "- §02 brain activation — per-channel STD, MWU, Cohen's d (`02_brain_activation/results_brain_activation_stats.csv`)",
        "- §06 GLM/HRF — per-channel canonical-β, MWU, Cohen's d (`06_glm_hrf/results_canonical_hrf_beta.csv`)",
        "",
        "## Spearman ρ over 23 channels",
        "",
        "| Comparison | ρ | p | n |",
        "|---|---:|---:|---:|",
    ]
    for r in rho_rows:
        md_lines.append(
            f"| {r['comparison']} | {r['spearman_rho']:+.3f} | {r['p_value']:.3f} | {r['n_channels']} |"
        )
    md_lines += [
        "",
        "## Top-10 inclusion vs C6 biological prior",
        "C6 = {S1_D1, S5_D5, S3_D3, S2_D1, S4_D5, S4_D7} per `docs/SPEC_xai_graph.md §11`.",
        "",
        "| Set | Count |",
        "|---|---:|",
    ]
    for k, v in c6_counts.items():
        md_lines.append(f"| {k} | {v} / 6 |")
    md_lines += [
        "",
        "## Top-10 ∩ Top-10 cross-method overlap",
        "",
        "| Comparison | Overlap |",
        "|---|---:|",
    ]
    for k, v in cross_overlap.items():
        md_lines.append(f"| {k} | {v} / 10 |")
    md_lines += [
        "",
        "## Notes",
        "- ρ is rank-based; computed via `scipy.stats.spearmanr` on the raw scalar vectors.",
        "- §02 / §06 Cohen's d are signed (HC − GAD; HC > GAD ⇒ d < 0); XAI cannot recover sign, so we compare against |d|.",
        "- C6 inclusion is by membership in the top-10 channel ranking, NOT rank-correlation.",
        f"- Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}",
    ]
    (STAT_OUT / "concordance_rho_table.md").write_text("\n".join(md_lines) + "\n")

    # ---- figure ----
    fig, axes = plt.subplots(
        3, 2, figsize=(11.5, 13.0),
        gridspec_kw={"hspace": 0.42, "wspace": 0.18, "height_ratios": [1.0, 1.0, 0.65]},
    )

    # Row A — XAI z-scored attention
    im00 = topomap(axes[0, 0], _z(attn_hbo), signed=True,
                   title="(A) ST attention z-score — HbO LOSO mt2 (52 subj, 98 trials)",
                   highlight=C6_PRIOR)
    im01 = topomap(axes[0, 1], _z(attn_hbr), signed=True,
                   title="(B) ST attention z-score — HbR LOSO mt2 (55 subj, 102 trials)",
                   highlight=C6_PRIOR)
    plt.colorbar(im00, ax=axes[0, 0], shrink=0.78, label="z-score")
    plt.colorbar(im01, ax=axes[0, 1], shrink=0.78, label="z-score")

    # Row B — statistics |d|
    im10 = topomap(axes[1, 0], abs_d_02, signed=False,
                   title="(C) |Cohen d| — §02 brain-activation HbO STD",
                   highlight=C6_PRIOR, cbar_label="|d|")
    im11 = topomap(axes[1, 1], abs_d_06, signed=False,
                   title="(D) |Cohen d| — §06 canonical-HRF β HbO",
                   highlight=C6_PRIOR, cbar_label="|d|")
    plt.colorbar(im10, ax=axes[1, 0], shrink=0.78, label="|d|")
    plt.colorbar(im11, ax=axes[1, 1], shrink=0.78, label="|d|")

    # Row C — temporal attention
    temporal_panel(axes[2, 0], *temp_hbo, n=52, color="#3066BE",
                   title="(E) Temporal attention — HbO LOSO mt2")
    temporal_panel(axes[2, 1], *temp_hbr, n=55, color="#C84B4B",
                   title="(F) Temporal attention — HbR LOSO mt2")

    # Footer / suptitle
    fig.suptitle(
        "Saliency–statistics concordance — ST attention vs HbO statistical analysis",
        fontsize=12, y=0.995,
    )
    footer = (
        f"Spearman ρ:  ST-HbO vs §02|d|={rho_rows[0]['spearman_rho']:+.2f} (p={rho_rows[0]['p_value']:.2f});  "
        f"vs §06|βd|={rho_rows[1]['spearman_rho']:+.2f} (p={rho_rows[1]['p_value']:.2f}).  "
        f"ST-HbR vs §02|d|={rho_rows[2]['spearman_rho']:+.2f};  vs §06|βd|={rho_rows[3]['spearman_rho']:+.2f}.  "
        f"§02|d| vs §06|βd|={rho_rows[5]['spearman_rho']:+.2f}.   "
        f"C6 top-10 hits — HbO:{c6_counts['ST attn HbO top-10 ∩ C6']}/6,  "
        f"HbR:{c6_counts['ST attn HbR top-10 ∩ C6']}/6,  "
        f"§02:{c6_counts['§02 |d| top-10 ∩ C6']}/6,  "
        f"§06:{c6_counts['§06 |β d| top-10 ∩ C6']}/6.   "
        f"C6 channels highlighted in yellow."
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=7.0,
             style="italic", wrap=True)

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    out_png = FIG_OUT / "concordance_triptych.png"
    out_svg = FIG_OUT / "concordance_triptych.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    # ---- console report ----
    print("\n=== Outputs ===")
    for p in (out_png, out_svg,
              STAT_OUT / "concordance_rho_table.csv",
              STAT_OUT / "concordance_rho_table.md"):
        print(f"  {p.relative_to(ROOT)}")

    print("\n=== Spearman ρ ===")
    for r in rho_rows:
        sig = " *" if r["p_value"] < 0.05 else ""
        print(f"  {r['comparison']:42s}  ρ = {r['spearman_rho']:+.3f}  "
              f"p = {r['p_value']:.3f}{sig}")

    print("\n=== Top-10 ∩ C6 biological prior ===")
    for k, v in c6_counts.items():
        print(f"  {k:38s}  {v} / 6")

    print("\n=== Top-10 ∩ Top-10 cross-method ===")
    for k, v in cross_overlap.items():
        print(f"  {k:42s}  {v} / 10")


if __name__ == "__main__":
    main()
