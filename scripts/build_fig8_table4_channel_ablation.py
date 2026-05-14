"""Build Table 4 (channel-ablation Δ-F1 matrix) + Figure 8 (K-consistency chart).

P1.7 deliverable. Consumes the 24-cell channel-ablation sweep and emits one figure
+ one table (CSV + Markdown) for the paper.

Inputs
------
- research/experiments/20260513/{mt2,mt4}/ST_GATv2_GNG_{hbo,hbr,hbt}_loso_mt{2,4}_noaug{,_K{08,12,16}}_2026051{3,4}/*_loso_overall.pkl

For HbT mt4 K=23 specifically: this script uses the 20260514 anomaly-check re-run
(F1=0.7860, bit-identical to 20260509 baseline) because the 20260513 in-batch run
(F1=0.8413) was confirmed as a non-deterministic outlier. See CHANNEL_ABLATION_RESULTS.md §6.2.

Outputs
-------
- research/paper-materials/figures/fig8_k_consistency.{png,svg}
- research/paper-materials/stats/table4_channel_ablation_delta_f1.csv
- research/paper-materials/stats/table4_channel_ablation_delta_f1.md

Layout (Figure 8 — 1 row × 2 cols)
  Left:  heatmap (3 K rows × 6 chromo×mt cols), cells coloured by Δ-F1 (pp) and
         annotated with the F1 (Δ-F1) numeric pair. Highlights which K wins each cell.
  Right: bar chart of mean Δ-F1 per K across the 6 configurations. Highlights K=12 as the
         overall most reliably beneficial subset.

Run
---
    src/.venv/bin/python scripts/build_fig8_table4_channel_ablation.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = ROOT / "research" / "experiments" / "20260513"
ANOMALY_RERUN = ROOT / "research" / "experiments" / "anomaly_check_hbt_mt4_k23"
FIG_OUT = ROOT / "research" / "paper-materials" / "figures"
STAT_OUT = ROOT / "research" / "paper-materials" / "stats"

CHROMOS = ["hbo", "hbr", "hbt"]
MTS = [2, 4]
KS = [23, 16, 12, 8]


def _find_overall_pkl(mt: int, hb: str, k: int) -> Path:
    """Locate the `*_loso_overall.pkl` for a given (mt, chromo, K) cell.

    HbT mt4 K=23 is special-cased: use the 20260514 anomaly-check re-run
    (canonical baseline) rather than the 20260513 in-batch outlier.
    """
    if hb == "hbt" and mt == 4 and k == 23:
        # canonical baseline (reproduces 20260509 bit-identically)
        run_dir = ANOMALY_RERUN / "20260514" / f"ST_GATv2_GNG_hbt_loso_mt4_noaug_20260514"
        return next(run_dir.glob("*_loso_overall.pkl"))

    mt_dir = ABLATION_ROOT / f"mt{mt}"
    k_token = "" if k == 23 else f"_K{k:02d}"
    # Try both 20260513 and 20260514 date suffixes
    for date in ("20260513", "20260514"):
        candidate = mt_dir / f"ST_GATv2_GNG_{hb}_loso_mt{mt}_noaug{k_token}_{date}"
        if candidate.exists():
            pkls = sorted(candidate.glob("*_loso_overall.pkl"))
            if pkls:
                return pkls[0]
    raise FileNotFoundError(f"No overall pickle for hb={hb} mt={mt} K={k}")


def collect_results() -> pd.DataFrame:
    """Load F1/Acc/Sens/Spec for every (chromo, mt, K) cell."""
    rows = []
    for hb in CHROMOS:
        for mt in MTS:
            for k in KS:
                pkl = _find_overall_pkl(mt, hb, k)
                with open(pkl, "rb") as f:
                    data = pickle.load(f)
                rows.append({
                    "chromo": hb.upper(),
                    "mt": mt,
                    "K": k,
                    "F1": float(data["overall_f1"]),
                    "Acc": float(data["overall_accuracy"]),
                    "Sens": float(data["overall_sensitivity"]),
                    "Spec": float(data["overall_specificity"]),
                    "source": str(pkl.relative_to(ROOT)),
                })
    return pd.DataFrame(rows)


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell Δ-F1 vs the matching (chromo, mt) K=23 baseline."""
    df = df.copy()
    baseline = df[df["K"] == 23].set_index(["chromo", "mt"])["F1"]
    df["F1_baseline"] = df.apply(lambda r: baseline.loc[(r["chromo"], r["mt"])], axis=1)
    df["dF1_pp"] = (df["F1"] - df["F1_baseline"]) * 100
    return df


def build_table4(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Build the wide-form Table 4 + a markdown rendering."""
    # Wide-form: rows = (chromo, mt), cols = K values, cells = "F1 (Δpp)"
    wide = df.pivot_table(index=["chromo", "mt"], columns="K", values="F1").reindex(
        columns=[23, 16, 12, 8]
    )
    wide_delta = df.pivot_table(index=["chromo", "mt"], columns="K", values="dF1_pp").reindex(
        columns=[23, 16, 12, 8]
    )

    def _fmt_cell(f1, dpp, is_best_in_row, is_baseline):
        if is_baseline:
            return f"{f1:.4f}"
        marker = "**" if is_best_in_row else ""
        return f"{marker}{f1:.4f} ({dpp:+.2f} pp){marker}"

    # Determine the best K per row (excluding K=23)
    nonbase_cols = [16, 12, 8]
    best_per_row = wide[nonbase_cols].idxmax(axis=1)

    md_lines = [
        "## Table 4. Channel-ablation Δ-F1 vs K=23 baseline (ST × LOSO, 24 cells)",
        "",
        "Each cell shows the held-out test F1 with the Δ vs the same (chromo × mt) K=23 baseline in parentheses.",
        "Bold = best non-baseline K within row. ★ = highest mean Δ-F1 across configurations.",
        "† HbT mt=4 K=23 uses the 2026-05-14 anomaly-check re-run (canonical 20260509 baseline reproduced bit-identically); the 20260513 in-batch value of 0.8413 was a non-deterministic outlier — see `CHANNEL_ABLATION_RESULTS.md §6.2`.",
        "",
        "| Chromo | mt | K=23 (baseline) | K=16 | K=12 | K=8 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (chromo, mt), row in wide.iterrows():
        delta_row = wide_delta.loc[(chromo, mt)]
        best_k = best_per_row.loc[(chromo, mt)]
        base_str = f"{row[23]:.4f}"
        if chromo == "HBT" and mt == 4:
            base_str += " †"
        cells = [base_str]
        for k in nonbase_cols:
            cells.append(_fmt_cell(row[k], delta_row[k], k == best_k, False))
        md_lines.append(f"| {chromo} | {mt} | " + " | ".join(cells) + " |")

    # Summary rows
    mean_delta = df[df["K"] != 23].groupby("K")["dF1_pp"].mean()
    sign_consistency = df[df["K"] != 23].assign(positive=lambda d: d["dF1_pp"] > 0)
    sign_counts = sign_consistency.groupby("K")["positive"].sum().astype(int)
    n_total = sign_consistency.groupby("K").size()

    best_mean_k = int(mean_delta.idxmax())

    md_lines.append("| **Mean Δ-F1** | | — | " + " | ".join(
        f"**{mean_delta[k]:+.2f} pp**" + (" ★" if k == best_mean_k else "")
        for k in nonbase_cols
    ) + " |")
    md_lines.append("| **Sign-consistency** | | — | " + " | ".join(
        f"{sign_counts[k]}/{n_total[k]}" for k in nonbase_cols
    ) + " |")
    md_lines.append("")
    md_lines.append(
        f"**Reading the K-consistency claim.** K={best_mean_k} has the highest mean Δ-F1 "
        f"({mean_delta[best_mean_k]:+.2f} pp) of any tested K, and matches the other K values on "
        f"sign-consistency (5/6 cells positive across the 3 chromophores × 2 trial-cap regimes). "
        f"HbR mt=4 is the single anti-parsimony cell for K={best_mean_k}; all other (chromo × mt) "
        f"configurations show positive Δ-F1 at K={best_mean_k}. The locked paper headline "
        f"(ST × HbO × LOSO × mt=2 × K={best_mean_k} = F1 0.8529) is the highest-F1 K={best_mean_k} cell "
        f"at the regime where the differential XAI was derived."
    )
    md_lines.append("")
    md_lines.append("**Source:** `research/experiments/20260513/CHANNEL_ABLATION_RESULTS.md` (full breakdown).")

    return wide, "\n".join(md_lines)


def build_figure(df: pd.DataFrame, fig_path_png: Path, fig_path_svg: Path) -> None:
    """Render Figure 8 — heatmap (left) + mean-bar (right)."""
    nonbase = df[df["K"] != 23].copy()
    # Pivot to a (K=8,12,16) x 6-cell matrix for the heatmap
    nonbase["cell"] = nonbase.apply(lambda r: f"{r['chromo']}\nmt{r['mt']}", axis=1)
    cell_order = [
        "HBO\nmt2", "HBR\nmt2", "HBT\nmt2",
        "HBO\nmt4", "HBR\nmt4", "HBT\nmt4",
    ]
    k_order = [16, 12, 8]  # top-to-bottom in the heatmap
    pivot = nonbase.pivot_table(index="K", columns="cell", values="dF1_pp").reindex(
        index=k_order, columns=cell_order
    )
    f1_pivot = nonbase.pivot_table(index="K", columns="cell", values="F1").reindex(
        index=k_order, columns=cell_order
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.4, 1.0]}
    )

    # ---- Left: heatmap ----
    ax = axes[0]
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(pivot.values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(cell_order)))
    ax.set_xticklabels(cell_order, fontsize=9)
    ax.set_yticks(range(len(k_order)))
    ax.set_yticklabels([f"K={k}" for k in k_order], fontsize=10)
    ax.set_title("Channel-ablation Δ-F1 vs K=23 baseline (pp)", fontsize=11, pad=8)

    # Annotate each cell with F1 (Δ pp)
    for i, k in enumerate(k_order):
        for j, cell in enumerate(cell_order):
            delta = pivot.loc[k, cell]
            f1 = f1_pivot.loc[k, cell]
            txt = f"{f1:.3f}\n({delta:+.2f})"
            # Choose text colour based on cell darkness
            color = "white" if abs(delta) > vmax * 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    # Highlight the K=12 row with a black rectangle
    k12_row = k_order.index(12)
    ax.add_patch(plt.Rectangle(
        (-0.5, k12_row - 0.5), len(cell_order), 1.0,
        fill=False, edgecolor="black", lw=2.0, zorder=10,
    ))

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Δ-F1 (percentage points)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ---- Right: mean Δ-F1 bar ----
    ax = axes[1]
    mean_dpp = nonbase.groupby("K")["dF1_pp"].mean().reindex(k_order)
    sign_count = nonbase.assign(pos=lambda d: d["dF1_pp"] > 0).groupby("K")["pos"].sum().reindex(k_order).astype(int)
    n_total = nonbase.groupby("K").size().reindex(k_order).astype(int)

    bar_colors = [
        ("#377eb8" if k == 12 else "#bdbdbd") for k in k_order
    ]
    bars = ax.barh(
        [f"K={k}" for k in k_order], mean_dpp.values,
        color=bar_colors, edgecolor="black", linewidth=0.6,
    )
    ax.invert_yaxis()  # match heatmap row order (K=16 on top, K=8 at bottom)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Mean Δ-F1 across 6 configs (pp)", fontsize=9)
    ax.set_title("K-consistency summary", fontsize=11, pad=8)
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, max(mean_dpp.values) * 1.4)

    # Annotate each bar with mean + sign-consistency
    for bar, k in zip(bars, k_order):
        v = bar.get_width()
        ax.text(
            v + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f} pp   ({sign_count[k]}/{n_total[k]} ↑)",
            va="center", ha="left", fontsize=9,
        )

    # Best K annotation
    best_k = int(mean_dpp.idxmax())
    fig.suptitle(
        f"Figure 8. K-consistency analysis of channel-ablation Δ-F1 (24 cells). "
        f"K={best_k} is the most beneficial subset across configurations.",
        fontsize=11, y=1.02,
    )

    plt.tight_layout()
    fig.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    STAT_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading 24 channel-ablation cells...")
    df = collect_results()
    df = compute_deltas(df)
    assert len(df) == 24, f"expected 24 rows, got {len(df)}"

    # Save the raw CSV (long form, 24 rows)
    csv_long = STAT_OUT / "table4_channel_ablation_delta_f1.csv"
    df.to_csv(csv_long, index=False, float_format="%.4f")
    print(f"  wrote {csv_long.relative_to(ROOT)}  ({len(df)} rows)")

    # Build the wide-form table + markdown
    wide, md = build_table4(df)
    md_path = STAT_OUT / "table4_channel_ablation_delta_f1.md"
    md_path.write_text(md + "\n")
    print(f"  wrote {md_path.relative_to(ROOT)}")

    # Render Figure 8
    fig_png = FIG_OUT / "fig8_k_consistency.png"
    fig_svg = FIG_OUT / "fig8_k_consistency.svg"
    build_figure(df, fig_png, fig_svg)
    print(f"  wrote {fig_png.relative_to(ROOT)}")
    print(f"  wrote {fig_svg.relative_to(ROOT)}")

    # Quick summary print
    print()
    print("Summary (mean Δ-F1 across 6 configs):")
    print(df[df["K"] != 23].groupby("K")["dF1_pp"].agg(["mean", lambda s: (s > 0).sum(), "count"])
          .rename(columns={"<lambda_0>": "n_pos", "count": "n_total"}).to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
