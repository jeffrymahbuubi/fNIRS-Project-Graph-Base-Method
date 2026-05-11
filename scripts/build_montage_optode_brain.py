"""Build fig8 — side-by-side optode diagram (Wang-style) + mne.viz.Brain view.

Panel A (this script): rectangular 2D optode diagram with S/D dots, S-D line
segments labelled 1-23 at midpoints. Source = red, Detector = blue. Source of
truth = `data/brainproducts-RNP-BA-128-custom.elc` via `src/xai/atlas.py:parse_elc`.

Panel B (separate): mne.viz.Brain('fsaverage') rendered in Jupyter, saved as
PNG, then composited here. See docstring of `composite_figure` for the input.

Outputs:
    research/paper-materials/figures/_intermediate/panel_A.{png,svg}
    research/paper-materials/figures/_intermediate/panel_B_brain.png  (built in JN)
    research/paper-materials/figures/fig8_montage_optode_brain.{png,svg}
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import numpy as np
from matplotlib import rcParams

# Project imports
import sys
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.xai.atlas import parse_elc  # noqa: E402
from src.xai.channels import CHANNEL_NAMES  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

ELC_PATH = REPO / "data" / "brainproducts-RNP-BA-128-custom.elc"
FIG_DIR = REPO / "research" / "paper-materials" / "figures"
INTERM_DIR = FIG_DIR / "_intermediate"

SOURCE_COLOR = "red"     # pure red (matches user's reference image)
DETECTOR_COLOR = "blue"  # pure blue (matches user's reference image)
BAR_COLOR = "#DCE3EA"       # soft cool gray with slight blue tint
BAR_EDGE_COLOR = "#B7C2CC"  # subtle bar border (1pt)
BAR_LINEWIDTH = 16          # balanced merge — thick enough to host channel numbers, thin enough not to dominate
CHANNEL_TEXT_COLOR = "#1A3A6E"  # rich navy — high contrast on the soft gray bar
CHANNEL_FONT_SIZE = 11
LABEL_FONT_SIZE = 11      # match channel-number font size (per user request)
CARD_BG_COLOR = "#FAFBFC"   # off-white card background
CARD_EDGE_COLOR = "#E5E9EE"

# Hard-coded row assignment AFTER 180° rotation of the sensor_arr.png layout
# (flip_y=True AND flip_x=True applied in load_optode_xy). Result:
#   - anterior optodes (S1/S2/D1/D2) sit at the BOTTOM of the figure
#   - posterior optodes (S6..D8) at the TOP
#   - subject's LEFT appears on viewer's RIGHT
# Labels are placed AWAY from the optode cluster (outside).
ROW_OF_OPTODE: Dict[str, str] = {
    "S1": "bottom", "S2": "bottom", "D1": "bottom", "D2": "bottom",
    "S3": "middle", "S4": "middle", "S5": "middle",
    "D3": "middle", "D4": "middle", "D5": "middle",
    "S6": "top",    "S7": "top",    "S8": "top",
    "D6": "top",    "D7": "top",    "D8": "top",
}

# Per-channel-number label offset overrides (mm) when default midpoint
# placement collides with an adjacent optode. Keys are 1-indexed channel
# numbers, values are (dx, dy) added to the midpoint position.
CHANNEL_LABEL_OFFSET: Dict[int, Tuple[float, float]] = {}


def _set_style() -> None:
    # Modern sans-serif stack (system) replaces the previous Times New Roman.
    rcParams["font.family"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
    rcParams["font.size"] = 10
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["savefig.dpi"] = 300
    rcParams["figure.dpi"] = 150


# --------------------------------------------------------------------------- #
# Geometry helpers                                                            #
# --------------------------------------------------------------------------- #

def load_optode_xy(elc_path: Path,
                   flip_y: bool = True,
                   flip_x: bool = True) -> Dict[str, Tuple[float, float]]:
    """Return ``{label: (x, y)}`` for S1-S8 and D1-D8 in mm.

    ELC stores positions in head-CTF mm: +x right, +y anterior, +z superior.
    We drop z and return (x, y).

    Defaults give a 180° rotation of the sensor_arr.png ground-truth layout
    (``flip_y=True`` AND ``flip_x=True``), i.e. forehead at the BOTTOM and
    subject's LEFT on viewer's RIGHT. This matches the rostral view of
    fig_montage_brain.png so the two panels read naturally side-by-side.

    Other combinations:
    - flip_y=False, flip_x=False → sensor_arr.png orientation (forehead UP,
      neurological convention, subject's left on viewer's left).
    - flip_y=True,  flip_x=False → vertical mirror only.
    - flip_y=False, flip_x=True  → horizontal mirror only (face-on, forehead
      still UP).
    """
    elc = parse_elc(elc_path)
    raw: Dict[str, Tuple[float, float]] = {}
    for label, pos in zip(elc.labels, elc.positions_mm):
        if label.startswith(("S", "D")) and len(label) == 2 and label[1].isdigit():
            raw[label] = (float(pos[0]), float(pos[1]))
    out = raw
    if flip_y:
        ys = [y for _, y in out.values()]
        y_mid = (max(ys) + min(ys)) / 2.0
        out = {lbl: (x, 2.0 * y_mid - y) for lbl, (x, y) in out.items()}
    if flip_x:
        xs = [x for x, _ in out.values()]
        x_mid = (max(xs) + min(xs)) / 2.0
        out = {lbl: (2.0 * x_mid - x, y) for lbl, (x, y) in out.items()}
    return out


def parse_channel_name(ch: str) -> Tuple[str, str]:
    src, det = ch.split("_")
    return src, det


# --------------------------------------------------------------------------- #
# Panel A — optode diagram                                                    #
# --------------------------------------------------------------------------- #

def build_optode_panel(ax: plt.Axes,
                       positions: Dict[str, Tuple[float, float]],
                       channel_names=CHANNEL_NAMES,
                       panel_letter: str | None = None) -> None:
    """Draw Wang-style optode layout on the given axis."""

    # Draw soft S-D connector bars (flat, no shadows — so all bars read as
    # one continuous mesh in the Image #4 style; rounded caps merge into a
    # hub at each optode).
    for ch_idx, ch in enumerate(channel_names, start=1):
        src, det = parse_channel_name(ch)
        x1, y1 = positions[src]
        x2, y2 = positions[det]
        ax.plot([x1, x2], [y1, y2], "-", color=BAR_COLOR,
                linewidth=BAR_LINEWIDTH,
                solid_capstyle="round", zorder=1.5)
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx_ov, dy_ov = CHANNEL_LABEL_OFFSET.get(ch_idx, (0.0, 0.0))
        ax.text(xm + dx_ov, ym + dy_ov, str(ch_idx), color=CHANNEL_TEXT_COLOR,
                fontsize=CHANNEL_FONT_SIZE, ha="center", va="center",
                fontweight="bold", zorder=3)

    # S/D label placement uses the hard-coded row map (avoids the percentile
    # mis-classification of optodes near the row boundary, e.g. D7).
    ys_all = np.array([p[1] for p in positions.values()])
    y_offset = 0.055 * float(np.ptp(ys_all))  # tight to the dot (~2.7 mm)

    # Plot source/detector dots — flat, no shadows (Image #4 style).
    for label, (x, y) in positions.items():
        is_src = label.startswith("S")
        color = SOURCE_COLOR if is_src else DETECTOR_COLOR
        ax.plot(x, y, "o", color=color, markersize=11,
                markeredgecolor=color, markeredgewidth=0, zorder=4)

        row = ROW_OF_OPTODE[label]
        if row == "top":
            dy, va = +y_offset, "bottom"
        elif row == "bottom":
            dy, va = -y_offset, "top"
        else:  # middle row: label above
            dy, va = +y_offset, "bottom"
        ax.text(x, y + dy, label, color=color, fontsize=LABEL_FONT_SIZE,
                ha="center", va=va, fontweight="semibold", zorder=5)

    # Legend handles (constructed manually so colors match the dots).
    src_handle = plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=SOURCE_COLOR, markersize=10,
                            markeredgecolor="white", label="Source")
    det_handle = plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=DETECTOR_COLOR, markersize=10,
                            markeredgecolor="white", label="Detector")
    ax.legend(handles=[src_handle, det_handle], loc="upper right",
              frameon=True, fontsize=10)

    # Make the axes "anatomical" — head-front-up, equal aspect, hide ticks.
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if panel_letter is not None:
        ax.text(-0.02, 1.02, panel_letter, transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="top", ha="left")


# --------------------------------------------------------------------------- #
# Standalone builder for Panel A (matplotlib-only; runs without Jupyter)      #
# --------------------------------------------------------------------------- #

def build_optode_layout_figure(out_dir: Path = FIG_DIR,
                               stem: str = "fig_optode_layout") -> Path:
    """Render the optode layout as its OWN figure (no composite).

    Saves both PNG (300 dpi) and SVG to ``out_dir``. Returns the PNG path.
    """
    _set_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    positions = load_optode_xy(ELC_PATH)
    if len(positions) != 16:
        raise RuntimeError(
            f"Expected 16 optodes (S1-S8 + D1-D8), got {len(positions)}: "
            f"{sorted(positions)}"
        )

    fig, ax = plt.subplots(figsize=(10.0, 5.6))

    # Compute extent first so we can draw the card behind the layout.
    xs = np.array([p[0] for p in positions.values()])
    ys = np.array([p[1] for p in positions.values()])
    x_pad = 0.07 * float(np.ptp(xs))
    y_pad = 0.16 * float(np.ptp(ys))
    xlim = (xs.min() - x_pad, xs.max() + x_pad)
    ylim = (ys.min() - y_pad, ys.max() + y_pad)

    # No background card — clean white background, optode area only.
    build_optode_panel(ax, positions)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Modernized legend — rounded corners, sans-serif.
    src_handle = plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=SOURCE_COLOR, markersize=11,
                            markeredgecolor=SOURCE_COLOR, markeredgewidth=0,
                            label="Source")
    det_handle = plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=DETECTOR_COLOR, markersize=11,
                            markeredgecolor=DETECTOR_COLOR, markeredgewidth=0,
                            label="Detector")
    # Legend stacked VERTICALLY (Source above Detector) outside the axes.
    leg = ax.legend(handles=[src_handle, det_handle],
                    loc="lower right",
                    bbox_to_anchor=(1.0, -0.06),
                    ncol=1,
                    frameon=True, fontsize=12,
                    fancybox=True, framealpha=0.95,
                    edgecolor=CARD_EDGE_COLOR,
                    labelspacing=0.6)
    leg.get_frame().set_linewidth(0.8)

    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


# --------------------------------------------------------------------------- #
# Compositing — Panel A + B (called after Panel B is rendered in Jupyter)     #
# --------------------------------------------------------------------------- #

def composite_figure(panel_b_png: Path,
                     out_dir: Path = FIG_DIR,
                     stem: str = "fig8_montage_optode_brain") -> Path:
    """Build the final side-by-side figure.

    Reads ``panel_b_png`` (the mne.viz.Brain screenshot from Jupyter) and
    composes it with Panel A into a single 1×2 figure with subplot labels.
    """
    _set_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not panel_b_png.exists():
        raise FileNotFoundError(
            f"Panel B image not found at {panel_b_png}. "
            "Build it in Jupyter first."
        )

    positions = load_optode_xy(ELC_PATH)

    # Compute Panel A's data aspect (width / height) so we can size the
    # subplots proportionally.
    xs_all = np.array([p[0] for p in positions.values()])
    ys_all = np.array([p[1] for p in positions.values()])
    x_pad_A = 0.07 * float(np.ptp(xs_all))
    y_pad_A = 0.20 * float(np.ptp(ys_all))   # extra room for above/below labels
    A_width = float(np.ptp(xs_all)) + 2 * x_pad_A
    A_height = float(np.ptp(ys_all)) + 2 * y_pad_A

    import matplotlib.image as mpimg
    brain_img = mpimg.imread(str(panel_b_png))
    B_h, B_w = brain_img.shape[:2]

    # Width ratio matches actual content aspect at equal panel height.
    A_panel_w_at_h1 = A_width / A_height
    B_panel_w_at_h1 = B_w / B_h
    total_w = A_panel_w_at_h1 + B_panel_w_at_h1
    fig_h = 6.0
    fig_w = fig_h * total_w * 1.05  # small extra for spacing
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [A_panel_w_at_h1, B_panel_w_at_h1]},
    )

    build_optode_panel(axA, positions, panel_letter="A")
    axA.set_xlim(xs_all.min() - x_pad_A, xs_all.max() + x_pad_A)
    axA.set_ylim(ys_all.min() - y_pad_A, ys_all.max() + y_pad_A)

    axB.imshow(brain_img)
    axB.set_xticks([]); axB.set_yticks([])
    for spine in axB.spines.values():
        spine.set_visible(False)
    axB.text(-0.02, 1.02, "B", transform=axB.transAxes,
             fontsize=22, fontweight="bold", va="top", ha="left")

    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    args = p.parse_args()
    out = build_optode_layout_figure()
    print(f"Optode-layout figure → {out}")


if __name__ == "__main__":
    main()
