"""Brodmann atlas registration for the 23-channel prefrontal montage (SPEC §16, rev. 5).

Pipeline:
    parse_elc  →  compute_channel_midpoints  →  register_to_fsaverage
              →  apply_registration  →  project_to_brodmann
              →  build_channel_to_brodmann  (orchestrator, persists §16.6 outputs)
              →  aggregate_to_regions       (§16.7 — re-aggregate every
                                              node_importance.csv + channel_pair_matrix.npy)

Inputs:  data/brainproducts-RNP-BA-128-custom.elc
         (16 optodes + 3 fiducials, validated 2026-05-08; mean S–D = 33.4 mm).
Outputs: research/xai/atlas/{channel_to_brodmann.csv,
                              channel_midpoints_mni.csv,
                              registration_run.json};
         per-cell  research/xai/atlas/{arch}/{regime}/mt{N}/{
                              region_importance.csv,
                              region_pair_matrix.npy,
                              region_keys.csv}.

Acceptance: SPEC §11 C8 — see tests/xai/test_atlas_registration.py.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.xai.channels import CHANNEL_NAMES, N_CH


# --------------------------------------------------------------------------- #
# Defaults (SPEC §16.6)                                                       #
# --------------------------------------------------------------------------- #

DEFAULT_SIGMA_MM: float = 5.0
DEFAULT_RADIUS_MM: float = 10.0
DEFAULT_ATLAS_PARC: str = "PALS_B12_Brodmann"
DEFAULT_PROJECTION_DISTANCE_BOUND_MM: float = 35.0   # used by C8.b
# Empirical: with the BrainProducts BA-128 cap aligned to fsaverage via 3 fiducials,
# the 23 prefrontal channel midpoints project to the pial surface at 14–33 mm
# (mean 22.5 mm). Dorsal-frontal probes (S7_D7, S8_D7, S7_D6) are at the upper
# end because the scalp-to-cortex distance is naturally largest over those
# regions (see Tsuzuki & Dan 2014, NeuroImage 85:92-103). 35 mm leaves head-room
# without admitting off-cortex projections.


# --------------------------------------------------------------------------- #
# §16.3 — pure-Python ELC parser (no MNE dependency at this step)             #
# --------------------------------------------------------------------------- #

_OPTODE_RE = re.compile(r"^[SD][1-8]$")


@dataclass(frozen=True)
class ElcContents:
    positions_mm: np.ndarray   # shape [N, 3], head-CTF mm
    labels: List[str]          # length N; first three are LPA, Nz/Nasion, RPA
    n_optodes: int             # count of S{1..8}/D{1..8} labels
    sha256: str                # of the file's text content (for reproducibility)


def parse_elc(path: Path) -> ElcContents:
    """Parse an ASA ``.elc`` electrode file. Validates per SPEC §16.3 / C8.a:

    * ``UnitPosition mm`` header is present.
    * ``NumberPositions=N`` matches both row count and label count.
    * First three labels are LPA, Nz (or Nasion), RPA.

    Raises ``ValueError`` on any violation.
    """
    text = Path(path).read_text()
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    lines = [l.rstrip() for l in text.splitlines()]

    n_pos: Optional[int] = None
    pos_start: Optional[int] = None
    label_start: Optional[int] = None
    has_mm = False
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith("UnitPosition"):
            if "mm" not in s.lower():
                raise ValueError(f"expected UnitPosition mm, got: {l!r}")
            has_mm = True
        elif s.startswith("NumberPositions"):
            try:
                n_pos = int(s.split("=", 1)[1].strip())
            except (IndexError, ValueError) as e:
                raise ValueError(f"malformed NumberPositions header: {l!r}") from e
        elif s == "Positions":
            pos_start = i + 1
        elif s == "Labels":
            label_start = i + 1

    if not has_mm:
        raise ValueError("ELC: UnitPosition mm header missing")
    if n_pos is None:
        raise ValueError("ELC: NumberPositions header missing")
    if pos_start is None:
        raise ValueError("ELC: Positions section missing")
    if label_start is None:
        raise ValueError("ELC: Labels section missing")

    if pos_start + n_pos > len(lines):
        raise ValueError(
            f"ELC: NumberPositions={n_pos} but only {len(lines) - pos_start} "
            f"position rows after header"
        )

    positions = np.empty((n_pos, 3), dtype=np.float64)
    for k in range(n_pos):
        parts = lines[pos_start + k].split()
        if len(parts) < 3:
            raise ValueError(
                f"ELC line {pos_start + k}: expected 3 floats, got {lines[pos_start + k]!r}"
            )
        positions[k] = [float(parts[0]), float(parts[1]), float(parts[2])]

    labels: List[str] = []
    li = label_start
    while len(labels) < n_pos and li < len(lines):
        s = lines[li].strip()
        if s:
            labels.append(s)
        li += 1
    if len(labels) != n_pos:
        raise ValueError(
            f"ELC: expected {n_pos} labels, found {len(labels)} non-empty lines"
        )

    if len(labels) < 3:
        raise ValueError("ELC: expected 3 fiducials at the start of Labels")
    fids_lower = [labels[0].lower(), labels[1].lower(), labels[2].lower()]
    if fids_lower[0] != "lpa":
        raise ValueError(f"ELC: first label must be LPA, got {labels[0]!r}")
    if fids_lower[1] not in ("nz", "nasion"):
        raise ValueError(f"ELC: second label must be Nz/Nasion, got {labels[1]!r}")
    if fids_lower[2] != "rpa":
        raise ValueError(f"ELC: third label must be RPA, got {labels[2]!r}")

    n_optodes = sum(1 for L in labels[3:] if _OPTODE_RE.match(L))

    return ElcContents(
        positions_mm=positions,
        labels=labels,
        n_optodes=n_optodes,
        sha256=sha,
    )


# --------------------------------------------------------------------------- #
# §16.4 — channel midpoints                                                   #
# --------------------------------------------------------------------------- #

def compute_channel_midpoints(
    elc: ElcContents,
    channel_names: Sequence[str] = CHANNEL_NAMES,
) -> Dict[str, np.ndarray]:
    """Return ``{channel_name: midpoint_mm}`` with midpoint = (S + D) / 2."""
    label_to_idx = {L: i for i, L in enumerate(elc.labels)}
    out: Dict[str, np.ndarray] = {}
    for ch in channel_names:
        s, d = ch.split("_")
        if s not in label_to_idx:
            raise KeyError(f"optode {s!r} not in ELC labels")
        if d not in label_to_idx:
            raise KeyError(f"optode {d!r} not in ELC labels")
        out[ch] = 0.5 * (
            elc.positions_mm[label_to_idx[s]] + elc.positions_mm[label_to_idx[d]]
        )
    return out


def compute_sd_distances(
    elc: ElcContents,
    channel_names: Sequence[str] = CHANNEL_NAMES,
) -> Dict[str, float]:
    """Per-channel source-detector Euclidean distance (mm)."""
    label_to_idx = {L: i for i, L in enumerate(elc.labels)}
    return {
        ch: float(
            np.linalg.norm(
                elc.positions_mm[label_to_idx[ch.split("_")[0]]]
                - elc.positions_mm[label_to_idx[ch.split("_")[1]]]
            )
        )
        for ch in channel_names
    }


# --------------------------------------------------------------------------- #
# §16.5 — head-CTF → MRI registration (rigid Procrustes via fiducials)        #
# --------------------------------------------------------------------------- #

@dataclass
class AtlasRegistration:
    trans_head_to_mri_mm: np.ndarray    # [4, 4] affine, mm in / mm out
    fiducial_residual_rmse_mm: float
    fsaverage_dir: str                  # absolute path to the fsaverage subject dir
    elc_sha256: str
    sigma_mm: float
    radius_mm: float


def _resolve_fsaverage(
    subjects_dir: Optional[Path],
    subject: str = "fsaverage",
) -> Tuple[Path, str]:
    """Return ``(subjects_dir, subject_dir_path_str)`` — fetching fsaverage if
    not cached yet."""
    import mne   # noqa: F401  (lazy)
    if subjects_dir is None:
        fs_root = mne.datasets.fetch_fsaverage(verbose="ERROR")
        subjects_dir = Path(fs_root).parent
    subjects_dir = Path(subjects_dir)
    return subjects_dir, str(subjects_dir / subject)


def register_to_fsaverage(
    elc: ElcContents,
    subjects_dir: Optional[Path] = None,
    subject: str = "fsaverage",
    sigma_mm: float = DEFAULT_SIGMA_MM,
    radius_mm: float = DEFAULT_RADIUS_MM,
) -> AtlasRegistration:
    """Solve a rigid-body Procrustes from ELC head-CTF → fsaverage MRI (mm).

    Uses the three fiducials only (LPA, Nz/Nasion, RPA). ``scale=False`` —
    we trust the BrainProducts head model's size.
    """
    import mne

    subjects_dir, fs_subject_dir = _resolve_fsaverage(subjects_dir, subject)

    fs_fids = mne.coreg.get_mni_fiducials(subject, subjects_dir=subjects_dir)
    # ident: 1=LPA, 2=NASION, 3=RPA. r is in metres; convert to mm.
    fid_by_ident = {int(p["ident"]): np.asarray(p["r"], dtype=np.float64) * 1000.0
                    for p in fs_fids}

    src_pts = elc.positions_mm[:3].astype(np.float64)        # ELC LPA, Nz, RPA (mm)
    tgt_pts = np.array([fid_by_ident[1], fid_by_ident[2], fid_by_ident[3]],
                       dtype=np.float64)                     # fsaverage MRI mm

    trans = mne.coreg.fit_matched_points(
        src_pts=src_pts, tgt_pts=tgt_pts,
        rotate=True, translate=True, scale=False,
    )
    trans = np.asarray(trans, dtype=np.float64)
    if trans.shape != (4, 4):
        raise RuntimeError(
            f"fit_matched_points returned shape {trans.shape}, expected (4, 4)"
        )

    src_homo = np.c_[src_pts, np.ones(3)]
    src_aligned = (trans @ src_homo.T).T[:, :3]
    rmse = float(np.sqrt(np.mean(np.sum((src_aligned - tgt_pts) ** 2, axis=1))))

    return AtlasRegistration(
        trans_head_to_mri_mm=trans,
        fiducial_residual_rmse_mm=rmse,
        fsaverage_dir=fs_subject_dir,
        elc_sha256=elc.sha256,
        sigma_mm=sigma_mm,
        radius_mm=radius_mm,
    )


def apply_registration(
    midpoints_head_mm: Dict[str, np.ndarray],
    reg: AtlasRegistration,
) -> Dict[str, np.ndarray]:
    """Apply ``reg.trans_head_to_mri_mm`` to every midpoint in head-CTF mm."""
    chs = list(midpoints_head_mm.keys())
    arr = np.stack([midpoints_head_mm[ch] for ch in chs])
    homo = np.c_[arr, np.ones(len(arr))]
    out = (reg.trans_head_to_mri_mm @ homo.T).T[:, :3]
    return {ch: out[i] for i, ch in enumerate(chs)}


# --------------------------------------------------------------------------- #
# §16.6 — cortical projection + Brodmann query (probabilistic)                #
# --------------------------------------------------------------------------- #

def _load_pial_surface(
    subjects_dir: Path, subject: str = "fsaverage",
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Both-hemisphere pial vertices in MRI mm.

    Returns ``(verts_all_mm[N,3], hemi_arr[N], n_lh, n_rh)``.
    Pial surface coords come from FreeSurfer in mm already.
    """
    import mne
    surf_dir = Path(subjects_dir) / subject / "surf"
    verts_lh, _ = mne.read_surface(str(surf_dir / "lh.pial"))
    verts_rh, _ = mne.read_surface(str(surf_dir / "rh.pial"))
    verts_all = np.concatenate([verts_lh, verts_rh], axis=0)
    hemi = np.concatenate(
        [np.full(len(verts_lh), "L"), np.full(len(verts_rh), "R")]
    )
    return verts_all, hemi, len(verts_lh), len(verts_rh)


def _filter_real_brodmann_labels(labels):
    """Keep only ``Brodmann.X-{lh,rh}`` labels; drop ``???``, ``Medial_wall``,
    placeholders, etc."""
    return [L for L in labels if L.name.startswith("Brodmann.")]


def project_to_brodmann(
    midpoints_mri_mm: Dict[str, np.ndarray],
    sd_distances_mm: Dict[str, float],
    subjects_dir: Path,
    subject: str = "fsaverage",
    sigma_mm: float = DEFAULT_SIGMA_MM,
    radius_mm: float = DEFAULT_RADIUS_MM,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute channel→BA probability table and per-channel projection diagnostics.

    Returns ``(channel_to_brodmann_long, channel_midpoints_mni)``.

    *channel_to_brodmann_long*: rows ``(channel, ba_label, hemi, probability)``;
    sums to 1 per channel (any unlabelled mass goes to ``ba_label='Unlabelled'``,
    ``hemi='U'``).

    *channel_midpoints_mni*: rows ``(channel, x_mni_mm, y_mni_mm, z_mni_mm,
    projected_vertex_id, hemi, projection_distance_mm, sd_distance_mm)``.
    """
    import mne
    from scipy.spatial import cKDTree

    subjects_dir = Path(subjects_dir)
    verts_all_mm, hemi_arr, n_lh, _ = _load_pial_surface(subjects_dir, subject)
    tree = cKDTree(verts_all_mm)

    raw_labels = mne.read_labels_from_annot(
        subject, parc=DEFAULT_ATLAS_PARC, hemi="both",
        subjects_dir=subjects_dir, verbose="ERROR",
    )
    labels = _filter_real_brodmann_labels(raw_labels)

    # Build (combined-vertex-index → list[(ba_clean, hemi)]) lookup.
    # Label.vertices is hemi-local; offset rh vertices by n_lh.
    vertex_to_ba: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    for L in labels:
        ba_clean = L.name.rsplit("-", 1)[0]                 # 'Brodmann.10-lh' → 'Brodmann.10'
        h = "L" if L.hemi == "lh" else "R"
        offset = 0 if L.hemi == "lh" else n_lh
        for v in L.vertices:
            vertex_to_ba[int(v) + offset].append((ba_clean, h))

    rows: List[Dict] = []
    diag_rows: List[Dict] = []
    for ch in CHANNEL_NAMES:
        mid = midpoints_mri_mm[ch]
        nearby = tree.query_ball_point(mid, r=radius_mm)
        if not nearby:
            _, i0 = tree.query(mid, k=1)
            nearby = [int(i0)]
        nearby = list(nearby)
        d = np.linalg.norm(verts_all_mm[nearby] - mid, axis=1)
        w = np.exp(-(d / sigma_mm) ** 2 / 2.0)
        w_sum = float(w.sum())
        if w_sum <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w_sum

        mass: Dict[Tuple[str, str], float] = defaultdict(float)
        for v_idx, weight in zip(nearby, w):
            for ba, h in vertex_to_ba.get(int(v_idx), []):
                mass[(ba, h)] += float(weight)

        labelled_mass = sum(mass.values())
        if labelled_mass < 1.0 - 1e-9:
            mass[("Unlabelled", "U")] = mass.get(("Unlabelled", "U"), 0.0) + (
                1.0 - labelled_mass
            )

        for (ba, h), p in sorted(mass.items(), key=lambda kv: -kv[1]):
            rows.append(
                {"channel": ch, "ba_label": ba, "hemi": h, "probability": float(p)}
            )

        d_nearest, i_nearest = tree.query(mid, k=1)
        diag_rows.append(
            {
                "channel": ch,
                "x_mni_mm": float(mid[0]),
                "y_mni_mm": float(mid[1]),
                "z_mni_mm": float(mid[2]),
                "projected_vertex_id": int(i_nearest),
                "hemi": str(hemi_arr[i_nearest]),
                "projection_distance_mm": float(d_nearest),
                "sd_distance_mm": float(sd_distances_mm[ch]),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


# --------------------------------------------------------------------------- #
# §16 — top-level orchestrator                                                #
# --------------------------------------------------------------------------- #

@dataclass
class BrodmannMapping:
    channel_to_ba_long: pd.DataFrame    # channel, ba_label, hemi, probability
    midpoints_mri_mm: pd.DataFrame      # channel + diagnostics (see project_to_brodmann)
    registration: AtlasRegistration


def build_channel_to_brodmann(
    elc_path: Path,
    output_dir: Optional[Path] = None,
    subjects_dir: Optional[Path] = None,
    subject: str = "fsaverage",
    sigma_mm: float = DEFAULT_SIGMA_MM,
    radius_mm: float = DEFAULT_RADIUS_MM,
) -> BrodmannMapping:
    """End-to-end §16.3 → §16.6. Persists CSVs + ``registration_run.json`` if
    ``output_dir`` is given. Returns the in-memory mapping either way."""
    import mne
    elc = parse_elc(Path(elc_path))
    midpoints_head = compute_channel_midpoints(elc)
    sd_distances = compute_sd_distances(elc)

    reg = register_to_fsaverage(
        elc, subjects_dir=subjects_dir, subject=subject,
        sigma_mm=sigma_mm, radius_mm=radius_mm,
    )
    midpoints_mri = apply_registration(midpoints_head, reg)

    ch2ba, diag = project_to_brodmann(
        midpoints_mri, sd_distances,
        subjects_dir=Path(reg.fsaverage_dir).parent,
        subject=subject,
        sigma_mm=sigma_mm, radius_mm=radius_mm,
    )

    mapping = BrodmannMapping(
        channel_to_ba_long=ch2ba, midpoints_mri_mm=diag, registration=reg,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # %.8f matches aggregate.py's CSV precision (commit d2e69e4).
        ch2ba.to_csv(out / "channel_to_brodmann.csv", index=False, float_format="%.8f")
        diag.to_csv(out / "channel_midpoints_mni.csv", index=False, float_format="%.8f")
        run_meta = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "elc_path": str(Path(elc_path).resolve()),
            "elc_sha256": elc.sha256,
            "fsaverage_dir": reg.fsaverage_dir,
            "mne_version": mne.__version__,
            "atlas_parc": DEFAULT_ATLAS_PARC,
            "sigma_mm": sigma_mm,
            "radius_mm": radius_mm,
            "fiducial_residual_rmse_mm": reg.fiducial_residual_rmse_mm,
            "trans_head_to_mri_mm": reg.trans_head_to_mri_mm.tolist(),
            "n_optodes": elc.n_optodes,
            "n_channels": N_CH,
        }
        with open(out / "registration_run.json", "w") as f:
            json.dump(run_meta, f, indent=2)

    return mapping


# --------------------------------------------------------------------------- #
# §16.7 — region-level re-aggregation                                         #
# --------------------------------------------------------------------------- #

def aggregate_to_regions(
    node_csv: Path,
    pair_npy: Path,
    channel_to_brodmann_csv: Path,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[Tuple[str, str]]]:
    """Re-aggregate channel-level XAI to BA region level.

    Returns ``(region_importance_df, region_pair_matrix, region_keys)``.
    ``region_keys`` is the row/col ordering of the pair matrix:
    ``list[(ba_label, hemi)]`` sorted alphabetically.

    Conservation invariant (asserted in tests):
        ``region_importance_df['mean'].sum() ≈ node_importance.csv['mean'].sum()``
    (mass is preserved by the probability split).
    """
    ch2ba_long = pd.read_csv(channel_to_brodmann_csv)
    ch_imp = pd.read_csv(node_csv).set_index("channel")["mean"]
    pair_M = np.load(pair_npy)
    if pair_M.shape != (N_CH, N_CH):
        raise ValueError(f"pair_matrix shape {pair_M.shape} != ({N_CH}, {N_CH})")

    # Channel-level → region-level
    merged = ch2ba_long.merge(ch_imp.reset_index(), on="channel")
    merged["weighted"] = merged["mean"] * merged["probability"]
    region_imp = (
        merged.groupby(["ba_label", "hemi"], as_index=False)
        .agg(
            mean=("weighted", "sum"),
            n_channels_contrib=("channel", "nunique"),
            p_mass_total=("probability", "sum"),
        )
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )

    # Stable region ordering for the matrix
    region_keys: List[Tuple[str, str]] = sorted(
        {(r["ba_label"], r["hemi"]) for _, r in ch2ba_long.iterrows()}
    )
    key_idx = {k: i for i, k in enumerate(region_keys)}
    n_ba = len(region_keys)

    # Per-channel distribution dict
    ch_dist: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
    for _, r in ch2ba_long.iterrows():
        ch_dist[r["channel"]][(r["ba_label"], r["hemi"])] = float(r["probability"])

    region_pair = np.zeros((n_ba, n_ba), dtype=np.float64)
    for i, ch_i in enumerate(CHANNEL_NAMES):
        d_i = ch_dist.get(ch_i, {})
        if not d_i:
            continue
        for j, ch_j in enumerate(CHANNEL_NAMES):
            if i == j:
                continue
            v = float(pair_M[i, j])
            if v == 0.0:
                continue
            d_j = ch_dist.get(ch_j, {})
            if not d_j:
                continue
            for k_i, p_i in d_i.items():
                for k_j, p_j in d_j.items():
                    region_pair[key_idx[k_i], key_idx[k_j]] += v * p_i * p_j
    region_pair = (region_pair + region_pair.T) / 2.0

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        region_imp.to_csv(out / "region_importance.csv", index=False, float_format="%.8f")
        np.save(out / "region_pair_matrix.npy", region_pair.astype(np.float32))
        pd.DataFrame(region_keys, columns=["ba_label", "hemi"]).to_csv(
            out / "region_keys.csv", index=False
        )

    return region_imp, region_pair, region_keys
