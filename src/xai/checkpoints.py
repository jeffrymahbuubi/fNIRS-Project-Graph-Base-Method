"""Checkpoint discovery + leak-free reload for the XAI pipeline.

SPEC §5.2 / §6.1 / §10 (rev. 4). Two public entry points:

- `discover_checkpoints(experiment_root, *, arch, hb, regime, mt)`
    Walks the regime subdirectory, returns a list of `CheckpointInfo`
    (one per fold or per LOSO subject) that match the (hb, regime, mt) cell.

- `load_checkpoint(info, cfg)`
    Reconstructs the model from `config.yaml`, loads its state_dict, builds
    the dataset + leak-free val transform from the same config, and returns
    a `LoadedCheckpoint` bundle ready for the explainer.

Auto-rebase: cloud-trained checkpoints (e.g. HbO mt4 LOSO 20260506) record a
container path like `/root/remote-training-setup/data/processed-new-mc` in
their `data_dir`. SPEC §10.4 specifies how to recover the local path; the
rewrite is captured in `LoadedCheckpoint.path_rebases` so the run.json
audit trail records what happened.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch_geometric.transforms import Compose

from src.xai.config import XAIRunConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KNOWN_CLOUD_PREFIXES: Tuple[str, ...] = (
    "/root/remote-training-setup/",
)

# Map XAIRunConfig.regime → relative subdirectory under experiment_root.
# SG keeps an extra `kfold/` parent ("kfold/5-fold"), ST does not ("5-fold").
_REGIME_SUBDIR: Dict[str, Dict[str, str]] = {
    "sg": {"kfold-5": "kfold/5-fold", "kfold-10": "kfold/10-fold", "loso": "loso"},
    "st": {"kfold-5": "5-fold", "kfold-10": "10-fold", "loso": "loso"},
}

# Experiment-directory name pattern (the directory CONTAINING config.yaml + .pt).
# SG: GATv2_GNG_{hb}_{regime_token}_mt{N}_noaug_{YYYYMMDD}
# ST: ST_GATv2_GNG_{hb}_{regime_token}_mt{N}_noaug_{YYYYMMDD}
# regime_token is 'kfold' for both kfold-5 and kfold-10, or 'loso' for LOSO.
_EXP_DIR_RE: Dict[str, re.Pattern] = {
    "sg": re.compile(
        r"^GATv2_GNG_(?P<hb>hbo|hbr|hbt)_(?P<regime_token>kfold|loso)"
        r"_mt(?P<mt>\d+)_noaug_(?P<date>\d{8})$"
    ),
    "st": re.compile(
        r"^ST_GATv2_GNG_(?P<hb>hbo|hbr|hbt)_(?P<regime_token>kfold|loso)"
        r"_mt(?P<mt>\d+)_noaug_(?P<date>\d{8})$"
    ),
}

# Checkpoint filename patterns: {exp_name}_fold_{F}.pt or {exp_name}_subj_{SID}.pt
_CKPT_FOLD_RE = re.compile(r"_fold_(?P<fold>\d+)\.pt$")
_CKPT_SUBJ_RE = re.compile(r"_subj_(?P<subj>[A-Za-z0-9_]+)\.pt$")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CheckpointInfo:
    """One checkpoint = one (fold or subject) of one experiment.

    `fold` is set for kfold checkpoints; `subject` is set for LOSO checkpoints.
    Exactly one of them is non-None.
    """

    arch: str                       # 'sg' | 'st'
    hb: str                         # 'hbo' | 'hbr' | 'hbt'
    regime: str                     # 'kfold-5' | 'kfold-10' | 'loso'
    mt: int                         # 2 | 4
    fold: Optional[int]             # kfold only
    subject: Optional[str]          # LOSO only
    exp_dir: Path                   # the directory containing config.yaml + .pt + .pkl
    exp_name: str                   # the experiment-directory base name
    ckpt_path: Path                 # the .pt file
    config_path: Path               # the config.yaml file
    pkl_path: Path                  # the .pkl results file (used for C1 acceptance)

    def __post_init__(self) -> None:
        if (self.fold is None) == (self.subject is None):
            raise ValueError(
                f"Exactly one of fold/subject must be set, got fold={self.fold}, subject={self.subject!r}"
            )

    @property
    def label(self) -> str:
        """Short human-readable label: 'fold-1' or 'subj-AA011'."""
        return f"fold-{self.fold}" if self.fold is not None else f"subj-{self.subject}"


@dataclass
class LoadedCheckpoint:
    """Fully-loaded XAI-ready checkpoint.

    The model is in eval mode and on `device`.  The dataset is the full,
    un-transformed dataset (transforms are applied per-trial inside the
    explainer via `val_transform`).  `train_indices` is what `compute_stats`
    was called with, so explainer code that re-derives stats can verify
    leak-freeness.
    """

    info: CheckpointInfo
    model: nn.Module
    dataset: Any                     # fNIRSGraphDataset (SG or ST flavour)
    train_indices: List[int]
    val_indices: List[int]
    val_subject: Optional[str]       # LOSO only
    val_transform: Compose
    config: Dict[str, Any]
    stored_metrics: Dict[str, Any]
    path_rebases: List[Dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Path resolution + audit
# ---------------------------------------------------------------------------


def _detect_project_root() -> Path:
    """Default project_root = src/xai/checkpoints.py → up 3 levels."""
    return Path(__file__).resolve().parents[2]


def _safe_exists(p: Path) -> bool:
    """Path.exists() can raise PermissionError on cloud-trained-only paths
    like /root/remote-training-setup/... when running as a non-root user.
    Treat any OS-level failure as 'does not exist' for resolution purposes.
    """
    try:
        return p.exists()
    except (PermissionError, OSError):
        return False


def _resolve_data_dir(
    raw: str, project_root: Path
) -> Tuple[Path, Optional[Dict[str, str]]]:
    """SPEC §10.4 rebase rule.

    Returns (resolved_path, rebase_log_entry_or_None). Raises FileNotFoundError
    if no candidate exists.
    """
    p = Path(raw)
    if _safe_exists(p):
        return p, None

    for prefix in _KNOWN_CLOUD_PREFIXES:
        if str(p).startswith(prefix):
            tail = str(p)[len(prefix):]
            candidate = project_root / tail
            if _safe_exists(candidate):
                return candidate, {
                    "raw": str(p),
                    "resolved": str(candidate),
                    "reason": f"cloud_prefix:{prefix}",
                }

    candidate = project_root / "data" / p.name
    if _safe_exists(candidate):
        return candidate, {
            "raw": str(p),
            "resolved": str(candidate),
            "reason": "basename_fallback",
        }

    raise FileNotFoundError(
        f"data_dir from config.yaml not resolvable: {raw!r} "
        f"(tried as-is, known cloud prefixes {_KNOWN_CLOUD_PREFIXES}, "
        f"and {project_root / 'data' / p.name})"
    )


def _resolve_splits_json(
    raw: Optional[str], project_root: Path
) -> Tuple[Optional[Path], Optional[Dict[str, str]]]:
    """Same rebase logic as data_dir, but None / 'null' is preserved (LOSO case)."""
    if raw is None:
        return None, None
    p = Path(raw)
    if _safe_exists(p):
        return p, None

    for prefix in _KNOWN_CLOUD_PREFIXES:
        if str(p).startswith(prefix):
            candidate = project_root / str(p)[len(prefix):]
            if _safe_exists(candidate):
                return candidate, {
                    "raw": str(p),
                    "resolved": str(candidate),
                    "reason": f"cloud_prefix:{prefix}",
                }

    candidate = project_root / "data" / "splits" / p.name
    if _safe_exists(candidate):
        return candidate, {
            "raw": str(p),
            "resolved": str(candidate),
            "reason": "basename_fallback",
        }

    raise FileNotFoundError(
        f"splits_json from config.yaml not resolvable: {raw!r} "
        f"(tried as-is, known cloud prefixes, and {project_root / 'data' / 'splits' / p.name})"
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _regime_subdir(arch: str, regime: str) -> Path:
    return Path(_REGIME_SUBDIR[arch][regime])


def discover_checkpoints(
    experiment_root: str | Path,
    *,
    arch: str,
    hb: str,
    regime: str,
    mt: int,
    subdir_override: Optional[str] = None,
) -> List[CheckpointInfo]:
    """Find every fold/subject checkpoint matching this (arch, hb, regime, mt) cell.

    `subdir_override` replaces the default `_REGIME_SUBDIR[arch][regime]`
    when provided. Use it to point at non-canonical layouts such as the
    20260509 ST kfold dirs (`st-kfold/{5,10}-fold/20260509/`). LOSO under
    20260509 still matches the default ('loso'), so leave override=None
    for LOSO cells.

    Sorts kfold checkpoints by fold number ascending; sorts LOSO checkpoints
    by subject ID lexicographically (matches the order the SG/ST datasets
    iterate subjects).
    """
    if arch not in _EXP_DIR_RE:
        raise ValueError(f"unknown arch {arch!r}")
    regime_token = "kfold" if regime.startswith("kfold-") else "loso"
    subdir = Path(subdir_override) if subdir_override else _regime_subdir(arch, regime)
    regime_dir = Path(experiment_root) / subdir
    if not regime_dir.is_dir():
        raise FileNotFoundError(f"regime dir not found: {regime_dir}")

    matches: List[CheckpointInfo] = []
    pattern = _EXP_DIR_RE[arch]
    for exp_dir in sorted(regime_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        m = pattern.match(exp_dir.name)
        if m is None:
            continue
        if m.group("hb") != hb or m.group("regime_token") != regime_token or int(m.group("mt")) != mt:
            continue

        config_path = exp_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"config.yaml missing in {exp_dir}")

        for ckpt_path in sorted(exp_dir.glob("*.pt")):
            stem = ckpt_path.name
            fold_match = _CKPT_FOLD_RE.search(stem)
            subj_match = _CKPT_SUBJ_RE.search(stem)
            if fold_match is not None and regime_token == "kfold":
                fold = int(fold_match.group("fold"))
                pkl_path = ckpt_path.with_suffix(".pkl")
                matches.append(CheckpointInfo(
                    arch=arch, hb=hb, regime=regime, mt=mt,
                    fold=fold, subject=None,
                    exp_dir=exp_dir, exp_name=exp_dir.name,
                    ckpt_path=ckpt_path, config_path=config_path,
                    pkl_path=pkl_path,
                ))
            elif subj_match is not None and regime_token == "loso":
                subj = subj_match.group("subj")
                pkl_path = ckpt_path.with_suffix(".pkl")
                matches.append(CheckpointInfo(
                    arch=arch, hb=hb, regime=regime, mt=mt,
                    fold=None, subject=subj,
                    exp_dir=exp_dir, exp_name=exp_dir.name,
                    ckpt_path=ckpt_path, config_path=config_path,
                    pkl_path=pkl_path,
                ))

    if regime_token == "kfold":
        matches.sort(key=lambda i: (i.exp_name, i.fold or -1))
    else:
        matches.sort(key=lambda i: (i.exp_name, i.subject or ""))
    return matches


# ---------------------------------------------------------------------------
# Model construction (per-arch dispatch)
# ---------------------------------------------------------------------------


def _build_sg_model(config: Dict[str, Any]) -> nn.Module:
    from src.core.models import FlexibleGATNet

    rwpe_walk_length = int(config.get("rwpe_walk_length", 0))
    in_channels = 6 + (rwpe_walk_length if config.get("use_rwpe", False) else 0)
    return FlexibleGATNet(
        in_channels=in_channels,
        edge_dim=2,
        n_layers=int(config["n_layers"]),
        n_filters=config["n_filters"],
        heads=config["n_heads"],
        fc_size=int(config["fc_size"]),
        dropout=float(config["dropout"]),
        n_classes=2,
        use_residual=bool(config["use_residual"]),
        use_norm=bool(config["use_norm"]),
        norm_type=str(config.get("norm_type", "batch")),
        use_gine_first_layer=bool(config.get("use_gine_first_layer", False)),
    )


def _build_st_model(config: Dict[str, Any]) -> nn.Module:
    from src.core_st.models import WindowedSpatioTemporalGATNet

    return WindowedSpatioTemporalGATNet(
        in_channels=6,
        edge_dim=2,
        window_size=int(config["window_size"]),
        window_stride=int(config["window_stride"]),
        n_layers=int(config["n_layers"]),
        n_filters=int(config["n_filters"]),
        heads=int(config["n_heads"]),
        temporal_hidden=int(config["temporal_hidden"]),
        temporal_layers=int(config["temporal_layers"]),
        fc_size=int(config["fc_size"]),
        dropout=float(config["dropout"]),
        n_classes=2,
        use_residual=bool(config["use_residual"]),
        use_norm=bool(config["use_norm"]),
        norm_type=str(config.get("norm_type", "batch")),
    )


# ---------------------------------------------------------------------------
# Dataset + split helpers
# ---------------------------------------------------------------------------


def _import_dataset_module(arch: str):
    if arch == "sg":
        from src.core import dataset as ds_mod  # type: ignore
        return ds_mod
    if arch == "st":
        from src.core_st import dataset as ds_mod  # type: ignore
        return ds_mod
    raise ValueError(f"unknown arch {arch!r}")


def _build_dataset(arch: str, data_dir: Path, config: Dict[str, Any]):
    ds_mod = _import_dataset_module(arch)
    return ds_mod.fNIRSGraphDataset(
        root=str(data_dir),
        task_type=str(config.get("task_type", "GNG")),
        data_type=str(config["data_type"]),
        max_trials=int(config["max_trials"]) if config.get("max_trials") is not None else None,
        directed=bool(config.get("directed", True)),
        corr_threshold=float(config.get("corr_threshold", 0.1)),
        self_loops=bool(config.get("self_loops", True)),
    )


def _resolve_split_indices(
    arch: str,
    info: CheckpointInfo,
    dataset,
    splits_json: Optional[Path],
    config: Dict[str, Any],
) -> Tuple[List[int], List[int], Optional[str]]:
    """Resolve (train_indices, val_indices, val_subject) for this checkpoint."""
    ds_mod = _import_dataset_module(arch)

    if info.regime == "loso":
        # LOSO splits are computed in-code; splits_json is null in config.
        if info.subject is None:
            raise ValueError("LOSO checkpoint missing subject id")
        for train_idx, val_idx, val_subj in ds_mod.get_loso_splits(dataset, verbose=False):
            if val_subj == info.subject:
                return list(train_idx), list(val_idx), val_subj
        raise ValueError(
            f"LOSO subject {info.subject!r} not found in dataset (dataset has "
            f"{len(set(str(dataset[i].subject_id) for i in range(len(dataset))))} subjects)"
        )

    # kfold path
    if splits_json is None:
        raise ValueError(
            f"{info.label}: kfold checkpoint requires a splits_json, but config.yaml had splits_json=null"
        )
    if info.fold is None:
        raise ValueError("kfold checkpoint missing fold number")
    n_splits = 5 if info.regime == "kfold-5" else 10
    folds = ds_mod.get_kfold_splits_from_json(
        dataset, str(splits_json), n_splits=n_splits, verbose=False
    )
    # get_kfold_splits_from_json returns an ordered list aligned with the JSON's
    # 'fold' values (1..n_splits). Use info.fold (1-indexed) directly.
    if not (1 <= info.fold <= len(folds)):
        raise ValueError(f"fold {info.fold} out of range [1, {len(folds)}]")
    train_idx, val_idx = folds[info.fold - 1]
    return list(train_idx), list(val_idx), None


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


def _load_pkl(pkl_path: Path) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_val_transform(arch: str, stats: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Compose:
    if arch == "sg":
        from src.core.transforms import get_transforms
    else:
        from src.core_st.transforms import get_transforms
    use_rwpe = bool(config.get("use_rwpe", False)) if arch == "sg" else False
    rwpe_walk_length = int(config.get("rwpe_walk_length", 4))
    return get_transforms(
        stats=stats,
        augment=False,
        edge_dropout_p=0.0,
        feature_mask_p=0.0,
        feature_mask_mode=str(config.get("feature_mask_mode", "all")),
        use_rwpe=use_rwpe,
        rwpe_walk_length=rwpe_walk_length,
    )


def load_checkpoint(info: CheckpointInfo, cfg: XAIRunConfig) -> LoadedCheckpoint:
    """Reconstruct a model + leak-free val transform + dataset for one checkpoint.

    Per SPEC §5.2 / §6.1 / §10. The returned bundle is what the explainer
    iterates over: the model (eval mode, on cfg.device), the dataset, and the
    val_indices that scope the explanation graphs.
    """
    project_root = Path(cfg.project_root) if cfg.project_root else _detect_project_root()
    config = _load_yaml(info.config_path)

    # --- coherence checks against XAIRunConfig ---------------------------
    if str(config.get("data_type")) != cfg.hb:
        raise ValueError(
            f"{info.label}: config.data_type={config.get('data_type')!r} but cfg.hb={cfg.hb!r}"
        )
    if int(config.get("max_trials", -1)) != cfg.mt:
        raise ValueError(
            f"{info.label}: config.max_trials={config.get('max_trials')} but cfg.mt={cfg.mt}"
        )

    # --- resolve data paths (rebase) -------------------------------------
    path_rebases: List[Dict[str, str]] = []
    raw_data_dir = cfg.data_dir_override or str(config["data_dir"])
    data_dir, data_rebase = _resolve_data_dir(raw_data_dir, project_root)
    if data_rebase is not None:
        data_rebase["field"] = "data_dir"
        data_rebase["checkpoint"] = info.label
        path_rebases.append(data_rebase)

    raw_splits = cfg.splits_json_override if cfg.splits_json_override is not None else config.get("splits_json")
    splits_json, splits_rebase = _resolve_splits_json(raw_splits, project_root)
    if splits_rebase is not None:
        splits_rebase["field"] = "splits_json"
        splits_rebase["checkpoint"] = info.label
        path_rebases.append(splits_rebase)

    # --- build dataset + splits + leak-free stats ------------------------
    dataset = _build_dataset(cfg.arch, data_dir, config)
    train_indices, val_indices, val_subject = _resolve_split_indices(
        cfg.arch, info, dataset, splits_json, config
    )
    stats = dataset.compute_stats(train_indices)
    val_transform = _build_val_transform(cfg.arch, stats, config)

    # --- model ------------------------------------------------------------
    if cfg.arch == "sg":
        model = _build_sg_model(config)
    else:
        model = _build_st_model(config)

    state_dict = torch.load(info.ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(cfg.device).eval()

    # --- stored metrics (for C1) -----------------------------------------
    stored_metrics = _load_pkl(info.pkl_path)

    return LoadedCheckpoint(
        info=info,
        model=model,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        val_subject=val_subject,
        val_transform=val_transform,
        config=config,
        stored_metrics=stored_metrics,
        path_rebases=path_rebases,
    )
