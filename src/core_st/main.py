import os
from argparse import ArgumentParser
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import torch
import yaml
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .config import ExperimentConfig, SystemConfig, load_config, save_config, setup_system
from .dataset import fNIRSGraphDataset
from .models import WindowedSpatioTemporalGATNet
from .training import (
    CosineWarmupScheduler,
    perform_holdout_training,
    perform_kfold_training,
    perform_loso_training,
)
from .utils import get_experiment_dir

TASK_CHOICES = ["GNG", "VF", "SS", "1backWM"]


def _resolve_channel_subset(value: Any) -> Optional[List[int]]:
    """Normalise a CLI/YAML channel-subset value to a list of canonical indices.

    Accepted inputs (all map to the 23-channel order in `src/xai/channels.py:CHANNEL_NAMES`):
      - None / "" / "all" / "ALL"           → None (use all 23 channels, baseline behaviour)
      - "0,3,5,7"                            → [0, 3, 5, 7]
      - "S1_D1,S2_D1,S3_D1"                  → [0, 3, 5]
      - [0, 3, 5, 7]                         → [0, 3, 5, 7]   (YAML list of ints)
      - ["S1_D1", "S2_D1"]                   → [0, 3]         (YAML list of names)
      - mixed list e.g. [0, "S2_D1"]         → [0, 3]
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "all":
            return None
        tokens: Sequence[Any] = [tok.strip() for tok in stripped.split(",") if tok.strip()]
    elif isinstance(value, (list, tuple)):
        tokens = value
    else:
        raise ValueError(f"Unsupported channel_subset type: {type(value).__name__}")

    # Import locally to avoid pulling `src.xai` at module load when subset is unused.
    from src.xai.channels import CH_TO_IDX, N_CH

    resolved: List[int] = []
    for tok in tokens:
        if isinstance(tok, bool):  # bool is a subclass of int — reject explicitly
            raise ValueError(f"channel_subset token {tok!r} is a bool, not an index")
        if isinstance(tok, int):
            idx = tok
        else:
            tok_str = str(tok).strip()
            try:
                idx = int(tok_str)
            except ValueError:
                if tok_str not in CH_TO_IDX:
                    raise ValueError(
                        f"channel_subset token {tok_str!r} is not an int and not in CHANNEL_NAMES"
                    )
                idx = CH_TO_IDX[tok_str]
        if not (0 <= idx < N_CH):
            raise ValueError(f"channel_subset index {idx} out of range [0, {N_CH - 1}]")
        if idx in resolved:
            raise ValueError(f"channel_subset has duplicate index {idx}")
        resolved.append(idx)
    if not resolved:
        return None
    return resolved


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="fNIRS ST Pipeline — windowed spatial-temporal GNN on fNIRS data")
    p.add_argument("--config", type=str, default=None,
                   help="Path to experiment_config.yaml; values override code defaults, CLI overrides YAML")
    # Runtime / general
    p.add_argument("--data_dir", type=str, required=True, help="Root data directory (e.g. data/processed)")
    p.add_argument("--save_dir", type=str, default="experiments")
    p.add_argument("--task", type=str, default="GNG", choices=TASK_CHOICES,
                   help="Cognitive task subdirectory under data_dir")
    p.add_argument("--data_type", type=str, default="hbo", choices=["hbo", "hbr", "hbt"])
    p.add_argument("--validation", type=str, default="holdout", choices=["holdout", "kfold", "loso"])
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--checkpoint_metric", type=str, default="f1", choices=["f1", "loss"],
                   help="Metric used for checkpoint saving and early stopping: 'f1' (best val F1) or 'loss' (lowest val loss)")
    p.add_argument("--scheduler", type=str, default=None,
                   choices=["cosine_annealing", "cosine_warmup", "reduce_on_plateau"],
                   help="LR scheduler (default: cosine_annealing, CORAL-validated best)")
    p.add_argument("--eta_min", type=float, default=1e-5,
                   help="CosineAnnealingLR eta_min (default: 1e-5, Optuna lr_cosine best Trial #36)")
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--sqrt_class_weights", action="store_true")
    p.add_argument("--use_focal_loss", action="store_true")
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    # Graph construction
    p.add_argument("--max_trials", type=int, default=None)
    p.add_argument("--directed", action="store_true")
    p.add_argument("--corr_threshold", type=float, default=None)
    p.add_argument("--self_loops", action="store_true")
    p.add_argument("--channel_subset", type=str, default=None,
                   help="Comma-separated list of channel indices OR names from src/xai/channels.py "
                        "CHANNEL_NAMES (e.g. '0,3,5' or 'S1_D1,S2_D1,S3_D1'). Pass 'all' or omit "
                        "to use all 23 channels. CLI overrides any value in --config YAML.")
    # Spatial model (GATv2)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--n_filters", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--fc_size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--no_residual", action="store_true")
    p.add_argument("--use_norm", action="store_true")
    p.add_argument("--norm_type", type=str, default=None, choices=["batch", "layer"])
    # Temporal model (GRU + attention)
    p.add_argument("--window_size", type=int, default=None,
                   help="Timepoints per window (default 32 ≈ 3.2s at 10Hz)")
    p.add_argument("--window_stride", type=int, default=None,
                   help="Window step size; 50%% overlap = window_size // 2")
    p.add_argument("--temporal_hidden", type=int, default=None,
                   help="GRU hidden dimension")
    p.add_argument("--temporal_layers", type=int, default=None,
                   help="Number of GRU layers")
    # Augmentation
    p.add_argument("--augment", action="store_true")
    p.add_argument("--edge_dropout_p", type=float, default=None)
    p.add_argument("--feature_mask_p", type=float, default=None)
    p.add_argument("--feature_mask_mode", type=str, default=None, choices=["all", "col", "row"])
    # Splits
    p.add_argument("--splits_json", type=str, default=None,
                   help="Path to pre-defined k-fold splits JSON")
    return p


def _load_yaml_flat(path: str) -> Dict[str, Any]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    flat: Dict[str, Any] = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def build_experiment_name(
    args, max_trials: Optional[int] = None, channel_subset: Optional[List[int]] = None
) -> str:
    aug_str = "aug" if args.augment else "noaug"
    today = date.today().strftime("%Y%m%d")
    mt_str = f"_mt{max_trials}" if max_trials is not None else ""
    # Suffix the experiment name only when an actual subset is used (skip for full 23-channel
    # baseline so historical names are unchanged).
    k_str = f"_K{len(channel_subset):02d}" if channel_subset is not None else ""
    return f"ST_GATv2_{args.task}_{args.data_type}_{args.validation}{mt_str}_{aug_str}{k_str}_{today}"


def _args_to_config(args, yaml_cfg: Dict[str, Any]) -> ExperimentConfig:
    def pick(cli_val, yaml_key, default):
        if cli_val is not None:
            return cli_val
        return yaml_cfg.get(yaml_key, default)

    # CLI value (str) takes precedence over YAML value (str or list).
    raw_subset = args.channel_subset if args.channel_subset is not None else yaml_cfg.get("channel_subset")
    channel_subset = _resolve_channel_subset(raw_subset)

    return ExperimentConfig(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        task_type=args.task,
        data_type=args.data_type,
        max_trials=pick(args.max_trials, "max_trials", None),
        directed=args.directed or yaml_cfg.get("directed", False),
        corr_threshold=pick(args.corr_threshold, "corr_threshold", 0.1),
        self_loops=args.self_loops or yaml_cfg.get("self_loops", False),
        channel_subset=channel_subset,
        n_layers=pick(args.n_layers, "n_layers", 2),
        n_filters=pick(args.n_filters, "n_filters", 64),
        n_heads=pick(args.n_heads, "n_heads", 4),
        fc_size=pick(args.fc_size, "fc_size", 64),
        dropout=pick(args.dropout, "dropout", 0.5),
        use_residual=(not args.no_residual) and yaml_cfg.get("use_residual", True),
        use_norm=args.use_norm or yaml_cfg.get("use_norm", False),
        norm_type=pick(args.norm_type, "norm_type", "batch"),
        window_size=pick(args.window_size, "window_size", 32),
        window_stride=pick(args.window_stride, "window_stride", 16),
        temporal_hidden=pick(args.temporal_hidden, "temporal_hidden", 64),
        temporal_layers=pick(args.temporal_layers, "temporal_layers", 1),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        checkpoint_metric=args.checkpoint_metric,
        scheduler=pick(args.scheduler, "scheduler", "cosine_annealing"),
        use_class_weights=args.use_class_weights,
        sqrt_class_weights=args.sqrt_class_weights,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        augment=args.augment or yaml_cfg.get("augment", False),
        edge_dropout_p=pick(args.edge_dropout_p, "edge_dropout_p", 0.1),
        feature_mask_p=pick(args.feature_mask_p, "feature_mask_p", 0.1),
        feature_mask_mode=pick(args.feature_mask_mode, "feature_mask_mode", "all"),
        use_rwpe=False,
        rwpe_walk_length=4,
        validation=args.validation,
        k_folds=args.k_folds,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
        splits_json=args.splits_json,
        num_workers=args.num_workers,
        resume=args.resume,
    )


def main() -> None:
    parser = build_parser()

    pre_args, _ = parser.parse_known_args()

    saved_cfg_path = os.path.join(pre_args.save_dir, "config.yaml")
    if pre_args.resume and os.path.exists(saved_cfg_path):
        cfg = load_config(saved_cfg_path)
        args = pre_args
        yaml_cfg: Dict[str, Any] = {}
        print(f"Resuming from saved config: {saved_cfg_path}")
    else:
        args = parser.parse_args()
        yaml_cfg = _load_yaml_flat(args.config) if args.config else {}
        if yaml_cfg:
            print(f"Loaded experiment config: {args.config}")
        cfg = _args_to_config(args, yaml_cfg)

    setup_system(SystemConfig(seed=args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_name = build_experiment_name(args, max_trials=cfg.max_trials, channel_subset=cfg.channel_subset)
    exp_dir = get_experiment_dir(exp_name, cfg.save_dir)
    save_config(cfg, os.path.join(exp_dir, "config.yaml"))
    print(f"Task: {cfg.task_type} | Experiment dir: {exp_dir}")
    if cfg.channel_subset is not None:
        print(f"Channel subset active: K={len(cfg.channel_subset)} indices {cfg.channel_subset}")

    print("Loading dataset...")
    dataset = fNIRSGraphDataset(
        root=cfg.data_dir,
        task_type=cfg.task_type,
        data_type=cfg.data_type,
        max_trials=cfg.max_trials,
        directed=cfg.directed,
        corr_threshold=cfg.corr_threshold,
        self_loops=cfg.self_loops,
        channel_subset=cfg.channel_subset,
    )
    print(f"Dataset: {len(dataset)} graphs loaded")

    # Stats and transforms are built per-fold inside training.py to keep normalization
    # leak-free (see SG_vs_ST_validation_comparison.md §7).

    n_channels = len(cfg.channel_subset) if cfg.channel_subset is not None else 23
    model = WindowedSpatioTemporalGATNet(
        n_channels=n_channels,
        in_channels=6,
        edge_dim=2,
        window_size=cfg.window_size,
        window_stride=cfg.window_stride,
        n_layers=cfg.n_layers,
        n_filters=cfg.n_filters,
        heads=cfg.n_heads,
        temporal_hidden=cfg.temporal_hidden,
        temporal_layers=cfg.temporal_layers,
        fc_size=cfg.fc_size,
        dropout=cfg.dropout,
        n_classes=2,
        use_residual=cfg.use_residual,
        use_norm=cfg.use_norm,
        norm_type=cfg.norm_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: WindowedSpatioTemporalGATNet | Parameters: {n_params:,}")

    optimizer_class = optim.Adam
    optimizer_params = {"lr": cfg.lr}
    if cfg.scheduler == "cosine_annealing":
        scheduler_class = CosineAnnealingLR
        scheduler_params = {"T_max": cfg.epochs, "eta_min": args.eta_min}
    elif cfg.scheduler == "cosine_warmup":
        scheduler_class = CosineWarmupScheduler
        scheduler_params = {"warmup": 5, "max_iters": cfg.epochs}
    else:  # reduce_on_plateau
        scheduler_class = ReduceLROnPlateau
        scheduler_params = {"mode": "max", "factor": 0.5, "patience": 5}

    shared = dict(
        model=model,
        dataset=dataset,
        cfg=cfg,
        device=device,
        exp_dir=exp_dir,
        model_name=exp_name,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        scheduler_class=scheduler_class,
        scheduler_params=scheduler_params,
    )

    if cfg.validation == "loso":
        perform_loso_training(**shared, resume=cfg.resume)
    elif cfg.validation == "kfold":
        perform_kfold_training(**shared, resume=cfg.resume, splits_json=cfg.splits_json)
    else:
        perform_holdout_training(**shared)

    print(f"\nDone. Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
