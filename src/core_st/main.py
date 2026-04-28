import os
from argparse import ArgumentParser
from datetime import date
from typing import Any, Dict, Optional

import torch
import yaml
from torch import optim

from .config import ExperimentConfig, SystemConfig, load_config, save_config, setup_system
from .dataset import fNIRSGraphDataset
from .models import WindowedSpatioTemporalGATNet
from .training import (
    CosineWarmupScheduler,
    perform_holdout_training,
    perform_kfold_training,
    perform_loso_training,
)
from .transforms import get_transforms
from .utils import get_experiment_dir

TASK_CHOICES = ["GNG", "VF", "SS", "1backWM"]


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


def build_experiment_name(args, max_trials: Optional[int] = None) -> str:
    aug_str = "aug" if args.augment else "noaug"
    today = date.today().strftime("%Y%m%d")
    mt_str = f"_mt{max_trials}" if max_trials is not None else ""
    return f"ST_GATv2_{args.task}_{args.data_type}_{args.validation}{mt_str}_{aug_str}_{today}"


def _args_to_config(args, yaml_cfg: Dict[str, Any]) -> ExperimentConfig:
    def pick(cli_val, yaml_key, default):
        if cli_val is not None:
            return cli_val
        return yaml_cfg.get(yaml_key, default)

    return ExperimentConfig(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        task_type=args.task,
        data_type=args.data_type,
        max_trials=pick(args.max_trials, "max_trials", None),
        directed=args.directed or yaml_cfg.get("directed", False),
        corr_threshold=pick(args.corr_threshold, "corr_threshold", 0.1),
        self_loops=args.self_loops or yaml_cfg.get("self_loops", False),
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

    exp_name = build_experiment_name(args, max_trials=cfg.max_trials)
    exp_dir = get_experiment_dir(exp_name, cfg.save_dir)
    save_config(cfg, os.path.join(exp_dir, "config.yaml"))
    print(f"Task: {cfg.task_type} | Experiment dir: {exp_dir}")

    print("Loading dataset...")
    dataset = fNIRSGraphDataset(
        root=cfg.data_dir,
        task_type=cfg.task_type,
        data_type=cfg.data_type,
        max_trials=cfg.max_trials,
        directed=cfg.directed,
        corr_threshold=cfg.corr_threshold,
        self_loops=cfg.self_loops,
    )
    print(f"Dataset: {len(dataset)} graphs loaded")

    stats = dataset.compute_stats()
    train_transform = get_transforms(
        stats,
        augment=cfg.augment,
        edge_dropout_p=cfg.edge_dropout_p,
        feature_mask_p=cfg.feature_mask_p,
        feature_mask_mode=cfg.feature_mask_mode,
    )
    val_transform = get_transforms(stats, augment=False)

    model = WindowedSpatioTemporalGATNet(
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
    scheduler_class = CosineWarmupScheduler
    scheduler_params = {"warmup": 5, "max_iters": cfg.epochs}

    shared = dict(
        model=model,
        dataset=dataset,
        cfg=cfg,
        device=device,
        exp_dir=exp_dir,
        model_name=exp_name,
        train_transform=train_transform,
        val_transform=val_transform,
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
