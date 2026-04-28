"""
fNIRS GATv2 optimization seed.

FIXED (do NOT change):
  data_type="hbo", max_trials=2, task_type="GNG"
  validation="holdout", val_ratio=0.2, random_state=42
  directed=True, self_loops=True

OPTIMIZE — roughly ordered by expected impact:
  1. Augmentation: augment, edge_dropout_p, feature_mask_p, feature_mask_mode, use_rwpe
  2. Loss: use_focal_loss, focal_alpha, focal_gamma, use_class_weights, sqrt_class_weights
  3. Training: lr, epochs, patience, batch_size
  4. Architecture: n_layers, n_filters, n_heads, fc_size, dropout, use_norm, norm_type
  5. core/models.py — add new layer types or pooling strategies if needed

BASELINE: Holdout F1 = 0.8065
"""
import os
import sys
import tempfile

import torch
from torch import optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import ExperimentConfig, SystemConfig, setup_system
from core.dataset import fNIRSGraphDataset
from core.models import FlexibleGATNet
from core.training import CosineWarmupScheduler, perform_holdout_training
from core.transforms import get_transforms


def run(data_dir: str) -> float:
    """Train the model and return the best validation F1 score."""
    setup_system(SystemConfig(seed=42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ExperimentConfig(
        data_dir=data_dir,
        save_dir=tempfile.mkdtemp(prefix="coral_fnirs_"),

        # --- FIXED ---
        task_type="GNG",
        data_type="hbo",
        max_trials=2,
        directed=True,
        corr_threshold=0.1,
        self_loops=True,
        validation="holdout",
        val_ratio=0.2,
        random_state=42,
        num_workers=0,
        pin_memory=False,

        # --- ARCHITECTURE (tune me) ---
        n_layers=2,
        n_filters=[112, 32],
        n_heads=[6, 4],
        fc_size=96,
        dropout=0.4,
        use_residual=True,
        use_norm=True,
        norm_type="batch",
        use_gine_first_layer=True,

        # --- TRAINING (tune me) ---
        epochs=100,
        batch_size=8,
        lr=1e-3,
        patience=9999,

        # --- LOSS (tune me) ---
        use_class_weights=False,
        sqrt_class_weights=False,
        use_focal_loss=False,
        focal_alpha=0.25,
        focal_gamma=2.0,

        # --- AUGMENTATION (tune me) ---
        augment=False,
        edge_dropout_p=0.1,
        feature_mask_p=0.1,
        feature_mask_mode="all",
        use_rwpe=False,
        rwpe_walk_length=4,
    )

    dataset = fNIRSGraphDataset(
        root=cfg.data_dir,
        task_type=cfg.task_type,
        data_type=cfg.data_type,
        max_trials=cfg.max_trials,
        directed=cfg.directed,
        corr_threshold=cfg.corr_threshold,
        self_loops=cfg.self_loops,
    )

    stats = dataset.compute_stats()
    train_transform = get_transforms(
        stats,
        augment=cfg.augment,
        edge_dropout_p=cfg.edge_dropout_p,
        feature_mask_p=cfg.feature_mask_p,
        feature_mask_mode=cfg.feature_mask_mode,
        use_rwpe=cfg.use_rwpe,
        rwpe_walk_length=cfg.rwpe_walk_length,
    )
    val_transform = get_transforms(
        stats,
        augment=False,
        use_rwpe=cfg.use_rwpe,
        rwpe_walk_length=cfg.rwpe_walk_length,
    )

    in_channels = 6 + (cfg.rwpe_walk_length if cfg.use_rwpe else 0)
    model = FlexibleGATNet(
        in_channels=in_channels,
        edge_dim=2,
        n_layers=cfg.n_layers,
        n_filters=cfg.n_filters,
        heads=cfg.n_heads,
        fc_size=cfg.fc_size,
        dropout=cfg.dropout,
        n_classes=2,
        use_residual=cfg.use_residual,
        use_norm=cfg.use_norm,
        norm_type=cfg.norm_type,
        use_gine_first_layer=cfg.use_gine_first_layer,
    ).to(device)

    results = perform_holdout_training(
        model=model,
        dataset=dataset,
        cfg=cfg,
        device=device,
        exp_dir=cfg.save_dir,
        model_name="coral_fnirs",
        train_transform=train_transform,
        val_transform=val_transform,
        optimizer_class=optim.Adam,
        optimizer_params={"lr": cfg.lr},
        scheduler_class=CosineWarmupScheduler,
        scheduler_params={"warmup": 5, "max_iters": cfg.epochs},
    )

    return float(results["f1_score"])
