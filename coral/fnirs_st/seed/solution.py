"""
fNIRS ST-GNN temporal module search — CORAL seed.

FIXED (do NOT change):
  Spatial encoder: n_layers=2, n_filters=80, heads=2
                   use_residual=False, use_norm=True, norm_type="batch"
  Windowing:       window_size=16, window_stride=8  (mt4)
  Dataset:         data_type="hbo", max_trials=4, task_type="GNG"
                   directed=True, self_loops=True, corr_threshold=0.1
  Training:        epochs=100, patience=20, batch_size=8, Adam + ReduceLROnPlateau

OPTIMIZE — explore temporal module type and its hyperparameters:
  temporal_type:    "gru" | "lstm" | "bigru" | "transformer" | "tcn"
  temporal_hidden:  [32..256, step=32]
  temporal_layers:  [1..3]
  Transformer:      transformer_heads {2,4,8}, ffn_ratio {2,4}
  TCN:              tcn_kernel_size {3,5,7}, tcn_dilation_base {1,2}
  dropout, fc_size, learning_rate may also be tuned

BASELINE: 5-fold CV mean F1 = 0.7792 (GRU, Optuna Trial #91, HBO mt4)
"""
import os
import sys

import torch
import torch.nn as nn
from torch import optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.dataset import fNIRSGraphDataset, get_kfold_loaders_from_json
from core.models import WindowedSpatioTemporalGATNet
from core.training import EarlyStopping, evaluate, train_epoch
from core.transforms import get_transforms
from core.utils import set_seed

_SPLITS_JSON = "/home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/data/splits/kfold_splits_processed_new_mc.json"


def run(data_dir: str) -> float:
    """Train ST-GNN and return 5-fold CV mean F1. Called by CORAL grader."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # TEMPORAL MODULE — primary search target
    # Change temporal_type to explore alternatives to the GRU baseline.
    # =========================================================================
    temporal_type    = "gru"    # "gru" | "lstm" | "bigru" | "transformer" | "tcn"
    temporal_hidden  = 192      # [32..256, step=32] — Optuna best was 192
    temporal_layers  = 1        # [1..3]

    # Transformer-specific (only active when temporal_type="transformer")
    transformer_heads = 4       # {2, 4, 8} — must divide temporal_hidden
    ffn_ratio         = 2       # {2, 4}

    # TCN-specific (only active when temporal_type="tcn")
    tcn_kernel_size   = 5       # {3, 5, 7}
    tcn_dilation_base = 2       # {1, 2}

    # =========================================================================
    # TUNABLE SHARED — may adjust these per temporal_type
    # =========================================================================
    dropout       = 0.30        # [0.1..0.5, step=0.1]
    fc_size       = 256         # [32..256, step=32]
    learning_rate = 0.000635    # [1e-5..1e-1 log] — Optuna best for GRU

    # =========================================================================
    # FIXED — do NOT change
    # =========================================================================
    n_layers      = 2
    n_filters     = 80
    heads         = 2
    use_residual  = False
    use_norm      = True
    norm_type     = "batch"
    window_size   = 16
    window_stride = 8
    epochs        = 100
    patience      = 20
    batch_size    = 8

    # --- Dataset ---
    dataset = fNIRSGraphDataset(
        root=data_dir,
        task_type="GNG",
        data_type="hbo",
        max_trials=4,
        directed=True,
        corr_threshold=0.1,
        self_loops=True,
    )
    stats = dataset.compute_stats()
    transform = get_transforms(stats, augment=False)

    fold_data = get_kfold_loaders_from_json(
        dataset,
        splits_json=_SPLITS_JSON,
        n_splits=5,
        batch_size=batch_size,
        shuffle_train=True,
        num_workers=0,
        pin_memory=False,
        train_transform=transform,
        val_transform=transform,
        verbose=False,
    )

    loss_fn = nn.CrossEntropyLoss()
    fold_f1s = []

    for fold_idx in range(len(fold_data)):
        train_loader, val_loader = fold_data[fold_idx]
        fold_data[fold_idx] = None  # release loader before building next fold

        model = WindowedSpatioTemporalGATNet(
            n_channels=23,
            in_channels=6,
            edge_dim=2,
            window_size=window_size,
            window_stride=window_stride,
            n_layers=n_layers,
            n_filters=n_filters,
            heads=heads,
            use_residual=use_residual,
            use_norm=use_norm,
            norm_type=norm_type,
            temporal_type=temporal_type,
            temporal_hidden=temporal_hidden,
            temporal_layers=temporal_layers,
            transformer_heads=transformer_heads,
            ffn_ratio=ffn_ratio,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dilation_base=tcn_dilation_base,
            fc_size=fc_size,
            dropout=dropout,
            n_classes=2,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        early_stopper = EarlyStopping(patience=patience, mode="max")

        best_val_f1 = 0.0
        best_state = None

        for epoch in range(epochs):
            train_epoch(
                model, train_loader, optimizer, loss_fn, device,
                epoch=epoch, n_epochs=epochs, verbose=False,
            )
            _, _, vl_f1, *_ = evaluate(
                model, val_loader, loss_fn, device,
                epoch=epoch, n_epochs=epochs, verbose=False,
            )
            scheduler.step(vl_f1)
            if vl_f1 > best_val_f1:
                best_val_f1 = vl_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if early_stopper(vl_f1, epoch):
                break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        _, _, final_f1, *_ = evaluate(model, val_loader, loss_fn, device, verbose=False)
        fold_f1s.append(final_f1)
        del train_loader, val_loader

    mean_f1 = sum(fold_f1s) / len(fold_f1s)
    print(f"[{temporal_type}] 5-fold F1s: {[f'{v:.4f}' for v in fold_f1s]} → mean={mean_f1:.4f}")
    return float(mean_f1)
