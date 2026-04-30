from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union
import os
import random

import numpy as np
import torch
import yaml


@dataclass
class SystemConfig:
    seed: int = 42
    use_cuda: bool = True
    deterministic: bool = True


@dataclass
class ExperimentConfig:
    # Data
    data_dir: str = ""
    save_dir: str = "experiments"
    task_type: str = "GNG"
    data_type: str = "hbo"
    max_trials: Optional[int] = None
    directed: bool = False
    corr_threshold: float = 0.1
    self_loops: bool = False
    # Model
    n_layers: int = 2
    n_filters: Union[int, List[int]] = 64
    n_heads: Union[int, List[int]] = 4
    fc_size: int = 64
    dropout: float = 0.5
    use_residual: bool = True
    use_norm: bool = False
    norm_type: str = "batch"
    use_gine_first_layer: bool = False
    # Training
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-3
    patience: int = 10
    checkpoint_metric: str = "f1"
    use_class_weights: bool = False
    sqrt_class_weights: bool = False
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # Augmentation
    augment: bool = False
    edge_dropout_p: float = 0.1
    feature_mask_p: float = 0.1
    feature_mask_mode: str = "all"
    use_rwpe: bool = False
    rwpe_walk_length: int = 4
    # Validation
    validation: str = "holdout"
    k_folds: int = 5
    val_ratio: float = 0.2
    random_state: int = 42
    splits_json: Optional[str] = None
    # Misc
    num_workers: int = 0
    pin_memory: bool = False
    resume: bool = False


def setup_system(cfg: SystemConfig) -> None:
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_config(cfg: ExperimentConfig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**{k: v for k, v in data.items() if k in ExperimentConfig.__dataclass_fields__})
