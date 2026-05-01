# pylint: disable=too-many-arguments, too-many-locals
"""
Optuna hyperparameter search for FlexibleGATNet (src/core baseline GNN).

Two evaluation strategies are supported:

  holdout  — single 80/20 subject-level split (fast, ~1× per trial)
  kfold    — inner k-fold CV using pre-defined JSON splits (stable, ~k× per trial)
             inner_folds=3 reads data["kfold_3"] from the splits JSON (lightweight)
             inner_folds=5 reads data["kfold_5"] from the splits JSON (full)

Run directly:

  # Holdout (default, fastest)
  python -m src.core.optuna_search \\
      --data_dir data/processed-new --data_type hbo \\
      --n_trials 300 --n_epochs 100

  # Lightweight 3-fold inner CV
  python -m src.core.optuna_search \\
      --data_dir data/processed-new --data_type hbo \\
      --n_trials 300 --n_epochs 100 \\
      --eval_strategy kfold --inner_folds 3 \\
      --splits_json data/splits/kfold_splits_processed_new.json

  # Full 5-fold inner CV
  python -m src.core.optuna_search \\
      --data_dir data/processed-new --data_type hbo \\
      --n_trials 300 --n_epochs 100 \\
      --eval_strategy kfold --inner_folds 5 \\
      --splits_json data/splits/kfold_splits_processed_new.json
"""
import argparse
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import optuna
import torch
import torch.nn as nn

from .dataset import fNIRSGraphDataset, get_holdout_loaders, get_kfold_loaders_from_json
from .models import FlexibleGATNet
from .training import FocalLoss, train_epoch, evaluate
from .transforms import get_transforms
from .utils import get_experiment_dir, set_seed


# ---------------------------------------------------------------------------
# Search Space
# ---------------------------------------------------------------------------

def design_search_space(trial: optuna.Trial, use_fl: bool = False) -> Dict:
    """
    Search space for FlexibleGATNet hyperparameter optimization.

    n_filters and heads are suggested independently per layer so that Optuna
    can discover asymmetric architectures (e.g. [112, 32] found by CORAL).
    norm_type is only suggested when use_norm=True.

    Fixed (not suggested):
      - in_channels=6  (mean, min, max, skewness, kurtosis, variance)
      - edge_dim=2     ([abs(corr), coherence])
      - n_classes=2
    """
    n_layers = trial.suggest_int("n_layers", 1, 3, step=1)

    # Per-layer filters and heads — allows [112, 32] asymmetric patterns
    n_filters_list: List[int] = []
    heads_list: List[int] = []
    for i in range(n_layers):
        n_filters_list.append(trial.suggest_int(f"n_filters_{i}", 16, 128, step=16))
        heads_list.append(trial.suggest_int(f"heads_{i}", 2, 8, step=2))

    fc_size = trial.suggest_int("fc_size", 32, 256, step=32)
    use_gine_first_layer = trial.suggest_categorical("use_gine_first_layer", [True, False])
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    use_residual = trial.suggest_categorical("use_residual", [True, False])
    use_norm = trial.suggest_categorical("use_norm", [True, False])
    norm_type = trial.suggest_categorical("norm_type", ["batch", "layer"]) if use_norm else None
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    if use_fl:
        focal_alpha = trial.suggest_float("focal_alpha", 0.1, 0.9)
        focal_gamma = trial.suggest_float("focal_gamma", 0.0, 5.0)
    else:
        focal_alpha = focal_gamma = None

    return {
        "n_layers": n_layers,
        "n_filters": n_filters_list,
        "heads": heads_list,
        "fc_size": fc_size,
        "use_gine_first_layer": use_gine_first_layer,
        "dropout": dropout,
        "use_residual": use_residual,
        "use_norm": use_norm,
        "norm_type": norm_type,
        "learning_rate": learning_rate,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
    }


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------

def _train_single_run(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    n_epochs: int,
    early_stop_patience: int,
    trial: Optional[optuna.Trial] = None,
    step_offset: int = 0,
) -> Tuple[float, int, float, float, Optional[Dict]]:
    """
    Train for up to n_epochs with early stopping; return best fold metrics.

    trial and step_offset enable per-epoch intermediate reporting to Optuna's
    MedianPruner. In the holdout objective step_offset=0 (epochs 0…N-1).
    In the kfold objective step_offset=fold_idx*n_epochs so pruning steps are
    globally ordered across folds (fold0: 0…N-1, fold1: N…2N-1, …).
    Pass trial=None to disable pruning (e.g. during final fold evaluation).

    Returns:
        best_val_f1, best_epoch, train_f1_at_best, train_acc_at_best, best_model_state
    """
    es_min_delta = 1e-4
    es_counter = 0
    es_best_f1 = 0.0

    best_val_f1 = 0.0
    best_epoch = 0
    best_model_state: Optional[Dict] = None
    train_f1_at_best = 0.0
    train_acc_at_best = 0.0

    for epoch in range(n_epochs):
        _, tr_acc, tr_f1 = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            epoch=epoch, n_epochs=n_epochs, verbose=False,
        )
        _, _, vl_f1, *_ = evaluate(
            model, val_loader, loss_fn, device,
            epoch=epoch, n_epochs=n_epochs, verbose=False,
        )
        scheduler.step(vl_f1)

        if trial is not None:
            trial.report(vl_f1, step_offset + epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_epoch = epoch
            train_f1_at_best = tr_f1
            train_acc_at_best = tr_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if vl_f1 > es_best_f1 + es_min_delta:
            es_best_f1 = vl_f1
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= early_stop_patience:
                break

    return best_val_f1, best_epoch, train_f1_at_best, train_acc_at_best, best_model_state


def _build_model(hparams: Dict, device: torch.device) -> FlexibleGATNet:
    return FlexibleGATNet(
        in_channels=6,
        edge_dim=2,
        n_layers=hparams["n_layers"],
        n_filters=hparams["n_filters"],
        heads=hparams["heads"],
        fc_size=hparams["fc_size"],
        dropout=hparams["dropout"],
        n_classes=2,
        use_residual=hparams["use_residual"],
        use_norm=hparams["use_norm"],
        norm_type=hparams["norm_type"] or "batch",
        use_gine_first_layer=hparams["use_gine_first_layer"],
        gine_train_eps=True,
    ).to(device)


def _make_loss_fn(hparams: Dict, use_fl: bool) -> nn.Module:
    if use_fl and hparams["focal_alpha"] is not None:
        return FocalLoss(alpha=hparams["focal_alpha"], gamma=hparams["focal_gamma"])
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Objective — holdout (single 80/20 split)
# ---------------------------------------------------------------------------

def objective(
    dataset: fNIRSGraphDataset,
    stats: Dict,
    trial: optuna.Trial,
    n_epochs: int,
    device: torch.device,
    use_fl: bool = False,
    early_stop_patience: int = 10,
    num_workers: int = 0,
) -> float:
    """Single subject-level holdout objective (val_ratio=0.2, random_state=42)."""
    hparams = design_search_space(trial, use_fl=use_fl)

    transform = get_transforms(stats, augment=False)
    train_loader, val_loader = get_holdout_loaders(
        dataset,
        batch_size=8,
        shuffle_train=True,
        num_workers=num_workers,
        pin_memory=True,
        val_ratio=0.2,
        random_state=42,
        train_transform=transform,
        val_transform=transform,
        verbose=False,
    )

    model = _build_model(hparams, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    loss_fn = _make_loss_fn(hparams, use_fl)

    _, best_epoch, train_f1_at_best, train_acc_at_best, best_model_state = (
        _train_single_run(
            model, train_loader, val_loader, loss_fn, optimizer, scheduler,
            device, n_epochs, early_stop_patience,
            trial=trial, step_offset=0,
        )
    )

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    final_loss, final_acc, final_f1, final_prec, final_sens, final_spec, *_ = evaluate(
        model, val_loader, loss_fn, device, verbose=False
    )

    trial.set_user_attr("val_accuracy", final_acc)
    trial.set_user_attr("val_loss", final_loss)
    trial.set_user_attr("val_f1", final_f1)
    trial.set_user_attr("val_precision", final_prec)
    trial.set_user_attr("val_sensitivity", final_sens)
    trial.set_user_attr("val_specificity", final_spec)
    trial.set_user_attr("train_f1", train_f1_at_best)
    trial.set_user_attr("train_accuracy", train_acc_at_best)
    trial.set_user_attr("total_params", total_params)
    trial.set_user_attr("trainable_params", trainable_params)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("n_filters_list", hparams["n_filters"])
    trial.set_user_attr("heads_list", hparams["heads"])

    return final_f1


# ---------------------------------------------------------------------------
# Objective — inner k-fold CV (3-fold lightweight or 5-fold full)
# ---------------------------------------------------------------------------

def objective_kfold(
    dataset: fNIRSGraphDataset,
    stats: Dict,
    trial: optuna.Trial,
    n_epochs: int,
    device: torch.device,
    splits_json: str,
    inner_folds: int = 3,
    use_fl: bool = False,
    early_stop_patience: int = 10,
    num_workers: int = 0,
) -> float:
    """
    Inner k-fold CV objective — more robust than single holdout.

    Reads fold definitions from splits_json["kfold_{inner_folds}"] (fully
    deterministic, subject-stratified). A fresh model is trained per fold;
    the trial objective is the mean val F1 across all folds.

    Per-epoch intermediate values are reported with globally ordered steps
    (fold_idx * n_epochs + epoch) so MedianPruner can prune within fold 1
    if the trial is clearly underperforming.

    inner_folds=3  → reads kfold_3 (lightweight; ~3× holdout cost)
    inner_folds=5  → reads kfold_5 (full; ~5× holdout cost)
    """
    hparams = design_search_space(trial, use_fl=use_fl)
    loss_fn = _make_loss_fn(hparams, use_fl)
    transform = get_transforms(stats, augment=False)

    fold_data = get_kfold_loaders_from_json(
        dataset,
        splits_json=splits_json,
        n_splits=inner_folds,
        batch_size=8,
        shuffle_train=True,
        num_workers=num_workers,
        pin_memory=True,
        train_transform=transform,
        val_transform=transform,
        verbose=False,
    )

    fold_f1_scores: List[float] = []
    fold_best_epochs: List[int] = []
    total_params: Optional[int] = None
    trainable_params: Optional[int] = None

    for fold_idx in range(len(fold_data)):
        train_loader, val_loader = fold_data[fold_idx]
        fold_data[fold_idx] = None  # release persistent workers from this fold before next

        model = _build_model(hparams, device)

        if total_params is None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

        # step_offset encodes fold position so pruner steps are globally ordered
        _, best_epoch, _, _, best_model_state = _train_single_run(
            model, train_loader, val_loader, loss_fn, optimizer, scheduler,
            device, n_epochs, early_stop_patience,
            trial=trial, step_offset=fold_idx * n_epochs,
        )

        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

        _, _, final_f1, *_ = evaluate(model, val_loader, loss_fn, device, verbose=False)

        fold_f1_scores.append(final_f1)
        fold_best_epochs.append(best_epoch)
        del train_loader, val_loader

    mean_f1 = sum(fold_f1_scores) / len(fold_f1_scores)

    trial.set_user_attr("val_f1_per_fold", fold_f1_scores)
    trial.set_user_attr("best_epoch_per_fold", fold_best_epochs)
    trial.set_user_attr("mean_val_f1", mean_f1)
    trial.set_user_attr("total_params", total_params)
    trial.set_user_attr("trainable_params", trainable_params)
    trial.set_user_attr("n_filters_list", hparams["n_filters"])
    trial.set_user_attr("heads_list", hparams["heads"])

    return mean_f1


# ---------------------------------------------------------------------------
# Logging / callback utilities
# ---------------------------------------------------------------------------

class ProgressCallback:
    """Prints a progress line every N trials without flooding the terminal."""

    def __init__(
        self,
        n_trials: int,
        update_interval: int = 50,
        log_file: Optional[str] = None,
    ) -> None:
        self.n_trials = n_trials
        self.update_interval = update_interval
        self.log_file = log_file
        self.start_time = datetime.now()

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        n = trial.number + 1
        if n % self.update_interval != 0 and n != self.n_trials:
            return
        elapsed = str(datetime.now() - self.start_time).split(".")[0]
        try:
            best_val = f"{study.best_value:.4f}"
            best_num = study.best_trial.number
        except Exception:
            best_val, best_num = "N/A", "N/A"
        val = trial.value
        val_str = f"{val:.4f}" if isinstance(val, float) else "Pruned"
        msg = (
            f"Progress: {n}/{self.n_trials} | Elapsed: {elapsed} | "
            f"Current F1: {val_str} | Best F1: {best_val} (Trial #{best_num})"
        )
        print(msg)
        if self.log_file:
            with open(self.log_file, "a") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts} - {msg}\n")


def setup_optuna_logging(save_dir: str, verbosity: int = logging.WARNING) -> str:
    """Redirect verbose Optuna output to a file; return the log path."""
    optuna.logging.set_verbosity(verbosity)
    log_file = os.path.join(save_dir, "optuna_detailed.log")
    logger = logging.getLogger("optuna")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    print(f"Optuna detailed logs → {log_file}")
    return log_file


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_optuna(
    data_dir: str,
    base_dir: str = "experiments/optuna_core",
    data_type: str = "hbo",
    task_type: str = "GNG",
    max_trials: Optional[int] = None,
    n_trials: int = 300,
    n_epochs: int = 100,
    seed: int = 42,
    use_fl: bool = False,
    early_stop_patience: int = 10,
    update_interval: int = 50,
    n_jobs: int = 1,
    num_workers: int = 0,
    eval_strategy: str = "holdout",
    inner_folds: int = 3,
    splits_json: Optional[str] = None,
    storage_url: Optional[str] = None,
) -> optuna.Study:
    """
    Initialize and run an Optuna study for FlexibleGATNet (src/core).

    eval_strategy="holdout"  → single 80/20 holdout split per trial (fast)
    eval_strategy="kfold"    → inner k-fold CV per trial (robust; requires splits_json)

    storage_url: optional PostgreSQL/MySQL URL for distributed multi-machine runs.
    If None, falls back to a local SQLite file (single-machine only).
    Example: "postgresql://user:pass@host:5432/optuna"
    """
    if eval_strategy == "kfold" and splits_json is None:
        raise ValueError("--splits_json is required when --eval_strategy kfold")
    if eval_strategy not in ("holdout", "kfold"):
        raise ValueError(f"Unknown eval_strategy '{eval_strategy}'; choose holdout or kfold")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    max_tag = max_trials if max_trials is not None else "all"
    kfold_tag = f"_kf{inner_folds}" if eval_strategy == "kfold" else ""
    study_name = (
        f"core_{data_type}_mt{max_tag}_ep{n_epochs}_tr{n_trials}"
        + kfold_tag
        + ("_fl" if use_fl else "")
    )
    save_dir = get_experiment_dir(
        experiment_name=study_name, base_dir=base_dir, overwrite=False
    )
    if storage_url and storage_url.startswith("journal:"):
        from optuna.storages import JournalStorage
        try:
            from optuna.storages import JournalFileBackend
        except ImportError:
            from optuna.storages import JournalFileStorage as JournalFileBackend  # Optuna <3.1
        journal_path = storage_url[len("journal:"):]
        storage_name = JournalStorage(JournalFileBackend(journal_path))
        storage_label = f"JournalFile ({journal_path}) — multi-process safe"
    else:
        storage_name = storage_url or f"sqlite:///{save_dir}/optuna_study.db"
        storage_label = "RDB (distributed)" if storage_url else "SQLite (local)"

    dataset = fNIRSGraphDataset(
        root=data_dir,
        task_type=task_type,
        data_type=data_type,
        max_trials=max_trials,
        directed=True,
        corr_threshold=0.1,
        self_loops=True,
    )
    stats = dataset.compute_stats()

    print(f"Experiment dir  : {save_dir}")
    print(f"Study name      : {study_name}")
    print(f"Storage         : {storage_label}")
    print(f"Eval strategy   : {eval_strategy}" + (f" ({inner_folds}-fold)" if eval_strategy == "kfold" else ""))
    print(f"Dataset size    : {len(dataset)} graphs")
    print(f"Device          : {device}")
    print(f"Trials / Epochs : {n_trials} / {n_epochs}")
    print()

    optuna_log_file = setup_optuna_logging(save_dir, verbosity=logging.WARNING)
    progress_log = os.path.join(save_dir, "progress.log")
    progress_cb = ProgressCallback(
        n_trials=n_trials,
        update_interval=update_interval,
        log_file=progress_log,
    )

    # For kfold, n_warmup_steps covers the first fold's epochs before pruning begins
    warmup_steps = 20 if eval_strategy == "holdout" else n_epochs + 20

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=15,
            n_warmup_steps=warmup_steps,
            interval_steps=1,
        ),
        sampler=optuna.samplers.TPESampler(seed=seed),
        load_if_exists=True,
    )

    if eval_strategy == "holdout":
        obj_fn = lambda trial: objective(
            dataset, stats, trial, n_epochs, device,
            use_fl=use_fl,
            early_stop_patience=early_stop_patience,
            num_workers=num_workers,
        )
    else:
        obj_fn = lambda trial: objective_kfold(
            dataset, stats, trial, n_epochs, device,
            splits_json=splits_json,
            inner_folds=inner_folds,
            use_fl=use_fl,
            early_stop_patience=early_stop_patience,
            num_workers=num_workers,
        )

    study.optimize(
        obj_fn,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[progress_cb],
        show_progress_bar=False,
    )

    from optuna.trial import TrialState
    pruned = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print()
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"  Total    : {len(study.trials)}")
    print(f"  Pruned   : {len(pruned)}")
    print(f"  Complete : {len(complete)}")
    print(f"  Best F1  : {study.best_value:.4f} (Trial #{study.best_trial.number})")
    print(f"  Best params:\n    {study.best_params}")
    print(f"  Logs     : {optuna_log_file}")
    print(f"  Progress : {progress_log}")
    return study


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for FlexibleGATNet (src/core)"
    )
    p.add_argument("--data_dir", required=True,
                   help="Root data directory (e.g. data/processed-new); dataset appends task_type internally")
    p.add_argument("--base_dir", default="experiments/optuna_core")
    p.add_argument("--data_type", default="hbo", choices=["hbo", "hbr", "hbt"])
    p.add_argument("--task_type", default="GNG")
    p.add_argument("--max_trials", type=int, default=None,
                   help="Max trial files per subject (None=all; 2 recommended based on k-fold findings)")
    p.add_argument("--n_trials", type=int, default=300, help="Number of Optuna trials")
    p.add_argument("--n_epochs", type=int, default=100, help="Epochs per trial")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_fl", action="store_true", help="Include focal-loss params in search")
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--update_interval", type=int, default=50,
                   help="Print progress every N trials")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel workers (keep 1 for SQLite; >1 requires PostgreSQL/MySQL)")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader worker subprocesses per trial (4 recommended on GPU)")
    # Evaluation strategy
    p.add_argument("--eval_strategy", default="holdout", choices=["holdout", "kfold"],
                   help="holdout: single 80/20 split (fast); kfold: inner CV (robust)")
    p.add_argument("--inner_folds", type=int, default=3, choices=[3, 5],
                   help="Inner folds for kfold strategy: 3 (lightweight) or 5 (full)")
    p.add_argument("--splits_json", type=str, default=None,
                   help="Path to pre-defined splits JSON (required for --eval_strategy kfold)")
    p.add_argument("--storage_url", type=str, default=None,
                   help="PostgreSQL/MySQL URL for distributed multi-machine runs (e.g. postgresql://user:pass@host:5432/optuna). Omit to use local SQLite.")
    args = p.parse_args()
    run_optuna(**vars(args))
