# pylint: disable=too-many-arguments, too-many-locals
import os
import pickle as pk
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import LRScheduler

from .dataset import get_holdout_loaders, get_kfold_loaders, get_kfold_loaders_from_json, get_loso_loaders


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce)
        return (self.alpha * (1 - p_t) ** self.gamma * ce).mean()


class CosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup: int, max_iters: int):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * self._factor(self.last_epoch) for base_lr in self.base_lrs]

    def _factor(self, epoch: int) -> float:
        f = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            f *= epoch / max(self.warmup, 1)
        return f


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "max": higher is better (F1); "min": lower is better (loss)
        self.counter = 0
        self.best_score: float = None
        self.best_epoch: int = 0
        self.early_stop: bool = False

    def __call__(self, score: float, epoch: int) -> bool:
        if self.mode == "min":
            improved = self.best_score is None or score < self.best_score - self.min_delta
        else:
            improved = self.best_score is None or score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience} no improvement")
        if self.counter >= self.patience:
            label = "Loss" if self.mode == "min" else "F1"
            print(f"  EarlyStopping triggered at epoch {epoch + 1}. "
                  f"Best {label}={self.best_score:.4f} at epoch {self.best_epoch + 1}")
            self.early_stop = True
        return self.early_stop


def _compute_class_weights(loader, device, use_sqrt: bool = False) -> torch.Tensor:
    labels = np.array([int(b.y[i].item()) for b in loader for i in range(b.y.shape[0])])
    w = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    if use_sqrt:
        w = np.sqrt(w)
        w = w / w.min()
    return torch.tensor(w, dtype=torch.float).to(device)


def _make_loss_fn(cfg, train_loader, device) -> nn.Module:
    if cfg.use_focal_loss:
        return FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma).to(device)
    if cfg.use_class_weights:
        w = _compute_class_weights(train_loader, device, use_sqrt=cfg.sqrt_class_weights)
        print(f"  Class weights: {w.cpu().numpy()} (sqrt={cfg.sqrt_class_weights})")
        return nn.CrossEntropyLoss(weight=w)
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Core training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model, train_loader, optimizer, loss_fn, device,
    epoch: int = None, n_epochs: int = None,
    verbose: bool = True, log_freq: int = 10,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    acc_m = torchmetrics.Accuracy(task="binary").to(device)
    f1_m = torchmetrics.F1Score(task="binary").to(device)
    n_batches = len(train_loader)
    ep_str = f"{epoch + 1}/{n_epochs}" if epoch is not None and n_epochs else "?"

    for idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = out.argmax(dim=-1)
        acc_m.update(preds, batch.y)
        f1_m.update(preds, batch.y)
        cur = idx + 1
        if verbose and (cur % log_freq == 0 or cur == n_batches):
            print(f"  Ep [{ep_str}] Step [{cur}/{n_batches}] "
                  f"Loss={total_loss / cur:.4f} Acc={acc_m.compute():.4f} F1={f1_m.compute():.4f}")

    return total_loss / n_batches, acc_m.compute().item(), f1_m.compute().item()


def evaluate(
    model, val_loader, loss_fn, device,
    epoch: int = None, n_epochs: int = None,
    verbose: bool = True,
) -> Tuple[float, float, float, float, float, float, np.ndarray, List, List]:
    model.eval()
    total_loss = 0.0
    acc_m = torchmetrics.Accuracy(task="binary").to(device)
    f1_m = torchmetrics.F1Score(task="binary").to(device)
    prec_m = torchmetrics.Precision(task="binary").to(device)
    rec_m = torchmetrics.Recall(task="binary").to(device)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            total_loss += loss_fn(out, batch.y).item()
            preds = out.argmax(dim=-1)
            acc_m.update(preds, batch.y)
            f1_m.update(preds, batch.y)
            prec_m.update(preds, batch.y)
            rec_m.update(preds, batch.y)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    accuracy = acc_m.compute().item()
    f1 = f1_m.compute().item()
    precision = prec_m.compute().item()
    sensitivity = rec_m.compute().item()
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp_v = (cm[0, 0], cm[0, 1]) if cm.shape == (2, 2) else (0, 0)
    specificity = tn / (tn + fp_v) if (tn + fp_v) > 0 else 0.0

    if verbose:
        ep_str = f" Ep [{epoch + 1}/{n_epochs}]" if epoch is not None and n_epochs else ""
        print(f"Val{ep_str}: Loss={avg_loss:.4f} Acc={accuracy:.4f} F1={f1:.4f} "
              f"Prec={precision:.4f} Sens={sensitivity:.4f} Spec={specificity:.4f}")

    return avg_loss, accuracy, f1, precision, sensitivity, specificity, cm, all_labels, all_preds


# ---------------------------------------------------------------------------
# Visualization / persistence helpers
# ---------------------------------------------------------------------------

def plot_training_curves(history: Dict, save_dir: str, name: str, best_epoch: int = None) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for key, title in [("loss", "Loss"), ("accuracy", "Accuracy"), ("f1", "F1 Score")]:
        tr_vals = history.get(f"train_{key}", [])
        vl_vals = history.get(f"val_{key}", [])
        if not tr_vals:
            continue
        epochs = range(1, len(tr_vals) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, tr_vals, label="Train", color="steelblue")
        ax.plot(epochs, vl_vals, label="Val", color="darkorange")
        if best_epoch is not None and best_epoch < len(vl_vals):
            ax.axvline(x=best_epoch + 1, color="green", linestyle="--", alpha=0.7,
                       label=f"Best (ep {best_epoch + 1})")
            ax.scatter([best_epoch + 1], [vl_vals[best_epoch]], color="green", s=80, zorder=5)
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.set_title(f"{title} — {name}")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{name}_{key}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_confusion_matrix(cm: np.ndarray, save_dir: str, name: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    labels = ["Healthy", "Anxiety"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{name}_cm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics(results: Dict, save_dir: str, name: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{name}.pkl"), "wb") as f:
        pk.dump(results, f)


def _reset_model(model: nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def _empty_fold_metrics() -> Dict:
    return {
        "train_loss": [], "train_accuracy": [], "train_f1": [],
        "val_loss": [], "val_accuracy": [], "val_f1": [],
        "accuracies": [], "precisions": [], "sensitivity": [], "specificity": [], "f1_scores": [],
        "conf_matrix": np.zeros((2, 2), dtype=int),
        "true_labels": [], "pred_labels": [],
    }


def _run_fold(model, optimizer, scheduler, train_loader, val_loader, device,
              cfg) -> Tuple[Dict, int, Dict]:
    checkpoint_metric = getattr(cfg, "checkpoint_metric", "f1")
    mode = "min" if checkpoint_metric == "loss" else "max"
    early_stopper = EarlyStopping(patience=cfg.patience, mode=mode)
    loss_fn = _make_loss_fn(cfg, train_loader, device)
    best_model_state = None
    best_val_score = float("inf") if checkpoint_metric == "loss" else -1.0
    best_epoch = 0
    history = {k: [] for k in ["train_loss", "train_accuracy", "train_f1",
                                "val_loss", "val_accuracy", "val_f1"]}

    for epoch in range(cfg.epochs):
        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            epoch=epoch, n_epochs=cfg.epochs, verbose=False,
        )
        vl_loss, vl_acc, vl_f1, *_ = evaluate(
            model, val_loader, loss_fn, device,
            epoch=epoch, n_epochs=cfg.epochs, verbose=False,
        )
        for k, v in zip(
            ["train_loss", "train_accuracy", "train_f1", "val_loss", "val_accuracy", "val_f1"],
            [tr_loss, tr_acc, tr_f1, vl_loss, vl_acc, vl_f1],
        ):
            history[k].append(v)
        print(f"Ep {epoch + 1:>4}: TR L={tr_loss:.4f} Acc={tr_acc:.4f} F1={tr_f1:.4f} | "
              f"VL L={vl_loss:.4f} Acc={vl_acc:.4f} F1={vl_f1:.4f}")
        monitor_val = vl_loss if checkpoint_metric == "loss" else vl_f1
        improved = (vl_loss < best_val_score) if checkpoint_metric == "loss" else (vl_f1 > best_val_score)
        if improved:
            best_val_score = monitor_val
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if early_stopper(monitor_val, epoch):
            break
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_val)
            else:
                scheduler.step()

    return history, best_epoch, best_model_state


def _collect_fold_results(model, val_loader, device, cfg, best_model_state,
                          fold_metrics: Dict, history: Dict) -> Tuple:
    loss_fn = _make_loss_fn(cfg, val_loader, device)
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    _, acc, f1, precision, sensitivity, specificity, cm, true_labels, pred_labels = evaluate(
        model, val_loader, loss_fn, device, verbose=True
    )
    for key in ["train_loss", "train_accuracy", "train_f1", "val_loss", "val_accuracy", "val_f1"]:
        fold_metrics[key].append(history[key])
    fold_metrics["accuracies"].append(acc)
    fold_metrics["precisions"].append(precision)
    fold_metrics["sensitivity"].append(sensitivity)
    fold_metrics["specificity"].append(specificity)
    fold_metrics["f1_scores"].append(f1)
    fold_metrics["conf_matrix"] += cm
    fold_metrics["true_labels"].extend(true_labels)
    fold_metrics["pred_labels"].extend(pred_labels)
    return acc, f1, precision, sensitivity, specificity, cm, true_labels, pred_labels


def _compute_overall_metrics(fold_metrics: Dict, save_dir: str, name: str, suffix: str) -> None:
    cm = fold_metrics["conf_matrix"]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    total = int(tn + fp + fn + tp)
    overall = {
        "overall_accuracy": (tp + tn) / total if total > 0 else 0.0,
        "overall_precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "overall_sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "overall_specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "overall_f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        "mean_accuracy": float(np.mean(fold_metrics["accuracies"])),
        "mean_f1": float(np.mean(fold_metrics["f1_scores"])),
        "mean_precision": float(np.mean(fold_metrics["precisions"])),
        "mean_sensitivity": float(np.mean(fold_metrics["sensitivity"])),
        "mean_specificity": float(np.mean(fold_metrics["specificity"])),
        "confusion_matrix": cm,
        "true_labels": fold_metrics["true_labels"],
        "pred_labels": fold_metrics["pred_labels"],
    }
    tag = f"{name}_{suffix}_overall"
    _plot_confusion_matrix(cm, save_dir, tag)
    save_metrics(overall, save_dir, tag)
    print(f"\n=== {suffix.upper()} Overall Results ===")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Public training entry points
# ---------------------------------------------------------------------------

def perform_holdout_training(model, dataset, cfg, device, exp_dir: str, model_name: str,
                              train_transform, val_transform,
                              optimizer_class, optimizer_params,
                              scheduler_class, scheduler_params) -> Dict:
    train_loader, val_loader = get_holdout_loaders(
        dataset,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        random_state=cfg.random_state,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        train_transform=train_transform,
        val_transform=val_transform,
    )
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    scheduler = scheduler_class(optimizer, **scheduler_params) if scheduler_class else None
    t0 = time.time()
    print(f"\n=== Holdout Training ===")
    history, best_epoch, best_model_state = _run_fold(
        model, optimizer, scheduler, train_loader, val_loader, device, cfg
    )
    print(f"Holdout done in {time.time() - t0:.1f}s. Best epoch: {best_epoch + 1}")
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        torch.save(model.state_dict(), os.path.join(exp_dir, f"{model_name}_holdout.pt"))
    loss_fn = _make_loss_fn(cfg, val_loader, device)
    _, acc, f1, precision, sensitivity, specificity, cm, true_labels, pred_labels = evaluate(
        model, val_loader, loss_fn, device, verbose=True
    )
    name = f"{model_name}_holdout"
    plot_training_curves(history, exp_dir, name, best_epoch)
    _plot_confusion_matrix(cm, exp_dir, name)
    results = {
        **history, "best_epoch": best_epoch,
        "accuracy": acc, "f1_score": f1, "precision": precision,
        "sensitivity": sensitivity, "specificity": specificity,
        "conf_matrix": cm, "true_labels": true_labels, "pred_labels": pred_labels,
    }
    save_metrics(results, exp_dir, name)
    return results


def perform_kfold_training(model, dataset, cfg, device, exp_dir: str, model_name: str,
                            train_transform, val_transform,
                            optimizer_class, optimizer_params,
                            scheduler_class, scheduler_params,
                            resume: bool = False,
                            splits_json: Optional[str] = None) -> Dict:
    if splits_json is not None:
        print(f"Using pre-defined splits from: {splits_json}")
        fold_data = get_kfold_loaders_from_json(
            dataset,
            splits_json=splits_json,
            n_splits=cfg.k_folds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            train_transform=train_transform,
            val_transform=val_transform,
        )
    else:
        fold_data = get_kfold_loaders(
            dataset,
            n_splits=cfg.k_folds,
            batch_size=cfg.batch_size,
            random_state=cfg.random_state,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            train_transform=train_transform,
            val_transform=val_transform,
        )
    os.makedirs(exp_dir, exist_ok=True)
    fold_metrics = _empty_fold_metrics()

    for fold_idx, (train_loader, val_loader) in enumerate(fold_data):
        fold_name = f"{model_name}_fold_{fold_idx + 1}"
        fold_pt = os.path.join(exp_dir, f"{fold_name}.pt")
        fold_pkl = os.path.join(exp_dir, f"{fold_name}.pkl")

        if resume and os.path.exists(fold_pt) and os.path.exists(fold_pkl):
            print(f"\nFold {fold_idx + 1}/{cfg.k_folds} [SKIPPED — already trained]")
            with open(fold_pkl, "rb") as f:
                saved = pk.load(f)
            for key in ["train_loss", "train_accuracy", "train_f1", "val_loss", "val_accuracy", "val_f1"]:
                fold_metrics[key].append(saved.get(key, []))
            fold_metrics["accuracies"].append(saved["accuracy"])
            fold_metrics["precisions"].append(saved["precision"])
            fold_metrics["sensitivity"].append(saved["sensitivity"])
            fold_metrics["specificity"].append(saved["specificity"])
            fold_metrics["f1_scores"].append(saved["f1_score"])
            fold_metrics["conf_matrix"] += saved["conf_matrix"]
            fold_metrics["true_labels"].extend(saved["true_labels"])
            fold_metrics["pred_labels"].extend(saved["pred_labels"])
            continue

        print(f"\n=== K-Fold {fold_idx + 1}/{cfg.k_folds} ===")
        t0 = time.time()
        _reset_model(model)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = scheduler_class(optimizer, **scheduler_params) if scheduler_class else None
        history, best_epoch, best_model_state = _run_fold(
            model, optimizer, scheduler, train_loader, val_loader, device, cfg
        )
        acc, f1, precision, sensitivity, specificity, cm, true_labels, pred_labels = _collect_fold_results(
            model, val_loader, device, cfg, best_model_state, fold_metrics, history
        )
        torch.save(model.state_dict(), fold_pt)
        plot_training_curves(history, exp_dir, fold_name, best_epoch)
        _plot_confusion_matrix(cm, exp_dir, fold_name)
        save_metrics({
            **history, "best_epoch": best_epoch,
            "accuracy": acc, "f1_score": f1, "precision": precision,
            "sensitivity": sensitivity, "specificity": specificity,
            "conf_matrix": cm, "true_labels": true_labels, "pred_labels": pred_labels,
        }, exp_dir, fold_name)
        print(f"Fold {fold_idx + 1} done in {time.time() - t0:.1f}s")

    _compute_overall_metrics(fold_metrics, exp_dir, model_name, "kfold")
    return fold_metrics


def perform_loso_training(model, dataset, cfg, device, exp_dir: str, model_name: str,
                           train_transform, val_transform,
                           optimizer_class, optimizer_params,
                           scheduler_class, scheduler_params,
                           resume: bool = False) -> Dict:
    fold_data = get_loso_loaders(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        train_transform=train_transform,
        val_transform=val_transform,
    )
    os.makedirs(exp_dir, exist_ok=True)
    fold_metrics = _empty_fold_metrics()
    n_folds = len(fold_data)

    for fold_idx, (train_loader, val_loader, val_subject) in enumerate(fold_data):
        subj_name = f"{model_name}_subj_{val_subject}"
        subj_pt = os.path.join(exp_dir, f"{subj_name}.pt")
        subj_pkl = os.path.join(exp_dir, f"{subj_name}.pkl")

        if resume and os.path.exists(subj_pt) and os.path.exists(subj_pkl):
            print(f"\nLOSO {fold_idx + 1}/{n_folds} — Subject {val_subject} [SKIPPED]")
            with open(subj_pkl, "rb") as f:
                saved = pk.load(f)
            for key in ["train_loss", "train_accuracy", "train_f1", "val_loss", "val_accuracy", "val_f1"]:
                fold_metrics[key].append(saved.get(key, []))
            fold_metrics["accuracies"].append(saved["accuracy"])
            fold_metrics["precisions"].append(saved["precision"])
            fold_metrics["sensitivity"].append(saved["sensitivity"])
            fold_metrics["specificity"].append(saved["specificity"])
            fold_metrics["f1_scores"].append(saved["f1_score"])
            fold_metrics["conf_matrix"] += saved["conf_matrix"]
            fold_metrics["true_labels"].extend(saved["true_labels"])
            fold_metrics["pred_labels"].extend(saved["pred_labels"])
            continue

        print(f"\n=== LOSO {fold_idx + 1}/{n_folds} — Subject: {val_subject} ===")
        t0 = time.time()
        _reset_model(model)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = scheduler_class(optimizer, **scheduler_params) if scheduler_class else None
        history, best_epoch, best_model_state = _run_fold(
            model, optimizer, scheduler, train_loader, val_loader, device, cfg
        )
        acc, f1, precision, sensitivity, specificity, cm, true_labels, pred_labels = _collect_fold_results(
            model, val_loader, device, cfg, best_model_state, fold_metrics, history
        )
        torch.save(model.state_dict(), subj_pt)
        plot_training_curves(history, exp_dir, subj_name, best_epoch)
        _plot_confusion_matrix(cm, exp_dir, subj_name)
        save_metrics({
            **history, "best_epoch": best_epoch, "val_subject": val_subject,
            "accuracy": acc, "f1_score": f1, "precision": precision,
            "sensitivity": sensitivity, "specificity": specificity,
            "conf_matrix": cm, "true_labels": true_labels, "pred_labels": pred_labels,
        }, exp_dir, subj_name)
        print(f"LOSO fold {fold_idx + 1} done in {time.time() - t0:.1f}s")

    _compute_overall_metrics(fold_metrics, exp_dir, model_name, "loso")
    return fold_metrics
