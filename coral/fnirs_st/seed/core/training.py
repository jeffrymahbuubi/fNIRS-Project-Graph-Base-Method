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
        self.mode = mode
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
