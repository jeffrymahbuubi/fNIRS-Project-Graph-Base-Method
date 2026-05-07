"""SG (FlexibleGATNet) explainer — SPEC §5.

Three estimators are supported via `cfg.estimator`:

- `'gnn'`        : GNNExplainer (primary, SPEC §5.2). Stochastic per-trial
                   optimisation; learns a [23, 6] attribute mask.
- `'captum_ig'`  : CaptumExplainer with `IntegratedGradients` (cross-check,
                   SPEC §5.3 / §15.7). Same mask shapes as GNNExplainer;
                   per-trial reductions identical. Requires `captum`.
- `'attention'`  : AttentionExplainer (cross-check, SPEC §5.3 / §15.4).
                   Auto-extracts GATv2 attention; essentially free at
                   inference. Edge-mask only — no node_mask is produced,
                   so channel_importance is derived from the row-sum of
                   the symmetric pair matrix and feature_importance is
                   reported as None.

SPEC §11 C4 compares Spearman ρ between any two of the three population-
level rankings. The notebook driver runs the cell three times with
different `cfg.estimator` values and computes ρ.

Public API:
- `ProbWrapper(model)` — wraps FlexibleGATNet so the Explainer sees a
  softmax-output, multiclass model (SPEC §5.2 step 4 / §3.1).
- `explain_checkpoint(loaded, cfg) -> List[TrialAttribution]` — runs the
  selected estimator on every val graph of one fold/subject.
- `run_sg(cfg) -> PopulationResult` — full matrix entry: discovers all
  checkpoints for the (arch, hb, regime, mt) cell, explains them, and
  aggregates into the SPEC §7.3 deliverable.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import (
    AttentionExplainer,
    CaptumExplainer,
    GNNExplainer,
)

from src.xai.aggregate import PopulationResult, TrialAttribution, aggregate_population
from src.xai.channels import N_CH
from src.xai.checkpoints import (
    CheckpointInfo,
    LoadedCheckpoint,
    discover_checkpoints,
    load_checkpoint,
)
from src.xai.config import XAIRunConfig


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class ProbWrapper(nn.Module):
    """SPEC §5.2 step 4 — softmax over the 2-class logits.

    `Explainer` with `model_config.return_type='probs'` expects probabilities;
    the underlying `FlexibleGATNet` returns raw logits. `batch=None` defaults
    to a single-graph batch vector so the wrapper works whether or not the
    explainer threads `batch` in via `**kwargs`.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if edge_attr is None:
            # FlexibleGATNet's forward is typed as edge_attr: Tensor (non-optional).
            # Defensive zero-feature: mirrors the dataset's empty-edge case.
            edge_attr = torch.zeros((edge_index.size(1), 2), device=x.device)
        return F.softmax(self.model(x, edge_index, edge_attr, batch), dim=-1)


# ---------------------------------------------------------------------------
# Per-trial explainer call
# ---------------------------------------------------------------------------


_MODEL_CONFIG = dict(
    mode="multiclass_classification",
    task_level="graph",
    return_type="probs",
)


def _build_gnn_explainer(model: nn.Module, cfg: XAIRunConfig) -> Explainer:
    """SPEC §5.2 step 5 — primary path."""
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=cfg.gnn_explainer_epochs, lr=cfg.gnn_explainer_lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=_MODEL_CONFIG,
    )


def _build_captum_explainer(model: nn.Module, cfg: XAIRunConfig) -> Explainer:
    """SPEC §5.3 / §15.7 — IntegratedGradients via Captum."""
    return Explainer(
        model=model,
        algorithm=CaptumExplainer("IntegratedGradients"),
        explanation_type="model",
        node_mask_type="attributes",     # CaptumExplainer accepts None | 'attributes'
        edge_mask_type="object",
        model_config=_MODEL_CONFIG,
    )


def _build_attention_explainer(model: nn.Module, cfg: XAIRunConfig) -> Explainer:
    """SPEC §15.4 — auto-extract GATv2 attention; edge-mask only.

    AttentionExplainer.supports() rejects any node_mask_type, so this
    estimator produces no node_mask. The reductions in
    `_per_trial_reductions` derive channel_importance from the row-sum of
    the symmetric pair matrix instead.
    """
    return Explainer(
        model=model,
        algorithm=AttentionExplainer(reduce="mean"),   # SPEC §6.3 default
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=_MODEL_CONFIG,
    )


def _build_explainer(model: nn.Module, cfg: XAIRunConfig) -> Explainer:
    if cfg.estimator == "gnn":
        return _build_gnn_explainer(model, cfg)
    if cfg.estimator == "captum_ig":
        return _build_captum_explainer(model, cfg)
    if cfg.estimator == "attention":
        return _build_attention_explainer(model, cfg)
    raise ValueError(f"unknown cfg.estimator={cfg.estimator!r}")


def _per_trial_reductions(
    node_mask: Optional[torch.Tensor],   # [N, F] for gnn/captum_ig; None for attention
    edge_mask: torch.Tensor,             # [E]
    edge_index: torch.Tensor,             # [2, E]
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """SPEC §5.2 step 7 — channel × feature × pair-matrix reductions.

    For estimators that produce a node_mask (gnn, captum_ig):
      channel_importance = |node_mask|.sum(features)
      feature_importance = |node_mask|.sum(channels)
    For attention (no node_mask):
      channel_importance = row-sum of the symmetric pair matrix
      feature_importance = None  (attention has no per-feature breakdown)

    The dataset is built directed (`directed=True`), so each channel pair
    contributes two edges (i→j and j→i) with identical edge_attr. Sum the
    edge_mask onto an asymmetric (i, j) accumulator and symmetrise — this
    is loss-free per SPEC §5.2 step 7.
    """
    pair = np.zeros((N_CH, N_CH), dtype=np.float32)
    ei = edge_index.cpu().numpy()
    em = edge_mask.detach().cpu().numpy()
    for e in range(em.shape[0]):
        pair[int(ei[0, e]), int(ei[1, e])] += float(em[e])
    pair = (pair + pair.T) / 2.0

    if node_mask is None:
        channel_importance = pair.sum(axis=1).astype(np.float32)
        feature_importance: Optional[np.ndarray] = None
    else:
        abs_node = node_mask.abs()
        channel_importance = abs_node.sum(dim=1).cpu().numpy().astype(np.float32)
        feature_importance = abs_node.sum(dim=0).cpu().numpy().astype(np.float32)
    return channel_importance, feature_importance, pair


def explain_checkpoint(loaded: LoadedCheckpoint, cfg: XAIRunConfig) -> List[TrialAttribution]:
    """Run the configured estimator (`cfg.estimator`) on every val graph.

    'gnn' / 'captum_ig' are stochastic / iterative; `torch.manual_seed(cfg.seed)`
    is reset per trial for reproducibility (SPEC §10.5). 'attention' is
    deterministic — `cfg.seed` is reset for consistency but has no effect
    on the output.
    """
    if cfg.arch != "sg":
        raise ValueError(f"explain_checkpoint(SG) requires cfg.arch='sg', got {cfg.arch!r}")

    device = torch.device(cfg.device)
    model = loaded.model.to(device).eval()
    prob_model = ProbWrapper(model).to(device).eval()
    explainer = _build_explainer(prob_model, cfg)

    trial_atts: List[TrialAttribution] = []
    trial_idx_per_subject: dict[str, int] = {}

    for idx in loaded.val_indices:
        raw_data: Data = loaded.dataset[idx]
        sid = str(raw_data.subject_id)
        t_idx = trial_idx_per_subject.get(sid, 0)
        trial_idx_per_subject[sid] = t_idx + 1

        data = loaded.val_transform(raw_data).to(device)
        true_label = int(data.y.item() if data.y.dim() == 0 else data.y[0].item())
        batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.edge_attr, batch_vec)
            pred_label = int(logits.argmax(dim=-1).item())

        # Reproducible explainer per trial (SPEC §10.5).
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # PyG 2.7.0: with explanation_type='model', `target` is auto-filled
        # from the model's argmax — passing it triggers a UserWarning.
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=batch_vec,
        )

        # AttentionExplainer doesn't populate a node_mask; the others do.
        node_mask = getattr(explanation, "node_mask", None)
        ch_imp, feat_imp, pair = _per_trial_reductions(
            node_mask, explanation.edge_mask, data.edge_index,
        )
        trial_atts.append(TrialAttribution(
            subject_id=sid,
            fold_or_subj_label=loaded.info.label,
            trial_idx_in_subject=t_idx,
            true_label=true_label,
            pred_label=pred_label,
            included=(pred_label == true_label) if not cfg.include_misclassified else True,
            channel_importance=ch_imp,
            pair_matrix=pair,
            feature_importance=feat_imp,
        ))

    return trial_atts


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def run_sg(cfg: XAIRunConfig) -> PopulationResult:
    """Discover → explain every checkpoint → aggregate to PopulationResult.

    Writes nothing on its own; the caller is expected to invoke
    `result.to_csv(cfg.output_dir)` to persist the §7.3 deliverable.
    """
    if cfg.arch != "sg":
        raise ValueError(f"run_sg requires cfg.arch='sg', got {cfg.arch!r}")

    infos: List[CheckpointInfo] = discover_checkpoints(
        cfg.experiment_root, arch="sg", hb=cfg.hb, regime=cfg.regime, mt=cfg.mt,
    )
    if not infos:
        raise FileNotFoundError(
            f"no SG checkpoints under {cfg.experiment_root!r} matching "
            f"hb={cfg.hb} regime={cfg.regime} mt={cfg.mt}"
        )

    all_trials: List[TrialAttribution] = []
    path_rebases: List[dict] = []
    for info in infos:
        loaded = load_checkpoint(info, cfg)
        path_rebases.extend(loaded.path_rebases)
        all_trials.extend(explain_checkpoint(loaded, cfg))

    extras: dict = {
        "estimator": cfg.estimator,
        "gnn_explainer_epochs": cfg.gnn_explainer_epochs,
        "gnn_explainer_lr": cfg.gnn_explainer_lr,
        "n_checkpoints": len(infos),
        "checkpoints": [i.label for i in infos],
        "path_rebases": path_rebases,
    }
    return aggregate_population(
        all_trials,
        arch="sg", hb=cfg.hb, regime=cfg.regime, mt=cfg.mt,
        only_included=not cfg.include_misclassified,
        extras=extras,
    )
