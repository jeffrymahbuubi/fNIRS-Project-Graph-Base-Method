"""ST (WindowedSpatioTemporalGATNet) explainer — SPEC §6.

Primary path (`explain_checkpoint` + `run_st`): native attention via the
model's built-in `model.explain()`. Essentially free at inference — no
per-trial optimisation, no extra training. The model already captures
GATv2 attention (per window, per layer, per head) and additive temporal
attention α_k during eval-mode forward; `model.explain()` just exposes
those tensors.

Per-trial reductions follow SPEC §6.1:
    spatial_attention[k][l]: [E, heads]
        ↓ mean / max over heads      (cfg.head_reduce)
        ↓ mean / last over layers     (cfg.layer_reduce)
    per-window edge attention [E]
        ↓ Σ_k  α_k * attn_k
    trial pair-edge scores [E]
        ↓ scatter onto 23×23 by edge_index, symmetrise
    pair_matrix [23, 23]
        ↓ row-sum
    channel_importance [23]

Supplementary path (`explain_supplementary_checkpoint` + `run_st_supplementary`):
GNNExplainer with `node_mask_type='object'` and `edge_mask_type='object'`
(SPEC §6.4). The 'object' node mask is `[23]` rather than `[23, T≈326]`,
which keeps the cross-check tractable on ST. Used as a method-independent
sanity check on which channels matter; not as a replacement for the native
attention path. Gated behind `cfg.run_supplementary_gnnexplainer` at the
notebook level.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

from src.xai.aggregate import PopulationResult, TrialAttribution, aggregate_population
from src.xai.channels import N_CH
from src.xai.checkpoints import (
    CheckpointInfo,
    LoadedCheckpoint,
    discover_checkpoints,
    load_checkpoint,
)
from src.xai.config import XAIRunConfig
from src.xai.sg_explainer import ProbWrapper   # generic softmax wrapper, reused

# Project convention: 10 Hz fNIRS sampling rate. Matches the
# `_get_fs()` fallback in src/core_st/dataset.py and the device files in
# data/processed-new-mc/. Window→time conversion only — has no effect on
# attention magnitudes, only the t_start_s / t_end_s columns of
# temporal_attention.csv.
_FNIRS_FS: float = 10.0


# ---------------------------------------------------------------------------
# Per-trial reductions
# ---------------------------------------------------------------------------


def _reduce_heads(t: torch.Tensor, mode: str) -> torch.Tensor:
    """Heads dimension is the last axis: [E, heads] → [E]."""
    if mode == "mean":
        return t.mean(dim=-1)
    if mode == "max":
        return t.max(dim=-1).values
    raise ValueError(f"head_reduce must be 'mean' or 'max', got {mode!r}")


def _reduce_layers(per_layer_E: List[torch.Tensor], mode: str) -> torch.Tensor:
    """per_layer_E[l] is [E]; reduce across layers → [E]."""
    if mode == "mean":
        return torch.stack(per_layer_E, dim=0).mean(dim=0)
    if mode == "last":
        return per_layer_E[-1]
    raise ValueError(f"layer_reduce must be 'mean' or 'last', got {mode!r}")


def _per_trial_reductions(
    temporal_attention: torch.Tensor,                # [K]
    spatial_attention: List[List[torch.Tensor]],     # spatial_attention[k][l]: [E, heads]
    edge_index: torch.Tensor,                        # [2, E]
    head_reduce: str,
    layer_reduce: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (channel_importance[23], pair_matrix[23,23] symmetric)."""
    K = temporal_attention.shape[0]
    pair_E: torch.Tensor | None = None
    for k in range(K):
        per_layer_E = [_reduce_heads(layer, head_reduce) for layer in spatial_attention[k]]
        per_window_E = _reduce_layers(per_layer_E, layer_reduce)
        weighted = float(temporal_attention[k].item()) * per_window_E
        pair_E = weighted if pair_E is None else pair_E + weighted
    assert pair_E is not None

    pair = np.zeros((N_CH, N_CH), dtype=np.float32)
    ei = edge_index.cpu().numpy()
    em = pair_E.detach().cpu().numpy()
    for e in range(em.shape[0]):
        pair[int(ei[0, e]), int(ei[1, e])] += float(em[e])
    pair = (pair + pair.T) / 2.0

    channel_importance = pair.sum(axis=1).astype(np.float32)
    return channel_importance, pair


def _window_times(K: int, window_size: int, window_stride: int) -> np.ndarray:
    """[(t_start_s, t_end_s) for each window]."""
    times = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        times[k, 0] = (k * window_stride) / _FNIRS_FS
        times[k, 1] = (k * window_stride + window_size) / _FNIRS_FS
    return times


# ---------------------------------------------------------------------------
# Per-checkpoint pass
# ---------------------------------------------------------------------------


def explain_checkpoint(loaded: LoadedCheckpoint, cfg: XAIRunConfig) -> List[TrialAttribution]:
    """Run native-attention explanation on every val graph of `loaded`.

    Deterministic — eval-mode forward + attention extraction is non-stochastic
    (dropout is identity in eval, GATv2 attention is a fixed function of
    the data + weights). cfg.seed is unused here.
    """
    if cfg.arch != "st":
        raise ValueError(f"explain_checkpoint(ST) requires cfg.arch='st', got {cfg.arch!r}")

    device = torch.device(cfg.device)
    model = loaded.model.to(device).eval()

    trial_atts: List[TrialAttribution] = []
    trial_idx_per_subject: dict[str, int] = {}

    for idx in loaded.val_indices:
        raw_data: Data = loaded.dataset[idx]
        sid = str(raw_data.subject_id)
        t_idx = trial_idx_per_subject.get(sid, 0)
        trial_idx_per_subject[sid] = t_idx + 1

        data = loaded.val_transform(raw_data).to(device)
        true_label = int(data.y.item() if data.y.dim() == 0 else data.y[0].item())

        # Prediction (separate forward — model.explain() returns attention only).
        batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.edge_attr, batch_vec)
            pred_label = int(logits.argmax(dim=-1).item())

        # Native attention extraction.
        out = model.explain(data, device=device)
        temporal_attention: torch.Tensor = out["temporal_attention"]
        spatial_attention: List[List[torch.Tensor]] = out["spatial_attention"]
        K: int = int(out["n_windows"])
        window_size: int = int(out["window_size"])
        window_stride: int = int(out["window_stride"])

        ch_imp, pair = _per_trial_reductions(
            temporal_attention, spatial_attention, data.edge_index,
            cfg.head_reduce, cfg.layer_reduce,
        )
        wt = _window_times(K, window_size, window_stride)

        trial_atts.append(TrialAttribution(
            subject_id=sid,
            fold_or_subj_label=loaded.info.label,
            trial_idx_in_subject=t_idx,
            true_label=true_label,
            pred_label=pred_label,
            included=(pred_label == true_label) if not cfg.include_misclassified else True,
            channel_importance=ch_imp,
            pair_matrix=pair,
            feature_importance=None,
            temporal_attention=temporal_attention.detach().cpu().numpy().astype(np.float32),
            window_times=wt,
        ))

    return trial_atts


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Supplementary path — GNNExplainer with object masks (SPEC §6.4)
# ---------------------------------------------------------------------------


def _build_st_object_explainer(model: torch.nn.Module, cfg: XAIRunConfig) -> Explainer:
    """SPEC §6.4 — `node_mask_type='object'` keeps the mask at [23] rather
    than the [23, T≈326] explosion that 'attributes' would produce on ST.
    """
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=cfg.gnn_explainer_epochs, lr=cfg.gnn_explainer_lr),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="probs",
        ),
    )


def _supplementary_per_trial_reductions(
    node_mask: torch.Tensor,             # [N] or [N, 1] (object mask; PyG returns [N, 1])
    edge_mask: torch.Tensor,             # [E]
    edge_index: torch.Tensor,             # [2, E]
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (channel_importance[23], pair_matrix[23,23] symmetric)."""
    nm = node_mask.detach().abs()
    # PyG 2.7's GNNExplainer reports `node_mask_type='object'` as [N, 1].
    if nm.ndim == 2 and nm.shape[1] == 1:
        nm = nm.squeeze(-1)
    channel_importance = nm.cpu().numpy().astype(np.float32)
    if channel_importance.ndim != 1 or channel_importance.shape[0] != N_CH:
        raise ValueError(
            f"object node_mask must reduce to shape ({N_CH},), got {channel_importance.shape}"
        )

    pair = np.zeros((N_CH, N_CH), dtype=np.float32)
    ei = edge_index.cpu().numpy()
    em = edge_mask.detach().cpu().numpy()
    for e in range(em.shape[0]):
        pair[int(ei[0, e]), int(ei[1, e])] += float(em[e])
    pair = (pair + pair.T) / 2.0
    return channel_importance, pair


def explain_supplementary_checkpoint(
    loaded: LoadedCheckpoint, cfg: XAIRunConfig,
) -> List[TrialAttribution]:
    """SPEC §6.4 supplementary cross-check: GNNExplainer with object masks.

    Stochastic — same per-trial seed reset as SG.
    """
    if cfg.arch != "st":
        raise ValueError(
            f"explain_supplementary_checkpoint(ST) requires cfg.arch='st', got {cfg.arch!r}"
        )

    device = torch.device(cfg.device)
    model = loaded.model.to(device).eval()
    prob_model = ProbWrapper(model).to(device).eval()
    explainer = _build_st_object_explainer(prob_model, cfg)

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

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # cuDNN's GRU backward path requires training mode. The model is in
        # eval mode here (so dropout etc. don't fire), so we disable cuDNN
        # for this call — falls back to the native autograd-friendly RNN
        # kernel. Slower per call but only matters at LOSO scale.
        with torch.backends.cudnn.flags(enabled=False):
            explanation = explainer(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=batch_vec,
            )
        ch_imp, pair = _supplementary_per_trial_reductions(
            explanation.node_mask, explanation.edge_mask, data.edge_index,
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
            feature_importance=None,
            temporal_attention=None,
            window_times=None,
        ))
    return trial_atts


def run_st_supplementary(cfg: XAIRunConfig) -> PopulationResult:
    """SPEC §6.4 supplementary path: GNNExplainer object masks across all checkpoints.

    Returns a PopulationResult separate from the native-attention `run_st(cfg)`
    output. The notebook driver should call both and compute Spearman ρ
    between their channel-importance rankings as a method-independent sanity
    check (SPEC §11 C5).
    """
    if cfg.arch != "st":
        raise ValueError(f"run_st_supplementary requires cfg.arch='st', got {cfg.arch!r}")

    infos: List[CheckpointInfo] = discover_checkpoints(
        cfg.experiment_root, arch="st", hb=cfg.hb, regime=cfg.regime, mt=cfg.mt,
        subdir_override=cfg.experiment_subdir,
    )
    if not infos:
        raise FileNotFoundError(
            f"no ST checkpoints under {cfg.experiment_root!r} matching "
            f"hb={cfg.hb} regime={cfg.regime} mt={cfg.mt}"
        )

    all_trials: List[TrialAttribution] = []
    path_rebases: List[dict] = []
    for info in infos:
        loaded = load_checkpoint(info, cfg)
        path_rebases.extend(loaded.path_rebases)
        all_trials.extend(explain_supplementary_checkpoint(loaded, cfg))

    extras: dict = {
        "estimator": "gnn_object_supplementary",
        "gnn_explainer_epochs": cfg.gnn_explainer_epochs,
        "gnn_explainer_lr": cfg.gnn_explainer_lr,
        "n_checkpoints": len(infos),
        "checkpoints": [i.label for i in infos],
        "path_rebases": path_rebases,
    }
    return aggregate_population(
        all_trials,
        arch="st", hb=cfg.hb, regime=cfg.regime, mt=cfg.mt,
        only_included=not cfg.include_misclassified,
        extras=extras,
    )


def run_st(cfg: XAIRunConfig) -> PopulationResult:
    """Discover → explain every checkpoint → aggregate to PopulationResult.

    Writes nothing on its own; the caller is expected to invoke
    `result.to_csv(cfg.output_dir)` to persist the §7.3 deliverable
    (incl. temporal_attention.csv for ST).
    """
    if cfg.arch != "st":
        raise ValueError(f"run_st requires cfg.arch='st', got {cfg.arch!r}")

    infos: List[CheckpointInfo] = discover_checkpoints(
        cfg.experiment_root, arch="st", hb=cfg.hb, regime=cfg.regime, mt=cfg.mt,
        subdir_override=cfg.experiment_subdir,
    )
    if not infos:
        raise FileNotFoundError(
            f"no ST checkpoints under {cfg.experiment_root!r} matching "
            f"hb={cfg.hb} regime={cfg.regime} mt={cfg.mt}"
        )

    all_trials: List[TrialAttribution] = []
    path_rebases: List[dict] = []
    for info in infos:
        loaded = load_checkpoint(info, cfg)
        path_rebases.extend(loaded.path_rebases)
        all_trials.extend(explain_checkpoint(loaded, cfg))

    extras: dict = {
        "estimator": "native_attention",
        "head_reduce": cfg.head_reduce,
        "layer_reduce": cfg.layer_reduce,
        "fnirs_fs": _FNIRS_FS,
        "n_checkpoints": len(infos),
        "checkpoints": [i.label for i in infos],
        "path_rebases": path_rebases,
    }
    return aggregate_population(
        all_trials,
        arch="st", hb=cfg.hb, regime=cfg.regime, mt=cfg.mt,
        only_included=not cfg.include_misclassified,
        extras=extras,
    )
