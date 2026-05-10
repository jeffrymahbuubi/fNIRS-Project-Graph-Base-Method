"""XAIRunConfig — one dataclass that drives one XAI run.

Per SPEC §9 (rev. 4): notebook cells construct an XAIRunConfig and pass it to
`run_sg(cfg)` / `run_st(cfg)` (Phase A only ships discovery + load; explainers
land in Phase A.4 / A.5).

Design notes:
- `data_dir` and `splits_json` are NOT required fields. Each checkpoint's own
  `config.yaml` is the source of truth (SPEC §10.4). The two `*_override`
  fields exist for the rare case where the auto-rebase logic in
  `src/xai/checkpoints.py:_resolve_data_dir` cannot recover the right path.
- `arch` / `regime` / `mt` / `hb` are validated at construction time so a bad
  notebook cell fails fast, not 30 minutes into a LOSO run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


_VALID_ARCH: Tuple[str, ...] = ("sg", "st")
_VALID_HB: Tuple[str, ...] = ("hbo", "hbr", "hbt")
_VALID_REGIME: Tuple[str, ...] = ("kfold-5", "kfold-10", "loso")
_VALID_MT: Tuple[int, ...] = (2, 4)
_VALID_HEAD_REDUCE: Tuple[str, ...] = ("mean", "max")
_VALID_LAYER_REDUCE: Tuple[str, ...] = ("mean", "last")
# SG estimators (SPEC §5.3 / §15.4 / §15.7). C4 (SPEC §11) compares ρ between
# any two of these three rankings at the population level.
_VALID_SG_ESTIMATOR: Tuple[str, ...] = ("gnn", "captum_ig", "attention")


@dataclass
class XAIRunConfig:
    """One XAI run = one (arch, hb, regime, mt) cell of the SPEC matrix.

    Required fields
    ---------------
    arch              : 'sg' | 'st'
    hb                : 'hbo' | 'hbr' | 'hbt'  (rev. 4 scope is HbO only;
                        HbR / HbT are accepted for forward-compat with the
                        future-work notebooks but not exercised in v1)
    regime            : 'kfold-5' | 'kfold-10' | 'loso'
    mt                : 2 | 4
    experiment_root   : path to the directory containing the per-regime subdirs
                        (`kfold/5-fold/`, `kfold/10-fold/`, `loso/` for SG;
                        `5-fold/`, `10-fold/`, `loso/` for ST)
    output_dir        : where this run's CSVs / NPY / run.json land

    Common fields
    -------------
    seed              : torch + numpy seed for GNNExplainer's stochastic mask
                        initialisation (SPEC §10.5)
    device            : torch device string ('cuda:0', 'cpu', ...)
    include_misclassified : if False (default), aggregation in §7 keeps only
                        trials where argmax(softmax(logits)) == y
    project_root      : absolute path to the project root, used by the data-dir
                        rebase logic in `checkpoints.py`. Auto-detected if None.

    SG-only fields
    --------------
    gnn_explainer_epochs       : default 200 (SPEC §5.2 step 5)
    gnn_explainer_lr           : default 0.01
    estimator                  : 'gnn' | 'captum_ig' | 'attention'
                                 — selects the SG estimator to run.
                                 'gnn'        → GNNExplainer (primary, SPEC §5.2)
                                 'captum_ig'  → CaptumExplainer-IG (cross-check,
                                                SPEC §5.3 / §15.7); requires
                                                `captum` installed
                                 'attention'  → AttentionExplainer (cross-check,
                                                SPEC §5.3 / §15.4); essentially
                                                free at inference, edge-mask only
                                 SPEC §11 C4 compares ρ between any two of these
                                 three rankings at the population level.
    run_attention_cross_check  : Legacy SPEC rev. 3 toggle. Use `estimator`
                                 field above instead — kept for backward
                                 compatibility, no longer read by the explainer.

    ST-only fields
    --------------
    head_reduce       : 'mean' | 'max'  — how to aggregate the per-edge
                        attention across heads (SPEC §6.2)
    layer_reduce      : 'mean' | 'last' — how to aggregate across layers
                        (SPEC §6.3)
    run_supplementary_gnnexplainer : whether to also run the GNNExplainer
                        cross-check pass with `node_mask_type='object'`
                        (SPEC §6.4). Default False — too slow at LOSO scale.

    Path overrides (rare)
    ---------------------
    data_dir_override   : if non-None, used in place of the data_dir from each
                          checkpoint's config.yaml (after rebase). Use only
                          when the rebase logic cannot recover the local path.
    splits_json_override: same idea for the splits JSON. Note: for LOSO, the
                          checkpoint's config.yaml has `splits_json: null`,
                          and overrides are still ignored — LOSO splits are
                          always derived in-code from dataset subject IDs.
    experiment_subdir   : if non-None, used as the relative path under
                          experiment_root in place of the per-arch/per-regime
                          default in `_REGIME_SUBDIR`. Needed for the
                          20260509 ST layout where kfold lives at
                          `<root>/st-kfold/{5,10}-fold/<date>/...` instead of
                          the original `<root>/{5,10}-fold/...`. LOSO under
                          20260509 still matches the default, so leave None
                          for LOSO cells.
    """

    # --- required fields (no defaults) -------------------------------------
    arch: str
    hb: str
    regime: str
    mt: int
    experiment_root: str
    output_dir: str

    # --- common defaults ---------------------------------------------------
    seed: int = 42
    device: str = "cuda:0"
    include_misclassified: bool = False
    project_root: Optional[str] = None

    # --- SG-only -----------------------------------------------------------
    gnn_explainer_epochs: int = 200
    gnn_explainer_lr: float = 0.01
    estimator: str = "gnn"
    run_attention_cross_check: bool = True   # legacy; superseded by `estimator`

    # --- ST-only -----------------------------------------------------------
    head_reduce: str = "mean"
    layer_reduce: str = "mean"
    run_supplementary_gnnexplainer: bool = False

    # --- path overrides ----------------------------------------------------
    data_dir_override: Optional[str] = None
    splits_json_override: Optional[str] = None
    experiment_subdir: Optional[str] = None

    # --- internal book-keeping --------------------------------------------
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arch not in _VALID_ARCH:
            raise ValueError(f"arch must be one of {_VALID_ARCH}, got {self.arch!r}")
        if self.hb not in _VALID_HB:
            raise ValueError(f"hb must be one of {_VALID_HB}, got {self.hb!r}")
        if self.regime not in _VALID_REGIME:
            raise ValueError(f"regime must be one of {_VALID_REGIME}, got {self.regime!r}")
        if self.mt not in _VALID_MT:
            raise ValueError(f"mt must be one of {_VALID_MT}, got {self.mt!r}")
        if self.head_reduce not in _VALID_HEAD_REDUCE:
            raise ValueError(
                f"head_reduce must be one of {_VALID_HEAD_REDUCE}, got {self.head_reduce!r}"
            )
        if self.layer_reduce not in _VALID_LAYER_REDUCE:
            raise ValueError(
                f"layer_reduce must be one of {_VALID_LAYER_REDUCE}, got {self.layer_reduce!r}"
            )
        if self.estimator not in _VALID_SG_ESTIMATOR:
            raise ValueError(
                f"estimator must be one of {_VALID_SG_ESTIMATOR}, got {self.estimator!r}"
            )
        if self.gnn_explainer_epochs <= 0:
            raise ValueError(f"gnn_explainer_epochs must be > 0, got {self.gnn_explainer_epochs}")
        if self.gnn_explainer_lr <= 0:
            raise ValueError(f"gnn_explainer_lr must be > 0, got {self.gnn_explainer_lr}")
