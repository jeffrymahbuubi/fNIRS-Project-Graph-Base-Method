"""Graph Explainability building blocks for the fNIRS GNN project.

See `docs/SPEC_xai_graph.md` (rev. 4) for the full specification.

This package is consumed by notebooks under `src/notebook/xai/`.
The public API is gradually built up across Phase A — what is exported here
matches what has been implemented and tested.
"""

from src.xai.channels import (
    CHANNEL_NAMES,
    GRID_POS,
    GRID_SHAPE,
    N_CH,
    CH_TO_IDX,
    IDX_TO_GRID,
)
from src.xai.config import XAIRunConfig
from src.xai.checkpoints import (
    CheckpointInfo,
    LoadedCheckpoint,
    discover_checkpoints,
    load_checkpoint,
)
from src.xai.aggregate import (
    TrialAttribution,
    PopulationResult,
    aggregate_population,
)
from src.xai.sg_explainer import (
    ProbWrapper,
    explain_checkpoint as explain_sg_checkpoint,
    run_sg,
)
from src.xai.st_explainer import (
    explain_checkpoint as explain_st_checkpoint,
    run_st,
)
from src.xai.visualize import (
    plot_montage_channel_importance,
    plot_pair_matrix,
    plot_temporal_attention,
    plot_sg_vs_st_scatter,
    plot_pair_matrix_diff,
)
from src.xai.io import write_run_json, read_run_json

__all__ = [
    "CHANNEL_NAMES",
    "GRID_POS",
    "GRID_SHAPE",
    "N_CH",
    "CH_TO_IDX",
    "IDX_TO_GRID",
    "XAIRunConfig",
    "CheckpointInfo",
    "LoadedCheckpoint",
    "discover_checkpoints",
    "load_checkpoint",
    "TrialAttribution",
    "PopulationResult",
    "aggregate_population",
    "ProbWrapper",
    "explain_sg_checkpoint",
    "run_sg",
    "explain_st_checkpoint",
    "run_st",
    "plot_montage_channel_importance",
    "plot_pair_matrix",
    "plot_temporal_attention",
    "plot_sg_vs_st_scatter",
    "plot_pair_matrix_diff",
    "write_run_json",
    "read_run_json",
]
