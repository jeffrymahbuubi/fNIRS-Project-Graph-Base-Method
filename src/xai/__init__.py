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

__all__ = [
    "CHANNEL_NAMES",
    "GRID_POS",
    "GRID_SHAPE",
    "N_CH",
    "CH_TO_IDX",
    "IDX_TO_GRID",
    "XAIRunConfig",
]
