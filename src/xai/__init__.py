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
    explain_supplementary_checkpoint as explain_st_supplementary_checkpoint,
    run_st,
    run_st_supplementary,
)
from src.xai.visualize import (
    plot_montage_channel_importance,
    plot_pair_matrix,
    plot_temporal_attention,
    plot_sg_vs_st_scatter,
    plot_pair_matrix_diff,
    # rev. 5: atlas-level figures (SPEC §16.8)
    plot_brodmann_montage,
    plot_brodmann_surface,
    plot_montage_with_atlas,
    plot_region_bar,
    plot_region_pair_heatmap,
)
from src.xai.io import write_run_json, read_run_json
from src.xai.atlas import (
    DEFAULT_ATLAS_PARC,
    DEFAULT_PROJECTION_DISTANCE_BOUND_MM,
    DEFAULT_RADIUS_MM,
    DEFAULT_SIGMA_MM,
    AtlasRegistration,
    BrodmannMapping,
    ElcContents,
    aggregate_to_regions,
    apply_registration,
    build_channel_to_brodmann,
    compute_channel_midpoints,
    compute_sd_distances,
    parse_elc,
    project_to_brodmann,
    register_to_fsaverage,
)

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
    "explain_st_supplementary_checkpoint",
    "run_st",
    "run_st_supplementary",
    "plot_montage_channel_importance",
    "plot_pair_matrix",
    "plot_temporal_attention",
    "plot_sg_vs_st_scatter",
    "plot_pair_matrix_diff",
    "plot_brodmann_montage",
    "plot_brodmann_surface",
    "plot_montage_with_atlas",
    "plot_region_bar",
    "plot_region_pair_heatmap",
    "write_run_json",
    "read_run_json",
    # rev. 5: atlas registration (SPEC §16)
    "DEFAULT_ATLAS_PARC",
    "DEFAULT_PROJECTION_DISTANCE_BOUND_MM",
    "DEFAULT_RADIUS_MM",
    "DEFAULT_SIGMA_MM",
    "AtlasRegistration",
    "BrodmannMapping",
    "ElcContents",
    "aggregate_to_regions",
    "apply_registration",
    "build_channel_to_brodmann",
    "compute_channel_midpoints",
    "compute_sd_distances",
    "parse_elc",
    "project_to_brodmann",
    "register_to_fsaverage",
]
