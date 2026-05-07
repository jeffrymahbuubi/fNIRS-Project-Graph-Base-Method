"""Canonical 23-channel prefrontal montage layout.

Single source of truth for the XAI module. Values copied verbatim from
`src/notebook/statistical-analysis/04_severity_correlation/04_severity_correlation.ipynb`
so XAI heatmaps line up channel-for-channel with the statistical-analysis figures.
"""
from typing import Dict, List, Tuple

CHANNEL_NAMES: List[str] = [
    "S1_D1", "S1_D3", "S2_D2", "S2_D1", "S2_D5", "S3_D1", "S3_D3", "S3_D4", "S3_D6",
    "S4_D4", "S4_D5", "S4_D7", "S5_D2", "S5_D5", "S5_D8", "S6_D3", "S6_D6",
    "S7_D4", "S7_D6", "S7_D7", "S8_D5", "S8_D7", "S8_D8",
]

GRID_POS: List[Tuple[int, int]] = [
    (0, 2), (1, 1), (0, 4), (0, 3), (1, 4), (1, 2), (2, 1), (2, 2), (3, 1),
    (2, 3), (2, 4), (3, 4), (1, 5), (2, 5), (3, 6), (3, 0), (4, 1),
    (3, 2), (4, 2), (4, 3), (3, 5), (4, 4), (4, 5),
]

GRID_SHAPE: Tuple[int, int] = (5, 7)

N_CH: int = len(CHANNEL_NAMES)

CH_TO_IDX: Dict[str, int] = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
IDX_TO_GRID: Dict[int, Tuple[int, int]] = dict(enumerate(GRID_POS))

assert N_CH == 23, f"expected 23 channels, got {N_CH}"
assert len(GRID_POS) == N_CH, "GRID_POS length must equal CHANNEL_NAMES length"
