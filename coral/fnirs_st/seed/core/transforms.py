from typing import Dict

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.utils import dropout_edge, mask_feature


class StandardizeGraphFeatures(BaseTransform):
    def __init__(self, mean_ea: torch.Tensor, std_ea: torch.Tensor, eps: float = 1e-8):
        self.mean_ea = mean_ea
        self.std_ea = std_ea
        self.eps = eps

    def forward(self, data: Data) -> Data:
        data = data.clone()
        if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            data.edge_attr = (data.edge_attr - self.mean_ea) / (self.std_ea + self.eps)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DropoutEdgeAugmentation(BaseTransform):
    def __init__(self, p: float = 0.1, force_undirected: bool = True):
        self.p = p
        self.force_undirected = force_undirected

    def forward(self, data: Data) -> Data:
        data = data.clone()
        edge_index, mask = dropout_edge(
            data.edge_index, p=self.p, force_undirected=self.force_undirected, training=True
        )
        data.edge_index = edge_index
        if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            data.edge_attr = data.edge_attr[mask]
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class MaskFeatureAugmentation(BaseTransform):
    def __init__(self, p: float = 0.1, mode: str = "all"):
        self.p = p
        self.mode = mode

    def forward(self, data: Data) -> Data:
        data = data.clone()
        data.x, _ = mask_feature(data.x, p=self.p, mode=self.mode, fill_value=0.0)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"


def get_transforms(
    stats: Dict[str, torch.Tensor],
    augment: bool = False,
    edge_dropout_p: float = 0.1,
    feature_mask_p: float = 0.1,
    feature_mask_mode: str = "all",
    use_rwpe: bool = False,
    rwpe_walk_length: int = 4,
) -> Compose:
    pipeline = [StandardizeGraphFeatures(mean_ea=stats["mean_ea"], std_ea=stats["std_ea"])]
    if augment:
        if feature_mask_p > 0.0:
            pipeline.append(MaskFeatureAugmentation(p=feature_mask_p, mode=feature_mask_mode))
        if edge_dropout_p > 0.0:
            pipeline.append(DropoutEdgeAugmentation(p=edge_dropout_p))
    return Compose(pipeline)
