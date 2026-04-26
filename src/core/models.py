from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv, GINEConv, global_mean_pool


class FlexibleGATNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        edge_dim: int = 2,
        n_layers: int = 2,
        n_filters: Union[int, List[int]] = 64,
        heads: Union[int, List[int]] = 4,
        fc_size: int = 64,
        dropout: float = 0.5,
        n_classes: int = 2,
        use_residual: bool = True,
        use_norm: bool = False,
        norm_type: str = "batch",
        use_gine_first_layer: bool = False,
        gine_train_eps: bool = True,
    ):
        super().__init__()

        n_filters_list = [n_filters] * n_layers if isinstance(n_filters, int) else list(n_filters)
        heads_list = [heads] * n_layers if isinstance(heads, int) else list(heads)
        assert len(n_filters_list) == n_layers, "n_filters length must equal n_layers"
        assert len(heads_list) == n_layers, "heads length must equal n_layers"

        self.n_layers = n_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_norm = use_norm

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        in_dim = in_channels
        for i in range(n_layers):
            f, h = n_filters_list[i], heads_list[i]

            if i == 0 and use_gine_first_layer:
                mlp = nn.Sequential(
                    nn.Linear(in_dim, f),
                    nn.ReLU(),
                    nn.BatchNorm1d(f),
                    nn.Linear(f, f),
                )
                conv = GINEConv(mlp, train_eps=gine_train_eps, edge_dim=edge_dim)
                out_dim = f
            else:
                conv = GATv2Conv(
                    in_channels=in_dim,
                    out_channels=f,
                    heads=h,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=True,
                )
                out_dim = f * h

            self.convs.append(conv)

            if use_norm:
                norm: nn.Module = (
                    BatchNorm(out_dim) if norm_type == "batch" else nn.LayerNorm(out_dim)
                )
            else:
                norm = nn.Identity()
            self.norms.append(norm)

            proj: nn.Module = (
                nn.Linear(in_dim, out_dim, bias=False) if (use_residual and in_dim != out_dim)
                else nn.Identity()
            )
            self.residual_projs.append(proj)

            in_dim = out_dim

        self.pre_pool = nn.Linear(in_dim, fc_size)
        self.classifier = nn.Linear(fc_size, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        for conv, norm, res_proj in zip(self.convs, self.norms, self.residual_projs):
            h = conv(x, edge_index, edge_attr)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h + res_proj(x) if self.use_residual else h

        x = F.elu(self.pre_pool(x))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
