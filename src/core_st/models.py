from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GATv2Conv, global_mean_pool


class WindowedSpatioTemporalGATNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 23,
        in_channels: int = 6,
        edge_dim: int = 2,
        window_size: int = 32,
        window_stride: int = 16,
        n_layers: int = 2,
        n_filters: int = 64,
        heads: int = 4,
        temporal_hidden: int = 64,
        temporal_layers: int = 1,
        fc_size: int = 64,
        dropout: float = 0.5,
        n_classes: int = 2,
        use_residual: bool = True,
        use_norm: bool = False,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.n_channels = n_channels

        # Spatial: GATv2 layers — weights shared across all K windows
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        in_dim = in_channels
        for _ in range(n_layers):
            out_dim = n_filters * heads
            conv = GATv2Conv(
                in_channels=in_dim,
                out_channels=n_filters,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=True,
            )
            self.convs.append(conv)

            if use_norm:
                norm: nn.Module = (
                    BatchNorm(out_dim) if norm_type == "batch" else nn.LayerNorm(out_dim)
                )
            else:
                norm = nn.Identity()
            self.norms.append(norm)

            proj: nn.Module = (
                nn.Linear(in_dim, out_dim, bias=False)
                if (use_residual and in_dim != out_dim)
                else nn.Identity()
            )
            self.residual_projs.append(proj)
            in_dim = out_dim

        spatial_out = in_dim  # n_filters * heads

        # Project pooled spatial embedding into temporal_hidden before GRU
        self.pre_gru = nn.Linear(spatial_out, temporal_hidden)

        # Temporal: GRU over K window embeddings
        self.gru = nn.GRU(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )

        # Additive attention over GRU outputs
        self.attn_v = nn.Linear(temporal_hidden, temporal_hidden)
        self.attn_u = nn.Linear(temporal_hidden, 1, bias=False)

        # Classifier head
        self.pre_cls = nn.Linear(temporal_hidden, fc_size)
        self.classifier = nn.Linear(fc_size, n_classes)

        # Populated during eval-mode forward for explain()
        self._last_temporal_attn: Optional[torch.Tensor] = None
        self._last_spatial_attn: Optional[List[List[torch.Tensor]]] = None
        self._last_n_windows: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_window_stats(windows: torch.Tensor) -> torch.Tensor:
        """
        windows : [N, K, W]
        returns  : [N, K, 6]  (mean, min, max, variance, skewness, kurtosis)
        """
        eps = 1e-8
        mean = windows.mean(dim=-1)
        min_ = windows.min(dim=-1).values
        max_ = windows.max(dim=-1).values
        centered = windows - mean.unsqueeze(-1)
        var = (centered ** 2).mean(dim=-1)
        std = var.sqrt().clamp(min=eps)
        skew = (centered ** 3).mean(dim=-1) / (std ** 3)
        kurt = (centered ** 4).mean(dim=-1) / (std ** 4)
        stats = torch.stack([mean, min_, max_, var, skew, kurt], dim=-1)
        return torch.nan_to_num(stats, nan=0.0)

    def _spatial_encode(
        self,
        x_k: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attn: bool = False,
    ):
        """GATv2 layers on one window's node features [N, in_dim] → [N, spatial_out]."""
        attn_list: List[torch.Tensor] = []
        for conv, norm, res_proj in zip(self.convs, self.norms, self.residual_projs):
            if return_attn:
                h, (_, attn_w) = conv(x_k, edge_index, edge_attr, return_attention_weights=True)
                attn_list.append(attn_w.detach().cpu())
            else:
                h = conv(x_k, edge_index, edge_attr)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x_k = h + res_proj(x_k) if self.use_residual else h
        return x_k, attn_list

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B*C, T]
        collect_attn = not self.training

        # Step 1: window the raw time series
        windows = x.unfold(dimension=1, size=self.window_size, step=self.window_stride)
        # windows: [B*C, K, W]

        # Step 2: per-window 6-stat node features → [B*C, K, 6]
        window_stats = self._compute_window_stats(windows)
        K = window_stats.shape[1]

        # Step 3: spatial encoding per window (shared GATv2 weights)
        window_embeddings: List[torch.Tensor] = []
        all_spatial_attn: List[List[torch.Tensor]] = []

        for k in range(K):
            x_k = window_stats[:, k, :]  # [B*C, 6]
            x_k, attn_k = self._spatial_encode(x_k, edge_index, edge_attr, return_attn=collect_attn)
            # Pool across nodes → [B, spatial_out]
            h_k = global_mean_pool(x_k, batch)
            # Project to temporal hidden → [B, temporal_hidden]
            h_k = F.elu(self.pre_gru(h_k))
            window_embeddings.append(h_k)
            all_spatial_attn.append(attn_k)

        # Step 4: temporal encoding
        seq = torch.stack(window_embeddings, dim=1)   # [B, K, temporal_hidden]
        gru_out, _ = self.gru(seq)                    # [B, K, temporal_hidden]

        e = torch.tanh(self.attn_v(gru_out))           # [B, K, temporal_hidden]
        alpha = torch.softmax(self.attn_u(e), dim=1)   # [B, K, 1]
        context = (alpha * gru_out).sum(dim=1)          # [B, temporal_hidden]

        # Step 5: classify
        context = F.dropout(context, p=self.dropout, training=self.training)
        logits = self.classifier(F.elu(self.pre_cls(context)))

        if collect_attn:
            self._last_temporal_attn = alpha.detach().cpu()
            self._last_spatial_attn = all_spatial_attn
            self._last_n_windows = K

        return logits

    # ------------------------------------------------------------------
    # XAI
    # ------------------------------------------------------------------

    def explain(self, data: Data, device: torch.device) -> dict:
        """
        Run a single-sample forward pass and return attention weights for XAI.
        Call after training; do not call during the training loop.

        Parameters
        ----------
        data   : a single un-batched PyG Data object (one trial)
        device : the device the model is on

        Returns
        -------
        dict with keys:
            temporal_attention : Tensor [K]
                softmax weight of each time window — higher = more influential
            spatial_attention  : List[List[Tensor]]
                spatial_attention[k][l] = GATv2 edge attention for window k, layer l
                shape of each tensor: [E, heads]
            window_size  : int
            window_stride: int
            n_windows    : int (K)
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            data = data.to(device)
            batch_vec = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
            _ = self.forward(data.x, data.edge_index, data.edge_attr, batch_vec)
        if was_training:
            self.train()

        temporal_attn = self._last_temporal_attn
        if temporal_attn is not None:
            temporal_attn = temporal_attn.squeeze(0).squeeze(-1)  # [1, K, 1] → [K]

        return {
            "temporal_attention": temporal_attn,
            "spatial_attention": self._last_spatial_attn,
            "window_size": self.window_size,
            "window_stride": self.window_stride,
            "n_windows": self._last_n_windows,
        }
