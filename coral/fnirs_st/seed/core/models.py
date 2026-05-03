"""
Temporal module factory for fNIRS ST-GNN CORAL search.

Five interchangeable temporal modules — all receive [B, K, temporal_hidden]
and return a context vector [B, temporal_hidden]:

  GRUTemporal         — GRU + additive attention (baseline)
  LSTMTemporal        — LSTM + additive attention
  BiGRUTemporal       — Bidirectional GRU + additive attention
  TransformerTemporal — Transformer encoder + mean pooling
  TCNTemporal         — Temporal Convolutional Network + last-step readout

WindowedSpatioTemporalGATNet wires GATv2 spatial encoder → temporal module.
The spatial encoder is fixed; only the temporal module is searchable.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv, global_mean_pool


# ---------------------------------------------------------------------------
# Temporal modules
# ---------------------------------------------------------------------------

class GRUTemporal(nn.Module):
    """GRU with additive attention readout (Optuna-tuned baseline)."""

    def __init__(self, temporal_hidden: int, temporal_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        self.attn_v = nn.Linear(temporal_hidden, temporal_hidden)
        self.attn_u = nn.Linear(temporal_hidden, 1, bias=False)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq)                      # [B, K, H]
        alpha = torch.softmax(self.attn_u(torch.tanh(self.attn_v(out))), dim=1)
        return (alpha * out).sum(dim=1)             # [B, H]


class LSTMTemporal(nn.Module):
    """LSTM with additive attention readout."""

    def __init__(self, temporal_hidden: int, temporal_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        self.attn_v = nn.Linear(temporal_hidden, temporal_hidden)
        self.attn_u = nn.Linear(temporal_hidden, 1, bias=False)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)                     # [B, K, H]
        alpha = torch.softmax(self.attn_u(torch.tanh(self.attn_v(out))), dim=1)
        return (alpha * out).sum(dim=1)             # [B, H]


class BiGRUTemporal(nn.Module):
    """Bidirectional GRU with additive attention.

    Uses hidden_size = temporal_hidden // 2 so the forward+backward
    concatenation stays at temporal_hidden dimensions.
    """

    def __init__(self, temporal_hidden: int, temporal_layers: int, dropout: float):
        super().__init__()
        h = max(1, temporal_hidden // 2)
        self.bigru = nn.GRU(
            input_size=temporal_hidden,
            hidden_size=h,
            num_layers=temporal_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        out_dim = h * 2
        self.attn_v = nn.Linear(out_dim, out_dim)
        self.attn_u = nn.Linear(out_dim, 1, bias=False)
        # Project back to temporal_hidden if 2h != temporal_hidden (odd inputs)
        self.proj = nn.Linear(out_dim, temporal_hidden) if out_dim != temporal_hidden else nn.Identity()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.bigru(seq)                    # [B, K, 2h]
        alpha = torch.softmax(self.attn_u(torch.tanh(self.attn_v(out))), dim=1)
        context = (alpha * out).sum(dim=1)          # [B, 2h]
        return self.proj(context)                   # [B, temporal_hidden]


class TransformerTemporal(nn.Module):
    """Transformer encoder with sinusoidal positional encoding + mean pooling.

    temporal_hidden must be divisible by n_heads. All values in the CORAL
    search space [32..256, step=32] are divisible by 8, so any {2,4,8} is safe.
    """

    def __init__(
        self,
        temporal_hidden: int,
        temporal_layers: int,
        n_heads: int,
        ffn_ratio: int,
        dropout: float,
    ):
        super().__init__()
        # Safety: round n_heads down until temporal_hidden is divisible
        while temporal_hidden % n_heads != 0 and n_heads > 1:
            n_heads //= 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_hidden,
            nhead=n_heads,
            dim_feedforward=temporal_hidden * ffn_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # pre-norm: more stable for small datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.register_buffer("pe", self._sinusoidal_pe(512, temporal_hidden))

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        K = seq.size(1)
        seq = seq + self.pe[:, :K, :]              # add positional encoding
        out = self.transformer(seq)                 # [B, K, H]
        return out.mean(dim=1)                      # mean pool → [B, H]


class _TCNBlock(nn.Module):
    """Single causal dilated conv block with residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.causal_pad = nn.ConstantPad1d((pad, 0), 0.0)
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(F.relu(self.bn(self.conv(self.causal_pad(x))))) + x


class TCNTemporal(nn.Module):
    """Temporal Convolutional Network with causal dilated convolutions.

    Input/output: [B, K, H] (seq-first).
    Internally transposed to [B, H, K] (channels-first) for Conv1d.
    Readout: last time step of the final conv output.
    """

    def __init__(
        self,
        temporal_hidden: int,
        temporal_layers: int,
        kernel_size: int,
        dilation_base: int,
        dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            _TCNBlock(
                channels=temporal_hidden,
                kernel_size=kernel_size,
                dilation=dilation_base ** i,
                dropout=dropout,
            )
            for i in range(temporal_layers)
        ])

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = seq.transpose(1, 2)                    # [B, H, K]
        for block in self.blocks:
            x = block(x)
        return x[:, :, -1]                         # last time step → [B, H]


# ---------------------------------------------------------------------------
# Temporal module factory
# ---------------------------------------------------------------------------

def build_temporal_module(
    temporal_type: str,
    temporal_hidden: int,
    temporal_layers: int,
    dropout: float,
    transformer_heads: int = 4,
    ffn_ratio: int = 2,
    tcn_kernel_size: int = 5,
    tcn_dilation_base: int = 2,
) -> nn.Module:
    if temporal_type == "gru":
        return GRUTemporal(temporal_hidden, temporal_layers, dropout)
    if temporal_type == "lstm":
        return LSTMTemporal(temporal_hidden, temporal_layers, dropout)
    if temporal_type == "bigru":
        return BiGRUTemporal(temporal_hidden, temporal_layers, dropout)
    if temporal_type == "transformer":
        return TransformerTemporal(temporal_hidden, temporal_layers, transformer_heads, ffn_ratio, dropout)
    if temporal_type == "tcn":
        return TCNTemporal(temporal_hidden, temporal_layers, tcn_kernel_size, tcn_dilation_base, dropout)
    raise ValueError(f"Unknown temporal_type '{temporal_type}'. "
                     f"Choose from: gru, lstm, bigru, transformer, tcn")


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class WindowedSpatioTemporalGATNet(nn.Module):
    """
    GATv2 spatial encoder (shared across K windows) + pluggable temporal module.

    FIXED spatial encoder params:
      n_layers=2, n_filters=80, heads=2, use_residual=False, use_norm=True

    Searchable temporal params (set via temporal_type + related args):
      temporal_type    — module architecture
      temporal_hidden  — hidden dimensionality
      temporal_layers  — depth
      + module-specific kwargs (transformer_heads, ffn_ratio, tcn_kernel_size, tcn_dilation_base)
    """

    def __init__(
        self,
        n_channels: int = 23,
        in_channels: int = 6,
        edge_dim: int = 2,
        window_size: int = 16,
        window_stride: int = 8,
        # --- FIXED spatial encoder ---
        n_layers: int = 2,
        n_filters: int = 80,
        heads: int = 2,
        use_residual: bool = False,
        use_norm: bool = True,
        norm_type: str = "batch",
        # --- Searchable temporal ---
        temporal_type: str = "gru",
        temporal_hidden: int = 192,
        temporal_layers: int = 1,
        transformer_heads: int = 4,
        ffn_ratio: int = 2,
        tcn_kernel_size: int = 5,
        tcn_dilation_base: int = 2,
        # --- Classifier head ---
        fc_size: int = 256,
        dropout: float = 0.3,
        n_classes: int = 2,
    ):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.n_channels = n_channels

        # --- Spatial: GATv2 layers (weights shared across all K windows) ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        in_dim = in_channels
        for _ in range(n_layers):
            out_dim = n_filters * heads
            self.convs.append(GATv2Conv(
                in_channels=in_dim, out_channels=n_filters, heads=heads,
                dropout=dropout, edge_dim=edge_dim, concat=True,
            ))
            self.norms.append(
                BatchNorm(out_dim) if (use_norm and norm_type == "batch")
                else (nn.LayerNorm(out_dim) if use_norm else nn.Identity())
            )
            self.residual_projs.append(
                nn.Linear(in_dim, out_dim, bias=False)
                if (use_residual and in_dim != out_dim) else nn.Identity()
            )
            in_dim = out_dim

        spatial_out = in_dim  # n_filters * heads

        # Project pooled spatial embedding to temporal_hidden before temporal module
        self.pre_temporal = nn.Linear(spatial_out, temporal_hidden)

        # --- Temporal module (pluggable) ---
        self.temporal = build_temporal_module(
            temporal_type=temporal_type,
            temporal_hidden=temporal_hidden,
            temporal_layers=temporal_layers,
            dropout=dropout,
            transformer_heads=transformer_heads,
            ffn_ratio=ffn_ratio,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dilation_base=tcn_dilation_base,
        )

        # --- Classifier head ---
        self.pre_cls = nn.Linear(temporal_hidden, fc_size)
        self.classifier = nn.Linear(fc_size, n_classes)

    @staticmethod
    def _compute_window_stats(windows: torch.Tensor) -> torch.Tensor:
        """windows [N, K, W] → stats [N, K, 6]  (mean, min, max, var, skew, kurt)."""
        eps = 1e-8
        mean = windows.mean(dim=-1)
        min_ = windows.min(dim=-1).values
        max_ = windows.max(dim=-1).values
        centered = windows - mean.unsqueeze(-1)
        var = (centered ** 2).mean(dim=-1)
        std = var.sqrt().clamp(min=eps)
        skew = (centered ** 3).mean(dim=-1) / std ** 3
        kurt = (centered ** 4).mean(dim=-1) / std ** 4
        return torch.nan_to_num(
            torch.stack([mean, min_, max_, var, skew, kurt], dim=-1), nan=0.0
        )

    def _spatial_encode(self, x_k, edge_index, edge_attr):
        for conv, norm, res_proj in zip(self.convs, self.norms, self.residual_projs):
            h = norm(conv(x_k, edge_index, edge_attr))
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x_k = h + res_proj(x_k) if self.use_residual else h
        return x_k

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B*C, T]
        windows = x.unfold(dimension=1, size=self.window_size, step=self.window_stride)
        window_stats = self._compute_window_stats(windows)  # [B*C, K, 6]
        K = window_stats.shape[1]

        window_embeddings: List[torch.Tensor] = []
        for k in range(K):
            x_k = window_stats[:, k, :]                         # [B*C, 6]
            x_k = self._spatial_encode(x_k, edge_index, edge_attr)
            h_k = global_mean_pool(x_k, batch)                  # [B, spatial_out]
            h_k = F.elu(self.pre_temporal(h_k))                 # [B, temporal_hidden]
            window_embeddings.append(h_k)

        seq = torch.stack(window_embeddings, dim=1)              # [B, K, temporal_hidden]
        context = self.temporal(seq)                             # [B, temporal_hidden]

        context = F.dropout(context, p=self.dropout, training=self.training)
        return self.classifier(F.elu(self.pre_cls(context)))     # [B, n_classes]
