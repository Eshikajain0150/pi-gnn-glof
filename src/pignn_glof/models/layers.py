from __future__ import annotations

import torch
from torch import nn


class PhysicsMessagePassing(nn.Module):
    """Directed edge-aware aggregation with a gated node update."""

    def __init__(self, hidden_dim: int, edge_features: int = 4, dropout: float = 0.3):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + edge_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        weight = edge_attr[:, 0:1]
        message = self.message(torch.cat([hidden[src], edge_attr], dim=-1)) * weight
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, dst, message)
        incoming = torch.zeros(hidden.shape[0], 1, device=hidden.device, dtype=hidden.dtype)
        incoming.index_add_(0, dst, weight)
        aggregate = aggregate / incoming.clamp_min(1e-6)
        updated = self.update(aggregate, hidden)
        return self.norm(hidden + self.dropout(updated))
