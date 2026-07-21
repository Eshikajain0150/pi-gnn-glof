from __future__ import annotations

import torch
from torch import nn

from pignn_glof.data.schema import GraphBatch

from .encoders import OpticalEncoder, SequenceEncoder
from .pignn import AutoregressiveTrajectoryHead


class CNNOnly(nn.Module):
    def __init__(
        self,
        *,
        optical_channels: int = 6,
        modality_embedding: int = 64,
        hidden_dim: int = 128,
        forecast_horizon: int = 6,
        dropout: float = 0.3,
        **_: object,
    ):
        super().__init__()
        self.encoder = OpticalEncoder(optical_channels, modality_embedding, dropout)
        self.project = nn.Sequential(
            nn.Linear(modality_embedding, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.trajectory = AutoregressiveTrajectoryHead(hidden_dim, forecast_horizon)

    def forward(self, batch: GraphBatch, **_: object) -> dict[str, torch.Tensor]:
        hidden = self.project(self.encoder(batch.optical))
        logits = self.classifier(hidden).squeeze(-1)
        trajectory_logits = self.trajectory(hidden, logits)
        return {
            "hazard_logits": logits,
            "hazard_probability": torch.sigmoid(logits),
            "trajectory_logits": trajectory_logits,
            "trajectory_probability": torch.sigmoid(trajectory_logits),
            "hidden": hidden,
        }


class TemporalGRU(nn.Module):
    def __init__(
        self,
        *,
        velocity_features: int = 3,
        thermal_features: int = 1,
        modality_embedding: int = 64,
        hidden_dim: int = 128,
        forecast_horizon: int = 6,
        dropout: float = 0.3,
        **_: object,
    ):
        super().__init__()
        self.velocity = SequenceEncoder(velocity_features, modality_embedding, dropout)
        self.thermal = SequenceEncoder(thermal_features, modality_embedding, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(modality_embedding * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.trajectory = AutoregressiveTrajectoryHead(hidden_dim, forecast_horizon)

    def forward(self, batch: GraphBatch, **_: object) -> dict[str, torch.Tensor]:
        hidden = self.fusion(torch.cat([self.velocity(batch.velocity), self.thermal(batch.thermal)], -1))
        logits = self.classifier(hidden).squeeze(-1)
        trajectory_logits = self.trajectory(hidden, logits)
        return {
            "hazard_logits": logits,
            "hazard_probability": torch.sigmoid(logits),
            "trajectory_logits": trajectory_logits,
            "trajectory_probability": torch.sigmoid(trajectory_logits),
            "hidden": hidden,
        }
