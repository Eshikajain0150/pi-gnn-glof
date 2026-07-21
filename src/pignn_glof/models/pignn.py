from __future__ import annotations

import torch
from torch import nn

from pignn_glof.data.schema import GraphBatch

from .encoders import MultimodalEncoder
from .layers import PhysicsMessagePassing


class AutoregressiveTrajectoryHead(nn.Module):
    def __init__(self, hidden_dim: int, horizon: int):
        super().__init__()
        self.horizon = int(horizon)
        self.input_projection = nn.Linear(1, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, initial_logit: torch.Tensor) -> torch.Tensor:
        state = hidden
        previous = torch.sigmoid(initial_logit).unsqueeze(-1)
        outputs = []
        for _ in range(self.horizon):
            state = self.cell(self.input_projection(previous), state)
            logit = self.output(state).squeeze(-1)
            outputs.append(logit)
            previous = torch.sigmoid(logit).unsqueeze(-1)
        return torch.stack(outputs, dim=1)


class PIGNN(nn.Module):
    def __init__(
        self,
        *,
        optical_channels: int = 6,
        velocity_features: int = 3,
        thermal_features: int = 1,
        modality_embedding: int = 64,
        hidden_dim: int = 128,
        graph_layers: int = 3,
        edge_features: int = 4,
        forecast_horizon: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = MultimodalEncoder(
            optical_channels=optical_channels,
            velocity_features=velocity_features,
            thermal_features=thermal_features,
            embedding_dim=modality_embedding,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [PhysicsMessagePassing(hidden_dim, edge_features, dropout) for _ in range(graph_layers)]
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.trajectory = AutoregressiveTrajectoryHead(hidden_dim, forecast_horizon)

    def forward(
        self,
        batch: GraphBatch,
        modality_mask: torch.Tensor | None = None,
        precomputed_embeddings: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if precomputed_embeddings is None:
            hidden, embeddings = self.encoder(
                batch.optical, batch.velocity, batch.thermal, modality_mask
            )
        else:
            embeddings = precomputed_embeddings
            hidden = self.encoder.fuse(embeddings, modality_mask)
        for layer in self.layers:
            hidden = layer(hidden, batch.edge_index, batch.edge_attr)
        hazard_logits = self.classifier(hidden).squeeze(-1)
        trajectory_logits = self.trajectory(hidden, hazard_logits)
        return {
            "hazard_logits": hazard_logits,
            "hazard_probability": torch.sigmoid(hazard_logits),
            "trajectory_logits": trajectory_logits,
            "trajectory_probability": torch.sigmoid(trajectory_logits),
            "hidden": hidden,
            "modality_embeddings": embeddings,
        }
