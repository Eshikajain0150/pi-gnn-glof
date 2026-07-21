from __future__ import annotations

import torch
from torch import nn


class OpticalEncoder(nn.Module):
    def __init__(self, channels: int = 6, embedding_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.features(x).flatten(1))


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, embedding_dim, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.dropout(self.norm(hidden[-1]))


class MultimodalEncoder(nn.Module):
    modality_names = ("reflectance", "velocity", "thermal")

    def __init__(
        self,
        *,
        optical_channels: int,
        velocity_features: int,
        thermal_features: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.optical = OpticalEncoder(optical_channels, embedding_dim, dropout)
        self.velocity = SequenceEncoder(velocity_features, embedding_dim, dropout)
        self.thermal = SequenceEncoder(thermal_features, embedding_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def encode_separate(
        self,
        optical: torch.Tensor,
        velocity: torch.Tensor,
        thermal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.optical(optical), self.velocity(velocity), self.thermal(thermal)

    def fuse(
        self,
        embeddings: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if modality_mask is None:
            modality_mask = torch.ones(3, device=embeddings[0].device, dtype=embeddings[0].dtype)
        if modality_mask.shape != (3,):
            raise ValueError("modality_mask must have shape [3]")
        masked = [embedding * modality_mask[i] for i, embedding in enumerate(embeddings)]
        return self.fusion(torch.cat(masked, dim=-1))

    def forward(
        self,
        optical: torch.Tensor,
        velocity: torch.Tensor,
        thermal: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        separate = self.encode_separate(optical, velocity, thermal)
        return self.fuse(separate, modality_mask), separate
