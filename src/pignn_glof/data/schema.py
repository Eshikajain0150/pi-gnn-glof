from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


@dataclass
class GraphSample:
    optical: torch.Tensor
    velocity: torch.Tensor
    thermal: torch.Tensor
    coords: torch.Tensor
    elevation: torch.Tensor
    flow: torch.Tensor
    labels: torch.Tensor
    trajectory: torch.Tensor | None
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    region_id: str
    timestamp: str

    @property
    def num_nodes(self) -> int:
        return int(self.optical.shape[0])

    def validate(self) -> None:
        n = self.num_nodes
        expected = {
            "velocity": self.velocity.shape[0],
            "thermal": self.thermal.shape[0],
            "coords": self.coords.shape[0],
            "elevation": self.elevation.shape[0],
            "flow": self.flow.shape[0],
            "labels": self.labels.shape[0],
        }
        bad = {name: size for name, size in expected.items() if size != n}
        if bad:
            raise ValueError(f"Node dimension mismatch; expected {n}: {bad}")
        if self.coords.shape[-1] != 2 or self.flow.shape[-1] != 2:
            raise ValueError("coords and flow must end in x/y components")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.edge_attr.ndim != 2 or self.edge_attr.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_attr must have shape [E, F]")
        if self.trajectory is not None and self.trajectory.shape[0] != n:
            raise ValueError("trajectory must have shape [N, horizon]")


@dataclass
class GraphBatch:
    optical: torch.Tensor
    velocity: torch.Tensor
    thermal: torch.Tensor
    coords: torch.Tensor
    elevation: torch.Tensor
    flow: torch.Tensor
    labels: torch.Tensor
    trajectory: torch.Tensor | None
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    graph_id: torch.Tensor
    region_ids: list[str]
    timestamps: list[str]

    @property
    def num_nodes(self) -> int:
        return int(self.optical.shape[0])

    def to(self, device: torch.device | str) -> "GraphBatch":
        values: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return GraphBatch(**values)

    def modality_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.optical, self.velocity, self.thermal
