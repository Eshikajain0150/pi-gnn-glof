from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pignn_glof.data.graph import EDGE_ALIGNMENT, EDGE_DOWNHILL, EDGE_WEIGHT
from pignn_glof.data.schema import GraphBatch


def _standardize_by_graph(values: torch.Tensor, graph_id: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(values)
    for graph in torch.unique(graph_id):
        mask = graph_id == graph
        subset = values[mask]
        output[mask] = (subset - subset.mean()) / subset.std(unbiased=False).clamp_min(1e-6)
    return output


def risk_flux_continuity_loss(
    probability: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> torch.Tensor:
    """Penalize net divergence of edge-weighted hazard flux at each node."""
    src, dst = edge_index
    weight = edge_attr[:, EDGE_WEIGHT]
    flux = weight * (probability[src] - probability[dst])
    divergence = torch.zeros_like(probability)
    divergence.index_add_(0, src, -flux)
    divergence.index_add_(0, dst, flux)
    return divergence.abs().mean()


def temperature_elevation_loss(
    probability: torch.Tensor,
    thermal: torch.Tensor,
    elevation: torch.Tensor,
    graph_id: torch.Tensor,
) -> torch.Tensor:
    """Match hazard to the warm-and-low temperature/elevation physical driver."""
    latest_temperature = thermal[:, -1, 0]
    z_temperature = _standardize_by_graph(latest_temperature, graph_id)
    z_elevation = _standardize_by_graph(elevation, graph_id)
    physical_driver = torch.sigmoid(z_temperature - z_elevation).detach()
    return F.mse_loss(probability, physical_driver)


def flow_propagation_loss(
    probability: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    margin: float = 0.02,
) -> torch.Tensor:
    """Penalize loss of hazard along aligned downhill propagation edges."""
    src, dst = edge_index
    aligned = edge_attr[:, EDGE_ALIGNMENT].clamp_min(0.0)
    downhill = (edge_attr[:, EDGE_DOWNHILL] > 0).to(probability.dtype)
    weight = edge_attr[:, EDGE_WEIGHT] * aligned * downhill
    violation = F.relu(probability[src] - probability[dst] - float(margin))
    denom = weight.sum().clamp_min(1e-6)
    return (weight * violation.square()).sum() / denom


@dataclass
class PhysicsLosses:
    total: torch.Tensor
    continuity: torch.Tensor
    temperature_elevation: torch.Tensor
    flow: torch.Tensor


def physics_losses(
    probability: torch.Tensor,
    batch: GraphBatch,
    config: dict,
) -> PhysicsLosses:
    continuity = risk_flux_continuity_loss(probability, batch.edge_index, batch.edge_attr)
    temperature = temperature_elevation_loss(
        probability, batch.thermal, batch.elevation, batch.graph_id
    )
    flow = flow_propagation_loss(
        probability,
        batch.edge_index,
        batch.edge_attr,
        margin=float(config.get("downstream_margin", 0.02)),
    )
    total = (
        float(config["continuity_weight"]) * continuity
        + float(config["temperature_elevation_weight"]) * temperature
        + float(config["flow_weight"]) * flow
    )
    return PhysicsLosses(
        total=total,
        continuity=continuity,
        temperature_elevation=temperature,
        flow=flow,
    )
