from __future__ import annotations

import torch
import torch.nn.functional as F


EDGE_WEIGHT = 0
EDGE_DISTANCE = 1
EDGE_ALIGNMENT = 2
EDGE_DOWNHILL = 3


def _pairwise_geometry(
    coords: torch.Tensor,
    elevation: torch.Tensor,
    flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = coords.unsqueeze(0) - coords.unsqueeze(1)  # [source, destination, xy]
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
    direction = delta / distance.unsqueeze(-1)
    flow_unit = F.normalize(flow, dim=-1, eps=1e-6)
    alignment = (flow_unit.unsqueeze(1) * direction).sum(-1).clamp(-1.0, 1.0)
    downhill = ((elevation.unsqueeze(1) - elevation.unsqueeze(0)) / distance).clamp_min(0.0)
    return distance, alignment, downhill


def build_graph(
    coords: torch.Tensor,
    elevation: torch.Tensor,
    flow: torch.Tensor,
    *,
    top_k: int = 6,
    mode: str = "physics",
    distance_scale: float = 1_000.0,
    slope_scale: float = 0.10,
    flow_power: float = 2.0,
    cross_flow_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a directed row-normalized graph.

    Edge attributes are [normalized weight, scaled distance, flow alignment,
    normalized downhill slope]. A small cross-flow floor prevents isolated
    nodes when velocity vectors are noisy or nearly orthogonal.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape [N, 2]")
    n = int(coords.shape[0])
    if n < 2:
        raise ValueError("At least two nodes are required")
    top_k = max(1, min(int(top_k), n - 1))
    distance, alignment, downhill = _pairwise_geometry(coords, elevation, flow)
    eye = torch.eye(n, dtype=torch.bool, device=coords.device)
    proximity = torch.exp(-distance / float(distance_scale))

    if mode == "physics":
        forward = alignment.clamp_min(0.0).pow(float(flow_power))
        flow_gate = float(cross_flow_floor) + (1.0 - float(cross_flow_floor)) * forward
        slope_gate = 1.0 + downhill / float(slope_scale)
        raw = proximity * flow_gate * slope_gate
    elif mode == "knn":
        raw = proximity
    else:
        raise ValueError(f"Unknown graph mode: {mode}")
    raw = raw.masked_fill(eye, float("-inf"))

    values, destinations = torch.topk(raw, k=top_k, dim=1)
    sources = torch.arange(n, device=coords.device).unsqueeze(1).expand_as(destinations)
    finite_values = values.clamp_min(1e-12)
    normalized = finite_values / finite_values.sum(dim=1, keepdim=True).clamp_min(1e-12)

    src = sources.reshape(-1)
    dst = destinations.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0).long()
    scaled_distance = (distance[src, dst] / float(distance_scale)).clamp_max(20.0)
    scaled_downhill = (downhill[src, dst] / float(slope_scale)).clamp_max(20.0)
    edge_attr = torch.stack(
        [normalized.reshape(-1), scaled_distance, alignment[src, dst], scaled_downhill],
        dim=1,
    ).float()
    return edge_index, edge_attr
