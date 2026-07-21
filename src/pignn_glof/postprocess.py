from __future__ import annotations

import torch

from pignn_glof.data.graph import EDGE_ALIGNMENT, EDGE_DOWNHILL, EDGE_WEIGHT
from pignn_glof.data.schema import GraphBatch


@torch.no_grad()
def physics_conformity_filter(
    probability: torch.Tensor,
    batch: GraphBatch,
    *,
    propagation_strength: float = 0.25,
    iterations: int = 2,
) -> torch.Tensor:
    """Apply a conservative downhill-flow propagation correction.

    The unfiltered prediction is retained as the lower bound; the filter only
    raises a downstream node when a high-risk upstream node is connected by a
    strongly aligned, downhill edge.
    """
    result = probability.clone()
    src, dst = batch.edge_index
    support = (
        batch.edge_attr[:, EDGE_WEIGHT]
        * batch.edge_attr[:, EDGE_ALIGNMENT].clamp_min(0.0)
        * (batch.edge_attr[:, EDGE_DOWNHILL] > 0).to(result.dtype)
    )
    for _ in range(int(iterations)):
        proposal = float(propagation_strength) * support * result[src]
        downstream = torch.zeros_like(result)
        if hasattr(downstream, "scatter_reduce_"):
            downstream.scatter_reduce_(0, dst, proposal, reduce="amax", include_self=False)
        else:  # pragma: no cover - compatibility for old PyTorch
            for index in range(dst.numel()):
                downstream[dst[index]] = torch.maximum(downstream[dst[index]], proposal[index])
        result = torch.maximum(result, downstream)
    return result.clamp(0.0, 1.0)
