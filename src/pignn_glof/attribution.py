from __future__ import annotations

import itertools
import math

import torch

from pignn_glof.data.schema import GraphBatch
from pignn_glof.models.pignn import PIGNN


MODALITIES = ("reflectance", "velocity", "thermal")


@torch.no_grad()
def exact_modality_shapley(model: PIGNN, batch: GraphBatch) -> dict[str, torch.Tensor]:
    """Compute exact three-player Shapley values by evaluating all coalitions."""
    model_was_training = model.training
    model.eval()
    embeddings = model.encoder.encode_separate(batch.optical, batch.velocity, batch.thermal)
    values: dict[tuple[int, ...], torch.Tensor] = {}
    for size in range(4):
        for coalition in itertools.combinations(range(3), size):
            mask = torch.zeros(3, device=batch.optical.device)
            if coalition:
                mask[list(coalition)] = 1.0
            output = model(batch, modality_mask=mask, precomputed_embeddings=embeddings)
            values[coalition] = output["hazard_probability"].detach()  # type: ignore[union-attr]

    contributions = []
    factorial = math.factorial
    for modality in range(3):
        contribution = torch.zeros_like(values[()])
        others = [item for item in range(3) if item != modality]
        for size in range(3):
            for coalition in itertools.combinations(others, size):
                expanded = tuple(sorted(coalition + (modality,)))
                weight = factorial(size) * factorial(2 - size) / factorial(3)
                contribution += weight * (values[expanded] - values[coalition])
        contributions.append(contribution)
    stacked = torch.stack(contributions, dim=1)
    if model_was_training:
        model.train()
    return {
        "values": stacked,
        "baseline": values[()],
        "full": values[(0, 1, 2)],
        "efficiency_residual": stacked.sum(1) - (values[(0, 1, 2)] - values[()]),
    }
