#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pignn_glof.attribution import MODALITIES, exact_modality_shapley
from pignn_glof.config import load_config
from pignn_glof.data.loader import GraphNPZDataset, collate_graph_samples, split_by_region
from pignn_glof.experiment import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute exact three-modality Shapley values.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Explain one dataset index; default explains every held-out test snapshot",
    )
    parser.add_argument("--output", default="outputs/shapley.npz")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = GraphNPZDataset(args.manifest, config["data"]["graph"], graph_mode="physics")
    model = build_model("pi_gnn", config).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if args.sample_index is None:
        split = config["data"]["split"]
        _, _, test_set = split_by_region(
            dataset,
            train_fraction=float(split["train"]),
            val_fraction=float(split["validation"]),
            seed=int(config["project"]["seed"]),
        )
        indices = list(test_set.indices)
    else:
        indices = [args.sample_index]

    collected: dict[str, list[np.ndarray]] = {
        "values": [],
        "baseline": [],
        "full": [],
        "efficiency_residual": [],
    }
    region_ids: list[str] = []
    for index in indices:
        sample = dataset[index]
        batch = collate_graph_samples([sample]).to(args.device)
        result = exact_modality_shapley(model, batch)
        for name in collected:
            collected[name].append(result[name].cpu().numpy())
        region_ids.extend([sample.region_id] * sample.num_nodes)

    arrays = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    arrays["modality_names"] = np.asarray(MODALITIES)
    arrays["region_ids"] = np.asarray(region_ids)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)

    absolute = np.abs(arrays["values"])
    mean_absolute = absolute.mean(axis=0)
    normalized = mean_absolute / max(float(mean_absolute.sum()), 1e-12)
    dominant = absolute.argmax(axis=1)
    summary = {
        "samples": len(indices),
        "nodes": int(absolute.shape[0]),
        "normalized_mean_absolute_contribution": {
            name: float(normalized[index]) for index, name in enumerate(MODALITIES)
        },
        "dominant_nodes_percent": {
            name: float(100.0 * np.mean(dominant == index))
            for index, name in enumerate(MODALITIES)
        },
        "maximum_efficiency_residual": float(
            np.max(np.abs(arrays["efficiency_residual"]))
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
