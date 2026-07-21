#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from pignn_glof.config import load_config
from pignn_glof.data.loader import GraphNPZDataset, split_by_region
from pignn_glof.experiment import MODEL_VARIANTS, build_model, graph_mode_for_variant
from pignn_glof.training import evaluate_predictions, make_loader, predict
from pignn_glof.utils import resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the region-held-out test set.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=MODEL_VARIANTS, default="pi_gnn")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_config(args.config, {"project": {"device": args.device}})
    device = resolve_device(args.device)
    dataset = GraphNPZDataset(
        args.manifest,
        config["data"]["graph"],
        graph_mode=graph_mode_for_variant(args.variant),
    )
    split = config["data"]["split"]
    _, _, test_set = split_by_region(
        dataset,
        train_fraction=float(split["train"]),
        val_fraction=float(split["validation"]),
        seed=int(config["project"]["seed"]),
    )
    loader = make_loader(
        test_set,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    model = build_model(args.variant, config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_predictions(
        predict(model, loader, device), float(config["training"]["decision_threshold"])
    )
    write_json(metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
