#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pignn_glof.attribution import exact_modality_shapley
from pignn_glof.config import load_config
from pignn_glof.data.loader import GraphNPZDataset, collate_graph_samples
from pignn_glof.data.synthetic import generate_synthetic_dataset
from pignn_glof.experiment import build_model, run_experiment
from pignn_glof.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small end-to-end synthetic smoke test; outputs are not paper results."
    )
    parser.add_argument("--workdir", default="outputs/quick_test")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(args.workdir)
    manifest = generate_synthetic_dataset(
        root / "data",
        regions=6,
        samples_per_region=1,
        num_nodes=12,
        optical_size=12,
        velocity_steps=5,
        thermal_steps=5,
        horizon=3,
        seed=42,
    )
    overrides = {
        "project": {"device": args.device},
        "data": {"forecast_horizon": 3, "graph": {"top_k": 3}},
        "model": {
            "modality_embedding": 12,
            "hidden_dim": 24,
            "graph_layers": 2,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 2,
            "patience": 2,
            "batch_size": 2,
            "num_workers": 0,
        },
    }
    config = load_config(args.config, overrides)
    result = run_experiment(
        config,
        variant="pi_gnn",
        output_dir=root / "run",
        manifest=manifest,
    )

    dataset = GraphNPZDataset(manifest, config["data"]["graph"], graph_mode="physics")
    batch = collate_graph_samples([dataset[0]])
    model = build_model("pi_gnn", config)
    checkpoint = torch.load(root / "run" / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    attribution = exact_modality_shapley(model, batch)
    residual = float(attribution["efficiency_residual"].abs().max())
    if not torch.isfinite(attribution["values"]).all() or residual > 1e-5:
        raise RuntimeError(f"Shapley verification failed (maximum residual={residual:.3e})")
    report = {
        "status": "passed",
        "scope": "synthetic smoke test only; not a reproduction of manuscript metrics",
        "experiment": result,
        "shapley_max_efficiency_residual": residual,
    }
    write_json(report, root / "quick_test_report.json")
    print(json.dumps(report, indent=2))
    print("QUICK TEST PASSED")


if __name__ == "__main__":
    main()
