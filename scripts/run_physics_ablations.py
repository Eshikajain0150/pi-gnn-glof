#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from pignn_glof.config import load_config
from pignn_glof.experiment import run_experiment
from pignn_glof.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the manuscript physics-loss ablations.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default="outputs/physics_ablations")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    base = load_config(args.config)
    if args.device:
        base["project"]["device"] = args.device
    root = Path(args.output)

    experiments = [
        ("full_pi_gnn", "pi_gnn", {}),
        ("without_continuity", "pi_gnn", {"continuity_weight": 0.0}),
        (
            "without_temperature_elevation",
            "pi_gnn",
            {"temperature_elevation_weight": 0.0},
        ),
        ("without_flow_alignment", "pi_gnn", {"flow_weight": 0.0}),
        ("no_physics", "plain_gnn", {"total_weight": 0.0}),
    ]

    results = {}
    rows = []
    for name, variant, physics_overrides in experiments:
        config = copy.deepcopy(base)
        config["physics"].update(physics_overrides)
        output = root / name
        result = run_experiment(
            config,
            variant=variant,
            output_dir=output,
            manifest=args.manifest,
        )
        result["ablation"] = name
        write_json(result, output / "metrics.json")
        results[name] = result
        metrics = result["metrics"]
        per_horizon = metrics.get("trajectory", {}).get("per_horizon_rmse", [])
        rows.append(
            {
                "configuration": name,
                "auc": metrics["auc"],
                "f1": metrics["f1"],
                "step_6_rmse": per_horizon[-1] if per_horizon else None,
                "composite_violation": metrics["physics_conformity"]["composite"],
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    write_json(results, root / "all_metrics.json")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
