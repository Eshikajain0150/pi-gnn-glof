#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment metrics into a CSV table.")
    parser.add_argument("--input", default="outputs/comparison")
    parser.add_argument("--output", default="outputs/comparison/summary.csv")
    args = parser.parse_args()
    root = Path(args.input)
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        metrics = record["metrics"]
        trajectory = metrics.get("trajectory", {})
        per_horizon_mad = trajectory.get("per_horizon_mad", [])
        per_horizon_rmse = trajectory.get("per_horizon_rmse", [])
        rows.append(
            {
                "model": record["variant"],
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "iou": metrics["iou"],
                "brier": metrics["brier"],
                "trajectory_rmse": trajectory.get("rmse"),
                "step_3_mad": per_horizon_mad[2] if len(per_horizon_mad) >= 3 else None,
                "step_3_rmse": per_horizon_rmse[2] if len(per_horizon_rmse) >= 3 else None,
                "step_6_mad": per_horizon_mad[5] if len(per_horizon_mad) >= 6 else None,
                "step_6_rmse": per_horizon_rmse[5] if len(per_horizon_rmse) >= 6 else None,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No variant metrics.json files found below {root}")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(destination)


if __name__ == "__main__":
    main()
