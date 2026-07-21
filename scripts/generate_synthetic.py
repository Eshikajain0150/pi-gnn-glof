#!/usr/bin/env python
from __future__ import annotations

import argparse

from pignn_glof.data.synthetic import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic PI-GNN smoke-test dataset (not paper results)."
    )
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--regions", type=int, default=6)
    parser.add_argument("--samples-per-region", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = generate_synthetic_dataset(
        args.output,
        regions=args.regions,
        samples_per_region=args.samples_per_region,
        num_nodes=args.nodes,
        horizon=args.horizon,
        seed=args.seed,
    )
    print(manifest)


if __name__ == "__main__":
    main()
