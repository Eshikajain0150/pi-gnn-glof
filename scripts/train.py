#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from pignn_glof.config import load_config
from pignn_glof.experiment import MODEL_VARIANTS, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one PI-GNN manuscript model variant.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--variant", choices=MODEL_VARIANTS, default="pi_gnn")
    parser.add_argument("--output", default="outputs/pi_gnn")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    overrides = {"project": {"device": args.device}} if args.device else None
    result = run_experiment(
        load_config(args.config, overrides),
        variant=args.variant,
        output_dir=args.output,
        manifest=args.manifest,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
