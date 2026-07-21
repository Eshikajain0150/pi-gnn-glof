#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pignn_glof.config import load_config
from pignn_glof.experiment import MODEL_VARIANTS, run_experiment
from pignn_glof.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four manuscript comparison models.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default="outputs/comparison")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    overrides = {"project": {"device": args.device}} if args.device else None
    config = load_config(args.config, overrides)
    root = Path(args.output)
    results = {}
    for variant in MODEL_VARIANTS:
        results[variant] = run_experiment(
            config,
            variant=variant,
            output_dir=root / variant,
            manifest=args.manifest,
        )
    write_json(results, root / "all_metrics.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
