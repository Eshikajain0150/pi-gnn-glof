#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pignn_glof.data.preprocess import PreparedSnapshot, minmax_per_band, write_manifest, write_snapshot


REQUIRED = ("optical", "velocity", "thermal", "coords", "elevation", "flow", "labels")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize standardized GLOFNet snapshots and build a JSONL manifest. "
            "See docs/data.md for the required arrays."
        )
    )
    parser.add_argument("--input", required=True, help="Directory containing source .npz snapshots")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument(
        "--keep-optical-scale",
        action="store_true",
        help="Do not apply per-snapshot, per-band min-max normalization",
    )
    args = parser.parse_args()
    source_root = Path(args.input)
    output_root = Path(args.output)
    files = sorted(source_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz snapshots found in {source_root}")
    snapshots = []
    for source in files:
        with np.load(source, allow_pickle=False) as data:
            missing = [name for name in REQUIRED if name not in data]
            if missing:
                raise ValueError(f"{source} is missing arrays: {missing}")
            optical = np.asarray(data["optical"], dtype=np.float32)
            if not args.keep_optical_scale:
                optical = minmax_per_band(optical)
            trajectory = np.asarray(data["trajectory"], dtype=np.float32) if "trajectory" in data else None
            destination = output_root / source.name
            write_snapshot(
                destination,
                optical=optical,
                velocity=data["velocity"],
                thermal=data["thermal"],
                coords=data["coords"],
                elevation=data["elevation"],
                flow=data["flow"],
                labels=data["labels"],
                trajectory=trajectory,
            )
        metadata_path = source.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        snapshots.append(
            PreparedSnapshot(
                path=destination.name,
                region_id=str(metadata.get("region_id", source.stem)),
                timestamp=str(metadata.get("timestamp", "unknown")),
            )
        )
    manifest = output_root / "manifest.jsonl"
    write_manifest(snapshots, manifest)
    print(manifest)


if __name__ == "__main__":
    main()
