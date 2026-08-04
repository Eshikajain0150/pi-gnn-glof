#!/usr/bin/env python3
"""Create an external binding among tag commit, Zenodo DOI and release bytes.

The output is intentionally generated after the Git commit exists and is not
committed back into that same commit, avoiding a self-referential hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

TAG = "v1.1.0-causal30-calendar"
VERSION = TAG


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--commit", required=True, help="Full 40-character Git commit SHA")
    p.add_argument("--doi", required=True, help="New Zenodo version DOI")
    p.add_argument("--asset", required=True, type=Path, help="Exact release ZIP")
    p.add_argument("--output", type=Path, default=Path("RELEASE_ATTESTATION.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    commit = args.commit.strip().lower()
    doi = args.doi.strip().lower().removeprefix("https://doi.org/")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("Commit must be the full 40-character hexadecimal SHA.")
    if not re.fullmatch(r"10\.5281/zenodo\.\d+", doi):
        raise SystemExit("DOI must look like 10.5281/zenodo.<record_id>.")
    if doi == "10.5281/zenodo.21761944":
        raise SystemExit("Use the NEW version DOI, not the v1.0.0-causal30 DOI.")
    if not args.asset.is_file():
        raise SystemExit(f"Release asset not found: {args.asset}")

    root = Path(__file__).resolve().parents[1]
    compact = root / "evidence" / "causal30_v1_calendar_release.zip"
    old = root / "evidence" / "causal30_v1_release.zip"
    record = {
        "schema": "pi-gnn-glof-release-attestation-v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repository": "https://github.com/Eshikajain0150/pi-gnn-glof",
        "tag": TAG,
        "version": VERSION,
        "commit_sha": commit,
        "zenodo_version_doi": doi,
        "previous_version_doi": "10.5281/zenodo.21761944",
        "release_asset": {
            "filename": args.asset.name,
            "bytes": args.asset.stat().st_size,
            "sha256": sha256(args.asset),
        },
        "embedded_evidence_archives": {
            "causal30_v1_release.zip": sha256(old),
            "causal30_v1_calendar_release.zip": sha256(compact),
        },
        "interpretation_boundary": "Nine held-out anchors are one correlated 2022 event sequence; the separate 125-scene rerun is excluded.",
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
