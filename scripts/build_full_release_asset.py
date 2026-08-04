#!/usr/bin/env python3
"""Build the deterministic full repository release asset."""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = "pi-gnn-glof-v1.1.0-causal30-calendar"
FIXED_TIME = (2026, 8, 4, 0, 0, 0)
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return (
        path.is_file()
        and not any(part in EXCLUDED_PARTS for part in rel.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT.parent / f"{TOP}.zip")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(p for p in ROOT.rglob("*") if include(p)):
            rel = f"{TOP}/{path.relative_to(ROOT).as_posix()}"
            info = zipfile.ZipInfo(rel, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, path.read_bytes())
    with zipfile.ZipFile(args.output) as zf:
        bad = zf.testzip()
        if bad:
            raise SystemExit(f"Corrupt ZIP entry: {bad}")
    print(f"{sha256(args.output)}  {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
