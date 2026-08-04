#!/usr/bin/env python3
"""Verify repository-level SHA-256 inventory and release-manifest bindings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_FROM_CRITICAL = {
    "SHA256SUMS.txt",
    "RELEASE_MANIFEST.json",
    "evidence/release_manifest.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def critical_tree_digest() -> str:
    rows = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file()):
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDE_FROM_CRITICAL or ".git/" in rel or "__pycache__/" in rel:
            continue
        rows.append(f"{sha256(path)}  {rel}")
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


def main() -> int:
    checksum_rows = (ROOT / "SHA256SUMS.txt").read_text().splitlines()
    for row in checksum_rows:
        digest, rel = row.split("  ", 1)
        path = ROOT / rel
        if not path.is_file():
            raise SystemExit(f"FAIL missing file: {rel}")
        if sha256(path) != digest:
            raise SystemExit(f"FAIL checksum mismatch: {rel}")

    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    if manifest != json.loads((ROOT / "evidence/release_manifest.json").read_text()):
        raise SystemExit("FAIL release manifest copies differ")
    if critical_tree_digest() != manifest["critical_file_tree_sha256"]:
        raise SystemExit("FAIL critical file-tree digest")
    for archive in manifest["archives"].values():
        path = ROOT / archive["filename"]
        if sha256(path) != archive["sha256"]:
            raise SystemExit(f"FAIL archive binding: {archive['filename']}")
    print("PASS repository checksums and manifest bindings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
