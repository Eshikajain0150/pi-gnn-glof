#!/usr/bin/env python3
"""Build a deterministic compact calendar-control evidence archive."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_ARCHIVE = ROOT / "evidence" / "causal30_v1_release.zip"
OUT_ARCHIVE = ROOT / "evidence" / "causal30_v1_calendar_release.zip"
TOP = "causal30_v1_calendar"
NEW_FILES = [
    ROOT / "Calendar_Control_Summary.csv",
    ROOT / "Calendar_Control_Anchor_Predictions.csv",
    ROOT / "Calendar_Control_Excluded_Anchors.csv",
    ROOT / "Calendar_Control_Run_Metadata.json",
    ROOT / "CALENDAR_CONTROL_PROTOCOL.md",
    ROOT / "evidence" / "claim_ledger.md",
]
FIXED_TIME = (2026, 8, 4, 0, 0, 0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_deterministic_zip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            relative = path.relative_to(source.parent).as_posix()
            info = zipfile.ZipInfo(relative, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, path.read_bytes())


def main() -> None:
    if not OLD_ARCHIVE.exists():
        raise FileNotFoundError(OLD_ARCHIVE)
    for path in NEW_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        with zipfile.ZipFile(OLD_ARCHIVE) as zf:
            if zf.testzip() is not None:
                raise ValueError("Existing causal30 archive is corrupt")
            zf.extractall(temp / "old")
        original = temp / "old" / "causal30_v1"
        target = temp / TOP
        shutil.copytree(original, target)
        # Replace the old internal claim ledger with the calendar-aware ledger.
        shutil.copy2(ROOT / "evidence" / "claim_ledger.md", target / "claim_ledger.md")
        shutil.copy2(ROOT / "CALENDAR_CONTROL_PROTOCOL.md", target / "calendar_control_protocol.md")
        for path in NEW_FILES[:4]:
            shutil.copy2(path, target / path.name)

        checksum_path = target / "checksums.sha256"
        rows = []
        for path in sorted(p for p in target.rglob("*") if p.is_file() and p != checksum_path):
            rows.append(f"{sha256(path)}  ./{path.relative_to(target).as_posix()}")
        checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        write_deterministic_zip(target, OUT_ARCHIVE)

    with zipfile.ZipFile(OUT_ARCHIVE) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"Built archive is corrupt at {bad}")
    print(json.dumps({
        "archive": str(OUT_ARCHIVE.relative_to(ROOT)),
        "sha256": sha256(OUT_ARCHIVE),
        "bytes": OUT_ARCHIVE.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
