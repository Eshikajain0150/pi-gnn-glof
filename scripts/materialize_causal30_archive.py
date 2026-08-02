#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, hashlib
from pathlib import Path
EXPECTED_SHA256 = "0fb298684852b69a3f7b03bbb6748adc59aa6d64b4436235f41e71a36a9e9aaa"

def materialize(parts_dir: Path, output: Path) -> Path:
    parts=sorted(parts_dir.glob("causal30_v1_release.zip.b64.part*"))
    if not parts:
        raise FileNotFoundError(f"No archive parts found in {parts_dir}")
    encoded=''.join(p.read_text(encoding='ascii').strip() for p in parts)
    data=base64.b64decode(encoded, validate=True)
    digest=hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"archive SHA-256 mismatch: {digest}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return output

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parts-dir', type=Path, default=Path('evidence/archive_parts'))
    p.add_argument('--output', type=Path, default=Path('evidence/causal30_v1_release.zip'))
    a=p.parse_args()
    print(materialize(a.parts_dir, a.output))
if __name__=='__main__': main()
