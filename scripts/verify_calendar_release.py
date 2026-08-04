#!/usr/bin/env python3
"""Verify the compact causal30_v1 + calendar-control archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

TOP = "causal30_v1_calendar/"
REQUIRED = {
    "anchor_manifest.jsonl",
    "protocol.json",
    "split_manifest.json",
    "claim_ledger.md",
    "calendar_control_protocol.md",
    "Calendar_Control_Summary.csv",
    "Calendar_Control_Anchor_Predictions.csv",
    "Calendar_Control_Excluded_Anchors.csv",
    "Calendar_Control_Run_Metadata.json",
    "checksums.sha256",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("evidence/causal30_v1_calendar_release.zip"))
    return p.parse_args()


def rows(zf: zipfile.ZipFile, name: str):
    text = zf.read(TOP + name).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def main() -> int:
    args = parse_args()
    with zipfile.ZipFile(args.archive) as zf:
        if zf.testzip() is not None:
            raise SystemExit("FAIL: corrupt ZIP")
        names = {name[len(TOP):] for name in zf.namelist() if name.startswith(TOP) and not name.endswith("/")}
        missing = REQUIRED - names
        if missing:
            raise SystemExit(f"FAIL: missing {sorted(missing)}")
        checks = zf.read(TOP + "checksums.sha256").decode("utf-8").splitlines()
        for line in checks:
            digest, relative = line.split("  ./", 1)
            actual = hashlib.sha256(zf.read(TOP + relative)).hexdigest()
            if actual != digest:
                raise SystemExit(f"FAIL: checksum mismatch {relative}")
        anchors = [json.loads(line) for line in zf.read(TOP + "anchor_manifest.jsonl").decode().splitlines() if line]
        if len(anchors) != 43:
            raise SystemExit("FAIL: anchor count")
        summary = {r["control_id"]: r for r in rows(zf, "Calendar_Control_Summary.csv")}
        if float(summary["doy_scalar_logistic"]["test_pr_auc"]) != 1.0:
            raise SystemExit("FAIL: scalar PR-AUC")
        if float(summary["doy_harmonic_logistic"]["test_roc_auc"]) != 1.0:
            raise SystemExit("FAIL: harmonic ROC-AUC")
        excluded = rows(zf, "Calendar_Control_Excluded_Anchors.csv")
        if [r["anchor_date"] for r in excluded] != ["2021-04-10", "2021-04-18"]:
            raise SystemExit("FAIL: excluded anchors")
    print(json.dumps({
        "status": "PASS",
        "anchors": 43,
        "calendar_controls": 3,
        "excluded_partial_overlap_anchors": 2,
        "archive": str(args.archive),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
