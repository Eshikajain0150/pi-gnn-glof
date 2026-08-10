from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_calendar_outputs_reproduce():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "reproduce_calendar_controls.py"),
            "--evidence",
            str(ROOT / "evidence" / "causal30_v1"),
            "--output-dir",
            str(ROOT),
            "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_calendar_control_scientific_values():
    rows = {row["control_id"]: row for row in read_rows(ROOT / "Calendar_Control_Summary.csv")}
    assert set(rows) == {
        "training_prevalence",
        "doy_scalar_logistic",
        "doy_harmonic_logistic",
    }
    assert abs(float(rows["training_prevalence"]["test_pr_auc"]) - 4 / 9) < 1e-12
    for key in ("doy_scalar_logistic", "doy_harmonic_logistic"):
        assert float(rows[key]["test_pr_auc"]) == 1.0
        assert float(rows[key]["test_roc_auc"]) == 1.0
        assert int(rows[key]["test_false_alarms"]) == 0
        assert int(rows[key]["first_true_alert_lead_days"]) == 11
    assert abs(float(rows["doy_scalar_logistic"]["test_brier"]) - 0.241421922506) < 1e-12
    assert abs(float(rows["doy_harmonic_logistic"]["test_brier"]) - 0.298775108961) < 1e-12


def test_partial_overlap_exclusions():
    rows = read_rows(ROOT / "Calendar_Control_Excluded_Anchors.csv")
    assert [row["anchor_date"] for row in rows] == ["2021-04-10", "2021-04-18"]
    assert all(row["split"] == "validation" for row in rows)
    assert all("partial_overlap" in row["exclusion_reason"] for row in rows)
