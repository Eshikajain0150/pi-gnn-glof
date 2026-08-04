#!/usr/bin/env python3
"""Reproduce leakage-controlled calendar baselines for causal30_v1.

The script fits all model parameters on the frozen 2019-2020 training anchors,
selects alert thresholds on the frozen 2021 validation anchors, and evaluates
once on the nine held-out 2022 anchors. It does not use the separate 125-scene
rerun and never creates a random split.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROTOCOL_ID = "CAUSAL_30DAY_GLOF_FORECAST_V1_1"
EXPECTED_COUNTS = {"train": 27, "validation": 7, "test": 9}
EVENT_DATE_2022 = date(2022, 5, 7)
OUTPUT_FILES = {
    "summary": "Calendar_Control_Summary.csv",
    "predictions": "Calendar_Control_Anchor_Predictions.csv",
    "excluded": "Calendar_Control_Excluded_Anchors.csv",
    "metadata": "Calendar_Control_Run_Metadata.json",
}


@dataclass(frozen=True)
class SummaryRow:
    control_id: str
    model: str
    train_records: int
    validation_records: int
    test_records: int
    test_positives: int
    test_negatives: int
    validation_selected_threshold: float
    validation_f1: float
    test_pr_auc: float
    test_roc_auc: float
    test_brier: float
    test_f1: float
    test_precision: float
    test_recall: float
    test_false_alarms: int
    test_true_positives: int
    test_false_negatives: int
    first_true_alert_date: str | None
    first_true_alert_lead_days: int | None
    interpretation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--archive",
        type=Path,
        default=Path("evidence/causal30_v1_release.zip"),
        help="Compact causal30_v1 evidence ZIP (default: evidence/causal30_v1_release.zip).",
    )
    source.add_argument(
        "--manifest",
        type=Path,
        help="Direct path to anchor_manifest.jsonl instead of the compact archive.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate to memory and fail if checked-in output files differ.",
    )
    return parser.parse_args()


def read_jsonl_from_archive(archive: Path) -> tuple[list[dict], dict]:
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise ValueError(f"Corrupt ZIP member: {zf.testzip()}")
        manifest_name = "causal30_v1/anchor_manifest.jsonl"
        protocol_name = "causal30_v1/protocol.json"
        rows = [
            json.loads(line)
            for line in zf.read(manifest_name).decode("utf-8").splitlines()
            if line.strip()
        ]
        protocol = json.loads(zf.read(protocol_name).decode("utf-8"))
    return rows, protocol


def read_jsonl_direct(manifest: Path) -> tuple[list[dict], dict]:
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, {"protocol_id": rows[0].get("protocol_id") if rows else None}


def validate(rows: list[dict], protocol: dict) -> None:
    if len(rows) != 43:
        raise ValueError(f"Expected 43 frozen anchors; observed {len(rows)}")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Unexpected protocol: {protocol.get('protocol_id')}")
    required = {"anchor_date", "split", "hazard_label", "event_start", "event_end", "protocol_id"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {index} missing fields: {sorted(missing)}")
        if row["protocol_id"] != PROTOCOL_ID:
            raise ValueError(f"Row {index} has a different protocol")
        if int(float(row["hazard_label"])) not in {0, 1}:
            raise ValueError(f"Row {index} has a non-binary label")
    counts = {split: sum(r["split"] == split for r in rows) for split in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Frozen split mismatch: {counts} != {EXPECTED_COUNTS}")
    ordered = sorted(rows, key=lambda r: r["anchor_date"])
    ranges = {
        split: [date.fromisoformat(r["anchor_date"]) for r in ordered if r["split"] == split]
        for split in EXPECTED_COUNTS
    }
    if max(ranges["train"]) >= min(ranges["validation"]):
        raise ValueError("Training and validation are not chronologically separated")
    if max(ranges["validation"]) >= min(ranges["test"]):
        raise ValueError("Validation and test are not chronologically separated")


def day_of_year(row: dict) -> int:
    return date.fromisoformat(row["anchor_date"]).timetuple().tm_yday


def features(rows: Iterable[dict], kind: str) -> np.ndarray:
    doy = np.asarray([day_of_year(r) for r in rows], dtype=float)
    if kind == "scalar":
        return doy[:, None]
    if kind == "harmonic":
        angle = 2.0 * math.pi * doy / 365.2425
        return np.column_stack([np.sin(angle), np.cos(angle)])
    raise ValueError(f"Unknown feature kind: {kind}")


def fit_logistic(train: list[dict], kind: str) -> Pipeline:
    y = np.asarray([int(float(r["hazard_label"])) for r in train], dtype=int)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    solver="liblinear",
                    C=1.0,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(features(train, kind), y)
    return model


def select_threshold(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Maximize validation F1; choose highest threshold under exact ties."""
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scored = [
        (float(f1_score(y, probabilities >= threshold, zero_division=0)), float(threshold))
        for threshold in candidates
    ]
    best_f1 = max(score for score, _ in scored)
    threshold = max(t for score, t in scored if math.isclose(score, best_f1, abs_tol=1e-12))
    return threshold, best_f1


def make_excluded_rows(rows: list[dict]) -> list[dict]:
    validation = sorted((r for r in rows if r["split"] == "validation"), key=lambda r: r["anchor_date"])
    event_start = date.fromisoformat(validation[0]["event_start"])
    event_end = date.fromisoformat(validation[0]["event_end"])
    retained = {date.fromisoformat(r["anchor_date"]) for r in validation}
    current = min(retained)
    final = max(retained)
    excluded: list[dict] = []
    while current <= final:
        horizon_end = current + timedelta(days=30)
        overlap = current < event_end and horizon_end >= event_start
        full_containment = current < event_start and horizon_end >= event_end
        if current not in retained and overlap and not full_containment:
            excluded.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "split": "validation",
                    "event_id": validation[0]["event_id"],
                    "anchor_date": current.isoformat(),
                    "forecast_window_start_exclusive": current.isoformat(),
                    "forecast_window_end_inclusive": horizon_end.isoformat(),
                    "documented_event_start": event_start.isoformat(),
                    "documented_event_end": event_end.isoformat(),
                    "exclusion_reason": "partial_overlap_without_full_interval_containment",
                }
            )
        current += timedelta(days=8)
    expected = ["2021-04-10", "2021-04-18"]
    observed = [r["anchor_date"] for r in excluded]
    if observed != expected:
        raise ValueError(f"Excluded-anchor reconstruction mismatch: {observed} != {expected}")
    return excluded


def csv_text(rows: list[dict], fieldnames: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def run(rows: list[dict], protocol: dict) -> dict[str, str]:
    validate(rows, protocol)
    split_rows = {
        split: sorted((r for r in rows if r["split"] == split), key=lambda r: r["anchor_date"])
        for split in EXPECTED_COUNTS
    }
    y = {
        split: np.asarray([int(float(r["hazard_label"])) for r in split_rows[split]], dtype=int)
        for split in split_rows
    }
    train_prevalence = float(np.mean(y["train"]))
    scalar_model = fit_logistic(split_rows["train"], "scalar")
    harmonic_model = fit_logistic(split_rows["train"], "harmonic")

    probabilities: dict[str, dict[str, np.ndarray]] = {
        "training_prevalence": {
            split: np.full(len(split_rows[split]), train_prevalence, dtype=float)
            for split in split_rows
        },
        "doy_scalar_logistic": {
            split: scalar_model.predict_proba(features(split_rows[split], "scalar"))[:, 1]
            for split in split_rows
        },
        "doy_harmonic_logistic": {
            split: harmonic_model.predict_proba(features(split_rows[split], "harmonic"))[:, 1]
            for split in split_rows
        },
    }
    display_names = {
        "training_prevalence": "Training-prevalence constant",
        "doy_scalar_logistic": "Day-of-year scalar logistic",
        "doy_harmonic_logistic": "Day-of-year harmonic logistic",
    }

    summary_rows: list[dict] = []
    prediction_rows: list[dict] = []
    thresholds: dict[str, float] = {}
    for control_id, by_split in probabilities.items():
        threshold, validation_f1 = select_threshold(y["validation"], by_split["validation"])
        thresholds[control_id] = threshold
        test_p = by_split["test"]
        test_pred = test_p >= threshold
        test_y = y["test"]
        alert_dates = [
            date.fromisoformat(r["anchor_date"])
            for r, predicted, label in zip(split_rows["test"], test_pred, test_y)
            if predicted and label == 1
        ]
        first_alert = min(alert_dates) if alert_dates else None
        summary = SummaryRow(
            control_id=control_id,
            model=display_names[control_id],
            train_records=len(split_rows["train"]),
            validation_records=len(split_rows["validation"]),
            test_records=len(split_rows["test"]),
            test_positives=int(np.sum(test_y == 1)),
            test_negatives=int(np.sum(test_y == 0)),
            validation_selected_threshold=threshold,
            validation_f1=validation_f1,
            test_pr_auc=float(average_precision_score(test_y, test_p)),
            test_roc_auc=float(roc_auc_score(test_y, test_p)),
            test_brier=float(brier_score_loss(test_y, test_p)),
            test_f1=float(f1_score(test_y, test_pred, zero_division=0)),
            test_precision=float(precision_score(test_y, test_pred, zero_division=0)),
            test_recall=float(recall_score(test_y, test_pred, zero_division=0)),
            test_false_alarms=int(np.sum((test_pred == 1) & (test_y == 0))),
            test_true_positives=int(np.sum((test_pred == 1) & (test_y == 1))),
            test_false_negatives=int(np.sum((test_pred == 0) & (test_y == 1))),
            first_true_alert_date=first_alert.isoformat() if first_alert else None,
            first_true_alert_lead_days=(EVENT_DATE_2022 - first_alert).days if first_alert else None,
            interpretation=(
                "Reference prevalence only; no temporal discrimination."
                if control_id == "training_prevalence"
                else "Calendar-only control; perfect ranking is compatible with monotonic seasonal ordering of the nine test anchors."
            ),
        )
        rounded_summary = {
            key: (f"{value:.12f}" if isinstance(value, float) else value)
            for key, value in asdict(summary).items()
        }
        summary_rows.append(rounded_summary)

        for split in ("train", "validation", "test"):
            for row, probability, label in zip(split_rows[split], by_split[split], y[split]):
                prediction_rows.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "control_id": control_id,
                        "model": display_names[control_id],
                        "split": split,
                        "anchor_date": row["anchor_date"],
                        "day_of_year": day_of_year(row),
                        "label": int(label),
                        "probability": f"{float(probability):.12f}",
                        "validation_selected_threshold": f"{threshold:.12f}",
                        "alert": int(probability >= threshold),
                        "event_id": row["event_id"],
                        "event_start": row["event_start"],
                        "event_end": row["event_end"],
                    }
                )

    summary_fields = list(asdict(SummaryRow(
        "", "", 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, None, None, ""
    )).keys())
    prediction_fields = [
        "protocol_id", "control_id", "model", "split", "anchor_date", "day_of_year",
        "label", "probability", "validation_selected_threshold", "alert", "event_id",
        "event_start", "event_end",
    ]
    excluded_rows = make_excluded_rows(rows)
    excluded_fields = list(excluded_rows[0].keys())

    logistic_parameters = {}
    for name, model in (("doy_scalar_logistic", scalar_model), ("doy_harmonic_logistic", harmonic_model)):
        lr = model.named_steps["logistic"]
        scaler = model.named_steps["scaler"]
        logistic_parameters[name] = {
            "scaler_mean": [float(v) for v in scaler.mean_],
            "scaler_scale": [float(v) for v in scaler.scale_],
            "coefficient": [float(v) for v in lr.coef_[0]],
            "intercept": float(lr.intercept_[0]),
        }
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "scope": "calendar-only leakage-control audit",
        "fitting": "training only (2019-2020)",
        "threshold_selection": "validation only (2021), maximum F1 with highest-threshold tie-break",
        "evaluation": "held-out test only (2022; nine correlated anchors from one event sequence)",
        "excluded_evidence": "separate 125-scene rerun",
        "models": list(probabilities),
        "thresholds": thresholds,
        "logistic_parameters": logistic_parameters,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    try:
        import sklearn
        metadata["software"]["scikit_learn"] = sklearn.__version__
    except Exception:
        pass

    return {
        OUTPUT_FILES["summary"]: csv_text(summary_rows, summary_fields),
        OUTPUT_FILES["predictions"]: csv_text(prediction_rows, prediction_fields),
        OUTPUT_FILES["excluded"]: csv_text(excluded_rows, excluded_fields),
        OUTPUT_FILES["metadata"]: json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    }


def main() -> int:
    args = parse_args()
    if args.manifest is not None:
        rows, protocol = read_jsonl_direct(args.manifest)
        source = args.manifest
    else:
        rows, protocol = read_jsonl_from_archive(args.archive)
        source = args.archive
    outputs = run(rows, protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.check:
        differences = []
        for filename, expected in outputs.items():
            path = args.output_dir / filename
            actual = path.read_text(encoding="utf-8") if path.exists() else None
            if actual != expected:
                differences.append(filename)
        if differences:
            print("Calendar-control outputs differ: " + ", ".join(differences), file=sys.stderr)
            return 1
        print(f"PASS: checked-in calendar-control outputs reproduce from {source}")
        return 0
    for filename, content in outputs.items():
        (args.output_dir / filename).write_text(content, encoding="utf-8", newline="")
        print(f"wrote {args.output_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
