from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SENTINEL2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


def minmax_per_band(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize [N, C, H, W] reflectance independently by band."""
    if array.ndim != 4:
        raise ValueError("Optical array must have shape [N, C, H, W]")
    low = np.nanmin(array, axis=(0, 2, 3), keepdims=True)
    high = np.nanmax(array, axis=(0, 2, 3), keepdims=True)
    return np.clip((array - low) / np.maximum(high - low, eps), 0.0, 1.0).astype(np.float32)


def velocity_features(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    if vx.shape != vy.shape:
        raise ValueError("vx and vy must have identical shapes")
    magnitude = np.sqrt(vx**2 + vy**2)
    return np.stack([vx, vy, magnitude], axis=-1).astype(np.float32)


def monthly_anomaly(values: pd.Series, timestamps: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"value": values.astype(float), "timestamp": pd.to_datetime(timestamps)})
    climatology = frame.groupby(frame["timestamp"].dt.month)["value"].transform("mean")
    return (frame["value"] - climatology).to_numpy(dtype=np.float32)


def interpolate_short_gaps(
    frame: pd.DataFrame,
    value_columns: list[str],
    *,
    time_column: str = "timestamp",
    max_gap_days: int = 5,
) -> pd.DataFrame:
    frame = frame.sort_values(time_column).copy()
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame = frame.set_index(time_column)
    full = frame.resample("1D").asfreq()
    full[value_columns] = full[value_columns].interpolate(
        method="time", limit=max_gap_days, limit_area="inside"
    )
    return full.reset_index()


@dataclass(frozen=True)
class PreparedSnapshot:
    path: Path
    region_id: str
    timestamp: str


def write_snapshot(
    output_path: str | Path,
    *,
    optical: np.ndarray,
    velocity: np.ndarray,
    thermal: np.ndarray,
    coords: np.ndarray,
    elevation: np.ndarray,
    flow: np.ndarray,
    labels: np.ndarray,
    trajectory: np.ndarray | None = None,
) -> None:
    """Validate and write the journal-repository interchange format."""
    arrays = {
        "optical": np.asarray(optical, dtype=np.float32),
        "velocity": np.asarray(velocity, dtype=np.float32),
        "thermal": np.asarray(thermal, dtype=np.float32),
        "coords": np.asarray(coords, dtype=np.float32),
        "elevation": np.asarray(elevation, dtype=np.float32),
        "flow": np.asarray(flow, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.float32),
    }
    n = arrays["optical"].shape[0]
    if any(array.shape[0] != n for array in arrays.values()):
        raise ValueError("Every array must share the first node dimension")
    if trajectory is not None:
        arrays["trajectory"] = np.asarray(trajectory, dtype=np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def write_manifest(snapshots: list[PreparedSnapshot], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in snapshots:
            handle.write(
                json.dumps(
                    {
                        "path": str(item.path),
                        "region_id": item.region_id,
                        "timestamp": item.timestamp,
                    }
                )
                + "\n"
            )
