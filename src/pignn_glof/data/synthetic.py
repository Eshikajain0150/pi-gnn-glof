from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _grid(num_nodes: int) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(num_nodes)))
    rows = int(np.ceil(num_nodes / cols))
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.column_stack([x.reshape(-1), y.reshape(-1)])[:num_nodes].astype(np.float32)


def generate_synthetic_dataset(
    output_dir: str | Path,
    *,
    regions: int = 6,
    samples_per_region: int = 2,
    num_nodes: int = 24,
    optical_size: int = 16,
    velocity_steps: int = 8,
    thermal_steps: int = 8,
    horizon: int = 3,
    seed: int = 42,
) -> Path:
    """Generate small physically structured graphs for tests and demonstrations.

    These files are deliberately synthetic and must never be reported as paper
    results. They exercise every repository code path without downloading data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = []
    base_grid = _grid(num_nodes)

    for region in range(regions):
        region_bias = rng.normal(0.0, 0.15)
        for step in range(samples_per_region):
            coords = base_grid * 250.0 + rng.normal(0.0, 8.0, base_grid.shape)
            elevation = 4_800.0 - coords[:, 1] * 0.35 + rng.normal(0.0, 12.0, num_nodes)
            flow = np.column_stack(
                [rng.normal(0.05, 0.02, num_nodes), rng.normal(1.0, 0.08, num_nodes)]
            ).astype(np.float32)
            temperature_driver = (
                -(elevation - elevation.mean()) / (elevation.std() + 1e-6)
                + region_bias
                + 0.15 * step
            )
            velocity_driver = np.linspace(0.0, 1.0, velocity_steps, dtype=np.float32)[None, :]
            velocity_driver = velocity_driver + rng.normal(0.0, 0.12, (num_nodes, velocity_steps))
            unstable_band = (coords[:, 1] > np.quantile(coords[:, 1], 0.55)).astype(np.float32)
            velocity_driver += unstable_band[:, None] * 0.65

            vx = 0.1 * velocity_driver + rng.normal(0.0, 0.03, velocity_driver.shape)
            vy = velocity_driver + rng.normal(0.0, 0.05, velocity_driver.shape)
            magnitude = np.sqrt(vx**2 + vy**2)
            velocity = np.stack([vx, vy, magnitude], axis=-1).astype(np.float32)
            thermal = (
                temperature_driver[:, None]
                + np.sin(np.linspace(0, np.pi, thermal_steps))[None, :]
                + rng.normal(0.0, 0.15, (num_nodes, thermal_steps))
            )[..., None].astype(np.float32)

            optical = rng.normal(0.35, 0.12, (num_nodes, 6, optical_size, optical_size)).astype(np.float32)
            optical[:, 3] += unstable_band[:, None, None] * 0.2
            optical[:, 4:] -= unstable_band[:, None, None, None] * 0.08
            optical = np.clip(optical, 0.0, 1.0)

            risk = 0.9 * temperature_driver + 1.3 * magnitude[:, -1] + 0.8 * unstable_band
            risk = (risk - np.median(risk)) / (np.std(risk) + 1e-6)
            probability = 1.0 / (1.0 + np.exp(-risk))
            labels = (probability > 0.52).astype(np.float32)
            trajectory = np.stack(
                [np.clip(probability + 0.04 * (h + 1) * unstable_band, 0.0, 1.0) for h in range(horizon)],
                axis=1,
            ).astype(np.float32)

            filename = f"region_{region:02d}_step_{step:02d}.npz"
            np.savez_compressed(
                output_dir / filename,
                optical=optical,
                velocity=velocity,
                thermal=thermal,
                coords=coords.astype(np.float32),
                elevation=elevation.astype(np.float32),
                flow=flow,
                labels=labels,
                trajectory=trajectory,
            )
            manifest.append(
                {
                    "path": filename,
                    "region_id": f"synthetic-region-{region:02d}",
                    "timestamp": f"2025-01-{step + 1:02d}",
                    "synthetic": True,
                }
            )

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record) + "\n")
    return manifest_path
