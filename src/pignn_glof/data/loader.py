from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .graph import build_graph
from .schema import GraphBatch, GraphSample


REQUIRED_KEYS = {
    "optical",
    "velocity",
    "thermal",
    "coords",
    "elevation",
    "flow",
    "labels",
}


class GraphNPZDataset(Dataset[GraphSample]):
    """Load prepared graph snapshots listed in a JSONL manifest."""

    def __init__(self, manifest: str | Path, graph_config: dict, graph_mode: str = "physics"):
        self.manifest_path = Path(manifest)
        self.root = self.manifest_path.parent
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Manifest is empty: {manifest}")
        self.graph_config = dict(graph_config)
        self.graph_mode = graph_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GraphSample:
        record = self.records[index]
        path = Path(record["path"])
        if not path.is_absolute():
            path = self.root / path
        with np.load(path, allow_pickle=False) as data:
            missing = REQUIRED_KEYS - set(data.files)
            if missing:
                raise ValueError(f"{path} is missing keys: {sorted(missing)}")
            tensors = {key: torch.from_numpy(np.asarray(data[key])).float() for key in REQUIRED_KEYS}
            trajectory = torch.from_numpy(np.asarray(data["trajectory"])).float() if "trajectory" in data else None
        edge_index, edge_attr = build_graph(
            tensors["coords"],
            tensors["elevation"],
            tensors["flow"],
            mode=self.graph_mode,
            **self.graph_config,
        )
        sample = GraphSample(
            optical=tensors["optical"],
            velocity=tensors["velocity"],
            thermal=tensors["thermal"],
            coords=tensors["coords"],
            elevation=tensors["elevation"],
            flow=tensors["flow"],
            labels=tensors["labels"],
            trajectory=trajectory,
            edge_index=edge_index,
            edge_attr=edge_attr,
            region_id=str(record.get("region_id", path.stem)),
            timestamp=str(record.get("timestamp", "unknown")),
        )
        sample.validate()
        return sample


def collate_graph_samples(samples: Sequence[GraphSample]) -> GraphBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    offsets = []
    total = 0
    for sample in samples:
        offsets.append(total)
        total += sample.num_nodes
    edge_index = torch.cat(
        [sample.edge_index + offset for sample, offset in zip(samples, offsets)], dim=1
    )
    trajectories = [sample.trajectory for sample in samples]
    if any(item is None for item in trajectories) and not all(item is None for item in trajectories):
        raise ValueError("Either every sample or no sample must contain trajectories")
    trajectory = None if trajectories[0] is None else torch.cat(trajectories, dim=0)  # type: ignore[arg-type]
    return GraphBatch(
        optical=torch.cat([sample.optical for sample in samples], dim=0),
        velocity=torch.cat([sample.velocity for sample in samples], dim=0),
        thermal=torch.cat([sample.thermal for sample in samples], dim=0),
        coords=torch.cat([sample.coords for sample in samples], dim=0),
        elevation=torch.cat([sample.elevation for sample in samples], dim=0),
        flow=torch.cat([sample.flow for sample in samples], dim=0),
        labels=torch.cat([sample.labels for sample in samples], dim=0),
        trajectory=trajectory,
        edge_index=edge_index,
        edge_attr=torch.cat([sample.edge_attr for sample in samples], dim=0),
        graph_id=torch.cat(
            [torch.full((sample.num_nodes,), i, dtype=torch.long) for i, sample in enumerate(samples)]
        ),
        region_ids=[sample.region_id for sample in samples],
        timestamps=[sample.timestamp for sample in samples],
    )


def split_by_region(
    dataset: GraphNPZDataset,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Create deterministic, disjoint region-level train/validation/test splits."""
    regions = sorted({str(record.get("region_id", i)) for i, record in enumerate(dataset.records)})
    rng = np.random.default_rng(seed)
    rng.shuffle(regions)
    n = len(regions)
    if n < 3:
        raise ValueError("At least three unique regions are needed for leakage-safe splitting")
    n_train = max(1, int(round(n * train_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_regions = set(regions[:n_train])
    val_regions = set(regions[n_train : n_train + n_val])
    test_regions = set(regions[n_train + n_val :])

    def indices(selected: set[str]) -> list[int]:
        return [
            i
            for i, record in enumerate(dataset.records)
            if str(record.get("region_id", i)) in selected
        ]

    return Subset(dataset, indices(train_regions)), Subset(dataset, indices(val_regions)), Subset(dataset, indices(test_regions))
