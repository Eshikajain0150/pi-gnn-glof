from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pignn_glof.data.loader import collate_graph_samples
from pignn_glof.data.schema import GraphBatch
from pignn_glof.metrics import classification_metrics, trajectory_metrics
from pignn_glof.physics import physics_losses


@dataclass
class EpochResult:
    loss: float
    supervised: float
    trajectory: float
    physics: float


def make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_graph_samples,
    )


def _positive_weight(loader: DataLoader, device: torch.device) -> torch.Tensor:
    positive = 0.0
    total = 0.0
    for batch in loader:
        positive += float(batch.labels.sum())
        total += float(batch.labels.numel())
    negative = total - positive
    return torch.tensor(negative / max(positive, 1.0), device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    physics_config: dict,
    trajectory_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 1.0,
    use_physics_loss: bool = True,
    pos_weight: torch.Tensor | None = None,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "supervised": 0.0, "trajectory": 0.0, "physics": 0.0}
    batches = 0
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch)
            logits = output["hazard_logits"]
            probability = output["hazard_probability"]
            supervised = F.binary_cross_entropy_with_logits(
                logits, batch.labels, pos_weight=pos_weight
            )
            trajectory = torch.zeros((), device=device)
            if batch.trajectory is not None:
                trajectory = F.mse_loss(output["trajectory_probability"], batch.trajectory)
            physical = physics_losses(probability, batch, physics_config)
            physics_term = physical.total if use_physics_loss else torch.zeros((), device=device)
            loss = (
                supervised
                + float(trajectory_weight) * trajectory
                + float(physics_config["total_weight"]) * physics_term
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
                optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["supervised"] += float(supervised.detach())
        totals["trajectory"] += float(trajectory.detach())
        totals["physics"] += float(physics_term.detach())
        batches += 1
    return EpochResult(**{key: value / max(1, batches) for key, value in totals.items()})


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    labels, probabilities, trajectories, trajectory_targets = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        labels.append(batch.labels.cpu().numpy())
        probabilities.append(output["hazard_probability"].cpu().numpy())
        if batch.trajectory is not None:
            trajectories.append(output["trajectory_probability"].cpu().numpy())
            trajectory_targets.append(batch.trajectory.cpu().numpy())
    result = {
        "labels": np.concatenate(labels),
        "probabilities": np.concatenate(probabilities),
    }
    if trajectories:
        result["trajectories"] = np.concatenate(trajectories)
        result["trajectory_targets"] = np.concatenate(trajectory_targets)
    return result


def evaluate_predictions(predictions: dict[str, np.ndarray], threshold: float) -> dict:
    metrics = classification_metrics(
        predictions["labels"], predictions["probabilities"], threshold
    )
    if "trajectories" in predictions:
        metrics["trajectory"] = trajectory_metrics(
            predictions["trajectory_targets"], predictions["trajectories"]
        )
    return metrics


@torch.no_grad()
def evaluate_physics_conformity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    physics_config: dict,
) -> dict[str, float]:
    """Evaluate unweighted physical-conformity terms on a fixed partition."""
    model.eval()
    totals = {"continuity": 0.0, "temperature_elevation": 0.0, "flow": 0.0}
    nodes = 0
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        terms = physics_losses(output["hazard_probability"], batch, physics_config)
        count = int(batch.labels.numel())
        totals["continuity"] += float(terms.continuity) * count
        totals["temperature_elevation"] += float(terms.temperature_elevation) * count
        totals["flow"] += float(terms.flow) * count
        nodes += count
    if nodes == 0:
        raise ValueError("Cannot evaluate physical conformity on an empty loader")
    means = {name: value / nodes for name, value in totals.items()}
    means["composite"] = sum(means.values()) / 3.0
    return means


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    config: dict,
    checkpoint_path: str | Path,
    use_physics_loss: bool,
) -> list[dict]:
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    pos_weight = _positive_weight(train_loader, device)
    best = float("inf")
    stale = 0
    history = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(training["epochs"]) + 1):
        train_result = run_epoch(
            model,
            train_loader,
            device=device,
            physics_config=config["physics"],
            trajectory_weight=float(training["trajectory_weight"]),
            optimizer=optimizer,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            use_physics_loss=use_physics_loss,
            pos_weight=pos_weight,
        )
        val_result = run_epoch(
            model,
            val_loader,
            device=device,
            physics_config=config["physics"],
            trajectory_weight=float(training["trajectory_weight"]),
            use_physics_loss=use_physics_loss,
            pos_weight=pos_weight,
        )
        history.append({"epoch": epoch, "train": train_result.__dict__, "validation": val_result.__dict__})
        if val_result.loss < best - 1e-8:
            best = val_result.loss
            stale = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation_loss": best}, checkpoint_path)
        else:
            stale += 1
            if stale >= int(training["patience"]):
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return history
