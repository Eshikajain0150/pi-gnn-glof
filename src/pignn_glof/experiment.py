from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from pignn_glof.config import save_config
from pignn_glof.data.loader import GraphNPZDataset, split_by_region
from pignn_glof.models import CNNOnly, PIGNN, TemporalGRU
from pignn_glof.training import (
    evaluate_physics_conformity,
    evaluate_predictions,
    fit,
    make_loader,
    predict,
)
from pignn_glof.utils import resolve_device, seed_everything, write_json


MODEL_VARIANTS = ("cnn", "temporal_gru", "plain_gnn", "pi_gnn")


def _model_arguments(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    model = config["model"]
    return {
        "optical_channels": int(data["optical_channels"]),
        "velocity_features": int(data["velocity_features"]),
        "thermal_features": int(data["thermal_features"]),
        "forecast_horizon": int(data["forecast_horizon"]),
        "modality_embedding": int(model["modality_embedding"]),
        "hidden_dim": int(model["hidden_dim"]),
        "graph_layers": int(model["graph_layers"]),
        "edge_features": int(model["edge_features"]),
        "dropout": float(model["dropout"]),
    }


def build_model(variant: str, config: dict[str, Any]) -> torch.nn.Module:
    """Construct one of the four manuscript model variants."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {MODEL_VARIANTS}")
    arguments = _model_arguments(config)
    if variant == "cnn":
        return CNNOnly(**arguments)
    if variant == "temporal_gru":
        return TemporalGRU(**arguments)
    return PIGNN(**arguments)


def graph_mode_for_variant(variant: str) -> str:
    return "physics" if variant == "pi_gnn" else "knn"


def run_experiment(
    config: dict[str, Any],
    *,
    variant: str,
    output_dir: str | Path,
    manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Train, validate, and test a single leakage-safe model variant."""
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = resolve_device(str(config["project"].get("device", "auto")))
    manifest_path = Path(manifest or config["data"]["manifest"]).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = GraphNPZDataset(
        manifest_path,
        graph_config=config["data"]["graph"],
        graph_mode=graph_mode_for_variant(variant),
    )
    split = config["data"]["split"]
    train_set, validation_set, test_set = split_by_region(
        dataset,
        train_fraction=float(split["train"]),
        val_fraction=float(split["validation"]),
        seed=seed,
    )
    training = config["training"]
    loader_options = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
    }
    train_loader = make_loader(train_set, shuffle=True, **loader_options)
    validation_loader = make_loader(validation_set, shuffle=False, **loader_options)
    test_loader = make_loader(test_set, shuffle=False, **loader_options)

    model = build_model(variant, config).to(device)
    history = fit(
        model,
        train_loader,
        validation_loader,
        device=device,
        config=config,
        checkpoint_path=output_dir / "best_model.pt",
        use_physics_loss=variant == "pi_gnn",
    )
    predictions = predict(model, test_loader, device)
    threshold = float(training["decision_threshold"])
    metrics = evaluate_predictions(predictions, threshold)
    metrics["physics_conformity"] = evaluate_physics_conformity(
        model, test_loader, device, config["physics"]
    )
    result: dict[str, Any] = {
        "variant": variant,
        "device": str(device),
        "manifest": str(manifest_path),
        "samples": {
            "train": len(train_set),
            "validation": len(validation_set),
            "test": len(test_set),
        },
        "metrics": metrics,
        "epochs_completed": len(history),
    }
    save_config(config, output_dir / "resolved_config.yaml")
    write_json(history, output_dir / "history.json")
    write_json(result, output_dir / "metrics.json")
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    return result
