from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def classification_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels = np.asarray(labels).astype(int).reshape(-1)
    probability = np.asarray(probability).astype(float).reshape(-1)
    prediction = (probability >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0
    )
    tp = int(np.sum((prediction == 1) & (labels == 1)))
    fp = int(np.sum((prediction == 1) & (labels == 0)))
    fn = int(np.sum((prediction == 0) & (labels == 1)))
    tn = int(np.sum((prediction == 0) & (labels == 0)))
    accuracy = float((tp + tn) / max(1, labels.size))
    iou = float(tp / max(1, tp + fp + fn))
    brier = float(np.mean((probability - labels) ** 2))
    auc = float(roc_auc_score(labels, probability)) if np.unique(labels).size == 2 else float("nan")
    return {
        "auc": auc,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": iou,
        "brier": brier,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": float(threshold),
    }


def trajectory_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.shape != prediction.shape:
        raise ValueError(f"Trajectory shapes differ: {target.shape} vs {prediction.shape}")
    error = prediction - target
    return {
        "mad": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "per_horizon_mad": np.mean(np.abs(error), axis=0).tolist(),
        "per_horizon_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
    }
