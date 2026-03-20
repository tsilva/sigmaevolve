from __future__ import annotations

from typing import Any

from sigmaevolve.models import OUTCOME_SUCCEEDED


def compute_classification_metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length.")
    if not labels:
        raise ValueError("Cannot score an empty validation split.")
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    accuracy = correct / len(labels)
    return {"accuracy": accuracy, "correct": correct, "num_examples": len(labels)}


def compute_score(metrics: dict[str, Any] | None, outcome_reason: str | None, scorer_config: dict[str, Any]) -> float:
    if outcome_reason != OUTCOME_SUCCEEDED or not metrics:
        return 0.0
    primary_metric = scorer_config.get("primary_metric", "accuracy")
    value = metrics.get(primary_metric)
    if value is None:
        raise ValueError(f"Primary metric {primary_metric!r} not present in metrics.")
    return float(value)
