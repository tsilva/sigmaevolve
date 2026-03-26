from __future__ import annotations

from typing import Any


def compute_classification_metrics(
    predictions: list[int],
    labels: list[int],
) -> dict[str, Any]:
    # Validate the scoring inputs before computing aggregates.
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length.")
    if not labels:
        raise ValueError("Cannot score an empty validation split.")

    # Derive the aggregate metrics from the aligned predictions and labels.
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    accuracy = correct / len(labels)

    # Return the metrics payload in a reviewable shape.
    return {
        "accuracy": accuracy,
        "correct": correct,
        "num_examples": len(labels),
    }


def compute_score(
    metrics: dict[str, Any] | None,
    outcome_reason: str | None,
    scorer_config: dict[str, Any],
) -> float:
    del outcome_reason

    # Fall back to zero when the trial never produced metrics.
    if not metrics:
        return 0.0

    # Resolve the configured metric before converting it to a float score.
    primary_metric = scorer_config.get("primary_metric", "accuracy")
    value = metrics.get(primary_metric)
    if value is None:
        raise ValueError(f"Primary metric {primary_metric!r} not present in metrics.")

    return float(value)
