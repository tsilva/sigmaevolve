from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sigmaevolve.scoring import compute_classification_metrics


DEBUG_METRIC_KEYS = (
    "early_stopped",
    "early_stop_epoch",
    "early_stopping_patience",
    "epochs_completed",
    "best_validation_accuracy_seen",
    "epochs_without_improvement",
)
EVAL_ARTIFACT_METRIC_KEYS = (
    "accuracy",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
)


def coerce_optional_scalar(value: Any, cast) -> Any | None:
    # Treat missing and empty array payloads as absent scalar values.
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, np.generic):
        scalar = scalar.item()

    # Coerce only the first scalar-like value and ignore invalid payloads.
    try:
        return cast(scalar)
    except (TypeError, ValueError):
        return None


def load_eval_artifacts(
    *,
    eval_dir: Path,
    labels_path: str,
) -> list[dict[str, Any]]:
    # Load the fixed validation labels once and reuse them for every artifact.
    labels = np.load(labels_path)
    label_list = labels.astype(int).tolist()
    artifacts: list[dict[str, Any]] = []

    # Build a normalized metrics payload for each completed evaluation artifact.
    for eval_path in sorted(eval_dir.glob("*.npz")):
        with np.load(eval_path) as payload:
            if "predictions" not in payload:
                continue

            # Normalize predictions before computing the canonical metrics.
            predictions = payload["predictions"]
            if predictions.ndim > 1:
                predictions = predictions.argmax(axis=1)

            metrics = compute_classification_metrics(
                predictions.astype(int).tolist(),
                label_list,
            )
            for key in EVAL_ARTIFACT_METRIC_KEYS:
                if key not in payload:
                    continue

                value = coerce_optional_scalar(payload[key], float)
                if value is not None:
                    metrics[key] = value

            # Preserve the artifact metadata alongside the derived metrics.
            metrics.setdefault("val_acc", metrics.get("accuracy"))
            eval_index = (
                coerce_optional_scalar(payload["eval_index"], int)
                if "eval_index" in payload
                else None
            )
            elapsed_time_sec = (
                coerce_optional_scalar(payload["elapsed_time_sec"], float)
                if "elapsed_time_sec" in payload
                else None
            )
            epoch = (
                coerce_optional_scalar(payload["epoch"], int)
                if "epoch" in payload
                else None
            )

            artifacts.append(
                {
                    "path": str(eval_path),
                    "eval_index": eval_index,
                    "elapsed_time_sec": elapsed_time_sec,
                    "epoch": epoch,
                    "metrics": metrics,
                }
            )

    return artifacts


def select_best_eval(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        artifacts,
        key=lambda artifact: (
            -float(artifact["metrics"]["accuracy"]),
            float(artifact.get("elapsed_time_sec") if artifact.get("elapsed_time_sec") is not None else float("inf")),
            int(artifact.get("eval_index") if artifact.get("eval_index") is not None else np.iinfo(np.int64).max),
        ),
    )


def select_last_completed_eval(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        artifacts,
        key=lambda artifact: (
            float(artifact.get("elapsed_time_sec") if artifact.get("elapsed_time_sec") is not None else -1.0),
            int(artifact.get("eval_index") if artifact.get("eval_index") is not None else -1),
        ),
    )


def apply_debug_metrics(metrics: dict[str, Any], debug_payload: dict[str, Any] | None) -> None:
    # Copy through the debug-only metrics that the runner reported explicitly.
    if not debug_payload:
        return
    for key in DEBUG_METRIC_KEYS:
        if key in debug_payload:
            metrics[key] = debug_payload[key]


def build_final_metrics_payload(
    *,
    artifacts: list[dict[str, Any]],
    progress_payload: dict[str, Any] | None,
    process_elapsed_sec: float,
    timed_out: bool,
    debug_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    # Derive the best and latest evaluations before composing the final payload.
    best_artifact = select_best_eval(artifacts)
    last_artifact = select_last_completed_eval(artifacts)

    last_completed_eval_sec = last_artifact.get("elapsed_time_sec")
    if last_completed_eval_sec is None and progress_payload:
        last_completed_eval_sec = progress_payload.get("last_completed_eval_sec")

    time_to_best_eval_sec = best_artifact.get("elapsed_time_sec")
    time_since_last_eval_sec = None
    if last_completed_eval_sec is not None:
        time_since_last_eval_sec = max(0.0, float(process_elapsed_sec) - float(last_completed_eval_sec))

    last_phase = None
    if progress_payload:
        last_phase = progress_payload.get("phase") or progress_payload.get("current_phase")

    # Flag timeouts that ended with a measurable amount of unevaluated work.
    had_unscored_work_at_timeout = bool(
        timed_out
        and time_since_last_eval_sec is not None
        and time_since_last_eval_sec > 0.05
        and (last_phase in {None, "train"})
    )

    # Merge the best evaluation metrics with run-level diagnostic fields.
    metrics = dict(best_artifact["metrics"])
    metrics.update(
        {
            "best_accuracy": best_artifact["metrics"]["accuracy"],
            "time_to_best_eval_sec": time_to_best_eval_sec,
            "best_eval_index": best_artifact.get("eval_index"),
            "best_eval_epoch": best_artifact.get("epoch"),
            "best_eval_path": best_artifact["path"],
            "last_completed_eval_sec": last_completed_eval_sec,
            "last_completed_eval_index": last_artifact.get("eval_index"),
            "timed_out": timed_out,
            "time_since_last_eval_sec": time_since_last_eval_sec,
            "had_unscored_work_at_timeout": had_unscored_work_at_timeout,
            "last_phase": last_phase,
            "eval_count": len(artifacts),
            "process_elapsed_sec": float(process_elapsed_sec),
        }
    )

    apply_debug_metrics(metrics, debug_payload)
    return metrics


def build_active_metrics_payload(
    *,
    artifacts: list[dict[str, Any]],
    progress_payload: dict[str, Any] | None,
    process_elapsed_sec: float,
    debug_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    # Start from the process-level progress fields that are always available.
    progress = dict(progress_payload or {})
    metrics: dict[str, Any] = {"process_elapsed_sec": float(process_elapsed_sec)}

    last_phase = progress.get("phase") or progress.get("current_phase")
    if last_phase is not None:
        metrics["last_phase"] = last_phase

    progress_eval_index = coerce_optional_scalar(progress.get("eval_index"), int)
    debug_eval_count = coerce_optional_scalar((debug_payload or {}).get("eval_count"), int)
    eval_count = max(len(artifacts), progress_eval_index or 0, debug_eval_count or 0)
    metrics["eval_count"] = eval_count

    # Reconcile runner progress with the most recent persisted evaluation artifact.
    last_completed_eval_sec = coerce_optional_scalar(progress.get("last_completed_eval_sec"), float)
    last_completed_eval_index = progress_eval_index

    if artifacts:
        best_artifact = select_best_eval(artifacts)
        last_artifact = select_last_completed_eval(artifacts)
        last_completed_eval_sec = coerce_optional_scalar(last_artifact.get("elapsed_time_sec"), float)
        last_completed_eval_index = coerce_optional_scalar(last_artifact.get("eval_index"), int)
        metrics.update(dict(best_artifact["metrics"]))
        metrics.update(
            {
                "best_accuracy": best_artifact["metrics"]["accuracy"],
                "time_to_best_eval_sec": coerce_optional_scalar(
                    best_artifact.get("elapsed_time_sec"),
                    float,
                ),
                "best_eval_index": coerce_optional_scalar(best_artifact.get("eval_index"), int),
                "best_eval_epoch": coerce_optional_scalar(best_artifact.get("epoch"), int),
            }
        )

    # Keep the last completed evaluation metadata in scalar form for the dashboard.
    if last_completed_eval_sec is not None:
        metrics["last_completed_eval_sec"] = float(last_completed_eval_sec)
    if last_completed_eval_index is not None:
        metrics["last_completed_eval_index"] = int(last_completed_eval_index)

    apply_debug_metrics(metrics, debug_payload)
    return metrics if metrics else None
