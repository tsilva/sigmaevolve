from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from sigmaevolve.core import (
    DEFAULT_TRIAL_HARD_TIMEOUT_SEC,
    OUTCOME_CRASHED,
    OUTCOME_EVAL_FAILED,
    OUTCOME_SUCCEEDED,
    OUTCOME_TIMEOUT,
    compute_classification_metrics,
    compute_score,
)
from sigmaevolve.env import load_env_file

WANDB_ENV_KEYS = (
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_BASE_URL",
)
_DISALLOWED_WANDB_MODES = {"disabled", "dryrun", "offline"}


def collect_wandb_env() -> dict[str, str]:
    # Capture only the non-empty WandB variables that are already present.
    collected: dict[str, str] = {}
    for key in WANDB_ENV_KEYS:
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            collected[key] = value
    return collected


def apply_wandb_env(overrides: dict[str, str] | None) -> None:
    if not overrides:
        return
    for key in WANDB_ENV_KEYS:
        value = overrides.get(key)
        if isinstance(value, str) and value.strip():
            os.environ[key] = value


def _import_wandb():
    try:
        return importlib.import_module("wandb")
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases support requires the 'wandb' package."
        ) from exc


def _env_first(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _wandb_metric_aliases(metrics: dict[str, Any] | None) -> dict[str, Any]:
    # Mirror canonical metrics under the field names expected by WandB dashboards.
    payload = dict(metrics or {})
    train_loss = payload.get("train_loss")
    train_acc = payload.get("train_acc")
    val_loss = payload.get("val_loss")
    val_acc = payload.get("val_acc")

    # Fall back to accuracy when the canonical validation metric is absent.
    if val_acc is None:
        val_acc = payload.get("accuracy")

    if train_loss is not None:
        payload["train/loss"] = train_loss
    if train_acc is not None:
        payload["train/acc"] = train_acc
    if val_loss is not None:
        payload["val/loss"] = val_loss

    if val_acc is not None:
        payload["val/acc"] = val_acc
    return payload


@dataclass(frozen=True)
class WandbSettings:
    api_key: str
    project: str
    entity: str | None
    base_url: str | None


def resolve_wandb_settings() -> WandbSettings:
    mode = _env_first("WANDB_MODE")

    # Reject local-only WandB modes that would block remote syncing.
    if mode is not None and mode.lower() in _DISALLOWED_WANDB_MODES:
        raise RuntimeError(
            "WANDB_MODE must allow remote sync; offline and disabled modes are not supported."
        )

    api_key = _env_first("WANDB_API_KEY")

    # Require an API key before attempting to initialize the run logger.
    if api_key is None:
        raise RuntimeError(
            "WANDB_API_KEY is required to log SigmaEvolve runs to Weights & Biases."
        )

    project = _env_first("WANDB_PROJECT", default="sigmaevolve") or "sigmaevolve"
    entity = _env_first("WANDB_ENTITY")
    base_url = _env_first("WANDB_BASE_URL")

    return WandbSettings(
        api_key=api_key,
        project=project,
        entity=entity,
        base_url=base_url,
    )


class WandbRunLogger:
    def __init__(
        self,
        *,
        repository,
        trial,
        track,
        manifest,
        runner_id: str,
    ) -> None:
        self.repository = repository
        self.trial = trial
        self.track = track
        self.manifest = manifest
        self.runner_id = runner_id
        self.step = 0

        wandb = _import_wandb()
        settings = resolve_wandb_settings()

        # Forward a custom base URL when the deployment points WandB at a mirror.
        if settings.base_url:
            os.environ["WANDB_BASE_URL"] = settings.base_url
        wandb.login(key=settings.api_key, relogin=True)

        wandb_config = {
            "sigmaevolve": {
                "trial_id": trial.trial_id,
                "track_id": track.track_id,
                "dataset_id": track.dataset_id,
                "runner_id": runner_id,
                "script_hash": trial.script_hash,
                "dispatch_attempts": trial.dispatch_attempts,
                "policy": dict(track.policy_json),
                "provenance": dict(trial.provenance_json),
                "dataset_metadata": dict(manifest.metadata),
            }
        }
        wandb_tags = [
            "sigmaevolve",
            f"track:{track.track_id}",
            f"dataset:{track.dataset_id}",
        ]
        run = wandb.init(
            project=settings.project,
            entity=settings.entity,
            job_type="trial",
            name=f"{track.track_id}:{trial.trial_id}",
            config=wandb_config,
            tags=wandb_tags,
        )
        self.run = run
        wandb_metadata = {
            "project": settings.project,
            "entity": getattr(run, "entity", None) or settings.entity,
            "run_id": getattr(run, "id", None),
            "run_name": getattr(run, "name", None),
            "run_url": getattr(run, "url", None),
        }
        self.repository.record_trial_wandb_metadata(
            trial.trial_id,
            wandb_metadata,
        )

    def log_metrics(self, metrics: dict[str, Any], *, state: str) -> None:
        payload = _wandb_metric_aliases(metrics)
        payload["trial_state"] = state
        self.step += 1
        self.run.log(payload, step=self.step)

    def finish(
        self,
        *,
        outcome_reason: str,
        metrics: dict[str, Any] | None,
        error_info: dict[str, Any] | None,
    ) -> None:
        # Build the terminal log entry in the same field order the dashboards expect.
        score = float(compute_score(metrics))
        payload = {
            "trial_state": "terminal",
            "outcome_reason": outcome_reason,
            "score": score,
        }
        if metrics:
            payload.update(_wandb_metric_aliases(metrics))
        self.step += 1
        self.run.log(payload, step=self.step)

        self.run.summary["trial_id"] = self.trial.trial_id
        self.run.summary["track_id"] = self.track.track_id
        self.run.summary["dataset_id"] = self.track.dataset_id
        self.run.summary["runner_id"] = self.runner_id
        self.run.summary["outcome_reason"] = outcome_reason
        self.run.summary["score"] = score

        # Copy metric fields into the summary so the final run page stays searchable.
        if metrics:
            for key, value in _wandb_metric_aliases(metrics).items():
                self.run.summary[key] = value

        # Retain structured error details only when the runner captured them.
        if error_info:
            reason = error_info.get("reason")
            detail = error_info.get("detail")
            if isinstance(reason, str) and reason:
                self.run.summary["error_reason"] = reason
            if isinstance(detail, str) and detail:
                self.run.summary["error_detail"] = detail

        exit_code = 0 if outcome_reason in {"succeeded", "timeout"} else 1
        self.run.finish(exit_code=exit_code)


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


def _read_eval_metadata(payload: dict[str, Any], key: str, cast) -> Any | None:
    if key not in payload:
        return None
    return coerce_optional_scalar(payload[key], cast)


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
            eval_index = _read_eval_metadata(payload, "eval_index", int)
            elapsed_time_sec = _read_eval_metadata(payload, "elapsed_time_sec", float)
            epoch = _read_eval_metadata(payload, "epoch", int)

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


def _best_eval_sort_key(artifact: dict[str, Any]) -> tuple[float, float, int]:
    accuracy = float(artifact["metrics"]["accuracy"])
    elapsed_time_sec = artifact.get("elapsed_time_sec")
    eval_index = artifact.get("eval_index")
    return (
        -accuracy,
        float(elapsed_time_sec) if elapsed_time_sec is not None else float("inf"),
        int(eval_index) if eval_index is not None else np.iinfo(np.int64).max,
    )


def _last_completed_eval_sort_key(artifact: dict[str, Any]) -> tuple[float, int]:
    elapsed_time_sec = artifact.get("elapsed_time_sec")
    eval_index = artifact.get("eval_index")
    return (
        float(elapsed_time_sec) if elapsed_time_sec is not None else -1.0,
        int(eval_index) if eval_index is not None else -1,
    )


def select_best_eval(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return min(artifacts, key=_best_eval_sort_key)


def select_last_completed_eval(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return max(artifacts, key=_last_completed_eval_sort_key)


def apply_debug_metrics(
    metrics: dict[str, Any], debug_payload: dict[str, Any] | None
) -> None:
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

    # Prefer the progress snapshot when the artifact metadata omitted a timestamp.
    if last_completed_eval_sec is None and progress_payload:
        last_completed_eval_sec = progress_payload.get("last_completed_eval_sec")

    time_to_best_eval_sec = best_artifact.get("elapsed_time_sec")
    time_since_last_eval_sec = None

    # Measure the gap since the last completed eval when a timestamp is available.
    if last_completed_eval_sec is not None:
        time_since_last_eval_sec = max(
            0.0, float(process_elapsed_sec) - float(last_completed_eval_sec)
        )

    last_phase = None

    # Preserve the last reported phase for downstream timeout classification.
    if progress_payload:
        last_phase = progress_payload.get("phase") or progress_payload.get(
            "current_phase"
        )

    # Flag timeouts that ended with a measurable amount of unevaluated work.
    is_timed_out = timed_out
    has_last_eval_gap = (
        time_since_last_eval_sec is not None and time_since_last_eval_sec > 0.05
    )
    is_training_phase = last_phase in {None, "train"}
    had_unscored_work_at_timeout = bool(
        is_timed_out and has_last_eval_gap and is_training_phase
    )

    # Merge the best evaluation metrics with run-level diagnostic fields.
    best_metrics = dict(best_artifact["metrics"])
    metrics_payload = {
        "best_accuracy": best_metrics["accuracy"],
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
    metrics = dict(best_metrics)
    metrics.update(metrics_payload)

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

    # Surface the most recent phase when the runner reported one explicitly.
    if last_phase is not None:
        metrics["last_phase"] = last_phase

    progress_eval_index = coerce_optional_scalar(progress.get("eval_index"), int)
    debug_eval_count = coerce_optional_scalar(
        (debug_payload or {}).get("eval_count"), int
    )
    eval_count = max(len(artifacts), progress_eval_index or 0, debug_eval_count or 0)
    metrics["eval_count"] = eval_count

    # Reconcile runner progress with the most recent persisted evaluation artifact.
    last_completed_eval_sec = coerce_optional_scalar(
        progress.get("last_completed_eval_sec"), float
    )
    last_completed_eval_index = progress_eval_index

    # Use persisted artifacts when they exist because they carry the authoritative metrics.
    if artifacts:
        best_artifact = select_best_eval(artifacts)
        last_artifact = select_last_completed_eval(artifacts)
        last_completed_eval_sec = coerce_optional_scalar(
            last_artifact.get("elapsed_time_sec"), float
        )
        last_completed_eval_index = coerce_optional_scalar(
            last_artifact.get("eval_index"), int
        )
        best_metrics = dict(best_artifact["metrics"])
        metrics.update(best_metrics)
        best_metrics_payload = {
            "best_accuracy": best_metrics["accuracy"],
            "time_to_best_eval_sec": coerce_optional_scalar(
                best_artifact.get("elapsed_time_sec"), float
            ),
            "best_eval_index": coerce_optional_scalar(
                best_artifact.get("eval_index"), int
            ),
            "best_eval_epoch": coerce_optional_scalar(best_artifact.get("epoch"), int),
        }
        metrics.update(best_metrics_payload)

    # Keep the last completed evaluation metadata in scalar form for the dashboard.
    if last_completed_eval_sec is not None:
        metrics["last_completed_eval_sec"] = float(last_completed_eval_sec)

    # Preserve the evaluation index as an integer when progress or artifacts provide it.
    if last_completed_eval_index is not None:
        metrics["last_completed_eval_index"] = int(last_completed_eval_index)

    apply_debug_metrics(metrics, debug_payload)
    return metrics if metrics else None


logger = logging.getLogger(__name__)
ACTIVE_METRICS_INTERVAL_SEC = 1.0


def _coerce_text(value: Any) -> str | None:
    # Normalize subprocess output chunks into UTF-8 text.
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


@dataclass(frozen=True)
class _StreamedProcessResult:
    returncode: int
    stdout: str | None
    stderr: str | None
    timed_out: bool


def _stream_pipe(pipe, sink, chunks: list[str]) -> None:
    # Forward streamed output to the parent process while capturing a copy.
    try:
        for chunk in iter(pipe.readline, ""):
            if not chunk:
                break
            sink.write(chunk)
            sink.flush()
            chunks.append(chunk)
    finally:
        pipe.close()


def _run_streamed_subprocess(
    command: list[str], timeout: float
) -> _StreamedProcessResult:
    # Launch the child process with unbuffered Python output for live streaming.
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=child_env,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    # Stream stdout and stderr concurrently while the process is running.
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, sys.stdout, stdout_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr, sys.stderr, stderr_chunks),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    # Kill the process on timeout and preserve the final exit status.
    timed_out = False
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        returncode = process.wait()

    # Join the stream threads and collapse the captured output into strings.
    stdout_thread.join()
    stderr_thread.join()
    stdout = "".join(stdout_chunks) or None
    stderr = "".join(stderr_chunks) or None
    return _StreamedProcessResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )


class RunnerService:
    def __init__(
        self,
        repository,
        dataset_manager,
        python_executable: str | None = None,
        hard_timeout_sec: float = DEFAULT_TRIAL_HARD_TIMEOUT_SEC,
    ) -> None:
        # Persist the dependencies and default interpreter used by runner workers.
        self.repository = repository
        self.dataset_manager = dataset_manager
        self.python_executable = python_executable or sys.executable
        self.hard_timeout_sec = float(hard_timeout_sec)

    def _start_heartbeat(
        self,
        trial_id: str,
        runner_id: str,
        interval_sec: int,
    ) -> tuple[threading.Event, threading.Thread]:
        stop_event = threading.Event()

        def loop() -> None:
            # Keep the active trial heartbeat alive until the stop event is set.
            while not stop_event.wait(interval_sec):
                try:
                    self.repository.heartbeat_trial(
                        trial_id, runner_id, {"status": "alive"}
                    )
                except Exception:
                    # Transient database disconnects should not permanently kill the
                    # heartbeat loop for a still-running worker.
                    logger.warning(
                        "Heartbeat update failed for trial %s runner %s; retrying.",
                        trial_id,
                        runner_id,
                        exc_info=True,
                    )
                    self._dispose_repository_engine()

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        return stop_event, thread

    def _dispose_repository_engine(self) -> None:
        # Dispose the repository engine when the worker suspects a broken connection.
        engine = getattr(self.repository, "engine", None)
        dispose = getattr(engine, "dispose", None)

        # Only call dispose on objects that actually expose it.
        if callable(dispose):
            try:
                dispose()
            except Exception:
                logger.warning(
                    "Disposing repository engine after failure also failed.",
                    exc_info=True,
                )

    def _read_json_object(self, path: Path) -> dict[str, Any] | None:
        # Return None for missing or malformed debug payloads.
        if not path.exists():
            return None

        # Read the file as JSON only after confirming it still exists.
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

        # Keep only object-shaped payloads so callers can use dict access safely.
        if not isinstance(payload, dict):
            return None
        return payload

    def _read_progress(self, progress_path: Path) -> dict[str, Any] | None:
        return self._read_json_object(progress_path)

    def _read_debug_payload(self, debug_path: Path) -> dict[str, Any] | None:
        return self._read_json_object(debug_path)

    def _load_eval_artifacts(
        self,
        eval_dir: Path,
        labels_path: str,
    ) -> list[dict[str, Any]]:
        return load_eval_artifacts(
            eval_dir=eval_dir,
            labels_path=labels_path,
        )

    def _select_best_eval(self, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        return select_best_eval(artifacts)

    def _select_last_completed_eval(
        self, artifacts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return select_last_completed_eval(artifacts)

    def _build_metrics_payload(
        self,
        artifacts: list[dict[str, Any]],
        progress_payload: dict[str, Any] | None,
        process_elapsed_sec: float,
        timed_out: bool,
        debug_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return build_final_metrics_payload(
            artifacts=artifacts,
            progress_payload=progress_payload,
            process_elapsed_sec=process_elapsed_sec,
            timed_out=timed_out,
            debug_payload=debug_payload,
        )

    def _apply_debug_metrics(
        self,
        metrics: dict[str, Any],
        debug_payload: dict[str, Any] | None,
    ) -> None:
        apply_debug_metrics(metrics, debug_payload)

    def _build_active_metrics_payload(
        self,
        artifacts: list[dict[str, Any]],
        progress_payload: dict[str, Any] | None,
        process_elapsed_sec: float,
        debug_payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return build_active_metrics_payload(
            artifacts=artifacts,
            progress_payload=progress_payload,
            process_elapsed_sec=process_elapsed_sec,
            debug_payload=debug_payload,
        )

    def _collect_active_metrics_payload(
        self,
        *,
        eval_dir: Path,
        progress_path: Path,
        debug_path: Path,
        labels_path: str,
        started_at: float,
    ) -> dict[str, Any] | None:
        # Load the latest progress, debug, and eval artifacts before building metrics.
        progress_payload = self._read_progress(progress_path)
        debug_payload = self._read_debug_payload(debug_path)
        artifacts: list[dict[str, Any]] = []
        try:
            artifacts = self._load_eval_artifacts(
                eval_dir=eval_dir,
                labels_path=labels_path,
            )
        except Exception:
            logger.warning(
                "Active metrics scan failed; continuing without eval artifacts.",
                exc_info=True,
            )

        # Skip reporter updates until there is at least one source of metrics.
        if not progress_payload and not debug_payload and not artifacts:
            return None

        # Delegate the actual payload shaping to the shared runner-metrics helper.
        return self._build_active_metrics_payload(
            artifacts=artifacts,
            progress_payload=progress_payload,
            process_elapsed_sec=time.monotonic() - started_at,
            debug_payload=debug_payload,
        )

    def _start_active_metrics_reporter(
        self,
        *,
        trial_id: str,
        runner_id: str,
        progress_path: Path,
        debug_path: Path,
        eval_dir: Path,
        labels_path: str,
        started_at: float,
        wandb_run_logger: WandbRunLogger | None = None,
        interval_sec: float = ACTIVE_METRICS_INTERVAL_SEC,
    ) -> tuple[threading.Event, threading.Thread]:
        stop_event = threading.Event()

        def report_once(last_metrics: dict[str, Any] | None) -> dict[str, Any] | None:
            metrics = self._collect_active_metrics_payload(
                eval_dir=eval_dir,
                progress_path=progress_path,
                debug_path=debug_path,
                labels_path=labels_path,
                started_at=started_at,
            )

            # Avoid redundant writes when the reporter sees the same payload twice.
            if metrics is None or metrics == last_metrics:
                return last_metrics
            self.repository.update_active_trial_metrics(
                trial_id=trial_id, runner_id=runner_id, metrics=metrics
            )
            self._log_wandb_metrics(wandb_run_logger, metrics, state="active")
            return metrics

        def loop() -> None:
            last_metrics: dict[str, Any] | None = None
            while not stop_event.is_set():
                try:
                    last_metrics = report_once(last_metrics)
                except Exception:
                    logger.warning(
                        "Active metrics update failed for trial %s runner %s; retrying.",
                        trial_id,
                        runner_id,
                        exc_info=True,
                    )
                    self._dispose_repository_engine()
                if stop_event.wait(interval_sec):
                    break

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        return stop_event, thread

    def _log_wandb_metrics(
        self,
        wandb_run_logger: WandbRunLogger | None,
        metrics: dict[str, Any] | None,
        *,
        state: str,
    ) -> None:

        # Skip WandB logging when there is no logger or no payload to record.
        if wandb_run_logger is None or metrics is None:
            return
        try:
            wandb_run_logger.log_metrics(metrics, state=state)
        except Exception:
            # Keep the trial running even if the telemetry backend is transiently unavailable.
            logger.warning(
                "W&B metrics update failed for trial %s.",
                wandb_run_logger.trial.trial_id,
                exc_info=True,
            )

    def _finalize_trial(
        self,
        *,
        trial_id: str,
        runner_id: str | None,
        outcome_reason: str,
        metrics: dict[str, Any] | None,
        error_info: dict[str, Any] | None,
        wandb_run_logger: WandbRunLogger | None,
    ) -> None:
        self.repository.finalize_trial(
            trial_id=trial_id,
            runner_id=runner_id,
            outcome_reason=outcome_reason,
            metrics=metrics,
            error_info=error_info,
        )
        if wandb_run_logger is None:
            return
        try:
            wandb_run_logger.finish(
                outcome_reason=outcome_reason,
                metrics=metrics,
                error_info=error_info,
            )
        except Exception:
            logger.warning(
                "W&B run finalization failed for trial %s.", trial_id, exc_info=True
            )

    def run_reserved_trial(
        self, trial_id: str, dispatch_token: str, runner_id: str
    ) -> None:
        logger.info("Claiming reserved trial %s with runner %s.", trial_id, runner_id)
        trial = self.repository.claim_trial(trial_id, dispatch_token, runner_id)

        # Stop immediately when the reservation has already been lost to another runner.
        if trial is None:
            logger.info(
                "Skipping trial %s because the reservation could not be claimed.",
                trial_id,
            )
            return
        logger.info("Claimed trial %s on track %s.", trial.trial_id, trial.track_id)
        track = self.repository.get_track(trial.track_id)

        # Fail fast when the track disappeared between reservation and execution.
        if track is None:
            raise RuntimeError(f"Track not found for trial {trial.trial_id}")
        policy = track.policy_json
        manifest = self.dataset_manager.verify(track.dataset_id)
        load_env_file()
        logger.info(
            "Verified dataset %s for trial %s.", track.dataset_id, trial.trial_id
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"sigmaevolve_{trial.trial_id}_"
            ) as temp_dir:
                temp_path = Path(temp_dir)
                train_script_path = temp_path / "train.py"
                config_path = temp_path / "run_config.json"
                progress_path = temp_path / "progress.json"
                eval_dir = temp_path / "evals"
                debug_path = temp_path / "debug.json"
                eval_dir.mkdir(parents=True, exist_ok=True)
                train_script_path.write_text(trial.source)
                run_config_payload = {
                    "train_split_path": manifest.train_split_path,
                    "validation_split_path": manifest.validation_split_path,
                    "validation_labels_path": manifest.validation_labels_path,
                    "epochs": int(policy["epochs"]),
                    "hard_timeout_sec": self.hard_timeout_sec,
                    "random_seed": 1234,
                    "progress_path": str(progress_path),
                    "eval_dir": str(eval_dir),
                    "debug_output_path": str(debug_path),
                    "dataset_metadata": manifest.metadata,
                }
                config_path.write_text(json.dumps(run_config_payload, sort_keys=True))

                def finalize_outcome(
                    outcome_reason: str,
                    *,
                    metrics: dict[str, Any] | None,
                    error_info: dict[str, Any] | None,
                ) -> None:
                    self._finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=outcome_reason,
                        metrics=metrics,
                        error_info=error_info,
                        wandb_run_logger=wandb_run_logger,
                    )
                    logger.info(
                        "Finalized trial %s with outcome=%s.",
                        trial.trial_id,
                        outcome_reason,
                    )

                try:
                    wandb_run_logger = WandbRunLogger(
                        repository=self.repository,
                        trial=trial,
                        track=track,
                        manifest=manifest,
                        runner_id=runner_id,
                    )
                except Exception as exc:
                    self._finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=OUTCOME_CRASHED,
                        metrics=None,
                        error_info={
                            "reason": "wandb_init_failed",
                            "detail": str(exc),
                        },
                        wandb_run_logger=None,
                    )
                    logger.info(
                        "Finalized trial %s with outcome=%s.",
                        trial.trial_id,
                        OUTCOME_CRASHED,
                    )
                    return
                heartbeat_stop, heartbeat_thread = self._start_heartbeat(
                    trial_id=trial.trial_id,
                    runner_id=runner_id,
                    interval_sec=int(policy["heartbeat_interval_sec"]),
                )
                metrics_stop = threading.Event()
                metrics_thread: threading.Thread | None = None
                command = [
                    self.python_executable,
                    "-u",
                    str(train_script_path),
                    "--config",
                    str(config_path),
                ]
                timed_out = False
                stdout: str | None = None
                stderr: str | None = None
                started_at = time.monotonic()
                logger.info(
                    "Starting child process for trial %s: %s",
                    trial.trial_id,
                    " ".join(command),
                )
                metrics_stop, metrics_thread = self._start_active_metrics_reporter(
                    trial_id=trial.trial_id,
                    runner_id=runner_id,
                    progress_path=progress_path,
                    debug_path=debug_path,
                    eval_dir=eval_dir,
                    labels_path=manifest.validation_labels_path,
                    started_at=started_at,
                    wandb_run_logger=wandb_run_logger,
                )
                completed = _run_streamed_subprocess(
                    command, timeout=self.hard_timeout_sec
                )
                metrics_stop.set()
                if metrics_thread is not None:
                    metrics_thread.join(timeout=1.0)
                timed_out = completed.timed_out
                stdout = _coerce_text(completed.stdout)
                stderr = _coerce_text(completed.stderr)
                process_elapsed_sec = time.monotonic() - started_at
                logger.info(
                    "Child process finished for trial %s with returncode=%s timed_out=%s elapsed=%.2fs.",
                    trial.trial_id,
                    completed.returncode,
                    timed_out,
                    process_elapsed_sec,
                )
                progress_payload = self._read_progress(progress_path)
                debug_payload = self._read_debug_payload(debug_path)
                timed_out = bool(timed_out or (debug_payload or {}).get("timed_out"))

                # Classify a nonzero exit before looking at evaluation artifacts.
                if completed.returncode != 0 and not timed_out:
                    failure_outcome = (debug_payload or {}).get("failure_outcome")

                    # Preserve explicit contract failures as eval failures instead of crashes.
                    if failure_outcome == OUTCOME_EVAL_FAILED:
                        error_info = {
                            "reason": (debug_payload or {}).get("failure_reason")
                            or "train_script_contract_violation",
                            "detail": (debug_payload or {}).get("detail"),
                            "stdout": stdout,
                            "stderr": stderr,
                            "returncode": completed.returncode,
                            "timed_out": timed_out,
                            "progress": progress_payload,
                        }
                        finalize_outcome(
                            OUTCOME_EVAL_FAILED,
                            metrics=None,
                            error_info=error_info,
                        )
                        return
                    error_info = {
                        "stdout": stdout,
                        "stderr": stderr,
                        "returncode": completed.returncode,
                        "timed_out": timed_out,
                        "progress": progress_payload,
                    }
                    finalize_outcome(
                        OUTCOME_CRASHED,
                        metrics=None,
                        error_info=error_info,
                    )
                    return

                try:
                    artifacts = self._load_eval_artifacts(
                        eval_dir=eval_dir,
                        labels_path=manifest.validation_labels_path,
                    )
                except Exception as exc:
                    error_info = {
                        "reason": "prediction_load_failed",
                        "detail": str(exc),
                        "timed_out": timed_out,
                        "stdout": stdout,
                        "stderr": stderr,
                        "progress": progress_payload,
                    }
                    finalize_outcome(
                        OUTCOME_EVAL_FAILED,
                        metrics=None,
                        error_info=error_info,
                    )
                    return

                # Handle the no-artifact case separately so timeout and missing-output cases stay distinct.
                if not artifacts:
                    outcome_reason = (
                        OUTCOME_TIMEOUT if timed_out else OUTCOME_EVAL_FAILED
                    )
                    error_info: dict[str, Any] = {
                        "reason": "completed_evals_missing"
                        if timed_out
                        else "predictions_missing",
                        "timed_out": timed_out,
                        "stdout": stdout,
                        "stderr": stderr,
                        "progress": progress_payload,
                        "eval_dir": str(eval_dir),
                    }
                    finalize_outcome(
                        outcome_reason,
                        metrics=None,
                        error_info=error_info,
                    )
                    return

                metrics = self._build_metrics_payload(
                    artifacts=artifacts,
                    progress_payload=progress_payload,
                    process_elapsed_sec=process_elapsed_sec,
                    timed_out=timed_out,
                    debug_payload=debug_payload,
                )
                outcome_reason = OUTCOME_TIMEOUT if timed_out else OUTCOME_SUCCEEDED
                error_info = {
                    "stdout": stdout,
                    "stderr": stderr,
                    "debug": debug_payload,
                    "debug_output_path": str(debug_path),
                    "progress": progress_payload,
                    "eval_dir": str(eval_dir),
                    "eval_artifacts": [artifact["path"] for artifact in artifacts],
                    "timed_out": timed_out,
                }
                self._log_wandb_metrics(wandb_run_logger, metrics, state="completed")
                self._finalize_trial(
                    trial_id=trial.trial_id,
                    runner_id=runner_id,
                    outcome_reason=outcome_reason,
                    metrics=metrics,
                    error_info=error_info,
                    wandb_run_logger=wandb_run_logger,
                )
                logger.info(
                    "Finalized trial %s with outcome=%s score=%.6f accuracy=%s.",
                    trial.trial_id,
                    outcome_reason,
                    compute_score(metrics),
                    metrics.get("accuracy"),
                )
        finally:
            if "metrics_stop" in locals():
                metrics_stop.set()
            if "metrics_thread" in locals() and metrics_thread is not None:
                metrics_thread.join(timeout=1.0)
            if "heartbeat_stop" in locals():
                heartbeat_stop.set()
            if "heartbeat_thread" in locals():
                heartbeat_thread.join(timeout=1.0)
            time.sleep(0.01)


logger = logging.getLogger(__name__)


class StrategyContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class StrategyContext:
    train_features: np.ndarray
    train_labels: np.ndarray
    validation_features: np.ndarray
    dataset_metadata: dict[str, Any]
    random_seed: int
    device: str
    epoch_index: int
    num_epochs: int
    epochs_remaining: int
    budget_sec: float
    remaining_budget_sec: float
    max_eval_gap_sec: float
    window_index: int


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, sort_keys=True))
    temp_path.replace(path)


def write_eval_atomic(
    eval_dir: Path,
    eval_index: int,
    predictions: np.ndarray,
    elapsed_time_sec: float,
    epoch: int | None,
) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    final_path = eval_dir / f"eval_{eval_index:04d}.npz"
    temp_path = eval_dir / f".eval_{eval_index:04d}.tmp.npz"
    payload: dict[str, Any] = {
        "predictions": np.asarray(predictions, dtype=np.int64),
        "eval_index": np.array(eval_index, dtype=np.int64),
        "elapsed_time_sec": np.array(elapsed_time_sec, dtype=np.float64),
    }

    # Store the epoch only when the harness has already advanced past the initial state.
    if epoch is not None:
        payload["epoch"] = np.array(epoch, dtype=np.int64)
    np.savez(temp_path, **payload)
    temp_path.replace(final_path)
    return final_path


def _load_strategy_module(strategy_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sigmaevolve_candidate_strategy", strategy_path
    )

    # Reject incomplete module specs before trying to execute the candidate.
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load strategy module from {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_strategy(strategy_path: Path) -> tuple[Any, Any, Any]:
    module = _load_strategy_module(strategy_path)
    initialize = getattr(module, "initialize", None)
    train_window = getattr(module, "train_window", None)
    predict_validation = getattr(module, "predict_validation", None)

    missing = [
        name
        for name, value in (
            ("initialize", initialize),
            ("train_window", train_window),
            ("predict_validation", predict_validation),
        )
        if not callable(value)
    ]

    # Surface the missing entry points together so strategy authors can fix them in one pass.
    if missing:
        raise StrategyContractError(
            f"Strategy is missing required callable exports: {', '.join(missing)}"
        )

    return initialize, train_window, predict_validation


def _normalize_predictions(
    raw_predictions: Any,
    *,
    num_examples: int,
    num_classes: int | None,
) -> np.ndarray:
    try:
        import torch  # noqa: PLC0415
    except (
        ImportError
    ):  # pragma: no cover - torch is expected but keep loader tolerant.
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(raw_predictions, torch.Tensor):
        array = raw_predictions.detach().cpu().numpy()
    else:
        array = np.asarray(raw_predictions)

    if array.ndim == 0:
        raise StrategyContractError(
            "predict_validation must return one prediction per validation example."
        )

    if array.shape[0] != num_examples:
        raise StrategyContractError(
            f"predict_validation returned {array.shape[0]} predictions for {num_examples} validation examples."
        )

    if array.ndim == 1:
        # Accept 1D float outputs only for binary tasks where thresholding is meaningful.
        if np.issubdtype(array.dtype, np.floating):
            if num_classes == 2:
                finite = array[np.isfinite(array)]
                has_bounded_probabilities = (
                    finite.size
                    and float(finite.min()) >= 0.0
                    and float(finite.max()) <= 1.0
                )
                if has_bounded_probabilities:
                    return (array >= 0.5).astype(np.int64)
                return (array >= 0.0).astype(np.int64)

            raise StrategyContractError(
                "predict_validation returned a 1D float array for a non-binary task; return class ids or logits."
            )
        return array.astype(np.int64)

    reshaped = array.reshape(num_examples, -1)
    if reshaped.shape[1] <= 1:
        return reshaped.reshape(num_examples).astype(np.int64)
    return reshaped.argmax(axis=1).astype(np.int64)


def _seed_everything(seed: int) -> str:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"

    # Seed CUDA explicitly when the runtime exposes it.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        return "cuda"
    return "cpu"


def _read_split(path: str) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    payload = np.load(path)
    features = payload["features"].astype(np.float32)

    # Preserve label arrays only when the split actually includes them.
    if "labels" in payload:
        return features, payload["labels"].astype(np.int64)
    return features


def _build_context(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    dataset_metadata: dict[str, Any],
    random_seed: int,
    device: str,
    num_epochs: int,
    epoch_index: int,
    hard_timeout_sec: float,
    start_time: float,
) -> StrategyContext:
    elapsed = time.monotonic() - start_time
    remaining = max(0.0, float(hard_timeout_sec) - float(elapsed))
    epochs_remaining = max(0, int(num_epochs) - int(epoch_index))
    return StrategyContext(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        dataset_metadata=dict(dataset_metadata),
        random_seed=int(random_seed),
        device=device,
        epoch_index=epoch_index,
        num_epochs=int(num_epochs),
        epochs_remaining=epochs_remaining,
        budget_sec=float(hard_timeout_sec),
        remaining_budget_sec=remaining,
        max_eval_gap_sec=float(hard_timeout_sec),
        window_index=epoch_index,
    )


def _write_progress(
    progress_path: Path,
    *,
    phase: str,
    elapsed_time_sec: float,
    last_completed_eval_sec: float | None,
    eval_index: int,
    epoch_index: int,
) -> None:
    write_json_atomic(
        progress_path,
        {
            "phase": phase,
            "elapsed_time_sec": float(elapsed_time_sec),
            "last_completed_eval_sec": last_completed_eval_sec,
            "eval_index": eval_index,
            "epoch_index": epoch_index,
        },
    )


def _run_harness(config: dict[str, Any]) -> int:
    strategy_path = Path(config["strategy_path"])
    progress_path = Path(config["progress_path"])
    eval_dir = Path(config["eval_dir"])
    debug_output_path = Path(config["debug_output_path"])
    num_epochs = int(config["epochs"])
    hard_timeout_sec = float(config["hard_timeout_sec"])
    random_seed = int(config["random_seed"])
    dataset_metadata = dict(config.get("dataset_metadata") or {})

    train_features, train_labels = _read_split(config["train_split_path"])
    validation_features = _read_split(config["validation_split_path"])

    # Validate the split shapes before entering the training loop.
    if not isinstance(train_features, np.ndarray) or not isinstance(
        train_labels, np.ndarray
    ):
        raise RuntimeError("Training split is invalid.")
    if not isinstance(validation_features, np.ndarray):
        raise RuntimeError("Validation split is invalid.")

    device = _seed_everything(random_seed)
    start_time = time.monotonic()
    eval_index = 0
    last_completed_eval_sec: float | None = None
    debug_payload: dict[str, Any] = {"timed_out": False, "eval_count": 0}
    num_classes = (
        int(dataset_metadata["num_classes"])
        if "num_classes" in dataset_metadata
        else None
    )

    def _build_epoch_context(epoch_index: int) -> StrategyContext:
        return _build_context(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            dataset_metadata=dataset_metadata,
            random_seed=random_seed,
            device=device,
            num_epochs=num_epochs,
            epoch_index=epoch_index,
            hard_timeout_sec=hard_timeout_sec,
            start_time=start_time,
        )

    try:
        # Load the candidate strategy and validate its required entry points first.
        initialize, train_window, predict_validation = load_strategy(strategy_path)
        init_ctx = _build_epoch_context(0)
        state = initialize(init_ctx)

        # Reject strategies that return an unexpected state container.
        if not isinstance(state, dict):
            raise StrategyContractError("initialize must return a dict state object.")

        # Write the initial progress snapshot before the epoch loop begins.
        _write_progress(
            progress_path,
            phase="train",
            elapsed_time_sec=0.0,
            last_completed_eval_sec=None,
            eval_index=eval_index,
            epoch_index=0,
        )

        # Alternate between train and eval work while keeping progress updates durable.
        for epoch_index in range(num_epochs):
            elapsed_before = time.monotonic() - start_time
            train_ctx = _build_epoch_context(epoch_index)
            _write_progress(
                progress_path,
                phase="train",
                elapsed_time_sec=elapsed_before,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                epoch_index=epoch_index,
            )

            train_window(train_ctx, state)

            predict_ctx = _build_epoch_context(epoch_index)
            _write_progress(
                progress_path,
                phase="eval",
                elapsed_time_sec=time.monotonic() - start_time,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                epoch_index=epoch_index,
            )
            raw_predictions = predict_validation(predict_ctx, state)
            predictions = _normalize_predictions(
                raw_predictions,
                num_examples=int(validation_features.shape[0]),
                num_classes=num_classes,
            )
            eval_index += 1
            elapsed_after_eval = time.monotonic() - start_time
            write_eval_atomic(
                eval_dir,
                eval_index=eval_index,
                predictions=predictions,
                elapsed_time_sec=elapsed_after_eval,
                epoch=epoch_index + 1,
            )
            last_completed_eval_sec = elapsed_after_eval
            debug_payload["eval_count"] = eval_index
            _write_progress(
                progress_path,
                phase="train",
                elapsed_time_sec=elapsed_after_eval,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                epoch_index=epoch_index + 1,
            )

        # Finish with a terminal snapshot and the final debug payload.
        _write_progress(
            progress_path,
            phase="finished",
            elapsed_time_sec=time.monotonic() - start_time,
            last_completed_eval_sec=last_completed_eval_sec,
            eval_index=eval_index,
            epoch_index=num_epochs,
        )
        write_json_atomic(debug_output_path, debug_payload)
        return 0
    except StrategyContractError as exc:
        # Persist strategy contract failures in the debug payload for the parent process.
        debug_payload.update(
            {
                "failure_outcome": "eval_failed",
                "failure_reason": "strategy_contract_violation",
                "detail": str(exc),
            }
        )
        write_json_atomic(debug_output_path, debug_payload)
        logger.error("%s", exc)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    return _run_harness(config)


if __name__ == "__main__":
    raise SystemExit(main())
