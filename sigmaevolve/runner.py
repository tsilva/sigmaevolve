from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sigmaevolve.models import OUTCOME_CRASHED, OUTCOME_EVAL_FAILED, OUTCOME_SUCCEEDED, OUTCOME_TIMEOUT
from sigmaevolve.runtime_config import DEFAULT_TRIAL_HARD_TIMEOUT_SEC
from sigmaevolve.scoring import compute_classification_metrics, compute_score


logger = logging.getLogger(__name__)
ACTIVE_METRICS_INTERVAL_SEC = 1.0
DEBUG_METRIC_KEYS = (
    "early_stopped",
    "early_stop_epoch",
    "early_stopping_patience",
    "epochs_completed",
    "best_validation_accuracy_seen",
    "epochs_without_improvement",
)


def _coerce_optional_scalar(value: Any, cast) -> Any | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    try:
        return cast(scalar)
    except (TypeError, ValueError):
        return None


def _coerce_text(value: Any) -> str | None:
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
    try:
        for chunk in iter(pipe.readline, ""):
            if not chunk:
                break
            sink.write(chunk)
            sink.flush()
            chunks.append(chunk)
    finally:
        pipe.close()


def _run_streamed_subprocess(command: list[str], timeout: float) -> _StreamedProcessResult:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_thread = threading.Thread(target=_stream_pipe, args=(process.stdout, sys.stdout, stdout_chunks), daemon=True)
    stderr_thread = threading.Thread(target=_stream_pipe, args=(process.stderr, sys.stderr, stderr_chunks), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        returncode = process.wait()

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
        self.repository = repository
        self.dataset_manager = dataset_manager
        self.python_executable = python_executable or sys.executable
        self.hard_timeout_sec = float(hard_timeout_sec)

    def _start_heartbeat(self, trial_id: str, runner_id: str, interval_sec: int) -> tuple[threading.Event, threading.Thread]:
        stop_event = threading.Event()

        def loop() -> None:
            while not stop_event.wait(interval_sec):
                try:
                    self.repository.heartbeat_trial(trial_id, runner_id, {"status": "alive"})
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
        engine = getattr(self.repository, "engine", None)
        dispose = getattr(engine, "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                logger.warning("Disposing repository engine after failure also failed.", exc_info=True)

    def _read_progress(self, progress_path: Path) -> dict[str, Any] | None:
        if not progress_path.exists():
            return None
        try:
            payload = json.loads(progress_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _read_debug_payload(self, debug_path: Path) -> dict[str, Any] | None:
        if not debug_path.exists():
            return None
        try:
            payload = json.loads(debug_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _load_eval_artifacts(
        self,
        eval_dir: Path,
        labels_path: str,
        fallback_predictions_path: Path | None,
        fallback_elapsed_time_sec: float,
    ) -> list[dict[str, Any]]:
        labels = np.load(labels_path)
        artifacts: list[dict[str, Any]] = []
        for eval_path in sorted(eval_dir.glob("*.npz")):
            with np.load(eval_path) as payload:
                if "predictions" not in payload:
                    continue
                predictions = payload["predictions"]
                if predictions.ndim > 1:
                    predictions = predictions.argmax(axis=1)
                metrics = compute_classification_metrics(predictions.astype(int).tolist(), labels.astype(int).tolist())
                artifacts.append(
                    {
                        "path": str(eval_path),
                        "eval_index": _coerce_optional_scalar(payload["eval_index"], int) if "eval_index" in payload else None,
                        "elapsed_time_sec": _coerce_optional_scalar(payload["elapsed_time_sec"], float)
                        if "elapsed_time_sec" in payload
                        else None,
                        "epoch": _coerce_optional_scalar(payload["epoch"], int) if "epoch" in payload else None,
                        "metrics": metrics,
                    }
                )

        if not artifacts and fallback_predictions_path is not None and fallback_predictions_path.exists():
            with np.load(fallback_predictions_path) as payload:
                predictions = payload["predictions"]
                if predictions.ndim > 1:
                    predictions = predictions.argmax(axis=1)
            metrics = compute_classification_metrics(predictions.astype(int).tolist(), labels.astype(int).tolist())
            artifacts.append(
                {
                    "path": str(fallback_predictions_path),
                    "eval_index": 0,
                    "elapsed_time_sec": float(fallback_elapsed_time_sec),
                    "epoch": None,
                    "metrics": metrics,
                }
            )
        return artifacts

    def _select_best_eval(self, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        return min(
            artifacts,
            key=lambda artifact: (
                -float(artifact["metrics"]["accuracy"]),
                float(artifact.get("elapsed_time_sec") if artifact.get("elapsed_time_sec") is not None else float("inf")),
                int(artifact.get("eval_index") if artifact.get("eval_index") is not None else sys.maxsize),
            ),
        )

    def _select_last_completed_eval(self, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        return max(
            artifacts,
            key=lambda artifact: (
                float(artifact.get("elapsed_time_sec") if artifact.get("elapsed_time_sec") is not None else -1.0),
                int(artifact.get("eval_index") if artifact.get("eval_index") is not None else -1),
            ),
        )

    def _build_metrics_payload(
        self,
        artifacts: list[dict[str, Any]],
        progress_payload: dict[str, Any] | None,
        process_elapsed_sec: float,
        timed_out: bool,
        debug_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        best_artifact = self._select_best_eval(artifacts)
        last_artifact = self._select_last_completed_eval(artifacts)
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
        had_unscored_work_at_timeout = bool(
            timed_out
            and time_since_last_eval_sec is not None
            and time_since_last_eval_sec > 0.05
            and (last_phase in {None, "train"})
        )

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
        self._apply_debug_metrics(metrics, debug_payload)
        return metrics

    def _apply_debug_metrics(self, metrics: dict[str, Any], debug_payload: dict[str, Any] | None) -> None:
        if not debug_payload:
            return
        for key in DEBUG_METRIC_KEYS:
            if key in debug_payload:
                metrics[key] = debug_payload[key]

    def _build_active_metrics_payload(
        self,
        artifacts: list[dict[str, Any]],
        progress_payload: dict[str, Any] | None,
        process_elapsed_sec: float,
        debug_payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        progress = dict(progress_payload or {})
        metrics: dict[str, Any] = {
            "process_elapsed_sec": float(process_elapsed_sec),
        }

        last_phase = progress.get("phase") or progress.get("current_phase")
        if last_phase is not None:
            metrics["last_phase"] = last_phase

        progress_eval_index = _coerce_optional_scalar(progress.get("eval_index"), int)
        debug_eval_count = _coerce_optional_scalar((debug_payload or {}).get("eval_count"), int)
        eval_count = max(len(artifacts), progress_eval_index or 0, debug_eval_count or 0)
        metrics["eval_count"] = eval_count

        last_completed_eval_sec = _coerce_optional_scalar(progress.get("last_completed_eval_sec"), float)
        last_completed_eval_index = progress_eval_index

        if artifacts:
            best_artifact = self._select_best_eval(artifacts)
            last_artifact = self._select_last_completed_eval(artifacts)
            last_completed_eval_sec = _coerce_optional_scalar(last_artifact.get("elapsed_time_sec"), float)
            last_completed_eval_index = _coerce_optional_scalar(last_artifact.get("eval_index"), int)
            metrics.update(dict(best_artifact["metrics"]))
            metrics.update(
                {
                    "best_accuracy": best_artifact["metrics"]["accuracy"],
                    "time_to_best_eval_sec": _coerce_optional_scalar(best_artifact.get("elapsed_time_sec"), float),
                    "best_eval_index": _coerce_optional_scalar(best_artifact.get("eval_index"), int),
                    "best_eval_epoch": _coerce_optional_scalar(best_artifact.get("epoch"), int),
                }
            )

        if last_completed_eval_sec is not None:
            metrics["last_completed_eval_sec"] = float(last_completed_eval_sec)
        if last_completed_eval_index is not None:
            metrics["last_completed_eval_index"] = int(last_completed_eval_index)

        self._apply_debug_metrics(metrics, debug_payload)
        return metrics if metrics else None

    def _collect_active_metrics_payload(
        self,
        *,
        eval_dir: Path,
        progress_path: Path,
        debug_path: Path,
        labels_path: str,
        started_at: float,
    ) -> dict[str, Any] | None:
        progress_payload = self._read_progress(progress_path)
        debug_payload = self._read_debug_payload(debug_path)
        artifacts: list[dict[str, Any]] = []
        try:
            artifacts = self._load_eval_artifacts(
                eval_dir=eval_dir,
                labels_path=labels_path,
                fallback_predictions_path=None,
                fallback_elapsed_time_sec=0.0,
            )
        except Exception:
            logger.warning("Active metrics scan failed; continuing without eval artifacts.", exc_info=True)
        if not progress_payload and not debug_payload and not artifacts:
            return None
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
            if metrics is None or metrics == last_metrics:
                return last_metrics
            self.repository.update_active_trial_metrics(trial_id=trial_id, runner_id=runner_id, metrics=metrics)
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

    def run_reserved_trial(self, trial_id: str, dispatch_token: str, runner_id: str) -> None:
        trial = self.repository.claim_trial(trial_id, dispatch_token, runner_id)
        if trial is None:
            return
        track = self.repository.get_track(trial.track_id)
        if track is None:
            raise RuntimeError(f"Track not found for trial {trial.trial_id}")
        policy = track.policy_json
        manifest = self.dataset_manager.verify(track.dataset_id)
        try:
            with tempfile.TemporaryDirectory(prefix=f"sigmaevolve_{trial.trial_id}_") as temp_dir:
                temp_path = Path(temp_dir)
                train_script_path = temp_path / "train.py"
                config_path = temp_path / "run_config.json"
                progress_path = temp_path / "progress.json"
                eval_dir = temp_path / "evals"
                debug_path = temp_path / "debug.json"
                eval_dir.mkdir(parents=True, exist_ok=True)
                train_script_path.write_text(trial.source)
                config_path.write_text(
                    json.dumps(
                        {
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
                        },
                        sort_keys=True,
                    )
                )
                heartbeat_stop, heartbeat_thread = self._start_heartbeat(
                    trial_id=trial.trial_id,
                    runner_id=runner_id,
                    interval_sec=int(policy["heartbeat_interval_sec"]),
                )
                metrics_stop = threading.Event()
                metrics_thread: threading.Thread | None = None
                command = [self.python_executable, str(train_script_path), "--config", str(config_path)]
                timed_out = False
                stdout: str | None = None
                stderr: str | None = None
                started_at = time.monotonic()
                metrics_stop, metrics_thread = self._start_active_metrics_reporter(
                    trial_id=trial.trial_id,
                    runner_id=runner_id,
                    progress_path=progress_path,
                    debug_path=debug_path,
                    eval_dir=eval_dir,
                    labels_path=manifest.validation_labels_path,
                    started_at=started_at,
                )
                completed = _run_streamed_subprocess(command, timeout=self.hard_timeout_sec)
                metrics_stop.set()
                if metrics_thread is not None:
                    metrics_thread.join(timeout=1.0)
                timed_out = completed.timed_out
                stdout = _coerce_text(completed.stdout)
                stderr = _coerce_text(completed.stderr)
                process_elapsed_sec = time.monotonic() - started_at
                progress_payload = self._read_progress(progress_path)
                debug_payload = self._read_debug_payload(debug_path)
                timed_out = bool(timed_out or (debug_payload or {}).get("timed_out"))

                if completed.returncode != 0 and not timed_out:
                    failure_outcome = (debug_payload or {}).get("failure_outcome")
                    if failure_outcome == OUTCOME_EVAL_FAILED:
                        self.repository.finalize_trial(
                            trial_id=trial.trial_id,
                            runner_id=runner_id,
                            outcome_reason=OUTCOME_EVAL_FAILED,
                            metrics=None,
                            score=0.0,
                            error_info={
                                "reason": (debug_payload or {}).get("failure_reason") or "train_script_contract_violation",
                                "detail": (debug_payload or {}).get("detail"),
                                "stdout": stdout,
                                "stderr": stderr,
                                "returncode": completed.returncode,
                                "timed_out": timed_out,
                                "progress": progress_payload,
                            },
                        )
                        return
                    self.repository.finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=OUTCOME_CRASHED,
                        metrics=None,
                        score=0.0,
                        error_info={
                            "stdout": stdout,
                            "stderr": stderr,
                            "returncode": completed.returncode,
                            "timed_out": timed_out,
                            "progress": progress_payload,
                        },
                    )
                    return

                try:
                    artifacts = self._load_eval_artifacts(
                        eval_dir=eval_dir,
                        labels_path=manifest.validation_labels_path,
                        fallback_predictions_path=temp_path / "unused_predictions.npz",
                        fallback_elapsed_time_sec=process_elapsed_sec,
                    )
                except Exception as exc:
                    self.repository.finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=OUTCOME_EVAL_FAILED,
                        metrics=None,
                        score=0.0,
                        error_info={
                            "reason": "prediction_load_failed",
                            "detail": str(exc),
                            "timed_out": timed_out,
                            "stdout": stdout,
                            "stderr": stderr,
                            "progress": progress_payload,
                        },
                    )
                    return

                if not artifacts:
                    outcome_reason = OUTCOME_TIMEOUT if timed_out else OUTCOME_EVAL_FAILED
                    error_info: dict[str, Any] = {
                        "reason": "completed_evals_missing" if timed_out else "predictions_missing",
                        "timed_out": timed_out,
                        "stdout": stdout,
                        "stderr": stderr,
                        "progress": progress_payload,
                        "eval_dir": str(eval_dir),
                    }
                    self.repository.finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=outcome_reason,
                        metrics=None,
                        score=0.0,
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
                score = compute_score(metrics, outcome_reason, policy["scorer_settings"])
                self.repository.finalize_trial(
                    trial_id=trial.trial_id,
                    runner_id=runner_id,
                    outcome_reason=outcome_reason,
                    metrics=metrics,
                    score=score,
                    error_info={
                        "stdout": stdout,
                        "stderr": stderr,
                        "debug": debug_payload,
                        "debug_output_path": str(debug_path),
                        "progress": progress_payload,
                        "eval_dir": str(eval_dir),
                        "eval_artifacts": [artifact["path"] for artifact in artifacts],
                        "timed_out": timed_out,
                    },
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
