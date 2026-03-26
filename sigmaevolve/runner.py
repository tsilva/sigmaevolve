from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sigmaevolve.env import load_env_file
from sigmaevolve.models import OUTCOME_CRASHED, OUTCOME_EVAL_FAILED, OUTCOME_SUCCEEDED, OUTCOME_TIMEOUT
from sigmaevolve.runner_metrics import (
    build_active_metrics_payload,
    build_final_metrics_payload,
    coerce_optional_scalar,
    load_eval_artifacts,
    select_best_eval,
    select_last_completed_eval,
)
from sigmaevolve.runtime_config import DEFAULT_TRIAL_HARD_TIMEOUT_SEC
from sigmaevolve.scoring import compute_score
from sigmaevolve.wandb_support import WandbRunLogger


logger = logging.getLogger(__name__)
ACTIVE_METRICS_INTERVAL_SEC = 1.0


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

    def _start_heartbeat(
        self,
        trial_id: str,
        runner_id: str,
        interval_sec: int,
    ) -> tuple[threading.Event, threading.Thread]:
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
    ) -> list[dict[str, Any]]:
        return load_eval_artifacts(
            eval_dir=eval_dir,
            labels_path=labels_path,
        )

    def _select_best_eval(self, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        return select_best_eval(artifacts)

    def _select_last_completed_eval(self, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
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
        from sigmaevolve.runner_metrics import apply_debug_metrics

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
        progress_payload = self._read_progress(progress_path)
        debug_payload = self._read_debug_payload(debug_path)
        artifacts: list[dict[str, Any]] = []
        try:
            artifacts = self._load_eval_artifacts(
                eval_dir=eval_dir,
                labels_path=labels_path,
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
            if metrics is None or metrics == last_metrics:
                return last_metrics
            self.repository.update_active_trial_metrics(trial_id=trial_id, runner_id=runner_id, metrics=metrics)
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
        if wandb_run_logger is None or metrics is None:
            return
        try:
            wandb_run_logger.log_metrics(metrics, state=state)
        except Exception:
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
        score: float,
        error_info: dict[str, Any] | None,
        wandb_run_logger: WandbRunLogger | None,
    ) -> None:
        self.repository.finalize_trial(
            trial_id=trial_id,
            runner_id=runner_id,
            outcome_reason=outcome_reason,
            metrics=metrics,
            score=score,
            error_info=error_info,
        )
        if wandb_run_logger is None:
            return
        try:
            wandb_run_logger.finish(
                outcome_reason=outcome_reason,
                metrics=metrics,
                score=score,
                error_info=error_info,
            )
        except Exception:
            logger.warning("W&B run finalization failed for trial %s.", trial_id, exc_info=True)

    def run_reserved_trial(self, trial_id: str, dispatch_token: str, runner_id: str) -> None:
        logger.info("Claiming reserved trial %s with runner %s.", trial_id, runner_id)
        trial = self.repository.claim_trial(trial_id, dispatch_token, runner_id)
        if trial is None:
            logger.info("Skipping trial %s because the reservation could not be claimed.", trial_id)
            return
        logger.info("Claimed trial %s on track %s.", trial.trial_id, trial.track_id)
        track = self.repository.get_track(trial.track_id)
        if track is None:
            raise RuntimeError(f"Track not found for trial {trial.trial_id}")
        policy = track.policy_json
        manifest = self.dataset_manager.verify(track.dataset_id)
        load_env_file()
        logger.info("Verified dataset %s for trial %s.", track.dataset_id, trial.trial_id)
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
                        score=0.0,
                        error_info={
                            "reason": "wandb_init_failed",
                            "detail": str(exc),
                        },
                        wandb_run_logger=None,
                    )
                    logger.info("Finalized trial %s with outcome=%s.", trial.trial_id, OUTCOME_CRASHED)
                    return
                heartbeat_stop, heartbeat_thread = self._start_heartbeat(
                    trial_id=trial.trial_id,
                    runner_id=runner_id,
                    interval_sec=int(policy["heartbeat_interval_sec"]),
                )
                metrics_stop = threading.Event()
                metrics_thread: threading.Thread | None = None
                command = [self.python_executable, "-u", str(train_script_path), "--config", str(config_path)]
                timed_out = False
                stdout: str | None = None
                stderr: str | None = None
                started_at = time.monotonic()
                logger.info("Starting child process for trial %s: %s", trial.trial_id, " ".join(command))
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
                completed = _run_streamed_subprocess(command, timeout=self.hard_timeout_sec)
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

                if completed.returncode != 0 and not timed_out:
                    failure_outcome = (debug_payload or {}).get("failure_outcome")
                    if failure_outcome == OUTCOME_EVAL_FAILED:
                        error_info = {
                            "reason": (debug_payload or {}).get("failure_reason") or "train_script_contract_violation",
                            "detail": (debug_payload or {}).get("detail"),
                            "stdout": stdout,
                            "stderr": stderr,
                            "returncode": completed.returncode,
                            "timed_out": timed_out,
                            "progress": progress_payload,
                        }
                        self._finalize_trial(
                            trial_id=trial.trial_id,
                            runner_id=runner_id,
                            outcome_reason=OUTCOME_EVAL_FAILED,
                            metrics=None,
                            score=0.0,
                            error_info=error_info,
                            wandb_run_logger=wandb_run_logger,
                        )
                        logger.info("Finalized trial %s with outcome=%s.", trial.trial_id, OUTCOME_EVAL_FAILED)
                        return
                    error_info = {
                        "stdout": stdout,
                        "stderr": stderr,
                        "returncode": completed.returncode,
                        "timed_out": timed_out,
                        "progress": progress_payload,
                    }
                    self._finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=OUTCOME_CRASHED,
                        metrics=None,
                        score=0.0,
                        error_info=error_info,
                        wandb_run_logger=wandb_run_logger,
                    )
                    logger.info("Finalized trial %s with outcome=%s.", trial.trial_id, OUTCOME_CRASHED)
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
                    self._finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=OUTCOME_EVAL_FAILED,
                        metrics=None,
                        score=0.0,
                        error_info=error_info,
                        wandb_run_logger=wandb_run_logger,
                    )
                    logger.info("Finalized trial %s with outcome=%s.", trial.trial_id, OUTCOME_EVAL_FAILED)
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
                    self._finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=outcome_reason,
                        metrics=None,
                        score=0.0,
                        error_info=error_info,
                        wandb_run_logger=wandb_run_logger,
                    )
                    logger.info("Finalized trial %s with outcome=%s.", trial.trial_id, outcome_reason)
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
                    score=score,
                    error_info=error_info,
                    wandb_run_logger=wandb_run_logger,
                )
                logger.info(
                    "Finalized trial %s with outcome=%s score=%.6f accuracy=%s.",
                    trial.trial_id,
                    outcome_reason,
                    score,
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
