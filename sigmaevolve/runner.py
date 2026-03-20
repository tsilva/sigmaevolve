from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from sigmaevolve.models import OUTCOME_CRASHED, OUTCOME_EVAL_FAILED, OUTCOME_SUCCEEDED, OUTCOME_TIMEOUT
from sigmaevolve.scoring import compute_classification_metrics, compute_score


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


class RunnerService:
    def __init__(self, repository, dataset_manager, python_executable: str | None = None) -> None:
        self.repository = repository
        self.dataset_manager = dataset_manager
        self.python_executable = python_executable or sys.executable

    def _start_heartbeat(self, trial_id: str, runner_id: str, interval_sec: int) -> tuple[threading.Event, threading.Thread]:
        stop_event = threading.Event()

        def loop() -> None:
            while not stop_event.wait(interval_sec):
                self.repository.heartbeat_trial(trial_id, runner_id, {"status": "alive"})

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        return stop_event, thread

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
        fallback_predictions_path: Path,
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

        if not artifacts and fallback_predictions_path.exists():
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
        return metrics

    def run_reserved_trial(self, trial_id: str, dispatch_token: str, runner_id: str) -> None:
        trial = self.repository.claim_trial(trial_id, dispatch_token, runner_id)
        if trial is None:
            return
        track = self.repository.get_track(trial.track_id)
        if track is None:
            raise RuntimeError(f"Track not found for trial {trial.trial_id}")
        policy = track.policy_json
        manifest = self.dataset_manager.verify(track.dataset_id)
        heartbeat_stop, heartbeat_thread = self._start_heartbeat(
            trial_id=trial.trial_id,
            runner_id=runner_id,
            interval_sec=int(policy["heartbeat_interval_sec"]),
        )
        try:
            with tempfile.TemporaryDirectory(prefix=f"sigmaevolve_{trial.trial_id}_") as temp_dir:
                temp_path = Path(temp_dir)
                strategy_path = temp_path / "strategy.py"
                config_path = temp_path / "run_config.json"
                progress_path = temp_path / "progress.json"
                eval_dir = temp_path / "evals"
                debug_path = temp_path / "debug.json"
                eval_dir.mkdir(parents=True, exist_ok=True)
                strategy_path.write_text(trial.source)
                config_path.write_text(
                    json.dumps(
                        {
                            "strategy_path": str(strategy_path),
                            "train_split_path": manifest.train_split_path,
                            "validation_split_path": manifest.validation_split_path,
                            "budget_sec": int(policy["budget_sec"]),
                            "max_eval_gap_sec": int(policy["max_eval_gap_sec"]),
                            "random_seed": 1234,
                            "progress_path": str(progress_path),
                            "eval_dir": str(eval_dir),
                            "debug_output_path": str(debug_path),
                            "dataset_metadata": manifest.metadata,
                        },
                        sort_keys=True,
                    )
                )
                command = [self.python_executable, "-m", "sigmaevolve.strategy_runtime", "--config", str(config_path)]
                timed_out = False
                completed: subprocess.CompletedProcess[str] | None = None
                stdout: str | None = None
                stderr: str | None = None
                started_at = time.monotonic()
                try:
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=float(policy["budget_sec"]) + float(policy["max_eval_gap_sec"]) + 1.0,
                        check=False,
                    )
                    stdout = _coerce_text(completed.stdout)
                    stderr = _coerce_text(completed.stderr)
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    stdout = _coerce_text(exc.stdout)
                    stderr = _coerce_text(exc.stderr)
                process_elapsed_sec = time.monotonic() - started_at
                progress_payload = self._read_progress(progress_path)
                debug_payload = self._read_debug_payload(debug_path)
                timed_out = bool(timed_out or (debug_payload or {}).get("timed_out"))

                if completed is not None and completed.returncode != 0:
                    failure_outcome = (debug_payload or {}).get("failure_outcome")
                    if failure_outcome == OUTCOME_EVAL_FAILED:
                        self.repository.finalize_trial(
                            trial_id=trial.trial_id,
                            runner_id=runner_id,
                            outcome_reason=OUTCOME_EVAL_FAILED,
                            metrics=None,
                            score=0.0,
                            error_info={
                                "reason": (debug_payload or {}).get("failure_reason") or "strategy_contract_violation",
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
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
            time.sleep(0.01)
