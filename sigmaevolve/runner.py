from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from sigmaevolve.models import OUTCOME_CRASHED, OUTCOME_EVAL_FAILED, OUTCOME_SUCCEEDED, OUTCOME_TIMEOUT
from sigmaevolve.scoring import compute_classification_metrics, compute_score


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
                script_path = temp_path / "train.py"
                config_path = temp_path / "run_config.json"
                predictions_path = temp_path / "predictions.npz"
                debug_path = temp_path / "debug.json"
                script_path.write_text(trial.source)
                config_path.write_text(
                    json.dumps(
                        {
                            "dataset_dir": manifest.root_dir,
                            "train_split_path": manifest.train_split_path,
                            "validation_split_path": manifest.validation_split_path,
                            "budget_sec": int(policy["budget_sec"]),
                            "random_seed": 1234,
                            "predictions_output_path": str(predictions_path),
                            "debug_output_path": str(debug_path),
                            "dataset_metadata": manifest.metadata,
                        },
                        sort_keys=True,
                    )
                )
                command = [self.python_executable, str(script_path), "--config", str(config_path)]
                try:
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=int(policy["budget_sec"]),
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    self.repository.finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=OUTCOME_TIMEOUT,
                        metrics=None,
                        score=0.0,
                        error_info={"stdout": exc.stdout, "stderr": exc.stderr},
                    )
                    return
                if completed.returncode != 0:
                    self.repository.finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=OUTCOME_CRASHED,
                        metrics=None,
                        score=0.0,
                        error_info={"stdout": completed.stdout, "stderr": completed.stderr, "returncode": completed.returncode},
                    )
                    return
                if not predictions_path.exists():
                    self.repository.finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=OUTCOME_EVAL_FAILED,
                        metrics=None,
                        score=0.0,
                        error_info={"reason": "predictions_missing"},
                    )
                    return
                try:
                    predictions_npz = np.load(predictions_path)
                    predictions = predictions_npz["predictions"]
                    if predictions.ndim > 1:
                        predictions = predictions.argmax(axis=1)
                    labels = np.load(manifest.validation_labels_path)
                    metrics = compute_classification_metrics(predictions.astype(int).tolist(), labels.astype(int).tolist())
                    score = compute_score(metrics, OUTCOME_SUCCEEDED, policy["scorer_settings"])
                except Exception as exc:
                    self.repository.finalize_trial(
                        trial_id=trial.trial_id,
                        runner_id=runner_id,
                        outcome_reason=OUTCOME_EVAL_FAILED,
                        metrics=None,
                        score=0.0,
                        error_info={"reason": "prediction_load_failed", "detail": str(exc)},
                    )
                    return
                self.repository.finalize_trial(
                    trial_id=trial.trial_id,
                    runner_id=runner_id,
                    outcome_reason=OUTCOME_SUCCEEDED,
                    metrics=metrics,
                    score=score,
                    error_info={
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                        "debug_output_path": str(debug_path),
                    },
                )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
            time.sleep(0.01)
