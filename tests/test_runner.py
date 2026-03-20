from __future__ import annotations

from sigmaevolve.orchestrator import InlineRunnerLauncher
from sigmaevolve.runner import RunnerService
from sigmaevolve.system import EvolutionSystem


SUCCESS_SCRIPT = """
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = json.loads(Path(args.config).read_text())
val = np.load(cfg["validation_split_path"])["features"]
predictions = (val.sum(axis=1) > 0).astype(int)
np.savez(cfg["predictions_output_path"], predictions=predictions)
Path(cfg["debug_output_path"]).write_text("{}")
"""

TIMEOUT_SCRIPT = """
import argparse
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = json.loads(Path(args.config).read_text())
time.sleep(cfg["budget_sec"] + 1)
"""

CRASH_SCRIPT = """
raise SystemExit(3)
"""

EVAL_FAILED_SCRIPT = """
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = json.loads(Path(args.config).read_text())
np.savez(cfg["predictions_output_path"], wrong=np.array([1, 2, 3]))
Path(cfg["debug_output_path"]).write_text("{}")
"""


def build_inline_system(repository, dataset_manager):
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    return EvolutionSystem(repository, dataset_manager, None, InlineRunnerLauncher(runner), runner)


def finalize_baseline(system, track_id):
    baseline = system.repository.list_trials(track_id)[0]
    system.repository.finalize_trial(
        trial_id=baseline.trial_id,
        runner_id=None,
        outcome_reason="stale",
        metrics=None,
        score=0.0,
        error_info={"reason": "test_setup"},
    )


def test_successful_run_produces_metrics_and_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("runner", "mnist:v1", {"budget_sec": 2})
    finalize_baseline(system, track.track_id)
    trial, created = system.repository.create_queued_trial_if_absent(track.track_id, SUCCESS_SCRIPT, {"backend": "test"})
    assert created is True
    reserved = system.repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    system.launcher.launch_trial(reserved.trial_id, reserved.dispatch_token)
    finished = system.repository.get_trial(reserved.trial_id)
    assert finished.outcome_reason == "succeeded"
    assert finished.metrics_json["accuracy"] >= 0.0
    assert finished.score == finished.metrics_json["accuracy"]


def test_timeout_finalizes_with_zero_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("timeout", "mnist:v1", {"budget_sec": 1})
    finalize_baseline(system, track.track_id)
    trial, _ = system.repository.create_queued_trial_if_absent(track.track_id, TIMEOUT_SCRIPT, {"backend": "test"})
    reserved = system.repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    system.launcher.launch_trial(reserved.trial_id, reserved.dispatch_token)
    finished = system.repository.get_trial(reserved.trial_id)
    assert finished.outcome_reason == "timeout"
    assert finished.score == 0.0


def test_crash_finalizes_with_zero_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("crash", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    trial, _ = system.repository.create_queued_trial_if_absent(track.track_id, CRASH_SCRIPT, {"backend": "test"})
    reserved = system.repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    system.launcher.launch_trial(reserved.trial_id, reserved.dispatch_token)
    finished = system.repository.get_trial(reserved.trial_id)
    assert finished.outcome_reason == "crashed"
    assert finished.score == 0.0


def test_eval_failure_finalizes_with_zero_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("eval", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    trial, _ = system.repository.create_queued_trial_if_absent(track.track_id, EVAL_FAILED_SCRIPT, {"backend": "test"})
    reserved = system.repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    system.launcher.launch_trial(reserved.trial_id, reserved.dispatch_token)
    finished = system.repository.get_trial(reserved.trial_id)
    assert finished.outcome_reason == "eval_failed"
    assert finished.score == 0.0


def test_rescore_updates_only_derived_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("rescore", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    trial, _ = system.repository.create_queued_trial_if_absent(track.track_id, SUCCESS_SCRIPT, {"backend": "test"})
    reserved = system.repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    system.launcher.launch_trial(reserved.trial_id, reserved.dispatch_token)
    before = system.repository.get_trial(reserved.trial_id)
    metrics = dict(before.metrics_json)
    migration = system.rescore(track.track_id, {"primary_metric": "accuracy"})
    after = system.repository.get_trial(reserved.trial_id)
    assert migration.updated_trials >= 1
    assert after.metrics_json == metrics
    assert after.score == before.metrics_json["accuracy"]
