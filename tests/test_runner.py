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

SALVAGED_TIMEOUT_SCRIPT = """
import argparse
import json
import time
from pathlib import Path
import numpy as np


def write_json_atomic(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    tmp.replace(path)


def write_eval(eval_dir, eval_index, predictions, elapsed_time_sec, epoch):
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    tmp = eval_dir / f".eval_{eval_index:04d}.tmp.npz"
    final_path = eval_dir / f"eval_{eval_index:04d}.npz"
    np.savez(
        tmp,
        predictions=np.asarray(predictions, dtype=np.int64),
        eval_index=np.array(eval_index, dtype=np.int64),
        elapsed_time_sec=np.array(elapsed_time_sec, dtype=np.float64),
        epoch=np.array(epoch, dtype=np.int64),
    )
    tmp.replace(final_path)


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = json.loads(Path(args.config).read_text())
val = np.load(cfg["validation_split_path"])["features"]
correct = (val.sum(axis=1) > 0).astype(int)
wrong = np.zeros_like(correct)
start = time.monotonic()
write_json_atomic(
    cfg["progress_path"],
    {"phase": "eval", "elapsed_time_sec": 0.05, "last_completed_eval_sec": None, "step": 0},
)
write_eval(cfg["eval_dir"], 1, wrong, 0.05, 0)
write_json_atomic(
    cfg["progress_path"],
    {"phase": "train", "elapsed_time_sec": 0.10, "last_completed_eval_sec": 0.05, "step": 1},
)
write_eval(cfg["eval_dir"], 2, correct, 0.10, 1)
write_json_atomic(
    cfg["progress_path"],
    {"phase": "train", "elapsed_time_sec": 0.20, "last_completed_eval_sec": 0.10, "step": 2},
)
time.sleep(cfg["budget_sec"] + 0.2)
"""

TIEBREAKER_SCRIPT = """
import argparse
import json
from pathlib import Path
import numpy as np


def write_json_atomic(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    tmp.replace(path)


def write_eval(eval_dir, eval_index, predictions, elapsed_time_sec):
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    tmp = eval_dir / f".eval_{eval_index:04d}.tmp.npz"
    final_path = eval_dir / f"eval_{eval_index:04d}.npz"
    np.savez(
        tmp,
        predictions=np.asarray(predictions, dtype=np.int64),
        eval_index=np.array(eval_index, dtype=np.int64),
        elapsed_time_sec=np.array(elapsed_time_sec, dtype=np.float64),
    )
    tmp.replace(final_path)


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = json.loads(Path(args.config).read_text())
val = np.load(cfg["validation_split_path"])["features"]
predictions = (val.sum(axis=1) > 0).astype(int)
write_eval(cfg["eval_dir"], 1, predictions, 0.25)
write_eval(cfg["eval_dir"], 2, predictions, 0.75)
write_json_atomic(
    cfg["progress_path"],
    {"phase": "finished", "elapsed_time_sec": 0.75, "last_completed_eval_sec": 0.75, "step": 2},
)
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


def _run_trial(system, track_id, source):
    _, created = system.repository.create_queued_trial_if_absent(track_id, source, {"backend": "test"})
    assert created is True
    reserved = system.repository.reserve_trials(track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    system.launcher.launch_trial(reserved.trial_id, reserved.dispatch_token)
    return system.repository.get_trial(reserved.trial_id)


def test_successful_run_produces_metrics_and_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("runner", "mnist:v1", {"budget_sec": 2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, SUCCESS_SCRIPT)
    assert finished.outcome_reason == "succeeded"
    assert finished.metrics_json["accuracy"] >= 0.0
    assert finished.score == finished.metrics_json["accuracy"]
    assert finished.error_json is None


def test_timeout_with_no_completed_eval_finalizes_with_zero_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("timeout", "mnist:v1", {"budget_sec": 1})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, TIMEOUT_SCRIPT)
    assert finished.outcome_reason == "timeout"
    assert finished.score == 0.0
    assert finished.metrics_json is None


def test_timeout_with_completed_eval_keeps_best_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("timeout-salvaged", "mnist:v1", {"budget_sec": 1})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, SALVAGED_TIMEOUT_SCRIPT)
    assert finished.outcome_reason == "timeout"
    assert finished.score == finished.metrics_json["accuracy"]
    assert finished.metrics_json["timed_out"] is True
    assert finished.metrics_json["accuracy"] == 1.0
    assert finished.metrics_json["best_eval_index"] == 2
    assert finished.metrics_json["time_to_best_eval_sec"] == 0.1
    assert finished.metrics_json["had_unscored_work_at_timeout"] is True
    assert finished.metrics_json["time_since_last_eval_sec"] > 0.0
    assert finished.error_json is None


def test_equal_accuracy_uses_lower_time_to_best_eval_as_tiebreaker(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("tiebreak", "mnist:v1", {"budget_sec": 2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, TIEBREAKER_SCRIPT)
    assert finished.outcome_reason == "succeeded"
    assert finished.metrics_json["accuracy"] == 1.0
    assert finished.metrics_json["time_to_best_eval_sec"] == 0.25
    assert finished.metrics_json["best_eval_index"] == 1
    assert finished.metrics_json["last_completed_eval_index"] == 2


def test_crash_finalizes_with_zero_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("crash", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, CRASH_SCRIPT)
    assert finished.outcome_reason == "crashed"
    assert finished.score == 0.0


def test_eval_failure_finalizes_with_zero_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("eval", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, EVAL_FAILED_SCRIPT)
    assert finished.outcome_reason == "eval_failed"
    assert finished.score == 0.0


def test_rescore_updates_only_derived_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("rescore", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    _ = _run_trial(system, track.track_id, SUCCESS_SCRIPT)
    before = system.repository.sample_trial_context(track.track_id, limit=1)[0]
    metrics = dict(before.metrics_json)
    migration = system.rescore(track.track_id, {"primary_metric": "accuracy"})
    after = system.repository.get_trial(before.trial_id)
    assert migration.updated_trials >= 1
    assert after.metrics_json == metrics
    assert after.score == before.metrics_json["accuracy"]
