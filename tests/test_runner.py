from __future__ import annotations

import pytest

from sigmaevolve.models import CANDIDATE_KIND_STRATEGY_V1
from sigmaevolve.orchestrator import InlineRunnerLauncher
from sigmaevolve.runner import RunnerService
from sigmaevolve.system import EvolutionSystem
from sigmaevolve.train_script_blocks import build_candidate_train_script, build_model_block
from tests.support import make_llm_provenance


SUCCESS_BLOCK = build_model_block(
    """
def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    return torch.stack((-scores, scores), dim=1)
"""
)

TIMEOUT_BLOCK = build_model_block(
    """
def forward(self, x):
    time.sleep(2.0)
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
    imports="import time\nimport torch",
)

SALVAGED_TIMEOUT_BLOCK = build_model_block(
    """
def __init__(self):
    super().__init__()
    self.epoch_index = 0

def on_epoch_start(self, *, epoch_index, num_epochs):
    self.epoch_index = epoch_index

def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    if self.training and self.epoch_index >= 2:
        time.sleep(2.0)
    if self.epoch_index == 0:
        return torch.zeros((x.shape[0], 2), dtype=torch.float32)
    return torch.stack((-scores, scores), dim=1)
""",
    imports="import time\nimport torch",
)

TIEBREAKER_BLOCK = build_model_block(
    """
def __init__(self):
    super().__init__()
    self.epoch_index = 0

def on_epoch_start(self, *, epoch_index, num_epochs):
    self.epoch_index = epoch_index

def forward(self, x):
    if self.training:
        if self.epoch_index == 0:
            time.sleep(0.05)
        elif self.epoch_index == 1:
            time.sleep(0.1)
        else:
            time.sleep(2.0)
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    return torch.stack((-scores, scores), dim=1)
""",
    imports="import time\nimport torch",
)

EARLY_STOP_BLOCK = build_model_block(
    """
def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
    imports="import torch",
)

CRASH_BLOCK = build_model_block(
    """
def __init__(self):
    super().__init__()
    raise RuntimeError("boom")

def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
"""
)

MISSING_EXPORT_BLOCK = "import torch\n"

LOGGING_BLOCK = build_model_block(
    """
def forward(self, x):
    print("stdout-marker", flush=True)
    print("stderr-marker", file=sys.stderr, flush=True)
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    return torch.stack((-scores, scores), dim=1)
""",
    imports="import sys\nimport torch",
)


def build_inline_system(repository, dataset_manager, hard_timeout_sec=5.0):
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager, hard_timeout_sec=hard_timeout_sec)
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
    _, created = system.repository.create_queued_trial_if_absent(
        track_id,
        build_candidate_train_script(source),
        make_llm_provenance(candidate_kind=CANDIDATE_KIND_STRATEGY_V1),
    )
    assert created is True
    reserved = system.repository.reserve_trials(track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    system.launcher.launch_trial(reserved.trial_id, reserved.dispatch_token)
    return system.repository.get_trial(reserved.trial_id)


def test_successful_run_produces_metrics_and_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("runner", "mnist:v1", {"epochs": 2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, SUCCESS_BLOCK)
    assert finished.outcome_reason == "succeeded"
    assert finished.metrics_json["accuracy"] >= 0.0
    assert finished.metrics_json["eval_count"] == 2
    assert finished.score == finished.metrics_json["accuracy"]
    assert finished.error_json is None


def test_timeout_with_no_completed_eval_finalizes_with_zero_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager, hard_timeout_sec=0.3)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("timeout", "mnist:v1", {"epochs": 3})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, TIMEOUT_BLOCK)
    assert finished.outcome_reason == "timeout"
    assert finished.score == 0.0
    assert finished.metrics_json is None


def test_timeout_with_completed_eval_keeps_best_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager, hard_timeout_sec=1.5)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("timeout-salvaged", "mnist:v1", {"epochs": 4})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, SALVAGED_TIMEOUT_BLOCK)
    assert finished.outcome_reason == "timeout"
    assert finished.score == finished.metrics_json["accuracy"]
    assert finished.metrics_json["timed_out"] is True
    assert finished.metrics_json["accuracy"] == 1.0
    assert finished.metrics_json["best_eval_index"] == 2
    assert finished.metrics_json["had_unscored_work_at_timeout"] is True
    assert finished.metrics_json["time_since_last_eval_sec"] > 0.0
    assert finished.error_json is None


def test_equal_accuracy_uses_lower_time_to_best_eval_as_tiebreaker(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager, hard_timeout_sec=1.5)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("tiebreak", "mnist:v1", {"epochs": 4})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, TIEBREAKER_BLOCK)
    assert finished.outcome_reason == "timeout"
    assert finished.metrics_json["accuracy"] == 1.0
    assert finished.metrics_json["time_to_best_eval_sec"] == pytest.approx(0.05, abs=0.08)
    assert finished.metrics_json["best_eval_index"] == 1
    assert finished.metrics_json["last_completed_eval_index"] == 2


def test_run_stops_early_when_validation_accuracy_plateaus(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("early-stop", "mnist:v1", {"epochs": 5})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, EARLY_STOP_BLOCK)
    assert finished.outcome_reason == "succeeded"
    assert finished.metrics_json["early_stopped"] is True
    assert finished.metrics_json["early_stopping_patience"] == 2
    assert finished.metrics_json["early_stop_epoch"] == 3
    assert finished.metrics_json["epochs_completed"] == 3
    assert finished.metrics_json["eval_count"] == 3


def test_crash_finalizes_with_zero_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("crash", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, CRASH_BLOCK)
    assert finished.outcome_reason == "crashed"
    assert finished.score == 0.0


def test_missing_required_exports_finalizes_as_eval_failed(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("eval", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, MISSING_EXPORT_BLOCK)
    assert finished.outcome_reason == "eval_failed"
    assert finished.error_json["reason"] == "train_script_contract_violation"
    assert finished.score == 0.0


def test_run_streams_child_output_to_parent_logs(repository, dataset_manager, capsys):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("logging", "mnist:v1", {"epochs": 1})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, LOGGING_BLOCK)
    captured = capsys.readouterr()
    assert finished.outcome_reason == "succeeded"
    assert "stdout-marker" in captured.out
    assert "stderr-marker" in captured.err


def test_rescore_updates_only_derived_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("rescore", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    _ = _run_trial(system, track.track_id, SUCCESS_BLOCK)
    before = system.repository.sample_trial_context(track.track_id, limit=1)[0]
    metrics = dict(before.metrics_json)
    migration = system.rescore(track.track_id, {"primary_metric": "accuracy"})
    after = system.repository.get_trial(before.trial_id)

    assert migration.updated_trials >= 1
    assert after is not None
    assert after.metrics_json == metrics
    assert after.score == metrics["accuracy"]
