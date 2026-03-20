from __future__ import annotations

import pytest

from sigmaevolve.models import CANDIDATE_KIND_STRATEGY_V1
from sigmaevolve.orchestrator import InlineRunnerLauncher
from sigmaevolve.runner import RunnerService
from sigmaevolve.system import EvolutionSystem


SUCCESS_STRATEGY = """
import numpy as np


def initialize(ctx):
    return {}


def train_window(ctx, state):
    state["window"] = state.get("window", 0) + 1
    state["done"] = True


def predict_validation(ctx, state):
    val = ctx.validation_features
    logits = np.stack([-(val.sum(axis=1)), val.sum(axis=1)], axis=1)
    return logits
"""

TIMEOUT_STRATEGY = """
import time


def initialize(ctx):
    return {}


def train_window(ctx, state):
    time.sleep(ctx.budget_sec + 0.2)


def predict_validation(ctx, state):
    return [0] * ctx.validation_features.shape[0]
"""

SALVAGED_TIMEOUT_STRATEGY = """
import numpy as np
import time


def initialize(ctx):
    return {"window": 0}


def train_window(ctx, state):
    state["window"] += 1
    if state["window"] >= 3:
        time.sleep(ctx.remaining_budget_sec + 0.2)


def predict_validation(ctx, state):
    val = ctx.validation_features
    correct = (val.sum(axis=1) > 0).astype(np.int64)
    if state["window"] == 1:
        return np.zeros_like(correct)
    return correct
"""

TIEBREAKER_STRATEGY = """
import numpy as np
import time


def initialize(ctx):
    return {"window": 0}


def train_window(ctx, state):
    state["window"] += 1
    if state["window"] == 1:
        time.sleep(0.25)
    elif state["window"] == 2:
        time.sleep(0.5)
    else:
        time.sleep(ctx.remaining_budget_sec + 0.2)


def predict_validation(ctx, state):
    val = ctx.validation_features
    return (val.sum(axis=1) > 0).astype(np.int64)
"""

CRASH_STRATEGY = """
def initialize(ctx):
    raise RuntimeError("boom")


def train_window(ctx, state):
    return None


def predict_validation(ctx, state):
    return []
"""

MISSING_EXPORT_STRATEGY = """
def initialize(ctx):
    return {}


def train_window(ctx, state):
    return None
"""

WINDOW_OVERRUN_STRATEGY = """
import time


def initialize(ctx):
    return {}


def train_window(ctx, state):
    time.sleep(ctx.max_eval_gap_sec + 0.1)


def predict_validation(ctx, state):
    return [0] * ctx.validation_features.shape[0]
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
    _, created = system.repository.create_queued_trial_if_absent(
        track_id,
        source,
        {"backend": "test", "candidate_kind": CANDIDATE_KIND_STRATEGY_V1},
    )
    assert created is True
    reserved = system.repository.reserve_trials(track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    system.launcher.launch_trial(reserved.trial_id, reserved.dispatch_token)
    return system.repository.get_trial(reserved.trial_id)


def test_successful_run_produces_metrics_and_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("runner", "mnist:v1", {"budget_sec": 2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, SUCCESS_STRATEGY)
    assert finished.outcome_reason == "succeeded"
    assert finished.metrics_json["accuracy"] >= 0.0
    assert finished.score == finished.metrics_json["accuracy"]
    assert finished.error_json is None


def test_timeout_with_no_completed_eval_finalizes_with_zero_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("timeout", "mnist:v1", {"budget_sec": 1, "max_eval_gap_sec": 2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, TIMEOUT_STRATEGY)
    assert finished.outcome_reason == "timeout"
    assert finished.score == 0.0
    assert finished.metrics_json is None


def test_timeout_with_completed_eval_keeps_best_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("timeout-salvaged", "mnist:v1", {"budget_sec": 1, "max_eval_gap_sec": 2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, SALVAGED_TIMEOUT_STRATEGY)
    assert finished.outcome_reason == "timeout"
    assert finished.score == finished.metrics_json["accuracy"]
    assert finished.metrics_json["timed_out"] is True
    assert finished.metrics_json["accuracy"] == 1.0
    assert finished.metrics_json["best_eval_index"] == 2
    assert finished.metrics_json["had_unscored_work_at_timeout"] is True
    assert finished.metrics_json["time_since_last_eval_sec"] > 0.0
    assert finished.error_json is None


def test_equal_accuracy_uses_lower_time_to_best_eval_as_tiebreaker(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("tiebreak", "mnist:v1", {"budget_sec": 1.1, "max_eval_gap_sec": 2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, TIEBREAKER_STRATEGY)
    assert finished.outcome_reason == "timeout"
    assert finished.metrics_json["accuracy"] == 1.0
    assert finished.metrics_json["time_to_best_eval_sec"] == pytest.approx(0.25, abs=0.15)
    assert finished.metrics_json["best_eval_index"] == 1
    assert finished.metrics_json["last_completed_eval_index"] == 2


def test_crash_finalizes_with_zero_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("crash", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, CRASH_STRATEGY)
    assert finished.outcome_reason == "crashed"
    assert finished.score == 0.0


def test_missing_required_exports_finalizes_as_eval_failed(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("eval", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, MISSING_EXPORT_STRATEGY)
    assert finished.outcome_reason == "eval_failed"
    assert finished.error_json["reason"] == "strategy_contract_violation"
    assert finished.score == 0.0


def test_window_overrun_finalizes_as_eval_failed(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("window-overrun", "mnist:v1", {"budget_sec": 1, "max_eval_gap_sec": 0.2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, WINDOW_OVERRUN_STRATEGY)
    assert finished.outcome_reason == "eval_failed"
    assert finished.error_json["reason"] == "eval_window_missed"
    assert finished.score == 0.0


def test_rescore_updates_only_derived_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = system.create_track("rescore", "mnist:v1", {})
    finalize_baseline(system, track.track_id)
    _ = _run_trial(system, track.track_id, SUCCESS_STRATEGY)
    before = system.repository.sample_trial_context(track.track_id, limit=1)[0]
    metrics = dict(before.metrics_json)
    migration = system.rescore(track.track_id, {"primary_metric": "accuracy"})
    after = system.repository.get_trial(before.trial_id)

    assert migration.updated_trials >= 1
    assert after is not None
    assert after.metrics_json == metrics
    assert after.score == metrics["accuracy"]
