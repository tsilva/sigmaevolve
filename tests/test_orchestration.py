from __future__ import annotations

import threading
import time

from sigmaevolve.baseline import build_baseline_linear_classifier
from sigmaevolve.generation import FixedGenerationBackend
from sigmaevolve.orchestrator import InlineRunnerLauncher, RecordingLauncher
from sigmaevolve.runner import RunnerService
from sigmaevolve.system import EvolutionSystem


def test_create_track_seeds_one_baseline_candidate(system):
    system.prepare_dataset("mnist:v1")
    track = system.create_track("baseline", "mnist:v1", {})
    trials = system.repository.list_trials(track.track_id)
    assert len(trials) == 1
    assert trials[0].status == "queued"
    assert trials[0].source == build_baseline_linear_classifier().replace("\r\n", "\n").rstrip("\n") + "\n"


def test_same_source_is_deduped_within_track_and_allowed_across_tracks(system):
    system.prepare_dataset("mnist:v1")
    first = system.create_track("a", "mnist:v1", {})
    second = system.create_track("b", "mnist:v1", {})
    duplicate_source = "print('candidate')\n"

    original, created = system.repository.create_queued_trial_if_absent(first.track_id, duplicate_source, {"backend": "test"})
    again, created_again = system.repository.create_queued_trial_if_absent(first.track_id, duplicate_source, {"backend": "test"})
    other_track, other_created = system.repository.create_queued_trial_if_absent(second.track_id, duplicate_source, {"backend": "test"})

    assert created is True
    assert created_again is False
    assert original is not None and again is not None
    assert original.trial_id == again.trial_id
    assert other_created is True
    assert other_track is not None


def test_two_orchestrators_cannot_reserve_same_trial(system):
    system.prepare_dataset("mnist:v1")
    track = system.create_track("reserve", "mnist:v1", {"max_parallelism": 1})
    reserved_ids = []
    lock = threading.Lock()

    def reserve_once():
        reserved = system.repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)
        with lock:
            reserved_ids.extend([trial.trial_id for trial in reserved])

    threads = [threading.Thread(target=reserve_once) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(reserved_ids) == 1


def test_two_runners_cannot_both_claim_same_dispatch(system):
    system.prepare_dataset("mnist:v1")
    track = system.create_track("claim", "mnist:v1", {})
    reserved = system.repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)
    trial = reserved[0]
    claims = []
    lock = threading.Lock()

    def claim_once(runner_id):
        claimed = system.claim_trial(trial.trial_id, trial.dispatch_token, runner_id)
        with lock:
            claims.append(claimed.trial_id if claimed else None)

    threads = [threading.Thread(target=claim_once, args=(f"runner_{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert claims.count(trial.trial_id) == 1
    assert claims.count(None) == 1


def test_reconcile_generates_duplicates_without_dispatching_more_work(repository, dataset_manager):
    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    system = EvolutionSystem(
        repository,
        dataset_manager,
        FixedGenerationBackend(source=build_baseline_linear_classifier()),
        InlineRunnerLauncher(runner),
        runner,
    )
    track = system.create_track("dup", "mnist:v1", {"ready_queue_threshold": 2, "dispatch_ttl_sec": 1, "budget_sec": 2})
    baseline = system.repository.list_trials(track.track_id)[0]
    system.repository.finalize_trial(
        trial_id=baseline.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": 0.5},
        score=0.5,
        error_info={"reason": "test_setup"},
    )

    result = system.reconcile_track(track.track_id)
    assert result.duplicate_hashes
    assert result.launched_trial_ids == []


def test_expired_dispatch_is_marked_stale_when_retries_exhausted(system):
    system.prepare_dataset("mnist:v1")
    track = system.create_track("stale", "mnist:v1", {"dispatch_ttl_sec": 0, "max_dispatch_retries": 1})
    system.repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=0, limit=1)
    time.sleep(0.05)
    result = system.reconcile_track(track.track_id)
    assert result.stale_trial_ids


def test_stale_active_trial_is_finalized(system):
    system.prepare_dataset("mnist:v1")
    track = system.create_track("active-stale", "mnist:v1", {"stale_ttl_sec": 0})
    reserved = system.repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    claimed = system.claim_trial(reserved.trial_id, reserved.dispatch_token, "runner")
    assert claimed is not None
    time.sleep(0.05)
    result = system.reconcile_track(track.track_id)
    trial = system.repository.get_trial(reserved.trial_id)
    assert result.stale_trial_ids == [reserved.trial_id]
    assert trial.outcome_reason == "stale"


def test_reconcile_passes_recent_crashes_back_as_negative_context(repository, dataset_manager):
    class CapturingGenerator:
        def __init__(self):
            self.negative_trials = None

        def generate(self, track, dataset_manifest, context_trials, negative_trials=None, generation_index=0):
            self.negative_trials = negative_trials or []
            return type(
                "Generated",
                (),
                {
                    "source": "print('candidate')\n",
                    "provenance_json": {"backend": "fixed", "model": "capture"},
                },
            )()

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    generator = CapturingGenerator()
    launcher = RecordingLauncher()
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    system = EvolutionSystem(repository, dataset_manager, generator, launcher, runner)
    track = system.create_track("negatives", "mnist:v1", {"ready_queue_threshold": 1})

    baseline = repository.list_trials(track.track_id)[0]
    repository.finalize_trial(
        trial_id=baseline.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": 0.5},
        score=0.5,
        error_info={"stdout": "", "stderr": ""},
    )
    failed, _ = repository.create_queued_trial_if_absent(
        track.track_id,
        "print('broken candidate')\n",
        {"backend": "openrouter", "model": "test/model"},
    )
    assert failed is not None
    repository.finalize_trial(
        trial_id=failed.trial_id,
        runner_id=None,
        outcome_reason="crashed",
        metrics=None,
        score=0.0,
        error_info={"returncode": 1, "stderr": "RuntimeError: mat1 and mat2 shapes cannot be multiplied"},
    )

    system.reconcile_track(track.track_id)

    assert generator.negative_trials is not None
    assert [trial.trial_id for trial in generator.negative_trials] == [failed.trial_id]
    assert generator.negative_trials[0].error_json == {
        "returncode": 1,
        "stderr": "RuntimeError: mat1 and mat2 shapes cannot be multiplied",
    }
