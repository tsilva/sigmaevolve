from __future__ import annotations

import threading
import time

import pytest

from sigmaevolve.baseline import build_baseline_linear_classifier
from sigmaevolve.generation import FixedGenerationBackend
from sigmaevolve.hashing import compute_script_hash, normalize_source
from sigmaevolve.models import CANDIDATE_KIND_STRATEGY_V1
from sigmaevolve.orchestrator import InlineRunnerLauncher, RecordingLauncher
from sigmaevolve.runner import RunnerService
from sigmaevolve.storage import trials_table
from sigmaevolve.system import EvolutionSystem
from sigmaevolve.train_script_blocks import build_candidate_train_script, build_model_block
from tests.support import make_llm_provenance


def test_create_track_seeds_one_baseline_candidate(system):
    system.prepare_dataset("mnist:v1")
    track = system.create_track("baseline", "mnist:v1", {})
    trials = system.repository.list_trials(track.track_id)
    assert len(trials) == 1
    assert trials[0].status == "queued"
    assert trials[0].source == build_baseline_linear_classifier().replace("\r\n", "\n").rstrip("\n") + "\n"
    assert trials[0].provenance_json["candidate_kind"] == CANDIDATE_KIND_STRATEGY_V1


def test_same_source_is_deduped_within_track_and_allowed_across_tracks(system):
    system.prepare_dataset("mnist:v1")
    first = system.create_track("a", "mnist:v1", {})
    second = system.create_track("b", "mnist:v1", {})
    duplicate_source = build_candidate_train_script(
        build_model_block(
            """
def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
"""
        )
    )
    provenance = make_llm_provenance(candidate_kind=CANDIDATE_KIND_STRATEGY_V1)

    original, created = system.repository.create_queued_trial_if_absent(first.track_id, duplicate_source, provenance)
    again, created_again = system.repository.create_queued_trial_if_absent(first.track_id, duplicate_source, provenance)
    other_track, other_created = system.repository.create_queued_trial_if_absent(second.track_id, duplicate_source, provenance)

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
    track = system.create_track("dup", "mnist:v1", {"ready_queue_threshold": 2, "dispatch_ttl_sec": 1, "epochs": 2})
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
    assert len(result.duplicate_hashes) == 4
    assert result.launched_trial_ids == []


def test_reconcile_retries_duplicate_generation_with_incremented_retry_count(repository, dataset_manager):
    class RetryingDuplicateGenerator:
        def __init__(self, source: str):
            self.source = source
            self.retry_counts = []

        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            self.retry_counts.append(duplicate_retry_count)
            return type(
                "Generated",
                (),
                {
                    "source": self.source,
                    "provenance_json": {
                        **make_llm_provenance(model="retry-capture"),
                        "duplicate_retry_count": duplicate_retry_count,
                    },
                },
            )()

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    generator = RetryingDuplicateGenerator(build_baseline_linear_classifier())
    system = EvolutionSystem(
        repository,
        dataset_manager,
        generator,
        InlineRunnerLauncher(runner),
        runner,
    )
    track = system.create_track(
        "dup-retries", "mnist:v1", {"ready_queue_threshold": 2, "dispatch_ttl_sec": 1, "epochs": 2}
    )
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

    assert generator.retry_counts == [0, 1, 2, 3]
    assert len(result.duplicate_hashes) == 4
    assert result.generated_trial_ids == []
    assert result.launched_trial_ids == []


def test_reconcile_persists_successful_retry_generation_params(repository, dataset_manager):
    class DuplicateThenUniqueGenerator:
        def __init__(self, duplicate_source: str, unique_source: str):
            self.duplicate_source = duplicate_source
            self.unique_source = unique_source
            self.retry_counts = []

        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            self.retry_counts.append(duplicate_retry_count)
            source = self.duplicate_source if duplicate_retry_count == 0 else self.unique_source
            return type(
                "Generated",
                (),
                {
                    "source": source,
                    "provenance_json": {
                        **make_llm_provenance(model="retry-capture"),
                        "generation_index": generation_index,
                        "duplicate_retry_count": duplicate_retry_count,
                        "generation_config": {
                            "model": "retry-capture",
                            "temperature": 0.2 + (0.1 * duplicate_retry_count),
                            "max_tokens": 1500,
                        },
                    },
                },
            )()

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    duplicate_source = build_baseline_linear_classifier()
    unique_source = build_candidate_train_script(
        build_model_block(
            """
def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    return torch.stack((-scores, scores), dim=1)
"""
        )
    )
    generator = DuplicateThenUniqueGenerator(duplicate_source=duplicate_source, unique_source=unique_source)
    system = EvolutionSystem(
        repository,
        dataset_manager,
        generator,
        RecordingLauncher(),
        runner,
    )
    track = system.create_track(
        "dup-success", "mnist:v1", {"ready_queue_threshold": 2, "dispatch_ttl_sec": 1, "epochs": 2}
    )
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
    trials = system.repository.list_trials(track.track_id)
    created_trial = next(trial for trial in trials if trial.source == unique_source)

    assert generator.retry_counts == [0, 1]
    assert len(result.duplicate_hashes) == 1
    assert result.generated_trial_ids == [created_trial.trial_id]
    assert created_trial.provenance_json["duplicate_retry_count"] == 1
    assert created_trial.provenance_json["generation_index"] == 1
    assert created_trial.provenance_json["generation_config"]["temperature"] == pytest.approx(0.3)


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


def test_weighted_successful_sampling_favors_higher_scores(repository, dataset_manager):
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
    track = system.create_track("weighted", "mnist:v1", {"sampling_settings": {"seed": 7}})

    trials = repository.list_trials(track.track_id)
    baseline = trials[0]
    mid, _ = repository.create_queued_trial_if_absent(
        track.track_id,
        build_candidate_train_script(
            build_model_block(
                """
def forward(self, x):
    return torch.tensor([[0.0, 1.0]], dtype=torch.float32).repeat(x.shape[0], 1)
"""
            )
        ),
        make_llm_provenance(model="mid", candidate_kind=CANDIDATE_KIND_STRATEGY_V1),
    )
    low, _ = repository.create_queued_trial_if_absent(
        track.track_id,
        build_candidate_train_script(
            build_model_block(
                """
def forward(self, x):
    return torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(x.shape[0], 1)
"""
            )
        ),
        make_llm_provenance(model="low", candidate_kind=CANDIDATE_KIND_STRATEGY_V1),
    )
    assert mid is not None and low is not None

    repository.finalize_trial(
        trial_id=baseline.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": 0.9},
        score=0.9,
        error_info=None,
    )
    repository.finalize_trial(
        trial_id=mid.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": 0.3},
        score=0.3,
        error_info=None,
    )
    repository.finalize_trial(
        trial_id=low.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": 0.1},
        score=0.1,
        error_info=None,
    )

    counts = {baseline.trial_id: 0, mid.trial_id: 0, low.trial_id: 0}
    for generation_index in range(300):
        sampled = system.orchestrator._sample_successful_context_trials(
            track.track_id,
            {"seed": 7},
            generation_index,
        )
        counts[sampled[0].trial_id] += 1

    assert counts[baseline.trial_id] > counts[mid.trial_id] > counts[low.trial_id]


def test_reconcile_never_passes_failed_trials_as_generation_context(repository, dataset_manager):
    class CapturingGenerator:
        def __init__(self):
            self.context_trials = None
            self.negative_trials = None

        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            self.context_trials = context_trials
            self.negative_trials = negative_trials or []
            return type(
                "Generated",
                (),
                {
                    "source": build_candidate_train_script(
                        build_model_block(
                            """
def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    return torch.stack((-scores, scores), dim=1)
"""
                        )
                    ),
                    "provenance_json": {
                        **make_llm_provenance(model="capture"),
                    },
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
        build_candidate_train_script(
            build_model_block(
                """
def __init__(self):
    super().__init__()
    raise RuntimeError("broken")

def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
"""
            )
        ),
        make_llm_provenance(model="test/model", candidate_kind=CANDIDATE_KIND_STRATEGY_V1),
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

    assert generator.context_trials is not None
    assert [trial.trial_id for trial in generator.context_trials] == [baseline.trial_id]
    assert generator.negative_trials is not None
    assert generator.negative_trials == []


def test_reconcile_does_not_mutate_legacy_successes(repository, dataset_manager):
    class CapturingGenerator:
        def __init__(self):
            self.called = False

        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            self.called = True
            return type(
                "Generated",
                (),
                {
                    "source": build_candidate_train_script(
                        "import torch\n"
                    ),
                    "provenance_json": {
                        **make_llm_provenance(model="capture"),
                    },
                },
            )()

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    generator = CapturingGenerator()
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    system = EvolutionSystem(repository, dataset_manager, generator, RecordingLauncher(), runner)
    track = system.create_track("legacy-only", "mnist:v1", {"ready_queue_threshold": 1})

    baseline = repository.list_trials(track.track_id)[0]
    repository.finalize_trial(
        trial_id=baseline.trial_id,
        runner_id=None,
        outcome_reason="stale",
        metrics=None,
        score=0.0,
        error_info={"reason": "test_setup"},
    )
    legacy_source = normalize_source("print('legacy train script')\n")
    legacy_trial_id = "trial_legacy_success"
    with repository.transaction() as conn:
        conn.execute(
            trials_table.insert().values(
                trial_id=legacy_trial_id,
                track_id=track.track_id,
                source=legacy_source,
                script_hash=compute_script_hash(legacy_source),
                provenance_json={"backend": "legacy"},
                status="queued",
                outcome_reason=None,
                dispatch_token=None,
                dispatch_deadline_at=None,
                runner_id=None,
                heartbeat_at=None,
                started_at=None,
                finished_at=None,
                metrics_json=None,
                score=0.0,
                error_json=None,
                dispatch_attempts=0,
                created_at=baseline.created_at,
            )
        )
    legacy = repository.get_trial(legacy_trial_id)
    assert legacy is not None
    repository.finalize_trial(
        trial_id=legacy.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": 0.8},
        score=0.8,
        error_info=None,
    )

    result = system.reconcile_track(track.track_id)

    assert generator.called is False
    assert result.generated_trial_ids == []


def test_reconcile_rejects_mutations_outside_evolve_blocks(repository, dataset_manager):
    class InvalidGenerator:
        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            invalid_source = context_trials[0].source.replace("import json\n", "import json\nIMMUTABLE_BREAK = True\n", 1)
            return type(
                "Generated",
                (),
                {
                    "source": invalid_source,
                    "provenance_json": {
                        **make_llm_provenance(model="invalid"),
                    },
                },
            )()

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    system = EvolutionSystem(repository, dataset_manager, InvalidGenerator(), RecordingLauncher(), runner)
    track = system.create_track("invalid-mutation", "mnist:v1", {"ready_queue_threshold": 1})

    baseline = repository.list_trials(track.track_id)[0]
    repository.finalize_trial(
        trial_id=baseline.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": 0.5},
        score=0.5,
        error_info=None,
    )

    result = system.reconcile_track(track.track_id)

    assert result.generated_trial_ids == []
    assert result.errors
    assert result.errors[0].startswith("invalid_mutation:")


def test_reconcile_persists_launcher_metadata_for_launched_trials(repository, dataset_manager):
    class MetadataLauncher:
        def launch_trial(self, trial_id: str, dispatch_token: str):
            return {
                "kind": "modal",
                "run_id": f"fc-{trial_id}",
                "run_url": f"https://modal.com/apps/test/runs/{trial_id}",
            }

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    system = EvolutionSystem(
        repository,
        dataset_manager,
        FixedGenerationBackend(source=build_baseline_linear_classifier()),
        MetadataLauncher(),
        runner,
    )
    track = system.create_track("launch-metadata", "mnist:v1", {"ready_queue_threshold": 0, "max_parallelism": 1})

    queued_trial = repository.list_trials(track.track_id)[0]
    result = system.reconcile_track(track.track_id)
    updated_trial = repository.get_trial(queued_trial.trial_id)

    assert result.launched_trial_ids == [queued_trial.trial_id]
    assert updated_trial is not None
    assert updated_trial.provenance_json["launcher"] == {
        "kind": "modal",
        "run_id": f"fc-{queued_trial.trial_id}",
        "run_url": f"https://modal.com/apps/test/runs/{queued_trial.trial_id}",
    }
