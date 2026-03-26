from __future__ import annotations

import threading
import time

import pytest

from sigmaevolve.generation import FixedGenerationBackend, OpenRouterGenerationBackend
from sigmaevolve.core import CANDIDATE_KIND_STRATEGY_V1, GenerationResult
from sigmaevolve.orchestration import InlineRunnerLauncher
from sigmaevolve.execution import RunnerService
from sigmaevolve.orchestration import EvolutionSystem
from sigmaevolve.generation import build_baseline_train_script, build_candidate_train_script, build_model_block
from tests.support import RecordingLauncherDouble, make_llm_provenance


def _prepare_repo_dataset(repository, dataset_manager, dataset_id: str = "mnist:v1") -> None:
    dataset_manager.prepare(dataset_id)
    repository.register_dataset(dataset_id, str(dataset_manager.manifest_path_for(dataset_id)))


def _build_system(repository, dataset_manager, generator, launcher):
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    return EvolutionSystem(repository, dataset_manager, generator, launcher, runner), runner


def _finalize_baseline_success(repository, track_id: str, score: float = 0.5):
    baseline = repository.list_trials(track_id)[0]
    repository.finalize_trial(
        trial_id=baseline.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": score},
        score=score,
        error_info=None,
    )
    return baseline


def test_create_track_seeds_one_baseline_candidate(system):
    system.prepare_dataset("mnist:v1")
    track = system.create_track("baseline", "mnist:v1", {})
    trials = system.repository.list_trials(track.track_id)
    assert len(trials) == 1
    assert trials[0].status == "queued"
    assert trials[0].source == build_baseline_train_script().replace("\r\n", "\n").rstrip("\n") + "\n"
    assert trials[0].provenance_json["candidate_kind"] == CANDIDATE_KIND_STRATEGY_V1


def test_reconcile_generates_from_queued_baseline_before_first_result(repository, dataset_manager):
    class CapturingGenerator:
        def __init__(self):
            self.context_trials = None

        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            del track, dataset_manifest, negative_trials, generation_index, duplicate_retry_count
            self.context_trials = context_trials
            return GenerationResult(
                source=build_candidate_train_script(
                    build_model_block(
                        """
def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    return torch.stack((-scores, scores), dim=1)
"""
                    )
                ),
                provenance_json=make_llm_provenance(
                    model="cold-start",
                    context_trial_ids=[trial.trial_id for trial in context_trials],
                ),
            )

    _prepare_repo_dataset(repository, dataset_manager)
    generator = CapturingGenerator()
    system, _ = _build_system(repository, dataset_manager, generator, RecordingLauncherDouble())
    track = system.create_track("cold-start", "mnist:v1", {})

    baseline = repository.list_trials(track.track_id)[0]
    result = system.reconcile_track(track.track_id, ready_queue_threshold=2, max_parallelism=0)
    trials = repository.list_trials(track.track_id)

    assert generator.context_trials is not None
    assert [trial.trial_id for trial in generator.context_trials] == [baseline.trial_id]
    assert len(result.generated_trial_ids) == 1
    assert len(trials) == 2
    assert trials[0].trial_id == baseline.trial_id
    assert trials[0].status == "queued"


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
    track = system.create_track("reserve", "mnist:v1", {})
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
    _prepare_repo_dataset(repository, dataset_manager)
    system, runner = _build_system(
        repository,
        dataset_manager,
        FixedGenerationBackend(source=build_baseline_train_script()),
        None,
    )
    system.launcher = InlineRunnerLauncher(runner)
    system.orchestrator.launcher = system.launcher
    track = system.create_track("dup", "mnist:v1", {"dispatch_ttl_sec": 1, "epochs": 2})
    baseline = system.repository.list_trials(track.track_id)[0]
    system.repository.finalize_trial(
        trial_id=baseline.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": 0.5},
        score=0.5,
        error_info={"reason": "test_setup"},
    )

    result = system.reconcile_track(track.track_id, ready_queue_threshold=2)
    trials = system.repository.list_trials(track.track_id)
    assert len(result.duplicate_hashes) == 4
    assert len(result.duplicate_trial_ids) == 4
    assert result.launched_trial_ids == []
    assert len(trials) == 5
    assert all(trial.outcome_reason == "duplicate" for trial in trials[1:])


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

    _prepare_repo_dataset(repository, dataset_manager)
    generator = RetryingDuplicateGenerator(build_baseline_train_script())
    system, runner = _build_system(repository, dataset_manager, generator, None)
    system.launcher = InlineRunnerLauncher(runner)
    system.orchestrator.launcher = system.launcher
    track = system.create_track(
        "dup-retries", "mnist:v1", {"dispatch_ttl_sec": 1, "epochs": 2}
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

    result = system.reconcile_track(track.track_id, ready_queue_threshold=2)

    assert sorted(generator.retry_counts) == [0, 0, 1, 1]
    assert len(result.duplicate_hashes) == 4
    assert len(result.duplicate_trial_ids) == 4
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

    _prepare_repo_dataset(repository, dataset_manager)
    duplicate_source = build_baseline_train_script()
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
    system, _ = _build_system(repository, dataset_manager, generator, RecordingLauncherDouble())
    track = system.create_track(
        "dup-success", "mnist:v1", {"dispatch_ttl_sec": 1, "epochs": 2}
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
    assert len(result.duplicate_trial_ids) == 1
    assert result.generated_trial_ids == [created_trial.trial_id]
    assert created_trial.provenance_json["duplicate_retry_count"] == 1
    assert created_trial.provenance_json["generation_index"] == 1
    assert created_trial.provenance_json["generation_config"]["temperature"] == pytest.approx(0.3)
    assert created_trial.provenance_json["generation"]["assertions_passed"] is True


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


def test_stale_active_modal_trial_requests_run_cancellation(repository, dataset_manager):
    class CancellationLauncher:
        def __init__(self):
            self.cancelled: list[dict[str, object]] = []

        def launch_trial(self, trial_id: str, dispatch_token: str, launch_policy: dict[str, object] | None = None):
            del trial_id, dispatch_token, launch_policy
            return None

        def cancel_run(self, launcher_metadata: dict[str, object]) -> None:
            self.cancelled.append(dict(launcher_metadata))

    _prepare_repo_dataset(repository, dataset_manager)
    launcher = CancellationLauncher()
    system, _ = _build_system(repository, dataset_manager, FixedGenerationBackend(source=build_baseline_train_script()), launcher)
    track = system.create_track("active-stale-modal", "mnist:v1", {"stale_ttl_sec": 0})
    reserved = repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    claimed = system.claim_trial(reserved.trial_id, reserved.dispatch_token, "runner")
    assert claimed is not None
    repository.record_trial_launcher_metadata(
        claimed.trial_id,
        {
            "kind": "modal",
            "run_id": "fc-123",
            "run_url": "https://modal.com/apps/test/runs/fc-123",
        },
    )

    time.sleep(0.05)
    result = system.reconcile_track(track.track_id)
    trial = repository.get_trial(claimed.trial_id)

    assert result.stale_trial_ids == [claimed.trial_id]
    assert launcher.cancelled == [
        {
            "kind": "modal",
            "run_id": "fc-123",
            "run_url": "https://modal.com/apps/test/runs/fc-123",
        }
    ]
    assert trial is not None
    assert trial.outcome_reason == "stale"
    assert trial.provenance_json["launcher"]["cancel_outcome"] == "requested"
    assert trial.provenance_json["launcher"]["cancel_attempted_at"]
    assert "cancel_error" not in trial.provenance_json["launcher"]


def test_stale_active_modal_trial_records_missing_run_id_skip(repository, dataset_manager):
    class CancellationLauncher:
        def __init__(self):
            self.cancel_calls = 0

        def launch_trial(self, trial_id: str, dispatch_token: str, launch_policy: dict[str, object] | None = None):
            del trial_id, dispatch_token, launch_policy
            return None

        def cancel_run(self, launcher_metadata: dict[str, object]) -> None:
            del launcher_metadata
            self.cancel_calls += 1

    _prepare_repo_dataset(repository, dataset_manager)
    launcher = CancellationLauncher()
    system, _ = _build_system(repository, dataset_manager, FixedGenerationBackend(source=build_baseline_train_script()), launcher)
    track = system.create_track("active-stale-missing-run", "mnist:v1", {"stale_ttl_sec": 0})
    reserved = repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    claimed = system.claim_trial(reserved.trial_id, reserved.dispatch_token, "runner")
    assert claimed is not None
    repository.record_trial_launcher_metadata(
        claimed.trial_id,
        {
            "kind": "modal",
            "run_url": "https://modal.com/apps/test/runs/fc-missing",
        },
    )

    time.sleep(0.05)
    system.reconcile_track(track.track_id)
    trial = repository.get_trial(claimed.trial_id)

    assert launcher.cancel_calls == 0
    assert trial is not None
    assert trial.outcome_reason == "stale"
    assert trial.provenance_json["launcher"]["cancel_outcome"] == "skipped_no_run_id"
    assert trial.provenance_json["launcher"]["cancel_attempted_at"]
    assert "cancel_error" not in trial.provenance_json["launcher"]


def test_stale_active_modal_trial_records_cancellation_failure(repository, dataset_manager):
    class FailingCancellationLauncher:
        def launch_trial(self, trial_id: str, dispatch_token: str, launch_policy: dict[str, object] | None = None):
            del trial_id, dispatch_token, launch_policy
            return None

        def cancel_run(self, launcher_metadata: dict[str, object]) -> None:
            del launcher_metadata
            raise RuntimeError("modal cancellation failed")

    _prepare_repo_dataset(repository, dataset_manager)
    launcher = FailingCancellationLauncher()
    system, _ = _build_system(repository, dataset_manager, FixedGenerationBackend(source=build_baseline_train_script()), launcher)
    track = system.create_track("active-stale-failed-cancel", "mnist:v1", {"stale_ttl_sec": 0})
    reserved = repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)[0]
    claimed = system.claim_trial(reserved.trial_id, reserved.dispatch_token, "runner")
    assert claimed is not None
    repository.record_trial_launcher_metadata(
        claimed.trial_id,
        {
            "kind": "modal",
            "run_id": "fc-bad",
        },
    )

    time.sleep(0.05)
    system.reconcile_track(track.track_id)
    trial = repository.get_trial(claimed.trial_id)

    assert trial is not None
    assert trial.outcome_reason == "stale"
    assert trial.provenance_json["launcher"]["cancel_outcome"] == "failed"
    assert trial.provenance_json["launcher"]["cancel_attempted_at"]
    assert trial.provenance_json["launcher"]["cancel_error"] == "modal cancellation failed"


def test_weighted_successful_sampling_favors_higher_scores(repository, dataset_manager):
    _prepare_repo_dataset(repository, dataset_manager)
    system, runner = _build_system(
        repository,
        dataset_manager,
        FixedGenerationBackend(source=build_baseline_train_script()),
        None,
    )
    system.launcher = InlineRunnerLauncher(runner)
    system.orchestrator.launcher = system.launcher
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

    draw_counts = {baseline.trial_id: 0, mid.trial_id: 0, low.trial_id: 0}
    current_counts = {baseline.trial_id: 0, mid.trial_id: 0, low.trial_id: 0}
    for generation_index in range(300):
        sampled = system.orchestrator._sample_successful_context_trials(
            track.track_id,
            {"seed": 7},
            generation_index,
        )
        assert len(sampled) == 2
        assert sampled[0].trial_id != sampled[1].trial_id
        assert sampled[0].score >= sampled[1].score
        for trial in sampled:
            draw_counts[trial.trial_id] += 1
        current_counts[sampled[0].trial_id] += 1

    assert draw_counts[baseline.trial_id] > draw_counts[mid.trial_id] > draw_counts[low.trial_id]
    assert current_counts[baseline.trial_id] > current_counts[mid.trial_id] > current_counts[low.trial_id]


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

    _prepare_repo_dataset(repository, dataset_manager)
    generator = CapturingGenerator()
    system, _ = _build_system(repository, dataset_manager, generator, RecordingLauncherDouble())
    track = system.create_track("negatives", "mnist:v1", {})

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
    system = EvolutionSystem(repository, dataset_manager, InvalidGenerator(), RecordingLauncherDouble(), runner)
    track = system.create_track("invalid-mutation", "mnist:v1", {})

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
    assert all(error.startswith("invalid_mutation:") for error in result.errors)
    assert len(result.failed_generation_trial_ids) == 2
    failed_trials = [repository.get_trial(trial_id) for trial_id in result.failed_generation_trial_ids]
    assert all(trial is not None for trial in failed_trials)
    assert all(trial.outcome_reason == "generation_failed" for trial in failed_trials if trial is not None)
    assert all(trial.provenance_json["generation"]["assertions_passed"] is False for trial in failed_trials if trial is not None)
    assert all(trial.provenance_json["generation"]["assertion_failures"] for trial in failed_trials if trial is not None)


def test_reconcile_applies_search_replace_response_before_queueing(repository, dataset_manager):
    class PatchGenerator:
        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            return type(
                "Generated",
                (),
                {
                    "source": """<<<<<<< SEARCH
    def forward(self, x):
        return self.network(x)
=======
    def forward(self, x):
        return self.network(x) * 0.5
>>>>>>> REPLACE
""",
                    "provenance_json": {
                        **make_llm_provenance(model="patch-generator"),
                    },
                },
            )()

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    system = EvolutionSystem(repository, dataset_manager, PatchGenerator(), RecordingLauncherDouble(), runner)
    track = system.create_track("patch-mutation", "mnist:v1", {})

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
    created_trial = repository.get_trial(result.generated_trial_ids[0])

    assert result.errors == []
    assert len(result.generated_trial_ids) == 1
    assert created_trial is not None
    assert "return self.network(x) * 0.5" in created_trial.source


def test_reconcile_rejects_search_replace_mutations_outside_evolve_blocks(repository, dataset_manager):
    class InvalidPatchGenerator:
        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            return type(
                "Generated",
                (),
                {
                    "source": """<<<<<<< SEARCH
import json
=======
import json
IMMUTABLE_BREAK = True
>>>>>>> REPLACE
""",
                    "provenance_json": {
                        **make_llm_provenance(model="invalid-patch"),
                    },
                },
            )()

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    system = EvolutionSystem(repository, dataset_manager, InvalidPatchGenerator(), RecordingLauncherDouble(), runner)
    track = system.create_track("invalid-patch-mutation", "mnist:v1", {})

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
    assert all(error.startswith("invalid_mutation:") for error in result.errors)
    assert len(result.failed_generation_trial_ids) == 2
    failed_trials = [repository.get_trial(trial_id) for trial_id in result.failed_generation_trial_ids]
    assert all(trial is not None for trial in failed_trials)
    assert all(trial.outcome_reason == "generation_failed" for trial in failed_trials if trial is not None)
    assert all(trial.provenance_json["generation"]["assertions_passed"] is False for trial in failed_trials if trial is not None)


def test_reconcile_persists_generation_failed_trial_when_backend_returns_error(repository, dataset_manager):
    class ProviderFailureGenerator:
        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            return GenerationResult(
                source=None,
                provenance_json=make_llm_provenance(model="provider-failure"),
                error_info={"reason": "provider_request_failed", "detail": "upstream timeout"},
            )

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    system = EvolutionSystem(repository, dataset_manager, ProviderFailureGenerator(), RecordingLauncherDouble(), runner)
    track = system.create_track("provider-failure", "mnist:v1", {})

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
    assert len(result.failed_generation_trial_ids) == 2
    failed_trials = [repository.get_trial(trial_id) for trial_id in result.failed_generation_trial_ids]
    assert all(trial is not None for trial in failed_trials)
    assert all(trial.outcome_reason == "generation_failed" for trial in failed_trials if trial is not None)
    assert all(trial.error_json["reason"] == "provider_request_failed" for trial in failed_trials if trial is not None)
    assert all(trial.error_json["detail"] == "upstream timeout" for trial in failed_trials if trial is not None)


def test_reconcile_persists_generation_failed_trial_when_response_cannot_materialize(repository, dataset_manager):
    class UnmaterializableGenerator:
        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            return GenerationResult(
                source="not python and not a valid patch",
                provenance_json=make_llm_provenance(model="bad-response"),
            )

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    system = EvolutionSystem(repository, dataset_manager, UnmaterializableGenerator(), RecordingLauncherDouble(), runner)
    track = system.create_track("bad-response", "mnist:v1", {})

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
    assert len(result.failed_generation_trial_ids) == 2
    failed_trials = [repository.get_trial(trial_id) for trial_id in result.failed_generation_trial_ids]
    assert all(trial is not None for trial in failed_trials)
    assert all(trial.outcome_reason == "generation_failed" for trial in failed_trials if trial is not None)
    assert all(trial.error_json["reason"] == "candidate_materialization_failed" for trial in failed_trials if trial is not None)


def test_reconcile_tags_length_limited_invalid_candidate_as_truncation(repository, dataset_manager):
    class TruncatedGenerator:
        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            return GenerationResult(
                source="<<<<<<< SEARCH\npartial patch",
                provenance_json=make_llm_provenance(
                    model="truncated-response",
                    generation={
                        "response_text": "<<<<<<< SEARCH\npartial patch",
                        "finish_reason": "length",
                        "native_finish_reason": "length",
                    },
                ),
            )

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    system = EvolutionSystem(repository, dataset_manager, TruncatedGenerator(), RecordingLauncherDouble(), runner)
    track = system.create_track("truncated-response", "mnist:v1", {})

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
    assert len(result.failed_generation_trial_ids) == 2
    failed_trials = [repository.get_trial(trial_id) for trial_id in result.failed_generation_trial_ids]
    assert all(trial is not None for trial in failed_trials)
    assert all(trial.outcome_reason == "generation_failed" for trial in failed_trials if trial is not None)
    assert all(trial.error_json["reason"] == "candidate_materialization_failed" for trial in failed_trials if trial is not None)
    assert all(trial.error_json["finish_reason"] == "length" for trial in failed_trials if trial is not None)
    assert all(trial.error_json["error_type"] == "generation_output_truncated" for trial in failed_trials if trial is not None)


def test_reconcile_persists_generation_failed_trial_when_api_key_missing(repository, dataset_manager, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    generator = OpenRouterGenerationBackend(api_key=None)
    system = EvolutionSystem(repository, dataset_manager, generator, RecordingLauncherDouble(), runner)
    track = system.create_track("missing-api-key", "mnist:v1", {})

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
    assert len(result.failed_generation_trial_ids) == 2
    failed_trials = [repository.get_trial(trial_id) for trial_id in result.failed_generation_trial_ids]
    assert all(trial is not None for trial in failed_trials)
    assert all(trial.error_json["reason"] == "missing_api_key" for trial in failed_trials if trial is not None)


def test_reconcile_generates_requested_candidates_in_parallel(repository, dataset_manager):
    class ParallelTrackingGenerator:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            del track, dataset_manifest, context_trials, negative_trials, duplicate_retry_count
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.05)
                return type(
                    "Generated",
                    (),
                    {
                        "source": build_candidate_train_script(
                            build_model_block(
                                f"""
def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1) + {generation_index}
    return torch.stack((-scores, scores), dim=1)
"""
                            )
                        ),
                        "provenance_json": {
                            **make_llm_provenance(model="parallel-capture"),
                            "generation_index": generation_index,
                            "duplicate_retry_count": 0,
                        },
                    },
                )()
            finally:
                with self.lock:
                    self.active -= 1

    dataset_manager.prepare("mnist:v1")
    repository.register_dataset("mnist:v1", str(dataset_manager.manifest_path_for("mnist:v1")))
    generator = ParallelTrackingGenerator()
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    system = EvolutionSystem(repository, dataset_manager, generator, RecordingLauncherDouble(), runner)
    track = system.create_track("parallel", "mnist:v1", {})

    baseline = repository.list_trials(track.track_id)[0]
    repository.finalize_trial(
        trial_id=baseline.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": 0.5},
        score=0.5,
        error_info=None,
    )

    result = system.reconcile_track(track.track_id, ready_queue_threshold=2, max_parallelism=0)

    assert len(result.generated_trial_ids) == 2
    assert generator.max_active == 2


def test_reconcile_launches_first_ready_candidate_before_slower_generation_finishes(repository, dataset_manager):
    class StaggeredGenerator:
        def __init__(self):
            self.finished_at: dict[int, float] = {}
            self.fast_started = threading.Event()

        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            del track, dataset_manifest, context_trials, negative_trials, duplicate_retry_count
            if generation_index == 1:
                self.fast_started.set()
                time.sleep(0.02)
            else:
                self.fast_started.wait(timeout=1.0)
                time.sleep(0.20)
            self.finished_at[generation_index] = time.monotonic()
            return type(
                "Generated",
                (),
                {
                    "source": build_candidate_train_script(
                        build_model_block(
                            f"""
def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1) + {generation_index}
    return torch.stack((-scores, scores), dim=1)
"""
                        )
                    ),
                    "provenance_json": make_llm_provenance(model="staggered-generator"),
                },
            )()

    class TimedLauncher:
        def __init__(self):
            self.launch_times: dict[str, float] = {}

        def launch_trial(self, trial_id: str, dispatch_token: str, launch_policy: dict[str, object] | None = None):
            del dispatch_token, launch_policy
            self.launch_times[trial_id] = time.monotonic()
            return None

    _prepare_repo_dataset(repository, dataset_manager)
    generator = StaggeredGenerator()
    launcher = TimedLauncher()
    system, _ = _build_system(repository, dataset_manager, generator, launcher)
    track = system.create_track("early-dispatch", "mnist:v1", {})
    _finalize_baseline_success(repository, track.track_id)

    result = system.reconcile_track(track.track_id, ready_queue_threshold=2, max_parallelism=1)

    assert len(result.generated_trial_ids) == 2
    assert len(result.launched_trial_ids) == 1
    launched_at = launcher.launch_times[result.launched_trial_ids[0]]
    assert launched_at < generator.finished_at[2]


def test_controller_relaunches_requeued_dispatch_while_generation_is_still_running(repository, dataset_manager):
    class BlockingGenerator:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()

        def generate(
            self,
            track,
            dataset_manifest,
            context_trials,
            negative_trials=None,
            generation_index=0,
            duplicate_retry_count=0,
        ):
            del track, dataset_manifest, context_trials, negative_trials, generation_index, duplicate_retry_count
            self.started.set()
            self.release.wait(timeout=2.0)
            self.finished.set()
            return type(
                "Generated",
                (),
                {
                    "source": build_candidate_train_script(
                        build_model_block(
                            """
def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1) + 5
    return torch.stack((-scores, scores), dim=1)
"""
                        )
                    ),
                    "provenance_json": make_llm_provenance(model="blocking-generator"),
                },
            )()

    class RelaunchRecordingLauncher:
        def __init__(self):
            self.launched: list[str] = []
            self.relaunched = threading.Event()

        def launch_trial(self, trial_id: str, dispatch_token: str, launch_policy: dict[str, object] | None = None):
            del dispatch_token, launch_policy
            self.launched.append(trial_id)
            self.relaunched.set()
            return None

    _prepare_repo_dataset(repository, dataset_manager)
    generator = BlockingGenerator()
    launcher = RelaunchRecordingLauncher()
    system, _ = _build_system(repository, dataset_manager, generator, launcher)
    track = system.create_track("stale-relaunch", "mnist:v1", {"dispatch_ttl_sec": 0, "max_dispatch_retries": 2})
    _finalize_baseline_success(repository, track.track_id)

    queued_trial, created = repository.create_queued_trial_if_absent(
        track.track_id,
        build_candidate_train_script(
            build_model_block(
                """
def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1) + 9
    return torch.stack((-scores, scores), dim=1)
"""
            )
        ),
        make_llm_provenance(model="queued-before-controller"),
    )
    assert created is True
    assert queued_trial is not None
    repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=0, limit=1)

    controller = system.start_track_controller(track.track_id, max_parallelism=2)
    try:
        assert generator.started.wait(timeout=1.0)
        assert launcher.relaunched.wait(timeout=1.0)
        assert queued_trial.trial_id in launcher.launched
        assert generator.finished.is_set() is False
    finally:
        generator.release.set()
        controller.stop()


def test_reconcile_uses_launch_executor_so_blocking_launches_do_not_block_other_launches(repository, dataset_manager):
    class BlockingLauncher:
        def __init__(self):
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.release_first = threading.Event()
            self.lock = threading.Lock()
            self.launch_order: list[str] = []

        def launch_trial(self, trial_id: str, dispatch_token: str, launch_policy: dict[str, object] | None = None):
            del dispatch_token, launch_policy
            with self.lock:
                self.launch_order.append(trial_id)
                launch_number = len(self.launch_order)
            if launch_number == 1:
                self.first_started.set()
                self.release_first.wait(timeout=2.0)
            else:
                self.second_started.set()
            return None

    _prepare_repo_dataset(repository, dataset_manager)
    launcher = BlockingLauncher()
    system, _ = _build_system(
        repository,
        dataset_manager,
        FixedGenerationBackend(source=build_baseline_train_script()),
        launcher,
    )
    track = system.create_track("blocking-launch", "mnist:v1", {})
    second_trial, created = repository.create_queued_trial_if_absent(
        track.track_id,
        build_candidate_train_script(
            build_model_block(
                """
def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1) + 11
    return torch.stack((-scores, scores), dim=1)
"""
            )
        ),
        make_llm_provenance(model="second-launchable"),
    )
    assert created is True
    assert second_trial is not None

    result_holder: dict[str, object] = {}

    def run_reconcile():
        result_holder["result"] = system.reconcile_track(track.track_id, ready_queue_threshold=0, max_parallelism=2)

    thread = threading.Thread(target=run_reconcile)
    thread.start()
    try:
        assert launcher.first_started.wait(timeout=1.0)
        assert launcher.second_started.wait(timeout=1.0)
    finally:
        launcher.release_first.set()
        thread.join(timeout=2.0)

    assert thread.is_alive() is False
    result = result_holder["result"]
    assert len(result.launched_trial_ids) == 2


def test_reconcile_persists_launcher_metadata_for_launched_trials(repository, dataset_manager):
    class MetadataLauncher:
        def launch_trial(self, trial_id: str, dispatch_token: str, launch_policy: dict[str, object] | None = None):
            del dispatch_token, launch_policy
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
        FixedGenerationBackend(source=build_baseline_train_script()),
        MetadataLauncher(),
        runner,
    )
    track = system.create_track("launch-metadata", "mnist:v1", {})

    queued_trial = repository.list_trials(track.track_id)[0]
    result = system.reconcile_track(track.track_id, ready_queue_threshold=0, max_parallelism=1)
    updated_trial = repository.get_trial(queued_trial.trial_id)

    assert result.launched_trial_ids == [queued_trial.trial_id]
    assert updated_trial is not None
    assert updated_trial.provenance_json["launcher"] == {
        "kind": "modal",
        "run_id": f"fc-{queued_trial.trial_id}",
        "run_url": f"https://modal.com/apps/test/runs/{queued_trial.trial_id}",
    }
