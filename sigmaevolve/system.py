from __future__ import annotations

from pathlib import Path

from sigmaevolve.baseline import build_baseline_linear_classifier
from sigmaevolve.datasets import DatasetManager, TorchvisionClassificationProvider
from sigmaevolve.generation import OpenRouterGenerationBackend
from sigmaevolve.models import (
    CANDIDATE_KIND_STRATEGY_V1,
    DatasetRecord,
    MigrationResult,
    TrackPolicy,
    TrackRecord,
    TrialRecord,
    TrialSummary,
)
from sigmaevolve.orchestrator import Orchestrator, RecordingLauncher
from sigmaevolve.runner import RunnerService
from sigmaevolve.scoring import compute_score
from sigmaevolve.storage import SQLAlchemyRepository


class EvolutionSystem:
    def __init__(
        self,
        repository: SQLAlchemyRepository,
        dataset_manager: DatasetManager,
        generator,
        launcher,
        runner_service: RunnerService,
    ) -> None:
        self.repository = repository
        self.dataset_manager = dataset_manager
        self.generator = generator
        self.launcher = launcher
        self.runner_service = runner_service
        self.orchestrator = Orchestrator(repository, dataset_manager, generator, launcher)

    def prepare_dataset(self, dataset_id: str) -> DatasetRecord:
        manifest = self.dataset_manager.prepare(dataset_id)
        return self.repository.register_dataset(dataset_id=dataset_id, manifest_path=str(Path(manifest.root_dir) / "manifest.json"))

    def create_track(self, name: str | None, dataset_id: str, policy_json: dict) -> TrackRecord:
        if self.repository.get_dataset(dataset_id) is None:
            raise KeyError(f"Dataset must be prepared before track creation: {dataset_id}")
        policy = TrackPolicy.from_dict(policy_json)
        track = self.repository.create_track(name=name, dataset_id=dataset_id, policy_json=policy.to_dict())
        baseline_source = build_baseline_linear_classifier()
        self.repository.create_queued_trial_if_absent(
            track_id=track.track_id,
            source=baseline_source,
            provenance_json={
                "backend": "baseline",
                "model": "linear-classifier",
                "candidate_kind": CANDIDATE_KIND_STRATEGY_V1,
                "parent_trial_ids": [],
            },
        )
        return track

    def reconcile_track(self, track_id: str):
        return self.orchestrator.reconcile_track(track_id)

    def sample_trial_context(self, track_id: str, limit: int) -> list[TrialSummary]:
        return self.repository.sample_trial_context(track_id=track_id, limit=limit)

    def claim_trial(self, trial_id: str, dispatch_token: str, runner_id: str) -> TrialRecord | None:
        return self.repository.claim_trial(trial_id=trial_id, dispatch_token=dispatch_token, runner_id=runner_id)

    def heartbeat_trial(self, trial_id: str, runner_id: str, meta: dict) -> None:
        self.repository.heartbeat_trial(trial_id=trial_id, runner_id=runner_id, meta=meta)

    def finalize_trial(
        self,
        trial_id: str,
        runner_id: str,
        outcome_reason: str,
        metrics: dict | None,
        score: float,
        error_info: dict | None,
    ) -> None:
        self.repository.finalize_trial(
            trial_id=trial_id,
            runner_id=runner_id,
            outcome_reason=outcome_reason,
            metrics=metrics,
            score=score,
            error_info=error_info,
        )

    def rescore(self, track_or_all, scorer_config: dict) -> MigrationResult:
        track_id = None if track_or_all == "all" else track_or_all
        return self.repository.rescore(track_id=track_id, scorer_config=scorer_config)


def build_system(
    database_url: str,
    dataset_root: str | Path,
    openrouter_api_key: str | None = None,
    providers: dict | None = None,
    launcher=None,
) -> EvolutionSystem:
    dataset_root = Path(dataset_root)
    providers = providers or {
        "mnist:v1": TorchvisionClassificationProvider("mnist"),
        "fashion_mnist:v1": TorchvisionClassificationProvider("fashion_mnist"),
    }
    repository = SQLAlchemyRepository(database_url)
    dataset_manager = DatasetManager(dataset_root=dataset_root, providers=providers)
    generator = OpenRouterGenerationBackend(api_key=openrouter_api_key)
    runner_service = RunnerService(repository=repository, dataset_manager=dataset_manager)
    launcher = launcher or RecordingLauncher()
    return EvolutionSystem(
        repository=repository,
        dataset_manager=dataset_manager,
        generator=generator,
        launcher=launcher,
        runner_service=runner_service,
    )
