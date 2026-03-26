from __future__ import annotations

from typing import Any, Callable

from sigmaevolve.controller import TrackController
from sigmaevolve.generation_coordinator import GenerationCoordinator
from sigmaevolve.launchers import InlineRunnerLauncher, ModalRemoteLauncher, RecordingLauncher, RunnerLauncher


def emit_report_event(
    reporter: Callable[[str, dict[str, Any]], None] | None,
    event: str,
    **payload: Any,
) -> None:
    if reporter is not None:
        reporter(event, payload)


class Orchestrator:
    GENERATION_FAILURE_LIMIT_MULTIPLIER = 2

    def __init__(self, repository, dataset_manager, generator, launcher) -> None:
        self.repository = repository
        self.dataset_manager = dataset_manager
        self.generator = generator
        self.launcher = launcher
        self.generation = GenerationCoordinator(repository=repository, generator=generator)

    def _sample_successful_context_trials(
        self,
        track_id: str,
        sampling_settings: dict[str, Any],
        generation_index: int,
    ):
        return self.generation.sample_successful_context_trials(
            track_id,
            sampling_settings,
            generation_index,
        )

    def _sample_generation_context_trials(
        self,
        track_id: str,
        sampling_settings: dict[str, Any],
        generation_index: int,
    ):
        return self.generation.sample_generation_context_trials(
            track_id,
            sampling_settings,
            generation_index,
        )

    def start_track_controller(
        self,
        track_id: str,
        reporter: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        max_parallelism: int,
        ready_queue_threshold: int = 0,
    ) -> TrackController:
        track = self.repository.get_track(track_id)
        if track is None:
            raise KeyError(f"Track not found: {track_id}")
        if max_parallelism < 0:
            raise ValueError("max_parallelism must be >= 0")
        controller = TrackController(
            repository=self.repository,
            dataset_manager=self.dataset_manager,
            generation=self.generation,
            launcher=self.launcher,
            generation_failure_limit_multiplier=self.GENERATION_FAILURE_LIMIT_MULTIPLIER,
            track=track,
            reporter=reporter,
            ready_queue_threshold=int(ready_queue_threshold),
            max_parallelism=int(max_parallelism),
            continuous=True,
        )
        controller.start()
        return controller

    def reconcile_track(
        self,
        track_id: str,
        reporter: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        ready_queue_threshold: int = 1,
        max_parallelism: int = 1,
    ):
        track = self.repository.get_track(track_id)
        if track is None:
            raise KeyError(f"Track not found: {track_id}")
        ready_queue_threshold = int(ready_queue_threshold)
        max_parallelism = int(max_parallelism)
        if ready_queue_threshold < 0:
            raise ValueError("ready_queue_threshold must be >= 0")
        if max_parallelism < 0:
            raise ValueError("max_parallelism must be >= 0")
        emit_report_event(
            reporter,
            "reconcile_started",
            track_id=track_id,
            launcher=self.launcher.__class__.__name__,
        )
        initial_queue_count = self.repository.count_trials(track_id, statuses={"queued"})
        if initial_queue_count >= ready_queue_threshold:
            emit_report_event(
                reporter,
                "queue_fill_skipped",
                queued_count=initial_queue_count,
                target_queue_count=ready_queue_threshold,
            )
        controller = TrackController(
            repository=self.repository,
            dataset_manager=self.dataset_manager,
            generation=self.generation,
            launcher=self.launcher,
            generation_failure_limit_multiplier=self.GENERATION_FAILURE_LIMIT_MULTIPLIER,
            track=track,
            reporter=reporter,
            ready_queue_threshold=ready_queue_threshold,
            max_parallelism=max_parallelism,
            continuous=False,
        )
        controller.start()
        try:
            result = controller.wait_until_one_shot_complete()
        finally:
            controller.stop()
        emit_report_event(
            reporter,
            "reconcile_finished",
            generated_count=len(result.generated_trial_ids),
            launched_count=len(result.launched_trial_ids),
            duplicate_count=len(result.duplicate_trial_ids),
            failed_generation_count=len(result.failed_generation_trial_ids),
            error_count=len(result.errors),
        )
        return result
