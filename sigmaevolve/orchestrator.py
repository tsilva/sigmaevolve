from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from sigmaevolve.models import ReconcileResult


class RunnerLauncher(Protocol):
    def launch_trial(self, trial_id: str, dispatch_token: str) -> None:
        ...


class RecordingLauncher:
    def __init__(self) -> None:
        self.launched: list[tuple[str, str]] = []

    def launch_trial(self, trial_id: str, dispatch_token: str) -> None:
        self.launched.append((trial_id, dispatch_token))


class InlineRunnerLauncher:
    def __init__(self, runner_service, runner_id_prefix: str = "inline") -> None:
        self.runner_service = runner_service
        self.runner_id_prefix = runner_id_prefix
        self.launch_count = 0

    def launch_trial(self, trial_id: str, dispatch_token: str) -> None:
        self.launch_count += 1
        runner_id = f"{self.runner_id_prefix}_{self.launch_count}"
        self.runner_service.run_reserved_trial(trial_id, dispatch_token, runner_id)


class ModalRemoteLauncher:
    def __init__(self, modal_function) -> None:
        self.modal_function = modal_function

    def launch_trial(self, trial_id: str, dispatch_token: str) -> None:
        self.modal_function.spawn(trial_id=trial_id, dispatch_token=dispatch_token)


class Orchestrator:
    def __init__(self, repository, dataset_manager, generator, launcher) -> None:
        self.repository = repository
        self.dataset_manager = dataset_manager
        self.generator = generator
        self.launcher = launcher

    def reconcile_track(self, track_id: str) -> ReconcileResult:
        track = self.repository.get_track(track_id)
        if track is None:
            raise KeyError(f"Track not found: {track_id}")
        policy = track.policy_json
        result = ReconcileResult()

        requeued, stale_dispatch = self.repository.sweep_expired_dispatches(
            track_id=track_id,
            max_dispatch_retries=int(policy["max_dispatch_retries"]),
        )
        stale_active = self.repository.sweep_stale_active_trials(
            track_id=track_id,
            stale_ttl_sec=int(policy["stale_ttl_sec"]),
        )
        result = replace(
            result,
            requeued_trial_ids=requeued,
            stale_trial_ids=stale_dispatch + stale_active,
        )

        queue_count = self.repository.count_trials(track_id, statuses={"queued"})
        if queue_count < int(policy["ready_queue_threshold"]):
            dataset_manifest = self.dataset_manager.verify(track.dataset_id)
            context_trials = self.repository.sample_trial_context(track_id, limit=5)
            if context_trials:
                try:
                    generated = self.generator.generate(track, dataset_manifest, context_trials)
                    trial, created = self.repository.create_queued_trial_if_absent(
                        track_id=track_id,
                        source=generated.source,
                        provenance_json=generated.provenance_json,
                    )
                    if created and trial is not None:
                        result.generated_trial_ids.append(trial.trial_id)
                    elif trial is not None:
                        result.duplicate_hashes.append(trial.script_hash)
                except Exception as exc:
                    result.errors.append(str(exc))

        reserved = self.repository.reserve_trials(
            track_id=track_id,
            max_parallelism=int(policy["max_parallelism"]),
            dispatch_ttl_sec=int(policy["dispatch_ttl_sec"]),
            limit=int(policy["max_parallelism"]),
        )
        for trial in reserved:
            try:
                self.launcher.launch_trial(trial.trial_id, trial.dispatch_token or "")
                result.launched_trial_ids.append(trial.trial_id)
            except Exception as exc:
                result.errors.append(f"launch_failed:{trial.trial_id}:{exc}")
        return result
