from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Protocol

from sigmaevolve.evolve_blocks import EvolveBlockError, assert_only_evolve_blocks_changed, materialize_candidate_source
from sigmaevolve.models import CANDIDATE_KIND_STRATEGY_V1, ReconcileResult, TrialSummary


class RunnerLauncher(Protocol):
    def launch_trial(self, trial_id: str, dispatch_token: str) -> dict[str, Any] | None:
        ...


class RecordingLauncher:
    def __init__(self) -> None:
        self.launched: list[tuple[str, str]] = []

    def launch_trial(self, trial_id: str, dispatch_token: str) -> dict[str, Any] | None:
        self.launched.append((trial_id, dispatch_token))
        return None


class InlineRunnerLauncher:
    def __init__(self, runner_service, runner_id_prefix: str = "inline") -> None:
        self.runner_service = runner_service
        self.runner_id_prefix = runner_id_prefix
        self.launch_count = 0

    def launch_trial(self, trial_id: str, dispatch_token: str) -> dict[str, Any] | None:
        self.launch_count += 1
        runner_id = f"{self.runner_id_prefix}_{self.launch_count}"
        self.runner_service.run_reserved_trial(trial_id, dispatch_token, runner_id)
        return None


class ModalRemoteLauncher:
    def __init__(self, modal_function) -> None:
        self.modal_function = modal_function

    def launch_trial(self, trial_id: str, dispatch_token: str) -> dict[str, Any] | None:
        function_call = self.modal_function.spawn(trial_id=trial_id, dispatch_token=dispatch_token)
        metadata: dict[str, Any] = {"kind": "modal"}
        object_id = getattr(function_call, "object_id", None)
        if isinstance(object_id, str) and object_id:
            metadata["run_id"] = object_id
        get_dashboard_url = getattr(function_call, "get_dashboard_url", None)
        if callable(get_dashboard_url):
            try:
                run_url = get_dashboard_url()
            except Exception:
                run_url = None
            if isinstance(run_url, str) and run_url:
                metadata["run_url"] = run_url
        return metadata if len(metadata) > 1 else None


class Orchestrator:
    MAX_DUPLICATE_RETRIES = 3

    def __init__(self, repository, dataset_manager, generator, launcher) -> None:
        self.repository = repository
        self.dataset_manager = dataset_manager
        self.generator = generator
        self.launcher = launcher

    def _sample_successful_context_trials(self, track_id: str, sampling_settings: dict, generation_index: int) -> list[TrialSummary]:
        candidates = self.repository.sample_trial_context(
            track_id,
            limit=self.repository.count_trials(track_id),
            candidate_kind=CANDIDATE_KIND_STRATEGY_V1,
        )
        if not candidates:
            return []
        if len(candidates) == 1:
            return [candidates[0]]

        seed = int(sampling_settings.get("seed", 0))
        rng = random.Random(seed + generation_index)
        weights = [max(float(trial.score), 0.0) for trial in candidates]
        if sum(weights) <= 0.0:
            return [rng.choice(candidates)]
        return [rng.choices(candidates, weights=weights, k=1)[0]]

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
            generation_index = self.repository.count_trials(track_id)
            context_trials = self._sample_successful_context_trials(
                track_id,
                policy.get("sampling_settings", {}),
                generation_index,
            )
            if context_trials:
                for duplicate_retry_count in range(self.MAX_DUPLICATE_RETRIES + 1):
                    try:
                        generated = self.generator.generate(
                            track,
                            dataset_manifest,
                            context_trials,
                            negative_trials=[],
                            generation_index=generation_index,
                            duplicate_retry_count=duplicate_retry_count,
                        )
                        try:
                            candidate_source = materialize_candidate_source(context_trials[0].source, generated.source)
                            assert_only_evolve_blocks_changed(context_trials[0].source, candidate_source)
                        except EvolveBlockError as exc:
                            result.errors.append(f"invalid_mutation:{exc}")
                            continue
                        trial, created = self.repository.create_queued_trial_if_absent(
                            track_id=track_id,
                            source=candidate_source,
                            provenance_json=generated.provenance_json,
                        )
                        if created and trial is not None:
                            result.generated_trial_ids.append(trial.trial_id)
                            break
                        if trial is not None:
                            result.duplicate_hashes.append(trial.script_hash)
                    except Exception as exc:
                        result.errors.append(str(exc))
                        break

        reserved = self.repository.reserve_trials(
            track_id=track_id,
            max_parallelism=int(policy["max_parallelism"]),
            dispatch_ttl_sec=int(policy["dispatch_ttl_sec"]),
            limit=int(policy["max_parallelism"]),
        )
        for trial in reserved:
            try:
                launch_metadata = self.launcher.launch_trial(trial.trial_id, trial.dispatch_token or "")
                if launch_metadata:
                    self.repository.record_trial_launcher_metadata(trial.trial_id, launch_metadata)
                result.launched_trial_ids.append(trial.trial_id)
            except Exception as exc:
                result.errors.append(f"launch_failed:{trial.trial_id}:{exc}")
        return result
