from __future__ import annotations

import random
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from sigmaevolve.controller import TrackController
from sigmaevolve.evolve_blocks import EvolveBlockError, assert_only_evolve_blocks_changed, materialize_candidate_source
from sigmaevolve.hashing import compute_script_hash
from sigmaevolve.models import (
    CANDIDATE_KIND_STRATEGY_V1,
    GenerationResult,
    OUTCOME_DUPLICATE,
    OUTCOME_GENERATION_FAILED,
    ReconcileResult,
    TrialSummary,
)


class RunnerLauncher(Protocol):
    def launch_trial(
        self,
        trial_id: str,
        dispatch_token: str,
        launch_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        ...

    def cancel_run(self, launcher_metadata: dict[str, Any]) -> None:
        ...


class RecordingLauncher:
    def __init__(self) -> None:
        self.launched: list[tuple[str, str]] = []

    def launch_trial(
        self,
        trial_id: str,
        dispatch_token: str,
        launch_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        del launch_policy
        self.launched.append((trial_id, dispatch_token))
        return None

    def cancel_run(self, launcher_metadata: dict[str, Any]) -> None:
        del launcher_metadata


class InlineRunnerLauncher:
    def __init__(self, runner_service, runner_id_prefix: str = "inline") -> None:
        self.runner_service = runner_service
        self.runner_id_prefix = runner_id_prefix
        self.launch_count = 0

    def launch_trial(
        self,
        trial_id: str,
        dispatch_token: str,
        launch_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        del launch_policy
        self.launch_count += 1
        runner_id = f"{self.runner_id_prefix}_{self.launch_count}"
        self.runner_service.run_reserved_trial(trial_id, dispatch_token, runner_id)
        return None

    def cancel_run(self, launcher_metadata: dict[str, Any]) -> None:
        del launcher_metadata


class ModalRemoteLauncher:
    def __init__(self, modal_function) -> None:
        self.modal_function = modal_function

    def launch_trial(
        self,
        trial_id: str,
        dispatch_token: str,
        launch_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        requested_gpus = (launch_policy or {}).get("modal_gpu_preferences")
        if requested_gpus is None:
            attempts: list[str | None] = [None]
        elif isinstance(requested_gpus, list) and requested_gpus:
            attempts = [str(gpu) for gpu in requested_gpus]
        else:
            raise ValueError("Track launch policy modal_gpu_preferences must be null or a non-empty list.")

        failures: list[str] = []
        attempted_gpus: list[str] = []
        for gpu in attempts:
            if gpu is not None:
                attempted_gpus.append(gpu)
            try:
                spawn_result = self.modal_function.spawn(
                    trial_id=trial_id,
                    dispatch_token=dispatch_token,
                    gpu=gpu,
                )
            except Exception as exc:
                failures.append(f"{gpu or 'cpu'}: {exc}")
                continue
            function_call = getattr(spawn_result, "function_call", spawn_result)
            effective_gpu = getattr(spawn_result, "effective_gpu", gpu)

            metadata: dict[str, Any] = {
                "kind": "modal",
                "gpu_attempts": list(attempted_gpus),
            }
            if effective_gpu is not None:
                metadata["gpu_selected"] = effective_gpu
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
            return metadata

        raise RuntimeError("Modal launch failed for all configured resources: " + "; ".join(failures))

    def cancel_run(self, launcher_metadata: dict[str, Any]) -> None:
        run_id = launcher_metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Modal cancellation requires launcher_metadata.run_id.")
        self.modal_function.cancel(run_id)


@dataclass(frozen=True)
class GenerationAttempt:
    slot_index: int
    generation_index: int
    duplicate_retry_count: int
    context_trials: list[TrialSummary]


class Orchestrator:
    GENERATION_FAILURE_LIMIT_MULTIPLIER = 2

    def __init__(self, repository, dataset_manager, generator, launcher) -> None:
        self.repository = repository
        self.dataset_manager = dataset_manager
        self.generator = generator
        self.launcher = launcher

    def _emit(self, reporter: Callable[[str, dict[str, Any]], None] | None, event: str, **payload: Any) -> None:
        if reporter is not None:
            reporter(event, payload)

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
        remaining = list(candidates)
        remaining_weights = [max(float(trial.score), 0.0) for trial in remaining]
        sampled: list[TrialSummary] = []

        for _ in range(min(2, len(remaining))):
            total_weight = sum(remaining_weights)
            if total_weight <= 0.0:
                selected_index = rng.randrange(len(remaining))
            else:
                selected_index = rng.choices(range(len(remaining)), weights=remaining_weights, k=1)[0]
            sampled.append(remaining.pop(selected_index))
            remaining_weights.pop(selected_index)

        candidate_ranks = {trial.trial_id: index for index, trial in enumerate(candidates)}
        sampled.sort(key=lambda trial: (-float(trial.score), candidate_ranks[trial.trial_id]))
        return sampled

    def _sample_generation_context_trials(self, track_id: str, sampling_settings: dict, generation_index: int) -> list[TrialSummary]:
        successful_context = self._sample_successful_context_trials(track_id, sampling_settings, generation_index)
        if successful_context:
            return successful_context

        if self.repository.sample_trial_context(track_id, limit=self.repository.count_trials(track_id)):
            return []

        # Cold-start tracks only have the system-seeded baseline. Use it as the
        # first parent before any scored results exist.
        for trial in self.repository.list_trials(track_id):
            provenance = dict(trial.provenance_json or {})
            if provenance.get("backend") != "baseline":
                continue
            return [
                TrialSummary(
                    trial_id=trial.trial_id,
                    score=float(trial.score or 0.0),
                    metrics_json=dict(trial.metrics_json) if trial.metrics_json else None,
                    source=trial.source,
                    provenance_json=provenance,
                    outcome_reason=trial.outcome_reason,
                    error_json=dict(trial.error_json) if trial.error_json else None,
                )
            ]
        return []

    def _with_generation_trace(
        self,
        provenance_json: dict[str, Any],
        *,
        generated_source: str | None,
        assertions_passed: bool,
        assertion_failures: list[str],
        candidate_hash: str | None,
    ) -> dict[str, Any]:
        payload = dict(provenance_json or {})
        generation_payload = dict(payload.get("generation") or {})
        request_messages = payload.get("request_messages")
        if isinstance(request_messages, list):
            if "system_prompt" not in generation_payload and request_messages:
                first = request_messages[0]
                if isinstance(first, dict) and isinstance(first.get("content"), str):
                    generation_payload["system_prompt"] = first["content"]
            if "user_prompt" not in generation_payload and len(request_messages) > 1:
                second = request_messages[1]
                if isinstance(second, dict) and isinstance(second.get("content"), str):
                    generation_payload["user_prompt"] = second["content"]
        generation_payload.setdefault("response_text", None)
        generation_payload["generated_source"] = generated_source
        generation_payload["assertions_passed"] = assertions_passed
        generation_payload["assertion_failures"] = list(assertion_failures)
        generation_payload["candidate_hash"] = candidate_hash
        payload["generation"] = generation_payload
        return payload

    def _normalize_generation_result(
        self,
        generated: Any,
    ) -> GenerationResult:
        if isinstance(generated, GenerationResult):
            return generated
        return GenerationResult(
            source=getattr(generated, "source", None),
            provenance_json=dict(getattr(generated, "provenance_json", {}) or {}),
            error_info=dict(getattr(generated, "error_info", {}) or {}) or None,
        )

    def _fallback_generation_provenance(
        self,
        track,
        context_trials: list[TrialSummary],
        *,
        generation_index: int,
        duplicate_retry_count: int,
    ) -> dict[str, Any]:
        generation_backend = dict(track.policy_json.get("generation_backend", {}))
        model = generation_backend.get("model")
        if not isinstance(model, str) or not model:
            model_pool = generation_backend.get("model_pool")
            if isinstance(model_pool, list) and model_pool and isinstance(model_pool[0], dict):
                pool_model = model_pool[0].get("model")
                model = str(pool_model) if pool_model else "unknown"
            else:
                model = "unknown"
        system_prompt = "Generation backend failed before prompts could be fully recorded."
        user_prompt = "No user prompt was captured because generation aborted before the provider call completed."
        return {
            "backend": "openrouter",
            "model": model,
            "candidate_kind": CANDIDATE_KIND_STRATEGY_V1,
            "generation_config": generation_backend,
            "generation_index": generation_index,
            "duplicate_retry_count": duplicate_retry_count,
            "request_messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "context_trial_ids": [trial.trial_id for trial in context_trials],
            "generation": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_text": None,
                "generated_source": None,
                "assertions_passed": False,
                "assertion_failures": [],
                "candidate_hash": None,
            },
        }

    def _record_generation_attempt_failure(
        self,
        track_id: str,
        result: ReconcileResult,
        provenance_json: dict[str, Any],
        *,
        reason: str,
        detail: str | None = None,
        generated_source: str | None = None,
        candidate_hash: str | None = None,
        extra_error_json: dict[str, Any] | None = None,
        result_error: str | None = None,
    ) -> ReconcileResult:
        assertion_failures = [detail] if detail else [reason]
        final_provenance = self._with_generation_trace(
            provenance_json,
            generated_source=generated_source,
            assertions_passed=False,
            assertion_failures=assertion_failures,
            candidate_hash=candidate_hash,
        )
        error_payload: dict[str, Any] = {"reason": reason}
        generation_payload = dict(final_provenance.get("generation") or {})
        if detail:
            error_payload["detail"] = detail
        finish_reason = generation_payload.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            error_payload["finish_reason"] = finish_reason
        native_finish_reason = generation_payload.get("native_finish_reason")
        if isinstance(native_finish_reason, str) and native_finish_reason:
            error_payload["native_finish_reason"] = native_finish_reason
        if extra_error_json:
            error_payload.update(extra_error_json)
        trial = self.repository.create_generation_attempt_trial(
            track_id=track_id,
            provenance_json=final_provenance,
            outcome_reason=OUTCOME_GENERATION_FAILED,
            error_json=error_payload,
        )
        result.failed_generation_trial_ids.append(trial.trial_id)
        result.errors.append(result_error or f"generation_failed:{reason}")
        return result

    def _schedule_generation_attempt(
        self,
        executor: ThreadPoolExecutor,
        track,
        dataset_manifest,
        sampling_settings: dict[str, Any],
        *,
        slot_index: int,
        generation_index: int,
        duplicate_retry_count: int,
    ) -> tuple[Future[Any], GenerationAttempt] | None:
        context_trials = self._sample_generation_context_trials(
            track.track_id,
            sampling_settings,
            generation_index,
        )
        if not context_trials:
            return None
        attempt = GenerationAttempt(
            slot_index=slot_index,
            generation_index=generation_index,
            duplicate_retry_count=duplicate_retry_count,
            context_trials=context_trials,
        )
        future = executor.submit(
            self.generator.generate,
            track,
            dataset_manifest,
            context_trials,
            [],
            generation_index,
            duplicate_retry_count,
        )
        return future, attempt

    def _record_duplicate_generation_attempt(
        self,
        track_id: str,
        result: ReconcileResult,
        provenance_json: dict[str, Any],
        *,
        candidate_hash: str,
        trial_id: str,
    ) -> ReconcileResult:
        duplicate_trial = self.repository.create_generation_attempt_trial(
            track_id=track_id,
            provenance_json=provenance_json,
            outcome_reason=OUTCOME_DUPLICATE,
            error_json={
                "reason": "duplicate_candidate",
                "detail": f"Candidate source already exists as {trial_id}.",
                "candidate_hash": candidate_hash,
                "existing_trial_id": trial_id,
            },
        )
        result.duplicate_hashes.append(candidate_hash)
        result.duplicate_trial_ids.append(duplicate_trial.trial_id)
        return result

    def _materialize_candidate_source(self, parent_source: str, generated_source: str) -> str:
        candidate_source = materialize_candidate_source(parent_source, generated_source)
        assert_only_evolve_blocks_changed(parent_source, candidate_source)
        return candidate_source

    def _accept_generated_candidate(
        self,
        *,
        track_id: str,
        result: ReconcileResult,
        generated: GenerationResult,
        attempt: GenerationAttempt,
        candidate_source: str,
    ) -> dict[str, Any]:
        candidate_hash = compute_script_hash(candidate_source)
        final_provenance = self._with_generation_trace(
            generated.provenance_json,
            generated_source=candidate_source,
            assertions_passed=True,
            assertion_failures=[],
            candidate_hash=candidate_hash,
        )
        trial, created = self.repository.create_queued_trial_if_absent(
            track_id=track_id,
            source=candidate_source,
            provenance_json=final_provenance,
        )
        if created and trial is not None:
            result.generated_trial_ids.append(trial.trial_id)
            return {
                "event": "generation_accepted",
                "payload": {"trial_id": trial.trial_id},
            }

        if trial is None:
            raise RuntimeError("Queued trial creation returned no trial record.")

        self._record_duplicate_generation_attempt(
            track_id=track_id,
            result=result,
            provenance_json=final_provenance,
            candidate_hash=candidate_hash,
            trial_id=trial.trial_id,
        )
        return {
            "event": "generation_duplicate",
            "payload": {"existing_trial_id": trial.trial_id},
        }

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
            self,
            track,
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
    ) -> ReconcileResult:
        track = self.repository.get_track(track_id)
        if track is None:
            raise KeyError(f"Track not found: {track_id}")
        ready_queue_threshold = int(ready_queue_threshold)
        max_parallelism = int(max_parallelism)
        if ready_queue_threshold < 0:
            raise ValueError("ready_queue_threshold must be >= 0")
        if max_parallelism < 0:
            raise ValueError("max_parallelism must be >= 0")
        self._emit(
            reporter,
            "reconcile_started",
            track_id=track_id,
            launcher=self.launcher.__class__.__name__,
        )
        initial_queue_count = self.repository.count_trials(track_id, statuses={"queued"})
        if initial_queue_count >= ready_queue_threshold:
            self._emit(
                reporter,
                "queue_fill_skipped",
                queued_count=initial_queue_count,
                target_queue_count=ready_queue_threshold,
            )
        controller = TrackController(
            self,
            track,
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
        self._emit(
            reporter,
            "reconcile_finished",
            generated_count=len(result.generated_trial_ids),
            launched_count=len(result.launched_trial_ids),
            duplicate_count=len(result.duplicate_trial_ids),
            failed_generation_count=len(result.failed_generation_trial_ids),
            error_count=len(result.errors),
        )
        return result
