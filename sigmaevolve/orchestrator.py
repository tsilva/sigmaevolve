from __future__ import annotations

import random
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol

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
        if detail:
            error_payload["detail"] = detail
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
        context_trials = self._sample_successful_context_trials(
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
        policy = track.policy_json
        ready_queue_threshold = int(ready_queue_threshold)
        max_parallelism = int(max_parallelism)
        if ready_queue_threshold < 0:
            raise ValueError("ready_queue_threshold must be >= 0")
        if max_parallelism < 0:
            raise ValueError("max_parallelism must be >= 0")
        result = ReconcileResult()
        self._emit(
            reporter,
            "reconcile_started",
            track_id=track_id,
            launcher=self.launcher.__class__.__name__,
        )

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
        self._emit(
            reporter,
            "sweep_completed",
            requeued_count=len(requeued),
            stale_count=len(stale_dispatch) + len(stale_active),
        )

        queue_count = self.repository.count_trials(track_id, statuses={"queued"})
        if queue_count < ready_queue_threshold:
            dataset_manifest = self.dataset_manager.verify(track.dataset_id)
            requested_generations = ready_queue_threshold - queue_count
            generation_index_base = self.repository.count_trials(track_id)
            sampling_settings = policy.get("sampling_settings", {})
            max_failures = requested_generations * self.GENERATION_FAILURE_LIMIT_MULTIPLIER
            failures = 0
            completed_slots: set[int] = set()
            pending: dict[Future[Any], GenerationAttempt] = {}
            self._emit(
                reporter,
                "queue_fill_started",
                queued_count=queue_count,
                target_queue_count=ready_queue_threshold,
                requested_generations=requested_generations,
                max_failures=max_failures,
            )

            with ThreadPoolExecutor(max_workers=requested_generations) as executor:
                for slot_index in range(requested_generations):
                    scheduled = self._schedule_generation_attempt(
                        executor,
                        track,
                        dataset_manifest,
                        sampling_settings,
                        slot_index=slot_index,
                        generation_index=generation_index_base + slot_index,
                        duplicate_retry_count=0,
                    )
                    if scheduled is not None:
                        future, attempt = scheduled
                        pending[future] = attempt
                        self._emit(
                            reporter,
                            "generation_scheduled",
                            slot_index=attempt.slot_index,
                            generation_index=attempt.generation_index,
                            duplicate_retry_count=attempt.duplicate_retry_count,
                            in_flight=len(pending),
                        )

                while pending and len(completed_slots) < requested_generations:
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        attempt = pending.pop(future)
                        retry_needed = False

                        try:
                            raw_generated = future.result()
                        except Exception as exc:
                            failures += 1
                            result = self._record_generation_attempt_failure(
                                track_id=track_id,
                                result=result,
                                provenance_json=self._fallback_generation_provenance(
                                    track,
                                    attempt.context_trials,
                                    generation_index=attempt.generation_index,
                                    duplicate_retry_count=attempt.duplicate_retry_count,
                                ),
                                reason="generator_exception",
                                detail=str(exc),
                                extra_error_json={"exception_type": type(exc).__name__},
                                result_error=str(exc),
                            )
                            self._emit(
                                reporter,
                                "generation_failed",
                                slot_index=attempt.slot_index,
                                generation_index=attempt.generation_index,
                                duplicate_retry_count=attempt.duplicate_retry_count,
                                reason="generator_exception",
                                detail=str(exc),
                                failures=failures,
                                max_failures=max_failures,
                                completed=len(completed_slots),
                                requested=requested_generations,
                                in_flight=len(pending),
                            )
                            retry_needed = True
                        else:
                            generated = self._normalize_generation_result(raw_generated)
                            if not generated.succeeded:
                                failures += 1
                                error_info = dict(generated.error_info or {})
                                result = self._record_generation_attempt_failure(
                                    track_id=track_id,
                                    result=result,
                                    provenance_json=generated.provenance_json,
                                    reason=str(error_info.get("reason") or "generation_failed"),
                                    detail=str(error_info["detail"]) if error_info.get("detail") is not None else None,
                                    extra_error_json=error_info,
                                )
                                self._emit(
                                    reporter,
                                    "generation_failed",
                                    slot_index=attempt.slot_index,
                                    generation_index=attempt.generation_index,
                                    duplicate_retry_count=attempt.duplicate_retry_count,
                                    reason=str(error_info.get("reason") or "generation_failed"),
                                    detail=str(error_info.get("detail") or ""),
                                    failures=failures,
                                    max_failures=max_failures,
                                    completed=len(completed_slots),
                                    requested=requested_generations,
                                    in_flight=len(pending),
                                )
                                retry_needed = True
                            else:
                                assert generated.source is not None
                                try:
                                    candidate_source = materialize_candidate_source(
                                        attempt.context_trials[0].source,
                                        generated.source,
                                    )
                                except EvolveBlockError as exc:
                                    failures += 1
                                    result = self._record_generation_attempt_failure(
                                        track_id=track_id,
                                        result=result,
                                        provenance_json=generated.provenance_json,
                                        reason="candidate_materialization_failed",
                                        detail=str(exc),
                                        result_error=f"invalid_mutation:{exc}",
                                    )
                                    self._emit(
                                        reporter,
                                        "generation_failed",
                                        slot_index=attempt.slot_index,
                                        generation_index=attempt.generation_index,
                                        duplicate_retry_count=attempt.duplicate_retry_count,
                                        reason="candidate_materialization_failed",
                                        detail=str(exc),
                                        failures=failures,
                                        max_failures=max_failures,
                                        completed=len(completed_slots),
                                        requested=requested_generations,
                                        in_flight=len(pending),
                                    )
                                    retry_needed = True
                                else:
                                    candidate_hash = compute_script_hash(candidate_source)
                                    try:
                                        assert_only_evolve_blocks_changed(
                                            attempt.context_trials[0].source,
                                            candidate_source,
                                        )
                                    except EvolveBlockError as exc:
                                        failures += 1
                                        result = self._record_generation_attempt_failure(
                                            track_id=track_id,
                                            result=result,
                                            provenance_json=generated.provenance_json,
                                            reason="generation_assertion_failed",
                                            detail=str(exc),
                                            generated_source=candidate_source,
                                            candidate_hash=candidate_hash,
                                            result_error=f"invalid_mutation:{exc}",
                                        )
                                        self._emit(
                                            reporter,
                                            "generation_failed",
                                            slot_index=attempt.slot_index,
                                            generation_index=attempt.generation_index,
                                            duplicate_retry_count=attempt.duplicate_retry_count,
                                            reason="generation_assertion_failed",
                                            detail=str(exc),
                                            failures=failures,
                                            max_failures=max_failures,
                                            completed=len(completed_slots),
                                            requested=requested_generations,
                                            in_flight=len(pending),
                                        )
                                        retry_needed = True
                                    else:
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
                                            completed_slots.add(attempt.slot_index)
                                            self._emit(
                                                reporter,
                                                "generation_accepted",
                                                slot_index=attempt.slot_index,
                                                generation_index=attempt.generation_index,
                                                duplicate_retry_count=attempt.duplicate_retry_count,
                                                trial_id=trial.trial_id,
                                                completed=len(completed_slots),
                                                requested=requested_generations,
                                                failures=failures,
                                                max_failures=max_failures,
                                                in_flight=len(pending),
                                            )
                                        elif trial is not None:
                                            failures += 1
                                            result = self._record_duplicate_generation_attempt(
                                                track_id=track_id,
                                                result=result,
                                                provenance_json=final_provenance,
                                                candidate_hash=candidate_hash,
                                                trial_id=trial.trial_id,
                                            )
                                            self._emit(
                                                reporter,
                                                "generation_duplicate",
                                                slot_index=attempt.slot_index,
                                                generation_index=attempt.generation_index,
                                                duplicate_retry_count=attempt.duplicate_retry_count,
                                                existing_trial_id=trial.trial_id,
                                                failures=failures,
                                                max_failures=max_failures,
                                                completed=len(completed_slots),
                                                requested=requested_generations,
                                                in_flight=len(pending),
                                            )
                                            retry_needed = True

                        if retry_needed and failures + len(pending) < max_failures:
                            scheduled = self._schedule_generation_attempt(
                                executor,
                                track,
                                dataset_manifest,
                                sampling_settings,
                                slot_index=attempt.slot_index,
                                generation_index=attempt.generation_index,
                                duplicate_retry_count=attempt.duplicate_retry_count + 1,
                            )
                            if scheduled is not None:
                                next_future, next_attempt = scheduled
                                pending[next_future] = next_attempt
                                self._emit(
                                    reporter,
                                    "generation_scheduled",
                                    slot_index=next_attempt.slot_index,
                                    generation_index=next_attempt.generation_index,
                                    duplicate_retry_count=next_attempt.duplicate_retry_count,
                                    in_flight=len(pending),
                                )
                if len(completed_slots) < requested_generations and failures >= max_failures:
                    self._emit(
                        reporter,
                        "queue_fill_stopped",
                        completed=len(completed_slots),
                        requested=requested_generations,
                        failures=failures,
                        max_failures=max_failures,
                    )
                else:
                    self._emit(
                        reporter,
                        "queue_fill_completed",
                        completed=len(completed_slots),
                        requested=requested_generations,
                        failures=failures,
                        max_failures=max_failures,
                    )
        else:
            self._emit(
                reporter,
                "queue_fill_skipped",
                queued_count=queue_count,
                target_queue_count=ready_queue_threshold,
            )

        reserved = self.repository.reserve_trials(
            track_id=track_id,
            max_parallelism=max_parallelism,
            dispatch_ttl_sec=int(policy["dispatch_ttl_sec"]),
            limit=max_parallelism,
        )
        self._emit(
            reporter,
            "launch_batch_started",
            reserved_count=len(reserved),
            max_parallelism=max_parallelism,
        )
        for trial in reserved:
            try:
                self._emit(
                    reporter,
                    "trial_launch_started",
                    trial_id=trial.trial_id,
                )
                launch_metadata = self.launcher.launch_trial(trial.trial_id, trial.dispatch_token or "")
                if launch_metadata:
                    self.repository.record_trial_launcher_metadata(trial.trial_id, launch_metadata)
                result.launched_trial_ids.append(trial.trial_id)
                self._emit(
                    reporter,
                    "trial_launched",
                    trial_id=trial.trial_id,
                    launch_metadata=launch_metadata or {},
                )
            except Exception as exc:
                result.errors.append(f"launch_failed:{trial.trial_id}:{exc}")
                self._emit(
                    reporter,
                    "trial_launch_failed",
                    trial_id=trial.trial_id,
                    detail=str(exc),
                )
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
