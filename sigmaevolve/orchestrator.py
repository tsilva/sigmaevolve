from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Protocol

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
                        raw_generated = self.generator.generate(
                            track,
                            dataset_manifest,
                            context_trials,
                            negative_trials=[],
                            generation_index=generation_index,
                            duplicate_retry_count=duplicate_retry_count,
                        )
                    except Exception as exc:
                        result = self._record_generation_attempt_failure(
                            track_id=track_id,
                            result=result,
                            provenance_json=self._fallback_generation_provenance(
                                track,
                                context_trials,
                                generation_index=generation_index,
                                duplicate_retry_count=duplicate_retry_count,
                            ),
                            reason="generator_exception",
                            detail=str(exc),
                            extra_error_json={"exception_type": type(exc).__name__},
                            result_error=str(exc),
                        )
                        break
                    generated = self._normalize_generation_result(raw_generated)
                    if not generated.succeeded:
                        error_info = dict(generated.error_info or {})
                        result = self._record_generation_attempt_failure(
                            track_id=track_id,
                            result=result,
                            provenance_json=generated.provenance_json,
                            reason=str(error_info.get("reason") or "generation_failed"),
                            detail=str(error_info["detail"]) if error_info.get("detail") is not None else None,
                            extra_error_json=error_info,
                        )
                        break
                    assert generated.source is not None
                    try:
                        candidate_source = materialize_candidate_source(context_trials[0].source, generated.source)
                    except EvolveBlockError as exc:
                        result = self._record_generation_attempt_failure(
                            track_id=track_id,
                            result=result,
                            provenance_json=generated.provenance_json,
                            reason="candidate_materialization_failed",
                            detail=str(exc),
                            result_error=f"invalid_mutation:{exc}",
                        )
                        break
                    candidate_hash = compute_script_hash(candidate_source)
                    try:
                        assert_only_evolve_blocks_changed(context_trials[0].source, candidate_source)
                    except EvolveBlockError as exc:
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
                        break
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
                        break
                    if trial is not None:
                        duplicate_trial = self.repository.create_generation_attempt_trial(
                            track_id=track_id,
                            provenance_json=final_provenance,
                            outcome_reason=OUTCOME_DUPLICATE,
                            error_json={
                                "reason": "duplicate_candidate",
                                "detail": f"Candidate source already exists as {trial.trial_id}.",
                                "candidate_hash": candidate_hash,
                                "existing_trial_id": trial.trial_id,
                            },
                        )
                        result.duplicate_hashes.append(candidate_hash)
                        result.duplicate_trial_ids.append(duplicate_trial.trial_id)

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
