from __future__ import annotations

from typing import Any

from sigmaevolve.hashing import normalize_source
from sigmaevolve.models import (
    ERROR_OUTCOMES,
    OUTCOME_GENERATION_FAILED,
    OUTCOME_STALE,
    TRIAL_STATUS_ERROR,
    TRIAL_STATUS_FINISHED,
)


ALLOWED_GENERATION_BACKENDS = frozenset({"openrouter"})


def _is_prompt_message(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False

    role = entry.get("role")
    content = entry.get("content")
    has_role = isinstance(role, str) and bool(role.strip())
    has_content = isinstance(content, str) and bool(content.strip())

    return has_role and has_content


def validate_trial_provenance(provenance_json: dict[str, Any]) -> dict[str, Any]:
    payload = dict(provenance_json or {})
    backend = payload.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("Queued trials require provenance_json.backend.")
    if backend == "baseline":
        return payload
    if backend not in ALLOWED_GENERATION_BACKENDS:
        raise ValueError(
            "Queued non-baseline trials must come from the recorded LLM prompting pipeline; "
            f"unsupported backend {backend!r}."
        )
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("LLM-generated trials require provenance_json.model.")
    generation_config = payload.get("generation_config")
    if not isinstance(generation_config, dict):
        raise ValueError("LLM-generated trials require provenance_json.generation_config.")
    request_messages = payload.get("request_messages")
    if not isinstance(request_messages, list) or not request_messages:
        raise ValueError("LLM-generated trials require non-empty provenance_json.request_messages.")
    if not all(_is_prompt_message(entry) for entry in request_messages):
        raise ValueError(
            "LLM-generated trials require provenance_json.request_messages entries with string role and content."
        )
    context_trial_ids = payload.get("context_trial_ids")
    if not isinstance(context_trial_ids, list):
        raise ValueError("LLM-generated trials require provenance_json.context_trial_ids.")
    candidate_kind = payload.get("candidate_kind")
    if not isinstance(candidate_kind, str) or not candidate_kind.strip():
        raise ValueError("LLM-generated trials require provenance_json.candidate_kind.")
    return payload


def has_error_signal(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    reason = payload.get("reason")
    if isinstance(reason, str) and reason.strip():
        return True
    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        return True
    stderr = payload.get("stderr")
    if isinstance(stderr, str) and stderr.strip():
        return True
    return payload.get("returncode") is not None


def build_generation_attempt_source(trial_id: str, outcome_reason: str) -> str:
    return normalize_source(
        "\n".join(
            [
                "# sigmaevolve generation attempt",
                f"# trial_id: {trial_id}",
                f"# outcome_reason: {outcome_reason}",
                "# diagnostic_source: true",
                "raise RuntimeError('diagnostic generation attempt source; see provenance_json.generation')",
            ]
        )
        + "\n"
    )


def status_for_outcome_reason(outcome_reason: str) -> str:
    if outcome_reason in ERROR_OUTCOMES:
        return TRIAL_STATUS_ERROR
    return TRIAL_STATUS_FINISHED


def classify_error_type(outcome_reason: str, error_json: dict[str, Any] | None) -> str | None:
    payload = dict(error_json or {})
    explicit = payload.get("error_type")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    reason = payload.get("reason")
    if not isinstance(reason, str):
        reason = None
    finish_reason = payload.get("finish_reason")
    native_finish_reason = payload.get("native_finish_reason")
    reached_length_limit = (
        (isinstance(finish_reason, str) and finish_reason == "length")
        or (isinstance(native_finish_reason, str) and native_finish_reason == "length")
    )

    if outcome_reason == OUTCOME_GENERATION_FAILED:
        if reason in {"candidate_materialization_failed", "generation_assertion_failed"} and reached_length_limit:
            return "generation_output_truncated"
        if reason in {"candidate_materialization_failed", "generation_assertion_failed"}:
            return "generation_invalid_candidate"
        if reason == "generator_exception":
            return "generation_backend_exception"
        if reason in {
            "provider_http_error",
            "provider_request_failed",
            "provider_response_invalid_json",
            "provider_response_missing_choices",
            "provider_response_missing_content",
        }:
            return "generation_provider_failure"
        return "generation_failed"

    if outcome_reason == "crashed":
        return "execution_crash"
    if outcome_reason == "eval_failed":
        if reason == "train_script_contract_violation":
            return "execution_contract_violation"
        if reason == "prediction_load_failed":
            return "evaluation_artifact_error"
        if reason == "predictions_missing":
            return "evaluation_predictions_missing"
        return "evaluation_failed"
    if outcome_reason == OUTCOME_STALE:
        if reason == "dispatch_deadline_expired":
            return "dispatch_stale"
        if reason == "heartbeat_stale":
            return "runner_stale"
        return "stale"
    return None


def prepare_error_payload(outcome_reason: str, error_json: dict[str, Any] | None) -> dict[str, Any] | None:
    payload = dict(error_json or {})
    error_type = classify_error_type(outcome_reason, payload)
    if error_type:
        payload["error_type"] = error_type
    return payload or None
