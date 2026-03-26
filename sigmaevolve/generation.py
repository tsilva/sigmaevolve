from __future__ import annotations

import json
import os
import random
import re
from difflib import SequenceMatcher
from functools import lru_cache
from importlib import resources
from dataclasses import dataclass
from typing import Protocol
from urllib import request
from urllib.error import HTTPError, URLError

from sigmaevolve.evolve_blocks import EVOLVE_BLOCK_END, EVOLVE_BLOCK_START
from sigmaevolve.models import CANDIDATE_KIND_STRATEGY_V1, DatasetManifest, GenerationResult, TrackRecord, TrialSummary


class GenerationBackend(Protocol):
    def generate(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary] | None = None,
        generation_index: int = 0,
        duplicate_retry_count: int = 0,
    ) -> GenerationResult:
        ...


@dataclass(frozen=True)
class FixedGenerationBackend:
    source: str
    model_name: str = "fixed/test"

    def generate(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary] | None = None,
        generation_index: int = 0,
        duplicate_retry_count: int = 0,
    ) -> GenerationResult:
        # Emit a deterministic provenance payload for fixed test generations.
        system_prompt = "Test-only fixed generator stub for a recorded LLM prompt."
        user_prompt = "Use this parent trial as the base candidate:\n```python\n# fixed backend stub\n```"
        return GenerationResult(
            source=self.source,
            provenance_json={
                "backend": "openrouter",
                "model": self.model_name,
                "candidate_kind": CANDIDATE_KIND_STRATEGY_V1,
                "generation_index": generation_index,
                "duplicate_retry_count": duplicate_retry_count,
                "generation_config": {
                    "model": self.model_name,
                    "temperature": 0.0,
                    "max_tokens": 0,
                },
                "request_messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "context_trial_ids": [trial.trial_id for trial in context_trials],
                "generation": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response_text": self.source,
                    "generated_source": None,
                    "assertions_passed": False,
                    "assertion_failures": [],
                    "candidate_hash": None,
                },
            },
        )


@lru_cache(maxsize=None)
def _load_prompt_template(name: str) -> str:
    return (resources.files("sigmaevolve.prompts") / name).read_text(encoding="utf-8").strip()


def _render_prompt_template(name: str, **variables: str) -> str:
    # Load the template once and replace every declared variable placeholder.
    template = _load_prompt_template(name)

    def replace_variable(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        if variable_name not in variables:
            raise ValueError(f"Prompt template {name!r} is missing variable {variable_name!r}.")
        return variables[variable_name]

    return re.sub(r"{{([a-zA-Z0-9_]+)}}", replace_variable, template)


class OpenRouterGenerationBackend:
    def __init__(
        self,
        api_key: str | None = None,
        site_url: str = "https://sigmaevolve.local",
        app_name: str = "sigmaevolve",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.site_url = site_url
        self.app_name = app_name

    def _normalize_generation_config(self, generation_policy: dict[str, object]) -> dict[str, object]:
        # Resolve model-pool selection strategies before building the request payload.
        model_pool = generation_policy.get("model_pool")
        if isinstance(model_pool, list) and model_pool:
            selection = generation_policy.get("selection", "round_robin")
            index = int(generation_policy.get("_generation_index", 0))
            seed = int(generation_policy.get("seed", 0))
            if selection == "random":
                rng = random.Random(seed + index)
                return dict(rng.choice(model_pool))
            if selection == "weighted_random":
                weights: list[float] = []
                normalized_pool: list[dict[str, object]] = []
                for entry in model_pool:
                    item = dict(entry)
                    raw_weight = item.get("probability", item.get("weight", 1.0))
                    weight = float(raw_weight)
                    if weight < 0:
                        raise ValueError("generation_backend model_pool probabilities must be non-negative.")
                    normalized_pool.append(item)
                    weights.append(weight)
                total_weight = sum(weights)
                if total_weight <= 0:
                    raise ValueError(
                        "generation_backend weighted_random selection requires "
                        "a positive total probability."
                    )
                rng = random.Random(seed + index)
                selected = dict(rng.choices(normalized_pool, weights=weights, k=1)[0])
                selected["selection_probability"] = (
                    float(selected.get("probability", 1.0)) / total_weight
                )
                return selected
            return dict(model_pool[index % len(model_pool)])

        # Fall back to the single-model configuration shape.
        return {
            "model": generation_policy["model"],
            "temperature": generation_policy.get("temperature", 0.2),
            "max_tokens": generation_policy.get("max_tokens", 2500),
            "retry_count": generation_policy.get("retry_count", 2),
        }

    def _format_scalar(self, value: object) -> str:
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _format_mapping(self, payload: dict[str, object], indent: int = 0) -> list[str]:
        # Render nested payloads as an indented bullet list for prompt text.
        lines: list[str] = []
        prefix = " " * indent
        for key, value in payload.items():
            label = str(key)
            if isinstance(value, dict):
                lines.append(f"{prefix}- {label}:")
                lines.extend(self._format_mapping(value, indent + 2))
                continue
            if isinstance(value, list):
                if not value:
                    lines.append(f"{prefix}- {label}: none")
                    continue
                if all(not isinstance(item, (dict, list)) for item in value):
                    rendered = ", ".join(self._format_scalar(item) for item in value)
                    lines.append(f"{prefix}- {label}: {rendered}")
                    continue
                lines.append(f"{prefix}- {label}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.extend(self._format_mapping(item, indent + 2))
                    elif isinstance(item, list):
                        nested = ", ".join(self._format_scalar(part) for part in item)
                        lines.append(f"{' ' * (indent + 2)}- {nested}")
                    else:
                        lines.append(f"{' ' * (indent + 2)}- {self._format_scalar(item)}")
                continue
            lines.append(f"{prefix}- {label}: {self._format_scalar(value)}")
        return lines

    def _summarize_error(self, error_json: dict[str, object] | None) -> list[str]:
        # Extract the most actionable error fields for prompt-side diagnostics.
        if not error_json:
            return []
        lines: list[str] = []
        reason = error_json.get("reason")
        if reason is not None:
            lines.append(f"- error reason: {self._format_scalar(reason)}")
        detail = error_json.get("detail")
        if detail is not None:
            lines.append(f"- error detail: {self._format_scalar(detail)}")
        returncode = error_json.get("returncode")
        if returncode is not None:
            lines.append(f"- returncode: {self._format_scalar(returncode)}")
        stderr = error_json.get("stderr")
        if isinstance(stderr, str) and stderr.strip():
            excerpt = stderr.strip().splitlines()[-1][:240]
            lines.append(f"- stderr excerpt: {excerpt}")
        return lines

    def _trial_prompt_metric(self, trial: TrialSummary, *names: str) -> str:
        # Return the first populated metric value from the preferred aliases.
        metrics = trial.metrics_json or {}
        for name in names:
            if name in metrics and metrics[name] is not None:
                return self._format_scalar(metrics[name])
        return "n/a"

    def _strip_evolve_block_tags(self, source: str) -> str:
        lines = source.splitlines()
        filtered_lines = [line for line in lines if line not in {EVOLVE_BLOCK_START, EVOLVE_BLOCK_END}]
        return "\n".join(filtered_lines) + ("\n" if source.endswith("\n") else "")

    def _collapse_matching_source(self, source: str, reference_source: str) -> str:
        # Collapse unchanged regions so the prompt emphasizes the novel edits.
        source_lines = source.splitlines()
        reference_lines = reference_source.splitlines()
        summarized_lines: list[str] = []
        matcher = SequenceMatcher(a=source_lines, b=reference_lines, autojunk=False)
        for tag, source_start, source_end, _, _ in matcher.get_opcodes():
            if tag == "equal":
                if source_start == source_end:
                    continue
                if not summarized_lines or summarized_lines[-1] != "[...]":
                    summarized_lines.append("[...]")
                continue
            if tag in {"replace", "delete"}:
                summarized_lines.extend(source_lines[source_start:source_end])
        if not summarized_lines:
            summarized_lines.append("[...]")
        return "\n".join(summarized_lines) + ("\n" if source.endswith("\n") else "")

    def _render_trial_prompt_block(
        self,
        trial: TrialSummary,
        *,
        strip_evolve_block_tags: bool = False,
        collapse_matching_against: str | None = None,
    ) -> list[str]:
        # Normalize the source snapshot before rendering the trial prompt block.
        source = trial.source
        if strip_evolve_block_tags:
            source = self._strip_evolve_block_tags(source)
        if collapse_matching_against is not None:
            source = self._collapse_matching_source(source, collapse_matching_against)
        rendered = _render_prompt_template(
            "trial.md",
            score=self._format_scalar(trial.score),
            val_acc=self._trial_prompt_metric(trial, "val_acc", "accuracy"),
            val_loss=self._trial_prompt_metric(trial, "val_loss"),
            source=source.rstrip(),
        )
        return rendered.splitlines()

    def _build_system_prompt_text(self) -> str:
        return _render_prompt_template(
            "system.md",
            EVOLVE_BLOCK_START=EVOLVE_BLOCK_START,
            EVOLVE_BLOCK_END=EVOLVE_BLOCK_END,
        )

    def _build_user_prompt_text(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary],
        selected_config: dict[str, object],
    ) -> str:
        del track, dataset_manifest, negative_trials, selected_config

        # Split the context into the current program and optional prior examples.
        current_program = context_trials[0] if context_trials else None
        prior_programs = context_trials[1:] if len(context_trials) > 1 else []
        current_program_stripped_source = None
        if current_program is not None:
            current_program_stripped_source = self._strip_evolve_block_tags(current_program.source)

        # Render prior programs in diff-focused form when the prompt has history.
        if prior_programs:
            prior_programs_text = "\n".join(
                "\n".join(
                    self._render_trial_prompt_block(
                        trial,
                        strip_evolve_block_tags=True,
                        collapse_matching_against=current_program_stripped_source,
                    )
                )
                for trial in prior_programs
            )
        else:
            prior_programs_text = "None."

        # Render the primary current program in full so the model has a base candidate.
        if current_program is not None:
            current_program_text = "\n".join(self._render_trial_prompt_block(current_program))
        else:
            current_program_text = "None."
        return _render_prompt_template(
            "user.md",
            prior_programs=prior_programs_text,
            current_program=current_program_text,
        )

    def _build_prompt(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary],
        selected_config: dict[str, object],
    ) -> list[dict[str, str]]:
        # Build the final two-message chat payload from the system and user prompts.
        system_prompt = self._build_system_prompt_text()
        user_prompt = self._build_user_prompt_text(
            track,
            dataset_manifest,
            context_trials,
            negative_trials,
            selected_config,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _base_provenance(
        self,
        selected_config: dict[str, object],
        request_messages: list[dict[str, str]],
        context_trials: list[TrialSummary],
        *,
        generation_index: int,
        duplicate_retry_count: int,
        provider_response_id: str | None = None,
        response_text: str | None = None,
        response_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        # Preserve the prompt text and response metadata in a single provenance shape.
        system_prompt = request_messages[0]["content"] if request_messages else ""
        user_prompt = request_messages[1]["content"] if len(request_messages) > 1 else ""
        generation_payload: dict[str, object] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_text": response_text,
            "generated_source": None,
            "assertions_passed": False,
            "assertion_failures": [],
            "candidate_hash": None,
        }
        if response_metadata:
            generation_payload.update(response_metadata)

        # Add the generation bookkeeping fields shared by success and failure cases.
        provenance_json: dict[str, object] = {
            "backend": "openrouter",
            "model": str(selected_config["model"]),
            "candidate_kind": CANDIDATE_KIND_STRATEGY_V1,
            "generation_config": dict(selected_config),
            "generation_index": generation_index,
            "duplicate_retry_count": duplicate_retry_count,
            "request_messages": request_messages,
            "context_trial_ids": [trial.trial_id for trial in context_trials],
            "generation": generation_payload,
        }
        if provider_response_id:
            provenance_json["provider_response_id"] = provider_response_id
        return provenance_json

    def _extract_response_metadata(
        self,
        body: dict[str, object],
        choice: dict[str, object],
    ) -> dict[str, object]:
        # Extract stable provider metadata fields from the provider response body.
        metadata: dict[str, object] = {}
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            metadata["finish_reason"] = finish_reason
        native_finish_reason = choice.get("native_finish_reason")
        if isinstance(native_finish_reason, str) and native_finish_reason:
            metadata["native_finish_reason"] = native_finish_reason
        provider = body.get("provider")
        if isinstance(provider, str) and provider:
            metadata["provider"] = provider
        provider_model = body.get("model")
        if isinstance(provider_model, str) and provider_model:
            metadata["provider_model"] = provider_model
        return metadata

    def _extract_source(self, raw_text: str) -> str:
        match = re.search(r"```(?:python)?\n(.*?)```", raw_text, flags=re.DOTALL)
        if match:
            return match.group(1).strip() + "\n"
        return raw_text.strip() + "\n"

    def _extract_message_content(self, message: object) -> str | None:
        # Support both string and structured content payloads from the provider.
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return None
        text_parts: list[str] = []
        for entry in content:
            if isinstance(entry, str) and entry:
                text_parts.append(entry)
                continue
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        if not text_parts:
            return None
        return "\n".join(text_parts)

    def _missing_content_error_info(
        self,
        body: dict[str, object],
        raw_body: str,
        choice: dict[str, object],
        message: dict[str, object] | None,
    ) -> dict[str, object]:
        # Start from the missing-content reason and attach provider response context.
        error_info: dict[str, object] = {
            "reason": "provider_response_missing_content",
            "response_body": raw_body,
        }
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            error_info["finish_reason"] = finish_reason
        native_finish_reason = choice.get("native_finish_reason")
        if isinstance(native_finish_reason, str) and native_finish_reason:
            error_info["native_finish_reason"] = native_finish_reason
        provider = body.get("provider")
        if isinstance(provider, str) and provider:
            error_info["provider"] = provider
        model = body.get("model")
        if isinstance(model, str) and model:
            error_info["provider_model"] = model

        usage = body.get("usage")
        if isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens")
            if completion_tokens is not None:
                error_info["completion_tokens"] = completion_tokens
            completion_details = usage.get("completion_tokens_details")
            if isinstance(completion_details, dict):
                reasoning_tokens = completion_details.get("reasoning_tokens")
                if reasoning_tokens is not None:
                    error_info["reasoning_tokens"] = reasoning_tokens

        # Detect whether the provider consumed its budget on hidden reasoning.
        reasoning_present = False
        if isinstance(message, dict):
            reasoning = message.get("reasoning")
            reasoning_present = isinstance(reasoning, str) and bool(reasoning.strip())
        if reasoning_present:
            error_info["reasoning_present"] = True

        completion_tokens = error_info.get("completion_tokens")
        reasoning_tokens = error_info.get("reasoning_tokens")
        exhausted_reasoning_budget = (
            finish_reason == "length"
            and reasoning_present
            and isinstance(completion_tokens, int)
            and isinstance(reasoning_tokens, int)
            and completion_tokens > 0
            and reasoning_tokens >= completion_tokens
        )
        if exhausted_reasoning_budget:
            error_info["error_type"] = "generation_reasoning_tokens_exhausted"
            error_info["detail"] = (
                "Provider exhausted the completion budget on reasoning "
                "without emitting assistant content."
            )
        elif finish_reason == "length":
            error_info["error_type"] = "generation_output_truncated"
            error_info["detail"] = "Provider reached the completion limit before emitting assistant content."
        else:
            error_info["error_type"] = "generation_provider_failure"
        return error_info

    def _error_result(
        self,
        *,
        selected_config: dict[str, object],
        request_messages: list[dict[str, str]],
        context_trials: list[TrialSummary],
        generation_index: int,
        duplicate_retry_count: int,
        error_info: dict[str, object],
        provider_response_id: str | None = None,
        response_text: str | None = None,
        response_metadata: dict[str, object] | None = None,
    ) -> GenerationResult:
        # Wrap provider failures in the same provenance shape as successful results.
        return GenerationResult(
            source=None,
            provenance_json=self._base_provenance(
                selected_config,
                request_messages,
                context_trials,
                generation_index=generation_index,
                duplicate_retry_count=duplicate_retry_count,
                provider_response_id=provider_response_id,
                response_text=response_text,
                response_metadata=response_metadata,
            ),
            error_info=error_info,
        )

    def generate(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary] | None = None,
        generation_index: int = 0,
        duplicate_retry_count: int = 0,
    ) -> GenerationResult:
        # Resolve the concrete generation config for this attempt and retry number.
        generation_policy = dict(track.policy_json["generation_backend"])
        generation_policy["_generation_index"] = generation_index + duplicate_retry_count
        selected_config = self._normalize_generation_config(generation_policy)
        selected_temperature = float(selected_config.get("temperature", 0.2))
        selected_config["temperature"] = selected_temperature + (0.1 * duplicate_retry_count)

        # Build the request messages before checking provider credentials.
        request_messages = self._build_prompt(
            track,
            dataset_manifest,
            context_trials,
            negative_trials or [],
            selected_config,
        )
        if not self.api_key:
            missing_api_key_error = {
                "reason": "missing_api_key",
                "detail": "OPENROUTER_API_KEY is required for OpenRouter generation.",
            }
            return self._error_result(
                selected_config=selected_config,
                request_messages=request_messages,
                context_trials=context_trials,
                generation_index=generation_index,
                duplicate_retry_count=duplicate_retry_count,
                error_info=missing_api_key_error,
            )

        # Build the OpenRouter request payload and HTTP request object.
        payload = {
            "model": selected_config["model"],
            "messages": request_messages,
            "temperature": selected_config.get("temperature", 0.2),
            "max_tokens": selected_config.get("max_tokens", 2500),
        }
        req = request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
            },
            method="POST",
        )
        raw_body: str | None = None
        try:
            # Execute the provider request and capture the raw response text.
            with request.urlopen(req, timeout=120) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="replace")
            return self._error_result(
                selected_config=selected_config,
                request_messages=request_messages,
                context_trials=context_trials,
                generation_index=generation_index,
                duplicate_retry_count=duplicate_retry_count,
                error_info={
                    "reason": "provider_http_error",
                    "detail": f"{exc.code} {exc.reason}",
                    "status_code": exc.code,
                    "response_body": raw_body,
                },
            )
        except URLError as exc:
            return self._error_result(
                selected_config=selected_config,
                request_messages=request_messages,
                context_trials=context_trials,
                generation_index=generation_index,
                duplicate_retry_count=duplicate_retry_count,
                error_info={"reason": "provider_request_failed", "detail": str(exc.reason)},
            )
        except Exception as exc:
            return self._error_result(
                selected_config=selected_config,
                request_messages=request_messages,
                context_trials=context_trials,
                generation_index=generation_index,
                duplicate_retry_count=duplicate_retry_count,
                error_info={"reason": "provider_request_failed", "detail": str(exc)},
            )

        # Parse the provider body before validating the choice and content fields.
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            return self._error_result(
                selected_config=selected_config,
                request_messages=request_messages,
                context_trials=context_trials,
                generation_index=generation_index,
                duplicate_retry_count=duplicate_retry_count,
                error_info={"reason": "provider_response_invalid_json", "detail": str(exc), "response_body": raw_body},
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return self._error_result(
                selected_config=selected_config,
                request_messages=request_messages,
                context_trials=context_trials,
                generation_index=generation_index,
                duplicate_retry_count=duplicate_retry_count,
                provider_response_id=body.get("id"),
                error_info={"reason": "provider_response_missing_choices", "response_body": raw_body},
            )

        # Extract the first assistant message and reject empty content payloads.
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        choice = choices[0] if isinstance(choices[0], dict) else {}
        response_metadata = self._extract_response_metadata(body, choice)
        content = self._extract_message_content(message)
        if not isinstance(content, str) or not content.strip():
            return self._error_result(
                selected_config=selected_config,
                request_messages=request_messages,
                context_trials=context_trials,
                generation_index=generation_index,
                duplicate_retry_count=duplicate_retry_count,
                provider_response_id=body.get("id"),
                response_text=content if isinstance(content, str) else None,
                response_metadata=response_metadata,
                error_info=self._missing_content_error_info(
                    body,
                    raw_body,
                    choice,
                    message if isinstance(message, dict) else None,
                ),
            )

        # Return the normalized source with a complete provenance record.
        return GenerationResult(
            source=self._extract_source(content),
            provenance_json=self._base_provenance(
                selected_config,
                request_messages,
                context_trials,
                generation_index=generation_index,
                duplicate_retry_count=duplicate_retry_count,
                provider_response_id=body.get("id"),
                response_text=content,
                response_metadata=response_metadata,
            ),
        )
