from __future__ import annotations

import json
import os
import random
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from textwrap import indent
from typing import Any, Protocol
from urllib import request
from urllib.error import HTTPError, URLError

from sigmaevolve.core import (
    CANDIDATE_KIND_STRATEGY_V1,
    OUTCOME_DUPLICATE,
    OUTCOME_GENERATION_FAILED,
    DatasetManifest,
    GenerationResult,
    ReconcileResult,
    TrackRecord,
    TrialSummary,
    compute_script_hash,
    normalize_source,
)

_BASELINE_TEMPLATE_PATH = Path(__file__).with_name("baseline_template.py")


def build_baseline_train_script() -> str:
    template_source = _BASELINE_TEMPLATE_PATH.read_text(encoding="utf-8")
    return normalize_source(template_source)


EVOLVE_BLOCK_START = "# EVOLVE-BLOCK-START"
EVOLVE_BLOCK_END = "# EVOLVE-BLOCK-END"

_EVOLVE_BLOCK_PATTERN = re.compile(
    rf"(?ms)^[ \t]*{re.escape(EVOLVE_BLOCK_START)}\n(.*?)^[ \t]*{re.escape(EVOLVE_BLOCK_END)}\n?"
)


class EvolveBlockError(ValueError):
    pass


@dataclass(frozen=True)
class SearchReplaceBlock:
    search: str
    replace: str


@dataclass(frozen=True)
class ParsedGenerationResponse:
    task_description: str | None
    patch_text: str


TASK_DESCRIPTION_HEADER = "TASK_DESCRIPTION:"
MAX_CONTEXT_TRIALS = 4
MAX_NEGATIVE_TRIALS = 4
INSPIRATION_POOL_SIZE = 8


def _contains_evolve_block_marker_line(text: str) -> bool:
    for line in text.splitlines():
        stripped_line = line.strip()
        if stripped_line in {EVOLVE_BLOCK_START, EVOLVE_BLOCK_END}:
            return True
    return False


def _line_indent(line: str) -> str:
    prefix_length = len(line) - len(line.lstrip(" \t"))
    return line[:prefix_length]


def _common_indent(lines: list[str]) -> str:
    indents = [_line_indent(line) for line in lines if line.strip()]
    if not indents:
        return ""
    return os.path.commonprefix(indents)


def _dedent_lines(lines: list[str]) -> tuple[list[str], str]:
    indent = _common_indent(lines)
    if not indent:
        return list(lines), ""

    dedented = [
        line[len(indent) :] if line.startswith(indent) and line.strip() else line
        for line in lines
    ]
    return dedented, indent


def _canonicalize_patch_text(text: str) -> tuple[list[str], str]:
    lines = normalize_source(text).splitlines(keepends=True)
    return _dedent_lines(lines)


def _reindent_lines(lines: list[str], indent: str) -> list[str]:
    if not indent:
        return list(lines)
    return [f"{indent}{line}" if line.strip() else line for line in lines]


def _find_matching_line_ranges(
    source_lines: list[str],
    search_lines: list[str],
) -> list[tuple[int, int, str]]:
    if not search_lines:
        return []

    canonical_search_lines, _ = _dedent_lines(search_lines)
    search_length = len(search_lines)
    matches: list[tuple[int, int, str]] = []
    for start in range(len(source_lines) - search_length + 1):
        candidate_lines = source_lines[start : start + search_length]
        canonical_candidate_lines, candidate_indent = _dedent_lines(candidate_lines)
        if canonical_candidate_lines == canonical_search_lines:
            matches.append((start, start + search_length, candidate_indent))
    return matches


def split_evolve_blocks(source: str) -> tuple[list[str], list[str]]:
    normalized = normalize_source(source)
    matches = list(_EVOLVE_BLOCK_PATTERN.finditer(normalized))
    if not matches:
        raise EvolveBlockError("source must contain at least one evolve block")

    immutable_parts: list[str] = []
    block_payloads: list[str] = []
    cursor = 0
    for match in matches:
        block_start, block_end = match.span(1)
        immutable_parts.append(normalized[cursor:block_start])
        block_payloads.append(match.group(1))
        cursor = block_end
    immutable_parts.append(normalized[cursor:])
    return immutable_parts, block_payloads


def _merge_payloads(immutable_parts: list[str], payloads: list[str]) -> str:
    merged: list[str] = []
    for immutable_part, payload in zip(immutable_parts, payloads):
        merged.append(immutable_part)
        merged.append(payload)
    merged.append(immutable_parts[-1])
    return "".join(merged)


def _extract_outer_evolve_payload(source: str) -> tuple[list[str], str]:
    immutable_parts, block_payloads = split_evolve_blocks(source)
    if len(block_payloads) != 1:
        raise EvolveBlockError("source must contain exactly one outer evolve block")
    return immutable_parts, block_payloads[0]


def extract_evolve_block_payloads(source: str) -> list[str]:
    _, outer_payload = _extract_outer_evolve_payload(source)
    return [normalize_source(outer_payload)]


def replace_evolve_block_payloads(
    template_source: str, block_payloads: list[str]
) -> str:
    outer_immutable_parts, current_payload = _extract_outer_evolve_payload(
        template_source
    )
    if len(block_payloads) != 1:
        raise EvolveBlockError(
            f"expected 1 evolve block payload, received {len(block_payloads)}"
        )
    current_lines = current_payload.splitlines(keepends=True)
    replacement_lines = normalize_source(block_payloads[0]).splitlines(keepends=True)
    block_indent = _common_indent(current_lines)
    indented_replacement = "".join(_reindent_lines(replacement_lines, block_indent))
    return normalize_source(
        _merge_payloads(outer_immutable_parts, [indented_replacement])
    )


def parse_generation_response(response_text: str) -> ParsedGenerationResponse:
    normalized = normalize_source(response_text)
    stripped = normalized.strip()
    if not stripped:
        return ParsedGenerationResponse(task_description=None, patch_text=normalized)

    starts_patch = stripped.startswith("<<<<<<< SEARCH")
    is_no_changes = stripped == "NO_CHANGES"
    if starts_patch or is_no_changes:
        raise EvolveBlockError(
            "generated response must begin with TASK_DESCRIPTION before "
            "SEARCH/REPLACE blocks or NO_CHANGES"
        )

    lines = normalized.splitlines(keepends=True)
    if lines[0].strip() != TASK_DESCRIPTION_HEADER:
        return ParsedGenerationResponse(task_description=None, patch_text=normalized)

    cursor = 1
    description_lines: list[str] = []
    while cursor < len(lines):
        current_line = lines[cursor]
        stripped_line = current_line.strip()
        if current_line == "<<<<<<< SEARCH\n" or stripped_line == "NO_CHANGES":
            break
        description_lines.append(current_line)
        cursor += 1

    task_description = "".join(description_lines).strip()
    if not task_description:
        raise EvolveBlockError(
            "generated response must include a non-empty TASK_DESCRIPTION before "
            "SEARCH/REPLACE blocks or NO_CHANGES"
        )

    patch_text = normalize_source("".join(lines[cursor:]))
    if not patch_text.strip():
        raise EvolveBlockError(
            "generated response must include SEARCH/REPLACE blocks or NO_CHANGES "
            "after TASK_DESCRIPTION"
        )
    return ParsedGenerationResponse(
        task_description=task_description,
        patch_text=patch_text,
    )


def extract_task_description(response_text: str) -> str | None:
    try:
        return parse_generation_response(response_text).task_description
    except EvolveBlockError:
        return None


def parse_search_replace_blocks(response_text: str) -> list[SearchReplaceBlock]:
    normalized = parse_generation_response(response_text).patch_text
    if normalized.strip() == "NO_CHANGES":
        return []

    lines = normalized.splitlines(keepends=True)
    blocks: list[SearchReplaceBlock] = []
    cursor = 0

    while cursor < len(lines):
        # Skip layout-only separators between blocks.
        is_separator_line = lines[cursor].strip() == ""
        if is_separator_line:
            cursor += 1
            continue

        # Require the explicit SEARCH header before collecting patch text.
        has_search_header = lines[cursor] == "<<<<<<< SEARCH\n"
        if not has_search_header:
            raise EvolveBlockError(
                "generated response must contain SEARCH/REPLACE blocks or NO_CHANGES"
            )
        cursor += 1

        # Collect the SEARCH payload up to the middle separator.
        search_lines: list[str] = []
        while cursor < len(lines) and lines[cursor] != "=======\n":
            search_lines.append(lines[cursor])
            cursor += 1
        if cursor >= len(lines):
            raise EvolveBlockError("SEARCH/REPLACE block is missing ======= separator")
        cursor += 1

        # Collect the REPLACE payload until the block terminator.
        replace_lines: list[str] = []
        while cursor < len(lines) and lines[cursor] != ">>>>>>> REPLACE\n":
            replace_lines.append(lines[cursor])
            cursor += 1
        if cursor >= len(lines):
            raise EvolveBlockError(
                "SEARCH/REPLACE block is missing >>>>>>> REPLACE terminator"
            )
        cursor += 1

        search = "".join(search_lines)
        if not search:
            raise EvolveBlockError(
                "SEARCH/REPLACE block must include non-empty SEARCH text"
            )
        replace = "".join(replace_lines)
        contains_marker_line = _contains_evolve_block_marker_line(
            search
        ) or _contains_evolve_block_marker_line(replace)
        if contains_marker_line:
            raise EvolveBlockError(
                "SEARCH/REPLACE blocks may not include evolve block marker lines"
            )
        blocks.append(SearchReplaceBlock(search=search, replace=replace))

    if not blocks:
        raise EvolveBlockError(
            "generated response must contain SEARCH/REPLACE blocks or NO_CHANGES"
        )
    return blocks


def apply_search_replace_blocks(
    current_source: str, blocks: list[SearchReplaceBlock]
) -> str:
    updated_lines = normalize_source(current_source).splitlines(keepends=True)
    for index, block in enumerate(blocks, start=1):
        search_lines = normalize_source(block.search).splitlines(keepends=True)
        replace_lines, _ = _canonicalize_patch_text(block.replace)
        matches = _find_matching_line_ranges(updated_lines, search_lines)
        if not matches:
            raise EvolveBlockError(
                f"SEARCH block {index} did not match the current program"
            )
        if len(matches) > 1:
            raise EvolveBlockError(
                f"SEARCH block {index} matched multiple locations in the current program"
            )

        start, end, indent = matches[0]
        updated_lines[start:end] = _reindent_lines(replace_lines, indent)
    return normalize_source("".join(updated_lines))


def materialize_candidate_source(current_source: str, generated_source: str) -> str:
    parsed_response = parse_generation_response(generated_source)
    normalized_generated = parsed_response.patch_text
    stripped_generated = normalized_generated.strip()

    # Treat SEARCH/REPLACE input as a patch rather than a full program.
    is_search_replace_patch = stripped_generated.startswith("<<<<<<< SEARCH")
    if stripped_generated == "NO_CHANGES" or is_search_replace_patch:
        return apply_search_replace_blocks(
            current_source,
            parse_search_replace_blocks(generated_source),
        )

    # Accept a full program only when both evolve-block markers are present.
    has_evolve_block_tags = (
        EVOLVE_BLOCK_START in normalized_generated
        and EVOLVE_BLOCK_END in normalized_generated
    )
    if has_evolve_block_tags:
        return normalized_generated
    raise EvolveBlockError(
        "generated response must be SEARCH/REPLACE blocks, NO_CHANGES, or a full program"
    )


def assert_only_evolve_blocks_changed(
    parent_source: str, candidate_source: str
) -> None:
    parent_parts, parent_payloads = split_evolve_blocks(parent_source)
    candidate_parts, candidate_payloads = split_evolve_blocks(candidate_source)
    if parent_parts != candidate_parts:
        raise EvolveBlockError(
            "candidate modified immutable text outside evolve blocks"
        )
    if len(parent_payloads) != len(candidate_payloads):
        raise EvolveBlockError("candidate changed the number of evolve blocks")


class GenerationBackend(Protocol):
    def generate(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary] | None = None,
        generation_index: int = 0,
        duplicate_retry_count: int = 0,
    ) -> GenerationResult: ...


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
        user_prompt = (
            "Use this parent trial as the base candidate:\n"
            "```python\n"
            "# fixed backend stub\n"
            "```"
        )
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
    return (
        (resources.files("sigmaevolve.prompts") / name)
        .read_text(encoding="utf-8")
        .strip()
    )


def _render_prompt_template(name: str, **variables: str) -> str:
    # Load the template once and replace every declared variable placeholder.
    template = _load_prompt_template(name)

    def replace_variable(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        if variable_name not in variables:
            raise ValueError(
                f"Prompt template {name!r} is missing variable {variable_name!r}."
            )
        return variables[variable_name]

    return re.sub(r"{{([a-zA-Z0-9_]+)}}", replace_variable, template)


@dataclass(frozen=True)
class _GenerationRequestContext:
    selected_config: dict[str, object]
    request_messages: list[dict[str, str]]
    context_trials: list[TrialSummary]
    generation_index: int
    duplicate_retry_count: int


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

    def _normalize_generation_config(
        self,
        generation_policy: dict[str, object],
    ) -> dict[str, object]:
        # Resolve model-pool selection strategies before building the request payload.
        model_pool = generation_policy.get("model_pool")
        has_model_pool = isinstance(model_pool, list) and model_pool
        if has_model_pool:
            selection_strategy = generation_policy.get("selection", "round_robin")
            generation_index = int(generation_policy.get("_generation_index", 0))
            seed = int(generation_policy.get("seed", 0))

            # Handle stochastic pool selection before falling back to round robin.
            if selection_strategy == "random":
                rng = random.Random(seed + generation_index)
                return dict(rng.choice(model_pool))

            # Weight the pool entries when the policy asks for weighted selection.
            if selection_strategy == "weighted_random":
                weights: list[float] = []
                normalized_pool: list[dict[str, object]] = []
                for entry in model_pool:
                    item = dict(entry)
                    raw_weight = item.get("probability", item.get("weight", 1.0))
                    weight = float(raw_weight)
                    if weight < 0:
                        raise ValueError(
                            "generation_backend model_pool probabilities must be non-negative."
                        )
                    normalized_pool.append(item)
                    weights.append(weight)

                total_weight = sum(weights)
                if total_weight <= 0:
                    raise ValueError(
                        "generation_backend weighted_random selection requires "
                        "a positive total probability."
                    )

                rng = random.Random(seed + generation_index)
                selected = dict(rng.choices(normalized_pool, weights=weights, k=1)[0])
                selected["selection_probability"] = (
                    float(selected.get("probability", 1.0)) / total_weight
                )
                return selected

            # Use the deterministic pool slot when no stochastic selector is requested.
            model_pool_index = generation_index % len(model_pool)
            return dict(model_pool[model_pool_index])

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
            # Expand nested mappings so each field stays readable in prompt text.
            if isinstance(value, dict):
                lines.append(f"{prefix}- {label}:")
                lines.extend(self._format_mapping(value, indent + 2))
                continue

            # Collapse flat lists into one line when the values are scalar-like.
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
                        lines.append(
                            f"{' ' * (indent + 2)}- {self._format_scalar(item)}"
                        )
                continue
            lines.append(f"{prefix}- {label}: {self._format_scalar(value)}")
        return lines

    def _summarize_error(self, error_json: dict[str, object] | None) -> list[str]:
        # Extract the most actionable error fields for prompt-side diagnostics.
        if not error_json:
            return []
        lines: list[str] = []
        # Record the concise failure reason when the provider reported one.
        reason = error_json.get("reason")
        if reason is not None:
            lines.append(f"- error reason: {self._format_scalar(reason)}")

        # Include the human-readable detail when the error payload has one.
        detail = error_json.get("detail")
        if detail is not None:
            lines.append(f"- error detail: {self._format_scalar(detail)}")

        # Surface the subprocess return code for execution failures.
        returncode = error_json.get("returncode")
        if returncode is not None:
            lines.append(f"- returncode: {self._format_scalar(returncode)}")

        # Include duplicate bookkeeping when the row points at an existing candidate.
        existing_trial_id = error_json.get("existing_trial_id")
        if existing_trial_id is not None:
            lines.append(
                f"- existing trial id: {self._format_scalar(existing_trial_id)}"
            )

        candidate_hash = error_json.get("candidate_hash")
        if candidate_hash is not None:
            lines.append(f"- candidate hash: {self._format_scalar(candidate_hash)}")

        # Capture the last stderr line as the shortest useful excerpt.
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

    def _prompt_trial_source(
        self,
        trial: TrialSummary,
        *,
        strip_evolve_block_tags: bool = False,
        prefer_generated_source: bool = False,
    ) -> str:
        source = trial.source
        if prefer_generated_source:
            generation_payload = dict(
                (trial.provenance_json or {}).get("generation") or {}
            )
            generated_source = generation_payload.get("generated_source")
            has_generated_source = isinstance(generated_source, str) and bool(
                generated_source.strip()
            )
            if has_generated_source:
                source = generated_source

        if strip_evolve_block_tags:
            source = self._strip_evolve_block_tags(source)
        return source

    def _strip_evolve_block_tags(self, source: str) -> str:
        lines = source.splitlines()
        filtered_lines = [
            line for line in lines if line not in {EVOLVE_BLOCK_START, EVOLVE_BLOCK_END}
        ]
        return "\n".join(filtered_lines) + ("\n" if source.endswith("\n") else "")

    def _extract_compact_evolve_source(self, source: str) -> str:
        try:
            payloads = extract_evolve_block_payloads(source)
        except EvolveBlockError:
            return self._strip_evolve_block_tags(source)

        compact_source = "\n\n".join(payloads).rstrip()
        return f"{compact_source}\n" if compact_source else ""

    def _render_trial_prompt_block(
        self,
        trial: TrialSummary,
        *,
        strip_evolve_block_tags: bool = False,
        compact_evolve_source: bool = False,
    ) -> list[str]:
        # Normalize the source snapshot before rendering the trial prompt block.
        source = self._prompt_trial_source(
            trial,
            strip_evolve_block_tags=strip_evolve_block_tags,
        )
        if compact_evolve_source:
            source = self._extract_compact_evolve_source(source)
        rendered = _render_prompt_template(
            "trial.md",
            val_acc=self._trial_prompt_metric(trial, "val_acc", "accuracy"),
            val_loss=self._trial_prompt_metric(trial, "val_loss"),
            source=source.rstrip(),
        )
        return rendered.splitlines()

    def _render_negative_trial_prompt_block(self, trial: TrialSummary) -> list[str]:
        source = self._prompt_trial_source(
            trial,
            prefer_generated_source=True,
        )
        source = self._extract_compact_evolve_source(source)
        lines = [
            "---",
            f"outcome_reason: {self._format_scalar(trial.outcome_reason or 'unknown')}",
        ]
        lines.extend(self._summarize_error(trial.error_json))
        lines.extend(("```python", source.rstrip(), "```"))
        return lines

    def _build_system_prompt_text(self) -> str:
        return _render_prompt_template(
            "system.md",
            EVOLVE_BLOCK_START=EVOLVE_BLOCK_START,
            EVOLVE_BLOCK_END=EVOLVE_BLOCK_END,
        )

    def _build_prompt_context_text(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
    ) -> str:
        prompt_context: dict[str, object] = {
            "dataset_id": track.dataset_id,
            "split_sizes": dict(dataset_manifest.split_sizes),
        }
        epochs = track.policy_json.get("epochs")
        if epochs is not None:
            prompt_context["epochs"] = epochs

        dataset_metadata: dict[str, object] = {}
        for key in ("num_classes", "feature_shape", "feature_dtype", "label_dtype"):
            value = dataset_manifest.metadata.get(key)
            if value is not None:
                dataset_metadata[key] = value
        if dataset_metadata:
            prompt_context["dataset_metadata"] = dataset_metadata

        return "\n".join(self._format_mapping(prompt_context))

    def _build_user_prompt_text(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary],
        selected_config: dict[str, object],
    ) -> str:
        del selected_config
        task_context_text = self._build_prompt_context_text(track, dataset_manifest)

        # Split the context into the current program and optional prior examples.
        current_program = context_trials[0] if context_trials else None
        prior_programs = context_trials[1:] if len(context_trials) > 1 else []

        # Compact prior programs to mutable payloads to avoid repeating boilerplate.
        if prior_programs:
            prior_program_blocks = []
            for trial in prior_programs:
                prior_program_blocks.append(
                    "\n".join(
                        self._render_trial_prompt_block(
                            trial,
                            compact_evolve_source=True,
                        )
                    )
                )
            prior_programs_text = "\n".join(prior_program_blocks)
        else:
            prior_programs_text = "None."

        # Keep the current program intact so SEARCH blocks match the real source.
        if current_program is not None:
            current_program_text = "\n".join(
                self._render_trial_prompt_block(current_program)
            )
        else:
            current_program_text = "None."
        if negative_trials:
            negative_trials_text = "\n".join(
                "\n".join(self._render_negative_trial_prompt_block(trial))
                for trial in negative_trials
            )
        else:
            negative_trials_text = "None."
        return _render_prompt_template(
            "user.md",
            task_context=task_context_text,
            prior_programs=prior_programs_text,
            negative_trials=negative_trials_text,
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

    def _build_request_context(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary],
        *,
        generation_index: int,
        duplicate_retry_count: int,
    ) -> _GenerationRequestContext:
        # Resolve the concrete generation config before constructing prompt text.
        generation_policy = dict(track.policy_json["generation_backend"])
        generation_policy["_generation_index"] = (
            generation_index + duplicate_retry_count
        )
        selected_config = self._normalize_generation_config(generation_policy)
        selected_temperature = float(selected_config.get("temperature", 0.2))
        selected_config["temperature"] = selected_temperature + (
            0.1 * duplicate_retry_count
        )

        request_messages = self._build_prompt(
            track,
            dataset_manifest,
            context_trials,
            negative_trials,
            selected_config,
        )
        return _GenerationRequestContext(
            selected_config=selected_config,
            request_messages=request_messages,
            context_trials=context_trials,
            generation_index=generation_index,
            duplicate_retry_count=duplicate_retry_count,
        )

    def _build_provenance(
        self,
        context: _GenerationRequestContext,
        *,
        generation_index: int,
        duplicate_retry_count: int,
        provider_response_id: str | None = None,
        task_description: str | None = None,
        response_text: str | None = None,
        reasoning_text: str | None = None,
        response_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        # Preserve the prompt text and response metadata in a single provenance shape.
        request_messages = context.request_messages
        system_prompt = request_messages[0]["content"] if request_messages else ""
        user_prompt = (
            request_messages[1]["content"] if len(request_messages) > 1 else ""
        )
        generation_payload: dict[str, object] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "task_description": task_description,
            "response_text": response_text,
            "reasoning_text": reasoning_text,
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
            "model": str(context.selected_config["model"]),
            "candidate_kind": CANDIDATE_KIND_STRATEGY_V1,
            "generation_config": dict(context.selected_config),
            "generation_index": generation_index,
            "duplicate_retry_count": duplicate_retry_count,
            "request_messages": request_messages,
            "context_trial_ids": [trial.trial_id for trial in context.context_trials],
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

    def _extract_reasoning_text(self, message: object) -> str | None:
        # Capture provider reasoning text from either direct fields or structured traces.
        if not isinstance(message, dict):
            return None

        direct_reasoning = message.get("reasoning")
        if isinstance(direct_reasoning, str) and direct_reasoning.strip():
            return direct_reasoning

        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return reasoning_content

        reasoning_details = message.get("reasoning_details")
        if not isinstance(reasoning_details, list):
            return None

        collected_parts: list[str] = []
        for entry in reasoning_details:
            if isinstance(entry, str) and entry.strip():
                collected_parts.append(entry)
                continue
            if not isinstance(entry, dict):
                continue
            for field in ("text", "reasoning", "content"):
                value = entry.get(field)
                if isinstance(value, str) and value.strip():
                    collected_parts.append(value)
                    break

        if collected_parts:
            return "\n\n".join(collected_parts)

        # Preserve some readable trace even when the provider only returned structured entries.
        return json.dumps(reasoning_details, indent=2)

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
        reasoning_text = self._extract_reasoning_text(message)
        reasoning_present = isinstance(reasoning_text, str) and bool(
            reasoning_text.strip()
        )
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
            error_info["detail"] = (
                "Provider reached the completion limit before emitting assistant content."
            )
        else:
            error_info["error_type"] = "generation_provider_failure"
        return error_info

    def _build_generation_result(
        self,
        *,
        context: _GenerationRequestContext,
        source: str | None = None,
        error_info: dict[str, object],
        provider_response_id: str | None = None,
        task_description: str | None = None,
        response_text: str | None = None,
        reasoning_text: str | None = None,
        response_metadata: dict[str, object] | None = None,
    ) -> GenerationResult:
        # Wrap provider failures in the same provenance shape as successful results.
        return GenerationResult(
            source=source,
            provenance_json=self._build_provenance(
                context,
                generation_index=context.generation_index,
                duplicate_retry_count=context.duplicate_retry_count,
                provider_response_id=provider_response_id,
                task_description=task_description,
                response_text=response_text,
                reasoning_text=reasoning_text,
                response_metadata=response_metadata,
            ),
            error_info=dict(error_info) or None,
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
        # Resolve the concrete generation config and prompt messages once up front.
        context = self._build_request_context(
            track,
            dataset_manifest,
            context_trials,
            negative_trials or [],
            generation_index=generation_index,
            duplicate_retry_count=duplicate_retry_count,
        )
        if not self.api_key:
            return self._build_generation_result(
                context=context,
                error_info={
                    "reason": "missing_api_key",
                    "detail": "OPENROUTER_API_KEY is required for OpenRouter generation.",
                },
            )

        # Build the OpenRouter request payload and HTTP request object.
        payload = {
            "model": context.selected_config["model"],
            "messages": context.request_messages,
            "temperature": context.selected_config.get("temperature", 0.2),
            "max_tokens": context.selected_config.get("max_tokens", 2500),
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
            return self._build_generation_result(
                context=context,
                error_info={
                    "reason": "provider_http_error",
                    "detail": f"{exc.code} {exc.reason}",
                    "status_code": exc.code,
                    "response_body": raw_body,
                },
            )
        except URLError as exc:
            return self._build_generation_result(
                context=context,
                error_info={
                    "reason": "provider_request_failed",
                    "detail": str(exc.reason),
                },
            )
        except Exception as exc:
            return self._build_generation_result(
                context=context,
                error_info={"reason": "provider_request_failed", "detail": str(exc)},
            )

        # Parse the provider body before validating the choice and content fields.
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            return self._build_generation_result(
                context=context,
                error_info={
                    "reason": "provider_response_invalid_json",
                    "detail": str(exc),
                    "response_body": raw_body,
                },
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return self._build_generation_result(
                context=context,
                provider_response_id=body.get("id"),
                error_info={
                    "reason": "provider_response_missing_choices",
                    "response_body": raw_body,
                },
            )

        # Extract the first assistant message and reject empty content payloads.
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        choice = choices[0] if isinstance(choices[0], dict) else {}
        response_metadata = self._extract_response_metadata(body, choice)
        content = self._extract_message_content(message)
        task_description = (
            extract_task_description(content) if isinstance(content, str) else None
        )
        reasoning_text = self._extract_reasoning_text(message)
        if not isinstance(content, str) or not content.strip():
            return self._build_generation_result(
                context=context,
                provider_response_id=body.get("id"),
                task_description=task_description,
                response_text=content if isinstance(content, str) else None,
                reasoning_text=reasoning_text,
                response_metadata=response_metadata,
                error_info=self._missing_content_error_info(
                    body,
                    raw_body,
                    choice,
                    message if isinstance(message, dict) else None,
                ),
            )

        # Return the normalized source with a complete provenance record.
        return self._build_generation_result(
            context=context,
            source=self._extract_source(content),
            provider_response_id=body.get("id"),
            task_description=task_description,
            response_text=content,
            reasoning_text=reasoning_text,
            response_metadata=response_metadata,
            error_info={},
        )


@dataclass(frozen=True)
class GenerationAttempt:
    slot_index: int
    generation_index: int
    duplicate_retry_count: int
    context_trials: list[TrialSummary]


class GenerationCoordinator:
    def __init__(self, repository, generator) -> None:
        self.repository = repository
        self.generator = generator

    def sample_successful_context_trials(
        self,
        track_id: str,
        sampling_seed: int,
        generation_index: int,
    ) -> list[TrialSummary]:
        # Prefer finished strategy variants that already have scored metrics.
        candidates = self.repository.sample_trial_context(
            track_id,
            limit=self.repository.count_trials(track_id),
            candidate_kind=CANDIDATE_KIND_STRATEGY_V1,
        )
        if not candidates:
            return []
        if len(candidates) == 1:
            return [candidates[0]]

        rng = random.Random(int(sampling_seed) + generation_index)
        remaining = list(candidates)
        remaining_weights = [max(float(trial.score), 0.0) for trial in remaining]
        total_weight = sum(remaining_weights)
        if total_weight <= 0.0:
            selected_index = rng.randrange(len(remaining))
        else:
            selected_index = rng.choices(
                range(len(remaining)),
                weights=remaining_weights,
                k=1,
            )[0]
        current_program = remaining.pop(selected_index)
        remaining_weights.pop(selected_index)

        inspiration_pool = remaining[: min(INSPIRATION_POOL_SIZE, len(remaining))]
        inspiration_count = min(MAX_CONTEXT_TRIALS - 1, len(inspiration_pool))
        inspiration_indices = rng.sample(
            range(len(inspiration_pool)), k=inspiration_count
        )
        inspirations = [inspiration_pool[index] for index in inspiration_indices]

        # Keep the current program first and sort inspirations in stable best-first order.
        candidate_ranks = {
            trial.trial_id: index for index, trial in enumerate(candidates)
        }
        inspirations.sort(
            key=lambda trial: (-float(trial.score), candidate_ranks[trial.trial_id])
        )
        return [current_program, *inspirations]

    def sample_negative_trials(
        self,
        track_id: str,
        *,
        limit: int = MAX_NEGATIVE_TRIALS,
    ) -> list[TrialSummary]:
        # Surface recent duplicate attempts as negative examples for the prompt.
        return self.repository.list_recent_trial_summaries(
            track_id,
            outcome_reasons={OUTCOME_DUPLICATE},
            limit=limit,
        )

    def sample_generation_context_trials(
        self,
        track_id: str,
        sampling_seed: int,
        generation_index: int,
    ) -> list[TrialSummary]:
        # Use successful strategy trials first whenever any exist.
        successful_context = self.sample_successful_context_trials(
            track_id,
            sampling_seed,
            generation_index,
        )
        if successful_context:
            return successful_context

        # Avoid mixing in unfinished or failed context once scored trials exist.
        has_scored_history = self.repository.sample_trial_context(
            track_id,
            limit=self.repository.count_trials(track_id),
        )
        if has_scored_history:
            return []

        # Fall back to the seeded baseline when the track has no scored history yet.
        for trial in self.repository.list_trials(track_id):
            provenance = dict(trial.provenance_json or {})
            if provenance.get("backend") != "baseline":
                continue
            return [
                TrialSummary(
                    trial_id=trial.trial_id,
                    metrics_json=dict(trial.metrics_json)
                    if trial.metrics_json
                    else None,
                    source=trial.source,
                    provenance_json=provenance,
                    outcome_reason=trial.outcome_reason,
                    error_json=dict(trial.error_json) if trial.error_json else None,
                )
            ]
        return []

    def with_generation_trace(
        self,
        provenance_json: dict[str, Any],
        *,
        generated_source: str | None,
        assertions_passed: bool,
        assertion_failures: list[str],
        candidate_hash: str | None,
    ) -> dict[str, Any]:
        # Start from the recorded provenance so retries preserve prior metadata.
        payload = dict(provenance_json or {})
        generation_payload = dict(payload.get("generation") or {})
        request_messages = payload.get("request_messages")

        # Backfill prompt text when older payloads only stored request messages.
        if isinstance(request_messages, list):
            if "system_prompt" not in generation_payload and request_messages:
                first = request_messages[0]
                if isinstance(first, dict) and isinstance(first.get("content"), str):
                    generation_payload["system_prompt"] = first["content"]

            if "user_prompt" not in generation_payload and len(request_messages) > 1:
                second = request_messages[1]
                if isinstance(second, dict) and isinstance(second.get("content"), str):
                    generation_payload["user_prompt"] = second["content"]

        # Record the generated candidate trace in one normalized generation block.
        generation_payload.setdefault("response_text", None)
        generation_payload.setdefault("task_description", None)
        generation_payload.setdefault("reasoning_text", None)
        generation_payload["generated_source"] = generated_source
        generation_payload["assertions_passed"] = assertions_passed
        generation_payload["assertion_failures"] = list(assertion_failures)
        generation_payload["candidate_hash"] = candidate_hash
        payload["generation"] = generation_payload

        return payload

    def normalize_generation_result(self, generated: Any) -> GenerationResult:
        if isinstance(generated, GenerationResult):
            return generated
        return GenerationResult(
            source=getattr(generated, "source", None),
            provenance_json=dict(getattr(generated, "provenance_json", {}) or {}),
            error_info=dict(getattr(generated, "error_info", {}) or {}) or None,
        )

    def fallback_generation_provenance(
        self,
        track,
        context_trials: list[TrialSummary],
        *,
        generation_index: int,
        duplicate_retry_count: int,
    ) -> dict[str, Any]:
        # Recover the configured model name when the provider failed before logging prompts.
        generation_backend = dict(track.policy_json.get("generation_backend", {}))
        model = generation_backend.get("model")
        if not isinstance(model, str) or not model:
            model_pool = generation_backend.get("model_pool")
            if (
                isinstance(model_pool, list)
                and model_pool
                and isinstance(model_pool[0], dict)
            ):
                pool_model = model_pool[0].get("model")
                model = str(pool_model) if pool_model else "unknown"
            else:
                model = "unknown"

        system_prompt = (
            "Generation backend failed before prompts could be fully recorded."
        )
        user_prompt = (
            "No user prompt was captured because generation aborted before the "
            "provider call completed."
        )

        # Emit a synthetic but schema-valid generation payload for failure records.
        generation_payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "task_description": None,
            "response_text": None,
            "reasoning_text": None,
            "generated_source": None,
            "assertions_passed": False,
            "assertion_failures": [],
            "candidate_hash": None,
        }

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
            "generation": generation_payload,
        }

    def record_generation_attempt_failure(
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
        # Turn the failure into a persisted generation-attempt trial with trace data.
        assertion_failures = [detail] if detail else [reason]
        final_provenance = self.with_generation_trace(
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

        # Record the failure in both storage and the in-memory reconcile result.
        trial = self.repository.create_generation_attempt_trial(
            track_id=track_id,
            provenance_json=final_provenance,
            outcome_reason=OUTCOME_GENERATION_FAILED,
            error_json=error_payload,
        )
        result.failed_generation_trial_ids.append(trial.trial_id)
        result.errors.append(result_error or f"generation_failed:{reason}")
        return result

    def schedule_generation_attempt(
        self,
        executor: ThreadPoolExecutor,
        track,
        dataset_manifest,
        sampling_seed: int,
        *,
        slot_index: int,
        generation_index: int,
        duplicate_retry_count: int,
    ) -> tuple[Future[Any], GenerationAttempt] | None:
        # Skip scheduling when there is no valid context to generate from.
        context_trials = self.sample_generation_context_trials(
            track.track_id,
            sampling_seed,
            generation_index,
        )
        if not context_trials:
            return None
        negative_trials = self.sample_negative_trials(track.track_id)

        # Submit the provider request together with the bookkeeping metadata.
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
            negative_trials,
            generation_index,
            duplicate_retry_count,
        )
        return future, attempt

    def record_duplicate_generation_attempt(
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

    def materialize_candidate_source(
        self, parent_source: str, generated_source: str
    ) -> str:
        candidate_source = materialize_candidate_source(parent_source, generated_source)
        assert_only_evolve_blocks_changed(parent_source, candidate_source)
        return candidate_source

    def _accepted_generation_trace_error(
        self, provenance_json: dict[str, Any], candidate_source: str
    ) -> str | None:
        generation_payload = provenance_json.get("generation")
        if not isinstance(generation_payload, dict):
            return (
                "Accepted generated candidates must include provenance_json.generation."
            )

        response_text = generation_payload.get("response_text")
        has_response_text = isinstance(response_text, str) and bool(
            response_text.strip()
        )
        if not has_response_text:
            return (
                "Accepted generated candidates must persist a non-empty "
                "provenance_json.generation.response_text."
            )

        generated_source = generation_payload.get("generated_source")
        has_generated_source = isinstance(generated_source, str) and bool(
            generated_source.strip()
        )
        if not has_generated_source:
            return (
                "Accepted generated candidates must persist a non-empty "
                "provenance_json.generation.generated_source."
            )

        if normalize_source(generated_source) != normalize_source(candidate_source):
            return (
                "Accepted generated candidates must persist a generation "
                "generated_source that matches the queued trial source."
            )
        return None

    def accept_generated_candidate(
        self,
        *,
        track_id: str,
        result: ReconcileResult,
        generated: GenerationResult,
        attempt: GenerationAttempt,
        candidate_source: str,
    ) -> dict[str, Any]:
        candidate_hash = compute_script_hash(candidate_source)
        final_provenance = self.with_generation_trace(
            generated.provenance_json,
            generated_source=candidate_source,
            assertions_passed=True,
            assertion_failures=[],
            candidate_hash=candidate_hash,
        )
        trace_error = self._accepted_generation_trace_error(
            final_provenance,
            candidate_source,
        )
        if trace_error is not None:
            self.record_generation_attempt_failure(
                track_id=track_id,
                result=result,
                provenance_json=final_provenance,
                reason="missing_generation_trace",
                detail=trace_error,
                generated_source=candidate_source,
                candidate_hash=candidate_hash,
                result_error=f"missing_generation_trace:{trace_error}",
            )
            return {
                "event": "generation_failed",
                "payload": {
                    "reason": "missing_generation_trace",
                    "detail": trace_error,
                },
            }

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

        self.record_duplicate_generation_attempt(
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


def _normalize_payload(payload: str) -> str:
    return payload.strip("\n") + "\n"


def build_candidate_train_script(
    block_payload: str | None = None,
    *,
    config_block_payload: str | None = None,
    model_block_payload: str | None = None,
    data_block_payload: str | None = None,
    optimization_block_payload: str | None = None,
    training_policy_block_payload: str | None = None,
) -> str:
    template_source = build_baseline_train_script()
    replacement_payloads = [
        payload
        for payload in (
            model_block_payload if model_block_payload is not None else block_payload,
            config_block_payload,
            data_block_payload,
            optimization_block_payload,
            training_policy_block_payload,
        )
        if payload is not None
    ]
    if len(replacement_payloads) > 1:
        raise EvolveBlockError(
            "single-block templates accept only one replacement payload at a time"
        )
    if not replacement_payloads:
        return template_source
    return replace_evolve_block_payloads(
        template_source,
        [_normalize_payload(replacement_payloads[0])],
    )


def build_config_block(body: str) -> str:
    return _normalize_payload(body)


def _default_data_section() -> str:
    return "\n".join(
        (
            "batch_size = 64",
            "train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)",
            "val_loader = DataLoader(val_ds, batch_size=batch_size)",
        )
    )


def _default_model_section() -> str:
    return "\n".join(
        (
            "flat_dim = int(train_ds[0][0].numel())",
            "num_classes = int(",
            "    torch.cat((train_ds.tensors[1], val_ds.tensors[1])).max().item()",
            ") + 1",
            "",
            "model = nn.Sequential(",
            "    nn.Flatten(),",
            "    nn.Linear(flat_dim, 128),",
            "    nn.ReLU(),",
            "    nn.Linear(128, num_classes),",
            ").to(device)",
        )
    )


def _default_optimization_section() -> str:
    return "\n".join(
        (
            "trainable_parameters = [",
            "    parameter for parameter in model.parameters() if parameter.requires_grad",
            "]",
            "optimizer = None",
            "if trainable_parameters:",
            "    optimizer = torch.optim.Adam(trainable_parameters, lr=1e-3)",
            "",
            "scheduler = None",
            "if optimizer is not None:",
            "    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(",
            '        optimizer, mode="min", factor=0.5, patience=1',
            "    )",
        )
    )


def _default_training_policy_section() -> str:
    return "\n".join(
        (
            "early_stopping_patience = 2",
            "min_delta = 0.0",
        )
    )


def _default_loss_section() -> str:
    return "\n".join(
        (
            "def loss_fn(batch):",
            "    x, y = (tensor.to(device) for tensor in batch)",
            "    logits = model(x)",
            "    loss = F.cross_entropy(logits, y)",
            "    return loss, logits, y",
        )
    )


def _default_return_section() -> str:
    return "\n".join(
        (
            "return {",
            '    "model": model,',
            '    "optimizer": optimizer,',
            '    "scheduler": scheduler,',
            '    "loss_fn": loss_fn,',
            '    "train_loader": train_loader,',
            '    "val_loader": val_loader,',
            '    "early_stopping_patience": early_stopping_patience,',
            '    "min_delta": min_delta,',
            "}",
        )
    )


def _assemble_experiment_block(
    *,
    imports: str = "",
    data_section: str | None = None,
    model_section: str | None = None,
    optimization_section: str | None = None,
    training_policy_section: str | None = None,
) -> str:
    parts: list[str] = []
    normalized_imports = imports.strip()
    if normalized_imports:
        parts.append(normalized_imports)
        parts.append("")
    parts.append((data_section or _default_data_section()).strip("\n"))
    parts.append("")
    parts.append((model_section or _default_model_section()).strip("\n"))
    parts.append("")
    parts.append((optimization_section or _default_optimization_section()).strip("\n"))
    parts.append("")
    parts.append(
        (training_policy_section or _default_training_policy_section()).strip("\n")
    )
    parts.append("")
    parts.append(_default_loss_section())
    parts.append("")
    parts.append(_default_return_section())
    parts.append("")
    return "\n".join(parts)


def build_model_block(
    body: str,
    *,
    imports: str = "import torch",
    build_body: str = "return self.network(x)",
) -> str:
    model_section = "\n".join(
        (
            "flat_dim = int(train_ds[0][0].numel())",
            "num_classes = int(",
            "    torch.cat((train_ds.tensors[1], val_ds.tensors[1])).max().item()",
            ") + 1",
            "",
            "class EvolvedModel(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            "        self.network = nn.Sequential(",
            "            nn.Flatten(),",
            "            nn.Linear(flat_dim, 128),",
            "            nn.ReLU(),",
            "            nn.Linear(128, num_classes),",
            "        )",
            "",
            indent(body.strip("\n"), "    "),
            "",
            "model = EvolvedModel().to(device)",
        )
    )
    if body.strip() == "":
        model_section = "\n".join(
            (
                "flat_dim = int(train_ds[0][0].numel())",
                "num_classes = int(",
                "    torch.cat((train_ds.tensors[1], val_ds.tensors[1])).max().item()",
                ") + 1",
                "",
                "class EvolvedModel(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "        self.network = nn.Sequential(",
                "            nn.Flatten(),",
                "            nn.Linear(flat_dim, 128),",
                "            nn.ReLU(),",
                "            nn.Linear(128, num_classes),",
                "        )",
                "",
                "    def forward(self, x):",
                f"        {build_body}",
                "",
                "model = EvolvedModel().to(device)",
            )
        )
    return _assemble_experiment_block(
        imports=imports,
        model_section=model_section,
    )


def build_data_block(
    body: str,
    *,
    imports: str = "import torch",
) -> str:
    return _assemble_experiment_block(
        imports=imports,
        data_section=body,
    )


def build_optimization_block(
    body: str,
    *,
    imports: str = "import torch",
) -> str:
    return _assemble_experiment_block(
        imports=imports,
        optimization_section=body,
    )


def build_training_policy_block(body: str) -> str:
    return _assemble_experiment_block(
        training_policy_section=body,
    )
