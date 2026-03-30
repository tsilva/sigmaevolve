from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from typing import Any

from sigmaevolve.core import normalize_source

SIGMAEVOLVE_METADATA_START = "/// sigmaevolve"
SIGMAEVOLVE_METADATA_END = "///"
SCRIPT_SPEC_VERSION = 1
PYTHON_TRAIN_RUNNER = "python_train_v1"
DEFAULT_OBJECTIVE = "accuracy,val_loss:min"
ALLOWED_POLICY_DEFAULT_KEYS = frozenset({"epochs"})

_COMMENT_LINE_RE = re.compile(r"^[ \t]*#(?: ?(.*))?$")
_ENCODING_COMMENT_RE = re.compile(r"^[ \t]*#.*coding[:=][ \t]*[-_.a-zA-Z0-9]+")
_EVOLVE_BLOCK_START_RE = re.compile(
    r"^[ \t]*# EVOLVE-BLOCK-START(?:: (?P<name>[-_.a-zA-Z0-9]+))?[ \t]*$"
)
_EVOLVE_BLOCK_END_RE = re.compile(
    r"^[ \t]*# EVOLVE-BLOCK-END(?:: (?P<name>[-_.a-zA-Z0-9]+))?[ \t]*$"
)


class ScriptSpecError(ValueError):
    pass


@dataclass(frozen=True)
class EvolutionSpec:
    task: str
    objective: str | None = None


@dataclass(frozen=True)
class ScriptSpec:
    version: int
    dataset_id: str
    runner: str
    defaults: dict[str, Any]
    track_policy: dict[str, Any]
    evolution: EvolutionSpec


@dataclass(frozen=True)
class EvolveBlock:
    name: str
    payload: str
    payload_start_line: int
    payload_end_line: int


@dataclass(frozen=True)
class ParsedSourceLayout:
    script_spec: ScriptSpec | None
    immutable_parts: list[str]
    blocks: list[EvolveBlock]


def _comment_body(line: str) -> str | None:
    match = _COMMENT_LINE_RE.match(line)
    if match is None:
        return None
    return match.group(1) or ""


def _is_comment_or_blank(line: str) -> bool:
    stripped_line = line.strip()
    if not stripped_line:
        return True
    return _comment_body(line) is not None


def _line_contains_metadata_start(line: str) -> bool:
    return _comment_body(line) == SIGMAEVOLVE_METADATA_START


def _find_metadata_start_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if _line_contains_metadata_start(line):
            return index
    return None


def _extract_metadata_lines(
    lines: list[str], start_index: int
) -> tuple[list[str], int]:
    metadata_lines: list[str] = []
    for index in range(start_index + 1, len(lines)):
        body = _comment_body(lines[index])
        if body is None:
            raise ScriptSpecError(
                "sigmaevolve metadata block must use comment-prefixed lines."
            )
        if body == SIGMAEVOLVE_METADATA_END:
            return metadata_lines, index
        metadata_lines.append(body)
    raise ScriptSpecError("sigmaevolve metadata block is missing a closing '# ///'.")


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    # Merge nested tables while allowing scalar values to replace the base entry.
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(value, dict) and isinstance(base_value, dict):
            merged[key] = _deep_merge_dict(base_value, value)
            continue

        merged[key] = value
    return merged


def parse_script_spec(source: str) -> ScriptSpec | None:
    normalized_source = normalize_source(source)
    lines = normalized_source.splitlines()
    metadata_start_index = _find_metadata_start_index(lines)
    preamble_lines = (
        lines[:metadata_start_index] if metadata_start_index is not None else lines
    )

    for line in preamble_lines:
        is_shebang = line.startswith("#!")
        is_encoding_comment = _ENCODING_COMMENT_RE.match(line) is not None
        if is_shebang or is_encoding_comment or _is_comment_or_blank(line):
            continue
        if metadata_start_index is not None:
            raise ScriptSpecError(
                "sigmaevolve metadata block must appear before the first Python statement."
            )
        return None

    if metadata_start_index is None:
        return None

    metadata_lines, _ = _extract_metadata_lines(lines, metadata_start_index)
    metadata_text = "\n".join(metadata_lines)
    try:
        parsed = tomllib.loads(metadata_text)
    except tomllib.TOMLDecodeError as exc:
        raise ScriptSpecError(f"Invalid sigmaevolve metadata block: {exc}.") from exc

    version = parsed.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ScriptSpecError("sigmaevolve metadata version must be an integer.")
    if version != SCRIPT_SPEC_VERSION:
        raise ScriptSpecError(f"Unsupported sigmaevolve metadata version: {version!r}.")

    dataset_id = parsed.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ScriptSpecError(
            "sigmaevolve metadata dataset_id must be a non-empty string."
        )

    runner = parsed.get("runner")
    if not isinstance(runner, str) or not runner.strip():
        raise ScriptSpecError("sigmaevolve metadata runner must be a non-empty string.")
    if runner != PYTHON_TRAIN_RUNNER:
        raise ScriptSpecError(f"Unsupported sigmaevolve runner: {runner!r}.")

    defaults = parsed.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ScriptSpecError("sigmaevolve metadata defaults must be a table.")

    track_policy = parsed.get("track") or {}
    if not isinstance(track_policy, dict):
        raise ScriptSpecError("sigmaevolve metadata track must be a table.")

    evolution = parsed.get("evolution")
    if not isinstance(evolution, dict):
        raise ScriptSpecError("sigmaevolve metadata must include an [evolution] table.")
    if "mutable_regions" in evolution:
        raise ScriptSpecError(
            "sigmaevolve metadata may not declare evolution.mutable_regions."
        )

    task = evolution.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ScriptSpecError(
            "sigmaevolve metadata evolution.task must be a non-empty string."
        )

    objective = evolution.get("objective")
    if objective is not None:
        if not isinstance(objective, str):
            raise ScriptSpecError(
                "sigmaevolve metadata evolution.objective must be a string."
            )
        if objective != DEFAULT_OBJECTIVE:
            raise ScriptSpecError(
                "sigmaevolve metadata evolution.objective must be "
                f"{DEFAULT_OBJECTIVE!r} when provided."
            )

    return ScriptSpec(
        version=version,
        dataset_id=dataset_id.strip(),
        runner=runner,
        defaults=dict(defaults),
        track_policy=dict(track_policy),
        evolution=EvolutionSpec(task=task.strip(), objective=objective),
    )


def require_script_spec(source: str) -> ScriptSpec:
    spec = parse_script_spec(source)
    if spec is None:
        raise ScriptSpecError("source must include a sigmaevolve metadata block.")
    return spec


def apply_script_policy_defaults(
    policy_json: dict[str, Any],
    script_spec: ScriptSpec,
) -> dict[str, Any]:
    merged_policy = _deep_merge_dict(script_spec.track_policy, dict(policy_json or {}))
    for key in ALLOWED_POLICY_DEFAULT_KEYS:
        if key in merged_policy or key not in script_spec.defaults:
            continue

        value = script_spec.defaults[key]
        if key == "epochs":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ScriptSpecError(
                    "sigmaevolve metadata defaults.epochs must be an integer."
                )
            merged_policy[key] = int(value)
            continue

        merged_policy[key] = value
    return merged_policy


def _block_name(match: re.Match[str]) -> str:
    return match.group("name") or "main"


def is_evolve_marker_line(line: str) -> bool:
    return bool(_EVOLVE_BLOCK_START_RE.match(line) or _EVOLVE_BLOCK_END_RE.match(line))


def parse_source_layout(source: str) -> ParsedSourceLayout:
    normalized_source = normalize_source(source)
    lines = normalized_source.splitlines(keepends=True)
    immutable_parts: list[str] = []
    blocks: list[EvolveBlock] = []
    block_names: set[str] = set()
    cursor = 0
    index = 0

    while index < len(lines):
        line_text = lines[index].rstrip("\n")
        start_match = _EVOLVE_BLOCK_START_RE.match(line_text)
        if start_match is None:
            index += 1
            continue

        block_name = _block_name(start_match)
        if block_name in block_names:
            raise ScriptSpecError(f"Duplicate evolve block name: {block_name!r}.")
        block_names.add(block_name)
        immutable_parts.append("".join(lines[cursor : index + 1]))
        payload_start_line = index + 1
        index += 1

        while index < len(lines):
            candidate_line = lines[index].rstrip("\n")
            if _EVOLVE_BLOCK_START_RE.match(candidate_line) is not None:
                raise ScriptSpecError("Nested evolve blocks are not supported.")

            end_match = _EVOLVE_BLOCK_END_RE.match(candidate_line)
            if end_match is None:
                index += 1
                continue

            end_name = _block_name(end_match)
            if end_name != block_name:
                raise ScriptSpecError(
                    f"Evolve block {block_name!r} closed by mismatched end marker {end_name!r}."
                )

            blocks.append(
                EvolveBlock(
                    name=block_name,
                    payload="".join(lines[payload_start_line:index]),
                    payload_start_line=payload_start_line,
                    payload_end_line=index,
                )
            )
            cursor = index
            index += 1
            break
        else:
            raise ScriptSpecError(
                f"Evolve block {block_name!r} is missing a matching end marker."
            )

    if not blocks:
        raise ScriptSpecError("source must contain at least one evolve block.")

    immutable_parts.append("".join(lines[cursor:]))
    return ParsedSourceLayout(
        script_spec=parse_script_spec(normalized_source),
        immutable_parts=immutable_parts,
        blocks=blocks,
    )
