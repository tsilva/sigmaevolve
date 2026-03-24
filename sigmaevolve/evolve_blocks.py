from __future__ import annotations

import os
import re
from dataclasses import dataclass

from sigmaevolve.hashing import normalize_source


EVOLVE_BLOCK_START = "# EVOLVE-BLOCK-START"
EVOLVE_BLOCK_END = "# EVOLVE-BLOCK-END"

_EVOLVE_BLOCK_PATTERN = re.compile(
    rf"(?ms)^{re.escape(EVOLVE_BLOCK_START)}\n(.*?)^{re.escape(EVOLVE_BLOCK_END)}\n?"
)


class EvolveBlockError(ValueError):
    pass


@dataclass(frozen=True)
class SearchReplaceBlock:
    search: str
    replace: str


def _contains_evolve_block_marker_line(text: str) -> bool:
    for line in text.splitlines():
        if line.strip() in {EVOLVE_BLOCK_START, EVOLVE_BLOCK_END}:
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
    dedented = [line[len(indent) :] if line.startswith(indent) and line.strip() else line for line in lines]
    return dedented, indent


def _canonicalize_patch_text(text: str) -> tuple[list[str], str]:
    lines = normalize_source(text).splitlines(keepends=True)
    return _dedent_lines(lines)


def _reindent_lines(lines: list[str], indent: str) -> list[str]:
    if not indent:
        return list(lines)
    return [f"{indent}{line}" if line.strip() else line for line in lines]


def _find_matching_line_ranges(source_lines: list[str], search_lines: list[str]) -> list[tuple[int, int, str]]:
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


def extract_evolve_block_payloads(source: str) -> list[str]:
    _, block_payloads = split_evolve_blocks(source)
    return block_payloads


def replace_evolve_block_payloads(template_source: str, block_payloads: list[str]) -> str:
    immutable_parts, current_payloads = split_evolve_blocks(template_source)
    if len(block_payloads) != len(current_payloads):
        raise EvolveBlockError(
            f"expected {len(current_payloads)} evolve block payloads, received {len(block_payloads)}"
        )
    merged: list[str] = []
    for immutable_part, block_payload in zip(immutable_parts, block_payloads):
        merged.append(immutable_part)
        merged.append(block_payload)
    merged.append(immutable_parts[-1])
    return normalize_source("".join(merged))


def parse_search_replace_blocks(response_text: str) -> list[SearchReplaceBlock]:
    normalized = normalize_source(response_text)
    if normalized.strip() == "NO_CHANGES":
        return []

    lines = normalized.splitlines(keepends=True)
    blocks: list[SearchReplaceBlock] = []
    cursor = 0

    while cursor < len(lines):
        if lines[cursor].strip() == "":
            cursor += 1
            continue
        if lines[cursor] != "<<<<<<< SEARCH\n":
            raise EvolveBlockError("generated response must contain SEARCH/REPLACE blocks or NO_CHANGES")
        cursor += 1

        search_lines: list[str] = []
        while cursor < len(lines) and lines[cursor] != "=======\n":
            search_lines.append(lines[cursor])
            cursor += 1
        if cursor >= len(lines):
            raise EvolveBlockError("SEARCH/REPLACE block is missing ======= separator")
        cursor += 1

        replace_lines: list[str] = []
        while cursor < len(lines) and lines[cursor] != ">>>>>>> REPLACE\n":
            replace_lines.append(lines[cursor])
            cursor += 1
        if cursor >= len(lines):
            raise EvolveBlockError("SEARCH/REPLACE block is missing >>>>>>> REPLACE terminator")
        cursor += 1

        search = "".join(search_lines)
        if not search:
            raise EvolveBlockError("SEARCH/REPLACE block must include non-empty SEARCH text")
        replace = "".join(replace_lines)
        if _contains_evolve_block_marker_line(search) or _contains_evolve_block_marker_line(replace):
            raise EvolveBlockError("SEARCH/REPLACE blocks may not include evolve block marker lines")
        blocks.append(SearchReplaceBlock(search=search, replace=replace))

    if not blocks:
        raise EvolveBlockError("generated response must contain SEARCH/REPLACE blocks or NO_CHANGES")
    return blocks


def apply_search_replace_blocks(current_source: str, blocks: list[SearchReplaceBlock]) -> str:
    updated_lines = normalize_source(current_source).splitlines(keepends=True)
    for index, block in enumerate(blocks, start=1):
        search_lines = normalize_source(block.search).splitlines(keepends=True)
        replace_lines, _ = _canonicalize_patch_text(block.replace)
        matches = _find_matching_line_ranges(updated_lines, search_lines)
        if not matches:
            raise EvolveBlockError(f"SEARCH block {index} did not match the current program")
        if len(matches) > 1:
            raise EvolveBlockError(f"SEARCH block {index} matched multiple locations in the current program")
        start, end, indent = matches[0]
        updated_lines[start:end] = _reindent_lines(replace_lines, indent)
    return normalize_source("".join(updated_lines))


def materialize_candidate_source(current_source: str, generated_source: str) -> str:
    normalized_generated = normalize_source(generated_source)
    stripped_generated = normalized_generated.strip()
    if stripped_generated == "NO_CHANGES" or stripped_generated.startswith("<<<<<<< SEARCH"):
        return apply_search_replace_blocks(current_source, parse_search_replace_blocks(normalized_generated))
    if EVOLVE_BLOCK_START in normalized_generated and EVOLVE_BLOCK_END in normalized_generated:
        return normalized_generated
    raise EvolveBlockError("generated response must be SEARCH/REPLACE blocks, NO_CHANGES, or a full program")


def assert_only_evolve_blocks_changed(parent_source: str, candidate_source: str) -> None:
    parent_parts, parent_payloads = split_evolve_blocks(parent_source)
    candidate_parts, candidate_payloads = split_evolve_blocks(candidate_source)
    if parent_parts != candidate_parts:
        raise EvolveBlockError("candidate modified immutable text outside evolve blocks")
    if len(parent_payloads) != len(candidate_payloads):
        raise EvolveBlockError("candidate changed the number of evolve blocks")
