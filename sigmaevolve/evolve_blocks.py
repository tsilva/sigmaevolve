from __future__ import annotations

import re

from sigmaevolve.hashing import normalize_source


EVOLVE_BLOCK_START = "# EVOLVE-BLOCK-START"
EVOLVE_BLOCK_END = "# EVOLVE-BLOCK-END"

_EVOLVE_BLOCK_PATTERN = re.compile(
    rf"(?ms)^{re.escape(EVOLVE_BLOCK_START)}\n(.*?)^{re.escape(EVOLVE_BLOCK_END)}\n?"
)


class EvolveBlockError(ValueError):
    pass


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


def assert_only_evolve_blocks_changed(parent_source: str, candidate_source: str) -> None:
    parent_parts, parent_payloads = split_evolve_blocks(parent_source)
    candidate_parts, candidate_payloads = split_evolve_blocks(candidate_source)
    if parent_parts != candidate_parts:
        raise EvolveBlockError("candidate modified immutable text outside evolve blocks")
    if len(parent_payloads) != len(candidate_payloads):
        raise EvolveBlockError("candidate changed the number of evolve blocks")
