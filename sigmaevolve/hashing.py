from __future__ import annotations

import hashlib


def normalize_source(source: str) -> str:
    normalized = source.encode("utf-8", errors="strict").decode("utf-8")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.rstrip("\n") + "\n"
    return normalized


def compute_script_hash(source: str) -> str:
    normalized = normalize_source(source)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
