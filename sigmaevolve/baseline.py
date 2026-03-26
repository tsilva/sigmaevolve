from __future__ import annotations

from pathlib import Path

from sigmaevolve.hashing import normalize_source


_BASELINE_TEMPLATE_PATH = Path(__file__).with_name("baseline_template.py")


def build_baseline_train_script() -> str:
    template_source = _BASELINE_TEMPLATE_PATH.read_text(encoding="utf-8")
    return normalize_source(template_source)


def build_baseline_linear_classifier() -> str:
    return build_baseline_train_script()
