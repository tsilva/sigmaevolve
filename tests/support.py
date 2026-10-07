from __future__ import annotations

import json
from typing import Any

from sigmaevolve.core import CANDIDATE_KIND_STRATEGY_V1
from sigmaevolve.generation import build_baseline_train_script


class RecordingLauncherDouble:
    def __init__(self) -> None:
        self.launched: list[tuple[str, str]] = []

    def launch_trial(
        self,
        trial_id: str,
        dispatch_token: str,
        launch_policy: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        del launch_policy
        self.launched.append((trial_id, dispatch_token))
        return None

    def cancel_run(self, launcher_metadata: dict[str, object]) -> None:
        del launcher_metadata


def make_llm_provenance(
    model: str = "test/model",
    *,
    request_messages: list[dict[str, str]] | None = None,
    context_trial_ids: list[str] | None = None,
    **extra: object,
) -> dict[str, object]:
    # Build the fixed provenance fields first so the payload shape stays obvious.
    generation_config = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 1500,
    }

    # Prefer caller-provided messages when present.
    if request_messages is not None:
        effective_request_messages = request_messages
    else:
        effective_request_messages = [
            {
                "role": "system",
                "content": "You are mutating an existing train.py candidate.",
            },
            {
                "role": "user",
                "content": "Use this parent trial as the base candidate:\n```python\nprint('parent')\n```",
            },
        ]

    # Prefer caller-provided trial context when present.
    if context_trial_ids is not None:
        effective_context_trial_ids = context_trial_ids
    else:
        effective_context_trial_ids = ["trial_parent"]

    payload = {
        "backend": "openrouter",
        "model": model,
        "candidate_kind": CANDIDATE_KIND_STRATEGY_V1,
        "generation_config": generation_config,
        "request_messages": effective_request_messages,
        "context_trial_ids": effective_context_trial_ids,
    }
    payload.update(extra)
    return payload


def make_generation_trace(
    source: str,
    *,
    response_text: str | None = None,
    task_description: str = "Record the generated candidate for test coverage.",
    reasoning_text: str = "Selected the stored candidate for the test scenario.",
    assertions_passed: bool = True,
) -> dict[str, object]:
    return {
        "task_description": task_description,
        "response_text": response_text or source,
        "reasoning_text": reasoning_text,
        "generated_source": source,
        "assertions_passed": assertions_passed,
        "assertion_failures": [],
    }


def _render_toml_scalar(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        rendered_items = ", ".join(_render_toml_scalar(item) for item in value)
        return f"[{rendered_items}]"
    raise TypeError(f"Unsupported TOML scalar value: {value!r}")


def _render_toml_table_lines(
    table_name: str,
    payload: dict[str, Any],
    *,
    array_item: bool = False,
) -> list[str]:
    lines: list[str] = []
    header = f"[[{table_name}]]" if array_item else f"[{table_name}]"
    lines.append(f"# {header}")

    scalar_items: list[tuple[str, Any]] = []
    nested_tables: list[tuple[str, dict[str, Any]]] = []
    array_tables: list[tuple[str, list[dict[str, Any]]]] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            nested_tables.append((key, value))
            continue
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, dict) for item in value)
        ):
            array_tables.append((key, value))
            continue
        scalar_items.append((key, value))

    for key, value in scalar_items:
        lines.append(f"# {key} = {_render_toml_scalar(value)}")

    for key, value in nested_tables:
        lines.append("#")
        lines.extend(_render_toml_table_lines(f"{table_name}.{key}", value))

    for key, values in array_tables:
        for item in values:
            lines.append("#")
            lines.extend(
                _render_toml_table_lines(f"{table_name}.{key}", item, array_item=True)
            )

    return lines


def build_selfcontained_train_script(
    source: str | None = None,
    *,
    dataset_id: str = "mnist:v1",
    task: str = "Maximize validation accuracy while keeping the script runnable.",
    epochs: int | None = None,
    track_policy: dict[str, Any] | None = None,
) -> str:
    body = source if source is not None else build_baseline_train_script()
    body_lines = body.splitlines(keepends=True)
    if body_lines and body_lines[0].startswith("# /// sigmaevolve"):
        for index, line in enumerate(body_lines[1:], start=1):
            if line.rstrip("\n") == "# ///":
                body = "".join(body_lines[index + 1 :])
                break

    metadata_lines = [
        "# /// sigmaevolve",
        "# version = 1",
        f'# dataset_id = "{dataset_id}"',
        '# runner = "python_train_v1"',
    ]
    if track_policy:
        metadata_lines.extend(["#"])
        metadata_lines.extend(_render_toml_table_lines("track", track_policy))
    if epochs is not None:
        metadata_lines.extend(
            [
                "#",
                "# [defaults]",
                f"# epochs = {epochs}",
            ]
        )
    metadata_lines.extend(
        [
            "#",
            "# [evolution]",
            f'# task = "{task}"',
            "# ///",
            "",
        ]
    )
    return "\n".join(metadata_lines) + body
