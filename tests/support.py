from __future__ import annotations

from sigmaevolve.core import CANDIDATE_KIND_STRATEGY_V1


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
