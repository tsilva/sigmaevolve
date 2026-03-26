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
    **extra,
):
    payload = {
        "backend": "openrouter",
        "model": model,
        "candidate_kind": CANDIDATE_KIND_STRATEGY_V1,
        "generation_config": {
            "model": model,
            "temperature": 0.1,
            "max_tokens": 1500,
        },
        "request_messages": request_messages
        if request_messages is not None
        else [
            {
                "role": "system",
                "content": "You are mutating an existing train.py candidate.",
            },
            {
                "role": "user",
                "content": "Use this parent trial as the base candidate:\n```python\nprint('parent')\n```",
            },
        ],
        "context_trial_ids": context_trial_ids if context_trial_ids is not None else ["trial_parent"],
    }
    payload.update(extra)
    return payload
