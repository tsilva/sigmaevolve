from __future__ import annotations

from sigmaevolve.core import CANDIDATE_KIND_STRATEGY_V1


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
