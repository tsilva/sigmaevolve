from __future__ import annotations

from typing import Any

DEFAULT_MODEL_POOL_ID = "default_openrouter_v1"
MNIST_MODEL_POOL_ID = "mnist_openrouter_v1"

MODEL_POOL_CONFIGS: dict[str, list[dict[str, Any]]] = {
    DEFAULT_MODEL_POOL_ID: [
        {
            "model": "x-ai/grok-4.1-fast",
            "temperature": 0.2,
            "max_tokens": 2500,
            "retry_count": 2,
            "probability": 0.5436,
        },
        {
            "model": "google/gemini-3.7-flash",
            "temperature": 0.2,
            "max_tokens": 2500,
            "retry_count": 2,
            "probability": 0.2446,
        },
        {
            "model": "moonshotai/kimi-k2.5",
            "temperature": 0.2,
            "max_tokens": 2500,
            "retry_count": 2,
            "probability": 0.1578,
        },
        {
            "model": "google/gemini-3.1-pro-preview",
            "temperature": 0.2,
            "max_tokens": 2500,
            "retry_count": 2,
            "probability": 0.0306,
        },
        {
            "model": "anthropic/claude-sonnet-4.6",
            "temperature": 0.2,
            "max_tokens": 2500,
            "retry_count": 2,
            "probability": 0.0233,
        },
    ],
    MNIST_MODEL_POOL_ID: [
        {
            "model": "x-ai/grok-4.1-fast",
            "pricing": {
                "prompt": "0.0000002",
                "completion": "0.0000005",
                "web_search": "0.005",
                "input_cache_read": "0.00000005",
            },
            "temperature": 0.5,
            "max_tokens": 2500,
            "retry_count": 2,
            "probability": 0.2106,
        },
        {
            "model": "google/gemini-3.7-flash",
            "pricing": {
                "prompt": "0.000000375",
                "completion": "0.000001875",
                "image": "0.000000375",
                "audio": "0.000000375",
                "input_audio_cache": "0.0000000375",
                "web_search": "0.007",
                "internal_reasoning": "0.000001875",
                "input_cache_read": "0.0000000375",
                "input_cache_write": "0.0000000208333333333333",
            },
            "temperature": 0.5,
            "max_tokens": 2500,
            "retry_count": 2,
            "probability": 0.1615,
        },
        {
            "model": "moonshotai/kimi-k2.5",
            "pricing": {
                "prompt": "0.00000045",
                "completion": "0.0000022",
                "input_cache_read": "0.000000225",
            },
            "temperature": 0.5,
            "max_tokens": 20000,
            "retry_count": 2,
            "probability": 0.0909,
        },
        {
            "model": "google/gemini-3.1-pro-preview",
            "pricing": {
                "prompt": "0.000002",
                "completion": "0.000012",
                "image": "0.000002",
                "audio": "0.000002",
                "internal_reasoning": "0.000012",
                "input_cache_read": "0.0000002",
                "input_cache_write": "0.000000375",
            },
            "temperature": 0.5,
            "max_tokens": 20000,
            "retry_count": 2,
            "probability": 0.0202,
        },
        {
            "model": "anthropic/claude-sonnet-4.6",
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000015",
                "web_search": "0.01",
                "input_cache_read": "0.0000003",
                "input_cache_write": "0.00000375",
            },
            "temperature": 0.5,
            "max_tokens": 20000,
            "retry_count": 2,
            "probability": 0.0136,
        },
        {
            "model": "openai/gpt-5.4-nano",
            "pricing": {
                "prompt": "0.0000002",
                "completion": "0.00000125",
                "web_search": "0.01",
            },
            "temperature": 0.5,
            "max_tokens": 2500,
            "retry_count": 2,
            "probability": 0.2013,
        },
        {
            "model": "minimax/minimax-m2.7",
            "pricing": {
                "prompt": "0.0000003",
                "completion": "0.0000012",
            },
            "temperature": 0.5,
            "max_tokens": 20000,
            "retry_count": 2,
            "probability": 0.1378,
        },
        {
            "model": "deepseek/deepseek-v3.2",
            "pricing": {
                "prompt": "0.00000026",
                "completion": "0.00000038",
            },
            "temperature": 0.5,
            "max_tokens": 20000,
            "retry_count": 2,
            "probability": 0.1641,
        },
    ],
}


def get_model_pool_config(pool_id: str) -> list[dict[str, Any]]:
    normalized_pool_id = pool_id.strip()
    pool = MODEL_POOL_CONFIGS.get(normalized_pool_id)
    if pool is None:
        available = ", ".join(sorted(MODEL_POOL_CONFIGS))
        raise ValueError(
            f"Unknown model pool id {pool_id!r}. Available model pools: {available}."
        )
    return [dict(entry) for entry in pool]
