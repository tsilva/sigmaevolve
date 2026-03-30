from __future__ import annotations

import pytest

from sigmaevolve.core import (
    TrackPolicy,
    compute_script_hash,
    normalize_source,
)
from sigmaevolve.model_pools import MNIST_MODEL_POOL_ID


def test_track_policy_does_not_persist_removed_modal_gpu_preferences():
    policy = TrackPolicy.from_dict({})

    assert "modal_gpu_preferences" not in policy.to_dict()


def test_track_policy_rejects_removed_modal_gpu_preferences():
    with pytest.raises(ValueError, match="no longer supported"):
        TrackPolicy.from_dict({"modal_gpu_preferences": ["T4", "L4", "A10"]})


def test_track_policy_defaults_sampling_seed_to_zero():
    policy = TrackPolicy.from_dict({})
    policy_dict = policy.to_dict()

    assert policy.sampling_seed == 0
    assert policy_dict["sampling_seed"] == 0


def test_track_policy_resolves_repo_model_pool_id():
    policy = TrackPolicy.from_dict(
        {
            "generation_backend": {
                "selection": "weighted_random",
                "model_pool_id": MNIST_MODEL_POOL_ID,
            }
        }
    )

    generation_backend = policy.to_dict()["generation_backend"]
    assert generation_backend["model_pool_id"] == MNIST_MODEL_POOL_ID
    assert generation_backend["model_pool"][0]["model"] == "x-ai/grok-4.1-fast"


def test_track_policy_rejects_unknown_model_pool_id():
    with pytest.raises(ValueError, match="Unknown model pool id"):
        TrackPolicy.from_dict(
            {"generation_backend": {"model_pool_id": "does_not_exist"}}
        )


def test_track_policy_defaults_to_repo_default_model_pool_id_when_unspecified():
    policy = TrackPolicy.from_dict({})
    generation_backend = policy.to_dict()["generation_backend"]

    assert generation_backend["model_pool"][0]["model"] == "x-ai/grok-4.1-fast"
    assert "model_pool_id" not in generation_backend


def test_source_normalization_is_stable():
    left = "print('x')\r\n"
    right = "print('x')\n\n"
    normalized_left = normalize_source(left)
    normalized_right = normalize_source(right)
    left_hash = compute_script_hash(left)
    right_hash = compute_script_hash(right)

    assert normalized_left == "print('x')\n"
    assert normalized_left == normalized_right
    assert left_hash == right_hash
