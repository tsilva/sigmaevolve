# ---- test_models.py ----

from __future__ import annotations

import pytest

from sigmaevolve.core import TrackPolicy


def test_track_policy_defaults_modal_gpu_preferences_to_none():
    policy = TrackPolicy.from_dict({})

    assert policy.modal_gpu_preferences is None
    assert policy.to_dict()["modal_gpu_preferences"] is None


def test_track_policy_persists_modal_gpu_preferences():
    policy = TrackPolicy.from_dict({"modal_gpu_preferences": ["T4", "L4", "A10"]})

    assert policy.modal_gpu_preferences == ["T4", "L4", "A10"]
    assert policy.to_dict()["modal_gpu_preferences"] == ["T4", "L4", "A10"]


def test_track_policy_rejects_invalid_modal_gpu_preferences():
    with pytest.raises(ValueError, match="modal_gpu_preferences"):
        TrackPolicy.from_dict({"modal_gpu_preferences": ["T4", 1]})

    with pytest.raises(ValueError, match="modal_gpu_preferences"):
        TrackPolicy.from_dict({"modal_gpu_preferences": []})


def test_track_policy_defaults_sampling_settings_to_seed_only():
    policy = TrackPolicy.from_dict({})

    assert policy.sampling_settings == {"seed": 0}
    assert policy.to_dict()["sampling_settings"] == {"seed": 0}


# ---- test_hashing.py ----

from sigmaevolve.core import compute_script_hash, normalize_source


def test_source_normalization_is_stable():
    left = "print('x')\r\n"
    right = "print('x')\n\n"
    assert normalize_source(left) == "print('x')\n"
    assert normalize_source(left) == normalize_source(right)
    assert compute_script_hash(left) == compute_script_hash(right)
