from __future__ import annotations

import pytest

from sigmaevolve.models import TrackPolicy


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
