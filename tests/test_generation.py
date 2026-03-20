from __future__ import annotations

import json

from sigmaevolve.generation import OpenRouterGenerationBackend
from sigmaevolve.models import DatasetManifest, TrackRecord, TrialSummary, now_utc


def _track_with_pool():
    return TrackRecord(
        track_id="track_1",
        name="pool",
        dataset_id="mnist:v1",
        policy_json={
            "budget_sec": 60,
            "generation_backend": {
                "backend": "openrouter",
                "selection": "round_robin",
                "model_pool": [
                    {"model": "openai/gpt-4o-mini", "temperature": 0.1, "max_tokens": 1200},
                    {"model": "anthropic/claude-3.5-sonnet", "temperature": 0.8, "max_tokens": 2200},
                ],
            },
        },
        created_at=now_utc(),
    )


def _manifest():
    return DatasetManifest(
        dataset_id="mnist:v1",
        root_dir="/tmp/dataset",
        train_split_path="/tmp/train.npz",
        validation_split_path="/tmp/validation.npz",
        validation_labels_path="/tmp/validation_labels.npy",
        test_split_path="/tmp/test.npz",
        test_labels_path="/tmp/test_labels.npy",
        split_sizes={"train": 1, "validation": 1, "test": 1},
        checksums={},
        fingerprint="fp",
        metadata={"num_classes": 10},
    )


def _context():
    return [
        TrialSummary(
            trial_id="trial_1",
            score=0.5,
            metrics_json={"accuracy": 0.5},
            source="print('old')\n",
            provenance_json={"backend": "baseline"},
        )
    ]


def test_openrouter_generation_uses_model_pool_round_robin(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")
    payloads = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "id": "resp_1",
                    "choices": [{"message": {"content": "print('new candidate')"}}],
                }
            ).encode("utf-8")

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    track = _track_with_pool()
    first_result = backend.generate(track, _manifest(), _context(), generation_index=0)
    second_result = backend.generate(track, _manifest(), _context(), generation_index=1)

    assert payloads[0]["model"] == "openai/gpt-4o-mini"
    assert payloads[1]["model"] == "anthropic/claude-3.5-sonnet"
    assert first_result.provenance_json["model"] == "openai/gpt-4o-mini"
    assert second_result.provenance_json["generation_config"]["temperature"] == 0.8
    assert first_result.provenance_json["request_messages"] == payloads[0]["messages"]

    first_prompt = payloads[0]["messages"][1]["content"]
    assert not first_prompt.lstrip().startswith("{")
    assert "Write a complete Python train.py for dataset mnist:v1." in first_prompt
    assert "Follow this task contract exactly:" in first_prompt
    assert "- max_eval_gap_sec: 15" in first_prompt
    assert "- progress_path: JSON heartbeat with current phase, elapsed_time_sec, and last_completed_eval_sec" in first_prompt
    assert "- validation_split_path: Path to the validation .npz file with features only." in first_prompt
    assert "Read the config JSON using the exact keys listed in config_keys" in first_prompt
    assert "No recent negative trials are available." in first_prompt
