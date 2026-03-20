from __future__ import annotations

import json

import pytest

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
            source="def initialize(ctx):\n    return {}\n",
            provenance_json={"backend": "baseline", "candidate_kind": "strategy_v1"},
        )
    ]


def _negative_trials():
    return [
        TrialSummary(
            trial_id="trial_failed",
            score=0.0,
            metrics_json=None,
            source="print('bad candidate')\n",
            provenance_json={"backend": "openrouter", "model": "test/model"},
            outcome_reason="crashed",
            error_json={
                "returncode": 1,
                "stderr": "RuntimeError: mat1 and mat2 shapes cannot be multiplied (55000x28 and 784x128)",
            },
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
    assert first_result.provenance_json["candidate_kind"] == "strategy_v1"
    assert second_result.provenance_json["generation_config"]["temperature"] == 0.8
    assert first_result.provenance_json["request_messages"] == payloads[0]["messages"]

    system_prompt = payloads[0]["messages"][0]["content"]
    first_prompt = payloads[0]["messages"][1]["content"]
    assert "candidate module: strategy.py" in system_prompt
    assert "Treat this as an evolutionary mutation task, not a rewrite from scratch." in system_prompt
    assert "Follow this contract exactly:" in system_prompt
    assert "- validation_features: Validation features array from the harness." in system_prompt
    assert "Do not parse CLI args, read config files, write files, or manage progress/eval artifacts" in system_prompt
    assert "Produce a mutated descendant of the parent strategy, not a fresh rewrite." in system_prompt
    assert not first_prompt.lstrip().startswith("{")
    assert "Write a complete Python strategy.py module for dataset mnist:v1." in first_prompt
    assert "- max_eval_gap_sec: 15" in first_prompt
    assert "Use this parent trial as the base candidate:" in first_prompt
    assert "No recent negative trials are available." in first_prompt


def test_openrouter_generation_bumps_temperature_on_duplicate_retry(monkeypatch):
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

    result = backend.generate(_track_with_pool(), _manifest(), _context(), generation_index=0, duplicate_retry_count=2)

    assert payloads[0]["temperature"] == pytest.approx(0.3)
    assert result.provenance_json["duplicate_retry_count"] == 2
    assert result.provenance_json["generation_config"]["temperature"] == pytest.approx(0.3)


def test_openrouter_generation_prompt_includes_failure_feedback(monkeypatch):
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

    backend.generate(_track_with_pool(), _manifest(), _context(), negative_trials=_negative_trials(), generation_index=0)

    system_prompt = payloads[0]["messages"][0]["content"]
    prompt = payloads[0]["messages"][1]["content"]
    assert "if you use linear layers, flatten both train and validation batches consistently" in system_prompt
    assert "Make exactly one substantive improvement likely to improve validation accuracy within the time budget." in system_prompt
    assert "Trial trial_failed:" in prompt
    assert "- returncode: 1" in prompt
    assert "mat1 and mat2 shapes cannot be multiplied" in prompt
