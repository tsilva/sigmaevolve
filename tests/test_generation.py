from __future__ import annotations

import json
import pytest

from sigmaevolve.generation import OpenRouterGenerationBackend
from sigmaevolve.models import DatasetManifest, TrackRecord, TrialSummary, now_utc
from sigmaevolve.train_script_blocks import build_candidate_train_script, build_model_block


def _track_with_pool():
    return TrackRecord(
        track_id="track_1",
        name="pool",
        dataset_id="mnist:v1",
        policy_json={
            "epochs": 5,
            "generation_backend": {
                "backend": "openrouter",
                "selection": "round_robin",
                "model_pool": [
                    {"model": "x-ai/grok-4.1-fast", "temperature": 0.1, "max_tokens": 1200},
                    {"model": "anthropic/claude-sonnet-4.6", "temperature": 0.8, "max_tokens": 2200},
                ],
            },
        },
        created_at=now_utc(),
    )


def _track_with_weighted_pool():
    return TrackRecord(
        track_id="track_weighted",
        name="weighted",
        dataset_id="mnist:v1",
        policy_json={
            "epochs": 5,
            "generation_backend": {
                "backend": "openrouter",
                "selection": "weighted_random",
                "seed": 11,
                "model_pool": [
                    {"model": "x-ai/grok-4.1-fast", "temperature": 0.1, "max_tokens": 1200, "probability": 0.0},
                    {"model": "moonshotai/kimi-k2.5", "temperature": 0.3, "max_tokens": 1600, "probability": 1.0},
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
            source=_mutated_script("return torch.zeros((x.shape[0], 10), dtype=torch.float32)"),
            provenance_json={"backend": "baseline", "candidate_kind": "strategy_v1"},
        )
    ]


def _context_with_prior_programs():
    return [
        TrialSummary(
            trial_id="trial_current",
            score=0.992,
            metrics_json={"accuracy": 0.992, "loss": 0.1},
            source=_mutated_script("return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.2"),
            provenance_json={"backend": "openrouter", "candidate_kind": "strategy_v1"},
        ),
        TrialSummary(
            trial_id="trial_prior",
            score=0.998,
            metrics_json={"accuracy": 0.998, "loss": 0.023},
            source=_mutated_script("return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.3"),
            provenance_json={"backend": "openrouter", "candidate_kind": "strategy_v1"},
        ),
    ]


def _negative_trials():
    return [
        TrialSummary(
            trial_id="trial_failed",
            score=0.0,
            metrics_json=None,
            source=_mutated_script("raise RuntimeError('bad candidate')"),
            provenance_json={"backend": "openrouter", "model": "test/model"},
            outcome_reason="crashed",
            error_json={
                "returncode": 1,
                "stderr": "RuntimeError: mat1 and mat2 shapes cannot be multiplied (55000x28 and 784x128)",
            },
        )
    ]


def _mutated_script(forward_body: str) -> str:
    return build_candidate_train_script(
        build_model_block(
            f"""
def forward(self, x):
    {forward_body.strip()}
"""
        )
    )


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
                    "choices": [
                        {
                            "message": {
                                "content": _mutated_script(
                                    "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.1"
                                )
                            }
                        }
                    ],
                }
            ).encode("utf-8")

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    track = _track_with_pool()
    first_result = backend.generate(track, _manifest(), _context(), generation_index=0)
    second_result = backend.generate(track, _manifest(), _context(), generation_index=1)

    assert payloads[0]["model"] == "x-ai/grok-4.1-fast"
    assert payloads[1]["model"] == "anthropic/claude-sonnet-4.6"
    assert first_result.provenance_json["model"] == "x-ai/grok-4.1-fast"
    assert first_result.provenance_json["candidate_kind"] == "strategy_v1"
    assert second_result.provenance_json["generation_config"]["temperature"] == 0.8
    assert first_result.provenance_json["request_messages"] == payloads[0]["messages"]

    system_prompt = payloads[0]["messages"][0]["content"]
    first_prompt = payloads[0]["messages"][1]["content"]
    assert "# EVOLVE-BLOCK-START" in system_prompt
    assert "# EVOLVE-BLOCK-END" in system_prompt
    assert not first_prompt.lstrip().startswith("{")
    assert "PRIOR PROGRAMS:" in first_prompt
    assert "CURRENT PROGRAM:" in first_prompt
    assert "Here is the current program we are trying to improve" in first_prompt
    assert "(you will need to propose a modification to it below)." in first_prompt
    assert "score: 0.5" in first_prompt
    assert "val_acc: 0.5" in first_prompt
    assert "val_loss: n/a" in first_prompt
    assert first_prompt.rstrip().endswith("REPLACEMENTS:")


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
                    "choices": [
                        {
                            "message": {
                                "content": _mutated_script("return torch.zeros((x.shape[0], 10), dtype=torch.float32)")
                            }
                        }
                    ],
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


def test_openrouter_generation_uses_weighted_random_probabilities(monkeypatch):
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
                    "choices": [
                        {
                            "message": {
                                "content": _mutated_script("return torch.zeros((x.shape[0], 10), dtype=torch.float32)")
                            }
                        }
                    ],
                }
            ).encode("utf-8")

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    result = backend.generate(_track_with_weighted_pool(), _manifest(), _context(), generation_index=0)

    assert payloads[0]["model"] == "moonshotai/kimi-k2.5"
    assert result.provenance_json["model"] == "moonshotai/kimi-k2.5"
    assert result.provenance_json["generation_config"]["selection_probability"] == pytest.approx(1.0)
    assert result.provenance_json["request_messages"] == payloads[0]["messages"]


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
                    "choices": [
                        {
                            "message": {
                                "content": _mutated_script("return torch.zeros((x.shape[0], 10), dtype=torch.float32)")
                            }
                        }
                    ],
                }
            ).encode("utf-8")

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    backend.generate(_track_with_pool(), _manifest(), _context(), negative_trials=_negative_trials(), generation_index=0)

    system_prompt = payloads[0]["messages"][0]["content"]
    prompt = payloads[0]["messages"][1]["content"]
    assert "# EVOLVE-BLOCK-START" in system_prompt
    assert "# EVOLVE-BLOCK-END" in system_prompt
    assert "CURRENT PROGRAM:" in prompt
    assert "REPLACEMENTS:" in prompt


def test_openrouter_generation_prompt_lists_prior_programs_before_current_program():
    backend = OpenRouterGenerationBackend(api_key="test-key")

    prompt = backend._build_user_prompt_text(
        _track_with_pool(),
        _manifest(),
        _context_with_prior_programs(),
        negative_trials=[],
        selected_config={"model": "test/model"},
    )

    assert prompt.startswith("PRIOR PROGRAMS:\n\n---\nscore: 0.998")
    assert "val_acc: 0.998" in prompt
    assert "val_loss: 0.023" in prompt
    assert "CURRENT PROGRAM:\n\nHere is the current program we are trying to improve" in prompt
    assert "score: 0.992" in prompt
    assert "val_acc: 0.992" in prompt
    assert "val_loss: 0.1" in prompt
    assert prompt.rstrip().endswith("REPLACEMENTS:")
