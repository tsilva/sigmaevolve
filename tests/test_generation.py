from __future__ import annotations

import io
import json
import pytest
from urllib.error import HTTPError, URLError

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


class _FakeResponse:
    def __init__(self, payload: object):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        if isinstance(self._payload, bytes):
            return self._payload
        if isinstance(self._payload, str):
            return self._payload.encode("utf-8")
        return json.dumps(self._payload).encode("utf-8")


def _fake_generation_content(forward_body: str = "return torch.zeros((x.shape[0], 10), dtype=torch.float32)"):
    return {
        "id": "resp_1",
        "choices": [{"message": {"content": _mutated_script(forward_body)}}],
    }


def test_openrouter_generation_uses_model_pool_round_robin(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")
    payloads = []

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(_fake_generation_content("return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.1"))

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
    assert first_result.provenance_json["generation"]["system_prompt"] == payloads[0]["messages"][0]["content"]
    assert first_result.provenance_json["generation"]["user_prompt"] == payloads[0]["messages"][1]["content"]
    assert "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.1" in first_result.provenance_json["generation"]["response_text"]
    assert first_result.error_info is None

    system_prompt = payloads[0]["messages"][0]["content"]
    first_prompt = payloads[0]["messages"][1]["content"]
    assert "# EVOLVE-BLOCK-START" in system_prompt
    assert "# EVOLVE-BLOCK-END" in system_prompt
    assert "Never wrap the response in triple backticks or fenced code blocks" in system_prompt
    assert "If emitting a patch, begin immediately with <<<<<<< SEARCH on the first line" in system_prompt
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

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(_fake_generation_content())

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    result = backend.generate(_track_with_pool(), _manifest(), _context(), generation_index=0, duplicate_retry_count=2)

    assert payloads[0]["temperature"] == pytest.approx(0.3)
    assert result.provenance_json["duplicate_retry_count"] == 2
    assert result.provenance_json["generation_config"]["temperature"] == pytest.approx(0.3)


def test_openrouter_generation_uses_weighted_random_probabilities(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")
    payloads = []

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(_fake_generation_content())

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    result = backend.generate(_track_with_weighted_pool(), _manifest(), _context(), generation_index=0)

    assert payloads[0]["model"] == "moonshotai/kimi-k2.5"
    assert result.provenance_json["model"] == "moonshotai/kimi-k2.5"
    assert result.provenance_json["generation_config"]["selection_probability"] == pytest.approx(1.0)
    assert result.provenance_json["request_messages"] == payloads[0]["messages"]


def test_openrouter_generation_prompt_includes_failure_feedback(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")
    payloads = []

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(_fake_generation_content())

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

    assert prompt.startswith("PRIOR PROGRAMS:\n---\nscore: 0.998")
    assert "val_acc: 0.998" in prompt
    assert "val_loss: 0.023" in prompt
    assert "CURRENT PROGRAM:\nHere is the current program we are trying to improve" in prompt
    assert "score: 0.992" in prompt
    assert "val_acc: 0.992" in prompt
    assert "val_loss: 0.1" in prompt
    assert prompt.rstrip().endswith("REPLACEMENTS:")


def test_openrouter_generation_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    backend = OpenRouterGenerationBackend(api_key=None)

    result = backend.generate(_track_with_pool(), _manifest(), _context(), generation_index=0)

    assert result.source is None
    assert result.error_info == {
        "reason": "missing_api_key",
        "detail": "OPENROUTER_API_KEY is required for OpenRouter generation.",
    }
    assert result.provenance_json["generation"]["response_text"] is None
    assert result.provenance_json["generation"]["system_prompt"]
    assert result.provenance_json["generation"]["user_prompt"]


def test_openrouter_generation_captures_http_errors(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")

    def fake_urlopen(req, timeout=0):
        raise HTTPError(req.full_url, 503, "Service Unavailable", hdrs=None, fp=io.BytesIO(b'{"error":"overloaded"}'))

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    result = backend.generate(_track_with_pool(), _manifest(), _context(), generation_index=0)

    assert result.source is None
    assert result.error_info is not None
    assert result.error_info["reason"] == "provider_http_error"
    assert result.error_info["status_code"] == 503
    assert result.error_info["response_body"] == '{"error":"overloaded"}'


@pytest.mark.parametrize(
    ("response_payload", "expected_reason", "expected_field", "expected_value"),
    [
        (b"{not-json", "provider_response_invalid_json", "response_body", "{not-json"),
        ({"id": "resp_1", "choices": []}, "provider_response_missing_choices", "response_body", '{"id": "resp_1", "choices": []}'),
        ({"id": "resp_1", "choices": [{"message": {"content": "   "}}]}, "provider_response_missing_content", "response_text", "   "),
    ],
)
def test_openrouter_generation_response_errors(monkeypatch, response_payload, expected_reason, expected_field, expected_value):
    backend = OpenRouterGenerationBackend(api_key="test-key")
    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", lambda req, timeout=0: _FakeResponse(response_payload))

    result = backend.generate(_track_with_pool(), _manifest(), _context(), generation_index=0)

    assert result.source is None
    assert result.error_info is not None
    assert result.error_info["reason"] == expected_reason
    if expected_field == "response_text":
        assert result.provenance_json["generation"]["response_text"] == expected_value
    else:
        assert result.error_info[expected_field] == expected_value
    if expected_reason == "provider_response_missing_choices":
        assert result.provenance_json["provider_response_id"] == "resp_1"


def test_openrouter_generation_classifies_reasoning_budget_exhaustion(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "id": "resp_1",
                    "provider": "Fireworks",
                    "model": "moonshotai/kimi-k2.5-0127",
                    "choices": [
                        {
                            "finish_reason": "length",
                            "native_finish_reason": "length",
                            "message": {
                                "content": None,
                                "reasoning": "internal chain of thought",
                            },
                        }
                    ],
                    "usage": {
                        "completion_tokens": 2500,
                        "completion_tokens_details": {
                            "reasoning_tokens": 2500,
                        },
                    },
                }
            ).encode("utf-8")

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", lambda req, timeout=0: FakeResponse())

    result = backend.generate(_track_with_pool(), _manifest(), _context(), generation_index=0)

    assert result.source is None
    assert result.error_info is not None
    assert result.error_info["reason"] == "provider_response_missing_content"
    assert result.error_info["error_type"] == "generation_reasoning_tokens_exhausted"
    assert result.error_info["finish_reason"] == "length"
    assert result.error_info["reasoning_present"] is True


def test_openrouter_generation_captures_transport_errors(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")

    def fake_urlopen(req, timeout=0):
        raise URLError("network down")

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    result = backend.generate(_track_with_pool(), _manifest(), _context(), generation_index=0)

    assert result.source is None
    assert result.error_info is not None
    assert result.error_info["reason"] == "provider_request_failed"
    assert "network down" in result.error_info["detail"]
