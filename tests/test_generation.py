from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import numpy as np
import pytest
import torch

from sigmaevolve.core import DatasetManifest, TrackRecord, TrialSummary, now_utc
from sigmaevolve.generation import (
    EvolveBlockError,
    OpenRouterGenerationBackend,
    apply_search_replace_blocks,
    assert_only_evolve_blocks_changed,
    build_baseline_train_script,
    build_candidate_train_script,
    build_config_block,
    build_data_block,
    build_model_block,
    build_optimization_block,
    build_training_policy_block,
    extract_evolve_block_payloads,
    extract_task_description,
    materialize_candidate_source,
    parse_generation_response,
    parse_search_replace_blocks,
    replace_evolve_block_payloads,
)


def _track_with_pool():
    return TrackRecord(
        track_id="track_1",
        dataset_id="mnist:v1",
        policy_json={
            "epochs": 5,
            "generation_backend": {
                "selection": "round_robin",
                "model_pool": [
                    {
                        "model": "x-ai/grok-4.1-fast",
                        "temperature": 0.1,
                        "max_tokens": 1200,
                    },
                    {
                        "model": "anthropic/claude-sonnet-4.6",
                        "temperature": 0.8,
                        "max_tokens": 2200,
                    },
                ],
            },
        },
        created_at=now_utc(),
    )


def _track_with_weighted_pool():
    return TrackRecord(
        track_id="track_weighted",
        dataset_id="mnist:v1",
        policy_json={
            "epochs": 5,
            "generation_backend": {
                "selection": "weighted_random",
                "seed": 11,
                "model_pool": [
                    {
                        "model": "x-ai/grok-4.1-fast",
                        "temperature": 0.1,
                        "max_tokens": 1200,
                        "probability": 0.0,
                    },
                    {
                        "model": "moonshotai/kimi-k2.5",
                        "temperature": 0.3,
                        "max_tokens": 1600,
                        "probability": 1.0,
                    },
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
            metrics_json={"accuracy": 0.5},
            source=_mutated_script(
                "return torch.zeros((x.shape[0], 10), dtype=torch.float32)"
            ),
            provenance_json={"backend": "baseline", "candidate_kind": "strategy_v1"},
        )
    ]


def _context_with_prior_programs():
    return [
        TrialSummary(
            trial_id="trial_current",
            metrics_json={"accuracy": 0.992, "val_loss": 0.1},
            source=_mutated_script(
                "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.2"
            ),
            provenance_json={"backend": "openrouter", "candidate_kind": "strategy_v1"},
        ),
        TrialSummary(
            trial_id="trial_prior",
            metrics_json={"accuracy": 0.998, "val_loss": 0.023},
            source=_mutated_script(
                "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.3"
            ),
            provenance_json={"backend": "openrouter", "candidate_kind": "strategy_v1"},
        ),
    ]


def _context_with_many_prior_programs():
    context = _context_with_prior_programs()
    context.extend(
        [
            TrialSummary(
                trial_id="trial_prior_2",
                metrics_json={"accuracy": 0.997, "val_loss": 0.031},
                source=_mutated_script(
                    "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.4"
                ),
                provenance_json={
                    "backend": "openrouter",
                    "candidate_kind": "strategy_v1",
                },
            ),
            TrialSummary(
                trial_id="trial_prior_3",
                metrics_json={"accuracy": 0.996, "val_loss": 0.045},
                source=_mutated_script(
                    "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.5"
                ),
                provenance_json={
                    "backend": "openrouter",
                    "candidate_kind": "strategy_v1",
                },
            ),
        ]
    )
    return context


def _context_with_alternate_current_and_prior_programs():
    return [
        TrialSummary(
            trial_id="trial_current_alt",
            metrics_json={"accuracy": 0.991, "val_loss": 0.11},
            source=_mutated_script(
                "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.6"
            ),
            provenance_json={"backend": "openrouter", "candidate_kind": "strategy_v1"},
        ),
        *_context_with_many_prior_programs()[1:],
    ]


def _negative_trials():
    return [
        TrialSummary(
            trial_id="trial_failed",
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


def _many_negative_trials():
    return [
        TrialSummary(
            trial_id=f"trial_failed_{index}",
            metrics_json=None,
            source=_mutated_script(f"raise RuntimeError('bad candidate {index}')"),
            provenance_json={"backend": "openrouter", "model": "test/model"},
            outcome_reason="crashed",
            error_json={
                "returncode": index + 1,
                "detail": (
                    "shape mismatch while evaluating candidate "
                    f"{index} with a deliberately long diagnostic string"
                ),
            },
        )
        for index in range(4)
    ]


def _mutated_script(forward_body: str) -> str:
    return build_candidate_train_script(
        build_model_block(f"""
def forward(self, x):
    {forward_body.strip()}
""")
    )


class _FakeResponse:
    def __init__(self, payload: object):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        # Preserve raw bytes payloads without re-encoding them.
        if isinstance(self._payload, bytes):
            return self._payload

        # Encode string payloads exactly once for the fake response body.
        if isinstance(self._payload, str):
            return self._payload.encode("utf-8")

        # Serialize structured payloads the same way the real provider does.
        return json.dumps(self._payload).encode("utf-8")


def _fake_generation_content(
    forward_body: str = "return torch.zeros((x.shape[0], 10), dtype=torch.float32)",
):
    return {
        "id": "resp_1",
        "choices": [{"message": {"content": _mutated_script(forward_body)}}],
    }


def test_openrouter_generation_uses_model_pool_round_robin(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")
    payloads = []

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(
            _fake_generation_content(
                "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.1"
            )
        )

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
    assert len(payloads[0]["messages"]) == 5
    assert (
        first_result.provenance_json["generation"]["system_prompt"]
        == payloads[0]["messages"][0]["content"]
    )
    assert first_result.provenance_json["generation"]["user_prompt"] == "\n\n".join(
        message["content"] for message in payloads[0]["messages"][1:]
    )
    assert (
        "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.1"
        in first_result.provenance_json["generation"]["response_text"]
    )
    assert first_result.error_info is None

    system_prompt = payloads[0]["messages"][0]["content"]
    first_prompt = payloads[0]["messages"][1]["content"]
    reference_appendix = payloads[0]["messages"][2]["content"]
    negative_appendix = payloads[0]["messages"][3]["content"]
    current_program_appendix = payloads[0]["messages"][4]["content"]
    assert "# EVOLVE-BLOCK-START" in system_prompt
    assert "# EVOLVE-BLOCK-END" in system_prompt
    assert "Return exactly:" in system_prompt
    assert "TASK_DESCRIPTION:" in system_prompt
    assert "Maximize val_acc" in system_prompt
    assert "Use lower val_loss only to break ties" in system_prompt
    assert "Appendix order: REFERENCE, NEGATIVE, CURRENT_PROGRAM" in (system_prompt)
    assert "Randomly choose either a focused improvement or a broader revamp" in (
        system_prompt
    )
    assert "Use REFERENCE appendices as inspiration only" in system_prompt
    assert "textually distinct from CURRENT_PROGRAM" in system_prompt
    assert "SEARCH must match exactly once in CURRENT_PROGRAM" in system_prompt
    assert not first_prompt.lstrip().startswith("{")
    assert "TASK CONTEXT:" in first_prompt
    assert "- dataset_id: mnist:v1" in first_prompt
    assert "- epochs: 5" in first_prompt
    assert "- split_sizes:" in first_prompt
    assert "- dataset_metadata:" in first_prompt
    assert "- num_classes: 10" in first_prompt
    assert "OBJECTIVE:" in first_prompt
    assert "Improve CURRENT_PROGRAM for higher val_acc." in first_prompt
    assert "Only CURRENT_PROGRAM is editable." in first_prompt
    assert "Later appendices are ordered as REFERENCE, NEGATIVE, CURRENT_PROGRAM." in (
        first_prompt
    )
    assert "score:" not in first_prompt
    assert reference_appendix == "REFERENCE APPENDIX\nNone."
    assert negative_appendix == "NEGATIVE APPENDIX\nNone."
    assert "CURRENT PROGRAM APPENDIX" in current_program_appendix
    assert "CURRENT_PROGRAM val_acc=0.5 val_loss=n/a" in current_program_appendix
    assert "# EVOLVE-BLOCK-START" in current_program_appendix
    assert "# EVOLVE-BLOCK-END" in current_program_appendix


def test_openrouter_generation_bumps_temperature_on_duplicate_retry(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")
    payloads = []

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(_fake_generation_content())

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    result = backend.generate(
        _track_with_pool(),
        _manifest(),
        _context(),
        generation_index=0,
        duplicate_retry_count=2,
    )

    assert payloads[0]["temperature"] == pytest.approx(0.3)
    assert result.provenance_json["duplicate_retry_count"] == 2
    assert result.provenance_json["generation_config"]["temperature"] == pytest.approx(
        0.3
    )


def test_openrouter_generation_uses_weighted_random_probabilities(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")
    payloads = []

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(_fake_generation_content())

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    result = backend.generate(
        _track_with_weighted_pool(), _manifest(), _context(), generation_index=0
    )

    assert payloads[0]["model"] == "moonshotai/kimi-k2.5"
    assert result.provenance_json["model"] == "moonshotai/kimi-k2.5"
    assert result.provenance_json["generation_config"][
        "selection_probability"
    ] == pytest.approx(1.0)
    assert result.provenance_json["request_messages"] == payloads[0]["messages"]


def test_openrouter_generation_prompt_includes_expected_sections(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")
    payloads = []

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(_fake_generation_content())

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    backend.generate(
        _track_with_pool(),
        _manifest(),
        _context(),
        negative_trials=_negative_trials(),
        generation_index=0,
    )

    system_prompt = payloads[0]["messages"][0]["content"]
    stable_prompt = payloads[0]["messages"][1]["content"]
    reference_appendix = payloads[0]["messages"][2]["content"]
    negative_appendix = payloads[0]["messages"][3]["content"]
    current_program_appendix = payloads[0]["messages"][4]["content"]
    assert "# EVOLVE-BLOCK-START" in system_prompt
    assert "# EVOLVE-BLOCK-END" in system_prompt
    assert "OBJECTIVE:" in stable_prompt
    assert "TASK CONTEXT:" in stable_prompt
    assert reference_appendix.startswith("REFERENCE APPENDIX")
    assert negative_appendix.startswith("NEGATIVE APPENDIX")
    assert current_program_appendix.startswith("CURRENT PROGRAM APPENDIX")


def test_openrouter_generation_prompt_compacts_prior_programs_before_current_program():
    backend = OpenRouterGenerationBackend(api_key="test-key")

    prompt_messages = backend._build_prompt(
        _track_with_pool(),
        _manifest(),
        _context_with_prior_programs(),
        negative_trials=[],
        selected_config={"model": "test/model"},
    )

    stable_prompt = prompt_messages[1]["content"]
    prior_section = prompt_messages[2]["content"]
    negative_section = prompt_messages[3]["content"]
    current_section = prompt_messages[4]["content"]

    assert "OBJECTIVE:" in stable_prompt
    assert "TASK CONTEXT:" in stable_prompt
    assert "REFERENCE APPENDIX" in prior_section
    assert "REFERENCE val_acc=0.998 val_loss=0.023" in prior_section
    assert "score:" not in prior_section
    assert "[...]" not in prior_section
    assert "def forward(self, x):" in prior_section
    assert "from __future__ import annotations" not in prior_section
    assert "class TrainScriptContractError(RuntimeError):" not in prior_section
    assert (
        "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.3"
        in prior_section
    )
    assert (
        "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.2"
        not in prior_section
    )
    assert "CURRENT PROGRAM APPENDIX" in current_section
    assert "CURRENT_PROGRAM val_acc=0.992 val_loss=0.1" in current_section
    assert "score:" not in current_section
    assert "# EVOLVE-BLOCK-START" not in prior_section
    assert "# EVOLVE-BLOCK-END" not in prior_section
    assert "# EVOLVE-BLOCK-START" in current_section
    assert "# EVOLVE-BLOCK-END" in current_section
    assert negative_section == "NEGATIVE APPENDIX\nNone."


def test_openrouter_generation_prompt_renders_recent_negative_trials():
    backend = OpenRouterGenerationBackend(api_key="test-key")

    prompt_messages = backend._build_prompt(
        _track_with_pool(),
        _manifest(),
        _context_with_prior_programs(),
        negative_trials=_negative_trials(),
        selected_config={"model": "test/model"},
    )

    negative_section = prompt_messages[3]["content"]

    assert negative_section.startswith("NEGATIVE APPENDIX")
    assert "NEGATIVE reason=crashed detail=" in negative_section
    assert "returncode=1" in negative_section
    assert "mat1 and mat2 shapes cannot be multiplied" in negative_section
    assert "raise RuntimeError('bad candidate')" in negative_section
    assert "```python" not in negative_section
    assert "class TrainScriptContractError(RuntimeError):" not in negative_section


def test_openrouter_generation_prompt_forbids_copying_reference_or_negative_examples():
    backend = OpenRouterGenerationBackend(api_key="test-key")

    prompt_messages = backend._build_prompt(
        _track_with_pool(),
        _manifest(),
        _context_with_prior_programs(),
        negative_trials=_negative_trials(),
        selected_config={"model": "test/model"},
    )

    system_prompt = prompt_messages[0]["content"]
    assert "Never copy any shown REFERENCE or NEGATIVE example verbatim" in (
        system_prompt
    )
    assert "Use REFERENCE appendices as inspiration only" in system_prompt
    assert "textually distinct from CURRENT_PROGRAM" in system_prompt


def test_openrouter_generation_prompt_keeps_stable_prefix_across_context_changes():
    backend = OpenRouterGenerationBackend(api_key="test-key")

    first_prompt = backend._build_prompt(
        _track_with_pool(),
        _manifest(),
        _context_with_prior_programs(),
        negative_trials=_negative_trials(),
        selected_config={"model": "test/model"},
    )
    second_prompt = backend._build_prompt(
        _track_with_pool(),
        _manifest(),
        _context_with_alternate_current_and_prior_programs(),
        negative_trials=_many_negative_trials(),
        selected_config={"model": "test/model"},
    )

    assert first_prompt[0]["content"] == second_prompt[0]["content"]
    assert first_prompt[0]["content"] == second_prompt[0]["content"]
    assert first_prompt[1]["content"] == second_prompt[1]["content"]
    assert first_prompt[2]["content"] != second_prompt[2]["content"]
    assert first_prompt[4]["content"] != second_prompt[4]["content"]


def test_openrouter_generation_prompt_applies_render_budgets():
    backend = OpenRouterGenerationBackend(api_key="test-key")

    prompt_messages = backend._build_prompt(
        _track_with_pool(),
        _manifest(),
        _context_with_many_prior_programs(),
        negative_trials=_many_negative_trials(),
        selected_config={"model": "test/model"},
    )

    reference_appendix = prompt_messages[2]["content"]
    negative_appendix = prompt_messages[3]["content"]

    assert reference_appendix.count("REFERENCE val_acc=") == 2
    assert "trial_prior_2" not in reference_appendix
    assert " + 0.5" not in reference_appendix
    assert " + 0.3" in reference_appendix
    assert " + 0.4" in reference_appendix
    assert negative_appendix.count("NEGATIVE reason=") == 4
    assert "bad candidate 2" in negative_appendix
    assert "bad candidate 3" in negative_appendix


def test_openrouter_generation_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    backend = OpenRouterGenerationBackend(api_key=None)

    result = backend.generate(
        _track_with_pool(), _manifest(), _context(), generation_index=0
    )

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
        raise HTTPError(
            req.full_url,
            503,
            "Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"overloaded"}'),
        )

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    result = backend.generate(
        _track_with_pool(), _manifest(), _context(), generation_index=0
    )

    assert result.source is None
    assert result.error_info is not None
    assert result.error_info["reason"] == "provider_http_error"
    assert result.error_info["status_code"] == 503
    assert result.error_info["response_body"] == '{"error":"overloaded"}'


@pytest.mark.parametrize(
    ("response_payload", "expected_reason", "expected_field", "expected_value"),
    [
        (b"{not-json", "provider_response_invalid_json", "response_body", "{not-json"),
        (
            {"id": "resp_1", "choices": []},
            "provider_response_missing_choices",
            "response_body",
            '{"id": "resp_1", "choices": []}',
        ),
        (
            {"id": "resp_1", "choices": [{"message": {"content": "   "}}]},
            "provider_response_missing_content",
            "response_text",
            "   ",
        ),
    ],
)
def test_openrouter_generation_response_errors(
    monkeypatch, response_payload, expected_reason, expected_field, expected_value
):
    backend = OpenRouterGenerationBackend(api_key="test-key")
    monkeypatch.setattr(
        "sigmaevolve.generation.request.urlopen",
        lambda req, timeout=0: _FakeResponse(response_payload),
    )

    result = backend.generate(
        _track_with_pool(), _manifest(), _context(), generation_index=0
    )

    assert result.source is None
    assert result.error_info is not None
    assert result.error_info["reason"] == expected_reason

    # Route the expected assertion to the field that failed for this case.
    if expected_field == "response_text":
        assert result.provenance_json["generation"]["response_text"] == expected_value
    else:
        assert result.error_info[expected_field] == expected_value

    # Preserve provider response ids for the missing-choice failure path.
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

    monkeypatch.setattr(
        "sigmaevolve.generation.request.urlopen", lambda req, timeout=0: FakeResponse()
    )

    result = backend.generate(
        _track_with_pool(), _manifest(), _context(), generation_index=0
    )

    assert result.source is None
    assert result.error_info is not None
    assert result.error_info["reason"] == "provider_response_missing_content"
    assert result.error_info["error_type"] == "generation_reasoning_tokens_exhausted"
    assert result.error_info["finish_reason"] == "length"
    assert result.error_info["reasoning_present"] is True
    assert (
        result.provenance_json["generation"]["reasoning_text"]
        == "internal chain of thought"
    )


def test_openrouter_generation_persists_reasoning_trace(monkeypatch):
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
                    "choices": [
                        {
                            "message": {
                                "content": _mutated_script(
                                    "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.1"
                                ),
                                "reasoning": "I will modify only the evolve block.",
                            },
                        }
                    ],
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "sigmaevolve.generation.request.urlopen", lambda req, timeout=0: FakeResponse()
    )

    result = backend.generate(
        _track_with_pool(), _manifest(), _context(), generation_index=0
    )

    assert result.error_info is None
    assert (
        result.provenance_json["generation"]["reasoning_text"]
        == "I will modify only the evolve block."
    )


def test_openrouter_generation_persists_finish_reason_for_contentful_response(
    monkeypatch,
):
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
                    "provider": "OpenRouter",
                    "model": "google/gemini-3.1-pro-preview",
                    "choices": [
                        {
                            "finish_reason": "length",
                            "native_finish_reason": "length",
                            "message": {
                                "content": "<<<<<<< SEARCH\npartial patch",
                            },
                        }
                    ],
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "sigmaevolve.generation.request.urlopen", lambda req, timeout=0: FakeResponse()
    )

    result = backend.generate(
        _track_with_pool(), _manifest(), _context(), generation_index=0
    )

    assert result.source == "<<<<<<< SEARCH\npartial patch\n"
    assert result.provenance_json["generation"]["finish_reason"] == "length"
    assert result.provenance_json["generation"]["native_finish_reason"] == "length"
    assert result.provenance_json["generation"]["provider"] == "OpenRouter"
    assert (
        result.provenance_json["generation"]["provider_model"]
        == "google/gemini-3.1-pro-preview"
    )


def test_openrouter_generation_captures_transport_errors(monkeypatch):
    backend = OpenRouterGenerationBackend(api_key="test-key")

    def fake_urlopen(req, timeout=0):
        raise URLError("network down")

    monkeypatch.setattr("sigmaevolve.generation.request.urlopen", fake_urlopen)

    result = backend.generate(
        _track_with_pool(), _manifest(), _context(), generation_index=0
    )

    assert result.source is None
    assert result.error_info is not None
    assert result.error_info["reason"] == "provider_request_failed"
    assert "network down" in result.error_info["detail"]


def test_replace_evolve_block_payloads_rewrites_only_block_contents():
    source = build_baseline_train_script()
    payloads = extract_evolve_block_payloads(source)
    updated = replace_evolve_block_payloads(
        source,
        [
            build_model_block(
                """
def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
            ),
        ],
    )

    assert len(payloads) == 1
    assert extract_evolve_block_payloads(updated) != payloads
    assert_only_evolve_blocks_changed(source, updated)


def test_baseline_template_uses_one_outer_evolve_block():
    source = build_baseline_train_script()

    assert source.count("# EVOLVE-BLOCK-START") == 1
    assert source.count("# EVOLVE-BLOCK-END") == 1
    assert len(extract_evolve_block_payloads(source)) == 1


def test_baseline_template_preserves_feature_shape_for_evolved_models():
    namespace: dict[str, object] = {}
    exec(build_baseline_train_script(), namespace)

    build_tensor_datasets = namespace["build_tensor_datasets"]
    make_experiment = namespace["make_experiment"]

    train_ds, validation_ds = build_tensor_datasets(
        np.zeros((4, 28, 28), dtype=np.float32),
        np.array([0, 1, 0, 1], dtype=np.int64),
        np.zeros((2, 28, 28), dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
    )
    experiment = make_experiment(torch.device("cpu"), train_ds, validation_ds)
    logits = experiment["model"](validation_ds.tensors[0])

    assert tuple(train_ds.tensors[0].shape) == (4, 28, 28)
    assert tuple(logits.shape) == (2, 2)
    assert isinstance(experiment["model"][0], torch.nn.Flatten)


def test_build_candidate_train_script_replaces_only_data_block():
    source = build_baseline_train_script()
    source_payloads = extract_evolve_block_payloads(source)
    updated = build_candidate_train_script(
        data_block_payload=build_data_block("""
batch_size = 8
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=1)
""")
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert len(source_payloads) == 1
    assert "batch_size = 8" in updated_payloads[0]
    assert "shuffle=False" in updated_payloads[0]
    assert updated_payloads[0] != source_payloads[0]
    assert_only_evolve_blocks_changed(source, updated)


def test_build_candidate_train_script_replaces_only_optimization_block():
    source = build_baseline_train_script()
    source_payloads = extract_evolve_block_payloads(source)
    updated = build_candidate_train_script(
        optimization_block_payload=build_optimization_block("""
optimizer = None
scheduler = None
""")
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert len(source_payloads) == 1
    assert "optimizer = None" in updated_payloads[0]
    assert "scheduler = None" in updated_payloads[0]
    assert updated_payloads[0] != source_payloads[0]
    assert_only_evolve_blocks_changed(source, updated)


def test_build_candidate_train_script_positional_model_payload_keeps_other_blocks():
    source = build_baseline_train_script()
    updated = build_candidate_train_script(
        build_model_block(
            """
def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
        )
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert len(updated_payloads) == 1
    assert (
        "return torch.zeros((x.shape[0], 2), dtype=torch.float32)"
        in updated_payloads[0]
    )
    assert_only_evolve_blocks_changed(source, updated)


def test_assert_only_evolve_blocks_changed_rejects_immutable_changes():
    source = build_baseline_train_script()
    invalid = source.replace("import json\n", "import json\nBROKEN = True\n", 1)

    # Reject any change that touches immutable text outside evolve blocks.
    with pytest.raises(EvolveBlockError, match="immutable text"):
        assert_only_evolve_blocks_changed(source, invalid)


def test_materialize_candidate_source_applies_search_replace_blocks():
    source = build_baseline_train_script()
    response = """TASK_DESCRIPTION:
Add a hidden layer so the baseline model keeps a little more capacity.

<<<<<<< SEARCH
    nn.Linear(128, num_classes),
=======
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, num_classes),
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert "nn.Linear(128, 64)" in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_materialize_candidate_source_matches_search_blocks_without_outer_indentation():
    source = build_baseline_train_script()
    response = """TASK_DESCRIPTION:
Adjust optimization defaults to try a smaller, more regularized update schedule.

<<<<<<< SEARCH
trainable_parameters = [
    parameter for parameter in model.parameters() if parameter.requires_grad
]
optimizer = None
if trainable_parameters:
    optimizer = torch.optim.Adam(trainable_parameters, lr=1e-3)
=======
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = None
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert "torch.optim.AdamW" in updated
    assert "scheduler = None" in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_assert_only_evolve_blocks_changed_accepts_outer_block_only_patch_layout():
    source = build_baseline_train_script()
    response = """TASK_DESCRIPTION:
Increase the batch size while keeping the rest of the program unchanged.

<<<<<<< SEARCH
batch_size = 64
=======
batch_size = 128
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert "batch_size = 128" in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_build_candidate_train_script_replaces_only_config_block():
    source = build_baseline_train_script()
    source_payloads = extract_evolve_block_payloads(source)
    updated = build_candidate_train_script(
        config_block_payload=build_config_block(
            build_training_policy_block(
                """
early_stopping_patience = 5
min_delta = 0.1
"""
            )
        )
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert "early_stopping_patience = 5" in updated_payloads[0]
    assert updated_payloads[0] != source_payloads[0]
    assert_only_evolve_blocks_changed(source, updated)


def test_apply_search_replace_blocks_preserves_internal_indentation():
    source = build_baseline_train_script()
    response = """TASK_DESCRIPTION:
Raise early stopping patience slightly so the model can train longer before stopping.

<<<<<<< SEARCH
early_stopping_patience = 2
min_delta = 0.0
=======
early_stopping_patience = 5
min_delta = 0.0
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert "    early_stopping_patience = 5" in updated
    assert "    min_delta = 0.0" in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_parse_and_apply_search_replace_blocks_support_no_changes():
    source = build_baseline_train_script()

    blocks = parse_search_replace_blocks(
        "TASK_DESCRIPTION:\nNo safe change stands out over the current program.\n\nNO_CHANGES\n"
    )
    updated = apply_search_replace_blocks(source, blocks)

    assert updated == source


def test_parse_generation_response_extracts_task_description():
    response = """TASK_DESCRIPTION:
Increase hidden layer width to add capacity while preserving the current training loop.

<<<<<<< SEARCH
    "hidden_dims": (256, 128),
=======
    "hidden_dims": (512, 256),
>>>>>>> REPLACE
"""

    parsed = parse_generation_response(response)

    assert (
        parsed.task_description
        == "Increase hidden layer width to add capacity while preserving the current training loop."
    )
    assert parsed.patch_text.startswith("<<<<<<< SEARCH\n")
    assert extract_task_description(response) == parsed.task_description


def test_parse_search_replace_blocks_rejects_evolve_markers_in_search_text():
    response = """TASK_DESCRIPTION:
Try a replacement, but this one is invalid because it targets evolve markers.

<<<<<<< SEARCH
# EVOLVE-BLOCK-START
=======
replacement
>>>>>>> REPLACE
"""

    # Reject search text that already contains evolve markers.
    with pytest.raises(
        EvolveBlockError, match="may not include evolve block marker lines"
    ):
        parse_search_replace_blocks(response)


def test_parse_search_replace_blocks_rejects_evolve_markers_in_replace_text():
    response = """TASK_DESCRIPTION:
Try a replacement, but this one is invalid because it emits evolve markers.

<<<<<<< SEARCH
original
=======
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
"""

    # Reject replacement text that already contains evolve markers.
    with pytest.raises(
        EvolveBlockError, match="may not include evolve block marker lines"
    ):
        parse_search_replace_blocks(response)


def test_materialize_candidate_source_rejects_non_patch_without_full_program():
    source = build_baseline_train_script()

    # Reject plain text responses that do not include SEARCH/REPLACE blocks.
    with pytest.raises(EvolveBlockError, match="SEARCH/REPLACE blocks"):
        materialize_candidate_source(source, "return self.network(x)\n")


def test_parse_search_replace_blocks_requires_task_description_prefix():
    response = """<<<<<<< SEARCH
original
=======
replacement
>>>>>>> REPLACE
"""

    with pytest.raises(EvolveBlockError, match="TASK_DESCRIPTION"):
        parse_search_replace_blocks(response)
