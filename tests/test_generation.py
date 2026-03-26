from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest

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
    extract_evolve_block_payloads,
    materialize_candidate_source,
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
    assert (
        first_result.provenance_json["generation"]["system_prompt"]
        == payloads[0]["messages"][0]["content"]
    )
    assert (
        first_result.provenance_json["generation"]["user_prompt"]
        == payloads[0]["messages"][1]["content"]
    )
    assert (
        "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.1"
        in first_result.provenance_json["generation"]["response_text"]
    )
    assert first_result.error_info is None

    system_prompt = payloads[0]["messages"][0]["content"]
    first_prompt = payloads[0]["messages"][1]["content"]
    assert "# EVOLVE-BLOCK-START" in system_prompt
    assert "# EVOLVE-BLOCK-END" in system_prompt
    assert (
        "Never wrap the response in triple backticks or fenced code blocks"
        in system_prompt
    )
    assert (
        "If emitting a patch, begin immediately with <<<<<<< SEARCH on the first line"
        in system_prompt
    )
    assert "If you cannot emit a complete SEARCH/REPLACE block, output NO_CHANGES" in (
        system_prompt
    )
    assert (
        "Do not emit leading spaces or tabs that only reflect surrounding block nesting"
        in system_prompt
    )
    assert (
        "SEARCH must match exactly one location in the CURRENT PROGRAM" in system_prompt
    )
    assert not first_prompt.lstrip().startswith("{")
    assert "OBJECTIVE:" in first_prompt
    assert "TASK CONTEXT:" in first_prompt
    assert "- dataset_id: mnist:v1" in first_prompt
    assert "- epochs: 5" in first_prompt
    assert "- split_sizes:" in first_prompt
    assert "- dataset_metadata:" in first_prompt
    assert "- num_classes: 10" in first_prompt
    assert "REFERENCE PROGRAMS:" in first_prompt
    assert "CURRENT PROGRAM:" in first_prompt
    assert "Optimize for higher val_acc." in first_prompt
    assert "Use REFERENCE PROGRAMS as inspiration only." in first_prompt
    assert (
        "Patch this program. SEARCH blocks must match text from CURRENT PROGRAM"
        in first_prompt
    )
    assert "score:" not in first_prompt
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
    prompt = payloads[0]["messages"][1]["content"]
    assert "# EVOLVE-BLOCK-START" in system_prompt
    assert "# EVOLVE-BLOCK-END" in system_prompt
    assert "OBJECTIVE:" in prompt
    assert "TASK CONTEXT:" in prompt
    assert "REFERENCE PROGRAMS:" in prompt
    assert "CURRENT PROGRAM:" in prompt
    assert "REPLACEMENTS:" in prompt


def test_openrouter_generation_prompt_lists_full_prior_programs_before_current_program():
    backend = OpenRouterGenerationBackend(api_key="test-key")

    prompt = backend._build_user_prompt_text(
        _track_with_pool(),
        _manifest(),
        _context_with_prior_programs(),
        negative_trials=[],
        selected_config={"model": "test/model"},
    )

    prior_section, current_section = prompt.split("CURRENT PROGRAM:\n", maxsplit=1)
    assert "OBJECTIVE:" in prior_section
    assert "TASK CONTEXT:" in prior_section
    assert "REFERENCE PROGRAMS:\n---\nval_acc: 0.998" in prior_section
    assert "score:" not in prior_section
    assert "val_acc: 0.998" in prior_section
    assert "val_loss: 0.023" in prior_section
    assert "[...]" not in prior_section
    assert "def forward(self, x):" in prior_section
    assert (
        "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.3"
        in prior_section
    )
    assert "return torch.zeros((x.shape[0], 10), dtype=torch.float32) + 0.2" not in (
        prior_section
    )
    assert (
        "CURRENT PROGRAM:\nPatch this program. SEARCH blocks must match text from CURRENT PROGRAM"
        in prompt
    )
    assert "score:" not in current_section
    assert "val_acc: 0.992" in current_section
    assert "val_loss: 0.1" in current_section
    assert "# EVOLVE-BLOCK-START" not in prior_section
    assert "# EVOLVE-BLOCK-END" not in prior_section
    assert "# EVOLVE-BLOCK-START" in current_section
    assert "# EVOLVE-BLOCK-END" in current_section
    assert prompt.rstrip().endswith("REPLACEMENTS:")


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
            payloads[0],
            build_model_block(
                """
def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
            ),
            payloads[2],
            payloads[3],
            payloads[4],
        ],
    )

    assert len(payloads) == 5
    assert extract_evolve_block_payloads(updated) != payloads
    assert_only_evolve_blocks_changed(source, updated)


def test_baseline_template_uses_one_outer_evolve_block():
    source = build_baseline_train_script()

    assert source.count("# EVOLVE-BLOCK-START") == 1
    assert source.count("# EVOLVE-BLOCK-END") == 1
    assert "# EVOLVE-SECTION-START:" not in source
    assert "# EVOLVE-SECTION-END:" not in source


def test_build_candidate_train_script_replaces_only_data_block():
    source = build_baseline_train_script()
    source_payloads = extract_evolve_block_payloads(source)
    updated = build_candidate_train_script(
        data_block_payload=build_data_block("""
batch_size = 8
return {
    "batch_size": batch_size,
    "train_loader": torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=False,
    ),
    "validation_loader": torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(validation_x),
        batch_size=1,
        shuffle=False,
    ),
}
""")
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert updated_payloads[0] == source_payloads[0]
    assert updated_payloads[1] == source_payloads[1]
    assert updated_payloads[3] == source_payloads[3]
    assert updated_payloads[4] == source_payloads[4]
    assert "batch_size = 8" in updated_payloads[2]
    assert_only_evolve_blocks_changed(source, updated)


def test_build_candidate_train_script_replaces_only_optimization_block():
    source = build_baseline_train_script()
    source_payloads = extract_evolve_block_payloads(source)
    updated = build_candidate_train_script(
        optimization_block_payload=build_optimization_block("""
return {
    "trainable_parameters": [parameter for parameter in model.parameters() if parameter.requires_grad],
    "optimizer": None,
    "scheduler": None,
    "label_smoothing": 0.0,
    "grad_clip_norm": None,
}
""")
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert updated_payloads[0] == source_payloads[0]
    assert updated_payloads[1] == source_payloads[1]
    assert updated_payloads[2] == source_payloads[2]
    assert updated_payloads[4] == source_payloads[4]
    assert '"optimizer": None' in updated_payloads[3]
    assert_only_evolve_blocks_changed(source, updated)


def test_build_candidate_train_script_positional_model_payload_keeps_other_blocks():
    source = build_baseline_train_script()
    source_payloads = extract_evolve_block_payloads(source)
    updated = build_candidate_train_script(
        build_model_block(
            """
def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
        )
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert updated_payloads[0] == source_payloads[0]
    assert (
        "return torch.zeros((x.shape[0], 2), dtype=torch.float32)"
        in updated_payloads[1]
    )
    assert updated_payloads[2:] == source_payloads[2:]
    assert_only_evolve_blocks_changed(source, updated)


def test_assert_only_evolve_blocks_changed_rejects_immutable_changes():
    source = build_baseline_train_script()
    invalid = source.replace("import json\n", "import json\nBROKEN = True\n", 1)

    # Reject any change that touches immutable text outside evolve blocks.
    with pytest.raises(EvolveBlockError, match="immutable text"):
        assert_only_evolve_blocks_changed(source, invalid)


def test_materialize_candidate_source_applies_search_replace_blocks():
    source = build_baseline_train_script()
    response = """<<<<<<< SEARCH
    def forward(self, x):
        return self.network(x)
=======
    def forward(self, x):
        return self.network(x) * 0.5
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert "return self.network(x) * 0.5" in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_materialize_candidate_source_matches_search_blocks_without_outer_indentation():
    source = build_baseline_train_script()
    response = """<<<<<<< SEARCH
    "learning_rate": 0.002,
    "weight_decay": 1e-4,
    "label_smoothing": 0.0,
    "grad_clip_norm": 1.0,
=======
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "label_smoothing": 0.1,
    "grad_clip_norm": 0.5,
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert '"learning_rate": 0.001' in updated
    assert '        "learning_rate": 0.001,' in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_assert_only_evolve_blocks_changed_accepts_outer_block_only_patch_layout():
    source = build_baseline_train_script()
    response = """<<<<<<< SEARCH
"model": {
    "hidden_dims": (256, 128),
},
=======
"model": {
    "hidden_dims": (512, 256),
},
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert '"hidden_dims": (512, 256)' in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_build_candidate_train_script_replaces_only_config_block():
    source = build_baseline_train_script()
    source_payloads = extract_evolve_block_payloads(source)
    updated = build_candidate_train_script(
        config_block_payload=build_config_block("""
CONFIG = {
    "normalization_std_floor": 1e-5,
    "binary_probability_threshold": 0.55,
    "binary_logit_threshold": 0.1,
    "initial_best_accuracy": -1.0,
    "accuracy_improvement_tol": 1e-8,
    "model": {
        "hidden_dims": (256, 128),
    },
    "data": {
        "max_batch_size": 512,
        "shuffle_train": True,
        "shuffle_validation": False,
    },
    "optimization": {
        "learning_rate": 0.002,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "grad_clip_norm": 1.0,
    },
    "training_policy": {
        "early_stopping_patience": 2,
    },
}
""")
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert '"binary_probability_threshold": 0.55' in updated_payloads[0]
    assert updated_payloads[1:] == source_payloads[1:]
    assert_only_evolve_blocks_changed(source, updated)


def test_apply_search_replace_blocks_preserves_internal_indentation():
    source = build_baseline_train_script()
    response = """<<<<<<< SEARCH
def configure_training_policy(*, num_epochs):
    del num_epochs
    training_policy = CONFIG["training_policy"]
    return {
        "early_stopping_patience": training_policy["early_stopping_patience"],
    }
=======
def configure_training_policy(*, num_epochs):
    del num_epochs
    training_policy = CONFIG["training_policy"]
    return {
        "early_stopping_patience": training_policy["early_stopping_patience"] + 3,
    }
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert (
        '        "early_stopping_patience": training_policy["early_stopping_patience"] + 3,'
        in updated
    )
    assert "    del num_epochs" in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_parse_and_apply_search_replace_blocks_support_no_changes():
    source = build_baseline_train_script()

    blocks = parse_search_replace_blocks("NO_CHANGES")
    updated = apply_search_replace_blocks(source, blocks)

    assert updated == source


def test_parse_search_replace_blocks_rejects_evolve_markers_in_search_text():
    response = """<<<<<<< SEARCH
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
    response = """<<<<<<< SEARCH
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
