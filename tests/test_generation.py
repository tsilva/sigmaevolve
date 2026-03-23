from __future__ import annotations

import json

import pytest

from sigmaevolve.baseline import build_baseline_train_script
from sigmaevolve.evolve_blocks import replace_evolve_block_payloads
from sigmaevolve.generation import OpenRouterGenerationBackend
from sigmaevolve.models import DatasetManifest, TrackRecord, TrialSummary, now_utc


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
            source=_mutated_script("return model(val_x)\n"),
            provenance_json={"backend": "baseline", "candidate_kind": "strategy_v1"},
        )
    ]


def _negative_trials():
    return [
        TrialSummary(
            trial_id="trial_failed",
            score=0.0,
            metrics_json=None,
            source=_mutated_script("raise RuntimeError('bad candidate')\n"),
            provenance_json={"backend": "openrouter", "model": "test/model"},
            outcome_reason="crashed",
            error_json={
                "returncode": 1,
                "stderr": "RuntimeError: mat1 and mat2 shapes cannot be multiplied (55000x28 and 784x128)",
            },
        )
    ]


def _mutated_script(predict_body: str) -> str:
    return replace_evolve_block_payloads(
        build_baseline_train_script(),
        [
            (
                "def build_state(*, train_features, train_labels, validation_features, dataset_metadata, random_seed, device):\n"
                "    train_x = train_features.reshape(train_features.shape[0], -1)\n"
                "    val_x = validation_features.reshape(validation_features.shape[0], -1)\n"
                "    train_y = train_labels.astype(np.int64)\n"
                "    num_classes = int(dataset_metadata.get(\"num_classes\") or (np.max(train_y) + 1))\n"
                "    torch.manual_seed(int(random_seed))\n"
                "    model = torch.nn.Linear(int(train_x.shape[1]), num_classes)\n"
                "    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)\n"
                "    criterion = torch.nn.CrossEntropyLoss()\n"
                "    return {\n"
                "        \"model\": model,\n"
                "        \"optimizer\": optimizer,\n"
                "        \"criterion\": criterion,\n"
                "        \"train_x\": torch.from_numpy(train_x),\n"
                "        \"train_y\": torch.from_numpy(train_y),\n"
                "        \"val_x\": torch.from_numpy(val_x),\n"
                "        \"steps_per_epoch\": 5,\n"
                "    }\n\n"
                "def train_epoch(state, *, epoch_index, num_epochs):\n"
                "    return None\n\n"
                "def predict_validation(state, validation_features):\n"
                f"    {predict_body.strip()}\n"
            )
        ],
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
                    "choices": [{"message": {"content": _mutated_script("return model(val_x) + 0.1\n")}}],
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
    assert "candidate module: train.py" in system_prompt
    assert "Treat this as an evolutionary mutation task, not a rewrite from scratch." in system_prompt
    assert "Follow this contract exactly:" in system_prompt
    assert "Only change code between matching evolve block markers." in system_prompt
    assert "Keep every non-evolve line identical to the parent source." in system_prompt
    assert "Produce a mutated descendant of the parent block implementation, not a fresh rewrite of the file." in system_prompt
    assert not first_prompt.lstrip().startswith("{")
    assert "Write a complete Python train.py module for dataset mnist:v1." in first_prompt
    assert "- epochs: 5" in first_prompt
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
                    "choices": [{"message": {"content": _mutated_script("return model(val_x)\n")}}],
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
                    "choices": [{"message": {"content": _mutated_script("return model(val_x)\n")}}],
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
    assert "Make exactly one substantive improvement likely to improve validation accuracy within the fixed epoch budget." in system_prompt
    assert "Trial trial_failed:" in prompt
    assert "- returncode: 1" in prompt
    assert "mat1 and mat2 shapes cannot be multiplied" in prompt
