from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Add the repository root so test imports resolve the local package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigmaevolve.datasets import ArrayDatasetProvider, DatasetManager  # noqa: E402
from sigmaevolve.execution import RunnerService  # noqa: E402
from sigmaevolve.generation import (  # noqa: E402
    FixedGenerationBackend,
    build_candidate_train_script,
    build_model_block,
)
from sigmaevolve.orchestration import EvolutionSystem  # noqa: E402
from sigmaevolve.storage import SQLAlchemyRepository  # noqa: E402
from tests.support import RecordingLauncherDouble  # noqa: E402


@pytest.fixture
def fake_wandb(monkeypatch):
    state: dict[str, object] = {
        "login_calls": [],
        "runs": [],
    }

    class FakeRun:
        def __init__(self, **kwargs) -> None:
            runs = state["runs"]
            assert isinstance(runs, list)
            index = len(runs) + 1
            self.project = kwargs.get("project")
            self.entity = kwargs.get("entity")
            self.id = f"wandb-run-{index}"
            self.name = kwargs.get("name") or self.id
            self.url = f"https://wandb.example/{self.project}/{self.id}"
            self.config = kwargs.get("config")
            self.tags = kwargs.get("tags")
            self.job_type = kwargs.get("job_type")
            self.logged: list[dict[str, object]] = []
            self.artifacts: list[dict[str, object]] = []
            self.summary: dict[str, object] = {}
            self.finished: dict[str, object] | None = None
            self.events: list[dict[str, object]] = []

        def log(self, payload, step=None) -> None:
            self.logged.append({"payload": dict(payload), "step": step})
            self.events.append({"kind": "log", "payload": dict(payload), "step": step})

        def log_artifact(self, artifact, aliases=None) -> None:
            entry = {
                "artifact": artifact,
                "aliases": list(aliases or []),
            }
            self.artifacts.append(entry)
            self.events.append({"kind": "artifact", **entry})

        def finish(self, exit_code=None) -> None:
            self.finished = {"exit_code": exit_code}
            self.events.append({"kind": "finish", "exit_code": exit_code})

    class FakeArtifact:
        def __init__(self, name, type, metadata=None) -> None:
            self.name = name
            self.type = type
            self.metadata = dict(metadata or {})
            self.files: list[dict[str, object]] = []

        def add_file(self, local_path, name=None) -> None:
            self.files.append({"local_path": local_path, "name": name})

    module = types.ModuleType("wandb")

    def login(*, key=None, relogin=None):
        login_calls = state["login_calls"]
        assert isinstance(login_calls, list)
        login_calls.append({"key": key, "relogin": relogin})
        return True

    def init(**kwargs):
        runs = state["runs"]
        assert isinstance(runs, list)
        run = FakeRun(**kwargs)
        runs.append(run)
        return run

    module.login = login  # type: ignore[attr-defined]
    module.init = init  # type: ignore[attr-defined]
    module.Artifact = FakeArtifact  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    return state


@pytest.fixture(autouse=True)
def _install_fake_wandb(fake_wandb):
    return fake_wandb


def make_policy(**overrides):
    policy = {
        "epochs": 3,
        "dispatch_ttl_sec": 1,
        "heartbeat_interval_sec": 1,
        "stale_ttl_sec": 1,
        "max_dispatch_retries": 1,
        "sampling_seed": 0,
        "generation_backend": {
            "selection": "round_robin",
            "model_pool": [
                {
                    "model": "test/model",
                    "temperature": 0.1,
                    "max_tokens": 1500,
                    "retry_count": 1,
                }
            ],
        },
    }
    policy.update(overrides)
    return policy


def make_provider(seed: int) -> ArrayDatasetProvider:
    rng = np.random.default_rng(seed)
    train_features = rng.normal(size=(12, 4)).astype(np.float32)
    train_labels = (train_features.sum(axis=1) > 0).astype(np.int64)
    validation_features = rng.normal(size=(6, 4)).astype(np.float32)
    validation_labels = (validation_features.sum(axis=1) > 0).astype(np.int64)
    test_features = rng.normal(size=(5, 4)).astype(np.float32)
    test_labels = (test_features.sum(axis=1) > 0).astype(np.int64)
    return ArrayDatasetProvider(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        test_features=test_features,
        test_labels=test_labels,
        metadata={"num_classes": 2, "feature_shape": [4]},
    )


@pytest.fixture
def providers():
    return {
        "mnist:v1": make_provider(seed=7),
        "fashion_mnist:v1": make_provider(seed=21),
    }


@pytest.fixture
def repository(tmp_path):
    return SQLAlchemyRepository(f"sqlite:///{tmp_path / 'sigmaevolve.sqlite'}")


@pytest.fixture
def dataset_manager(tmp_path, providers):
    root = Path(tmp_path) / "datasets"
    return DatasetManager(root, providers)


@pytest.fixture
def system(repository, dataset_manager):
    # Build the launcher double used by orchestration tests.
    launcher = RecordingLauncherDouble()

    # Build the default candidate generator used in integration fixtures.
    model_source = build_model_block(
        """
def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    return torch.stack((-scores, scores), dim=1)
"""
    )
    generation_source = build_candidate_train_script(model_source)
    generator = FixedGenerationBackend(
        source=generation_source,
    )

    # Assemble the service layer used by end-to-end tests.
    runner_service = RunnerService(
        repository=repository, dataset_manager=dataset_manager
    )
    return EvolutionSystem(
        repository,
        dataset_manager,
        generator,
        launcher,
        runner_service,
    )
