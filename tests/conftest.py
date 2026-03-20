from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigmaevolve.datasets import ArrayDatasetProvider, DatasetManager
from sigmaevolve.generation import FixedGenerationBackend
from sigmaevolve.orchestrator import RecordingLauncher
from sigmaevolve.runner import RunnerService
from sigmaevolve.storage import SQLAlchemyRepository
from sigmaevolve.system import EvolutionSystem


def make_policy(**overrides):
    policy = {
        "budget_sec": 2,
        "max_eval_gap_sec": 1,
        "max_parallelism": 1,
        "ready_queue_threshold": 1,
        "dispatch_ttl_sec": 1,
        "heartbeat_interval_sec": 1,
        "stale_ttl_sec": 1,
        "max_dispatch_retries": 1,
        "scorer_settings": {"primary_metric": "accuracy"},
        "sampling_settings": {"strategy": "top_then_random", "top_k": 3, "seed": 0},
        "generation_backend": {
            "backend": "openrouter",
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
    launcher = RecordingLauncher()
    generator = FixedGenerationBackend(
        source=(
            "import argparse, json\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "parser=argparse.ArgumentParser(); parser.add_argument('--config', required=True); "
            "args=parser.parse_args(); "
            "cfg=json.loads(Path(args.config).read_text()); "
            "labels=np.load(cfg['validation_split_path'])['features']; "
            "preds=(labels.sum(axis=1) > 0).astype(int); "
            "np.savez(cfg['predictions_output_path'], predictions=preds)\n"
        )
    )
    runner_service = RunnerService(repository=repository, dataset_manager=dataset_manager)
    return EvolutionSystem(repository, dataset_manager, generator, launcher, runner_service)
