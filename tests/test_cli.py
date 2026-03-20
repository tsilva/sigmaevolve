from __future__ import annotations

import json
import os

from sigmaevolve.cli import main


def test_cli_create_track_and_list_trials(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'cli.sqlite'}"
    dataset_root = tmp_path / "datasets"

    from sigmaevolve.datasets import ArrayDatasetProvider
    import numpy as np

    provider = ArrayDatasetProvider(
        train_features=np.ones((4, 2), dtype=np.float32),
        train_labels=np.array([0, 1, 0, 1], dtype=np.int64),
        validation_features=np.ones((2, 2), dtype=np.float32),
        validation_labels=np.array([0, 1], dtype=np.int64),
        test_features=np.ones((2, 2), dtype=np.float32),
        test_labels=np.array([0, 1], dtype=np.int64),
        metadata={"num_classes": 2},
    )

    # Reuse the CLI entrypoint but patch system construction to avoid torchvision downloads.
    from sigmaevolve import cli as cli_module
    from sigmaevolve.system import build_system

    def fake_make_system(args):
        return build_system(
            database_url=args.database_url,
            dataset_root=args.dataset_root,
            providers={"mnist:v1": provider, "fashion_mnist:v1": provider},
        )

    monkeypatch.setattr(cli_module, "_make_system", fake_make_system)

    assert main(["--database-url", db_url, "--dataset-root", str(dataset_root), "prepare-dataset", "mnist:v1"]) == 0

    from io import StringIO
    import contextlib

    create_out = StringIO()
    with contextlib.redirect_stdout(create_out):
        assert (
            main(
                [
                    "--database-url",
                    db_url,
                    "--dataset-root",
                    str(dataset_root),
                    "create-track",
                    "mnist:v1",
                    "--name",
                    "cli-test",
                ]
            )
            == 0
        )
    track_id = json.loads(create_out.getvalue())["track_id"]

    list_out = StringIO()
    with contextlib.redirect_stdout(list_out):
        assert main(["--database-url", db_url, "--dataset-root", str(dataset_root), "list-trials", track_id]) == 0
    trials = json.loads(list_out.getvalue())
    assert len(trials) == 1
    assert trials[0]["status"] == "queued"


def test_cli_create_track_from_policy_file(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'cli-policy.sqlite'}"
    dataset_root = tmp_path / "datasets"
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "generation_backend": {
                    "backend": "openrouter",
                    "selection": "round_robin",
                    "model_pool": [
                        {"model": "openai/gpt-4o-mini", "temperature": 0.2, "max_tokens": 1000, "retry_count": 1},
                        {"model": "anthropic/claude-3.5-sonnet", "temperature": 0.7, "max_tokens": 2000, "retry_count": 1},
                    ],
                }
            }
        )
    )

    from sigmaevolve.datasets import ArrayDatasetProvider
    import numpy as np
    from sigmaevolve import cli as cli_module
    from sigmaevolve.system import build_system

    provider = ArrayDatasetProvider(
        train_features=np.ones((4, 2), dtype=np.float32),
        train_labels=np.array([0, 1, 0, 1], dtype=np.int64),
        validation_features=np.ones((2, 2), dtype=np.float32),
        validation_labels=np.array([0, 1], dtype=np.int64),
        test_features=np.ones((2, 2), dtype=np.float32),
        test_labels=np.array([0, 1], dtype=np.int64),
        metadata={"num_classes": 2},
    )

    def fake_make_system(args):
        return build_system(
            database_url=args.database_url,
            dataset_root=args.dataset_root,
            providers={"mnist:v1": provider, "fashion_mnist:v1": provider},
        )

    monkeypatch.setattr(cli_module, "_make_system", fake_make_system)

    assert main(["--database-url", db_url, "--dataset-root", str(dataset_root), "prepare-dataset", "mnist:v1"]) == 0

    from io import StringIO
    import contextlib

    create_out = StringIO()
    with contextlib.redirect_stdout(create_out):
        assert (
            main(
                [
                    "--database-url",
                    db_url,
                    "--dataset-root",
                    str(dataset_root),
                    "create-track",
                    "mnist:v1",
                    "--policy-file",
                    str(policy_file),
                ]
            )
            == 0
        )
    track = json.loads(create_out.getvalue())
    pool = track["policy_json"]["generation_backend"]["model_pool"]
    assert len(pool) == 2
    assert pool[1]["model"] == "anthropic/claude-3.5-sonnet"


def test_cli_loads_env_file_for_defaults(tmp_path, monkeypatch):
    env_dir = tmp_path / ".config" / "sigmaevolve"
    env_dir.mkdir(parents=True)
    env_file = env_dir / ".env"
    db_path = tmp_path / "from-env.sqlite"
    dataset_root = tmp_path / "env-datasets"
    env_file.write_text(
        "\n".join(
            [
                f"OPENROUTER_API_KEY=test-key",
                f"SIGMAEVOLVE_SENTINEL=loaded",
                f"UNRELATED_PATH={dataset_root}",
            ]
        )
    )

    from sigmaevolve import cli as cli_module
    from sigmaevolve.env import load_env_file

    original_loader = load_env_file

    def fake_loader(path=None, override=False):
        return original_loader(env_file, override=override)

    monkeypatch.setattr(cli_module, "load_env_file", fake_loader)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SIGMAEVOLVE_SENTINEL", raising=False)

    assert main(["--database-url", f"sqlite:///{db_path}", "--dataset-root", str(dataset_root), "list-trials", "missing"]) == 0
    assert os.environ["OPENROUTER_API_KEY"] == "test-key"
    assert os.environ["SIGMAEVOLVE_SENTINEL"] == "loaded"
