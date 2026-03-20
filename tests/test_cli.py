from __future__ import annotations

import json
import os
from types import SimpleNamespace

from sigmaevolve.cli import main
from sigmaevolve.storage import normalize_database_url


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


def test_cli_modal_commands_call_support_helpers(tmp_path, monkeypatch):
    from sigmaevolve import cli as cli_module

    deployed = {}
    synced = {}

    def fake_deploy_modal_app(**kwargs):
        deployed.update(kwargs)
        return {"ok": True, **kwargs}

    def fake_sync_dataset_to_modal(**kwargs):
        synced.update(kwargs)
        return {"ok": True, **kwargs}

    monkeypatch.setattr(cli_module, "deploy_modal_app", fake_deploy_modal_app)
    monkeypatch.setattr(cli_module, "sync_dataset_to_modal", fake_sync_dataset_to_modal)

    assert main(["--modal-app-name", "custom-app", "modal-deploy"]) == 0
    assert deployed["app_name"] == "custom-app"

    assert main(["--dataset-root", str(tmp_path / "datasets"), "modal-sync-dataset", "mnist:v1"]) == 0
    assert synced["dataset_id"] == "mnist:v1"


def test_make_system_with_modal_launcher_uses_modal_proxy(monkeypatch, tmp_path):
    from sigmaevolve import cli as cli_module

    captured = {}

    def fake_create_modal_launcher(**kwargs):
        captured.update(kwargs)
        return object()

    def fake_build_system(**kwargs):
        orchestrator = SimpleNamespace(launcher=None)
        return SimpleNamespace(launcher=None, orchestrator=orchestrator)

    monkeypatch.setattr(cli_module, "create_modal_launcher", fake_create_modal_launcher)
    monkeypatch.setattr(cli_module, "build_system", fake_build_system)
    args = cli_module.build_parser().parse_args(
        [
            "--database-url",
            "postgresql://example/db",
            "--launcher",
            "modal",
            "--modal-app-name",
            "sigmaevolve-runner",
            "list-trials",
            "track_1",
        ]
    )
    system = cli_module._make_system(args)
    assert captured["database_url"] == "postgresql://example/db"
    assert system.launcher is not None


def test_database_url_defaults_from_env(monkeypatch):
    from sigmaevolve import cli as cli_module

    monkeypatch.setenv("SIGMAEVOLVE_DATABASE_URL", "postgresql://example/db")
    parser = cli_module.build_parser()
    args = parser.parse_args(["list-trials", "track_1"])
    assert args.database_url == "postgresql://example/db"


def test_normalize_database_url_accepts_neon_postgres_scheme():
    assert normalize_database_url("postgresql://example/db").startswith("postgresql+psycopg://")
    assert normalize_database_url("postgres://example/db").startswith("postgresql+psycopg://")
