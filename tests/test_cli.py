from __future__ import annotations

import contextlib
import json
import os
import re
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest

from sigmaevolve import cli as cli_module
from sigmaevolve.cli import CliReconcileReporter, main
from sigmaevolve.core import DEFAULT_GENERATION_MODEL
from sigmaevolve.datasets import ArrayDatasetProvider
from sigmaevolve.env import load_env_file, resolve_runtime_config
from sigmaevolve.generation import build_baseline_train_script
from sigmaevolve.model_pools import MNIST_MODEL_POOL_ID
from sigmaevolve.orchestration import build_system
from sigmaevolve.storage import SQLAlchemyRepository, normalize_database_url
from tests.support import build_selfcontained_train_script


def _write_script_file(tmp_path, source: str, filename: str = "train.py") -> str:
    script_path = tmp_path / filename
    script_path.write_text(source)
    return str(script_path)


def _make_provider():
    return ArrayDatasetProvider(
        train_features=np.ones((4, 2), dtype=np.float32),
        train_labels=np.array([0, 1, 0, 1], dtype=np.int64),
        validation_features=np.ones((2, 2), dtype=np.float32),
        validation_labels=np.array([0, 1], dtype=np.int64),
        test_features=np.ones((2, 2), dtype=np.float32),
        test_labels=np.array([0, 1], dtype=np.int64),
        metadata={"num_classes": 2},
    )


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out = StringIO()
    err = StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _load_trailing_json(stdout: str) -> dict[str, object]:
    payload_start = stdout.find("{")
    assert payload_start >= 0
    return json.loads(stdout[payload_start:])


def _track_id_from_stderr(stderr: str) -> str:
    match = re.search(r"Created track (\S+)\.", stderr)
    assert match is not None
    return match.group(1)


def _set_runtime_env(
    monkeypatch,
    *,
    database_url: str | None = None,
    dataset_root: str | None = None,
    openrouter_api_key: str | None = None,
    modal_app_name: str | None = None,
    modal_function_name: str | None = None,
    modal_dataset_volume: str | None = None,
    modal_dataset_mount: str | None = None,
    modal_environment_name: str | None = None,
) -> None:
    # Clear both scoped and fallback env names so each test owns its runtime config.
    for key in (
        "SIGMAEVOLVE_DATABASE_URL",
        "DATABASE_URL",
        "SIGMAEVOLVE_DATASET_ROOT",
        "SIGMAEVOLVE_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
        "SIGMAEVOLVE_MODAL_APP_NAME",
        "SIGMAEVOLVE_MODAL_FUNCTION_NAME",
        "SIGMAEVOLVE_MODAL_DATASET_VOLUME",
        "SIGMAEVOLVE_MODAL_DATASET_MOUNT",
        "SIGMAEVOLVE_MODAL_ENVIRONMENT_NAME",
    ):
        monkeypatch.delenv(key, raising=False)

    values = {
        "SIGMAEVOLVE_DATABASE_URL": database_url,
        "SIGMAEVOLVE_DATASET_ROOT": None if dataset_root is None else str(dataset_root),
        "SIGMAEVOLVE_OPENROUTER_API_KEY": openrouter_api_key,
        "SIGMAEVOLVE_MODAL_APP_NAME": modal_app_name,
        "SIGMAEVOLVE_MODAL_FUNCTION_NAME": modal_function_name,
        "SIGMAEVOLVE_MODAL_DATASET_VOLUME": modal_dataset_volume,
        "SIGMAEVOLVE_MODAL_DATASET_MOUNT": modal_dataset_mount,
        "SIGMAEVOLVE_MODAL_ENVIRONMENT_NAME": modal_environment_name,
    }
    for key, value in values.items():
        # Set only the config values that the test explicitly opted into.
        if value is not None:
            monkeypatch.setenv(key, value)


@pytest.fixture
def patched_cli_system(monkeypatch):
    provider = _make_provider()

    def fake_make_system(args):
        return build_system(
            database_url=args.database_url,
            dataset_root=args.dataset_root,
            providers={"mnist:v1": provider, "fashion_mnist:v1": provider},
        )

    monkeypatch.setattr(cli_module, "_make_system", fake_make_system)


def test_cli_create_track_and_list_trials(tmp_path, patched_cli_system, monkeypatch):
    del patched_cli_system
    db_url = f"sqlite:///{tmp_path / 'cli.sqlite'}"
    dataset_root = tmp_path / "datasets"
    _set_runtime_env(monkeypatch, database_url=db_url, dataset_root=dataset_root)
    script_path = _write_script_file(tmp_path, build_baseline_train_script())

    code, stdout, stderr = _run_cli(["create-track", script_path])
    assert code == 0
    assert stdout == ""
    track_id = _track_id_from_stderr(stderr)
    code, stdout, _ = _run_cli(["list-trials", track_id])
    assert code == 0
    trials = json.loads(stdout)
    assert len(trials) == 1
    assert trials[0]["status"] == "queued"
    assert trials[0]["time_to_best_eval_sec"] is None


def test_cli_create_track_uses_default_generation_model(
    tmp_path, patched_cli_system, monkeypatch
):
    del patched_cli_system
    db_url = f"sqlite:///{tmp_path / 'cli-default-policy.sqlite'}"
    dataset_root = tmp_path / "datasets"
    _set_runtime_env(monkeypatch, database_url=db_url, dataset_root=dataset_root)
    script_path = _write_script_file(tmp_path, build_baseline_train_script())

    code, stdout, stderr = _run_cli(["create-track", script_path])
    assert code == 0
    assert stdout == ""
    track_id = _track_id_from_stderr(stderr)
    track = SQLAlchemyRepository(db_url).get_track(track_id)
    assert track is not None
    assert "max_parallelism" not in track.policy_json
    assert "ready_queue_threshold" not in track.policy_json
    assert (
        track.policy_json["generation_backend"]["model_pool_id"] == MNIST_MODEL_POOL_ID
    )
    pool = track.policy_json["generation_backend"]["model_pool"]
    assert pool[0]["model"] == DEFAULT_GENERATION_MODEL


def test_cli_create_track_from_script_file(tmp_path, patched_cli_system, monkeypatch):
    del patched_cli_system
    db_url = f"sqlite:///{tmp_path / 'cli-policy.sqlite'}"
    dataset_root = tmp_path / "datasets"
    _set_runtime_env(monkeypatch, database_url=db_url, dataset_root=dataset_root)
    script_path = _write_script_file(
        tmp_path,
        build_selfcontained_train_script(
            track_policy={
                "generation_backend": {
                    "selection": "round_robin",
                    "model_pool_id": MNIST_MODEL_POOL_ID,
                }
            }
        ),
    )

    code, stdout, stderr = _run_cli(["create-track", script_path])
    assert code == 0
    assert stdout == ""
    track_id = _track_id_from_stderr(stderr)
    track = SQLAlchemyRepository(db_url).get_track(track_id)
    assert track is not None
    assert (
        track.policy_json["generation_backend"]["model_pool_id"] == MNIST_MODEL_POOL_ID
    )
    pool = track.policy_json["generation_backend"]["model_pool"]
    assert len(pool) == 8
    assert pool[1]["model"] == "google/gemini-3.1-flash-lite-preview"


def test_cli_create_track_uses_script_defaults(
    tmp_path, patched_cli_system, monkeypatch
):
    del patched_cli_system
    db_url = f"sqlite:///{tmp_path / 'cli-source.sqlite'}"
    dataset_root = tmp_path / "datasets"
    _set_runtime_env(monkeypatch, database_url=db_url, dataset_root=dataset_root)
    script_path = tmp_path / "train.py"
    script_path.write_text(build_selfcontained_train_script(epochs=7))

    code, stdout, stderr = _run_cli(["create-track", str(script_path)])
    assert code == 0
    assert stdout == ""
    track_id = _track_id_from_stderr(stderr)
    repository = SQLAlchemyRepository(db_url)
    track = repository.get_track(track_id)
    baseline = repository.list_trials(track_id)[0]

    assert track is not None
    assert track.policy_json["epochs"] == 7
    assert baseline.source == script_path.read_text()
    assert baseline.provenance_json["candidate_kind"] == "selfcontained_script_v1"
    assert baseline.provenance_json["model"] == "python_train_v1"


def test_cli_rejects_json_track_file(tmp_path, patched_cli_system, monkeypatch):
    del patched_cli_system
    db_url = f"sqlite:///{tmp_path / 'cli-policy-json.sqlite'}"
    dataset_root = tmp_path / "datasets"
    _set_runtime_env(monkeypatch, database_url=db_url, dataset_root=dataset_root)
    track_file = tmp_path / "mnist.json"
    track_file.write_text(json.dumps({"dataset_id": "mnist:v1", "epochs": 5}))

    code, _, stderr = _run_cli(["create-track", str(track_file)])
    assert code == 1
    assert "self-contained script path, not a JSON track file" in stderr


def test_cli_create_track_reports_progress_to_stderr(
    tmp_path, patched_cli_system, monkeypatch
):
    del patched_cli_system
    db_url = f"sqlite:///{tmp_path / 'cli-create-track-progress.sqlite'}"
    dataset_root = tmp_path / "datasets"
    _set_runtime_env(monkeypatch, database_url=db_url, dataset_root=dataset_root)
    script_path = _write_script_file(
        tmp_path,
        build_baseline_train_script(),
        "create-track-progress.py",
    )

    code, stdout, stderr = _run_cli(["create-track", script_path])
    assert code == 0
    assert stdout == ""
    track_id = _track_id_from_stderr(stderr)
    assert "Loading script definition" in stderr
    assert "Ensuring dataset mnist:v1 is prepared." in stderr
    assert "Prepared dataset mnist:v1" in stderr
    assert (
        "Creating track for dataset mnist:v1 and seeding the baseline trial." in stderr
    )
    assert f"Created track {track_id}." in stderr
    assert "Run it with:" in stderr
    assert f"sigmaevolve launch {track_id} 1" in stderr
    assert str(script_path) not in stderr
    assert "--dataset-root" not in stderr
    assert "--launcher" not in stderr


def test_cli_loads_env_file_for_defaults(tmp_path, monkeypatch):
    env_dir = tmp_path / ".config" / "sigmaevolve"
    env_dir.mkdir(parents=True)
    env_file = env_dir / ".env"
    db_path = tmp_path / "from-env.sqlite"
    dataset_root = tmp_path / "env-datasets"
    env_file.write_text(
        "\n".join(
            [
                f"SIGMAEVOLVE_DATABASE_URL=sqlite:///{db_path}",
                f"SIGMAEVOLVE_DATASET_ROOT={dataset_root}",
                "SIGMAEVOLVE_OPENROUTER_API_KEY=test-key",
                "SENTINEL=loaded",
            ]
        )
    )

    original_loader = load_env_file

    def fake_loader(path=None, override=False):
        return original_loader(env_file, override=override)

    monkeypatch.setattr(cli_module, "load_env_file", fake_loader)
    _set_runtime_env(monkeypatch)
    monkeypatch.delenv("SIGMAEVOLVE_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SENTINEL", raising=False)

    assert main(["list-trials", "missing"]) == 0
    assert os.environ["SIGMAEVOLVE_OPENROUTER_API_KEY"] == "test-key"
    assert os.environ["SENTINEL"] == "loaded"


def test_cli_modal_commands_call_support_helpers(tmp_path, monkeypatch):
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

    _set_runtime_env(
        monkeypatch,
        dataset_root=tmp_path / "datasets",
        modal_app_name="custom-app",
        modal_function_name="custom-function",
        modal_dataset_volume="custom-volume",
        modal_dataset_mount="/mnt/custom-datasets",
        modal_environment_name="prod",
    )

    assert main(["modal-deploy"]) == 0
    assert deployed["app_name"] == "custom-app"
    assert deployed["function_name"] == "custom-function"
    assert deployed["dataset_volume_name"] == "custom-volume"
    assert deployed["dataset_mount_path"] == "/mnt/custom-datasets"
    assert deployed["environment_name"] == "prod"

    assert main(["modal-sync-dataset", "mnist:v1"]) == 0
    assert synced["dataset_id"] == "mnist:v1"
    assert synced["dataset_root"] == str(tmp_path / "datasets")
    assert synced["volume_name"] == "custom-volume"
    assert synced["environment_name"] == "prod"


def test_cli_reconcile_reporter_dispatch_table_logs_known_events_and_ignores_unknown(
    monkeypatch,
):
    messages: list[str] = []

    def fake_info(message, *args):
        messages.append(message % args if args else message)

    monkeypatch.setattr("sigmaevolve.cli.logger.info", fake_info)
    reporter = CliReconcileReporter()

    reporter(
        "generation_scheduled",
        {
            "slot_index": 0,
            "generation_index": 7,
            "duplicate_retry_count": 0,
            "sampled_candidates": [
                {
                    "rank": 1,
                    "trial_id": "trial_high",
                    "score": 0.9,
                    "selection_probability": 0.75,
                    "selected_role": "current",
                },
                {
                    "rank": 2,
                    "trial_id": "trial_mid",
                    "score": 0.3,
                    "selection_probability": 0.25,
                    "selected_role": "inspiration",
                },
            ],
        },
    )
    reporter(
        "generation_failed",
        {
            "slot_index": 0,
            "reason": "provider_response_missing_content",
            "duplicate_retry_count": 1,
            "detail": "empty",
            "completed": 0,
            "requested": 1,
            "failures": 1,
            "max_failures": 2,
            "in_flight": 0,
        },
    )
    reporter("unknown_event", {"ignored": True})

    assert any("Sampled candidates:" in message for message in messages)
    assert any(
        "| rank | trial_id | score | p(current) | selected |" in message
        for message in messages
    )
    assert any(
        "| 1 | trial_high | 0.9000 | 0.7500 | current |" in message
        for message in messages
    )
    assert any("Generation failed for slot 1" in message for message in messages)
    assert any("Queue fill [" in message for message in messages)


def test_cli_launch_count_reports_progress_to_stderr(
    tmp_path, patched_cli_system, monkeypatch
):
    del patched_cli_system
    db_url = f"sqlite:///{tmp_path / 'cli-launch-progress.sqlite'}"
    dataset_root = tmp_path / "datasets"
    _set_runtime_env(monkeypatch, database_url=db_url, dataset_root=dataset_root)
    script_path = _write_script_file(
        tmp_path,
        build_baseline_train_script(),
        "launch-track.py",
    )

    _, stdout, stderr = _run_cli(["create-track", script_path])
    assert stdout == ""
    track_id = _track_id_from_stderr(stderr)
    code, stdout, stderr = _run_cli(["launch", track_id, "1"])
    assert code == 0
    payload = _load_trailing_json(stdout)
    assert "launched_trial_ids" in payload
    assert payload["mode"] == "count"
    assert "Running launch pass" in stderr
    assert "Launching reserved trials" in stderr
    assert "Launch pass finished" in stderr


def test_cli_launch_maintain_running_stops_after_max_cycles(
    tmp_path, patched_cli_system, monkeypatch
):
    del patched_cli_system
    db_url = f"sqlite:///{tmp_path / 'cli-launch-maintain.sqlite'}"
    dataset_root = tmp_path / "datasets"
    _set_runtime_env(monkeypatch, database_url=db_url, dataset_root=dataset_root)
    script_path = _write_script_file(
        tmp_path,
        build_baseline_train_script(),
        "maintain-track.py",
    )

    _, stdout, stderr = _run_cli(["create-track", script_path])
    assert stdout == ""
    track_id = _track_id_from_stderr(stderr)
    code, stdout, _ = _run_cli(
        [
            "--launcher",
            "inline",
            "launch",
            track_id,
            "1",
            "--daemon",
            "--max-cycles",
            "1",
        ]
    )
    assert code == 0
    payload = _load_trailing_json(stdout)
    assert payload["mode"] == "daemon"
    assert payload["cycles_completed"] == 1
    assert payload["target_running"] == 1
    assert payload["stopped_reason"] == "max_cycles_reached"


def test_cli_daemon_reports_controller_mode_in_stderr(
    tmp_path, patched_cli_system, monkeypatch
):
    del patched_cli_system
    db_url = f"sqlite:///{tmp_path / 'cli-launch-controller.sqlite'}"
    dataset_root = tmp_path / "datasets"
    _set_runtime_env(monkeypatch, database_url=db_url, dataset_root=dataset_root)
    script_path = _write_script_file(
        tmp_path,
        build_baseline_train_script(),
        "daemon-track.py",
    )

    _, stdout, stderr = _run_cli(["create-track", script_path])
    assert stdout == ""
    track_id = _track_id_from_stderr(stderr)
    code, stdout, stderr = _run_cli(
        [
            "--launcher",
            "inline",
            "launch",
            track_id,
            "1",
            "--daemon",
            "--max-cycles",
            "1",
        ]
    )

    assert code == 0
    payload = _load_trailing_json(stdout)
    assert payload["mode"] == "daemon"
    assert "Starting controller" in stderr
    assert "Running launch pass" not in stderr


def test_make_system_with_modal_launcher_uses_modal_proxy(monkeypatch, tmp_path):
    captured = {}

    def fake_create_modal_launcher(**kwargs):
        captured.update(kwargs)
        return object()

    def fake_build_system(**kwargs):
        orchestrator = SimpleNamespace(launcher=None)
        return SimpleNamespace(launcher=None, orchestrator=orchestrator)

    monkeypatch.setattr(cli_module, "create_modal_launcher", fake_create_modal_launcher)
    monkeypatch.setattr(cli_module, "build_system", fake_build_system)
    _set_runtime_env(
        monkeypatch,
        database_url="postgresql://example/db",
        dataset_root=tmp_path / "datasets",
        modal_app_name="sigmaevolve-runner",
    )
    args = cli_module.build_parser().parse_args(
        [
            "--launcher",
            "modal",
            "list-trials",
            "track_1",
        ]
    )
    args = cli_module._apply_runtime_config(args)
    system = cli_module._make_system(args)
    assert captured["database_url"] == "postgresql://example/db"
    assert system.launcher is not None


def test_runtime_config_prefers_sigmaevolve_env_names(monkeypatch):
    _set_runtime_env(
        monkeypatch,
        database_url="postgresql://scoped/db",
        openrouter_api_key="scoped-key",
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://fallback/db")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fallback-key")

    runtime_config = resolve_runtime_config()

    assert runtime_config.database_url == "postgresql://scoped/db"
    assert runtime_config.openrouter_api_key == "scoped-key"


def test_cli_rejects_removed_runtime_config_flags():
    parser = cli_module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--database-url", "postgresql://example/db", "list-trials", "track_1"]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(["--modal-app-name", "custom-app", "modal-deploy"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--launcher", "recording", "list-trials", "track_1"])

    with pytest.raises(SystemExit):
        parser.parse_args(["sample-context", "track_1"])

    with pytest.raises(SystemExit):
        parser.parse_args(["prepare-dataset", "mnist:v1"])

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "rescore",
                "--all-tracks",
                "--scorer-json",
                '{"primary_metric":"accuracy"}',
            ]
        )


def test_cli_help_documents_env_runtime_config(capsys):
    parser = cli_module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])

    captured = capsys.readouterr()
    assert "SIGMAEVOLVE_DATABASE_URL or DATABASE_URL" in captured.out
    assert "SIGMAEVOLVE_MODAL_APP_NAME" in captured.out
    assert "--database-url" not in captured.out


def test_normalize_database_url_accepts_neon_postgres_scheme():
    assert normalize_database_url("postgresql://example/db").startswith(
        "postgresql+psycopg://"
    )
    assert normalize_database_url("postgres://example/db").startswith(
        "postgresql+psycopg://"
    )
