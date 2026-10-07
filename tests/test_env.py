from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from sigmaevolve import cli
from sigmaevolve import env as runtime_env


def secret_row(**changes):
    return {
        "key": "DATABASE_URL",
        "value": "managed-database",
        "workspace": runtime_env.INFISICAL_PROJECT,
        "secretPath": "/",
        "type": "shared",
        **changes,
    }


@pytest.fixture
def managed_root(tmp_path):
    settings = {
        "workspaceId": runtime_env.INFISICAL_PROJECT,
        "domain": runtime_env.INFISICAL_DOMAIN,
        "defaultEnvironment": "dev",
    }
    (tmp_path / ".infisical.json").write_text(json.dumps(settings))
    return tmp_path


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in runtime_env.PRIVATE_ENV_KEYS:
        monkeypatch.setenv(key, "obsolete-shell-secret")
    for key in list(os.environ):
        if key.startswith("INFISICAL_") or key == "BWS_ACCESS_TOKEN":
            monkeypatch.delenv(key)
    monkeypatch.setenv("INFISICAL_TOKEN", "manager-secret")
    monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "bootstrap-secret")


@pytest.mark.parametrize("override", [False, True])
def test_userfile_never_restores_private_credentials(tmp_path, monkeypatch, override):
    for key in runtime_env.PRIVATE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("SIGMAEVOLVE_DATASET_ROOT", raising=False)
    settings = tmp_path / ".env"
    settings.write_text(
        "\n".join(f"{key}=old-userfile" for key in runtime_env.PRIVATE_ENV_KEYS)
        + "\nINFISICAL_OTHER_AUTH=old-manager\nSIGMAEVOLVE_DATASET_ROOT=/datasets\n"
    )
    runtime_env.load_env_file(settings, override=override)
    for key in runtime_env.PRIVATE_ENV_KEYS:
        assert key not in os.environ
    assert "INFISICAL_OTHER_AUTH" not in os.environ
    assert os.environ["SIGMAEVOLVE_DATASET_ROOT"] == "/datasets"


def test_export_is_pinned_and_only_experiment_credentials_are_injected(
    managed_root, monkeypatch
):
    requests = []
    rows = [
        secret_row(),
        secret_row(key="SENTRY_AUTH_TOKEN", value="dashboard-only"),
        secret_row(key="PATH", value="untrusted-path"),
        secret_row(key="INFISICAL_TOKEN", value="untrusted-auth"),
    ]

    def export(command, **kwargs):
        requests.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")

    original_path = os.environ.get("PATH")
    monkeypatch.setattr(runtime_env.subprocess, "run", export)
    runtime_env.load_managed_secrets(managed_root)

    command, options = requests[0]
    assert command[command.index("--projectId") + 1] == runtime_env.INFISICAL_PROJECT
    assert command[command.index("--env") + 1] == "dev"
    assert command[command.index("--path") + 1] == "/"
    assert command[command.index("--domain") + 1] == runtime_env.INFISICAL_DOMAIN
    for flag in (
        "--expand=false",
        "--include-imports=false",
        "--secret-overriding=false",
    ):
        assert flag in command
    assert options["capture_output"] is True
    assert options["timeout"] == 120
    assert options["cwd"] == managed_root
    assert not any(
        key in runtime_env.PRIVATE_ENV_KEYS or runtime_env._is_manager_setting(key)
        for key in options["env"]
    )
    assert os.environ["DATABASE_URL"] == "managed-database"
    for key in runtime_env.PRIVATE_ENV_KEYS - {"DATABASE_URL"}:
        assert os.environ[key] == ""
    assert os.environ.get("PATH") == original_path
    assert "INFISICAL_TOKEN" not in os.environ
    assert "INFISICAL_CLIENT_SECRET" not in os.environ
    assert "BWS_ACCESS_TOKEN" not in os.environ


@pytest.mark.parametrize(
    "payload",
    [
        "raw-provider-secret",
        "{}",
        json.dumps([secret_row(workspace="foreign-project")]),
        json.dumps([secret_row(secretPath="/other")]),
        json.dumps([secret_row(type="personal")]),
        json.dumps([secret_row(), secret_row()]),
        json.dumps([secret_row(value="invalid\0secret")]),
        json.dumps([secret_row(value=123)]),
    ],
)
def test_malformed_or_foreign_responses_fail_closed(managed_root, monkeypatch, payload):
    result = SimpleNamespace(returncode=0, stdout=payload, stderr="provider-secret")
    monkeypatch.setattr(runtime_env.subprocess, "run", lambda *a, **k: result)
    with pytest.raises(runtime_env.ManagedSecretsError) as failure:
        runtime_env.load_managed_secrets(managed_root)
    assert "provider-secret" not in str(failure.value)
    for key in runtime_env.PRIVATE_ENV_KEYS:
        assert os.environ[key] == ""


@pytest.mark.parametrize("field", ["workspaceId", "domain", "defaultEnvironment"])
def test_changed_configuration_fails_before_fetch(managed_root, monkeypatch, field):
    path = managed_root / ".infisical.json"
    settings = json.loads(path.read_text())
    settings[field] = "foreign-source"
    path.write_text(json.dumps(settings))

    def forbidden_fetch(*args, **kwargs):
        pytest.fail("A changed source must fail before making a request")

    monkeypatch.setattr(runtime_env.subprocess, "run", forbidden_fetch)
    with pytest.raises(runtime_env.ManagedSecretsError):
        runtime_env.load_managed_secrets(managed_root)


def test_failed_fetch_does_not_invoke_cli_handler(managed_root, monkeypatch, capsys):
    calls = []
    args = SimpleNamespace(func=lambda arguments: calls.append(arguments))
    parser = SimpleNamespace(parse_args=lambda argv: args)
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.setattr(
        cli,
        "load_managed_secrets",
        lambda: runtime_env.load_managed_secrets(managed_root),
    )

    def failed_export(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            "private-command", 120, output="provider-secret"
        )

    monkeypatch.setattr(runtime_env.subprocess, "run", failed_export)
    assert cli.main(["list-trials", "example"]) == 1
    assert not calls
    assert "provider-secret" not in capsys.readouterr().err
    assert "INFISICAL_TOKEN" not in os.environ


@pytest.mark.parametrize("arguments", [["--help"], ["launch", "--help"]])
def test_help_does_not_load_userfile_or_fetch_secrets(monkeypatch, arguments):
    def forbidden_load():
        pytest.fail("Help must work offline without loading credentials")

    monkeypatch.setattr(cli, "load_env_file", forbidden_load)
    monkeypatch.setattr(cli, "load_managed_secrets", forbidden_load)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(arguments)
    assert exit_info.value.code == 0
