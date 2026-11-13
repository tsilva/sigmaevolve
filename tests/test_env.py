from __future__ import annotations

import os
import subprocess

import pytest

from sigmaevolve import cli
from sigmaevolve import env as runtime_env


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


@pytest.mark.parametrize("arguments", [["--help"], ["launch", "--help"]])
def test_help_does_not_load_userfile_or_fetch_secrets(monkeypatch, arguments):
    def forbidden_load():
        pytest.fail("Help must work offline without loading credentials")

    monkeypatch.setattr(cli, "load_env_file", forbidden_load)
    monkeypatch.setattr(cli, "prepare_runtime_environment", forbidden_load)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(arguments)
    assert exit_info.value.code == 0


def test_supplied_process_credentials_survive_without_a_provider(monkeypatch):
    expected = {
        key: "synthetic-explicit-value" for key in runtime_env.EXPERIMENT_SECRET_KEYS
    }
    for key, value in expected.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("INFISICAL_TOKEN", "synthetic-manager")
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: pytest.fail("No secret-manager call")
    )
    runtime_env.prepare_runtime_environment()
    assert {key: os.environ.get(key) for key in expected} == expected
    assert "INFISICAL_TOKEN" not in os.environ
