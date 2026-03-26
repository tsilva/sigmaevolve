from __future__ import annotations

import pytest

from sigmaevolve.wandb_support import collect_wandb_env, resolve_wandb_settings


def test_collect_wandb_env_uses_standard_wandb_keys(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_BASE_URL", "https://wandb.example")

    assert collect_wandb_env() == {
        "WANDB_API_KEY": "key",
        "WANDB_PROJECT": "proj",
        "WANDB_ENTITY": "team",
        "WANDB_BASE_URL": "https://wandb.example",
    }


def test_resolve_wandb_settings_reads_standard_wandb_keys(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_BASE_URL", "https://wandb.example")

    settings = resolve_wandb_settings()

    assert settings.api_key == "key"
    assert settings.project == "proj"
    assert settings.entity == "team"
    assert settings.base_url == "https://wandb.example"


def test_resolve_wandb_settings_requires_wandb_api_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="WANDB_API_KEY is required"):
        resolve_wandb_settings()
