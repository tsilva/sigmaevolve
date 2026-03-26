from __future__ import annotations

from types import SimpleNamespace

import pytest

from sigmaevolve import modal as modal_module


def test_deploy_modal_app_allows_default_names(monkeypatch):
    deployed: list[tuple[str, str | None]] = []

    class FakeApp:
        def deploy(self, *, name: str, environment_name: str | None = None) -> None:
            deployed.append((name, environment_name))

    class _OutputContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_modal = SimpleNamespace(enable_output=lambda: _OutputContext())
    monkeypatch.setattr(modal_module, "require_modal", lambda: fake_modal)
    monkeypatch.setattr(modal_module, "app", FakeApp())

    payload = modal_module.deploy_modal_app()

    assert deployed == [(modal_module.DEFAULT_MODAL_APP_NAME, None)]
    assert payload["app_name"] == modal_module.DEFAULT_MODAL_APP_NAME


def test_deploy_modal_app_rejects_custom_names():
    with pytest.raises(ValueError, match="Custom Modal app/function/volume names"):
        modal_module.deploy_modal_app(app_name="custom-app")
