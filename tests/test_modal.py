from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
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


def test_modal_image_uses_project_dependencies_and_copies_source(monkeypatch):
    calls: dict[str, object] = {}

    class FakeImageBuilder:
        def pip_install_from_pyproject(
            self,
            path: str,
            *,
            optional_dependencies: list[str] | None = None,
        ):
            calls["pyproject"] = (path, optional_dependencies)
            return self

        def add_local_python_source(self, package: str, *, copy: bool = False):
            calls["local_source"] = (package, copy)
            return self

    class FakeImageAPI:
        @staticmethod
        def debian_slim(*, python_version: str):
            calls["python_version"] = python_version
            return FakeImageBuilder()

    class FakeVolumeAPI:
        @staticmethod
        def from_name(
            name: str,
            create_if_missing: bool = False,
            environment_name: str | None = None,
        ):
            del create_if_missing, environment_name
            return SimpleNamespace(name=name)

    class FakeApp:
        def __init__(self, name: str):
            self.name = name

        def cls(self, **kwargs):
            calls["app_cls_kwargs"] = kwargs

            def decorator(cls):
                return cls

            return decorator

    def fake_method():
        def decorator(fn):
            return fn

        return decorator

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImageAPI,
        Volume=FakeVolumeAPI,
        method=fake_method,
    )
    module_path = Path(modal_module.__file__)
    spec = importlib.util.spec_from_file_location(
        "sigmaevolve_modal_image_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None

    test_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, spec.name, test_module)
    spec.loader.exec_module(test_module)

    assert calls["python_version"] == "3.11"
    assert calls["pyproject"] == ("pyproject.toml", ["datasets"])
    assert calls["local_source"] == ("sigmaevolve", True)
