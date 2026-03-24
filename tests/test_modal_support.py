from __future__ import annotations

import pytest

from sigmaevolve.modal_support import create_modal_launcher


def test_modal_launcher_spawns_named_method_without_gpu_override(monkeypatch):
    captured = {}

    class FakeFunctionCall:
        object_id = "fc-123"

        def get_dashboard_url(self):
            return "https://modal.com/apps/test/runs/fc-123"

    class FakeMethodHandle:
        def __init__(self, gpu=None):
            self.gpu = gpu

        def spawn(self, **kwargs):
            captured["spawn"] = {"gpu": self.gpu, **kwargs}
            return FakeFunctionCall()

    class FakeObjectHandle:
        def __init__(self, gpu=None):
            self.run_trial = FakeMethodHandle(gpu=gpu)

    class FakeClassHandle:
        def __init__(self, gpu=None):
            self.gpu = gpu

        def with_options(self, *, gpu=None, **_kwargs):
            captured.setdefault("with_options", []).append(gpu)
            return FakeClassHandle(gpu=gpu)

        def __call__(self):
            captured["instantiated_gpu"] = self.gpu
            return FakeObjectHandle(gpu=self.gpu)

    class FakeCls:
        @staticmethod
        def from_name(app_name, name, environment_name=None):
            captured["lookup"] = {
                "app_name": app_name,
                "name": name,
                "environment_name": environment_name,
            }
            return FakeClassHandle()

    class FakeModal:
        Cls = FakeCls

    monkeypatch.setattr("sigmaevolve.modal_support.require_modal", lambda: FakeModal)

    launcher = create_modal_launcher(
        app_name="sigmaevolve-runner",
        function_name="run_trial",
        database_url="postgresql://example/db",
        dataset_root="/mnt/datasets",
        environment_name="main",
    )
    metadata = launcher.launch_trial("trial_1", "dispatch_1")

    assert captured["lookup"]["app_name"] == "sigmaevolve-runner"
    assert captured["lookup"]["name"] == "TrialRunner"
    assert captured["spawn"]["trial_id"] == "trial_1"
    assert captured["spawn"]["dispatch_token"] == "dispatch_1"
    assert captured["spawn"]["database_url"] == "postgresql://example/db"
    assert captured["spawn"]["dataset_root"] == "/mnt/datasets"
    assert captured["spawn"]["gpu"] is None
    assert "with_options" not in captured
    assert metadata == {
        "kind": "modal",
        "run_id": "fc-123",
        "run_url": "https://modal.com/apps/test/runs/fc-123",
        "gpu_attempts": [],
    }


def test_modal_launcher_retries_gpu_preferences_in_order(monkeypatch):
    captured = {"spawn_attempts": []}

    class FakeFunctionCall:
        object_id = "fc-456"

        def get_dashboard_url(self):
            return "https://modal.com/apps/test/runs/fc-456"

    class FakeMethodHandle:
        def __init__(self, gpu=None):
            self.gpu = gpu

        def spawn(self, **kwargs):
            captured["spawn_attempts"].append({"gpu": self.gpu, **kwargs})
            if self.gpu == "T4":
                raise RuntimeError("T4 capacity unavailable")
            return FakeFunctionCall()

    class FakeObjectHandle:
        def __init__(self, gpu=None):
            self.run_trial = FakeMethodHandle(gpu=gpu)

    class FakeClassHandle:
        def __init__(self, gpu=None):
            self.gpu = gpu

        def with_options(self, *, gpu=None, **_kwargs):
            return FakeClassHandle(gpu=gpu)

        def __call__(self):
            return FakeObjectHandle(gpu=self.gpu)

    class FakeCls:
        @staticmethod
        def from_name(app_name, name, environment_name=None):
            captured["lookup"] = {
                "app_name": app_name,
                "name": name,
                "environment_name": environment_name,
            }
            return FakeClassHandle()

    class FakeModal:
        Cls = FakeCls

    monkeypatch.setattr("sigmaevolve.modal_support.require_modal", lambda: FakeModal)

    launcher = create_modal_launcher(
        app_name="sigmaevolve-runner",
        function_name="run_trial",
        database_url="postgresql://example/db",
        dataset_root="/mnt/datasets",
        environment_name="main",
    )
    metadata = launcher.launch_trial(
        "trial_1",
        "dispatch_1",
        launch_policy={"modal_gpu_preferences": ["T4", "L4", "A10"]},
    )

    assert [attempt["gpu"] for attempt in captured["spawn_attempts"]] == ["T4", "L4"]
    assert metadata == {
        "kind": "modal",
        "run_id": "fc-456",
        "run_url": "https://modal.com/apps/test/runs/fc-456",
        "gpu_selected": "L4",
        "gpu_attempts": ["T4", "L4"],
    }


def test_modal_launcher_surfaces_combined_gpu_failures(monkeypatch):
    class FakeMethodHandle:
        def __init__(self, gpu=None):
            self.gpu = gpu

        def spawn(self, **kwargs):
            del kwargs
            raise RuntimeError(f"{self.gpu} unavailable")

    class FakeObjectHandle:
        def __init__(self, gpu=None):
            self.run_trial = FakeMethodHandle(gpu=gpu)

    class FakeClassHandle:
        def __init__(self, gpu=None):
            self.gpu = gpu

        def with_options(self, *, gpu=None, **_kwargs):
            return FakeClassHandle(gpu=gpu)

        def __call__(self):
            return FakeObjectHandle(gpu=self.gpu)

    class FakeCls:
        @staticmethod
        def from_name(app_name, name, environment_name=None):
            del app_name, name, environment_name
            return FakeClassHandle()

    class FakeModal:
        Cls = FakeCls

    monkeypatch.setattr("sigmaevolve.modal_support.require_modal", lambda: FakeModal)

    launcher = create_modal_launcher(
        app_name="sigmaevolve-runner",
        function_name="run_trial",
        database_url="postgresql://example/db",
        dataset_root="/mnt/datasets",
        environment_name="main",
    )

    with pytest.raises(RuntimeError, match="T4: .*L4: .*A10:"):
        launcher.launch_trial(
            "trial_1",
            "dispatch_1",
            launch_policy={"modal_gpu_preferences": ["T4", "L4", "A10"]},
        )
