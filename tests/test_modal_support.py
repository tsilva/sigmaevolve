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
        wandb_env={"WANDB_API_KEY": "wandb-test-key", "WANDB_PROJECT": "sigmaevolve"},
    )
    metadata = launcher.launch_trial("trial_1", "dispatch_1")

    assert captured["lookup"]["app_name"] == "sigmaevolve-runner"
    assert captured["lookup"]["name"] == "TrialRunner"
    assert captured["spawn"]["trial_id"] == "trial_1"
    assert captured["spawn"]["dispatch_token"] == "dispatch_1"
    assert captured["spawn"]["database_url"] == "postgresql://example/db"
    assert captured["spawn"]["dataset_root"] == "/mnt/datasets"
    assert captured["spawn"]["wandb_env"] == {"WANDB_API_KEY": "wandb-test-key", "WANDB_PROJECT": "sigmaevolve"}
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
        wandb_env={"WANDB_API_KEY": "wandb-test-key"},
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
        wandb_env={"WANDB_API_KEY": "wandb-test-key"},
    )

    with pytest.raises(RuntimeError, match="T4: .*L4: .*A10:"):
        launcher.launch_trial(
            "trial_1",
            "dispatch_1",
            launch_policy={"modal_gpu_preferences": ["T4", "L4", "A10"]},
        )


def test_modal_launcher_cancels_function_call_by_run_id(monkeypatch):
    captured = {}

    class FakeFunctionCallHandle:
        def __init__(self, run_id):
            self.run_id = run_id

        def cancel(self):
            captured["cancelled_run_id"] = self.run_id

    class FakeFunctionCall:
        @staticmethod
        def from_id(run_id):
            captured["lookup_run_id"] = run_id
            return FakeFunctionCallHandle(run_id)

    class FakeModal:
        FunctionCall = FakeFunctionCall

    monkeypatch.setattr("sigmaevolve.modal_support.require_modal", lambda: FakeModal)

    launcher = create_modal_launcher(
        app_name="sigmaevolve-runner",
        function_name="run_trial",
        database_url="postgresql://example/db",
        dataset_root="/mnt/datasets",
        environment_name="main",
        wandb_env={"WANDB_API_KEY": "wandb-test-key"},
    )

    launcher.cancel_run({"kind": "modal", "run_id": "fc-789"})

    assert captured == {
        "lookup_run_id": "fc-789",
        "cancelled_run_id": "fc-789",
    }


def test_modal_launcher_falls_back_to_legacy_function_when_class_lookup_is_missing(monkeypatch):
    captured = {}

    class FakeFunctionCall:
        object_id = "fc-legacy"

        def get_dashboard_url(self):
            return "https://modal.com/apps/test/runs/fc-legacy"

    class FakeFunctionHandle:
        def spawn(self, **kwargs):
            captured["spawn"] = kwargs
            return FakeFunctionCall()

    class FakeCls:
        @staticmethod
        def from_name(app_name, name, environment_name=None):
            captured["class_lookup"] = {
                "app_name": app_name,
                "name": name,
                "environment_name": environment_name,
            }
            raise RuntimeError(
                "Lookup failed for Cls 'TrialRunner' from the 'sigmaevolve-runner' app: "
                "Class 'TrialRunner' not found in app 'sigmaevolve-runner'."
            )

    class FakeFunction:
        @staticmethod
        def from_name(app_name, name, environment_name=None):
            captured["function_lookup"] = {
                "app_name": app_name,
                "name": name,
                "environment_name": environment_name,
            }
            return FakeFunctionHandle()

    class FakeModal:
        Cls = FakeCls
        Function = FakeFunction

    monkeypatch.setattr("sigmaevolve.modal_support.require_modal", lambda: FakeModal)

    launcher = create_modal_launcher(
        app_name="sigmaevolve-runner",
        function_name="run_trial",
        database_url="postgresql://example/db",
        dataset_root="/mnt/datasets",
        environment_name="main",
        wandb_env={"WANDB_API_KEY": "wandb-test-key"},
    )
    metadata = launcher.launch_trial(
        "trial_1",
        "dispatch_1",
        launch_policy={"modal_gpu_preferences": ["T4", "L4", "A10"]},
    )

    assert captured["class_lookup"] == {
        "app_name": "sigmaevolve-runner",
        "name": "TrialRunner",
        "environment_name": "main",
    }
    assert captured["function_lookup"] == {
        "app_name": "sigmaevolve-runner",
        "name": "run_trial",
        "environment_name": "main",
    }
    assert captured["spawn"] == {
        "trial_id": "trial_1",
        "dispatch_token": "dispatch_1",
        "database_url": "postgresql://example/db",
        "dataset_root": "/mnt/datasets",
        "wandb_env": {"WANDB_API_KEY": "wandb-test-key"},
    }
    assert metadata == {
        "kind": "modal",
        "run_id": "fc-legacy",
        "run_url": "https://modal.com/apps/test/runs/fc-legacy",
        "gpu_attempts": ["T4"],
    }


def test_modal_launcher_does_not_fallback_to_legacy_function_for_non_lookup_errors(monkeypatch):
    captured = {"function_lookups": 0}

    class FakeMethodHandle:
        def __init__(self, gpu=None):
            self.gpu = gpu

        def spawn(self, **kwargs):
            del kwargs
            raise RuntimeError(f"{self.gpu} capacity unavailable")

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

    class FakeFunction:
        @staticmethod
        def from_name(app_name, name, environment_name=None):
            del app_name, name, environment_name
            captured["function_lookups"] += 1
            raise AssertionError("legacy function lookup should not be used")

    class FakeModal:
        Cls = FakeCls
        Function = FakeFunction

    monkeypatch.setattr("sigmaevolve.modal_support.require_modal", lambda: FakeModal)

    launcher = create_modal_launcher(
        app_name="sigmaevolve-runner",
        function_name="run_trial",
        database_url="postgresql://example/db",
        dataset_root="/mnt/datasets",
        environment_name="main",
        wandb_env={"WANDB_API_KEY": "wandb-test-key"},
    )

    with pytest.raises(RuntimeError, match="T4: .*L4: .*A10:"):
        launcher.launch_trial(
            "trial_1",
            "dispatch_1",
            launch_policy={"modal_gpu_preferences": ["T4", "L4", "A10"]},
        )

    assert captured["function_lookups"] == 0
