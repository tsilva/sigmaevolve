from __future__ import annotations

from sigmaevolve.modal_support import create_modal_launcher


def test_modal_launcher_spawns_named_function(monkeypatch):
    captured = {}

    class FakeFunctionHandle:
        def spawn(self, **kwargs):
            captured["spawn"] = kwargs

    class FakeFunction:
        @staticmethod
        def from_name(app_name, name, environment_name=None):
            captured["lookup"] = {
                "app_name": app_name,
                "name": name,
                "environment_name": environment_name,
            }
            return FakeFunctionHandle()

    class FakeModal:
        Function = FakeFunction

    monkeypatch.setattr("sigmaevolve.modal_support.require_modal", lambda: FakeModal)

    launcher = create_modal_launcher(
        app_name="sigmaevolve-runner",
        function_name="run_trial",
        database_url="postgresql://example/db",
        dataset_root="/mnt/datasets",
        environment_name="main",
    )
    launcher.launch_trial("trial_1", "dispatch_1")

    assert captured["lookup"]["app_name"] == "sigmaevolve-runner"
    assert captured["lookup"]["name"] == "run_trial"
    assert captured["spawn"]["trial_id"] == "trial_1"
    assert captured["spawn"]["dispatch_token"] == "dispatch_1"
    assert captured["spawn"]["database_url"] == "postgresql://example/db"
    assert captured["spawn"]["dataset_root"] == "/mnt/datasets"
