import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from sigmaevolve import execution as strategy_runtime
from sigmaevolve.core import CANDIDATE_KIND_STRATEGY_V1
from sigmaevolve.execution import (
    RunnerService,
    collect_wandb_env,
    resolve_wandb_settings,
)
from sigmaevolve.generation import (
    build_candidate_train_script,
    build_data_block,
    build_model_block,
    build_optimization_block,
)
from sigmaevolve.modal import create_modal_launcher
from sigmaevolve.orchestration import (
    EvolutionSystem,
    InlineRunnerLauncher,
)
from sigmaevolve.storage import classify_error_type
from tests.support import make_generation_trace, make_llm_provenance

SUCCESS_BLOCK = build_model_block(
    """
def forward(self, x):
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    return torch.stack((-scores, scores), dim=1)
"""
)

TIMEOUT_BLOCK = build_model_block(
    """
def forward(self, x):
    time.sleep(2.0)
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
    imports="import time\nimport torch",
)

SALVAGED_TIMEOUT_BLOCK = build_model_block(
    """
def __init__(self):
    super().__init__()
    self.train_calls = 0

def forward(self, x):
    if self.training:
        self.train_calls += 1
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    if self.training and self.train_calls >= 3:
        time.sleep(2.0)
    if self.train_calls == 1:
        return torch.zeros((x.shape[0], 2), dtype=torch.float32)
    return torch.stack((-scores, scores), dim=1)
""",
    imports="import time\nimport torch",
)

TIEBREAKER_BLOCK = build_model_block(
    """
def __init__(self):
    super().__init__()
    self.train_calls = 0

def forward(self, x):
    if self.training:
        self.train_calls += 1
        if self.train_calls == 1:
            time.sleep(0.05)
        elif self.train_calls == 2:
            time.sleep(0.1)
        else:
            time.sleep(2.0)
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    return torch.stack((-scores, scores), dim=1)
""",
    imports="import time\nimport torch",
)

EARLY_STOP_BLOCK = build_model_block(
    """
def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
    imports="import torch",
)

CRASH_BLOCK = build_model_block(
    """
def __init__(self):
    super().__init__()
    raise RuntimeError("boom")

def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
"""
)

LOGGING_BLOCK = build_model_block(
    """
def forward(self, x):
    print("stdout-marker", flush=True)
    print("stderr-marker", file=sys.stderr, flush=True)
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    return torch.stack((-scores, scores), dim=1)
""",
    imports="import sys\nimport torch",
)

LIVE_METRICS_BLOCK = build_model_block(
    """
def __init__(self):
    super().__init__()
    self.train_calls = 0

def forward(self, x):
    if self.training:
        self.train_calls += 1
    if self.training and self.train_calls >= 2:
        time.sleep(1.3)
    flat = x.reshape(x.shape[0], -1)
    scores = flat.sum(dim=1)
    return torch.stack((-scores, scores), dim=1)
""",
    imports="import time\nimport torch",
)

SMALL_BATCH_DATA_BLOCK = build_data_block(
    """
batch_size = 2
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=1)
""",
    imports="from torch.utils.data import DataLoader",
)

NOOP_OPTIMIZATION_BLOCK = build_optimization_block(
    """
optimizer = None
scheduler = None
"""
)

INVALID_EXPERIMENT_BLOCK = """
model = object()
optimizer = None
scheduler = None
loss_fn = 123
train_loader = None
val_loader = None
early_stopping_patience = 0
min_delta = 0.0
"""


def build_inline_system(repository, dataset_manager, hard_timeout_sec=5.0):
    runner = RunnerService(
        repository=repository,
        dataset_manager=dataset_manager,
        hard_timeout_sec=hard_timeout_sec,
    )
    return EvolutionSystem(
        repository, dataset_manager, None, InlineRunnerLauncher(runner), runner
    )


def _create_track(
    system, policy_json: dict | None = None, dataset_id: str = "mnist:v1"
):
    return system.create_track(dataset_id, policy_json or {})


def finalize_baseline(system, track_id):
    baseline = system.repository.list_trials(track_id)[0]
    system.repository.finalize_trial(
        trial_id=baseline.trial_id,
        runner_id=None,
        outcome_reason="stale",
        metrics=None,
        error_info={"reason": "test_setup"},
    )


def _run_trial(system, track_id, source):
    trial_source = (
        source
        if "# EVOLVE-BLOCK-START" in source
        else build_candidate_train_script(source)
    )
    _, created = system.repository.create_queued_trial_if_absent(
        track_id,
        trial_source,
        make_llm_provenance(
            candidate_kind=CANDIDATE_KIND_STRATEGY_V1,
            generation=make_generation_trace(trial_source),
        ),
    )
    assert created is True
    reserved = system.repository.reserve_trials(
        track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1
    )[0]
    system.launcher.launch_trial(reserved.trial_id, reserved.dispatch_token)
    return system.repository.get_trial(reserved.trial_id)


def test_heartbeat_thread_retries_after_transient_failure():
    class FlakyRepository:
        def __init__(self) -> None:
            self.calls = 0

        def heartbeat_trial(self, trial_id, runner_id, meta):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient disconnect")

    repository = FlakyRepository()
    runner = RunnerService(repository=repository, dataset_manager=object())

    stop_event, thread = runner._start_heartbeat(
        "trial-1", "runner-1", interval_sec=0.05
    )
    time.sleep(0.18)
    stop_event.set()
    thread.join(timeout=1.0)

    assert repository.calls >= 2
    assert thread.is_alive() is False


def test_successful_run_produces_metrics_and_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, SUCCESS_BLOCK)
    assert finished.outcome_reason == "succeeded"
    assert finished.metrics_json["accuracy"] >= 0.0
    assert finished.metrics_json["val_loss"] >= 0.0
    assert finished.metrics_json["eval_count"] == 2
    assert finished.metrics_json["epochs_completed"] == 2
    assert finished.metrics_json["best_eval_epoch"] == 1
    assert finished.score == finished.metrics_json["accuracy"]
    assert finished.error_json is None


def test_successful_run_creates_wandb_experiment(
    repository, dataset_manager, fake_wandb
):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, SUCCESS_BLOCK)

    runs = fake_wandb["runs"]
    assert isinstance(runs, list)
    assert len(runs) == 1
    run = runs[0]
    assert finished.outcome_reason == "succeeded"
    assert finished.provenance_json["wandb"]["project"] == "sigmaevolve"
    assert finished.provenance_json["wandb"]["run_id"] == run.id
    assert finished.provenance_json["wandb"]["run_url"] == run.url
    terminal_entries = [
        entry for entry in run.logged if entry["payload"]["trial_state"] == "terminal"
    ]
    assert terminal_entries
    terminal_payload = terminal_entries[-1]["payload"]
    assert terminal_payload["train/loss"] >= 0.0
    assert 0.0 <= terminal_payload["train/acc"] <= 1.0
    assert terminal_payload["val/loss"] == pytest.approx(
        finished.metrics_json["val_loss"]
    )
    assert terminal_payload["val/acc"] == pytest.approx(
        finished.metrics_json["accuracy"]
    )
    assert run.summary["outcome_reason"] == "succeeded"
    assert run.summary["train/loss"] >= 0.0
    assert 0.0 <= run.summary["train/acc"] <= 1.0
    assert run.summary["val/loss"] == pytest.approx(finished.metrics_json["val_loss"])
    assert run.summary["val/acc"] == pytest.approx(finished.metrics_json["accuracy"])
    assert len(run.artifacts) == 1
    artifact_entry = run.artifacts[0]
    artifact = artifact_entry["artifact"]
    assert artifact.name == f"{finished.trial_id}-best-model"
    assert artifact.type == "model"
    assert artifact.metadata["trial_id"] == finished.trial_id
    assert artifact.metadata["track_id"] == finished.track_id
    assert artifact.metadata["dataset_id"] == "mnist:v1"
    assert artifact.metadata["accuracy"] == pytest.approx(
        finished.metrics_json["accuracy"]
    )
    assert artifact_entry["aliases"] == ["best", "latest"]
    assert artifact.files == [
        {
            "local_path": run.summary["best_model_artifact_path"],
            "name": "best_model.pt",
        }
    ]
    assert [event["kind"] for event in run.events][-2:] == ["artifact", "finish"]
    assert run.finished == {"exit_code": 0}


def test_successful_run_with_custom_data_block(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(
        system,
        track.track_id,
        build_candidate_train_script(data_block_payload=SMALL_BATCH_DATA_BLOCK),
    )
    assert finished.outcome_reason == "succeeded"
    assert finished.metrics_json["eval_count"] == 2
    assert finished.error_json is None


def test_successful_run_with_custom_optimization_block(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 2})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(
        system,
        track.track_id,
        build_candidate_train_script(
            optimization_block_payload=NOOP_OPTIMIZATION_BLOCK
        ),
    )
    assert finished.outcome_reason == "succeeded"
    assert finished.metrics_json["eval_count"] == 2
    assert finished.error_json is None


def test_timeout_with_no_completed_eval_finalizes_with_zero_score(
    repository, dataset_manager
):
    system = build_inline_system(repository, dataset_manager, hard_timeout_sec=0.3)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 3})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, TIMEOUT_BLOCK)
    assert finished.outcome_reason == "timeout"
    assert finished.score == 0.0
    assert finished.metrics_json is None


def test_timeout_with_completed_eval_keeps_best_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager, hard_timeout_sec=1.5)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 4})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, SALVAGED_TIMEOUT_BLOCK)
    assert finished.outcome_reason == "timeout"
    assert finished.score == finished.metrics_json["accuracy"]
    assert finished.metrics_json["timed_out"] is True
    assert finished.metrics_json["accuracy"] == 1.0
    assert finished.metrics_json["had_unscored_work_at_timeout"] is True
    assert finished.metrics_json["time_since_last_eval_sec"] > 0.0
    assert finished.error_json is None


def test_timeout_with_completed_eval_uploads_best_model_artifact(
    repository, dataset_manager, fake_wandb
):
    system = build_inline_system(repository, dataset_manager, hard_timeout_sec=1.5)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 4})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, SALVAGED_TIMEOUT_BLOCK)

    runs = fake_wandb["runs"]
    assert isinstance(runs, list)
    assert len(runs) == 1
    run = runs[0]
    assert finished.outcome_reason == "timeout"
    assert len(run.artifacts) == 1
    artifact = run.artifacts[0]["artifact"]
    assert artifact.metadata["accuracy"] == pytest.approx(
        finished.metrics_json["accuracy"]
    )
    assert [event["kind"] for event in run.events][-2:] == ["artifact", "finish"]


def test_equal_accuracy_uses_lower_time_to_best_eval_as_tiebreaker(
    repository, dataset_manager
):
    system = build_inline_system(repository, dataset_manager, hard_timeout_sec=1.5)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 4})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, TIEBREAKER_BLOCK)
    assert finished.outcome_reason == "timeout"
    assert finished.metrics_json["accuracy"] == 1.0
    assert finished.metrics_json["time_to_best_eval_sec"] == pytest.approx(
        0.05, abs=0.08
    )
    assert finished.metrics_json["eval_count"] == 2


def test_run_stops_early_when_validation_accuracy_plateaus(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 5})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, EARLY_STOP_BLOCK)
    assert finished.outcome_reason == "succeeded"
    assert finished.metrics_json["eval_count"] == 3
    assert finished.metrics_json["last_phase"] == "finished"


def test_crash_finalizes_with_zero_score(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system)
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, CRASH_BLOCK)
    assert finished.status == "error"
    assert finished.outcome_reason == "crashed"
    assert finished.score == 0.0


def test_invalid_experiment_contract_finalizes_as_crashed(repository, dataset_manager):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system)
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, INVALID_EXPERIMENT_BLOCK)
    assert finished.status == "error"
    assert finished.outcome_reason == "crashed"
    assert (
        classify_error_type(finished.outcome_reason or "", finished.error_json)
        == "execution_crash"
    )
    assert finished.score == 0.0


def test_run_streams_child_output_to_parent_logs(repository, dataset_manager, capsys):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 1})
    finalize_baseline(system, track.track_id)
    finished = _run_trial(system, track.track_id, LOGGING_BLOCK)
    captured = capsys.readouterr()
    assert finished.outcome_reason == "succeeded"
    assert "stdout-marker" in captured.out
    assert "stderr-marker" in captured.err


def test_run_emits_lifecycle_logs(repository, dataset_manager, caplog):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 1})
    finalize_baseline(system, track.track_id)
    with caplog.at_level(logging.INFO, logger="sigmaevolve.execution"):
        finished = _run_trial(system, track.track_id, SUCCESS_BLOCK)
    messages = [record.getMessage() for record in caplog.records]
    assert finished.outcome_reason == "succeeded"
    assert any("Claimed trial" in message for message in messages)
    assert any("Verified dataset" in message for message in messages)
    assert any("Starting child process" in message for message in messages)
    assert any("Child process finished" in message for message in messages)
    assert any("Finalized trial" in message for message in messages)


def test_run_uses_unbuffered_python_for_child_process(
    repository, dataset_manager, monkeypatch
):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 1})
    finalize_baseline(system, track.track_id)
    reserved_source = build_candidate_train_script(SUCCESS_BLOCK)
    _, created = system.repository.create_queued_trial_if_absent(
        track.track_id,
        reserved_source,
        make_llm_provenance(
            candidate_kind=CANDIDATE_KIND_STRATEGY_V1,
            generation=make_generation_trace(reserved_source),
        ),
    )
    assert created is True
    reserved = system.repository.reserve_trials(
        track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1
    )[0]

    seen: dict[str, object] = {}

    def fake_run_streamed_subprocess(command, timeout):
        seen["command"] = command
        seen["timeout"] = timeout
        train_script_path = Path(command[2])
        runtime_dir = train_script_path.parent / "sigmaevolve"
        seen["runtime_init_exists"] = (runtime_dir / "__init__.py").exists()
        seen["runtime_init_source"] = (runtime_dir / "__init__.py").read_text()
        seen["runtime_module_exists"] = (
            runtime_dir / "train_script_runtime.py"
        ).exists()
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False},
        )()

    monkeypatch.setattr(
        "sigmaevolve.execution._run_streamed_subprocess", fake_run_streamed_subprocess
    )

    system.runner_service.run_reserved_trial(
        reserved.trial_id, reserved.dispatch_token, "runner-unbuffered"
    )

    command = seen["command"]
    assert isinstance(command, list)
    assert command[1] == "-u"
    assert str(command[2]).endswith("train.py")
    assert seen["runtime_init_exists"] is True
    assert seen["runtime_init_source"] == ""
    assert seen["runtime_module_exists"] is True


def test_active_run_persists_live_metrics_before_finalization(
    repository, dataset_manager, fake_wandb
):
    system = build_inline_system(repository, dataset_manager)
    system.prepare_dataset("mnist:v1")
    track = _create_track(system, {"epochs": 3})
    finalize_baseline(system, track.track_id)
    live_metrics_source = build_candidate_train_script(LIVE_METRICS_BLOCK)
    _, created = system.repository.create_queued_trial_if_absent(
        track.track_id,
        live_metrics_source,
        make_llm_provenance(
            candidate_kind=CANDIDATE_KIND_STRATEGY_V1,
            generation=make_generation_trace(live_metrics_source),
        ),
    )
    assert created is True

    reserved = system.repository.reserve_trials(
        track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1
    )[0]
    runner_id = "runner-live"
    worker = threading.Thread(
        target=system.runner_service.run_reserved_trial,
        args=(reserved.trial_id, reserved.dispatch_token, runner_id),
        daemon=True,
    )
    worker.start()

    active_snapshot = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        current = system.repository.get_trial(reserved.trial_id)
        if (
            current is not None
            and current.status == "active"
            and current.metrics_json is not None
        ):
            active_snapshot = current
            break
        time.sleep(0.1)

    worker.join(timeout=10.0)
    assert worker.is_alive() is False
    assert active_snapshot is not None
    assert active_snapshot.finished_at is None
    assert active_snapshot.metrics_json["accuracy"] == 1.0
    assert active_snapshot.metrics_json["val_loss"] >= 0.0
    assert active_snapshot.metrics_json["eval_count"] >= 1
    assert active_snapshot.metrics_json["best_eval_epoch"] == 1
    assert active_snapshot.metrics_json["last_phase"] in {"train", "eval", "finished"}
    assert "timed_out" not in active_snapshot.metrics_json

    finished = system.repository.get_trial(reserved.trial_id)
    assert finished is not None
    assert finished.status == "finished"
    assert finished.metrics_json["timed_out"] is False
    assert finished.metrics_json["accuracy"] == 1.0
    assert finished.metrics_json["epochs_completed"] == 3
    assert finished.metrics_json["best_eval_epoch"] == 1
    assert finished.metrics_json != active_snapshot.metrics_json

    runs = fake_wandb["runs"]
    assert isinstance(runs, list)
    assert len(runs) == 1
    run = runs[0]
    active_entries = [
        entry for entry in run.logged if entry["payload"]["trial_state"] == "active"
    ]
    assert active_entries
    active_payload = active_entries[-1]["payload"]
    assert "train/loss" in active_payload
    assert "train/acc" in active_payload
    assert "val/loss" in active_payload
    assert "val/acc" in active_payload
    assert run.finished == {"exit_code": 0}


def test_collect_active_metrics_payload_uses_eval_artifacts(
    repository, dataset_manager
):
    runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
    manifest = dataset_manager.prepare("mnist:v1")

    eval_dir = dataset_manager.dataset_root / "active-metrics-evals"
    eval_dir.mkdir(parents=True)
    progress_path = eval_dir / "progress.json"

    labels = np.load(manifest.validation_labels_path)
    np.savez(
        eval_dir / "eval-001.npz",
        predictions=labels,
        eval_index=1,
        elapsed_time_sec=0.25,
        epoch=1,
        train_loss=0.1,
        train_acc=1.0,
        val_loss=0.2,
        val_acc=1.0,
    )
    progress_path.write_text(
        json.dumps(
            {
                "phase": "eval",
                "eval_index": 1,
                "last_completed_eval_sec": 0.25,
                "eval_count": 1,
            }
        )
    )

    metrics = runner._collect_active_metrics_payload(
        eval_dir=eval_dir,
        progress_path=progress_path,
        labels_path=manifest.validation_labels_path,
        started_at=time.monotonic() - 0.3,
    )

    assert metrics is not None
    assert metrics["accuracy"] == 1.0
    assert metrics["best_accuracy"] == 1.0
    assert metrics["best_eval_index"] == 1
    assert metrics["best_eval_epoch"] == 1
    assert metrics["eval_count"] == 1
    assert metrics["last_phase"] == "eval"
    assert "timed_out" not in metrics


def test_seed_everything_returns_cpu_when_cuda_unavailable(monkeypatch):
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def manual_seed_all(seed):
                raise AssertionError(f"manual_seed_all should not be called: {seed}")

        @staticmethod
        def manual_seed(seed):
            return None

    monkeypatch.setitem(strategy_runtime.sys.modules, "torch", FakeTorch)

    assert strategy_runtime._seed_everything(1234) == "cpu"


def test_seed_everything_returns_cuda_when_available(monkeypatch):
    state = {"manual_seed": [], "manual_seed_all": []}

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def manual_seed_all(seed):
                state["manual_seed_all"].append(seed)

        @staticmethod
        def manual_seed(seed):
            state["manual_seed"].append(seed)

    monkeypatch.setitem(strategy_runtime.sys.modules, "torch", FakeTorch)

    assert strategy_runtime._seed_everything(7) == "cuda"
    assert state == {"manual_seed": [7], "manual_seed_all": [7]}


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

        def with_options(self, *, secrets=None, **_kwargs):
            captured.setdefault("with_options", []).append({"secrets": secrets})
            return FakeClassHandle()

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

    class FakeSecret:
        @staticmethod
        def from_dict(payload):
            captured["secret_payload"] = dict(payload)
            return {"secret_payload": dict(payload)}

    class FakeModal:
        Cls = FakeCls
        Secret = FakeSecret

    monkeypatch.setattr("sigmaevolve.modal.require_modal", lambda: FakeModal)

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
    assert captured["secret_payload"] == {
        "WANDB_API_KEY": "wandb-test-key",
        "WANDB_PROJECT": "sigmaevolve",
    }
    assert captured["with_options"] == [
        {
            "secrets": [
                {
                    "secret_payload": {
                        "WANDB_API_KEY": "wandb-test-key",
                        "WANDB_PROJECT": "sigmaevolve",
                    }
                }
            ],
        }
    ]
    assert metadata == {
        "kind": "modal",
        "run_id": "fc-123",
        "run_url": "https://modal.com/apps/test/runs/fc-123",
    }


def test_modal_launcher_surfaces_spawn_failure(monkeypatch):
    captured = {}

    class FakeMethodHandle:
        def __init__(self):
            pass

        def spawn(self, **kwargs):
            captured["spawn"] = dict(kwargs)
            raise RuntimeError("capacity unavailable")

    class FakeObjectHandle:
        def __init__(self):
            self.run_trial = FakeMethodHandle()

    class FakeClassHandle:
        def with_options(self, *, secrets=None, **_kwargs):
            captured["with_options"] = {"secrets": secrets}
            return FakeClassHandle()

        def __call__(self):
            return FakeObjectHandle()

    class FakeCls:
        @staticmethod
        def from_name(app_name, name, environment_name=None):
            captured["lookup"] = {
                "app_name": app_name,
                "name": name,
                "environment_name": environment_name,
            }
            return FakeClassHandle()

    class FakeSecret:
        @staticmethod
        def from_dict(payload):
            return {"secret_payload": dict(payload)}

    class FakeModal:
        Cls = FakeCls
        Secret = FakeSecret

    monkeypatch.setattr("sigmaevolve.modal.require_modal", lambda: FakeModal)

    launcher = create_modal_launcher(
        app_name="sigmaevolve-runner",
        function_name="run_trial",
        database_url="postgresql://example/db",
        dataset_root="/mnt/datasets",
        environment_name="main",
        wandb_env={"WANDB_API_KEY": "wandb-test-key"},
    )

    with pytest.raises(RuntimeError, match="capacity unavailable"):
        launcher.launch_trial("trial_1", "dispatch_1")

    assert captured["spawn"]["trial_id"] == "trial_1"
    assert captured["spawn"]["dispatch_token"] == "dispatch_1"
    assert captured["spawn"]["database_url"] == "postgresql://example/db"
    assert captured["spawn"]["dataset_root"] == "/mnt/datasets"


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

    monkeypatch.setattr("sigmaevolve.modal.require_modal", lambda: FakeModal)

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


def test_modal_launcher_surfaces_class_lookup_errors(monkeypatch):

    class FakeCls:
        @staticmethod
        def from_name(app_name, name, environment_name=None):
            del app_name, name, environment_name
            raise RuntimeError(
                "Class 'TrialRunner' not found in app 'sigmaevolve-runner'."
            )

    class FakeModal:
        Cls = FakeCls
        Secret = type(
            "FakeSecret",
            (),
            {
                "from_dict": staticmethod(
                    lambda payload: {"secret_payload": dict(payload)}
                )
            },
        )

    monkeypatch.setattr("sigmaevolve.modal.require_modal", lambda: FakeModal)

    launcher = create_modal_launcher(
        app_name="sigmaevolve-runner",
        function_name="run_trial",
        database_url="postgresql://example/db",
        dataset_root="/mnt/datasets",
        environment_name="main",
        wandb_env={"WANDB_API_KEY": "wandb-test-key"},
    )

    with pytest.raises(RuntimeError, match="TrialRunner"):
        launcher.launch_trial("trial_1", "dispatch_1")
