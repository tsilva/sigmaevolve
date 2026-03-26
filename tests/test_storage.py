from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sigmaevolve.storage import trials_table
from tests.support import make_llm_provenance


class _RecordingConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def execute(self, statement, params):
        self.calls.append((str(statement), params))


def test_dashboard_notify_is_postgres_only(repository):
    conn = _RecordingConnection()

    repository._notify_dashboard(conn, track_id="track_sqlite", reason="trial_changed")
    assert conn.calls == []

    repository.engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    repository._notify_dashboard(conn, track_id="track_pg", reason="track_changed")

    assert len(conn.calls) == 1
    statement, params = conn.calls[0]
    assert "pg_notify" in statement
    assert params["channel"] == "sigmaevolve_dashboard"
    assert json.loads(params["payload"]) == {
        "trackId": "track_pg",
        "reason": "track_changed",
    }


def test_track_and_trial_mutations_publish_dashboard_notifications(repository):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")

    notifications: list[tuple[str, str]] = []
    repository._notify_dashboard = lambda conn, track_id, reason: notifications.append((track_id, reason))  # type: ignore[method-assign]

    track = repository.create_track(
        name="dashboard", dataset_id="mnist:v1", policy_json={}
    )
    assert notifications[-1] == (track.track_id, "track_changed")

    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None
    assert notifications[-1] == (track.track_id, "trial_changed")

    reserved = repository.reserve_trials(
        track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1
    )
    assert len(reserved) == 1
    assert notifications[-1] == (track.track_id, "trial_changed")

    claimed = repository.claim_trial(
        reserved[0].trial_id, reserved[0].dispatch_token, "runner-1"
    )
    assert claimed is not None
    assert notifications[-1] == (track.track_id, "trial_changed")

    repository.finalize_trial(
        trial_id=reserved[0].trial_id,
        runner_id="runner-1",
        outcome_reason="succeeded",
        metrics={"accuracy": 0.75},
        score=0.75,
        error_info=None,
    )
    assert notifications[-1] == (track.track_id, "trial_changed")


def test_update_active_trial_metrics_updates_only_matching_active_runner_and_notifies(
    repository,
):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")

    notifications: list[tuple[str, str]] = []
    repository._notify_dashboard = lambda conn, track_id, reason: notifications.append((track_id, reason))  # type: ignore[method-assign]

    track = repository.create_track(
        name="live-metrics", dataset_id="mnist:v1", policy_json={}
    )
    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None

    reserved = repository.reserve_trials(
        track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1
    )
    claimed = repository.claim_trial(
        reserved[0].trial_id, reserved[0].dispatch_token, "runner-1"
    )
    assert claimed is not None

    repository.update_active_trial_metrics(
        trial_id=claimed.trial_id,
        runner_id="runner-1",
        metrics={"accuracy": 0.5, "eval_count": 1, "last_phase": "train"},
    )
    updated = repository.get_trial(claimed.trial_id)
    assert updated is not None
    assert updated.metrics_json == {
        "accuracy": 0.5,
        "eval_count": 1,
        "last_phase": "train",
    }
    assert notifications[-1] == (track.track_id, "trial_changed")

    notify_count = len(notifications)
    repository.update_active_trial_metrics(
        trial_id=claimed.trial_id,
        runner_id="runner-1",
        metrics={"accuracy": 0.5, "eval_count": 1, "last_phase": "train"},
    )
    assert len(notifications) == notify_count

    notify_count = len(notifications)
    repository.update_active_trial_metrics(
        trial_id=claimed.trial_id,
        runner_id="runner-2",
        metrics={"accuracy": 0.9},
    )
    assert len(notifications) == notify_count
    unchanged = repository.get_trial(claimed.trial_id)
    assert unchanged is not None
    assert unchanged.metrics_json == {
        "accuracy": 0.5,
        "eval_count": 1,
        "last_phase": "train",
    }


def test_finalize_trial_overwrites_interim_active_metrics(repository):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")
    track = repository.create_track(
        name="final-metrics", dataset_id="mnist:v1", policy_json={}
    )

    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None

    reserved = repository.reserve_trials(
        track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1
    )
    claimed = repository.claim_trial(
        reserved[0].trial_id, reserved[0].dispatch_token, "runner-1"
    )
    assert claimed is not None

    repository.update_active_trial_metrics(
        trial_id=claimed.trial_id,
        runner_id="runner-1",
        metrics={"accuracy": 0.5, "eval_count": 1, "last_phase": "train"},
    )
    repository.finalize_trial(
        trial_id=claimed.trial_id,
        runner_id="runner-1",
        outcome_reason="succeeded",
        metrics={"accuracy": 0.75, "eval_count": 2, "timed_out": False},
        score=0.75,
        error_info=None,
    )

    updated = repository.get_trial(claimed.trial_id)
    assert updated is not None
    assert updated.status == "finished"
    assert updated.metrics_json == {
        "accuracy": 0.75,
        "eval_count": 2,
        "timed_out": False,
    }
    assert updated.score == 0.75


def test_failed_trials_persist_error_status_and_error_type(repository):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")
    track = repository.create_track(
        name="errors", dataset_id="mnist:v1", policy_json={}
    )

    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None

    repository.finalize_trial(
        trial_id=trial.trial_id,
        runner_id=None,
        outcome_reason="crashed",
        metrics=None,
        score=0.0,
        error_info={"reason": "boom"},
    )

    updated = repository.get_trial(trial.trial_id)
    assert updated is not None
    assert updated.status == "error"
    assert updated.error_json == {"reason": "boom", "error_type": "execution_crash"}


def test_eval_failed_prediction_load_is_classified_as_artifact_error(repository):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")
    track = repository.create_track(
        name="errors", dataset_id="mnist:v1", policy_json={}
    )

    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None

    repository.finalize_trial(
        trial_id=trial.trial_id,
        runner_id=None,
        outcome_reason="eval_failed",
        metrics=None,
        score=0.0,
        error_info={"reason": "prediction_load_failed"},
    )

    updated = repository.get_trial(trial.trial_id)
    assert updated is not None
    assert updated.status == "error"
    assert updated.error_json == {
        "reason": "prediction_load_failed",
        "error_type": "evaluation_artifact_error",
    }


def test_generation_attempt_trial_uses_specific_error_type_when_present(repository):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")
    track = repository.create_track(
        name="generation-errors", dataset_id="mnist:v1", policy_json={}
    )

    trial = repository.create_generation_attempt_trial(
        track_id=track.track_id,
        provenance_json=make_llm_provenance(model="worker"),
        outcome_reason="generation_failed",
        error_json={
            "reason": "provider_response_missing_content",
            "error_type": "generation_reasoning_tokens_exhausted",
        },
    )

    assert trial.status == "error"
    assert trial.error_json == {
        "reason": "provider_response_missing_content",
        "error_type": "generation_reasoning_tokens_exhausted",
    }


def test_generation_attempt_trial_classifies_length_limited_invalid_candidate_as_truncated(
    repository,
):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")
    track = repository.create_track(
        name="generation-errors", dataset_id="mnist:v1", policy_json={}
    )

    trial = repository.create_generation_attempt_trial(
        track_id=track.track_id,
        provenance_json=make_llm_provenance(model="worker"),
        outcome_reason="generation_failed",
        error_json={
            "reason": "candidate_materialization_failed",
            "finish_reason": "length",
        },
    )

    assert trial.status == "error"
    assert trial.error_json == {
        "reason": "candidate_materialization_failed",
        "finish_reason": "length",
        "error_type": "generation_output_truncated",
    }


def test_record_trial_launcher_metadata_merges_into_provenance_and_notifies(repository):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")

    notifications: list[tuple[str, str]] = []
    repository._notify_dashboard = lambda conn, track_id, reason: notifications.append((track_id, reason))  # type: ignore[method-assign]

    track = repository.create_track(
        name="launcher", dataset_id="mnist:v1", policy_json={}
    )
    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None

    repository.record_trial_launcher_metadata(
        trial.trial_id,
        {
            "kind": "modal",
            "run_id": "fc-123",
            "run_url": "https://modal.com/apps/test/runs/fc-123",
        },
    )
    updated = repository.get_trial(trial.trial_id)

    assert updated is not None
    assert updated.provenance_json["backend"] == "openrouter"
    assert updated.provenance_json["launcher"] == {
        "kind": "modal",
        "run_id": "fc-123",
        "run_url": "https://modal.com/apps/test/runs/fc-123",
    }
    assert notifications[-1] == (track.track_id, "trial_changed")

    repository.record_trial_launcher_metadata(
        trial.trial_id,
        {
            "cancel_outcome": "requested",
            "cancel_attempted_at": "2026-03-24T12:00:00+00:00",
        },
    )
    updated = repository.get_trial(trial.trial_id)

    assert updated is not None
    assert updated.provenance_json["launcher"] == {
        "kind": "modal",
        "run_id": "fc-123",
        "run_url": "https://modal.com/apps/test/runs/fc-123",
        "cancel_outcome": "requested",
        "cancel_attempted_at": "2026-03-24T12:00:00+00:00",
    }


def test_record_trial_wandb_metadata_merges_into_provenance_and_notifies(repository):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")

    notifications: list[tuple[str, str]] = []
    repository._notify_dashboard = lambda conn, track_id, reason: notifications.append((track_id, reason))  # type: ignore[method-assign]

    track = repository.create_track(name="wandb", dataset_id="mnist:v1", policy_json={})
    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None

    repository.record_trial_wandb_metadata(
        trial.trial_id,
        {
            "project": "sigmaevolve",
            "run_id": "wandb-run-1",
            "run_url": "https://wandb.example/sigmaevolve/wandb-run-1",
        },
    )
    updated = repository.get_trial(trial.trial_id)

    assert updated is not None
    assert updated.provenance_json["wandb"] == {
        "project": "sigmaevolve",
        "run_id": "wandb-run-1",
        "run_url": "https://wandb.example/sigmaevolve/wandb-run-1",
    }
    assert notifications[-1] == (track.track_id, "trial_changed")

    repository.record_trial_wandb_metadata(
        trial.trial_id,
        {
            "run_name": "track_1:trial_1",
        },
    )
    updated = repository.get_trial(trial.trial_id)

    assert updated is not None
    assert updated.provenance_json["wandb"] == {
        "project": "sigmaevolve",
        "run_id": "wandb-run-1",
        "run_url": "https://wandb.example/sigmaevolve/wandb-run-1",
        "run_name": "track_1:trial_1",
    }


def test_trial_indexes_exist():
    index_names = {index.name for index in trials_table.indexes}

    assert "ix_trials_track_created_at_desc" in index_names
    assert "ix_trials_track_status_created_at_desc" in index_names

    status_index = next(
        index
        for index in trials_table.indexes
        if index.name == "ix_trials_track_status_created_at_desc"
    )
    rendered = [str(expression) for expression in status_index.expressions]
    assert rendered[0].endswith("track_id")
    assert rendered[1].endswith("status")
    assert "created_at" in rendered[2]


def test_sample_trial_context_includes_scored_timeouts_and_uses_time_to_best_eval_tiebreak(
    repository,
):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")
    track = repository.create_track(
        name="context", dataset_id="mnist:v1", policy_json={}
    )

    fast_timeout, _ = repository.create_queued_trial_if_absent(
        track.track_id,
        "print('fast timeout')\n",
        make_llm_provenance(model="fast"),
    )
    slow_success, _ = repository.create_queued_trial_if_absent(
        track.track_id,
        "print('slow success')\n",
        make_llm_provenance(model="slow"),
    )
    assert fast_timeout is not None and slow_success is not None

    repository.finalize_trial(
        trial_id=fast_timeout.trial_id,
        runner_id=None,
        outcome_reason="timeout",
        metrics={"accuracy": 0.9, "time_to_best_eval_sec": 1.0},
        score=0.9,
        error_info=None,
    )
    repository.finalize_trial(
        trial_id=slow_success.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": 0.9, "time_to_best_eval_sec": 3.0},
        score=0.9,
        error_info=None,
    )

    context = repository.sample_trial_context(track.track_id, limit=2)
    assert [trial.trial_id for trial in context] == [
        fast_timeout.trial_id,
        slow_success.trial_id,
    ]
    assert context[0].outcome_reason == "timeout"


def test_create_queued_trial_rejects_non_llm_candidate_provenance(repository):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")
    track = repository.create_track(
        name="guardrail", dataset_id="mnist:v1", policy_json={}
    )

    with pytest.raises(ValueError, match="recorded LLM prompting pipeline"):
        repository.create_queued_trial_if_absent(
            track_id=track.track_id,
            source="print('candidate')\n",
            provenance_json={"backend": "manual-curated", "model": "cnn-residual-ish"},
        )


def test_runner_finalize_does_not_overwrite_stale_terminal_state(repository):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")
    track = repository.create_track(
        name="stale-guard", dataset_id="mnist:v1", policy_json={}
    )
    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None

    reserved = repository.reserve_trials(
        track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1
    )
    claimed = repository.claim_trial(
        reserved[0].trial_id, reserved[0].dispatch_token, "runner-1"
    )
    assert claimed is not None

    with repository.transaction() as conn:
        conn.execute(
            trials_table.update()
            .where(trials_table.c.trial_id == claimed.trial_id)
            .values(
                status="finished",
                outcome_reason="stale",
                error_json={"reason": "heartbeat_stale"},
            )
        )

    repository.finalize_trial(
        trial_id=claimed.trial_id,
        runner_id="runner-1",
        outcome_reason="succeeded",
        metrics={"accuracy": 1.0},
        score=1.0,
        error_info=None,
    )
    updated = repository.get_trial(claimed.trial_id)

    assert updated is not None
    assert updated.status == "finished"
    assert updated.outcome_reason == "stale"
    assert updated.error_json == {"reason": "heartbeat_stale"}
