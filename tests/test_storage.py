from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from sigmaevolve.storage import (
    classify_error_type,
    tracks_table,
    trials_table,
)
from tests.support import make_llm_provenance


class _RecordingConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def execute(self, statement, params):
        self.calls.append((str(statement), params))


def _capture_dashboard_notifications(repository):
    notifications: list[tuple[str, str]] = []
    repository._notify_dashboard = (  # type: ignore[method-assign]
        lambda conn, track_id, reason: notifications.append((track_id, reason))
    )
    return notifications


def _create_track(repository):
    return repository.create_track(dataset_id="mnist:v1", policy_json={})


def test_fresh_schema_bootstrap_creates_only_tracks_and_trials(repository):
    inspector = sa.inspect(repository.engine)

    assert sorted(inspector.get_table_names()) == ["tracks", "trials"]


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
    notifications = _capture_dashboard_notifications(repository)

    track = _create_track(repository)
    assert notifications[-1] == (track.track_id, "track_changed")

    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None
    assert notifications[-1] == (track.track_id, "trial_changed")

    reserved = repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)
    assert len(reserved) == 1
    assert notifications[-1] == (track.track_id, "trial_changed")

    claimed = repository.claim_trial(reserved[0].trial_id, reserved[0].dispatch_token, "runner-1")
    assert claimed is not None
    assert notifications[-1] == (track.track_id, "trial_changed")

    repository.finalize_trial(
        trial_id=reserved[0].trial_id,
        runner_id="runner-1",
        outcome_reason="succeeded",
        metrics={"accuracy": 0.75},
        error_info=None,
    )
    assert notifications[-1] == (track.track_id, "trial_changed")


def test_update_active_trial_metrics_updates_only_matching_active_runner_and_notifies(repository):
    notifications = _capture_dashboard_notifications(repository)
    track = _create_track(repository)
    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None

    reserved = repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)
    claimed = repository.claim_trial(reserved[0].trial_id, reserved[0].dispatch_token, "runner-1")
    assert claimed is not None

    repository.update_active_trial_metrics(
        trial_id=claimed.trial_id,
        runner_id="runner-1",
        metrics={"accuracy": 0.5, "eval_count": 1, "last_phase": "train", "best_accuracy": 0.5},
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

    repository.update_active_trial_metrics(
        trial_id=claimed.trial_id,
        runner_id="runner-2",
        metrics={"accuracy": 0.9},
    )
    unchanged = repository.get_trial(claimed.trial_id)
    assert unchanged is not None
    assert unchanged.metrics_json == {
        "accuracy": 0.5,
        "eval_count": 1,
        "last_phase": "train",
    }


def test_finalize_trial_overwrites_interim_metrics_and_slims_payload(repository):
    track = _create_track(repository)
    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None

    reserved = repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)
    claimed = repository.claim_trial(reserved[0].trial_id, reserved[0].dispatch_token, "runner-1")
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
        metrics={
            "accuracy": 0.75,
            "eval_count": 2,
            "timed_out": False,
            "best_eval_index": 2,
            "process_elapsed_sec": 12.0,
        },
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


def test_failed_trials_persist_compact_error_payload_and_classify_on_read(repository):
    track = _create_track(repository)
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
        error_info={
            "reason": "prediction_load_failed",
            "detail": "missing predictions.npy",
            "stderr": "trace",
            "returncode": 1,
            "native_finish_reason": "length",
        },
    )

    updated = repository.get_trial(trial.trial_id)
    assert updated is not None
    assert updated.status == "error"
    assert updated.error_json == {
        "reason": "prediction_load_failed",
        "detail": "missing predictions.npy",
        "stderr": "trace",
        "returncode": 1,
    }
    assert classify_error_type(updated.outcome_reason or "", updated.error_json) == "evaluation_artifact_error"


def test_generation_attempt_trial_persists_slim_failure_provenance(repository):
    track = _create_track(repository)

    trial = repository.create_generation_attempt_trial(
        track_id=track.track_id,
        provenance_json=make_llm_provenance(
            model="worker",
            generation_index=3,
            duplicate_retry_count=2,
            provider_response_id="resp_1",
            generation={
                "system_prompt": "system",
                "user_prompt": "user",
                "response_text": "partial response",
                "generated_source": "print('x')\n",
                "assertions_passed": False,
                "assertion_failures": ["bad patch"],
            },
        ),
        outcome_reason="generation_failed",
        error_json={
            "reason": "candidate_materialization_failed",
            "finish_reason": "length",
            "native_finish_reason": "length",
            "error_type": "generation_output_truncated",
        },
    )

    assert trial.status == "error"
    assert trial.error_json == {
        "reason": "candidate_materialization_failed",
        "finish_reason": "length",
    }
    assert classify_error_type(trial.outcome_reason or "", trial.error_json) == "generation_output_truncated"
    assert trial.provenance_json == {
        "backend": "openrouter",
        "model": "worker",
        "candidate_kind": "strategy_v1",
        "generation_config": {
            "model": "worker",
            "temperature": 0.1,
            "max_tokens": 1500,
        },
        "request_messages": make_llm_provenance(model="worker")["request_messages"],
        "context_trial_ids": ["trial_parent"],
        "generation": {"response_text": "partial response"},
    }


def test_record_trial_launcher_metadata_merges_filtered_keys_and_notifies(repository):
    notifications = _capture_dashboard_notifications(repository)
    track = _create_track(repository)
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
    assert updated.provenance_json["launcher"] == {
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
        "run_id": "fc-123",
        "run_url": "https://modal.com/apps/test/runs/fc-123",
    }


def test_record_trial_wandb_metadata_merges_filtered_keys_and_notifies(repository):
    notifications = _capture_dashboard_notifications(repository)
    track = _create_track(repository)
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
            "entity": "team",
            "run_id": "wandb-run-1",
            "run_url": "https://wandb.example/sigmaevolve/wandb-run-1",
            "artifact_id": "ignored",
        },
    )
    updated = repository.get_trial(trial.trial_id)

    assert updated is not None
    assert updated.provenance_json["wandb"] == {
        "project": "sigmaevolve",
        "entity": "team",
        "run_id": "wandb-run-1",
        "run_url": "https://wandb.example/sigmaevolve/wandb-run-1",
    }
    assert notifications[-1] == (track.track_id, "trial_changed")


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


def test_sample_trial_context_includes_scored_timeouts_and_uses_time_to_best_eval_tiebreak(repository):
    track = _create_track(repository)

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
        error_info=None,
    )
    repository.finalize_trial(
        trial_id=slow_success.trial_id,
        runner_id=None,
        outcome_reason="succeeded",
        metrics={"accuracy": 0.9, "time_to_best_eval_sec": 3.0},
        error_info=None,
    )

    context = repository.sample_trial_context(track.track_id, limit=2)
    assert [trial.trial_id for trial in context] == [
        fast_timeout.trial_id,
        slow_success.trial_id,
    ]
    assert context[0].outcome_reason == "timeout"


def test_create_queued_trial_rejects_non_llm_candidate_provenance(repository):
    track = _create_track(repository)

    with pytest.raises(ValueError, match="recorded LLM prompting pipeline"):
        repository.create_queued_trial_if_absent(
            track_id=track.track_id,
            source="print('candidate')\n",
            provenance_json={"backend": "manual-curated", "model": "cnn-residual-ish"},
        )


def test_runner_finalize_does_not_overwrite_stale_terminal_state(repository):
    track = _create_track(repository)
    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json=make_llm_provenance(model="worker"),
    )
    assert created is True
    assert trial is not None

    reserved = repository.reserve_trials(track.track_id, max_parallelism=1, dispatch_ttl_sec=60, limit=1)
    claimed = repository.claim_trial(reserved[0].trial_id, reserved[0].dispatch_token, "runner-1")
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
        error_info=None,
    )
    updated = repository.get_trial(claimed.trial_id)

    assert updated is not None
    assert updated.status == "finished"
    assert updated.outcome_reason == "stale"
    assert updated.error_json == {"reason": "heartbeat_stale"}


def test_migrate_reduced_schema_rewrites_old_tables(repository):
    with repository.transaction() as conn:
        conn.execute(sa.text("DROP TABLE IF EXISTS trials"))
        conn.execute(sa.text("DROP TABLE IF EXISTS tracks"))
        conn.execute(sa.text("DROP TABLE IF EXISTS datasets"))
        conn.execute(
            sa.text(
                """
                CREATE TABLE datasets (
                    dataset_id TEXT PRIMARY KEY,
                    manifest_path TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        conn.execute(
            sa.text(
                """
                CREATE TABLE tracks (
                    track_id TEXT PRIMARY KEY,
                    name TEXT,
                    dataset_id TEXT NOT NULL,
                    policy_json JSON NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        conn.execute(
            sa.text(
                """
                CREATE TABLE trials (
                    trial_id TEXT PRIMARY KEY,
                    track_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    script_hash TEXT NOT NULL,
                    provenance_json JSON NOT NULL,
                    status TEXT NOT NULL,
                    outcome_reason TEXT,
                    dispatch_token TEXT,
                    dispatch_deadline_at TEXT,
                    runner_id TEXT,
                    heartbeat_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    metrics_json JSON,
                    score FLOAT NOT NULL DEFAULT 0,
                    error_json JSON,
                    dispatch_attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO tracks (track_id, name, dataset_id, policy_json, created_at)
                VALUES (
                    'track_1',
                    'legacy',
                    'mnist:v1',
                    :policy_json,
                    '2026-03-26T10:00:00+00:00'
                )
                """
            ),
            {
                "policy_json": json.dumps(
                    {
                        "epochs": 3,
                        "scorer_settings": {"primary_metric": "accuracy"},
                        "sampling_settings": {"seed": 7},
                        "generation_backend": {
                            "backend": "openrouter",
                            "selection": "round_robin",
                            "model_pool": [{"model": "test/model", "temperature": 0.1, "max_tokens": 1500}],
                        },
                    }
                )
            },
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO trials (
                    trial_id, track_id, source, script_hash, provenance_json, status, outcome_reason,
                    metrics_json, score, error_json, dispatch_attempts, created_at
                ) VALUES (
                    'trial_1',
                    'track_1',
                    'print(''candidate'')\n',
                    'hash_1',
                    :provenance_json,
                    'error',
                    'generation_failed',
                    :metrics_json,
                    0.91,
                    :error_json,
                    2,
                    '2026-03-26T10:01:00+00:00'
                )
                """
            ),
            {
                "provenance_json": json.dumps(
                    {
                        **make_llm_provenance(model="worker"),
                        "provider_response_id": "resp_1",
                        "launcher": {
                            "kind": "modal",
                            "run_id": "fc-123",
                            "run_url": "https://modal.com/apps/test/runs/fc-123",
                            "cancel_outcome": "requested",
                        },
                        "generation": {
                            "system_prompt": "system",
                            "user_prompt": "user",
                            "response_text": "partial response",
                            "generated_source": "print('candidate')\n",
                        },
                    }
                ),
                "metrics_json": json.dumps(
                    {
                        "accuracy": 0.91,
                        "val_loss": 0.12,
                        "best_eval_index": 2,
                        "process_elapsed_sec": 14.0,
                    }
                ),
                "error_json": json.dumps(
                    {
                        "reason": "candidate_materialization_failed",
                        "detail": "bad patch",
                        "finish_reason": "length",
                        "native_finish_reason": "length",
                        "error_type": "generation_output_truncated",
                    }
                ),
            },
        )

    payload = repository.migrate_reduced_schema()
    assert payload == {"migrated_tracks": 1, "migrated_trials": 1}

    inspector = sa.inspect(repository.engine)
    assert sorted(inspector.get_table_names()) == ["tracks", "trials"]
    assert {column["name"] for column in inspector.get_columns("tracks")} == {
        "track_id",
        "dataset_id",
        "policy_json",
        "created_at",
    }
    assert {column["name"] for column in inspector.get_columns("trials")} == {
        "trial_id",
        "track_id",
        "source",
        "script_hash",
        "provenance_json",
        "status",
        "outcome_reason",
        "dispatch_token",
        "dispatch_deadline_at",
        "runner_id",
        "heartbeat_at",
        "started_at",
        "finished_at",
        "metrics_json",
        "error_json",
        "dispatch_attempts",
        "created_at",
    }

    migrated_track = repository.get_track("track_1")
    assert migrated_track is not None
    assert migrated_track.policy_json["sampling_seed"] == 7
    assert "scorer_settings" not in migrated_track.policy_json
    assert "sampling_settings" not in migrated_track.policy_json
    assert "backend" not in migrated_track.policy_json["generation_backend"]

    migrated_trial = repository.get_trial("trial_1")
    assert migrated_trial is not None
    assert migrated_trial.metrics_json == {
        "accuracy": 0.91,
        "val_loss": 0.12,
    }
    assert migrated_trial.error_json == {
        "reason": "candidate_materialization_failed",
        "detail": "bad patch",
        "finish_reason": "length",
    }
    assert migrated_trial.provenance_json == {
        "backend": "openrouter",
        "model": "worker",
        "candidate_kind": "strategy_v1",
        "generation_config": {
            "model": "worker",
            "temperature": 0.1,
            "max_tokens": 1500,
        },
        "request_messages": make_llm_provenance(model="worker")["request_messages"],
        "context_trial_ids": ["trial_parent"],
        "launcher": {
            "run_id": "fc-123",
            "run_url": "https://modal.com/apps/test/runs/fc-123",
        },
        "generation": {"response_text": "partial response"},
    }
