from __future__ import annotations

import json
from types import SimpleNamespace

from sigmaevolve.storage import trials_table


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
    assert json.loads(params["payload"]) == {"trackId": "track_pg", "reason": "track_changed"}


def test_track_and_trial_mutations_publish_dashboard_notifications(repository):
    repository.register_dataset("mnist:v1", "/tmp/manifest.json")

    notifications: list[tuple[str, str]] = []
    repository._notify_dashboard = lambda conn, track_id, reason: notifications.append((track_id, reason))  # type: ignore[method-assign]

    track = repository.create_track(name="dashboard", dataset_id="mnist:v1", policy_json={})
    assert notifications[-1] == (track.track_id, "track_changed")

    trial, created = repository.create_queued_trial_if_absent(
        track_id=track.track_id,
        source="print('candidate')\n",
        provenance_json={"backend": "test", "model": "worker"},
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
        score=0.75,
        error_info=None,
    )
    assert notifications[-1] == (track.track_id, "trial_changed")

    repository.rescore(track_id=track.track_id, scorer_config={"primary_metric": "accuracy"})
    assert notifications[-1] == (track.track_id, "trial_changed")


def test_trial_indexes_exist():
    index_names = {index.name for index in trials_table.indexes}

    assert "ix_trials_track_created_at_desc" in index_names
    assert "ix_trials_track_status_created_at_desc" in index_names

    status_index = next(index for index in trials_table.indexes if index.name == "ix_trials_track_status_created_at_desc")
    rendered = [str(expression) for expression in status_index.expressions]
    assert rendered[0].endswith("track_id")
    assert rendered[1].endswith("status")
    assert "created_at" in rendered[2]
