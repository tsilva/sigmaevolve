from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import json
from typing import Any, Iterable

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from sigmaevolve.hashing import compute_script_hash, normalize_source
from sigmaevolve.models import (
    ACTIVE_STATUSES,
    ERROR_OUTCOMES,
    OUTCOME_DUPLICATE,
    OUTCOME_GENERATION_FAILED,
    OUTCOME_STALE,
    SUCCESS_OUTCOMES,
    TERMINAL_STATUSES,
    TERMINAL_OUTCOMES,
    TRIAL_STATUS_ACTIVE,
    TRIAL_STATUS_DISPATCHING,
    TRIAL_STATUS_ERROR,
    TRIAL_STATUS_FINISHED,
    TRIAL_STATUS_QUEUED,
    DatasetRecord,
    MigrationResult,
    TrackRecord,
    TrialRecord,
    TrialSummary,
    make_id,
    now_utc,
)
from sigmaevolve.scoring import compute_score


metadata = sa.MetaData()


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://") :]
    if database_url.startswith("postgresql://") and "+psycopg" not in database_url:
        database_url = "postgresql+psycopg://" + database_url[len("postgresql://") :]
    return database_url


ALLOWED_GENERATION_BACKENDS = frozenset({"openrouter"})


def _is_prompt_message(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    role = entry.get("role")
    content = entry.get("content")
    return isinstance(role, str) and bool(role.strip()) and isinstance(content, str) and bool(content.strip())


def _validate_trial_provenance(provenance_json: dict[str, Any]) -> dict[str, Any]:
    payload = dict(provenance_json or {})
    backend = payload.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("Queued trials require provenance_json.backend.")
    if backend == "baseline":
        return payload
    if backend not in ALLOWED_GENERATION_BACKENDS:
        raise ValueError(
            "Queued non-baseline trials must come from the recorded LLM prompting pipeline; "
            f"unsupported backend {backend!r}."
        )
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("LLM-generated trials require provenance_json.model.")
    generation_config = payload.get("generation_config")
    if not isinstance(generation_config, dict):
        raise ValueError("LLM-generated trials require provenance_json.generation_config.")
    request_messages = payload.get("request_messages")
    if not isinstance(request_messages, list) or not request_messages:
        raise ValueError("LLM-generated trials require non-empty provenance_json.request_messages.")
    if not all(_is_prompt_message(entry) for entry in request_messages):
        raise ValueError(
            "LLM-generated trials require provenance_json.request_messages entries with string role and content."
        )
    context_trial_ids = payload.get("context_trial_ids")
    if not isinstance(context_trial_ids, list):
        raise ValueError("LLM-generated trials require provenance_json.context_trial_ids.")
    candidate_kind = payload.get("candidate_kind")
    if not isinstance(candidate_kind, str) or not candidate_kind.strip():
        raise ValueError("LLM-generated trials require provenance_json.candidate_kind.")
    return payload

datasets_table = sa.Table(
    "datasets",
    metadata,
    sa.Column("dataset_id", sa.String(255), primary_key=True),
    sa.Column("manifest_path", sa.Text(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

tracks_table = sa.Table(
    "tracks",
    metadata,
    sa.Column("track_id", sa.String(255), primary_key=True),
    sa.Column("name", sa.String(255), nullable=True),
    sa.Column("dataset_id", sa.String(255), sa.ForeignKey("datasets.dataset_id"), nullable=False),
    sa.Column("policy_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

trials_table = sa.Table(
    "trials",
    metadata,
    sa.Column("trial_id", sa.String(255), primary_key=True),
    sa.Column("track_id", sa.String(255), sa.ForeignKey("tracks.track_id"), nullable=False),
    sa.Column("source", sa.Text(), nullable=False),
    sa.Column("script_hash", sa.String(64), nullable=False),
    sa.Column("provenance_json", sa.JSON(), nullable=False),
    sa.Column("status", sa.String(32), nullable=False),
    sa.Column("outcome_reason", sa.String(32), nullable=True),
    sa.Column("dispatch_token", sa.String(255), nullable=True),
    sa.Column("dispatch_deadline_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("runner_id", sa.String(255), nullable=True),
    sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("metrics_json", sa.JSON(), nullable=True),
    sa.Column("score", sa.Float(), nullable=False, server_default="0"),
    sa.Column("error_json", sa.JSON(), nullable=True),
    sa.Column("dispatch_attempts", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("track_id", "script_hash", name="uq_trials_track_script_hash"),
)

sa.Index("ix_trials_track_created_at_desc", trials_table.c.track_id, trials_table.c.created_at.desc())
sa.Index(
    "ix_trials_track_status_created_at_desc",
    trials_table.c.track_id,
    trials_table.c.status,
    trials_table.c.created_at.desc(),
)


def _row_to_dataset(row: sa.Row[Any]) -> DatasetRecord:
    return DatasetRecord(
        dataset_id=row.dataset_id,
        manifest_path=row.manifest_path,
        created_at=row.created_at,
    )


def _row_to_track(row: sa.Row[Any]) -> TrackRecord:
    return TrackRecord(
        track_id=row.track_id,
        name=row.name,
        dataset_id=row.dataset_id,
        policy_json=dict(row.policy_json),
        created_at=row.created_at,
    )


def _row_to_trial(row: sa.Row[Any]) -> TrialRecord:
    return TrialRecord(
        trial_id=row.trial_id,
        track_id=row.track_id,
        source=row.source,
        script_hash=row.script_hash,
        provenance_json=dict(row.provenance_json or {}),
        status=row.status,
        outcome_reason=row.outcome_reason,
        dispatch_token=row.dispatch_token,
        dispatch_deadline_at=row.dispatch_deadline_at,
        runner_id=row.runner_id,
        heartbeat_at=row.heartbeat_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        metrics_json=dict(row.metrics_json) if row.metrics_json else None,
        score=float(row.score or 0.0),
        error_json=dict(row.error_json) if row.error_json else None,
        dispatch_attempts=int(row.dispatch_attempts),
        created_at=row.created_at,
    )


def _trial_summary_sort_key(summary: TrialSummary) -> tuple[float, float, float]:
    metrics = summary.metrics_json or {}
    accuracy = float(metrics.get("accuracy") or 0.0)
    time_to_best = metrics.get("time_to_best_eval_sec")
    if time_to_best is None:
        time_to_best = float("inf")
    return (-accuracy, float(time_to_best), -summary.score)


def _has_error_signal(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    reason = payload.get("reason")
    if isinstance(reason, str) and reason.strip():
        return True
    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        return True
    stderr = payload.get("stderr")
    if isinstance(stderr, str) and stderr.strip():
        return True
    return payload.get("returncode") is not None


def _build_generation_attempt_source(trial_id: str, outcome_reason: str) -> str:
    return normalize_source(
        "\n".join(
            [
                "# sigmaevolve generation attempt",
                f"# trial_id: {trial_id}",
                f"# outcome_reason: {outcome_reason}",
                "# diagnostic_source: true",
                "raise RuntimeError('diagnostic generation attempt source; see provenance_json.generation')",
            ]
        )
        + "\n"
    )


def _status_for_outcome_reason(outcome_reason: str) -> str:
    if outcome_reason in ERROR_OUTCOMES:
        return TRIAL_STATUS_ERROR
    return TRIAL_STATUS_FINISHED


def _classify_error_type(outcome_reason: str, error_json: dict[str, Any] | None) -> str | None:
    payload = dict(error_json or {})
    explicit = payload.get("error_type")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    reason = payload.get("reason")
    if not isinstance(reason, str):
        reason = None

    if outcome_reason == OUTCOME_GENERATION_FAILED:
        if reason in {"candidate_materialization_failed", "generation_assertion_failed"}:
            return "generation_invalid_candidate"
        if reason == "generator_exception":
            return "generation_backend_exception"
        if reason in {
            "provider_http_error",
            "provider_request_failed",
            "provider_response_invalid_json",
            "provider_response_missing_choices",
            "provider_response_missing_content",
        }:
            return "generation_provider_failure"
        return "generation_failed"

    if outcome_reason == "crashed":
        return "execution_crash"

    if outcome_reason == "eval_failed":
        if reason == "train_script_contract_violation":
            return "execution_contract_violation"
        if reason == "prediction_load_failed":
            return "evaluation_artifact_error"
        if reason == "predictions_missing":
            return "evaluation_predictions_missing"
        return "evaluation_failed"

    if outcome_reason == OUTCOME_STALE:
        if reason == "dispatch_deadline_expired":
            return "dispatch_stale"
        if reason == "heartbeat_stale":
            return "runner_stale"
        return "stale"

    return None


def _prepare_error_payload(outcome_reason: str, error_json: dict[str, Any] | None) -> dict[str, Any] | None:
    payload = dict(error_json or {})
    error_type = _classify_error_type(outcome_reason, payload)
    if error_type:
        payload["error_type"] = error_type
    return payload or None


def _row_to_trial_summary(row: sa.Row[Any]) -> TrialSummary:
    return TrialSummary(
        trial_id=row.trial_id,
        score=float(row.score or 0.0),
        metrics_json=dict(row.metrics_json) if row.metrics_json else None,
        source=row.source,
        provenance_json=dict(row.provenance_json or {}),
        outcome_reason=row.outcome_reason,
        error_json=dict(row.error_json) if row.error_json else None,
    )


class SQLAlchemyRepository:
    def __init__(self, database_url: str) -> None:
        database_url = normalize_database_url(database_url)
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        engine_kwargs: dict[str, Any] = {"future": True, "connect_args": connect_args}
        if not database_url.startswith("sqlite"):
            engine_kwargs["pool_pre_ping"] = True
        self.engine: Engine = sa.create_engine(database_url, **engine_kwargs)
        metadata.create_all(self.engine)

    @contextmanager
    def transaction(self) -> Iterable[Connection]:
        if self.engine.dialect.name == "sqlite":
            with self.engine.connect() as conn:
                conn.exec_driver_sql("BEGIN IMMEDIATE")
                try:
                    yield conn
                except Exception:
                    conn.rollback()
                    raise
                else:
                    conn.commit()
        else:
            with self.engine.begin() as conn:
                yield conn

    def _notify_dashboard(self, conn: Connection, track_id: str, reason: str) -> None:
        if self.engine.dialect.name not in {"postgresql", "postgres"}:
            return
        payload = {"trackId": track_id, "reason": reason}
        conn.execute(
            sa.text("SELECT pg_notify(:channel, :payload)"),
            {"channel": "sigmaevolve_dashboard", "payload": json.dumps(payload, sort_keys=True)},
        )

    def _update_trial_state(
        self,
        conn: Connection,
        *,
        trial_id: str,
        values: dict[str, Any],
        where: list[Any] | None = None,
        notify: bool = True,
    ) -> int:
        conditions = [trials_table.c.trial_id == trial_id]
        if where:
            conditions.extend(where)
        result = conn.execute(sa.update(trials_table).where(sa.and_(*conditions)).values(**values))
        if notify and result.rowcount:
            track_id = conn.execute(
                sa.select(trials_table.c.track_id).where(trials_table.c.trial_id == trial_id)
            ).scalar_one_or_none()
            if track_id is not None:
                self._notify_dashboard(conn, track_id=track_id, reason="trial_changed")
        return int(result.rowcount)

    def register_dataset(self, dataset_id: str, manifest_path: str | None) -> DatasetRecord:
        created_at = now_utc()
        with self.transaction() as conn:
            if self.engine.dialect.name == "sqlite":
                conn.execute(
                    sa.insert(datasets_table)
                    .values(
                        dataset_id=dataset_id,
                        manifest_path=manifest_path,
                        created_at=created_at,
                    )
                    .prefix_with("OR REPLACE")
                )
            else:
                existing = conn.execute(
                    sa.select(datasets_table).where(datasets_table.c.dataset_id == dataset_id)
                ).fetchone()
                if existing:
                    conn.execute(
                        sa.update(datasets_table)
                        .where(datasets_table.c.dataset_id == dataset_id)
                        .values(manifest_path=manifest_path, created_at=created_at)
                    )
                else:
                    conn.execute(
                        sa.insert(datasets_table).values(
                            dataset_id=dataset_id,
                            manifest_path=manifest_path,
                            created_at=created_at,
                        )
                    )
            row = conn.execute(
                sa.select(datasets_table).where(datasets_table.c.dataset_id == dataset_id)
            ).one()
        return _row_to_dataset(row)

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                sa.select(datasets_table).where(datasets_table.c.dataset_id == dataset_id)
            ).fetchone()
        return _row_to_dataset(row) if row else None

    def create_track(self, name: str | None, dataset_id: str, policy_json: dict[str, Any]) -> TrackRecord:
        track_id = make_id("track")
        created_at = now_utc()
        with self.transaction() as conn:
            conn.execute(
                sa.insert(tracks_table).values(
                    track_id=track_id,
                    name=name,
                    dataset_id=dataset_id,
                    policy_json=policy_json,
                    created_at=created_at,
                )
            )
            row = conn.execute(sa.select(tracks_table).where(tracks_table.c.track_id == track_id)).one()
            self._notify_dashboard(conn, track_id=track_id, reason="track_changed")
        return _row_to_track(row)

    def get_track(self, track_id: str) -> TrackRecord | None:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(tracks_table).where(tracks_table.c.track_id == track_id)).fetchone()
        return _row_to_track(row) if row else None

    def create_queued_trial_if_absent(
        self,
        track_id: str,
        source: str,
        provenance_json: dict[str, Any],
    ) -> tuple[TrialRecord | None, bool]:
        validated_provenance = _validate_trial_provenance(provenance_json)
        normalized_source = normalize_source(source)
        script_hash = compute_script_hash(normalized_source)
        created_at = now_utc()
        trial_id = make_id("trial")
        with self.transaction() as conn:
            existing = conn.execute(
                sa.select(trials_table).where(
                    sa.and_(
                        trials_table.c.track_id == track_id,
                        trials_table.c.script_hash == script_hash,
                    )
                )
            ).fetchone()
            if existing:
                return _row_to_trial(existing), False
            conn.execute(
                sa.insert(trials_table).values(
                    trial_id=trial_id,
                    track_id=track_id,
                    source=normalized_source,
                    script_hash=script_hash,
                    provenance_json=validated_provenance,
                    status=TRIAL_STATUS_QUEUED,
                    outcome_reason=None,
                    dispatch_token=None,
                    dispatch_deadline_at=None,
                    runner_id=None,
                    heartbeat_at=None,
                    started_at=None,
                    finished_at=None,
                    metrics_json=None,
                    score=0.0,
                    error_json=None,
                    dispatch_attempts=0,
                    created_at=created_at,
                )
            )
            row = conn.execute(sa.select(trials_table).where(trials_table.c.trial_id == trial_id)).one()
            self._notify_dashboard(conn, track_id=track_id, reason="trial_changed")
        return _row_to_trial(row), True

    def create_generation_attempt_trial(
        self,
        track_id: str,
        provenance_json: dict[str, Any],
        *,
        outcome_reason: str,
        error_json: dict[str, Any] | None,
    ) -> TrialRecord:
        if outcome_reason not in {OUTCOME_DUPLICATE, OUTCOME_GENERATION_FAILED}:
            raise ValueError(f"Unsupported generation attempt outcome_reason: {outcome_reason}")
        validated_provenance = _validate_trial_provenance(provenance_json)
        trial_id = make_id("trial")
        source = _build_generation_attempt_source(trial_id, outcome_reason)
        script_hash = compute_script_hash(source)
        created_at = now_utc()
        with self.transaction() as conn:
            conn.execute(
                sa.insert(trials_table).values(
                    trial_id=trial_id,
                    track_id=track_id,
                    source=source,
                    script_hash=script_hash,
                    provenance_json=validated_provenance,
                    status=_status_for_outcome_reason(outcome_reason),
                    outcome_reason=outcome_reason,
                    dispatch_token=None,
                    dispatch_deadline_at=None,
                    runner_id=None,
                    heartbeat_at=created_at,
                    started_at=None,
                    finished_at=created_at,
                    metrics_json=None,
                    score=0.0,
                    error_json=_prepare_error_payload(outcome_reason, error_json),
                    dispatch_attempts=0,
                    created_at=created_at,
                )
            )
            row = conn.execute(sa.select(trials_table).where(trials_table.c.trial_id == trial_id)).one()
            self._notify_dashboard(conn, track_id=track_id, reason="trial_changed")
        return _row_to_trial(row)

    def get_trial(self, trial_id: str) -> TrialRecord | None:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(trials_table).where(trials_table.c.trial_id == trial_id)).fetchone()
        return _row_to_trial(row) if row else None

    def record_trial_launcher_metadata(self, trial_id: str, launcher_metadata: dict[str, Any]) -> None:
        payload = dict(launcher_metadata)
        with self.transaction() as conn:
            row = conn.execute(
                sa.select(trials_table.c.track_id, trials_table.c.provenance_json).where(trials_table.c.trial_id == trial_id)
            ).fetchone()
            if row is None:
                raise KeyError(f"Trial not found: {trial_id}")
            provenance_json = dict(row.provenance_json or {})
            updated_provenance_json = dict(provenance_json)
            updated_provenance_json["launcher"] = payload
            if updated_provenance_json == provenance_json:
                return
            conn.execute(
                sa.update(trials_table)
                .where(trials_table.c.trial_id == trial_id)
                .values(provenance_json=updated_provenance_json)
            )
            self._notify_dashboard(conn, track_id=row.track_id, reason="trial_changed")

    def list_trials(self, track_id: str, statuses: set[str] | None = None) -> list[TrialRecord]:
        stmt = sa.select(trials_table).where(trials_table.c.track_id == track_id).order_by(trials_table.c.created_at)
        if statuses:
            stmt = stmt.where(trials_table.c.status.in_(sorted(statuses)))
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_to_trial(row) for row in rows]

    def sample_trial_context(self, track_id: str, limit: int, candidate_kind: str | None = None) -> list[TrialSummary]:
        stmt = (
            sa.select(trials_table)
            .where(
                sa.and_(
                    trials_table.c.track_id == track_id,
                    trials_table.c.status == TRIAL_STATUS_FINISHED,
                    trials_table.c.outcome_reason.in_(sorted(SUCCESS_OUTCOMES)),
                    trials_table.c.metrics_json.is_not(None),
                )
            )
            .order_by(trials_table.c.finished_at.desc(), trials_table.c.created_at.desc())
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        summaries = [
            _row_to_trial_summary(row)
            for row in rows
            if candidate_kind is None or dict(row.provenance_json or {}).get("candidate_kind") == candidate_kind
        ]
        return sorted(summaries, key=_trial_summary_sort_key)[:limit]

    def list_recent_trial_summaries(
        self,
        track_id: str,
        *,
        outcome_reasons: set[str] | None = None,
        require_metrics: bool | None = None,
        limit: int = 5,
    ) -> list[TrialSummary]:
        stmt = sa.select(trials_table).where(
            sa.and_(
                trials_table.c.track_id == track_id,
                trials_table.c.status.in_(sorted(TERMINAL_STATUSES)),
            )
        )
        if outcome_reasons:
            stmt = stmt.where(trials_table.c.outcome_reason.in_(sorted(outcome_reasons)))
        if require_metrics is True:
            stmt = stmt.where(trials_table.c.metrics_json.is_not(None))
        elif require_metrics is False:
            stmt = stmt.where(trials_table.c.metrics_json.is_(None))
        stmt = stmt.order_by(trials_table.c.finished_at.desc(), trials_table.c.created_at.desc()).limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_to_trial_summary(row) for row in rows]

    def count_trials(self, track_id: str, statuses: set[str] | None = None) -> int:
        stmt = sa.select(sa.func.count()).select_from(trials_table).where(trials_table.c.track_id == track_id)
        if statuses:
            stmt = stmt.where(trials_table.c.status.in_(sorted(statuses)))
        with self.engine.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def reserve_trials(
        self,
        track_id: str,
        max_parallelism: int,
        dispatch_ttl_sec: int,
        limit: int | None = None,
    ) -> list[TrialRecord]:
        reserved: list[TrialRecord] = []
        limit = limit or max_parallelism
        with self.transaction() as conn:
            active_count = int(
                conn.execute(
                    sa.select(sa.func.count())
                    .select_from(trials_table)
                    .where(
                        sa.and_(
                            trials_table.c.track_id == track_id,
                            trials_table.c.status.in_(sorted(ACTIVE_STATUSES)),
                        )
                    )
                ).scalar_one()
            )
            available = max(0, max_parallelism - active_count)
            for _ in range(min(limit, available)):
                stmt = (
                    sa.select(trials_table)
                    .where(
                        sa.and_(
                            trials_table.c.track_id == track_id,
                            trials_table.c.status == TRIAL_STATUS_QUEUED,
                        )
                    )
                    .order_by(trials_table.c.created_at, trials_table.c.trial_id)
                    .limit(1)
                )
                if self.engine.dialect.name != "sqlite":
                    stmt = stmt.with_for_update(skip_locked=True)
                row = conn.execute(stmt).fetchone()
                if not row:
                    break
                dispatch_token = make_id("dispatch")
                deadline = now_utc() + timedelta(seconds=dispatch_ttl_sec)
                conn.execute(
                    sa.update(trials_table)
                    .where(
                        sa.and_(
                            trials_table.c.trial_id == row.trial_id,
                            trials_table.c.status == TRIAL_STATUS_QUEUED,
                        )
                    )
                    .values(
                        status=TRIAL_STATUS_DISPATCHING,
                        dispatch_token=dispatch_token,
                        dispatch_deadline_at=deadline,
                        dispatch_attempts=int(row.dispatch_attempts) + 1,
                    )
                )
                updated = conn.execute(
                    sa.select(trials_table).where(trials_table.c.trial_id == row.trial_id)
                ).one()
                reserved.append(_row_to_trial(updated))
            if reserved:
                self._notify_dashboard(conn, track_id=track_id, reason="trial_changed")
        return reserved

    def claim_trial(self, trial_id: str, dispatch_token: str, runner_id: str) -> TrialRecord | None:
        with self.transaction() as conn:
            now = now_utc()
            updated = self._update_trial_state(
                conn,
                trial_id=trial_id,
                where=[
                    trials_table.c.status == TRIAL_STATUS_DISPATCHING,
                    trials_table.c.dispatch_token == dispatch_token,
                ],
                values={
                    "status": TRIAL_STATUS_ACTIVE,
                    "runner_id": runner_id,
                    "started_at": now,
                    "heartbeat_at": now,
                },
            )
            if updated != 1:
                return None
            row = conn.execute(sa.select(trials_table).where(trials_table.c.trial_id == trial_id)).one()
        return _row_to_trial(row)

    def heartbeat_trial(self, trial_id: str, runner_id: str, meta: dict[str, Any] | None = None) -> None:
        payload = dict(meta or {})
        with self.transaction() as conn:
            result = conn.execute(
                sa.update(trials_table)
                .where(
                    sa.and_(
                        trials_table.c.trial_id == trial_id,
                        trials_table.c.status == TRIAL_STATUS_ACTIVE,
                        trials_table.c.runner_id == runner_id,
                    )
                )
                .values(heartbeat_at=now_utc(), error_json=payload if _has_error_signal(payload) else None)
            )

    def update_active_trial_metrics(self, trial_id: str, runner_id: str, metrics: dict[str, Any]) -> None:
        payload = dict(metrics or {})
        with self.transaction() as conn:
            row = conn.execute(
                sa.select(trials_table.c.track_id, trials_table.c.metrics_json).where(
                    sa.and_(
                        trials_table.c.trial_id == trial_id,
                        trials_table.c.status == TRIAL_STATUS_ACTIVE,
                        trials_table.c.runner_id == runner_id,
                    )
                )
            ).fetchone()
            if row is None:
                return
            existing = dict(row.metrics_json) if row.metrics_json else None
            if existing == payload:
                return
            result = conn.execute(
                sa.update(trials_table)
                .where(
                    sa.and_(
                        trials_table.c.trial_id == trial_id,
                        trials_table.c.status == TRIAL_STATUS_ACTIVE,
                        trials_table.c.runner_id == runner_id,
                    )
                )
                .values(metrics_json=payload)
            )
            if result.rowcount:
                self._notify_dashboard(conn, track_id=row.track_id, reason="trial_changed")

    def finalize_trial(
        self,
        trial_id: str,
        runner_id: str | None,
        outcome_reason: str,
        metrics: dict[str, Any] | None,
        score: float,
        error_info: dict[str, Any] | None,
    ) -> None:
        if outcome_reason not in TERMINAL_OUTCOMES:
            raise ValueError(f"Unsupported outcome_reason: {outcome_reason}")
        if metrics is None:
            score = 0.0
        persisted_error_info = dict(error_info) if error_info else None
        if outcome_reason in SUCCESS_OUTCOMES and not _has_error_signal(persisted_error_info):
            persisted_error_info = None
        with self.transaction() as conn:
            where = [trials_table.c.trial_id == trial_id]
            if runner_id is not None:
                where.append(trials_table.c.runner_id == runner_id)
                where.append(trials_table.c.status == TRIAL_STATUS_ACTIVE)
            now = now_utc()
            updated = self._update_trial_state(
                conn,
                trial_id=trial_id,
                where=where[1:],
                values={
                    "status": _status_for_outcome_reason(outcome_reason),
                    "outcome_reason": outcome_reason,
                    "finished_at": now,
                    "dispatch_token": None,
                    "dispatch_deadline_at": None,
                    "heartbeat_at": now,
                    "metrics_json": metrics,
                    "score": score,
                    "error_json": _prepare_error_payload(outcome_reason, persisted_error_info),
                },
            )
            if not updated:
                return

    def sweep_expired_dispatches(self, track_id: str, max_dispatch_retries: int) -> tuple[list[str], list[str]]:
        requeued: list[str] = []
        stale: list[str] = []
        with self.transaction() as conn:
            rows = conn.execute(
                sa.select(trials_table).where(
                    sa.and_(
                        trials_table.c.track_id == track_id,
                        trials_table.c.status == TRIAL_STATUS_DISPATCHING,
                        trials_table.c.dispatch_deadline_at.is_not(None),
                        trials_table.c.dispatch_deadline_at < now_utc(),
                    )
                )
            ).fetchall()
            for row in rows:
                if int(row.dispatch_attempts) < max_dispatch_retries:
                    conn.execute(
                        sa.update(trials_table)
                        .where(trials_table.c.trial_id == row.trial_id)
                        .values(
                            status=TRIAL_STATUS_QUEUED,
                            dispatch_token=None,
                            dispatch_deadline_at=None,
                            runner_id=None,
                        )
                    )
                    requeued.append(row.trial_id)
                else:
                    conn.execute(
                        sa.update(trials_table)
                        .where(trials_table.c.trial_id == row.trial_id)
                        .values(
                            status=TRIAL_STATUS_ERROR,
                            outcome_reason=OUTCOME_STALE,
                            finished_at=now_utc(),
                            dispatch_token=None,
                            dispatch_deadline_at=None,
                            score=0.0,
                            error_json=_prepare_error_payload(
                                OUTCOME_STALE,
                                {"reason": "dispatch_deadline_expired"},
                            ),
                        )
                    )
                    stale.append(row.trial_id)
            if requeued or stale:
                self._notify_dashboard(conn, track_id=track_id, reason="trial_changed")
        return requeued, stale

    def sweep_stale_active_trials(self, track_id: str, stale_ttl_sec: int) -> list[str]:
        stale: list[str] = []
        cutoff = now_utc() - timedelta(seconds=stale_ttl_sec)
        with self.transaction() as conn:
            rows = conn.execute(
                sa.select(trials_table).where(
                    sa.and_(
                        trials_table.c.track_id == track_id,
                        trials_table.c.status == TRIAL_STATUS_ACTIVE,
                        sa.or_(
                            trials_table.c.heartbeat_at < cutoff,
                            sa.and_(
                                trials_table.c.heartbeat_at.is_(None),
                                trials_table.c.started_at.is_not(None),
                                trials_table.c.started_at < cutoff,
                            ),
                        ),
                    )
                )
            ).fetchall()
            for row in rows:
                conn.execute(
                    sa.update(trials_table)
                    .where(trials_table.c.trial_id == row.trial_id)
                    .values(
                        status=TRIAL_STATUS_ERROR,
                        outcome_reason=OUTCOME_STALE,
                        finished_at=now_utc(),
                        score=0.0,
                        error_json=_prepare_error_payload(
                            OUTCOME_STALE,
                            {"reason": "heartbeat_stale"},
                        ),
                    )
                )
                stale.append(row.trial_id)
            if stale:
                self._notify_dashboard(conn, track_id=track_id, reason="trial_changed")
        return stale

    def rescore(self, track_id: str | None, scorer_config: dict[str, Any]) -> MigrationResult:
        updated = 0
        touched_track_ids: set[str] = set()
        with self.transaction() as conn:
            stmt = sa.select(trials_table).where(trials_table.c.status.in_(sorted(TERMINAL_STATUSES)))
            if track_id is not None:
                stmt = stmt.where(trials_table.c.track_id == track_id)
            rows = conn.execute(stmt).fetchall()
            for row in rows:
                new_score = compute_score(row.metrics_json, row.outcome_reason, scorer_config)
                conn.execute(
                    sa.update(trials_table)
                    .where(trials_table.c.trial_id == row.trial_id)
                    .values(score=new_score)
                )
                updated += 1
                touched_track_ids.add(row.track_id)
            for touched_track_id in sorted(touched_track_ids):
                self._notify_dashboard(conn, track_id=touched_track_id, reason="trial_changed")
        return MigrationResult(updated_trials=updated, scorer_config=dict(scorer_config))
