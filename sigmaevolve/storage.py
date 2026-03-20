from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Iterable

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from sigmaevolve.hashing import compute_script_hash, normalize_source
from sigmaevolve.models import (
    ACTIVE_STATUSES,
    OUTCOME_STALE,
    OUTCOME_SUCCEEDED,
    SUCCESS_OUTCOMES,
    TERMINAL_OUTCOMES,
    TRIAL_STATUS_ACTIVE,
    TRIAL_STATUS_DISPATCHING,
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


class SQLAlchemyRepository:
    def __init__(self, database_url: str) -> None:
        database_url = normalize_database_url(database_url)
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine: Engine = sa.create_engine(database_url, future=True, connect_args=connect_args)
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
                    provenance_json=provenance_json,
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
        return _row_to_trial(row), True

    def get_trial(self, trial_id: str) -> TrialRecord | None:
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(trials_table).where(trials_table.c.trial_id == trial_id)).fetchone()
        return _row_to_trial(row) if row else None

    def list_trials(self, track_id: str, statuses: set[str] | None = None) -> list[TrialRecord]:
        stmt = sa.select(trials_table).where(trials_table.c.track_id == track_id).order_by(trials_table.c.created_at)
        if statuses:
            stmt = stmt.where(trials_table.c.status.in_(sorted(statuses)))
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_to_trial(row) for row in rows]

    def sample_trial_context(self, track_id: str, limit: int) -> list[TrialSummary]:
        stmt = (
            sa.select(trials_table)
            .where(
                sa.and_(
                    trials_table.c.track_id == track_id,
                    trials_table.c.status == TRIAL_STATUS_FINISHED,
                    trials_table.c.outcome_reason.in_(sorted(SUCCESS_OUTCOMES)),
                )
            )
            .order_by(trials_table.c.score.desc(), trials_table.c.finished_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [
            TrialSummary(
                trial_id=row.trial_id,
                score=float(row.score or 0.0),
                metrics_json=dict(row.metrics_json) if row.metrics_json else None,
                source=row.source,
                provenance_json=dict(row.provenance_json or {}),
            )
            for row in rows
        ]

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
        return reserved

    def claim_trial(self, trial_id: str, dispatch_token: str, runner_id: str) -> TrialRecord | None:
        with self.transaction() as conn:
            now = now_utc()
            result = conn.execute(
                sa.update(trials_table)
                .where(
                    sa.and_(
                        trials_table.c.trial_id == trial_id,
                        trials_table.c.status == TRIAL_STATUS_DISPATCHING,
                        trials_table.c.dispatch_token == dispatch_token,
                    )
                )
                .values(
                    status=TRIAL_STATUS_ACTIVE,
                    runner_id=runner_id,
                    started_at=now,
                    heartbeat_at=now,
                )
            )
            if result.rowcount != 1:
                return None
            row = conn.execute(sa.select(trials_table).where(trials_table.c.trial_id == trial_id)).one()
        return _row_to_trial(row)

    def heartbeat_trial(self, trial_id: str, runner_id: str, meta: dict[str, Any] | None = None) -> None:
        payload = dict(meta or {})
        with self.transaction() as conn:
            conn.execute(
                sa.update(trials_table)
                .where(
                    sa.and_(
                        trials_table.c.trial_id == trial_id,
                        trials_table.c.status == TRIAL_STATUS_ACTIVE,
                        trials_table.c.runner_id == runner_id,
                    )
                )
                .values(heartbeat_at=now_utc(), error_json=payload or None)
            )

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
        if outcome_reason != OUTCOME_SUCCEEDED:
            score = 0.0
        with self.transaction() as conn:
            where = [trials_table.c.trial_id == trial_id]
            if runner_id is not None:
                where.append(trials_table.c.runner_id == runner_id)
            conn.execute(
                sa.update(trials_table)
                .where(sa.and_(*where))
                .values(
                    status=TRIAL_STATUS_FINISHED,
                    outcome_reason=outcome_reason,
                    finished_at=now_utc(),
                    dispatch_token=None,
                    dispatch_deadline_at=None,
                    heartbeat_at=now_utc(),
                    metrics_json=metrics,
                    score=score,
                    error_json=error_info,
                )
            )

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
                            status=TRIAL_STATUS_FINISHED,
                            outcome_reason=OUTCOME_STALE,
                            finished_at=now_utc(),
                            dispatch_token=None,
                            dispatch_deadline_at=None,
                            score=0.0,
                            error_json={"reason": "dispatch_deadline_expired"},
                        )
                    )
                    stale.append(row.trial_id)
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
                        status=TRIAL_STATUS_FINISHED,
                        outcome_reason=OUTCOME_STALE,
                        finished_at=now_utc(),
                        score=0.0,
                        error_json={"reason": "heartbeat_stale"},
                    )
                )
                stale.append(row.trial_id)
        return stale

    def rescore(self, track_id: str | None, scorer_config: dict[str, Any]) -> MigrationResult:
        updated = 0
        with self.transaction() as conn:
            stmt = sa.select(trials_table).where(trials_table.c.status == TRIAL_STATUS_FINISHED)
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
        return MigrationResult(updated_trials=updated, scorer_config=dict(scorer_config))
