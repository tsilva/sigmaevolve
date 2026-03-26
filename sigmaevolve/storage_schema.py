from __future__ import annotations

import sqlalchemy as sa


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

sa.Index("ix_trials_track_created_at_desc", trials_table.c.track_id, trials_table.c.created_at.desc())
sa.Index(
    "ix_trials_track_status_created_at_desc",
    trials_table.c.track_id,
    trials_table.c.status,
    trials_table.c.created_at.desc(),
)
