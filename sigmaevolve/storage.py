from __future__ import annotations


# ---- storage_schema.py ----

import sqlalchemy as sa


metadata = sa.MetaData()


def normalize_database_url(database_url: str) -> str:
    # Accept historical postgres:// URLs by rewriting them to SQLAlchemy's canonical form.
    normalized_url = database_url
    postgres_prefix = "postgres://"
    if normalized_url.startswith(postgres_prefix):
        normalized_url = "postgresql://" + normalized_url[len(postgres_prefix) :]

    # Default PostgreSQL connections to the psycopg driver when no driver is specified.
    if normalized_url.startswith("postgresql://") and "+psycopg" not in normalized_url:
        normalized_url = "postgresql+psycopg://" + normalized_url[len("postgresql://") :]
    return normalized_url


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


# ---- storage_validation.py ----

from typing import Any

from sigmaevolve.core import normalize_source
from sigmaevolve.core import (
    ERROR_OUTCOMES,
    OUTCOME_GENERATION_FAILED,
    OUTCOME_STALE,
    TRIAL_STATUS_ERROR,
    TRIAL_STATUS_FINISHED,
)


ALLOWED_GENERATION_BACKENDS = frozenset({"openrouter"})


def _is_prompt_message(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False

    role = entry.get("role")
    content = entry.get("content")
    has_role = isinstance(role, str) and bool(role.strip())
    has_content = isinstance(content, str) and bool(content.strip())

    return has_role and has_content


def validate_trial_provenance(provenance_json: dict[str, Any]) -> dict[str, Any]:
    # Copy the caller payload before enforcing the persisted provenance contract.
    payload = dict(provenance_json or {})
    backend = payload.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("Queued trials require provenance_json.backend.")

    # Allow the system-seeded baseline candidate to bypass LLM request requirements.
    if backend == "baseline":
        return payload

    # Reject non-baseline candidates that did not come from a supported generation backend.
    if backend not in ALLOWED_GENERATION_BACKENDS:
        raise ValueError(
            "Queued non-baseline trials must come from the recorded LLM prompting pipeline; "
            f"unsupported backend {backend!r}."
        )

    # Require the model and generation config needed to audit candidate provenance.
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("LLM-generated trials require provenance_json.model.")
    generation_config = payload.get("generation_config")
    if not isinstance(generation_config, dict):
        raise ValueError("LLM-generated trials require provenance_json.generation_config.")

    # Require the prompt message history used to produce the generated candidate.
    request_messages = payload.get("request_messages")
    if not isinstance(request_messages, list) or not request_messages:
        raise ValueError("LLM-generated trials require non-empty provenance_json.request_messages.")
    if not all(_is_prompt_message(entry) for entry in request_messages):
        raise ValueError(
            "LLM-generated trials require provenance_json.request_messages entries with string role and content."
        )

    # Require the track-context metadata that shaped the generation request.
    context_trial_ids = payload.get("context_trial_ids")
    if not isinstance(context_trial_ids, list):
        raise ValueError("LLM-generated trials require provenance_json.context_trial_ids.")
    candidate_kind = payload.get("candidate_kind")
    if not isinstance(candidate_kind, str) or not candidate_kind.strip():
        raise ValueError("LLM-generated trials require provenance_json.candidate_kind.")
    return payload


def has_error_signal(payload: dict[str, Any] | None) -> bool:
    # Treat missing or empty payloads as the absence of an error signal.
    if not payload:
        return False

    # Recognize explicit textual error fields before falling back to a return code.
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


def build_generation_attempt_source(trial_id: str, outcome_reason: str) -> str:
    # Build a minimal diagnostic source file that points reviewers back to provenance_json.
    lines = [
        "# sigmaevolve generation attempt",
        f"# trial_id: {trial_id}",
        f"# outcome_reason: {outcome_reason}",
        "# diagnostic_source: true",
        "raise RuntimeError('diagnostic generation attempt source; see provenance_json.generation')",
    ]
    source = "\n".join(lines) + "\n"
    return normalize_source(source)


def status_for_outcome_reason(outcome_reason: str) -> str:
    if outcome_reason in ERROR_OUTCOMES:
        return TRIAL_STATUS_ERROR
    return TRIAL_STATUS_FINISHED


def classify_error_type(outcome_reason: str, error_json: dict[str, Any] | None) -> str | None:
    # Honor an explicit stored error type before deriving one from lower-level fields.
    payload = dict(error_json or {})
    explicit = payload.get("error_type")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    # Normalize the shared classifier inputs used across outcome-specific branches.
    reason = payload.get("reason")
    if not isinstance(reason, str):
        reason = None
    finish_reason = payload.get("finish_reason")
    native_finish_reason = payload.get("native_finish_reason")
    reached_length_limit = (
        (isinstance(finish_reason, str) and finish_reason == "length")
        or (isinstance(native_finish_reason, str) and native_finish_reason == "length")
    )

    # Classify generation failures by whether the issue came from output shape or provider behavior.
    if outcome_reason == OUTCOME_GENERATION_FAILED:
        if reason in {"candidate_materialization_failed", "generation_assertion_failed"} and reached_length_limit:
            return "generation_output_truncated"
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

    # Map execution-time failures to more precise runner or evaluation buckets.
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

    # Split stale outcomes by whether dispatching or the active runner stopped making progress.
    if outcome_reason == OUTCOME_STALE:
        if reason == "dispatch_deadline_expired":
            return "dispatch_stale"
        if reason == "heartbeat_stale":
            return "runner_stale"
        return "stale"
    return None


def prepare_error_payload(outcome_reason: str, error_json: dict[str, Any] | None) -> dict[str, Any] | None:
    # Attach the derived error type while preserving any original error fields.
    payload = dict(error_json or {})
    error_type = classify_error_type(outcome_reason, payload)
    if error_type:
        payload["error_type"] = error_type
    return payload or None


# ---- storage.py ----

from contextlib import contextmanager
from datetime import timedelta
import json
from typing import Any, Iterable

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from sigmaevolve.core import compute_script_hash, normalize_source
from sigmaevolve.core import (
    ACTIVE_STATUSES,
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
from sigmaevolve.core import compute_score


def _row_to_dataset(row: sa.Row[Any]) -> DatasetRecord:
    return DatasetRecord(
        dataset_id=row.dataset_id,
        manifest_path=row.manifest_path,
        created_at=row.created_at,
    )


def _copy_json_dict(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _copy_optional_json_dict(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    return dict(value)


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
        provenance_json=_copy_json_dict(row.provenance_json),
        status=row.status,
        outcome_reason=row.outcome_reason,
        dispatch_token=row.dispatch_token,
        dispatch_deadline_at=row.dispatch_deadline_at,
        runner_id=row.runner_id,
        heartbeat_at=row.heartbeat_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        metrics_json=_copy_optional_json_dict(row.metrics_json),
        score=float(row.score or 0.0),
        error_json=_copy_optional_json_dict(row.error_json),
        dispatch_attempts=int(row.dispatch_attempts),
        created_at=row.created_at,
    )


def _trial_summary_sort_key(summary: TrialSummary) -> tuple[float, float, float]:
    # Rank by accuracy first, then faster time-to-best, then final score.
    metrics = summary.metrics_json or {}
    accuracy = float(metrics.get("accuracy") or 0.0)
    time_to_best = metrics.get("time_to_best_eval_sec")
    if time_to_best is None:
        time_to_best = float("inf")

    return (-accuracy, float(time_to_best), -summary.score)


def _row_to_trial_summary(row: sa.Row[Any]) -> TrialSummary:
    return TrialSummary(
        trial_id=row.trial_id,
        score=float(row.score or 0.0),
        metrics_json=_copy_optional_json_dict(row.metrics_json),
        source=row.source,
        provenance_json=_copy_json_dict(row.provenance_json),
        outcome_reason=row.outcome_reason,
        error_json=_copy_optional_json_dict(row.error_json),
    )


class SQLAlchemyRepository:
    def __init__(self, database_url: str) -> None:
        # Normalize the database URL before configuring the SQLAlchemy engine.
        database_url = normalize_database_url(database_url)
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        engine_kwargs: dict[str, Any] = {"future": True, "connect_args": connect_args}
        if not database_url.startswith("sqlite"):
            engine_kwargs["pool_pre_ping"] = True

        # Create the engine and ensure the schema exists for the current process.
        self.engine: Engine = sa.create_engine(database_url, **engine_kwargs)
        metadata.create_all(self.engine)

    @contextmanager
    def transaction(self) -> Iterable[Connection]:
        # Use BEGIN IMMEDIATE on SQLite so concurrent workers serialize writes cleanly.
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
            # Use the dialect-native transaction helper everywhere else.
            with self.engine.begin() as conn:
                yield conn

    def _notify_dashboard(self, conn: Connection, track_id: str, reason: str) -> None:
        # Skip dashboard notifications when the backend does not support LISTEN/NOTIFY.
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
        # Start from the target trial id and extend the filter for state-sensitive updates.
        conditions = [trials_table.c.trial_id == trial_id]
        if where:
            conditions.extend(where)
        result = conn.execute(sa.update(trials_table).where(sa.and_(*conditions)).values(**values))

        # Notify the dashboard only when the trial row was actually changed.
        if notify and result.rowcount:
            track_id = conn.execute(
                sa.select(trials_table.c.track_id).where(trials_table.c.trial_id == trial_id)
            ).scalar_one_or_none()
            if track_id is not None:
                self._notify_dashboard(conn, track_id=track_id, reason="trial_changed")
        return int(result.rowcount)

    def _load_trial_row(
        self,
        conn: Connection,
        *,
        trial_id: str,
        columns: tuple[Any, ...] | None = None,
    ) -> sa.Row[Any] | None:
        selected_columns = columns or (trials_table,)
        stmt = sa.select(*selected_columns).where(trials_table.c.trial_id == trial_id)
        return conn.execute(stmt).fetchone()

    def _mutate_trial(
        self,
        conn: Connection,
        *,
        trial_id: str,
        values: dict[str, Any],
        where: list[Any] | None = None,
        row_columns: tuple[Any, ...] | None = None,
        notify: bool = True,
    ) -> sa.Row[Any] | None:
        # Apply the state-sensitive update before reloading the requested row shape.
        updated = self._update_trial_state(
            conn,
            trial_id=trial_id,
            values=values,
            where=where,
            notify=notify,
        )
        if updated != 1:
            return None
        return self._load_trial_row(conn, trial_id=trial_id, columns=row_columns)

    def _set_terminal_trial_state(
        self,
        conn: Connection,
        *,
        trial_id: str,
        outcome_reason: str,
        score: float,
        error_json: dict[str, Any] | None,
        metrics_json: dict[str, Any] | None = None,
        finished_at: Any | None = None,
        extra_values: dict[str, Any] | None = None,
    ) -> None:
        # Persist the normalized terminal payload used by finalize and stale sweeps.
        values = {
            "status": status_for_outcome_reason(outcome_reason),
            "outcome_reason": outcome_reason,
            "finished_at": finished_at if finished_at is not None else now_utc(),
            "score": score,
            "error_json": prepare_error_payload(outcome_reason, error_json),
        }
        if metrics_json is not None or outcome_reason in TERMINAL_OUTCOMES:
            values["metrics_json"] = metrics_json
        if extra_values:
            values.update(extra_values)
        conn.execute(
            sa.update(trials_table)
            .where(trials_table.c.trial_id == trial_id)
            .values(**values)
        )

    def register_dataset(self, dataset_id: str, manifest_path: str | None) -> DatasetRecord:
        # Upsert the dataset manifest path while preserving the latest registration time.
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

            # Reload the canonical row shape before returning the record.
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
        # Create a fresh track id and persist the track metadata in one transaction.
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
            row = conn.execute(
                sa.select(tracks_table).where(tracks_table.c.track_id == track_id)
            ).fetchone()
        return _row_to_track(row) if row else None

    def create_queued_trial_if_absent(
        self,
        track_id: str,
        source: str,
        provenance_json: dict[str, Any],
    ) -> tuple[TrialRecord | None, bool]:
        # Normalize provenance and source before checking for duplicate scripts.
        validated_provenance = validate_trial_provenance(provenance_json)
        normalized_source = normalize_source(source)
        script_hash = compute_script_hash(normalized_source)
        created_at = now_utc()
        trial_id = make_id("trial")
        with self.transaction() as conn:
            # Reuse the existing trial when the track already has this script hash.
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

            # Insert the queued trial in its initial unclaimed state.
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
        # Restrict generation-attempt rows to duplicate and generation-failure outcomes.
        if outcome_reason not in {OUTCOME_DUPLICATE, OUTCOME_GENERATION_FAILED}:
            raise ValueError(f"Unsupported generation attempt outcome_reason: {outcome_reason}")

        # Materialize the diagnostic source and terminal row payload once up front.
        validated_provenance = validate_trial_provenance(provenance_json)
        trial_id = make_id("trial")
        source = build_generation_attempt_source(trial_id, outcome_reason)
        script_hash = compute_script_hash(source)
        created_at = now_utc()
        with self.transaction() as conn:
            # Persist the diagnostic row as a terminal non-runnable trial record.
            conn.execute(
                sa.insert(trials_table).values(
                    trial_id=trial_id,
                    track_id=track_id,
                    source=source,
                    script_hash=script_hash,
                    provenance_json=validated_provenance,
                    status=status_for_outcome_reason(outcome_reason),
                    outcome_reason=outcome_reason,
                    dispatch_token=None,
                    dispatch_deadline_at=None,
                    runner_id=None,
                    heartbeat_at=created_at,
                    started_at=None,
                    finished_at=created_at,
                    metrics_json=None,
                    score=0.0,
                    error_json=prepare_error_payload(outcome_reason, error_json),
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
        self._record_trial_provenance_section(trial_id, "launcher", launcher_metadata)

    def record_trial_wandb_metadata(self, trial_id: str, wandb_metadata: dict[str, Any]) -> None:
        self._record_trial_provenance_section(trial_id, "wandb", wandb_metadata)

    def _record_trial_provenance_section(self, trial_id: str, section: str, payload: dict[str, Any]) -> None:
        # Merge the section payload into the stored provenance document.
        payload = dict(payload)
        with self.transaction() as conn:
            row = conn.execute(
                sa.select(trials_table.c.track_id, trials_table.c.provenance_json).where(
                    trials_table.c.trial_id == trial_id
                )
            ).fetchone()
            if row is None:
                raise KeyError(f"Trial not found: {trial_id}")

            # Preserve any existing section fields when extending the payload.
            provenance_json = dict(row.provenance_json or {})
            updated_provenance_json = dict(provenance_json)
            existing_section = updated_provenance_json.get(section)
            if isinstance(existing_section, dict):
                merged_section = dict(existing_section)
                merged_section.update(payload)
                updated_provenance_json[section] = merged_section
            else:
                updated_provenance_json[section] = payload
            if updated_provenance_json == provenance_json:
                return

            # Write the merged provenance back only when the payload changed.
            conn.execute(
                sa.update(trials_table)
                .where(trials_table.c.trial_id == trial_id)
                .values(provenance_json=updated_provenance_json)
            )
            self._notify_dashboard(conn, track_id=row.track_id, reason="trial_changed")

    def list_trials(self, track_id: str, statuses: set[str] | None = None) -> list[TrialRecord]:
        # Apply the optional status filter before loading trials in creation order.
        stmt = (
            sa.select(trials_table)
            .where(trials_table.c.track_id == track_id)
            .order_by(trials_table.c.created_at)
        )
        if statuses:
            stmt = stmt.where(trials_table.c.status.in_(sorted(statuses)))
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_to_trial(row) for row in rows]

    def sample_trial_context(self, track_id: str, limit: int, candidate_kind: str | None = None) -> list[TrialSummary]:
        # Start from recent finished successful trials that include metrics payloads.
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

        # Apply candidate-kind filtering in Python after the rows are materialized.
        summaries: list[TrialSummary] = []
        for row in rows:
            row_candidate_kind = _copy_json_dict(row.provenance_json).get("candidate_kind")
            if candidate_kind is not None and row_candidate_kind != candidate_kind:
                continue
            summaries.append(_row_to_trial_summary(row))
        return sorted(summaries, key=_trial_summary_sort_key)[:limit]

    def list_recent_trial_summaries(
        self,
        track_id: str,
        *,
        outcome_reasons: set[str] | None = None,
        require_metrics: bool | None = None,
        limit: int = 5,
    ) -> list[TrialSummary]:
        # Start from terminal trials on the requested track and add optional filters.
        terminal_statuses = sorted(TERMINAL_STATUSES)
        stmt = sa.select(trials_table).where(
            sa.and_(
                trials_table.c.track_id == track_id,
                trials_table.c.status.in_(terminal_statuses),
            )
        )
        if outcome_reasons:
            stmt = stmt.where(trials_table.c.outcome_reason.in_(sorted(outcome_reasons)))
        if require_metrics is True:
            stmt = stmt.where(trials_table.c.metrics_json.is_not(None))
        elif require_metrics is False:
            stmt = stmt.where(trials_table.c.metrics_json.is_(None))
        stmt = stmt.order_by(trials_table.c.finished_at.desc(), trials_table.c.created_at.desc()).limit(limit)

        # Return lightweight summaries for the filtered result set.
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_to_trial_summary(row) for row in rows]

    def count_trials(self, track_id: str, statuses: set[str] | None = None) -> int:
        stmt = (
            sa.select(sa.func.count())
            .select_from(trials_table)
            .where(trials_table.c.track_id == track_id)
        )
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
        # Reserve at most the remaining dispatch capacity for this track.
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

            # Claim queued trials one by one so each reservation gets a fresh token.
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

                # Move the trial into dispatching state with a deadline and token.
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

            # Emit one dashboard notification for the whole reserved batch.
            if reserved:
                self._notify_dashboard(conn, track_id=track_id, reason="trial_changed")
        return reserved

    def claim_trial(self, trial_id: str, dispatch_token: str, runner_id: str) -> TrialRecord | None:
        # Claim only trials that are still dispatching with the expected token.
        with self.transaction() as conn:
            now = now_utc()
            row = self._mutate_trial(
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
            if row is None:
                return None
        return _row_to_trial(row)

    def heartbeat_trial(self, trial_id: str, runner_id: str, meta: dict[str, Any] | None = None) -> None:
        # Refresh the heartbeat and persist only error-bearing metadata payloads.
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
                .values(heartbeat_at=now_utc(), error_json=payload if has_error_signal(payload) else None)
            )

    def update_active_trial_metrics(self, trial_id: str, runner_id: str, metrics: dict[str, Any]) -> None:
        # Skip writes when the active trial already has the same metrics payload.
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
            existing = _copy_optional_json_dict(row.metrics_json)
            if existing == payload:
                return

            # Persist the updated metrics and notify the dashboard once.
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
        # Reject outcome reasons that do not map to terminal storage states.
        if outcome_reason not in TERMINAL_OUTCOMES:
            raise ValueError(f"Unsupported outcome_reason: {outcome_reason}")
        if metrics is None:
            score = 0.0

        # Drop empty error payloads for successful trials so dashboards stay clean.
        persisted_error_info = dict(error_info) if error_info else None
        if outcome_reason in SUCCESS_OUTCOMES and not has_error_signal(persisted_error_info):
            persisted_error_info = None

        with self.transaction() as conn:
            # Require the expected active runner state when finalizing from a worker.
            requires_runner_state = runner_id is not None
            state_filters: list[Any] = []
            if requires_runner_state:
                state_filters.append(trials_table.c.runner_id == runner_id)
                state_filters.append(trials_table.c.status == TRIAL_STATUS_ACTIVE)

            # Clear dispatch state and persist terminal metrics in one update.
            now = now_utc()
            row = self._mutate_trial(
                conn,
                trial_id=trial_id,
                where=state_filters,
                values={
                    "finished_at": now,
                    "dispatch_token": None,
                    "dispatch_deadline_at": None,
                    "heartbeat_at": now,
                    "status": status_for_outcome_reason(outcome_reason),
                    "outcome_reason": outcome_reason,
                    "metrics_json": metrics,
                    "score": score,
                    "error_json": prepare_error_payload(outcome_reason, persisted_error_info),
                },
            )
            if row is None:
                return

    def sweep_expired_dispatches(self, track_id: str, max_dispatch_retries: int) -> tuple[list[str], list[str]]:
        # Requeue expired dispatches until the retry budget is exhausted.
        requeued: list[str] = []
        stale: list[str] = []
        now = now_utc()

        with self.transaction() as conn:
            rows = conn.execute(
                sa.select(trials_table).where(
                    sa.and_(
                        trials_table.c.track_id == track_id,
                        trials_table.c.status == TRIAL_STATUS_DISPATCHING,
                        trials_table.c.dispatch_deadline_at.is_not(None),
                        trials_table.c.dispatch_deadline_at < now,
                    )
                )
            ).fetchall()
            for row in rows:
                can_retry = int(row.dispatch_attempts) < max_dispatch_retries
                if can_retry:
                    # Return retryable trials to the queued state.
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
                    # Mark exhausted dispatch attempts as stale terminal failures.
                    self._set_terminal_trial_state(
                        conn,
                        trial_id=row.trial_id,
                        outcome_reason=OUTCOME_STALE,
                        score=0.0,
                        error_json={"reason": "dispatch_deadline_expired"},
                        finished_at=now,
                        extra_values={
                            "dispatch_token": None,
                            "dispatch_deadline_at": None,
                        },
                    )
                    stale.append(row.trial_id)
            if requeued or stale:
                self._notify_dashboard(conn, track_id=track_id, reason="trial_changed")
        return requeued, stale

    def sweep_stale_active_trials(self, track_id: str, stale_ttl_sec: int) -> list[str]:
        # Find active trials whose heartbeat or start time has gone stale.
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
                # Convert stale active trials into terminal stale failures.
                self._set_terminal_trial_state(
                    conn,
                    trial_id=row.trial_id,
                    outcome_reason=OUTCOME_STALE,
                    score=0.0,
                    error_json={"reason": "heartbeat_stale"},
                )
                stale.append(row.trial_id)
            if stale:
                self._notify_dashboard(conn, track_id=track_id, reason="trial_changed")
        return stale

    def rescore(self, track_id: str | None, scorer_config: dict[str, Any]) -> MigrationResult:
        # Recompute scores for all terminal trials in the requested scope.
        updated = 0
        touched_track_ids: set[str] = set()
        with self.transaction() as conn:
            stmt = sa.select(trials_table).where(trials_table.c.status.in_(sorted(TERMINAL_STATUSES)))
            if track_id is not None:
                stmt = stmt.where(trials_table.c.track_id == track_id)
            rows = conn.execute(stmt).fetchall()
            for row in rows:
                # Persist the rescored value and remember which tracks changed.
                new_score = compute_score(row.metrics_json, row.outcome_reason, scorer_config)
                conn.execute(
                    sa.update(trials_table)
                    .where(trials_table.c.trial_id == row.trial_id)
                    .values(score=new_score)
                )
                updated += 1
                touched_track_ids.add(row.track_id)

            # Emit one dashboard notification per changed track.
            for touched_track_id in sorted(touched_track_ids):
                self._notify_dashboard(conn, track_id=touched_track_id, reason="trial_changed")
        return MigrationResult(updated_trials=updated, scorer_config=dict(scorer_config))
