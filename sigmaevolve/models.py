from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


TRIAL_STATUS_QUEUED = "queued"
TRIAL_STATUS_DISPATCHING = "dispatching"
TRIAL_STATUS_ACTIVE = "active"
TRIAL_STATUS_FINISHED = "finished"

OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_DUPLICATE = "duplicate"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_CRASHED = "crashed"
OUTCOME_EVAL_FAILED = "eval_failed"
OUTCOME_STALE = "stale"

SUCCESS_OUTCOMES = {OUTCOME_SUCCEEDED, OUTCOME_TIMEOUT}
ACTIVE_STATUSES = {TRIAL_STATUS_DISPATCHING, TRIAL_STATUS_ACTIVE}
TERMINAL_OUTCOMES = {
    OUTCOME_SUCCEEDED,
    OUTCOME_DUPLICATE,
    OUTCOME_TIMEOUT,
    OUTCOME_CRASHED,
    OUTCOME_EVAL_FAILED,
    OUTCOME_STALE,
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    manifest_path: str | None
    created_at: datetime


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    root_dir: str
    train_split_path: str
    validation_split_path: str
    validation_labels_path: str
    test_split_path: str
    test_labels_path: str
    split_sizes: dict[str, int]
    checksums: dict[str, str]
    fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "root_dir": self.root_dir,
            "train_split_path": self.train_split_path,
            "validation_split_path": self.validation_split_path,
            "validation_labels_path": self.validation_labels_path,
            "test_split_path": self.test_split_path,
            "test_labels_path": self.test_labels_path,
            "split_sizes": dict(self.split_sizes),
            "checksums": dict(self.checksums),
            "fingerprint": self.fingerprint,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetManifest":
        return cls(
            dataset_id=raw["dataset_id"],
            root_dir=raw["root_dir"],
            train_split_path=raw["train_split_path"],
            validation_split_path=raw["validation_split_path"],
            validation_labels_path=raw["validation_labels_path"],
            test_split_path=raw["test_split_path"],
            test_labels_path=raw["test_labels_path"],
            split_sizes=dict(raw["split_sizes"]),
            checksums=dict(raw["checksums"]),
            fingerprint=raw["fingerprint"],
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class TrackPolicy:
    budget_sec: int = 60
    max_eval_gap_sec: int = 15
    max_parallelism: int = 1
    ready_queue_threshold: int = 1
    dispatch_ttl_sec: int = 300
    heartbeat_interval_sec: int = 15
    stale_ttl_sec: int = 120
    max_dispatch_retries: int = 2
    scorer_settings: dict[str, Any] = field(default_factory=lambda: {"primary_metric": "accuracy"})
    sampling_settings: dict[str, Any] = field(
        default_factory=lambda: {"strategy": "top_then_random", "top_k": 3, "seed": 0}
    )
    generation_backend: dict[str, Any] = field(
        default_factory=lambda: {
            "backend": "openrouter",
            "selection": "round_robin",
            "model_pool": [
                {
                    "model": "openai/gpt-4o-mini",
                    "temperature": 0.2,
                    "max_tokens": 2500,
                    "retry_count": 2,
                }
            ],
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_sec": self.budget_sec,
            "max_eval_gap_sec": self.max_eval_gap_sec,
            "max_parallelism": self.max_parallelism,
            "ready_queue_threshold": self.ready_queue_threshold,
            "dispatch_ttl_sec": self.dispatch_ttl_sec,
            "heartbeat_interval_sec": self.heartbeat_interval_sec,
            "stale_ttl_sec": self.stale_ttl_sec,
            "max_dispatch_retries": self.max_dispatch_retries,
            "scorer_settings": dict(self.scorer_settings),
            "sampling_settings": dict(self.sampling_settings),
            "generation_backend": dict(self.generation_backend),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrackPolicy":
        base = cls()
        merged = base.to_dict()
        merged = _deep_merge_dict(merged, raw or {})
        return cls(
            budget_sec=int(merged["budget_sec"]),
            max_eval_gap_sec=int(merged["max_eval_gap_sec"]),
            max_parallelism=int(merged["max_parallelism"]),
            ready_queue_threshold=int(merged["ready_queue_threshold"]),
            dispatch_ttl_sec=int(merged["dispatch_ttl_sec"]),
            heartbeat_interval_sec=int(merged["heartbeat_interval_sec"]),
            stale_ttl_sec=int(merged["stale_ttl_sec"]),
            max_dispatch_retries=int(merged["max_dispatch_retries"]),
            scorer_settings=dict(merged["scorer_settings"]),
            sampling_settings=dict(merged["sampling_settings"]),
            generation_backend=dict(merged["generation_backend"]),
        )


@dataclass(frozen=True)
class TrackRecord:
    track_id: str
    name: str | None
    dataset_id: str
    policy_json: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    track_id: str
    source: str
    script_hash: str
    provenance_json: dict[str, Any]
    status: str
    outcome_reason: str | None
    dispatch_token: str | None
    dispatch_deadline_at: datetime | None
    runner_id: str | None
    heartbeat_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    metrics_json: dict[str, Any] | None
    score: float
    error_json: dict[str, Any] | None
    dispatch_attempts: int
    created_at: datetime

    @property
    def succeeded(self) -> bool:
        return self.status == TRIAL_STATUS_FINISHED and self.outcome_reason in SUCCESS_OUTCOMES and self.metrics_json is not None


@dataclass(frozen=True)
class TrialSummary:
    trial_id: str
    score: float
    metrics_json: dict[str, Any] | None
    source: str
    provenance_json: dict[str, Any]
    outcome_reason: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    source: str
    provenance_json: dict[str, Any]


@dataclass(frozen=True)
class ReconcileResult:
    generated_trial_ids: list[str] = field(default_factory=list)
    launched_trial_ids: list[str] = field(default_factory=list)
    duplicate_hashes: list[str] = field(default_factory=list)
    requeued_trial_ids: list[str] = field(default_factory=list)
    stale_trial_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MigrationResult:
    updated_trials: int
    scorer_config: dict[str, Any]
