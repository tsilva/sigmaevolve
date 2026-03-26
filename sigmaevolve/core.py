from __future__ import annotations


# ---- hashing.py ----

import hashlib


def normalize_source(source: str) -> str:
    # Round-trip through UTF-8 so invalid text fails fast at the boundary.
    normalized = source.encode("utf-8", errors="strict").decode("utf-8")

    # Canonicalize all newline variants to the repository's single-line-ending form.
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Guarantee one trailing newline so hashes and persisted sources stay stable.
    normalized = normalized.rstrip("\n") + "\n"
    return normalized


def compute_script_hash(source: str) -> str:
    normalized = normalize_source(source)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---- scoring.py ----

from typing import Any


def compute_classification_metrics(
    predictions: list[int],
    labels: list[int],
) -> dict[str, Any]:
    # Validate the scoring inputs before computing aggregates.
    num_examples = len(labels)
    if len(predictions) != num_examples:
        raise ValueError("Predictions and labels must have the same length.")
    if num_examples == 0:
        raise ValueError("Cannot score an empty validation split.")

    # Derive the aggregate metrics from the aligned predictions and labels.
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    accuracy = correct / num_examples

    # Return the metrics payload in a reviewable shape.
    return {
        "accuracy": accuracy,
        "correct": correct,
        "num_examples": num_examples,
    }


def compute_score(
    metrics: dict[str, Any] | None,
    scorer_config: dict[str, Any],
) -> float:
    # Fall back to zero when the trial never produced metrics.
    has_metrics = bool(metrics)
    if not has_metrics:
        return 0.0

    # Resolve the configured metric before converting it to a float score.
    primary_metric = scorer_config.get("primary_metric", "accuracy")
    score_value = metrics.get(primary_metric)
    if score_value is None:
        raise ValueError(f"Primary metric {primary_metric!r} not present in metrics.")

    return float(score_value)


# ---- runtime_config.py ----

DEFAULT_TRIAL_HARD_TIMEOUT_SEC = 60 * 60


# ---- models.py ----

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


TRIAL_STATUS_QUEUED = "queued"
TRIAL_STATUS_DISPATCHING = "dispatching"
TRIAL_STATUS_ACTIVE = "active"
TRIAL_STATUS_FINISHED = "finished"
TRIAL_STATUS_ERROR = "error"

OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_DUPLICATE = "duplicate"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_CRASHED = "crashed"
OUTCOME_EVAL_FAILED = "eval_failed"
OUTCOME_STALE = "stale"
OUTCOME_GENERATION_FAILED = "generation_failed"

SUCCESS_OUTCOMES = {OUTCOME_SUCCEEDED, OUTCOME_TIMEOUT}
ACTIVE_STATUSES = {TRIAL_STATUS_DISPATCHING, TRIAL_STATUS_ACTIVE}
TERMINAL_STATUSES = {TRIAL_STATUS_FINISHED, TRIAL_STATUS_ERROR}
ERROR_OUTCOMES = {OUTCOME_CRASHED, OUTCOME_EVAL_FAILED, OUTCOME_STALE, OUTCOME_GENERATION_FAILED}
TERMINAL_OUTCOMES = {
    OUTCOME_SUCCEEDED,
    OUTCOME_DUPLICATE,
    OUTCOME_TIMEOUT,
    OUTCOME_CRASHED,
    OUTCOME_EVAL_FAILED,
    OUTCOME_STALE,
    OUTCOME_GENERATION_FAILED,
}

CANDIDATE_KIND_STRATEGY_V1 = "strategy_v1"
DEFAULT_GENERATION_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_GENERATION_SELECTION = "weighted_random"
DEFAULT_GENERATION_MODEL_POOL = [
    {
        "model": "x-ai/grok-4.1-fast",
        "temperature": 0.2,
        "max_tokens": 2500,
        "retry_count": 2,
        "probability": 0.5436,
    },
    {
        "model": "google/gemini-3.1-flash-lite-preview",
        "temperature": 0.2,
        "max_tokens": 2500,
        "retry_count": 2,
        "probability": 0.2446,
    },
    {
        "model": "moonshotai/kimi-k2.5",
        "temperature": 0.2,
        "max_tokens": 2500,
        "retry_count": 2,
        "probability": 0.1578,
    },
    {
        "model": "google/gemini-3.1-pro-preview",
        "temperature": 0.2,
        "max_tokens": 2500,
        "retry_count": 2,
        "probability": 0.0306,
    },
    {
        "model": "anthropic/claude-sonnet-4.6",
        "temperature": 0.2,
        "max_tokens": 2500,
        "retry_count": 2,
        "probability": 0.0233,
    },
]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    # Merge nested dictionaries recursively while replacing non-dict values.
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(value, dict) and isinstance(base_value, dict):
            merged[key] = _deep_merge_dict(base_value, value)
        else:
            merged[key] = value
    return merged


def _reject_removed_policy_fields(raw: dict[str, Any]) -> None:
    # Fail fast when callers still send policy knobs that are no longer supported.
    if "modal_gpu_preferences" in raw:
        raise ValueError("Track policy modal_gpu_preferences is no longer supported.")

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
        # Return a plain dict so manifests can be serialized directly.
        payload = {
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
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetManifest":
        # Rebuild the strongly typed manifest from a persisted dict payload.
        split_sizes = dict(raw["split_sizes"])
        checksums = dict(raw["checksums"])
        metadata = dict(raw.get("metadata", {}))
        return cls(
            dataset_id=raw["dataset_id"],
            root_dir=raw["root_dir"],
            train_split_path=raw["train_split_path"],
            validation_split_path=raw["validation_split_path"],
            validation_labels_path=raw["validation_labels_path"],
            test_split_path=raw["test_split_path"],
            test_labels_path=raw["test_labels_path"],
            split_sizes=split_sizes,
            checksums=checksums,
            fingerprint=raw["fingerprint"],
            metadata=metadata,
        )


@dataclass(frozen=True)
class TrackPolicy:
    epochs: int = 5
    dispatch_ttl_sec: int = 300
    heartbeat_interval_sec: int = 15
    stale_ttl_sec: int = 120
    max_dispatch_retries: int = 2
    scorer_settings: dict[str, Any] = field(default_factory=lambda: {"primary_metric": "accuracy"})
    sampling_settings: dict[str, Any] = field(default_factory=lambda: {"seed": 0})
    generation_backend: dict[str, Any] = field(
        default_factory=lambda: {
            "backend": "openrouter",
            "selection": DEFAULT_GENERATION_SELECTION,
            "seed": 0,
            "model_pool": [dict(entry) for entry in DEFAULT_GENERATION_MODEL_POOL],
        }
    )

    def to_dict(self) -> dict[str, Any]:
        # Copy mutable policy containers before returning the serialized shape.
        payload = {
            "epochs": self.epochs,
            "dispatch_ttl_sec": self.dispatch_ttl_sec,
            "heartbeat_interval_sec": self.heartbeat_interval_sec,
            "stale_ttl_sec": self.stale_ttl_sec,
            "max_dispatch_retries": self.max_dispatch_retries,
            "scorer_settings": dict(self.scorer_settings),
            "sampling_settings": dict(self.sampling_settings),
            "generation_backend": dict(self.generation_backend),
        }
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrackPolicy":
        _reject_removed_policy_fields(raw or {})

        # Merge overrides onto the default policy before coercing field types.
        base = cls()
        merged = _deep_merge_dict(base.to_dict(), raw or {})

        # Rebuild the dataclass with normalized scalar and nested policy fields.
        return cls(
            epochs=int(merged["epochs"]),
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
        # Require terminal finished state, a success outcome, and metrics payload.
        is_finished = self.status == TRIAL_STATUS_FINISHED
        has_success_outcome = self.outcome_reason in SUCCESS_OUTCOMES
        has_metrics = self.metrics_json is not None

        return is_finished and has_success_outcome and has_metrics


@dataclass(frozen=True)
class TrialSummary:
    trial_id: str
    score: float
    metrics_json: dict[str, Any] | None
    source: str
    provenance_json: dict[str, Any]
    outcome_reason: str | None = None
    error_json: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationResult:
    source: str | None
    provenance_json: dict[str, Any]
    error_info: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        has_error = self.error_info is not None
        has_source = self.source is not None

        return not has_error and has_source


@dataclass(frozen=True)
class ReconcileResult:
    generated_trial_ids: list[str] = field(default_factory=list)
    launched_trial_ids: list[str] = field(default_factory=list)
    duplicate_hashes: list[str] = field(default_factory=list)
    duplicate_trial_ids: list[str] = field(default_factory=list)
    failed_generation_trial_ids: list[str] = field(default_factory=list)
    requeued_trial_ids: list[str] = field(default_factory=list)
    stale_trial_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
