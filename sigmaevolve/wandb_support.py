from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any


WANDB_ENV_KEYS = (
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_BASE_URL",
    "SIGMAEVOLVE_WANDB_API_KEY",
    "SIGMAEVOLVE_WANDB_PROJECT",
    "SIGMAEVOLVE_WANDB_ENTITY",
    "SIGMAEVOLVE_WANDB_BASE_URL",
)
_DISALLOWED_WANDB_MODES = {"disabled", "dryrun", "offline"}


def collect_wandb_env() -> dict[str, str]:
    return {
        key: value
        for key in WANDB_ENV_KEYS
        if isinstance((value := os.environ.get(key)), str) and value.strip()
    }


def apply_wandb_env(overrides: dict[str, str] | None) -> None:
    if not overrides:
        return
    for key in WANDB_ENV_KEYS:
        value = overrides.get(key)
        if isinstance(value, str) and value.strip():
            os.environ[key] = value


def _import_wandb():
    try:
        return importlib.import_module("wandb")
    except ImportError as exc:
        raise RuntimeError("Weights & Biases support requires the 'wandb' package.") from exc


def _env_first(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


@dataclass(frozen=True)
class WandbSettings:
    api_key: str
    project: str
    entity: str | None
    base_url: str | None


def resolve_wandb_settings() -> WandbSettings:
    mode = _env_first("WANDB_MODE")
    if mode is not None and mode.lower() in _DISALLOWED_WANDB_MODES:
        raise RuntimeError("WANDB_MODE must allow remote sync; offline and disabled modes are not supported.")

    api_key = _env_first("SIGMAEVOLVE_WANDB_API_KEY", "WANDB_API_KEY")
    if api_key is None:
        raise RuntimeError("WANDB_API_KEY is required to log SigmaEvolve runs to Weights & Biases.")

    return WandbSettings(
        api_key=api_key,
        project=_env_first("SIGMAEVOLVE_WANDB_PROJECT", "WANDB_PROJECT", default="sigmaevolve") or "sigmaevolve",
        entity=_env_first("SIGMAEVOLVE_WANDB_ENTITY", "WANDB_ENTITY"),
        base_url=_env_first("SIGMAEVOLVE_WANDB_BASE_URL", "WANDB_BASE_URL"),
    )


class WandbRunLogger:
    def __init__(
        self,
        *,
        repository,
        trial,
        track,
        manifest,
        runner_id: str,
    ) -> None:
        self.repository = repository
        self.trial = trial
        self.track = track
        self.manifest = manifest
        self.runner_id = runner_id
        self.step = 0

        wandb = _import_wandb()
        settings = resolve_wandb_settings()
        if settings.base_url:
            os.environ["WANDB_BASE_URL"] = settings.base_url
        wandb.login(key=settings.api_key, relogin=True)

        run = wandb.init(
            project=settings.project,
            entity=settings.entity,
            job_type="trial",
            name=f"{track.track_id}:{trial.trial_id}",
            config={
                "sigmaevolve": {
                    "trial_id": trial.trial_id,
                    "track_id": track.track_id,
                    "track_name": track.name,
                    "dataset_id": track.dataset_id,
                    "runner_id": runner_id,
                    "script_hash": trial.script_hash,
                    "dispatch_attempts": trial.dispatch_attempts,
                    "policy": dict(track.policy_json),
                    "provenance": dict(trial.provenance_json),
                    "dataset_metadata": dict(manifest.metadata),
                }
            },
            tags=[
                "sigmaevolve",
                f"track:{track.track_id}",
                f"dataset:{track.dataset_id}",
            ],
        )
        self.run = run
        self.repository.record_trial_wandb_metadata(
            trial.trial_id,
            {
                "project": settings.project,
                "entity": getattr(run, "entity", None) or settings.entity,
                "run_id": getattr(run, "id", None),
                "run_name": getattr(run, "name", None),
                "run_url": getattr(run, "url", None),
            },
        )

    def log_metrics(self, metrics: dict[str, Any], *, state: str) -> None:
        payload = dict(metrics or {})
        payload["trial_state"] = state
        self.step += 1
        self.run.log(payload, step=self.step)

    def finish(
        self,
        *,
        outcome_reason: str,
        metrics: dict[str, Any] | None,
        score: float,
        error_info: dict[str, Any] | None,
    ) -> None:
        payload = {
            "trial_state": "terminal",
            "outcome_reason": outcome_reason,
            "score": float(score),
        }
        if metrics:
            payload.update(dict(metrics))
        self.step += 1
        self.run.log(payload, step=self.step)

        self.run.summary["trial_id"] = self.trial.trial_id
        self.run.summary["track_id"] = self.track.track_id
        self.run.summary["dataset_id"] = self.track.dataset_id
        self.run.summary["runner_id"] = self.runner_id
        self.run.summary["outcome_reason"] = outcome_reason
        self.run.summary["score"] = float(score)
        if metrics:
            for key, value in metrics.items():
                self.run.summary[key] = value
        if error_info:
            reason = error_info.get("reason")
            detail = error_info.get("detail")
            if isinstance(reason, str) and reason:
                self.run.summary["error_reason"] = reason
            if isinstance(detail, str) and detail:
                self.run.summary["error_detail"] = detail

        exit_code = 0 if outcome_reason in {"succeeded", "timeout"} else 1
        self.run.finish(exit_code=exit_code)
