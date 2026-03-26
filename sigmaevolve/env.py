from __future__ import annotations

import os
from dataclasses import dataclass

from sigmaevolve.core import load_env_file
from sigmaevolve.modal import (
    DEFAULT_MODAL_APP_NAME,
    DEFAULT_MODAL_DATASET_MOUNT,
    DEFAULT_MODAL_DATASET_VOLUME,
    DEFAULT_MODAL_FUNCTION_NAME,
)


DEFAULT_DATASET_ROOT = "./artifacts/datasets"


@dataclass(frozen=True)
class RuntimeConfig:
    database_url: str | None
    dataset_root: str
    openrouter_api_key: str | None
    modal_app_name: str
    modal_function_name: str
    modal_dataset_volume: str
    modal_dataset_mount: str
    modal_environment_name: str | None


def resolve_runtime_config() -> RuntimeConfig:
    # Resolve each runtime setting from SigmaEvolve-scoped env vars first.
    return RuntimeConfig(
        database_url=_resolve_optional_env("SIGMAEVOLVE_DATABASE_URL", "DATABASE_URL"),
        dataset_root=_resolve_required_default(DEFAULT_DATASET_ROOT, "SIGMAEVOLVE_DATASET_ROOT"),
        openrouter_api_key=_resolve_optional_env("SIGMAEVOLVE_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
        modal_app_name=_resolve_required_default(DEFAULT_MODAL_APP_NAME, "SIGMAEVOLVE_MODAL_APP_NAME"),
        modal_function_name=_resolve_required_default(
            DEFAULT_MODAL_FUNCTION_NAME,
            "SIGMAEVOLVE_MODAL_FUNCTION_NAME",
        ),
        modal_dataset_volume=_resolve_required_default(
            DEFAULT_MODAL_DATASET_VOLUME,
            "SIGMAEVOLVE_MODAL_DATASET_VOLUME",
        ),
        modal_dataset_mount=_resolve_required_default(
            DEFAULT_MODAL_DATASET_MOUNT,
            "SIGMAEVOLVE_MODAL_DATASET_MOUNT",
        ),
        modal_environment_name=_resolve_optional_env("SIGMAEVOLVE_MODAL_ENVIRONMENT_NAME"),
    )


def _resolve_optional_env(*names: str) -> str | None:
    # Treat unset and blank env vars as absent so defaults remain predictable.
    for name in names:
        value = os.getenv(name)
        if isinstance(value, str) and value.strip():
            return value

    return None


def _resolve_required_default(default: str, *names: str) -> str:
    value = _resolve_optional_env(*names)
    if value is not None:
        return value

    return default
