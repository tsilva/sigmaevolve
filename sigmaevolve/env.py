from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENV_PATH = Path.home() / ".config" / "sigmaevolve" / ".env"
DEFAULT_DATASET_ROOT = "./artifacts/datasets"
INFISICAL_ROOT = Path(__file__).resolve().parents[1]
INFISICAL_PROJECT = "adceee68-d107-4399-a6c6-65bc6f09ced1"
INFISICAL_DOMAIN = "https://app.infisical.com"
EXPERIMENT_SECRET_KEYS = frozenset(
    {
        "DATABASE_URL",
        "SIGMAEVOLVE_DATABASE_URL",
        "OPENROUTER_API_KEY",
        "SIGMAEVOLVE_OPENROUTER_API_KEY",
        "WANDB_API_KEY",
    }
)
DASHBOARD_SECRET_KEYS = frozenset({"SENTRY_AUTH_TOKEN", "SENTRY_SMOKE_TOKEN"})
PRIVATE_ENV_KEYS = EXPERIMENT_SECRET_KEYS | DASHBOARD_SECRET_KEYS


class ManagedSecretsError(Exception):
    """Expose only fixed, credential-free errors at the CLI boundary."""


def _is_manager_setting(key: str) -> bool:
    return key.startswith("INFISICAL_") or key == "BWS_ACCESS_TOKEN"


def _clear_private_environment() -> None:
    for key in list(os.environ):
        if key in PRIVATE_ENV_KEYS or _is_manager_setting(key):
            os.environ.pop(key, None)
    for key in PRIVATE_ENV_KEYS:
        os.environ[key] = ""


def _validate_managed_configuration(root: Path) -> None:
    try:
        settings = json.loads((root / ".infisical.json").read_text())
        if (
            settings["workspaceId"] != INFISICAL_PROJECT
            or settings.get("domain") != INFISICAL_DOMAIN
            or settings.get("defaultEnvironment") != "dev"
        ):
            raise ValueError()
    except Exception:
        raise ManagedSecretsError(
            "SigmaEvolve's development Infisical configuration is invalid."
        ) from None


def _parse_managed_secrets(payload: str) -> dict[str, str]:
    try:
        rows = json.loads(payload)
        if not isinstance(rows, list):
            raise ValueError()
        values: dict[str, str] = {}
        for row in rows:
            key, value = row["key"], row["value"]
            if (
                not isinstance(key, str)
                or not isinstance(value, str)
                or "\0" in key
                or "\0" in value
                or row["workspace"] != INFISICAL_PROJECT
                or row["secretPath"] != "/"
                or row["type"] != "shared"
                or key in values
            ):
                raise ValueError()
            values[key] = value
        return {key: values.get(key, "") for key in EXPERIMENT_SECRET_KEYS}
    except Exception:
        raise ManagedSecretsError(
            "Unexpected Infisical response; provider details were suppressed."
        ) from None


def load_managed_secrets(root: Path = INFISICAL_ROOT) -> None:
    """Replace private settings using the repository's fixed development source."""
    # Clear stale credentials even if configuration or the provider request fails.
    _clear_private_environment()
    _validate_managed_configuration(root)
    command = [
        "infisical",
        "export",
        "--format",
        "json",
        "--expand=false",
        "--include-imports=false",
        "--secret-overriding=false",
        "--projectId",
        INFISICAL_PROJECT,
        "--domain",
        INFISICAL_DOMAIN,
        "--env",
        "dev",
        "--path",
        "/",
        "--silent",
        "--telemetry=false",
    ]
    environment = {
        key: value for key, value in os.environ.items() if key not in PRIVATE_ENV_KEYS
    }
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode:
            raise ValueError()
    except Exception:
        raise ManagedSecretsError(
            "Infisical request failed. Check installation, login and project access; "
            "provider details were suppressed."
        ) from None
    os.environ.update(_parse_managed_secrets(result.stdout))


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


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> None:

    # Resolve the default user-scoped env file and exit quietly when it is absent.
    env_path = Path(path) if path is not None else DEFAULT_ENV_PATH

    # Skip missing env files instead of treating them as a configuration error.
    if not env_path.exists():
        return

    # Parse simple shell-style KEY=VALUE lines while ignoring blanks and comments.
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()

        # Ignore blank lines and comment lines in the env file.
        if not line or line.startswith("#"):
            continue

        # Support `export KEY=VALUE` lines from shell-style env files.
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        # Skip malformed lines that do not contain an assignment.
        if "=" not in line:
            continue

        # Normalize the key/value tokens before applying optional quote stripping.
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Userfile settings must never restore credentials or manager authentication.
        if not key or key in PRIVATE_ENV_KEYS or _is_manager_setting(key):
            continue

        # Remove matching quotes around the value when the assignment uses them.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        # Preserve existing environment values unless override mode is active.
        if override or key not in os.environ:
            os.environ[key] = value


def resolve_runtime_config() -> RuntimeConfig:

    # Resolve Modal defaults lazily so env loading does not participate in import cycles.
    from sigmaevolve.modal import (  # noqa: PLC0415
        DEFAULT_MODAL_APP_NAME,
        DEFAULT_MODAL_DATASET_MOUNT,
        DEFAULT_MODAL_DATASET_VOLUME,
        DEFAULT_MODAL_FUNCTION_NAME,
    )

    # Resolve each runtime setting from SigmaEvolve-scoped env vars first.
    database_url = _resolve_optional_env("SIGMAEVOLVE_DATABASE_URL", "DATABASE_URL")
    dataset_root = _resolve_required_default(
        DEFAULT_DATASET_ROOT,
        "SIGMAEVOLVE_DATASET_ROOT",
    )
    openrouter_api_key = _resolve_optional_env(
        "SIGMAEVOLVE_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
    )
    modal_app_name = _resolve_required_default(
        DEFAULT_MODAL_APP_NAME,
        "SIGMAEVOLVE_MODAL_APP_NAME",
    )
    modal_function_name = _resolve_required_default(
        DEFAULT_MODAL_FUNCTION_NAME,
        "SIGMAEVOLVE_MODAL_FUNCTION_NAME",
    )
    modal_dataset_volume = _resolve_required_default(
        DEFAULT_MODAL_DATASET_VOLUME,
        "SIGMAEVOLVE_MODAL_DATASET_VOLUME",
    )
    modal_dataset_mount = _resolve_required_default(
        DEFAULT_MODAL_DATASET_MOUNT,
        "SIGMAEVOLVE_MODAL_DATASET_MOUNT",
    )
    modal_environment_name = _resolve_optional_env("SIGMAEVOLVE_MODAL_ENVIRONMENT_NAME")

    # Assemble the final runtime config once each field has been resolved.
    return RuntimeConfig(
        database_url=database_url,
        dataset_root=dataset_root,
        openrouter_api_key=openrouter_api_key,
        modal_app_name=modal_app_name,
        modal_function_name=modal_function_name,
        modal_dataset_volume=modal_dataset_volume,
        modal_dataset_mount=modal_dataset_mount,
        modal_environment_name=modal_environment_name,
    )


def _resolve_optional_env(*names: str) -> str | None:

    # Treat unset and blank env vars as absent so defaults remain predictable.
    for name in names:
        value = os.getenv(name)
        has_non_blank_value = isinstance(value, str) and bool(value.strip())

        # Return the first configured environment value that still contains content.
        if has_non_blank_value:
            return value

    return None


def _resolve_required_default(default: str, *names: str) -> str:
    value = _resolve_optional_env(*names)
    has_override = value is not None

    # Use the explicit override when one is present, otherwise fall back to the default.
    if has_override:
        return value

    return default
