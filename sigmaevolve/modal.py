from __future__ import annotations


# ---- modal_support.py ----

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sigmaevolve.datasets import DatasetManager
from sigmaevolve.orchestration import ModalRemoteLauncher


DEFAULT_MODAL_APP_NAME = "sigmaevolve-runner"
DEFAULT_MODAL_FUNCTION_NAME = "run_trial"
DEFAULT_MODAL_CLASS_NAME = "TrialRunner"
DEFAULT_MODAL_DATASET_VOLUME = "sigmaevolve-datasets"
DEFAULT_MODAL_DATASET_MOUNT = "/mnt/datasets"


def require_modal():
    # Import Modal lazily so local workflows do not require the optional dependency.
    try:
        import modal  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Modal support requires the 'modal' package. Install it with: "
            "pip install '.[modal]'"
        ) from exc

    return modal


@dataclass(frozen=True)
class ModalSpawnResult:
    function_call: Any


def _apply_runtime_options(
    handle: Any,
    *,
    modal,
    wandb_env: dict[str, str] | None = None,
):
    # Collect the optional WandB settings before mutating the handle.
    options: dict[str, Any] = {}
    has_wandb_secret = bool(wandb_env)
    if has_wandb_secret:
        options["secrets"] = [modal.Secret.from_dict(dict(wandb_env))]
    if not options:
        return handle

    # Apply the collected runtime options only when at least one was requested.
    return handle.with_options(**options)


class _ModalClassProxy:
    def __init__(
        self,
        app_name: str,
        class_name: str,
        method_name: str,
        database_url: str,
        dataset_root: str,
        environment_name: str | None = None,
        wandb_env: dict[str, str] | None = None,
    ) -> None:
        self.app_name = app_name
        self.class_name = class_name
        self.method_name = method_name
        self.database_url = database_url
        self.dataset_root = dataset_root
        self.environment_name = environment_name
        self.wandb_env = dict(wandb_env or {})

    def spawn(self, trial_id: str, dispatch_token: str):
        # Resolve the deployed Modal class handle and apply runtime overrides.
        modal = require_modal()
        cls = modal.Cls.from_name(
            self.app_name,
            self.class_name,
            environment_name=self.environment_name,
        )
        runtime_cls = _apply_runtime_options(
            cls,
            modal=modal,
            wandb_env=self.wandb_env,
        )
        method = getattr(runtime_cls(), self.method_name)

        # Spawn the remote function call with the database and dataset parameters.
        return ModalSpawnResult(
            function_call=method.spawn(
                trial_id=trial_id,
                dispatch_token=dispatch_token,
                database_url=self.database_url,
                dataset_root=self.dataset_root,
            ),
        )

    def cancel(self, run_id: str) -> None:
        modal = require_modal()
        modal.FunctionCall.from_id(run_id).cancel()


def create_modal_launcher(
    app_name: str,
    function_name: str,
    database_url: str,
    dataset_root: str = DEFAULT_MODAL_DATASET_MOUNT,
    environment_name: str | None = None,
    wandb_env: dict[str, str] | None = None,
) -> ModalRemoteLauncher:
    # Wrap the Modal class proxy in the launcher interface used by orchestration code.
    class_proxy = _ModalClassProxy(
        app_name=app_name,
        class_name=DEFAULT_MODAL_CLASS_NAME,
        method_name=function_name,
        database_url=database_url,
        dataset_root=dataset_root,
        environment_name=environment_name,
        wandb_env=wandb_env,
    )
    return ModalRemoteLauncher(class_proxy)


def deploy_modal_app(
    app_name: str = DEFAULT_MODAL_APP_NAME,
    function_name: str = DEFAULT_MODAL_FUNCTION_NAME,
    dataset_volume_name: str = DEFAULT_MODAL_DATASET_VOLUME,
    dataset_mount_path: str = DEFAULT_MODAL_DATASET_MOUNT,
    environment_name: str | None = None,
) -> dict[str, Any]:
    # Reject unsupported deployment names until the deployed module supports them.
    modal = require_modal()
    is_default_deployment = (
        app_name != DEFAULT_MODAL_APP_NAME
        or function_name != DEFAULT_MODAL_FUNCTION_NAME
        or dataset_volume_name != DEFAULT_MODAL_DATASET_VOLUME
        or dataset_mount_path != DEFAULT_MODAL_DATASET_MOUNT
    )
    if not is_default_deployment:
        raise ValueError(
            "Custom Modal app/function/volume names are not yet supported by the deployed app module. "
            "Use the defaults for now."
        )

    # Import the deployed app object only after the configuration has been validated.
    from sigmaevolve.modal import app

    # Deploy the app with Modal's progress output enabled.
    with modal.enable_output():
        app.deploy(name=app_name, environment_name=environment_name)
    deployment_payload = {
        "app_name": app_name,
        "function_name": function_name,
        "dataset_volume_name": dataset_volume_name,
        "dataset_mount_path": dataset_mount_path,
        "environment_name": environment_name,
    }
    return deployment_payload


def sync_dataset_to_modal(
    dataset_id: str,
    dataset_root: str | Path,
    volume_name: str = DEFAULT_MODAL_DATASET_VOLUME,
    environment_name: str | None = None,
) -> dict[str, Any]:
    # Resolve the prepared local dataset directory before uploading to Modal.
    modal = require_modal()
    local_dataset_root = Path(dataset_root)
    manager = DatasetManager(local_dataset_root, providers={})
    manifest_path = manager.manifest_path_for(dataset_id)
    local_dir = manifest_path.parent
    manifest_exists = manifest_path.exists()
    if not manifest_exists:
        raise FileNotFoundError(f"Dataset manifest not found locally for {dataset_id!r}: {manifest_path}")

    # Open the target volume and upload the dataset directory in one batch.
    volume = modal.Volume.from_name(
        volume_name,
        create_if_missing=True,
        environment_name=environment_name,
    )
    remote_dir = DatasetManager.safe_dir_name(dataset_id)
    with modal.enable_output():
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(local_dir), remote_path=remote_dir)

    # Return the local and remote paths so callers can report the sync result.
    return {
        "dataset_id": dataset_id,
        "local_dir": str(local_dir),
        "remote_dir": remote_dir,
        "volume_name": volume_name,
        "environment_name": environment_name,
    }


# ---- modal_app.py ----

import logging
import os
from pathlib import Path
from uuid import uuid4

import modal

from sigmaevolve.core import DEFAULT_TRIAL_HARD_TIMEOUT_SEC
from sigmaevolve.execution import apply_wandb_env


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)


app = modal.App(DEFAULT_MODAL_APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "psycopg[binary]>=3.1",
        "sqlalchemy>=2.0",
        "torch>=2.0",
        "torchvision>=0.18",
        "wandb>=0.19",
    )
    .add_local_python_source("sigmaevolve")
)
dataset_volume = modal.Volume.from_name(DEFAULT_MODAL_DATASET_VOLUME, create_if_missing=True)


@app.cls(
    image=image,
    volumes={DEFAULT_MODAL_DATASET_MOUNT: dataset_volume},
    timeout=DEFAULT_TRIAL_HARD_TIMEOUT_SEC,
)
class TrialRunner:
    @modal.method()
    def run_trial(
        self,
        trial_id: str,
        dispatch_token: str,
        database_url: str,
        dataset_root: str = DEFAULT_MODAL_DATASET_MOUNT,
        wandb_env: dict[str, str] | None = None,
    ) -> None:
        # Import the heavy runtime dependencies lazily inside the Modal worker.
        from sigmaevolve.datasets import DatasetManager
        from sigmaevolve.execution import RunnerService
        from sigmaevolve.storage import SQLAlchemyRepository

        # Reconstruct the repository and dataset manager from the remote runtime inputs.
        apply_wandb_env(wandb_env)
        has_database_url = isinstance(database_url, str) and bool(database_url)
        if has_database_url:
            os.environ.setdefault("DATABASE_URL", database_url)
        repository = SQLAlchemyRepository(database_url)
        dataset_manager = DatasetManager(Path(dataset_root), providers={})
        runner = RunnerService(repository=repository, dataset_manager=dataset_manager)

        # Run the reserved trial with a unique Modal-scoped runner identity.
        runner_id = f"modal_{uuid4().hex}"
        runner.run_reserved_trial(trial_id=trial_id, dispatch_token=dispatch_token, runner_id=runner_id)
