from __future__ import annotations

from pathlib import Path
from typing import Any

from sigmaevolve.datasets import DatasetManager
from sigmaevolve.orchestrator import ModalRemoteLauncher


DEFAULT_MODAL_APP_NAME = "sigmaevolve-runner"
DEFAULT_MODAL_FUNCTION_NAME = "run_trial"
DEFAULT_MODAL_DATASET_VOLUME = "sigmaevolve-datasets"
DEFAULT_MODAL_DATASET_MOUNT = "/mnt/datasets"


def require_modal():
    try:
        import modal  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Modal support requires the 'modal' package. Install it with: pip install '.[modal]'") from exc
    return modal


class _ModalFunctionProxy:
    def __init__(
        self,
        app_name: str,
        function_name: str,
        database_url: str,
        dataset_root: str,
        environment_name: str | None = None,
    ) -> None:
        self.app_name = app_name
        self.function_name = function_name
        self.database_url = database_url
        self.dataset_root = dataset_root
        self.environment_name = environment_name

    def spawn(self, trial_id: str, dispatch_token: str):
        modal = require_modal()
        function = modal.Function.from_name(
            self.app_name,
            self.function_name,
            environment_name=self.environment_name,
        )
        return function.spawn(
            trial_id=trial_id,
            dispatch_token=dispatch_token,
            database_url=self.database_url,
            dataset_root=self.dataset_root,
        )


def create_modal_launcher(
    app_name: str,
    function_name: str,
    database_url: str,
    dataset_root: str = DEFAULT_MODAL_DATASET_MOUNT,
    environment_name: str | None = None,
) -> ModalRemoteLauncher:
    proxy = _ModalFunctionProxy(
        app_name=app_name,
        function_name=function_name,
        database_url=database_url,
        dataset_root=dataset_root,
        environment_name=environment_name,
    )
    return ModalRemoteLauncher(proxy)


def deploy_modal_app(
    app_name: str = DEFAULT_MODAL_APP_NAME,
    function_name: str = DEFAULT_MODAL_FUNCTION_NAME,
    dataset_volume_name: str = DEFAULT_MODAL_DATASET_VOLUME,
    dataset_mount_path: str = DEFAULT_MODAL_DATASET_MOUNT,
    environment_name: str | None = None,
) -> dict[str, Any]:
    modal = require_modal()
    from sigmaevolve.modal_app import build_modal_app

    app = build_modal_app(
        app_name=app_name,
        function_name=function_name,
        dataset_volume_name=dataset_volume_name,
        dataset_mount_path=dataset_mount_path,
    )
    with modal.enable_output():
        app.deploy(name=app_name, environment_name=environment_name)
    return {
        "app_name": app_name,
        "function_name": function_name,
        "dataset_volume_name": dataset_volume_name,
        "dataset_mount_path": dataset_mount_path,
        "environment_name": environment_name,
    }


def sync_dataset_to_modal(
    dataset_id: str,
    dataset_root: str | Path,
    volume_name: str = DEFAULT_MODAL_DATASET_VOLUME,
    environment_name: str | None = None,
) -> dict[str, Any]:
    modal = require_modal()
    local_dataset_root = Path(dataset_root)
    manager = DatasetManager(local_dataset_root, providers={})
    manifest_path = manager.manifest_path_for(dataset_id)
    local_dir = manifest_path.parent
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found locally for {dataset_id!r}: {manifest_path}")
    volume = modal.Volume.from_name(
        volume_name,
        create_if_missing=True,
        environment_name=environment_name,
    )
    remote_dir = DatasetManager.safe_dir_name(dataset_id)
    with modal.enable_output():
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(local_dir), remote_path=remote_dir)
        volume.commit()
    return {
        "dataset_id": dataset_id,
        "local_dir": str(local_dir),
        "remote_dir": remote_dir,
        "volume_name": volume_name,
        "environment_name": environment_name,
    }
