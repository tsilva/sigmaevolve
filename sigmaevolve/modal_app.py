from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import modal

from sigmaevolve.modal_support import (
    DEFAULT_MODAL_APP_NAME,
    DEFAULT_MODAL_DATASET_MOUNT,
    DEFAULT_MODAL_DATASET_VOLUME,
    DEFAULT_MODAL_FUNCTION_NAME,
)


def build_modal_app(
    app_name: str = DEFAULT_MODAL_APP_NAME,
    function_name: str = DEFAULT_MODAL_FUNCTION_NAME,
    dataset_volume_name: str = DEFAULT_MODAL_DATASET_VOLUME,
    dataset_mount_path: str = DEFAULT_MODAL_DATASET_MOUNT,
) -> modal.App:
    app = modal.App(app_name)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("numpy>=1.26", "sqlalchemy>=2.0", "torch>=2.0", "torchvision>=0.18")
        .add_local_python_source("sigmaevolve")
    )
    dataset_volume = modal.Volume.from_name(dataset_volume_name, create_if_missing=True)

    @app.function(
        name=function_name,
        image=image,
        volumes={dataset_mount_path: dataset_volume},
        timeout=15 * 60,
    )
    def run_trial(trial_id: str, dispatch_token: str, database_url: str, dataset_root: str = dataset_mount_path) -> None:
        from sigmaevolve.datasets import DatasetManager
        from sigmaevolve.runner import RunnerService
        from sigmaevolve.storage import SQLAlchemyRepository

        repository = SQLAlchemyRepository(database_url)
        dataset_manager = DatasetManager(Path(dataset_root), providers={})
        runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
        runner_id = f"modal_{uuid4().hex}"
        runner.run_reserved_trial(trial_id=trial_id, dispatch_token=dispatch_token, runner_id=runner_id)

    return app


app = build_modal_app()
