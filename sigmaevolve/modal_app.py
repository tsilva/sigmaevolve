from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import uuid4

import modal

from sigmaevolve.modal_support import (
    DEFAULT_MODAL_APP_NAME,
    DEFAULT_MODAL_DATASET_MOUNT,
    DEFAULT_MODAL_DATASET_VOLUME,
    DEFAULT_MODAL_FUNCTION_NAME,
)
from sigmaevolve.runtime_config import DEFAULT_TRIAL_HARD_TIMEOUT_SEC
from sigmaevolve.wandb_support import apply_wandb_env


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
        from sigmaevolve.datasets import DatasetManager
        from sigmaevolve.runner import RunnerService
        from sigmaevolve.storage import SQLAlchemyRepository

        apply_wandb_env(wandb_env)
        if isinstance(database_url, str) and database_url:
            os.environ.setdefault("SIGMAEVOLVE_DATABASE_URL", database_url)
        repository = SQLAlchemyRepository(database_url)
        dataset_manager = DatasetManager(Path(dataset_root), providers={})
        runner = RunnerService(repository=repository, dataset_manager=dataset_manager)
        runner_id = f"modal_{uuid4().hex}"
        runner.run_reserved_trial(trial_id=trial_id, dispatch_token=dispatch_token, runner_id=runner_id)
