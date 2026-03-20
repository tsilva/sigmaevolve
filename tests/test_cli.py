from __future__ import annotations

import json

from sigmaevolve.cli import main


def test_cli_create_track_and_list_trials(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'cli.sqlite'}"
    dataset_root = tmp_path / "datasets"

    from sigmaevolve.datasets import ArrayDatasetProvider
    import numpy as np

    provider = ArrayDatasetProvider(
        train_features=np.ones((4, 2), dtype=np.float32),
        train_labels=np.array([0, 1, 0, 1], dtype=np.int64),
        validation_features=np.ones((2, 2), dtype=np.float32),
        validation_labels=np.array([0, 1], dtype=np.int64),
        test_features=np.ones((2, 2), dtype=np.float32),
        test_labels=np.array([0, 1], dtype=np.int64),
        metadata={"num_classes": 2},
    )

    # Reuse the CLI entrypoint but patch system construction to avoid torchvision downloads.
    from sigmaevolve import cli as cli_module
    from sigmaevolve.system import build_system

    def fake_make_system(args):
        return build_system(
            database_url=args.database_url,
            dataset_root=args.dataset_root,
            providers={"mnist:v1": provider, "fashion_mnist:v1": provider},
        )

    monkeypatch.setattr(cli_module, "_make_system", fake_make_system)

    assert main(["--database-url", db_url, "--dataset-root", str(dataset_root), "prepare-dataset", "mnist:v1"]) == 0

    from io import StringIO
    import contextlib

    create_out = StringIO()
    with contextlib.redirect_stdout(create_out):
        assert (
            main(
                [
                    "--database-url",
                    db_url,
                    "--dataset-root",
                    str(dataset_root),
                    "create-track",
                    "mnist:v1",
                    "--name",
                    "cli-test",
                ]
            )
            == 0
        )
    track_id = json.loads(create_out.getvalue())["track_id"]

    list_out = StringIO()
    with contextlib.redirect_stdout(list_out):
        assert main(["--database-url", db_url, "--dataset-root", str(dataset_root), "list-trials", track_id]) == 0
    trials = json.loads(list_out.getvalue())
    assert len(trials) == 1
    assert trials[0]["status"] == "queued"
