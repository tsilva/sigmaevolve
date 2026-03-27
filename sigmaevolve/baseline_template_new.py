from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def make_experiment(device, train_ds, val_ds):
    # EVOLVE-BLOCK-START
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    flat_dim = int(train_ds[0][0].numel())
    num_classes = int(train_ds.tensors[1].max().item()) + 1

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat_dim, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    ).to(device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = None
    if trainable_parameters:
        optimizer = torch.optim.Adam(trainable_parameters, lr=1e-3)

    scheduler = None
    if optimizer is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1
        )

    early_stopping_patience = 2
    min_delta = 0.0

    def loss_fn(batch):
        x, y = (tensor.to(device) for tensor in batch)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        return loss, logits, y

    # EVOLVE-BLOCK-END

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_fn": loss_fn,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "early_stopping_patience": early_stopping_patience,
        "min_delta": min_delta,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    serialized_payload = json.dumps(payload, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(serialized_payload)
    temp_path.replace(path)


def write_eval_atomic(
    eval_dir: Path,
    eval_index: int,
    predictions: np.ndarray,
    elapsed_time_sec: float,
    epoch: int,
    metrics: dict[str, float] | None = None,
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    temp_path = eval_dir / f".eval_{eval_index:04d}.tmp.npz"
    payload: dict[str, Any] = {
        "predictions": np.asarray(predictions, dtype=np.int64),
        "eval_index": np.array(eval_index, dtype=np.int64),
        "elapsed_time_sec": np.array(elapsed_time_sec, dtype=np.float64),
        "epoch": np.array(epoch, dtype=np.int64),
    }
    for key, value in (metrics or {}).items():
        if value is None:
            continue
        payload[key] = np.array(float(value), dtype=np.float64)
    np.savez(temp_path, **payload)
    temp_path.replace(eval_dir / f"eval_{eval_index:04d}.npz")


def write_best_model_atomic(
    best_model_path: Path,
    *,
    model: torch.nn.Module,
    metrics: dict[str, float],
    eval_index: int,
    epoch: int,
    elapsed_time_sec: float,
) -> None:
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = best_model_path.with_suffix(best_model_path.suffix + ".tmp")
    state_dict = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    payload = {
        "state_dict": state_dict,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "eval_index": int(eval_index),
        "epoch": int(epoch),
        "elapsed_time_sec": float(elapsed_time_sec),
    }
    torch.save(payload, temp_path)
    temp_path.replace(best_model_path)


def seed_everything(seed: int) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        return torch.device("cuda")
    return torch.device("cpu")


def read_split(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    payload = np.load(path)
    features = payload["features"].astype(np.float32)
    labels = payload["labels"].astype(np.int64) if "labels" in payload else None
    return features, labels


def load_run_config(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = json.loads(Path(parser.parse_args(argv).config).read_text())
    return {
        "progress_path": Path(config["progress_path"]),
        "eval_dir": Path(config["eval_dir"]),
        "best_model_path": Path(config["best_model_path"]),
        "num_epochs": int(config["epochs"]),
        "random_seed": int(config["random_seed"]),
        "dataset_metadata": dict(config.get("dataset_metadata") or {}),
        "train_split_path": Path(config["train_split_path"]),
        "validation_split_path": Path(config["validation_split_path"]),
        "validation_labels_path": Path(config["validation_labels_path"]),
    }


def load_dataset_splits(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_features, train_labels = read_split(config["train_split_path"])
    validation_features, _ = read_split(config["validation_split_path"])
    validation_labels = np.load(config["validation_labels_path"]).astype(np.int64)
    return train_features, train_labels, validation_features, validation_labels


def build_tensor_datasets(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[TensorDataset, TensorDataset]:
    train_ds = TensorDataset(
        torch.from_numpy(train_features).contiguous(),
        torch.from_numpy(train_labels).contiguous(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(validation_features).contiguous(),
        torch.from_numpy(validation_labels).contiguous(),
    )
    return train_ds, val_ds


def build_debug_payload() -> dict[str, Any]:
    return {
        "timed_out": False,
        "eval_count": 0,
        "early_stopped": False,
        "early_stopping_patience": 0,
        "epochs_completed": 0,
    }


def write_progress(
    progress_path: Path,
    *,
    phase: str,
    elapsed_time_sec: float,
    last_eval_sec: float | None,
    eval_index: int,
    epoch_index: int,
    extras: dict[str, Any] | None = None,
) -> None:
    payload = {
        "phase": phase,
        "elapsed_time_sec": float(elapsed_time_sec),
        "last_completed_eval_sec": last_eval_sec,
        "eval_index": eval_index,
        "epoch_index": epoch_index,
    }
    payload.update(extras or {})
    write_json_atomic(progress_path, payload)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_batch,
    *,
    optimizer=None,
    training: bool | None = None,
) -> dict[str, Any]:
    is_training = optimizer is not None if training is None else training
    model.train(is_training)
    context = torch.enable_grad() if is_training else torch.no_grad()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    predictions: list[torch.Tensor] = []

    with context:
        for batch in loader:
            loss, logits, labels = loss_batch(batch)
            if optimizer is not None and loss.requires_grad:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_predictions = logits.argmax(dim=1)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((batch_predictions == labels).sum().item())
            total_examples += batch_size
            predictions.append(batch_predictions.detach().cpu())

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "predictions": torch.cat(predictions, dim=0).numpy(),
    }


def fit(config: dict[str, Any]) -> dict[str, Any]:
    train_features, train_labels, validation_features, validation_labels = (
        load_dataset_splits(config)
    )
    train_ds, val_ds = build_tensor_datasets(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )
    device = seed_everything(config["random_seed"])
    experiment = make_experiment(device, train_ds, val_ds)

    model = experiment["model"]
    optimizer = experiment["optimizer"]
    scheduler = experiment["scheduler"]
    loss_fn = experiment["loss_fn"]
    train_loader = experiment["train_loader"]
    val_loader = experiment["val_loader"]
    patience = int(experiment.get("early_stopping_patience", 0))
    min_delta = float(experiment.get("min_delta", 0.0))

    debug_payload = build_debug_payload()
    debug_payload["early_stopping_patience"] = patience
    start_time = time.monotonic()
    best_state = None
    best_metrics = None
    best_accuracy = -1.0
    bad_epochs = 0
    eval_index = 0
    last_eval_sec: float | None = None

    write_progress(
        config["progress_path"],
        phase="train",
        elapsed_time_sec=0.0,
        last_eval_sec=None,
        eval_index=0,
        epoch_index=0,
        extras=debug_payload,
    )

    for epoch_index in range(config["num_epochs"]):
        write_progress(
            config["progress_path"],
            phase="train",
            elapsed_time_sec=time.monotonic() - start_time,
            last_eval_sec=last_eval_sec,
            eval_index=eval_index,
            epoch_index=epoch_index,
            extras=debug_payload,
        )

        train_result = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer=optimizer,
            training=True,
        )
        write_progress(
            config["progress_path"],
            phase="eval",
            elapsed_time_sec=time.monotonic() - start_time,
            last_eval_sec=last_eval_sec,
            eval_index=eval_index,
            epoch_index=epoch_index,
            extras=debug_payload,
        )

        val_result = run_epoch(model, val_loader, loss_fn, training=False)
        if scheduler is not None:
            try:
                scheduler.step(val_result["loss"])
            except TypeError:
                scheduler.step()

        eval_index += 1
        elapsed_after_eval = time.monotonic() - start_time
        metrics = {
            "train_loss": float(train_result["loss"]),
            "train_acc": float(train_result["accuracy"]),
            "val_loss": float(val_result["loss"]),
            "val_acc": float(val_result["accuracy"]),
            "accuracy": float(val_result["accuracy"]),
        }
        write_eval_atomic(
            config["eval_dir"],
            eval_index=eval_index,
            predictions=val_result["predictions"],
            elapsed_time_sec=elapsed_after_eval,
            epoch=epoch_index + 1,
            metrics=metrics,
        )

        debug_payload.update(
            {
                "eval_count": eval_index,
                "epochs_completed": epoch_index + 1,
                "best_validation_accuracy_seen": max(
                    best_accuracy, float(val_result["accuracy"])
                ),
            }
        )
        improved = float(val_result["accuracy"]) > best_accuracy + min_delta
        if improved:
            best_accuracy = float(val_result["accuracy"])
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dict(metrics)
            bad_epochs = 0
            write_best_model_atomic(
                config["best_model_path"],
                model=model,
                metrics=metrics,
                eval_index=eval_index,
                epoch=epoch_index + 1,
                elapsed_time_sec=elapsed_after_eval,
            )
        else:
            bad_epochs += 1
        debug_payload["epochs_without_improvement"] = bad_epochs

        last_eval_sec = elapsed_after_eval
        write_progress(
            config["progress_path"],
            phase="train",
            elapsed_time_sec=elapsed_after_eval,
            last_eval_sec=last_eval_sec,
            eval_index=eval_index,
            epoch_index=epoch_index + 1,
            extras=debug_payload,
        )

        print(
            f"epoch {epoch_index + 1:02d} | "
            f"train_loss={train_result['loss']:.4f} "
            f"train_acc={train_result['accuracy']:.4f} | "
            f"val_loss={val_result['loss']:.4f} "
            f"val_acc={val_result['accuracy']:.4f}",
            flush=True,
        )

        if (
            patience
            and bad_epochs >= patience
            and epoch_index + 1 < config["num_epochs"]
        ):
            debug_payload["early_stopped"] = True
            debug_payload["early_stop_epoch"] = epoch_index + 1
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    write_progress(
        config["progress_path"],
        phase="finished",
        elapsed_time_sec=time.monotonic() - start_time,
        last_eval_sec=last_eval_sec,
        eval_index=eval_index,
        epoch_index=int(debug_payload["epochs_completed"]),
        extras=debug_payload,
    )

    result = dict(best_metrics)
    result.update(
        {
            "eval_count": eval_index,
            "epochs_completed": int(debug_payload["epochs_completed"]),
            "early_stopped": bool(debug_payload["early_stopped"]),
        }
    )
    print(
        f"best_val_loss={best_metrics['val_loss']:.4f} "
        f"best_val_acc={best_metrics['val_acc']:.4f}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    config = load_run_config(argv)
    fit(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
