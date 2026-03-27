from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sigmaevolve.train_script_runtime import (
    seed_everything,
    write_best_model_atomic,
    write_eval_atomic,
    write_json_atomic,
)

logger = logging.getLogger(__name__)


def make_experiment(device, train_ds, val_ds):
    # EVOLVE-BLOCK-START
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Create model
    flat_dim = int(train_ds[0][0].numel())
    num_classes = int(train_ds.tensors[1].max().item()) + 1
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat_dim, num_classes),
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create loss function
    def loss_fn(batch):
        x, y = (tensor.to(device) for tensor in batch)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        return loss, logits, y

    # EVOLVE-BLOCK-END

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": None,
        "loss_fn": loss_fn,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_batch,
    *,
    optimizer=None,
    training: bool | None = None,
) -> dict[str, Any]:
    # Set model in training mode if optimizer is provided
    # (otherwise it will be in evaluation mode)
    is_training = optimizer is not None if training is None else training
    model.train(is_training)
    context = torch.enable_grad() if is_training else torch.no_grad()

    # Start epoch
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    predictions: list[torch.Tensor] = []
    with context:
        # Iterate over batches
        for batch in loader:
            # Compute loss, logits and targets
            loss, logits, labels = loss_batch(batch)

            # If optimizer is provided,
            # backprop loss and step optimizer
            if optimizer is not None and loss.requires_grad:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # Update total loss, correct and predictions
            batch_predictions = logits.argmax(dim=1)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((batch_predictions == labels).sum().item())
            total_examples += batch_size
            predictions.append(batch_predictions.detach().cpu())

    # Return average loss, accuracy and predictions
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "predictions": torch.cat(predictions, dim=0).numpy(),
    }


def _read_split(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    payload = np.load(path)
    features = payload["features"].astype(np.float32)
    labels = payload["labels"].astype(np.int64) if "labels" in payload else None
    return features, labels


def fit(config: dict[str, Any]) -> dict[str, Any]:
    # Set random seed
    device = seed_everything(config["random_seed"])

    # Load dataset
    x_train, y_train = _read_split(config["train_split_path"])
    x_val, _ = _read_split(config["validation_split_path"])
    y_val = np.load(config["validation_labels_path"]).astype(np.int64)
    train_ds = TensorDataset(
        torch.from_numpy(x_train).contiguous(),
        torch.from_numpy(y_train).contiguous(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val).contiguous(),
        torch.from_numpy(y_val).contiguous(),
    )

    # Create experiment configuration
    experiment = make_experiment(device, train_ds, val_ds)
    model = experiment["model"]
    optimizer = experiment["optimizer"]
    scheduler = experiment["scheduler"]
    loss_fn = experiment["loss_fn"]
    train_loader = experiment["train_loader"]
    val_loader = experiment["val_loader"]

    # Write the initial progress payload before training starts
    write_json_atomic(
        config["progress_path"],
        {
            "phase": "train",
            "elapsed_time_sec": 0.0,
            "last_completed_eval_sec": None,
            "eval_index": 0,
            "epoch_index": 0,
        },
    )

    # Start training loop    patience = 2
    min_delta = 0.0
    start_time = time.monotonic()
    best_state = None
    best_metrics = None
    best_accuracy = -1.0
    bad_epochs = 0
    eval_index = 0
    last_eval_sec: float | None = None
    patience = 3
    early_stopped = False
    epochs_completed = 0
    for epoch_index in range(config["num_epochs"]):
        # Update progress before running the next training epoch
        write_json_atomic(
            config["progress_path"],
            {
                "phase": "train",
                "elapsed_time_sec": float(time.monotonic() - start_time),
                "last_completed_eval_sec": last_eval_sec,
                "eval_index": eval_index,
                "epoch_index": epoch_index,
            },
        )

        # Run training epoch
        train_result = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer=optimizer,
            training=True,
        )

        # Mark the switch from training to evaluation
        write_json_atomic(
            config["progress_path"],
            {
                "phase": "eval",
                "elapsed_time_sec": float(time.monotonic() - start_time),
                "last_completed_eval_sec": last_eval_sec,
                "eval_index": eval_index,
                "epoch_index": epoch_index,
            },
        )

        # Run validation epoch
        val_result = run_epoch(model, val_loader, loss_fn, training=False)

        # If scheduler is available take a step
        if scheduler is not None:
            try:
                scheduler.step(val_result["loss"])
            except TypeError:
                scheduler.step()

        # Persist the evaluation metrics and predictions for this epoch
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

        # If validation accuracy improved store the best state and metrics
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
            # Otherwise increment the bad epoch counter
            bad_epochs += 1

        # Store the latest completed evaluation checkpoint in progress
        last_eval_sec = elapsed_after_eval
        epochs_completed = epoch_index + 1
        write_json_atomic(
            config["progress_path"],
            {
                "phase": "train",
                "elapsed_time_sec": float(elapsed_after_eval),
                "last_completed_eval_sec": last_eval_sec,
                "eval_index": eval_index,
                "epoch_index": epochs_completed,
            },
        )

        # Print training and validation metrics
        print(
            f"epoch {epoch_index + 1:02d} | "
            f"train_loss={train_result['loss']:.4f} "
            f"train_acc={train_result['accuracy']:.4f} | "
            f"val_loss={val_result['loss']:.4f} "
            f"val_acc={val_result['accuracy']:.4f}",
            flush=True,
        )

        # If patience is exhausted, mark the run as early stopped
        if (
            patience
            and bad_epochs >= patience
            and epoch_index + 1 < config["num_epochs"]
        ):
            early_stopped = True
            break

    # If we found a best state, load it into the model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Write the final progress payload after training completes
    write_json_atomic(
        config["progress_path"],
        {
            "phase": "finished",
            "elapsed_time_sec": float(time.monotonic() - start_time),
            "last_completed_eval_sec": last_eval_sec,
            "eval_index": eval_index,
            "epoch_index": epochs_completed,
            "early_stopped": early_stopped,
        },
    )

    write_json_atomic(
        config["debug_output_path"],
        {
            "timed_out": False,
            "eval_count": eval_index,
            "epochs_completed": epochs_completed,
            "early_stopped": early_stopped,
            "early_stopping_patience": patience,
        },
    )

    # Print the best validation metrics seen during training
    print(
        f"best_val_loss={best_metrics['val_loss']:.4f} "
        f"best_val_acc={best_metrics['val_acc']:.4f}",
        flush=True,
    )
    return best_metrics


def main(argv: list[str] | None = None) -> int:
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = json.loads(Path(parser.parse_args(argv).config).read_text())
    config = {
        "progress_path": Path(config["progress_path"]),
        "eval_dir": Path(config["eval_dir"]),
        "best_model_path": Path(config["best_model_path"]),
        "debug_output_path": Path(config["debug_output_path"]),
        "num_epochs": int(config["epochs"]),
        "random_seed": int(config["random_seed"]),
        "dataset_metadata": dict(config.get("dataset_metadata") or {}),
        "train_split_path": Path(config["train_split_path"]),
        "validation_split_path": Path(config["validation_split_path"]),
        "validation_labels_path": Path(config["validation_labels_path"]),
    }

    # Start training
    fit(config)

    # Return success
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
