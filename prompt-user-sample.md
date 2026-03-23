# Sample User Prompt

~~~text
Write a complete Python train.py module for dataset mnist:v1.

Use the dataset metadata below when choosing the model, augmentation, and loss setup:
- num_classes: 10
- feature_shape: 28, 28
- feature_dtype: float32
- label_dtype: int64
- source: mnist

Use these track-specific runtime settings:
- epochs: 5

This attempt was selected with the following generation settings:
- model: x-ai/grok-4.1-fast
- temperature: 0.2
- max_tokens: 2500
- retry_count: 2
- selection_probability: 0.5436

The broader generation policy for the track is:
- backend: openrouter
- selection: weighted_random
- seed: 0
- model_pool:
  - model: x-ai/grok-4.1-fast
  - temperature: 0.2
  - max_tokens: 2500
  - retry_count: 2
  - probability: 0.5436
  - model: google/gemini-3.1-flash-lite-preview
  - temperature: 0.2
  - max_tokens: 2500
  - retry_count: 2
  - probability: 0.2446
  - model: moonshotai/kimi-k2.5
  - temperature: 0.2
  - max_tokens: 2500
  - retry_count: 2
  - probability: 0.1578
  - model: google/gemini-3.1-pro-preview
  - temperature: 0.2
  - max_tokens: 2500
  - retry_count: 2
  - probability: 0.0306
  - model: anthropic/claude-sonnet-4.6
  - temperature: 0.2
  - max_tokens: 2500
  - retry_count: 2
  - probability: 0.0233

Use this parent trial as the base candidate:
Trial trial_parent:
- score: 0.988
- outcome reason: succeeded
- metrics:
  - accuracy: 0.988

Parent source:
```python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


class TrainScriptContractError(RuntimeError):
    pass


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, sort_keys=True))
    temp_path.replace(path)


def write_eval_atomic(
    eval_dir: Path,
    eval_index: int,
    predictions: np.ndarray,
    elapsed_time_sec: float,
    epoch: int,
) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    final_path = eval_dir / f"eval_{eval_index:04d}.npz"
    temp_path = eval_dir / f".eval_{eval_index:04d}.tmp.npz"
    np.savez(
        temp_path,
        predictions=np.asarray(predictions, dtype=np.int64),
        eval_index=np.array(eval_index, dtype=np.int64),
        elapsed_time_sec=np.array(elapsed_time_sec, dtype=np.float64),
        epoch=np.array(epoch, dtype=np.int64),
    )
    temp_path.replace(final_path)
    return final_path


def seed_everything(seed: int) -> str:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return "cpu"


def read_split(path: str) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    payload = np.load(path)
    features = payload["features"].astype(np.float32)
    if "labels" in payload:
        return features, payload["labels"].astype(np.int64)
    return features


def read_labels(path: str) -> np.ndarray:
    return np.load(path).astype(np.int64)


def infer_input_shape(features: np.ndarray) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in features.shape[1:])
    if not shape:
        raise TrainScriptContractError("training features must include at least one non-batch dimension")
    return shape


def to_feature_tensor(features: np.ndarray, *, input_shape: tuple[int, ...]) -> torch.Tensor:
    tensor = torch.from_numpy(features.astype(np.float32))
    if len(input_shape) == 2:
        tensor = tensor.unsqueeze(1)
    return tensor.contiguous()


def normalize_feature_tensors(train_x: torch.Tensor, validation_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if train_x.ndim <= 1:
        raise TrainScriptContractError("feature tensors must be at least 2D including the batch axis")
    if train_x.ndim == 2:
        reduce_dims = (0,)
    else:
        reduce_dims = (0,) + tuple(range(2, train_x.ndim))
    mean = train_x.mean(dim=reduce_dims, keepdim=True)
    std = train_x.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (validation_x - mean) / std


def normalize_predictions(raw_predictions: object, *, num_examples: int, num_classes: int | None) -> np.ndarray:
    if isinstance(raw_predictions, torch.Tensor):
        array = raw_predictions.detach().cpu().numpy()
    else:
        array = np.asarray(raw_predictions)
    if array.ndim == 0:
        raise TrainScriptContractError("model evaluation must return one prediction per validation example.")
    if array.shape[0] != num_examples:
        raise TrainScriptContractError(
            f"model evaluation returned {array.shape[0]} predictions for {num_examples} validation examples."
        )
    if array.ndim == 1:
        if np.issubdtype(array.dtype, np.floating):
            if num_classes == 2:
                finite = array[np.isfinite(array)]
                if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
                    return (array >= 0.5).astype(np.int64)
                return (array >= 0.0).astype(np.int64)
            raise TrainScriptContractError(
                "model evaluation returned a 1D float array for a non-binary task; return class ids or logits."
            )
        return array.astype(np.int64)
    reshaped = array.reshape(num_examples, -1)
    if reshaped.shape[1] <= 1:
        return reshaped.reshape(num_examples).astype(np.int64)
    return reshaped.argmax(axis=1).astype(np.int64)


def coerce_model_logits(raw_output: object, *, batch_size: int, num_classes: int) -> torch.Tensor:
    logits = raw_output if isinstance(raw_output, torch.Tensor) else torch.as_tensor(raw_output, dtype=torch.float32)
    if logits.ndim == 1:
        if batch_size == 1 and logits.shape[0] == num_classes:
            return logits.reshape(1, num_classes)
        if num_classes == 2 and logits.shape[0] == batch_size:
            return torch.stack((-logits, logits), dim=1)
    if logits.ndim == 2 and logits.shape[0] == batch_size:
        return logits
    raise TrainScriptContractError(
        f"model forward must return logits shaped [batch, num_classes], received {tuple(logits.shape)}"
    )


# EVOLVE-BLOCK-START
class EvolvedModel(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        if len(input_shape) <= 1:
            flat_dim = int(np.prod(input_shape))
            self.network = torch.nn.Sequential(
                torch.nn.Linear(flat_dim, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 128),
                torch.nn.GELU(),
                torch.nn.Linear(128, num_classes),
            )
        else:
            channels = 1 if len(input_shape) == 2 else int(input_shape[0])
            self.network = torch.nn.Sequential(
                torch.nn.Conv2d(channels, 24, kernel_size=5, padding=2),
                torch.nn.GELU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(24, 48, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.MaxPool2d(2),
                torch.nn.AdaptiveAvgPool2d((4, 4)),
                torch.nn.Flatten(),
                torch.nn.Linear(48 * 4 * 4, 64),
                torch.nn.GELU(),
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(64, num_classes),
            )

    def forward(self, x):
        return self.network(x)


def build_model(*, input_shape, num_classes):
    return EvolvedModel(input_shape=input_shape, num_classes=num_classes)


# EVOLVE-BLOCK-END


def load_model_builder():
    build_model_fn = globals().get("build_model")
    if not callable(build_model_fn):
        raise TrainScriptContractError("train.py is missing required evolve-block callable: build_model")
    return build_model_fn


def write_progress(
    progress_path: Path,
    *,
    phase: str,
    elapsed_time_sec: float,
    last_completed_eval_sec: float | None,
    eval_index: int,
    epoch_index: int,
) -> None:
    write_json_atomic(
        progress_path,
        {
            "phase": phase,
            "elapsed_time_sec": float(elapsed_time_sec),
            "last_completed_eval_sec": last_completed_eval_sec,
            "eval_index": eval_index,
            "epoch_index": epoch_index,
        },
    )


def maybe_call_epoch_hook(model: torch.nn.Module, *, epoch_index: int, num_epochs: int) -> None:
    hook = getattr(model, "on_epoch_start", None)
    if callable(hook):
        hook(epoch_index=epoch_index, num_epochs=num_epochs)


def run_validation(
    model: torch.nn.Module,
    validation_x: torch.Tensor,
    *,
    batch_size: int,
    num_classes: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    validation_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(validation_x),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    with torch.no_grad():
        for (batch_x,) in validation_loader:
            outputs.append(coerce_model_logits(model(batch_x), batch_size=int(batch_x.shape[0]), num_classes=num_classes))
    return torch.cat(outputs, dim=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())

    progress_path = Path(config["progress_path"])
    eval_dir = Path(config["eval_dir"])
    debug_output_path = Path(config["debug_output_path"])
    num_epochs = int(config["epochs"])
    random_seed = int(config["random_seed"])
    dataset_metadata = dict(config.get("dataset_metadata") or {})

    train_features, train_labels = read_split(config["train_split_path"])
    validation_features = read_split(config["validation_split_path"])
    validation_labels = read_labels(config["validation_labels_path"])
    if not isinstance(train_features, np.ndarray) or not isinstance(train_labels, np.ndarray):
        raise RuntimeError("Training split is invalid.")
    if not isinstance(validation_features, np.ndarray):
        raise RuntimeError("Validation split is invalid.")
    if not isinstance(validation_labels, np.ndarray):
        raise RuntimeError("Validation labels are invalid.")

    start_time = time.monotonic()
    eval_index = 0
    last_completed_eval_sec: float | None = None
    early_stopping_patience = 2 if num_epochs > 2 else 0
    best_validation_accuracy = -1.0
    epochs_without_improvement = 0
    epochs_completed = 0
    debug_payload: dict[str, object] = {
        "timed_out": False,
        "eval_count": 0,
        "early_stopped": False,
        "early_stopping_patience": early_stopping_patience,
        "epochs_completed": 0,
    }

    try:
        _ = seed_everything(random_seed)
        input_shape = infer_input_shape(train_features)
        num_classes = int(dataset_metadata.get("num_classes") or (np.max(train_labels) + 1))
        train_x = to_feature_tensor(train_features, input_shape=input_shape)
        validation_x = to_feature_tensor(validation_features, input_shape=input_shape)
        train_x, validation_x = normalize_feature_tensors(train_x, validation_x)
        train_y = torch.from_numpy(train_labels.astype(np.int64))
        build_model_fn = load_model_builder()
        model = build_model_fn(input_shape=input_shape, num_classes=num_classes)
        if not isinstance(model, torch.nn.Module):
            raise TrainScriptContractError("build_model must return a torch.nn.Module instance.")
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        batch_size = max(1, min(512, int(train_x.shape[0])))
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
        )
        optimizer = None
        scheduler = None
        if trainable_parameters:
            optimizer = torch.optim.AdamW(trainable_parameters, lr=0.002, weight_decay=1e-4)
            total_steps = max(1, int(num_epochs) * max(1, len(train_loader)))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.002,
                total_steps=total_steps,
                pct_start=0.2,
            )
        label_smoothing = 0.02 if num_classes > 2 else 0.0

        write_progress(
            progress_path,
            phase="train",
            elapsed_time_sec=0.0,
            last_completed_eval_sec=None,
            eval_index=eval_index,
            epoch_index=0,
        )

        for epoch_index in range(num_epochs):
            maybe_call_epoch_hook(model, epoch_index=epoch_index, num_epochs=num_epochs)
            write_progress(
                progress_path,
                phase="train",
                elapsed_time_sec=time.monotonic() - start_time,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                epoch_index=epoch_index,
            )
            model.train()
            for batch_x, batch_y in train_loader:
                logits = coerce_model_logits(model(batch_x), batch_size=int(batch_x.shape[0]), num_classes=num_classes)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(logits, batch_y, label_smoothing=label_smoothing)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
            write_progress(
                progress_path,
                phase="eval",
                elapsed_time_sec=time.monotonic() - start_time,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                epoch_index=epoch_index,
            )
            raw_predictions = run_validation(
                model,
                validation_x,
                batch_size=batch_size,
                num_classes=num_classes,
            )
            predictions = normalize_predictions(
                raw_predictions,
                num_examples=int(validation_features.shape[0]),
                num_classes=num_classes,
            )
            eval_index += 1
            elapsed_after_eval = time.monotonic() - start_time
            write_eval_atomic(
                eval_dir,
                eval_index=eval_index,
                predictions=predictions,
                elapsed_time_sec=elapsed_after_eval,
                epoch=epoch_index + 1,
            )
            validation_accuracy = float((predictions == validation_labels).mean())
            if validation_accuracy > best_validation_accuracy + 1e-9:
                best_validation_accuracy = validation_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            last_completed_eval_sec = elapsed_after_eval
            epochs_completed = epoch_index + 1
            debug_payload["eval_count"] = eval_index
            debug_payload["epochs_completed"] = epochs_completed
            debug_payload["best_validation_accuracy_seen"] = best_validation_accuracy
            debug_payload["epochs_without_improvement"] = epochs_without_improvement
            write_progress(
                progress_path,
                phase="train",
                elapsed_time_sec=elapsed_after_eval,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                epoch_index=epoch_index + 1,
            )
            if (
                early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
                and (epoch_index + 1) < num_epochs
            ):
                debug_payload["early_stopped"] = True
                debug_payload["early_stop_epoch"] = epoch_index + 1
                break

        write_progress(
            progress_path,
            phase="finished",
            elapsed_time_sec=time.monotonic() - start_time,
            last_completed_eval_sec=last_completed_eval_sec,
            eval_index=eval_index,
            epoch_index=epochs_completed,
        )
        write_json_atomic(debug_output_path, debug_payload)
        return 0
    except TrainScriptContractError as exc:
        debug_payload.update(
            {
                "failure_outcome": "eval_failed",
                "failure_reason": "train_script_contract_violation",
                "detail": str(exc),
            }
        )
        write_json_atomic(debug_output_path, debug_payload)
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

Avoid the failure modes seen in these recent negative trials:
No recent negative trials are available.
~~~
