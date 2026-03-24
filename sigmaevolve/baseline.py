from __future__ import annotations

from textwrap import dedent


DEFAULT_MODEL_BLOCK = dedent(
    """
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
    """
).strip("\n")


DEFAULT_DATA_BLOCK = dedent(
    """
    def configure_data(*, train_x, train_y, validation_x, random_seed):
        del random_seed
        batch_size = max(1, min(512, int(train_x.shape[0])))
        return {
            "batch_size": batch_size,
            "train_loader": torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_x, train_y),
                batch_size=batch_size,
                shuffle=True,
            ),
            "validation_loader": torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(validation_x),
                batch_size=batch_size,
                shuffle=False,
            ),
        }
    """
).strip("\n")


DEFAULT_OPTIMIZATION_BLOCK = dedent(
    """
    def configure_optimization(*, model, train_loader, num_epochs, num_classes):
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = scheduler = None
        if trainable_parameters:
            optimizer = torch.optim.AdamW(trainable_parameters, lr=0.002, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.002,
                total_steps=max(1, num_epochs * max(1, len(train_loader))),
                pct_start=0.2,
            )
        return {
            "trainable_parameters": trainable_parameters,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "label_smoothing": 0.02 if num_classes > 2 else 0.0,
            "grad_clip_norm": 1.0,
        }
    """
).strip("\n")


DEFAULT_TRAINING_POLICY_BLOCK = dedent(
    """
    def configure_training_policy(*, num_epochs):
        patience = 2 if num_epochs > 2 else 0
        return {
            "early_stopping_patience": patience,
        }
    """
).strip("\n")


def _wrap_evolve_block(payload: str) -> str:
    normalized_payload = payload.strip("\n")
    return f"# EVOLVE-BLOCK-START\n{normalized_payload}\n# EVOLVE-BLOCK-END\n"


def build_baseline_train_script() -> str:
    parts = [
        dedent(
            """
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


            def write_json_atomic(path, payload):
                path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = path.with_suffix(path.suffix + ".tmp")
                temp_path.write_text(json.dumps(payload, sort_keys=True))
                temp_path.replace(path)


            def write_eval_atomic(eval_dir, eval_index, predictions, elapsed_time_sec, epoch):
                eval_dir.mkdir(parents=True, exist_ok=True)
                temp_path = eval_dir / f".eval_{eval_index:04d}.tmp.npz"
                np.savez(
                    temp_path,
                    predictions=np.asarray(predictions, dtype=np.int64),
                    eval_index=np.array(eval_index, dtype=np.int64),
                    elapsed_time_sec=np.array(elapsed_time_sec, dtype=np.float64),
                    epoch=np.array(epoch, dtype=np.int64),
                )
                temp_path.replace(eval_dir / f"eval_{eval_index:04d}.npz")


            def seed_everything(seed):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)


            def read_split(path):
                payload = np.load(path)
                labels = payload["labels"].astype(np.int64) if "labels" in payload else None
                return payload["features"].astype(np.float32), labels


            def prepare_feature_tensor(features, *, input_shape=None):
                input_shape = input_shape or tuple(int(dim) for dim in features.shape[1:])
                if not input_shape:
                    raise TrainScriptContractError("training features must include at least one non-batch dimension")
                tensor = torch.from_numpy(features.astype(np.float32))
                if len(input_shape) == 2:
                    tensor = tensor.unsqueeze(1)
                return input_shape, tensor.contiguous()


            def normalize_feature_tensors(train_x, validation_x):
                if train_x.ndim <= 1:
                    raise TrainScriptContractError("feature tensors must be at least 2D including the batch axis")
                reduce_dims = (0,) if train_x.ndim == 2 else (0,) + tuple(range(2, train_x.ndim))
                mean = train_x.mean(dim=reduce_dims, keepdim=True)
                std = train_x.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1e-6)
                return (train_x - mean) / std, (validation_x - mean) / std


            def normalize_predictions(raw_predictions, *, num_examples, num_classes):
                array = raw_predictions.detach().cpu().numpy() if isinstance(raw_predictions, torch.Tensor) else np.asarray(raw_predictions)
                if array.ndim == 0:
                    raise TrainScriptContractError("model evaluation must return one prediction per validation example.")
                if array.shape[0] != num_examples:
                    raise TrainScriptContractError(f"model evaluation returned {array.shape[0]} predictions for {num_examples} validation examples.")
                if array.ndim == 1:
                    if not np.issubdtype(array.dtype, np.floating):
                        return array.astype(np.int64)
                    if num_classes != 2:
                        raise TrainScriptContractError("model evaluation returned a 1D float array for a non-binary task; return class ids or logits.")
                    finite = array[np.isfinite(array)]
                    threshold = 0.5 if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0 else 0.0
                    return (array >= threshold).astype(np.int64)
                reshaped = array.reshape(num_examples, -1)
                return (reshaped.reshape(num_examples) if reshaped.shape[1] <= 1 else reshaped.argmax(axis=1)).astype(np.int64)


            def coerce_model_logits(raw_output, *, batch_size, num_classes):
                logits = raw_output if isinstance(raw_output, torch.Tensor) else torch.as_tensor(raw_output, dtype=torch.float32)
                if logits.ndim == 1:
                    if batch_size == 1 and logits.shape[0] == num_classes:
                        return logits.reshape(1, num_classes)
                    if num_classes == 2 and logits.shape[0] == batch_size:
                        return torch.stack((-logits, logits), dim=1)
                if logits.ndim == 2 and logits.shape[0] == batch_size:
                    return logits
                raise TrainScriptContractError(f"model forward must return logits shaped [batch, num_classes], received {tuple(logits.shape)}")


            def require_callable(name):
                value = globals().get(name)
                if not callable(value):
                    raise TrainScriptContractError(f"train.py is missing required evolve-block callable: {name}")
                return value


            def require_mapping(name, value):
                if not isinstance(value, dict):
                    raise TrainScriptContractError(f"{name} must return a dict.")
                return value
            """
        ).lstrip(),
        _wrap_evolve_block(DEFAULT_MODEL_BLOCK),
        dedent(
            """
            def run_validation(model, validation_loader, *, num_classes):
                model.eval()
                predictions = []
                with torch.no_grad():
                    for (batch_x,) in validation_loader:
                        predictions.append(
                            coerce_model_logits(model(batch_x), batch_size=int(batch_x.shape[0]), num_classes=num_classes)
                        )
                if not predictions:
                    raise TrainScriptContractError("validation loader must yield at least one batch")
                return torch.cat(predictions, dim=0)
            """
        ),
        _wrap_evolve_block(DEFAULT_DATA_BLOCK),
        _wrap_evolve_block(DEFAULT_OPTIMIZATION_BLOCK),
        _wrap_evolve_block(DEFAULT_TRAINING_POLICY_BLOCK),
        dedent(
            """
            def main(argv=None):
                parser = argparse.ArgumentParser()
                parser.add_argument("--config", required=True)
                config = json.loads(Path(parser.parse_args(argv).config).read_text())
                progress_path = Path(config["progress_path"])
                eval_dir = Path(config["eval_dir"])
                debug_output_path = Path(config["debug_output_path"])
                num_epochs = int(config["epochs"])
                random_seed = int(config["random_seed"])
                dataset_metadata = dict(config.get("dataset_metadata") or {})

                train_features, train_labels = read_split(config["train_split_path"])
                validation_features, _ = read_split(config["validation_split_path"])
                validation_labels = np.load(config["validation_labels_path"]).astype(np.int64)
                if train_labels is None or not all(
                    isinstance(value, np.ndarray) for value in (train_features, train_labels, validation_features, validation_labels)
                ):
                    raise RuntimeError("Dataset splits are invalid.")

                start_time = time.monotonic()
                eval_index = epochs_completed = stale_epochs = 0
                last_eval_sec = None
                best_accuracy = -1.0
                debug_payload = {
                    "timed_out": False,
                    "eval_count": 0,
                    "early_stopped": False,
                    "early_stopping_patience": 0,
                    "epochs_completed": 0,
                }

                try:
                    seed_everything(random_seed)
                    input_shape, train_x = prepare_feature_tensor(train_features)
                    _, validation_x = prepare_feature_tensor(validation_features, input_shape=input_shape)
                    train_x, validation_x = normalize_feature_tensors(train_x, validation_x)
                    train_y = torch.from_numpy(train_labels.astype(np.int64))
                    num_classes = int(dataset_metadata.get("num_classes") or (np.max(train_labels) + 1))

                    model = require_callable("build_model")(input_shape=input_shape, num_classes=num_classes)
                    if not isinstance(model, torch.nn.Module):
                        raise TrainScriptContractError("build_model must return a torch.nn.Module instance.")

                    data_config = require_mapping(
                        "configure_data",
                        require_callable("configure_data")(
                            train_x=train_x,
                            train_y=train_y,
                            validation_x=validation_x,
                            random_seed=random_seed,
                        ),
                    )
                    train_loader = data_config.get("train_loader")
                    validation_loader = data_config.get("validation_loader")
                    if train_loader is None or validation_loader is None:
                        raise TrainScriptContractError("configure_data must return train_loader and validation_loader.")

                    optimization_config = require_mapping(
                        "configure_optimization",
                        require_callable("configure_optimization")(
                            model=model,
                            train_loader=train_loader,
                            num_epochs=num_epochs,
                            num_classes=num_classes,
                        ),
                    )
                    trainable_parameters = optimization_config.get("trainable_parameters")
                    if trainable_parameters is None:
                        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                    trainable_parameters = list(trainable_parameters)
                    optimizer = optimization_config.get("optimizer")
                    scheduler = optimization_config.get("scheduler")
                    label_smoothing = float(optimization_config.get("label_smoothing", 0.0))
                    grad_clip_norm = optimization_config.get("grad_clip_norm", 1.0)

                    training_policy = require_mapping(
                        "configure_training_policy",
                        require_callable("configure_training_policy")(num_epochs=num_epochs),
                    )
                    patience = int(training_policy.get("early_stopping_patience", 0))
                    debug_payload["early_stopping_patience"] = patience

                    def report(phase, *, elapsed_time_sec, epoch_index):
                        write_json_atomic(
                            progress_path,
                            {
                                "phase": phase,
                                "elapsed_time_sec": float(elapsed_time_sec),
                                "last_completed_eval_sec": last_eval_sec,
                                "eval_index": eval_index,
                                "epoch_index": epoch_index,
                            },
                        )

                    report("train", elapsed_time_sec=0.0, epoch_index=0)
                    for epoch_index in range(num_epochs):
                        hook = getattr(model, "on_epoch_start", None)
                        if callable(hook):
                            hook(epoch_index=epoch_index, num_epochs=num_epochs)
                        model.train()
                        report("train", elapsed_time_sec=time.monotonic() - start_time, epoch_index=epoch_index)
                        for batch_x, batch_y in train_loader:
                            logits = coerce_model_logits(model(batch_x), batch_size=int(batch_x.shape[0]), num_classes=num_classes)
                            if optimizer is None:
                                continue
                            optimizer.zero_grad(set_to_none=True)
                            torch.nn.functional.cross_entropy(logits, batch_y, label_smoothing=label_smoothing).backward()
                            if grad_clip_norm is not None and trainable_parameters:
                                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=float(grad_clip_norm))
                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()
                        report("eval", elapsed_time_sec=time.monotonic() - start_time, epoch_index=epoch_index)
                        predictions = normalize_predictions(
                            run_validation(model, validation_loader, num_classes=num_classes),
                            num_examples=int(validation_features.shape[0]),
                            num_classes=num_classes,
                        )
                        eval_index += 1
                        elapsed_after_eval = time.monotonic() - start_time
                        write_eval_atomic(eval_dir, eval_index, predictions, elapsed_after_eval, epoch_index + 1)
                        val_acc = float((predictions == validation_labels).mean())
                        stale_epochs = 0 if val_acc > best_accuracy + 1e-9 else stale_epochs + 1
                        best_accuracy = max(best_accuracy, val_acc)
                        last_eval_sec = elapsed_after_eval
                        epochs_completed = epoch_index + 1
                        debug_payload.update(
                            {
                                "eval_count": eval_index,
                                "epochs_completed": epochs_completed,
                                "best_validation_accuracy_seen": best_accuracy,
                                "epochs_without_improvement": stale_epochs,
                            }
                        )
                        report("train", elapsed_time_sec=elapsed_after_eval, epoch_index=epochs_completed)
                        if patience and stale_epochs >= patience and epochs_completed < num_epochs:
                            debug_payload["early_stopped"] = True
                            debug_payload["early_stop_epoch"] = epochs_completed
                            break
                    report("finished", elapsed_time_sec=time.monotonic() - start_time, epoch_index=epochs_completed)
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
            """
        ),
    ]
    return "".join(parts)


def build_baseline_linear_classifier() -> str:
    return build_baseline_train_script()
