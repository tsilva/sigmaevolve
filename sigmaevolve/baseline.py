from __future__ import annotations


def build_baseline_train_script() -> str:
    return """from __future__ import annotations

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


def normalize_predictions(raw_predictions: object, *, num_examples: int, num_classes: int | None) -> np.ndarray:
    if isinstance(raw_predictions, torch.Tensor):
        array = raw_predictions.detach().cpu().numpy()
    else:
        array = np.asarray(raw_predictions)
    if array.ndim == 0:
        raise TrainScriptContractError("predict_validation must return one prediction per validation example.")
    if array.shape[0] != num_examples:
        raise TrainScriptContractError(
            f"predict_validation returned {array.shape[0]} predictions for {num_examples} validation examples."
        )
    if array.ndim == 1:
        if np.issubdtype(array.dtype, np.floating):
            if num_classes == 2:
                finite = array[np.isfinite(array)]
                if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
                    return (array >= 0.5).astype(np.int64)
                return (array >= 0.0).astype(np.int64)
            raise TrainScriptContractError(
                "predict_validation returned a 1D float array for a non-binary task; return class ids or logits."
            )
        return array.astype(np.int64)
    reshaped = array.reshape(num_examples, -1)
    if reshaped.shape[1] <= 1:
        return reshaped.reshape(num_examples).astype(np.int64)
    return reshaped.argmax(axis=1).astype(np.int64)


# EVOLVE-BLOCK-START
def build_state(*, train_features, train_labels, validation_features, dataset_metadata, random_seed, device):
    train_x = train_features.reshape(train_features.shape[0], -1)
    val_x = validation_features.reshape(validation_features.shape[0], -1)
    train_y = train_labels.astype(np.int64)
    num_classes = int(dataset_metadata.get("num_classes") or (np.max(train_y) + 1))
    torch.manual_seed(int(random_seed))
    model = torch.nn.Linear(int(train_x.shape[1]), num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()
    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "train_x": torch.from_numpy(train_x),
        "train_y": torch.from_numpy(train_y),
        "val_x": torch.from_numpy(val_x),
        "steps_per_epoch": 5,
    }


def train_epoch(state, *, epoch_index, num_epochs):
    model = state["model"]
    optimizer = state["optimizer"]
    criterion = state["criterion"]
    train_x = state["train_x"]
    train_y = state["train_y"]
    for _ in range(int(state["steps_per_epoch"])):
        optimizer.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()


def predict_validation(state, validation_features):
    model = state["model"]
    val_x = state["val_x"]
    model.eval()
    with torch.no_grad():
        return model(val_x)


# EVOLVE-BLOCK-END


def load_evolvable_functions():
    build_state_fn = globals().get("build_state")
    train_epoch_fn = globals().get("train_epoch")
    predict_validation_fn = globals().get("predict_validation")
    missing = [
        name
        for name, value in (
            ("build_state", build_state_fn),
            ("train_epoch", train_epoch_fn),
            ("predict_validation", predict_validation_fn),
        )
        if not callable(value)
    ]
    if missing:
        raise TrainScriptContractError(
            f"train.py is missing required evolve-block callables: {', '.join(missing)}"
        )
    return build_state_fn, train_epoch_fn, predict_validation_fn


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
    if not isinstance(train_features, np.ndarray) or not isinstance(train_labels, np.ndarray):
        raise RuntimeError("Training split is invalid.")
    if not isinstance(validation_features, np.ndarray):
        raise RuntimeError("Validation split is invalid.")

    start_time = time.monotonic()
    eval_index = 0
    last_completed_eval_sec: float | None = None
    debug_payload: dict[str, object] = {"timed_out": False, "eval_count": 0}

    try:
        device = seed_everything(random_seed)
        build_state_fn, train_epoch_fn, predict_validation_fn = load_evolvable_functions()
        state = build_state_fn(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            dataset_metadata=dataset_metadata,
            random_seed=random_seed,
            device=device,
        )
        if not isinstance(state, dict):
            raise TrainScriptContractError("build_state must return a dict state object.")

        write_progress(
            progress_path,
            phase="train",
            elapsed_time_sec=0.0,
            last_completed_eval_sec=None,
            eval_index=eval_index,
            epoch_index=0,
        )

        for epoch_index in range(num_epochs):
            write_progress(
                progress_path,
                phase="train",
                elapsed_time_sec=time.monotonic() - start_time,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                epoch_index=epoch_index,
            )
            train_epoch_fn(state, epoch_index=epoch_index, num_epochs=num_epochs)
            write_progress(
                progress_path,
                phase="eval",
                elapsed_time_sec=time.monotonic() - start_time,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                epoch_index=epoch_index,
            )
            raw_predictions = predict_validation_fn(state, validation_features)
            predictions = normalize_predictions(
                raw_predictions,
                num_examples=int(validation_features.shape[0]),
                num_classes=int(dataset_metadata["num_classes"]) if "num_classes" in dataset_metadata else None,
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
            last_completed_eval_sec = elapsed_after_eval
            debug_payload["eval_count"] = eval_index
            write_progress(
                progress_path,
                phase="train",
                elapsed_time_sec=elapsed_after_eval,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                epoch_index=epoch_index + 1,
            )

        write_progress(
            progress_path,
            phase="finished",
            elapsed_time_sec=time.monotonic() - start_time,
            last_completed_eval_sec=last_completed_eval_sec,
            eval_index=eval_index,
            epoch_index=num_epochs,
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
"""


def build_baseline_linear_classifier() -> str:
    return build_baseline_train_script()
