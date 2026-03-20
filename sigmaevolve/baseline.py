from __future__ import annotations


def build_baseline_linear_classifier() -> str:
    return """import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def load_training_split(path: str):
    data = np.load(path)
    return data["features"].astype(np.float32), data["labels"].astype(np.int64)


def load_validation_inputs(path: str):
    data = np.load(path)
    return data["features"].astype(np.float32)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, sort_keys=True))
    temp_path.replace(path)


def write_eval_atomic(eval_dir: Path, eval_index: int, predictions, elapsed_time_sec: float, epoch: int) -> Path:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    train_x, train_y = load_training_split(config["train_split_path"])
    val_x = load_validation_inputs(config["validation_split_path"])

    train_x = train_x.reshape(train_x.shape[0], -1)
    val_x = val_x.reshape(val_x.shape[0], -1)
    input_dim = int(train_x.shape[1])
    num_classes = int(np.max(train_y)) + 1

    torch.manual_seed(int(config["random_seed"]))
    model = torch.nn.Linear(input_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()

    train_tensor = torch.from_numpy(train_x)
    labels_tensor = torch.from_numpy(train_y)
    val_tensor = torch.from_numpy(val_x)
    start_time = time.monotonic()
    progress_path = Path(config["progress_path"])
    eval_dir = Path(config["eval_dir"])
    best_predictions = None
    last_completed_eval_sec = None

    def progress(phase: str, step: int, eval_index: int) -> None:
        elapsed = time.monotonic() - start_time
        payload = {
            "phase": phase,
            "step": step,
            "elapsed_time_sec": elapsed,
            "last_completed_eval_sec": last_completed_eval_sec,
            "eval_index": eval_index,
        }
        write_json_atomic(progress_path, payload)

    def run_eval(eval_index: int, step: int) -> None:
        nonlocal best_predictions, last_completed_eval_sec
        progress("eval", step, eval_index)
        with torch.no_grad():
            predictions = model(val_tensor).argmax(dim=1).cpu().numpy()
        elapsed = time.monotonic() - start_time
        write_eval_atomic(eval_dir, eval_index, predictions, elapsed, epoch=step)
        last_completed_eval_sec = elapsed
        best_predictions = predictions
        progress("train", step, eval_index)

    eval_index = 0
    progress("train", 0, eval_index)
    for step in range(1, 26):
        optimizer.zero_grad()
        logits = model(train_tensor)
        loss = criterion(logits, labels_tensor)
        loss.backward()
        optimizer.step()
        progress("train", step, eval_index)
        if step % 5 == 0:
            eval_index += 1
            run_eval(eval_index, step)

    if best_predictions is None:
        eval_index += 1
        run_eval(eval_index, 25)

    np.savez(config["predictions_output_path"], predictions=np.asarray(best_predictions, dtype=np.int64))
    debug_payload = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "eval_count": eval_index,
    }
    Path(config["debug_output_path"]).write_text(json.dumps(debug_payload, sort_keys=True))
    progress("finished", 25, eval_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
