from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
        "metrics": {
            key: float(value)
            for key, value in (metrics or {}).items()
            if value is not None
        },
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
