from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


class StrategyContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class StrategyContext:
    train_features: np.ndarray
    train_labels: np.ndarray
    validation_features: np.ndarray
    dataset_metadata: dict[str, Any]
    random_seed: int
    device: str
    budget_sec: float
    remaining_budget_sec: float
    max_eval_gap_sec: float
    window_index: int


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, sort_keys=True))
    temp_path.replace(path)


def write_eval_atomic(
    eval_dir: Path,
    eval_index: int,
    predictions: np.ndarray,
    elapsed_time_sec: float,
    epoch: int | None,
) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    final_path = eval_dir / f"eval_{eval_index:04d}.npz"
    temp_path = eval_dir / f".eval_{eval_index:04d}.tmp.npz"
    payload: dict[str, Any] = {
        "predictions": np.asarray(predictions, dtype=np.int64),
        "eval_index": np.array(eval_index, dtype=np.int64),
        "elapsed_time_sec": np.array(elapsed_time_sec, dtype=np.float64),
    }
    if epoch is not None:
        payload["epoch"] = np.array(epoch, dtype=np.int64)
    np.savez(temp_path, **payload)
    temp_path.replace(final_path)
    return final_path


def _load_strategy_module(strategy_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("sigmaevolve_candidate_strategy", strategy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load strategy module from {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_strategy(strategy_path: Path) -> tuple[Any, Any, Any]:
    module = _load_strategy_module(strategy_path)
    initialize = getattr(module, "initialize", None)
    train_window = getattr(module, "train_window", None)
    predict_validation = getattr(module, "predict_validation", None)
    missing = [
        name
        for name, value in (
            ("initialize", initialize),
            ("train_window", train_window),
            ("predict_validation", predict_validation),
        )
        if not callable(value)
    ]
    if missing:
        raise StrategyContractError(f"Strategy is missing required callable exports: {', '.join(missing)}")
    return initialize, train_window, predict_validation


def _normalize_predictions(
    raw_predictions: Any,
    *,
    num_examples: int,
    num_classes: int | None,
) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is expected but keep loader tolerant.
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(raw_predictions, torch.Tensor):
        array = raw_predictions.detach().cpu().numpy()
    else:
        array = np.asarray(raw_predictions)

    if array.ndim == 0:
        raise StrategyContractError("predict_validation must return one prediction per validation example.")

    if array.shape[0] != num_examples:
        raise StrategyContractError(
            f"predict_validation returned {array.shape[0]} predictions for {num_examples} validation examples."
        )

    if array.ndim == 1:
        if np.issubdtype(array.dtype, np.floating):
            if num_classes == 2:
                finite = array[np.isfinite(array)]
                if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
                    return (array >= 0.5).astype(np.int64)
                return (array >= 0.0).astype(np.int64)
            raise StrategyContractError(
                "predict_validation returned a 1D float array for a non-binary task; return class ids or logits."
            )
        return array.astype(np.int64)

    reshaped = array.reshape(num_examples, -1)
    if reshaped.shape[1] <= 1:
        return reshaped.reshape(num_examples).astype(np.int64)
    return reshaped.argmax(axis=1).astype(np.int64)


def _seed_everything(seed: int) -> str:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return "cpu"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return "cpu"


def _read_split(path: str) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    payload = np.load(path)
    features = payload["features"].astype(np.float32)
    if "labels" in payload:
        return features, payload["labels"].astype(np.int64)
    return features


def _build_context(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    dataset_metadata: dict[str, Any],
    random_seed: int,
    device: str,
    budget_sec: float,
    start_time: float,
    max_eval_gap_sec: float,
    window_index: int,
) -> StrategyContext:
    elapsed = time.monotonic() - start_time
    remaining = max(0.0, float(budget_sec) - float(elapsed))
    return StrategyContext(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        dataset_metadata=dict(dataset_metadata),
        random_seed=int(random_seed),
        device=device,
        budget_sec=float(budget_sec),
        remaining_budget_sec=remaining,
        max_eval_gap_sec=float(max_eval_gap_sec),
        window_index=window_index,
    )


def _write_progress(
    progress_path: Path,
    *,
    phase: str,
    elapsed_time_sec: float,
    last_completed_eval_sec: float | None,
    eval_index: int,
    window_index: int,
) -> None:
    write_json_atomic(
        progress_path,
        {
            "phase": phase,
            "elapsed_time_sec": float(elapsed_time_sec),
            "last_completed_eval_sec": last_completed_eval_sec,
            "eval_index": eval_index,
            "window_index": window_index,
        },
    )


def _run_harness(config: dict[str, Any]) -> int:
    strategy_path = Path(config["strategy_path"])
    progress_path = Path(config["progress_path"])
    eval_dir = Path(config["eval_dir"])
    debug_output_path = Path(config["debug_output_path"])
    budget_sec = float(config["budget_sec"])
    max_eval_gap_sec = float(config["max_eval_gap_sec"])
    random_seed = int(config["random_seed"])
    dataset_metadata = dict(config.get("dataset_metadata") or {})

    train_features, train_labels = _read_split(config["train_split_path"])
    validation_features = _read_split(config["validation_split_path"])
    if not isinstance(train_features, np.ndarray) or not isinstance(train_labels, np.ndarray):
        raise RuntimeError("Training split is invalid.")
    if not isinstance(validation_features, np.ndarray):
        raise RuntimeError("Validation split is invalid.")

    device = _seed_everything(random_seed)
    start_time = time.monotonic()
    eval_index = 0
    last_completed_eval_sec: float | None = None
    debug_payload: dict[str, Any] = {"timed_out": False, "eval_count": 0}

    try:
        initialize, train_window, predict_validation = load_strategy(strategy_path)
        init_ctx = _build_context(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            dataset_metadata=dataset_metadata,
            random_seed=random_seed,
            device=device,
            budget_sec=budget_sec,
            start_time=start_time,
            max_eval_gap_sec=max_eval_gap_sec,
            window_index=0,
        )
        state = initialize(init_ctx)
        if not isinstance(state, dict):
            raise StrategyContractError("initialize must return a dict state object.")

        window_index = 0
        _write_progress(
            progress_path,
            phase="train",
            elapsed_time_sec=0.0,
            last_completed_eval_sec=None,
            eval_index=eval_index,
            window_index=window_index,
        )

        while True:
            elapsed_before = time.monotonic() - start_time
            if elapsed_before >= budget_sec:
                debug_payload["timed_out"] = True
                break

            window_ctx = _build_context(
                train_features=train_features,
                train_labels=train_labels,
                validation_features=validation_features,
                dataset_metadata=dataset_metadata,
                random_seed=random_seed,
                device=device,
                budget_sec=budget_sec,
                start_time=start_time,
                max_eval_gap_sec=max_eval_gap_sec,
                window_index=window_index,
            )
            _write_progress(
                progress_path,
                phase="train",
                elapsed_time_sec=elapsed_before,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                window_index=window_index,
            )

            window_started = time.monotonic()
            train_window(window_ctx, state)
            window_elapsed = time.monotonic() - window_started
            if window_elapsed > max_eval_gap_sec:
                raise StrategyContractError(
                    f"train_window exceeded max_eval_gap_sec ({window_elapsed:.3f}s > {max_eval_gap_sec:.3f}s)."
                )

            elapsed_after_train = time.monotonic() - start_time
            if elapsed_after_train > budget_sec:
                debug_payload["timed_out"] = True
                break

            predict_ctx = _build_context(
                train_features=train_features,
                train_labels=train_labels,
                validation_features=validation_features,
                dataset_metadata=dataset_metadata,
                random_seed=random_seed,
                device=device,
                budget_sec=budget_sec,
                start_time=start_time,
                max_eval_gap_sec=max_eval_gap_sec,
                window_index=window_index,
            )
            _write_progress(
                progress_path,
                phase="eval",
                elapsed_time_sec=elapsed_after_train,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                window_index=window_index,
            )
            raw_predictions = predict_validation(predict_ctx, state)
            predictions = _normalize_predictions(
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
                epoch=window_index,
            )
            last_completed_eval_sec = elapsed_after_eval
            debug_payload["eval_count"] = eval_index
            _write_progress(
                progress_path,
                phase="train",
                elapsed_time_sec=elapsed_after_eval,
                last_completed_eval_sec=last_completed_eval_sec,
                eval_index=eval_index,
                window_index=window_index,
            )

            if state.get("done"):
                break

            if elapsed_after_eval >= budget_sec:
                debug_payload["timed_out"] = True
                break

            window_index += 1

        _write_progress(
            progress_path,
            phase="finished" if not debug_payload["timed_out"] else "train",
            elapsed_time_sec=time.monotonic() - start_time,
            last_completed_eval_sec=last_completed_eval_sec,
            eval_index=eval_index,
            window_index=window_index,
        )
        write_json_atomic(debug_output_path, debug_payload)
        return 0
    except StrategyContractError as exc:
        debug_payload.update(
            {
                "failure_outcome": "eval_failed",
                "failure_reason": "eval_window_missed" if "max_eval_gap_sec" in str(exc) else "strategy_contract_violation",
                "detail": str(exc),
            }
        )
        write_json_atomic(debug_output_path, debug_payload)
        print(str(exc), file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    return _run_harness(config)


if __name__ == "__main__":
    raise SystemExit(main())
