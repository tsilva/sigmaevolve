from __future__ import annotations


def build_baseline_linear_classifier() -> str:
    return """import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_training_split(path: str):
    data = np.load(path)
    return data["features"].astype(np.float32), data["labels"].astype(np.int64)


def load_validation_inputs(path: str):
    data = np.load(path)
    return data["features"].astype(np.float32)


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
    for _ in range(25):
        optimizer.zero_grad()
        logits = model(train_tensor)
        loss = criterion(logits, labels_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = model(torch.from_numpy(val_x)).argmax(dim=1).cpu().numpy()

    np.savez(config["predictions_output_path"], predictions=predictions)
    debug_payload = {"input_dim": input_dim, "num_classes": num_classes}
    Path(config["debug_output_path"]).write_text(json.dumps(debug_payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
