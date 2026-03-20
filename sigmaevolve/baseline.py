from __future__ import annotations


def build_baseline_linear_classifier() -> str:
    return """import numpy as np
import torch


def initialize(ctx):
    train_x = ctx.train_features.reshape(ctx.train_features.shape[0], -1)
    val_x = ctx.validation_features.reshape(ctx.validation_features.shape[0], -1)
    train_y = ctx.train_labels.astype(np.int64)
    num_classes = int(ctx.dataset_metadata.get("num_classes") or (np.max(train_y) + 1))
    torch.manual_seed(int(ctx.random_seed))
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
        "steps_per_window": 5,
        "window_count": 0,
    }


def train_window(ctx, state):
    model = state["model"]
    optimizer = state["optimizer"]
    criterion = state["criterion"]
    train_x = state["train_x"]
    train_y = state["train_y"]
    state["window_count"] += 1
    for _ in range(int(state["steps_per_window"])):
        optimizer.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()
    if state["window_count"] >= 5:
        state["done"] = True


def predict_validation(ctx, state):
    model = state["model"]
    val_x = state["val_x"]
    with torch.no_grad():
        return model(val_x)
"""
