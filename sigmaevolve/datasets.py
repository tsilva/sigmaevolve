from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from sigmaevolve.models import DatasetManifest, DatasetRecord


class DatasetProvider(Protocol):
    def materialize(self, dataset_id: str, output_dir: Path) -> DatasetManifest:
        ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(dataset_id: str, checksums: dict[str, str]) -> str:
    payload = json.dumps({"dataset_id": dataset_id, "checksums": checksums}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_training_split(path: Path, features: np.ndarray, labels: np.ndarray) -> None:
    np.savez(path, features=features, labels=labels)


def _write_input_split(path: Path, features: np.ndarray) -> None:
    np.savez(path, features=features)


def _write_labels(path: Path, labels: np.ndarray) -> None:
    np.save(path, labels)


@dataclass(frozen=True)
class ArrayDatasetProvider:
    train_features: np.ndarray
    train_labels: np.ndarray
    validation_features: np.ndarray
    validation_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    metadata: dict[str, object] | None = None

    def materialize(self, dataset_id: str, output_dir: Path) -> DatasetManifest:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        train_path = output_dir / "train.npz"
        val_path = output_dir / "validation_inputs.npz"
        val_labels_path = output_dir / "validation_labels.npy"
        test_path = output_dir / "test_inputs.npz"
        test_labels_path = output_dir / "test_labels.npy"

        _write_training_split(train_path, self.train_features, self.train_labels)
        _write_input_split(val_path, self.validation_features)
        _write_labels(val_labels_path, self.validation_labels)
        _write_input_split(test_path, self.test_features)
        _write_labels(test_labels_path, self.test_labels)

        checksums = {
            "train_split": _sha256_file(train_path),
            "validation_split": _sha256_file(val_path),
            "validation_labels": _sha256_file(val_labels_path),
            "test_split": _sha256_file(test_path),
            "test_labels": _sha256_file(test_labels_path),
        }
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            root_dir=str(output_dir),
            train_split_path=str(train_path),
            validation_split_path=str(val_path),
            validation_labels_path=str(val_labels_path),
            test_split_path=str(test_path),
            test_labels_path=str(test_labels_path),
            split_sizes={
                "train": int(self.train_features.shape[0]),
                "validation": int(self.validation_features.shape[0]),
                "test": int(self.test_features.shape[0]),
            },
            checksums=checksums,
            fingerprint=_fingerprint(dataset_id, checksums),
            metadata=dict(self.metadata or {}),
        )
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return manifest


class TorchvisionClassificationProvider:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    def materialize(self, dataset_id: str, output_dir: Path) -> DatasetManifest:
        try:
            from torchvision import datasets
        except ImportError as exc:
            raise RuntimeError("torchvision is required to materialize torchvision datasets.") from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        dataset_cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST}[self.dataset_name]

        train_ds = dataset_cls(root=output_dir / "downloads", train=True, download=True)
        test_ds = dataset_cls(root=output_dir / "downloads", train=False, download=True)

        train_features = np.asarray(train_ds.data, dtype=np.float32) / 255.0
        train_labels = np.asarray(train_ds.targets, dtype=np.int64)
        test_features = np.asarray(test_ds.data, dtype=np.float32) / 255.0
        test_labels = np.asarray(test_ds.targets, dtype=np.int64)

        validation_size = 5000
        validation_features = train_features[-validation_size:]
        validation_labels = train_labels[-validation_size:]
        train_features = train_features[:-validation_size]
        train_labels = train_labels[:-validation_size]

        provider = ArrayDatasetProvider(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            test_features=test_features,
            test_labels=test_labels,
            metadata={"source": self.dataset_name, "num_classes": 10},
        )
        manifest = provider.materialize(dataset_id, output_dir)
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return manifest


class DatasetManager:
    def __init__(self, dataset_root: Path, providers: dict[str, DatasetProvider]) -> None:
        self.dataset_root = dataset_root
        self.providers = providers

    @staticmethod
    def safe_dir_name(dataset_id: str) -> str:
        return dataset_id.replace(":", "__")

    def manifest_path_for(self, dataset_id: str) -> Path:
        return self.dataset_root / self.safe_dir_name(dataset_id) / "manifest.json"

    def prepare(self, dataset_id: str) -> DatasetManifest:
        provider = self.providers.get(dataset_id)
        if provider is None:
            raise KeyError(f"No dataset provider registered for {dataset_id!r}.")
        dataset_dir = self.manifest_path_for(dataset_id).parent
        manifest = provider.materialize(dataset_id, dataset_dir)
        self.manifest_path_for(dataset_id).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return manifest

    def load_manifest(self, dataset_id: str) -> DatasetManifest:
        manifest_path = self.manifest_path_for(dataset_id)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found for {dataset_id!r}: {manifest_path}")
        return DatasetManifest.from_dict(json.loads(manifest_path.read_text()))

    def verify(self, dataset_id: str) -> DatasetManifest:
        manifest = self.load_manifest(dataset_id)
        paths = {
            "train_split": Path(manifest.train_split_path),
            "validation_split": Path(manifest.validation_split_path),
            "validation_labels": Path(manifest.validation_labels_path),
            "test_split": Path(manifest.test_split_path),
            "test_labels": Path(manifest.test_labels_path),
        }
        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Dataset artifact missing for {dataset_id!r}: {path}")
            checksum = _sha256_file(path)
            expected = manifest.checksums[key]
            if checksum != expected:
                raise ValueError(f"Checksum mismatch for {key} in {dataset_id!r}: {checksum} != {expected}")
        fingerprint = _fingerprint(dataset_id, manifest.checksums)
        if fingerprint != manifest.fingerprint:
            raise ValueError(f"Fingerprint mismatch for {dataset_id!r}.")
        return manifest

    def to_record(self, dataset_id: str) -> DatasetRecord:
        manifest_path = self.manifest_path_for(dataset_id)
        return DatasetRecord(dataset_id=dataset_id, manifest_path=str(manifest_path), created_at=None)  # type: ignore[arg-type]
