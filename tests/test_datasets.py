from __future__ import annotations

from pathlib import Path

import pytest


def test_prepare_and_verify_multiple_datasets(system, dataset_manager):
    # Prepare two datasets so the verification path covers multiple manifests.
    mnist_record = system.prepare_dataset("mnist:v1")
    fashion_record = system.prepare_dataset("fashion_mnist:v1")

    # Verify both dataset manifests through the manager API.
    mnist_manifest = dataset_manager.verify("mnist:v1")
    fashion_manifest = dataset_manager.verify("fashion_mnist:v1")

    assert mnist_record.dataset_id == "mnist:v1"
    assert fashion_record.dataset_id == "fashion_mnist:v1"
    assert mnist_manifest.split_sizes["train"] == 12
    assert fashion_manifest.split_sizes["validation"] == 6

    # Reload the manifest to confirm the persisted fingerprint is stable.
    reloaded = dataset_manager.load_manifest("mnist:v1")
    assert reloaded.fingerprint == mnist_manifest.fingerprint


def test_runner_side_verification_detects_tampering(system, dataset_manager):
    # Create a manifest, then corrupt the validation split on disk.
    system.prepare_dataset("mnist:v1")
    manifest = dataset_manager.load_manifest("mnist:v1")
    validation_split_path = Path(manifest.validation_split_path)
    validation_split_path.write_bytes(b"corrupted")

    # Verification should reject the tampered artifact.
    with pytest.raises(ValueError):
        dataset_manager.verify("mnist:v1")
