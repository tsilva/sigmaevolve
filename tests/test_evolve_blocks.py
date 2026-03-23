from sigmaevolve.baseline import build_baseline_train_script
from sigmaevolve.evolve_blocks import (
    EvolveBlockError,
    assert_only_evolve_blocks_changed,
    extract_evolve_block_payloads,
    replace_evolve_block_payloads,
)


def test_replace_evolve_block_payloads_rewrites_only_block_contents():
    source = build_baseline_train_script()
    updated = replace_evolve_block_payloads(
        source,
        [
            "def build_state(*, train_features, train_labels, validation_features, dataset_metadata, random_seed, device):\n"
            "    return {}\n\n"
            "def train_epoch(state, *, epoch_index, num_epochs):\n"
            "    return None\n\n"
            "def predict_validation(state, validation_features):\n"
            "    return [0] * validation_features.shape[0]\n"
        ],
    )

    assert len(extract_evolve_block_payloads(source)) == 1
    assert extract_evolve_block_payloads(updated) != extract_evolve_block_payloads(source)
    assert_only_evolve_blocks_changed(source, updated)


def test_assert_only_evolve_blocks_changed_rejects_immutable_changes():
    source = build_baseline_train_script()
    invalid = source.replace("import json\n", "import json\nBROKEN = True\n", 1)

    try:
        assert_only_evolve_blocks_changed(source, invalid)
    except EvolveBlockError as exc:
        assert "immutable text" in str(exc)
    else:
        raise AssertionError("expected immutable change to be rejected")
