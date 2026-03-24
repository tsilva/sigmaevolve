from sigmaevolve.baseline import build_baseline_train_script
from sigmaevolve.evolve_blocks import (
    EvolveBlockError,
    apply_search_replace_blocks,
    assert_only_evolve_blocks_changed,
    extract_evolve_block_payloads,
    materialize_candidate_source,
    parse_search_replace_blocks,
    replace_evolve_block_payloads,
)
from sigmaevolve.train_script_blocks import (
    build_candidate_train_script,
    build_config_block,
    build_data_block,
    build_model_block,
    build_optimization_block,
)


def test_replace_evolve_block_payloads_rewrites_only_block_contents():
    source = build_baseline_train_script()
    payloads = extract_evolve_block_payloads(source)
    updated = replace_evolve_block_payloads(
        source,
        [
            payloads[0],
            build_model_block(
                """
def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
            ),
            payloads[2],
            payloads[3],
            payloads[4],
        ],
    )

    assert len(payloads) == 5
    assert extract_evolve_block_payloads(updated) != payloads
    assert_only_evolve_blocks_changed(source, updated)


def test_build_candidate_train_script_replaces_only_data_block():
    source = build_baseline_train_script()
    source_payloads = extract_evolve_block_payloads(source)
    updated = build_candidate_train_script(
        data_block_payload=build_data_block(
            """
batch_size = 8
return {
    "batch_size": batch_size,
    "train_loader": torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=False,
    ),
    "validation_loader": torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(validation_x),
        batch_size=1,
        shuffle=False,
    ),
}
"""
        )
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert updated_payloads[0] == source_payloads[0]
    assert updated_payloads[1] == source_payloads[1]
    assert updated_payloads[3] == source_payloads[3]
    assert updated_payloads[4] == source_payloads[4]
    assert "batch_size = 8" in updated_payloads[2]
    assert_only_evolve_blocks_changed(source, updated)


def test_build_candidate_train_script_replaces_only_optimization_block():
    source = build_baseline_train_script()
    source_payloads = extract_evolve_block_payloads(source)
    updated = build_candidate_train_script(
        optimization_block_payload=build_optimization_block(
            """
return {
    "trainable_parameters": [parameter for parameter in model.parameters() if parameter.requires_grad],
    "optimizer": None,
    "scheduler": None,
    "label_smoothing": 0.0,
    "grad_clip_norm": None,
}
"""
        )
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert updated_payloads[0] == source_payloads[0]
    assert updated_payloads[1] == source_payloads[1]
    assert updated_payloads[2] == source_payloads[2]
    assert updated_payloads[4] == source_payloads[4]
    assert '"optimizer": None' in updated_payloads[3]
    assert_only_evolve_blocks_changed(source, updated)


def test_build_candidate_train_script_positional_model_payload_keeps_other_blocks():
    source = build_baseline_train_script()
    source_payloads = extract_evolve_block_payloads(source)
    updated = build_candidate_train_script(
        build_model_block(
            """
def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
        )
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert updated_payloads[0] == source_payloads[0]
    assert "return torch.zeros((x.shape[0], 2), dtype=torch.float32)" in updated_payloads[1]
    assert updated_payloads[2:] == source_payloads[2:]
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


def test_materialize_candidate_source_applies_search_replace_blocks():
    source = build_baseline_train_script()
    response = """<<<<<<< SEARCH
    def forward(self, x):
        return self.network(x)
=======
    def forward(self, x):
        return self.network(x) * 0.5
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert "return self.network(x) * 0.5" in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_materialize_candidate_source_matches_search_blocks_without_outer_indentation():
    source = build_baseline_train_script()
    response = """<<<<<<< SEARCH
    "learning_rate": 0.002,
    "weight_decay": 1e-4,
    "scheduler_max_lr": 0.002,
    "scheduler_pct_start": 0.2,
=======
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "scheduler_max_lr": 0.005,
    "scheduler_pct_start": 0.25,
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert '"learning_rate": 0.001' in updated
    assert '        "learning_rate": 0.001,' in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_build_candidate_train_script_replaces_only_config_block():
    source = build_baseline_train_script()
    source_payloads = extract_evolve_block_payloads(source)
    updated = build_candidate_train_script(
        config_block_payload=build_config_block(
            """
CONFIG = {
    "normalization_std_floor": 1e-5,
    "binary_probability_threshold": 0.55,
    "binary_logit_threshold": 0.1,
    "initial_best_accuracy": -1.0,
    "accuracy_improvement_tol": 1e-8,
    "model": {
        "mlp_hidden_dims": (256, 128),
        "cnn_channels": (24, 48),
        "cnn_kernel_sizes": (5, 3),
        "cnn_paddings": (2, 1),
        "cnn_pool_kernel_size": 2,
        "cnn_adaptive_pool_size": (4, 4),
        "cnn_projection_dim": 64,
        "dropout_p": 0.1,
    },
    "data": {
        "max_batch_size": 512,
        "shuffle_train": True,
        "shuffle_validation": False,
    },
    "optimization": {
        "learning_rate": 0.002,
        "weight_decay": 1e-4,
        "scheduler_max_lr": 0.002,
        "scheduler_pct_start": 0.2,
        "label_smoothing_multiclass": 0.02,
        "label_smoothing_binary": 0.0,
        "grad_clip_norm": 1.0,
    },
    "training_policy": {
        "patience_threshold_epochs": 2,
        "early_stopping_patience": 2,
        "short_run_patience": 0,
    },
}
"""
        )
    )

    updated_payloads = extract_evolve_block_payloads(updated)
    assert '"binary_probability_threshold": 0.55' in updated_payloads[0]
    assert updated_payloads[1:] == source_payloads[1:]
    assert_only_evolve_blocks_changed(source, updated)


def test_apply_search_replace_blocks_preserves_internal_indentation():
    source = build_baseline_train_script()
    response = """<<<<<<< SEARCH
def configure_training_policy(*, num_epochs):
    training_policy = CONFIG["training_policy"]
    patience = (
        training_policy["early_stopping_patience"]
        if num_epochs > training_policy["patience_threshold_epochs"]
        else training_policy["short_run_patience"]
    )
    return {
        "early_stopping_patience": patience,
    }
=======
def configure_training_policy(*, num_epochs):
    training_policy = CONFIG["training_policy"]
    patience = (
        training_policy["early_stopping_patience"] + 3
        if num_epochs > 5
        else max(1, num_epochs // 2)
    )
    return {
        "early_stopping_patience": patience,
    }
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert '        training_policy["early_stopping_patience"] + 3' in updated
    assert '        "early_stopping_patience": patience,' in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_parse_and_apply_search_replace_blocks_support_no_changes():
    source = build_baseline_train_script()

    blocks = parse_search_replace_blocks("NO_CHANGES")
    updated = apply_search_replace_blocks(source, blocks)

    assert updated == source


def test_parse_search_replace_blocks_rejects_evolve_markers_in_search_text():
    response = """<<<<<<< SEARCH
# EVOLVE-BLOCK-START
=======
replacement
>>>>>>> REPLACE
"""

    try:
        parse_search_replace_blocks(response)
    except EvolveBlockError as exc:
        assert "may not include evolve block marker lines" in str(exc)
    else:
        raise AssertionError("expected evolve block markers in SEARCH text to be rejected")


def test_parse_search_replace_blocks_rejects_evolve_markers_in_replace_text():
    response = """<<<<<<< SEARCH
original
=======
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
"""

    try:
        parse_search_replace_blocks(response)
    except EvolveBlockError as exc:
        assert "may not include evolve block marker lines" in str(exc)
    else:
        raise AssertionError("expected evolve block markers in REPLACE text to be rejected")


def test_materialize_candidate_source_rejects_non_patch_without_full_program():
    source = build_baseline_train_script()

    try:
        materialize_candidate_source(source, "return self.network(x)\n")
    except EvolveBlockError as exc:
        assert "SEARCH/REPLACE blocks" in str(exc)
    else:
        raise AssertionError("expected invalid generated response to be rejected")
