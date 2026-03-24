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
            build_model_block(
                """
def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
            ),
            payloads[1],
            payloads[2],
            payloads[3],
        ],
    )

    assert len(payloads) == 4
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
    assert updated_payloads[2] == source_payloads[2]
    assert updated_payloads[3] == source_payloads[3]
    assert "batch_size = 8" in updated_payloads[1]
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
    assert updated_payloads[3] == source_payloads[3]
    assert '"optimizer": None' in updated_payloads[2]
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
    assert "return torch.zeros((x.shape[0], 2), dtype=torch.float32)" in updated_payloads[0]
    assert updated_payloads[1:] == source_payloads[1:]
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
optimizer = torch.optim.AdamW(trainable_parameters, lr=0.002, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.002,
    total_steps=max(1, num_epochs * max(1, len(train_loader))),
    pct_start=0.2,
)
=======
optimizer = torch.optim.AdamW(trainable_parameters, lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.005,
    total_steps=max(1, num_epochs * max(1, len(train_loader))),
    pct_start=0.25,
)
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert "optimizer = torch.optim.AdamW(trainable_parameters, lr=0.001, weight_decay=1e-5)" in updated
    assert "        optimizer = torch.optim.AdamW(trainable_parameters, lr=0.001, weight_decay=1e-5)" in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_apply_search_replace_blocks_preserves_internal_indentation():
    source = build_baseline_train_script()
    response = """<<<<<<< SEARCH
def configure_training_policy(*, num_epochs):
    patience = 2 if num_epochs > 2 else 0
    return {
        "early_stopping_patience": patience,
    }
=======
def configure_training_policy(*, num_epochs):
    patience = 5 if num_epochs > 5 else max(1, num_epochs // 2)
    return {
        "early_stopping_patience": patience,
    }
>>>>>>> REPLACE
"""

    updated = materialize_candidate_source(source, response)

    assert '    patience = 5 if num_epochs > 5 else max(1, num_epochs // 2)' in updated
    assert '        "early_stopping_patience": patience,' in updated
    assert_only_evolve_blocks_changed(source, updated)


def test_parse_and_apply_search_replace_blocks_support_no_changes():
    source = build_baseline_train_script()

    blocks = parse_search_replace_blocks("NO_CHANGES")
    updated = apply_search_replace_blocks(source, blocks)

    assert updated == source


def test_materialize_candidate_source_rejects_non_patch_without_full_program():
    source = build_baseline_train_script()

    try:
        materialize_candidate_source(source, "return self.network(x)\n")
    except EvolveBlockError as exc:
        assert "SEARCH/REPLACE blocks" in str(exc)
    else:
        raise AssertionError("expected invalid generated response to be rejected")
