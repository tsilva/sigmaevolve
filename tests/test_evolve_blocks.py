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
from sigmaevolve.train_script_blocks import build_model_block


def test_replace_evolve_block_payloads_rewrites_only_block_contents():
    source = build_baseline_train_script()
    updated = replace_evolve_block_payloads(
        source,
        [
            build_model_block(
                """
def forward(self, x):
    return torch.zeros((x.shape[0], 2), dtype=torch.float32)
""",
            )
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
