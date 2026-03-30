from __future__ import annotations

import pytest

from sigmaevolve.generation import build_baseline_train_script
from sigmaevolve.script_spec import (
    DEFAULT_OBJECTIVE,
    PYTHON_TRAIN_RUNNER,
    ScriptSpecError,
    apply_script_policy_defaults,
    parse_script_spec,
    parse_source_layout,
    require_script_spec,
)
from tests.support import build_selfcontained_train_script


def test_parse_script_spec_reads_bundled_baseline():
    spec = require_script_spec(build_baseline_train_script())

    assert spec.dataset_id == "mnist:v1"
    assert spec.runner == PYTHON_TRAIN_RUNNER
    assert spec.defaults == {}
    assert spec.track_policy["epochs"] == 20


def test_build_baseline_train_script_accepts_path(tmp_path):
    baseline_path = tmp_path / "custom_mnist.py"
    baseline_path.write_text(build_selfcontained_train_script(epochs=7))

    source = build_baseline_train_script(baseline_path)
    spec = require_script_spec(source)

    assert spec.defaults == {"epochs": 7}


def test_parse_script_spec_reads_comment_prefixed_toml():
    source = build_selfcontained_train_script(
        epochs=7,
        track_policy={"dispatch_ttl_sec": 180},
    )

    spec = require_script_spec(source)

    assert spec.version == 1
    assert spec.dataset_id == "mnist:v1"
    assert spec.runner == PYTHON_TRAIN_RUNNER
    assert spec.defaults == {"epochs": 7}
    assert spec.track_policy == {"dispatch_ttl_sec": 180}
    assert spec.evolution.task == (
        "Maximize validation accuracy while keeping the script runnable."
    )


def test_parse_script_spec_allows_shebang_and_encoding_comment():
    source = build_selfcontained_train_script()
    source = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n" + source

    spec = require_script_spec(source)

    assert spec.runner == PYTHON_TRAIN_RUNNER


@pytest.mark.parametrize(
    ("metadata_lines", "message"),
    [
        (
            [
                "# version = 1",
                '# dataset_id = "mnist:v1"',
                '# runner = "shell"',
                "#",
                "# [evolution]",
                '# task = "Maximize validation accuracy while keeping the script runnable."',
            ],
            "Unsupported sigmaevolve runner",
        ),
        (
            [
                "# version = 1",
                '# dataset_id = "mnist:v1"',
                '# runner = "python_train_v1"',
                "#",
                "# [evolution]",
                '# task = "Maximize validation accuracy while keeping the script runnable."',
                '# objective = "val_loss:min"',
            ],
            "evolution.objective",
        ),
        (
            [
                "# version = 1",
                '# dataset_id = "mnist:v1"',
                '# runner = "python_train_v1"',
                "#",
                "# [evolution]",
            ],
            "evolution.task",
        ),
        (
            [
                "# version = 1",
                '# dataset_id = "mnist:v1"',
                '# runner = "python_train_v1"',
                "#",
                "# [evolution]",
                "# not = [valid toml",
            ],
            "Invalid sigmaevolve metadata block",
        ),
        (
            [
                "# version = 1",
                '# dataset_id = "mnist:v1"',
                '# runner = "python_train_v1"',
                "#",
                "# [evolution]",
                '# task = "Maximize validation accuracy while keeping the script runnable."',
                '# mutable_regions = ["experiment"]',
            ],
            "mutable_regions",
        ),
        (
            [
                "# version = 1",
                '# runner = "python_train_v1"',
                "#",
                "# [evolution]",
                '# task = "Maximize validation accuracy while keeping the script runnable."',
            ],
            "dataset_id",
        ),
    ],
)
def test_parse_script_spec_rejects_invalid_metadata(metadata_lines, message):
    body = "\n".join(
        [
            "# EVOLVE-BLOCK-START",
            "x = 1",
            "# EVOLVE-BLOCK-END",
            "",
        ]
    )
    source = "\n".join(
        [
            "# /// sigmaevolve",
            *metadata_lines,
            "# ///",
            "",
            body,
        ]
    )

    with pytest.raises(ScriptSpecError, match=message):
        require_script_spec(source)


def test_parse_script_spec_rejects_metadata_after_python_statement():
    source = (
        "from __future__ import annotations\n"
        "# /// sigmaevolve\n"
        "# version = 1\n"
        '# dataset_id = "mnist:v1"\n'
        '# runner = "python_train_v1"\n'
        "#\n"
        "# [evolution]\n"
        '# task = "Maximize validation accuracy while keeping the script runnable."\n'
        "# ///\n"
    )

    with pytest.raises(
        ScriptSpecError,
        match="must appear before the first Python statement",
    ):
        parse_script_spec(source)


def test_apply_script_policy_defaults_sets_epochs_only_when_missing():
    spec = require_script_spec(build_selfcontained_train_script(epochs=9))

    assert apply_script_policy_defaults({}, spec) == {"epochs": 9}
    assert apply_script_policy_defaults({"epochs": 3}, spec) == {"epochs": 3}


def test_apply_script_policy_defaults_starts_from_embedded_track_policy():
    spec = require_script_spec(
        build_selfcontained_train_script(
            epochs=9,
            track_policy={
                "dispatch_ttl_sec": 180,
                "generation_backend": {
                    "selection": "round_robin",
                    "model_pool": [{"model": "test/model", "temperature": 0.1}],
                },
            },
        )
    )

    assert apply_script_policy_defaults({}, spec) == {
        "dispatch_ttl_sec": 180,
        "epochs": 9,
        "generation_backend": {
            "selection": "round_robin",
            "model_pool": [{"model": "test/model", "temperature": 0.1}],
        },
    }
    assert apply_script_policy_defaults(
        {"generation_backend": {"selection": "weighted_random"}},
        spec,
    ) == {
        "dispatch_ttl_sec": 180,
        "epochs": 9,
        "generation_backend": {
            "selection": "weighted_random",
            "model_pool": [{"model": "test/model", "temperature": 0.1}],
        },
    }


def test_parse_source_layout_reads_named_blocks():
    source = "\n".join(
        [
            "# /// sigmaevolve",
            "# version = 1",
            '# dataset_id = "mnist:v1"',
            '# runner = "python_train_v1"',
            "#",
            "# [evolution]",
            '# task = "Maximize validation accuracy while keeping the script runnable."',
            "# ///",
            "",
            "# EVOLVE-BLOCK-START: model",
            "x = 1",
            "# EVOLVE-BLOCK-END: model",
            "",
            "# EVOLVE-BLOCK-START: training",
            "y = 2",
            "# EVOLVE-BLOCK-END: training",
            "",
        ]
    )

    layout = parse_source_layout(source)

    assert [block.name for block in layout.blocks] == ["model", "training"]
    assert [block.payload for block in layout.blocks] == ["x = 1\n", "y = 2\n"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "\n".join(
                [
                    "# EVOLVE-BLOCK-START: model",
                    "x = 1",
                    "# EVOLVE-BLOCK-END: training",
                    "",
                ]
            ),
            "mismatched end marker",
        ),
        (
            "\n".join(
                [
                    "# EVOLVE-BLOCK-START: model",
                    "x = 1",
                    "# EVOLVE-BLOCK-START: training",
                    "y = 2",
                    "# EVOLVE-BLOCK-END: training",
                    "",
                ]
            ),
            "Nested evolve blocks",
        ),
        (
            "\n".join(
                [
                    "# EVOLVE-BLOCK-START",
                    "x = 1",
                    "# EVOLVE-BLOCK-END",
                    "# EVOLVE-BLOCK-START",
                    "y = 2",
                    "# EVOLVE-BLOCK-END",
                    "",
                ]
            ),
            "Duplicate evolve block name",
        ),
    ],
)
def test_parse_source_layout_rejects_invalid_marker_layout(source, message):
    with pytest.raises(ScriptSpecError, match=message):
        parse_source_layout(source)


def test_parse_script_spec_accepts_default_objective_when_present():
    body = "\n".join(
        [
            "# EVOLVE-BLOCK-START",
            "x = 1",
            "# EVOLVE-BLOCK-END",
            "",
        ]
    )
    source = "\n".join(
        [
            "# /// sigmaevolve",
            "# version = 1",
            '# dataset_id = "mnist:v1"',
            '# runner = "python_train_v1"',
            "#",
            "# [evolution]",
            '# task = "Maximize validation accuracy while keeping the script runnable."',
            f'# objective = "{DEFAULT_OBJECTIVE}"',
            "# ///",
            "",
            body,
        ]
    )

    spec = require_script_spec(source)

    assert spec.evolution.objective == DEFAULT_OBJECTIVE
