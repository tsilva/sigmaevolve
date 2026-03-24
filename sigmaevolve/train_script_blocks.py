from __future__ import annotations

from textwrap import indent

from sigmaevolve.baseline import build_baseline_train_script
from sigmaevolve.evolve_blocks import extract_evolve_block_payloads, replace_evolve_block_payloads


CONFIG_BLOCK_INDEX = 0
MODEL_BLOCK_INDEX = 1
DATA_BLOCK_INDEX = 2
OPTIMIZATION_BLOCK_INDEX = 3
TRAINING_POLICY_BLOCK_INDEX = 4


def _normalize_payload(payload: str) -> str:
    return payload.strip("\n") + "\n"


def build_candidate_train_script(
    block_payload: str | None = None,
    *,
    config_block_payload: str | None = None,
    model_block_payload: str | None = None,
    data_block_payload: str | None = None,
    optimization_block_payload: str | None = None,
    training_policy_block_payload: str | None = None,
) -> str:
    template_source = build_baseline_train_script()
    payloads = extract_evolve_block_payloads(template_source)

    replacements: dict[int, str | None] = {
        CONFIG_BLOCK_INDEX: config_block_payload,
        MODEL_BLOCK_INDEX: model_block_payload if model_block_payload is not None else block_payload,
        DATA_BLOCK_INDEX: data_block_payload,
        OPTIMIZATION_BLOCK_INDEX: optimization_block_payload,
        TRAINING_POLICY_BLOCK_INDEX: training_policy_block_payload,
    }
    for index, payload in replacements.items():
        if payload is not None:
            payloads[index] = _normalize_payload(payload)

    return replace_evolve_block_payloads(template_source, payloads)


def build_config_block(body: str) -> str:
    return _normalize_payload(body)


def build_model_block(
    body: str,
    *,
    imports: str = "import torch",
    build_body: str = "return EvolvedModel()",
) -> str:
    parts: list[str] = []
    imports = imports.strip()
    if imports:
        parts.append(imports)
        parts.append("")
    parts.append("class EvolvedModel(torch.nn.Module):")
    parts.append(indent(body.strip("\n"), "    "))
    parts.append("")
    parts.append("")
    parts.append("def build_model(*, input_shape, num_classes):")
    parts.append(indent(build_body.strip("\n"), "    "))
    parts.append("")
    return "\n".join(parts)


def _build_function_block(
    body: str,
    *,
    function_name: str,
    signature: str,
    imports: str = "",
) -> str:
    parts: list[str] = []
    imports = imports.strip()
    if imports:
        parts.append(imports)
        parts.append("")
    parts.append(f"def {function_name}({signature}):")
    parts.append(indent(body.strip("\n"), "    "))
    parts.append("")
    return "\n".join(parts)


def build_data_block(
    body: str,
    *,
    imports: str = "import torch",
) -> str:
    return _build_function_block(
        body,
        function_name="configure_data",
        signature="*, train_x, train_y, validation_x, random_seed",
        imports=imports,
    )


def build_optimization_block(
    body: str,
    *,
    imports: str = "import torch",
) -> str:
    return _build_function_block(
        body,
        function_name="configure_optimization",
        signature="*, model, train_loader, num_epochs, num_classes",
        imports=imports,
    )


def build_training_policy_block(body: str) -> str:
    return _build_function_block(
        body,
        function_name="configure_training_policy",
        signature="*, num_epochs",
    )
