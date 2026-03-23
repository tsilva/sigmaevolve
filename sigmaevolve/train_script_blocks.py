from __future__ import annotations

from textwrap import indent

from sigmaevolve.baseline import build_baseline_train_script
from sigmaevolve.evolve_blocks import replace_evolve_block_payloads


def build_candidate_train_script(block_payload: str) -> str:
    return replace_evolve_block_payloads(
        build_baseline_train_script(),
        [block_payload.strip("\n") + "\n"],
    )


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
