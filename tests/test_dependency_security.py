from importlib.metadata import version
from io import BytesIO

import cbor2
import pytest
import torch
from PIL import Image, UnidentifiedImageError


def _numeric_version(distribution: str) -> tuple[int, ...]:
    release = version(distribution).split("+", maxsplit=1)[0]
    return tuple(int(part) for part in release.split(".") if part.isdigit())


@pytest.mark.parametrize(
    ("distribution", "minimum"),
    [
        ("aiohttp", (3, 14, 3)),
        ("cbor2", (5, 9, 0)),
        ("click", (8, 3, 3)),
        ("h2", (4, 4, 1)),
        ("idna", (3, 15)),
        ("pillow", (12, 3, 0)),
        ("pygments", (2, 20, 0)),
        ("requests", (2, 33, 0)),
        ("setuptools", (83, 0, 0)),
        ("torch", (2, 13, 0)),
        ("torchvision", (0, 28, 0)),
        ("urllib3", (2, 7, 0)),
        ("wandb", (0, 28, 1)),
    ],
)
def test_audited_dependency_floors(distribution: str, minimum: tuple[int, ...]) -> None:
    assert _numeric_version(distribution) >= minimum


def test_binary_decoders_reject_invalid_input_and_accept_valid_input() -> None:
    with pytest.raises(cbor2.CBORDecodeError):
        cbor2.loads(b"\x9f")

    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(b"not an image")).verify()

    assert cbor2.loads(cbor2.dumps({"score": 1.0})) == {"score": 1.0}

    image_buffer = BytesIO()
    Image.new("RGB", (2, 2), "white").save(image_buffer, format="PNG")
    image_buffer.seek(0)
    with Image.open(image_buffer) as image:
        image.verify()


def test_torch_runtime_still_executes_a_legitimate_model() -> None:
    model = torch.nn.Linear(2, 1)
    output = model(torch.tensor([[1.0, 2.0]]))
    assert output.shape == (1, 1)
