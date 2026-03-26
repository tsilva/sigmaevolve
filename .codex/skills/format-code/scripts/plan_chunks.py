#!/usr/bin/env python3
"""Plan disjoint Python file chunks for parallel SigmaEvolve formatting work."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_TARGETS = ("sigmaevolve", "tests")
SMALL_FILE_MAX_LINES = 80
SMALL_BATCH_MAX_FILES = 4
SMALL_BATCH_MAX_LINES = 240


@dataclass(frozen=True)
class FileInfo:
    path: str
    lines: int
    role: str


@dataclass(frozen=True)
class Chunk:
    name: str
    reason: str
    total_lines: int
    files: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend disjoint file chunks for parallel Python formatting work.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="Files or directories to inspect. Defaults to sigmaevolve tests.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    return parser.parse_args()


def iter_python_files(targets: list[str]) -> list[Path]:
    files: dict[Path, None] = {}
    for raw_target in targets:
        target = Path(raw_target)
        if not target.exists():
            continue
        if target.is_file():
            if target.suffix == ".py":
                files[target] = None
            continue
        for path in target.rglob("*.py"):
            if path.is_file():
                files[path] = None
    return sorted(files)


def classify_role(path: Path) -> str:
    parts = path.parts
    if parts and parts[0] == "tests":
        return "test"
    if parts and parts[0] == "sigmaevolve":
        return "source"
    return "other"


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def build_file_infos(paths: list[Path]) -> list[FileInfo]:
    return [
        FileInfo(
            path=path.as_posix(),
            lines=count_lines(path),
            role=classify_role(path),
        )
        for path in paths
    ]


def build_test_index(files: list[FileInfo]) -> dict[str, FileInfo]:
    return {Path(file.path).name: file for file in files if file.role == "test"}


def pair_name(source: FileInfo) -> str:
    stem = Path(source.path).stem.replace("_", "-")
    return f"{stem}-chunk"


def misc_name(root: str, index: int) -> str:
    return f"{root}-misc-{index}"


def should_batch_tiny(file: FileInfo) -> bool:
    if file.lines > SMALL_FILE_MAX_LINES:
        return False

    return Path(file.path).name in {"__init__.py", "support.py"}


def build_chunks(files: list[FileInfo]) -> list[Chunk]:
    remaining = {file.path: file for file in files}
    test_index = build_test_index(files)
    chunks: list[Chunk] = []

    for file in files:
        if (
            file.role != "source"
            or file.path not in remaining
            or should_batch_tiny(file)
        ):
            continue

        chunk_files = [file.path]
        reason = "single source file"
        test_name = f"test_{Path(file.path).stem}.py"
        matching_test = test_index.get(test_name)

        if matching_test is not None and matching_test.path in remaining:
            chunk_files.append(matching_test.path)
            reason = "paired source file with matching test"

        for path in chunk_files:
            remaining.pop(path, None)

        total_lines = sum(
            next(item.lines for item in files if item.path == path)
            for path in chunk_files
        )
        chunks.append(
            Chunk(
                name=pair_name(file),
                reason=reason,
                total_lines=total_lines,
                files=sorted(chunk_files),
            )
        )

    small_batches: dict[str, list[FileInfo]] = {"source": [], "test": [], "other": []}
    large_leftovers: list[FileInfo] = []
    for file in remaining.values():
        if should_batch_tiny(file):
            small_batches[file.role].append(file)
        else:
            large_leftovers.append(file)

    for file in sorted(large_leftovers, key=lambda item: (-item.lines, item.path)):
        chunks.append(
            Chunk(
                name=pair_name(file),
                reason=f"unpaired {file.role} file",
                total_lines=file.lines,
                files=[file.path],
            )
        )

    for role, files_in_role in small_batches.items():
        files_in_role = sorted(files_in_role, key=lambda item: item.path)
        if not files_in_role:
            continue

        batch: list[FileInfo] = []
        batch_lines = 0
        batch_index = 1
        for file in files_in_role:
            next_line_total = batch_lines + file.lines
            if batch and (
                len(batch) >= SMALL_BATCH_MAX_FILES
                or next_line_total > SMALL_BATCH_MAX_LINES
            ):
                chunks.append(
                    Chunk(
                        name=misc_name(role, batch_index),
                        reason=f"batched tiny {role} support files",
                        total_lines=batch_lines,
                        files=[item.path for item in batch],
                    )
                )
                batch = []
                batch_lines = 0
                batch_index += 1

            batch.append(file)
            batch_lines += file.lines

        if batch:
            chunks.append(
                Chunk(
                    name=misc_name(role, batch_index),
                    reason=f"batched tiny {role} support files",
                    total_lines=batch_lines,
                    files=[item.path for item in batch],
                )
            )

    return sorted(chunks, key=lambda chunk: (-chunk.total_lines, chunk.name))


def render_text(chunks: list[Chunk]) -> str:
    lines = [f"Recommended chunks: {len(chunks)}"]
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"{index}. {chunk.name} ({chunk.total_lines} lines) - {chunk.reason}"
        )
        for path in chunk.files:
            lines.append(f"   - {path}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    files = build_file_infos(iter_python_files(args.targets))
    chunks = build_chunks(files)

    if args.json:
        print(json.dumps([asdict(chunk) for chunk in chunks], indent=2))
    else:
        print(render_text(chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
