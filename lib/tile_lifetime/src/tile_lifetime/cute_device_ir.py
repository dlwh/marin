# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured snapshots of the retained CuTe compiler module."""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CuteDeviceIrSnapshot:
    """Content identity and operation inventory for one retained MLIR module."""

    bytecode_sha256: str
    operation_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if len(self.bytecode_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.bytecode_sha256
        ):
            raise ValueError("CuTe device-IR bytecode requires a SHA-256 digest")
        if not self.operation_counts:
            raise ValueError("CuTe device-IR operation inventory must be nonempty")
        if any(not name or "\0" in name or count <= 0 for name, count in self.operation_counts):
            raise ValueError("CuTe device-IR operation counts must be positive and named")
        if tuple(sorted(self.operation_counts)) != self.operation_counts:
            raise ValueError("CuTe device-IR operation counts must be sorted")
        if len({name for name, _ in self.operation_counts}) != len(self.operation_counts):
            raise ValueError("CuTe device-IR operation names must be unique")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "bytecode_sha256": self.bytecode_sha256,
            "operation_counts": tuple({"name": name, "count": count} for name, count in self.operation_counts),
        }


def capture_cute_module(module: Any) -> CuteDeviceIrSnapshot:
    """Inventory a retained CuTe MLIR module without converting it to text."""
    if module is None:
        raise ValueError("CuTe compilation did not retain its structured MLIR module")
    with module.context:
        counts = Counter(operation.name for operation in _operations(module.operation))
        buffer = io.BytesIO()
        module.operation.write_bytecode(buffer)
        bytecode = buffer.getvalue()
    if not counts:
        raise ValueError("CuTe structured MLIR contains no operations")
    return CuteDeviceIrSnapshot(
        bytecode_sha256=hashlib.sha256(bytecode).hexdigest(),
        operation_counts=tuple(sorted(counts.items())),
    )


def compile_cute_with_structured_snapshot(
    cute: Any,
    function: Any,
    *arguments: Any,
    options: str,
) -> tuple[Any, CuteDeviceIrSnapshot]:
    """Compile once and snapshot the exact structured module backing it."""
    executable = cute.compile(function, *arguments, options=options)
    snapshot = capture_cute_module(executable.ir_module)
    return executable, snapshot


def _operations(operation: Any) -> Iterator[Any]:
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for nested in block.operations:
                yield from _operations(nested.operation)
