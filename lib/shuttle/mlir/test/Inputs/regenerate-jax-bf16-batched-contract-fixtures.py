# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ordinary-JAX BF16 projected batched Contract fixtures."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"


@dataclass(frozen=True)
class ShapeFamily:
    batch: int
    rows: int
    heads: int
    reduction: int
    columns: int

    @property
    def shape_id(self) -> str:
        return f"{self.batch}x{self.rows}x{self.heads}x{self.reduction}x{self.columns}"

    @property
    def lhs_shape(self) -> tuple[int, ...]:
        return (self.batch, self.rows, self.heads, self.reduction)

    @property
    def rhs_shape(self) -> tuple[int, ...]:
        return (self.batch, self.columns, self.heads, self.reduction)

    @property
    def result_shape(self) -> tuple[int, ...]:
        return (self.batch, self.heads, self.rows, self.columns)


SHAPE_FAMILIES = (
    ShapeFamily(batch=2, rows=5, heads=3, reduction=7, columns=11),
    ShapeFamily(batch=3, rows=4, heads=2, reduction=13, columns=6),
)


def forward(lhs, rhs):
    lhs_f32 = jnp.transpose(lhs.astype(jnp.float32), (0, 2, 1, 3))
    rhs_f32 = jnp.transpose(rhs.astype(jnp.float32), (0, 2, 1, 3))
    result = jnp.einsum("bhmk,bhnk->bhmn", lhs_f32, rhs_f32)
    return result.astype(jnp.bfloat16)


def backward(lhs, rhs, output_cotangent):
    _, pullback = jax.vjp(forward, lhs, rhs)
    return pullback(output_cotangent)


def composed(lhs, rhs, output_cotangent):
    result, pullback = jax.vjp(forward, lhs, rhs)
    lhs_cotangent, rhs_cotangent = pullback(output_cotangent)
    return result, lhs_cotangent, rhs_cotangent


def fixture_text(shape: ShapeFamily, boundary: str) -> str:
    inputs = (
        jax.ShapeDtypeStruct(shape.lhs_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(shape.rhs_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(shape.result_shape, jnp.bfloat16),
    )
    function = {"forward": forward, "backward": backward, "composed": composed}[boundary]
    arguments = inputs[:2] if boundary == "forward" else inputs
    module = jax.jit(function).lower(*arguments).compiler_ir(dialect="stablehlo")
    header = f"""// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary JAX {JAX_VERSION} / jaxlib {JAXLIB_VERSION} projected batched Contract fixture.
// Source: jax.jit({boundary}).lower(...).compiler_ir(dialect="stablehlo")
// Structural shape: batch={shape.batch}, rows={shape.rows}, heads={shape.heads}
// Reduction shape: reduction={shape.reduction}, columns={shape.columns}
"""
    return header + f"{module}".rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        parser.error(
            f"requires JAX {JAX_VERSION} and jaxlib {JAXLIB_VERSION}; "
            f"found {jax.__version__} and {jaxlib.__version__}"
        )

    mismatches = []
    for shape in SHAPE_FAMILIES:
        for boundary in ("forward", "backward", "composed"):
            filename = f"jax-{JAX_VERSION}-bf16-batched_contract-{shape.shape_id}-{boundary}.mlir"
            path = arguments.output_dir / filename
            expected = fixture_text(shape, boundary)
            if arguments.write:
                path.write_text(expected)
            elif not path.exists() or path.read_text() != expected:
                mismatches.append(filename)
    if mismatches:
        parser.error("ordinary-JAX batched Contract fixture drift: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
