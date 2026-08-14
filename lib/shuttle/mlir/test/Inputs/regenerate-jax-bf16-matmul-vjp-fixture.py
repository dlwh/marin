# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the ordinary-JAX BF16 matmul VJP fixture."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
ROWS = 17
REDUCTION = 33
COLUMNS = 19
BACKWARD_FILENAME = "jax-0.10.1-bf16-matmul-vjp-17x33x19-backward.mlir"
COMPOSED_FILENAME = "jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir"


def forward(lhs, rhs):
    return jnp.matmul(lhs, rhs, preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def backward(lhs, rhs, output_cotangent):
    _, pullback = jax.vjp(forward, lhs, rhs)
    return pullback(output_cotangent)


def composed(lhs, rhs, output_cotangent):
    output, pullback = jax.vjp(forward, lhs, rhs)
    lhs_cotangent, rhs_cotangent = pullback(output_cotangent)
    return output, lhs_cotangent, rhs_cotangent


def fixture_text(function, boundary: str) -> str:
    arguments = (
        jax.ShapeDtypeStruct((ROWS, REDUCTION), jnp.bfloat16),
        jax.ShapeDtypeStruct((REDUCTION, COLUMNS), jnp.bfloat16),
        jax.ShapeDtypeStruct((ROWS, COLUMNS), jnp.bfloat16),
    )
    module = jax.jit(function).lower(*arguments).compiler_ir(dialect="stablehlo")
    fixture_name = "VJP" if boundary == "backward" else boundary
    header = f"""// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary JAX 0.10.1 / jaxlib 0.10.1 matmul {fixture_name} fixture.
// Source: jax.jit({boundary}).lower(...).compiler_ir(dialect=\"stablehlo\")
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
    for filename, function, boundary in (
        (BACKWARD_FILENAME, backward, "backward"),
        (COMPOSED_FILENAME, composed, "composed"),
    ):
        path = arguments.output_dir / filename
        expected = fixture_text(function, boundary)
        if arguments.write:
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            mismatches.append(filename)
    if mismatches:
        parser.error("ordinary-JAX BF16 matmul VJP fixture drift: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
