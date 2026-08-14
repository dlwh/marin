# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the ordinary-JAX ABI15 composed matmul evidence fixture."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
ROWS = 33
REDUCTION = 2
COLUMNS = 34
FIXTURE_FILENAME = "jax-0.10.1-bf16-matmul-vjp-33x2x34-composed.mlir"


def forward(lhs, rhs):
    """Return the BF16 matmul with an explicit FP32 accumulator."""
    return jnp.matmul(lhs, rhs, preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def composed(lhs, rhs, output_cotangent):
    """Return the primal and both cotangents from ordinary JAX AD."""
    output, pullback = jax.vjp(forward, lhs, rhs)
    lhs_cotangent, rhs_cotangent = pullback(output_cotangent)
    return output, lhs_cotangent, rhs_cotangent


def fixture_text() -> str:
    """Return the canonical StableHLO text produced by pinned ordinary JAX."""
    arguments = (
        jax.ShapeDtypeStruct((ROWS, REDUCTION), jnp.bfloat16),
        jax.ShapeDtypeStruct((REDUCTION, COLUMNS), jnp.bfloat16),
        jax.ShapeDtypeStruct((ROWS, COLUMNS), jnp.bfloat16),
    )
    module = jax.jit(composed).lower(*arguments).compiler_ir(dialect="stablehlo")
    header = """// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary JAX 0.10.1 / jaxlib 0.10.1 matmul composed fixture.
// Source: jax.jit(composed).lower(...).compiler_ir(dialect="stablehlo")
"""
    return header + f"{module}".rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        parser.error(
            f"requires JAX {JAX_VERSION} and jaxlib {JAXLIB_VERSION}; "
            f"found {jax.__version__} and {jaxlib.__version__}"
        )
    path = arguments.output_dir / FIXTURE_FILENAME
    expected = fixture_text()
    if arguments.write:
        path.write_text(expected)
        return 0
    if not path.exists() or path.read_text() != expected:
        parser.error(f"ordinary-JAX ABI15 composed fixture drift: {FIXTURE_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
