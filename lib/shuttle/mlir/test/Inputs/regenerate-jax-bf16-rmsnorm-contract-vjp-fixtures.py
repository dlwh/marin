# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ordinary-JAX BF16 RMSNorm-to-Contract composed VJP fixtures."""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
from jax import lax
from jaxlib.mlir.dialects import stablehlo

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
JAX_REVISION = "619764c15117fbefc4ba13ab941871cb514c23f6"
XLA_REVISION = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
STABLEHLO_VERSION = "1.17.0"


@dataclass(frozen=True)
class StructuralShape:
    activation: tuple[int, ...]
    columns: int

    @property
    def rows(self) -> int:
        return math.prod(self.activation[:-1])

    @property
    def reduction(self) -> int:
        return self.activation[-1]

    @property
    def shape_id(self) -> str:
        return "x".join(str(dimension) for dimension in (*self.activation, self.columns))


SHAPES = (
    StructuralShape((5, 19), 7),
    StructuralShape((5, 19), 257),
    StructuralShape((2, 5, 19), 7),
)


def rmsnorm_contract(x, weight, projection):
    local = x.astype(jnp.float32)
    mean_square = jnp.mean(local * local, axis=-1, keepdims=True)
    inverse_root = lax.rsqrt(mean_square + jnp.float32(1e-6))
    normalized = (local * inverse_root * weight.astype(jnp.float32)).astype(jnp.bfloat16)
    result = jnp.matmul(normalized, projection, preferred_element_type=jnp.float32)
    return result.astype(jnp.bfloat16)


def composed(x, weight, projection, cotangent):
    result, pullback = jax.vjp(rmsnorm_contract, x, weight, projection)
    return (result, *pullback(cotangent))


def projection_vjp(x, weight, projection, cotangent):
    result, pullback = jax.vjp(rmsnorm_contract, x, weight, projection)
    return result, pullback(cotangent)[2]


def audited_fixture(shape: StructuralShape, *, projection_only: bool = False) -> str:
    arguments = (
        jax.ShapeDtypeStruct(shape.activation, jnp.bfloat16),
        jax.ShapeDtypeStruct((shape.reduction,), jnp.bfloat16),
        jax.ShapeDtypeStruct((shape.reduction, shape.columns), jnp.bfloat16),
        jax.ShapeDtypeStruct((*shape.activation[:-1], shape.columns), jnp.bfloat16),
    )
    boundary = projection_vjp if projection_only else composed
    module = jax.jit(boundary).lower(*arguments).compiler_ir(dialect="stablehlo")
    payload = f"{module}".rstrip() + "\n"
    fields = json.dumps(
        {
            "activation": shape.activation,
            "columns": shape.columns,
            "reduction": shape.reduction,
            "rows": shape.rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    header = "\n".join(
        (
            "// Copyright The Marin Authors",
            "// SPDX-License-Identifier: Apache-2.0",
            "",
            "// Artifact classification: ordinary_jax_fixture_only",
            "// Evaluation oracle status: hardware_gate_pending",
            "// Hardware evidence: none",
            f"// Generator: {Path(__file__).name}",
            f"// Generator source SHA-256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()}",
            "// Source: jax.jit(ordinary JAX BF16 RMSNorm-to-Contract "
            + ("projection-only VJP)" if projection_only else "composed VJP)"),
            '//   .lower(...).compiler_ir(dialect="stablehlo")',
            f"// Structural fields: {fields}",
            f"// JAX: {JAX_VERSION}; jaxlib: {JAXLIB_VERSION}; JAX revision: {JAX_REVISION}",
            f"// XLA revision: {XLA_REVISION}; StableHLO current version: {STABLEHLO_VERSION}",
            f"// Raw StableHLO SHA-256: {hashlib.sha256(payload.encode()).hexdigest().upper()}",
            "",
        )
    )
    return header + payload


def verify_toolchain() -> None:
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError(
            f"fixtures require JAX {JAX_VERSION} and jaxlib {JAXLIB_VERSION}; "
            f"found {jax.__version__} and {jaxlib.__version__}"
        )
    if stablehlo.get_current_version() != STABLEHLO_VERSION:
        raise RuntimeError("fixture StableHLO version drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    verify_toolchain()
    mismatches = []
    for shape in SHAPES:
        filename = f"jax-{JAX_VERSION}-bf16-rmsnorm_contract-{shape.shape_id}-composed.mlir"
        path = arguments.output_dir / filename
        expected = audited_fixture(shape)
        if arguments.write:
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            mismatches.append(filename)
    projection_shape = StructuralShape((5, 19), 257)
    filename = f"jax-{JAX_VERSION}-bf16-rmsnorm_contract-" f"{projection_shape.shape_id}-projection-vjp.mlir"
    path = arguments.output_dir / filename
    expected = audited_fixture(projection_shape, projection_only=True)
    if arguments.write:
        path.write_text(expected)
    elif not path.exists() or path.read_text() != expected:
        mismatches.append(filename)
    if mismatches:
        parser.error("ordinary-JAX RMSNorm Contract VJP fixture drift: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
