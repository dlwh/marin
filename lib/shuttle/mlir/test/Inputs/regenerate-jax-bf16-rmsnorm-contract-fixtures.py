# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ordinary-JAX BF16 RMSNorm-to-Contract fixtures."""

import argparse
import hashlib
import json
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
    def reduction(self) -> int:
        return self.activation[-1]

    @property
    def weight(self) -> tuple[int, ...]:
        return (self.reduction,)

    @property
    def projection(self) -> tuple[int, ...]:
        return (self.reduction, self.columns)

    @property
    def result(self) -> tuple[int, ...]:
        return (*self.activation[:-1], self.columns)

    @property
    def shape_id(self) -> str:
        return "x".join(str(dimension) for dimension in (*self.activation, self.columns))

    @property
    def case_id(self) -> str:
        fields = {"activation": self.activation, "columns": self.columns}
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return "rmsnorm_contract_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


SHAPES = (
    StructuralShape(activation=(5, 19), columns=7),
    StructuralShape(activation=(5, 65), columns=7),
    StructuralShape(activation=(5, 19), columns=9),
    StructuralShape(activation=(5, 19), columns=257),
    StructuralShape(activation=(3, 2, 5, 11, 263), columns=521),
)


def rmsnorm_contract(x, weight, projection):
    local = x.astype(jnp.float32)
    mean_square = jnp.mean(local * local, axis=-1, keepdims=True)
    inverse_root = lax.rsqrt(mean_square + jnp.float32(1e-6))
    normalized = (local * inverse_root * weight.astype(jnp.float32)).astype(jnp.bfloat16)
    result = jnp.matmul(normalized, projection, preferred_element_type=jnp.float32)
    return result.astype(jnp.bfloat16)


def signature(shape: tuple[int, ...]) -> str:
    return f"{shape}:bfloat16"


def audited_fixture(shape: StructuralShape) -> str:
    arguments = (
        jax.ShapeDtypeStruct(shape.activation, jnp.bfloat16),
        jax.ShapeDtypeStruct(shape.weight, jnp.bfloat16),
        jax.ShapeDtypeStruct(shape.projection, jnp.bfloat16),
    )
    module = jax.jit(rmsnorm_contract).lower(*arguments).compiler_ir(dialect="stablehlo")
    payload = f"{module}".rstrip() + "\n"
    generator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()
    fields = json.dumps(
        {
            "activation": shape.activation,
            "columns": shape.columns,
            "projection": shape.projection,
            "weight": shape.weight,
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
            f"// Generator source SHA-256: {generator_digest}",
            "// Source: jax.jit(ordinary JAX BF16 RMSNorm followed by matmul)",
            '//   .lower(...).compiler_ir(dialect="stablehlo")',
            f"// Case ID: {shape.case_id}",
            f"// Structural fields: {fields}",
            "// Inputs: "
            f"x={signature(shape.activation)}, weight={signature(shape.weight)}, "
            f"projection={signature(shape.projection)}",
            f"// Outputs: result={signature(shape.result)}",
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
        filename = f"jax-{JAX_VERSION}-bf16-rmsnorm_contract-{shape.shape_id}-forward.mlir"
        path = arguments.output_dir / filename
        expected = audited_fixture(shape)
        if arguments.write:
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            mismatches.append(filename)
    if mismatches:
        parser.error("ordinary-JAX RMSNorm Contract fixture drift: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
