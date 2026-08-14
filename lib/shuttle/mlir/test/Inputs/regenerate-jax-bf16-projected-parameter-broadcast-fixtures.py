# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ordinary-JAX BF16 projected-parameter broadcast fixtures."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
from jax import lax
from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir import ir, passmanager
from jaxlib.mlir.dialects import stablehlo

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
JAX_REVISION = "619764c15117fbefc4ba13ab941871cb514c23f6"
XLA_REVISION = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
STABLEHLO_VERSION = "1.17.0"
HOOK_PIPELINE = "builtin.module(func.func(stablehlo-complex-math-expander))"


@dataclass(frozen=True)
class StructuralShape:
    dimensions: tuple[int, ...]

    @property
    def weight_dimensions(self) -> tuple[int, ...]:
        return (self.dimensions[-1],)

    @property
    def shape_id(self) -> str:
        return "x".join(str(dimension) for dimension in self.dimensions)

    @property
    def case_id(self) -> str:
        fields = {"dimensions": self.dimensions, "weight_dimensions": self.weight_dimensions}
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return "projected_parameter_broadcast_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


SHAPES = (
    StructuralShape(dimensions=(2, 7, 19)),
    StructuralShape(dimensions=(3, 2, 5, 11, 263)),
)


def rms_norm_forward(x, weight):
    local = x.astype(jnp.float32)
    mean_square = jnp.mean(local * local, axis=-1, keepdims=True)
    inverse_root = lax.rsqrt(mean_square + jnp.float32(1e-6))
    return (local * inverse_root * weight.astype(jnp.float32)).astype(jnp.bfloat16)


def filename(shape: StructuralShape) -> str:
    return f"jax-{JAX_VERSION}-bf16-projected_parameter_broadcast-{shape.shape_id}-forward.mlir"


def export_stablehlo(shape: StructuralShape) -> str:
    arguments = (
        jax.ShapeDtypeStruct(shape.dimensions, jnp.bfloat16),
        jax.ShapeDtypeStruct(shape.weight_dimensions, jnp.bfloat16),
    )
    module = jax.jit(rms_norm_forward).lower(*arguments).compiler_ir(dialect="stablehlo")
    return f"{module}".rstrip() + "\n"


def hook_boundary_stablehlo(payload: str) -> str:
    stablehlo.register_stablehlo_passes()
    with make_ir_context():
        module = ir.Module.parse(payload)
        passmanager.PassManager.parse(HOOK_PIPELINE).run(module.operation)
        return f"{module}".rstrip() + "\n"


def audited_fixture(shape: StructuralShape) -> str:
    payload = export_stablehlo(shape)
    hook_payload = hook_boundary_stablehlo(payload)
    generator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()
    fields = json.dumps(
        {"boundary": "forward", "dimensions": shape.dimensions, "weight_dimensions": shape.weight_dimensions},
        sort_keys=True,
        separators=(",", ":"),
    )
    header = "\n".join(
        (
            "// Copyright The Marin Authors",
            "// SPDX-License-Identifier: Apache-2.0",
            "",
            "// Artifact classification: ordinary_jax_fixture_only",
            "// Evaluation oracle status: not_pinned",
            "// Hardware evidence: none",
            f"// Generator: {Path(__file__).name}",
            f"// Generator source SHA-256: {generator_digest}",
            "// Source: jax.jit(ordinary JAX RMS normalization with projected BF16",
            "// weight).lower(...).compiler_ir(StableHLO)",
            f"// Case ID: {shape.case_id}",
            f"// Structural fields: {fields}",
            f"// Inputs: x={shape.dimensions}:bfloat16, weight={shape.weight_dimensions}:bfloat16",
            f"// Outputs: result={shape.dimensions}:bfloat16",
            f"// JAX: {JAX_VERSION}; jaxlib: {JAXLIB_VERSION}; JAX revision: {JAX_REVISION}",
            f"// XLA revision: {XLA_REVISION}; StableHLO current version: {STABLEHLO_VERSION}",
            f"// Raw StableHLO SHA-256: {hashlib.sha256(payload.encode()).hexdigest().upper()}",
            "// XLA hook-boundary preprocessing: stablehlo-complex-math-expander",
            f"// XLA hook-boundary StableHLO SHA-256: {hashlib.sha256(hook_payload.encode()).hexdigest().upper()}",
            "",
        )
    )
    return header + payload


def verify_toolchain() -> None:
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError(
            f"fixture inventory requires JAX {JAX_VERSION} and jaxlib {JAXLIB_VERSION}; "
            f"found {jax.__version__} and {jaxlib.__version__}"
        )
    if stablehlo.get_current_version() != STABLEHLO_VERSION:
        raise RuntimeError(
            f"fixture inventory requires StableHLO {STABLEHLO_VERSION}; found {stablehlo.get_current_version()}"
        )
    mlir_root = Path(__file__).resolve().parents[2]
    pinned_jax = (mlir_root / "jax_patch" / "PINNED_JAX_REVISION").read_text().strip()
    pinned_xla = (mlir_root / "xla_patch" / "PINNED_XLA_REVISION").read_text().strip()
    if pinned_jax != JAX_REVISION or pinned_xla != XLA_REVISION:
        raise RuntimeError("fixture inventory source revision pins drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    verify_toolchain()
    mismatches = []
    for shape in SHAPES:
        expected = audited_fixture(shape)
        path = arguments.output_dir / filename(shape)
        if arguments.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            mismatches.append(path.name)
    if mismatches:
        parser.error("BF16 projected-parameter broadcast fixture inventory drift: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
