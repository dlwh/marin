# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ordinary-JAX BF16 LayerNorm forward and VJP fixtures."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir import ir, passmanager
from jaxlib.mlir.dialects import stablehlo

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
JAX_REVISION = "619764c15117fbefc4ba13ab941871cb514c23f6"
XLA_REVISION = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
STABLEHLO_VERSION = "1.17.0"
HOOK_PIPELINE = "builtin.module(func.func(stablehlo-complex-math-expander))"
EPSILON = 1e-5
BOUNDARIES = ("forward", "backward", "composed")


@dataclass(frozen=True)
class StructuralShape:
    rows: int
    features: int
    role: str

    @property
    def case_id(self) -> str:
        fields = {"features": self.features, "rows": self.rows}
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return "layer_norm_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


SHAPES = (
    StructuralShape(rows=2048, features=4096, role="target2_representative_shape"),
    StructuralShape(rows=7, features=13, role="structural_shape_mutation"),
)


def layer_norm(x, gamma, beta):
    """Apply explicit ordinary-JAX two-pass LayerNorm."""
    local = x.astype(jnp.float32)
    mean = jnp.mean(local, axis=-1, keepdims=True)
    centered = local - mean
    inverse = jax.lax.rsqrt(jnp.mean(centered * centered, axis=-1, keepdims=True) + jnp.float32(EPSILON))
    return (centered * inverse * gamma.astype(jnp.float32) + beta.astype(jnp.float32)).astype(jnp.bfloat16)


def boundary_function(boundary: str):
    if boundary == "forward":

        def forward(x, gamma, beta):
            return layer_norm(x, gamma, beta)

        return forward
    if boundary == "backward":

        def backward(x, gamma, dy):
            beta = jnp.zeros_like(gamma)
            _, pullback = jax.vjp(layer_norm, x, gamma, beta)
            return pullback(dy)

        return backward
    if boundary == "composed":

        def composed(x, gamma, beta, dy):
            y, pullback = jax.vjp(layer_norm, x, gamma, beta)
            dx, dgamma, dbeta = pullback(dy)
            return y, dx, dgamma, dbeta

        return composed
    raise ValueError(f"unsupported boundary {boundary!r}")


def input_signature(shape: StructuralShape, boundary: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    inputs = (
        ("x", (shape.rows, shape.features)),
        ("gamma", (shape.features,)),
    )
    if boundary == "forward":
        return (*inputs, ("beta", (shape.features,)))
    if boundary == "backward":
        return (*inputs, ("dy", (shape.rows, shape.features)))
    return (
        *inputs,
        ("beta", (shape.features,)),
        ("dy", (shape.rows, shape.features)),
    )


def output_signature(shape: StructuralShape, boundary: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if boundary == "forward":
        return (("y", (shape.rows, shape.features)),)
    gradients = (
        ("dx", (shape.rows, shape.features)),
        ("dgamma", (shape.features,)),
        ("dbeta", (shape.features,)),
    )
    if boundary == "backward":
        return gradients
    return (("y", (shape.rows, shape.features)), *gradients)


def filename(shape: StructuralShape, boundary: str) -> str:
    return f"jax-{JAX_VERSION}-bf16-{shape.case_id}-{boundary}.mlir"


def export_stablehlo(shape: StructuralShape, boundary: str) -> str:
    arguments = tuple(
        jax.ShapeDtypeStruct(dimensions, jnp.bfloat16) for _, dimensions in input_signature(shape, boundary)
    )
    module = jax.jit(boundary_function(boundary)).lower(*arguments).compiler_ir(dialect="stablehlo")
    return f"{module}".rstrip() + "\n"


def hook_boundary_stablehlo(payload: str) -> str:
    stablehlo.register_stablehlo_passes()
    with make_ir_context():
        module = ir.Module.parse(payload)
        passmanager.PassManager.parse(HOOK_PIPELINE).run(module.operation)
        return f"{module}".rstrip() + "\n"


def _signature_header(
    signature: tuple[tuple[str, tuple[int, ...]], ...],
) -> str:
    return ", ".join(f"{role}={dimensions}:bfloat16" for role, dimensions in signature)


def audited_fixture(shape: StructuralShape, boundary: str) -> str:
    payload = export_stablehlo(shape, boundary)
    hook_payload = hook_boundary_stablehlo(payload)
    generator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()
    fields = json.dumps(
        {
            "boundary": boundary,
            "epsilon": EPSILON,
            "features": shape.features,
            "rows": shape.rows,
            "shape_role": shape.role,
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
            "// Evaluation oracle status: not_pinned",
            "// Hardware evidence: none",
            f"// Generator: {Path(__file__).name}",
            f"// Generator source SHA-256: {generator_digest}",
            "// Source: jax.jit(ordinary JAX LayerNorm plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)",
            f"// Case ID: {shape.case_id}",
            f"// Structural fields: {fields}",
            f"// Inputs: {_signature_header(input_signature(shape, boundary))}",
            f"// Outputs: {_signature_header(output_signature(shape, boundary))}",
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
            f"fixture inventory requires StableHLO {STABLEHLO_VERSION}; " f"found {stablehlo.get_current_version()}"
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
        for boundary in BOUNDARIES:
            expected = audited_fixture(shape, boundary)
            path = arguments.output_dir / filename(shape, boundary)
            if arguments.write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(expected)
            elif not path.exists() or path.read_text() != expected:
                mismatches.append(path.name)
    if mismatches:
        parser.error("BF16 LayerNorm fixture inventory drift: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
