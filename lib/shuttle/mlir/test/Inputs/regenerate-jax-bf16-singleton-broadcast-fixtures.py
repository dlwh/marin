# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ordinary-JAX BF16 expanding-singleton broadcast fixtures."""

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
BOUNDARIES = ("forward", "backward", "composed")


@dataclass(frozen=True)
class StructuralShape:
    dimensions: tuple[int, ...]

    @property
    def shape_id(self) -> str:
        return "x".join(str(dimension) for dimension in self.dimensions)

    @property
    def case_id(self) -> str:
        fields = {"dimensions": self.dimensions}
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return "singleton_broadcast_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


SHAPES = (
    StructuralShape(dimensions=(2, 7, 19)),
    StructuralShape(dimensions=(3, 2, 5, 11, 263)),
)


def singleton_broadcast(x):
    local = x.astype(jnp.float32)
    folded = jnp.sum(local, axis=-1, keepdims=True)
    return (local * folded).astype(jnp.bfloat16)


def boundary_function(boundary: str):
    if boundary == "forward":

        def forward(x):
            return singleton_broadcast(x)

        return forward
    if boundary == "backward":

        def backward(x, output_cotangent):
            _, pullback = jax.vjp(singleton_broadcast, x)
            return pullback(output_cotangent)[0]

        return backward
    if boundary == "composed":

        def composed(x, output_cotangent):
            result, pullback = jax.vjp(singleton_broadcast, x)
            return result, pullback(output_cotangent)[0]

        return composed
    raise ValueError(f"unsupported boundary {boundary!r}")


def input_signature(shape: StructuralShape, boundary: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    inputs = (("x", shape.dimensions),)
    if boundary == "forward":
        return inputs
    return (*inputs, ("output_cotangent", shape.dimensions))


def output_signature(shape: StructuralShape, boundary: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if boundary == "composed":
        return (("result", shape.dimensions), ("input_cotangent", shape.dimensions))
    role = "result" if boundary == "forward" else "input_cotangent"
    return ((role, shape.dimensions),)


def filename(shape: StructuralShape, boundary: str) -> str:
    return f"jax-{JAX_VERSION}-bf16-singleton_broadcast-{shape.shape_id}-{boundary}.mlir"


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


def signature_header(signature: tuple[tuple[str, tuple[int, ...]], ...]) -> str:
    return ", ".join(f"{role}={dimensions}:bfloat16" for role, dimensions in signature)


def audited_fixture(shape: StructuralShape, boundary: str) -> str:
    payload = export_stablehlo(shape, boundary)
    hook_payload = hook_boundary_stablehlo(payload)
    generator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()
    fields = json.dumps(
        {"boundary": boundary, "dimensions": shape.dimensions},
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
            "// Source: jax.jit(ordinary JAX terminal-axis keepdims sum plus JAX-owned",
            "// VJP).lower(...).compiler_ir(StableHLO)",
            f"// Case ID: {shape.case_id}",
            f"// Structural fields: {fields}",
            f"// Inputs: {signature_header(input_signature(shape, boundary))}",
            f"// Outputs: {signature_header(output_signature(shape, boundary))}",
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
        for boundary in BOUNDARIES:
            expected = audited_fixture(shape, boundary)
            path = arguments.output_dir / filename(shape, boundary)
            if arguments.write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(expected)
            elif not path.exists() or path.read_text() != expected:
                mismatches.append(path.name)
    if mismatches:
        parser.error("BF16 singleton broadcast fixture inventory drift: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
