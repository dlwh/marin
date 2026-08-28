# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ordinary-JAX normalized-exponential contraction fixtures."""

import argparse
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
PINNED_XLA_REVISION = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"


def causal_gqa(q, k, v):
    """Normalized causal GQA with a rank-five score domain."""
    scores = jnp.einsum("bhgqd,bhkd->bhgqk", q, k) * jnp.float32(0.125)
    query = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 17, 1)
    key = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 1, 17)
    predicate = jnp.broadcast_to(key <= query, scores.shape)
    masked = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
    maximum = jnp.max(masked, axis=4, keepdims=True)
    numerator = jnp.exp(masked - maximum)
    denominator = jnp.sum(numerator, axis=4, keepdims=True)
    weights = numerator / denominator
    return jnp.einsum("bhgqk,bhkv->bhgqv", weights, v)


def causal_gqa_with_output_epilogue(q, k, v):
    """Normalized causal GQA followed by a live scalar output epilogue."""
    output = causal_gqa(q, k, v)
    return jnp.tanh(output * jnp.float32(0.5)) + jnp.float32(0.25)


def causal_gqa_benchmark(q, k, v):
    """Representative normalized causal GQA with shape-derived extents."""
    scores = jnp.einsum("bhgqd,bhkd->bhgqk", q, k) * jnp.float32(q.shape[-1] ** -0.5)
    query_extent = q.shape[-2]
    key_extent = k.shape[-2]
    query = jnp.arange(query_extent, dtype=jnp.int32).reshape(1, 1, 1, query_extent, 1)
    key = jnp.arange(key_extent, dtype=jnp.int32).reshape(1, 1, 1, 1, key_extent)
    predicate = jnp.broadcast_to(key <= query, scores.shape)
    masked = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
    maximum = jnp.max(masked, axis=4, keepdims=True)
    numerator = jnp.exp(masked - maximum)
    denominator = jnp.sum(numerator, axis=4, keepdims=True)
    weights = numerator / denominator
    return jnp.einsum("bhgqk,bhkv->bhgqv", weights, v)


def causal_gqa_bf16_benchmark(q, k, v):
    """BF16 public attention with FP32 score and sufficient-state arithmetic."""
    return causal_gqa_benchmark(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)).astype(jnp.bfloat16)


def causal_gqa_bf16_axis4(q, k, v):
    """Tiny BF16 public fixture preserving the accepted axis-4 semantics."""
    return causal_gqa(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)).astype(jnp.bfloat16)


def causal_gqa_long_key(q, k, v):
    """Normalized causal GQA whose key extent exceeds the online scheduler."""
    scores = jnp.einsum("bhgqd,bhkd->bhgqk", q, k) * jnp.float32(0.125)
    query = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 17, 1)
    key = jnp.arange(257, dtype=jnp.int32).reshape(1, 1, 1, 1, 257)
    predicate = jnp.broadcast_to(key <= query, scores.shape)
    masked = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
    maximum = jnp.max(masked, axis=4, keepdims=True)
    numerator = jnp.exp(masked - maximum)
    denominator = jnp.sum(numerator, axis=4, keepdims=True)
    weights = numerator / denominator
    return jnp.einsum("bhgqk,bhkv->bhgqv", weights, v)


def causal_gqa_rank9(q, k, v):
    """Normalized causal GQA whose weighted Contract domain has rank nine."""
    scores = jnp.einsum("abcdhgqz,abcdhkz->abcdhgqk", q, k) * jnp.float32(0.125)
    query = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 1, 1, 1, 17, 1)
    key = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 1, 1, 1, 1, 17)
    predicate = jnp.broadcast_to(key <= query, scores.shape)
    masked = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
    maximum = jnp.max(masked, axis=7, keepdims=True)
    numerator = jnp.exp(masked - maximum)
    denominator = jnp.sum(numerator, axis=7, keepdims=True)
    weights = numerator / denominator
    return jnp.einsum("abcdhgqk,abcdhkv->abcdhgqv", weights, v)


def offset_softcap_gqa(q, k, v, *, scale: float, softcap: float):
    """The same computation with a different layout and predicate offset."""
    scores = jnp.einsum("qbhgd,kbhd->qkbhg", q, k) * jnp.float32(scale)
    scores = jnp.float32(softcap) * jnp.tanh(scores / jnp.float32(softcap))
    query = jnp.arange(19, dtype=jnp.int32).reshape(19, 1, 1, 1, 1)
    key = jnp.arange(23, dtype=jnp.int32).reshape(1, 23, 1, 1, 1)
    predicate = jnp.broadcast_to(key <= query + jnp.int32(4), scores.shape)
    masked = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
    maximum = jnp.max(masked, axis=1, keepdims=True)
    numerator = jnp.exp(masked - maximum)
    denominator = jnp.sum(numerator, axis=1, keepdims=True)
    weights = numerator / denominator
    return jnp.einsum("qkbhg,kbhv->qbhgv", weights, v)


def offset_softcap_gqa_primary(q, k, v):
    return offset_softcap_gqa(q, k, v, scale=0.1767766952966369, softcap=12.0)


def offset_softcap_gqa_mutation(q, k, v):
    return offset_softcap_gqa(q, k, v, scale=0.25, softcap=7.0)


def causal_gqa_with_public_scores(q, k, v):
    """Near miss whose public score result requires materialized fallback."""
    scores = jnp.einsum("bhgqd,bhkd->bhgqk", q, k) * jnp.float32(0.125)
    query = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 17, 1)
    key = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 1, 17)
    predicate = jnp.broadcast_to(key <= query, scores.shape)
    masked = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
    maximum = jnp.max(masked, axis=4, keepdims=True)
    numerator = jnp.exp(masked - maximum)
    denominator = jnp.sum(numerator, axis=4, keepdims=True)
    weights = numerator / denominator
    output = jnp.einsum("bhgqk,bhkv->bhgqv", weights, v)
    return output, masked


def causal_gqa_without_maximum(q, k, v):
    """Near miss without the maximum component of the online state."""
    scores = jnp.einsum("bhgqd,bhkd->bhgqk", q, k) * jnp.float32(0.125)
    query = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 17, 1)
    key = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 1, 17)
    predicate = jnp.broadcast_to(key <= query, scores.shape)
    masked = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
    numerator = jnp.exp(masked)
    denominator = jnp.sum(numerator, axis=4, keepdims=True)
    weights = numerator / denominator
    return jnp.einsum("bhgqk,bhkv->bhgqv", weights, v)


def causal_gqa_with_mismatched_reduction(q, k, v):
    """Near miss whose maximum and denominator reduce different coordinates."""
    scores = jnp.einsum("bhgqd,bhkd->bhgqk", q, k) * jnp.float32(0.125)
    query = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 17, 1)
    key = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 1, 17)
    predicate = jnp.broadcast_to(key <= query, scores.shape)
    masked = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
    maximum = jnp.max(masked, axis=3, keepdims=True)
    numerator = jnp.exp(masked - maximum)
    denominator = jnp.sum(numerator, axis=4, keepdims=True)
    weights = numerator / denominator
    return jnp.einsum("bhgqk,bhkv->bhgqv", weights, v)


def causal_gqa_with_alternate_score_contract(q, k, v):
    """Near miss whose representative score has two Contract ancestors."""
    primary = jnp.einsum("bhgqd,bhkd->bhgqk", q, k)
    alternate = jnp.einsum("bhgqd,bhkd->bhgqk", q, k * jnp.float32(0.5))
    scores = (primary + alternate) * jnp.float32(0.125)
    query = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 17, 1)
    key = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 1, 17)
    predicate = jnp.broadcast_to(key <= query, scores.shape)
    masked = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
    maximum = jnp.max(masked, axis=4, keepdims=True)
    numerator = jnp.exp(masked - maximum)
    denominator = jnp.sum(numerator, axis=4, keepdims=True)
    weights = numerator / denominator
    return jnp.einsum("bhgqk,bhkv->bhgqv", weights, v)


@dataclass(frozen=True)
class Fixture:
    filename: str
    function: Callable[..., object]
    shapes: tuple[tuple[int, ...], ...]
    expression: str
    alpha_rename: bool = False
    argument_dtype: str = "float32"


AXIS4_SHAPES = ((1, 2, 2, 17, 8), (1, 2, 17, 8), (1, 2, 17, 7))
BENCHMARK_256_SHAPES = (
    (1, 8, 4, 256, 64),
    (1, 8, 256, 64),
    (1, 8, 256, 64),
)
BENCHMARK_2048_SHAPES = (
    (1, 8, 4, 2048, 128),
    (1, 8, 2048, 128),
    (1, 8, 2048, 128),
)
LONG_KEY_SHAPES = ((1, 2, 2, 17, 8), (1, 2, 257, 8), (1, 2, 257, 7))
RANK9_SHAPES = (
    (1, 1, 1, 1, 2, 2, 17, 8),
    (1, 1, 1, 1, 2, 17, 8),
    (1, 1, 1, 1, 2, 17, 7),
)
AXIS1_SHAPES = ((19, 2, 3, 2, 5), (23, 2, 3, 5), (23, 2, 3, 11))

FIXTURES = (
    Fixture(
        "jax-0.10.1-f32-causal-gqa-online-axis4.mlir",
        causal_gqa,
        AXIS4_SHAPES,
        "causal GQA [batch, kv_head, group, query, key], reduction axis 4",
    ),
    Fixture(
        "jax-0.10.1-f32-offset-softcap-gqa-online-axis1.mlir",
        offset_softcap_gqa_primary,
        AXIS1_SHAPES,
        "offset causal softcap GQA [query, key, batch, kv_head, group], reduction axis 1",
        alpha_rename=True,
    ),
    Fixture(
        "jax-0.10.1-f32-offset-softcap-gqa-online-axis1-mutation.mlir",
        offset_softcap_gqa_mutation,
        AXIS1_SHAPES,
        "semantic scale/softcap mutation of the axis-1 fixture",
        alpha_rename=True,
    ),
    Fixture(
        "jax-0.10.1-f32-causal-gqa-online-output-epilogue.mlir",
        causal_gqa_with_output_epilogue,
        AXIS4_SHAPES,
        "causal GQA with a live tanh/multiply/add output epilogue",
    ),
    Fixture(
        "jax-0.10.1-f32-causal-gqa-online-benchmark-s256-d64.mlir",
        causal_gqa_benchmark,
        BENCHMARK_256_SHAPES,
        "representative causal GQA B1 Hq32 Hkv8 S256 D=Dv64",
    ),
    Fixture(
        "jax-0.10.1-f32-causal-gqa-online-benchmark-s2048-d128.mlir",
        causal_gqa_benchmark,
        BENCHMARK_2048_SHAPES,
        "representative causal GQA B1 Hq32 Hkv8 S2048 D=Dv128",
    ),
    Fixture(
        "jax-0.10.1-bf16-causal-gqa-online-benchmark-s256-d64.mlir",
        causal_gqa_bf16_benchmark,
        BENCHMARK_256_SHAPES,
        "BF16 causal GQA B1 Hq32 Hkv8 S256 D=Dv64 with FP32 state",
        argument_dtype="bfloat16",
    ),
    Fixture(
        "jax-0.10.1-bf16-causal-gqa-online-axis4.mlir",
        causal_gqa_bf16_axis4,
        AXIS4_SHAPES,
        "BF16 causal GQA axis-4 fixture with FP32 sufficient state",
        argument_dtype="bfloat16",
    ),
    Fixture(
        "jax-0.10.1-bf16-causal-gqa-online-benchmark-s2048-d128.mlir",
        causal_gqa_bf16_benchmark,
        BENCHMARK_2048_SHAPES,
        "BF16 causal GQA B1 Hq32 Hkv8 S2048 D=Dv128 with FP32 state",
        argument_dtype="bfloat16",
    ),
    Fixture(
        "jax-0.10.1-f32-causal-gqa-online-key257-fallback.mlir",
        causal_gqa_long_key,
        LONG_KEY_SHAPES,
        "causal GQA with key extent 257 requiring generic fallback",
    ),
    Fixture(
        "jax-0.10.1-f32-causal-gqa-online-rank9-fallback.mlir",
        causal_gqa_rank9,
        RANK9_SHAPES,
        "causal GQA with a rank-nine weighted Contract domain requiring generic fallback",
    ),
    Fixture(
        "jax-0.10.1-f32-causal-gqa-public-score-near-miss.mlir",
        causal_gqa_with_public_scores,
        AXIS4_SHAPES,
        "near miss with a public representative-score result",
    ),
    Fixture(
        "jax-0.10.1-f32-causal-gqa-no-maximum-near-miss.mlir",
        causal_gqa_without_maximum,
        AXIS4_SHAPES,
        "near miss without maximum sufficient-state semantics",
    ),
    Fixture(
        "jax-0.10.1-f32-causal-gqa-mismatched-reduction-near-miss.mlir",
        causal_gqa_with_mismatched_reduction,
        AXIS4_SHAPES,
        "near miss whose maximum and denominator reduce different coordinates",
    ),
    Fixture(
        "jax-0.10.1-f32-causal-gqa-alternate-score-contract-near-miss.mlir",
        causal_gqa_with_alternate_score_contract,
        AXIS4_SHAPES,
        "near miss whose representative score has two Contract ancestors",
    ),
)


def export_stablehlo(fixture: Fixture) -> str:
    dtype = getattr(jnp, fixture.argument_dtype)
    arguments = tuple(jax.ShapeDtypeStruct(shape, dtype) for shape in fixture.shapes)
    module = jax.jit(fixture.function).lower(*arguments).compiler_ir(dialect="stablehlo")
    return f"{module}".rstrip() + "\n"


def alpha_rename_ssa(payload: str) -> str:
    names = re.findall(r"(?m)^    (%(?:[0-9]+|cst(?:_[0-9]+)?|c)) =", payload)
    replacements = {name: f"%layout_b_value_{ordinal}" for ordinal, name in enumerate(names)}
    replacements.update({f"%arg{ordinal}": f"%layout_b_input_{ordinal}" for ordinal in range(3)})
    for source in sorted(replacements, key=len, reverse=True):
        payload = re.sub(rf"{re.escape(source)}(?![A-Za-z0-9_])", replacements[source], payload)
    return payload


def audited_fixture(fixture: Fixture) -> str:
    ordinary_jax = export_stablehlo(fixture)
    payload = alpha_rename_ssa(ordinary_jax) if fixture.alpha_rename else ordinary_jax
    ordinary_digest = hashlib.sha256(ordinary_jax.encode()).hexdigest().upper()
    fixture_digest = hashlib.sha256(payload.encode()).hexdigest().upper()
    shapes = ", ".join(str(shape) for shape in fixture.shapes)
    header = (
        "// Copyright The Marin Authors",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        "// Ordinary-JAX export audit",
        f"// Generator: {Path(__file__).name} ({fixture.filename})",
        '// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")',
        f"// Expression: {fixture.expression}",
        f"// Inputs: {shapes}:f32",
        f"// JAX: {JAX_VERSION}; jaxlib: {JAXLIB_VERSION}; XLA: {PINNED_XLA_REVISION}",
        f"// Raw ordinary-JAX StableHLO SHA-256: {ordinary_digest}",
        f"// Fixture StableHLO SHA-256: {fixture_digest}",
        f"// Textual SSA alpha-renaming: {str(fixture.alpha_rename).lower()}",
        "",
    )
    return "\n".join(header) + payload


def main() -> None:
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise SystemExit(
            "fixture regeneration requires "
            f"jax=={JAX_VERSION} and jaxlib=={JAXLIB_VERSION}; found "
            f"jax=={jax.__version__} and jaxlib=={jaxlib.__version__}"
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent
    stale = []
    for fixture in FIXTURES:
        expected = audited_fixture(fixture)
        path = root / fixture.filename
        if args.write:
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            stale.append(path.name)
    if stale:
        raise SystemExit("stale ordinary-JAX fixtures: " + ", ".join(stale))


if __name__ == "__main__":
    main()
