# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ordinary-JAX causal relation and Fold conversion fixtures."""

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


def causal_score_relation(scores):
    """Return a causal row maximum and a live i1-to-f32 relation projection."""
    query_position = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 17, 1)
    key_position = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 1, 17)
    valid = key_position <= query_position
    valid_f32 = valid.astype(jnp.float32)
    predicate = jnp.broadcast_to(valid, scores.shape)
    scaled_scores = scores * jnp.float32(0.125)
    masked_scores = jax.lax.select(
        predicate,
        scaled_scores,
        jnp.full(scores.shape, -jnp.inf, dtype=jnp.float32),
    )
    return jnp.max(masked_scores, axis=4), jnp.sum(valid_f32, axis=4)


def offset_causal_score_relation(scores, *, scale: float, softcap: float):
    """Return the same relation semantics in a different axis arrangement."""
    query_position = jnp.arange(19, dtype=jnp.int32).reshape(19, 1, 1, 1, 1)
    key_position = jnp.arange(23, dtype=jnp.int32).reshape(1, 23, 1, 1, 1)
    valid = key_position <= query_position + jnp.int32(4)
    valid_f32 = valid.astype(jnp.float32)
    predicate = jnp.broadcast_to(valid, scores.shape)
    scaled_scores = scores * jnp.float32(scale)
    capped_scores = jnp.float32(softcap) * jnp.tanh(scaled_scores / jnp.float32(softcap))
    masked_scores = jax.lax.select(
        predicate,
        capped_scores,
        jnp.full(scores.shape, -jnp.inf, dtype=jnp.float32),
    )
    return jnp.max(masked_scores, axis=1), jnp.sum(valid_f32, axis=1)


def offset_causal_score_relation_primary(scores):
    return offset_causal_score_relation(scores, scale=0.1767766952966369, softcap=12.0)


def offset_causal_score_relation_mutation(scores):
    return offset_causal_score_relation(scores, scale=0.25, softcap=7.0)


@dataclass(frozen=True)
class Fixture:
    filename: str
    function: Callable[..., object]
    shape: tuple[int, ...]
    expression: str
    alpha_rename: bool = False


FIXTURES = (
    Fixture(
        "jax-0.10.1-f32-causal-relation-max-axis4.mlir",
        causal_score_relation,
        (1, 2, 2, 17, 17),
        "causal mask over [batch, kv_head, group, query, key], max/sum on axis 4",
    ),
    Fixture(
        "jax-0.10.1-f32-offset-causal-softcap-max-axis1.mlir",
        offset_causal_score_relation_primary,
        (19, 23, 2, 3, 2),
        "offset causal mask over [query, key, batch, kv_head, group], softcap 12, max/sum on axis 1",
        alpha_rename=True,
    ),
    Fixture(
        "jax-0.10.1-f32-offset-causal-softcap-max-axis1-mutation.mlir",
        offset_causal_score_relation_mutation,
        (19, 23, 2, 3, 2),
        "semantic mutation of the axis-1 fixture: scale 0.25 and softcap 7",
        alpha_rename=True,
    ),
)


def export_stablehlo(fixture: Fixture) -> str:
    argument = jax.ShapeDtypeStruct(fixture.shape, jnp.float32)
    module = jax.jit(fixture.function).lower(argument).compiler_ir(dialect="stablehlo")
    return f"{module}".rstrip() + "\n"


def alpha_rename_ssa(payload: str) -> str:
    """Apply a textual alpha-renaming that cannot change parsed MLIR semantics."""
    names = re.findall(r"(?m)^    (%(?:[0-9]+|cst(?:_[0-9]+)?|c)) =", payload)
    replacements = {name: f"%renamed_value_{ordinal}" for ordinal, name in enumerate(names)}
    replacements["%arg0"] = "%renamed_scores"
    for source in sorted(replacements, key=len, reverse=True):
        payload = re.sub(rf"{re.escape(source)}(?![A-Za-z0-9_])", replacements[source], payload)
    return payload


def audited_fixture(fixture: Fixture) -> str:
    ordinary_jax = export_stablehlo(fixture)
    payload = alpha_rename_ssa(ordinary_jax) if fixture.alpha_rename else ordinary_jax
    ordinary_digest = hashlib.sha256(ordinary_jax.encode()).hexdigest().upper()
    fixture_digest = hashlib.sha256(payload.encode()).hexdigest().upper()
    header_lines = (
        "// Copyright The Marin Authors",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        "// Ordinary-JAX export audit",
        f"// Generator: {Path(__file__).name} ({fixture.filename})",
        '// Export: jax.jit(fixture.function).lower(f32_shape_struct).compiler_ir(dialect="stablehlo")',
        f"// Expression: {fixture.expression}",
        f"// Input: {fixture.shape}:f32",
        f"// JAX: {JAX_VERSION}; jaxlib: {JAXLIB_VERSION}; XLA: {PINNED_XLA_REVISION}",
        f"// Raw ordinary-JAX StableHLO SHA-256: {ordinary_digest}",
        f"// Fixture StableHLO SHA-256: {fixture_digest}",
        f"// Textual SSA alpha-renaming: {str(fixture.alpha_rename).lower()}",
    )
    return "\n".join(header_lines) + "\n\n" + payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError(
            f"fixture audit requires JAX {JAX_VERSION} and jaxlib {JAXLIB_VERSION}; "
            f"found {jax.__version__} and {jaxlib.__version__}"
        )
    mismatches = []
    for fixture in FIXTURES:
        expected = audited_fixture(fixture)
        path = arguments.output_dir / fixture.filename
        if arguments.write:
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            mismatches.append(fixture.filename)
    if mismatches:
        parser.error("fixture audit failed: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
