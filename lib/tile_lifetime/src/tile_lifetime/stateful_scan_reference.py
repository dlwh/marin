# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordinary JAX affine-state scan used by the natural StableHLO frontend."""

from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp

from shuttle.ir import DType
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    EvaluationOrder,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    ScalarExpressionKind,
    ScanDirection,
    ScanInputBinding,
    ScanOutputBinding,
    ScanPrimitive,
    ScanStateBinding,
    TensorAxis,
    TensorProgram,
    index_axis,
    scalar_binary,
    scalar_input,
    scalar_unary,
)


class ScanDecayAxes(StrEnum):
    """Logical domain of the multiplicative state decay."""

    SCALAR = "scalar"
    KEY = "key"


class ScanDiagonalOperation(StrEnum):
    """Natural JAX expression used to form the diagonal state transition."""

    EXP = "exp"
    EXP_SQUARED = "exp_squared"


@dataclass(frozen=True)
class NaturalAffineScanConfig:
    """Static shape and algebra choices for an ordinary affine recurrence."""

    batch: int = 1
    sequence: int = 4
    heads: int = 2
    key_dimension: int = 8
    value_dimension: int = 12
    update_rank: int = 1
    decay_axes: ScanDecayAxes = ScanDecayAxes.SCALAR
    diagonal_operation: ScanDiagonalOperation = ScanDiagonalOperation.EXP

    def __post_init__(self) -> None:
        dimensions = (
            self.batch,
            self.sequence,
            self.heads,
            self.key_dimension,
            self.value_dimension,
            self.update_rank,
        )
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("stateful-scan dimensions must be positive")


NATURAL_AFFINE_SCAN_INPUT_NAMES = (
    "query",
    "key",
    "value",
    "log_decay",
    "beta",
    "initial_state",
)
NATURAL_AFFINE_SCAN_BACKWARD_INPUT_NAMES = (
    *NATURAL_AFFINE_SCAN_INPUT_NAMES,
    "output_cotangent",
    "final_state_cotangent",
)


def natural_affine_scan_region(config: NaturalAffineScanConfig):
    """Return natural JAX tensor/state math for a bounded-rank affine scan."""

    def region(query, key, value, log_decay, beta, initial_state):
        def step(state, inputs):
            query_token, key_token, value_token, log_decay_token, beta_token = inputs
            decay = jnp.exp(log_decay_token)
            if config.diagonal_operation is ScanDiagonalOperation.EXP_SQUARED:
                decay = decay * decay
            if config.decay_axes is ScanDecayAxes.SCALAR:
                decay = decay[..., None, None]
            else:
                decay = decay[..., :, None]
            decayed_state = state * decay
            prediction = jnp.einsum(
                "bhkv,bhrk->bhrv",
                decayed_state,
                key_token,
                preferred_element_type=jnp.float32,
            )
            delta = beta_token[..., None] * (value_token.astype(jnp.float32) - prediction)
            state_next = decayed_state + jnp.einsum(
                "bhrk,bhrv->bhkv",
                key_token,
                delta,
                preferred_element_type=jnp.float32,
            )
            output = jnp.einsum(
                "bhkv,bhk->bhv",
                state_next,
                query_token,
                preferred_element_type=jnp.float32,
            )
            return state_next, output

        scan_inputs = tuple(jnp.swapaxes(value, 0, 1) for value in (query, key, value, log_decay, beta))
        final_state, output = jax.lax.scan(step, initial_state, scan_inputs)
        return jnp.swapaxes(output, 0, 1), final_state

    return region


def natural_affine_scan_body_program(config: NaturalAffineScanConfig) -> TensorProgram:
    """Build the typed generic Map/Contract body of one scan step."""
    batch = TensorAxis(0, config.batch, "batch")
    head = TensorAxis(2, config.heads, "head")
    key_axis = TensorAxis(3, config.key_dimension, "key")
    value_axis = TensorAxis(4, config.value_dimension, "value")
    rank = TensorAxis(5, config.update_rank, "update_rank")
    state_axes = (batch, head, key_axis, value_axis)
    ranked_key_axes = (batch, head, rank, key_axis)
    ranked_value_axes = (batch, head, rank, value_axis)
    rank_axes = (batch, head, rank)
    decay_axes = (batch, head) if config.decay_axes is ScanDecayAxes.SCALAR else (batch, head, key_axis)

    query = ProgramValue("query", (batch, head, key_axis), DType.BF16)
    key = ProgramValue("key", ranked_key_axes, DType.BF16)
    value = ProgramValue("value", ranked_value_axes, DType.BF16)
    log_decay = ProgramValue("log_decay", decay_axes, DType.FP32)
    beta = ProgramValue("beta", rank_axes, DType.FP32)
    state_prev = ProgramValue("state_prev", state_axes, DType.FP32)
    decay = ProgramValue("decay", decay_axes, DType.FP32)
    if config.diagonal_operation is ScanDiagonalOperation.EXP_SQUARED:
        exponential = scalar_unary(ScalarExpressionKind.EXP, scalar_input("log_decay"))
        decay_expression = scalar_binary(ScalarExpressionKind.MULTIPLY, exponential, exponential)
    else:
        decay_expression = scalar_unary(ScalarExpressionKind.EXP, scalar_input("log_decay"))
    state_decayed = ProgramValue("state_decayed", state_axes, DType.FP32)
    prediction = ProgramValue("prediction", ranked_value_axes, DType.FP32)
    residual = ProgramValue("residual", ranked_value_axes, DType.FP32)
    scaled_residual = ProgramValue("scaled_residual", ranked_value_axes, DType.FP32)
    correction = ProgramValue("correction", state_axes, DType.FP32)
    state_next = ProgramValue("state_next", state_axes, DType.FP32)
    output = ProgramValue("output", (batch, head, value_axis), DType.FP32)
    operations = (
        MapPrimitive("decay", (log_decay,), decay, decay_expression),
        MapPrimitive(
            "decay_state",
            (decay, state_prev),
            state_decayed,
            scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_input("decay"),
                scalar_input("state_prev"),
            ),
        ),
        ContractPrimitive(
            "predict",
            (state_decayed, key),
            prediction,
            (key_axis,),
            DType.FP32,
        ),
        MapPrimitive(
            "residual",
            (value, prediction),
            residual,
            scalar_binary(
                ScalarExpressionKind.SUBTRACT,
                scalar_input("value"),
                scalar_input("prediction"),
            ),
        ),
        MapPrimitive(
            "scale_residual",
            (beta, residual),
            scaled_residual,
            scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_input("beta"),
                scalar_input("residual"),
            ),
        ),
        ContractPrimitive(
            "rank_update",
            (key, scaled_residual),
            correction,
            (rank,),
            DType.FP32,
        ),
        MapPrimitive(
            "write_state",
            (state_decayed, correction),
            state_next,
            scalar_binary(
                ScalarExpressionKind.ADD,
                scalar_input("state_decayed"),
                scalar_input("correction"),
            ),
        ),
        ContractPrimitive(
            "read_state",
            (state_next, query),
            output,
            (key_axis,),
            DType.FP32,
        ),
    )
    return TensorProgram(
        inputs=(query, key, value, log_decay, beta, state_prev),
        operations=operations,
        outputs=(output, state_next, state_prev),
    )


def natural_affine_scan_program(
    config: NaturalAffineScanConfig,
    *,
    direction: ScanDirection = ScanDirection.FORWARD,
    numerical_policy: NumericalPolicy = NumericalPolicy.SOURCE_ORDERED,
) -> TensorProgram:
    """Translate the ordinary recurrence to one generic typed Scan primitive."""
    body = natural_affine_scan_body_program(config)
    query_body, key_body, value_body, log_decay_body, beta_body, state_body = body.inputs
    output_body, state_next_body, state_prev_body = body.outputs
    batch = TensorAxis(0, config.batch, "batch")
    ordered = TensorAxis(1, config.sequence, "ordered")
    head = TensorAxis(2, config.heads, "parallel")
    key_axis = TensorAxis(3, config.key_dimension, "state_row")
    value_axis = TensorAxis(4, config.value_dimension, "state_column")
    rank = TensorAxis(5, config.update_rank, "update_rank")
    query = ProgramValue("query", (batch, ordered, head, key_axis), DType.BF16)
    key = ProgramValue("key", (batch, ordered, head, rank, key_axis), DType.BF16)
    value = ProgramValue("value", (batch, ordered, head, rank, value_axis), DType.BF16)
    log_decay_axes = (
        (batch, ordered, head) if config.decay_axes is ScanDecayAxes.SCALAR else (batch, ordered, head, key_axis)
    )
    log_decay = ProgramValue("log_decay", log_decay_axes, DType.FP32)
    beta = ProgramValue("beta", (batch, ordered, head, rank), DType.FP32)
    initial_state = ProgramValue("initial_state", (batch, head, key_axis, value_axis), DType.FP32)
    final_state = ProgramValue("final_state", initial_state.axes, DType.FP32)
    output = ProgramValue("output", (batch, ordered, head, value_axis), DType.FP32)
    state_history = ProgramValue(
        "state_history",
        (batch, ordered, head, key_axis, value_axis),
        DType.FP32,
    )

    def identity_indexing(operand: ProgramValue) -> OperandIndexing:
        return OperandIndexing(
            operand=operand,
            domain_axes=operand.axes,
            indices=tuple(index_axis(axis) for axis in operand.axes),
        )

    scan = ScanPrimitive(
        name="ordered_state",
        axis=ordered,
        body=body,
        state=(ScanStateBinding(initial_state, state_body, state_next_body, final_state),),
        scanned_inputs=(
            ScanInputBinding(query, query_body, identity_indexing(query)),
            ScanInputBinding(key, key_body, identity_indexing(key)),
            ScanInputBinding(value, value_body, identity_indexing(value)),
            ScanInputBinding(log_decay, log_decay_body, identity_indexing(log_decay)),
            ScanInputBinding(beta, beta_body, identity_indexing(beta)),
        ),
        emitted_outputs=(
            ScanOutputBinding(output_body, output, identity_indexing(output)),
            ScanOutputBinding(state_prev_body, state_history, identity_indexing(state_history)),
        ),
        direction=direction,
        order=EvaluationOrder.SOURCE_ORDERED,
    )
    return TensorProgram(
        inputs=(query, key, value, log_decay, beta, initial_state),
        operations=(scan,),
        outputs=(output, final_state),
        numerical_policy=numerical_policy,
    )


def export_natural_affine_scan(
    config: NaturalAffineScanConfig = NaturalAffineScanConfig(),
) -> bytes:
    """Export the natural recurrence as portable StableHLO containing ``while``."""
    exported = jax.export.export(jax.jit(natural_affine_scan_region(config)))(
        *natural_affine_scan_input_specifications(config)
    )
    return exported.mlir_module_serialized


def natural_affine_scan_value_and_grad(config: NaturalAffineScanConfig):
    """Return a scalar objective and its JAX-owned input gradients."""
    region = natural_affine_scan_region(config)

    def objective(*arguments):
        inputs = arguments[: len(NATURAL_AFFINE_SCAN_INPUT_NAMES)]
        output_cotangent, final_state_cotangent = arguments[len(NATURAL_AFFINE_SCAN_INPUT_NAMES) :]
        output, final_state = region(*inputs)
        return jnp.vdot(output, output_cotangent) + jnp.vdot(final_state, final_state_cotangent)

    differentiated = jax.value_and_grad(
        objective,
        argnums=tuple(range(len(NATURAL_AFFINE_SCAN_INPUT_NAMES))),
    )

    def value_and_grad(*arguments):
        value, gradients = differentiated(*arguments)
        return value, *gradients

    return value_and_grad


def natural_affine_scan_training_step(config: NaturalAffineScanConfig):
    """Compose the ordinary forward with its JAX-owned pullback."""
    region = natural_affine_scan_region(config)

    def training_step(*arguments):
        inputs = arguments[: len(NATURAL_AFFINE_SCAN_INPUT_NAMES)]
        output_cotangent, final_state_cotangent = arguments[len(NATURAL_AFFINE_SCAN_INPUT_NAMES) :]
        (output, final_state), pullback = jax.vjp(region, *inputs)
        gradients = pullback((output_cotangent, final_state_cotangent))
        return output, final_state, *gradients

    return training_step


def export_natural_affine_scan_value_and_grad(
    config: NaturalAffineScanConfig = NaturalAffineScanConfig(),
) -> bytes:
    """Export the objective and JAX-owned reverse as portable StableHLO."""
    exported = jax.export.export(jax.jit(natural_affine_scan_value_and_grad(config)))(
        *_natural_affine_scan_backward_specifications(config)
    )
    return exported.mlir_module_serialized


def export_natural_affine_scan_value_and_grad_text(
    config: NaturalAffineScanConfig = NaturalAffineScanConfig(),
) -> str:
    """Render the exported JAX-owned reverse for provenance checks."""
    exported = jax.export.export(jax.jit(natural_affine_scan_value_and_grad(config)))(
        *_natural_affine_scan_backward_specifications(config)
    )
    return exported.mlir_module()


def export_natural_affine_scan_training_step(
    config: NaturalAffineScanConfig = NaturalAffineScanConfig(),
) -> bytes:
    """Export the composed forward-and-reverse boundary as StableHLO."""
    exported = jax.export.export(jax.jit(natural_affine_scan_training_step(config)))(
        *_natural_affine_scan_backward_specifications(config)
    )
    return exported.mlir_module_serialized


def export_natural_affine_scan_training_step_text(
    config: NaturalAffineScanConfig = NaturalAffineScanConfig(),
) -> str:
    """Render the composed forward-and-reverse boundary for provenance checks."""
    exported = jax.export.export(jax.jit(natural_affine_scan_training_step(config)))(
        *_natural_affine_scan_backward_specifications(config)
    )
    return exported.mlir_module()


def natural_affine_scan_input_specifications(
    config: NaturalAffineScanConfig,
) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Return the public input contract for the ordinary scan boundary."""
    bf16 = jnp.bfloat16
    decay_shape = (
        (config.batch, config.sequence, config.heads)
        if config.decay_axes is ScanDecayAxes.SCALAR
        else (config.batch, config.sequence, config.heads, config.key_dimension)
    )
    return (
        jax.ShapeDtypeStruct((config.batch, config.sequence, config.heads, config.key_dimension), bf16),
        jax.ShapeDtypeStruct(
            (config.batch, config.sequence, config.heads, config.update_rank, config.key_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(
            (config.batch, config.sequence, config.heads, config.update_rank, config.value_dimension),
            bf16,
        ),
        jax.ShapeDtypeStruct(decay_shape, jnp.float32),
        jax.ShapeDtypeStruct(
            (config.batch, config.sequence, config.heads, config.update_rank),
            jnp.float32,
        ),
        jax.ShapeDtypeStruct(
            (config.batch, config.heads, config.key_dimension, config.value_dimension),
            jnp.float32,
        ),
    )


def _natural_affine_scan_backward_specifications(
    config: NaturalAffineScanConfig,
) -> tuple[jax.ShapeDtypeStruct, ...]:
    output_cotangent = jax.ShapeDtypeStruct(
        (config.batch, config.sequence, config.heads, config.value_dimension),
        jnp.float32,
    )
    final_state_cotangent = jax.ShapeDtypeStruct(
        (config.batch, config.heads, config.key_dimension, config.value_dimension),
        jnp.float32,
    )
    return (*natural_affine_scan_input_specifications(config), output_cotangent, final_state_cotangent)
