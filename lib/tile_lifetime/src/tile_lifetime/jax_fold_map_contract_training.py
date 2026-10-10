# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX composition for structurally provided Fold/Map/Contract training plans."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from shuttle.ir import DType
from tile_lifetime.fold_map_contract_training_provider import FoldMapContractTrainingProvider
from tile_lifetime.jax_axis_fold_ffi import call_cuda_axis_fold_ffi
from tile_lifetime.row_normalization_training import FoldMapContractTrainingPlan
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    IndexExpression,
    IndexExpressionKind,
    MapPrimitive,
    ProgramValue,
    RelationPrimitive,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    primitive_outputs,
)


def execute_fold_map_contract_training_jax(
    provider: FoldMapContractTrainingProvider,
    plan: FoldMapContractTrainingPlan,
    source_inputs: Mapping[str, jax.Array],
    output_cotangents: Mapping[str, jax.Array],
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Execute forward algebra and its automatic adjoint with generated Fold regions."""
    forward = execute_jax_tensor_program_values(plan.source, source_inputs)
    backward_inputs: dict[str, jax.Array] = {}
    for value in plan.backward.inputs:
        if value.name in source_inputs:
            backward_inputs[value.name] = source_inputs[value.name]
        elif value.name in forward:
            backward_inputs[value.name] = forward[value.name]
        elif value.name in output_cotangents:
            backward_inputs[value.name] = output_cotangents[value.name]
        else:
            raise ValueError(f"missing automatic-adjoint boundary value {value.name!r}")
    backward = execute_jax_tensor_program_values(plan.backward, backward_inputs, provider=provider)
    return (
        tuple(forward[value.name] for value in plan.source.outputs),
        tuple(backward[value.name] for value in plan.backward.outputs),
    )


def execute_jax_tensor_program_values(
    program: TensorProgram,
    inputs: Mapping[str, jax.Array],
    *,
    provider: FoldMapContractTrainingProvider | None = None,
) -> dict[str, jax.Array]:
    """Execute the supported generic TensorProgram subset using JAX operations and provided Folds."""
    environment = {value: inputs[value.name] for value in program.inputs}
    regions = {} if provider is None else {region.launch_operation_ordinal: region for region in provider.regions}
    provided_folds = {
        binding.fold: region
        for region in (() if provider is None else provider.regions)
        for binding in region.fold_bindings
    }
    for ordinal, operation in enumerate(program.operations):
        region = regions.get(ordinal)
        if region is not None:
            parameter_values: dict[str, ProgramValue] = {}
            for binding in region.fold_bindings:
                for item in binding.inputs:
                    parameter_values[item.parameter] = item.value
            ffi_inputs = {buffer.name: environment[parameter_values[buffer.name]] for buffer in region.generated.inputs}
            results = call_cuda_axis_fold_ffi(region.generated, ffi_inputs)
            for binding, result in zip(region.fold_bindings, results, strict=True):
                environment[binding.fold.output] = result
        if isinstance(operation, FoldPrimitive) and operation in provided_folds:
            continue
        results = _execute_primitive(operation, environment)
        for value, result in zip(primitive_outputs(operation), results, strict=True):
            environment[value] = result.astype(_jax_dtype(value.dtype))
    return {value.name: array for value, array in environment.items()}


def _execute_primitive(
    operation: TensorPrimitive,
    environment: Mapping[ProgramValue, jax.Array],
) -> tuple[jax.Array, ...]:
    if isinstance(operation, MapPrimitive):
        if operation.predicate is not None:
            raise ValueError("JAX training composition does not yet lower predicated scalar Maps")
        values = {
            value.name: _broadcast(environment[value], value.axes, operation.output.axes) for value in operation.inputs
        }
        return (jnp.broadcast_to(_evaluate_scalar(operation.expression, values), operation.output.shape),)
    if isinstance(operation, FoldPrimitive):
        dimensions = tuple(operation.input.axes.index(axis) for axis in operation.reduction_axes)
        value = environment[operation.input].astype(_jax_dtype(operation.accumulation_dtype))
        if operation.reducer is FoldReducer.SUM:
            return (jnp.sum(value, axis=dimensions),)
        return (jnp.max(value, axis=dimensions),)
    if isinstance(operation, RelationPrimitive):
        return _execute_relation(operation, environment)
    if isinstance(operation, ContractPrimitive):
        if len(operation.inputs) != 2:
            raise ValueError("JAX training composition currently requires binary Contracts")
        arrays: list[jax.Array] = []
        operand_axes: list[tuple[TensorAxis, ...]] = []
        for input_index, value in enumerate(operation.inputs):
            array = environment[value].astype(_jax_dtype(operation.accumulation_dtype))
            axes = list(value.axes)
            for mapping in operation.index_maps_for_input(input_index):
                dimension = axes.index(mapping.operand_axis)
                array = jnp.take(array, jnp.asarray(mapping.indices()), axis=dimension)
                axes[dimension] = mapping.domain_axis
            arrays.append(array)
            operand_axes.append(tuple(axes))
        left_axes, right_axes = operand_axes
        batch_axes = tuple(axis for axis in operation.output.axes if axis in left_axes and axis in right_axes)
        dimension_numbers = (
            (
                tuple(left_axes.index(axis) for axis in operation.reduction_axes),
                tuple(right_axes.index(axis) for axis in operation.reduction_axes),
            ),
            (
                tuple(left_axes.index(axis) for axis in batch_axes),
                tuple(right_axes.index(axis) for axis in batch_axes),
            ),
        )
        result = jax.lax.dot_general(
            arrays[0],
            arrays[1],
            dimension_numbers,
            preferred_element_type=_jax_dtype(operation.accumulation_dtype),
        )
        physical_axes = (
            *batch_axes,
            *(axis for axis in left_axes if axis not in operation.reduction_axes and axis not in batch_axes),
            *(axis for axis in right_axes if axis not in operation.reduction_axes and axis not in batch_axes),
        )
        if physical_axes != operation.output.axes:
            result = jnp.transpose(result, tuple(physical_axes.index(axis) for axis in operation.output.axes))
        return (result,)
    raise ValueError(f"JAX training composition does not support {type(operation).__name__}")


def _execute_relation(
    operation: RelationPrimitive,
    environment: Mapping[ProgramValue, jax.Array],
) -> tuple[jax.Array, ...]:
    if operation.predicate is not None or any(indexing.predicate is not None for indexing in operation.input_indexing):
        raise ValueError("JAX training composition does not yet lower predicated Relations")
    if any(expression is not None for expression in operation.resolved_result_expressions):
        raise ValueError("JAX training composition requires Relation results sourced from operands")
    coordinates = {
        axis: jnp.broadcast_to(
            jnp.arange(axis.extent, dtype=jnp.int32).reshape(
                tuple(axis.extent if index == position else 1 for index in range(len(operation.domain_axes)))
            ),
            tuple(axis.extent for axis in operation.domain_axes),
        )
        for position, axis in enumerate(operation.domain_axes)
    }
    gathered = {
        indexing.operand: environment[indexing.operand][
            tuple(_evaluate_index(expression, coordinates) for expression in indexing.indices)
        ]
        for indexing in operation.input_indexing
    }
    results: list[jax.Array] = []
    for output, source in zip(operation.outputs, operation.resolved_result_sources, strict=True):
        if source is None:
            raise ValueError("JAX training composition requires Relation results sourced from operands")
        results.append(jnp.broadcast_to(gathered[source], output.shape))
    return tuple(results)


def _evaluate_index(expression: IndexExpression, coordinates: Mapping[TensorAxis, jax.Array]) -> jax.Array:
    if expression.kind is IndexExpressionKind.AXIS:
        assert expression.axis is not None
        return coordinates[expression.axis]
    if expression.kind is IndexExpressionKind.CONSTANT:
        assert expression.constant is not None
        return jnp.asarray(expression.constant, dtype=jnp.int32)
    raise ValueError(f"JAX training composition does not yet lower {expression.kind.value} index expressions")


def _broadcast(array: jax.Array, source_axes: tuple[TensorAxis, ...], output_axes: tuple[TensorAxis, ...]) -> jax.Array:
    positions = [output_axes.index(axis) for axis in source_axes]
    permutation = tuple(index for index, _ in sorted(enumerate(positions), key=lambda item: item[1]))
    ordered = jnp.transpose(array, permutation) if permutation != tuple(range(array.ndim)) else array
    shape = [1] * len(output_axes)
    for dimension, position in enumerate(sorted(positions)):
        shape[position] = ordered.shape[dimension]
    return jnp.broadcast_to(ordered.reshape(shape), tuple(axis.extent for axis in output_axes))


def _evaluate_scalar(expression: ScalarExpression, inputs: Mapping[str, jax.Array]) -> jax.Array:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return inputs[expression.input_name]
    if expression.kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        return jnp.asarray(expression.constant)
    operands = tuple(_evaluate_scalar(operand, inputs) for operand in expression.operands)
    if expression.kind is ScalarExpressionKind.CONVERT:
        assert expression.result_dtype is not None
        return operands[0].astype(_jax_dtype(expression.result_dtype))
    function = {
        ScalarExpressionKind.ADD: jnp.add,
        ScalarExpressionKind.SUBTRACT: jnp.subtract,
        ScalarExpressionKind.MULTIPLY: jnp.multiply,
        ScalarExpressionKind.DIVIDE: jnp.divide,
        ScalarExpressionKind.NEGATE: jnp.negative,
        ScalarExpressionKind.EXP: jnp.exp,
        ScalarExpressionKind.LOG: jnp.log,
        ScalarExpressionKind.RSQRT: jax.lax.rsqrt,
        ScalarExpressionKind.TANH: jnp.tanh,
        ScalarExpressionKind.MAXIMUM: jnp.maximum,
        ScalarExpressionKind.MINIMUM: jnp.minimum,
        ScalarExpressionKind.LESS: jnp.less,
        ScalarExpressionKind.LESS_EQUAL: jnp.less_equal,
        ScalarExpressionKind.GREATER: jnp.greater,
        ScalarExpressionKind.GREATER_EQUAL: jnp.greater_equal,
        ScalarExpressionKind.EQUAL: jnp.equal,
        ScalarExpressionKind.NOT_EQUAL: jnp.not_equal,
        ScalarExpressionKind.SELECT: jnp.where,
    }.get(expression.kind)
    if function is None:
        raise ValueError(f"JAX training composition does not support scalar {expression.kind.value}")
    return function(*operands)


def _jax_dtype(dtype: DType) -> np.dtype:
    return np.dtype(
        {
            DType.BOOL: jnp.bool_,
            DType.BF16: jnp.bfloat16,
            DType.FP32: jnp.float32,
            DType.FP64: jnp.float64,
            DType.INT32: jnp.int32,
        }[dtype]
    )
