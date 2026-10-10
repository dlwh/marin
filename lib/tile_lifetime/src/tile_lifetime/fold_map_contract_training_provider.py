# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Physical providers for generic Fold/Map/Contract automatic adjoints."""

from __future__ import annotations

from dataclasses import dataclass

from tile_lifetime.cuda_axis_fold_codegen import (
    AxisFoldDirection,
    AxisFoldInput,
    AxisFoldInputLayout,
    AxisFoldOutputKind,
    AxisFoldPipeline,
    AxisFoldPipelineSchedule,
    AxisFoldPipelineStage,
    AxisFoldProgram,
    AxisFoldReassociation,
    AxisFoldReduction,
    GeneratedCudaAxisFoldFfi,
    generate_cuda_axis_fold_pipeline_ffi,
)
from tile_lifetime.row_normalization_training import (
    FoldMapContractTrainingPlan,
    GeneratedContractSkeleton,
    GeneratedMapSkeleton,
    GeneratedRelationSkeleton,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    RelationPrimitive,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorPrimitive,
    primitive_outputs,
    scalar_expression_inputs,
    scalar_input,
)


@dataclass(frozen=True)
class AxisFoldValueBinding:
    """Bind one generated AxisFold parameter to an algebra value."""

    parameter: str
    value: ProgramValue
    layout: AxisFoldInputLayout


@dataclass(frozen=True)
class AxisFoldStageBinding:
    """Bind one algebra Fold to its generated pipeline stage."""

    fold: FoldPrimitive
    stage: AxisFoldPipelineStage
    inputs: tuple[AxisFoldValueBinding, ...]


@dataclass(frozen=True)
class AxisFoldRegionBinding:
    """One launchable topological region of Fold stages and inlined scalar Maps."""

    pipeline: AxisFoldPipeline
    generated: GeneratedCudaAxisFoldFfi
    fold_bindings: tuple[AxisFoldStageBinding, ...]
    launch_operation_ordinal: int


@dataclass(frozen=True)
class FoldMapContractTrainingProvider:
    """Generated Fold regions plus structurally retained terminal components."""

    regions: tuple[AxisFoldRegionBinding, ...]
    terminal_components: tuple[GeneratedMapSkeleton | GeneratedRelationSkeleton | GeneratedContractSkeleton, ...]
    task_ids: tuple[int, ...]
    event_count: int
    saved_values: tuple[str, ...]
    recomputed_values: tuple[str, ...]


def provide_fold_map_contract_training(
    plan: FoldMapContractTrainingPlan,
    *,
    target_name: str,
    threads: int = 256,
    maximum_fold_outputs_per_region: int = 8,
) -> FoldMapContractTrainingProvider:
    """Generate AxisFold stages selected only from the automatic adjoint DAG."""
    if threads <= 0 or threads & (threads - 1):
        raise ValueError("AxisFold provider threads must be a positive power of two")
    producer_by_value = {
        output: operation for operation in plan.backward.operations for output in primitive_outputs(operation)
    }
    operation_ordinal = {id(operation): ordinal for ordinal, operation in enumerate(plan.backward.operations)}
    bindings = tuple(
        _axis_fold_stage(
            operation,
            producer_by_value=producer_by_value,
            operation_ordinal=operation_ordinal,
            threads=threads,
        )
        for operation in plan.backward.operations
        if isinstance(operation, FoldPrimitive)
    )
    if not bindings:
        raise ValueError("automatic adjoint has no Fold to provide")
    if maximum_fold_outputs_per_region < 1:
        raise ValueError("AxisFold provider Fold-output resource limit must be positive")
    binding_groups = _partition_launchable_regions(
        bindings,
        producer_by_value=producer_by_value,
        operation_ordinal=operation_ordinal,
        maximum_fold_outputs_per_region=maximum_fold_outputs_per_region,
    )
    regions = tuple(
        _generate_region(group, target_name=f"{target_name}.region_{index}", operation_ordinal=operation_ordinal)
        for index, group in enumerate(binding_groups)
    )
    terminal_components = tuple(
        step
        for step in plan.backward_steps
        if isinstance(step, (GeneratedMapSkeleton, GeneratedRelationSkeleton, GeneratedContractSkeleton))
    )
    return FoldMapContractTrainingProvider(
        regions=regions,
        terminal_components=terminal_components,
        task_ids=tuple(task.id for task in plan.backward_task_plan.tasks),
        event_count=len(plan.backward_event_program.event_plans),
        saved_values=plan.saved_values,
        recomputed_values=plan.recomputed_values,
    )


def _axis_fold_stage(
    fold: FoldPrimitive,
    *,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
    operation_ordinal: dict[int, int],
    threads: int,
) -> AxisFoldStageBinding:
    if fold.reducer is not FoldReducer.SUM:
        raise ValueError(f"AxisFold provider only supports sum reductions, found {fold.reducer.value}")
    if len(fold.input.axes) != 2 or len(fold.reduction_axes) != 1 or len(fold.output.axes) != 1:
        raise ValueError("AxisFold provider requires a contiguous rank-two input and one rank-one result")
    if fold.output.axes != tuple(axis for axis in fold.input.axes if axis not in fold.reduction_axes):
        raise ValueError("AxisFold provider requires the unreduced axis to retain logical order")
    expression, leaves = _inline_scalar_value(fold.input, fold.input.axes, producer_by_value, frozenset())
    reduction_axis = (
        AxisFoldDirection.ROWS if fold.reduction_axes[0] == fold.input.axes[0] else AxisFoldDirection.COLUMNS
    )
    ordinal = operation_ordinal[id(fold)]
    inputs = tuple(
        AxisFoldValueBinding(
            _pipeline_value_name(value, producer_by_value, operation_ordinal, ordinal, index),
            value,
            _input_layout(value, fold.input.axes),
        )
        for index, value in enumerate(leaves)
    )
    renamed = _rename_scalar_inputs(expression, {binding.value.name: binding.parameter for binding in inputs})
    reduction_name = "reduction_0"
    program = AxisFoldProgram(
        rows=fold.input.axes[0].extent,
        columns=fold.input.axes[1].extent,
        inputs=tuple(AxisFoldInput(binding.parameter, binding.value.dtype, binding.layout) for binding in inputs),
        reductions=(AxisFoldReduction(reduction_name, renamed),),
        reduction_axis=reduction_axis,
        output_kind=AxisFoldOutputKind.REDUCED,
        output_expression=scalar_input(reduction_name),
        output_dtype=fold.output.dtype,
        threads=1 if fold.order is EvaluationOrder.SOURCE_ORDERED else threads,
        reassociation=(
            AxisFoldReassociation.SOURCE_ORDERED
            if fold.order is EvaluationOrder.SOURCE_ORDERED
            else AxisFoldReassociation.DETERMINISTIC_TREE
        ),
    )
    stage = AxisFoldPipelineStage(output_name=f"fold_{ordinal}", program=program, expose_output=True)
    return AxisFoldStageBinding(fold, stage, inputs)


def _pipeline_value_name(
    value: ProgramValue,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
    operation_ordinal: dict[int, int],
    consumer_ordinal: int,
    input_index: int,
) -> str:
    producer = producer_by_value.get(value)
    if isinstance(producer, FoldPrimitive):
        producer_ordinal = operation_ordinal[id(producer)]
        if producer_ordinal >= consumer_ordinal:
            raise ValueError("AxisFold pipeline requires topologically ordered Fold dependencies")
        return f"fold_{producer_ordinal}"
    return f"stage_{consumer_ordinal}_input_{input_index}"


def _partition_launchable_regions(
    bindings: tuple[AxisFoldStageBinding, ...],
    *,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
    operation_ordinal: dict[int, int],
    maximum_fold_outputs_per_region: int,
) -> tuple[tuple[AxisFoldStageBinding, ...], ...]:
    groups: list[tuple[AxisFoldStageBinding, ...]] = []
    pending: list[AxisFoldStageBinding] = []
    first_ordinal = -1
    for binding in bindings:
        ordinal = operation_ordinal[id(binding.fold)]
        if not pending:
            pending = [binding]
            first_ordinal = ordinal
            continue
        external_producers = tuple(
            operation_ordinal[id(producer)]
            for item in binding.inputs
            if (producer := producer_by_value.get(item.value)) is not None and not isinstance(producer, FoldPrimitive)
        )
        if len(pending) >= maximum_fold_outputs_per_region or any(
            producer_ordinal >= first_ordinal for producer_ordinal in external_producers
        ):
            groups.append(tuple(pending))
            pending = [binding]
            first_ordinal = ordinal
        else:
            pending.append(binding)
    groups.append(tuple(pending))
    return tuple(groups)


def _generate_region(
    bindings: tuple[AxisFoldStageBinding, ...],
    *,
    target_name: str,
    operation_ordinal: dict[int, int],
) -> AxisFoldRegionBinding:
    pipeline = AxisFoldPipeline(tuple(binding.stage for binding in bindings))
    generated = generate_cuda_axis_fold_pipeline_ffi(
        pipeline,
        target_name=target_name,
        schedule=AxisFoldPipelineSchedule.SEPARATE_STAGES,
    )
    return AxisFoldRegionBinding(
        pipeline=pipeline,
        generated=generated,
        fold_bindings=bindings,
        launch_operation_ordinal=operation_ordinal[id(bindings[0].fold)],
    )


def _inline_scalar_value(
    value: ProgramValue,
    target_axes: tuple[TensorAxis, ...],
    producer_by_value: dict[ProgramValue, TensorPrimitive],
    visiting: frozenset[ProgramValue],
) -> tuple[ScalarExpression, tuple[ProgramValue, ...]]:
    if value in visiting:
        raise ValueError("scalar Map DAG contains a cycle")
    producer = producer_by_value.get(value)
    if isinstance(producer, MapPrimitive):
        replacements: dict[str, ScalarExpression] = {}
        leaves: list[ProgramValue] = []
        for operand in producer.inputs:
            expression, operand_leaves = _inline_scalar_value(
                operand, target_axes, producer_by_value, visiting | {value}
            )
            replacements[operand.name] = expression
            leaves.extend(operand_leaves)
        return _substitute_scalar_inputs(producer.expression, replacements), tuple(dict.fromkeys(leaves))
    if isinstance(producer, RelationPrimitive):
        if (
            len(producer.inputs) != 1
            or producer.outputs != (value,)
            or producer.resolved_result_sources != producer.inputs
            or producer.resolved_result_expressions != (None,)
            or producer.predicate is not None
            or producer.input_indexing[0].predicate is not None
        ):
            raise ValueError("AxisFold scalar inlining requires an unpredicated single-source Relation")
        return _inline_scalar_value(producer.inputs[0], target_axes, producer_by_value, visiting | {value})
    if isinstance(producer, ContractPrimitive):
        return scalar_input(value.name), (value,)
    if isinstance(producer, FoldPrimitive) or producer is None:
        return scalar_input(value.name), (value,)
    raise ValueError(f"AxisFold scalar inlining does not support {type(producer).__name__}")


def _input_layout(value: ProgramValue, target_axes: tuple[TensorAxis, ...]) -> AxisFoldInputLayout:
    if value.axes == target_axes:
        return AxisFoldInputLayout.ELEMENT
    if not value.axes:
        return AxisFoldInputLayout.SCALAR
    non_unit_axes = tuple(axis for axis in value.axes if axis.extent != 1)
    if non_unit_axes == (target_axes[0],):
        return AxisFoldInputLayout.ROW
    if non_unit_axes == (target_axes[1],):
        return AxisFoldInputLayout.COLUMN
    raise ValueError(f"AxisFold input {value.name!r} axes are noncontiguous or cannot broadcast over the Fold domain")


def _substitute_scalar_inputs(
    expression: ScalarExpression, replacements: dict[str, ScalarExpression]
) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return replacements.get(expression.input_name, expression)
    return ScalarExpression(
        kind=expression.kind,
        operands=tuple(_substitute_scalar_inputs(operand, replacements) for operand in expression.operands),
        input_name=expression.input_name,
        constant=expression.constant,
        result_dtype=expression.result_dtype,
        conversion_semantics=expression.conversion_semantics,
    )


def _rename_scalar_inputs(expression: ScalarExpression, names: dict[str, str]) -> ScalarExpression:
    unknown = scalar_expression_inputs(expression) - set(names)
    if unknown:
        raise ValueError(f"AxisFold scalar DAG has unavailable leaves {sorted(unknown)}")
    return _substitute_scalar_inputs(expression, {source: scalar_input(target) for source, target in names.items()})
