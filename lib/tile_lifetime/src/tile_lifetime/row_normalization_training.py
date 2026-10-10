# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training lowering for Fold-dependent Map/Contract tensor algebra.

There is deliberately no normalization classifier here. The lowering finds a
dataflow region, differentiates the original TensorProgram, then applies a
generic save/recompute cut to that adjoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from shuttle.ir import DType
from tile_lifetime.autodiff import (
    DifferentiatedTensorProgram,
    differentiate_tensor_program,
    extract_backward_tensor_program,
)
from tile_lifetime.dense_flow_planner import derive_composed_task_plan, eventize_composed_task_plan
from tile_lifetime.event_dataflow import EventDataflowProgram
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND
from tile_lifetime.physical_task_contract import CanonicalTaskPlan, PhysicalNumericalPolicy
from tile_lifetime.plan import GemmSkeleton, NumericalEquivalence
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    RelationPrimitive,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    primitive_inputs,
    primitive_outputs,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
)


class ReverseMaterializationPolicy(StrEnum):
    """Generic forward values retained across a forward/reverse boundary."""

    SAVE_ALL_INTERMEDIATES = "save_all_intermediates"
    SAVE_FOLD_STATES = "save_fold_states"
    RECOMPUTE = "recompute"


@dataclass(frozen=True)
class GeneratedMapSkeleton:
    name: str
    inputs: tuple[str, ...]
    output: str
    output_shape: tuple[int, ...]
    expression: ScalarExpression
    tile_shape: tuple[int, ...]


@dataclass(frozen=True)
class GeneratedFoldSkeleton:
    name: str
    input: str
    output: str
    input_shape: tuple[int, ...]
    reduction_axes: tuple[str, ...]
    reducer: FoldReducer
    accumulation_dtype: DType
    tile_extent: int


@dataclass(frozen=True)
class GeneratedContractSkeleton:
    name: str
    operands: tuple[str, str]
    operand_views: tuple[str, str]
    output: str
    reduction_axis: str
    skeleton: GemmSkeleton


@dataclass(frozen=True)
class GeneratedRelationSkeleton:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    domain_shape: tuple[int, ...]


GeneratedTrainingSkeleton = (
    GeneratedMapSkeleton | GeneratedFoldSkeleton | GeneratedContractSkeleton | GeneratedRelationSkeleton
)


@dataclass(frozen=True)
class FoldMapContractRegion:
    """A dataflow-derived Fold/Map/Contract region and explicit dimensions."""

    contract: ContractPrimitive
    mapped_operand: ProgramValue
    other_operand: ProgramValue
    fold_states: tuple[ProgramValue, ...]
    scalar_maps: tuple[MapPrimitive, ...]
    operations: tuple[TensorPrimitive, ...]
    owner_axes: tuple[TensorAxis, ...]
    contraction_axes: tuple[TensorAxis, ...]


@dataclass(frozen=True)
class FoldMapContractTrainingPlan:
    """Generic region, automatic adjoint, and lowered physical skeletons."""

    source: TensorProgram
    region: FoldMapContractRegion
    automatic_adjoint: DifferentiatedTensorProgram
    backward: TensorProgram
    input_gradients: tuple[ProgramValue, ...]
    materialization_policy: ReverseMaterializationPolicy
    numerical_equivalence: NumericalEquivalence
    saved_values: tuple[str, ...]
    recomputed_values: tuple[str, ...]
    forward_contract: GemmSkeleton
    backward_steps: tuple[GeneratedTrainingSkeleton, ...]
    backward_task_plan: CanonicalTaskPlan
    backward_event_program: EventDataflowProgram

    @property
    def maps(self) -> tuple[GeneratedMapSkeleton, ...]:
        return tuple(step for step in self.backward_steps if isinstance(step, GeneratedMapSkeleton))

    @property
    def folds(self) -> tuple[GeneratedFoldSkeleton, ...]:
        return tuple(step for step in self.backward_steps if isinstance(step, GeneratedFoldSkeleton))

    @property
    def contracts(self) -> tuple[GeneratedContractSkeleton, ...]:
        return tuple(step for step in self.backward_steps if isinstance(step, GeneratedContractSkeleton))


def build_one_fold_contract_fixture(
    *, rows: int, hidden: int, features: int, epsilon: float = 1e-5, dtype: DType = DType.FP32
) -> TensorProgram:
    """Build a small one-Fold fixture; compilation does not inspect its identity."""
    return _build_fold_contract_fixture(rows, hidden, features, epsilon, dtype, leading_fold=False)


def build_two_fold_contract_fixture(
    *, rows: int, hidden: int, features: int, epsilon: float = 1e-5, dtype: DType = DType.FP32
) -> TensorProgram:
    """Build a small two-Fold fixture; compilation does not inspect its identity."""
    return _build_fold_contract_fixture(rows, hidden, features, epsilon, dtype, leading_fold=True)


def _build_fold_contract_fixture(
    rows: int, hidden: int, features: int, epsilon: float, dtype: DType, *, leading_fold: bool
) -> TensorProgram:
    if min(rows, hidden, features) <= 0 or epsilon < 0:
        raise ValueError("fixture dimensions must be positive and epsilon non-negative")
    row = TensorAxis(0, rows, "row")
    reduction = TensorAxis(1, hidden, "reduction")
    feature = TensorAxis(2, features, "feature")
    x = ProgramValue("input", (row, reduction), dtype)
    scale = ProgramValue("feature_scale", (reduction,), dtype)
    weight = ProgramValue("weight", (reduction, feature), dtype)
    operations: list[TensorPrimitive] = []
    local = x
    if leading_fold:
        total = ProgramValue("row_sum", (row,), DType.FP32)
        mean = ProgramValue("row_mean", (row,), DType.FP32)
        local = ProgramValue("centered", (row, reduction), dtype)
        operations.extend(
            (
                FoldPrimitive("Fold input", x, total, (reduction,), FoldReducer.SUM, DType.FP32),
                MapPrimitive(
                    "scale Fold", (total,), mean, _divide(scalar_input(total.name), scalar_constant(float(hidden)))
                ),
                MapPrimitive(
                    "subtract Fold", (x, mean), local, _subtract(scalar_input(x.name), scalar_input(mean.name))
                ),
            )
        )
    square = ProgramValue("squared", (row, reduction), DType.FP32)
    total_square = ProgramValue("sum_square", (row,), DType.FP32)
    mean_square = ProgramValue("mean_square", (row,), DType.FP32)
    inverse = ProgramValue("inverse_scale", (row,), DType.FP32)
    transformed = ProgramValue("standardized", (row, reduction), dtype)
    contracted_input = ProgramValue("scaled", (row, reduction), dtype)
    output = ProgramValue("output", (row, feature), dtype)
    operations.extend(
        (
            MapPrimitive("square", (local,), square, _multiply(scalar_input(local.name), scalar_input(local.name))),
            FoldPrimitive("Fold square", square, total_square, (reduction,), FoldReducer.SUM, DType.FP32),
            MapPrimitive(
                "average Fold",
                (total_square,),
                mean_square,
                _divide(scalar_input(total_square.name), scalar_constant(float(hidden))),
            ),
            MapPrimitive(
                "inverse Fold",
                (mean_square,),
                inverse,
                scalar_unary(ScalarExpressionKind.RSQRT, _add(scalar_input(mean_square.name), scalar_constant(epsilon))),
            ),
            MapPrimitive(
                "transform",
                (local, inverse),
                transformed,
                _multiply(scalar_input(local.name), scalar_input(inverse.name)),
            ),
            MapPrimitive(
                "scale",
                (transformed, scale),
                contracted_input,
                _multiply(scalar_input(transformed.name), scalar_input(scale.name)),
            ),
            ContractPrimitive("Contract", (contracted_input, weight), output, (reduction,), DType.FP32),
        )
    )
    return TensorProgram(inputs=(x, scale, weight), operations=tuple(operations), outputs=(output,))


def compile_fold_map_contract_training(
    source: TensorProgram,
    *,
    materialization_policy: ReverseMaterializationPolicy = ReverseMaterializationPolicy.SAVE_FOLD_STATES,
    physical_tile_shape: tuple[int, int, int] = (128, 128, 64),
    map_tile_elements: int = 256,
    fold_tile_extent: int = 256,
    task_tile_extent: int = 32,
) -> FoldMapContractTrainingPlan:
    """Differentiate and lower a generic Fold-dependent output Contract region."""
    region = derive_fold_map_contract_region(source)
    differentiated = differentiate_tensor_program(source, with_respect_to=tuple(value.name for value in source.inputs))
    saved_names = _saved_value_names(source, region, differentiated, materialization_policy)
    extracted = extract_backward_tensor_program(differentiated, saved_values=saved_names)
    steps = tuple(
        _lower_training_operation(
            operation,
            physical_tile_shape=physical_tile_shape,
            map_tile_elements=map_tile_elements,
            fold_tile_extent=fold_tile_extent,
        )
        for operation in extracted.program.operations
    )
    axes = {
        axis
        for operation in extracted.program.operations
        for value in (*primitive_inputs(operation), *primitive_outputs(operation))
        for axis in value.axes
    }
    task_plan = derive_composed_task_plan(
        extracted.program,
        tile_extent_by_axis={axis.id: min(axis.extent, task_tile_extent) for axis in axes},
        numerical_policy=PhysicalNumericalPolicy(source.numerical_policy.value),
    )
    return FoldMapContractTrainingPlan(
        source=source,
        region=region,
        automatic_adjoint=differentiated,
        backward=extracted.program,
        input_gradients=differentiated.input_gradients,
        materialization_policy=materialization_policy,
        numerical_equivalence=NumericalEquivalence.BITWISE_EXACT,
        saved_values=tuple(value.name for value in extracted.saved_values),
        recomputed_values=tuple(
            output.name for operation in extracted.recomputed_operations for output in primitive_outputs(operation)
        ),
        forward_contract=_forward_contract_skeleton(region, physical_tile_shape),
        backward_steps=steps,
        backward_task_plan=task_plan,
        backward_event_program=eventize_composed_task_plan(task_plan),
    )


def derive_fold_map_contract_region(source: TensorProgram) -> FoldMapContractRegion:
    """Derive the output Contract region solely from values and dependencies."""
    producer_by_value = {output: operation for operation in source.operations for output in primitive_outputs(operation)}
    output_operations = {
        operation for output in source.outputs for operation in _value_ancestors(output, producer_by_value)
    }
    contracts = tuple(
        operation
        for operation in source.operations
        if operation in output_operations and isinstance(operation, ContractPrimitive)
    )
    terminal_contracts = tuple(
        contract
        for contract in contracts
        if not any(
            contract is not other and contract in _operation_ancestors(other, producer_by_value) for other in contracts
        )
    )
    if len(terminal_contracts) != 1:
        raise ValueError(
            f"generic Fold/Map/Contract training requires one terminal output Contract; found {len(terminal_contracts)}"
        )
    contract = terminal_contracts[0]
    if len(contract.inputs) != 2 or len(contract.reduction_axes) != 1 or len(contract.output.axes) != 2:
        raise ValueError("generic GEMM training requires a two-input rank-two Contract with one contraction axis")

    operand_ancestors = tuple(_value_ancestors(value, producer_by_value) for value in contract.inputs)
    fold_operand_indices = tuple(
        index
        for index, ancestors in enumerate(operand_ancestors)
        if any(isinstance(operation, FoldPrimitive) for operation in ancestors)
    )
    if len(fold_operand_indices) != 1:
        raise ValueError("exactly one Contract operand must depend on one or more Fold states")
    mapped_index = fold_operand_indices[0]
    region_operations = tuple(
        operation for operation in source.operations if operation in operand_ancestors[mapped_index]
    )
    folds = tuple(operation for operation in region_operations if isinstance(operation, FoldPrimitive))
    fold_states = tuple(fold.output for fold in folds)
    return FoldMapContractRegion(
        contract=contract,
        mapped_operand=contract.inputs[mapped_index],
        other_operand=contract.inputs[1 - mapped_index],
        fold_states=fold_states,
        scalar_maps=tuple(operation for operation in region_operations if isinstance(operation, MapPrimitive)),
        operations=region_operations,
        owner_axes=tuple(dict.fromkeys(axis for state in fold_states for axis in state.axes)),
        contraction_axes=contract.reduction_axes,
    )


def _saved_value_names(
    source: TensorProgram,
    region: FoldMapContractRegion,
    differentiated: DifferentiatedTensorProgram,
    policy: ReverseMaterializationPolicy,
) -> tuple[str, ...]:
    if policy is ReverseMaterializationPolicy.RECOMPUTE:
        return ()
    if policy is ReverseMaterializationPolicy.SAVE_FOLD_STATES:
        return tuple(value.name for value in region.fold_states)

    forward_outputs = {output.name for operation in source.operations for output in primitive_outputs(operation)}
    backward_operations = differentiated.program.operations[len(source.operations) :]
    referenced = {
        value.name
        for operation in backward_operations
        for value in primitive_inputs(operation)
        if value.name in forward_outputs
    }
    return tuple(
        output.name
        for operation in source.operations
        for output in primitive_outputs(operation)
        if output.name in referenced
    )


def _forward_contract_skeleton(region: FoldMapContractRegion, tile_shape: tuple[int, int, int]) -> GemmSkeleton:
    contract = region.contract
    m_axis, n_axis, k_axis, _, _, left_view, right_view = _contract_matrix_views(contract)
    return GemmSkeleton(
        name=contract.name,
        input=left_view,
        weight=right_view,
        output=contract.output.name,
        shape=(m_axis.extent, n_axis.extent, k_axis.extent),
        accumulation_dtype=contract.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=tile_shape,
        cluster_shape=(1, 1, 1),
        pingpong=True,
    )


def _lower_training_operation(
    operation: TensorPrimitive,
    *,
    physical_tile_shape: tuple[int, int, int],
    map_tile_elements: int,
    fold_tile_extent: int,
) -> GeneratedTrainingSkeleton:
    if isinstance(operation, MapPrimitive):
        return GeneratedMapSkeleton(
            operation.name,
            tuple(value.name for value in operation.inputs),
            operation.output.name,
            operation.output.shape,
            operation.expression,
            tuple(min(extent, map_tile_elements) for extent in operation.output.shape),
        )
    if isinstance(operation, FoldPrimitive):
        return GeneratedFoldSkeleton(
            operation.name,
            operation.input.name,
            operation.output.name,
            operation.input.shape,
            tuple(axis.label or f"axis_{axis.id}" for axis in operation.reduction_axes),
            operation.reducer,
            operation.accumulation_dtype,
            min(fold_tile_extent, max(axis.extent for axis in operation.reduction_axes)),
        )
    if isinstance(operation, RelationPrimitive):
        return GeneratedRelationSkeleton(
            operation.name,
            tuple(value.name for value in operation.inputs),
            tuple(value.name for value in operation.outputs),
            tuple(axis.extent for axis in operation.domain_axes),
        )
    return _lower_contract(operation, physical_tile_shape)


def _lower_contract(operation: ContractPrimitive, tile_shape: tuple[int, int, int]) -> GeneratedContractSkeleton:
    if len(operation.inputs) != 2 or len(operation.output.axes) != 2 or len(operation.reduction_axes) != 1:
        raise ValueError("generic GEMM legalization requires a two-input rank-two Contract with one reduction axis")
    m_axis, n_axis, k_axis, left, right, left_view, right_view = _contract_matrix_views(operation)
    skeleton = GemmSkeleton(
        name=operation.name,
        input=left_view,
        weight=right_view,
        output=operation.output.name,
        shape=(m_axis.extent, n_axis.extent, k_axis.extent),
        accumulation_dtype=operation.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=tile_shape,
        cluster_shape=(1, 1, 1),
        pingpong=True,
    )
    return GeneratedContractSkeleton(
        operation.name,
        (left.name, right.name),
        (left_view, right_view),
        operation.output.name,
        k_axis.label or f"axis_{k_axis.id}",
        skeleton,
    )


def _contract_matrix_views(
    operation: ContractPrimitive,
) -> tuple[TensorAxis, TensorAxis, TensorAxis, ProgramValue, ProgramValue, str, str]:
    k_axis = operation.reduction_axes[0]
    for left, right in (operation.inputs, tuple(reversed(operation.inputs))):
        left_free = tuple(axis for axis in left.axes if axis != k_axis)
        right_free = tuple(axis for axis in right.axes if axis != k_axis)
        if len(left_free) != 1 or len(right_free) != 1:
            continue
        m_axis, n_axis = left_free[0], right_free[0]
        if sorted((m_axis.extent, n_axis.extent)) != sorted(operation.output.shape):
            continue
        return (
            m_axis,
            n_axis,
            k_axis,
            left,
            right,
            _matrix_view(left, (m_axis, k_axis)),
            _matrix_view(right, (k_axis, n_axis)),
        )
    raise ValueError(f"Contract {operation.name!r} operands cannot be represented by one rank-two GEMM")


def _matrix_view(value: ProgramValue, axes: tuple[TensorAxis, TensorAxis]) -> str:
    if value.axes == axes:
        return value.name
    if value.axes == tuple(reversed(axes)):
        return f"transpose.{value.name}"
    raise ValueError(f"value {value.name!r} cannot legalize to matrix axes {[axis.label for axis in axes]}")


def _value_ancestors(
    value: ProgramValue,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
) -> frozenset[TensorPrimitive]:
    producer = producer_by_value.get(value)
    if producer is None:
        return frozenset()
    return frozenset((producer, *_operation_ancestors(producer, producer_by_value)))


def _operation_ancestors(
    operation: TensorPrimitive,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
) -> frozenset[TensorPrimitive]:
    result: set[TensorPrimitive] = set()
    pending = list(primitive_inputs(operation))
    while pending:
        value = pending.pop()
        producer = producer_by_value.get(value)
        if producer is None or producer in result:
            continue
        result.add(producer)
        pending.extend(primitive_inputs(producer))
    return frozenset(result)


def _add(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.ADD, left, right)


def _subtract(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.SUBTRACT, left, right)


def _multiply(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.MULTIPLY, left, right)


def _divide(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.DIVIDE, left, right)
