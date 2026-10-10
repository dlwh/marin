# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach generic two-stage Fold/Map/Contract requests to CUDA SM90 code."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from shuttle.ir import DType
from tile_lifetime.backends.h100.cute_map_load_contract import (
    GeneratedSm90MapLoadContract,
    Sm90MapLoadContractSchedule,
    generate_sm90_map_load_contract,
)
from tile_lifetime.cuda_axis_fold_codegen import render_cuda_scalar_expression
from tile_lifetime.event_buffering import (
    EventRealizationKind,
    erased_event_realization,
    verify_event_realizations,
)
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventStorageBinding,
    TaskFamily,
    derive_event_tensor_plan,
)
from tile_lifetime.physical_task_contract import (
    CanonicalTaskPlan,
    EventErasureWitness,
    EventRealizationProof,
    EventTensorAllocation,
    ExecutablePhysicalRequest,
    PhysicalOrderKind,
    PhysicalTarget,
    PhysicalTaskOrder,
    TaskCapability,
    TaskOperationKind,
    TiledTask,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    IndexExpressionKind,
    MapPrimitive,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    primitive_outputs,
    scalar_convert,
    scalar_expression_inputs,
    scalar_input,
    tensor_program_value_ordinals,
)


@dataclass(frozen=True)
class Sm90FoldMapContractBinding:
    """A structural two-kernel binding recovered from generic algebra and tasks."""

    rows: int
    reduction: int
    features: int
    raw_input: ProgramValue
    mapped_input: ProgramValue
    weight: ProgramValue
    output: ProgramValue
    folds: tuple[FoldPrimitive, ...]
    output_contract: ContractPrimitive
    producer_operations: tuple[TensorPrimitive, ...]
    consumer_operations: tuple[TensorPrimitive, ...]
    row_materializations: tuple[ProgramValue, ...]
    task_ids: tuple[int, int]

    @property
    def boundary_values(self) -> tuple[ProgramValue, ...]:
        """Values returned by the producer side of the two-kernel cut.

        The source tensor is an alias of an existing public allocation while
        Fold states and row-only scalar Maps are materialized. Recording all
        of them here prevents the consumer from acquiring undeclared inputs.
        """
        return (
            self.raw_input,
            *(fold.output for fold in self.folds),
            *self.row_materializations,
        )

    @property
    def fold(self) -> FoldPrimitive:
        """Final Fold in producer order, retained as the reduction-axis anchor."""
        return self.folds[-1]

    @property
    def fold_state(self) -> ProgramValue:
        """Final Fold state, retained for one-state callers."""
        return self.fold.output


@dataclass(frozen=True)
class GeneratedSm90FoldMapContract:
    """Generated two-kernel source bound to one common physical request."""

    source: str
    request_fingerprint: str
    algebra_fingerprint: str
    source_sha256: str
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[DType, ...]
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: DType
    physical_request: ExecutablePhysicalRequest
    intermediate_names: tuple[str, ...]
    intermediate_storage: tuple[str, ...]
    consumer: GeneratedSm90MapLoadContract
    fold_state_names: tuple[str, ...]
    row_materialization_names: tuple[str, ...]
    kernel_count: int = 2
    performance_class: str = "tiled_map_contract"


def attach_sm90_fold_map_contract(
    program: TensorProgram,
    request: ExecutablePhysicalRequest,
    *,
    threads: int = 256,
    consumer_schedule: Sm90MapLoadContractSchedule | None = None,
) -> GeneratedSm90FoldMapContract:
    """Generate a two-kernel SM90 implementation of one common request.

    Kernel zero evaluates a generic scalar Map into a row Fold. The raw input
    remains live through that task boundary alongside the compact Fold state.
    Kernel one compiles the Fold-dependent scalar Map DAG into the input side
    of a generic Contract.
    """
    if threads not in {128, 256, 512}:
        raise ValueError("SM90 Fold/Map/Contract codegen requires 128, 256, or 512 threads")
    if request.target is not PhysicalTarget.SM90:
        raise ValueError("the two-kernel CUDA attachment requires an SM90 physical request")
    if request.algebra_fingerprint != program.semantic_fingerprint:
        raise ValueError("physical request algebra fingerprint does not match the supplied tensor program")
    binding = recover_sm90_fold_map_contract_binding(program, request)
    if binding.raw_input.dtype is not DType.BF16 or binding.weight.dtype is not DType.BF16:
        raise ValueError("the SM90 WGMMA Map-load/Contract attachment requires BF16 activation and weight operands")
    producer_by_value = {
        output: operation for operation in program.operations for output in primitive_outputs(operation)
    }
    consumer_allowed = set(binding.consumer_operations) | {
        operation
        for operation in binding.producer_operations
        if isinstance(operation, (MapPrimitive, RelationPrimitive))
    }
    consumer_allowed -= {producer_by_value[value] for value in binding.row_materializations}
    consumer_leaves = tuple(
        sorted(
            _scalar_leaves(binding.mapped_input, producer_by_value, consumer_allowed),
            key=lambda value: (value.name, value.shape),
        )
    )
    consumer = generate_sm90_map_load_contract(
        expression=_fused_scalar_expression(binding.mapped_input, producer_by_value, consumer_allowed),
        activation=binding.raw_input,
        leaves=consumer_leaves,
        row_axis=binding.fold_state.axes[0],
        reduction_axis=binding.fold.reduction_axes[0],
        weight_feature_axis=binding.weight.axes[1],
        output_row_axis=binding.output.axes[0],
        output_feature_axis=binding.output.axes[1],
        weight=binding.weight,
        output=binding.output,
        schedule=consumer_schedule,
    )
    physical_request = _serial_launch_request(program, request, binding, consumer_schedule=consumer.schedule)
    source = _source(program, binding, physical_request, threads=threads)
    return GeneratedSm90FoldMapContract(
        source=source,
        request_fingerprint=physical_request.fingerprint,
        algebra_fingerprint=request.algebra_fingerprint,
        source_sha256=hashlib.sha256((source + consumer.source).encode()).hexdigest(),
        input_names=tuple(value.name for value in program.inputs),
        input_shapes=tuple(tuple(axis.extent for axis in value.axes) for value in program.inputs),
        input_dtypes=tuple(value.dtype for value in program.inputs),
        output_name=binding.output.name,
        output_shape=(binding.rows, binding.features),
        output_dtype=binding.output.dtype,
        physical_request=physical_request,
        intermediate_names=tuple(value.name for value in binding.boundary_values),
        intermediate_storage=(
            "input_alias",
            *(("materialized_fp32",) * len(binding.folds)),
            *(("materialized_fp32_row_map",) * len(binding.row_materializations)),
        ),
        consumer=consumer,
        fold_state_names=tuple(fold.output.name for fold in binding.folds),
        row_materialization_names=tuple(value.name for value in binding.row_materializations),
        performance_class=consumer.schedule.family,
    )


def recover_sm90_fold_map_contract_binding(
    program: TensorProgram,
    request: ExecutablePhysicalRequest,
) -> Sm90FoldMapContractBinding:
    """Select one dependency-convex physical region from generic task metadata."""
    candidates = discover_sm90_fold_map_contract_bindings(program, request)
    structural_pairs = _candidate_task_pairs(request)
    if not candidates and len(structural_pairs) == 1:
        _binding_for_task_pair(program, *structural_pairs[0])
    if len(candidates) != 1:
        raise ValueError(f"expected one dependency-convex Fold/Map/Contract region, found {len(candidates)}")
    return candidates[0]


def discover_sm90_fold_map_contract_bindings(
    program: TensorProgram,
    request: ExecutablePhysicalRequest,
) -> tuple[Sm90FoldMapContractBinding, ...]:
    """Find physicalizable regions without matching an enclosing graph topology."""
    candidates = []
    for producer_task, consumer_task in _candidate_task_pairs(request):
        try:
            candidates.append(_binding_for_task_pair(program, producer_task, consumer_task))
        except ValueError:
            continue
    return tuple(candidates)


def _candidate_task_pairs(request: ExecutablePhysicalRequest) -> tuple[tuple[TiledTask, TiledTask], ...]:
    successors = _task_successors(request)
    return tuple(
        (producer_task, consumer_task)
        for producer_task in request.task_plan.tasks
        if _task_has_capability(producer_task, TaskCapability.ASSOCIATIVE_FOLD)
        for consumer_task in request.task_plan.tasks
        if producer_task.id != consumer_task.id
        and _task_has_capability(consumer_task, TaskCapability.MATRIX_CONTRACT)
        and consumer_task.id in successors[producer_task.id]
        and _task_subset_is_dependency_convex({producer_task.id, consumer_task.id}, successors)
    )


def _binding_for_task_pair(
    program: TensorProgram,
    producer_task: TiledTask,
    consumer_task: TiledTask,
) -> Sm90FoldMapContractBinding:
    if producer_task.structure.principal_kind is not TaskOperationKind.FOLD:
        raise ValueError("the producer task must have a Fold principal operation")
    if consumer_task.structure.principal_kind is not TaskOperationKind.CONTRACT:
        raise ValueError("the consumer task must have a Contract principal operation")
    producer_operations = tuple(program.operations[index] for index in producer_task.operation_ordinals)
    consumer_operations = tuple(program.operations[index] for index in consumer_task.operation_ordinals)
    if any(
        not isinstance(operation, (MapPrimitive, FoldPrimitive, RelationPrimitive)) for operation in producer_operations
    ):
        raise ValueError("the producer task contains a phase outside the current Map/Fold/Relation physical capability")
    if any(
        not isinstance(operation, (MapPrimitive, ContractPrimitive, RelationPrimitive))
        for operation in consumer_operations
    ):
        raise ValueError(
            "the consumer task contains a phase outside the current Map/Contract/Relation physical capability"
        )
    folds = tuple(operation for operation in producer_operations if isinstance(operation, FoldPrimitive))
    contracts = tuple(operation for operation in consumer_operations if isinstance(operation, ContractPrimitive))
    if not 1 <= len(folds) <= 2:
        raise ValueError(
            "the current SM90 producer supports one or two ordered Folds; additional Fold state "
            "exceeds its producer pass and materialization resource model"
        )
    if len(contracts) != 1:
        raise ValueError("the current SM90 consumer requires exactly one Contract")
    fold = folds[-1]
    contract = contracts[0]
    if any(item.reducer is not FoldReducer.SUM or item.accumulation_dtype is not DType.FP32 for item in folds):
        raise ValueError("the current SM90 producer requires ordered FP32 sum Folds")
    if contract.accumulation_dtype is not DType.FP32 or len(contract.reduction_axes) != 1:
        raise ValueError("the current SM90 consumer requires one FP32 Contract reduction axis")
    if any(len(item.input.axes) != 2 or len(item.output.axes) != 1 for item in folds):
        raise ValueError("the current SM90 Folds require rank-two inputs and rank-one states")
    row_axis = fold.output.axes[0]
    reduction_axis = fold.reduction_axes[0]
    if any(
        item.output.axes != (row_axis,)
        or item.reduction_axes != (reduction_axis,)
        or item.input.axes != (row_axis, reduction_axis)
        for item in folds
    ):
        raise ValueError("the current SM90 Folds require common contiguous logical (row, reduction) axes")

    producer_by_value = {
        output: operation for operation in program.operations for output in primitive_outputs(operation)
    }
    contract_left, contract_right = contract.inputs
    if contract_left.axes == (row_axis, reduction_axis):
        mapped_input, weight = contract_left, contract_right
        mapped_input_index, weight_index = 0, 1
    elif contract_right.axes == (row_axis, reduction_axis):
        mapped_input, weight = contract_right, contract_left
        mapped_input_index, weight_index = 1, 0
    else:
        raise ValueError("the current SM90 Contract requires one (row, reduction) mapped operand")
    output_row_axis = _contract_domain_axis(contract, mapped_input_index, row_axis)
    weight = _neutral_public_operand(weight, producer_by_value, set(consumer_operations), set(program.inputs))
    if len(weight.axes) != 2 or weight.axes[0] != reduction_axis:
        raise ValueError("the current SM90 Contract weight must have logical (reduction, feature) axes")
    output_feature_axis = _contract_domain_axis(contract, weight_index, weight.axes[1])
    if len(contract.output.axes) != 2 or contract.output.axes != (output_row_axis, output_feature_axis):
        raise ValueError("the current SM90 Contract output must have logical (row, feature) axes")
    feature_axis = output_feature_axis
    if weight not in program.inputs:
        raise ValueError("the current SM90 Contract requires a public weight operand")

    producer_allowed = set(producer_operations)
    consumer_allowed = set(consumer_operations) | {
        operation for operation in producer_operations if isinstance(operation, (MapPrimitive, RelationPrimitive))
    }
    producer_leaves = frozenset(
        leaf for item in folds for leaf in _scalar_leaves(item.input, producer_by_value, producer_allowed)
    )
    consumer_leaves = _scalar_leaves(mapped_input, producer_by_value, consumer_allowed)
    raw_candidates = tuple(
        value
        for value in program.inputs
        if value.axes == (row_axis, reduction_axis) and value in producer_leaves and value in consumer_leaves
    )
    if len(raw_candidates) != 1:
        raise ValueError("the two-kernel cut requires one raw (row, reduction) value live through the Fold producer")
    raw_input = raw_candidates[0]
    missing_fold_states = tuple(item.output.name for item in folds if item.output not in consumer_leaves)
    if missing_fold_states:
        raise ValueError(f"the Contract-side scalar Map does not depend on producer Fold states {missing_fold_states}")
    if mapped_input.dtype not in {DType.BF16, DType.FP32} or contract.output.dtype not in {DType.BF16, DType.FP32}:
        raise ValueError("the current SM90 consumer supports BF16 or FP32 mapped and output values")
    if any(value.dtype not in {DType.BF16, DType.FP32} for value in program.inputs):
        raise ValueError("the current SM90 public boundary supports BF16 and FP32 tensors")
    row_materializations = _row_map_materializations(
        program,
        mapped_input=mapped_input,
        row_axis=row_axis,
    )
    return Sm90FoldMapContractBinding(
        rows=row_axis.extent,
        reduction=reduction_axis.extent,
        features=feature_axis.extent,
        raw_input=raw_input,
        mapped_input=mapped_input,
        weight=weight,
        output=contract.output,
        folds=folds,
        output_contract=contract,
        producer_operations=producer_operations,
        consumer_operations=consumer_operations,
        row_materializations=row_materializations,
        task_ids=(producer_task.id, consumer_task.id),
    )


def _row_map_materializations(
    program: TensorProgram,
    *,
    mapped_input: ProgramValue,
    row_axis: TensorAxis,
) -> tuple[ProgramValue, ...]:
    """Select row-dependent Map values immediately consumed by a wider value."""
    producer_by_value = {
        output: operation for operation in program.operations for output in primitive_outputs(operation)
    }
    ancestors: set[ProgramValue] = set()

    def visit(value: ProgramValue) -> None:
        if value in ancestors:
            return
        ancestors.add(value)
        producer = producer_by_value.get(value)
        if isinstance(producer, (MapPrimitive, RelationPrimitive)):
            for operand in producer.inputs:
                visit(operand)

    visit(mapped_input)
    consumers: dict[ProgramValue, list[ProgramValue]] = {}
    for value in ancestors:
        producer = producer_by_value.get(value)
        if not isinstance(producer, (MapPrimitive, RelationPrimitive)):
            continue
        for operand in producer.inputs:
            consumers.setdefault(operand, []).append(value)
    selected = []
    for operation in program.operations:
        if not isinstance(operation, MapPrimitive) or operation.output not in ancestors:
            continue
        value = operation.output
        if not _is_row_only(value, row_axis) or value.dtype is not DType.FP32:
            continue
        if any(not _is_row_only(consumer, row_axis) for consumer in consumers.get(value, ())):
            selected.append(value)
    return tuple(selected)


def _is_row_only(value: ProgramValue, row_axis: TensorAxis) -> bool:
    """Whether a value varies only with the row index.

    StableHLO commonly preserves reduced dimensions as fresh extent-one axes
    before broadcasting them back over the reduction axis. Those axes carry no
    index dependence and therefore do not change the materialization domain.
    """
    return row_axis in value.axes and all(axis == row_axis or axis.extent == 1 for axis in value.axes)


def _neutral_public_operand(
    value: ProgramValue,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
    allowed: set[TensorPrimitive],
    public_inputs: set[ProgramValue],
) -> ProgramValue:
    producer = producer_by_value.get(value)
    if producer not in allowed or not isinstance(producer, MapPrimitive):
        return value
    source = _neutral_widening_map_source(producer)
    return source if source is not None and source in public_inputs else value


def _neutral_widening_map_source(operation: MapPrimitive) -> ProgramValue | None:
    if len(operation.inputs) != 1:
        return None
    expression = operation.expression
    if (
        expression.kind is not ScalarExpressionKind.CONVERT
        or expression.result_dtype is not DType.FP32
        or expression.conversion_semantics is not ScalarConversionSemantics.ROUND_NEAREST_EVEN
        or operation.inputs[0].dtype is not DType.BF16
        or operation.output.dtype is not DType.FP32
    ):
        return None
    return operation.inputs[0]


def _contract_domain_axis(
    contract: ContractPrimitive,
    input_index: int,
    operand_axis: TensorAxis,
) -> TensorAxis:
    matches = tuple(
        mapping.domain_axis
        for mapping in contract.index_maps_for_input(input_index)
        if mapping.operand_axis == operand_axis
    )
    if len(matches) == 1:
        return matches[0]
    if not matches and operand_axis in contract.output.axes:
        return operand_axis
    raise ValueError("the current SM90 Contract requires one affine output-to-operand axis relation")


def _source(
    program: TensorProgram,
    binding: Sm90FoldMapContractBinding,
    request: ExecutablePhysicalRequest,
    *,
    threads: int,
) -> str:
    producer_by_value = {
        output: operation for operation in program.operations for output in primitive_outputs(operation)
    }
    producer_allowed = set(binding.producer_operations)
    contributions = []
    for index, fold in enumerate(binding.folds):
        producer_aliases = {
            value: _load(value, binding, row="row", reduction="reduction")
            for value in _scalar_leaves(fold.input, producer_by_value, producer_allowed)
        }
        contributions.append(
            _render_fold_contribution(
                fold.input,
                producer_by_value,
                producer_allowed,
                producer_aliases,
                index=index,
            )
        )
    arguments = ",\n    ".join(f"torch::Tensor {_parameter_name(value)}" for value in program.inputs)
    kernel_parameters = ",\n    ".join(
        f"const {_pointer_type(value.dtype)}* __restrict__ {_parameter_name(value)}" for value in program.inputs
    )
    pointer_arguments = ",\n      ".join(
        f"reinterpret_cast<const {_pointer_type(value.dtype)}*>({_parameter_name(value)}.data_ptr())"
        for value in program.inputs
    )
    checks = "\n".join(_input_check(value, binding) for value in program.inputs)
    use_wgmma = binding.mapped_input.dtype is DType.BF16 and binding.weight.dtype is DType.BF16
    metadata = json.dumps(
        {
            "algebra_fingerprint": program.semantic_fingerprint,
            "physical_request_fingerprint": request.fingerprint,
            "physical_family": "sm90_two_kernel_fold_map_contract",
            "producer_results": [
                {"value": binding.raw_input.name, "storage": "input_alias"},
                *({"value": fold.output.name, "storage": "materialized_fp32"} for fold in binding.folds),
                *(
                    {"value": value.name, "storage": "materialized_fp32_row_map"}
                    for value in binding.row_materializations
                ),
            ],
            "contract_implementation": "cute_wgmma_map_load_contract" if use_wgmma else "scalar_fallback",
            "producer_operations": [program.operations.index(operation) for operation in binding.producer_operations],
            "consumer_operations": [program.operations.index(operation) for operation in binding.consumer_operations],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    fold_output_parameters = ",\n    ".join(
        f"float* __restrict__ fold_state_{index}" for index in range(len(binding.folds))
    )
    row_output_parameters = ",\n    ".join(
        f"float* __restrict__ row_map_{index}" for index in range(len(binding.row_materializations))
    )
    output_parameters = ",\n    ".join(
        parameter for parameter in (fold_output_parameters, row_output_parameters) if parameter
    )
    fold_passes = "\n".join(
        _fold_pass_source(index, statements, contribution)
        for index, (statements, contribution) in enumerate(contributions)
    )
    fold_allocations = "\n".join(
        f"  auto fold_state_{index} = torch::empty({{kRows}}, "
        f"{_parameter_name(binding.raw_input)}.options().dtype(torch::kFloat32));"
        for index in range(len(binding.folds))
    )
    row_allocations = "\n".join(
        f"  auto row_map_{index} = torch::empty({{kRows}}, "
        f"{_parameter_name(binding.raw_input)}.options().dtype(torch::kFloat32));"
        for index in range(len(binding.row_materializations))
    )
    fold_pointer_arguments = ",\n      ".join(
        f"fold_state_{index}.data_ptr<float>()" for index in range(len(binding.folds))
    )
    row_pointer_arguments = ",\n      ".join(
        f"row_map_{index}.data_ptr<float>()" for index in range(len(binding.row_materializations))
    )
    output_pointer_arguments = ",\n      ".join(
        argument for argument in (fold_pointer_arguments, row_pointer_arguments) if argument
    )
    row_map_source = _row_materialization_source(program, binding, producer_by_value)
    output_returns = ", ".join(
        (
            *(f"fold_state_{index}" for index in range(len(binding.folds))),
            *(f"row_map_{index}" for index in range(len(binding.row_materializations))),
        )
    )
    output_return_types = ", ".join(
        "torch::Tensor" for _ in range(len(binding.folds) + len(binding.row_materializations))
    )
    return f"""
// Generated from generic Fold/Map/Contract algebra and a common physical request; do not edit.
// request_metadata={metadata}
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tuple>

namespace {{

constexpr int kRows = {binding.rows};
constexpr int kReduction = {binding.reduction};
constexpr int kFeatures = {binding.features};
constexpr int kThreads = {threads};
constexpr int kWarps = kThreads / 32;
__global__ __launch_bounds__(kThreads) void shuttle_fold_producer(
    {kernel_parameters},
    {output_parameters}) {{
  __shared__ float partial[kWarps];
  const int row = blockIdx.x;
{fold_passes}
{row_map_source}
}}

}}  // namespace

std::tuple<{output_return_types}> fold(
    {arguments}) {{
{checks}
  const c10::cuda::CUDAGuard device_guard({_parameter_name(binding.raw_input)}.device());
{fold_allocations}
{row_allocations}
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shuttle_fold_producer<<<kRows, kThreads, 0, stream>>>(
      {pointer_arguments},
      {output_pointer_arguments});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple({output_returns});
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
  module.def("fold", &fold);
}}
""".strip()


def _fold_pass_source(index: int, statements: str, contribution: str) -> str:
    lanes = []
    for item in range(4):
        offset = "" if item == 0 else f" + {item} * kThreads"
        indented_statements = "\n".join(f"      {line}" for line in statements.splitlines())
        lanes.append(
            f"""
    if (base_{index}{offset} < kReduction) {{
      const int reduction = base_{index}{offset};
{indented_statements}
      local_{index}_{item} = __fadd_rn(local_{index}_{item}, {contribution});
    }}""".rstrip()
        )
    lane_source = "\n".join(lanes)
    return f"""
  float local_{index}_0 = 0.0f;
  float local_{index}_1 = 0.0f;
  float local_{index}_2 = 0.0f;
  float local_{index}_3 = 0.0f;
  for (int base_{index} = threadIdx.x; base_{index} < kReduction; base_{index} += 4 * kThreads) {{
{lane_source}
  }}
  float local_{index} = __fadd_rn(
      __fadd_rn(local_{index}_0, local_{index}_1),
      __fadd_rn(local_{index}_2, local_{index}_3));
  for (int offset = 16; offset > 0; offset /= 2) {{
    local_{index} = __fadd_rn(local_{index}, __shfl_down_sync(0xffffffff, local_{index}, offset));
  }}
  const int lane_{index} = threadIdx.x % 32;
  const int warp_{index} = threadIdx.x / 32;
  if (lane_{index} == 0) {{
    partial[warp_{index}] = local_{index};
  }}
  __syncthreads();
  if (warp_{index} == 0) {{
    local_{index} = lane_{index} < kWarps ? partial[lane_{index}] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) {{
      local_{index} = __fadd_rn(local_{index}, __shfl_down_sync(0xffffffff, local_{index}, offset));
    }}
    if (lane_{index} == 0) {{
      fold_state_{index}[row] = local_{index};
    }}
  }}
  __syncthreads();
""".rstrip()


def _row_materialization_source(
    program: TensorProgram,
    binding: Sm90FoldMapContractBinding,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
) -> str:
    if not binding.row_materializations:
        return ""
    allowed = {operation for operation in program.operations if isinstance(operation, (MapPrimitive, RelationPrimitive))}
    aliases = {fold.output: f"fold_state_{index}[row]" for index, fold in enumerate(binding.folds)}
    statements = []
    for index, value in enumerate(binding.row_materializations):
        expression = _render_value(value, producer_by_value, allowed, aliases)
        statements.append(f"    const float row_value_{index} = {expression};")
        statements.append(f"    row_map_{index}[row] = row_value_{index};")
        aliases[value] = f"row_value_{index}"
    return "\n".join(("  if (threadIdx.x == 0) {", *statements, "  }"))


def _serial_launch_request(
    program: TensorProgram,
    request: ExecutablePhysicalRequest,
    binding: Sm90FoldMapContractBinding,
    *,
    consumer_schedule: Sm90MapLoadContractSchedule,
) -> ExecutablePhysicalRequest:
    if set(binding.task_ids) != {task.id for task in request.task_plan.tasks}:
        raise ValueError(
            "the Fold/Map/Contract region was discovered, but the standalone generated attachment "
            "does not yet execute residual tasks"
        )
    schedule_candidates = tuple(
        schedule for schedule in request.schedules if set(binding.task_ids) <= set(schedule.task_ids)
    )
    if len(schedule_candidates) != 1:
        raise ValueError("the current serial attachment requires one schedule covering the selected task region")
    schedule = schedule_candidates[0]
    tasks_by_id = {task.id: task for task in request.task_plan.tasks}
    producer_task, consumer_task = (tasks_by_id[task_id] for task_id in binding.task_ids)
    data_dependences = tuple(
        dependence
        for dependence in request.task_plan.dependences
        if dependence.relation.source == producer_task.family and dependence.relation.target == consumer_task.family
    )
    if len(data_dependences) != 1:
        raise ValueError("the serial two-kernel attachment requires one exact producer-to-consumer dependence")
    data_dependence = data_dependences[0]
    event_plan = derive_event_tensor_plan(
        data_dependence,
        name="full_fold_and_row_map_state_ready",
        generation_policy=EventGenerationPolicy.PER_INVOCATION,
    )
    task_plan = CanonicalTaskPlan(
        request.task_plan.tasks,
        (data_dependence,),
        (),
    )
    allocation = EventTensorAllocation(
        EventStorageBinding(
            event_plan.domain,
            tuple(range(len(event_plan.domain.coordinates))),
            (0,) * len(event_plan.domain.coordinates),
        ),
        capacity=len(event_plan.domain.coordinates),
        generation_count=1,
    )
    order = PhysicalTaskOrder(
        PhysicalOrderKind.STREAM_ORDER,
        producer_task.id,
        consumer_task.id,
        event_plan.scheduled_dependence,
        {
            "stream": 0,
            "mechanism": "full_producer_launch_then_full_consumer_launch",
            "fold_state_storage": "full_materialization",
            "row_map_materialization_storage": "full_materialization",
        },
    )
    schedule = replace(
        schedule,
        parameters={
            **schedule.parameters,
            "kernel_count": 2,
            "contract_implementation": (
                "cute_wgmma_map_load_contract"
                if binding.raw_input.dtype is DType.BF16 and binding.weight.dtype is DType.BF16
                else "scalar_fallback"
            ),
            "contract_tile_m": consumer_schedule.tile_m,
            "contract_tile_n": consumer_schedule.tile_n,
            "contract_tile_k": consumer_schedule.tile_k,
            "contract_threads": consumer_schedule.threads_per_cta,
            "contract_cluster_m": consumer_schedule.cluster_m,
            "contract_cluster_n": consumer_schedule.cluster_n,
            "contract_pingpong": consumer_schedule.pingpong,
            "fold_state_storage": "full_materialization",
            "producer_results": _producer_result_records(program, binding),
            "row_map_materializations": tuple(
                tensor_program_value_ordinals(program)[value] for value in binding.row_materializations
            ),
        },
        enforced_orders=(order,),
    )
    event_program = EventDataflowProgram(
        task_plan.task_families,
        task_plan.dependences,
        (event_plan,),
    )
    audit = verify_event_realizations(
        event_program,
        (
            erased_event_realization(
                event_plan,
                kind=EventRealizationKind.ERASED_STREAM_ORDER,
                mechanism="declared serial CUDA launch order on one stream",
                reason="the full producer kernel is submitted before the full consumer kernel on the same stream",
            ),
        ),
    )
    return replace(
        request,
        task_plan=task_plan,
        event_plans=(event_plan,),
        event_allocations=(allocation,),
        schedules=(schedule,),
        event_realizations=EventRealizationProof(audit, (EventErasureWitness(event_plan.name, 0, 0),)),
        provenance={
            **request.provenance,
            "physical_attachment": "two_kernel_tiled_fold_map_contract",
        },
    )


def _producer_result_records(
    program: TensorProgram,
    binding: Sm90FoldMapContractBinding,
) -> tuple[dict[str, object], ...]:
    value_ids = tensor_program_value_ordinals(program)
    return (
        {"value_id": value_ids[binding.raw_input], "storage": "input_alias"},
        *({"value_id": value_ids[fold.output], "storage": "materialized_fp32"} for fold in binding.folds),
        *(
            {"value_id": value_ids[value], "storage": "materialized_fp32_row_map"}
            for value in binding.row_materializations
        ),
    )


def _task_has_capability(task: TiledTask, capability: TaskCapability) -> bool:
    return any(phase.capability is capability for phase in task.structure.phases)


def _task_successors(request: ExecutablePhysicalRequest) -> dict[int, set[int]]:
    task_ids = {task.family: task.id for task in request.task_plan.tasks}
    successors = {task.id: set() for task in request.task_plan.tasks}
    for dependence in request.task_plan.dependences:
        source = dependence.relation.source
        target = dependence.relation.target
        if not isinstance(source, TaskFamily) or not isinstance(target, TaskFamily):
            raise TypeError("physical task dependences must connect task-family domains")
        successors[task_ids[source]].add(task_ids[target])
    return successors


def _reachable(successors: dict[int, set[int]], starts: set[int]) -> set[int]:
    reached = set()
    frontier = list(starts)
    while frontier:
        task_id = frontier.pop()
        if task_id in reached:
            continue
        reached.add(task_id)
        frontier.extend(successors[task_id] - reached)
    return reached


def _task_subset_is_dependency_convex(owned: set[int], successors: dict[int, set[int]]) -> bool:
    for source in owned:
        frontier = list(successors[source])
        visited = set()
        while frontier:
            task_id = frontier.pop()
            if task_id in visited:
                continue
            visited.add(task_id)
            if task_id in owned:
                continue
            if _reachable(successors, {task_id}) & owned:
                return False
            frontier.extend(successors[task_id] - visited)
    return True


def _scalar_leaves(
    value: ProgramValue,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
    allowed: set[TensorPrimitive],
) -> frozenset[ProgramValue]:
    producer = producer_by_value.get(value)
    if producer is None or producer not in allowed:
        return frozenset((value,))
    if isinstance(producer, RelationPrimitive):
        return _scalar_leaves(_affine_relation_source(producer, value), producer_by_value, allowed)
    if not isinstance(producer, MapPrimitive):
        return frozenset((value,))
    return frozenset(leaf for operand in producer.inputs for leaf in _scalar_leaves(operand, producer_by_value, allowed))


def _render_value(
    value: ProgramValue,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
    allowed: set[TensorPrimitive],
    aliases: dict[ProgramValue, str],
) -> str:
    if value in aliases:
        return aliases[value]
    producer = producer_by_value.get(value)
    if producer is None or producer not in allowed:
        try:
            return aliases[value]
        except KeyError as error:
            raise ValueError(f"no SM90 scalar binding for value {value.name!r}") from error
    if isinstance(producer, RelationPrimitive):
        return _render_value(
            _affine_relation_source(producer, value),
            producer_by_value,
            allowed,
            aliases,
        )
    if not isinstance(producer, MapPrimitive):
        raise ValueError(f"SM90 scalar fusion cannot inline {type(producer).__name__}")
    operand_aliases = {
        operand.name: _render_value(operand, producer_by_value, allowed, aliases) for operand in producer.inputs
    }
    missing = scalar_expression_inputs(producer.expression) - set(operand_aliases)
    if missing:
        raise ValueError(f"scalar Map {producer.name!r} has unbound inputs {sorted(missing)}")
    if _neutral_widening_map_source(producer) is not None:
        return f"static_cast<float>({operand_aliases[producer.inputs[0].name]})"
    rendered = render_cuda_scalar_expression(producer.expression, operand_aliases)
    return _round_to_dtype(rendered, value.dtype)


def _render_fold_contribution(
    value: ProgramValue,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
    allowed: set[TensorPrimitive],
    aliases: dict[ProgramValue, str],
    *,
    index: int,
) -> tuple[str, str]:
    """Render a Fold scalar DAG once per element, preserving shared SSA values."""
    resolved = dict(aliases)
    statements = []

    def render(current: ProgramValue) -> str:
        if current in resolved:
            return resolved[current]
        producer = producer_by_value.get(current)
        if producer is None or producer not in allowed:
            raise ValueError(f"no SM90 scalar binding for value {current.name!r}")
        if isinstance(producer, RelationPrimitive):
            rendered = render(_affine_relation_source(producer, current))
            resolved[current] = rendered
            return rendered
        if not isinstance(producer, MapPrimitive):
            raise ValueError(f"SM90 scalar fusion cannot inline {type(producer).__name__}")
        operand_aliases = {operand.name: render(operand) for operand in producer.inputs}
        missing = scalar_expression_inputs(producer.expression) - set(operand_aliases)
        if missing:
            raise ValueError(f"scalar Map {producer.name!r} has unbound inputs {sorted(missing)}")
        if _neutral_widening_map_source(producer) is not None:
            expression = f"static_cast<float>({operand_aliases[producer.inputs[0].name]})"
        else:
            expression = _round_to_dtype(
                render_cuda_scalar_expression(producer.expression, operand_aliases),
                current.dtype,
            )
        temporary = f"fold_{index}_value_{len(statements)}"
        statements.append(f"const float {temporary} = {expression};")
        resolved[current] = temporary
        return temporary

    contribution = render(value)
    return "\n".join(statements), contribution


def _fused_scalar_expression(
    value: ProgramValue,
    producer_by_value: dict[ProgramValue, TensorPrimitive],
    allowed: set[TensorPrimitive],
) -> ScalarExpression:
    producer = producer_by_value.get(value)
    if producer is None or producer not in allowed:
        return scalar_input(value.name)
    if isinstance(producer, RelationPrimitive):
        return _fused_scalar_expression(_affine_relation_source(producer, value), producer_by_value, allowed)
    if not isinstance(producer, MapPrimitive):
        raise ValueError(f"SM90 scalar fusion cannot inline {type(producer).__name__}")
    replacements = {
        operand.name: _fused_scalar_expression(operand, producer_by_value, allowed) for operand in producer.inputs
    }
    return scalar_convert(
        _substitute_scalar_inputs(producer.expression, replacements),
        result_dtype=value.dtype,
        semantics=producer.output_conversion,
    )


def _substitute_scalar_inputs(
    expression: ScalarExpression,
    replacements: dict[str, ScalarExpression],
) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        try:
            return replacements[expression.input_name]
        except KeyError as error:
            raise ValueError(f"scalar Map input {expression.input_name!r} has no fused value") from error
    if not expression.operands:
        return expression
    return replace(
        expression,
        operands=tuple(_substitute_scalar_inputs(operand, replacements) for operand in expression.operands),
    )


def _affine_relation_source(operation: RelationPrimitive, output: ProgramValue) -> ProgramValue:
    if operation.predicate is not None:
        raise ValueError("SM90 scalar fusion cannot inline a predicated Relation")
    try:
        output_index = operation.outputs.index(output)
    except ValueError as error:
        raise ValueError(f"Relation {operation.name!r} does not produce {output.name!r}") from error
    if operation.resolved_result_expressions[output_index] is not None:
        raise ValueError("SM90 scalar fusion cannot inline a value-producing index expression")
    source = operation.resolved_result_sources[output_index]
    if source is None:
        raise ValueError("SM90 scalar fusion requires one indexed Relation source")
    indexing = next((item for item in operation.input_indexing if item.operand == source), None)
    if indexing is None or indexing.predicate is not None:
        raise ValueError("SM90 scalar fusion requires an unpredicated operand indexing relation")
    for axis, expression in zip(source.axes, indexing.indices, strict=True):
        if expression.kind is IndexExpressionKind.AXIS and expression.axis == axis:
            continue
        if expression.kind is IndexExpressionKind.CONSTANT and expression.constant == 0 and axis.extent == 1:
            continue
        raise ValueError("SM90 scalar fusion supports only identity and singleton-broadcast affine Relations")
    return source


def _load(
    value: ProgramValue,
    binding: Sm90FoldMapContractBinding,
    *,
    row: str,
    reduction: str,
    feature: str | None = None,
) -> str:
    for index, fold in enumerate(binding.folds):
        if value is fold.output:
            return f"fold_state_{index}[{row}]"
    if value is binding.weight:
        if feature is None:
            raise ValueError("the Contract weight requires a feature-domain index")
        loaded = f"{_parameter_name(value)}[{reduction} * kFeatures + {feature}]"
        return f"__bfloat162float({loaded})" if value.dtype is DType.BF16 else loaded
    axes = value.axes
    row_axis = binding.fold_state.axes[0]
    reduction_axis = binding.fold.reduction_axes[0]
    feature_axis = binding.output.axes[1]
    if axes == (row_axis, reduction_axis):
        index = f"{row} * kReduction + {reduction}"
    elif axes == (reduction_axis,):
        index = reduction
    elif axes == (row_axis,):
        index = row
    elif axes == (reduction_axis, feature_axis) and feature is not None:
        index = f"{reduction} * kFeatures + {feature}"
    elif not axes:
        index = "0"
    else:
        raise ValueError(f"unsupported SM90 scalar input axes for {value.name!r}: {[axis.id for axis in axes]}")
    loaded = f"{_parameter_name(value)}[{index}]"
    return f"__bfloat162float({loaded})" if value.dtype is DType.BF16 else loaded


def _round_to_dtype(expression: str, dtype: DType) -> str:
    if dtype is DType.BF16:
        return f"__bfloat162float(__float2bfloat16_rn({expression}))"
    if dtype is DType.FP32:
        return expression
    raise ValueError(f"unsupported SM90 scalar result dtype {dtype.value!r}")


def _pointer_type(dtype: DType) -> str:
    if dtype is DType.BF16:
        return "__nv_bfloat16"
    if dtype is DType.FP32:
        return "float"
    raise ValueError(f"unsupported SM90 pointer dtype {dtype.value!r}")


def _torch_dtype(dtype: DType) -> str:
    if dtype is DType.BF16:
        return "torch::kBFloat16"
    if dtype is DType.FP32:
        return "torch::kFloat32"
    raise ValueError(f"unsupported SM90 Torch dtype {dtype.value!r}")


def _input_check(value: ProgramValue, binding: Sm90FoldMapContractBinding) -> str:
    parameter = _parameter_name(value)
    shape = tuple(axis.extent for axis in value.axes)
    dimensions = " && ".join(f"{parameter}.size({index}) == {extent}" for index, extent in enumerate(shape))
    shape_check = f"{parameter}.dim() == {len(shape)}" + (f" && {dimensions}" if dimensions else "")
    return (
        f'  TORCH_CHECK({parameter}.is_cuda(), "input {value.name} must be CUDA");\n'
        f"  TORCH_CHECK({parameter}.scalar_type() == {_torch_dtype(value.dtype)}, "
        f'"input {value.name} dtype mismatch");\n'
        f'  TORCH_CHECK({parameter}.is_contiguous(), "input {value.name} must be contiguous");\n'
        f'  TORCH_CHECK({shape_check}, "input {value.name} shape mismatch");'
    )


def _parameter_name(value: ProgramValue) -> str:
    if value.name.isidentifier():
        return value.name
    digest = hashlib.sha256(value.name.encode()).hexdigest()[:12]
    return f"tensor_{digest}"
