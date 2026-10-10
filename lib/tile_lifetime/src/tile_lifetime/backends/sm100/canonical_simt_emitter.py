# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit an executable SIMT fallback from canonical scheduled components."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from shuttle.ir import DType
from tile_lifetime.backends.sm100.refined_schedule import (
    REFINED_GATHERED_NORMALIZED_EXP_SCHEDULE_FAMILY,
    REFINED_SELECTION_SCHEDULE_FAMILY,
)
from tile_lifetime.canonical_kernel_composition import (
    CanonicalKernelComponent,
    CanonicalKernelComposition,
    CuteKernelComponentFragment,
    compose_cute_pipeline_launch_recipe,
    lower_canonical_kernel_composition,
)
from tile_lifetime.cuda_map_fold_codegen import render_cuda_scalar_expression
from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelFunctionAttribute,
    CuteKernelFunctionAttributeKind,
    CuteKernelLaunch,
    CutePipelineElementType,
    CutePipelineLaunchRecipe,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
)
from tile_lifetime.physical_task_contract import (
    ExecutablePhysicalRequest,
    PhysicalNumericalPolicy,
    PhysicalSchedule,
    PhysicalTarget,
    install_physical_schedule_cover,
)
from tile_lifetime.sm100_canonical_kernel_composition import (
    SM100CanonicalKernelComposition,
    SM100ConcreteEventBoundary,
    SM100KernelComponent,
    SM100ParametricEventBoundary,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    IndexExpression,
    IndexExpressionKind,
    IndexPredicate,
    IndexPredicateKind,
    MapPrimitive,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    SelectionComparison,
    SelectionOrder,
    SelectionPrimitive,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    TransportPrimitive,
    deserialize_tensor_program,
    execute_tensor_program,
    primitive_inputs,
    primitive_outputs,
    tensor_program_value_ordinals,
)

SIMT_SCHEDULE_FAMILY = "sm100_simt_contract_relations_maps"
SIMT_CANONICAL_SCHEDULE_FAMILY = "sm100_simt_canonical_multi_launch"
SIMT_THREADS = 256


@dataclass(frozen=True)
class GeneratedSm100SimtComponent:
    """Generated CUDA and launch fragment for one canonical component."""

    component_id: int
    source: str
    source_sha256: str
    fragment: CuteKernelComponentFragment


@dataclass(frozen=True)
class Sm100CanonicalEmitterCoverage:
    """Executable fragments plus exact unsupported and Event boundaries."""

    generated: tuple[GeneratedSm100SimtComponent, ...]
    unsupported_components: tuple[tuple[int, str], ...]
    concrete_events: tuple[SM100ConcreteEventBoundary, ...]
    parametric_events: tuple[SM100ParametricEventBoundary, ...]

    @property
    def fragments(self) -> tuple[CuteKernelComponentFragment, ...]:
        return tuple(item.fragment for item in self.generated)

    @property
    def complete(self) -> bool:
        return not self.unsupported_components


@dataclass(frozen=True)
class RuntimeEventStreamOrderErasure:
    """A runtime-parametric Event relation covered by whole-kernel stream order."""

    dependence_program_id: int
    relation_program_fingerprint: str
    producer_launch: int
    consumer_launch: int

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "dependence_program_id": self.dependence_program_id,
            "relation_program_fingerprint": self.relation_program_fingerprint,
            "producer_launch": self.producer_launch,
            "consumer_launch": self.consumer_launch,
            "realization": "stream_order",
        }


@dataclass(frozen=True)
class CompleteSm100SimtPipeline:
    """Complete generated source, launch recipe, and parametric Event proofs."""

    source: str
    source_sha256: str
    recipe: CutePipelineLaunchRecipe
    runtime_event_erasures: tuple[RuntimeEventStreamOrderErasure, ...]


def emit_sm100_canonical_simt_components(
    composition: SM100CanonicalKernelComposition,
) -> Sm100CanonicalEmitterCoverage:
    """Emit every component selected for the generic multi-launch SIMT family."""
    program = _tensor_program(composition)
    ordinals = tensor_program_value_ordinals(program)
    task_by_operation = {
        operation_ordinal: task.id
        for task in composition.scheduled_request.task_plan.tasks
        for operation_ordinal in task.operation_ordinals
    }
    generated: list[GeneratedSm100SimtComponent] = []
    unsupported: list[tuple[int, str]] = []
    for component in composition.components:
        schedule = composition.scheduled_request.schedules[component.schedule_index]
        reason = _unsupported_schedule_reason(schedule.family.name, schedule.parameters)
        if reason is not None:
            unsupported.append((component.id, reason))
            continue
        threads = schedule.parameters["threads"]
        assert type(threads) is int
        generated.append(
            _emit_component(
                component,
                program,
                ordinals,
                task_by_operation,
                schedule_family=schedule.family.name,
                schedule_parameters=schedule.parameters,
                threads=threads,
            )
        )
    return Sm100CanonicalEmitterCoverage(
        tuple(generated),
        tuple(unsupported),
        composition.concrete_events,
        composition.parametric_events,
    )


def compose_sm100_canonical_simt_pipeline(
    composition: SM100CanonicalKernelComposition | CanonicalKernelComposition,
) -> CompleteSm100SimtPipeline:
    """Compose a complete SIMT source and erase cross-kernel Events by stream order."""
    if isinstance(composition, CanonicalKernelComposition):
        return _compose_native_simt_pipeline(composition)
    coverage = emit_sm100_canonical_simt_components(composition)
    if not coverage.complete:
        raise ValueError(f"canonical SM100 SIMT emission is incomplete: {coverage.unsupported_components}")
    canonical = lower_canonical_kernel_composition(composition.scheduled_request)
    generated_by_tasks = {composition.components[item.component_id].task_ids: item for item in coverage.generated}
    fragments = []
    sources = []
    for component in canonical.components:
        try:
            generated = generated_by_tasks[component.task_ids]
        except KeyError as error:
            raise ValueError(f"canonical SM100 component tasks {component.task_ids} have no emitted source") from error
        fragments.append(
            replace(
                generated.fragment,
                component_id=component.id,
                schedule_fingerprint=component.schedule_fingerprint,
            )
        )
        sources.append(generated.source)
    recipe = compose_cute_pipeline_launch_recipe(canonical, tuple(fragments))
    launch_indices_by_task: dict[int, list[int]] = {}
    for launch_index, launch in enumerate(recipe.launches):
        for task_id in launch.source_tasks:
            launch_indices_by_task.setdefault(task_id, []).append(launch_index)
    runtime_erasures = tuple(
        RuntimeEventStreamOrderErasure(
            event.dependence_program_id,
            event.relation.relation_program_fingerprint,
            max(launch_indices_by_task[event.relation.source_task_id]),
            min(launch_indices_by_task[event.relation.target_task_id]),
        )
        for event in composition.parametric_events
    )
    if any(item.producer_launch >= item.consumer_launch for item in runtime_erasures):
        raise ValueError(
            "runtime-parametric Event relation is not covered by forward stream order: "
            f"{tuple(item.canonical_record for item in runtime_erasures)}"
        )
    source = "\n".join(sources)
    return CompleteSm100SimtPipeline(
        source,
        hashlib.sha256(source.encode()).hexdigest(),
        recipe,
        runtime_erasures,
    )


def lower_sm100_native_simt_kernel_composition(
    request: ExecutablePhysicalRequest,
) -> CanonicalKernelComposition:
    """Install the generic ordered SIMT family on an arbitrary SM100 request."""
    if request.target is not PhysicalTarget.SM100:
        raise ValueError("canonical SM100 SIMT composition requires an SM100 request")
    if request.numerical_policy not in {PhysicalNumericalPolicy.FAST, PhysicalNumericalPolicy.SOURCE_ORDERED}:
        raise ValueError(f"unsupported canonical SM100 numerical policy {request.numerical_policy.value!r}")
    schedules = tuple(
        PhysicalSchedule(
            family=replace(schedule.family, name=SIMT_CANONICAL_SCHEDULE_FAMILY),
            task_ids=schedule.task_ids,
            parameters={
                "matrix_instruction": "cuda_core_fma",
                "operation_realization": "ordered_multi_launch",
                "threads": SIMT_THREADS,
            },
            enforced_orders=schedule.enforced_orders,
        )
        for schedule in request.schedules
    )
    installed = install_physical_schedule_cover(
        request,
        schedules,
        source_request_fingerprint=request.fingerprint,
    )
    composition = lower_canonical_kernel_composition(installed)
    return replace(composition, source_request_fingerprint=request.fingerprint)


def _compose_native_simt_pipeline(composition: CanonicalKernelComposition) -> CompleteSm100SimtPipeline:
    program = _tensor_program_for_request(composition.scheduled_request)
    ordinals = tensor_program_value_ordinals(program)
    task_by_operation: dict[int, int] = {}
    for task in composition.scheduled_request.task_plan.tasks:
        for operation_ordinal in task.operation_ordinals:
            previous = task_by_operation.setdefault(operation_ordinal, task.id)
            if previous != task.id:
                raise ValueError("canonical SM100 SIMT emission requires every algebra operation to have one task owner")
    fragments = []
    sources = []
    for component in composition.components:
        schedule = composition.scheduled_request.schedules[component.schedule_index]
        reason = _unsupported_schedule_reason(schedule.family.name, schedule.parameters)
        if reason is not None:
            raise ValueError(f"canonical SM100 SIMT component {component.id} is unsupported: {reason}")
        generated = _emit_component(
            component,
            program,
            ordinals,
            task_by_operation,
            schedule_family=schedule.family.name,
            schedule_parameters=schedule.parameters,
            threads=SIMT_THREADS,
        )
        fragments.append(generated.fragment)
        sources.append(generated.source)
    recipe = compose_cute_pipeline_launch_recipe(composition, tuple(fragments))
    launch_indices_by_task: dict[int, list[int]] = {}
    for launch_index, launch in enumerate(recipe.launches):
        for task_id in launch.source_tasks:
            launch_indices_by_task.setdefault(task_id, []).append(launch_index)
    runtime_erasures = tuple(
        RuntimeEventStreamOrderErasure(
            event.dependence_program_index,
            dependence.relation_program.semantic_fingerprint,
            max(launch_indices_by_task[dependence.source_task_id]),
            min(launch_indices_by_task[dependence.target_task_id]),
        )
        for event in composition.scheduled_request.event_programs
        for dependence in (composition.scheduled_request.task_plan.dependence_programs[event.dependence_program_index],)
    )
    if any(item.producer_launch >= item.consumer_launch for item in runtime_erasures):
        raise ValueError(
            "runtime-parametric Event relation is not covered by forward stream order: "
            f"{tuple(item.canonical_record for item in runtime_erasures)}"
        )
    source = "\n".join(sources)
    return CompleteSm100SimtPipeline(
        source,
        hashlib.sha256(source.encode()).hexdigest(),
        recipe,
        runtime_erasures,
    )


def _unsupported_schedule_reason(family: str, parameters: Mapping[str, Any]) -> str | None:
    if family == REFINED_GATHERED_NORMALIZED_EXP_SCHEDULE_FAMILY:
        expected = {
            "matrix_instruction": "tcgen05_m64n128k16_bf16_fp32_pair",
            "component_realization": "dependency_convex_refinement",
            "threads": SIMT_THREADS,
            "tensor_warps": 8,
            "resident_warps_per_cta": 8,
            "active_tensor_warps": 1,
            "normalized_passes": 3,
            "pipeline_stages": 1,
            "tensor_memory_columns": 256,
            "tensor_core_warps": 1,
        }
        mismatches = tuple(key for key, value in expected.items() if parameters.get(key) != value)
        owner_rows = parameters.get("owner_rows")
        if not isinstance(owner_rows, int) or not 0 < owner_rows <= 16:
            mismatches = (*mismatches, "owner_rows")
        return (
            None if not mismatches else f"SM100 gathered normalized-exp schedule has unsupported parameters {mismatches}"
        )
    if family == REFINED_SELECTION_SCHEDULE_FAMILY:
        expected = {
            "matrix_instruction": "tcgen05_m64n128k16_bf16_fp32",
            "component_realization": "dependency_convex_refinement",
            "threads": SIMT_THREADS,
            "score_tile_rows": 64,
            "score_tile_columns": 128,
            "contraction_tile": 128,
            "pipeline_stages": 1,
            "tensor_memory_columns": 128,
            "tensor_core_warps": 1,
            "selection_warps": 2,
        }
        mismatches = tuple(key for key, value in expected.items() if parameters.get(key) != value)
        return None if not mismatches else f"SM100 refined Selection schedule has unsupported parameters {mismatches}"
    if family not in {SIMT_SCHEDULE_FAMILY, SIMT_CANONICAL_SCHEDULE_FAMILY}:
        return f"schedule family {family!r} has no canonical SM100 emitter"
    expected = {
        "matrix_instruction": "cuda_core_fma",
        "operation_realization": "ordered_multi_launch",
        "threads": SIMT_THREADS,
    }
    mismatches = tuple(key for key, value in expected.items() if parameters.get(key) != value)
    if mismatches:
        return f"SM100 SIMT schedule has unsupported parameters {mismatches}"
    return None


def _tensor_program(composition: SM100CanonicalKernelComposition) -> TensorProgram:
    return _tensor_program_for_request(composition.scheduled_request)


def _tensor_program_for_request(request: ExecutablePhysicalRequest) -> TensorProgram:
    record = request.algebra_record
    if record.get("format") == "shuttle_domain_restricted_tensor_program_v1":
        record = record.get("program")
    if not isinstance(record, dict) or record.get("format") != "shuttle_tensor_program_v1":
        raise ValueError("canonical SM100 emission requires a TensorProgram algebra record")
    return deserialize_tensor_program(json.dumps(record, sort_keys=True, separators=(",", ":")))


def _emit_component(
    component: SM100KernelComponent | CanonicalKernelComponent,
    program: TensorProgram,
    ordinals: dict[ProgramValue, int],
    task_by_operation: dict[int, int],
    *,
    schedule_family: str,
    schedule_parameters: Mapping[str, Any],
    threads: int,
) -> GeneratedSm100SimtComponent:
    operations = tuple(program.operations[ordinal] for ordinal in component.operation_ordinals)
    if schedule_family == REFINED_SELECTION_SCHEDULE_FAMILY:
        return _emit_refined_selection_component(component, program, ordinals, operations, threads=threads)
    if schedule_family == REFINED_GATHERED_NORMALIZED_EXP_SCHEDULE_FAMILY:
        return _emit_refined_gathered_normalized_exp_component(
            component,
            program,
            ordinals,
            operations,
            schedule_parameters=schedule_parameters,
            threads=threads,
        )
    supported = (
        RelationPrimitive,
        MapPrimitive,
        ContractPrimitive,
        FoldPrimitive,
        SelectionPrimitive,
        TransportPrimitive,
    )
    if any(not isinstance(operation, supported) for operation in operations):
        kinds = tuple(type(operation).__name__ for operation in operations if not isinstance(operation, supported))
        raise ValueError(f"SM100 SIMT component {component.id} contains unsupported operations {kinds}")
    values = tuple(
        sorted(
            {
                value
                for operation in operations
                for value in (*primitive_inputs(operation), *primitive_outputs(operation))
            },
            key=ordinals.__getitem__,
        )
    )
    slots = {
        ordinals[value]: CutePipelineSlot(
            name=f"local_value_{ordinals[value]}",
            element_type=_element_type(value.dtype),
            shape=value.shape,
            kind=CutePipelineSlotKind.SCRATCH,
            alignment=16,
        )
        for value in values
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "schedule": component.schedule_fingerprint,
                "operations": component.operations,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    source_parts = [
        "#include <cuda_bf16.h>",
        "#include <stdint.h>",
        "#include <math.h>",
        """#ifndef SHUTTLE_TOTAL_ORDER_KEY_DEFINED
#define SHUTTLE_TOTAL_ORDER_KEY_DEFINED
static __device__ __forceinline__ uint32_t shuttle_total_order_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
#endif""",
    ]
    launches = []
    for local_ordinal, (operation_ordinal, operation) in enumerate(
        zip(component.operation_ordinals, operations, strict=True)
    ):
        symbol = f"shuttle_sm100_simt_{identity}_{local_ordinal}"
        source_parts.append(render_canonical_simt_kernel(symbol, operation, ordinals))
        operation_values = (*primitive_inputs(operation), *primitive_outputs(operation))
        outputs = set(primitive_outputs(operation))
        arguments = tuple(
            CutePipelineSlotArgument(
                name=f"argument_{ordinals[value]}",
                slot=slots[ordinals[value]].name,
                access=CutePipelineSlotAccess.WRITE if value in outputs else CutePipelineSlotAccess.READ,
            )
            for value in operation_values
        )
        output_elements = _operation_work_items(operation)
        launches.append(
            CuteKernelLaunch(
                name=f"operation_{local_ordinal}",
                kernel_symbol=symbol,
                grid=((output_elements + threads - 1) // threads, 1, 1),
                block=(threads, 1, 1),
                dynamic_shared_memory_bytes=0,
                function_attributes=(),
                arguments=arguments,
                source_tasks=(task_by_operation[operation_ordinal],),
            )
        )
    source = "\n\n".join(source_parts) + "\n"
    fragment = CuteKernelComponentFragment(
        component.id,
        component.schedule_fingerprint,
        slots,
        tuple(launches),
    )
    return GeneratedSm100SimtComponent(
        component.id,
        source,
        hashlib.sha256(source.encode()).hexdigest(),
        fragment,
    )


def _emit_refined_selection_component(
    component: SM100KernelComponent | CanonicalKernelComponent,
    program: TensorProgram,
    ordinals: dict[ProgramValue, int],
    operations: tuple[TensorPrimitive, ...],
    *,
    threads: int,
) -> GeneratedSm100SimtComponent:
    """Emit one tcgen05 Contract/block-Fold/Selection kernel from typed algebra roles."""
    contracts = tuple(operation for operation in operations if isinstance(operation, ContractPrimitive))
    folds = tuple(operation for operation in operations if isinstance(operation, FoldPrimitive))
    selections = tuple(operation for operation in operations if isinstance(operation, SelectionPrimitive))
    relations = tuple(operation for operation in operations if isinstance(operation, RelationPrimitive))
    layout_relations = tuple(relation for relation in relations for fold in folds if fold.input in relation.outputs)
    if len(contracts) != 1 or len(folds) != 1 or len(selections) != 1 or len(layout_relations) != 1:
        raise ValueError("refined SM100 Selection emission requires one Contract, Fold, Selection, and layout Relation")
    contract, fold, selection, relation = contracts[0], folds[0], selections[0], layout_relations[0]
    if threads != 256 or contract.accumulation_dtype is not DType.FP32:
        raise ValueError("refined SM100 Selection tcgen05 requires 256 threads and FP32 accumulation")
    if fold.reducer is not FoldReducer.MAXIMUM or fold.input not in relation.outputs:
        raise ValueError("refined SM100 Selection requires a relation-view maximum Fold")
    if selection.order is not SelectionOrder.DESCENDING or selection.comparison is not SelectionComparison.TOTAL_ORDER:
        raise ValueError("refined SM100 Selection requires descending total order")
    if not selection.stable or selection.tie_break.value != "source_index_ascending":
        raise ValueError("refined SM100 Selection requires stable source-index tie breaking")
    if len(contract.output.axes) != 3 or len(fold.output.axes) != 3 or len(selection.values.axes) != 3:
        raise ValueError("refined SM100 Selection currently requires rank-three score, Fold, and result tensors")
    query_extent, group_extent, right_extent = contract.output.shape
    block_count = selection.dimension.extent
    fold_extent = fold.reduction_axes[0].extent
    feature_extent = contract.reduction_axes[0].extent
    if feature_extent != 128 or fold_extent != 128 or right_extent != block_count * fold_extent:
        raise ValueError("refined SM100 Selection requires 128-wide feature and Fold tiles")
    if query_extent % 64:
        raise ValueError("refined SM100 Selection tcgen05 requires query extent divisible by 64")
    if len(relation.inputs) != 1 or len(relation.outputs) != 1:
        raise ValueError("refined SM100 Selection requires one exact score-layout relation")
    relation_input = np.arange(math.prod(relation.inputs[0].shape), dtype=np.float32).reshape(relation.inputs[0].shape)
    relation_program = TensorProgram(
        relation.inputs,
        (relation,),
        relation.outputs,
        program.numerical_policy,
    )
    relation_output = execute_tensor_program(relation_program, {relation.inputs[0].name: relation_input})[
        relation.outputs[0].name
    ]
    if not np.array_equal(relation_output.reshape(-1), relation_input.reshape(-1)):
        raise ValueError("refined SM100 Selection requires an exact row-major score-layout relation")

    produced_by = {value: operation for operation in operations for value in primitive_outputs(operation)}
    internal_outputs = set(produced_by)
    internal_contract_inputs = tuple(value for value in contract.inputs if value in internal_outputs)
    if len(internal_contract_inputs) != 1:
        raise ValueError("refined SM100 Selection requires one fused Contract input conversion")
    converted_right = internal_contract_inputs[0]
    conversion = produced_by[converted_right]
    if not isinstance(conversion, MapPrimitive) or converted_right.dtype is not DType.BF16:
        raise ValueError("refined SM100 Selection Contract input must be a BF16 scalar Map")
    left = next(value for value in contract.inputs if value != converted_right)
    if left.dtype is not DType.BF16 or left.shape != (query_extent, group_extent, feature_extent):
        raise ValueError("refined SM100 Selection left operand must be row-major [owner,group,feature] BF16")
    if len(conversion.inputs) != 1 or conversion.inputs[0].shape != (right_extent, feature_extent):
        raise ValueError("refined SM100 Selection right conversion must preserve [right,feature]")
    right_source = conversion.inputs[0]

    relation_source = relation.inputs[0]
    finalizer = produced_by.get(selection.score)
    if not isinstance(finalizer, MapPrimitive) or fold.output not in finalizer.inputs:
        raise ValueError("refined SM100 Selection requires one Fold-dependent scalar finalizer")

    # Intermediates produced before this component are not necessarily public program inputs.
    external_inputs = tuple(
        sorted(
            {item for operation in operations for item in primitive_inputs(operation)} - internal_outputs,
            key=ordinals.__getitem__,
        )
    )
    if {ordinals[value] for value in external_inputs} != set(component.live_input_value_ids):
        raise ValueError("refined SM100 Selection live inputs disagree with its typed algebra closure")
    outputs = tuple(
        value for value in (selection.values, selection.indices) if ordinals[value] in component.live_output_value_ids
    )
    if selection.indices not in outputs:
        raise ValueError("refined SM100 Selection requires a live bounded index result")
    slots = {
        ordinals[value]: CutePipelineSlot(
            name=f"local_value_{ordinals[value]}",
            element_type=_element_type(value.dtype),
            shape=value.shape,
            kind=CutePipelineSlotKind.SCRATCH,
            alignment=16,
        )
        for value in (*external_inputs, *outputs)
    }

    score_coordinates = {
        contract.output.axes[0]: "query",
        contract.output.axes[1]: "group",
        contract.output.axes[2]: "right",
    }
    resolver = _SymbolicComponentExpression(
        operations,
        ordinals,
        {contract.output: "score_value", fold.output: "block_maximum"},
    )
    score_expression = resolver.value(relation_source, score_coordinates)
    fold_coordinates = {
        fold.output.axes[0]: "query",
        fold.output.axes[1]: "group",
        fold.output.axes[2]: "right_block",
    }
    finalized_expression = resolver.value(selection.score, fold_coordinates)
    converted_expression = render_cuda_scalar_expression(
        conversion.expression,
        {right_source.name: "right_source_value"},
    )
    parameters = _parameters(external_inputs, outputs, ordinals)
    value_store = (
        f"argument_{ordinals[selection.values]}[output] = best_value[slot];" if selection.values in outputs else ""
    )
    identity = hashlib.sha256(
        json.dumps(
            {"schedule": component.schedule_fingerprint, "operations": component.operations},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    symbol = f"shuttle_sm100_refined_selection_{identity}"
    source = f"""#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#ifndef SHUTTLE_TOTAL_ORDER_KEY_DEFINED
#define SHUTTLE_TOTAL_ORDER_KEY_DEFINED
static __device__ __forceinline__ uint32_t shuttle_total_order_key(float value) {{
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}}
#endif
extern "C" __global__ void {symbol}({parameters}) {{
  using namespace cute;
  using Element = cutlass::bfloat16_t;
  using TileShape = Shape<_64, _128, _128>;
  using Mma = SM100_MMA_F16BF16_SS<
      Element, Element, float, 64, 128, UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Mma{{}}));
  using SmemLayoutLeft = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<Element>{{}}, Shape<_64, _128>{{}}));
  using SmemLayoutRight = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<Element>{{}}, Shape<_128, _128>{{}}));
  constexpr int feature_extent = {feature_extent};
  constexpr int fold_extent = {fold_extent};
  constexpr int block_count = {block_count};
  constexpr int selected_count = {selection.count};
  const int query_base = static_cast<int>(blockIdx.x) * 64;
  const int group = static_cast<int>(blockIdx.y);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  __shared__ alignas(128) Element shared_left_storage[cosize_v<SmemLayoutLeft>];
  __shared__ alignas(128) Element shared_right_storage[cosize_v<SmemLayoutRight>];
  __shared__ float shared_score[64 * fold_extent];
  __shared__ float shared_maximum[64 * block_count];
  __shared__ alignas(16) uint64_t tensor_core_barrier;
  __shared__ uint32_t tensor_memory_base;
  Tensor shared_left = make_tensor(make_smem_ptr(shared_left_storage), SmemLayoutLeft{{}});
  Tensor shared_right = make_tensor(make_smem_ptr(shared_right_storage), SmemLayoutRight{{}});
  for (int index = threadIdx.x; index < 64 * feature_extent; index += blockDim.x) {{
    const int local_query = index / feature_extent;
    const int feature = index % feature_extent;
    const int query = query_base + local_query;
    shared_left(local_query, feature) = Element(
        argument_{ordinals[left]}[(query * {group_extent} + group) * feature_extent + feature]);
  }}
  if (warp == 0) {{
    TMEM::Allocator1Sm allocator;
    allocator.allocate(128, &tensor_memory_base);
    __syncwarp();
  }}
  __syncthreads();
  for (int right_block = 0; right_block < block_count; ++right_block) {{
    for (int index = threadIdx.x; index < fold_extent * feature_extent; index += blockDim.x) {{
      const int local_right = index / feature_extent;
      const int feature = index % feature_extent;
      const int right = right_block * fold_extent + local_right;
      const float right_source_value = static_cast<float>(
          argument_{ordinals[right_source]}[right * feature_extent + feature]);
      shared_right(local_right, feature) = Element({converted_expression});
    }}
    cutlass::arch::fence_view_async_shared();
    if (threadIdx.x == 0) {{
      cutlass::arch::ClusterBarrier::init(&tensor_core_barrier, 1);
      cutlass::arch::fence_barrier_init();
    }}
    __syncthreads();
    if (warp == 0) {{
      TiledMma mma;
      auto thread_mma = mma.get_slice(0);
      Tensor partitioned_left = thread_mma.partition_A(shared_left);
      Tensor partitioned_right = thread_mma.partition_B(shared_right);
      Tensor fragment_left = thread_mma.make_fragment_A(partitioned_left);
      Tensor fragment_right = thread_mma.make_fragment_B(partitioned_right);
      Tensor fragment_score = partition_fragment_C(mma, Shape<_64, _128>{{}});
      fragment_score.data() = tensor_memory_base;
      mma.accumulate_ = UMMA::ScaleOut::Zero;
      #pragma unroll
      for (int feature_block = 0; feature_block < size<2>(fragment_left); ++feature_block) {{
        cute::gemm(mma, fragment_left(_, _, feature_block), fragment_right(_, _, feature_block), fragment_score);
        mma.accumulate_ = UMMA::ScaleOut::One;
      }}
      cutlass::arch::umma_arrive(&tensor_core_barrier);
    }}
    cutlass::arch::ClusterBarrier::wait(&tensor_core_barrier, 0);
    if (threadIdx.x < 128) {{
      Tensor tensor_score = partition_fragment_C(TiledMma{{}}, Shape<_64, _128>{{}});
      tensor_score.data() = tensor_memory_base;
      Tensor score_coordinates = make_identity_tensor(Shape<_64, _128>{{}});
      Tensor partitioned_score_coordinates = TiledMma{{}}.get_slice(0).partition_C(score_coordinates);
      using TensorMemoryLoad = SM100_TMEM_LOAD_16dp32b32x;
      auto tiled_load = make_tmem_copy(TensorMemoryLoad{{}}, tensor_score);
      auto thread_load = tiled_load.get_slice(threadIdx.x);
      Tensor load_source = thread_load.partition_S(tensor_score);
      Tensor load_coordinates = thread_load.partition_D(partitioned_score_coordinates);
      Tensor loaded_score = make_tensor<float>(shape(load_coordinates));
      copy(tiled_load, load_source, loaded_score);
      #pragma unroll
      for (int item = 0; item < size(loaded_score); ++item) {{
        const int local_query = int(get<0>(load_coordinates(item)));
        const int local_right = int(get<1>(load_coordinates(item)));
        shared_score[local_query * fold_extent + local_right] = loaded_score(item);
      }}
    }}
    __syncthreads();
    for (int local_query = threadIdx.x; local_query < 64; local_query += blockDim.x) {{
      const int query = query_base + local_query;
      float block_maximum = -INFINITY;
      for (int local_right = 0; local_right < fold_extent; ++local_right) {{
        const int right = right_block * fold_extent + local_right;
        const float score_value = shared_score[local_query * fold_extent + local_right];
        const float mapped_score = static_cast<float>({score_expression});
        block_maximum = (isnan(block_maximum) || isnan(mapped_score)) ? NAN : fmaxf(block_maximum, mapped_score);
      }}
      shared_maximum[local_query * block_count + right_block] = static_cast<float>({finalized_expression});
    }}
    __syncthreads();
  }}
  if (threadIdx.x < 64) {{
    const int local_query = threadIdx.x;
    const int query = query_base + local_query;
    float best_value[selected_count];
    int best_index[selected_count];
    #pragma unroll
    for (int slot = 0; slot < selected_count; ++slot) {{
      best_value[slot] = -INFINITY;
      best_index[slot] = -1;
    }}
    for (int candidate = 0; candidate < block_count; ++candidate) {{
      const float value = shared_maximum[local_query * block_count + candidate];
      const uint32_t key = shuttle_total_order_key(value);
      int position = selected_count;
      #pragma unroll
      for (int slot = 0; slot < selected_count; ++slot) {{
        const uint32_t incumbent = shuttle_total_order_key(best_value[slot]);
        if (key > incumbent || (key == incumbent && (best_index[slot] < 0 || candidate < best_index[slot]))) {{
          position = slot;
          break;
        }}
      }}
      if (position < selected_count) {{
        #pragma unroll
        for (int slot = selected_count - 1; slot > position; --slot) {{
          best_value[slot] = best_value[slot - 1];
          best_index[slot] = best_index[slot - 1];
        }}
        best_value[position] = value;
        best_index[position] = candidate;
      }}
    }}
    #pragma unroll
    for (int slot = 0; slot < selected_count; ++slot) {{
      const int output = (query * {group_extent} + group) * selected_count + slot;
      {value_store}
      argument_{ordinals[selection.indices]}[output] = best_index[slot];
    }}
  }}
  __syncthreads();
  if (warp == 0) {{
    TMEM::Allocator1Sm allocator;
    allocator.free(tensor_memory_base, 128);
  }}
}}
"""
    arguments = tuple(
        CutePipelineSlotArgument(
            name=f"argument_{ordinals[value]}",
            slot=slots[ordinals[value]].name,
            access=CutePipelineSlotAccess.WRITE if value in outputs else CutePipelineSlotAccess.READ,
        )
        for value in (*external_inputs, *outputs)
    )
    launch = CuteKernelLaunch(
        name="refined_contract_fold_selection",
        kernel_symbol=symbol,
        grid=(query_extent // 64, group_extent, 1),
        block=(threads, 1, 1),
        dynamic_shared_memory_bytes=0,
        function_attributes=(),
        arguments=arguments,
        source_tasks=component.task_ids,
    )
    fragment = CuteKernelComponentFragment(component.id, component.schedule_fingerprint, slots, (launch,))
    return GeneratedSm100SimtComponent(
        component.id,
        source,
        hashlib.sha256(source.encode()).hexdigest(),
        fragment,
    )


class _SymbolicComponentExpression:
    """Render one scalar from a dependency-closed Map/Relation component."""

    def __init__(
        self,
        operations: tuple[TensorPrimitive, ...],
        ordinals: dict[ProgramValue, int],
        placeholders: Mapping[ProgramValue, str],
    ) -> None:
        self.produced_by = {value: operation for operation in operations for value in primitive_outputs(operation)}
        self.ordinals = ordinals
        self.placeholders = dict(placeholders)
        self.seen_coordinates: dict[ProgramValue, dict[TensorAxis, str]] = {}

    def value(self, value: ProgramValue, coordinates: dict[TensorAxis, str]) -> str:
        if value in self.placeholders:
            self.seen_coordinates[value] = dict(coordinates)
            return self.placeholders[value]
        operation = self.produced_by.get(value)
        if operation is None:
            return _load(value, _axis_offset(value.axes, coordinates), self.ordinals)
        if isinstance(operation, MapPrimitive):
            bindings = {
                item.name: self.value(item, _positional_coordinates(item.axes, operation.output.axes, coordinates))
                for item in operation.inputs
            }
            expression = render_cuda_scalar_expression(operation.expression, bindings)
            if operation.output.dtype is DType.INT32:
                expression = f"static_cast<int32_t>({expression})"
            elif operation.output.dtype is DType.BOOL:
                expression = f"static_cast<bool>({expression})"
            if operation.predicate is None:
                return expression
            predicate = self.predicate(operation.predicate, coordinates)
            return f"(({predicate}) ? ({expression}) : ({_cuda_literal(operation.rejected_value)}))"
        if isinstance(operation, RelationPrimitive):
            output_index = operation.outputs.index(value)
            gathered: dict[ProgramValue, str] = {}
            allowed = self.predicate(operation.predicate, coordinates)
            for indexing in operation.input_indexing:
                operand_coordinates = {
                    axis: self.index(expression, coordinates)
                    for axis, expression in zip(indexing.operand.axes, indexing.indices, strict=True)
                }
                loaded = self.value(indexing.operand, operand_coordinates)
                indexing_allowed = self.predicate(indexing.predicate, coordinates)
                rejected = f"static_cast<{_cuda_type(indexing.operand.dtype)}>(0)"
                gathered[indexing.operand] = f"(({allowed}) && ({indexing_allowed}) ? ({loaded}) : ({rejected}))"
            expression = operation.resolved_result_expressions[output_index]
            source = operation.resolved_result_sources[output_index]
            if expression is None:
                if source is None:
                    raise ValueError("refined SM100 Relation result has neither expression nor source")
                return gathered[source]
            if isinstance(expression, IndexExpression):
                result = self.index(expression, coordinates)
            else:
                result = self.predicate(expression, coordinates)
            rejected = f"static_cast<{_cuda_type(value.dtype)}>(0)"
            return f"({allowed}) ? ({result}) : ({rejected})"
        raise ValueError(
            "refined SM100 scalar dependency crosses an unsupported internal " f"{type(operation).__name__} result"
        )

    def index(self, expression: IndexExpression, coordinates: dict[TensorAxis, str]) -> str:
        if expression.kind is IndexExpressionKind.AXIS:
            if expression.axis is None:
                raise ValueError("axis index expression has no axis")
            return coordinates[expression.axis]
        if expression.kind is IndexExpressionKind.CONSTANT:
            if expression.constant is None:
                raise ValueError("constant index expression has no value")
            return str(int(expression.constant))
        if expression.kind is IndexExpressionKind.TENSOR_ELEMENT:
            if expression.tensor is None:
                raise ValueError("tensor-element index expression has no tensor")
            indices = tuple(self.index(item, coordinates) for item in expression.operands)
            return self.value(expression.tensor, dict(zip(expression.tensor.axes, indices, strict=True)))
        if expression.kind is IndexExpressionKind.SELECT:
            if expression.predicate is None:
                raise ValueError("select index expression has no predicate")
            predicate = self.predicate(expression.predicate, coordinates)
            left, right = (self.index(item, coordinates) for item in expression.operands)
            return f"(({predicate}) ? ({left}) : ({right}))"
        left, right = (self.index(item, coordinates) for item in expression.operands)
        if expression.kind is IndexExpressionKind.MINIMUM:
            return f"min(({left}), ({right}))"
        if expression.kind is IndexExpressionKind.MAXIMUM:
            return f"max(({left}), ({right}))"
        symbol = {
            IndexExpressionKind.ADD: "+",
            IndexExpressionKind.SUBTRACT: "-",
            IndexExpressionKind.MULTIPLY: "*",
            IndexExpressionKind.FLOOR_DIVIDE: "/",
            IndexExpressionKind.MODULO: "%",
        }.get(expression.kind)
        if symbol is None:
            raise ValueError(f"unsupported refined SM100 index expression {expression.kind.value}")
        return f"(({left}) {symbol} ({right}))"

    def predicate(self, predicate: IndexPredicate | None, coordinates: dict[TensorAxis, str]) -> str:
        if predicate is None:
            return "true"
        if predicate.kind is IndexPredicateKind.CONSTANT:
            return "true" if predicate.constant else "false"
        if predicate.kind is IndexPredicateKind.IS_TRUE:
            return f"static_cast<bool>({self.index(predicate.expressions[0], coordinates)})"
        if predicate.kind is IndexPredicateKind.NOT:
            return f"!({self.predicate(predicate.predicates[0], coordinates)})"
        if predicate.kind in {IndexPredicateKind.AND, IndexPredicateKind.OR}:
            symbol = "&&" if predicate.kind is IndexPredicateKind.AND else "||"
            left, right = (self.predicate(item, coordinates) for item in predicate.predicates)
            return f"(({left}) {symbol} ({right}))"
        left, right = (self.index(item, coordinates) for item in predicate.expressions)
        symbol = {
            IndexPredicateKind.EQUAL: "==",
            IndexPredicateKind.NOT_EQUAL: "!=",
            IndexPredicateKind.LESS: "<",
            IndexPredicateKind.LESS_EQUAL: "<=",
            IndexPredicateKind.GREATER: ">",
            IndexPredicateKind.GREATER_EQUAL: ">=",
        }[predicate.kind]
        return f"(({left}) {symbol} ({right}))"


def _emit_refined_gathered_normalized_exp_component(
    component: SM100KernelComponent | CanonicalKernelComponent,
    program: TensorProgram,
    ordinals: dict[ProgramValue, int],
    operations: tuple[TensorPrimitive, ...],
    *,
    schedule_parameters: Mapping[str, Any],
    threads: int,
) -> GeneratedSm100SimtComponent:
    """Emit a gathered score/Fold/weighted-Contract component from scalar and index ASTs."""
    contracts = tuple(operation for operation in operations if isinstance(operation, ContractPrimitive))
    folds = tuple(operation for operation in operations if isinstance(operation, FoldPrimitive))
    if len(contracts) != 2 or len(folds) != 2 or threads != 256:
        raise ValueError("refined SM100 gathered normalized-exp requires two Contracts, two Folds, and 256 threads")
    producer, terminal = contracts
    maximums = tuple(fold for fold in folds if fold.reducer is FoldReducer.MAXIMUM)
    summations = tuple(fold for fold in folds if fold.reducer is FoldReducer.SUM)
    if len(maximums) != 1 or len(summations) != 1:
        raise ValueError("refined SM100 gathered normalized-exp requires maximum and sum Folds")
    maximum, summation = maximums[0], summations[0]
    if producer.accumulation_dtype is not DType.FP32 or terminal.accumulation_dtype is not DType.FP32:
        raise ValueError("refined SM100 gathered normalized-exp requires FP32 Contract accumulation")
    if any(value.dtype is not DType.BF16 for contract in contracts for value in contract.inputs):
        raise ValueError("refined SM100 gathered normalized-exp tcgen05 requires BF16 Contract operands")
    if len(producer.reduction_axes) != 1 or producer.reduction_axes[0].extent != 128:
        raise ValueError("refined SM100 gathered normalized-exp requires a 128-wide score Contract")
    if len(terminal.reduction_axes) != 1 or terminal.reduction_axes[0].extent != maximum.reduction_axes[0].extent:
        raise ValueError("refined SM100 gathered normalized-exp Contract and Fold dimensions disagree")
    if summation.reduction_axes != maximum.reduction_axes:
        raise ValueError("refined SM100 gathered normalized-exp maximum and sum dimensions disagree")
    normalized_extent = terminal.reduction_axes[0].extent
    if normalized_extent % 128 or terminal.output.shape[-1] != 128:
        raise ValueError(
            "refined SM100 gathered normalized-exp tcgen05 requires normalized divisible by 128 and value=128"
        )

    produced_by = {value: operation for operation in operations for value in primitive_outputs(operation)}
    internal_outputs = set(produced_by)
    external_inputs = tuple(
        sorted(
            {item for operation in operations for item in primitive_inputs(operation)} - internal_outputs,
            key=ordinals.__getitem__,
        )
    )
    outputs = tuple(value for value in program.outputs if ordinals[value] in component.live_output_value_ids)
    if {ordinals[value] for value in external_inputs} != set(component.live_input_value_ids):
        raise ValueError("refined SM100 gathered normalized-exp live inputs disagree with its algebra closure")
    if len(outputs) != 1:
        raise ValueError("refined SM100 gathered normalized-exp requires one live program output")
    output = outputs[0]

    weight_inputs = tuple(value for value in terminal.inputs if _depends_on(value, summation.output, produced_by))
    if len(weight_inputs) != 1:
        raise ValueError("refined SM100 terminal Contract requires one normalized Fold-dependent operand")
    weight = weight_inputs[0]
    value_operand = next(value for value in terminal.inputs if value != weight)
    value_operand_index = terminal.inputs.index(value_operand)
    value_domain_axes = {
        *value_operand.axes,
        *(mapping.domain_axis for mapping in terminal.index_maps_for_input(value_operand_index)),
    }
    owner_group_axes = tuple(axis for axis in terminal.output.axes[:-1] if axis not in value_domain_axes)
    if len(owner_group_axes) != 1 or not 0 < owner_group_axes[0].extent <= 16:
        raise ValueError("refined SM100 tensor-core normalized-exp requires at most sixteen grouped owner rows")
    owner_group_axis = owner_group_axes[0]
    owner_group_extent = owner_group_axis.extent
    if schedule_parameters.get("owner_rows") != owner_group_extent:
        raise ValueError("refined SM100 tensor-core owner-row schedule disagrees with the Contract index relations")
    if schedule_parameters.get("normalized_tile") != normalized_extent:
        raise ValueError("refined SM100 normalized tile disagrees with the Contract reduction dimension")

    resolver = _SymbolicComponentExpression(
        operations,
        ordinals,
        {
            producer.output: "score_value",
            maximum.output: "row_maximum",
            summation.output: "row_sum",
            terminal.output: "output_accumulator",
        },
    )
    if len(output.axes) != 3 or output.shape[1] != terminal.output.shape[1] * owner_group_extent:
        raise ValueError("refined SM100 grouped owner rows require one exact flattened output axis")
    output_coordinates = {
        output.axes[0]: "output_coordinate_0",
        output.axes[1]: f"(output_coordinate_1 * {owner_group_extent} + owner_group)",
        output.axes[2]: "value_feature",
    }
    output_expression = resolver.value(output, output_coordinates)
    try:
        terminal_coordinates = resolver.seen_coordinates[terminal.output]
    except KeyError as error:
        raise ValueError("refined SM100 finalizer does not depend on the terminal Contract") from error

    reduction_coordinates = {terminal.reduction_axes[0]: "normalized_index"}
    weight_coordinates = _contract_operand_coordinates(terminal, weight, terminal_coordinates, reduction_coordinates)
    value_coordinates = _contract_operand_coordinates(
        terminal, value_operand, terminal_coordinates, reduction_coordinates
    )
    maximum_input_coordinates = {
        **{axis: weight_coordinates[axis] for axis in maximum.output.axes},
        maximum.reduction_axes[0]: "normalized_index",
    }
    normalized_score_expression = resolver.value(maximum.input, maximum_input_coordinates)
    try:
        producer_coordinates = resolver.seen_coordinates[producer.output]
    except KeyError as error:
        raise ValueError("refined SM100 maximum Fold does not depend on the producer Contract") from error
    producer_reduction = {producer.reduction_axes[0]: "score_feature"}
    score_left_coordinates = _contract_operand_coordinates(
        producer, producer.inputs[0], producer_coordinates, producer_reduction
    )
    score_right_coordinates = _contract_operand_coordinates(
        producer, producer.inputs[1], producer_coordinates, producer_reduction
    )
    score_left = resolver.value(producer.inputs[0], score_left_coordinates)
    score_right = resolver.value(producer.inputs[1], score_right_coordinates)

    resolver.placeholders[maximum.input] = (
        "shared_values[owner_group * normalized_wave + normalized_index % normalized_wave]"
    )
    numerator_expression = resolver.value(
        summation.input,
        {
            **{axis: weight_coordinates[axis] for axis in summation.output.axes},
            summation.reduction_axes[0]: "normalized_index",
        },
    )
    # The terminal Contract consumes the sum-Fold input normalized by the Fold
    # output.  Recomputing the score in the output pass only restores the input
    # of the numerator Map; preserve that Map here instead of accidentally
    # feeding the raw mapped score to the terminal Contract.
    resolver.placeholders[summation.input] = f"static_cast<float>({numerator_expression})"
    weight_expression = resolver.value(weight, weight_coordinates)
    value_expression = resolver.value(value_operand, value_coordinates)

    slots = {
        ordinals[value]: CutePipelineSlot(
            name=f"local_value_{ordinals[value]}",
            element_type=_element_type(value.dtype),
            shape=value.shape,
            kind=CutePipelineSlotKind.SCRATCH,
            alignment=16,
        )
        for value in (*external_inputs, output)
    }
    identity = hashlib.sha256(
        json.dumps(
            {"schedule": component.schedule_fingerprint, "operations": component.operations},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    symbol = f"shuttle_sm100_refined_gathered_normalized_exp_{identity}"
    outer_axes = tuple(axis for axis in terminal.output.axes[:-1] if axis != owner_group_axis)
    outer_lines = _coordinate_lines(outer_axes, source="outer_linear").replace("coordinate_", "output_coordinate_")
    output_offset = _axis_offset(output.axes, output_coordinates)
    stored_output = _store(output_expression, output.dtype)
    score_tile_body = f"""  for (int matrix_index = threadIdx.x;
       matrix_index < normalized_wave * score_feature_extent;
       matrix_index += blockDim.x) {{
    const int local_right = matrix_index / score_feature_extent;
    const int score_feature = matrix_index % score_feature_extent;
    const int normalized_index = normalized_wave_base + local_right;
    const int owner_group = 0;
    shared_right(local_right, score_feature) = Element({score_right});
  }}
  cutlass::arch::fence_view_async_shared();
  if (threadIdx.x == 0) {{
    cutlass::arch::ClusterBarrier::init(&tensor_core_barrier, 1);
    cutlass::arch::fence_barrier_init();
  }}
  __syncthreads();
  if (warp == 0) {{
    TiledMma mma;
    auto thread_mma = mma.get_slice(0);
    Tensor partitioned_left = thread_mma.partition_A(shared_left);
    Tensor partitioned_right = thread_mma.partition_B(shared_right);
    Tensor fragment_left = thread_mma.make_fragment_A(partitioned_left);
    Tensor fragment_right = thread_mma.make_fragment_B(partitioned_right);
    Tensor fragment_score = partition_fragment_C(mma, Shape<_64, _128>{{}});
    fragment_score.data() = tensor_memory_score;
    mma.accumulate_ = UMMA::ScaleOut::Zero;
    #pragma unroll
    for (int feature_block = 0; feature_block < size<2>(fragment_left); ++feature_block) {{
      cute::gemm(mma, fragment_left(_, _, feature_block), fragment_right(_, _, feature_block), fragment_score);
      mma.accumulate_ = UMMA::ScaleOut::One;
    }}
    cutlass::arch::umma_arrive(&tensor_core_barrier);
  }}
  cutlass::arch::ClusterBarrier::wait(&tensor_core_barrier, 0);
  if (threadIdx.x < 64) {{
    Tensor tensor_score = partition_fragment_C(TiledMma{{}}, Shape<_64, _128>{{}});
    tensor_score.data() = tensor_memory_score;
    Tensor score_coordinates = make_identity_tensor(Shape<_64, _128>{{}});
    Tensor partitioned_score_coordinates = TiledMma{{}}.get_slice(0).partition_C(score_coordinates);
    auto tiled_load = make_tmem_copy(SM100_TMEM_LOAD_16dp32b32x{{}}, tensor_score);
    auto thread_load = tiled_load.get_slice(threadIdx.x);
    Tensor load_source = thread_load.partition_S(tensor_score);
    Tensor load_coordinates = thread_load.partition_D(partitioned_score_coordinates);
    Tensor loaded_score = make_tensor<float>(shape(load_coordinates));
    copy(tiled_load, load_source, loaded_score);
    #pragma unroll
    for (int item = 0; item < size(loaded_score); ++item) {{
      const int owner_group = int(get<0>(load_coordinates(item)));
      const int normalized_index = normalized_wave_base + int(get<1>(load_coordinates(item)));
      if (owner_group < owner_group_extent) {{
        const float score_value = loaded_score(item);
        const float mapped_score = static_cast<float>({normalized_score_expression});
        shared_values[owner_group * normalized_wave + normalized_index % normalized_wave] = mapped_score;
      }}
    }}
  }}
  __syncthreads();"""
    source = f"""#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
extern "C" __global__ void {symbol}({_parameters(external_inputs, (output,), ordinals)}) {{
  using namespace cute;
  using Element = cutlass::bfloat16_t;
  using Mma = SM100_MMA_F16BF16_SS<
      Element, Element, float, 64, 128, UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Mma{{}}));
  using SmemLayoutLeft = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<Element>{{}}, Shape<_64, _128>{{}}));
  using SmemLayoutRight = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<Element>{{}}, Shape<_128, _128>{{}}));
  constexpr int normalized_extent = {normalized_extent};
  constexpr int matrix_extent = 16;
  constexpr int tensor_warps = 8;
  constexpr int normalized_wave = tensor_warps * matrix_extent;
  constexpr int score_feature_extent = 128;
  constexpr int owner_group_extent = {owner_group_extent};
  extern __shared__ float shared_values[];
  __shared__ float shared_row_maximum[owner_group_extent];
  __shared__ float shared_row_sum[owner_group_extent];
  __shared__ alignas(128) Element shared_left_storage[cosize_v<SmemLayoutLeft>];
  __shared__ alignas(128) Element shared_right_storage[cosize_v<SmemLayoutRight>];
  __shared__ alignas(16) uint64_t tensor_core_barrier;
  __shared__ uint32_t tensor_memory_base;
  Tensor shared_left = make_tensor(make_smem_ptr(shared_left_storage), SmemLayoutLeft{{}});
  Tensor shared_right = make_tensor(make_smem_ptr(shared_right_storage), SmemLayoutRight{{}});
  const int64_t outer_linear = blockIdx.x;
{outer_lines}
  const int value_feature = 0;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (warp == 0) {{
    TMEM::Allocator1Sm allocator;
    allocator.allocate(256, &tensor_memory_base);
    __syncwarp();
  }}
  __syncthreads();
  const uint32_t tensor_memory_score = tensor_memory_base;
  const uint32_t tensor_memory_output = tensor_memory_base + 128;
  for (int matrix_index = threadIdx.x;
       matrix_index < 64 * score_feature_extent;
       matrix_index += blockDim.x) {{
    const int score_row = matrix_index / score_feature_extent;
    const int score_feature = matrix_index % score_feature_extent;
    const int owner_group = score_row;
    const int normalized_index = 0;
    shared_left(score_row, score_feature) = score_row < owner_group_extent
        ? Element({score_left})
        : Element(0.0f);
  }}
  __syncthreads();
  float thread_maximum[2] = {{-INFINITY, -INFINITY}};
  for (int normalized_wave_base = 0;
       normalized_wave_base < normalized_extent;
       normalized_wave_base += normalized_wave) {{
{score_tile_body}
    for (int owner_slot = 0; owner_slot < 2; ++owner_slot) {{
      const int owner_group = warp + owner_slot * tensor_warps;
      if (owner_group < owner_group_extent) {{
        for (int local = lane; local < normalized_wave; local += 32) {{
          const float mapped_score = shared_values[owner_group * normalized_wave + local];
          thread_maximum[owner_slot] =
              (isnan(thread_maximum[owner_slot]) || isnan(mapped_score))
              ? NAN
              : fmaxf(thread_maximum[owner_slot], mapped_score);
        }}
      }}
    }}
    __syncthreads();
  }}
  for (int owner_slot = 0; owner_slot < 2; ++owner_slot) {{
    const int owner_group = warp + owner_slot * tensor_warps;
    for (int offset = 16; offset > 0; offset >>= 1) {{
      const float other = __shfl_down_sync(0xffffffff, thread_maximum[owner_slot], offset);
      thread_maximum[owner_slot] =
          (isnan(thread_maximum[owner_slot]) || isnan(other))
          ? NAN
          : fmaxf(thread_maximum[owner_slot], other);
    }}
    if (lane == 0 && owner_group < owner_group_extent)
      shared_row_maximum[owner_group] = thread_maximum[owner_slot];
  }}
  __syncthreads();
  float thread_sum[2] = {{0.0f, 0.0f}};
  for (int normalized_wave_base = 0;
       normalized_wave_base < normalized_extent;
       normalized_wave_base += normalized_wave) {{
{score_tile_body}
    for (int owner_slot = 0; owner_slot < 2; ++owner_slot) {{
      const int owner_group = warp + owner_slot * tensor_warps;
      if (owner_group < owner_group_extent) {{
        const float row_maximum = shared_row_maximum[owner_group];
        for (int local = lane; local < normalized_wave; local += 32) {{
          const int normalized_index = normalized_wave_base + local;
          const float numerator = static_cast<float>({numerator_expression});
          thread_sum[owner_slot] = __fadd_rn(thread_sum[owner_slot], numerator);
        }}
      }}
    }}
    __syncthreads();
  }}
  for (int owner_slot = 0; owner_slot < 2; ++owner_slot) {{
    const int owner_group = warp + owner_slot * tensor_warps;
    for (int offset = 16; offset > 0; offset >>= 1)
      thread_sum[owner_slot] =
          __fadd_rn(thread_sum[owner_slot], __shfl_down_sync(0xffffffff, thread_sum[owner_slot], offset));
    if (lane == 0 && owner_group < owner_group_extent)
      shared_row_sum[owner_group] = thread_sum[owner_slot];
  }}
  __syncthreads();
  for (int normalized_wave_base = 0;
       normalized_wave_base < normalized_extent;
       normalized_wave_base += normalized_wave) {{
    for (int matrix_index = threadIdx.x;
         matrix_index < 64 * score_feature_extent;
         matrix_index += blockDim.x) {{
      const int score_row = matrix_index / score_feature_extent;
      const int score_feature = matrix_index % score_feature_extent;
      const int owner_group = score_row;
      const int normalized_index = 0;
      shared_left(score_row, score_feature) = score_row < owner_group_extent
          ? Element({score_left})
          : Element(0.0f);
    }}
    __syncthreads();
{score_tile_body}
    for (int matrix_index = threadIdx.x; matrix_index < 64 * normalized_wave; matrix_index += blockDim.x) {{
      const int owner_group = matrix_index / normalized_wave;
      const int normalized_index = normalized_wave_base + matrix_index % normalized_wave;
      const int value_feature = 0;
      if (owner_group < owner_group_extent) {{
        const float row_maximum = shared_row_maximum[owner_group];
        const float row_sum = shared_row_sum[owner_group];
        const float normalized_weight = static_cast<float>({weight_expression});
        shared_left(owner_group, matrix_index % normalized_wave) = Element(normalized_weight);
      }} else {{
        shared_left(owner_group, matrix_index % normalized_wave) = Element(0.0f);
      }}
    }}
    for (int matrix_index = threadIdx.x; matrix_index < normalized_wave * 128; matrix_index += blockDim.x) {{
      const int normalized_index = normalized_wave_base + matrix_index / 128;
      const int value_feature = matrix_index % 128;
      const int owner_group = 0;
      shared_right(value_feature, matrix_index / 128) = Element({value_expression});
    }}
    cutlass::arch::fence_view_async_shared();
    if (threadIdx.x == 0) {{
      cutlass::arch::ClusterBarrier::init(&tensor_core_barrier, 1);
      cutlass::arch::fence_barrier_init();
    }}
    __syncthreads();
    if (warp == 0) {{
      TiledMma mma;
      auto thread_mma = mma.get_slice(0);
      Tensor partitioned_weight = thread_mma.partition_A(shared_left);
      Tensor partitioned_value = thread_mma.partition_B(shared_right);
      Tensor fragment_weight = thread_mma.make_fragment_A(partitioned_weight);
      Tensor fragment_value = thread_mma.make_fragment_B(partitioned_value);
      Tensor fragment_output = partition_fragment_C(mma, Shape<_64, _128>{{}});
      fragment_output.data() = tensor_memory_output;
      mma.accumulate_ = normalized_wave_base == 0 ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
      #pragma unroll
      for (int normalized_block = 0; normalized_block < size<2>(fragment_weight); ++normalized_block) {{
        cute::gemm(
            mma,
            fragment_weight(_, _, normalized_block),
            fragment_value(_, _, normalized_block),
            fragment_output);
        mma.accumulate_ = UMMA::ScaleOut::One;
      }}
      cutlass::arch::umma_arrive(&tensor_core_barrier);
    }}
    cutlass::arch::ClusterBarrier::wait(&tensor_core_barrier, 0);
    __syncthreads();
  }}
  if (threadIdx.x < 64) {{
    Tensor tensor_output = partition_fragment_C(TiledMma{{}}, Shape<_64, _128>{{}});
    tensor_output.data() = tensor_memory_output;
    Tensor output_coordinates = make_identity_tensor(Shape<_64, _128>{{}});
    Tensor partitioned_output_coordinates = TiledMma{{}}.get_slice(0).partition_C(output_coordinates);
    auto tiled_load = make_tmem_copy(SM100_TMEM_LOAD_16dp32b32x{{}}, tensor_output);
    auto thread_load = tiled_load.get_slice(threadIdx.x);
    Tensor load_source = thread_load.partition_S(tensor_output);
    Tensor load_coordinates = thread_load.partition_D(partitioned_output_coordinates);
    Tensor loaded_output = make_tensor<float>(shape(load_coordinates));
    copy(tiled_load, load_source, loaded_output);
    #pragma unroll
    for (int item = 0; item < size(loaded_output); ++item) {{
      const int owner_group = int(get<0>(load_coordinates(item)));
      const int value_feature = int(get<1>(load_coordinates(item)));
      if (owner_group < owner_group_extent) {{
        const float output_accumulator = loaded_output(item);
        argument_{ordinals[output]}[{output_offset}] = {stored_output};
      }}
    }}
  }}
  __syncthreads();
  if (warp == 0) {{
    TMEM::Allocator1Sm allocator;
    allocator.free(tensor_memory_base, 256);
  }}
}}
"""
    arguments = tuple(
        CutePipelineSlotArgument(
            name=f"argument_{ordinals[value]}",
            slot=slots[ordinals[value]].name,
            access=CutePipelineSlotAccess.WRITE if value == output else CutePipelineSlotAccess.READ,
        )
        for value in (*external_inputs, output)
    )
    launch = CuteKernelLaunch(
        name="refined_gathered_normalized_exp_contract",
        kernel_symbol=symbol,
        grid=(math.prod(axis.extent for axis in outer_axes), 1, 1),
        block=(threads, 1, 1),
        dynamic_shared_memory_bytes=owner_group_extent * 128 * 4,
        function_attributes=(
            CuteKernelFunctionAttribute(
                CuteKernelFunctionAttributeKind.MAX_DYNAMIC_SHARED_MEMORY_BYTES,
                owner_group_extent * 128 * 4,
            ),
        ),
        arguments=arguments,
        source_tasks=component.task_ids,
    )
    fragment = CuteKernelComponentFragment(component.id, component.schedule_fingerprint, slots, (launch,))
    return GeneratedSm100SimtComponent(
        component.id,
        source,
        hashlib.sha256(source.encode()).hexdigest(),
        fragment,
    )


def _depends_on(
    value: ProgramValue,
    ancestor: ProgramValue,
    produced_by: Mapping[ProgramValue, TensorPrimitive],
    seen: set[ProgramValue] | None = None,
) -> bool:
    if value == ancestor:
        return True
    if seen is None:
        seen = set()
    if value in seen:
        return False
    operation = produced_by.get(value)
    return operation is not None and any(
        _depends_on(item, ancestor, produced_by, seen | {value}) for item in primitive_inputs(operation)
    )


def _contract_operand_coordinates(
    contract: ContractPrimitive,
    operand: ProgramValue,
    output_coordinates: dict[TensorAxis, str],
    reduction_coordinates: dict[TensorAxis, str],
) -> dict[TensorAxis, str]:
    operand_index = contract.inputs.index(operand)
    coordinates = {**output_coordinates, **reduction_coordinates}
    for mapping in contract.index_maps_for_input(operand_index):
        source = coordinates[mapping.domain_axis]
        mapped = f"(({source}) / {mapping.divisor} + {mapping.offset})"
        if mapping.modulus is not None:
            mapped = f"(({mapped}) % {mapping.modulus})"
        coordinates[mapping.operand_axis] = mapped
    try:
        return {axis: coordinates[axis] for axis in operand.axes}
    except KeyError as error:
        raise ValueError("refined SM100 Contract operand has an unmapped physical axis") from error


def _positional_coordinates(
    input_axes: tuple[TensorAxis, ...],
    output_axes: tuple[TensorAxis, ...],
    output_coordinates: dict[TensorAxis, str],
) -> dict[TensorAxis, str]:
    if all(axis in output_coordinates for axis in input_axes):
        return {axis: output_coordinates[axis] for axis in input_axes}
    if len(input_axes) != len(output_axes):
        raise ValueError("refined SM100 scalar Map changes rank without an index Relation")
    return {
        input_axis: output_coordinates[output_axis]
        for input_axis, output_axis in zip(input_axes, output_axes, strict=True)
    }


def render_canonical_simt_kernel(
    symbol: str,
    operation: TensorPrimitive,
    ordinals: dict[ProgramValue, int],
) -> str:
    """Render one generic scalar, index, selection, Fold, or Contract kernel."""
    if isinstance(operation, MapPrimitive):
        return _render_map(symbol, operation, ordinals)
    if isinstance(operation, RelationPrimitive):
        return _render_relation(symbol, operation, ordinals)
    if isinstance(operation, ContractPrimitive):
        return _render_contract(symbol, operation, ordinals)
    if isinstance(operation, FoldPrimitive):
        return _render_fold(symbol, operation, ordinals)
    if isinstance(operation, SelectionPrimitive):
        return _render_selection(symbol, operation, ordinals)
    if isinstance(operation, TransportPrimitive):
        return _render_transport(symbol, operation, ordinals)
    raise TypeError(f"unsupported canonical SIMT operation {type(operation).__name__}")


def _operation_work_items(operation: TensorPrimitive) -> int:
    if isinstance(operation, SelectionPrimitive):
        return math.prod(axis.extent for axis in operation.score.axes if axis != operation.dimension)
    if isinstance(operation, TransportPrimitive):
        return math.prod(axis.extent for axis in operation.domain_axes)
    outputs = primitive_outputs(operation)
    return math.prod(outputs[0].shape)


def _parameters(
    inputs: tuple[ProgramValue, ...], outputs: tuple[ProgramValue, ...], ordinals: dict[ProgramValue, int]
) -> str:
    parameters = [f"const {_cuda_type(value.dtype)}* argument_{ordinals[value]}" for value in inputs]
    parameters.extend(f"{_cuda_type(value.dtype)}* argument_{ordinals[value]}" for value in outputs)
    return ", ".join(parameters)


def _render_map(symbol: str, operation: MapPrimitive, ordinals: dict[ProgramValue, int]) -> str:
    coordinates = _coordinate_lines(operation.output.axes)
    bindings = []
    for value in operation.inputs:
        offset = _axis_offset(
            value.axes, {axis: f"coordinate_{index}" for index, axis in enumerate(operation.output.axes)}
        )
        bindings.append(f"  const auto scalar_{ordinals[value]} = {_load(value, offset, ordinals)};")
    expression = render_cuda_scalar_expression(
        operation.expression, {value.name: f"scalar_{ordinals[value]}" for value in operation.inputs}
    )
    return f"""extern "C" __global__ void {symbol}({_parameters(operation.inputs, (operation.output,), ordinals)}) {{
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= {math.prod(operation.output.shape)}) return;
{coordinates}
{chr(10).join(bindings)}
  argument_{ordinals[operation.output]}[linear] = {_store(expression, operation.output.dtype)};
}}"""


def _render_relation(symbol: str, operation: RelationPrimitive, ordinals: dict[ProgramValue, int]) -> str:
    if len(operation.outputs) != 1:
        raise ValueError("the SM100 SIMT fallback currently requires one Relation result")
    output = operation.outputs[0]
    coordinates_by_axis = {axis: f"coordinate_{index}" for index, axis in enumerate(operation.domain_axes)}
    gathered: dict[ProgramValue, str] = {}
    bindings = []
    operation_predicate = _index_predicate(operation.predicate, coordinates_by_axis, ordinals)
    for indexing in operation.input_indexing:
        offset = _indexing_offset(indexing, coordinates_by_axis, ordinals)
        name = f"scalar_{ordinals[indexing.operand]}"
        predicate = _index_predicate(indexing.predicate, coordinates_by_axis, ordinals)
        predicate = f"({operation_predicate}) && ({predicate})"
        value = _load(indexing.operand, offset, ordinals)
        bindings.append(f"  const auto {name} = ({predicate}) ? ({value}) : {_store('0', indexing.operand.dtype)};")
        gathered[indexing.operand] = name
    expression = operation.resolved_result_expressions[0]
    source = operation.resolved_result_sources[0]
    if expression is None:
        assert source is not None
        result = gathered[source]
    elif isinstance(expression, IndexExpression):
        result = _index_expression(expression, coordinates_by_axis, ordinals)
    else:
        result = _index_predicate(expression, coordinates_by_axis, ordinals)
    allowed = operation_predicate
    return f"""extern "C" __global__ void {symbol}({_parameters(operation.inputs, operation.outputs, ordinals)}) {{
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= {math.prod(output.shape)}) return;
{_coordinate_lines(operation.domain_axes)}
{chr(10).join(bindings)}
  argument_{ordinals[output]}[linear] = ({allowed}) ? {_store(result, output.dtype)} : {_store("0", output.dtype)};
}}"""


def _render_transport(symbol: str, operation: TransportPrimitive, ordinals: dict[ProgramValue, int]) -> str:
    _require_permutation_transport_destination(operation)
    coordinates = {axis: f"coordinate_{index}" for index, axis in enumerate(operation.domain_axes)}
    source_offset = _indexing_offset(operation.source_indexing, coordinates, ordinals)
    destination_offset = _indexing_offset(operation.destination_indexing, coordinates, ordinals)
    predicates = (
        operation.predicate,
        operation.source_indexing.predicate,
        operation.destination_indexing.predicate,
    )
    allowed = " && ".join(f"({_index_predicate(predicate, coordinates, ordinals)})" for predicate in predicates)
    value = f"({allowed}) ? ({_load(operation.source, source_offset, ordinals)}) : (0)"
    return f"""extern "C" __global__ void {symbol}({_parameters(operation.inputs, operation.outputs, ordinals)}) {{
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= {math.prod(axis.extent for axis in operation.domain_axes)}) return;
{_coordinate_lines(operation.domain_axes)}
  argument_{ordinals[operation.destination]}[{destination_offset}] = {_store(value, operation.destination.dtype)};
}}"""


def _require_permutation_transport_destination(operation: TransportPrimitive) -> None:
    """Prove a static one-writer destination map before emitting plain stores."""
    mapped_axes = []
    for destination_axis, expression in zip(
        operation.destination.axes,
        operation.destination_indexing.indices,
        strict=True,
    ):
        if expression.kind is not IndexExpressionKind.AXIS or expression.axis is None:
            raise ValueError(
                "SM100 SIMT Transport destination is not proven injective; "
                "plain-store lowering requires an axis permutation"
            )
        if expression.axis.extent != destination_axis.extent:
            raise ValueError("SM100 SIMT Transport destination axis extents do not preserve a bijection")
        mapped_axes.append(expression.axis)
    if len(mapped_axes) != len(operation.domain_axes) or set(mapped_axes) != set(operation.domain_axes):
        raise ValueError(
            "SM100 SIMT Transport destination is not proven injective; "
            "every transport-domain axis must map to one destination axis"
        )


def _render_contract(symbol: str, operation: ContractPrimitive, ordinals: dict[ProgramValue, int]) -> str:
    if len(operation.inputs) != 2 or operation.accumulation_dtype is not DType.FP32:
        raise ValueError("the SM100 SIMT fallback currently requires binary Contract with FP32 accumulation")
    output_coordinates = {axis: f"coordinate_{index}" for index, axis in enumerate(operation.output.axes)}
    reduction_size = math.prod(axis.extent for axis in operation.reduction_axes)
    reduction_lines = _coordinate_lines(operation.reduction_axes, source="reduction_linear", indent="    ")
    reduction_coordinates = {
        axis: f"reduction_coordinate_{index}" for index, axis in enumerate(operation.reduction_axes)
    }
    loads = []
    for input_index, value in enumerate(operation.inputs):
        coordinate_by_axis = {**output_coordinates, **reduction_coordinates}
        for mapping in operation.index_maps_for_input(input_index):
            source_coordinate = coordinate_by_axis[mapping.domain_axis]
            mapped = f"(({source_coordinate}) / {mapping.divisor} + {mapping.offset})"
            if mapping.modulus is not None:
                mapped = f"(({mapped}) % {mapping.modulus})"
            coordinate_by_axis[mapping.operand_axis] = mapped
        loads.append(_load(value, _axis_offset(value.axes, coordinate_by_axis), ordinals))
    return f"""extern "C" __global__ void {symbol}({_parameters(operation.inputs, (operation.output,), ordinals)}) {{
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= {math.prod(operation.output.shape)}) return;
{_coordinate_lines(operation.output.axes)}
  float accumulator = 0.0f;
  for (int64_t reduction_linear = 0; reduction_linear < {reduction_size}; ++reduction_linear) {{
{reduction_lines}
    accumulator = __fadd_rn(
        accumulator,
        __fmul_rn(static_cast<float>({loads[0]}), static_cast<float>({loads[1]})));
  }}
  argument_{ordinals[operation.output]}[linear] = {_store("accumulator", operation.output.dtype)};
}}"""


def _render_fold(symbol: str, operation: FoldPrimitive, ordinals: dict[ProgramValue, int]) -> str:
    if operation.accumulation_dtype not in {DType.BF16, DType.FP32}:
        raise ValueError(
            "the canonical SIMT fallback supports BF16 or FP32 Fold accumulation, "
            f"got {operation.accumulation_dtype.value}"
        )
    output_coordinates = {axis: f"coordinate_{index}" for index, axis in enumerate(operation.output.axes)}
    reduction_size = math.prod(axis.extent for axis in operation.reduction_axes)
    reduction_lines = _coordinate_lines(operation.reduction_axes, source="reduction_linear", indent="    ")
    reduction_coordinates = {
        axis: f"reduction_coordinate_{index}" for index, axis in enumerate(operation.reduction_axes)
    }
    offset = _axis_offset(operation.input.axes, {**output_coordinates, **reduction_coordinates})
    loaded = _load(operation.input, offset, ordinals)
    accumulator_type = _cuda_type(operation.accumulation_dtype)
    if operation.reducer is FoldReducer.SUM:
        initial = _store("0.0f", operation.accumulation_dtype)
        update_expression = f"static_cast<float>(accumulator) + static_cast<float>({loaded})"
        update = f"accumulator = {_store(update_expression, operation.accumulation_dtype)};"
    elif operation.reducer is FoldReducer.MAXIMUM:
        initial = _store("-INFINITY", operation.accumulation_dtype)
        maximum = "(isnan(accumulated) || isnan(value)) ? NAN : fmaxf(accumulated, value)"
        update = (
            f"const float value = static_cast<float>({loaded}); "
            "const float accumulated = static_cast<float>(accumulator); "
            f"accumulator = {_store(maximum, operation.accumulation_dtype)};"
        )
    else:
        raise ValueError(f"unsupported SM100 SIMT Fold reducer {operation.reducer.value}")
    return f"""extern "C" __global__ void {symbol}({_parameters((operation.input,), (operation.output,), ordinals)}) {{
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= {math.prod(operation.output.shape)}) return;
{_coordinate_lines(operation.output.axes)}
  {accumulator_type} accumulator = {initial};
  for (int64_t reduction_linear = 0; reduction_linear < {reduction_size}; ++reduction_linear) {{
{reduction_lines}
    {update}
  }}
  argument_{ordinals[operation.output]}[linear] = {_store("accumulator", operation.output.dtype)};
}}"""


def _render_selection(symbol: str, operation: SelectionPrimitive, ordinals: dict[ProgramValue, int]) -> str:
    if operation.comparison is not SelectionComparison.TOTAL_ORDER:
        raise ValueError("the SM100 SIMT fallback requires total-order Selection comparison")
    if len(operation.outputs) not in {2, 3}:
        raise ValueError("the SM100 SIMT fallback requires Selection value/index[/validity] results")
    outer_axes = tuple(axis for axis in operation.score.axes if axis != operation.dimension)
    outer_coordinates = {axis: f"coordinate_{index}" for index, axis in enumerate(outer_axes)}
    selected_position = operation.score.axes.index(operation.dimension)
    source_coordinates = dict(outer_coordinates)
    source_coordinates[operation.dimension] = "candidate"
    score_offset = _axis_offset(operation.score.axes, source_coordinates)
    validity = "true"
    if operation.validity is not None:
        validity = f"argument_{ordinals[operation.validity]}[{score_offset}]"
    comparison = ">" if operation.order is SelectionOrder.DESCENDING else "<"
    output_coordinates = dict(outer_coordinates)
    selected_axis = operation.values.axes[selected_position]
    output_coordinates[selected_axis] = "slot"
    output_offset = _axis_offset(operation.values.axes, output_coordinates)
    output_validity = ""
    if operation.output_validity is not None:
        output_validity = (
            f"\n    argument_{ordinals[operation.output_validity]}[{output_offset}] = selected_valid[slot];"
        )
    return f"""extern "C" __global__ void {symbol}({_parameters(operation.inputs, operation.outputs, ordinals)}) {{
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= {math.prod(axis.extent for axis in outer_axes)}) return;
{_coordinate_lines(outer_axes)}
  float selected_value[{operation.count}];
  int32_t selected_index[{operation.count}];
  bool selected_valid[{operation.count}];
  #pragma unroll
  for (int slot = 0; slot < {operation.count}; ++slot) {{
    selected_value[slot] = {_cuda_literal(operation.invalid_value)};
    selected_index[slot] = {operation.invalid_index};
    selected_valid[slot] = false;
  }}
  for (int candidate = 0; candidate < {operation.dimension.extent}; ++candidate) {{
    if (!({validity})) continue;
    const float candidate_value = static_cast<float>({_load(operation.score, score_offset, ordinals)});
    const uint32_t candidate_key = shuttle_total_order_key(candidate_value);
    int insertion = {operation.count};
    #pragma unroll
    for (int slot = 0; slot < {operation.count}; ++slot) {{
      const uint32_t selected_key = shuttle_total_order_key(selected_value[slot]);
      if (!selected_valid[slot] || candidate_key {comparison} selected_key ||
          (candidate_key == selected_key && candidate < selected_index[slot])) {{
        insertion = slot;
        break;
      }}
    }}
    if (insertion == {operation.count}) continue;
    #pragma unroll
    for (int slot = {operation.count - 1}; slot > insertion; --slot) {{
      selected_value[slot] = selected_value[slot - 1];
      selected_index[slot] = selected_index[slot - 1];
      selected_valid[slot] = selected_valid[slot - 1];
    }}
    selected_value[insertion] = candidate_value;
    selected_index[insertion] = candidate;
    selected_valid[insertion] = true;
  }}
  #pragma unroll
  for (int slot = 0; slot < {operation.count}; ++slot) {{
    argument_{ordinals[operation.values]}[{output_offset}] = {_store("selected_value[slot]", operation.values.dtype)};
    argument_{ordinals[operation.indices]}[{output_offset}] = selected_index[slot];{output_validity}
  }}
}}"""


def _coordinate_lines(axes: tuple[TensorAxis, ...], *, source: str = "linear", indent: str = "  ") -> str:
    lines = []
    divisor = math.prod(axis.extent for axis in axes)
    for index, axis in enumerate(axes):
        divisor //= axis.extent
        expression = f"({source} / {divisor}) % {axis.extent}" if divisor != 1 else f"{source} % {axis.extent}"
        prefix = "reduction_" if source == "reduction_linear" else ""
        lines.append(f"{indent}const int64_t {prefix}coordinate_{index} = {expression};")
    return "\n".join(lines)


def _axis_offset(axes: tuple[TensorAxis, ...], coordinates: dict[TensorAxis, str]) -> str:
    terms = []
    stride = 1
    for axis in reversed(axes):
        terms.append(f"({coordinates[axis]}) * {stride}")
        stride *= axis.extent
    return " + ".join(reversed(terms)) or "0"


def _indexing_offset(
    indexing: OperandIndexing,
    coordinates: dict[TensorAxis, str],
    ordinals: dict[ProgramValue, int],
) -> str:
    mapped = tuple(_index_expression(expression, coordinates, ordinals) for expression in indexing.indices)
    return _axis_offset(indexing.operand.axes, dict(zip(indexing.operand.axes, mapped, strict=True)))


def _index_expression(
    expression: IndexExpression,
    coordinates: dict[TensorAxis, str],
    ordinals: dict[ProgramValue, int],
) -> str:
    if expression.kind is IndexExpressionKind.AXIS:
        assert expression.axis is not None
        return coordinates[expression.axis]
    if expression.kind is IndexExpressionKind.CONSTANT:
        assert expression.constant is not None
        return str(int(expression.constant))
    if expression.kind is IndexExpressionKind.TENSOR_ELEMENT:
        assert expression.tensor is not None
        indices = tuple(_index_expression(item, coordinates, ordinals) for item in expression.operands)
        offset = _axis_offset(expression.tensor.axes, dict(zip(expression.tensor.axes, indices, strict=True)))
        return _load(expression.tensor, offset, ordinals)
    if expression.kind is IndexExpressionKind.SELECT:
        assert expression.predicate is not None
        predicate = _index_predicate(expression.predicate, coordinates, ordinals)
        left, right = (_index_expression(item, coordinates, ordinals) for item in expression.operands)
        return f"(({predicate}) ? ({left}) : ({right}))"
    left, right = (_index_expression(item, coordinates, ordinals) for item in expression.operands)
    if expression.kind is IndexExpressionKind.MINIMUM:
        return f"min(({left}), ({right}))"
    if expression.kind is IndexExpressionKind.MAXIMUM:
        return f"max(({left}), ({right}))"
    symbol = {
        IndexExpressionKind.ADD: "+",
        IndexExpressionKind.SUBTRACT: "-",
        IndexExpressionKind.MULTIPLY: "*",
        IndexExpressionKind.FLOOR_DIVIDE: "/",
        IndexExpressionKind.MODULO: "%",
    }.get(expression.kind)
    if symbol is None:
        raise ValueError(f"unsupported SM100 SIMT index expression {expression.kind.value}")
    return f"(({left}) {symbol} ({right}))"


def _index_predicate(
    predicate: IndexPredicate | None,
    coordinates: dict[TensorAxis, str],
    ordinals: dict[ProgramValue, int],
) -> str:
    if predicate is None:
        return "true"
    if predicate.kind is IndexPredicateKind.CONSTANT:
        return "true" if predicate.constant else "false"
    if predicate.kind is IndexPredicateKind.IS_TRUE:
        return f"static_cast<bool>({_index_expression(predicate.expressions[0], coordinates, ordinals)})"
    if predicate.kind is IndexPredicateKind.NOT:
        return f"!({_index_predicate(predicate.predicates[0], coordinates, ordinals)})"
    if predicate.kind in {IndexPredicateKind.AND, IndexPredicateKind.OR}:
        symbol = "&&" if predicate.kind is IndexPredicateKind.AND else "||"
        left, right = (_index_predicate(item, coordinates, ordinals) for item in predicate.predicates)
        return f"(({left}) {symbol} ({right}))"
    left, right = (_index_expression(item, coordinates, ordinals) for item in predicate.expressions)
    symbol = {
        IndexPredicateKind.EQUAL: "==",
        IndexPredicateKind.NOT_EQUAL: "!=",
        IndexPredicateKind.LESS: "<",
        IndexPredicateKind.LESS_EQUAL: "<=",
        IndexPredicateKind.GREATER: ">",
        IndexPredicateKind.GREATER_EQUAL: ">=",
    }[predicate.kind]
    return f"(({left}) {symbol} ({right}))"


def _load(value: ProgramValue, offset: str, ordinals: dict[ProgramValue, int]) -> str:
    return f"argument_{ordinals[value]}[{offset}]"


def _store(expression: str, dtype: DType) -> str:
    if dtype is DType.BF16:
        return f"__float2bfloat16_rn(static_cast<float>({expression}))"
    return f"static_cast<{_cuda_type(dtype)}>({expression})"


def _cuda_literal(value: float | int | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if math.isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    if math.isnan(value):
        raise ValueError("SM100 SIMT literals cannot be NaN")
    rendered = repr(value)
    if "." not in rendered and "e" not in rendered:
        rendered += ".0"
    return f"{rendered}f"


def _cuda_type(dtype: DType) -> str:
    return {
        DType.BOOL: "bool",
        DType.INT32: "int32_t",
        DType.BF16: "__nv_bfloat16",
        DType.FP32: "float",
        DType.FP64: "double",
    }[dtype]


def _element_type(dtype: DType) -> CutePipelineElementType:
    return {
        DType.BOOL: CutePipelineElementType.PRED,
        DType.INT32: CutePipelineElementType.I32,
        DType.BF16: CutePipelineElementType.BF16,
        DType.FP32: CutePipelineElementType.F32,
        DType.FP64: CutePipelineElementType.F64,
    }[dtype]
