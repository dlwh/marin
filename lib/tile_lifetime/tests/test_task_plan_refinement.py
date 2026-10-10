# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from shuttle.ir import DType
from tile_lifetime.backends.h100.triton_ordered_state import (
    generate_cuda_ordered_state_artifact,
    ordered_state_scan_component,
)
from tile_lifetime.dense_flow_planner import (
    derive_composed_task_plan,
    eventize_composed_task_plan,
    physical_request_for_composed_tasks,
)
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventDomain,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    IndexPair,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    coarsen_event_tensor_plan,
    derive_event_tensor_plan,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.msa_frontend import MSADebugConfig
from tile_lifetime.msa_reference_pipeline import build_msa_reference_oracle, derive_msa_reference_dataflow
from tile_lifetime.native_task_plan_adapter import (
    canonical_task_event_plan_from_native_request,
    executable_physical_request_from_native_request,
    executable_scan_physical_request_from_native_request,
    refine_native_scan_task_event_plan,
)
from tile_lifetime.physical_artifact_compiler import decode_gpu_physical_artifact_request
from tile_lifetime.physical_task_contract import (
    CanonicalTaskPlan,
    PhysicalNumericalPolicy,
    PhysicalPhase,
    PhysicalPhaseEdge,
    PhysicalTarget,
    PhysicalTaskStructure,
    RelationAccessClass,
    TaskAxisTile,
    TaskCapability,
    TaskOperationKind,
    TaskResourceEstimate,
    TaskTraversal,
    TaskValueAccessKind,
    TiledTask,
    canonical_fingerprint,
    instantiate_parametric_event_tensor,
)
from tile_lifetime.stateful_scan_reference import NaturalAffineScanConfig, natural_affine_scan_program
from tile_lifetime.stateful_scan_task_pipeline import refine_whole_program_scan_tasks
from tile_lifetime.task_plan_refinement import (
    CanonicalTaskPlanRefinement,
    canonical_task_event_boundary_fingerprint,
    identity_task_refinement_binding,
)
from tile_lifetime.tensor_program import (
    ContractOperandPrecision,
    ContractPrimitive,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarExpressionKind,
    ScanDirection,
    ScanInputBinding,
    ScanOutputBinding,
    ScanPrimitive,
    ScanStateBinding,
    TensorAxis,
    TensorProgram,
    index_axis,
    index_constant,
    index_tensor_element,
    primitive_inputs,
    primitive_outputs,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
)
from tile_lifetime.tensor_program_rewrites import remove_dead_operations


def _task(
    task_id: int,
    name: str,
    operations: tuple[tuple[int, TaskOperationKind, TaskCapability], ...],
) -> TiledTask:
    family = TaskFamily(name, (TaskAxis(f"{name}.row", 2),), placement="device")
    phases = tuple(
        PhysicalPhase(phase, capability, (ordinal,), (128,), ())
        for phase, (ordinal, _, capability) in enumerate(operations)
    )
    structure = PhysicalTaskStructure(
        principal_kind=operations[0][1],
        phases=phases,
        phase_edges=tuple(
            PhysicalPhaseEdge(phase, phase + 1, RelationAccessClass.IDENTITY) for phase in range(len(phases) - 1)
        ),
        owner_axes=(TaskAxisTile(0, 128, 64),),
        streamed_axes=(),
        reduction_axes=(),
        contraction_axes=(),
        traversal=TaskTraversal.UNORDERED,
        element_types=("fp32",),
        layout_classes=("logical_row_major",),
        resources=TaskResourceEstimate(32, 0, 0),
    )
    return TiledTask(
        task_id,
        family,
        tuple(operation[0] for operation in operations),
        tuple(operation[1] for operation in operations),
        structure,
        (),
    )


def _dependence(source: TiledTask, target: TiledTask, pairs=None) -> TaskDependence:
    relation_pairs = (
        tuple(IndexPair((row,), (row,)) for row in range(2))
        if pairs is None
        else tuple(IndexPair(source_coordinate, target_coordinate) for source_coordinate, target_coordinate in pairs)
    )
    return TaskDependence(
        TaskRelation(source.family, target.family, relation_pairs),
        MemoryVisibility(EventMemoryScope.DEVICE),
    )


def _refinement(
    *,
    label: str = "plan",
    omit_external_pair: bool = False,
    coarsen_external_event: bool = False,
) -> CanonicalTaskPlanRefinement:
    contract = (0, TaskOperationKind.CONTRACT, TaskCapability.MATRIX_CONTRACT)
    fold = (1, TaskOperationKind.FOLD, TaskCapability.ASSOCIATIVE_FOLD)
    scalar_map = (2, TaskOperationKind.MAP, TaskCapability.SCALAR_MAP)
    coarse_core = _task(0, f"{label}.coarse.core", (contract, fold))
    coarse_output = _task(1, f"{label}.coarse.output", (scalar_map,))
    coarse_dependence = _dependence(coarse_core, coarse_output)
    coarse = CanonicalTaskPlan((coarse_core, coarse_output), (coarse_dependence,))

    refined_contract = _task(0, f"{label}.refined.contract", (contract,))
    refined_fold = _task(1, f"{label}.refined.fold", (fold,))
    refined_output = _task(2, f"{label}.refined.output", (scalar_map,))
    internal = _dependence(refined_contract, refined_fold)
    external_pairs = (((0,), (0,)),) if omit_external_pair else None
    external = _dependence(refined_fold, refined_output, external_pairs)
    refined = CanonicalTaskPlan((refined_contract, refined_fold, refined_output), (internal, external))
    external_event = derive_event_tensor_plan(external, name=f"{label}.refined.ready")
    if coarsen_external_event:
        external_event = coarsen_event_tensor_plan(
            external_event,
            domain=EventDomain(f"{label}.coarsened", (TaskAxis("all", 1),)),
            project=lambda _: (0,),
            name=f"{label}.refined.coarsened",
        )

    bindings = (
        identity_task_refinement_binding(
            0,
            0,
            refined_family=refined_contract.family,
            coarse_family=coarse_core.family,
        ),
        identity_task_refinement_binding(
            1,
            0,
            refined_family=refined_fold.family,
            coarse_family=coarse_core.family,
        ),
        identity_task_refinement_binding(
            2,
            1,
            refined_family=refined_output.family,
            coarse_family=coarse_output.family,
        ),
    )
    return CanonicalTaskPlanRefinement(
        canonical_fingerprint({"operations": ("contract", "fold", "map")}),
        PhysicalTarget.SM90,
        PhysicalNumericalPolicy.FAST,
        coarse,
        refined,
        bindings,
        coarse_events=(derive_event_tensor_plan(coarse_dependence, name=f"{label}.coarse.ready"),),
        refined_events=(
            derive_event_tensor_plan(internal, name=f"{label}.refined.internal"),
            external_event,
        ),
    )


def test_task_event_refinement_has_one_canonical_quotient_identity() -> None:
    refinement = _refinement(label="first")
    renamed = _refinement(label="renamed")

    assert refinement.coarse_plan.fingerprint != refinement.refined_plan.fingerprint
    assert refinement.quotient_record["task_plan"] == refinement.coarse_plan.canonical_record
    assert refinement.quotient_fingerprint == renamed.quotient_fingerprint
    assert len(refinement.proof_record["internalized_dependencies"]) == 1
    assert refinement.proof_record["quotient_fingerprint"] == refinement.quotient_fingerprint


def test_task_event_refinement_rejects_inexact_projected_relation() -> None:
    with pytest.raises(ValueError, match="dependencies do not quotient"):
        _refinement(omit_external_pair=True)


def test_task_refinement_rejects_incomplete_coordinate_cover() -> None:
    refinement = _refinement()
    binding = refinement.bindings[0]
    incomplete = replace(
        binding, coordinate_relation=replace(binding.coordinate_relation, pairs=binding.coordinate_relation.pairs[:1])
    )

    with pytest.raises(ValueError, match="coordinate relation must be total"):
        replace(refinement, bindings=(incomplete, *refinement.bindings[1:]))


def test_task_refinement_rejects_event_coarsening_not_present_in_quotient() -> None:
    with pytest.raises(ValueError, match="Event relations do not quotient"):
        _refinement(coarsen_external_event=True)


def _native_operation(
    name: str,
    *,
    attributes: dict[str, object] | None = None,
    operands: list[int] | None = None,
    results: list[dict[str, object]] | None = None,
    operations: list[dict[str, object]] | None = None,
    arguments: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "attributes": {} if attributes is None else attributes,
        "operands": [] if operands is None else operands,
        "results": [] if results is None else results,
        "regions": (
            []
            if operations is None
            else [[{"arguments": [] if arguments is None else arguments, "operations": operations}]]
        ),
    }


def _source(reference: int) -> dict[str, str]:
    return {"kind": "mlir_attribute", "value": f"#shuttle.source_ref<{reference}>"}


def _native_request_record() -> dict[str, Any]:
    tensor_type = {"kind": "ranked_tensor", "element_type": "f32", "shape": [2, 4]}
    row_type = {"kind": "ranked_tensor", "element_type": "f32", "shape": [2]}
    scalar_type = {"kind": "scalar", "element_type": "f32"}

    def identity_map(source: int, operand: int, result: int) -> dict[str, object]:
        argument = result * 10
        return _native_operation(
            "shuttle.map",
            attributes={"source": _source(source)},
            operands=[operand],
            results=[{"id": result, "type": row_type}],
            operations=[_native_operation("shuttle.yield", operands=[argument])],
            arguments=[{"id": argument, "type": scalar_type}],
        )

    def affine_map(source: int, operand: int, result: int) -> dict[str, object]:
        argument = result * 10
        scale = argument + 1
        scaled = argument + 2
        offset = argument + 3
        value = argument + 4
        return _native_operation(
            "shuttle.map",
            attributes={"source": _source(source)},
            operands=[operand],
            results=[{"id": result, "type": row_type}],
            operations=[
                _native_operation(
                    "arith.constant",
                    attributes={"value": {"kind": "float", "element_type": "f32", "bits": "3f000000"}},
                    results=[{"id": scale, "type": scalar_type}],
                ),
                _native_operation(
                    "arith.mulf", operands=[argument, scale], results=[{"id": scaled, "type": scalar_type}]
                ),
                _native_operation(
                    "arith.constant",
                    attributes={"value": {"kind": "float", "element_type": "f32", "bits": "3f800000"}},
                    results=[{"id": offset, "type": scalar_type}],
                ),
                _native_operation("arith.addf", operands=[scaled, offset], results=[{"id": value, "type": scalar_type}]),
                _native_operation("shuttle.yield", operands=[value]),
            ],
            arguments=[{"id": argument, "type": scalar_type}],
        )

    algebra_operations = [
        _native_operation(
            "shuttle.relation",
            attributes={
                "source": _source(0),
                "semantics": {
                    "kind": "mlir_attribute",
                    "value": "#shuttle.relation_semantics<broadcast_in_dim>",
                },
                "parameters": [0],
            },
            operands=[0],
            results=[{"id": 1, "type": tensor_type}],
        ),
        _native_operation(
            "shuttle.fold",
            attributes={"source": _source(1), "reduction_dimensions": [1]},
            operands=[1],
            results=[{"id": 2, "type": row_type}],
            operations=[_native_operation("arith.addf")],
        ),
        affine_map(2, 2, 3),
        identity_map(3, 3, 4),
        _native_operation("shuttle.yield", operands=[4]),
    ]

    def materialization_task(ordinal: int, sources: list[int], dependencies: list[int]) -> dict[str, object]:
        return _native_operation(
            "shuttle.materialization_task",
            attributes={
                "ordinal": ordinal,
                "composition_sources": [_source(source) for source in sources],
                "input_buffers": [],
                "output_buffers": [],
                "dependencies": dependencies,
                "domain_shape": [2, 4, 8],
                "reduction_dimensions": [2],
                "composition_owner_dimensions": [0],
                "online_vector_dimensions": [1],
                "composition_stream_dimensions": [2],
                "composition_producer_contraction_dimensions": [1],
                "composition_consumer_contraction_dimensions": [1],
            },
        )

    materialization_tasks = [materialization_task(0, [0, 1, 2], []), materialization_task(1, [3], [0])]

    def schedule_task(ordinal: int, dependencies: list[int]) -> dict[str, object]:
        return _native_operation(
            "shuttle.schedule_task",
            attributes={
                "ordinal": ordinal,
                "source_task": ordinal,
                "kind": {"kind": "mlir_attribute", "value": "#shuttle.schedule_task_kind<contract>"},
                "domain_shape": [2, 4, 8],
                "grid_shape": [2],
                "tile_shape": [1, 4, 8],
                "serial_tiles": 1,
                "workgroup_threads": 128,
                "subgroup_size": 32,
                "scratch_bytes": 0,
                "input_buffers": [],
                "output_buffers": [],
                "dependencies": dependencies,
                "reduction_axis": 2,
            },
        )

    affine_dimension = {"kind": "dimension", "position": 0}
    affine_zero = {"kind": "constant", "value": 0}
    identity_map = {
        "kind": "affine_map",
        "dimensions": 1,
        "symbols": 0,
        "results": [affine_dimension],
    }
    zero_map = {**identity_map, "results": [affine_zero]}
    event_metadata = {
        "event": 0,
        "producer_task": 0,
        "consumer_task": 1,
        "relation_domain_shape": [2],
        "producer_indices": identity_map,
        "consumer_indices": identity_map,
        "storage_capacity": 2,
        "storage_slot": identity_map,
        "generation_index": zero_map,
        "generations": 1,
        "notify_count": 1,
        "scope": {"kind": "mlir_attribute", "value": "#shuttle.event_scope<device>"},
    }
    event_operations = [
        _native_operation(
            "shuttle.event_realization",
            attributes={
                "event": 0,
                "kind": {
                    "kind": "mlir_attribute",
                    "value": "#shuttle.event_realization_kind<imperative_wait_notify>",
                },
                "covering_launch_order": [],
            },
        ),
        _native_operation("shuttle.imperative_event_notify", attributes={**event_metadata, "release": True}),
        _native_operation("shuttle.imperative_event_wait", attributes={**event_metadata, "acquire": True}),
        _native_operation("shuttle.event_lowering_plan_yield"),
    ]
    request = {
        "schema_version": 3,
        "target": "sm_90",
        "fingerprint": "f" * 64,
        "binding_indices": [],
        "region_result_indices": [],
        "reuse_groups": [],
        "region": {
            **_native_operation("shuttle.region", results=[{"id": 4, "type": row_type}]),
            "regions": [[{"arguments": [{"id": 0, "type": row_type}], "operations": algebra_operations}]],
        },
        "materialization": _native_operation("shuttle.materialization_plan", operations=materialization_tasks),
        "schedule": _native_operation(
            "shuttle.schedule_plan",
            operations=[
                schedule_task(0, []),
                schedule_task(1, [0]),
                _native_operation(
                    "shuttle.event_tensor",
                    attributes={
                        **{key: value for key, value in event_metadata.items() if key != "event"},
                        "ordinal": 0,
                        "release": True,
                        "acquire": True,
                    },
                ),
            ],
        ),
        "event_lowering_plan": _native_operation(
            "shuttle.event_lowering_plan",
            operations=event_operations,
        ),
    }
    return request


def _canonical_algebra_program(*, broadcast_source: bool = True) -> TensorProgram:
    row = TensorAxis(0, 2, "row")
    feature = TensorAxis(1, 4, "feature")
    source_axes = (row,) if broadcast_source else (row, feature)
    source = ProgramValue("source", source_axes, DType.FP32)
    broadcast = ProgramValue("broadcast", (row, feature), DType.FP32)
    reduced = ProgramValue("reduced", (row,), DType.FP32)
    mapped = ProgramValue("mapped", (row,), DType.FP32)
    output = ProgramValue("output", (row,), DType.FP32)
    return TensorProgram(
        (source,),
        (
            RelationPrimitive(
                "broadcast",
                (source,),
                (broadcast,),
                (row, feature),
                (
                    OperandIndexing(
                        source,
                        (row, feature),
                        tuple(index_axis(axis) for axis in source_axes),
                    ),
                ),
                result_sources=(source,),
            ),
            FoldPrimitive(
                "reduce",
                broadcast,
                reduced,
                (feature,),
                FoldReducer.SUM,
                DType.FP32,
                EvaluationOrder.ORDER_FREE,
            ),
            MapPrimitive(
                "affine",
                (reduced,),
                mapped,
                scalar_binary(
                    ScalarExpressionKind.ADD,
                    scalar_binary(
                        ScalarExpressionKind.MULTIPLY,
                        scalar_input(reduced.name),
                        scalar_constant(0.5),
                    ),
                    scalar_constant(1.0),
                ),
            ),
            MapPrimitive("identity.final", (mapped,), output, scalar_input(mapped.name)),
        ),
        (output,),
        NumericalPolicy.FAST,
    )


def _canonical_map_contract_map_program() -> TensorProgram:
    row = TensorAxis(0, 2, "row")
    column = TensorAxis(1, 4, "column")
    reduction = TensorAxis(2, 3, "reduction")
    left = ProgramValue("left", (row, reduction), DType.FP32)
    right = ProgramValue("right", (reduction, column), DType.FP32)
    mapped = ProgramValue("mapped", (row, reduction), DType.FP32)
    contracted = ProgramValue("contracted", (row, column), DType.FP32)
    output = ProgramValue("output", (row, column), DType.FP32)
    return TensorProgram(
        (left, right),
        (
            MapPrimitive(
                "negate.left",
                (left,),
                mapped,
                scalar_unary(ScalarExpressionKind.NEGATE, scalar_input(left.name)),
            ),
            ContractPrimitive("contract", (mapped, right), contracted, (reduction,), DType.FP32),
            MapPrimitive(
                "negate.output",
                (contracted,),
                output,
                scalar_unary(ScalarExpressionKind.NEGATE, scalar_input(contracted.name)),
            ),
        ),
        (mapped, output),
        NumericalPolicy.SOURCE_ORDERED,
    )


def _native_map_contract_map_request_record() -> dict[str, Any]:
    left_type = {"kind": "ranked_tensor", "element_type": "f32", "shape": [2, 3]}
    right_type = {"kind": "ranked_tensor", "element_type": "f32", "shape": [3, 4]}
    output_type = {"kind": "ranked_tensor", "element_type": "f32", "shape": [2, 4]}
    scalar_type = {"kind": "scalar", "element_type": "f32"}

    def dimension(position: int) -> dict[str, object]:
        return {"kind": "dimension", "position": position}

    def affine_map(dimensions: int, *positions: int) -> dict[str, object]:
        return {
            "kind": "affine_map",
            "dimensions": dimensions,
            "symbols": 0,
            "results": [dimension(position) for position in positions],
        }

    def negate(source: int, operand: int, result: int, tensor_type: dict[str, object]) -> dict[str, object]:
        argument = result + 100
        negated = result + 200
        return _native_operation(
            "shuttle.map",
            attributes={"source": _source(source)},
            operands=[operand],
            results=[{"id": result, "type": tensor_type}],
            operations=[
                _native_operation("arith.negf", operands=[argument], results=[{"id": negated, "type": scalar_type}]),
                _native_operation("shuttle.yield", operands=[negated]),
            ],
            arguments=[{"id": argument, "type": scalar_type}],
        )

    algebra_operations = [
        negate(0, 0, 2, left_type),
        _native_operation(
            "shuttle.contract",
            attributes={
                "source": _source(1),
                "indexing_maps": [affine_map(3, 0, 2), affine_map(3, 2, 1), affine_map(3, 0, 1)],
                "accumulator_types": [scalar_type],
            },
            operands=[2, 1],
            results=[{"id": 3, "type": output_type}],
        ),
        negate(2, 3, 4, output_type),
        _native_operation("shuttle.yield", operands=[2, 4]),
    ]
    phases = [
        {
            "source": _source(0),
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.materialization_task_kind<map>"},
            "domain_shape": [2, 3],
            "reduction_dimensions": [],
        },
        {
            "source": _source(1),
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.materialization_task_kind<contract>"},
            "domain_shape": [2, 4, 3],
            "reduction_dimensions": [2],
        },
        {
            "source": _source(2),
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.materialization_task_kind<map>"},
            "domain_shape": [2, 4],
            "reduction_dimensions": [],
        },
    ]
    phase_edges = [
        {
            "source_phase": 0,
            "source_result": 0,
            "target_phase": 1,
            "target_operand": 0,
            "access_map": affine_map(3, 0, 2),
        },
        {
            "source_phase": 1,
            "source_result": 0,
            "target_phase": 2,
            "target_operand": 0,
            "access_map": affine_map(2, 0, 1),
        },
    ]
    materialization_task = _native_operation(
        "shuttle.materialization_task",
        attributes={
            "ordinal": 0,
            "source": _source(1),
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.materialization_task_kind<contract>"},
            "composition_kind": "map_contract",
            "composition_sources": [_source(0), _source(1), _source(2)],
            "composition_phases": phases,
            "composition_phase_edges": phase_edges,
            "composition_result_bindings": [
                {"phase": 0, "source": _source(0), "result": 0, "buffer": 2},
                {"phase": 2, "source": _source(2), "result": 0, "buffer": 3},
            ],
            "input_buffers": [0, 1],
            "output_buffers": [2, 3],
            "dependencies": [],
            "domain_shape": [2, 4, 3],
            "reduction_dimensions": [2],
        },
    )
    schedule_task = _native_operation(
        "shuttle.schedule_task",
        attributes={
            "ordinal": 0,
            "source_task": 0,
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.schedule_task_kind<contract>"},
            "domain_shape": [2, 4, 3],
            "grid_shape": [2],
            "tile_shape": [1, 4, 3],
            "serial_tiles": 1,
            "workgroup_threads": 128,
            "subgroup_size": 32,
            "scratch_bytes": 0,
            "input_buffers": [0, 1],
            "output_buffers": [2, 3],
            "dependencies": [],
            "reduction_axis": 2,
        },
    )
    return {
        "schema_version": 3,
        "target": "sm_90",
        "fingerprint": "c" * 64,
        "binding_indices": [],
        "region_result_indices": [],
        "reuse_groups": [],
        "region": {
            **_native_operation(
                "shuttle.region",
                attributes={"policy": {"kind": "mlir_attribute", "value": "#shuttle.policy<source_ordered>"}},
                results=[{"id": 2, "type": left_type}, {"id": 4, "type": output_type}],
            ),
            "regions": [
                [
                    {
                        "arguments": [{"id": 0, "type": left_type}, {"id": 1, "type": right_type}],
                        "operations": algebra_operations,
                    }
                ]
            ],
        },
        "materialization": _native_operation(
            "shuttle.materialization_plan",
            operations=[materialization_task, _native_operation("shuttle.materialization_plan_yield")],
        ),
        "schedule": _native_operation(
            "shuttle.schedule_plan",
            operations=[schedule_task, _native_operation("shuttle.schedule_plan_yield")],
        ),
        "event_lowering_plan": _native_operation(
            "shuttle.event_lowering_plan",
            operations=[_native_operation("shuttle.event_lowering_plan_yield")],
        ),
    }


def test_native_multi_phase_schedule_matches_executable_physical_fingerprint() -> None:
    program = _canonical_map_contract_map_program()
    request = decode_gpu_physical_artifact_request(json.dumps(_native_map_contract_map_request_record()))

    native = canonical_task_event_plan_from_native_request(request, algebra_program=program)
    task = native.task_plan.tasks[0]
    assert tuple(
        (phase.operation_ordinals, phase.domain_shape, phase.reduction_dimensions) for phase in task.structure.phases
    ) == (
        ((0,), (2, 3), ()),
        ((1,), (2, 4, 3), (2,)),
        ((2,), (2, 4), ()),
    )
    assert task.structure.phase_edges == (
        PhysicalPhaseEdge(0, 1, RelationAccessClass.AFFINE),
        PhysicalPhaseEdge(1, 2, RelationAccessClass.IDENTITY),
    )
    assert tuple(
        (access.phase_id, access.value_id) for access in task.accesses if access.kind is TaskValueAccessKind.LIVE_OUT
    ) == ((0, 2), (2, 4))

    physical = executable_physical_request_from_native_request(request, algebra_program=program)
    reference = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=native.task_plan,
        event_program=EventDataflowProgram(
            native.task_plan.task_families,
            native.task_plan.dependences,
            native.events,
        ),
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
    )
    assert physical.fingerprint == reference.fingerprint
    assert physical.task_plan.fingerprint == reference.task_plan.fingerprint
    assert tuple(schedule.fingerprint for schedule in physical.schedules) == tuple(
        schedule.fingerprint for schedule in reference.schedules
    )
    assert physical.event_plans == physical.event_allocations == physical.event_programs == ()


def _canonical_ordered_scan_program() -> TensorProgram:
    ordered = TensorAxis(0, 4, "ordered")
    feature = TensorAxis(1, 2, "feature")
    initial = ProgramValue("initial", (feature,), DType.FP32)
    sequence = ProgramValue("sequence", (ordered, feature), DType.FP32)
    body_state = ProgramValue("body_state", (feature,), DType.FP32)
    body_element = ProgramValue("body_element", (feature,), DType.FP32)
    next_state = ProgramValue("next_state", (feature,), DType.FP32)
    emitted_element = ProgramValue("emitted_element", (feature,), DType.FP32)
    body = TensorProgram(
        (body_state, body_element),
        (
            MapPrimitive(
                "state_update",
                (body_state, body_element),
                next_state,
                scalar_binary(
                    ScalarExpressionKind.ADD,
                    scalar_input(body_state.name),
                    scalar_input(body_element.name),
                ),
            ),
            MapPrimitive("emit", (next_state,), emitted_element, scalar_input(next_state.name)),
        ),
        (next_state, emitted_element),
        NumericalPolicy.SOURCE_ORDERED,
    )
    final = ProgramValue("final", (feature,), DType.FP32)
    emitted = ProgramValue("emitted", (ordered, feature), DType.FP32)
    indexing = OperandIndexing(sequence, (ordered, feature), (index_axis(ordered), index_axis(feature)))
    emitted_indexing = OperandIndexing(emitted, (ordered, feature), (index_axis(ordered), index_axis(feature)))
    scan = ScanPrimitive(
        "ordered_recurrence",
        ordered,
        body,
        (ScanStateBinding(initial, body_state, next_state, final),),
        (ScanInputBinding(sequence, body_element, indexing),),
        (ScanOutputBinding(emitted_element, emitted, emitted_indexing),),
        direction=ScanDirection.FORWARD,
    )
    return TensorProgram(
        (initial, sequence),
        (scan,),
        (final, emitted),
        NumericalPolicy.SOURCE_ORDERED,
    )


def _native_ordered_scan_request_record() -> dict[str, Any]:
    vector_type = {"kind": "ranked_tensor", "element_type": "f32", "shape": [2]}
    sequence_type = {"kind": "ranked_tensor", "element_type": "f32", "shape": [4, 2]}
    scalar_type = {"kind": "scalar", "element_type": "f32"}

    update = _native_operation(
        "shuttle.map",
        attributes={"source": _source(2)},
        operands=[10, 11],
        results=[{"id": 12, "type": vector_type}],
        operations=[
            _native_operation("arith.addf", operands=[100, 101], results=[{"id": 102, "type": scalar_type}]),
            _native_operation("shuttle.yield", operands=[102]),
        ],
        arguments=[{"id": 100, "type": scalar_type}, {"id": 101, "type": scalar_type}],
    )
    emit = _native_operation(
        "shuttle.map",
        attributes={"source": _source(3)},
        operands=[12],
        results=[{"id": 13, "type": vector_type}],
        operations=[_native_operation("shuttle.yield", operands=[103])],
        arguments=[{"id": 103, "type": scalar_type}],
    )
    scan = _native_operation(
        "shuttle.scan",
        attributes={
            "axis": 0,
            "extent": 4,
            "direction": {"kind": "mlir_attribute", "value": "#shuttle.scan_direction<forward>"},
            "state_operands": [0],
            "state_body_arguments": [0],
            "state_body_results": [0],
            "state_results": [0],
            "invariant_operands": [],
            "invariant_body_arguments": [],
            "scanned_operands": [1],
            "scanned_body_arguments": [1],
            "scanned_dimensions": [0],
            "emitted_body_results": [1],
            "emitted_results": [1],
            "emitted_dimensions": [0],
            "sources": [_source(0), _source(1)],
        },
        operands=[0, 1],
        results=[{"id": 2, "type": vector_type}, {"id": 3, "type": sequence_type}],
        operations=[update, emit, _native_operation("shuttle.yield", operands=[12, 13])],
        arguments=[{"id": 10, "type": vector_type}, {"id": 11, "type": vector_type}],
    )
    materialization_task = _native_operation(
        "shuttle.materialization_task",
        attributes={
            "ordinal": 0,
            "source": _source(0),
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.materialization_task_kind<scan>"},
            "composition_sources": [_source(0), _source(1)],
            "input_buffers": [0, 1],
            "output_buffers": [2, 3],
            "dependencies": [],
            "domain_shape": [1, 4, 2],
            "reduction_dimensions": [],
            "composition_owner_dimensions": [0, 2],
            "online_vector_dimensions": [],
            "composition_stream_dimensions": [1],
            "composition_state_inputs": [0],
            "composition_state_outputs": [2],
            "composition_producer_contraction_dimensions": [],
            "composition_consumer_contraction_dimensions": [],
        },
    )
    schedule_task = _native_operation(
        "shuttle.schedule_task",
        attributes={
            "ordinal": 0,
            "source_task": 0,
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.schedule_task_kind<ordered_state>"},
            "domain_shape": [1, 4, 2],
            "grid_shape": [2],
            "tile_shape": [1, 4, 1],
            "serial_tiles": 1,
            "workgroup_threads": 32,
            "subgroup_size": 32,
            "scratch_bytes": 8,
            "composition_state_inputs": [0],
            "composition_state_outputs": [2],
            "input_buffers": [0, 1],
            "output_buffers": [2, 3],
            "dependencies": [],
            "reduction_axis": 1,
        },
    )
    return {
        "schema_version": 3,
        "target": "sm_90",
        "fingerprint": "e" * 64,
        "binding_indices": [],
        "region_result_indices": [],
        "reuse_groups": [],
        "region": {
            **_native_operation(
                "shuttle.region",
                results=[{"id": 2, "type": vector_type}, {"id": 3, "type": sequence_type}],
            ),
            "regions": [
                [
                    {
                        "arguments": [{"id": 0, "type": vector_type}, {"id": 1, "type": sequence_type}],
                        "operations": [scan, _native_operation("shuttle.yield", operands=[2, 3])],
                    }
                ]
            ],
        },
        "materialization": _native_operation(
            "shuttle.materialization_plan",
            operations=[materialization_task, _native_operation("shuttle.materialization_plan_yield")],
        ),
        "schedule": _native_operation(
            "shuttle.schedule_plan",
            operations=[schedule_task, _native_operation("shuttle.schedule_plan_yield")],
        ),
        "event_lowering_plan": _native_operation(
            "shuttle.event_lowering_plan",
            operations=[_native_operation("shuttle.event_lowering_plan_yield")],
        ),
    }


def test_native_ordered_scan_binds_canonical_nested_algebra_without_workload_identity() -> None:
    program = _canonical_ordered_scan_program()
    request = decode_gpu_physical_artifact_request(json.dumps(_native_ordered_scan_request_record()))

    native = canonical_task_event_plan_from_native_request(request, algebra_program=program)

    assert native.algebra_fingerprint == program.semantic_fingerprint
    assert native.task_plan.tasks[0].operation_kinds == (TaskOperationKind.SCAN,)
    assert native.task_plan.tasks[0].structure.streamed_axes == (TaskAxisTile(1, 4, 4),)
    assert native.events == ()
    assert native.event_programs == ()

    mutated = _native_ordered_scan_request_record()
    mutated["region"]["regions"][0][0]["operations"][0]["attributes"]["direction"] = {
        "kind": "mlir_attribute",
        "value": "#shuttle.scan_direction<reverse>",
    }
    with pytest.raises(ValueError, match="traversal directions disagree"):
        canonical_task_event_plan_from_native_request(
            decode_gpu_physical_artifact_request(json.dumps(mutated)), algebra_program=program
        )


def _native_type(value: ProgramValue) -> dict[str, object]:
    return {"kind": "ranked_tensor", "element_type": value.dtype.value, "shape": list(value.shape)}


def _native_scalar_type(dtype: DType) -> dict[str, object]:
    return {"kind": "scalar", "element_type": dtype.value}


def _native_scan_body(
    program: TensorProgram, source_start: int
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[ProgramValue, int]]:
    value_ids = {value: index + 100 for index, value in enumerate(program.inputs)}
    arguments = [{"id": value_ids[value], "type": _native_type(value)} for value in program.inputs]
    operations: list[dict[str, object]] = []
    next_id = 100 + len(program.inputs)

    def scalar_body(expression, inputs: tuple[ProgramValue, ...], output_dtype: DType):
        nonlocal next_id
        scalar_ids = {value.name: next_id + index for index, value in enumerate(inputs)}
        next_id += len(inputs)
        body_arguments = [{"id": scalar_ids[value.name], "type": _native_scalar_type(value.dtype)} for value in inputs]
        body_operations: list[dict[str, object]] = []

        def emit(node) -> int:
            nonlocal next_id
            if node.kind is ScalarExpressionKind.INPUT:
                return scalar_ids[node.input_name]
            operands = [emit(operand) for operand in node.operands]
            result = next_id
            next_id += 1
            if node.kind is ScalarExpressionKind.CONSTANT:
                bits = struct.pack(">f", float(node.constant)).hex()
                body_operations.append(
                    _native_operation(
                        "arith.constant",
                        attributes={"value": {"kind": "float", "element_type": "f32", "bits": bits}},
                        results=[{"id": result, "type": _native_scalar_type(DType.FP32)}],
                    )
                )
                return result
            name = {
                ScalarExpressionKind.ADD: "arith.addf",
                ScalarExpressionKind.SUBTRACT: "arith.subf",
                ScalarExpressionKind.MULTIPLY: "arith.mulf",
                ScalarExpressionKind.DIVIDE: "arith.divf",
                ScalarExpressionKind.EXP: "math.exp",
                ScalarExpressionKind.NEGATE: "arith.negf",
            }.get(node.kind)
            if name is None:
                raise ValueError(f"test native Scan encoder does not support {node.kind.value}")
            dtype = node.result_dtype or output_dtype
            body_operations.append(
                _native_operation(name, operands=operands, results=[{"id": result, "type": _native_scalar_type(dtype)}])
            )
            return result

        result = emit(expression)
        body_operations.append(_native_operation("shuttle.yield", operands=[result]))
        return body_arguments, body_operations

    for ordinal, operation in enumerate(program.operations):
        outputs = primitive_outputs(operation)
        result_ids = list(range(next_id, next_id + len(outputs)))
        next_id += len(outputs)
        for value, value_id in zip(outputs, result_ids, strict=True):
            value_ids[value] = value_id
        source = _source(source_start + ordinal)
        if isinstance(operation, MapPrimitive):
            body_arguments, body_operations = scalar_body(operation.expression, operation.inputs, operation.output.dtype)
            operations.append(
                _native_operation(
                    "shuttle.map",
                    attributes={"source": source},
                    operands=[value_ids[value] for value in operation.inputs],
                    results=[{"id": result_ids[0], "type": _native_type(operation.output)}],
                    operations=body_operations,
                    arguments=body_arguments,
                )
            )
            continue
        if isinstance(operation, ContractPrimitive):
            aliases = []
            for input_index, value in enumerate(operation.inputs):
                mapping = {item.operand_axis: item.domain_axis for item in operation.index_maps_for_input(input_index)}
                aliases.append(tuple(mapping.get(axis, axis) for axis in value.axes))
            domain_axes = tuple(dict.fromkeys((*operation.output.axes, *operation.reduction_axes)))

            def affine_map(axes, *, iteration_axes=domain_axes):
                return {
                    "kind": "affine_map",
                    "dimensions": len(iteration_axes),
                    "symbols": 0,
                    "results": [{"kind": "dimension", "position": iteration_axes.index(axis)} for axis in axes],
                }

            operations.append(
                _native_operation(
                    "shuttle.contract",
                    attributes={
                        "source": source,
                        "indexing_maps": [
                            *(affine_map(axes) for axes in aliases),
                            affine_map(operation.output.axes),
                        ],
                        "accumulator_types": [_native_scalar_type(operation.accumulation_dtype)],
                    },
                    operands=[value_ids[value] for value in operation.inputs],
                    results=[{"id": result_ids[0], "type": _native_type(operation.output)}],
                )
            )
            continue
        raise ValueError(f"test native Scan encoder does not support {type(operation).__name__}")
    operations.append(_native_operation("shuttle.yield", operands=[value_ids[value] for value in program.outputs]))
    return arguments, operations, value_ids


def _native_natural_scan_request(program: TensorProgram) -> dict[str, Any]:
    scan = program.operations[0]
    assert isinstance(scan, ScanPrimitive)
    body_arguments, body_operations, _ = _native_scan_body(scan.body, 20)
    outer_ids = {value: index for index, value in enumerate(program.inputs)}
    outputs = primitive_outputs(scan)
    result_ids = {value: index + len(program.inputs) for index, value in enumerate(outputs)}
    inputs = primitive_inputs(scan)
    input_position = {value: index for index, value in enumerate(inputs)}
    output_position = {value: index for index, value in enumerate(outputs)}
    body_input_position = {value: index for index, value in enumerate(scan.body.inputs)}
    body_output_position = {value: index for index, value in enumerate(scan.body.outputs)}
    sources = [_source(0), _source(1)]
    native_scan = _native_operation(
        "shuttle.scan",
        attributes={
            "axis": scan.axis.id,
            "extent": scan.axis.extent,
            "direction": {"kind": "mlir_attribute", "value": f"#shuttle.scan_direction<{scan.direction.value}>"},
            "state_operands": [input_position[b.initial] for b in scan.state],
            "state_body_arguments": [body_input_position[b.body_argument] for b in scan.state],
            "state_body_results": [body_output_position[b.body_result] for b in scan.state],
            "state_results": [output_position[b.final] for b in scan.state],
            "invariant_operands": [input_position[b.source] for b in scan.invariants],
            "invariant_body_arguments": [body_input_position[b.body_argument] for b in scan.invariants],
            "scanned_operands": [input_position[b.source] for b in scan.scanned_inputs],
            "scanned_body_arguments": [body_input_position[b.body_argument] for b in scan.scanned_inputs],
            "scanned_dimensions": [b.indexing.domain_axes.index(scan.axis) for b in scan.scanned_inputs],
            "emitted_body_results": [body_output_position[b.body_result] for b in scan.emitted_outputs],
            "emitted_results": [output_position[b.destination] for b in scan.emitted_outputs],
            "emitted_dimensions": [b.indexing.domain_axes.index(scan.axis) for b in scan.emitted_outputs],
            "sources": sources,
        },
        operands=[outer_ids[value] for value in inputs],
        results=[{"id": result_ids[value], "type": _native_type(value)} for value in outputs],
        operations=body_operations,
        arguments=body_arguments,
    )
    domain = [1, scan.axis.extent, 1, 2, 3, 1]
    materialization = _native_operation(
        "shuttle.materialization_task",
        attributes={
            "ordinal": 0,
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.materialization_task_kind<scan>"},
            "composition_sources": sources,
            "input_buffers": list(range(len(program.inputs))),
            "output_buffers": [result_ids[value] for value in outputs],
            "dependencies": [],
            "domain_shape": domain,
            "reduction_dimensions": [],
            "composition_owner_dimensions": [0, 2, 4],
            "online_vector_dimensions": [3, 5],
            "composition_stream_dimensions": [1],
            "composition_state_inputs": [outer_ids[scan.state[0].initial]],
            "composition_state_outputs": [result_ids[scan.state[0].final]],
            "composition_producer_contraction_dimensions": [3, 5],
            "composition_consumer_contraction_dimensions": [3],
        },
    )
    schedule = _native_operation(
        "shuttle.schedule_task",
        attributes={
            "ordinal": 0,
            "source_task": 0,
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.schedule_task_kind<ordered_state>"},
            "domain_shape": domain,
            "grid_shape": [1, 1, 1],
            "tile_shape": [1, scan.axis.extent, 1, 2, 3, 1],
            "serial_tiles": 1,
            "workgroup_threads": 128,
            "subgroup_size": 32,
            "scratch_bytes": 48,
            "input_buffers": list(range(len(program.inputs))),
            "output_buffers": [result_ids[value] for value in outputs],
            "dependencies": [],
            "reduction_axis": 1,
            "composition_state_inputs": [outer_ids[scan.state[0].initial]],
            "composition_state_outputs": [result_ids[scan.state[0].final]],
        },
    )
    region_results = [{"id": result_ids[value], "type": _native_type(value)} for value in outputs]
    return {
        "schema_version": 3,
        "target": "sm_90",
        "fingerprint": "d" * 64,
        "binding_indices": [],
        "region_result_indices": [],
        "reuse_groups": [],
        "region": {
            **_native_operation("shuttle.region", results=region_results),
            "regions": [
                [
                    {
                        "arguments": [{"id": outer_ids[value], "type": _native_type(value)} for value in program.inputs],
                        "operations": [
                            native_scan,
                            _native_operation("shuttle.yield", operands=list(result_ids.values())),
                        ],
                    }
                ]
            ],
        },
        "materialization": _native_operation("shuttle.materialization_plan", operations=[materialization]),
        "schedule": _native_operation("shuttle.schedule_plan", operations=[schedule]),
        "event_lowering_plan": _native_operation(
            "shuttle.event_lowering_plan", operations=[_native_operation("shuttle.event_lowering_plan_yield")]
        ),
    }


def test_native_scan_refinement_matches_generic_chunk_and_exact_event_fingerprints() -> None:
    program = remove_dead_operations(
        natural_affine_scan_program(
            NaturalAffineScanConfig(batch=1, sequence=4, heads=1, key_dimension=2, value_dimension=3, update_rank=1)
        )
    )
    request = decode_gpu_physical_artifact_request(json.dumps(_native_natural_scan_request(program)))

    native = refine_native_scan_task_event_plan(
        request,
        algebra_program=program,
        chunk_extent=2,
        state_column_tile=2,
    )
    direct_coarse = canonical_task_event_plan_from_native_request(request, algebra_program=program).task_plan
    direct = refine_whole_program_scan_tasks(program, direct_coarse, chunk_extent=2, state_column_tile=2)

    assert native.algebra_fingerprint == program.semantic_fingerprint
    assert native.task_plan.fingerprint == direct.task_plan.fingerprint
    assert native.event_fingerprint == canonical_task_event_boundary_fingerprint(
        direct.task_plan, direct.event_plans, ()
    )
    assert tuple(event.name for event in native.events) == ("scan.0.state_carry", "scan.0.state_commit")
    assert native.task_plan.bounded_buffers[0].capacity == 2
    physical = executable_scan_physical_request_from_native_request(
        request,
        algebra_program=program,
        chunk_extent=2,
        state_column_tile=2,
    )
    artifact = generate_cuda_ordered_state_artifact(ordered_state_scan_component(program, 0), physical)

    assert physical.task_plan.fingerprint == native.task_plan.fingerprint
    assert (
        canonical_task_event_boundary_fingerprint(physical.task_plan, physical.event_plans, ())
        == native.event_fingerprint
    )
    assert physical.provenance["provider_request_fingerprint"] == request.fingerprint
    assert artifact.request_fingerprint == physical.fingerprint
    assert "ordered_state_kernel" in artifact.cuda_source


def _canonical_composed_relation_program() -> TensorProgram:
    row = TensorAxis(0, 2, "row")
    feature = TensorAxis(1, 4, "feature")
    source = ProgramValue("source", (row,), DType.FP32)
    reindexed = ProgramValue("reindexed", (feature, row), DType.FP32)
    reduced = ProgramValue("reduced", (row,), DType.FP32)
    mapped = ProgramValue("mapped", (row,), DType.FP32)
    output = ProgramValue("output", (row,), DType.FP32)
    return TensorProgram(
        (source,),
        (
            RelationPrimitive(
                "broadcast.transpose",
                (source,),
                (reindexed,),
                (feature, row),
                (OperandIndexing(source, (feature, row), (index_axis(row),)),),
                result_sources=(source,),
            ),
            FoldPrimitive(
                "reduce",
                reindexed,
                reduced,
                (feature,),
                FoldReducer.SUM,
                DType.FP32,
                EvaluationOrder.ORDER_FREE,
            ),
            MapPrimitive(
                "affine",
                (reduced,),
                mapped,
                scalar_binary(
                    ScalarExpressionKind.ADD,
                    scalar_binary(
                        ScalarExpressionKind.MULTIPLY,
                        scalar_input(reduced.name),
                        scalar_constant(0.5),
                    ),
                    scalar_constant(1.0),
                ),
            ),
            MapPrimitive("identity.final", (mapped,), output, scalar_input(mapped.name)),
        ),
        (output,),
        NumericalPolicy.FAST,
    )


def _native_composed_relation_request_record() -> dict[str, Any]:
    request_record = _native_request_record()
    operations = request_record["region"]["regions"][0][0]["operations"]
    broadcast = operations[0]
    broadcast["results"] = [{"id": 5, "type": {"kind": "ranked_tensor", "element_type": "f32", "shape": [2, 4]}}]
    transpose = _native_operation(
        "shuttle.relation",
        attributes={
            "source": _source(1),
            "semantics": {"kind": "mlir_attribute", "value": "#shuttle.relation_semantics<transpose>"},
            "parameters": [1, 0],
        },
        operands=[5],
        results=[{"id": 1, "type": {"kind": "ranked_tensor", "element_type": "f32", "shape": [4, 2]}}],
    )
    operations.insert(1, transpose)
    fold, affine, finalizer = operations[2:5]
    fold["attributes"]["source"] = _source(2)
    fold["attributes"]["reduction_dimensions"] = [0]
    affine["attributes"]["source"] = _source(3)
    finalizer["attributes"]["source"] = _source(4)
    materialization_tasks = request_record["materialization"]["regions"][0][0]["operations"]
    materialization_tasks[0]["attributes"]["composition_sources"] = [_source(value) for value in [0, 1, 2, 3]]
    materialization_tasks[1]["attributes"]["composition_sources"] = [_source(4)]
    return request_record


def test_native_relation_source_chain_groups_by_canonical_ssa_boundary() -> None:
    request_record = _native_composed_relation_request_record()

    decoded = decode_gpu_physical_artifact_request(json.dumps(request_record))
    program = _canonical_composed_relation_program()
    native = canonical_task_event_plan_from_native_request(decoded, algebra_program=program)

    assert native.algebra_fingerprint == program.semantic_fingerprint
    assert native.task_plan.tasks[0].operation_ordinals == (0, 1, 2)
    assert native.task_plan.tasks[0].operation_kinds == (
        TaskOperationKind.RELATION,
        TaskOperationKind.FOLD,
        TaskOperationKind.MAP,
    )


def test_native_fold_region_sources_define_materialization_composition() -> None:
    request_record = _native_request_record()
    first_task = request_record["materialization"]["regions"][0][0]["operations"][0]
    attributes = first_task["attributes"]
    attributes["fold_region_sources"] = attributes.pop("composition_sources")
    attributes["source"] = _source(1)

    program = _canonical_algebra_program()
    native = canonical_task_event_plan_from_native_request(
        decode_gpu_physical_artifact_request(json.dumps(request_record)),
        algebra_program=program,
    )

    assert native.algebra_fingerprint == program.semantic_fingerprint
    assert native.task_plan.tasks[0].operation_ordinals == (0, 1, 2)


def test_native_fold_region_sources_must_include_principal_source() -> None:
    request_record = _native_request_record()
    first_task = request_record["materialization"]["regions"][0][0]["operations"][0]
    attributes = first_task["attributes"]
    attributes["fold_region_sources"] = attributes.pop("composition_sources")
    attributes["source"] = _source(9)

    with pytest.raises(ValueError, match="principal source is outside its composition"):
        canonical_task_event_plan_from_native_request(
            decode_gpu_physical_artifact_request(json.dumps(request_record)),
            algebra_program=_canonical_algebra_program(),
        )


def test_native_two_fold_region_sources_preserve_sequential_algebra_grouping() -> None:
    request_record = _native_request_record()
    row = TensorAxis(0, 2, "row")
    feature = TensorAxis(1, 4, "feature")
    source = ProgramValue("source", (row,), DType.FP32)
    right = ProgramValue("right", (), DType.FP32)
    broadcast = ProgramValue("broadcast", (row, feature), DType.FP32)
    first = ProgramValue("first", (row,), DType.FP32)
    mapped = ProgramValue("mapped", (row,), DType.FP32)
    second = ProgramValue("second", (), DType.FP32)
    finalized = ProgramValue("finalized", (), DType.FP32)
    output = ProgramValue("output", (), DType.FP32)
    program = TensorProgram(
        (source, right),
        (
            RelationPrimitive(
                "broadcast",
                (source,),
                (broadcast,),
                (row, feature),
                (OperandIndexing(source, (row, feature), (index_axis(row),)),),
                result_sources=(source,),
            ),
            FoldPrimitive("first", broadcast, first, (feature,), FoldReducer.SUM, DType.FP32),
            MapPrimitive("between", (first,), mapped, scalar_input(first.name)),
            FoldPrimitive("second", mapped, second, (row,), FoldReducer.SUM, DType.FP32),
            MapPrimitive("finalize", (second,), finalized, scalar_input(second.name)),
            ContractPrimitive("terminal", (finalized, right), output, (), DType.FP32),
        ),
        (output,),
        NumericalPolicy.FAST,
    )
    tensor_type = {"kind": "ranked_tensor", "element_type": "f32", "shape": [2, 4]}
    row_type = {"kind": "ranked_tensor", "element_type": "f32", "shape": [2]}
    scalar_tensor = {"kind": "ranked_tensor", "element_type": "f32", "shape": []}
    scalar_type = {"kind": "scalar", "element_type": "f32"}

    def identity_map(source_ordinal: int, operand: int, result: int, result_type: dict[str, object]):
        argument = result * 10
        return _native_operation(
            "shuttle.map",
            attributes={"source": _source(source_ordinal)},
            operands=[operand],
            results=[{"id": result, "type": result_type}],
            operations=[_native_operation("shuttle.yield", operands=[argument])],
            arguments=[{"id": argument, "type": scalar_type}],
        )

    initializer = _native_operation(
        "shuttle.map",
        attributes={"source": _source(3)},
        results=[{"id": 4, "type": scalar_tensor}],
        operations=[
            _native_operation(
                "arith.constant",
                attributes={"value": {"kind": "float", "element_type": "f32", "bits": "0"}},
                results=[{"id": 40, "type": scalar_type}],
            ),
            _native_operation("shuttle.yield", operands=[40]),
        ],
    )
    scalar_map = {"kind": "affine_map", "dimensions": 0, "symbols": 0, "results": []}
    operations = [
        _native_operation(
            "shuttle.relation",
            attributes={
                "source": _source(0),
                "semantics": {"kind": "mlir_attribute", "value": "#shuttle.relation_semantics<broadcast_in_dim>"},
                "parameters": [0],
            },
            operands=[0],
            results=[{"id": 2, "type": tensor_type}],
        ),
        _native_operation(
            "shuttle.fold",
            attributes={"source": _source(1), "reduction_dimensions": [1]},
            operands=[2],
            results=[{"id": 3, "type": row_type}],
            operations=[_native_operation("arith.addf")],
        ),
        identity_map(2, 3, 5, row_type),
        initializer,
        _native_operation(
            "shuttle.fold",
            attributes={"source": _source(4), "reduction_dimensions": [0]},
            operands=[5, 4],
            results=[{"id": 6, "type": scalar_tensor}],
            operations=[_native_operation("arith.addf")],
        ),
        identity_map(5, 6, 7, scalar_tensor),
        _native_operation(
            "shuttle.contract",
            attributes={
                "source": _source(6),
                "indexing_maps": [scalar_map, scalar_map, scalar_map],
                "reduction_dimensions": [],
                "accumulator_types": [scalar_type],
            },
            operands=[7, 1],
            results=[{"id": 8, "type": scalar_tensor}],
        ),
        _native_operation("shuttle.yield", operands=[8]),
    ]
    request_record["region"] = {
        **_native_operation("shuttle.region", results=[{"id": 8, "type": scalar_tensor}]),
        "regions": [
            [
                {
                    "arguments": [
                        {"id": 0, "type": row_type},
                        {"id": 1, "type": scalar_tensor},
                    ],
                    "operations": operations,
                }
            ]
        ],
    }
    materialization = request_record["materialization"]["regions"][0][0]["operations"]
    materialization[0]["attributes"]["fold_region_sources"] = [_source(index) for index in range(5)]
    materialization[0]["attributes"].pop("composition_sources")
    materialization[0]["attributes"]["source"] = _source(1)
    materialization[1]["attributes"]["composition_sources"] = [_source(5), _source(6)]
    materialization[1]["attributes"]["composition_recomputed_sources"] = [_source(2)]
    materialization[1]["attributes"]["source"] = _source(6)

    native = canonical_task_event_plan_from_native_request(
        decode_gpu_physical_artifact_request(json.dumps(request_record)),
        algebra_program=program,
    )

    assert native.algebra_fingerprint == program.semantic_fingerprint
    assert native.task_plan.tasks[0].operation_ordinals == (0, 1, 2, 3)
    assert native.task_plan.tasks[1].operation_ordinals == (4, 5)


def test_native_reshape_relation_uses_canonical_index_expressions() -> None:
    request_record = _native_request_record()
    region = request_record["region"]
    region["regions"][0][0]["arguments"][0]["type"] = {
        "kind": "ranked_tensor",
        "element_type": "f32",
        "shape": [2, 4],
    }
    relation = region["regions"][0][0]["operations"][0]
    relation["attributes"]["semantics"]["value"] = "#shuttle.relation_semantics<reshape>"
    relation["attributes"]["parameters"] = []
    decoded = decode_gpu_physical_artifact_request(json.dumps(request_record))

    native = canonical_task_event_plan_from_native_request(
        decoded,
        algebra_program=_canonical_algebra_program(broadcast_source=False),
    )

    assert native.algebra_fingerprint == _canonical_algebra_program(broadcast_source=False).semantic_fingerprint


def _canonical_dynamic_gather_program() -> TensorProgram:
    expert = TensorAxis(0, 3, "expert")
    row = TensorAxis(1, 2, "row")
    feature = TensorAxis(2, 2, "feature")
    token = TensorAxis(3, 2, "token")
    slot = TensorAxis(4, 1, "slot")
    vector = TensorAxis(5, 1, "vector")
    source = ProgramValue("source", (expert, row, feature), DType.BF16)
    indices = ProgramValue("indices", (token, slot, vector), DType.INT32)
    output = ProgramValue("output", (token, slot, row, feature), DType.BF16)
    dynamic_expert = index_tensor_element(indices, index_axis(token), index_axis(slot), index_constant(0))
    return TensorProgram(
        (source, indices),
        (
            RelationPrimitive(
                "dynamic.gather",
                (source, indices),
                (output,),
                output.axes,
                (
                    OperandIndexing(
                        source,
                        output.axes,
                        (dynamic_expert, index_axis(row), index_axis(feature)),
                    ),
                    OperandIndexing(
                        indices,
                        output.axes,
                        (index_axis(token), index_axis(slot), index_constant(0)),
                    ),
                ),
                result_sources=(source,),
            ),
        ),
        (output,),
        NumericalPolicy.FAST,
    )


def _native_dynamic_gather_request_record() -> dict[str, Any]:
    request = _native_request_record()
    source_type = {"kind": "ranked_tensor", "element_type": "bf16", "shape": [3, 2, 2]}
    indices_type = {"kind": "ranked_tensor", "element_type": "i32", "shape": [2, 1, 1]}
    output_type = {"kind": "ranked_tensor", "element_type": "bf16", "shape": [2, 1, 2, 2]}
    gather = _native_operation(
        "shuttle.relation",
        attributes={
            "source": _source(0),
            "semantics": {"kind": "mlir_attribute", "value": "#shuttle.relation_semantics<gather>"},
            "parameters": [
                {
                    "offset_dims": [2, 3],
                    "collapsed_slice_dims": [0],
                    "operand_batching_dims": [],
                    "start_indices_batching_dims": [],
                    "start_index_map": [0],
                    "index_vector_dim": 2,
                    "slice_sizes": [1, 2, 2],
                    "indices_are_sorted": False,
                }
            ],
        },
        operands=[0, 1],
        results=[{"id": 2, "type": output_type}],
    )
    request["region"] = {
        **_native_operation("shuttle.region", results=[{"id": 2, "type": output_type}]),
        "regions": [
            [
                {
                    "arguments": [{"id": 0, "type": source_type}, {"id": 1, "type": indices_type}],
                    "operations": [gather, _native_operation("shuttle.yield", operands=[2])],
                }
            ]
        ],
    }
    materialization_task = _native_operation(
        "shuttle.materialization_task",
        attributes={
            "ordinal": 0,
            "composition_sources": [_source(0)],
            "input_buffers": [],
            "output_buffers": [],
            "dependencies": [],
            "domain_shape": [2, 1, 2, 2],
            "reduction_dimensions": [],
            "composition_owner_dimensions": [0, 1, 2, 3],
            "online_vector_dimensions": [],
            "composition_stream_dimensions": [],
            "composition_producer_contraction_dimensions": [],
            "composition_consumer_contraction_dimensions": [],
        },
    )
    schedule_task = _native_operation(
        "shuttle.schedule_task",
        attributes={
            "ordinal": 0,
            "source_task": 0,
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.schedule_task_kind<elementwise>"},
            "domain_shape": [2, 1, 2, 2],
            "grid_shape": [1, 1, 1, 1],
            "tile_shape": [2, 1, 2, 2],
            "serial_tiles": 1,
            "workgroup_threads": 32,
            "subgroup_size": 32,
            "scratch_bytes": 0,
            "input_buffers": [],
            "output_buffers": [],
            "dependencies": [],
        },
    )
    request["materialization"] = _native_operation("shuttle.materialization_plan", operations=[materialization_task])
    request["schedule"] = _native_operation("shuttle.schedule_plan", operations=[schedule_task])
    request["event_lowering_plan"] = _native_operation(
        "shuttle.event_lowering_plan", operations=[_native_operation("shuttle.event_lowering_plan_yield")]
    )
    return request


def test_native_gather_relation_preserves_dynamic_index_program() -> None:
    request = _native_dynamic_gather_request_record()
    program = _canonical_dynamic_gather_program()

    native = canonical_task_event_plan_from_native_request(
        decode_gpu_physical_artifact_request(json.dumps(request)), algebra_program=program
    )

    assert native.algebra_fingerprint == program.semantic_fingerprint
    assert native.task_plan.tasks[0].operation_kinds == (TaskOperationKind.RELATION,)
    assert native.events == ()
    assert native.task_plan.tasks[1].operation_kinds == (TaskOperationKind.INPUT_STAGE,)
    assert len(native.task_plan.dependence_programs) == len(native.event_programs) == 1
    dependence = native.task_plan.dependence_programs[0]
    assert dependence.source_task_id == 1
    assert dependence.target_task_id == 0
    assert tuple(binding.algebra_value_ordinal for binding in dependence.runtime_bindings) == (1,)
    first_event, first_allocation = instantiate_parametric_event_tensor(
        native.event_programs[0],
        task_plan=native.task_plan,
        algebra_values={1: jnp.asarray([[[0]], [[2]]], dtype=jnp.int32)},
        name="native.dynamic.gather",
    )
    second_event, second_allocation = instantiate_parametric_event_tensor(
        native.event_programs[0],
        task_plan=native.task_plan,
        algebra_values={1: jnp.asarray([[[1]], [[1]]], dtype=jnp.int32)},
        name="native.dynamic.gather",
    )
    assert first_event.scheduled_dependence.pairs == (
        IndexPair((0, 0, 0), (0, 0, 0, 0)),
        IndexPair((2, 0, 0), (0, 0, 0, 0)),
    )
    assert second_event.scheduled_dependence.pairs == (IndexPair((1, 0, 0), (0, 0, 0, 0)),)
    assert first_allocation.capacity == second_allocation.capacity == 1
    assert first_allocation.generation_count == second_allocation.generation_count == 1
    assert first_allocation.storage.slots == second_allocation.storage.slots == (0,)
    assert first_allocation.storage.generations == second_allocation.storage.generations == (0,)
    serialized = json.dumps(native.canonical_record, sort_keys=True)
    assert "dependence_programs" in serialized
    assert "event_programs" in serialized
    assert "moe" not in serialized.lower()

    with pytest.raises(ValueError, match="capacity and generation count must be positive"):
        replace(native.event_programs[0], capacity=0)
    with pytest.raises(ValueError, match="generation is outside"):
        replace(native.event_programs[0], generations=(1,))

    request["region"]["regions"][0][0]["operations"][0]["attributes"]["parameters"][0]["start_index_map"] = [1]
    with pytest.raises(ValueError, match="full-slice form"):
        canonical_task_event_plan_from_native_request(
            decode_gpu_physical_artifact_request(json.dumps(request)), algebra_program=program
        )


def _native_indexed_contributions_request_record() -> dict[str, Any]:
    request = _native_request_record()
    destination_type = {"kind": "ranked_tensor", "element_type": "bf16", "shape": [3, 2]}
    indices_type = {"kind": "ranked_tensor", "element_type": "i32", "shape": [2, 1]}
    updates_type = {"kind": "ranked_tensor", "element_type": "bf16", "shape": [2, 2]}
    contributions_type = {"kind": "ranked_tensor", "element_type": "bf16", "shape": [3, 2, 2]}
    scalar_type = {"kind": "scalar", "element_type": "bf16"}

    def expansion(member: int) -> dict[str, int]:
        return {"member": member, "size": 4}

    relation = _native_operation(
        "shuttle.relation",
        attributes={
            "source": _source(0),
            "source_group": expansion(0),
            "semantics": {
                "kind": "mlir_attribute",
                "value": "#shuttle.relation_semantics<indexed_contributions>",
            },
            "parameters": [
                {
                    "output_shape": [3, 2],
                    "update_window_dims": [1],
                    "inserted_window_dims": [0],
                    "input_batching_dims": [],
                    "scatter_indices_batching_dims": [],
                    "scatter_dims_to_operand_dims": [0],
                    "index_vector_dim": 1,
                    "indices_are_sorted": False,
                    "unique_indices": False,
                    "conflict_resolution": "source_ordered_add",
                }
            ],
        },
        operands=[2, 1],
        results=[{"id": 3, "type": contributions_type}],
    )
    initializer = _native_operation(
        "shuttle.map",
        attributes={"source": _source(0), "source_group": expansion(1)},
        results=[{"id": 4, "type": scalar_type}],
        operations=[
            _native_operation(
                "arith.constant",
                attributes={"value": {"kind": "float", "element_type": "f32", "bits": "00000000"}},
                results=[{"id": 40, "type": scalar_type}],
            ),
            _native_operation("shuttle.yield", operands=[40]),
        ],
    )
    fold = _native_operation(
        "shuttle.fold",
        attributes={"source": _source(0), "source_group": expansion(2), "reduction_dimensions": [2]},
        operands=[3, 4],
        results=[{"id": 5, "type": destination_type}],
        operations=[_native_operation("arith.addf")],
    )
    finalizer = _native_operation(
        "shuttle.map",
        attributes={"source": _source(0), "source_group": expansion(3)},
        operands=[0, 5],
        results=[{"id": 6, "type": destination_type}],
        operations=[
            _native_operation("arith.addf", operands=[60, 61], results=[{"id": 62, "type": scalar_type}]),
            _native_operation("shuttle.yield", operands=[62]),
        ],
        arguments=[{"id": 60, "type": scalar_type}, {"id": 61, "type": scalar_type}],
    )
    request["region"] = {
        **_native_operation("shuttle.region", results=[{"id": 6, "type": destination_type}]),
        "regions": [
            [
                {
                    "arguments": [
                        {"id": 0, "type": destination_type},
                        {"id": 1, "type": indices_type},
                        {"id": 2, "type": updates_type},
                    ],
                    "operations": [
                        relation,
                        initializer,
                        fold,
                        finalizer,
                        _native_operation("shuttle.yield", operands=[6]),
                    ],
                }
            ]
        ],
    }
    materialization_task = _native_operation(
        "shuttle.materialization_task",
        attributes={
            "ordinal": 0,
            "source": _source(0),
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.materialization_task_kind<fold>"},
            "composition_kind": "indexed_accumulation",
            "composition_sources": [_source(0)] * 4,
            "source_expansion_group": [expansion(member) for member in range(4)],
            "input_buffers": [0, 1, 2],
            "output_buffers": [3],
            "dependencies": [],
            "domain_shape": [3, 2, 2],
            "reduction_dimensions": [2],
            "composition_owner_dimensions": [0, 1],
            "online_vector_dimensions": [],
            "composition_stream_dimensions": [2],
            "composition_producer_contraction_dimensions": [],
            "composition_consumer_contraction_dimensions": [],
        },
    )
    schedule_task = _native_operation(
        "shuttle.schedule_task",
        attributes={
            "ordinal": 0,
            "source_task": 0,
            "kind": {"kind": "mlir_attribute", "value": "#shuttle.schedule_task_kind<row_fold>"},
            "domain_shape": [3, 2, 2],
            "grid_shape": [6],
            "tile_shape": [1, 1, 2],
            "serial_tiles": 1,
            "workgroup_threads": 32,
            "subgroup_size": 32,
            "scratch_bytes": 128,
            "reduction_axis": 2,
            "input_buffers": [0, 1, 2],
            "output_buffers": [3],
            "dependencies": [],
        },
    )
    request["materialization"] = _native_operation("shuttle.materialization_plan", operations=[materialization_task])
    request["schedule"] = _native_operation("shuttle.schedule_plan", operations=[schedule_task])
    request["event_lowering_plan"] = _native_operation(
        "shuttle.event_lowering_plan", operations=[_native_operation("shuttle.event_lowering_plan_yield")]
    )
    return request


def test_native_indexed_contributions_bind_jax_owned_scatter_reverse() -> None:
    dimensions = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)
    )

    def scatter(operand, indices, updates):
        return jax.lax.scatter_add(operand, indices, updates, dimensions, mode="promise_in_bounds")

    program = jax_function_to_tensor_program(
        scatter,
        jnp.ones((3, 2), dtype=jnp.bfloat16),
        jnp.asarray([[0], [2]], dtype=jnp.int32),
        jnp.ones((2, 2), dtype=jnp.bfloat16),
    )
    request = _native_indexed_contributions_request_record()
    native = canonical_task_event_plan_from_native_request(
        decode_gpu_physical_artifact_request(json.dumps(request)), algebra_program=program
    )

    assert native.algebra_fingerprint == program.semantic_fingerprint
    assert native.task_plan.tasks[0].operation_kinds == (
        TaskOperationKind.RELATION,
        TaskOperationKind.FOLD,
        TaskOperationKind.MAP,
    )
    assert all(task.operation_kinds == (TaskOperationKind.INPUT_STAGE,) for task in native.task_plan.tasks[1:])
    assert len(native.task_plan.dependence_programs) == len(native.event_programs) == 2
    first_pairs = tuple(
        instantiate_parametric_event_tensor(
            event,
            task_plan=native.task_plan,
            algebra_values={1: jnp.asarray([[0], [2]], dtype=jnp.int32)},
            name="native.indexed-contributions",
        )[0].scheduled_dependence.pairs
        for event in native.event_programs
    )
    second_pairs = tuple(
        instantiate_parametric_event_tensor(
            event,
            task_plan=native.task_plan,
            algebra_values={1: jnp.asarray([[1], [1]], dtype=jnp.int32)},
            name="native.indexed-contributions",
        )[0].scheduled_dependence.pairs
        for event in native.event_programs
    )
    assert first_pairs != second_pairs
    assert all(event.capacity == 6 for event in native.event_programs)
    assert all(event.generation_count == 1 for event in native.event_programs)
    assert all(event.slots == tuple(range(6)) for event in native.event_programs)
    assert all(event.generations == (0,) * 6 for event in native.event_programs)
    assert "indexed_contributions" not in json.dumps(native.canonical_record, sort_keys=True)

    request["region"]["regions"][0][0]["operations"][0]["attributes"]["parameters"][0][
        "conflict_resolution"
    ] = "last_writer"
    with pytest.raises(ValueError, match="source-ordered additive conflicts"):
        canonical_task_event_plan_from_native_request(
            decode_gpu_physical_artifact_request(json.dumps(request)), algebra_program=program
        )


def test_decoded_native_task_event_plan_conforms_to_finer_reference_plan() -> None:
    request_record = _native_composed_relation_request_record()
    decoded = decode_gpu_physical_artifact_request(json.dumps(request_record))
    algebra_program = _canonical_composed_relation_program()
    native = canonical_task_event_plan_from_native_request(decoded, algebra_program=algebra_program)
    native_core, native_output = native.task_plan.tasks
    assert native_core.structure.contraction_axes == (TaskAxisTile(1, 4, 4),)
    relation = _task(0, "reference.relation", ((0, TaskOperationKind.RELATION, TaskCapability.INDEX_RELATION),))
    fold = _task(1, "reference.fold", ((1, TaskOperationKind.FOLD, TaskCapability.ASSOCIATIVE_FOLD),))
    mapped = _task(2, "reference.map", ((2, TaskOperationKind.MAP, TaskCapability.SCALAR_MAP),))
    output = _task(3, "reference.output", ((3, TaskOperationKind.MAP, TaskCapability.SCALAR_MAP),))
    internal_relation = _dependence(relation, fold)
    internal = _dependence(fold, mapped)
    external = _dependence(mapped, output)
    reference = CanonicalTaskPlan((relation, fold, mapped, output), (internal_relation, internal, external))
    reference_events = (
        derive_event_tensor_plan(internal_relation, name="reference.internal.relation"),
        derive_event_tensor_plan(internal, name="reference.internal"),
        derive_event_tensor_plan(
            external,
            name="reference.external",
            generation_policy=EventGenerationPolicy.PER_INVOCATION,
            scheduling_mode=EventSchedulingMode.DYNAMIC,
        ),
    )
    refinement = CanonicalTaskPlanRefinement(
        native.algebra_fingerprint,
        PhysicalTarget.SM90,
        PhysicalNumericalPolicy.FAST,
        native.task_plan,
        reference,
        (
            identity_task_refinement_binding(0, 0, refined_family=relation.family, coarse_family=native_core.family),
            identity_task_refinement_binding(1, 0, refined_family=fold.family, coarse_family=native_core.family),
            identity_task_refinement_binding(2, 0, refined_family=mapped.family, coarse_family=native_core.family),
            identity_task_refinement_binding(3, 1, refined_family=output.family, coarse_family=native_output.family),
        ),
        coarse_events=native.events,
        refined_events=reference_events,
    )

    assert native.task_plan.fingerprint != reference.fingerprint
    assert refinement.quotient_record["task_plan"] == native.task_plan.canonical_record
    assert len(refinement.proof_record["internalized_dependencies"]) == 2

    request_record["fingerprint"] = "0" * 64
    renamed_request = decode_gpu_physical_artifact_request(json.dumps(request_record))
    renamed_native = canonical_task_event_plan_from_native_request(renamed_request, algebra_program=algebra_program)
    assert renamed_native.algebra_fingerprint == native.algebra_fingerprint
    assert native.algebra_fingerprint == algebra_program.semantic_fingerprint


def test_real_native_dense_capture_conforms_to_generic_algebra_and_exact_events() -> None:
    def forward(lhs, rhs):
        lhs_f32 = jnp.transpose(lhs.astype(jnp.float32), (0, 2, 1, 3))
        rhs_f32 = jnp.transpose(rhs.astype(jnp.float32), (0, 2, 1, 3))
        result = jnp.einsum("bhmk,bhnk->bhmn", lhs_f32, rhs_f32)
        return result.astype(jnp.bfloat16)

    shape = (2, 5, 3, 7)
    rhs_shape = (2, 11, 3, 7)
    program = jax_function_to_tensor_program(
        forward,
        jax.ShapeDtypeStruct(shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(rhs_shape, jnp.bfloat16),
        numerical_policy=NumericalPolicy.FAST,
    )
    fixture = Path(__file__).parent / "fixtures/native_schema_v3/jax-bf16-batched-contract-forward-sm90.json"
    request = decode_gpu_physical_artifact_request(fixture.read_text())

    native = canonical_task_event_plan_from_native_request(request, algebra_program=program)

    assert native.algebra_fingerprint == program.semantic_fingerprint
    for precision in (ContractOperandPrecision.HIGH, ContractOperandPrecision.HIGHEST):
        explicit_precision = replace(
            program,
            operations=tuple(
                (
                    replace(operation, operand_precision=(precision,) * len(operation.inputs))
                    if isinstance(operation, ContractPrimitive)
                    else operation
                )
                for operation in program.operations
            ),
        )
        with pytest.raises(ValueError, match="cannot preserve explicit HIGH/HIGHEST operand precision"):
            canonical_task_event_plan_from_native_request(request, algebra_program=explicit_precision)
    contract_task = next(task for task in native.task_plan.tasks if TaskOperationKind.CONTRACT in task.operation_kinds)
    assert contract_task.structure.streamed_axes == ()
    assert contract_task.structure.reduction_axes == ()
    assert tuple((axis.extent, axis.tile_extent) for axis in contract_task.structure.contraction_axes) == ((7, 7),)
    assert any(TaskOperationKind.RELATION in task.operation_kinds for task in native.task_plan.tasks)
    assert any(count.value > 1 for event in native.events for count in event.initial_count.counts)
    assert {requirement.notify_count for requirement in request.event_requirements} == {1}

    values = (*program.inputs, *(value for operation in program.operations for value in primitive_outputs(operation)))
    generic_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={axis.id: axis.extent for value in values for axis in value.axes},
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    assert tuple(task.operation_ordinals for task in native.task_plan.tasks) == tuple(
        task.operation_ordinals for task in generic_plan.tasks
    )

    def task_edges(plan: CanonicalTaskPlan) -> set[tuple[int, int]]:
        task_by_family = {task.family: task.id for task in plan.tasks}
        return {
            (task_by_family[dependence.relation.source], task_by_family[dependence.relation.target])
            for dependence in plan.dependences
        }

    assert task_edges(native.task_plan) == task_edges(generic_plan)
    generic_events = eventize_composed_task_plan(generic_plan)
    physical_request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=generic_plan,
        event_program=generic_events,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    assert physical_request.algebra_fingerprint == native.algebra_fingerprint
    assert len(physical_request.schedules) == len(native.task_plan.tasks)


def test_real_native_attention_capture_canonicalizes_source_and_index_decomposition() -> None:
    def attention(query, key, value):
        query_f32 = query.astype(jnp.float32)
        key_f32 = key.astype(jnp.float32)
        value_f32 = value.astype(jnp.float32)
        scores = jnp.einsum("bhgqd,bhkd->bhgqk", query_f32, key_f32) * jnp.float32(0.125)
        query_index = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 17, 1)
        key_index = jnp.arange(17, dtype=jnp.int32).reshape(1, 1, 1, 1, 17)
        predicate = jnp.broadcast_to(key_index <= query_index, scores.shape)
        masked = jax.lax.select(predicate, scores, jnp.full(scores.shape, -jnp.inf))
        maximum = jnp.max(masked, axis=4, keepdims=True)
        numerator = jnp.exp(masked - maximum)
        denominator = jnp.sum(numerator, axis=4, keepdims=True)
        weights = numerator / denominator
        return jnp.einsum("bhgqk,bhkv->bhgqv", weights, value_f32).astype(jnp.bfloat16)

    shapes = ((1, 2, 2, 17, 8), (1, 2, 17, 8), (1, 2, 17, 7))
    program = jax_function_to_tensor_program(
        attention,
        *(jax.ShapeDtypeStruct(shape, jnp.bfloat16) for shape in shapes),
        numerical_policy=NumericalPolicy.FAST,
    )
    fixture = Path(__file__).parent / "fixtures/native_schema_v3/jax-bf16-causal-gqa-online-axis4-sm90.json"
    request = decode_gpu_physical_artifact_request(fixture.read_text())

    native = canonical_task_event_plan_from_native_request(request, algebra_program=program)

    assert native.algebra_fingerprint == program.semantic_fingerprint
    assert len(request.normalized_exp_contract_regions) == 1
    declared_region = request.normalized_exp_contract_regions[0]
    source_operations = tuple(
        operation
        for operation in request.region.body().operations
        if operation.name
        in {
            "shuttle.contract",
            "shuttle.fold",
            "shuttle.map",
            "shuttle.relation",
            "shuttle.selection",
        }
    )
    assert source_operations[declared_region.producer_operation].name == "shuttle.contract"
    assert source_operations[declared_region.consumer_operation].name == "shuttle.contract"
    assert all(source_operations[ordinal].name == "shuttle.fold" for ordinal in declared_region.fold_operations)
    assert len(native.task_plan.tasks) == 1
    assert native.task_plan.tasks[0].structure.principal_kind is TaskOperationKind.CONTRACT
    assert native.task_plan.tasks[0].operation_ordinals == tuple(range(len(program.operations)))
    assert native.events == ()
    assert set(native.task_plan.tasks[0].operation_kinds) >= {
        TaskOperationKind.CONTRACT,
        TaskOperationKind.FOLD,
        TaskOperationKind.MAP,
        TaskOperationKind.RELATION,
    }

    values = (*program.inputs, *(value for operation in program.operations for value in primitive_outputs(operation)))
    generic_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis={axis.id: axis.extent for value in values for axis in value.axes},
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    generic_events = eventize_composed_task_plan(generic_plan)
    physical_request = physical_request_for_composed_tasks(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=generic_plan,
        event_program=generic_events,
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )
    assert physical_request.algebra_fingerprint == native.algebra_fingerprint
    assert {ordinal for task in physical_request.task_plan.tasks for ordinal in task.operation_ordinals} == set(
        native.task_plan.tasks[0].operation_ordinals
    )


def test_real_native_scalar_pullback_capture_retains_all_axis_fold_and_exact_roles() -> None:
    fixture = (
        Path(__file__).parent / "fixtures/native_schema_v3/jax-bf16-causal-gqa-scalar-value-and-grad-s128-d128-sm90.json"
    )
    request = decode_gpu_physical_artifact_request(fixture.read_text())

    assert len(request.normalized_exp_contract_pullbacks) == 1
    pullback = request.normalized_exp_contract_pullbacks[0]
    assert (
        pullback.forward_task,
        pullback.objective_task,
        pullback.owner_pullback_task,
        pullback.streamed_pullback_task,
    ) == (0, 1, 2, 3)
    assert (
        pullback.maximum_state_buffer,
        pullback.normalizer_state_buffer,
        pullback.output_dependent_state_buffer,
    ) == (8, 9, 10)
    assert pullback.forward_result_buffer == 11
    assert pullback.output_cotangent_buffer == 3
    assert pullback.terminal_cotangent_buffers == (7, 6, 5)

    schedule_tasks = tuple(
        operation for operation in request.schedule.body().operations if operation.name == "shuttle.schedule_task"
    )
    objective = schedule_tasks[pullback.objective_task]
    assert objective.attributes["kind"]["value"] == "#shuttle.schedule_task_kind<all_axis_fold>"
    assert objective.attributes["domain_shape"] == [1, 8, 4, 128, 128]
    assert objective.attributes["reduction_dimensions"] == [0, 1, 2, 3, 4]
    assert objective.attributes["grid_shape"] == [128]
    assert objective.attributes["tile_shape"] == [4096]
    assert objective.attributes["serial_tiles"] == 16
    assert objective.attributes["reduction_partial_buffer_shape"] == [128]
    assert objective.attributes["reduction_merge_grid_shape"] == [1]
    assert objective.attributes["reduction_merge_tile_shape"] == [256]
    assert objective.attributes["reduction_merge_serial_tiles"] == 1
    assert len(request.event_relations) == 7
    assert len(request.event_requirements) == 7
    partial_to_merge = request.event_relations[-1]
    assert partial_to_merge.producer_task == pullback.objective_task
    assert partial_to_merge.consumer_task == pullback.objective_task
    assert partial_to_merge.producer_phase == "partial"
    assert partial_to_merge.consumer_phase == "merge"
    assert partial_to_merge.relation_domain_shape == (128,)
    assert partial_to_merge.storage_capacity == 128
    assert partial_to_merge.generations == 1
    assert partial_to_merge.scope == "device"
    assert partial_to_merge.release
    assert partial_to_merge.acquire


def test_native_adapter_retains_logical_event_erased_by_physical_order() -> None:
    request_record = _native_request_record()
    event_operations = request_record["event_lowering_plan"]["regions"][0][0]["operations"]
    event_operations[0]["attributes"]["kind"] = {
        "kind": "mlir_attribute",
        "value": "#shuttle.event_realization_kind<algebraic_refinement_erased>",
    }
    event_operations[0]["attributes"]["covering_launch_order"] = [0, 0, 1, 0]
    del event_operations[1:3]
    decoded = decode_gpu_physical_artifact_request(json.dumps(request_record))

    native = canonical_task_event_plan_from_native_request(decoded, algebra_program=_canonical_algebra_program())

    assert len(native.events) == 1
    assert native.events[0].scheduling_mode is EventSchedulingMode.STATIC
    assert native.physical_order_witnesses[0].kind == "algebraic_refinement_erased"
    assert native.physical_order_witnesses[0].covering_launch_order == (0, 0, 1, 0)


def test_msa_dsl_request_supports_runtime_parametric_refinement_identity() -> None:
    oracle = build_msa_reference_oracle(MSADebugConfig())
    dataflow = derive_msa_reference_dataflow(oracle)
    bindings = tuple(
        identity_task_refinement_binding(
            task.id,
            task.id,
            refined_family=task.family,
            coarse_family=task.family,
        )
        for task in dataflow.task_plan.tasks
    )
    refinement = CanonicalTaskPlanRefinement(
        oracle.canonical_program.semantic_fingerprint,
        PhysicalTarget.SM100,
        PhysicalNumericalPolicy.FAST,
        dataflow.task_plan,
        dataflow.task_plan,
        bindings,
        coarse_events=dataflow.program.event_plans,
        refined_events=dataflow.program.event_plans,
        coarse_event_programs=dataflow.event_programs,
        refined_event_programs=dataflow.event_programs,
    )

    assert len(dataflow.task_plan.dependence_programs) == 2
    assert refinement.quotient_record["algebra_fingerprint"] == oracle.canonical_program.semantic_fingerprint
