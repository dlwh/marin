# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable relation, Transport, task, and EventTensor plans for routed experts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from shuttle.ir import DType
from tile_lifetime.dense_flow_planner import derive_composed_task_plan, eventize_composed_task_plan
from tile_lifetime.event_buffering import physical_event_realization, verify_event_realizations
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    EventStorageBinding,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
    event_tensor_plan_record,
)
from tile_lifetime.physical_task_contract import (
    BoundedTaskCoordinate,
    CanonicalTaskPlan,
    DependenceRuntimeBinding,
    EventRealizationProof,
    EventTensorAllocation,
    ExecutablePhysicalRequest,
    ParametricEventDerivation,
    ParametricEventRealization,
    ParametricEventTensorPlan,
    PhaseEdgeConstraint,
    PhysicalNumericalPolicy,
    PhysicalPhase,
    PhysicalPhaseEdge,
    PhysicalSchedule,
    PhysicalScheduleFamily,
    PhysicalTarget,
    PhysicalTaskConstraint,
    PhysicalTaskStructure,
    RelationAccessClass,
    TaskAxisTile,
    TaskCapability,
    TaskDependenceProgram,
    TaskOperationKind,
    TaskResourceEstimate,
    TaskTraversal,
    TiledTask,
    physical_phase_iteration_space,
)
from tile_lifetime.relation import RelationPlacement, RelationPlan, build_fixed_capacity_relation_plan
from tile_lifetime.relation_transport import (
    RelationTilePipelinePlan,
    RelationTransportTemplate,
    RelationTransportTrainingPlan,
    TilePipelineEdge,
    TilePipelineGraph,
    TilePipelineStage,
    TransportCapacityMode,
    TransportEpochProtocol,
    TransportPayloadDomain,
    TransportPayloadField,
    bind_transport_epoch,
    derive_relation_tile_pipeline,
    derive_relation_transport_training_plan,
    execute_transport_field_flow_reference,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    EvaluationOrder,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpressionKind,
    SelectionComparison,
    SelectionOrder,
    SelectionPrimitive,
    SelectionTieBreak,
    TensorAxis,
    TensorProgram,
    TransportPrimitive,
    index_axis,
    index_tensor_element,
    primitive_outputs,
    scalar_binary,
    scalar_constant,
    scalar_convert,
    scalar_input,
    scalar_unary,
    tensor_program_value_ordinals,
)


@dataclass(frozen=True)
class ExpertRouteSchedule:
    """Finite tiling and bounded-storage choices for a generic routed relation."""

    source_placement: RelationPlacement
    destination_placement: RelationPlacement
    tile_rows: int
    macrobatch_rows: int
    phase_count: int = 2
    generation_count: int = 2

    def __post_init__(self) -> None:
        if self.source_placement.rank_count != self.destination_placement.rank_count:
            raise ValueError("source and destination relations must share one rank domain")
        if self.tile_rows <= 0:
            raise ValueError("route tile rows must be positive")
        if self.macrobatch_rows < self.tile_rows or self.macrobatch_rows % self.tile_rows:
            raise ValueError("route macrobatch rows must be a multiple of tile rows")


@dataclass(frozen=True)
class ExpertRouteDataflow:
    """Primal and cotangent task graphs derived from one concrete route relation."""

    relation: RelationPlan
    transport: RelationTransportTrainingPlan
    forward: RelationTilePipelinePlan
    reverse: RelationTilePipelinePlan

    @property
    def task_fingerprint(self) -> str:
        """Return route-value-independent identity for both bounded tiled task graphs."""
        return _fingerprint(
            {
                "schema": "shuttle.generic-routed-task-plan.v1",
                "transport_template": _transport_template_record(self.transport.template),
                "forward": _static_tile_pipeline_record(self.forward),
                "reverse": _static_tile_pipeline_record(self.reverse),
            }
        )

    @property
    def event_fingerprint(self) -> str:
        """Return route-value-independent identity for bounded EventTensor programs."""
        return _fingerprint(
            {
                "schema": "shuttle.generic-routed-event-plan.v1",
                "forward": _static_event_records(self.forward),
                "reverse": _static_event_records(self.reverse),
                "phase_count": self.transport.template.epoch_protocol.phase_count,
                "generation_count": self.transport.template.epoch_protocol.generation_modulus,
            }
        )

    def canonical_record(self) -> dict[str, Any]:
        """Return the lossless runtime relation and task/EventTensor request."""
        runtime = self.transport.runtime
        protocol = self.transport.template.epoch_protocol
        return {
            "schema": "shuttle.generic-routed-executable-plan.v1",
            "relation": {
                "source_item_count": self.relation.source_item_count,
                "route_slots": self.relation.route_slots,
                "destination_count": self.relation.destination_count,
                "edge_source_item": tuple(int(value) for value in runtime.edge_source_item),
                "edge_route_slot": tuple(int(value) for value in runtime.edge_route_slot),
                "edge_destination_item": tuple(int(value) for value in runtime.edge_destination_item),
                "edge_valid": tuple(bool(value) for value in runtime.edge_valid),
                "edge_to_destination_row": tuple(int(value) for value in runtime.edge_to_destination_row),
            },
            "transport_template": _transport_template_record(self.transport.template),
            "epoch": {
                "phase": self.transport.epoch.phase,
                "stored_generation": self.transport.epoch.stored_generation,
            },
            "task_fingerprint": self.task_fingerprint,
            "event_fingerprint": self.event_fingerprint,
            "forward": _tile_pipeline_record(
                self.forward,
                include_events=True,
                phase_count=protocol.phase_count,
                generation_count=protocol.generation_modulus,
            ),
            "reverse": _tile_pipeline_record(
                self.reverse,
                include_events=True,
                phase_count=protocol.phase_count,
                generation_count=protocol.generation_modulus,
            ),
        }

    def serialize(self) -> str:
        """Serialize the exact plan canonically for artifact provenance."""
        return json.dumps(self.canonical_record(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        """Return canonical identity for relation data and all lowering boundaries."""
        return hashlib.sha256(self.serialize().encode()).hexdigest()


def expert_tensor_program(
    *,
    token_count: int,
    hidden_dimension: int,
    intermediate_dimension: int,
    expert_count: int,
    selected_count: int,
    numerical_policy: NumericalPolicy = NumericalPolicy.SOURCE_ORDERED,
) -> TensorProgram:
    """Build the generic Contract/Map/Fold/Selection/Relation algebra for routed experts."""
    if min(token_count, hidden_dimension, intermediate_dimension, expert_count, selected_count) <= 0:
        raise ValueError("tensor-program dimensions must be positive")
    if selected_count > expert_count:
        raise ValueError("selected count cannot exceed the selection dimension")
    token = TensorAxis(0, token_count, "token")
    hidden = TensorAxis(1, hidden_dimension, "hidden")
    expert = TensorAxis(2, expert_count, "expert")
    intermediate = TensorAxis(3, intermediate_dimension, "intermediate")
    selected = TensorAxis(4, selected_count, "selected")

    source = ProgramValue("source", (token, hidden), DType.BF16)
    router_weight = ProgramValue("router.weight", (hidden, expert), DType.BF16)
    shared_gate_weight = ProgramValue("shared.gate_weight", (intermediate, hidden), DType.BF16)
    shared_up_weight = ProgramValue("shared.up_weight", (intermediate, hidden), DType.BF16)
    shared_down_weight = ProgramValue("shared.down_weight", (hidden, intermediate), DType.BF16)
    gate_weight = ProgramValue("routed.gate_weight", (expert, intermediate, hidden), DType.BF16)
    up_weight = ProgramValue("routed.up_weight", (expert, intermediate, hidden), DType.BF16)
    down_weight = ProgramValue("routed.down_weight", (expert, hidden, intermediate), DType.BF16)

    logits = ProgramValue("router.logits", (token, expert), DType.BF16)
    selected_values = ProgramValue("selection.values", (token, selected), DType.BF16)
    selected_indices = ProgramValue("selection.indices", (token, selected), DType.INT32)
    selected_fp32 = ProgramValue("selection.values_fp32", (token, selected), DType.FP32)
    route_max = ProgramValue("route.maximum", (token,), DType.FP32)
    route_exp = ProgramValue("route.exponential", (token, selected), DType.FP32)
    route_sum = ProgramValue("route.sum", (token,), DType.FP32)
    route_weight = ProgramValue("route.weight", (token, selected), DType.FP32)

    shared_gate = ProgramValue("shared.gate", (token, intermediate), DType.BF16)
    shared_up = ProgramValue("shared.up", (token, intermediate), DType.BF16)
    shared_hidden = ProgramValue("shared.hidden", (token, intermediate), DType.BF16)
    shared_output = ProgramValue("shared.output", (token, hidden), DType.BF16)

    edge_source = ProgramValue("edge.source", (token, selected, hidden), DType.BF16)
    selected_gate_weight = ProgramValue(
        "selected.gate_weight",
        (token, selected, intermediate, hidden),
        DType.BF16,
    )
    selected_up_weight = ProgramValue(
        "selected.up_weight",
        (token, selected, intermediate, hidden),
        DType.BF16,
    )
    selected_down_weight = ProgramValue(
        "selected.down_weight",
        (token, selected, hidden, intermediate),
        DType.BF16,
    )
    routed_gate = ProgramValue("routed.gate", (token, selected, intermediate), DType.BF16)
    routed_up = ProgramValue("routed.up", (token, selected, intermediate), DType.BF16)
    routed_hidden = ProgramValue("routed.hidden", (token, selected, intermediate), DType.BF16)
    routed_output = ProgramValue("routed.output", (token, selected, hidden), DType.BF16)
    weighted_output = ProgramValue("weighted.output", (token, selected, hidden), DType.FP32)
    returned_weighted_output = ProgramValue("returned.weighted_output", (token, selected, hidden), DType.FP32)
    routed_sum = ProgramValue("weighted.sum", (token, hidden), DType.FP32)
    routed_sum_bf16 = ProgramValue("weighted.sum_bf16", (token, hidden), DType.BF16)
    output = ProgramValue("output", (token, hidden), DType.BF16)

    operations = (
        ContractPrimitive("contract.router", (source, router_weight), logits, (hidden,), DType.FP32),
        SelectionPrimitive(
            "selection.routes",
            logits,
            selected_values,
            selected_indices,
            expert,
            selected_count,
            order=SelectionOrder.DESCENDING,
            comparison=SelectionComparison.TOTAL_ORDER,
            stable=True,
            tie_break=SelectionTieBreak.SOURCE_INDEX_ASCENDING,
            evaluation_order=EvaluationOrder.SOURCE_ORDERED,
        ),
        MapPrimitive(
            "map.selection_to_fp32",
            (selected_values,),
            selected_fp32,
            scalar_convert(
                scalar_input(selected_values.name),
                result_dtype=DType.FP32,
                semantics=ScalarConversionSemantics.EXACT,
            ),
        ),
        FoldPrimitive(
            "fold.route_maximum",
            selected_fp32,
            route_max,
            (selected,),
            FoldReducer.MAXIMUM,
            DType.FP32,
            EvaluationOrder.SOURCE_ORDERED,
        ),
        MapPrimitive(
            "map.route_exponential",
            (selected_fp32, route_max),
            route_exp,
            scalar_unary(
                ScalarExpressionKind.EXP,
                scalar_binary(
                    ScalarExpressionKind.SUBTRACT,
                    scalar_input(selected_fp32.name),
                    scalar_input(route_max.name),
                ),
            ),
        ),
        FoldPrimitive(
            "fold.route_sum",
            route_exp,
            route_sum,
            (selected,),
            FoldReducer.SUM,
            DType.FP32,
            EvaluationOrder.SOURCE_ORDERED,
        ),
        MapPrimitive(
            "map.route_weight",
            (route_exp, route_sum),
            route_weight,
            scalar_binary(
                ScalarExpressionKind.DIVIDE,
                scalar_input(route_exp.name),
                scalar_input(route_sum.name),
            ),
        ),
        ContractPrimitive(
            "contract.shared_gate",
            (source, shared_gate_weight),
            shared_gate,
            (hidden,),
            DType.FP32,
        ),
        ContractPrimitive(
            "contract.shared_up",
            (source, shared_up_weight),
            shared_up,
            (hidden,),
            DType.FP32,
        ),
        MapPrimitive(
            "map.shared_activation_pair",
            (shared_gate, shared_up),
            shared_hidden,
            _activation_pair_expression(shared_gate.name, shared_up.name),
        ),
        ContractPrimitive(
            "contract.shared_down",
            (shared_hidden, shared_down_weight),
            shared_output,
            (intermediate,),
            DType.FP32,
        ),
        TransportPrimitive(
            "transport.dispatch_source",
            source,
            edge_source,
            (selected_indices,),
            (token, selected, hidden),
            OperandIndexing(
                source,
                (token, selected, hidden),
                (index_axis(token), index_axis(hidden)),
            ),
            OperandIndexing(
                edge_source,
                (token, selected, hidden),
                (index_axis(token), index_axis(selected), index_axis(hidden)),
            ),
            order=EvaluationOrder.SOURCE_ORDERED,
        ),
        _selected_weight_relation(
            "relation.select_gate_weight",
            gate_weight,
            selected_indices,
            selected_gate_weight,
            (token, selected, intermediate, hidden),
        ),
        _selected_weight_relation(
            "relation.select_up_weight",
            up_weight,
            selected_indices,
            selected_up_weight,
            (token, selected, intermediate, hidden),
        ),
        _selected_weight_relation(
            "relation.select_down_weight",
            down_weight,
            selected_indices,
            selected_down_weight,
            (token, selected, hidden, intermediate),
        ),
        ContractPrimitive(
            "contract.routed_gate",
            (edge_source, selected_gate_weight),
            routed_gate,
            (hidden,),
            DType.FP32,
        ),
        ContractPrimitive(
            "contract.routed_up",
            (edge_source, selected_up_weight),
            routed_up,
            (hidden,),
            DType.FP32,
        ),
        MapPrimitive(
            "map.routed_activation_pair",
            (routed_gate, routed_up),
            routed_hidden,
            _activation_pair_expression(routed_gate.name, routed_up.name),
        ),
        ContractPrimitive(
            "contract.routed_down",
            (routed_hidden, selected_down_weight),
            routed_output,
            (intermediate,),
            DType.FP32,
        ),
        MapPrimitive(
            "map.weighted_output",
            (routed_output, route_weight),
            weighted_output,
            scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_convert(
                    scalar_input(routed_output.name),
                    result_dtype=DType.FP32,
                    semantics=ScalarConversionSemantics.EXACT,
                ),
                scalar_input(route_weight.name),
            ),
        ),
        TransportPrimitive(
            "transport.return_weighted_output",
            weighted_output,
            returned_weighted_output,
            (selected_indices,),
            (token, selected, hidden),
            OperandIndexing(
                weighted_output,
                (token, selected, hidden),
                (index_axis(token), index_axis(selected), index_axis(hidden)),
            ),
            OperandIndexing(
                returned_weighted_output,
                (token, selected, hidden),
                (index_axis(token), index_axis(selected), index_axis(hidden)),
            ),
            order=EvaluationOrder.SOURCE_ORDERED,
        ),
        FoldPrimitive(
            "fold.selected_outputs",
            returned_weighted_output,
            routed_sum,
            (selected,),
            FoldReducer.SUM,
            DType.FP32,
            EvaluationOrder.SOURCE_ORDERED,
        ),
        MapPrimitive(
            "map.routed_to_bf16",
            (routed_sum,),
            routed_sum_bf16,
            scalar_convert(
                scalar_input(routed_sum.name),
                result_dtype=DType.BF16,
                semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
            ),
        ),
        MapPrimitive(
            "map.output",
            (shared_output, routed_sum_bf16),
            output,
            scalar_binary(
                ScalarExpressionKind.ADD,
                scalar_input(shared_output.name),
                scalar_input(routed_sum_bf16.name),
            ),
        ),
    )
    return TensorProgram(
        inputs=(
            source,
            router_weight,
            shared_gate_weight,
            shared_up_weight,
            shared_down_weight,
            gate_weight,
            up_weight,
            down_weight,
        ),
        operations=operations,
        outputs=(output, selected_indices, route_weight),
        numerical_policy=numerical_policy,
    )


def expert_physical_request(
    program: TensorProgram,
    *,
    schedule: ExpertRouteSchedule,
) -> ExecutablePhysicalRequest:
    """Build a route-parametric SM90 task/EventTensor request from generic algebra."""
    if program.numerical_policy is not NumericalPolicy.SOURCE_ORDERED:
        raise ValueError("the reference expert schedule requires SOURCE_ORDERED numerical policy")
    axes = {axis.label: axis for value in (*program.inputs, *program.outputs) for axis in value.axes}
    token = axes["token"]
    selected = axes["selected"]
    expert = next(
        axis
        for operation in program.operations
        for value in primitive_outputs(operation)
        for axis in value.axes
        if axis.label == "expert"
    )

    selected_indices = next(
        value
        for operation in program.operations
        for value in primitive_outputs(operation)
        if value.name == "selection.indices"
    )
    task_specs = (
        ("route_selection", (token, selected), tuple(range(0, 7))),
        ("shared_contracts", (token,), tuple(range(7, 11))),
        ("routed_contracts", (expert, token, selected), tuple(range(11, 21))),
        ("return_and_finalize", (token,), tuple(range(21, 24))),
    )
    tasks = tuple(
        _reference_tiled_task(task_id, name, task_axes, ordinals, program, schedule=schedule)
        for task_id, (name, task_axes, ordinals) in enumerate(task_specs)
    )
    shared_to_final = TaskDependence(
        TaskRelation.from_pairs(
            tasks[1].family,
            tasks[3].family,
            tuple((((token_index,), (token_index,))) for token_index in range(token.extent)),
        ),
        MemoryVisibility(EventMemoryScope.DEVICE),
    )
    coordinate_program, token_coordinate, selected_coordinate, expert_coordinate = _route_coordinate_program(
        selected_indices,
        token=token,
        selected=selected,
    )
    value_ordinals = tensor_program_value_ordinals(program)
    binding = DependenceRuntimeBinding(selected_indices, value_ordinals[selected_indices])
    route_to_expert = TaskDependenceProgram(
        source_task_id=0,
        target_task_id=2,
        relation_program=coordinate_program,
        runtime_bindings=(binding,),
        source_coordinates=(
            BoundedTaskCoordinate(token_coordinate, 0, token.extent),
            BoundedTaskCoordinate(selected_coordinate, 0, selected.extent),
        ),
        target_coordinates=(
            BoundedTaskCoordinate(expert_coordinate, 0, expert.extent),
            BoundedTaskCoordinate(token_coordinate, 0, token.extent),
            BoundedTaskCoordinate(selected_coordinate, 0, selected.extent),
        ),
        validity=None,
        visibility=MemoryVisibility(EventMemoryScope.DEVICE),
    )
    expert_to_final = TaskDependenceProgram(
        source_task_id=2,
        target_task_id=3,
        relation_program=coordinate_program,
        runtime_bindings=(binding,),
        source_coordinates=(
            BoundedTaskCoordinate(expert_coordinate, 0, expert.extent),
            BoundedTaskCoordinate(token_coordinate, 0, token.extent),
            BoundedTaskCoordinate(selected_coordinate, 0, selected.extent),
        ),
        target_coordinates=(BoundedTaskCoordinate(token_coordinate, 0, token.extent),),
        validity=None,
        visibility=MemoryVisibility(EventMemoryScope.DEVICE),
    )
    task_plan = CanonicalTaskPlan(
        tasks,
        (shared_to_final,),
        dependence_programs=(route_to_expert, expert_to_final),
    )
    finite_event = derive_event_tensor_plan(
        shared_to_final,
        name="event.shared_to_final",
        memory_scope=EventMemoryScope.DEVICE,
        generation_policy=EventGenerationPolicy.PER_INVOCATION,
        scheduling_mode=EventSchedulingMode.STATIC,
    )
    finite_allocation = EventTensorAllocation(
        EventStorageBinding(finite_event.domain, tuple(range(token.extent)), (0,) * token.extent),
        capacity=token.extent,
        generation_count=1,
    )
    routed_event_count = expert.extent * token.extent * selected.extent
    event_programs = (
        _parametric_event(0, routed_event_count, schedule=schedule),
        _parametric_event(1, token.extent, schedule=schedule),
    )
    event_dataflow = EventDataflowProgram(task_plan.task_families, task_plan.dependences, (finite_event,))
    audit = verify_event_realizations(
        event_dataflow,
        (
            physical_event_realization(
                finite_event,
                mechanism="imperative_wait_notify",
                reason="the shared and routed results join at the final task without a stronger physical order",
            ),
        ),
    )
    schedules = tuple(_reference_schedule(task, schedule=schedule) for task in tasks)
    return ExecutablePhysicalRequest(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_plans=(finite_event,),
        event_allocations=(finite_allocation,),
        target=PhysicalTarget.SM90,
        numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED,
        schedules=schedules,
        event_realizations=EventRealizationProof(audit, ()),
        provenance={
            "source_boundary": "ordinary_jax",
            "differentiation_owner": "jax",
            "route_values": "runtime_inputs",
        },
        event_programs=event_programs,
    )


def generic_tensor_program_physical_request(
    program: TensorProgram,
    *,
    schedule: ExpertRouteSchedule,
) -> ExecutablePhysicalRequest:
    """Lower imported algebra through the common ownership/task/Event machinery."""
    numerical_policy = PhysicalNumericalPolicy(program.numerical_policy.value)
    axes = {
        axis.id: axis
        for value in (
            *program.inputs,
            *(result for operation in program.operations for result in primitive_outputs(operation)),
        )
        for axis in value.axes
    }
    tile_extent_by_axis = {axis_id: min(axis.extent, schedule.tile_rows) for axis_id, axis in axes.items()}
    task_plan = derive_composed_task_plan(
        program,
        tile_extent_by_axis=tile_extent_by_axis,
        numerical_policy=numerical_policy,
    )
    event_dataflow = eventize_composed_task_plan(task_plan)
    event_plans = event_dataflow.event_plans
    allocations = tuple(
        EventTensorAllocation(
            EventStorageBinding(
                plan.domain,
                tuple(range(len(plan.domain.coordinates))),
                (0,) * len(plan.domain.coordinates),
            ),
            capacity=len(plan.domain.coordinates),
            generation_count=1,
        )
        for plan in event_plans
    )
    audit = verify_event_realizations(
        event_dataflow,
        tuple(
            physical_event_realization(
                plan,
                mechanism="imperative_wait_notify",
                reason="the generic schedule retains the exact crossing task dependence",
            )
            for plan in event_plans
        ),
    )
    event_programs = tuple(
        _parametric_event(
            index,
            len(task_plan.tasks[dependence.target_task_id].family.coordinates),
            schedule=schedule,
        )
        for index, dependence in enumerate(task_plan.dependence_programs)
    )
    return ExecutablePhysicalRequest(
        algebra_record=program.canonical_record,
        algebra_fingerprint=program.semantic_fingerprint,
        task_plan=task_plan,
        event_plans=event_plans,
        event_allocations=allocations,
        target=PhysicalTarget.SM90,
        numerical_policy=numerical_policy,
        schedules=tuple(_reference_schedule(task, schedule=schedule) for task in task_plan.tasks),
        event_realizations=EventRealizationProof(audit, ()),
        provenance={
            "source_boundary": "ordinary_jax",
            "differentiation_owner": "jax",
            "task_decomposition": "ownership_and_dependency_convex_phase_dag",
        },
        event_programs=event_programs,
    )


def _route_coordinate_program(
    selected_indices: ProgramValue,
    *,
    token: TensorAxis,
    selected: TensorAxis,
) -> tuple[TensorProgram, ProgramValue, ProgramValue, ProgramValue]:
    token_coordinate = ProgramValue("coordinate.token", (token, selected), DType.INT32)
    selected_coordinate = ProgramValue("coordinate.selected", (token, selected), DType.INT32)
    expert_coordinate = ProgramValue("coordinate.expert", (token, selected), DType.INT32)
    relation = RelationPrimitive(
        "relation.route_task_coordinates",
        (selected_indices,),
        (token_coordinate, selected_coordinate, expert_coordinate),
        (token, selected),
        (
            OperandIndexing(
                selected_indices,
                (token, selected),
                (index_axis(token), index_axis(selected)),
            ),
        ),
        result_expressions=(
            index_axis(token),
            index_axis(selected),
            index_tensor_element(selected_indices, index_axis(token), index_axis(selected)),
        ),
        order=EvaluationOrder.ORDER_FREE,
    )
    coordinate_program = TensorProgram(
        (selected_indices,),
        (relation,),
        (token_coordinate, selected_coordinate, expert_coordinate),
        NumericalPolicy.SOURCE_ORDERED,
    )
    return coordinate_program, token_coordinate, selected_coordinate, expert_coordinate


def _reference_tiled_task(
    task_id: int,
    name: str,
    axes: tuple[TensorAxis, ...],
    operation_ordinals: tuple[int, ...],
    program: TensorProgram,
    *,
    schedule: ExpertRouteSchedule,
) -> TiledTask:
    operations = tuple(program.operations[ordinal] for ordinal in operation_ordinals)
    kinds = tuple(_task_operation_kind(operation) for operation in operations)
    capabilities = tuple(_task_capability(kind) for kind in kinds)
    phases = tuple(
        PhysicalPhase(
            index,
            capability,
            (ordinal,),
            *physical_phase_iteration_space(program.operations[ordinal]),
        )
        for index, (capability, ordinal) in enumerate(zip(capabilities, operation_ordinals, strict=True))
    )
    phase_edges = tuple(
        PhysicalPhaseEdge(index, index + 1, RelationAccessClass.IDENTITY) for index in range(len(phases) - 1)
    )
    task_axes = tuple(TaskAxis(axis.label or f"axis_{axis.id}", axis.extent) for axis in axes)
    tile_by_name = {
        "token": min(schedule.tile_rows, axes[0].extent) if axes else 1,
        "selected": 1,
        "expert": 1,
    }
    owner_axes = tuple(TaskAxisTile(axis.id, axis.extent, tile_by_name[axis.label]) for axis in axes)
    element_types = tuple(
        sorted({value.dtype.value for operation in operations for value in primitive_outputs(operation)})
    )
    structure = PhysicalTaskStructure(
        principal_kind=kinds[0],
        phases=phases,
        phase_edges=phase_edges,
        owner_axes=owner_axes,
        streamed_axes=(),
        reduction_axes=(),
        contraction_axes=(),
        traversal=TaskTraversal.SOURCE_ORDERED,
        element_types=element_types,
        layout_classes=("strided",),
        resources=TaskResourceEstimate(64, 0, max(1, len(phases) - 1)),
    )
    return TiledTask(task_id, TaskFamily(name, task_axes), operation_ordinals, kinds, structure, ())


def _task_operation_kind(operation) -> TaskOperationKind:
    if isinstance(operation, ContractPrimitive):
        return TaskOperationKind.CONTRACT
    if isinstance(operation, FoldPrimitive):
        return TaskOperationKind.FOLD
    if isinstance(operation, MapPrimitive):
        return TaskOperationKind.MAP
    if isinstance(operation, SelectionPrimitive):
        return TaskOperationKind.SELECTION
    if isinstance(operation, RelationPrimitive):
        return TaskOperationKind.RELATION
    if isinstance(operation, TransportPrimitive):
        return TaskOperationKind.TRANSPORT
    raise TypeError(f"unsupported routed task operation {type(operation).__name__}")


def _task_capability(kind: TaskOperationKind) -> TaskCapability:
    return {
        TaskOperationKind.MAP: TaskCapability.SCALAR_MAP,
        TaskOperationKind.FOLD: TaskCapability.ASSOCIATIVE_FOLD,
        TaskOperationKind.CONTRACT: TaskCapability.MATRIX_CONTRACT,
        TaskOperationKind.SELECTION: TaskCapability.BOUNDED_SELECTION,
        TaskOperationKind.RELATION: TaskCapability.INDEX_RELATION,
        TaskOperationKind.TRANSPORT: TaskCapability.DATA_TRANSPORT,
    }[kind]


def _parametric_event(
    dependence_program_index: int,
    event_count: int,
    *,
    schedule: ExpertRouteSchedule,
) -> ParametricEventTensorPlan:
    return ParametricEventTensorPlan(
        dependence_program_index=dependence_program_index,
        derivation=ParametricEventDerivation.EXACT_CONSUMER,
        capacity=event_count,
        slots=tuple(range(event_count)),
        generations=(0,) * event_count,
        generation_count=schedule.generation_count,
        memory_scope=EventMemoryScope.DEVICE,
        generation_policy=EventGenerationPolicy.PHASED,
        scheduling_mode=EventSchedulingMode.DYNAMIC,
        realization=ParametricEventRealization.IMPERATIVE_WAIT_NOTIFY,
    )


def _reference_schedule(task: TiledTask, *, schedule: ExpertRouteSchedule) -> PhysicalSchedule:
    structure = task.structure
    constraint = PhysicalTaskConstraint(
        principal_kind=structure.principal_kind,
        required_capabilities=tuple(phase.capability for phase in structure.phases),
        supported_capabilities=tuple(dict.fromkeys(phase.capability for phase in structure.phases)),
        required_edges=tuple(
            PhaseEdgeConstraint(
                structure.phases[edge.source_phase].capability,
                structure.phases[edge.target_phase].capability,
                edge.relation_class,
            )
            for edge in structure.phase_edges
        ),
        owner_rank=len(structure.owner_axes),
        streamed_rank=len(structure.streamed_axes),
        reduction_rank=len(structure.reduction_axes),
        contraction_rank=len(structure.contraction_axes),
        supported_relation_classes=tuple(RelationAccessClass),
        supported_element_types=structure.element_types,
        supported_layout_classes=structure.layout_classes,
        maximum_resources=structure.resources,
    )
    return PhysicalSchedule(
        PhysicalScheduleFamily("generic_relation_contract_pipeline", PhysicalTarget.SM90, (constraint,)),
        (task.id,),
        {
            "tile_rows": schedule.tile_rows,
            "macrobatch_rows": schedule.macrobatch_rows,
            "phase_count": schedule.phase_count,
            "generation_count": schedule.generation_count,
            "traversal": structure.traversal.value,
        },
    )


def _selected_weight_relation(
    name: str,
    weight: ProgramValue,
    selected_indices: ProgramValue,
    output: ProgramValue,
    domain_axes: tuple[TensorAxis, ...],
) -> RelationPrimitive:
    token, selected = domain_axes[:2]
    expert_index = index_tensor_element(selected_indices, index_axis(token), index_axis(selected))
    weight_indices = (expert_index, *(index_axis(axis) for axis in domain_axes[2:]))
    return RelationPrimitive(
        name,
        (weight, selected_indices),
        (output,),
        domain_axes,
        (
            OperandIndexing(weight, domain_axes, weight_indices),
            OperandIndexing(
                selected_indices,
                domain_axes,
                (index_axis(token), index_axis(selected)),
            ),
        ),
        result_sources=(weight,),
        order=EvaluationOrder.ORDER_FREE,
    )


def _activation_pair_expression(gate_name: str, up_name: str):
    gate = scalar_input(gate_name)
    exponential = _round_bf16(scalar_unary(ScalarExpressionKind.EXP, scalar_unary(ScalarExpressionKind.NEGATE, gate)))
    denominator = _round_bf16(
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_constant(1.0),
            exponential,
        )
    )
    sigmoid_product = _round_bf16(
        scalar_binary(
            ScalarExpressionKind.DIVIDE,
            gate,
            denominator,
        )
    )
    return _round_bf16(
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            sigmoid_product,
            scalar_input(up_name),
        )
    )


def _round_bf16(expression):
    return scalar_convert(
        expression,
        result_dtype=DType.BF16,
        semantics=ScalarConversionSemantics.ROUND_NEAREST_EVEN,
    )


def derive_expert_route_dataflow(
    expert_indices: np.ndarray,
    route_weights: np.ndarray,
    *,
    hidden_dimension: int,
    schedule: ExpertRouteSchedule,
) -> ExpertRouteDataflow:
    """Derive exact primal/reverse tasks from selected indices and an index placement."""
    indices = np.asarray(expert_indices)
    weights = np.asarray(route_weights, dtype=np.float32)
    if indices.ndim != 2:
        raise ValueError("selected expert indices must have [source_item, route_slot] shape")
    if weights.shape != indices.shape:
        raise ValueError("route weights must match selected-index shape")
    if indices.shape[0] != schedule.source_placement.destination_count:
        raise ValueError("source placement must cover every routed source item")
    if hidden_dimension <= 0:
        raise ValueError("hidden dimension must be positive")

    destination_rank = np.asarray(schedule.destination_placement.rank_by_item, dtype=np.int32)
    destination_local = np.asarray(schedule.destination_placement.local_item_by_item, dtype=np.int32)
    route_count = indices.size
    relation = build_fixed_capacity_relation_plan(
        indices,
        weights,
        destination_rank_by_item=destination_rank,
        destination_local_item_by_item=destination_local,
        source_placement=schedule.source_placement,
        destination_capacity=route_count,
    )
    template = _transport_template(relation, schedule=schedule)
    epoch = bind_transport_epoch(
        template.epoch_protocol,
        0,
        completed_epochs=frozenset(),
        completed_resets=frozenset(),
    )
    source_rank = np.asarray(schedule.source_placement.rank_by_item, dtype=np.int32)
    source_local = np.asarray(schedule.source_placement.local_item_by_item, dtype=np.int32)
    primal_source = TransportPayloadField(
        "source.activation",
        TransportPayloadDomain.SOURCE_ITEM,
        (hidden_dimension,),
        DType.BF16,
    )
    primal_weight = TransportPayloadField(
        "relation.weight",
        TransportPayloadDomain.RELATION_EDGE,
        (),
        DType.FP32,
    )
    primal_result = TransportPayloadField(
        "relation.output",
        TransportPayloadDomain.RELATION_EDGE,
        (hidden_dimension,),
        DType.BF16,
    )
    output_cotangent = TransportPayloadField(
        "source.output_cotangent",
        TransportPayloadDomain.SOURCE_ITEM,
        (hidden_dimension,),
        DType.BF16,
    )
    input_cotangent = TransportPayloadField(
        "relation.input_cotangent",
        TransportPayloadDomain.RELATION_EDGE,
        (hidden_dimension,),
        DType.BF16,
    )
    weight_cotangent = TransportPayloadField(
        "relation.weight_cotangent",
        TransportPayloadDomain.RELATION_EDGE,
        (),
        DType.FP32,
    )
    transport = derive_relation_transport_training_plan(
        relation,
        template=template,
        source_rank_by_item=source_rank,
        source_local_item_by_item=source_local,
        epoch=epoch,
        primal_dispatch_fields=(primal_source, primal_weight),
        primal_return_fields=(primal_result,),
        cotangent_dispatch_fields=(output_cotangent,),
        cotangent_return_fields=(input_cotangent, weight_cotangent),
    )
    forward = derive_relation_tile_pipeline(
        transport.runtime,
        dispatch_flows=transport.primal.dispatch_flows,
        return_flows=transport.primal.return_flows,
        graph=_linear_tile_graph(
            (
                "contract.gate_up",
                "map.activation_pair",
                "contract.down",
                "map.route_weight",
            ),
            split_after="map.activation_pair",
        ),
        epoch=epoch,
    )
    reverse = derive_relation_tile_pipeline(
        transport.runtime,
        dispatch_flows=transport.cotangent.dispatch_flows,
        return_flows=transport.cotangent.return_flows,
        graph=_linear_tile_graph(
            (
                "map.output_adjoint",
                "contract.down_adjoint",
                "map.activation_pair_vjp",
                "contract.gate_up_adjoint",
                "fold.route_weight_adjoint",
            ),
            split_after="map.activation_pair_vjp",
        ),
        epoch=epoch,
    )
    return ExpertRouteDataflow(relation, transport, forward, reverse)


def execute_expert_route_forward(
    plan: ExpertRouteDataflow,
    source: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
) -> np.ndarray:
    """Execute the primal relation and task dataflow as an independent CPU oracle."""
    source = np.asarray(source, dtype=np.float32)
    gate_weight = np.asarray(gate_weight, dtype=np.float32)
    up_weight = np.asarray(up_weight, dtype=np.float32)
    down_weight = np.asarray(down_weight, dtype=np.float32)
    hidden_dimension = plan.transport.primal.dispatch_flows[0].field.trailing_shape
    expected_weight_shape = (
        plan.relation.destination_count,
        gate_weight.shape[1],
        hidden_dimension[0],
    )
    if source.shape != (plan.relation.source_item_count, hidden_dimension[0]):
        raise ValueError("source activation shape disagrees with the routed task plan")
    if gate_weight.shape != expected_weight_shape or up_weight.shape != expected_weight_shape:
        raise ValueError("gate and up weights must have [destination, intermediate, hidden] shape")
    if down_weight.shape != (
        plan.relation.destination_count,
        hidden_dimension[0],
        gate_weight.shape[1],
    ):
        raise ValueError("down weights must have [destination, hidden, intermediate] shape")

    source_flow = next(flow for flow in plan.transport.primal.dispatch_flows if flow.field.name == "source.activation")
    weight_flow = next(flow for flow in plan.transport.primal.dispatch_flows if flow.field.name == "relation.weight")
    result_flow = plan.transport.primal.return_flows[0]
    destination_source = execute_transport_field_flow_reference(plan.transport.runtime, source_flow, source)
    route_weights = plan.relation.weight.astype(np.float32, copy=False)
    destination_weight = execute_transport_field_flow_reference(plan.transport.runtime, weight_flow, route_weights)
    destination_item = plan.transport.runtime.destination_row_destination_item
    gate = np.einsum("rh,rih->ri", destination_source, gate_weight[destination_item])
    up = np.einsum("rh,rih->ri", destination_source, up_weight[destination_item])
    hidden = (gate / (1.0 + np.exp(-gate))) * up
    output = np.einsum("ri,rhi->rh", hidden, down_weight[destination_item])
    weighted = output * destination_weight[:, None]
    source_edges = execute_transport_field_flow_reference(plan.transport.runtime, result_flow, weighted)
    return np.sum(source_edges, axis=1, dtype=np.float32)


def _linear_tile_graph(names: tuple[str, ...], *, split_after: str) -> TilePipelineGraph:
    return TilePipelineGraph(
        stages=tuple(TilePipelineStage(name) for name in names),
        edges=tuple(TilePipelineEdge(source, target) for source, target in pairwise(names)),
        entry_stage=names[0],
        exit_stage=names[-1],
        tile_group_split_after=split_after,
    )


def _transport_template(relation: RelationPlan, *, schedule: ExpertRouteSchedule) -> RelationTransportTemplate:
    rank_count = schedule.destination_placement.rank_count
    source_counts = schedule.source_placement.item_count_by_rank
    source_edge_capacity = tuple(count * relation.route_slots for count in source_counts)
    exact_capacity = tuple(
        tuple(source_edge_capacity[source] for _ in range(rank_count)) for source in range(rank_count)
    )
    coalesced_capacity = tuple(tuple(source_counts[source] for _ in range(rank_count)) for source in range(rank_count))
    per_destination_capacity = relation.source_item_count * relation.route_slots
    return RelationTransportTemplate(
        world_rank_count=rank_count,
        source_rank_count=rank_count,
        source_item_capacity_by_rank=source_counts,
        destination_rank_by_item=schedule.destination_placement.rank_by_item,
        destination_local_item_by_item=schedule.destination_placement.local_item_by_item,
        destination_row_capacity_by_item=(per_destination_capacity,) * relation.destination_count,
        coalesced_capacity_by_rank_pair=coalesced_capacity,
        exact_edge_capacity_by_rank_pair=exact_capacity,
        capacity_mode=TransportCapacityMode.FIXED,
        tile_rows=schedule.tile_rows,
        macrobatch_rows=schedule.macrobatch_rows,
        epoch_protocol=TransportEpochProtocol(schedule.phase_count, schedule.generation_count),
    )


def _tile_pipeline_record(
    plan: RelationTilePipelinePlan,
    *,
    include_events: bool,
    phase_count: int,
    generation_count: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_families": tuple(
            {
                "name": family.name,
                "axes": tuple({"name": axis.name, "extent": axis.extent} for axis in family.axes),
                "placement": family.placement,
            }
            for family in plan.program.task_families
        ),
        "dependences": tuple(
            {
                "source": dependence.relation.source.name,
                "target": dependence.relation.target.name,
                "pairs": tuple({"source": pair.source, "target": pair.target} for pair in dependence.relation.pairs),
                "visibility": {
                    "scope": dependence.visibility.scope.value,
                    "release": dependence.visibility.release_on_notify,
                    "acquire": dependence.visibility.acquire_before_consumer,
                },
            }
            for dependence in plan.program.dependences
        ),
        "kernelization_candidates": tuple(
            {"policy": candidate.policy.value, "groups": candidate.groups} for candidate in plan.kernelization_candidates
        ),
        "active_tasks": plan.tile_domain.active_by_task,
    }
    if include_events:
        record["event_tensors"] = _event_records(
            plan,
            phase_count=phase_count,
            generation_count=generation_count,
        )
    return record


def _static_tile_pipeline_record(plan: RelationTilePipelinePlan) -> dict[str, Any]:
    """Describe a bounded task template without invocation-specific relation pairs."""
    family_ordinal = {family: ordinal for ordinal, family in enumerate(plan.program.task_families)}
    return {
        "task_families": tuple(
            {
                "axis_extents": tuple(axis.extent for axis in family.axes),
                "placement": family.placement,
            }
            for family in plan.program.task_families
        ),
        "dependence_domains": tuple(
            {
                "source_task": family_ordinal[dependence.relation.source],
                "target_task": family_ordinal[dependence.relation.target],
                "scope": dependence.visibility.scope.value,
                "release": dependence.visibility.release_on_notify,
                "acquire": dependence.visibility.acquire_before_consumer,
            }
            for dependence in plan.program.dependences
        ),
        "kernelization_candidates": tuple(
            {
                "policy": candidate.policy.value,
                "group_sizes": tuple(len(group) for group in candidate.groups),
            }
            for candidate in plan.kernelization_candidates
        ),
        "padding_is_masked": plan.tile_domain.padding_is_masked,
    }


def _static_event_records(plan: RelationTilePipelinePlan) -> tuple[dict[str, Any], ...]:
    family_ordinal = {family: ordinal for ordinal, family in enumerate(plan.program.task_families)}
    return tuple(
        {
            "producer_task": family_ordinal[event.required_dependence.relation.source],
            "consumer_task": family_ordinal[event.required_dependence.relation.target],
            "event_extents": tuple(axis.extent for axis in event.domain.axes),
            "memory_scope": event.memory_scope.value,
            "generation_policy": event.generation_policy.value,
            "scheduling_mode": None if event.scheduling_mode is None else event.scheduling_mode.value,
        }
        for event in plan.program.event_plans
    )


def _event_records(
    plan: RelationTilePipelinePlan,
    *,
    phase_count: int,
    generation_count: int,
) -> tuple[dict[str, Any], ...]:
    owned_by_name = {owned.plan.name: owned for owned in plan.owned_events}
    return tuple(
        {
            "plan": event_tensor_plan_record(event_plan),
            "owner_rank_by_event": owned_by_name[event_plan.name].owner_rank_by_event,
            "owner_placement": owned_by_name[event_plan.name].owner_placement,
            "storage_namespace": owned_by_name[event_plan.name].storage_namespace,
            "storage_slots": owned_by_name[event_plan.name].runtime_inputs.event_storage_slots,
            "generations": owned_by_name[event_plan.name].runtime_inputs.event_generations,
            "storage_capacity": len(event_plan.domain.coordinates) * phase_count,
            "generation_count": generation_count,
        }
        for event_plan in plan.program.event_plans
    )


def _transport_template_record(template: RelationTransportTemplate) -> dict[str, Any]:
    return {
        "world_rank_count": template.world_rank_count,
        "source_item_capacity_by_rank": template.source_item_capacity_by_rank,
        "destination_rank_by_item": template.destination_rank_by_item,
        "destination_local_item_by_item": template.destination_local_item_by_item,
        "destination_row_capacity_by_item": template.destination_row_capacity_by_item,
        "coalesced_capacity_by_rank_pair": template.coalesced_capacity_by_rank_pair,
        "exact_edge_capacity_by_rank_pair": template.exact_edge_capacity_by_rank_pair,
        "capacity_mode": template.capacity_mode.value,
        "tile_rows": template.tile_rows,
        "macrobatch_rows": template.macrobatch_rows,
        "phase_count": template.epoch_protocol.phase_count,
        "generation_count": template.epoch_protocol.generation_modulus,
    }


def _fingerprint(record: dict[str, Any]) -> str:
    serialized = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()
