# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime.event_buffering import (
    EventRealizationAudit,
    EventRealizationKind,
    derive_bounded_buffer_plan,
    erased_event_realization,
    physical_event_realization,
)
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventGenerationPolicy,
    EventMemoryScope,
    EventSchedulingMode,
    EventStorageBinding,
    ImperativeEventOpKind,
    IndexPair,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
)
from tile_lifetime.physical_task_contract import (
    BoundedTaskCoordinate,
    CanonicalTaskPlan,
    DependenceRuntimeBinding,
    EventErasureWitness,
    EventRealizationProof,
    EventTensorAllocation,
    ExecutablePhysicalRequest,
    ParametricEventDerivation,
    ParametricEventTensorPlan,
    PhaseEdgeConstraint,
    PhysicalNumericalPolicy,
    PhysicalOrderKind,
    PhysicalPhase,
    PhysicalPhaseEdge,
    PhysicalSchedule,
    PhysicalScheduleFamily,
    PhysicalTarget,
    PhysicalTaskConstraint,
    PhysicalTaskOrder,
    PhysicalTaskStructure,
    RelationAccessClass,
    TaskAxisTile,
    TaskCapability,
    TaskDependenceProgram,
    TaskOperationKind,
    TaskResourceEstimate,
    TaskTraversal,
    TaskValueAccess,
    TaskValueAccessKind,
    TiledTask,
    canonical_fingerprint,
    install_physical_schedule_cover,
    instantiate_parametric_event_tensor,
    instantiate_task_dependence,
    lower_executable_event_operations,
    physical_schedule_is_eligible,
)
from tile_lifetime.tensor_program import (
    EvaluationOrder,
    IndexExpressionKind,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    TensorAxis,
    TensorProgram,
    index_axis,
    index_binary,
    index_constant,
)


def _request(
    *,
    label: str,
    expression: str = "identity",
    access_offset: int = 0,
) -> ExecutablePhysicalRequest:
    producer = TaskFamily(f"{label}.producer", (TaskAxis(f"{label}.row", 2),), placement="device")
    consumer = TaskFamily(f"{label}.consumer", (TaskAxis(f"{label}.output", 2),), placement="device")
    fused_structure = PhysicalTaskStructure(
        principal_kind=TaskOperationKind.CONTRACT,
        phases=(
            PhysicalPhase(0, TaskCapability.MATRIX_CONTRACT, (0,), (128, 2048, 128), (2,)),
            PhysicalPhase(1, TaskCapability.ASSOCIATIVE_FOLD, (1,), (128, 2048), (1,)),
            PhysicalPhase(2, TaskCapability.SCALAR_MAP, (2,), (128,), ()),
            PhysicalPhase(3, TaskCapability.MATRIX_CONTRACT, (3,), (128, 2048, 128), (2,)),
        ),
        phase_edges=(
            PhysicalPhaseEdge(0, 1, RelationAccessClass.IDENTITY),
            PhysicalPhaseEdge(1, 2, RelationAccessClass.IDENTITY),
            PhysicalPhaseEdge(2, 3, RelationAccessClass.IDENTITY),
        ),
        owner_axes=(TaskAxisTile(0, 128, 64),),
        streamed_axes=(TaskAxisTile(1, 2048, 128),),
        reduction_axes=(),
        contraction_axes=(TaskAxisTile(2, 128, 128),),
        traversal=TaskTraversal.FORWARD,
        element_types=("bf16", "f32"),
        layout_classes=("row_major",),
        resources=TaskResourceEstimate(192, 96 * 1024, 4),
    )
    producer_task = TiledTask(
        0,
        producer,
        (0, 1, 2, 3),
        (
            TaskOperationKind.CONTRACT,
            TaskOperationKind.FOLD,
            TaskOperationKind.MAP,
            TaskOperationKind.CONTRACT,
        ),
        fused_structure,
        (
            TaskValueAccess(
                0,
                TaskValueAccessKind.LIVE_IN,
                0,
                RelationAccessClass.AFFINE,
                {"kind": "affine", "offset": access_offset},
            ),
            TaskValueAccess(4, TaskValueAccessKind.LIVE_OUT, 3, RelationAccessClass.IDENTITY, {"kind": "identity"}),
        ),
    )
    consumer_structure = PhysicalTaskStructure(
        principal_kind=TaskOperationKind.MAP,
        phases=(PhysicalPhase(0, TaskCapability.SCALAR_MAP, (4,), (128,), ()),),
        phase_edges=(),
        owner_axes=(TaskAxisTile(0, 128, 64),),
        streamed_axes=(),
        reduction_axes=(),
        contraction_axes=(),
        traversal=TaskTraversal.UNORDERED,
        element_types=("bf16",),
        layout_classes=("row_major",),
        resources=TaskResourceEstimate(24, 0, 0),
    )
    consumer_task = TiledTask(
        1,
        consumer,
        (4,),
        (TaskOperationKind.MAP,),
        consumer_structure,
        (
            TaskValueAccess(4, TaskValueAccessKind.LIVE_IN, 0, RelationAccessClass.IDENTITY, {"kind": "identity"}),
            TaskValueAccess(5, TaskValueAccessKind.LIVE_OUT, 0, RelationAccessClass.IDENTITY, {"kind": "identity"}),
        ),
    )
    relation = TaskRelation(producer, consumer, tuple(IndexPair((index,), (index,)) for index in range(2)))
    visibility = MemoryVisibility(EventMemoryScope.DEVICE)
    dependence = TaskDependence(relation, visibility)
    event = derive_event_tensor_plan(dependence, name=f"{label}.ready")
    program = EventDataflowProgram((producer, consumer), (dependence,), (event,))
    bounded = derive_bounded_buffer_plan(
        name=f"{label}.buffer",
        program=program,
        producer=producer,
        uses=(relation,),
        capacity=1,
        allocation_owner_for={(index,): (index,) for index in range(2)},
        slot_for={(index,): 0 for index in range(2)},
        generation_for={(index,): 0 for index in range(2)},
        visibility=visibility,
    )
    plan = CanonicalTaskPlan((producer_task, consumer_task), (dependence,), (bounded,))
    map_constraint = PhysicalTaskConstraint(
        TaskOperationKind.MAP,
        (TaskCapability.SCALAR_MAP,),
        (TaskCapability.SCALAR_MAP,),
        (),
        1,
        0,
        0,
        0,
        (RelationAccessClass.IDENTITY,),
        ("bf16",),
        ("row_major",),
        TaskResourceEstimate(64, 0, 0),
    )
    fused_family = PhysicalScheduleFamily(
        "sm90_contract_fold_map_contract",
        PhysicalTarget.SM90,
        (
            PhysicalTaskConstraint(
                TaskOperationKind.CONTRACT,
                (
                    TaskCapability.MATRIX_CONTRACT,
                    TaskCapability.MATRIX_CONTRACT,
                    TaskCapability.ASSOCIATIVE_FOLD,
                    TaskCapability.SCALAR_MAP,
                ),
                (
                    TaskCapability.MATRIX_CONTRACT,
                    TaskCapability.ASSOCIATIVE_FOLD,
                    TaskCapability.SCALAR_MAP,
                ),
                (
                    PhaseEdgeConstraint(
                        TaskCapability.ASSOCIATIVE_FOLD,
                        TaskCapability.SCALAR_MAP,
                        RelationAccessClass.IDENTITY,
                    ),
                    PhaseEdgeConstraint(
                        TaskCapability.SCALAR_MAP,
                        TaskCapability.MATRIX_CONTRACT,
                        RelationAccessClass.IDENTITY,
                    ),
                ),
                1,
                1,
                0,
                1,
                (RelationAccessClass.IDENTITY, RelationAccessClass.AFFINE),
                ("bf16", "f32"),
                ("row_major",),
                TaskResourceEstimate(255, 228 * 1024, 16),
            ),
            map_constraint,
        ),
    )
    schedules = (
        PhysicalSchedule(
            fused_family,
            (0, 1),
            {"stages": 3, "threads": 384},
            (
                PhysicalTaskOrder(
                    PhysicalOrderKind.PROGRAM_ORDER,
                    0,
                    1,
                    relation,
                    {"kind": "launch_phase_order", "producer_phase": 0, "consumer_phase": 1},
                ),
            ),
        ),
    )
    realization = erased_event_realization(
        event,
        kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
        mechanism="same_launch_phase_order",
        reason=f"{label} producer phase precedes consumer phase",
        ordering=relation,
    )
    proof = EventRealizationProof(
        EventRealizationAudit((realization,)),
        (
            EventErasureWitness(
                event.name,
                0,
                0,
            ),
        ),
    )
    allocation = EventTensorAllocation(
        EventStorageBinding(event.domain, (0, 1), (0, 0)),
        capacity=2,
        generation_count=1,
    )
    algebra = {
        "schema_version": 1,
        "operations": (
            {"kind": "contract"},
            {"kind": "fold"},
            {"kind": "map", "expression": expression},
            {"kind": "contract"},
            {"kind": "map", "expression": "identity"},
        ),
    }
    return ExecutablePhysicalRequest(
        algebra,
        canonical_fingerprint(algebra),
        plan,
        (event,),
        (allocation,),
        PhysicalTarget.SM90,
        PhysicalNumericalPolicy.FAST,
        schedules,
        proof,
        {"source_origin": label, "toolchain": "test"},
    )


def test_equivalent_origins_share_task_request_and_schedule_family_identity() -> None:
    first = _request(label="first")
    renamed = _request(label="renamed")

    assert first.task_plan.fingerprint == renamed.task_plan.fingerprint
    assert first.semantic_fingerprint == renamed.semantic_fingerprint
    assert tuple(schedule.family.fingerprint for schedule in first.schedules) == tuple(
        schedule.family.fingerprint for schedule in renamed.schedules
    )
    assert all(physical_schedule_is_eligible(schedule, renamed.task_plan) for schedule in first.schedules)
    assert first.artifact_fingerprint != renamed.artifact_fingerprint


def test_scalar_and_index_ast_changes_affect_request_not_schedule_family() -> None:
    identity = _request(label="same", expression="identity")
    nonlinear = _request(label="same", expression="tanh")
    remapped = _request(label="same", expression="identity", access_offset=1)

    assert identity.semantic_fingerprint != nonlinear.semantic_fingerprint
    assert identity.task_plan.fingerprint == nonlinear.task_plan.fingerprint
    assert identity.semantic_fingerprint != remapped.semantic_fingerprint
    assert identity.task_plan.fingerprint != remapped.task_plan.fingerprint
    assert (
        tuple(schedule.family.fingerprint for schedule in identity.schedules)
        == tuple(schedule.family.fingerprint for schedule in nonlinear.schedules)
        == tuple(schedule.family.fingerprint for schedule in remapped.schedules)
    )
    assert all(physical_schedule_is_eligible(schedule, remapped.task_plan) for schedule in remapped.schedules)


def test_source_ordered_request_rejects_an_unordered_task() -> None:
    fast = _request(label="policy")

    with pytest.raises(ValueError, match="SOURCE_ORDERED request cannot contain unordered task 1"):
        replace(fast, numerical_policy=PhysicalNumericalPolicy.SOURCE_ORDERED)


def test_schedule_capability_ceiling_accepts_supported_maps_and_rejects_unsupported_phases() -> None:
    request = _request(label="coverage")
    task = request.task_plan.tasks[0]

    def with_extra_phase(capability: TaskCapability, kind: TaskOperationKind) -> CanonicalTaskPlan:
        phase = PhysicalPhase(len(task.structure.phases), capability, (10,), (128,), ())
        structure = replace(task.structure, phases=(*task.structure.phases, phase))
        extended = replace(
            task,
            operation_ordinals=(*task.operation_ordinals, 10),
            operation_kinds=(*task.operation_kinds, kind),
            structure=structure,
        )
        return replace(request.task_plan, tasks=(extended, request.task_plan.tasks[1]))

    extra_map = with_extra_phase(TaskCapability.SCALAR_MAP, TaskOperationKind.MAP)
    unsupported_relation = with_extra_phase(TaskCapability.INDEX_RELATION, TaskOperationKind.RELATION)

    assert physical_schedule_is_eligible(request.schedules[0], extra_map)
    assert not physical_schedule_is_eligible(request.schedules[0], unsupported_relation)
    original_schedule = request.schedules[0]
    expanded_constraint = replace(
        original_schedule.family.tasks[0],
        supported_capabilities=(
            *original_schedule.family.tasks[0].supported_capabilities,
            TaskCapability.INDEX_RELATION,
        ),
    )
    expanded_family = replace(
        original_schedule.family,
        tasks=(expanded_constraint, original_schedule.family.tasks[1]),
    )
    expanded_schedule = replace(original_schedule, family=expanded_family)
    assert expanded_family.fingerprint != original_schedule.family.fingerprint
    assert physical_schedule_is_eligible(expanded_schedule, unsupported_relation)


def test_schedule_cover_installation_erases_only_a_fully_covered_event_relation() -> None:
    request = _request(label="installation")
    fused = install_physical_schedule_cover(request, request.schedules)

    assert fused.event_realizations.audit.entries[0].kind is EventRealizationKind.ERASED_PROGRAM_ORDER
    assert not lower_executable_event_operations(fused)[0].retained_operations

    source_schedule = request.schedules[0]
    singleton_schedules = tuple(
        PhysicalSchedule(
            PhysicalScheduleFamily(
                f"sm90_single_task_{task_id}",
                PhysicalTarget.SM90,
                (source_schedule.family.tasks[task_id],),
            ),
            (task_id,),
            {"launches": 1},
        )
        for task_id in range(2)
    )
    split = install_physical_schedule_cover(request, singleton_schedules)

    assert split.event_realizations.audit.entries[0].kind is EventRealizationKind.PHYSICAL
    assert lower_executable_event_operations(split)[0].retained_operations

    order = source_schedule.enforced_orders[0]
    partial_relation = TaskRelation(order.relation.source, order.relation.target, order.relation.pairs[:1])
    partial_schedule = replace(
        source_schedule,
        enforced_orders=(replace(order, relation=partial_relation),),
    )
    partial = install_physical_schedule_cover(request, (partial_schedule,))

    assert partial.event_realizations.audit.entries[0].kind is EventRealizationKind.PHYSICAL


def test_provenance_changes_only_artifact_identity() -> None:
    request = _request(label="same")
    changed = replace(request, provenance={"source_origin": "same", "toolchain": "new"})

    assert request.semantic_fingerprint == changed.semantic_fingerprint
    assert request.artifact_fingerprint != changed.artifact_fingerprint


def test_erased_event_requires_covering_physical_order_witness() -> None:
    request = _request(label="same")
    proof = replace(request.event_realizations, erasure_witnesses=())

    with pytest.raises(ValueError, match="cover every erased Event"):
        replace(request, event_realizations=proof)


def test_physical_event_requires_no_erasure_witness_and_remains_in_request() -> None:
    request = _request(label="same")
    event = request.event_plans[0]
    retained = replace(
        request,
        event_realizations=EventRealizationProof(
            EventRealizationAudit(
                (physical_event_realization(event, mechanism="mbarrier", reason="cross-owner visibility"),)
            ),
            (),
        ),
    )

    realization = retained.canonical_record["event_realizations"]
    assert isinstance(realization, dict)
    assert realization["audit"] == ({"event": 0, "kind": "physical", "mechanism": "mbarrier"},)
    assert realization["erasure_witnesses"] == ()

    lowering = lower_executable_event_operations(retained)[0]
    assert lowering.retained_operations == lowering.imperative_operations
    assert {operation.kind for operation in lowering.retained_operations} == {
        ImperativeEventOpKind.INITIALIZE,
        ImperativeEventOpKind.WAIT,
        ImperativeEventOpKind.NOTIFY,
    }


def test_verified_program_order_erases_storage_bound_imperative_event_operations() -> None:
    request = _request(label="same")

    lowering = lower_executable_event_operations(request)[0]

    assert lowering.retained_operations == ()
    assert lowering.erasure_proof == {
        "kind": "selected_physical_order",
        "schedule_index": 0,
        "order_index": 0,
        "order_kind": "program_order",
        "mechanism": {"kind": "launch_phase_order", "producer_phase": 0, "consumer_phase": 1},
    }
    assert {operation.storage_slot for operation in lowering.imperative_operations} == {0, 1}
    assert {operation.generation for operation in lowering.imperative_operations} == {0}
    assert {operation.kind for operation in lowering.imperative_operations} == {
        ImperativeEventOpKind.INITIALIZE,
        ImperativeEventOpKind.WAIT,
        ImperativeEventOpKind.NOTIFY,
    }


def test_task_plan_rejects_duplicate_family_display_names() -> None:
    request = _request(label="same")
    first, second = request.task_plan.tasks
    duplicate = replace(second, family=replace(second.family, name=first.family.name))

    with pytest.raises(ValueError, match="display names must be unique"):
        CanonicalTaskPlan((first, duplicate), (), ())


def test_parametric_dependence_keeps_runtime_routes_out_of_semantic_identity() -> None:
    base = _request(label="dynamic")
    edge = TensorAxis(0, 4)
    route = ProgramValue("selected_expert", (edge,), DType.INT32)
    valid = ProgramValue("selected_valid", (edge,), DType.BOOL)
    source_coordinate = ProgramValue("source_task", (edge,), DType.INT32)
    target_coordinate = ProgramValue("target_task", (edge,), DType.INT32)
    output_validity = ProgramValue("edge_valid", (edge,), DType.BOOL)
    edge_index = index_axis(edge)
    relation = RelationPrimitive(
        "runtime route relation",
        (route, valid),
        (source_coordinate, target_coordinate, output_validity),
        (edge,),
        (
            OperandIndexing(route, (edge,), (edge_index,)),
            OperandIndexing(valid, (edge,), (edge_index,)),
        ),
        (
            index_binary(IndexExpressionKind.FLOOR_DIVIDE, edge_index, index_constant(2)),
            None,
            None,
        ),
        (None, route, valid),
        order=EvaluationOrder.SOURCE_ORDERED,
    )
    relation_program = TensorProgram(
        (route, valid),
        (relation,),
        (source_coordinate, target_coordinate, output_validity),
        NumericalPolicy.SOURCE_ORDERED,
    )
    dependence_program = TaskDependenceProgram(
        0,
        1,
        relation_program,
        (DependenceRuntimeBinding(route, 10), DependenceRuntimeBinding(valid, 11)),
        (BoundedTaskCoordinate(source_coordinate, 0, 2),),
        (BoundedTaskCoordinate(target_coordinate, 0, 2),),
        output_validity,
        MemoryVisibility(EventMemoryScope.DEVICE),
    )
    task_plan = CanonicalTaskPlan(base.task_plan.tasks, (), (), (dependence_program,))
    event_program = ParametricEventTensorPlan(
        0,
        ParametricEventDerivation.EXACT_CONSUMER,
        2,
        (0, 1),
        (0, 0),
        1,
        EventMemoryScope.DEVICE,
        EventGenerationPolicy.PER_INVOCATION,
        base.event_plans[0].scheduling_mode or EventSchedulingMode.STATIC,
    )
    request = replace(
        base,
        task_plan=task_plan,
        event_plans=(),
        event_allocations=(),
        event_realizations=EventRealizationProof(EventRealizationAudit(()), ()),
        event_programs=(event_program,),
    )
    first_values = {10: np.asarray([0, 1, 1, 0], dtype=np.int32), 11: np.ones(4, dtype=np.bool_)}
    second_values = {
        # Invalid Selection slots may carry an out-of-domain sentinel. Only
        # validity=true points participate in the exact runtime relation.
        10: np.asarray([1, -1, 0, 0], dtype=np.int32),
        11: np.asarray([True, False, True, True]),
    }

    first = instantiate_task_dependence(dependence_program, task_plan=task_plan, algebra_values=first_values)
    second = instantiate_task_dependence(dependence_program, task_plan=task_plan, algebra_values=second_values)
    assert first.relation.pairs != second.relation.pairs
    assert task_plan.fingerprint == request.task_plan.fingerprint
    assert "event_initial_counts" not in request.canonical_json
    instantiated_event, allocation = instantiate_parametric_event_tensor(
        event_program,
        task_plan=task_plan,
        algebra_values=second_values,
        name="runtime",
    )
    assert instantiated_event.required_dependence == second
    assert allocation.capacity == 2
    assert request.canonical_record["event_programs"][0]["realization"] == "imperative_wait_notify"
    installed = install_physical_schedule_cover(request, request.schedules)
    assert installed.event_programs == request.event_programs
    assert installed.canonical_record["event_programs"][0]["realization"] == "imperative_wait_notify"
