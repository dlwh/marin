# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt generic executable tasks to the SM90 normalized-exp kernel boundary."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

from tile_lifetime.cute_pipeline_artifact import CutePipelineCompileRequest
from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.event_dataflow import TaskFamily, verify_event_dataflow_program
from tile_lifetime.event_dataflow_adapters import streaming_contract_fold_event_descriptor
from tile_lifetime.h100_scalar_ast import finalizer_storage_conversion
from tile_lifetime.h100_streaming_lowering import (
    Sm90ForwardPipelineSchedule,
    select_sm90_contract_schedule,
    sm90_relation_packed_layout,
)
from tile_lifetime.index_domain_canonicalization import (
    DOMAIN_POSITION_COORDINATE_BASIS,
    TENSOR_AXIS_ID_COORDINATE_BASIS,
    CanonicalNormalizedExpTaskAlgebra,
    DomainCoordinateRemapProof,
    affine_predicate_in_domain_position_basis,
)
from tile_lifetime.internal_event_allocation import (
    InternalEventDeviceLegalizationRegion,
    internal_event_device_legalization_region,
    internal_event_device_legalization_region_record,
)
from tile_lifetime.layout_relation_canonicalization import (
    LayoutRelationKind,
    canonicalize_contract_layout_relations,
)
from tile_lifetime.normalized_exp_contract_region import (
    AffineIndexExpression,
    AffineIndexPredicate,
    IndexComparisonKind,
    NormalizedExpContractionDimensions,
    NormalizedExpContractProgram,
    NormalizedExpContractRegion,
    NormalizedExpFoldState,
    derive_forward_ownership_plan,
)
from tile_lifetime.physical_task_contract import (
    ExecutableEventLowering,
    ExecutablePhysicalRequest,
    PhysicalNumericalPolicy,
    PhysicalSchedule,
    PhysicalTarget,
    lower_executable_event_operations,
)
from tile_lifetime.streaming_event_schedule import (
    StreamingPhysicalEventSchedule,
    canonical_streaming_physical_event_schedule_record,
    derive_streaming_physical_event_schedule,
)
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractPrimitive,
    EvaluationOrder,
    FoldPrimitive,
    MapPrimitive,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    deserialize_tensor_program,
    primitive_inputs,
    primitive_outputs,
    scalar_convert,
    serialize_scalar_expression,
    tensor_program_value_ordinals,
)

_SM90_FORWARD_DIRECTION = "forward"


def declared_normalized_exp_algebra(
    request: ExecutablePhysicalRequest,
    schedule: PhysicalSchedule,
) -> CanonicalNormalizedExpTaskAlgebra:
    """Rebind a planner-declared region without rediscovering graph topology."""
    algebra_record = request.algebra_record
    if algebra_record.get("format") == "shuttle_domain_restricted_tensor_program_v1":
        algebra_record = algebra_record.get("program")
    if not isinstance(algebra_record, dict) or algebra_record.get("format") != "shuttle_tensor_program_v1":
        raise ValueError("normalized-exp executable preparation requires a canonical TensorProgram")
    if len(request.normalized_exp_contract_regions) != 1:
        raise ValueError("normalized-exp executable preparation requires one declared algebra region")
    source = deserialize_tensor_program(json.dumps(algebra_record, sort_keys=True, separators=(",", ":")))
    capability = request.normalized_exp_contract_regions[0]
    operations = source.operations
    values = {ordinal: value for value, ordinal in tensor_program_value_ordinals(source).items()}
    try:
        producer = operations[capability.producer_operation]
        score_map = operations[capability.score_operation]
        fold_operations = tuple(operations[ordinal] for ordinal in capability.fold_operations)
        normalized_weight = operations[capability.normalized_weight_operation]
        consumer = operations[capability.consumer_operation]
        finalizer = operations[capability.finalizer_operation]
        score_value = values[capability.score_value]
        maximum_value = values[capability.maximum_value]
        normalizer_value = values[capability.normalizer_value]
        normalized_weight_value = values[capability.normalized_weight_value]
        finalizer_value = values[capability.finalizer_value]
    except (IndexError, KeyError) as error:
        raise ValueError("normalized-exp executable capability contains a stale source ordinal") from error
    if (
        not isinstance(producer, ContractPrimitive)
        or not isinstance(score_map, MapPrimitive)
        or len(fold_operations) != 4
        or not isinstance(fold_operations[0], FoldPrimitive)
        or not isinstance(fold_operations[1], MapPrimitive)
        or not isinstance(fold_operations[2], MapPrimitive)
        or not isinstance(fold_operations[3], FoldPrimitive)
        or not isinstance(normalized_weight, MapPrimitive)
        or not isinstance(consumer, ContractPrimitive)
        or not isinstance(finalizer, MapPrimitive)
    ):
        raise ValueError("normalized-exp executable capability references unsupported algebra operations")
    maximum, center, exponential, normalizer = fold_operations
    if (
        score_map.output != score_value
        or maximum.output != maximum_value
        or normalizer.output != normalizer_value
        or normalized_weight.output != normalized_weight_value
        or finalizer.output != finalizer_value
        or capability.normalized_operand >= len(consumer.inputs)
        or consumer.inputs[capability.normalized_operand] != normalized_weight_value
    ):
        raise ValueError("normalized-exp executable capability disagrees with its declared source values")
    if capability.producer_owner_operand >= len(producer.inputs):
        raise ValueError("normalized-exp executable capability has an invalid producer owner operand")
    if capability.producer_owner_operand != 0:
        producer_permutation = (
            capability.producer_owner_operand,
            *(index for index in range(len(producer.inputs)) if index != capability.producer_owner_operand),
        )
        producer_maps = producer.input_index_maps or ((),) * len(producer.inputs)
        producer = replace(
            producer,
            inputs=tuple(producer.inputs[index] for index in producer_permutation),
            input_index_maps=tuple(producer_maps[index] for index in producer_permutation),
        )
    if capability.normalized_operand != 0:
        consumer_permutation = (
            capability.normalized_operand,
            *(index for index in range(len(consumer.inputs)) if index != capability.normalized_operand),
        )
        consumer_maps = consumer.input_index_maps or ((),) * len(consumer.inputs)
        consumer = replace(
            consumer,
            inputs=tuple(consumer.inputs[index] for index in consumer_permutation),
            input_index_maps=tuple(consumer_maps[index] for index in consumer_permutation),
        )
    axes_by_reference = {
        f"a{axis.id}": axis
        for value in (*source.inputs, *(result for operation in operations for result in primitive_outputs(operation)))
        for axis in value.axes
    }

    def axes(references: tuple[str, ...]) -> tuple[TensorAxis, ...]:
        try:
            return tuple(axes_by_reference[reference] for reference in references)
        except KeyError as error:
            raise ValueError("normalized-exp executable capability contains a stale axis reference") from error

    predicate = _declared_domain_predicate(schedule.parameters, score_map.output.axes)
    region = NormalizedExpContractRegion(
        producer=producer,
        score_map=score_map,
        fold_state=NormalizedExpFoldState(maximum, center, exponential, normalizer),
        consumer=consumer,
        finalizer=finalizer,
        dimensions=NormalizedExpContractionDimensions(
            producer_features=axes(capability.producer_features),
            normalized_axes=axes(capability.normalized_axes),
            consumer_features=axes(capability.consumer_features),
            shared_relation_axes=axes(capability.shared_relation_axes),
            relation_group_axes=axes(capability.relation_group_axes),
        ),
        domain_predicate=predicate,
    )
    proof = DomainCoordinateRemapProof(
        DOMAIN_POSITION_COORDINATE_BASIS,
        TENSOR_AXIS_ID_COORDINATE_BASIS,
        tuple((position, axis.id) for position, axis in enumerate(score_map.output.axes)),
        tuple(axis.extent for axis in score_map.output.axes),
    )
    return CanonicalNormalizedExpTaskAlgebra(source, region, proof)


def _declared_domain_predicate(parameters: Any, domain_axes: tuple[TensorAxis, ...]) -> AffineIndexPredicate | None:
    if not isinstance(parameters, dict):
        raise ValueError("SM90 schedule parameters must be a canonical mapping")
    encoded = parameters.get("domain_predicates")
    if encoded == ():
        return None
    if not isinstance(encoded, (tuple, list)) or len(encoded) != 1 or not isinstance(encoded[0], dict):
        raise ValueError("SM90 schedule must carry zero or one affine domain predicate")
    record = encoded[0]

    def expression(name: str) -> AffineIndexExpression:
        value = record.get(name)
        if not isinstance(value, dict) or type(value.get("constant")) is not int:
            raise ValueError("SM90 schedule domain predicate contains a malformed affine expression")
        coefficients = value.get("coefficients")
        if not isinstance(coefficients, (tuple, list)):
            raise ValueError("SM90 schedule domain predicate contains malformed affine coefficients")
        terms: list[tuple[int, int]] = []
        for term in coefficients:
            if (
                not isinstance(term, (tuple, list))
                or len(term) != 2
                or type(term[0]) is not int
                or type(term[1]) is not int
                or term[0] < 0
                or term[0] >= len(domain_axes)
            ):
                raise ValueError("SM90 schedule domain predicate references an invalid domain position")
            terms.append((domain_axes[term[0]].id, term[1]))
        return AffineIndexExpression(tuple(terms), value["constant"])

    try:
        comparison = IndexComparisonKind(record["comparison"])
    except (KeyError, ValueError) as error:
        raise ValueError("SM90 schedule domain predicate has an unsupported comparison") from error
    return AffineIndexPredicate(comparison, expression("left"), expression("right"))


@dataclass(frozen=True)
class ScheduleLocalTaskBinding:
    """Exact algebra operations implemented by one schedule-local task role."""

    role: str
    task_family: str
    operation_ordinals: tuple[int, ...]
    physical_auxiliary: bool = False

    def __post_init__(self) -> None:
        if not self.role or not self.task_family:
            raise ValueError("schedule-local task bindings require a role and task family")
        if self.physical_auxiliary and self.operation_ordinals:
            raise ValueError("physical schedule auxiliaries cannot claim algebra operations")


@dataclass(frozen=True)
class ScheduleLocalTaskRefinement:
    """Proof that one kernel-local task graph covers its coarse owned algebra exactly."""

    coarse_task_ids: tuple[int, ...]
    coarse_operation_ordinals: tuple[int, ...]
    bindings: tuple[ScheduleLocalTaskBinding, ...]
    event_schedule: StreamingPhysicalEventSchedule

    def __post_init__(self) -> None:
        dataflow = self.event_schedule.dataflow.program
        verify_event_dataflow_program(dataflow)
        family_names = {family.name for family in dataflow.task_families}
        bound_names = tuple(binding.task_family for binding in self.bindings)
        if len(bound_names) != len(set(bound_names)) or set(bound_names) != family_names:
            raise ValueError("schedule-local task bindings must cover every local task family exactly once")
        claimed = tuple(ordinal for binding in self.bindings for ordinal in binding.operation_ordinals)
        if len(claimed) != len(set(claimed)):
            raise ValueError("schedule-local task bindings claim an algebra operation more than once")
        if tuple(sorted(claimed)) != tuple(sorted(self.coarse_operation_ordinals)):
            raise ValueError("schedule-local task bindings must cover every coarse algebra operation exactly once")

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "format": "sm90_schedule_local_task_refinement_v1",
            "coarse_task_ids": self.coarse_task_ids,
            "coarse_operation_ordinals": self.coarse_operation_ordinals,
            "instantiation": "one_local_event_graph_per_coarse_task_coordinate",
            "bindings": tuple(
                {
                    "role": binding.role,
                    "task_family": binding.task_family,
                    "operation_ordinals": binding.operation_ordinals,
                    "physical_auxiliary": binding.physical_auxiliary,
                }
                for binding in self.bindings
            ),
            "event_schedule": canonical_streaming_physical_event_schedule_record(self.event_schedule),
        }


def _forward_schedules(request: ExecutablePhysicalRequest) -> tuple[PhysicalSchedule, ...]:
    """Select the forward schedule by its derived direction, not its family label."""
    return tuple(
        schedule
        for schedule in request.schedules
        if schedule.family.target is PhysicalTarget.SM90
        and schedule.parameters.get("direction") == _SM90_FORWARD_DIRECTION
    )


@dataclass(frozen=True)
class ExecutableNormalizedExpContractPlan:
    """One emitter-covered task region plus its retained executable boundary."""

    source_request: ExecutablePhysicalRequest
    program: NormalizedExpContractProgram[Sm90ForwardPipelineSchedule]
    compile_request: CutePipelineCompileRequest
    owned_task_ids: tuple[int, ...]
    residual_task_ids: tuple[int, ...]
    boundary_event_names: tuple[str, ...]
    finalizer_fusion: dict[str, Any] | None
    relation_view_proofs: tuple[dict[str, Any], ...]
    event_erasure_witnesses: tuple[dict[str, Any], ...]
    event_lowerings: tuple[ExecutableEventLowering, ...]
    schedule_local_refinement: ScheduleLocalTaskRefinement
    internal_event_legalization: InternalEventDeviceLegalizationRegion

    @property
    def covered_task_ids(self) -> frozenset[int]:
        """Source task ids covered by the emitted schedule component."""
        return frozenset(self.owned_task_ids)

    @property
    def whole_request_runtime_supported(self) -> bool:
        """Whether every source task and Event has an exact covering witness."""
        source_tasks = {task.id for task in self.source_request.task_plan.tasks}
        source_events = {event.name for event in self.source_request.event_plans}
        witnessed_events = {cast(str, witness["event"]) for witness in self.event_erasure_witnesses}
        return (
            self.covered_task_ids == source_tasks
            and witnessed_events == source_events
            and not self.residual_task_ids
            and not self.boundary_event_names
        )

    def require_whole_request_runtime(self) -> None:
        """Reject invocation when any source task or boundary Event lacks a covering witness."""
        if self.whole_request_runtime_supported:
            return
        tasks = ", ".join(str(task_id) for task_id in self.residual_task_ids) or "none"
        events = ", ".join(self.boundary_event_names) or "none"
        raise ValueError(
            "the SM90 kernel compiles only the normalized-exp task region; runtime composition for residual "
            f"public tasks [{tasks}] and boundary Events [{events}] is not implemented"
        )


def normalized_exp_emitter_program(
    request: ExecutablePhysicalRequest,
) -> NormalizedExpContractProgram[Sm90ForwardPipelineSchedule]:
    """Reconstruct the typed normalized-exp program selected by an SM90 schedule.

    This boundary performs algebra, domain, layout, and schedule validation but
    deliberately does not claim coverage of the surrounding source request.
    Callers that replace a larger algebraic region must provide their own exact
    task and Event coverage proof.
    """
    if request.target is not PhysicalTarget.SM90:
        raise ValueError("the normalized-exp CuTe adapter requires an SM90 executable request")
    if request.numerical_policy is not PhysicalNumericalPolicy.FAST:
        raise ValueError("the normalized-exp CuTe adapter requires FAST numerical policy")
    schedules = _forward_schedules(request)
    if len(schedules) != 1:
        raise ValueError("the executable request must contain one attached SM90 Contract/Fold forward schedule")
    physical_schedule = schedules[0]
    algebra = declared_normalized_exp_algebra(request, physical_schedule)
    _verify_domain_schedule(algebra.region, physical_schedule.parameters)
    contract_parameters = physical_schedule.parameters.get("contract")
    if not isinstance(contract_parameters, dict):
        raise ValueError("the attached SM90 schedule is missing Contract parameters")
    relation_layout = _relation_layout(algebra.region, contract_parameters)
    contract_schedule = select_sm90_contract_schedule(
        owner_extent=relation_layout.owner_extent,
        streamed_extent=relation_layout.streamed_extent,
        producer_feature_extent=relation_layout.producer_feature_extent,
        consumer_feature_extent=relation_layout.consumer_feature_extent,
        threads=_required_int(contract_parameters, "threads"),
        owner_tile=_required_int(contract_parameters, "tile_m"),
        streamed_tile=_required_int(contract_parameters, "tile_n"),
        relation_layout=relation_layout,
    )
    schedule = Sm90ForwardPipelineSchedule(
        contract=contract_schedule,
        stages=_required_int(physical_schedule.parameters, "stages"),
    )
    terminal_conversion, _ = _terminal_output_conversion(algebra)
    emitter_source, emitter_region = _emitter_algebra(
        algebra,
        relation_layout,
        terminal_conversion=terminal_conversion,
    )
    return NormalizedExpContractProgram(
        source=emitter_source,
        region=emitter_region,
        ownership=derive_forward_ownership_plan(emitter_region),
        schedule=schedule,
        materialized_values=(emitter_region.finalizer.output,),
    )


def prepare_sm90_normalized_exp_contract(
    request: ExecutablePhysicalRequest,
) -> ExecutableNormalizedExpContractPlan:
    """Prepare a CuTe request from canonical algebra, tasks, Events, and schedule."""
    if request.target is not PhysicalTarget.SM90:
        raise ValueError("the normalized-exp CuTe adapter requires an SM90 executable request")
    if request.numerical_policy is not PhysicalNumericalPolicy.FAST:
        raise ValueError("the normalized-exp CuTe adapter requires FAST numerical policy")
    schedules = _forward_schedules(request)
    if len(schedules) != 1:
        raise ValueError("the executable request must contain one attached SM90 Contract/Fold forward schedule")
    physical_schedule = schedules[0]
    algebra = declared_normalized_exp_algebra(request, physical_schedule)
    _verify_emitter_owned_operations(request, algebra, physical_schedule.task_ids)
    _verify_domain_schedule(algebra.region, physical_schedule.parameters)

    contract_parameters = physical_schedule.parameters.get("contract")
    if not isinstance(contract_parameters, dict):
        raise ValueError("the attached SM90 schedule is missing Contract parameters")
    relation_layout = _relation_layout(algebra.region, contract_parameters)
    contract_schedule = select_sm90_contract_schedule(
        owner_extent=relation_layout.owner_extent,
        streamed_extent=relation_layout.streamed_extent,
        producer_feature_extent=relation_layout.producer_feature_extent,
        consumer_feature_extent=relation_layout.consumer_feature_extent,
        threads=_required_int(contract_parameters, "threads"),
        owner_tile=_required_int(contract_parameters, "tile_m"),
        streamed_tile=_required_int(contract_parameters, "tile_n"),
        relation_layout=relation_layout,
    )
    stages = _required_int(physical_schedule.parameters, "stages")
    schedule = Sm90ForwardPipelineSchedule(contract=contract_schedule, stages=stages)
    terminal_conversion, terminal_relation_ordinals = _terminal_output_conversion(algebra)
    emitter_source, emitter_region = _emitter_algebra(
        algebra,
        relation_layout,
        terminal_conversion=terminal_conversion,
    )
    program = NormalizedExpContractProgram(
        source=emitter_source,
        region=emitter_region,
        ownership=derive_forward_ownership_plan(emitter_region),
        schedule=schedule,
        materialized_values=(emitter_region.finalizer.output,),
    )

    owned = frozenset(physical_schedule.task_ids)
    ordinal_to_task = {ordinal: task.id for task in request.task_plan.tasks for ordinal in task.operation_ordinals}
    family_ids = {task.family: task.id for task in request.task_plan.tasks}
    terminal_task = None if terminal_conversion is None else ordinal_to_task[terminal_conversion[0]]
    terminal_event = next(
        (
            (index, event)
            for index, event in enumerate(request.event_plans)
            if terminal_task is not None
            and isinstance(event.required_dependence.relation.target, TaskFamily)
            and family_ids.get(event.required_dependence.relation.target) == terminal_task
        ),
        None,
    )
    candidate_residual = tuple(
        task.id for task in request.task_plan.tasks if task.id not in owned and task.id != terminal_task
    )
    relation_view_proofs = _relation_view_proofs(
        request,
        algebra,
        relation_layout=relation_layout,
        residual_task_ids=candidate_residual,
        terminal_relation_ordinals=terminal_relation_ordinals,
    )
    legalized_relation_tasks = {
        cast(int, proof["task_id"]) for proof in relation_view_proofs if proof["zero_copy"] and proof["abi_compatible"]
    }
    residual = tuple(task.id for task in request.task_plan.tasks if task.id not in owned)
    covered_tasks = owned | legalized_relation_tasks | ({terminal_task} if terminal_task is not None else set())
    boundary_events: list[str] = []
    event_erasure_witnesses: list[dict[str, Any]] = []
    event_lowerings = list(lower_executable_event_operations(request))
    for event_index, (event, realization) in enumerate(
        zip(
            request.event_plans,
            request.event_realizations.audit.entries,
            strict=True,
        )
    ):
        source_family = event.required_dependence.relation.source
        target_family = event.required_dependence.relation.target
        if not isinstance(source_family, TaskFamily) or not isinstance(target_family, TaskFamily):
            raise TypeError("executable task Events must connect tiled task families")
        endpoints = {
            family_ids[source_family],
            family_ids[target_family],
        }
        internal = endpoints <= owned
        if internal and realization.kind is not EventRealizationKind.ERASED_PROGRAM_ORDER:
            raise ValueError(f"emitter-internal Event {event.name!r} was not erased by the fused schedule")
        if not internal and realization.kind is not EventRealizationKind.PHYSICAL:
            raise ValueError(f"Event {event.name!r} outside the emitter-owned region must remain physical")
        if internal:
            source_cover = _task_cover_record(
                family_ids[source_family],
                owned=owned,
                relation_view_proofs=relation_view_proofs,
                terminal_task=terminal_task,
                terminal_conversion=terminal_conversion,
            )
            target_cover = _task_cover_record(
                family_ids[target_family],
                owned=owned,
                relation_view_proofs=relation_view_proofs,
                terminal_task=terminal_task,
                terminal_conversion=terminal_conversion,
            )
            witness = {
                "event": event.name,
                "event_plan": cast(tuple[Any, ...], request.canonical_record["event_plans"])[event_index],
                "source_task_id": family_ids[source_family],
                "target_task_id": family_ids[target_family],
                "source_request_realization": realization.kind.value,
                "adapter_realization": "emitter_schedule_program_order",
                "source_cover": source_cover,
                "target_cover": target_cover,
                "covering_launch_phase_order": tuple(sorted(covered_tasks)),
            }
            event_erasure_witnesses.append(witness)
        if not internal and len(endpoints & owned) == 1:
            boundary_events.append(event.name)

    finalized_event_lowerings = tuple(event_lowerings)
    internal_event_schedule = derive_streaming_physical_event_schedule(streaming_contract_fold_event_descriptor(program))
    schedule_local_refinement = _schedule_local_task_refinement(
        request,
        algebra,
        owned_task_ids=physical_schedule.task_ids,
        event_schedule=internal_event_schedule,
    )
    internal_event_legalization = internal_event_device_legalization_region(
        0,
        internal_event_schedule.allocations,
    )
    compile_request = _compile_request(
        request,
        emitter_region,
        schedule,
        physical_schedule_record=physical_schedule.canonical_record,
        boundary_event_names=tuple(boundary_events),
        event_lowerings=finalized_event_lowerings,
        schedule_local_refinement=schedule_local_refinement,
        internal_event_legalization=internal_event_legalization,
    )
    return ExecutableNormalizedExpContractPlan(
        request,
        program,
        compile_request,
        physical_schedule.task_ids,
        residual,
        tuple(boundary_events),
        (
            None
            if terminal_conversion is None
            else {
                "task_id": terminal_task,
                "operation_ordinal": terminal_conversion[0],
                "commuted_relation_ordinals": terminal_relation_ordinals,
                "proof": "pointwise conversion commutes with exact one-to-one Relations",
                "conversion": {
                    "source_dtype": terminal_conversion[1].inputs[0].dtype.value,
                    "result_dtype": terminal_conversion[1].output.dtype.value,
                    "semantics": terminal_conversion[1].expression.conversion_semantics.value,
                },
                "event_witness": (
                    None
                    if terminal_event is None
                    else {
                        "event": terminal_event[1].name,
                        "event_plan": cast(tuple[Any, ...], request.canonical_record["event_plans"])[terminal_event[0]],
                        "realization": "erased_by_bijective_relation_commutation_and_kernel_output_conversion",
                    }
                ),
            }
        ),
        relation_view_proofs,
        tuple(event_erasure_witnesses),
        finalized_event_lowerings,
        schedule_local_refinement,
        internal_event_legalization,
    )


def _schedule_local_task_refinement(
    request: ExecutablePhysicalRequest,
    algebra: CanonicalNormalizedExpTaskAlgebra,
    *,
    owned_task_ids: tuple[int, ...],
    event_schedule: StreamingPhysicalEventSchedule,
) -> ScheduleLocalTaskRefinement:
    region = algebra.region
    ordinal_by_output = {
        value: ordinal
        for ordinal, operation in enumerate(algebra.task_program.operations)
        for value in primitive_outputs(operation)
    }

    def ordinal(operation: object) -> int:
        outputs = primitive_outputs(operation)
        if len(outputs) != 1 or outputs[0] not in ordinal_by_output:
            raise ValueError("schedule-local normalized-exp operation is absent from the canonical algebra")
        return ordinal_by_output[outputs[0]]

    producer = (ordinal(region.producer),)
    fold = tuple(
        ordinal(operation)
        for operation in (
            region.score_map,
            region.fold_state.maximum,
            region.fold_state.center,
            region.fold_state.exponential,
            region.fold_state.normalizer,
        )
    )
    consumer = (ordinal(region.consumer),)
    finalizer = (ordinal(region.finalizer),)
    coarse_ordinals = tuple(
        sorted(ordinal for task_id in owned_task_ids for ordinal in request.task_plan.tasks[task_id].operation_ordinals)
    )
    classified = set((*producer, *fold, *consumer, *finalizer))
    finalizer = (*finalizer, *(ordinal for ordinal in coarse_ordinals if ordinal not in classified))
    dataflow = event_schedule.dataflow
    return ScheduleLocalTaskRefinement(
        coarse_task_ids=owned_task_ids,
        coarse_operation_ordinals=coarse_ordinals,
        bindings=(
            ScheduleLocalTaskBinding("resident_input_stage", dataflow.resident_input_stage.name, (), True),
            ScheduleLocalTaskBinding(
                "first_streamed_input_stage",
                dataflow.first_streamed_input_stage.name,
                (),
                True,
            ),
            ScheduleLocalTaskBinding(
                "second_streamed_input_stage",
                dataflow.second_streamed_input_stage.name,
                (),
                True,
            ),
            ScheduleLocalTaskBinding("producer_contract", dataflow.first_contract.name, producer),
            ScheduleLocalTaskBinding("normalized_fold_update", dataflow.fold_update.name, fold),
            ScheduleLocalTaskBinding("consumer_contract", dataflow.second_contract.name, consumer),
            ScheduleLocalTaskBinding("finalizer", dataflow.finalize.name, finalizer),
        ),
        event_schedule=event_schedule,
    )


def _task_cover_record(
    task_id: int,
    *,
    owned: frozenset[int],
    relation_view_proofs: tuple[dict[str, Any], ...],
    terminal_task: int | None,
    terminal_conversion: tuple[int, MapPrimitive] | None,
) -> dict[str, object]:
    if task_id in owned:
        return {"kind": "emitted_schedule_task", "task_id": task_id}
    relation = next(
        (
            proof
            for proof in relation_view_proofs
            if proof["task_id"] == task_id and proof["zero_copy"] and proof["abi_compatible"]
        ),
        None,
    )
    if relation is not None:
        return {
            "kind": "public_abi_relation",
            "task_id": task_id,
            "operation_ordinal": cast(int, relation["operation_ordinal"]),
            "public_abi": relation["public_abi"],
        }
    if task_id == terminal_task and terminal_conversion is not None:
        return {
            "kind": "fused_finalizer_map",
            "task_id": task_id,
            "operation_ordinal": terminal_conversion[0],
        }
    raise ValueError(f"Event endpoint task {task_id} has no emitted or legalized cover")


def _verify_emitter_owned_operations(
    request: ExecutablePhysicalRequest,
    algebra: CanonicalNormalizedExpTaskAlgebra,
    owned_task_ids: tuple[int, ...],
) -> None:
    region = algebra.region
    operations = (
        region.producer,
        region.score_map,
        region.fold_state.maximum,
        region.fold_state.center,
        region.fold_state.exponential,
        region.fold_state.normalizer,
        region.consumer,
        region.finalizer,
    )
    ordinal_by_output = {
        value: ordinal
        for ordinal, operation in enumerate(algebra.task_program.operations)
        for value in primitive_outputs(operation)
    }
    try:
        expected = {ordinal_by_output[primitive_outputs(operation)[0]] for operation in operations}
    except (IndexError, KeyError) as error:
        raise ValueError("normalized-exp region operation is absent from the canonical task program") from error
    for operation in operations:
        ordinal = ordinal_by_output[primitive_outputs(operation)[0]]
        source = algebra.task_program.operations[ordinal]
        if source == operation:
            continue
        if not _is_contract_operand_permutation(source, operation):
            raise ValueError("normalized-exp extraction changed an operation beyond a Contract operand permutation")
    actual = {ordinal for task_id in owned_task_ids for ordinal in request.task_plan.tasks[task_id].operation_ordinals}
    extra = actual - expected
    if any(
        not _is_index_only_relation_legalization(algebra, ordinal)
        and not _is_contract_operand_conversion(algebra, ordinal, region)
        for ordinal in extra
    ):
        raise ValueError(
            "attached SM90 schedule owns operations outside the Contract/Map/Fold/Contract/Map region and its "
            f"typed Contract operand conversions (owned={sorted(actual)}, region={sorted(expected)})"
        )
    if not expected <= actual:
        raise ValueError(
            "attached SM90 schedule does not own every normalized-exp region operation "
            f"(owned={sorted(actual)}, region={sorted(expected)})"
        )


def _is_contract_operand_permutation(source: object, extracted: object) -> bool:
    if not isinstance(source, ContractPrimitive) or not isinstance(extracted, ContractPrimitive):
        return False
    source_operands = tuple(zip(source.inputs, source.input_index_maps, strict=True))
    extracted_operands = tuple(zip(extracted.inputs, extracted.input_index_maps, strict=True))
    unmatched = list(source_operands)
    for operand in extracted_operands:
        try:
            unmatched.remove(operand)
        except ValueError:
            return False
    return (
        not unmatched
        and replace(
            source,
            inputs=extracted.inputs,
            input_index_maps=extracted.input_index_maps,
        )
        == extracted
    )


def _is_contract_operand_conversion(
    algebra: CanonicalNormalizedExpTaskAlgebra,
    ordinal: int,
    region: NormalizedExpContractRegion,
) -> bool:
    operation = algebra.task_program.operations[ordinal]
    if not isinstance(operation, MapPrimitive) or operation.expression.kind is not ScalarExpressionKind.CONVERT:
        return False
    output = operation.output
    return output in (*region.producer.inputs, *region.consumer.inputs)


def _is_index_only_relation_legalization(algebra: CanonicalNormalizedExpTaskAlgebra, ordinal: int) -> bool:
    operation = algebra.task_program.operations[ordinal]
    if (
        not isinstance(operation, RelationPrimitive)
        or operation.predicate is not None
        or any(expression is not None for expression in operation.resolved_result_expressions)
        or any(source is None for source in operation.resolved_result_sources)
    ):
        return False
    return any(
        ordinal in proof.relation_operation_ordinals
        for proof in canonicalize_contract_layout_relations(algebra.task_program).proofs
    )


def _relation_layout(region: NormalizedExpContractRegion, parameters: dict[str, Any]):
    """Derive packed ABI extents from algebraic axes, not a workload mode."""
    owner_extent = _required_int(parameters, "owner_extent")
    streamed_extent = _required_int(parameters, "streamed_extent")
    producer_feature_extent = _required_int(parameters, "producer_feature_extent")
    consumer_feature_extent = _required_int(parameters, "consumer_feature_extent")
    owner_axes = region.owner_axes
    normalized = set(region.dimensions.normalized_axes)
    producer_feature = set(region.dimensions.producer_features)
    consumer_feature = set(region.dimensions.consumer_features)
    producer_axes = set().union(*(set(value.axes) for value in region.producer.inputs))
    consumer_operand_axes = set(region.consumer.inputs[1].axes)
    common = tuple(
        axis
        for axis in owner_axes
        if axis in producer_axes
        and axis in consumer_operand_axes
        and axis not in normalized | producer_feature | consumer_feature
    )
    if len(common) < 2:
        raise ValueError("the current SM90 packed ABI requires explicit batch and shared relation axes")
    batch_axes, shared_axis = common[:-1], common[-1]
    grouped_axes = tuple(axis for axis in owner_axes if axis not in common and axis.extent != owner_extent)
    grouped_shape = tuple(axis.extent for axis in grouped_axes) or (1,)
    owner_candidates = tuple(axis for axis in owner_axes if axis not in common and axis not in grouped_axes)
    if len(owner_candidates) != 1 or owner_candidates[0].extent != owner_extent:
        raise ValueError("the SM90 packed ABI cannot identify one owner axis from the canonical contraction dimensions")
    return sm90_relation_packed_layout(
        batch_shape=tuple(axis.extent for axis in batch_axes),
        shared_relation_extent=shared_axis.extent,
        relation_group_shape=grouped_shape,
        owner_extent=owner_extent,
        streamed_extent=streamed_extent,
        producer_feature_extent=producer_feature_extent,
        consumer_feature_extent=consumer_feature_extent,
    )


def _terminal_output_conversion(
    algebra: CanonicalNormalizedExpTaskAlgebra,
) -> tuple[tuple[int, MapPrimitive] | None, tuple[int, ...]]:
    uses: dict[ProgramValue, list[tuple[int, object]]] = {}
    for ordinal, operation in enumerate(algebra.task_program.operations):
        for value in primitive_inputs(operation):
            uses.setdefault(value, []).append((ordinal, operation))
    current = algebra.region.finalizer.output
    relation_ordinals: list[int] = []
    while True:
        consumers = uses.get(current, ())
        if len(consumers) != 1:
            return None, ()
        ordinal, operation = consumers[0]
        if isinstance(operation, RelationPrimitive):
            if (
                operation.predicate is not None
                or len(operation.outputs) != 1
                or operation.resolved_result_sources != (current,)
            ):
                return None, ()
            relation_ordinals.append(ordinal)
            current = operation.outputs[0]
            continue
        if not isinstance(operation, MapPrimitive):
            return None, ()
        expression = operation.expression
        if (
            expression.kind is not ScalarExpressionKind.CONVERT
            or expression.result_dtype is not operation.output.dtype
            or expression.conversion_semantics is not ScalarConversionSemantics.ROUND_NEAREST_EVEN
            or operation.inputs != (current,)
            or operation.output not in algebra.task_program.outputs
        ):
            return None, ()
        return (ordinal, operation), tuple(relation_ordinals)


def _relation_view_proofs(
    request: ExecutablePhysicalRequest,
    algebra: CanonicalNormalizedExpTaskAlgebra,
    *,
    relation_layout,
    residual_task_ids: tuple[int, ...],
    terminal_relation_ordinals: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    proofs = canonicalize_contract_layout_relations(algebra.task_program).proofs
    proof_by_ordinal = {
        ordinal: proof
        for proof in proofs
        for ordinal in proof.relation_operation_ordinals
        if len(proof.relation_operation_ordinals) == 1
    }
    records: list[dict[str, Any]] = []
    region = algebra.region
    normalized = region.dimensions.normalized_axes[0]
    streamed_inputs = tuple(
        index
        for index in range(2)
        if normalized in region.producer.inputs[index].axes
        or any(mapping.domain_axis == normalized for mapping in region.producer.index_maps_for_input(index))
    )
    if len(streamed_inputs) != 1:
        raise ValueError("the SM90 public ABI cannot identify one streamed Contract operand")
    owner_source = region.producer.inputs[1 - streamed_inputs[0]]
    consumer_source = region.consumer.inputs[1]
    owner_axis = next(axis for axis in region.owner_axes if axis.extent == relation_layout.owner_extent)
    common = tuple(axis for axis in region.owner_axes if axis in owner_source.axes and axis in consumer_source.axes)
    batch_axes, shared_axis = common[:-1], common[-1]
    group_axes = tuple(axis for axis in region.owner_axes if axis not in (*common, owner_axis))
    producer_feature = region.dimensions.producer_features[0]
    consumer_feature = region.dimensions.consumer_features[0]
    physical_owner_shape = relation_layout.owner_physical_shape(producer_feature.extent)
    physical_output_shape = relation_layout.owner_physical_shape(consumer_feature.extent)
    physical_owner_strides = _row_major_strides(physical_owner_shape)
    physical_output_strides = _row_major_strides(physical_output_shape)
    query_result_shape = (
        *relation_layout.batch_shape,
        relation_layout.owner_extent,
        shared_axis.extent,
        *relation_layout.relation_group_shape,
        producer_feature.extent,
    )
    output_preflatten_shape = (*query_result_shape[:-1], consumer_feature.extent)
    output_chain = set(terminal_relation_ordinals)
    for task_id in residual_task_ids:
        task = request.task_plan.tasks[task_id]
        if len(task.operation_ordinals) != 1:
            continue
        ordinal = task.operation_ordinals[0]
        operation = algebra.task_program.operations[ordinal]
        if not isinstance(operation, RelationPrimitive):
            continue
        proof = proof_by_ordinal.get(ordinal)
        if proof is None:
            records.append(
                {
                    "task_id": task_id,
                    "operation_ordinal": ordinal,
                    "zero_copy": False,
                    "abi_compatible": False,
                    "reason": "Relation has no exact permutation/row-major-reshape proof",
                }
            )
            continue
        zero_copy = proof.kind in (LayoutRelationKind.AXIS_PERMUTATION, LayoutRelationKind.ROW_MAJOR_RESHAPE)
        query_reshape = (
            proof.kind is LayoutRelationKind.ROW_MAJOR_RESHAPE
            and operation.outputs == (owner_source,)
            and operation.inputs[0] in algebra.task_program.inputs
            and operation.inputs[0].shape == physical_owner_shape
            and operation.outputs[0].shape == query_result_shape
        )
        output_view = ordinal in output_chain
        output_chain_complete = _output_relation_chain_matches_public_abi(
            algebra,
            terminal_relation_ordinals,
            output_preflatten_shape=output_preflatten_shape,
            physical_output_shape=physical_output_shape,
        )
        abi_compatible = query_reshape or (output_view and output_chain_complete)
        reason = (
            "row-major public tensor is the kernel's flat relation-coordinate ABI"
            if query_reshape
            else (
                "exact output view chain is the inverse flat relation-coordinate ABI"
                if output_view and output_chain_complete
                else "zero-copy Relation is not covered by the kernel's public affine ABI"
            )
        )
        records.append(
            {
                "task_id": task_id,
                "operation_ordinal": ordinal,
                "kind": proof.kind.value,
                "zero_copy": zero_copy,
                "abi_compatible": abi_compatible,
                "source_shape": proof.source_shape,
                "result_shape": proof.result_shape,
                "source_to_result_dimensions": proof.source_to_result_dimensions,
                "public_abi": (
                    {
                        "physical_shape": physical_owner_shape,
                        "physical_strides": physical_owner_strides,
                        "physical_axes": ("batch", "owner", "flat_relation", "feature"),
                        "algebra_axes": tuple(
                            axis.id for axis in (*batch_axes, shared_axis, *group_axes, owner_axis, producer_feature)
                        ),
                        "coordinate_map": {
                            "shared_relation": {
                                "source": "flat_relation",
                                "floor_divisor": relation_layout.relation_group_size,
                            },
                            "relation_groups": tuple(
                                {
                                    "axis": axis.id,
                                    "source": "flat_relation",
                                    "floor_divisor": mapping.divisor,
                                    "modulus": mapping.modulus,
                                }
                                for axis, mapping in zip(group_axes, relation_layout.group_relation_maps, strict=True)
                            ),
                            "owner": "owner",
                            "feature": "feature",
                        },
                    }
                    if query_reshape
                    else (
                        {
                            "physical_shape": physical_output_shape,
                            "physical_strides": physical_output_strides,
                            "physical_axes": ("batch", "owner", "flat_relation", "feature"),
                            "algebra_axes": tuple(
                                axis.id for axis in (*batch_axes, shared_axis, *group_axes, owner_axis, consumer_feature)
                            ),
                            "inverse_relation_chain": terminal_relation_ordinals,
                        }
                        if output_view and output_chain_complete
                        else None
                    )
                ),
                "reason": reason,
            }
        )
    return tuple(records)


def _output_relation_chain_matches_public_abi(
    algebra: CanonicalNormalizedExpTaskAlgebra,
    relation_ordinals: tuple[int, ...],
    *,
    output_preflatten_shape: tuple[int, ...],
    physical_output_shape: tuple[int, ...],
) -> bool:
    if not relation_ordinals:
        return False
    current = algebra.region.finalizer.output
    operations: list[RelationPrimitive] = []
    for ordinal in relation_ordinals:
        operation = algebra.task_program.operations[ordinal]
        if not isinstance(operation, RelationPrimitive) or operation.inputs != (current,):
            return False
        operations.append(operation)
        current = operation.outputs[0]
    if current.shape != physical_output_shape:
        return False
    final = operations[-1]
    if final.inputs[0].shape != output_preflatten_shape:
        return False
    canonical = canonicalize_contract_layout_relations(algebra.task_program)
    proof_kinds = {
        proof.relation_operation_ordinals[0]: proof.kind
        for proof in canonical.proofs
        if len(proof.relation_operation_ordinals) == 1
    }
    if proof_kinds.get(relation_ordinals[-1]) is not LayoutRelationKind.ROW_MAJOR_RESHAPE:
        return False
    return all(
        proof_kinds.get(ordinal) in (LayoutRelationKind.AXIS_PERMUTATION, LayoutRelationKind.ROW_MAJOR_RESHAPE)
        for ordinal in relation_ordinals
    )


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    reversed_strides: list[int] = []
    for extent in reversed(shape):
        reversed_strides.append(stride)
        stride *= extent
    return tuple(reversed(reversed_strides))


def _emitter_algebra(
    algebra: CanonicalNormalizedExpTaskAlgebra,
    relation_layout,
    *,
    terminal_conversion: tuple[int, MapPrimitive] | None,
):
    """Project logical axes onto the existing packed SM90 kernel ABI."""
    region = algebra.region
    if len(region.producer.inputs) != 2 or len(region.consumer.inputs) != 2:
        raise ValueError("the current SM90 emitter requires binary producer and consumer Contracts")
    normalized = region.dimensions.normalized_axes[0]
    streamed_inputs = tuple(
        index
        for index in range(2)
        if normalized in region.producer.inputs[index].axes
        or any(mapping.domain_axis == normalized for mapping in region.producer.index_maps_for_input(index))
    )
    if len(streamed_inputs) != 1:
        raise ValueError("the SM90 emitter cannot identify one streamed producer operand from Contract index maps")
    streamed_index = streamed_inputs[0]
    owner_index = 1 - streamed_index
    owner_source = region.producer.inputs[owner_index]
    streamed_source = region.producer.inputs[streamed_index]
    consumer_source = region.consumer.inputs[1]

    score_axes = region.producer.output.axes
    referenced = {
        dimension
        for expression in (
            region.domain_predicate.left if region.domain_predicate is not None else None,
            region.domain_predicate.right if region.domain_predicate is not None else None,
        )
        if expression is not None
        for dimension, _ in expression.coefficients
    }
    owner_coordinates = referenced - {normalized.id}
    if len(owner_coordinates) != 1:
        raise ValueError("the current SM90 emitter requires one explicit owner coordinate in the domain predicate")
    owner_axis = next(axis for axis in score_axes if axis.id == next(iter(owner_coordinates)))
    common = tuple(axis for axis in region.owner_axes if axis in owner_source.axes and axis in consumer_source.axes)
    if len(common) < 2:
        raise ValueError("the current SM90 emitter requires batch and shared relation axes")
    batch_axes, shared_axis = common[:-1], common[-1]
    group_axes = tuple(axis for axis in region.owner_axes if axis not in (*common, owner_axis))
    physical_group_axes = group_axes
    if not physical_group_axes and relation_layout.relation_group_size == 1:
        physical_group_axes = relation_layout.relation_group_axes
    if tuple(axis.extent for axis in physical_group_axes) != relation_layout.relation_group_shape:
        raise ValueError("canonical relation-group axes disagree with the selected SM90 relation packing")
    producer_feature = region.dimensions.producer_features[0]
    consumer_feature = region.dimensions.consumer_features[0]
    owner = ProgramValue(
        owner_source.name,
        (*batch_axes, shared_axis, *physical_group_axes, owner_axis, producer_feature),
        owner_source.dtype,
    )
    streamed = ProgramValue(
        streamed_source.name,
        (*batch_axes, shared_axis, normalized, producer_feature),
        streamed_source.dtype,
    )
    consumer_operand = ProgramValue(
        consumer_source.name,
        (*batch_axes, shared_axis, normalized, consumer_feature),
        consumer_source.dtype,
    )
    score_domain = (*batch_axes, shared_axis, *physical_group_axes, owner_axis, normalized)
    row_domain = score_domain[:-1]
    output_domain = (*row_domain, consumer_feature)
    finalizer_dtype = (
        region.finalizer.output.dtype if terminal_conversion is None else terminal_conversion[1].output.dtype
    )
    replacements = {
        region.producer.output: ProgramValue(region.producer.output.name, score_domain, region.producer.output.dtype),
        region.score_map.output: ProgramValue(region.score_map.output.name, score_domain, region.score_map.output.dtype),
        region.fold_state.maximum.output: ProgramValue(
            region.fold_state.maximum.output.name, row_domain, region.fold_state.maximum.output.dtype
        ),
        region.fold_state.center.output: ProgramValue(
            region.fold_state.center.output.name, score_domain, region.fold_state.center.output.dtype
        ),
        region.fold_state.exponential.output: ProgramValue(
            region.fold_state.exponential.output.name, score_domain, region.fold_state.exponential.output.dtype
        ),
        region.fold_state.normalizer.output: ProgramValue(
            region.fold_state.normalizer.output.name, row_domain, region.fold_state.normalizer.output.dtype
        ),
        region.consumer.output: ProgramValue(region.consumer.output.name, output_domain, region.consumer.output.dtype),
        region.finalizer.output: ProgramValue(region.finalizer.output.name, output_domain, finalizer_dtype),
    }

    def values(items: tuple[ProgramValue, ...]) -> tuple[ProgramValue, ...]:
        return tuple(replacements.get(value, value) for value in items)

    producer = replace(
        region.producer,
        inputs=(owner, streamed),
        output=replacements[region.producer.output],
        input_index_maps=((), ()),
        order=EvaluationOrder.ORDER_FREE,
    )
    score_map = replace(
        region.score_map, inputs=values(region.score_map.inputs), output=replacements[region.score_map.output]
    )
    maximum = replace(
        region.fold_state.maximum,
        input=replacements[region.score_map.output],
        output=replacements[region.fold_state.maximum.output],
        reduction_axes=(normalized,),
        order=EvaluationOrder.ORDER_FREE,
    )
    center = replace(
        region.fold_state.center,
        inputs=values(region.fold_state.center.inputs),
        output=replacements[region.fold_state.center.output],
        expression=region.fold_state.center.expression,
    )
    exponential = replace(
        region.fold_state.exponential,
        inputs=values(region.fold_state.exponential.inputs),
        output=replacements[region.fold_state.exponential.output],
    )
    normalizer = replace(
        region.fold_state.normalizer,
        input=replacements[region.fold_state.exponential.output],
        output=replacements[region.fold_state.normalizer.output],
        reduction_axes=(normalized,),
        order=EvaluationOrder.ORDER_FREE,
    )
    consumer = replace(
        region.consumer,
        inputs=(replacements[region.fold_state.exponential.output], consumer_operand),
        output=replacements[region.consumer.output],
        reduction_axes=(normalized,),
        input_index_maps=((), ()),
        order=EvaluationOrder.ORDER_FREE,
    )
    finalizer_expression = region.finalizer.expression
    if terminal_conversion is not None:
        conversion = terminal_conversion[1]
        finalizer_expression = scalar_convert(
            finalizer_expression,
            result_dtype=conversion.output.dtype,
            semantics=conversion.expression.conversion_semantics,
        )
    finalizer = replace(
        region.finalizer,
        inputs=values(region.finalizer.inputs),
        output=replacements[region.finalizer.output],
        expression=finalizer_expression,
    )
    dimensions = NormalizedExpContractionDimensions(
        producer_features=(producer_feature,),
        normalized_axes=(normalized,),
        consumer_features=(consumer_feature,),
        shared_relation_axes=(shared_axis,),
        relation_group_axes=physical_group_axes,
    )
    emitter_region = NormalizedExpContractRegion(
        producer,
        score_map,
        NormalizedExpFoldState(maximum, center, exponential, normalizer),
        consumer,
        finalizer,
        dimensions,
        region.domain_predicate,
    )
    source = TensorProgram(
        (owner, streamed, consumer_operand),
        (producer, score_map, maximum, center, exponential, normalizer, consumer, finalizer),
        (finalizer.output,),
        algebra.task_program.numerical_policy,
    )
    if relation_layout.public_owner_shape != owner.shape:
        raise ValueError("canonical owner axes disagree with the selected SM90 relation packing")
    return source, emitter_region


def _rename_scalar_input(expression: ScalarExpression, source_name: str, target_name: str) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        return replace(expression, input_name=target_name) if expression.input_name == source_name else expression
    return replace(
        expression,
        operands=tuple(_rename_scalar_input(operand, source_name, target_name) for operand in expression.operands),
    )


def _verify_domain_schedule(region: NormalizedExpContractRegion, parameters: Any) -> None:
    if not isinstance(parameters, dict):
        raise ValueError("SM90 schedule parameters must be a canonical mapping")
    expected = (
        ()
        if region.domain_predicate is None
        else (
            affine_predicate_in_domain_position_basis(
                region.domain_predicate,
                region.score_map.output.axes,
            ).canonical_record,
        )
    )
    actual = parameters.get("domain_predicates")
    if actual != expected:
        raise ValueError("attached SM90 schedule domain predicate disagrees with canonical algebra")


def _required_int(record: Any, key: str) -> int:
    if not isinstance(record, dict) or type(record.get(key)) is not int:
        raise ValueError(f"attached SM90 schedule requires integer parameter {key!r}")
    return record[key]


def _compile_request(
    request: ExecutablePhysicalRequest,
    region: NormalizedExpContractRegion,
    schedule: Sm90ForwardPipelineSchedule,
    *,
    physical_schedule_record: dict[str, Any],
    boundary_event_names: tuple[str, ...],
    event_lowerings: tuple[ExecutableEventLowering, ...],
    schedule_local_refinement: ScheduleLocalTaskRefinement | None = None,
    internal_event_legalization: InternalEventDeviceLegalizationRegion | None = None,
) -> CutePipelineCompileRequest:
    relation_layout = schedule.contract.relation_layout
    assert relation_layout is not None
    storage_conversion = finalizer_storage_conversion(
        region.finalizer.expression,
        destination_dtype=region.finalizer.output.dtype,
    )
    values = (*region.producer.inputs, region.consumer.inputs[1], region.finalizer.output)
    return CutePipelineCompileRequest(
        schema_version=4,
        target=request.target.value,
        numerical_policy=request.numerical_policy.value,
        physical_template="sm90_contract_fold_contract",
        tensors=tuple({"name": value.name, "shape": value.shape, "element_type": value.dtype.value} for value in values),
        scalar_programs={
            "score_map": json.loads(serialize_scalar_expression(region.score_map.expression)),
            "finalizer": json.loads(serialize_scalar_expression(region.finalizer.expression)),
        },
        index_relations=(
            {"contract": "producer", "maps": _contract_index_maps(region.producer.input_index_maps)},
            {"contract": "consumer", "maps": _contract_index_maps(region.consumer.input_index_maps)},
            {
                "contract": "domain",
                "predicate": None if region.domain_predicate is None else region.domain_predicate.canonical_record,
            },
            {
                "contract": "public_abi",
                "mapping": _public_abi_record(relation_layout),
            },
        ),
        contraction_dimensions={
            "producer_features": _axes_record(region.dimensions.producer_features),
            "normalized_axes": _axes_record(region.dimensions.normalized_axes),
            "consumer_features": _axes_record(region.dimensions.consumer_features),
            "shared_relation_axes": _axes_record(region.dimensions.shared_relation_axes),
            "relation_group_axes": _axes_record(region.dimensions.relation_group_axes),
        },
        schedule={
            "family": schedule.contract.family,
            "contract": asdict(schedule.contract),
            "pipeline": {
                "stages": schedule.stages,
                "resident_left_in_registers": schedule.resident_left_in_registers,
                "intra_warpgroup_overlap": schedule.intra_warpgroup_overlap,
                "consumer_register_source": schedule.consumer_register_source,
            },
            "attached_schedule": physical_schedule_record,
            "boundary_event_names": boundary_event_names,
            "source_event_lowering": tuple(lowering.canonical_record for lowering in event_lowerings),
            "schedule_local_refinement": (
                None if schedule_local_refinement is None else schedule_local_refinement.canonical_record
            ),
            "internal_event_legalization": (
                None
                if internal_event_legalization is None
                else internal_event_device_legalization_region_record(internal_event_legalization)
            ),
            "finalizer_storage_conversion": None if storage_conversion is None else storage_conversion.canonical_record,
        },
        algebra_identity={
            "request_algebra_fingerprint": request.algebra_fingerprint,
            "source_algebra_fingerprint": request.provenance.get("source_algebra_fingerprint"),
            "transformed_algebra_fingerprint": request.provenance.get("transformed_algebra_fingerprint"),
        },
    )


def _contract_index_maps(index_maps: tuple[tuple[AxisIndexMap, ...], ...]) -> tuple[object, ...]:
    return tuple(
        tuple(
            {
                "domain_axis": mapping.domain_axis.id,
                "operand_axis": mapping.operand_axis.id,
                "divisor": mapping.divisor,
                "offset": mapping.offset,
                "modulus": mapping.modulus,
            }
            for mapping in operand
        )
        for operand in index_maps
    )


def _public_abi_record(relation_layout) -> dict[str, object]:
    owner_shape = relation_layout.owner_physical_shape(relation_layout.producer_feature_extent)
    output_shape = relation_layout.owner_physical_shape(relation_layout.consumer_feature_extent)
    return {
        "coordinate_basis": "physical_dimension_position",
        "owner": {
            "shape": owner_shape,
            "required_strides": _row_major_strides(owner_shape),
            "dimensions": ("batch", "owner", "flat_relation", "producer_feature"),
        },
        "streamed_producer": {
            "shape": relation_layout.streamed_physical_shape(relation_layout.producer_feature_extent),
            "dimensions": ("batch", "streamed", "shared_relation", "producer_feature"),
        },
        "streamed_consumer": {
            "shape": relation_layout.streamed_physical_shape(relation_layout.consumer_feature_extent),
            "dimensions": ("batch", "streamed", "shared_relation", "consumer_feature"),
        },
        "output": {
            "shape": output_shape,
            "required_strides": _row_major_strides(output_shape),
            "dimensions": ("batch", "owner", "flat_relation", "consumer_feature"),
        },
        "flat_relation_maps": {
            "shared": _axis_index_map_record(relation_layout.shared_relation_map),
            "groups": tuple(_axis_index_map_record(mapping) for mapping in relation_layout.group_relation_maps),
        },
    }


def _axis_index_map_record(mapping: AxisIndexMap) -> dict[str, object]:
    return {
        "domain_axis": mapping.domain_axis.id,
        "operand_axis": mapping.operand_axis.id,
        "divisor": mapping.divisor,
        "offset": mapping.offset,
        "modulus": mapping.modulus,
    }


def _axes_record(axes: tuple[TensorAxis, ...]) -> tuple[dict[str, object], ...]:
    return tuple({"id": axis.id, "extent": axis.extent, "label": axis.label} for axis in axes)
