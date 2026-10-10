# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose independently eligible generic SM100 schedule regions."""

from dataclasses import dataclass, replace
from itertools import product
from typing import Any

from tile_lifetime.event_buffering import (
    EventRealizationKind,
    erased_event_realization,
    physical_event_realization,
    verify_event_realizations,
)
from tile_lifetime.event_dataflow import EventDataflowProgram
from tile_lifetime.experimental_msa_recovery import NaturalProjectedRoutedAttentionCompilation
from tile_lifetime.physical_task_contract import (
    EventErasureWitness,
    EventRealizationProof,
    ExecutablePhysicalRequest,
    PhysicalNumericalPolicy,
    PhysicalSchedule,
    PhysicalTarget,
    TaskCapability,
    canonical_fingerprint,
)
from tile_lifetime.sm100_routed_lowering import (
    SM100RoutedStreamingLowering,
    SM100RoutedTaskLowering,
    default_sm100_routed_schedules,
    lower_sm100_routed_streaming_program,
    select_sm100_routed_lowerings,
)
from tile_lifetime.sm100_selection_lowering import (
    SM100ProjectedSelectionLowering,
    SM100SelectionTaskLowering,
    default_sm100_selection_schedules,
    lower_sm100_projected_selection,
    select_sm100_selection_lowerings,
)


@dataclass(frozen=True)
class SM100ComposedTaskLowering:
    """Compatible generic Selection and bounded-relation schedule attachments."""

    selection: SM100SelectionTaskLowering
    routed: SM100RoutedTaskLowering


@dataclass(frozen=True)
class SM100ComposedTaskLoweringSet:
    """All compatible SM100 physical regions found in one canonical request."""

    source_request_fingerprint: str
    candidates: tuple[SM100ComposedTaskLowering, ...]


@dataclass(frozen=True)
class SM100RuntimeRelationABI:
    """Runtime relation inputs retained at the final physical boundary."""

    source_task_id: int
    target_task_id: int
    relation_program_fingerprint: str
    relation_program: dict[str, Any]
    algebra_value_ordinals: tuple[int, ...]
    source_coordinate_bounds: tuple[tuple[int, int], ...]
    target_coordinate_bounds: tuple[tuple[int, int], ...]
    has_validity: bool
    event_capacity: int
    event_generation_count: int
    event_realization: str

    def __post_init__(self) -> None:
        if canonical_fingerprint(self.relation_program) != self.relation_program_fingerprint:
            raise ValueError("runtime relation fingerprint does not match its canonical program")
        if self.event_capacity <= 0 or self.event_generation_count <= 0:
            raise ValueError("runtime relation Events require bounded positive storage")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "source_task_id": self.source_task_id,
            "target_task_id": self.target_task_id,
            "relation_program_fingerprint": self.relation_program_fingerprint,
            "relation_program": self.relation_program,
            "algebra_value_ordinals": self.algebra_value_ordinals,
            "source_coordinate_bounds": self.source_coordinate_bounds,
            "target_coordinate_bounds": self.target_coordinate_bounds,
            "has_validity": self.has_validity,
            "event_capacity": self.event_capacity,
            "event_generation_count": self.event_generation_count,
            "event_realization": self.event_realization,
        }


@dataclass(frozen=True)
class SM100ScalarProgramABI:
    """One scalar phase retained with its canonical task/phase provenance."""

    task_id: int
    phase_id: int
    operation_ordinal: int
    operation: dict[str, Any]

    def __post_init__(self) -> None:
        if min(self.task_id, self.phase_id, self.operation_ordinal) < 0:
            raise ValueError("scalar-program task, phase, and operation ids must be non-negative")
        if self.operation.get("kind") != "map" or "expression" not in self.operation:
            raise ValueError("a scalar-program ABI entry must retain one canonical Map expression")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "phase_id": self.phase_id,
            "operation_ordinal": self.operation_ordinal,
            "operation": self.operation,
        }


@dataclass(frozen=True)
class SM100CanonicalEmitterABI:
    """Canonical task geometry presented to an SM100 emitter.

    Scalar programs remain canonical algebra records.  This boundary does not
    recover score scales, causal modes, or workload identities from them.
    """

    source_request_fingerprint: str
    scheduled_request_fingerprint: str
    algebra_fingerprint: str
    task_plan_fingerprint: str
    selection_schedule_fingerprint: str
    routed_schedule_fingerprint: str
    selection_task_ids: tuple[int, ...]
    routed_task_ids: tuple[int, ...]
    left_extent: int
    group_extent: int
    left_lane_extent: int
    selected_extent: int
    score_feature_extent: int
    value_feature_extent: int
    fold_extent: int
    selection_parameters: dict[str, Any]
    routed_parameters: dict[str, Any]
    scalar_programs: tuple[SM100ScalarProgramABI, ...]
    runtime_relations: tuple[SM100RuntimeRelationABI, ...]
    residual_task_ids: tuple[int, ...]
    retained_event_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        dimensions = (
            self.left_extent,
            self.group_extent,
            self.left_lane_extent,
            self.selected_extent,
            self.score_feature_extent,
            self.value_feature_extent,
            self.fold_extent,
        )
        if min(dimensions) <= 0:
            raise ValueError("SM100 emitter dimensions must be positive")
        if set(self.selection_task_ids) & set(self.routed_task_ids):
            raise ValueError("SM100 emitter regions must own disjoint task sets")

    @property
    def complete_existing_emitter_coverage(self) -> bool:
        """Whether no task or synchronization boundary remains outside the emitter."""
        return not self.residual_task_ids and not self.retained_event_ids and not self.runtime_relations

    @property
    def existing_emitter_constructor(self) -> dict[str, Any]:
        """Return only geometry understood by the extracted physical class."""
        return {
            "head_dim": self.value_feature_extent,
            "qheadperkv": self.left_lane_extent,
            "m_block_size": self.routed_parameters["packed_left_rows"],
            "n_block_size": self.routed_parameters["right_block_size"],
            "use_prepare_scheduler": True,
        }

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "source_request_fingerprint": self.source_request_fingerprint,
            "scheduled_request_fingerprint": self.scheduled_request_fingerprint,
            "algebra_fingerprint": self.algebra_fingerprint,
            "task_plan_fingerprint": self.task_plan_fingerprint,
            "selection_schedule_fingerprint": self.selection_schedule_fingerprint,
            "routed_schedule_fingerprint": self.routed_schedule_fingerprint,
            "selection_task_ids": self.selection_task_ids,
            "routed_task_ids": self.routed_task_ids,
            "geometry": {
                "left_extent": self.left_extent,
                "group_extent": self.group_extent,
                "left_lane_extent": self.left_lane_extent,
                "selected_extent": self.selected_extent,
                "score_feature_extent": self.score_feature_extent,
                "value_feature_extent": self.value_feature_extent,
                "fold_extent": self.fold_extent,
            },
            "selection_parameters": self.selection_parameters,
            "routed_parameters": self.routed_parameters,
            "scalar_programs": tuple(program.canonical_record for program in self.scalar_programs),
            "runtime_relations": tuple(relation.canonical_record for relation in self.runtime_relations),
            "residual_task_ids": self.residual_task_ids,
            "retained_event_ids": self.retained_event_ids,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_record)


@dataclass(frozen=True)
class SM100ComposedPhysicalLowering:
    """A full scheduled request and its final canonical emitter ABI."""

    request: ExecutablePhysicalRequest
    selected: SM100ComposedTaskLowering
    emitter_abi: SM100CanonicalEmitterABI


def select_sm100_composed_task_lowerings(
    request: ExecutablePhysicalRequest,
) -> SM100ComposedTaskLoweringSet:
    """Select compatible regions without recognizing a source workload graph."""
    if request.target is not PhysicalTarget.SM100:
        return SM100ComposedTaskLoweringSet(request.fingerprint, ())
    selection_candidates = select_sm100_selection_lowerings(request)
    routed_candidates = select_sm100_routed_lowerings(request)
    candidates = tuple(
        SM100ComposedTaskLowering(selection, routed)
        for selection, routed in product(selection_candidates, routed_candidates)
        if _generic_regions_are_compatible(request, selection, routed)
    )
    return SM100ComposedTaskLoweringSet(request.fingerprint, candidates)


def lower_sm100_composed_physical_request(
    request: ExecutablePhysicalRequest,
    selected: SM100ComposedTaskLowering,
) -> SM100ComposedPhysicalLowering:
    """Install two independently selected regions into one complete request.

    Events internal to either fused schedule erase only when one of that
    schedule's exact program-order relations covers the Event relation.  All
    crossing Events and runtime-parametric relations remain physical.
    """
    if request.target is not PhysicalTarget.SM100:
        raise ValueError("SM100 schedule attachment requires an SM100 request")
    if request.numerical_policy is not PhysicalNumericalPolicy.FAST:
        raise ValueError("the current reassociated SM100 normalized-Fold schedule requires FAST numerical policy")
    if selected.selection.source_request_fingerprint != request.fingerprint:
        raise ValueError("Selection candidate does not derive from the supplied request")
    if selected.routed.source_request_fingerprint != request.fingerprint:
        raise ValueError("routed candidate does not derive from the supplied request")
    if not _generic_regions_are_compatible(request, selected.selection, selected.routed):
        raise ValueError("selected SM100 task regions are not structurally compatible")

    replacement_schedules = (
        selected.selection.physical_schedule,
        selected.routed.physical_schedule,
    )
    replaced_task_ids = {task_id for schedule in replacement_schedules for task_id in schedule.task_ids}
    retained = []
    removed_task_ids: set[int] = set()
    for schedule in request.schedules:
        overlap = set(schedule.task_ids) & replaced_task_ids
        if not overlap:
            retained.append(schedule)
            continue
        if overlap != set(schedule.task_ids):
            raise ValueError("selected schedule partially overlaps an existing physical schedule")
        removed_task_ids.update(schedule.task_ids)
    if removed_task_ids != replaced_task_ids:
        raise ValueError("selected schedules do not replace the existing cover exactly")
    schedules = tuple(sorted((*retained, *replacement_schedules), key=lambda schedule: min(schedule.task_ids)))

    program = EventDataflowProgram(request.task_plan.task_families, request.task_plan.dependences, request.event_plans)
    realizations = []
    witnesses = []
    for plan in request.event_plans:
        covering = _covering_schedule_order(plan.scheduled_dependence, schedules)
        if covering is None:
            realizations.append(
                physical_event_realization(
                    plan,
                    mechanism="generation_counted_event",
                    reason="the selected fused schedules do not own both endpoints of this exact relation",
                )
            )
            continue
        schedule_index, order_index = covering
        order = schedules[schedule_index].enforced_orders[order_index]
        realizations.append(
            erased_event_realization(
                plan,
                kind=EventRealizationKind.ERASED_PROGRAM_ORDER,
                mechanism="selected fused phase order",
                reason="one selected schedule owns both tasks and covers the exact Event relation",
                ordering=order.relation,
            )
        )
        witnesses.append(EventErasureWitness(plan.name, schedule_index, order_index))
    audit = verify_event_realizations(program, tuple(realizations))
    scheduled = replace(
        request,
        schedules=schedules,
        event_realizations=EventRealizationProof(audit, tuple(witnesses)),
        provenance={
            **request.provenance,
            "physical_attachment": {
                "source_request_fingerprint": request.fingerprint,
                "selection_schedule_fingerprint": selected.selection.physical_schedule.fingerprint,
                "routed_schedule_fingerprint": selected.routed.physical_schedule.fingerprint,
            },
        },
    )
    emitter_abi = _canonical_emitter_abi(request, scheduled, selected)
    return SM100ComposedPhysicalLowering(scheduled, selected, emitter_abi)


def _covering_schedule_order(relation, schedules: tuple[PhysicalSchedule, ...]) -> tuple[int, int] | None:
    matches = []
    for schedule_index, schedule in enumerate(schedules):
        for order_index, order in enumerate(schedule.enforced_orders):
            if order.source_task_id not in schedule.task_ids or order.target_task_id not in schedule.task_ids:
                continue
            if order.relation.source != relation.source or order.relation.target != relation.target:
                continue
            if set(relation.pairs) <= set(order.relation.pairs):
                matches.append((schedule_index, order_index))
    if len(matches) > 1:
        raise ValueError("one Event relation has multiple fused program-order witnesses")
    return matches[0] if matches else None


def _canonical_emitter_abi(
    source: ExecutablePhysicalRequest,
    scheduled: ExecutablePhysicalRequest,
    selected: SM100ComposedTaskLowering,
) -> SM100CanonicalEmitterABI:
    covered_task_ids = (*selected.selection.task_ids, *selected.routed.task_ids)
    covered_task_id_set = set(covered_task_ids)
    scalar_locations = tuple(
        (task_id, phase.id, ordinal)
        for task_id in covered_task_ids
        for phase in source.task_plan.tasks[task_id].structure.phases
        if phase.capability is TaskCapability.SCALAR_MAP
        for ordinal in phase.operation_ordinals
    )
    algebra_operations = source.algebra_record.get("operations")
    if not isinstance(algebra_operations, (list, tuple)):
        raise ValueError("canonical algebra record does not contain an operation sequence")
    if not all(isinstance(algebra_operations[ordinal], dict) for _, _, ordinal in scalar_locations):
        raise ValueError("canonical scalar operation records must be mappings")
    scalar_programs = tuple(
        SM100ScalarProgramABI(task_id, phase_id, ordinal, dict(algebra_operations[ordinal]))
        for task_id, phase_id, ordinal in scalar_locations
    )

    event_by_dependence = {event.dependence_program_index: event for event in source.event_programs}
    runtime_relations = tuple(
        SM100RuntimeRelationABI(
            source_task_id=dependence.source_task_id,
            target_task_id=dependence.target_task_id,
            relation_program_fingerprint=dependence.relation_program.semantic_fingerprint,
            relation_program=dependence.relation_program.canonical_record,
            algebra_value_ordinals=tuple(binding.algebra_value_ordinal for binding in dependence.runtime_bindings),
            source_coordinate_bounds=tuple(
                (coordinate.lower_bound, coordinate.upper_bound) for coordinate in dependence.source_coordinates
            ),
            target_coordinate_bounds=tuple(
                (coordinate.lower_bound, coordinate.upper_bound) for coordinate in dependence.target_coordinates
            ),
            has_validity=dependence.validity is not None,
            event_capacity=event_by_dependence[dependence_index].capacity,
            event_generation_count=event_by_dependence[dependence_index].generation_count,
            event_realization=event_by_dependence[dependence_index].realization.value,
        )
        for dependence_index, dependence in enumerate(source.task_plan.dependence_programs)
        if dependence.target_task_id in covered_task_id_set or dependence.source_task_id in covered_task_id_set
    )
    retained_event_names = {realization.plan_name for realization in scheduled.event_realizations.audit.physical}
    return SM100CanonicalEmitterABI(
        source_request_fingerprint=source.fingerprint,
        scheduled_request_fingerprint=scheduled.fingerprint,
        algebra_fingerprint=source.algebra_fingerprint,
        task_plan_fingerprint=source.task_plan.fingerprint,
        selection_schedule_fingerprint=selected.selection.physical_schedule.fingerprint,
        routed_schedule_fingerprint=selected.routed.physical_schedule.fingerprint,
        selection_task_ids=selected.selection.task_ids,
        routed_task_ids=selected.routed.task_ids,
        left_extent=selected.routed.left_extent,
        group_extent=selected.routed.group_extent,
        left_lane_extent=selected.routed.left_lane_extent,
        selected_extent=selected.routed.selected_extent,
        score_feature_extent=selected.routed.score_feature_extent,
        value_feature_extent=selected.routed.value_feature_extent,
        fold_extent=selected.routed.fold_extent,
        selection_parameters=dict(selected.selection.physical_schedule.parameters),
        routed_parameters=dict(selected.routed.physical_schedule.parameters),
        scalar_programs=scalar_programs,
        runtime_relations=runtime_relations,
        residual_task_ids=tuple(task.id for task in source.task_plan.tasks if task.id not in covered_task_id_set),
        retained_event_ids=tuple(
            event_id for event_id, plan in enumerate(source.event_plans) if plan.name in retained_event_names
        ),
    )


def _generic_regions_are_compatible(
    request: ExecutablePhysicalRequest,
    selection: SM100SelectionTaskLowering,
    routed: SM100RoutedTaskLowering,
) -> bool:
    if set(selection.task_ids) & set(routed.task_ids):
        return False
    selection_task = request.task_plan.tasks[selection.task_ids[-1]]
    relation_task = request.task_plan.tasks[routed.task_ids[0]]
    connected = any(
        dependence.relation.source == selection_task.family and dependence.relation.target == relation_task.family
        for dependence in request.task_plan.dependences
    )
    selection_owner = selection_task.structure.owner_axes
    relation_owner = relation_task.structure.owner_axes
    return (
        connected
        and tuple(axis.axis_id for axis in selection_owner) == tuple(axis.axis_id for axis in relation_owner)
        and selection.left_extent == routed.left_extent
        and selection.group_extent == routed.group_extent
        and selection.selected_extent == routed.selected_extent
        and selection.right_block_size == routed.fold_extent
    )


@dataclass(frozen=True)
class LoweredAffineIndexDomain:
    """Physical binding of generic left/right positions to an index predicate."""

    left_axis: str
    right_axis: str
    predicate: str
    left_count: int
    right_count: int
    left_position_offset: int
    right_position_offset: int

    @property
    def bottom_right_aligned(self) -> bool:
        """Whether the final left and right positions are identical."""
        left_stop = self.left_position_offset + self.left_count
        right_stop = self.right_position_offset + self.right_count
        return left_stop == right_stop

    def left_position(self, left_index: int) -> int:
        """Map a left-domain index to its absolute position."""
        if left_index < 0 or left_index >= self.left_count:
            raise ValueError(f"left index {left_index} is outside [0, {self.left_count})")
        return self.left_position_offset + left_index

    def right_position(self, right_index: int) -> int:
        """Map a right-domain index to its absolute position."""
        if right_index < 0 or right_index >= self.right_count:
            raise ValueError(f"right index {right_index} is outside [0, {self.right_count})")
        return self.right_position_offset + right_index

    def allows(self, left_index: int, right_index: int) -> bool:
        """Evaluate the lowered predicate for one logical index pair."""
        if self.predicate != "left_greater_equal_right":
            raise ValueError(f"unsupported index predicate {self.predicate!r}")
        return self.left_position(left_index) >= self.right_position(right_index)


@dataclass(frozen=True)
class SM100ProjectedRoutedCandidate:
    """One cross-product point over generic Selection and streaming skeletons."""

    selection: SM100ProjectedSelectionLowering
    streaming: SM100RoutedStreamingLowering
    position_domain: LoweredAffineIndexDomain


@dataclass(frozen=True)
class SM100ProjectedRoutedCandidateSet:
    """Quarantined legacy candidates derived from recovered workload state."""

    source: NaturalProjectedRoutedAttentionCompilation
    candidates: tuple[SM100ProjectedRoutedCandidate, ...]


def lower_sm100_projected_routed_candidates(
    compilation: NaturalProjectedRoutedAttentionCompilation,
) -> SM100ProjectedRoutedCandidateSet:
    """Enumerate legacy benchmark candidates from recovered workload state.

    New compiler paths use :func:`select_sm100_composed_task_lowerings` with a
    canonical executable request. This adapter remains only to preserve the
    already-recorded SM100 physical-kernel evidence.
    """
    selection_program = compilation.recovered.relation_selection
    selection_candidates = tuple(
        lower_sm100_projected_selection(selection_program, schedule) for schedule in default_sm100_selection_schedules()
    )
    streaming_candidates = tuple(
        lower_sm100_routed_streaming_program(compilation.streaming_program, compilation.relation, schedule)
        for schedule in default_sm100_routed_schedules()
    )
    position_domain = LoweredAffineIndexDomain(
        left_axis=selection_program.token_restriction.left_axis,
        right_axis=selection_program.token_restriction.right_axis,
        predicate=selection_program.token_restriction.predicate,
        left_count=selection_program.source_count,
        right_count=selection_program.resolved_right_count,
        left_position_offset=selection_program.left_position_offset,
        right_position_offset=selection_program.right_position_offset,
    )
    _validate_composition(selection_candidates, streaming_candidates, position_domain)
    candidates = tuple(
        SM100ProjectedRoutedCandidate(selection=selection, streaming=streaming, position_domain=position_domain)
        for selection, streaming in product(selection_candidates, streaming_candidates)
    )
    return SM100ProjectedRoutedCandidateSet(source=compilation, candidates=candidates)


def _validate_composition(
    selection_candidates: tuple[SM100ProjectedSelectionLowering, ...],
    streaming_candidates: tuple[SM100RoutedStreamingLowering, ...],
    position_domain: LoweredAffineIndexDomain,
) -> None:
    if not selection_candidates or not streaming_candidates:
        raise ValueError("projected routed lowering requires nonempty candidate families")
    selection = selection_candidates[0]
    streaming = streaming_candidates[0]
    if selection.program.source_count != streaming.query_length:
        raise ValueError("projected Selection left domain does not match streaming query domain")
    if selection.program.resolved_right_count != streaming.key_length:
        raise ValueError("projected Selection right domain does not match streaming key domain")
    if selection.program.group_count != streaming.key_value_heads:
        raise ValueError("projected Selection groups do not match streaming key/value heads")
    if selection.program.selected_count != streaming.selected_count:
        raise ValueError("projected Selection width does not match routed streaming edge slots")
    if streaming.score_map.domain_traversal.value != "lower_triangular":
        raise ValueError("the projected causal relation requires a causal streaming DomainRestriction")
    if position_domain.predicate != "left_greater_equal_right":
        raise ValueError("the initial SM100 composition requires left-position >= right-position")
    if not position_domain.bottom_right_aligned:
        raise ValueError("the initial asymmetric causal composition requires bottom-right alignment")
