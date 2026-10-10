# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frontend-neutral physical schedule semantics at a logical task quotient."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from tile_lifetime.physical_task_contract import ExecutablePhysicalRequest, canonical_fingerprint
from tile_lifetime.task_plan_refinement import CanonicalTaskPlanRefinement


class PhysicalSemanticsStatus(StrEnum):
    """Whether two owner-scoped physical realizations are comparable and equal."""

    EXACT = "exact"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CanonicalPhysicalOwnerSemantics:
    """Physical realization of one logical quotient owner."""

    owner_task_id: int
    task_realizations: tuple[dict[str, object], ...]
    schedules: tuple[dict[str, object], ...]
    concrete_events: tuple[dict[str, object], ...]
    parametric_events: tuple[dict[str, object], ...]
    generated_module_bindings: tuple[dict[str, object], ...] | None

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return stable semantics without schedule-family display names."""
        return {
            "schema": "shuttle.canonical-physical-owner-semantics.v2",
            "owner_task": self.owner_task_id,
            "logical_coverage": tuple(
                {
                    "task_slot": task["task_slot"],
                    "operation_ordinals": task["operation_ordinals"],
                    "operation_kinds": task["operation_kinds"],
                }
                for task in self.task_realizations
            ),
            "physical_dimensions": self.physical_dimensions,
        }

    @property
    def physical_dimensions(self) -> dict[str, object]:
        """Return the independent physical dimensions that must converge."""
        return {
            "phases": tuple(
                {"task_slot": task["task_slot"], "phase_graph": task["phase_graph"]} for task in self.task_realizations
            ),
            "ownership_and_tiles": tuple(
                {
                    "task_slot": task["task_slot"],
                    **cast(dict[str, object], task["ownership_and_tiles"]),
                }
                for task in self.task_realizations
            ),
            "traversal_and_order": {
                "tasks": tuple(
                    {"task_slot": task["task_slot"], "traversal": task["traversal"]} for task in self.task_realizations
                ),
                "schedules": tuple(
                    {
                        "schedule_slot": schedule["schedule_slot"],
                        "enforced_orders": schedule["enforced_orders"],
                    }
                    for schedule in self.schedules
                ),
            },
            "dtype_and_layout": tuple(
                {"task_slot": task["task_slot"], **cast(dict[str, object], task["dtype_and_layout"])}
                for task in self.task_realizations
            ),
            "relation_accesses": tuple(
                {"task_slot": task["task_slot"], "accesses": task["accesses"]} for task in self.task_realizations
            ),
            "resources": tuple(
                {"task_slot": task["task_slot"], **cast(dict[str, object], task["resources"])}
                for task in self.task_realizations
            ),
            "schedule_implementation": tuple(
                {
                    "schedule_slot": schedule["schedule_slot"],
                    "task_slots": schedule["task_slots"],
                    "family_semantics": schedule["family_semantics"],
                    "parameters": schedule["parameters"],
                }
                for schedule in self.schedules
            ),
            "event_allocation_and_realization": {
                "concrete": self.concrete_events,
                "parametric": self.parametric_events,
            },
            "generated_module_bindings": self.generated_module_bindings,
        }

    @property
    def fingerprint(self) -> str:
        """Fingerprint this owner-scoped physical realization."""
        return canonical_fingerprint(self.canonical_record)


@dataclass(frozen=True)
class PhysicalOwnerSemanticsComparison:
    """Exact field comparison for one logical owner."""

    owner_task_id: int
    status: PhysicalSemanticsStatus
    executable_fingerprint: str
    native_fingerprint: str
    differing_fields: tuple[str, ...]
    differing_paths: tuple[str, ...]

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return the positive or negative owner-scoped evidence."""
        return {
            "owner_task": self.owner_task_id,
            "status": self.status.value,
            "executable_fingerprint": self.executable_fingerprint,
            "native_fingerprint": self.native_fingerprint,
            "differing_fields": self.differing_fields,
            "differing_paths": self.differing_paths,
        }


@dataclass(frozen=True)
class CrossPipelinePhysicalSemanticsComparison:
    """Owner-by-owner physical comparison beneath frontend family names."""

    logical_quotient_fingerprint: str
    executable_request_fingerprint: str
    native_request_fingerprint: str
    owners: tuple[PhysicalOwnerSemanticsComparison, ...]

    @property
    def accepted(self) -> bool:
        """Return whether every owner has complete exact physical semantics."""
        return bool(self.owners) and all(owner.status is PhysicalSemanticsStatus.EXACT for owner in self.owners)

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return stable comparison evidence without replacing endpoint identity."""
        return {
            "schema": "shuttle.cross-pipeline-physical-semantics.v2",
            "accepted": self.accepted,
            "logical_quotient_fingerprint": self.logical_quotient_fingerprint,
            "executable_request_fingerprint": self.executable_request_fingerprint,
            "native_request_fingerprint": self.native_request_fingerprint,
            "owners": tuple(owner.canonical_record for owner in self.owners),
        }

    @property
    def fingerprint(self) -> str:
        """Fingerprint the complete physical comparison evidence."""
        return canonical_fingerprint(self.canonical_record)


def project_physical_schedule_semantics(
    request: ExecutablePhysicalRequest,
    refinement: CanonicalTaskPlanRefinement,
    *,
    generated_module_bindings: Mapping[int, str] | None = None,
) -> tuple[CanonicalPhysicalOwnerSemantics, ...]:
    """Project a physical request onto its proved logical quotient owners."""
    if refinement.refined_plan.fingerprint != request.task_plan.fingerprint:
        raise ValueError("physical semantics projection refinement does not cover its request")
    if generated_module_bindings is not None and set(generated_module_bindings) != set(range(len(request.schedules))):
        raise ValueError("generated module bindings must cover every physical schedule exactly once")
    coarse_by_task = {binding.refined_task_id: binding.coarse_task_id for binding in refinement.bindings}
    owners = tuple(task.id for task in refinement.coarse_plan.tasks)
    task_ids_by_owner = {
        owner: tuple(task.id for task in request.task_plan.tasks if coarse_by_task[task.id] == owner) for owner in owners
    }
    task_slot_by_owner = {
        owner: {task_id: slot for slot, task_id in enumerate(task_ids)} for owner, task_ids in task_ids_by_owner.items()
    }
    schedule_indices_by_owner: dict[int, list[int]] = {owner: [] for owner in owners}
    for schedule_index, schedule in enumerate(request.schedules):
        schedule_owners = {coarse_by_task[task_id] for task_id in schedule.task_ids}
        if len(schedule_owners) != 1:
            raise ValueError("one physical schedule cannot span multiple logical quotient owners")
        schedule_indices_by_owner[next(iter(schedule_owners))].append(schedule_index)
    request_record = request.canonical_record
    event_plan_records = cast(tuple[object, ...], request_record["event_plans"])
    event_allocation_records = cast(tuple[object, ...], request_record["event_allocations"])
    event_realization_records = cast(
        tuple[object, ...],
        cast(dict[str, object], request_record["event_realizations"])["audit"],
    )
    event_program_records = cast(tuple[object, ...], request_record["event_programs"])
    result = []
    for owner in owners:
        task_ids = task_ids_by_owner[owner]
        task_slot = {task_id: slot for slot, task_id in enumerate(task_ids)}
        schedule_indices = tuple(schedule_indices_by_owner[owner])
        concrete_event_indices = tuple(
            index
            for index, event in enumerate(request.event_plans)
            if owner
            in {
                coarse_by_task[_task_id_for_family(request, event.required_dependence.relation.source)],
                coarse_by_task[_task_id_for_family(request, event.required_dependence.relation.target)],
            }
        )
        parametric_event_indices = tuple(
            index
            for index, event in enumerate(request.event_programs)
            if owner
            in {
                coarse_by_task[request.task_plan.dependence_programs[event.dependence_program_index].source_task_id],
                coarse_by_task[request.task_plan.dependence_programs[event.dependence_program_index].target_task_id],
            }
        )
        result.append(
            CanonicalPhysicalOwnerSemantics(
                owner,
                tuple(
                    {
                        "task_slot": task_slot[task_id],
                        "operation_ordinals": request.task_plan.tasks[task_id].operation_ordinals,
                        "operation_kinds": tuple(
                            kind.value for kind in request.task_plan.tasks[task_id].operation_kinds
                        ),
                        "phase_graph": {
                            "phases": request.task_plan.tasks[task_id].structure.schedule_record["phases"],
                            "edges": request.task_plan.tasks[task_id].structure.schedule_record["phase_edges"],
                        },
                        "ownership_and_tiles": {
                            role: request.task_plan.tasks[task_id].structure.schedule_record[role]
                            for role in (
                                "owner_axes",
                                "streamed_axes",
                                "reduction_axes",
                                "contraction_axes",
                            )
                        },
                        "traversal": request.task_plan.tasks[task_id].structure.traversal.value,
                        "dtype_and_layout": {
                            "element_types": request.task_plan.tasks[task_id].structure.element_types,
                            "layout_classes": request.task_plan.tasks[task_id].structure.layout_classes,
                        },
                        "resources": request.task_plan.tasks[task_id].structure.schedule_record["resources"],
                        "accesses": tuple(
                            {
                                "value": access.value_id,
                                "kind": access.kind.value,
                                "phase": access.phase_id,
                                "relation_class": access.relation_class.value,
                                "relation": dict(access.relation_record),
                            }
                            for access in request.task_plan.tasks[task_id].accesses
                        ),
                    }
                    for task_id in task_ids
                ),
                tuple(
                    {
                        "schedule_slot": slot,
                        "task_slots": tuple(task_slot[task_id] for task_id in request.schedules[index].task_ids),
                        "family_semantics": _without_key(request.schedules[index].family.canonical_record, "name"),
                        "parameters": dict(request.schedules[index].parameters),
                        "enforced_orders": tuple(
                            {
                                "kind": order.kind.value,
                                "source_task_slot": task_slot[order.source_task_id],
                                "target_task_slot": task_slot[order.target_task_id],
                                "mechanism": dict(order.mechanism_record),
                            }
                            for order in request.schedules[index].enforced_orders
                        ),
                    }
                    for slot, index in enumerate(schedule_indices)
                ),
                tuple(
                    {
                        "plan": _normalize_concrete_event_plan(
                            event_plan_records[index],
                            coarse_by_task,
                            task_slot_by_owner,
                            owner,
                        ),
                        "allocation": _without_key(event_allocation_records[index], "event"),
                        "realization": _without_key(event_realization_records[index], "event"),
                        "erasure_witness": _normalized_event_witness(
                            request,
                            event_index=index,
                            schedule_indices_by_owner=schedule_indices_by_owner,
                            coarse_by_task=coarse_by_task,
                        ),
                    }
                    for index in concrete_event_indices
                ),
                tuple(
                    {
                        "source": _normalized_task_endpoint(
                            request.task_plan.dependence_programs[event.dependence_program_index].source_task_id,
                            coarse_by_task,
                            task_slot_by_owner,
                            owner,
                        ),
                        "target": _normalized_task_endpoint(
                            request.task_plan.dependence_programs[event.dependence_program_index].target_task_id,
                            coarse_by_task,
                            task_slot_by_owner,
                            owner,
                        ),
                        "event": _without_key(event_program_records[index], "dependence_program"),
                    }
                    for index in parametric_event_indices
                    for event in (request.event_programs[index],)
                ),
                (
                    None
                    if generated_module_bindings is None
                    else tuple(
                        {"schedule_slot": slot, "module_binding": generated_module_bindings[index]}
                        for slot, index in enumerate(schedule_indices)
                    )
                ),
            )
        )
    return tuple(result)


def compare_physical_schedule_semantics(
    executable_request: ExecutablePhysicalRequest,
    executable_refinement: CanonicalTaskPlanRefinement,
    native_request: ExecutablePhysicalRequest,
    native_refinement: CanonicalTaskPlanRefinement,
    *,
    executable_module_bindings: Mapping[int, str] | None = None,
    native_module_bindings: Mapping[int, str] | None = None,
) -> CrossPipelinePhysicalSemanticsComparison:
    """Compare canonical physical semantics owner-by-owner."""
    if executable_refinement.quotient_fingerprint != native_refinement.quotient_fingerprint:
        raise ValueError("physical semantics comparison requires one logical quotient")
    executable = project_physical_schedule_semantics(
        executable_request,
        executable_refinement,
        generated_module_bindings=executable_module_bindings,
    )
    native = project_physical_schedule_semantics(
        native_request,
        native_refinement,
        generated_module_bindings=native_module_bindings,
    )
    if tuple(owner.owner_task_id for owner in executable) != tuple(owner.owner_task_id for owner in native):
        raise ValueError("physical semantics projections disagree on logical owners")
    comparisons = []
    for executable_owner, native_owner in zip(executable, native, strict=True):
        executable_dimensions = executable_owner.physical_dimensions
        native_dimensions = native_owner.physical_dimensions
        differing = tuple(
            field for field in executable_dimensions if executable_dimensions[field] != native_dimensions[field]
        )
        differing_paths = differing
        unavailable = (
            executable_owner.generated_module_bindings is None or native_owner.generated_module_bindings is None
        )
        if unavailable:
            differing = (
                *(field for field in differing if field != "generated_module_bindings"),
                "generated_module_bindings:unavailable",
            )
        semantic_mismatch = any(field != "generated_module_bindings:unavailable" for field in differing)
        status = (
            PhysicalSemanticsStatus.MISMATCH
            if semantic_mismatch
            else PhysicalSemanticsStatus.UNAVAILABLE if unavailable else PhysicalSemanticsStatus.EXACT
        )
        comparisons.append(
            PhysicalOwnerSemanticsComparison(
                executable_owner.owner_task_id,
                status,
                executable_owner.fingerprint,
                native_owner.fingerprint,
                differing,
                differing_paths,
            )
        )
    return CrossPipelinePhysicalSemanticsComparison(
        executable_refinement.quotient_fingerprint,
        executable_request.fingerprint,
        native_request.fingerprint,
        tuple(comparisons),
    )


def _task_id_for_family(request: ExecutablePhysicalRequest, family: object) -> int:
    for task in request.task_plan.tasks:
        if task.family == family:
            return task.id
    raise ValueError("Event relation references an unavailable task family")


def _without_key(record: object, key: str) -> dict[str, object]:
    if not isinstance(record, dict):
        raise TypeError("canonical physical request records must be dictionaries")
    return {name: value for name, value in record.items() if name != key}


def _normalize_concrete_event_plan(
    record: object,
    coarse_by_task: Mapping[int, int],
    task_slot_by_owner: Mapping[int, Mapping[int, int]],
    projected_owner: int,
) -> dict[str, object]:
    if not isinstance(record, dict):
        raise TypeError("canonical Event plan record must be a dictionary")
    result = _without_key(record, "event")
    required = dict(result["required_dependence"])
    relation: dict[str, object] = dict(required["relation"])
    relation["source_task"] = _normalized_task_endpoint(
        relation["source_task"], coarse_by_task, task_slot_by_owner, projected_owner
    )
    relation["target_task"] = _normalized_task_endpoint(
        relation["target_task"], coarse_by_task, task_slot_by_owner, projected_owner
    )
    required["relation"] = relation
    result["required_dependence"] = required
    notify: dict[str, object] = dict(result["notify_relation"])
    notify["producer_task"] = _normalized_task_endpoint(
        notify["producer_task"], coarse_by_task, task_slot_by_owner, projected_owner
    )
    result["notify_relation"] = notify
    trigger: dict[str, object] = dict(result["trigger_relation"])
    trigger["consumer_task"] = _normalized_task_endpoint(
        trigger["consumer_task"], coarse_by_task, task_slot_by_owner, projected_owner
    )
    result["trigger_relation"] = trigger
    return result


def _normalized_task_endpoint(
    task_id: int,
    coarse_by_task: Mapping[int, int],
    task_slot_by_owner: Mapping[int, Mapping[int, int]],
    projected_owner: int,
) -> dict[str, object]:
    owner = coarse_by_task[task_id]
    return {
        "owner_task": owner,
        "local_task_slot": task_slot_by_owner[owner][task_id] if owner == projected_owner else None,
    }


def _normalized_event_witness(
    request: ExecutablePhysicalRequest,
    *,
    event_index: int,
    schedule_indices_by_owner: Mapping[int, list[int]],
    coarse_by_task: Mapping[int, int],
) -> dict[str, object] | None:
    plan_name = request.event_plans[event_index].name
    witness = next(
        (item for item in request.event_realizations.erasure_witnesses if item.plan_name == plan_name),
        None,
    )
    if witness is None:
        return None
    schedule = request.schedules[witness.schedule_index]
    owners = {coarse_by_task[task_id] for task_id in schedule.task_ids}
    if len(owners) != 1:
        raise ValueError("an Event erasure schedule must belong to one logical quotient owner")
    owner = next(iter(owners))
    return {
        "owner_task": owner,
        "schedule_slot": schedule_indices_by_owner[owner].index(witness.schedule_index),
        "order_slot": witness.order_index,
    }
