# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register a graph-derived physical compiler with Shuttle's JAX boundary."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from shuttle.physical_artifact import (
    activate_gpu_physical_artifact_providers,
    register_gpu_physical_artifact_provider,
)
from tile_lifetime.cute_pipeline_artifact import CuteAffineMap, CuteIntegerSet, CutePipelineArtifact
from tile_lifetime.internal_event_allocation import internal_event_device_legalization_region_record
from tile_lifetime.physical_event_cover import affine_expression_from_record


@dataclass(frozen=True)
class ShuttleTypeRecord:
    """One scalar or statically ranked tensor type in a provider request."""

    kind: str
    element_type: str
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class ShuttleValueRecord:
    """One SSA value with an encoder-local stable identifier."""

    id: int
    type: ShuttleTypeRecord


@dataclass(frozen=True)
class ShuttleBlockRecord:
    """One structured block in a Shuttle operation record."""

    arguments: tuple[ShuttleValueRecord, ...]
    operations: tuple[ShuttleOperationRecord, ...]


@dataclass(frozen=True)
class ShuttleOperationRecord:
    """Typed, recursively structured Shuttle/MLIR operation data."""

    name: str
    attributes: Mapping[str, Any]
    operands: tuple[int, ...]
    results: tuple[ShuttleValueRecord, ...]
    regions: tuple[tuple[ShuttleBlockRecord, ...], ...]

    def walk(self) -> Iterator[ShuttleOperationRecord]:
        """Yield this operation and all nested operations in source order."""
        yield self
        for region in self.regions:
            for block in region:
                for operation in block.operations:
                    yield from operation.walk()

    def body(self) -> ShuttleBlockRecord:
        """Return the sole block of a single-block operation."""
        if len(self.regions) != 1 or len(self.regions[0]) != 1:
            raise ValueError(f"{self.name} does not have exactly one region block")
        return self.regions[0][0]


@dataclass(frozen=True)
class ShuttleExactEventRequirement:
    """One exact bounded-buffer Event requirement awaiting physical realization."""

    ordinal: int
    producer_task: int
    consumer_task: int
    relation_domain_shape: tuple[int, ...]
    relation_predicate: CuteIntegerSet | None
    producer_indices: CuteAffineMap
    consumer_indices: CuteAffineMap
    storage_capacity: int
    storage_slot: CuteAffineMap
    generation_index: CuteAffineMap
    generations: int
    notify_count: int
    scope: str
    release: bool
    acquire: bool
    producer_phase: str | None = None
    consumer_phase: str | None = None


@dataclass(frozen=True)
class ShuttleEventRealizationRecord:
    """Typed physical realization or erasure proof for one logical Event."""

    ordinal: int
    kind: str
    covering_launch_order: tuple[int, ...]


@dataclass(frozen=True)
class ShuttleNormalizedExpContractRegionRecord:
    """Native source bindings for one planner-declared algebraic region."""

    ordinal: int
    producer_operation: int
    score_operation: int
    score_result: int
    fold_operations: tuple[int, ...]
    maximum_operation: int
    maximum_result: int
    normalizer_operation: int
    normalizer_result: int
    normalized_weight_operation: int
    normalized_weight_result: int
    consumer_operation: int
    normalized_operand: int
    finalizer_operation: int
    finalizer_result: int
    output_result: int | None


@dataclass(frozen=True)
class ShuttleNormalizedExpContractPullbackRecord:
    """Native source bindings for one planner-declared pullback."""

    ordinal: int
    forward_region_ordinal: int
    forward_task: int
    objective_task: int | None
    owner_pullback_task: int
    streamed_pullback_task: int
    maximum_state_buffer: int
    normalizer_state_buffer: int
    output_dependent_state_buffer: int
    forward_result_buffer: int | None
    output_cotangent_buffer: int
    terminal_cotangent_buffers: tuple[int, ...]
    normalized_output_operation: int
    post_normalization_vjp_operation: int
    output_product_operation: int
    output_dependent_fold_operation: int
    consumer_weight_cotangent_operation: int
    normalized_weight_operation: int
    normalized_exp_pullback_operation: int
    score_map_vjp_operation: int
    terminal_contract_operations: tuple[int, ...]


@dataclass(frozen=True)
class GpuPhysicalArtifactRequest:
    """Structured Shuttle algebra and verified physical plans from XLA."""

    target: str
    fingerprint: str
    region: ShuttleOperationRecord
    materialization: ShuttleOperationRecord
    schedule: ShuttleOperationRecord
    event_lowering_plan: ShuttleOperationRecord
    binding_indices: tuple[int | None, ...]
    region_result_indices: tuple[int | None, ...]
    reuse_groups: tuple[int, ...]
    event_relations: tuple[ShuttleExactEventRequirement, ...] = ()
    event_realizations: tuple[ShuttleEventRealizationRecord, ...] = ()
    event_requirements: tuple[ShuttleExactEventRequirement, ...] = ()
    normalized_exp_contract_regions: tuple[ShuttleNormalizedExpContractRegionRecord, ...] = ()
    normalized_exp_contract_pullbacks: tuple[ShuttleNormalizedExpContractPullbackRecord, ...] = ()


GpuPhysicalArtifactCompiler = Callable[[GpuPhysicalArtifactRequest], CutePipelineArtifact | bytes | None]


def register_gpu_physical_artifact_compiler(
    compiler: GpuPhysicalArtifactCompiler,
) -> None:
    """Bind one compile-time provider to the process-wide JAX Shuttle hook."""
    register_gpu_physical_artifact_provider(native_gpu_physical_artifact_provider(compiler))
    activate_gpu_physical_artifact_providers(discover=False)


def decode_gpu_physical_artifact_request(serialized: str) -> GpuPhysicalArtifactRequest:
    """Decode and validate the typed request emitted by Shuttle's MLIR pass."""
    record = json.loads(serialized)
    if (
        not isinstance(record, dict)
        or set(record)
        != {
            "schema_version",
            "target",
            "fingerprint",
            "binding_indices",
            "region_result_indices",
            "reuse_groups",
            "region",
            "materialization",
            "schedule",
            "event_lowering_plan",
        }
        or record["schema_version"] != 3
    ):
        raise ValueError("unsupported Shuttle physical artifact request")
    target = record["target"]
    fingerprint = record["fingerprint"]
    if target not in ("sm_90", "sm_100"):
        raise ValueError(f"unsupported Shuttle GPU target: {target!r}")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Shuttle physical artifact request has an invalid fingerprint")
    region = _decode_operation(record["region"])
    materialization = _decode_operation(record["materialization"])
    schedule = _decode_operation(record["schedule"])
    event_lowering_plan = _decode_operation(record["event_lowering_plan"])
    binding_indices = _optional_integer_tuple(record["binding_indices"], "physical binding indices")
    region_result_indices = _optional_integer_tuple(record["region_result_indices"], "region result indices")
    reuse_groups = _integer_tuple(record["reuse_groups"], "physical reuse groups")
    expected_names = (
        (region, "shuttle.region"),
        (materialization, "shuttle.materialization_plan"),
        (schedule, "shuttle.schedule_plan"),
        (event_lowering_plan, "shuttle.event_lowering_plan"),
    )
    for operation, expected_name in expected_names:
        if operation.name != expected_name:
            raise ValueError(f"physical artifact request expected {expected_name}, found {operation.name}")
    event_relations = _decode_schedule_event_relations(schedule)
    event_requirements, event_realizations = _decode_exact_event_requirements(event_lowering_plan)
    normalized_exp_contract_regions = _decode_normalized_exp_contract_regions(materialization)
    normalized_exp_contract_pullbacks = _decode_normalized_exp_contract_pullbacks(materialization)
    return GpuPhysicalArtifactRequest(
        target=target,
        fingerprint=fingerprint,
        region=region,
        materialization=materialization,
        schedule=schedule,
        event_lowering_plan=event_lowering_plan,
        binding_indices=binding_indices,
        region_result_indices=region_result_indices,
        reuse_groups=reuse_groups,
        event_relations=event_relations,
        event_realizations=event_realizations,
        event_requirements=event_requirements,
        normalized_exp_contract_regions=normalized_exp_contract_regions,
        normalized_exp_contract_pullbacks=normalized_exp_contract_pullbacks,
    )


def _decode_normalized_exp_contract_regions(
    materialization: ShuttleOperationRecord,
) -> tuple[ShuttleNormalizedExpContractRegionRecord, ...]:
    encoded = materialization.attributes.get("normalized_exp_contract_regions")
    if encoded is None:
        return ()
    records = _sequence(encoded, "normalized-exp Contract regions")
    decoded: list[ShuttleNormalizedExpContractRegionRecord] = []
    required_fields = {
        "ordinal",
        "producer_operation",
        "score_operation",
        "score_result",
        "fold_operations",
        "maximum_operation",
        "maximum_result",
        "normalizer_operation",
        "normalizer_result",
        "normalized_weight_operation",
        "normalized_weight_result",
        "consumer_operation",
        "normalized_operand",
        "finalizer_operation",
        "finalizer_result",
    }
    for encoded_record in records:
        record = _object(encoded_record, "normalized-exp Contract region")
        if set(record) not in (required_fields, {*required_fields, "output_result"}):
            raise ValueError("normalized-exp Contract region has unknown or missing fields")
        decoded.append(
            ShuttleNormalizedExpContractRegionRecord(
                ordinal=_nonnegative_integer(record["ordinal"], "normalized-exp Contract region ordinal"),
                producer_operation=_nonnegative_integer(
                    record["producer_operation"], "normalized-exp producer operation"
                ),
                score_operation=_nonnegative_integer(record["score_operation"], "normalized-exp score operation"),
                score_result=_nonnegative_integer(record["score_result"], "normalized-exp score result"),
                fold_operations=_integer_tuple(record["fold_operations"], "normalized-exp Fold operations"),
                maximum_operation=_nonnegative_integer(record["maximum_operation"], "normalized-exp maximum operation"),
                maximum_result=_nonnegative_integer(record["maximum_result"], "normalized-exp maximum result"),
                normalizer_operation=_nonnegative_integer(
                    record["normalizer_operation"], "normalized-exp normalizer operation"
                ),
                normalizer_result=_nonnegative_integer(record["normalizer_result"], "normalized-exp normalizer result"),
                normalized_weight_operation=_nonnegative_integer(
                    record["normalized_weight_operation"], "normalized-exp normalized-weight operation"
                ),
                normalized_weight_result=_nonnegative_integer(
                    record["normalized_weight_result"], "normalized-exp normalized-weight result"
                ),
                consumer_operation=_nonnegative_integer(
                    record["consumer_operation"], "normalized-exp consumer operation"
                ),
                normalized_operand=_nonnegative_integer(record["normalized_operand"], "normalized-exp consumer operand"),
                finalizer_operation=_nonnegative_integer(
                    record["finalizer_operation"], "normalized-exp finalizer operation"
                ),
                finalizer_result=_nonnegative_integer(record["finalizer_result"], "normalized-exp finalizer result"),
                output_result=(
                    None
                    if "output_result" not in record
                    else _nonnegative_integer(record["output_result"], "normalized-exp output result")
                ),
            )
        )
    if tuple(region.ordinal for region in decoded) != tuple(range(len(decoded))):
        raise ValueError("normalized-exp Contract region ordinals must be dense and source ordered")
    if any(not region.fold_operations for region in decoded):
        raise ValueError("normalized-exp Contract regions require at least one declared Fold operation")
    return tuple(decoded)


def _decode_normalized_exp_contract_pullbacks(
    materialization: ShuttleOperationRecord,
) -> tuple[ShuttleNormalizedExpContractPullbackRecord, ...]:
    encoded = materialization.attributes.get("normalized_exp_contract_pullbacks")
    if encoded is None:
        return ()
    records = _sequence(encoded, "normalized-exp Contract pullbacks")
    required_fields = {
        "ordinal",
        "forward_region_ordinal",
        "forward_task",
        "owner_pullback_task",
        "streamed_pullback_task",
        "maximum_state_buffer",
        "normalizer_state_buffer",
        "output_dependent_state_buffer",
        "output_cotangent_buffer",
        "terminal_cotangent_buffers",
        "normalized_output_operation",
        "post_normalization_vjp_operation",
        "output_product_operation",
        "output_dependent_fold_operation",
        "consumer_weight_cotangent_operation",
        "normalized_weight_operation",
        "normalized_exp_pullback_operation",
        "score_map_vjp_operation",
        "terminal_contract_operations",
    }
    decoded: list[ShuttleNormalizedExpContractPullbackRecord] = []
    for encoded_record in records:
        record = _object(encoded_record, "normalized-exp Contract pullback")
        if not required_fields <= set(record) or set(record) - required_fields != {
            field for field in ("objective_task", "forward_result_buffer") if field in record
        }:
            raise ValueError("normalized-exp Contract pullback has unknown or missing fields")
        decoded.append(
            ShuttleNormalizedExpContractPullbackRecord(
                ordinal=_nonnegative_integer(record["ordinal"], "normalized-exp pullback ordinal"),
                forward_region_ordinal=_nonnegative_integer(
                    record["forward_region_ordinal"], "normalized-exp pullback forward region"
                ),
                forward_task=_nonnegative_integer(record["forward_task"], "normalized-exp forward task"),
                objective_task=(
                    None
                    if "objective_task" not in record
                    else _nonnegative_integer(record["objective_task"], "normalized-exp objective task")
                ),
                owner_pullback_task=_nonnegative_integer(
                    record["owner_pullback_task"], "normalized-exp owner pullback task"
                ),
                streamed_pullback_task=_nonnegative_integer(
                    record["streamed_pullback_task"], "normalized-exp streamed pullback task"
                ),
                maximum_state_buffer=_nonnegative_integer(
                    record["maximum_state_buffer"], "normalized-exp maximum state buffer"
                ),
                normalizer_state_buffer=_nonnegative_integer(
                    record["normalizer_state_buffer"], "normalized-exp normalizer state buffer"
                ),
                output_dependent_state_buffer=_nonnegative_integer(
                    record["output_dependent_state_buffer"], "normalized-exp output-dependent state buffer"
                ),
                forward_result_buffer=(
                    None
                    if "forward_result_buffer" not in record
                    else _nonnegative_integer(record["forward_result_buffer"], "normalized-exp forward result buffer")
                ),
                output_cotangent_buffer=_nonnegative_integer(
                    record["output_cotangent_buffer"], "normalized-exp output cotangent buffer"
                ),
                terminal_cotangent_buffers=_integer_tuple(
                    record["terminal_cotangent_buffers"], "normalized-exp terminal cotangent buffers"
                ),
                normalized_output_operation=_nonnegative_integer(
                    record["normalized_output_operation"], "normalized-exp normalized-output operation"
                ),
                post_normalization_vjp_operation=_nonnegative_integer(
                    record["post_normalization_vjp_operation"], "normalized-exp finalizer VJP operation"
                ),
                output_product_operation=_nonnegative_integer(
                    record["output_product_operation"], "normalized-exp output-product operation"
                ),
                output_dependent_fold_operation=_nonnegative_integer(
                    record["output_dependent_fold_operation"], "normalized-exp output-dependent Fold operation"
                ),
                consumer_weight_cotangent_operation=_nonnegative_integer(
                    record["consumer_weight_cotangent_operation"],
                    "normalized-exp consumer-weight cotangent operation",
                ),
                normalized_weight_operation=_nonnegative_integer(
                    record["normalized_weight_operation"], "normalized-exp normalized-weight operation"
                ),
                normalized_exp_pullback_operation=_nonnegative_integer(
                    record["normalized_exp_pullback_operation"], "normalized-exp pullback Map operation"
                ),
                score_map_vjp_operation=_nonnegative_integer(
                    record["score_map_vjp_operation"], "normalized-exp score-Map VJP operation"
                ),
                terminal_contract_operations=_integer_tuple(
                    record["terminal_contract_operations"], "normalized-exp terminal Contract operations"
                ),
            )
        )
    if tuple(pullback.ordinal for pullback in decoded) != tuple(range(len(decoded))):
        raise ValueError("normalized-exp Contract pullback ordinals must be dense and source ordered")
    if any(
        pullback.forward_region_ordinal >= len(_decode_normalized_exp_contract_regions(materialization))
        for pullback in decoded
    ):
        raise ValueError("normalized-exp Contract pullback references an absent forward region")
    if any(
        len(pullback.terminal_contract_operations) != 3 or len(pullback.terminal_cotangent_buffers) != 3
        for pullback in decoded
    ):
        raise ValueError("normalized-exp Contract pullback requires three terminal Contract/buffer bindings")
    return tuple(decoded)


_EVENT_COMMON_ATTRIBUTES = frozenset(
    {
        "event",
        "producer_task",
        "consumer_task",
        "relation_domain_shape",
        "producer_indices",
        "consumer_indices",
        "storage_capacity",
        "storage_slot",
        "generation_index",
        "generations",
        "notify_count",
        "scope",
    }
)
_EVENT_PHASE_ATTRIBUTES = frozenset({"producer_phase", "consumer_phase"})
_EVENT_ERASURE_KINDS = frozenset(
    {
        "program_order_erased",
        "stream_order_erased",
        "algebraic_refinement_erased",
    }
)
_IMPERATIVE_EVENT_KIND = "imperative_wait_notify"


def _decode_exact_event_requirements(
    event_lowering_plan: ShuttleOperationRecord,
) -> tuple[tuple[ShuttleExactEventRequirement, ...], tuple[ShuttleEventRealizationRecord, ...]]:
    realizations: dict[int, str] = {}
    notifications: dict[int, ShuttleOperationRecord] = {}
    waits: dict[int, ShuttleOperationRecord] = {}
    for operation in event_lowering_plan.body().operations:
        if operation.name == "shuttle.event_lowering_plan_yield":
            continue
        if operation.name == "shuttle.event_realization":
            attributes = operation.attributes
            if set(attributes) != {"event", "kind", "covering_launch_order"}:
                raise ValueError("Event realization has unknown or missing fields")
            ordinal = _nonnegative_integer(attributes["event"], "Event realization ordinal")
            kind = _event_realization_kind(attributes["kind"])
            witnesses = _integer_tuple(attributes["covering_launch_order"], "Event realization witnesses")
            if kind == _IMPERATIVE_EVENT_KIND:
                if witnesses:
                    raise ValueError("imperative Event realization cannot carry erasure witnesses")
            elif kind in _EVENT_ERASURE_KINDS:
                if not witnesses or len(witnesses) % 4:
                    raise ValueError("erased Event realization requires launch/phase witnesses")
            else:
                raise ValueError(f"unsupported Event realization kind: {kind!r}")
            if ordinal in realizations:
                raise ValueError(f"duplicate Event realization ordinal {ordinal}")
            realizations[ordinal] = kind
            continue
        if operation.name == "shuttle.imperative_event_notify":
            ordinal = _event_operation_ordinal(operation, action_attribute="release")
            if ordinal in notifications:
                raise ValueError(f"duplicate imperative Event notify for ordinal {ordinal}")
            notifications[ordinal] = operation
            continue
        if operation.name == "shuttle.imperative_event_wait":
            ordinal = _event_operation_ordinal(operation, action_attribute="acquire")
            if ordinal in waits:
                raise ValueError(f"duplicate imperative Event wait for ordinal {ordinal}")
            waits[ordinal] = operation
            continue
        raise ValueError(f"unsupported Event lowering operation: {operation.name}")

    operation_events = notifications.keys() | waits.keys()
    if any(realizations.get(ordinal) != _IMPERATIVE_EVENT_KIND for ordinal in operation_events):
        raise ValueError("imperative Event operation does not match an imperative realization")
    if tuple(sorted(realizations)) != tuple(range(len(realizations))):
        raise ValueError("Event realization ordinals must be contiguous from zero")
    requirements: list[ShuttleExactEventRequirement] = []
    for ordinal, kind in sorted(realizations.items()):
        notification = notifications.get(ordinal)
        wait = waits.get(ordinal)
        if kind != _IMPERATIVE_EVENT_KIND:
            if notification is not None or wait is not None:
                raise ValueError("erased Event realization cannot retain imperative operations")
            continue
        if notification is None or wait is None:
            raise ValueError("imperative Event realization requires exactly one notify and wait")
        requirements.append(_decode_exact_event_requirement(ordinal, notification, wait))
    return (
        tuple(requirements),
        tuple(
            ShuttleEventRealizationRecord(
                ordinal,
                kind,
                _integer_tuple(
                    next(
                        operation.attributes["covering_launch_order"]
                        for operation in event_lowering_plan.body().operations
                        if operation.name == "shuttle.event_realization" and operation.attributes["event"] == ordinal
                    ),
                    "Event realization witnesses",
                ),
            )
            for ordinal, kind in sorted(realizations.items())
        ),
    )


def _decode_schedule_event_relations(
    schedule: ShuttleOperationRecord,
) -> tuple[ShuttleExactEventRequirement, ...]:
    relations: list[ShuttleExactEventRequirement] = []
    expected = (_EVENT_COMMON_ATTRIBUTES - {"event"}) | {"ordinal", "release", "acquire"}
    for operation in schedule.body().operations:
        if operation.name != "shuttle.event_tensor":
            continue
        attributes = operation.attributes
        admitted = (
            expected,
            expected | {"relation_predicate"},
            expected | _EVENT_PHASE_ATTRIBUTES,
            expected | {"relation_predicate"} | _EVENT_PHASE_ATTRIBUTES,
        )
        if set(attributes) not in admitted:
            raise ValueError("schedule EventTensor has unknown or missing exact Event fields")
        if type(attributes["release"]) is not bool or type(attributes["acquire"]) is not bool:
            raise ValueError("schedule EventTensor release/acquire fields must be booleans")
        ordinal = _nonnegative_integer(attributes["ordinal"], "schedule EventTensor ordinal")
        relations.append(
            _decode_exact_event_attributes(
                ordinal,
                attributes,
                release=attributes["release"],
                acquire=attributes["acquire"],
            )
        )
    relations.sort(key=lambda relation: relation.ordinal)
    if tuple(relation.ordinal for relation in relations) != tuple(range(len(relations))):
        raise ValueError("schedule EventTensor ordinals must be contiguous from zero")
    return tuple(relations)


def _event_operation_ordinal(operation: ShuttleOperationRecord, *, action_attribute: str) -> int:
    expected = _EVENT_COMMON_ATTRIBUTES | {action_attribute}
    attributes = operation.attributes
    admitted = (
        expected,
        expected | {"relation_predicate"},
        expected | _EVENT_PHASE_ATTRIBUTES,
        expected | {"relation_predicate"} | _EVENT_PHASE_ATTRIBUTES,
    )
    if set(attributes) not in admitted:
        raise ValueError(f"{operation.name} has unknown or missing exact Event fields")
    if type(attributes[action_attribute]) is not bool:
        raise ValueError(f"{operation.name} {action_attribute} must be a boolean")
    return _nonnegative_integer(attributes["event"], f"{operation.name} Event ordinal")


def _decode_exact_event_requirement(
    ordinal: int,
    notification: ShuttleOperationRecord,
    wait: ShuttleOperationRecord,
) -> ShuttleExactEventRequirement:
    notification_common = {key: value for key, value in notification.attributes.items() if key != "release"}
    wait_common = {key: value for key, value in wait.attributes.items() if key != "acquire"}
    if notification_common != wait_common:
        raise ValueError("imperative Event notify and wait metadata must match exactly")
    return _decode_exact_event_attributes(
        ordinal,
        notification_common,
        release=notification.attributes["release"],
        acquire=wait.attributes["acquire"],
    )


def _decode_exact_event_attributes(
    ordinal: int,
    attributes: Mapping[str, Any],
    *,
    release: bool,
    acquire: bool,
) -> ShuttleExactEventRequirement:
    shape = _integer_tuple(attributes["relation_domain_shape"], "Event relation domain shape")
    if not shape or any(extent <= 0 for extent in shape):
        raise ValueError("Event relation domain must have positive static extents")
    dimensions = len(shape)
    predicate = attributes.get("relation_predicate")
    relation_predicate = None if predicate is None else _integer_set(predicate, dimensions)
    producer_indices = _affine_map(attributes["producer_indices"], dimensions, "Event producer indices")
    consumer_indices = _affine_map(attributes["consumer_indices"], dimensions, "Event consumer indices")
    storage_slot = _affine_map(attributes["storage_slot"], dimensions, "Event storage slot")
    generation_index = _affine_map(attributes["generation_index"], dimensions, "Event generation index")
    if len(storage_slot.results) != 1 or len(generation_index.results) != 1:
        raise ValueError("Event storage slot and generation index must be scalar affine maps")
    storage_capacity = _positive_integer(attributes["storage_capacity"], "Event storage capacity")
    generations = _positive_integer(attributes["generations"], "Event generation count")
    notify_count = _positive_integer(attributes["notify_count"], "Event notify count")
    scope = _event_scope(attributes["scope"])
    producer_phase = attributes.get("producer_phase")
    consumer_phase = attributes.get("consumer_phase")
    if (producer_phase is None) != (consumer_phase is None):
        raise ValueError("Event producer/consumer phases must be present together")
    if producer_phase is not None and (
        not isinstance(producer_phase, str)
        or not producer_phase
        or not isinstance(consumer_phase, str)
        or not consumer_phase
        or producer_phase == consumer_phase
    ):
        raise ValueError("Event phases must be distinct non-empty strings")
    return ShuttleExactEventRequirement(
        ordinal=ordinal,
        producer_task=_nonnegative_integer(attributes["producer_task"], "Event producer task"),
        consumer_task=_nonnegative_integer(attributes["consumer_task"], "Event consumer task"),
        producer_phase=producer_phase,
        consumer_phase=consumer_phase,
        relation_domain_shape=shape,
        relation_predicate=relation_predicate,
        producer_indices=producer_indices,
        consumer_indices=consumer_indices,
        storage_capacity=storage_capacity,
        storage_slot=storage_slot,
        generation_index=generation_index,
        generations=generations,
        notify_count=notify_count,
        scope=scope,
        release=release,
        acquire=acquire,
    )


def _affine_map(encoded: object, dimensions: int, label: str) -> CuteAffineMap:
    record = _object(encoded, label)
    if (
        set(record) != {"kind", "dimensions", "symbols", "results"}
        or record["kind"] != "affine_map"
        or record["dimensions"] != dimensions
        or record["symbols"] != 0
    ):
        raise ValueError(f"{label} must be a symbol-free affine map over the Event relation")
    results = _sequence(record["results"], f"{label} results")
    return CuteAffineMap(dimensions, tuple(affine_expression_from_record(result) for result in results))


def _integer_set(encoded: object, dimensions: int) -> CuteIntegerSet:
    record = _object(encoded, "Event relation predicate")
    if (
        set(record) != {"kind", "dimensions", "symbols", "constraints", "equalities"}
        or record["kind"] != "integer_set"
        or record["dimensions"] != dimensions
        or record["symbols"] != 0
    ):
        raise ValueError("Event relation predicate must be a symbol-free integer set over the Event relation")
    constraints = _sequence(record["constraints"], "Event relation constraints")
    equalities = _sequence(record["equalities"], "Event relation equality flags")
    if any(type(equality) is not bool for equality in equalities):
        raise ValueError("Event relation equality flags must be booleans")
    return CuteIntegerSet(
        dimensions,
        tuple(affine_expression_from_record(constraint) for constraint in constraints),
        tuple(equalities),
    )


def _event_realization_kind(encoded: object) -> str:
    value = _mlir_attribute(encoded, "Event realization kind")
    prefix = "#shuttle.event_realization_kind<"
    if not value.startswith(prefix) or not value.endswith(">"):
        raise ValueError("Event realization kind must be a typed Shuttle attribute")
    return value[len(prefix) : -1]


def _event_scope(encoded: object) -> str:
    value = _mlir_attribute(encoded, "Event scope")
    prefix = "#shuttle.event_scope<"
    if not value.startswith(prefix) or not value.endswith(">"):
        raise ValueError("Event scope must be a typed Shuttle attribute")
    scope = value[len(prefix) : -1]
    if scope not in {"thread", "workgroup", "device", "system"}:
        raise ValueError(f"unsupported Event scope: {scope!r}")
    return scope


def _mlir_attribute(encoded: object, label: str) -> str:
    record = _object(encoded, label)
    if set(record) != {"kind", "value"} or record["kind"] != "mlir_attribute" or not isinstance(record["value"], str):
        raise ValueError(f"{label} must be a typed MLIR attribute")
    return record["value"]


def _nonnegative_integer(encoded: object, label: str) -> int:
    if type(encoded) is not int or encoded < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return encoded


def _positive_integer(encoded: object, label: str) -> int:
    if type(encoded) is not int or encoded <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return encoded


def native_gpu_physical_artifact_provider(
    compiler: GpuPhysicalArtifactCompiler,
) -> Callable[[str], bytes | tuple[bytes, tuple[dict[str, Any], ...]] | None]:
    """Adapt a typed compiler to Shuttle's serialized jaxlib callback."""

    def compile_request(serialized: str) -> bytes | tuple[bytes, tuple[dict[str, Any], ...]] | None:
        request = decode_gpu_physical_artifact_request(serialized)
        result = compiler(request)
        if result is None or isinstance(result, bytes):
            return result
        if not isinstance(result, CutePipelineArtifact):
            raise TypeError("physical artifact compiler must return CutePipelineArtifact, bytes, or None")
        legalizations = tuple(
            record
            for region in result.internal_event_legalizations
            for record in internal_event_device_legalization_region_record(region)
        )
        if not legalizations:
            return result.native_transport
        return result.native_transport, legalizations

    return compile_request


def _decode_operation(encoded: object) -> ShuttleOperationRecord:
    record = _object(encoded, "operation")
    if set(record) != {"name", "attributes", "operands", "results", "regions"}:
        raise ValueError("structured operation has unknown or missing fields")
    name = record["name"]
    attributes = record["attributes"]
    if not isinstance(name, str) or not isinstance(attributes, dict):
        raise ValueError("structured operation name and attributes have invalid types")
    operands = _integer_tuple(record["operands"], "operation operands")
    results = tuple(_decode_value(value) for value in _sequence(record["results"], "operation results"))
    regions = tuple(
        tuple(_decode_block(block) for block in _sequence(region, "operation region"))
        for region in _sequence(record["regions"], "operation regions")
    )
    return ShuttleOperationRecord(
        name=name,
        attributes=attributes,
        operands=operands,
        results=results,
        regions=regions,
    )


def _decode_block(encoded: object) -> ShuttleBlockRecord:
    record = _object(encoded, "block")
    if set(record) != {"arguments", "operations"}:
        raise ValueError("structured block has unknown or missing fields")
    return ShuttleBlockRecord(
        arguments=tuple(_decode_value(value) for value in _sequence(record["arguments"], "block arguments")),
        operations=tuple(
            _decode_operation(operation) for operation in _sequence(record["operations"], "block operations")
        ),
    )


def _decode_value(encoded: object) -> ShuttleValueRecord:
    record = _object(encoded, "value")
    if set(record) != {"id", "type"} or type(record["id"]) is not int:
        raise ValueError("structured value has invalid fields")
    return ShuttleValueRecord(id=record["id"], type=_decode_type(record["type"]))


def _decode_type(encoded: object) -> ShuttleTypeRecord:
    record = _object(encoded, "type")
    kind = record.get("kind")
    element_type = record.get("element_type")
    if kind not in ("scalar", "ranked_tensor") or not isinstance(element_type, str):
        raise ValueError("structured type has invalid kind or element type")
    expected_fields = {"kind", "element_type"} if kind == "scalar" else {"kind", "element_type", "shape"}
    if set(record) != expected_fields:
        raise ValueError("structured type has unknown or missing fields")
    shape = () if kind == "scalar" else _integer_tuple(record["shape"], "tensor shape")
    if kind == "ranked_tensor" and any(extent <= 0 for extent in shape):
        raise ValueError("physical artifact requests require static positive tensor shapes")
    return ShuttleTypeRecord(kind=kind, element_type=element_type, shape=shape)


def _object(encoded: object, label: str) -> dict[str, Any]:
    if not isinstance(encoded, dict):
        raise ValueError(f"structured {label} must be an object")
    return encoded


def _sequence(encoded: object, label: str) -> Sequence[object]:
    if not isinstance(encoded, list):
        raise ValueError(f"structured {label} must be an array")
    return encoded


def _integer_tuple(encoded: object, label: str) -> tuple[int, ...]:
    values = _sequence(encoded, label)
    if any(type(value) is not int for value in values):
        raise ValueError(f"structured {label} must contain only integers")
    return tuple(int(value) for value in values)


def _optional_integer_tuple(encoded: object, label: str) -> tuple[int | None, ...]:
    values = _sequence(encoded, label)
    if any(value is not None and type(value) is not int for value in values):
        raise ValueError(f"structured {label} must contain only integers or nulls")
    return tuple(None if value is None else int(value) for value in values)
