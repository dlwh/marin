# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Natural JAX value-and-gradient lowering for a normalized-exp Contract region."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp

from shuttle.ir import DType
from tile_lifetime.canonical_kernel_composition import (
    CanonicalKernelComposition,
    coalesce_dependency_convex_elementwise_components,
    lower_canonical_kernel_composition,
)
from tile_lifetime.executable_normalized_exp_contract import (
    ExecutableNormalizedExpContractPlan,
    prepare_sm90_normalized_exp_contract,
)
from tile_lifetime.index_domain_canonicalization import normalized_exp_task_algebra_from_record
from tile_lifetime.jax_normalized_exp_contract_pullback import (
    ExecutableNormalizedExpPullbackPlan,
    JaxOwnedPullbackTrace,
    prepare_sm90_jax_owned_normalized_exp_pullback,
    trace_jax_owned_pullback,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.normalized_exp_contract_region import (
    normalized_exp_contract_pullback_tensor_program,
    normalized_exp_contract_region,
)
from tile_lifetime.normalized_exp_regional_pullback import (
    NormalizedExpRegionalPullbackReplacement,
    replace_normalized_exp_regional_pullback,
)
from tile_lifetime.physical_task_contract import (
    ExecutableEventLowering,
    ExecutablePhysicalRequest,
    PhysicalNumericalPolicy,
    PhysicalTarget,
    canonical_fingerprint,
    lower_executable_event_operations,
)
from tile_lifetime.streaming_attention_executable import (
    attach_sm90_contract_fold_schedule,
    jax_tensor_program_executable_request,
)
from tile_lifetime.tensor_program import (
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    NumericalPolicy,
    OperandIndexing,
    ProgramValue,
    RelationPrimitive,
    ScalarConversionSemantics,
    ScalarExpressionKind,
    TensorProgram,
    deserialize_tensor_program,
    index_axis,
    scalar_binary,
    scalar_convert,
    scalar_input,
    tensor_program_value_ordinals,
)
from tile_lifetime.tensor_program_rewrites import (
    FoldInvariantDivisionBoundary,
    fold_invariant_division_boundaries,
)


@dataclass(frozen=True)
class JaxOwnedNormalizedExpValueAndGradTrace:
    """A scalar loss differentiated by JAX and its equivalent composed VJP."""

    forward: TensorProgram
    value_and_grad: TensorProgram
    composed_vjp: JaxOwnedPullbackTrace

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "format": "shuttle_jax_owned_normalized_exp_value_and_grad_trace_v1",
            "differentiation_owner": "jax",
            "objective": "fp32_inner_product_of_forward_result_and_explicit_cotangent",
            "forward_fingerprint": self.forward.semantic_fingerprint,
            "value_and_grad_fingerprint": self.value_and_grad.semantic_fingerprint,
            "composed_vjp_fingerprint": self.composed_vjp.fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.canonical_record, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class ExecutableNormalizedExpValueAndGradPlan:
    """Exact source replacement and canonical physical component composition."""

    source_request: ExecutablePhysicalRequest
    trace: JaxOwnedNormalizedExpValueAndGradTrace
    pullback_plan: ExecutableNormalizedExpPullbackPlan
    physical_program: TensorProgram
    physical_request: ExecutablePhysicalRequest
    preselection_event_lowerings: tuple[ExecutableEventLowering, ...]
    composition: CanonicalKernelComposition

    @property
    def crossing_event_names(self) -> tuple[str, ...]:
        return tuple(
            self.physical_request.event_plans[event.event_id].name for event in self.composition.concrete_events
        )

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "format": "shuttle_normalized_exp_value_and_grad_plan_v1",
            "source_request_fingerprint": self.source_request.fingerprint,
            "trace_fingerprint": self.trace.fingerprint,
            "replacement_fingerprint": self.pullback_plan.replacement.fingerprint,
            "physical_algebra_fingerprint": self.physical_program.semantic_fingerprint,
            "physical_request_fingerprint": self.physical_request.fingerprint,
            "component_composition_fingerprint": self.composition.fingerprint,
            "crossing_event_names": self.crossing_event_names,
            "preselection_event_lowerings": tuple(
                lowering.canonical_record for lowering in self.preselection_event_lowerings
            ),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_record)

    def require_whole_request_runtime(self) -> None:
        self.pullback_plan.require_whole_request_runtime()
        source_outputs = self.trace.value_and_grad.outputs
        physical_outputs = self.physical_program.outputs
        if tuple((value.shape, value.dtype) for value in physical_outputs) != tuple(
            (value.shape, value.dtype) for value in source_outputs
        ):
            raise ValueError("physical value-and-gradient results do not match the JAX-owned public boundary")
        if not self.crossing_event_names:
            raise ValueError("value-and-gradient composition must retain a crossing Event for the scalar loss")
        if any(not lowering.retained_operations for lowering in self.preselection_event_lowerings):
            raise ValueError("physical value-and-gradient Events must remain imperative before schedule selection")
        if any(lowering.retained_operations for lowering in self.composition.legalized_event_lowerings):
            raise ValueError("canonical component composition did not discharge every physical Event")


@dataclass(frozen=True)
class ExecutableSplitNormalizedExpValueAndGradPlan:
    """A JAX-owned program with disjoint forward and saved-state pullback schedules."""

    source_request: ExecutablePhysicalRequest
    trace: JaxOwnedNormalizedExpValueAndGradTrace
    forward_plan: ExecutableNormalizedExpContractPlan
    forward_conversion_boundary: FoldInvariantDivisionBoundary
    replacement: NormalizedExpRegionalPullbackReplacement
    program: TensorProgram
    saved_values: tuple[ProgramValue, ...]
    physical_request: ExecutablePhysicalRequest
    preselection_event_lowerings: tuple[ExecutableEventLowering, ...]
    composition: CanonicalKernelComposition

    @property
    def directional_schedules(self) -> tuple[object, object]:
        schedules = tuple(
            schedule
            for schedule in self.physical_request.schedules
            if schedule.parameters.get("direction") in {"forward", "pullback"}
        )
        if len(schedules) != 2:
            raise ValueError("split normalized-exp plan requires one forward and one pullback schedule")
        return schedules

    @property
    def crossing_event_names(self) -> tuple[str, ...]:
        return tuple(
            self.physical_request.event_plans[event.event_id].name for event in self.composition.concrete_events
        )

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "format": "shuttle_split_normalized_exp_value_and_grad_plan_v1",
                "source_request_fingerprint": self.source_request.fingerprint,
                "trace_fingerprint": self.trace.fingerprint,
                "replacement_program_fingerprint": self.replacement.program.semantic_fingerprint,
                "forward_conversion_boundary": self.forward_conversion_boundary.canonical_record,
                "program_fingerprint": self.program.semantic_fingerprint,
                "saved_value_ids": tuple(value.name for value in self.saved_values),
                "physical_request_fingerprint": self.physical_request.fingerprint,
                "composition_fingerprint": self.composition.fingerprint,
            }
        )

    def require_whole_request_runtime(self) -> None:
        forward, pullback = self.directional_schedules
        if set(forward.task_ids) & set(pullback.task_ids):
            raise ValueError("split normalized-exp schedules overlap")
        if not self.crossing_event_names:
            raise ValueError("split normalized-exp schedules require exact crossing Events")
        if any(not lowering.retained_operations for lowering in self.preselection_event_lowerings):
            raise ValueError("split normalized-exp Events must remain imperative before schedule selection")
        if any(lowering.retained_operations for lowering in self.composition.legalized_event_lowerings):
            raise ValueError("split normalized-exp composition did not legalize every crossing Event")


def trace_jax_owned_normalized_exp_value_and_grad(
    function: Callable[..., Any],
    *primals: Any,
    output_cotangent: Any,
    numerical_policy: NumericalPolicy = NumericalPolicy.SOURCE_ORDERED,
) -> JaxOwnedNormalizedExpValueAndGradTrace:
    """Trace an ordinary scalar loss through ``jax.value_and_grad``.

    The explicit output cotangent is a normal tensor argument to the loss.  It
    makes the scalar objective equivalent to the composed VJP without a custom
    differentiation rule or compiler marker.
    """

    primal_count = len(primals)
    if primal_count == 0:
        raise ValueError("normalized-exp value_and_grad requires at least one primal")

    def objective(*arguments: Any) -> Any:
        primal_arguments = arguments[:primal_count]
        cotangent = arguments[primal_count]
        output = function(*primal_arguments)
        return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))

    differentiated = jax.value_and_grad(objective, argnums=tuple(range(primal_count)))

    def flattened_value_and_grad(*arguments: Any) -> Any:
        value, gradients = differentiated(*arguments)
        return value, *gradients

    forward = jax_function_to_tensor_program(function, *primals, numerical_policy=numerical_policy)
    value_and_grad = jax_function_to_tensor_program(
        flattened_value_and_grad,
        *primals,
        output_cotangent,
        numerical_policy=numerical_policy,
    )
    composed_vjp = trace_jax_owned_pullback(
        function,
        *primals,
        output_cotangent=output_cotangent,
        numerical_policy=numerical_policy,
    )
    if forward.semantic_fingerprint != composed_vjp.forward.semantic_fingerprint:
        raise ValueError("value_and_grad and composed VJP did not trace the same forward program")
    if len(value_and_grad.outputs) != primal_count + 1 or value_and_grad.outputs[0].shape:
        raise ValueError("JAX value_and_grad must return one scalar loss followed by every primal gradient")
    for primal, gradient in zip(primals, value_and_grad.outputs[1:], strict=True):
        if tuple(primal.shape) != gradient.shape:
            raise ValueError("JAX value_and_grad result shape must equal its primal shape")
    return JaxOwnedNormalizedExpValueAndGradTrace(forward, value_and_grad, composed_vjp)


def normalized_exp_value_and_grad_tensor_program(
    pullback_plan: ExecutableNormalizedExpPullbackPlan,
) -> TensorProgram:
    """Add the scalar objective as a residual Map/Fold physical component."""
    base = normalized_exp_contract_pullback_tensor_program(pullback_plan.program.pullback)
    region = pullback_plan.program.forward.region
    output = region.finalizer.output
    output_cotangent = pullback_plan.program.pullback.output_cotangent
    product = ProgramValue("normalized_exp_objective.product", output.axes, DType.FP32)
    product_map = MapPrimitive(
        "normalized-exp objective product",
        (output, output_cotangent),
        product,
        scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_convert(
                scalar_input(output.name),
                result_dtype=DType.FP32,
                semantics=ScalarConversionSemantics.EXACT,
            ),
            scalar_convert(
                scalar_input(output_cotangent.name),
                result_dtype=DType.FP32,
                semantics=ScalarConversionSemantics.EXACT,
            ),
        ),
    )
    loss = ProgramValue("normalized_exp_objective.loss", (), DType.FP32)
    loss_fold = FoldPrimitive(
        "normalized-exp scalar objective Fold",
        product,
        loss,
        output.axes,
        FoldReducer.SUM,
        DType.FP32,
    )
    dimensions = region.dimensions
    query_gradient, key_gradient, value_gradient = base.outputs[:3]
    common_gradient_axes = set(query_gradient.axes) & set(key_gradient.axes) & set(value_gradient.axes)
    batch_axes = tuple(
        axis
        for axis in query_gradient.axes
        if axis in common_gradient_axes and axis not in dimensions.shared_relation_axes
    )
    shared_axis = dimensions.shared_relation_axes[0]
    owner_axis = next(
        axis for axis in region.owner_axes if axis not in (*batch_axes, shared_axis, *dimensions.relation_group_axes)
    )
    query_axes = (
        *batch_axes,
        owner_axis,
        shared_axis,
        *dimensions.relation_group_axes,
        *dimensions.producer_features,
    )
    key_axes = (*batch_axes, *dimensions.normalized_axes, shared_axis, *dimensions.producer_features)
    value_axes = (*batch_axes, *dimensions.normalized_axes, shared_axis, *dimensions.consumer_features)
    output_axes = (*batch_axes, owner_axis, shared_axis, *dimensions.relation_group_axes, *dimensions.consumer_features)
    canonical_inputs = (
        region.producer.inputs[0],
        region.producer.inputs[1],
        region.consumer.inputs[1],
        output_cotangent,
    )
    public_input_axes = (query_axes, key_axes, value_axes, output_axes)
    public_inputs: list[ProgramValue] = []
    input_relations: list[RelationPrimitive] = []
    for ordinal, (canonical_input, axes) in enumerate(zip(canonical_inputs, public_input_axes, strict=True)):
        if set(canonical_input.axes) != set(axes):
            raise ValueError("normalized-exp input cannot be expressed by the explicit grouped boundary")
        public_input = ProgramValue(f"normalized_exp_objective.input.{ordinal}", axes, canonical_input.dtype)
        input_relations.append(
            RelationPrimitive(
                f"normalized-exp input {ordinal} layout relation",
                (public_input,),
                (canonical_input,),
                canonical_input.axes,
                (
                    OperandIndexing(
                        public_input,
                        canonical_input.axes,
                        tuple(index_axis(axis) for axis in public_input.axes),
                    ),
                ),
            )
        )
        public_inputs.append(public_input)
    reordered_gradients: list[ProgramValue] = []
    gradient_relations: list[RelationPrimitive] = []
    for ordinal, (gradient, axes) in enumerate(
        zip((query_gradient, key_gradient, value_gradient), (query_axes, key_axes, value_axes), strict=True)
    ):
        if set(gradient.axes) != set(axes):
            raise ValueError("terminal gradient axes cannot be expressed by the explicit grouped boundary")
        reordered = ProgramValue(f"normalized_exp_objective.gradient.{ordinal}", axes, gradient.dtype)
        gradient_relations.append(
            RelationPrimitive(
                f"normalized-exp terminal gradient {ordinal} layout relation",
                (gradient,),
                (reordered,),
                axes,
                (
                    OperandIndexing(
                        gradient,
                        axes,
                        tuple(index_axis(axis) for axis in gradient.axes),
                    ),
                ),
            )
        )
        reordered_gradients.append(reordered)
    return replace(
        base,
        inputs=tuple(public_inputs),
        operations=(*input_relations, *base.operations, product_map, loss_fold, *gradient_relations),
        outputs=(loss, *reordered_gradients),
        numerical_policy=NumericalPolicy.FAST,
    )


def prepare_sm90_jax_owned_normalized_exp_value_and_grad(
    request: ExecutablePhysicalRequest,
    trace: JaxOwnedNormalizedExpValueAndGradTrace,
    *,
    tile_size: int,
) -> ExecutableNormalizedExpValueAndGradPlan:
    """Replace natural JAX AD with one generic pullback plus scalar-loss component."""
    pullback_plan = prepare_sm90_jax_owned_normalized_exp_pullback(
        request,
        trace.composed_vjp,
        tile_size=tile_size,
        source_program=trace.value_and_grad,
        replacement_identity="jax_value_and_grad_of_fp32_output_inner_product",
    )
    candidate = normalized_exp_value_and_grad_tensor_program(pullback_plan)
    unscheduled = jax_tensor_program_executable_request(
        candidate,
        tile_size=tile_size,
        target=PhysicalTarget.SM90,
        normalized_exp_pullback=pullback_plan.program.pullback,
    )
    preselection_event_lowerings = lower_executable_event_operations(unscheduled)
    if any(not lowering.retained_operations for lowering in preselection_event_lowerings):
        raise ValueError("value-and-gradient Events must remain imperative before physical schedule selection")
    attached = attach_sm90_contract_fold_schedule(unscheduled)
    if attached.physical_schedule.parameters.get("direction") != "pullback":
        raise ValueError("value-and-gradient algebra did not select the generic SM90 pullback schedule")
    if attached.request.algebra_record.get("format") == "shuttle_domain_restricted_tensor_program_v1":
        physical_program = normalized_exp_task_algebra_from_record(attached.request.algebra_record).task_program
    else:
        physical_program = deserialize_tensor_program(
            json.dumps(attached.request.algebra_record, sort_keys=True, separators=(",", ":"))
        )
    composition = lower_canonical_kernel_composition(attached.request)
    plan = ExecutableNormalizedExpValueAndGradPlan(
        request,
        trace,
        pullback_plan,
        physical_program,
        attached.request,
        preselection_event_lowerings,
        composition,
    )
    plan.require_whole_request_runtime()
    return plan


def prepare_sm90_split_normalized_exp_value_and_grad(
    request: ExecutablePhysicalRequest,
    trace: JaxOwnedNormalizedExpValueAndGradTrace,
    *,
    tile_size: int,
) -> ExecutableSplitNormalizedExpValueAndGradPlan:
    """Install separate forward and recomputing pullback schedules in a JAX program."""
    if request.target is not PhysicalTarget.SM90 or request.numerical_policy is not PhysicalNumericalPolicy.FAST:
        raise ValueError("split normalized-exp value-and-gradient lowering requires FAST SM90")
    if request.provenance.get("source_algebra_fingerprint") != trace.value_and_grad.semantic_fingerprint:
        raise ValueError("split normalized-exp request is not derived from the supplied JAX-owned program")
    if len(request.normalized_exp_contract_regions) != 1:
        raise ValueError("split normalized-exp lowering requires one planner-declared forward region")
    conversion_boundaries = fold_invariant_division_boundaries(trace.forward)
    if len(conversion_boundaries) != 1:
        raise ValueError(
            "conversion-faithful split lowering requires one typed Fold-invariant division boundary, "
            f"found {len(conversion_boundaries)}"
        )
    conversion_boundary = conversion_boundaries[0]
    if (
        conversion_boundary.numerator.dtype is not DType.BF16
        or conversion_boundary.denominator.dtype is not DType.BF16
        or conversion_boundary.fold_state.dtype is not DType.FP32
    ):
        raise ValueError(
            "conversion-faithful split lowering currently requires BF16 normalized weights with FP32 Fold state"
        )
    forward_request = jax_tensor_program_executable_request(
        trace.forward,
        tile_size=tile_size,
        target=request.target,
    )
    forward_plan = prepare_sm90_normalized_exp_contract(attach_sm90_contract_fold_schedule(forward_request).request)
    source_record: object = request.algebra_record
    if request.algebra_record.get("format") == "shuttle_domain_restricted_tensor_program_v1":
        source_record = request.algebra_record.get("program")
    if not isinstance(source_record, dict) or source_record.get("format") != "shuttle_tensor_program_v1":
        raise ValueError("split normalized-exp lowering requires a canonical TensorProgram")
    source = deserialize_tensor_program(json.dumps(source_record, sort_keys=True, separators=(",", ":")))
    capability = request.normalized_exp_contract_regions[0]
    values_by_ordinal = {ordinal: value for value, ordinal in tensor_program_value_ordinals(source).items()}
    try:
        finalizer_value = values_by_ordinal[capability.finalizer_value]
    except KeyError as error:
        raise ValueError("split normalized-exp finalizer value ordinal is stale") from error
    region = normalized_exp_contract_region(source, output=finalizer_value)
    replacement = replace_normalized_exp_regional_pullback(source, region, capability)
    unscheduled = jax_tensor_program_executable_request(
        replacement.program,
        tile_size=tile_size,
        target=PhysicalTarget.SM90,
        normalized_exp_pullback=replacement.pullback,
    )
    executable_record: object = unscheduled.algebra_record
    if unscheduled.algebra_record.get("format") == "shuttle_domain_restricted_tensor_program_v1":
        executable_record = unscheduled.algebra_record.get("program")
    if not isinstance(executable_record, dict):
        raise ValueError("split normalized-exp lowering did not retain a canonical executable program")
    executable_program = deserialize_tensor_program(json.dumps(executable_record, sort_keys=True, separators=(",", ":")))
    preselection_event_lowerings = lower_executable_event_operations(unscheduled)
    if any(not lowering.retained_operations for lowering in preselection_event_lowerings):
        raise ValueError("split normalized-exp Events must remain imperative before schedule selection")
    attached = attach_sm90_contract_fold_schedule(unscheduled)
    forward_schedule = next(
        schedule for schedule in attached.request.schedules if schedule.parameters.get("direction") == "forward"
    )
    conversion_faithful_schedule = replace(
        forward_schedule,
        parameters={
            **forward_schedule.parameters,
            "execution_graph": "contract_fold_then_recomputed_map_contract",
            "pass_count": 2,
            "producer_contract_traversals": 3,
            "statistics_traversals": ("maximum", "centered_exponential_sum"),
            "weighted_contract_traversals": ("normalized_weighted_contract",),
            "conversion_boundary": conversion_boundary.canonical_record,
            "selection": "typed_conversion_boundaries_and_task_resource_estimates",
        },
    )
    scheduled_request = replace(
        attached.request,
        schedules=tuple(
            conversion_faithful_schedule if schedule is forward_schedule else schedule
            for schedule in attached.request.schedules
        ),
    )
    composition = lower_canonical_kernel_composition(scheduled_request)
    pullback_schedule = next(
        schedule for schedule in scheduled_request.schedules if schedule.parameters.get("direction") == "pullback"
    )
    pullback_component = next(
        component
        for component in composition.components
        if component.schedule_index == scheduled_request.schedules.index(pullback_schedule)
    )
    values_by_ordinal = {ordinal: value for value, ordinal in tensor_program_value_ordinals(executable_program).items()}
    pullback_inputs = tuple(values_by_ordinal[ordinal] for ordinal in pullback_component.live_input_value_ids)
    canonical_saved_values = tuple(
        _one_value_with_signature(pullback_inputs, saved_value) for saved_value in replacement.saved_values
    )
    composition = coalesce_dependency_convex_elementwise_components(composition)
    scheduled_request = composition.scheduled_request
    plan = ExecutableSplitNormalizedExpValueAndGradPlan(
        request,
        trace,
        forward_plan,
        conversion_boundary,
        replacement,
        executable_program,
        canonical_saved_values,
        scheduled_request,
        preselection_event_lowerings,
        composition,
    )
    plan.require_whole_request_runtime()
    return plan


def _one_value_with_signature(values: tuple[ProgramValue, ...], reference: ProgramValue) -> ProgramValue:
    signature = (tuple((axis.id, axis.extent) for axis in reference.axes), reference.dtype)
    matches = tuple(
        value for value in values if (tuple((axis.id, axis.extent) for axis in value.axes), value.dtype) == signature
    )
    if len(matches) != 1:
        raise ValueError(
            f"saved-state boundary for {reference.name!r} requires one canonical value, found {len(matches)}"
        )
    return matches[0]
