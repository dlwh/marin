# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower a JAX-owned pullback to the generic normalized-exp SM90 boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import jax

from tile_lifetime.canonical_kernel_composition import CanonicalKernelComposition, lower_canonical_kernel_composition
from tile_lifetime.event_buffering import EventRealizationKind
from tile_lifetime.executable_normalized_exp_contract import (
    ExecutableNormalizedExpContractPlan,
    prepare_sm90_normalized_exp_contract,
)
from tile_lifetime.h100_normalized_exp_backward_lowering import (
    H100NormalizedExpBackwardLowering,
    lower_h100_normalized_exp_backward,
)
from tile_lifetime.h100_streaming_lowering import (
    Sm90BackwardAccumulatorSchedule,
    select_sm90_contract_schedule,
)
from tile_lifetime.internal_event_allocation import (
    InternalEventDeviceLegalizationRegion,
    internal_event_device_legalization_region,
)
from tile_lifetime.jax_tensor_program import jax_function_to_tensor_program
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    NormalizedExpContractPullbackProgram,
    normalized_exp_contract_pullback_program,
    normalized_exp_contract_pullback_tensor_program,
)
from tile_lifetime.physical_task_contract import (
    ExecutableEventLowering,
    ExecutablePhysicalRequest,
    PhysicalNumericalPolicy,
    PhysicalTarget,
    canonical_fingerprint,
    erase_executable_event_operations,
    lower_executable_event_operations,
)
from tile_lifetime.streaming_attention_executable import (
    attach_sm90_contract_fold_schedule,
    jax_tensor_program_executable_request,
)
from tile_lifetime.tensor_program import NumericalPolicy, TensorProgram


@dataclass(frozen=True)
class JaxOwnedPullbackTrace:
    """Forward, reverse, and composed programs produced by one ``jax.vjp``."""

    forward: TensorProgram
    reverse: TensorProgram
    training: TensorProgram

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "format": "shuttle_jax_owned_pullback_trace_v1",
            "differentiation_owner": "jax",
            "forward_fingerprint": self.forward.semantic_fingerprint,
            "reverse_fingerprint": self.reverse.semantic_fingerprint,
            "training_fingerprint": self.training.semantic_fingerprint,
            "primal_count": len(self.forward.inputs),
            "forward_result_count": len(self.forward.outputs),
            "reverse_result_count": len(self.reverse.outputs),
            "training_result_count": len(self.training.outputs),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.canonical_record, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class JaxPullbackAlgebraicReplacement:
    """Exact source task/Event coverage for one JAX-owned AD replacement."""

    source_algebra_fingerprint: str
    trace_fingerprint: str
    covered_task_ids: tuple[int, ...]
    covered_operation_ordinals: tuple[int, ...]
    event_witnesses: tuple[Mapping[str, object], ...]
    identity: str
    numerical_effect: str

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "format": "shuttle_jax_pullback_algebraic_replacement_v1",
            "source_algebra_fingerprint": self.source_algebra_fingerprint,
            "trace_fingerprint": self.trace_fingerprint,
            "covered_task_ids": self.covered_task_ids,
            "covered_operation_ordinals": self.covered_operation_ordinals,
            "event_witnesses": self.event_witnesses,
            "identity": self.identity,
            "legality": (
                "trace reverse and training programs were constructed by jax.vjp",
                "request source fingerprint equals the composed JAX trace",
                "forward request has complete generated-kernel and public-ABI coverage",
                "each source task operation ordinal is covered exactly once",
                "each source Event is discharged by the algebraic replacement",
            ),
            "numerical_effect": self.numerical_effect,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_record)


@dataclass(frozen=True)
class ExecutableNormalizedExpPullbackPlan:
    """JAX provenance, source coverage, and the generic backward emitter input."""

    source_request: ExecutablePhysicalRequest
    trace: JaxOwnedPullbackTrace
    forward_plan: ExecutableNormalizedExpContractPlan
    program: NormalizedExpContractPullbackProgram[object, Sm90BackwardAccumulatorSchedule]
    lowering: H100NormalizedExpBackwardLowering
    replacement: JaxPullbackAlgebraicReplacement
    source_event_lowerings: tuple[ExecutableEventLowering, ...]
    replacement_event_lowerings: tuple[ExecutableEventLowering, ...]
    pullback_preselection_event_lowerings: tuple[ExecutableEventLowering, ...]
    pullback_request: ExecutablePhysicalRequest
    pullback_composition: CanonicalKernelComposition
    backward_internal_event_legalization: InternalEventDeviceLegalizationRegion

    @property
    def whole_request_runtime_supported(self) -> bool:
        source_tasks = tuple(task.id for task in self.source_request.task_plan.tasks)
        source_operations = tuple(
            sorted(ordinal for task in self.source_request.task_plan.tasks for ordinal in task.operation_ordinals)
        )
        source_events = tuple(event.name for event in self.source_request.event_plans)
        witnessed_events = tuple(str(witness["event"]) for witness in self.replacement.event_witnesses)
        pullback_tasks = tuple(task.id for task in self.pullback_request.task_plan.tasks)
        composed_tasks = tuple(
            sorted(task_id for component in self.pullback_composition.components for task_id in component.task_ids)
        )
        return (
            self.replacement.covered_task_ids == source_tasks
            and self.replacement.covered_operation_ordinals == source_operations
            and witnessed_events == source_events
            and all(lowering.retained_operations for lowering in self.source_event_lowerings)
            and all(not lowering.retained_operations for lowering in self.replacement_event_lowerings)
            and all(lowering.retained_operations for lowering in self.pullback_preselection_event_lowerings)
            and pullback_tasks == composed_tasks
        )

    def require_whole_request_runtime(self) -> None:
        if not self.whole_request_runtime_supported:
            raise ValueError("the JAX-owned pullback replacement lacks exact task or Event coverage")


def trace_jax_owned_pullback(
    function: Callable[..., Any],
    *primals: Any,
    output_cotangent: Any,
    numerical_policy: NumericalPolicy = NumericalPolicy.SOURCE_ORDERED,
) -> JaxOwnedPullbackTrace:
    """Ask JAX for a VJP and import its forward, reverse, and composed JAXprs.

    The output cotangent remains an ordinary public tensor argument.  Shuttle
    does not install a custom derivative or a workload marker.
    """

    def reverse(*arguments: Any) -> Any:
        primal_arguments = arguments[:-1]
        cotangent = arguments[-1]
        _, pullback = jax.vjp(function, *primal_arguments)
        return pullback(cotangent)

    def training(*arguments: Any) -> Any:
        primal_arguments = arguments[:-1]
        cotangent = arguments[-1]
        output, pullback = jax.vjp(function, *primal_arguments)
        return output, *pullback(cotangent)

    forward = jax_function_to_tensor_program(function, *primals, numerical_policy=numerical_policy)
    reverse_program = jax_function_to_tensor_program(
        reverse,
        *primals,
        output_cotangent,
        numerical_policy=numerical_policy,
    )
    training_program = jax_function_to_tensor_program(
        training,
        *primals,
        output_cotangent,
        numerical_policy=numerical_policy,
    )
    if len(forward.outputs) != 1:
        raise ValueError("the current normalized-exp pullback trace requires one forward tensor result")
    if len(reverse_program.outputs) != len(primals):
        raise ValueError("JAX pullback result count must equal the primal argument count")
    if len(training_program.outputs) != 1 + len(primals):
        raise ValueError("JAX composed result must contain the forward result followed by every primal cotangent")
    if output_cotangent.shape != forward.outputs[0].shape:
        raise ValueError("output cotangent shape must equal the forward result shape")
    for primal, cotangent in zip(primals, reverse_program.outputs, strict=True):
        if tuple(primal.shape) != cotangent.shape:
            raise ValueError("JAX pullback cotangent shape must equal its primal shape")
    return JaxOwnedPullbackTrace(forward, reverse_program, training_program)


def prepare_sm90_jax_owned_normalized_exp_pullback(
    request: ExecutablePhysicalRequest,
    trace: JaxOwnedPullbackTrace,
    *,
    tile_size: int,
    source_program: TensorProgram | None = None,
    replacement_identity: str = "jax_vjp_of_same_forward_to_normalized_exp_contract_pullback",
) -> ExecutableNormalizedExpPullbackPlan:
    """Replace a certified JAX pullback with the shared generic SM90 algebra.

    Region extraction is performed only on the ordinary forward algebra.  The
    reverse source is accepted because this trace constructed it with
    ``jax.vjp`` from that exact forward callable, not because its decomposed
    equation topology resembles a named workload.
    """
    if request.target is not PhysicalTarget.SM90:
        raise ValueError("the JAX-owned normalized-exp pullback adapter requires SM90")
    if request.numerical_policy is not PhysicalNumericalPolicy.FAST:
        raise ValueError("the JAX-owned normalized-exp pullback adapter requires FAST numerical policy")
    source_program = trace.training if source_program is None else source_program
    if request.provenance.get("source_algebra_fingerprint") != source_program.semantic_fingerprint:
        raise ValueError("executable request is not derived from the supplied JAX-owned differentiated program")

    forward_request = jax_tensor_program_executable_request(
        trace.forward,
        tile_size=tile_size,
        target=request.target,
    )
    forward_attachment = attach_sm90_contract_fold_schedule(forward_request)
    forward_plan = prepare_sm90_normalized_exp_contract(forward_attachment.request)
    relation_layout = forward_plan.program.schedule.contract.relation_layout
    if relation_layout is None:
        raise ValueError("the selected SM90 forward schedule lacks a relation-packed layout")
    reverse_contract = select_sm90_contract_schedule(
        owner_extent=relation_layout.owner_extent,
        streamed_extent=relation_layout.streamed_extent,
        producer_feature_extent=relation_layout.producer_feature_extent,
        consumer_feature_extent=relation_layout.consumer_feature_extent,
        threads=384,
        owner_tile=64,
        streamed_tile=128,
        relation_layout=relation_layout,
    )
    program = normalized_exp_contract_pullback_program(
        forward_plan.program,
        schedule=Sm90BackwardAccumulatorSchedule(contract=reverse_contract),
        traversal=ContractDomainTraversal.FULL,
    )
    lowering = lower_h100_normalized_exp_backward(program)
    task_ids = tuple(task.id for task in request.task_plan.tasks)
    operation_ordinals = tuple(
        sorted(ordinal for task in request.task_plan.tasks for ordinal in task.operation_ordinals)
    )
    if len(operation_ordinals) != len(set(operation_ordinals)):
        raise ValueError("source task decomposition assigns an operation ordinal more than once")
    source_record: object = request.algebra_record
    if request.algebra_record.get("format") == "shuttle_domain_restricted_tensor_program_v1":
        source_record = request.algebra_record.get("program")
    if not isinstance(source_record, Mapping) or source_record.get("format") != "shuttle_tensor_program_v1":
        raise ValueError("JAX-owned pullback source must retain one canonical TensorProgram")
    source_operations = source_record.get("operations")
    if not isinstance(source_operations, (tuple, list)):
        raise ValueError("JAX-owned pullback source must declare its canonical operations")
    if operation_ordinals != tuple(range(len(source_operations))):
        raise ValueError("source task decomposition does not cover every transformed JAX operation exactly once")
    source_event_lowerings = lower_executable_event_operations(request)
    if any(not event.retained_operations for event in source_event_lowerings):
        raise ValueError("JAX-owned pullback source Events must remain imperative before algebraic replacement")
    replacement_event_lowerings = tuple(
        erase_executable_event_operations(
            event_lowering,
            event,
            kind=EventRealizationKind.ERASED_ALGEBRAIC_REFINEMENT,
            mechanism="jax_vjp_algebraic_refinement",
            reason="the exact JAX VJP task cover is replaced by the canonical pullback algebra",
            proof={
                "kind": "jax_vjp_exact_task_event_cover",
                "trace_fingerprint": trace.fingerprint,
                "source_algebra_fingerprint": request.algebra_fingerprint,
                "covered_task_ids": task_ids,
                "covered_operation_ordinals": operation_ordinals,
                "source_task": event.scheduled_dependence.source.name,
                "target_task": event.scheduled_dependence.target.name,
                "scheduled_pairs": tuple(
                    {"source": pair.source, "target": pair.target} for pair in event.scheduled_dependence.pairs
                ),
            },
        )
        for event_lowering, event in zip(source_event_lowerings, request.event_plans, strict=True)
    )
    event_witnesses = tuple(
        {
            "event": event.plan_name,
            "imperative_before_replacement": source.canonical_record,
            "algebraic_refinement": event.canonical_record,
        }
        for source, event in zip(source_event_lowerings, replacement_event_lowerings, strict=True)
    )
    replacement = JaxPullbackAlgebraicReplacement(
        source_algebra_fingerprint=request.algebra_fingerprint,
        trace_fingerprint=trace.fingerprint,
        covered_task_ids=task_ids,
        covered_operation_ordinals=operation_ordinals,
        event_witnesses=event_witnesses,
        identity=replacement_identity,
        numerical_effect=(
            "FAST replaces JAX's source-ordered quotient and maximum adjoints with the algebraically equivalent "
            "normalized-exponential pullback and schedule-defined FP32 reduction order"
        ),
    )
    pullback_tensor_program = replace(
        normalized_exp_contract_pullback_tensor_program(program.pullback),
        numerical_policy=NumericalPolicy.FAST,
    )
    unscheduled_pullback_request = jax_tensor_program_executable_request(
        pullback_tensor_program,
        tile_size=tile_size,
        target=request.target,
        normalized_exp_pullback=program.pullback,
    )
    pullback_preselection_event_lowerings = lower_executable_event_operations(unscheduled_pullback_request)
    if any(not event.retained_operations for event in pullback_preselection_event_lowerings):
        raise ValueError("canonical pullback Events must remain imperative before physical schedule selection")
    pullback_attachment = attach_sm90_contract_fold_schedule(unscheduled_pullback_request)
    if pullback_attachment.physical_schedule.parameters.get("direction") != "pullback":
        raise ValueError("canonical pullback task graph did not select the SM90 pullback schedule")
    pullback_composition = lower_canonical_kernel_composition(pullback_attachment.request)
    backward_internal_event_legalization = internal_event_device_legalization_region(
        1,
        lowering.input_event_schedule.allocations,
    )
    plan = ExecutableNormalizedExpPullbackPlan(
        request,
        trace,
        forward_plan,
        program,
        lowering,
        replacement,
        source_event_lowerings,
        replacement_event_lowerings,
        pullback_preselection_event_lowerings,
        pullback_attachment.request,
        pullback_composition,
        backward_internal_event_legalization,
    )
    plan.require_whole_request_runtime()
    return plan
