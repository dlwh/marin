# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-frontend conformance at canonical Shuttle compiler boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.native_task_plan_adapter import (
    NativeCanonicalTaskEventPlan,
    canonical_algebra_fingerprint_from_native_request,
    canonical_task_event_plan_from_native_request,
)
from tile_lifetime.physical_artifact_compiler import GpuPhysicalArtifactRequest
from tile_lifetime.physical_schedule_semantics import CrossPipelinePhysicalSemanticsComparison
from tile_lifetime.physical_task_contract import ExecutablePhysicalRequest, TaskOperationKind, canonical_fingerprint
from tile_lifetime.task_plan_refinement import (
    CanonicalTaskPlanRefinement,
    canonical_task_event_boundary_fingerprint,
)
from tile_lifetime.tensor_program import TensorProgram


class ConformanceStatus(StrEnum):
    """Strength of proof available at one canonical compiler boundary."""

    EXACT = "exact"
    PROVEN_REFINEMENT = "proven_refinement"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CanonicalBoundaryComparison:
    """One semantic comparison between records using the same schema."""

    boundary: str
    status: ConformanceStatus
    executable_fingerprint: str
    native_fingerprint: str | None
    detail: str

    def __post_init__(self) -> None:
        if not self.boundary or not self.detail:
            raise ValueError("canonical boundary comparisons require a boundary and detail")
        if len(self.executable_fingerprint) != 64:
            raise ValueError("executable boundary fingerprint must be a canonical SHA-256 digest")
        if self.native_fingerprint is not None and len(self.native_fingerprint) != 64:
            raise ValueError("native boundary fingerprint must be a canonical SHA-256 digest")
        if self.status is ConformanceStatus.EXACT and self.native_fingerprint != self.executable_fingerprint:
            raise ValueError("exact conformance requires identical canonical fingerprints")
        if self.status is ConformanceStatus.UNAVAILABLE and self.native_fingerprint is not None:
            raise ValueError("an unavailable canonical boundary cannot claim a native fingerprint")

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return the diagnostic comparison without redefining semantic identity."""
        return {
            "boundary": self.boundary,
            "status": self.status.value,
            "executable_fingerprint": self.executable_fingerprint,
            "native_fingerprint": self.native_fingerprint,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SourceContainerFingerprints:
    """Outer encodings retained separately from canonical semantics."""

    executable_artifact: str
    native_request: str

    def __post_init__(self) -> None:
        if len(self.executable_artifact) != 64 or len(self.native_request) != 64:
            raise ValueError("source-container fingerprints must be SHA-256 digests")

    @property
    def same_encoding_identity(self) -> bool:
        """Return exact outer-container identity, not semantic equivalence."""
        return self.executable_artifact == self.native_request


@dataclass(frozen=True)
class PhysicalScheduleCover:
    """Connect schedules implementing the same logical coarse task set."""

    native_schedule_index: int
    executable_schedule_indices: tuple[int, ...]
    parameter_equivalences: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.native_schedule_index < 0 or not self.executable_schedule_indices:
            raise ValueError("physical schedule covers require concrete schedule indices")
        if min(self.executable_schedule_indices) < 0:
            raise ValueError("physical schedule cover indices must be non-negative")
        if len(set(self.executable_schedule_indices)) != len(self.executable_schedule_indices):
            raise ValueError("a physical schedule cover cannot repeat an executable schedule")
        if not self.parameter_equivalences:
            raise ValueError("physical schedule covers require explicit parameter equivalences")


@dataclass(frozen=True)
class CrossPipelineTaskRefinement:
    """Two exact frontend refinements onto one logical task/Event quotient."""

    executable_refinement: CanonicalTaskPlanRefinement
    native_refinement: CanonicalTaskPlanRefinement

    def __post_init__(self) -> None:
        if self.executable_refinement.quotient_fingerprint != self.native_refinement.quotient_fingerprint:
            raise ValueError("cross-pipeline refinements require one shared task/Event quotient")

    @property
    def proof_record(self) -> dict[str, object]:
        """Return both endpoint proofs without merging their identities."""
        return {
            "schema": "shuttle.cross-pipeline-task-refinement.v1",
            "quotient_fingerprint": self.executable_refinement.quotient_fingerprint,
            "executable_proof": self.executable_refinement.proof_record,
            "native_proof": self.native_refinement.proof_record,
        }

    @property
    def fingerprint(self) -> str:
        """Fingerprint the complete two-endpoint proof."""
        return canonical_fingerprint(self.proof_record)


@dataclass(frozen=True)
class CrossPipelinePhysicalRefinement:
    """Two task quotients plus schedule covers proving physical refinement."""

    executable_task_refinement: CanonicalTaskPlanRefinement
    native_coordinate_refinement: CanonicalTaskPlanRefinement
    schedule_covers: tuple[PhysicalScheduleCover, ...]
    executable_physical_fingerprint: str
    native_physical_fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.executable_task_refinement.coarse_plan.fingerprint
            != self.native_coordinate_refinement.coarse_plan.fingerprint
        ):
            raise ValueError("cross-pipeline physical refinement requires one shared logical coarse plan")
        if self.executable_task_refinement.algebra_fingerprint != self.native_coordinate_refinement.algebra_fingerprint:
            raise ValueError("cross-pipeline physical refinements disagree on algebra identity")
        if self.executable_task_refinement.target is not self.native_coordinate_refinement.target:
            raise ValueError("cross-pipeline physical refinements disagree on target")
        if self.executable_task_refinement.numerical_policy is not self.native_coordinate_refinement.numerical_policy:
            raise ValueError("cross-pipeline physical refinements disagree on numerical policy")
        if not self.schedule_covers:
            raise ValueError("cross-pipeline physical refinement requires schedule covers")
        if len(self.executable_physical_fingerprint) != 64 or len(self.native_physical_fingerprint) != 64:
            raise ValueError("cross-pipeline physical refinement requires canonical endpoint fingerprints")

    @property
    def proof_record(self) -> dict[str, object]:
        """Return source-specific evidence without merging request identities."""
        return {
            "schema": "shuttle.cross-pipeline-physical-refinement.v1",
            "logical_quotient_fingerprint": self.executable_task_refinement.quotient_fingerprint,
            "executable_task_proof": self.executable_task_refinement.proof_record,
            "native_coordinate_proof": self.native_coordinate_refinement.proof_record,
            "executable_physical_fingerprint": self.executable_physical_fingerprint,
            "native_physical_fingerprint": self.native_physical_fingerprint,
            "schedule_covers": tuple(
                {
                    "native_schedule": cover.native_schedule_index,
                    "executable_schedules": cover.executable_schedule_indices,
                    "parameter_equivalences": cover.parameter_equivalences,
                }
                for cover in self.schedule_covers
            ),
        }

    @property
    def fingerprint(self) -> str:
        """Fingerprint the complete refinement evidence."""
        return canonical_fingerprint(self.proof_record)


@dataclass(frozen=True)
class CrossPipelineConformance:
    """Proof report for executable-DSL and native-MLIR compiler boundaries."""

    algebra: CanonicalBoundaryComparison
    ownership_task_plan: CanonicalBoundaryComparison
    event_tensor: CanonicalBoundaryComparison
    physical_request: CanonicalBoundaryComparison
    source_containers: SourceContainerFingerprints
    native_adaptation_error: str | None = None

    @property
    def accepted(self) -> bool:
        """Return whether every semantic boundary has a positive proof."""
        positive = {ConformanceStatus.EXACT, ConformanceStatus.PROVEN_REFINEMENT}
        return all(
            comparison.status in positive
            for comparison in (
                self.algebra,
                self.ownership_task_plan,
                self.event_tensor,
                self.physical_request,
            )
        )

    @property
    def canonical_record(self) -> dict[str, object]:
        """Return a stable audit record whose verdict is not artifact identity."""
        return {
            "schema": "shuttle.cross-pipeline-conformance.v1",
            "accepted": self.accepted,
            "boundaries": tuple(
                comparison.canonical_record
                for comparison in (
                    self.algebra,
                    self.ownership_task_plan,
                    self.event_tensor,
                    self.physical_request,
                )
            ),
            "source_containers": {
                "executable_artifact": self.source_containers.executable_artifact,
                "native_request": self.source_containers.native_request,
                "same_encoding_identity": self.source_containers.same_encoding_identity,
            },
            "native_adaptation_error": self.native_adaptation_error,
        }

    @property
    def fingerprint(self) -> str:
        """Fingerprint this audit result, including negative evidence."""
        return canonical_fingerprint(self.canonical_record)


def compare_cross_pipeline_conformance(
    executable: ExecutablePhysicalRequest,
    native_request: GpuPhysicalArtifactRequest,
    *,
    algebra_program: TensorProgram,
    refinement: CanonicalTaskPlanRefinement | None = None,
    native_physical_request: ExecutablePhysicalRequest | None = None,
    cross_pipeline_refinement: CrossPipelineTaskRefinement | None = None,
    physical_refinement: CrossPipelinePhysicalRefinement | None = None,
    physical_semantics: CrossPipelinePhysicalSemanticsComparison | None = None,
) -> CrossPipelineConformance:
    """Compare executable and native paths without equating outer encodings.

    ``native_physical_request`` must use the shared ``ExecutablePhysicalRequest``
    schema. A raw native provider request is deliberately insufficient to prove
    physical-request equivalence because it serializes MLIR containers and
    source plans rather than the executable DSL's canonical request record.
    """
    if physical_semantics is not None:
        quotient_fingerprints = {
            proof.quotient_fingerprint
            for proof in (
                refinement,
                None if cross_pipeline_refinement is None else cross_pipeline_refinement.executable_refinement,
            )
            if proof is not None
        }
        if quotient_fingerprints != {physical_semantics.logical_quotient_fingerprint}:
            raise ValueError("physical semantics comparison is not bound to the selected logical quotient proof")
    containers = SourceContainerFingerprints(executable.artifact_fingerprint, native_request.fingerprint)
    try:
        native_algebra_fingerprint = canonical_algebra_fingerprint_from_native_request(
            native_request,
            algebra_program=algebra_program,
        )
    except ValueError as error:
        unavailable = str(error)
        return CrossPipelineConformance(
            _unavailable("algebra", executable.algebra_fingerprint, unavailable),
            _unavailable("ownership_task_plan", executable.task_plan.fingerprint, unavailable),
            _unavailable("event_tensor", _executable_event_fingerprint(executable), unavailable),
            _physical_comparison(executable, native_physical_request, physical_refinement, physical_semantics),
            containers,
            native_adaptation_error=unavailable,
        )

    algebra = _exact_or_mismatch(
        "algebra",
        executable.algebra_fingerprint,
        native_algebra_fingerprint,
        mismatch="native and executable canonical algebra records differ",
    )
    try:
        native = canonical_task_event_plan_from_native_request(native_request, algebra_program=algebra_program)
    except ValueError as error:
        unavailable = str(error)
        return CrossPipelineConformance(
            algebra,
            _unavailable("ownership_task_plan", executable.task_plan.fingerprint, unavailable),
            _unavailable("event_tensor", _executable_event_fingerprint(executable), unavailable),
            _physical_comparison(executable, native_physical_request, physical_refinement, physical_semantics),
            containers,
            native_adaptation_error=unavailable,
        )
    if physical_refinement is not None:
        if native_physical_request is None:
            raise ValueError("physical refinement requires a native ExecutablePhysicalRequest")
        _verify_physical_refinement(physical_refinement, executable, native_physical_request, native)
    if refinement is not None and cross_pipeline_refinement is not None:
        raise ValueError("choose a direct or two-endpoint task refinement, not both")
    if cross_pipeline_refinement is not None:
        _verify_cross_pipeline_task_refinement(cross_pipeline_refinement, executable, native)
    tasks = _task_comparison(executable, native, refinement, cross_pipeline_refinement, physical_refinement)
    events = _event_comparison(executable, native, refinement, cross_pipeline_refinement, physical_refinement)
    return CrossPipelineConformance(
        algebra,
        tasks,
        events,
        _physical_comparison(executable, native_physical_request, physical_refinement, physical_semantics),
        containers,
    )


def _task_comparison(
    executable: ExecutablePhysicalRequest,
    native: NativeCanonicalTaskEventPlan,
    refinement: CanonicalTaskPlanRefinement | None,
    cross_pipeline_refinement: CrossPipelineTaskRefinement | None,
    physical_refinement: CrossPipelinePhysicalRefinement | None,
) -> CanonicalBoundaryComparison:
    executable_fingerprint = executable.task_plan.fingerprint
    native_fingerprint = native.task_plan.fingerprint
    if executable_fingerprint == native_fingerprint:
        return _exact_or_mismatch("ownership_task_plan", executable_fingerprint, native_fingerprint, mismatch="")
    if refinement is not None:
        _verify_refinement_endpoints(refinement, executable, native, physical_refinement)
        return CanonicalBoundaryComparison(
            "ownership_task_plan",
            ConformanceStatus.PROVEN_REFINEMENT,
            executable_fingerprint,
            native_fingerprint,
            f"exact quotient proof {_refinement_fingerprint(refinement, physical_refinement)}",
        )
    if cross_pipeline_refinement is not None:
        return CanonicalBoundaryComparison(
            "ownership_task_plan",
            ConformanceStatus.PROVEN_REFINEMENT,
            executable_fingerprint,
            native_fingerprint,
            f"shared two-endpoint quotient proof {cross_pipeline_refinement.fingerprint}",
        )
    return CanonicalBoundaryComparison(
        "ownership_task_plan",
        ConformanceStatus.MISMATCH,
        executable_fingerprint,
        native_fingerprint,
        "task ownership, tiling, dependencies, or resource records differ without a quotient proof",
    )


def _event_comparison(
    executable: ExecutablePhysicalRequest,
    native: NativeCanonicalTaskEventPlan,
    refinement: CanonicalTaskPlanRefinement | None,
    cross_pipeline_refinement: CrossPipelineTaskRefinement | None,
    physical_refinement: CrossPipelinePhysicalRefinement | None,
) -> CanonicalBoundaryComparison:
    executable_fingerprint = _executable_event_fingerprint(executable)
    native_fingerprint = native.event_fingerprint
    if executable_fingerprint == native_fingerprint:
        return _exact_or_mismatch("event_tensor", executable_fingerprint, native_fingerprint, mismatch="")
    if refinement is not None:
        _verify_refinement_endpoints(refinement, executable, native, physical_refinement)
        return CanonicalBoundaryComparison(
            "event_tensor",
            ConformanceStatus.PROVEN_REFINEMENT,
            executable_fingerprint,
            native_fingerprint,
            f"exact quotient proof {_refinement_fingerprint(refinement, physical_refinement)}",
        )
    if cross_pipeline_refinement is not None:
        return CanonicalBoundaryComparison(
            "event_tensor",
            ConformanceStatus.PROVEN_REFINEMENT,
            executable_fingerprint,
            native_fingerprint,
            f"shared two-endpoint quotient proof {cross_pipeline_refinement.fingerprint}",
        )
    return CanonicalBoundaryComparison(
        "event_tensor",
        ConformanceStatus.MISMATCH,
        executable_fingerprint,
        native_fingerprint,
        "logical EventTensor relations, visibility, or bounded generations differ without a quotient proof",
    )


def _physical_comparison(
    executable: ExecutablePhysicalRequest,
    native: ExecutablePhysicalRequest | None,
    refinement: CrossPipelinePhysicalRefinement | None,
    semantics: CrossPipelinePhysicalSemanticsComparison | None,
) -> CanonicalBoundaryComparison:
    if native is None:
        return _unavailable(
            "physical_request",
            executable.fingerprint,
            "native path has not emitted the shared ExecutablePhysicalRequest canonical schema",
        )
    if executable.fingerprint == native.fingerprint:
        return _exact_or_mismatch("physical_request", executable.fingerprint, native.fingerprint, mismatch="")
    if semantics is not None:
        if semantics.executable_request_fingerprint != executable.fingerprint:
            raise ValueError("physical semantics executable endpoint disagrees with the request")
        if semantics.native_request_fingerprint != native.fingerprint:
            raise ValueError("physical semantics native endpoint disagrees with the request")
        if semantics.accepted:
            return CanonicalBoundaryComparison(
                "physical_request",
                ConformanceStatus.PROVEN_REFINEMENT,
                executable.fingerprint,
                native.fingerprint,
                f"exact owner-scoped physical semantics {semantics.fingerprint}",
            )
    if refinement is not None:
        return CanonicalBoundaryComparison(
            "physical_request",
            ConformanceStatus.PROVEN_REFINEMENT,
            executable.fingerprint,
            native.fingerprint,
            f"verified physical schedule/Event cover {refinement.fingerprint}",
        )
    executable_record = executable.canonical_record
    native_record = native.canonical_record
    differing_fields = tuple(key for key in executable_record if executable_record.get(key) != native_record.get(key))
    owner_detail = ""
    if semantics is not None:
        owner_detail = "; owner semantics: " + ", ".join(
            f"{owner.owner_task_id}=[{', '.join(owner.differing_fields)}]" for owner in semantics.owners
        )
    return CanonicalBoundaryComparison(
        "physical_request",
        ConformanceStatus.MISMATCH,
        executable.fingerprint,
        native.fingerprint,
        f"canonical physical fields differ: {', '.join(differing_fields)}{owner_detail}",
    )


def _verify_physical_refinement(
    refinement: CrossPipelinePhysicalRefinement,
    executable: ExecutablePhysicalRequest,
    native_physical: ExecutablePhysicalRequest,
    native: NativeCanonicalTaskEventPlan,
) -> None:
    executable_refinement = refinement.executable_task_refinement
    native_refinement = refinement.native_coordinate_refinement
    if refinement.executable_physical_fingerprint != executable.fingerprint:
        raise ValueError("physical refinement executable endpoint fingerprint disagrees with the request")
    if refinement.native_physical_fingerprint != native_physical.fingerprint:
        raise ValueError("physical refinement native endpoint fingerprint disagrees with the request")
    if executable_refinement.refined_plan.fingerprint != executable.task_plan.fingerprint:
        raise ValueError("physical refinement does not cover the executable task plan")
    if native_refinement.refined_plan.fingerprint != native_physical.task_plan.fingerprint:
        raise ValueError("physical refinement does not cover the native physical task plan")
    if native_physical.task_plan.fingerprint != native.task_plan.fingerprint:
        raise ValueError("native physical schema disagrees with the adapted native task plan")
    if executable_refinement.refined_events != executable.event_plans:
        raise ValueError("physical refinement does not cover executable Event allocations")
    if native_refinement.refined_events != native_physical.event_plans:
        raise ValueError("physical refinement does not cover native Event allocations")
    if executable_refinement.refined_event_programs != executable.event_programs:
        raise ValueError("physical refinement does not cover executable parametric Event allocations")
    if native_refinement.refined_event_programs != native_physical.event_programs:
        raise ValueError("physical refinement does not cover native parametric Event allocations")
    if executable_refinement.target is not executable.target or native_refinement.target is not native_physical.target:
        raise ValueError("physical refinement target disagrees with an endpoint request")
    if (
        executable_refinement.numerical_policy is not executable.numerical_policy
        or native_refinement.numerical_policy is not native_physical.numerical_policy
    ):
        raise ValueError("physical refinement numerical policy disagrees with an endpoint request")
    if (
        executable_refinement.algebra_fingerprint != executable.algebra_fingerprint
        or native_refinement.algebra_fingerprint != native_physical.algebra_fingerprint
    ):
        raise ValueError("physical refinement algebra identity disagrees with an endpoint request")

    executable_coarse_by_task = {
        binding.refined_task_id: binding.coarse_task_id for binding in executable_refinement.bindings
    }
    native_coarse_by_task = {binding.refined_task_id: binding.coarse_task_id for binding in native_refinement.bindings}
    native_indices = tuple(cover.native_schedule_index for cover in refinement.schedule_covers)
    executable_indices = tuple(
        schedule_index for cover in refinement.schedule_covers for schedule_index in cover.executable_schedule_indices
    )
    if tuple(sorted(native_indices)) != tuple(range(len(native_physical.schedules))):
        raise ValueError("physical schedule covers must partition every native schedule")
    if tuple(sorted(executable_indices)) != tuple(range(len(executable.schedules))):
        raise ValueError("physical schedule covers must partition every executable schedule")
    for cover in refinement.schedule_covers:
        native_schedule = native_physical.schedules[cover.native_schedule_index]
        executable_schedules = tuple(executable.schedules[index] for index in cover.executable_schedule_indices)
        native_logical_tasks = {native_coarse_by_task[task_id] for task_id in native_schedule.task_ids}
        executable_logical_tasks = {
            executable_coarse_by_task[task_id] for schedule in executable_schedules for task_id in schedule.task_ids
        }
        if native_logical_tasks != executable_logical_tasks:
            raise ValueError("physical schedule cover endpoints implement different logical task sets")
        for executable_parameter, native_parameter in cover.parameter_equivalences:
            if native_parameter not in native_schedule.parameters:
                raise ValueError(f"native schedule has no parameter {native_parameter!r}")
            native_value = native_schedule.parameters[native_parameter]
            if any(schedule.parameters.get(executable_parameter) != native_value for schedule in executable_schedules):
                raise ValueError(
                    f"physical schedule parameter {executable_parameter!r} does not refine "
                    f"native parameter {native_parameter!r}"
                )


def _verify_cross_pipeline_task_refinement(
    refinement: CrossPipelineTaskRefinement,
    executable: ExecutablePhysicalRequest,
    native: NativeCanonicalTaskEventPlan,
) -> None:
    executable_refinement = refinement.executable_refinement
    native_refinement = refinement.native_refinement
    if executable_refinement.refined_plan.fingerprint != executable.task_plan.fingerprint:
        raise ValueError("cross-pipeline refinement does not cover the executable task plan")
    if native_refinement.refined_plan.fingerprint != native.task_plan.fingerprint:
        raise ValueError("cross-pipeline refinement does not cover the native task plan")
    if executable_refinement.refined_events != executable.event_plans:
        raise ValueError("cross-pipeline refinement does not cover executable concrete Events")
    if native_refinement.refined_events != native.events:
        raise ValueError("cross-pipeline refinement does not cover native concrete Events")
    if executable_refinement.refined_event_programs != executable.event_programs:
        raise ValueError("cross-pipeline refinement does not cover executable parametric Events")
    if native_refinement.refined_event_programs != native.event_programs:
        raise ValueError("cross-pipeline refinement does not cover native parametric Events")
    if (
        executable_refinement.algebra_fingerprint != executable.algebra_fingerprint
        or native_refinement.algebra_fingerprint != native.algebra_fingerprint
    ):
        raise ValueError("cross-pipeline refinement algebra identity disagrees with an endpoint")
    helper_ids = {
        task.id for task in native.task_plan.tasks if task.operation_kinds == (TaskOperationKind.IMPLEMENTATION_HELPER,)
    }
    helper_proofs = {proof.helper_task_id: proof for proof in native.implementation_helper_refinements}
    if set(helper_proofs) != helper_ids:
        raise ValueError("cross-pipeline refinement requires an exact proof for every implementation helper")
    coarse_by_native = {binding.refined_task_id: binding.coarse_task_id for binding in native_refinement.bindings}
    for helper_id, proof in helper_proofs.items():
        helper_coarse = coarse_by_native[helper_id]
        if any(coarse_by_native[owner_id] != helper_coarse for _, owner_id in proof.operation_owners):
            raise ValueError("implementation helper and canonical owners must internalize to one coarse task")


def _verify_refinement_endpoints(
    refinement: CanonicalTaskPlanRefinement,
    executable: ExecutablePhysicalRequest,
    native: NativeCanonicalTaskEventPlan,
    physical_refinement: CrossPipelinePhysicalRefinement | None,
) -> None:
    native_endpoint = (
        native.task_plan.fingerprint
        if physical_refinement is None
        else physical_refinement.native_coordinate_refinement.coarse_plan.fingerprint
    )
    plans = {refinement.coarse_plan.fingerprint, refinement.refined_plan.fingerprint}
    if plans != {executable.task_plan.fingerprint, native_endpoint}:
        raise ValueError("task refinement does not connect the compared executable and native plans")
    if refinement.algebra_fingerprint != executable.algebra_fingerprint:
        raise ValueError("task refinement algebra identity disagrees with the compared executable request")


def _refinement_fingerprint(
    refinement: CanonicalTaskPlanRefinement,
    physical_refinement: CrossPipelinePhysicalRefinement | None,
) -> str:
    return refinement.quotient_fingerprint if physical_refinement is None else physical_refinement.fingerprint


def _executable_event_fingerprint(executable: ExecutablePhysicalRequest) -> str:
    return canonical_task_event_boundary_fingerprint(
        executable.task_plan,
        executable.event_plans,
        executable.event_programs,
    )


def _exact_or_mismatch(
    boundary: str,
    executable_fingerprint: str,
    native_fingerprint: str,
    *,
    mismatch: str,
) -> CanonicalBoundaryComparison:
    if executable_fingerprint == native_fingerprint:
        return CanonicalBoundaryComparison(
            boundary,
            ConformanceStatus.EXACT,
            executable_fingerprint,
            native_fingerprint,
            "identical canonical records",
        )
    return CanonicalBoundaryComparison(
        boundary,
        ConformanceStatus.MISMATCH,
        executable_fingerprint,
        native_fingerprint,
        mismatch,
    )


def _unavailable(boundary: str, executable_fingerprint: str, detail: str) -> CanonicalBoundaryComparison:
    return CanonicalBoundaryComparison(
        boundary,
        ConformanceStatus.UNAVAILABLE,
        executable_fingerprint,
        None,
        detail,
    )
