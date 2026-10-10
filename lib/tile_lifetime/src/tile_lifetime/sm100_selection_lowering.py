# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach bounded SM100 Contract/Fold/Selection schedules to generic tasks."""

from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from tile_lifetime.physical_task_contract import (
    ExecutablePhysicalRequest,
    PhaseEdgeConstraint,
    PhysicalOrderKind,
    PhysicalSchedule,
    PhysicalScheduleFamily,
    PhysicalTarget,
    PhysicalTaskConstraint,
    PhysicalTaskOrder,
    RelationAccessClass,
    TaskCapability,
    TaskOperationKind,
    TaskResourceEstimate,
    TiledTask,
    physical_schedule_is_eligible,
)
from tile_lifetime.routed_attention import ProjectedBlockSelectionProgram


class SM100SelectionStrategy(StrEnum):
    """Where block scores cross the maximum-Fold/Selection boundary."""

    MATERIALIZED_BLOCK_SCORES = "materialized_block_scores"
    FUSED_STREAMING_TOP_K = "fused_streaming_top_k"


@dataclass(frozen=True)
class SM100SelectionSchedule:
    """Physical candidate for Contract, block maximum, and top-k selection."""

    strategy: SM100SelectionStrategy
    left_tile_size: int
    right_block_size: int
    feature_tile_size: int
    pipeline_stages: int
    matrix_threads: int
    selection_threads_per_row: int

    def __post_init__(self) -> None:
        if (self.left_tile_size, self.right_block_size, self.feature_tile_size) != (128, 128, 128):
            raise ValueError("the initial SM100 projected-selection skeleton requires 128x128x128 tiles")
        if self.pipeline_stages not in (1, 2, 3):
            raise ValueError("the initial SM100 projected-selection pipeline supports one to three stages")
        if self.matrix_threads <= 0 or self.matrix_threads % 32:
            raise ValueError("matrix worker count must be a positive number of warps")
        if self.selection_threads_per_row != 32:
            raise ValueError("the initial register top-k template uses one warp per logical row")


@dataclass(frozen=True)
class SM100SelectionTaskLowering:
    """One schedule attachment selected solely from generic task structure."""

    schedule: SM100SelectionSchedule
    physical_schedule: PhysicalSchedule
    source_request_fingerprint: str
    left_extent: int
    group_extent: int
    right_block_count: int
    selected_extent: int
    feature_extent: int
    right_block_size: int

    @property
    def task_ids(self) -> tuple[int, ...]:
        """Generic tasks covered by this fused physical schedule."""
        return self.physical_schedule.task_ids


def sm100_selection_schedule_family() -> PhysicalScheduleFamily:
    """Return the workload-neutral eligibility contract of the SM100 template."""
    relations = (RelationAccessClass.IDENTITY, RelationAccessClass.AFFINE)
    element_types = ("bf16", "bool", "fp32", "int32")
    layouts = ("row_major",)
    resources = TaskResourceEstimate(255, 232_448, 32)
    return PhysicalScheduleFamily(
        name="sm100_contract_fold_bounded_selection_128",
        target=PhysicalTarget.SM100,
        tasks=(
            PhysicalTaskConstraint(
                principal_kind=TaskOperationKind.CONTRACT,
                required_capabilities=(
                    TaskCapability.MATRIX_CONTRACT,
                    TaskCapability.INDEX_RELATION,
                    TaskCapability.SCALAR_MAP,
                ),
                supported_capabilities=(
                    TaskCapability.MATRIX_CONTRACT,
                    TaskCapability.INDEX_RELATION,
                    TaskCapability.SCALAR_MAP,
                ),
                required_edges=(
                    PhaseEdgeConstraint(
                        TaskCapability.MATRIX_CONTRACT,
                        TaskCapability.INDEX_RELATION,
                        RelationAccessClass.IDENTITY,
                    ),
                    PhaseEdgeConstraint(
                        TaskCapability.INDEX_RELATION,
                        TaskCapability.SCALAR_MAP,
                        RelationAccessClass.IDENTITY,
                    ),
                ),
                owner_rank=2,
                streamed_rank=1,
                reduction_rank=0,
                contraction_rank=1,
                supported_relation_classes=relations,
                supported_element_types=element_types,
                supported_layout_classes=layouts,
                maximum_resources=resources,
            ),
            PhysicalTaskConstraint(
                principal_kind=TaskOperationKind.FOLD,
                required_capabilities=(TaskCapability.ASSOCIATIVE_FOLD,),
                supported_capabilities=(TaskCapability.ASSOCIATIVE_FOLD,),
                required_edges=(),
                owner_rank=3,
                streamed_rank=0,
                reduction_rank=1,
                contraction_rank=0,
                supported_relation_classes=relations,
                supported_element_types=element_types,
                supported_layout_classes=layouts,
                maximum_resources=resources,
            ),
            PhysicalTaskConstraint(
                principal_kind=TaskOperationKind.SELECTION,
                required_capabilities=(TaskCapability.SCALAR_MAP, TaskCapability.BOUNDED_SELECTION),
                supported_capabilities=(
                    TaskCapability.INDEX_RELATION,
                    TaskCapability.SCALAR_MAP,
                    TaskCapability.BOUNDED_SELECTION,
                ),
                required_edges=(
                    PhaseEdgeConstraint(
                        TaskCapability.SCALAR_MAP,
                        TaskCapability.BOUNDED_SELECTION,
                        RelationAccessClass.IDENTITY,
                    ),
                ),
                owner_rank=3,
                streamed_rank=0,
                reduction_rank=1,
                contraction_rank=0,
                supported_relation_classes=relations,
                supported_element_types=element_types,
                supported_layout_classes=layouts,
                maximum_resources=resources,
            ),
        ),
    )


def select_sm100_selection_lowerings(
    request: ExecutablePhysicalRequest,
) -> tuple[SM100SelectionTaskLowering, ...]:
    """Discover every generic task chain eligible for the SM100 template.

    Source names, operation ordinals, and the full algebra operation sequence
    are deliberately ignored. Candidate discovery uses the declarative phase,
    axis, access, dtype, layout, resource, and exact task-dependence contracts.
    """
    if request.target is not PhysicalTarget.SM100:
        return ()
    family = sm100_selection_schedule_family()
    candidates = []
    for task_ids in _eligible_task_id_products(request, family):
        contract, fold, selection = (request.task_plan.tasks[task_id] for task_id in task_ids)
        first = _dependence_order(request, contract.id, fold.id)
        second = _dependence_order(request, fold.id, selection.id)
        if first is None or second is None:
            continue
        if not _selection_axes_are_compatible(contract, fold, selection):
            continue
        feature_extent = contract.structure.contraction_axes[0].extent
        right_block_size = fold.structure.reduction_axes[0].extent
        if feature_extent != 128 or right_block_size != 128:
            continue
        left_extent, group_extent = (axis.extent for axis in contract.structure.owner_axes)
        right_block_count = contract.structure.streamed_axes[0].extent
        selected_extent = selection.structure.owner_axes[2].extent
        if selected_extent not in (4, 8, 16, 32):
            continue
        for schedule in default_sm100_selection_schedules():
            physical = PhysicalSchedule(
                family=family,
                task_ids=task_ids,
                parameters={
                    "strategy": schedule.strategy.value,
                    "left_tile_size": schedule.left_tile_size,
                    "right_block_size": schedule.right_block_size,
                    "feature_tile_size": schedule.feature_tile_size,
                    "pipeline_stages": schedule.pipeline_stages,
                    "matrix_threads": schedule.matrix_threads,
                    "selection_threads_per_row": schedule.selection_threads_per_row,
                },
                enforced_orders=(first, second),
            )
            candidates.append(
                SM100SelectionTaskLowering(
                    schedule=schedule,
                    physical_schedule=physical,
                    source_request_fingerprint=request.fingerprint,
                    left_extent=left_extent,
                    group_extent=group_extent,
                    right_block_count=right_block_count,
                    selected_extent=selected_extent,
                    feature_extent=feature_extent,
                    right_block_size=right_block_size,
                )
            )
    return tuple(candidates)


def _eligible_task_id_products(
    request: ExecutablePhysicalRequest,
    family: PhysicalScheduleFamily,
) -> tuple[tuple[int, ...], ...]:
    eligible_by_constraint = []
    for constraint in family.tasks:
        singleton = PhysicalScheduleFamily(family.name, family.target, (constraint,))
        eligible_by_constraint.append(
            tuple(
                task.id
                for task in request.task_plan.tasks
                if physical_schedule_is_eligible(
                    PhysicalSchedule(singleton, (task.id,), {"eligibility_probe": True}),
                    request.task_plan,
                )
            )
        )
    return tuple(task_ids for task_ids in product(*eligible_by_constraint) if len(set(task_ids)) == len(task_ids))


def _dependence_order(
    request: ExecutablePhysicalRequest,
    source_task_id: int,
    target_task_id: int,
) -> PhysicalTaskOrder | None:
    source = request.task_plan.tasks[source_task_id]
    target = request.task_plan.tasks[target_task_id]
    matches = tuple(
        dependence.relation
        for dependence in request.task_plan.dependences
        if dependence.relation.source == source.family and dependence.relation.target == target.family
    )
    if len(matches) != 1:
        return None
    return PhysicalTaskOrder(
        kind=PhysicalOrderKind.PROGRAM_ORDER,
        source_task_id=source_task_id,
        target_task_id=target_task_id,
        relation=matches[0],
        mechanism_record={"kind": "fused_phase_order"},
    )


def _selection_axes_are_compatible(contract: TiledTask, fold: TiledTask, selection: TiledTask) -> bool:
    contract_owner = tuple(axis.axis_id for axis in contract.structure.owner_axes)
    fold_owner = tuple(axis.axis_id for axis in fold.structure.owner_axes)
    selection_owner = tuple(axis.axis_id for axis in selection.structure.owner_axes)
    block_axis = contract.structure.streamed_axes[0].axis_id
    return (
        fold_owner[:2] == contract_owner
        and fold_owner[2] == block_axis
        and fold.structure.reduction_axes[0].axis_id != block_axis
        and selection_owner[:2] == contract_owner
        and selection.structure.reduction_axes[0].axis_id == block_axis
    )


@dataclass(frozen=True)
class SM100ProjectedSelectionLowering:
    """Legacy benchmark geometry adapter; not the canonical request path."""

    program: ProjectedBlockSelectionProgram
    schedule: SM100SelectionSchedule
    right_block_count: int
    token_score_materialization_bytes: int
    block_score_materialization_bytes: int
    relation_index_bytes: int

    def dump(self) -> str:
        """Render the semantic and materialization decisions."""
        return "\n".join(
            (
                "SM100 projected relation selection",
                "  semantics: Contract(index Q/K) -> scale Map -> DomainRestriction "
                "-> block maximum Fold -> top-k Selection -> Relation",
                f"  strategy: {self.schedule.strategy.value}",
                (
                    f"  domains: left={self.program.source_count} groups={self.program.group_count} "
                    f"right_blocks={self.right_block_count} feature={self.program.relation_feature_count}"
                ),
                (
                    f"  materialization: token_scores={self.token_score_materialization_bytes} "
                    f"block_scores={self.block_score_materialization_bytes} "
                    f"relation_indices={self.relation_index_bytes}"
                ),
                f"  local block forced: {str(self.program.force_local_block).lower()}",
                f"  selection policy: {self.program.selection_semantics.scheduling_key}",
                "  external semantics: none",
            )
        )


def default_sm100_selection_schedules() -> tuple[SM100SelectionSchedule, ...]:
    """Return materialized and fused candidates without choosing by workload name."""
    common = {
        "left_tile_size": 128,
        "right_block_size": 128,
        "feature_tile_size": 128,
        "pipeline_stages": 2,
        "matrix_threads": 384,
        "selection_threads_per_row": 32,
    }
    return tuple(
        SM100SelectionSchedule(strategy=strategy, **common)
        for strategy in (
            SM100SelectionStrategy.MATERIALIZED_BLOCK_SCORES,
            SM100SelectionStrategy.FUSED_STREAMING_TOP_K,
        )
    )


def lower_sm100_projected_selection(
    program: ProjectedBlockSelectionProgram,
    schedule: SM100SelectionSchedule,
) -> SM100ProjectedSelectionLowering:
    """Build the quarantined legacy benchmark adapter from a semantic program."""
    if program.accumulation_dtype != "fp32":
        raise ValueError("the initial SM100 index Contract and maximum Fold accumulate in FP32")
    if program.relation_feature_count != schedule.feature_tile_size:
        raise ValueError("the initial SM100 index Contract requires relation feature dimension 128")
    if program.right_block_size != schedule.right_block_size:
        raise ValueError("the semantic Fold block must match the physical right tile")
    if program.selected_count not in (4, 8, 16, 32):
        raise ValueError("the initial SM100 top-k templates support k=4/8/16/32")
    if not program.force_local_block:
        raise ValueError("the primary selection candidate requires an explicit forced-local relation edge")

    right_block_count = program.right_block_count
    token_score_bytes = program.source_count * program.group_count * program.resolved_right_count * 4
    block_score_bytes = program.source_count * program.group_count * right_block_count * 4
    if schedule.strategy is SM100SelectionStrategy.FUSED_STREAMING_TOP_K:
        token_score_bytes = 0
    relation_index_bytes = program.source_count * program.group_count * program.selected_count * 4
    return SM100ProjectedSelectionLowering(
        program=program,
        schedule=schedule,
        right_block_count=right_block_count,
        token_score_materialization_bytes=token_score_bytes,
        block_score_materialization_bytes=block_score_bytes,
        relation_index_bytes=relation_index_bytes,
    )
