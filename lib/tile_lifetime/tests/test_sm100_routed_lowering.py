# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime import StreamingTileSchedule, apply_causal_score_mask, build_attention_tensor_program
from tile_lifetime.msa_frontend import MSADebugConfig
from tile_lifetime.msa_reference_pipeline import (
    build_msa_reference_oracle,
    derive_msa_reference_dataflow,
    msa_reference_physical_request,
)
from tile_lifetime.physical_task_contract import (
    PhysicalNumericalPolicy,
    RelationAccessClass,
    TaskOperationKind,
    canonical_fingerprint,
    physical_schedule_is_eligible,
)
from tile_lifetime.relation import build_relation_plan
from tile_lifetime.sm100_routed_lowering import (
    SM100RelationOrientation,
    default_sm100_routed_schedules,
    lower_sm100_routed_streaming_program,
    select_sm100_routed_lowerings,
)
from tile_lifetime.streaming_attention import derive_streaming_attention, scaled_score_map


def _program_and_relation(*, causal: bool = True):
    query_length = 128
    key_value_heads = 2
    right_block_count = 1
    selected_count = 4
    score_map = scaled_score_map(128**-0.5)
    if causal:
        score_map = apply_causal_score_mask(score_map)
    tensor_program = build_attention_tensor_program(
        batch_size=1,
        query_length=query_length,
        key_length=128,
        query_heads=8,
        key_value_heads=key_value_heads,
        key_dimension=128,
        value_dimension=128,
        score_map=score_map,
        input_dtype=DType.BF16,
    )
    program = derive_streaming_attention(
        tensor_program,
        schedule=StreamingTileSchedule(query_tile_size=128, key_value_tile_size=128, pipeline_depth=2),
    )
    source_count = query_length
    destinations = np.zeros((source_count, key_value_heads * selected_count), dtype=np.int32)
    edge_valid = np.zeros(destinations.shape, dtype=np.bool_)
    edge_valid[:, ::selected_count] = True
    relation = build_relation_plan(
        destinations,
        np.ones(destinations.shape, dtype=np.float32),
        edge_valid=edge_valid,
        destination_rank_by_item=np.zeros(right_block_count, dtype=np.int32),
        destination_local_item_by_item=np.arange(right_block_count, dtype=np.int32),
        padding_quantum=1,
    )
    return program, relation


def _physical_request(*, head_dimension: int = 128):
    config = MSADebugConfig(
        query_length=128,
        key_value_length=512,
        hidden_dimension=128,
        query_heads=8,
        key_value_heads=2,
        head_dimension=head_dimension,
        index_dimension=128,
        block_size=128,
        selected_blocks=4,
    )
    oracle = build_msa_reference_oracle(config)
    return msa_reference_physical_request(
        oracle,
        derive_msa_reference_dataflow(oracle, numerical_policy=PhysicalNumericalPolicy.FAST),
        numerical_policy=PhysicalNumericalPolicy.FAST,
    )


def test_sm100_routed_schedule_attaches_from_generic_phase_and_access_contracts() -> None:
    request = _physical_request()

    left_major, right_major = select_sm100_routed_lowerings(request)

    assert left_major.schedule.orientation is SM100RelationOrientation.LEFT_MAJOR
    assert right_major.schedule.orientation is SM100RelationOrientation.RIGHT_MAJOR
    assert left_major.physical_schedule.family == right_major.physical_schedule.family
    assert physical_schedule_is_eligible(right_major.physical_schedule, request.task_plan)
    assert tuple(request.task_plan.tasks[task_id].structure.principal_kind for task_id in right_major.task_ids) == (
        TaskOperationKind.RELATION,
        TaskOperationKind.CONTRACT,
        TaskOperationKind.FOLD,
        TaskOperationKind.CONTRACT,
        TaskOperationKind.MAP,
    )
    relation_task = request.task_plan.tasks[right_major.task_ids[0]]
    assert {access.relation_class for access in relation_task.accesses} == {RelationAccessClass.BOUNDED_DYNAMIC}
    assert right_major.left_extent == 128
    assert right_major.group_extent == 2
    assert right_major.left_lane_extent == 4
    assert right_major.selected_extent == 4
    assert right_major.score_feature_extent == right_major.value_feature_extent == right_major.fold_extent == 128
    assert len(right_major.physical_schedule.enforced_orders) == 6


def test_sm100_routed_schedule_rejects_unsupported_independent_value_and_score_dimension() -> None:
    assert select_sm100_routed_lowerings(_physical_request(head_dimension=64)) == ()


def test_sm100_routed_schedule_is_independent_of_domain_predicate_ast() -> None:
    source = _physical_request()
    records = tuple(
        {
            **source.algebra_record,
            "domain_restrictions": (
                {
                    "predicate": {
                        "kind": comparison,
                        "operands": (
                            {"kind": "input", "name": "left.position"},
                            {"kind": "input", "name": "right.position"},
                        ),
                    }
                },
            ),
        }
        for comparison in ("less_equal", "less_than")
    )
    variants = tuple(
        replace(source, algebra_record=record, algebra_fingerprint=canonical_fingerprint(record)) for record in records
    )

    selections = tuple(select_sm100_routed_lowerings(request) for request in variants)

    assert variants[0].fingerprint != variants[1].fingerprint
    assert tuple(candidate.physical_schedule for candidate in selections[0]) == tuple(
        candidate.physical_schedule for candidate in selections[1]
    )


def test_sm100_lowering_exposes_generic_semantics_and_both_orientations() -> None:
    program, relation = _program_and_relation()
    schedules = default_sm100_routed_schedules()

    lowered = tuple(lower_sm100_routed_streaming_program(program, relation, schedule) for schedule in schedules)

    assert [item.schedule.orientation for item in lowered] == [
        SM100RelationOrientation.LEFT_MAJOR,
        SM100RelationOrientation.RIGHT_MAJOR,
    ]
    assert lowered[0].head_group_size == 4
    assert lowered[0].query_tokens_per_task == 32
    assert lowered[0].selected_count == 4
    assert lowered[0].edge_group(5) == 1
    assert lowered[0].edge_selected_slot(5) == 1
    assert lowered[0].right_task_key(5, 0) == (1, 0)
    assert lowered[0].canonical_right_indices().shape == (2, 128, 4)
    assert np.array_equal(lowered[0].canonical_right_indices()[:, :, 0], np.zeros((2, 128), dtype=np.int32))
    assert np.all(lowered[0].canonical_right_indices()[:, :, 1:] == -1)
    assert "Contract -> score Map -> DomainRestriction -> normalized-exp Fold -> PV Contract" in lowered[0].dump()
    assert "external semantics: none" in lowered[0].dump()


def test_sm100_lowering_rejects_incomplete_group_edge_slots() -> None:
    program, relation = _program_and_relation()
    destinations = relation.destination_item.reshape(relation.source_item_count, relation.route_slots)[:, :-1].copy()
    invalid = build_relation_plan(
        destinations,
        np.ones(destinations.shape, dtype=np.float32),
        edge_valid=relation.edge_valid.reshape(relation.source_item_count, relation.route_slots)[:, :-1],
        destination_rank_by_item=np.zeros(1, dtype=np.int32),
        destination_local_item_by_item=np.arange(1, dtype=np.int32),
        padding_quantum=1,
    )

    with pytest.raises(ValueError, match="divide evenly"):
        lower_sm100_routed_streaming_program(program, invalid, default_sm100_routed_schedules()[1])


def test_sm100_lowering_mutates_domain_restriction_without_a_workload_switch() -> None:
    causal_program, relation = _program_and_relation(causal=True)
    unrestricted_program, _ = _program_and_relation(causal=False)
    schedule = default_sm100_routed_schedules()[1]

    causal = lower_sm100_routed_streaming_program(causal_program, relation, schedule)
    unrestricted = lower_sm100_routed_streaming_program(unrestricted_program, relation, schedule)

    assert causal.schedule == unrestricted.schedule
    assert causal.score_map.domain_traversal.value == "lower_triangular"
    assert unrestricted.score_map.domain_traversal.value == "full"
    assert causal.dump() != unrestricted.dump()
