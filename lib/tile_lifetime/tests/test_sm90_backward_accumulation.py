# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU logical proofs; these tests do not compile or execute CUDA kernels."""

import math
import struct
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from tile_lifetime.cute_pipeline_artifact import (
    CuteKernelLaunch,
    CutePipelineElementType,
    CutePipelineLaunchRecipe,
    CutePipelineSlot,
    CutePipelineSlotAccess,
    CutePipelineSlotArgument,
    CutePipelineSlotKind,
    launch_dependency_event_schedule,
)
from tile_lifetime.executable_normalized_exp_contract import _relation_layout
from tile_lifetime.h100_normalized_exp_backward_lowering import _accumulation_plan, _input_event_schedule
from tile_lifetime.h100_streaming_lowering import Sm90BackwardAccumulatorSchedule, select_sm90_contract_schedule
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    NormalizedExpContractProgram,
    derive_forward_ownership_plan,
    normalized_exp_contract_pullback_program,
    normalized_exp_contract_region,
)
from tile_lifetime.sm90_backward_accumulation import (
    Sm90BackwardAccumulationPlan,
    ordered_backward_kernel_binding,
    ordered_backward_launch_roles,
    validate_ordered_backward_recipe,
    validate_ordered_backward_variant,
)
from tile_lifetime.tensor_program import (
    ContractOperandPrecision,
    EvaluationOrder,
    NumericalPolicy,
    deserialize_tensor_program,
)


def _plan(**changes):
    return replace(
        Sm90BackwardAccumulationPlan(2048, 2048, 64, 128, 1, 32, 128, 128, ContractDomainTraversal.FULL), **changes
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("extent", [1, 63, 64, 65, 127, 128, 129, 2048])
def test_contributor_ranks_cover_exact_dense_or_causal_storage(causal, extent):
    traversal = ContractDomainTraversal.LOWER_TRIANGULAR if causal else ContractDomainTraversal.FULL
    plan = _plan(owner_extent=extent, streamed_extent=extent, traversal=traversal)
    independent = {}
    for owner in range(math.ceil(extent / 64)):
        active = [
            streamed
            for streamed in range(math.ceil(extent / 128))
            if not causal
            or any(
                k <= q
                for q in range(owner * 64, min((owner + 1) * 64, extent))
                for k in range(streamed * 128, min((streamed + 1) * 128, extent))
            )
        ]
        for rank, streamed in enumerate(sorted(active, reverse=causal)):
            independent[owner, streamed] = rank
    assert plan.contributor_ranks == independent


@pytest.mark.parametrize(
    "batch,heads,owner,key",
    [(1, 1, 2048, 513), (2, 3, 2048, 2048), (3, 11, 2048, 2048), (1, 32, 2048, 2048), (2, 5, 65536, 257)],
)
@pytest.mark.parametrize("causal", [False, True])
def test_lpt_grid_is_a_bijection_including_swizzle_residuals(batch, heads, owner, key, causal):
    if causal:
        key = owner
    plan = _plan(
        batch=batch,
        heads=heads,
        owner_extent=owner,
        streamed_extent=key,
        traversal=ContractDomainTraversal.LOWER_TRIANGULAR if causal else ContractDomainTraversal.FULL,
    )
    actual = [plan.logical_tile(block) for block in range(plan.total_blocks)]
    expected = {(n, h, b) for b in range(batch) for h in range(heads) for n in range(math.ceil(key / 128))}
    assert len(actual) == len(set(actual)) == len(expected)
    assert set(actual) == expected
    for b in range(batch):
        for h in range(heads):
            sequence = [n for n, head, replica in actual if (head, replica) == (h, b)]
            assert sequence == sorted(sequence, reverse=causal)
    assert plan.counter_shape == (batch, heads, math.ceil(owner / 64), 1)
    assert plan.counter_bytes == batch * heads * math.ceil(owner / 64) * np.dtype(np.int32).itemsize
    with pytest.raises(ValueError, match="outside"):
        plan.logical_tile(plan.total_blocks)


def test_scheduler_uses_typed_fields_and_existing_fast_division_encoding():
    plan = _plan(batch=3, heads=11)
    arguments = plan.scheduler_arguments
    assert [argument.value for argument in arguments[:3]] == [528, 16, 16]
    assert arguments[-1].value == 2
    for argument, divisor in zip(arguments[3:7], (11, 16, 256, 1), strict=True):
        divisor_field, multiplier, shift1, shift2 = struct.unpack("<IIBBxx", argument.data)
        assert divisor_field == divisor and argument.alignment == 4
        for dividend in [0, 1, divisor - 1, divisor, divisor + 1, 527, 2**31 - 1]:
            high = dividend * multiplier >> 32
            quotient = (high + ((dividend - high) >> shift1)) >> shift2
            assert (quotient, dividend - quotient * divisor) == divmod(dividend, divisor)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"batch": 0}, "positive"),
        ({"traversal": "full"}, "traversal"),
        ({"heads": 2**31}, "int32"),
        ({"owner_tile": 32}, "64x128"),
        ({"producer_features": 64}, "D128"),
        ({"streamed_extent": 1024, "traversal": ContractDomainTraversal.LOWER_TRIANGULAR}, "equal"),
    ],
)
def test_unproved_scheduler_domains_reject(changes, message):
    with pytest.raises(ValueError, match=message):
        _plan(**changes)


@pytest.mark.parametrize(
    "group,transport,combined,message",
    [(4, False, False, "grouped"), (1, True, False, "transport"), (1, False, True, "combined")],
)
def test_unproved_ordered_variants_reject(group, transport, combined, message):
    with pytest.raises(ValueError, match=message):
        validate_ordered_backward_variant(relation_group_size=group, layout_transport=transport, combined=combined)


def _dense_boundary(traversal):
    program = deserialize_tensor_program(
        (Path(__file__).parent / "fixtures/dense_compile_boundary_program.json").read_text()
    )
    region = normalized_exp_contract_region(program)
    layout = _relation_layout(
        region,
        {
            "owner_extent": region.producer.inputs[0].shape[1],
            "streamed_extent": math.prod(axis.extent for axis in region.dimensions.normalized_axes),
            "producer_feature_extent": math.prod(axis.extent for axis in region.dimensions.producer_features),
            "consumer_feature_extent": math.prod(axis.extent for axis in region.dimensions.consumer_features),
        },
    )
    contract = select_sm90_contract_schedule(
        owner_extent=layout.owner_extent,
        streamed_extent=layout.streamed_extent,
        producer_feature_extent=layout.producer_feature_extent,
        consumer_feature_extent=layout.consumer_feature_extent,
        threads=384,
        owner_tile=64,
        streamed_tile=128,
        relation_layout=layout,
    )
    schedule = Sm90BackwardAccumulatorSchedule(
        contract=contract, consumer_output_cotangent_stages=1, deterministic_accumulation=True
    )
    forward = NormalizedExpContractProgram(
        program, region, derive_forward_ownership_plan(region), None, (region.finalizer.output,)
    )
    backward = normalized_exp_contract_pullback_program(forward, schedule=schedule, traversal=traversal)
    return backward, schedule, layout


@pytest.mark.parametrize(
    "traversal,expected_tiles,expected_edges",
    [(ContractDomainTraversal.FULL, 512, 480), (ContractDomainTraversal.LOWER_TRIANGULAR, 272, 240)],
)
def test_exact_dense_source_boundary_events_match_physical_contributor_generations(
    traversal, expected_tiles, expected_edges
):
    backward, schedule, layout = _dense_boundary(traversal)
    source_before = backward.forward.source.semantic_fingerprint
    plan = _accumulation_plan(backward, schedule, layout)
    events = _input_event_schedule(backward, schedule, layout)
    (queue,) = events.device_accumulation_queues
    actual = dict(zip(queue.allocation.active_coordinates, queue.allocation.generations, strict=True))
    assert actual == plan.contributor_ranks
    assert len(actual) == expected_tiles
    expected = {
        ((owner, left), (owner, right))
        for owner in range(plan.owner_blocks)
        for left, right in pairwise(plan.contributors(owner))
    }
    pairs = {(pair.source, pair.target) for pair in queue.allocation.plan.required_dependence.relation.pairs}
    assert pairs == expected and len(pairs) == expected_edges
    assert plan.grid == (512, 1, 1) and plan.counter_bytes == 4096
    assert backward.forward.source.semantic_fingerprint == source_before
    assert not _input_event_schedule(
        backward, replace(schedule, deterministic_accumulation=False), layout
    ).device_accumulation_queues


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("order", EvaluationOrder.SOURCE_ORDERED, "SOURCE_ORDERED"),
        ("operand_precision", (ContractOperandPrecision.HIGH, ContractOperandPrecision.DEFAULT), "DEFAULT"),
    ],
)
def test_ordered_physical_choice_does_not_silently_change_source_contract(field, value, message):
    backward, schedule, layout = _dense_boundary(ContractDomainTraversal.FULL)
    region = backward.forward.region
    producer = replace(region.producer, **{field: value})
    source = replace(
        backward.forward.source,
        operations=tuple(
            producer if operation == region.producer else operation for operation in backward.forward.source.operations
        ),
    )
    forward = replace(backward.forward, source=source, region=replace(region, producer=producer))
    invalid = normalized_exp_contract_pullback_program(
        forward, schedule=schedule, traversal=ContractDomainTraversal.FULL
    )
    with pytest.raises(ValueError, match=message):
        _accumulation_plan(invalid, schedule, layout)


# These strings exercise rejection/binding of compiler role metadata, not GPU
# execution. The unobserved reset/VJP spellings still need hardware capture.
PREFIXES = {
    "reset": "tile_lifetimebackendsh100cute_streaming_backward_emitter_Int32SemaphoreZeroFill",
    "vjp": "tile_lifetimebackendsh100cute_post_normalization_pullbackCutePostNormalizationPullback",
    "preprocess": "flash_attncuteflash_bwd_preprocessFlashAttentionBackwardPreprocess",
    "main": "tile_lifetimebackendsh100cute_normalized_exp_backward_sm90ShuttleNormalizedExponentialBackwardSm90",
    "postprocess": "flash_attncuteflash_bwd_postprocessFlashAttentionBackwardPostprocess",
}


def _recipe(expanded):
    plan = _plan(batch=3, heads=11)
    roles = (
        ("reset", "vjp", "preprocess", "vjp", "main", "postprocess")
        if expanded
        else ("reset", "preprocess", "main", "postprocess")
    )
    slots = (
        CutePipelineSlot(
            "query_accumulation_semaphore",
            CutePipelineElementType.I32,
            plan.counter_shape,
            CutePipelineSlotKind.SCRATCH,
            16,
        ),
        CutePipelineSlot(
            "query_accumulator",
            CutePipelineElementType.F32,
            (plan.batch, plan.heads, plan.owner_extent * 128),
            CutePipelineSlotKind.SCRATCH,
            16,
        ),
    )
    launches = []
    for index, role in enumerate(roles):
        accesses = {
            "reset": (("query_accumulation_semaphore", CutePipelineSlotAccess.WRITE),),
            "preprocess": (("query_accumulator", CutePipelineSlotAccess.WRITE),),
            "main": (
                ("query_accumulator", CutePipelineSlotAccess.READ_WRITE),
                ("query_accumulation_semaphore", CutePipelineSlotAccess.READ_WRITE),
            ),
            "vjp": (),
            "postprocess": (("query_accumulator", CutePipelineSlotAccess.READ),),
        }[role]
        args = tuple(CutePipelineSlotArgument(slot, slot, access) for slot, access in accesses)
        if role == "main":
            args += plan.scheduler_arguments
        grid = (
            plan.grid
            if role == "main"
            else (((plan.counter_bytes // 4 + 127) // 128, 1, 1) if role == "reset" else (1, 1, 1))
        )
        launches.append(
            CuteKernelLaunch(
                f"{role}_{index}",
                f"kernel_cutlass_kernel_{PREFIXES[role]}_object_at__fixture_{index}",
                grid,
                (128, 1, 1),
                0,
                (),
                args,
            )
        )
    return plan, CutePipelineLaunchRecipe(2, slots, tuple(launches))


@pytest.mark.parametrize("expanded", [False, True])
def test_each_invocation_resets_counter_and_accumulator_with_exact_shifted_event_indices(expanded):
    plan, recipe = _recipe(expanded)
    validate_ordered_backward_recipe(plan, recipe, expanded_cotangent=expanded)
    binding = ordered_backward_kernel_binding(
        tuple(launch.kernel_symbol for launch in recipe.launches), expanded_cotangent=expanded
    )
    assert binding.reset == recipe.launches[0].kernel_symbol
    assert binding.main == recipe.launches[-2].kernel_symbol
    assert (binding.prime_vjp, binding.vjp) == (
        (recipe.launches[1].kernel_symbol, recipe.launches[3].kernel_symbol) if expanded else (None, None)
    )
    # Model repeated recipe invocations on one stream using actual typed slots.
    repeated = tuple(
        replace(launch, name=f"call_{call}_{launch.name}") for call in range(2) for launch in recipe.launches
    )
    events = launch_dependency_event_schedule(repeated, slots=recipe.slots)
    pairs = set(events["predecessor_pairs"])
    required = {
        (tuple(p["source"]), tuple(p["target"]))
        for p in events["realizations"][0]["event"]["required_dependence"]["relation"]["pairs"]
    }
    main_index = 4 if expanded else 2
    preprocess = 2 if expanded else 1
    count = len(recipe.launches)
    for start in (0, count):
        for edge in (((start,), (start + main_index,)), ((start + preprocess,), (start + main_index,))):
            assert edge in pairs and edge in required
    assert ((main_index,), (count,)) in pairs


@pytest.mark.parametrize(
    "fault", ["missing", "unknown", "reordered", "duplicate", "grid", "counter", "access", "arguments"]
)
def test_malformed_ordered_recipe_rejects(fault):
    plan, recipe = _recipe(True)
    launches = list(recipe.launches)
    slots = recipe.slots
    if fault == "missing":
        launches.pop(0)
    elif fault == "unknown":
        launches[0] = replace(launches[0], kernel_symbol="unknown_kernel")
    elif fault == "reordered":
        launches[1], launches[2] = launches[2], launches[1]
    elif fault == "duplicate":
        launches[3] = replace(launches[3], kernel_symbol=launches[1].kernel_symbol)
    elif fault == "grid":
        launches[4] = replace(launches[4], grid=(16, 33, 1))
    elif fault == "counter":
        slots = (replace(slots[0], shape=(1, 33, 32, 1)), slots[1])
    elif fault == "access":
        launches[0] = replace(
            launches[0], arguments=(replace(launches[0].arguments[0], access=CutePipelineSlotAccess.READ_WRITE),)
        )
    elif fault == "arguments":
        launches[4] = replace(launches[4], arguments=launches[4].arguments[:-1])
    with pytest.raises(ValueError):
        validate_ordered_backward_recipe(
            plan, replace(recipe, launches=tuple(launches), slots=slots), expanded_cotangent=True
        )


def test_role_sequence_rejects_an_unexpected_optional_finalizer():
    _, recipe = _recipe(True)
    with pytest.raises(ValueError, match="roles must"):
        ordered_backward_launch_roles(
            tuple(launch.kernel_symbol for launch in recipe.launches), expanded_cotangent=False
        )


def test_ordered_accumulation_rejects_source_ordered_program_policy():
    backward, schedule, layout = _dense_boundary(ContractDomainTraversal.FULL)
    source = replace(backward.forward.source, numerical_policy=NumericalPolicy.SOURCE_ORDERED)
    invalid = replace(backward, forward=replace(backward.forward, source=source))
    with pytest.raises(ValueError, match="explicit FAST"):
        _accumulation_plan(invalid, schedule, layout)


def test_large_owner_replication_cannot_overflow_semaphore_zero_fill():
    with pytest.raises(ValueError, match="semaphore count or padded reset index"):
        _plan(owner_extent=2**31 - 1, streamed_extent=1, batch=1, heads=128)
    # Largest admitted whole owner-block/head product just below signed int32;
    # no tensors or Event domain arrays are allocated for this geometry check.
    plan = _plan(owner_extent=(2**31 - 64), streamed_extent=1, batch=1, heads=64)
    assert plan.counter_elements == 2**31 - 64
    assert plan.reset_grid[0] * plan.reset_block[0] - 1 == 2**31 - 1
