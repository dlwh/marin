# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU plan for the admitted dense SM90 backward accumulation protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.cute_fast_divmod import encode_cute_fast_divmod_divisor
from tile_lifetime.cute_pipeline_artifact import (
    CutePipelineElementType,
    CutePipelineLaunchRecipe,
    CutePipelinePackedBytesArgument,
    CutePipelineScalarArgument,
    CutePipelineSlotKind,
    _launch_slot_accesses,
    launch_dependency_event_schedule,
)
from tile_lifetime.normalized_exp_contract_region import ContractDomainTraversal


class Sm90AccumulationMode(StrEnum):
    UNORDERED = "unordered"
    ORDERED = "ordered"


@dataclass(frozen=True)
class Sm90BackwardAccumulationPlan:
    """Dense dQ contributor order and the existing static LPT scheduler ABI.

    A queue describes one batch/head replica. Each invocation resets all of its
    replicas. This is a fixed FAST physical order, not SOURCE_ORDERED arithmetic
    and not a concurrency guarantee for shared workspace.
    """

    owner_extent: int
    streamed_extent: int
    owner_tile: int
    streamed_tile: int
    batch: int
    heads: int
    producer_features: int
    consumer_features: int
    traversal: ContractDomainTraversal

    def __post_init__(self) -> None:
        dimensions = (
            self.owner_extent,
            self.streamed_extent,
            self.owner_tile,
            self.streamed_tile,
            self.batch,
            self.heads,
            self.producer_features,
            self.consumer_features,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("ordered SM90 dimensions must be positive integers")
        if not isinstance(self.traversal, ContractDomainTraversal) or self.traversal not in (
            ContractDomainTraversal.FULL,
            ContractDomainTraversal.LOWER_TRIANGULAR,
        ):
            raise ValueError("ordered SM90 accumulation requires dense FULL or causal traversal")
        if self.descending and self.owner_extent != self.streamed_extent:
            raise ValueError("ordered causal accumulation requires equal owner and streamed extents")
        if (self.owner_tile, self.streamed_tile) != (64, 128):
            raise ValueError("ordered SM90 accumulation requires the admitted 64x128 physical tile")
        if (self.producer_features, self.consumer_features) != (128, 128):
            raise ValueError("ordered SM90 accumulation requires the admitted BF16 D128 physical layout")
        if any(
            value > (1 << 31) - 1 for value in (*dimensions, self.total_blocks, self.l2_minor * self.streamed_blocks)
        ):
            raise ValueError("ordered scheduler arguments exceed signed int32")
        last_reset_index = self.reset_grid[0] * self.reset_block[0] - 1
        if self.counter_elements > (1 << 31) - 1 or last_reset_index > (1 << 31) - 1:
            raise ValueError("ordered semaphore count or padded reset index exceeds signed int32")

    @property
    def descending(self) -> bool:
        return self.traversal is ContractDomainTraversal.LOWER_TRIANGULAR

    @property
    def owner_blocks(self) -> int:
        return (self.owner_extent + self.owner_tile - 1) // self.owner_tile

    @property
    def streamed_blocks(self) -> int:
        return (self.streamed_extent + self.streamed_tile - 1) // self.streamed_tile

    def contributors(self, owner: int) -> tuple[int, ...]:
        if not 0 <= owner < self.owner_blocks:
            raise ValueError("ordered accumulation owner is outside the tiled domain")
        limit = self.streamed_blocks
        if self.descending:
            last_row = min((owner + 1) * self.owner_tile, self.owner_extent)
            limit = min(limit, (last_row + self.streamed_tile - 1) // self.streamed_tile)
        return tuple(range(limit - 1, -1, -1)) if self.descending else tuple(range(limit))

    @property
    def contributor_ranks(self) -> dict[tuple[int, int], int]:
        return {
            (owner, contributor): rank
            for owner in range(self.owner_blocks)
            for rank, contributor in enumerate(self.contributors(owner))
        }

    @property
    def counter_shape(self) -> tuple[int, int, int, int]:
        return self.batch, self.heads, self.owner_blocks, 1

    @property
    def counter_elements(self) -> int:
        return self.batch * self.heads * self.owner_blocks

    @property
    def counter_bytes(self) -> int:
        return self.counter_elements * 4

    @property
    def reset_block(self) -> tuple[int, int, int]:
        return 128, 1, 1

    @property
    def reset_grid(self) -> tuple[int, int, int]:
        return (self.counter_elements + self.reset_block[0] - 1) // self.reset_block[0], 1, 1

    @property
    def total_blocks(self) -> int:
        return self.streamed_blocks * self.batch * self.heads

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.total_blocks, 1, 1

    @property
    def l2_minor(self) -> int:
        # This is the pinned SingleTileLPTBwdScheduler policy, not kernel tuning.
        one_head_bytes = self.owner_extent * (
            (self.producer_features + self.consumer_features) * 2 + self.producer_features * 4
        )
        resident_heads = max(1, (50 * 1024 * 1024) // one_head_bytes)
        return 1 << (resident_heads.bit_length() - 1)

    @property
    def head_batch_quotient(self) -> int:
        return self.heads * self.batch // self.l2_minor

    @property
    def head_batch_residual(self) -> int:
        return max(self.heads * self.batch % self.l2_minor, 1)

    def logical_tile(self, block: int) -> tuple[int, int, int]:
        """Return (streamed block, head, batch) for one admitted physical CTA."""
        if not 0 <= block < self.total_blocks:
            raise ValueError("ordered scheduler block is outside the launch grid")
        section, offset = divmod(block, self.l2_minor * self.streamed_blocks)
        width = self.l2_minor if section < self.head_batch_quotient else self.head_batch_residual
        streamed, residual = divmod(offset, width)
        batch, head = divmod(section * self.l2_minor + residual, self.heads)
        if self.descending:
            streamed = self.streamed_blocks - 1 - streamed
        return streamed, head, batch

    @property
    def scheduler_arguments(self) -> tuple[CutePipelineScalarArgument | CutePipelinePackedBytesArgument, ...]:
        """Encode fields flattened by pinned ParamsBase, never one guessed struct.

        The runtime constructs every scalar as Int32 and each fast divisor from
        Int32. ParamsBase extracts the eight dynamic fields in declaration order;
        each FastDivmod is its existing single 12-byte value. Strict CUDA argument
        validation and captured parameter comparison remain hardware gates.
        """
        return (
            CutePipelineScalarArgument("scheduler_total_blocks", CutePipelineElementType.I32, self.total_blocks),
            CutePipelineScalarArgument("scheduler_num_block", CutePipelineElementType.I32, self.streamed_blocks),
            CutePipelineScalarArgument("scheduler_l2_minor", CutePipelineElementType.I32, self.l2_minor),
            *(
                CutePipelinePackedBytesArgument(name, encode_cute_fast_divmod_divisor(divisor), 4)
                for name, divisor in (
                    ("scheduler_num_head_divmod", self.heads),
                    ("scheduler_l2_minor_divmod", self.l2_minor),
                    ("scheduler_l2_major_divmod", self.l2_minor * self.streamed_blocks),
                    ("scheduler_l2_minor_residual_divmod", self.head_batch_residual),
                )
            ),
            CutePipelineScalarArgument(
                "scheduler_num_hb_quotient", CutePipelineElementType.I32, self.head_batch_quotient
            ),
        )

    @property
    def record(self) -> dict:
        return {
            "contributor_order": "descending" if self.descending else "ascending",
            "owner_extent": self.owner_extent,
            "streamed_extent": self.streamed_extent,
            "owner_tile": self.owner_tile,
            "streamed_tile": self.streamed_tile,
            "batch": self.batch,
            "heads": self.heads,
            "counter_shape": self.counter_shape,
            "counter_bytes": self.counter_bytes,
            "scheduler": "SingleTileLPTBwdScheduler",
            "grid": self.grid,
            "l2_minor": self.l2_minor,
            "num_hb_quotient": self.head_batch_quotient,
            "num_hb_residual": self.head_batch_residual,
            "reset": "every_invocation",
            "reset_block": self.reset_block,
            "reset_grid": self.reset_grid,
        }


def validate_ordered_backward_variant(*, relation_group_size: int, layout_transport: bool, combined: bool) -> None:
    """Reject paths whose ordered launch/reset contract has not been admitted."""
    if relation_group_size != 1:
        raise ValueError("ordered grouped dK/dV accumulation is not admitted")
    if layout_transport:
        raise ValueError("ordered affine output transport is not admitted")
    if combined:
        raise ValueError("ordered combined forward/backward execution is not admitted")


def ordered_backward_launch_roles(symbols: tuple[str, ...], *, expanded_cotangent: bool) -> tuple[str, ...]:
    """Validate qualified compiler kernel roles before binding a recipe.

    The symbol grammar is that of the pinned CuTe compiler. Unknown spellings
    reject; no ordinal suffix or generic kernel count can establish a role.
    """
    prefixes = {
        "reset": "tile_lifetimebackendsh100cute_streaming_backward_emitter_Int32SemaphoreZeroFill",
        "vjp": "tile_lifetimebackendsh100cute_post_normalization_pullbackCutePostNormalizationPullback",
        "preprocess": "flash_attncuteflash_bwd_preprocessFlashAttentionBackwardPreprocess",
        "main": "tile_lifetimebackendsh100cute_normalized_exp_backward_sm90ShuttleNormalizedExponentialBackwardSm90",
        "postprocess": "flash_attncuteflash_bwd_postprocessFlashAttentionBackwardPostprocess",
    }
    roles = []
    for symbol in symbols:
        matched = tuple(
            role for role, prefix in prefixes.items() if symbol.startswith(f"kernel_cutlass_kernel_{prefix}_object_at__")
        )
        if len(matched) != 1:
            raise ValueError(f"unrecognized ordered backward kernel role: {symbol}")
        roles.append(matched[0])
    expected = (
        ("reset", "vjp", "preprocess", "vjp", "main", "postprocess")
        if expanded_cotangent
        else ("reset", "preprocess", "main", "postprocess")
    )
    if tuple(roles) != expected or len(set(symbols)) != len(symbols):
        raise ValueError(f"ordered backward kernel roles must be {expected}, got {tuple(roles)}")
    return tuple(roles)


@dataclass(frozen=True)
class Sm90BackwardKernelBinding:
    reset: str
    preprocess: str
    main: str
    postprocess: str
    prime_vjp: str | None
    vjp: str | None


def ordered_backward_kernel_binding(symbols: tuple[str, ...], *, expanded_cotangent: bool) -> Sm90BackwardKernelBinding:
    """Bind only the uniquely validated roles, including both optional VJP sites."""
    roles = ordered_backward_launch_roles(symbols, expanded_cotangent=expanded_cotangent)
    by_role = {role: symbol for role, symbol in zip(roles, symbols, strict=True)}
    maps = tuple(symbol for role, symbol in zip(roles, symbols, strict=True) if role == "vjp")
    return Sm90BackwardKernelBinding(
        by_role["reset"],
        by_role["preprocess"],
        by_role["main"],
        by_role["postprocess"],
        maps[0] if maps else None,
        maps[1] if maps else None,
    )


def validate_ordered_backward_recipe(
    plan: Sm90BackwardAccumulationPlan,
    recipe: CutePipelineLaunchRecipe,
    *,
    expanded_cotangent: bool,
) -> None:
    """Check logical launch roles, resets, and both typed dependency records.

    This does not establish compiled CUDA parameter mapping or alignment.
    """
    roles = ordered_backward_launch_roles(
        tuple(launch.kernel_symbol for launch in recipe.launches), expanded_cotangent=expanded_cotangent
    )
    by_role = {role: index for index, role in enumerate(roles)}
    main_index = by_role["main"]
    main = recipe.launches[main_index]
    reset = recipe.launches[0]
    if main.grid != plan.grid:
        raise ValueError("ordered main grid differs from its shared scheduler plan")
    actual_scheduler = tuple(argument for argument in main.arguments if argument.name.startswith("scheduler_"))
    if actual_scheduler != plan.scheduler_arguments:
        raise ValueError("ordered scheduler argument fields differ from their shared logical plan")
    counters = tuple(slot for slot in recipe.slots if slot.name == "query_accumulation_semaphore")
    if (
        len(counters) != 1
        or counters[0].shape != plan.counter_shape
        or counters[0].element_type is not CutePipelineElementType.I32
        or counters[0].kind is not CutePipelineSlotKind.SCRATCH
    ):
        raise ValueError("ordered recipe counter storage differs from its replica domain")
    if counters[0].reuse_group is not None:
        aliases = tuple(slot for slot in recipe.slots if slot.reuse_group == counters[0].reuse_group)
        if len(aliases) != 1:
            raise ValueError("ordered accumulation counter cannot alias another recipe slot")
    if reset.block != plan.reset_block or reset.dynamic_shared_memory_bytes != 0 or reset.grid != plan.reset_grid:
        raise ValueError("ordered reset grid must cover every counter replica")
    accesses = tuple(_launch_slot_accesses(launch) for launch in recipe.launches)
    if (
        accesses[0] != {"query_accumulation_semaphore": 2}
        or accesses[main_index].get("query_accumulation_semaphore") != 3
    ):
        raise ValueError("ordered semaphore must be reset then read/write by the main launch")
    if any(
        "query_accumulation_semaphore" in access for index, access in enumerate(accesses) if index not in (0, main_index)
    ):
        raise ValueError("unexpected ordered semaphore reader/writer")
    preprocess_index = by_role["preprocess"]
    if accesses[preprocess_index].get("query_accumulator") != 2 or accesses[main_index].get("query_accumulator") != 3:
        raise ValueError("ordered dQ accumulator must be reset before every main launch")
    dependencies = launch_dependency_event_schedule(recipe.launches, slots=recipe.slots)
    edges = set(dependencies["predecessor_pairs"])
    required = {
        (tuple(pair["source"]), tuple(pair["target"]))
        for pair in dependencies["realizations"][0]["event"]["required_dependence"]["relation"]["pairs"]
    }
    for edge in (((0,), (main_index,)), ((preprocess_index,), (main_index,))):
        if edge not in edges or edge not in required:
            raise ValueError("ordered reset must precede main in both typed dependence records")
