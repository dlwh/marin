# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free legalization of streaming tensor programs for SM90."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

from shuttle.ir import DType
from tile_lifetime.autodiff import scalar_expression_vjp
from tile_lifetime.normalized_exp_contract_region import (
    ContractDomainTraversal,
    NormalizedExpContractProgram,
    NormalizedExpContractRegion,
    relation_group_size,
)
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    scalar_constant,
    scalar_expression_inputs,
)


@dataclass(frozen=True)
class Sm90RelationPackedLayout:
    """Explicit public-to-SM90 layout for one relation-grouped Contract.

    Public owner operands retain ``[B, Hshared, *G, S, D]``.  The current
    Hopper kernel consumes an equivalent ``[B, S, Hshared*prod(G), D]`` view.
    The quotient and remainder maps below make that physical flattening part
    of schedule identity instead of rediscovering it from tensor extents.
    """

    batch_shape: tuple[int, ...]
    shared_relation_axis: TensorAxis
    relation_group_axes: tuple[TensorAxis, ...]
    owner_extent: int
    streamed_extent: int
    producer_feature_extent: int
    consumer_feature_extent: int
    flattened_relation_axis: TensorAxis
    shared_relation_map: AxisIndexMap
    group_relation_maps: tuple[AxisIndexMap, ...]

    def __post_init__(self) -> None:
        extents = (
            *self.batch_shape,
            self.owner_extent,
            self.streamed_extent,
            self.producer_feature_extent,
            self.consumer_feature_extent,
        )
        if any(extent <= 0 for extent in extents):
            raise ValueError("SM90 relation-packed layout extents must be positive")
        if not self.relation_group_axes:
            raise ValueError("SM90 relation-packed layout requires explicit relation-group axes")
        if self.shared_relation_map.domain_axis != self.flattened_relation_axis:
            raise ValueError("shared relation map must consume the flattened physical relation axis")
        if self.shared_relation_map.operand_axis != self.shared_relation_axis:
            raise ValueError("shared relation map must produce the public shared relation axis")
        if tuple(mapping.operand_axis for mapping in self.group_relation_maps) != self.relation_group_axes:
            raise ValueError("group relation maps must reproduce every public relation-group axis")
        if any(mapping.domain_axis != self.flattened_relation_axis for mapping in self.group_relation_maps):
            raise ValueError("group relation maps must consume the flattened physical relation axis")

    @property
    def batch_extent(self) -> int:
        return math.prod(self.batch_shape)

    @property
    def relation_group_shape(self) -> tuple[int, ...]:
        return tuple(axis.extent for axis in self.relation_group_axes)

    @property
    def relation_group_size(self) -> int:
        return math.prod(self.relation_group_shape)

    @property
    def public_owner_shape(self) -> tuple[int, ...]:
        return (
            *self.batch_shape,
            self.shared_relation_axis.extent,
            *self.relation_group_shape,
            self.owner_extent,
            self.producer_feature_extent,
        )

    @property
    def public_streamed_producer_shape(self) -> tuple[int, ...]:
        return (
            *self.batch_shape,
            self.shared_relation_axis.extent,
            self.streamed_extent,
            self.producer_feature_extent,
        )

    @property
    def public_streamed_consumer_shape(self) -> tuple[int, ...]:
        return (
            *self.batch_shape,
            self.shared_relation_axis.extent,
            self.streamed_extent,
            self.consumer_feature_extent,
        )

    @property
    def public_output_shape(self) -> tuple[int, ...]:
        return (*self.public_owner_shape[:-1], self.consumer_feature_extent)

    def owner_physical_shape(self, feature_extent: int) -> tuple[int, int, int, int]:
        return (
            self.batch_extent,
            self.owner_extent,
            self.flattened_relation_axis.extent,
            feature_extent,
        )

    def owner_physical_strides(self, feature_extent: int) -> tuple[int, int, int, int]:
        return (
            self.flattened_relation_axis.extent * self.owner_extent * feature_extent,
            feature_extent,
            self.owner_extent * feature_extent,
            1,
        )

    def streamed_physical_shape(self, feature_extent: int) -> tuple[int, int, int, int]:
        return (
            self.batch_extent,
            self.streamed_extent,
            self.shared_relation_axis.extent,
            feature_extent,
        )

    def streamed_physical_strides(self, feature_extent: int) -> tuple[int, int, int, int]:
        return (
            self.shared_relation_axis.extent * self.streamed_extent * feature_extent,
            feature_extent,
            self.streamed_extent * feature_extent,
            1,
        )

    def owner_coordinate_shape(self, feature_extent: int) -> tuple[int, int, int, int]:
        """Return the owner tensor shape in the CuTe kernel coordinate order."""
        return self.owner_extent, feature_extent, self.flattened_relation_axis.extent, self.batch_extent

    def streamed_coordinate_shape(self, feature_extent: int) -> tuple[int, int, int, int]:
        """Return the streamed tensor shape in the CuTe kernel coordinate order."""
        return self.streamed_extent, feature_extent, self.shared_relation_axis.extent, self.batch_extent


@dataclass(frozen=True)
class Sm90PackedRowStateWriteProof:
    """Finite proof that packed owner rows cover saved Fold state exactly once."""

    owner_tile: int
    logical_row_count: int
    emitted_write_count: int
    unique_write_count: int

    def __post_init__(self) -> None:
        if self.owner_tile <= 0:
            raise ValueError("packed row-state owner tile must be positive")
        if self.emitted_write_count != self.logical_row_count:
            raise ValueError("packed row-state writer has holes or out-of-bounds writes")
        if self.unique_write_count != self.logical_row_count:
            raise ValueError("packed row-state writer overlaps logical rows")

    @property
    def canonical_record(self) -> dict[str, int]:
        return {
            "owner_tile": self.owner_tile,
            "logical_row_count": self.logical_row_count,
            "emitted_write_count": self.emitted_write_count,
            "unique_write_count": self.unique_write_count,
        }


def prove_sm90_packed_row_state_writes(
    layout: Sm90RelationPackedLayout,
    *,
    owner_tile: int,
) -> Sm90PackedRowStateWriteProof:
    """Prove the quotient/remainder row-state address map is a bijection."""
    packed_owner_extent = layout.owner_extent * layout.relation_group_size
    coordinates = []
    for batch in range(layout.batch_extent):
        for shared_relation in range(layout.shared_relation_axis.extent):
            for owner_block in range(math.ceil(packed_owner_extent / owner_tile)):
                for tile_row in range(owner_tile):
                    packed_owner = owner_block * owner_tile + tile_row
                    if packed_owner >= packed_owner_extent:
                        continue
                    owner, group_relation = divmod(packed_owner, layout.relation_group_size)
                    relation = shared_relation * layout.relation_group_size + group_relation
                    coordinates.append((batch, relation, owner))
    logical_row_count = layout.batch_extent * layout.flattened_relation_axis.extent * layout.owner_extent
    return Sm90PackedRowStateWriteProof(
        owner_tile=owner_tile,
        logical_row_count=logical_row_count,
        emitted_write_count=len(coordinates),
        unique_write_count=len(set(coordinates)),
    )


def sm90_relation_packed_layout(
    *,
    batch_shape: tuple[int, ...],
    shared_relation_extent: int,
    relation_group_shape: tuple[int, ...],
    owner_extent: int,
    streamed_extent: int,
    producer_feature_extent: int,
    consumer_feature_extent: int,
) -> Sm90RelationPackedLayout:
    """Build explicit relation flattening maps for the current SM90 schedule."""
    if not relation_group_shape:
        relation_group_shape = (1,)
    group_size = math.prod(relation_group_shape)
    flattened = TensorAxis(-1, shared_relation_extent * group_size, "physical_flattened_relation")
    shared = TensorAxis(-2, shared_relation_extent, "public_shared_relation")
    groups = tuple(
        TensorAxis(-3 - index, extent, f"public_relation_group_{index}")
        for index, extent in enumerate(relation_group_shape)
    )
    group_maps = tuple(
        AxisIndexMap(
            flattened,
            axis,
            divisor=math.prod(relation_group_shape[index + 1 :]),
            modulus=axis.extent,
        )
        for index, axis in enumerate(groups)
    )
    return Sm90RelationPackedLayout(
        batch_shape=batch_shape,
        shared_relation_axis=shared,
        relation_group_axes=groups,
        owner_extent=owner_extent,
        streamed_extent=streamed_extent,
        producer_feature_extent=producer_feature_extent,
        consumer_feature_extent=consumer_feature_extent,
        flattened_relation_axis=flattened,
        shared_relation_map=AxisIndexMap(flattened, shared, divisor=group_size),
        group_relation_maps=group_maps,
    )


def derive_sm90_relation_packed_layout(region: NormalizedExpContractRegion) -> Sm90RelationPackedLayout:
    """Derive public relation axes and the selected SM90 physical packing."""
    dimensions = region.dimensions
    owner_input, streamed_producer_input = region.producer.inputs
    streamed_consumer_input = region.consumer.inputs[1]
    if dimensions.shared_relation_axes:
        if len(dimensions.shared_relation_axes) != 1 or not dimensions.relation_group_axes:
            raise ValueError("SM90 relation packing requires one shared axis and explicit group axes")
        if len(dimensions.producer_features) != 1 or len(dimensions.normalized_axes) != 1:
            raise ValueError("SM90 relation packing requires one producer feature and one streamed axis")
        if len(dimensions.consumer_features) != 1:
            raise ValueError("SM90 relation packing requires one consumer feature axis")
        shared = dimensions.shared_relation_axes[0]
        groups = dimensions.relation_group_axes
        producer_feature = dimensions.producer_features[0]
        streamed = dimensions.normalized_axes[0]
        consumer_feature = dimensions.consumer_features[0]
        relation_start = owner_input.axes.index(shared)
        batch_axes = owner_input.axes[:relation_start]
        owner_suffix = owner_input.axes[relation_start:]
        if len(owner_suffix) != len(groups) + 3:
            raise ValueError("owner operand must have [Hshared,*G,S,D] after its batch axes")
        owner = owner_suffix[-2]
        if owner_suffix != (shared, *groups, owner, producer_feature):
            raise ValueError("owner operand axes must be [B...,Hshared,*G,S,Dproducer]")
        if streamed_producer_input.axes != (*batch_axes, shared, streamed, producer_feature):
            raise ValueError("streamed producer axes must be [B...,Hshared,T,Dproducer]")
        if streamed_consumer_input.axes != (*batch_axes, shared, streamed, consumer_feature):
            raise ValueError("streamed consumer axes must be [B...,Hshared,T,Dconsumer]")
        return sm90_relation_packed_layout(
            batch_shape=tuple(axis.extent for axis in batch_axes),
            shared_relation_extent=shared.extent,
            relation_group_shape=tuple(axis.extent for axis in groups),
            owner_extent=owner.extent,
            streamed_extent=streamed.extent,
            producer_feature_extent=producer_feature.extent,
            consumer_feature_extent=consumer_feature.extent,
        )

    producer_relations = region.producer.index_maps_for_input(1)
    consumer_relations = region.consumer.index_maps_for_input(1)
    if len(producer_relations) != 1 or consumer_relations != producer_relations:
        raise ValueError("SM90 legacy flat operands require one shared Contract index relation")
    relation = producer_relations[0]
    group_size = relation_group_size(producer_relations)
    if relation.offset != 0 or relation.modulus is not None:
        raise ValueError("SM90 relation packing supports a zero-offset floor relation")
    if relation.domain_axis.extent != relation.operand_axis.extent * group_size:
        raise ValueError("SM90 flat relation does not exactly cover the owner relation axis")
    if len(owner_input.shape) != 4 or len(streamed_producer_input.shape) != 4:
        raise ValueError("legacy flat SM90 operands must have rank four")
    return sm90_relation_packed_layout(
        batch_shape=(owner_input.shape[0],),
        shared_relation_extent=relation.operand_axis.extent,
        relation_group_shape=(group_size,),
        owner_extent=owner_input.shape[1],
        streamed_extent=streamed_producer_input.shape[1],
        producer_feature_extent=owner_input.shape[-1],
        consumer_feature_extent=streamed_consumer_input.shape[-1],
    )


def replace_contract_relation_layout(
    schedule: Sm90ForwardPipelineSchedule,
    relation_layout: Sm90RelationPackedLayout,
) -> Sm90ForwardPipelineSchedule:
    """Bind one validated relation layout to a previously selected schedule."""
    return replace(schedule, contract=replace(schedule.contract, relation_layout=relation_layout))


@dataclass(frozen=True)
class LoweredScoreMap:
    """Closed score scalar AST and its physical coordinate bindings."""

    expression: ScalarExpression
    score_input: str
    owner_inputs: tuple[str, ...]
    streamed_inputs: tuple[str, ...]
    domain_traversal: ContractDomainTraversal


@dataclass(frozen=True)
class Sm90ContractSchedule:
    """Shared SM90 Contract/Fold tile, WGMMA, and worker geometry."""

    tile_m: int
    tile_n: int
    threads: int = 384
    wgmma_instruction: str = "sm90.wgmma.mma_async"
    accumulator_dtype: DType = DType.FP32
    relation_layout: Sm90RelationPackedLayout | None = None

    def __post_init__(self) -> None:
        if self.tile_m not in (64, 128) or self.tile_n not in (64, 128):
            raise ValueError("the extracted SM90 Contract schedule supports 64/128 owner and stream tiles")
        if self.threads not in (128, 256, 384):
            raise ValueError("the extracted SM90 Contract schedule supports 128, 256, or 384 threads")

    @property
    def family(self) -> str:
        return "sm90_contract_fold_v1"


def select_sm90_contract_schedule(
    *,
    owner_extent: int,
    streamed_extent: int,
    producer_feature_extent: int,
    consumer_feature_extent: int,
    threads: int | None = None,
    owner_tile: int | None = None,
    streamed_tile: int | None = None,
    relation_layout: Sm90RelationPackedLayout | None = None,
) -> Sm90ContractSchedule:
    """Select the shared SM90 Contract/Fold schedule from physical dimensions.

    Scalar Maps are intentionally absent from this boundary. The current
    implementation admits the FA4-derived 64/128 tile family and retains
    precise failures for feature dimensions outside that physical family.
    """
    if producer_feature_extent not in (64, 128):
        raise ValueError("the current SM90 Contract/Fold family supports producer features 64 or 128")
    if consumer_feature_extent not in (64, 128):
        raise ValueError("the current SM90 Contract/Fold family supports consumer features 64 or 128")
    selected_owner_tile = owner_tile if owner_tile is not None else (128 if owner_extent >= 128 else 64)
    # The verified FA4-derived forward families both retain a 128-wide
    # streamed tile.  A short streamed domain is predicated inside that tile;
    # selecting 64 here would create a distinct pipeline/shared-memory family.
    selected_streamed_tile = streamed_tile if streamed_tile is not None else 128
    selected_threads = threads if threads is not None else 128 * (selected_owner_tile // 64 + 1)
    return Sm90ContractSchedule(
        tile_m=selected_owner_tile,
        tile_n=selected_streamed_tile,
        threads=selected_threads,
        relation_layout=relation_layout,
    )


class Sm90ContractReductionTraversal(StrEnum):
    """Physical WGMMA traversal of one Contract reduction tile."""

    FORWARD = "forward"
    REVERSE = "reverse"
    EVEN_ODD = "even_odd"


class Sm90ProducerOperandOrientation(StrEnum):
    """Physical owner/streamed orientation of the producer Contract."""

    OWNER_STREAMED = "owner_streamed"
    STREAMED_OWNER = "streamed_owner"


@dataclass(frozen=True)
class Sm90ProducerConsumerHandoffProof:
    """Coverage and resource proof for a producer-fragment handoff."""

    orientation: Sm90ProducerOperandOrientation
    logical_element_count: int
    emitted_element_count: int
    unique_element_count: int
    shared_memory_bytes: int
    requires_register_to_shared_transpose: bool
    producer_fold_rows_per_thread: int
    consumer_fold_rows_per_thread: int
    required_schedule_local_handoffs: int
    producer_fragment_axes: tuple[str, str]
    fold_fragment_axes: tuple[str, str]
    scratch_axes: tuple[str, str]
    barrier_generations: tuple[str, str]
    barrier_participants: int
    estimated_shared_storage_bytes: int | None

    def __post_init__(self) -> None:
        if self.emitted_element_count != self.logical_element_count:
            raise ValueError("producer/consumer handoff leaves logical elements uncovered")
        if self.unique_element_count != self.logical_element_count:
            raise ValueError("producer/consumer handoff aliases logical elements")

    @property
    def canonical_record(self) -> dict[str, object]:
        return {
            "orientation": self.orientation.value,
            "logical_element_count": self.logical_element_count,
            "emitted_element_count": self.emitted_element_count,
            "unique_element_count": self.unique_element_count,
            "shared_memory_bytes": self.shared_memory_bytes,
            "requires_register_to_shared_transpose": self.requires_register_to_shared_transpose,
            "producer_fold_rows_per_thread": self.producer_fold_rows_per_thread,
            "consumer_fold_rows_per_thread": self.consumer_fold_rows_per_thread,
            "required_schedule_local_handoffs": self.required_schedule_local_handoffs,
            "producer_fragment_axes": self.producer_fragment_axes,
            "fold_fragment_axes": self.fold_fragment_axes,
            "scratch_axes": self.scratch_axes,
            "barrier_generations": self.barrier_generations,
            "barrier_participants": self.barrier_participants,
            "estimated_shared_storage_bytes": self.estimated_shared_storage_bytes,
        }


def prove_sm90_producer_consumer_handoff(
    contract: Sm90ContractSchedule,
    orientation: Sm90ProducerOperandOrientation,
    *,
    stages: int = 2,
    resident_left_in_registers: bool = False,
) -> Sm90ProducerConsumerHandoffProof:
    """Prove the logical owner/streamed tile survives physical orientation."""
    logical = tuple((owner, streamed) for owner in range(contract.tile_m) for streamed in range(contract.tile_n))
    physical = (
        logical
        if orientation is Sm90ProducerOperandOrientation.OWNER_STREAMED
        else tuple((streamed, owner) for owner, streamed in logical)
    )
    recovered = (
        physical
        if orientation is Sm90ProducerOperandOrientation.OWNER_STREAMED
        else tuple((owner, streamed) for streamed, owner in physical)
    )
    relation_layout = contract.relation_layout
    estimated_shared_storage_bytes = None
    if relation_layout is not None:
        query_bytes = contract.tile_m * relation_layout.producer_feature_extent * 2
        key_bytes = contract.tile_n * relation_layout.producer_feature_extent * stages * 2
        value_bytes = contract.tile_n * relation_layout.consumer_feature_extent * stages * 2
        resident_bytes = max(query_bytes, value_bytes) if resident_left_in_registers else query_bytes + value_bytes
        scratch_bytes = (
            0 if orientation is Sm90ProducerOperandOrientation.OWNER_STREAMED else contract.tile_m * contract.tile_n * 2
        )
        pipeline_barrier_bytes = (1 + 2 * stages) * 2 * 8
        aligned_pipeline_barrier_bytes = math.ceil(pipeline_barrier_bytes / 1024) * 1024
        estimated_shared_storage_bytes = aligned_pipeline_barrier_bytes + resident_bytes + key_bytes + scratch_bytes
    return Sm90ProducerConsumerHandoffProof(
        orientation=orientation,
        logical_element_count=len(logical),
        emitted_element_count=len(recovered),
        unique_element_count=len(set(recovered)),
        shared_memory_bytes=(
            0 if orientation is Sm90ProducerOperandOrientation.OWNER_STREAMED else contract.tile_m * contract.tile_n * 2
        ),
        requires_register_to_shared_transpose=orientation is Sm90ProducerOperandOrientation.STREAMED_OWNER,
        producer_fold_rows_per_thread=(
            2 if orientation is Sm90ProducerOperandOrientation.OWNER_STREAMED else contract.tile_m // 4
        ),
        consumer_fold_rows_per_thread=2,
        required_schedule_local_handoffs=(0 if orientation is Sm90ProducerOperandOrientation.OWNER_STREAMED else 3),
        producer_fragment_axes=(
            ("owner", "streamed")
            if orientation is Sm90ProducerOperandOrientation.OWNER_STREAMED
            else ("streamed", "owner")
        ),
        fold_fragment_axes=("owner", "streamed"),
        scratch_axes=("owner", "streamed"),
        barrier_generations=("empty_before_reuse", "full_after_store"),
        barrier_participants=contract.threads - 128,
        estimated_shared_storage_bytes=estimated_shared_storage_bytes,
    )


@dataclass(frozen=True)
class Sm90ForwardPipelineSchedule:
    """Forward-only pipeline choices around a shared Contract schedule."""

    contract: Sm90ContractSchedule
    stages: int
    resident_left_in_registers: bool = False
    intra_warpgroup_overlap: bool = True
    consumer_register_source: bool = True
    producer_reduction_traversal: Sm90ContractReductionTraversal = Sm90ContractReductionTraversal.FORWARD
    producer_operand_orientation: Sm90ProducerOperandOrientation = Sm90ProducerOperandOrientation.OWNER_STREAMED

    def __post_init__(self) -> None:
        if self.stages not in (1, 2, 3):
            raise ValueError("the extracted SM90 forward pipeline supports one to three stages")
        expected_threads = 128 * (self.contract.tile_m // 64 + 1)
        if self.contract.threads != expected_threads:
            raise ValueError(
                "the SM90 forward pipeline requires one producer warpgroup plus one matrix warpgroup per 64 owner rows"
            )
        if self.producer_operand_orientation is Sm90ProducerOperandOrientation.STREAMED_OWNER:
            if self.consumer_register_source:
                raise ValueError(
                    "streamed-owner producer fragments require an explicit register-to-shared transpose "
                    "before the consumer Contract"
                )
            if self.producer_reduction_traversal is not Sm90ContractReductionTraversal.FORWARD:
                raise ValueError(
                    "the current swapped-operand WGMMA realization supports only forward reduction traversal"
                )
            if self.contract.tile_m != self.contract.tile_n:
                raise ValueError("the current streamed-owner WGMMA realization requires square owner/streamed tiles")
        proof = prove_sm90_producer_consumer_handoff(
            self.contract,
            self.producer_operand_orientation,
            stages=self.stages,
            resident_left_in_registers=self.resident_left_in_registers,
        )
        if proof.estimated_shared_storage_bytes is not None and proof.estimated_shared_storage_bytes > 227_328:
            raise ValueError(
                "streamed-owner redistribution exceeds the SM90 shared-memory boundary: "
                f"{proof.estimated_shared_storage_bytes} > 227328 bytes"
            )

    @property
    def tile_m(self) -> int:
        return self.contract.tile_m

    @property
    def tile_n(self) -> int:
        return self.contract.tile_n

    @property
    def threads(self) -> int:
        return self.contract.threads


@dataclass(frozen=True)
class Sm90BackwardAccumulatorSchedule:
    """Backward-only accumulator, operand-layout, and epilogue choices."""

    contract: Sm90ContractSchedule
    resident_left_stages: int = 2
    consumer_output_cotangent_stages: int = 2
    post_normalization_cotangent_expansion_terms: int = 2
    score_cotangent_stages: int = 2
    score_cotangent_swap_ab: bool = True
    right_gradient_swap_ab: bool = False
    left_gradient_swap_ab: bool = False
    score_atom_layout_m: int = 1
    right_gradient_atom_layout_n: int = 2
    left_atom_layout_m: int = 1
    resident_consumer_operand_in_registers: bool = False
    left_gradient_single_warpgroup: bool = False
    deterministic_accumulation: bool = False
    owner_subtile_factor: int = 1
    epilogue_tile_m: int = 64
    epilogue_threads: int = 128
    epilogue_cluster_size: int = 1
    use_padded_epilogue_offsets: bool = True
    compute_capability: int = 90
    mainloop_dynamic_shared_memory_bytes: int = 198_656
    epilogue_dynamic_shared_memory_bytes: int = 32_768

    def __post_init__(self) -> None:
        stages = (self.resident_left_stages, self.consumer_output_cotangent_stages, self.score_cotangent_stages)
        if any(stage <= 0 for stage in stages):
            raise ValueError("SM90 backward pipeline stages must be positive")
        if self.post_normalization_cotangent_expansion_terms != 2:
            raise ValueError("the current SM90 post-normalization cotangent expansion requires exactly two terms")
        if self.owner_subtile_factor <= 0:
            raise ValueError("SM90 backward owner subtile factor must be positive")
        if self.epilogue_tile_m <= 0 or self.epilogue_threads <= 0 or self.epilogue_cluster_size <= 0:
            raise ValueError("SM90 backward epilogue geometry must be positive")
        if self.compute_capability != 90:
            raise ValueError("the extracted backward schedule targets compute capability 90")
        if self.mainloop_dynamic_shared_memory_bytes <= 0 or self.epilogue_dynamic_shared_memory_bytes <= 0:
            raise ValueError("SM90 backward shared-memory reservations must be positive")

    @property
    def normalized_weight_register_source(self) -> bool:
        """Return whether the consumer-operand gradient Contract reads its weight tile from registers."""
        matrix_warpgroups = self.contract.threads // 128 - 1
        return (
            self.score_atom_layout_m == 1
            and self.right_gradient_atom_layout_n == matrix_warpgroups
            and self.score_cotangent_swap_ab
            and not self.right_gradient_swap_ab
        )


H100StreamingSchedule = Sm90ForwardPipelineSchedule


@dataclass(frozen=True)
class H100StreamingLowering:
    """CUDA-independent contract consumed by the optional CuTe emitter."""

    score_map: LoweredScoreMap
    schedule: Sm90ForwardPipelineSchedule
    relation_layout: Sm90RelationPackedLayout
    finalizer_expression: ScalarExpression


def lower_h100_streaming_program(
    program: NormalizedExpContractProgram[Sm90ForwardPipelineSchedule],
) -> H100StreamingLowering:
    """Legalize generic Contract/Map/Fold structure without importing CUDA."""
    region = program.region
    query_value, key_value = region.producer.inputs
    value_value = region.consumer.inputs[1]
    query_shape = query_value.shape
    key_shape = key_value.shape
    value_shape = value_value.shape
    if query_value.dtype.value != "bf16" or key_value.dtype.value != "bf16" or value_value.dtype.value != "bf16":
        raise ValueError("the initial SM90 skeleton accepts BF16 contraction operands")
    relation_layout = derive_sm90_relation_packed_layout(region)
    scheduled_layout = program.schedule.contract.relation_layout
    if scheduled_layout is not None and scheduled_layout != relation_layout:
        raise ValueError("the selected SM90 relation layout does not match the Contract index relations")
    if query_shape[-1] != key_shape[-1]:
        raise ValueError("the producer Contract requires equal left and right feature dimensions")
    dimensions = region.dimensions
    if len(dimensions.producer_features) != 1 or dimensions.producer_features[0].extent != query_shape[-1]:
        raise ValueError("the initial SM90 skeleton requires one explicit producer feature dimension")
    if len(dimensions.consumer_features) != 1 or dimensions.consumer_features[0].extent != value_shape[-1]:
        raise ValueError("the initial SM90 skeleton requires one explicit consumer feature dimension")
    if query_shape[-1] not in (64, 128):
        raise ValueError("the initial SM90 skeleton supports producer feature dimensions 64 or 128")
    if value_shape[-1] not in (64, 128):
        raise ValueError("the initial SM90 skeleton supports consumer feature dimensions 64 or 128")

    if region.dimensions.shared_relation_axes:
        expected_shapes = (
            relation_layout.public_owner_shape,
            relation_layout.public_streamed_producer_shape,
            relation_layout.public_streamed_consumer_shape,
        )
    else:
        expected_shapes = (
            relation_layout.owner_physical_shape(query_shape[-1]),
            relation_layout.streamed_physical_shape(key_shape[-1]),
            relation_layout.streamed_physical_shape(value_shape[-1]),
        )
    if (query_shape, key_shape, value_shape) != expected_shapes:
        raise ValueError("the SM90 relation layout does not reproduce the producer and consumer operand shapes")

    return H100StreamingLowering(
        score_map=lower_normalized_exp_score_map(region),
        schedule=replace_contract_relation_layout(program.schedule, relation_layout),
        relation_layout=relation_layout,
        finalizer_expression=region.finalizer.expression,
    )


def lower_score_map(program: NormalizedExpContractProgram[object]) -> LoweredScoreMap:
    """Lower the score AST and its domain relation without workload-name dispatch."""
    return lower_normalized_exp_score_map(program.region)


def lower_normalized_exp_score_map(region: NormalizedExpContractRegion) -> LoweredScoreMap:
    """Separate a supported index-domain predicate from the closed score AST."""
    expression, traversal, _ = _lower_normalized_exp_score_domain(region)

    values = {value.name: value for value in region.score_map.inputs}
    score_input = region.producer.output.name
    owner_inputs: list[str] = []
    streamed_inputs: list[str] = []
    for name in scalar_expression_inputs(expression) - {score_input}:
        value = values.get(name)
        if value is None:
            raise ValueError(f"score scalar input {name!r} is not a score Map operand")
        axes = set(value.axes)
        if axes and axes <= set(region.owner_axes):
            owner_inputs.append(name)
        elif axes and axes <= set(region.dimensions.normalized_axes):
            streamed_inputs.append(name)
        else:
            raise ValueError(f"the current CuTe score hook cannot bind axes {value.axes} for input {name!r}")
    return LoweredScoreMap(
        expression=expression,
        score_input=score_input,
        owner_inputs=tuple(sorted(owner_inputs)),
        streamed_inputs=tuple(sorted(streamed_inputs)),
        domain_traversal=traversal,
    )


def restrict_normalized_exp_score_vjp(
    region: NormalizedExpContractRegion,
    source_vjp_expression: ScalarExpression,
    *,
    cotangent_input: str,
) -> ScalarExpression:
    """Apply the certified score-domain assumption to an autodiff-produced VJP AST."""
    expression, _, domain_predicate = _lower_normalized_exp_score_domain(region)
    if domain_predicate is None:
        restricted = source_vjp_expression
    else:

        def restrict(node: ScalarExpression) -> ScalarExpression:
            if node == domain_predicate:
                return scalar_constant(True)
            operands = tuple(restrict(operand) for operand in node.operands)
            if (
                node.kind is ScalarExpressionKind.SELECT
                and operands[0].kind is ScalarExpressionKind.CONSTANT
                and operands[0].constant is True
            ):
                return operands[1]
            return ScalarExpression(
                kind=node.kind,
                operands=operands,
                input_name=node.input_name,
                constant=node.constant,
            )

        restricted = restrict(source_vjp_expression)

    expected = scalar_expression_vjp(
        expression,
        input_name=region.producer.output.name,
        cotangent_name=cotangent_input,
    )
    if restricted != expected:
        raise ValueError("score Map VJP does not match the certified index-domain restriction")
    return restricted


def _lower_normalized_exp_score_domain(
    region: NormalizedExpContractRegion,
) -> tuple[ScalarExpression, ContractDomainTraversal, ScalarExpression | None]:
    traversal = ContractDomainTraversal.FULL
    domain_predicate: ScalarExpression | None = None

    def remove_domain(node: ScalarExpression) -> ScalarExpression:
        nonlocal domain_predicate, traversal
        if node.kind is not ScalarExpressionKind.SELECT:
            return replace(node, operands=tuple(remove_domain(operand) for operand in node.operands))
        if traversal is not ContractDomainTraversal.FULL:
            raise ValueError("the current SM90 score lowering supports one index-domain predicate")
        predicate, selected, rejected = node.operands
        if rejected.kind is not ScalarExpressionKind.CONSTANT or rejected.constant != float("-inf"):
            raise ValueError("the SM90 score domain only supports selection of negative infinity")
        if predicate.kind is not ScalarExpressionKind.LESS_EQUAL:
            raise ValueError("the SM90 score domain only supports a lower-triangular comparison")
        left, right = predicate.operands
        if not _is_lower_triangular_predicate(region, left, right):
            raise ValueError("the SM90 lower-triangular domain requires streamed_position <= owner_position")
        traversal = ContractDomainTraversal.LOWER_TRIANGULAR
        domain_predicate = predicate
        return remove_domain(selected)

    expression = remove_domain(region.score_map.expression)
    # A typed index-domain predicate is executable scalar/index algebra, not a
    # traversal selector.  Tile-envelope optimization may be added separately,
    # but it must preserve this predicate as the source of truth and may not
    # change schedule-family selection.
    return expression, traversal, domain_predicate


def _is_lower_triangular_predicate(
    region: NormalizedExpContractRegion,
    streamed: ScalarExpression,
    owner: ScalarExpression,
) -> bool:
    streamed_name = streamed.input_name if streamed.kind is ScalarExpressionKind.INPUT else None
    owner_name = owner.input_name if owner.kind is ScalarExpressionKind.INPUT else None
    values = {value.name: value for value in region.score_map.inputs}
    streamed_value = values.get(streamed_name) if streamed_name is not None else None
    owner_value = values.get(owner_name) if owner_name is not None else None
    if streamed_value is None or owner_value is None:
        return False
    streamed_axes = set(streamed_value.axes)
    owner_axes = set(owner_value.axes)
    return (
        bool(streamed_axes)
        and streamed_axes <= set(region.dimensions.normalized_axes)
        and bool(owner_axes)
        and owner_axes <= set(region.owner_axes)
    )
