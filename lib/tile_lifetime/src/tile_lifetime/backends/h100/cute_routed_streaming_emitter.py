# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower routed Contract/Map/Fold semantics to the SM90 streaming skeleton.

The relation supplies only the sparse fold domain.  This module does not call
an attention interface: it derives compact block lists from ``RelationPlan``
and instantiates the same Shuttle-owned QK/online-state/PV skeleton used by the
dense path.  The physical skeleton performs TMA K/V copies into a circular
shared-memory pipeline before the contractions consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    normalize_block_sparse_config,
    to_cute_block_sparse_tensors,
)
from flash_attn.cute.cute_dsl_utils import to_cute_tensor

from tile_lifetime.h100_scalar_ast import normalized_finalizer_expression, score_input_roles
from tile_lifetime.h100_streaming_lowering import (
    LoweredScoreMap,
    Sm90ForwardPipelineSchedule,
    lower_h100_streaming_program,
    select_sm90_contract_schedule,
)
from tile_lifetime.normalized_exp_contract_region import ContractDomainTraversal
from tile_lifetime.routed_attention_plan import (
    RoutedAttentionOrientation,
    RoutedStreamingAttentionCompilation,
    query_major_block_index_plan,
)

from .cute_scalar_ast import cute_finalizer_map, cute_score_map
from .cute_streaming_emitter import _validate_runtime_tensor
from .cute_streaming_sm90 import ShuttleNormalizedExponentialContractSm90


@dataclass
class CompiledH100RoutedStreamingProgram:
    """One relation-specialized query-major executable."""

    compilation: RoutedStreamingAttentionCompilation
    score_map: LoweredScoreMap
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    block_sparse_tensors: BlockSparseTensorsTorch
    executable: Any

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        log_sum_exp: torch.Tensor,
    ) -> None:
        _validate_runtime_tensor("query", query, self.query_shape)
        _validate_runtime_tensor("key", key, self.key_shape)
        _validate_runtime_tensor("value", value, self.value_shape)
        _validate_runtime_tensor("output", output, (*self.query_shape[:-1], self.value_shape[-1]))
        expected_lse = (self.query_shape[0], self.query_shape[2], self.query_shape[1])
        if tuple(log_sum_exp.shape) != expected_lse or log_sum_exp.dtype is not torch.float32:
            raise ValueError(f"log_sum_exp must be FP32 with shape {expected_lse}")
        sparse_arguments = (
            self.block_sparse_tensors.mask_block_cnt,
            self.block_sparse_tensors.mask_block_idx,
            self.block_sparse_tensors.full_block_cnt,
            self.block_sparse_tensors.full_block_idx,
            self.block_sparse_tensors.cu_total_m_blocks,
            self.block_sparse_tensors.cu_block_idx_offsets,
            self.block_sparse_tensors.dq_write_order,
            self.block_sparse_tensors.dq_write_order_full,
        )
        self.executable(
            query,
            key,
            value,
            output,
            log_sum_exp,
            *([None] * 8),
            sparse_arguments,
            None,
        )


def compile_h100_routed_streaming_program(
    compilation: RoutedStreamingAttentionCompilation,
    *,
    orientation: RoutedAttentionOrientation,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    log_sum_exp: torch.Tensor,
) -> CompiledH100RoutedStreamingProgram:
    """Instantiate shared-memory streaming for one generic sparse relation."""
    if orientation is not RoutedAttentionOrientation.QUERY_MAJOR:
        raise ValueError("the first SM90 routed emitter implements only the query-major candidate")
    candidate = next((item for item in compilation.candidates if item.orientation is orientation), None)
    if candidate is None:
        raise ValueError(f"compilation does not contain the {orientation.value} candidate")

    program = compilation.program
    region = program.region
    query_shape = region.producer.inputs[0].shape
    key_shape = region.producer.inputs[1].shape
    value_shape = region.consumer.inputs[1].shape
    for name, tensor, shape in (
        ("query", query, query_shape),
        ("key", key, key_shape),
        ("value", value, value_shape),
        ("output", output, region.finalizer.output.shape),
    ):
        _validate_runtime_tensor(name, tensor, shape)

    physical_program = replace(
        program,
        schedule=Sm90ForwardPipelineSchedule(
            contract=select_sm90_contract_schedule(
                owner_extent=query_shape[1],
                streamed_extent=key_shape[1],
                producer_feature_extent=query_shape[-1],
                consumer_feature_extent=value_shape[-1],
                owner_tile=candidate.config.query_block_size,
                streamed_tile=candidate.config.key_value_block_size,
            ),
            stages=program.schedule.pipeline_depth,
        ),
    )
    lowering = lower_h100_streaming_program(physical_program)
    schedule = lowering.schedule
    if candidate.config.query_block_size != schedule.tile_m:
        raise ValueError("relation query block size does not match the streaming tile")
    if candidate.config.key_value_block_size != schedule.tile_n:
        raise ValueError("relation KV block size does not match the streaming tile")

    index_plan = query_major_block_index_plan(compilation.relation)
    block_count = torch.as_tensor(index_plan.block_count, device=query.device)[None, None]
    block_index = torch.as_tensor(index_plan.block_index, device=query.device)[None, None]
    raw_sparse_tensors = BlockSparseTensorsTorch(
        mask_block_cnt=block_count,
        mask_block_idx=block_index,
        # The generated dynamic branch must carry concrete tensors for both
        # lists even when every selected block uses the masking path.
        full_block_cnt=torch.zeros_like(block_count),
        full_block_idx=torch.zeros_like(block_index),
        block_size=(schedule.tile_m, schedule.tile_n),
    )
    sparse_tensors, _, query_subtile_factor = normalize_block_sparse_config(
        raw_sparse_tensors,
        batch_size=query_shape[0],
        num_head=query_shape[2],
        seqlen_q=query_shape[1],
        seqlen_k=key_shape[1],
        block_size=(schedule.tile_m, schedule.tile_n),
        q_stage=1,
    )
    sparse_cute_tensors = to_cute_block_sparse_tensors(sparse_tensors)
    assert sparse_cute_tensors is not None

    score_map = lowering.score_map
    score_mod = cute_score_map(
        score_map.expression,
        score_input_roles(
            score_map.expression,
            score_input=score_map.score_input,
            owner_inputs=score_map.owner_inputs,
            streamed_inputs=score_map.streamed_inputs,
        ),
    )
    finalizer = normalized_finalizer_expression(
        lowering.finalizer_expression,
        accumulator_input=program.region.consumer.output.name,
        normalizer_input=program.region.fold_state.normalizer.output.name,
    )
    skeleton = ShuttleNormalizedExponentialContractSm90(
        cutlass.BFloat16,
        query_shape[-1],
        value_shape[-1],
        lowering.relation_layout.relation_group_size,
        lower_triangular_domain=score_map.domain_traversal is ContractDomainTraversal.LOWER_TRIANGULAR,
        is_local=False,
        pack_index_relation=True,
        tile_m=schedule.tile_m,
        tile_n=schedule.tile_n,
        num_stages=schedule.stages,
        num_threads=schedule.threads,
        Q_in_regs=schedule.resident_left_in_registers,
        score_mod=score_mod,
        mask_mod=None,
        has_aux_tensors=False,
        output_scale=1.0,
        finalizer_map=cute_finalizer_map(finalizer),
        intra_wg_overlap=schedule.intra_warpgroup_overlap,
        mma_pv_is_rs=schedule.consumer_register_source,
        q_subtile_factor=query_subtile_factor,
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    cute_arguments = [to_cute_tensor(tensor) for tensor in (query, key, value, output)]
    cute_arguments.append(to_cute_tensor(log_sum_exp, assumed_align=4))
    cute_arguments.extend([None] * 8)
    cute_arguments.extend((sparse_cute_tensors, None, fake_stream))
    executable = cute.compile(skeleton, *cute_arguments, options="--enable-tvm-ffi")
    return CompiledH100RoutedStreamingProgram(
        compilation=compilation,
        score_map=score_map,
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        block_sparse_tensors=sparse_tensors,
        executable=executable,
    )
