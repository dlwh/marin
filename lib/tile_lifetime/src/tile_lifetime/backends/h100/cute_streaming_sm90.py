# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# ruff: noqa
# fmt: off
# SM90 (Hopper) forward pass for flash attention, extracted from flash_fwd.py.

import math
import operator
from types import SimpleNamespace
from typing import Callable, Literal, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.block_sparse_utils import (
    produce_block_sparse_loads,
    consume_block_sparse_loads,
)
from flash_attn.cute import pipeline as pipeline_custom
# These pinned FA4 helpers retain their upstream names. Shuttle feeds them only
# relation group sizes and relation-packed layouts at this external boundary.
from flash_attn.cute.pack_gqa import PackGQA, pack_gqa_layout, make_packgqa_tiled_tma_atom
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.named_barrier import NamedBarrierFwd
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
)
from cutlass.cute import FastDivmodDivisor

from .cute_domain_restriction import DomainRestriction
from .cute_normalized_exp import NormalizedExpFoldState, apply_score_map_inner
from .cute_streaming_base import ShuttleNormalizedExponentialContractBase


@cute.jit
def _gemm_zero_init_permuted_k(
    tiled_mma,
    shape,
    tCrA,
    tCrB,
    B_idx=None,
    wg_wait=-1,
    traversal: Literal["reverse", "even_odd"] = "reverse",
):
    """Accumulate one WGMMA Contract tile in a structural K permutation."""
    acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
    rA = tCrA
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, False)
    reduction_tiles = cutlass.const_expr(cute.size(rA.shape[2]))
    for iteration in cutlass.range_constexpr(reduction_tiles):
        if const_expr(traversal == "reverse"):
            k = reduction_tiles - 1 - iteration
        elif const_expr(traversal == "even_odd"):
            even_tiles = (reduction_tiles + 1) // 2
            k = iteration * 2 if const_expr(iteration < even_tiles) else (iteration - even_tiles) * 2 + 1
        cute.gemm(mma_atom, acc, rA[None, None, k], rB[None, None, k], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    if const_expr(wg_wait >= 0):
        warpgroup.wait_group(wg_wait)
    return acc


class ShuttleNormalizedExponentialContractSm90(ShuttleNormalizedExponentialContractBase):
    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        query_pipeline_stages: int = 1,
        key_pipeline_stages: int | None = None,
        value_pipeline_stages: int | None = None,
        pipeline_barriers_per_stage: int = 2,
        transfer_warps: int = 1,
        matrix_warpgroups: int | None = None,
        scheduler_arrival_threads: int | None = None,
        query_transaction_bytes: int | None = None,
        key_transaction_bytes: int | None = None,
        value_transaction_bytes: int | None = None,
        finalizer_map: Callable | None = None,
        conversion_faithful_normalized_map: Callable | None = None,
        conversion_faithful_score_map: Callable | None = None,
        conversion_faithful_statistics_score_map: Callable | None = None,
        conversion_faithful_statistics_sum_map: Callable | None = None,
        conversion_faithful_statistics_lse_map: Callable | None = None,
        qk_reduction_traversal: Literal["forward", "reverse", "even_odd"] = "forward",
        qk_operand_orientation: Literal["owner_streamed", "streamed_owner"] = "owner_streamed",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs
        self.buffer_align_bytes = 1024
        self.query_pipeline_stages = query_pipeline_stages
        self.key_pipeline_stages = self.num_stages if key_pipeline_stages is None else key_pipeline_stages
        self.value_pipeline_stages = self.num_stages if value_pipeline_stages is None else value_pipeline_stages
        self.pipeline_barriers_per_stage = pipeline_barriers_per_stage
        self.event_transfer_warps = transfer_warps
        self.event_matrix_warpgroups = self.tile_m // 64 if matrix_warpgroups is None else matrix_warpgroups
        self.scheduler_arrival_threads = (
            2 * 128 if self.event_matrix_warpgroups > 1 else 0
        ) if scheduler_arrival_threads is None else scheduler_arrival_threads
        self.event_transaction_bytes = {
            "Q": query_transaction_bytes,
            "K": key_transaction_bytes,
            "V": value_transaction_bytes,
        }
        self.finalizer_map = finalizer_map
        self.conversion_faithful_normalized_map = conversion_faithful_normalized_map
        self.conversion_faithful_score_map = conversion_faithful_score_map
        statistics_hooks = (
            conversion_faithful_statistics_score_map,
            conversion_faithful_statistics_sum_map,
            conversion_faithful_statistics_lse_map,
        )
        assert all(hook is None for hook in statistics_hooks) or all(
            hook is not None for hook in statistics_hooks
        ), "conversion-faithful statistics scalar hooks must be supplied together"
        self.conversion_faithful_statistics_score_map = conversion_faithful_statistics_score_map
        self.conversion_faithful_statistics_sum_map = conversion_faithful_statistics_sum_map
        self.conversion_faithful_statistics_lse_map = conversion_faithful_statistics_lse_map
        self.conversion_faithful_statistics_only = conversion_faithful_statistics_sum_map is not None
        if qk_reduction_traversal not in ("forward", "reverse", "even_odd"):
            raise ValueError(f"unsupported QK reduction traversal {qk_reduction_traversal!r}")
        self.qk_reduction_traversal = qk_reduction_traversal
        if qk_operand_orientation not in ("owner_streamed", "streamed_owner"):
            raise ValueError(f"unsupported QK operand orientation {qk_operand_orientation!r}")
        self.qk_swap_ab = qk_operand_orientation == "streamed_owner"
        if self.qk_swap_ab:
            assert self.tile_m == self.tile_n, "streamed-owner QK currently requires square tiles"
            assert not self.mma_pv_is_rs, "streamed-owner QK requires shared-memory consumer handoff"
            assert qk_reduction_traversal == "forward", (
                "streamed-owner QK currently supports only forward reduction traversal"
            )
            assert self.conversion_faithful_statistics_only or self.conversion_faithful_normalized_map is not None, (
                "streamed-owner QK currently requires the explicit two-pass normalized-exp schedule"
            )
        self.qk_gemm = (
            sm90_utils.gemm_zero_init
            if qk_reduction_traversal == "forward"
            else partial(_gemm_zero_init_permuted_k, traversal=qk_reduction_traversal)
        )
        assert self.query_pipeline_stages == 1
        assert self.key_pipeline_stages == self.num_stages
        assert self.value_pipeline_stages == self.num_stages
        assert self.pipeline_barriers_per_stage == 2
        assert self.event_transfer_warps == 1
        self.use_tma_KV = not paged_kv_non_tma
        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )
        self.cluster_shape_mn = (1, 1)
        assert self.arch.is_family_of(Arch.sm_90a), "Only SM 9.x is supported"

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype, self.tile_n
                ),
                self.dtype,
            )
        else:
            sP_layout_atom = None
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM
            if self.mma_pv_is_rs
            else warpgroup.OperandSource.SMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        # Event Tensor lowering assigns one full and one empty barrier per
        # bounded-buffer stage. Barrier IDs remain a backend allocation choice.
        mbar_ptr_Q_struct = cute.struct.MemRange[
            cutlass.Int64, self.query_pipeline_stages * self.pipeline_barriers_per_stage
        ]
        mbar_ptr_K_struct = cute.struct.MemRange[
            cutlass.Int64, self.key_pipeline_stages * self.pipeline_barriers_per_stage
        ]
        mbar_ptr_V_struct = cute.struct.MemRange[
            cutlass.Int64, self.value_pipeline_stages * self.pipeline_barriers_per_stage
        ]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        mNormalizedO: Optional[cute.Tensor] = None,
        mRowMax: Optional[cute.Tensor] = None,
        mRowSum: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """

        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
            )
        )
        if const_expr(mNormalizedO is not None and mNormalizedO.element_type != Float32):
            raise TypeError("normalized Contract output scratch must be Float32")
        if const_expr((mRowMax is None) != (mRowSum is None)):
            raise TypeError("conversion-faithful row maximum and sum must be supplied together")

        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        if const_expr(mNormalizedO is not None):
            mNormalizedO = assume_tensor_aligned(mNormalizedO)
        if const_expr(mRowMax is not None):
            mRowMax, mRowSum = (assume_tensor_aligned(mRowMax), assume_tensor_aligned(mRowSum))
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
        if const_expr(mNormalizedO is not None):
            mNormalizedO = layout_utils.select(mNormalizedO, QO_layout_transpose)
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE = (
            layout_utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )
        mRowMax = (
            layout_utils.select(mRowMax, LSE_layout_transpose)
            if const_expr(mRowMax is not None)
            else None
        )
        mRowSum = (
            layout_utils.select(mRowSum, LSE_layout_transpose)
            if const_expr(mRowSum is not None)
            else None
        )

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2, 3]
        assert self.num_wg_mma == self.event_matrix_warpgroups
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        assert self.num_producer_threads == self.event_transfer_warps * cute.arch.WARP_SIZE
        expected_scheduler_arrivals = (
            2 * self.num_threads_per_warp_group if self.num_wg_mma > 1 else 0
        )
        assert self.scheduler_arrival_threads == expected_scheduler_arrivals
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[
            self.num_wg_mma
        ]
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if const_expr(self.conversion_faithful_statistics_only):
            assert self.use_tma_KV, "conversion-faithful statistics currently requires TMA K loads"
            assert not self.use_block_sparsity, (
                "conversion-faithful statistics currently requires a dense streamed domain"
            )
            assert mRowMax is not None and mRowSum is not None and mLSE is not None, (
                "conversion-faithful statistics requires maximum, sum, and log-normalizer outputs"
            )

        self.use_scheduler_barrier = False if const_expr(self.conversion_faithful_statistics_only) else (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (
            self.pack_index_relation and self.tile_m % self.relation_group_size != 0
        )
        self.use_tma_O = self.use_tma_Q
        # Producer needs more registers when doing cp.async Q or KV loads
        if const_expr(self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)):
            self.num_mma_regs, self.num_producer_regs = 224, 40
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
        self._setup_attributes()
        # TODO: we prob don't need most of what's in _setup_attributes
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
            for mX, shape, stage in [
                (mQ, (self.tile_m, self.tile_hdim), None),
                (mK, (self.tile_n, self.tile_hdim), self.num_stages),
                (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
                (mO, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(
                mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )

        SharedStorage = self._get_shared_storage_cls()
        self.dynamic_shared_memory_bytes = SharedStorage.size_in_bytes()

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_index_relation):
            shared_relation_extent = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.relation_group_size, shared_relation_extent, head_idx=2)
            mO = pack_gqa_layout(mO, self.relation_group_size, shared_relation_extent, head_idx=2)
            if const_expr(mNormalizedO is not None):
                mNormalizedO = pack_gqa_layout(
                    mNormalizedO,
                    self.relation_group_size,
                    shared_relation_extent,
                    head_idx=2,
                )
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.relation_group_size, shared_relation_extent, head_idx=1)

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        for name, expected_bytes in self.event_transaction_bytes.items():
            if const_expr(expected_bytes is not None):
                assert self.tma_copy_bytes[name] == expected_bytes
        make_tiled_tma_atom_fn = (
            partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.relation_group_size, head_idx=2)
            if const_expr(self.pack_index_relation)
            else cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_index_relation) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),  # No mcast
            )
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        if const_expr(self.use_tma_KV):
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,  # No mcast for now
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,  # No mcast for now
            )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_index_relation) else mO
            if const_expr(self.varlen_q):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),  # No mcast
            )
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler
                if const_expr(not self.lower_triangular_domain or self.is_local)
                else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            1,  # num_splits
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.relation_group_size if const_expr(self.pack_index_relation) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.lower_triangular_domain or self.is_local,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        # The scalar score Map has already applied every score transformation.
        # This is only the base-e to base-2 conversion used by the Fold's exp2.
        softmax_scale_log2 = Float32(math.log2(math.e))
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.relation_group_size, self.pack_index_relation, aux_tensors, mPageTable
        )

        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mNormalizedO,
            mLSE,
            mRowMax,
            mRowSum,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            aux_tensors,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mNormalizedO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mRowMax: Optional[cute.Tensor],
        mRowSum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_tensors=Optional[list[cute.Tensor]],
        fastdiv_mods=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)
            if const_expr(not self.conversion_faithful_statistics_only):
                for tma_atom in (tma_atom_V, tma_atom_O):
                    if const_expr(tma_atom is not None):
                        cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier / pipeline init
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=self.query_pipeline_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=self.query_pipeline_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        if const_expr(self.use_tma_KV):
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.key_pipeline_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.value_pipeline_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )
        else:
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.key_pipeline_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
            pipeline_v = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.value_pipeline_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        # reuse sQ's data iterator
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.lower_triangular_domain,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.relation_group_size if const_expr(self.pack_index_relation) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_index_relation) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks if blocksparse_tensors is not None else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if blocksparse_tensors is not None
                else None
            ),
            # Don't need to pass in tile_mn because we won't access offset_padded
        )
        DomainRestrictionCls = partial(
            DomainRestriction,
            self.tile_m,
            self.tile_n,
            window_left=window_size_left,
            window_right=window_size_right,
            head_group_size=self.relation_group_size if const_expr(self.pack_index_relation) else 1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Cluster wait before starting
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_Q,
                mPageTable,
                blocksparse_tensors,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mNormalizedO,
                mLSE,
                mRowMax,
                mRowSum,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                DomainRestrictionCls,
                TileSchedulerCls,
                blocksparse_tensors,
                aux_tensors,
                fastdiv_mods,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()

        # TMA: only warp 0 loads. cp_async: all warps load.
        # When not use_tma_Q, all 128 producer threads participate in Q loading.
        is_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV or not self.use_tma_Q)
        # KV loading restricted to warp 0 for TMA, all warps for non-TMA KV
        is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)

        if is_load_warp:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                streamed_relation_idx = (
                    head_idx // self.relation_group_size if const_expr(not self.pack_index_relation) else head_idx
                )

                load_Q = None
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                    )

                paged_kv_manager = None
                tma_load_K_fn = None
                tma_load_V_fn = None
                if const_expr(self.use_tma_KV):
                    # === TMA path (non-paged and paged with page_size == n_block_size) ===
                    if const_expr(mPageTable is not None):
                        # Paged TMA: keep page dimension indexable
                        mK_cur = mK[None, None, streamed_relation_idx, None]
                        mV_cur = mV[None, None, streamed_relation_idx, None]
                        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (0, 0, None))
                        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (0, 0, None))
                    else:
                        # Non-paged TMA
                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                            None, None, streamed_relation_idx
                        ]
                        mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                            None, None, streamed_relation_idx
                        ]
                        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
                    # TODO: mcast
                    tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K, 0, cute.make_layout(1), gK, sK
                    )
                    tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)
                    tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V, 0, cute.make_layout(1), gV, sV
                    )
                    tma_load_V_fn = copy_utils.tma_producer_copy_fn(tma_load_V_fn, pipeline_v)
                else:
                    # === cp_async path (paged KV with page_size != n_block_size) ===
                    paged_kv_manager = PagedKVManager.create(
                        mPageTable,
                        mK,
                        mV,
                        FastDivmodDivisor(mK.shape[0]),
                        batch_idx,
                        streamed_relation_idx,
                        tidx,
                        seqlen.seqlen_k,
                        0,  # leftpad_k
                        self.tile_n,
                        self.tile_hdim,
                        self.tile_hdimv,
                        self.num_threads_per_warp_group,
                        mK.element_type,
                        arch=self.arch.major * 10 + self.arch.minor,
                    )

                load_K = partial(
                    self.load_KV,
                    tma_load_K_fn,
                    paged_kv_manager,
                    sK,
                    pipeline_kv=pipeline_k,
                    K_or_V="K",
                )
                load_V = partial(
                    self.load_KV,
                    tma_load_V_fn,
                    paged_kv_manager,
                    sV,
                    pipeline_kv=pipeline_v,
                    K_or_V="V",
                )

                relation_packer = None
                if const_expr(not self.use_tma_Q):
                    relation_packer = PackGQA(
                        self.tile_m, self.tile_hdim, self.check_hdim_oob, self.relation_group_size
                    )

                if const_expr(not self.use_block_sparsity):
                    n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
                    # if cute.arch.thread_idx()[0] == 0:
                    #     cute.printf("m_block = %d, n_block_min: %d, n_block_max: %d", m_block, n_block_min, n_block_max)
                    # Clamp n_block to 0 when n_block_max == 0 (can happen with causal
                    # + packed relation load when the streamed extent is smaller than tile_n). TMA handles n_block=-1
                    # gracefully (fills zeros), but cp.async would crash on
                    # out-of-bounds page table access.
                    n_block = (
                        n_block_max - 1
                        if const_expr(self.use_tma_KV)
                        else cutlass.max(n_block_max - 1, 0)
                    )
                    page_idx = (
                        mPageTable[batch_idx, n_block]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )

                    # First iteration: load K on pipeline_k, Q on pipeline_q
                    if is_kv_load_warp:
                        pipeline_k.producer_acquire(kv_producer_state)
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                        load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        relation_packer.load_Q(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1

                    if is_kv_load_warp:
                        if const_expr(self.conversion_faithful_statistics_only):
                            # The resident Q tile remains live while K is streamed twice:
                            # first for the global maximum, then for the exact typed sum.
                            kv_producer_state.advance()
                            for i in cutlass.range(n_block_max - n_block_min - 1, unroll=1):
                                n_block = n_block_max - 2 - i
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                kv_producer_state.advance()
                            for i in cutlass.range(n_block_max - n_block_min, unroll=1):
                                n_block = n_block_max - 1 - i
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                kv_producer_state.advance()
                        elif const_expr(not self.intra_wg_overlap or not self.use_tma_KV):
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                            kv_producer_state.advance()
                            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                                n_block = n_block_max - 1 - i - 1
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(mPageTable is not None and self.use_tma_KV)
                                    else None
                                )
                                if const_expr(not self.use_tma_KV):
                                    paged_kv_manager.load_page_table(n_block)
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                pipeline_v.producer_acquire(kv_producer_state)
                                load_V(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                kv_producer_state.advance()
                        else:
                            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                                n_block_prev = n_block_max - i - 1
                                n_block = n_block_prev - 1
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                page_idx_prev = (
                                    mPageTable[batch_idx, n_block_prev]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                kv_producer_state_prev = kv_producer_state.clone()
                                kv_producer_state.advance()
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                pipeline_v.producer_acquire(kv_producer_state_prev)
                                load_V(
                                    block=n_block_prev,
                                    producer_state=kv_producer_state_prev,
                                    page_idx=page_idx_prev,
                                )
                            n_block = n_block_min
                            page_idx = (
                                mPageTable[batch_idx, n_block]
                                if const_expr(mPageTable is not None)
                                else None
                            )
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                            kv_producer_state.advance()
                else:
                    # Block sparsity: use TMA closures directly (not paged)
                    # Load Q on pipeline_q, separate from K/V pipeline
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        relation_packer.load_Q(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1
                    if is_kv_load_warp:
                        kv_producer_state = produce_block_sparse_loads(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            m_block,
                            seqlen,
                            kv_producer_state,
                            tma_load_K_fn,
                            tma_load_V_fn,
                            pipeline_k,
                            pipeline_v,
                            self.intra_wg_overlap,
                            self.relation_group_size if const_expr(self.pack_index_relation) else 1,
                            self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                        )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            # Producer tail is only useful for cluster to avoid early exit of blocks.
            # We only need producer_tail on V since that's the last that's loaded, we don't
            # need it for Q (no cluster) and K.
            if is_kv_load_warp:
                if const_expr(self.conversion_faithful_statistics_only):
                    pipeline_k.producer_tail(kv_producer_state)
                else:
                    pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def load_KV(
        self,
        tma_load_fn: Optional[Callable],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
    ):
        if const_expr(self.use_tma_KV):
            src_idx = block if const_expr(page_idx is None) else page_idx
            tma_load_fn(src_idx=src_idx, producer_state=producer_state)
        else:
            paged_kv_manager.load_KV(block, sX[None, None, producer_state.index], K_or_V)
            cute.arch.cp_async_commit_group()
        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mNormalizedO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mRowMax: Optional[cute.Tensor],
        mRowSum: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        DomainRestrictionCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        aux_tensors: Optional[list],
        fastdiv_mods=None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk,
            (self.tile_m, self.tile_n, self.tile_hdim),
            sQ,
            sK,
            swap_AB=self.qk_swap_ab,
        )
        mma_qk_fn = (
            partial(
                self.qk_gemm,
                tiled_mma_qk,
                (self.tile_m, self.tile_n),
                tSrQ,
                tSrK,
                swap_AB=True,
            )
            if const_expr(self.qk_swap_ab)
            else partial(self.qk_gemm, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK)
        )
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        copy_P_r2s = None
        copy_P_s2r = None
        smem_thr_copy_P = None
        tPsP = None
        if const_expr(sP is not None and self.qk_swap_ab):
            copy_P_r2s, smem_thr_copy_P, tPsP = copy_utils.get_smem_store_C(
                tiled_mma_qk,
                layout_utils.transpose_view(sP),
                tidx,
                transpose=True,
                position_independent=True,
                major_mode_size=self.tile_n // self.num_wg_mma,
            )
            copy_P_s2r, _, _ = copy_utils.get_smem_load_C(
                tiled_mma_qk,
                sP,
                tidx,
                position_independent=True,
            )
        elif const_expr(sP is not None):
            smem_copy_atom_P = utils.get_smem_store_atom(
                self.arch.major * 10 + self.arch.minor, self.dtype
            )
            smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
            tPsP = smem_thr_copy_P.partition_D(sP)
        smem_copy_params = SimpleNamespace(
            copy_P_r2s=copy_P_r2s,
            copy_P_s2r=copy_P_s2r,
            smem_thr_copy_P=smem_thr_copy_P,
            tPsP=tPsP,
        )

        self.mma_init()

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        softmax = NormalizedExpFoldState.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
        )

        if const_expr(self.conversion_faithful_statistics_only):
            self.mma_conversion_faithful_statistics_tiles(
                mma_qk_fn,
                pipeline_k,
                pipeline_q,
                q_consumer_phase,
                kv_consumer_state,
                softmax.row_max,
                softmax.row_sum,
                mRowMax,
                mRowSum,
                mLSE,
                tiled_mma_qk,
                thr_mma_qk,
                tidx,
                block_info,
                SeqlenInfoCls,
                DomainRestrictionCls,
                TileSchedulerCls,
                aux_tensors,
                fastdiv_mods,
                smem_copy_params,
            )
            return

        # For RescaleOBeforeGemm: persistent scores_scale across iterations
        scores_scale = None
        if const_expr(self.rescale_O_before_gemm):
            scores_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)

        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap
            if const_expr(self.intra_wg_overlap)
            else self.mma_one_n_block,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            check_inf=True,
            scores_scale=scores_scale,
        )

        process_first_half_block = partial(
            self.first_half_block_overlap,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )
        process_last_half_block = partial(
            self.last_half_block_overlap,
            pipeline_v=pipeline_v,
            mma_pv_fn=mma_pv_fn,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )
        while work_tile.is_valid_tile:
            # if work_tile.is_valid_tile:

            # shape: (atom_v_m * rest_m)
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            # Recompute fastdiv_mods if necessary for varlen with aux_tensors
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )
            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen.seqlen_k),
                )

            mask = DomainRestrictionCls(domain_info=seqlen)
            mask_fn = partial(
                mask.apply,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.lower_triangular_domain,
                mask_local=self.is_local,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
            )
            score_map_fn = None
            if const_expr(self.conversion_faithful_score_map is not None):
                score_map_fn = self.apply_conversion_faithful_score_map
            elif const_expr(self.score_map is not None):
                score_map_fn = partial(
                    self.apply_score_map,
                    thr_mma_qk,
                    batch_idx,
                    head_idx,
                    m_block,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=fastdiv_mods,
                )
            normalized_map_fn = None
            if const_expr(self.conversion_faithful_normalized_map is not None):
                normalized_map_fn = partial(
                    self.apply_conversion_faithful_normalized_map,
                    thr_mma_qk,
                    batch_idx,
                    head_idx,
                    m_block,
                    mRowMax,
                    mRowSum,
                )
            mma_one_n_block = partial(
                mma_one_n_block_all,
                seqlen=seqlen,
                softmax=softmax,
                score_map_fn=score_map_fn,
                normalized_map_fn=normalized_map_fn,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of tile_n.
            # We also need masking on S if it's causal, for the last several blocks.
            # softmax.reset()  # Don't need reset as we explicitly call softmax w is_first=True
            O_should_accumulate = False

            # ==========================================
            # MAINLOOP
            # ==========================================
            if const_expr(not self.use_block_sparsity):
                # ==========================================
                # No block-sparsity (original path)
                # ==========================================
                # First iteration with seqlen masking
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_first_half_block(
                        n_block=n_block_max - 1,
                        seqlen=seqlen,
                        kv_consumer_state=kv_consumer_state,
                        mask_fn=partial(mask_fn, mask_mod=self.domain_predicate),
                        score_mod_fn=score_map_fn,
                        is_first_block=True,
                    )
                else:
                    self.warp_scheduler_barrier_sync()
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                        is_first_n_block=True,
                        mask_fn=partial(mask_fn, mask_mod=self.domain_predicate, mask_seqlen=True),
                    )
                    O_should_accumulate = True
                # if cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block_max = {}, n_block_min = {}", m_block, n_block_max, n_block_min)
                n_block_max -= 1
                # Next couple of iterations with causal masking
                if const_expr(self.lower_triangular_domain or self.is_local):
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_causal_local_mask = {}", n_block_min_causal_local_mask)
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_causal_local_mask, unroll=1
                    ):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                            mask_fn=partial(mask_fn, mask_mod=self.domain_predicate, mask_seqlen=False),
                        )
                        O_should_accumulate = True
                    n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                # The remaining iterations have no masking
                n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                    seqlen, m_block, n_block_min
                )
                # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_before_local_mask = {}, n_block_min = {}", n_block_min_before_local_mask, n_block_min)
                for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1 - n_tile,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_mod=self.domain_predicate, mask_seqlen=False),
                    )
                    O_should_accumulate = True
                # Separate iterations with local masking on the left
                if const_expr(self.is_local and block_info.window_size_left is not None):
                    n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                    for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                            mask_fn=partial(mask_fn, mask_mod=self.domain_predicate, mask_seqlen=False),
                        )
                        O_should_accumulate = True
                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)
                # Last "half" iteration
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_last_half_block(
                        kv_consumer_state=kv_consumer_state,
                        zero_init=not O_should_accumulate,
                    )
                    O_should_accumulate = True
                else:
                    self.warp_scheduler_barrier_arrive()

            else:
                # ==========================================
                # Block sparsity
                # ==========================================
                kv_consumer_state, O_should_accumulate, processed_any = consume_block_sparse_loads(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    kv_consumer_state,
                    mma_pv_fn,
                    mma_one_n_block,
                    process_first_half_block,
                    process_last_half_block,
                    mask_fn,
                    score_map_fn,
                    O_should_accumulate,
                    self.domain_predicate,
                    fastdiv_mods,
                    self.intra_wg_overlap,
                    self.warp_scheduler_barrier_sync,
                    self.warp_scheduler_barrier_arrive,
                    self.relation_group_size if const_expr(self.pack_index_relation) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )

                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)

                # Handle empty case (when no blocks to process)
                if not processed_any:
                    softmax.reset()
                    acc_O.fill(0.0)

            q_consumer_phase ^= 1

            sink_val = None
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_index_relation):
                    sink_val = Float32(learnable_sink[head_idx])
                else:  # Relation-packed rows can address different sink values within one thread.
                    sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.tile_m + tScS_mn[r][0]
                        owner_relation_idx = row % self.relation_group_size + head_idx * self.relation_group_size
                        sink_val[r] = Float32(learnable_sink[owner_relation_idx])

            # normalize acc_O by row_sum and calculate the lse
            if const_expr(mRowMax is not None and self.conversion_faithful_normalized_map is None):
                self.store_conversion_faithful_row_state(
                    mRowMax,
                    mRowSum,
                    softmax.row_max,
                    softmax.row_sum,
                    tiled_mma_pv,
                    tidx,
                    m_block,
                    head_idx,
                    batch_idx,
                    seqlen,
                )
            if const_expr(self.conversion_faithful_normalized_map is None):
                row_scale = softmax.finalize(output_scale=self.output_scale, extra_logit=sink_val)
                softmax.rescale_weighted_accumulator(acc_O, row_scale)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mNormalizedO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: NormalizedExpFoldState,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        score_mod_fn: Optional[Callable] = None,
        is_first_block: bool = False,
    ):
        """Processes the first half block when using intra-warpgroup-overlap"""

        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        # Apply score modification if present
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)

        # Apply mask; mask_seqlen always True for first block
        # Caveat: if full block further right than mask block, seqlen masking is redundant;
        # however, masking is being applied anyway, so essentially no perf hit
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True, transpose=self.qk_swap_ab)

        row_scale = softmax.update(acc_S, is_first=is_first_block)

        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        tOrP_cur.store(tOrP_acc.load().to(self.dtype))

        if const_expr(not self.mma_pv_is_rs):
            if const_expr(self.qk_swap_ab):
                self.store_qk_fragment_for_consumer(tOrP_cur, smem_copy_params)
            else:
                tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
                cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
                # Fence and barrier to make smem store visible to WGMMA
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()

        # For RescaleOBeforeGemm: initialize acc_O
        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())

        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[NormalizedExpFoldState] = None,
        acc_O: Optional[cute.Tensor] = None,
    ):
        """Processes the final PV GEMM when using intra-warpgroup-overlap"""

        # For RescaleOBeforeGemm: rescale O before the final PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_weighted_accumulator(acc_O, scores_scale)

        pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: NormalizedExpFoldState,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,  # not used
        score_map_fn: Optional[Callable] = None,
        normalized_map_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle score mods and masking
        if const_expr(score_map_fn is not None):
            score_map_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block, transpose=self.qk_swap_ab)
        if const_expr(normalized_map_fn is not None):
            normalized_map_fn(acc_S, n_block=n_block, seqlen=seqlen)
        else:
            row_scale = softmax.update(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            if const_expr(self.qk_swap_ab):
                self.store_qk_fragment_for_consumer(tOrP_cur, smem_copy_params)
            else:
                tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
                cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(normalized_map_fn is None):
            softmax.rescale_weighted_accumulator(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs and not self.qk_swap_ab):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_conversion_faithful_statistics_tiles(
        self,
        mma_qk_fn,
        pipeline_k,
        pipeline_q,
        q_consumer_phase,
        kv_consumer_state,
        row_max,
        row_sum,
        mRowMax,
        mRowSum,
        mLSE,
        tiled_mma_qk,
        thr_mma_qk,
        tidx,
        block_info,
        SeqlenInfoCls,
        DomainRestrictionCls,
        TileSchedulerCls,
        aux_tensors,
        fastdiv_mods,
        smem_copy_params,
    ):
        """Execute the two exact statistics traversals for every owned tile."""
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )
            tile_fastdiv_mods = fastdiv_mods
            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                tile_fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen.seqlen_k),
                )
            mask = DomainRestrictionCls(domain_info=seqlen)
            mask_fn = partial(
                mask.apply,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.lower_triangular_domain,
                mask_local=self.is_local,
                aux_tensors=aux_tensors,
                fastdiv_mods=tile_fastdiv_mods,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            self.warp_scheduler_barrier_sync()
            kv_consumer_state = self.mma_conversion_faithful_statistics(
                mma_qk_fn,
                pipeline_k,
                kv_consumer_state,
                mask_fn,
                seqlen,
                n_block_min,
                n_block_max,
                row_max,
                row_sum,
                smem_copy_params,
            )
            pipeline_q.consumer_release_w_index(0)
            q_consumer_phase ^= 1
            self.store_conversion_faithful_row_state(
                mRowMax,
                mRowSum,
                row_max,
                row_sum,
                tiled_mma_qk,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                seqlen,
                mLSE=mLSE,
                lse_map=self.conversion_faithful_statistics_lse_map,
                coordinate_column_extent=self.tile_n,
            )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma_conversion_faithful_statistics(
        self,
        mma_qk_fn,
        pipeline_k,
        consumer_state,
        mask_fn,
        seqlen,
        n_block_min,
        n_block_max,
        row_max,
        row_sum,
        smem_copy_params,
    ):
        """Run exact max and typed-sum sweeps over tiled producer Contracts."""
        row_max.fill(-Float32.inf)
        row_sum.fill(0.0)
        for sweep in cutlass.range_constexpr(2):
            consumer_state = self.mma_conversion_faithful_statistics_n_block(
                mma_qk_fn,
                pipeline_k,
                consumer_state,
                mask_fn,
                seqlen,
                n_block_max - 1,
                row_max,
                row_sum,
                sweep=sweep,
                mask_seqlen=True,
                smem_copy_params=smem_copy_params,
            )
            for tile in cutlass.range(n_block_max - n_block_min - 1, unroll=1):
                consumer_state = self.mma_conversion_faithful_statistics_n_block(
                    mma_qk_fn,
                    pipeline_k,
                    consumer_state,
                    mask_fn,
                    seqlen,
                    n_block_max - 2 - tile,
                    row_max,
                    row_sum,
                    sweep=sweep,
                    mask_seqlen=False,
                    smem_copy_params=smem_copy_params,
                )
            if const_expr(sweep == 0):
                for row in cutlass.range(cute.size(row_max), unroll_full=True):
                    row_max[row] = Float32(row_max[row].to(self.dtype))
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        return consumer_state

    @cute.jit
    def mma_conversion_faithful_statistics_n_block(
        self,
        mma_qk_fn,
        pipeline_k,
        consumer_state,
        mask_fn,
        seqlen,
        n_block,
        row_max,
        row_sum,
        sweep: cutlass.Constexpr[int],
        mask_seqlen: cutlass.Constexpr[bool],
        smem_copy_params,
    ):
        pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
        acc_S = mma_qk_fn(B_idx=consumer_state.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(consumer_state)
        consumer_state.advance()

        for offset in cutlass.range(cute.size(acc_S), unroll_full=True):
            acc_S[offset] = self.conversion_faithful_score_map(acc_S[offset])
        mask_fn(
            acc_S=acc_S,
            n_block=n_block,
            mask_mod=self.domain_predicate,
            mask_seqlen=mask_seqlen,
            transpose=self.qk_swap_ab,
        )
        if const_expr(self.qk_swap_ab):
            acc_S = self.redistribute_qk_fragment_for_fold(acc_S, smem_copy_params)

        values = layout_utils.reshape_acc_to_mn(acc_S)
        if const_expr(sweep == 0):
            for row in cutlass.range(cute.size(row_max), unroll_full=True):
                maximum = utils.fmax_reduce(
                    values[row, None].load(),
                    init_val=row_max[row],
                    arch=self.arch.major * 10 + self.arch.minor,
                )
                row_max[row] = cute.arch.warp_reduction_max(maximum, threads_in_group=4)
        else:
            for row in cutlass.range(cute.size(row_sum), unroll_full=True):
                row_values = values[row, None]
                for column in cutlass.range(cute.size(row_values), unroll_full=True):
                    row_values[column] = self.conversion_faithful_statistics_sum_map(
                        row_values[column], row_max[row]
                    )
                row_sum[row] += utils.fadd_reduce(
                    row_values.load(),
                    init_val=None,
                    arch=self.arch.major * 10 + self.arch.minor,
                )
        self.warp_scheduler_barrier_sync()
        return consumer_state

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: NormalizedExpFoldState,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        score_map_fn: Optional[Callable] = None,
        normalized_map_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
    ):
        assert normalized_map_fn is None, "conversion-faithful normalization requires non-overlapped tile traversal"
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        # RescaleOBeforeGemm: rescale O while QK GEMM is in flight, before PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_weighted_accumulator(acc_O, scores_scale)
        pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v))
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle score mods and masking
        if const_expr(score_map_fn is not None):
            score_map_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block, transpose=self.qk_swap_ab)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))

        row_scale = softmax.update(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP_cur.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            if const_expr(self.qk_swap_ab):
                self.store_qk_fragment_for_consumer(tOrP_cur, smem_copy_params)
            else:
                tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
                cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_weighted_accumulator(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        if const_expr(not self.mma_pv_is_rs and not self.qk_swap_ab):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=self.scheduler_arrival_threads,
                )

    @cute.jit
    def schedule_local_redistribution_barrier(self, barrier_id: cutlass.Constexpr[int]):
        """Advance one generation of the reusable fragment-handoff Event."""
        cute.arch.barrier(
            barrier_id=barrier_id,
            number_of_threads=self.num_mma_threads,
        )

    @cute.jit
    def redistribute_qk_fragment_for_fold(self, acc_S, smem_copy_params):
        """Transpose streamed-owner QK registers through shared memory into Fold ownership."""
        self.schedule_local_redistribution_barrier(int(NamedBarrierFwd.PEmpty))
        source = layout_utils.reshape_acc_to_frgA(acc_S)
        stored = cute.make_rmem_tensor_like(source, self.dtype)
        utils.cvt_f16(source, stored)
        smem_copy_params.copy_P_r2s(stored)
        cute.arch.fence_view_async_shared()
        self.schedule_local_redistribution_barrier(int(NamedBarrierFwd.PFull))
        loaded = smem_copy_params.copy_P_s2r()
        result = cute.make_rmem_tensor_like(acc_S, Float32)
        result_fragment = layout_utils.reshape_acc_to_frgA(result)
        result_fragment.store(loaded.load().to(Float32))
        return result

    @cute.jit
    def store_qk_fragment_for_consumer(self, fragment, smem_copy_params):
        """Publish one streamed-owner fragment as the consumer's shared operand."""
        self.schedule_local_redistribution_barrier(int(NamedBarrierFwd.PEmpty))
        smem_copy_params.copy_P_r2s(fragment)
        cute.arch.fence_view_async_shared()
        self.schedule_local_redistribution_barrier(int(NamedBarrierFwd.PFull))

    @cute.jit
    def apply_score_map(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=None,
    ):
        # Prepare index tensor
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)

        apply_score_map_inner(
            acc_S,
            tScS,
            self.score_map,
            batch_idx,
            head_idx,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_query_idx=None,
            head_group_size=self.relation_group_size if const_expr(self.pack_index_relation) else 1,
        )

    @cute.jit
    def apply_conversion_faithful_score_map(self, acc_S, n_block, seqlen):
        del n_block, seqlen
        for offset in cutlass.range(cute.size(acc_S), unroll_full=True):
            acc_S[offset] = self.conversion_faithful_score_map(acc_S[offset])

    @cute.jit
    def apply_conversion_faithful_normalized_map(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        mRowMax,
        mRowSum,
        acc_S,
        n_block,
        seqlen,
    ):
        del seqlen
        assert mRowMax is not None and mRowSum is not None
        physical_shape = (
            (self.tile_n, self.tile_m) if const_expr(self.qk_swap_ab) else (self.tile_m, self.tile_n)
        )
        physical_offset = (
            (n_block * self.tile_n, m_block * self.tile_m)
            if const_expr(self.qk_swap_ab)
            else (m_block * self.tile_m, n_block * self.tile_n)
        )
        cS = cute.make_identity_tensor(physical_shape)
        cS = cute.domain_offset(physical_offset, cS)
        tScS = thr_mma_qk.partition_C(cS)
        value_count = cutlass.const_expr(cute.size(acc_S.shape))
        for offset in cutlass.range(value_count, unroll_full=True):
            packed_owner = tScS[offset][1] if const_expr(self.qk_swap_ab) else tScS[offset][0]
            owner_idx = packed_owner // self.relation_group_size
            group_idx = packed_owner - owner_idx * self.relation_group_size
            relation_idx = head_idx * self.relation_group_size + group_idx
            acc_S[offset] = self.conversion_faithful_normalized_map(
                acc_S[offset],
                mRowMax[owner_idx, relation_idx, batch_idx],
                mRowSum[owner_idx, relation_idx, batch_idx],
            )

    @cute.jit
    def store_conversion_faithful_row_state(
        self,
        mRowMax,
        mRowSum,
        row_max,
        row_sum,
        tiled_mma,
        tidx,
        m_block,
        head_idx,
        batch_idx,
        seqlen,
        mLSE=None,
        lse_map=None,
        coordinate_column_extent=None,
    ):
        assert mRowMax is not None and mRowSum is not None
        lse = None
        if const_expr(mLSE is not None):
            assert lse_map is not None
            lse = cute.make_fragment_like(row_sum, Float32)
            for m in cutlass.range(cute.size(lse), unroll_full=True):
                lse[m] = lse_map(row_max[m], row_sum[m])
        thr_mma = tiled_mma.get_slice(tidx)
        column_extent = (
            self.tile_hdimv if const_expr(coordinate_column_extent is None) else coordinate_column_extent
        )
        cO = cute.make_identity_tensor((self.tile_m, column_extent))
        taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
        t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
        if taccOcO[0][1] == 0:
            for m in cutlass.range(cute.size(row_max), unroll_full=True):
                if t0accOcO[m, 0][0] < seqlen.seqlen_q * self.relation_group_size - taccOcO[0][0]:
                    packed_owner = m_block * self.tile_m + taccOcO[m, 0][0]
                    owner_idx = packed_owner // self.relation_group_size
                    group_idx = packed_owner - owner_idx * self.relation_group_size
                    relation_idx = head_idx * self.relation_group_size + group_idx
                    mRowMax[owner_idx, relation_idx, batch_idx] = row_max[m].to(self.dtype)
                    mRowSum[owner_idx, relation_idx, batch_idx] = row_sum[m]
        if const_expr(mLSE is not None):
            relation_packer = PackGQA(
                self.tile_m, column_extent, False, self.relation_group_size
            )
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
            relation_packer.store_LSE(
                mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q
            )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=self.scheduler_arrival_threads,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_wg_mma == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=self.scheduler_arrival_threads,
            )
