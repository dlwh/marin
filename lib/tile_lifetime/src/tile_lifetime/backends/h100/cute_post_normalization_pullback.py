# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic pointwise Map plus feature Fold for normalized-output pullbacks."""

from __future__ import annotations

import operator
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32
from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from quack import layout_utils


class CutePostNormalizationPullback:
    """Apply a scalar VJP and reduce ``U * dU`` over one feature dimension."""

    def __init__(self, feature_dimension: int, vjp: Callable):
        if feature_dimension % cute.arch.WARP_SIZE != 0:
            raise ValueError("the current SM90 post-normalization pullback requires a warp-multiple feature dimension")
        self.feature_dimension = feature_dimension
        self.vjp = vjp
        self.num_threads = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        normalized_output: cute.Tensor,
        output_cotangent: cute.Tensor,
        normalized_output_cotangent: cute.Tensor,
        normalized_output_cotangent_residual: cute.Tensor,
        output_dot: cute.Tensor,
        stream: cuda.CUstream,
    ):
        (
            normalized_output,
            output_cotangent,
            normalized_output_cotangent,
            normalized_output_cotangent_residual,
            output_dot,
        ) = [
            assume_tensor_aligned(tensor)
            for tensor in (
                normalized_output,
                output_cotangent,
                normalized_output_cotangent,
                normalized_output_cotangent_residual,
                output_dot,
            )
        ]
        normalized_output, output_cotangent, normalized_output_cotangent, normalized_output_cotangent_residual = [
            layout_utils.select(tensor, [1, 3, 2, 0])
            for tensor in (
                normalized_output,
                output_cotangent,
                normalized_output_cotangent,
                normalized_output_cotangent_residual,
            )
        ]
        output_dot = layout_utils.select(output_dot, [2, 1, 0])
        self.kernel(
            normalized_output,
            output_cotangent,
            normalized_output_cotangent,
            normalized_output_cotangent_residual,
            output_dot,
        ).launch(
            grid=[normalized_output.shape[0], normalized_output.shape[2], normalized_output.shape[3]],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        normalized_output: cute.Tensor,
        output_cotangent: cute.Tensor,
        normalized_output_cotangent: cute.Tensor,
        normalized_output_cotangent_residual: cute.Tensor,
        output_dot: cute.Tensor,
    ):
        lane, _, _ = cute.arch.thread_idx()
        owner, relation, batch = cute.arch.block_idx()
        dot = Float32(0.0)
        for feature_offset in cutlass.range_constexpr(self.feature_dimension // cute.arch.WARP_SIZE):
            feature = lane + feature_offset * cute.arch.WARP_SIZE
            primal = Float32(normalized_output[owner, feature, relation, batch])
            cotangent = Float32(output_cotangent[owner, feature, relation, batch])
            normalized_cotangent = self.vjp(cotangent, primal)
            primary = cutlass.BFloat16(normalized_cotangent)
            residual = cutlass.BFloat16(Float32(normalized_cotangent) - Float32(primary))
            normalized_output_cotangent[owner, feature, relation, batch] = primary
            normalized_output_cotangent_residual[owner, feature, relation, batch] = residual
            reconstructed_cotangent = Float32(primary) + Float32(residual)
            dot += primal * reconstructed_cotangent
        dot = utils.warp_reduce(dot, operator.add)
        if lane == 0:
            output_dot[owner, relation, batch] = dot
