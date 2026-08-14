// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary JAX 0.10.1 / jaxlib 0.10.1 projected batched Contract fixture.
// Source: jax.jit(backward).lower(...).compiler_ir(dialect="stablehlo")
// Structural shape: batch=2, rows=5, heads=3
// Reduction shape: reduction=7, columns=11
module @jit_backward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x5x3x7xbf16>, %arg1: tensor<2x11x3x7xbf16>, %arg2: tensor<2x3x5x11xbf16>) -> (tensor<2x5x3x7xbf16> {jax.result_info = "result[0]"}, tensor<2x11x3x7xbf16> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2x5x3x7xbf16>) -> tensor<2x5x3x7xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<2x5x3x7xf32>) -> tensor<2x3x5x7xf32>
    %2 = stablehlo.convert %arg1 : (tensor<2x11x3x7xbf16>) -> tensor<2x11x3x7xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<2x11x3x7xf32>) -> tensor<2x3x11x7xf32>
    %4 = stablehlo.convert %arg2 : (tensor<2x3x5x11xbf16>) -> tensor<2x3x5x11xf32>
    %5 = stablehlo.dot_general %4, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x3x5x11xf32>, tensor<2x3x5x7xf32>) -> tensor<2x3x11x7xf32>
    %6 = stablehlo.dot_general %4, %3, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x3x5x11xf32>, tensor<2x3x11x7xf32>) -> tensor<2x3x5x7xf32>
    %7 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<2x3x11x7xf32>) -> tensor<2x11x3x7xf32>
    %8 = stablehlo.convert %7 : (tensor<2x11x3x7xf32>) -> tensor<2x11x3x7xbf16>
    %9 = stablehlo.transpose %6, dims = [0, 2, 1, 3] : (tensor<2x3x5x7xf32>) -> tensor<2x5x3x7xf32>
    %10 = stablehlo.convert %9 : (tensor<2x5x3x7xf32>) -> tensor<2x5x3x7xbf16>
    return %10, %8 : tensor<2x5x3x7xbf16>, tensor<2x11x3x7xbf16>
  }
}
