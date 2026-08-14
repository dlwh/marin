// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary JAX 0.10.1 / jaxlib 0.10.1 projected batched Contract fixture.
// Source: jax.jit(forward).lower(...).compiler_ir(dialect="stablehlo")
// Structural shape: batch=2, rows=5, heads=3
// Reduction shape: reduction=7, columns=11
module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x5x3x7xbf16>, %arg1: tensor<2x11x3x7xbf16>) -> (tensor<2x3x5x11xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2x5x3x7xbf16>) -> tensor<2x5x3x7xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<2x5x3x7xf32>) -> tensor<2x3x5x7xf32>
    %2 = stablehlo.convert %arg1 : (tensor<2x11x3x7xbf16>) -> tensor<2x11x3x7xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<2x11x3x7xf32>) -> tensor<2x3x11x7xf32>
    %4 = stablehlo.dot_general %1, %3, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x3x5x7xf32>, tensor<2x3x11x7xf32>) -> tensor<2x3x5x11xf32>
    %5 = stablehlo.convert %4 : (tensor<2x3x5x11xf32>) -> tensor<2x3x5x11xbf16>
    return %5 : tensor<2x3x5x11xbf16>
  }
}
