// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary JAX 0.10.1 / jaxlib 0.10.1 projected batched Contract fixture.
// Source: jax.jit(backward).lower(...).compiler_ir(dialect="stablehlo")
// Structural shape: batch=3, rows=4, heads=2
// Reduction shape: reduction=13, columns=6
module @jit_backward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x4x2x13xbf16>, %arg1: tensor<3x6x2x13xbf16>, %arg2: tensor<3x2x4x6xbf16>) -> (tensor<3x4x2x13xbf16> {jax.result_info = "result[0]"}, tensor<3x6x2x13xbf16> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<3x4x2x13xbf16>) -> tensor<3x4x2x13xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<3x4x2x13xf32>) -> tensor<3x2x4x13xf32>
    %2 = stablehlo.convert %arg1 : (tensor<3x6x2x13xbf16>) -> tensor<3x6x2x13xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<3x6x2x13xf32>) -> tensor<3x2x6x13xf32>
    %4 = stablehlo.convert %arg2 : (tensor<3x2x4x6xbf16>) -> tensor<3x2x4x6xf32>
    %5 = stablehlo.dot_general %4, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<3x2x4x6xf32>, tensor<3x2x4x13xf32>) -> tensor<3x2x6x13xf32>
    %6 = stablehlo.dot_general %4, %3, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<3x2x4x6xf32>, tensor<3x2x6x13xf32>) -> tensor<3x2x4x13xf32>
    %7 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<3x2x6x13xf32>) -> tensor<3x6x2x13xf32>
    %8 = stablehlo.convert %7 : (tensor<3x6x2x13xf32>) -> tensor<3x6x2x13xbf16>
    %9 = stablehlo.transpose %6, dims = [0, 2, 1, 3] : (tensor<3x2x4x13xf32>) -> tensor<3x4x2x13xf32>
    %10 = stablehlo.convert %9 : (tensor<3x4x2x13xf32>) -> tensor<3x4x2x13xbf16>
    return %10, %8 : tensor<3x4x2x13xbf16>, tensor<3x6x2x13xbf16>
  }
}
