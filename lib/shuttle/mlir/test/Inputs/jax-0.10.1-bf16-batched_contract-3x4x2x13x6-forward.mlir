// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary JAX 0.10.1 / jaxlib 0.10.1 projected batched Contract fixture.
// Source: jax.jit(forward).lower(...).compiler_ir(dialect="stablehlo")
// Structural shape: batch=3, rows=4, heads=2
// Reduction shape: reduction=13, columns=6
module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x4x2x13xbf16>, %arg1: tensor<3x6x2x13xbf16>) -> (tensor<3x2x4x6xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<3x4x2x13xbf16>) -> tensor<3x4x2x13xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<3x4x2x13xf32>) -> tensor<3x2x4x13xf32>
    %2 = stablehlo.convert %arg1 : (tensor<3x6x2x13xbf16>) -> tensor<3x6x2x13xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<3x6x2x13xf32>) -> tensor<3x2x6x13xf32>
    %4 = stablehlo.dot_general %1, %3, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<3x2x4x13xf32>, tensor<3x2x6x13xf32>) -> tensor<3x2x4x6xf32>
    %5 = stablehlo.convert %4 : (tensor<3x2x4x6xf32>) -> tensor<3x2x4x6xbf16>
    return %5 : tensor<3x2x4x6xbf16>
  }
}
