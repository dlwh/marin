// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary JAX 0.10.1 / jaxlib 0.10.1 matmul VJP fixture.
// Source: jax.jit(backward).lower(...).compiler_ir(dialect="stablehlo")
module @jit_backward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<17x33xbf16>, %arg1: tensor<33x19xbf16>, %arg2: tensor<17x19xbf16>) -> (tensor<17x33xbf16> {jax.result_info = "result[0]"}, tensor<33x19xbf16> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.convert %arg2 : (tensor<17x19xbf16>) -> tensor<17x19xf32>
    %1 = stablehlo.convert %0 : tensor<17x19xf32>
    %2 = stablehlo.convert %arg0 : (tensor<17x33xbf16>) -> tensor<17x33xf32>
    %3 = stablehlo.dot_general %1, %2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<17x19xf32>, tensor<17x33xf32>) -> tensor<19x33xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<19x33xf32>) -> tensor<33x19xf32>
    %5 = stablehlo.convert %4 : (tensor<33x19xf32>) -> tensor<33x19xbf16>
    %6 = stablehlo.convert %0 : tensor<17x19xf32>
    %7 = stablehlo.convert %arg1 : (tensor<33x19xbf16>) -> tensor<33x19xf32>
    %8 = stablehlo.dot_general %6, %7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<17x19xf32>, tensor<33x19xf32>) -> tensor<17x33xf32>
    %9 = stablehlo.convert %8 : (tensor<17x33xf32>) -> tensor<17x33xbf16>
    return %9, %5 : tensor<17x33xbf16>, tensor<33x19xbf16>
  }
}
