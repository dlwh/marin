// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary JAX 0.10.1 / jaxlib 0.10.1 matmul composed fixture.
// Source: jax.jit(composed).lower(...).compiler_ir(dialect="stablehlo")
module @jit_composed attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<17x33xbf16>, %arg1: tensor<33x19xbf16>, %arg2: tensor<17x19xbf16>) -> (tensor<17x19xbf16> {jax.result_info = "result[0]"}, tensor<17x33xbf16> {jax.result_info = "result[1]"}, tensor<33x19xbf16> {jax.result_info = "result[2]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<17x33xbf16>, tensor<33x19xbf16>) -> tensor<17x19xf32>
    %1 = stablehlo.convert %0 : (tensor<17x19xf32>) -> tensor<17x19xbf16>
    %2 = stablehlo.convert %arg2 : (tensor<17x19xbf16>) -> tensor<17x19xf32>
    %3 = stablehlo.convert %2 : tensor<17x19xf32>
    %4 = stablehlo.convert %arg0 : (tensor<17x33xbf16>) -> tensor<17x33xf32>
    %5 = stablehlo.dot_general %3, %4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<17x19xf32>, tensor<17x33xf32>) -> tensor<19x33xf32>
    %6 = stablehlo.transpose %5, dims = [1, 0] : (tensor<19x33xf32>) -> tensor<33x19xf32>
    %7 = stablehlo.convert %6 : (tensor<33x19xf32>) -> tensor<33x19xbf16>
    %8 = stablehlo.convert %2 : tensor<17x19xf32>
    %9 = stablehlo.convert %arg1 : (tensor<33x19xbf16>) -> tensor<33x19xf32>
    %10 = stablehlo.dot_general %8, %9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<17x19xf32>, tensor<33x19xf32>) -> tensor<17x33xf32>
    %11 = stablehlo.convert %10 : (tensor<17x33xf32>) -> tensor<17x33xbf16>
    return %1, %11, %7 : tensor<17x19xbf16>, tensor<17x33xbf16>, tensor<33x19xbf16>
  }
}
