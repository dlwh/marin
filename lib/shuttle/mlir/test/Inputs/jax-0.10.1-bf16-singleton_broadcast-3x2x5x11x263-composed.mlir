// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-singleton-broadcast-fixtures.py
// Generator source SHA-256: 305FFDADE616B4F498BEA71CD09F54ECD073A95A7E6F51F9C97BD48A05499F5D
// Source: jax.jit(ordinary JAX terminal-axis keepdims sum plus JAX-owned
// VJP).lower(...).compiler_ir(StableHLO)
// Case ID: singleton_broadcast_fe5d7743e1867457
// Structural fields: {"boundary":"composed","dimensions":[3,2,5,11,263]}
// Inputs: x=(3, 2, 5, 11, 263):bfloat16, output_cotangent=(3, 2, 5, 11, 263):bfloat16
// Outputs: result=(3, 2, 5, 11, 263):bfloat16, input_cotangent=(3, 2, 5, 11, 263):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 69C4C15CCF986A76951DB787441027B31BE8391FBD9FA49C422D2DC11647D187
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 0600181DD79006C76FC819C24B6CF8DEB25496288C914A439E69DB7CC6CAF770
module @jit_composed attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x2x5x11x263xbf16>, %arg1: tensor<3x2x5x11x263xbf16>) -> (tensor<3x2x5x11x263xbf16> {jax.result_info = "result[0]"}, tensor<3x2x5x11x263xbf16> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<3x2x5x11x263xbf16>) -> tensor<3x2x5x11x263xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [4] : (tensor<3x2x5x11x263xf32>, tensor<f32>) -> tensor<3x2x5x11xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<3x2x5x11xf32>) -> tensor<3x2x5x11x1xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3, 4] : (tensor<3x2x5x11x1xf32>) -> tensor<3x2x5x11x263xf32>
    %4 = stablehlo.multiply %0, %3 : tensor<3x2x5x11x263xf32>
    %5 = stablehlo.convert %4 : (tensor<3x2x5x11x263xf32>) -> tensor<3x2x5x11x263xbf16>
    %6 = stablehlo.convert %arg1 : (tensor<3x2x5x11x263xbf16>) -> tensor<3x2x5x11x263xf32>
    %7 = stablehlo.multiply %0, %6 : tensor<3x2x5x11x263xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [4] : (tensor<3x2x5x11x263xf32>, tensor<f32>) -> tensor<3x2x5x11xf32>
    %9 = stablehlo.reshape %8 : (tensor<3x2x5x11xf32>) -> tensor<3x2x5x11x1xf32>
    %10 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3, 4] : (tensor<3x2x5x11x1xf32>) -> tensor<3x2x5x11x263xf32>
    %11 = stablehlo.multiply %6, %10 : tensor<3x2x5x11x263xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%9 init: %cst_1) applies stablehlo.add across dimensions = [4] : (tensor<3x2x5x11x1xf32>, tensor<f32>) -> tensor<3x2x5x11xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2, 3] : (tensor<3x2x5x11xf32>) -> tensor<3x2x5x11x263xf32>
    %14 = stablehlo.add %11, %13 : tensor<3x2x5x11x263xf32>
    %15 = stablehlo.convert %14 : (tensor<3x2x5x11x263xf32>) -> tensor<3x2x5x11x263xbf16>
    return %5, %15 : tensor<3x2x5x11x263xbf16>, tensor<3x2x5x11x263xbf16>
  }
}
