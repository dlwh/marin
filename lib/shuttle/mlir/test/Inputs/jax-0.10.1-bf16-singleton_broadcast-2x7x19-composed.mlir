// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-singleton-broadcast-fixtures.py
// Generator source SHA-256: 305FFDADE616B4F498BEA71CD09F54ECD073A95A7E6F51F9C97BD48A05499F5D
// Source: jax.jit(ordinary JAX terminal-axis keepdims sum plus JAX-owned
// VJP).lower(...).compiler_ir(StableHLO)
// Case ID: singleton_broadcast_fe71077dbb7b6fcf
// Structural fields: {"boundary":"composed","dimensions":[2,7,19]}
// Inputs: x=(2, 7, 19):bfloat16, output_cotangent=(2, 7, 19):bfloat16
// Outputs: result=(2, 7, 19):bfloat16, input_cotangent=(2, 7, 19):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 599743A219250C6B205AFDA630D45540FC6C8BA18A328D8A52AFC9C2E08AB95E
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: A075B4DF7F1C02E392E0EF0C83C5C7E93277E8092F54DAA94E1DF367596E4421
module @jit_composed attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x7x19xbf16>, %arg1: tensor<2x7x19xbf16>) -> (tensor<2x7x19xbf16> {jax.result_info = "result[0]"}, tensor<2x7x19xbf16> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %4 = stablehlo.multiply %0, %3 : tensor<2x7x19xf32>
    %5 = stablehlo.convert %4 : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    %6 = stablehlo.convert %arg1 : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %7 = stablehlo.multiply %0, %6 : tensor<2x7x19xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %9 = stablehlo.reshape %8 : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %10 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %11 = stablehlo.multiply %6, %10 : tensor<2x7x19xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%9 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<2x7x1xf32>, tensor<f32>) -> tensor<2x7xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x19xf32>
    %14 = stablehlo.add %11, %13 : tensor<2x7x19xf32>
    %15 = stablehlo.convert %14 : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %5, %15 : tensor<2x7x19xbf16>, tensor<2x7x19xbf16>
  }
}
