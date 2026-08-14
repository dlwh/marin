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
// Structural fields: {"boundary":"forward","dimensions":[2,7,19]}
// Inputs: x=(2, 7, 19):bfloat16
// Outputs: result=(2, 7, 19):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 6C7F259E7B024EEE87A4E6C8919821B4BB9DAC9BBB074FBF16B622FEE4B3FC86
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 49804449059A325C0F3A2290D732BDE94D8CB64A9C35E66E3F47F1476C4E23E0
module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x7x19xbf16>) -> (tensor<2x7x19xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %4 = stablehlo.multiply %0, %3 : tensor<2x7x19xf32>
    %5 = stablehlo.convert %4 : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %5 : tensor<2x7x19xbf16>
  }
}
