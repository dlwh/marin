// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: hardware_gate_pending
// Hardware evidence: none
// Generator: regenerate-jax-bf16-rmsnorm-contract-fixtures.py
// Generator source SHA-256: 0F2118B74CE0307BC6D8BA77CF93D1EBB2EB69829D743AC29C4216214E0F6737
// Source: jax.jit(ordinary JAX BF16 RMSNorm followed by matmul)
//   .lower(...).compiler_ir(dialect="stablehlo")
// Case ID: rmsnorm_contract_d3d61db6290a8ae6
// Structural fields: {"activation":[3,2,5,11,263],"columns":521,"projection":[263,521],"weight":[263]}
// Inputs: x=(3, 2, 5, 11, 263):bfloat16, weight=(263,):bfloat16, projection=(263, 521):bfloat16
// Outputs: result=(3, 2, 5, 11, 521):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: E9F9304744F2AFB98F1FA34DFDB21D225816B16BA438EA4CDBF63D3086FFB924
module @jit_rmsnorm_contract attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x2x5x11x263xbf16>, %arg1: tensor<263xbf16>, %arg2: tensor<263x521xbf16>) -> (tensor<3x2x5x11x521xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<3x2x5x11x263xbf16>) -> tensor<3x2x5x11x263xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<3x2x5x11x263xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [4] : (tensor<3x2x5x11x263xf32>, tensor<f32>) -> tensor<3x2x5x11xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3] : (tensor<3x2x5x11xf32>) -> tensor<3x2x5x11x1xf32>
    %cst_0 = stablehlo.constant dense<2.630000e+02> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3x2x5x11x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<3x2x5x11x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<3x2x5x11x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<3x2x5x11x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<3x2x5x11x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3, 4] : (tensor<3x2x5x11x1xf32>) -> tensor<3x2x5x11x263xf32>
    %10 = stablehlo.multiply %0, %9 : tensor<3x2x5x11x263xf32>
    %11 = stablehlo.convert %arg1 : (tensor<263xbf16>) -> tensor<263xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [4] : (tensor<263xf32>) -> tensor<1x1x1x1x263xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2, 3, 4] : (tensor<1x1x1x1x263xf32>) -> tensor<3x2x5x11x263xf32>
    %14 = stablehlo.multiply %10, %13 : tensor<3x2x5x11x263xf32>
    %15 = stablehlo.convert %14 : (tensor<3x2x5x11x263xf32>) -> tensor<3x2x5x11x263xbf16>
    %16 = stablehlo.dot_general %15, %arg2, contracting_dims = [4] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x2x5x11x263xbf16>, tensor<263x521xbf16>) -> tensor<3x2x5x11x521xf32>
    %17 = stablehlo.convert %16 : (tensor<3x2x5x11x521xf32>) -> tensor<3x2x5x11x521xbf16>
    return %17 : tensor<3x2x5x11x521xbf16>
  }
}
