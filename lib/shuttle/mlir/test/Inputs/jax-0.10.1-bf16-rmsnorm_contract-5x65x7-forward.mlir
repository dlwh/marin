// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: hardware_gate_pending
// Hardware evidence: none
// Generator: regenerate-jax-bf16-rmsnorm-contract-fixtures.py
// Generator source SHA-256: F030F627F82526AAFAE0277AE90B7A49D4D1692AE524C706A551D0C6E90A64DA
// Source: jax.jit(ordinary JAX BF16 RMSNorm followed by matmul)
//   .lower(...).compiler_ir(dialect="stablehlo")
// Case ID: rmsnorm_contract_36a34e0d6c3a73ee
// Structural fields: {"activation":[5,65],"columns":7,"projection":[65,7],"weight":[65]}
// Inputs: x=(5, 65):bfloat16, weight=(65,):bfloat16, projection=(65, 7):bfloat16
// Outputs: result=(5, 7):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: C3FC17EADB07DF6653EFF925A587CFCC5CB33E84CC3AF3D92F4F9828DF4E3F90
module @jit_rmsnorm_contract attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<5x65xbf16>, %arg1: tensor<65xbf16>, %arg2: tensor<65x7xbf16>) -> (tensor<5x7xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<5x65xbf16>) -> tensor<5x65xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<5x65xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<5x65xf32>, tensor<f32>) -> tensor<5xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<5xf32>) -> tensor<5x1xf32>
    %cst_0 = stablehlo.constant dense<6.500000e+01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<5x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<5x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<5x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<5x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<5x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<5x1xf32>) -> tensor<5x65xf32>
    %10 = stablehlo.multiply %0, %9 : tensor<5x65xf32>
    %11 = stablehlo.convert %arg1 : (tensor<65xbf16>) -> tensor<65xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<65xf32>) -> tensor<1x65xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x65xf32>) -> tensor<5x65xf32>
    %14 = stablehlo.multiply %10, %13 : tensor<5x65xf32>
    %15 = stablehlo.convert %14 : (tensor<5x65xf32>) -> tensor<5x65xbf16>
    %16 = stablehlo.dot_general %15, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<5x65xbf16>, tensor<65x7xbf16>) -> tensor<5x7xf32>
    %17 = stablehlo.convert %16 : (tensor<5x7xf32>) -> tensor<5x7xbf16>
    return %17 : tensor<5x7xbf16>
  }
}
