// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: hardware_gate_pending
// Hardware evidence: none
// Generator: regenerate-jax-bf16-rmsnorm-contract-vjp-fixtures.py
// Generator source SHA-256: 1BE378B0CECB4BACD8698FF05938ED1430EEC4FC7B3F4035329C377ABF499355
// Source: jax.jit(ordinary JAX BF16 RMSNorm-to-Contract composed VJP)
//   .lower(...).compiler_ir(dialect="stablehlo")
// Structural fields: {"activation":[5,19],"columns":7,"reduction":19,"rows":5}
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: D54F3B77EFCF769AE8C4635B621831E482D25B1B9B9782C11148C1C665C649D0
module @jit_composed attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<5x19xbf16>, %arg1: tensor<19xbf16>, %arg2: tensor<19x7xbf16>, %arg3: tensor<5x7xbf16>) -> (tensor<5x7xbf16> {jax.result_info = "result[0]"}, tensor<5x19xbf16> {jax.result_info = "result[1]"}, tensor<19xbf16> {jax.result_info = "result[2]"}, tensor<19x7xbf16> {jax.result_info = "result[3]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<5x19xbf16>) -> tensor<5x19xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<5x19xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<5x19xf32>, tensor<f32>) -> tensor<5xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<5xf32>) -> tensor<5x1xf32>
    %cst_0 = stablehlo.constant dense<1.900000e+01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<5x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<5x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<5x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<5x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<5x1xf32>
    %9 = stablehlo.divide %8, %7 : tensor<5x1xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<5x1xf32>
    %11 = stablehlo.multiply %10, %9 : tensor<5x1xf32>
    %12 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<5x1xf32>) -> tensor<5x19xf32>
    %13 = stablehlo.multiply %0, %12 : tensor<5x19xf32>
    %14 = stablehlo.convert %arg1 : (tensor<19xbf16>) -> tensor<19xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<19xf32>) -> tensor<1x19xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<1x19xf32>) -> tensor<5x19xf32>
    %17 = stablehlo.multiply %13, %16 : tensor<5x19xf32>
    %18 = stablehlo.convert %17 : (tensor<5x19xf32>) -> tensor<5x19xbf16>
    %19 = stablehlo.dot_general %18, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<5x19xbf16>, tensor<19x7xbf16>) -> tensor<5x7xf32>
    %20 = stablehlo.convert %19 : (tensor<5x7xf32>) -> tensor<5x7xbf16>
    %21 = stablehlo.convert %arg3 : (tensor<5x7xbf16>) -> tensor<5x7xf32>
    %22 = stablehlo.convert %21 : tensor<5x7xf32>
    %23 = stablehlo.convert %18 : (tensor<5x19xbf16>) -> tensor<5x19xf32>
    %24 = stablehlo.dot_general %22, %23, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<5x7xf32>, tensor<5x19xf32>) -> tensor<7x19xf32>
    %25 = stablehlo.transpose %24, dims = [1, 0] : (tensor<7x19xf32>) -> tensor<19x7xf32>
    %26 = stablehlo.convert %25 : (tensor<19x7xf32>) -> tensor<19x7xbf16>
    %27 = stablehlo.convert %21 : tensor<5x7xf32>
    %28 = stablehlo.convert %arg2 : (tensor<19x7xbf16>) -> tensor<19x7xf32>
    %29 = stablehlo.dot_general %27, %28, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x7xf32>, tensor<19x7xf32>) -> tensor<5x19xf32>
    %30 = stablehlo.convert %29 : (tensor<5x19xf32>) -> tensor<5x19xbf16>
    %31 = stablehlo.convert %30 : (tensor<5x19xbf16>) -> tensor<5x19xf32>
    %32 = stablehlo.multiply %13, %31 : tensor<5x19xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %33 = stablehlo.reduce(%32 init: %cst_3) applies stablehlo.add across dimensions = [0] : (tensor<5x19xf32>, tensor<f32>) -> tensor<19xf32>
    %34 = stablehlo.reshape %33 : (tensor<19xf32>) -> tensor<1x19xf32>
    %35 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<1x19xf32>) -> tensor<5x19xf32>
    %36 = stablehlo.multiply %31, %35 : tensor<5x19xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %37 = stablehlo.reduce(%34 init: %cst_4) applies stablehlo.add across dimensions = [0] : (tensor<1x19xf32>, tensor<f32>) -> tensor<19xf32>
    %38 = stablehlo.convert %37 : (tensor<19xf32>) -> tensor<19xbf16>
    %39 = stablehlo.multiply %0, %36 : tensor<5x19xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %40 = stablehlo.reduce(%39 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<5x19xf32>, tensor<f32>) -> tensor<5xf32>
    %41 = stablehlo.reshape %40 : (tensor<5xf32>) -> tensor<5x1xf32>
    %42 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<5x1xf32>) -> tensor<5x19xf32>
    %43 = stablehlo.multiply %36, %42 : tensor<5x19xf32>
    %44 = stablehlo.multiply %41, %11 : tensor<5x1xf32>
    %cst_6 = stablehlo.constant dense<1.900000e+01> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<5x1xf32>
    %46 = stablehlo.divide %44, %45 : tensor<5x1xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %47 = stablehlo.reduce(%46 init: %cst_7) applies stablehlo.add across dimensions = [1] : (tensor<5x1xf32>, tensor<f32>) -> tensor<5xf32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [0] : (tensor<5xf32>) -> tensor<5x19xf32>
    %49 = stablehlo.multiply %0, %48 : tensor<5x19xf32>
    %50 = stablehlo.add %43, %49 : tensor<5x19xf32>
    %51 = stablehlo.multiply %48, %0 : tensor<5x19xf32>
    %52 = stablehlo.add %50, %51 : tensor<5x19xf32>
    %53 = stablehlo.convert %52 : (tensor<5x19xf32>) -> tensor<5x19xbf16>
    return %20, %53, %38, %26 : tensor<5x7xbf16>, tensor<5x19xbf16>, tensor<19xbf16>, tensor<19x7xbf16>
  }
}
