// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Test-only structural fixture derived from the ABI18 terminal Fold and
// expanding-singleton boundary. It is anonymous and is not an evaluation or
// hardware oracle.
module @abi19_exp_contract {
  func.func public @main(%arg0: tensor<2x7x19xbf16>) -> tensor<2x7x19xbf16> {
    %input = stablehlo.convert %arg0 : (tensor<2x7x19xbf16>) -> tensor<2x7x19xf32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = stablehlo.reduce(%input init: %zero) applies stablehlo.add across dimensions = [2] : (tensor<2x7x19xf32>, tensor<f32>) -> tensor<2x7xf32>
    %singleton = stablehlo.broadcast_in_dim %sum, dims = [0, 1] : (tensor<2x7xf32>) -> tensor<2x7x1xf32>
    %expanded = stablehlo.broadcast_in_dim %singleton, dims = [0, 1, 2] : (tensor<2x7x1xf32>) -> tensor<2x7x19xf32>
    %centered = stablehlo.subtract %input, %expanded : tensor<2x7x19xf32>
    %exponential = stablehlo.exponential %centered : tensor<2x7x19xf32>
    %result = stablehlo.convert %exponential : (tensor<2x7x19xf32>) -> tensor<2x7x19xbf16>
    return %result : tensor<2x7x19xbf16>
  }

  func.func public @secondary(%arg0: tensor<3x2x5x11x263xbf16>) -> tensor<3x2x5x11x263xbf16> {
    %input = stablehlo.convert %arg0 : (tensor<3x2x5x11x263xbf16>) -> tensor<3x2x5x11x263xf32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = stablehlo.reduce(%input init: %zero) applies stablehlo.add across dimensions = [4] : (tensor<3x2x5x11x263xf32>, tensor<f32>) -> tensor<3x2x5x11xf32>
    %singleton = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2, 3] : (tensor<3x2x5x11xf32>) -> tensor<3x2x5x11x1xf32>
    %expanded = stablehlo.broadcast_in_dim %singleton, dims = [0, 1, 2, 3, 4] : (tensor<3x2x5x11x1xf32>) -> tensor<3x2x5x11x263xf32>
    %centered = stablehlo.subtract %input, %expanded : tensor<3x2x5x11x263xf32>
    %exponential = stablehlo.exponential %centered : tensor<3x2x5x11x263xf32>
    %result = stablehlo.convert %exponential : (tensor<3x2x5x11x263xf32>) -> tensor<3x2x5x11x263xbf16>
    return %result : tensor<3x2x5x11x263xbf16>
  }
}
