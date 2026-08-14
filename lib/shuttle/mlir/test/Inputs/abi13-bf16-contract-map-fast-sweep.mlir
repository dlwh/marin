// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Minimal StableHLO produced by ordinary JAX
// jnp.matmul(lhs, rhs, preferred_element_type=jnp.float32).astype(jnp.bfloat16).
// Symbols and dimensions are anonymous structural coverage, not dispatch keys.

module @anonymous_a {
  func.func public @main(%lhs: tensor<1x16xbf16>, %rhs: tensor<16x1xbf16>) -> tensor<1x1xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16xbf16>, tensor<16x1xbf16>) -> tensor<1x1xf32>
    %result = stablehlo.convert %contract : (tensor<1x1xf32>) -> tensor<1x1xbf16>
    return %result : tensor<1x1xbf16>
  }
}

// -----

module @anonymous_b {
  func.func public @main(%lhs: tensor<1x32xbf16>, %rhs: tensor<32x9xbf16>) -> tensor<1x9xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x32xbf16>, tensor<32x9xbf16>) -> tensor<1x9xf32>
    %result = stablehlo.convert %contract : (tensor<1x9xf32>) -> tensor<1x9xbf16>
    return %result : tensor<1x9xbf16>
  }
}

// -----

module @anonymous_c {
  func.func public @main(%lhs: tensor<2x33xbf16>, %rhs: tensor<33x3xbf16>) -> tensor<2x3xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x33xbf16>, tensor<33x3xbf16>) -> tensor<2x3xf32>
    %result = stablehlo.convert %contract : (tensor<2x3xf32>) -> tensor<2x3xbf16>
    return %result : tensor<2x3xbf16>
  }
}

// -----

module @anonymous_d {
  func.func public @main(%lhs: tensor<17x65xbf16>, %rhs: tensor<65x19xbf16>) -> tensor<17x19xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<17x65xbf16>, tensor<65x19xbf16>) -> tensor<17x19xf32>
    %result = stablehlo.convert %contract : (tensor<17x19xf32>) -> tensor<17x19xbf16>
    return %result : tensor<17x19xbf16>
  }
}
