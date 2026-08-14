// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Minimal StableHLO produced by ordinary JAX
// jnp.matmul(lhs, rhs, preferred_element_type=jnp.float32).astype(jnp.bfloat16).
// The symbols are deliberately anonymous and do not encode shapes or workloads.

module @anonymous_a {
  func.func public @main(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x5xbf16>) -> tensor<2x5xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x3xbf16>, tensor<3x5xbf16>) -> tensor<2x5xf32>
    %result = stablehlo.convert %contract : (tensor<2x5xf32>) -> tensor<2x5xbf16>
    return %result : tensor<2x5xbf16>
  }
}

// -----

module @anonymous_b {
  func.func public @main(%lhs: tensor<3x5xbf16>, %rhs: tensor<5x2xbf16>) -> tensor<3x2xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x5xbf16>, tensor<5x2xbf16>) -> tensor<3x2xf32>
    %result = stablehlo.convert %contract : (tensor<3x2xf32>) -> tensor<3x2xbf16>
    return %result : tensor<3x2xbf16>
  }
}

// -----

module @anonymous_c {
  func.func public @main(%lhs: tensor<7x13xbf16>, %rhs: tensor<13x3xbf16>) -> tensor<7x3xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<7x13xbf16>, tensor<13x3xbf16>) -> tensor<7x3xf32>
    %result = stablehlo.convert %contract : (tensor<7x3xf32>) -> tensor<7x3xbf16>
    return %result : tensor<7x3xbf16>
  }
}

// -----

module @anonymous_d {
  func.func public @main(%lhs: tensor<11x3xbf16>, %rhs: tensor<3x17xbf16>) -> tensor<11x17xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<11x3xbf16>, tensor<3x17xbf16>) -> tensor<11x17xf32>
    %result = stablehlo.convert %contract : (tensor<11x17xf32>) -> tensor<11x17xbf16>
    return %result : tensor<11x17xbf16>
  }
}
