// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module {
  func.func @bf16_default(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x4xbf16>) -> tensor<2x4xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
    return %0 : tensor<2x4xbf16>
  }
  func.func @bf16_high_batched(%lhs: tensor<2x3x5xbf16>, %rhs: tensor<2x5x7xbf16>) -> tensor<2x3x7xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [HIGH, HIGH] : (tensor<2x3x5xbf16>, tensor<2x5x7xbf16>) -> tensor<2x3x7xbf16>
    return %0 : tensor<2x3x7xbf16>
  }
  func.func @bf16_highest_reordered(%lhs: tensor<2x3x5x7xbf16>, %rhs: tensor<5x7x3x11xbf16>) -> tensor<7x2x11xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, batching_dims = [3] x [1], contracting_dims = [2, 1] x [0, 2], precision = [HIGHEST, HIGHEST] : (tensor<2x3x5x7xbf16>, tensor<5x7x3x11xbf16>) -> tensor<7x2x11xbf16>
    return %0 : tensor<7x2x11xbf16>
  }
  func.func @bf16_asymmetric_precision(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x4xbf16>) -> tensor<2x4xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [HIGH, HIGHEST] : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
    return %0 : tensor<2x4xbf16>
  }
  func.func @f32_dot_then_convert(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x4xbf16>) -> tensor<2x4xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xf32>
    %1 = stablehlo.convert %0 : (tensor<2x4xf32>) -> tensor<2x4xbf16>
    return %1 : tensor<2x4xbf16>
  }
  func.func @f32_high_precision(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [HIGH, DEFAULT] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  func.func @bf16_scalar_result(%lhs: tensor<3xbf16>, %rhs: tensor<3xbf16>) -> tensor<bf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xbf16>, tensor<3xbf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }
  func.func @bf16_omitted_precision(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x4xbf16>) -> tensor<2x4xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0] : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
    return %0 : tensor<2x4xbf16>
  }
  func.func @bf16_empty_precision(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x4xbf16>) -> tensor<2x4xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [] : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
    return %0 : tensor<2x4xbf16>
  }
  func.func @f32_omitted_precision(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  func.func @f32_empty_precision(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}
