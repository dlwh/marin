// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-test-opt %s --split-input-file --shuttle-gpu-executable-bundle-pipeline -o /dev/null 2>&1 | FileCheck %s

module @multiple_reductions {
  func.func public @main(%lhs: tensor<2x3x5xbf16>, %rhs: tensor<3x5x7xbf16>) -> tensor<2x7xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1, 2] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x3x5xbf16>, tensor<3x5x7xbf16>) -> tensor<2x7xf32>
    %1 = stablehlo.convert %0 : (tensor<2x7xf32>) -> tensor<2x7xbf16>
    return %1 : tensor<2x7xbf16>
  }
}

// CHECK: requires a bounded set of independently connected static GPU algebra regions

// -----

module @dynamic_batch {
  func.func public @main(%lhs: tensor<?x3x5xbf16>, %rhs: tensor<?x5x7xbf16>) -> tensor<?x3x7xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<?x3x5xbf16>, tensor<?x5x7xbf16>) -> tensor<?x3x7xf32>
    %1 = stablehlo.convert %0 : (tensor<?x3x7xf32>) -> tensor<?x3x7xbf16>
    return %1 : tensor<?x3x7xbf16>
  }
}

// CHECK: requires a bounded set of independently connected static GPU algebra regions

// -----

module @rank_nine {
  func.func public @main(%lhs: tensor<2x2x2x2x2x2x2x3x5xbf16>, %rhs: tensor<2x2x2x2x2x2x2x5x7xbf16>) -> tensor<2x2x2x2x2x2x2x3x7xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, batching_dims = [0, 1, 2, 3, 4, 5, 6] x [0, 1, 2, 3, 4, 5, 6], contracting_dims = [8] x [7], precision = [DEFAULT, DEFAULT] : (tensor<2x2x2x2x2x2x2x3x5xbf16>, tensor<2x2x2x2x2x2x2x5x7xbf16>) -> tensor<2x2x2x2x2x2x2x3x7xf32>
    %1 = stablehlo.convert %0 : (tensor<2x2x2x2x2x2x2x3x7xf32>) -> tensor<2x2x2x2x2x2x2x3x7xbf16>
    return %1 : tensor<2x2x2x2x2x2x2x3x7xbf16>
  }
}

// CHECK: requires a bounded set of independently connected static GPU algebra regions

// -----

module @public_f32 {
  func.func public @main(%lhs: tensor<2x3x5xf32>, %rhs: tensor<2x5x7xf32>) -> tensor<2x3x7xf32> {
    %0 = stablehlo.dot_general %lhs, %rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3x5xf32>, tensor<2x5x7xf32>) -> tensor<2x3x7xf32>
    return %0 : tensor<2x3x7xf32>
  }
}

// CHECK: requires a bounded set of independently connected static GPU algebra regions

// -----

// ABI16 admits matrix-style Contracts with at least two parallel dimensions.
// Batched vector dot remains outside the frozen boundary.
module @batched_vector_dot {
  func.func public @main(%lhs: tensor<3x13xbf16>, %rhs: tensor<3x13xbf16>) -> tensor<3xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x13xbf16>, tensor<3x13xbf16>) -> tensor<3xf32>
    %1 = stablehlo.convert %0 : (tensor<3xf32>) -> tensor<3xbf16>
    return %1 : tensor<3xbf16>
  }
}

// CHECK: requires a bounded set of independently connected static GPU algebra regions
