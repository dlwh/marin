// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt %s --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure -o /dev/null
// RUN: not shuttle-opt %s --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization -o /dev/null 2>&1 | FileCheck %s

// Semantic admission does not claim either of these physical realizations.
// CHECK: cannot derive a bounded generic task plan for this algebra component
// CHECK: requires a bounded set of independently connected static GPU algebra regions
// CHECK: cannot derive a bounded generic task plan for this algebra component
// CHECK: requires a bounded set of independently connected static GPU algebra regions

module {
  func.func @bf16_result(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x4xbf16>) -> tensor<2x4xbf16> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
    return %0 : tensor<2x4xbf16>
  }
}

// -----

module {
  func.func @high_precision(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [HIGH, HIGH] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}
