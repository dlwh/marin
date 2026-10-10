// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %s --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=ALGEBRA
// RUN: shuttle-test-opt %s --shuttle-stablehlo-source-ordered-pipeline | FileCheck %s --check-prefix=LOWERED

module {
  func.func @elementwise_maximum(%lhs: tensor<2x3xf32>,
      %rhs: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %result = stablehlo.maximum %lhs, %rhs : tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }
}

// ALGEBRA: "shuttle.map"
// ALGEBRA-SAME: indexing_maps = [#map, #map, #map]
// ALGEBRA-SAME: semantics = #shuttle.map_semantics<pointwise>
// ALGEBRA: "arith.maximumf"
// ALGEBRA-NOT: stablehlo.maximum

// LOWERED-NOT: shuttle.
// LOWERED: stablehlo.maximum
