// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-test-opt --shuttle-plan-row-fold-materialization %s -o /dev/null 2>&1 | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>

module attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%input: tensor<?x4xf32>, %init: tensor<f32>)
      -> tensor<?xf32> {
    %result = "shuttle.region"(%input, %init) ({
    ^bb0(%region_input: tensor<?x4xf32>, %region_init: tensor<f32>):
      %first = "shuttle.map"(%region_input) ({
      ^bb0(%element: f32):
        %negative = arith.negf %element : f32
        "shuttle.yield"(%negative) : (f32) -> ()
      }) {indexing_maps = [#id, #id], source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<?x4xf32>) -> tensor<?x4xf32>
      %second = "shuttle.map"(%first) ({
      ^bb0(%element: f32):
        %square = arith.mulf %element, %element : f32
        "shuttle.yield"(%square) : (f32) -> ()
      }) {indexing_maps = [#id, #id], source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<?x4xf32>) -> tensor<?x4xf32>
      %folded = "shuttle.fold"(%second, %region_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<?x4xf32>, tensor<f32>) -> tensor<?xf32>
      "shuttle.yield"(%folded) : (tensor<?xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>]} : (tensor<?x4xf32>, tensor<f32>) -> tensor<?xf32>
    return %result : tensor<?xf32>
  }
}

// CHECK: requires a bounded set of independently connected static GPU algebra regions
