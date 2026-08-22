// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Shape, source ordinal, and symbol metamorphism of the rank-2 fixture.

#contract0_lhs = affine_map<(b, m, n, k) -> (b, m, k)>
#contract0_rhs = affine_map<(b, m, n, k) -> (k, n)>
#contract0_out = affine_map<(b, m, n, k) -> (b, m, n)>
#identity = affine_map<(b, m, n) -> (b, m, n)>
#prefix = affine_map<(b, m, n) -> (b, m)>
#contract1_lhs = affine_map<(b, m, o, n) -> (b, m, n)>
#contract1_rhs = affine_map<(b, m, o, n) -> (n, o)>
#contract1_out = affine_map<(b, m, o, n) -> (b, m, o)>

module @different_anonymous_graph attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @other_entry(%left: tensor<2x3x4xf32>, %right: tensor<4x11xf32>,
      %projection: tensor<11x13xf32>, %zero: tensor<f32>)
      -> tensor<2x3x13xf32> {
    %result = "shuttle.region"(%left, %right, %projection, %zero) ({
    ^bb0(%region_left: tensor<2x3x4xf32>,
         %region_right: tensor<4x11xf32>,
         %region_projection: tensor<11x13xf32>, %region_zero: tensor<f32>):
      %h = "shuttle.contract"(%region_left, %region_right) {
        accumulator_types = [f32], algorithm = "dot_general",
        indexing_maps = [#contract0_lhs, #contract0_rhs, #contract0_out],
        iterator_kinds = ["parallel", "parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 4>
      } : (tensor<2x3x4xf32>, tensor<4x11xf32>) -> tensor<2x3x11xf32>
      %s = "shuttle.fold"(%h, %region_zero) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>, order_free = true,
        reduction_dimensions = array<i64: 2>,
        source = #shuttle.source_ref<0, 0, 0, 6>
      } : (tensor<2x3x11xf32>, tensor<f32>) -> tensor<2x3xf32>
      %z = "shuttle.map"(%h, %s) ({
      ^bb0(%element: f32, %scale: f32):
        %scaled = arith.mulf %element, %scale : f32
        "shuttle.yield"(%scaled) : (f32) -> ()
      }) {
        indexing_maps = [#identity, #prefix, #identity],
        source = #shuttle.source_ref<0, 0, 0, 9>
      } : (tensor<2x3x11xf32>, tensor<2x3xf32>) -> tensor<2x3x11xf32>
      %y = "shuttle.contract"(%z, %region_projection) {
        accumulator_types = [f32], algorithm = "dot_general",
        indexing_maps = [#contract1_lhs, #contract1_rhs, #contract1_out],
        iterator_kinds = ["parallel", "parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 13>
      } : (tensor<2x3x11xf32>, tensor<11x13xf32>) -> tensor<2x3x13xf32>
      "shuttle.yield"(%y) : (tensor<2x3x13xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 4>,
                     #shuttle.source_ref<0, 0, 0, 6>,
                     #shuttle.source_ref<0, 0, 0, 9>,
                     #shuttle.source_ref<0, 0, 0, 13>]
    } : (tensor<2x3x4xf32>, tensor<4x11xf32>, tensor<11x13xf32>, tensor<f32>)
        -> tensor<2x3x13xf32>
    return %result : tensor<2x3x13xf32>
  }
}
