// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#identity = affine_map<(m, n) -> (m, n)>
#row = affine_map<(m, n) -> (m)>
#contract_lhs = affine_map<(m, o, n) -> (m, n)>
#contract_rhs = affine_map<(m, o, n) -> (n, o)>
#contract_out = affine_map<(m, o, n) -> (m, o)>

module @input_producer attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @entry(%x: tensor<2x5xf32>, %p: tensor<5x7xf32>,
      %init: tensor<f32>) -> tensor<2x7xf32> {
    %result = "shuttle.region"(%x, %p, %init) ({
    ^bb0(%region_x: tensor<2x5xf32>, %region_p: tensor<5x7xf32>,
         %region_init: tensor<f32>):
      %state = "shuttle.fold"(%region_x, %region_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {

        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>, order_free = true,
        reduction_dimensions = array<i64: 1>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x5xf32>, tensor<f32>) -> tensor<2xf32>
      %scaled = "shuttle.map"(%region_x, %state) ({
      ^bb0(%element: f32, %scale: f32):
        %value = arith.mulf %element, %scale : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {
        indexing_maps = [#identity, #row, #identity],
        source = #shuttle.source_ref<0, 0, 1, 0>
      } : (tensor<2x5xf32>, tensor<2xf32>) -> tensor<2x5xf32>
      %output = "shuttle.contract"(%scaled, %region_p) {
        accumulator_types = [f32], algorithm = "dot_general",
        indexing_maps = [#contract_lhs, #contract_rhs, #contract_out],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 2, 0>
      } : (tensor<2x5xf32>, tensor<5x7xf32>) -> tensor<2x7xf32>
      "shuttle.yield"(%output) : (tensor<2x7xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>,
                     #shuttle.source_ref<0, 0, 1, 0>,
                     #shuttle.source_ref<0, 0, 2, 0>]
    } : (tensor<2x5xf32>, tensor<5x7xf32>, tensor<f32>) -> tensor<2x7xf32>
    return %result : tensor<2x7xf32>
  }
}
