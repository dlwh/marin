// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Anonymous algebra fixture for Fold(x); Map(x, state0); Fold(mapped);
// Map(mapped, state1); Contract. The mapped tensor is recomputed in the
// consumer task; only the compact Fold states cross the task boundary.

#identity = affine_map<(m, n) -> (m, n)>
#row = affine_map<(m, n) -> (m)>
#contract_lhs = affine_map<(m, o, n) -> (m, n)>
#contract_rhs = affine_map<(m, o, n) -> (n, o)>
#contract_out = affine_map<(m, o, n) -> (m, o)>

module @anonymous_two_fold_graph attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @entry(%x: tensor<5x19xf32>, %p: tensor<19x7xf32>) -> tensor<5x7xf32> {
    %result = "shuttle.region"(%x, %p) ({
    ^bb0(%region_x: tensor<5x19xf32>, %region_p: tensor<19x7xf32>):
      %zero = "shuttle.map"() ({
      ^bb0:
        %value = arith.constant 0.0 : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {
        indexing_maps = [affine_map<() -> ()>],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : () -> tensor<f32>
      %state0 = "shuttle.fold"(%region_x, %zero) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        indexing_maps = [#identity, #row],
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>, order_free = true,
        reduction_dimensions = array<i64: 1>,
        source = #shuttle.source_ref<0, 0, 1, 0>
      } : (tensor<5x19xf32>, tensor<f32>) -> tensor<5xf32>
      %centered = "shuttle.map"(%region_x, %state0) ({
      ^bb0(%element: f32, %state: f32):
        %value = arith.subf %element, %state : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {
        indexing_maps = [#identity, #row, #identity],
        source = #shuttle.source_ref<0, 0, 2, 0>
      } : (tensor<5x19xf32>, tensor<5xf32>) -> tensor<5x19xf32>
      %squared = "shuttle.map"(%centered) ({
      ^bb0(%element: f32):
        %value = arith.mulf %element, %element : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {
        indexing_maps = [#identity, #identity],
        source = #shuttle.source_ref<0, 0, 3, 0>
      } : (tensor<5x19xf32>) -> tensor<5x19xf32>
      %state1 = "shuttle.fold"(%squared, %zero) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        indexing_maps = [#identity, #row],
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>, order_free = true,
        reduction_dimensions = array<i64: 1>,
        source = #shuttle.source_ref<0, 0, 4, 0>
      } : (tensor<5x19xf32>, tensor<f32>) -> tensor<5xf32>
      %normalized = "shuttle.map"(%centered, %state1) ({
      ^bb0(%element: f32, %state: f32):
        %value = arith.divf %element, %state : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {
        indexing_maps = [#identity, #row, #identity],
        source = #shuttle.source_ref<0, 0, 5, 0>
      } : (tensor<5x19xf32>, tensor<5xf32>) -> tensor<5x19xf32>
      %output = "shuttle.contract"(%normalized, %region_p) {
        accumulator_types = [f32], algorithm = "dot_general",
        indexing_maps = [#contract_lhs, #contract_rhs, #contract_out],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 6, 0>
      } : (tensor<5x19xf32>, tensor<19x7xf32>) -> tensor<5x7xf32>
      "shuttle.yield"(%output) : (tensor<5x7xf32>) -> ()
    }) {
      policy = #shuttle.policy<fast>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>,
                     #shuttle.source_ref<0, 0, 1, 0>,
                     #shuttle.source_ref<0, 0, 2, 0>,
                     #shuttle.source_ref<0, 0, 3, 0>,
                     #shuttle.source_ref<0, 0, 4, 0>,
                     #shuttle.source_ref<0, 0, 5, 0>,
                     #shuttle.source_ref<0, 0, 6, 0>]
    } : (tensor<5x19xf32>, tensor<19x7xf32>) -> tensor<5x7xf32>
    return %result : tensor<5x7xf32>
  }
}
