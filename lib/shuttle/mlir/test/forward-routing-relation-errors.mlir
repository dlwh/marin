// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --split-input-file --verify-diagnostics %s

module {
  func.func @bad_slice(%arg0: tensor<2x4xi32>) {
    %region = "shuttle.region"(%arg0) <{
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    }> ({
    ^bb0(%input: tensor<2x4xi32>):
      // expected-error @+1 {{has invalid static slice bounds or result shape}}
      %0 = "shuttle.relation"(%input) <{
        parameters = [array<i64: 0, 0>, array<i64: 2, 2>,
                      array<i64: 1, 1>],
        semantics = #shuttle.relation_semantics<slice>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      }> : (tensor<2x4xi32>) -> tensor<2x3xi32>
      "shuttle.yield"(%0) : (tensor<2x3xi32>) -> ()
    }) : (tensor<2x4xi32>) -> tensor<2x3xi32>
    return
  }
}

// -----

module {
  func.func @overlapping_gather_dimensions(
      %data: tensor<2x4x3xf32>, %indices: tensor<2x2x1xi32>) {
    %region = "shuttle.region"(%data, %indices) <{
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    }> ({
    ^bb0(%input: tensor<2x4x3xf32>, %starts: tensor<2x2x1xi32>):
      // expected-error @+1 {{requires collapsed-slice and start-index dimensions disjoint from operand-batching dimensions}}
      %0 = "shuttle.relation"(%input, %starts) <{
        parameters = [{
          collapsed_slice_dims = array<i64: 1>,
          index_vector_dim = 2 : i64,
          indices_are_sorted = false,
          offset_dims = array<i64: 2>,
          operand_batching_dims = array<i64: 0>,
          slice_sizes = array<i64: 1, 1, 3>,
          start_index_map = array<i64: 0>,
          start_indices_batching_dims = array<i64: 0>
        }],
        semantics = #shuttle.relation_semantics<gather>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      }> : (tensor<2x4x3xf32>, tensor<2x2x1xi32>) -> tensor<2x2x3xf32>
      "shuttle.yield"(%0) : (tensor<2x2x3xf32>) -> ()
    }) : (tensor<2x4x3xf32>, tensor<2x2x1xi32>) -> tensor<2x2x3xf32>
    return
  }
}

// -----

module {
  func.func @bad_gather_result(
      %data: tensor<2x4x3xf32>, %indices: tensor<2x2x1xi32>) {
    %region = "shuttle.region"(%data, %indices) <{
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    }> ({
    ^bb0(%input: tensor<2x4x3xf32>, %starts: tensor<2x2x1xi32>):
      // expected-error @+1 {{has a gather result with an incorrect shape}}
      %0 = "shuttle.relation"(%input, %starts) <{
        parameters = [{
          collapsed_slice_dims = array<i64: 1>,
          index_vector_dim = 2 : i64,
          indices_are_sorted = false,
          offset_dims = array<i64: 2>,
          operand_batching_dims = array<i64: 0>,
          slice_sizes = array<i64: 1, 1, 3>,
          start_index_map = array<i64: 1>,
          start_indices_batching_dims = array<i64: 0>
        }],
        semantics = #shuttle.relation_semantics<gather>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      }> : (tensor<2x4x3xf32>, tensor<2x2x1xi32>) -> tensor<2x2x2xf32>
      "shuttle.yield"(%0) : (tensor<2x2x2xf32>) -> ()
    }) : (tensor<2x4x3xf32>, tensor<2x2x1xi32>) -> tensor<2x2x2xf32>
    return
  }
}
