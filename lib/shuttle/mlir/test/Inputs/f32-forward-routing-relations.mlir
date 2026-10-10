// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module @forward_routing_relations {
  func.func public @main(
      %data: tensor<2x4x3xf32>, %indices: tensor<2x4xi32>,
      %predicate: tensor<2x2xi1>, %alternate: tensor<2x2xi32>,
      %scale: tensor<2x2xi32>) -> tensor<2x2x3xf32> {
    %sliced = stablehlo.slice %indices [0:2, 0:2]
        : (tensor<2x4xi32>) -> tensor<2x2xi32>
    %transposed = stablehlo.transpose %sliced, dims = [1, 0]
        : (tensor<2x2xi32>) -> tensor<2x2xi32>
    %selected = stablehlo.select %predicate, %transposed, %alternate
        : tensor<2x2xi1>, tensor<2x2xi32>
    %multiplied = stablehlo.multiply %selected, %scale : tensor<2x2xi32>
    %vectors = stablehlo.reshape %multiplied
        : (tensor<2x2xi32>) -> tensor<2x2x1xi32>
    %gathered = "stablehlo.gather"(%data, %vectors) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1, 3>
    }> : (tensor<2x4x3xf32>, tensor<2x2x1xi32>) -> tensor<2x2x3xf32>
    return %gathered : tensor<2x2x3xf32>
  }

  func.func public @float_transpose(%arg0: tensor<2x3xf32>)
      -> tensor<3x2xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0]
        : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}
