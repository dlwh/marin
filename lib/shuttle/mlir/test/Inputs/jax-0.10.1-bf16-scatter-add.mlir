// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module @jit_scatter_add attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%operand: tensor<3x2xbf16>, %indices: tensor<2x1xi32>, %updates: tensor<2x2xbf16>) -> tensor<3x2xbf16> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%lhs: tensor<bf16>, %rhs: tensor<bf16>):
      %sum = stablehlo.add %lhs, %rhs : tensor<bf16>
      stablehlo.return %sum : tensor<bf16>
    }) : (tensor<3x2xbf16>, tensor<2x1xi32>, tensor<2x2xbf16>) -> tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}
