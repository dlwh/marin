// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module {
  func.func public @output_dependency_mutation(
      %input: tensor<2x3xf32>, %alternate: tensor<2x3xf32>)
      -> tensor<2x3xf32> {
    %sum = stablehlo.add %input, %alternate : tensor<2x3xf32>
    %result = stablehlo.multiply %sum, %alternate : tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }
}
