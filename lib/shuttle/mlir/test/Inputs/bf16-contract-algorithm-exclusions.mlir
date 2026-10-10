// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module {
  func.func @bf16_explicit_algorithm(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x4xbf16>) -> tensor<2x4xbf16> {
    %0 = "stablehlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
      algorithm = #stablehlo.dot_algorithm<lhs_precision_type = bf16, rhs_precision_type = bf16, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
    } : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
    %1 = stablehlo.negate %0 : tensor<2x4xbf16>
    return %1 : tensor<2x4xbf16>
  }
  func.func @f32_explicit_algorithm(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = "stablehlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
      algorithm = #stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %1 = stablehlo.negate %0 : tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}
