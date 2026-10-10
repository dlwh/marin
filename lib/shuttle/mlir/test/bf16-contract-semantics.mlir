// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/bf16-contract-semantics.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=ALGEBRA
// RUN: shuttle-test-opt %S/Inputs/bf16-contract-semantics.mlir --shuttle-stablehlo-source-ordered-pipeline | FileCheck %s --check-prefix=LOWERED
// RUN: shuttle-test-opt %S/Inputs/bf16-contract-semantics.mlir --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.original
// RUN: shuttle-test-opt %S/Inputs/bf16-contract-semantics.mlir --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.lowered
// RUN: diff %t.original %t.lowered
// RUN: shuttle-test-opt %S/Inputs/bf16-contract-semantics.mlir --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.fast
// RUN: diff %t.original %t.fast

// Source accumulation follows each result type. It is not a declaration of
// physical FP32 accumulation. Exact fingerprint equality above also checks
// batching/contraction dimension order, precision arrays, types, and operands.
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: accumulator_types = [bf16]
// ALGEBRA-SAME: precision = ["DEFAULT", "DEFAULT"]
// ALGEBRA-SAME: source = #shuttle.source_ref<0, 0, 0, 0>
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: accumulator_types = [bf16]
// ALGEBRA-SAME: precision = ["HIGH", "HIGH"]
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: accumulator_types = [bf16]
// ALGEBRA-SAME: precision = ["HIGHEST", "HIGHEST"]
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: accumulator_types = [bf16]
// ALGEBRA-SAME: precision = ["HIGH", "HIGHEST"]
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: accumulator_types = [f32]
// ALGEBRA-SAME: precision = ["DEFAULT", "DEFAULT"]
// ALGEBRA-SAME: source = #shuttle.source_ref<4, 0, 0, 0>
// ALGEBRA: "shuttle.map"
// ALGEBRA-SAME: source = #shuttle.source_ref<4, 0, 1, 0>
// ALGEBRA: "shuttle.scalar_convert"
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: accumulator_types = [f32]
// ALGEBRA-SAME: precision = ["HIGH", "DEFAULT"]
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: accumulator_types = [bf16]
// ALGEBRA-SAME: precision = ["DEFAULT", "DEFAULT"]
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: accumulator_types = [bf16]
// ALGEBRA-SAME: precision = ["DEFAULT", "DEFAULT"]
// ALGEBRA-SAME: source_precision_form = "omitted"
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: accumulator_types = [bf16]
// ALGEBRA-SAME: precision = ["DEFAULT", "DEFAULT"]
// ALGEBRA-SAME: source_precision_form = "empty"
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: accumulator_types = [f32]
// ALGEBRA-SAME: precision = ["DEFAULT", "DEFAULT"]
// ALGEBRA-SAME: source_precision_form = "omitted"
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: accumulator_types = [f32]
// ALGEBRA-SAME: precision = ["DEFAULT", "DEFAULT"]
// ALGEBRA-SAME: source_precision_form = "empty"
// ALGEBRA: shuttle.coverage_manifest
// ALGEBRA-SAME: complete = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<1, 0, 0, 0>, #shuttle.source_ref<2, 0, 0, 0>, #shuttle.source_ref<3, 0, 0, 0>, #shuttle.source_ref<4, 0, 0, 0>, #shuttle.source_ref<4, 0, 1, 0>, #shuttle.source_ref<5, 0, 0, 0>, #shuttle.source_ref<6, 0, 0, 0>, #shuttle.source_ref<7, 0, 0, 0>, #shuttle.source_ref<8, 0, 0, 0>, #shuttle.source_ref<9, 0, 0, 0>, #shuttle.source_ref<10, 0, 0, 0>]
// ALGEBRA-SAME: excluded = []

// LOWERED-LABEL: func.func @bf16_default
// LOWERED: stablehlo.dot_general
// LOWERED-SAME: precision = [DEFAULT, DEFAULT]
// LOWERED-SAME: -> tensor<2x4xbf16>
// LOWERED-LABEL: func.func @bf16_high_batched
// LOWERED: batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [HIGH, HIGH]
// LOWERED-LABEL: func.func @bf16_highest_reordered
// LOWERED: batching_dims = [3] x [1], contracting_dims = [2, 1] x [0, 2], precision = [HIGHEST, HIGHEST]
// LOWERED-LABEL: func.func @bf16_asymmetric_precision
// LOWERED: precision = [HIGH, HIGHEST]
// LOWERED-LABEL: func.func @f32_dot_then_convert
// LOWERED: %[[DOT:.*]] = stablehlo.dot_general
// LOWERED-SAME: -> tensor<2x4xf32>
// LOWERED-NEXT: %[[CONVERT:.*]] = stablehlo.convert %[[DOT]] : (tensor<2x4xf32>) -> tensor<2x4xbf16>
// LOWERED-NEXT: return %[[CONVERT]]
// LOWERED-LABEL: func.func @f32_high_precision
// LOWERED: precision = [HIGH, DEFAULT]
// LOWERED-LABEL: func.func @bf16_scalar_result
// LOWERED: stablehlo.dot_general
// LOWERED-SAME: -> tensor<bf16>
// LOWERED-LABEL: func.func @bf16_omitted_precision
// LOWERED: stablehlo.dot_general
// LOWERED-SAME: contracting_dims = [1] x [0] :
// LOWERED-SAME: -> tensor<2x4xbf16>
// LOWERED-LABEL: func.func @bf16_empty_precision
// LOWERED: stablehlo.dot_general
// LOWERED-SAME: contracting_dims = [1] x [0], precision = [] :
// LOWERED-SAME: -> tensor<2x4xbf16>
// LOWERED-LABEL: func.func @f32_omitted_precision
// LOWERED: stablehlo.dot_general
// LOWERED-SAME: contracting_dims = [1] x [0] :
// LOWERED-SAME: -> tensor<2x4xf32>
// LOWERED-LABEL: func.func @f32_empty_precision
// LOWERED: stablehlo.dot_general
// LOWERED-SAME: contracting_dims = [1] x [0], precision = [] :
// LOWERED-SAME: -> tensor<2x4xf32>
