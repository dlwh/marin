// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/f32-forward-routing-relations.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=ALGEBRA
// RUN: shuttle-test-opt %S/Inputs/f32-forward-routing-relations.mlir --shuttle-stablehlo-source-ordered-pipeline | FileCheck %s --check-prefix=LOWERED
// RUN: shuttle-test-opt %S/Inputs/f32-forward-routing-relations.mlir --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.source
// RUN: shuttle-test-opt %S/Inputs/f32-forward-routing-relations.mlir --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.roundtrip
// RUN: diff %t.source %t.roundtrip

// ALGEBRA: "shuttle.relation"
// ALGEBRA-SAME: semantics = #shuttle.relation_semantics<slice>
// ALGEBRA: "shuttle.relation"
// ALGEBRA-SAME: semantics = #shuttle.relation_semantics<transpose>
// ALGEBRA: "shuttle.relation"
// ALGEBRA-SAME: semantics = #shuttle.relation_semantics<select>
// ALGEBRA: "shuttle.relation"
// ALGEBRA-SAME: semantics = #shuttle.relation_semantics<multiply>
// ALGEBRA: "shuttle.relation"
// ALGEBRA-SAME: semantics = #shuttle.relation_semantics<reshape>
// ALGEBRA: "shuttle.relation"
// ALGEBRA-SAME: collapsed_slice_dims = array<i64: 1>
// ALGEBRA-SAME: index_vector_dim = 2 : i64
// ALGEBRA-SAME: indices_are_sorted = false
// ALGEBRA-SAME: offset_dims = array<i64: 2>
// ALGEBRA-SAME: operand_batching_dims = array<i64: 0>
// ALGEBRA-SAME: slice_sizes = array<i64: 1, 1, 3>
// ALGEBRA-SAME: start_index_map = array<i64: 1>
// ALGEBRA-SAME: start_indices_batching_dims = array<i64: 0>
// ALGEBRA-SAME: semantics = #shuttle.relation_semantics<gather>
// ALGEBRA: "shuttle.map"
// ALGEBRA-SAME: semantics = #shuttle.map_semantics<transpose>

// LOWERED: stablehlo.slice
// LOWERED: stablehlo.transpose
// LOWERED: stablehlo.select
// LOWERED: stablehlo.multiply
// LOWERED: stablehlo.reshape
// LOWERED: "stablehlo.gather"
// LOWERED-SAME: dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>
// LOWERED-SAME: indices_are_sorted = false
// LOWERED-SAME: slice_sizes = array<i64: 1, 1, 3>
