// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/bf16-contract-algorithm-exclusions.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s
// RUN: shuttle-test-opt %S/Inputs/bf16-contract-algorithm-exclusions.mlir --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.original
// RUN: shuttle-test-opt %S/Inputs/bf16-contract-algorithm-exclusions.mlir --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.lowered
// RUN: diff %t.original %t.lowered

// Each unsupported algorithm remains a source dot, while the following negate
// is selected. Full fingerprint equality also preserves algorithm fields.
// CHECK-NOT: "shuttle.contract"
// CHECK: "stablehlo.dot_general"
// CHECK-SAME: accumulation_type = f32
// CHECK-SAME: allow_imprecise_accumulation = false
// CHECK: "shuttle.map"
// CHECK: "arith.negf"
// CHECK-NOT: "shuttle.contract"
// CHECK: "stablehlo.dot_general"
// CHECK-SAME: accumulation_type = f32
// CHECK-SAME: allow_imprecise_accumulation = false
// CHECK: "shuttle.map"
// CHECK: "arith.negf"
// CHECK-NOT: "shuttle.contract"
// CHECK: shuttle.coverage_manifest
// CHECK-SAME: complete = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 1, 0>, #shuttle.source_ref<1, 0, 0, 0>, #shuttle.source_ref<1, 0, 1, 0>]
// CHECK-SAME: reason = "unsupported_operation", source = #shuttle.source_ref<0, 0, 0, 0>
// CHECK-SAME: reason = "unsupported_operation", source = #shuttle.source_ref<1, 0, 0, 0>
// CHECK-SAME: selected_regions = {{\[\[}}#shuttle.source_ref<0, 0, 1, 0>], [#shuttle.source_ref<1, 0, 1, 0>]]
