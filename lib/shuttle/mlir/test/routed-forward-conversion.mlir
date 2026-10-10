// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-routed-forward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-canonicalize --mlir-print-op-generic | FileCheck %s

// CHECK-DAG: "shuttle.selection"{{.*}}comparison = #shuttle.selection_comparison<total_order>{{.*}}stable = true{{.*}}tie_break = #shuttle.selection_tie_break<source_index_ascending>{{.*}}(tensor<2x2xbf16>) -> (tensor<2x1xbf16>, tensor<2x1xi32>)
// CHECK-DAG: "shuttle.fold"
// CHECK-DAG: "shuttle.relation"({{.*}}semantics = #shuttle.relation_semantics<gather>
// CHECK-NOT: moe
