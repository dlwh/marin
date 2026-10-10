// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-scatter-add.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic | FileCheck %s
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-scatter-add.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=PLAN

// CHECK: "shuttle.region"
// CHECK: "shuttle.relation"{{.*}}semantics = #shuttle.relation_semantics<indexed_contributions>{{.*}}source_group = {member = 0 : i64, size = 4 : i64}
// CHECK: "shuttle.map"{{.*}}source_group = {member = 1 : i64, size = 4 : i64}
// CHECK: "arith.constant"{{.*}}value = 0.000000e+00 : bf16
// CHECK: "shuttle.fold"{{.*}}order_free = false{{.*}}reduction_dimensions = array<i64: 2>{{.*}}source_group = {member = 2 : i64, size = 4 : i64}
// CHECK: "arith.addf"
// CHECK: "shuttle.map"{{.*}}source_group = {member = 3 : i64, size = 4 : i64}
// CHECK: "arith.addf"
// CHECK-NOT: moe

// PLAN: "shuttle.materialization_task"() <{composition_kind = "indexed_accumulation", composition_owner_dimensions = array<i64: 0, 1>, composition_sources = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 0>], composition_stream_dimensions = array<i64: 2>, dependencies = array<i64>, domain_shape = array<i64: 3, 2, 2>, input_buffers = array<i64: 0, 1, 2>, kind = #shuttle.materialization_task_kind<fold>, order_free = false, ordinal = 0 : i64, output_buffers = array<i64: 3>, reduction_dimensions = array<i64: 2>, semantic_fingerprint = {{.*}}, source = #shuttle.source_ref<0, 0, 0, 0>, source_expansion_group = [{member = 0 : i64, size = 4 : i64}, {member = 1 : i64, size = 4 : i64}, {member = 2 : i64, size = 4 : i64}, {member = 3 : i64, size = 4 : i64}]}> : () -> ()
// PLAN-NOT: "shuttle.materialization_task"
// PLAN: "shuttle.schedule_task"() <{composition_owner_dimensions = array<i64: 0, 1>, composition_stream_dimensions = array<i64: 2>{{.*}}kind = #shuttle.schedule_task_kind<row_fold>{{.*}}reduction_axis = 2 : i64{{.*}}tile_shape = array<i64: 1, 1, 2>
// PLAN-NOT: moe
