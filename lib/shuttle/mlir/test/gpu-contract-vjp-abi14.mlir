// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=MATERIALIZATION
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=SOURCE-SCHEDULE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-backward.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic | FileCheck %s --check-prefix=FAST-SCHEDULE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-backward.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=SOURCE-DESCRIPTOR
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-backward.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=FAST-DESCRIPTOR

// MATERIALIZATION: "shuttle.materialization_plan"() <{{.*}}policy = #shuttle.policy<source_ordered>{{.*}}schema_version = 2
// MATERIALIZATION-COUNT-13: "shuttle.materialization_buffer"
// MATERIALIZATION-COUNT-2: kind = #shuttle.materialization_task_kind<contract>

// SOURCE-SCHEDULE: "shuttle.schedule_plan"() <{{.*}}policy = #shuttle.policy<source_ordered>{{.*}}schema_version = 2
// SOURCE-SCHEDULE-COUNT-2: reduction_order = #shuttle.schedule_reduction_order<serial_ascending_fma_rn_f32>

// FAST-SCHEDULE: "shuttle.schedule_plan"() <{{.*}}policy = #shuttle.policy<fast>{{.*}}schema_version = 2
// FAST-SCHEDULE-COUNT-2: reduction_order = #shuttle.schedule_reduction_order<leaf_order_free>

// SOURCE-DESCRIPTOR: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// SOURCE-DESCRIPTOR-SAME: api_version = 4
// SOURCE-DESCRIPTOR-SAME: bundle_schema_version = 2 : i64
// SOURCE-DESCRIPTOR-SAME: device_schema_version = 3 : i64
// SOURCE-DESCRIPTOR-SAME: invocation_abi_schema_version = 3 : i64
// SOURCE-DESCRIPTOR-SAME: transport_schema_version = 2 : i64
// SOURCE-DESCRIPTOR-NOT: shuttle.region
// SOURCE-DESCRIPTOR-NOT: stablehlo.dot_general

// FAST-DESCRIPTOR: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// FAST-DESCRIPTOR-SAME: api_version = 4
// FAST-DESCRIPTOR-NOT: shuttle.region
// FAST-DESCRIPTOR-NOT: stablehlo.dot_general
