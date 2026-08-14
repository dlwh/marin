// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=PLAN
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=SOURCE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=FAST

// PLAN-COUNT-2: "shuttle.materialization_plan"
// PLAN-COUNT-17: "shuttle.materialization_buffer"
// PLAN-COUNT-12: "shuttle.materialization_task"
// PLAN-COUNT-3: kind = #shuttle.materialization_task_kind<contract>

// SOURCE-LABEL: module @jit_composed
// SOURCE-COUNT-2: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// SOURCE-COUNT-2: api_version = 4 : i32
// SOURCE: return %{{.*}}#0, %{{.*}}#1, %{{.*}}#0
// SOURCE-NOT: shuttle.
// SOURCE-NOT: stablehlo.dot_general

// FAST-LABEL: module @jit_composed
// FAST-COUNT-2: stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// FAST-COUNT-2: api_version = 4 : i32
// FAST: return %{{.*}}#0, %{{.*}}#1, %{{.*}}#0
// FAST-NOT: shuttle.
// FAST-NOT: stablehlo.dot_general
