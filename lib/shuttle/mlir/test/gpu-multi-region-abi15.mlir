// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=PLAN
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s --check-prefix=CONTRACTS
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=SOURCE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir --shuttle-gpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=FAST
// RUN: shuttle-test-opt %s --shuttle-gpu-executable-bundle-pipeline | FileCheck %s --check-prefix=IDENTICAL

// PLAN: "shuttle.materialization_plan"
// PLAN-COUNT-4: "shuttle.materialization_buffer"
// PLAN-COUNT-2: "shuttle.materialization_task"
// PLAN: "shuttle.materialization_plan_yield"
// PLAN: "shuttle.materialization_plan"
// PLAN-COUNT-13: "shuttle.materialization_buffer"
// PLAN-COUNT-10: "shuttle.materialization_task"
// PLAN: "shuttle.materialization_plan_yield"

// CONTRACTS-COUNT-3: kind = #shuttle.materialization_task_kind<contract>

// SOURCE-LABEL: module @jit_composed
// SOURCE: %[[FORWARD:[0-9]+]] = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// SOURCE-SAME: api_version = 4 : i32
// SOURCE: %[[BACKWARD:[0-9]+]]:2 = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// SOURCE-SAME: api_version = 4 : i32
// SOURCE: return %[[FORWARD]], %[[BACKWARD]]#1, %[[BACKWARD]]#0
// SOURCE-NOT: shuttle.
// SOURCE-NOT: stablehlo.dot_general

// FAST-LABEL: module @jit_composed
// FAST: %[[FORWARD:[0-9]+]] = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// FAST-SAME: api_version = 4 : i32
// FAST: %[[BACKWARD:[0-9]+]]:2 = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// FAST-SAME: api_version = 4 : i32
// FAST: return %[[FORWARD]], %[[BACKWARD]]#1, %[[BACKWARD]]#0
// FAST-NOT: shuttle.
// FAST-NOT: stablehlo.dot_general

// IDENTICAL-LABEL: module @jit_identical_regions
// IDENTICAL: %[[FIRST:[0-9]+]] = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// IDENTICAL: %[[SECOND:[0-9]+]] = stablehlo.custom_call @shuttle.gpu.executable_bundle.v2
// IDENTICAL: return %[[FIRST]], %[[SECOND]]
// IDENTICAL-NOT: shuttle.
// IDENTICAL-NOT: stablehlo.dot_general

module @jit_identical_regions attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<17x33xbf16>, %arg1: tensor<33x19xbf16>) -> (tensor<17x19xbf16>, tensor<17x19xbf16>) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<17x33xbf16>, tensor<33x19xbf16>) -> tensor<17x19xf32>
    %1 = stablehlo.convert %0 : (tensor<17x19xf32>) -> tensor<17x19xbf16>
    %2 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<17x33xbf16>, tensor<33x19xbf16>) -> tensor<17x19xf32>
    %3 = stablehlo.convert %2 : (tensor<17x19xf32>) -> tensor<17x19xbf16>
    return %1, %3 : tensor<17x19xbf16>, tensor<17x19xbf16>
  }
}
