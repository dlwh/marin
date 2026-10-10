// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// The physical capability is the canonical Contract/Fold region carried by
// the materialization plan. Scalar score refactorings and symbol spellings
// change semantic identity without changing the two-task ownership cut.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-softcap-gqa-online-axis1-mutation.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic | FileCheck %s

// CHECK: "shuttle.materialization_plan"() <{{.*}}normalized_exp_contract_regions = [
// CHECK: "shuttle.materialization_task"() <{{.*}}composition_kind = "contract_fold"
// CHECK: "shuttle.materialization_task"() <{{.*}}composition_kind = "map_contract"
// CHECK-NOT: online_state

module {}
