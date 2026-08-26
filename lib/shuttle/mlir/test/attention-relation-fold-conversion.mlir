// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// This RED stops at lossless algebra conversion. It deliberately does not
// constrain region, operation, task, launch, slot, grid, or kernel counts.

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-relation-max-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=AXIS4
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=AXIS1
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1-mutation.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=AXIS1

// Complete coverage is not enough if selected semantics are dead. Require the
// union of transitive public-output dependencies to equal manifest.complete.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-relation-max-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-verify-output-dependency-coverage -o /dev/null | FileCheck %s --check-prefix=DEPENDENCIES
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-verify-output-dependency-coverage -o /dev/null | FileCheck %s --check-prefix=DEPENDENCIES
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1-mutation.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-verify-output-dependency-coverage -o /dev/null | FileCheck %s --check-prefix=DEPENDENCIES

// Source-ordered conversion is a lossless StableHLO roundtrip. The FAST named
// pipeline must also preserve the public boundary and validate its own policy
// identity; it may later choose a semantically legal online-state form.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-relation-max-axis4.mlir --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.axis4-source
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-relation-max-axis4.mlir --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.axis4-roundtrip
// RUN: diff %t.axis4-source %t.axis4-roundtrip
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-relation-max-axis4.mlir --shuttle-stablehlo-fast-pipeline | FileCheck %s --check-prefix=AXIS4-BOUNDARY
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1.mlir --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.axis1-source
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1.mlir --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.axis1-roundtrip
// RUN: diff %t.axis1-source %t.axis1-roundtrip
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1.mlir --shuttle-stablehlo-fast-pipeline | FileCheck %s --check-prefix=AXIS1-BOUNDARY

// Symbol alpha-renaming cannot affect the generic algebra identity. Changing
// scale and softcap is semantic and must affect it.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.primary
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.renamed
// RUN: diff %t.primary %t.renamed
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1-mutation.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.mutated
// RUN: not diff %t.primary %t.mutated

// A represented but dead selected operation must not be hidden by aggregate
// region provenance. The positive uses the same verifier before rewiring.
// RUN: shuttle-test-opt %S/Inputs/output-dependency-mutation.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-verify-output-dependency-coverage -o /dev/null | FileCheck %s --check-prefix=DEPENDENCIES
// RUN: not shuttle-test-opt %S/Inputs/output-dependency-mutation.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-rewire-return --shuttle-test-verify-output-dependency-coverage -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEAD

// A selected scalar combiner operation must likewise reach the exact Fold
// result yield; merely remaining present in the combiner is not sufficient.
// RUN: shuttle-test-opt %S/Inputs/output-dependency-fold-mutation.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-verify-output-dependency-coverage -o /dev/null | FileCheck %s --check-prefix=DEPENDENCIES
// RUN: not shuttle-test-opt %S/Inputs/output-dependency-fold-mutation.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-rewire-fold-yield --shuttle-test-verify-output-dependency-coverage -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEAD

// Relation is generic index-domain algebra. The check intentionally permits
// one lossless materialized relation or another equivalent representation.
// AXIS4-DAG: policy = #shuttle.policy<source_ordered>
// AXIS4-DAG: "shuttle.relation"
// AXIS4-DAG: #shuttle.relation_semantics<compare>
// AXIS4-DAG: parameters = ["SIGNED", "LE"]
// AXIS4-DAG: "shuttle.fold"{{.*}}order_free = false{{.*}}reduction_dimensions = array<i64: 4>
// AXIS4-DAG: "arith.maximumf"
// AXIS4-DAG: order_free = true
// AXIS4-DAG: "arith.addf"
// AXIS4-DAG: shuttle.coverage_manifest
// AXIS4-DAG: excluded = []
// AXIS4-DAG: policy = "source_ordered"
// AXIS4-DAG: policy_digest = "{{[0-9a-f]+}}"

// AXIS1-DAG: policy = #shuttle.policy<source_ordered>
// AXIS1-DAG: "shuttle.relation"
// AXIS1-DAG: #shuttle.relation_semantics<compare>
// AXIS1-DAG: parameters = ["SIGNED", "LE"]
// AXIS1-DAG: "shuttle.fold"{{.*}}order_free = false{{.*}}reduction_dimensions = array<i64: 1>
// AXIS1-DAG: "arith.maximumf"
// AXIS1-DAG: order_free = true
// AXIS1-DAG: "arith.addf"
// AXIS1-DAG: shuttle.coverage_manifest
// AXIS1-DAG: excluded = []
// AXIS1-DAG: policy = "source_ordered"
// AXIS1-DAG: policy_digest = "{{[0-9a-f]+}}"

// DEPENDENCIES: output_dependency_coverage = complete
// DEAD: covered source results do not form a complete public-output dependency trace

// AXIS4-BOUNDARY: func.func public @main
// AXIS4-BOUNDARY-SAME: tensor<1x2x2x17x17xf32>
// AXIS4-BOUNDARY-SAME: tensor<1x2x2x17xf32>
// AXIS4-BOUNDARY-SAME: tensor<1x1x1x17xf32>
// AXIS4-BOUNDARY-NOT: shuttle.

// AXIS1-BOUNDARY: func.func public @main
// AXIS1-BOUNDARY-SAME: tensor<19x23x2x3x2xf32>
// AXIS1-BOUNDARY-SAME: tensor<19x2x3x2xf32>
// AXIS1-BOUNDARY-SAME: tensor<19x1x1x1xf32>
// AXIS1-BOUNDARY-NOT: shuttle.
