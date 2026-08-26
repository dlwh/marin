// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// This RED stops at lossless algebra conversion. It deliberately does not
// constrain region, operation, task, launch, slot, grid, or kernel counts.

// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-relation-max-axis3.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=AXIS3
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=AXIS1
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1-mutation.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic | FileCheck %s --check-prefix=AXIS1

// Complete coverage is not enough if selected semantics are dead. Require the
// union of transitive public-output dependencies to equal manifest.complete.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-relation-max-axis3.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-verify-output-dependency-coverage -o /dev/null | FileCheck %s --check-prefix=DEPENDENCIES
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-verify-output-dependency-coverage -o /dev/null | FileCheck %s --check-prefix=DEPENDENCIES
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-causal-softcap-max-axis1-mutation.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-verify-output-dependency-coverage -o /dev/null | FileCheck %s --check-prefix=DEPENDENCIES

// Source-ordered conversion is a lossless StableHLO roundtrip. The FAST named
// pipeline must also preserve the public boundary and validate its own policy
// identity; it may later choose a semantically legal online-state form.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-relation-max-axis3.mlir --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.axis3-source
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-relation-max-axis3.mlir --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.axis3-roundtrip
// RUN: diff %t.axis3-source %t.axis3-roundtrip
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-relation-max-axis3.mlir --shuttle-stablehlo-fast-pipeline | FileCheck %s --check-prefix=AXIS3-BOUNDARY
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

// Relation is generic index-domain algebra. The check intentionally permits
// one lossless materialized relation or another equivalent representation.
// AXIS3: policy = #shuttle.policy<source_ordered>
// AXIS3: "shuttle.relation"
// AXIS3: "shuttle.fold"
// AXIS3-SAME: reduction_dimensions = array<i64: 3>
// AXIS3: "arith.maximumf"
// AXIS3: "shuttle.fold"
// AXIS3: "arith.addf"
// AXIS3: shuttle.coverage_manifest
// AXIS3-SAME: excluded = []
// AXIS3-SAME: policy = "source_ordered"
// AXIS3-SAME: policy_digest = "{{[0-9a-f]{64}}"
// AXIS3-NOT: "stablehlo.iota"
// AXIS3-NOT: "stablehlo.compare"
// AXIS3-NOT: "stablehlo.select"
// AXIS3-NOT: "stablehlo.maximum"

// AXIS1: policy = #shuttle.policy<source_ordered>
// AXIS1: "shuttle.relation"
// AXIS1: "shuttle.fold"
// AXIS1-SAME: reduction_dimensions = array<i64: 1>
// AXIS1: "arith.maximumf"
// AXIS1: "shuttle.fold"
// AXIS1: "arith.addf"
// AXIS1: shuttle.coverage_manifest
// AXIS1-SAME: excluded = []
// AXIS1-SAME: policy = "source_ordered"
// AXIS1-SAME: policy_digest = "{{[0-9a-f]{64}}"
// AXIS1-NOT: "stablehlo.iota"
// AXIS1-NOT: "stablehlo.compare"
// AXIS1-NOT: "stablehlo.select"
// AXIS1-NOT: "stablehlo.maximum"

// DEPENDENCIES: output_dependency_coverage = complete

// AXIS3-BOUNDARY: func.func public @main
// AXIS3-BOUNDARY-SAME: tensor<1x4x17x17xf32>
// AXIS3-BOUNDARY-SAME: tensor<1x4x17xf32>
// AXIS3-BOUNDARY-SAME: tensor<1x1x17xf32>
// AXIS3-BOUNDARY-NOT: shuttle.

// AXIS1-BOUNDARY: func.func public @main
// AXIS1-BOUNDARY-SAME: tensor<19x23x2x6xf32>
// AXIS1-BOUNDARY-SAME: tensor<19x2x6xf32>
// AXIS1-BOUNDARY-SAME: tensor<19x1x1xf32>
// AXIS1-BOUNDARY-NOT: shuttle.
