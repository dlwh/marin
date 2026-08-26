// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX source coverage and public-output closure are prerequisites.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage -o /dev/null | FileCheck %s --check-prefix=COVERAGE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-softcap-gqa-online-axis1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage -o /dev/null | FileCheck %s --check-prefix=COVERAGE

// FAST may select any dependence-driven plan that carries the full semantic
// cut, implements the online {m,l,o} state, avoids a representative-score
// temporary, and improves both temporary bytes and traffic without increasing
// launch cost. No task, launch, slot, grid, or algebra-op count is observable.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-verify-online-sufficient-state -o /dev/null | FileCheck %s --check-prefix=ONLINE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-softcap-gqa-online-axis1.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-verify-online-sufficient-state -o /dev/null | FileCheck %s --check-prefix=ONLINE

// Symbol and SSA spellings cannot participate in recognition.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-verify-online-sufficient-state -o /dev/null | FileCheck %s --check-prefix=ONLINE
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-softcap-gqa-online-axis1.mlir --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-verify-online-sufficient-state -o /dev/null | FileCheck %s --check-prefix=ONLINE

// Scale and softcap change semantic identity but not generic eligibility.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-softcap-gqa-online-axis1-mutation.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-verify-online-sufficient-state -o /dev/null | FileCheck %s --check-prefix=ONLINE

// A public representative-score result is an unsupported near miss. It must
// remain on a complete materialized fallback instead of being misclassified.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-public-score-near-miss.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-verify-online-fallback -o /dev/null | FileCheck %s --check-prefix=FALLBACK

// A normalized contraction without maximum-state semantics is also a generic
// fallback, even though its remaining dataflow is structurally similar.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-no-maximum-near-miss.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-verify-online-fallback -o /dev/null | FileCheck %s --check-prefix=FALLBACK

// Maximum and additive state updates must reduce the same score coordinate.
// Equal query/key extents make the wrong-axis computation shape-correct, so
// only semantic-coordinate checking distinguishes this generic fallback.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-mismatched-reduction-near-miss.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-verify-online-fallback -o /dev/null | FileCheck %s --check-prefix=FALLBACK

// A maximum operation that remains represented but no longer feeds the Fold
// yield is not maximum-state semantics and must retain the generic fallback.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --shuttle-test-verify-output-dependency-coverage --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-test-rewire-online-maximum-yield --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-verify-online-fallback -o /dev/null | FileCheck %s --check-prefix=FALLBACK

// SOURCE_ORDERED retains the exact StableHLO roundtrip for both layouts; it
// is not required to use, or forbidden from using, any planner topology.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.axis4.raw
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.axis4.roundtrip
// RUN: diff %t.axis4.raw %t.axis4.roundtrip
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-softcap-gqa-online-axis1.mlir --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.axis1.raw
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-softcap-gqa-online-axis1.mlir --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.axis1.roundtrip
// RUN: diff %t.axis1.raw %t.axis1.roundtrip
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-offset-softcap-gqa-online-axis1-mutation.mlir --shuttle-test-report-normalized-fingerprint -o /dev/null > %t.mutation.raw
// RUN: not diff %t.axis1.raw %t.mutation.raw

// COVERAGE: output_dependency_coverage = complete
// ONLINE: online_state = {m,l,o}; source_cut = complete; representative_score_materialized = false; temporary_bytes = improved; temporary_traffic = improved
// FALLBACK: online_state = fallback
