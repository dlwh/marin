// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Identity and tanh-composed scalar score Maps both reach the same generic
// Contract/Fold schedule and verified executable boundary. The scalar AST is
// retained in device identity; it is not a physical mode.
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-online-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.identity
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-f32-causal-gqa-softcap-score-axis4.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-test-set-prefer-fusion-tuning --shuttle-canonicalize --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-plan-source-event-lowering --shuttle-build-gpu-executable-bundle --shuttle-verify-gpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint -o /dev/null > %t.tanh
// RUN: not diff %t.identity %t.tanh

module {}
