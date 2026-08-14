// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-test-opt %S/Inputs/abi20-projected-parameter-broadcast-exclusions.mlir --split-input-file --shuttle-gpu-executable-bundle-pipeline 2>&1 | FileCheck %s

// CHECK-COUNT-10: requires a bounded set of independently connected static GPU algebra regions
