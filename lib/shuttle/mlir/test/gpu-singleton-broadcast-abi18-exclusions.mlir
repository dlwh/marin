// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-test-opt %S/Inputs/abi18-singleton-broadcast-exclusions.mlir --split-input-file --shuttle-gpu-executable-bundle-pipeline 2>&1 | FileCheck %s --check-prefix=ABI18
// RUN: not shuttle-test-opt %S/Inputs/abi17-rank-n-fold-exclusions.mlir --split-input-file --shuttle-gpu-executable-bundle-pipeline 2>&1 | FileCheck %s --check-prefix=ABI17

// Existing valid schema-2 packets continue to own byte-level cap checks. ABI18
// changes no 32 MiB slot, 256 MiB temporary/position, 16 KiB shared-memory,
// 256-record, 512 KiB entry, 8 MiB code, 16 MiB transport, or rank-eight cap.

// ABI18-COUNT-8: requires a bounded set of independently connected static GPU algebra regions
// ABI17-COUNT-7: requires a bounded set of independently connected static GPU algebra regions
// ABI17: must be statically shaped
// ABI17-COUNT-3: requires a bounded set of independently connected static GPU algebra regions
