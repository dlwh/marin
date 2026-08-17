// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-test-opt %S/Inputs/abi21-full-prefix-fold-exclusions.mlir --split-input-file --shuttle-gpu-executable-bundle-pipeline > %t 2>&1
// RUN: FileCheck %s --input-file=%t --check-prefix=BOUNDED
// RUN: FileCheck %s --input-file=%t --check-prefix=STATIC
// RUN: FileCheck %s --input-file=%t --check-prefix=CAP
// RUN: not shuttle-test-opt %S/Inputs/abi21-full-prefix-fold-2x7x19.mlir --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-add-fold-fastmath --shuttle-plan-row-fold-materialization 2>&1 | FileCheck %s --check-prefix=FASTMATH

// Every semantic negative keeps a positive-static BF16 public boundary. The
// maximum, non-add, and initializer cases differ from the admitted cell only
// in the named Fold property. Existing ABI15/ABI17 tests retain checked cap,
// rank, dynamic-shape, and mixed Contract/Fold coverage; the final packet here
// crosses the same unchanged 32 MiB per-slot limit.

// BOUNDED-COUNT-11: requires a bounded set of independently connected static GPU algebra regions
// STATIC: {{requires a bounded set of independently connected static GPU algebra regions|must be statically shaped}}
// CAP: serialized GPU transport failed public validation: invalid Shuttle GPU executable transport: slot descriptor
// FASTMATH: requires a bounded set of independently connected static GPU algebra regions
