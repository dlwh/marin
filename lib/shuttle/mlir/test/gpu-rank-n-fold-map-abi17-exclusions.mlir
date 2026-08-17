// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-test-opt %S/Inputs/abi17-rank-n-fold-exclusions.mlir --split-input-file --shuttle-gpu-executable-bundle-pipeline 2>&1 | FileCheck %s --check-prefix=EXCLUDED

// ABI17 does not widen caps: public slots remain 32 MiB, temporaries 256 MiB,
// logical positions 256 Mi, shared memory 16 KiB, records 256, descriptor
// entries 512 KiB, device code 8 MiB, and transport payloads 16 MiB. Runtime
// decoder mutation tests own the byte-level enforcement of those unchanged
// schema-2/schema-3 limits.

// EXCLUDED-COUNT-7: requires a bounded set of independently connected static GPU algebra regions
// EXCLUDED: must be statically shaped
// EXCLUDED-COUNT-3: requires a bounded set of independently connected static GPU algebra regions
