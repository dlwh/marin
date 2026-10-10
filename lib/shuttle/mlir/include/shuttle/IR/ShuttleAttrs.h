// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_IR_SHUTTLEATTRS_H_
#define SHUTTLE_IR_SHUTTLEATTRS_H_

#include <cstdint>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"

#include "shuttle/IR/ShuttleEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "shuttle/IR/ShuttleAttrs.h.inc"

namespace mlir::shuttle {

// SM90 and SM100 permit up to 227 KiB per block. Initialization also checks
// static plus dynamic memory against the concrete device opt-in capacity.
inline constexpr uint64_t kMaximumGpuDynamicSharedMemoryBytes = 227 * 1024;

} // namespace mlir::shuttle

#endif // SHUTTLE_IR_SHUTTLEATTRS_H_
