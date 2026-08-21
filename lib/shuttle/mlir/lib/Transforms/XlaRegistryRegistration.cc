// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/XlaRegistration.h"

#include <mutex>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/stablehlo_module_transform.h"

namespace mlir::shuttle {

absl::Status registerShuttleXlaTransform() {
  static std::once_flag once;
  static absl::Status status;
  std::call_once(once, [] {
    status = xla::StablehloModuleTransformRegistry::Global().Register(
        "shuttle",
        [](mlir::ModuleOp module, absl::string_view serializedOptions) {
          return runShuttleXlaTransform(module, serializedOptions);
        });
  });
  return status;
}

} // namespace mlir::shuttle
