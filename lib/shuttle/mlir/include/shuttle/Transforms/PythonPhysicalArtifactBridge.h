// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TRANSFORMS_PYTHONPHYSICALARTIFACTBRIDGE_H_
#define SHUTTLE_TRANSFORMS_PYTHONPHYSICALARTIFACTBRIDGE_H_

namespace nanobind {
class module_;
}

namespace mlir::shuttle {

// Adds the production compile-time physical compiler registration surface to
// jaxlib's _jax module. Registration is process-wide and single-assignment.
void registerShuttlePhysicalArtifactBindings(nanobind::module_ &module);

} // namespace mlir::shuttle

#endif // SHUTTLE_TRANSFORMS_PYTHONPHYSICALARTIFACTBRIDGE_H_
