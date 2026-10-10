// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TRANSFORMS_PHYSICALARTIFACT_H_
#define SHUTTLE_TRANSFORMS_PHYSICALARTIFACT_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleOps.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "shuttle/Transforms/Observer.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::shuttle {

// A source-bound request for a physical compiler. The provider receives only
// generic Shuttle algebra and its verified plans; it must not recover or name a
// workload. The fingerprint closes the exact region/plan/schedule/target tuple
// and is part of the provider cache identity.
struct GpuPhysicalArtifactRequest {
  RegionOp region;
  MaterializationPlanOp materialization;
  SchedulePlanOp schedule;
  EventLoweringPlanOp eventPlan;
  GpuTarget target;
  std::string fingerprint;
  llvm::SmallVector<std::optional<int64_t>> bindingIndices;
  llvm::SmallVector<std::optional<int64_t>> regionResultIndices;
  llvm::SmallVector<int64_t> reuseGroups;
};

// Content identity for the exact generic algebra, verified plans, and target
// presented to a physical compiler.  This intentionally includes RegionOp
// rather than assuming that a materialization fingerprint closes every scalar
// or relation operation in the source algebra.
std::string gpuPhysicalArtifactRequestFingerprint(
    RegionOp region, MaterializationPlanOp materialization,
    SchedulePlanOp schedule, EventLoweringPlanOp eventPlan, GpuTarget target);

// Canonical structured wire record for a language-runtime physical compiler.
// This serializes Shuttle algebra and verified plans, not StableHLO text. SSA
// edges are numeric value identities and operations remain dialect-generic.
std::string
gpuPhysicalArtifactRequestJson(const GpuPhysicalArtifactRequest &request);

// Compiler-owned device-IR operations for one source Event realization. This
// record is deliberately outside serialized transport: only an in-process
// provider can return it to the native build pass.
struct GpuSourceEventLegalization {
  int64_t eventOrdinal;
  std::vector<GpuEventOperation> operations;
};

enum class GpuInternalEventOperationKind : uint8_t {
  Notify = 0,
  Wait = 1,
};

// Compiler-owned device-IR operations for one refined internal Event
// allocation. Unlike source Event legalizations, these records are keyed by
// the distinct internal region/allocation identity and never consume source
// Event ordinals. Only an in-process provider can return them; serialized
// physical provenance remains descriptive and cannot establish trust.
struct GpuInternalEventOperation {
  int64_t ordinal = -1;
  GpuInternalEventOperationKind kind = GpuInternalEventOperationKind::Notify;
  int64_t relationOrdinal = -1;
  std::string taskFamily;
  GpuInternalEventCoordinate taskCoordinate;
  GpuInternalEventCoordinate mechanismCoordinate;
  GpuInternalEventAllocation::LegalizationKind legalizationKind =
      GpuInternalEventAllocation::LegalizationKind::CtaMbarrier;
  GpuEventPhysicalScope scope = GpuEventPhysicalScope::ThreadBlock;
  GpuInternalEventCoordinate allocationOwner;
  int64_t slot = -1;
  int64_t generation = -1;
  int64_t generationCount = 0;
  int64_t operationCount = 0;
  int64_t initialCount = 0;
  bool releaseAcquire = false;
  std::optional<int64_t> transactionBytes;
};

struct GpuInternalEventLegalization {
  int64_t regionOrdinal = -1;
  std::string allocationId;
  std::vector<GpuInternalEventOperation> operations;
};

struct GpuPhysicalArtifactCompilation {
  GpuFatbinExecutableBundle bundle;
  std::vector<GpuSourceEventLegalization> sourceEventLegalizations;
  std::vector<GpuInternalEventLegalization> internalEventLegalizations;
};

class GpuPhysicalArtifactProvider {
public:
  virtual ~GpuPhysicalArtifactProvider() = default;

  // No value means that this provider has no legal physical lowering for the
  // generic request. Provider and artifact failures are errors and cannot
  // silently select another semantic implementation.
  virtual absl::StatusOr<std::optional<GpuPhysicalArtifactCompilation>>
  compile(const GpuPhysicalArtifactRequest &request) const = 0;
};

// The process-wide provider is single-assignment so ordinary XLA compilation
// cannot change physical compilers after registration.
absl::Status registerGpuPhysicalArtifactProvider(
    std::shared_ptr<const GpuPhysicalArtifactProvider> provider);

std::shared_ptr<const GpuPhysicalArtifactProvider>
gpuPhysicalArtifactProvider();

} // namespace mlir::shuttle

#endif // SHUTTLE_TRANSFORMS_PHYSICALARTIFACT_H_
