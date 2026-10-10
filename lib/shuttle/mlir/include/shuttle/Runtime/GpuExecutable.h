// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_RUNTIME_GPUEXECUTABLE_H_
#define SHUTTLE_RUNTIME_GPUEXECUTABLE_H_

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::shuttle {

inline constexpr char kGpuExecutableBundleFfiTarget[] =
    "shuttle.gpu.executable_bundle.v2";
inline constexpr uint64_t kMaximumGpuTransportBytes = 16 * 1024 * 1024;
inline constexpr uint64_t kMaximumGpuExecutableRecords = 256;
inline constexpr uint64_t kMaximumGpuCodeBytes = 8 * 1024 * 1024;
inline constexpr uint64_t kMaximumGpuEntryCodeBytes = 512 * 1024;
// External buffers are caller-owned; only native temporary allocations use
// these byte budgets. Their tensor sizes remain checked signed 64-bit values.
inline constexpr uint64_t kMaximumGpuTemporarySlotBytes = 32 * 1024 * 1024;
inline constexpr uint64_t kMaximumGpuTemporaryBytes = 256 * 1024 * 1024;
inline constexpr uint64_t kMaximumGpuEventTransactionBytes = 32 * 1024 * 1024;
inline constexpr uint64_t kMaximumGpuAggregateTaskPositions = 256 * 1024 * 1024;
inline constexpr uint64_t kMaximumGpuKernelArgumentBytes = 4096;

enum class GpuKernelArgumentKind : uint8_t {
  SlotPointer = 0,
  ScalarImmediate = 1,
  PackedBytes = 2,
  TensorMap = 3,
  StructuredBytes = 4,
};

enum class GpuScalarImmediateType : uint8_t {
  I32 = 0,
  U32 = 1,
  I64 = 2,
  U64 = 3,
  F32 = 4,
  F64 = 5,
};

enum class GpuTensorMapInterleave : uint8_t {
  None = 0,
  Bytes16 = 1,
  Bytes32 = 2,
};

enum class GpuTensorMapSwizzle : uint8_t {
  None = 0,
  Bytes32 = 1,
  Bytes64 = 2,
  Bytes128 = 3,
};

enum class GpuTensorMapL2Promotion : uint8_t {
  None = 0,
  Bytes64 = 1,
  Bytes128 = 2,
  Bytes256 = 3,
};

enum class GpuTensorMapOobFill : uint8_t {
  None = 0,
  NanRequestZeroFma = 1,
};

struct GpuTensorMapDescriptor {
  int64_t slotOrdinal = -1;
  uint64_t byteOffset = 0;
  uint8_t elementByteWidth = 0;
  std::vector<uint64_t> globalDimensions;
  std::vector<uint64_t> globalByteStrides;
  std::vector<uint32_t> boxDimensions;
  std::vector<uint32_t> elementStrides;
  GpuTensorMapInterleave interleave = GpuTensorMapInterleave::None;
  GpuTensorMapSwizzle swizzle = GpuTensorMapSwizzle::None;
  GpuTensorMapL2Promotion l2Promotion = GpuTensorMapL2Promotion::None;
  GpuTensorMapOobFill oobFill = GpuTensorMapOobFill::None;
  uint64_t alignment = 64;
  ExecutableAccess access = ExecutableAccess::Read;
};

enum class GpuStructuredFieldKind : uint8_t {
  SlotAddress = 0,
  ImmediateBytes = 1,
};

struct GpuStructuredField {
  GpuStructuredFieldKind kind;
  uint64_t offset = 0;
  int64_t slotOrdinal = -1;
  uint64_t slotByteOffset = 0;
  std::vector<uint8_t> bytes;
  ExecutableAccess access = ExecutableAccess::Read;
};

struct GpuStructuredArgument {
  uint64_t size = 0;
  uint64_t alignment = 1;
  std::vector<GpuStructuredField> fields;
};

struct GpuKernelArgument {
  GpuKernelArgumentKind kind;
  int64_t slotOrdinal = -1;
  GpuScalarImmediateType scalarType = GpuScalarImmediateType::I32;
  std::vector<uint8_t> bytes;
  uint64_t alignment = 1;
  GpuTensorMapDescriptor tensorMap;
  GpuStructuredArgument structure;
  uint64_t slotByteOffset = 0;
  ExecutableAccess access = ExecutableAccess::Read;
};

enum class GpuFunctionAttributeKind : uint8_t {
  MaximumDynamicSharedMemoryBytes = 0,
  PreferredSharedMemoryCarveout = 1,
  NonPortableClusterSizeAllowed = 2,
};

struct GpuFunctionAttribute {
  GpuFunctionAttributeKind kind;
  uint64_t value;
};

struct GpuLaunch {
  int64_t taskOrdinal;
  std::string kernelName;
  int64_t codeOffset;
  int64_t codeLength;
  std::string codeDigest;
  std::array<uint64_t, 3> grid;
  std::array<uint64_t, 3> block;
  uint64_t dynamicSharedMemoryBytes;
  std::vector<GpuKernelArgument> arguments;
  std::vector<GpuFunctionAttribute> functionAttributes;
  std::vector<int64_t> inputSlots;
  std::vector<int64_t> outputSlots;
  std::vector<int64_t> dependencies;
  // Structural source tasks implemented by this physical launch. Multiple
  // source tasks may fuse into one launch, and one source task may be split
  // across multiple launches.
  std::vector<int64_t> sourceTaskOrdinals;
  uint64_t imageIndex = 0;
};

enum class GpuEventRealizationKind : uint8_t {
  ProgramOrder = 0,
  StreamOrder = 1,
  AlgebraicRefinement = 2,
  MBarrier = 3,
  Pipeline = 4,
  Queue = 5,
};

enum class GpuAffineExpressionKind : uint8_t {
  Dimension = 0,
  Constant = 1,
  Add = 2,
  Multiply = 3,
  Modulo = 4,
  FloorDivide = 5,
  CeilDivide = 6,
};

// A symbol-free affine expression over one EventTensor relation domain. The
// payload is a dimension position or constant for leaves; binary expressions
// carry exactly two operands.
struct GpuAffineExpression {
  GpuAffineExpressionKind kind = GpuAffineExpressionKind::Constant;
  int64_t payload = 0;
  std::vector<GpuAffineExpression> operands;
};

struct GpuAffineMap {
  int64_t dimensions = 0;
  std::vector<GpuAffineExpression> results;
};

struct GpuIntegerSet {
  int64_t dimensions = 0;
  std::vector<GpuAffineExpression> constraints;
  // One byte per constraint: zero means expression >= 0; one means
  // expression == 0.
  std::vector<uint8_t> equalities;
};

struct GpuEventCoverEndpoint {
  int64_t launchOrdinal = -1;
  GpuAffineMap owner;
  GpuAffineMap localTime;
};

// One relation subset and the physical producer/consumer points covering it.
// An absent predicate denotes the full source EventTensor relation. The source
// schedule fingerprint and event ordinal supply the relation domain and its
// logical producer/consumer index maps.
struct GpuEventCoverPiece {
  std::optional<GpuIntegerSet> relationSubset;
  GpuEventCoverEndpoint producer;
  GpuEventCoverEndpoint consumer;
};

enum class GpuEventPhysicalScope : uint8_t {
  ThreadBlock = 0,
  Cluster = 1,
  Device = 2,
  System = 3,
};

// A bounded physical synchronization allocation used by a non-erased Event
// realization. Both maps are over the source EventTensor relation domain. The
// slot map selects storage within the allocation; generationPhase selects the
// bounded generation and optional mechanism-local phase.
struct GpuEventMechanismAllocation {
  int64_t allocationOrdinal = -1;
  GpuEventPhysicalScope scope = GpuEventPhysicalScope::ThreadBlock;
  int64_t capacity = 0;
  GpuAffineMap slot;
  GpuAffineMap generationPhase;
  int64_t generationCount = 0;
  int64_t arrivalCount = 0;
  std::optional<int64_t> transactionBytes;
};

enum class GpuEventOperationKind : uint8_t {
  Notify = 0,
  Wait = 1,
};

// One synchronization operation in compiler IR before device-code emission.
// The cover-piece reference binds the operation to an exact subset of the
// source EventTensor relation. The remaining fields describe the operation at
// the physical execution point and bounded mechanism allocation.
struct GpuEventOperation {
  int64_t ordinal = -1;
  GpuEventOperationKind kind = GpuEventOperationKind::Notify;
  int64_t coverPieceOrdinal = -1;
  GpuEventCoverEndpoint executionPoint;
  GpuAffineMap slot;
  GpuAffineMap generationPhase;
  GpuEventPhysicalScope scope = GpuEventPhysicalScope::ThreadBlock;
  int64_t count = 0;
  bool releaseAcquire = false;
};

// A structural certificate emitted by a compiler-IR lowering stage. The
// compilerIrDigest is the canonical digest of operations, while
// deviceImageDigest binds that IR declaration to one device image. Transport
// presence alone does not establish that the certificate came from a trusted
// in-process lowering stage.
struct GpuEventOperationCertificate {
  std::string compilerIrDigest;
  std::string deviceImageDigest;
  std::vector<GpuEventOperation> operations;
};

// A physical compiler's source-bound certificate that one exact EventTensor
// relation is erased by physical execution order/algebraic refinement or
// assigned a concrete synchronization mechanism. coverPieces carry the
// instance-level proof. Erased realizations retain a launch-order summary whose
// groups are
// [producer launch, producer owner-local phase, consumer launch, consumer
// owner-local phase]. Algebraic-refinement endpoints name the fused launch and
// may use the same phase. The source schedule fingerprint and event ordinal
// bind the certificate to the full logical relation; sourceTaskOrdinals are
// only provenance and are not used as an erasure proof.
struct GpuEventRealization {
  int64_t eventOrdinal;
  GpuEventRealizationKind kind;
  std::vector<int64_t> coveringLaunchOrder;
  std::vector<GpuEventCoverPiece> coverPieces;
  std::optional<GpuEventMechanismAllocation> mechanism;
  std::optional<GpuEventOperationCertificate> operationCertificate;
};

using GpuInternalEventCoordinate = std::vector<int64_t>;

struct GpuInternalEventAxis {
  std::string name;
  int64_t extent = 0;
};

enum class GpuInternalEventDomainKind : uint8_t {
  TaskFamily = 0,
  EventDomain = 1,
};

struct GpuInternalEventDomain {
  GpuInternalEventDomainKind kind = GpuInternalEventDomainKind::TaskFamily;
  std::string name;
  std::vector<GpuInternalEventAxis> axes;
  std::optional<std::string> placement;
};

struct GpuInternalEventRelationPair {
  GpuInternalEventCoordinate source;
  GpuInternalEventCoordinate target;
};

struct GpuInternalEventRelation {
  GpuInternalEventDomain source;
  GpuInternalEventDomain target;
  std::vector<GpuInternalEventRelationPair> pairs;
};

struct GpuInternalEventVisibility {
  GpuEventPhysicalScope scope = GpuEventPhysicalScope::ThreadBlock;
  bool releaseOnNotify = false;
  bool acquireBeforeConsumer = false;
};

enum class GpuInternalEventGenerationPolicy : uint8_t {
  PerInvocation = 0,
  Phased = 1,
};

struct GpuInternalEventInitialCount {
  GpuInternalEventCoordinate coordinate;
  int64_t value = 0;
};

struct GpuInternalEventTensorPlan {
  std::string name;
  GpuInternalEventRelation requiredDependence;
  GpuInternalEventVisibility requiredVisibility;
  GpuInternalEventDomain eventDomain;
  GpuInternalEventRelation notifyRelation;
  GpuInternalEventRelation triggerRelation;
  std::vector<GpuInternalEventInitialCount> initialCounts;
  std::string initialCountProvenance;
  GpuEventPhysicalScope memoryScope = GpuEventPhysicalScope::ThreadBlock;
  GpuInternalEventGenerationPolicy generationPolicy =
      GpuInternalEventGenerationPolicy::PerInvocation;
  GpuInternalEventVisibility visibility;
  std::optional<std::string> schedulingMode;
};

struct GpuInternalEventAllocation {
  struct NotifyRelation {
    std::string taskFamily;
    GpuInternalEventCoordinate taskCoordinate;
    GpuInternalEventCoordinate mechanismCoordinate;
  };
  struct WaitRelation {
    GpuInternalEventCoordinate mechanismCoordinate;
    std::string taskFamily;
    GpuInternalEventCoordinate taskCoordinate;
  };

  std::string allocationId;
  std::vector<std::string> coveredPlans;
  GpuEventPhysicalScope scope = GpuEventPhysicalScope::ThreadBlock;
  bool releaseOnNotify = false;
  bool acquireBeforeConsumer = false;
  int64_t capacity = 0;
  std::vector<GpuInternalEventCoordinate> mechanismCoordinates;
  std::vector<GpuInternalEventCoordinate> allocationOwners;
  std::vector<int64_t> slots;
  std::vector<int64_t> generations;
  int64_t generationCount = 0;
  std::vector<NotifyRelation> notifyRelations;
  std::vector<WaitRelation> waitRelations;
  std::vector<int64_t> initialCounts;
  std::optional<std::vector<int64_t>> transactionBytesByCoordinate;
  // Machine mechanism required by the physical allocation. This is
  // deliberately independent of the realization's descriptive mechanism
  // string and tensor workload. Capacity and generations separately describe
  // bounded pipeline composition over the selected primitive.
  enum class LegalizationKind : uint8_t {
    CtaMbarrier = 0,
    CtaTransactionMbarrier = 1,
    CtaAsyncCompletionQueue = 2,
    DeviceSemaphoreQueue = 3,
  } legalizationKind = LegalizationKind::CtaMbarrier;
};

enum class GpuInternalEventRealizationKind : uint8_t {
  ErasedProgramOrder = 0,
  ErasedStreamOrder = 1,
  ErasedAlgebraicRefinement = 2,
  Physical = 3,
};

struct GpuInternalEventRealization {
  std::string plan;
  GpuInternalEventTensorPlan event;
  GpuInternalEventRealizationKind kind =
      GpuInternalEventRealizationKind::ErasedProgramOrder;
  std::string mechanism;
  std::vector<std::string> allocationIds;
};

// One canonical refined physical region. Combined envelopes are flattened in
// their declared region order, while each region retains its own exact digest.
struct GpuInternalEventRegion {
  int64_t schemaVersion = 0;
  std::string fingerprint;
  std::vector<GpuInternalEventDomain> taskFamilies;
  std::vector<GpuInternalEventTensorPlan> events;
  std::vector<GpuInternalEventAllocation> allocations;
  std::vector<GpuInternalEventRealization> realizations;
};

struct GpuInternalEventSchedule {
  int64_t schemaVersion = 0;
  std::string physicalScheduleFingerprint;
  std::string fingerprint;
  std::vector<GpuInternalEventRegion> regions;
};

struct GpuDeviceIrOperationCount {
  std::string operationName;
  int64_t count = 0;
};

// Descriptive evidence from one structured compiler IR module before device
// code emission. This inventory does not map operations to Event allocations
// and cannot establish an Event legalization certificate.
struct GpuDeviceIrSnapshot {
  std::string mlirBytecodeDigest;
  std::vector<GpuDeviceIrOperationCount> operationCounts;
};

struct GpuSlot {
  int64_t ordinal;
  Type tensorType;
  int64_t requiredBytes;
  llvm::SmallVector<int64_t> byteStrides;
  int64_t byteOffset;
  int64_t alignment;
  ExecutableAddressSpace addressSpace;
  ExecutableAccess access;
  MaterializationStorage storage;
  int64_t aliasGroup;
  int64_t reuseGroup;
  ExecutableBindingKind bindingKind;
  int64_t bindingIndex;
};

// A workload-independent CUDA image table and launch recipe. Every launch names
// a kernel in its selected image and supplies its device ABI in exact parameter
// order.  Slot descriptors carry storage, access, reuse, and alignment; this
// keeps scratch and read-modify-write behavior out of kernel-name dispatch.
struct GpuFatbinExecutableBundle {
  int64_t deviceSchemaVersion;
  ExecutableCodeFormat codeFormat;
  std::string scheduleFingerprint;
  std::string planFingerprint;
  std::vector<std::vector<uint8_t>> images;
  std::vector<GpuLaunch> launches;
  std::vector<GpuSlot> slots;
  std::vector<GpuEventRealization> eventRealizations;
  // Canonical refined internal Event schedule. This is deliberately separate
  // from source-bound eventRealizations and does not add source Event ordinals.
  std::string physicalEventProvenance;
  std::string physicalEventProvenanceDigest;
  std::vector<GpuDeviceIrSnapshot> deviceIrSnapshots;
};

struct GpuExternalBinding {
  ExecutableBindingKind kind;
  int64_t index;
  int64_t slotOrdinal;
  Type tensorType;
  int64_t requiredBytes;
  int64_t alignment;
};

class GpuExecutable {
public:
  static absl::StatusOr<std::shared_ptr<const GpuExecutable>>
  Load(llvm::ArrayRef<uint8_t> bytes);

  // PTX bytes or the sole CUDA image; empty for a multi-image executable.
  llvm::ArrayRef<uint8_t> codeBytes() const;
  llvm::ArrayRef<std::vector<uint8_t>> codeImages() const;
  llvm::ArrayRef<uint8_t> codeImage(uint64_t index) const;
  uint64_t totalCodeBytes() const;
  int64_t deviceSchemaVersion() const;
  ExecutableCodeFormat codeFormat() const;
  llvm::StringRef scheduleFingerprint() const;
  llvm::StringRef planFingerprint() const;
  llvm::ArrayRef<GpuLaunch> launches() const;
  llvm::ArrayRef<GpuSlot> slots() const;
  llvm::ArrayRef<GpuEventRealization> eventRealizations() const;
  llvm::StringRef physicalEventProvenance() const;
  llvm::StringRef physicalEventProvenanceDigest() const;
  const std::optional<GpuInternalEventSchedule> &internalEventSchedule() const;
  llvm::ArrayRef<GpuDeviceIrSnapshot> deviceIrSnapshots() const;
  llvm::ArrayRef<GpuExternalBinding> externalBindings() const;

private:
  class Impl;
  explicit GpuExecutable(std::shared_ptr<const Impl> implementation);
  std::shared_ptr<const Impl> implementation;
};

FailureOr<llvm::SmallVector<uint8_t>>
serializeGpuExecutableBundle(ModuleOp module);
absl::StatusOr<llvm::SmallVector<uint8_t>>
serializeGpuFatbinExecutableBundle(const GpuFatbinExecutableBundle &bundle);
absl::StatusOr<GpuFatbinExecutableBundle>
deserializeGpuFatbinExecutableBundle(llvm::ArrayRef<uint8_t> bytes,
                                     MLIRContext *context);
absl::StatusOr<GpuInternalEventSchedule>
parseGpuInternalEventSchedule(llvm::StringRef canonicalProvenance);
std::string gpuExecutableBundleDigest(llvm::ArrayRef<uint8_t> bytes);
// Preserve raw single-image identity. Multiple images bind their ordered,
// length-delimited bytes and every launch's image index.
absl::StatusOr<llvm::SmallVector<uint8_t>>
gpuFatbinCodeIdentity(llvm::ArrayRef<std::vector<uint8_t>> images,
                      llvm::ArrayRef<GpuLaunch> launches);
std::string gpuEventOperationCertificateDigest(
    const GpuEventOperationCertificate &certificate);

} // namespace mlir::shuttle

#endif // SHUTTLE_RUNTIME_GPUEXECUTABLE_H_
