// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Runtime/GpuFfi.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/profiler/lib/traceme.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::shuttle {
namespace {

namespace ffi = xla::ffi;
namespace se = stream_executor;

struct LoadedExecutable {
  std::vector<std::string> ptx;
  std::vector<std::unique_ptr<se::Kernel>> kernels;
};

struct InitializationCache {
  std::mutex mutex;
  llvm::DenseMap<se::StreamExecutor *, std::shared_ptr<const LoadedExecutable>>
      executables;
};

struct InstantiateState {
  std::vector<uint8_t> transport;
  std::string digest;
  std::shared_ptr<const GpuExecutable> executable;
  std::vector<GpuExternalBinding> operands;
  std::vector<GpuExternalBinding> results;
  // Instantiate state has compilation lifetime. Keep immutable driver-loaded
  // kernels here so XLA's execution-scoped Initialize stage does not reload
  // the same PTX for every invocation. The StreamExecutor key binds a cache
  // entry to its CUDA device/context; invocation-local temporary storage stays
  // in PrepareState.
  std::shared_ptr<InitializationCache> initializationCache;
};

struct PrepareState {
  std::vector<se::ScopedDeviceAddress<uint8_t>> temporaries;
  llvm::DenseMap<int64_t, se::DeviceAddressBase> slots;
};

struct InitializeState {
  std::shared_ptr<const LoadedExecutable> executable;
};

absl::Status invalid(StringRef message) {
  return absl::InvalidArgumentError(message.str());
}

struct ExternalElement {
  xla::PrimitiveType primitive;
  int64_t width;
};

std::optional<ExternalElement> externalElement(RankedTensorType type) {
  if (!type || !type.hasStaticShape() || type.getRank() > 8)
    return std::nullopt;
  if (type.getElementType().isBF16() && type.getRank() >= 1)
    return ExternalElement{xla::PrimitiveType::BF16, 2};
  if (type.getElementType().isF32())
    return ExternalElement{xla::PrimitiveType::F32, 4};
  return std::nullopt;
}

absl::Status projectBindings(ArrayRef<GpuExternalBinding> bindings,
                             std::vector<GpuExternalBinding> &operands,
                             std::vector<GpuExternalBinding> &results) {
  llvm::SmallDenseSet<int64_t> slots;
  for (const GpuExternalBinding &binding : bindings) {
    std::vector<GpuExternalBinding> *projection = nullptr;
    if (binding.kind == ExecutableBindingKind::Operand)
      projection = &operands;
    else if (binding.kind == ExecutableBindingKind::Result)
      projection = &results;
    else
      return invalid("GPU executable has an invalid external projection");
    if (binding.index < 0)
      return invalid("GPU executable external projection index is invalid");
    auto type = dyn_cast<RankedTensorType>(binding.tensorType);
    std::optional<ExternalElement> element = externalElement(type);
    int64_t elements = 1;
    if (!element || binding.slotOrdinal < 0 ||
        !slots.insert(binding.slotOrdinal).second ||
        binding.alignment != element->width)
      return invalid("GPU executable external projection type is invalid");
    for (int64_t extent : type.getShape()) {
      if (extent <= 0 ||
          elements > std::numeric_limits<int64_t>::max() / extent)
        return invalid("GPU executable external projection shape is invalid");
      elements *= extent;
    }
    if (elements > std::numeric_limits<int64_t>::max() / element->width ||
        binding.requiredBytes != elements * element->width)
      return invalid("GPU executable external projection bytes are invalid");
    projection->push_back(binding);
  }
  auto orderBindings = [](std::vector<GpuExternalBinding> &projection) {
    llvm::sort(projection, [](const GpuExternalBinding &left,
                              const GpuExternalBinding &right) {
      return left.index < right.index;
    });
    for (auto [index, binding] : llvm::enumerate(projection))
      if (binding.index != static_cast<int64_t>(index))
        return false;
    return true;
  };
  if (!orderBindings(operands) || !orderBindings(results))
    return invalid("GPU executable external projection is not contiguous");
  if (operands.empty() || results.empty() ||
      bindings.size() > kMaximumGpuExecutableRecords)
    return invalid("GPU executable projection count is invalid");
  return absl::OkStatus();
}

absl::Status bufferMatches(const GpuExternalBinding &binding,
                           const ffi::AnyBuffer &buffer, bool requireData) {
  auto type = dyn_cast<RankedTensorType>(binding.tensorType);
  std::optional<ExternalElement> element = externalElement(type);
  if (!element || buffer.element_type() != element->primitive ||
      buffer.dimensions().size() != static_cast<size_t>(type.getRank()) ||
      !std::equal(buffer.dimensions().begin(), buffer.dimensions().end(),
                  type.getShape().begin(), type.getShape().end()) ||
      buffer.size_bytes() != static_cast<size_t>(binding.requiredBytes))
    return invalid("typed FFI view does not match the GPU executable ABI");
  if (requireData && (buffer.untyped_data() == nullptr ||
                      reinterpret_cast<uintptr_t>(buffer.untyped_data()) %
                              static_cast<uint64_t>(binding.alignment) !=
                          0))
    return invalid("typed FFI device pointer does not satisfy alignment");
  return absl::OkStatus();
}

absl::Status
externalAddresses(ffi::RemainingArgs arguments, ffi::RemainingRets results,
                  const InstantiateState &state,
                  llvm::DenseMap<int64_t, se::DeviceAddressBase> &slots) {
  if (arguments.size() != state.operands.size() ||
      results.size() != state.results.size())
    return invalid("typed FFI GPU argument count is invalid");
  struct Span {
    uintptr_t begin;
    uintptr_t end;
  };
  SmallVector<Span> spans;
  auto add = [&](const GpuExternalBinding &binding,
                 const ffi::AnyBuffer &buffer) -> absl::Status {
    if (absl::Status status = bufferMatches(binding, buffer, true);
        !status.ok())
      return status;
    uintptr_t begin = reinterpret_cast<uintptr_t>(buffer.untyped_data());
    if (begin > std::numeric_limits<uintptr_t>::max() -
                    static_cast<uint64_t>(binding.requiredBytes))
      return invalid("typed FFI GPU buffer address overflows");
    Span current{begin, begin + static_cast<uint64_t>(binding.requiredBytes)};
    for (Span prior : spans)
      if (current.begin < prior.end && prior.begin < current.end)
        return invalid("typed FFI GPU buffers must not alias");
    spans.push_back(current);
    slots[binding.slotOrdinal] =
        se::DeviceAddressBase(buffer.untyped_data(), binding.requiredBytes);
    return absl::OkStatus();
  };
  for (size_t index = 0; index < state.operands.size(); ++index) {
    absl::StatusOr<ffi::AnyBuffer> buffer =
        arguments.get<ffi::AnyBuffer>(index);
    if (!buffer.ok())
      return buffer.status();
    if (absl::Status status = add(state.operands[index], *buffer); !status.ok())
      return status;
  }
  for (size_t index = 0; index < state.results.size(); ++index) {
    absl::StatusOr<ffi::Result<ffi::AnyBuffer>> buffer =
        results.get<ffi::AnyBuffer>(index);
    if (!buffer.ok())
      return buffer.status();
    if (absl::Status status = add(state.results[index], **buffer); !status.ok())
      return status;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<InstantiateState>>
instantiate(int64_t transportSchemaVersion, absl::string_view bundleBytes,
            int64_t bundleSize, absl::string_view bundleSha256,
            int64_t deviceSchemaVersion, int64_t invocationAbiSchemaVersion,
            int64_t bundleSchemaVersion, absl::string_view completion) {
  tsl::profiler::TraceMe trace("ShuttleGpuFfi::Instantiate");
  try {
    if (transportSchemaVersion != 2 ||
        (deviceSchemaVersion != 3 && deviceSchemaVersion != 4) ||
        invocationAbiSchemaVersion != 3 || bundleSchemaVersion != 2 ||
        completion != "stream_ordered" || bundleSize < 0 ||
        static_cast<size_t>(bundleSize) != bundleBytes.size() ||
        bundleBytes.size() > kMaximumGpuTransportBytes)
      return invalid("invalid Shuttle GPU executable transport metadata");
    ArrayRef<uint8_t> borrowed(
        reinterpret_cast<const uint8_t *>(bundleBytes.data()),
        bundleBytes.size());
    std::string digest = gpuExecutableBundleDigest(borrowed);
    if (digest != bundleSha256)
      return invalid("Shuttle GPU executable transport digest mismatch");
    absl::StatusOr<std::shared_ptr<const GpuExecutable>> executable =
        GpuExecutable::Load(borrowed);
    if (!executable.ok())
      return executable.status();
    if ((*executable)->deviceSchemaVersion() != deviceSchemaVersion)
      return invalid("Shuttle GPU device schema metadata mismatch");
    std::vector<GpuExternalBinding> operands;
    std::vector<GpuExternalBinding> results;
    if (absl::Status status = projectBindings((*executable)->externalBindings(),
                                              operands, results);
        !status.ok())
      return status;
    std::vector<uint8_t> owned(borrowed.begin(), borrowed.end());
    return std::make_unique<InstantiateState>(InstantiateState{
        std::move(owned), std::move(digest), std::move(*executable),
        std::move(operands), std::move(results),
        std::make_shared<InitializationCache>()});
  } catch (const std::bad_alloc &) {
    return absl::ResourceExhaustedError(
        "insufficient memory to instantiate Shuttle GPU executable");
  } catch (...) {
    return absl::InternalError(
        "unexpected failure while instantiating Shuttle GPU executable");
  }
}

absl::StatusOr<std::unique_ptr<PrepareState>>
prepare(InstantiateState *state, se::DeviceAddressAllocator *allocator,
        int32_t deviceOrdinal) {
  tsl::profiler::TraceMe trace("ShuttleGpuFfi::Prepare");
  try {
    if (state == nullptr || allocator == nullptr || deviceOrdinal < 0)
      return invalid("GPU executable Prepare context is invalid");
    auto prepared = std::make_unique<PrepareState>();
    struct TemporaryGroup {
      int64_t reuseGroup;
      int64_t requiredBytes;
      int64_t alignment;
    };
    std::vector<TemporaryGroup> groups;
    llvm::DenseMap<int64_t, size_t> groupIndexes;
    uint64_t aggregateBytes = 0;
    for (const GpuSlot &slot : state->executable->slots()) {
      if (slot.storage != MaterializationStorage::Temporary)
        continue;
      auto inserted = groupIndexes.try_emplace(slot.reuseGroup, groups.size());
      if (inserted.second) {
        if (slot.requiredBytes <= 0 ||
            static_cast<uint64_t>(slot.requiredBytes) >
                kMaximumGpuTemporaryBytes - aggregateBytes)
          return invalid("GPU temporary allocation budget is invalid");
        aggregateBytes += slot.requiredBytes;
        groups.push_back({slot.reuseGroup, slot.requiredBytes, slot.alignment});
      } else {
        TemporaryGroup &group = groups[inserted.first->second];
        if (slot.requiredBytes > group.requiredBytes) {
          uint64_t increase = slot.requiredBytes - group.requiredBytes;
          if (increase > kMaximumGpuTemporaryBytes - aggregateBytes)
            return invalid("GPU temporary allocation budget is invalid");
          aggregateBytes += increase;
          group.requiredBytes = slot.requiredBytes;
        }
        group.alignment = std::max(group.alignment, slot.alignment);
      }
    }
    llvm::DenseMap<int64_t, se::DeviceAddressBase> groupAddresses;
    for (const TemporaryGroup &group : groups) {
      absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> allocation =
          allocator->Allocate(deviceOrdinal, group.requiredBytes);
      if (!allocation.ok())
        return allocation.status();
      if (allocation->is_null() ||
          reinterpret_cast<uintptr_t>(allocation->cref().opaque()) %
                  static_cast<uint64_t>(group.alignment) !=
              0)
        return invalid("GPU temporary allocation does not satisfy alignment");
      groupAddresses[group.reuseGroup] = allocation->cref();
      prepared->temporaries.push_back(std::move(*allocation));
    }
    for (const GpuSlot &slot : state->executable->slots()) {
      if (slot.storage != MaterializationStorage::Temporary)
        continue;
      auto address = groupAddresses.find(slot.reuseGroup);
      if (address == groupAddresses.end())
        return invalid("GPU temporary reuse group is not allocated");
      prepared->slots[slot.ordinal] =
          se::DeviceAddressBase(address->second.opaque(), slot.requiredBytes);
    }
    return prepared;
  } catch (const std::bad_alloc &) {
    return absl::ResourceExhaustedError(
        "insufficient memory to prepare Shuttle GPU executable");
  } catch (...) {
    return absl::InternalError(
        "unexpected failure while preparing Shuttle GPU executable");
  }
}

absl::StatusOr<std::unique_ptr<InitializeState>>
initialize(InstantiateState *state, se::Stream *stream,
           const se::GpuComputeCapability *capability) {
  tsl::profiler::TraceMe trace("ShuttleGpuFfi::Initialize");
  try {
    if (state == nullptr || stream == nullptr || stream->parent() == nullptr ||
        capability == nullptr || !capability->IsCuda())
      return invalid("GPU executable Initialize context is invalid");
    const se::CudaComputeCapability *cuda =
        capability->cuda_compute_capability();
    const ExecutableCodeFormat format = state->executable->codeFormat();
    const bool capabilityMatches =
        cuda != nullptr && ((format == ExecutableCodeFormat::CudaPtxSm90V1 &&
                             cuda->major == 9 && cuda->minor == 0) ||
                            (format == ExecutableCodeFormat::CudaPtxSm100V1 &&
                             cuda->major == 10 && cuda->minor == 0));
    if (!capabilityMatches)
      return invalid(
          "GPU executable target does not match CUDA compute capability");
    if (state->initializationCache == nullptr)
      return invalid("GPU executable Initialize cache is invalid");
    const bool requiresSharedMemoryOptIn = llvm::any_of(
        state->executable->launches(), [](const GpuLaunch &launch) {
          return launch.dynamicSharedMemoryBytes > 48 * 1024;
        });
    const int64_t dynamicSharedMemoryLimit =
        requiresSharedMemoryOptIn
            ? stream->parent()
                  ->GetDeviceDescription()
                  .shared_memory_per_block_optin()
            : 48 * 1024;
    if (requiresSharedMemoryOptIn && dynamicSharedMemoryLimit <= 48 * 1024)
      return invalid(
          "GPU executable requires unavailable dynamic shared-memory opt-in");
    std::lock_guard lock(state->initializationCache->mutex);
    auto cached = state->initializationCache->executables.find(stream->parent());
    if (cached != state->initializationCache->executables.end()) {
      tsl::profiler::TraceMe cacheHit("ShuttleGpuFfi::InitializeCacheHit");
      return std::make_unique<InitializeState>(
          InitializeState{cached->second});
    }

    tsl::profiler::TraceMe cacheMiss("ShuttleGpuFfi::InitializeCacheMiss");
    auto loaded = std::make_shared<LoadedExecutable>();
    loaded->ptx.reserve(state->executable->launches().size());
    loaded->kernels.reserve(state->executable->launches().size());
    for (const GpuLaunch &launch : state->executable->launches()) {
      if (launch.dynamicSharedMemoryBytes > 48 * 1024 &&
          launch.dynamicSharedMemoryBytes >
          static_cast<uint64_t>(dynamicSharedMemoryLimit))
        return invalid(
            "GPU executable dynamic shared memory exceeds device opt-in capacity");
      ArrayRef<uint8_t> slice = state->executable->codeBytes().slice(
          launch.codeOffset, launch.codeLength);
      loaded->ptx.emplace_back(
          reinterpret_cast<const char *>(slice.data()), slice.size());
      se::KernelLoaderSpec spec =
          se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
              loaded->ptx.back(), "shuttle_entry",
              launch.inputSlots.size() + launch.outputSlots.size());
      absl::StatusOr<std::unique_ptr<se::Kernel>> kernel =
          stream->parent()->LoadKernel(spec);
      if (!kernel.ok())
        return kernel.status();
      se::KernelMetadata metadata;
      metadata.set_shared_memory_bytes(launch.dynamicSharedMemoryBytes);
      (*kernel)->set_metadata(metadata);
      loaded->kernels.push_back(std::move(*kernel));
    }
    if (loaded->kernels.size() != state->executable->launches().size())
      return invalid("GPU executable kernel projection is invalid");
    std::shared_ptr<const LoadedExecutable> immutable = std::move(loaded);
    state->initializationCache->executables[stream->parent()] = immutable;
    return std::make_unique<InitializeState>(InitializeState{immutable});
  } catch (const std::bad_alloc &) {
    return absl::ResourceExhaustedError(
        "insufficient memory to initialize Shuttle GPU executable");
  } catch (...) {
    return absl::InternalError(
        "unexpected failure while initializing Shuttle GPU executable");
  }
}

absl::Status execute(ffi::RemainingArgs arguments, ffi::RemainingRets results,
                     InstantiateState *state, PrepareState *prepared,
                     InitializeState *initialized, se::Stream *stream) {
  tsl::profiler::TraceMe trace("ShuttleGpuFfi::Execute");
  try {
    if (state == nullptr || prepared == nullptr || initialized == nullptr ||
        initialized->executable == nullptr || stream == nullptr ||
        initialized->executable->kernels.size() !=
            state->executable->launches().size())
      return invalid("GPU executable Execute state is invalid");
    llvm::DenseMap<int64_t, se::DeviceAddressBase> slots = prepared->slots;
    {
      tsl::profiler::TraceMe bind("ShuttleGpuFfi::BindArguments");
      if (absl::Status status =
              externalAddresses(arguments, results, *state, slots);
          !status.ok())
        return status;
    }
    tsl::profiler::TraceMe launches("ShuttleGpuFfi::EnqueueKernels");
    for (const GpuLaunch &launch : state->executable->launches()) {
      SmallVector<se::KernelArg> kernelArguments;
      for (int64_t slot : launch.inputSlots) {
        if (!slots.count(slot))
          return invalid("GPU executable launch input is not bound");
        kernelArguments.push_back(slots.lookup(slot));
      }
      for (int64_t slot : launch.outputSlots) {
        if (!slots.count(slot))
          return invalid("GPU executable launch output is not bound");
        kernelArguments.push_back(slots.lookup(slot));
      }
      xla::gpu::LaunchDimensions dimensions(
          se::BlockDim(launch.grid[0], launch.grid[1], launch.grid[2]),
          se::ThreadDim(launch.block[0], launch.block[1], launch.block[2]));
      if (absl::Status status = xla::gpu::ExecuteKernelOnStream(
              *initialized->executable->kernels[launch.taskOrdinal],
              kernelArguments,
              dimensions, std::nullopt, stream);
          !status.ok())
        return status;
    }
    return absl::OkStatus();
  } catch (const std::bad_alloc &) {
    return absl::ResourceExhaustedError(
        "insufficient memory to execute Shuttle GPU executable");
  } catch (...) {
    return absl::InternalError(
        "unexpected failure while executing Shuttle GPU executable");
  }
}

XLA_FFI_DEFINE_HANDLER(kInstantiate, instantiate,
                       ffi::Ffi::BindInstantiate()
                           .Attr<int64_t>("transport_schema_version")
                           .Attr<absl::string_view>("bundle_bytes")
                           .Attr<int64_t>("bundle_size")
                           .Attr<absl::string_view>("bundle_sha256")
                           .Attr<int64_t>("device_schema_version")
                           .Attr<int64_t>("invocation_abi_schema_version")
                           .Attr<int64_t>("bundle_schema_version")
                           .Attr<absl::string_view>("completion"));

XLA_FFI_DEFINE_HANDLER(kPrepare, prepare,
                       ffi::Ffi::BindPrepare()
                           .Ctx<ffi::State<InstantiateState>>()
                           .Ctx<ffi::Allocator>()
                           .Ctx<ffi::DeviceOrdinal>());

XLA_FFI_DEFINE_HANDLER(kInitialize, initialize,
                       ffi::Ffi::BindInitialize()
                           .Ctx<ffi::State<InstantiateState>>()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::TargetGpuComputeCapability>());

XLA_FFI_DEFINE_HANDLER(kExecute, execute,
                       ffi::Ffi::Bind()
                           .RemainingArgs()
                           .RemainingRets()
                           .Ctx<ffi::State<InstantiateState>>()
                           .Ctx<ffi::Prepared<PrepareState>>()
                           .Ctx<ffi::Initialized<InitializeState>>()
                           .Ctx<ffi::Stream>());

} // namespace

XLA_FFI_Handler_Bundle gpuExecutableBundleFfiHandlerBundle() {
  return {/*instantiate=*/kInstantiate,
          /*prepare=*/kPrepare,
          /*initialize=*/kInitialize,
          /*execute=*/kExecute};
}

} // namespace mlir::shuttle
