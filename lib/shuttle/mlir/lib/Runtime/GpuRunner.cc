// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Runtime/GpuRunner.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "shuttle/Runtime/GpuFfi.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/invoke.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::shuttle {
namespace {

namespace ffi = xla::ffi;
namespace se = stream_executor;

absl::Status invalid(llvm::StringRef message) {
  return absl::InvalidArgumentError(message.str());
}

struct ExternalProjection {
  std::vector<GpuExternalBinding> operands;
  std::vector<GpuExternalBinding> results;
};

absl::StatusOr<ExternalProjection>
projectExternalBindings(const GpuExecutable &executable) {
  ExternalProjection projection;
  for (const GpuExternalBinding &binding : executable.externalBindings()) {
    std::vector<GpuExternalBinding> *bindings = nullptr;
    if (binding.kind == ExecutableBindingKind::Operand)
      bindings = &projection.operands;
    else if (binding.kind == ExecutableBindingKind::Result)
      bindings = &projection.results;
    else
      return invalid("GPU runner external binding kind is invalid");
    if (binding.index < 0 ||
        static_cast<size_t>(binding.index) != bindings->size())
      return invalid("GPU runner external bindings are not contiguous");
    bindings->push_back(binding);
  }
  if (projection.operands.empty() || projection.results.empty())
    return invalid("GPU runner requires operands and results");
  return projection;
}

void addTransportAttributes(ffi::CallFrameBuilder &builder,
                            llvm::ArrayRef<uint8_t> transport) {
  ffi::CallFrameBuilder::AttributesBuilder attributes;
  attributes.Insert(
      "bundle_bytes",
      std::string(reinterpret_cast<const char *>(transport.data()),
                  transport.size()));
  attributes.Insert("bundle_sha256", gpuExecutableBundleDigest(transport));
  attributes.Insert("bundle_size", static_cast<int64_t>(transport.size()));
  attributes.Insert("transport_schema_version", int64_t{2});
  attributes.Insert("device_schema_version", int64_t{3});
  attributes.Insert("invocation_abi_schema_version", int64_t{3});
  attributes.Insert("bundle_schema_version", int64_t{2});
  attributes.Insert("completion", std::string("stream_ordered"));
  builder.AddAttributes(attributes.Build());
}

absl::Status addBuffer(ffi::CallFrameBuilder &builder,
                       const GpuExternalBinding &binding,
                       const GpuRunnerBuffer &buffer, bool result) {
  auto tensor = dyn_cast<RankedTensorType>(binding.tensorType);
  if (!tensor || !tensor.hasStaticShape() ||
      !tensor.getElementType().isBF16() || buffer.sizeBytes <= 0 ||
      buffer.sizeBytes != binding.requiredBytes)
    return invalid("GPU runner buffer does not match the executable ABI");
  se::DeviceAddressBase address(buffer.data, buffer.sizeBytes);
  if (result)
    builder.AddBufferRet(address, xla::PrimitiveType::BF16,
                         tensor.getShape());
  else
    builder.AddBufferArg(address, xla::PrimitiveType::BF16,
                         tensor.getShape());
  return absl::OkStatus();
}

absl::StatusOr<ffi::CallFrame>
buildCallFrame(llvm::ArrayRef<uint8_t> transport,
               const ExternalProjection &projection,
               llvm::ArrayRef<GpuRunnerBuffer> operands,
               llvm::ArrayRef<GpuRunnerBuffer> results) {
  if (operands.size() != projection.operands.size() ||
      results.size() != projection.results.size())
    return invalid("GPU runner external buffer count is invalid");
  ffi::CallFrameBuilder builder(operands.size(), results.size());
  for (auto [binding, buffer] : llvm::zip_equal(projection.operands, operands))
    if (absl::Status status = addBuffer(builder, binding, buffer, false);
        !status.ok())
      return status;
  for (auto [binding, buffer] : llvm::zip_equal(projection.results, results))
    if (absl::Status status = addBuffer(builder, binding, buffer, true);
        !status.ok())
      return status;
  addTransportAttributes(builder, transport);
  return builder.Build();
}

std::pair<llvm::SmallVector<GpuRunnerBuffer>,
          llvm::SmallVector<GpuRunnerBuffer>>
placeholderBuffers(const ExternalProjection &projection) {
  llvm::SmallVector<GpuRunnerBuffer> operands;
  llvm::SmallVector<GpuRunnerBuffer> results;
  auto append = [](llvm::ArrayRef<GpuExternalBinding> bindings,
                   llvm::SmallVectorImpl<GpuRunnerBuffer> &buffers) {
    for (const GpuExternalBinding &binding : bindings) {
      uintptr_t address = UINT64_C(0x100000000) +
                          static_cast<uintptr_t>(binding.slotOrdinal) *
                              UINT64_C(0x10000000);
      buffers.push_back(
          {reinterpret_cast<void *>(address), binding.requiredBytes});
    }
  };
  append(projection.operands, operands);
  append(projection.results, results);
  return {std::move(operands), std::move(results)};
}

GpuRunnerResources runnerResources(const GpuExecutable &executable,
                                   const ExternalProjection &projection) {
  llvm::DenseMap<int64_t, uint64_t> temporaryBytes;
  for (const GpuSlot &slot : executable.slots()) {
    if (slot.storage != MaterializationStorage::Temporary)
      continue;
    uint64_t &bytes = temporaryBytes[slot.reuseGroup];
    bytes = std::max(bytes, static_cast<uint64_t>(slot.requiredBytes));
  }
  uint64_t aggregateTemporaryBytes = 0;
  for (const auto &entry : temporaryBytes)
    aggregateTemporaryBytes += entry.second;
  return {executable.codeBytes().size(),
          executable.launches().size(),
          executable.slots().size(),
          projection.operands.size(),
          projection.results.size(),
          temporaryBytes.size(),
          aggregateTemporaryBytes};
}

} // namespace

class GpuRunner::Impl {
public:
  std::vector<uint8_t> transport;
  std::shared_ptr<const GpuExecutable> executable;
  ExternalProjection projection;
  GpuRunnerResources resources;
  se::DeviceAddressAllocator *allocator;
  int32_t deviceOrdinal;
  se::Stream *stream;
  se::GpuComputeCapability capability;
  XLA_FFI_Handler_Bundle handlers;
  ffi::ExecutionState instantiateState;
  ffi::ExecutionState prepareState;
  ffi::ExecutionState initializeState;
  std::mutex executeMutex;

  ffi::InvokeContext context() {
    ffi::InvokeContext result;
    result.device_ordinal = deviceOrdinal;
    result.backend_context = ffi::InvokeContext::GpuContext{
        stream, allocator, nullptr, nullptr, nullptr, nullptr, nullptr,
        &capability};
    result.state_context = {&instantiateState, &prepareState, &initializeState};
    return result;
  }
};

GpuRunner::GpuRunner(std::unique_ptr<Impl> implementation)
    : implementation(std::move(implementation)) {}

GpuRunner::~GpuRunner() = default;
GpuRunner::GpuRunner(GpuRunner &&) noexcept = default;
GpuRunner &GpuRunner::operator=(GpuRunner &&) noexcept = default;

absl::StatusOr<std::unique_ptr<GpuRunner>> GpuRunner::Create(
    llvm::ArrayRef<uint8_t> transport, se::DeviceAddressAllocator *allocator,
    int32_t deviceOrdinal, se::Stream *stream,
    const se::GpuComputeCapability *capability) {
  if (transport.empty() || allocator == nullptr || deviceOrdinal < 0 ||
      stream == nullptr || capability == nullptr)
    return invalid("GPU runner creation context is invalid");
  absl::StatusOr<std::shared_ptr<const GpuExecutable>> executable =
      GpuExecutable::Load(transport);
  if (!executable.ok())
    return executable.status();
  absl::StatusOr<ExternalProjection> projection =
      projectExternalBindings(**executable);
  if (!projection.ok())
    return projection.status();

  auto implementation = std::make_unique<Impl>();
  implementation->transport.assign(transport.begin(), transport.end());
  implementation->executable = std::move(*executable);
  implementation->projection = std::move(*projection);
  implementation->resources = runnerResources(*implementation->executable,
                                               implementation->projection);
  implementation->allocator = allocator;
  implementation->deviceOrdinal = deviceOrdinal;
  implementation->stream = stream;
  implementation->capability = *capability;
  implementation->handlers = gpuExecutableBundleFfiHandlerBundle();

  auto [placeholderOperands, placeholderResults] =
      placeholderBuffers(implementation->projection);
  absl::StatusOr<ffi::CallFrame> frame = buildCallFrame(
      implementation->transport, implementation->projection,
      placeholderOperands, placeholderResults);
  if (!frame.ok())
    return frame.status();
  ffi::InvokeContext context = implementation->context();
  if (absl::Status status = ffi::Invoke(
          ffi::GetXlaFfiApi(), implementation->handlers.instantiate, *frame,
          context, XLA_FFI_ExecutionStage_INSTANTIATE);
      !status.ok())
    return status;
  if (absl::Status status = ffi::Invoke(
          ffi::GetXlaFfiApi(), implementation->handlers.prepare, *frame,
          context, XLA_FFI_ExecutionStage_PREPARE);
      !status.ok())
    return status;
  if (absl::Status status = ffi::Invoke(
          ffi::GetXlaFfiApi(), implementation->handlers.initialize, *frame,
          context, XLA_FFI_ExecutionStage_INITIALIZE);
      !status.ok())
    return status;
  return std::unique_ptr<GpuRunner>(
      new GpuRunner(std::move(implementation)));
}

absl::Status GpuRunner::Execute(llvm::ArrayRef<GpuRunnerBuffer> operands,
                                llvm::ArrayRef<GpuRunnerBuffer> results) {
  std::lock_guard lock(implementation->executeMutex);
  absl::StatusOr<ffi::CallFrame> frame =
      buildCallFrame(implementation->transport, implementation->projection,
                     operands, results);
  if (!frame.ok())
    return frame.status();
  ffi::InvokeContext context = implementation->context();
  return ffi::Invoke(ffi::GetXlaFfiApi(), implementation->handlers.execute,
                     *frame, context, XLA_FFI_ExecutionStage_EXECUTE);
}

const GpuRunnerResources &GpuRunner::resources() const {
  return implementation->resources;
}

} // namespace mlir::shuttle
