// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_RUNTIME_GPURUNNER_H_
#define SHUTTLE_RUNTIME_GPURUNNER_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"

namespace stream_executor {
class DeviceAddressAllocator;
class GpuComputeCapability;
class Stream;
} // namespace stream_executor

namespace mlir::shuttle {

// A non-owning device buffer supplied to one runner invocation.
struct GpuRunnerBuffer {
  void *data;
  int64_t sizeBytes;
};

// Resource counts derived from the sealed Shuttle executable bundle.
struct GpuRunnerResources {
  uint64_t codeBytes;
  uint64_t launchCount;
  uint64_t slotCount;
  uint64_t externalOperandCount;
  uint64_t externalResultCount;
  uint64_t temporaryAllocationCount;
  uint64_t temporaryBytes;
};

// Owns one prepared and initialized target-bound Shuttle GPU invocation.
// The caller-owned allocator and stream must outlive the runner.
class GpuRunner {
public:
  static absl::StatusOr<std::unique_ptr<GpuRunner>>
  Create(llvm::ArrayRef<uint8_t> transport,
         stream_executor::DeviceAddressAllocator *allocator,
         int32_t deviceOrdinal, stream_executor::Stream *stream,
         const stream_executor::GpuComputeCapability *capability);

  // Retries pending completion and terminates if quiescence cannot be proven.
  ~GpuRunner();
  GpuRunner(const GpuRunner &) = delete;
  GpuRunner &operator=(const GpuRunner &) = delete;
  GpuRunner(GpuRunner &&) = delete;
  GpuRunner &operator=(GpuRunner &&) = delete;

  // Enqueues the sealed launch graph and waits for stream completion. If the
  // completion fence fails, every buffer and caller-owned dependency must
  // remain alive until AwaitCompletion succeeds.
  absl::Status Execute(llvm::ArrayRef<GpuRunnerBuffer> operands,
                       llvm::ArrayRef<GpuRunnerBuffer> results);
  // Measures only the stream interval containing the sealed launch graph.
  absl::StatusOr<int64_t>
  ExecuteTimed(llvm::ArrayRef<GpuRunnerBuffer> operands,
               llvm::ArrayRef<GpuRunnerBuffer> results);
  // Retries a failed Execute completion fence without enqueueing new work.
  absl::Status AwaitCompletion();
  const GpuRunnerResources &resources() const;

private:
  class Impl;
  explicit GpuRunner(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation;
};

} // namespace mlir::shuttle

#endif // SHUTTLE_RUNTIME_GPURUNNER_H_
