// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "shuttle/Runtime/GpuRunner.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

namespace {

namespace se = stream_executor;
using mlir::shuttle::GpuRunner;
using mlir::shuttle::GpuRunnerBuffer;

constexpr std::array<int64_t, 6> kOperandBytes{
    1'048'576, 262'144, 262'144, 1'048'576, 1'048'576, 16'384};
constexpr std::array<int64_t, 3> kResultBytes{1'048'576, 262'144, 262'144};
constexpr std::array<int64_t, 3> kForwardOperandBytes{1'048'576, 262'144,
                                                     262'144};
constexpr std::array<int64_t, 2> kForwardResultBytes{1'048'576, 16'384};

std::vector<uint8_t> readBytes(const char *path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good())
    return {};
  return {std::istreambuf_iterator<char>(input), {}};
}

void expectBfloat16Close(llvm::ArrayRef<uint8_t> actual,
                         llvm::ArrayRef<uint8_t> expected, int64_t ordinal) {
  ASSERT_EQ(actual.size(), expected.size());
  ASSERT_EQ(actual.size() % sizeof(uint16_t), 0);
  for (size_t offset = 0; offset < actual.size(); offset += sizeof(uint16_t)) {
    uint16_t actualBits;
    uint16_t expectedBits;
    std::memcpy(&actualBits, actual.data() + offset, sizeof(actualBits));
    std::memcpy(&expectedBits, expected.data() + offset, sizeof(expectedBits));
    uint32_t actualFloatBits = static_cast<uint32_t>(actualBits) << 16;
    uint32_t expectedFloatBits = static_cast<uint32_t>(expectedBits) << 16;
    float actualValue;
    float expectedValue;
    std::memcpy(&actualValue, &actualFloatBits, sizeof(actualValue));
    std::memcpy(&expectedValue, &expectedFloatBits, sizeof(expectedValue));
    EXPECT_LE(std::abs(actualValue - expectedValue),
              2e-5f + 2e-5f * std::abs(expectedValue))
        << "result " << ordinal << " element " << offset / sizeof(uint16_t);
  }
}

TEST(GpuFatbinAttentionH100Test, ExecutesExportedSevenLaunchRecipe) {
  const char *bundlePath = std::getenv("SHUTTLE_TEST_ATTENTION_BUNDLE");
  const char *fixturePath = std::getenv("SHUTTLE_TEST_ATTENTION_FIXTURE");
  if (bundlePath == nullptr || fixturePath == nullptr)
    GTEST_SKIP() << "set Shuttle attention bundle and fixture paths";
  std::vector<uint8_t> bundle = readBytes(bundlePath);
  std::vector<uint8_t> fixture = readBytes(fixturePath);
  ASSERT_FALSE(bundle.empty());
  int64_t expectedFixtureBytes = 0;
  for (int64_t size : kOperandBytes)
    expectedFixtureBytes += size;
  for (int64_t size : kResultBytes)
    expectedFixtureBytes += size;
  ASSERT_EQ(fixture.size(), expectedFixtureBytes);

  auto platform = se::PlatformManager::PlatformWithName("CUDA");
  ASSERT_TRUE(platform.ok()) << platform.status();
  auto executor = (*platform)->ExecutorForDevice(0);
  ASSERT_TRUE(executor.ok()) << executor.status();
  ASSERT_NE((*executor)->GetDeviceDescription().name().find("H100"),
            std::string::npos);
  auto stream = (*executor)->CreateStream();
  ASSERT_TRUE(stream.ok()) << stream.status();
  se::StreamExecutorAddressAllocator allocator(*executor);
  const auto &capability =
      (*executor)->GetDeviceDescription().gpu_compute_capability();
  auto runner = GpuRunner::Create(bundle, &allocator, 0, stream->get(),
                                  &capability);
  ASSERT_TRUE(runner.ok()) << runner.status();
  EXPECT_EQ((*runner)->resources().launchCount, 7);
  EXPECT_EQ((*runner)->resources().slotCount, 14);
  EXPECT_EQ((*runner)->resources().externalOperandCount, 6);
  EXPECT_EQ((*runner)->resources().externalResultCount, 3);

  std::vector<se::DeviceMemory<uint8_t>> operandAllocations;
  std::vector<se::DeviceMemory<uint8_t>> resultAllocations;
  llvm::SmallVector<GpuRunnerBuffer> operands;
  llvm::SmallVector<GpuRunnerBuffer> results;
  int64_t fixtureOffset = 0;
  for (int64_t size : kOperandBytes) {
    auto allocation = (*executor)->AllocateArray<uint8_t>(size);
    ASSERT_TRUE(allocation);
    ASSERT_TRUE((*executor)
                    ->SynchronousMemcpyH2D(fixture.data() + fixtureOffset,
                                           size, &allocation)
                    .ok());
    operands.push_back({allocation.opaque(), size});
    operandAllocations.push_back(allocation);
    fixtureOffset += size;
  }
  for (int64_t size : kResultBytes) {
    auto allocation = (*executor)->AllocateArray<uint8_t>(size);
    ASSERT_TRUE(allocation);
    results.push_back({allocation.opaque(), size});
    resultAllocations.push_back(allocation);
  }

  absl::Status executeStatus = (*runner)->Execute(operands, results);
  ASSERT_TRUE(executeStatus.ok()) << executeStatus;
  for (auto [ordinal, size] : llvm::enumerate(kResultBytes)) {
    std::vector<uint8_t> actual(size);
    ASSERT_TRUE((*executor)
                    ->SynchronousMemcpyD2H(resultAllocations[ordinal], size,
                                           actual.data())
                    .ok());
    auto expected = llvm::ArrayRef<uint8_t>(fixture).slice(fixtureOffset, size);
    expectBfloat16Close(actual, expected, ordinal);
    fixtureOffset += size;
  }

  runner->reset();
  for (auto &allocation : resultAllocations)
    (*executor)->Deallocate(&allocation);
  for (auto &allocation : operandAllocations)
    (*executor)->Deallocate(&allocation);
}

TEST(GpuFatbinAttentionH100Test, ExecutesExportedForwardStateRecipe) {
  const char *bundlePath = std::getenv("SHUTTLE_TEST_ATTENTION_FORWARD_BUNDLE");
  const char *fixturePath = std::getenv("SHUTTLE_TEST_ATTENTION_FORWARD_FIXTURE");
  if (bundlePath == nullptr || fixturePath == nullptr)
    GTEST_SKIP() << "set Shuttle attention forward bundle and fixture paths";
  std::vector<uint8_t> bundle = readBytes(bundlePath);
  std::vector<uint8_t> fixture = readBytes(fixturePath);
  ASSERT_FALSE(bundle.empty());
  ASSERT_EQ(fixture.size(), 2'637'824);

  auto platform = se::PlatformManager::PlatformWithName("CUDA");
  ASSERT_TRUE(platform.ok()) << platform.status();
  auto executor = (*platform)->ExecutorForDevice(0);
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto stream = (*executor)->CreateStream();
  ASSERT_TRUE(stream.ok()) << stream.status();
  se::StreamExecutorAddressAllocator allocator(*executor);
  const auto &capability =
      (*executor)->GetDeviceDescription().gpu_compute_capability();
  auto runner = GpuRunner::Create(bundle, &allocator, 0, stream->get(),
                                  &capability);
  ASSERT_TRUE(runner.ok()) << runner.status();

  std::vector<se::DeviceMemory<uint8_t>> allocations;
  llvm::SmallVector<GpuRunnerBuffer> operands;
  llvm::SmallVector<GpuRunnerBuffer> results;
  int64_t fixtureOffset = 0;
  for (int64_t size : kForwardOperandBytes) {
    auto allocation = (*executor)->AllocateArray<uint8_t>(size);
    ASSERT_TRUE(allocation);
    ASSERT_TRUE((*executor)
                    ->SynchronousMemcpyH2D(fixture.data() + fixtureOffset,
                                           size, &allocation)
                    .ok());
    operands.push_back({allocation.opaque(), size});
    allocations.push_back(allocation);
    fixtureOffset += size;
  }
  for (int64_t size : kForwardResultBytes) {
    auto allocation = (*executor)->AllocateArray<uint8_t>(size);
    ASSERT_TRUE(allocation);
    results.push_back({allocation.opaque(), size});
    allocations.push_back(allocation);
  }
  ASSERT_TRUE((*runner)->Execute(operands, results).ok());
  for (auto [ordinal, size] : llvm::enumerate(kForwardResultBytes)) {
    std::vector<uint8_t> actual(size);
    ASSERT_TRUE((*executor)
                    ->SynchronousMemcpyD2H(allocations[operands.size() + ordinal],
                                           size, actual.data())
                    .ok());
    auto expected = llvm::ArrayRef<uint8_t>(fixture).slice(fixtureOffset, size);
    EXPECT_EQ(actual, std::vector<uint8_t>(expected.begin(), expected.end()))
        << "forward result " << ordinal;
    fixtureOffset += size;
  }
}

} // namespace
