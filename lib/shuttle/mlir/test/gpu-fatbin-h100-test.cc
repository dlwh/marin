// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "shuttle/Runtime/GpuRunner.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {

using namespace mlir;
using namespace mlir::shuttle;
namespace se = stream_executor;

::testing::AssertionResult statusSucceeded(const absl::Status &status) {
  if (status.ok())
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << status.ToString();
}

GpuKernelArgument slotArgument(int64_t slot, ExecutableAccess access) {
  GpuKernelArgument argument;
  argument.kind = GpuKernelArgumentKind::SlotPointer;
  argument.slotOrdinal = slot;
  argument.access = access;
  return argument;
}

GpuFatbinExecutableBundle smokeBundle(MLIRContext &context,
                                      std::vector<uint8_t> fatbin) {
  auto tensor = RankedTensorType::get({16}, Float32Type::get(&context));
  std::vector<GpuSlot> slots{
      {0,
       tensor,
       64,
       {4},
       0,
       16,
       ExecutableAddressSpace::Device,
       ExecutableAccess::Read,
       MaterializationStorage::External,
       0,
       0,
       ExecutableBindingKind::Operand,
       0},
      {1,
       tensor,
       64,
       {4},
       0,
       16,
       ExecutableAddressSpace::Device,
       ExecutableAccess::ReadWrite,
       MaterializationStorage::Temporary,
       1,
       1,
       ExecutableBindingKind::None,
       -1},
      {2,
       tensor,
       64,
       {4},
       0,
       16,
       ExecutableAddressSpace::Device,
       ExecutableAccess::Write,
       MaterializationStorage::External,
       2,
       2,
       ExecutableBindingKind::Result,
       0},
  };

  GpuKernelArgument zero;
  zero.kind = GpuKernelArgumentKind::ScalarImmediate;
  zero.scalarType = GpuScalarImmediateType::I32;
  zero.bytes = {0, 0, 0, 0};
  zero.alignment = alignof(int32_t);

  GpuKernelArgument outputView;
  outputView.kind = GpuKernelArgumentKind::StructuredBytes;
  outputView.structure.size = 24;
  outputView.structure.alignment = alignof(int64_t);
  outputView.structure.fields = {
      {GpuStructuredFieldKind::SlotAddress,
       0,
       2,
       0,
       {},
       ExecutableAccess::Write},
      {GpuStructuredFieldKind::ImmediateBytes,
       8,
       -1,
       0,
       {16, 0, 0, 0, 0, 0, 0, 0}},
      {GpuStructuredFieldKind::ImmediateBytes,
       16,
       -1,
       0,
       {1, 0, 0, 0, 0, 0, 0, 0}},
  };

  GpuKernelArgument tag;
  tag.kind = GpuKernelArgumentKind::PackedBytes;
  tag.bytes = {0, 0, 0, 0};
  tag.alignment = alignof(uint32_t);

  GpuLaunch stage0{
      0,
      "shuttle_stage0",
      0,
      0,
      "",
      {1, 1, 1},
      {32, 1, 1},
      0,
      {slotArgument(0, ExecutableAccess::Read),
       slotArgument(1, ExecutableAccess::Write), zero},
      {},
      {0},
      {1},
      {},
      {0},
  };
  GpuLaunch stage1{
      1,
      "shuttle_stage1",
      0,
      0,
      "",
      {1, 1, 1},
      {32, 1, 1},
      0,
      {slotArgument(1, ExecutableAccess::ReadWrite), outputView, tag},
      {},
      {1},
      {1, 2},
      {0},
      {1},
  };
  return {5,
          ExecutableCodeFormat::CudaFatbinSm90V1,
          std::string(64, 'c'),
          std::string(64, 'd'),
          {std::move(fatbin)},
          {std::move(stage0), std::move(stage1)},
          std::move(slots)};
}

TEST(GpuFatbinH100Test,
     ExecutesNamedMultiLaunchScratchAndStructuredRelocation) {
  const char *fatbinPath = std::getenv("SHUTTLE_TEST_FATBIN");
  if (fatbinPath == nullptr)
    GTEST_SKIP() << "set SHUTTLE_TEST_FATBIN to the smoke fatbin path";
  std::ifstream input(fatbinPath, std::ios::binary);
  ASSERT_TRUE(input.good()) << fatbinPath;
  std::vector<uint8_t> fatbin{std::istreambuf_iterator<char>(input), {}};
  ASSERT_FALSE(fatbin.empty());

  MLIRContext context;
  auto transport = serializeGpuFatbinExecutableBundle(
      smokeBundle(context, std::move(fatbin)));
  ASSERT_TRUE(transport.ok()) << transport.status();

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
  auto runner =
      GpuRunner::Create(*transport, &allocator, 0, stream->get(), &capability);
  ASSERT_TRUE(runner.ok()) << runner.status();
  EXPECT_EQ((*runner)->resources().launchCount, 2);
  EXPECT_EQ((*runner)->resources().temporaryAllocationCount, 1);
  EXPECT_EQ((*runner)->resources().temporaryBytes, 64);

  std::vector<float> hostInput(16);
  for (int64_t index = 0; index < 16; ++index)
    hostInput[index] = static_cast<float>(index) - 8.0f;
  std::vector<float> actual(16, -999.0f);
  auto deviceInput = (*executor)->AllocateArray<float>(hostInput.size());
  auto deviceOutput = (*executor)->AllocateArray<float>(actual.size());
  ASSERT_TRUE(deviceInput);
  ASSERT_TRUE(deviceOutput);
  ASSERT_TRUE(statusSucceeded(
      (*executor)->SynchronousMemcpyH2D(hostInput.data(), 64, &deviceInput)));
  ASSERT_TRUE(statusSucceeded(
      (*executor)->SynchronousMemcpyH2D(actual.data(), 64, &deviceOutput)));
  llvm::SmallVector<GpuRunnerBuffer> operands{{deviceInput.opaque(), 64}};
  llvm::SmallVector<GpuRunnerBuffer> results{{deviceOutput.opaque(), 64}};

  ASSERT_TRUE(statusSucceeded((*runner)->Execute(operands, results)));
  ASSERT_TRUE(statusSucceeded(
      (*executor)->SynchronousMemcpyD2H(deviceOutput, 64, actual.data())));
  for (int64_t index = 0; index < 16; ++index)
    EXPECT_EQ(actual[index], hostInput[index] * 2.0f + 1.0f) << index;

  runner->reset();
  (*executor)->Deallocate(&deviceOutput);
  (*executor)->Deallocate(&deviceInput);
}

TEST(GpuFatbinH100Test, ResolvesDuplicateKernelNamesPerImageWithScalarResult) {
  const char *firstPath = std::getenv("SHUTTLE_TEST_FIRST_IMAGE");
  const char *secondPath = std::getenv("SHUTTLE_TEST_SECOND_IMAGE");
  if (firstPath == nullptr || secondPath == nullptr)
    GTEST_SKIP() << "set both independently compiled Shuttle image paths";
  std::ifstream firstInput(firstPath, std::ios::binary);
  std::ifstream secondInput(secondPath, std::ios::binary);
  ASSERT_TRUE(firstInput.good()) << firstPath;
  ASSERT_TRUE(secondInput.good()) << secondPath;
  std::vector<uint8_t> first{std::istreambuf_iterator<char>(firstInput), {}};
  std::vector<uint8_t> second{std::istreambuf_iterator<char>(secondInput), {}};
  ASSERT_FALSE(first.empty());
  ASSERT_FALSE(second.empty());
  ASSERT_NE(first, second);

  MLIRContext context;
  auto bundle = smokeBundle(context, std::move(first));
  bundle.images.push_back(std::move(second));
  for (auto &slot : bundle.slots) {
    slot.tensorType = RankedTensorType::get({}, Float32Type::get(&context));
    slot.requiredBytes = sizeof(float);
    slot.byteStrides.clear();
  }
  for (auto &launch : bundle.launches) {
    launch.kernelName = "shuttle_shared_symbol";
    launch.arguments = {
        slotArgument(launch.taskOrdinal, ExecutableAccess::Read),
        slotArgument(launch.taskOrdinal + 1, ExecutableAccess::Write)};
    launch.inputSlots = {launch.taskOrdinal};
    launch.outputSlots = {launch.taskOrdinal + 1};
    launch.imageIndex = launch.taskOrdinal;
  }
  auto transport = serializeGpuFatbinExecutableBundle(bundle);
  ASSERT_TRUE(transport.ok()) << transport.status();
  auto platform = se::PlatformManager::PlatformWithName("CUDA");
  ASSERT_TRUE(platform.ok()) << platform.status();
  auto executor = (*platform)->ExecutorForDevice(0);
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto stream = (*executor)->CreateStream();
  ASSERT_TRUE(stream.ok()) << stream.status();
  se::StreamExecutorAddressAllocator allocator(*executor);
  const auto &capability =
      (*executor)->GetDeviceDescription().gpu_compute_capability();
  auto runner =
      GpuRunner::Create(*transport, &allocator, 0, stream->get(), &capability);
  ASSERT_TRUE(runner.ok()) << runner.status();
  EXPECT_EQ((*runner)->resources().externalResultCount, 1);
  EXPECT_EQ((*runner)->resources().launchCount, 2);
  EXPECT_EQ((*runner)->resources().temporaryBytes, sizeof(float));

  auto input = (*executor)->AllocateArray<float>(1);
  auto output = (*executor)->AllocateArray<float>(1);
  ASSERT_TRUE(input);
  ASSERT_TRUE(output);
  float value = 3.0f;
  ASSERT_TRUE(statusSucceeded(
      (*executor)->SynchronousMemcpyH2D(&value, sizeof(value), &input)));
  llvm::SmallVector<GpuRunnerBuffer> operands{{input.opaque(), sizeof(float)}};
  llvm::SmallVector<GpuRunnerBuffer> results{{output.opaque(), sizeof(float)}};
  ASSERT_TRUE(statusSucceeded((*runner)->Execute(operands, results)));
  float result = 0.0f;
  ASSERT_TRUE(statusSucceeded(
      (*executor)->SynchronousMemcpyD2H(output, sizeof(result), &result)));
  // Image zero adds one; image one doubles. A symbol-only cache produces five.
  EXPECT_EQ(result, 8.0f);

  auto alternateInput = (*executor)->AllocateArray<float>(1);
  auto alternateOutput = (*executor)->AllocateArray<float>(1);
  ASSERT_TRUE(alternateInput);
  ASSERT_TRUE(alternateOutput);
  for (int iteration = 0; iteration < 6; ++iteration) {
    auto &currentInput = iteration % 2 ? alternateInput : input;
    auto &currentOutput = iteration % 2 ? alternateOutput : output;
    auto &unusedOutput = iteration % 2 ? output : alternateOutput;
    value = static_cast<float>(iteration) - 4.0f;
    float sentinel = -999.0f;
    ASSERT_TRUE(statusSucceeded((*executor)->SynchronousMemcpyH2D(
        &value, sizeof(value), &currentInput)));
    ASSERT_TRUE(statusSucceeded((*executor)->SynchronousMemcpyH2D(
        &sentinel, sizeof(sentinel), &unusedOutput)));
    operands[0].data = currentInput.opaque();
    results[0].data = currentOutput.opaque();

    // Reject invalid bindings before mutating the reusable call frame. The
    // following valid call must still use its own input and output addresses.
    auto wrongSize = results;
    wrongSize[0].sizeBytes += sizeof(float);
    EXPECT_EQ((*runner)->Execute(operands, wrongSize).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ((*runner)->Execute(operands, operands).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ((*runner)->Execute({}, results).code(),
              absl::StatusCode::kInvalidArgument);
    auto misaligned = results;
    misaligned[0].data = static_cast<uint8_t *>(currentOutput.opaque()) + 1;
    EXPECT_EQ((*runner)->ExecuteTimed(operands, misaligned).status().code(),
              absl::StatusCode::kInvalidArgument);
    auto nullOutput = results;
    nullOutput[0].data = nullptr;
    EXPECT_EQ((*runner)->ExecuteProfiled(operands, nullOutput).status().code(),
              absl::StatusCode::kInvalidArgument);

    if (iteration % 3 == 0) {
      ASSERT_TRUE(statusSucceeded((*runner)->Execute(operands, results)));
    } else if (iteration % 3 == 1) {
      auto timed = (*runner)->ExecuteTimed(operands, results);
      ASSERT_TRUE(timed.ok()) << timed.status();
    } else {
      auto profiled = (*runner)->ExecuteProfiled(operands, results);
      ASSERT_TRUE(profiled.ok()) << profiled.status();
    }
    ASSERT_TRUE(statusSucceeded((*executor)->SynchronousMemcpyD2H(
        currentOutput, sizeof(result), &result)));
    EXPECT_EQ(result, 2.0f * (value + 1.0f));
    ASSERT_TRUE(statusSucceeded((*executor)->SynchronousMemcpyD2H(
        unusedOutput, sizeof(result), &result)));
    EXPECT_EQ(result, sentinel);
  }
  runner->reset();
  (*executor)->Deallocate(&alternateOutput);
  (*executor)->Deallocate(&alternateInput);
  (*executor)->Deallocate(&output);
  (*executor)->Deallocate(&input);
}

TEST(GpuFatbinH100Test, ExecutesExportedComposedPublicBoundary) {
  const char *directory =
      std::getenv("SHUTTLE_TEST_COMPOSED_FIXTURE_DIRECTORY");
  if (directory == nullptr)
    GTEST_SKIP() << "set composed fixture directory containing manifest.json";
  auto read = [directory](llvm::StringRef name) {
    std::ifstream input(std::filesystem::path(directory) / name.str(),
                        std::ios::binary);
    return std::vector<uint8_t>{std::istreambuf_iterator<char>(input), {}};
  };
  auto manifestBytes = read("manifest.json");
  auto manifest = llvm::json::parse(
      llvm::StringRef(reinterpret_cast<const char *>(manifestBytes.data()),
                      manifestBytes.size()));
  ASSERT_TRUE(static_cast<bool>(manifest));
  auto *object = manifest->getAsObject();
  ASSERT_NE(object, nullptr);
  auto bundlePath = object->getString("bundle");
  auto *inputPaths = object->getArray("inputs");
  auto *outputPaths = object->getArray("expected_outputs");
  auto tolerance = object->getNumber("max_absolute_error");
  ASSERT_TRUE(bundlePath.has_value());
  ASSERT_NE(inputPaths, nullptr);
  ASSERT_NE(outputPaths, nullptr);
  ASSERT_TRUE(tolerance.has_value());
  ASSERT_TRUE(std::isfinite(*tolerance));
  ASSERT_GE(*tolerance, 0.0);
  auto transport = read(*bundlePath);
  MLIRContext context;
  auto bundle = deserializeGpuFatbinExecutableBundle(transport, &context);
  ASSERT_TRUE(bundle.ok()) << bundle.status();
  std::vector<GpuSlot> operandSlots;
  std::vector<GpuSlot> resultSlots;
  for (const auto &slot : bundle->slots) {
    if (slot.bindingKind == ExecutableBindingKind::Operand)
      operandSlots.push_back(slot);
    else if (slot.bindingKind == ExecutableBindingKind::Result)
      resultSlots.push_back(slot);
  }
  auto bindingOrder = [](const GpuSlot &left, const GpuSlot &right) {
    return left.bindingIndex < right.bindingIndex;
  };
  std::sort(operandSlots.begin(), operandSlots.end(), bindingOrder);
  std::sort(resultSlots.begin(), resultSlots.end(), bindingOrder);
  ASSERT_EQ(inputPaths->size(), operandSlots.size());
  ASSERT_EQ(outputPaths->size(), resultSlots.size());
  auto platform = se::PlatformManager::PlatformWithName("CUDA");
  ASSERT_TRUE(platform.ok()) << platform.status();
  auto executor = (*platform)->ExecutorForDevice(0);
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto stream = (*executor)->CreateStream();
  ASSERT_TRUE(stream.ok()) << stream.status();
  se::StreamExecutorAddressAllocator allocator(*executor);
  const auto &capability =
      (*executor)->GetDeviceDescription().gpu_compute_capability();
  auto runner =
      GpuRunner::Create(transport, &allocator, 0, stream->get(), &capability);
  ASSERT_TRUE(runner.ok()) << runner.status();
  std::vector<se::DeviceMemory<uint8_t>> allocations;
  llvm::SmallVector<GpuRunnerBuffer> operands;
  llvm::SmallVector<GpuRunnerBuffer> results;
  for (const auto &slot : operandSlots) {
    ASSERT_EQ(slot.bindingIndex, operands.size());
    auto path = (*inputPaths)[slot.bindingIndex].getAsString();
    ASSERT_TRUE(path.has_value());
    auto bytes = read(*path);
    ASSERT_EQ(bytes.size(), slot.requiredBytes);
    auto allocation = (*executor)->AllocateArray<uint8_t>(slot.requiredBytes);
    ASSERT_TRUE(allocation);
    ASSERT_TRUE(statusSucceeded((*executor)->SynchronousMemcpyH2D(
        bytes.data(), bytes.size(), &allocation)));
    operands.push_back({allocation.opaque(), slot.requiredBytes});
    allocations.push_back(allocation);
  }
  std::vector<se::DeviceMemory<uint8_t>> resultAllocations;
  for (const auto &slot : resultSlots) {
    ASSERT_EQ(slot.bindingIndex, results.size());
    auto allocation = (*executor)->AllocateArray<uint8_t>(slot.requiredBytes);
    ASSERT_TRUE(allocation);
    results.push_back({allocation.opaque(), slot.requiredBytes});
    resultAllocations.push_back(allocation);
  }
  ASSERT_TRUE(statusSucceeded((*runner)->Execute(operands, results)));
  llvm::json::Array outputErrors;
  bool accepted = true;
  for (size_t ordinal = 0; ordinal < resultSlots.size(); ++ordinal) {
    auto path = (*outputPaths)[ordinal].getAsString();
    ASSERT_TRUE(path.has_value());
    auto expected = read(*path);
    std::vector<uint8_t> actual(resultSlots[ordinal].requiredBytes);
    ASSERT_EQ(expected.size(), actual.size());
    ASSERT_TRUE(statusSucceeded((*executor)->SynchronousMemcpyD2H(
        resultAllocations[ordinal], actual.size(), actual.data())));
    auto elementType =
        llvm::cast<RankedTensorType>(resultSlots[ordinal].tensorType)
            .getElementType();
    ASSERT_TRUE(elementType.isF32() || elementType.isBF16());
    size_t width = elementType.isBF16() ? 2 : 4;
    auto value = [width](const uint8_t *bytes) {
      uint32_t bits;
      if (width == 2) {
        uint16_t narrow;
        std::memcpy(&narrow, bytes, sizeof(narrow));
        bits = static_cast<uint32_t>(narrow) << 16;
      } else {
        std::memcpy(&bits, bytes, sizeof(bits));
      }
      float decoded;
      std::memcpy(&decoded, &bits, sizeof(decoded));
      return decoded;
    };
    double maximum = 0.0;
    bool finite = true;
    for (size_t offset = 0; offset < actual.size(); offset += width) {
      float result = value(actual.data() + offset);
      float reference = value(expected.data() + offset);
      finite &= std::isfinite(result) && std::isfinite(reference);
      maximum =
          std::max(maximum, std::abs(static_cast<double>(result) - reference));
    }
    outputErrors.push_back(
        llvm::json::Object{{"output", static_cast<int64_t>(ordinal)},
                           {"max_abs", maximum},
                           {"finite", finite}});
    accepted &= finite && maximum <= *tolerance;
  }
  llvm::json::Object report{{"numerical_acceptance", accepted},
                            {"errors", std::move(outputErrors)},
                            {"max_absolute_error", *tolerance}};
  if (accepted) {
    std::vector<double> hostSamples;
    std::vector<double> deviceSamples;
    llvm::json::Array profiles;
    for (int iteration = 0; iteration < 10; ++iteration)
      ASSERT_TRUE(statusSucceeded((*runner)->Execute(operands, results)));
    for (int iteration = 0; iteration < 30; ++iteration) {
      auto start = std::chrono::steady_clock::now();
      ASSERT_TRUE(statusSucceeded((*runner)->Execute(operands, results)));
      hostSamples.push_back(std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count());
      auto deviceTime = (*runner)->ExecuteTimed(operands, results);
      ASSERT_TRUE(deviceTime.ok()) << deviceTime.status();
      deviceSamples.push_back(*deviceTime / 1e6);
      auto profile = (*runner)->ExecuteProfiled(operands, results);
      ASSERT_TRUE(profile.ok()) << profile.status();
      profiles.push_back(llvm::json::Object{
          {"frame_ms", profile->frameNanoseconds / 1e6},
          {"enqueue_ms", profile->enqueueNanoseconds / 1e6},
          {"completion_ms", profile->completionNanoseconds / 1e6}});
    }
    report["host_samples_ms"] = llvm::json::Array(hostSamples);
    report["stream_samples_ms"] = llvm::json::Array(deviceSamples);
    report["host_profiles"] = std::move(profiles);
    report["warmup_iterations"] = 10;
    std::sort(hostSamples.begin(), hostSamples.end());
    std::sort(deviceSamples.begin(), deviceSamples.end());
    report["median_host_ms"] = (hostSamples[14] + hostSamples[15]) / 2;
    report["median_device_ms"] = (deviceSamples[14] + deviceSamples[15]) / 2;
    report["timing_note"] =
        "Host includes GpuRunner::Execute and completion "
        "fence; device uses ExecuteTimed stream interval. "
        "Profiles use separate invocations without device "
        "timer events; enqueue includes argument preparation "
        "and submission; completion is the remaining wait.";
  }
  llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(std::move(report)))
               << "\n";
  // GTest writes through a different stdout buffer. Finish the JSON before
  // its test-status lines can appear between raw_ostream buffer chunks.
  llvm::outs().flush();
  EXPECT_TRUE(accepted);
  runner->reset();
  for (auto &allocation : resultAllocations)
    (*executor)->Deallocate(&allocation);
  for (auto &allocation : allocations)
    (*executor)->Deallocate(&allocation);
}

} // namespace
