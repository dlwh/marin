// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/IR/ShuttleOps.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "shuttle/Runtime/GpuFfi.h"
#include "shuttle/Runtime/GpuRunner.h"
#include "shuttle/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "tools/cpp/runfiles/runfiles.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/invoke.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "gtest/gtest.h"

namespace {

std::vector<mlir::shuttle::GpuExternalBinding>
externalBindingsBySlot(const mlir::shuttle::GpuExecutable &executable) {
  std::vector<mlir::shuttle::GpuExternalBinding> bindings(
      executable.externalBindings().begin(),
      executable.externalBindings().end());
  llvm::sort(bindings, [](const auto &left, const auto &right) {
    return left.slotOrdinal < right.slotOrdinal;
  });
  return bindings;
}

class Reader {
public:
  explicit Reader(llvm::ArrayRef<uint8_t> bytes) : bytes(bytes) {}

  uint8_t u8() { return integer<uint8_t>(); }
  uint16_t u16() { return integer<uint16_t>(); }
  uint32_t u32() { return integer<uint32_t>(); }
  uint64_t u64() { return integer<uint64_t>(); }
  int64_t i64() { return static_cast<int64_t>(u64()); }
  llvm::StringRef text(uint64_t *payloadOffset = nullptr) {
    uint64_t size = u64();
    if (payloadOffset)
      *payloadOffset = position();
    auto value = take(size);
    return {reinterpret_cast<const char *>(value.data()), value.size()};
  }
  llvm::ArrayRef<uint8_t> blob(uint64_t *payloadOffset = nullptr) {
    uint64_t size = u64();
    if (payloadOffset)
      *payloadOffset = position();
    return take(size);
  }
  llvm::SmallVector<int64_t> integers() {
    uint64_t count = u64();
    if (count > 256) {
      failed = true;
      return {};
    }
    llvm::SmallVector<int64_t> values;
    for (uint64_t index = 0; index < count; ++index)
      values.push_back(i64());
    return values;
  }
  bool ok() const { return !failed; }
  bool done() const { return ok() && offset == bytes.size(); }
  uint64_t position() const { return offset; }

private:
  llvm::ArrayRef<uint8_t> take(uint64_t count) {
    if (failed || count > bytes.size() - offset) {
      failed = true;
      return {};
    }
    auto value = bytes.slice(offset, count);
    offset += count;
    return value;
  }
  template <typename T> T integer() {
    auto encoded = take(sizeof(T));
    if (encoded.size() != sizeof(T))
      return 0;
    T value = 0;
    for (unsigned index = 0; index < sizeof(T); ++index)
      value |= static_cast<T>(encoded[index]) << (index * 8);
    return value;
  }

  llvm::ArrayRef<uint8_t> bytes;
  uint64_t offset = 0;
  bool failed = false;
};

std::string sha256(llvm::ArrayRef<uint8_t> bytes) {
  llvm::SHA256 sha;
  sha.update(bytes);
  return llvm::toHex(sha.final(), true);
}

std::string sha256(llvm::ArrayRef<uint8_t> prefix,
                   llvm::ArrayRef<uint8_t> suffix) {
  llvm::SHA256 sha;
  sha.update(prefix);
  sha.update(suffix);
  return llvm::toHex(sha.final(), true);
}

std::string performanceWorkloadDigest(
    llvm::StringRef caseName, llvm::StringRef policy,
    llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<std::vector<uint16_t>> operands) {
  llvm::SHA256 sha;
  auto updateInteger = [&](uint64_t value) {
    std::array<uint8_t, 8> bytes;
    for (size_t index = 0; index < bytes.size(); ++index)
      bytes[index] = static_cast<uint8_t>(value >> (index * 8));
    sha.update(bytes);
  };
  auto updateText = [&](llvm::StringRef value) {
    updateInteger(value.size());
    sha.update(llvm::ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(value.data()), value.size()));
  };
  updateText("shuttle-h100-performance-workload-v1");
  updateText(caseName);
  updateText(policy);
  updateInteger(shape.size());
  for (int64_t extent : shape)
    updateInteger(static_cast<uint64_t>(extent));
  updateInteger(operands.size());
  for (const std::vector<uint16_t> &operand : operands) {
    updateInteger(operand.size());
    for (uint16_t bits : operand) {
      std::array<uint8_t, 2> bytes{static_cast<uint8_t>(bits),
                                   static_cast<uint8_t>(bits >> 8)};
      sha.update(bytes);
    }
  }
  return llvm::toHex(sha.final(), true);
}

TEST(GpuTransportV2Test,
     PerformanceWorkloadDigestMatchesProducerIndependentProtocol) {
  std::vector<int64_t> shape{2, 2};
  std::vector<std::vector<uint16_t>> operands{{1, 2, 3, 4}, {5, 6}};
  EXPECT_EQ(performanceWorkloadDigest("probe", "fast", shape, operands),
            "1a39c50a5f3ab75d02bad51a11a33175b8910c8292d7570ccc64d76f932ee1c2");
}

class PacketWriter {
public:
  void u8(uint8_t value) { bytes.push_back(value); }
  void u16(uint16_t value) { integer(value); }
  void u32(uint32_t value) { integer(value); }
  void u64(uint64_t value) { integer(value); }
  void i64(int64_t value) { u64(static_cast<uint64_t>(value)); }
  void append(llvm::ArrayRef<uint8_t> value) {
    bytes.append(value.begin(), value.end());
  }
  void append(llvm::StringRef value) {
    append(llvm::ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(value.data()), value.size()));
  }
  void text(llvm::StringRef value) {
    u64(value.size());
    append(value);
  }
  void blob(llvm::ArrayRef<uint8_t> value) {
    u64(value.size());
    append(value);
  }
  void integers(llvm::ArrayRef<int64_t> values) {
    u64(values.size());
    for (int64_t value : values)
      i64(value);
  }
  llvm::ArrayRef<uint8_t> view() const { return bytes; }
  llvm::SmallVector<uint8_t> take() { return std::move(bytes); }

private:
  template <typename T> void integer(T value) {
    for (size_t index = 0; index < sizeof(T); ++index)
      bytes.push_back(static_cast<uint8_t>(value >> (index * 8)));
  }
  llvm::SmallVector<uint8_t> bytes;
};

llvm::SmallVector<uint8_t>
completeSchema2Packet(llvm::ArrayRef<uint64_t> codeLengths,
                      uint32_t gridX = 1, uint32_t sharedBytes = 0,
                      uint64_t temporaryRank = 1) {
  if (codeLengths.size() < 2)
    return {};
  const std::string schedule(64, 'a');
  const std::string plan(64, 'b');
  constexpr llvm::StringLiteral ptxMarker =
      ".version 8.0\n.target sm_90\n.address_size 64\n"
      ".visible .entry shuttle_entry(\n)\n{\n  ret;\n}\n";

  llvm::SmallVector<uint8_t> code;
  llvm::SmallVector<std::string> entryDigests;
  for (uint64_t length : codeLengths) {
    if (length < ptxMarker.size())
      return {};
    llvm::SmallVector<uint8_t> slice(length, ' ');
    llvm::copy(ptxMarker, slice.begin());
    slice.back() = '\n';
    entryDigests.push_back(sha256(slice));
    code.append(slice.begin(), slice.end());
  }

  PacketWriter devicePrefix;
  devicePrefix.i64(3);
  devicePrefix.u8(2);
  devicePrefix.u8(0);
  devicePrefix.text(schedule);
  devicePrefix.blob(code);
  devicePrefix.text(sha256(code));
  PacketWriter deviceSuffix;
  deviceSuffix.u64(codeLengths.size());
  int64_t codeOffset = 0;
  for (auto [ordinal, length] : llvm::enumerate(codeLengths)) {
    const int64_t input = ordinal;
    const int64_t output = ordinal + 1;
    deviceSuffix.i64(ordinal);
    deviceSuffix.i64(ordinal);
    deviceSuffix.i64(codeOffset);
    deviceSuffix.i64(length);
    deviceSuffix.integers(llvm::ArrayRef<int64_t>(input));
    deviceSuffix.integers(llvm::ArrayRef<int64_t>(output));
    deviceSuffix.u64(1);
    deviceSuffix.u8(0);
    deviceSuffix.u64(1);
    deviceSuffix.u8(1);
    if (ordinal == 0) {
      deviceSuffix.integers({});
    } else {
      const int64_t dependency = ordinal - 1;
      deviceSuffix.integers(llvm::ArrayRef<int64_t>(dependency));
    }
    deviceSuffix.u8(1);
    deviceSuffix.u8(0);
    deviceSuffix.text(entryDigests[ordinal]);
    deviceSuffix.u32(gridX);
    deviceSuffix.u32(1);
    deviceSuffix.u32(1);
    deviceSuffix.u16(1);
    deviceSuffix.u16(1);
    deviceSuffix.u16(1);
    deviceSuffix.u32(sharedBytes);
    deviceSuffix.u16(2);
    codeOffset += length;
  }
  const std::string deviceRoot =
      sha256(devicePrefix.view(), deviceSuffix.view());

  PacketWriter invocationPrefix;
  invocationPrefix.i64(3);
  invocationPrefix.text(plan);
  invocationPrefix.text(schedule);
  PacketWriter invocationSuffix;
  const uint64_t slotCount = codeLengths.size() + 1;
  invocationSuffix.u64(slotCount);
  for (uint64_t ordinal = 0; ordinal < slotCount; ++ordinal) {
    const bool external = ordinal == 0 || ordinal + 1 == slotCount;
    const uint8_t element = external ? 0 : 1;
    const uint64_t rank = !external && ordinal == 1 ? temporaryRank : 1;
    const int64_t width = element == 0 ? 2 : 4;
    invocationSuffix.i64(ordinal);
    invocationSuffix.i64(ordinal);
    invocationSuffix.u8(element);
    invocationSuffix.u64(rank);
    for (uint64_t axis = 0; axis < rank; ++axis)
      invocationSuffix.i64(1);
    invocationSuffix.i64(width);
    llvm::SmallVector<int64_t> strides(rank, width);
    invocationSuffix.integers(strides);
    invocationSuffix.i64(0);
    invocationSuffix.i64(width);
    invocationSuffix.u8(1);
    invocationSuffix.u8(ordinal == 0 ? 0 : (external ? 1 : 2));
    invocationSuffix.u8(external ? 0 : 1);
    invocationSuffix.i64(ordinal);
    invocationSuffix.i64(ordinal);
    invocationSuffix.u8(ordinal == 0 ? 1 : (external ? 2 : 0));
    invocationSuffix.u8(external ? 1 : 0);
    if (external)
      invocationSuffix.i64(0);
  }
  const std::string invocationRoot =
      sha256(invocationPrefix.view(), invocationSuffix.view());

  PacketWriter packet;
  packet.append(llvm::StringRef("SHUTGPU\0", 8));
  packet.u32(2);
  packet.append(devicePrefix.view());
  packet.text(deviceRoot);
  packet.append(deviceSuffix.view());
  packet.append(invocationPrefix.view());
  packet.text(invocationRoot);
  packet.append(invocationSuffix.view());
  packet.i64(2);
  packet.text(schedule);
  packet.text(deviceRoot);
  packet.text(invocationRoot);
  packet.u8(1);
  packet.text(sha256(packet.view()));
  return packet.take();
}

bool isDigest(llvm::StringRef value) {
  return value.size() == 64 && llvm::all_of(value, [](char character) {
           return llvm::isDigit(character) ||
                  (character >= 'a' && character <= 'f');
         });
}

struct Decoded {
  llvm::SmallVector<uint8_t> code;
  llvm::SmallVector<mlir::shuttle::GpuLaunch> launches;
  uint8_t policy = 0;
  uint64_t devicePolicyOffset = 0;
  llvm::SmallVector<int64_t> slotReuseGroups;
  llvm::SmallVector<uint64_t> slotReuseOffsets;
  llvm::SmallVector<uint8_t> slotStorage;
  llvm::SmallVector<uint8_t> slotBinding;
  llvm::SmallVector<int64_t> slotBindingIndex;
  llvm::SmallVector<uint64_t> slotBindingIndexOffsets;
  llvm::SmallVector<llvm::SmallVector<int64_t>> slotDimensions;
  llvm::SmallVector<llvm::SmallVector<int64_t>> slotStrides;
  llvm::SmallVector<int64_t> slotRequiredBytes;
  llvm::SmallVector<int64_t> slotProducers;
  llvm::SmallVector<int64_t> slotFinalUses;
  llvm::SmallVector<uint64_t> launchFirstInputOffsets;
  llvm::SmallVector<int64_t> temporaryGroupOrder;
  llvm::SmallVector<uint64_t> temporaryGroupBytes;
  int64_t slots = 0;
  uint64_t temporaryBytes = 0;
  int64_t scalarSlots = 0;
  int64_t externalSlots = 0;
  uint64_t deviceRootOffset = 0;
  uint64_t deviceRootPayloadOffset = 0;
  uint64_t deviceSuffixStart = 0;
  uint64_t invocationStart = 0;
  uint64_t invocationRootOffset = 0;
  uint64_t invocationRootPayloadOffset = 0;
  uint64_t invocationSuffixStart = 0;
  uint64_t bundleStart = 0;
  uint64_t bundleDeviceRootPayloadOffset = 0;
  uint64_t bundleInvocationRootPayloadOffset = 0;
  uint64_t bundleRootOffset = 0;
  uint64_t bundleRootPayloadOffset = 0;
  llvm::SmallVector<uint64_t> launchGridOffsets;
  uint64_t codeLengthOffset = 0;
  uint64_t codePayloadOffset = 0;
  uint64_t codeDigestPayloadOffset = 0;
  uint64_t deviceEntryCountOffset = 0;
  llvm::SmallVector<uint64_t> entryLengthOffsets;
  llvm::SmallVector<uint64_t> entryDigestPayloadOffsets;
  llvm::SmallVector<uint64_t> launchSharedOffsets;
  uint64_t invocationSlotCountOffset = 0;
  llvm::SmallVector<uint64_t> slotRankOffsets;
  llvm::SmallVector<llvm::SmallVector<uint64_t>> slotDimensionOffsets;
  llvm::SmallVector<uint64_t> slotRequiredOffsets;
};

struct DecodeLimits {
  uint64_t transportBytes = 16 * 1024 * 1024;
  uint64_t codeBytes = 8 * 1024 * 1024;
  uint64_t records = 256;
  uint64_t entryCodeBytes = 512 * 1024;
  uint64_t taskPositions = 256 * 1024 * 1024;
  uint64_t sharedBytes = 16 * 1024;
  uint64_t rank = 8;
  uint64_t slotBytes = 32 * 1024 * 1024;
  uint64_t temporaryBytes = 256 * 1024 * 1024;
};

std::optional<Decoded>
decodeInlineSchema2(llvm::ArrayRef<uint8_t> bytes,
                    DecodeLimits limits = {}) {
  if (bytes.size() > limits.transportBytes)
    return std::nullopt;
  Reader reader(bytes);
  for (char expected : llvm::StringRef("SHUTGPU\0", 8))
    if (reader.u8() != static_cast<uint8_t>(expected))
      return std::nullopt;
  if (reader.u32() != 2)
    return std::nullopt;
  const int64_t deviceSchema = reader.i64();
  const uint8_t codeFormat = reader.u8();
  if (!((deviceSchema == 3 && codeFormat == 2) ||
        (deviceSchema == 4 && codeFormat == 3)))
    return std::nullopt;
  uint64_t devicePolicyOffset = reader.position();
  uint8_t policy = reader.u8();
  if (policy > 1)
    return std::nullopt;
  constexpr uint64_t deviceStart = 12;
  llvm::StringRef schedule = reader.text();
  if (!isDigest(schedule))
    return std::nullopt;
  uint64_t codeLengthOffset = reader.position();
  uint64_t codePayloadOffset = 0;
  auto code = reader.blob(&codePayloadOffset);
  uint64_t codeDigestPayloadOffset = 0;
  if (code.size() > limits.codeBytes ||
      reader.text(&codeDigestPayloadOffset) != sha256(code))
    return std::nullopt;
  uint64_t deviceRootOffset = reader.position();
  uint64_t deviceRootPayloadOffset = 0;
  llvm::StringRef deviceRoot = reader.text(&deviceRootPayloadOffset);
  uint64_t deviceSuffixStart = reader.position();
  uint64_t deviceEntryCountOffset = reader.position();
  uint64_t entryCount = reader.u64();
  if (!isDigest(deviceRoot) || entryCount == 0 || entryCount > limits.records)
    return std::nullopt;

  llvm::SmallVector<mlir::shuttle::GpuLaunch> launches;
  llvm::SmallVector<uint64_t> launchGridOffsets;
  llvm::SmallVector<uint64_t> launchFirstInputOffsets;
  llvm::SmallVector<uint64_t> entryLengthOffsets;
  llvm::SmallVector<uint64_t> entryDigestPayloadOffsets;
  llvm::SmallVector<uint64_t> launchSharedOffsets;
  int64_t nextOffset = 0;
  uint64_t taskPositions = 0;
  for (int64_t ordinal = 0; ordinal < static_cast<int64_t>(entryCount);
       ++ordinal) {
    if (reader.i64() != ordinal || reader.i64() != ordinal ||
        reader.i64() != nextOffset)
      return std::nullopt;
    entryLengthOffsets.push_back(reader.position());
    int64_t length = reader.i64();
    if (length <= 0 || length > static_cast<int64_t>(limits.entryCodeBytes) ||
        nextOffset > static_cast<int64_t>(code.size()) - length)
      return std::nullopt;
    auto inputs = reader.integers();
    uint64_t firstInputOffset =
        reader.position() - inputs.size() * sizeof(uint64_t);
    auto outputs = reader.integers();
    if (llvm::any_of(inputs, [](int64_t slot) { return slot < 0; }) ||
        llvm::any_of(outputs, [](int64_t slot) { return slot < 0; }))
      return std::nullopt;
    if (reader.u64() != inputs.size())
      return std::nullopt;
    for (size_t index = 0; index < inputs.size(); ++index)
      if (reader.u8() != 0)
        return std::nullopt;
    if (reader.u64() != outputs.size())
      return std::nullopt;
    for (size_t index = 0; index < outputs.size(); ++index)
      if (reader.u8() != 1)
        return std::nullopt;
    auto dependencies = reader.integers();
    if (llvm::any_of(dependencies,
                     [ordinal](int64_t dependency) {
                       return dependency < 0 || dependency >= ordinal;
                     }) ||
        reader.u8() > 1)
      return std::nullopt;
    uint8_t hasReduction = reader.u8();
    if (hasReduction > 1 || (hasReduction && reader.u8() > 0))
      return std::nullopt;
    uint64_t entryDigestPayloadOffset = 0;
    llvm::StringRef codeDigest = reader.text(&entryDigestPayloadOffset);
    auto slice = code.slice(nextOffset, length);
    if (codeDigest != sha256(slice))
      return std::nullopt;
    nextOffset += length;
    uint64_t launchGridOffset = reader.position();
    std::array<uint64_t, 3> grid{reader.u32(), reader.u32(), reader.u32()};
    std::array<uint64_t, 3> block{reader.u16(), reader.u16(), reader.u16()};
    launchSharedOffsets.push_back(reader.position());
    uint32_t shared = reader.u32();
    if (llvm::is_contained(grid, 0) || llvm::is_contained(block, 0) ||
        block[0] * block[1] * block[2] > 1024 ||
        shared > limits.sharedBytes ||
        reader.u16() != inputs.size() + outputs.size())
      return std::nullopt;
    uint64_t positions = 1;
    for (uint64_t extent : llvm::concat<uint64_t>(grid, block)) {
      if (extent > limits.taskPositions / positions)
        return std::nullopt;
      positions *= extent;
    }
    if (positions > limits.taskPositions - taskPositions)
      return std::nullopt;
    taskPositions += positions;
    mlir::shuttle::GpuLaunch launch;
    launch.taskOrdinal = ordinal;
    launch.codeOffset = nextOffset - length;
    launch.codeLength = length;
    launch.codeDigest = codeDigest.str();
    launch.grid = grid;
    launch.block = block;
    launch.dynamicSharedMemoryBytes = shared;
    launch.inputSlots.assign(inputs.begin(), inputs.end());
    launch.outputSlots.assign(outputs.begin(), outputs.end());
    launch.dependencies.assign(dependencies.begin(), dependencies.end());
    launches.push_back(std::move(launch));
    entryDigestPayloadOffsets.push_back(entryDigestPayloadOffset);
    launchFirstInputOffsets.push_back(firstInputOffset);
    launchGridOffsets.push_back(launchGridOffset);
  }
  uint64_t invocationStart = reader.position();
  if (deviceRoot !=
      sha256(
          bytes.slice(deviceStart, deviceRootOffset - deviceStart),
          bytes.slice(deviceSuffixStart, invocationStart - deviceSuffixStart)))
    return std::nullopt;
  if (!reader.ok() || nextOffset != code.size() || reader.i64() != 3 ||
      !isDigest(reader.text()) || reader.text() != schedule)
    return std::nullopt;
  uint64_t invocationRootOffset = reader.position();
  uint64_t invocationRootPayloadOffset = 0;
  llvm::StringRef invocationRoot = reader.text(&invocationRootPayloadOffset);
  uint64_t invocationSuffixStart = reader.position();
  uint64_t invocationSlotCountOffset = reader.position();
  uint64_t slotCount = reader.u64();
  if (!isDigest(invocationRoot) || slotCount == 0 ||
      slotCount > limits.records)
    return std::nullopt;
  uint64_t temporaryBytes = 0;
  int64_t scalarSlots = 0;
  int64_t externalSlots = 0;
  llvm::SmallVector<int64_t> slotReuseGroups;
  llvm::SmallVector<uint64_t> slotReuseOffsets;
  llvm::SmallVector<int64_t> temporaryGroupOrder;
  llvm::SmallVector<uint64_t> temporaryGroupBytes;
  llvm::DenseMap<int64_t, size_t> temporaryGroupIndexes;
  llvm::SmallVector<uint8_t> slotStorage;
  llvm::SmallVector<uint8_t> slotBinding;
  llvm::SmallVector<llvm::SmallVector<int64_t>> slotDimensions;
  llvm::SmallVector<llvm::SmallVector<int64_t>> slotStrides;
  llvm::SmallVector<int64_t> slotRequiredBytes;
  llvm::SmallVector<uint64_t> slotRankOffsets;
  llvm::SmallVector<llvm::SmallVector<uint64_t>> slotDimensionOffsets;
  llvm::SmallVector<uint64_t> slotRequiredOffsets;
  llvm::SmallVector<int64_t> slotBindingIndex;
  llvm::SmallVector<uint64_t> slotBindingIndexOffsets;
  for (int64_t ordinal = 0; ordinal < static_cast<int64_t>(slotCount);
       ++ordinal) {
    if (reader.i64() != ordinal || reader.i64() != ordinal)
      return std::nullopt;
    uint8_t elementType = reader.u8();
    if (elementType > 1)
      return std::nullopt;
    slotRankOffsets.push_back(reader.position());
    uint64_t rank = reader.u64();
    if (rank > limits.rank)
      return std::nullopt;
    llvm::SmallVector<int64_t> dimensions;
    llvm::SmallVector<uint64_t> dimensionOffsets;
    int64_t expectedBytes = elementType == 0 ? 2 : 4;
    for (uint64_t axis = 0; axis < rank; ++axis) {
      dimensionOffsets.push_back(reader.position());
      int64_t extent = reader.i64();
      if (extent <= 0 ||
          expectedBytes > std::numeric_limits<int64_t>::max() / extent)
        return std::nullopt;
      dimensions.push_back(extent);
      expectedBytes *= extent;
    }
    slotDimensionOffsets.push_back(std::move(dimensionOffsets));
    slotRequiredOffsets.push_back(reader.position());
    int64_t required = reader.i64();
    auto strides = reader.integers();
    int64_t offset = reader.i64();
    int64_t alignment = reader.i64();
    uint8_t addressSpace = reader.u8();
    uint8_t access = reader.u8();
    if (required != expectedBytes ||
        required > static_cast<int64_t>(limits.slotBytes) ||
        strides.size() != rank || offset != 0 ||
        alignment != (elementType == 0 ? 2 : 4) || addressSpace != 1 ||
        access > 2)
      return std::nullopt;
    uint8_t storage = reader.u8();
    int64_t aliasGroup = reader.i64();
    uint64_t reuseOffset = reader.position();
    int64_t reuseGroup = reader.i64();
    if (storage > 1 || aliasGroup != ordinal || reuseGroup < 0)
      return std::nullopt;
    slotReuseGroups.push_back(reuseGroup);
    slotReuseOffsets.push_back(reuseOffset);
    uint8_t binding = reader.u8();
    uint8_t hasBindingIndex = reader.u8();
    if (binding > 2 || hasBindingIndex > 1)
      return std::nullopt;
    uint64_t bindingIndexOffset = reader.position();
    int64_t bindingIndex = hasBindingIndex ? reader.i64() : -1;
    slotStorage.push_back(storage);
    slotBinding.push_back(binding);
    slotBindingIndexOffsets.push_back(hasBindingIndex ? bindingIndexOffset : 0);
    slotDimensions.push_back(std::move(dimensions));
    slotStrides.push_back(std::move(strides));
    slotRequiredBytes.push_back(required);
    slotBindingIndex.push_back(bindingIndex);
    if (rank == 0)
      ++scalarSlots;
    if (storage == 1) {
      if (binding != 0 || hasBindingIndex || access != 2)
        return std::nullopt;
      auto inserted = temporaryGroupIndexes.try_emplace(
          reuseGroup, temporaryGroupOrder.size());
      if (inserted.second) {
        temporaryGroupOrder.push_back(reuseGroup);
        temporaryGroupBytes.push_back(required);
        temporaryBytes += required;
      } else if (static_cast<uint64_t>(required) >
                 temporaryGroupBytes[inserted.first->second]) {
        temporaryBytes +=
            required - temporaryGroupBytes[inserted.first->second];
        temporaryGroupBytes[inserted.first->second] = required;
      }
    } else {
      ++externalSlots;
      if (reuseGroup != ordinal || binding == 0 || !hasBindingIndex ||
          (binding == 1 && access != 0) || (binding == 2 && access != 1) ||
          elementType != 0 || rank == 0 || rank > 8)
        return std::nullopt;
    }
  }
  uint64_t bundleStart = reader.position();
  if (invocationRoot !=
      sha256(
          bytes.slice(invocationStart, invocationRootOffset - invocationStart),
          bytes.slice(invocationSuffixStart,
                      bundleStart - invocationSuffixStart)))
    return std::nullopt;
  if (temporaryBytes > limits.temporaryBytes || externalSlots == 0 ||
      reader.i64() != 2 || reader.text() != schedule)
    return std::nullopt;
  uint64_t bundleDeviceRootPayloadOffset = 0;
  if (reader.text(&bundleDeviceRootPayloadOffset) != deviceRoot)
    return std::nullopt;
  uint64_t bundleInvocationRootPayloadOffset = 0;
  if (reader.text(&bundleInvocationRootPayloadOffset) != invocationRoot ||
      reader.u8() != 1)
    return std::nullopt;
  uint64_t rootOffset = reader.position();
  uint64_t bundleRootPayloadOffset = 0;
  llvm::StringRef bundleRoot = reader.text(&bundleRootPayloadOffset);
  if (!reader.done() || bundleRoot != sha256(bytes.take_front(rootOffset)))
    return std::nullopt;
  llvm::SmallVector<int64_t> producers(slotCount, -1);
  llvm::SmallVector<int64_t> firstUses(slotCount, -1);
  llvm::SmallVector<int64_t> finalUses(slotCount, -1);
  for (const auto &launch : launches) {
    for (int64_t slot : launch.inputSlots) {
      if (slot >= static_cast<int64_t>(slotCount))
        return std::nullopt;
      if (firstUses[slot] == -1)
        firstUses[slot] = launch.taskOrdinal;
      finalUses[slot] = std::max(finalUses[slot], launch.taskOrdinal);
    }
    for (int64_t slot : launch.outputSlots) {
      if (slot >= static_cast<int64_t>(slotCount) || producers[slot] != -1)
        return std::nullopt;
      producers[slot] = launch.taskOrdinal;
      finalUses[slot] = std::max(finalUses[slot], launch.taskOrdinal);
    }
  }
  llvm::SmallDenseSet<int64_t> operandIndices;
  llvm::SmallDenseSet<int64_t> resultIndices;
  for (int64_t ordinal = 0; ordinal < static_cast<int64_t>(slotCount);
       ++ordinal) {
    if (slotStorage[ordinal] == 1) {
      if (producers[ordinal] < 0 || firstUses[ordinal] <= producers[ordinal] ||
          finalUses[ordinal] <= producers[ordinal])
        return std::nullopt;
      continue;
    }
    if (slotBinding[ordinal] == 1) {
      if (slotBindingIndex[ordinal] < 0 ||
          !operandIndices.insert(slotBindingIndex[ordinal]).second ||
          producers[ordinal] != -1 || finalUses[ordinal] < 0)
        return std::nullopt;
    } else if (slotBinding[ordinal] == 2) {
      if (slotBindingIndex[ordinal] < 0 ||
          !resultIndices.insert(slotBindingIndex[ordinal]).second ||
          producers[ordinal] < 0 ||
          (firstUses[ordinal] != -1 &&
           firstUses[ordinal] <= producers[ordinal]))
        return std::nullopt;
    } else {
      return std::nullopt;
    }
  }
  auto isContiguous = [](const llvm::SmallDenseSet<int64_t> &indices) {
    for (int64_t index = 0; index < static_cast<int64_t>(indices.size());
         ++index)
      if (!indices.contains(index))
        return false;
    return true;
  };
  if (operandIndices.empty() || resultIndices.empty() ||
      !isContiguous(operandIndices) || !isContiguous(resultIndices))
    return std::nullopt;
  for (int64_t left = 0; left < static_cast<int64_t>(slotCount); ++left) {
    if (slotStorage[left] != 1)
      continue;
    for (int64_t right = left + 1; right < static_cast<int64_t>(slotCount);
         ++right) {
      if (slotStorage[right] != 1 ||
          slotReuseGroups[left] != slotReuseGroups[right])
        continue;
      if (producers[left] <= finalUses[right] &&
          producers[right] <= finalUses[left])
        return std::nullopt;
    }
  }
  return Decoded{llvm::SmallVector<uint8_t>(code),
                 std::move(launches),
                 policy,
                 devicePolicyOffset,
                 std::move(slotReuseGroups),
                 std::move(slotReuseOffsets),
                 std::move(slotStorage),
                 std::move(slotBinding),
                 std::move(slotBindingIndex),
                 std::move(slotBindingIndexOffsets),
                 std::move(slotDimensions),
                 std::move(slotStrides),
                 std::move(slotRequiredBytes),
                 std::move(producers),
                 std::move(finalUses),
                 std::move(launchFirstInputOffsets),
                 std::move(temporaryGroupOrder),
                 std::move(temporaryGroupBytes),
                 static_cast<int64_t>(slotCount),
                 temporaryBytes,
                 scalarSlots,
                 externalSlots,
                 deviceRootOffset,
                 deviceRootPayloadOffset,
                 deviceSuffixStart,
                 invocationStart,
                 invocationRootOffset,
                 invocationRootPayloadOffset,
                 invocationSuffixStart,
                 bundleStart,
                 bundleDeviceRootPayloadOffset,
                 bundleInvocationRootPayloadOffset,
                 rootOffset,
                 bundleRootPayloadOffset,
                 std::move(launchGridOffsets),
                 codeLengthOffset,
                 codePayloadOffset,
                 codeDigestPayloadOffset,
                 deviceEntryCountOffset,
                 std::move(entryLengthOffsets),
                 std::move(entryDigestPayloadOffsets),
                 std::move(launchSharedOffsets),
                 invocationSlotCountOffset,
                 std::move(slotRankOffsets),
                 std::move(slotDimensionOffsets),
                 std::move(slotRequiredOffsets)};
}

void writeU32(llvm::MutableArrayRef<uint8_t> bytes, uint64_t offset,
              uint32_t value) {
  for (unsigned byte = 0; byte < sizeof(value); ++byte)
    bytes[offset + byte] = static_cast<uint8_t>(value >> (byte * 8));
}

void writeU64(llvm::MutableArrayRef<uint8_t> bytes, uint64_t offset,
              uint64_t value) {
  for (unsigned byte = 0; byte < sizeof(value); ++byte)
    bytes[offset + byte] = static_cast<uint8_t>(value >> (byte * 8));
}

void writeDigest(llvm::MutableArrayRef<uint8_t> bytes, uint64_t offset,
                 llvm::StringRef digest) {
  assert(digest.size() == 64);
  llvm::copy(digest, bytes.begin() + offset);
}

void rehashDeviceAndBundle(llvm::MutableArrayRef<uint8_t> bytes,
                           const Decoded &decoded) {
  std::string deviceRoot =
      sha256(bytes.slice(/*deviceStart=*/12, decoded.deviceRootOffset - 12),
             bytes.slice(decoded.deviceSuffixStart,
                         decoded.invocationStart - decoded.deviceSuffixStart));
  writeDigest(bytes, decoded.deviceRootPayloadOffset, deviceRoot);
  writeDigest(bytes, decoded.bundleDeviceRootPayloadOffset, deviceRoot);
  writeDigest(bytes, decoded.bundleRootPayloadOffset,
              sha256(bytes.take_front(decoded.bundleRootOffset)));
}

llvm::SmallVector<uint8_t>
replaceEntryCodeAndRehash(llvm::ArrayRef<uint8_t> bytes,
                          const Decoded &decoded, int64_t entry,
                          llvm::StringRef from, llvm::StringRef to) {
  if (entry < 0 || entry >= static_cast<int64_t>(decoded.launches.size()) ||
      from.size() != to.size())
    return {};
  const auto &launch = decoded.launches[entry];
  llvm::StringRef ptx(
      reinterpret_cast<const char *>(decoded.code.data() + launch.codeOffset),
      launch.codeLength);
  size_t relative = ptx.find(from);
  if (relative == llvm::StringRef::npos ||
      ptx.find(from, relative + 1) != llvm::StringRef::npos)
    return {};
  llvm::SmallVector<uint8_t> result(bytes);
  llvm::copy(to, result.begin() + decoded.codePayloadOffset +
                     launch.codeOffset + relative);
  auto code = llvm::ArrayRef<uint8_t>(result).slice(decoded.codePayloadOffset,
                                                    decoded.code.size());
  writeDigest(result, decoded.codeDigestPayloadOffset, sha256(code));
  writeDigest(result, decoded.entryDigestPayloadOffsets[entry],
              sha256(code.slice(launch.codeOffset, launch.codeLength)));
  rehashDeviceAndBundle(result, decoded);
  return result;
}

std::optional<std::string>
projectedF32LoadedRegister(llvm::StringRef ptx, int64_t inputPointer) {
  std::string suffix =
      (", [%rd" + llvm::Twine(inputPointer) + "];").str();
  std::optional<std::string> loaded;
  llvm::SmallVector<llvm::StringRef> lines;
  ptx.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (!line.starts_with("ld.global.f32 ") || !line.ends_with(suffix))
      continue;
    if (loaded)
      return std::nullopt;
    llvm::StringRef destination =
        line.drop_front(llvm::StringRef("ld.global.f32 ").size())
            .drop_back(suffix.size());
    if (!destination.starts_with("%f") || destination.contains(','))
      return std::nullopt;
    loaded = destination.str();
  }
  return loaded;
}

void rehashInvocationAndBundle(llvm::MutableArrayRef<uint8_t> bytes,
                               const Decoded &decoded) {
  std::string invocationRoot =
      sha256(bytes.slice(decoded.invocationStart, decoded.invocationRootOffset -
                                                      decoded.invocationStart),
             bytes.slice(decoded.invocationSuffixStart,
                         decoded.bundleStart - decoded.invocationSuffixStart));
  writeDigest(bytes, decoded.invocationRootPayloadOffset, invocationRoot);
  writeDigest(bytes, decoded.bundleInvocationRootPayloadOffset, invocationRoot);
  writeDigest(bytes, decoded.bundleRootPayloadOffset,
              sha256(bytes.take_front(decoded.bundleRootOffset)));
}

llvm::SmallVector<uint8_t>
mutateBindingIndexAndRehash(llvm::ArrayRef<uint8_t> bytes,
                            const Decoded &decoded, int64_t slot,
                            int64_t bindingIndex) {
  if (slot < 0 || slot >= static_cast<int64_t>(decoded.slots) ||
      decoded.slotBinding[slot] == 0 ||
      decoded.slotBindingIndexOffsets[slot] == 0)
    return {};
  llvm::SmallVector<uint8_t> result(bytes);
  writeU64(result, decoded.slotBindingIndexOffsets[slot],
           static_cast<uint64_t>(bindingIndex));
  rehashInvocationAndBundle(result, decoded);
  return result;
}

llvm::SmallVector<uint8_t> oversizedLaunchGrid(llvm::ArrayRef<uint8_t> bytes,
                                               const Decoded &decoded) {
  llvm::SmallVector<uint8_t> result(bytes);
  uint64_t gridOffset = decoded.launchGridOffsets.front();
  writeU32(result, gridOffset, uint32_t{1} << 31);
  writeU32(result, gridOffset + 4, uint32_t{1} << 31);
  writeU32(result, gridOffset + 8, 4);
  rehashDeviceAndBundle(result, decoded);
  return result;
}

llvm::SmallVector<uint8_t>
overlappingTemporaryReuse(llvm::ArrayRef<uint8_t> bytes,
                          const Decoded &decoded) {
  for (size_t left = 0; left < decoded.slotStorage.size(); ++left) {
    if (decoded.slotStorage[left] != 1)
      continue;
    for (size_t right = left + 1; right < decoded.slotStorage.size(); ++right) {
      if (decoded.slotStorage[right] != 1 ||
          decoded.slotReuseGroups[left] == decoded.slotReuseGroups[right])
        continue;
      if (decoded.slotProducers[left] > decoded.slotFinalUses[right] ||
          decoded.slotProducers[right] > decoded.slotFinalUses[left])
        continue;
      llvm::SmallVector<uint8_t> result(bytes);
      writeU64(result, decoded.slotReuseOffsets[right],
               decoded.slotReuseGroups[left]);
      rehashInvocationAndBundle(result, decoded);
      return result;
    }
  }
  return {};
}

llvm::SmallVector<uint8_t> futureProducedInput(llvm::ArrayRef<uint8_t> bytes,
                                               const Decoded &decoded) {
  for (const auto &launch : decoded.launches) {
    if (launch.inputSlots.empty())
      continue;
    for (int64_t slot = 0;
         slot < static_cast<int64_t>(decoded.slotStorage.size()); ++slot) {
      if (decoded.slotStorage[slot] != 1 ||
          decoded.slotProducers[slot] <= launch.taskOrdinal ||
          decoded.slotFinalUses[slot] <= decoded.slotProducers[slot])
        continue;
      llvm::SmallVector<uint8_t> result(bytes);
      writeU64(result, decoded.launchFirstInputOffsets[launch.taskOrdinal],
               slot);
      rehashDeviceAndBundle(result, decoded);
      return result;
    }
  }
  return {};
}

llvm::SmallVector<uint8_t>
futureProducedResultInput(llvm::ArrayRef<uint8_t> bytes,
                          const Decoded &decoded) {
  for (const auto &launch : decoded.launches) {
    if (launch.inputSlots.empty())
      continue;
    for (int64_t slot = 0;
         slot < static_cast<int64_t>(decoded.slotStorage.size()); ++slot) {
      if (decoded.slotBinding[slot] != 2 ||
          decoded.slotProducers[slot] <= launch.taskOrdinal)
        continue;
      llvm::SmallVector<uint8_t> result(bytes);
      writeU64(result, decoded.launchFirstInputOffsets[launch.taskOrdinal],
               slot);
      rehashDeviceAndBundle(result, decoded);
      return result;
    }
  }
  return {};
}

struct BuiltBundle {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

std::optional<std::string> fixtureSource(llvm::StringRef filename) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  if (!runfiles)
    return std::nullopt;
  std::ifstream input(
      runfiles->Rlocation(("shuttle_mlir/test/Inputs/" + filename).str()));
  std::ostringstream source;
  source << input.rdbuf();
  if (!input)
    return std::nullopt;
  return source.str();
}

std::string contractMapModule(int64_t rows, int64_t reduction, int64_t columns,
                              llvm::StringRef moduleSymbol,
                              llvm::StringRef precision = "DEFAULT",
                              bool convertToBf16 = true) {
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module @" << moduleSymbol << " {\n"
         << "  func.func public @main(%lhs: tensor<" << rows << "x" << reduction
         << "xbf16>, %rhs: tensor<" << reduction << "x" << columns
         << "xbf16>) -> tensor<" << rows << "x" << columns
         << (convertToBf16 ? "xbf16" : "xf32") << "> {\n"
         << "    %contract = stablehlo.dot_general %lhs, %rhs, "
            "contracting_dims = [1] x [0], precision = ["
         << precision << ", " << precision << "] : (tensor<" << rows << "x"
         << reduction << "xbf16>, tensor<" << reduction << "x" << columns
         << "xbf16>) -> tensor<" << rows << "x" << columns << "xf32>\n";
  if (convertToBf16) {
    stream << "    %result = stablehlo.convert %contract : (tensor<" << rows
           << "x" << columns << "xf32>) -> tensor<" << rows << "x" << columns
           << "xbf16>\n"
           << "    return %result : tensor<" << rows << "x" << columns
           << "xbf16>\n";
  } else {
    stream << "    return %contract : tensor<" << rows << "x" << columns
           << "xf32>\n";
  }
  stream << "  }\n}\n";
  stream.flush();
  return source;
}

constexpr llvm::StringLiteral kContractMapWithLiveF32Result = R"mlir(
module @live_f32_and_bf16 {
  func.func @main(%lhs: tensor<3x5xbf16>, %rhs: tensor<5x2xbf16>) -> (tensor<3x2xf32>, tensor<3x2xbf16>) {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x5xbf16>, tensor<5x2xbf16>) -> tensor<3x2xf32>
    %result = stablehlo.convert %contract : (tensor<3x2xf32>) -> tensor<3x2xbf16>
    return %contract, %result : tensor<3x2xf32>, tensor<3x2xbf16>
  }
}
)mlir";

std::unique_ptr<BuiltBundle> buildFixture(
    llvm::StringRef source,
    mlir::shuttle::NumericalPolicy policy =
        mlir::shuttle::NumericalPolicy::SourceOrdered,
    llvm::StringRef canonicalOptions = {},
    llvm::StringRef canonicalTuning =
        R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json",
    mlir::shuttle::GpuTarget gpuTarget = mlir::shuttle::GpuTarget::Sm90) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::shuttle::ShuttleDialect>();
  auto result = std::make_unique<BuiltBundle>();
  result->context = std::make_unique<mlir::MLIRContext>(registry);
  result->module =
      mlir::parseSourceString<mlir::ModuleOp>(source, result->context.get());
  if (!result->module)
    return {};
  mlir::PassManager manager(result->context.get());
  manager.addPass(mlir::shuttle::createAnnotateSourcePass());
  if (!canonicalOptions.empty()) {
    manager.addPass(mlir::shuttle::createFormStructuralRegionsPass(
        policy, canonicalOptions.str(), canonicalTuning.str()));
  } else if (policy == mlir::shuttle::NumericalPolicy::Fast) {
    manager.addPass(mlir::shuttle::createFormStructuralRegionsPass(
        policy,
        R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":18,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
        R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json"));
  } else {
    manager.addPass(mlir::shuttle::createFormStructuralRegionsPass());
  }
  manager.addPass(mlir::shuttle::createConvertStablehloToAlgebraPass());
  manager.addPass(mlir::shuttle::createPlanRowFoldMaterializationPass());
  manager.addPass(mlir::shuttle::createPlanSimt32RowFoldSchedulePass());
  manager.addPass(mlir::shuttle::createBuildGpuExecutableBundlePass(gpuTarget));
  manager.addPass(mlir::shuttle::createVerifyGpuExecutableBundlePass());
  if (mlir::failed(manager.run(*result->module)))
    return {};
  return result;
}

TEST(GpuTransportV2Test, Sm100CodegenAndTransportAreTargetBound) {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir");
  ASSERT_TRUE(source);
  auto sm90 = buildFixture(*source);
  auto sm100 =
      buildFixture(*source, mlir::shuttle::NumericalPolicy::SourceOrdered, {},
                   {}, mlir::shuttle::GpuTarget::Sm100);
  ASSERT_TRUE(sm90);
  ASSERT_TRUE(sm100);
  auto sm90Device =
      *sm90->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
  auto sm100Device =
      *sm100->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
  EXPECT_EQ(sm90Device.getSchemaVersion(), 3);
  EXPECT_EQ(sm100Device.getSchemaVersion(), 4);
  EXPECT_EQ(sm90Device.getCodeFormat(),
            mlir::shuttle::ExecutableCodeFormat::CudaPtxSm90V1);
  EXPECT_EQ(sm100Device.getCodeFormat(),
            mlir::shuttle::ExecutableCodeFormat::CudaPtxSm100V1);
  llvm::StringRef sm90Ptx(
      reinterpret_cast<const char *>(sm90Device.getCode().data()),
      sm90Device.getCode().size());
  llvm::StringRef sm100Ptx(
      reinterpret_cast<const char *>(sm100Device.getCode().data()),
      sm100Device.getCode().size());
  EXPECT_TRUE(sm90Ptx.starts_with(".version 8.0\n.target sm_90\n"));
  EXPECT_TRUE(sm100Ptx.starts_with(".version 8.7\n.target sm_100\n"));
  llvm::SmallVector<mlir::shuttle::DeviceEntryOp> sm90Entries(
      sm90Device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>());
  llvm::SmallVector<mlir::shuttle::DeviceEntryOp> sm100Entries(
      sm100Device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>());
  ASSERT_EQ(sm90Entries.size(), sm100Entries.size());
  ASSERT_GT(sm90Entries.size(), 1);
  constexpr llvm::StringLiteral sm90Header = ".version 8.0\n.target sm_90\n";
  constexpr llvm::StringLiteral sm100Header = ".version 8.7\n.target sm_100\n";
  for (auto [sm90Entry, sm100Entry] :
       llvm::zip_equal(sm90Entries, sm100Entries)) {
    llvm::StringRef sm90EntryPtx =
        sm90Ptx.substr(sm90Entry.getCodeOffset(), sm90Entry.getCodeLength());
    llvm::StringRef sm100EntryPtx =
        sm100Ptx.substr(sm100Entry.getCodeOffset(), sm100Entry.getCodeLength());
    ASSERT_TRUE(sm90EntryPtx.starts_with(sm90Header));
    ASSERT_TRUE(sm100EntryPtx.starts_with(sm100Header));
    EXPECT_EQ(sm90EntryPtx.drop_front(sm90Header.size()),
              sm100EntryPtx.drop_front(sm100Header.size()));
  }
  EXPECT_NE(sm90Device.getFingerprint(), sm100Device.getFingerprint());

  auto sm90Transport =
      mlir::shuttle::serializeGpuExecutableBundle(*sm90->module);
  auto sm100Transport =
      mlir::shuttle::serializeGpuExecutableBundle(*sm100->module);
  ASSERT_TRUE(mlir::succeeded(sm90Transport));
  ASSERT_TRUE(mlir::succeeded(sm100Transport));
  EXPECT_NE(mlir::shuttle::gpuExecutableBundleDigest(*sm90Transport),
            mlir::shuttle::gpuExecutableBundleDigest(*sm100Transport));
  auto loaded = mlir::shuttle::GpuExecutable::Load(*sm100Transport);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_EQ((*loaded)->deviceSchemaVersion(), 4);
  EXPECT_EQ((*loaded)->codeFormat(),
            mlir::shuttle::ExecutableCodeFormat::CudaPtxSm100V1);

  auto decoded = decodeInlineSchema2(*sm100Transport);
  ASSERT_TRUE(decoded);
  auto wrongHeader = replaceEntryCodeAndRehash(*sm100Transport, *decoded, 0,
                                               ".version 8.7", ".version 8.0");
  ASSERT_FALSE(wrongHeader.empty());
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(wrongHeader).ok());
  llvm::SmallVector<uint8_t> crossedPair(*sm100Transport);
  writeU64(crossedPair, /*device schema offset=*/12, 3);
  rehashDeviceAndBundle(crossedPair, *decoded);
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(crossedPair).ok());
}

std::unique_ptr<BuiltBundle>
buildGpuContractFixture(llvm::StringRef source,
                        mlir::shuttle::NumericalPolicy policy) {
  constexpr llvm::StringLiteral sourceOrdered =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":18,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral fast =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":18,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  return buildFixture(
      source, policy,
      policy == mlir::shuttle::NumericalPolicy::Fast ? fast : sourceOrdered);
}

std::optional<llvm::SmallVector<llvm::SmallVector<uint8_t>>>
serializeAllGpuBundles(mlir::ModuleOp module) {
  llvm::SmallVector<llvm::SmallVector<uint8_t>> serialized;
  for (mlir::shuttle::ExecutableBundleOp bundle :
       module.getOps<mlir::shuttle::ExecutableBundleOp>()) {
    mlir::shuttle::DeviceModuleOp device;
    for (mlir::shuttle::DeviceModuleOp candidate :
         module.getOps<mlir::shuttle::DeviceModuleOp>()) {
      if (candidate.getFingerprint() != bundle.getDeviceModuleFingerprint())
        continue;
      if (device)
        return std::nullopt;
      device = candidate;
    }
    mlir::shuttle::InvocationAbiOp abi;
    for (mlir::shuttle::InvocationAbiOp candidate :
         module.getOps<mlir::shuttle::InvocationAbiOp>()) {
      if (candidate.getFingerprint() != bundle.getInvocationAbiFingerprint())
        continue;
      if (abi)
        return std::nullopt;
      abi = candidate;
    }
    if (!device || !abi)
      return std::nullopt;
    mlir::OwningOpRef<mlir::ModuleOp> tuple =
        mlir::ModuleOp::create(module.getLoc());
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(tuple->getBody());
    builder.insert(device->clone());
    builder.insert(abi->clone());
    builder.insert(bundle->clone());
    auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*tuple);
    if (mlir::failed(bytes))
      return std::nullopt;
    serialized.push_back(std::move(*bytes));
  }
  return serialized;
}

std::unique_ptr<BuiltBundle> buildFixture() {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir");
  if (!source)
    return {};
  return buildFixture(*source);
}

std::unique_ptr<BuiltBundle> buildAlternateFixture() {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir");
  if (!source)
    return {};
  constexpr llvm::StringLiteral original = "9.99999974E-6";
  constexpr llvm::StringLiteral replacement = "2.49999994E-6";
  size_t position = source->find(original.str());
  if (position == std::string::npos)
    return {};
  source->replace(position, original.size(), replacement.data(),
                  replacement.size());
  return buildFixture(*source);
}

std::string mapOnlyModule(int64_t inputs, bool permutedResult) {
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << (permutedResult ? "#indexing = affine_map<(d0, d1) -> (d1, d0)>\n"
                            : "#indexing = affine_map<(d0, d1) -> (d0, d1)>\n")
         << "module {\n  func.func @map_only(";
  for (int64_t index = 0; index < inputs; ++index) {
    if (index)
      stream << ", ";
    stream << "%arg" << index << ": tensor<2x3xf32>";
  }
  stream << ") -> tensor<2xf32> {\n"
            "    %result = \"shuttle.region\"(";
  for (int64_t index = 0; index < inputs; ++index) {
    if (index)
      stream << ", ";
    stream << "%arg" << index;
  }
  stream << ") ({\n    ^bb0(";
  for (int64_t index = 0; index < inputs; ++index) {
    if (index)
      stream << ", ";
    stream << "%region" << index << ": tensor<2x3xf32>";
  }
  stream << "):\n"
            "      %zero = \"shuttle.map\"() ({\n"
            "      ^bb0:\n"
            "        %constant = arith.constant 0.0 : f32\n"
            "        \"shuttle.yield\"(%constant) : (f32) -> ()\n"
            "      }) {indexing_maps = [affine_map<() -> ()>], source = "
            "#shuttle.source_ref<0, 0, 0, 0>} : () -> tensor<f32>\n"
            "      %mapped = \"shuttle.map\"(";
  for (int64_t index = 0; index < inputs; ++index) {
    if (index)
      stream << ", ";
    stream << "%region" << index;
  }
  stream << ") ({\n      ^bb0(";
  for (int64_t index = 0; index < inputs; ++index) {
    if (index)
      stream << ", ";
    stream << "%value" << index << ": f32";
  }
  stream << "):\n        \"shuttle.yield\"(%value0) : (f32) -> ()\n"
            "      }) {indexing_maps = [";
  for (int64_t index = 0; index <= inputs; ++index) {
    if (index)
      stream << ", ";
    stream << "#indexing";
  }
  stream << "], source = #shuttle.source_ref<0, 0, 1, 0>} : (";
  for (int64_t index = 0; index < inputs; ++index) {
    if (index)
      stream << ", ";
    stream << "tensor<2x3xf32>";
  }
  stream << ") -> tensor<2x3xf32>\n"
            "      %folded = \"shuttle.fold\"(%mapped, %zero) ({\n"
            "      ^bb0(%left: f32, %right: f32):\n"
            "        %sum = arith.addf %left, %right : f32\n"
            "        \"shuttle.yield\"(%sum) : (f32) -> ()\n"
            "      }) {accumulator_types = [f32], operandSegmentSizes = "
            "array<i32: 1, 1>, order_free = true, reduction_dimensions = "
            "array<i64: 1>, source = #shuttle.source_ref<0, 0, 2, 0>} : "
            "(tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>\n"
            "      \"shuttle.yield\"(%folded) : (tensor<2xf32>) -> ()\n"
            "    }) {policy = #shuttle.policy<source_ordered>, "
            "source_refs = [#shuttle.source_ref<0, 0, 0, 0>, "
            "#shuttle.source_ref<0, 0, 1, 0>, "
            "#shuttle.source_ref<0, 0, 2, 0>]} : (";
  for (int64_t index = 0; index < inputs; ++index) {
    if (index)
      stream << ", ";
    stream << "tensor<2x3xf32>";
  }
  stream << ") -> tensor<2xf32>\n"
            "    return %result : tensor<2xf32>\n  }\n}\n";
  stream.flush();
  return source;
}

std::string roundedScalarMapModule() {
  std::string source = mapOnlyModule(1, false);
  constexpr llvm::StringLiteral yield =
      "        \"shuttle.yield\"(%value0) : (f32) -> ()\n";
  constexpr llvm::StringLiteral rounded =
      "        %rounded = \"shuttle.scalar_convert\"(%value0) "
      "{semantics = #shuttle.scalar_convert_semantics<round_nearest_even>} "
      ": (f32) -> bf16\n"
      "        %expanded = \"shuttle.scalar_convert\"(%rounded) "
      "{semantics = #shuttle.scalar_convert_semantics<exact>} "
      ": (bf16) -> f32\n"
      "        \"shuttle.yield\"(%expanded) : (f32) -> ()\n";
  size_t position = source.find(yield.str());
  if (position == std::string::npos)
    return {};
  source.replace(position, yield.size(), rounded.str());
  return source;
}

std::string bf16ArithmeticMapModule() {
  std::string source = roundedScalarMapModule();
  constexpr llvm::StringLiteral expanded =
      "        %expanded = \"shuttle.scalar_convert\"(%rounded) "
      "{semantics = #shuttle.scalar_convert_semantics<exact>} "
      ": (bf16) -> f32\n";
  constexpr llvm::StringLiteral arithmetic =
      "        %summed = arith.addf %rounded, %rounded : bf16\n"
      "        %expanded = \"shuttle.scalar_convert\"(%summed) "
      "{semantics = #shuttle.scalar_convert_semantics<exact>} "
      ": (bf16) -> f32\n";
  size_t position = source.find(expanded.str());
  if (position == std::string::npos)
    return {};
  source.replace(position, expanded.size(), arithmetic.str());
  return source;
}

std::unique_ptr<BuiltBundle> buildAlgebraFixture(llvm::StringRef source) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::shuttle::ShuttleDialect>();
  auto result = std::make_unique<BuiltBundle>();
  result->context = std::make_unique<mlir::MLIRContext>(registry);
  result->module =
      mlir::parseSourceString<mlir::ModuleOp>(source, result->context.get());
  if (!result->module)
    return {};
  mlir::Builder builder(result->context.get());
  result->module->getOperation()->setAttr(
      "shuttle.coverage_manifest",
      builder.getDictionaryAttr(builder.getNamedAttr(
          "canonical_tuning",
          builder.getStringAttr(
              R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json"))));
  mlir::PassManager manager(result->context.get());
  manager.addPass(mlir::shuttle::createPlanRowFoldMaterializationPass());
  manager.addPass(mlir::shuttle::createPlanSimt32RowFoldSchedulePass());
  manager.addPass(mlir::shuttle::createBuildGpuExecutableBundlePass());
  manager.addPass(mlir::shuttle::createVerifyGpuExecutableBundlePass());
  if (mlir::failed(manager.run(*result->module)))
    return {};
  return result;
}

TEST(GpuTransportV2Test, IndependentInlineDecoderMatchesPublicDescriptor) {
  auto built = buildFixture();
  ASSERT_TRUE(built);
  auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  auto independent = decodeInlineSchema2(*bytes);
  ASSERT_TRUE(independent);
  auto publicDescriptor = mlir::shuttle::GpuExecutable::Load(*bytes);
  ASSERT_TRUE(publicDescriptor.ok()) << publicDescriptor.status();
  ASSERT_EQ(independent->launches.size(), 19);
  ASSERT_EQ((*publicDescriptor)->launches().size(), 19);
  EXPECT_EQ(independent->slots, 21);
  EXPECT_EQ(independent->temporaryGroupOrder.size(), 4);
  EXPECT_EQ(independent->temporaryBytes, 100671488);
  EXPECT_EQ(independent->scalarSlots, 3);
  EXPECT_EQ(independent->externalSlots, 3);
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(independent->code),
            (*publicDescriptor)->codeBytes());
  for (auto [decoded, loaded] : llvm::zip_equal(
           independent->launches, (*publicDescriptor)->launches())) {
    EXPECT_EQ(decoded.taskOrdinal, loaded.taskOrdinal);
    EXPECT_EQ(decoded.codeOffset, loaded.codeOffset);
    EXPECT_EQ(decoded.codeLength, loaded.codeLength);
    EXPECT_EQ(decoded.codeDigest, loaded.codeDigest);
    EXPECT_EQ(decoded.grid, loaded.grid);
    EXPECT_EQ(decoded.block, loaded.block);
    EXPECT_EQ(decoded.dynamicSharedMemoryBytes,
              loaded.dynamicSharedMemoryBytes);
    EXPECT_EQ(decoded.inputSlots, loaded.inputSlots);
    EXPECT_EQ(decoded.outputSlots, loaded.outputSlots);
    EXPECT_EQ(decoded.dependencies, loaded.dependencies);
  }
  llvm::SmallVector<uint8_t> corruptRoot(*bytes);
  corruptRoot.back() ^= 1;
  EXPECT_FALSE(decodeInlineSchema2(corruptRoot));
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(corruptRoot).ok());

  auto oversized = oversizedLaunchGrid(*bytes, *independent);
  EXPECT_FALSE(decodeInlineSchema2(oversized));
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(oversized).ok());

  auto overlappingReuse = overlappingTemporaryReuse(*bytes, *independent);
  ASSERT_FALSE(overlappingReuse.empty());
  EXPECT_FALSE(decodeInlineSchema2(overlappingReuse));
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(overlappingReuse).ok());

  auto futureInput = futureProducedInput(*bytes, *independent);
  ASSERT_FALSE(futureInput.empty());
  EXPECT_FALSE(decodeInlineSchema2(futureInput));
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(futureInput).ok());
  auto futureResultInput = futureProducedResultInput(*bytes, *independent);
  ASSERT_FALSE(futureResultInput.empty());
  EXPECT_FALSE(decodeInlineSchema2(futureResultInput));
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(futureResultInput).ok());
}

TEST(GpuTransportV2Test, GenericLayerNormGraphsHaveClosedDynamicDescriptors) {
  struct Case {
    const char *filename;
    int64_t launches;
    int64_t slots;
    int64_t externalSlots;
    int64_t temporaryGroups;
    uint64_t temporaryBytes;
  };
  constexpr Case cases[] = {
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir", 31, 34, 4, 4,
       100671488},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-backward.mlir", 73, 76, 6,
       9, 167804932},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir", 80, 84, 8,
       9, 167804932},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir", 31, 34, 4, 4,
       1120},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-backward.mlir", 73, 76, 6,
       9, 1932},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-composed.mlir", 80, 84, 8,
       9, 1932},
  };
  for (const Case &test : cases) {
    SCOPED_TRACE(test.filename);
    auto source = fixtureSource(test.filename);
    ASSERT_TRUE(source);
    auto built = buildFixture(*source);
    ASSERT_TRUE(built);
    auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
    ASSERT_TRUE(mlir::succeeded(bytes));
    auto independent = decodeInlineSchema2(*bytes);
    ASSERT_TRUE(independent);
    auto publicDescriptor = mlir::shuttle::GpuExecutable::Load(*bytes);
    ASSERT_TRUE(publicDescriptor.ok()) << publicDescriptor.status();
    EXPECT_EQ(independent->launches.size(), test.launches);
    EXPECT_EQ(independent->slots, test.slots);
    EXPECT_EQ(independent->externalSlots, test.externalSlots);
    EXPECT_EQ(independent->temporaryGroupOrder.size(), test.temporaryGroups);
    EXPECT_EQ(independent->temporaryBytes, test.temporaryBytes);
    ASSERT_EQ((*publicDescriptor)->slots().size(), independent->slots);
    EXPECT_EQ((*publicDescriptor)->externalBindings().size(),
              independent->externalSlots);
    llvm::StringRef ptx(
        reinterpret_cast<const char *>((*publicDescriptor)->codeBytes().data()),
        (*publicDescriptor)->codeBytes().size());
    EXPECT_TRUE(ptx.contains("sqrt.rn.f32"));
    EXPECT_TRUE(ptx.contains(", 0f3F800000, %f"));
    EXPECT_FALSE(ptx.contains("rcp.rn.f32"));
    for (auto [slot, reuse] : llvm::zip_equal((*publicDescriptor)->slots(),
                                              independent->slotReuseGroups))
      EXPECT_EQ(slot.reuseGroup, reuse);
  }
}

TEST(GpuTransportV2Test, FastPolicyChangesClosureWithoutSelectingCode) {
  auto source =
      fixtureSource("jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir");
  ASSERT_TRUE(source);
  auto sourceOrdered = buildFixture(*source);
  auto fast = buildFixture(*source, mlir::shuttle::NumericalPolicy::Fast);
  ASSERT_TRUE(sourceOrdered);
  ASSERT_TRUE(fast);
  auto sourceOrderedBytes =
      mlir::shuttle::serializeGpuExecutableBundle(*sourceOrdered->module);
  auto fastBytes = mlir::shuttle::serializeGpuExecutableBundle(*fast->module);
  ASSERT_TRUE(mlir::succeeded(sourceOrderedBytes));
  ASSERT_TRUE(mlir::succeeded(fastBytes));
  auto sourceOrderedDecoded = decodeInlineSchema2(*sourceOrderedBytes);
  auto fastDecoded = decodeInlineSchema2(*fastBytes);
  ASSERT_TRUE(sourceOrderedDecoded);
  ASSERT_TRUE(fastDecoded);
  EXPECT_EQ(sourceOrderedDecoded->policy, 0);
  EXPECT_EQ(fastDecoded->policy, 1);
  EXPECT_EQ(sourceOrderedDecoded->code, fastDecoded->code);
  EXPECT_NE(*sourceOrderedBytes, *fastBytes);
  EXPECT_NE(mlir::shuttle::gpuExecutableBundleDigest(*sourceOrderedBytes),
            mlir::shuttle::gpuExecutableBundleDigest(*fastBytes));

  auto unsealedMutation = *sourceOrderedBytes;
  unsealedMutation[sourceOrderedDecoded->devicePolicyOffset] = 1;
  EXPECT_FALSE(decodeInlineSchema2(unsealedMutation));
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(unsealedMutation).ok());
}

TEST(GpuTransportV2Test, RejectsPermutedResultMapBeforeGpuCodeEmission) {
  ASSERT_TRUE(buildAlgebraFixture(mapOnlyModule(1, false)));
  EXPECT_FALSE(buildAlgebraFixture(mapOnlyModule(1, true)));
}

TEST(GpuTransportV2Test, WideMapDeclaresEveryReferencedRegister) {
  auto built = buildAlgebraFixture(mapOnlyModule(58, false));
  ASSERT_TRUE(built);
  auto devices =
      llvm::to_vector(built->module->getOps<mlir::shuttle::DeviceModuleOp>());
  ASSERT_EQ(devices.size(), 1);
  llvm::StringRef ptx(
      reinterpret_cast<const char *>(devices[0].getCode().data()),
      devices[0].getCode().size());
  EXPECT_TRUE(ptx.contains(".reg .b64 %rd<128>;"));
  EXPECT_TRUE(ptx.contains(".reg .f32 %f<128>;"));
  EXPECT_TRUE(ptx.contains("%rd126"));
  EXPECT_TRUE(ptx.contains("%f57"));
  EXPECT_FALSE(buildAlgebraFixture(mapOnlyModule(59, false)));
}

TEST(GpuTransportV2Test, ScalarConvertPreservesBf16RoundingBoundary) {
  auto built = buildAlgebraFixture(roundedScalarMapModule());
  ASSERT_TRUE(built);
  auto devices =
      llvm::to_vector(built->module->getOps<mlir::shuttle::DeviceModuleOp>());
  ASSERT_EQ(devices.size(), 1);
  llvm::StringRef ptx(
      reinterpret_cast<const char *>(devices[0].getCode().data()),
      devices[0].getCode().size());
  EXPECT_TRUE(ptx.contains("cvt.rn.bf16.f32 %h2, %f0;"));
  EXPECT_TRUE(ptx.contains("cvt.f32.bf16 %f1, %h2;"));
  EXPECT_FALSE(buildAlgebraFixture(bf16ArithmeticMapModule()));
}

TEST(GpuTransportV2Test,
     Abi14ContractMapShapeSweepHasClosedGenericDescriptors) {
  struct Shape {
    int64_t rows;
    int64_t reduction;
    int64_t columns;
  };
  constexpr Shape shapes[] = {
      {2, 3, 5}, {3, 5, 2}, {7, 13, 3}, {11, 3, 17}, {17, 3, 19},
  };
  bool sawMultipleBlocks = false;
  for (const Shape &shape : shapes) {
    SCOPED_TRACE(testing::Message()
                 << "M=" << shape.rows << " K=" << shape.reduction
                 << " N=" << shape.columns);
    auto built = buildFixture(contractMapModule(shape.rows, shape.reduction,
                                                shape.columns, "anonymous"));
    ASSERT_TRUE(built);
    auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
    ASSERT_TRUE(mlir::succeeded(bytes));
    auto independent = decodeInlineSchema2(*bytes);
    ASSERT_TRUE(independent);
    auto publicDescriptor = mlir::shuttle::GpuExecutable::Load(*bytes);
    ASSERT_TRUE(publicDescriptor.ok()) << publicDescriptor.status();

    ASSERT_EQ(independent->launches.size(), 2);
    ASSERT_EQ((*publicDescriptor)->launches().size(), 2);
    ASSERT_EQ((*publicDescriptor)->slots().size(), 4);
    EXPECT_EQ(independent->slots, 4);
    EXPECT_EQ(independent->externalSlots, 3);
    EXPECT_EQ(independent->scalarSlots, 0);
    EXPECT_EQ(independent->temporaryGroupOrder.size(), 1);
    EXPECT_EQ(independent->temporaryBytes,
              static_cast<uint64_t>(shape.rows * shape.columns * 4));
    auto temporaryType = cast<mlir::RankedTensorType>(
        (*publicDescriptor)->slots()[2].tensorType);
    EXPECT_EQ(temporaryType.getShape(),
              llvm::ArrayRef<int64_t>({shape.rows, shape.columns}));
    EXPECT_TRUE(temporaryType.getElementType().isF32());
    EXPECT_EQ((*publicDescriptor)->slots()[2].requiredBytes,
              shape.rows * shape.columns * 4);
    EXPECT_EQ(independent->launches[0].inputSlots,
              (std::vector<int64_t>{0, 1}));
    EXPECT_EQ(independent->launches[0].outputSlots, (std::vector<int64_t>{2}));
    EXPECT_TRUE(independent->launches[0].dependencies.empty());
    EXPECT_EQ(independent->launches[1].inputSlots, (std::vector<int64_t>{2}));
    EXPECT_EQ(independent->launches[1].outputSlots, (std::vector<int64_t>{3}));
    EXPECT_EQ(independent->launches[1].dependencies, (std::vector<int64_t>{0}));
    const uint64_t outputElements = shape.rows * shape.columns;
    const uint64_t workgroupThreads =
        std::min<uint64_t>(256, ((outputElements + 31) / 32) * 32);
    const std::array<uint64_t, 3> expectedGrid{
        (outputElements + workgroupThreads - 1) / workgroupThreads, 1, 1};
    const std::array<uint64_t, 3> expectedBlock{workgroupThreads, 1, 1};
    for (const auto &launch : independent->launches) {
      EXPECT_EQ(launch.grid, expectedGrid);
      EXPECT_EQ(launch.block, expectedBlock);
      EXPECT_EQ(launch.dynamicSharedMemoryBytes, 0);
    }
    sawMultipleBlocks |= expectedGrid[0] > 1;
  }
  EXPECT_TRUE(sawMultipleBlocks);
}

TEST(GpuTransportV2Test,
     Abi14SourceOrderedContractMapUsesAscendingFmaAndExplicitBf16Map) {
  auto built = buildFixture(contractMapModule(7, 13, 3, "first_name"));
  auto renamed = buildFixture(contractMapModule(7, 13, 3, "second_name"));
  ASSERT_TRUE(built);
  ASSERT_TRUE(renamed);
  auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  auto renamedBytes =
      mlir::shuttle::serializeGpuExecutableBundle(*renamed->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  ASSERT_TRUE(mlir::succeeded(renamedBytes));
  auto decoded = decodeInlineSchema2(*bytes);
  auto renamedDecoded = decodeInlineSchema2(*renamedBytes);
  ASSERT_TRUE(decoded);
  ASSERT_TRUE(renamedDecoded);
  ASSERT_EQ(decoded->launches.size(), 2);
  ASSERT_EQ(renamedDecoded->launches.size(), 2);
  EXPECT_EQ(decoded->code, renamedDecoded->code);

  const auto &contractLaunch = decoded->launches[0];
  llvm::StringRef contractPtx(
      reinterpret_cast<const char *>(decoded->code.data() +
                                     contractLaunch.codeOffset),
      contractLaunch.codeLength);
  const auto &mapLaunch = decoded->launches[1];
  llvm::StringRef mapPtx(reinterpret_cast<const char *>(decoded->code.data() +
                                                        mapLaunch.codeOffset),
                         mapLaunch.codeLength);
  size_t zero = contractPtx.find("mov.f32 %f0, 0f00000000;");
  size_t induction = contractPtx.find("mov.u64 %rd10, 0;");
  size_t loop = contractPtx.find("$contract_loop:");
  size_t bound = contractPtx.find("setp.ge.u64 %p1, %rd10, 13;", loop);
  size_t lhsRow = contractPtx.find("mad.lo.u64 %rd11, %rd11, 7, %rd8;", bound);
  size_t lhsIndex =
      contractPtx.find("mad.lo.u64 %rd11, %rd11, 13, %rd10;", lhsRow);
  size_t rhsReduction =
      contractPtx.find("mad.lo.u64 %rd13, %rd13, 13, %rd10;", lhsIndex);
  size_t rhsIndex =
      contractPtx.find("mad.lo.u64 %rd13, %rd13, 3, %rd9;", rhsReduction);
  size_t fma = contractPtx.find("fma.rn.f32 %f0, %f1, %f2, %f0;", rhsIndex);
  size_t increment = contractPtx.find("add.u64 %rd10, %rd10, 1;", fma);
  size_t branch = contractPtx.find("bra $contract_loop;", increment);
  ASSERT_NE(zero, llvm::StringRef::npos);
  ASSERT_NE(induction, llvm::StringRef::npos);
  ASSERT_NE(loop, llvm::StringRef::npos);
  ASSERT_NE(bound, llvm::StringRef::npos);
  ASSERT_NE(lhsRow, llvm::StringRef::npos);
  ASSERT_NE(lhsIndex, llvm::StringRef::npos);
  ASSERT_NE(rhsReduction, llvm::StringRef::npos);
  ASSERT_NE(rhsIndex, llvm::StringRef::npos);
  ASSERT_NE(fma, llvm::StringRef::npos);
  EXPECT_LT(zero, induction);
  EXPECT_LT(induction, loop);
  EXPECT_LT(loop, bound);
  EXPECT_LT(bound, lhsRow);
  EXPECT_LT(lhsRow, lhsIndex);
  EXPECT_LT(lhsIndex, rhsReduction);
  EXPECT_LT(rhsReduction, rhsIndex);
  EXPECT_LT(rhsIndex, fma);
  ASSERT_NE(increment, llvm::StringRef::npos);
  EXPECT_LT(fma, increment);
  ASSERT_NE(branch, llvm::StringRef::npos);
  EXPECT_LT(increment, branch);
  EXPECT_EQ(contractPtx.find("fma.rn.f32", fma + 1), llvm::StringRef::npos);
  EXPECT_FALSE(contractPtx.contains("mul.rn.f32"));
  EXPECT_TRUE(mapPtx.contains("cvt.rn.bf16.f32"));
  EXPECT_FALSE(mapPtx.contains("fma.rn.f32"));
}

TEST(GpuTransportV2Test,
     Abi14FastContractMapWarpGeometryCoversReductionBoundaries) {
  struct Shape {
    int64_t rows;
    int64_t reduction;
    int64_t columns;
    uint64_t contractGrid;
    uint64_t contractThreads;
  };
  constexpr Shape shapes[] = {
      {1, 16, 1, 1, 32},
      {1, 32, 9, 2, 256},
      {2, 33, 3, 1, 192},
      {17, 65, 19, 41, 256},
  };
  for (const Shape &shape : shapes) {
    SCOPED_TRACE(testing::Message()
                 << "M=" << shape.rows << " K=" << shape.reduction
                 << " N=" << shape.columns);
    auto built =
        buildGpuContractFixture(contractMapModule(shape.rows, shape.reduction,
                                                  shape.columns, "anonymous"),
                                mlir::shuttle::NumericalPolicy::Fast);
    ASSERT_TRUE(built);
    auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
    ASSERT_TRUE(mlir::succeeded(bytes));
    auto decoded = decodeInlineSchema2(*bytes);
    ASSERT_TRUE(decoded);
    auto publicDescriptor = mlir::shuttle::GpuExecutable::Load(*bytes);
    ASSERT_TRUE(publicDescriptor.ok()) << publicDescriptor.status();
    ASSERT_EQ(decoded->launches.size(), 2);
    EXPECT_EQ(decoded->policy, 1);
    EXPECT_EQ(decoded->slots, 4);
    EXPECT_EQ(decoded->externalSlots, 3);
    EXPECT_EQ(decoded->temporaryGroupOrder.size(), 1);
    EXPECT_EQ(decoded->temporaryBytes,
              static_cast<uint64_t>(shape.rows * shape.columns * 4));
    EXPECT_EQ(decoded->launches[0].grid,
              (std::array<uint64_t, 3>{shape.contractGrid, 1, 1}));
    EXPECT_EQ(decoded->launches[0].block,
              (std::array<uint64_t, 3>{shape.contractThreads, 1, 1}));
    EXPECT_EQ(decoded->launches[0].dynamicSharedMemoryBytes, 0);

    const uint64_t outputs = shape.rows * shape.columns;
    const uint64_t mapThreads =
        std::min<uint64_t>(256, ((outputs + 31) / 32) * 32);
    EXPECT_EQ(decoded->launches[1].grid,
              (std::array<uint64_t, 3>{(outputs + mapThreads - 1) / mapThreads,
                                       1, 1}));
    EXPECT_EQ(decoded->launches[1].block,
              (std::array<uint64_t, 3>{mapThreads, 1, 1}));
    EXPECT_EQ(decoded->launches[1].dynamicSharedMemoryBytes, 0);
  }
}

TEST(GpuTransportV2Test,
     Abi14FastContractMapPtxUsesLaneStridedFmaAndFixedShuffleTree) {
  auto built = buildGpuContractFixture(contractMapModule(2, 33, 3, "anonymous"),
                                       mlir::shuttle::NumericalPolicy::Fast);
  ASSERT_TRUE(built);
  auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  auto decoded = decodeInlineSchema2(*bytes);
  ASSERT_TRUE(decoded);
  ASSERT_EQ(decoded->launches.size(), 2);

  const auto &contractLaunch = decoded->launches[0];
  llvm::StringRef contractPtx(
      reinterpret_cast<const char *>(decoded->code.data() +
                                     contractLaunch.codeOffset),
      contractLaunch.codeLength);
  const auto &mapLaunch = decoded->launches[1];
  llvm::StringRef mapPtx(reinterpret_cast<const char *>(decoded->code.data() +
                                                        mapLaunch.codeOffset),
                         mapLaunch.codeLength);

  EXPECT_TRUE(contractPtx.contains("and.b32"));
  EXPECT_TRUE(contractPtx.contains(", 31;"));
  EXPECT_TRUE(contractPtx.contains(", 32;"));
  EXPECT_EQ(contractPtx.count("fma.rn.f32"), 1u);
  EXPECT_EQ(contractPtx.count("shfl.sync.down.b32"), 5u);
  EXPECT_EQ(contractPtx.count("add.rn.f32"), 5u);
  EXPECT_FALSE(contractPtx.contains("atom."));
  EXPECT_FALSE(contractPtx.contains("mma."));

  size_t cursor = contractPtx.find("fma.rn.f32");
  ASSERT_NE(cursor, llvm::StringRef::npos);
  for (int offset : {16, 8, 4, 2, 1}) {
    size_t shuffle = contractPtx.find("shfl.sync.down.b32", cursor);
    ASSERT_NE(shuffle, llvm::StringRef::npos);
    size_t immediate = contractPtx.find(
        (", " + llvm::Twine(offset) + ", 31, 0xffffffff;").str(), shuffle);
    ASSERT_NE(immediate, llvm::StringRef::npos);
    size_t add = contractPtx.find("add.rn.f32", immediate);
    ASSERT_NE(add, llvm::StringRef::npos);
    EXPECT_LT(shuffle, immediate);
    EXPECT_LT(immediate, add);
    cursor = add + 1;
  }
  size_t laneGuard = contractPtx.find("setp.ne.u32", cursor);
  size_t store = contractPtx.find("st.global.f32", laneGuard);
  ASSERT_NE(laneGuard, llvm::StringRef::npos);
  ASSERT_NE(store, llvm::StringRef::npos);
  EXPECT_LT(laneGuard, store);
  EXPECT_TRUE(mapPtx.contains("cvt.rn.bf16.f32"));
  EXPECT_FALSE(mapPtx.contains("shfl.sync.down.b32"));
}

TEST(GpuTransportV2Test,
     Abi14ContractPoliciesSeparateCodeDeviceAndBundleRoots) {
  std::string source = contractMapModule(2, 33, 3, "anonymous");
  auto sourceOrdered = buildGpuContractFixture(
      source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  auto fast =
      buildGpuContractFixture(source, mlir::shuttle::NumericalPolicy::Fast);
  ASSERT_TRUE(sourceOrdered);
  ASSERT_TRUE(fast);
  auto sourceBytes =
      mlir::shuttle::serializeGpuExecutableBundle(*sourceOrdered->module);
  auto fastBytes = mlir::shuttle::serializeGpuExecutableBundle(*fast->module);
  ASSERT_TRUE(mlir::succeeded(sourceBytes));
  ASSERT_TRUE(mlir::succeeded(fastBytes));
  auto sourceDecoded = decodeInlineSchema2(*sourceBytes);
  auto fastDecoded = decodeInlineSchema2(*fastBytes);
  ASSERT_TRUE(sourceDecoded);
  ASSERT_TRUE(fastDecoded);
  EXPECT_EQ(sourceDecoded->policy, 0);
  EXPECT_EQ(fastDecoded->policy, 1);
  EXPECT_NE(sourceDecoded->code, fastDecoded->code);
  llvm::ArrayRef<uint8_t> sourcePayload(*sourceBytes);
  llvm::ArrayRef<uint8_t> fastPayload(*fastBytes);
  EXPECT_NE(sourcePayload.slice(sourceDecoded->deviceRootPayloadOffset, 64),
            fastPayload.slice(fastDecoded->deviceRootPayloadOffset, 64));
  EXPECT_NE(sourcePayload.slice(sourceDecoded->bundleRootPayloadOffset, 64),
            fastPayload.slice(fastDecoded->bundleRootPayloadOffset, 64));
  EXPECT_NE(mlir::shuttle::gpuExecutableBundleDigest(*sourceBytes),
            mlir::shuttle::gpuExecutableBundleDigest(*fastBytes));

  llvm::StringRef sourcePtx(
      reinterpret_cast<const char *>(sourceDecoded->code.data()),
      sourceDecoded->launches.front().codeLength);
  llvm::StringRef fastPtx(
      reinterpret_cast<const char *>(fastDecoded->code.data()),
      fastDecoded->launches.front().codeLength);
  EXPECT_TRUE(sourcePtx.contains("add.u64 %rd10, %rd10, 1;"));
  EXPECT_FALSE(sourcePtx.contains("shfl.sync.down.b32"));
  EXPECT_TRUE(fastPtx.contains("shfl.sync.down.b32"));
  EXPECT_FALSE(fastPtx.contains("add.u64 %rd10, %rd10, 1;"));
}

TEST(GpuTransportV2Test, Abi14ContractMapTransportMutationsFailClosed) {
  auto built = buildFixture(contractMapModule(3, 5, 2, "anonymous"));
  ASSERT_TRUE(built);
  auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  auto decoded = decodeInlineSchema2(*bytes);
  ASSERT_TRUE(decoded);

  llvm::SmallVector<uint8_t> corruptRoot(*bytes);
  corruptRoot.back() ^= 1;
  EXPECT_FALSE(decodeInlineSchema2(corruptRoot));
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(corruptRoot).ok());

  auto oversized = oversizedLaunchGrid(*bytes, *decoded);
  EXPECT_FALSE(decodeInlineSchema2(oversized));
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(oversized).ok());

  auto futureResultInput = futureProducedResultInput(*bytes, *decoded);
  ASSERT_FALSE(futureResultInput.empty());
  EXPECT_FALSE(decodeInlineSchema2(futureResultInput));
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(futureResultInput).ok());
}

TEST(GpuTransportV2Test, Abi14ContractMapRejectsUnsupportedShapes) {
  constexpr llvm::StringLiteral multipleContractDimensions = R"mlir(
module @multiple_contract_dimensions {
  func.func @main(%lhs: tensor<2x3x5xbf16>, %rhs: tensor<3x5x7xbf16>) -> tensor<2x7xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1, 2] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<2x3x5xbf16>, tensor<3x5x7xbf16>) -> tensor<2x7xf32>
    %result = stablehlo.convert %contract : (tensor<2x7xf32>) -> tensor<2x7xbf16>
    return %result : tensor<2x7xbf16>
  }
}
)mlir";
  constexpr llvm::StringLiteral f32Inputs = R"mlir(
module @f32_inputs {
  func.func @main(%lhs: tensor<3x5xf32>, %rhs: tensor<5x2xf32>) -> tensor<3x2xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x5xf32>, tensor<5x2xf32>) -> tensor<3x2xf32>
    %result = stablehlo.convert %contract : (tensor<3x2xf32>) -> tensor<3x2xbf16>
    return %result : tensor<3x2xbf16>
  }
}
)mlir";
  EXPECT_FALSE(
      buildFixture(contractMapModule(3, 5, 2, "high_precision", "HIGH", true)));
  EXPECT_FALSE(buildFixture(
      contractMapModule(3, 5, 2, "external_f32", "DEFAULT", false)));
  EXPECT_FALSE(buildFixture(kContractMapWithLiveF32Result));
  EXPECT_FALSE(buildFixture(multipleContractDimensions));
  EXPECT_FALSE(buildFixture(f32Inputs));
}

TEST(GpuTransportV2Test, Abi17ThroughAbi21PreserveEveryTransportAndRuntimeCap) {
  auto built = buildFixture();
  ASSERT_TRUE(built);
  auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  auto decoded = decodeInlineSchema2(*bytes);
  ASSERT_TRUE(decoded);
  auto expectOnlyCapsRejected = [](llvm::StringRef label,
                                   llvm::ArrayRef<uint8_t> mutation,
                                   DecodeLimits raised) {
    SCOPED_TRACE(label.str());
    EXPECT_TRUE(decodeInlineSchema2(mutation, raised));
    EXPECT_FALSE(decodeInlineSchema2(mutation));
    EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(mutation).ok());
  };

  const auto baselinePacket = completeSchema2Packet({128, 128});
  ASSERT_TRUE(decodeInlineSchema2(baselinePacket));
  ASSERT_TRUE(mlir::shuttle::GpuExecutable::Load(baselinePacket).ok());

  EXPECT_EQ(mlir::shuttle::kMaximumGpuTransportBytes, 16 * 1024 * 1024);
  EXPECT_EQ(mlir::shuttle::kMaximumGpuCodeBytes, 8 * 1024 * 1024);
  EXPECT_EQ(mlir::shuttle::kMaximumGpuEntryCodeBytes, 512 * 1024);
  EXPECT_EQ(mlir::shuttle::kMaximumGpuExecutableRecords, 256);
  EXPECT_EQ(mlir::shuttle::kMaximumGpuAggregateTaskPositions,
            256 * 1024 * 1024);

  llvm::SmallVector<uint64_t> transportLengths(32, 512 * 1024);
  const auto transportPacket = completeSchema2Packet(transportLengths);
  ASSERT_GT(transportPacket.size(),
            mlir::shuttle::kMaximumGpuTransportBytes);
  DecodeLimits raisedTransport;
  raisedTransport.transportBytes = transportPacket.size();
  raisedTransport.codeBytes = 16 * 1024 * 1024;
  expectOnlyCapsRejected("transport", transportPacket, raisedTransport);

  llvm::SmallVector<uint64_t> codeLengths(
      17, mlir::shuttle::kMaximumGpuCodeBytes / 17 + 1);
  const auto codePacket = completeSchema2Packet(codeLengths);
  ASSERT_LT(codePacket.size(), mlir::shuttle::kMaximumGpuTransportBytes);
  DecodeLimits raisedCode;
  raisedCode.codeBytes = mlir::shuttle::kMaximumGpuCodeBytes + 17;
  expectOnlyCapsRejected("code", codePacket, raisedCode);

  // Device closure requires one distinct produced slot per entry, so the
  // shared 256-record ceiling necessarily couples entry and slot counts.
  llvm::SmallVector<uint64_t> recordLengths(257, 128);
  DecodeLimits raisedRecords;
  raisedRecords.records = 258;
  expectOnlyCapsRejected("records", completeSchema2Packet(recordLengths),
                         raisedRecords);

  DecodeLimits raisedEntry;
  raisedEntry.entryCodeBytes =
      mlir::shuttle::kMaximumGpuEntryCodeBytes + 1;
  expectOnlyCapsRejected(
      "entry",
      completeSchema2Packet(
          {mlir::shuttle::kMaximumGpuEntryCodeBytes + 1, 128}),
      raisedEntry);

  llvm::SmallVector<uint8_t> sharedCap(*bytes);
  writeU32(sharedCap, decoded->launchSharedOffsets.front(), 16 * 1024 + 4);
  rehashDeviceAndBundle(sharedCap, *decoded);
  DecodeLimits raisedShared;
  raisedShared.sharedBytes = 16 * 1024 + 4;
  expectOnlyCapsRejected("shared", sharedCap, raisedShared);

  const auto positionsPacket = completeSchema2Packet(
      {128, 128}, mlir::shuttle::kMaximumGpuAggregateTaskPositions + 1);
  DecodeLimits raisedPositions;
  raisedPositions.taskPositions =
      2 * (mlir::shuttle::kMaximumGpuAggregateTaskPositions + 1);
  expectOnlyCapsRejected("positions", positionsPacket, raisedPositions);

  llvm::SmallVector<uint64_t> slotCountLengths(256, 128);
  DecodeLimits raisedSlots;
  raisedSlots.records = 257;
  expectOnlyCapsRejected("slots", completeSchema2Packet(slotCountLengths),
                         raisedSlots);

  DecodeLimits raisedRank;
  raisedRank.rank = 9;
  expectOnlyCapsRejected("rank", completeSchema2Packet({128, 128}, 1, 0, 9),
                         raisedRank);

  llvm::SmallVector<uint8_t> slotByteCap(*bytes);
  const uint64_t publicSlot = llvm::find(decoded->slotStorage, 0) -
                              decoded->slotStorage.begin();
  ASSERT_LT(publicSlot, decoded->slotStorage.size());
  int64_t oversizedExtent = decoded->slotDimensions[publicSlot].front();
  int64_t oversizedBytes = decoded->slotRequiredBytes[publicSlot];
  while (oversizedBytes <= 32 * 1024 * 1024) {
    oversizedExtent *= 2;
    oversizedBytes *= 2;
  }
  writeU64(slotByteCap,
           decoded->slotDimensionOffsets[publicSlot].front(),
           oversizedExtent);
  writeU64(slotByteCap, decoded->slotRequiredOffsets[publicSlot],
           oversizedBytes);
  rehashInvocationAndBundle(slotByteCap, *decoded);
  DecodeLimits raisedSlotBytes;
  raisedSlotBytes.slotBytes = oversizedBytes;
  expectOnlyCapsRejected("slot bytes", slotByteCap, raisedSlotBytes);

  llvm::SmallVector<uint8_t> temporaryCap(*bytes);
  int64_t widenedTemporaries = 0;
  int64_t widenedTemporaryBytes = 0;
  for (size_t slot = 0; slot < decoded->slotStorage.size() &&
                        widenedTemporaries < 9;
       ++slot) {
    if (decoded->slotStorage[slot] != 1 ||
        decoded->slotDimensionOffsets[slot].empty())
      continue;
    const int64_t priorExtent = decoded->slotDimensions[slot].front();
    ASSERT_EQ(decoded->slotRequiredBytes[slot] % priorExtent, 0);
    const int64_t bytesPerLeadingElement =
        decoded->slotRequiredBytes[slot] / priorExtent;
    const int64_t widenedExtent =
        (32 * 1024 * 1024) / bytesPerLeadingElement;
    const int64_t widenedBytes = widenedExtent * bytesPerLeadingElement;
    writeU64(temporaryCap, decoded->slotDimensionOffsets[slot].front(),
             widenedExtent);
    writeU64(temporaryCap, decoded->slotRequiredOffsets[slot], widenedBytes);
    writeU64(temporaryCap, decoded->slotReuseOffsets[slot], 1000 + slot);
    widenedTemporaryBytes += widenedBytes;
    ++widenedTemporaries;
  }
  ASSERT_EQ(widenedTemporaries, 9);
  ASSERT_GT(widenedTemporaryBytes, 256 * 1024 * 1024);
  rehashInvocationAndBundle(temporaryCap, *decoded);
  DecodeLimits raisedTemporaryBytes;
  raisedTemporaryBytes.temporaryBytes =
      widenedTemporaryBytes + decoded->temporaryBytes;
  expectOnlyCapsRejected("temporary bytes", temporaryCap,
                         raisedTemporaryBytes);
}

TEST(GpuTransportV2Test,
     Abi16ProjectedBatchedContractClosesTransportGeometryAndAddressing) {
  struct Linearization {
    int64_t linearRegister;
    std::array<int64_t, 4> extents;
    std::array<int64_t, 4> coordinateRegisters;
  };
  struct Contract {
    llvm::StringRef orientation;
    int64_t ordinal;
    std::vector<int64_t> inputSlots;
    uint64_t sourceGrid;
    uint64_t sourceBlock;
    uint64_t fastGrid;
    uint64_t fastBlock;
    std::array<Linearization, 2> inputs;
  };
  struct Boundary {
    llvm::StringRef filename;
    int64_t launches;
    int64_t slots;
    int64_t operandCount;
    std::vector<int64_t> externalBytes;
    std::vector<Contract> contracts;
  };

  const std::vector<Boundary> boundaries = {
      {"jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir",
       6,
       8,
       2,
       {420, 924, 660},
       {{"forward",
         4,
         {3, 5},
         2,
         256,
         42,
         256,
         {{{13, {2, 3, 5, 7}, {8, 9, 10, 12}},
           {15, {2, 3, 11, 7}, {8, 9, 11, 12}}}}}}},
      {"jax-0.10.1-bf16-batched_contract-2x5x3x7x11-backward.mlir",
       11,
       14,
       3,
       {420, 924, 660, 924, 420},
       {{"dRHS",
         5,
         {7, 4},
         2,
         256,
         58,
         256,
         {{{13, {2, 3, 5, 11}, {8, 9, 12, 10}},
           {15, {2, 3, 5, 7}, {8, 9, 12, 11}}}}},
        {"dLHS",
         6,
         {7, 6},
         1,
         224,
         27,
         256,
         {{{13, {2, 3, 5, 11}, {8, 9, 10, 12}},
           {15, {2, 3, 11, 7}, {8, 9, 12, 11}}}}}}},
      {"jax-0.10.1-bf16-batched_contract-2x5x3x7x11-composed.mlir",
       13,
       16,
       3,
       {420, 924, 660, 660, 924, 420},
       {{"forward",
         4,
         {4, 6},
         2,
         256,
         42,
         256,
         {{{13, {2, 3, 5, 7}, {8, 9, 10, 12}},
           {15, {2, 3, 11, 7}, {8, 9, 11, 12}}}}},
        {"dRHS",
         7,
         {9, 4},
         2,
         256,
         58,
         256,
         {{{13, {2, 3, 5, 11}, {8, 9, 12, 10}},
           {15, {2, 3, 5, 7}, {8, 9, 12, 11}}}}},
        {"dLHS",
         8,
         {9, 6},
         1,
         224,
         27,
         256,
         {{{13, {2, 3, 5, 11}, {8, 9, 10, 12}},
           {15, {2, 3, 11, 7}, {8, 9, 12, 11}}}}}}},
      {"jax-0.10.1-bf16-batched_contract-3x4x2x13x6-forward.mlir",
       6,
       8,
       2,
       {624, 936, 288},
       {{"forward",
         4,
         {3, 5},
         1,
         160,
         18,
         256,
         {{{13, {3, 2, 4, 13}, {8, 9, 10, 12}},
           {15, {3, 2, 6, 13}, {8, 9, 11, 12}}}}}}},
      {"jax-0.10.1-bf16-batched_contract-3x4x2x13x6-backward.mlir",
       11,
       14,
       3,
       {624, 936, 288, 936, 624},
       {{"dRHS",
         5,
         {7, 4},
         2,
         256,
         59,
         256,
         {{{13, {3, 2, 4, 6}, {8, 9, 12, 10}},
           {15, {3, 2, 4, 13}, {8, 9, 12, 11}}}}},
        {"dLHS",
         6,
         {7, 6},
         2,
         256,
         39,
         256,
         {{{13, {3, 2, 4, 6}, {8, 9, 10, 12}},
           {15, {3, 2, 6, 13}, {8, 9, 12, 11}}}}}}},
      {"jax-0.10.1-bf16-batched_contract-3x4x2x13x6-composed.mlir",
       13,
       16,
       3,
       {624, 936, 288, 288, 936, 624},
       {{"forward",
         4,
         {4, 6},
         1,
         160,
         18,
         256,
         {{{13, {3, 2, 4, 13}, {8, 9, 10, 12}},
           {15, {3, 2, 6, 13}, {8, 9, 11, 12}}}}},
        {"dRHS",
         7,
         {9, 4},
         2,
         256,
         59,
         256,
         {{{13, {3, 2, 4, 6}, {8, 9, 12, 10}},
           {15, {3, 2, 4, 13}, {8, 9, 12, 11}}}}},
        {"dLHS",
         8,
         {9, 6},
         2,
         256,
         39,
         256,
         {{{13, {3, 2, 4, 6}, {8, 9, 10, 12}},
           {15, {3, 2, 6, 13}, {8, 9, 12, 11}}}}}}},
  };

  auto contractPtx = [](const Decoded &decoded, int64_t ordinal) {
    const auto &launch = decoded.launches[ordinal];
    return llvm::StringRef(
        reinterpret_cast<const char *>(decoded.code.data() + launch.codeOffset),
        launch.codeLength);
  };
  auto expectLinearization = [](llvm::StringRef ptx,
                                const Linearization &input) {
    size_t cursor = 0;
    for (auto [extent, coordinate] :
         llvm::zip_equal(input.extents, input.coordinateRegisters)) {
      std::string expected =
          ("mad.lo.u64 %rd" + llvm::Twine(input.linearRegister) + ", %rd" +
           llvm::Twine(input.linearRegister) + ", " + llvm::Twine(extent) +
           ", %rd" + llvm::Twine(coordinate) + ";")
              .str();
      cursor = ptx.find(expected, cursor);
      ASSERT_NE(cursor, llvm::StringRef::npos) << expected;
      ++cursor;
    }
    const int64_t pointerRegister = input.linearRegister + 1;
    const int64_t parameter = input.linearRegister == 13 ? 0 : 1;
    std::string address =
        ("ld.param.u64 %rd" + llvm::Twine(pointerRegister) + ", [p" +
         llvm::Twine(parameter) + "];\n  mad.lo.u64 %rd" +
         llvm::Twine(pointerRegister) + ", %rd" +
         llvm::Twine(input.linearRegister) + ", 4, %rd" +
         llvm::Twine(pointerRegister) + ";")
            .str();
    EXPECT_TRUE(ptx.contains(address)) << address;
  };

  for (const Boundary &boundary : boundaries) {
    SCOPED_TRACE(boundary.filename.str());
    auto source = fixtureSource(boundary.filename);
    ASSERT_TRUE(source);
    auto sourceOrdered = buildGpuContractFixture(
        *source, mlir::shuttle::NumericalPolicy::SourceOrdered);
    auto fast =
        buildGpuContractFixture(*source, mlir::shuttle::NumericalPolicy::Fast);
    ASSERT_TRUE(sourceOrdered);
    ASSERT_TRUE(fast);

    auto sourceBytes =
        mlir::shuttle::serializeGpuExecutableBundle(*sourceOrdered->module);
    auto fastBytes = mlir::shuttle::serializeGpuExecutableBundle(*fast->module);
    ASSERT_TRUE(mlir::succeeded(sourceBytes));
    ASSERT_TRUE(mlir::succeeded(fastBytes));
    auto sourceDecoded = decodeInlineSchema2(*sourceBytes);
    auto fastDecoded = decodeInlineSchema2(*fastBytes);
    ASSERT_TRUE(sourceDecoded);
    ASSERT_TRUE(fastDecoded);
    auto sourcePublic = mlir::shuttle::GpuExecutable::Load(*sourceBytes);
    auto fastPublic = mlir::shuttle::GpuExecutable::Load(*fastBytes);
    ASSERT_TRUE(sourcePublic.ok()) << sourcePublic.status();
    ASSERT_TRUE(fastPublic.ok()) << fastPublic.status();

    for (const Decoded *decoded : {&*sourceDecoded, &*fastDecoded}) {
      ASSERT_EQ(decoded->launches.size(), boundary.launches);
      EXPECT_EQ(decoded->slots, boundary.slots);
      EXPECT_EQ(decoded->externalSlots, boundary.externalBytes.size());
      EXPECT_EQ(decoded->scalarSlots, 0);
      for (int64_t ordinal = 0; ordinal < boundary.launches; ++ordinal) {
        EXPECT_EQ(decoded->launches[ordinal].taskOrdinal, ordinal);
        EXPECT_EQ(decoded->launches[ordinal].outputSlots,
                  (std::vector<int64_t>{boundary.operandCount + ordinal}));
      }
    }
    EXPECT_EQ(sourceDecoded->policy, 0);
    EXPECT_EQ(fastDecoded->policy, 1);

    for (const auto *loaded : {&*sourcePublic, &*fastPublic}) {
      ASSERT_EQ((*loaded)->launches().size(), boundary.launches);
      ASSERT_EQ((*loaded)->slots().size(), boundary.slots);
      ASSERT_EQ((*loaded)->externalBindings().size(),
                boundary.externalBytes.size());
      for (auto [ordinal, binding] :
           llvm::enumerate((*loaded)->externalBindings())) {
        const bool isOperand = ordinal < boundary.operandCount;
        EXPECT_EQ(binding.kind,
                  isOperand ? mlir::shuttle::ExecutableBindingKind::Operand
                            : mlir::shuttle::ExecutableBindingKind::Result);
        EXPECT_EQ(binding.index,
                  isOperand ? ordinal : ordinal - boundary.operandCount);
      }
      std::vector<mlir::shuttle::GpuExternalBinding> structuralBindings =
          externalBindingsBySlot(**loaded);
      for (auto [binding, expectedBytes] : llvm::zip_equal(
               structuralBindings, boundary.externalBytes))
        EXPECT_EQ(binding.requiredBytes, expectedBytes);
    }

    for (const Contract &contract : boundary.contracts) {
      SCOPED_TRACE(contract.orientation.str());
      const auto &sourceLaunch = sourceDecoded->launches[contract.ordinal];
      const auto &fastLaunch = fastDecoded->launches[contract.ordinal];
      EXPECT_EQ(sourceLaunch.inputSlots, contract.inputSlots);
      EXPECT_EQ(fastLaunch.inputSlots, contract.inputSlots);
      EXPECT_EQ(sourceLaunch.grid,
                (std::array<uint64_t, 3>{contract.sourceGrid, 1, 1}));
      EXPECT_EQ(sourceLaunch.block,
                (std::array<uint64_t, 3>{contract.sourceBlock, 1, 1}));
      EXPECT_EQ(fastLaunch.grid,
                (std::array<uint64_t, 3>{contract.fastGrid, 1, 1}));
      EXPECT_EQ(fastLaunch.block,
                (std::array<uint64_t, 3>{contract.fastBlock, 1, 1}));
      EXPECT_EQ(sourceLaunch.dynamicSharedMemoryBytes, 0);
      EXPECT_EQ(fastLaunch.dynamicSharedMemoryBytes, 0);

      llvm::StringRef sourcePtx = contractPtx(*sourceDecoded, contract.ordinal);
      llvm::StringRef fastPtx = contractPtx(*fastDecoded, contract.ordinal);
      for (llvm::StringRef ptx : {sourcePtx, fastPtx}) {
        for (const Linearization &input : contract.inputs)
          expectLinearization(ptx, input);
        EXPECT_EQ(ptx.count("fma.rn.f32"), 1u);
        EXPECT_EQ(ptx.count("ld.global.f32"), 2u);
        EXPECT_FALSE(ptx.contains("ld.global.u16"));
      }
      EXPECT_TRUE(sourcePtx.contains("add.u64 %rd12, %rd12, 1;"));
      EXPECT_FALSE(sourcePtx.contains("shfl.sync.down.b32"));
      EXPECT_FALSE(fastPtx.contains("add.u64 %rd12, %rd12, 1;"));
      EXPECT_EQ(fastPtx.count("shfl.sync.down.b32"), 5u);
    }
    EXPECT_NE(sourceDecoded->code, fastDecoded->code);
    EXPECT_NE(*sourceBytes, *fastBytes);

    auto futureResultInput =
        futureProducedResultInput(*sourceBytes, *sourceDecoded);
    ASSERT_FALSE(futureResultInput.empty());
    EXPECT_FALSE(decodeInlineSchema2(futureResultInput));
    EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(futureResultInput).ok());
  }
}

TEST(GpuTransportV2Test, Abi16BatchedContractFingerprintsIgnoreModuleSymbols) {
  struct Rename {
    llvm::StringRef filename;
    llvm::StringRef original;
    llvm::StringRef replacement;
  };
  constexpr Rename renames[] = {
      {"jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir",
       "module @jit_forward", "module @renamed_forward"},
      {"jax-0.10.1-bf16-batched_contract-2x5x3x7x11-composed.mlir",
       "module @jit_composed", "module @renamed_composed"},
  };
  for (const Rename &rename : renames) {
    SCOPED_TRACE(rename.filename.str());
    auto source = fixtureSource(rename.filename);
    ASSERT_TRUE(source);
    llvm::StringRef sourceRef(*source);
    const size_t first = sourceRef.find(rename.original);
    ASSERT_NE(first, llvm::StringRef::npos);
    ASSERT_EQ(sourceRef.find(rename.original, first + rename.original.size()),
              llvm::StringRef::npos);
    std::string renamed = *source;
    renamed.replace(first, rename.original.size(), rename.replacement.str());

    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      auto originalBuilt = buildGpuContractFixture(*source, policy);
      auto renamedBuilt = buildGpuContractFixture(renamed, policy);
      ASSERT_TRUE(originalBuilt);
      ASSERT_TRUE(renamedBuilt);
      auto originalBytes =
          mlir::shuttle::serializeGpuExecutableBundle(*originalBuilt->module);
      auto renamedBytes =
          mlir::shuttle::serializeGpuExecutableBundle(*renamedBuilt->module);
      ASSERT_TRUE(mlir::succeeded(originalBytes));
      ASSERT_TRUE(mlir::succeeded(renamedBytes));
      EXPECT_EQ(*originalBytes, *renamedBytes);
    }
  }
}

TEST(GpuTransportV2Test,
     Abi17RankNFoldClosesTransportGeometryBytesAndAddressing) {
  struct Shape {
    llvm::StringRef stem;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> resultShape;
    std::vector<int64_t> inputStrides;
    std::vector<int64_t> resultStrides;
    std::vector<int64_t> publicStrides;
    int64_t outputCount;
    int64_t reductionExtent;
    uint64_t threads;
    uint32_t scratch;
    int64_t inputBytes;
    int64_t resultBytes;
    int64_t publicBytes;
    int64_t sampleOutput;
    int64_t sampleLeaf;
    int64_t sampleInputByte;
    int64_t sampleOutputByte;
  };
  struct Boundary {
    llvm::StringRef suffix;
    int64_t launches;
    int64_t folds;
    int64_t externalSlots;
  };
  const Shape shapes[] = {
      {"2x5x17",
       {2, 5, 17},
       {2, 5},
       {340, 68, 4},
       {20, 4},
       {170, 34, 2},
       10,
       17,
       32,
       128,
       680,
       40,
       340,
       7,
       11,
       520,
       28},
      {"3x2x4x7x269",
       {3, 2, 4, 7, 269},
       {3, 2, 4, 7},
       {60256, 30128, 7532, 1076, 4},
       {224, 112, 28, 4},
       {30128, 15064, 3766, 538, 2},
       168,
       269,
       256,
       1024,
       180768,
       672,
       90384,
       137,
       268,
       148484,
       548},
  };
  constexpr Boundary boundaries[] = {
      {"forward", 6, 1, 2},
      {"backward", 12, 2, 3},
      {"composed", 14, 2, 4},
  };

  for (const Shape &shape : shapes) {
    for (const Boundary &boundary : boundaries) {
      std::string filename =
          ("jax-0.10.1-bf16-rank_n_fold_map-" + shape.stem + "-" +
           boundary.suffix + ".mlir")
              .str();
      SCOPED_TRACE(filename);
      auto source = fixtureSource(filename);
      ASSERT_TRUE(source);
      for (mlir::shuttle::NumericalPolicy policy :
           {mlir::shuttle::NumericalPolicy::SourceOrdered,
            mlir::shuttle::NumericalPolicy::Fast}) {
        auto built = buildGpuContractFixture(*source, policy);
        ASSERT_TRUE(built);
        auto bytes =
            mlir::shuttle::serializeGpuExecutableBundle(*built->module);
        ASSERT_TRUE(mlir::succeeded(bytes));
        auto decoded = decodeInlineSchema2(*bytes);
        ASSERT_TRUE(decoded);
        auto loaded = mlir::shuttle::GpuExecutable::Load(*bytes);
        ASSERT_TRUE(loaded.ok()) << loaded.status();
        EXPECT_EQ(decoded->launches.size(), boundary.launches);
        EXPECT_EQ(decoded->externalSlots, boundary.externalSlots);
        ASSERT_EQ((*loaded)->externalBindings().size(),
                  boundary.externalSlots);
        for (const auto &binding : (*loaded)->externalBindings())
          EXPECT_EQ(binding.requiredBytes, shape.publicBytes);
        int64_t inputSlots = 0;
        int64_t resultSlots = 0;
        for (size_t slot = 0; slot < decoded->slotDimensions.size(); ++slot) {
          if (decoded->slotDimensions[slot] ==
                  llvm::ArrayRef<int64_t>(shape.inputShape) &&
              decoded->slotRequiredBytes[slot] == shape.inputBytes) {
            ++inputSlots;
            EXPECT_EQ(decoded->slotStrides[slot],
                      llvm::ArrayRef<int64_t>(shape.inputStrides));
            int64_t remainingOutput = shape.sampleOutput;
            int64_t decodedByte =
                shape.sampleLeaf * decoded->slotStrides[slot].back();
            for (int64_t axis = shape.resultShape.size(); axis > 0; --axis) {
              const int64_t coordinate =
                  remainingOutput % shape.resultShape[axis - 1];
              remainingOutput /= shape.resultShape[axis - 1];
              decodedByte += coordinate * decoded->slotStrides[slot][axis - 1];
            }
            EXPECT_EQ(remainingOutput, 0);
            EXPECT_EQ(decodedByte, shape.sampleInputByte);
          }
          if (decoded->slotDimensions[slot] ==
                  llvm::ArrayRef<int64_t>(shape.resultShape) &&
              decoded->slotRequiredBytes[slot] == shape.resultBytes) {
            ++resultSlots;
            EXPECT_EQ(decoded->slotStrides[slot],
                      llvm::ArrayRef<int64_t>(shape.resultStrides));
            int64_t remainingOutput = shape.sampleOutput;
            int64_t decodedByte = 0;
            for (int64_t axis = shape.resultShape.size(); axis > 0; --axis) {
              const int64_t coordinate =
                  remainingOutput % shape.resultShape[axis - 1];
              remainingOutput /= shape.resultShape[axis - 1];
              decodedByte += coordinate * decoded->slotStrides[slot][axis - 1];
            }
            EXPECT_EQ(remainingOutput, 0);
            EXPECT_EQ(decodedByte, shape.sampleOutputByte);
          }
          if (decoded->slotStorage[slot] == 0) {
            EXPECT_EQ(decoded->slotDimensions[slot],
                      llvm::ArrayRef<int64_t>(shape.inputShape));
            EXPECT_EQ(decoded->slotStrides[slot],
                      llvm::ArrayRef<int64_t>(shape.publicStrides));
          }
        }
        EXPECT_GE(inputSlots, boundary.folds);
        EXPECT_GE(resultSlots, boundary.folds);

        int64_t foldLaunches = 0;
        for (const auto &launch : decoded->launches) {
          if (launch.dynamicSharedMemoryBytes == 0)
            continue;
          ++foldLaunches;
          EXPECT_EQ(launch.grid,
                    (std::array<uint64_t, 3>{
                        static_cast<uint64_t>(shape.outputCount), 1, 1}));
          EXPECT_EQ(launch.block,
                    (std::array<uint64_t, 3>{shape.threads, 1, 1}));
          EXPECT_EQ(launch.dynamicSharedMemoryBytes, shape.scratch);
          llvm::StringRef ptx(
              reinterpret_cast<const char *>(decoded->code.data() +
                                             launch.codeOffset),
              launch.codeLength);
          EXPECT_TRUE(ptx.contains(
              ("setp.ge.u32 %p0, %r1, " + llvm::Twine(shape.outputCount) +
               ";")
                  .str()));
          EXPECT_TRUE(ptx.contains(
              ("mad.lo.u64 %rd3, %rd2, " +
               llvm::Twine(shape.reductionExtent) + ", %rd1;")
                  .str()));
          EXPECT_TRUE(
              ptx.contains("mad.lo.u64 %rd3, %rd3, 4, %rd0;"));
          EXPECT_TRUE(ptx.contains("mul.lo.u64 %rd9, %rd9, 4;"));
          EXPECT_EQ(ptx.count("ld.global.f32"),
                    static_cast<size_t>(
                        (shape.reductionExtent + shape.threads - 1) /
                            shape.threads +
                        1));
          EXPECT_FALSE(ptx.contains("ld.global.u16"));
        }
        EXPECT_EQ(foldLaunches, boundary.folds);
      }
    }
  }
}

TEST(GpuTransportV2Test,
     Abi18SingletonBroadcastClosesEveryBoundaryAndPhysicalAddress) {
  struct Shape {
    llvm::StringRef stem;
    std::vector<int64_t> fullShape;
    std::vector<int64_t> singletonShape;
    std::vector<int64_t> fullStrides;
    std::vector<int64_t> singletonStrides;
    std::vector<int64_t> publicStrides;
    std::vector<int64_t> sampleCoordinates;
    int64_t reductionExtent;
    uint64_t mapGrid;
    uint64_t foldGrid;
    uint64_t foldThreads;
    uint32_t foldScratch;
    int64_t fullBytes;
    int64_t singletonBytes;
    int64_t publicBytes;
    int64_t sampleInputByte;
    int64_t sampleOutputByte;
  };
  struct Boundary {
    llvm::StringRef suffix;
    int64_t launches;
    int64_t fullFolds;
    int64_t singletonFolds;
    int64_t expansions;
    int64_t externalSlots;
  };
  const Shape shapes[] = {
      {"2x7x19",
       {2, 7, 19},
       {2, 7, 1},
       {532, 76, 4},
       {28, 4, 4},
       {266, 38, 2},
       {1, 5, 18},
       19,
       2,
       14,
       32,
       128,
       1064,
       56,
       532,
       48,
       984},
      {"3x2x5x11x263",
       {3, 2, 5, 11, 263},
       {3, 2, 5, 11, 1},
       {115720, 57860, 11572, 1052, 4},
       {440, 220, 44, 4, 4},
       {57860, 28930, 5786, 526, 2},
       {2, 1, 4, 10, 262},
       263,
       340,
       330,
       256,
       1024,
       347160,
       1320,
       173580,
       1316,
       347156},
  };
  constexpr Boundary boundaries[] = {
      {"forward", 7, 1, 0, 1, 2},
      {"backward", 16, 2, 1, 1, 3},
      {"composed", 19, 2, 1, 2, 4},
  };

  for (const Shape &shape : shapes) {
    for (const Boundary &boundary : boundaries) {
      std::string filename =
          ("jax-0.10.1-bf16-singleton_broadcast-" + shape.stem + "-" +
           boundary.suffix + ".mlir")
              .str();
      SCOPED_TRACE(filename);
      auto source = fixtureSource(filename);
      ASSERT_TRUE(source);
      std::optional<llvm::SmallVector<uint8_t>> sourceTransport;
      for (mlir::shuttle::NumericalPolicy policy :
           {mlir::shuttle::NumericalPolicy::SourceOrdered,
            mlir::shuttle::NumericalPolicy::Fast}) {
        auto built = buildGpuContractFixture(*source, policy);
        ASSERT_TRUE(built);
        auto bytes =
            mlir::shuttle::serializeGpuExecutableBundle(*built->module);
        ASSERT_TRUE(mlir::succeeded(bytes));
        auto decoded = decodeInlineSchema2(*bytes);
        ASSERT_TRUE(decoded);
        auto loaded = mlir::shuttle::GpuExecutable::Load(*bytes);
        ASSERT_TRUE(loaded.ok()) << loaded.status();
        EXPECT_EQ(decoded->policy,
                  policy == mlir::shuttle::NumericalPolicy::SourceOrdered ? 0
                                                                          : 1);
        EXPECT_EQ(decoded->launches.size(), boundary.launches);
        EXPECT_EQ(decoded->externalSlots, boundary.externalSlots);
        ASSERT_EQ((*loaded)->externalBindings().size(),
                  boundary.externalSlots);
        for (const auto &binding : (*loaded)->externalBindings()) {
          EXPECT_EQ(binding.requiredBytes, shape.publicBytes);
        }

        int64_t expansionLaunches = 0;
        int64_t fullFoldLaunches = 0;
        int64_t singletonFoldLaunches = 0;
        for (const auto &launch : decoded->launches) {
          llvm::StringRef ptx(
              reinterpret_cast<const char *>(decoded->code.data() +
                                             launch.codeOffset),
              launch.codeLength);
          bool isExpansion = launch.inputSlots.size() == 1 &&
                             launch.outputSlots.size() == 1;
          if (isExpansion) {
            int64_t input = launch.inputSlots.front();
            int64_t output = launch.outputSlots.front();
            isExpansion =
                decoded->slotDimensions[input] ==
                    llvm::ArrayRef<int64_t>(shape.singletonShape) &&
                decoded->slotDimensions[output] ==
                    llvm::ArrayRef<int64_t>(shape.fullShape);
          }
          if (isExpansion) {
            ++expansionLaunches;
            const int64_t input = launch.inputSlots.front();
            const int64_t output = launch.outputSlots.front();
            EXPECT_EQ(decoded->slotDimensions[input],
                      llvm::ArrayRef<int64_t>(shape.singletonShape));
            EXPECT_EQ(decoded->slotStrides[input],
                      llvm::ArrayRef<int64_t>(shape.singletonStrides));
            EXPECT_EQ(decoded->slotRequiredBytes[input], shape.singletonBytes);
            EXPECT_EQ(decoded->slotDimensions[output],
                      llvm::ArrayRef<int64_t>(shape.fullShape));
            EXPECT_EQ(decoded->slotStrides[output],
                      llvm::ArrayRef<int64_t>(shape.fullStrides));
            EXPECT_EQ(decoded->slotRequiredBytes[output], shape.fullBytes);
            EXPECT_EQ(launch.grid,
                      (std::array<uint64_t, 3>{shape.mapGrid, 1, 1}));
            EXPECT_EQ(launch.block,
                      (std::array<uint64_t, 3>{256, 1, 1}));
            EXPECT_EQ(launch.dynamicSharedMemoryBytes, 0);
            llvm::SmallVector<llvm::StringRef> lines;
            ptx.split(lines, '\n');
            int64_t divisionsByReductionExtent = 0;
            int64_t coordinateDivisionsByReductionExtent = 0;
            std::string suffix =
                (", " + llvm::Twine(shape.reductionExtent) + ";").str();
            for (llvm::StringRef line : lines) {
              if (!line.starts_with("  div.u64 ") || !line.ends_with(suffix)) {
                continue;
              }
              ++divisionsByReductionExtent;
              if (line.contains(", %rd0, ")) {
                ++coordinateDivisionsByReductionExtent;
              }
            }
            EXPECT_EQ(divisionsByReductionExtent, 2);
            EXPECT_EQ(coordinateDivisionsByReductionExtent, 1);

            int64_t inputByte = 0;
            int64_t outputByte = 0;
            for (size_t axis = 0; axis < shape.sampleCoordinates.size(); ++axis) {
              const int64_t inputCoordinate =
                  axis + 1 == shape.sampleCoordinates.size()
                      ? 0
                      : shape.sampleCoordinates[axis];
              inputByte += inputCoordinate * shape.singletonStrides[axis];
              outputByte +=
                  shape.sampleCoordinates[axis] * shape.fullStrides[axis];
            }
            EXPECT_EQ(inputByte, shape.sampleInputByte);
            EXPECT_EQ(outputByte, shape.sampleOutputByte);
          }
          if (launch.dynamicSharedMemoryBytes != 0) {
            EXPECT_EQ(launch.grid,
                      (std::array<uint64_t, 3>{shape.foldGrid, 1, 1}));
            ASSERT_EQ(launch.inputSlots.size(), 2);
            const int64_t input = launch.inputSlots.front();
            const int64_t initializer = launch.inputSlots.back();
            EXPECT_TRUE(decoded->slotDimensions[initializer].empty());
            EXPECT_TRUE(decoded->slotStrides[initializer].empty());
            EXPECT_EQ(decoded->slotRequiredBytes[initializer], 4);
            if (decoded->slotDimensions[input] ==
                llvm::ArrayRef<int64_t>(shape.fullShape)) {
              ++fullFoldLaunches;
              EXPECT_EQ(launch.block,
                        (std::array<uint64_t, 3>{shape.foldThreads, 1, 1}));
              EXPECT_EQ(launch.dynamicSharedMemoryBytes, shape.foldScratch);
            } else {
              ++singletonFoldLaunches;
              EXPECT_EQ(decoded->slotDimensions[input],
                        llvm::ArrayRef<int64_t>(shape.singletonShape));
              EXPECT_EQ(launch.block,
                        (std::array<uint64_t, 3>{32, 1, 1}));
              EXPECT_EQ(launch.dynamicSharedMemoryBytes, 128);
            }
          }
        }
        EXPECT_EQ(expansionLaunches, boundary.expansions);
        EXPECT_EQ(fullFoldLaunches, boundary.fullFolds);
        EXPECT_EQ(singletonFoldLaunches, boundary.singletonFolds);

        for (size_t slot = 0; slot < decoded->slotDimensions.size(); ++slot) {
          if (decoded->slotStorage[slot] != 0) {
            continue;
          }
          EXPECT_EQ(decoded->slotDimensions[slot],
                    llvm::ArrayRef<int64_t>(shape.fullShape));
          EXPECT_EQ(decoded->slotStrides[slot],
                    llvm::ArrayRef<int64_t>(shape.publicStrides));
          EXPECT_EQ(decoded->slotRequiredBytes[slot], shape.publicBytes);
        }

        if (policy == mlir::shuttle::NumericalPolicy::SourceOrdered) {
          sourceTransport = *bytes;
        } else {
          ASSERT_TRUE(sourceTransport);
          EXPECT_NE(*bytes, *sourceTransport);
        }
      }
    }
  }
}

TEST(GpuTransportV2Test,
     Abi18SingletonBroadcastTransportMutationsFailBothDecoders) {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-composed.mlir");
  ASSERT_TRUE(source);
  auto built = buildGpuContractFixture(
      *source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(built);
  auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  auto decoded = decodeInlineSchema2(*bytes);
  ASSERT_TRUE(decoded);

  auto expectRejected = [](llvm::ArrayRef<uint8_t> mutation) {
    EXPECT_FALSE(decodeInlineSchema2(mutation));
    EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(mutation).ok());
  };
  llvm::SmallVector<uint8_t> codeDigest(*bytes);
  ASSERT_FALSE(decoded->launches.empty());
  codeDigest[decoded->codeLengthOffset + sizeof(uint64_t) +
             decoded->launches.front().codeOffset] ^= 1;
  expectRejected(codeDigest);

  llvm::SmallVector<uint8_t> requiredBytes(*bytes);
  const int64_t temporary = llvm::find(decoded->slotStorage, 1) -
                            decoded->slotStorage.begin();
  ASSERT_LT(temporary, decoded->slotStorage.size());
  writeU64(requiredBytes, decoded->slotRequiredOffsets[temporary],
           decoded->slotRequiredBytes[temporary] + 4);
  rehashInvocationAndBundle(requiredBytes, *decoded);
  expectRejected(requiredBytes);

  auto reuse = overlappingTemporaryReuse(*bytes, *decoded);
  ASSERT_FALSE(reuse.empty());
  expectRejected(reuse);

  auto futureResult = futureProducedResultInput(*bytes, *decoded);
  ASSERT_FALSE(futureResult.empty());
  expectRejected(futureResult);

  auto oversizedGrid = oversizedLaunchGrid(*bytes, *decoded);
  expectRejected(oversizedGrid);
}

TEST(GpuTransportV2Test,
     Abi20ProjectedParameterBroadcastClosesGeometryAndAddresses) {
  struct Shape {
    llvm::StringRef stem;
    std::vector<int64_t> full;
    std::vector<int64_t> inserted;
    std::vector<int64_t> weight;
    std::vector<int64_t> coordinates;
    std::vector<int64_t> divisors;
    uint64_t fullGrid;
    uint64_t parameterGrid;
    int64_t fullBytes;
    int64_t insertedBytes;
    int64_t weightBytes;
    int64_t sampleInputByte;
    int64_t sampleOutputByte;
  };
  const Shape shapes[] = {
      {"2x7x19", {2, 7, 19}, {1, 1, 19}, {19}, {1, 6, 18}, {2, 7},
       2, 1, 1064, 76, 38, 72, 1060},
      {"3x2x5x11x263", {3, 2, 5, 11, 263}, {1, 1, 1, 1, 263},
       {263}, {2, 1, 4, 10, 262}, {3, 2, 5, 11}, 340, 2, 347160,
       1052, 526, 1048, 347156},
  };
  for (const Shape &shape : shapes) {
    auto source = fixtureSource(
        ("jax-0.10.1-bf16-projected_parameter_broadcast-" + shape.stem +
         "-forward.mlir")
            .str());
    ASSERT_TRUE(source);
    std::optional<llvm::SmallVector<uint8_t>> sourceTransport;
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      auto built = buildGpuContractFixture(*source, policy);
      ASSERT_TRUE(built);
      auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
      ASSERT_TRUE(mlir::succeeded(bytes));
      auto decoded = decodeInlineSchema2(*bytes);
      ASSERT_TRUE(decoded);
      auto loaded = mlir::shuttle::GpuExecutable::Load(*bytes);
      ASSERT_TRUE(loaded.ok()) << loaded.status();
      EXPECT_EQ(decoded->launches.size(), 19);
      EXPECT_EQ(decoded->slots, 21);
      EXPECT_EQ(decoded->externalSlots, 3);
      EXPECT_EQ((*loaded)->externalBindings().size(), 3);
      int64_t projectedLaunches = 0;
      for (const auto &launch : decoded->launches) {
        if (launch.inputSlots.size() != 1 || launch.outputSlots.size() != 1)
          continue;
        int64_t input = launch.inputSlots.front();
        int64_t output = launch.outputSlots.front();
        if (decoded->slotDimensions[input] !=
                llvm::ArrayRef<int64_t>(shape.inserted) ||
            decoded->slotDimensions[output] !=
                llvm::ArrayRef<int64_t>(shape.full))
          continue;
        ++projectedLaunches;
        EXPECT_EQ(decoded->slotRequiredBytes[input], shape.insertedBytes);
        EXPECT_EQ(decoded->slotRequiredBytes[output], shape.fullBytes);
        EXPECT_EQ(launch.grid,
                  (std::array<uint64_t, 3>{shape.fullGrid, 1, 1}));
        EXPECT_EQ(launch.block,
                  (std::array<uint64_t, 3>{256, 1, 1}));
        llvm::StringRef ptx(
            reinterpret_cast<const char *>(decoded->code.data() +
                                           launch.codeOffset),
            launch.codeLength);
        int64_t elementStride = 1;
        llvm::SmallVector<int64_t> elementStrides(shape.full.size());
        for (size_t axis = shape.full.size(); axis > 0; --axis) {
          elementStrides[axis - 1] = elementStride;
          elementStride *= shape.full[axis - 1];
        }
        const int64_t linearRegister = 8 + shape.full.size();
        for (size_t axis = 0; axis < shape.full.size(); ++axis) {
          EXPECT_TRUE(ptx.contains(
              ("div.u64 %rd" + llvm::Twine(8 + axis) + ", %rd0, " +
               llvm::Twine(elementStrides[axis]) + ";")
                  .str()));
          int64_t mappedRegister = 8 + axis;
          if (axis < shape.divisors.size()) {
            mappedRegister = linearRegister + 1 + axis;
            EXPECT_TRUE(ptx.contains(
                ("div.u64 %rd" + llvm::Twine(mappedRegister) + ", %rd" +
                 llvm::Twine(8 + axis) + ", " +
                 llvm::Twine(shape.divisors[axis]) + ";")
                    .str()));
          }
          EXPECT_TRUE(ptx.contains(
              ("mad.lo.u64 %rd" + llvm::Twine(linearRegister) + ", %rd" +
               llvm::Twine(linearRegister) + ", " +
               llvm::Twine(shape.inserted[axis]) + ", %rd" +
               llvm::Twine(mappedRegister) + ";")
                  .str()));
        }
        const int64_t inputPointer = 8 + 2 * shape.full.size();
        EXPECT_TRUE(ptx.contains(
            ("mad.lo.u64 %rd" + llvm::Twine(inputPointer) + ", %rd" +
             llvm::Twine(linearRegister) + ", 4, %rd" +
             llvm::Twine(inputPointer) + ";")
                .str()));
        auto loaded = projectedF32LoadedRegister(ptx, inputPointer);
        ASSERT_TRUE(loaded);
        llvm::SmallVector<llvm::StringRef> ptxLines;
        ptx.split(ptxLines, '\n');
        int64_t matchingStores = llvm::count_if(
            ptxLines, [&](llvm::StringRef line) {
              return line.contains("st.global.f32") &&
                     line.contains(*loaded);
            });
        EXPECT_EQ(matchingStores, 1);
        const int64_t outputPointer = inputPointer + 1;
        EXPECT_TRUE(ptx.contains(
            ("mad.lo.u64 %rd" + llvm::Twine(outputPointer) +
             ", %rd0, 4, %rd" + llvm::Twine(outputPointer) + ";")
                .str()));
        EXPECT_TRUE(ptx.contains(
            ("st.global.f32 [%rd" + llvm::Twine(outputPointer) + "], " +
             *loaded + ";")
                .str()));
        EXPECT_FALSE(ptx.contains("st.global.u16"));
        int64_t inputByte = 0;
        int64_t outputByte = 0;
        for (size_t axis = 0; axis < shape.full.size(); ++axis) {
          int64_t inputCoordinate =
              axis < shape.divisors.size()
                  ? shape.coordinates[axis] / shape.divisors[axis]
                  : shape.coordinates[axis];
          inputByte += inputCoordinate * decoded->slotStrides[input][axis];
          outputByte +=
              shape.coordinates[axis] * decoded->slotStrides[output][axis];
        }
        EXPECT_EQ(inputByte, shape.sampleInputByte);
        EXPECT_EQ(outputByte, shape.sampleOutputByte);
      }
      EXPECT_EQ(projectedLaunches, 1);
      int64_t publicWeightBindings = 0;
      int64_t publicFullBindings = 0;
      for (size_t slot = 0; slot < decoded->slotDimensions.size(); ++slot) {
        if (decoded->slotStorage[slot] != 0)
          continue;
        if (decoded->slotDimensions[slot] ==
            llvm::ArrayRef<int64_t>(shape.full)) {
          ++publicFullBindings;
          EXPECT_EQ(decoded->slotRequiredBytes[slot], shape.fullBytes / 2);
        }
        if (
            decoded->slotDimensions[slot] ==
                llvm::ArrayRef<int64_t>(shape.weight)) {
          ++publicWeightBindings;
          EXPECT_EQ(decoded->slotRequiredBytes[slot], shape.weightBytes);
        }
      }
      EXPECT_EQ(publicFullBindings, 2);
      EXPECT_EQ(publicWeightBindings, 1);
      if (policy == mlir::shuttle::NumericalPolicy::SourceOrdered)
        sourceTransport = *bytes;
      else {
        ASSERT_TRUE(sourceTransport);
        EXPECT_NE(*bytes, *sourceTransport);
      }
    }
  }
}

TEST(GpuTransportV2Test,
     Abi20ProjectedParameterTransportMutationsAreBehaviorSensitive) {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-projected_parameter_broadcast-"
      "3x2x5x11x263-forward.mlir");
  ASSERT_TRUE(source);
  auto built = buildGpuContractFixture(
      *source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(built);
  auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  auto decoded = decodeInlineSchema2(*bytes);
  ASSERT_TRUE(decoded);

  const std::vector<int64_t> inserted{1, 1, 1, 1, 263};
  const std::vector<int64_t> full{3, 2, 5, 11, 263};
  int64_t projectedEntry = -1;
  int64_t projectedEntries = 0;
  std::string projectedLoad;
  for (const auto &launch : decoded->launches) {
    if (launch.inputSlots.size() != 1 || launch.outputSlots.size() != 1 ||
        decoded->slotDimensions[launch.inputSlots.front()] !=
            llvm::ArrayRef<int64_t>(inserted) ||
        decoded->slotDimensions[launch.outputSlots.front()] !=
            llvm::ArrayRef<int64_t>(full))
      continue;
    ++projectedEntries;
    llvm::StringRef ptx(
        reinterpret_cast<const char *>(decoded->code.data() +
                                       launch.codeOffset),
        launch.codeLength);
    auto loaded = projectedF32LoadedRegister(ptx, /*inputPointer=*/18);
    ASSERT_TRUE(loaded);
    projectedEntry = launch.taskOrdinal;
    projectedLoad = "ld.global.f32 " + *loaded + ", [%rd18];";
  }
  ASSERT_EQ(projectedEntries, 1);
  ASSERT_GE(projectedEntry, 0);
  std::string wrongProjectedLoad = projectedLoad;
  size_t pointer = wrongProjectedLoad.find("%rd18");
  ASSERT_NE(pointer, std::string::npos);
  wrongProjectedLoad.replace(pointer, llvm::StringRef("%rd18").size(),
                             "%rd17");
  auto wrongPointer = replaceEntryCodeAndRehash(
      *bytes, *decoded, projectedEntry, projectedLoad,
      wrongProjectedLoad);
  ASSERT_FALSE(wrongPointer.empty());
  auto wrongPointerDecoded = decodeInlineSchema2(wrongPointer);
  ASSERT_TRUE(wrongPointerDecoded);
  auto wrongPointerLoaded = mlir::shuttle::GpuExecutable::Load(wrongPointer);
  ASSERT_TRUE(wrongPointerLoaded.ok()) << wrongPointerLoaded.status();
  const auto &wrongLaunch = wrongPointerDecoded->launches[projectedEntry];
  llvm::StringRef wrongPtx(
      reinterpret_cast<const char *>(wrongPointerDecoded->code.data() +
                                     wrongLaunch.codeOffset),
      wrongLaunch.codeLength);
  EXPECT_FALSE(projectedF32LoadedRegister(wrongPtx, /*inputPointer=*/18));
  EXPECT_TRUE(projectedF32LoadedRegister(wrongPtx, /*inputPointer=*/17));

  auto expectRejected = [](llvm::ArrayRef<uint8_t> mutation) {
    EXPECT_FALSE(decodeInlineSchema2(mutation));
    EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(mutation).ok());
  };
  llvm::SmallVector<uint8_t> codeDigest(*bytes);
  codeDigest[decoded->codeLengthOffset + sizeof(uint64_t) +
             decoded->launches.back().codeOffset] ^= 1;
  expectRejected(codeDigest);
  auto reuse = overlappingTemporaryReuse(*bytes, *decoded);
  ASSERT_FALSE(reuse.empty());
  expectRejected(reuse);
  auto futureResult = futureProducedResultInput(*bytes, *decoded);
  ASSERT_FALSE(futureResult.empty());
  expectRejected(futureResult);
  expectRejected(oversizedLaunchGrid(*bytes, *decoded));
}

TEST(GpuTransportV2Test,
     Abi21FullPrefixFoldClosesGeometryAddressingAndTransport) {
  struct Shape {
    llvm::StringRef stem;
    std::vector<int64_t> full;
    std::vector<int64_t> singletonPrefix;
    std::vector<int64_t> feature;
    int64_t reductionElements;
    int64_t featureElements;
    uint64_t threads;
    uint64_t serialTiles;
    uint64_t scratchBytes;
    int64_t finalInputByte;
    int64_t fullPublicBytes;
    int64_t featurePublicBytes;
    int64_t backwardRawTemporaryBytes;
    int64_t composedRawTemporaryBytes;
  };
  const Shape shapes[] = {
      {"2x7x19", {2, 7, 19}, {1, 1, 19}, {19}, 14, 19, 32, 1,
       128, 1060, 532, 38, -1, -1},
      {"3x2x5x11x263", {3, 2, 5, 11, 263}, {1, 1, 1, 1, 263},
       {263}, 330, 263, 256, 2, 1024, 347156, 173580, 526,
       5580976, 6275296},
  };
  struct Boundary {
    llvm::StringRef name;
    int64_t tasks;
    int64_t slots;
    int64_t bindings;
  };
  constexpr Boundary boundaries[] = {
      {"backward", 48, 51, 5},
      {"composed", 51, 54, 6},
  };
  for (const Shape &shape : shapes) {
    for (const Boundary &boundary : boundaries) {
      auto source = fixtureSource(
          ("jax-0.10.1-bf16-projected_parameter_broadcast-" + shape.stem +
           "-" + boundary.name + ".mlir")
              .str());
      ASSERT_TRUE(source);
      std::optional<llvm::SmallVector<uint8_t>> sourceTransport;
      for (mlir::shuttle::NumericalPolicy policy :
           {mlir::shuttle::NumericalPolicy::SourceOrdered,
            mlir::shuttle::NumericalPolicy::Fast}) {
        SCOPED_TRACE((shape.stem + "/" + boundary.name).str());
        auto built = buildGpuContractFixture(*source, policy);
        ASSERT_TRUE(built);
        auto bytes =
            mlir::shuttle::serializeGpuExecutableBundle(*built->module);
        ASSERT_TRUE(mlir::succeeded(bytes));
        auto decoded = decodeInlineSchema2(*bytes);
        ASSERT_TRUE(decoded);
        auto loaded = mlir::shuttle::GpuExecutable::Load(*bytes);
        ASSERT_TRUE(loaded.ok()) << loaded.status();
        EXPECT_EQ(decoded->launches.size(), boundary.tasks);
        EXPECT_EQ(decoded->slots, boundary.slots);
        EXPECT_EQ(decoded->externalSlots, boundary.bindings);
        EXPECT_EQ((*loaded)->externalBindings().size(), boundary.bindings);

        const std::vector<int64_t> expectedBindingBytes =
            boundary.name == "backward"
                ? std::vector<int64_t>{shape.fullPublicBytes,
                                       shape.featurePublicBytes,
                                       shape.fullPublicBytes,
                                       shape.featurePublicBytes,
                                       shape.fullPublicBytes}
                : std::vector<int64_t>{shape.fullPublicBytes,
                                       shape.featurePublicBytes,
                                       shape.fullPublicBytes,
                                       shape.fullPublicBytes,
                                       shape.featurePublicBytes,
                                       shape.fullPublicBytes};
        ASSERT_EQ((*loaded)->externalBindings().size(),
                  expectedBindingBytes.size());
        std::vector<mlir::shuttle::GpuExternalBinding> structuralBindings =
            externalBindingsBySlot(**loaded);
        for (auto [binding, expectedBytes] : llvm::zip_equal(
                 structuralBindings, expectedBindingBytes))
          EXPECT_EQ(binding.requiredBytes, expectedBytes);

        int64_t prefixFolds = 0;
        int64_t rowFolds = 0;
        int64_t fullPrefixFolds = 0;
        for (const auto &launch : decoded->launches) {
          if (launch.dynamicSharedMemoryBytes == 0)
            continue;
          ASSERT_EQ(launch.outputSlots.size(), 1u);
          const int64_t output = launch.outputSlots.front();
          if (decoded->slotDimensions[output] !=
              llvm::ArrayRef<int64_t>(shape.feature)) {
            ++rowFolds;
            continue;
          }
          ++prefixFolds;
          int64_t tensorInput = -1;
          for (int64_t input : launch.inputSlots) {
            if (decoded->slotDimensions[input] ==
                    llvm::ArrayRef<int64_t>(shape.full) ||
                decoded->slotDimensions[input] ==
                    llvm::ArrayRef<int64_t>(shape.singletonPrefix)) {
              ASSERT_EQ(tensorInput, -1);
              tensorInput = input;
            }
          }
          ASSERT_GE(tensorInput, 0);
          if (decoded->slotDimensions[tensorInput] !=
              llvm::ArrayRef<int64_t>(shape.full))
            continue;
          ++fullPrefixFolds;
          EXPECT_EQ(launch.grid,
                    (std::array<uint64_t, 3>{
                        static_cast<uint64_t>(shape.featureElements), 1, 1}));
          EXPECT_EQ(launch.block,
                    (std::array<uint64_t, 3>{shape.threads, 1, 1}));
          EXPECT_EQ(launch.dynamicSharedMemoryBytes, shape.scratchBytes);
          llvm::StringRef ptx(
              reinterpret_cast<const char *>(decoded->code.data() +
                                               launch.codeOffset),
              launch.codeLength);
          EXPECT_TRUE(ptx.contains(
              ("mul.lo.u32 %r4, %r0, " + llvm::Twine(shape.serialTiles) +
               ";")
                  .str()));
          EXPECT_TRUE(ptx.contains(
              ("setp.ge.u32 %p1, %r4, " +
               llvm::Twine(shape.reductionElements) + ";")
                  .str()));
          EXPECT_TRUE(ptx.contains(
              ("mad.lo.u64 %rd3, %rd1, " +
               llvm::Twine(shape.featureElements) + ", %rd2;\n"
                "  mad.lo.u64 %rd3, %rd3, 4, %rd0;\n"
                "  ld.global.f32 %f0, [%rd3];")
                  .str()));
          EXPECT_TRUE(ptx.contains(
              "mul.lo.u64 %rd9, %rd9, 4;\n"
              "  add.u64 %rd8, %rd8, %rd9;\n"
              "  st.global.f32 [%rd8], %f37;"));
          int64_t finalByte = 0;
          for (size_t axis = 0; axis < shape.full.size(); ++axis)
            finalByte +=
                (shape.full[axis] - 1) * decoded->slotStrides[tensorInput][axis];
          EXPECT_EQ(finalByte, shape.finalInputByte);
          EXPECT_EQ((shape.reductionElements - 1) * shape.featureElements * 4 +
                        (shape.featureElements - 1) * 4,
                    shape.finalInputByte);
        }
        EXPECT_EQ(prefixFolds, 2);
        EXPECT_EQ(rowFolds, 3);
        EXPECT_EQ(fullPrefixFolds, 1);

        int64_t rawTemporaryBytes = 0;
        for (size_t slot = 0; slot < decoded->slotStorage.size(); ++slot)
          if (decoded->slotStorage[slot] == 1)
            rawTemporaryBytes += decoded->slotRequiredBytes[slot];
        int64_t expectedRaw = boundary.name == "backward"
                                  ? shape.backwardRawTemporaryBytes
                                  : shape.composedRawTemporaryBytes;
        if (expectedRaw >= 0)
          EXPECT_EQ(rawTemporaryBytes, expectedRaw);
        EXPECT_LE(decoded->temporaryBytes, 256 * 1024 * 1024);
        EXPECT_LE(bytes->size(), 16 * 1024 * 1024);
        if (policy == mlir::shuttle::NumericalPolicy::SourceOrdered)
          sourceTransport = *bytes;
        else {
          ASSERT_TRUE(sourceTransport);
          EXPECT_NE(*bytes, *sourceTransport);
        }
      }
    }
  }
}

TEST(GpuTransportV2Test,
     Abi21FullPrefixFoldTransportMutationsAreBehaviorSensitive) {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-projected_parameter_broadcast-"
      "3x2x5x11x263-backward.mlir");
  ASSERT_TRUE(source);
  auto built = buildGpuContractFixture(
      *source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(built);
  auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  auto decoded = decodeInlineSchema2(*bytes);
  ASSERT_TRUE(decoded);
  const std::vector<int64_t> full{3, 2, 5, 11, 263};
  const std::vector<int64_t> feature{263};
  int64_t prefixEntry = -1;
  for (const auto &launch : decoded->launches) {
    if (launch.dynamicSharedMemoryBytes != 1024 ||
        launch.outputSlots.size() != 1 ||
        decoded->slotDimensions[launch.outputSlots.front()] !=
            llvm::ArrayRef<int64_t>(feature))
      continue;
    for (int64_t input : launch.inputSlots)
      if (decoded->slotDimensions[input] == llvm::ArrayRef<int64_t>(full))
        prefixEntry = launch.taskOrdinal;
  }
  ASSERT_GE(prefixEntry, 0);
  const auto &prefixLaunch = decoded->launches[prefixEntry];
  constexpr llvm::StringLiteral address =
      "mad.lo.u64 %rd3, %rd1, 263, %rd2;\n"
      "  mad.lo.u64 %rd3, %rd3, 4, %rd0;\n"
      "  ld.global.f32 %f0, [%rd3];";
  constexpr llvm::StringLiteral deadAddress =
      "mad.lo.u64 %rd6, %rd1, 263, %rd2;\n"
      "  mad.lo.u64 %rd3, %rd3, 4, %rd0;\n"
      "  ld.global.f32 %f0, [%rd3];";
  auto dead = replaceEntryCodeAndRehash(*bytes, *decoded, prefixEntry, address,
                                        deadAddress);
  ASSERT_FALSE(dead.empty());
  auto deadDecoded = decodeInlineSchema2(dead);
  ASSERT_TRUE(deadDecoded);
  ASSERT_TRUE(mlir::shuttle::GpuExecutable::Load(dead).ok());
  const auto &deadLaunch = deadDecoded->launches[prefixEntry];
  llvm::StringRef deadPtx(
      reinterpret_cast<const char *>(deadDecoded->code.data() +
                                     deadLaunch.codeOffset),
      deadLaunch.codeLength);
  EXPECT_FALSE(deadPtx.contains(address));
  EXPECT_TRUE(deadPtx.contains(deadAddress));

  llvm::SmallVector<uint8_t> wrongGrid(*bytes);
  writeU32(wrongGrid, decoded->launchGridOffsets[prefixEntry], 264);
  rehashDeviceAndBundle(wrongGrid, *decoded);
  auto wrongGridDecoded = decodeInlineSchema2(wrongGrid);
  ASSERT_TRUE(wrongGridDecoded);
  ASSERT_TRUE(mlir::shuttle::GpuExecutable::Load(wrongGrid).ok());
  EXPECT_EQ(wrongGridDecoded->launches[prefixEntry].grid,
            (std::array<uint64_t, 3>{264, 1, 1}));
  EXPECT_NE(wrongGridDecoded->launches[prefixEntry].grid, prefixLaunch.grid);

  auto expectRejected = [](llvm::ArrayRef<uint8_t> mutation) {
    EXPECT_FALSE(decodeInlineSchema2(mutation));
    EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(mutation).ok());
  };
  auto reuse = overlappingTemporaryReuse(*bytes, *decoded);
  ASSERT_FALSE(reuse.empty());
  expectRejected(reuse);
  auto futureInput = futureProducedResultInput(*bytes, *decoded);
  ASSERT_FALSE(futureInput.empty());
  expectRejected(futureInput);
}

TEST(GpuTransportV2Test,
     Abi14OrdinaryJaxMatmulVjpHasClosedMultiContractTransport) {
  auto source =
      fixtureSource("jax-0.10.1-bf16-matmul-vjp-17x33x19-backward.mlir");
  ASSERT_TRUE(source);
  auto sourceOrdered = buildGpuContractFixture(
      *source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  auto fast =
      buildGpuContractFixture(*source, mlir::shuttle::NumericalPolicy::Fast);
  ASSERT_TRUE(sourceOrdered);
  ASSERT_TRUE(fast);

  auto sourceBytes =
      mlir::shuttle::serializeGpuExecutableBundle(*sourceOrdered->module);
  auto fastBytes = mlir::shuttle::serializeGpuExecutableBundle(*fast->module);
  ASSERT_TRUE(mlir::succeeded(sourceBytes));
  ASSERT_TRUE(mlir::succeeded(fastBytes));
  EXPECT_NE(*sourceBytes, *fastBytes);

  auto sourceDecoded = decodeInlineSchema2(*sourceBytes);
  auto fastDecoded = decodeInlineSchema2(*fastBytes);
  ASSERT_TRUE(sourceDecoded);
  ASSERT_TRUE(fastDecoded);
  auto sourcePublic = mlir::shuttle::GpuExecutable::Load(*sourceBytes);
  auto fastPublic = mlir::shuttle::GpuExecutable::Load(*fastBytes);
  ASSERT_TRUE(sourcePublic.ok()) << sourcePublic.status();
  ASSERT_TRUE(fastPublic.ok()) << fastPublic.status();

  constexpr int64_t kTaskCount = 10;
  constexpr int64_t kSlotCount = 13;
  constexpr int64_t kExternalSlots = 5;
  constexpr int64_t kContractOrdinals[] = {3, 8};
  const std::vector<std::vector<int64_t>> expectedInputs = {
      {0}, {3}, {1}, {4, 5}, {6}, {7}, {3}, {2}, {9, 10}, {11},
  };
  ASSERT_EQ(expectedInputs.size(), kTaskCount);

  for (const Decoded *decoded : {&*sourceDecoded, &*fastDecoded}) {
    ASSERT_EQ(decoded->launches.size(), kTaskCount);
    EXPECT_EQ(decoded->slots, kSlotCount);
    EXPECT_EQ(decoded->externalSlots, kExternalSlots);
    EXPECT_EQ(decoded->scalarSlots, 0);
    for (int64_t ordinal = 0; ordinal < kTaskCount; ++ordinal) {
      const auto &launch = decoded->launches[ordinal];
      EXPECT_EQ(launch.taskOrdinal, ordinal);
      EXPECT_EQ(launch.inputSlots, expectedInputs[ordinal]);
      EXPECT_EQ(launch.outputSlots, (std::vector<int64_t>{3 + ordinal}));
    }
  }
  EXPECT_EQ(sourceDecoded->policy, 0);
  EXPECT_EQ(fastDecoded->policy, 1);
  EXPECT_EQ((*sourcePublic)->launches().size(), kTaskCount);
  EXPECT_EQ((*fastPublic)->launches().size(), kTaskCount);
  EXPECT_EQ((*sourcePublic)->slots().size(), kSlotCount);
  EXPECT_EQ((*fastPublic)->slots().size(), kSlotCount);

  for (int64_t ordinal : kContractOrdinals) {
    const auto sourceLaunch = sourceDecoded->launches[ordinal];
    const auto fastLaunch = fastDecoded->launches[ordinal];
    llvm::StringRef sourcePtx(
        reinterpret_cast<const char *>(sourceDecoded->code.data() +
                                       sourceLaunch.codeOffset),
        sourceLaunch.codeLength);
    llvm::StringRef fastPtx(
        reinterpret_cast<const char *>(fastDecoded->code.data() +
                                       fastLaunch.codeOffset),
        fastLaunch.codeLength);
    EXPECT_EQ(sourcePtx.count("fma.rn.f32"), 1u);
    EXPECT_FALSE(sourcePtx.contains("shfl.sync.down.b32"));
    EXPECT_EQ(fastPtx.count("fma.rn.f32"), 1u);
    EXPECT_EQ(fastPtx.count("shfl.sync.down.b32"), 5u);
    EXPECT_EQ(sourcePtx.count("ld.global.f32"), 2u);
    EXPECT_FALSE(sourcePtx.contains("ld.global.u16"));
    EXPECT_EQ(fastPtx.count("ld.global.f32"), 2u);
    EXPECT_FALSE(fastPtx.contains("ld.global.u16"));
    auto expectMappedAddresses = [&](llvm::StringRef ptx) {
      if (ordinal == 3) {
        // lhs=(d2,d0), rhs=(d2,d1), with d2 in %rd10.
        EXPECT_TRUE(ptx.contains("mad.lo.u64 %rd11, %rd11, 17, %rd10;\n"
                                 "  mad.lo.u64 %rd11, %rd11, 19, %rd8;"));
        EXPECT_TRUE(ptx.contains("mad.lo.u64 %rd13, %rd13, 17, %rd10;\n"
                                 "  mad.lo.u64 %rd13, %rd13, 33, %rd9;"));
      } else {
        // lhs=(d0,d2), rhs=(d1,d2), with d2 in %rd10.
        EXPECT_TRUE(ptx.contains("mad.lo.u64 %rd11, %rd11, 17, %rd8;\n"
                                 "  mad.lo.u64 %rd11, %rd11, 19, %rd10;"));
        EXPECT_TRUE(ptx.contains("mad.lo.u64 %rd13, %rd13, 33, %rd9;\n"
                                 "  mad.lo.u64 %rd13, %rd13, 19, %rd10;"));
      }
    };
    expectMappedAddresses(sourcePtx);
    expectMappedAddresses(fastPtx);
  }
  const auto transposeLaunch = sourceDecoded->launches[4];
  llvm::StringRef transposePtx(
      reinterpret_cast<const char *>(sourceDecoded->code.data() +
                                     transposeLaunch.codeOffset),
      transposeLaunch.codeLength);
  // The transpose Map consumes its input through (d1,d0), not identity order.
  EXPECT_TRUE(transposePtx.contains("mad.lo.u64 %rd10, %rd10, 19, %rd9;\n"
                                    "  mad.lo.u64 %rd10, %rd10, 33, %rd8;"));
}

TEST(GpuTransportV2Test,
     Abi15ComposedMatmulUsesTwoIndependentClosedTransportsTransactionally) {
  auto source =
      fixtureSource("jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir");
  ASSERT_TRUE(source);
  auto built = buildGpuContractFixture(
      *source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(built);
  auto serialized = serializeAllGpuBundles(*built->module);
  ASSERT_TRUE(serialized);
  ASSERT_EQ(serialized->size(), 2u);

  constexpr std::array<int64_t, 2> expectedLaunches = {2, 10};
  constexpr std::array<int64_t, 2> expectedSlots = {4, 13};
  constexpr std::array<std::array<int64_t, 5>, 2> expectedBytes = {{
      {1122, 1254, 646, 0, 0},
      {646, 1122, 1254, 1254, 1122},
  }};
  constexpr std::array<int64_t, 2> expectedBindings = {3, 5};
  for (size_t index = 0; index < serialized->size(); ++index) {
    auto decoded = decodeInlineSchema2((*serialized)[index]);
    ASSERT_TRUE(decoded);
    auto loaded = mlir::shuttle::GpuExecutable::Load((*serialized)[index]);
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    EXPECT_EQ(decoded->launches.size(), expectedLaunches[index]);
    EXPECT_EQ(decoded->slots, expectedSlots[index]);
    ASSERT_EQ((*loaded)->externalBindings().size(), expectedBindings[index]);
    std::vector<mlir::shuttle::GpuExternalBinding> structuralBindings =
        externalBindingsBySlot(**loaded);
    for (auto [binding, expectedBytes] :
         llvm::zip_equal(structuralBindings,
                         llvm::ArrayRef<int64_t>(expectedBytes[index])
                             .take_front(expectedBindings[index]))) {
      EXPECT_EQ(binding.requiredBytes, expectedBytes);
    }
    for (auto [ordinal, binding] :
         llvm::enumerate((*loaded)->externalBindings())) {
      const int64_t operands = index == 0 ? 2 : 3;
      EXPECT_EQ(binding.kind,
                ordinal < operands
                    ? mlir::shuttle::ExecutableBindingKind::Operand
                    : mlir::shuttle::ExecutableBindingKind::Result);
      EXPECT_EQ(binding.index,
                ordinal < operands ? ordinal : ordinal - operands);
    }
  }

  auto broken = buildGpuContractFixture(
      *source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(broken);
  llvm::SmallVector<mlir::shuttle::ExecutableBundleOp> bundles(
      broken->module->getOps<mlir::shuttle::ExecutableBundleOp>());
  ASSERT_EQ(bundles.size(), 2u);
  bundles[1].setSourceScheduleFingerprint(
      bundles[0].getSourceScheduleFingerprint());
  bundles[1].setDeviceModuleFingerprint(
      bundles[0].getDeviceModuleFingerprint());
  bundles[1].setInvocationAbiFingerprint(
      bundles[0].getInvocationAbiFingerprint());
  bundles[1].setFingerprint(
      mlir::shuttle::executableBundleFingerprint(bundles[1]));

  mlir::PassManager replace(broken->context.get());
  replace.addPass(mlir::shuttle::createReplaceWithGpuExecutableBundlePass());
  EXPECT_TRUE(mlir::failed(replace.run(*broken->module)));
  int64_t regions = 0;
  int64_t customCalls = 0;
  broken->module->walk([&](mlir::Operation *operation) {
    regions += mlir::isa<mlir::shuttle::RegionOp>(operation);
    customCalls +=
        operation->getName().getStringRef() == "stablehlo.custom_call";
  });
  EXPECT_EQ(regions, 2);
  EXPECT_EQ(customCalls, 0);
}

TEST(GpuTransportV2Test, FoldDynamicSharedMemoryIsDeclaredAtModuleScope) {
  auto built = buildFixture();
  ASSERT_TRUE(built);
  auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  auto decoded = decodeInlineSchema2(*bytes);
  ASSERT_TRUE(decoded);

  size_t foldCount = 0;
  for (const auto &launch : decoded->launches) {
    if (launch.dynamicSharedMemoryBytes == 0)
      continue;
    ++foldCount;
    llvm::StringRef ptx(reinterpret_cast<const char *>(decoded->code.data() +
                                                       launch.codeOffset),
                        launch.codeLength);
    size_t declaration =
        ptx.find(".extern .shared .align 4 .b8 shuttle_smem[];");
    size_t version = ptx.find(".version 8.0");
    size_t target = ptx.find(".target sm_90");
    size_t addressSize = ptx.find(".address_size 64");
    size_t entry = ptx.find(".visible .entry shuttle_entry(");
    ASSERT_EQ(version, 0);
    ASSERT_NE(target, llvm::StringRef::npos);
    ASSERT_NE(addressSize, llvm::StringRef::npos);
    ASSERT_NE(declaration, llvm::StringRef::npos);
    ASSERT_NE(entry, llvm::StringRef::npos);
    EXPECT_LT(version, target);
    EXPECT_LT(target, addressSize);
    EXPECT_LT(addressSize, declaration);
    EXPECT_LT(declaration, entry);
    EXPECT_EQ(ptx.find(".extern .shared", entry), llvm::StringRef::npos);
    size_t conversion = 0;
    size_t conversionCount = 0;
    while ((conversion = ptx.find("cvta.to.shared.u64 ", conversion)) !=
           llvm::StringRef::npos) {
      llvm::StringRef remaining = ptx.drop_front(conversion);
      llvm::StringRef instruction = remaining.take_front(remaining.find('\n'));
      auto operands = instruction.drop_front(
          llvm::StringRef("cvta.to.shared.u64 ").size()).split(',');
      llvm::StringRef destination = operands.first.trim();
      llvm::StringRef source = operands.second.trim(" ;");
      EXPECT_EQ(source, destination);
      std::string symbolMove =
          (llvm::Twine("mov.u64 ") + destination + ", shuttle_smem;").str();
      size_t symbolMovePosition = ptx.take_front(conversion).rfind(symbolMove);
      ASSERT_NE(symbolMovePosition, llvm::StringRef::npos);
      EXPECT_LT(conversion - symbolMovePosition, 64u);
      ++conversionCount;
      conversion += instruction.size();
    }
    EXPECT_GT(conversionCount, 0u);
  }
  EXPECT_EQ(foldCount, 1);
}

TEST(GpuTransportV2Test, FoldRejectsNonPositiveZeroInitializer) {
  auto source =
      fixtureSource("jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir");
  ASSERT_TRUE(source);
  ASSERT_TRUE(buildFixture(*source));
  constexpr llvm::StringLiteral positiveZero = "dense<0.000000e+00>";
  size_t position = source->find(positiveZero.str());
  ASSERT_NE(position, std::string::npos);
  for (llvm::StringRef replacement :
       {"dense<-0.000000e+00>", "dense<1.000000e+00>"}) {
    std::string mutated = *source;
    mutated.replace(position, positiveZero.size(), replacement.str());
    EXPECT_FALSE(buildFixture(mutated));
  }
}

TEST(GpuTransportV2Test, GeneratedBundleFingerprintBindsEveryClosedRoot) {
  auto built = buildFixture();
  ASSERT_TRUE(built);
  auto device = *built->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
  auto entry =
      *device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>().begin();
  auto abi = *built->module->getOps<mlir::shuttle::InvocationAbiOp>().begin();
  auto slot =
      *abi.getBody().front().getOps<mlir::shuttle::InvocationSlotOp>().begin();
  auto bundle =
      *built->module->getOps<mlir::shuttle::ExecutableBundleOp>().begin();
  const std::string baseline =
      mlir::shuttle::generatedGpuBundleFingerprint(*built->module);
  ASSERT_FALSE(baseline.empty());

  auto expectMutationChangesFingerprint = [&](mlir::Operation *operation,
                                              mlir::StringRef attribute,
                                              mlir::Attribute mutation) {
    mlir::Attribute original = operation->getAttr(attribute);
    ASSERT_TRUE(original);
    operation->setAttr(attribute, mutation);
    EXPECT_NE(mlir::shuttle::generatedGpuBundleFingerprint(*built->module),
              baseline);
    operation->setAttr(attribute, original);
    EXPECT_EQ(mlir::shuttle::generatedGpuBundleFingerprint(*built->module),
              baseline);
  };

  llvm::SmallVector<int8_t> code(device.getCode());
  code.front() ^= 1;
  expectMutationChangesFingerprint(
      device, "code", mlir::DenseI8ArrayAttr::get(built->context.get(), code));
  expectMutationChangesFingerprint(
      device, "schema_version",
      mlir::IntegerAttr::get(mlir::IntegerType::get(built->context.get(), 64),
                             device.getSchemaVersion() + 1));
  llvm::SmallVector<int64_t> grid(*entry.getGrid());
  ++grid.front();
  expectMutationChangesFingerprint(
      entry, "grid", mlir::DenseI64ArrayAttr::get(built->context.get(), grid));
  expectMutationChangesFingerprint(
      slot, "required_bytes",
      mlir::IntegerAttr::get(mlir::IntegerType::get(built->context.get(), 64),
                             slot.getRequiredBytes() + 1));
  expectMutationChangesFingerprint(
      bundle, "device_module_fingerprint",
      mlir::StringAttr::get(built->context.get(), std::string(64, '0')));
}

class SyntheticAllocator final
    : public stream_executor::DeviceAddressAllocator {
public:
  explicit SyntheticAllocator(uintptr_t base = 0x10000000)
      : DeviceAddressAllocator(nullptr), base(base) {}

  absl::StatusOr<stream_executor::ScopedDeviceAddress<uint8_t>>
  Allocate(int deviceOrdinal, uint64_t size, bool, int64_t) final {
    if (failAt && sizes.size() == *failAt)
      return absl::ResourceExhaustedError("synthetic allocation failure");
    sizes.push_back(size);
    uintptr_t address = base;
    for (uint64_t previous : sizes)
      address += previous + 256;
    return stream_executor::ScopedDeviceAddress<uint8_t>(
        stream_executor::DeviceAddressBase(reinterpret_cast<void *>(address),
                                           size),
        deviceOrdinal, this);
  }
  absl::Status Deallocate(int, stream_executor::DeviceAddressBase) final {
    ++deallocations;
    return absl::OkStatus();
  }
  absl::StatusOr<stream_executor::Stream *> GetStream(int) final {
    return absl::UnimplementedError("test allocator has no private stream");
  }

  llvm::SmallVector<uint64_t> sizes;
  int64_t deallocations = 0;
  std::optional<size_t> failAt;

private:
  uintptr_t base;
};

struct LoadedKernel {
  std::string ptx;
  std::string symbol;
  size_t arity = 0;
};

struct KernelLaunch {
  size_t ordinal = 0;
  std::array<uint64_t, 3> grid;
  std::array<uint64_t, 3> block;
  uint64_t sharedMemoryBytes = 0;
  llvm::SmallVector<uintptr_t> arguments;
};

struct ExecutionTrace {
  std::mutex mutex;
  llvm::SmallVector<LoadedKernel> loads;
  llvm::SmallVector<KernelLaunch> launches;
  std::atomic<int64_t> launchAttempts{0};
  std::atomic<int64_t> failAt{-1};
};

class RecordingKernel final : public stream_executor::Kernel {
public:
  RecordingKernel(unsigned arity, size_t ordinal,
                  std::shared_ptr<ExecutionTrace> trace)
      : arity(arity), ordinal(ordinal), trace(std::move(trace)) {}
  unsigned Arity() const final { return arity; }
  absl::StatusOr<int32_t>
  GetMaxOccupiedBlocksPerCore(stream_executor::ThreadDim, size_t) const final {
    return 1;
  }
  absl::Status Launch(const stream_executor::ThreadDim &thread,
                      const stream_executor::BlockDim &block,
                      const std::optional<stream_executor::ClusterDim> &,
                      stream_executor::Stream *,
                      const stream_executor::KernelArgs &args) final {
    int64_t attempt = trace->launchAttempts.fetch_add(1);
    if (trace->failAt.load() == attempt)
      return absl::InternalError("synthetic enqueue failure");
    auto *packed =
        stream_executor::DynCast<stream_executor::KernelArgsPackedArrayBase>(
            &args);
    KernelLaunch launch{ordinal,
                        {block.x, block.y, block.z},
                        {thread.x, thread.y, thread.z},
                        args.number_of_shared_bytes(),
                        {}};
    for (const void *argument : packed->argument_addresses()) {
      void *address = nullptr;
      std::memcpy(&address, argument, sizeof(address));
      launch.arguments.push_back(reinterpret_cast<uintptr_t>(address));
    }
    std::lock_guard lock(trace->mutex);
    trace->launches.push_back(std::move(launch));
    return absl::OkStatus();
  }

private:
  unsigned arity;
  size_t ordinal;
  std::shared_ptr<ExecutionTrace> trace;
};

xla::ffi::CallFrame callFrame(llvm::ArrayRef<uint8_t> transport,
                              llvm::StringRef transportDigest) {
  xla::ffi::CallFrameBuilder builder(/*num_args=*/2, /*num_rets=*/1);
  constexpr std::array<int64_t, 2> matrix{2048, 4096};
  constexpr std::array<int64_t, 1> vector{4096};
  builder.AddBufferArg(
      stream_executor::DeviceAddressBase(reinterpret_cast<void *>(0x40000000),
                                         2048 * 4096 * 2),
      xla::PrimitiveType::BF16, matrix);
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           reinterpret_cast<void *>(0x50000000), 4096 * 2),
                       xla::PrimitiveType::BF16, vector);
  builder.AddBufferRet(
      stream_executor::DeviceAddressBase(reinterpret_cast<void *>(0x60000000),
                                         2048 * 4096 * 2),
      xla::PrimitiveType::BF16, matrix);
  xla::ffi::CallFrameBuilder::AttributesBuilder attributes;
  attributes.Insert("bundle_bytes", std::string(reinterpret_cast<const char *>(
                                                    transport.data()),
                                                transport.size()));
  attributes.Insert("bundle_sha256", transportDigest.str());
  attributes.Insert("bundle_size", static_cast<int64_t>(transport.size()));
  attributes.Insert("transport_schema_version", int64_t{2});
  attributes.Insert("device_schema_version", int64_t{3});
  attributes.Insert("invocation_abi_schema_version", int64_t{3});
  attributes.Insert("bundle_schema_version", int64_t{2});
  attributes.Insert("completion", std::string("stream_ordered"));
  builder.AddAttributes(attributes.Build());
  return builder.Build();
}

struct DynamicFrame {
  xla::ffi::CallFrame frame;
  llvm::DenseMap<int64_t, uintptr_t> externalSlots;
};

std::optional<DynamicFrame>
dynamicCallFrame(const mlir::shuttle::GpuExecutable &executable,
                 llvm::ArrayRef<uint8_t> transport,
                 llvm::StringRef transportDigest) {
  llvm::SmallVector<const mlir::shuttle::GpuExternalBinding *> operands;
  llvm::SmallVector<const mlir::shuttle::GpuExternalBinding *> results;
  for (const auto &binding : executable.externalBindings()) {
    auto *projection =
        binding.kind == mlir::shuttle::ExecutableBindingKind::Operand
            ? &operands
        : binding.kind == mlir::shuttle::ExecutableBindingKind::Result
            ? &results
            : nullptr;
    if (!projection ||
        binding.index != static_cast<int64_t>(projection->size()))
      return std::nullopt;
    projection->push_back(&binding);
  }
  if (operands.empty() || results.empty())
    return std::nullopt;
  xla::ffi::CallFrameBuilder builder(operands.size(), results.size());
  llvm::DenseMap<int64_t, uintptr_t> externalSlots;
  auto add = [&](const mlir::shuttle::GpuExternalBinding &binding,
                 bool result) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(binding.tensorType);
    if (!tensor || !tensor.getElementType().isBF16())
      return false;
    uintptr_t address =
        UINT64_C(0x100000000) +
        static_cast<uintptr_t>(binding.slotOrdinal) * UINT64_C(0x10000000);
    stream_executor::DeviceAddressBase deviceAddress(
        reinterpret_cast<void *>(address), binding.requiredBytes);
    if (result)
      builder.AddBufferRet(deviceAddress, xla::PrimitiveType::BF16,
                           tensor.getShape());
    else
      builder.AddBufferArg(deviceAddress, xla::PrimitiveType::BF16,
                           tensor.getShape());
    externalSlots[binding.slotOrdinal] = address;
    return true;
  };
  for (const auto *binding : operands)
    if (!add(*binding, false))
      return std::nullopt;
  for (const auto *binding : results)
    if (!add(*binding, true))
      return std::nullopt;
  xla::ffi::CallFrameBuilder::AttributesBuilder attributes;
  attributes.Insert("bundle_bytes", std::string(reinterpret_cast<const char *>(
                                                    transport.data()),
                                                transport.size()));
  attributes.Insert("bundle_sha256", transportDigest.str());
  attributes.Insert("bundle_size", static_cast<int64_t>(transport.size()));
  attributes.Insert("transport_schema_version", int64_t{2});
  attributes.Insert("device_schema_version", int64_t{3});
  attributes.Insert("invocation_abi_schema_version", int64_t{3});
  attributes.Insert("bundle_schema_version", int64_t{2});
  attributes.Insert("completion", std::string("stream_ordered"));
  builder.AddAttributes(attributes.Build());
  return DynamicFrame{builder.Build(), std::move(externalSlots)};
}

std::unique_ptr<stream_executor::Kernel>
recordLoadedKernel(const stream_executor::KernelLoaderSpec &spec,
                   const std::shared_ptr<ExecutionTrace> &trace) {
  auto ptx = spec.cuda_ptx_in_memory();
  if (!ptx)
    return nullptr;
  size_t ordinal = 0;
  {
    std::lock_guard lock(trace->mutex);
    ordinal = trace->loads.size();
    trace->loads.push_back(
        {std::string(ptx->ptx), spec.kernel_name(), spec.arity()});
  }
  return std::make_unique<RecordingKernel>(spec.arity(), ordinal, trace);
}

void expectLoadedKernels(const Decoded &decoded, const ExecutionTrace &trace,
                         size_t start) {
  ASSERT_GE(trace.loads.size(), start + decoded.launches.size());
  for (size_t ordinal = 0; ordinal < decoded.launches.size(); ++ordinal) {
    const auto &expected = decoded.launches[ordinal];
    const auto &actual = trace.loads[start + ordinal];
    EXPECT_EQ(llvm::StringRef(actual.ptx),
              llvm::StringRef(reinterpret_cast<const char *>(
                                  decoded.code.data() + expected.codeOffset),
                              expected.codeLength));
    EXPECT_EQ(actual.symbol, "shuttle_entry");
    EXPECT_EQ(actual.arity,
              expected.inputSlots.size() + expected.outputSlots.size());
  }
}

void expectKernelLaunches(const Decoded &decoded, const ExecutionTrace &trace,
                          size_t start,
                          const llvm::DenseMap<int64_t, uintptr_t> &slots) {
  ASSERT_GE(trace.launches.size(), start + decoded.launches.size());
  for (size_t ordinal = 0; ordinal < decoded.launches.size(); ++ordinal) {
    const auto &expected = decoded.launches[ordinal];
    const auto &actual = trace.launches[start + ordinal];
    EXPECT_EQ(actual.ordinal, ordinal);
    EXPECT_EQ(actual.grid, expected.grid);
    EXPECT_EQ(actual.block, expected.block);
    EXPECT_EQ(actual.sharedMemoryBytes, expected.dynamicSharedMemoryBytes);
    llvm::SmallVector<uintptr_t> expectedArguments;
    for (int64_t slot : expected.inputSlots)
      expectedArguments.push_back(slots.lookup(slot));
    for (int64_t slot : expected.outputSlots)
      expectedArguments.push_back(slots.lookup(slot));
    EXPECT_EQ(actual.arguments, expectedArguments);
  }
}

llvm::DenseMap<int64_t, uintptr_t>
slotAddresses(const Decoded &decoded, const SyntheticAllocator &allocator,
              uintptr_t allocationBase) {
  llvm::DenseMap<int64_t, uintptr_t> groups;
  uintptr_t address = allocationBase;
  for (auto [group, bytes] :
       llvm::zip_equal(decoded.temporaryGroupOrder, allocator.sizes)) {
    address += bytes + 256;
    groups[group] = address;
  }
  llvm::DenseMap<int64_t, uintptr_t> slots{
      {0, 0x40000000}, {1, 0x50000000}, {20, 0x60000000}};
  for (int64_t ordinal = 2; ordinal < 20; ++ordinal)
    slots[ordinal] = groups.lookup(decoded.slotReuseGroups[ordinal]);
  return slots;
}

llvm::DenseMap<int64_t, uintptr_t>
dynamicSlotAddresses(const Decoded &decoded,
                     const SyntheticAllocator &allocator,
                     uintptr_t allocationBase,
                     llvm::DenseMap<int64_t, uintptr_t> externalSlots) {
  llvm::DenseMap<int64_t, uintptr_t> groups;
  uintptr_t address = allocationBase;
  for (auto [group, bytes] :
       llvm::zip_equal(decoded.temporaryGroupOrder, allocator.sizes)) {
    address += bytes + 256;
    groups[group] = address;
  }
  for (int64_t ordinal = 0; ordinal < decoded.slots; ++ordinal)
    if (decoded.slotStorage[ordinal] == 1)
      externalSlots[ordinal] = groups.lookup(decoded.slotReuseGroups[ordinal]);
  return externalSlots;
}

TEST(GpuTransportV2Test, Sm100RuntimeRequiresExactComputeCapability) {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir");
  ASSERT_TRUE(source);
  auto built =
      buildFixture(*source, mlir::shuttle::NumericalPolicy::SourceOrdered, {},
                   {}, mlir::shuttle::GpuTarget::Sm100);
  ASSERT_TRUE(built);
  auto transport = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));
  auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
  ASSERT_TRUE(executable.ok()) << executable.status();

  SyntheticAllocator allocator;
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  auto trace = std::make_shared<ExecutionTrace>();
  EXPECT_CALL(stream, parent()).WillRepeatedly(testing::Return(&executor));
  EXPECT_CALL(executor, LoadKernel(testing::_))
      .Times((*executable)->launches().size())
      .WillRepeatedly([trace](const stream_executor::KernelLoaderSpec &spec) {
        return recordLoadedKernel(spec, trace);
      });
  stream_executor::GpuComputeCapability sm100{
      stream_executor::CudaComputeCapability{10, 0}};
  auto runner = mlir::shuttle::GpuRunner::Create(*transport, &allocator, 0,
                                                 &stream, &sm100);
  ASSERT_TRUE(runner.ok()) << runner.status();
  ASSERT_EQ(trace->loads.size(), (*executable)->launches().size());
  EXPECT_TRUE(llvm::all_of(trace->loads, [](const LoadedKernel &kernel) {
    return llvm::StringRef(kernel.ptx)
        .starts_with(".version 8.7\n.target sm_100\n.address_size 64\n");
  }));

  stream_executor::GpuComputeCapability sm90{
      stream_executor::CudaComputeCapability{9, 0}};
  EXPECT_FALSE(mlir::shuttle::GpuRunner::Create(*transport, &allocator, 0,
                                                &stream, &sm90)
                   .ok());

  auto legacyBuilt = buildFixture(*source);
  ASSERT_TRUE(legacyBuilt);
  auto legacyTransport =
      mlir::shuttle::serializeGpuExecutableBundle(*legacyBuilt->module);
  ASSERT_TRUE(mlir::succeeded(legacyTransport));
  EXPECT_FALSE(mlir::shuttle::GpuRunner::Create(*legacyTransport, &allocator, 0,
                                                &stream, &sm100)
                   .ok());
}

TEST(GpuTransportV2Test, ExportedFfiBundleOwnsEveryLifecycleStage) {
  auto built = buildFixture();
  ASSERT_TRUE(built);
  auto transport = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));
  std::string transportDigest =
      mlir::shuttle::gpuExecutableBundleDigest(*transport);
  auto frame = callFrame(*transport, transportDigest);
  auto decoded = decodeInlineSchema2(*transport);
  ASSERT_TRUE(decoded);
  auto alternateBuilt = buildAlternateFixture();
  ASSERT_TRUE(alternateBuilt);
  auto alternateTransport =
      mlir::shuttle::serializeGpuExecutableBundle(*alternateBuilt->module);
  ASSERT_TRUE(mlir::succeeded(alternateTransport));
  auto alternateDecoded = decodeInlineSchema2(*alternateTransport);
  ASSERT_TRUE(alternateDecoded);
  ASSERT_NE(decoded->code, alternateDecoded->code);
  XLA_FFI_Handler_Bundle handlers =
      mlir::shuttle::gpuExecutableBundleFfiHandlerBundle();
  ASSERT_NE(handlers.instantiate, nullptr);
  ASSERT_NE(handlers.prepare, nullptr);
  ASSERT_NE(handlers.initialize, nullptr);
  ASSERT_NE(handlers.execute, nullptr);

  SyntheticAllocator allocator;
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  stream_executor::GpuComputeCapability capability{
      stream_executor::CudaComputeCapability{9, 0}};
  auto trace = std::make_shared<ExecutionTrace>();
  EXPECT_CALL(stream, parent()).WillRepeatedly(testing::Return(&executor));
  EXPECT_CALL(executor, LoadKernel(testing::_))
      .Times(38)
      .WillRepeatedly([trace](const stream_executor::KernelLoaderSpec &spec) {
        return recordLoadedKernel(spec, trace);
      });

  xla::ffi::ExecutionState instantiateState;
  xla::ffi::ExecutionState prepareState;
  xla::ffi::ExecutionState initializeState;
  xla::ffi::InvokeContext context;
  context.device_ordinal = 0;
  context.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &allocator, nullptr, nullptr,
      nullptr, nullptr,    nullptr, &capability};
  context.state_context = {&instantiateState, &prepareState, &initializeState};

  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               frame, context,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               frame, context, XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_EQ(allocator.sizes.size(), decoded->temporaryGroupOrder.size());
  EXPECT_EQ(allocator.sizes, decoded->temporaryGroupBytes);
  EXPECT_EQ(llvm::accumulate(allocator.sizes, uint64_t{0}), 100671488);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.initialize,
                               frame, context,
                               XLA_FFI_ExecutionStage_INITIALIZE)
                  .ok());
  expectLoadedKernels(*decoded, *trace, 0);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                               frame, context, XLA_FFI_ExecutionStage_EXECUTE)
                  .ok());
  EXPECT_EQ(trace->launchAttempts.load(), 19);
  llvm::DenseMap<int64_t, uintptr_t> firstSlots =
      slotAddresses(*decoded, allocator, 0x10000000);
  expectKernelLaunches(*decoded, *trace, 0, firstSlots);

  trace->launchAttempts.store(0);
  trace->failAt.store(7);
  EXPECT_FALSE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                                frame, context, XLA_FFI_ExecutionStage_EXECUTE)
                   .ok());
  EXPECT_EQ(trace->launchAttempts.load(), 8);
  trace->failAt.store(-1);

  trace->launchAttempts.store(0);
  SyntheticAllocator secondAllocator(0x70000000);
  xla::ffi::ExecutionState secondPrepareState;
  xla::ffi::InvokeContext secondContext = context;
  secondContext.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &secondAllocator, nullptr, nullptr,
      nullptr, nullptr,          nullptr, &capability};
  secondContext.state_context.prepare = &secondPrepareState;
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               frame, secondContext,
                               XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_EQ(secondAllocator.sizes, allocator.sizes);
  size_t concurrentStart = 0;
  {
    std::lock_guard lock(trace->mutex);
    concurrentStart = trace->launches.size();
  }
  auto firstFrame = frame.Copy();
  auto secondFrame = frame.Copy();
  auto first = std::async(std::launch::async, [&] {
    return xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                            firstFrame, context,
                            XLA_FFI_ExecutionStage_EXECUTE);
  });
  auto second = std::async(std::launch::async, [&] {
    return xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                            secondFrame, secondContext,
                            XLA_FFI_ExecutionStage_EXECUTE);
  });
  EXPECT_TRUE(first.get().ok());
  EXPECT_TRUE(second.get().ok());
  EXPECT_EQ(trace->launchAttempts.load(), 38);
  llvm::DenseMap<int64_t, uintptr_t> secondSlots =
      slotAddresses(*decoded, secondAllocator, 0x70000000);
  // Independent allocators must yield distinct invocation-local addresses.
  EXPECT_NE(firstSlots.lookup(2), secondSlots.lookup(2));
  llvm::SmallVector<int64_t> firstLaunchesPerOrdinal(19);
  llvm::SmallVector<int64_t> secondLaunchesPerOrdinal(19);
  {
    std::lock_guard lock(trace->mutex);
    ASSERT_EQ(trace->launches.size() - concurrentStart, 38);
    for (const auto &actual :
         llvm::drop_begin(trace->launches, concurrentStart)) {
      ASSERT_LT(actual.ordinal, decoded->launches.size());
      const auto &expected = decoded->launches[actual.ordinal];
      EXPECT_EQ(actual.grid, expected.grid);
      EXPECT_EQ(actual.block, expected.block);
      EXPECT_EQ(actual.sharedMemoryBytes, expected.dynamicSharedMemoryBytes);
      auto expectedArguments =
          [&](const llvm::DenseMap<int64_t, uintptr_t> &slots) {
            llvm::SmallVector<uintptr_t> arguments;
            for (int64_t slot : expected.inputSlots)
              arguments.push_back(slots.lookup(slot));
            for (int64_t slot : expected.outputSlots)
              arguments.push_back(slots.lookup(slot));
            return arguments;
          };
      auto firstArguments = expectedArguments(firstSlots);
      auto secondArguments = expectedArguments(secondSlots);
      ASSERT_NE(firstArguments, secondArguments);
      bool usesFirstState = actual.arguments == firstArguments;
      bool usesSecondState = actual.arguments == secondArguments;
      EXPECT_NE(usesFirstState, usesSecondState);
      if (usesFirstState)
        ++firstLaunchesPerOrdinal[actual.ordinal];
      if (usesSecondState)
        ++secondLaunchesPerOrdinal[actual.ordinal];
    }
  }
  EXPECT_TRUE(llvm::all_of(firstLaunchesPerOrdinal,
                           [](int64_t count) { return count == 1; }));
  EXPECT_TRUE(llvm::all_of(secondLaunchesPerOrdinal,
                           [](int64_t count) { return count == 1; }));

  auto alternateFrame =
      callFrame(*alternateTransport,
                mlir::shuttle::gpuExecutableBundleDigest(*alternateTransport));
  xla::ffi::ExecutionState alternateInstantiateState;
  xla::ffi::ExecutionState alternatePrepareState;
  xla::ffi::ExecutionState alternateInitializeState;
  SyntheticAllocator alternateAllocator(0x80000000);
  xla::ffi::InvokeContext alternateContext = context;
  alternateContext.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &alternateAllocator, nullptr, nullptr, nullptr, nullptr,
      nullptr, &capability};
  alternateContext.state_context = {&alternateInstantiateState,
                                    &alternatePrepareState,
                                    &alternateInitializeState};
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               alternateFrame, alternateContext,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               alternateFrame, alternateContext,
                               XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.initialize,
                               alternateFrame, alternateContext,
                               XLA_FFI_ExecutionStage_INITIALIZE)
                  .ok());
  expectLoadedKernels(*alternateDecoded, *trace, decoded->launches.size());

  auto corrupt = callFrame(*transport, std::string(64, '0'));
  EXPECT_FALSE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                                corrupt, context,
                                XLA_FFI_ExecutionStage_INSTANTIATE)
                   .ok());

  SyntheticAllocator failingAllocator;
  failingAllocator.failAt = 2;
  xla::ffi::ExecutionState failedPrepare;
  context.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &failingAllocator, nullptr, nullptr,
      nullptr, nullptr,           nullptr, &capability};
  context.state_context.prepare = &failedPrepare;
  EXPECT_FALSE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                                frame, context, XLA_FFI_ExecutionStage_PREPARE)
                   .ok());
  EXPECT_EQ(failingAllocator.sizes.size(), 2);
  EXPECT_EQ(failingAllocator.deallocations, 2);
}

TEST(GpuTransportV2Test,
     StandaloneRunnerExecutesAbi22FusedRmsNormForwardAndBackward) {
  static_assert(!std::is_move_constructible_v<mlir::shuttle::GpuRunner>);
  static_assert(!std::is_move_assignable_v<mlir::shuttle::GpuRunner>);

  struct Boundary {
    const char *filename;
    int64_t launches;
    int64_t slots;
    int64_t temporaryGroups;
    uint64_t temporaryBytes;
  };
  constexpr Boundary boundaries[] = {
      {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir",
       1, 3, 0, 0},
      {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir",
       3, 6, 1, 56},
      {"jax-0.10.1-bf16-projected_parameter_broadcast-"
       "3x2x5x11x263-backward.mlir",
       3, 6, 1, 1320},
  };
  constexpr llvm::StringLiteral options =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral tuning =
      R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";

  for (const Boundary &boundary : boundaries) {
    SCOPED_TRACE(boundary.filename);
    auto source = fixtureSource(boundary.filename);
    ASSERT_TRUE(source);
    auto built = buildFixture(
        *source, mlir::shuttle::NumericalPolicy::SourceOrdered, options,
        tuning);
    ASSERT_TRUE(built);
    auto transport =
        mlir::shuttle::serializeGpuExecutableBundle(*built->module);
    ASSERT_TRUE(mlir::succeeded(transport));
    auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
    ASSERT_TRUE(executable.ok()) << executable.status();
    auto decoded = decodeInlineSchema2(*transport);
    ASSERT_TRUE(decoded);
    ASSERT_EQ(decoded->launches.size(), boundary.launches);
    ASSERT_EQ(decoded->slots, boundary.slots);
    EXPECT_EQ(decoded->temporaryGroupOrder.size(), boundary.temporaryGroups);
    EXPECT_EQ(llvm::accumulate(decoded->temporaryGroupBytes, uint64_t{0}),
              boundary.temporaryBytes);

    llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> operands;
    llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> results;
    llvm::DenseMap<int64_t, uintptr_t> externalSlots;
    for (const mlir::shuttle::GpuExternalBinding &binding :
         (*executable)->externalBindings()) {
      uintptr_t address = UINT64_C(0x100000000) +
                          static_cast<uintptr_t>(binding.slotOrdinal) *
                              UINT64_C(0x10000000);
      mlir::shuttle::GpuRunnerBuffer buffer{
          reinterpret_cast<void *>(address), binding.requiredBytes};
      if (binding.kind ==
          mlir::shuttle::ExecutableBindingKind::Operand) {
        ASSERT_EQ(binding.index, static_cast<int64_t>(operands.size()));
        operands.push_back(buffer);
      } else {
        ASSERT_EQ(binding.kind,
                  mlir::shuttle::ExecutableBindingKind::Result);
        ASSERT_EQ(binding.index, static_cast<int64_t>(results.size()));
        results.push_back(buffer);
      }
      externalSlots[binding.slotOrdinal] = address;
    }

    SyntheticAllocator allocator(0xa0000000);
    stream_executor::MockStreamExecutor executor;
    stream_executor::MockStream stream;
    stream_executor::GpuComputeCapability capability{
        stream_executor::CudaComputeCapability{9, 0}};
    auto trace = std::make_shared<ExecutionTrace>();
    EXPECT_CALL(stream, parent()).WillRepeatedly(testing::Return(&executor));
    EXPECT_CALL(stream, BlockHostUntilDone())
        .Times(4)
        .WillOnce(testing::Return(absl::OkStatus()))
        .WillOnce(testing::Return(
            absl::InternalError("synthetic completion failure")))
        .WillRepeatedly(testing::Return(absl::OkStatus()));
    EXPECT_CALL(executor, LoadKernel(testing::_))
        .Times(boundary.launches)
        .WillRepeatedly([trace](
                            const stream_executor::KernelLoaderSpec &spec) {
          return recordLoadedKernel(spec, trace);
        });

    auto runner = mlir::shuttle::GpuRunner::Create(
        *transport, &allocator, 0, &stream, &capability);
    ASSERT_TRUE(runner.ok()) << runner.status();
    const mlir::shuttle::GpuRunnerResources &resources =
        (*runner)->resources();
    EXPECT_EQ(resources.codeBytes, decoded->code.size());
    EXPECT_EQ(resources.launchCount, boundary.launches);
    EXPECT_EQ(resources.slotCount, decoded->slots);
    EXPECT_EQ(resources.externalOperandCount, operands.size());
    EXPECT_EQ(resources.externalResultCount, results.size());
    EXPECT_EQ(resources.temporaryAllocationCount,
              boundary.temporaryGroups);
    EXPECT_EQ(resources.temporaryBytes, boundary.temporaryBytes);
    EXPECT_EQ(allocator.sizes, decoded->temporaryGroupBytes);
    expectLoadedKernels(*decoded, *trace, 0);

    absl::Status execution = (*runner)->Execute(operands, results);
    ASSERT_TRUE(execution.ok()) << execution;
    EXPECT_EQ(trace->launchAttempts.load(), boundary.launches);
    auto slots = dynamicSlotAddresses(*decoded, allocator, 0xa0000000,
                                      std::move(externalSlots));
    expectKernelLaunches(*decoded, *trace, 0, slots);

    llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> wrongOperands(operands);
    wrongOperands.front().sizeBytes -= 2;
    EXPECT_FALSE((*runner)->Execute(wrongOperands, results).ok());
    EXPECT_EQ(trace->launchAttempts.load(), boundary.launches);

    if (operands.size() > 1) {
      llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> aliased(operands);
      aliased[1].data = aliased[0].data;
      EXPECT_FALSE((*runner)->Execute(aliased, results).ok());
      EXPECT_EQ(trace->launchAttempts.load(), boundary.launches);
    }

    EXPECT_FALSE((*runner)->Execute(operands, results).ok());
    EXPECT_EQ(trace->launchAttempts.load(), boundary.launches * 2);
    EXPECT_FALSE((*runner)->Execute(operands, results).ok());
    EXPECT_EQ(trace->launchAttempts.load(), boundary.launches * 2);
    EXPECT_TRUE((*runner)->AwaitCompletion().ok());
    EXPECT_TRUE((*runner)->Execute(operands, results).ok());
    EXPECT_EQ(trace->launchAttempts.load(), boundary.launches * 3);

    stream_executor::GpuComputeCapability unsupported{
        stream_executor::CudaComputeCapability{8, 0}};
    EXPECT_FALSE(mlir::shuttle::GpuRunner::Create(
                     *transport, &allocator, 0, &stream, &unsupported)
                     .ok());
  }
}

uint16_t roundToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += UINT32_C(0x7fff) + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float expandBf16(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

int32_t orderedBf16(uint16_t value) {
  if (value & UINT16_C(0x8000))
    return INT32_C(0x8000) - static_cast<int32_t>(value & UINT16_C(0x7fff));
  return INT32_C(0x8000) + static_cast<int32_t>(value);
}

struct LayerNormInputs {
  std::vector<uint16_t> x;
  std::vector<uint16_t> gamma;
  std::vector<uint16_t> beta;
};

struct LayerNormForwardOracle {
  std::vector<uint16_t> output;
  std::vector<float> means;
  std::vector<float> variances;
};

LayerNormInputs layerNormInputs(int64_t rows, int64_t features) {
  LayerNormInputs inputs;
  inputs.x.resize(rows * features);
  inputs.gamma.resize(features);
  inputs.beta.resize(features);
  for (int64_t feature = 0; feature < features; ++feature) {
    inputs.gamma[feature] = roundToBf16(
        0.625f + static_cast<float>((feature * 11) % 23) * 0.03125f);
    inputs.beta[feature] = roundToBf16(
        -0.25f + static_cast<float>((feature * 7) % 19) * 0.0234375f);
  }
  for (int64_t row = 0; row < rows; ++row) {
    const float rowOffset = -0.55f + static_cast<float>(row % 9) * 0.1375f;
    const float rowScale = 0.35f + static_cast<float>(row % 7) * 0.0875f;
    for (int64_t feature = 0; feature < features; ++feature) {
      const int64_t featurePattern = (feature * 37) % 257 - 128;
      const int64_t skewPattern = (feature * feature + row * 17) % 29 - 14;
      const float value =
          rowOffset + rowScale * static_cast<float>(featurePattern) / 128.0f +
          static_cast<float>(skewPattern) * 0.0046875f;
      inputs.x[row * features + feature] = roundToBf16(value);
    }
  }
  return inputs;
}

LayerNormForwardOracle layerNormForwardReference(const LayerNormInputs &inputs,
                                                 int64_t rows,
                                                 int64_t features) {
  assert(inputs.x.size() == static_cast<size_t>(rows * features));
  assert(inputs.gamma.size() == static_cast<size_t>(features));
  assert(inputs.beta.size() == static_cast<size_t>(features));
  LayerNormForwardOracle oracle;
  oracle.output.resize(inputs.x.size());
  oracle.means.resize(rows);
  oracle.variances.resize(rows);
  for (int64_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (int64_t feature = 0; feature < features; ++feature)
      sum += expandBf16(inputs.x[row * features + feature]);
    const float mean = sum / static_cast<float>(features);
    float squaredSum = 0.0f;
    for (int64_t feature = 0; feature < features; ++feature) {
      const float centered =
          expandBf16(inputs.x[row * features + feature]) - mean;
      squaredSum += centered * centered;
    }
    const float variance = squaredSum / static_cast<float>(features);
    const float inverseStandardDeviation = 1.0f / std::sqrt(variance + 1.0e-5f);
    oracle.means[row] = mean;
    oracle.variances[row] = variance;
    for (int64_t feature = 0; feature < features; ++feature) {
      const float centered =
          expandBf16(inputs.x[row * features + feature]) - mean;
      oracle.output[row * features + feature] =
          roundToBf16(centered * inverseStandardDeviation *
                          expandBf16(inputs.gamma[feature]) +
                      expandBf16(inputs.beta[feature]));
    }
  }
  return oracle;
}

TEST(GpuTransportV2Test,
     PublicResultBindingPermutationRejectsMalformedSerializedIndices) {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-backward.mlir");
  ASSERT_TRUE(source);
  constexpr llvm::StringLiteral options =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral tuning =
      R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
  auto built = buildFixture(*source,
                            mlir::shuttle::NumericalPolicy::SourceOrdered,
                            options, tuning);
  ASSERT_TRUE(built);
  auto transport =
      mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));
  auto decoded = decodeInlineSchema2(*transport);
  ASSERT_TRUE(decoded);
  ASSERT_TRUE(mlir::shuttle::GpuExecutable::Load(*transport).ok());

  llvm::SmallVector<int64_t> resultSlots;
  for (int64_t slot = 0; slot < decoded->slots; ++slot)
    if (decoded->slotBinding[slot] == 2)
      resultSlots.push_back(slot);
  ASSERT_EQ(resultSlots.size(), 3u);
  EXPECT_EQ(decoded->slotBindingIndex[resultSlots[0]], 2);
  EXPECT_EQ(decoded->slotBindingIndex[resultSlots[1]], 1);
  EXPECT_EQ(decoded->slotBindingIndex[resultSlots[2]], 0);

  auto expectRejected = [](llvm::ArrayRef<uint8_t> mutation) {
    ASSERT_FALSE(mutation.empty());
    EXPECT_FALSE(decodeInlineSchema2(mutation));
    EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(mutation).ok());
  };
  expectRejected(mutateBindingIndexAndRehash(
      *transport, *decoded, resultSlots[1],
      decoded->slotBindingIndex[resultSlots[2]]));
  expectRejected(
      mutateBindingIndexAndRehash(*transport, *decoded, resultSlots[0], 3));
  expectRejected(
      mutateBindingIndexAndRehash(*transport, *decoded, resultSlots[0], -1));
}

TEST(GpuTransportV2Test,
     LayerNormForwardHostOracleRejectsMissingMeanVarianceAndFeatureIndexing) {
  constexpr int64_t rows = 7;
  constexpr int64_t features = 13;
  LayerNormInputs inputs = layerNormInputs(rows, features);
  LayerNormForwardOracle oracle =
      layerNormForwardReference(inputs, rows, features);
  int64_t missingMeanDifferences = 0;
  int64_t missingVarianceDifferences = 0;
  int64_t rotatedAffineDifferences = 0;
  float minimumMean = std::numeric_limits<float>::infinity();
  float maximumMean = -std::numeric_limits<float>::infinity();
  float absoluteMeanSum = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    minimumMean = std::min(minimumMean, oracle.means[row]);
    maximumMean = std::max(maximumMean, oracle.means[row]);
    absoluteMeanSum += std::abs(oracle.means[row]);
    EXPECT_GT(oracle.variances[row], 0.01f);
    EXPECT_LT(oracle.variances[row], 0.5f);
    const float inverseStandardDeviation =
        1.0f / std::sqrt(oracle.variances[row] + 1.0e-5f);
    for (int64_t feature = 0; feature < features; ++feature) {
      const int64_t element = row * features + feature;
      const float value = expandBf16(inputs.x[element]);
      const float gamma = expandBf16(inputs.gamma[feature]);
      const float beta = expandBf16(inputs.beta[feature]);
      const uint16_t missingMean =
          roundToBf16(value * inverseStandardDeviation * gamma + beta);
      const uint16_t missingVariance =
          roundToBf16((value - oracle.means[row]) * gamma + beta);
      const int64_t rotatedFeature = (feature + 1) % features;
      const uint16_t rotatedAffine =
          roundToBf16((value - oracle.means[row]) * inverseStandardDeviation *
                          expandBf16(inputs.gamma[rotatedFeature]) +
                      expandBf16(inputs.beta[rotatedFeature]));
      missingMeanDifferences += missingMean != oracle.output[element];
      missingVarianceDifferences += missingVariance != oracle.output[element];
      rotatedAffineDifferences += rotatedAffine != oracle.output[element];
    }
  }
  const int64_t majority = rows * features / 2;
  EXPECT_LT(minimumMean, -0.25f);
  EXPECT_GT(maximumMean, 0.1f);
  EXPECT_GT(absoluteMeanSum, 1.0f);
  EXPECT_GT(missingMeanDifferences, majority);
  EXPECT_GT(missingVarianceDifferences, majority);
  EXPECT_GT(rotatedAffineDifferences, majority);
}

enum class ProspectiveLayerNormCase { Balanced, PositiveOffset };

struct ProspectiveLayerNormCell {
  ProspectiveLayerNormCase testCase;
  int64_t rows;
  int64_t features;
  llvm::StringLiteral xDigest;
  llvm::StringLiteral gammaDigest;
  llvm::StringLiteral betaDigest;
  llvm::StringLiteral binary64Digest;
  llvm::StringLiteral scheduleDigest;
  double scheduleMaxAbsoluteError;
  double scheduleMeanAbsoluteError;
  double scheduleRelativeLinfError;
  int32_t scheduleMaxBf16UlpError;
};

struct ProspectiveLayerNormBoundary {
  llvm::StringLiteral fixture;
  int64_t rows;
  int64_t features;
};

constexpr ProspectiveLayerNormCell prospectiveLayerNormCells[] = {
    {ProspectiveLayerNormCase::Balanced, 7, 13,
     "2e43a818f671624fccb9795c962c5ebcafc27ea1e9e8de51f99a9037aabae0f2",
     "34effe46fe6716c40d0ae04f50309d60e9743b32c86d890a64b1181378f54ef9",
     "fcc835f6d26f1dddc481c20f646176ad694e6174115a5599fdd165668c8f5de8",
     "9852b2a458e0a2c04a02eeaf23665c4fddafa0dc5e27c078f432e2c5f6cdce87",
     "9852b2a458e0a2c04a02eeaf23665c4fddafa0dc5e27c078f432e2c5f6cdce87", 0.0,
     0.0, 0.0, 0},
    {ProspectiveLayerNormCase::Balanced, 2048, 4096,
     "15e4956b1c4d383ba6fddf1b125c9bb9cb7c2ff98b209b4d1c6d1a0dfa918a81",
     "872b4391b6ea27f0ad4741ecd49609658ad243341f6d8f99ae76b04164f6c94e",
     "a9da8f908c0517f02b32fa19b2150614f27bb165dc90b96a28c1c9782006dec5",
     "958b574bed7b59b3e772d3bfe75e3dd60f57bd61e3ca21ad38045c0ed705f27c",
     "9126627880838eb8205d217555045de1c9c5621e10f5bd74623b9d5801f558ac",
     0.015625, 3.1493970653606596e-08, 0.004975124378109453, 9},
    {ProspectiveLayerNormCase::PositiveOffset, 7, 13,
     "2e43a818f671624fccb9795c962c5ebcafc27ea1e9e8de51f99a9037aabae0f2",
     "34effe46fe6716c40d0ae04f50309d60e9743b32c86d890a64b1181378f54ef9",
     "91f2179937df9a6204846408a95194b24fbf38d49300696ddb1a6fd5c5087985",
     "b33fad9beb38cf7af1bd937665bdbf508171e4f458759886f981a12d039219e8",
     "b33fad9beb38cf7af1bd937665bdbf508171e4f458759886f981a12d039219e8", 0.0,
     0.0, 0.0, 0},
    {ProspectiveLayerNormCase::PositiveOffset, 2048, 4096,
     "15e4956b1c4d383ba6fddf1b125c9bb9cb7c2ff98b209b4d1c6d1a0dfa918a81",
     "872b4391b6ea27f0ad4741ecd49609658ad243341f6d8f99ae76b04164f6c94e",
     "73eb4840bac211062417bea20b7633dd51e525f744a20d87b91a3f0283e5ba2a",
     "878c03fd5b00d131debc718b80a9d208bb23096f80db5c67e44ef6f88f08e97c",
     "8410a8e6435fdd1dd953cc0056770b8ccc909149a705172344411a8d753d2068", 0.0625,
     3.725290298461914e-08, 0.005376344086021506, 1},
};

constexpr ProspectiveLayerNormBoundary prospectiveLayerNormBoundaries[] = {
    {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir", 7, 13},
    {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir", 2048, 4096},
};

constexpr mlir::shuttle::NumericalPolicy prospectiveLayerNormPolicies[] = {
    mlir::shuttle::NumericalPolicy::SourceOrdered,
    mlir::shuttle::NumericalPolicy::Fast,
};

bool prospectiveLayerNormMatrixIsExact(
    llvm::ArrayRef<ProspectiveLayerNormBoundary> boundaries,
    llvm::ArrayRef<ProspectiveLayerNormCell> cells,
    llvm::ArrayRef<mlir::shuttle::NumericalPolicy> policies) {
  if (boundaries.size() != 2 || cells.size() != 4 || policies.size() != 2)
    return false;
  struct ExpectedBoundary {
    int64_t rows;
    int64_t features;
    llvm::StringLiteral fixture;
  };
  constexpr ExpectedBoundary expectedBoundaries[] = {
      {7, 13, "jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir"},
      {2048, 4096,
       "jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir"},
  };
  for (const ExpectedBoundary &expected : expectedBoundaries) {
    if (llvm::count_if(boundaries, [&](const auto &boundary) {
          return boundary.rows == expected.rows &&
                 boundary.features == expected.features &&
                 boundary.fixture == expected.fixture;
        }) != 1)
      return false;
  }
  struct ExpectedCell {
    int64_t rows;
    int64_t features;
    ProspectiveLayerNormCase testCase;
  };
  constexpr ExpectedCell expectedCells[] = {
      {7, 13, ProspectiveLayerNormCase::Balanced},
      {2048, 4096, ProspectiveLayerNormCase::Balanced},
      {7, 13, ProspectiveLayerNormCase::PositiveOffset},
      {2048, 4096, ProspectiveLayerNormCase::PositiveOffset},
  };
  for (const ExpectedCell &expected : expectedCells) {
    if (llvm::count_if(cells, [&](const auto &cell) {
          return cell.rows == expected.rows &&
                 cell.features == expected.features &&
                 cell.testCase == expected.testCase;
        }) != 1)
      return false;
  }
  for (mlir::shuttle::NumericalPolicy expected :
       {mlir::shuttle::NumericalPolicy::SourceOrdered,
        mlir::shuttle::NumericalPolicy::Fast}) {
    if (std::count(policies.begin(), policies.end(), expected) != 1)
      return false;
  }
  int64_t executions = 0;
  for (const auto &boundary : boundaries)
    for (const auto &cell : cells)
      if (cell.rows == boundary.rows && cell.features == boundary.features)
        executions += policies.size();
  return executions == 8;
}

std::string prospectiveLayerNormCoordinate(
    const ProspectiveLayerNormCell &cell,
    mlir::shuttle::NumericalPolicy policy) {
  return llvm::formatv(
             "{0}x{1}_{2}_{3}", cell.rows, cell.features,
             cell.testCase == ProspectiveLayerNormCase::Balanced
                 ? "balanced"
                 : "positive_offset",
             policy == mlir::shuttle::NumericalPolicy::SourceOrdered
                 ? "source_ordered"
                 : "fast")
      .str();
}

uint32_t prospectiveLayerNormMix32(uint32_t index, uint32_t stream) {
  uint32_t value = index + UINT32_C(0x6A09E667) + stream * UINT32_C(0x9E3779B9);
  value ^= value >> 16;
  value *= UINT32_C(0x7FEB352D);
  value ^= value >> 15;
  value *= UINT32_C(0x846CA68B);
  value ^= value >> 16;
  return value;
}

LayerNormInputs
prospectiveLayerNormInputs(const ProspectiveLayerNormCell &cell) {
  LayerNormInputs inputs;
  inputs.x.resize(cell.rows * cell.features);
  inputs.gamma.resize(cell.features);
  inputs.beta.resize(cell.features);
  for (size_t index = 0; index < inputs.x.size(); ++index) {
    int32_t quantized =
        static_cast<int32_t>(prospectiveLayerNormMix32(index, 1) %
                             UINT32_C(513)) -
        256;
    inputs.x[index] = roundToBf16(static_cast<float>(quantized) / 128.0f);
  }
  const int32_t betaOrigin =
      cell.testCase == ProspectiveLayerNormCase::Balanced ? -64 : 1024;
  for (int64_t feature = 0; feature < cell.features; ++feature) {
    int32_t gammaQuantized =
        64 + static_cast<int32_t>(prospectiveLayerNormMix32(feature, 2) %
                                  UINT32_C(129));
    int32_t betaQuantized =
        betaOrigin + static_cast<int32_t>(
                         prospectiveLayerNormMix32(feature, 3) % UINT32_C(129));
    inputs.gamma[feature] =
        roundToBf16(static_cast<float>(gammaQuantized) / 128.0f);
    inputs.beta[feature] =
        roundToBf16(static_cast<float>(betaQuantized) / 128.0f);
  }
  return inputs;
}

std::string prospectiveBf16Digest(llvm::ArrayRef<uint16_t> values,
                                  llvm::ArrayRef<int64_t> shape) {
  std::string shapeText = "(";
  for (auto [ordinal, extent] : llvm::enumerate(shape)) {
    if (ordinal != 0)
      shapeText += ", ";
    shapeText += std::to_string(extent);
  }
  if (shape.size() == 1)
    shapeText += ",";
  shapeText += ")";
  llvm::SHA256 digest;
  constexpr llvm::StringLiteral dtype = "bfloat16";
  digest.update(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(dtype.data()), dtype.size()));
  digest.update(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(shapeText.data()), shapeText.size()));
  for (uint16_t value : values) {
    std::array<uint8_t, 2> bytes{static_cast<uint8_t>(value),
                                 static_cast<uint8_t>(value >> 8)};
    digest.update(bytes);
  }
  return llvm::toHex(digest.final(), true);
}

float prospectiveAdd(float left, float right) {
  volatile float result = left + right;
  return result;
}

float prospectiveSubtract(float left, float right) {
  volatile float result = left - right;
  return result;
}

float prospectiveMultiply(float left, float right) {
  volatile float result = left * right;
  return result;
}

float prospectiveDivide(float numerator, float denominator) {
  volatile float result = numerator / denominator;
  return result;
}

float prospectiveSqrt(float value) {
  volatile float result = std::sqrt(value);
  return result;
}

std::vector<float> prospectiveScheduledFold(llvm::ArrayRef<float> values,
                                            int64_t rows, int64_t features) {
  assert(values.size() == static_cast<size_t>(rows * features));
  const int64_t tile = std::min<int64_t>(features, 256);
  const int64_t threads = (tile + 31) / 32 * 32;
  const int64_t serialTiles = (features + tile - 1) / tile;
  std::vector<float> result(rows);
  std::vector<float> partials(threads);
  for (int64_t row = 0; row < rows; ++row) {
    std::fill(partials.begin(), partials.end(), 0.0f);
    for (int64_t thread = 0; thread < threads; ++thread) {
      for (int64_t leaf = 0; leaf < serialTiles; ++leaf) {
        const int64_t feature = thread * serialTiles + leaf;
        if (feature < features) {
          partials[thread] = prospectiveAdd(partials[thread],
                                            values[row * features + feature]);
        }
      }
    }
    int64_t active = threads;
    while (active > 1) {
      const int64_t pairs = active / 2;
      for (int64_t pair = 0; pair < pairs; ++pair) {
        partials[pair] =
            prospectiveAdd(partials[pair * 2], partials[pair * 2 + 1]);
      }
      if (active % 2 != 0)
        partials[pairs] = partials[active - 1];
      active = (active + 1) / 2;
    }
    result[row] = partials[0];
  }
  return result;
}

std::vector<uint16_t>
prospectiveScheduledLayerNorm(const LayerNormInputs &inputs, int64_t rows,
                              int64_t features) {
  std::vector<float> values(inputs.x.size());
  for (auto [index, bits] : llvm::enumerate(inputs.x))
    values[index] = expandBf16(bits);
  std::vector<float> sums = prospectiveScheduledFold(values, rows, features);
  std::vector<float> means(rows);
  std::vector<float> centered(values.size());
  std::vector<float> squared(values.size());
  for (int64_t row = 0; row < rows; ++row) {
    means[row] = prospectiveDivide(sums[row], static_cast<float>(features));
    for (int64_t feature = 0; feature < features; ++feature) {
      const int64_t element = row * features + feature;
      centered[element] = prospectiveSubtract(values[element], means[row]);
      squared[element] =
          prospectiveMultiply(centered[element], centered[element]);
    }
  }
  std::vector<float> squaredSums =
      prospectiveScheduledFold(squared, rows, features);
  std::vector<uint16_t> output(values.size());
  for (int64_t row = 0; row < rows; ++row) {
    float variance =
        prospectiveDivide(squaredSums[row], static_cast<float>(features));
    float root = prospectiveSqrt(prospectiveAdd(variance, 1.0e-5f));
    float inverse = prospectiveDivide(1.0f, root);
    for (int64_t feature = 0; feature < features; ++feature) {
      const int64_t element = row * features + feature;
      float value = prospectiveMultiply(centered[element], inverse);
      value = prospectiveMultiply(value, expandBf16(inputs.gamma[feature]));
      value = prospectiveAdd(value, expandBf16(inputs.beta[feature]));
      output[element] = roundToBf16(value);
    }
  }
  return output;
}

double prospectiveBinary64Epsilon() {
  return static_cast<double>(1.0e-5f);
}

std::vector<uint16_t>
prospectiveBinary64LayerNorm(const LayerNormInputs &inputs, int64_t rows,
                             int64_t features,
                             std::vector<double> *unroundedOutput = nullptr) {
  std::vector<uint16_t> output(inputs.x.size());
  if (unroundedOutput)
    unroundedOutput->assign(inputs.x.size(), 0.0);
  for (int64_t row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (int64_t feature = 0; feature < features; ++feature)
      sum += expandBf16(inputs.x[row * features + feature]);
    double mean = sum / static_cast<double>(features);
    double squaredSum = 0.0;
    for (int64_t feature = 0; feature < features; ++feature) {
      double centered =
          static_cast<double>(expandBf16(inputs.x[row * features + feature])) -
          mean;
      squaredSum += centered * centered;
    }
    double inverse = 1.0 / std::sqrt(squaredSum / static_cast<double>(features) +
                                     prospectiveBinary64Epsilon());
    for (int64_t feature = 0; feature < features; ++feature) {
      const int64_t element = row * features + feature;
      double centered =
          static_cast<double>(expandBf16(inputs.x[element])) - mean;
      double value =
          centered * inverse *
              static_cast<double>(expandBf16(inputs.gamma[feature])) +
          static_cast<double>(expandBf16(inputs.beta[feature]));
      if (unroundedOutput)
        (*unroundedOutput)[element] = value;
      output[element] = roundToBf16(static_cast<float>(value));
    }
  }
  return output;
}

struct ProspectiveLayerNormMetrics {
  double maxAbsoluteError = 0.0;
  double meanAbsoluteError = 0.0;
  double relativeLinfError = 0.0;
  int32_t maxBf16UlpError = 0;
  size_t maxAbsoluteElement = 0;
  size_t maxUlpElement = 0;
};

std::optional<ProspectiveLayerNormMetrics>
prospectiveLayerNormMetrics(llvm::ArrayRef<uint16_t> actual,
                            llvm::ArrayRef<uint16_t> reference) {
  if (actual.size() != reference.size() || actual.empty())
    return std::nullopt;
  ProspectiveLayerNormMetrics metrics;
  double absoluteSum = 0.0;
  double referenceScale = 1.0 / 128.0;
  for (size_t element = 0; element < actual.size(); ++element) {
    double actualValue = expandBf16(actual[element]);
    double referenceValue = expandBf16(reference[element]);
    if (!std::isfinite(actualValue) || !std::isfinite(referenceValue))
      return std::nullopt;
    double absolute = std::abs(actualValue - referenceValue);
    absoluteSum += absolute;
    referenceScale = std::max(referenceScale, std::abs(referenceValue));
    if (absolute > metrics.maxAbsoluteError) {
      metrics.maxAbsoluteError = absolute;
      metrics.maxAbsoluteElement = element;
    }
    int32_t ulp = std::abs(orderedBf16(actual[element]) -
                           orderedBf16(reference[element]));
    if (ulp > metrics.maxBf16UlpError) {
      metrics.maxBf16UlpError = ulp;
      metrics.maxUlpElement = element;
    }
  }
  metrics.meanAbsoluteError = absoluteSum / actual.size();
  metrics.relativeLinfError = metrics.maxAbsoluteError / referenceScale;
  return metrics;
}

void expectProspectiveLayerNormMetrics(
    const ProspectiveLayerNormMetrics &metrics,
    const ProspectiveLayerNormCell &cell) {
  const bool balanced = cell.testCase == ProspectiveLayerNormCase::Balanced;
  EXPECT_LE(metrics.maxAbsoluteError, balanced ? 0.015625 : 0.0625)
      << "element " << metrics.maxAbsoluteElement;
  EXPECT_LE(metrics.meanAbsoluteError, 1.0e-6);
  EXPECT_LE(metrics.relativeLinfError, 0.0078125);
  EXPECT_LE(metrics.maxBf16UlpError, balanced ? 16 : 1)
      << "element " << metrics.maxUlpElement;
}

TEST(GpuTransportV2Test,
     ProspectiveLayerNormBinary64OraclePromotesPublicF32Epsilon) {
  LayerNormInputs inputs{
      {roundToBf16(0.0f), roundToBf16(2.0f)},
      {roundToBf16(1.0f), roundToBf16(1.0f)},
      {roundToBf16(0.0f), roundToBf16(0.0f)},
  };
  std::vector<double> unrounded;
  prospectiveBinary64LayerNorm(inputs, 1, 2, &unrounded);
  const double promotedF32 =
      -1.0 / std::sqrt(1.0 + static_cast<double>(1.0e-5f));
  const double nativeBinary64 = -1.0 / std::sqrt(1.0 + 1.0e-5);
  ASSERT_EQ(unrounded.size(), 2u);
  EXPECT_NE(promotedF32, nativeBinary64);
  EXPECT_DOUBLE_EQ(unrounded[0], promotedF32);
}

TEST(GpuTransportV2Test,
     ProspectiveLayerNormMatrixRejectsRemovalAndDuplication) {
  EXPECT_TRUE(prospectiveLayerNormMatrixIsExact(
      prospectiveLayerNormBoundaries, prospectiveLayerNormCells,
      prospectiveLayerNormPolicies));

  std::vector<ProspectiveLayerNormBoundary> boundaries(
      std::begin(prospectiveLayerNormBoundaries),
      std::end(prospectiveLayerNormBoundaries));
  boundaries.pop_back();
  EXPECT_FALSE(prospectiveLayerNormMatrixIsExact(
      boundaries, prospectiveLayerNormCells, prospectiveLayerNormPolicies));
  boundaries.assign(std::begin(prospectiveLayerNormBoundaries),
                    std::end(prospectiveLayerNormBoundaries));
  boundaries[1] = boundaries[0];
  EXPECT_FALSE(prospectiveLayerNormMatrixIsExact(
      boundaries, prospectiveLayerNormCells, prospectiveLayerNormPolicies));

  std::vector<ProspectiveLayerNormCell> cells(
      std::begin(prospectiveLayerNormCells),
      std::end(prospectiveLayerNormCells));
  cells.pop_back();
  EXPECT_FALSE(prospectiveLayerNormMatrixIsExact(
      prospectiveLayerNormBoundaries, cells, prospectiveLayerNormPolicies));
  cells.assign(std::begin(prospectiveLayerNormCells),
               std::end(prospectiveLayerNormCells));
  cells[3] = cells[0];
  EXPECT_FALSE(prospectiveLayerNormMatrixIsExact(
      prospectiveLayerNormBoundaries, cells, prospectiveLayerNormPolicies));

  std::vector<mlir::shuttle::NumericalPolicy> policies(
      std::begin(prospectiveLayerNormPolicies),
      std::end(prospectiveLayerNormPolicies));
  policies.pop_back();
  EXPECT_FALSE(prospectiveLayerNormMatrixIsExact(
      prospectiveLayerNormBoundaries, prospectiveLayerNormCells, policies));
  policies.assign(std::begin(prospectiveLayerNormPolicies),
                  std::end(prospectiveLayerNormPolicies));
  policies[1] = policies[0];
  EXPECT_FALSE(prospectiveLayerNormMatrixIsExact(
      prospectiveLayerNormBoundaries, prospectiveLayerNormCells, policies));
}

TEST(GpuTransportV2Test,
     ProspectiveLayerNormForwardContractMatchesSealedHostModels) {
  for (const ProspectiveLayerNormCell &cell : prospectiveLayerNormCells) {
    SCOPED_TRACE(
        llvm::formatv("{0}x{1} {2}", cell.rows, cell.features,
                      cell.testCase == ProspectiveLayerNormCase::Balanced
                          ? "balanced"
                          : "positive_offset")
            .str());
    LayerNormInputs inputs = prospectiveLayerNormInputs(cell);
    EXPECT_EQ(prospectiveBf16Digest(inputs.x, {cell.rows, cell.features}),
              cell.xDigest);
    EXPECT_EQ(prospectiveBf16Digest(inputs.gamma, {cell.features}),
              cell.gammaDigest);
    EXPECT_EQ(prospectiveBf16Digest(inputs.beta, {cell.features}),
              cell.betaDigest);
    std::vector<uint16_t> binary64 =
        prospectiveBinary64LayerNorm(inputs, cell.rows, cell.features);
    std::vector<uint16_t> scheduled =
        prospectiveScheduledLayerNorm(inputs, cell.rows, cell.features);
    EXPECT_EQ(prospectiveBf16Digest(binary64, {cell.rows, cell.features}),
              cell.binary64Digest);
    EXPECT_EQ(prospectiveBf16Digest(scheduled, {cell.rows, cell.features}),
              cell.scheduleDigest);
    auto metrics = prospectiveLayerNormMetrics(scheduled, binary64);
    ASSERT_TRUE(metrics);
    EXPECT_EQ(metrics->maxAbsoluteError, cell.scheduleMaxAbsoluteError);
    EXPECT_EQ(metrics->meanAbsoluteError, cell.scheduleMeanAbsoluteError);
    EXPECT_EQ(metrics->relativeLinfError, cell.scheduleRelativeLinfError);
    EXPECT_EQ(metrics->maxBf16UlpError, cell.scheduleMaxBf16UlpError);
    expectProspectiveLayerNormMetrics(*metrics, cell);
  }
}

TEST(GpuTransportV2Test, H100RunnerExecutesThreeMapBf16Chain) {
  if (std::getenv("SHUTTLE_RUN_H100_NUMERICS") == nullptr)
    GTEST_SKIP() << "set SHUTTLE_RUN_H100_NUMERICS=1 on an H100 worker";

  constexpr llvm::StringLiteral source = R"mlir(
module @three_map_bf16_chain {
  func.func public @main(%arg0: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<2x3xbf16>) -> tensor<2x3xf32>
    %1 = stablehlo.negate %0 : tensor<2x3xf32>
    %2 = stablehlo.convert %1 : (tensor<2x3xf32>) -> tensor<2x3xbf16>
    return %2 : tensor<2x3xbf16>
  }
}
)mlir";
  constexpr llvm::StringLiteral options =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral tuning =
      R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
  auto built = buildFixture(source, mlir::shuttle::NumericalPolicy::SourceOrdered,
                            options, tuning);
  ASSERT_TRUE(built);
  auto transport = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));

  auto platform = stream_executor::PlatformManager::PlatformWithName("CUDA");
  ASSERT_TRUE(platform.ok()) << platform.status();
  auto executor = (*platform)->ExecutorForDevice(0);
  ASSERT_TRUE(executor.ok()) << executor.status();
  ASSERT_NE((*executor)->GetDeviceDescription().name().find("H100"),
            std::string::npos);
  auto stream = (*executor)->CreateStream();
  ASSERT_TRUE(stream.ok()) << stream.status();
  stream_executor::StreamExecutorAddressAllocator allocator(*executor);
  const auto &capability =
      (*executor)->GetDeviceDescription().gpu_compute_capability();
  auto runner = mlir::shuttle::GpuRunner::Create(
      *transport, &allocator, 0, stream->get(), &capability);
  ASSERT_TRUE(runner.ok()) << runner.status();
  ASSERT_EQ((*runner)->resources().launchCount, 3);

  std::vector<uint16_t> input = {
      roundToBf16(-3.0f), roundToBf16(-1.5f), roundToBf16(-0.5f),
      roundToBf16(0.5f),  roundToBf16(1.5f),  roundToBf16(3.0f),
  };
  std::vector<uint16_t> actual(input.size(), UINT16_C(0x7FC1));
  auto deviceInput = (*executor)->AllocateArray<uint16_t>(input.size());
  auto deviceResult = (*executor)->AllocateArray<uint16_t>(actual.size());
  ASSERT_TRUE(deviceInput);
  ASSERT_TRUE(deviceResult);
  ASSERT_TRUE((*executor)
                  ->SynchronousMemcpyH2D(input.data(),
                                         input.size() * sizeof(uint16_t),
                                         &deviceInput)
                  .ok());
  ASSERT_TRUE((*executor)
                  ->SynchronousMemcpyH2D(actual.data(),
                                         actual.size() * sizeof(uint16_t),
                                         &deviceResult)
                  .ok());
  llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> operands{{
      deviceInput.opaque(), static_cast<int64_t>(deviceInput.size())}};
  llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> results{{
      deviceResult.opaque(), static_cast<int64_t>(deviceResult.size())}};

  ASSERT_TRUE((*runner)->Execute(operands, results).ok());
  ASSERT_TRUE((*runner)->AwaitCompletion().ok());
  ASSERT_TRUE((*executor)
                  ->SynchronousMemcpyD2H(deviceResult,
                                         actual.size() * sizeof(uint16_t),
                                         actual.data())
                  .ok());
  for (auto [index, actualBits] : llvm::enumerate(actual))
    EXPECT_EQ(actualBits, roundToBf16(-expandBf16(input[index])))
        << "element " << index;

  runner->reset();
  (*executor)->Deallocate(&deviceResult);
  (*executor)->Deallocate(&deviceInput);
}

TEST(GpuTransportV2Test, H100RunnerMatchesIndependentLayerNormForwardOracle) {
  if (std::getenv("SHUTTLE_RUN_H100_NUMERICS") == nullptr)
    GTEST_SKIP() << "set SHUTTLE_RUN_H100_NUMERICS=1 on an H100 worker";

  auto platform = stream_executor::PlatformManager::PlatformWithName("CUDA");
  ASSERT_TRUE(platform.ok()) << platform.status();
  auto executor = (*platform)->ExecutorForDevice(0);
  ASSERT_TRUE(executor.ok()) << executor.status();
  ASSERT_NE((*executor)->GetDeviceDescription().name().find("H100"),
            std::string::npos);
  auto stream = (*executor)->CreateStream();
  ASSERT_TRUE(stream.ok()) << stream.status();
  stream_executor::StreamExecutorAddressAllocator allocator(*executor);
  const auto &capability =
      (*executor)->GetDeviceDescription().gpu_compute_capability();

  constexpr llvm::StringLiteral sourceOrderedOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral fastOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral tuning =
      R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
  struct Boundary {
    llvm::StringLiteral fixture;
    int64_t rows;
    int64_t features;
  };
  constexpr Boundary boundaries[] = {
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir", 7, 13},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir", 2048, 4096},
  };
  for (const Boundary &boundary : boundaries) {
    SCOPED_TRACE(boundary.fixture.str());
    LayerNormInputs inputs = layerNormInputs(boundary.rows, boundary.features);
    LayerNormForwardOracle oracle =
        layerNormForwardReference(inputs, boundary.rows, boundary.features);
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      SCOPED_TRACE(policy == mlir::shuttle::NumericalPolicy::SourceOrdered
                       ? "source_ordered"
                       : "fast");
      auto source = fixtureSource(boundary.fixture);
      ASSERT_TRUE(source);
      auto built =
          buildFixture(*source, policy,
                       policy == mlir::shuttle::NumericalPolicy::SourceOrdered
                           ? sourceOrderedOptions
                           : fastOptions,
                       tuning);
      ASSERT_TRUE(built);
      auto transport =
          mlir::shuttle::serializeGpuExecutableBundle(*built->module);
      ASSERT_TRUE(mlir::succeeded(transport));
      auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
      ASSERT_TRUE(executable.ok()) << executable.status();
      const auto &bindings = (*executable)->externalBindings();
      ASSERT_EQ(bindings.size(), 4u);
      constexpr mlir::shuttle::ExecutableBindingKind kinds[] = {
          mlir::shuttle::ExecutableBindingKind::Operand,
          mlir::shuttle::ExecutableBindingKind::Operand,
          mlir::shuttle::ExecutableBindingKind::Operand,
          mlir::shuttle::ExecutableBindingKind::Result,
      };
      const int64_t bindingBytes[] = {
          static_cast<int64_t>(inputs.x.size() * sizeof(uint16_t)),
          static_cast<int64_t>(inputs.gamma.size() * sizeof(uint16_t)),
          static_cast<int64_t>(inputs.beta.size() * sizeof(uint16_t)),
          static_cast<int64_t>(oracle.output.size() * sizeof(uint16_t)),
      };
      for (auto [ordinal, binding] : llvm::enumerate(bindings)) {
        EXPECT_EQ(binding.kind, kinds[ordinal]);
        EXPECT_EQ(binding.index, ordinal == 3 ? 0 : ordinal);
        EXPECT_EQ(binding.requiredBytes, bindingBytes[ordinal]);
      }

      auto runner = mlir::shuttle::GpuRunner::Create(
          *transport, &allocator, 0, stream->get(), &capability);
      ASSERT_TRUE(runner.ok()) << runner.status();
      const mlir::shuttle::GpuRunnerResources &resources =
          (*runner)->resources();
      EXPECT_EQ(resources.launchCount, 1);
      EXPECT_EQ(resources.externalOperandCount, 3);
      EXPECT_EQ(resources.externalResultCount, 1);
      EXPECT_EQ(resources.temporaryAllocationCount, 0);
      EXPECT_EQ(resources.temporaryBytes, 0);

      const std::vector<const std::vector<uint16_t> *> hostOperands = {
          &inputs.x, &inputs.gamma, &inputs.beta};
      std::vector<stream_executor::DeviceAddressBase> deviceOperands;
      llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> operands;
      for (const std::vector<uint16_t> *host : hostOperands) {
        auto device = (*executor)->AllocateArray<uint16_t>(host->size());
        ASSERT_TRUE(device);
        ASSERT_TRUE((*executor)
                        ->SynchronousMemcpyH2D(host->data(),
                                               host->size() * sizeof(uint16_t),
                                               &device)
                        .ok());
        deviceOperands.push_back(device);
        operands.push_back(
            {device.opaque(), static_cast<int64_t>(device.size())});
      }
      std::vector<uint16_t> actual(oracle.output.size(), UINT16_C(0x7FC1));
      auto deviceResult = (*executor)->AllocateArray<uint16_t>(actual.size());
      ASSERT_TRUE(deviceResult);
      ASSERT_TRUE((*executor)
                      ->SynchronousMemcpyH2D(actual.data(),
                                             actual.size() * sizeof(uint16_t),
                                             &deviceResult)
                      .ok());
      llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> results{
          {deviceResult.opaque(), static_cast<int64_t>(deviceResult.size())}};
      absl::Status execution = (*runner)->Execute(operands, results);
      ASSERT_TRUE(execution.ok()) << execution;
      ASSERT_TRUE((*runner)->AwaitCompletion().ok());
      ASSERT_TRUE((*executor)
                      ->SynchronousMemcpyD2H(deviceResult,
                                             actual.size() * sizeof(uint16_t),
                                             actual.data())
                      .ok());

      int64_t poisoned = 0;
      int64_t nonfinite = 0;
      int32_t maximumUlpDistance = 0;
      size_t maximumUlpElement = 0;
      for (auto [element, actualBits] : llvm::enumerate(actual)) {
        poisoned += actualBits == UINT16_C(0x7FC1);
        nonfinite += !std::isfinite(expandBf16(actualBits));
        int32_t ulpDistance = std::abs(orderedBf16(actualBits) -
                                       orderedBf16(oracle.output[element]));
        if (ulpDistance > maximumUlpDistance) {
          maximumUlpDistance = ulpDistance;
          maximumUlpElement = element;
        }
      }
      EXPECT_EQ(poisoned, 0);
      EXPECT_EQ(nonfinite, 0);
      const int32_t allowedUlps =
          policy == mlir::shuttle::NumericalPolicy::SourceOrdered ? 1 : 4;
      EXPECT_LE(maximumUlpDistance, allowedUlps)
          << "element " << maximumUlpElement << ", actual "
          << expandBf16(actual[maximumUlpElement]) << ", expected "
          << expandBf16(oracle.output[maximumUlpElement]);

      runner->reset();
      (*executor)->Deallocate(&deviceResult);
      for (auto &device : deviceOperands)
        (*executor)->Deallocate(&device);
    }
  }
}

TEST(GpuTransportV2Test,
     H100RunnerMatchesProspectiveLayerNormForwardContractV1) {
  if (std::getenv("SHUTTLE_RUN_H100_NUMERICS") == nullptr)
    GTEST_SKIP() << "set SHUTTLE_RUN_H100_NUMERICS=1 on an H100 worker";

  auto platform = stream_executor::PlatformManager::PlatformWithName("CUDA");
  ASSERT_TRUE(platform.ok()) << platform.status();
  auto executor = (*platform)->ExecutorForDevice(0);
  ASSERT_TRUE(executor.ok()) << executor.status();
  ASSERT_NE((*executor)->GetDeviceDescription().name().find("H100"),
            std::string::npos);
  auto stream = (*executor)->CreateStream();
  ASSERT_TRUE(stream.ok()) << stream.status();
  stream_executor::StreamExecutorAddressAllocator allocator(*executor);
  const auto &capability =
      (*executor)->GetDeviceDescription().gpu_compute_capability();

  constexpr llvm::StringLiteral sourceOrderedOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral fastOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral tuning =
      R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
  ASSERT_TRUE(prospectiveLayerNormMatrixIsExact(
      prospectiveLayerNormBoundaries, prospectiveLayerNormCells,
      prospectiveLayerNormPolicies));
  std::vector<std::string> executedCoordinates;
  for (const ProspectiveLayerNormBoundary &boundary :
       prospectiveLayerNormBoundaries) {
    SCOPED_TRACE(boundary.fixture.str());
    for (mlir::shuttle::NumericalPolicy policy :
         prospectiveLayerNormPolicies) {
      const llvm::StringRef policyName =
          policy == mlir::shuttle::NumericalPolicy::SourceOrdered
              ? "source_ordered"
              : "fast";
      SCOPED_TRACE(policyName.str());
      auto source = fixtureSource(boundary.fixture);
      ASSERT_TRUE(source);
      auto built =
          buildFixture(*source, policy,
                       policy == mlir::shuttle::NumericalPolicy::SourceOrdered
                           ? sourceOrderedOptions
                           : fastOptions,
                       tuning);
      ASSERT_TRUE(built);
      auto transport =
          mlir::shuttle::serializeGpuExecutableBundle(*built->module);
      ASSERT_TRUE(mlir::succeeded(transport));
      auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
      ASSERT_TRUE(executable.ok()) << executable.status();
      const auto &bindings = (*executable)->externalBindings();
      ASSERT_EQ(bindings.size(), 4u);
      constexpr mlir::shuttle::ExecutableBindingKind kinds[] = {
          mlir::shuttle::ExecutableBindingKind::Operand,
          mlir::shuttle::ExecutableBindingKind::Operand,
          mlir::shuttle::ExecutableBindingKind::Operand,
          mlir::shuttle::ExecutableBindingKind::Result,
      };
      const int64_t bindingBytes[] = {
          boundary.rows * boundary.features *
              static_cast<int64_t>(sizeof(uint16_t)),
          boundary.features * static_cast<int64_t>(sizeof(uint16_t)),
          boundary.features * static_cast<int64_t>(sizeof(uint16_t)),
          boundary.rows * boundary.features *
              static_cast<int64_t>(sizeof(uint16_t)),
      };
      for (auto [ordinal, binding] : llvm::enumerate(bindings)) {
        EXPECT_EQ(binding.kind, kinds[ordinal]);
        EXPECT_EQ(binding.index, ordinal == 3 ? 0 : ordinal);
        EXPECT_EQ(binding.requiredBytes, bindingBytes[ordinal]);
      }

      auto runner = mlir::shuttle::GpuRunner::Create(
          *transport, &allocator, 0, stream->get(), &capability);
      ASSERT_TRUE(runner.ok()) << runner.status();
      const mlir::shuttle::GpuRunnerResources &resources =
          (*runner)->resources();
      EXPECT_EQ(resources.launchCount, 1);
      EXPECT_EQ(resources.externalOperandCount, 3);
      EXPECT_EQ(resources.externalResultCount, 1);
      EXPECT_EQ(resources.temporaryAllocationCount, 0);
      EXPECT_EQ(resources.temporaryBytes, 0);

      for (const ProspectiveLayerNormCell &cell : prospectiveLayerNormCells) {
        if (cell.rows != boundary.rows || cell.features != boundary.features)
          continue;
        const llvm::StringRef caseName =
            cell.testCase == ProspectiveLayerNormCase::Balanced
                ? "balanced"
                : "positive_offset";
        SCOPED_TRACE(caseName.str());
        LayerNormInputs inputs = prospectiveLayerNormInputs(cell);
        std::vector<uint16_t> scheduled =
            prospectiveScheduledLayerNorm(inputs, cell.rows, cell.features);
        std::vector<uint16_t> binary64 =
            prospectiveBinary64LayerNorm(inputs, cell.rows, cell.features);
        ASSERT_EQ(prospectiveBf16Digest(inputs.x, {cell.rows, cell.features}),
                  cell.xDigest);
        ASSERT_EQ(prospectiveBf16Digest(inputs.gamma, {cell.features}),
                  cell.gammaDigest);
        ASSERT_EQ(prospectiveBf16Digest(inputs.beta, {cell.features}),
                  cell.betaDigest);
        ASSERT_EQ(prospectiveBf16Digest(binary64, {cell.rows, cell.features}),
                  cell.binary64Digest);
        ASSERT_EQ(prospectiveBf16Digest(scheduled, {cell.rows, cell.features}),
                  cell.scheduleDigest);

        const std::vector<const std::vector<uint16_t> *> hostOperands = {
            &inputs.x, &inputs.gamma, &inputs.beta};
        std::vector<stream_executor::DeviceAddressBase> deviceOperands;
        llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> operands;
        for (const std::vector<uint16_t> *host : hostOperands) {
          auto device = (*executor)->AllocateArray<uint16_t>(host->size());
          ASSERT_TRUE(device);
          ASSERT_TRUE(
              (*executor)
                  ->SynchronousMemcpyH2D(
                      host->data(), host->size() * sizeof(uint16_t), &device)
                  .ok());
          deviceOperands.push_back(device);
          operands.push_back(
              {device.opaque(), static_cast<int64_t>(device.size())});
        }
        std::vector<std::vector<uint16_t>> actualRuns(
            2, std::vector<uint16_t>(scheduled.size()));
        auto deviceResult =
            (*executor)->AllocateArray<uint16_t>(scheduled.size());
        ASSERT_TRUE(deviceResult);
        llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> results{
            {deviceResult.opaque(), static_cast<int64_t>(deviceResult.size())}};
        for (std::vector<uint16_t> &actual : actualRuns) {
          std::fill(actual.begin(), actual.end(), UINT16_C(0x7FC1));
          ASSERT_TRUE((*executor)
                          ->SynchronousMemcpyH2D(
                              actual.data(), actual.size() * sizeof(uint16_t),
                              &deviceResult)
                          .ok());
          absl::Status execution = (*runner)->Execute(operands, results);
          ASSERT_TRUE(execution.ok()) << execution;
          ASSERT_TRUE((*runner)->AwaitCompletion().ok());
          ASSERT_TRUE((*executor)
                          ->SynchronousMemcpyD2H(
                              deviceResult, actual.size() * sizeof(uint16_t),
                              actual.data())
                          .ok());
        }

        int64_t repeatabilityDifferences = 0;
        size_t firstRepeatabilityDifference = 0;
        int64_t scheduleDifferences = 0;
        size_t firstScheduleDifference = 0;
        for (size_t element = 0; element < scheduled.size(); ++element) {
          if (actualRuns[0][element] != actualRuns[1][element]) {
            if (repeatabilityDifferences == 0)
              firstRepeatabilityDifference = element;
            ++repeatabilityDifferences;
          }
          if (actualRuns[0][element] != scheduled[element]) {
            if (scheduleDifferences == 0)
              firstScheduleDifference = element;
            ++scheduleDifferences;
          }
        }
        EXPECT_EQ(repeatabilityDifferences, 0)
            << "first element " << firstRepeatabilityDifference;
        EXPECT_EQ(scheduleDifferences, 0)
            << "first element " << firstScheduleDifference << ", actual "
            << expandBf16(actualRuns[0][firstScheduleDifference])
            << ", scheduled " << expandBf16(scheduled[firstScheduleDifference]);
        auto metrics = prospectiveLayerNormMetrics(actualRuns[0], binary64);
        ASSERT_TRUE(metrics)
            << "all actual and reference outputs must be finite";
        expectProspectiveLayerNormMetrics(*metrics, cell);
        executedCoordinates.push_back(prospectiveLayerNormCoordinate(cell,
                                                                      policy));
        RecordProperty(
            llvm::formatv("metrics_{0}x{1}_{2}_{3}", cell.rows, cell.features,
                          caseName, policyName)
                .str(),
            llvm::formatv(
                "max_absolute={0},mean_absolute={1},relative_linf={2},"
                "max_bf16_ulp={3},schedule_differences={4},"
                "repeatability_differences={5}",
                metrics->maxAbsoluteError, metrics->meanAbsoluteError,
                metrics->relativeLinfError, metrics->maxBf16UlpError,
                scheduleDifferences, repeatabilityDifferences)
                .str());

        (*executor)->Deallocate(&deviceResult);
        for (auto &device : deviceOperands)
          (*executor)->Deallocate(&device);
      }
      runner->reset();
    }
  }
  std::sort(executedCoordinates.begin(), executedCoordinates.end());
  const std::vector<std::string> expectedCoordinates = {
      "2048x4096_balanced_fast",
      "2048x4096_balanced_source_ordered",
      "2048x4096_positive_offset_fast",
      "2048x4096_positive_offset_source_ordered",
      "7x13_balanced_fast",
      "7x13_balanced_source_ordered",
      "7x13_positive_offset_fast",
      "7x13_positive_offset_source_ordered",
  };
  EXPECT_EQ(executedCoordinates, expectedCoordinates);
}

std::vector<uint16_t> rmsNormForwardReference(
    llvm::ArrayRef<uint16_t> x, llvm::ArrayRef<uint16_t> weight,
    int64_t rows, int64_t features) {
  std::vector<uint16_t> result(x.size());
  for (int64_t row = 0; row < rows; ++row) {
    float squareSum = 0.0f;
    for (int64_t feature = 0; feature < features; ++feature) {
      float value = expandBf16(x[row * features + feature]);
      squareSum += value * value;
    }
    float inverseRoot =
        1.0f / std::sqrt(squareSum / static_cast<float>(features) + 1e-6f);
    for (int64_t feature = 0; feature < features; ++feature) {
      float value = expandBf16(x[row * features + feature]);
      float scale = expandBf16(weight[feature]);
      result[row * features + feature] =
          roundToBf16(value * inverseRoot * scale);
    }
  }
  return result;
}

std::pair<std::vector<uint16_t>, std::vector<uint16_t>>
rmsNormBackwardReference(llvm::ArrayRef<uint16_t> x,
                         llvm::ArrayRef<uint16_t> weight,
                         llvm::ArrayRef<uint16_t> cotangent, int64_t rows,
                         int64_t features) {
  std::vector<uint16_t> inputCotangent(x.size());
  std::vector<float> weightCotangent(features, 0.0f);
  for (int64_t row = 0; row < rows; ++row) {
    float squareSum = 0.0f;
    for (int64_t feature = 0; feature < features; ++feature) {
      float value = expandBf16(x[row * features + feature]);
      squareSum += value * value;
    }
    float inverseRoot =
        1.0f / std::sqrt(squareSum / static_cast<float>(features) + 1e-6f);
    float innerProduct = 0.0f;
    for (int64_t feature = 0; feature < features; ++feature) {
      float value = expandBf16(x[row * features + feature]);
      float upstream = expandBf16(cotangent[row * features + feature]);
      float scale = expandBf16(weight[feature]);
      innerProduct += value * upstream * scale;
      weightCotangent[feature] += upstream * value * inverseRoot;
    }
    float derivativeScale = inverseRoot * inverseRoot * inverseRoot /
                            static_cast<float>(features);
    for (int64_t feature = 0; feature < features; ++feature) {
      float value = expandBf16(x[row * features + feature]);
      float upstream = expandBf16(cotangent[row * features + feature]);
      float scale = expandBf16(weight[feature]);
      inputCotangent[row * features + feature] = roundToBf16(
          upstream * scale * inverseRoot -
          value * derivativeScale * innerProduct);
    }
  }
  std::vector<uint16_t> roundedWeightCotangent(features);
  for (int64_t feature = 0; feature < features; ++feature)
    roundedWeightCotangent[feature] =
        roundToBf16(weightCotangent[feature]);
  return {std::move(inputCotangent), std::move(roundedWeightCotangent)};
}

TEST(GpuTransportV2Test,
     H100RunnerMatchesIndependentRmsNormForwardAndVjpOracle) {
  if (std::getenv("SHUTTLE_RUN_H100_NUMERICS") == nullptr)
    GTEST_SKIP() << "set SHUTTLE_RUN_H100_NUMERICS=1 on an H100 worker";

  auto platform = stream_executor::PlatformManager::PlatformWithName("CUDA");
  ASSERT_TRUE(platform.ok()) << platform.status();
  auto executor = (*platform)->ExecutorForDevice(0);
  ASSERT_TRUE(executor.ok()) << executor.status();
  ASSERT_NE((*executor)->GetDeviceDescription().name().find("H100"),
            std::string::npos);
  auto stream = (*executor)->CreateStream();
  ASSERT_TRUE(stream.ok()) << stream.status();
  stream_executor::StreamExecutorAddressAllocator allocator(*executor);
  const auto &capability =
      (*executor)->GetDeviceDescription().gpu_compute_capability();

  constexpr int64_t rows = 14;
  constexpr int64_t features = 19;
  std::vector<uint16_t> x(rows * features);
  std::vector<uint16_t> weight(features);
  std::vector<uint16_t> cotangent(rows * features);
  for (size_t index = 0; index < x.size(); ++index) {
    if (index < features) {
      x[index] = roundToBf16(-0.001f + 0.002f * index / (features - 1));
    } else {
      x[index] =
          roundToBf16(-0.75f + 1.625f * index / (x.size() - 1));
    }
  }
  for (size_t index = 0; index < weight.size(); ++index)
    weight[index] =
        roundToBf16(0.5f + 0.75f * index / (weight.size() - 1));
  for (size_t index = 0; index < cotangent.size(); ++index)
    cotangent[index] =
        roundToBf16(-0.5f + 1.125f * index / (cotangent.size() - 1));

  constexpr llvm::StringLiteral sourceOrderedOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral fastOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral tuning =
      R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
  for (bool backward : {false, true}) {
    SCOPED_TRACE(backward ? "VJP" : "forward");
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      SCOPED_TRACE(policy == mlir::shuttle::NumericalPolicy::SourceOrdered
                       ? "source_ordered"
                       : "fast");
    auto source = fixtureSource(
        backward
            ? "jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-"
              "backward.mlir"
            : "jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-"
              "forward.mlir");
    ASSERT_TRUE(source);
    auto built = buildFixture(
        *source, policy,
        policy == mlir::shuttle::NumericalPolicy::SourceOrdered
            ? sourceOrderedOptions
            : fastOptions,
        tuning);
    ASSERT_TRUE(built);
    auto transport =
        mlir::shuttle::serializeGpuExecutableBundle(*built->module);
    ASSERT_TRUE(mlir::succeeded(transport));
    auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
    ASSERT_TRUE(executable.ok()) << executable.status();
    llvm::SmallVector<int64_t> resultBytes;
    for (const auto &binding : (*executable)->externalBindings())
      if (binding.kind == mlir::shuttle::ExecutableBindingKind::Result)
        resultBytes.push_back(binding.requiredBytes);
    llvm::SmallVector<int64_t> expectedResultBytes;
    if (backward)
      expectedResultBytes.push_back(weight.size() * sizeof(uint16_t));
    expectedResultBytes.push_back(x.size() * sizeof(uint16_t));
    ASSERT_EQ(resultBytes, expectedResultBytes);
    auto runner = mlir::shuttle::GpuRunner::Create(
        *transport, &allocator, 0, stream->get(), &capability);
    ASSERT_TRUE(runner.ok()) << runner.status();
    EXPECT_EQ((*runner)->resources().launchCount, backward ? 3 : 1);

    std::vector<std::vector<uint16_t>> hostOperands{x, weight};
    if (backward)
      hostOperands.push_back(cotangent);
    std::vector<std::vector<uint16_t>> hostResults;
    if (backward) {
      hostResults.emplace_back(weight.size());
      hostResults.emplace_back(x.size());
    } else {
      hostResults.emplace_back(x.size());
    }
    std::vector<stream_executor::DeviceAddressBase> deviceOperands;
    std::vector<stream_executor::DeviceAddressBase> deviceResults;
    llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> operands;
    llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> results;
    for (const auto &host : hostOperands) {
      auto device = (*executor)->AllocateArray<uint16_t>(host.size());
      ASSERT_TRUE(device);
      ASSERT_TRUE((*executor)
                      ->SynchronousMemcpyH2D(
                          host.data(), host.size() * sizeof(uint16_t), &device)
                      .ok());
      deviceOperands.push_back(device);
      operands.push_back(
          {device.opaque(), static_cast<int64_t>(device.size())});
    }
    for (auto &host : hostResults) {
      auto device = (*executor)->AllocateArray<uint16_t>(host.size());
      ASSERT_TRUE(device);
      std::vector<uint16_t> poison(host.size(), UINT16_C(0x7FC1));
      ASSERT_TRUE((*executor)
                      ->SynchronousMemcpyH2D(
                          poison.data(), poison.size() * sizeof(uint16_t),
                          &device)
                      .ok());
      deviceResults.push_back(device);
      results.push_back(
          {device.opaque(), static_cast<int64_t>(device.size())});
    }
    absl::Status execution = (*runner)->Execute(operands, results);
    ASSERT_TRUE(execution.ok()) << execution;
    ASSERT_TRUE((*runner)->AwaitCompletion().ok());
    for (auto [index, host] : llvm::enumerate(hostResults))
      ASSERT_TRUE((*executor)
                      ->SynchronousMemcpyD2H(
                          deviceResults[index], host.size() * sizeof(uint16_t),
                          host.data())
                      .ok());

    std::vector<std::vector<uint16_t>> expected;
    if (backward) {
      auto [dx, dw] = rmsNormBackwardReference(x, weight, cotangent, rows,
                                               features);
      expected = {std::move(dw), std::move(dx)};
    } else {
      expected = {rmsNormForwardReference(x, weight, rows, features)};
    }
    ASSERT_EQ(hostResults.size(), expected.size());
    for (auto [resultIndex, actual] : llvm::enumerate(hostResults)) {
      ASSERT_EQ(actual.size(), expected[resultIndex].size());
      for (auto [element, actualBits] : llvm::enumerate(actual)) {
        EXPECT_NE(actualBits, UINT16_C(0x7FC1));
        float actualValue = expandBf16(actualBits);
        EXPECT_TRUE(std::isfinite(actualValue));
        uint16_t expectedBits = expected[resultIndex][element];
        int32_t ulpDistance =
            std::abs(orderedBf16(actualBits) - orderedBf16(expectedBits));
        int32_t maximumUlps =
            policy == mlir::shuttle::NumericalPolicy::SourceOrdered ? 1 : 4;
        EXPECT_LE(ulpDistance, maximumUlps)
            << "result " << resultIndex << " element " << element;
      }
    }
    runner->reset();
    for (auto &device : deviceResults)
      (*executor)->Deallocate(&device);
    for (auto &device : deviceOperands)
      (*executor)->Deallocate(&device);
    }
  }
}

std::vector<uint16_t> rmsNormContractReference(
    llvm::ArrayRef<uint16_t> x, llvm::ArrayRef<uint16_t> weight,
    llvm::ArrayRef<uint16_t> projection, int64_t rows, int64_t reduction,
    int64_t columns, bool fast, float epsilon = 1e-6f) {
  std::vector<uint16_t> result(rows * columns);
  for (int64_t row = 0; row < rows; ++row) {
    std::vector<float> squareLeaves(reduction);
    for (int64_t index = 0; index < reduction; ++index) {
      float value = expandBf16(x[row * reduction + index]);
      squareLeaves[index] = value * value;
    }
    while (squareLeaves.size() > 1) {
      std::vector<float> compacted((squareLeaves.size() + 1) / 2);
      for (size_t index = 0; index < compacted.size(); ++index) {
        compacted[index] = squareLeaves[index * 2];
        if (index * 2 + 1 < squareLeaves.size())
          compacted[index] += squareLeaves[index * 2 + 1];
      }
      squareLeaves = std::move(compacted);
    }
    float inverseRoot =
        1.0f / std::sqrt(squareLeaves.front() /
                             static_cast<float>(reduction) +
                         epsilon);
    for (int64_t column = 0; column < columns; ++column) {
      std::array<float, 32> accumulators{};
      for (int64_t index = 0; index < reduction; ++index) {
        float normalized = expandBf16(roundToBf16(
            expandBf16(x[row * reduction + index]) * inverseRoot *
            expandBf16(weight[index])));
        int64_t lane = fast ? index % 32 : 0;
        accumulators[lane] =
            std::fma(normalized,
                     expandBf16(projection[index * columns + column]),
                     accumulators[lane]);
      }
      if (fast) {
        for (int64_t offset : {16, 8, 4, 2, 1}) {
          std::array<float, 32> previous = accumulators;
          for (int64_t lane = 0; lane + offset < 32; ++lane)
            accumulators[lane] = previous[lane] + previous[lane + offset];
        }
      }
      result[row * columns + column] = roundToBf16(accumulators.front());
    }
  }
  return result;
}

TEST(GpuTransportV2Test,
     H100RunnerMatchesIndependentRmsNormContractOracle) {
  if (std::getenv("SHUTTLE_RUN_H100_NUMERICS") == nullptr)
    GTEST_SKIP() << "set SHUTTLE_RUN_H100_NUMERICS=1 on an H100 worker";

  auto platform = stream_executor::PlatformManager::PlatformWithName("CUDA");
  ASSERT_TRUE(platform.ok()) << platform.status();
  auto executor = (*platform)->ExecutorForDevice(0);
  ASSERT_TRUE(executor.ok()) << executor.status();
  ASSERT_NE((*executor)->GetDeviceDescription().name().find("H100"),
            std::string::npos);
  auto stream = (*executor)->CreateStream();
  ASSERT_TRUE(stream.ok()) << stream.status();
  stream_executor::StreamExecutorAddressAllocator allocator(*executor);
  const auto &capability =
      (*executor)->GetDeviceDescription().gpu_compute_capability();

  constexpr llvm::StringLiteral sourceOrderedOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral fastOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral tuning =
      R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
  struct Boundary {
    const char *filename;
    int64_t columns;
    int64_t launches;
    int64_t temporaryGroups;
    int64_t temporaryBytes;
  };
  constexpr Boundary boundaries[] = {
      {"jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir", 7, 1, 0, 0},
      {"jax-0.10.1-bf16-rmsnorm_contract-5x19x257-forward.mlir", 257, 2, 1,
       20},
  };
  constexpr int64_t rows = 5;
  constexpr int64_t reduction = 19;
  for (const Boundary &boundary : boundaries) {
    SCOPED_TRACE(boundary.filename);
    auto source = fixtureSource(boundary.filename);
    ASSERT_TRUE(source);
    std::vector<uint16_t> x(rows * reduction);
    std::vector<uint16_t> weight(reduction);
    std::vector<uint16_t> projection(reduction * boundary.columns);
    for (size_t index = 0; index < x.size(); ++index) {
      x[index] = index < reduction
                     ? roundToBf16(-0.001f + 0.002f * index / (reduction - 1))
                     : roundToBf16(-0.75f +
                                   1.625f * index / (x.size() - 1));
    }
    for (size_t index = 0; index < weight.size(); ++index)
      weight[index] =
          roundToBf16(0.5f + 0.75f * index / (weight.size() - 1));
    for (size_t index = 0; index < projection.size(); ++index)
      projection[index] = roundToBf16(
          -0.25f + 0.625f * index / (projection.size() - 1));

    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      const bool fast = policy == mlir::shuttle::NumericalPolicy::Fast;
      SCOPED_TRACE(fast ? "fast" : "source_ordered");
      std::vector<uint16_t> expected = rmsNormContractReference(
          x, weight, projection, rows, reduction, boundary.columns, fast);
      std::vector<uint16_t> withoutEpsilon = rmsNormContractReference(
          x, weight, projection, rows, reduction, boundary.columns, fast,
          0.0f);
      EXPECT_TRUE(llvm::any_of(llvm::zip_equal(expected, withoutEpsilon),
                               [](auto values) {
                                 return std::abs(orderedBf16(std::get<0>(values)) -
                                                 orderedBf16(std::get<1>(values))) >
                                        4;
                               }));

      auto built = buildFixture(
          *source, policy,
          fast ? fastOptions : sourceOrderedOptions, tuning);
      ASSERT_TRUE(built);
      auto transport =
          mlir::shuttle::serializeGpuExecutableBundle(*built->module);
      ASSERT_TRUE(mlir::succeeded(transport));
      auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
      ASSERT_TRUE(executable.ok()) << executable.status();
      if (boundary.launches == 2) {
        ASSERT_EQ((*executable)->launches().size(), 2u);
        EXPECT_TRUE((*executable)->launches()[0].dependencies.empty());
        EXPECT_EQ((*executable)->launches()[1].dependencies,
                  std::vector<int64_t>({0}));
      }
      auto runner = mlir::shuttle::GpuRunner::Create(
          *transport, &allocator, 0, stream->get(), &capability);
      ASSERT_TRUE(runner.ok()) << runner.status();
      EXPECT_EQ((*runner)->resources().launchCount, boundary.launches);
      EXPECT_EQ((*runner)->resources().temporaryAllocationCount,
                boundary.temporaryGroups);
      EXPECT_EQ((*runner)->resources().temporaryBytes,
                boundary.temporaryBytes);

      std::vector<std::vector<uint16_t>> hostOperands{x, weight, projection};
      std::vector<uint16_t> actual(expected.size());
      std::vector<stream_executor::DeviceAddressBase> deviceOperands;
      llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> operands;
      for (const auto &host : hostOperands) {
        auto device = (*executor)->AllocateArray<uint16_t>(host.size());
        ASSERT_TRUE(device);
        ASSERT_TRUE((*executor)
                        ->SynchronousMemcpyH2D(
                            host.data(), host.size() * sizeof(uint16_t),
                            &device)
                        .ok());
        operands.push_back(
            {device.opaque(), static_cast<int64_t>(device.size())});
        deviceOperands.push_back(device);
      }
      auto deviceResult = (*executor)->AllocateArray<uint16_t>(actual.size());
      ASSERT_TRUE(deviceResult);
      std::vector<uint16_t> poison(actual.size(), UINT16_C(0x7FC1));
      ASSERT_TRUE((*executor)
                      ->SynchronousMemcpyH2D(
                          poison.data(), poison.size() * sizeof(uint16_t),
                          &deviceResult)
                      .ok());
      llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> results{{
          deviceResult.opaque(), static_cast<int64_t>(deviceResult.size())}};
      ASSERT_TRUE((*runner)->Execute(operands, results).ok());
      ASSERT_TRUE((*runner)->AwaitCompletion().ok());
      ASSERT_TRUE((*executor)
                      ->SynchronousMemcpyD2H(deviceResult,
                                             actual.size() * sizeof(uint16_t),
                                             actual.data())
                      .ok());
      for (auto [element, actualBits] : llvm::enumerate(actual)) {
        EXPECT_NE(actualBits, UINT16_C(0x7FC1));
        EXPECT_TRUE(std::isfinite(expandBf16(actualBits)));
        int32_t ulpDistance = std::abs(orderedBf16(actualBits) -
                                       orderedBf16(expected[element]));
        EXPECT_LE(ulpDistance, fast ? 4 : 1) << "element " << element;
      }
      runner->reset();
      (*executor)->Deallocate(&deviceResult);
      for (auto &device : deviceOperands)
        (*executor)->Deallocate(&device);
    }
  }
}

TEST(GpuTransportV2Test, H100RunnerWritesCounterbalancedPerformanceReport) {
  const char *runPerformance =
      std::getenv("SHUTTLE_RUN_H100_PERFORMANCE");
  if (runPerformance == nullptr || llvm::StringRef(runPerformance) != "1")
    GTEST_SKIP() << "set SHUTTLE_RUN_H100_PERFORMANCE=1 on an H100 worker";
  const char *reportPath = std::getenv("SHUTTLE_H100_PERF_REPORT");
  const char *compilerIdentity = std::getenv("SHUTTLE_COMPILER_IDENTITY");
  ASSERT_NE(reportPath, nullptr);
  ASSERT_NE(compilerIdentity, nullptr);
  ASSERT_FALSE(llvm::StringRef(compilerIdentity).empty());

  auto platform = stream_executor::PlatformManager::PlatformWithName("CUDA");
  ASSERT_TRUE(platform.ok()) << platform.status();
  auto executor = (*platform)->ExecutorForDevice(0);
  ASSERT_TRUE(executor.ok()) << executor.status();
  const std::string deviceName = (*executor)->GetDeviceDescription().name();
  ASSERT_NE(deviceName.find("H100"), std::string::npos);
  auto stream = (*executor)->CreateStream();
  ASSERT_TRUE(stream.ok()) << stream.status();
  stream_executor::StreamExecutorAddressAllocator allocator(*executor);
  const auto &capability =
      (*executor)->GetDeviceDescription().gpu_compute_capability();

  constexpr llvm::StringLiteral sourceOrderedOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral fastOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral tuning =
      R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
  constexpr int64_t warmupIterations = 3;
  constexpr int64_t measuredIterations = 20;
  constexpr int64_t normRows = 14;
  constexpr int64_t normFeatures = 19;

  struct PerformanceSpec {
    const char *caseName;
    const char *fixture;
    std::vector<int64_t> shape;
    bool backward;
    int64_t columns;
    uint64_t launches;
    uint64_t slots;
    uint64_t operands;
    uint64_t results;
    uint64_t temporaryAllocations;
    uint64_t temporaryBytes;
  };
  const PerformanceSpec specs[] = {
      {"rmsnorm_forward",
       "jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir",
       {2, 7, 19}, false, 0, 1, 3, 2, 1, 0, 0},
      {"rmsnorm_backward",
       "jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir",
       {2, 7, 19}, true, 0, 3, 6, 3, 2, 1, 56},
      {"rmsnorm_contract_inline",
       "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir",
       {5, 19, 7}, false, 7, 1, 4, 3, 1, 0, 0},
      {"rmsnorm_contract_compact",
       "jax-0.10.1-bf16-rmsnorm_contract-5x19x257-forward.mlir",
       {5, 19, 257}, false, 257, 2, 5, 3, 1, 1, 20},
  };

  struct PerformanceCell {
    const PerformanceSpec *spec;
    mlir::shuttle::NumericalPolicy policy;
    std::string fixture;
    std::vector<uint8_t> transport;
    std::shared_ptr<const mlir::shuttle::GpuExecutable> executable;
    std::unique_ptr<mlir::shuttle::GpuRunner> runner;
    mlir::shuttle::GpuRunnerResources resources{};
    std::vector<std::vector<uint16_t>> hostOperands;
    std::vector<std::vector<uint16_t>> expected;
    std::vector<std::vector<uint16_t>> actual;
    std::vector<stream_executor::DeviceAddressBase> deviceOperands;
    std::vector<stream_executor::DeviceAddressBase> deviceResults;
    llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> operandBindings;
    llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> resultBindings;
    std::vector<int64_t> samples;
    std::vector<int64_t> hostSamples;
  };
  std::vector<PerformanceCell> cells;

  std::vector<uint16_t> normX(normRows * normFeatures);
  std::vector<uint16_t> normWeight(normFeatures);
  std::vector<uint16_t> normCotangent(normRows * normFeatures);
  for (size_t index = 0; index < normX.size(); ++index) {
    normX[index] = index < normFeatures
                       ? roundToBf16(-0.001f +
                                     0.002f * index / (normFeatures - 1))
                       : roundToBf16(-0.75f +
                                     1.625f * index / (normX.size() - 1));
    normCotangent[index] =
        roundToBf16(-0.5f + 1.125f * index / (normX.size() - 1));
  }
  for (size_t index = 0; index < normWeight.size(); ++index)
    normWeight[index] =
        roundToBf16(0.5f + 0.75f * index / (normWeight.size() - 1));

  for (const PerformanceSpec &spec : specs) {
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      PerformanceCell cell;
      cell.spec = &spec;
      cell.policy = policy;
      auto source = fixtureSource(spec.fixture);
      ASSERT_TRUE(source);
      cell.fixture = *source;
      auto built = buildFixture(
          *source, policy,
          policy == mlir::shuttle::NumericalPolicy::SourceOrdered
              ? sourceOrderedOptions
              : fastOptions,
          tuning);
      ASSERT_TRUE(built);
      auto serialized =
          mlir::shuttle::serializeGpuExecutableBundle(*built->module);
      ASSERT_TRUE(mlir::succeeded(serialized));
      cell.transport.assign(serialized->begin(), serialized->end());
      auto executable = mlir::shuttle::GpuExecutable::Load(cell.transport);
      ASSERT_TRUE(executable.ok()) << executable.status();
      cell.executable = *executable;
      auto runner = mlir::shuttle::GpuRunner::Create(
          cell.transport, &allocator, 0, stream->get(), &capability);
      ASSERT_TRUE(runner.ok()) << runner.status();
      cell.runner = std::move(*runner);
      cell.resources = cell.runner->resources();
      EXPECT_EQ(cell.resources.launchCount, spec.launches);
      EXPECT_EQ(cell.resources.slotCount, spec.slots);
      EXPECT_EQ(cell.resources.externalOperandCount, spec.operands);
      EXPECT_EQ(cell.resources.externalResultCount, spec.results);
      EXPECT_EQ(cell.resources.temporaryAllocationCount,
                spec.temporaryAllocations);
      EXPECT_EQ(cell.resources.temporaryBytes, spec.temporaryBytes);

      if (spec.columns == 0) {
        cell.hostOperands = {normX, normWeight};
        if (spec.backward) {
          cell.hostOperands.push_back(normCotangent);
          auto [dx, dw] = rmsNormBackwardReference(
              normX, normWeight, normCotangent, normRows, normFeatures);
          cell.expected = {std::move(dw), std::move(dx)};
        } else {
          cell.expected = {rmsNormForwardReference(normX, normWeight, normRows,
                                                   normFeatures)};
        }
      } else {
        constexpr int64_t rows = 5;
        constexpr int64_t reduction = 19;
        std::vector<uint16_t> x(rows * reduction);
        std::vector<uint16_t> weight(reduction);
        std::vector<uint16_t> projection(reduction * spec.columns);
        for (size_t index = 0; index < x.size(); ++index)
          x[index] = index < reduction
                         ? roundToBf16(-0.001f +
                                       0.002f * index / (reduction - 1))
                         : roundToBf16(-0.75f +
                                       1.625f * index / (x.size() - 1));
        for (size_t index = 0; index < weight.size(); ++index)
          weight[index] =
              roundToBf16(0.5f + 0.75f * index / (weight.size() - 1));
        for (size_t index = 0; index < projection.size(); ++index)
          projection[index] = roundToBf16(
              -0.25f + 0.625f * index / (projection.size() - 1));
        cell.hostOperands = {x, weight, projection};
        cell.expected = {rmsNormContractReference(
            x, weight, projection, rows, reduction, spec.columns,
            policy == mlir::shuttle::NumericalPolicy::Fast)};
      }
      size_t resultBindingIndex = 0;
      for (const auto &binding : cell.executable->externalBindings()) {
        if (binding.kind != mlir::shuttle::ExecutableBindingKind::Result)
          continue;
        ASSERT_LT(resultBindingIndex, cell.expected.size());
        EXPECT_EQ(binding.requiredBytes,
                  static_cast<int64_t>(cell.expected[resultBindingIndex].size() *
                                       sizeof(uint16_t)));
        ++resultBindingIndex;
      }
      ASSERT_EQ(resultBindingIndex, cell.expected.size());
      for (const auto &expected : cell.expected)
        cell.actual.emplace_back(expected.size());
      for (const auto &host : cell.hostOperands) {
        auto device = (*executor)->AllocateArray<uint16_t>(host.size());
        ASSERT_TRUE(device);
        ASSERT_TRUE((*executor)
                        ->SynchronousMemcpyH2D(
                            host.data(), host.size() * sizeof(uint16_t),
                            &device)
                        .ok());
        cell.operandBindings.push_back(
            {device.opaque(), static_cast<int64_t>(device.size())});
        cell.deviceOperands.push_back(device);
      }
      for (auto &actual : cell.actual) {
        auto device = (*executor)->AllocateArray<uint16_t>(actual.size());
        ASSERT_TRUE(device);
        std::vector<uint16_t> poison(actual.size(), UINT16_C(0x7FC1));
        ASSERT_TRUE((*executor)
                        ->SynchronousMemcpyH2D(
                            poison.data(), poison.size() * sizeof(uint16_t),
                            &device)
                        .ok());
        cell.resultBindings.push_back(
            {device.opaque(), static_cast<int64_t>(device.size())});
        cell.deviceResults.push_back(device);
      }
      cells.push_back(std::move(cell));
    }
  }
  ASSERT_EQ(cells.size(), 8u);

  for (PerformanceCell &cell : cells) {
    for (int64_t iteration = 0; iteration < warmupIterations; ++iteration) {
      absl::Status execution =
          cell.runner->Execute(cell.operandBindings, cell.resultBindings);
      ASSERT_TRUE(execution.ok())
          << execution << " " << cell.spec->caseName << " "
          << mlir::shuttle::stringifyNumericalPolicy(cell.policy).str();
    }
  }
  for (int64_t round = 0; round < measuredIterations; ++round) {
    for (size_t position = 0; position < cells.size(); ++position) {
      size_t cellIndex = round % 2 == 0 ? position : cells.size() - 1 - position;
      PerformanceCell &cell = cells[cellIndex];
      auto elapsed =
          cell.runner->ExecuteTimed(cell.operandBindings, cell.resultBindings);
      ASSERT_TRUE(elapsed.ok()) << elapsed.status();
      cell.samples.push_back(*elapsed);
    }
  }
  for (PerformanceCell &cell : cells)
    for (int64_t iteration = 0; iteration < warmupIterations; ++iteration)
      ASSERT_TRUE(cell.runner
                      ->Execute(cell.operandBindings, cell.resultBindings)
                      .ok());
  for (int64_t round = 0; round < measuredIterations; ++round) {
    for (size_t position = 0; position < cells.size(); ++position) {
      size_t cellIndex = round % 2 == 0 ? position : cells.size() - 1 - position;
      PerformanceCell &cell = cells[cellIndex];
      auto start = std::chrono::steady_clock::now();
      ASSERT_TRUE(cell.runner
                      ->Execute(cell.operandBindings, cell.resultBindings)
                      .ok());
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
      ASSERT_GT(elapsed, 0);
      cell.hostSamples.push_back(elapsed);
    }
  }

  llvm::json::Array reportCells;
  for (auto [cellIndex, cell] : llvm::enumerate(cells)) {
    llvm::StringRef policyName =
        cell.policy == mlir::shuttle::NumericalPolicy::SourceOrdered
            ? "source_ordered"
            : "fast";
    float maximumAbsoluteError = 0.0f;
    int32_t maximumUlp = 0;
    llvm::json::Array outputDigests;
    for (auto [resultIndex, actual] : llvm::enumerate(cell.actual)) {
      ASSERT_TRUE((*executor)
                      ->SynchronousMemcpyD2H(
                          cell.deviceResults[resultIndex],
                          actual.size() * sizeof(uint16_t), actual.data())
                      .ok());
      ASSERT_EQ(actual.size(), cell.expected[resultIndex].size());
      for (auto [element, actualBits] : llvm::enumerate(actual)) {
        EXPECT_NE(actualBits, UINT16_C(0x7FC1));
        float actualValue = expandBf16(actualBits);
        float expectedValue = expandBf16(cell.expected[resultIndex][element]);
        EXPECT_TRUE(std::isfinite(actualValue));
        maximumAbsoluteError =
            std::max(maximumAbsoluteError,
                     std::abs(actualValue - expectedValue));
        maximumUlp =
            std::max(maximumUlp,
                     std::abs(orderedBf16(actualBits) -
                              orderedBf16(cell.expected[resultIndex][element])));
      }
      auto bytes = llvm::ArrayRef<uint8_t>(
          reinterpret_cast<const uint8_t *>(actual.data()),
          actual.size() * sizeof(uint16_t));
      outputDigests.push_back(sha256(bytes));
    }
    EXPECT_LE(maximumUlp,
              cell.policy == mlir::shuttle::NumericalPolicy::SourceOrdered
                  ? 1
                  : 4);

    llvm::json::Array samples;
    llvm::json::Array hostSamples;
    llvm::json::Array rounds;
    llvm::json::Array positions;
    llvm::json::Array hostRounds;
    llvm::json::Array hostPositions;
    ASSERT_EQ(cell.samples.size(), measuredIterations);
    ASSERT_EQ(cell.hostSamples.size(), measuredIterations);
    for (int64_t round = 0; round < measuredIterations; ++round) {
      samples.push_back(cell.samples[round]);
      hostSamples.push_back(cell.hostSamples[round]);
      rounds.push_back(round);
      hostRounds.push_back(round);
      positions.push_back(round % 2 == 0
                              ? static_cast<int64_t>(cellIndex)
                              : static_cast<int64_t>(cells.size() - 1 -
                                                     cellIndex));
      hostPositions.push_back(round % 2 == 0
                                  ? static_cast<int64_t>(cellIndex)
                                  : static_cast<int64_t>(cells.size() - 1 -
                                                         cellIndex));
    }
    llvm::json::Array shape;
    for (int64_t extent : cell.spec->shape)
      shape.push_back(extent);
    llvm::json::Array ptxDigests;
    for (const mlir::shuttle::GpuLaunch &launch : cell.executable->launches())
      ptxDigests.push_back(launch.codeDigest);
    llvm::json::Object resources{
        {"code_bytes", cell.resources.codeBytes},
        {"launch_count", cell.resources.launchCount},
        {"slot_count", cell.resources.slotCount},
        {"external_operand_count", cell.resources.externalOperandCount},
        {"external_result_count", cell.resources.externalResultCount},
        {"temporary_allocation_count",
         cell.resources.temporaryAllocationCount},
        {"temporary_bytes", cell.resources.temporaryBytes},
    };
    llvm::json::Object output{
        {"digests", std::move(outputDigests)},
        {"max_abs_error", maximumAbsoluteError},
        {"max_bf16_ulp", maximumUlp},
    };
    llvm::json::Object identity{
        {"workload_digest",
         performanceWorkloadDigest(cell.spec->caseName, policyName,
                                   cell.spec->shape, cell.hostOperands)},
        {"fixture_digest", sha256(llvm::ArrayRef<uint8_t>(
                               reinterpret_cast<const uint8_t *>(
                                   cell.fixture.data()),
                               cell.fixture.size()))},
        {"executable_digest",
         mlir::shuttle::gpuExecutableBundleDigest(cell.transport)},
        {"ptx_digests", std::move(ptxDigests)},
        {"module_load_digest", sha256(cell.executable->codeBytes())},
    };
    reportCells.push_back(llvm::json::Object{
        {"case", cell.spec->caseName},
        {"policy", policyName},
        {"shape", std::move(shape)},
        {"samples_ns", std::move(samples)},
        {"host_samples_ns", std::move(hostSamples)},
        {"sample_rounds", std::move(rounds)},
        {"sample_positions", std::move(positions)},
        {"host_sample_rounds", std::move(hostRounds)},
        {"host_sample_positions", std::move(hostPositions)},
        {"resources", std::move(resources)},
        {"output", std::move(output)},
        {"identity", std::move(identity)},
    });
  }

  std::string options(sourceOrderedOptions);
  options.append(fastOptions);
  options.append(tuning);
  llvm::json::Object root{
      {"schema", "shuttle_h100_performance_v1"},
      {"producer", "shuttle"},
      {"device", llvm::json::Object{{"model", deviceName}, {"ordinal", 0}}},
      {"compiler",
       llvm::json::Object{
           {"identity", compilerIdentity},
           {"options_digest",
            sha256(llvm::ArrayRef<uint8_t>(
                reinterpret_cast<const uint8_t *>(options.data()),
                options.size()))}}},
      {"protocol",
       llvm::json::Object{
           {"clock", "cuda_event"},
           {"counterbalance", "forward_reverse_alternating"},
           {"warmup_iterations", warmupIterations},
           {"measured_iterations", measuredIterations}}},
      {"host_protocol",
       llvm::json::Object{
           {"mode", "exploratory_host_v1"},
           {"clock", "host_block_until_ready"},
           {"counterbalance", "forward_reverse_alternating"},
           {"warmup_iterations", warmupIterations},
           {"measured_iterations", measuredIterations}}},
      {"cells", std::move(reportCells)},
  };
  std::ofstream report(reportPath, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(report.good());
  report << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root))).str();
  report.close();
  ASSERT_TRUE(report.good());

  for (PerformanceCell &cell : cells) {
    cell.runner.reset();
    for (auto &device : cell.deviceResults)
      (*executor)->Deallocate(&device);
    for (auto &device : cell.deviceOperands)
      (*executor)->Deallocate(&device);
  }
}

TEST(GpuTransportV2Test,
     H100RepresentativeRmsNormRunnerWritesOneAuthenticatedRawCell) {
  const char *run = std::getenv("SHUTTLE_RUN_H100_REPRESENTATIVE");
  if (run == nullptr || llvm::StringRef(run) != "1")
    GTEST_SKIP() << "set SHUTTLE_RUN_H100_REPRESENTATIVE=1 on an H100 worker";
  const char *caseRoot = std::getenv("SHUTTLE_REP_CASE_ROOT");
  const char *rawRoot = std::getenv("SHUTTLE_REP_RAW_ROOT");
  const char *policyValue = std::getenv("SHUTTLE_REP_POLICY");
  const char *boundaryValue = std::getenv("SHUTTLE_REP_BOUNDARY");
  const char *repeatValue = std::getenv("SHUTTLE_REP_REPEAT");
  ASSERT_NE(caseRoot, nullptr);
  ASSERT_NE(rawRoot, nullptr);
  ASSERT_NE(policyValue, nullptr);
  ASSERT_NE(boundaryValue, nullptr);
  ASSERT_NE(repeatValue, nullptr);
  llvm::StringRef policyName(policyValue);
  llvm::StringRef boundary(boundaryValue);
  ASSERT_TRUE(policyName == "source_ordered" || policyName == "fast");
  ASSERT_TRUE(boundary == "forward" || boundary == "backward_recompute" ||
              boundary == "composed");
  int64_t repeat = -1;
  ASSERT_FALSE(llvm::StringRef(repeatValue).getAsInteger(10, repeat));
  ASSERT_TRUE(repeat == 0 || repeat == 1);

  llvm::StringRef fixture =
      boundary == "forward"
          ? "jax-0.10.1-bf16-projected_parameter_broadcast-2048x4096-"
            "forward.mlir"
      : boundary == "backward_recompute"
          ? "jax-0.10.1-bf16-projected_parameter_broadcast-2048x4096-"
            "backward.mlir"
          : "jax-0.10.1-bf16-projected_parameter_broadcast-2048x4096-"
            "composed.mlir";
  auto source = fixtureSource(fixture);
  ASSERT_TRUE(source)
      << "representative fixture unavailable: " << fixture
      << " RUNFILES_DIR="
      << (std::getenv("RUNFILES_DIR") == nullptr
              ? "<unset>"
              : std::getenv("RUNFILES_DIR"))
      << " TEST_SRCDIR="
      << (std::getenv("TEST_SRCDIR") == nullptr
              ? "<unset>"
              : std::getenv("TEST_SRCDIR"));
  constexpr llvm::StringLiteral sourceOrderedOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral fastOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  constexpr llvm::StringLiteral tuning =
      R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
  mlir::shuttle::NumericalPolicy policy =
      policyName == "source_ordered"
          ? mlir::shuttle::NumericalPolicy::SourceOrdered
          : mlir::shuttle::NumericalPolicy::Fast;
  auto built = buildFixture(
      *source, policy,
      policyName == "source_ordered" ? sourceOrderedOptions : fastOptions,
      tuning);
  ASSERT_TRUE(built);
  auto serialized = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(serialized));
  auto executable = mlir::shuttle::GpuExecutable::Load(*serialized);
  ASSERT_TRUE(executable.ok()) << executable.status();

  auto platform = stream_executor::PlatformManager::PlatformWithName("CUDA");
  ASSERT_TRUE(platform.ok()) << platform.status();
  auto executor = (*platform)->ExecutorForDevice(0);
  ASSERT_TRUE(executor.ok()) << executor.status();
  ASSERT_NE((*executor)->GetDeviceDescription().name().find("H100"),
            std::string::npos);
  auto stream = (*executor)->CreateStream();
  ASSERT_TRUE(stream.ok()) << stream.status();
  stream_executor::StreamExecutorAddressAllocator allocator(*executor);
  const auto &capability =
      (*executor)->GetDeviceDescription().gpu_compute_capability();
  auto runner = mlir::shuttle::GpuRunner::Create(*serialized, &allocator, 0,
                                                 stream->get(), &capability);
  ASSERT_TRUE(runner.ok()) << runner.status();

  constexpr int64_t rows = 2048;
  constexpr int64_t features = 4096;
  constexpr int64_t fullElements = rows * features;
  const std::filesystem::path cases =
      std::filesystem::path(caseRoot) / "heldout_splitmix64_v3";
  auto read = [&](llvm::StringRef name,
                  int64_t elements) -> std::vector<uint16_t> {
    std::vector<uint16_t> values(elements);
    std::ifstream input(cases / name.str(), std::ios::binary);
    EXPECT_TRUE(input.good());
    input.read(reinterpret_cast<char *>(values.data()),
               values.size() * sizeof(uint16_t));
    EXPECT_EQ(input.gcount(),
              static_cast<std::streamsize>(values.size() * sizeof(uint16_t)));
    EXPECT_EQ(input.peek(), std::ifstream::traits_type::eof());
    return values;
  };
  std::vector<std::vector<uint16_t>> hostOperands{read("x.bf16", fullElements),
                                                  read("gamma.bf16", features)};
  if (boundary != "forward")
    hostOperands.push_back(read("dy.bf16", fullElements));
  std::vector<int64_t> resultElements;
  if (boundary == "forward")
    resultElements = {fullElements};
  else if (boundary == "backward_recompute")
    resultElements = {fullElements, features};
  else
    resultElements = {fullElements, fullElements, features};

  std::vector<stream_executor::DeviceAddressBase> deviceOperands;
  std::vector<stream_executor::DeviceAddressBase> deviceResults;
  llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> operands;
  llvm::SmallVector<mlir::shuttle::GpuRunnerBuffer> results;
  for (const auto &host : hostOperands) {
    auto device = (*executor)->AllocateArray<uint16_t>(host.size());
    ASSERT_TRUE(device);
    ASSERT_TRUE((*executor)
                    ->SynchronousMemcpyH2D(
                        host.data(), host.size() * sizeof(uint16_t), &device)
                    .ok());
    operands.push_back({device.opaque(), static_cast<int64_t>(device.size())});
    deviceOperands.push_back(device);
  }
  std::vector<std::vector<uint16_t>> hostResults;
  for (int64_t elements : resultElements) {
    hostResults.emplace_back(elements);
    auto device = (*executor)->AllocateArray<uint16_t>(elements);
    ASSERT_TRUE(device);
    std::vector<uint16_t> poison(elements, UINT16_C(0x7FC1));
    ASSERT_TRUE((*executor)
                    ->SynchronousMemcpyH2D(poison.data(),
                                           poison.size() * sizeof(uint16_t),
                                           &device)
                    .ok());
    results.push_back({device.opaque(), static_cast<int64_t>(device.size())});
    deviceResults.push_back(device);
  }

  constexpr int64_t warmupIterations = 10;
  constexpr int64_t measuredIterations = 20;
  for (int64_t iteration = 0; iteration < warmupIterations; ++iteration)
    ASSERT_TRUE((*runner)->Execute(operands, results).ok());
  llvm::json::Array hostSamples;
  for (int64_t iteration = 0; iteration < measuredIterations; ++iteration) {
    auto start = std::chrono::steady_clock::now();
    ASSERT_TRUE((*runner)->Execute(operands, results).ok());
    int64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    ASSERT_GT(elapsed, 0);
    hostSamples.push_back(elapsed);
  }
  llvm::json::Array eventSamples;
  for (int64_t iteration = 0; iteration < measuredIterations; ++iteration) {
    auto elapsed = (*runner)->ExecuteTimed(operands, results);
    ASSERT_TRUE(elapsed.ok()) << elapsed.status();
    ASSERT_GT(*elapsed, 0);
    eventSamples.push_back(*elapsed);
  }

  const std::filesystem::path outputDirectory =
      std::filesystem::path(rawRoot) / policyName.str() / boundary.str() /
      ("repeat-" + std::to_string(repeat));
  ASSERT_TRUE(std::filesystem::create_directories(outputDirectory) ||
              std::filesystem::is_directory(outputDirectory));
  llvm::json::Array externalBindings;
  llvm::StringRef operandRoles[] = {"x", "gamma", "dy"};
  llvm::SmallVector<llvm::StringRef> resultRoles;
  if (boundary == "forward")
    resultRoles = {"y"};
  else if (boundary == "backward_recompute")
    resultRoles = {"dx", "dgamma"};
  else
    resultRoles = {"y", "dx", "dgamma"};
  size_t resultIndex = 0;
  for (const auto &binding : (*executable)->externalBindings()) {
    llvm::json::Object record{
        {"kind", binding.kind == mlir::shuttle::ExecutableBindingKind::Operand
                     ? "operand"
                     : "result"},
        {"runner_index", binding.index},
        {"slot_ordinal", binding.slotOrdinal},
        {"required_bytes", binding.requiredBytes},
        {"tensor_type", binding.requiredBytes == features * 2
                            ? "bf16[4096]"
                            : "bf16[2048,4096]"},
        {"alignment", binding.alignment},
    };
    if (binding.kind == mlir::shuttle::ExecutableBindingKind::Operand) {
      ASSERT_LT(binding.index, static_cast<int64_t>(std::size(operandRoles)));
      record["role"] = operandRoles[binding.index];
    } else {
      ASSERT_LT(resultIndex, hostResults.size());
      ASSERT_EQ(binding.index, static_cast<int64_t>(resultIndex));
      record["role"] = resultRoles[resultIndex];
      record["public_result_index"] = binding.index;
      ASSERT_TRUE((*executor)
                      ->SynchronousMemcpyD2H(deviceResults[resultIndex],
                                             hostResults[resultIndex].size() *
                                                 sizeof(uint16_t),
                                             hostResults[resultIndex].data())
                      .ok());
      std::filesystem::path outputPath =
          outputDirectory /
          ("result-" + std::to_string(binding.index) + ".bf16");
      std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
      ASSERT_TRUE(output.good());
      output.write(
          reinterpret_cast<const char *>(hostResults[resultIndex].data()),
          hostResults[resultIndex].size() * sizeof(uint16_t));
      output.close();
      ASSERT_TRUE(output.good());
      auto bytes = llvm::ArrayRef<uint8_t>(
          reinterpret_cast<const uint8_t *>(hostResults[resultIndex].data()),
          hostResults[resultIndex].size() * sizeof(uint16_t));
      record["artifact"] = llvm::json::Object{
          {"path", outputPath.string()},
          {"byte_count", static_cast<int64_t>(bytes.size())},
          {"sha256", sha256(bytes)},
      };
      ++resultIndex;
    }
    externalBindings.push_back(std::move(record));
  }
  ASSERT_EQ(resultIndex, hostResults.size());

  const auto &resources = (*runner)->resources();
  llvm::json::Array ptxDigests;
  for (const auto &launch : (*executable)->launches())
    ptxDigests.push_back(launch.codeDigest);
  llvm::json::Object raw{
      {"schema", "shuttle_abi22_rmsnorm_representative_raw_cell_v1"},
      {"policy", policyName},
      {"boundary", boundary},
      {"repeat", repeat},
      {"runner_api",
       llvm::json::Object{
           {"host", "GpuRunner::Execute_BlockHostUntilDone"},
           {"cuda_event", "GpuRunner::ExecuteTimed"},
       }},
      {"external_bindings", std::move(externalBindings)},
      {"resources",
       llvm::json::Object{
           {"launch_count", resources.launchCount},
           {"slot_count", resources.slotCount},
           {"external_operand_count", resources.externalOperandCount},
           {"external_result_count", resources.externalResultCount},
           {"temporary_allocation_count", resources.temporaryAllocationCount},
           {"temporary_bytes", resources.temporaryBytes},
       }},
      {"transport",
       llvm::json::Object{
           {"contract_operation_count", 0},
           {"code_bytes",
            static_cast<int64_t>((*executable)->codeBytes().size())},
           {"executable_digest",
            mlir::shuttle::gpuExecutableBundleDigest(*serialized)},
           {"module_load_digest", sha256((*executable)->codeBytes())},
           {"ptx_digests", std::move(ptxDigests)},
       }},
      {"timing",
       llvm::json::Object{
           {"warmup_iterations", warmupIterations},
           {"measured_iterations", measuredIterations},
           {"host_clock", "steady_clock_execute_block_host_until_done_ns"},
           {"host_samples_ns", std::move(hostSamples)},
           {"cuda_event_status", "supplemental_not_acceptance"},
           {"cuda_event_samples_ns", std::move(eventSamples)},
       }},
  };
  std::ofstream rawOutput(outputDirectory / "cell.json",
                          std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(rawOutput.good());
  rawOutput << llvm::formatv("{0}\n", llvm::json::Value(std::move(raw))).str();
  rawOutput.close();
  ASSERT_TRUE(rawOutput.good());

  runner->reset();
  for (auto &device : deviceResults)
    (*executor)->Deallocate(&device);
  for (auto &device : deviceOperands)
    (*executor)->Deallocate(&device);
}

TEST(GpuTransportV2Test,
     ExportedFfiBundleExecutesDynamicLayerNormProjectionAndLaunchCount) {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-composed.mlir");
  ASSERT_TRUE(source);
  auto built = buildFixture(*source);
  ASSERT_TRUE(built);
  auto transport = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));
  auto decoded = decodeInlineSchema2(*transport);
  ASSERT_TRUE(decoded);
  ASSERT_EQ(decoded->launches.size(), 80);
  auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
  ASSERT_TRUE(executable.ok()) << executable.status();
  ASSERT_EQ((*executable)->externalBindings().size(), 8);
  auto dynamicFrame =
      dynamicCallFrame(**executable, *transport,
                       mlir::shuttle::gpuExecutableBundleDigest(*transport));
  ASSERT_TRUE(dynamicFrame);

  XLA_FFI_Handler_Bundle handlers =
      mlir::shuttle::gpuExecutableBundleFfiHandlerBundle();
  SyntheticAllocator allocator(0x90000000);
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  stream_executor::GpuComputeCapability capability{
      stream_executor::CudaComputeCapability{9, 0}};
  auto trace = std::make_shared<ExecutionTrace>();
  EXPECT_CALL(stream, parent()).WillRepeatedly(testing::Return(&executor));
  EXPECT_CALL(executor, LoadKernel(testing::_))
      .Times(decoded->launches.size())
      .WillRepeatedly([trace](const stream_executor::KernelLoaderSpec &spec) {
        return recordLoadedKernel(spec, trace);
      });

  xla::ffi::ExecutionState instantiateState;
  xla::ffi::ExecutionState prepareState;
  xla::ffi::ExecutionState initializeState;
  xla::ffi::InvokeContext context;
  context.device_ordinal = 0;
  context.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &allocator, nullptr, nullptr,
      nullptr, nullptr,    nullptr, &capability};
  context.state_context = {&instantiateState, &prepareState, &initializeState};

  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_EQ(allocator.sizes, decoded->temporaryGroupBytes);
  EXPECT_EQ(llvm::accumulate(allocator.sizes, uint64_t{0}), 1932);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.initialize,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INITIALIZE)
                  .ok());
  expectLoadedKernels(*decoded, *trace, 0);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_EXECUTE)
                  .ok());
  EXPECT_EQ(trace->launchAttempts.load(), decoded->launches.size());
  auto slots = dynamicSlotAddresses(*decoded, allocator, 0x90000000,
                                    dynamicFrame->externalSlots);
  expectKernelLaunches(*decoded, *trace, 0, slots);
}

TEST(GpuTransportV2Test,
     Abi14ContractMapUsesUnchangedFfiSchemasAndTwoLaunchLifecycle) {
  auto built = buildFixture(contractMapModule(3, 5, 2, "ffi_anonymous"));
  ASSERT_TRUE(built);
  auto transport = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));
  auto decoded = decodeInlineSchema2(*transport);
  ASSERT_TRUE(decoded);
  ASSERT_EQ(decoded->launches.size(), 2);
  auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
  ASSERT_TRUE(executable.ok()) << executable.status();
  ASSERT_EQ((*executable)->externalBindings().size(), 3);
  auto dynamicFrame =
      dynamicCallFrame(**executable, *transport,
                       mlir::shuttle::gpuExecutableBundleDigest(*transport));
  ASSERT_TRUE(dynamicFrame);

  XLA_FFI_Handler_Bundle handlers =
      mlir::shuttle::gpuExecutableBundleFfiHandlerBundle();
  SyntheticAllocator allocator(0xa0000000);
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  stream_executor::GpuComputeCapability capability{
      stream_executor::CudaComputeCapability{9, 0}};
  auto trace = std::make_shared<ExecutionTrace>();
  EXPECT_CALL(stream, parent()).WillRepeatedly(testing::Return(&executor));
  EXPECT_CALL(executor, LoadKernel(testing::_))
      .Times(2)
      .WillRepeatedly([trace](const stream_executor::KernelLoaderSpec &spec) {
        return recordLoadedKernel(spec, trace);
      });

  xla::ffi::ExecutionState instantiateState;
  xla::ffi::ExecutionState prepareState;
  xla::ffi::ExecutionState initializeState;
  xla::ffi::InvokeContext context;
  context.device_ordinal = 0;
  context.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &allocator, nullptr, nullptr,
      nullptr, nullptr,    nullptr, &capability};
  context.state_context = {&instantiateState, &prepareState, &initializeState};

  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_EQ(allocator.sizes, (llvm::SmallVector<uint64_t>{24}));
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.initialize,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INITIALIZE)
                  .ok());
  expectLoadedKernels(*decoded, *trace, 0);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_EXECUTE)
                  .ok());
  EXPECT_EQ(trace->launchAttempts.load(), 2);
  auto slots = dynamicSlotAddresses(*decoded, allocator, 0xa0000000,
                                    dynamicFrame->externalSlots);
  expectKernelLaunches(*decoded, *trace, 0, slots);
}

TEST(GpuTransportV2Test,
     Abi16ProjectedBatchedContractUsesUnchangedFfiSchemasAndSixLaunches) {
  auto source =
      fixtureSource("jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir");
  ASSERT_TRUE(source);
  auto built = buildGpuContractFixture(
      *source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(built);
  auto transport = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));
  auto decoded = decodeInlineSchema2(*transport);
  ASSERT_TRUE(decoded);
  ASSERT_EQ(decoded->launches.size(), 6);
  auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
  ASSERT_TRUE(executable.ok()) << executable.status();
  ASSERT_EQ((*executable)->externalBindings().size(), 3);
  auto dynamicFrame =
      dynamicCallFrame(**executable, *transport,
                       mlir::shuttle::gpuExecutableBundleDigest(*transport));
  ASSERT_TRUE(dynamicFrame);

  XLA_FFI_Handler_Bundle handlers =
      mlir::shuttle::gpuExecutableBundleFfiHandlerBundle();
  SyntheticAllocator allocator(0xb0000000);
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  stream_executor::GpuComputeCapability capability{
      stream_executor::CudaComputeCapability{9, 0}};
  auto trace = std::make_shared<ExecutionTrace>();
  EXPECT_CALL(stream, parent()).WillRepeatedly(testing::Return(&executor));
  EXPECT_CALL(executor, LoadKernel(testing::_))
      .Times(6)
      .WillRepeatedly([trace](const stream_executor::KernelLoaderSpec &spec) {
        return recordLoadedKernel(spec, trace);
      });

  xla::ffi::ExecutionState instantiateState;
  xla::ffi::ExecutionState prepareState;
  xla::ffi::ExecutionState initializeState;
  xla::ffi::InvokeContext context;
  context.device_ordinal = 0;
  context.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &allocator, nullptr, nullptr,
      nullptr, nullptr,    nullptr, &capability};
  context.state_context = {&instantiateState, &prepareState, &initializeState};

  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_EQ(allocator.sizes, decoded->temporaryGroupBytes);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.initialize,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INITIALIZE)
                  .ok());
  expectLoadedKernels(*decoded, *trace, 0);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_EXECUTE)
                  .ok());
  EXPECT_EQ(trace->launchAttempts.load(), 6);
  auto slots = dynamicSlotAddresses(*decoded, allocator, 0xb0000000,
                                    dynamicFrame->externalSlots);
  expectKernelLaunches(*decoded, *trace, 0, slots);
}

TEST(GpuTransportV2Test,
     Abi17RankNFoldUsesUnchangedFfiSchemasAndRankFiveLaunches) {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-forward.mlir");
  ASSERT_TRUE(source);
  auto built = buildGpuContractFixture(
      *source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(built);
  auto transport = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));
  auto decoded = decodeInlineSchema2(*transport);
  ASSERT_TRUE(decoded);
  ASSERT_EQ(decoded->launches.size(), 6);
  auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
  ASSERT_TRUE(executable.ok()) << executable.status();
  ASSERT_EQ((*executable)->externalBindings().size(), 2);
  auto dynamicFrame =
      dynamicCallFrame(**executable, *transport,
                       mlir::shuttle::gpuExecutableBundleDigest(*transport));
  ASSERT_TRUE(dynamicFrame);

  XLA_FFI_Handler_Bundle handlers =
      mlir::shuttle::gpuExecutableBundleFfiHandlerBundle();
  SyntheticAllocator allocator(0xc0000000);
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  stream_executor::GpuComputeCapability capability{
      stream_executor::CudaComputeCapability{9, 0}};
  auto trace = std::make_shared<ExecutionTrace>();
  EXPECT_CALL(stream, parent()).WillRepeatedly(testing::Return(&executor));
  EXPECT_CALL(executor, LoadKernel(testing::_))
      .Times(6)
      .WillRepeatedly([trace](const stream_executor::KernelLoaderSpec &spec) {
        return recordLoadedKernel(spec, trace);
      });

  xla::ffi::ExecutionState instantiateState;
  xla::ffi::ExecutionState prepareState;
  xla::ffi::ExecutionState initializeState;
  xla::ffi::InvokeContext context;
  context.device_ordinal = 0;
  context.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &allocator, nullptr, nullptr,
      nullptr, nullptr,    nullptr, &capability};
  context.state_context = {&instantiateState, &prepareState, &initializeState};

  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_EQ(allocator.sizes, decoded->temporaryGroupBytes);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.initialize,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INITIALIZE)
                  .ok());
  expectLoadedKernels(*decoded, *trace, 0);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_EXECUTE)
                  .ok());
  EXPECT_EQ(trace->launchAttempts.load(), 6);
  auto slots = dynamicSlotAddresses(*decoded, allocator, 0xc0000000,
                                    dynamicFrame->externalSlots);
  expectKernelLaunches(*decoded, *trace, 0, slots);
}

TEST(GpuTransportV2Test,
     Abi18SingletonBroadcastUsesUnchangedFfiSchemasAndRankFiveLaunches) {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-forward.mlir");
  ASSERT_TRUE(source);
  auto built = buildGpuContractFixture(
      *source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(built);
  auto transport = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));
  auto decoded = decodeInlineSchema2(*transport);
  ASSERT_TRUE(decoded);
  ASSERT_EQ(decoded->launches.size(), 7);
  auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
  ASSERT_TRUE(executable.ok()) << executable.status();
  ASSERT_EQ((*executable)->externalBindings().size(), 2);
  auto dynamicFrame =
      dynamicCallFrame(**executable, *transport,
                       mlir::shuttle::gpuExecutableBundleDigest(*transport));
  ASSERT_TRUE(dynamicFrame);

  XLA_FFI_Handler_Bundle handlers =
      mlir::shuttle::gpuExecutableBundleFfiHandlerBundle();
  SyntheticAllocator allocator(0xd0000000);
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  stream_executor::GpuComputeCapability capability{
      stream_executor::CudaComputeCapability{9, 0}};
  auto trace = std::make_shared<ExecutionTrace>();
  EXPECT_CALL(stream, parent()).WillRepeatedly(testing::Return(&executor));
  EXPECT_CALL(executor, LoadKernel(testing::_))
      .Times(7)
      .WillRepeatedly([trace](const stream_executor::KernelLoaderSpec &spec) {
        return recordLoadedKernel(spec, trace);
      });

  xla::ffi::ExecutionState instantiateState;
  xla::ffi::ExecutionState prepareState;
  xla::ffi::ExecutionState initializeState;
  xla::ffi::InvokeContext context;
  context.device_ordinal = 0;
  context.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &allocator, nullptr, nullptr,
      nullptr, nullptr,    nullptr, &capability};
  context.state_context = {&instantiateState, &prepareState, &initializeState};

  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_EQ(allocator.sizes, decoded->temporaryGroupBytes);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.initialize,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INITIALIZE)
                  .ok());
  expectLoadedKernels(*decoded, *trace, 0);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_EXECUTE)
                  .ok());
  EXPECT_EQ(trace->launchAttempts.load(), 7);
  auto slots = dynamicSlotAddresses(*decoded, allocator, 0xd0000000,
                                    dynamicFrame->externalSlots);
  expectKernelLaunches(*decoded, *trace, 0, slots);
}

TEST(GpuTransportV2Test,
     Abi20ProjectedParameterUsesUnchangedFfiSchemasAndThreeBindings) {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-projected_parameter_broadcast-"
      "3x2x5x11x263-forward.mlir");
  ASSERT_TRUE(source);
  auto built = buildGpuContractFixture(
      *source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(built);
  auto transport = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));
  auto decoded = decodeInlineSchema2(*transport);
  ASSERT_TRUE(decoded);
  ASSERT_EQ(decoded->launches.size(), 19);
  auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
  ASSERT_TRUE(executable.ok()) << executable.status();
  ASSERT_EQ((*executable)->externalBindings().size(), 3);
  auto dynamicFrame =
      dynamicCallFrame(**executable, *transport,
                       mlir::shuttle::gpuExecutableBundleDigest(*transport));
  ASSERT_TRUE(dynamicFrame);

  XLA_FFI_Handler_Bundle handlers =
      mlir::shuttle::gpuExecutableBundleFfiHandlerBundle();
  SyntheticAllocator allocator(0xe0000000);
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  stream_executor::GpuComputeCapability capability{
      stream_executor::CudaComputeCapability{9, 0}};
  auto trace = std::make_shared<ExecutionTrace>();
  EXPECT_CALL(stream, parent()).WillRepeatedly(testing::Return(&executor));
  EXPECT_CALL(executor, LoadKernel(testing::_))
      .Times(19)
      .WillRepeatedly([trace](const stream_executor::KernelLoaderSpec &spec) {
        return recordLoadedKernel(spec, trace);
      });

  xla::ffi::ExecutionState instantiateState;
  xla::ffi::ExecutionState prepareState;
  xla::ffi::ExecutionState initializeState;
  xla::ffi::InvokeContext context;
  context.device_ordinal = 0;
  context.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &allocator, nullptr, nullptr,
      nullptr, nullptr,    nullptr, &capability};
  context.state_context = {&instantiateState, &prepareState, &initializeState};

  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_EQ(allocator.sizes, decoded->temporaryGroupBytes);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.initialize,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INITIALIZE)
                  .ok());
  expectLoadedKernels(*decoded, *trace, 0);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_EXECUTE)
                  .ok());
  EXPECT_EQ(trace->launchAttempts.load(), 19);
  auto slots = dynamicSlotAddresses(*decoded, allocator, 0xe0000000,
                                    dynamicFrame->externalSlots);
  expectKernelLaunches(*decoded, *trace, 0, slots);
}

TEST(GpuTransportV2Test,
     Abi21FullPrefixFoldUsesUnchangedFfiSchemasAndFiveBindings) {
  auto source = fixtureSource(
      "jax-0.10.1-bf16-projected_parameter_broadcast-"
      "3x2x5x11x263-backward.mlir");
  ASSERT_TRUE(source);
  auto built = buildGpuContractFixture(
      *source, mlir::shuttle::NumericalPolicy::Fast);
  ASSERT_TRUE(built);
  auto transport = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));
  auto decoded = decodeInlineSchema2(*transport);
  ASSERT_TRUE(decoded);
  ASSERT_EQ(decoded->launches.size(), 48);
  auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
  ASSERT_TRUE(executable.ok()) << executable.status();
  ASSERT_EQ((*executable)->externalBindings().size(), 5);
  auto dynamicFrame =
      dynamicCallFrame(**executable, *transport,
                       mlir::shuttle::gpuExecutableBundleDigest(*transport));
  ASSERT_TRUE(dynamicFrame);

  XLA_FFI_Handler_Bundle handlers =
      mlir::shuttle::gpuExecutableBundleFfiHandlerBundle();
  SyntheticAllocator allocator(0xf0000000);
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  stream_executor::GpuComputeCapability capability{
      stream_executor::CudaComputeCapability{9, 0}};
  auto trace = std::make_shared<ExecutionTrace>();
  EXPECT_CALL(stream, parent()).WillRepeatedly(testing::Return(&executor));
  EXPECT_CALL(executor, LoadKernel(testing::_))
      .Times(48)
      .WillRepeatedly([trace](const stream_executor::KernelLoaderSpec &spec) {
        return recordLoadedKernel(spec, trace);
      });

  xla::ffi::ExecutionState instantiateState;
  xla::ffi::ExecutionState prepareState;
  xla::ffi::ExecutionState initializeState;
  xla::ffi::InvokeContext context;
  context.device_ordinal = 0;
  context.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &allocator, nullptr, nullptr,
      nullptr, nullptr,    nullptr, &capability};
  context.state_context = {&instantiateState, &prepareState, &initializeState};

  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_EQ(allocator.sizes, decoded->temporaryGroupBytes);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.initialize,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_INITIALIZE)
                  .ok());
  expectLoadedKernels(*decoded, *trace, 0);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                               dynamicFrame->frame, context,
                               XLA_FFI_ExecutionStage_EXECUTE)
                  .ok());
  EXPECT_EQ(trace->launchAttempts.load(), 48);
  auto slots = dynamicSlotAddresses(*decoded, allocator, 0xf0000000,
                                    dynamicFrame->externalSlots);
  expectKernelLaunches(*decoded, *trace, 0, slots);
}

} // namespace
