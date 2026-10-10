// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Runtime/GpuExecutable.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "shuttle/IR/ShuttleOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

namespace mlir::shuttle {
namespace {

constexpr std::array<uint8_t, 8> kMagic{'S', 'H', 'U', 'T', 'G', 'P', 'U', 0};

std::string digest(ArrayRef<uint8_t> bytes) {
  llvm::SHA256 sha;
  sha.update(bytes);
  return llvm::toHex(sha.final(), true);
}

std::string digest(ArrayRef<uint8_t> prefix, ArrayRef<uint8_t> suffix) {
  llvm::SHA256 sha;
  sha.update(prefix);
  sha.update(suffix);
  return llvm::toHex(sha.final(), true);
}

bool isDigest(StringRef value) {
  return value.size() == 64 && llvm::all_of(value, [](char character) {
           return llvm::isDigit(character) ||
                  (character >= 'a' && character <= 'f');
         });
}

bool isScheduleReductionOrder(uint8_t value) {
  switch (static_cast<ScheduleReductionOrder>(value)) {
  case ScheduleReductionOrder::TreeAssociationFreeLeafOrderFixed:
  case ScheduleReductionOrder::LeafOrderFree:
  case ScheduleReductionOrder::SerialAscendingFmaRnF32:
  case ScheduleReductionOrder::StableTotalOrderLeafOrderFixed:
    return true;
  }
  return false;
}

class Writer {
public:
  void u8(uint8_t value) { bytes.push_back(value); }
  void u16(uint16_t value) { integer(value); }
  void u32(uint32_t value) { integer(value); }
  void u64(uint64_t value) { integer(value); }
  void i64(int64_t value) { u64(static_cast<uint64_t>(value)); }
  void text(StringRef value) {
    u64(value.size());
    bytes.append(value.bytes_begin(), value.bytes_end());
  }
  void blob(ArrayRef<int8_t> value) {
    u64(value.size());
    for (int8_t byte : value)
      bytes.push_back(static_cast<uint8_t>(byte));
  }
  void blobBytes(ArrayRef<uint8_t> value) {
    u64(value.size());
    append(value);
  }
  void integers(ArrayRef<int64_t> values) {
    u64(values.size());
    for (int64_t value : values)
      i64(value);
  }
  void append(ArrayRef<uint8_t> values) {
    bytes.append(values.begin(), values.end());
  }
  ArrayRef<uint8_t> view() const { return bytes; }
  SmallVector<uint8_t> take() { return std::move(bytes); }

private:
  template <typename T> void integer(T value) {
    for (unsigned byte = 0; byte < sizeof(T); ++byte)
      bytes.push_back(static_cast<uint8_t>(value >> (byte * 8)));
  }
  SmallVector<uint8_t> bytes;
};

class Reader {
public:
  explicit Reader(ArrayRef<uint8_t> bytes) : bytes(bytes) {}
  std::optional<uint8_t> u8() { return integer<uint8_t>(); }
  std::optional<uint16_t> u16() { return integer<uint16_t>(); }
  std::optional<uint32_t> u32() { return integer<uint32_t>(); }
  std::optional<uint64_t> u64() { return integer<uint64_t>(); }
  std::optional<int64_t> i64() {
    auto value = u64();
    return value ? std::optional<int64_t>(static_cast<int64_t>(*value))
                 : std::nullopt;
  }
  std::optional<StringRef> text() {
    auto size = u64();
    if (!size || !bounded(*size))
      return std::nullopt;
    StringRef value(reinterpret_cast<const char *>(bytes.data() + offset),
                    *size);
    offset += *size;
    return value;
  }
  std::optional<ArrayRef<uint8_t>> blob() {
    auto size = u64();
    if (!size || !bounded(*size))
      return std::nullopt;
    ArrayRef<uint8_t> value = bytes.slice(offset, *size);
    offset += *size;
    return value;
  }
  std::optional<SmallVector<int64_t>> integers() {
    auto size = u64();
    if (!size || *size > kMaximumGpuExecutableRecords ||
        *size > (bytes.size() - offset) / sizeof(int64_t))
      return std::nullopt;
    SmallVector<int64_t> values;
    values.reserve(*size);
    for (uint64_t index = 0; index < *size; ++index) {
      auto value = i64();
      if (!value)
        return std::nullopt;
      values.push_back(*value);
    }
    return values;
  }
  size_t position() const { return offset; }
  bool done() const { return offset == bytes.size(); }

private:
  template <typename T> std::optional<T> integer() {
    if (!bounded(sizeof(T)))
      return std::nullopt;
    T value = 0;
    for (unsigned byte = 0; byte < sizeof(T); ++byte)
      value |= static_cast<T>(bytes[offset++]) << (byte * 8);
    return value;
  }
  bool bounded(uint64_t size) const { return size <= bytes.size() - offset; }
  ArrayRef<uint8_t> bytes;
  size_t offset = 0;
};

void writeAccesses(Writer &writer, ArrayAttr accesses) {
  writer.u64(accesses.size());
  for (Attribute attribute : accesses)
    writer.u8(
        static_cast<uint8_t>(cast<ExecutableAccessAttr>(attribute).getValue()));
}

void writeTensor(Writer &writer, Type type) {
  auto tensor = cast<RankedTensorType>(type);
  Type elementType = tensor.getElementType();
  assert((elementType.isBF16() || elementType.isF32() ||
          elementType.isInteger(1) || elementType.isInteger(32)) &&
         "unsupported GPU executable tensor element type");
  writer.u8(elementType.isBF16()       ? 0
            : elementType.isF32()      ? 1
            : elementType.isInteger(1) ? 2
                                       : 3);
  writer.u64(tensor.getRank());
  for (int64_t extent : tensor.getShape())
    writer.i64(extent);
}

void writeEntry(Writer &writer, DeviceEntryOp entry) {
  writer.i64(entry.getOrdinal());
  writer.i64(entry.getSourceTask());
  writer.i64(entry.getCodeOffset());
  writer.i64(entry.getCodeLength());
  writer.integers(entry.getInputBuffers());
  writer.integers(entry.getOutputBuffers());
  writeAccesses(writer, entry.getInputAccesses());
  writeAccesses(writer, entry.getOutputAccesses());
  writer.integers(entry.getDependencies());
  writer.u8(static_cast<uint8_t>(entry.getPredication()));
  writer.u8(entry.getReductionOrder().has_value());
  if (entry.getReductionOrder())
    writer.u8(static_cast<uint8_t>(*entry.getReductionOrder()));
  writer.text(entry.getCodeDigest());
  for (int64_t extent : *entry.getGrid())
    writer.u32(extent);
  for (int64_t extent : *entry.getBlock())
    writer.u16(extent);
  writer.u32(*entry.getDynamicSharedBytes());
  writer.u16(*entry.getKernelArity());
}

void writeSlot(Writer &writer, InvocationSlotOp slot) {
  writer.i64(slot.getOrdinal());
  writer.i64(slot.getSourceBuffer());
  writeTensor(writer, slot.getTensorType());
  writer.i64(slot.getRequiredBytes());
  writer.integers(slot.getStrides());
  writer.i64(slot.getOffset());
  writer.i64(slot.getAlignment());
  writer.u8(static_cast<uint8_t>(slot.getAddressSpace()));
  writer.u8(static_cast<uint8_t>(slot.getAccess()));
  writer.u8(static_cast<uint8_t>(slot.getStorage()));
  writer.i64(slot.getAliasGroup());
  writer.i64(slot.getReuseGroup());
  writer.u8(static_cast<uint8_t>(slot.getBinding()));
  writer.u8(slot.getBindingIndex().has_value());
  if (slot.getBindingIndex())
    writer.i64(*slot.getBindingIndex());
}

absl::Status invalidTransport() {
  return absl::InvalidArgumentError("invalid Shuttle GPU executable transport");
}

absl::Status invalidTransport(StringRef detail) {
  return absl::InvalidArgumentError(
      (Twine("invalid Shuttle GPU executable transport: ") + detail).str());
}

struct Parsed {
  std::unique_ptr<MLIRContext> context;
  int64_t deviceSchemaVersion;
  ExecutableCodeFormat codeFormat;
  std::string scheduleFingerprint;
  std::string planFingerprint;
  SmallVector<uint8_t> code;
  std::vector<std::vector<uint8_t>> images;
  std::vector<GpuLaunch> launches;
  std::vector<GpuEventRealization> eventRealizations;
  std::string physicalEventProvenance;
  std::string physicalEventProvenanceDigest;
  std::optional<GpuInternalEventSchedule> internalEventSchedule;
  std::vector<GpuDeviceIrSnapshot> deviceIrSnapshots;
  std::vector<GpuSlot> slots;
  std::vector<GpuExternalBinding> bindings;
};

std::optional<uint64_t> checkedProduct(ArrayRef<uint64_t> values,
                                       uint64_t maximum) {
  uint64_t product = 1;
  for (uint64_t value : values) {
    if (!value || value > maximum / product)
      return std::nullopt;
    product *= value;
  }
  return product;
}

absl::StatusOr<Parsed> parseV2(ArrayRef<uint8_t> bytes) {
  if (bytes.size() > kMaximumGpuTransportBytes)
    return invalidTransport("transport size");
  Reader reader(bytes);
  for (uint8_t expected : kMagic)
    if (reader.u8() != expected)
      return invalidTransport("magic");
  if (reader.u32() != 2)
    return invalidTransport("schema");
  constexpr size_t deviceStart = 12;
  auto deviceSchema = reader.i64();
  auto format = reader.u8();
  auto policy = reader.u8();
  auto schedule = reader.text();
  auto code = reader.blob();
  auto codeDigest = reader.text();
  size_t deviceRootOffset = reader.position();
  auto deviceRoot = reader.text();
  size_t deviceSuffixStart = reader.position();
  auto entryCount = reader.u64();
  const bool supportedTarget =
      (deviceSchema == 3 && format == 2) || (deviceSchema == 4 && format == 3);
  if (!supportedTarget || policy > 1 || !schedule || !isDigest(*schedule) ||
      !code || code->empty() || code->size() > kMaximumGpuCodeBytes ||
      !codeDigest || *codeDigest != digest(*code) || !deviceRoot ||
      !isDigest(*deviceRoot) || !entryCount || *entryCount == 0 ||
      *entryCount > kMaximumGpuExecutableRecords)
    return invalidTransport("device header");

  Parsed parsed;
  parsed.context = std::make_unique<MLIRContext>();
  parsed.deviceSchemaVersion = *deviceSchema;
  parsed.codeFormat = static_cast<ExecutableCodeFormat>(*format);
  parsed.scheduleFingerprint = *schedule;
  parsed.code.assign(code->begin(), code->end());
  int64_t expectedOffset = 0;
  uint64_t aggregatePositions = 0;
  DenseMap<int64_t, int64_t> producers;
  for (int64_t ordinal = 0; ordinal < static_cast<int64_t>(*entryCount);
       ++ordinal) {
    auto entryOrdinal = reader.i64();
    auto sourceTask = reader.i64();
    auto codeOffset = reader.i64();
    auto codeLength = reader.i64();
    auto inputs = reader.integers();
    auto outputs = reader.integers();
    auto inputCount = reader.u64();
    if (entryOrdinal != ordinal || sourceTask != ordinal ||
        codeOffset != expectedOffset || !codeLength || *codeLength <= 0 ||
        *codeLength > static_cast<int64_t>(kMaximumGpuEntryCodeBytes) ||
        !inputs || !outputs || !inputCount || *inputCount != inputs->size())
      return invalidTransport("entry header");
    for (uint64_t index = 0; index < *inputCount; ++index)
      if (reader.u8() != 0)
        return invalidTransport("entry input access");
    auto outputCount = reader.u64();
    if (!outputCount || *outputCount != outputs->size() || outputs->empty())
      return invalidTransport("entry output count");
    for (uint64_t index = 0; index < *outputCount; ++index)
      if (reader.u8() != 1)
        return invalidTransport("entry output access");
    auto dependencies = reader.integers();
    auto predication = reader.u8();
    auto hasReduction = reader.u8();
    std::optional<uint8_t> reduction;
    if (hasReduction == 1)
      reduction = reader.u8();
    else if (hasReduction != 0)
      return invalidTransport("entry reduction marker");
    auto entryDigest = reader.text();
    if (!dependencies || !predication || *predication > 1 ||
        (reduction && !isScheduleReductionOrder(*reduction)) || !entryDigest ||
        expectedOffset > static_cast<int64_t>(code->size()) - *codeLength)
      return invalidTransport("entry closure");
    ArrayRef<uint8_t> slice = code->slice(expectedOffset, *codeLength);
    StringRef ptx(reinterpret_cast<const char *>(slice.data()), slice.size());
    const StringRef expectedHeader =
        *format == 2 ? ".version 8.0\n.target sm_90\n.address_size 64\n"
                     : ".version 8.7\n.target sm_100\n.address_size 64\n";
    if (*entryDigest != digest(slice) || !ptx.starts_with(expectedHeader) ||
        !ptx.ends_with("\n") ||
        !ptx.contains(".visible .entry shuttle_entry(") ||
        !ptx.contains("ret;"))
      return invalidTransport("entry code");
    std::array<uint64_t, 3> grid;
    std::array<uint64_t, 3> block;
    for (uint64_t &extent : grid) {
      auto value = reader.u32();
      if (!value)
        return invalidTransport("grid encoding");
      extent = *value;
    }
    for (uint64_t &extent : block) {
      auto value = reader.u16();
      if (!value)
        return invalidTransport("block encoding");
      extent = *value;
    }
    auto shared = reader.u32();
    auto arity = reader.u16();
    SmallVector<uint64_t> geometry(grid.begin(), grid.end());
    geometry.append(block.begin(), block.end());
    auto blockThreads = checkedProduct(block, 1024);
    auto positions =
        checkedProduct(geometry, kMaximumGpuAggregateTaskPositions);
    if (!blockThreads || !positions ||
        *positions > kMaximumGpuAggregateTaskPositions - aggregatePositions ||
        !shared || *shared > kMaximumGpuDynamicSharedMemoryBytes || !arity ||
        *arity != inputs->size() + outputs->size())
      return invalidTransport("launch resource budget");
    aggregatePositions += *positions;
    if (llvm::any_of(*inputs, [](int64_t slot) { return slot < 0; }) ||
        llvm::any_of(*outputs, [](int64_t slot) { return slot < 0; }))
      return invalidTransport("negative slot ordinal");
    SmallVector<int64_t> expectedDependencies;
    for (int64_t input : *inputs) {
      auto producer = producers.find(input);
      if (producer != producers.end() &&
          !llvm::is_contained(expectedDependencies, producer->second))
        expectedDependencies.push_back(producer->second);
    }
    llvm::sort(expectedDependencies);
    llvm::SmallDenseSet<int64_t> uniqueDependencies;
    for (int64_t dependency : *dependencies)
      if (dependency < 0 || dependency >= ordinal ||
          !uniqueDependencies.insert(dependency).second)
        return invalidTransport("entry dependency range");
    if (*dependencies != expectedDependencies)
      return invalidTransport("entry dependencies");
    for (int64_t output : *outputs)
      if (!producers.try_emplace(output, ordinal).second)
        return invalidTransport("duplicate output producer");
    std::vector<GpuKernelArgument> arguments;
    arguments.reserve(inputs->size() + outputs->size());
    auto appendPointer = [&](int64_t slot, ExecutableAccess access) {
      GpuKernelArgument argument;
      argument.kind = GpuKernelArgumentKind::SlotPointer;
      argument.slotOrdinal = slot;
      argument.access = access;
      arguments.push_back(std::move(argument));
    };
    for (int64_t slot : *inputs)
      appendPointer(slot, ExecutableAccess::Read);
    for (int64_t slot : *outputs)
      appendPointer(slot, ExecutableAccess::Write);
    parsed.launches.push_back(
        {ordinal,
         "shuttle_entry",
         expectedOffset,
         *codeLength,
         entryDigest->str(),
         grid,
         block,
         *shared,
         std::move(arguments),
         {{GpuFunctionAttributeKind::MaximumDynamicSharedMemoryBytes, *shared}},
         std::vector<int64_t>(inputs->begin(), inputs->end()),
         std::vector<int64_t>(outputs->begin(), outputs->end()),
         std::vector<int64_t>(dependencies->begin(), dependencies->end()),
         {*sourceTask}});
    expectedOffset += *codeLength;
  }
  size_t invocationStart = reader.position();
  if (expectedOffset != code->size() ||
      *deviceRoot !=
          digest(bytes.slice(deviceStart, deviceRootOffset - deviceStart),
                 bytes.slice(deviceSuffixStart,
                             invocationStart - deviceSuffixStart)))
    return invalidTransport("device closure");

  auto abiSchema = reader.i64();
  auto plan = reader.text();
  auto abiSchedule = reader.text();
  size_t abiRootOffset = reader.position();
  auto abiRoot = reader.text();
  size_t abiSuffixStart = reader.position();
  auto slotCount = reader.u64();
  if (abiSchema != 3 || !plan || !isDigest(*plan) || !abiSchedule ||
      *abiSchedule != *schedule || !abiRoot || !isDigest(*abiRoot) ||
      !slotCount || *slotCount == 0 ||
      *slotCount > kMaximumGpuExecutableRecords)
    return invalidTransport("invocation header");
  parsed.planFingerprint = *plan;
  DenseMap<int64_t, uint64_t> temporaryGroupBytes;
  uint64_t temporaryBytes = 0;
  for (int64_t ordinal = 0; ordinal < static_cast<int64_t>(*slotCount);
       ++ordinal) {
    auto slotOrdinal = reader.i64();
    auto sourceBuffer = reader.i64();
    auto element = reader.u8();
    auto rank = reader.u64();
    if (slotOrdinal != ordinal || sourceBuffer != ordinal || !element ||
        *element > 3 || !rank || *rank > 8)
      return invalidTransport("slot header");
    SmallVector<int64_t> shape;
    const int64_t elementBytes = *element == 0 ? 2 : *element == 2 ? 1 : 4;
    int64_t required = elementBytes;
    for (uint64_t axis = 0; axis < *rank; ++axis) {
      auto extent = reader.i64();
      if (!extent || *extent <= 0 ||
          required > std::numeric_limits<int64_t>::max() / *extent)
        return invalidTransport("slot extent");
      shape.push_back(*extent);
      required *= *extent;
    }
    auto encodedRequired = reader.i64();
    auto strides = reader.integers();
    auto offset = reader.i64();
    auto alignment = reader.i64();
    auto addressSpace = reader.u8();
    auto access = reader.u8();
    auto storage = reader.u8();
    auto alias = reader.i64();
    auto reuse = reader.i64();
    auto binding = reader.u8();
    auto hasIndex = reader.u8();
    std::optional<int64_t> index;
    if (hasIndex == 1)
      index = reader.i64();
    else if (hasIndex != 0)
      return invalidTransport();
    if (encodedRequired != required || !strides || strides->size() != *rank ||
        offset != 0 || alignment != elementBytes || addressSpace != 1 ||
        !access || *access > 2 || !storage || *storage > 1 ||
        alias != ordinal || !reuse || *reuse < 0 || !binding || *binding > 2)
      return invalidTransport("slot descriptor");
    SmallVector<int64_t> expectedStrides(*rank);
    int64_t stride = elementBytes;
    for (int64_t axis = *rank; axis > 0; --axis) {
      expectedStrides[axis - 1] = stride;
      stride *= shape[axis - 1];
    }
    if (*strides != expectedStrides)
      return invalidTransport("slot strides");
    Type elementType;
    if (*element == 0)
      elementType = BFloat16Type::get(parsed.context.get());
    else if (*element == 1)
      elementType = Float32Type::get(parsed.context.get());
    else if (*element == 2)
      elementType = IntegerType::get(parsed.context.get(), 1);
    else
      elementType = IntegerType::get(parsed.context.get(), 32);
    auto tensor = RankedTensorType::get(shape, elementType);
    GpuSlot slot{ordinal,
                 tensor,
                 required,
                 std::move(*strides),
                 0,
                 *alignment,
                 ExecutableAddressSpace::Device,
                 static_cast<ExecutableAccess>(*access),
                 static_cast<MaterializationStorage>(*storage),
                 ordinal,
                 *reuse,
                 static_cast<ExecutableBindingKind>(*binding),
                 index.value_or(-1)};
    if (*storage == 1) {
      if (required > kMaximumGpuTemporarySlotBytes)
        return invalidTransport("temporary slot budget");
      if (*binding != 0 || index || *access != 2)
        return invalidTransport("temporary binding");
      uint64_t prior = temporaryGroupBytes.lookup(*reuse);
      if (static_cast<uint64_t>(required) > prior) {
        uint64_t increase = static_cast<uint64_t>(required) - prior;
        if (increase > kMaximumGpuTemporaryBytes - temporaryBytes)
          return invalidTransport("temporary allocation budget");
        temporaryBytes += increase;
        temporaryGroupBytes[*reuse] = required;
      }
    } else {
      if (*reuse != ordinal || !index || *binding == 0 ||
          (*binding == 1 && *access != 0) || (*binding == 2 && *access != 1))
        return invalidTransport("external binding");
      parsed.bindings.push_back({static_cast<ExecutableBindingKind>(*binding),
                                 *index, ordinal, tensor, required,
                                 *alignment});
    }
    parsed.slots.push_back(std::move(slot));
  }
  size_t bundleStart = reader.position();
  if (*abiRoot !=
      digest(bytes.slice(invocationStart, abiRootOffset - invocationStart),
             bytes.slice(abiSuffixStart, bundleStart - abiSuffixStart)))
    return invalidTransport("invocation closure");
  SmallVector<int64_t> producer(*slotCount, -1);
  SmallVector<int64_t> firstUse(*slotCount, -1);
  SmallVector<int64_t> finalUse(*slotCount, -1);
  for (const GpuLaunch &launch : parsed.launches) {
    for (int64_t slot : launch.inputSlots) {
      if (slot < 0 || slot >= static_cast<int64_t>(*slotCount))
        return invalidTransport("input slot range");
      if (firstUse[slot] == -1)
        firstUse[slot] = launch.taskOrdinal;
      finalUse[slot] = std::max(finalUse[slot], launch.taskOrdinal);
    }
    for (int64_t slot : launch.outputSlots) {
      if (slot < 0 || slot >= static_cast<int64_t>(*slotCount) ||
          producer[slot] != -1)
        return invalidTransport("output slot producer");
      producer[slot] = launch.taskOrdinal;
      finalUse[slot] = std::max(finalUse[slot], launch.taskOrdinal);
    }
  }
  llvm::SmallDenseSet<int64_t> operandIndices;
  llvm::SmallDenseSet<int64_t> resultIndices;
  llvm::SmallDenseSet<int64_t> externalReuseGroups;
  for (const GpuSlot &slot : parsed.slots) {
    if (slot.storage == MaterializationStorage::Temporary) {
      if (producer[slot.ordinal] < 0 ||
          firstUse[slot.ordinal] <= producer[slot.ordinal] ||
          finalUse[slot.ordinal] <= producer[slot.ordinal])
        return invalidTransport("temporary lifetime");
      continue;
    }
    externalReuseGroups.insert(slot.reuseGroup);
    auto tensor = dyn_cast<RankedTensorType>(slot.tensorType);
    if (!tensor || (!tensor.getElementType().isBF16() &&
                    !tensor.getElementType().isF32() &&
                    !tensor.getElementType().isInteger(1) &&
                    !tensor.getElementType().isInteger(32)))
      return invalidTransport("external type");
    if (slot.bindingKind == ExecutableBindingKind::Operand) {
      if (slot.bindingIndex < 0 ||
          !operandIndices.insert(slot.bindingIndex).second ||
          producer[slot.ordinal] != -1 || finalUse[slot.ordinal] < 0)
        return invalidTransport("operand projection");
    } else if (slot.bindingKind == ExecutableBindingKind::Result) {
      if (slot.bindingIndex < 0 ||
          !resultIndices.insert(slot.bindingIndex).second ||
          producer[slot.ordinal] < 0 ||
          (firstUse[slot.ordinal] != -1 &&
           firstUse[slot.ordinal] <= producer[slot.ordinal]))
        return invalidTransport("result projection");
    } else {
      return invalidTransport("external binding kind");
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
      !isContiguous(operandIndices) || !isContiguous(resultIndices) ||
      parsed.bindings.size() != operandIndices.size() + resultIndices.size())
    return invalidTransport("binding count");
  for (size_t left = 0; left < parsed.slots.size(); ++left) {
    const GpuSlot &first = parsed.slots[left];
    if (first.storage == MaterializationStorage::Temporary &&
        externalReuseGroups.contains(first.reuseGroup))
      return invalidTransport("external reuse collision");
    for (size_t right = left + 1; right < parsed.slots.size(); ++right) {
      const GpuSlot &second = parsed.slots[right];
      if (second.storage != MaterializationStorage::Temporary ||
          first.storage != MaterializationStorage::Temporary ||
          first.reuseGroup != second.reuseGroup)
        continue;
      if (producer[first.ordinal] <= finalUse[second.ordinal] &&
          producer[second.ordinal] <= finalUse[first.ordinal])
        return invalidTransport("overlapping temporary reuse");
    }
  }
  auto bundleSchema = reader.i64();
  auto bundleSchedule = reader.text();
  auto bundleDevice = reader.text();
  auto bundleAbi = reader.text();
  auto completion = reader.u8();
  size_t bundleRootOffset = reader.position();
  auto bundleRoot = reader.text();
  if (bundleSchema != 2 || !bundleSchedule || *bundleSchedule != *schedule ||
      !bundleDevice || *bundleDevice != *deviceRoot || !bundleAbi ||
      *bundleAbi != *abiRoot || completion != 1 || !bundleRoot ||
      *bundleRoot != digest(bytes.take_front(bundleRootOffset)) ||
      !reader.done())
    return invalidTransport("bundle closure");
  llvm::sort(parsed.bindings, [](const GpuExternalBinding &left,
                                 const GpuExternalBinding &right) {
    if (left.kind != right.kind)
      return left.kind == ExecutableBindingKind::Operand;
    return left.index < right.index;
  });
  return parsed;
}

uint64_t scalarImmediateWidth(GpuScalarImmediateType type) {
  switch (type) {
  case GpuScalarImmediateType::I32:
  case GpuScalarImmediateType::U32:
  case GpuScalarImmediateType::F32:
    return 4;
  case GpuScalarImmediateType::I64:
  case GpuScalarImmediateType::U64:
  case GpuScalarImmediateType::F64:
    return 8;
  }
  return 0;
}

bool isPowerOfTwo(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool isKernelName(StringRef name) {
  return !name.empty() && name.size() <= 1024 &&
         llvm::all_of(name, [](char character) {
           return llvm::isAlnum(character) || character == '_' ||
                  character == '$' || character == '.';
         });
}

bool isCudaGeometry(const std::array<uint64_t, 3> &grid,
                    const std::array<uint64_t, 3> &block) {
  return grid[0] > 0 && grid[0] <= INT32_MAX && grid[1] > 0 &&
         grid[1] <= UINT16_MAX && grid[2] > 0 && grid[2] <= UINT16_MAX &&
         block[0] > 0 && block[0] <= 1024 && block[1] > 0 && block[1] <= 1024 &&
         block[2] > 0 && block[2] <= 64;
}

uint8_t accessMask(ExecutableAccess access) {
  switch (access) {
  case ExecutableAccess::Read:
    return 1;
  case ExecutableAccess::Write:
    return 2;
  case ExecutableAccess::ReadWrite:
    return 3;
  }
  return 0;
}

constexpr uint64_t kMaximumGpuAffineRank = 16;
constexpr uint64_t kMaximumGpuAffineResults = 32;
constexpr uint64_t kMaximumGpuAffineExpressionDepth = 32;
constexpr uint64_t kMaximumGpuAffineExpressionNodes = 4096;

bool validateAffineExpression(const GpuAffineExpression &expression,
                              int64_t dimensions, uint64_t depth,
                              uint64_t &nodes) {
  if (++nodes > kMaximumGpuAffineExpressionNodes ||
      depth > kMaximumGpuAffineExpressionDepth)
    return false;
  switch (expression.kind) {
  case GpuAffineExpressionKind::Dimension:
    return expression.operands.empty() && expression.payload >= 0 &&
           expression.payload < dimensions;
  case GpuAffineExpressionKind::Constant:
    return expression.operands.empty();
  case GpuAffineExpressionKind::Add:
    if (expression.payload != 0)
      return false;
    break;
  case GpuAffineExpressionKind::Multiply:
    if (expression.payload != 0 || expression.operands.size() != 2 ||
        (expression.operands[0].kind != GpuAffineExpressionKind::Constant &&
         expression.operands[1].kind != GpuAffineExpressionKind::Constant))
      return false;
    break;
  case GpuAffineExpressionKind::Modulo:
  case GpuAffineExpressionKind::FloorDivide:
  case GpuAffineExpressionKind::CeilDivide:
    if (expression.payload != 0 || expression.operands.size() != 2 ||
        expression.operands[1].kind != GpuAffineExpressionKind::Constant ||
        expression.operands[1].payload <= 0)
      return false;
    break;
  }
  if (expression.operands.size() != 2)
    return false;
  return validateAffineExpression(expression.operands[0], dimensions,
                                  depth + 1, nodes) &&
         validateAffineExpression(expression.operands[1], dimensions,
                                  depth + 1, nodes);
}

bool validateAffineMap(const GpuAffineMap &map, int64_t dimensions,
                       uint64_t &nodes) {
  if (dimensions < 0 ||
      dimensions > static_cast<int64_t>(kMaximumGpuAffineRank) ||
      map.dimensions != dimensions || map.results.empty() ||
      map.results.size() > kMaximumGpuAffineResults)
    return false;
  return llvm::all_of(map.results, [&](const GpuAffineExpression &result) {
    return validateAffineExpression(result, dimensions, 0, nodes);
  });
}

bool validateIntegerSet(const GpuIntegerSet &set, int64_t dimensions,
                        uint64_t &nodes) {
  if (set.dimensions != dimensions || set.constraints.empty() ||
      set.constraints.size() > kMaximumGpuAffineResults ||
      set.equalities.size() != set.constraints.size() ||
      llvm::any_of(set.equalities, [](uint8_t equality) {
        return equality > 1;
      }))
    return false;
  return llvm::all_of(
      set.constraints, [&](const GpuAffineExpression &constraint) {
        return validateAffineExpression(constraint, dimensions, 0, nodes);
      });
}

bool validateEventCoverPiece(const GpuEventCoverPiece &piece,
                             uint64_t launchCount) {
  const int64_t dimensions = piece.producer.owner.dimensions;
  uint64_t nodes = 0;
  if (piece.producer.launchOrdinal < 0 ||
      piece.producer.launchOrdinal >= static_cast<int64_t>(launchCount) ||
      piece.consumer.launchOrdinal < 0 ||
      piece.consumer.launchOrdinal >= static_cast<int64_t>(launchCount) ||
      !validateAffineMap(piece.producer.owner, dimensions, nodes) ||
      !validateAffineMap(piece.producer.localTime, dimensions, nodes) ||
      !validateAffineMap(piece.consumer.owner, dimensions, nodes) ||
      !validateAffineMap(piece.consumer.localTime, dimensions, nodes) ||
      (piece.relationSubset &&
       !validateIntegerSet(*piece.relationSubset, dimensions, nodes)))
    return false;
  return true;
}

bool isErasedEventRealization(GpuEventRealizationKind kind) {
  return kind == GpuEventRealizationKind::ProgramOrder ||
         kind == GpuEventRealizationKind::StreamOrder ||
         kind == GpuEventRealizationKind::AlgebraicRefinement;
}

bool validateEventMechanism(const GpuEventMechanismAllocation &mechanism,
                            int64_t dimensions) {
  uint64_t nodes = 0;
  const bool validTransaction =
      !mechanism.transactionBytes ||
      (*mechanism.transactionBytes > 0 &&
       *mechanism.transactionBytes <=
           static_cast<int64_t>(kMaximumGpuEventTransactionBytes));
  if (mechanism.allocationOrdinal < 0 ||
      static_cast<uint8_t>(mechanism.scope) >
          static_cast<uint8_t>(GpuEventPhysicalScope::System) ||
      mechanism.capacity <= 0 ||
      mechanism.capacity >
          static_cast<int64_t>(kMaximumGpuAggregateTaskPositions) ||
      mechanism.generationCount <= 0 ||
      mechanism.generationCount >
          static_cast<int64_t>(kMaximumGpuAggregateTaskPositions) ||
      mechanism.arrivalCount <= 0 || mechanism.arrivalCount > UINT32_MAX ||
      !validTransaction ||
      !validateAffineMap(mechanism.slot, dimensions, nodes) ||
      mechanism.slot.results.size() != 1 ||
      !validateAffineMap(mechanism.generationPhase, dimensions, nodes) ||
      mechanism.generationPhase.results.empty() ||
      mechanism.generationPhase.results.size() > 2)
    return false;
  return true;
}

bool validateEventOperation(const GpuEventOperation &operation,
                            int64_t dimensions, uint64_t launchCount,
                            uint64_t coverPieceCount) {
  uint64_t nodes = 0;
  return operation.ordinal >= 0 &&
         static_cast<uint8_t>(operation.kind) <=
             static_cast<uint8_t>(GpuEventOperationKind::Wait) &&
         operation.coverPieceOrdinal >= 0 &&
         operation.coverPieceOrdinal < static_cast<int64_t>(coverPieceCount) &&
         validateEventCoverPiece(
             {std::nullopt, operation.executionPoint,
              operation.executionPoint},
             launchCount) &&
         validateAffineMap(operation.slot, dimensions, nodes) &&
         operation.slot.results.size() == 1 &&
         validateAffineMap(operation.generationPhase, dimensions, nodes) &&
         !operation.generationPhase.results.empty() &&
         operation.generationPhase.results.size() <= 2 &&
         static_cast<uint8_t>(operation.scope) <=
             static_cast<uint8_t>(GpuEventPhysicalScope::System) &&
         operation.count > 0 && operation.count <= UINT32_MAX;
}

void writeAffineExpression(Writer &writer,
                           const GpuAffineExpression &expression) {
  writer.u8(static_cast<uint8_t>(expression.kind));
  writer.i64(expression.payload);
  writer.u8(expression.operands.size());
  for (const GpuAffineExpression &operand : expression.operands)
    writeAffineExpression(writer, operand);
}

void writeAffineMap(Writer &writer, const GpuAffineMap &map) {
  writer.u8(map.dimensions);
  writer.u8(map.results.size());
  for (const GpuAffineExpression &result : map.results)
    writeAffineExpression(writer, result);
}

void writeIntegerSet(Writer &writer, const GpuIntegerSet &set) {
  writer.u8(set.dimensions);
  writer.u8(set.constraints.size());
  for (auto [constraint, equality] :
       llvm::zip_equal(set.constraints, set.equalities)) {
    writer.u8(equality);
    writeAffineExpression(writer, constraint);
  }
}

void writeEventCoverEndpoint(Writer &writer,
                             const GpuEventCoverEndpoint &endpoint) {
  writer.i64(endpoint.launchOrdinal);
  writeAffineMap(writer, endpoint.owner);
  writeAffineMap(writer, endpoint.localTime);
}

void writeEventMechanism(Writer &writer,
                         const GpuEventMechanismAllocation &mechanism) {
  writer.i64(mechanism.allocationOrdinal);
  writer.u8(static_cast<uint8_t>(mechanism.scope));
  writer.i64(mechanism.capacity);
  writeAffineMap(writer, mechanism.slot);
  writeAffineMap(writer, mechanism.generationPhase);
  writer.i64(mechanism.generationCount);
  writer.i64(mechanism.arrivalCount);
  writer.u8(mechanism.transactionBytes.has_value());
  if (mechanism.transactionBytes)
    writer.i64(*mechanism.transactionBytes);
}

void writeEventOperation(Writer &writer,
                         const GpuEventOperation &operation) {
  writer.i64(operation.ordinal);
  writer.u8(static_cast<uint8_t>(operation.kind));
  writer.i64(operation.coverPieceOrdinal);
  writeEventCoverEndpoint(writer, operation.executionPoint);
  writeAffineMap(writer, operation.slot);
  writeAffineMap(writer, operation.generationPhase);
  writer.u8(static_cast<uint8_t>(operation.scope));
  writer.i64(operation.count);
  writer.u8(operation.releaseAcquire);
}

void writeEventOperationCertificateOperations(
    Writer &writer, const GpuEventOperationCertificate &certificate) {
  writer.u64(certificate.operations.size());
  for (const GpuEventOperation &operation : certificate.operations)
    writeEventOperation(writer, operation);
}

void writeEventOperationCertificate(
    Writer &writer, const GpuEventOperationCertificate &certificate) {
  writer.text(certificate.compilerIrDigest);
  writer.text(certificate.deviceImageDigest);
  writeEventOperationCertificateOperations(writer, certificate);
}

std::optional<GpuAffineExpression>
readAffineExpression(Reader &reader, int64_t dimensions, uint64_t depth,
                     uint64_t &nodes) {
  auto encodedKind = reader.u8();
  auto payload = reader.i64();
  auto operandCount = reader.u8();
  if (!encodedKind || *encodedKind >
                          static_cast<uint8_t>(
                              GpuAffineExpressionKind::CeilDivide) ||
      !payload || !operandCount || ++nodes > kMaximumGpuAffineExpressionNodes ||
      depth > kMaximumGpuAffineExpressionDepth || *operandCount > 2)
    return std::nullopt;
  GpuAffineExpression expression{
      static_cast<GpuAffineExpressionKind>(*encodedKind), *payload, {}};
  expression.operands.reserve(*operandCount);
  for (uint64_t index = 0; index < *operandCount; ++index) {
    std::optional<GpuAffineExpression> operand = readAffineExpression(
        reader, dimensions, depth + 1, nodes);
    if (!operand)
      return std::nullopt;
    expression.operands.push_back(std::move(*operand));
  }
  uint64_t validationNodes = 0;
  if (!validateAffineExpression(expression, dimensions, 0, validationNodes))
    return std::nullopt;
  return expression;
}

std::optional<GpuAffineMap> readAffineMap(Reader &reader, int64_t dimensions,
                                          uint64_t &nodes) {
  auto encodedDimensions = reader.u8();
  auto resultCount = reader.u8();
  if (!encodedDimensions || *encodedDimensions != dimensions ||
      *encodedDimensions > kMaximumGpuAffineRank || !resultCount ||
      *resultCount == 0 || *resultCount > kMaximumGpuAffineResults)
    return std::nullopt;
  GpuAffineMap map{*encodedDimensions, {}};
  map.results.reserve(*resultCount);
  for (uint64_t index = 0; index < *resultCount; ++index) {
    std::optional<GpuAffineExpression> result =
        readAffineExpression(reader, dimensions, 0, nodes);
    if (!result)
      return std::nullopt;
    map.results.push_back(std::move(*result));
  }
  return map;
}

std::optional<GpuIntegerSet> readIntegerSet(Reader &reader,
                                            int64_t dimensions,
                                            uint64_t &nodes) {
  auto encodedDimensions = reader.u8();
  auto constraintCount = reader.u8();
  if (!encodedDimensions || *encodedDimensions != dimensions ||
      *encodedDimensions > kMaximumGpuAffineRank || !constraintCount ||
      *constraintCount == 0 ||
      *constraintCount > kMaximumGpuAffineResults)
    return std::nullopt;
  GpuIntegerSet set{*encodedDimensions, {}, {}};
  set.constraints.reserve(*constraintCount);
  set.equalities.reserve(*constraintCount);
  for (uint64_t index = 0; index < *constraintCount; ++index) {
    auto equality = reader.u8();
    if (!equality || *equality > 1)
      return std::nullopt;
    std::optional<GpuAffineExpression> constraint =
        readAffineExpression(reader, dimensions, 0, nodes);
    if (!constraint)
      return std::nullopt;
    set.equalities.push_back(*equality);
    set.constraints.push_back(std::move(*constraint));
  }
  return set;
}

std::optional<GpuEventCoverEndpoint>
readEventCoverEndpoint(Reader &reader, int64_t dimensions, uint64_t launchCount,
                       uint64_t &nodes) {
  auto launch = reader.i64();
  if (!launch || *launch < 0 || *launch >= static_cast<int64_t>(launchCount))
    return std::nullopt;
  std::optional<GpuAffineMap> owner =
      readAffineMap(reader, dimensions, nodes);
  std::optional<GpuAffineMap> localTime =
      readAffineMap(reader, dimensions, nodes);
  if (!owner || !localTime)
    return std::nullopt;
  return GpuEventCoverEndpoint{*launch, std::move(*owner),
                               std::move(*localTime)};
}

std::optional<GpuEventMechanismAllocation>
readEventMechanism(Reader &reader, int64_t dimensions) {
  auto allocation = reader.i64();
  auto scope = reader.u8();
  auto capacity = reader.i64();
  uint64_t nodes = 0;
  std::optional<GpuAffineMap> slot =
      readAffineMap(reader, dimensions, nodes);
  std::optional<GpuAffineMap> generationPhase =
      readAffineMap(reader, dimensions, nodes);
  auto generationCount = reader.i64();
  auto arrivalCount = reader.i64();
  auto hasTransactionBytes = reader.u8();
  if (!allocation || !scope || !capacity || !slot || !generationPhase ||
      !generationCount || !arrivalCount || !hasTransactionBytes ||
      *hasTransactionBytes > 1)
    return std::nullopt;
  std::optional<int64_t> transactionBytes;
  if (*hasTransactionBytes) {
    transactionBytes = reader.i64();
    if (!transactionBytes)
      return std::nullopt;
  }
  GpuEventMechanismAllocation mechanism{
      *allocation,
      static_cast<GpuEventPhysicalScope>(*scope),
      *capacity,
      std::move(*slot),
      std::move(*generationPhase),
      *generationCount,
      *arrivalCount,
      transactionBytes,
  };
  if (!validateEventMechanism(mechanism, dimensions))
    return std::nullopt;
  return mechanism;
}

std::optional<GpuEventOperation>
readEventOperation(Reader &reader, int64_t dimensions, uint64_t launchCount,
                   uint64_t coverPieceCount) {
  auto ordinal = reader.i64();
  auto kind = reader.u8();
  auto coverPieceOrdinal = reader.i64();
  uint64_t nodes = 0;
  std::optional<GpuEventCoverEndpoint> executionPoint =
      readEventCoverEndpoint(reader, dimensions, launchCount, nodes);
  std::optional<GpuAffineMap> slot =
      readAffineMap(reader, dimensions, nodes);
  std::optional<GpuAffineMap> generationPhase =
      readAffineMap(reader, dimensions, nodes);
  auto scope = reader.u8();
  auto count = reader.i64();
  auto releaseAcquire = reader.u8();
  if (!ordinal || !kind || !coverPieceOrdinal || !executionPoint || !slot ||
      !generationPhase || !scope || !count || !releaseAcquire ||
      *releaseAcquire > 1)
    return std::nullopt;
  GpuEventOperation operation{
      *ordinal,
      static_cast<GpuEventOperationKind>(*kind),
      *coverPieceOrdinal,
      std::move(*executionPoint),
      std::move(*slot),
      std::move(*generationPhase),
      static_cast<GpuEventPhysicalScope>(*scope),
      *count,
      static_cast<bool>(*releaseAcquire),
  };
  if (!validateEventOperation(operation, dimensions, launchCount,
                              coverPieceCount))
    return std::nullopt;
  return operation;
}

std::optional<GpuEventOperationCertificate> readEventOperationCertificate(
    Reader &reader, int64_t dimensions, uint64_t launchCount,
    uint64_t coverPieceCount) {
  auto compilerIrDigest = reader.text();
  auto deviceImageDigest = reader.text();
  auto operationCount = reader.u64();
  if (!compilerIrDigest || !deviceImageDigest ||
      !isDigest(*compilerIrDigest) || !isDigest(*deviceImageDigest) ||
      !operationCount || *operationCount == 0 ||
      *operationCount > kMaximumGpuExecutableRecords)
    return std::nullopt;
  GpuEventOperationCertificate certificate{
      compilerIrDigest->str(), deviceImageDigest->str(), {}};
  certificate.operations.reserve(*operationCount);
  for (uint64_t ordinal = 0; ordinal < *operationCount; ++ordinal) {
    std::optional<GpuEventOperation> operation = readEventOperation(
        reader, dimensions, launchCount, coverPieceCount);
    if (!operation || operation->ordinal != static_cast<int64_t>(ordinal))
      return std::nullopt;
    certificate.operations.push_back(std::move(*operation));
  }
  if (gpuEventOperationCertificateDigest(certificate) !=
      certificate.compilerIrDigest)
    return std::nullopt;
  return certificate;
}

absl::StatusOr<Parsed> parseFatbin(ArrayRef<uint8_t> bytes,
                                  uint32_t transportVersion) {
  if (bytes.size() > kMaximumGpuTransportBytes)
    return invalidTransport();
  Reader reader(bytes);
  for (uint8_t expected : kMagic)
    if (reader.u8() != expected)
      return invalidTransport();
  if (transportVersion != 12 || reader.u32() != transportVersion)
    return invalidTransport();
  auto deviceSchema = reader.i64();
  auto format = reader.u8();
  auto schedule = reader.text();
  auto plan = reader.text();
  auto imageCount = reader.u64();
  const bool supportedTarget =
      (deviceSchema == 5 && format == 4) || (deviceSchema == 6 && format == 5);
  if (!supportedTarget || !schedule || !isDigest(*schedule) || !plan ||
      !isDigest(*plan) || !imageCount || *imageCount == 0 ||
      *imageCount > kMaximumGpuExecutableRecords)
    return invalidTransport("fatbin header");

  Parsed parsed;
  parsed.context = std::make_unique<MLIRContext>();
  parsed.deviceSchemaVersion = *deviceSchema;
  parsed.codeFormat = static_cast<ExecutableCodeFormat>(*format);
  parsed.scheduleFingerprint = *schedule;
  parsed.planFingerprint = *plan;
  uint64_t totalCodeBytes = 0;
  for (uint64_t index = 0; index < *imageCount; ++index) {
    auto code = reader.blob();
    auto codeDigest = reader.text();
    if (!code || code->empty() || !codeDigest ||
        code->size() > kMaximumGpuCodeBytes - totalCodeBytes ||
        *codeDigest != digest(*code))
      return invalidTransport("fatbin image");
    totalCodeBytes += code->size();
    parsed.images.emplace_back(code->begin(), code->end());
  }
  auto launchCount = reader.u64();
  if (!launchCount || *launchCount == 0 ||
      *launchCount > kMaximumGpuExecutableRecords)
    return invalidTransport("fatbin launch count");
  uint64_t aggregatePositions = 0;
  for (int64_t ordinal = 0; ordinal < static_cast<int64_t>(*launchCount);
       ++ordinal) {
    auto launchOrdinal = reader.i64();
    auto imageIndex = reader.u64();
    if (!imageIndex || *imageIndex >= parsed.images.size())
      return invalidTransport("fatbin launch image index");
    ArrayRef<uint8_t> code = parsed.images[*imageIndex];
    auto sourceTasks = reader.integers();
    auto kernelName = reader.text();
    std::array<uint64_t, 3> grid;
    std::array<uint64_t, 3> block;
    for (uint64_t &extent : grid) {
      auto value = reader.u32();
      if (!value)
        return invalidTransport("fatbin grid");
      extent = *value;
    }
    for (uint64_t &extent : block) {
      auto value = reader.u16();
      if (!value)
        return invalidTransport("fatbin block");
      extent = *value;
    }
    auto shared = reader.u32();
    auto argumentCount = reader.u16();
    SmallVector<uint64_t> geometry(grid.begin(), grid.end());
    geometry.append(block.begin(), block.end());
    auto blockThreads = checkedProduct(block, 1024);
    auto positions =
        checkedProduct(geometry, kMaximumGpuAggregateTaskPositions);
    llvm::SmallDenseSet<int64_t> uniqueSourceTasks;
    if (launchOrdinal != ordinal || !sourceTasks || sourceTasks->empty() ||
        sourceTasks->size() > kMaximumGpuExecutableRecords ||
        llvm::any_of(*sourceTasks,
                     [&](int64_t sourceTask) {
                       return sourceTask < 0 ||
                              !uniqueSourceTasks.insert(sourceTask).second;
                     }) ||
        !kernelName || !isKernelName(*kernelName) ||
        !isCudaGeometry(grid, block) || !shared ||
        *shared > kMaximumGpuDynamicSharedMemoryBytes || !argumentCount ||
        *argumentCount > kMaximumGpuExecutableRecords || !blockThreads ||
        !positions ||
        *positions > kMaximumGpuAggregateTaskPositions - aggregatePositions)
      return invalidTransport("fatbin launch resources");
    aggregatePositions += *positions;
    std::vector<GpuKernelArgument> arguments;
    arguments.reserve(*argumentCount);
    for (uint64_t index = 0; index < *argumentCount; ++index) {
      auto encodedKind = reader.u8();
      if (!encodedKind || *encodedKind > 4)
        return invalidTransport("fatbin argument kind");
      auto kind = static_cast<GpuKernelArgumentKind>(*encodedKind);
      if (kind == GpuKernelArgumentKind::SlotPointer) {
        auto slot = reader.i64();
        auto byteOffset = reader.u64();
        auto access = reader.u8();
        if (!slot || *slot < 0 || !byteOffset || !access || *access > 2)
          return invalidTransport("fatbin pointer argument");
        GpuKernelArgument argument;
        argument.kind = kind;
        argument.slotOrdinal = *slot;
        argument.slotByteOffset = *byteOffset;
        argument.access = static_cast<ExecutableAccess>(*access);
        arguments.push_back(std::move(argument));
        continue;
      }
      if (kind == GpuKernelArgumentKind::TensorMap) {
        auto slot = reader.i64();
        auto byteOffset = reader.u64();
        auto elementWidth = reader.u8();
        auto rank = reader.u8();
        if (!slot || *slot < 0 || !byteOffset || !elementWidth ||
            (*elementWidth != 1 && *elementWidth != 2 && *elementWidth != 4 &&
             *elementWidth != 8) ||
            !rank || *rank < 1 || *rank > 5)
          return invalidTransport("fatbin tensor-map header");
        GpuTensorMapDescriptor tensorMap;
        tensorMap.slotOrdinal = *slot;
        tensorMap.byteOffset = *byteOffset;
        tensorMap.elementByteWidth = *elementWidth;
        tensorMap.globalDimensions.reserve(*rank);
        tensorMap.globalByteStrides.reserve(*rank - 1);
        tensorMap.boxDimensions.reserve(*rank);
        tensorMap.elementStrides.reserve(*rank);
        for (uint64_t dimension = 0; dimension < *rank; ++dimension) {
          auto value = reader.u64();
          if (!value || *value == 0 || *value > UINT32_MAX)
            return invalidTransport("fatbin tensor-map dimensions");
          tensorMap.globalDimensions.push_back(*value);
        }
        for (uint64_t dimension = 1; dimension < *rank; ++dimension) {
          auto value = reader.u64();
          if (!value || *value < 16 || *value >= (UINT64_C(1) << 40) ||
              *value % 16 != 0)
            return invalidTransport("fatbin tensor-map global strides");
          tensorMap.globalByteStrides.push_back(*value);
        }
        for (uint64_t dimension = 0; dimension < *rank; ++dimension) {
          auto value = reader.u32();
          if (!value || *value == 0 || *value > 256)
            return invalidTransport("fatbin tensor-map box dimensions");
          tensorMap.boxDimensions.push_back(*value);
        }
        for (uint64_t dimension = 0; dimension < *rank; ++dimension) {
          auto value = reader.u32();
          if (!value || *value == 0 || *value > 8)
            return invalidTransport("fatbin tensor-map element strides");
          tensorMap.elementStrides.push_back(*value);
        }
        auto interleave = reader.u8();
        auto swizzle = reader.u8();
        auto l2Promotion = reader.u8();
        auto oobFill = reader.u8();
        auto alignment = reader.u16();
        auto access = reader.u8();
        if (!interleave || *interleave > 2 || !swizzle || *swizzle > 3 ||
            !l2Promotion || *l2Promotion > 3 || !oobFill || *oobFill > 1 ||
            !alignment || !isPowerOfTwo(*alignment) || *alignment < 64 ||
            *alignment > 256 || !access || *access > 2)
          return invalidTransport("fatbin tensor-map options");
        tensorMap.interleave = static_cast<GpuTensorMapInterleave>(*interleave);
        tensorMap.swizzle = static_cast<GpuTensorMapSwizzle>(*swizzle);
        tensorMap.l2Promotion =
            static_cast<GpuTensorMapL2Promotion>(*l2Promotion);
        tensorMap.oobFill = static_cast<GpuTensorMapOobFill>(*oobFill);
        tensorMap.alignment = *alignment;
        tensorMap.access = static_cast<ExecutableAccess>(*access);
        GpuKernelArgument argument;
        argument.kind = kind;
        argument.tensorMap = std::move(tensorMap);
        arguments.push_back(std::move(argument));
        continue;
      }
      if (kind == GpuKernelArgumentKind::StructuredBytes) {
        auto size = reader.u32();
        auto alignment = reader.u16();
        auto fieldCount = reader.u16();
        if (!size || *size == 0 || *size > kMaximumGpuKernelArgumentBytes ||
            !alignment || !isPowerOfTwo(*alignment) || *alignment > 256 ||
            !fieldCount || *fieldCount == 0 ||
            *fieldCount > kMaximumGpuExecutableRecords)
          return invalidTransport("fatbin structured argument header");
        GpuStructuredArgument structure;
        structure.size = *size;
        structure.alignment = *alignment;
        SmallVector<std::pair<uint64_t, uint64_t>> spans;
        for (uint64_t field = 0; field < *fieldCount; ++field) {
          auto fieldKind = reader.u8();
          auto offset = reader.u32();
          if (!fieldKind || *fieldKind > 1 || !offset)
            return invalidTransport("fatbin structured field header");
          GpuStructuredField value;
          value.kind = static_cast<GpuStructuredFieldKind>(*fieldKind);
          value.offset = *offset;
          uint64_t width = 0;
          if (value.kind == GpuStructuredFieldKind::SlotAddress) {
            auto fieldSlot = reader.i64();
            auto fieldByteOffset = reader.u64();
            auto access = reader.u8();
            if (!fieldSlot || *fieldSlot < 0 || !fieldByteOffset || !access ||
                *access > 2)
              return invalidTransport("fatbin structured pointer field");
            value.slotOrdinal = *fieldSlot;
            value.slotByteOffset = *fieldByteOffset;
            value.access = static_cast<ExecutableAccess>(*access);
            width = sizeof(uintptr_t);
          } else {
            auto fieldBytes = reader.blob();
            if (!fieldBytes || fieldBytes->empty() ||
                fieldBytes->size() > kMaximumGpuKernelArgumentBytes)
              return invalidTransport("fatbin structured immediate field");
            value.bytes.assign(fieldBytes->begin(), fieldBytes->end());
            width = fieldBytes->size();
          }
          if (value.offset > structure.size ||
              width > structure.size - value.offset)
            return invalidTransport("fatbin structured field range");
          for (auto [begin, end] : spans)
            if (value.offset < end && begin < value.offset + width)
              return invalidTransport("fatbin structured field overlap");
          spans.push_back({value.offset, value.offset + width});
          structure.fields.push_back(std::move(value));
        }
        GpuKernelArgument argument;
        argument.kind = kind;
        argument.structure = std::move(structure);
        arguments.push_back(std::move(argument));
        continue;
      }
      auto encodedScalar = reader.u8();
      auto alignment = reader.u16();
      auto value = reader.blob();
      if (!encodedScalar || *encodedScalar > 5 || !alignment ||
          !isPowerOfTwo(*alignment) || *alignment > 256 || !value ||
          value->empty() || value->size() > kMaximumGpuKernelArgumentBytes)
        return invalidTransport("fatbin value argument");
      auto scalar = static_cast<GpuScalarImmediateType>(*encodedScalar);
      if (kind == GpuKernelArgumentKind::ScalarImmediate &&
          (value->size() != scalarImmediateWidth(scalar) ||
           *alignment > value->size()))
        return invalidTransport("fatbin scalar argument");
      if (kind == GpuKernelArgumentKind::PackedBytes &&
          value->size() % *alignment != 0)
        return invalidTransport("fatbin packed argument");
      arguments.push_back({kind, -1, scalar,
                           std::vector<uint8_t>(value->begin(), value->end()),
                           *alignment});
    }
    auto attributeCount = reader.u16();
    if (!attributeCount || *attributeCount > kMaximumGpuExecutableRecords)
      return invalidTransport("fatbin function attributes");
    std::vector<GpuFunctionAttribute> attributes;
    llvm::SmallDenseSet<uint8_t> attributeKinds;
    for (uint64_t index = 0; index < *attributeCount; ++index) {
      auto kind = reader.u8();
      auto value = reader.u64();
      if (!kind || *kind > 2 || !value ||
          !attributeKinds.insert(*kind).second ||
          (*kind == 0 && (*value < *shared ||
                          *value > kMaximumGpuDynamicSharedMemoryBytes)) ||
          (*kind == 1 && *value > 100) || (*kind == 2 && *value > 1))
        return invalidTransport("fatbin function attribute");
      attributes.push_back(
          {static_cast<GpuFunctionAttributeKind>(*kind), *value});
    }
    auto inputs = reader.integers();
    auto outputs = reader.integers();
    auto dependencies = reader.integers();
    if (!inputs || !outputs || !dependencies)
      return invalidTransport("fatbin launch dataflow");
    llvm::SmallDenseSet<int64_t> uniqueDependencies;
    for (int64_t dependency : *dependencies)
      if (dependency < 0 || dependency >= ordinal ||
          !uniqueDependencies.insert(dependency).second)
        return invalidTransport("fatbin launch dependency");
    parsed.launches.push_back(
        {ordinal, kernelName->str(), 0, static_cast<int64_t>(code.size()),
         digest(code), grid, block, *shared, std::move(arguments),
         std::move(attributes),
         std::vector<int64_t>(inputs->begin(), inputs->end()),
         std::vector<int64_t>(outputs->begin(), outputs->end()),
         std::vector<int64_t>(dependencies->begin(), dependencies->end()),
         std::vector<int64_t>(sourceTasks->begin(), sourceTasks->end()),
         *imageIndex});
  }

  if (transportVersion >= 5) {
    auto realizationCount = reader.u64();
    if (!realizationCount ||
        *realizationCount > kMaximumGpuExecutableRecords)
      return invalidTransport("fatbin Event realization count");
    int64_t nextMechanismAllocation = 0;
    for (int64_t ordinal = 0;
         ordinal < static_cast<int64_t>(*realizationCount); ++ordinal) {
      auto eventOrdinal = reader.i64();
      auto kind = reader.u8();
      auto witnesses = reader.integers();
      const uint8_t maximumKind = transportVersion >= 8 ? 5
                                  : transportVersion >= 6 ? 2
                                                          : 1;
      if (eventOrdinal != ordinal || !kind || *kind > maximumKind ||
          !witnesses)
        return invalidTransport("fatbin Event realization descriptor");
      const auto realizationKind =
          static_cast<GpuEventRealizationKind>(*kind);
      const bool erased = isErasedEventRealization(realizationKind);
      if ((erased && (witnesses->empty() || witnesses->size() % 4 != 0)) ||
          (!erased && !witnesses->empty()))
        return invalidTransport("fatbin Event realization witnesses");
      const bool algebraicRefinement =
          *kind == static_cast<uint8_t>(
                       GpuEventRealizationKind::AlgebraicRefinement);
      bool crossesLaunch = false;
      for (size_t index = 0; index < witnesses->size(); index += 4) {
        const int64_t producerLaunch = (*witnesses)[index];
        const int64_t producerPhase = (*witnesses)[index + 1];
        const int64_t consumerLaunch = (*witnesses)[index + 2];
        const int64_t consumerPhase = (*witnesses)[index + 3];
        if (producerLaunch < 0 || producerLaunch >= *launchCount ||
            producerPhase < 0 || consumerLaunch < 0 ||
            consumerLaunch >= *launchCount || consumerPhase < 0)
          return invalidTransport("fatbin Event realization order");
        if (algebraicRefinement ? producerLaunch != consumerLaunch
                                : producerLaunch > consumerLaunch ||
                                      (producerLaunch == consumerLaunch &&
                                       producerPhase >= consumerPhase))
          return invalidTransport("fatbin Event realization order");
        crossesLaunch |= producerLaunch < consumerLaunch;
      }
      if ((*kind ==
               static_cast<uint8_t>(GpuEventRealizationKind::ProgramOrder) &&
           crossesLaunch) ||
          (*kind ==
               static_cast<uint8_t>(GpuEventRealizationKind::StreamOrder) &&
           !crossesLaunch))
        return invalidTransport("fatbin Event realization kind");
      parsed.eventRealizations.push_back(
          {*eventOrdinal, realizationKind,
           std::vector<int64_t>(witnesses->begin(), witnesses->end()), {},
           std::nullopt});
      if (transportVersion >= 7) {
        auto pieceCount = reader.u64();
        if (!pieceCount || *pieceCount == 0 ||
            *pieceCount > kMaximumGpuExecutableRecords)
          return invalidTransport("fatbin Event cover-piece count");
        GpuEventRealization &realization = parsed.eventRealizations.back();
        realization.coverPieces.reserve(*pieceCount);
        for (uint64_t pieceOrdinal = 0; pieceOrdinal < *pieceCount;
             ++pieceOrdinal) {
          auto dimensions = reader.u8();
          auto hasSubset = reader.u8();
          if (!dimensions || *dimensions > kMaximumGpuAffineRank || !hasSubset ||
              *hasSubset > 1)
            return invalidTransport("fatbin Event cover-piece header");
          uint64_t nodes = 0;
          std::optional<GpuIntegerSet> subset;
          if (*hasSubset) {
            subset = readIntegerSet(reader, *dimensions, nodes);
            if (!subset)
              return invalidTransport("fatbin Event cover-piece predicate");
          }
          std::optional<GpuEventCoverEndpoint> producer =
              readEventCoverEndpoint(reader, *dimensions, *launchCount, nodes);
          std::optional<GpuEventCoverEndpoint> consumer =
              readEventCoverEndpoint(reader, *dimensions, *launchCount, nodes);
          if (!producer || !consumer)
            return invalidTransport("fatbin Event cover-piece endpoint");
          GpuEventCoverPiece piece{std::move(subset), std::move(*producer),
                                   std::move(*consumer)};
          if (!validateEventCoverPiece(piece, *launchCount))
            return invalidTransport("fatbin Event cover-piece descriptor");
          realization.coverPieces.push_back(std::move(piece));
        }
        const int64_t relationDimensions =
            realization.coverPieces.front().producer.owner.dimensions;
        if (llvm::any_of(realization.coverPieces,
                         [&](const GpuEventCoverPiece &piece) {
                           return piece.producer.owner.dimensions !=
                                  relationDimensions;
                         }))
          return invalidTransport("fatbin Event cover-piece relation rank");
        if (transportVersion >= 8) {
          auto hasMechanism = reader.u8();
          if (!hasMechanism || *hasMechanism > 1 ||
              (erased && *hasMechanism) || (!erased && !*hasMechanism))
            return invalidTransport("fatbin Event realization mechanism");
          if (*hasMechanism) {
            std::optional<GpuEventMechanismAllocation> mechanism =
                readEventMechanism(reader, relationDimensions);
            if (!mechanism ||
                mechanism->allocationOrdinal != nextMechanismAllocation++)
              return invalidTransport(
                  "fatbin Event mechanism allocation ordinal");
            realization.mechanism = std::move(*mechanism);
          }
        }
        if (transportVersion >= 9) {
          auto hasCertificate = reader.u8();
          if (!hasCertificate || *hasCertificate > 1 ||
              (erased && *hasCertificate))
            return invalidTransport(
                "fatbin Event operation certificate presence");
          if (*hasCertificate) {
            std::optional<GpuEventOperationCertificate> certificate =
                readEventOperationCertificate(
                    reader, relationDimensions, *launchCount,
                    realization.coverPieces.size());
            if (!certificate)
              return invalidTransport(
                  "fatbin Event operation certificate descriptor");
            realization.operationCertificate = std::move(*certificate);
          }
        }
      }
    }
  }

  if (transportVersion >= 10) {
    auto hasPhysicalEventProvenance = reader.u8();
    if (!hasPhysicalEventProvenance || *hasPhysicalEventProvenance > 1)
      return invalidTransport("fatbin physical Event provenance presence");
    if (*hasPhysicalEventProvenance) {
      auto provenance = reader.text();
      auto provenanceDigest = reader.text();
      if (!provenance || provenance->empty() || !provenanceDigest ||
          !isDigest(*provenanceDigest) ||
          *provenanceDigest !=
              digest(ArrayRef<uint8_t>(
                  reinterpret_cast<const uint8_t *>(provenance->data()),
                  provenance->size())))
        return invalidTransport("fatbin physical Event provenance");
      parsed.physicalEventProvenance = provenance->str();
      parsed.physicalEventProvenanceDigest = provenanceDigest->str();
      absl::StatusOr<GpuInternalEventSchedule> internalSchedule =
          parseGpuInternalEventSchedule(*provenance);
      if (!internalSchedule.ok())
        return internalSchedule.status();
      parsed.internalEventSchedule = std::move(*internalSchedule);
    }
  }

  if (transportVersion >= 11) {
    auto snapshotCount = reader.u64();
    if (!snapshotCount || *snapshotCount > kMaximumGpuExecutableRecords)
      return invalidTransport("fatbin device IR snapshot count");
    parsed.deviceIrSnapshots.reserve(*snapshotCount);
    for (uint64_t snapshotOrdinal = 0; snapshotOrdinal < *snapshotCount;
         ++snapshotOrdinal) {
      auto bytecodeDigest = reader.text();
      auto operationCount = reader.u64();
      if (!bytecodeDigest || !isDigest(*bytecodeDigest) || !operationCount ||
          *operationCount == 0 ||
          *operationCount > kMaximumGpuExecutableRecords)
        return invalidTransport("fatbin device IR snapshot descriptor");
      GpuDeviceIrSnapshot snapshot{bytecodeDigest->str(), {}};
      snapshot.operationCounts.reserve(*operationCount);
      StringRef previousName;
      for (uint64_t operationOrdinal = 0; operationOrdinal < *operationCount;
           ++operationOrdinal) {
        auto operationName = reader.text();
        auto count = reader.u64();
        if (!operationName || operationName->empty() ||
            operationName->size() > 1024 || operationName->contains('\0') ||
            (!previousName.empty() && *operationName <= previousName) ||
            !count || *count == 0 ||
            *count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
          return invalidTransport("fatbin device IR operation inventory");
        snapshot.operationCounts.push_back(
            {operationName->str(), static_cast<int64_t>(*count)});
        previousName = *operationName;
      }
      parsed.deviceIrSnapshots.push_back(std::move(snapshot));
    }
  }

  auto slotCount = reader.u64();
  if (!slotCount || *slotCount == 0 ||
      *slotCount > kMaximumGpuExecutableRecords)
    return invalidTransport("fatbin slot count");
  DenseMap<int64_t, uint64_t> temporaryGroupBytes;
  uint64_t temporaryBytes = 0;
  for (int64_t ordinal = 0; ordinal < static_cast<int64_t>(*slotCount);
       ++ordinal) {
    auto slotOrdinal = reader.i64();
    auto element = reader.u8();
    auto rank = reader.u64();
    if (slotOrdinal != ordinal || !element || *element > 3 || !rank ||
        *rank > 8)
      return invalidTransport("fatbin slot header");
    SmallVector<int64_t> shape;
    const int64_t elementBytes = *element == 0 ? 2 : *element == 2 ? 1 : 4;
    int64_t required = elementBytes;
    for (uint64_t axis = 0; axis < *rank; ++axis) {
      auto extent = reader.i64();
      if (!extent || *extent <= 0 ||
          required > std::numeric_limits<int64_t>::max() / *extent)
        return invalidTransport("fatbin slot shape");
      shape.push_back(*extent);
      required *= *extent;
    }
    auto encodedRequired = reader.i64();
    auto strides = reader.integers();
    auto offset = reader.i64();
    auto alignment = reader.i64();
    auto addressSpace = reader.u8();
    auto access = reader.u8();
    auto storage = reader.u8();
    auto alias = reader.i64();
    auto reuse = reader.i64();
    auto binding = reader.u8();
    auto bindingIndex = reader.i64();
    if (encodedRequired != required || !strides || strides->size() != *rank ||
        !offset || *offset != 0 || !alignment || !isPowerOfTwo(*alignment) ||
        *alignment > 4096 || !addressSpace || *addressSpace != 1 || !access ||
        *access > 2 || !storage || *storage > 1 || alias != ordinal || !reuse ||
        *reuse < 0 || !binding || *binding > 2 || !bindingIndex)
      return invalidTransport("fatbin slot descriptor");
    SmallVector<int64_t> expectedStrides(*rank);
    int64_t stride = elementBytes;
    for (int64_t axis = *rank; axis > 0; --axis) {
      expectedStrides[axis - 1] = stride;
      stride *= shape[axis - 1];
    }
    if (*strides != expectedStrides)
      return invalidTransport("fatbin slot strides");
    Type elementType;
    if (*element == 0)
      elementType = BFloat16Type::get(parsed.context.get());
    else if (*element == 1)
      elementType = Float32Type::get(parsed.context.get());
    else if (*element == 2)
      elementType = IntegerType::get(parsed.context.get(), 1);
    else
      elementType = IntegerType::get(parsed.context.get(), 32);
    auto tensor = RankedTensorType::get(shape, elementType);
    GpuSlot slot{ordinal,
                 tensor,
                 required,
                 std::move(*strides),
                 0,
                 *alignment,
                 ExecutableAddressSpace::Device,
                 static_cast<ExecutableAccess>(*access),
                 static_cast<MaterializationStorage>(*storage),
                 ordinal,
                 *reuse,
                 static_cast<ExecutableBindingKind>(*binding),
                 *bindingIndex};
    if (*storage == 1) {
      if (required > kMaximumGpuTemporarySlotBytes)
        return invalidTransport("temporary slot budget");
      if (*binding != 0 || *bindingIndex != -1)
        return invalidTransport("fatbin temporary binding");
      uint64_t prior = temporaryGroupBytes.lookup(*reuse);
      if (static_cast<uint64_t>(required) > prior) {
        uint64_t increase = static_cast<uint64_t>(required) - prior;
        if (increase > kMaximumGpuTemporaryBytes - temporaryBytes)
          return invalidTransport("fatbin temporary budget");
        temporaryBytes += increase;
        temporaryGroupBytes[*reuse] = required;
      }
    } else {
      if (*reuse != ordinal || *binding == 0 || *bindingIndex < 0 ||
          (*binding == 1 && *access != 0) || (*binding == 2 && *access != 1))
        return invalidTransport("fatbin external binding");
      parsed.bindings.push_back({static_cast<ExecutableBindingKind>(*binding),
                                 *bindingIndex, ordinal, tensor, required,
                                 *alignment});
    }
    parsed.slots.push_back(std::move(slot));
  }

  SmallVector<int64_t> firstUse(*slotCount, -1);
  SmallVector<int64_t> finalUse(*slotCount, -1);
  SmallVector<uint8_t> aggregateAccess(*slotCount, 0);
  SmallVector<int64_t> lastWriter(*slotCount, -1);
  for (const GpuLaunch &launch : parsed.launches) {
    llvm::SmallDenseSet<int64_t> pointerSlots;
    llvm::DenseMap<int64_t, uint8_t> launchAccess;
    auto observePointer = [&](int64_t slot, uint64_t byteOffset, uint64_t width,
                              ExecutableAccess access) -> absl::Status {
      if (slot < 0 || slot >= static_cast<int64_t>(*slotCount) ||
          byteOffset >=
              static_cast<uint64_t>(parsed.slots[slot].requiredBytes) ||
          width > static_cast<uint64_t>(parsed.slots[slot].requiredBytes) -
                      byteOffset)
        return invalidTransport("fatbin argument slot range");
      pointerSlots.insert(slot);
      launchAccess[slot] |= accessMask(access);
      aggregateAccess[slot] |= accessMask(access);
      if (firstUse[slot] < 0)
        firstUse[slot] = launch.taskOrdinal;
      finalUse[slot] = launch.taskOrdinal;
      return absl::OkStatus();
    };
    for (const GpuKernelArgument &argument : launch.arguments) {
      if (argument.kind == GpuKernelArgumentKind::SlotPointer) {
        if (absl::Status status =
                observePointer(argument.slotOrdinal, argument.slotByteOffset, 1,
                               argument.access);
            !status.ok())
          return status;
        continue;
      }
      if (argument.kind == GpuKernelArgumentKind::TensorMap) {
        const GpuTensorMapDescriptor &tensorMap = argument.tensorMap;
        if (tensorMap.slotOrdinal < 0 ||
            tensorMap.slotOrdinal >= static_cast<int64_t>(*slotCount))
          return invalidTransport("fatbin tensor-map slot range");
        const GpuSlot &slot = parsed.slots[tensorMap.slotOrdinal];
        auto tensor = dyn_cast<RankedTensorType>(slot.tensorType);
        const uint64_t elementWidth =
            tensor && tensor.getElementType().isBF16() ? 2 : 4;
        if (!tensor || tensorMap.elementByteWidth != elementWidth ||
            slot.alignment < 16 || tensorMap.byteOffset % 16 != 0)
          return invalidTransport("fatbin tensor-map slot type");
        uint64_t span =
            tensorMap.elementByteWidth * tensorMap.globalDimensions.front();
        for (auto [extent, stride] : llvm::zip_equal(
                 ArrayRef<uint64_t>(tensorMap.globalDimensions).drop_front(),
                 tensorMap.globalByteStrides)) {
          if ((extent - 1) >
              (std::numeric_limits<uint64_t>::max() - span) / stride)
            return invalidTransport("fatbin tensor-map span");
          span += (extent - 1) * stride;
        }
        if (absl::Status status =
                observePointer(tensorMap.slotOrdinal, tensorMap.byteOffset,
                               span, tensorMap.access);
            !status.ok())
          return status;
        continue;
      }
      if (argument.kind == GpuKernelArgumentKind::StructuredBytes) {
        for (const GpuStructuredField &field : argument.structure.fields) {
          if (field.kind != GpuStructuredFieldKind::SlotAddress)
            continue;
          if (absl::Status status = observePointer(
                  field.slotOrdinal, field.slotByteOffset, 1, field.access);
              !status.ok())
            return status;
        }
      }
    }
    llvm::SmallDenseSet<int64_t> inputs;
    llvm::SmallDenseSet<int64_t> outputs;
    for (int64_t slot : launch.inputSlots)
      if (slot < 0 || slot >= static_cast<int64_t>(*slotCount) ||
          !inputs.insert(slot).second || !pointerSlots.contains(slot))
        return invalidTransport("fatbin input slot range");
    for (int64_t slot : launch.outputSlots)
      if (slot < 0 || slot >= static_cast<int64_t>(*slotCount) ||
          !outputs.insert(slot).second || !pointerSlots.contains(slot))
        return invalidTransport("fatbin output slot range");
    for (int64_t slot : pointerSlots) {
      uint8_t mask = launchAccess.lookup(slot);
      if (((mask & 1) != 0) != inputs.contains(slot) ||
          ((mask & 2) != 0) != outputs.contains(slot))
        return invalidTransport("fatbin slot access projection");
      if ((mask & 1) != 0 &&
          parsed.slots[slot].storage == MaterializationStorage::Temporary &&
          lastWriter[slot] < 0)
        return invalidTransport("fatbin scratch read before write");
      if ((mask & 1) != 0 && lastWriter[slot] >= 0 &&
          !llvm::is_contained(launch.dependencies, lastWriter[slot]))
        return invalidTransport("fatbin missing data dependency");
    }
    for (int64_t slot : outputs)
      lastWriter[slot] = launch.taskOrdinal;
  }
  llvm::SmallDenseSet<int64_t> operandIndices;
  llvm::SmallDenseSet<int64_t> resultIndices;
  for (const GpuSlot &slot : parsed.slots) {
    if (firstUse[slot.ordinal] < 0)
      return invalidTransport("fatbin unused slot");
    const uint8_t aggregate = aggregateAccess[slot.ordinal];
    const uint8_t abi = accessMask(slot.access);
    const bool resultReadAfterWrite =
        slot.storage == MaterializationStorage::External &&
        slot.bindingKind == ExecutableBindingKind::Result &&
        aggregate == (accessMask(ExecutableAccess::Read) |
                      accessMask(ExecutableAccess::Write));
    if (aggregate != abi && !resultReadAfterWrite)
      return invalidTransport("fatbin slot access closure");
    if (slot.storage == MaterializationStorage::Temporary)
      continue;
    llvm::SmallDenseSet<int64_t> *indices =
        slot.bindingKind == ExecutableBindingKind::Operand  ? &operandIndices
        : slot.bindingKind == ExecutableBindingKind::Result ? &resultIndices
                                                            : nullptr;
    if (!indices || !indices->insert(slot.bindingIndex).second)
      return invalidTransport("fatbin binding projection");
  }
  auto contiguous = [](const llvm::SmallDenseSet<int64_t> &indices) {
    for (int64_t index = 0; index < static_cast<int64_t>(indices.size());
         ++index)
      if (!indices.contains(index))
        return false;
    return true;
  };
  if (operandIndices.empty() || resultIndices.empty() ||
      !contiguous(operandIndices) || !contiguous(resultIndices))
    return invalidTransport("fatbin binding count");
  for (size_t left = 0; left < parsed.slots.size(); ++left) {
    const GpuSlot &first = parsed.slots[left];
    if (first.storage != MaterializationStorage::Temporary)
      continue;
    for (size_t right = left + 1; right < parsed.slots.size(); ++right) {
      const GpuSlot &second = parsed.slots[right];
      if (second.storage == MaterializationStorage::Temporary &&
          first.reuseGroup == second.reuseGroup &&
          firstUse[first.ordinal] <= finalUse[second.ordinal] &&
          firstUse[second.ordinal] <= finalUse[first.ordinal])
        return invalidTransport("fatbin overlapping scratch reuse");
    }
  }
  auto root = reader.text();
  if (!root ||
      *root != digest(bytes.take_front(reader.position() - sizeof(uint64_t) -
                                       root->size())) ||
      !reader.done())
    return invalidTransport("fatbin closure");
  llvm::sort(parsed.bindings, [](const GpuExternalBinding &left,
                                 const GpuExternalBinding &right) {
    if (left.kind != right.kind)
      return left.kind == ExecutableBindingKind::Operand;
    return left.index < right.index;
  });
  return parsed;
}

absl::StatusOr<Parsed> parse(ArrayRef<uint8_t> bytes) {
  if (bytes.size() < kMagic.size() + sizeof(uint32_t))
    return invalidTransport();
  Reader reader(bytes);
  for (uint8_t expected : kMagic)
    if (reader.u8() != expected)
      return invalidTransport();
  auto version = reader.u32();
  if (version == 2)
    return parseV2(bytes);
  if (version == 12)
    return parseFatbin(bytes, *version);
  return invalidTransport();
}

} // namespace

class GpuExecutable::Impl {
public:
  explicit Impl(Parsed parsed) : parsed(std::move(parsed)) {}
  Parsed parsed;
};

GpuExecutable::GpuExecutable(std::shared_ptr<const Impl> implementation)
    : implementation(std::move(implementation)) {}

absl::StatusOr<std::shared_ptr<const GpuExecutable>>
GpuExecutable::Load(ArrayRef<uint8_t> bytes) {
  try {
    absl::StatusOr<Parsed> parsed = parse(bytes);
    if (!parsed.ok())
      return parsed.status();
    auto implementation = std::make_shared<const Impl>(std::move(*parsed));
    return std::shared_ptr<const GpuExecutable>(
        new GpuExecutable(std::move(implementation)));
  } catch (const std::exception &) {
    return invalidTransport();
  } catch (...) {
    return invalidTransport();
  }
}

ArrayRef<uint8_t> GpuExecutable::codeBytes() const {
  const auto &parsed = implementation->parsed;
  if (parsed.images.size() == 1)
    return parsed.images.front();
  return parsed.code;
}
ArrayRef<std::vector<uint8_t>> GpuExecutable::codeImages() const {
  return implementation->parsed.images;
}
ArrayRef<uint8_t> GpuExecutable::codeImage(uint64_t index) const {
  return implementation->parsed.images.at(index);
}
uint64_t GpuExecutable::totalCodeBytes() const {
  uint64_t total = implementation->parsed.code.size();
  for (const auto &image : implementation->parsed.images)
    total += image.size();
  return total;
}
int64_t GpuExecutable::deviceSchemaVersion() const {
  return implementation->parsed.deviceSchemaVersion;
}
ExecutableCodeFormat GpuExecutable::codeFormat() const {
  return implementation->parsed.codeFormat;
}
llvm::StringRef GpuExecutable::scheduleFingerprint() const {
  return implementation->parsed.scheduleFingerprint;
}
llvm::StringRef GpuExecutable::planFingerprint() const {
  return implementation->parsed.planFingerprint;
}
ArrayRef<GpuLaunch> GpuExecutable::launches() const {
  return implementation->parsed.launches;
}
ArrayRef<GpuSlot> GpuExecutable::slots() const {
  return implementation->parsed.slots;
}
ArrayRef<GpuEventRealization> GpuExecutable::eventRealizations() const {
  return implementation->parsed.eventRealizations;
}
StringRef GpuExecutable::physicalEventProvenance() const {
  return implementation->parsed.physicalEventProvenance;
}
StringRef GpuExecutable::physicalEventProvenanceDigest() const {
  return implementation->parsed.physicalEventProvenanceDigest;
}
const std::optional<GpuInternalEventSchedule> &
GpuExecutable::internalEventSchedule() const {
  return implementation->parsed.internalEventSchedule;
}
ArrayRef<GpuDeviceIrSnapshot> GpuExecutable::deviceIrSnapshots() const {
  return implementation->parsed.deviceIrSnapshots;
}
ArrayRef<GpuExternalBinding> GpuExecutable::externalBindings() const {
  return implementation->parsed.bindings;
}

std::string gpuExecutableBundleDigest(ArrayRef<uint8_t> bytes) {
  return digest(bytes);
}

absl::StatusOr<SmallVector<uint8_t>>
gpuFatbinCodeIdentity(ArrayRef<std::vector<uint8_t>> images,
                      ArrayRef<GpuLaunch> launches) {
  if (images.empty() || images.size() > kMaximumGpuExecutableRecords ||
      launches.empty() || launches.size() > kMaximumGpuExecutableRecords)
    return invalidTransport("fatbin code identity count");
  uint64_t codeBytes = 0;
  for (const auto &image : images) {
    if (image.empty() || image.size() > kMaximumGpuCodeBytes - codeBytes)
      return invalidTransport("fatbin code identity image size");
    codeBytes += image.size();
  }
  for (const GpuLaunch &launch : launches)
    if (launch.imageIndex >= images.size())
      return invalidTransport("fatbin code identity launch image");
  if (images.size() == 1)
    return SmallVector<uint8_t>(images.front().begin(), images.front().end());
  Writer writer;
  writer.text("shuttle.fatbin-code-identity.v1");
  writer.u64(images.size());
  for (const auto &image : images)
    writer.blobBytes(image);
  writer.u64(launches.size());
  for (const GpuLaunch &launch : launches)
    writer.u64(launch.imageIndex);
  return writer.take();
}

std::string gpuEventOperationCertificateDigest(
    const GpuEventOperationCertificate &certificate) {
  Writer writer;
  writeEventOperationCertificateOperations(writer, certificate);
  return digest(writer.view());
}

FailureOr<SmallVector<uint8_t>> serializeGpuExecutableBundle(ModuleOp module) {
  SmallVector<DeviceModuleOp> devices(module.getOps<DeviceModuleOp>());
  SmallVector<InvocationAbiOp> abis(module.getOps<InvocationAbiOp>());
  SmallVector<ExecutableBundleOp> bundles(module.getOps<ExecutableBundleOp>());
  if (devices.size() != 1 || abis.size() != 1 || bundles.size() != 1 ||
      failed(devices.front().verifyRegions()) ||
      failed(abis.front().verifyRegions()) || failed(bundles.front().verify()))
    return failure();
  DeviceModuleOp device = devices.front();
  InvocationAbiOp abi = abis.front();
  ExecutableBundleOp bundle = bundles.front();
  const bool fatbin =
      (device.getSchemaVersion() == 5 &&
       device.getCodeFormat() == ExecutableCodeFormat::CudaFatbinSm90V1) ||
      (device.getSchemaVersion() == 6 &&
       device.getCodeFormat() == ExecutableCodeFormat::CudaFatbinSm100V1);
  if (fatbin) {
    DenseI8ArrayAttr transport = device.getNativeTransportAttr();
    if (!transport || !device.getPhysicalRequestFingerprintAttr())
      return failure();
    ArrayRef<int8_t> signedBytes = transport.asArrayRef();
    ArrayRef<uint8_t> bytes(
        reinterpret_cast<const uint8_t *>(signedBytes.data()),
        signedBytes.size());
    absl::StatusOr<std::shared_ptr<const GpuExecutable>> executable =
        GpuExecutable::Load(bytes);
    if (!executable.ok())
      return failure();
    auto codeIdentity = gpuFatbinCodeIdentity((*executable)->codeImages(),
                                              (*executable)->launches());
    if (!codeIdentity.ok() ||
        (*executable)->deviceSchemaVersion() != device.getSchemaVersion() ||
        (*executable)->codeFormat() != device.getCodeFormat() ||
        (*executable)->scheduleFingerprint() !=
            device.getSourceScheduleFingerprint() ||
        (*executable)->planFingerprint() != abi.getSourcePlanFingerprint() ||
        ArrayRef<uint8_t>(*codeIdentity) !=
            ArrayRef<uint8_t>(
                reinterpret_cast<const uint8_t *>(device.getCode().data()),
                device.getCode().size()))
      return failure();
    SmallVector<InvocationSlotOp> invocationSlots(
        abi.getBody().front().getOps<InvocationSlotOp>());
    if (invocationSlots.size() != (*executable)->slots().size())
      return failure();
    for (auto [actual, encoded] :
         llvm::zip_equal(invocationSlots, (*executable)->slots())) {
      auto actualType = dyn_cast<RankedTensorType>(actual.getTensorType());
      auto encodedType = dyn_cast<RankedTensorType>(encoded.tensorType);
      if (!actualType || !encodedType ||
          actualType.getShape() != encodedType.getShape() ||
          actualType.getElementType().isBF16() !=
              encodedType.getElementType().isBF16() ||
          actualType.getElementType().isF32() !=
              encodedType.getElementType().isF32() ||
          actualType.getElementType().isInteger(1) !=
              encodedType.getElementType().isInteger(1) ||
          actualType.getElementType().isInteger(32) !=
              encodedType.getElementType().isInteger(32) ||
          actual.getOrdinal() != encoded.ordinal ||
          actual.getRequiredBytes() != encoded.requiredBytes ||
          actual.getStrides() != ArrayRef<int64_t>(encoded.byteStrides) ||
          actual.getOffset() != encoded.byteOffset ||
          actual.getAlignment() != encoded.alignment ||
          actual.getAddressSpace() != encoded.addressSpace ||
          actual.getAccess() != encoded.access ||
          actual.getStorage() != encoded.storage ||
          actual.getAliasGroup() != encoded.aliasGroup ||
          actual.getReuseGroup() != encoded.reuseGroup ||
          actual.getBinding() != encoded.bindingKind ||
          actual.getBindingIndex().value_or(-1) != encoded.bindingIndex)
        return failure();
    }
    return SmallVector<uint8_t>(bytes.begin(), bytes.end());
  }
  const bool supportedTarget =
      (device.getSchemaVersion() == 3 &&
       device.getCodeFormat() == ExecutableCodeFormat::CudaPtxSm90V1) ||
      (device.getSchemaVersion() == 4 &&
       device.getCodeFormat() == ExecutableCodeFormat::CudaPtxSm100V1);
  if (!supportedTarget || abi.getSchemaVersion() != 3 ||
      bundle.getSchemaVersion() != 2)
    return failure();

  Writer devicePrefix;
  devicePrefix.i64(device.getSchemaVersion());
  devicePrefix.u8(static_cast<uint8_t>(device.getCodeFormat()));
  devicePrefix.u8(static_cast<uint8_t>(device.getPolicy()));
  devicePrefix.text(device.getSourceScheduleFingerprint());
  devicePrefix.blob(device.getCode());
  devicePrefix.text(device.getCodeDigest());
  Writer deviceSuffix;
  SmallVector<DeviceEntryOp> entries(
      device.getBody().front().getOps<DeviceEntryOp>());
  deviceSuffix.u64(entries.size());
  for (DeviceEntryOp entry : entries)
    writeEntry(deviceSuffix, entry);
  std::string deviceRoot = digest(devicePrefix.view(), deviceSuffix.view());

  Writer abiPrefix;
  abiPrefix.i64(3);
  abiPrefix.text(abi.getSourcePlanFingerprint());
  abiPrefix.text(abi.getSourceScheduleFingerprint());
  Writer abiSuffix;
  SmallVector<InvocationSlotOp> slots(
      abi.getBody().front().getOps<InvocationSlotOp>());
  abiSuffix.u64(slots.size());
  for (InvocationSlotOp slot : slots)
    writeSlot(abiSuffix, slot);
  std::string abiRoot = digest(abiPrefix.view(), abiSuffix.view());

  Writer writer;
  writer.append(kMagic);
  writer.u32(2);
  writer.append(devicePrefix.view());
  writer.text(deviceRoot);
  writer.append(deviceSuffix.view());
  writer.append(abiPrefix.view());
  writer.text(abiRoot);
  writer.append(abiSuffix.view());
  writer.i64(2);
  writer.text(bundle.getSourceScheduleFingerprint());
  writer.text(deviceRoot);
  writer.text(abiRoot);
  writer.u8(1);
  writer.text(digest(writer.view()));
  SmallVector<uint8_t> bytes = writer.take();
  if (bytes.size() > kMaximumGpuTransportBytes) {
    device.emitOpError(
        "serialized GPU transport exceeds the closed size limit");
    return failure();
  }
  absl::Status loaded = GpuExecutable::Load(bytes).status();
  if (!loaded.ok()) {
    device.emitOpError("serialized GPU transport failed public validation: ")
        << loaded.message();
    return failure();
  }
  return bytes;
}

absl::StatusOr<SmallVector<uint8_t>>
serializeGpuFatbinExecutableBundle(const GpuFatbinExecutableBundle &bundle) {
  try {
    const bool supportedTarget =
        (bundle.deviceSchemaVersion == 5 &&
         bundle.codeFormat == ExecutableCodeFormat::CudaFatbinSm90V1) ||
        (bundle.deviceSchemaVersion == 6 &&
         bundle.codeFormat == ExecutableCodeFormat::CudaFatbinSm100V1);
    if (!supportedTarget || !isDigest(bundle.scheduleFingerprint) ||
        !isDigest(bundle.planFingerprint) || bundle.images.empty() ||
        bundle.images.size() > kMaximumGpuExecutableRecords ||
        bundle.launches.empty() ||
        bundle.launches.size() > kMaximumGpuExecutableRecords ||
        bundle.slots.empty() ||
        bundle.slots.size() > kMaximumGpuExecutableRecords)
      return invalidTransport("fatbin descriptor");

    Writer writer;
    writer.append(kMagic);
    writer.u32(12);
    writer.i64(bundle.deviceSchemaVersion);
    writer.u8(static_cast<uint8_t>(bundle.codeFormat));
    writer.text(bundle.scheduleFingerprint);
    writer.text(bundle.planFingerprint);
    writer.u64(bundle.images.size());
    uint64_t totalCodeBytes = 0;
    for (const auto &image : bundle.images) {
      if (image.empty() || image.size() > kMaximumGpuCodeBytes - totalCodeBytes)
        return invalidTransport("fatbin image descriptor");
      totalCodeBytes += image.size();
      writer.blobBytes(image);
      writer.text(digest(image));
    }
    writer.u64(bundle.launches.size());
    for (auto [ordinal, launch] : llvm::enumerate(bundle.launches)) {
      if (launch.imageIndex >= bundle.images.size() ||
          launch.taskOrdinal != static_cast<int64_t>(ordinal) ||
          launch.sourceTaskOrdinals.empty() ||
          launch.sourceTaskOrdinals.size() > kMaximumGpuExecutableRecords ||
          !isKernelName(launch.kernelName) ||
          !isCudaGeometry(launch.grid, launch.block) ||
          launch.dynamicSharedMemoryBytes >
              kMaximumGpuDynamicSharedMemoryBytes ||
          launch.arguments.empty() ||
          launch.arguments.size() > kMaximumGpuExecutableRecords ||
          launch.functionAttributes.size() > kMaximumGpuExecutableRecords)
        return invalidTransport("fatbin launch descriptor");
      writer.i64(launch.taskOrdinal);
      writer.u64(launch.imageIndex);
      writer.integers(launch.sourceTaskOrdinals);
      writer.text(launch.kernelName);
      for (uint64_t extent : launch.grid)
        writer.u32(extent);
      for (uint64_t extent : launch.block)
        writer.u16(extent);
      writer.u32(launch.dynamicSharedMemoryBytes);
      writer.u16(launch.arguments.size());
      for (const GpuKernelArgument &argument : launch.arguments) {
        writer.u8(static_cast<uint8_t>(argument.kind));
        if (argument.kind == GpuKernelArgumentKind::SlotPointer) {
          writer.i64(argument.slotOrdinal);
          writer.u64(argument.slotByteOffset);
          writer.u8(static_cast<uint8_t>(argument.access));
          continue;
        }
        if (argument.kind == GpuKernelArgumentKind::TensorMap) {
          const GpuTensorMapDescriptor &tensorMap = argument.tensorMap;
          writer.i64(tensorMap.slotOrdinal);
          writer.u64(tensorMap.byteOffset);
          writer.u8(tensorMap.elementByteWidth);
          writer.u8(tensorMap.globalDimensions.size());
          for (uint64_t value : tensorMap.globalDimensions)
            writer.u64(value);
          for (uint64_t value : tensorMap.globalByteStrides)
            writer.u64(value);
          for (uint32_t value : tensorMap.boxDimensions)
            writer.u32(value);
          for (uint32_t value : tensorMap.elementStrides)
            writer.u32(value);
          writer.u8(static_cast<uint8_t>(tensorMap.interleave));
          writer.u8(static_cast<uint8_t>(tensorMap.swizzle));
          writer.u8(static_cast<uint8_t>(tensorMap.l2Promotion));
          writer.u8(static_cast<uint8_t>(tensorMap.oobFill));
          writer.u16(tensorMap.alignment);
          writer.u8(static_cast<uint8_t>(tensorMap.access));
          continue;
        }
        if (argument.kind == GpuKernelArgumentKind::StructuredBytes) {
          writer.u32(argument.structure.size);
          writer.u16(argument.structure.alignment);
          writer.u16(argument.structure.fields.size());
          for (const GpuStructuredField &field : argument.structure.fields) {
            writer.u8(static_cast<uint8_t>(field.kind));
            writer.u32(field.offset);
            if (field.kind == GpuStructuredFieldKind::SlotAddress) {
              writer.i64(field.slotOrdinal);
              writer.u64(field.slotByteOffset);
              writer.u8(static_cast<uint8_t>(field.access));
            } else {
              writer.blobBytes(field.bytes);
            }
          }
          continue;
        }
        writer.u8(static_cast<uint8_t>(argument.scalarType));
        writer.u16(argument.alignment);
        writer.blobBytes(argument.bytes);
      }
      writer.u16(launch.functionAttributes.size());
      for (const GpuFunctionAttribute &attribute : launch.functionAttributes) {
        writer.u8(static_cast<uint8_t>(attribute.kind));
        writer.u64(attribute.value);
      }
      writer.integers(launch.inputSlots);
      writer.integers(launch.outputSlots);
      writer.integers(launch.dependencies);
    }
    if (bundle.eventRealizations.size() > kMaximumGpuExecutableRecords)
      return invalidTransport("fatbin Event realization count");
    writer.u64(bundle.eventRealizations.size());
    int64_t nextMechanismAllocation = 0;
    for (auto [ordinal, realization] :
         llvm::enumerate(bundle.eventRealizations)) {
      const bool validKind =
          static_cast<uint8_t>(realization.kind) <=
          static_cast<uint8_t>(GpuEventRealizationKind::Queue);
      const bool erased =
          validKind && isErasedEventRealization(realization.kind);
      if (realization.eventOrdinal != static_cast<int64_t>(ordinal) ||
          !validKind || realization.coverPieces.empty() ||
          realization.coverPieces.size() > kMaximumGpuExecutableRecords ||
          (erased &&
           (realization.coveringLaunchOrder.empty() ||
            realization.coveringLaunchOrder.size() % 4 != 0 ||
            realization.mechanism || realization.operationCertificate)) ||
          (!erased && (!realization.coveringLaunchOrder.empty() ||
                       !realization.mechanism)))
        return invalidTransport("fatbin Event realization descriptor");
      writer.i64(realization.eventOrdinal);
      writer.u8(static_cast<uint8_t>(realization.kind));
      writer.integers(realization.coveringLaunchOrder);
      writer.u64(realization.coverPieces.size());
      const int64_t relationDimensions =
          realization.coverPieces.front().producer.owner.dimensions;
      for (const GpuEventCoverPiece &piece : realization.coverPieces) {
        if (!validateEventCoverPiece(piece, bundle.launches.size()) ||
            piece.producer.owner.dimensions != relationDimensions)
          return invalidTransport("fatbin Event cover-piece descriptor");
        const int64_t dimensions = piece.producer.owner.dimensions;
        writer.u8(dimensions);
        writer.u8(piece.relationSubset.has_value());
        if (piece.relationSubset)
          writeIntegerSet(writer, *piece.relationSubset);
        writeEventCoverEndpoint(writer, piece.producer);
        writeEventCoverEndpoint(writer, piece.consumer);
      }
      writer.u8(realization.mechanism.has_value());
      if (realization.mechanism) {
        if (realization.mechanism->allocationOrdinal !=
                nextMechanismAllocation++ ||
            !validateEventMechanism(*realization.mechanism,
                                    relationDimensions))
          return invalidTransport("fatbin Event mechanism descriptor");
        writeEventMechanism(writer, *realization.mechanism);
      }
      writer.u8(realization.operationCertificate.has_value());
      if (realization.operationCertificate) {
        const GpuEventOperationCertificate &certificate =
            *realization.operationCertificate;
        if (!isDigest(certificate.compilerIrDigest) ||
            !isDigest(certificate.deviceImageDigest) ||
            certificate.compilerIrDigest !=
                gpuEventOperationCertificateDigest(certificate) ||
            certificate.operations.empty() ||
            certificate.operations.size() > kMaximumGpuExecutableRecords)
          return invalidTransport(
              "fatbin Event operation certificate descriptor");
        for (auto [operationOrdinal, operation] :
             llvm::enumerate(certificate.operations))
          if (operation.ordinal != static_cast<int64_t>(operationOrdinal) ||
              !validateEventOperation(operation, relationDimensions,
                                      bundle.launches.size(),
                                      realization.coverPieces.size()))
            return invalidTransport(
                "fatbin Event operation certificate operation");
        writeEventOperationCertificate(writer, certificate);
      }
    }
    const bool hasPhysicalEventProvenance =
        !bundle.physicalEventProvenance.empty();
    if (hasPhysicalEventProvenance !=
            !bundle.physicalEventProvenanceDigest.empty() ||
        (hasPhysicalEventProvenance &&
         (!isDigest(bundle.physicalEventProvenanceDigest) ||
          bundle.physicalEventProvenanceDigest !=
              digest(
                  ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(
                                        bundle.physicalEventProvenance.data()),
                                    bundle.physicalEventProvenance.size())))))
      return invalidTransport("fatbin physical Event provenance");
    writer.u8(hasPhysicalEventProvenance);
    if (hasPhysicalEventProvenance) {
      writer.text(bundle.physicalEventProvenance);
      writer.text(bundle.physicalEventProvenanceDigest);
    }
    if (bundle.deviceIrSnapshots.size() > kMaximumGpuExecutableRecords)
      return invalidTransport("fatbin device IR snapshot count");
    writer.u64(bundle.deviceIrSnapshots.size());
    for (const GpuDeviceIrSnapshot &snapshot : bundle.deviceIrSnapshots) {
      if (!isDigest(snapshot.mlirBytecodeDigest) ||
          snapshot.operationCounts.empty() ||
          snapshot.operationCounts.size() > kMaximumGpuExecutableRecords)
        return invalidTransport("fatbin device IR snapshot descriptor");
      writer.text(snapshot.mlirBytecodeDigest);
      writer.u64(snapshot.operationCounts.size());
      StringRef previousName;
      for (const GpuDeviceIrOperationCount &operation :
           snapshot.operationCounts) {
        if (operation.operationName.empty() ||
            operation.operationName.size() > 1024 ||
            StringRef(operation.operationName).contains('\0') ||
            (!previousName.empty() &&
             operation.operationName <= previousName) ||
            operation.count <= 0)
          return invalidTransport("fatbin device IR operation inventory");
        writer.text(operation.operationName);
        writer.u64(operation.count);
        previousName = operation.operationName;
      }
    }
    writer.u64(bundle.slots.size());
    for (const GpuSlot &slot : bundle.slots) {
      writer.i64(slot.ordinal);
      writeTensor(writer, slot.tensorType);
      writer.i64(slot.requiredBytes);
      writer.integers(slot.byteStrides);
      writer.i64(slot.byteOffset);
      writer.i64(slot.alignment);
      writer.u8(static_cast<uint8_t>(slot.addressSpace));
      writer.u8(static_cast<uint8_t>(slot.access));
      writer.u8(static_cast<uint8_t>(slot.storage));
      writer.i64(slot.aliasGroup);
      writer.i64(slot.reuseGroup);
      writer.u8(static_cast<uint8_t>(slot.bindingKind));
      writer.i64(slot.bindingIndex);
    }
    writer.text(digest(writer.view()));
    SmallVector<uint8_t> bytes = writer.take();
    if (bytes.size() > kMaximumGpuTransportBytes)
      return invalidTransport("fatbin serialized size");
    absl::Status loaded = GpuExecutable::Load(bytes).status();
    if (!loaded.ok())
      return loaded;
    return bytes;
  } catch (const std::bad_alloc &) {
    return absl::ResourceExhaustedError(
        "insufficient memory to serialize Shuttle GPU fatbin executable");
  } catch (...) {
    return absl::InternalError(
        "unexpected failure while serializing Shuttle GPU fatbin executable");
  }
}

absl::StatusOr<GpuFatbinExecutableBundle>
deserializeGpuFatbinExecutableBundle(ArrayRef<uint8_t> bytes,
                                     MLIRContext *context) {
  if (context == nullptr)
    return absl::InvalidArgumentError(
        "Shuttle GPU fatbin deserialization requires an MLIR context");
  absl::StatusOr<std::shared_ptr<const GpuExecutable>> executable =
      GpuExecutable::Load(bytes);
  if (!executable.ok())
    return executable.status();
  const bool fatbin =
      ((*executable)->deviceSchemaVersion() == 5 &&
       (*executable)->codeFormat() == ExecutableCodeFormat::CudaFatbinSm90V1) ||
      ((*executable)->deviceSchemaVersion() == 6 &&
       (*executable)->codeFormat() == ExecutableCodeFormat::CudaFatbinSm100V1);
  if (!fatbin)
    return absl::InvalidArgumentError(
        "Shuttle physical compiler returned a non-fatbin transport");

  GpuFatbinExecutableBundle bundle;
  bundle.deviceSchemaVersion = (*executable)->deviceSchemaVersion();
  bundle.codeFormat = (*executable)->codeFormat();
  bundle.scheduleFingerprint = (*executable)->scheduleFingerprint().str();
  bundle.planFingerprint = (*executable)->planFingerprint().str();
  bundle.images.assign((*executable)->codeImages().begin(),
                       (*executable)->codeImages().end());
  bundle.launches.assign((*executable)->launches().begin(),
                         (*executable)->launches().end());
  bundle.eventRealizations.assign((*executable)->eventRealizations().begin(),
                                  (*executable)->eventRealizations().end());
  bundle.physicalEventProvenance =
      (*executable)->physicalEventProvenance().str();
  bundle.physicalEventProvenanceDigest =
      (*executable)->physicalEventProvenanceDigest().str();
  bundle.deviceIrSnapshots.assign((*executable)->deviceIrSnapshots().begin(),
                                  (*executable)->deviceIrSnapshots().end());
  bundle.slots.reserve((*executable)->slots().size());
  for (const GpuSlot &encoded : (*executable)->slots()) {
    auto tensor = cast<RankedTensorType>(encoded.tensorType);
    GpuSlot slot = encoded;
    Type elementType;
    if (tensor.getElementType().isBF16())
      elementType = BFloat16Type::get(context);
    else if (tensor.getElementType().isF32())
      elementType = Float32Type::get(context);
    else if (tensor.getElementType().isInteger(1))
      elementType = IntegerType::get(context, 1);
    else if (tensor.getElementType().isInteger(32))
      elementType = IntegerType::get(context, 32);
    else
      return absl::InvalidArgumentError(
          "Shuttle fatbin transport has an unsupported slot element type");
    slot.tensorType = RankedTensorType::get(tensor.getShape(), elementType);
    bundle.slots.push_back(std::move(slot));
  }
  return bundle;
}

} // namespace mlir::shuttle
