// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

namespace {

using namespace mlir;
using namespace mlir::shuttle;

GpuKernelArgument
slotArgument(int64_t slot, ExecutableAccess access = ExecutableAccess::Read) {
  GpuKernelArgument argument;
  argument.kind = GpuKernelArgumentKind::SlotPointer;
  argument.slotOrdinal = slot;
  argument.access = access;
  return argument;
}

GpuAffineExpression dimension(int64_t position) {
  return {GpuAffineExpressionKind::Dimension, position, {}};
}

GpuAffineExpression constant(int64_t value) {
  return {GpuAffineExpressionKind::Constant, value, {}};
}

GpuEventCoverPiece fullRelationCover(int64_t producerLaunch,
                                     int64_t consumerLaunch) {
  GpuAffineMap owner{2, {dimension(0)}};
  GpuAffineMap producerTime{2, {constant(0)}};
  GpuAffineMap consumerTime{2, {constant(1)}};
  return {std::nullopt,
          {producerLaunch, owner, producerTime},
          {consumerLaunch, owner, consumerTime}};
}

GpuEventMechanismAllocation pipelineMechanism() {
  return {0,
          GpuEventPhysicalScope::ThreadBlock,
          16,
          {2, {dimension(0)}},
          {2, {constant(0), dimension(1)}},
          2,
          1,
          128};
}

GpuEventOperationCertificate pipelineCertificate(
    llvm::ArrayRef<uint8_t> image, const GpuEventCoverPiece &piece,
    const GpuEventMechanismAllocation &mechanism) {
  GpuEventOperationCertificate certificate{
      {}, gpuExecutableBundleDigest(image),
      {
          {0,
           GpuEventOperationKind::Notify,
           0,
           piece.producer,
           mechanism.slot,
           mechanism.generationPhase,
           mechanism.scope,
           mechanism.arrivalCount,
           true},
          {1,
           GpuEventOperationKind::Wait,
           0,
           piece.consumer,
           mechanism.slot,
           mechanism.generationPhase,
           mechanism.scope,
           mechanism.arrivalCount,
           true},
      }};
  certificate.compilerIrDigest =
      gpuEventOperationCertificateDigest(certificate);
  return certificate;
}

GpuFatbinExecutableBundle testBundle(MLIRContext &context) {
  auto bf16Matrix =
      RankedTensorType::get({16, 16}, BFloat16Type::get(&context));
  auto f32Vector = RankedTensorType::get({16}, Float32Type::get(&context));
  std::vector<GpuSlot> slots{
      {0,
       bf16Matrix,
       512,
       {32, 2},
       0,
       64,
       ExecutableAddressSpace::Device,
       ExecutableAccess::Read,
       MaterializationStorage::External,
       0,
       0,
       ExecutableBindingKind::Operand,
       0},
      {1,
       f32Vector,
       64,
       {4},
       0,
       64,
       ExecutableAddressSpace::Device,
       ExecutableAccess::ReadWrite,
       MaterializationStorage::Temporary,
       1,
       7,
       ExecutableBindingKind::None,
       -1},
      {2,
       bf16Matrix,
       512,
       {32, 2},
       0,
       64,
       ExecutableAddressSpace::Device,
       ExecutableAccess::Write,
       MaterializationStorage::External,
       2,
       2,
       ExecutableBindingKind::Result,
       0},
  };

  GpuKernelArgument scalar;
  scalar.kind = GpuKernelArgumentKind::ScalarImmediate;
  scalar.scalarType = GpuScalarImmediateType::I32;
  scalar.bytes = {0, 0, 0, 0};
  scalar.alignment = 4;

  GpuKernelArgument packed;
  packed.kind = GpuKernelArgumentKind::PackedBytes;
  packed.bytes = {1, 2, 3, 4, 5, 6, 7, 8};
  packed.alignment = 8;

  GpuKernelArgument tensorMap;
  tensorMap.kind = GpuKernelArgumentKind::TensorMap;
  tensorMap.tensorMap.slotOrdinal = 0;
  tensorMap.tensorMap.elementByteWidth = 2;
  tensorMap.tensorMap.globalDimensions = {16, 16};
  tensorMap.tensorMap.globalByteStrides = {32};
  tensorMap.tensorMap.boxDimensions = {16, 16};
  tensorMap.tensorMap.elementStrides = {1, 1};
  tensorMap.tensorMap.alignment = 64;

  GpuKernelArgument structured;
  structured.kind = GpuKernelArgumentKind::StructuredBytes;
  structured.structure.size = 32;
  structured.structure.alignment = 8;
  structured.structure.fields = {
      {GpuStructuredFieldKind::SlotAddress, 0, 0, 0, {}},
      {GpuStructuredFieldKind::ImmediateBytes,
       8,
       -1,
       0,
       {16, 0, 0, 0, 16, 0, 0, 0}},
  };

  GpuLaunch prepare{
      0,
      "prepare_tiles",
      0,
      0,
      "",
      {1, 1, 1},
      {128, 1, 1},
      0,
      {slotArgument(0), scalar, packed, tensorMap, structured,
       slotArgument(1, ExecutableAccess::Write)},
      {},
      {0},
      {1},
      {},
      {0},
  };
  GpuLaunch finish{
      1,
      "finish_tiles",
      0,
      0,
      "",
      {2, 1, 1},
      {64, 1, 1},
      0,
      {slotArgument(1, ExecutableAccess::ReadWrite),
       slotArgument(2, ExecutableAccess::Write)},
      {{GpuFunctionAttributeKind::MaximumDynamicSharedMemoryBytes, 0}},
      {1},
      {1, 2},
      {0},
      {1},
  };
  return {5,
          ExecutableCodeFormat::CudaFatbinSm90V1,
          std::string(64, 'a'),
          std::string(64, 'b'),
          {{0x50, 0xed, 0x55, 0xba, 1, 0, 0, 0}},
          {std::move(prepare), std::move(finish)},
          std::move(slots)};
}

TEST(GpuFatbinExecutableTest, RoundTripsDynamicTypedLaunchRecipe) {
  MLIRContext context;
  GpuFatbinExecutableBundle bundle = testBundle(context);
  auto serialized = serializeGpuFatbinExecutableBundle(bundle);
  ASSERT_TRUE(serialized.ok()) << serialized.status();
  auto executable = GpuExecutable::Load(*serialized);
  ASSERT_TRUE(executable.ok()) << executable.status();

  EXPECT_EQ((*executable)->codeFormat(),
            ExecutableCodeFormat::CudaFatbinSm90V1);
  ASSERT_EQ((*executable)->codeImages().size(), 1);
  EXPECT_EQ((*executable)->codeBytes(),
            llvm::ArrayRef<uint8_t>(bundle.images[0]));
  auto identity = gpuFatbinCodeIdentity((*executable)->codeImages(),
                                        (*executable)->launches());
  ASSERT_TRUE(identity.ok()) << identity.status();
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(*identity), (*executable)->codeBytes());
  ASSERT_EQ((*executable)->launches().size(), 2);
  const GpuLaunch &prepare = (*executable)->launches()[0];
  EXPECT_EQ(prepare.kernelName, "prepare_tiles");
  EXPECT_EQ(prepare.dynamicSharedMemoryBytes, 0);
  EXPECT_TRUE(prepare.functionAttributes.empty());
  ASSERT_EQ(prepare.arguments.size(), 6);
  EXPECT_EQ(prepare.arguments[0].kind, GpuKernelArgumentKind::SlotPointer);
  EXPECT_EQ(prepare.arguments[1].scalarType, GpuScalarImmediateType::I32);
  EXPECT_EQ(prepare.arguments[1].bytes, (std::vector<uint8_t>{0, 0, 0, 0}));
  EXPECT_EQ(prepare.arguments[3].kind, GpuKernelArgumentKind::TensorMap);
  EXPECT_EQ(prepare.arguments[3].tensorMap.globalByteStrides,
            (std::vector<uint64_t>{32}));
  EXPECT_EQ(prepare.arguments[4].kind, GpuKernelArgumentKind::StructuredBytes);
  EXPECT_EQ(prepare.arguments[4].structure.fields[0].kind,
            GpuStructuredFieldKind::SlotAddress);
  EXPECT_EQ((*executable)->slots()[1].access, ExecutableAccess::ReadWrite);
  EXPECT_EQ((*executable)->slots()[1].storage,
            MaterializationStorage::Temporary);

  llvm::SmallVector<uint8_t> corrupted(*serialized);
  corrupted.back() ^= 1;
  EXPECT_FALSE(GpuExecutable::Load(corrupted).ok());
}

TEST(GpuFatbinExecutableTest,
     AdmitsSupportedSharedMemoryOptInAndRejectsOverflow) {
  MLIRContext context;
  for (ExecutableCodeFormat format :
       {ExecutableCodeFormat::CudaFatbinSm90V1,
        ExecutableCodeFormat::CudaFatbinSm100V1}) {
    for (uint64_t sharedBytes : {225 * 1024, 227 * 1024}) {
      auto bundle = testBundle(context);
      bundle.codeFormat = format;
      bundle.deviceSchemaVersion =
          format == ExecutableCodeFormat::CudaFatbinSm90V1 ? 5 : 6;
      bundle.launches[1].dynamicSharedMemoryBytes = sharedBytes;
      bundle.launches[1].functionAttributes[0].value = 227 * 1024;
      auto serialized = serializeGpuFatbinExecutableBundle(bundle);
      ASSERT_TRUE(serialized.ok()) << serialized.status();
      auto executable = GpuExecutable::Load(*serialized);
      ASSERT_TRUE(executable.ok()) << executable.status();
      EXPECT_EQ((*executable)->launches()[1].dynamicSharedMemoryBytes,
                sharedBytes);
      EXPECT_EQ((*executable)->launches()[1].functionAttributes[0].value,
                227 * 1024);
    }
    auto bundle = testBundle(context);
    bundle.codeFormat = format;
    bundle.deviceSchemaVersion =
        format == ExecutableCodeFormat::CudaFatbinSm90V1 ? 5 : 6;
    bundle.launches[1].dynamicSharedMemoryBytes = 227 * 1024 + 1;
    bundle.launches[1].functionAttributes[0].value = 227 * 1024 + 1;
    EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
    bundle.launches[1].dynamicSharedMemoryBytes = 225 * 1024;
    EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
    bundle.launches[1].functionAttributes[0].value = 225 * 1024 - 1;
    EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
  }
}

TEST(GpuFatbinExecutableTest, LargeExternalBuffersRetainExactTypedDescriptors) {
  MLIRContext context;
  for (ExecutableCodeFormat format :
       {ExecutableCodeFormat::CudaFatbinSm90V1,
        ExecutableCodeFormat::CudaFatbinSm100V1}) {
    auto bundle = testBundle(context);
    bundle.codeFormat = format;
    bundle.deviceSchemaVersion =
        format == ExecutableCodeFormat::CudaFatbinSm90V1 ? 5 : 6;
    for (size_t ordinal : {0, 2}) {
      auto &slot = bundle.slots[ordinal];
      slot.tensorType =
          RankedTensorType::get({4096, 14336}, BFloat16Type::get(&context));
      slot.requiredBytes = 117440512;
      slot.byteStrides = {28672, 2};
    }
    auto serialized = serializeGpuFatbinExecutableBundle(bundle);
    ASSERT_TRUE(serialized.ok()) << serialized.status();
    auto executable = GpuExecutable::Load(*serialized);
    ASSERT_TRUE(executable.ok()) << executable.status();
    ASSERT_EQ((*executable)->externalBindings().size(), 2);
    for (const auto &binding : (*executable)->externalBindings()) {
      EXPECT_EQ(binding.requiredBytes, 117440512);
      EXPECT_EQ(cast<RankedTensorType>(binding.tensorType).getShape(),
                (ArrayRef<int64_t>{4096, 14336}));
    }
    auto invalid = bundle;
    invalid.slots[0].requiredBytes += 2;
    EXPECT_FALSE(serializeGpuFatbinExecutableBundle(invalid).ok());
    invalid = bundle;
    invalid.slots[0].byteStrides[0] += 2;
    EXPECT_FALSE(serializeGpuFatbinExecutableBundle(invalid).ok());
    invalid = bundle;
    invalid.slots[0].bindingIndex = 1;
    EXPECT_FALSE(serializeGpuFatbinExecutableBundle(invalid).ok());
    invalid = bundle;
    invalid.slots[0].tensorType = RankedTensorType::get(
        {std::numeric_limits<int64_t>::max(), 2}, BFloat16Type::get(&context));
    EXPECT_FALSE(serializeGpuFatbinExecutableBundle(invalid).ok());
  }
}

TEST(GpuFatbinExecutableTest, NativeTemporarySlotsKeepTheirAllocationLimit) {
  MLIRContext context;
  auto bundle = testBundle(context);
  auto &temporary = bundle.slots[1];
  temporary.tensorType =
      RankedTensorType::get({8388608}, Float32Type::get(&context));
  temporary.requiredBytes = 33554432;
  temporary.byteStrides = {4};
  auto serialized = serializeGpuFatbinExecutableBundle(bundle);
  ASSERT_TRUE(serialized.ok()) << serialized.status();
  auto executable = GpuExecutable::Load(*serialized);
  ASSERT_TRUE(executable.ok()) << executable.status();
  EXPECT_EQ((*executable)->slots()[1].requiredBytes, 33554432);
  temporary.tensorType =
      RankedTensorType::get({8388609}, Float32Type::get(&context));
  temporary.requiredBytes += 4;
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
  temporary.tensorType =
      RankedTensorType::get({29360128}, Float32Type::get(&context));
  temporary.requiredBytes = 117440512;
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
}

TEST(GpuFatbinExecutableTest, SelectsImagePerLaunchWithIdenticalSymbols) {
  MLIRContext context;
  auto bundle = testBundle(context);
  bundle.images.push_back({0x50, 0xed, 0x55, 0xba, 2, 0, 0, 0});
  bundle.launches[0].kernelName = "shared_kernel";
  bundle.launches[1].kernelName = "shared_kernel";
  bundle.launches[1].imageIndex = 1;
  auto &output = bundle.slots[2];
  output.tensorType = RankedTensorType::get({}, Float32Type::get(&context));
  output.requiredBytes = 4;
  output.byteStrides.clear();
  auto serialized = serializeGpuFatbinExecutableBundle(bundle);
  ASSERT_TRUE(serialized.ok()) << serialized.status();
  auto executable = GpuExecutable::Load(*serialized);
  ASSERT_TRUE(executable.ok()) << executable.status();
  ASSERT_EQ((*executable)->codeImages().size(), 2);
  EXPECT_TRUE((*executable)->codeBytes().empty());
  EXPECT_EQ((*executable)->totalCodeBytes(), 16);
  for (size_t index = 0; index < 2; ++index) {
    const auto &launch = (*executable)->launches()[index];
    EXPECT_EQ(launch.kernelName, "shared_kernel");
    EXPECT_EQ(launch.imageIndex, index);
    EXPECT_EQ((*executable)->codeImage(launch.imageIndex),
              llvm::ArrayRef<uint8_t>(bundle.images[index]));
    EXPECT_EQ(launch.codeDigest, gpuExecutableBundleDigest(bundle.images[index]));
  }
  EXPECT_EQ(cast<RankedTensorType>((*executable)->slots()[2].tensorType).getRank(), 0);
  EXPECT_EQ((*executable)->slots()[2].requiredBytes, 4);
  auto decoded = deserializeGpuFatbinExecutableBundle(*serialized, &context);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  auto reserialized = serializeGpuFatbinExecutableBundle(*decoded);
  ASSERT_TRUE(reserialized.ok()) << reserialized.status();
  EXPECT_EQ(*reserialized, *serialized);
  auto identity = gpuFatbinCodeIdentity(decoded->images, decoded->launches);
  ASSERT_TRUE(identity.ok()) << identity.status();
  auto changedBoundary = *decoded;
  changedBoundary.images[0].push_back(changedBoundary.images[1].front());
  changedBoundary.images[1].erase(changedBoundary.images[1].begin());
  auto changedIdentity =
      gpuFatbinCodeIdentity(changedBoundary.images, changedBoundary.launches);
  ASSERT_TRUE(changedIdentity.ok()) << changedIdentity.status();
  // Concatenation alone would lose the image boundary and alias these bundles.
  EXPECT_NE(*changedIdentity, *identity);
}

TEST(GpuFatbinExecutableTest, RejectsMissingLaunchImageAndCorruptImageDigest) {
  MLIRContext context;
  auto bundle = testBundle(context);
  bundle.launches[1].imageIndex = 1;
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
  bundle.launches[1].imageIndex = 0;
  auto serialized = serializeGpuFatbinExecutableBundle(bundle);
  ASSERT_TRUE(serialized.ok()) << serialized.status();
  // v12 image bytes follow magic, version, target, two digests, image count,
  // and the first blob length. Preserve the outer digest to test image binding.
  constexpr size_t imageOffset = 8 + 4 + 8 + 1 + 2 * (8 + 64) + 8 + 8;
  const size_t imageIndexOffset = imageOffset + bundle.images[0].size() +
                                  (8 + 64) + 8 + 8;
  auto invalidMapping = *serialized;
  invalidMapping[imageIndexOffset] = 1;
  const size_t mappingRootOffset = invalidMapping.size() - 64;
  const auto mappingRoot = gpuExecutableBundleDigest(
      llvm::ArrayRef<uint8_t>(invalidMapping).take_front(mappingRootOffset - 8));
  std::copy(mappingRoot.begin(), mappingRoot.end(),
            invalidMapping.begin() + mappingRootOffset);
  EXPECT_FALSE(GpuExecutable::Load(invalidMapping).ok());
  (*serialized)[imageOffset] ^= 1;
  const size_t rootOffset = serialized->size() - 64;
  const auto root = gpuExecutableBundleDigest(
      llvm::ArrayRef<uint8_t>(*serialized).take_front(rootOffset - 8));
  std::copy(root.begin(), root.end(), serialized->begin() + rootOffset);
  EXPECT_FALSE(GpuExecutable::Load(*serialized).ok());
}

TEST(GpuFatbinExecutableTest, RejectsMissingScratchDependency) {
  MLIRContext context;
  GpuFatbinExecutableBundle bundle = testBundle(context);
  bundle.launches[1].dependencies.clear();
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
}

TEST(GpuFatbinExecutableTest, RejectsTensorMapOutsideBoundSlot) {
  MLIRContext context;
  GpuFatbinExecutableBundle bundle = testBundle(context);
  bundle.launches[0].arguments[3].tensorMap.globalDimensions = {17, 16};
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
}

TEST(GpuFatbinExecutableTest, RejectsOverlappingStructuredFields) {
  MLIRContext context;
  GpuFatbinExecutableBundle bundle = testBundle(context);
  bundle.launches[0].arguments[4].structure.fields[1].offset = 4;
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
}

TEST(GpuFatbinExecutableTest, RoundTripsScalarEventWithoutInventingDimensions) {
  MLIRContext context;
  auto bundle = testBundle(context);
  GpuEventCoverPiece cover{
      GpuIntegerSet{0, {constant(0)}, {true}},
      {0, {0, {constant(0)}}, {0, {constant(0)}}},
      {1, {0, {constant(0)}}, {0, {constant(1)}}}};
  bundle.eventRealizations = {
      {0, GpuEventRealizationKind::StreamOrder, {0, 0, 1, 0}, {cover}}};
  auto serialized = serializeGpuFatbinExecutableBundle(bundle);
  ASSERT_TRUE(serialized.ok()) << serialized.status();
  auto executable = GpuExecutable::Load(*serialized);
  ASSERT_TRUE(executable.ok()) << executable.status();
  const auto &decoded = (*executable)->eventRealizations()[0].coverPieces[0];
  EXPECT_EQ(decoded.producer.owner.dimensions, 0);
  EXPECT_EQ(decoded.consumer.localTime.results[0].payload, 1);
  ASSERT_TRUE(decoded.relationSubset.has_value());
  EXPECT_EQ(decoded.relationSubset->dimensions, 0);
  // A rank-zero relation has one point, but has no coordinate dimension zero.
  bundle.eventRealizations[0].coverPieces[0].producer.owner.results[0] = dimension(0);
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
}

TEST(GpuFatbinExecutableTest, RoundTripsPipelineEventRealization) {
  MLIRContext context;
  GpuFatbinExecutableBundle bundle = testBundle(context);
  GpuEventCoverPiece cover = fullRelationCover(0, 1);
  GpuEventMechanismAllocation mechanism = pipelineMechanism();
  bundle.eventRealizations = {
      {0,
       GpuEventRealizationKind::Pipeline,
       {},
       {cover},
       mechanism,
       pipelineCertificate(bundle.images.front(), cover, mechanism)},
  };
  auto serialized = serializeGpuFatbinExecutableBundle(bundle);
  ASSERT_TRUE(serialized.ok()) << serialized.status();
  ASSERT_GE(serialized->size(), 12u);
  EXPECT_EQ((*serialized)[8], 12u);
  auto executable = GpuExecutable::Load(*serialized);
  ASSERT_TRUE(executable.ok()) << executable.status();
  ASSERT_EQ((*executable)->eventRealizations().size(), 1u);
  const GpuEventRealization &realization =
      (*executable)->eventRealizations().front();
  EXPECT_EQ(realization.kind, GpuEventRealizationKind::Pipeline);
  EXPECT_TRUE(realization.coveringLaunchOrder.empty());
  ASSERT_EQ(realization.coverPieces.size(), 1u);
  ASSERT_TRUE(realization.mechanism);
  EXPECT_EQ(realization.mechanism->allocationOrdinal, 0);
  EXPECT_EQ(realization.mechanism->scope,
            GpuEventPhysicalScope::ThreadBlock);
  EXPECT_EQ(realization.mechanism->capacity, 16);
  EXPECT_EQ(realization.mechanism->generationPhase.results.size(), 2u);
  EXPECT_EQ(realization.mechanism->generationCount, 2);
  EXPECT_EQ(realization.mechanism->arrivalCount, 1);
  EXPECT_EQ(realization.mechanism->transactionBytes, 128);
  ASSERT_TRUE(realization.operationCertificate);
  EXPECT_EQ(realization.operationCertificate->deviceImageDigest,
            gpuExecutableBundleDigest((*executable)->codeBytes()));
  ASSERT_EQ(realization.operationCertificate->operations.size(), 2u);
  EXPECT_EQ(realization.operationCertificate->operations[0].kind,
            GpuEventOperationKind::Notify);
  EXPECT_EQ(realization.operationCertificate->operations[1].kind,
            GpuEventOperationKind::Wait);
}

TEST(GpuFatbinExecutableTest, EventTransactionsKeepTheirIndependentByteLimit) {
  MLIRContext context;
  auto bundle = testBundle(context);
  auto cover = fullRelationCover(0, 1);
  auto mechanism = pipelineMechanism();
  mechanism.transactionBytes = 33554432;
  bundle.eventRealizations = {
      {0,
       GpuEventRealizationKind::Pipeline,
       {},
       {cover},
       mechanism,
       pipelineCertificate(bundle.images.front(), cover, mechanism)}};
  auto serialized = serializeGpuFatbinExecutableBundle(bundle);
  ASSERT_TRUE(serialized.ok()) << serialized.status();
  auto executable = GpuExecutable::Load(*serialized);
  ASSERT_TRUE(executable.ok()) << executable.status();
  EXPECT_EQ((*executable)->eventRealizations()[0].mechanism->transactionBytes,
            33554432);
  bundle.eventRealizations[0].mechanism->transactionBytes = 33554433;
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
}

TEST(GpuFatbinExecutableTest, RoundTripsStructuredDeviceIrSnapshots) {
  MLIRContext context;
  GpuFatbinExecutableBundle bundle = testBundle(context);
  bundle.deviceIrSnapshots = {
      {std::string(64, 'a'), {{"arith.addf", 7}, {"gpu.barrier", 2}}},
      {std::string(64, 'b'), {{"nvgpu.mbarrier.arrive", 3}}},
  };

  auto serialized = serializeGpuFatbinExecutableBundle(bundle);
  ASSERT_TRUE(serialized.ok()) << serialized.status();
  auto executable = GpuExecutable::Load(*serialized);
  ASSERT_TRUE(executable.ok()) << executable.status();
  ASSERT_EQ((*executable)->deviceIrSnapshots().size(), 2u);
  EXPECT_EQ((*executable)->deviceIrSnapshots()[0].mlirBytecodeDigest,
            std::string(64, 'a'));
  EXPECT_EQ((*executable)->deviceIrSnapshots()[0].operationCounts.size(), 2u);
  EXPECT_EQ(
      (*executable)->deviceIrSnapshots()[0].operationCounts[0].operationName,
      "arith.addf");
  EXPECT_EQ((*executable)->deviceIrSnapshots()[0].operationCounts[0].count, 7);
  auto decoded = deserializeGpuFatbinExecutableBundle(*serialized, &context);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->deviceIrSnapshots.size(), 2u);

  auto invalid = bundle;
  std::swap(invalid.deviceIrSnapshots[0].operationCounts[0],
            invalid.deviceIrSnapshots[0].operationCounts[1]);
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(invalid).ok());
  invalid = bundle;
  invalid.deviceIrSnapshots[0].operationCounts[1].operationName = "arith.addf";
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(invalid).ok());
  invalid = bundle;
  invalid.deviceIrSnapshots[0].operationCounts[0].count = 0;
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(invalid).ok());
  invalid = bundle;
  invalid.deviceIrSnapshots[0].operationCounts.clear();
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(invalid).ok());
  invalid = bundle;
  invalid.deviceIrSnapshots[0].mlirBytecodeDigest = std::string(64, 'z');
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(invalid).ok());
}

TEST(GpuFatbinExecutableTest,
     RoundTripsPhysicalEventProvenanceSeparatelyFromSourceRealizations) {
  MLIRContext context;
  GpuFatbinExecutableBundle bundle = testBundle(context);
  bundle.physicalEventProvenance =
      R"json({"fingerprint":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","physical_schedule_fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","schedule":{"allocations":[],"fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","realizations":[{"allocation_ids":[],"event":{"event_domain":{"axes":[],"kind":"event_domain","name":"internal events"},"generation_policy":"per_invocation","initial_count":{"counts":[{"coordinate":[],"value":1}],"provenance":"notify indegree"},"memory_scope":"cta","name":"internal","notify_relation":{"pairs":[{"source":[],"target":[]}],"source_domain":{"axes":[],"kind":"task_family","name":"producer","placement":"worker"},"target_domain":{"axes":[],"kind":"event_domain","name":"internal events"}},"required_dependence":{"relation":{"pairs":[{"source":[],"target":[]}],"source_domain":{"axes":[],"kind":"task_family","name":"producer","placement":"worker"},"target_domain":{"axes":[],"kind":"task_family","name":"consumer","placement":"worker"}},"visibility":{"acquire_before_consumer":true,"release_on_notify":true,"scope":"cta"}},"scheduling_mode":null,"trigger_relation":{"pairs":[{"source":[],"target":[]}],"source_domain":{"axes":[],"kind":"event_domain","name":"internal events"},"target_domain":{"axes":[],"kind":"task_family","name":"consumer","placement":"worker"}},"visibility":{"acquire_before_consumer":true,"release_on_notify":true,"scope":"cta"}},"kind":"erased_program_order","mechanism":"program order","plan":"internal"}],"schema_version":3},"schema_version":1})json";
  bundle.physicalEventProvenanceDigest = gpuExecutableBundleDigest(
      ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(
                            bundle.physicalEventProvenance.data()),
                        bundle.physicalEventProvenance.size()));

  auto serialized = serializeGpuFatbinExecutableBundle(bundle);
  ASSERT_TRUE(serialized.ok()) << serialized.status();
  auto executable = GpuExecutable::Load(*serialized);
  ASSERT_TRUE(executable.ok()) << executable.status();
  EXPECT_TRUE((*executable)->eventRealizations().empty());
  EXPECT_EQ((*executable)->physicalEventProvenance(),
            bundle.physicalEventProvenance);
  EXPECT_EQ((*executable)->physicalEventProvenanceDigest(),
            bundle.physicalEventProvenanceDigest);
  ASSERT_TRUE((*executable)->internalEventSchedule());
  ASSERT_EQ((*executable)->internalEventSchedule()->regions.size(), 1u);
  EXPECT_EQ(
      (*executable)->internalEventSchedule()->regions[0].realizations[0].kind,
      GpuInternalEventRealizationKind::ErasedProgramOrder);
}

TEST(GpuFatbinExecutableTest, RejectsMalformedEventOperationCertificate) {
  MLIRContext context;
  GpuFatbinExecutableBundle bundle = testBundle(context);
  GpuEventCoverPiece cover = fullRelationCover(0, 1);
  GpuEventMechanismAllocation mechanism = pipelineMechanism();
  GpuEventOperationCertificate certificate =
      pipelineCertificate(bundle.images.front(), cover, mechanism);
  bundle.eventRealizations = {{0,
                               GpuEventRealizationKind::Pipeline,
                               {},
                               {cover},
                               mechanism,
                               certificate}};

  bundle.eventRealizations[0].operationCertificate->compilerIrDigest =
      std::string(64, '0');
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());

  bundle.eventRealizations[0].operationCertificate = certificate;
  bundle.eventRealizations[0]
      .operationCertificate->operations[1]
      .coverPieceOrdinal = 1;
  bundle.eventRealizations[0].operationCertificate->compilerIrDigest =
      gpuEventOperationCertificateDigest(
          *bundle.eventRealizations[0].operationCertificate);
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
}

TEST(GpuFatbinExecutableTest, RejectsMissingOrExtraEventMechanism) {
  MLIRContext context;
  GpuFatbinExecutableBundle bundle = testBundle(context);
  bundle.eventRealizations = {
      {0,
       GpuEventRealizationKind::Pipeline,
       {},
       {fullRelationCover(0, 1)},
       std::nullopt},
  };
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());

  bundle.eventRealizations = {
      {0,
       GpuEventRealizationKind::ProgramOrder,
       {0, 0, 0, 1},
       {fullRelationCover(0, 0)},
       pipelineMechanism()},
  };
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
}

TEST(GpuFatbinExecutableTest, RejectsEmptyGenerationPhaseMap) {
  MLIRContext context;
  GpuFatbinExecutableBundle bundle = testBundle(context);
  GpuEventMechanismAllocation mechanism = pipelineMechanism();
  mechanism.generationPhase.results.clear();
  bundle.eventRealizations = {
      {0,
       GpuEventRealizationKind::Pipeline,
       {},
       {fullRelationCover(0, 1)},
       std::move(mechanism)},
  };
  EXPECT_FALSE(serializeGpuFatbinExecutableBundle(bundle).ok());
}

} // namespace
