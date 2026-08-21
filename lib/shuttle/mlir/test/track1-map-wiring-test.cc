// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "shuttle/Runtime/GpuFfi.h"
#include "shuttle/Transforms/Observer.h"
#include "shuttle/Transforms/XlaRegistration.h"
#include "stablehlo/dialect/Register.h"
#include "tools/cpp/runfiles/runfiles.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/invoke.h"
#include "xla/pjrt/stablehlo_module_transform.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {

constexpr llvm::StringLiteral kRoundTripSourceOrderedOptions =
    R"json({"execution_mode":"stablehlo_round_trip","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kRoundTripFastOptions =
    R"json({"execution_mode":"stablehlo_round_trip","numerics":"fast","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kAbi21GpuSourceOrderedOptions =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kAbi21GpuFastOptions =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kAbi22GpuSourceOrderedOptions =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kAbi22GpuFastOptions =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr std::array<llvm::StringLiteral, 4> kGpuPolicies = {
    kAbi21GpuSourceOrderedOptions, kAbi21GpuFastOptions,
    kAbi22GpuSourceOrderedOptions, kAbi22GpuFastOptions};
constexpr llvm::StringLiteral kAbi22FoldRegionSourceOrderedOptions =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kAbi22FoldRegionFastOptions =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr std::array<llvm::StringLiteral, 2> kAbi22FoldRegionPolicies = {
    kAbi22FoldRegionSourceOrderedOptions, kAbi22FoldRegionFastOptions};

// Emitted by JAX 0.10.1 from:
// jax.jit(lambda x: jnp.abs((-x.astype(jnp.float32)).astype(jnp.bfloat16)))
constexpr llvm::StringLiteral kRankTwoMixedMapProgram = R"mlir(
module @jit__lambda attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x3xbf16>) -> (tensor<2x3xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2x3xbf16>) -> tensor<2x3xf32>
    %1 = stablehlo.negate %0 : tensor<2x3xf32>
    %2 = stablehlo.convert %1 : (tensor<2x3xf32>) -> tensor<2x3xbf16>
    %3 = stablehlo.abs %2 : tensor<2x3xbf16>
    return %3 : tensor<2x3xbf16>
  }
}
)mlir";

constexpr llvm::StringLiteral kRankThreeMixedMapProgram = R"mlir(
module @jit__lambda attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x3x5xbf16>) -> (tensor<2x3x5xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2x3x5xbf16>) -> tensor<2x3x5xf32>
    %1 = stablehlo.negate %0 : tensor<2x3x5xf32>
    %2 = stablehlo.convert %1 : (tensor<2x3x5xf32>) -> tensor<2x3x5xbf16>
    %3 = stablehlo.abs %2 : tensor<2x3x5xbf16>
    return %3 : tensor<2x3x5xbf16>
  }
}
)mlir";

// Emitted by JAX 0.10.1 from the Track 1 value and analytic-gradient subject.
constexpr llvm::StringLiteral kRankTwoValueAndGradientProgram = R"mlir(
module @jit_mixed_map_value_and_gradient attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x3xbf16>) -> (tensor<2x3xbf16> {jax.result_info = "result[0]"}, tensor<2x3xbf16> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2x3xbf16>) -> tensor<2x3xf32>
    %1 = stablehlo.negate %0 : tensor<2x3xf32>
    %2 = stablehlo.convert %1 : (tensor<2x3xf32>) -> tensor<2x3xbf16>
    %3 = stablehlo.abs %2 : tensor<2x3xbf16>
    %4 = stablehlo.convert %3 : (tensor<2x3xbf16>) -> tensor<2x3xf32>
    %5 = stablehlo.convert %arg0 : (tensor<2x3xbf16>) -> tensor<2x3xf32>
    %6 = stablehlo.divide %4, %5 : tensor<2x3xf32>
    %7 = stablehlo.convert %6 : (tensor<2x3xf32>) -> tensor<2x3xbf16>
    return %3, %7 : tensor<2x3xbf16>, tensor<2x3xbf16>
  }
}
)mlir";

constexpr llvm::StringLiteral kRankThreeValueAndGradientProgram = R"mlir(
module @jit_mixed_map_value_and_gradient attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x3x5xbf16>) -> (tensor<2x3x5xbf16> {jax.result_info = "result[0]"}, tensor<2x3x5xbf16> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2x3x5xbf16>) -> tensor<2x3x5xf32>
    %1 = stablehlo.negate %0 : tensor<2x3x5xf32>
    %2 = stablehlo.convert %1 : (tensor<2x3x5xf32>) -> tensor<2x3x5xbf16>
    %3 = stablehlo.abs %2 : tensor<2x3x5xbf16>
    %4 = stablehlo.convert %3 : (tensor<2x3x5xbf16>) -> tensor<2x3x5xf32>
    %5 = stablehlo.convert %arg0 : (tensor<2x3x5xbf16>) -> tensor<2x3x5xf32>
    %6 = stablehlo.divide %4, %5 : tensor<2x3x5xf32>
    %7 = stablehlo.convert %6 : (tensor<2x3x5xf32>) -> tensor<2x3x5xbf16>
    return %3, %7 : tensor<2x3x5xbf16>, tensor<2x3x5xbf16>
  }
}
)mlir";

constexpr llvm::StringLiteral kUnsupportedMapBodyProgram = R"mlir(
module @jit__lambda {
  func.func public @main(%arg0: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<2x3xbf16>) -> tensor<2x3xf32>
    %1 = stablehlo.tanh %0 : tensor<2x3xf32>
    %2 = stablehlo.convert %1 : (tensor<2x3xf32>) -> tensor<2x3xbf16>
    %3 = stablehlo.abs %2 : tensor<2x3xbf16>
    return %3 : tensor<2x3xbf16>
  }
}
)mlir";

constexpr llvm::StringLiteral kF32BoundaryProgram = R"mlir(
module @jit__lambda {
  func.func public @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.negate %arg0 : tensor<2x3xf32>
    %1 = stablehlo.negate %0 : tensor<2x3xf32>
    %2 = stablehlo.abs %1 : tensor<2x3xf32>
    return %2 : tensor<2x3xf32>
  }
}
)mlir";

constexpr llvm::StringLiteral kInternalBf16Program = R"mlir(
module @jit__lambda {
  func.func public @main(%arg0: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<2x3xbf16>) -> tensor<2x3xf32>
    %1 = stablehlo.convert %0 : (tensor<2x3xf32>) -> tensor<2x3xbf16>
    %2 = stablehlo.convert %1 : (tensor<2x3xbf16>) -> tensor<2x3xf32>
    %3 = stablehlo.convert %2 : (tensor<2x3xf32>) -> tensor<2x3xbf16>
    %4 = stablehlo.abs %3 : tensor<2x3xbf16>
    return %4 : tensor<2x3xbf16>
  }
}
)mlir";

class RecordingObserver final : public mlir::shuttle::ShuttlePipelineObserver {
public:
  void observe(const mlir::shuttle::ShuttlePipelineEvent &event) const final {
    std::lock_guard<std::mutex> lock(mutex);
    events.push_back(event);
  }

  std::vector<mlir::shuttle::ShuttlePipelineEvent> snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return events;
  }

private:
  mutable std::mutex mutex;
  mutable std::vector<mlir::shuttle::ShuttlePipelineEvent> events;
};

mlir::OwningOpRef<mlir::ModuleOp> parseModule(mlir::MLIRContext &context,
                                              llvm::StringRef source) {
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::func::FuncDialect>();
  mlir::stablehlo::registerAllDialects(dialects);
  context.appendDialectRegistry(dialects);
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream output(result);
  module.print(output);
  output.flush();
  return result;
}

std::optional<std::string> fixtureSource(llvm::StringRef filename) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  if (!runfiles)
    return std::nullopt;
  std::ifstream input(runfiles->Rlocation(
      (llvm::Twine("shuttle_mlir/test/Inputs/") + filename).str()));
  std::ostringstream source;
  source << input.rdbuf();
  if (!input)
    return std::nullopt;
  return source.str();
}

absl::Status runRegisteredShuttleTransform(mlir::ModuleOp module,
                                           llvm::StringRef policy) {
  absl::Status registration = mlir::shuttle::registerShuttleXlaTransform();
  if (!registration.ok())
    return registration;
  return xla::StablehloModuleTransformRegistry::Global().Run(
      module, policy.str(), "shuttle");
}

size_t countOccurrences(llvm::StringRef text, llvm::StringRef pattern) {
  size_t count = 0;
  size_t offset = 0;
  while ((offset = text.find(pattern, offset)) != llvm::StringRef::npos) {
    ++count;
    offset += pattern.size();
  }
  return count;
}

void expectObservedSuccess(
    const std::vector<mlir::shuttle::ShuttlePipelineEvent> &events,
    mlir::shuttle::ShuttlePipelinePhase middlePhase) {
  using mlir::shuttle::ShuttlePipelinePhase;
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0].phase(), ShuttlePipelinePhase::AlgebraCoverage);
  EXPECT_EQ(events[1].phase(), middlePhase);
  EXPECT_EQ(events[2].phase(), ShuttlePipelinePhase::FinalErasure);
  EXPECT_GT(events[0].invocationId(), 0);
  EXPECT_EQ(events[0].invocationId(), events[1].invocationId());
  EXPECT_EQ(events[0].invocationId(), events[2].invocationId());
  EXPECT_EQ(events[0].identity(), events[2].identity());
  EXPECT_FALSE(events[0].regionMembership().empty());
  EXPECT_EQ(events[0].regionMembership(), events[1].regionMembership());
  EXPECT_FALSE(events[0].coverageManifest().empty());
  EXPECT_EQ(events[0].coverageManifest(), events[1].coverageManifest());
  for (int64_t operation = 0; operation < 3; ++operation) {
    EXPECT_NE(
        events[0].regionMembership().find(("#shuttle.source_ref<0, 0, " +
                                           std::to_string(operation) + ", 0>")),
        std::string::npos);
  }
  EXPECT_EQ(
      events[0].regionMembership().find("#shuttle.source_ref<0, 0, 3, 0>"),
      std::string::npos);
  EXPECT_NE(
      events[0].coverageManifest().find("#shuttle.source_ref<0, 0, 3, 0>"),
      std::string::npos);
  EXPECT_NE(events[0].coverageManifest().find("unsupported_operation"),
            std::string::npos);
  EXPECT_FALSE(events[0].unsupportedFingerprint().empty());
  EXPECT_EQ(events[0].unsupportedFingerprint(),
            events[1].unsupportedFingerprint());
  EXPECT_FALSE(events[0].noShuttleSemantics());
  EXPECT_FALSE(events[1].noShuttleSemantics());
  EXPECT_TRUE(events[2].regionMembership().empty());
  EXPECT_TRUE(events[2].coverageManifest().empty());
  EXPECT_TRUE(events[2].unsupportedFingerprint().empty());
  EXPECT_TRUE(events[2].noShuttleSemantics());
  EXPECT_FALSE(events[2].normalizedModuleFingerprint().empty());
}

void expectUnsupportedAbsConsumes(mlir::ModuleOp module,
                                  mlir::Operation *producer) {
  mlir::Operation *absolute = nullptr;
  mlir::func::ReturnOp functionReturn;
  module.walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "stablehlo.abs")
      absolute = operation;
    if (auto returnOperation = mlir::dyn_cast<mlir::func::ReturnOp>(operation))
      functionReturn = returnOperation;
  });
  ASSERT_NE(absolute, nullptr);
  ASSERT_TRUE(functionReturn);
  ASSERT_EQ(absolute->getNumOperands(), 1);
  ASSERT_EQ(functionReturn.getNumOperands(), 1);
  EXPECT_EQ(absolute->getOperand(0).getDefiningOp(), producer);
  EXPECT_EQ(functionReturn.getOperand(0).getDefiningOp(), absolute);
}

xla::ffi::CallFrame ffiInstantiationFrame(mlir::DictionaryAttr config) {
  auto integer = [&](llvm::StringRef name) {
    auto attribute = config.getAs<mlir::IntegerAttr>(name);
    EXPECT_TRUE(attribute) << name.str();
    return attribute ? attribute.getInt() : int64_t{-1};
  };
  auto string = [&](llvm::StringRef name) {
    auto attribute = config.getAs<mlir::StringAttr>(name);
    EXPECT_TRUE(attribute) << name.str();
    return attribute ? attribute.getValue().str() : std::string{};
  };
  xla::ffi::CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  xla::ffi::CallFrameBuilder::AttributesBuilder attributes;
  attributes.Insert("transport_schema_version",
                    integer("transport_schema_version"));
  attributes.Insert("bundle_bytes", string("bundle_bytes"));
  attributes.Insert("bundle_size", integer("bundle_size"));
  attributes.Insert("bundle_sha256", string("bundle_sha256"));
  attributes.Insert("device_schema_version", integer("device_schema_version"));
  attributes.Insert("invocation_abi_schema_version",
                    integer("invocation_abi_schema_version"));
  attributes.Insert("bundle_schema_version", integer("bundle_schema_version"));
  attributes.Insert("completion", string("completion"));
  builder.AddAttributes(attributes.Build());
  return builder.Build();
}

void expectFfiInstantiationAccepts(mlir::DictionaryAttr backendConfig) {
  xla::ffi::CallFrame frame = ffiInstantiationFrame(backendConfig);
  XLA_FFI_Handler_Bundle handlers =
      mlir::shuttle::gpuExecutableBundleFfiHandlerBundle();
  ASSERT_NE(handlers.instantiate, nullptr);
  xla::ffi::ExecutionState instantiateState;
  xla::ffi::ExecutionState prepareState;
  xla::ffi::ExecutionState initializeState;
  xla::ffi::InvokeContext context;
  context.state_context = {&instantiateState, &prepareState, &initializeState};
  EXPECT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               frame, context,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
}

TEST(TrackOneMapWiringTest,
     RoundTripsJaxMixedGraphsAndLeavesUnsupportedStablehloForXla) {
  constexpr std::array<llvm::StringLiteral, 2> programs = {
      kRankTwoMixedMapProgram, kRankThreeMixedMapProgram};
  constexpr std::array<llvm::StringLiteral, 2> policies = {
      kRoundTripSourceOrderedOptions, kRoundTripFastOptions};
  for (llvm::StringRef program : programs) {
    for (llvm::StringRef policy : policies) {
      SCOPED_TRACE(policy.str());
      auto observer = std::make_shared<RecordingObserver>();
      auto subscription =
          mlir::shuttle::subscribeShuttlePipelineObserver(observer);
      mlir::MLIRContext context;
      mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, program);
      ASSERT_TRUE(module);
      absl::Status status = runRegisteredShuttleTransform(*module, policy);
      ASSERT_TRUE(status.ok()) << status;

      std::string transformed = moduleText(*module);
      EXPECT_EQ(countOccurrences(transformed, "stablehlo.convert"), 2);
      EXPECT_EQ(countOccurrences(transformed, "stablehlo.negate"), 1);
      EXPECT_EQ(countOccurrences(transformed, "stablehlo.abs"), 1);
      EXPECT_EQ(transformed.find("stablehlo.custom_call"), std::string::npos);
      EXPECT_EQ(transformed.find("shuttle."), std::string::npos);
      mlir::Operation *negate = nullptr;
      module->walk([&](mlir::Operation *operation) {
        if (operation->getName().getStringRef() == "stablehlo.negate")
          negate = operation;
      });
      ASSERT_NE(negate, nullptr);
      mlir::Operation *roundTripProducer = negate;
      for (mlir::Operation *user : negate->getUsers())
        if (user->getName().getStringRef() == "stablehlo.convert")
          roundTripProducer = user;
      expectUnsupportedAbsConsumes(*module, roundTripProducer);
      expectObservedSuccess(
          observer->snapshot(),
          mlir::shuttle::ShuttlePipelinePhase::LoweredCoverage);
    }
  }
}

TEST(TrackOneMapWiringTest,
     BundlesJaxMapIslandsAndLeavesUnsupportedStablehloForXla) {
  constexpr std::array<llvm::StringLiteral, 2> programs = {
      kRankTwoMixedMapProgram, kRankThreeMixedMapProgram};
  for (llvm::StringRef program : programs) {
    for (llvm::StringRef policy : kGpuPolicies) {
      SCOPED_TRACE(policy.str());
      auto observer = std::make_shared<RecordingObserver>();
      auto subscription =
          mlir::shuttle::subscribeShuttlePipelineObserver(observer);
      mlir::MLIRContext context;
      mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, program);
      ASSERT_TRUE(module);
      absl::Status status = runRegisteredShuttleTransform(*module, policy);
      ASSERT_TRUE(status.ok()) << status;

      std::string transformed = moduleText(*module);
      EXPECT_EQ(
          countOccurrences(transformed, "shuttle.gpu.executable_bundle.v2"), 1);
      EXPECT_EQ(countOccurrences(transformed, "stablehlo.abs"), 1);
      EXPECT_EQ(transformed.find("stablehlo.convert"), std::string::npos);
      EXPECT_EQ(transformed.find("stablehlo.negate"), std::string::npos);
      EXPECT_EQ(transformed.find("shuttle.region"), std::string::npos);
      EXPECT_NE(transformed.find("api_version = 4 : i32"), std::string::npos);

      mlir::Operation *customCall = nullptr;
      module->walk([&](mlir::Operation *operation) {
        if (operation->getName().getStringRef() == "stablehlo.custom_call") {
          customCall = operation;
        }
      });
      ASSERT_NE(customCall, nullptr);
      auto backendConfig =
          customCall->getAttrOfType<mlir::DictionaryAttr>("backend_config");
      ASSERT_TRUE(backendConfig);
      expectUnsupportedAbsConsumes(*module, customCall);
      expectFfiInstantiationAccepts(backendConfig);
      auto bundleBytes = backendConfig.getAs<mlir::StringAttr>("bundle_bytes");
      ASSERT_TRUE(bundleBytes);
      llvm::StringRef serialized = bundleBytes.getValue();
      auto loaded = mlir::shuttle::GpuExecutable::Load(llvm::ArrayRef<uint8_t>(
          reinterpret_cast<const uint8_t *>(serialized.data()),
          serialized.size()));
      ASSERT_TRUE(loaded.ok()) << loaded.status();
      EXPECT_EQ((*loaded)->launches().size(), 3);
      EXPECT_EQ((*loaded)->slots().size(), 4);
      EXPECT_EQ((*loaded)->externalBindings().size(), 2);
      EXPECT_EQ(std::count_if(
                    (*loaded)->slots().begin(), (*loaded)->slots().end(),
                    [](const auto &slot) {
                      return slot.storage ==
                             mlir::shuttle::MaterializationStorage::Temporary;
                    }),
                2);
      expectObservedSuccess(
          observer->snapshot(),
          mlir::shuttle::ShuttlePipelinePhase::GeneratedBundle);
    }
  }
}

TEST(TrackOneMapWiringTest,
     BundlesBothValueAndGradientMapIslandsThroughRegisteredFfi) {
  constexpr std::array<llvm::StringLiteral, 2> programs = {
      kRankTwoValueAndGradientProgram, kRankThreeValueAndGradientProgram};
  for (llvm::StringRef program : programs) {
    for (llvm::StringRef policy : kGpuPolicies) {
      SCOPED_TRACE(policy.str());
      auto observer = std::make_shared<RecordingObserver>();
      auto subscription =
          mlir::shuttle::subscribeShuttlePipelineObserver(observer);
      mlir::MLIRContext context;
      mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, program);
      ASSERT_TRUE(module);
      absl::Status status = runRegisteredShuttleTransform(*module, policy);
      ASSERT_TRUE(status.ok()) << status;

      llvm::SmallVector<mlir::Operation *> customCalls;
      mlir::Operation *absolute = nullptr;
      mlir::func::ReturnOp functionReturn;
      module->walk([&](mlir::Operation *operation) {
        if (operation->getName().getStringRef() == "stablehlo.custom_call")
          customCalls.push_back(operation);
        if (operation->getName().getStringRef() == "stablehlo.abs")
          absolute = operation;
        if (auto returnOperation =
                mlir::dyn_cast<mlir::func::ReturnOp>(operation))
          functionReturn = returnOperation;
      });
      ASSERT_EQ(customCalls.size(), 2);
      ASSERT_NE(absolute, nullptr);
      ASSERT_TRUE(functionReturn);
      ASSERT_EQ(functionReturn.getNumOperands(), 2);
      EXPECT_EQ(absolute->getOperand(0).getDefiningOp(), customCalls[0]);
      EXPECT_EQ(functionReturn.getOperand(0).getDefiningOp(), absolute);
      EXPECT_EQ(functionReturn.getOperand(1).getDefiningOp(), customCalls[1]);

      size_t launches = 0;
      for (mlir::Operation *customCall : customCalls) {
        auto backendConfig =
            customCall->getAttrOfType<mlir::DictionaryAttr>("backend_config");
        ASSERT_TRUE(backendConfig);
        expectFfiInstantiationAccepts(backendConfig);
        auto bundleBytes =
            backendConfig.getAs<mlir::StringAttr>("bundle_bytes");
        ASSERT_TRUE(bundleBytes);
        llvm::StringRef serialized = bundleBytes.getValue();
        auto loaded =
            mlir::shuttle::GpuExecutable::Load(llvm::ArrayRef<uint8_t>(
                reinterpret_cast<const uint8_t *>(serialized.data()),
                serialized.size()));
        ASSERT_TRUE(loaded.ok()) << loaded.status();
        launches += (*loaded)->launches().size();
      }
      EXPECT_EQ(launches, 7);
      auto events = observer->snapshot();
      expectObservedSuccess(
          events, mlir::shuttle::ShuttlePipelinePhase::GeneratedBundle);
      for (int64_t operation : {4, 5, 6, 7}) {
        EXPECT_NE(events[0].regionMembership().find(
                      ("#shuttle.source_ref<0, 0, " +
                       std::to_string(operation) + ", 0>")),
                  std::string::npos);
      }
    }
  }
}

TEST(TrackOneMapWiringTest,
     BundlesAbi22RmsNormFoldRegionsThroughRegisteredFfi) {
  struct Boundary {
    const char *filename;
    int64_t inputs;
    int64_t outputs;
    int64_t launches;
    int64_t temporaries;
  };
  constexpr Boundary boundaries[] = {
      {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir",
       2, 1, 1, 0},
      {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir",
       3, 2, 3, 1},
      {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-composed.mlir",
       3, 3, 4, 1},
      {"jax-0.10.1-bf16-projected_parameter_broadcast-"
       "3x2x5x11x263-forward.mlir",
       2, 1, 1, 0},
      {"jax-0.10.1-bf16-projected_parameter_broadcast-"
       "3x2x5x11x263-backward.mlir",
       3, 2, 3, 1},
      {"jax-0.10.1-bf16-projected_parameter_broadcast-"
       "3x2x5x11x263-composed.mlir",
       3, 3, 4, 1},
  };
  for (const Boundary &boundary : boundaries) {
    SCOPED_TRACE(boundary.filename);
    std::optional<std::string> source = fixtureSource(boundary.filename);
    ASSERT_TRUE(source);
    for (llvm::StringRef policy : kAbi22FoldRegionPolicies) {
      SCOPED_TRACE(policy.str());
      auto observer = std::make_shared<RecordingObserver>();
      auto subscription =
          mlir::shuttle::subscribeShuttlePipelineObserver(observer);
      mlir::MLIRContext context;
      mlir::OwningOpRef<mlir::ModuleOp> module =
          parseModule(context, *source);
      ASSERT_TRUE(module);
      absl::Status status = runRegisteredShuttleTransform(*module, policy);
      ASSERT_TRUE(status.ok()) << status;

      llvm::SmallVector<mlir::Operation *> customCalls;
      module->walk([&](mlir::Operation *operation) {
        if (operation->getName().getStringRef() == "stablehlo.custom_call")
          customCalls.push_back(operation);
      });
      ASSERT_EQ(customCalls.size(), 1);
      mlir::Operation *customCall = customCalls.front();
      EXPECT_EQ(customCall->getNumOperands(), boundary.inputs);
      EXPECT_EQ(customCall->getNumResults(), boundary.outputs);
      auto backendConfig =
          customCall->getAttrOfType<mlir::DictionaryAttr>("backend_config");
      ASSERT_TRUE(backendConfig);
      expectFfiInstantiationAccepts(backendConfig);
      auto bundleBytes =
          backendConfig.getAs<mlir::StringAttr>("bundle_bytes");
      ASSERT_TRUE(bundleBytes);
      llvm::StringRef serialized = bundleBytes.getValue();
      auto loaded = mlir::shuttle::GpuExecutable::Load(llvm::ArrayRef<uint8_t>(
          reinterpret_cast<const uint8_t *>(serialized.data()),
          serialized.size()));
      ASSERT_TRUE(loaded.ok()) << loaded.status();
      EXPECT_EQ((*loaded)->launches().size(), boundary.launches);
      EXPECT_EQ((*loaded)->externalBindings().size(),
                boundary.inputs + boundary.outputs);
      EXPECT_EQ(std::count_if(
                    (*loaded)->slots().begin(), (*loaded)->slots().end(),
                    [](const auto &slot) {
                      return slot.storage ==
                             mlir::shuttle::MaterializationStorage::Temporary;
                    }),
                boundary.temporaries);

      std::string transformed = moduleText(*module);
      EXPECT_EQ(transformed.find("shuttle.region"), std::string::npos);
      EXPECT_EQ(transformed.find("stablehlo.reduce"), std::string::npos);
      auto events = observer->snapshot();
      ASSERT_EQ(events.size(), 3);
      EXPECT_EQ(events[0].phase(),
                mlir::shuttle::ShuttlePipelinePhase::AlgebraCoverage);
      EXPECT_EQ(events[1].phase(),
                mlir::shuttle::ShuttlePipelinePhase::GeneratedBundle);
      EXPECT_EQ(events[2].phase(),
                mlir::shuttle::ShuttlePipelinePhase::FinalErasure);
      EXPECT_GT(events[0].invocationId(), 0);
      EXPECT_EQ(events[0].invocationId(), events[1].invocationId());
      EXPECT_EQ(events[0].invocationId(), events[2].invocationId());
      EXPECT_FALSE(events[0].regionMembership().empty());
      EXPECT_EQ(events[0].regionMembership(), events[1].regionMembership());
      EXPECT_FALSE(events[0].noShuttleSemantics());
      EXPECT_FALSE(events[1].noShuttleSemantics());
      EXPECT_TRUE(events[2].noShuttleSemantics());
      EXPECT_TRUE(events[2].regionMembership().empty());
    }
  }
}

TEST(TrackOneMapWiringTest, LateMapCodegenFailureLeavesTheJaxGraphUnchanged) {
  auto observer = std::make_shared<RecordingObserver>();
  auto subscription = mlir::shuttle::subscribeShuttlePipelineObserver(observer);
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, kUnsupportedMapBodyProgram);
  ASSERT_TRUE(module);
  std::string before = moduleText(*module);

  absl::Status status = mlir::shuttle::runShuttleXlaTransform(
      *module, kAbi21GpuSourceOrderedOptions.str());

  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(moduleText(*module), before);
  auto events = observer->snapshot();
  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[0].phase(),
            mlir::shuttle::ShuttlePipelinePhase::AlgebraCoverage);
  EXPECT_EQ(events[1].phase(), mlir::shuttle::ShuttlePipelinePhase::Failure);
  EXPECT_EQ(events[1].failurePass(), "shuttle-build-gpu-executable-bundle");
}

TEST(TrackOneMapWiringTest,
     RejectsF32BoundariesAndBf16TemporariesTransactionally) {
  constexpr std::array<llvm::StringLiteral, 2> invalidPrograms = {
      kF32BoundaryProgram, kInternalBf16Program};
  for (llvm::StringRef program : invalidPrograms) {
    auto observer = std::make_shared<RecordingObserver>();
    auto subscription =
        mlir::shuttle::subscribeShuttlePipelineObserver(observer);
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, program);
    ASSERT_TRUE(module);
    std::string before = moduleText(*module);

    absl::Status status = mlir::shuttle::runShuttleXlaTransform(
        *module, kAbi21GpuSourceOrderedOptions.str());

    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_EQ(moduleText(*module), before);
    auto events = observer->snapshot();
    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(events[0].phase(),
              mlir::shuttle::ShuttlePipelinePhase::AlgebraCoverage);
    EXPECT_EQ(events[1].phase(), mlir::shuttle::ShuttlePipelinePhase::Failure);
    EXPECT_EQ(events[1].failurePass(), "shuttle-plan-row-fold-materialization");
  }
}

} // namespace
