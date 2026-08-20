// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "shuttle/Transforms/Observer.h"
#include "shuttle/Transforms/XlaRegistration.h"
#include "stablehlo/dialect/Register.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {

constexpr llvm::StringLiteral kRoundTripSourceOrderedOptions =
    R"json({"execution_mode":"stablehlo_round_trip","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kRoundTripFastOptions =
    R"json({"execution_mode":"stablehlo_round_trip","numerics":"fast","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kGpuSourceOrderedOptions =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kGpuFastOptions =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";

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
  void observe(
      const mlir::shuttle::ShuttlePipelineEvent &event) const final {
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
  EXPECT_EQ(events[0].invocationId(), events[2].invocationId());
  EXPECT_EQ(events[0].identity(), events[2].identity());
  EXPECT_FALSE(events[0].regionMembership().empty());
  EXPECT_EQ(events[0].regionMembership(), events[1].regionMembership());
  EXPECT_FALSE(events[0].coverageManifest().empty());
  EXPECT_EQ(events[0].coverageManifest(), events[1].coverageManifest());
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
      absl::Status status =
          mlir::shuttle::runShuttleXlaTransform(*module, policy.str());
      ASSERT_TRUE(status.ok()) << status;

      std::string transformed = moduleText(*module);
      EXPECT_EQ(countOccurrences(transformed, "stablehlo.convert"), 2);
      EXPECT_EQ(countOccurrences(transformed, "stablehlo.negate"), 1);
      EXPECT_EQ(countOccurrences(transformed, "stablehlo.abs"), 1);
      EXPECT_EQ(transformed.find("stablehlo.custom_call"), std::string::npos);
      EXPECT_EQ(transformed.find("shuttle."), std::string::npos);
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
  constexpr std::array<llvm::StringLiteral, 2> policies = {
      kGpuSourceOrderedOptions, kGpuFastOptions};
  for (llvm::StringRef program : programs) {
    for (llvm::StringRef policy : policies) {
      SCOPED_TRACE(policy.str());
      auto observer = std::make_shared<RecordingObserver>();
      auto subscription =
          mlir::shuttle::subscribeShuttlePipelineObserver(observer);
      mlir::MLIRContext context;
      mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, program);
      ASSERT_TRUE(module);
      absl::Status status =
          mlir::shuttle::runShuttleXlaTransform(*module, policy.str());
      ASSERT_TRUE(status.ok()) << status;

      std::string transformed = moduleText(*module);
      EXPECT_EQ(countOccurrences(transformed,
                                 "shuttle.gpu.executable_bundle.v2"),
                1);
      EXPECT_EQ(countOccurrences(transformed, "stablehlo.abs"), 1);
      EXPECT_EQ(transformed.find("stablehlo.convert"), std::string::npos);
      EXPECT_EQ(transformed.find("stablehlo.negate"), std::string::npos);
      EXPECT_EQ(transformed.find("shuttle.region"), std::string::npos);
      expectObservedSuccess(
          observer->snapshot(),
          mlir::shuttle::ShuttlePipelinePhase::GeneratedBundle);
    }
  }
}

TEST(TrackOneMapWiringTest,
     LateMapCodegenFailureLeavesTheJaxGraphUnchanged) {
  auto observer = std::make_shared<RecordingObserver>();
  auto subscription =
      mlir::shuttle::subscribeShuttlePipelineObserver(observer);
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, kUnsupportedMapBodyProgram);
  ASSERT_TRUE(module);
  std::string before = moduleText(*module);

  absl::Status status = mlir::shuttle::runShuttleXlaTransform(
      *module, kGpuSourceOrderedOptions.str());

  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(moduleText(*module), before);
  auto events = observer->snapshot();
  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[0].phase(),
            mlir::shuttle::ShuttlePipelinePhase::AlgebraCoverage);
  EXPECT_EQ(events[1].phase(), mlir::shuttle::ShuttlePipelinePhase::Failure);
  EXPECT_EQ(events[1].failurePass(),
            "shuttle-build-gpu-executable-bundle");
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
        *module, kGpuSourceOrderedOptions.str());

    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_EQ(moduleText(*module), before);
    auto events = observer->snapshot();
    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(events[0].phase(),
              mlir::shuttle::ShuttlePipelinePhase::AlgebraCoverage);
    EXPECT_EQ(events[1].phase(),
              mlir::shuttle::ShuttlePipelinePhase::Failure);
    EXPECT_EQ(events[1].failurePass(),
              "shuttle-plan-row-fold-materialization");
  }
}

} // namespace
