// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Testing/ObserverTestBridge.h"

#include <array>
#include <atomic>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "shuttle/Transforms/Passes.h"
#include "shuttle/Transforms/XlaRegistration.h"
#include "stablehlo/dialect/Register.h"
#include "tools/cpp/runfiles/runfiles.h"
#include "gtest/gtest.h"

namespace {

constexpr char kProgram[] = R"mlir(
module {
  func.func @main(%arg0: tensor<7xf32>) -> tensor<7xf32> {
    %0 = stablehlo.tanh %arg0 : tensor<7xf32>
    return %0 : tensor<7xf32>
  }
}
)mlir";

TEST(ShuttleObserverTestBridgeTest, CopiesCompleteSuccessfulInvocation) {
  auto capture = mlir::shuttle::testing::subscribeShuttleObserverForTesting();

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kProgram, &context);
  ASSERT_TRUE(module);

  mlir::shuttle::ShuttlePipelineOptions options;
  ASSERT_TRUE(
      mlir::succeeded(mlir::shuttle::runShuttleXlaTransform(*module, options)));
  std::vector<mlir::shuttle::testing::ShuttleObserverTestEvent> events =
      capture->snapshot();

  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0].phase, "algebra_coverage");
  EXPECT_EQ(events[1].phase, "lowered_coverage");
  EXPECT_EQ(events[2].phase, "final_erasure");
  EXPECT_EQ(events[0].invocationId, events[2].invocationId);
  EXPECT_EQ(events[0].policy, "source_ordered");
  EXPECT_FALSE(events[0].policyDigest.empty());
  EXPECT_FALSE(events[0].tuningDigest.empty());
  EXPECT_EQ(events[0].regionMembership, "[[#shuttle.source_ref<0, 0, 0, 0>]]");
  EXPECT_EQ(events[0].regionMembership, events[1].regionMembership);
  EXPECT_FALSE(events[0].coverageManifest.empty());
  EXPECT_EQ(events[0].coverageManifest, events[1].coverageManifest);
  EXPECT_EQ(events[0].unsupportedFingerprint,
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945");
  EXPECT_EQ(events[0].unsupportedFingerprint, events[1].unsupportedFingerprint);
  EXPECT_TRUE(events[0].normalizedModuleFingerprint.empty());
  EXPECT_TRUE(events[1].normalizedModuleFingerprint.empty());
  EXPECT_FALSE(events[0].noShuttleSemantics);
  EXPECT_FALSE(events[1].noShuttleSemantics);
  EXPECT_TRUE(events[2].regionMembership.empty());
  EXPECT_TRUE(events[2].coverageManifest.empty());
  EXPECT_TRUE(events[2].unsupportedFingerprint.empty());
  EXPECT_EQ(events[2].normalizedModuleFingerprint,
            mlir::shuttle::normalizedStablehloFingerprint(*module));
  EXPECT_TRUE(events[2].noShuttleSemantics);
  for (const auto &event : events) {
    EXPECT_TRUE(event.failurePass.empty());
  }

  capture->close();
  capture->close();
  mlir::OwningOpRef<mlir::ModuleOp> afterClose =
      mlir::parseSourceString<mlir::ModuleOp>(kProgram, &context);
  ASSERT_TRUE(afterClose);
  ASSERT_TRUE(mlir::succeeded(
      mlir::shuttle::runShuttleXlaTransform(*afterClose, options)));
  EXPECT_EQ(capture->snapshot().size(), 3);
}

TEST(ShuttleObserverTestBridgeTest,
     GpuPipelineExposesGeneratedBundleBeforeFinalErasure) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  std::ifstream input(runfiles->Rlocation(
      "shuttle_mlir/test/Inputs/"
      "jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir"));
  std::ostringstream source;
  source << input.rdbuf();
  ASSERT_TRUE(input);

  auto capture = mlir::shuttle::testing::subscribeShuttleObserverForTesting();
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source.str(), &context);
  ASSERT_TRUE(module);
  mlir::shuttle::ShuttlePipelineOptions options;
  options.executionMode = mlir::shuttle::ExecutionMode::GpuExecutableBundle;
  options.canonicalOptions =
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":10,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  ASSERT_TRUE(
      mlir::succeeded(mlir::shuttle::runShuttleXlaTransform(*module, options)));
  auto events = capture->snapshot();
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0].phase, "algebra_coverage");
  EXPECT_EQ(events[1].phase, "generated_bundle");
  EXPECT_EQ(events[2].phase, "final_erasure");
  EXPECT_EQ(events[0].coverageManifest, events[1].coverageManifest);
  EXPECT_FALSE(events[1].normalizedModuleFingerprint.empty());
  EXPECT_FALSE(events[1].noShuttleSemantics);
  EXPECT_TRUE(events[2].noShuttleSemantics);
  EXPECT_EQ(events[1].normalizedModuleFingerprint,
            "fbdca3266a5a927e0f27813b53df7f37305e9a1561f5e295d560c2e66a9c0cdb");
  EXPECT_EQ(events[2].normalizedModuleFingerprint,
            "a81d64b4bfe67da8865ab686918a88bdd3b008f1c7d7b9f0d5a871413310086c");
}

TEST(ShuttleObserverTestBridgeTest,
     ConcurrentCloseWaitsForCapturedInvocationAndEachOther) {
  auto capture = mlir::shuttle::testing::subscribeShuttleObserverForTesting();
  capture->blockNextCallbackForTesting();

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kProgram, &context);
  ASSERT_TRUE(module);
  mlir::shuttle::ShuttlePipelineOptions options;
  std::atomic<bool> pipelineSucceeded = false;
  std::thread pipeline([&] {
    pipelineSucceeded = mlir::succeeded(
        mlir::shuttle::runShuttleXlaTransform(*module, options));
  });
  capture->waitForBlockedCallbackForTesting();

  std::array<std::size_t, 2> eventCountsAtCloseReturn{};
  std::thread firstCloser([&] {
    capture->close();
    eventCountsAtCloseReturn[0] = capture->snapshot().size();
  });
  std::thread secondCloser([&] {
    capture->close();
    eventCountsAtCloseReturn[1] = capture->snapshot().size();
  });
  capture->waitForCloseCallersForTesting(2);
  capture->releaseBlockedCallbackForTesting();

  firstCloser.join();
  secondCloser.join();
  pipeline.join();
  ASSERT_TRUE(pipelineSucceeded);
  EXPECT_EQ(eventCountsAtCloseReturn, (std::array<std::size_t, 2>{3, 3}));

  mlir::OwningOpRef<mlir::ModuleOp> afterClose =
      mlir::parseSourceString<mlir::ModuleOp>(kProgram, &context);
  ASSERT_TRUE(afterClose);
  ASSERT_TRUE(mlir::succeeded(
      mlir::shuttle::runShuttleXlaTransform(*afterClose, options)));
  EXPECT_EQ(capture->snapshot().size(), 3);
}

} // namespace
