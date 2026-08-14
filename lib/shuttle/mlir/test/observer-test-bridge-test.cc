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
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":11,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
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
            "35c8ea524578d03d8a8b87241dec0540a8992f7e04862f94eb47a79ef7ad300e");
  EXPECT_EQ(events[2].normalizedModuleFingerprint,
            "a5b42bfca0adc315873ffa9a965c7003d92c021234ec68222e44a296750a15dc");
}

TEST(ShuttleObserverTestBridgeTest,
     LayerNormGeneratedAndFinalFingerprintsAreClosedPerCell) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  struct Fingerprints {
    const char *fixture;
    const char *generated[2];
    const char *final[2];
  };
  constexpr std::array<Fingerprints, 6> fixtures = {{
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir",
       {"51a675acd6bf1254176267c8a0d01ccd04357710f564233d35a70a14a780fbb5",
        "b03a5d84794682f2eca360820fbc937656882f547cc9725395e75b7c8b944d76"},
       {"4eceb5ea70cd2e919c9823af01be735a5937b3d25b539be050dc81083691a2cd",
        "055c5a4bd1e22e71037003cbe85ac2baa2a3589c53be7237acc6c10545d58858"}},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-backward.mlir",
       {"3b2f78ad26b3cad2758d78d5adefab0c16756838979469af41733870245bb55d",
        "91c44297721ebcb838332a9826e3fa93c07e55c5506fe16aa30da5e622012597"},
       {"fa32168a74fc038ba7a11a768f168eea57d19ae70ac0ac23c6c22a08d02ba9b7",
        "18e412711edc31de7fe46375b94ea95692495ae43202e9f257c264e63b04319d"}},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir",
       {"8c60d2debd1344bffcd251517f9c05c9b3a886f2780c97377d93ae63be86c865",
        "a6fea4448ad3a907dcec72e94013132d370ee5c9fc58966912856eac5a8db954"},
       {"537b96019fa2cdc021ff70252f588c8e1eddaabaaa754076e98a4fed6e708a94",
        "74a606047a0a88279adae7efa80d4eaca9c614957fc806e75eb63601277e8961"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir",
       {"9709ec5748d1782201c66b757b277e581c4b6ba7d22e4f7c0dcca6d8671e5a73",
        "38f9c6b58ea83d426a297e0e2e499e547f1ffc6e2af3eac4acdb30fa2f5d6cd3"},
       {"1c3f73a65db2df46631437f30f781de0b7e94e0c04d0cea91285178b681cf0bf",
        "64a17a580ab728cfc3ff5eb381508112f9c322a69681c545d288bc7bdb0437fe"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-backward.mlir",
       {"ddc1f20c8db96585d2ae722a13fcdbdb764b3d965a52a43ebe7c62d6de9d5b76",
        "139587df1e0fbe5e9c000533c1b1395a4a447c895b82c306831c76639a1dcb3f"},
       {"6aa2dd0cd4a8bfdfb9e530ff2e7d0ed8c05b1a8fdb924bc91538272f8b062fb1",
        "ae00099b7fea645587f925f49e98ba2f344f60e137c20a7941e4fd012e3df000"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-composed.mlir",
       {"6b59d6a04b3e8366571b08e5ed6278ebc8594eba66d5a7dc52749ae6fdd023d4",
        "539ccc57b4cc30004f6ba3b756c3c09aaf29bfea4d2ab2a9f1d1bfb8a91af551"},
       {"966722fb8e7ed4f2fcb8c3747b7e3eb678a71d7fe8a7f3d04658ab89872d8019",
        "e9657c7c406ccb0037bc990287fe5f261f17c994e1e02d78c03b8086cc9b0038"}},
  }};
  constexpr std::array<const char *, 2> policies = {
      "source_ordered", "fast",
  };
  for (const Fingerprints &fingerprints : fixtures) {
    for (std::size_t policyIndex = 0; policyIndex < policies.size();
         ++policyIndex) {
      const char *policy = policies[policyIndex];
      std::ifstream input(runfiles->Rlocation(
          std::string("shuttle_mlir/test/Inputs/") + fingerprints.fixture));
      std::ostringstream source;
      source << input.rdbuf();
      ASSERT_TRUE(input);
      auto capture =
          mlir::shuttle::testing::subscribeShuttleObserverForTesting();
      mlir::DialectRegistry registry;
      mlir::stablehlo::registerAllDialects(registry);
      registry.insert<mlir::func::FuncDialect>();
      mlir::MLIRContext context(registry);
      auto module =
          mlir::parseSourceString<mlir::ModuleOp>(source.str(), &context);
      ASSERT_TRUE(module);
      mlir::shuttle::ShuttlePipelineOptions options;
      options.executionMode =
          mlir::shuttle::ExecutionMode::GpuExecutableBundle;
      options.numerics = policy == std::string("fast")
                             ? mlir::shuttle::NumericalPolicy::Fast
                             : mlir::shuttle::NumericalPolicy::SourceOrdered;
      options.canonicalOptions =
          std::string("{\"execution_mode\":\"gpu_executable_bundle\","
                      "\"numerics\":\"") +
          policy +
          "\",\"pipeline_abi_version\":11,\"schema_version\":1,"
          "\"tuning\":{\"cluster_shape\":[],\"materialization\":"
          "\"automatic\",\"maximum_candidates\":1,\"pipeline_stages\":1,"
          "\"tile_sizes\":[]}}";
      ASSERT_TRUE(mlir::succeeded(
          mlir::shuttle::runShuttleXlaTransform(*module, options)));
      auto events = capture->snapshot();
      ASSERT_EQ(events.size(), 3);
      EXPECT_EQ(events[1].normalizedModuleFingerprint,
                fingerprints.generated[policyIndex]);
      EXPECT_EQ(events[2].normalizedModuleFingerprint,
                fingerprints.final[policyIndex]);
    }
  }
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
