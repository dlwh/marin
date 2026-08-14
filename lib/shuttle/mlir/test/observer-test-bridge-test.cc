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
            "4dc859ba10a6b52fe8f20fcbaa75df13b4e4c8c40e5f730a34e6432e1f2da00f");
  EXPECT_EQ(events[2].normalizedModuleFingerprint,
            "b97870ed629fde9df6ba1494314fa2062293d85de56e53f98a4417e3a24e958d");
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
       {"14dd27fbea0fb542923447a84c401862972d51b882d437db351f9017b2e9c9a1",
        "7ea3cd47aa4ebee558d87b9c38806f1d4aa4d45f8d5d71a300cb11cec0cf94cf"},
       {"3dea82023efde83bd2f0cc42940eef3ee78c1dd7cbed26c69228ff8e1d08cf06",
        "e5a44e4947fd16d004d5d156f539fa4a2b5e751d6b957dd0b984062141626887"}},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-backward.mlir",
       {"70212b005312cecc45de474b2572306e895e7429b3783c1c569b00c2ac59c688",
        "f81bebdc98bc8db0177bf961809602cb69378ca4689599c07be6a25d30368d6c"},
       {"2538cfec50123bcff3c9d13efaa1f262dc972e4fa9d3ef91bc3f86d66f61e71e",
        "092dfd91fd44248286a518598115f3104881b05755b65d873dfbe296b3c6ca7a"}},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir",
       {"264e568277f36cb22c4ef362b8f37c934f8e05844a53e321f9fce13b12f1716c",
        "bac6f333ae444eab24b9fbfb286a4d1eac2be780f1c0469adcb8a7a0426a73b1"},
       {"57b365ea9f022632cdc127016f65fb884c86e09b3d33c2d2ce988865003060fb",
        "feb4450affed2cefe1a60a631fa368fc2e8417534f5e232e2b844639aee90511"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir",
       {"7c9528eb9f0b70c94398dc88d0ca9299a3fb4f73e4656b0c69f6d4eda29027f4",
        "f637184d311568d777f4a06844661560cf35a2f9a7361fb8cf3b8174d5a9e2d9"},
       {"e04ef75d200893ec138459d7d197cba47941b4cf2e4718b45cd679772244d4de",
        "ed18f687d1d62283859071c493270c1911e916eb22dd6a836a6b52a4777ea8a4"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-backward.mlir",
       {"a3e917830b43cbe6a6b88ee2b1037c5a1e90329bb6f3e3f3ad1e07fb6f4881b4",
        "59eec815eac7bc70e34d43c989d2de0ef3b0630fca6dd8c8807918ec6cd878fb"},
       {"4e65e61b30890170096cf8f38f55eb58bb77b15fe0fe1ddfebe1a454b2d5df3b",
        "df75fedb775547f2e32e915fbe88053b6305065adbf48491e7cb7eddf3c159a4"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-composed.mlir",
       {"73e7fd2631e83f8fcda2600ace109783175fb312c37a4abfc55dc8741eed127b",
        "c6c63421ab8b1be82f0ad7f1c935dbae043b59af428d5c9261c119074feae04e"},
       {"63f722e6e20356c452e03513137d420cce87126902bfbc88d439236c0e3fadf1",
        "669d6168dea3b47713af143255c4940421a5d989ec30b423dcbbf4a562f25475"}},
  }};
  constexpr std::array<const char *, 2> policies = {
      "source_ordered",
      "fast",
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
      options.executionMode = mlir::shuttle::ExecutionMode::GpuExecutableBundle;
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
