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
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  ASSERT_TRUE(
      mlir::succeeded(mlir::shuttle::runShuttleXlaTransform(*module, options)));
  auto events = capture->snapshot();
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0].phase, "algebra_coverage");
  EXPECT_EQ(events[1].phase, "generated_bundle");
  EXPECT_EQ(events[2].phase, "final_erasure");
  EXPECT_EQ(events[1].policyDigest,
            "731a06b7bd067960293e47448e483e07ca7373a6ff09bfa94fb3dc804d7de280");
  EXPECT_EQ(events[0].coverageManifest, events[1].coverageManifest);
  EXPECT_FALSE(events[1].normalizedModuleFingerprint.empty());
  EXPECT_FALSE(events[1].noShuttleSemantics);
  EXPECT_TRUE(events[2].noShuttleSemantics);
  EXPECT_EQ(events[1].normalizedModuleFingerprint,
            "0b46a104e0e1c9307a2dcdc1aa9e9a0b49129970f0864d8ed8b1b34d55cd2c38");
  EXPECT_EQ(events[2].normalizedModuleFingerprint,
            "6298245e05a8f2b2efd123554f9593d2722f44261f5fec241f8763b4131ddfba");
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
       {"1b12fa78e65428ec211aab65c599e24ed6f04cfd70acddfa3011f8d735e3fc29",
        "51620b20c889c69a6146a4bce1050ab214e89a9c1fb5a23fe552f61ed5f0218a"},
       {"2ad7c508f530841f953fa1e81aed473fbb8fe7532780f1748a51a9fce729279a",
        "bb776bd06f293636976ab3ad218064a38a959060f43a32307153199d53210397"}},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-backward.mlir",
       {"6896d9b7999b00520c446df6d7c8b956cdfc93781087430453c0159d25966e41",
        "0497915627df04286a33ac59ecdf63a36a1e2cdffb9e12ded0853726916ed65b"},
       {"7297e1176a5879d5f695814d45735e78623f68b6332793b92dac62eb07de36e2",
        "3e339542ad763530c072a1327918a5158674ccdd7f1510cb9bad3d5c54d01c98"}},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir",
       {"722d980e05155e01076fa8e1c7f0cfab486ccd2fb46e394444d2866455acbf01",
        "e25cd4eea88e5f2a6734fd4aa9185ee8d6856770384b2be636ba4b072a63fb75"},
       {"333dbbcbc020d8360ad9d496e57580a948129c10307d590494bc87138680c2fc",
        "0c5bf1845b0d4fd7c33e3328409453f758571a7e92a343c41ee4d10f79379aa4"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir",
       {"3458b44c014c7ac9e0509f58d80685a3292f968bd74194e569224d4295e7ecb2",
        "7407ce943168a419408522c6f1dfc11b88e608ebe6349d7695f46a27e522ce68"},
       {"ac2f23ad49f825f1f557315192c50a471507afb3297ab38042fbadf0415e4078",
        "dca22b2b44c183b6ab47458de651db719498bebcaf11635a234386adf8749fda"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-backward.mlir",
       {"d59335ef534bcd66ba996fa93696f82d1a17951616d864bd5b06d05a788711dd",
        "6690438f1f4fd377526267e2fe03dffd3bf35db50547db4eb46caf1b51c23283"},
       {"9ba321e8759a599eda38c88d9d3e336398f69d160003053076164e89053a8e7d",
        "4747249de56e9b6c5f8fd53bf12a462bcb74b2fd6c61e039108c4084c53a34f8"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-composed.mlir",
       {"14552e42efe9dbfdaba1ed89a0768596676df658860f0faa59ff2d4f96d13643",
        "9fe0c290bb82cad835a7a1f4c4654f5087418939cadc0d5e7cff2a7c6b5e00b5"},
       {"e7e70523f54d32fa3c287b3da189fa2cb10d037935005f8d6290a3b23256ec33",
        "34c38893e3986d096aea78baf3ca3be24231099ecb0073a1a4f037bc0c80a763"}},
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
          "\",\"pipeline_abi_version\":16,\"schema_version\":1,"
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
     MultiRegionGeneratedFingerprintConsumesEveryBundleTuple) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;

  struct Fingerprints {
    const char *policy;
    const char *generated;
    const char *final;
  };
  struct FixtureFingerprints {
    const char *fixture;
    std::array<Fingerprints, 2> policies;
  };
  constexpr std::array<FixtureFingerprints, 2> fixtures = {{
      {"jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir",
       {{{"source_ordered",
          "458d941bc3a5cd636b4a836bc2c4b515246add12d17180205c0524343dfd7ae0",
          "22d4520b8a9337ccb192272126323e7d1b8a3629007ba02de5342a7026b198ca"},
         {"fast",
          "6b5d9bc0141c4555a8397aae21c6bb097d3637d2fab957713a82ebf6dc863eec",
          "b5746a610b98f683f28378090ff48d163b27e0d92bee20af8f5f144e95b1acf"
          "4"}}}},
      {"jax-0.10.1-bf16-matmul-vjp-33x2x34-composed.mlir",
       {{{"source_ordered",
          "d5f91111b5e05c7cdf4af807702287b1dc55233e697269c04769af438b4f8540",
          "036085a583909874d0c4e6c134926880af03d5b54ceffb3084d838362fe13fcc"},
         {"fast",
          "96e8a120c055843446523819aa1e085f1391c7c790cb6f607c5c4339bd6cb571",
          "9b8de0ccb413b410c0176acd9bd6b7ac1bbca1759c1fab8e3bee60fd18e4bea"
          "a"}}}},
  }};
  for (const FixtureFingerprints &fixture : fixtures) {
    std::ifstream input(runfiles->Rlocation(
        std::string("shuttle_mlir/test/Inputs/") + fixture.fixture));
    std::ostringstream source;
    source << input.rdbuf();
    ASSERT_TRUE(input);
    for (const Fingerprints &expected : fixture.policies) {
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
      options.numerics = expected.policy == std::string("fast")
                             ? mlir::shuttle::NumericalPolicy::Fast
                             : mlir::shuttle::NumericalPolicy::SourceOrdered;
      options.canonicalOptions =
          std::string("{\"execution_mode\":\"gpu_executable_bundle\","
                      "\"numerics\":\"") +
          expected.policy +
          "\",\"pipeline_abi_version\":16,\"schema_version\":1,"
          "\"tuning\":{\"cluster_shape\":[],\"materialization\":"
          "\"automatic\",\"maximum_candidates\":1,"
          "\"pipeline_stages\":1,\"tile_sizes\":[]}}";
      ASSERT_TRUE(mlir::succeeded(
          mlir::shuttle::runShuttleXlaTransform(*module, options)));
      auto events = capture->snapshot();
      ASSERT_EQ(events.size(), 3);
      EXPECT_EQ(events[0].phase, "algebra_coverage");
      EXPECT_EQ(events[1].phase, "generated_bundle");
      EXPECT_EQ(events[2].phase, "final_erasure");
      EXPECT_EQ(events[0].invocationId, events[1].invocationId);
      EXPECT_EQ(events[1].invocationId, events[2].invocationId);
      EXPECT_EQ(events[0].policy, expected.policy);
      EXPECT_EQ(events[1].policy, expected.policy);
      EXPECT_EQ(events[2].policy, expected.policy);
      EXPECT_EQ(events[0].coverageManifest, events[1].coverageManifest);
      EXPECT_FALSE(events[1].noShuttleSemantics);
      EXPECT_TRUE(events[2].noShuttleSemantics);
      EXPECT_EQ(events[1].normalizedModuleFingerprint, expected.generated);
      EXPECT_EQ(events[2].normalizedModuleFingerprint, expected.final);
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
