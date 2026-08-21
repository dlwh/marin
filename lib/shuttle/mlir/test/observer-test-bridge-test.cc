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
       {"3539f1c3cf340e4b2a5108a886d95a2519923091720950bd78c3a969ee4018a8",
        "71c3e53d46bbad773d880595f6822622426d1bef58fc00c1589f99dccb99193c"},
       {"98421ff068aba658c1598ea9e0546ae7c4123238ef71e72ca85a0a6f057b9a89",
        "9a7edd24ad75dfe261fde7df86ae04043378f1ffd2017cd30ed3a1829f89a77b"}},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir",
       {"520188f4b87b925f38272a0c5de8bee62a7a3f4528a81358e7e15d4b15f262e4",
        "6f00db056d8aa6e6fc3809d715d2bc5485cb909c4e9f53522fdaacb85d0988be"},
       {"035113f60ca84d578ee35ee1372fc5bc3281eb4d8e4a5fa16525a975c267dae8",
        "48f35334a65b295718eb5c7be64fc3494b2fe0636de73a994f1ddd47bb927d5d"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir",
       {"3458b44c014c7ac9e0509f58d80685a3292f968bd74194e569224d4295e7ecb2",
        "7407ce943168a419408522c6f1dfc11b88e608ebe6349d7695f46a27e522ce68"},
       {"ac2f23ad49f825f1f557315192c50a471507afb3297ab38042fbadf0415e4078",
        "dca22b2b44c183b6ab47458de651db719498bebcaf11635a234386adf8749fda"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-backward.mlir",
       {"9dce0c0db7b78df9bf760cb3012dbc0bf955662e4bb012fcc5d0cd270ebfd26b",
        "45bcef4520779e758a6e82987ede24ce4e38374a93e7d5795779cef04bcd5a5b"},
       {"3e6dc9243fc6878b27f9a568e8b095058483f7f961790b3f5846bc7227f593b8",
        "ad323539acef4728f862a2094a4561c90a3c970031f4ccb794fb0c0dfc3b80d6"}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-composed.mlir",
       {"42e736da83f1785ed45d15f2f1c3124acda73038bac503a69cf6ac9e66dc3c64",
        "ad2a88e809ae32dd037e2649968a22643d927fbe625cad184eb6defdb6582dfa"},
       {"42abd2b73f313ec10d412b5a375803dc30d578b8104f2d055b1d92bf25d079c5",
        "1ce0d2974af7db97137ba1f4a6e4d76d49e95db4dd79988465b8072349c81d28"}},
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
          "7d026ded5d5bcbc5f1924fd21abf51469f61abb2d7a130cb007cd0f1d2655ddc",
          "2774dbe1741915134f9dd8b18b417ed5f8178ae5c8230b0c698b2af9d338b963"},
         {"fast",
          "265abdea60365311d94def217b299d610514f3a82c0f61b63a1143faf6be7c89",
          "2a0dfc1c37020a565a59726b680ed2cc9809c800244ca10bade93a6e5536bde2"}}}},
      {"jax-0.10.1-bf16-matmul-vjp-33x2x34-composed.mlir",
       {{{"source_ordered",
          "30d14aec6dbd740608c1e3e5df0777ff0d144dfbe9967650d174de4d1c441b11",
          "b9ac764d74486ee6d5a9e11b611810d440f891fb6b050950fe858bc5cc8b1c10"},
         {"fast",
          "d6d49e4f2cef8ab0c92cff335976e282a2fb67d512d149ce632d217eec476f53",
          "4f229a88f48c0a44d518ba03ba0d24a45ee9c45625f64ff6db494cc7c8e167a2"}}}},
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
