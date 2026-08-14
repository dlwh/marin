// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <sstream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "shuttle/Transforms/Observer.h"
#include "shuttle/Transforms/XlaRegistration.h"
#include "stablehlo/dialect/Register.h"
#include "tools/cpp/runfiles/runfiles.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {

constexpr char kDefaultOptions[] =
    R"json({"execution_mode":"stablehlo_round_trip","numerics":"source_ordered","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kSourceOrderedOptions[] =
    R"json({"execution_mode":"stablehlo_round_trip","numerics":"source_ordered","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}})json";
constexpr char kFastOptions[] =
    R"json({"execution_mode":"stablehlo_round_trip","numerics":"fast","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kCpuExecutableBundleOptions[] =
    R"json({"execution_mode":"cpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}})json";
constexpr char kCpuExecutableBundleFastOptions[] =
    R"json({"execution_mode":"cpu_executable_bundle","numerics":"fast","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}})json";
constexpr char kGpuExecutableBundleOptions[] =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}})json";
constexpr char kGpuExecutableBundleFastOptions[] =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}})json";
constexpr char kAbi14ContractSourceOrderedOptions[] =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kAbi14ContractFastOptions[] =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kProgram[] = R"mlir(
module {
  func.func @main(%arg0: tensor<7xf32>) -> tensor<7xf32> {
    %0 = stablehlo.tanh %arg0 : tensor<7xf32>
    return %0 : tensor<7xf32>
  }
}
)mlir";
constexpr char kContractMapProgram[] = R"mlir(
module @anonymous {
  func.func public @main(%lhs: tensor<3x5xbf16>, %rhs: tensor<5x2xbf16>) -> tensor<3x2xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x5xbf16>, tensor<5x2xbf16>) -> tensor<3x2xf32>
    %result = stablehlo.convert %contract : (tensor<3x2xf32>) -> tensor<3x2xbf16>
    return %result : tensor<3x2xbf16>
  }
}
)mlir";

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(mlir::MLIRContext &context, llvm::StringRef source = kProgram) {
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::func::FuncDialect>();
  mlir::stablehlo::registerAllDialects(dialects);
  context.appendDialectRegistry(dialects);
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

TEST(ShuttleXlaOptionsTest, DefaultOptionsBindCurrentPipelineAbi) {
  mlir::shuttle::ShuttlePipelineOptions options;
  EXPECT_EQ(options.canonicalOptions, kDefaultOptions);
  auto parsed = mlir::shuttle::parseShuttleXlaOptions(options.canonicalOptions);
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  EXPECT_EQ(mlir::shuttle::shuttlePipelineIdentity(options),
            mlir::shuttle::shuttlePipelineIdentity(*parsed));
}

TEST(ShuttleXlaOptionsTest, ParsesCanonicalPythonWireFormat) {
  absl::StatusOr<mlir::shuttle::ShuttlePipelineOptions> source =
      mlir::shuttle::parseShuttleXlaOptions(kSourceOrderedOptions);
  ASSERT_TRUE(source.ok()) << source.status();
  EXPECT_EQ(source->numerics, mlir::shuttle::NumericalPolicy::SourceOrdered);
  EXPECT_EQ(source->executionMode,
            mlir::shuttle::ExecutionMode::StablehloRoundTrip);
  EXPECT_EQ(source->canonicalOptions, kSourceOrderedOptions);
  EXPECT_EQ(
      source->canonicalTuning,
      R"json({"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]})json");
  mlir::shuttle::ShuttlePipelineIdentity sourceIdentity =
      mlir::shuttle::shuttlePipelineIdentity(*source);
  EXPECT_EQ(sourceIdentity.policyDigest,
            "77e4bd42cd65b53a894b0213585c5ed789989b11f351250c9bcf8715ddad671b");
  EXPECT_EQ(sourceIdentity.tuningDigest,
            "ae69cb474b1ddc91067687e7351ee27afe4e3b0814ae59e310a42bec5911326f");

  absl::StatusOr<mlir::shuttle::ShuttlePipelineOptions> fast =
      mlir::shuttle::parseShuttleXlaOptions(kFastOptions);
  ASSERT_TRUE(fast.ok()) << fast.status();
  EXPECT_EQ(fast->numerics, mlir::shuttle::NumericalPolicy::Fast);
  EXPECT_NE(mlir::shuttle::shuttlePipelineIdentity(*source).policyDigest,
            mlir::shuttle::shuttlePipelineIdentity(*fast).policyDigest);

  absl::StatusOr<mlir::shuttle::ShuttlePipelineOptions> cpuBundle =
      mlir::shuttle::parseShuttleXlaOptions(kCpuExecutableBundleOptions);
  ASSERT_TRUE(cpuBundle.ok()) << cpuBundle.status();
  EXPECT_EQ(cpuBundle->executionMode,
            mlir::shuttle::ExecutionMode::CpuExecutableBundle);
  EXPECT_NE(mlir::shuttle::shuttlePipelineIdentity(*source).policyDigest,
            mlir::shuttle::shuttlePipelineIdentity(*cpuBundle).policyDigest);

  absl::StatusOr<mlir::shuttle::ShuttlePipelineOptions> cpuFast =
      mlir::shuttle::parseShuttleXlaOptions(kCpuExecutableBundleFastOptions);
  ASSERT_TRUE(cpuFast.ok()) << cpuFast.status();
  EXPECT_EQ(cpuFast->executionMode,
            mlir::shuttle::ExecutionMode::CpuExecutableBundle);
  EXPECT_EQ(cpuFast->numerics, mlir::shuttle::NumericalPolicy::Fast);
  EXPECT_NE(mlir::shuttle::shuttlePipelineIdentity(*cpuBundle).policyDigest,
            mlir::shuttle::shuttlePipelineIdentity(*cpuFast).policyDigest);

  auto gpuBundle =
      mlir::shuttle::parseShuttleXlaOptions(kGpuExecutableBundleOptions);
  ASSERT_TRUE(gpuBundle.ok()) << gpuBundle.status();
  EXPECT_EQ(gpuBundle->executionMode,
            mlir::shuttle::ExecutionMode::GpuExecutableBundle);

  auto gpuFast =
      mlir::shuttle::parseShuttleXlaOptions(kGpuExecutableBundleFastOptions);
  ASSERT_TRUE(gpuFast.ok()) << gpuFast.status();
  EXPECT_EQ(gpuFast->executionMode,
            mlir::shuttle::ExecutionMode::GpuExecutableBundle);
  EXPECT_EQ(gpuFast->numerics, mlir::shuttle::NumericalPolicy::Fast);
  EXPECT_NE(mlir::shuttle::shuttlePipelineIdentity(*gpuBundle).policyDigest,
            mlir::shuttle::shuttlePipelineIdentity(*gpuFast).policyDigest);
}

TEST(ShuttleXlaOptionsTest, RejectsAdjacentGpuPipelineAbis) {
  constexpr const char *invalid[] = {
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":13,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":15,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
  };
  for (const char *serialized : invalid)
    EXPECT_EQ(mlir::shuttle::parseShuttleXlaOptions(serialized).status().code(),
              absl::StatusCode::kInvalidArgument);
}

TEST(ShuttleXlaOptionsTest, Abi14RunsCanonicalSourceOrderedContractMapOptions) {
  auto options =
      mlir::shuttle::parseShuttleXlaOptions(kAbi14ContractSourceOrderedOptions);
  ASSERT_TRUE(options.ok()) << options.status();
  EXPECT_EQ(options->canonicalOptions, kAbi14ContractSourceOrderedOptions);
  EXPECT_EQ(
      options->canonicalTuning,
      R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json");
  EXPECT_EQ(mlir::shuttle::shuttlePipelineIdentity(*options).policyDigest,
            "afc685fdac734de383e70a49de385ca744287f151816fbff419668b719dd5689");

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, kContractMapProgram);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(
      mlir::shuttle::runShuttleXlaTransform(*module, *options)));
  std::string transformed;
  llvm::raw_string_ostream output(transformed);
  module->print(output);
  output.flush();
  EXPECT_EQ(transformed.find("shuttle.region"), std::string::npos);
  EXPECT_EQ(transformed.find("stablehlo.dot_general"), std::string::npos);
  EXPECT_NE(transformed.find("shuttle.gpu.executable_bundle.v2"),
            std::string::npos);
}

TEST(ShuttleXlaOptionsTest, Abi14PreservesFastMapFold) {
  auto options =
      mlir::shuttle::parseShuttleXlaOptions(kAbi14ContractFastOptions);
  ASSERT_TRUE(options.ok()) << options.status();

  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  std::ifstream input(runfiles->Rlocation(
      "shuttle_mlir/test/Inputs/"
      "jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir"));
  std::ostringstream source;
  source << input.rdbuf();
  ASSERT_TRUE(input);

  mlir::MLIRContext mapFoldContext;
  mlir::OwningOpRef<mlir::ModuleOp> mapFold =
      parseModule(mapFoldContext, source.str());
  ASSERT_TRUE(mapFold);
  EXPECT_TRUE(mlir::succeeded(
      mlir::shuttle::runShuttleXlaTransform(*mapFold, *options)));
}

TEST(ShuttleXlaOptionsTest, Abi14ContractRejectsNonDefaultTuning) {
  constexpr const char *nonDefault[] = {
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":2,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":2,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":14,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[64]}})json",
  };
  for (const char *serialized : nonDefault) {
    auto options = mlir::shuttle::parseShuttleXlaOptions(serialized);
    ASSERT_TRUE(options.ok()) << options.status();
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, kContractMapProgram);
    ASSERT_TRUE(module);
    EXPECT_TRUE(
        mlir::failed(mlir::shuttle::runShuttleXlaTransform(*module, *options)));
  }
}

TEST(ShuttleXlaOptionsTest,
     Abi14ParsesSeparateSourceOrderedAndFastContractCacheIdentities) {
  auto sourceOrdered =
      mlir::shuttle::parseShuttleXlaOptions(kAbi14ContractSourceOrderedOptions);
  auto fast = mlir::shuttle::parseShuttleXlaOptions(kAbi14ContractFastOptions);
  ASSERT_TRUE(sourceOrdered.ok()) << sourceOrdered.status();
  ASSERT_TRUE(fast.ok()) << fast.status();
  EXPECT_EQ(sourceOrdered->canonicalOptions,
            kAbi14ContractSourceOrderedOptions);
  EXPECT_EQ(fast->canonicalOptions, kAbi14ContractFastOptions);
  EXPECT_EQ(mlir::shuttle::shuttlePipelineIdentity(*sourceOrdered).policyDigest,
            "afc685fdac734de383e70a49de385ca744287f151816fbff419668b719dd5689");
  EXPECT_EQ(mlir::shuttle::shuttlePipelineIdentity(*fast).policyDigest,
            "06807ada40a32b66c33d8f4f682c1e30afd9918311c02b2f17447ae72a9290dc");
  EXPECT_NE(mlir::shuttle::shuttlePipelineIdentity(*sourceOrdered).policyDigest,
            mlir::shuttle::shuttlePipelineIdentity(*fast).policyDigest);
}

TEST(ShuttleXlaOptionsTest, Abi14RejectsAdjacentContractPipelineAbis) {
  constexpr const char *invalid[] = {
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":13,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":15,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
  };
  for (const char *serialized : invalid)
    EXPECT_EQ(mlir::shuttle::parseShuttleXlaOptions(serialized).status().code(),
              absl::StatusCode::kInvalidArgument);
}

TEST(ShuttleXlaOptionsTest,
     Abi14RunsOrdinaryJaxContractMapThroughBothPolicyEntries) {
  constexpr const char *serializedPolicies[] = {
      kAbi14ContractSourceOrderedOptions,
      kAbi14ContractFastOptions,
  };
  for (const char *serialized : serializedPolicies) {
    auto options = mlir::shuttle::parseShuttleXlaOptions(serialized);
    ASSERT_TRUE(options.ok()) << options.status();
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, kContractMapProgram);
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(
        mlir::shuttle::runShuttleXlaTransform(*module, *options)));
    std::string transformed;
    llvm::raw_string_ostream output(transformed);
    module->print(output);
    output.flush();
    EXPECT_EQ(transformed.find("shuttle.region"), std::string::npos);
    EXPECT_EQ(transformed.find("stablehlo.dot_general"), std::string::npos);
    EXPECT_NE(transformed.find("shuttle.gpu.executable_bundle.v2"),
              std::string::npos);
  }
}

TEST(ShuttleXlaOptionsTest,
     Abi14RunsOrdinaryJaxMatmulVjpThroughBothPolicyEntries) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  std::ifstream input(
      runfiles->Rlocation("shuttle_mlir/test/Inputs/"
                          "jax-0.10.1-bf16-matmul-vjp-17x33x19-backward.mlir"));
  std::ostringstream source;
  source << input.rdbuf();
  ASSERT_TRUE(input);

  constexpr const char *serializedPolicies[] = {
      kAbi14ContractSourceOrderedOptions,
      kAbi14ContractFastOptions,
  };
  for (const char *serialized : serializedPolicies) {
    auto options = mlir::shuttle::parseShuttleXlaOptions(serialized);
    ASSERT_TRUE(options.ok()) << options.status();
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, source.str());
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(
        mlir::shuttle::runShuttleXlaTransform(*module, *options)));
    std::string transformed;
    llvm::raw_string_ostream output(transformed);
    module->print(output);
    output.flush();
    EXPECT_EQ(transformed.find("shuttle.region"), std::string::npos);
    EXPECT_EQ(transformed.find("stablehlo.dot_general"), std::string::npos);
    EXPECT_NE(transformed.find("shuttle.gpu.executable_bundle.v2"),
              std::string::npos);
  }
}

TEST(ShuttleXlaOptionsTest,
     RunsOrdinaryJaxComposedMatmulAsTwoGenericCalls) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  std::ifstream input(
      runfiles->Rlocation("shuttle_mlir/test/Inputs/"
                          "jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir"));
  std::ostringstream source;
  source << input.rdbuf();
  ASSERT_TRUE(input);

  constexpr const char *serializedPolicies[] = {
      kAbi14ContractSourceOrderedOptions,
      kAbi14ContractFastOptions,
  };
  for (const char *serialized : serializedPolicies) {
    auto options = mlir::shuttle::parseShuttleXlaOptions(serialized);
    ASSERT_TRUE(options.ok()) << options.status();
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, source.str());
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(
        mlir::shuttle::runShuttleXlaTransform(*module, *options)));
    std::string transformed;
    llvm::raw_string_ostream output(transformed);
    module->print(output);
    output.flush();
    constexpr llvm::StringLiteral target =
        "shuttle.gpu.executable_bundle.v2";
    size_t first = transformed.find(target.str());
    ASSERT_NE(first, std::string::npos);
    size_t second = transformed.find(target.str(), first + target.size());
    ASSERT_NE(second, std::string::npos);
    EXPECT_EQ(transformed.find(target.str(), second + target.size()),
              std::string::npos);
    EXPECT_EQ(transformed.find("shuttle.region"), std::string::npos);
    EXPECT_EQ(transformed.find("stablehlo.dot_general"), std::string::npos);
  }
}

TEST(ShuttleXlaOptionsTest, RejectsInvalidOrNoncanonicalWireFormats) {
  constexpr const char *invalidOptions[] =
      {
          R"json({"execution_mode":"stablehlo_round_trip","numerics":"source_ordered","numerics":"fast","pipeline_abi_version":7,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"execution_mode":"stablehlo_round_trip","numerics":"source_ordered","pipeline_abi_version":7,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]},"workload":"named"})json",
          R"json({"execution_mode":"cpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":6,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"execution_mode":"cpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"execution_mode":"cpu_executable_bundle","numerics":"fast","pipeline_abi_version":7,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"execution_mode":"stablehlo_round_trip","numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"execution_mode":"future_consumer","numerics":"source_ordered","pipeline_abi_version":7,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":7,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":4,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":7,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":5,"schema_version":2,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[],"warps":4}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[1,1,1,1],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[1,1,1,1,1,1,1,1,1]}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":0,"tile_sizes":[]}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":true,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":2147483648,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"always_fuse","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({ "numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
          R"json({"schema_version":1,"numerics":"source_ordered","pipeline_abi_version":5,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      };
  for (const char *serialized : invalidOptions) {
    absl::StatusOr<mlir::shuttle::ShuttlePipelineOptions> parsed =
        mlir::shuttle::parseShuttleXlaOptions(serialized);
    EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument)
        << serialized << "\n"
        << parsed.status();
  }
}

TEST(ShuttleXlaOptionsTest, RejectsPreviousPipelineAbiForBothCpuPolicies) {
  constexpr const char *abi8Options[] = {
      R"json({"execution_mode":"cpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":8,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"cpu_executable_bundle","numerics":"fast","pipeline_abi_version":8,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
  };
  for (const char *serialized : abi8Options) {
    EXPECT_EQ(mlir::shuttle::parseShuttleXlaOptions(serialized).status().code(),
              absl::StatusCode::kInvalidArgument)
        << serialized;
  }
}

TEST(ShuttleXlaOptionsTest, RunsTheSharedPipelineFromSerializedOptions) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(
      mlir::shuttle::runShuttleXlaTransform(*module, kSourceOrderedOptions)
          .ok());

  std::string transformed;
  llvm::raw_string_ostream output(transformed);
  module->print(output);
  output.flush();
  EXPECT_EQ(transformed.find("shuttle."), std::string::npos);
  EXPECT_NE(transformed.find("stablehlo.tanh"), std::string::npos);
}

TEST(ShuttleXlaOptionsTest, RunsGpuFastPipelineFromSerializedOptions) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  std::ifstream input(runfiles->Rlocation(
      "shuttle_mlir/test/Inputs/"
      "jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir"));
  std::ostringstream source;
  source << input.rdbuf();
  ASSERT_TRUE(input);
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source.str());
  ASSERT_TRUE(module);
  EXPECT_TRUE(mlir::shuttle::runShuttleXlaTransform(
                  *module, kGpuExecutableBundleFastOptions)
                  .ok());

  std::string transformed;
  llvm::raw_string_ostream output(transformed);
  module->print(output);
  output.flush();
  EXPECT_EQ(transformed.find("shuttle.region"), std::string::npos);
  EXPECT_EQ(transformed.find("stablehlo.tanh"), std::string::npos);
  EXPECT_NE(transformed.find("shuttle.gpu.executable_bundle.v2"),
            std::string::npos);
}

} // namespace
