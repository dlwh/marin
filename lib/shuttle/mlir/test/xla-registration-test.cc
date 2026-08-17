// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <sstream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "shuttle/Transforms/Observer.h"
#include "shuttle/Transforms/XlaRegistration.h"
#include "stablehlo/dialect/Register.h"
#include "tools/cpp/runfiles/runfiles.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {

constexpr char kDefaultOptions[] =
    R"json({"execution_mode":"stablehlo_round_trip","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kSourceOrderedOptions[] =
    R"json({"execution_mode":"stablehlo_round_trip","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}})json";
constexpr char kFastOptions[] =
    R"json({"execution_mode":"stablehlo_round_trip","numerics":"fast","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kRetiredCpuExecutableBundleOptions[] =
    R"json({"execution_mode":"cpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}})json";
constexpr char kAbi21DefaultCpuExecutableBundleOptions[] =
    R"json({"execution_mode":"cpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kGpuExecutableBundleOptions[] =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}})json";
constexpr char kGpuExecutableBundleFastOptions[] =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}})json";
constexpr char kAbi21ProjectedParameterSourceOrderedOptions[] =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kAbi21ProjectedParameterFastOptions[] =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kHistoricalAbi17GpuOptions[] =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":17,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
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
            "95a589038ad56dc8db7734cea3ef87863a32860003db22f2b8077ff77f9cf8b3");
  EXPECT_EQ(sourceIdentity.tuningDigest,
            "ae69cb474b1ddc91067687e7351ee27afe4e3b0814ae59e310a42bec5911326f");

  absl::StatusOr<mlir::shuttle::ShuttlePipelineOptions> fast =
      mlir::shuttle::parseShuttleXlaOptions(kFastOptions);
  ASSERT_TRUE(fast.ok()) << fast.status();
  EXPECT_EQ(fast->numerics, mlir::shuttle::NumericalPolicy::Fast);
  EXPECT_NE(mlir::shuttle::shuttlePipelineIdentity(*source).policyDigest,
            mlir::shuttle::shuttlePipelineIdentity(*fast).policyDigest);

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

TEST(ShuttleXlaOptionsTest, RejectsRetiredCpuExecutableMode) {
  auto parsed =
      mlir::shuttle::parseShuttleXlaOptions(kRetiredCpuExecutableBundleOptions);
  EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument)
      << parsed.status();
}

TEST(ShuttleXlaOptionsTest, RejectsAdjacentGpuPipelineAbis) {
  constexpr const char *invalid[] = {
      kHistoricalAbi17GpuOptions,
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":18,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":19,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":20,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
  };
  for (const char *serialized : invalid) {
    auto parsed = mlir::shuttle::parseShuttleXlaOptions(serialized);
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(std::string(parsed.status().message()).find("integer 21"),
              std::string::npos);
  }
}

TEST(ShuttleXlaOptionsTest, Abi21PreservesCanonicalSourceOrderedFoldMapOptions) {
  auto options =
      mlir::shuttle::parseShuttleXlaOptions(
          kAbi21ProjectedParameterSourceOrderedOptions);
  ASSERT_TRUE(options.ok()) << options.status();
  EXPECT_EQ(options->canonicalOptions,
            kAbi21ProjectedParameterSourceOrderedOptions);
  EXPECT_EQ(
      options->canonicalTuning,
      R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json");
  EXPECT_EQ(mlir::shuttle::shuttlePipelineIdentity(*options).policyDigest,
            "731a06b7bd067960293e47448e483e07ca7373a6ff09bfa94fb3dc804d7de280");

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

TEST(ShuttleXlaOptionsTest, Abi21PreservesFastMapFold) {
  auto options =
      mlir::shuttle::parseShuttleXlaOptions(
          kAbi21ProjectedParameterFastOptions);
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

TEST(ShuttleXlaOptionsTest, Abi21FoldAndBroadcastRejectNonDefaultTuning) {
  constexpr const char *nonDefault[] = {
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":2,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":2,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[64]}})json",
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
     Abi21ParsesSeparateSourceOrderedAndFastCacheIdentities) {
  auto sourceOrdered =
      mlir::shuttle::parseShuttleXlaOptions(
          kAbi21ProjectedParameterSourceOrderedOptions);
  auto fast = mlir::shuttle::parseShuttleXlaOptions(
      kAbi21ProjectedParameterFastOptions);
  ASSERT_TRUE(sourceOrdered.ok()) << sourceOrdered.status();
  ASSERT_TRUE(fast.ok()) << fast.status();
  EXPECT_EQ(sourceOrdered->canonicalOptions,
            kAbi21ProjectedParameterSourceOrderedOptions);
  EXPECT_EQ(fast->canonicalOptions, kAbi21ProjectedParameterFastOptions);
  EXPECT_EQ(mlir::shuttle::shuttlePipelineIdentity(*sourceOrdered).policyDigest,
            "731a06b7bd067960293e47448e483e07ca7373a6ff09bfa94fb3dc804d7de280");
  EXPECT_EQ(mlir::shuttle::shuttlePipelineIdentity(*fast).policyDigest,
            "c658672602d79f23894da60c8300ecdaf40905acefe9f33ad58216571f48d2da");
  EXPECT_NE(mlir::shuttle::shuttlePipelineIdentity(*sourceOrdered).policyDigest,
            mlir::shuttle::shuttlePipelineIdentity(*fast).policyDigest);
}

TEST(ShuttleXlaOptionsTest, Abi21RejectsAdjacentPipelineAbis) {
  constexpr const char *invalid[] = {
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":19,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":20,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
  };
  for (const char *serialized : invalid)
    EXPECT_EQ(mlir::shuttle::parseShuttleXlaOptions(serialized).status().code(),
              absl::StatusCode::kInvalidArgument);
}

TEST(ShuttleXlaOptionsTest,
     Abi21PreservesOrdinaryJaxContractMapThroughBothPolicyEntries) {
  constexpr const char *serializedPolicies[] = {
      kAbi21ProjectedParameterSourceOrderedOptions,
      kAbi21ProjectedParameterFastOptions,
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
     Abi21PreservesOrdinaryJaxMatmulVjpThroughBothPolicyEntries) {
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
      kAbi21ProjectedParameterSourceOrderedOptions,
      kAbi21ProjectedParameterFastOptions,
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
     Abi21PreservesProjectedBatchedContractThroughBothPolicyEntries) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  constexpr const char *filenames[] = {
      "jax-0.10.1-bf16-batched_contract-2x5x3x7x11-forward.mlir",
      "jax-0.10.1-bf16-batched_contract-2x5x3x7x11-backward.mlir",
      "jax-0.10.1-bf16-batched_contract-2x5x3x7x11-composed.mlir",
      "jax-0.10.1-bf16-batched_contract-3x4x2x13x6-forward.mlir",
      "jax-0.10.1-bf16-batched_contract-3x4x2x13x6-backward.mlir",
      "jax-0.10.1-bf16-batched_contract-3x4x2x13x6-composed.mlir",
  };
  constexpr const char *serializedPolicies[] = {
      kAbi21ProjectedParameterSourceOrderedOptions,
      kAbi21ProjectedParameterFastOptions,
  };
  constexpr llvm::StringLiteral target =
      "shuttle.gpu.executable_bundle.v2";

  for (const char *filename : filenames) {
    SCOPED_TRACE(filename);
    std::ifstream input(runfiles->Rlocation(
        (llvm::Twine("shuttle_mlir/test/Inputs/") + filename).str()));
    std::ostringstream source;
    source << input.rdbuf();
    ASSERT_TRUE(input);
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
      size_t first = transformed.find(target.str());
      ASSERT_NE(first, std::string::npos);
      EXPECT_EQ(transformed.find(target.str(), first + target.size()),
                std::string::npos);
      EXPECT_EQ(transformed.find("shuttle.region"), std::string::npos);
      EXPECT_EQ(transformed.find("stablehlo.dot_general"), std::string::npos);
    }
  }
}

TEST(ShuttleXlaOptionsTest,
     Abi21PreservesOrdinaryJaxRankNFoldMatrixThroughBothPolicyEntries) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  constexpr const char *filenames[] = {
      "jax-0.10.1-bf16-rank_n_fold_map-2x5x17-forward.mlir",
      "jax-0.10.1-bf16-rank_n_fold_map-2x5x17-backward.mlir",
      "jax-0.10.1-bf16-rank_n_fold_map-2x5x17-composed.mlir",
      "jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-forward.mlir",
      "jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-backward.mlir",
      "jax-0.10.1-bf16-rank_n_fold_map-3x2x4x7x269-composed.mlir",
  };
  constexpr const char *serializedPolicies[] = {
      kAbi21ProjectedParameterSourceOrderedOptions,
      kAbi21ProjectedParameterFastOptions,
  };
  for (const char *filename : filenames) {
    SCOPED_TRACE(filename);
    std::ifstream input(runfiles->Rlocation(
        (llvm::Twine("shuttle_mlir/test/Inputs/") + filename).str()));
    std::ostringstream source;
    source << input.rdbuf();
    ASSERT_TRUE(input);
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
      EXPECT_EQ(transformed.find("stablehlo.reduce"), std::string::npos);
      EXPECT_NE(transformed.find("shuttle.gpu.executable_bundle.v2"),
                std::string::npos);
    }
  }
}

TEST(ShuttleXlaOptionsTest,
     Abi21RunsOrdinaryJaxSingletonBroadcastMatrixThroughBothPolicyEntries) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  constexpr const char *filenames[] = {
      "jax-0.10.1-bf16-singleton_broadcast-2x7x19-forward.mlir",
      "jax-0.10.1-bf16-singleton_broadcast-2x7x19-backward.mlir",
      "jax-0.10.1-bf16-singleton_broadcast-2x7x19-composed.mlir",
      "jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-forward.mlir",
      "jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-backward.mlir",
      "jax-0.10.1-bf16-singleton_broadcast-3x2x5x11x263-composed.mlir",
  };
  constexpr const char *serializedPolicies[] = {
      kAbi21ProjectedParameterSourceOrderedOptions,
      kAbi21ProjectedParameterFastOptions,
  };
  for (const char *filename : filenames) {
    SCOPED_TRACE(filename);
    std::ifstream input(runfiles->Rlocation(
        (llvm::Twine("shuttle_mlir/test/Inputs/") + filename).str()));
    std::ostringstream source;
    source << input.rdbuf();
    ASSERT_TRUE(input);
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
      EXPECT_EQ(transformed.find("stablehlo.broadcast_in_dim"),
                std::string::npos);
      EXPECT_NE(transformed.find("shuttle.gpu.executable_bundle.v2"),
                std::string::npos);
    }
  }
}

TEST(ShuttleXlaOptionsTest,
     Abi21RunsOrdinaryJaxProjectedParameterRmsNormThroughBothPolicies) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  constexpr const char *filenames[] = {
      "jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir",
      "jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir",
      "jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-composed.mlir",
      "jax-0.10.1-bf16-projected_parameter_broadcast-"
      "3x2x5x11x263-forward.mlir",
      "jax-0.10.1-bf16-projected_parameter_broadcast-"
      "3x2x5x11x263-backward.mlir",
      "jax-0.10.1-bf16-projected_parameter_broadcast-"
      "3x2x5x11x263-composed.mlir",
  };
  constexpr const char *serializedPolicies[] = {
      kAbi21ProjectedParameterSourceOrderedOptions,
      kAbi21ProjectedParameterFastOptions,
  };
  for (const char *filename : filenames) {
    SCOPED_TRACE(filename);
    std::ifstream input(runfiles->Rlocation(
        (llvm::Twine("shuttle_mlir/test/Inputs/") + filename).str()));
    std::ostringstream source;
    source << input.rdbuf();
    ASSERT_TRUE(input);
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
      EXPECT_EQ(transformed.find("stablehlo.broadcast_in_dim"),
                std::string::npos);
      EXPECT_EQ(transformed.find("stablehlo.reduce"), std::string::npos);
      EXPECT_NE(transformed.find("shuttle.gpu.executable_bundle.v2"),
                std::string::npos);
    }
  }
}

TEST(ShuttleXlaOptionsTest,
     Abi21CpuRejectsFullPrefixFoldWithoutPartiallyRewritingModule) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  std::ifstream input(runfiles->Rlocation(
      "shuttle_mlir/test/Inputs/"
      "jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir"));
  std::ostringstream source;
  source << input.rdbuf();
  ASSERT_TRUE(input);
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, source.str());
  ASSERT_TRUE(module);
  std::string before;
  llvm::raw_string_ostream beforeOutput(before);
  module->print(beforeOutput);
  beforeOutput.flush();
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream output(diagnostics);
        diagnostic.print(output);
        output << '\n';
        output.flush();
        return mlir::success();
      });
  absl::Status status = mlir::shuttle::runShuttleXlaTransform(
      *module, kAbi21DefaultCpuExecutableBundleOptions);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(diagnostics.find(
                "requires a bounded generated Map/Fold CPU bytecode subset"),
            std::string::npos)
      << diagnostics;
  std::string after;
  llvm::raw_string_ostream afterOutput(after);
  module->print(afterOutput);
  afterOutput.flush();
  EXPECT_EQ(after, before);
  EXPECT_EQ(after.find("shuttle.cpu.executable_bundle.v1"), std::string::npos);
}

TEST(ShuttleXlaOptionsTest,
     Abi21CpuStillRejectsRankNFoldWithoutPartiallyRewritingTheModule) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  std::ifstream input(runfiles->Rlocation(
      "shuttle_mlir/test/Inputs/"
      "jax-0.10.1-bf16-rank_n_fold_map-2x5x17-forward.mlir"));
  std::ostringstream source;
  source << input.rdbuf();
  ASSERT_TRUE(input);
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, source.str());
  ASSERT_TRUE(module);
  std::string before;
  llvm::raw_string_ostream beforeOutput(before);
  module->print(beforeOutput);
  beforeOutput.flush();

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream output(diagnostics);
        diagnostic.print(output);
        output << '\n';
        output.flush();
        return mlir::success();
      });
  absl::Status status = mlir::shuttle::runShuttleXlaTransform(
      *module, kAbi21DefaultCpuExecutableBundleOptions);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(diagnostics.find(
                "requires a bounded generated Map/Fold CPU bytecode subset"),
            std::string::npos)
      << diagnostics;
  std::string after;
  llvm::raw_string_ostream afterOutput(after);
  module->print(afterOutput);
  afterOutput.flush();
  EXPECT_EQ(after, before);
  EXPECT_EQ(after.find("shuttle.cpu.executable_bundle.v1"), std::string::npos);
}

TEST(ShuttleXlaOptionsTest,
     Abi21CpuRejectsSingletonBroadcastWithoutPartiallyRewritingTheModule) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  ASSERT_TRUE(runfiles) << error;
  std::ifstream input(runfiles->Rlocation(
      "shuttle_mlir/test/Inputs/"
      "jax-0.10.1-bf16-singleton_broadcast-2x7x19-forward.mlir"));
  std::ostringstream source;
  source << input.rdbuf();
  ASSERT_TRUE(input);
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, source.str());
  ASSERT_TRUE(module);
  std::string before;
  llvm::raw_string_ostream beforeOutput(before);
  module->print(beforeOutput);
  beforeOutput.flush();

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream output(diagnostics);
        diagnostic.print(output);
        output << '\n';
        output.flush();
        return mlir::success();
      });
  absl::Status status = mlir::shuttle::runShuttleXlaTransform(
      *module, kAbi21DefaultCpuExecutableBundleOptions);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(diagnostics.find(
                "requires a bounded generated Map/Fold CPU bytecode subset"),
            std::string::npos)
      << diagnostics;
  std::string after;
  llvm::raw_string_ostream afterOutput(after);
  module->print(afterOutput);
  afterOutput.flush();
  EXPECT_EQ(after, before);
  EXPECT_EQ(after.find("shuttle.cpu.executable_bundle.v1"), std::string::npos);
}

TEST(ShuttleXlaOptionsTest, RunsOrdinaryJaxComposedMatmulAsTwoGenericCalls) {
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
      kAbi21ProjectedParameterSourceOrderedOptions,
      kAbi21ProjectedParameterFastOptions,
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
    constexpr llvm::StringLiteral target = "shuttle.gpu.executable_bundle.v2";
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
