// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/IR/ShuttleOps.h"
#include "shuttle/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace {

constexpr llvm::StringLiteral kSourceOrderedOptions =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":19,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kFastOptions =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":19,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kDefaultTuning =
    R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";

struct BuiltModule {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

std::optional<std::string> fixtureSource() {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  if (!runfiles)
    return std::nullopt;
  std::ifstream input(runfiles->Rlocation(
      "shuttle_mlir/test/Inputs/abi19-bf16-terminal-fold-exp.mlir"));
  std::ostringstream source;
  source << input.rdbuf();
  if (!input)
    return std::nullopt;
  return source.str();
}

std::unique_ptr<BuiltModule>
buildAbi19Exp(llvm::StringRef source,
              mlir::shuttle::NumericalPolicy policy) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::shuttle::ShuttleDialect>();
  auto result = std::make_unique<BuiltModule>();
  result->context = std::make_unique<mlir::MLIRContext>(registry);
  result->module =
      mlir::parseSourceString<mlir::ModuleOp>(source, result->context.get());
  if (!result->module)
    return {};

  mlir::PassManager manager(result->context.get());
  manager.addPass(mlir::shuttle::createAnnotateSourcePass());
  manager.addPass(mlir::shuttle::createFormStructuralRegionsPass(
      policy,
      policy == mlir::shuttle::NumericalPolicy::Fast ? kFastOptions.str()
                                                     : kSourceOrderedOptions.str(),
      kDefaultTuning.str()));
  manager.addPass(mlir::shuttle::createConvertStablehloToAlgebraPass());
  manager.addPass(mlir::shuttle::createPlanRowFoldMaterializationPass());
  manager.addPass(mlir::shuttle::createPlanSimt32RowFoldSchedulePass());
  manager.addPass(mlir::shuttle::createBuildGpuExecutableBundlePass());
  manager.addPass(mlir::shuttle::createVerifyGpuExecutableBundlePass());
  if (mlir::failed(manager.run(*result->module)))
    return {};
  return result;
}

std::vector<std::string> ptxSlices(mlir::ModuleOp module) {
  std::vector<std::string> slices;
  for (mlir::shuttle::DeviceModuleOp device :
       module.getOps<mlir::shuttle::DeviceModuleOp>()) {
    llvm::ArrayRef<int8_t> code = device.getCode();
    for (mlir::shuttle::DeviceEntryOp entry :
         device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>()) {
      const int64_t offset = entry.getCodeOffset();
      const int64_t length = entry.getCodeLength();
      if (offset < 0 || length <= 0 ||
          static_cast<uint64_t>(offset) > code.size() ||
          static_cast<uint64_t>(length) > code.size() - offset)
        return {};
      slices.emplace_back(reinterpret_cast<const char *>(code.data() + offset),
                          length);
    }
  }
  return slices;
}

testing::AssertionResult isClosedDefaultExpPtx(llvm::StringRef ptx) {
  if (!ptx.starts_with(
          ".version 8.0\n.target sm_90\n.address_size 64\n"))
    return testing::AssertionFailure() << "wrong PTX target preamble";
  if (ptx.contains(".extern .func"))
    return testing::AssertionFailure() << "unresolved device function";
  if (ptx.contains("__nv_fast_expf"))
    return testing::AssertionFailure() << "fast exp symbol";
  if (ptx.contains(".ftz"))
    return testing::AssertionFailure() << "flush-to-zero instruction";
  if (!ptx.contains(".visible .entry shuttle_entry("))
    return testing::AssertionFailure() << "missing Shuttle entry";
  return testing::AssertionSuccess();
}

std::string replaceOnce(llvm::StringRef text, llvm::StringRef before,
                        llvm::StringRef after) {
  std::string result = text.str();
  const size_t position = result.find(before.str());
  if (position != std::string::npos)
    result.replace(position, before.size(), after.str());
  return result;
}

TEST(Abi19LibdeviceContractTest,
     FixtureRequestsOneImplicitDefaultAccuracyExponential) {
  auto source = fixtureSource();
  ASSERT_TRUE(source);
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(*source, &context);
  ASSERT_TRUE(module);
  int64_t exponentials = 0;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() != "stablehlo.exponential")
      return;
    ++exponentials;
    EXPECT_FALSE(operation->getAttr("result_accuracy"));
  });
  EXPECT_EQ(exponentials, 1);
}

TEST(Abi19LibdeviceContractTest,
     DefaultExpProducesDeterministicClosedPtxForBothPolicies) {
  auto source = fixtureSource();
  ASSERT_TRUE(source);
  for (mlir::shuttle::NumericalPolicy policy :
       {mlir::shuttle::NumericalPolicy::SourceOrdered,
        mlir::shuttle::NumericalPolicy::Fast}) {
    auto first = buildAbi19Exp(*source, policy);
    ASSERT_TRUE(first)
        << "ABI19 remains red until the official libdevice artifact is sealed";
    auto second = buildAbi19Exp(*source, policy);
    ASSERT_TRUE(second);
    std::vector<std::string> firstPtx = ptxSlices(*first->module);
    std::vector<std::string> secondPtx = ptxSlices(*second->module);
    ASSERT_FALSE(firstPtx.empty());
    ASSERT_EQ(firstPtx, secondPtx);

    std::string renamed = replaceOnce(*source, "@abi19_exp_contract",
                                      "@anonymous_module");
    renamed = replaceOnce(renamed, "@main", "@anonymous_entry");
    auto renamedBuild = buildAbi19Exp(renamed, policy);
    ASSERT_TRUE(renamedBuild);
    EXPECT_EQ(firstPtx, ptxSlices(*renamedBuild->module));

    for (const std::string &ptx : firstPtx) {
      ASSERT_TRUE(isClosedDefaultExpPtx(ptx));

      EXPECT_FALSE(isClosedDefaultExpPtx(
          replaceOnce(ptx, ".version 8.0", ".version 8.1")));
      EXPECT_FALSE(isClosedDefaultExpPtx(
          replaceOnce(ptx, ".target sm_90", ".target sm_100")));
      EXPECT_FALSE(isClosedDefaultExpPtx(
          replaceOnce(ptx, ".address_size 64\n",
                      ".address_size 64\n.extern .func "
                      "__shuttle_unresolved();\n")));
    }
  }
}

} // namespace
