// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
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
              mlir::shuttle::NumericalPolicy policy,
              bool buildBundle = true) {
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
  if (buildBundle) {
    manager.addPass(mlir::shuttle::createBuildGpuExecutableBundlePass());
    manager.addPass(mlir::shuttle::createVerifyGpuExecutableBundlePass());
  }
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

size_t skipBalancedParentheses(llvm::StringRef text, size_t position) {
  int64_t depth = 0;
  for (; position < text.size(); ++position) {
    if (text[position] == '(')
      ++depth;
    else if (text[position] == ')' && --depth == 0)
      return position + 1;
  }
  return llvm::StringRef::npos;
}

std::optional<std::string> functionName(llvm::StringRef text, size_t position) {
  position = text.find_first_not_of(" \t\r\n", position);
  if (position == llvm::StringRef::npos)
    return std::nullopt;
  if (text[position] == '(') {
    position = skipBalancedParentheses(text, position);
    if (position == llvm::StringRef::npos)
      return std::nullopt;
    position = text.find_first_not_of(" \t\r\n,", position);
  }
  if (position == llvm::StringRef::npos)
    return std::nullopt;
  size_t end = text.find_first_of(" \t\r\n,(", position);
  if (end == llvm::StringRef::npos)
    end = text.size();
  if (end == position)
    return std::nullopt;
  return text.slice(position, end).str();
}

std::set<std::string> definedFunctions(llvm::StringRef ptx) {
  std::set<std::string> names;
  size_t position = 0;
  while ((position = ptx.find(".func", position)) != llvm::StringRef::npos) {
    auto name = functionName(ptx, position + 5);
    if (!name)
      break;
    const size_t body = ptx.find('{', position);
    const size_t declaration = ptx.find(';', position);
    if (body != llvm::StringRef::npos &&
        (declaration == llvm::StringRef::npos || body < declaration))
      names.insert(*name);
    position += 5;
  }
  return names;
}

testing::AssertionResult hasClosedCalls(llvm::StringRef ptx) {
  const std::set<std::string> definitions = definedFunctions(ptx);
  size_t position = 0;
  while ((position = ptx.find("call", position)) != llvm::StringRef::npos) {
    size_t suffix = position + 4;
    if (ptx.substr(suffix).starts_with(".uni"))
      suffix += 4;
    auto target = functionName(ptx, suffix);
    if (!target)
      return testing::AssertionFailure() << "malformed call";
    if (definitions.find(*target) == definitions.end())
      return testing::AssertionFailure() << "undefined call target " << *target;
    position = suffix;
  }
  return testing::AssertionSuccess();
}

std::optional<std::string> firstOperand(llvm::StringRef instruction) {
  const size_t space = instruction.find_first_of(" \t");
  if (space == llvm::StringRef::npos)
    return std::nullopt;
  const size_t start = instruction.find_first_not_of(" \t", space);
  const size_t comma = instruction.find(',', start);
  if (start == llvm::StringRef::npos || comma == llvm::StringRef::npos)
    return std::nullopt;
  return instruction.slice(start, comma).trim().str();
}

testing::AssertionResult hasExpResultStoreDataflow(llvm::StringRef ptx) {
  const size_t call = ptx.find("call.uni (");
  if (call == llvm::StringRef::npos)
    return testing::AssertionFailure() << "missing __nv_expf return call";
  const size_t returnStart = call + llvm::StringRef("call.uni (").size();
  const size_t returnEnd = ptx.find(')', returnStart);
  if (returnEnd == llvm::StringRef::npos ||
      !ptx.substr(returnEnd).starts_with("), __nv_expf,"))
    return testing::AssertionFailure() << "missing __nv_expf return call";
  const std::string returned = ptx.slice(returnStart, returnEnd).trim().str();

  const size_t load = ptx.find("ld.param", returnEnd);
  const size_t loadEnd = ptx.find(';', load);
  if (load == llvm::StringRef::npos || loadEnd == llvm::StringRef::npos)
    return testing::AssertionFailure() << "missing __nv_expf return load";
  const llvm::StringRef loadInstruction = ptx.slice(load, loadEnd);
  if (!loadInstruction.contains("[" + returned + "]"))
    return testing::AssertionFailure() << "missing __nv_expf return load";
  auto f32Result = firstOperand(loadInstruction);
  if (!f32Result)
    return testing::AssertionFailure() << "malformed __nv_expf return load";

  const size_t conversion = ptx.find("cvt.rn.bf16.f32", loadEnd);
  const size_t conversionEnd = ptx.find(';', conversion);
  if (conversion == llvm::StringRef::npos ||
      conversionEnd == llvm::StringRef::npos)
    return testing::AssertionFailure()
           << "__nv_expf result does not feed BF16 conversion";
  const llvm::StringRef conversionInstruction =
      ptx.slice(conversion, conversionEnd);
  if (!conversionInstruction.ends_with(", " + *f32Result))
    return testing::AssertionFailure()
           << "__nv_expf result does not feed BF16 conversion";
  auto bf16Result = firstOperand(conversionInstruction);
  if (!bf16Result)
    return testing::AssertionFailure() << "malformed BF16 conversion";

  const size_t store = ptx.find("st.global", conversionEnd);
  const size_t storeEnd = ptx.find(';', store);
  if (store == llvm::StringRef::npos || storeEnd == llvm::StringRef::npos ||
      !ptx.slice(store, storeEnd).ends_with(", " + *bf16Result))
    return testing::AssertionFailure()
           << "converted __nv_expf result does not feed output store";
  return testing::AssertionSuccess();
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
  const std::set<std::string> definitions = definedFunctions(ptx);
  if (definitions.find("__nv_expf") == definitions.end())
    return testing::AssertionFailure() << "missing linked __nv_expf definition";
  testing::AssertionResult closedCalls = hasClosedCalls(ptx);
  if (!closedCalls)
    return closedCalls;
  testing::AssertionResult expDataflow = hasExpResultStoreDataflow(ptx);
  if (!expDataflow)
    return expDataflow;
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
     FixtureRequestsTwoImplicitDefaultAccuracyExponentials) {
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
  EXPECT_EQ(exponentials, 2);
}

TEST(Abi19LibdeviceContractTest,
     ScheduledAlgebraRetainsBothExponentialBodiesForBothPolicies) {
  auto source = fixtureSource();
  ASSERT_TRUE(source);
  for (mlir::shuttle::NumericalPolicy policy :
       {mlir::shuttle::NumericalPolicy::SourceOrdered,
        mlir::shuttle::NumericalPolicy::Fast}) {
    auto scheduled = buildAbi19Exp(*source, policy, false);
    ASSERT_TRUE(scheduled);
    int64_t exponentials = 0;
    scheduled->module->walk([&](mlir::math::ExpOp exponential) {
      EXPECT_TRUE(exponential->getParentOfType<mlir::shuttle::MapOp>());
      ++exponentials;
    });
    EXPECT_EQ(exponentials, 2);
  }
}

TEST(Abi19LibdeviceContractTest, ClosedPtxOracleRejectsSemanticMutations) {
  constexpr llvm::StringLiteral ptx = R"ptx(.version 8.0
.target sm_90
.address_size 64
.func (.param .b32 __ret) __nv_expf(.param .b32 __arg) {
  ret;
}
.visible .entry shuttle_entry() {
  .param .b32 __exp_input;
  .param .b32 __exp_return;
  st.param.b32 [__exp_input], %f1;
  call.uni (__exp_return), __nv_expf, (__exp_input);
  ld.param.b32 %f2, [__exp_return];
  cvt.rn.bf16.f32 %rs1, %f2;
  st.global.u16 [%rd1], %rs1;
  ret;
}
)ptx";
  ASSERT_TRUE(isClosedDefaultExpPtx(ptx));
  EXPECT_FALSE(isClosedDefaultExpPtx(
      replaceOnce(ptx, "call.uni (__exp_return), __nv_expf, (__exp_input);",
                  "mov.f32 %f2, %f1;")));
  EXPECT_FALSE(isClosedDefaultExpPtx(
      replaceOnce(ptx, "cvt.rn.bf16.f32 %rs1, %f2;",
                  "cvt.rn.bf16.f32 %rs1, %f1;")));
  EXPECT_FALSE(isClosedDefaultExpPtx(replaceOnce(
      ptx, "call.uni (__exp_return), __nv_expf, (__exp_input);",
      "call.uni (__exp_return), __shuttle_missing, (__exp_input);")));
  EXPECT_FALSE(isClosedDefaultExpPtx(
      replaceOnce(ptx, "__nv_expf", "__nv_fast_expf")));
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
