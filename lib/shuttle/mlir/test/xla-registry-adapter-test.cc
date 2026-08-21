// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "shuttle/Transforms/Observer.h"
#include "shuttle/Transforms/XlaRegistration.h"
#include "stablehlo/dialect/Register.h"
#include "xla/pjrt/stablehlo_module_transform.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {

constexpr char kOptions[] =
    R"json({"execution_mode":"stablehlo_round_trip","numerics":"source_ordered","pipeline_abi_version":21,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kSm100GpuOptions[] =
    R"json({"execution_mode":"gpu_executable_bundle","gpu_target":"sm_100","numerics":"source_ordered","pipeline_abi_version":23,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kProgram[] = R"mlir(
module {
  func.func @main(%arg0: tensor<7xf32>) -> tensor<7xf32> {
    %0 = stablehlo.tanh %arg0 : tensor<7xf32>
    return %0 : tensor<7xf32>
  }
}
)mlir";
constexpr char kContractMapProgram[] = R"mlir(
module {
  func.func @main(%lhs: tensor<3x5xbf16>, %rhs: tensor<5x2xbf16>) -> tensor<3x2xbf16> {
    %contract = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x5xbf16>, tensor<5x2xbf16>) -> tensor<3x2xf32>
    %result = stablehlo.convert %contract : (tensor<3x2xf32>) -> tensor<3x2xbf16>
    return %result : tensor<3x2xbf16>
  }
}
)mlir";
constexpr char kFailingProgram[] = R"mlir(
module {
  func.func @main(%arg0: tensor<7xf32>) -> tensor<7xf32> {
    return %arg0 : tensor<7xf32>
  ^bb1(%value: tensor<7xf32>):
    return %value : tensor<7xf32>
  }
}
)mlir";

class RecordingObserver final : public mlir::shuttle::ShuttlePipelineObserver {
public:
  struct Snapshot {
    std::vector<mlir::shuttle::ShuttlePipelinePhase> phases;
    std::vector<std::string> policyDigests;
  };

  void observe(const mlir::shuttle::ShuttlePipelineEvent &event) const final {
    std::lock_guard<std::mutex> lock(mutex);
    phases.push_back(event.phase());
    policyDigests.push_back(event.identity().policyDigest);
  }

  Snapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return Snapshot{phases, policyDigests};
  }

private:
  mutable std::mutex mutex;
  mutable std::vector<mlir::shuttle::ShuttlePipelinePhase> phases;
  mutable std::vector<std::string> policyDigests;
};

mlir::OwningOpRef<mlir::ModuleOp> parseModule(mlir::MLIRContext &context,
                                              llvm::StringRef source) {
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
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

TEST(ShuttleXlaRegistryAdapterTest,
     RegistersAndRunsTheExactObservedPipelineTransactionally) {
  ASSERT_TRUE(mlir::shuttle::registerShuttleXlaTransform().ok());
  ASSERT_TRUE(mlir::shuttle::registerShuttleXlaTransform().ok());
  auto observer = std::make_shared<RecordingObserver>();
  auto subscription = mlir::shuttle::subscribeShuttlePipelineObserver(observer);
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, kProgram);
  ASSERT_TRUE(module);

  absl::Status status = xla::StablehloModuleTransformRegistry::Global().Run(
      *module, kOptions, "shuttle");
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(moduleText(*module).find("shuttle."), std::string::npos);
  RecordingObserver::Snapshot observed = observer->snapshot();
  ASSERT_EQ(observed.phases.size(), 3);
  EXPECT_EQ(observed.phases[0],
            mlir::shuttle::ShuttlePipelinePhase::AlgebraCoverage);
  EXPECT_EQ(observed.phases[1],
            mlir::shuttle::ShuttlePipelinePhase::LoweredCoverage);
  EXPECT_EQ(observed.phases[2],
            mlir::shuttle::ShuttlePipelinePhase::FinalErasure);
  EXPECT_EQ(observed.policyDigests[0], observed.policyDigests[2]);

  mlir::MLIRContext invalidContext;
  mlir::OwningOpRef<mlir::ModuleOp> invalid =
      parseModule(invalidContext, kProgram);
  ASSERT_TRUE(invalid);
  std::string invalidBefore = moduleText(*invalid);
  status = xla::StablehloModuleTransformRegistry::Global().Run(*invalid, "{}",
                                                               "shuttle");
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(moduleText(*invalid), invalidBefore);
  EXPECT_EQ(observer->snapshot().phases.size(), 3);

  mlir::MLIRContext failingContext;
  mlir::OwningOpRef<mlir::ModuleOp> failing =
      parseModule(failingContext, kFailingProgram);
  ASSERT_TRUE(failing);
  std::string failingBefore = moduleText(*failing);
  status = xla::StablehloModuleTransformRegistry::Global().Run(
      *failing, kOptions, "shuttle");
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(moduleText(*failing), failingBefore);
  observed = observer->snapshot();
  ASSERT_EQ(observed.phases.size(), 4);
  EXPECT_EQ(observed.phases.back(),
            mlir::shuttle::ShuttlePipelinePhase::Failure);
}

TEST(ShuttleXlaRegistryAdapterTest,
     Abi23Sm100PreferFusionRunsThroughPlanningAndCodegen) {
  ASSERT_TRUE(mlir::shuttle::registerShuttleXlaTransform().ok());
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, kContractMapProgram);
  ASSERT_TRUE(module);

  absl::Status status = xla::StablehloModuleTransformRegistry::Global().Run(
      *module, kSm100GpuOptions, "shuttle");
  ASSERT_TRUE(status.ok()) << status;
  std::string transformed = moduleText(*module);
  EXPECT_EQ(transformed.find("shuttle.region"), std::string::npos);
  EXPECT_EQ(transformed.find("stablehlo.dot_general"), std::string::npos);
  EXPECT_NE(transformed.find("shuttle.gpu.executable_bundle.v2"),
            std::string::npos);
  mlir::Operation *customCall = nullptr;
  int64_t customCallCount = 0;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "stablehlo.custom_call") {
      customCall = operation;
      ++customCallCount;
    }
  });
  EXPECT_EQ(customCallCount, 1);
  ASSERT_NE(customCall, nullptr);
  auto backendConfig =
      customCall->getAttrOfType<mlir::DictionaryAttr>("backend_config");
  ASSERT_TRUE(backendConfig);
  auto deviceSchemaVersion =
      backendConfig.getAs<mlir::IntegerAttr>("device_schema_version");
  ASSERT_TRUE(deviceSchemaVersion);
  EXPECT_EQ(deviceSchemaVersion.getInt(), 4);
  auto bundleBytes = backendConfig.getAs<mlir::StringAttr>("bundle_bytes");
  ASSERT_TRUE(bundleBytes);
  llvm::StringRef serialized = bundleBytes.getValue();
  auto executable = mlir::shuttle::GpuExecutable::Load(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(serialized.data()), serialized.size()));
  ASSERT_TRUE(executable.ok()) << executable.status();
  EXPECT_EQ((*executable)->deviceSchemaVersion(), 4);
  EXPECT_EQ((*executable)->codeFormat(),
            mlir::shuttle::ExecutableCodeFormat::CudaPtxSm100V1);
  llvm::StringRef ptx(
      reinterpret_cast<const char *>((*executable)->codeBytes().data()),
      (*executable)->codeBytes().size());
  EXPECT_TRUE(
      ptx.starts_with(".version 8.7\n.target sm_100\n.address_size 64\n"));
}

} // namespace
