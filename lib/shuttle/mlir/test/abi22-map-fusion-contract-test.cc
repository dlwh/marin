// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/IR/ShuttleOps.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "shuttle/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace {

constexpr llvm::StringLiteral kAutomaticTuning =
    R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
constexpr llvm::StringLiteral kFusionTuning =
    R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
constexpr llvm::StringLiteral kMaterializationTuning =
    R"json({"cluster_shape":[],"materialization":"prefer_materialization","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";

struct FixtureExpectation {
  const char *filename;
  int64_t maps;
  int64_t folds;
  int64_t materializedTasks;
  int64_t materializedBuffers;
  int64_t fusedTasks;
  int64_t fusedBuffers;
};

// Counts are consequences of the anonymous algebra SSA graph. They are test
// oracles and are never compiler selectors.
constexpr std::array<FixtureExpectation, 12> kFixtures = {{
    {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir",
     18, 1, 19, 21, 11, 13},
    {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir",
     43, 5, 48, 51, 30, 33},
    {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-composed.mlir",
     46, 5, 51, 54, 33, 36},
    {"jax-0.10.1-bf16-projected_parameter_broadcast-"
     "3x2x5x11x263-forward.mlir",
     18, 1, 19, 21, 11, 13},
    {"jax-0.10.1-bf16-projected_parameter_broadcast-"
     "3x2x5x11x263-backward.mlir",
     43, 5, 48, 51, 30, 33},
    {"jax-0.10.1-bf16-projected_parameter_broadcast-"
     "3x2x5x11x263-composed.mlir",
     46, 5, 51, 54, 33, 36},
    {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir", 29, 2,
     31, 34, 18, 21},
    {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-backward.mlir", 63,
     10, 73, 76, 49, 52},
    {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-composed.mlir", 70,
     10, 80, 84, 54, 58},
    {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir", 29, 2,
     31, 34, 18, 21},
    {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-backward.mlir", 63,
     10, 73, 76, 49, 52},
    {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir", 70,
     10, 80, 84, 54, 58},
}};

struct BuiltModule {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

struct PlanSummary {
  int64_t maps = 0;
  int64_t folds = 0;
  int64_t contracts = 0;
  int64_t tasks = 0;
  int64_t buffers = 0;
  int64_t mapTasks = 0;
  int64_t foldTasks = 0;
  int64_t contractTasks = 0;
  int64_t eligibleFoldMapEdges = 0;
  std::vector<int64_t> fusedMapGroupSizes;
  std::vector<int64_t> fusedFoldMapGroupSizes;
  bool exactSourceCoverage = false;
  std::string fingerprint;
};

int64_t staticElementCount(mlir::RankedTensorType type) {
  int64_t count = 1;
  for (int64_t extent : type.getShape())
    count *= extent;
  return count;
}

bool isOneToOneFoldMapEdge(mlir::shuttle::FoldOp fold,
                           mlir::shuttle::MapOp map,
                           unsigned operandNumber) {
  if (fold.getNumResults() != 1 || !fold.getResult(0).hasOneUse() ||
      fold.getInputs().size() != 1 || fold.getInitializers().size() != 1 ||
      !fold.getOrderFree() ||
      map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != map.getInputs().size() + 1)
    return false;
  auto foldInput = mlir::dyn_cast<mlir::RankedTensorType>(
      fold.getInputs().front().getType());
  auto initializer = mlir::dyn_cast<mlir::RankedTensorType>(
      fold.getInitializers().front().getType());
  auto folded = mlir::dyn_cast<mlir::RankedTensorType>(
      fold.getResult(0).getType());
  auto output =
      mlir::dyn_cast<mlir::RankedTensorType>(map.getResult(0).getType());
  auto edge = mlir::dyn_cast<mlir::AffineMapAttr>(
      map.getIndexingMaps()[operandNumber]);
  auto resultMap = mlir::dyn_cast<mlir::AffineMapAttr>(
      map.getIndexingMaps().getValue().back());
  if (!foldInput || !initializer || !folded || !output ||
      !foldInput.hasStaticShape() || !foldInput.getElementType().isF32() ||
      initializer.getRank() != 0 || !initializer.getElementType().isF32() ||
      !folded.hasStaticShape() ||
      !output.hasStaticShape() || !folded.getElementType().isF32() ||
      !output.getElementType().isF32() || !edge || !resultMap ||
      !resultMap.getValue().isIdentity() ||
      staticElementCount(folded) != staticElementCount(output) ||
      edge.getValue().getNumDims() != output.getRank() ||
      edge.getValue().getNumSymbols() != 0 ||
      edge.getValue().getNumResults() != folded.getRank())
    return false;
  for (mlir::Type type : map.getOperandTypes()) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() ||
        !tensor.getElementType().isF32())
      return false;
  }
  llvm::SmallDenseSet<unsigned> mapped;
  std::optional<unsigned> previousDimension;
  for (auto [axis, expression] :
       llvm::enumerate(edge.getValue().getResults())) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension || !mapped.insert(dimension.getPosition()).second ||
        (previousDimension &&
         dimension.getPosition() <= *previousDimension) ||
        folded.getDimSize(axis) !=
            output.getDimSize(dimension.getPosition()))
      return false;
    previousDimension = dimension.getPosition();
  }
  return llvm::all_of(llvm::enumerate(output.getShape()),
                      [&](auto item) {
    return mapped.contains(item.index()) || item.value() == 1;
  });
}

struct FusionAudit {
  int64_t groups = 0;
  bool sawComposedProjectedInput = false;
};

bool sourcePrecedes(mlir::shuttle::SourceRefAttr left,
                    mlir::shuttle::SourceRefAttr right) {
  return std::tuple(left.getFunctionOrdinal(), left.getBlockOrdinal(),
                    left.getOperationOrdinal(), left.getResultOrdinal()) <
         std::tuple(right.getFunctionOrdinal(), right.getBlockOrdinal(),
                    right.getOperationOrdinal(), right.getResultOrdinal());
}

std::string printType(mlir::Type type) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  type.print(stream);
  return result;
}

std::vector<int64_t>
expectedFusionGroupSizes(const FixtureExpectation &fixture) {
  if (fixture.maps == 18 && fixture.folds == 1)
    return {6, 4};
  if (fixture.maps == 43 && fixture.folds == 5)
    return {5, 7, 3, 2, 6};
  if (fixture.maps == 46 && fixture.folds == 5)
    return {5, 7, 2, 2, 2, 6};
  if (fixture.maps == 29 && fixture.folds == 2)
    return {3, 2, 6, 6};
  if (fixture.maps == 63 && fixture.folds == 10)
    return {3, 2, 5, 7, 3, 2, 6, 3, 2};
  if (fixture.maps == 70 && fixture.folds == 10)
    return {3, 2, 5, 7, 2, 4, 2, 6, 3, 2};
  return {};
}

std::string options(mlir::shuttle::NumericalPolicy policy,
                    llvm::StringRef tuning) {
  const llvm::StringRef numerics =
      policy == mlir::shuttle::NumericalPolicy::Fast ? "fast"
                                                     : "source_ordered";
  return (llvm::Twine("{\"execution_mode\":\"gpu_executable_bundle\","
                      "\"numerics\":\"") +
          numerics +
          "\",\"pipeline_abi_version\":22,\"schema_version\":1,"
          "\"tuning\":" +
          tuning + "}")
      .str();
}

std::optional<std::string> fixtureSource(llvm::StringRef filename) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  if (!runfiles)
    return std::nullopt;
  std::ifstream input(runfiles->Rlocation(
      (llvm::Twine("shuttle_mlir/test/Inputs/") + filename).str()));
  std::ostringstream source;
  source << input.rdbuf();
  if (!input)
    return std::nullopt;
  return source.str();
}

void renameSymbols(mlir::ModuleOp module) {
  module.setSymName("anonymous_module");
  int64_t ordinal = 0;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
    function.setSymName((llvm::Twine("anonymous_function_") +
                         llvm::Twine(ordinal++))
                            .str());
}

std::unique_ptr<BuiltModule>
build(llvm::StringRef source, mlir::shuttle::NumericalPolicy policy,
      llvm::StringRef tuning, bool renamed, bool executable) {
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
  if (renamed)
    renameSymbols(*result->module);

  mlir::PassManager manager(result->context.get());
  manager.addPass(mlir::shuttle::createAnnotateSourcePass());
  manager.addPass(mlir::shuttle::createFormStructuralRegionsPass(
      policy, options(policy, tuning), tuning.str()));
  manager.addPass(mlir::shuttle::createConvertStablehloToAlgebraPass());
  manager.addPass(mlir::shuttle::createVerifySourceCoveragePass());
  manager.addPass(mlir::shuttle::createVerifySemanticErasurePass());
  manager.addPass(mlir::shuttle::createPlanRowFoldMaterializationPass());
  manager.addPass(mlir::shuttle::createVerifyMaterializationPlanPass());
  if (executable) {
    manager.addPass(mlir::shuttle::createPlanSimt32RowFoldSchedulePass());
    manager.addPass(mlir::shuttle::createVerifySimt32RowFoldSchedulePass());
    manager.addPass(mlir::shuttle::createBuildGpuExecutableBundlePass());
    manager.addPass(mlir::shuttle::createVerifyGpuExecutableBundlePass());
  }
  if (mlir::failed(manager.run(*result->module)))
    return {};
  return result;
}

std::unique_ptr<BuiltModule>
buildAlgebra(llvm::StringRef source, mlir::shuttle::NumericalPolicy policy) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::shuttle::ShuttleDialect>();
  auto result = std::make_unique<BuiltModule>();
  result->context = std::make_unique<mlir::MLIRContext>(registry);
  result->module =
      mlir::parseSourceString<mlir::ModuleOp>(source, result->context.get());
  if (!result->module)
    return {};
  result->module->walk([&](mlir::shuttle::RegionOp region) {
    region.setPolicy(policy);
  });
  mlir::PassManager manager(result->context.get());
  manager.addPass(mlir::shuttle::createPlanRowFoldMaterializationPass());
  manager.addPass(mlir::shuttle::createVerifyMaterializationPlanPass());
  manager.addPass(mlir::shuttle::createPlanSimt32RowFoldSchedulePass());
  manager.addPass(mlir::shuttle::createVerifySimt32RowFoldSchedulePass());
  manager.addPass(mlir::shuttle::createBuildGpuExecutableBundlePass());
  manager.addPass(mlir::shuttle::createVerifyGpuExecutableBundlePass());
  if (mlir::failed(manager.run(*result->module)))
    return {};
  return result;
}

std::optional<PlanSummary> summary(mlir::ModuleOp module) {
  PlanSummary result;
  std::vector<std::pair<mlir::shuttle::SourceRefAttr, int64_t>> fusedGroups;
  llvm::SmallDenseSet<mlir::Attribute> algebraSources;
  module.walk([&](mlir::shuttle::MapOp) { ++result.maps; });
  module.walk([&](mlir::shuttle::FoldOp) { ++result.folds; });
  module.walk([&](mlir::shuttle::ContractOp) { ++result.contracts; });
  module.walk([&](mlir::Operation *operation) {
    if (auto map = mlir::dyn_cast<mlir::shuttle::MapOp>(operation))
      algebraSources.insert(map.getSource());
    else if (auto fold = mlir::dyn_cast<mlir::shuttle::FoldOp>(operation))
      algebraSources.insert(fold.getSource());
    else if (auto contract =
                 mlir::dyn_cast<mlir::shuttle::ContractOp>(operation))
      algebraSources.insert(contract.getSource());
  });
  module.walk([&](mlir::shuttle::FoldOp fold) {
    if (fold.getNumResults() != 1 || !fold.getResult(0).hasOneUse())
      return;
    mlir::OpOperand &use = *fold.getResult(0).getUses().begin();
    auto map = mlir::dyn_cast<mlir::shuttle::MapOp>(use.getOwner());
    if (map && isOneToOneFoldMapEdge(fold, map, use.getOperandNumber()))
      ++result.eligibleFoldMapEdges;
  });
  llvm::SmallVector<mlir::shuttle::MaterializationPlanOp> plans(
      module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return std::nullopt;
  result.fingerprint = plans.front().getFingerprint().str();
  llvm::SmallDenseSet<mlir::Attribute> plannedSources;
  for (mlir::Operation &operation : plans.front().getBody().front()) {
    if (mlir::isa<mlir::shuttle::MaterializationBufferOp>(operation)) {
      ++result.buffers;
      continue;
    }
    auto task = mlir::dyn_cast<mlir::shuttle::MaterializationTaskOp>(operation);
    if (!task)
      continue;
    auto fusedSources = task->getAttrOfType<mlir::ArrayAttr>("fused_sources");
    auto foldEpilogueSources =
        task->getAttrOfType<mlir::ArrayAttr>("fold_epilogue_sources");
    if (fusedSources && foldEpilogueSources)
      return std::nullopt;
    if (foldEpilogueSources) {
      if (task.getKind() != mlir::shuttle::MaterializationTaskKind::Fold ||
          foldEpilogueSources.empty() ||
          !plannedSources.insert(task.getSource()).second)
        return std::nullopt;
      auto firstSource = mlir::dyn_cast<mlir::shuttle::SourceRefAttr>(
          foldEpilogueSources.getValue().front());
      if (!firstSource)
        return std::nullopt;
      if (foldEpilogueSources.size() > 1)
        fusedGroups.emplace_back(firstSource, foldEpilogueSources.size());
      result.fusedFoldMapGroupSizes.push_back(foldEpilogueSources.size());
      for (mlir::Attribute source : foldEpilogueSources) {
        if (!mlir::isa<mlir::shuttle::SourceRefAttr>(source) ||
            !plannedSources.insert(source).second)
          return std::nullopt;
      }
    } else if (fusedSources) {
      if (fusedSources.size() < 2 ||
          fusedSources.getValue().back() != task.getSource())
        return std::nullopt;
      auto firstSource = mlir::dyn_cast<mlir::shuttle::SourceRefAttr>(
          fusedSources.getValue().front());
      if (!firstSource)
        return std::nullopt;
      fusedGroups.emplace_back(firstSource, fusedSources.size());
      for (mlir::Attribute source : fusedSources) {
        if (!mlir::isa<mlir::shuttle::SourceRefAttr>(source) ||
            !plannedSources.insert(source).second)
          return std::nullopt;
      }
    } else if (!plannedSources.insert(task.getSource()).second) {
      return std::nullopt;
    }
    ++result.tasks;
    switch (task.getKind()) {
    case mlir::shuttle::MaterializationTaskKind::Map:
      ++result.mapTasks;
      break;
    case mlir::shuttle::MaterializationTaskKind::Fold:
      ++result.foldTasks;
      break;
    case mlir::shuttle::MaterializationTaskKind::Contract:
      ++result.contractTasks;
      break;
    }
  }
  result.exactSourceCoverage =
      plannedSources.size() == algebraSources.size() &&
      llvm::all_of(plannedSources, [&](mlir::Attribute source) {
        return algebraSources.contains(source);
      });
  llvm::sort(fusedGroups, [](const auto &left, const auto &right) {
    return sourcePrecedes(left.first, right.first);
  });
  for (const auto &group : fusedGroups)
    result.fusedMapGroupSizes.push_back(group.second);
  return result;
}

std::optional<llvm::SmallVector<int64_t>> mapDomainShape(
    mlir::shuttle::MapOp map) {
  if (map.getNumResults() != 1)
    return std::nullopt;
  auto result = mlir::dyn_cast<mlir::RankedTensorType>(map.getResult(0).getType());
  if (!result || !result.hasStaticShape())
    return std::nullopt;
  auto resultMap = mlir::cast<mlir::AffineMapAttr>(
                       map.getIndexingMaps().getValue().back())
                       .getValue();
  if (!resultMap.isIdentity())
    return std::nullopt;
  return llvm::SmallVector<int64_t>(result.getShape());
}

std::optional<FusionAudit> auditFusionGroups(mlir::ModuleOp module) {
  llvm::SmallDenseMap<mlir::Attribute, mlir::shuttle::MapOp> maps;
  module.walk([&](mlir::shuttle::MapOp map) {
    maps.try_emplace(map.getSource(), map);
  });
  llvm::SmallVector<mlir::shuttle::MaterializationPlanOp> plans(
      module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return std::nullopt;

  FusionAudit audit;
  for (mlir::shuttle::MaterializationTaskOp task :
       plans.front().getBody().front().getOps<
           mlir::shuttle::MaterializationTaskOp>()) {
    auto sources = task->getAttrOfType<mlir::ArrayAttr>("fused_sources");
    if (!sources)
      continue;
    ++audit.groups;
    llvm::SmallDenseSet<mlir::Attribute> group;
    std::optional<mlir::shuttle::SourceRefAttr> previousSource;
    for (mlir::Attribute source : sources) {
      auto sourceRef = mlir::dyn_cast<mlir::shuttle::SourceRefAttr>(source);
      auto found = maps.find(source);
      if (!sourceRef || found == maps.end() || !group.insert(source).second ||
          (previousSource && !sourcePrecedes(*previousSource, sourceRef)))
        return std::nullopt;
      previousSource = sourceRef;
      mlir::shuttle::MapOp map = found->second;
      std::optional<llvm::SmallVector<int64_t>> domain = mapDomainShape(map);
      if (!domain)
        return std::nullopt;
      llvm::SmallVector<mlir::Type> boundaryTypes(map.getOperandTypes());
      llvm::append_range(boundaryTypes, map.getResultTypes());
      for (mlir::Type type : boundaryTypes) {
        auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
        if (!tensor || !tensor.hasStaticShape() ||
            !tensor.getElementType().isF32())
          return std::nullopt;
      }
    }
    for (mlir::Attribute source : sources.getValue().drop_back()) {
      mlir::shuttle::MapOp producer = maps.lookup(source);
      if (producer.getNumResults() != 1 || !producer.getResult(0).hasOneUse())
        return std::nullopt;
      mlir::OpOperand &edge = *producer.getResult(0).getUses().begin();
      auto consumer = mlir::dyn_cast<mlir::shuttle::MapOp>(edge.getOwner());
      if (!consumer || !group.contains(consumer.getSource()))
        return std::nullopt;
      auto edgeMap = mlir::cast<mlir::AffineMapAttr>(
                         consumer.getIndexingMaps()[edge.getOperandNumber()])
                         .getValue();
      if (!edgeMap.isIdentity() ||
          mapDomainShape(producer) != mapDomainShape(consumer))
        return std::nullopt;
      for (auto [operand, indexing] : llvm::zip_equal(
               producer.getInputs(), producer.getIndexingMaps().getValue().take_front(
                                         producer.getInputs().size()))) {
        auto inputMap = mlir::cast<mlir::AffineMapAttr>(indexing).getValue();
        const bool projected = llvm::any_of(
            inputMap.getResults(), [](mlir::AffineExpr expression) {
              return expression.getKind() == mlir::AffineExprKind::FloorDiv;
            });
        auto inputProducer = operand.getDefiningOp<mlir::shuttle::MapOp>();
        if (projected &&
            (!inputProducer || !group.contains(inputProducer.getSource())))
          audit.sawComposedProjectedInput = true;
      }
    }
  }
  return audit;
}

bool mutateFusedTask(mlir::ModuleOp module, llvm::StringRef mutation) {
  llvm::SmallVector<mlir::shuttle::MaterializationPlanOp> plans(
      module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return false;
  for (mlir::shuttle::MaterializationTaskOp task :
       plans.front()
           .getBody()
           .front()
           .getOps<mlir::shuttle::MaterializationTaskOp>()) {
    auto sources = task->getAttrOfType<mlir::ArrayAttr>("fused_sources");
    auto maps = task->getAttrOfType<mlir::ArrayAttr>("composed_input_maps");
    if (!sources || sources.size() < 2 || !maps || maps.size() < 2)
      continue;
    mlir::Builder builder(module.getContext());
    llvm::SmallVector<mlir::Attribute> changed;
    if (mutation == "duplicate-source") {
      llvm::append_range(changed, sources);
      changed[1] = changed[0];
      task->setAttr("fused_sources", builder.getArrayAttr(changed));
    } else if (mutation == "reordered-source") {
      llvm::append_range(changed, llvm::reverse(sources.getValue()));
      task->setAttr("fused_sources", builder.getArrayAttr(changed));
    } else if (mutation == "stale-source") {
      llvm::append_range(changed, sources);
      changed[0] = mlir::shuttle::SourceRefAttr::get(
          module.getContext(), 99, 99, 99, 99);
      task->setAttr("fused_sources", builder.getArrayAttr(changed));
    } else if (mutation == "missing-map") {
      task->setAttr("composed_input_maps",
                    builder.getArrayAttr(maps.getValue().drop_back()));
    } else if (mutation == "reordered-map") {
      llvm::append_range(changed, llvm::reverse(maps.getValue()));
      task->setAttr("composed_input_maps", builder.getArrayAttr(changed));
    } else {
      return false;
    }
    plans.front().setFingerprint(
        mlir::shuttle::materializationPlanFingerprint(plans.front()));
    return true;
  }
  return false;
}

bool mutateFoldEpilogueTask(mlir::ModuleOp module,
                            llvm::StringRef mutation) {
  llvm::SmallVector<mlir::shuttle::MaterializationPlanOp> plans(
      module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return false;
  for (mlir::shuttle::MaterializationTaskOp task :
       plans.front()
           .getBody()
           .front()
           .getOps<mlir::shuttle::MaterializationTaskOp>()) {
    auto sources =
        task->getAttrOfType<mlir::ArrayAttr>("fold_epilogue_sources");
    auto maps = task->getAttrOfType<mlir::ArrayAttr>("epilogue_input_maps");
    if (!sources || sources.size() < 2 || !maps || maps.size() < 2)
      continue;
    mlir::Builder builder(module.getContext());
    llvm::SmallVector<mlir::Attribute> changed;
    if (mutation == "duplicate-source") {
      llvm::append_range(changed, sources);
      changed[1] = changed[0];
      task->setAttr("fold_epilogue_sources", builder.getArrayAttr(changed));
    } else if (mutation == "reordered-source") {
      llvm::append_range(changed, llvm::reverse(sources.getValue()));
      task->setAttr("fold_epilogue_sources", builder.getArrayAttr(changed));
    } else if (mutation == "stale-source") {
      llvm::append_range(changed, sources);
      changed[0] = mlir::shuttle::SourceRefAttr::get(
          module.getContext(), 99, 99, 99, 99);
      task->setAttr("fold_epilogue_sources", builder.getArrayAttr(changed));
    } else if (mutation == "missing-map") {
      task->setAttr("epilogue_input_maps",
                    builder.getArrayAttr(maps.getValue().drop_back()));
    } else if (mutation == "reordered-map") {
      llvm::append_range(changed, llvm::reverse(maps.getValue()));
      if (maps.getValue() == llvm::ArrayRef<mlir::Attribute>(changed))
        continue;
      task->setAttr("epilogue_input_maps", builder.getArrayAttr(changed));
    } else {
      return false;
    }
    plans.front().setFingerprint(
        mlir::shuttle::materializationPlanFingerprint(plans.front()));
    return true;
  }
  return false;
}

bool verifyMaterializationPlan(mlir::ModuleOp module) {
  mlir::PassManager manager(module.getContext());
  manager.addPass(mlir::shuttle::createVerifyMaterializationPlanPass());
  return mlir::succeeded(manager.run(module));
}

void expectCounts(const PlanSummary &actual,
                  const FixtureExpectation &expected, bool fused) {
  EXPECT_EQ(actual.maps, expected.maps);
  EXPECT_EQ(actual.folds, expected.folds);
  EXPECT_EQ(actual.contracts, 0);
  EXPECT_EQ(actual.tasks,
            fused ? expected.fusedTasks - actual.eligibleFoldMapEdges
                  : expected.materializedTasks);
  EXPECT_EQ(actual.buffers,
            fused ? expected.fusedBuffers - actual.eligibleFoldMapEdges
                  : expected.materializedBuffers);
  EXPECT_EQ(actual.foldTasks, expected.folds);
  EXPECT_EQ(actual.contractTasks, 0);
  EXPECT_EQ(actual.mapTasks, actual.tasks - actual.foldTasks);
  EXPECT_TRUE(actual.exactSourceCoverage);
  EXPECT_EQ(actual.fusedMapGroupSizes,
            fused ? expectedFusionGroupSizes(expected)
                  : std::vector<int64_t>{});
  EXPECT_EQ(actual.fusedFoldMapGroupSizes.size(),
            fused ? actual.eligibleFoldMapEdges : 0);
}

TEST(Abi22MapFusionContract,
     PinnedOrdinaryJaxMatrixBindsStructuralTaskAndBufferReductions) {
  constexpr mlir::shuttle::NumericalPolicy policies[] = {
      mlir::shuttle::NumericalPolicy::SourceOrdered,
      mlir::shuttle::NumericalPolicy::Fast,
  };
  for (const FixtureExpectation &fixture : kFixtures) {
    SCOPED_TRACE(fixture.filename);
    std::optional<std::string> source = fixtureSource(fixture.filename);
    ASSERT_TRUE(source);
    std::optional<PlanSummary> sourceOrderedFusion;
    for (mlir::shuttle::NumericalPolicy policy : policies) {
      SCOPED_TRACE(policy == mlir::shuttle::NumericalPolicy::Fast ? "fast"
                                                                 : "source");
      std::unique_ptr<BuiltModule> automatic =
          build(*source, policy, kAutomaticTuning, false, false);
      ASSERT_TRUE(automatic);
      std::optional<PlanSummary> automaticSummary =
          summary(*automatic->module);
      ASSERT_TRUE(automaticSummary);
      expectCounts(*automaticSummary, fixture, false);

      std::unique_ptr<BuiltModule> materialized =
          build(*source, policy, kMaterializationTuning, false, false);
      ASSERT_TRUE(materialized);
      std::optional<PlanSummary> materializedSummary =
          summary(*materialized->module);
      ASSERT_TRUE(materializedSummary);
      expectCounts(*materializedSummary, fixture, false);

      std::unique_ptr<BuiltModule> fused =
          build(*source, policy, kFusionTuning, false, false);
      ASSERT_TRUE(fused);
      std::optional<PlanSummary> fusedSummary = summary(*fused->module);
      ASSERT_TRUE(fusedSummary);
      expectCounts(*fusedSummary, fixture, true);
      std::optional<FusionAudit> audit = auditFusionGroups(*fused->module);
      ASSERT_TRUE(audit);
      const int64_t epilogueMapGroups = llvm::count_if(
          fusedSummary->fusedFoldMapGroupSizes,
          [](int64_t size) { return size > 1; });
      EXPECT_EQ(audit->groups + epilogueMapGroups,
                static_cast<int64_t>(expectedFusionGroupSizes(fixture).size()));
      if (llvm::StringRef(fixture.filename).contains(
              "projected_parameter_broadcast"))
        EXPECT_TRUE(audit->sawComposedProjectedInput);
      EXPECT_EQ(fixture.materializedTasks - fixture.fusedTasks,
                fixture.materializedBuffers - fixture.fusedBuffers);

      std::unique_ptr<BuiltModule> renamed =
          build(*source, policy, kFusionTuning, true, false);
      ASSERT_TRUE(renamed);
      std::optional<PlanSummary> renamedSummary = summary(*renamed->module);
      ASSERT_TRUE(renamedSummary);
      expectCounts(*renamedSummary, fixture, true);
      EXPECT_EQ(renamedSummary->fingerprint, fusedSummary->fingerprint);

      if (sourceOrderedFusion) {
        EXPECT_EQ(sourceOrderedFusion->tasks, fusedSummary->tasks);
        EXPECT_EQ(sourceOrderedFusion->buffers, fusedSummary->buffers);
        EXPECT_NE(sourceOrderedFusion->fingerprint,
                  fusedSummary->fingerprint);
      } else {
        sourceOrderedFusion = *fusedSummary;
      }
    }
  }
}

TEST(Abi22MapFusionContract,
     FusedPhysicalTasksBindPtxEntriesAndTransportSlots) {
  constexpr std::array<FixtureExpectation, 2> representatives = {
      kFixtures[0], kFixtures[6]};
  constexpr mlir::shuttle::NumericalPolicy policies[] = {
      mlir::shuttle::NumericalPolicy::SourceOrdered,
      mlir::shuttle::NumericalPolicy::Fast,
  };
  for (const FixtureExpectation &fixture : representatives) {
    SCOPED_TRACE(fixture.filename);
    std::optional<std::string> source = fixtureSource(fixture.filename);
    ASSERT_TRUE(source);
    for (mlir::shuttle::NumericalPolicy policy : policies) {
      std::unique_ptr<BuiltModule> built =
          build(*source, policy, kFusionTuning, false, true);
      ASSERT_TRUE(built);
      std::optional<PlanSummary> plan = summary(*built->module);
      ASSERT_TRUE(plan);
      expectCounts(*plan, fixture, true);

      llvm::SmallVector<mlir::shuttle::DeviceModuleOp> devices(
          built->module->getOps<mlir::shuttle::DeviceModuleOp>());
      llvm::SmallVector<mlir::shuttle::InvocationAbiOp> invocationAbis(
          built->module->getOps<mlir::shuttle::InvocationAbiOp>());
      llvm::SmallVector<mlir::shuttle::ExecutableBundleOp> bundles(
          built->module->getOps<mlir::shuttle::ExecutableBundleOp>());
      ASSERT_EQ(devices.size(), 1);
      ASSERT_EQ(invocationAbis.size(), 1);
      ASSERT_EQ(bundles.size(), 1);
      EXPECT_FALSE(devices.front().getCode().empty());
      EXPECT_EQ(llvm::range_size(
                    devices.front().getBody().front().getOps<
                        mlir::shuttle::DeviceEntryOp>()),
                fixture.fusedTasks - plan->eligibleFoldMapEdges);
      EXPECT_EQ(llvm::range_size(
                    invocationAbis.front().getBody().front().getOps<
                        mlir::shuttle::InvocationSlotOp>()),
                fixture.fusedBuffers - plan->eligibleFoldMapEdges);
      for (mlir::shuttle::DeviceEntryOp entry :
           devices.front().getBody().front().getOps<
               mlir::shuttle::DeviceEntryOp>()) {
        EXPECT_GT(entry.getCodeLength(), 0);
      }

      mlir::FailureOr<llvm::SmallVector<uint8_t>> transport =
          mlir::shuttle::serializeGpuExecutableBundle(*built->module);
      ASSERT_TRUE(mlir::succeeded(transport));
      auto decoded = mlir::shuttle::GpuExecutable::Load(*transport);
      ASSERT_TRUE(decoded.ok()) << decoded.status();
      EXPECT_EQ((*decoded)->codeBytes().size(), devices.front().getCode().size());
      EXPECT_EQ((*decoded)->launches().size(),
                fixture.fusedTasks - plan->eligibleFoldMapEdges);
      EXPECT_EQ((*decoded)->slots().size(),
                fixture.fusedBuffers - plan->eligibleFoldMapEdges);

      auto entries = llvm::to_vector(devices.front()
                                         .getBody()
                                         .front()
                                         .getOps<mlir::shuttle::DeviceEntryOp>());
      for (auto [ordinal, entry, launch] :
           llvm::enumerate(entries, (*decoded)->launches())) {
        EXPECT_EQ(launch.taskOrdinal, ordinal);
        EXPECT_EQ(launch.codeOffset, entry.getCodeOffset());
        EXPECT_EQ(launch.codeLength, entry.getCodeLength());
        EXPECT_EQ(launch.codeDigest, entry.getCodeDigest());
        EXPECT_EQ(launch.inputSlots,
                  std::vector<int64_t>(entry.getInputBuffers()));
        EXPECT_EQ(launch.outputSlots,
                  std::vector<int64_t>(entry.getOutputBuffers()));
        EXPECT_EQ(launch.dependencies,
                  std::vector<int64_t>(entry.getDependencies()));
        ASSERT_TRUE(entry.getGrid());
        ASSERT_TRUE(entry.getBlock());
        EXPECT_EQ(launch.grid,
                  (std::array<uint64_t, 3>{
                      static_cast<uint64_t>((*entry.getGrid())[0]),
                      static_cast<uint64_t>((*entry.getGrid())[1]),
                      static_cast<uint64_t>((*entry.getGrid())[2])}));
        EXPECT_EQ(launch.block,
                  (std::array<uint64_t, 3>{
                      static_cast<uint64_t>((*entry.getBlock())[0]),
                      static_cast<uint64_t>((*entry.getBlock())[1]),
                      static_cast<uint64_t>((*entry.getBlock())[2])}));
        EXPECT_EQ(launch.dynamicSharedMemoryBytes,
                  static_cast<uint64_t>(*entry.getDynamicSharedBytes()));
      }

      auto slots = llvm::to_vector(
          invocationAbis.front()
              .getBody()
              .front()
              .getOps<mlir::shuttle::InvocationSlotOp>());
      for (auto [ordinal, slot, decodedSlot] :
           llvm::enumerate(slots, (*decoded)->slots())) {
        EXPECT_EQ(decodedSlot.ordinal, ordinal);
        EXPECT_EQ(printType(decodedSlot.tensorType),
                  printType(slot.getTensorType()));
        EXPECT_EQ(decodedSlot.requiredBytes, slot.getRequiredBytes());
        EXPECT_EQ(decodedSlot.byteStrides,
                  llvm::SmallVector<int64_t>(slot.getStrides()));
        EXPECT_EQ(decodedSlot.byteOffset, slot.getOffset());
        EXPECT_EQ(decodedSlot.alignment, slot.getAlignment());
        EXPECT_EQ(decodedSlot.addressSpace, slot.getAddressSpace());
        EXPECT_EQ(decodedSlot.access, slot.getAccess());
        EXPECT_EQ(decodedSlot.storage, slot.getStorage());
        EXPECT_EQ(decodedSlot.aliasGroup, slot.getAliasGroup());
        EXPECT_EQ(decodedSlot.reuseGroup, slot.getReuseGroup());
        EXPECT_EQ(decodedSlot.bindingKind, slot.getBinding());
        EXPECT_EQ(decodedSlot.bindingIndex, slot.getBindingIndex().value_or(-1));
      }
    }
  }
}

TEST(Abi22MapFusionContract,
     RejectsCorruptOrderedProvenanceAndComposedInputMaps) {
  std::optional<std::string> source = fixtureSource(kFixtures[0].filename);
  std::optional<std::string> foldSource =
      fixtureSource(kFixtures[1].filename);
  ASSERT_TRUE(source);
  ASSERT_TRUE(foldSource);
  constexpr llvm::StringLiteral mutations[] = {
      "duplicate-source", "reordered-source", "stale-source", "missing-map",
      "reordered-map"};
  for (llvm::StringRef mutation : mutations) {
    for (bool foldEpilogue : {false, true}) {
      SCOPED_TRACE((llvm::Twine(foldEpilogue ? "Fold epilogue " : "Map ") +
                    mutation)
                       .str());
      std::unique_ptr<BuiltModule> built = build(
          foldEpilogue ? *foldSource : *source,
          mlir::shuttle::NumericalPolicy::SourceOrdered,
          kFusionTuning, false, false);
      ASSERT_TRUE(built);
      ASSERT_TRUE(foldEpilogue
                      ? mutateFoldEpilogueTask(*built->module, mutation)
                      : mutateFusedTask(*built->module, mutation));
      EXPECT_FALSE(verifyMaterializationPlan(*built->module));
    }
  }
}

TEST(Abi22MapFusionContract,
     FusedPtxPreservesDependentScalarOrderAndFingerprintsEveryOperation) {
  std::optional<std::string> source =
      fixtureSource("abi22-map-fusion-source-order.mlir");
  ASSERT_TRUE(source);
  for (mlir::shuttle::NumericalPolicy policy :
       {mlir::shuttle::NumericalPolicy::SourceOrdered,
        mlir::shuttle::NumericalPolicy::Fast}) {
    std::unique_ptr<BuiltModule> built = buildAlgebra(*source, policy);
    ASSERT_TRUE(built);
    std::optional<PlanSummary> originalSummary = summary(*built->module);
    ASSERT_TRUE(originalSummary);
    EXPECT_EQ(originalSummary->tasks, 4);
    EXPECT_EQ(originalSummary->buffers, 6);
    EXPECT_EQ(originalSummary->fusedMapGroupSizes,
              std::vector<int64_t>({2}));
    llvm::SmallVector<mlir::shuttle::DeviceModuleOp> devices(
        built->module->getOps<mlir::shuttle::DeviceModuleOp>());
    ASSERT_EQ(devices.size(), 1);
    llvm::StringRef ptx(
        reinterpret_cast<const char *>(devices.front().getCode().data()),
        devices.front().getCode().size());
    size_t subtract = ptx.find("sub.rn.f32");
    size_t divide = ptx.find("div.rn.f32");
    ASSERT_NE(subtract, llvm::StringRef::npos);
    ASSERT_NE(divide, llvm::StringRef::npos);
    EXPECT_LT(subtract, divide);

    std::string changed = *source;
    size_t operation = changed.find("arith.subf");
    ASSERT_NE(operation, std::string::npos);
    changed.replace(operation, std::string("arith.subf").size(), "arith.addf");
    std::unique_ptr<BuiltModule> mutated = buildAlgebra(changed, policy);
    ASSERT_TRUE(mutated);
    std::optional<PlanSummary> mutatedSummary = summary(*mutated->module);
    ASSERT_TRUE(mutatedSummary);
    EXPECT_NE(originalSummary->fingerprint, mutatedSummary->fingerprint);
  }
}

TEST(Abi22MapFusionContract, RejectsPermutedFoldResultEdge) {
  constexpr llvm::StringLiteral source = R"mlir(
#input = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
#output = affine_map<(d0, d1) -> (d0, d1)>
#scalar = affine_map<() -> ()>
module attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%input: tensor<2x3x4xf32>) -> tensor<3x2xf32> {
    %result = "shuttle.region"(%input) ({
    ^bb0(%region_input: tensor<2x3x4xf32>):
      %init = "shuttle.map"() ({
      ^bb0:
        %zero = arith.constant 0.000000e+00 : f32
        "shuttle.yield"(%zero) : (f32) -> ()
      }) {indexing_maps = [#scalar], source = #shuttle.source_ref<0, 0, 0, 0>} : () -> tensor<f32>
      %folded = "shuttle.fold"(%region_input, %init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 2>, source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x3xf32>
      %transposed = "shuttle.map"(%folded) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {indexing_maps = [#transpose, #output], source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
      "shuttle.yield"(%transposed) : (tensor<3x2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>]} : (tensor<2x3x4xf32>) -> tensor<3x2xf32>
    return %result : tensor<3x2xf32>
  }
}
)mlir";
  std::unique_ptr<BuiltModule> built =
      buildAlgebra(source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(built);
  std::optional<PlanSummary> plan = summary(*built->module);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->tasks, 3);
  EXPECT_EQ(plan->buffers, 4);
  EXPECT_TRUE(plan->fusedFoldMapGroupSizes.empty());
}

} // namespace
