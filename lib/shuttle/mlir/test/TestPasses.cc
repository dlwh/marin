// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "TestPasses.h"

#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleOps.h"
#include "shuttle/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::shuttle::test {
namespace {

constexpr llvm::StringLiteral kManifest = "shuttle.coverage_manifest";
constexpr llvm::StringLiteral kSourceRefs = "shuttle.source_refs";
constexpr llvm::StringLiteral kOperationRef = "shuttle.operation_ref";
constexpr llvm::StringLiteral kFastFusionTuning =
    R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
constexpr llvm::StringLiteral kFastFusionOptions =
    R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr llvm::StringLiteral kFastFusionTuningDigest =
    "528d87798df49d74838ca85396578213dc949a9d7ad76d9db7b8be4d0b64868a";
constexpr llvm::StringLiteral kFastFusionPolicyDigest =
    "10de636a98b8aef9f56270c33f288459fd3bf64d9206daf809ccbf8f04425bec";

llvm::SmallDenseSet<Attribute> excludedSources(ModuleOp module) {
  llvm::SmallDenseSet<Attribute> sources;
  auto manifest = module->getAttrOfType<DictionaryAttr>(kManifest);
  auto excluded =
      manifest ? manifest.getAs<ArrayAttr>("excluded") : ArrayAttr{};
  if (!excluded) {
    return sources;
  }
  for (Attribute attribute : excluded) {
    if (auto record = dyn_cast<DictionaryAttr>(attribute)) {
      if (Attribute source = record.get("source")) {
        sources.insert(source);
      }
    }
  }
  return sources;
}

bool isExcluded(Operation *operation,
                const llvm::SmallDenseSet<Attribute> &excluded) {
  auto refs = operation->getAttrOfType<ArrayAttr>(kSourceRefs);
  return refs && llvm::any_of(refs, [&](Attribute ref) {
           return excluded.contains(ref);
         });
}

template <typename Derived>
class MutationPass : public PassWrapper<Derived, OperationPass<ModuleOp>> {};

class ExcludedKindPass : public MutationPass<ExcludedKindPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExcludedKindPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-excluded-kind";
  }
  StringRef getDescription() const final {
    return "Mutate one excluded binary operation kind";
  }
  void runOnOperation() override {
    auto excluded = excludedSources(getOperation());
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      for (Operation &operation :
           llvm::make_early_inc_range(function.getBody().front())) {
        if (!isExcluded(&operation, excluded) ||
            operation.getNumOperands() != 2 || operation.getNumResults() != 1 ||
            operation.getOperand(0).getType() !=
                operation.getOperand(1).getType()) {
          continue;
        }
        OpBuilder builder(&operation);
        OperationState state(operation.getLoc(),
                             stablehlo::AddOp::getOperationName());
        state.addOperands(operation.getOperands());
        state.addTypes(operation.getResultTypes());
        state.addAttribute(kSourceRefs, operation.getAttr(kSourceRefs));
        Operation *replacement = builder.create(state);
        operation.replaceAllUsesWith(replacement);
        operation.erase();
        return;
      }
    }
    getOperation().emitError("test fixture has no excluded binary operation");
    this->signalPassFailure();
  }
};

class ExcludedAttributePass : public MutationPass<ExcludedAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExcludedAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-excluded-attribute";
  }
  StringRef getDescription() const final {
    return "Mutate one excluded operation attribute";
  }
  void runOnOperation() override {
    auto excluded = excludedSources(getOperation());
    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (!isExcluded(operation, excluded)) {
        return WalkResult::advance();
      }
      operation->setAttr("stablehlo.mutated_test",
                         UnitAttr::get(operation->getContext()));
      return WalkResult::interrupt();
    });
    if (!result.wasInterrupted()) {
      getOperation().emitError("test fixture has no excluded operation");
      this->signalPassFailure();
    }
  }
};

class NestedExcludedAttributePass
    : public MutationPass<NestedExcludedAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NestedExcludedAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-nested-excluded-attribute";
  }
  StringRef getDescription() const final {
    return "Mutate one excluded operation nested in a source region";
  }
  void runOnOperation() override {
    auto excluded = excludedSources(getOperation());
    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (!isExcluded(operation, excluded) ||
          isa<func::FuncOp>(operation->getParentOp())) {
        return WalkResult::advance();
      }
      operation->setAttr("stablehlo.mutated_nested_test",
                         UnitAttr::get(operation->getContext()));
      return WalkResult::interrupt();
    });
    if (!result.wasInterrupted()) {
      getOperation().emitError("test fixture has no nested excluded operation");
      this->signalPassFailure();
    }
  }
};

class ExcludedOperandPass : public MutationPass<ExcludedOperandPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExcludedOperandPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-excluded-operand";
  }
  StringRef getDescription() const final {
    return "Rewire one excluded operation operand";
  }
  void runOnOperation() override {
    auto excluded = excludedSources(getOperation());
    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (!isExcluded(operation, excluded) || operation->getNumOperands() < 2 ||
          operation->getOperand(0).getType() !=
              operation->getOperand(1).getType()) {
        return WalkResult::advance();
      }
      Value first = operation->getOperand(0);
      operation->setOperand(0, operation->getOperand(1));
      operation->setOperand(1, first);
      return WalkResult::interrupt();
    });
    if (!result.wasInterrupted()) {
      getOperation().emitError(
          "test fixture has no excluded same-type binary operation");
      this->signalPassFailure();
    }
  }
};

class UnsupportedAbsorptionPass
    : public MutationPass<UnsupportedAbsorptionPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnsupportedAbsorptionPass)
  StringRef getArgument() const final {
    return "shuttle-test-absorb-excluded-source";
  }
  StringRef getDescription() const final {
    return "Add one excluded source to a selected region";
  }
  void runOnOperation() override {
    auto excluded = excludedSources(getOperation());
    RegionOp region;
    getOperation().walk([&](RegionOp candidate) {
      if (!region) {
        region = candidate;
      }
    });
    if (!region || excluded.empty()) {
      getOperation().emitError("test fixture has no region or excluded source");
      this->signalPassFailure();
      return;
    }
    SmallVector<Attribute> refs(region.getSourceRefs().begin(),
                                region.getSourceRefs().end());
    refs.push_back(*excluded.begin());
    region->setAttr("source_refs",
                    ArrayAttr::get(getOperation().getContext(), refs));
  }
};

class ReturnRewirePass : public MutationPass<ReturnRewirePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReturnRewirePass)
  StringRef getArgument() const final { return "shuttle-test-rewire-return"; }
  StringRef getDescription() const final {
    return "Rewire one return to a same-type function argument";
  }
  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      auto returnOp =
          dyn_cast<func::ReturnOp>(function.getBody().front().back());
      if (!returnOp) {
        continue;
      }
      for (OpOperand &returned : returnOp->getOpOperands()) {
        for (BlockArgument argument : function.getArguments()) {
          if (returned.get().getType() == argument.getType() &&
              returned.get() != argument) {
            returned.set(argument);
            return;
          }
        }
      }
    }
    getOperation().emitError(
        "test fixture has no same-type return alternative");
    this->signalPassFailure();
  }
};

class ManifestDigestPass : public MutationPass<ManifestDigestPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ManifestDigestPass)
  StringRef getArgument() const final {
    return "shuttle-test-corrupt-policy-digest";
  }
  StringRef getDescription() const final {
    return "Corrupt the policy digest in a coverage manifest";
  }
  void runOnOperation() override {
    auto manifest = getOperation()->getAttrOfType<DictionaryAttr>(kManifest);
    if (!manifest) {
      getOperation().emitError("test fixture has no coverage manifest");
      this->signalPassFailure();
      return;
    }
    NamedAttrList fields(manifest);
    fields.set("policy_digest",
               StringAttr::get(getOperation().getContext(), "corrupted"));
    getOperation()->setAttr(
        kManifest, DictionaryAttr::get(getOperation().getContext(), fields));
  }
};

template <typename Derived>
class FoldMutationPass : public MutationPass<Derived> {
protected:
  FoldOp fold() {
    FoldOp result;
    this->getOperation().walk([&](FoldOp candidate) {
      if (!result) {
        result = candidate;
      }
    });
    if (!result) {
      this->getOperation().emitError("test fixture has no shuttle.fold");
      this->signalPassFailure();
    }
    return result;
  }
};

class FoldOwnerPass : public FoldMutationPass<FoldOwnerPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldOwnerPass)
  StringRef getArgument() const final {
    return "shuttle-test-remove-fold-owner-ref";
  }
  void runOnOperation() override {
    if (FoldOp operation = fold()) {
      operation->removeAttr(kOperationRef);
    }
  }
};

class FoldOwnerMismatchPass : public FoldMutationPass<FoldOwnerMismatchPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldOwnerMismatchPass)
  StringRef getArgument() const final {
    return "shuttle-test-mismatch-fold-owner-ref";
  }
  void runOnOperation() override {
    if (FoldOp operation = fold()) {
      operation->setAttr(kOperationRef, DenseI64ArrayAttr::get(
                                            operation.getContext(), {9, 9, 9}));
    }
  }
};

class FoldAddSourcePass : public FoldMutationPass<FoldAddSourcePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddSourcePass)
  StringRef getArgument() const final {
    return "shuttle-test-remove-fold-add-source";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().walk(
          [&](arith::AddFOp add) { add->removeAttr(kSourceRefs); });
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldAddOwnerMismatchPass
    : public FoldMutationPass<FoldAddOwnerMismatchPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddOwnerMismatchPass)
  StringRef getArgument() const final {
    return "shuttle-test-mismatch-fold-add-owner-ref";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().walk([&](arith::AddFOp add) {
        add->setAttr(kOperationRef,
                     DenseI64ArrayAttr::get(add.getContext(), {8, 8, 8}));
      });
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldAddOwnerMissingPass
    : public FoldMutationPass<FoldAddOwnerMissingPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddOwnerMissingPass)
  StringRef getArgument() const final {
    return "shuttle-test-remove-fold-add-owner-ref";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().walk(
          [&](arith::AddFOp add) { add->removeAttr(kOperationRef); });
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldAddDuplicateSourcePass
    : public FoldMutationPass<FoldAddDuplicateSourcePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddDuplicateSourcePass)
  StringRef getArgument() const final {
    return "shuttle-test-duplicate-fold-add-source";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().walk([&](arith::AddFOp add) {
        add->setAttr(kSourceRefs,
                     ArrayAttr::get(add.getContext(), {operation.getSource()}));
      });
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldYieldOwnerPass : public FoldMutationPass<FoldYieldOwnerPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldYieldOwnerPass)
  StringRef getArgument() const final {
    return "shuttle-test-remove-fold-yield-ref";
  }
  void runOnOperation() override {
    if (FoldOp operation = fold()) {
      operation.getCombiner().front().getTerminator()->removeAttr(
          kOperationRef);
    }
  }
};

class FoldOwnerDuplicatePass : public FoldMutationPass<FoldOwnerDuplicatePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldOwnerDuplicatePass)
  StringRef getArgument() const final {
    return "shuttle-test-duplicate-fold-owner-ref";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (!operation) {
      return;
    }
    Attribute owner = operation->getAttr(kOperationRef);
    operation.getCombiner().walk(
        [&](arith::AddFOp add) { add->setAttr(kOperationRef, owner); });
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldYieldRewirePass : public FoldMutationPass<FoldYieldRewirePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldYieldRewirePass)
  StringRef getArgument() const final {
    return "shuttle-test-rewire-fold-yield";
  }
  void runOnOperation() override {
    if (FoldOp operation = fold()) {
      auto yield = cast<YieldOp>(operation.getCombiner().front().back());
      yield.getValuesMutable().assign(
          operation.getCombiner().front().getArgument(0));
    }
  }
};

class FoldAddFastMathPass : public FoldMutationPass<FoldAddFastMathPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddFastMathPass)
  StringRef getArgument() const final {
    return "shuttle-test-add-fold-fastmath";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().walk([&](arith::AddFOp add) {
        add.setFastmathAttr(arith::FastMathFlagsAttr::get(
            add.getContext(), arith::FastMathFlags::fast));
      });
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldYieldAttributePass : public FoldMutationPass<FoldYieldAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldYieldAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-fold-yield-attribute";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().front().getTerminator()->setAttr(
          "shuttle.test_semantic",
          IntegerAttr::get(IntegerType::get(operation.getContext(), 64), 7));
    }
  }
};

class FoldAttributePass : public FoldMutationPass<FoldAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-fold-attribute";
  }
  void runOnOperation() override {
    if (FoldOp operation = fold()) {
      operation->setAttr(
          "shuttle.test_semantic",
          IntegerAttr::get(IntegerType::get(operation.getContext(), 64), 7));
    }
  }
};

template <typename Derived>
class MapMutationPass : public MutationPass<Derived> {
protected:
  MapOp map(MapSemantics semantics) {
    MapOp result;
    this->getOperation().walk([&](MapOp candidate) {
      if (!result && candidate.getSemantics() == semantics) {
        result = candidate;
      }
    });
    if (!result) {
      this->getOperation().emitError(
          "test fixture has no matching shuttle.map");
      this->signalPassFailure();
    }
    return result;
  }

  void replaceInputMap(MapOp operation, AffineMap inputMap) {
    SmallVector<Attribute> maps(operation.getIndexingMaps().begin(),
                                operation.getIndexingMaps().end());
    maps[0] = AffineMapAttr::get(inputMap);
    operation->setAttr("indexing_maps",
                       ArrayAttr::get(operation.getContext(), maps));
  }

  MapOp singletonBroadcast() {
    MapOp result;
    this->getOperation().walk([&](MapOp candidate) {
      if (result || candidate.getSemantics() != MapSemantics::BroadcastInDim ||
          candidate.getInputs().size() != 1) {
        return;
      }
      auto input =
          dyn_cast<RankedTensorType>(candidate.getInputs()[0].getType());
      auto output =
          dyn_cast<RankedTensorType>(candidate.getResult(0).getType());
      if (!input || !output || candidate.getIndexingMaps().size() != 2) {
        return;
      }
      AffineMap inputMap =
          cast<AffineMapAttr>(candidate.getIndexingMaps()[0]).getValue();
      unsigned expandingSingletons = 0;
      for (AffineExpr expression : inputMap.getResults()) {
        auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
        expandingSingletons +=
            binary && binary.getKind() == AffineExprKind::FloorDiv;
      }
      if (expandingSingletons == 1) {
        result = candidate;
      }
    });
    if (!result) {
      this->getOperation().emitError(
          "test fixture has no exactly-one expanding singleton broadcast Map");
      this->signalPassFailure();
    }
    return result;
  }

  MapOp projectedParameterBroadcast() {
    MapOp result;
    this->getOperation().walk([&](MapOp candidate) {
      if (result || candidate.getSemantics() != MapSemantics::BroadcastInDim ||
          candidate.getInputs().size() != 1 ||
          candidate.getIndexingMaps().size() != 2) {
        return;
      }
      AffineMap inputMap =
          cast<AffineMapAttr>(candidate.getIndexingMaps()[0]).getValue();
      unsigned expandingSingletons = llvm::count_if(
          inputMap.getResults(), [](AffineExpr expression) {
            auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
            return binary && binary.getKind() == AffineExprKind::FloorDiv;
          });
      if (expandingSingletons > 1) {
        result = candidate;
      }
    });
    if (!result) {
      this->getOperation().emitError(
          "test fixture has no multiply-expanded projected parameter Map");
      this->signalPassFailure();
    }
    return result;
  }

  std::pair<unsigned, AffineBinaryOpExpr>
  expandedInputExpression(MapOp operation) {
    AffineMap inputMap =
        cast<AffineMapAttr>(operation.getIndexingMaps()[0]).getValue();
    for (auto [position, expression] :
         llvm::enumerate(inputMap.getResults())) {
      auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
      if (binary && binary.getKind() == AffineExprKind::FloorDiv) {
        return {position, binary};
      }
    }
    llvm_unreachable("singletonBroadcast guarantees one floorDiv expression");
  }
};

class ProjectedParameterMapMutationPass
    : public MapMutationPass<ProjectedParameterMapMutationPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ProjectedParameterMapMutationPass)
  ProjectedParameterMapMutationPass() = default;
  ProjectedParameterMapMutationPass(
      const ProjectedParameterMapMutationPass &other)
      : MapMutationPass<ProjectedParameterMapMutationPass>(other) {}
  Option<std::string> kind{*this, "kind", llvm::cl::desc("Mutation kind"),
                           llvm::cl::init("")};
  StringRef getArgument() const final {
    return "shuttle-test-mutate-projected-parameter-map";
  }
  void runOnOperation() override {
    MapOp operation = projectedParameterBroadcast();
    if (!operation) {
      return;
    }
    SmallVector<Attribute> maps(operation.getIndexingMaps().begin(),
                                operation.getIndexingMaps().end());
    AffineMap inputMap = cast<AffineMapAttr>(maps.front()).getValue();
    SmallVector<AffineExpr> expressions(inputMap.getResults());
    auto expanded = llvm::find_if(expressions, [](AffineExpr expression) {
      auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
      return binary && binary.getKind() == AffineExprKind::FloorDiv;
    });
    if (expanded == expressions.end()) {
      llvm_unreachable("projectedParameterBroadcast has expanded axes");
    }
    auto division = cast<AffineBinaryOpExpr>(*expanded);
    int64_t divisor = cast<AffineConstantExpr>(division.getRHS()).getValue();
    StringRef mutation = kind;
    unsigned numSymbols = 0;
    if (mutation == "divisor") {
      *expanded = division.getLHS().floorDiv(divisor - 1);
    } else if (mutation == "composite") {
      *expanded = (division.getLHS() + 1).floorDiv(divisor);
    } else if (mutation == "ceildiv") {
      *expanded = division.getLHS().ceilDiv(divisor);
    } else if (mutation == "mod") {
      *expanded = division.getLHS() % divisor;
    } else if (mutation == "symbol") {
      *expanded = getAffineSymbolExpr(0, operation.getContext());
      numSymbols = 1;
    } else if (mutation == "duplicate") {
      auto second = llvm::find_if(
          llvm::make_range(std::next(expanded), expressions.end()),
          [](AffineExpr expression) {
            auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
            return binary && binary.getKind() == AffineExprKind::FloorDiv;
          });
      if (second == expressions.end()) {
        operation.emitOpError("test fixture has fewer than two expansions");
        return signalPassFailure();
      }
      auto secondDivision = cast<AffineBinaryOpExpr>(*second);
      *second = division.getLHS().floorDiv(
          cast<AffineConstantExpr>(secondDivision.getRHS()).getValue());
    } else if (mutation == "direct") {
      *expanded = division.getLHS();
    } else if (mutation == "result") {
      AffineMap resultMap = cast<AffineMapAttr>(maps.back()).getValue();
      SmallVector<AffineExpr> resultExpressions(resultMap.getResults());
      std::swap(resultExpressions.front(), resultExpressions.back());
      maps.back() = AffineMapAttr::get(AffineMap::get(
          resultMap.getNumDims(), 0, resultExpressions,
          operation.getContext()));
    } else {
      operation.emitOpError("unknown projected parameter mutation kind");
      return signalPassFailure();
    }
    maps.front() = AffineMapAttr::get(AffineMap::get(
        inputMap.getNumDims(), numSymbols, expressions,
        operation.getContext()));
    operation->setAttr("indexing_maps",
                       ArrayAttr::get(operation.getContext(), maps));
  }
};

class BroadcastMappedAxesSwapPass
    : public MapMutationPass<BroadcastMappedAxesSwapPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastMappedAxesSwapPass)
  StringRef getArgument() const final {
    return "shuttle-test-swap-mapped-broadcast-axes";
  }
  void runOnOperation() override {
    MapOp operation = singletonBroadcast();
    if (!operation) {
      return;
    }
    AffineMap inputMap =
        cast<AffineMapAttr>(operation.getIndexingMaps()[0]).getValue();
    SmallVector<AffineExpr> expressions(inputMap.getResults());
    unsigned expandedPosition = expandedInputExpression(operation).first;
    auto direct = llvm::find_if(expressions, [](AffineExpr expression) {
      return isa<AffineDimExpr>(expression);
    });
    if (direct == expressions.end()) {
      operation.emitOpError("test fixture has no direct mapped broadcast axis");
      return signalPassFailure();
    }
    std::swap(*direct, expressions[expandedPosition]);
    replaceInputMap(operation,
                    AffineMap::get(inputMap.getNumDims(), 0, expressions,
                                   operation.getContext()));
  }
};

class BroadcastMappedAxisDuplicatePass
    : public MapMutationPass<BroadcastMappedAxisDuplicatePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastMappedAxisDuplicatePass)
  StringRef getArgument() const final {
    return "shuttle-test-duplicate-mapped-broadcast-axis";
  }
  void runOnOperation() override {
    MapOp operation = singletonBroadcast();
    if (!operation) {
      return;
    }
    AffineMap inputMap =
        cast<AffineMapAttr>(operation.getIndexingMaps()[0]).getValue();
    SmallVector<AffineExpr> expressions(inputMap.getResults());
    auto [expandedPosition, expanded] = expandedInputExpression(operation);
    auto direct = llvm::find_if(expressions, [](AffineExpr expression) {
      return isa<AffineDimExpr>(expression);
    });
    if (direct == expressions.end()) {
      operation.emitOpError("test fixture has no direct mapped broadcast axis");
      return signalPassFailure();
    }
    expressions[expandedPosition] =
        (*direct).floorDiv(cast<AffineConstantExpr>(expanded.getRHS()).getValue());
    replaceInputMap(operation,
                    AffineMap::get(inputMap.getNumDims(), 0, expressions,
                                   operation.getContext()));
  }
};

class BroadcastWrongDivisorPass
    : public MapMutationPass<BroadcastWrongDivisorPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastWrongDivisorPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-wrong-broadcast-divisor";
  }
  void runOnOperation() override {
    MapOp operation = singletonBroadcast();
    if (!operation) {
      return;
    }
    AffineMap inputMap =
        cast<AffineMapAttr>(operation.getIndexingMaps()[0]).getValue();
    SmallVector<AffineExpr> expressions(inputMap.getResults());
    auto [expandedPosition, expanded] = expandedInputExpression(operation);
    int64_t divisor = cast<AffineConstantExpr>(expanded.getRHS()).getValue();
    expressions[expandedPosition] = expanded.getLHS().floorDiv(divisor - 1);
    replaceInputMap(operation,
                    AffineMap::get(inputMap.getNumDims(), 0, expressions,
                                   operation.getContext()));
  }
};

class BroadcastLiteralZeroPass
    : public MapMutationPass<BroadcastLiteralZeroPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastLiteralZeroPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-broadcast-literal-zero";
  }
  void runOnOperation() override {
    MapOp operation = singletonBroadcast();
    if (!operation) {
      return;
    }
    AffineMap inputMap =
        cast<AffineMapAttr>(operation.getIndexingMaps()[0]).getValue();
    SmallVector<AffineExpr> expressions(inputMap.getResults());
    expressions[expandedInputExpression(operation).first] =
        getAffineConstantExpr(0, operation.getContext());
    replaceInputMap(operation,
                    AffineMap::get(inputMap.getNumDims(), 0, expressions,
                                   operation.getContext()));
  }
};

class BroadcastCompositeDividendPass
    : public MapMutationPass<BroadcastCompositeDividendPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastCompositeDividendPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-broadcast-composite-dividend";
  }
  void runOnOperation() override {
    MapOp operation = singletonBroadcast();
    if (!operation) {
      return;
    }
    AffineMap inputMap =
        cast<AffineMapAttr>(operation.getIndexingMaps()[0]).getValue();
    SmallVector<AffineExpr> expressions(inputMap.getResults());
    auto [expandedPosition, expanded] = expandedInputExpression(operation);
    AffineExpr shifted =
        expanded.getLHS() + getAffineConstantExpr(1, operation.getContext());
    expressions[expandedPosition] = shifted.floorDiv(
        cast<AffineConstantExpr>(expanded.getRHS()).getValue());
    replaceInputMap(operation,
                    AffineMap::get(inputMap.getNumDims(), 0, expressions,
                                   operation.getContext()));
  }
};

class BroadcastWrongResultExtentPass
    : public MapMutationPass<BroadcastWrongResultExtentPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastWrongResultExtentPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-wrong-broadcast-result-extent";
  }
  void runOnOperation() override {
    if (MapOp operation = singletonBroadcast()) {
      auto output = cast<RankedTensorType>(operation.getResult(0).getType());
      SmallVector<int64_t> shape(output.getShape());
      AffineBinaryOpExpr expanded = expandedInputExpression(operation).second;
      unsigned axis = cast<AffineDimExpr>(expanded.getLHS()).getPosition();
      --shape[axis];
      operation.getResult(0).setType(
          RankedTensorType::get(shape, output.getElementType()));
    }
  }
};

class BroadcastDirectSingletonPass
    : public MapMutationPass<BroadcastDirectSingletonPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastDirectSingletonPass)
  StringRef getArgument() const final {
    return "shuttle-test-direct-expanded-singleton";
  }
  void runOnOperation() override {
    MapOp operation = singletonBroadcast();
    if (!operation) {
      return;
    }
    AffineMap inputMap =
        cast<AffineMapAttr>(operation.getIndexingMaps()[0]).getValue();
    SmallVector<AffineExpr> expressions(inputMap.getResults());
    auto [expandedPosition, expanded] = expandedInputExpression(operation);
    expressions[expandedPosition] = expanded.getLHS();
    replaceInputMap(operation,
                    AffineMap::get(inputMap.getNumDims(), 0, expressions,
                                   operation.getContext()));
  }
};

class BroadcastExpandNonSingletonPass
    : public MapMutationPass<BroadcastExpandNonSingletonPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastExpandNonSingletonPass)
  StringRef getArgument() const final {
    return "shuttle-test-expand-nonsingleton-axis";
  }
  void runOnOperation() override {
    MapOp operation = singletonBroadcast();
    if (!operation) {
      return;
    }
    AffineMap inputMap =
        cast<AffineMapAttr>(operation.getIndexingMaps()[0]).getValue();
    auto output = cast<RankedTensorType>(operation.getResult(0).getType());
    SmallVector<AffineExpr> expressions(inputMap.getResults());
    for (AffineExpr &expression : expressions) {
      auto direct = dyn_cast<AffineDimExpr>(expression);
      if (!direct) {
        continue;
      }
      expression = direct.floorDiv(output.getDimSize(direct.getPosition()));
      replaceInputMap(operation,
                      AffineMap::get(inputMap.getNumDims(), 0, expressions,
                                     operation.getContext()));
      return;
    }
    operation.emitOpError("test fixture has no non-singleton broadcast axis");
    signalPassFailure();
  }
};

class BroadcastResultMapMutationPass
    : public MapMutationPass<BroadcastResultMapMutationPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastResultMapMutationPass)
  BroadcastResultMapMutationPass() = default;
  BroadcastResultMapMutationPass(const BroadcastResultMapMutationPass &other)
      : MapMutationPass<BroadcastResultMapMutationPass>(other) {}
  Option<std::string> kind{*this, "kind", llvm::cl::desc("Mutation kind"),
                           llvm::cl::init("")};
  StringRef getArgument() const final {
    return "shuttle-test-mutate-broadcast-result-map";
  }
  void runOnOperation() override {
    MapOp operation = singletonBroadcast();
    if (!operation) {
      return;
    }
    SmallVector<Attribute> maps(operation.getIndexingMaps().begin(),
                                operation.getIndexingMaps().end());
    AffineMap resultMap = cast<AffineMapAttr>(maps.back()).getValue();
    SmallVector<AffineExpr> expressions(resultMap.getResults());
    StringRef mutation = kind;
    if (mutation == "permutation") {
      if (expressions.size() < 2) {
        operation.emitOpError("test fixture has no permutable result map");
        return signalPassFailure();
      }
      std::swap(expressions.front(), expressions.back());
    } else if (mutation == "projection") {
      expressions.pop_back();
    } else {
      operation.emitOpError("unknown broadcast result map mutation kind");
      return signalPassFailure();
    }
    maps.back() = AffineMapAttr::get(AffineMap::get(
        resultMap.getNumDims(), 0, expressions, operation.getContext()));
    operation->setAttr("indexing_maps",
                       ArrayAttr::get(operation.getContext(), maps));
  }
};

class MapConstantZeroPointwisePass
    : public MapMutationPass<MapConstantZeroPointwisePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapConstantZeroPointwisePass)
  StringRef getArgument() const final {
    return "shuttle-test-set-pointwise-constant-zero";
  }
  void runOnOperation() override {
    MapOp operation = map(MapSemantics::Pointwise);
    if (!operation) {
      return;
    }
    auto input = cast<RankedTensorType>(operation.getInputs()[0].getType());
    MLIRContext *context = operation.getContext();
    SmallVector<AffineExpr> expressions;
    expressions.push_back(getAffineConstantExpr(0, context));
    for (unsigned dimension = 1; dimension < input.getRank(); ++dimension) {
      expressions.push_back(getAffineDimExpr(dimension, context));
    }
    replaceInputMap(operation,
                    AffineMap::get(input.getRank(), 0, expressions, context));
  }
};

class BroadcastReplayAsReshapePass
    : public MapMutationPass<BroadcastReplayAsReshapePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastReplayAsReshapePass)
  StringRef getArgument() const final {
    return "shuttle-test-replay-broadcast-as-reshape";
  }
  void runOnOperation() override {
    if (MapOp operation = singletonBroadcast()) {
      operation->setAttr(
          "semantics",
          MapSemanticsAttr::get(operation.getContext(), MapSemantics::Reshape));
    }
  }
};

class ReshapeReplayAsBroadcastPass
    : public MapMutationPass<ReshapeReplayAsBroadcastPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReshapeReplayAsBroadcastPass)
  StringRef getArgument() const final {
    return "shuttle-test-replay-reshape-as-broadcast";
  }
  void runOnOperation() override {
    MapOp operation;
    getOperation().walk([&](MapOp candidate) {
      if (operation || candidate.getSemantics() != MapSemantics::Reshape) {
        return;
      }
      auto input =
          dyn_cast<RankedTensorType>(candidate.getInputs()[0].getType());
      auto result =
          dyn_cast<RankedTensorType>(candidate.getResult(0).getType());
      if (input && result && input.getRank() > result.getRank()) {
        operation = candidate;
      }
    });
    if (!operation) {
      getOperation().emitError("test fixture has no rank-reducing reshape Map");
      signalPassFailure();
      return;
    }
    operation->setAttr("semantics",
                       MapSemanticsAttr::get(operation.getContext(),
                                             MapSemantics::BroadcastInDim));
  }
};

class BroadcastDuplicatePass : public MapMutationPass<BroadcastDuplicatePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastDuplicatePass)
  StringRef getArgument() const final {
    return "shuttle-test-duplicate-broadcast-dimension";
  }
  void runOnOperation() override {
    MapOp operation = map(MapSemantics::BroadcastInDim);
    if (!operation) {
      return;
    }
    MLIRContext *context = operation.getContext();
    replaceInputMap(operation, AffineMap::get(3, 0,
                                              {getAffineDimExpr(0, context),
                                               getAffineDimExpr(0, context)},
                                              context));
  }
};

class ReshapeAmbiguousPass : public MapMutationPass<ReshapeAmbiguousPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReshapeAmbiguousPass)
  StringRef getArgument() const final {
    return "shuttle-test-ambiguate-reshape-map";
  }
  void runOnOperation() override {
    MapOp operation;
    getOperation().walk([&](MapOp candidate) {
      auto input =
          dyn_cast<RankedTensorType>(candidate.getInputs()[0].getType());
      auto result =
          dyn_cast<RankedTensorType>(candidate.getResult(0).getType());
      if (!operation && candidate.getSemantics() == MapSemantics::Reshape &&
          input && result && input.getRank() == 2 && result.getRank() == 3) {
        operation = candidate;
      }
    });
    if (!operation) {
      getOperation().emitError("test fixture has no rank-two reshape Map");
      signalPassFailure();
      return;
    }
    MLIRContext *context = operation.getContext();
    replaceInputMap(operation, AffineMap::get(3, 0,
                                              {getAffineDimExpr(2, context),
                                               getAffineDimExpr(0, context)},
                                              context));
  }
};

class MapAttributePass : public MapMutationPass<MapAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-map-attribute";
  }
  void runOnOperation() override {
    if (MapOp operation = map(MapSemantics::BroadcastInDim)) {
      operation->setAttr(
          "shuttle.test_semantic",
          IntegerAttr::get(IntegerType::get(operation.getContext(), 64), 7));
    }
  }
};

class MapYieldAttributePass : public MapMutationPass<MapYieldAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapYieldAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-map-yield-attribute";
  }
  void runOnOperation() override {
    if (MapOp operation = map(MapSemantics::BroadcastInDim)) {
      operation.getBody().front().getTerminator()->setAttr(
          "shuttle.test_semantic",
          IntegerAttr::get(IntegerType::get(operation.getContext(), 64), 7));
    }
  }
};

class MapStructuralSemanticPass
    : public MapMutationPass<MapStructuralSemanticPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapStructuralSemanticPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-structural-map-semantic";
  }
  void runOnOperation() override {
    if (MapOp operation = map(MapSemantics::Pointwise)) {
      operation->setAttr(
          "semantics",
          MapSemanticsAttr::get(operation.getContext(), MapSemantics::Reshape));
    }
  }
};

class ManifestVersionPass : public MutationPass<ManifestVersionPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ManifestVersionPass)
  StringRef getArgument() const final {
    return "shuttle-test-downgrade-manifest-version";
  }
  void runOnOperation() override {
    auto manifest = getOperation()->getAttrOfType<DictionaryAttr>(kManifest);
    if (!manifest) {
      getOperation().emitError("test fixture has no coverage manifest");
      signalPassFailure();
      return;
    }
    NamedAttrList fields(manifest);
    fields.set(
        "version",
        IntegerAttr::get(IntegerType::get(getOperation().getContext(), 64), 1));
    getOperation()->setAttr(
        kManifest, DictionaryAttr::get(getOperation().getContext(), fields));
  }
};

class ManifestVersionMissingPass
    : public MutationPass<ManifestVersionMissingPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ManifestVersionMissingPass)
  StringRef getArgument() const final {
    return "shuttle-test-remove-manifest-version";
  }
  void runOnOperation() override {
    auto manifest = getOperation()->getAttrOfType<DictionaryAttr>(kManifest);
    if (!manifest) {
      getOperation().emitError("test fixture has no coverage manifest");
      signalPassFailure();
      return;
    }
    NamedAttrList fields(manifest);
    fields.erase("version");
    getOperation()->setAttr(
        kManifest, DictionaryAttr::get(getOperation().getContext(), fields));
  }
};

class ReportNormalizedFingerprintPass
    : public MutationPass<ReportNormalizedFingerprintPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReportNormalizedFingerprintPass)
  StringRef getArgument() const final {
    return "shuttle-test-report-normalized-fingerprint";
  }
  StringRef getDescription() const final {
    return "Report the normalized structural StableHLO fingerprint";
  }
  void runOnOperation() override {
    llvm::outs() << normalizedStablehloFingerprint(getOperation()) << '\n';
  }
};

MaterializationPlanOp materializationPlan(ModuleOp module) {
  auto plans = module.getOps<MaterializationPlanOp>();
  return plans.empty() ? MaterializationPlanOp{} : *plans.begin();
}

bool isExpandingSingletonMapTask(
    MaterializationTaskOp task,
    MutableArrayRef<MaterializationBufferOp> buffers) {
  if (task.getKind() != MaterializationTaskKind::Map ||
      task.getDomainShape().size() < 3 || task.getInputBuffers().size() != 1) {
    return false;
  }
  int64_t inputOrdinal = task.getInputBuffers().front();
  if (inputOrdinal < 0 || inputOrdinal >= static_cast<int64_t>(buffers.size())) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(buffers[inputOrdinal].getTensorType());
  if (!input || input.getRank() !=
                    static_cast<int64_t>(task.getDomainShape().size())) {
    return false;
  }
  unsigned expansions = 0;
  for (auto [inputExtent, outputExtent] :
       llvm::zip_equal(input.getShape(), task.getDomainShape())) {
    if (inputExtent == outputExtent) {
      continue;
    }
    if (inputExtent != 1 || outputExtent <= 1) {
      return false;
    }
    ++expansions;
  }
  return expansions == 1;
}

bool isProjectedParameterMapTask(
    MaterializationTaskOp task,
    MutableArrayRef<MaterializationBufferOp> buffers) {
  if (task.getKind() != MaterializationTaskKind::Map ||
      task.getDomainShape().size() < 3 || task.getInputBuffers().size() != 1) {
    return false;
  }
  int64_t inputOrdinal = task.getInputBuffers().front();
  if (inputOrdinal < 0 || inputOrdinal >= static_cast<int64_t>(buffers.size())) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(buffers[inputOrdinal].getTensorType());
  if (!input || input.getRank() !=
                    static_cast<int64_t>(task.getDomainShape().size())) {
    return false;
  }
  unsigned expansions = 0;
  for (auto [inputExtent, outputExtent] :
       llvm::zip_equal(input.getShape(), task.getDomainShape())) {
    if (inputExtent == outputExtent) {
      continue;
    }
    if (inputExtent != 1 || outputExtent <= 1) {
      return false;
    }
    ++expansions;
  }
  return expansions > 1;
}

void refreshMaterializationFingerprint(MaterializationPlanOp plan) {
  plan.setFingerprint(materializationPlanFingerprint(plan));
}

SchedulePlanOp schedulePlan(ModuleOp module) {
  auto plans = module.getOps<SchedulePlanOp>();
  return plans.empty() ? SchedulePlanOp{} : *plans.begin();
}

void refreshScheduleFingerprint(SchedulePlanOp plan) {
  plan.setFingerprint(schedulePlanFingerprint(plan));
}

void refreshMaterializationEdges(MaterializationPlanOp plan) {
  SmallVector<MaterializationBufferOp> buffers;
  SmallVector<MaterializationTaskOp> tasks;
  llvm::append_range(buffers,
                     plan.getBody().front().getOps<MaterializationBufferOp>());
  llvm::append_range(tasks,
                     plan.getBody().front().getOps<MaterializationTaskOp>());
  SmallVector<SmallVector<int64_t>> consumers(buffers.size());
  for (auto [taskOrdinal, task] : llvm::enumerate(tasks)) {
    SmallVector<int64_t> dependencies;
    for (int64_t bufferOrdinal : task.getInputBuffers()) {
      if (consumers[bufferOrdinal].empty() ||
          consumers[bufferOrdinal].back() !=
              static_cast<int64_t>(taskOrdinal)) {
        consumers[bufferOrdinal].push_back(taskOrdinal);
      }
      if (auto producer =
              buffers[bufferOrdinal]->getAttrOfType<IntegerAttr>("producer")) {
        if (!llvm::is_contained(dependencies, producer.getInt())) {
          dependencies.push_back(producer.getInt());
        }
      }
    }
    llvm::sort(dependencies);
    task->setAttr("dependencies",
                  DenseI64ArrayAttr::get(task.getContext(), dependencies));
  }
  for (auto [ordinal, buffer] : llvm::enumerate(buffers)) {
    buffer->setAttr("consumers", DenseI64ArrayAttr::get(buffer.getContext(),
                                                        consumers[ordinal]));
    int64_t lifetimeEnd =
        buffer.getLiveOut()
            ? static_cast<int64_t>(tasks.size())
            : (consumers[ordinal].empty() ? buffer.getLifetimeStart()
                                          : consumers[ordinal].back());
    buffer.setLifetimeEnd(lifetimeEnd);
  }
  refreshMaterializationFingerprint(plan);
}

class ReportMaterializationFingerprintPass
    : public MutationPass<ReportMaterializationFingerprintPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ReportMaterializationFingerprintPass)
  StringRef getArgument() const final {
    return "shuttle-test-report-materialization-fingerprint";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no materialization plan");
      return signalPassFailure();
    }
    llvm::outs() << plan.getFingerprint() << '\n';
  }
};

class SetPreferFusionTuningPass
    : public MutationPass<SetPreferFusionTuningPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetPreferFusionTuningPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-prefer-fusion-tuning";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto manifest = module->getAttrOfType<DictionaryAttr>(kManifest);
    if (!manifest) {
      module.emitError("test fixture has no coverage manifest");
      return signalPassFailure();
    }
    NamedAttrList fields(manifest);
    fields.set("canonical_tuning",
               StringAttr::get(module.getContext(), kFastFusionTuning));
    fields.set("canonical_options",
               StringAttr::get(module.getContext(), kFastFusionOptions));
    fields.set("policy", StringAttr::get(module.getContext(), "fast"));
    fields.set("policy_digest",
               StringAttr::get(module.getContext(), kFastFusionPolicyDigest));
    fields.set("tuning_digest",
               StringAttr::get(module.getContext(), kFastFusionTuningDigest));
    module->setAttr(kManifest,
                    DictionaryAttr::get(module.getContext(), fields));
  }
};

bool valueDependsOn(Value value, Operation *target) {
  llvm::DenseSet<Value> visited;
  SmallVector<Value> worklist{value};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second) {
      continue;
    }
    Operation *producer = current.getDefiningOp();
    if (!producer) {
      continue;
    }
    if (producer == target) {
      return true;
    }
    llvm::append_range(worklist, producer->getOperands());
  }
  return false;
}

bool foldYieldIs(FoldOp fold, StringRef operationName) {
  if (fold.getCombiner().empty()) {
    return false;
  }
  Block &body = fold.getCombiner().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1 || body.getNumArguments() != 2) {
    return false;
  }
  Operation *combiner = yield.getValues().front().getDefiningOp();
  if (!combiner || combiner->getName().getStringRef() != operationName ||
      combiner->getNumOperands() != 2) {
    return false;
  }
  return llvm::all_of(combiner->getOperands(), [&](Value operand) {
    return operand == body.getArgument(0) || operand == body.getArgument(1);
  }) &&
         combiner->getOperand(0) != combiner->getOperand(1);
}

bool mapYieldIs(MapOp map, StringRef operationName) {
  if (map.getBody().empty()) {
    return false;
  }
  auto yield = dyn_cast<YieldOp>(map.getBody().front().getTerminator());
  return yield && yield.getValues().size() == 1 &&
         yield.getValues().front().getDefiningOp() &&
         yield.getValues().front().getDefiningOp()->getName().getStringRef() ==
             operationName;
}

std::optional<unsigned> mapYieldUnaryArgument(MapOp map,
                                              StringRef operationName) {
  if (!mapYieldIs(map, operationName)) {
    return std::nullopt;
  }
  auto yield = cast<YieldOp>(map.getBody().front().getTerminator());
  Operation *unary = yield.getValues().front().getDefiningOp();
  if (unary->getNumOperands() != 1) {
    return std::nullopt;
  }
  auto argument = dyn_cast<BlockArgument>(unary->getOperand(0));
  if (!argument || argument.getOwner() != &map.getBody().front() ||
      argument.getArgNumber() >= map.getInputs().size()) {
    return std::nullopt;
  }
  return argument.getArgNumber();
}

std::optional<std::pair<unsigned, unsigned>>
mapYieldBinaryArguments(MapOp map, StringRef operationName) {
  if (!mapYieldIs(map, operationName)) {
    return std::nullopt;
  }
  auto yield = cast<YieldOp>(map.getBody().front().getTerminator());
  Operation *binary = yield.getValues().front().getDefiningOp();
  if (binary->getNumOperands() != 2) {
    return std::nullopt;
  }
  auto lhs = dyn_cast<BlockArgument>(binary->getOperand(0));
  auto rhs = dyn_cast<BlockArgument>(binary->getOperand(1));
  if (!lhs || !rhs || lhs.getOwner() != &map.getBody().front() ||
      rhs.getOwner() != &map.getBody().front() ||
      lhs.getArgNumber() == rhs.getArgNumber() ||
      lhs.getArgNumber() >= map.getInputs().size() ||
      rhs.getArgNumber() >= map.getInputs().size()) {
    return std::nullopt;
  }
  return std::pair(lhs.getArgNumber(), rhs.getArgNumber());
}

class RewireOnlineMaximumYieldPass
    : public MutationPass<RewireOnlineMaximumYieldPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewireOnlineMaximumYieldPass)
  StringRef getArgument() const final {
    return "shuttle-test-rewire-online-maximum-yield";
  }
  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](FoldOp fold) {
      if (!foldYieldIs(fold, arith::MaximumFOp::getOperationName())) {
        return WalkResult::advance();
      }
      auto yield = cast<YieldOp>(fold.getCombiner().front().getTerminator());
      yield->setOperand(0, fold.getCombiner().front().getArgument(0));
      return WalkResult::interrupt();
    });
    if (!result.wasInterrupted()) {
      getOperation().emitError("test fixture has no live maximum Fold yield");
      return signalPassFailure();
    }
  }
};

FailureOr<int64_t> staticTensorBytes(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape() ||
      (!tensor.getElementType().isF32() &&
       !tensor.getElementType().isBF16() &&
       !tensor.getElementType().isInteger(32) &&
       !tensor.getElementType().isInteger(1))) {
    return failure();
  }
  int64_t elements = 1;
  for (int64_t extent : tensor.getShape()) {
    if (extent <= 0 || elements > INT64_MAX / extent) {
      return failure();
    }
    elements *= extent;
  }
  const int64_t elementBytes =
      tensor.getElementType().isF32() ||
              tensor.getElementType().isInteger(32)
          ? 4
          : (tensor.getElementType().isBF16() ? 2 : 1);
  if (elements > INT64_MAX / elementBytes) {
    return failure();
  }
  return elements * elementBytes;
}

LogicalResult checkedAccumulate(int64_t value, int64_t &total) {
  if (value < 0 || total > INT64_MAX - value) {
    return failure();
  }
  total += value;
  return success();
}

LogicalResult checkedAccumulateTraffic(int64_t bytes, int64_t accesses,
                                       int64_t &total) {
  if (bytes < 0 || accesses < 0 ||
      (bytes != 0 && accesses > INT64_MAX / bytes)) {
    return failure();
  }
  return checkedAccumulate(bytes * accesses, total);
}

FailureOr<AffineMap> mapCoordinatesToAncestor(Value value, Value ancestor) {
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  if (!valueType) {
    return failure();
  }
  if (value == ancestor) {
    return AffineMap::getMultiDimIdentityMap(valueType.getRank(),
                                             value.getContext());
  }
  auto map = value.getDefiningOp<MapOp>();
  if (!map || map.getInputs().size() != 1 || map.getResults().size() != 1 ||
      map.getIndexingMaps().size() != 2) {
    return failure();
  }
  auto inputMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[0]);
  auto resultMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[1]);
  if (!inputMap || !resultMap || !resultMap.getValue().isIdentity()) {
    return failure();
  }
  FailureOr<AffineMap> upstream =
      mapCoordinatesToAncestor(map.getInputs().front(), ancestor);
  if (failed(upstream) ||
      upstream->getNumDims() != inputMap.getValue().getNumResults()) {
    return failure();
  }
  return upstream->compose(inputMap.getValue());
}

FailureOr<AffineMap> projectedReductionCoordinates(Type scoreType,
                                                   int64_t reduction) {
  auto score = dyn_cast<RankedTensorType>(scoreType);
  if (!score || reduction < 0 || reduction >= score.getRank()) {
    return failure();
  }
  SmallVector<AffineExpr> coordinates;
  for (int64_t axis = 0; axis < score.getRank(); ++axis) {
    if (axis != reduction) {
      coordinates.push_back(getAffineDimExpr(axis, score.getContext()));
    }
  }
  return AffineMap::get(score.getRank(), 0, coordinates, score.getContext());
}

FailureOr<SmallVector<int64_t>> onlineRelationDomainAxes(
    MaterializationTaskOp task, RegionOp region,
    ArrayRef<RelationOp> relations, int64_t relationRank);

class VerifyOnlineSufficientStatePass
    : public MutationPass<VerifyOnlineSufficientStatePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyOnlineSufficientStatePass)
  StringRef getArgument() const final {
    return "shuttle-test-verify-online-sufficient-state";
  }
  StringRef getDescription() const final {
    return "Verify a topology-neutral online normalized contraction plan";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto manifest = module->getAttrOfType<DictionaryAttr>(kManifest);
    auto manifestPolicy = manifest ? manifest.getAs<StringAttr>("policy")
                                   : StringAttr{};
    auto canonicalOptions =
        manifest ? manifest.getAs<StringAttr>("canonical_options")
                 : StringAttr{};
    auto canonicalTuning =
        manifest ? manifest.getAs<StringAttr>("canonical_tuning")
                 : StringAttr{};
    auto policyDigest = manifest ? manifest.getAs<StringAttr>("policy_digest")
                                 : StringAttr{};
    auto tuningDigest = manifest ? manifest.getAs<StringAttr>("tuning_digest")
                                 : StringAttr{};
    if (!manifestPolicy || manifestPolicy.getValue() != "fast" ||
        !canonicalOptions ||
        canonicalOptions.getValue() != kFastFusionOptions ||
        !canonicalTuning || canonicalTuning.getValue() != kFastFusionTuning ||
        !policyDigest || policyDigest.getValue() != kFastFusionPolicyDigest ||
        !tuningDigest || tuningDigest.getValue() != kFastFusionTuningDigest) {
      module.emitError("online planning lacks exact FAST policy identity");
      return signalPassFailure();
    }
    SmallVector<RegionOp> regions;
    module.walk([&](RegionOp region) { regions.push_back(region); });
    constexpr llvm::StringLiteral kind =
        "online_normalized_exponential_weighted_contract";
    MaterializationPlanOp plan;
    llvm::SmallDenseSet<Attribute> plannedSources;
    llvm::SmallDenseSet<Attribute> plannedRecomputedSources;
    DictionaryAttr cost;
    llvm::SmallDenseSet<int64_t> compositionTaskOrdinals;
    for (MaterializationPlanOp candidate :
         module.getOps<MaterializationPlanOp>()) {
      if (candidate.getPolicy() != NumericalPolicy::Fast) {
        continue;
      }
      llvm::SmallDenseSet<Attribute> candidateSources;
      llvm::SmallDenseMap<Attribute, int64_t> primaryOwner;
      SmallVector<std::pair<int64_t, ArrayAttr>> recomputationDeclarations;
      llvm::SmallDenseSet<int64_t> candidateTaskOrdinals;
      DictionaryAttr candidateCost;
      for (MaterializationTaskOp task :
           candidate.getBody().front().getOps<MaterializationTaskOp>()) {
        auto taskKind = task.getCompositionKind();
        if (!taskKind || *taskKind != kind) {
          continue;
        }
        std::optional<ArrayAttr> sources = task.getCompositionSources();
        if (!sources || sources->empty()) {
          module.emitError("online composition has no semantic source closure");
          return signalPassFailure();
        }
        for (Attribute source : *sources) {
          if (!candidateSources.insert(source).second ||
              !primaryOwner.try_emplace(source, task.getOrdinal()).second) {
            module.emitError(
                "online composition sources do not uniquely partition the cut");
            return signalPassFailure();
          }
        }
        candidateTaskOrdinals.insert(task.getOrdinal());
        if (auto recomputed = task->getAttrOfType<ArrayAttr>(
                "composition_recomputed_sources"))
          recomputationDeclarations.push_back({task.getOrdinal(), recomputed});
        if (std::optional<DictionaryAttr> taskCost =
                task.getCompositionCost()) {
          if (candidateCost && candidateCost != *taskCost) {
            module.emitError(
                "online composition has inconsistent cost evidence");
            return signalPassFailure();
          }
          candidateCost = *taskCost;
        }
      }
      llvm::SmallDenseSet<Attribute> candidateRecomputedSources;
      for (auto [taskOrdinal, sources] : recomputationDeclarations) {
        for (Attribute source : sources) {
          auto owner = primaryOwner.find(source);
          if (owner == primaryOwner.end() || owner->second == taskOrdinal ||
              !candidateRecomputedSources.insert(source).second) {
            module.emitError(
                "online recomputation does not bind an earlier source cut");
            return signalPassFailure();
          }
        }
      }
      if (!candidateSources.empty()) {
        plan = candidate;
        plannedSources = std::move(candidateSources);
        plannedRecomputedSources = std::move(candidateRecomputedSources);
        compositionTaskOrdinals = std::move(candidateTaskOrdinals);
        cost = candidateCost;
        break;
      }
    }
    if (!plan || plannedSources.empty() || !cost) {
      module.emitError("requires online composition semantics and cost evidence");
      return signalPassFailure();
    }

    RegionOp region;
    SmallVector<Operation *> algebra;
    SmallVector<Operation *> onlineAlgebra;
    for (RegionOp candidate : regions) {
      llvm::SmallDenseSet<Attribute> candidateSources;
      SmallVector<Operation *> candidateAlgebra;
      for (Operation &operation :
           candidate.getBody().front().without_terminator()) {
        Attribute source = operation.getAttr("source");
        if (!source) {
          module.emitError("algebra operation lacks source identity");
          return signalPassFailure();
        }
        candidateSources.insert(source);
        candidateAlgebra.push_back(&operation);
      }
      if (llvm::all_of(plannedSources, [&](Attribute source) {
            return candidateSources.contains(source);
          })) {
        region = candidate;
        algebra = std::move(candidateAlgebra);
        llvm::copy_if(algebra, std::back_inserter(onlineAlgebra),
                      [&](Operation *operation) {
                        return plannedSources.contains(
                            operation->getAttr("source"));
                      });
        break;
      }
    }
    if (!region) {
      module.emitError("online composition does not close the algebra cut");
      return signalPassFailure();
    }
    llvm::SmallDenseSet<Operation *> onlineOperations(onlineAlgebra.begin(),
                                                       onlineAlgebra.end());
    for (Operation *operation : onlineAlgebra) {
      for (Value operand : operation->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (!producer || onlineOperations.contains(producer) ||
            producer->getParentOfType<RegionOp>() != region) {
          continue;
        }
        if (llvm::any_of(onlineAlgebra, [&](Operation *onlineOperation) {
              return valueDependsOn(operand, onlineOperation);
            })) {
          module.emitError("online composition has a non-closed reentry cut");
          return signalPassFailure();
        }
      }
    }

    FoldOp maximumFold;
    FoldOp additiveFold;
    MapOp exponentialMap;
    ContractOp weightedContract;
    SmallVector<RelationOp> relationOperations;
    auto regionYield = cast<YieldOp>(region.getBody().front().getTerminator());
    for (Operation *operation : onlineAlgebra) {
      if (auto fold = dyn_cast<FoldOp>(operation)) {
        if (foldYieldIs(fold, arith::MaximumFOp::getOperationName())) {
          maximumFold = fold;
        } else if (foldYieldIs(fold, arith::AddFOp::getOperationName())) {
          additiveFold = fold;
        }
      } else if (auto map = dyn_cast<MapOp>(operation)) {
        if (mapYieldIs(map, "math.exp")) {
          exponentialMap = map;
        }
      } else if (auto contract = dyn_cast<ContractOp>(operation)) {
        const bool reachesOutput =
            llvm::any_of(regionYield.getValues(), [&](Value output) {
              return valueDependsOn(output, contract);
            });
        const bool feedsAnotherContract =
            llvm::any_of(algebra, [&](Operation *consumer) {
              auto other = dyn_cast<ContractOp>(consumer);
              return other && other != contract &&
                     llvm::any_of(other.getInputs(), [&](Value input) {
                       return valueDependsOn(input, contract);
                     });
            });
        if (reachesOutput && !feedsAnotherContract) {
          weightedContract = contract;
        }
      } else if (auto relation = dyn_cast<RelationOp>(operation)) {
        relationOperations.push_back(relation);
      }
    }
    bool relationFeedsScore = false;
    bool contractFeedsScore = false;
    if (maximumFold && maximumFold.getInputs().size() == 1) {
      for (Operation *operation : onlineAlgebra) {
        relationFeedsScore |=
            isa<RelationOp>(operation) &&
            valueDependsOn(maximumFold.getInputs().front(), operation);
        contractFeedsScore |=
            isa<ContractOp>(operation) && operation != weightedContract &&
            valueDependsOn(maximumFold.getInputs().front(), operation);
      }
    }
    const bool exponentialUsesMaximum =
        exponentialMap && maximumFold &&
        llvm::any_of(exponentialMap.getInputs(), [&](Value input) {
          return valueDependsOn(input, maximumFold);
        });
    MapOp scoreMinusMaximumMap;
    Value maximumExpansion;
    if (exponentialMap && maximumFold &&
        maximumFold.getInputs().size() == 1) {
      auto exponentialArgument =
          mapYieldUnaryArgument(exponentialMap, "math.exp");
      if (exponentialArgument) {
        Value input = exponentialMap.getInputs()[*exponentialArgument];
        auto candidate = input.getDefiningOp<MapOp>();
        auto arguments = candidate ? mapYieldBinaryArguments(
                                         candidate,
                                         arith::SubFOp::getOperationName())
                                   : std::nullopt;
        if (arguments &&
            candidate.getInputs()[arguments->first] ==
                maximumFold.getInputs().front() &&
            valueDependsOn(candidate.getInputs()[arguments->second],
                           maximumFold)) {
          scoreMinusMaximumMap = candidate;
          maximumExpansion = candidate.getInputs()[arguments->second];
        }
      }
    }
    const bool denominatorUsesExponential =
        additiveFold && exponentialMap &&
        additiveFold.getInputs().size() == 1 &&
        exponentialMap.getResults().size() == 1 &&
        additiveFold.getInputs().front() ==
            exponentialMap.getResults().front();
    std::optional<unsigned> normalizedWeightOperand;
    Value denominatorExpansion;
    if (weightedContract && additiveFold && exponentialMap) {
      for (auto [operandNumber, input] :
           llvm::enumerate(weightedContract.getInputs())) {
        auto normalization = input.getDefiningOp<MapOp>();
        auto arguments = normalization
                             ? mapYieldBinaryArguments(
                                   normalization, arith::DivFOp::getOperationName())
                             : std::nullopt;
        if (arguments &&
            exponentialMap.getResults().size() == 1 &&
            normalization.getInputs()[arguments->first] ==
                exponentialMap.getResults().front() &&
            valueDependsOn(normalization.getInputs()[arguments->second],
                           additiveFold)) {
          normalizedWeightOperand = operandNumber;
          denominatorExpansion =
              normalization.getInputs()[arguments->second];
          break;
        }
      }
    }
    Type scoreType = maximumFold && maximumFold.getInputs().size() == 1
                         ? maximumFold.getInputs().front().getType()
                         : Type{};
    llvm::SmallDenseSet<Attribute> requiredRecomputedSources;
    if (maximumFold && maximumFold.getInputs().size() == 1) {
      llvm::DenseSet<Value> visited;
      SmallVector<Value> worklist{maximumFold.getInputs().front()};
      while (!worklist.empty()) {
        Value value = worklist.pop_back_val();
        if (!visited.insert(value).second)
          continue;
        Operation *producer = value.getDefiningOp();
        if (!producer || producer->getParentOfType<RegionOp>() != region)
          continue;
        if (Attribute source = producer->getAttr("source"))
          requiredRecomputedSources.insert(source);
        llvm::append_range(worklist, producer->getOperands());
      }
    }
    const bool reductionsAlign =
        maximumFold && additiveFold &&
        maximumFold.getReductionDimensions() ==
            additiveFold.getReductionDimensions() &&
        maximumFold.getInputs().size() == 1 &&
        additiveFold.getInputs().size() == 1 &&
        maximumFold.getInputs().front().getType() ==
            additiveFold.getInputs().front().getType() &&
        maximumFold.getResults().size() == 1 &&
        additiveFold.getResults().size() == 1 &&
        maximumFold.getResults().front().getType() ==
            additiveFold.getResults().front().getType();
    bool stateCoordinatesAlign = false;
    if (reductionsAlign && maximumExpansion && denominatorExpansion &&
        maximumFold.getReductionDimensions().size() == 1) {
      const int64_t reduction = maximumFold.getReductionDimensions().front();
      FailureOr<AffineMap> expected =
          projectedReductionCoordinates(scoreType, reduction);
      FailureOr<AffineMap> maximumCoordinates = mapCoordinatesToAncestor(
          maximumExpansion, maximumFold.getResults().front());
      FailureOr<AffineMap> denominatorCoordinates = mapCoordinatesToAncestor(
          denominatorExpansion, additiveFold.getResults().front());
      stateCoordinatesAlign =
          succeeded(expected) && succeeded(maximumCoordinates) &&
          succeeded(denominatorCoordinates) &&
          *maximumCoordinates == *expected &&
          *denominatorCoordinates == *expected;
    }
    bool contractionReducesStateCoordinate = false;
    if (reductionsAlign && weightedContract &&
        maximumFold.getReductionDimensions().size() == 1) {
      const int64_t reduction = maximumFold.getReductionDimensions().front();
      for (auto [operandNumber, input] :
           llvm::enumerate(weightedContract.getInputs())) {
        if (!normalizedWeightOperand ||
            operandNumber != *normalizedWeightOperand ||
            operandNumber >= weightedContract.getIndexingMaps().size()) {
          continue;
        }
        auto map = dyn_cast<AffineMapAttr>(
            weightedContract.getIndexingMaps()[operandNumber]);
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        if (!map || !inputType || reduction < 0 ||
            reduction >= inputType.getRank()) {
          continue;
        }
        auto domain = dyn_cast<AffineDimExpr>(map.getValue().getResult(reduction));
        if (!domain ||
            domain.getPosition() >= weightedContract.getIteratorKinds().size()) {
          continue;
        }
        auto iterator = dyn_cast<StringAttr>(
            weightedContract.getIteratorKinds()[domain.getPosition()]);
        contractionReducesStateCoordinate =
            iterator && iterator.getValue() == "reduction";
      }
    }
    const bool scoreSourcesOwned = plannedRecomputedSources.empty() &&
        llvm::all_of(requiredRecomputedSources, [&](Attribute source) {
          return plannedSources.contains(source);
        });
    if (!relationFeedsScore || !contractFeedsScore ||
        !exponentialUsesMaximum || !scoreMinusMaximumMap ||
        !denominatorUsesExponential ||
        !normalizedWeightOperand || !reductionsAlign ||
        !scoreSourcesOwned ||
        !stateCoordinatesAlign || !contractionReducesStateCoordinate) {
      module.emitError(
          "online composition does not prove {m,l,o} state semantics");
      return signalPassFailure();
    }

    auto rankedScoreType = dyn_cast<RankedTensorType>(scoreType);
    if (!rankedScoreType || relationOperations.empty()) {
      module.emitError("online composition lacks a Relation score domain");
      return signalPassFailure();
    }
    for (MaterializationTaskOp task :
         plan.getBody().front().getOps<MaterializationTaskOp>()) {
      if (!compositionTaskOrdinals.contains(task.getOrdinal()))
        continue;
      if (failed(onlineRelationDomainAxes(task, region, relationOperations,
                                          rankedScoreType.getRank()))) {
        module.emitError(
            "online composition has a mismatched Relation score axis map");
        return signalPassFailure();
      }
    }

    int64_t actualCandidateBytes = 0;
    int64_t actualCandidateTraffic = 0;
    for (MaterializationBufferOp buffer :
         plan.getBody().front().getOps<MaterializationBufferOp>()) {
      if (buffer.getStorage() == MaterializationStorage::Temporary &&
          buffer.getTensorType() == scoreType) {
        module.emitError("online composition materializes representative scores");
        return signalPassFailure();
      }
      auto producer = buffer.getProducer();
      const bool producedByOnline =
          producer && compositionTaskOrdinals.contains(*producer);
      const bool consumedByOnline = llvm::any_of(
          buffer.getConsumers(), [&](int64_t consumer) {
            return compositionTaskOrdinals.contains(consumer);
          });
      // Charge both sides of the composition cut, including all accesses to a
      // temporary that crosses either boundary.
      if (buffer.getStorage() != MaterializationStorage::Temporary ||
          (!producedByOnline && !consumedByOnline)) {
        continue;
      }
      FailureOr<int64_t> bytes = staticTensorBytes(buffer.getTensorType());
      if (failed(bytes) ||
          failed(checkedAccumulate(*bytes, actualCandidateBytes)) ||
          failed(checkedAccumulateTraffic(
              *bytes,
              static_cast<int64_t>(buffer.getConsumers().size()) +
                  (producer ? 1 : 0),
              actualCandidateTraffic))) {
        module.emitError("online candidate buffer cost is not statically bounded");
        return signalPassFailure();
      }
    }
    int64_t actualBaselineBytes = 0;
    int64_t actualBaselineTraffic = 0;
    // Derive the materialized comparison directly from algebra SSA, charging
    // values produced or consumed across the selected composition boundary.
    for (Operation *operation : algebra) {
      for (Value result : operation->getResults()) {
        const bool liveOut = llvm::any_of(result.getUses(), [&](OpOperand &use) {
          return use.getOwner() == regionYield;
        });
        llvm::SmallDenseSet<Operation *> consumers;
        bool consumedByOnline = false;
        for (OpOperand &use : result.getUses()) {
          consumers.insert(use.getOwner());
          consumedByOnline |= onlineOperations.contains(use.getOwner());
        }
        if (liveOut || consumers.empty() ||
            (!onlineOperations.contains(operation) && !consumedByOnline)) {
          continue;
        }
        FailureOr<int64_t> bytes = staticTensorBytes(result.getType());
        if (failed(bytes) ||
            failed(checkedAccumulate(*bytes, actualBaselineBytes)) ||
            failed(checkedAccumulateTraffic(
                *bytes, static_cast<int64_t>(consumers.size()) + 1,
                actualBaselineTraffic))) {
          module.emitError("materialized baseline cost is not statically bounded");
          return signalPassFailure();
        }
      }
    }
    FailureOr<int64_t> scoreBytes = staticTensorBytes(scoreType);
    auto integerCost = [&](StringRef name) -> std::optional<int64_t> {
      auto value = cost.getAs<IntegerAttr>(name);
      return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
    };
    auto baselineLaunches = integerCost("baseline_launches");
    auto candidateLaunches = integerCost("candidate_launches");
    auto baselineBytes = integerCost("baseline_temporary_bytes");
    auto candidateBytes = integerCost("candidate_temporary_bytes");
    auto baselineTraffic = integerCost("baseline_temporary_traffic_bytes");
    auto candidateTraffic = integerCost("candidate_temporary_traffic_bytes");
    auto contractEvaluations = integerCost("contract_evaluations");
    const int64_t expectedContractEvaluations =
        llvm::count_if(onlineAlgebra, [](Operation *operation) {
          return isa<ContractOp>(operation);
        }) +
        llvm::count_if(plannedRecomputedSources, [&](Attribute source) {
          return llvm::any_of(algebra, [&](Operation *operation) {
            return operation->getAttr("source") == source &&
                   isa<ContractOp>(operation);
          });
        });
    if (failed(scoreBytes) || !baselineLaunches || !candidateLaunches ||
        !baselineBytes || !candidateBytes || !baselineTraffic ||
        !candidateTraffic || !contractEvaluations ||
        *candidateLaunches > *baselineLaunches ||
        *baselineBytes != actualBaselineBytes ||
        *candidateBytes != actualCandidateBytes ||
        *baselineTraffic != actualBaselineTraffic ||
        *candidateTraffic != actualCandidateTraffic ||
        actualCandidateBytes >= actualBaselineBytes ||
        actualCandidateTraffic >= actualBaselineTraffic ||
        actualCandidateBytes >= *scoreBytes ||
        *contractEvaluations != expectedContractEvaluations) {
      module.emitError(
          "online composition lacks relative cost and traffic improvement");
      return signalPassFailure();
    }
    llvm::outs() << "online_state = {m,l,o}; source_cut = complete; "
                    "representative_score_materialized = false; "
                    "temporary_bytes = improved; temporary_traffic = improved\n";
  }
};

struct OnlinePtxInstruction {
  int64_t ordinal = -1;
  int64_t block = -1;
  std::string role;
  std::string sourceRef;
  std::optional<int64_t> logicalDimension;
  std::string predicate;
  bool negatedPredicate = false;
  std::string opcode;
  std::vector<std::string> operands;
  std::string destination;
  std::string branchTarget;
};

using OnlineDefinitionState =
    std::map<std::string, std::set<int64_t>>;
constexpr int64_t kOnlineUndefinedDefinition = -1;

struct OnlinePtxCfg {
  std::vector<OnlinePtxInstruction> instructions;
  std::vector<std::vector<int64_t>> blocks;
  std::map<std::string, int64_t> labels;
  std::vector<std::vector<int64_t>> successors;
  std::vector<std::vector<int64_t>> predecessors;
  std::vector<OnlineDefinitionState> before;
  std::map<std::string, int64_t> roles;
  std::map<std::string, std::vector<int64_t>> sources;
};

std::string onlineSourceKey(SourceRefAttr source) {
  return (Twine(source.getFunctionOrdinal()) + ":" +
          Twine(source.getBlockOrdinal()) + ":" +
          Twine(source.getOperationOrdinal()) + ":" +
          Twine(source.getResultOrdinal()))
      .str();
}

std::vector<std::string> onlinePtxTokens(StringRef text) {
  std::vector<std::string> tokens;
  auto isIdentifierContinuation = [](char character) {
    return llvm::isAlnum(character) || character == '_' || character == '.' ||
           character == '$';
  };
  for (size_t index = 0; index < text.size();) {
    const bool registerToken = text[index] == '%';
    const bool parameterToken = text[index] == 'p' && index + 1 < text.size() &&
                                llvm::isDigit(text[index + 1]);
    const bool sharedSymbolToken =
        text.substr(index).starts_with("shuttle_smem") &&
        (index == 0 || !isIdentifierContinuation(text[index - 1]));
    const bool constantToken =
        (text[index] == '0' && index + 2 < text.size() &&
         text[index + 1] == 'f' && llvm::isHexDigit(text[index + 2])) ||
        llvm::isDigit(text[index]) || text[index] == '-';
    if (!registerToken && !parameterToken && !sharedSymbolToken &&
        !constantToken) {
      ++index;
      continue;
    }
    size_t end = index + (text[index] == '0' && index + 1 < text.size() &&
                                  text[index + 1] == 'f'
                              ? 2
                              : 1);
    while (end < text.size() && isIdentifierContinuation(text[end]))
      ++end;
    tokens.push_back(text.slice(index, end).str());
    index = end;
  }
  return tokens;
}

bool onlinePtxDefinesValue(StringRef opcode,
                           ArrayRef<std::string> operands) {
  return !operands.empty() && StringRef(operands.front()).starts_with("%") &&
         !opcode.starts_with("st.") && !opcode.starts_with("bra") &&
         !opcode.starts_with("bar.") && opcode != "ret" && opcode != "exit";
}

void onlineTransfer(const OnlinePtxInstruction &instruction,
                    OnlineDefinitionState &state) {
  if (instruction.destination.empty())
    return;
  std::set<int64_t> &definitions = state[instruction.destination];
  if (instruction.predicate.empty()) {
    definitions.clear();
  } else if (definitions.empty()) {
    definitions.insert(kOnlineUndefinedDefinition);
  }
  definitions.insert(instruction.ordinal);
}

FailureOr<OnlinePtxCfg> onlinePtxCfg(StringRef ptx) {
  OnlinePtxCfg cfg;
  cfg.blocks.emplace_back();
  int64_t block = 0;
  std::string pendingRole;
  std::string pendingSource;
  std::optional<int64_t> pendingLogicalDimension;
  while (!ptx.empty()) {
    auto [line, remainder] = ptx.split('\n');
    ptx = remainder;
    line = line.trim();
    if (line.starts_with("// shuttle.online.role ")) {
      pendingRole =
          line.drop_front(StringRef("// shuttle.online.role ").size()).str();
      continue;
    }
    if (line.starts_with("// shuttle.online.source_ref ")) {
      pendingSource =
          line.drop_front(StringRef("// shuttle.online.source_ref ").size())
              .str();
      continue;
    }
    if (line.starts_with("// shuttle.online.logical_dimension ")) {
      int64_t dimension = -1;
      if (line
              .drop_front(
                  StringRef("// shuttle.online.logical_dimension ").size())
              .getAsInteger(10, dimension) ||
          dimension < 0)
        return failure();
      pendingLogicalDimension = dimension;
      continue;
    }
    if (line.empty() || line.starts_with("//") || line.starts_with(".") ||
        line == "{" || line == "}")
      continue;
    if (line.ends_with(":")) {
      if (!cfg.blocks[block].empty()) {
        cfg.blocks.emplace_back();
        block = cfg.blocks.size() - 1;
      }
      cfg.labels.try_emplace(line.drop_back().str(), block);
      continue;
    }
    OnlinePtxInstruction instruction;
    instruction.ordinal = cfg.instructions.size();
    instruction.block = block;
    instruction.role = std::move(pendingRole);
    instruction.sourceRef = std::move(pendingSource);
    instruction.logicalDimension = pendingLogicalDimension;
    pendingRole.clear();
    pendingSource.clear();
    pendingLogicalDimension.reset();
    if (line.starts_with("@")) {
      auto [guard, body] = line.split(' ');
      instruction.negatedPredicate = guard.starts_with("@!");
      guard.consume_front(instruction.negatedPredicate ? "@!" : "@");
      instruction.predicate = guard.str();
      line = body.trim();
    }
    auto [opcode, operands] = line.split(' ');
    instruction.opcode = opcode.rtrim(";").str();
    instruction.operands = onlinePtxTokens(operands);
    if (onlinePtxDefinesValue(instruction.opcode, instruction.operands))
      instruction.destination = instruction.operands.front();
    if (StringRef(instruction.opcode).starts_with("bra")) {
      StringRef target = operands.trim().rtrim(";");
      instruction.branchTarget = target.str();
    } else if (StringRef(instruction.opcode).starts_with("br") ||
               StringRef(instruction.opcode).starts_with("call")) {
      return failure();
    }
    if (!instruction.role.empty() &&
        !cfg.roles.try_emplace(instruction.role, instruction.ordinal).second)
      return failure();
    if (!instruction.sourceRef.empty())
      cfg.sources[instruction.sourceRef].push_back(instruction.ordinal);
    cfg.blocks[block].push_back(instruction.ordinal);
    cfg.instructions.push_back(std::move(instruction));
    if (StringRef(cfg.instructions.back().opcode).starts_with("bra") ||
        StringRef(cfg.instructions.back().opcode).starts_with("ret") ||
        StringRef(cfg.instructions.back().opcode).starts_with("exit")) {
      cfg.blocks.emplace_back();
      block = cfg.blocks.size() - 1;
    }
  }
  if (cfg.blocks.back().empty())
    cfg.blocks.pop_back();
  if (cfg.blocks.empty())
    return failure();
  cfg.successors.resize(cfg.blocks.size());
  cfg.predecessors.resize(cfg.blocks.size());
  for (auto [blockOrdinal, instructions] : llvm::enumerate(cfg.blocks)) {
    if (instructions.empty())
      continue;
    const OnlinePtxInstruction &last = cfg.instructions[instructions.back()];
    if (StringRef(last.opcode).starts_with("bra")) {
      auto target = cfg.labels.find(last.branchTarget);
      if (target == cfg.labels.end())
        return failure();
      cfg.successors[blockOrdinal].push_back(target->second);
      if (!last.predicate.empty() && blockOrdinal + 1 < cfg.blocks.size())
        cfg.successors[blockOrdinal].push_back(blockOrdinal + 1);
    } else if (!StringRef(last.opcode).starts_with("ret") &&
               !StringRef(last.opcode).starts_with("exit") &&
               blockOrdinal + 1 < cfg.blocks.size()) {
      cfg.successors[blockOrdinal].push_back(blockOrdinal + 1);
    }
  }
  for (auto [blockOrdinal, successors] : llvm::enumerate(cfg.successors))
    for (int64_t successor : successors)
      cfg.predecessors[successor].push_back(blockOrdinal);

  std::vector<OnlineDefinitionState> incoming(cfg.blocks.size());
  std::vector<OnlineDefinitionState> outgoing(cfg.blocks.size());
  bool changed = true;
  while (changed) {
    changed = false;
    for (int64_t blockOrdinal = 0;
         blockOrdinal < static_cast<int64_t>(cfg.blocks.size());
         ++blockOrdinal) {
      OnlineDefinitionState nextIncoming;
      std::set<std::string> predecessorValues;
      for (int64_t predecessor : cfg.predecessors[blockOrdinal])
        for (const auto &[value, definitions] : outgoing[predecessor])
          predecessorValues.insert(value);
      for (const std::string &value : predecessorValues) {
        for (int64_t predecessor : cfg.predecessors[blockOrdinal]) {
          auto definitions = outgoing[predecessor].find(value);
          if (definitions == outgoing[predecessor].end()) {
            nextIncoming[value].insert(kOnlineUndefinedDefinition);
          } else {
            nextIncoming[value].insert(definitions->second.begin(),
                                       definitions->second.end());
          }
        }
      }
      OnlineDefinitionState nextOutgoing = nextIncoming;
      for (int64_t instruction : cfg.blocks[blockOrdinal])
        onlineTransfer(cfg.instructions[instruction], nextOutgoing);
      if (nextIncoming != incoming[blockOrdinal] ||
          nextOutgoing != outgoing[blockOrdinal]) {
        incoming[blockOrdinal] = std::move(nextIncoming);
        outgoing[blockOrdinal] = std::move(nextOutgoing);
        changed = true;
      }
    }
  }
  cfg.before.resize(cfg.instructions.size());
  for (auto [blockOrdinal, instructions] : llvm::enumerate(cfg.blocks)) {
    OnlineDefinitionState state = incoming[blockOrdinal];
    for (int64_t instruction : instructions) {
      cfg.before[instruction] = state;
      onlineTransfer(cfg.instructions[instruction], state);
    }
  }
  return cfg;
}

bool onlineInstructionDominates(const OnlinePtxCfg &cfg, int64_t dominator,
                                int64_t dominated);

std::set<int64_t> onlineReachingDefinitions(const OnlinePtxCfg &cfg,
                                            int64_t instruction,
                                            StringRef operand) {
  if (!operand.starts_with("%"))
    return {};
  auto found = cfg.before[instruction].find(operand.str());
  if (found == cfg.before[instruction].end())
    return {};
  std::set<int64_t> definitions = found->second;
  if (definitions.count(kOnlineUndefinedDefinition) != 0 &&
      llvm::any_of(definitions, [&](int64_t definition) {
        return definition != kOnlineUndefinedDefinition &&
               cfg.instructions[definition].predicate.empty() &&
               onlineInstructionDominates(cfg, definition, instruction);
      }))
    definitions.erase(kOnlineUndefinedDefinition);
  return definitions;
}

bool onlineBlockReachable(const OnlinePtxCfg &cfg, int64_t from,
                          int64_t to) {
  std::set<int64_t> seen;
  std::vector<int64_t> worklist{from};
  while (!worklist.empty()) {
    int64_t block = worklist.back();
    worklist.pop_back();
    if (!seen.insert(block).second)
      continue;
    if (block == to)
      return true;
    llvm::append_range(worklist, cfg.successors[block]);
  }
  return false;
}

bool onlineRoleDirectlyReaches(const OnlinePtxCfg &cfg, StringRef role,
                               int64_t consumer) {
  auto producer = cfg.roles.find(role.str());
  if (producer == cfg.roles.end())
    return false;
  if (!cfg.instructions[consumer].predicate.empty() &&
      onlineReachingDefinitions(cfg, consumer,
                                cfg.instructions[consumer].predicate) ==
          std::set<int64_t>{producer->second})
    return true;
  return llvm::any_of(
      llvm::drop_begin(cfg.instructions[consumer].operands),
      [&](const std::string &operand) {
        return onlineReachingDefinitions(cfg, consumer, operand) ==
               std::set<int64_t>{producer->second};
      });
}

bool onlineInstructionTransitivelyDependsOn(
    const OnlinePtxCfg &cfg, int64_t consumer, int64_t producer,
    std::set<std::pair<int64_t, int64_t>> &active) {
  if (consumer == producer)
    return true;
  if (!active.insert({consumer, producer}).second)
    return false;
  const OnlinePtxInstruction &instruction = cfg.instructions[consumer];
  SmallVector<std::string> dependencies(
      llvm::drop_begin(instruction.operands));
  if (!instruction.predicate.empty())
    dependencies.push_back(instruction.predicate);
  for (const std::string &dependency : dependencies) {
    std::set<int64_t> definitions =
        onlineReachingDefinitions(cfg, consumer, dependency);
    if (definitions.empty() ||
        definitions.count(kOnlineUndefinedDefinition) != 0)
      continue;
    for (int64_t definition : definitions)
      if (onlineInstructionTransitivelyDependsOn(cfg, definition, producer,
                                                 active)) {
        active.erase({consumer, producer});
        return true;
      }
  }
  active.erase({consumer, producer});
  return false;
}

bool onlineRoleTransitivelyDependsOn(const OnlinePtxCfg &cfg,
                                     StringRef consumerRole,
                                     StringRef producerRole) {
  auto consumer = cfg.roles.find(consumerRole.str());
  auto producer = cfg.roles.find(producerRole.str());
  if (consumer == cfg.roles.end() || producer == cfg.roles.end())
    return false;
  std::set<std::pair<int64_t, int64_t>> active;
  return onlineInstructionTransitivelyDependsOn(
      cfg, consumer->second, producer->second, active);
}

bool onlineDefinitionDependsOnSource(
    const OnlinePtxCfg &cfg, int64_t definition, int64_t source,
    std::set<std::pair<int64_t, int64_t>> &active) {
  if (definition == source)
    return true;
  if (definition == kOnlineUndefinedDefinition ||
      !active.insert({definition, source}).second)
    return false;
  const OnlinePtxInstruction &instruction = cfg.instructions[definition];
  for (const std::string &operand : llvm::drop_begin(instruction.operands)) {
    std::set<int64_t> reaching =
        onlineReachingDefinitions(cfg, definition, operand);
    if (reaching.empty() || reaching.count(kOnlineUndefinedDefinition) != 0)
      continue;
    if (llvm::all_of(reaching, [&](int64_t candidate) {
          return onlineDefinitionDependsOnSource(cfg, candidate, source,
                                                 active);
        })) {
      active.erase({definition, source});
      return true;
    }
  }
  active.erase({definition, source});
  return false;
}

bool onlineRoleHasSourceBoundOperand(const OnlinePtxCfg &cfg, StringRef role,
                                     StringRef source) {
  auto consumer = cfg.roles.find(role.str());
  auto producer = cfg.sources.find(source.str());
  if (consumer == cfg.roles.end() || producer == cfg.sources.end() ||
      producer->second.size() != 1)
    return false;
  const int64_t sourceInstruction = producer->second.front();
  const StringRef sourceOpcode = cfg.instructions[sourceInstruction].opcode;
  if (sourceOpcode != "fma.rn.f32" && sourceOpcode != "add.rn.f32" &&
      sourceOpcode != "mul.rn.f32")
    return false;
  if (consumer->second == sourceInstruction)
    return true;
  for (const std::string &operand :
       llvm::drop_begin(cfg.instructions[consumer->second].operands)) {
    std::set<int64_t> reaching =
        onlineReachingDefinitions(cfg, consumer->second, operand);
    if (reaching.empty() || reaching.count(kOnlineUndefinedDefinition) != 0)
      continue;
    std::set<std::pair<int64_t, int64_t>> active;
    if (llvm::all_of(reaching, [&](int64_t definition) {
          return onlineDefinitionDependsOnSource(cfg, definition,
                                                 sourceInstruction, active);
        }))
      return true;
  }
  return false;
}

bool onlineRoleOperandIs(const OnlinePtxCfg &cfg, StringRef consumerRole,
                         size_t operandIndex, StringRef producerRole) {
  auto consumer = cfg.roles.find(consumerRole.str());
  auto producer = cfg.roles.find(producerRole.str());
  if (consumer == cfg.roles.end() || producer == cfg.roles.end() ||
      operandIndex >= cfg.instructions[consumer->second].operands.size())
    return false;
  return onlineReachingDefinitions(
             cfg, consumer->second,
             cfg.instructions[consumer->second].operands[operandIndex]) ==
         std::set<int64_t>{producer->second};
}

bool onlineSourceOperandIs(const OnlinePtxCfg &cfg, int64_t consumer,
                           size_t operandIndex, StringRef source) {
  auto producer = cfg.sources.find(source.str());
  if (producer == cfg.sources.end() ||
      operandIndex >= cfg.instructions[consumer].operands.size())
    return false;
  std::set<int64_t> definitions = onlineReachingDefinitions(
      cfg, consumer, cfg.instructions[consumer].operands[operandIndex]);
  return definitions.size() == 1 &&
         llvm::is_contained(producer->second, *definitions.begin());
}

FailureOr<int64_t> onlineIotaLogicalDimension(RelationOp iota) {
  if (iota.getParameters().size() != 1)
    return failure();
  auto parameter = dyn_cast<IntegerAttr>(iota.getParameters()[0]);
  if (!parameter)
    return failure();
  int64_t dimension = parameter.getInt();
  Value value = iota.getResult(0);
  llvm::SmallDenseSet<Operation *> visited;
  while (true) {
    SmallVector<RelationOp> users;
    for (Operation *user : value.getUsers())
      if (auto relation = dyn_cast<RelationOp>(user))
        users.push_back(relation);
    if (users.size() != 1 || !visited.insert(users.front()).second)
      return failure();
    RelationOp user = users.front();
    if (user.getSemantics() == RelationSemantics::Compare)
      return dimension;
    if (user.getSemantics() == RelationSemantics::Reshape) {
      auto inputType = dyn_cast<RankedTensorType>(value.getType());
      auto outputType =
          dyn_cast<RankedTensorType>(user.getResult(0).getType());
      if (!inputType || !outputType || dimension < 0 ||
          dimension >= inputType.getRank())
        return failure();
      SmallVector<int64_t> candidates;
      for (int64_t output = 0; output < outputType.getRank(); ++output)
        if (outputType.getDimSize(output) == inputType.getDimSize(dimension) &&
            outputType.getDimSize(output) != 1)
          candidates.push_back(output);
      if (candidates.size() != 1)
        return failure();
      dimension = candidates.front();
    } else if (user.getSemantics() ==
               RelationSemantics::BroadcastInDim) {
      if (dimension < 0 || dimension >= user.getParameters().size())
        return failure();
      auto mapped = dyn_cast<IntegerAttr>(user.getParameters()[dimension]);
      if (!mapped)
        return failure();
      dimension = mapped.getInt();
    } else if (user.getSemantics() != RelationSemantics::Add) {
      return failure();
    }
    value = user.getResult(0);
  }
}

struct OnlineIndexAlternatives {
  bool valid = true;
  SmallVector<AffineExpr> values;
  SmallVector<AffineExpr> intermediates;
};

OnlineIndexAlternatives onlineIndexAlternatives(
    const OnlinePtxCfg &cfg, int64_t instruction, StringRef value,
    MLIRContext *context, std::set<std::pair<int64_t, std::string>> &active,
    const llvm::SmallDenseMap<int64_t, AffineExpr> *symbolicDefinitions =
        nullptr) {
  auto dimension = [&](unsigned ordinal) {
    return getAffineDimExpr(ordinal, context);
  };
  if (value == "%ctaid.x")
    return {true, {dimension(0)}};
  if (value == "%ctaid.y")
    return {true, {dimension(1)}};
  if (value == "%ctaid.z")
    return {true, {dimension(2)}};
  if (value == "%tid.x")
    return {true, {dimension(3)}};
  if (value == "%tid.y")
    return {true, {dimension(4)}};
  if (value == "%tid.z")
    return {true, {dimension(5)}};
  if (!value.starts_with("%")) {
    int64_t constant = 0;
    if (value.getAsInteger(10, constant))
      return {false, {}};
    return {true, {getAffineConstantExpr(constant, context)}};
  }
  if (!active.insert({instruction, value.str()}).second)
    return {false, {}};
  std::set<int64_t> definitions =
      onlineReachingDefinitions(cfg, instruction, value);
  OnlineIndexAlternatives result;
  for (int64_t definition : definitions) {
    if (definition == kOnlineUndefinedDefinition) {
      result.valid = false;
      continue;
    }
    if (symbolicDefinitions) {
      auto symbolic = symbolicDefinitions->find(definition);
      if (symbolic != symbolicDefinitions->end()) {
        result.values.push_back(symbolic->second);
        continue;
      }
    }
    const OnlinePtxInstruction &producer = cfg.instructions[definition];
    if (!producer.predicate.empty() || producer.operands.size() < 2) {
      result.valid = false;
      continue;
    }
    SmallVector<OnlineIndexAlternatives> operands;
    for (const std::string &operand : llvm::drop_begin(producer.operands))
      operands.push_back(
          onlineIndexAlternatives(cfg, definition, operand, context, active,
                                  symbolicDefinitions));
    if (llvm::any_of(operands, [](const OnlineIndexAlternatives &operand) {
          return !operand.valid;
        })) {
      result.valid = false;
      continue;
    }
    for (const OnlineIndexAlternatives &operand : operands)
      llvm::append_range(result.intermediates, operand.intermediates);
    StringRef opcode = producer.opcode;
    for (AffineExpr left : operands.front().values) {
      if (opcode == "mov.u32" || opcode == "mov.s32" ||
          opcode == "mov.b32") {
        if (operands.size() != 1)
          result.valid = false;
        else
          result.values.push_back(left);
        continue;
      }
      if (operands.size() != 2) {
        result.valid = false;
        continue;
      }
      for (AffineExpr right : operands[1].values) {
        auto leftConstant = dyn_cast<AffineConstantExpr>(left);
        auto rightConstant = dyn_cast<AffineConstantExpr>(right);
        if (opcode == "add.u32" || opcode == "add.s32") {
          AffineExpr value = left + right;
          result.values.push_back(value);
          result.intermediates.push_back(value);
        } else if ((opcode == "mul.lo.u32" || opcode == "mul.lo.s32") &&
                   (leftConstant || rightConstant)) {
          AffineExpr value = left * right;
          result.values.push_back(value);
          result.intermediates.push_back(value);
        } else if (opcode == "div.u32" && rightConstant &&
                   rightConstant.getValue() > 0) {
          AffineExpr value = left.floorDiv(rightConstant.getValue());
          result.values.push_back(value);
          result.intermediates.push_back(value);
        } else if (opcode == "rem.u32" && rightConstant &&
                   rightConstant.getValue() > 0) {
          AffineExpr value = left % rightConstant.getValue();
          result.values.push_back(value);
          result.intermediates.push_back(value);
        } else {
          result.valid = false;
        }
      }
    }
  }
  active.erase({instruction, value.str()});
  return result;
}

FailureOr<int64_t> evaluateOnlineAffineExpr(AffineExpr expression,
                                            ArrayRef<int64_t> dimensions) {
  if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
    if (dimension.getPosition() >= dimensions.size())
      return failure();
    return dimensions[dimension.getPosition()];
  }
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return constant.getValue();
  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return failure();
  FailureOr<int64_t> lhs =
      evaluateOnlineAffineExpr(binary.getLHS(), dimensions);
  FailureOr<int64_t> rhs =
      evaluateOnlineAffineExpr(binary.getRHS(), dimensions);
  if (failed(lhs) || failed(rhs))
    return failure();
  switch (binary.getKind()) {
  case AffineExprKind::Add:
    return *lhs + *rhs;
  case AffineExprKind::Mul:
    return *lhs * *rhs;
  case AffineExprKind::FloorDiv:
    if (*rhs <= 0)
      return failure();
    return *lhs / *rhs;
  case AffineExprKind::Mod:
    if (*rhs <= 0)
      return failure();
    return *lhs % *rhs;
  default:
    return failure();
  }
}

FailureOr<int64_t> evaluateOnlineIndexExpr(AffineExpr expression,
                                           ArrayRef<int64_t> dimensions) {
  FailureOr<int64_t> value;
  if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
    if (dimension.getPosition() >= dimensions.size())
      return failure();
    value = dimensions[dimension.getPosition()];
  } else if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
    value = constant.getValue();
  } else {
    auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
    if (!binary)
      return failure();
    FailureOr<int64_t> lhs =
        evaluateOnlineIndexExpr(binary.getLHS(), dimensions);
    FailureOr<int64_t> rhs =
        evaluateOnlineIndexExpr(binary.getRHS(), dimensions);
    if (failed(lhs) || failed(rhs))
      return failure();
    switch (binary.getKind()) {
    case AffineExprKind::Add:
      if (*lhs > std::numeric_limits<int32_t>::max() - *rhs)
        return failure();
      value = *lhs + *rhs;
      break;
    case AffineExprKind::Mul:
      if (*lhs != 0 && *rhs > std::numeric_limits<int32_t>::max() / *lhs)
        return failure();
      value = *lhs * *rhs;
      break;
    case AffineExprKind::FloorDiv:
      if (*rhs <= 0)
        return failure();
      value = *lhs / *rhs;
      break;
    case AffineExprKind::Mod:
      if (*rhs <= 0)
        return failure();
      value = *lhs % *rhs;
      break;
    default:
      return failure();
    }
  }
  if (failed(value) || *value < 0 ||
      *value > std::numeric_limits<int32_t>::max())
    return failure();
  return value;
}

bool onlineIndexExprFitsLaunchDomain(AffineExpr expression,
                                     ArrayRef<int64_t> launchExtents) {
  int64_t launches = 1;
  for (int64_t extent : launchExtents) {
    if (extent <= 0 || launches > 1000000 / extent)
      return false;
    launches *= extent;
  }
  for (int64_t linear = 0; linear < launches; ++linear) {
    int64_t remainder = linear;
    SmallVector<int64_t> physical(launchExtents.size());
    for (int64_t axis = launchExtents.size() - 1; axis >= 0; --axis) {
      physical[axis] = remainder % launchExtents[axis];
      remainder /= launchExtents[axis];
    }
    if (failed(evaluateOnlineIndexExpr(expression, physical)))
      return false;
  }
  return true;
}

bool onlineIndexExprFitsBuffer(AffineExpr expression,
                               ArrayRef<int64_t> launchExtents,
                               int64_t elementCount) {
  if (elementCount <= 0)
    return false;
  int64_t launches = 1;
  for (int64_t extent : launchExtents) {
    if (extent <= 0 || launches > 1000000 / extent)
      return false;
    launches *= extent;
  }
  for (int64_t linear = 0; linear < launches; ++linear) {
    int64_t remainder = linear;
    SmallVector<int64_t> physical(launchExtents.size());
    for (int64_t axis = launchExtents.size() - 1; axis >= 0; --axis) {
      physical[axis] = remainder % launchExtents[axis];
      remainder /= launchExtents[axis];
    }
    FailureOr<int64_t> index =
        evaluateOnlineIndexExpr(expression, physical);
    if (failed(index) || *index >= elementCount)
      return false;
  }
  return true;
}

bool onlineBlocksAreAcyclic(const OnlinePtxCfg &cfg,
                            const llvm::SmallDenseSet<int64_t> &blocks,
                            std::optional<std::pair<int64_t, int64_t>>
                                certifiedBackedge = std::nullopt) {
  llvm::SmallDenseMap<int64_t, int64_t> indegree;
  for (int64_t block : blocks)
    indegree.try_emplace(block, 0);
  for (int64_t block : blocks)
    for (int64_t successor : cfg.successors[block])
      if (blocks.contains(successor) &&
          certifiedBackedge !=
              std::pair<int64_t, int64_t>{block, successor})
        ++indegree[successor];
  SmallVector<int64_t> ready;
  for (auto [block, degree] : indegree)
    if (degree == 0)
      ready.push_back(block);
  int64_t visited = 0;
  while (!ready.empty()) {
    int64_t block = ready.pop_back_val();
    ++visited;
    for (int64_t successor : cfg.successors[block]) {
      if (!blocks.contains(successor) ||
          certifiedBackedge ==
              std::pair<int64_t, int64_t>{block, successor})
        continue;
      if (--indegree[successor] == 0)
        ready.push_back(successor);
    }
  }
  return visited == static_cast<int64_t>(blocks.size());
}

struct OnlineLogicalCoordinateProof {
  SmallVector<AffineExpr> coordinates;
  SmallVector<int64_t> launchExtents;
};

FailureOr<OnlineLogicalCoordinateProof> onlineLogicalCoordinateMaps(
    DeviceEntryOp entry, ScheduleTaskOp schedule) {
  auto maps = entry->getAttrOfType<ArrayAttr>("logical_coordinate_maps");
  if (!maps || maps.size() != schedule.getDomainShape().size())
    return failure();
  SmallVector<AffineExpr> coordinates;
  for (Attribute attribute : maps) {
    auto map = dyn_cast<AffineMapAttr>(attribute);
    if (!map || map.getValue().getNumDims() != 7 ||
        map.getValue().getNumSymbols() != 0 ||
        map.getValue().getNumResults() != 1)
      return failure();
    coordinates.push_back(map.getValue().getResult(0));
  }
  SmallVector<int64_t> grid(3, 1);
  SmallVector<int64_t> block(3, 1);
  llvm::copy(entry.getGrid().value_or(ArrayRef<int64_t>()), grid.begin());
  llvm::copy(entry.getBlock().value_or(ArrayRef<int64_t>()), block.begin());
  SmallVector<int64_t> launchExtents{
      grid[0],  grid[1],  grid[2], block[0], block[1], block[2],
      static_cast<int64_t>(schedule.getSerialTiles())};
  int64_t launches = 1;
  for (int64_t extent : launchExtents) {
    if (extent <= 0 || launches > 1000000 / extent)
      return failure();
    launches *= extent;
  }
  std::set<SmallVector<int64_t>> covered;
  for (int64_t linear = 0; linear < launches; ++linear) {
    int64_t remainder = linear;
    SmallVector<int64_t> physical(7);
    for (int64_t axis = 6; axis >= 0; --axis) {
      physical[axis] = remainder % launchExtents[axis];
      remainder /= launchExtents[axis];
    }
    SmallVector<int64_t> logical;
    bool inBounds = true;
    for (auto [axis, expression] : llvm::enumerate(coordinates)) {
      FailureOr<int64_t> coordinate =
          evaluateOnlineAffineExpr(expression, physical);
      if (failed(coordinate))
        return failure();
      inBounds &= *coordinate >= 0 &&
                  *coordinate < schedule.getDomainShape()[axis];
      logical.push_back(*coordinate);
    }
    if (inBounds && !covered.insert(logical).second)
      return failure();
  }
  int64_t domainElements = 1;
  for (int64_t extent : schedule.getDomainShape()) {
    if (extent <= 0 || domainElements > 1000000 / extent)
      return failure();
    domainElements *= extent;
  }
  if (covered.size() != static_cast<size_t>(domainElements))
    return failure();
  return OnlineLogicalCoordinateProof{coordinates, launchExtents};
}

FailureOr<SmallVector<int64_t>> onlineRelationDomainAxes(
    MaterializationTaskOp task, RegionOp region,
    ArrayRef<RelationOp> relations, int64_t relationRank) {
  Operation *anchor = nullptr;
  for (Operation &operation : region.getBody().front().without_terminator())
    if (operation.getAttr("source") == task.getSource())
      anchor = &operation;
  if (auto fold = dyn_cast_or_null<FoldOp>(anchor)) {
    if (fold.getInputs().empty() ||
        cast<RankedTensorType>(fold.getInputs().front().getType()).getRank() !=
            relationRank ||
        task.getDomainShape().size() != relationRank)
      return failure();
    SmallVector<int64_t> identity;
    for (int64_t axis = 0; axis < relationRank; ++axis)
      identity.push_back(axis);
    return identity;
  }
  auto contract = dyn_cast_or_null<ContractOp>(anchor);
  if (!contract || contract.getIndexingMaps().empty())
    return failure();
  std::optional<unsigned> scoreOperand;
  for (auto [operand, input] : llvm::enumerate(contract.getInputs())) {
    Value inputValue = input;
    if (!llvm::any_of(relations, [&](RelationOp relation) {
          return valueDependsOn(inputValue, relation);
        }))
      continue;
    if (scoreOperand)
      return failure();
    scoreOperand = operand;
  }
  if (!scoreOperand || *scoreOperand >= contract.getIndexingMaps().size())
    return failure();
  AffineMap scoreMap = cast<AffineMapAttr>(
                           contract.getIndexingMaps()[*scoreOperand])
                           .getValue();
  if (scoreMap.getNumResults() != relationRank ||
      scoreMap.getNumDims() != task.getDomainShape().size())
    return failure();
  SmallVector<int64_t> axes;
  for (AffineExpr result : scoreMap.getResults()) {
    auto dimension = dyn_cast<AffineDimExpr>(result);
    if (!dimension)
      return failure();
    axes.push_back(dimension.getPosition());
  }
  return axes;
}

bool onlineRoleIs(const OnlinePtxCfg &cfg, StringRef role,
                  ArrayRef<StringRef> opcodes,
                  ArrayRef<StringRef> dependencies = {}) {
  auto found = cfg.roles.find(role.str());
  if (found == cfg.roles.end())
    return false;
  const OnlinePtxInstruction &instruction = cfg.instructions[found->second];
  if (!onlineBlockReachable(cfg, 0, instruction.block))
    return false;
  if (!llvm::any_of(opcodes, [&](StringRef opcode) {
        return StringRef(instruction.opcode).starts_with(opcode);
      }))
    return false;
  return llvm::all_of(dependencies, [&](StringRef dependency) {
    return onlineRoleDirectlyReaches(cfg, dependency, found->second);
  });
}

FailureOr<std::string> onlineComparisonOpcode(RelationOp relation);

bool onlineRelationSourceClosure(
    const OnlinePtxCfg &cfg, ArrayRef<RelationOp> relations,
    ArrayRef<AffineExpr> logicalCoordinates,
    ArrayRef<int64_t> launchExtents,
    ArrayRef<int64_t> relationDomainAxes, MLIRContext *context,
    DictionaryAttr reductionCertificate = DictionaryAttr(),
    std::optional<int64_t> reductionDimension = std::nullopt) {
  llvm::SmallDenseSet<int64_t> certifiedLoopCoordinates;
  if (reductionCertificate) {
    if (auto nodes = reductionCertificate.getAs<ArrayAttr>("nodes"))
      for (Attribute attribute : nodes) {
        auto node = dyn_cast<DictionaryAttr>(attribute);
        auto kind = node ? node.getAs<StringAttr>("kind") : StringAttr();
        auto realization = node && kind && kind.getValue() == "leaf_family"
                               ? node.getAs<DictionaryAttr>("realization")
                               : DictionaryAttr();
        auto coordinate =
            realization
                ? realization.getAs<StringAttr>("coordinate_role")
                : StringAttr();
        auto found = coordinate
                         ? cfg.roles.find(coordinate.getValue().str())
                         : cfg.roles.end();
        if (found != cfg.roles.end())
          certifiedLoopCoordinates.insert(found->second);
      }
  }
  for (RelationOp relation : relations) {
    SourceRefAttr source = relation->getAttrOfType<SourceRefAttr>("source");
    auto occurrences = cfg.sources.find(onlineSourceKey(source));
    if (!source || occurrences == cfg.sources.end() ||
        occurrences->second.empty())
      return false;
    llvm::SmallDenseSet<int64_t> validatedLoopCoordinates;
    for (int64_t sourceInstruction : occurrences->second) {
      const OnlinePtxInstruction &ptx = cfg.instructions[sourceInstruction];
      if (!ptx.predicate.empty())
        return false;
      auto relationOperandSource = [&](size_t index, size_t ptxIndex) {
        if (index >= relation.getInputs().size())
          return false;
        auto producer = relation.getInputs()[index].getDefiningOp<RelationOp>();
        SourceRefAttr operandSource =
            producer ? producer->getAttrOfType<SourceRefAttr>("source")
                     : SourceRefAttr();
        return operandSource &&
               onlineSourceOperandIs(cfg, sourceInstruction, ptxIndex,
                                     onlineSourceKey(operandSource));
      };
      switch (relation.getSemantics()) {
    case RelationSemantics::Iota: {
      FailureOr<int64_t> logicalDimension =
          onlineIotaLogicalDimension(relation);
      if (!StringRef(ptx.opcode).starts_with("mov.u32") &&
          !StringRef(ptx.opcode).starts_with("mov.s32"))
        return false;
      if (failed(logicalDimension) || *logicalDimension < 0 ||
          *logicalDimension >= relationDomainAxes.size() ||
          relationDomainAxes[*logicalDimension] < 0 ||
          relationDomainAxes[*logicalDimension] >= logicalCoordinates.size() ||
          ptx.operands.size() != 2 ||
          !StringRef(ptx.operands[1]).starts_with("%"))
        return false;
      std::set<std::pair<int64_t, std::string>> active;
      OnlineIndexAlternatives alternatives = onlineIndexAlternatives(
          cfg, sourceInstruction, ptx.operands[1], context, active);
      AffineExpr expected =
          logicalCoordinates[relationDomainAxes[*logicalDimension]];
      const bool physicalCoordinate =
          alternatives.valid && !alternatives.values.empty() &&
          llvm::all_of(alternatives.values, [&](AffineExpr value) {
            return value == expected &&
                   onlineIndexExprFitsLaunchDomain(value, launchExtents);
          }) &&
          llvm::all_of(alternatives.intermediates, [&](AffineExpr value) {
            return onlineIndexExprFitsLaunchDomain(value, launchExtents);
          });
      const bool loopCoordinate =
          reductionDimension &&
          relationDomainAxes[*logicalDimension] == *reductionDimension &&
          certifiedLoopCoordinates.contains(sourceInstruction);
      if (!physicalCoordinate && !loopCoordinate)
        return false;
      if (loopCoordinate)
        validatedLoopCoordinates.insert(sourceInstruction);
      break;
    }
    case RelationSemantics::Constant: {
      if (!StringRef(ptx.opcode).starts_with("mov.") ||
          relation.getParameters().size() != 1)
        return false;
      auto value = dyn_cast<IntegerAttr>(relation.getParameters()[0]);
      if (!value || !llvm::is_contained(ptx.operands,
                                        Twine(value.getInt()).str()))
        return false;
      break;
    }
    case RelationSemantics::Reshape:
    case RelationSemantics::BroadcastInDim:
      if (!StringRef(ptx.opcode).starts_with("mov.") ||
          !relationOperandSource(0, 1))
        return false;
      break;
    case RelationSemantics::Add:
      if ((!StringRef(ptx.opcode).starts_with("add.s32") &&
           !StringRef(ptx.opcode).starts_with("add.u32")) ||
          !relationOperandSource(0, 1) || !relationOperandSource(1, 2))
        return false;
      break;
    case RelationSemantics::Compare: {
      FailureOr<std::string> opcode = onlineComparisonOpcode(relation);
      if (failed(opcode) || !StringRef(ptx.opcode).starts_with(*opcode) ||
          !relationOperandSource(0, 1) || !relationOperandSource(1, 2))
        return false;
      break;
    }
    case RelationSemantics::Convert: {
      const bool select =
          StringRef(ptx.opcode) == "selp.f32" &&
          ptx.operands.size() == 4 &&
          ptx.operands[1] == "0f3F800000" &&
          ptx.operands[2] == "0f00000000" &&
          relationOperandSource(0, 3);
      if (!select)
        return false;
      break;
    }
    case RelationSemantics::Select: {
      if (!StringRef(ptx.opcode).starts_with("selp.f32"))
        return false;
      auto predicate = relation.getInputs().front().getDefiningOp<RelationOp>();
      SourceRefAttr predicateSource =
          predicate ? predicate->getAttrOfType<SourceRefAttr>("source")
                    : SourceRefAttr();
      if (!predicateSource ||
          !onlineSourceOperandIs(cfg, sourceInstruction, 3,
                                 onlineSourceKey(predicateSource)))
        return false;
      break;
    }
      }
    if (relation.getSemantics() == RelationSemantics::Iota) {
      FailureOr<int64_t> logicalDimension = onlineIotaLogicalDimension(relation);
      if (failed(logicalDimension) || *logicalDimension < 0 ||
          *logicalDimension >= relationDomainAxes.size())
        return false;
      if (reductionDimension &&
          relationDomainAxes[*logicalDimension] == *reductionDimension &&
          validatedLoopCoordinates != certifiedLoopCoordinates)
        return false;
    }
  }
  }
  return true;
}

struct OnlinePointerAlternative {
  std::optional<std::string> base;
};

struct OnlinePointerAlternatives {
  bool valid = true;
  std::vector<OnlinePointerAlternative> values;
};

OnlinePointerAlternatives onlinePointerAlternatives(
    const OnlinePtxCfg &cfg, int64_t instruction, StringRef value,
    std::set<std::pair<int64_t, std::string>> &active) {
  if (value.starts_with("p"))
    return {true, {{{value.str()}}}};
  if (!value.starts_with("%"))
    return {true, {{{std::nullopt}}}};
  if (!active.insert({instruction, value.str()}).second)
    return {false, {}};
  std::set<int64_t> definitions =
      onlineReachingDefinitions(cfg, instruction, value);
  if (definitions.empty()) {
    const bool builtin = value.starts_with("%tid") ||
                         value.starts_with("%ctaid") ||
                         value.starts_with("%ntid") ||
                         value.starts_with("%laneid");
    active.erase({instruction, value.str()});
    return {builtin, builtin
                         ? std::vector<OnlinePointerAlternative>{{std::nullopt}}
                         : std::vector<OnlinePointerAlternative>{}};
  }
  OnlinePointerAlternatives result;
  for (int64_t definition : definitions) {
    if (definition == kOnlineUndefinedDefinition) {
      result.valid = false;
      continue;
    }
    const OnlinePtxInstruction &producer = cfg.instructions[definition];
    if (!producer.predicate.empty()) {
      result.valid = false;
      continue;
    }
    std::vector<std::vector<OnlinePointerAlternative>> operandAlternatives;
    for (const std::string &operand : llvm::drop_begin(producer.operands)) {
      OnlinePointerAlternatives child =
          onlinePointerAlternatives(cfg, definition, operand, active);
      if (!child.valid) {
        result.valid = false;
        operandAlternatives.clear();
        break;
      }
      operandAlternatives.push_back(std::move(child.values));
    }
    if (operandAlternatives.empty())
      continue;
    std::vector<std::vector<OnlinePointerAlternative>> combinations{{}};
    for (const auto &alternatives : operandAlternatives) {
      std::vector<std::vector<OnlinePointerAlternative>> next;
      for (const auto &combination : combinations)
        for (const OnlinePointerAlternative &alternative : alternatives) {
          auto joined = combination;
          joined.push_back(alternative);
          next.push_back(std::move(joined));
        }
      combinations = std::move(next);
    }
    StringRef opcode = producer.opcode;
    for (const auto &combination : combinations) {
      SmallVector<StringRef> bases;
      for (const OnlinePointerAlternative &operand : combination)
        if (operand.base)
          bases.push_back(*operand.base);
      std::optional<std::string> outputBase;
      if (opcode.starts_with("ld.param.u64")) {
        if (combination.size() != 1 || bases.size() != 1)
          result.valid = false;
        else
          outputBase = bases.front().str();
      } else if (opcode.starts_with("mov.u64") ||
                 opcode.starts_with("cvta.to.global.u64")) {
        if (combination.size() != 1)
          result.valid = false;
        else
          outputBase = combination.front().base;
      } else if (opcode.starts_with("add.u64") ||
                 opcode.starts_with("add.s64")) {
        if (bases.size() > 1)
          result.valid = false;
        else if (bases.size() == 1)
          outputBase = bases.front().str();
      } else if (opcode.starts_with("mov.u32") ||
                 opcode.starts_with("cvt.u64.u32") ||
                 opcode.starts_with("cvt.u32.") ||
                 opcode.starts_with("add.u32") ||
                 opcode.starts_with("add.s32") ||
                 opcode.starts_with("mul.lo.u32") ||
                 opcode.starts_with("div.u32") ||
                 opcode.starts_with("rem.u32") ||
                 opcode.starts_with("mad.lo.u32") ||
                 opcode.starts_with("mad.wide.u32") ||
                 opcode.starts_with("shl.b32") ||
                 opcode.starts_with("and.b32") ||
                 opcode.starts_with("or.b32")) {
        if (!bases.empty())
          result.valid = false;
      } else {
        result.valid = false;
      }
      if (result.valid)
        result.values.push_back({std::move(outputBase)});
    }
  }
  active.erase({instruction, value.str()});
  return result;
}

bool onlinePointerIsOutputParameter(const OnlinePtxCfg &cfg,
                                    int64_t instruction, StringRef pointer,
                                    int64_t parameterIndex) {
  std::set<std::pair<int64_t, std::string>> active;
  OnlinePointerAlternatives alternatives =
      onlinePointerAlternatives(cfg, instruction, pointer, active);
  const std::string expected = (Twine("p") + Twine(parameterIndex)).str();
  return alternatives.valid && !alternatives.values.empty() &&
         llvm::all_of(alternatives.values,
                      [&](const OnlinePointerAlternative &alternative) {
                        return alternative.base == expected;
                      });
}

std::string onlineAddressFingerprint(const OnlinePtxCfg &cfg,
                                     int64_t instruction, StringRef value,
                                     std::set<std::pair<int64_t, std::string>>
                                         &active) {
  if (!value.starts_with("%"))
    return value.str();
  if (!active.insert({instruction, value.str()}).second)
    return "$cycle";
  std::set<std::string> alternatives;
  for (int64_t definition :
       onlineReachingDefinitions(cfg, instruction, value)) {
    if (definition == kOnlineUndefinedDefinition) {
      alternatives.insert("$undef");
      continue;
    }
    const OnlinePtxInstruction &producer = cfg.instructions[definition];
    std::string fingerprint = producer.opcode;
    for (const std::string &operand : llvm::drop_begin(producer.operands)) {
      fingerprint += "(";
      fingerprint += onlineAddressFingerprint(cfg, definition, operand, active);
      fingerprint += ")";
    }
    alternatives.insert(std::move(fingerprint));
  }
  active.erase({instruction, value.str()});
  return (Twine("{") + llvm::join(alternatives, "|") + "}").str();
}

bool onlineInstructionReachable(const OnlinePtxCfg &cfg, int64_t from,
                                int64_t to) {
  const OnlinePtxInstruction &fromInstruction = cfg.instructions[from];
  const OnlinePtxInstruction &toInstruction = cfg.instructions[to];
  if (fromInstruction.block == toInstruction.block)
    return from <= to;
  return llvm::any_of(cfg.successors[fromInstruction.block],
                      [&](int64_t successor) {
                        return onlineBlockReachable(
                            cfg, successor, toInstruction.block);
                      });
}

bool onlineInstructionDominates(const OnlinePtxCfg &cfg, int64_t dominator,
                                int64_t dominated) {
  const OnlinePtxInstruction &left = cfg.instructions[dominator];
  const OnlinePtxInstruction &right = cfg.instructions[dominated];
  if (!onlineBlockReachable(cfg, 0, left.block) ||
      !onlineBlockReachable(cfg, 0, right.block))
    return false;
  if (left.block == right.block)
    return dominator <= dominated;
  std::set<int64_t> seen{left.block};
  std::vector<int64_t> worklist{0};
  while (!worklist.empty()) {
    int64_t block = worklist.back();
    worklist.pop_back();
    if (!seen.insert(block).second)
      continue;
    if (block == right.block)
      return false;
    llvm::append_range(worklist, cfg.successors[block]);
  }
  return true;
}

bool onlineSharedPairIsClosed(const OnlinePtxCfg &cfg, StringRef storeRole,
                              StringRef producerRole, StringRef barrierRole,
                              StringRef loadRole) {
  auto store = cfg.roles.find(storeRole.str());
  auto barrier = cfg.roles.find(barrierRole.str());
  auto load = cfg.roles.find(loadRole.str());
  if (store == cfg.roles.end() || barrier == cfg.roles.end() ||
      load == cfg.roles.end())
    return false;
  const OnlinePtxInstruction &storeInstruction =
      cfg.instructions[store->second];
  const OnlinePtxInstruction &barrierInstruction =
      cfg.instructions[barrier->second];
  const OnlinePtxInstruction &loadInstruction = cfg.instructions[load->second];
  StringRef barrierOpcode = barrierInstruction.opcode;
  const bool synchronizingBarrier =
      barrierOpcode.starts_with("bar.sync") ||
      barrierOpcode.starts_with("bar.warp.sync") ||
      barrierOpcode.starts_with("barrier.cluster.wait");
  if (!StringRef(storeInstruction.opcode).starts_with("st.shared.") ||
      !synchronizingBarrier ||
      !StringRef(loadInstruction.opcode).starts_with("ld.shared.") ||
      !storeInstruction.predicate.empty() ||
      !barrierInstruction.predicate.empty() ||
      !loadInstruction.predicate.empty() ||
      storeInstruction.operands.size() < 2 ||
      loadInstruction.operands.size() < 2 ||
      !onlineRoleOperandIs(cfg, storeRole, 1, producerRole))
    return false;
  std::set<std::pair<int64_t, std::string>> storeActive;
  std::set<std::pair<int64_t, std::string>> loadActive;
  if (onlineAddressFingerprint(cfg, store->second,
                               storeInstruction.operands[0], storeActive) !=
      onlineAddressFingerprint(cfg, load->second,
                               loadInstruction.operands[1], loadActive))
    return false;
  std::set<std::pair<int64_t, std::string>> storePredicateActive;
  std::set<std::pair<int64_t, std::string>> loadPredicateActive;
  if (storeInstruction.negatedPredicate != loadInstruction.negatedPredicate ||
      onlineAddressFingerprint(cfg, store->second, storeInstruction.predicate,
                               storePredicateActive) !=
          onlineAddressFingerprint(cfg, load->second,
                                   loadInstruction.predicate,
                                   loadPredicateActive))
    return false;
  if (!onlineInstructionDominates(cfg, store->second, barrier->second) ||
      !onlineInstructionDominates(cfg, barrier->second, load->second))
    return false;
  const std::string address = onlineAddressFingerprint(
      cfg, store->second, storeInstruction.operands[0], storeActive);
  for (const OnlinePtxInstruction &instruction : cfg.instructions) {
    if (instruction.ordinal == store->second ||
        !StringRef(instruction.opcode).starts_with("st.shared.") ||
        instruction.operands.size() < 2 ||
        !onlineInstructionReachable(cfg, store->second, instruction.ordinal) ||
        !onlineInstructionReachable(cfg, instruction.ordinal, load->second))
      continue;
    std::set<std::pair<int64_t, std::string>> active;
    if (onlineAddressFingerprint(cfg, instruction.ordinal,
                                 instruction.operands[0], active) == address)
      return false;
  }
  return true;
}

bool onlineRoleStoresToOutput(const OnlinePtxCfg &cfg, StringRef role,
                              int64_t parameterIndex) {
  auto found = cfg.roles.find(role.str());
  if (found == cfg.roles.end())
    return false;
  const OnlinePtxInstruction &store = cfg.instructions[found->second];
  if (!StringRef(store.opcode).starts_with("st.global.") ||
      store.operands.size() < 2 || !store.predicate.empty())
    return false;
  return onlinePointerIsOutputParameter(cfg, found->second, store.operands[0],
                                        parameterIndex);
}

FailureOr<std::string> onlineComparisonOpcode(RelationOp relation) {
  ArrayAttr parameters = relation.getParameters();
  if (!parameters || parameters.size() != 2)
    return failure();
  auto type = dyn_cast<StringAttr>(parameters[0]);
  auto direction = dyn_cast<StringAttr>(parameters[1]);
  if (!type || !direction)
    return failure();
  StringRef suffix =
      type.getValue() == "SIGNED"
          ? "s32"
          : type.getValue() == "UNSIGNED" ? "u32" : StringRef();
  StringRef predicate =
      llvm::StringSwitch<StringRef>(direction.getValue())
          .Case("EQ", "eq")
          .Case("NE", "ne")
          .Case("GE", "ge")
          .Case("GT", "gt")
          .Case("LE", "le")
          .Case("LT", "lt")
          .Default("");
  if (suffix.empty() || predicate.empty())
    return failure();
  return (Twine("setp.") + predicate + "." + suffix).str();
}

bool onlineRoleIsExactFloatConstant(const OnlinePtxCfg &cfg, StringRef role,
                                    StringRef bits) {
  auto found = cfg.roles.find(role.str());
  if (found == cfg.roles.end())
    return false;
  int64_t instruction = found->second;
  llvm::SmallDenseSet<int64_t> active;
  while (active.insert(instruction).second) {
    const OnlinePtxInstruction &producer = cfg.instructions[instruction];
    if (!producer.predicate.empty() || producer.operands.size() != 2 ||
        (producer.opcode != "mov.f32" && producer.opcode != "mov.b32"))
      return false;
    if (producer.operands[1] == bits)
      return true;
    std::set<int64_t> definitions = onlineReachingDefinitions(
        cfg, instruction, producer.operands[1]);
    if (definitions.size() != 1 ||
        definitions.count(kOnlineUndefinedDefinition))
      return false;
    instruction = *definitions.begin();
  }
  return false;
}

bool onlineMaskedScoreProtocol(const OnlinePtxCfg &cfg,
                               StringRef comparisonOpcode) {
  auto compare = cfg.roles.find("relation_compare");
  return compare != cfg.roles.end() &&
         StringRef(cfg.instructions[compare->second].opcode)
             .starts_with(comparisonOpcode) &&
         onlineRoleIs(cfg, "relation_compare", {"setp."},
                      {"relation_lhs", "relation_rhs"}) &&
         onlineRoleOperandIs(cfg, "relation_compare", 1, "relation_lhs") &&
         onlineRoleOperandIs(cfg, "relation_compare", 2, "relation_rhs") &&
         onlineRoleIs(cfg, "negative_infinity", {"mov.f32"}) &&
         onlineRoleIsExactFloatConstant(cfg, "negative_infinity",
                                        "0fFF800000") &&
         onlineRoleIs(cfg, "mask_select", {"selp."},
                      {"score_value", "negative_infinity"}) &&
         onlineRoleOperandIs(cfg, "mask_select", 1, "score_value") &&
         onlineRoleOperandIs(cfg, "mask_select", 2, "negative_infinity") &&
         onlineRoleTransitivelyDependsOn(cfg, "mask_select",
                                         "relation_compare");
}

bool onlineCompactOutputAddressIsProjected(
    const OnlinePtxCfg &cfg, int64_t outputParameter,
    ArrayRef<AffineExpr> logicalCoordinates, ArrayRef<int64_t> launchExtents,
    ArrayRef<int64_t> scoreShape, int64_t reductionDimension,
    MLIRContext *context) {
  auto linear = cfg.roles.find("compact_output_linear_index");
  auto byteOffset = cfg.roles.find("compact_output_byte_offset");
  auto pointer = cfg.roles.find("compact_output_pointer");
  auto store = cfg.roles.find("compact_store");
  if (linear == cfg.roles.end() || byteOffset == cfg.roles.end() ||
      pointer == cfg.roles.end() || store == cfg.roles.end() ||
      logicalCoordinates.size() != scoreShape.size() ||
      reductionDimension < 0 ||
      reductionDimension >= static_cast<int64_t>(scoreShape.size()))
    return false;
  const OnlinePtxInstruction &byteOffsetInstruction =
      cfg.instructions[byteOffset->second];
  const OnlinePtxInstruction &pointerInstruction =
      cfg.instructions[pointer->second];
  const OnlinePtxInstruction &storeInstruction = cfg.instructions[store->second];
  if (byteOffsetInstruction.opcode != "mul.wide.u32" ||
      byteOffsetInstruction.operands.size() != 3 ||
      byteOffsetInstruction.operands[2] != "4" ||
      !onlineRoleOperandIs(cfg, "compact_output_byte_offset", 1,
                           "compact_output_linear_index") ||
      pointerInstruction.opcode != "add.u64" ||
      pointerInstruction.operands.size() != 3 ||
      storeInstruction.operands.size() < 2 ||
      !onlineRoleOperandIs(cfg, "compact_store", 0,
                           "compact_output_pointer"))
    return false;
  const bool baseThenOffset =
      onlinePointerIsOutputParameter(cfg, pointer->second,
                                     pointerInstruction.operands[1],
                                     outputParameter) &&
      onlineRoleOperandIs(cfg, "compact_output_pointer", 2,
                          "compact_output_byte_offset");
  const bool offsetThenBase =
      onlinePointerIsOutputParameter(cfg, pointer->second,
                                     pointerInstruction.operands[2],
                                     outputParameter) &&
      onlineRoleOperandIs(cfg, "compact_output_pointer", 1,
                          "compact_output_byte_offset");
  if (!baseThenOffset && !offsetThenBase)
    return false;
  std::set<std::pair<int64_t, std::string>> active;
  OnlineIndexAlternatives alternatives = onlineIndexAlternatives(
      cfg, byteOffset->second, byteOffsetInstruction.operands[1], context,
      active);
  AffineExpr expected = getAffineConstantExpr(0, context);
  for (auto [dimension, extent] : llvm::enumerate(scoreShape)) {
    if (static_cast<int64_t>(dimension) == reductionDimension)
      continue;
    if (extent <= 0)
      return false;
    expected = expected * extent + logicalCoordinates[dimension];
  }
  return alternatives.valid && !alternatives.values.empty() &&
         llvm::all_of(alternatives.values, [&](AffineExpr value) {
           return value == expected &&
                  onlineIndexExprFitsLaunchDomain(value, launchExtents);
         }) &&
         llvm::all_of(alternatives.intermediates, [&](AffineExpr value) {
           return onlineIndexExprFitsLaunchDomain(value, launchExtents);
         });
}

bool onlineCompactWriterIsUnique(
    const OnlinePtxCfg &cfg, ArrayRef<AffineExpr> logicalCoordinates,
    ArrayRef<int64_t> launchExtents, ArrayRef<int64_t> scoreShape,
    int64_t reductionDimension, MLIRContext *context,
    DictionaryAttr writerCertificate = DictionaryAttr()) {
  auto store = cfg.roles.find("compact_store");
  if (store == cfg.roles.end() || logicalCoordinates.size() != scoreShape.size())
    return false;
  const OnlinePtxInstruction &storeInstruction = cfg.instructions[store->second];
  std::optional<AffineExpr> selector;
  llvm::SmallDenseMap<int64_t, int64_t> boundPredicates;
  if (!storeInstruction.predicate.empty()) {
    auto finalName = writerCertificate
                         ? writerCertificate.getAs<StringAttr>("predicate_role")
                         : StringAttr();
    auto selectorName = writerCertificate
                            ? writerCertificate.getAs<StringAttr>("selector_role")
                            : StringAttr();
    auto selectorPredicateName =
        writerCertificate
            ? writerCertificate.getAs<StringAttr>("selector_predicate_role")
            : StringAttr();
    auto bounds = writerCertificate
                      ? writerCertificate.getAs<ArrayAttr>("bounds")
                      : ArrayAttr();
    auto finalPredicate = finalName
                              ? cfg.roles.find(finalName.getValue().str())
                              : cfg.roles.end();
    auto predicate = selectorPredicateName
                         ? cfg.roles.find(selectorPredicateName.getValue().str())
                         : cfg.roles.end();
    auto selectorRole = selectorName
                            ? cfg.roles.find(selectorName.getValue().str())
                            : cfg.roles.end();
    if (!bounds || finalPredicate == cfg.roles.end() ||
        predicate == cfg.roles.end() || selectorRole == cfg.roles.end() ||
        storeInstruction.negatedPredicate ||
        onlineReachingDefinitions(cfg, store->second,
                                  storeInstruction.predicate) !=
            std::set<int64_t>{finalPredicate->second})
      return false;
    const OnlinePtxInstruction &predicateInstruction =
        cfg.instructions[predicate->second];
    if ((predicateInstruction.opcode != "setp.eq.u32" &&
         predicateInstruction.opcode != "setp.eq.s32") ||
        predicateInstruction.operands.size() != 3 ||
        predicateInstruction.operands[2] != "0" ||
        onlineReachingDefinitions(cfg, predicate->second,
                                  predicateInstruction.operands[1]) !=
            std::set<int64_t>{selectorRole->second})
      return false;
    std::set<std::pair<int64_t, std::string>> active;
    OnlineIndexAlternatives alternatives = onlineIndexAlternatives(
        cfg, predicate->second, predicateInstruction.operands[1], context,
        active);
    if (!alternatives.valid || alternatives.values.size() != 1 ||
        llvm::any_of(alternatives.intermediates, [&](AffineExpr expression) {
          return !onlineIndexExprFitsLaunchDomain(expression, launchExtents);
        }))
      return false;
    selector = alternatives.values.front();
    for (Attribute attribute : bounds) {
      auto bound = dyn_cast<DictionaryAttr>(attribute);
      auto dimension = bound ? bound.getAs<IntegerAttr>("dimension")
                             : IntegerAttr();
      auto roleName = bound ? bound.getAs<StringAttr>("predicate_role")
                            : StringAttr();
      auto role = roleName ? cfg.roles.find(roleName.getValue().str())
                           : cfg.roles.end();
      if (!dimension || dimension.getInt() < 0 ||
          dimension.getInt() >= scoreShape.size() ||
          dimension.getInt() == reductionDimension ||
          role == cfg.roles.end() ||
          !boundPredicates.try_emplace(dimension.getInt(), role->second).second)
        return false;
      const OnlinePtxInstruction &boundInstruction =
          cfg.instructions[role->second];
      if ((boundInstruction.opcode != "setp.lt.u32" &&
           boundInstruction.opcode != "setp.lt.s32") ||
          boundInstruction.operands.size() != 3 ||
          boundInstruction.operands[2] !=
              Twine(scoreShape[dimension.getInt()]).str())
        return false;
      std::set<std::pair<int64_t, std::string>> boundActive;
      OnlineIndexAlternatives boundAlternatives = onlineIndexAlternatives(
          cfg, role->second, boundInstruction.operands[1], context,
          boundActive);
      if (!boundAlternatives.valid || boundAlternatives.values.empty() ||
          llvm::any_of(boundAlternatives.values, [&](AffineExpr value) {
            return value != logicalCoordinates[dimension.getInt()] ||
                   !onlineIndexExprFitsLaunchDomain(value, launchExtents);
          }))
        return false;
    }
    for (int64_t dimension = 0; dimension < scoreShape.size(); ++dimension)
      if (dimension != reductionDimension &&
          !boundPredicates.contains(dimension))
        return false;
    llvm::SmallDenseSet<int64_t> expectedPredicates{predicate->second};
    for (auto [dimension, instruction] : boundPredicates)
      expectedPredicates.insert(instruction);
    llvm::SmallDenseSet<int64_t> foundPredicates;
    llvm::SmallDenseSet<int64_t> activePredicates;
    std::function<bool(int64_t)> collect = [&](int64_t instruction) {
      if (expectedPredicates.contains(instruction))
        return foundPredicates.insert(instruction).second;
      if (!activePredicates.insert(instruction).second)
        return false;
      const OnlinePtxInstruction &candidate = cfg.instructions[instruction];
      if (candidate.opcode != "and.pred" || candidate.operands.size() != 3)
        return false;
      for (size_t operand = 1; operand <= 2; ++operand) {
        std::set<int64_t> definitions = onlineReachingDefinitions(
            cfg, instruction, candidate.operands[operand]);
        if (definitions.size() != 1 || !collect(*definitions.begin()))
          return false;
      }
      activePredicates.erase(instruction);
      return true;
    };
    if (!collect(finalPredicate->second) ||
        foundPredicates != expectedPredicates)
      return false;
  }

  int64_t launchInstances = 1;
  for (int64_t extent : launchExtents) {
    if (extent <= 0 || launchInstances > 1000000 / extent)
      return false;
    launchInstances *= extent;
  }
  std::map<std::vector<int64_t>, std::pair<int64_t, int64_t>> fibers;
  for (int64_t linear = 0; linear < launchInstances; ++linear) {
    int64_t remainder = linear;
    SmallVector<int64_t> physical(launchExtents.size());
    for (int64_t axis = launchExtents.size() - 1; axis >= 0; --axis) {
      physical[axis] = remainder % launchExtents[axis];
      remainder /= launchExtents[axis];
    }
    std::vector<int64_t> output;
    bool outputInBounds = true;
    for (auto [dimension, expression] : llvm::enumerate(logicalCoordinates)) {
      if (static_cast<int64_t>(dimension) == reductionDimension)
        continue;
      FailureOr<int64_t> coordinate =
          evaluateOnlineAffineExpr(expression, physical);
      if (failed(coordinate))
        return false;
      outputInBounds &= *coordinate >= 0 &&
                        *coordinate < scoreShape[dimension];
      output.push_back(*coordinate);
    }
    if (!outputInBounds) {
      if (!selector)
        return false;
      continue;
    }
    auto &fiber = fibers[output];
    ++fiber.first;
    if (!selector) {
      ++fiber.second;
      continue;
    }
    FailureOr<int64_t> selected =
        evaluateOnlineIndexExpr(*selector, physical);
    if (failed(selected))
      return false;
    if (*selected == 0)
      ++fiber.second;
  }
  int64_t expectedOutputs = 1;
  for (auto [dimension, extent] : llvm::enumerate(scoreShape)) {
    if (static_cast<int64_t>(dimension) == reductionDimension)
      continue;
    if (extent <= 0 || expectedOutputs > 1000000 / extent)
      return false;
    expectedOutputs *= extent;
  }
  return fibers.size() == static_cast<size_t>(expectedOutputs) &&
         llvm::all_of(fibers, [](const auto &item) {
           return item.second.first > 0 && item.second.second == 1;
         });
}

struct OnlineMaxReductionNode {
  int64_t id = -1;
  std::string kind;
  std::string role;
  llvm::SmallDenseSet<int64_t> coverage;
  SmallVector<int64_t> children;
  DictionaryAttr realization;
};

struct OnlineScoreContractGeometry {
  SmallVector<AffineExpr> inputLinearIndices;
  SmallVector<int64_t> inputElementCounts;
  int64_t reductionExtent = 0;
};

bool onlineValueDependsOnOperation(Value value, Operation *operation,
                                   llvm::SmallDenseSet<Operation *> &active) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  if (definition == operation)
    return true;
  if (!active.insert(definition).second)
    return false;
  const bool depends = llvm::any_of(definition->getOperands(), [&](Value input) {
    return onlineValueDependsOnOperation(input, operation, active);
  });
  active.erase(definition);
  return depends;
}

FailureOr<SmallVector<AffineExpr>> onlineScoreContractOutputCoordinates(
    Value scoreValue, ContractOp contract,
    ArrayRef<AffineExpr> scoreCoordinates, MLIRContext *context) {
  if (scoreValue.getDefiningOp() == contract.getOperation())
    return SmallVector<AffineExpr>(scoreCoordinates.begin(),
                                   scoreCoordinates.end());
  auto map = scoreValue.getDefiningOp<MapOp>();
  ArrayAttr maps = map ? map.getIndexingMaps() : ArrayAttr();
  if (!map || map.getNumResults() != 1 || map.getResult(0) != scoreValue ||
      !maps || maps.size() != map.getInputs().size() + 1)
    return failure();
  auto outputMap = dyn_cast<AffineMapAttr>(maps[maps.size() - 1]);
  if (!outputMap ||
      outputMap.getValue().getNumResults() != scoreCoordinates.size())
    return failure();
  SmallVector<AffineExpr> domainCoordinates(
      outputMap.getValue().getNumDims(), AffineExpr());
  for (auto [resultNumber, result] :
       llvm::enumerate(outputMap.getValue().getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(result);
    if (!dimension || dimension.getPosition() >= domainCoordinates.size() ||
        domainCoordinates[dimension.getPosition()])
      return failure();
    domainCoordinates[dimension.getPosition()] = scoreCoordinates[resultNumber];
  }
  if (llvm::any_of(domainCoordinates,
                   [](AffineExpr expression) { return !expression; }))
    return failure();

  std::optional<unsigned> contractInput;
  for (auto [index, input] : llvm::enumerate(map.getInputs())) {
    llvm::SmallDenseSet<Operation *> active;
    if (!onlineValueDependsOnOperation(input, contract.getOperation(), active))
      continue;
    if (contractInput)
      return failure();
    contractInput = index;
  }
  if (!contractInput)
    return failure();
  auto inputMap = dyn_cast<AffineMapAttr>(maps[*contractInput]);
  auto inputType = dyn_cast<RankedTensorType>(
      map.getInputs()[*contractInput].getType());
  if (!inputMap || !inputType ||
      inputMap.getValue().getNumResults() != inputType.getRank())
    return failure();
  SmallVector<AffineExpr> inputCoordinates;
  for (AffineExpr result : inputMap.getValue().getResults())
    inputCoordinates.push_back(
        result.replaceDimsAndSymbols(domainCoordinates, {}));
  return onlineScoreContractOutputCoordinates(
      map.getInputs()[*contractInput], contract, inputCoordinates, context);
}

FailureOr<OnlineScoreContractGeometry> onlineScoreContractGeometry(
    Value scoreValue, ContractOp contract,
    ArrayRef<AffineExpr> scoreCoordinates,
    int64_t scoreReductionDimension, MLIRContext *context) {
  ArrayAttr maps = contract.getIndexingMaps();
  ArrayAttr iteratorKinds = contract.getIteratorKinds();
  if (!maps || maps.size() != contract.getInputs().size() + 1 ||
      !iteratorKinds || maps.empty())
    return failure();
  if (scoreReductionDimension < 0 ||
      scoreReductionDimension >= static_cast<int64_t>(scoreCoordinates.size()))
    return failure();
  SmallVector<AffineExpr> symbolicScoreCoordinates(scoreCoordinates.begin(),
                                                   scoreCoordinates.end());
  symbolicScoreCoordinates[scoreReductionDimension] =
      getAffineDimExpr(7, context);
  FailureOr<SmallVector<AffineExpr>> contractOutputCoordinates =
      onlineScoreContractOutputCoordinates(scoreValue, contract,
                                           symbolicScoreCoordinates, context);
  auto outputMap = dyn_cast<AffineMapAttr>(maps[maps.size() - 1]);
  if (!outputMap ||
      failed(contractOutputCoordinates) ||
      outputMap.getValue().getNumResults() != contractOutputCoordinates->size())
    return failure();
  SmallVector<int64_t> reductionDomains;
  for (auto [dimension, attribute] : llvm::enumerate(iteratorKinds)) {
    auto kind = dyn_cast<StringAttr>(attribute);
    if (!kind)
      return failure();
    if (kind.getValue() == "reduction")
      reductionDomains.push_back(dimension);
    else if (kind.getValue() != "parallel")
      return failure();
  }
  if (reductionDomains.size() != 1)
    return failure();
  const int64_t contractReductionDimension = reductionDomains.front();
  SmallVector<AffineExpr> domainCoordinates(
      outputMap.getValue().getNumDims(), AffineExpr());
  for (auto [scoreDimension, result] :
       llvm::enumerate(outputMap.getValue().getResults())) {
    auto domain = dyn_cast<AffineDimExpr>(result);
    if (!domain || domain.getPosition() >= domainCoordinates.size() ||
        domain.getPosition() == contractReductionDimension ||
        domainCoordinates[domain.getPosition()])
      return failure();
    domainCoordinates[domain.getPosition()] =
        (*contractOutputCoordinates)[scoreDimension];
  }
  domainCoordinates[contractReductionDimension] =
      getAffineDimExpr(8, context);
  if (llvm::any_of(domainCoordinates,
                   [](AffineExpr expression) { return !expression; }))
    return failure();

  OnlineScoreContractGeometry geometry;
  std::optional<int64_t> reductionExtent;
  for (auto [operand, input] : llvm::enumerate(contract.getInputs())) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto inputMap = dyn_cast<AffineMapAttr>(maps[operand]);
    if (!inputType || !inputType.hasStaticShape() || !inputMap ||
        inputMap.getValue().getNumResults() != inputType.getRank())
      return failure();
    AffineExpr linear = getAffineConstantExpr(0, context);
    for (auto [dimension, result] :
         llvm::enumerate(inputMap.getValue().getResults())) {
      AffineExpr composed = result.replaceDimsAndSymbols(
          domainCoordinates, {});
      linear = linear * inputType.getDimSize(dimension) + composed;
      if (auto domain = dyn_cast<AffineDimExpr>(result))
        if (domain.getPosition() == contractReductionDimension) {
          int64_t extent = inputType.getDimSize(dimension);
          if (extent <= 0 ||
              (reductionExtent && *reductionExtent != extent))
            return failure();
          reductionExtent = extent;
        }
    }
    geometry.inputLinearIndices.push_back(linear);
    geometry.inputElementCounts.push_back(inputType.getNumElements());
  }
  if (!reductionExtent)
    return failure();
  geometry.reductionExtent = *reductionExtent;
  return geometry;
}

FailureOr<SmallVector<OnlineMaxReductionNode>> onlineMaxReductionNodes(
    DictionaryAttr certificate, int64_t reductionDimension,
    int64_t reductionExtent, ArrayRef<Attribute> expectedSources) {
  auto version = certificate.getAs<IntegerAttr>("version");
  auto dimension = certificate.getAs<IntegerAttr>("reduction_dimension");
  auto extent = certificate.getAs<IntegerAttr>("reduction_extent");
  auto nodes = certificate.getAs<ArrayAttr>("nodes");
  auto root = certificate.getAs<IntegerAttr>("root");
  auto sources = certificate.getAs<ArrayAttr>("source_refs");
  if (!version || version.getInt() != 1 || !dimension ||
      dimension.getInt() != reductionDimension || !extent ||
      extent.getInt() != reductionExtent || reductionExtent <= 0 || !nodes ||
      !root || !sources || sources.size() != expectedSources.size())
    return failure();
  llvm::SmallDenseSet<Attribute> expectedSourceSet(expectedSources.begin(),
                                                   expectedSources.end());
  llvm::SmallDenseSet<Attribute> certificateSourceSet(sources.begin(),
                                                      sources.end());
  if (expectedSourceSet.size() != expectedSources.size() ||
      certificateSourceSet.size() != sources.size() ||
      certificateSourceSet != expectedSourceSet)
    return failure();

  SmallVector<OnlineMaxReductionNode> result;
  llvm::SmallDenseMap<int64_t, size_t> positions;
  for (Attribute attribute : nodes) {
    auto node = dyn_cast<DictionaryAttr>(attribute);
    auto id = node ? node.getAs<IntegerAttr>("id") : IntegerAttr();
    auto role = node ? node.getAs<StringAttr>("role") : StringAttr();
    auto kind = node ? node.getAs<StringAttr>("kind") : StringAttr();
    auto children =
        node ? node.getAs<DenseI64ArrayAttr>("children")
             : DenseI64ArrayAttr();
    if (!id || id.getInt() < 0 || !kind ||
        (kind.getValue() != "leaf_family" && kind.getValue() != "merge") ||
        !role || role.getValue().empty() || !children ||
        positions.contains(id.getInt()))
      return failure();
    OnlineMaxReductionNode parsed;
    parsed.id = id.getInt();
    parsed.kind = kind.getValue().str();
    parsed.role = role.getValue().str();
    parsed.realization = node.getAs<DictionaryAttr>("realization");
    llvm::append_range(parsed.children, children.asArrayRef());
    positions.try_emplace(parsed.id, result.size());
    result.push_back(std::move(parsed));
  }
  if (!positions.contains(root.getInt()))
    return failure();

  for (OnlineMaxReductionNode &node : result) {
    if (node.children.empty()) {
      if (node.kind != "leaf_family" || !node.realization)
        return failure();
      auto partition = node.realization.getAs<DictionaryAttr>("partition");
      auto offset =
          partition ? partition.getAs<IntegerAttr>("offset") : IntegerAttr();
      auto stride =
          partition ? partition.getAs<IntegerAttr>("stride") : IntegerAttr();
      auto count =
          partition ? partition.getAs<IntegerAttr>("count") : IntegerAttr();
      if (!offset || !stride || !count || offset.getInt() < 0 ||
          stride.getInt() <= 0 || count.getInt() <= 0)
        return failure();
      for (int64_t ordinal = 0; ordinal < count.getInt(); ++ordinal) {
        if (ordinal >
            (std::numeric_limits<int64_t>::max() - offset.getInt()) /
                stride.getInt())
          return failure();
        int64_t coordinate = offset.getInt() + ordinal * stride.getInt();
        if (coordinate < 0 || coordinate >= reductionExtent ||
            !node.coverage.insert(coordinate).second)
          return failure();
      }
      continue;
    }
    if (node.kind != "merge")
      return failure();
    if (node.children.size() < 2)
      return failure();
    llvm::SmallDenseSet<int64_t> childIds;
    for (int64_t childId : node.children) {
      auto child = positions.find(childId);
      if (child == positions.end() || child->second >= positions.lookup(node.id) ||
          !childIds.insert(childId).second)
        return failure();
      for (int64_t coordinate : result[child->second].coverage)
        if (!node.coverage.insert(coordinate).second)
          return failure();
    }
  }
  const OnlineMaxReductionNode &rootNode = result[positions.lookup(root.getInt())];
  if (rootNode.coverage.size() != static_cast<size_t>(reductionExtent))
    return failure();
  for (int64_t coordinate = 0; coordinate < reductionExtent; ++coordinate)
    if (!rootNode.coverage.contains(coordinate))
      return failure();
  llvm::SmallDenseSet<int64_t> reachable;
  std::function<void(int64_t)> visit = [&](int64_t id) {
    if (!reachable.insert(id).second)
      return;
    for (int64_t child : result[positions.lookup(id)].children)
      visit(child);
  };
  visit(root.getInt());
  if (reachable.size() != result.size())
    return failure();
  return result;
}

bool onlineLeafCoordinatesCarrySource(
    const OnlinePtxCfg &cfg, ArrayRef<OnlineMaxReductionNode> nodes,
    StringRef source) {
  auto occurrences = cfg.sources.find(source.str());
  if (occurrences == cfg.sources.end())
    return false;
  llvm::SmallDenseSet<int64_t> expected;
  for (const OnlineMaxReductionNode &node : nodes) {
    if (node.kind != "leaf_family" || !node.realization)
      continue;
    auto role = node.realization.getAs<StringAttr>("coordinate_role");
    auto found = role ? cfg.roles.find(role.getValue().str()) : cfg.roles.end();
    if (found == cfg.roles.end() ||
        cfg.instructions[found->second].sourceRef != source ||
        !expected.insert(found->second).second)
      return false;
  }
  return expected == llvm::SmallDenseSet<int64_t>(
                         occurrences->second.begin(),
                         occurrences->second.end());
}

bool onlineScoreContractBindings(
    const OnlinePtxCfg &cfg, const OnlineMaxReductionNode &node,
    const OnlineScoreContractGeometry &geometry, StringRef scoreSource,
    ArrayRef<int64_t> launchExtents, int64_t scoreReductionExtent,
    ArrayRef<int64_t> inputParameters, MLIRContext *context) {
  DictionaryAttr score =
      node.realization.getAs<DictionaryAttr>("score_contract");
  if (!score || inputParameters.size() != 2 ||
      geometry.inputLinearIndices.size() != 2 ||
      geometry.inputElementCounts.size() != 2)
    return false;
  auto role = [&](StringRef name) -> std::optional<int64_t> {
    auto attribute = score.getAs<StringAttr>(name);
    if (!attribute)
      return std::nullopt;
    auto found = cfg.roles.find(attribute.getValue().str());
    return found == cfg.roles.end() ? std::nullopt
                                    : std::optional<int64_t>(found->second);
  };
  auto indexInit = role("index_init_role");
  auto indexGuard = role("index_guard_role");
  auto indexStep = role("index_step_role");
  auto exitBranch = role("exit_branch_role");
  auto backedge = role("backedge_role");
  auto coordinate = role("coordinate_role");
  auto accumulatorInit = role("accumulator_init_role");
  auto accumulatorUpdate = role("accumulator_update_role");
  auto result = role("result_role");
  if (!indexInit || !indexGuard || !indexStep || !exitBranch || !backedge ||
      !coordinate || !accumulatorInit || !accumulatorUpdate || !result)
    return false;
  const OnlinePtxInstruction &initInstruction = cfg.instructions[*indexInit];
  const OnlinePtxInstruction &guardInstruction = cfg.instructions[*indexGuard];
  const OnlinePtxInstruction &stepInstruction = cfg.instructions[*indexStep];
  const OnlinePtxInstruction &exitInstruction = cfg.instructions[*exitBranch];
  const OnlinePtxInstruction &backedgeInstruction = cfg.instructions[*backedge];
  const OnlinePtxInstruction &coordinateInstruction =
      cfg.instructions[*coordinate];
  const OnlinePtxInstruction &accumulatorInitInstruction =
      cfg.instructions[*accumulatorInit];
  const OnlinePtxInstruction &updateInstruction =
      cfg.instructions[*accumulatorUpdate];
  const OnlinePtxInstruction &resultInstruction = cfg.instructions[*result];
  auto reachesInduction = [&](int64_t consumer, StringRef operand) {
    return onlineReachingDefinitions(cfg, consumer, operand) ==
           std::set<int64_t>{*indexInit, *indexStep};
  };
  auto reachesAccumulator = [&](int64_t consumer, StringRef operand) {
    return onlineReachingDefinitions(cfg, consumer, operand) ==
           std::set<int64_t>{*accumulatorInit, *accumulatorUpdate};
  };
  if ((initInstruction.opcode != "mov.u32" &&
       initInstruction.opcode != "mov.s32") ||
      !initInstruction.predicate.empty() || initInstruction.operands.size() != 2 ||
      initInstruction.operands[1] != "0" ||
      (guardInstruction.opcode != "setp.ge.u32" &&
       guardInstruction.opcode != "setp.ge.s32") ||
      !guardInstruction.predicate.empty() || guardInstruction.operands.size() != 3 ||
      !reachesInduction(*indexGuard, guardInstruction.operands[1]) ||
      guardInstruction.operands[2] !=
          Twine(geometry.reductionExtent).str() ||
      !StringRef(exitInstruction.opcode).starts_with("bra") ||
      exitInstruction.predicate.empty() || exitInstruction.negatedPredicate ||
      onlineReachingDefinitions(cfg, *exitBranch,
                                exitInstruction.predicate) !=
          std::set<int64_t>{*indexGuard} ||
      (stepInstruction.opcode != "add.u32" &&
       stepInstruction.opcode != "add.s32") ||
      !stepInstruction.predicate.empty() || stepInstruction.operands.size() != 3 ||
      !reachesInduction(*indexStep, stepInstruction.operands[1]) ||
      stepInstruction.operands[2] != "1" ||
      !StringRef(backedgeInstruction.opcode).starts_with("bra") ||
      !backedgeInstruction.predicate.empty() ||
      coordinateInstruction.predicate.size() != 0 ||
      coordinateInstruction.operands.size() != 2 ||
      !reachesInduction(*coordinate, coordinateInstruction.operands[1]) ||
      accumulatorInitInstruction.opcode != "mov.f32" ||
      !accumulatorInitInstruction.predicate.empty() ||
      !llvm::is_contained(accumulatorInitInstruction.operands,
                          "0f00000000") ||
      updateInstruction.opcode != "fma.rn.f32" ||
      !updateInstruction.predicate.empty() || updateInstruction.operands.size() != 4 ||
      !reachesAccumulator(*accumulatorUpdate,
                          updateInstruction.operands[3]) ||
      resultInstruction.opcode != "mov.f32" ||
      !resultInstruction.predicate.empty() || resultInstruction.operands.size() != 2 ||
      !reachesAccumulator(*result, resultInstruction.operands[1]) ||
      resultInstruction.sourceRef != scoreSource)
    return false;

  auto exitTarget = cfg.labels.find(exitInstruction.branchTarget);
  auto backedgeTarget = cfg.labels.find(backedgeInstruction.branchTarget);
  int64_t bodyBlock = -1;
  if (exitTarget == cfg.labels.end() ||
      exitTarget->second != resultInstruction.block ||
      backedgeTarget == cfg.labels.end() ||
      backedgeTarget->second != guardInstruction.block ||
      cfg.successors[guardInstruction.block].size() != 2 ||
      cfg.successors[backedgeInstruction.block] !=
          std::vector<int64_t>{guardInstruction.block} ||
      !onlineInstructionDominates(cfg, *indexGuard, *accumulatorUpdate) ||
      !onlineInstructionDominates(cfg, *accumulatorUpdate, *indexStep) ||
      !onlineInstructionDominates(cfg, *indexStep, *backedge) ||
      !onlineInstructionDominates(cfg, *exitBranch, *result))
    return false;
  for (int64_t successor : cfg.successors[guardInstruction.block])
    if (successor != resultInstruction.block)
      bodyBlock = successor;
  if (bodyBlock < 0)
    return false;
  llvm::SmallDenseSet<int64_t> bodyBlocks;
  SmallVector<int64_t> worklist{bodyBlock};
  while (!worklist.empty()) {
    int64_t block = worklist.pop_back_val();
    if (block == guardInstruction.block || block == resultInstruction.block ||
        !bodyBlocks.insert(block).second)
      continue;
    for (int64_t successor : cfg.successors[block])
      if (successor != guardInstruction.block &&
          successor != resultInstruction.block)
        worklist.push_back(successor);
  }
  for (int64_t block : bodyBlocks)
    for (int64_t successor : cfg.successors[block])
      if (successor == resultInstruction.block ||
          (successor == guardInstruction.block &&
           block != backedgeInstruction.block))
        return false;
  if (!onlineBlocksAreAcyclic(cfg, bodyBlocks))
    return false;

  llvm::SmallDenseMap<int64_t, AffineExpr> symbols;
  auto outerCoordinate = node.realization.getAs<StringAttr>("coordinate_role");
  auto outer = outerCoordinate
                   ? cfg.roles.find(outerCoordinate.getValue().str())
                   : cfg.roles.end();
  if (outer == cfg.roles.end())
    return false;
  symbols.try_emplace(outer->second, getAffineDimExpr(7, context));
  symbols.try_emplace(*coordinate, getAffineDimExpr(8, context));
  SmallVector<int64_t> expressionExtents(launchExtents.begin(),
                                         launchExtents.end());
  if (scoreReductionExtent <= 0)
    return false;
  expressionExtents.push_back(scoreReductionExtent);
  expressionExtents.push_back(geometry.reductionExtent);
  for (int64_t operand = 0; operand < 2; ++operand) {
    StringRef prefix = operand == 0 ? "lhs_" : "rhs_";
    auto linear = role((Twine(prefix) + "linear_index_role").str());
    auto byteOffset = role((Twine(prefix) + "byte_offset_role").str());
    auto pointer = role((Twine(prefix) + "pointer_role").str());
    auto load = role((Twine(prefix) + "load_role").str());
    if (!linear || !byteOffset || !pointer || !load)
      return false;
    const OnlinePtxInstruction &linearInstruction = cfg.instructions[*linear];
    const OnlinePtxInstruction &byteInstruction = cfg.instructions[*byteOffset];
    const OnlinePtxInstruction &pointerInstruction = cfg.instructions[*pointer];
    const OnlinePtxInstruction &loadInstruction = cfg.instructions[*load];
    if (linearInstruction.operands.size() < 2 ||
        byteInstruction.opcode != "mul.wide.u32" ||
        byteInstruction.operands.size() != 3 || byteInstruction.operands[2] != "4" ||
        onlineReachingDefinitions(cfg, *byteOffset,
                                  byteInstruction.operands[1]) !=
            std::set<int64_t>{*linear} ||
        pointerInstruction.opcode != "add.u64" ||
        pointerInstruction.operands.size() != 3 ||
        loadInstruction.opcode != "ld.global.f32" ||
        loadInstruction.operands.size() != 2 ||
        onlineReachingDefinitions(cfg, *load,
                                  loadInstruction.operands[1]) !=
            std::set<int64_t>{*pointer} ||
        onlineReachingDefinitions(cfg, *accumulatorUpdate,
                                  updateInstruction.operands[1 + operand]) !=
            std::set<int64_t>{*load})
      return false;
    std::set<std::pair<int64_t, std::string>> active;
    OnlineIndexAlternatives alternatives = onlineIndexAlternatives(
        cfg, *linear, linearInstruction.operands[1], context, active,
        &symbols);
    if (!alternatives.valid || alternatives.values.empty() ||
        llvm::any_of(alternatives.values, [&](AffineExpr value) {
          return value != geometry.inputLinearIndices[operand] ||
                 !onlineIndexExprFitsBuffer(
                     value, expressionExtents,
                     geometry.inputElementCounts[operand]);
        }) ||
        llvm::any_of(alternatives.intermediates, [&](AffineExpr value) {
          return !onlineIndexExprFitsLaunchDomain(value, expressionExtents);
        }))
      return false;
    const bool baseFirst =
        onlinePointerIsOutputParameter(cfg, *pointer,
                                       pointerInstruction.operands[1],
                                       inputParameters[operand]) &&
        onlineReachingDefinitions(cfg, *pointer,
                                  pointerInstruction.operands[2]) ==
            std::set<int64_t>{*byteOffset};
    const bool baseSecond =
        onlinePointerIsOutputParameter(cfg, *pointer,
                                       pointerInstruction.operands[2],
                                       inputParameters[operand]) &&
        onlineReachingDefinitions(cfg, *pointer,
                                  pointerInstruction.operands[1]) ==
            std::set<int64_t>{*byteOffset};
    if (!baseFirst && !baseSecond)
      return false;
  }
  return true;
}

FailureOr<Attribute> onlineScalarScoreSemantics(Value scalar, MapOp map,
                                                Builder &builder) {
  if (auto argument = dyn_cast<BlockArgument>(scalar)) {
    if (argument.getOwner() != &map.getBody().front() ||
        argument.getArgNumber() >= map.getInputs().size())
      return failure();
    return DictionaryAttr::get(
        builder.getContext(),
        {builder.getNamedAttr("kind", builder.getStringAttr("argument")),
         builder.getNamedAttr(
             "index", builder.getI64IntegerAttr(argument.getArgNumber()))});
  }
  Operation *operation = scalar.getDefiningOp();
  if (!operation || operation->getNumResults() != 1)
    return failure();
  if (auto constant = dyn_cast<arith::ConstantOp>(operation))
    return DictionaryAttr::get(
        builder.getContext(),
        {builder.getNamedAttr("kind", builder.getStringAttr("constant")),
         builder.getNamedAttr("value", constant.getValue())});
  StringRef name = operation->getName().getStringRef();
  if (name != arith::AddFOp::getOperationName() &&
      name != arith::SubFOp::getOperationName() &&
      name != arith::MulFOp::getOperationName() &&
      name != arith::DivFOp::getOperationName() &&
      name != arith::MaximumFOp::getOperationName() && name != "math.tanh")
    return failure();
  SmallVector<Attribute> operands;
  for (Value operand : operation->getOperands()) {
    FailureOr<Attribute> semantic =
        onlineScalarScoreSemantics(operand, map, builder);
    if (failed(semantic))
      return failure();
    operands.push_back(*semantic);
  }
  return DictionaryAttr::get(
      builder.getContext(),
      {builder.getNamedAttr("kind", builder.getStringAttr("operation")),
       builder.getNamedAttr("name", builder.getStringAttr(name)),
       builder.getNamedAttr("attributes", operation->getAttrDictionary()),
       builder.getNamedAttr("operands", builder.getArrayAttr(operands))});
}

FailureOr<Attribute> onlineScoreSemantics(
    Value value, ContractOp scoreContract, Builder &builder,
    llvm::SmallDenseSet<Operation *> &active) {
  if (value.getDefiningOp() == scoreContract.getOperation()) {
    SourceRefAttr source =
        scoreContract->getAttrOfType<SourceRefAttr>("source");
    if (!source)
      return failure();
    return DictionaryAttr::get(
        builder.getContext(),
        {builder.getNamedAttr("kind", builder.getStringAttr("contract")),
         builder.getNamedAttr("source", source)});
  }
  auto map = value.getDefiningOp<MapOp>();
  if (!map || map.getNumResults() != 1 || map.getResult(0) != value ||
      map.getBody().empty() || !active.insert(map.getOperation()).second)
    return failure();
  SourceRefAttr source = map->getAttrOfType<SourceRefAttr>("source");
  auto yield = dyn_cast<YieldOp>(map.getBody().front().getTerminator());
  FailureOr<Attribute> scalar =
      yield && yield.getOperands().size() == 1
          ? onlineScalarScoreSemantics(yield.getOperands().front(), map,
                                       builder)
          : FailureOr<Attribute>(failure());
  SmallVector<Attribute> inputs;
  for (Value input : map.getInputs()) {
    FailureOr<Attribute> semantic =
        onlineScoreSemantics(input, scoreContract, builder, active);
    if (failed(semantic)) {
      active.erase(map.getOperation());
      return failure();
    }
    inputs.push_back(*semantic);
  }
  active.erase(map.getOperation());
  if (!source || failed(scalar))
    return failure();
  return DictionaryAttr::get(
      builder.getContext(),
      {builder.getNamedAttr("kind", builder.getStringAttr("map")),
       builder.getNamedAttr("source", source),
       builder.getNamedAttr("semantics", map->getAttr("semantics")),
       builder.getNamedAttr("indexing_maps", map.getIndexingMaps()),
       builder.getNamedAttr("scalar", *scalar),
       builder.getNamedAttr("inputs", builder.getArrayAttr(inputs))});
}

FailureOr<std::pair<Value, ContractOp>> onlineScoreSourceChain(ModuleOp module) {
  FoldOp maximumFold;
  module.walk([&](FoldOp fold) {
    if (!maximumFold &&
        foldYieldIs(fold, arith::MaximumFOp::getOperationName()))
      maximumFold = fold;
  });
  if (!maximumFold)
    return failure();
  SmallVector<ContractOp> contracts;
  llvm::SmallDenseSet<Operation *> ancestors;
  std::function<void(Value)> collect = [&](Value value) {
    Operation *definition = value.getDefiningOp();
    if (!definition || !ancestors.insert(definition).second)
      return;
    if (auto contract = dyn_cast<ContractOp>(definition))
      contracts.push_back(contract);
    for (Value operand : definition->getOperands())
      collect(operand);
  };
  collect(maximumFold.getInputs().front());
  auto select = maximumFold.getInputs().front().getDefiningOp<RelationOp>();
  if (contracts.size() != 1 || !select ||
      select.getSemantics() != RelationSemantics::Select ||
      select.getInputs().size() != 3)
    return failure();
  Value score;
  for (Value candidate : select.getInputs().drop_front()) {
    llvm::SmallDenseSet<Operation *> active;
    if (!onlineValueDependsOnOperation(candidate,
                                       contracts.front().getOperation(), active))
      continue;
    if (score)
      return failure();
    score = candidate;
  }
  if (!score)
    return failure();
  return std::pair<Value, ContractOp>{score, contracts.front()};
}

bool onlineScoreSemanticCertificateMatches(Value score, ContractOp contract,
                                           Attribute certificate) {
  Builder builder(score.getContext());
  llvm::SmallDenseSet<Operation *> active;
  FailureOr<Attribute> expected =
      onlineScoreSemantics(score, contract, builder, active);
  return succeeded(expected) && certificate == *expected;
}

Attribute onlineMutateScoreSemanticCertificate(
    Attribute attribute, StringRef key, std::optional<StringRef> value,
    Builder &builder, bool &changed) {
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute)) {
    SmallVector<NamedAttribute> fields;
    for (NamedAttribute field : dictionary) {
      Attribute replacement = field.getValue();
      auto string = dyn_cast<StringAttr>(field.getValue());
      if (!changed && field.getName().strref() == key &&
          (!value || (string && string.getValue() == *value))) {
        replacement = builder.getStringAttr("test.mutated");
        changed = true;
      } else {
        replacement = onlineMutateScoreSemanticCertificate(
            field.getValue(), key, value, builder, changed);
      }
      fields.push_back(builder.getNamedAttr(field.getName(), replacement));
    }
    return builder.getDictionaryAttr(fields);
  }
  if (auto array = dyn_cast<ArrayAttr>(attribute)) {
    SmallVector<Attribute> elements;
    for (Attribute element : array)
      elements.push_back(onlineMutateScoreSemanticCertificate(
          element, key, value, builder, changed));
    return builder.getArrayAttr(elements);
  }
  return attribute;
}

bool onlineScoreSemanticCertificateContains(Attribute attribute, StringRef key,
                                            StringRef value) {
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute)) {
    for (NamedAttribute field : dictionary) {
      if (field.getName().strref() == key)
        if (auto string = dyn_cast<StringAttr>(field.getValue());
            string && string.getValue() == value)
          return true;
      if (onlineScoreSemanticCertificateContains(field.getValue(), key, value))
        return true;
    }
  }
  if (auto array = dyn_cast<ArrayAttr>(attribute))
    return llvm::any_of(array, [&](Attribute element) {
      return onlineScoreSemanticCertificateContains(element, key, value);
    });
  return false;
}

struct OnlineFloatExpression {
  std::string operation;
  std::string atom;
  std::vector<OnlineFloatExpression> operands;
};

std::string onlineFloatExpressionText(const OnlineFloatExpression &expression) {
  if (!expression.atom.empty())
    return expression.atom;
  SmallVector<std::string> operands;
  for (const OnlineFloatExpression &operand : expression.operands)
    operands.push_back(onlineFloatExpressionText(operand));
  if (expression.operation == "add" || expression.operation == "mul")
    llvm::sort(operands);
  std::string result = expression.operation;
  result += "(";
  result += llvm::join(operands, ",");
  result += ")";
  return result;
}

OnlineFloatExpression onlineFloatOperation(
    StringRef operation, ArrayRef<OnlineFloatExpression> operands) {
  OnlineFloatExpression result;
  result.operation = operation.str();
  for (const OnlineFloatExpression &operand : operands) {
    if ((operation == "add" || operation == "mul") &&
        operand.operation == operation && operand.atom.empty())
      llvm::append_range(result.operands, operand.operands);
    else
      result.operands.push_back(operand);
  }
  return result;
}

FailureOr<OnlineFloatExpression> onlineSourceFloatExpression(
    Value value, ContractOp contract,
    llvm::SmallDenseSet<Operation *> &active);

FailureOr<OnlineFloatExpression> onlineSourceScalarExpression(
    Value scalar, MapOp map, ContractOp contract,
    llvm::SmallDenseSet<Operation *> &active) {
  if (auto argument = dyn_cast<BlockArgument>(scalar)) {
    if (argument.getOwner() != &map.getBody().front() ||
        argument.getArgNumber() >= map.getInputs().size())
      return failure();
    return onlineSourceFloatExpression(
        map.getInputs()[argument.getArgNumber()], contract, active);
  }
  Operation *operation = scalar.getDefiningOp();
  if (!operation)
    return failure();
  if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
    auto value = dyn_cast<FloatAttr>(constant.getValue());
    if (!value || !value.getType().isF32())
      return failure();
    std::string bits = llvm::utohexstr(
        value.getValue().bitcastToAPInt().getZExtValue(), true);
    bits.insert(bits.begin(), 8 - bits.size(), '0');
    return OnlineFloatExpression{
        "", (Twine("0f") + bits).str(), {}};
  }
  SmallVector<OnlineFloatExpression> operands;
  for (Value operand : operation->getOperands()) {
    FailureOr<OnlineFloatExpression> expression =
        onlineSourceScalarExpression(operand, map, contract, active);
    if (failed(expression))
      return failure();
    operands.push_back(*expression);
  }
  StringRef name = operation->getName().getStringRef();
  if (name == arith::AddFOp::getOperationName())
    return onlineFloatOperation("add", operands);
  if (name == arith::MulFOp::getOperationName())
    return onlineFloatOperation("mul", operands);
  if (name == arith::SubFOp::getOperationName())
    return onlineFloatOperation("sub", operands);
  if (name == arith::DivFOp::getOperationName())
    return onlineFloatOperation(
        "mul", {operands[0], onlineFloatOperation("recip", {operands[1]})});
  if (name == arith::MaximumFOp::getOperationName())
    return onlineFloatOperation("max", operands);
  if (name == "math.tanh")
    return onlineFloatOperation("tanh", operands);
  return failure();
}

FailureOr<OnlineFloatExpression> onlineSourceFloatExpression(
    Value value, ContractOp contract,
    llvm::SmallDenseSet<Operation *> &active) {
  if (value.getDefiningOp() == contract.getOperation())
    return OnlineFloatExpression{"", "raw_score", {}};
  auto map = value.getDefiningOp<MapOp>();
  if (!map || map.getBody().empty() ||
      !active.insert(map.getOperation()).second)
    return failure();
  auto yield = dyn_cast<YieldOp>(map.getBody().front().getTerminator());
  FailureOr<OnlineFloatExpression> result =
      yield && yield.getOperands().size() == 1
          ? onlineSourceScalarExpression(yield.getOperands().front(), map,
                                         contract, active)
          : FailureOr<OnlineFloatExpression>(failure());
  active.erase(map.getOperation());
  return result;
}

FailureOr<OnlineFloatExpression> onlinePtxFloatExpression(
    const OnlinePtxCfg &cfg, int64_t instruction, int64_t rawInstruction,
    llvm::SmallDenseSet<int64_t> &active);

FailureOr<OnlineFloatExpression> onlinePtxFloatOperand(
    const OnlinePtxCfg &cfg, int64_t instruction, StringRef operand,
    int64_t rawInstruction, llvm::SmallDenseSet<int64_t> &active) {
  if (operand.starts_with("0f"))
    return OnlineFloatExpression{"", operand.lower(), {}};
  std::set<int64_t> definitions =
      onlineReachingDefinitions(cfg, instruction, operand);
  if (definitions.size() != 1)
    return failure();
  return onlinePtxFloatExpression(cfg, *definitions.begin(), rawInstruction,
                                  active);
}

FailureOr<OnlineFloatExpression> onlinePtxFloatExpression(
    const OnlinePtxCfg &cfg, int64_t instruction, int64_t rawInstruction,
    llvm::SmallDenseSet<int64_t> &active) {
  if (instruction == rawInstruction)
    return OnlineFloatExpression{"", "raw_score", {}};
  if (instruction < 0 ||
      instruction >= static_cast<int64_t>(cfg.instructions.size()) ||
      !active.insert(instruction).second)
    return failure();
  const OnlinePtxInstruction &ptx = cfg.instructions[instruction];
  if (!ptx.predicate.empty() || ptx.operands.size() < 2) {
    active.erase(instruction);
    return failure();
  }
  SmallVector<OnlineFloatExpression> operands;
  for (StringRef operand : ArrayRef<std::string>(ptx.operands).drop_front()) {
    FailureOr<OnlineFloatExpression> expression = onlinePtxFloatOperand(
        cfg, instruction, operand, rawInstruction, active);
    if (failed(expression)) {
      active.erase(instruction);
      return failure();
    }
    operands.push_back(*expression);
  }
  active.erase(instruction);
  StringRef opcode = ptx.opcode;
  if (opcode == "mov.f32" && operands.size() == 1)
    return operands.front();
  if (opcode == "add.rn.f32" && operands.size() == 2)
    return onlineFloatOperation("add", operands);
  if (opcode == "mul.rn.f32" && operands.size() == 2)
    return onlineFloatOperation("mul", operands);
  if (opcode == "sub.rn.f32" && operands.size() == 2)
    return onlineFloatOperation("sub", operands);
  if (opcode == "div.rn.f32" && operands.size() == 2)
    return onlineFloatOperation(
        "mul", {operands[0], onlineFloatOperation("recip", {operands[1]})});
  if (opcode == "rcp.rn.f32" && operands.size() == 1)
    return onlineFloatOperation("recip", operands);
  if (opcode == "max.f32" && operands.size() == 2)
    return onlineFloatOperation("max", operands);
  if (opcode == "tanh.approx.f32" && operands.size() == 1)
    return onlineFloatOperation("tanh", operands);
  if (opcode == "fma.rn.f32" && operands.size() == 3)
    return onlineFloatOperation(
        "add", {onlineFloatOperation("mul", {operands[0], operands[1]}),
                operands[2]});
  return failure();
}

std::string onlineRenderFloatExpression(const OnlineFloatExpression &expression,
                                        int64_t &nextRegister,
                                        std::string &instructions) {
  if (expression.atom == "raw_score")
    return "%f0";
  if (!expression.atom.empty()) {
    std::string result = (Twine("%f") + Twine(nextRegister++)).str();
    instructions += "mov.f32 " + result + ", " + expression.atom + ";\n";
    return result;
  }
  SmallVector<std::string> operands;
  for (const OnlineFloatExpression &operand : expression.operands)
    operands.push_back(
        onlineRenderFloatExpression(operand, nextRegister, instructions));
  if ((expression.operation == "add" || expression.operation == "mul") &&
      operands.size() >= 2) {
    StringRef opcode = expression.operation == "add" ? "add.rn.f32"
                                                       : "mul.rn.f32";
    std::string accumulated = operands.front();
    for (StringRef operand : ArrayRef<std::string>(operands).drop_front()) {
      std::string result = (Twine("%f") + Twine(nextRegister++)).str();
      instructions += opcode.str() + " " + result + ", " + accumulated +
                      ", " + operand.str() + ";\n";
      accumulated = std::move(result);
    }
    return accumulated;
  }
  StringRef opcode =
      llvm::StringSwitch<StringRef>(expression.operation)
          .Case("add", "add.rn.f32")
          .Case("mul", "mul.rn.f32")
          .Case("sub", "sub.rn.f32")
          .Case("recip", "rcp.rn.f32")
          .Case("max", "max.f32")
          .Case("tanh", "tanh.approx.f32")
          .Default("");
  if (opcode.empty())
    return {};
  std::string result = (Twine("%f") + Twine(nextRegister++)).str();
  instructions += opcode.str() + " " + result + ", " +
                  llvm::join(operands, ", ") + ";\n";
  return result;
}

std::string onlineExpressionPtx(const OnlineFloatExpression &expression,
                                StringRef rawSource,
                                std::string &finalOperand) {
  int64_t nextRegister = 1;
  std::string body;
  finalOperand =
      onlineRenderFloatExpression(expression, nextRegister, body);
  if (finalOperand.empty())
    return {};
  return (Twine(".version 8.0\n.target sm_90\n.address_size 64\n"
                ".visible .entry expression() {\n"
                "// shuttle.online.role raw_score\n"
                "// shuttle.online.source_ref ") +
          rawSource + "\nmov.f32 %f0, %f99;\n" + body +
          "// shuttle.online.role score_value\nmov.f32 %f999, " +
          finalOperand + ";\nret;\n}\n")
      .str();
}

bool onlinePtxMatchesFloatExpression(
    StringRef ptx, const OnlineFloatExpression &expected) {
  FailureOr<OnlinePtxCfg> cfg = onlinePtxCfg(ptx);
  if (failed(cfg))
    return false;
  auto raw = cfg->roles.find("raw_score");
  auto final = cfg->roles.find("score_value");
  if (raw == cfg->roles.end() || final == cfg->roles.end())
    return false;
  llvm::SmallDenseSet<int64_t> active;
  FailureOr<OnlineFloatExpression> actual = onlinePtxFloatExpression(
      *cfg, final->second, raw->second, active);
  return succeeded(actual) &&
         onlineFloatExpressionText(expected) ==
             onlineFloatExpressionText(*actual);
}

bool onlineScoreMapChainBindings(const OnlinePtxCfg &cfg,
                                 const OnlineMaxReductionNode &node,
                                 Value scoreValue, ContractOp scoreContract) {
  auto scoreContractCertificate =
      node.realization.getAs<DictionaryAttr>("score_contract");
  auto rawName = scoreContractCertificate
                     ? scoreContractCertificate.getAs<StringAttr>("result_role")
                     : StringAttr();
  auto finalName = node.realization.getAs<StringAttr>("score_value_role");
  auto raw = rawName ? cfg.roles.find(rawName.getValue().str()) : cfg.roles.end();
  auto final = finalName ? cfg.roles.find(finalName.getValue().str())
                         : cfg.roles.end();
  if (raw == cfg.roles.end() || final == cfg.roles.end())
    return false;
  if (!scoreValue)
    return raw->second == final->second;
  auto scoreSource =
      scoreContract->getAttrOfType<SourceRefAttr>("source");
  if (!scoreSource || !cfg.instructions[raw->second].predicate.empty() ||
      cfg.instructions[raw->second].sourceRef !=
          onlineSourceKey(scoreSource))
    return false;
  Attribute certificate = node.realization.get("score_semantics");
  Builder builder(scoreValue.getContext());
  llvm::SmallDenseSet<Operation *> active;
  FailureOr<Attribute> expected =
      onlineScoreSemantics(scoreValue, scoreContract, builder, active);
  llvm::SmallDenseSet<Operation *> sourceActive;
  FailureOr<OnlineFloatExpression> sourceExpression =
      onlineSourceFloatExpression(scoreValue, scoreContract, sourceActive);
  llvm::SmallDenseSet<int64_t> ptxActive;
  FailureOr<OnlineFloatExpression> ptxExpression = onlinePtxFloatExpression(
      cfg, final->second, raw->second, ptxActive);
  return certificate && succeeded(expected) && certificate == *expected &&
         succeeded(sourceExpression) && succeeded(ptxExpression) &&
         onlineFloatExpressionText(*sourceExpression) ==
             onlineFloatExpressionText(*ptxExpression);
}

bool onlineCanonicalLeafFamilyBindings(const OnlinePtxCfg &cfg,
                                       const OnlineMaxReductionNode &node,
                                       StringRef scoreSource,
                                       StringRef comparisonOpcode,
                                       const std::set<std::string> &relationSources,
                                       const OnlineScoreContractGeometry &scoreGeometry,
                                       ArrayRef<int64_t> launchExtents,
                                       int64_t scoreReductionExtent,
                                       ArrayRef<int64_t> inputParameters,
                                       Value scoreValueTensor,
                                       ContractOp scoreContract,
                                       MLIRContext *context) {
  auto kind = node.realization.getAs<StringAttr>("kind");
  auto role = [&](StringRef name) -> std::optional<int64_t> {
    auto attribute = node.realization.getAs<StringAttr>(name);
    if (!attribute)
      return std::nullopt;
    auto found = cfg.roles.find(attribute.getValue().str());
    return found == cfg.roles.end() ? std::nullopt
                                    : std::optional<int64_t>(found->second);
  };
  auto indexInit = role("index_init_role");
  auto indexGuard = role("index_guard_role");
  auto indexStep = role("index_step_role");
  auto exitBranch = role("exit_branch_role");
  auto backedge = role("backedge_role");
  auto accumulatorInit = role("accumulator_init_role");
  auto accumulatorUpdate = role("accumulator_update_role");
  auto coordinate = role("coordinate_role");
  auto scoreValue = role("score_value_role");
  auto relationCompare = role("relation_compare_role");
  auto maskSelect = role("mask_select_role");
  auto negativeInfinity = role("negative_infinity_role");
  auto result = cfg.roles.find(node.role);
  if (!kind || kind.getValue() != "canonical_loop" || !indexInit ||
      !indexGuard || !indexStep || !exitBranch || !backedge ||
      !accumulatorInit || !accumulatorUpdate || !coordinate ||
      !scoreValue || !relationCompare || !maskSelect || !negativeInfinity ||
      result == cfg.roles.end())
    return false;
  SmallVector<int64_t> ordered(node.coverage.begin(), node.coverage.end());
  llvm::sort(ordered);
  auto declaredStep = node.realization.getAs<IntegerAttr>("step");
  const int64_t step = ordered.size() > 1 ? ordered[1] - ordered[0]
                                         : declaredStep
                                               ? declaredStep.getInt()
                                               : 1;
  if (step <= 0 || (declaredStep && declaredStep.getInt() != step))
    return false;
  const std::string lower = Twine(ordered.front()).str();
  const std::string upper = Twine(ordered.back() + 1).str();
  const std::string stepText = Twine(step).str();
  const OnlinePtxInstruction &initInstruction = cfg.instructions[*indexInit];
  const OnlinePtxInstruction &guardInstruction = cfg.instructions[*indexGuard];
  const OnlinePtxInstruction &stepInstruction = cfg.instructions[*indexStep];
  const OnlinePtxInstruction &exitInstruction = cfg.instructions[*exitBranch];
  const OnlinePtxInstruction &backedgeInstruction = cfg.instructions[*backedge];
  const OnlinePtxInstruction &coordinateInstruction =
      cfg.instructions[*coordinate];
  const OnlinePtxInstruction &accumulatorInitInstruction =
      cfg.instructions[*accumulatorInit];
  const OnlinePtxInstruction &accumulatorInstruction =
      cfg.instructions[*accumulatorUpdate];
  const OnlinePtxInstruction &resultInstruction =
      cfg.instructions[result->second];
  auto reachesEitherIteration = [&](int64_t consumer, StringRef operand) {
    std::set<int64_t> definitions =
        onlineReachingDefinitions(cfg, consumer, operand);
    return definitions == std::set<int64_t>{*indexInit, *indexStep};
  };
  auto reachesEitherAccumulator = [&](int64_t consumer, StringRef operand) {
    std::set<int64_t> definitions =
        onlineReachingDefinitions(cfg, consumer, operand);
    return definitions ==
           std::set<int64_t>{*accumulatorInit, *accumulatorUpdate};
  };
  std::set<std::pair<int64_t, int64_t>> scoreCoordinateActive;
  if (accumulatorInitInstruction.opcode != "mov.f32" ||
      !accumulatorInitInstruction.predicate.empty() ||
      !llvm::is_contained(accumulatorInitInstruction.operands,
                          "0fFF800000") ||
      (initInstruction.opcode != "mov.u32" &&
       initInstruction.opcode != "mov.s32") ||
      !initInstruction.predicate.empty() ||
      !llvm::is_contained(initInstruction.operands, lower) ||
      (guardInstruction.opcode != "setp.ge.u32" &&
       guardInstruction.opcode != "setp.ge.s32") ||
      guardInstruction.operands.size() != 3 ||
      !guardInstruction.predicate.empty() ||
      !reachesEitherIteration(*indexGuard, guardInstruction.operands[1]) ||
      guardInstruction.operands[2] != upper ||
      !StringRef(exitInstruction.opcode).starts_with("bra") ||
      exitInstruction.predicate.empty() || exitInstruction.negatedPredicate ||
      onlineReachingDefinitions(cfg, *exitBranch,
                                exitInstruction.predicate) !=
          std::set<int64_t>{*indexGuard} ||
      (stepInstruction.opcode != "add.u32" &&
       stepInstruction.opcode != "add.s32") ||
      !stepInstruction.predicate.empty() ||
      stepInstruction.operands.size() != 3 ||
      !reachesEitherIteration(*indexStep, stepInstruction.operands[1]) ||
      stepInstruction.operands[2] != stepText ||
      !StringRef(backedgeInstruction.opcode).starts_with("bra") ||
      !backedgeInstruction.predicate.empty() ||
      !coordinateInstruction.predicate.empty() ||
      coordinateInstruction.operands.size() != 2 ||
      !reachesEitherIteration(*coordinate,
                              coordinateInstruction.operands[1]) ||
      !onlineRoleTransitivelyDependsOn(
          cfg,
          node.realization.getAs<StringAttr>("relation_compare_role")
              .getValue(),
          node.realization.getAs<StringAttr>("coordinate_role").getValue()) ||
      !cfg.instructions[*scoreValue].predicate.empty() ||
      !StringRef(cfg.instructions[*relationCompare].opcode)
           .starts_with(comparisonOpcode) ||
      !cfg.instructions[*relationCompare].predicate.empty() ||
      !relationSources.count(cfg.instructions[*relationCompare].sourceRef) ||
      cfg.instructions[*negativeInfinity].opcode != "mov.f32" ||
      !cfg.instructions[*negativeInfinity].predicate.empty() ||
      !llvm::is_contained(cfg.instructions[*negativeInfinity].operands,
                          "0fFF800000") ||
      cfg.instructions[*maskSelect].opcode != "selp.f32" ||
      !cfg.instructions[*maskSelect].predicate.empty() ||
      !relationSources.count(cfg.instructions[*maskSelect].sourceRef) ||
      !onlineRoleOperandIs(
          cfg,
          node.realization.getAs<StringAttr>("mask_select_role").getValue(),
          1,
          node.realization.getAs<StringAttr>("score_value_role").getValue()) ||
      !onlineRoleOperandIs(
          cfg,
          node.realization.getAs<StringAttr>("mask_select_role").getValue(),
          2,
          node.realization.getAs<StringAttr>("negative_infinity_role")
              .getValue()) ||
      !onlineRoleOperandIs(
          cfg,
          node.realization.getAs<StringAttr>("mask_select_role").getValue(),
          3,
          node.realization.getAs<StringAttr>("relation_compare_role")
              .getValue()) ||
      accumulatorInstruction.opcode != "max.f32" ||
      !accumulatorInstruction.predicate.empty() ||
      accumulatorInstruction.operands.size() != 3 ||
      !reachesEitherAccumulator(*accumulatorUpdate,
                                accumulatorInstruction.operands[1]) ||
      !onlineRoleOperandIs(
          cfg,
          node.realization.getAs<StringAttr>("accumulator_update_role")
              .getValue(),
          2,
          node.realization.getAs<StringAttr>("mask_select_role").getValue()) ||
      resultInstruction.opcode != "mov.f32" ||
      !resultInstruction.predicate.empty() ||
      resultInstruction.operands.size() != 2 ||
      !reachesEitherAccumulator(result->second,
                                resultInstruction.operands[1]))
    return false;
  if (!onlineScoreContractBindings(
          cfg, node, scoreGeometry, scoreSource, launchExtents,
          scoreReductionExtent, inputParameters, context))
    return false;
  if (!onlineScoreMapChainBindings(cfg, node, scoreValueTensor,
                                   scoreContract))
    return false;
  auto exitTarget = cfg.labels.find(exitInstruction.branchTarget);
  auto backedgeTarget = cfg.labels.find(backedgeInstruction.branchTarget);
  if (exitTarget == cfg.labels.end() ||
      exitTarget->second != resultInstruction.block ||
      backedgeTarget == cfg.labels.end() ||
      backedgeTarget->second != guardInstruction.block)
    return false;
  int64_t bodyBlock = -1;
  llvm::SmallDenseSet<int64_t> guardSuccessors(
      cfg.successors[guardInstruction.block].begin(),
      cfg.successors[guardInstruction.block].end());
  if (guardSuccessors.size() != 2 ||
      !guardSuccessors.contains(resultInstruction.block))
    return false;
  for (int64_t successor : guardSuccessors)
    if (successor != resultInstruction.block)
      bodyBlock = successor;
  if (bodyBlock < 0)
    return false;
  llvm::SmallDenseSet<int64_t> loopBodyBlocks;
  SmallVector<int64_t> worklist{bodyBlock};
  while (!worklist.empty()) {
    int64_t block = worklist.pop_back_val();
    if (block == guardInstruction.block || block == resultInstruction.block ||
        !loopBodyBlocks.insert(block).second)
      continue;
    for (int64_t successor : cfg.successors[block])
      if (successor != guardInstruction.block &&
          successor != resultInstruction.block)
        worklist.push_back(successor);
  }
  for (int64_t block : loopBodyBlocks)
    for (int64_t successor : cfg.successors[block])
      if (successor == resultInstruction.block ||
          (successor == guardInstruction.block &&
           block != backedgeInstruction.block))
        return false;
  auto scoreContractCertificate =
      node.realization.getAs<DictionaryAttr>("score_contract");
  auto contractGuardName = scoreContractCertificate
                               ? scoreContractCertificate.getAs<StringAttr>(
                                     "index_guard_role")
                               : StringAttr();
  auto contractBackedgeName = scoreContractCertificate
                                  ? scoreContractCertificate.getAs<StringAttr>(
                                        "backedge_role")
                                  : StringAttr();
  auto contractGuard = contractGuardName
                           ? cfg.roles.find(contractGuardName.getValue().str())
                           : cfg.roles.end();
  auto contractBackedge =
      contractBackedgeName
          ? cfg.roles.find(contractBackedgeName.getValue().str())
          : cfg.roles.end();
  if (contractGuard == cfg.roles.end() ||
      contractBackedge == cfg.roles.end() ||
      !onlineBlocksAreAcyclic(
          cfg, loopBodyBlocks,
          std::pair<int64_t, int64_t>{
              cfg.instructions[contractBackedge->second].block,
              cfg.instructions[contractGuard->second].block}))
    return false;
  if (cfg.successors[backedgeInstruction.block] !=
      std::vector<int64_t>{guardInstruction.block})
    return false;
  if (!onlineInstructionDominates(cfg, *indexInit, *indexGuard) ||
      !onlineInstructionDominates(cfg, *indexGuard, *accumulatorUpdate) ||
      !onlineInstructionDominates(cfg, *accumulatorUpdate, *indexStep) ||
      !onlineInstructionDominates(cfg, *indexStep, *backedge) ||
      !onlineInstructionDominates(cfg, *exitBranch, result->second))
    return false;
  return true;
}

bool onlineCollectMaxLeaves(const OnlinePtxCfg &cfg, int64_t instruction,
                            const llvm::SmallDenseSet<int64_t> &expected,
                            llvm::SmallDenseSet<int64_t> &found,
                            llvm::SmallDenseSet<int64_t> &active) {
  if (expected.contains(instruction))
    return found.insert(instruction).second;
  if (!active.insert(instruction).second)
    return false;
  const OnlinePtxInstruction &candidate = cfg.instructions[instruction];
  if (candidate.opcode != "max.f32" || candidate.operands.size() != 3) {
    active.erase(instruction);
    return false;
  }
  for (size_t operand = 1; operand <= 2; ++operand) {
    std::set<int64_t> definitions = onlineReachingDefinitions(
        cfg, instruction, candidate.operands[operand]);
    if (definitions.size() != 1 ||
        !onlineCollectMaxLeaves(cfg, *definitions.begin(), expected, found,
                               active)) {
      active.erase(instruction);
      return false;
    }
  }
  active.erase(instruction);
  return true;
}

FailureOr<int64_t> onlineTransportedChild(
    const OnlinePtxCfg &cfg, const OnlineMaxReductionNode &child,
    DictionaryAttr transport) {
  auto kind = transport.getAs<StringAttr>("kind");
  auto role = transport.getAs<StringAttr>("role");
  auto childId = transport.getAs<IntegerAttr>("child");
  if (!kind || !role || !childId || childId.getInt() != child.id)
    return failure();
  auto transported = cfg.roles.find(role.getValue().str());
  if (transported == cfg.roles.end())
    return failure();
  if (kind.getValue() == "register") {
    if (!onlineRoleIs(cfg, role.getValue(), {"mov.f32"}, {child.role}) ||
        !onlineRoleOperandIs(cfg, role.getValue(), 1, child.role))
      return failure();
    return transported->second;
  }
  // Cross-instance transports need an ownership proof that composes the
  // producer and consumer partitions with the shared address or lane map.
  // Until that certificate exists, only same-instance register transport is
  // admitted here.
  return failure();
}

bool onlineMaxReductionBindings(
    const OnlinePtxCfg &cfg, ArrayRef<OnlineMaxReductionNode> nodes,
    DictionaryAttr certificate, StringRef scoreSource,
    StringRef comparisonOpcode,
    const std::set<std::string> &relationSources,
    const OnlineScoreContractGeometry &scoreGeometry,
    ArrayRef<int64_t> launchExtents, int64_t scoreReductionExtent,
    ArrayRef<int64_t> inputParameters, Value scoreValue,
    ContractOp scoreContract, MLIRContext *context) {
  auto root = certificate.getAs<IntegerAttr>("root");
  if (!root)
    return false;
  const OnlineMaxReductionNode *rootNode = nullptr;
  llvm::SmallDenseMap<int64_t, const OnlineMaxReductionNode *> byId;
  for (const OnlineMaxReductionNode &node : nodes) {
    byId.try_emplace(node.id, &node);
    if (node.id == root.getInt())
      rootNode = &node;
    auto role = cfg.roles.find(node.role);
    if (role == cfg.roles.end())
      return false;
    if (node.children.empty()) {
      if (node.kind == "leaf_family") {
        if (!onlineCanonicalLeafFamilyBindings(
                cfg, node, scoreSource, comparisonOpcode, relationSources,
                scoreGeometry, launchExtents,
                scoreReductionExtent, inputParameters, scoreValue,
                scoreContract, context))
          return false;
        continue;
      }
      if (!onlineRoleIs(cfg, node.role, {"max.f32"},
                        {"max_init", "mask_select"}))
        return false;
      continue;
    }
    llvm::SmallDenseMap<int64_t, DictionaryAttr> transports;
    if (node.realization)
      if (auto attributes = node.realization.getAs<ArrayAttr>("transports"))
        for (Attribute attribute : attributes) {
          auto transport = dyn_cast<DictionaryAttr>(attribute);
          auto child =
              transport ? transport.getAs<IntegerAttr>("child")
                        : IntegerAttr();
          if (!child || !transports.try_emplace(child.getInt(), transport).second)
            return false;
        }
    llvm::SmallDenseSet<int64_t> expectedLeaves;
    for (int64_t childId : node.children) {
      const OnlineMaxReductionNode *child = byId.lookup(childId);
      if (!child)
        return false;
      auto childRole = cfg.roles.find(child->role);
      if (childRole == cfg.roles.end())
        return false;
      int64_t leaf = childRole->second;
      if (DictionaryAttr transport = transports.lookup(childId)) {
        FailureOr<int64_t> transported =
            onlineTransportedChild(cfg, *child, transport);
        if (failed(transported))
          return false;
        leaf = *transported;
      }
      if (leaf < 0 || !expectedLeaves.insert(leaf).second)
        return false;
    }
    if (transports.size() > node.children.size())
      return false;
    llvm::SmallDenseSet<int64_t> foundLeaves;
    llvm::SmallDenseSet<int64_t> active;
    if (!onlineCollectMaxLeaves(cfg, role->second, expectedLeaves, foundLeaves,
                               active) ||
        foundLeaves != expectedLeaves)
      return false;
  }
  return rootNode &&
         onlineRoleIs(cfg, "compact_store", {"st.global."},
                      {rootNode->role}) &&
         onlineRoleOperandIs(cfg, "compact_store", 1, rootNode->role);
}

bool onlineMaximumProtocol(const OnlinePtxCfg &cfg, int64_t outputParameter,
                           StringRef comparisonOpcode,
                           DictionaryAttr reductionCertificate,
                           int64_t reductionDimension,
                           int64_t reductionExtent,
                           ArrayRef<Attribute> expectedSources,
                           StringRef scoreSource,
                           Value scoreValue,
                           ContractOp scoreContract,
                           ArrayRef<int64_t> scoreInputParameters,
                           ArrayRef<RelationOp> relations,
                           ArrayRef<AffineExpr> logicalCoordinates,
                           ArrayRef<int64_t> launchExtents,
                           ArrayRef<int64_t> scoreShape,
                           MLIRContext *context) {
  FailureOr<SmallVector<OnlineMaxReductionNode>> nodes =
      onlineMaxReductionNodes(reductionCertificate, reductionDimension,
                              reductionExtent, expectedSources);
  FailureOr<OnlineScoreContractGeometry> scoreGeometry =
      onlineScoreContractGeometry(scoreValue, scoreContract, logicalCoordinates,
                                  reductionDimension, context);
  std::set<std::string> relationSourceKeys;
  for (RelationOp relation : relations)
    relationSourceKeys.insert(onlineSourceKey(
        relation->getAttrOfType<SourceRefAttr>("source")));
  return onlineMaskedScoreProtocol(cfg, comparisonOpcode) &&
         onlineRoleIs(cfg, "max_init", {"mov.f32"}) &&
         llvm::is_contained(
             cfg.instructions[cfg.roles.at("max_init")].operands,
             "0fFF800000") &&
         succeeded(nodes) && succeeded(scoreGeometry) &&
         onlineMaxReductionBindings(cfg, *nodes, reductionCertificate,
                                    scoreSource, comparisonOpcode,
                                    relationSourceKeys,
                                    *scoreGeometry, launchExtents,
                                    reductionExtent, scoreInputParameters,
                                    scoreValue, scoreContract, context) &&
         onlineCompactOutputAddressIsProjected(
             cfg, outputParameter, logicalCoordinates, launchExtents,
             scoreShape, reductionDimension, context) &&
         onlineCompactWriterIsUnique(cfg, logicalCoordinates, launchExtents,
                                     scoreShape, reductionDimension, context,
                                     reductionCertificate.getAs<DictionaryAttr>(
                                         "writer")) &&
         onlineRoleStoresToOutput(cfg, "compact_store", outputParameter);
}

FailureOr<SmallVector<int64_t>> onlineScoreInputParameters(
    DeviceEntryOp entry, MaterializationPlanOp plan, RegionOp region,
    ContractOp contract) {
  SmallVector<MaterializationBufferOp> liveIns;
  for (MaterializationBufferOp buffer :
       plan.getBody().front().getOps<MaterializationBufferOp>())
    if (buffer.getStorage() == MaterializationStorage::External &&
        buffer.getLiveIn() && !buffer.getLiveOut())
      liveIns.push_back(buffer);
  llvm::sort(liveIns, [](MaterializationBufferOp left,
                         MaterializationBufferOp right) {
    return left.getOrdinal() < right.getOrdinal();
  });
  Block &body = region.getBody().front();
  if (liveIns.size() != body.getNumArguments())
    return failure();
  for (auto [index, argument] : llvm::enumerate(body.getArguments()))
    if (liveIns[index].getTensorType() != argument.getType())
      return failure();
  SmallVector<int64_t> parameters;
  for (Value input : contract.getInputs()) {
    auto argument = dyn_cast<BlockArgument>(input);
    if (!argument || argument.getOwner() != &body ||
        argument.getArgNumber() >= liveIns.size())
      return failure();
    int64_t buffer = liveIns[argument.getArgNumber()].getOrdinal();
    auto found = llvm::find(entry.getInputBuffers(), buffer);
    if (found == entry.getInputBuffers().end())
      return failure();
    parameters.push_back(std::distance(entry.getInputBuffers().begin(), found));
  }
  return parameters;
}

bool onlineScaledUpdate(const OnlinePtxCfg &cfg, StringRef scaledRole,
                        StringRef updateRole, StringRef scaleRole,
                        StringRef stateRole, StringRef newTermRole) {
  const bool separate =
      onlineRoleIs(cfg, scaledRole, {"mul.rn.f32"},
                   {scaleRole, stateRole}) &&
      onlineRoleIs(cfg, updateRole, {"add.rn.f32"},
                   {scaledRole, newTermRole});
  const bool orderedFmaOperands =
      (onlineRoleOperandIs(cfg, updateRole, 1, scaleRole) &&
       onlineRoleOperandIs(cfg, updateRole, 2, stateRole)) ||
      (onlineRoleOperandIs(cfg, updateRole, 1, stateRole) &&
       onlineRoleOperandIs(cfg, updateRole, 2, scaleRole));
  const bool fused = onlineRoleIs(cfg, updateRole, {"fma.rn.f32"}) &&
                     orderedFmaOperands &&
                     onlineRoleOperandIs(cfg, updateRole, 3, newTermRole);
  return separate || fused;
}

bool onlineExpOf(const OnlinePtxCfg &cfg, StringRef resultRole,
                 StringRef argumentRole, StringRef scaledArgumentRole) {
  if (onlineRoleIs(cfg, resultRole, {"exp."}, {argumentRole}))
    return true;
  auto scaled = cfg.roles.find(scaledArgumentRole.str());
  return scaled != cfg.roles.end() &&
         onlineRoleIs(cfg, scaledArgumentRole, {"mul.rn.f32"},
                      {argumentRole}) &&
         llvm::is_contained(cfg.instructions[scaled->second].operands,
                            "0f3FB8AA3B") &&
         onlineRoleIs(cfg, resultRole, {"ex2."}, {scaledArgumentRole});
}

bool onlineRoleOperandDefinitionsAre(const OnlinePtxCfg &cfg, StringRef role,
                                     unsigned operand,
                                     ArrayRef<StringRef> definitions) {
  auto consumer = cfg.roles.find(role.str());
  if (consumer == cfg.roles.end() ||
      cfg.instructions[consumer->second].operands.size() <= operand)
    return false;
  std::set<int64_t> expected;
  for (StringRef definition : definitions) {
    auto producer = cfg.roles.find(definition.str());
    if (producer == cfg.roles.end())
      return false;
    expected.insert(producer->second);
  }
  return onlineReachingDefinitions(
             cfg, consumer->second,
             cfg.instructions[consumer->second].operands[operand]) == expected;
}

bool onlineRoleIsUnpredicated(const OnlinePtxCfg &cfg, StringRef role,
                              ArrayRef<StringRef> opcodes) {
  auto found = cfg.roles.find(role.str());
  return found != cfg.roles.end() &&
         cfg.instructions[found->second].predicate.empty() &&
         onlineRoleIs(cfg, role, opcodes);
}

bool onlineSharedStateAddressesAreClosed(const OnlinePtxCfg &cfg,
                                         int64_t threads) {
  auto literalOperand = [&](StringRef role, StringRef literal) {
    auto found = cfg.roles.find(role.str());
    return found != cfg.roles.end() &&
           llvm::is_contained(cfg.instructions[found->second].operands,
                              literal);
  };
  const std::string lOffset = std::to_string(threads * 4);
  const std::string oOffset = std::to_string(threads * 8);
  return onlineRoleIsUnpredicated(cfg, "shared_symbol", {"mov.u64"}) &&
         llvm::is_contained(
             cfg.instructions[cfg.roles.at("shared_symbol")].operands,
             "shuttle_smem") &&
         onlineRoleIsUnpredicated(cfg, "shared_base",
                                  {"cvta.to.shared.u64"}) &&
         onlineRoleOperandIs(cfg, "shared_base", 1, "shared_symbol") &&
         onlineRoleIsUnpredicated(cfg, "leaf_index", {"mov.u32"}) &&
         llvm::is_contained(
             cfg.instructions[cfg.roles.at("leaf_index")].operands,
             "%tid.x") &&
         onlineRoleIsUnpredicated(cfg, "leaf_index_wide", {"cvt.u64.u32"}) &&
         onlineRoleOperandIs(cfg, "leaf_index_wide", 1, "leaf_index") &&
         onlineRoleIsUnpredicated(cfg, "leaf_index_byte_offset",
                                  {"mul.lo.u64"}) &&
         onlineRoleOperandIs(cfg, "leaf_index_byte_offset", 1,
                             "leaf_index_wide") &&
         literalOperand("leaf_index_byte_offset", "4") &&
         onlineRoleIsUnpredicated(cfg, "leaf_m_pointer", {"add.u64"}) &&
         onlineRoleOperandIs(cfg, "leaf_m_pointer", 1, "shared_base") &&
         onlineRoleOperandIs(cfg, "leaf_m_pointer", 2,
                             "leaf_index_byte_offset") &&
         onlineRoleIsUnpredicated(cfg, "leaf_l_pointer", {"add.u64"}) &&
         onlineRoleOperandIs(cfg, "leaf_l_pointer", 1, "leaf_m_pointer") &&
         literalOperand("leaf_l_pointer", lOffset) &&
         onlineRoleIsUnpredicated(cfg, "leaf_o_pointer", {"add.u64"}) &&
         onlineRoleOperandIs(cfg, "leaf_o_pointer", 1, "leaf_m_pointer") &&
         literalOperand("leaf_o_pointer", oOffset) &&
         onlineRoleIsUnpredicated(cfg, "partial_index_wide",
                                  {"cvt.u64.u32"}) &&
         onlineRoleOperandDefinitionsAre(
             cfg, "partial_index_wide", 1,
             {"partial_index_init", "partial_index_step"}) &&
         onlineRoleIsUnpredicated(cfg, "partial_index_byte_offset",
                                  {"mul.lo.u64"}) &&
         onlineRoleOperandIs(cfg, "partial_index_byte_offset", 1,
                             "partial_index_wide") &&
         literalOperand("partial_index_byte_offset", "4") &&
         onlineRoleIsUnpredicated(cfg, "partial_m_pointer", {"add.u64"}) &&
         onlineRoleOperandIs(cfg, "partial_m_pointer", 1, "shared_base") &&
         onlineRoleOperandIs(cfg, "partial_m_pointer", 2,
                             "partial_index_byte_offset") &&
         onlineRoleIsUnpredicated(cfg, "partial_l_pointer", {"add.u64"}) &&
         onlineRoleOperandIs(cfg, "partial_l_pointer", 1,
                             "partial_m_pointer") &&
         literalOperand("partial_l_pointer", lOffset) &&
         onlineRoleIsUnpredicated(cfg, "partial_o_pointer", {"add.u64"}) &&
         onlineRoleOperandIs(cfg, "partial_o_pointer", 1,
                             "partial_m_pointer") &&
         literalOperand("partial_o_pointer", oOffset) &&
         onlineRoleOperandIs(cfg, "leaf_m_store", 0, "leaf_m_pointer") &&
         onlineRoleOperandIs(cfg, "leaf_l_store", 0, "leaf_l_pointer") &&
         onlineRoleOperandIs(cfg, "leaf_o_store", 0, "leaf_o_pointer") &&
         onlineRoleOperandIs(cfg, "partial_m_load", 1,
                             "partial_m_pointer") &&
         onlineRoleOperandIs(cfg, "partial_l_load", 1,
                             "partial_l_pointer") &&
         onlineRoleOperandIs(cfg, "partial_o_load", 1,
                             "partial_o_pointer");
}

bool onlinePartialStateReductionProtocol(const OnlinePtxCfg &cfg,
                                         int64_t threads) {
  auto init = cfg.roles.find("partial_index_init");
  auto guard = cfg.roles.find("partial_index_guard");
  auto exit = cfg.roles.find("partial_exit_branch");
  auto loadM = cfg.roles.find("partial_m_load");
  auto mergeM = cfg.roles.find("partial_merge_max");
  auto step = cfg.roles.find("partial_index_step");
  auto backedge = cfg.roles.find("partial_backedge");
  auto result = cfg.roles.find("state_m");
  auto leaderPredicate = cfg.roles.find("leader_predicate");
  auto leaderExit = cfg.roles.find("leader_exit_branch");
  if (init == cfg.roles.end() || guard == cfg.roles.end() ||
      exit == cfg.roles.end() || loadM == cfg.roles.end() ||
      mergeM == cfg.roles.end() || step == cfg.roles.end() ||
      backedge == cfg.roles.end() || result == cfg.roles.end() ||
      leaderPredicate == cfg.roles.end() || leaderExit == cfg.roles.end() ||
      threads <= 0)
    return false;
  auto check = [](bool value) { return value; };
  const OnlinePtxInstruction &exitInstruction = cfg.instructions[exit->second];
  const OnlinePtxInstruction &backedgeInstruction =
      cfg.instructions[backedge->second];
  auto exitTarget = cfg.labels.find(exitInstruction.branchTarget);
  auto backedgeTarget = cfg.labels.find(backedgeInstruction.branchTarget);
  if (!check(
             onlineRoleIsExactFloatConstant(cfg, "partial_m_init",
                                            "0fFF800000") &&
                 onlineRoleIsExactFloatConstant(cfg, "partial_l_init",
                                                "0f00000000") &&
                 onlineRoleIsExactFloatConstant(cfg, "partial_o_init",
                                                "0f00000000")) ||
      !check(onlineRoleIs(cfg, "partial_index_init", {"mov.u32"}) &&
                               llvm::is_contained(
                                   cfg.instructions[init->second].operands,
                                   "1")) ||
      !check(
             onlineRoleIs(cfg, "partial_index_guard", {"setp.ge.u32"}) &&
                 llvm::is_contained(cfg.instructions[guard->second].operands,
                                    std::to_string(threads)) &&
                 onlineRoleOperandDefinitionsAre(
                     cfg, "partial_index_guard", 1,
                     {"partial_index_init", "partial_index_step"})) ||
      !check(
             exitInstruction.opcode == "bra" &&
                 !exitInstruction.predicate.empty() &&
                 !exitInstruction.negatedPredicate &&
                 onlineRoleDirectlyReaches(cfg, "partial_index_guard",
                                           exit->second) &&
                 backedgeInstruction.opcode == "bra" &&
                 backedgeInstruction.predicate.empty() &&
                 exitTarget != cfg.labels.end() &&
                 exitTarget->second == cfg.instructions[result->second].block &&
                 backedgeTarget != cfg.labels.end() &&
                 backedgeTarget->second ==
                     cfg.instructions[guard->second].block) ||
      !check(
             onlineRoleIs(cfg, "partial_m_load", {"ld.shared.f32"}) &&
                 onlineRoleIs(cfg, "partial_l_load", {"ld.shared.f32"}) &&
                 onlineRoleIs(cfg, "partial_o_load", {"ld.shared.f32"}) &&
                 cfg.instructions[loadM->second].predicate.empty() &&
                 cfg.instructions[cfg.roles.at("partial_l_load")]
                     .predicate.empty() &&
                 cfg.instructions[cfg.roles.at("partial_o_load")]
                     .predicate.empty() &&
                 onlineSharedStateAddressesAreClosed(cfg, threads)) ||
      !check(
             onlineRoleIs(cfg, "partial_merge_max", {"max.f32"},
                          {"partial_m_load"}) &&
                 onlineRoleOperandDefinitionsAre(
                     cfg, "partial_merge_max", 1,
                     {"partial_m_init", "partial_m_update"})) ||
      !check(
             onlineRoleIs(cfg, "partial_left_delta", {"sub.rn.f32"},
                          {"partial_merge_max"}) &&
                 onlineRoleOperandDefinitionsAre(
                     cfg, "partial_left_delta", 1,
                     {"partial_m_init", "partial_m_update"}) &&
                 onlineRoleOperandIs(cfg, "partial_left_delta", 2,
                                     "partial_merge_max") &&
                 onlineRoleIs(cfg, "partial_right_delta", {"sub.rn.f32"},
                              {"partial_m_load", "partial_merge_max"}) &&
                 onlineRoleOperandIs(cfg, "partial_right_delta", 1,
                                     "partial_m_load") &&
                 onlineRoleOperandIs(cfg, "partial_right_delta", 2,
                                     "partial_merge_max") &&
                 onlineExpOf(cfg, "partial_left_rescale",
                             "partial_left_delta",
                             "partial_left_exp_argument") &&
                 onlineExpOf(cfg, "partial_right_rescale",
                             "partial_right_delta",
                             "partial_right_exp_argument") &&
                 onlineRoleIs(cfg, "partial_merge_left_l", {"mul.rn.f32"},
                              {"partial_left_rescale"}) &&
                 onlineRoleOperandDefinitionsAre(
                     cfg, "partial_merge_left_l", 2,
                     {"partial_l_init", "partial_l_update"}) &&
                 onlineRoleIs(cfg, "partial_merge_right_l", {"mul.rn.f32"},
                              {"partial_right_rescale", "partial_l_load"}) &&
                 onlineRoleIs(cfg, "partial_merge_l", {"add.rn.f32"},
                              {"partial_merge_left_l",
                               "partial_merge_right_l"})) ||
      !check(
             onlineRoleIs(cfg, "partial_merge_left_o", {"mul.rn.f32"},
                          {"partial_left_rescale"}) &&
                 onlineRoleOperandDefinitionsAre(
                     cfg, "partial_merge_left_o", 2,
                     {"partial_o_init", "partial_o_update"}) &&
                 onlineRoleIs(cfg, "partial_merge_right_o", {"mul.rn.f32"},
                              {"partial_right_rescale", "partial_o_load"}) &&
                 onlineRoleIs(cfg, "partial_merge_o", {"add.rn.f32"},
                              {"partial_merge_left_o",
                               "partial_merge_right_o"})) ||
      !check(
             onlineRoleIs(cfg, "partial_m_update", {"mov.f32"},
                          {"partial_merge_max"}) &&
                 onlineRoleIs(cfg, "partial_l_update", {"mov.f32"},
                              {"partial_merge_l"}) &&
                 onlineRoleIs(cfg, "partial_o_update", {"mov.f32"},
                              {"partial_merge_o"})) ||
      !check(
             onlineRoleIs(cfg, "partial_index_step", {"add.u32"}) &&
                 cfg.instructions[step->second].predicate.empty() &&
                 onlineRoleOperandDefinitionsAre(
                     cfg, "partial_index_step", 1,
                     {"partial_index_init", "partial_index_step"}) &&
                 llvm::is_contained(cfg.instructions[step->second].operands,
                                    "1")) ||
      !check(
             onlineInstructionDominates(cfg, init->second, guard->second) &&
                 onlineInstructionDominates(cfg, guard->second,
                                            loadM->second) &&
                 onlineInstructionDominates(cfg, loadM->second,
                                            mergeM->second) &&
                 onlineInstructionDominates(cfg, mergeM->second,
                                            step->second) &&
                 onlineInstructionDominates(cfg, step->second,
                                            backedge->second) &&
                 onlineInstructionDominates(cfg, exit->second,
                                            result->second)))
    return false;
  const OnlinePtxInstruction &leaderExitInstruction =
      cfg.instructions[leaderExit->second];
  auto leaderTarget = cfg.labels.find(leaderExitInstruction.branchTarget);
  auto doneTarget = cfg.labels.find("$done");
  return check(
             onlineRoleIs(cfg, "leader_predicate", {"setp.ne.u32"},
                          {"leaf_index"}) &&
                 llvm::is_contained(
                     cfg.instructions[leaderPredicate->second].operands,
                     "0") &&
                 leaderExitInstruction.opcode == "bra" &&
                 !leaderExitInstruction.predicate.empty() &&
                 !leaderExitInstruction.negatedPredicate &&
                 onlineRoleDirectlyReaches(cfg, "leader_predicate",
                                           leaderExit->second) &&
                 leaderTarget != cfg.labels.end() &&
                 doneTarget != cfg.labels.end() &&
                 leaderTarget->second == doneTarget->second &&
                 onlineInstructionDominates(cfg, leaderExit->second,
                                            init->second)) &&
         check(
               onlineRoleIs(cfg, "leaf_state_m", {"mov.f32"},
                            {"max_init"}) &&
                   onlineRoleIs(cfg, "leaf_state_l", {"mov.f32"},
                                {"zero_init"}) &&
                   onlineRoleIs(cfg, "leaf_state_o", {"mov.f32"},
                                {"zero_init"}) &&
                   onlineRoleIs(cfg, "leaf_max_update", {"max.f32"},
                                {"leaf_state_m", "mask_select"}) &&
                   onlineRoleIs(cfg, "leaf_old_delta", {"sub.rn.f32"},
                                {"leaf_state_m", "leaf_max_update"}) &&
                   onlineRoleOperandIs(cfg, "leaf_old_delta", 1,
                                       "leaf_state_m") &&
                   onlineRoleOperandIs(cfg, "leaf_old_delta", 2,
                                       "leaf_max_update") &&
                   onlineExpOf(cfg, "leaf_old_rescale", "leaf_old_delta",
                               "leaf_old_exp_argument") &&
                   onlineRoleIs(cfg, "leaf_score_delta", {"sub.rn.f32"},
                                {"mask_select", "leaf_max_update"}) &&
                   onlineRoleOperandIs(cfg, "leaf_score_delta", 1,
                                       "mask_select") &&
                   onlineRoleOperandIs(cfg, "leaf_score_delta", 2,
                                       "leaf_max_update") &&
                   onlineExpOf(cfg, "leaf_score_weight", "leaf_score_delta",
                               "leaf_score_exp_argument") &&
                   onlineRoleIs(cfg, "leaf_value_weight", {"mul.rn.f32"},
                                {"leaf_score_weight", "value_load"})) &&
         check(
               onlineScaledUpdate(cfg, "leaf_old_l_scaled", "leaf_l_update",
                                  "leaf_old_rescale", "leaf_state_l",
                                  "leaf_score_weight")) &&
         check(
               onlineScaledUpdate(cfg, "leaf_old_o_scaled", "leaf_o_update",
                                  "leaf_old_rescale", "leaf_state_o",
                                  "leaf_value_weight")) &&
         check(
               onlineRoleIs(cfg, "leaf_m_store", {"st.shared.f32"},
                            {"leaf_max_update"}) &&
                   cfg.instructions[cfg.roles.at("leaf_m_store")]
                       .predicate.empty() &&
                   onlineRoleIs(cfg, "leaf_l_store", {"st.shared.f32"},
                                {"leaf_l_update"}) &&
                   cfg.instructions[cfg.roles.at("leaf_l_store")]
                       .predicate.empty() &&
                   onlineRoleIs(cfg, "leaf_o_store", {"st.shared.f32"},
                                {"leaf_o_update"}) &&
                   cfg.instructions[cfg.roles.at("leaf_o_store")]
                       .predicate.empty()) &&
         check(
               onlineRoleIs(cfg, "leaf_barrier", {"bar.sync"}) &&
                   cfg.instructions[cfg.roles.at("leaf_barrier")]
                       .predicate.empty() &&
                   onlineInstructionDominates(
                       cfg, cfg.roles.at("leaf_m_store"),
                       cfg.roles.at("leaf_barrier")) &&
                   onlineInstructionDominates(
                       cfg, cfg.roles.at("leaf_l_store"),
                       cfg.roles.at("leaf_barrier")) &&
                   onlineInstructionDominates(
                       cfg, cfg.roles.at("leaf_o_store"),
                       cfg.roles.at("leaf_barrier")) &&
                   onlineInstructionDominates(cfg,
                                              cfg.roles.at("leaf_barrier"),
                                              loadM->second) &&
                   onlineInstructionDominates(
                       cfg, cfg.roles.at("leaf_barrier"),
                       cfg.roles.at("partial_l_load")) &&
                   onlineInstructionDominates(
                       cfg, cfg.roles.at("leaf_barrier"),
                       cfg.roles.at("partial_o_load"))) &&
         check(
             onlineRoleIs(cfg, "partial_m_result", {"max.f32"},
                          {"max_init"}) &&
                 onlineRoleOperandDefinitionsAre(
                     cfg, "partial_m_result", 1,
                     {"partial_m_init", "partial_m_update"}) &&
                 onlineRoleIs(cfg, "partial_l_result", {"add.rn.f32"},
                              {"zero_init"}) &&
                 onlineRoleOperandDefinitionsAre(
                     cfg, "partial_l_result", 1,
                     {"partial_l_init", "partial_l_update"}) &&
                 onlineRoleIs(cfg, "partial_o_result", {"add.rn.f32"},
                              {"zero_init"}) &&
                 onlineRoleOperandDefinitionsAre(
                     cfg, "partial_o_result", 1,
                     {"partial_o_init", "partial_o_update"}) &&
                 onlineRoleIs(cfg, "state_m", {"mov.f32"},
                              {"partial_m_result"}) &&
                 onlineRoleIs(cfg, "state_l", {"mov.f32"},
                              {"partial_l_result"}) &&
                 onlineRoleIs(cfg, "state_o", {"mov.f32"},
                              {"partial_o_result"}));
}

bool onlineFinalizeProtocol(const OnlinePtxCfg &cfg) {
  const bool mergedDivision =
      onlineRoleIs(cfg, "finalize", {"div.rn.f32"},
                   {"merge_o", "merge_l"}) &&
      onlineRoleOperandIs(cfg, "finalize", 1, "merge_o") &&
      onlineRoleOperandIs(cfg, "finalize", 2, "merge_l");
  const bool mergedReciprocal =
      onlineRoleIs(cfg, "denominator_reciprocal", {"rcp.rn.f32"},
                   {"merge_l"}) &&
      onlineRoleIs(cfg, "finalize", {"mul.rn.f32"},
                   {"merge_o", "denominator_reciprocal"});
  const bool updateDivision =
      onlineRoleIs(cfg, "finalize", {"div.rn.f32"},
                   {"o_update", "l_update"}) &&
      onlineRoleOperandIs(cfg, "finalize", 1, "o_update") &&
      onlineRoleOperandIs(cfg, "finalize", 2, "l_update");
  const bool updateReciprocal =
      onlineRoleIs(cfg, "denominator_reciprocal", {"rcp.rn.f32"},
                   {"l_update"}) &&
      onlineRoleIs(cfg, "finalize", {"mul.rn.f32"},
                   {"o_update", "denominator_reciprocal"});
  return mergedDivision || mergedReciprocal || updateDivision ||
         updateReciprocal;
}

bool onlineStateProtocol(const OnlinePtxCfg &cfg, int64_t outputParameter,
                         StringRef comparisonOpcode, int64_t threads) {
  return onlinePartialStateReductionProtocol(cfg, threads) &&
         onlineMaskedScoreProtocol(cfg, comparisonOpcode) &&
         onlineRoleIs(cfg, "max_init", {"mov.f32"}) &&
         llvm::is_contained(
             cfg.instructions[cfg.roles.at("max_init")].operands,
             "0fFF800000") &&
         onlineRoleIs(cfg, "zero_init", {"mov.f32"}) &&
         llvm::is_contained(
             cfg.instructions[cfg.roles.at("zero_init")].operands,
             "0f00000000") &&
         onlineRoleIs(cfg, "state_m", {"mov.f32"}) &&
         onlineRoleTransitivelyDependsOn(cfg, "state_m", "max_init") &&
         onlineRoleIs(cfg, "state_l", {"mov.f32"}) &&
         onlineRoleTransitivelyDependsOn(cfg, "state_l", "zero_init") &&
         onlineRoleIs(cfg, "state_o", {"mov.f32"}) &&
         onlineRoleTransitivelyDependsOn(cfg, "state_o", "zero_init") &&
         onlineRoleIs(cfg, "max_update", {"max.f32"},
                      {"state_m", "mask_select"}) &&
         onlineRoleIs(cfg, "old_delta", {"sub.rn.f32"},
                      {"state_m", "max_update"}) &&
         onlineRoleOperandIs(cfg, "old_delta", 1, "state_m") &&
         onlineRoleOperandIs(cfg, "old_delta", 2, "max_update") &&
         onlineExpOf(cfg, "old_rescale", "old_delta",
                     "old_exp_argument") &&
         onlineRoleIs(cfg, "score_delta", {"sub.rn.f32"},
                      {"mask_select", "max_update"}) &&
         onlineRoleOperandIs(cfg, "score_delta", 1, "mask_select") &&
         onlineRoleOperandIs(cfg, "score_delta", 2, "max_update") &&
         onlineExpOf(cfg, "score_weight", "score_delta",
                     "score_exp_argument") &&
         onlineRoleIs(cfg, "value_weight", {"mul.rn.f32"},
                      {"score_weight", "value_load"}) &&
         onlineScaledUpdate(cfg, "old_l_scaled", "l_update",
                            "old_rescale", "state_l", "score_weight") &&
         onlineScaledUpdate(cfg, "old_o_scaled", "o_update",
                            "old_rescale", "state_o", "value_weight") &&
         onlineFinalizeProtocol(cfg) &&
         onlineRoleIs(cfg, "public_store", {"st.global."}, {"finalize"}) &&
         onlineRoleStoresToOutput(cfg, "public_store", outputParameter);
}

class VerifyOnlinePtxOraclePass
    : public MutationPass<VerifyOnlinePtxOraclePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyOnlinePtxOraclePass)
  StringRef getArgument() const final {
    return "shuttle-test-verify-online-ptx-oracle";
  }
  void runOnOperation() override {
    MLIRContext *context = getOperation().getContext();
    Builder builder(context);
    const std::vector<std::string> sharedSymbol =
        onlinePtxTokens("%rd0, shuttle_smem");
    const std::vector<std::string> prefixedSharedSymbol =
        onlinePtxTokens("%rd0, evilshuttle_smem");
    const std::vector<std::string> dottedSharedSymbol =
        onlinePtxTokens("%rd0, evil.shuttle_smem");
    const std::vector<std::string> dollarSharedSymbol =
        onlinePtxTokens("%rd0, evil$shuttle_smem");
    if (!llvm::is_contained(sharedSymbol, "shuttle_smem") ||
        llvm::is_contained(prefixedSharedSymbol, "shuttle_smem") ||
        llvm::is_contained(dottedSharedSymbol, "shuttle_smem") ||
        llvm::is_contained(dollarSharedSymbol, "shuttle_smem")) {
      getOperation().emitError(
          "online PTX shared-memory symbol tokenization is not exact");
      return signalPassFailure();
    }
    const OnlineFloatExpression raw{"", "raw_score", {}};
    const OnlineFloatExpression two{"", "0f40000000", {}};
    const OnlineFloatExpression three{"", "0f40400000", {}};
    const OnlineFloatExpression divided = onlineFloatOperation(
        "mul", {raw, onlineFloatOperation("recip", {two})});
    const OnlineFloatExpression fused = onlineFloatOperation(
        "add", {onlineFloatOperation("mul", {raw, two}), three});
    const OnlineFloatExpression reassociated =
        onlineFloatOperation("mul", {raw, two, three});
    constexpr StringLiteral directDividePtx = R"ptx(
.visible .entry expression() {
// shuttle.online.role raw_score
mov.f32 %f0, %f99;
// shuttle.online.role score_value
div.rn.f32 %f1, %f0, 0f40000000;
ret;
}
)ptx";
    constexpr StringLiteral fusedPtx = R"ptx(
.visible .entry expression() {
// shuttle.online.role raw_score
mov.f32 %f0, %f99;
// shuttle.online.role score_value
fma.rn.f32 %f1, 0f40000000, %f0, 0f40400000;
ret;
}
)ptx";
    constexpr StringLiteral reassociatedPtx = R"ptx(
.visible .entry expression() {
// shuttle.online.role raw_score
mov.f32 %f0, %f99;
mul.rn.f32 %f1, 0f40400000, %f0;
// shuttle.online.role score_value
mul.rn.f32 %f2, 0f40000000, %f1;
ret;
}
)ptx";
    constexpr StringLiteral ambiguousPtx = R"ptx(
.visible .entry expression() {
// shuttle.online.role raw_score
mov.f32 %f0, %f99;
mov.f32 %f1, %f0;
@%p0 mov.f32 %f1, 0f40000000;
// shuttle.online.role score_value
mov.f32 %f2, %f1;
ret;
}
)ptx";
    constexpr StringLiteral unsupportedPtx = R"ptx(
.visible .entry expression() {
// shuttle.online.role raw_score
mov.f32 %f0, %f99;
// shuttle.online.role score_value
sin.approx.f32 %f1, %f0;
ret;
}
)ptx";
    if (!onlinePtxMatchesFloatExpression(directDividePtx, divided) ||
        !onlinePtxMatchesFloatExpression(fusedPtx, fused) ||
        !onlinePtxMatchesFloatExpression(reassociatedPtx, reassociated) ||
        onlinePtxMatchesFloatExpression(ambiguousPtx, raw) ||
        onlinePtxMatchesFloatExpression(unsupportedPtx, raw)) {
      getOperation().emitError(
          "normalized FAST expression alternatives are not verifier-bound");
      return signalPassFailure();
    }
    bool hasAlgebraRegion = false;
    getOperation().walk([&](RegionOp) { hasAlgebraRegion = true; });
    if (hasAlgebraRegion) {
      FailureOr<std::pair<Value, ContractOp>> source =
          onlineScoreSourceChain(getOperation());
      if (failed(source)) {
        getOperation().emitError("score semantic certificate has no source chain");
        return signalPassFailure();
      }
      llvm::SmallDenseSet<Operation *> active;
      FailureOr<Attribute> certificate = onlineScoreSemantics(
          source->first, source->second, builder, active);
      auto scoreType = dyn_cast<RankedTensorType>(source->first.getType());
      FoldOp maximumFold;
      getOperation().walk([&](FoldOp fold) {
        if (!maximumFold &&
            foldYieldIs(fold, arith::MaximumFOp::getOperationName()))
          maximumFold = fold;
      });
      SmallVector<AffineExpr> scoreCoordinates;
      if (scoreType)
        for (int64_t dimension = 0; dimension < scoreType.getRank(); ++dimension)
          scoreCoordinates.push_back(getAffineDimExpr(dimension, context));
      ArrayRef<int64_t> reductionDimensions =
          maximumFold ? maximumFold.getReductionDimensions()
                      : ArrayRef<int64_t>();
      FailureOr<OnlineScoreContractGeometry> geometry =
          reductionDimensions.size() == 1
              ? onlineScoreContractGeometry(
                    source->first, source->second, scoreCoordinates,
                    reductionDimensions.front(), context)
              : FailureOr<OnlineScoreContractGeometry>(failure());
      if (failed(certificate) || failed(geometry) ||
          !onlineScoreSemanticCertificateMatches(
              source->first, source->second, *certificate)) {
        getOperation().emitError(
            "score semantic certificate does not compose to its Contract");
        return signalPassFailure();
      }
      SmallVector<std::pair<StringRef, std::optional<StringRef>>> mutations{
          {"indexing_maps", std::nullopt},
          {"source", std::nullopt},
          {"value", std::nullopt},
          {"operands", std::nullopt},
          {"name", arith::MulFOp::getOperationName()},
          {"name", arith::DivFOp::getOperationName()},
          {"name", "math.tanh"}};
      for (auto [key, value] : mutations) {
        if (value && !onlineScoreSemanticCertificateContains(
                         *certificate, key, *value))
          continue;
        bool changed = false;
        Attribute mutated = onlineMutateScoreSemanticCertificate(
            *certificate, key, value, builder, changed);
        if (!changed || onlineScoreSemanticCertificateMatches(
                            source->first, source->second, mutated)) {
          getOperation().emitError(
              "score semantic certificate mutation was not rejected");
          return signalPassFailure();
        }
      }
      llvm::SmallDenseSet<Operation *> expressionActive;
      FailureOr<OnlineFloatExpression> sourceExpression =
          onlineSourceFloatExpression(source->first, source->second,
                                      expressionActive);
      std::string finalOperand;
      SourceRefAttr scoreSource =
          source->second->getAttrOfType<SourceRefAttr>("source");
      const std::string scoreSourceKey =
          scoreSource ? onlineSourceKey(scoreSource) : std::string();
      std::string expressionPtx =
          succeeded(sourceExpression) && !scoreSourceKey.empty()
              ? onlineExpressionPtx(*sourceExpression, scoreSourceKey,
                                    finalOperand)
              : std::string();
      DictionaryAttr scoreContractCertificate = DictionaryAttr::get(
          context,
          {builder.getNamedAttr("result_role",
                                builder.getStringAttr("raw_score"))});
      OnlineMaxReductionNode expressionNode;
      expressionNode.realization = DictionaryAttr::get(
          context,
          {builder.getNamedAttr("score_contract", scoreContractCertificate),
           builder.getNamedAttr("score_value_role",
                                builder.getStringAttr("score_value")),
           builder.getNamedAttr("score_semantics", *certificate)});
      auto acceptsImplementation = [&](StringRef ptx) {
        FailureOr<OnlinePtxCfg> cfg = onlinePtxCfg(ptx);
        return succeeded(cfg) && onlineScoreMapChainBindings(
                                     *cfg, expressionNode, source->first,
                                     source->second);
      };
      const std::string finalLine =
          (Twine("mov.f32 %f999, ") + finalOperand + ";").str();
      auto replaceFinal = [&](StringRef replacement) {
        std::string mutated = expressionPtx;
        size_t position = mutated.find(finalLine);
        if (position != std::string::npos)
          mutated.replace(position, finalLine.size(), replacement.str());
        return mutated;
      };
      std::string wrongTransform = expressionPtx;
      size_t tanh = wrongTransform.find("tanh.approx.f32");
      if (tanh != std::string::npos)
        wrongTransform.replace(tanh, StringRef("tanh.approx.f32").size(),
                               "mov.f32        ");
      std::string wrongRawSource = expressionPtx;
      size_t rawSource = wrongRawSource.find(scoreSourceKey);
      if (rawSource != std::string::npos)
        wrongRawSource.replace(rawSource, scoreSourceKey.size(),
                               "test:wrong:source:0");
      bool everyConstantRejected = true;
      size_t constant = 0;
      bool foundConstant = false;
      while ((constant = expressionPtx.find("0f", constant)) !=
             std::string::npos) {
        foundConstant = true;
        std::string mutated = expressionPtx;
        if (constant + 9 >= mutated.size()) {
          everyConstantRejected = false;
          break;
        }
        mutated[constant + 9] =
            mutated[constant + 9] == '0' ? '1' : '0';
        everyConstantRejected &= !acceptsImplementation(mutated);
        constant += 10;
      }
      bool transformedOperationRejected = true;
      for (StringRef opcode : {StringRef("mul.rn.f32"),
                               StringRef("add.rn.f32"),
                               StringRef("sub.rn.f32"),
                               StringRef("rcp.rn.f32")}) {
        size_t position = 0;
        while ((position = expressionPtx.find(opcode.str(), position)) !=
               std::string::npos) {
          std::string mutated = expressionPtx;
          StringRef replacement = opcode == "mul.rn.f32" ? "add.rn.f32"
                                  : opcode == "add.rn.f32" ? "mul.rn.f32"
                                  : opcode == "sub.rn.f32" ? "add.rn.f32"
                                                            : "mov.f32    ";
          mutated.replace(position, opcode.size(), replacement.str());
          transformedOperationRejected &= !acceptsImplementation(mutated);
          position += opcode.size();
        }
      }
      if (expressionPtx.empty() || !acceptsImplementation(expressionPtx) ||
          acceptsImplementation(replaceFinal("mov.f32 %f999, %f0;")) ||
          acceptsImplementation(
              replaceFinal("mul.rn.f32 %f999, %f0, 0f00000000;")) ||
          acceptsImplementation(
              replaceFinal("mov.f32 %f999, 0f3F800000;")) ||
          !foundConstant || !everyConstantRejected ||
          rawSource == std::string::npos ||
          acceptsImplementation(wrongRawSource) ||
          !transformedOperationRejected ||
          (tanh != std::string::npos &&
           acceptsImplementation(wrongTransform))) {
        getOperation().emitError(
            "score implementation is not bound to source semantics");
        return signalPassFailure();
      }
    }
    auto reductionNode = [&](int64_t id, StringRef role,
                             ArrayRef<int64_t> coverage,
                             ArrayRef<int64_t> children) -> Attribute {
      const int64_t stride = coverage.size() > 1 ? coverage[1] - coverage[0] : 1;
      DictionaryAttr partition = DictionaryAttr::get(
          context,
          {builder.getNamedAttr("offset",
                                builder.getI64IntegerAttr(coverage.front())),
           builder.getNamedAttr("stride", builder.getI64IntegerAttr(stride)),
           builder.getNamedAttr(
               "count", builder.getI64IntegerAttr(coverage.size()))});
      DictionaryAttr syntheticLeafRealization = DictionaryAttr::get(
          context,
          {builder.getNamedAttr("kind",
                                builder.getStringAttr("canonical_loop")),
           builder.getNamedAttr("partition", partition)});
      SmallVector<NamedAttribute> attributes{
          builder.getNamedAttr("id", builder.getI64IntegerAttr(id)),
          builder.getNamedAttr("kind", builder.getStringAttr(
                                           children.empty() ? "leaf_family"
                                                            : "merge")),
          builder.getNamedAttr("role", builder.getStringAttr(role)),
          builder.getNamedAttr(
              "children", DenseI64ArrayAttr::get(context, children))};
      if (children.empty())
        attributes.push_back(builder.getNamedAttr(
            "realization", syntheticLeafRealization));
      return DictionaryAttr::get(
          context, attributes);
    };
    auto reductionLeafFamily =
        [&](int64_t id, StringRef role, ArrayRef<int64_t> coverage,
            DictionaryAttr realization) -> Attribute {
      return DictionaryAttr::get(
          context,
          {builder.getNamedAttr("id", builder.getI64IntegerAttr(id)),
           builder.getNamedAttr("kind", builder.getStringAttr("leaf_family")),
           builder.getNamedAttr("role", builder.getStringAttr(role)),
           builder.getNamedAttr("children",
                                DenseI64ArrayAttr::get(context, {})),
           builder.getNamedAttr("realization", realization)});
    };
    auto reductionCertificate =
        [&](ArrayRef<Attribute> nodes, int64_t root,
            int64_t extent) -> DictionaryAttr {
      ArrayAttr sources = builder.getArrayAttr(
          {builder.getStringAttr("synthetic-source-bound-score-and-relation")});
      return DictionaryAttr::get(
          context,
          {builder.getNamedAttr("version", builder.getI64IntegerAttr(1)),
           builder.getNamedAttr("reduction_dimension",
                                builder.getI64IntegerAttr(4)),
           builder.getNamedAttr("reduction_extent",
                                builder.getI64IntegerAttr(extent)),
           builder.getNamedAttr("nodes", builder.getArrayAttr(nodes)),
           builder.getNamedAttr("source_refs", sources),
           builder.getNamedAttr("root", builder.getI64IntegerAttr(root))});
    };
    SmallVector<Attribute> syntheticSources{
        builder.getStringAttr("synthetic-source-bound-score-and-relation")};
    SmallVector<Attribute> leftDeepNodes{
        reductionNode(0, "leaf0", {0}, {}),
        reductionNode(1, "leaf1", {1}, {}),
        reductionNode(2, "merge01", {0, 1}, {0, 1}),
        reductionNode(3, "leaf2", {2}, {}),
        reductionNode(4, "root", {0, 1, 2}, {2, 3})};
    SmallVector<Attribute> balancedNodes{
        reductionNode(0, "leaf0", {0}, {}),
        reductionNode(1, "leaf1", {1}, {}),
        reductionNode(2, "leaf2", {2}, {}),
        reductionNode(3, "leaf3", {3}, {}),
        reductionNode(4, "left", {0, 1}, {0, 1}),
        reductionNode(5, "right", {2, 3}, {2, 3}),
        reductionNode(6, "root", {0, 1, 2, 3}, {4, 5})};
    SmallVector<Attribute> stridedRegisterNodes{
        reductionNode(0, "even_partial", {0, 2}, {}),
        reductionNode(1, "odd_partial", {1, 3}, {}),
        reductionNode(2, "register_root", {0, 1, 2, 3}, {0, 1})};
    SmallVector<Attribute> naryNodes{
        reductionNode(0, "leaf0", {0}, {}),
        reductionNode(1, "leaf1", {1}, {}),
        reductionNode(2, "leaf2", {2}, {}),
        reductionNode(3, "leaf3", {3}, {}),
        reductionNode(4, "nary_root", {0, 1, 2, 3}, {0, 1, 2, 3})};
    auto canonicalLoopRealization =
        [&](StringRef prefix, int64_t offset, int64_t stride,
            int64_t count) -> DictionaryAttr {
      auto roleName = [&](StringRef role) {
        return builder.getStringAttr(
            prefix.empty() ? role.str() : (Twine(prefix) + "_" + role).str());
      };
      DictionaryAttr scoreContract = DictionaryAttr::get(
          context,
          {builder.getNamedAttr("index_init_role",
                                roleName("contract_index_init")),
           builder.getNamedAttr("index_guard_role",
                                roleName("contract_index_guard")),
           builder.getNamedAttr("index_step_role",
                                roleName("contract_index_step")),
           builder.getNamedAttr("exit_branch_role",
                                roleName("contract_exit_branch")),
           builder.getNamedAttr("backedge_role",
                                roleName("contract_backedge")),
           builder.getNamedAttr("coordinate_role",
                                roleName("contract_coordinate")),
           builder.getNamedAttr("accumulator_init_role",
                                roleName("contract_accumulator_init")),
           builder.getNamedAttr("accumulator_update_role",
                                roleName("contract_accumulator_update")),
           builder.getNamedAttr("result_role", roleName("score_value")),
           builder.getNamedAttr("lhs_linear_index_role",
                                roleName("score_lhs_linear_index")),
           builder.getNamedAttr("lhs_byte_offset_role",
                                roleName("score_lhs_byte_offset")),
           builder.getNamedAttr("lhs_pointer_role",
                                roleName("score_lhs_pointer")),
           builder.getNamedAttr("lhs_load_role",
                                roleName("score_lhs_load")),
           builder.getNamedAttr("rhs_linear_index_role",
                                roleName("score_rhs_linear_index")),
           builder.getNamedAttr("rhs_byte_offset_role",
                                roleName("score_rhs_byte_offset")),
           builder.getNamedAttr("rhs_pointer_role",
                                roleName("score_rhs_pointer")),
           builder.getNamedAttr("rhs_load_role",
                                roleName("score_rhs_load"))});
      return DictionaryAttr::get(
        context,
        {builder.getNamedAttr("kind", builder.getStringAttr("canonical_loop")),
         builder.getNamedAttr(
             "partition",
             DictionaryAttr::get(
                 context,
                 {builder.getNamedAttr("offset", builder.getI64IntegerAttr(offset)),
                  builder.getNamedAttr("stride", builder.getI64IntegerAttr(stride)),
                  builder.getNamedAttr("count", builder.getI64IntegerAttr(count))})),
         builder.getNamedAttr("index_init_role",
                              roleName("loop_index_init")),
         builder.getNamedAttr("index_guard_role",
                              roleName("loop_index_guard")),
         builder.getNamedAttr("index_step_role",
                              roleName("loop_index_step")),
         builder.getNamedAttr("exit_branch_role",
                              roleName("loop_exit_branch")),
         builder.getNamedAttr("backedge_role",
                              roleName("loop_backedge")),
         builder.getNamedAttr("accumulator_init_role",
                              roleName("max_init")),
         builder.getNamedAttr("accumulator_update_role",
                              roleName("loop_max_update")),
         builder.getNamedAttr("coordinate_role",
                              roleName("reduction_coordinate")),
         builder.getNamedAttr("score_lhs_load_role",
                              roleName("score_lhs_load")),
         builder.getNamedAttr("score_rhs_load_role",
                              roleName("score_rhs_load")),
         builder.getNamedAttr("score_lhs_offset_role",
                              roleName("score_lhs_offset")),
         builder.getNamedAttr("score_rhs_offset_role",
                              roleName("score_rhs_offset")),
         builder.getNamedAttr("score_lhs_pointer_role",
                              roleName("score_lhs_pointer")),
         builder.getNamedAttr("score_rhs_pointer_role",
                              roleName("score_rhs_pointer")),
         builder.getNamedAttr("score_zero_role",
                              roleName("score_zero")),
         builder.getNamedAttr("score_value_role",
                              roleName("score_value")),
         builder.getNamedAttr("relation_compare_role",
                              roleName("relation_compare")),
         builder.getNamedAttr("mask_select_role",
                              roleName("mask_select")),
         builder.getNamedAttr("negative_infinity_role",
                              roleName("negative_infinity")),
         builder.getNamedAttr("score_contract", scoreContract)});
    };
    DictionaryAttr canonicalLoop =
        canonicalLoopRealization("", 0, 1, 4);
    SmallVector<Attribute> loopNodes{reductionLeafFamily(
        0, "loop_result", {0, 1, 2, 3}, canonicalLoop)};
    DictionaryAttr evenLoop =
        canonicalLoopRealization("even", 0, 2, 2);
    DictionaryAttr oddLoop =
        canonicalLoopRealization("odd", 1, 2, 2);
    DictionaryAttr registerMerge = DictionaryAttr::get(context, {});
    SmallVector<Attribute> executableStridedNodes{
        reductionLeafFamily(0, "even_loop_result", {0, 2}, evenLoop),
        reductionLeafFamily(1, "odd_loop_result", {1, 3}, oddLoop),
        DictionaryAttr::get(
            context,
            {builder.getNamedAttr("id", builder.getI64IntegerAttr(2)),
             builder.getNamedAttr("kind", builder.getStringAttr("merge")),
             builder.getNamedAttr("role",
                                  builder.getStringAttr("strided_root")),
             builder.getNamedAttr(
                 "children", DenseI64ArrayAttr::get(context, {0, 1})),
             builder.getNamedAttr("realization", registerMerge)})};
    SmallVector<Attribute> missingLeafNodes(leftDeepNodes);
    missingLeafNodes.erase(missingLeafNodes.begin() + 3);
    SmallVector<Attribute> duplicateLeafNodes(leftDeepNodes);
    duplicateLeafNodes[1] = reductionNode(1, "leaf1", {0}, {});
    SmallVector<Attribute> wrongCoordinateNodes(leftDeepNodes);
    wrongCoordinateNodes[3] = reductionNode(3, "leaf2", {3}, {});
    SmallVector<Attribute> lostMergeNodes(leftDeepNodes);
    lostMergeNodes.back() = reductionNode(4, "root", {0, 1, 2}, {0, 3});
    SmallVector<Attribute> partialRootNodes(leftDeepNodes);
    partialRootNodes.back() = reductionNode(4, "root", {0, 1}, {0, 1});
    if (failed(onlineMaxReductionNodes(
            reductionCertificate(leftDeepNodes, 4, 3), 4, 3,
            syntheticSources)) ||
        failed(onlineMaxReductionNodes(
            reductionCertificate(balancedNodes, 6, 4), 4, 4,
            syntheticSources)) ||
        failed(onlineMaxReductionNodes(
            reductionCertificate(loopNodes, 0, 4), 4, 4,
            syntheticSources)) ||
        failed(onlineMaxReductionNodes(
            reductionCertificate(stridedRegisterNodes, 2, 4), 4, 4,
            syntheticSources)) ||
        failed(onlineMaxReductionNodes(
            reductionCertificate(naryNodes, 4, 4), 4, 4,
            syntheticSources)) ||
        succeeded(onlineMaxReductionNodes(
            reductionCertificate(missingLeafNodes, 4, 3), 4, 3,
            syntheticSources)) ||
        succeeded(onlineMaxReductionNodes(
            reductionCertificate(duplicateLeafNodes, 4, 3), 4, 3,
            syntheticSources)) ||
        succeeded(onlineMaxReductionNodes(
            reductionCertificate(wrongCoordinateNodes, 4, 3), 4, 3,
            syntheticSources)) ||
        succeeded(onlineMaxReductionNodes(
            reductionCertificate(lostMergeNodes, 4, 3), 4, 3,
            syntheticSources)) ||
        succeeded(onlineMaxReductionNodes(
            reductionCertificate(partialRootNodes, 4, 3), 4, 3,
            syntheticSources))) {
      getOperation().emitError(
          "online max-reduction coverage accepted a partition near miss");
      return signalPassFailure();
    }
    auto parse = [&](StringRef ptx) -> FailureOr<OnlinePtxCfg> {
      FailureOr<OnlinePtxCfg> cfg = onlinePtxCfg(ptx);
      if (failed(cfg))
        getOperation().emitError("synthetic online PTX oracle case did not parse");
      return cfg;
    };
    FailureOr<OnlinePtxCfg> predicated = parse(R"ptx(
// shuttle.online.role maybe_value
@%p0 mov.f32 %f0, 0f00000000;
// shuttle.online.role use
mov.f32 %f1, %f0;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> branch = parse(R"ptx(
bra.uni LIVE;
// shuttle.online.role dead
mov.f32 %f0, 0f00000000;
LIVE:
// shuttle.online.role live
mov.f32 %f1, 0f00000000;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> correctFma = parse(R"ptx(
// shuttle.online.role scale
mov.f32 %f0, 0f3F800000;
// shuttle.online.role state
mov.f32 %f1, 0f3F800000;
// shuttle.online.role term
mov.f32 %f2, 0f3F800000;
// shuttle.online.role update
fma.rn.f32 %f3, %f0, %f1, %f2;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> wrongFma = parse(R"ptx(
// shuttle.online.role scale
mov.f32 %f0, 0f3F800000;
// shuttle.online.role state
mov.f32 %f1, 0f3F800000;
// shuttle.online.role term
mov.f32 %f2, 0f3F800000;
// shuttle.online.role update
fma.rn.f32 %f3, %f0, %f2, %f1;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> pointer = parse(R"ptx(
ld.param.u64 %rd0, [p0];
setp.eq.u32 %p0, 0, 0;
@%p0 mov.u64 %rd0, 0;
// shuttle.online.role store
st.global.f32 [%rd0], %f0;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> transformedPointer = parse(R"ptx(
ld.param.u64 %rd0, [p0];
and.b64 %rd1, %rd0, 0;
// shuttle.online.role store
st.global.f32 [%rd1], %f0;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> predicatedStore = parse(R"ptx(
ld.param.u64 %rd0, [p0];
setp.eq.u32 %p0, 0, 0;
// shuttle.online.role store
@%p0 st.global.f32 [%rd0], %f0;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> paddedWriter = parse(R"ptx(
// shuttle.online.role compact_writer_selector
mov.u32 %r0, %tid.x;
// shuttle.online.role compact_writer_selector_predicate
setp.eq.u32 %p0, %r0, 0;
// shuttle.online.role compact_output_coordinate
mov.u32 %r1, %ctaid.x;
// shuttle.online.role compact_output_bound
setp.lt.u32 %p1, %r1, 3;
// shuttle.online.role compact_writer_predicate
and.pred %p2, %p0, %p1;
ld.param.u64 %rd0, [p0];
// shuttle.online.role compact_store
@%p2 st.global.f32 [%rd0], %f0;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> coordinate = parse(R"ptx(
// shuttle.online.role iota
mov.u32 %r0, %tid.x;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> zeroedCoordinate = parse(R"ptx(
mul.lo.u32 %r0, %tid.x, 0;
// shuttle.online.role iota
mov.u32 %r1, %r0;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> alternateCoordinate = parse(R"ptx(
mov.u32 %r0, %tid.x;
setp.eq.u32 %p0, 0, 0;
@%p0 mov.u32 %r0, %ctaid.x;
// shuttle.online.role iota
mov.u32 %r1, %r0;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> typeChangedCoordinate = parse(R"ptx(
cvt.rn.f32.u32 %f0, %tid.x;
// shuttle.online.role iota
mov.u32 %r1, %f0;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> floatingAddCoordinate = parse(R"ptx(
add.f32 %f0, %tid.x, 0f00000000;
// shuttle.online.role iota
mov.u32 %r1, %f0;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> overflowingCoordinate = parse(R"ptx(
mul.lo.u32 %r0, %tid.x, 268435456;
div.u32 %r1, %r0, 268435456;
// shuttle.online.role iota
mov.u32 %r2, %r1;
ret;
)ptx");
    FailureOr<OnlinePtxCfg> shared = parse(R"ptx(
mov.u64 %rd0, 0;
// shuttle.online.role expected
mov.f32 %f0, 0f3F800000;
// shuttle.online.role wrong
mov.f32 %f1, 0f00000000;
// shuttle.online.role shared_store
st.shared.f32 [%rd0], %f0;
// shuttle.online.role barrier
bar.sync 0;
// shuttle.online.role shared_load
ld.shared.f32 %f2, [%rd0];
ret;
)ptx");
    const std::string canonicalReductionPtx = R"ptx(
// shuttle.online.role max_init
mov.f32 %f0, 0fFF800000;
// shuttle.online.role loop_index_init
mov.u32 %r0, 0;
LOOP:
// shuttle.online.role loop_index_guard
setp.ge.u32 %p0, %r0, 4;
// shuttle.online.role loop_exit_branch
@%p0 bra.uni DONE;
// shuttle.online.role reduction_coordinate
// shuttle.online.source_ref synthetic-reduction-iota
mov.u32 %r1, %r0;
// shuttle.online.role relation_rhs
mov.u32 %r2, 3;
// shuttle.online.role relation_compare
// shuttle.online.source_ref synthetic-compare
setp.le.u32 %p1, %r1, %r2;
// shuttle.online.role contract_accumulator_init
mov.f32 %f7, 0f00000000;
// shuttle.online.role contract_index_init
mov.u32 %r6, 0;
CONTRACT_LOOP:
// shuttle.online.role contract_index_guard
setp.ge.u32 %p2, %r6, 2;
// shuttle.online.role contract_exit_branch
@%p2 bra.uni CONTRACT_DONE;
// shuttle.online.role contract_coordinate
mov.u32 %r7, %r6;
mul.lo.u32 %r20, %ctaid.x, 8;
mul.lo.u32 %r21, %r1, 2;
add.u32 %r22, %r20, %r21;
add.u32 %r23, %r22, %r7;
// shuttle.online.role score_lhs_linear_index
mov.u32 %r24, %r23;
ld.param.u64 %rd1, [p1];
// shuttle.online.role score_lhs_byte_offset
mul.wide.u32 %rd2, %r24, 4;
// shuttle.online.role score_lhs_pointer
add.u64 %rd3, %rd1, %rd2;
// shuttle.online.role score_lhs_load
ld.global.f32 %f5, [%rd3];
mul.lo.u32 %r25, %r1, 2;
add.u32 %r26, %r25, %r7;
// shuttle.online.role score_rhs_linear_index
mov.u32 %r27, %r26;
ld.param.u64 %rd7, [p2];
// shuttle.online.role score_rhs_byte_offset
mul.wide.u32 %rd8, %r27, 4;
// shuttle.online.role score_rhs_pointer
add.u64 %rd9, %rd7, %rd8;
// shuttle.online.role score_rhs_load
ld.global.f32 %f6, [%rd9];
// shuttle.online.role contract_accumulator_update
fma.rn.f32 %f7, %f5, %f6, %f7;
// shuttle.online.role contract_index_step
add.u32 %r6, %r6, 1;
// shuttle.online.role contract_backedge
bra.uni CONTRACT_LOOP;
CONTRACT_DONE:
// shuttle.online.role score_value
// shuttle.online.source_ref synthetic-source-bound-score-and-relation
mov.f32 %f1, %f7;
// shuttle.online.role negative_infinity
mov.f32 %f2, 0fFF800000;
// shuttle.online.role mask_select
// shuttle.online.source_ref synthetic-select
selp.f32 %f3, %f1, %f2, %p1;
// shuttle.online.role loop_max_update
max.f32 %f0, %f0, %f3;
// shuttle.online.role loop_index_step
add.u32 %r0, %r0, 1;
// shuttle.online.role loop_backedge
bra.uni LOOP;
DONE:
// shuttle.online.role loop_result
mov.f32 %f4, %f0;
ld.param.u64 %rd5, [p0];
// shuttle.online.role compact_output_linear_index
mov.u32 %r10, %ctaid.x;
// shuttle.online.role compact_output_byte_offset
mul.wide.u32 %rd6, %r10, 4;
// shuttle.online.role compact_output_pointer
add.u64 %rd0, %rd5, %rd6;
// shuttle.online.role compact_store
st.global.f32 [%rd0], %f4;
ret;
)ptx";
    FailureOr<OnlinePtxCfg> canonicalReduction = parse(canonicalReductionPtx);
    FailureOr<OnlinePtxCfg> stridedRegisterReduction = parse(R"ptx(
// shuttle.online.role even_max_init
mov.f32 %f0, 0fFF800000;
// shuttle.online.role even_loop_index_init
mov.u32 %r0, 0;
EVEN_LOOP:
// shuttle.online.role even_loop_index_guard
setp.ge.u32 %p0, %r0, 3;
// shuttle.online.role even_loop_exit_branch
@%p0 bra.uni EVEN_DONE;
// shuttle.online.role even_reduction_coordinate
// shuttle.online.source_ref synthetic-reduction-iota
mov.u32 %r1, %r0;
mov.u32 %r2, 3;
// shuttle.online.role even_relation_compare
// shuttle.online.source_ref synthetic-compare
setp.le.u32 %p1, %r1, %r2;
// shuttle.online.role even_contract_accumulator_init
mov.f32 %f20, 0f00000000;
// shuttle.online.role even_contract_index_init
mov.u32 %r6, 0;
EVEN_CONTRACT_LOOP:
// shuttle.online.role even_contract_index_guard
setp.ge.u32 %p4, %r6, 2;
// shuttle.online.role even_contract_exit_branch
@%p4 bra.uni EVEN_CONTRACT_DONE;
// shuttle.online.role even_contract_coordinate
mov.u32 %r7, %r6;
mul.lo.u32 %r20, %ctaid.x, 8;
mul.lo.u32 %r21, %r1, 2;
add.u32 %r22, %r20, %r21;
add.u32 %r23, %r22, %r7;
// shuttle.online.role even_score_lhs_linear_index
mov.u32 %r24, %r23;
ld.param.u64 %rd1, [p1];
// shuttle.online.role even_score_lhs_byte_offset
mul.wide.u32 %rd2, %r24, 4;
// shuttle.online.role even_score_lhs_pointer
add.u64 %rd3, %rd1, %rd2;
// shuttle.online.role even_score_lhs_load
ld.global.f32 %f1, [%rd3];
mul.lo.u32 %r25, %r1, 2;
add.u32 %r26, %r25, %r7;
// shuttle.online.role even_score_rhs_linear_index
mov.u32 %r27, %r26;
ld.param.u64 %rd4, [p2];
// shuttle.online.role even_score_rhs_byte_offset
mul.wide.u32 %rd5, %r27, 4;
// shuttle.online.role even_score_rhs_pointer
add.u64 %rd6, %rd4, %rd5;
// shuttle.online.role even_score_rhs_load
ld.global.f32 %f2, [%rd6];
// shuttle.online.role even_contract_accumulator_update
fma.rn.f32 %f20, %f1, %f2, %f20;
// shuttle.online.role even_contract_index_step
add.u32 %r6, %r6, 1;
// shuttle.online.role even_contract_backedge
bra.uni EVEN_CONTRACT_LOOP;
EVEN_CONTRACT_DONE:
// shuttle.online.role even_score_value
// shuttle.online.source_ref synthetic-source-bound-score-and-relation
mov.f32 %f4, %f20;
// shuttle.online.role even_negative_infinity
mov.f32 %f5, 0fFF800000;
// shuttle.online.role even_mask_select
// shuttle.online.source_ref synthetic-select
selp.f32 %f6, %f4, %f5, %p1;
// shuttle.online.role even_loop_max_update
max.f32 %f0, %f0, %f6;
// shuttle.online.role even_loop_index_step
add.u32 %r0, %r0, 2;
// shuttle.online.role even_loop_backedge
bra.uni EVEN_LOOP;
EVEN_DONE:
// shuttle.online.role even_loop_result
mov.f32 %f7, %f0;
// shuttle.online.role odd_max_init
mov.f32 %f8, 0fFF800000;
// shuttle.online.role odd_loop_index_init
mov.u32 %r3, 1;
ODD_LOOP:
// shuttle.online.role odd_loop_index_guard
setp.ge.u32 %p2, %r3, 4;
// shuttle.online.role odd_loop_exit_branch
@%p2 bra.uni ODD_DONE;
// shuttle.online.role odd_reduction_coordinate
// shuttle.online.source_ref synthetic-reduction-iota
mov.u32 %r4, %r3;
mov.u32 %r5, 3;
// shuttle.online.role odd_relation_compare
// shuttle.online.source_ref synthetic-compare
setp.le.u32 %p3, %r4, %r5;
// shuttle.online.role odd_contract_accumulator_init
mov.f32 %f21, 0f00000000;
// shuttle.online.role odd_contract_index_init
mov.u32 %r8, 0;
ODD_CONTRACT_LOOP:
// shuttle.online.role odd_contract_index_guard
setp.ge.u32 %p5, %r8, 2;
// shuttle.online.role odd_contract_exit_branch
@%p5 bra.uni ODD_CONTRACT_DONE;
// shuttle.online.role odd_contract_coordinate
mov.u32 %r9, %r8;
mul.lo.u32 %r30, %ctaid.x, 8;
mul.lo.u32 %r31, %r4, 2;
add.u32 %r32, %r30, %r31;
add.u32 %r33, %r32, %r9;
// shuttle.online.role odd_score_lhs_linear_index
mov.u32 %r34, %r33;
ld.param.u64 %rd7, [p1];
// shuttle.online.role odd_score_lhs_byte_offset
mul.wide.u32 %rd8, %r34, 4;
// shuttle.online.role odd_score_lhs_pointer
add.u64 %rd9, %rd7, %rd8;
// shuttle.online.role odd_score_lhs_load
ld.global.f32 %f9, [%rd9];
mul.lo.u32 %r35, %r4, 2;
add.u32 %r36, %r35, %r9;
// shuttle.online.role odd_score_rhs_linear_index
mov.u32 %r37, %r36;
ld.param.u64 %rd10, [p2];
// shuttle.online.role odd_score_rhs_byte_offset
mul.wide.u32 %rd11, %r37, 4;
// shuttle.online.role odd_score_rhs_pointer
add.u64 %rd12, %rd10, %rd11;
// shuttle.online.role odd_score_rhs_load
ld.global.f32 %f10, [%rd12];
// shuttle.online.role odd_contract_accumulator_update
fma.rn.f32 %f21, %f9, %f10, %f21;
// shuttle.online.role odd_contract_index_step
add.u32 %r8, %r8, 1;
// shuttle.online.role odd_contract_backedge
bra.uni ODD_CONTRACT_LOOP;
ODD_CONTRACT_DONE:
// shuttle.online.role odd_score_value
// shuttle.online.source_ref synthetic-source-bound-score-and-relation
mov.f32 %f12, %f21;
// shuttle.online.role odd_negative_infinity
mov.f32 %f13, 0fFF800000;
// shuttle.online.role odd_mask_select
// shuttle.online.source_ref synthetic-select
selp.f32 %f14, %f12, %f13, %p3;
// shuttle.online.role odd_loop_max_update
max.f32 %f8, %f8, %f14;
// shuttle.online.role odd_loop_index_step
add.u32 %r3, %r3, 2;
// shuttle.online.role odd_loop_backedge
bra.uni ODD_LOOP;
ODD_DONE:
// shuttle.online.role odd_loop_result
mov.f32 %f15, %f8;
// shuttle.online.role strided_root
max.f32 %f18, %f7, %f15;
ld.param.u64 %rd13, [p0];
// shuttle.online.role compact_store
st.global.f32 [%rd13], %f18;
ret;
)ptx");
    std::string wrongLoopPolarityPtx = canonicalReductionPtx;
    size_t loopPolarity = wrongLoopPolarityPtx.find("@%p0 bra.uni DONE");
    if (loopPolarity != std::string::npos)
      wrongLoopPolarityPtx.replace(loopPolarity, 1, "@!");
    FailureOr<OnlinePtxCfg> wrongLoopPolarity = parse(wrongLoopPolarityPtx);
    std::string wrongLoopTargetPtx = canonicalReductionPtx;
    size_t loopTarget = wrongLoopTargetPtx.find("@%p0 bra.uni DONE");
    if (loopTarget != std::string::npos)
      wrongLoopTargetPtx.replace(
          loopTarget, StringRef("@%p0 bra.uni DONE").size(),
          "@%p0 bra.uni LOOP");
    FailureOr<OnlinePtxCfg> wrongLoopTarget = parse(wrongLoopTargetPtx);
    std::string wrongLoopBoundPtx = canonicalReductionPtx;
    size_t loopBound =
        wrongLoopBoundPtx.find("setp.ge.u32 %p0, %r0, 4");
    if (loopBound != std::string::npos)
      wrongLoopBoundPtx.replace(
          loopBound, StringRef("setp.ge.u32 %p0, %r0, 4").size(),
          "setp.ge.u32 %p0, %r0, 3");
    FailureOr<OnlinePtxCfg> wrongLoopBound = parse(wrongLoopBoundPtx);
    std::string wrongLoopStepPtx = canonicalReductionPtx;
    size_t loopStep = wrongLoopStepPtx.find("add.u32 %r0, %r0, 1");
    if (loopStep != std::string::npos)
      wrongLoopStepPtx.replace(
          loopStep, StringRef("add.u32 %r0, %r0, 1").size(),
          "add.u32 %r0, %r0, 2");
    FailureOr<OnlinePtxCfg> wrongLoopStep = parse(wrongLoopStepPtx);
    std::string alternateBackedgePtx = canonicalReductionPtx;
    size_t alternateBackedge =
        alternateBackedgePtx.find("// shuttle.online.role loop_index_step");
    if (alternateBackedge != std::string::npos)
      alternateBackedgePtx.insert(alternateBackedge,
                                  "@%p1 bra.uni LOOP;\n");
    FailureOr<OnlinePtxCfg> alternateBackedgeValue =
        parse(alternateBackedgePtx);
    std::string internalCyclePtx = canonicalReductionPtx;
    size_t cycleStart =
        internalCyclePtx.find("// shuttle.online.role loop_max_update");
    size_t cycleEnd =
        internalCyclePtx.find("// shuttle.online.role loop_index_step");
    if (cycleStart != std::string::npos && cycleEnd != std::string::npos) {
      internalCyclePtx.insert(cycleStart, "INTERNAL_CYCLE:\n");
      cycleEnd = internalCyclePtx.find(
          "// shuttle.online.role loop_index_step", cycleStart);
      internalCyclePtx.insert(cycleEnd, "@%p1 bra.uni INTERNAL_CYCLE;\n");
    }
    FailureOr<OnlinePtxCfg> internalCycleValue = parse(internalCyclePtx);
    std::string wrongContractBoundPtx = canonicalReductionPtx;
    size_t wrongContractBound =
        wrongContractBoundPtx.find("setp.ge.u32 %p2, %r6, 2");
    if (wrongContractBound != std::string::npos)
      wrongContractBoundPtx.replace(
          wrongContractBound,
          StringRef("setp.ge.u32 %p2, %r6, 2").size(),
          "setp.ge.u32 %p2, %r6, 1");
    FailureOr<OnlinePtxCfg> wrongContractBoundValue =
        parse(wrongContractBoundPtx);
    std::string omittedOuterCoordinatePtx = canonicalReductionPtx;
    size_t omittedOuterCoordinate =
        omittedOuterCoordinatePtx.find("mul.lo.u32 %r20, %ctaid.x, 8");
    if (omittedOuterCoordinate != std::string::npos)
      omittedOuterCoordinatePtx.replace(
          omittedOuterCoordinate,
          StringRef("mul.lo.u32 %r20, %ctaid.x, 8").size(),
          "mov.u32 %r20, 0                  ");
    FailureOr<OnlinePtxCfg> omittedOuterCoordinateValue =
        parse(omittedOuterCoordinatePtx);
    std::string swappedContractAbiPtx = canonicalReductionPtx;
    size_t swappedContractAbi =
        swappedContractAbiPtx.find("ld.param.u64 %rd1, [p1]");
    if (swappedContractAbi != std::string::npos)
      swappedContractAbiPtx.replace(
          swappedContractAbi,
          StringRef("ld.param.u64 %rd1, [p1]").size(),
          "ld.param.u64 %rd1, [p2]");
    FailureOr<OnlinePtxCfg> swappedContractAbiValue =
        parse(swappedContractAbiPtx);
    std::string predicatedUpdatePtx = canonicalReductionPtx;
    size_t predicatedUpdate =
        predicatedUpdatePtx.find("max.f32 %f0, %f0, %f3");
    if (predicatedUpdate != std::string::npos)
      predicatedUpdatePtx.insert(predicatedUpdate, "@%p1 ");
    FailureOr<OnlinePtxCfg> predicatedUpdateValue =
        parse(predicatedUpdatePtx);
    std::string predicatedStepPtx = canonicalReductionPtx;
    size_t predicatedStep = predicatedStepPtx.find("add.u32 %r0, %r0, 1");
    if (predicatedStep != std::string::npos)
      predicatedStepPtx.insert(predicatedStep, "@%p1 ");
    FailureOr<OnlinePtxCfg> predicatedStepValue = parse(predicatedStepPtx);
    std::string wrongInitializerPtx = canonicalReductionPtx;
    size_t wrongInitializer =
        wrongInitializerPtx.find("mov.f32 %f0, 0fFF800000");
    if (wrongInitializer != std::string::npos)
      wrongInitializerPtx.replace(
          wrongInitializer, StringRef("mov.f32 %f0, 0fFF800000").size(),
          "mov.f32 %f0, 0f00000000");
    FailureOr<OnlinePtxCfg> wrongInitializerValue =
        parse(wrongInitializerPtx);
    std::string constantScorePtx = canonicalReductionPtx;
    size_t scoreFma = constantScorePtx.find("mov.f32 %f1, %f7");
    if (scoreFma != std::string::npos)
      constantScorePtx.replace(
          scoreFma, StringRef("mov.f32 %f1, %f7").size(),
          "mov.f32 %f1, 0f40000000");
    FailureOr<OnlinePtxCfg> constantScore = parse(constantScorePtx);
    std::string annihilatedScorePtx = canonicalReductionPtx;
    size_t annihilatedScore = annihilatedScorePtx.find(
        "fma.rn.f32 %f7, %f5, %f6, %f7");
    if (annihilatedScore != std::string::npos)
      annihilatedScorePtx.replace(
          annihilatedScore,
          StringRef("fma.rn.f32 %f7, %f5, %f6, %f7").size(),
          "fma.rn.f32 %f7, %f5, 0f00000000, %f7");
    FailureOr<OnlinePtxCfg> annihilatedScoreValue =
        parse(annihilatedScorePtx);
    std::string staleScorePtx = canonicalReductionPtx;
    size_t staleScore = staleScorePtx.find("mul.lo.u32 %r21, %r1, 2");
    if (staleScore != std::string::npos)
      staleScorePtx.replace(
          staleScore, StringRef("mul.lo.u32 %r21, %r1, 2").size(),
          "mul.lo.u32 %r21, %r2, 2");
    FailureOr<OnlinePtxCfg> staleScoreValue = parse(staleScorePtx);
    std::string wrongRootValuePtx = canonicalReductionPtx;
    size_t rootValue = wrongRootValuePtx.find("[%rd0], %f4");
    if (rootValue != std::string::npos)
      wrongRootValuePtx.replace(rootValue, StringRef("[%rd0], %f4").size(),
                                "[%rd0], %f1");
    FailureOr<OnlinePtxCfg> wrongRootValue = parse(wrongRootValuePtx);
    std::string wrongOutputCoordinatePtx = canonicalReductionPtx;
    size_t outputCoordinate =
        wrongOutputCoordinatePtx.find("mov.u32 %r10, %ctaid.x");
    if (outputCoordinate != std::string::npos)
      wrongOutputCoordinatePtx.replace(
          outputCoordinate, StringRef("mov.u32 %r10, %ctaid.x").size(),
          "mov.u32 %r10, 0       ");
    FailureOr<OnlinePtxCfg> wrongOutputCoordinate =
        parse(wrongOutputCoordinatePtx);
    if (failed(predicated) || failed(branch) || failed(correctFma) ||
        failed(wrongFma) || failed(pointer) || failed(transformedPointer) ||
        failed(predicatedStore) || failed(coordinate) ||
        failed(paddedWriter) ||
        failed(zeroedCoordinate) || failed(alternateCoordinate) ||
        failed(typeChangedCoordinate) || failed(floatingAddCoordinate) ||
        failed(overflowingCoordinate) || failed(shared) ||
        failed(canonicalReduction) || failed(stridedRegisterReduction) ||
        failed(wrongLoopPolarity) ||
        failed(wrongLoopTarget) || failed(wrongLoopBound) ||
        failed(wrongLoopStep) || failed(alternateBackedgeValue) ||
        failed(internalCycleValue) ||
        failed(wrongContractBoundValue) ||
        failed(omittedOuterCoordinateValue) ||
        failed(swappedContractAbiValue) || failed(predicatedUpdateValue) ||
        failed(predicatedStepValue) || failed(wrongInitializerValue) ||
        failed(constantScore) ||
        failed(annihilatedScoreValue) ||
        failed(staleScoreValue) || failed(wrongRootValue) ||
        failed(wrongOutputCoordinate))
      return signalPassFailure();
    FailureOr<SmallVector<OnlineMaxReductionNode>> loopCertificateNodes =
        onlineMaxReductionNodes(reductionCertificate(loopNodes, 0, 4), 4, 4,
                                syntheticSources);
    FailureOr<SmallVector<OnlineMaxReductionNode>> stridedCertificateNodes =
        onlineMaxReductionNodes(
            reductionCertificate(executableStridedNodes, 2, 4), 4, 4,
            syntheticSources);
    OnlinePtxCfg missingFamilyIota = *stridedRegisterReduction;
    missingFamilyIota.sources["synthetic-reduction-iota"].pop_back();
    SmallVector<AffineExpr> syntheticLogicalCoordinates{
        getAffineDimExpr(0, context), getAffineDimExpr(6, context)};
    SmallVector<int64_t> syntheticLaunchExtents{3, 1, 1, 1, 1, 1, 1};
    SmallVector<int64_t> duplicateWriterLaunchExtents{3, 1, 1, 1, 1, 1, 2};
    SmallVector<int64_t> syntheticScoreShape{3, 4};
    DictionaryAttr paddedWriterCertificate = DictionaryAttr::get(
        context,
        {builder.getNamedAttr("predicate_role",
                              builder.getStringAttr("compact_writer_predicate")),
         builder.getNamedAttr("selector_role",
                              builder.getStringAttr("compact_writer_selector")),
         builder.getNamedAttr(
             "selector_predicate_role",
             builder.getStringAttr("compact_writer_selector_predicate")),
         builder.getNamedAttr(
             "bounds",
             builder.getArrayAttr({DictionaryAttr::get(
                 context,
                 {builder.getNamedAttr("dimension",
                                       builder.getI64IntegerAttr(0)),
                  builder.getNamedAttr(
                      "predicate_role",
                      builder.getStringAttr("compact_output_bound"))})}))});
    SmallVector<AffineExpr> paddedLogicalCoordinates{
        getAffineDimExpr(0, context), getAffineConstantExpr(0, context)};
    SmallVector<int64_t> paddedLaunchExtents{4, 1, 1, 2, 1, 1, 1};
    OnlineScoreContractGeometry syntheticScoreGeometry{
        {getAffineDimExpr(0, context) * 8 +
             getAffineDimExpr(7, context) * 2 +
             getAffineDimExpr(8, context),
         getAffineDimExpr(7, context) * 2 + getAffineDimExpr(8, context)},
        {24, 8},
        2};
    SmallVector<int64_t> syntheticScoreParameters{1, 2};
    std::set<std::string> syntheticRelationSources{
        "synthetic-compare", "synthetic-select"};
    auto syntheticLeafBindings = [&](const OnlinePtxCfg &cfg,
                                     const OnlineMaxReductionNode &node) {
      return onlineCanonicalLeafFamilyBindings(
          cfg, node, "synthetic-source-bound-score-and-relation",
          "setp.le.u32", syntheticRelationSources,
          syntheticScoreGeometry, syntheticLaunchExtents, 4,
          syntheticScoreParameters, Value(), ContractOp(), context);
    };
    auto syntheticMaxBindings =
        [&](const OnlinePtxCfg &cfg,
            ArrayRef<OnlineMaxReductionNode> nodes,
            DictionaryAttr certificate) {
          return onlineMaxReductionBindings(
              cfg, nodes, certificate,
              "synthetic-source-bound-score-and-relation",
              "setp.le.u32", syntheticRelationSources,
              syntheticScoreGeometry, syntheticLaunchExtents, 4,
              syntheticScoreParameters, Value(), ContractOp(), context);
        };
    auto indexMatches = [&](const OnlinePtxCfg &cfg, AffineExpr expected) {
      int64_t iota = cfg.roles.at("iota");
      std::set<std::pair<int64_t, std::string>> active;
      OnlineIndexAlternatives alternatives = onlineIndexAlternatives(
          cfg, iota, cfg.instructions[iota].operands[1],
          getOperation().getContext(), active);
      SmallVector<int64_t> launchExtents{1, 1, 1, 17, 1, 1, 1};
      return alternatives.valid && !alternatives.values.empty() &&
             llvm::all_of(alternatives.values, [&](AffineExpr value) {
               return value == expected &&
                      onlineIndexExprFitsLaunchDomain(value, launchExtents);
             }) &&
             llvm::all_of(alternatives.intermediates, [&](AffineExpr value) {
               return onlineIndexExprFitsLaunchDomain(value, launchExtents);
             });
    };
    if (succeeded(stridedCertificateNodes) &&
        (!onlineLeafCoordinatesCarrySource(
             *stridedRegisterReduction, *stridedCertificateNodes,
             "synthetic-reduction-iota") ||
         !onlineLeafCoordinatesCarrySource(
             *canonicalReduction, *loopCertificateNodes,
             "synthetic-reduction-iota") ||
         onlineLeafCoordinatesCarrySource(
             missingFamilyIota, *stridedCertificateNodes,
             "synthetic-reduction-iota"))) {
      getOperation().emitError(
          "leaf-family coordinates are not exact Relation source occurrences");
      return signalPassFailure();
    }
    if (succeeded(stridedCertificateNodes) &&
        !onlineScoreContractBindings(
            *stridedRegisterReduction, (*stridedCertificateNodes)[0],
            syntheticScoreGeometry,
            "synthetic-source-bound-score-and-relation",
            syntheticLaunchExtents, 4, syntheticScoreParameters, context)) {
      getOperation().emitError("even score Contract is not executable");
      return signalPassFailure();
    }
    if (succeeded(stridedCertificateNodes) &&
        !syntheticLeafBindings(*stridedRegisterReduction,
                               (*stridedCertificateNodes)[0])) {
      getOperation().emitError("even strided leaf is not executable");
      return signalPassFailure();
    }
    if (succeeded(stridedCertificateNodes)) {
      llvm::SmallDenseSet<int64_t> expected{
          stridedRegisterReduction->roles.at("even_loop_result"),
          stridedRegisterReduction->roles.at("odd_loop_result")};
      llvm::SmallDenseSet<int64_t> found;
      llvm::SmallDenseSet<int64_t> active;
      bool collected = onlineCollectMaxLeaves(
          *stridedRegisterReduction,
          stridedRegisterReduction->roles.at("strided_root"), expected, found,
          active);
      bool storeClosed = onlineRoleIs(*stridedRegisterReduction,
                                      "compact_store", {"st.global."},
                                      {"strided_root"});
      bool storedRoot = onlineRoleOperandIs(
          *stridedRegisterReduction, "compact_store", 1, "strided_root");
      if (!collected || found != expected || !storeClosed || !storedRoot) {
        const auto &rootInstruction = stridedRegisterReduction->instructions[
            stridedRegisterReduction->roles.at("strided_root")];
        std::set<int64_t> leftDefs = onlineReachingDefinitions(
            *stridedRegisterReduction,
            stridedRegisterReduction->roles.at("strided_root"),
            rootInstruction.operands[1]);
        std::set<int64_t> rightDefs = onlineReachingDefinitions(
            *stridedRegisterReduction,
            stridedRegisterReduction->roles.at("strided_root"),
            rootInstruction.operands[2]);
        getOperation().emitError()
            << "strided register merge is not executable: collect="
            << collected << " coverage=" << (found == expected)
            << " store=" << storeClosed << " value=" << storedRoot
            << " left_defs=" << leftDefs.size()
            << " left_first=" << (leftDefs.empty() ? -2 : *leftDefs.begin())
            << " right_defs=" << rightDefs.size()
            << " right_first="
            << (rightDefs.empty() ? -2 : *rightDefs.begin());
        return signalPassFailure();
      }
    }
    if (succeeded(stridedCertificateNodes) &&
        !syntheticLeafBindings(*stridedRegisterReduction,
                               (*stridedCertificateNodes)[1])) {
      getOperation().emitError("odd strided leaf is not executable");
      return signalPassFailure();
    }
    if (failed(stridedCertificateNodes) ||
        !syntheticMaxBindings(
            *stridedRegisterReduction, *stridedCertificateNodes,
            reductionCertificate(executableStridedNodes, 2, 4))) {
      getOperation().emitError(
          "strided register max-reduction certificate is not executable");
      return signalPassFailure();
    }
    if (onlineRoleDirectlyReaches(*predicated, "maybe_value",
                                  predicated->roles.at("use")) ||
        onlineRoleIs(*branch, "dead", {"mov.f32"}) ||
        !onlineRoleIs(*branch, "live", {"mov.f32"}) ||
        !onlineScaledUpdate(*correctFma, "unused", "update", "scale",
                            "state", "term") ||
        onlineScaledUpdate(*wrongFma, "unused", "update", "scale",
                           "state", "term") ||
        onlineRoleStoresToOutput(*pointer, "store", 0) ||
        onlineRoleStoresToOutput(*transformedPointer, "store", 0) ||
        onlineRoleStoresToOutput(*predicatedStore, "store", 0) ||
        !indexMatches(*coordinate,
          getAffineDimExpr(3, context)) ||
        indexMatches(*zeroedCoordinate,
                     getAffineDimExpr(3, context)) ||
        indexMatches(*alternateCoordinate,
                     getAffineDimExpr(3, context)) ||
        indexMatches(*typeChangedCoordinate,
                     getAffineDimExpr(3, context)) ||
        indexMatches(*floatingAddCoordinate,
                     getAffineDimExpr(3, context)) ||
        indexMatches(*overflowingCoordinate,
                     getAffineDimExpr(3, context)) ||
        !onlineSharedPairIsClosed(*shared, "shared_store", "expected",
                                  "barrier", "shared_load") ||
        onlineSharedPairIsClosed(*shared, "shared_store", "wrong", "barrier",
                                 "shared_load") ||
        failed(loopCertificateNodes) ||
        !syntheticMaxBindings(*canonicalReduction, *loopCertificateNodes,
                              reductionCertificate(loopNodes, 0, 4)) ||
        onlineMaxReductionBindings(
            *canonicalReduction, *loopCertificateNodes,
            reductionCertificate(loopNodes, 0, 4),
            "synthetic-source-bound-score-and-relation", "setp.le.u32",
            syntheticRelationSources, syntheticScoreGeometry,
            paddedLaunchExtents, 4, syntheticScoreParameters, Value(),
            ContractOp(), context) ||
        syntheticMaxBindings(*wrongLoopPolarity, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*wrongLoopTarget, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*wrongLoopBound, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*wrongLoopStep, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*alternateBackedgeValue, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*internalCycleValue, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*wrongContractBoundValue, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*omittedOuterCoordinateValue,
                             *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*swappedContractAbiValue, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*predicatedUpdateValue, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*predicatedStepValue, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*wrongInitializerValue, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*constantScore, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*annihilatedScoreValue, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*staleScoreValue, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        syntheticMaxBindings(*wrongRootValue, *loopCertificateNodes,
                             reductionCertificate(loopNodes, 0, 4)) ||
        !onlineCompactOutputAddressIsProjected(
            *canonicalReduction, 0, syntheticLogicalCoordinates,
            syntheticLaunchExtents, syntheticScoreShape, 1, context) ||
        !onlineCompactWriterIsUnique(
            *canonicalReduction, syntheticLogicalCoordinates,
            syntheticLaunchExtents, syntheticScoreShape, 1, context) ||
        onlineCompactWriterIsUnique(
            *canonicalReduction, syntheticLogicalCoordinates,
            duplicateWriterLaunchExtents, syntheticScoreShape, 1, context) ||
        !onlineCompactWriterIsUnique(
            *paddedWriter, paddedLogicalCoordinates, paddedLaunchExtents,
            syntheticScoreShape, 1, context, paddedWriterCertificate) ||
        onlineCompactOutputAddressIsProjected(
            *wrongOutputCoordinate, 0, syntheticLogicalCoordinates,
            syntheticLaunchExtents, syntheticScoreShape, 1, context)) {
      getOperation().emitError(
          "online PTX oracle accepted a control/dataflow near miss");
      return signalPassFailure();
    }
    llvm::outs() << "online_ptx_oracle_mutations = complete\n";
  }
};

void refreshExecutableBundle(ModuleOp module, bool refreshCodeDigest);

class VerifyOnlineGpuExecutionPass
    : public MutationPass<VerifyOnlineGpuExecutionPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyOnlineGpuExecutionPass)
  StringRef getArgument() const final {
    return "shuttle-test-verify-online-gpu-execution";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    constexpr llvm::StringLiteral kind =
        "online_normalized_exponential_weighted_contract";

    MaterializationPlanOp plan;
    llvm::SmallDenseSet<int64_t> onlineTasks;
    llvm::SmallDenseMap<int64_t, MaterializationTaskOp> materializationTasks;
    llvm::SmallDenseSet<Attribute> certifiedSources;
    for (MaterializationPlanOp candidate :
         module.getOps<MaterializationPlanOp>()) {
      if (candidate.getPolicy() != NumericalPolicy::Fast)
        continue;
      for (MaterializationTaskOp task :
           candidate.getBody().front().getOps<MaterializationTaskOp>()) {
        if (!task.getCompositionKind() || *task.getCompositionKind() != kind)
          continue;
        plan = candidate;
        onlineTasks.insert(task.getOrdinal());
        materializationTasks.try_emplace(task.getOrdinal(), task);
        std::optional<ArrayAttr> sources = task.getCompositionSources();
        for (Attribute source : *sources)
          certifiedSources.insert(source);
        if (auto recomputed = task->getAttrOfType<ArrayAttr>(
                "composition_recomputed_sources"))
          certifiedSources.insert(recomputed.begin(), recomputed.end());
      }
    }
    if (!plan || onlineTasks.empty()) {
      module.emitError("requires one FAST online execution certificate");
      return signalPassFailure();
    }

    RegionOp region;
    FoldOp maximumFold;
    Type representativeScoreType;
    llvm::SmallDenseSet<Attribute> relationSources;
    SmallVector<RelationOp> relationOperations;
    std::optional<std::string> comparisonOpcode;
    module.walk([&](RegionOp candidate) {
      for (Operation &operation :
           candidate.getBody().front().without_terminator()) {
        Attribute source = operation.getAttr("source");
        if (!source || !certifiedSources.contains(source))
          continue;
        if (auto fold = dyn_cast<FoldOp>(operation);
            fold && foldYieldIs(fold, arith::MaximumFOp::getOperationName())) {
          region = candidate;
          maximumFold = fold;
          representativeScoreType = fold.getInputs().front().getType();
        }
        if (auto relation = dyn_cast<RelationOp>(operation)) {
          relationSources.insert(source);
          relationOperations.push_back(relation);
          if (relation.getSemantics() == RelationSemantics::Compare) {
            FailureOr<std::string> opcode = onlineComparisonOpcode(relation);
            if (failed(opcode) ||
                (comparisonOpcode && *comparisonOpcode != *opcode)) {
              comparisonOpcode.reset();
              return;
            }
            comparisonOpcode = std::move(*opcode);
          }
        }
      }
    });
    SmallVector<ContractOp> scoreContracts;
    llvm::SmallDenseSet<Operation *> scoreAncestors;
    std::function<void(Value)> collectScoreContracts = [&](Value value) {
      Operation *definition = value.getDefiningOp();
      if (!definition || !scoreAncestors.insert(definition).second)
        return;
      if (auto contract = dyn_cast<ContractOp>(definition))
        scoreContracts.push_back(contract);
      for (Value operand : definition->getOperands())
        collectScoreContracts(operand);
    };
    if (maximumFold)
      collectScoreContracts(maximumFold.getInputs().front());
    Value scoreValue;
    if (maximumFold && scoreContracts.size() == 1) {
      auto select = maximumFold.getInputs().front().getDefiningOp<RelationOp>();
      if (select && select.getSemantics() == RelationSemantics::Select &&
          select.getInputs().size() == 3) {
        for (Value candidate : select.getInputs().drop_front()) {
          llvm::SmallDenseSet<Operation *> active;
          if (!onlineValueDependsOnOperation(
                  candidate, scoreContracts.front().getOperation(), active))
            continue;
          if (scoreValue) {
            scoreValue = Value();
            break;
          }
          scoreValue = candidate;
        }
      }
    }
    SourceRefAttr scoreContractSource =
        scoreContracts.size() == 1
            ? scoreContracts.front()->getAttrOfType<SourceRefAttr>("source")
            : SourceRefAttr();
    if (!region || !maximumFold || !representativeScoreType ||
        !scoreContractSource || !scoreValue || relationSources.empty() ||
        !comparisonOpcode ||
        llvm::any_of(relationSources, [&](Attribute source) {
          return !certifiedSources.contains(source);
        })) {
      module.emitError("online GPU certificate does not apply its Relation");
      return signalPassFailure();
    }
    for (MaterializationBufferOp buffer :
         plan.getBody().front().getOps<MaterializationBufferOp>()) {
      if (buffer.getStorage() == MaterializationStorage::Temporary &&
          buffer.getTensorType() == representativeScoreType) {
        module.emitError("online GPU execution materializes representative scores");
        return signalPassFailure();
      }
    }

    SchedulePlanOp schedule;
    for (SchedulePlanOp candidate : module.getOps<SchedulePlanOp>())
      if (candidate.getSourcePlanFingerprint() == plan.getFingerprint())
        schedule = candidate;
    DeviceModuleOp device;
    for (DeviceModuleOp candidate : module.getOps<DeviceModuleOp>())
      if (schedule && candidate.getSourceScheduleFingerprint() ==
                          schedule.getFingerprint())
        device = candidate;
    InvocationAbiOp abi;
    for (InvocationAbiOp candidate : module.getOps<InvocationAbiOp>())
      if (schedule && candidate.getSourcePlanFingerprint() ==
                          plan.getFingerprint() &&
          candidate.getSourceScheduleFingerprint() == schedule.getFingerprint())
        abi = candidate;
    if (!schedule || !device || !abi ||
        device.getPolicy() != NumericalPolicy::Fast) {
      module.emitError("online GPU source, schedule, device, and ABI are not closed");
      return signalPassFailure();
    }

    llvm::SmallDenseMap<int64_t, int64_t> scheduleSources;
    llvm::SmallDenseMap<int64_t, ScheduleTaskOp> scheduleTasks;
    for (ScheduleTaskOp task :
         schedule.getBody().front().getOps<ScheduleTaskOp>()) {
      scheduleSources.try_emplace(task.getOrdinal(), task.getSourceTask());
      scheduleTasks.try_emplace(task.getOrdinal(), task);
    }
    llvm::SmallDenseMap<int64_t, MaterializationBufferOp> planBuffers;
    for (MaterializationBufferOp buffer :
         plan.getBody().front().getOps<MaterializationBufferOp>())
      planBuffers.try_emplace(buffer.getOrdinal(), buffer);
    llvm::SmallDenseSet<int64_t> coveredTasks;
    bool compactState = false;
    bool publicState = false;
    for (DeviceEntryOp entry :
         device.getBody().front().getOps<DeviceEntryOp>()) {
      auto source = scheduleSources.find(entry.getSourceTask());
      if (source == scheduleSources.end() ||
          !onlineTasks.contains(source->second))
        continue;
      auto materialization = materializationTasks.find(source->second);
      if (materialization == materializationTasks.end()) {
        module.emitError("online GPU entry has no source-bound semantics");
        return signalPassFailure();
      }
      coveredTasks.insert(source->second);
      if (entry.getCodeOffset() < 0 || entry.getCodeLength() <= 0 ||
          entry.getCodeOffset() >
              static_cast<int64_t>(device.getCode().size()) -
                  entry.getCodeLength()) {
        module.emitError("online GPU entry has an invalid code slice");
        return signalPassFailure();
      }
      StringRef ptx(reinterpret_cast<const char *>(device.getCode().data() +
                                                   entry.getCodeOffset()),
                    entry.getCodeLength());
      FailureOr<OnlinePtxCfg> cfg = onlinePtxCfg(ptx);
      if (failed(cfg)) {
        module.emitError("online GPU entry has malformed control flow");
        return signalPassFailure();
      }
      auto scheduled = scheduleTasks.find(entry.getSourceTask());
      FailureOr<OnlineLogicalCoordinateProof> logicalCoordinates =
          scheduled == scheduleTasks.end()
              ? FailureOr<OnlineLogicalCoordinateProof>(failure())
              : onlineLogicalCoordinateMaps(entry, scheduled->second);
      const int64_t relationRank =
          cast<RankedTensorType>(representativeScoreType).getRank();
      FailureOr<SmallVector<int64_t>> relationAxes =
          onlineRelationDomainAxes(materialization->second, region,
                                   relationOperations, relationRank);
      auto entryReductionCertificate =
          entry->getAttrOfType<DictionaryAttr>(
              "online_max_reduction_certificate");
      ArrayRef<int64_t> maximumReductionDimensions =
          maximumFold.getReductionDimensions();
      std::optional<int64_t> maximumReductionDimension =
          maximumReductionDimensions.size() == 1
              ? std::optional<int64_t>(maximumReductionDimensions.front())
              : std::nullopt;
      if (failed(logicalCoordinates) || failed(relationAxes) ||
          !onlineRelationSourceClosure(*cfg, relationOperations,
                                       logicalCoordinates->coordinates,
                                       logicalCoordinates->launchExtents,
                                       *relationAxes,
                                       module.getContext(),
                                       entryReductionCertificate,
                                       maximumReductionDimension)) {
        module.emitError("online GPU Relation index dataflow is not source-bound");
        return signalPassFailure();
      }
      bool entryHasSemanticOutput = false;
      bool entryClosed = true;
      for (auto [outputIndex, output] :
           llvm::enumerate(entry.getOutputBuffers())) {
        auto buffer = planBuffers.find(output);
        if (buffer == planBuffers.end())
          continue;
        const int64_t parameter =
            entry.getInputBuffers().size() + outputIndex;
        if (buffer->second.getStorage() == MaterializationStorage::Temporary) {
          entryHasSemanticOutput = true;
          ArrayRef<int64_t> reductionDimensions =
              maximumFold.getReductionDimensions();
          auto scoreType = cast<RankedTensorType>(
              maximumFold.getInputs().front().getType());
          DictionaryAttr certificate = entryReductionCertificate;
          SmallVector<Attribute> expectedSources;
          if (auto sources = materialization->second.getCompositionSources())
            llvm::append_range(expectedSources, *sources);
          FailureOr<SmallVector<int64_t>> scoreInputParameters =
              onlineScoreInputParameters(entry, plan, region,
                                         scoreContracts.front());
          const bool valid =
              reductionDimensions.size() == 1 && certificate &&
              !expectedSources.empty() && succeeded(scoreInputParameters) &&
              onlineMaximumProtocol(
                  *cfg, parameter, *comparisonOpcode, certificate,
                  reductionDimensions.front(),
                  scoreType.getDimSize(reductionDimensions.front()),
                  expectedSources, onlineSourceKey(scoreContractSource),
                  scoreValue,
                  scoreContracts.front(), *scoreInputParameters,
                  relationOperations,
                  logicalCoordinates->coordinates,
                  logicalCoordinates->launchExtents, scoreType.getShape(),
                  module.getContext());
          compactState |= valid;
          entryClosed &= valid;
        }
        if (buffer->second.getLiveOut()) {
          entryHasSemanticOutput = true;
          const bool valid =
              scheduled != scheduleTasks.end() &&
              onlineStateProtocol(*cfg, parameter, *comparisonOpcode,
                                  scheduled->second.getWorkgroupThreads());
          publicState |= valid;
          entryClosed &= valid;
        }
      }
      if (!entryHasSemanticOutput || !entryClosed) {
        module.emitError(
            "online GPU CFG does not prove predicate/update/merge/finalize");
        return signalPassFailure();
      }
    }
    if (coveredTasks.size() != onlineTasks.size() || compactState ||
        !publicState) {
      module.emitError(
          "online GPU CFG does not prove predicate/update/merge/finalize");
      return signalPassFailure();
    }

    llvm::SmallDenseMap<int64_t, Type> operandTypes;
    llvm::SmallDenseMap<int64_t, Type> resultTypes;
    SmallVector<int64_t> expectedOperands;
    SmallVector<int64_t> expectedResults;
    for (MaterializationBufferOp buffer :
         plan.getBody().front().getOps<MaterializationBufferOp>()) {
      if (buffer.getStorage() != MaterializationStorage::External)
        continue;
      if (buffer.getLiveIn() && !buffer.getLiveOut())
        expectedOperands.push_back(buffer.getOrdinal());
      if (buffer.getLiveOut())
        expectedResults.push_back(buffer.getOrdinal());
    }
    for (InvocationSlotOp slot :
         abi.getBody().front().getOps<InvocationSlotOp>()) {
      if (slot.getStorage() == MaterializationStorage::Temporary &&
          slot.getTensorType() == representativeScoreType) {
        module.emitError("online GPU ABI exposes representative-score storage");
        return signalPassFailure();
      }
      if (!slot.getBindingIndex())
        continue;
      const int64_t index = *slot.getBindingIndex();
      if (slot.getBinding() == ExecutableBindingKind::Operand) {
        if (index < 0 || index >= static_cast<int64_t>(expectedOperands.size()) ||
            slot.getSourceBuffer() != expectedOperands[index]) {
          module.emitError("online GPU operand ABI is not source-bound");
          return signalPassFailure();
        }
        operandTypes.try_emplace(*slot.getBindingIndex(), slot.getTensorType());
      } else if (slot.getBinding() == ExecutableBindingKind::Result) {
        if (index < 0 || index >= static_cast<int64_t>(expectedResults.size()) ||
            slot.getSourceBuffer() != expectedResults[index]) {
          module.emitError("online GPU result ABI is not source-bound");
          return signalPassFailure();
        }
        resultTypes.try_emplace(*slot.getBindingIndex(), slot.getTensorType());
      }
    }
    Block &body = region.getBody().front();
    auto yield = cast<YieldOp>(body.getTerminator());
    if (operandTypes.size() != body.getNumArguments() ||
        resultTypes.size() != yield.getValues().size() ||
        llvm::any_of(llvm::enumerate(body.getArguments()), [&](auto item) {
          return operandTypes.lookup(item.index()) != item.value().getType();
        }) ||
        llvm::any_of(llvm::enumerate(yield.getValues()), [&](auto item) {
          return resultTypes.lookup(item.index()) != item.value().getType();
        })) {
      module.emitError("online GPU public ABI is not an exact source projection");
      return signalPassFailure();
    }
    llvm::outs() << "online_gpu = semantic_state_closed; relation = applied; "
                    "public_abi = exact; representative_score_materialized = "
                    "false; resources = observational\n";
  }
};

class CorruptOnlineGpuSemanticsPass
    : public MutationPass<CorruptOnlineGpuSemanticsPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CorruptOnlineGpuSemanticsPass)
  CorruptOnlineGpuSemanticsPass() = default;
  CorruptOnlineGpuSemanticsPass(const CorruptOnlineGpuSemanticsPass &other)
      : MutationPass<CorruptOnlineGpuSemanticsPass>(other) {}
  Option<std::string> kind{*this, "kind", llvm::cl::desc("Mutation kind"),
                           llvm::cl::init("")};
  StringRef getArgument() const final {
    return "shuttle-test-corrupt-online-gpu-semantics";
  }
  void runOnOperation() override {
    DeviceModuleOp device;
    for (DeviceModuleOp candidate : getOperation().getOps<DeviceModuleOp>()) {
      device = candidate;
      break;
    }
    if (!device) {
      getOperation().emitError("test fixture has no online GPU device module");
      return signalPassFailure();
    }

    const bool hasMaximumCertificate = llvm::any_of(
        device.getBody().front().getOps<DeviceEntryOp>(),
        [](DeviceEntryOp entry) {
          return static_cast<bool>(entry->getAttr(
              "online_max_reduction_certificate"));
        });
    if (hasMaximumCertificate &&
        (kind == "max-leaf-drop" || kind == "max-leaf-duplicate" ||
         kind == "max-leaf-coordinate" || kind == "max-merge-lost" ||
         kind == "max-root-partial")) {
      bool changed = false;
      for (DeviceEntryOp entry :
           device.getBody().front().getOps<DeviceEntryOp>()) {
        auto certificate = entry->getAttrOfType<DictionaryAttr>(
            "online_max_reduction_certificate");
        auto nodes = certificate
                         ? certificate.getAs<ArrayAttr>("nodes")
                         : ArrayAttr();
        auto root = certificate
                        ? certificate.getAs<IntegerAttr>("root")
                        : IntegerAttr();
        if (!nodes || !root)
          continue;
        SmallVector<Attribute> mutated(nodes.begin(), nodes.end());
        auto replaceNode = [&](size_t index, DictionaryAttr node,
                               StringRef name, Attribute value) {
          SmallVector<NamedAttribute> attributes(node.begin(), node.end());
          llvm::erase_if(attributes, [&](NamedAttribute attribute) {
            return attribute.getName() == name;
          });
          attributes.emplace_back(StringAttr::get(entry.getContext(), name),
                                  value);
          mutated[index] = DictionaryAttr::get(entry.getContext(), attributes);
        };
        auto replacePartition = [&](size_t index, DictionaryAttr node,
                                    DictionaryAttr partition) {
          auto realization = node.getAs<DictionaryAttr>("realization");
          SmallVector<NamedAttribute> realizationAttributes(
              realization.begin(), realization.end());
          llvm::erase_if(realizationAttributes, [](NamedAttribute attribute) {
            return attribute.getName() == "partition";
          });
          realizationAttributes.emplace_back(
              StringAttr::get(entry.getContext(), "partition"), partition);
          replaceNode(index, node, "realization",
                      DictionaryAttr::get(entry.getContext(),
                                          realizationAttributes));
        };
        if (kind == "max-leaf-drop") {
          auto leaf = llvm::find_if(mutated, [](Attribute attribute) {
            auto node = cast<DictionaryAttr>(attribute);
            auto children = node.getAs<DenseI64ArrayAttr>("children");
            return children && children.empty();
          });
          if (leaf != mutated.end()) {
            mutated.erase(leaf);
            changed = true;
          }
        } else if (kind == "max-leaf-duplicate") {
          DictionaryAttr firstPartition;
          for (auto [index, attribute] : llvm::enumerate(mutated)) {
            auto node = cast<DictionaryAttr>(attribute);
            auto children = node.getAs<DenseI64ArrayAttr>("children");
            auto realization = node.getAs<DictionaryAttr>("realization");
            auto partition = realization
                                 ? realization.getAs<DictionaryAttr>("partition")
                                 : DictionaryAttr();
            if (!children || !children.empty() || !partition)
              continue;
            if (!firstPartition) {
              firstPartition = partition;
              continue;
            }
            replacePartition(index, node, firstPartition);
            changed = true;
            break;
          }
        } else if (kind == "max-leaf-coordinate") {
          auto extent = certificate.getAs<IntegerAttr>("reduction_extent");
          for (auto [index, attribute] : llvm::enumerate(mutated)) {
            auto node = cast<DictionaryAttr>(attribute);
            auto children = node.getAs<DenseI64ArrayAttr>("children");
            auto realization = node.getAs<DictionaryAttr>("realization");
            auto partition = realization
                                 ? realization.getAs<DictionaryAttr>("partition")
                                 : DictionaryAttr();
            if (!children || !children.empty() || !extent || !partition)
              continue;
            SmallVector<NamedAttribute> partitionAttributes(partition.begin(),
                                                            partition.end());
            llvm::erase_if(partitionAttributes, [](NamedAttribute attribute) {
              return attribute.getName() == "offset";
            });
            partitionAttributes.emplace_back(
                StringAttr::get(entry.getContext(), "offset"), extent);
            replacePartition(
                index, node,
                DictionaryAttr::get(entry.getContext(), partitionAttributes));
            changed = true;
            break;
          }
        } else if (kind == "max-merge-lost") {
          for (auto [index, attribute] : llvm::enumerate(mutated)) {
            auto node = cast<DictionaryAttr>(attribute);
            auto children = node.getAs<DenseI64ArrayAttr>("children");
            if (!children || children.size() != 2)
              continue;
            replaceNode(index, node, "children",
                        DenseI64ArrayAttr::get(entry.getContext(),
                                               {children[0], children[0]}));
            changed = true;
            break;
          }
        } else {
          for (auto [index, attribute] : llvm::enumerate(mutated)) {
            auto node = cast<DictionaryAttr>(attribute);
            auto id = node.getAs<IntegerAttr>("id");
            auto children = node.getAs<DenseI64ArrayAttr>("children");
            if (!id || id.getInt() != root.getInt() || !children ||
                children.empty())
              continue;
            SmallVector<int64_t> partial(children.asArrayRef().drop_back());
            replaceNode(index, node, "children",
                        DenseI64ArrayAttr::get(entry.getContext(), partial));
            changed = true;
            break;
          }
        }
        if (!changed)
          continue;
        SmallVector<NamedAttribute> attributes(certificate.begin(),
                                               certificate.end());
        llvm::erase_if(attributes, [](NamedAttribute attribute) {
          return attribute.getName() == "nodes";
        });
        attributes.emplace_back(StringAttr::get(entry.getContext(), "nodes"),
                                ArrayAttr::get(entry.getContext(), mutated));
        entry->setAttr("online_max_reduction_certificate",
                       DictionaryAttr::get(entry.getContext(), attributes));
        break;
      }
      if (!changed) {
        getOperation().emitError(
            "online GPU entry has no max-reduction certificate target");
        return signalPassFailure();
      }
      refreshExecutableBundle(getOperation(), /*refreshCodeDigest=*/false);
      return;
    }

    if (kind == "relation-axis-map") {
      bool changed = false;
      for (DeviceEntryOp entry :
           device.getBody().front().getOps<DeviceEntryOp>()) {
        auto maps = entry->getAttrOfType<ArrayAttr>("logical_coordinate_maps");
        if (!maps || maps.size() < 2)
          continue;
        SmallVector<Attribute> mutated(maps.begin(), maps.end());
        std::swap(mutated[0], mutated[1]);
        entry->setAttr("logical_coordinate_maps",
                       ArrayAttr::get(entry.getContext(), mutated));
        changed = true;
      }
      if (!changed) {
        getOperation().emitError(
            "online GPU entry has no logical coordinate maps");
        return signalPassFailure();
      }
      refreshExecutableBundle(getOperation(), /*refreshCodeDigest=*/false);
      return;
    }

    StringRef phaseRole = "public_store";
    StringRef role;
    StringRef needle;
    StringRef replacement;
    std::string sourceMarker;
    if (kind == "maximum") {
      phaseRole = "public_store";
      role = "partial_merge_max";
      needle = "max";
      replacement = "add";
    } else if (kind == "max-merge-lost") {
      role = "partial_m_update";
      needle = "mov";
      replacement = "max";
    } else if (kind == "max-init") {
      role = "partial_m_init";
      needle = "0fFF800000";
      replacement = "0f00000000";
    } else if (kind == "loop-polarity") {
      role = "partial_index_guard";
      needle = ".ge.";
      replacement = ".lt.";
    } else if (kind == "loop-bound") {
      role = "partial_index_guard";
    } else if (kind == "loop-step") {
      role = "partial_index_step";
    } else if (kind == "loop-backedge") {
      role = "partial_backedge";
      needle = "loop";
      replacement = "done";
    } else if (kind == "leaf-source") {
      RegionOp region;
      getOperation().walk([&](RegionOp candidate) {
        if (!region)
          region = candidate;
      });
      if (region) {
        for (Operation &operation :
             region.getBody().front().without_terminator()) {
          auto contract = dyn_cast<ContractOp>(operation);
          SourceRefAttr source =
              contract ? contract->getAttrOfType<SourceRefAttr>("source")
                       : SourceRefAttr();
          if (source) {
            sourceMarker = (Twine("// shuttle.online.source_ref ") +
                            onlineSourceKey(source))
                               .str();
            break;
          }
        }
      }
      needle = "fma";
      replacement = "mov";
    } else if (kind == "root-value" || kind == "max-root-partial") {
      role = "public_store";
    } else if (kind == "output-coordinate") {
      role = "public_output_linear_index";
    } else if (kind == "max-leaf-drop") {
      role = "leaf_m_store";
    } else if (kind == "max-leaf-duplicate") {
      role = "partial_merge_max";
    } else if (kind == "max-leaf-coordinate") {
      role = "partial_index_wide";
    } else if (kind == "leaf-max") {
      role = "leaf_max_update";
      needle = "max";
      replacement = "add";
    } else if (kind == "leaf-exp") {
      role = "leaf_score_weight";
      needle = "ex";
      replacement = "lg";
    } else if (kind == "leaf-update") {
      role = "leaf_l_update";
      needle = "fma";
      replacement = "mad";
    } else if (kind == "zero-init") {
      role = "zero_init";
      needle = "0f00000000";
      replacement = "0f3F800000";
    } else if (kind == "compare") {
      role = "relation_compare";
      needle = "setp";
      replacement = "setx";
    } else if (kind == "select") {
      role = "mask_select";
      needle = "selp";
      replacement = "maxx";
    } else if (kind == "exp") {
      role = "score_weight";
      needle = "ex";
      replacement = "lg";
    } else if (kind == "update") {
      role = "l_update";
      needle = "add";
      replacement = "sub";
    } else if (kind == "shared-merge") {
      role = "partial_l_load";
      needle = "shared";
      replacement = "global";
    } else if (kind == "shared-value") {
      role = "leaf_l_store";
    } else if (kind == "output-store") {
      role = "public_store";
      needle = "global";
      replacement = "shared";
    } else if (kind == "relation-offset" || kind == "relation-iota") {
      RelationSemantics target = kind == "relation-offset"
                                     ? RelationSemantics::Add
                                     : RelationSemantics::Iota;
      getOperation().walk([&](RelationOp relation) {
        if (!sourceMarker.empty() || relation.getSemantics() != target)
          return;
        SourceRefAttr source =
            relation->getAttrOfType<SourceRefAttr>("source");
        if (source)
          sourceMarker = (Twine("// shuttle.online.source_ref ") +
                          onlineSourceKey(source))
                             .str();
      });
      needle = kind == "relation-offset" ? "add" : "";
      replacement = kind == "relation-offset" ? "sub" : "";
    } else if (kind == "finalize") {
      role = "finalize";
      needle = "div";
      replacement = "mul";
    } else {
      getOperation().emitError("unknown online GPU semantic mutation");
      return signalPassFailure();
    }

    SmallVector<int8_t> code(device.getCode());
    StringRef bytes(reinterpret_cast<const char *>(code.data()), code.size());
    SmallVector<std::pair<size_t, std::string>> mutations;
    const std::string phaseMarker =
        (Twine("// shuttle.online.role ") + phaseRole).str();
    const std::string roleMarker = sourceMarker.empty()
                                       ? (Twine("// shuttle.online.role ") + role)
                                             .str()
                                       : sourceMarker;
    for (DeviceEntryOp entry :
         device.getBody().front().getOps<DeviceEntryOp>()) {
      StringRef slice = bytes.slice(entry.getCodeOffset(),
                                    entry.getCodeOffset() + entry.getCodeLength());
      if (!slice.contains(phaseMarker))
        continue;
      size_t marker = slice.find(roleMarker);
      if (marker == StringRef::npos ||
          slice.find(roleMarker, marker + roleMarker.size()) != StringRef::npos) {
        getOperation().emitError(
            "online GPU semantic role is not unique in its protocol phase");
        return signalPassFailure();
      }
      size_t instructionStart = slice.find('\n', marker);
      size_t instructionEnd =
          instructionStart == StringRef::npos
              ? StringRef::npos
              : slice.find('\n', instructionStart + 1);
      if (instructionStart == StringRef::npos)
        continue;
      if (instructionEnd == StringRef::npos)
        instructionEnd = slice.size();
      StringRef instruction =
          slice.slice(instructionStart + 1, instructionEnd);
      std::string localNeedleStorage = needle.str();
      std::string localReplacementStorage = replacement.str();
      StringRef localNeedleText = localNeedleStorage;
      StringRef localReplacement = localReplacementStorage;
      size_t localNeedle = instruction.find(localNeedleText);
      if (kind == "shared-value" || kind == "relation-iota" ||
          kind == "loop-bound" || kind == "loop-step" ||
          kind == "root-value" || kind == "output-coordinate" ||
          kind == "max-leaf-drop" || kind == "max-leaf-duplicate" ||
          kind == "max-leaf-coordinate" || kind == "max-root-partial") {
        std::vector<std::string> tokens = onlinePtxTokens(instruction);
        if (tokens.empty())
          continue;
        size_t tokenIndex =
            (kind == "root-value" || kind == "max-root-partial" ||
             kind == "max-leaf-drop")
                ? 1
            : kind == "max-leaf-duplicate"
                ? 2
            : (kind == "loop-bound" || kind == "loop-step")
                ? tokens.size() - 1
                : 1;
        if (tokens.size() <= tokenIndex || tokens[tokenIndex].empty() ||
            (!llvm::isDigit(tokens[tokenIndex].back()) &&
             !StringRef(tokens[tokenIndex]).ends_with(".x")))
          continue;
        localNeedleStorage = tokens[tokenIndex];
        std::string changed =
            kind == "max-leaf-duplicate" && tokens.size() > 2
                ? tokens[1]
                : tokens[tokenIndex];
        if (kind != "max-leaf-duplicate")
          changed.back() = llvm::isDigit(changed.back())
                               ? (changed.back() == '0' ? '1' : '0')
                               : 'y';
        localReplacementStorage = changed;
        localNeedleText = localNeedleStorage;
        localReplacement = localReplacementStorage;
        localNeedle = instruction.find(localNeedleText);
      }
      if (localNeedle == StringRef::npos && kind == "exp") {
        localNeedleText = "exp";
        localReplacement = "log";
        localNeedle = instruction.find(localNeedleText);
      }
      if (localNeedle == StringRef::npos && kind == "update") {
        localNeedleText = "fma";
        localReplacement = "mad";
        localNeedle = instruction.find(localNeedleText);
      }
      if (localNeedle == StringRef::npos && kind == "finalize") {
        localNeedleText = "rcp";
        localReplacement = "cos";
        localNeedle = instruction.find(localNeedleText);
      }
      if (localNeedle == StringRef::npos ||
          localNeedleText.size() != localReplacement.size())
        continue;
      const size_t absolute = entry.getCodeOffset() + instructionStart + 1 +
                              localNeedle;
      mutations.emplace_back(absolute, localReplacement.str());
    }
    if (mutations.empty()) {
      getOperation().emitError(
          "test fixture lacks the requested reaching semantic role");
      return signalPassFailure();
    }
    for (const auto &[offset, changed] : mutations)
      std::copy(changed.begin(), changed.end(), code.begin() + offset);
    device.setCodeAttr(DenseI8ArrayAttr::get(device.getContext(), code));
    refreshExecutableBundle(getOperation(), true);
  }
};

class VerifyOnlineFallbackPass
    : public MutationPass<VerifyOnlineFallbackPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyOnlineFallbackPass)
  StringRef getArgument() const final {
    return "shuttle-test-verify-online-fallback";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool sawPlan = false;
    for (MaterializationPlanOp plan : module.getOps<MaterializationPlanOp>()) {
      sawPlan = true;
      for (MaterializationTaskOp task :
           plan.getBody().front().getOps<MaterializationTaskOp>()) {
        if (task.getCompositionKind() &&
            *task.getCompositionKind() ==
                "online_normalized_exponential_weighted_contract") {
          module.emitError("near miss was incorrectly classified as online");
          return signalPassFailure();
        }
      }
    }
    if (!sawPlan) {
      module.emitError("near miss has no lossless materialized fallback");
      return signalPassFailure();
    }
    llvm::outs() << "online_state = fallback\n";
  }
};

class DropOnlineRecomputationPass
    : public MutationPass<DropOnlineRecomputationPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DropOnlineRecomputationPass)
  StringRef getArgument() const final {
    return "shuttle-test-drop-online-recomputation";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    for (MaterializationTaskOp task :
         plan.getBody().front().getOps<MaterializationTaskOp>()) {
      if (!task->hasAttr("composition_recomputed_sources"))
        continue;
      task->removeAttr("composition_recomputed_sources");
      refreshMaterializationFingerprint(plan);
      return;
    }
    getOperation().emitError("test fixture has no online recomputation");
    signalPassFailure();
  }
};

class SwapOnlineRecomputationPass
    : public MutationPass<SwapOnlineRecomputationPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwapOnlineRecomputationPass)
  StringRef getArgument() const final {
    return "shuttle-test-swap-online-recomputation";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    for (MaterializationTaskOp task :
         plan.getBody().front().getOps<MaterializationTaskOp>()) {
      auto sources =
          task->getAttrOfType<ArrayAttr>("composition_recomputed_sources");
      if (!sources || sources.size() < 2)
        continue;
      SmallVector<Attribute> swapped(sources.begin(), sources.end());
      std::swap(swapped[0], swapped[1]);
      task->setAttr("composition_recomputed_sources",
                    ArrayAttr::get(task.getContext(), swapped));
      refreshMaterializationFingerprint(plan);
      return;
    }
    getOperation().emitError("test fixture has no swappable recomputation");
    signalPassFailure();
  }
};

class MaterializationDeleteTaskPass
    : public MutationPass<MaterializationDeleteTaskPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializationDeleteTaskPass)
  StringRef getArgument() const final {
    return "shuttle-test-delete-materialization-task";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    MaterializationTaskOp task;
    if (plan) {
      for (MaterializationTaskOp candidate :
           plan.getBody().front().getOps<MaterializationTaskOp>()) {
        task = candidate;
        break;
      }
    }
    if (!task) {
      getOperation().emitError("test fixture has no materialization task");
      return signalPassFailure();
    }
    task.erase();
    refreshMaterializationFingerprint(plan);
  }
};

class MaterializationReorderTasksPass
    : public MutationPass<MaterializationReorderTasksPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializationReorderTasksPass)
  StringRef getArgument() const final {
    return "shuttle-test-reorder-materialization-tasks";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    SmallVector<MaterializationTaskOp> tasks;
    if (plan) {
      llvm::append_range(
          tasks, plan.getBody().front().getOps<MaterializationTaskOp>());
    }
    if (tasks.size() < 2) {
      getOperation().emitError("test fixture has fewer than two tasks");
      return signalPassFailure();
    }
    tasks[1]->moveBefore(tasks[0]);
    refreshMaterializationFingerprint(plan);
  }
};

class MaterializationSelfConsistentReorderPass
    : public MutationPass<MaterializationSelfConsistentReorderPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializationSelfConsistentReorderPass)
  StringRef getArgument() const final {
    return "shuttle-test-reorder-materialization-tasks-consistently";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    SmallVector<MaterializationTaskOp> tasks;
    SmallVector<MaterializationBufferOp> buffers;
    if (plan) {
      llvm::append_range(
          tasks, plan.getBody().front().getOps<MaterializationTaskOp>());
      llvm::append_range(
          buffers, plan.getBody().front().getOps<MaterializationBufferOp>());
    }
    size_t left = 0;
    while (left < tasks.size() && !tasks[left].getDependencies().empty()) {
      ++left;
    }
    size_t right = left + 1;
    while (right < tasks.size() && !tasks[right].getDependencies().empty()) {
      ++right;
    }
    if (right >= tasks.size()) {
      getOperation().emitError("test fixture has no independent task pair");
      return signalPassFailure();
    }
    SmallVector<int64_t> oldToNew(tasks.size());
    for (auto [ordinal, ignored] : llvm::enumerate(tasks)) {
      (void)ignored;
      oldToNew[ordinal] = ordinal;
    }
    oldToNew[right] = left;
    for (size_t ordinal = left; ordinal < right; ++ordinal) {
      oldToNew[ordinal] = ordinal + 1;
    }
    tasks[right]->moveBefore(tasks[left]);
    tasks.clear();
    llvm::append_range(tasks,
                       plan.getBody().front().getOps<MaterializationTaskOp>());
    for (auto [ordinal, task] : llvm::enumerate(tasks)) {
      task.setOrdinal(ordinal);
      SmallVector<int64_t> dependencies;
      for (int64_t dependency : task.getDependencies()) {
        dependencies.push_back(oldToNew[dependency]);
      }
      llvm::sort(dependencies);
      task->setAttr("dependencies",
                    DenseI64ArrayAttr::get(task.getContext(), dependencies));
    }
    for (MaterializationBufferOp buffer : buffers) {
      if (auto producer = buffer->getAttrOfType<IntegerAttr>("producer")) {
        buffer->setAttr(
            "producer",
            IntegerAttr::get(producer.getType(), oldToNew[producer.getInt()]));
        buffer.setLifetimeStart(oldToNew[producer.getInt()]);
      }
      SmallVector<int64_t> consumers;
      for (int64_t consumer : buffer.getConsumers()) {
        consumers.push_back(oldToNew[consumer]);
      }
      llvm::sort(consumers);
      buffer->setAttr("consumers",
                      DenseI64ArrayAttr::get(buffer.getContext(), consumers));
      if (!buffer.getLiveOut() && !consumers.empty()) {
        buffer.setLifetimeEnd(consumers.back());
      }
    }
    refreshMaterializationFingerprint(plan);
  }
};

class MaterializationReplaySourcePass
    : public MutationPass<MaterializationReplaySourcePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializationReplaySourcePass)
  StringRef getArgument() const final {
    return "shuttle-test-replay-materialization-source";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    SmallVector<MaterializationTaskOp> tasks;
    if (plan) {
      llvm::append_range(
          tasks, plan.getBody().front().getOps<MaterializationTaskOp>());
    }
    if (tasks.size() < 2) {
      getOperation().emitError("test fixture has fewer than two tasks");
      return signalPassFailure();
    }
    tasks[1]->setAttr("source", tasks[0].getSource());
    refreshMaterializationFingerprint(plan);
  }
};

class MaterializationDuplicateAlgebraSourcePass
    : public MutationPass<MaterializationDuplicateAlgebraSourcePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializationDuplicateAlgebraSourcePass)
  StringRef getArgument() const final {
    return "shuttle-test-duplicate-materialization-algebra-source";
  }
  void runOnOperation() override {
    SmallVector<MapOp> maps;
    getOperation().walk([&](MapOp map) { maps.push_back(map); });
    if (maps.size() < 2) {
      getOperation().emitError("test fixture has fewer than two Maps");
      return signalPassFailure();
    }
    maps[1]->setAttr("source", maps[0].getSource());
  }
};

class MaterializationSwapSameTypeEdgesPass
    : public MutationPass<MaterializationSwapSameTypeEdgesPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializationSwapSameTypeEdgesPass)
  StringRef getArgument() const final {
    return "shuttle-test-swap-materialization-edges";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    SmallVector<MaterializationBufferOp> buffers;
    if (plan) {
      llvm::append_range(
          buffers, plan.getBody().front().getOps<MaterializationBufferOp>());
    }
    if (plan) {
      for (MaterializationTaskOp task :
           plan.getBody().front().getOps<MaterializationTaskOp>()) {
        SmallVector<int64_t> inputs(task.getInputBuffers().begin(),
                                    task.getInputBuffers().end());
        for (size_t left = 0; left < inputs.size(); ++left) {
          for (size_t right = left + 1; right < inputs.size(); ++right) {
            if (inputs[left] == inputs[right] ||
                buffers[inputs[left]].getTensorType() !=
                    buffers[inputs[right]].getTensorType()) {
              continue;
            }
            std::swap(inputs[left], inputs[right]);
            task->setAttr("input_buffers",
                          DenseI64ArrayAttr::get(task.getContext(), inputs));
            refreshMaterializationEdges(plan);
            return;
          }
        }
      }
    }
    getOperation().emitError("test fixture has no same-type task edge pair");
    signalPassFailure();
  }
};

template <bool EmptyToScalar>
class MaterializationDomainPass
    : public MutationPass<MaterializationDomainPass<EmptyToScalar>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializationDomainPass<EmptyToScalar>)
  StringRef getArgument() const final {
    return EmptyToScalar ? "shuttle-test-add-scalar-materialization-domain"
                         : "shuttle-test-empty-tensor-materialization-domain";
  }
  void runOnOperation() override {
    ModuleOp module = this->getOperation();
    MaterializationPlanOp plan = materializationPlan(module);
    if (plan) {
      for (MaterializationTaskOp task :
           plan.getBody().front().getOps<MaterializationTaskOp>()) {
        if (task.getKind() != MaterializationTaskKind::Map ||
            task.getDomainShape().empty() != EmptyToScalar) {
          continue;
        }
        task->setAttr("domain_shape", DenseI64ArrayAttr::get(
                                          task.getContext(),
                                          EmptyToScalar ? ArrayRef<int64_t>{1}
                                                        : ArrayRef<int64_t>{}));
        refreshMaterializationFingerprint(plan);
        return;
      }
    }
    module.emitError("test fixture has no matching Map domain");
    this->signalPassFailure();
  }
};

class MaterializationUnknownAttributePass
    : public MutationPass<MaterializationUnknownAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializationUnknownAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-materialization-attribute";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no materialization plan");
      return signalPassFailure();
    }
    plan->setAttr("shuttle.test_semantic", UnitAttr::get(plan.getContext()));
  }
};

class MaterializationContractShapePass
    : public MutationPass<MaterializationContractShapePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializationContractShapePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-contract-materialization-shape";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    if (plan) {
      for (MaterializationTaskOp task :
           plan.getBody().front().getOps<MaterializationTaskOp>()) {
        if (task.getKind() != MaterializationTaskKind::Contract) {
          continue;
        }
        SmallVector<int64_t> domain(task.getDomainShape().begin(),
                                    task.getDomainShape().end());
        for (auto [axis, extent] : llvm::enumerate(domain)) {
          if (llvm::is_contained(task.getReductionDimensions(),
                                 static_cast<int64_t>(axis))) {
            continue;
          }
          domain[axis] = extent + 1;
          task->setAttr("domain_shape",
                        DenseI64ArrayAttr::get(task.getContext(), domain));
          refreshMaterializationFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError(
        "test fixture has no Contract materialization task");
    signalPassFailure();
  }
};

class MaterializationFoldMutationPass
    : public MutationPass<MaterializationFoldMutationPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializationFoldMutationPass)
  MaterializationFoldMutationPass() = default;
  MaterializationFoldMutationPass(const MaterializationFoldMutationPass &other)
      : MutationPass<MaterializationFoldMutationPass>(other) {}
  Option<std::string> kind{*this, "kind", llvm::cl::desc("Mutation kind"),
                           llvm::cl::init("")};
  StringRef getArgument() const final {
    return "shuttle-test-mutate-fold-materialization";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no materialization plan");
      return signalPassFailure();
    }
    for (MaterializationTaskOp task :
         plan.getBody().front().getOps<MaterializationTaskOp>()) {
      if (task.getKind() != MaterializationTaskKind::Fold) {
        continue;
      }
      SmallVector<int64_t> domain(task.getDomainShape());
      StringRef mutation = kind;
      bool prefix = mutation.consume_front("prefix-");
      if (prefix) {
        ArrayRef<int64_t> dimensions = task.getReductionDimensions();
        if (domain.size() < 3 || dimensions.size() + 1 != domain.size()) {
          continue;
        }
        bool fullPrefix = llvm::all_of(
            llvm::enumerate(dimensions), [](auto indexedDimension) {
              return indexedDimension.value() ==
                     static_cast<int64_t>(indexedDimension.index());
            });
        if (!fullPrefix) {
          continue;
        }
      }
      if (mutation == "parallel") {
        ++domain[prefix ? domain.size() - 1 : 0];
        task->setAttr("domain_shape",
                      DenseI64ArrayAttr::get(task.getContext(), domain));
      } else if (mutation == "reduction") {
        ++domain[prefix ? 0 : domain.size() - 1];
        task->setAttr("domain_shape",
                      DenseI64ArrayAttr::get(task.getContext(), domain));
      } else if (mutation == "axis") {
        task->setAttr("reduction_dimensions", DenseI64ArrayAttr::get(
                                                  task.getContext(), {0}));
      } else if (mutation == "omit") {
        SmallVector<int64_t> dimensions(task.getReductionDimensions());
        dimensions.pop_back();
        task->setAttr("reduction_dimensions",
                      DenseI64ArrayAttr::get(task.getContext(), dimensions));
      } else if (mutation == "swap") {
        SmallVector<int64_t> dimensions(task.getReductionDimensions());
        if (dimensions.size() < 2) {
          continue;
        }
        std::swap(dimensions[0], dimensions[1]);
        task->setAttr("reduction_dimensions",
                      DenseI64ArrayAttr::get(task.getContext(), dimensions));
      } else if (mutation == "dependency") {
        if (task.getDependencies().empty()) {
          continue;
        }
        task->setAttr("dependencies",
                      DenseI64ArrayAttr::get(task.getContext(), {}));
      } else {
        getOperation().emitError("unknown Fold materialization mutation kind");
        return signalPassFailure();
      }
      refreshMaterializationFingerprint(plan);
      return;
    }
    getOperation().emitError("test fixture has no Fold materialization task");
    signalPassFailure();
  }
};

class MaterializationMapMutationPass
    : public MutationPass<MaterializationMapMutationPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializationMapMutationPass)
  MaterializationMapMutationPass() = default;
  MaterializationMapMutationPass(const MaterializationMapMutationPass &other)
      : MutationPass<MaterializationMapMutationPass>(other) {}
  Option<std::string> kind{*this, "kind", llvm::cl::desc("Mutation kind"),
                           llvm::cl::init("")};
  StringRef getArgument() const final {
    return "shuttle-test-mutate-map-materialization";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no materialization plan");
      return signalPassFailure();
    }
    SmallVector<MaterializationBufferOp> buffers(
        plan.getBody().front().getOps<MaterializationBufferOp>());
    StringRef mutation = kind;
    bool projected = mutation.consume_front("projected-");
    for (MaterializationTaskOp task :
         plan.getBody().front().getOps<MaterializationTaskOp>()) {
      if (!(projected ? isProjectedParameterMapTask(task, buffers)
                      : isExpandingSingletonMapTask(task, buffers))) {
        continue;
      }
      if (mutation == "domain") {
        SmallVector<int64_t> domain(task.getDomainShape());
        ++domain.back();
        task->setAttr("domain_shape",
                      DenseI64ArrayAttr::get(task.getContext(), domain));
      } else if (mutation == "dependency") {
        if (task.getDependencies().empty()) {
          continue;
        }
        task->setAttr("dependencies",
                      DenseI64ArrayAttr::get(task.getContext(), {}));
      } else if (mutation == "output") {
        SmallVector<int64_t> outputs(task.getOutputBuffers());
        ++outputs.front();
        task->setAttr("output_buffers",
                      DenseI64ArrayAttr::get(task.getContext(), outputs));
      } else {
        getOperation().emitError("unknown Map materialization mutation kind");
        return signalPassFailure();
      }
      refreshMaterializationFingerprint(plan);
      return;
    }
    getOperation().emitError(
        "test fixture has no expanding-singleton rank-N Map task");
    signalPassFailure();
  }
};

class ReportScheduleFingerprintPass
    : public MutationPass<ReportScheduleFingerprintPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReportScheduleFingerprintPass)
  StringRef getArgument() const final {
    return "shuttle-test-report-simt32-schedule-fingerprint";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no schedule plan");
      return signalPassFailure();
    }
    llvm::outs() << plan.getFingerprint() << '\n';
  }
};

class ScheduleIndexingPass : public MutationPass<ScheduleIndexingPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleIndexingPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-indexing";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleBufferOp buffer :
           plan.getBody().front().getOps<ScheduleBufferOp>()) {
        auto type = cast<RankedTensorType>(buffer.getTensorType());
        if (type.getRank() == 2) {
          buffer->setAttr("iteration_order",
                          DenseI64ArrayAttr::get(buffer.getContext(), {1, 0}));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no rank-two schedule buffer");
    signalPassFailure();
  }
};

class ScheduleAxisPass : public MutationPass<ScheduleAxisPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleAxisPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-axis";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (task.getKind() == ScheduleTaskKind::RowFold) {
          task->setAttr(
              "reduction_axis",
              IntegerAttr::get(task.getReductionAxisAttr().getType(), 0));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no row Fold schedule");
    signalPassFailure();
  }
};

class ScheduleTilePass : public MutationPass<ScheduleTilePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleTilePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-tile";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (task.getKind() == ScheduleTaskKind::RowFold) {
          task->setAttr("tile_shape",
                        DenseI64ArrayAttr::get(task.getContext(), {1, 12}));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no row Fold schedule");
    signalPassFailure();
  }
};

class ScheduleResourcePass : public MutationPass<ScheduleResourcePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleResourcePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-resource";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (task.getKind() == ScheduleTaskKind::RowFold) {
          task.setScratchBytes(task.getScratchBytes() + 4);
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no row Fold schedule");
    signalPassFailure();
  }
};

class ScheduleDependencyPass : public MutationPass<ScheduleDependencyPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleDependencyPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-dependency";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (!task.getDependencies().empty()) {
          SmallVector<int64_t> dependencies(task.getDependencies().begin(),
                                            task.getDependencies().end());
          dependencies.pop_back();
          task->setAttr("dependencies", DenseI64ArrayAttr::get(
                                            task.getContext(), dependencies));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no dependent schedule task");
    signalPassFailure();
  }
};

class ScheduleDeleteTaskPass : public MutationPass<ScheduleDeleteTaskPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleDeleteTaskPass)
  StringRef getArgument() const final {
    return "shuttle-test-delete-schedule-task";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no schedule plan");
      return signalPassFailure();
    }
    auto tasks = plan.getBody().front().getOps<ScheduleTaskOp>();
    if (tasks.empty()) {
      getOperation().emitError("test fixture has no schedule task");
      return signalPassFailure();
    }
    (*tasks.begin()).erase();
    refreshScheduleFingerprint(plan);
  }
};

class ScheduleSelfConsistentReorderPass
    : public MutationPass<ScheduleSelfConsistentReorderPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ScheduleSelfConsistentReorderPass)
  StringRef getArgument() const final {
    return "shuttle-test-reorder-schedule-tasks";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    SmallVector<ScheduleTaskOp> tasks;
    if (plan) {
      llvm::append_range(tasks,
                         plan.getBody().front().getOps<ScheduleTaskOp>());
    }
    if (tasks.size() < 3 ||
        !llvm::is_contained(tasks[1].getDependencies(), 0) ||
        !tasks[2].getDependencies().empty()) {
      getOperation().emitError(
          "test fixture lacks the expected independent task pair");
      return signalPassFailure();
    }
    tasks[2]->moveBefore(tasks[1]);
    tasks[2].setOrdinal(1);
    tasks[2].setSourceTask(1);
    tasks[1].setOrdinal(2);
    tasks[1].setSourceTask(2);
    refreshScheduleFingerprint(plan);
  }
};

class ScheduleReplaySourceTaskPass
    : public MutationPass<ScheduleReplaySourceTaskPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleReplaySourceTaskPass)
  StringRef getArgument() const final {
    return "shuttle-test-replay-schedule-source-task";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    SmallVector<ScheduleTaskOp> tasks;
    if (plan) {
      llvm::append_range(tasks,
                         plan.getBody().front().getOps<ScheduleTaskOp>());
    }
    if (tasks.size() < 2) {
      getOperation().emitError(
          "test fixture has fewer than two schedule tasks");
      return signalPassFailure();
    }
    tasks[1].setSourceTask(tasks[0].getSourceTask());
    refreshScheduleFingerprint(plan);
  }
};

class ScheduleSourcePlanFingerprintPass
    : public MutationPass<ScheduleSourcePlanFingerprintPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ScheduleSourcePlanFingerprintPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-source-plan";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no schedule plan");
      return signalPassFailure();
    }
    plan.setSourcePlanFingerprint(std::string(64, '0'));
    refreshScheduleFingerprint(plan);
  }
};

class ScheduleTypePass : public MutationPass<ScheduleTypePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleTypePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-type";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleBufferOp buffer :
           plan.getBody().front().getOps<ScheduleBufferOp>()) {
        auto type = cast<RankedTensorType>(buffer.getTensorType());
        if (type.getElementType().isBF16()) {
          buffer->setAttr(
              "tensor_type",
              TypeAttr::get(RankedTensorType::get(
                  type.getShape(), Float32Type::get(buffer.getContext()))));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no bf16 schedule buffer");
    signalPassFailure();
  }
};

template <typename PlanOp>
class ClonePlanPass : public MutationPass<ClonePlanPass<PlanOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClonePlanPass<PlanOp>)
  StringRef getArgument() const final {
    return std::is_same_v<PlanOp, MaterializationPlanOp>
               ? "shuttle-test-clone-materialization-plan"
               : "shuttle-test-clone-schedule-plan";
  }
  void runOnOperation() override {
    ModuleOp module = this->getOperation();
    auto plans = module.getOps<PlanOp>();
    if (plans.empty()) {
      module.emitError("test fixture has no plan to clone");
      return this->signalPassFailure();
    }
    PlanOp plan = *plans.begin();
    Operation *clone = plan->clone();
    module.getBody()->getOperations().push_back(clone);
  }
};

class ScheduleUnknownAttributePass
    : public MutationPass<ScheduleUnknownAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleUnknownAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-schedule-attribute";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no schedule plan");
      return signalPassFailure();
    }
    plan->setAttr("shuttle.test_semantic", UnitAttr::get(plan.getContext()));
  }
};

class ScheduleTaskUnknownAttributePass
    : public MutationPass<ScheduleTaskUnknownAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleTaskUnknownAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-schedule-task-attribute";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no schedule plan");
      return signalPassFailure();
    }
    auto tasks = plan.getBody().front().getOps<ScheduleTaskOp>();
    if (tasks.empty()) {
      getOperation().emitError("test fixture has no schedule task");
      return signalPassFailure();
    }
    (*tasks.begin())
        ->setAttr("shuttle.test_semantic", UnitAttr::get(plan.getContext()));
  }
};

class ScheduleTargetProfilePass
    : public MutationPass<ScheduleTargetProfilePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleTargetProfilePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-target";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (task.getKind() == ScheduleTaskKind::RowFold) {
          task.setSubgroupSize(64);
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no row Fold schedule");
    signalPassFailure();
  }
};

class ScheduleReductionOrderPass
    : public MutationPass<ScheduleReductionOrderPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleReductionOrderPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-order";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (task.getKind() == ScheduleTaskKind::RowFold ||
            task.getKind() == ScheduleTaskKind::Contract) {
          ScheduleReductionOrder replacement =
              task.getReductionOrder() == ScheduleReductionOrder::LeafOrderFree
                  ? ScheduleReductionOrder::SerialAscendingFmaRnF32
                  : ScheduleReductionOrder::LeafOrderFree;
          task->setAttr("reduction_order", ScheduleReductionOrderAttr::get(
                                               task.getContext(), replacement));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no reducible schedule task");
    signalPassFailure();
  }
};

class ContractScheduleMutationPass
    : public MutationPass<ContractScheduleMutationPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ContractScheduleMutationPass)
  ContractScheduleMutationPass() = default;
  ContractScheduleMutationPass(const ContractScheduleMutationPass &other)
      : MutationPass<ContractScheduleMutationPass>(other) {}
  Option<std::string> kind{*this, "kind", llvm::cl::desc("Mutation kind"),
                           llvm::cl::init("")};
  StringRef getArgument() const final {
    return "shuttle-test-mutate-contract-schedule";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no schedule plan");
      return signalPassFailure();
    }
    for (ScheduleTaskOp task :
         plan.getBody().front().getOps<ScheduleTaskOp>()) {
      if (task.getKind() != ScheduleTaskKind::Contract) {
        continue;
      }
      StringRef mutation = kind;
      if (mutation == "grid") {
        SmallVector<int64_t> grid(task.getGridShape());
        ++grid.front();
        task->setAttr("grid_shape",
                      DenseI64ArrayAttr::get(task.getContext(), grid));
      } else if (mutation == "tile") {
        SmallVector<int64_t> tile(task.getTileShape());
        ++tile.front();
        task->setAttr("tile_shape",
                      DenseI64ArrayAttr::get(task.getContext(), tile));
      } else if (mutation == "serial") {
        task.setSerialTiles(task.getSerialTiles() + 1);
      } else if (mutation == "threads") {
        task.setWorkgroupThreads(task.getWorkgroupThreads() + 32);
      } else if (mutation == "subgroup") {
        task.setSubgroupSize(64);
      } else if (mutation == "axis") {
        task->setAttr(
            "reduction_axis",
            IntegerAttr::get(task.getReductionAxisAttr().getType(), 1));
      } else if (mutation == "output") {
        SmallVector<int64_t> domain(task.getDomainShape());
        for (auto [axis, extent] : llvm::enumerate(domain)) {
          if (static_cast<int64_t>(axis) == task.getReductionAxis()) {
            continue;
          }
          domain[axis] = extent + 1;
          task->setAttr("domain_shape",
                        DenseI64ArrayAttr::get(task.getContext(), domain));
          refreshScheduleFingerprint(plan);
          return;
        }
        getOperation().emitError(
            "Contract schedule has no parallel output axis");
        return signalPassFailure();
      } else {
        getOperation().emitError("unknown Contract schedule mutation kind");
        return signalPassFailure();
      }
      refreshScheduleFingerprint(plan);
      return;
    }
    getOperation().emitError("test fixture has no Contract schedule task");
    signalPassFailure();
  }
};

class FoldScheduleMutationPass
    : public MutationPass<FoldScheduleMutationPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldScheduleMutationPass)
  FoldScheduleMutationPass() = default;
  FoldScheduleMutationPass(const FoldScheduleMutationPass &other)
      : MutationPass<FoldScheduleMutationPass>(other) {}
  Option<std::string> kind{*this, "kind", llvm::cl::desc("Mutation kind"),
                           llvm::cl::init("")};
  StringRef getArgument() const final {
    return "shuttle-test-mutate-fold-schedule";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no schedule plan");
      return signalPassFailure();
    }
    for (ScheduleTaskOp task :
         plan.getBody().front().getOps<ScheduleTaskOp>()) {
      StringRef mutation = kind;
      bool prefix = mutation.consume_front("prefix-");
      if (task.getKind() != (prefix ? ScheduleTaskKind::ColumnFold
                                    : ScheduleTaskKind::RowFold)) {
        continue;
      }
      if (mutation == "grid") {
        SmallVector<int64_t> grid(task.getGridShape());
        ++grid.front();
        task->setAttr("grid_shape",
                      DenseI64ArrayAttr::get(task.getContext(), grid));
      } else if (mutation == "tile") {
        SmallVector<int64_t> tile(task.getTileShape());
        ++tile[prefix ? 0 : tile.size() - 1];
        task->setAttr("tile_shape",
                      DenseI64ArrayAttr::get(task.getContext(), tile));
      } else if (mutation == "serial") {
        task.setSerialTiles(task.getSerialTiles() + 1);
      } else if (mutation == "threads") {
        task.setWorkgroupThreads(task.getWorkgroupThreads() + 32);
      } else if (mutation == "subgroup") {
        task.setSubgroupSize(64);
      } else if (mutation == "scratch") {
        task.setScratchBytes(task.getScratchBytes() + 4);
      } else if (mutation == "axis") {
        task->setAttr(
            "reduction_axis",
            IntegerAttr::get(task.getReductionAxisAttr().getType(),
                             prefix ? 1 : *task.getReductionAxis() - 1));
      } else if (mutation == "output") {
        SmallVector<int64_t> domain(task.getDomainShape());
        ++domain[prefix ? domain.size() - 1 : 0];
        task->setAttr("domain_shape",
                      DenseI64ArrayAttr::get(task.getContext(), domain));
      } else if (mutation == "order") {
        ScheduleReductionOrder replacement =
            task.getReductionOrder() == ScheduleReductionOrder::LeafOrderFree
                ? ScheduleReductionOrder::SerialAscendingFmaRnF32
                : ScheduleReductionOrder::LeafOrderFree;
        task->setAttr("reduction_order", ScheduleReductionOrderAttr::get(
                                               task.getContext(), replacement));
      } else {
        getOperation().emitError("unknown Fold schedule mutation kind");
        return signalPassFailure();
      }
      refreshScheduleFingerprint(plan);
      return;
    }
    getOperation().emitError("test fixture has no matching Fold schedule task");
    signalPassFailure();
  }
};

class MapScheduleMutationPass : public MutationPass<MapScheduleMutationPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapScheduleMutationPass)
  MapScheduleMutationPass() = default;
  MapScheduleMutationPass(const MapScheduleMutationPass &other)
      : MutationPass<MapScheduleMutationPass>(other) {}
  Option<std::string> kind{*this, "kind", llvm::cl::desc("Mutation kind"),
                           llvm::cl::init("")};
  StringRef getArgument() const final {
    return "shuttle-test-mutate-map-schedule";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no schedule plan");
      return signalPassFailure();
    }
    MaterializationPlanOp materialization = materializationPlan(getOperation());
    if (!materialization) {
      getOperation().emitError("test fixture has no materialization plan");
      return signalPassFailure();
    }
    SmallVector<MaterializationBufferOp> buffers(
        materialization.getBody().front().getOps<MaterializationBufferOp>());
    SmallVector<MaterializationTaskOp> materializationTasks(
        materialization.getBody().front().getOps<MaterializationTaskOp>());
    StringRef mutation = kind;
    bool projected = mutation.consume_front("projected-");
    for (ScheduleTaskOp task :
         plan.getBody().front().getOps<ScheduleTaskOp>()) {
      int64_t sourceTask = task.getSourceTask();
      if (task.getKind() != ScheduleTaskKind::Elementwise || sourceTask < 0 ||
          sourceTask >= static_cast<int64_t>(materializationTasks.size()) ||
          !(projected
                ? isProjectedParameterMapTask(materializationTasks[sourceTask],
                                              buffers)
                : isExpandingSingletonMapTask(materializationTasks[sourceTask],
                                               buffers))) {
        continue;
      }
      if (mutation == "grid") {
        SmallVector<int64_t> grid(task.getGridShape());
        ++grid.front();
        task->setAttr("grid_shape",
                      DenseI64ArrayAttr::get(task.getContext(), grid));
      } else if (mutation == "tile") {
        SmallVector<int64_t> tile(task.getTileShape());
        --tile.front();
        task->setAttr("tile_shape",
                      DenseI64ArrayAttr::get(task.getContext(), tile));
      } else if (mutation == "threads") {
        task.setWorkgroupThreads(task.getWorkgroupThreads() - 32);
      } else if (mutation == "output") {
        SmallVector<int64_t> domain(task.getDomainShape());
        ++domain.front();
        task->setAttr("domain_shape",
                      DenseI64ArrayAttr::get(task.getContext(), domain));
      } else {
        getOperation().emitError("unknown Map schedule mutation kind");
        return signalPassFailure();
      }
      refreshScheduleFingerprint(plan);
      return;
    }
    getOperation().emitError(
        "test fixture has no expanding-singleton rank-N elementwise task");
    signalPassFailure();
  }
};

class ScheduleLifetimePass : public MutationPass<ScheduleLifetimePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleLifetimePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-lifetime";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleBufferOp buffer :
           plan.getBody().front().getOps<ScheduleBufferOp>()) {
        buffer.setLifetimeEnd(buffer.getLifetimeEnd() + 1);
        refreshScheduleFingerprint(plan);
        return;
      }
    }
    getOperation().emitError("test fixture has no mutable schedule lifetime");
    signalPassFailure();
  }
};

class RenameSymbolsPass : public MutationPass<RenameSymbolsPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RenameSymbolsPass)
  StringRef getArgument() const final { return "shuttle-test-rename-symbols"; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.setSymName("renamed_module");
    int64_t ordinal = 0;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      function.setSymName(("renamed_function_" + Twine(ordinal++)).str());
    }
  }
};

class SetFastRegionPolicyPass : public MutationPass<SetFastRegionPolicyPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetFastRegionPolicyPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-fast-region-policy";
  }
  void runOnOperation() override {
    bool changed = false;
    getOperation().walk([&](RegionOp region) {
      region.setPolicy(NumericalPolicy::Fast);
      changed = true;
    });
    if (!changed) {
      getOperation().emitError("test fixture has no Shuttle region");
      signalPassFailure();
    }
  }
};

class VerifyOutputDependencyCoveragePass
    : public MutationPass<VerifyOutputDependencyCoveragePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VerifyOutputDependencyCoveragePass)
  StringRef getArgument() const final {
    return "shuttle-test-verify-output-dependency-coverage";
  }
  StringRef getDescription() const final {
    return "Verify every covered source result reaches a public output";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto manifest = module->getAttrOfType<DictionaryAttr>(kManifest);
    auto complete = manifest ? manifest.getAs<ArrayAttr>("complete")
                             : ArrayAttr{};
    auto excluded = manifest ? manifest.getAs<ArrayAttr>("excluded")
                             : ArrayAttr{};
    if (!complete || !excluded || !excluded.empty()) {
      module.emitError(
          "output dependency coverage requires complete selected coverage");
      return signalPassFailure();
    }

    llvm::SmallDenseSet<Attribute> expected(complete.begin(), complete.end());
    llvm::SmallDenseSet<Attribute> reachable;
    llvm::DenseSet<Value> seenValues;
    llvm::DenseSet<Operation *> seenOperations;
    auto collectOperationSources = [&](Operation *operation) {
      if (Attribute source = operation->getAttr("source")) {
        reachable.insert(source);
      }
      if (auto sources = operation->getAttrOfType<ArrayAttr>(kSourceRefs)) {
        reachable.insert(sources.begin(), sources.end());
      }
    };

    std::function<void(Value)> collectValue;
    collectValue = [&](Value value) {
      if (!seenValues.insert(value).second) {
        return;
      }
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        Operation *parent = argument.getOwner()->getParentOp();
        if (auto region = dyn_cast_or_null<RegionOp>(parent);
            region && argument.getOwner() == &region.getBody().front()) {
          collectValue(region.getInputs()[argument.getArgNumber()]);
          return;
        }
        if (auto fold = dyn_cast_or_null<FoldOp>(parent);
            fold && argument.getOwner() == &fold.getCombiner().front()) {
          const unsigned argumentNumber = argument.getArgNumber();
          if (argumentNumber < fold.getInputs().size()) {
            collectValue(fold.getInputs()[argumentNumber]);
          } else {
            collectValue(fold.getInitializers()[argumentNumber -
                                                fold.getInputs().size()]);
          }
        }
        return;
      }

      Operation *operation = value.getDefiningOp();
      if (!operation) {
        return;
      }
      if (auto region = dyn_cast<RegionOp>(operation)) {
        auto result = cast<OpResult>(value);
        auto yield = cast<YieldOp>(region.getBody().front().getTerminator());
        collectValue(yield.getValues()[result.getResultNumber()]);
        return;
      }
      if (auto fold = dyn_cast<FoldOp>(operation)) {
        auto result = cast<OpResult>(value);
        auto yield = cast<YieldOp>(fold.getCombiner().front().getTerminator());
        collectOperationSources(operation);
        collectValue(yield.getValues()[result.getResultNumber()]);
        return;
      }
      if (!seenOperations.insert(operation).second) {
        return;
      }
      collectOperationSources(operation);
      for (Value operand : operation->getOperands()) {
        collectValue(operand);
      }
    };

    bool hasOutput = false;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (!function.isPublic()) {
        continue;
      }
      for (Block &block : function.getBody()) {
        auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
        if (!returnOp) {
          continue;
        }
        for (Value output : returnOp.getOperands()) {
          hasOutput = true;
          collectValue(output);
        }
      }
    }
    const bool completeTrace =
        reachable.size() == expected.size() &&
        llvm::all_of(expected, [&](Attribute source) {
          return reachable.contains(source);
        });
    if (!hasOutput || !completeTrace) {
      module.emitError(
          "covered source results do not form a complete public-output dependency trace");
      return signalPassFailure();
    }
    llvm::outs() << "output_dependency_coverage = complete\n";
  }
};

void refreshExecutableBundle(ModuleOp module, bool refreshCodeDigest) {
  auto device = *module.getOps<DeviceModuleOp>().begin();
  auto bundle = *module.getOps<ExecutableBundleOp>().begin();
  if (refreshCodeDigest) {
    std::string digest = executableCodeDigest(device.getCode());
    device.setCodeDigest(digest);
    for (DeviceEntryOp entry :
         device.getBody().front().getOps<DeviceEntryOp>()) {
      entry.setCodeDigest(digest);
    }
  }
  device.setFingerprint(deviceModuleFingerprint(device));
  bundle.setDeviceModuleFingerprint(device.getFingerprint());
  auto abi = *module.getOps<InvocationAbiOp>().begin();
  abi.setFingerprint(invocationAbiFingerprint(abi));
  bundle.setInvocationAbiFingerprint(abi.getFingerprint());
  bundle.setFingerprint(executableBundleFingerprint(bundle));
}

class ReportExecutableBundleFingerprintPass
    : public MutationPass<ReportExecutableBundleFingerprintPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ReportExecutableBundleFingerprintPass)
  StringRef getArgument() const final {
    return "shuttle-test-report-executable-bundle-fingerprint";
  }
  void runOnOperation() override {
    auto bundles = getOperation().getOps<ExecutableBundleOp>();
    if (bundles.empty()) {
      getOperation().emitError("test fixture has no executable bundle");
      return signalPassFailure();
    }
    llvm::outs() << (*bundles.begin()).getFingerprint() << '\n';
  }
};

class CorruptExecutablePass : public MutationPass<CorruptExecutablePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CorruptExecutablePass)
  CorruptExecutablePass() = default;
  CorruptExecutablePass(const CorruptExecutablePass &other)
      : MutationPass<CorruptExecutablePass>(other) {}
  Option<std::string> kind{*this, "kind", llvm::cl::desc("Corruption kind"),
                           llvm::cl::init("")};
  StringRef getArgument() const final {
    return "shuttle-test-corrupt-executable";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto devices = module.getOps<DeviceModuleOp>();
    auto abis = module.getOps<InvocationAbiOp>();
    auto bundles = module.getOps<ExecutableBundleOp>();
    if (devices.empty() || abis.empty() || bundles.empty()) {
      module.emitError("test fixture has no executable bundle");
      return signalPassFailure();
    }
    DeviceModuleOp device = *devices.begin();
    InvocationAbiOp abi = *abis.begin();
    ExecutableBundleOp bundle = *bundles.begin();
    SmallVector<InvocationSlotOp> slots(
        abi.getBody().front().getOps<InvocationSlotOp>());
    SmallVector<DeviceEntryOp> entries(
        device.getBody().front().getOps<DeviceEntryOp>());
    StringRef mutation = kind;
    if (mutation == "slot") {
      slots.front().setSourceBuffer(1);
    } else if (mutation == "stride") {
      slots.front().setStridesAttr(
          DenseI64ArrayAttr::get(module.getContext(), {2, 26}));
    } else if (mutation == "offset") {
      slots.front().setOffset(2);
    } else if (mutation == "alignment") {
      slots.front().setAlignment(1);
    } else if (mutation == "address-space") {
      slots.front().setAddressSpace(ExecutableAddressSpace::Device);
    } else if (mutation == "access") {
      slots.front().setAccess(ExecutableAccess::Write);
    } else if (mutation == "alias") {
      slots[1].setAliasGroup(0);
    } else if (mutation == "dependency") {
      entries[4].setDependenciesAttr(
          DenseI64ArrayAttr::get(module.getContext(), {}));
    } else if (mutation == "predication") {
      entries.front().setPredication(ExecutablePredication::None);
    } else if (mutation == "body") {
      SmallVector<int8_t> code(device.getCode());
      code[entries[1].getCodeOffset() + entries[1].getCodeLength() - 1] ^= 1;
      device.setCodeAttr(DenseI8ArrayAttr::get(module.getContext(), code));
      refreshExecutableBundle(module, true);
      return;
    } else if (mutation == "contract-body") {
      DeviceEntryOp contract = entries.front();
      SmallVector<int8_t> code(device.getCode());
      code[contract.getCodeOffset() + contract.getCodeLength() - 1] ^= 1;
      device.setCodeAttr(DenseI8ArrayAttr::get(module.getContext(), code));
      device.setCodeDigest(executableCodeDigest(device.getCode()));
      contract.setCodeDigest(executableCodeDigest(device.getCode().slice(
          contract.getCodeOffset(), contract.getCodeLength())));
      refreshExecutableBundle(module, false);
      return;
    } else if (mutation == "singleton-broadcast-body") {
      SchedulePlanOp schedule = schedulePlan(module);
      MaterializationPlanOp materialization = materializationPlan(module);
      if (!schedule || !materialization) {
        module.emitError("test fixture has no schedule/materialization plan");
        return signalPassFailure();
      }
      SmallVector<ScheduleTaskOp> scheduleTasks(
          schedule.getBody().front().getOps<ScheduleTaskOp>());
      SmallVector<MaterializationTaskOp> materializationTasks(
          materialization.getBody().front().getOps<MaterializationTaskOp>());
      SmallVector<MaterializationBufferOp> buffers(
          materialization.getBody().front().getOps<MaterializationBufferOp>());
      SmallVector<int8_t> code(device.getCode());
      for (DeviceEntryOp entry : entries) {
        int64_t sourceTask = entry.getSourceTask();
        if (sourceTask < 0 ||
            sourceTask >= static_cast<int64_t>(scheduleTasks.size())) {
          continue;
        }
        int64_t materializationTask =
            scheduleTasks[sourceTask].getSourceTask();
        if (materializationTask < 0 ||
            materializationTask >=
                static_cast<int64_t>(materializationTasks.size()) ||
            !isExpandingSingletonMapTask(
                materializationTasks[materializationTask], buffers)) {
          continue;
        }
        code[entry.getCodeOffset() + entry.getCodeLength() - 1] ^= 1;
        device.setCodeAttr(DenseI8ArrayAttr::get(module.getContext(), code));
        device.setCodeDigest(executableCodeDigest(device.getCode()));
        entry.setCodeDigest(executableCodeDigest(device.getCode().slice(
            entry.getCodeOffset(), entry.getCodeLength())));
        refreshExecutableBundle(module, false);
        return;
      }
      module.emitError("test fixture has no singleton broadcast PTX entry");
      return signalPassFailure();
    } else if (mutation == "projected-parameter-body") {
      SchedulePlanOp schedule = schedulePlan(module);
      MaterializationPlanOp materialization = materializationPlan(module);
      if (!schedule || !materialization) {
        module.emitError("test fixture has no schedule/materialization plan");
        return signalPassFailure();
      }
      SmallVector<ScheduleTaskOp> scheduleTasks(
          schedule.getBody().front().getOps<ScheduleTaskOp>());
      SmallVector<MaterializationTaskOp> materializationTasks(
          materialization.getBody().front().getOps<MaterializationTaskOp>());
      SmallVector<MaterializationBufferOp> buffers(
          materialization.getBody().front().getOps<MaterializationBufferOp>());
      SmallVector<int8_t> code(device.getCode());
      for (DeviceEntryOp entry : entries) {
        int64_t sourceTask = entry.getSourceTask();
        if (sourceTask < 0 ||
            sourceTask >= static_cast<int64_t>(scheduleTasks.size())) {
          continue;
        }
        int64_t materializationTask = scheduleTasks[sourceTask].getSourceTask();
        if (materializationTask < 0 ||
            materializationTask >=
                static_cast<int64_t>(materializationTasks.size()) ||
            !isProjectedParameterMapTask(
                materializationTasks[materializationTask], buffers)) {
          continue;
        }
        code[entry.getCodeOffset() + entry.getCodeLength() - 1] ^= 1;
        device.setCodeAttr(DenseI8ArrayAttr::get(module.getContext(), code));
        device.setCodeDigest(executableCodeDigest(device.getCode()));
        entry.setCodeDigest(executableCodeDigest(device.getCode().slice(
            entry.getCodeOffset(), entry.getCodeLength())));
        refreshExecutableBundle(module, false);
        return;
      }
      module.emitError("test fixture has no projected parameter PTX entry");
      return signalPassFailure();
    } else if (mutation == "prefix-fold-body") {
      SchedulePlanOp schedule = schedulePlan(module);
      MaterializationPlanOp materialization = materializationPlan(module);
      if (!schedule || !materialization) {
        module.emitError("test fixture has no schedule/materialization plan");
        return signalPassFailure();
      }
      SmallVector<ScheduleTaskOp> scheduleTasks(
          schedule.getBody().front().getOps<ScheduleTaskOp>());
      SmallVector<MaterializationTaskOp> materializationTasks(
          materialization.getBody().front().getOps<MaterializationTaskOp>());
      SmallVector<int8_t> code(device.getCode());
      for (DeviceEntryOp entry : entries) {
        int64_t sourceTask = entry.getSourceTask();
        if (sourceTask < 0 ||
            sourceTask >= static_cast<int64_t>(scheduleTasks.size()) ||
            scheduleTasks[sourceTask].getKind() !=
                ScheduleTaskKind::ColumnFold) {
          continue;
        }
        int64_t materializationTask = scheduleTasks[sourceTask].getSourceTask();
        if (materializationTask < 0 ||
            materializationTask >=
                static_cast<int64_t>(materializationTasks.size())) {
          continue;
        }
        MaterializationTaskOp fold =
            materializationTasks[materializationTask];
        if (fold.getKind() != MaterializationTaskKind::Fold ||
            fold.getDomainShape().size() < 3 ||
            fold.getReductionDimensions().size() + 1 !=
                fold.getDomainShape().size()) {
          continue;
        }
        code[entry.getCodeOffset() + entry.getCodeLength() - 1] ^= 1;
        device.setCodeAttr(DenseI8ArrayAttr::get(module.getContext(), code));
        device.setCodeDigest(executableCodeDigest(device.getCode()));
        entry.setCodeDigest(executableCodeDigest(device.getCode().slice(
            entry.getCodeOffset(), entry.getCodeLength())));
        refreshExecutableBundle(module, false);
        return;
      }
      module.emitError("test fixture has no full-prefix Fold PTX entry");
      return signalPassFailure();
    } else if (mutation == "output-binding") {
      SmallVector<InvocationSlotOp> outputs;
      for (InvocationSlotOp slot : slots) {
        if (slot.getAccess() == ExecutableAccess::Write) {
          outputs.push_back(slot);
        }
      }
      std::optional<std::pair<int64_t, int64_t>> pair;
      for (auto [index, first] : llvm::enumerate(outputs)) {
        for (InvocationSlotOp second :
             ArrayRef(outputs).drop_front(index + 1)) {
          if (first.getTensorType() == second.getTensorType()) {
            pair = std::make_pair(first.getOrdinal(), second.getOrdinal());
            break;
          }
        }
        if (pair) {
          break;
        }
      }
      if (!pair) {
        module.emitError(
            "test fixture has no two same-typed executable outputs");
        return signalPassFailure();
      }
      DeviceEntryOp firstProducer;
      DeviceEntryOp secondProducer;
      for (DeviceEntryOp entry : entries) {
        if (entry.getOutputBuffers() == ArrayRef<int64_t>{pair->first}) {
          firstProducer = entry;
        } else if (entry.getOutputBuffers() ==
                   ArrayRef<int64_t>{pair->second}) {
          secondProducer = entry;
        }
      }
      if (!firstProducer || !secondProducer) {
        module.emitError("test fixture outputs lack unique producers");
        return signalPassFailure();
      }
      firstProducer.setOutputBuffersAttr(
          DenseI64ArrayAttr::get(module.getContext(), {pair->second}));
      secondProducer.setOutputBuffersAttr(
          DenseI64ArrayAttr::get(module.getContext(), {pair->first}));
    } else if (mutation == "code-byte") {
      SmallVector<int8_t> code(device.getCode());
      code.back() ^= 1;
      device.setCodeAttr(DenseI8ArrayAttr::get(module.getContext(), code));
      return;
    } else if (mutation == "code-digest") {
      device.setCodeDigest(std::string(64, '0'));
      return;
    } else if (mutation == "cross-object") {
      entries.front().setCodeDigest(std::string(64, '0'));
      return;
    } else if (mutation == "root") {
      bundle.setDeviceModuleFingerprint(std::string(64, '0'));
      bundle.setFingerprint(executableBundleFingerprint(bundle));
      return;
    } else if (mutation == "unknown-attr") {
      bundle->setAttr(
          "shuttle.test_semantic",
          IntegerAttr::get(IntegerType::get(module.getContext(), 64), 7));
      return;
    } else {
      module.emitError("unknown executable corruption kind");
      return signalPassFailure();
    }
    refreshExecutableBundle(module, false);
  }
};

} // namespace

void registerMutationPasses() {
  PassRegistration<ExcludedKindPass>();
  PassRegistration<ExcludedAttributePass>();
  PassRegistration<NestedExcludedAttributePass>();
  PassRegistration<ExcludedOperandPass>();
  PassRegistration<UnsupportedAbsorptionPass>();
  PassRegistration<ReturnRewirePass>();
  PassRegistration<ManifestDigestPass>();
  PassRegistration<FoldOwnerPass>();
  PassRegistration<FoldOwnerMismatchPass>();
  PassRegistration<FoldAddSourcePass>();
  PassRegistration<FoldAddOwnerMismatchPass>();
  PassRegistration<FoldAddOwnerMissingPass>();
  PassRegistration<FoldAddDuplicateSourcePass>();
  PassRegistration<FoldYieldOwnerPass>();
  PassRegistration<FoldOwnerDuplicatePass>();
  PassRegistration<FoldYieldRewirePass>();
  PassRegistration<FoldAddFastMathPass>();
  PassRegistration<FoldYieldAttributePass>();
  PassRegistration<FoldAttributePass>();
  PassRegistration<BroadcastDuplicatePass>();
  PassRegistration<BroadcastMappedAxesSwapPass>();
  PassRegistration<BroadcastMappedAxisDuplicatePass>();
  PassRegistration<BroadcastWrongDivisorPass>();
  PassRegistration<BroadcastLiteralZeroPass>();
  PassRegistration<BroadcastCompositeDividendPass>();
  PassRegistration<BroadcastWrongResultExtentPass>();
  PassRegistration<BroadcastDirectSingletonPass>();
  PassRegistration<BroadcastExpandNonSingletonPass>();
  PassRegistration<BroadcastResultMapMutationPass>();
  PassRegistration<ProjectedParameterMapMutationPass>();
  PassRegistration<MapConstantZeroPointwisePass>();
  PassRegistration<BroadcastReplayAsReshapePass>();
  PassRegistration<ReshapeReplayAsBroadcastPass>();
  PassRegistration<ReshapeAmbiguousPass>();
  PassRegistration<MapAttributePass>();
  PassRegistration<MapYieldAttributePass>();
  PassRegistration<MapStructuralSemanticPass>();
  PassRegistration<ManifestVersionPass>();
  PassRegistration<ManifestVersionMissingPass>();
  PassRegistration<ReportNormalizedFingerprintPass>();
  PassRegistration<ReportMaterializationFingerprintPass>();
  PassRegistration<SetPreferFusionTuningPass>();
  PassRegistration<RewireOnlineMaximumYieldPass>();
  PassRegistration<VerifyOnlineSufficientStatePass>();
  PassRegistration<VerifyOnlinePtxOraclePass>();
  PassRegistration<VerifyOnlineGpuExecutionPass>();
  PassRegistration<VerifyOnlineFallbackPass>();
  PassRegistration<CorruptOnlineGpuSemanticsPass>();
  PassRegistration<DropOnlineRecomputationPass>();
  PassRegistration<SwapOnlineRecomputationPass>();
  PassRegistration<MaterializationDeleteTaskPass>();
  PassRegistration<MaterializationReorderTasksPass>();
  PassRegistration<MaterializationSelfConsistentReorderPass>();
  PassRegistration<MaterializationReplaySourcePass>();
  PassRegistration<MaterializationDuplicateAlgebraSourcePass>();
  PassRegistration<MaterializationSwapSameTypeEdgesPass>();
  PassRegistration<MaterializationContractShapePass>();
  PassRegistration<MaterializationFoldMutationPass>();
  PassRegistration<MaterializationMapMutationPass>();
  PassRegistration<MaterializationDomainPass<true>>();
  PassRegistration<MaterializationDomainPass<false>>();
  PassRegistration<MaterializationUnknownAttributePass>();
  PassRegistration<ReportScheduleFingerprintPass>();
  PassRegistration<ScheduleIndexingPass>();
  PassRegistration<ScheduleAxisPass>();
  PassRegistration<ScheduleTilePass>();
  PassRegistration<ScheduleResourcePass>();
  PassRegistration<ScheduleDependencyPass>();
  PassRegistration<ScheduleDeleteTaskPass>();
  PassRegistration<ScheduleSelfConsistentReorderPass>();
  PassRegistration<ScheduleReplaySourceTaskPass>();
  PassRegistration<ScheduleSourcePlanFingerprintPass>();
  PassRegistration<ScheduleTypePass>();
  PassRegistration<ClonePlanPass<MaterializationPlanOp>>();
  PassRegistration<ClonePlanPass<SchedulePlanOp>>();
  PassRegistration<ScheduleUnknownAttributePass>();
  PassRegistration<ScheduleTaskUnknownAttributePass>();
  PassRegistration<ScheduleTargetProfilePass>();
  PassRegistration<ScheduleReductionOrderPass>();
  PassRegistration<ContractScheduleMutationPass>();
  PassRegistration<FoldScheduleMutationPass>();
  PassRegistration<MapScheduleMutationPass>();
  PassRegistration<ScheduleLifetimePass>();
  PassRegistration<RenameSymbolsPass>();
  PassRegistration<SetFastRegionPolicyPass>();
  PassRegistration<VerifyOutputDependencyCoveragePass>();
  PassRegistration<ReportExecutableBundleFingerprintPass>();
  PassRegistration<CorruptExecutablePass>();
}

} // namespace mlir::shuttle::test
