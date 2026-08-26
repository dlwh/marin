// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "TestPasses.h"

#include <functional>
#include <string>
#include <type_traits>

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
    const bool recomputationMatches =
        plannedRecomputedSources.size() == requiredRecomputedSources.size() &&
        llvm::all_of(requiredRecomputedSources, [&](Attribute source) {
          return plannedRecomputedSources.contains(source);
        });
    if (!relationFeedsScore || !contractFeedsScore ||
        !exponentialUsesMaximum || !scoreMinusMaximumMap ||
        !denominatorUsesExponential ||
        !normalizedWeightOperand || !reductionsAlign ||
        !recomputationMatches ||
        !stateCoordinatesAlign || !contractionReducesStateCoordinate) {
      module.emitError(
          "online composition does not prove {m,l,o} state semantics");
      return signalPassFailure();
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
        *contractEvaluations <= 0) {
      module.emitError(
          "online composition lacks relative cost and traffic improvement");
      return signalPassFailure();
    }
    llvm::outs() << "online_state = {m,l,o}; source_cut = complete; "
                    "representative_score_materialized = false; "
                    "temporary_bytes = improved; temporary_traffic = improved\n";
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
  PassRegistration<VerifyOnlineFallbackPass>();
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
