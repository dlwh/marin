// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/Passes.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "ObserverInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/IR/ShuttleOps.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::shuttle {

#define GEN_PASS_DEF_SHUTTLEANNOTATESOURCEPASS
#define GEN_PASS_DEF_SHUTTLEFORMSTRUCTURALREGIONSPASS
#define GEN_PASS_DEF_SHUTTLECONVERTSTABLEHLOTOALGEBRAPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYSOURCECOVERAGEPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYSEMANTICERASUREPASS
#define GEN_PASS_DEF_SHUTTLECANONICALIZEPASS
#define GEN_PASS_DEF_SHUTTLELOWERALGEBRATOSTABLEHLOPASS
#define GEN_PASS_DEF_SHUTTLESTRIPSOURCEPROVENANCEPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYNOSHUTTLEOPSPASS
#define GEN_PASS_DEF_SHUTTLEPLANROWFOLDMATERIALIZATIONPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYMATERIALIZATIONPLANPASS
#define GEN_PASS_DEF_SHUTTLEPLANSIMT32ROWFOLDSCHEDULEPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYSIMT32ROWFOLDSCHEDULEPASS
#define GEN_PASS_DEF_SHUTTLEBUILDGPUEXECUTABLEBUNDLEPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYGPUEXECUTABLEBUNDLEPASS
#define GEN_PASS_DEF_SHUTTLEREPLACEWITHGPUEXECUTABLEBUNDLEPASS
#include "shuttle/Transforms/Passes.h.inc"

namespace {

ShuttlePipelineOptions commandLinePipelineOptions(NumericalPolicy numerics) {
  ShuttlePipelineOptions options;
  options.numerics = numerics;
  if (numerics == NumericalPolicy::Fast) {
    options.canonicalOptions =
        R"json({"execution_mode":"stablehlo_round_trip","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  }
  return options;
}

constexpr llvm::StringLiteral kSourceRefsAttribute = "shuttle.source_refs";
constexpr llvm::StringLiteral kSelectedAttribute = "shuttle.selected";
constexpr llvm::StringLiteral kFunctionOrdinalAttribute =
    "shuttle.function_ordinal";
constexpr llvm::StringLiteral kOperationRefAttribute = "shuttle.operation_ref";
constexpr llvm::StringLiteral kCoverageManifestAttribute =
    "shuttle.coverage_manifest";
constexpr llvm::StringLiteral kRegionResultSourcesAttribute =
    "shuttle.result_source_refs";
constexpr llvm::StringLiteral kDefaultCanonicalTuning =
    R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
constexpr llvm::StringLiteral kFusionCanonicalTuning =
    R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
constexpr llvm::StringLiteral kMaterializationCanonicalTuning =
    R"json({"cluster_shape":[],"materialization":"prefer_materialization","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";

constexpr llvm::StringLiteral kManifestComplete = "complete";
constexpr llvm::StringLiteral kManifestSelectedRegions = "selected_regions";
constexpr llvm::StringLiteral kManifestExcluded = "excluded";
constexpr llvm::StringLiteral kManifestZeroResultOperations =
    "zero_result_operations";
constexpr llvm::StringLiteral kManifestFunctionResults = "function_results";

std::string sha256(llvm::StringRef value) {
  llvm::SHA256 digest;
  digest.update(value);
  return llvm::toHex(digest.final(), true);
}

StringRef policyName(NumericalPolicy numerics) {
  return numerics == NumericalPolicy::SourceOrdered ? "source_ordered" : "fast";
}

struct CandidateComponent {
  SmallVector<Operation *> operations;
};

LogicalResult
walkSourcePreorder(Region &region,
                   llvm::function_ref<LogicalResult(Operation *)> visitor) {
  for (Block &block : region) {
    for (Operation &operation : block) {
      if (failed(visitor(&operation))) {
        return failure();
      }
      for (Region &nested : operation.getRegions()) {
        if (failed(walkSourcePreorder(nested, visitor))) {
          return failure();
        }
      }
    }
  }
  return success();
}

LogicalResult
walkSourcePreorder(Operation *operation,
                   llvm::function_ref<LogicalResult(Operation *)> visitor) {
  if (failed(visitor(operation))) {
    return failure();
  }
  for (Region &nested : operation->getRegions()) {
    if (failed(walkSourcePreorder(nested, visitor))) {
      return failure();
    }
  }
  return success();
}

bool containsShuttleAttribute(Attribute attribute) {
  bool found = false;
  attribute.walk([&](Attribute nested) {
    if (nested.getDialect().getNamespace() ==
        ShuttleDialect::getDialectNamespace()) {
      found = true;
      return;
    }
    if (auto opaque = dyn_cast<OpaqueAttr>(nested);
        opaque && opaque.getDialectNamespace().getValue() ==
                      ShuttleDialect::getDialectNamespace()) {
      found = true;
    }
  });
  return found;
}

ArrayAttr sourceRefs(Operation *operation) {
  return operation->getAttrOfType<ArrayAttr>(kSourceRefsAttribute);
}

SourceRefAttr singleSourceRef(Operation *operation) {
  ArrayAttr refs = sourceRefs(operation);
  if (!refs || refs.size() != 1) {
    return {};
  }
  return dyn_cast<SourceRefAttr>(refs[0]);
}

DenseI64ArrayAttr operationRefForSource(SourceRefAttr source) {
  return DenseI64ArrayAttr::get(
      source.getContext(),
      {static_cast<int64_t>(source.getFunctionOrdinal()),
       static_cast<int64_t>(source.getBlockOrdinal()),
       static_cast<int64_t>(source.getOperationOrdinal())});
}

DictionaryAttr sourceAttributes(Operation *operation) {
  SmallVector<NamedAttribute> attributes;
  for (NamedAttribute attribute : operation->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name == kSourceRefsAttribute || name == kSelectedAttribute ||
        name == kFunctionOrdinalAttribute || name == kOperationRefAttribute ||
        name == kCoverageManifestAttribute ||
        name == kRegionResultSourcesAttribute) {
      continue;
    }
    attributes.push_back(attribute);
  }
  return DictionaryAttr::get(operation->getContext(), attributes);
}

DictionaryAttr normalizedOperationFingerprint(Operation *operation) {
  MLIRContext *context = operation->getContext();
  SmallVector<Attribute> resultTypes;
  for (Type type : operation->getResultTypes()) {
    resultTypes.push_back(TypeAttr::get(type));
  }
  NamedAttribute fields[] = {
      NamedAttribute(StringAttr::get(context, "attributes"),
                     sourceAttributes(operation)),
      NamedAttribute(
          StringAttr::get(context, "name"),
          StringAttr::get(context, operation->getName().getStringRef())),
      NamedAttribute(StringAttr::get(context, "result_types"),
                     ArrayAttr::get(context, resultTypes)),
  };
  return DictionaryAttr::get(context, fields);
}

Attribute valueAnchor(Value value) {
  MLIRContext *context = value.getContext();
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *parent = argument.getOwner()->getParentOp();
    auto function = dyn_cast<func::FuncOp>(parent);
    auto functionOrdinal =
        function
            ? function->getAttrOfType<IntegerAttr>(kFunctionOrdinalAttribute)
            : IntegerAttr{};
    if (!functionOrdinal ||
        &function.getBody().front() != argument.getOwner()) {
      auto ownerRef =
          parent
              ? parent->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute)
              : DenseI64ArrayAttr{};
      if (!ownerRef) {
        return {};
      }
      unsigned regionOrdinal = 0;
      unsigned blockOrdinal = 0;
      bool found = false;
      for (auto [regionIndex, region] : llvm::enumerate(parent->getRegions())) {
        for (auto [blockIndex, block] : llvm::enumerate(region)) {
          if (&block == argument.getOwner()) {
            regionOrdinal = regionIndex;
            blockOrdinal = blockIndex;
            found = true;
          }
        }
      }
      if (!found) {
        return {};
      }
      NamedAttribute nestedFields[] = {
          NamedAttribute(StringAttr::get(context, "owner"), ownerRef),
          NamedAttribute(
              StringAttr::get(context, "region"),
              IntegerAttr::get(IntegerType::get(context, 64), regionOrdinal)),
          NamedAttribute(
              StringAttr::get(context, "block"),
              IntegerAttr::get(IntegerType::get(context, 64), blockOrdinal)),
          NamedAttribute(StringAttr::get(context, "argument"),
                         IntegerAttr::get(IntegerType::get(context, 64),
                                          argument.getArgNumber())),
      };
      return DictionaryAttr::get(context, nestedFields);
    }
    NamedAttribute fields[] = {
        NamedAttribute(StringAttr::get(context, "argument"),
                       UnitAttr::get(context)),
        NamedAttribute(StringAttr::get(context, "function"), functionOrdinal),
        NamedAttribute(StringAttr::get(context, "ordinal"),
                       IntegerAttr::get(IntegerType::get(context, 64),
                                        argument.getArgNumber())),
    };
    return DictionaryAttr::get(context, fields);
  }

  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    return {};
  }
  Operation *owner = result.getOwner();
  ArrayAttr refs = sourceRefs(owner);
  if (!refs) {
    refs = owner->getAttrOfType<ArrayAttr>(kRegionResultSourcesAttribute);
  }
  if (!refs || result.getResultNumber() >= refs.size()) {
    return {};
  }
  return refs[result.getResultNumber()];
}

bool hasOnlyRankedSupportedFloats(ValueRange values) {
  return llvm::all_of(values, [](Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type &&
           (type.getElementType().isF32() || type.getElementType().isBF16());
  });
}

Type elementType(Value value) {
  return cast<RankedTensorType>(value.getType()).getElementType();
}

bool hasEqualShapes(Operation *operation) {
  auto first = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = cast<RankedTensorType>(operation->getResult(0).getType());
  return first.getShape() == result.getShape();
}

bool hasValidBroadcastDimensions(Operation *operation,
                                 DenseI64ArrayAttr dimensions) {
  auto result = cast<RankedTensorType>(operation->getResult(0).getType());
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t dimension : dimensions.asArrayRef()) {
    if (dimension < 0 || dimension >= result.getRank() ||
        !seen.insert(dimension).second) {
      return false;
    }
  }
  return true;
}

bool isSupportedAffineBroadcast(Operation *operation) {
  if (operation->getNumOperands() != 1 || operation->getNumResults() != 1) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  auto dimensions =
      operation->getAttrOfType<DenseI64ArrayAttr>("broadcast_dimensions");
  DictionaryAttr attributes = sourceAttributes(operation);
  if (!input || !result || !input.getElementType().isF32() ||
      !result.getElementType().isF32() || !input.hasStaticShape() ||
      !result.hasStaticShape() || !dimensions ||
      dimensions.size() != static_cast<size_t>(input.getRank()) ||
      attributes.size() != 1 || !attributes.get("broadcast_dimensions") ||
      input == result || !hasValidBroadcastDimensions(operation, dimensions)) {
    return false;
  }
  unsigned expandingSingletons = 0;
  for (auto [inputDimension, resultDimension] :
       llvm::enumerate(dimensions.asArrayRef())) {
    const int64_t inputExtent = input.getDimSize(inputDimension);
    const int64_t resultExtent = result.getDimSize(resultDimension);
    if (inputExtent == resultExtent) {
      continue;
    }
    if (inputExtent != 1 || resultExtent <= 1) {
      return false;
    }
    ++expandingSingletons;
  }
  if (expandingSingletons == 0) {
    return true;
  }
  const bool rankTwoExpansion = input.getRank() == 2 && result.getRank() == 2;
  const bool rankNExpansion = input.getRank() == result.getRank() &&
                              input.getRank() >= 3 && input.getRank() <= 8;
  return (rankTwoExpansion && expandingSingletons == 1) || rankNExpansion;
}

bool isCopyFreeSingletonReshape(Operation *operation) {
  if (operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
      !sourceAttributes(operation).empty()) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!input || !result || !input.getElementType().isF32() ||
      !result.getElementType().isF32() || !input.hasStaticShape() ||
      !result.hasStaticShape()) {
    return false;
  }
  SmallVector<int64_t> inputNonSingleton;
  SmallVector<int64_t> resultNonSingleton;
  llvm::copy_if(input.getShape(), std::back_inserter(inputNonSingleton),
                [](int64_t extent) { return extent != 1; });
  llvm::copy_if(result.getShape(), std::back_inserter(resultNonSingleton),
                [](int64_t extent) { return extent != 1; });
  return inputNonSingleton == resultNonSingleton && input != result;
}

bool hasDefaultDotPrecision(Operation *operation) {
  auto precision = operation->getAttrOfType<ArrayAttr>("precision_config");
  if (!precision) {
    return true;
  }
  return precision.size() == operation->getNumOperands() &&
         llvm::all_of(precision, [](Attribute attribute) {
           auto value = dyn_cast<stablehlo::PrecisionAttr>(attribute);
           return value && value.getValue() == stablehlo::Precision::DEFAULT;
         });
}

bool isSupportedStablehloReduce(Operation *operation) {
  if (operation->getName().getStringRef() !=
          stablehlo::ReduceOp::getOperationName() ||
      operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
      operation->getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation->getRegion(0))) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto init = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  auto dimensions = operation->getAttrOfType<DenseI64ArrayAttr>("dimensions");
  DictionaryAttr reduceAttributes = sourceAttributes(operation);
  if (!input || input.getRank() == 0 || !input.getElementType().isF32() ||
      !init || init.getRank() != 0 || !init.getElementType().isF32() ||
      !result || !result.getElementType().isF32() || !dimensions ||
      dimensions.empty() || reduceAttributes.size() != 1 ||
      !reduceAttributes.get("dimensions")) {
    return false;
  }
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t dimension : dimensions.asArrayRef()) {
    if (dimension < 0 || dimension >= input.getRank() ||
        !seen.insert(dimension).second) {
      return false;
    }
  }
  SmallVector<int64_t> expected;
  for (int64_t dimension = 0; dimension < input.getRank(); ++dimension) {
    if (!seen.contains(dimension)) {
      expected.push_back(input.getDimSize(dimension));
    }
  }
  Block &body = operation->getRegion(0).front();
  if (result.getShape() != ArrayRef<int64_t>(expected) ||
      body.getNumArguments() != 2 ||
      !llvm::all_of(body.getArguments(),
                    [](BlockArgument argument) {
                      auto type =
                          dyn_cast<RankedTensorType>(argument.getType());
                      return type && type.getRank() == 0 &&
                             type.getElementType().isF32();
                    }) ||
      body.getOperations().size() != 2) {
    return false;
  }
  Operation &add = body.front();
  Operation &terminator = body.back();
  return add.getName().getStringRef() == stablehlo::AddOp::getOperationName() &&
         sourceAttributes(&add).empty() && add.getNumOperands() == 2 &&
         add.getNumResults() == 1 && add.getOperand(0) == body.getArgument(0) &&
         add.getOperand(1) == body.getArgument(1) &&
         add.getResult(0).getType() == body.getArgument(0).getType() &&
         terminator.getName().getStringRef() ==
             stablehlo::ReturnOp::getOperationName() &&
         sourceAttributes(&terminator).empty() &&
         terminator.getNumOperands() == 1 &&
         terminator.getOperand(0) == add.getResult(0);
}

bool isSupportedStablehlo(Operation *operation) {
  if (isSupportedStablehloReduce(operation)) {
    return true;
  }
  if (operation->getNumRegions() != 0 || !isMemoryEffectFree(operation) ||
      !hasOnlyRankedSupportedFloats(operation->getOperands()) ||
      !hasOnlyRankedSupportedFloats(operation->getResults())) {
    return false;
  }

  StringRef name = operation->getName().getStringRef();
  if (name == stablehlo::DotGeneralOp::getOperationName()) {
    if (operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
        operation->hasAttr("algorithm") || !hasDefaultDotPrecision(operation)) {
      return false;
    }
    return elementType(operation->getOperand(0)) ==
               elementType(operation->getOperand(1)) &&
           (elementType(operation->getOperand(0)).isF32() ||
            elementType(operation->getOperand(0)).isBF16()) &&
           elementType(operation->getResult(0)).isF32() &&
           static_cast<bool>(
               operation->getAttrOfType<stablehlo::DotDimensionNumbersAttr>(
                   "dot_dimension_numbers"));
  }
  if (name == stablehlo::ConvertOp::getOperationName()) {
    if (operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
        !hasEqualShapes(operation)) {
      return false;
    }
    Type input = elementType(operation->getOperand(0));
    Type result = elementType(operation->getResult(0));
    return (input.isF32() && result.isF32()) ||
           (input.isF32() && result.isBF16()) ||
           (input.isBF16() && result.isF32());
  }
  if (name == stablehlo::TanhOp::getOperationName() ||
      name == stablehlo::ExpOp::getOperationName() ||
      name == stablehlo::RsqrtOp::getOperationName() ||
      name == stablehlo::NegOp::getOperationName()) {
    return operation->getNumOperands() == 1 &&
           operation->getNumResults() == 1 &&
           !operation->hasAttr("result_accuracy") &&
           sourceAttributes(operation).empty() &&
           elementType(operation->getOperand(0)).isF32() &&
           operation->getOperand(0).getType() ==
               operation->getResult(0).getType();
  }
  if (name == stablehlo::AddOp::getOperationName() ||
      name == stablehlo::MulOp::getOperationName() ||
      name == stablehlo::SubtractOp::getOperationName() ||
      name == stablehlo::DivOp::getOperationName()) {
    return operation->getNumOperands() == 2 &&
           operation->getNumResults() == 1 &&
           elementType(operation->getOperand(0)).isF32() &&
           operation->getOperand(0).getType() ==
               operation->getOperand(1).getType() &&
           operation->getOperand(0).getType() ==
               operation->getResult(0).getType();
  }
  if (name == stablehlo::TransposeOp::getOperationName()) {
    auto permutation =
        operation->getAttrOfType<DenseI64ArrayAttr>("permutation");
    auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
    return operation->getNumOperands() == 1 &&
           operation->getNumResults() == 1 && permutation &&
           elementType(operation->getOperand(0)).isF32() &&
           permutation.size() == static_cast<size_t>(input.getRank());
  }
  if (name == stablehlo::ConstantOp::getOperationName()) {
    auto value = operation->getAttrOfType<DenseElementsAttr>("value");
    auto result = cast<RankedTensorType>(operation->getResult(0).getType());
    return operation->getNumOperands() == 0 &&
           operation->getNumResults() == 1 && result.getRank() == 0 &&
           result.getElementType().isF32() && value && value.isSplat();
  }
  if (name == stablehlo::BroadcastInDimOp::getOperationName()) {
    return isSupportedAffineBroadcast(operation);
  }
  if (name == stablehlo::ReshapeOp::getOperationName()) {
    return isCopyFreeSingletonReshape(operation);
  }
  return false;
}

SmallVector<CandidateComponent>
partitionSupportedInterval(ArrayRef<Operation *> interval) {
  llvm::SmallPtrSet<Operation *, 16> members(interval.begin(), interval.end());
  llvm::SmallPtrSet<Operation *, 16> visited;
  SmallVector<CandidateComponent> components;
  for (Operation *seed : interval) {
    if (!visited.insert(seed).second) {
      continue;
    }
    CandidateComponent component;
    SmallVector<Operation *> worklist{seed};
    while (!worklist.empty()) {
      Operation *operation = worklist.pop_back_val();
      component.operations.push_back(operation);
      for (Value operand : operation->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (producer && members.contains(producer) &&
            visited.insert(producer).second) {
          worklist.push_back(producer);
        }
      }
      for (Value result : operation->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (members.contains(user) && visited.insert(user).second) {
            worklist.push_back(user);
          }
        }
      }
    }
    llvm::sort(component.operations,
               [interval](Operation *left, Operation *right) {
                 auto position = [interval](Operation *operation) {
                   return llvm::find(interval, operation) - interval.begin();
                 };
                 return position(left) < position(right);
               });
    components.push_back(std::move(component));
  }
  llvm::DenseMap<Operation *, unsigned> componentFor;
  for (auto [componentOrdinal, component] : llvm::enumerate(components)) {
    for (Operation *operation : component.operations) {
      componentFor[operation] = componentOrdinal;
    }
  }
  SmallVector<CandidateComponent> sourceContiguousComponents;
  unsigned previousComponent = componentFor[interval.front()];
  sourceContiguousComponents.emplace_back();
  for (Operation *operation : interval) {
    unsigned component = componentFor[operation];
    if (component != previousComponent) {
      sourceContiguousComponents.emplace_back();
      previousComponent = component;
    }
    sourceContiguousComponents.back().operations.push_back(operation);
  }
  return sourceContiguousComponents;
}

unsigned rankNExpandingSingletonBroadcasts(Operation *operation) {
  if (operation->getName().getStringRef() !=
          stablehlo::BroadcastInDimOp::getOperationName() ||
      !isSupportedAffineBroadcast(operation)) {
    return 0;
  }
  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = cast<RankedTensorType>(operation->getResult(0).getType());
  if (input.getRank() < 3 || input.getRank() > 8 ||
      input.getRank() != result.getRank()) {
    return 0;
  }
  auto dimensions =
      operation->getAttrOfType<DenseI64ArrayAttr>("broadcast_dimensions");
  return llvm::count_if(llvm::enumerate(dimensions.asArrayRef()),
                        [&](auto indexedDimension) {
                          return input.getDimSize(indexedDimension.index()) !=
                                 result.getDimSize(indexedDimension.value());
                        });
}

bool isTerminalRankNAdditiveReduce(Operation *operation) {
  if (!isSupportedStablehloReduce(operation)) {
    return false;
  }
  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto dimensions = operation->getAttrOfType<DenseI64ArrayAttr>("dimensions");
  return input.getRank() >= 3 && input.getRank() <= 8 &&
         dimensions.asArrayRef() == ArrayRef<int64_t>({input.getRank() - 1});
}

bool isFullPrefixRankNAdditiveReduce(Operation *operation) {
  if (!isSupportedStablehloReduce(operation)) {
    return false;
  }
  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto dimensions =
      operation->getAttrOfType<DenseI64ArrayAttr>("dimensions");
  return input.hasStaticShape() && input.getRank() >= 3 &&
         input.getRank() <= 8 &&
         llvm::all_of(input.getShape(),
                      [](int64_t extent) { return extent > 0; }) &&
         dimensions.size() == static_cast<size_t>(input.getRank() - 1) &&
         llvm::all_of(llvm::enumerate(dimensions.asArrayRef()),
                      [](auto indexedDimension) {
                        return indexedDimension.value() ==
                               static_cast<int64_t>(indexedDimension.index());
                      });
}

bool isRankTwoAdditiveReduce(Operation *operation) {
  if (!isSupportedStablehloReduce(operation)) {
    return false;
  }
  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto dimensions =
      operation->getAttrOfType<DenseI64ArrayAttr>("dimensions");
  return input.hasStaticShape() && input.getRank() == 2 &&
         llvm::all_of(input.getShape(),
                      [](int64_t extent) { return extent > 0; }) &&
         dimensions.size() == 1 && dimensions[0] >= 0 &&
         dimensions[0] < input.getRank();
}

bool isClosedRankOneBf16Result(
    Operation *operation,
    const llvm::SmallPtrSetImpl<Operation *> &componentOperations) {
  if (operation->getName().getStringRef() !=
          stablehlo::ConvertOp::getOperationName() ||
      !isSupportedStablehlo(operation) || operation->getNumOperands() != 1 ||
      operation->getNumResults() != 1) {
    return false;
  }
  Operation *producer = operation->getOperand(0).getDefiningOp();
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  return producer && componentOperations.contains(producer) &&
         (isFullPrefixRankNAdditiveReduce(producer) ||
          isRankTwoAdditiveReduce(producer)) &&
         input && result &&
         input.hasStaticShape() && result.hasStaticShape() &&
         input.getRank() == 1 && result.getRank() == 1 &&
         input.getShape() == result.getShape() &&
         input.getElementType().isF32() && result.getElementType().isBF16() &&
         llvm::all_of(result.getShape(),
                      [](int64_t extent) { return extent > 0; });
}

bool isClosedDirectStablehloMap(Operation *operation) {
  if (!isSupportedStablehlo(operation)) {
    return false;
  }
  StringRef name = operation->getName().getStringRef();
  if (name == stablehlo::BroadcastInDimOp::getOperationName()) {
    return rankNExpandingSingletonBroadcasts(operation) == 0;
  }
  return name == stablehlo::ReshapeOp::getOperationName() ||
         name == stablehlo::ConvertOp::getOperationName() ||
         name == stablehlo::TanhOp::getOperationName() ||
         name == stablehlo::ExpOp::getOperationName() ||
         name == stablehlo::RsqrtOp::getOperationName() ||
         name == stablehlo::NegOp::getOperationName() ||
         name == stablehlo::AddOp::getOperationName() ||
         name == stablehlo::MulOp::getOperationName() ||
         name == stablehlo::SubtractOp::getOperationName() ||
         name == stablehlo::DivOp::getOperationName() ||
         name == stablehlo::TransposeOp::getOperationName() ||
         name == stablehlo::ConstantOp::getOperationName();
}

bool isProjectedParameterInsertionStep(Operation *operation) {
  if (operation->getName().getStringRef() !=
          stablehlo::BroadcastInDimOp::getOperationName() ||
      !isSupportedAffineBroadcast(operation)) {
    return false;
  }
  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = cast<RankedTensorType>(operation->getResult(0).getType());
  auto dimensions =
      operation->getAttrOfType<DenseI64ArrayAttr>("broadcast_dimensions");
  if (input.getRank() < 1 || input.getRank() >= result.getRank() ||
      result.getRank() > 8) {
    return false;
  }
  llvm::SmallDenseSet<int64_t> mapped(dimensions.asArrayRef().begin(),
                                      dimensions.asArrayRef().end());
  for (auto [inputAxis, resultAxis] :
       llvm::enumerate(dimensions.asArrayRef())) {
    if (input.getDimSize(inputAxis) != result.getDimSize(resultAxis)) {
      return false;
    }
  }
  for (auto [axis, extent] : llvm::enumerate(result.getShape())) {
    if (!mapped.contains(axis) && extent != 1) {
      return false;
    }
  }
  return true;
}

bool isProjectedBf16ParameterConversion(Operation *operation,
                                        int64_t expansionRank) {
  if (operation->getName().getStringRef() !=
          stablehlo::ConvertOp::getOperationName() ||
      !isSupportedStablehlo(operation) ||
      operation->getOperand(0).getDefiningOp()) {
    return false;
  }
  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = cast<RankedTensorType>(operation->getResult(0).getType());
  return input.hasStaticShape() && result.hasStaticShape() &&
         input.getRank() >= 1 && input.getRank() < expansionRank &&
         input.getShape() == result.getShape() &&
         input.getElementType().isBF16() && result.getElementType().isF32() &&
         llvm::all_of(input.getShape(),
                      [](int64_t extent) { return extent > 0; });
}

bool isExpansionDrivenByProjectedParameter(
    Operation *expansion,
    const llvm::SmallPtrSetImpl<Operation *> &componentOperations) {
  auto expansionType =
      cast<RankedTensorType>(expansion->getResult(0).getType());
  Value producerValue = expansion->getOperand(0);
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (Operation *producer = producerValue.getDefiningOp()) {
    if (!componentOperations.contains(producer) ||
        !visited.insert(producer).second) {
      return false;
    }
    if (isProjectedBf16ParameterConversion(producer,
                                           expansionType.getRank())) {
      return true;
    }
    if (!isProjectedParameterInsertionStep(producer) ||
        producer->getNumOperands() != 1) {
      return false;
    }
    producerValue = producer->getOperand(0);
  }
  return false;
}

bool isAxisAlignedRankNExpansion(Operation *operation) {
  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto dimensions =
      operation->getAttrOfType<DenseI64ArrayAttr>("broadcast_dimensions");
  return dimensions.size() == static_cast<size_t>(input.getRank()) &&
         llvm::all_of(llvm::enumerate(dimensions.asArrayRef()),
                      [](auto indexedDimension) {
                        return indexedDimension.index() ==
                               static_cast<size_t>(indexedDimension.value());
                      });
}

bool isExpansionDrivenByTerminalFold(
    Operation *expansion,
    const llvm::SmallPtrSetImpl<Operation *> &componentOperations) {
  Operation *root = expansion->getOperand(0).getDefiningOp();
  if (!root) {
    return false;
  }
  SmallVector<Operation *> worklist{root};
  llvm::SmallPtrSet<Operation *, 8> visited;
  Operation *terminalFold = nullptr;
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val();
    if (!componentOperations.contains(producer)) {
      return false;
    }
    if (!visited.insert(producer).second) {
      continue;
    }
    if (isTerminalRankNAdditiveReduce(producer)) {
      if (terminalFold && terminalFold != producer) {
        return false;
      }
      terminalFold = producer;
      continue;
    }
    if (!isClosedDirectStablehloMap(producer)) {
      return false;
    }
    if (producer->getNumOperands() == 0) {
      if (producer->getName().getStringRef() !=
          stablehlo::ConstantOp::getOperationName()) {
        return false;
      }
      continue;
    }
    for (Value operand : producer->getOperands()) {
      Operation *operandProducer = operand.getDefiningOp();
      if (!operandProducer || !componentOperations.contains(operandProducer)) {
        return false;
      }
      worklist.push_back(operandProducer);
    }
  }
  return terminalFold != nullptr;
}

bool hasRankNExpansionWithoutClosedAnchor(const CandidateComponent &component) {
  llvm::SmallPtrSet<Operation *, 16> componentOperations(
      component.operations.begin(), component.operations.end());
  const bool hasTerminalFold = llvm::any_of(
      component.operations, isTerminalRankNAdditiveReduce);
  return llvm::any_of(component.operations, [&](Operation *operation) {
    unsigned expansions = rankNExpandingSingletonBroadcasts(operation);
    if (expansions == 0) {
      return false;
    }
    if (expansions == 1) {
      return !isExpansionDrivenByTerminalFold(operation,
                                              componentOperations);
    }
    return !hasTerminalFold || !isAxisAlignedRankNExpansion(operation) ||
           !isExpansionDrivenByProjectedParameter(operation,
                                                  componentOperations);
  });
}

bool hasUnclosedRankOneResult(const CandidateComponent &component) {
  llvm::SmallPtrSet<Operation *, 16> componentOperations(
      component.operations.begin(), component.operations.end());
  return llvm::any_of(component.operations, [&](Operation *operation) {
    return llvm::any_of(operation->getResults(), [&](Value result) {
      auto type = dyn_cast<RankedTensorType>(result.getType());
      SmallVector<Operation *> outsideUsers;
      llvm::copy_if(result.getUsers(), std::back_inserter(outsideUsers),
                    [&](Operation *user) {
                      return !componentOperations.contains(user);
                    });
      if (!type || type.getRank() != 1 ||
          !type.getElementType().isBF16() || outsideUsers.empty()) {
        return false;
      }
      return !isClosedRankOneBf16Result(operation, componentOperations) ||
             !llvm::all_of(outsideUsers,
                           [](Operation *user) {
                             return isa<func::ReturnOp>(user);
                           });
    });
  });
}

SmallVector<CandidateComponent> candidateComponents(ModuleOp module) {
  SmallVector<CandidateComponent> components;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    for (Block &block : function.getBody()) {
      SmallVector<Operation *> interval;
      auto flushInterval = [&]() {
        if (interval.empty()) {
          return;
        }
        for (CandidateComponent &component :
             partitionSupportedInterval(interval)) {
          if (!hasRankNExpansionWithoutClosedAnchor(component) &&
              !hasUnclosedRankOneResult(component)) {
            components.push_back(std::move(component));
          }
        }
        interval.clear();
      };
      for (Operation &operation : block) {
        if (isSupportedStablehlo(&operation)) {
          interval.push_back(&operation);
        } else {
          flushInterval();
        }
      }
      flushInterval();
    }
  }
  return components;
}

ArrayAttr operationSourceRefs(Operation *operation) {
  return sourceRefs(operation);
}

DictionaryAttr zeroResultRecord(Operation *operation) {
  MLIRContext *context = operation->getContext();
  SmallVector<Attribute> operands;
  operands.reserve(operation->getNumOperands());
  for (Value operand : operation->getOperands()) {
    Attribute anchor = valueAnchor(operand);
    if (!anchor) {
      return {};
    }
    operands.push_back(anchor);
  }
  NamedAttribute fields[] = {
      NamedAttribute(
          StringAttr::get(context, "classification"),
          StringAttr::get(context, operation->hasTrait<OpTrait::IsTerminator>()
                                       ? "terminator"
                                       : "zero_result_operation")),
      NamedAttribute(StringAttr::get(context, "fingerprint"),
                     normalizedOperationFingerprint(operation)),
      NamedAttribute(StringAttr::get(context, "operation_ref"),
                     operation->getAttr(kOperationRefAttribute)),
      NamedAttribute(StringAttr::get(context, "operands"),
                     ArrayAttr::get(context, operands)),
  };
  return DictionaryAttr::get(context, fields);
}

FailureOr<DictionaryAttr>
buildCoverageManifest(ModuleOp module, ArrayRef<CandidateComponent> components,
                      NumericalPolicy numerics, StringRef canonicalOptions,
                      StringRef canonicalTuning) {
  MLIRContext *context = module.getContext();
  llvm::SmallDenseSet<Attribute> selected;
  SmallVector<Attribute> selectedRegions;
  for (const CandidateComponent &component : components) {
    SmallVector<Attribute> refs;
    for (Operation *operation : component.operations) {
      LogicalResult nested =
          walkSourcePreorder(operation, [&](Operation *sourceOperation) {
            ArrayAttr operationRefs = operationSourceRefs(sourceOperation);
            if (sourceOperation->getNumResults() != 0 && !operationRefs) {
              sourceOperation->emitOpError(
                  "is missing structural source references");
              return failure();
            }
            if (operationRefs) {
              for (Attribute ref : operationRefs) {
                if (!selected.insert(ref).second) {
                  module.emitError("a source result belongs to two regions");
                  return failure();
                }
                refs.push_back(ref);
              }
            }
            return success();
          });
      if (failed(nested)) {
        return failure();
      }
    }
    selectedRegions.push_back(ArrayAttr::get(context, refs));
  }

  SmallVector<Attribute> complete;
  SmallVector<Attribute> excluded;
  SmallVector<Attribute> zeroResultOperations;
  SmallVector<Attribute> functionResults;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto functionOrdinal =
        function->getAttrOfType<IntegerAttr>(kFunctionOrdinalAttribute);
    LogicalResult sourceWalk =
        walkSourcePreorder(function.getBody(), [&](Operation *operation) {
          if (operation->getNumResults() == 0) {
            DictionaryAttr record = zeroResultRecord(operation);
            if (!record) {
              operation->emitOpError(
                  "has an operand without a structural source anchor");
              return failure();
            }
            zeroResultOperations.push_back(record);
            return success();
          }
          ArrayAttr operationRefs = operationSourceRefs(operation);
          if (!operationRefs) {
            operation->emitOpError("is missing structural source references");
            return failure();
          }
          SmallVector<Attribute> operandAnchors;
          for (Value operand : operation->getOperands()) {
            Attribute anchor = valueAnchor(operand);
            if (!anchor) {
              operation->emitOpError(
                  "has an operand without a structural source anchor");
              return failure();
            }
            operandAnchors.push_back(anchor);
          }
          for (Attribute ref : operationRefs) {
            complete.push_back(ref);
            if (selected.contains(ref)) {
              continue;
            }
            NamedAttribute fields[] = {
                NamedAttribute(StringAttr::get(context, "fingerprint"),
                               normalizedOperationFingerprint(operation)),
                NamedAttribute(StringAttr::get(context, "operands"),
                               ArrayAttr::get(context, operandAnchors)),
                NamedAttribute(StringAttr::get(context, "source"), ref),
                NamedAttribute(
                    StringAttr::get(context, "reason"),
                    StringAttr::get(
                        context,
                        operation->getParentOp() &&
                                !isa<func::FuncOp>(operation->getParentOp()) &&
                                operation->getParentOp()->getNumRegions() != 0
                            ? "enclosing_region_excluded"
                            : "unsupported_operation")),
            };
            excluded.push_back(DictionaryAttr::get(context, fields));
          }
          return success();
        });
    if (failed(sourceWalk)) {
      return failure();
    }
    auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().front().back());
    if (!returnOp) {
      function.emitOpError(
          "the first offline slice requires a direct func.return");
      return failure();
    }
    for (auto [ordinal, operand] : llvm::enumerate(returnOp.getOperands())) {
      Attribute anchor = valueAnchor(operand);
      if (!anchor) {
        returnOp.emitOpError("has a result without a structural source anchor");
        return failure();
      }
      NamedAttribute fields[] = {
          NamedAttribute(StringAttr::get(context, "function"), functionOrdinal),
          NamedAttribute(
              StringAttr::get(context, "result"),
              IntegerAttr::get(IntegerType::get(context, 64), ordinal)),
          NamedAttribute(StringAttr::get(context, "anchor"), anchor),
      };
      functionResults.push_back(DictionaryAttr::get(context, fields));
    }
  }

  std::string tuningDigest = sha256(canonicalTuning);
  NamedAttribute fields[] = {
      NamedAttribute(StringAttr::get(context, "version"),
                     IntegerAttr::get(IntegerType::get(context, 64), 2)),
      NamedAttribute(StringAttr::get(context, "policy"),
                     StringAttr::get(context, policyName(numerics))),
      NamedAttribute(StringAttr::get(context, "policy_digest"),
                     StringAttr::get(context, sha256(canonicalOptions))),
      NamedAttribute(StringAttr::get(context, "canonical_options"),
                     StringAttr::get(context, canonicalOptions)),
      NamedAttribute(StringAttr::get(context, "canonical_tuning"),
                     StringAttr::get(context, canonicalTuning)),
      NamedAttribute(StringAttr::get(context, kManifestComplete),
                     ArrayAttr::get(context, complete)),
      NamedAttribute(StringAttr::get(context, kManifestSelectedRegions),
                     ArrayAttr::get(context, selectedRegions)),
      NamedAttribute(StringAttr::get(context, kManifestExcluded),
                     ArrayAttr::get(context, excluded)),
      NamedAttribute(StringAttr::get(context, kManifestZeroResultOperations),
                     ArrayAttr::get(context, zeroResultOperations)),
      NamedAttribute(StringAttr::get(context, kManifestFunctionResults),
                     ArrayAttr::get(context, functionResults)),
      NamedAttribute(StringAttr::get(context, "tuning_digest"),
                     StringAttr::get(context, tuningDigest)),
  };
  return DictionaryAttr::get(context, fields);
}

void annotateRegion(Region &region, uint64_t functionOrdinal,
                    uint64_t &nextBlockOrdinal) {
  for (Block &block : region) {
    const uint64_t blockOrdinal = nextBlockOrdinal++;
    uint64_t operationOrdinal = 0;
    for (Operation &operation : block) {
      Dialect *dialect = operation.getDialect();
      if (dialect == nullptr ||
          dialect->getNamespace() != ShuttleDialect::getDialectNamespace()) {
        operation.setAttr(
            kOperationRefAttribute,
            DenseI64ArrayAttr::get(operation.getContext(),
                                   {static_cast<int64_t>(functionOrdinal),
                                    static_cast<int64_t>(blockOrdinal),
                                    static_cast<int64_t>(operationOrdinal)}));
        SmallVector<Attribute> sourceRefs;
        sourceRefs.reserve(operation.getNumResults());
        for (uint64_t resultOrdinal = 0;
             resultOrdinal < operation.getNumResults(); ++resultOrdinal) {
          sourceRefs.push_back(SourceRefAttr::get(
              operation.getContext(), functionOrdinal, blockOrdinal,
              operationOrdinal, resultOrdinal));
        }
        if (!sourceRefs.empty()) {
          operation.setAttr(kSourceRefsAttribute,
                            ArrayAttr::get(operation.getContext(), sourceRefs));
        }
      }
      for (Region &nested : operation.getRegions()) {
        annotateRegion(nested, functionOrdinal, nextBlockOrdinal);
      }
      ++operationOrdinal;
    }
  }
}

struct AnnotateSourcePass
    : impl::ShuttleAnnotateSourcePassBase<AnnotateSourcePass> {
  void runOnOperation() override {
    uint64_t functionOrdinal = 0;
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      function->setAttr(
          kFunctionOrdinalAttribute,
          IntegerAttr::get(IntegerType::get(function.getContext(), 64),
                           functionOrdinal));
      uint64_t nextBlockOrdinal = 0;
      annotateRegion(function.getBody(), functionOrdinal++, nextBlockOrdinal);
    }
  }
};

struct FormStructuralRegionsPass
    : impl::ShuttleFormStructuralRegionsPassBase<FormStructuralRegionsPass> {
  FormStructuralRegionsPass() = default;
  FormStructuralRegionsPass(NumericalPolicy numerics,
                            std::string canonicalOptions,
                            std::string canonicalTuning)
      : numerics(numerics), canonicalOptions(std::move(canonicalOptions)),
        canonicalTuning(std::move(canonicalTuning)) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(kCoverageManifestAttribute)) {
      module.emitError("already has a Shuttle coverage manifest");
      signalPassFailure();
      return;
    }
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (!function->hasAttr(kFunctionOrdinalAttribute)) {
        function.emitOpError(
            "requires shuttle-annotate-source before region formation");
        signalPassFailure();
        return;
      }
      if (!llvm::hasSingleElement(function.getBody())) {
        function.emitOpError(
            "the first offline slice requires one function block");
        signalPassFailure();
        return;
      }
    }

    SmallVector<CandidateComponent> components = candidateComponents(module);
    FailureOr<DictionaryAttr> manifest = buildCoverageManifest(
        module, components, numerics, canonicalOptions, canonicalTuning);
    if (failed(manifest)) {
      signalPassFailure();
      return;
    }
    module->setAttr(kCoverageManifestAttribute, *manifest);

    for (const CandidateComponent &component : components) {
      if (failed(materializeRegion(component))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult materializeRegion(const CandidateComponent &component) {
    assert(!component.operations.empty());
    MLIRContext *context = component.operations.front()->getContext();
    llvm::SmallPtrSet<Operation *, 16> members(component.operations.begin(),
                                               component.operations.end());
    llvm::SmallDenseSet<Value> seenInputs;
    SmallVector<Value> inputs;
    for (Operation *operation : component.operations) {
      for (Value operand : operation->getOperands()) {
        if (!members.contains(operand.getDefiningOp()) &&
            seenInputs.insert(operand).second) {
          inputs.push_back(operand);
        }
      }
    }

    SmallVector<Value> outputs;
    for (Operation *operation : component.operations) {
      for (Value result : operation->getResults()) {
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !members.contains(user);
            })) {
          outputs.push_back(result);
        }
      }
    }

    SmallVector<Attribute> declaredSources;
    for (Operation *operation : component.operations) {
      LogicalResult nested =
          walkSourcePreorder(operation, [&](Operation *sourceOperation) {
            ArrayAttr refs = operationSourceRefs(sourceOperation);
            if (sourceOperation->getNumResults() != 0 && !refs) {
              sourceOperation->emitOpError(
                  "is missing structural source references");
              return failure();
            }
            if (refs) {
              llvm::append_range(declaredSources, refs);
              sourceOperation->setAttr(kSelectedAttribute,
                                       UnitAttr::get(context));
            }
            return success();
          });
      if (failed(nested)) {
        return failure();
      }
    }
    SmallVector<Attribute> outputSources;
    for (Value output : outputs) {
      Attribute anchor = valueAnchor(output);
      if (!anchor || !isa<SourceRefAttr>(anchor)) {
        return component.operations.front()->emitOpError(
            "has a live-out without a source-result anchor");
      }
      outputSources.push_back(anchor);
    }

    OpBuilder builder(component.operations.front());
    OperationState state(component.operations.front()->getLoc(),
                         RegionOp::getOperationName());
    state.addOperands(inputs);
    state.addTypes(TypeRange(outputs));
    state.addAttribute("policy", NumericalPolicyAttr::get(context, numerics));
    state.addAttribute("source_refs", ArrayAttr::get(context, declaredSources));
    state.addAttribute(kRegionResultSourcesAttribute,
                       ArrayAttr::get(context, outputSources));
    state.addRegion();
    auto region = cast<RegionOp>(builder.create(state));
    Block *body = new Block();
    region.getBody().push_back(body);
    for (Value input : inputs) {
      body->addArgument(input.getType(), input.getLoc());
    }

    for (auto [input, argument] :
         llvm::zip_equal(inputs, body->getArguments())) {
      for (Operation *operation : component.operations) {
        for (OpOperand &operand : operation->getOpOperands()) {
          if (operand.get() == input) {
            operand.set(argument);
          }
        }
      }
    }
    for (Operation *operation : component.operations) {
      operation->moveBefore(body, body->end());
    }

    builder.setInsertionPointToEnd(body);
    OperationState yieldState(region.getLoc(), YieldOp::getOperationName());
    yieldState.addOperands(outputs);
    builder.create(yieldState);

    for (auto [output, replacement] :
         llvm::zip_equal(outputs, region.getResults())) {
      output.replaceUsesWithIf(replacement, [&](OpOperand &use) {
        return !region->isAncestor(use.getOwner());
      });
    }
    return success();
  }

  NumericalPolicy numerics = NumericalPolicy::SourceOrdered;
  std::string canonicalOptions = ShuttlePipelineOptions{}.canonicalOptions;
  std::string canonicalTuning = ShuttlePipelineOptions{}.canonicalTuning;
};

ArrayAttr affineMapsAttr(MLIRContext *context, ArrayRef<AffineMap> maps) {
  SmallVector<Attribute> attributes;
  attributes.reserve(maps.size());
  for (AffineMap map : maps) {
    attributes.push_back(AffineMapAttr::get(map));
  }
  return ArrayAttr::get(context, attributes);
}

FailureOr<SmallVector<AffineMap>> dotIndexingMaps(Operation *operation) {
  MLIRContext *context = operation->getContext();
  auto dimensions =
      operation->getAttrOfType<stablehlo::DotDimensionNumbersAttr>(
          "dot_dimension_numbers");
  auto lhsType = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto rhsType = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
  auto resultType =
      dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!dimensions || !lhsType || !rhsType || !resultType ||
      dimensions.getLhsBatchingDimensions().size() !=
          dimensions.getRhsBatchingDimensions().size() ||
      dimensions.getLhsContractingDimensions().size() !=
          dimensions.getRhsContractingDimensions().size()) {
    operation->emitOpError("has unsupported dot dimension metadata");
    return failure();
  }

  SmallVector<int64_t> lhsDomain(lhsType.getRank(), -1);
  SmallVector<int64_t> rhsDomain(rhsType.getRank(), -1);
  int64_t nextDomain = 0;
  for (auto [lhs, rhs] :
       llvm::zip_equal(dimensions.getLhsBatchingDimensions(),
                       dimensions.getRhsBatchingDimensions())) {
    lhsDomain[lhs] = nextDomain;
    rhsDomain[rhs] = nextDomain++;
  }
  for (int64_t dimension = 0; dimension < lhsType.getRank(); ++dimension) {
    if (lhsDomain[dimension] < 0 &&
        !llvm::is_contained(dimensions.getLhsContractingDimensions(),
                            dimension)) {
      lhsDomain[dimension] = nextDomain++;
    }
  }
  for (int64_t dimension = 0; dimension < rhsType.getRank(); ++dimension) {
    if (rhsDomain[dimension] < 0 &&
        !llvm::is_contained(dimensions.getRhsContractingDimensions(),
                            dimension)) {
      rhsDomain[dimension] = nextDomain++;
    }
  }
  const int64_t parallelDimensions = nextDomain;
  for (auto [lhs, rhs] :
       llvm::zip_equal(dimensions.getLhsContractingDimensions(),
                       dimensions.getRhsContractingDimensions())) {
    lhsDomain[lhs] = nextDomain;
    rhsDomain[rhs] = nextDomain++;
  }
  if (llvm::is_contained(lhsDomain, -1) || llvm::is_contained(rhsDomain, -1) ||
      resultType.getRank() != parallelDimensions) {
    operation->emitOpError("has an incomplete normalized dot domain");
    return failure();
  }

  auto mapFor = [&](ArrayRef<int64_t> domainPositions) {
    SmallVector<AffineExpr> expressions;
    expressions.reserve(domainPositions.size());
    for (int64_t position : domainPositions) {
      expressions.push_back(getAffineDimExpr(position, context));
    }
    return AffineMap::get(nextDomain, 0, expressions, context);
  };
  SmallVector<int64_t> resultDomain(parallelDimensions);
  for (int64_t position = 0; position < parallelDimensions; ++position) {
    resultDomain[position] = position;
  }
  return SmallVector<AffineMap>{mapFor(lhsDomain), mapFor(rhsDomain),
                                mapFor(resultDomain)};
}

FailureOr<SmallVector<AffineMap>> mapIndexingMaps(Operation *operation) {
  MLIRContext *context = operation->getContext();
  auto resultType =
      dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!resultType) {
    operation->emitOpError("requires a ranked map result");
    return failure();
  }
  int64_t rank = resultType.getRank();
  AffineMap identity = AffineMap::getMultiDimIdentityMap(rank, context);
  if (operation->getName().getStringRef() ==
      stablehlo::ReshapeOp::getOperationName()) {
    auto inputType = cast<RankedTensorType>(operation->getOperand(0).getType());
    SmallVector<unsigned> resultNonSingleton;
    for (auto [position, extent] : llvm::enumerate(resultType.getShape())) {
      if (extent != 1) {
        resultNonSingleton.push_back(position);
      }
    }
    SmallVector<AffineExpr> inputExpressions;
    unsigned nextNonSingleton = 0;
    for (int64_t extent : inputType.getShape()) {
      if (extent == 1) {
        inputExpressions.push_back(getAffineConstantExpr(0, context));
        continue;
      }
      if (nextNonSingleton >= resultNonSingleton.size()) {
        operation->emitOpError("has incompatible singleton reshape shapes");
        return failure();
      }
      inputExpressions.push_back(
          getAffineDimExpr(resultNonSingleton[nextNonSingleton++], context));
    }
    if (nextNonSingleton != resultNonSingleton.size()) {
      operation->emitOpError("has incompatible singleton reshape shapes");
      return failure();
    }
    return SmallVector<AffineMap>{
        AffineMap::get(rank, 0, inputExpressions, context), identity};
  }
  if (operation->getName().getStringRef() ==
      stablehlo::BroadcastInDimOp::getOperationName()) {
    auto dimensions =
        operation->getAttrOfType<DenseI64ArrayAttr>("broadcast_dimensions");
    if (!dimensions) {
      operation->emitOpError("has no broadcast dimension metadata");
      return failure();
    }
    SmallVector<AffineExpr> inputExpressions;
    if (!hasValidBroadcastDimensions(operation, dimensions)) {
      operation->emitOpError("requires unique in-range broadcast dimensions");
      return failure();
    }
    for (int64_t dimension : dimensions.asArrayRef()) {
      if (dimension < 0 || dimension >= rank) {
        operation->emitOpError("has an out-of-range broadcast dimension");
        return failure();
      }
      AffineExpr expression = getAffineDimExpr(dimension, context);
      auto inputType =
          cast<RankedTensorType>(operation->getOperand(0).getType());
      if (inputType.getDimSize(inputExpressions.size()) == 1 &&
          resultType.getDimSize(dimension) > 1) {
        expression = expression.floorDiv(resultType.getDimSize(dimension));
      }
      inputExpressions.push_back(expression);
    }
    return SmallVector<AffineMap>{
        AffineMap::get(rank, 0, inputExpressions, context), identity};
  }
  if (operation->getName().getStringRef() !=
      stablehlo::TransposeOp::getOperationName()) {
    SmallVector<AffineMap> maps(operation->getNumOperands() + 1, identity);
    return maps;
  }

  auto permutation = operation->getAttrOfType<DenseI64ArrayAttr>("permutation");
  auto inputType =
      dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  if (!permutation || !inputType || inputType.getRank() != rank) {
    operation->emitOpError("has an invalid transpose permutation");
    return failure();
  }
  SmallVector<int64_t> inverse(rank, -1);
  for (auto [outputDimension, inputDimension] :
       llvm::enumerate(permutation.asArrayRef())) {
    if (inputDimension < 0 || inputDimension >= rank ||
        inverse[inputDimension] >= 0) {
      operation->emitOpError("has a non-permutation transpose map");
      return failure();
    }
    inverse[inputDimension] = outputDimension;
  }
  SmallVector<AffineExpr> inputExpressions;
  for (int64_t outputDimension : inverse) {
    inputExpressions.push_back(getAffineDimExpr(outputDimension, context));
  }
  return SmallVector<AffineMap>{
      AffineMap::get(rank, 0, inputExpressions, context), identity};
}

Operation *createScalarOperation(OpBuilder &builder, Operation *operation,
                                 ValueRange arguments) {
  Location location = operation->getLoc();
  StringRef stablehloName = operation->getName().getStringRef();
  StringRef scalarName;
  if (stablehloName == stablehlo::TanhOp::getOperationName()) {
    scalarName = math::TanhOp::getOperationName();
  } else if (stablehloName == stablehlo::ExpOp::getOperationName()) {
    scalarName = math::ExpOp::getOperationName();
  } else if (stablehloName == stablehlo::RsqrtOp::getOperationName()) {
    scalarName = math::RsqrtOp::getOperationName();
  } else if (stablehloName == stablehlo::NegOp::getOperationName()) {
    scalarName = arith::NegFOp::getOperationName();
  } else if (stablehloName == stablehlo::MulOp::getOperationName()) {
    scalarName = arith::MulFOp::getOperationName();
  } else if (stablehloName == stablehlo::AddOp::getOperationName()) {
    scalarName = arith::AddFOp::getOperationName();
  } else if (stablehloName == stablehlo::SubtractOp::getOperationName()) {
    scalarName = arith::SubFOp::getOperationName();
  } else if (stablehloName == stablehlo::DivOp::getOperationName()) {
    scalarName = arith::DivFOp::getOperationName();
  } else if (stablehloName == stablehlo::ConvertOp::getOperationName()) {
    OperationState state(location, ScalarConvertOp::getOperationName());
    state.addOperands(arguments);
    Type input = cast<RankedTensorType>(operation->getOperand(0).getType())
                     .getElementType();
    Type result = cast<RankedTensorType>(operation->getResult(0).getType())
                      .getElementType();
    ScalarConvertSemantics semantics =
        input.isF32() && result.isBF16()
            ? ScalarConvertSemantics::RoundNearestEven
            : ScalarConvertSemantics::Exact;
    state.addTypes(result);
    state.addAttribute("semantics", ScalarConvertSemanticsAttr::get(
                                        operation->getContext(), semantics));
    return builder.create(state);
  } else if (stablehloName == stablehlo::ConstantOp::getOperationName()) {
    auto value = operation->getAttrOfType<DenseElementsAttr>("value");
    if (!value || !value.isSplat()) {
      return nullptr;
    }
    OperationState state(location, arith::ConstantOp::getOperationName());
    Type result = cast<RankedTensorType>(operation->getResult(0).getType())
                      .getElementType();
    state.addTypes(result);
    state.addAttribute("value", value.getSplatValue<Attribute>());
    return builder.create(state);
  } else {
    return nullptr;
  }
  OperationState state(location, scalarName);
  state.addOperands(arguments);
  state.addTypes(cast<RankedTensorType>(operation->getResult(0).getType())
                     .getElementType());
  return builder.create(state);
}

LogicalResult convertMapOperation(Operation *operation) {
  FailureOr<SmallVector<AffineMap>> maps = mapIndexingMaps(operation);
  if (failed(maps)) {
    return failure();
  }
  SourceRefAttr source = singleSourceRef(operation);
  if (!source) {
    return operation->emitOpError("requires exactly one source result");
  }

  StringRef stablehloName = operation->getName().getStringRef();
  OpBuilder builder(operation);
  OperationState state(operation->getLoc(), MapOp::getOperationName());
  state.addOperands(operation->getOperands());
  state.addTypes(operation->getResultTypes());
  state.addAttribute("indexing_maps",
                     affineMapsAttr(operation->getContext(), *maps));
  MapSemantics semantics =
      llvm::StringSwitch<MapSemantics>(stablehloName)
          .Case(stablehlo::BroadcastInDimOp::getOperationName(),
                MapSemantics::BroadcastInDim)
          .Case(stablehlo::ReshapeOp::getOperationName(), MapSemantics::Reshape)
          .Case(stablehlo::TransposeOp::getOperationName(),
                MapSemantics::Transpose)
          .Default(MapSemantics::Pointwise);
  state.addAttribute("semantics",
                     MapSemanticsAttr::get(operation->getContext(), semantics));
  state.addAttribute("source", source);
  state.addRegion();
  auto map = cast<MapOp>(builder.create(state));

  Block *body = new Block();
  map.getBody().push_back(body);
  for (Value input : operation->getOperands()) {
    auto inputType = cast<RankedTensorType>(input.getType());
    body->addArgument(inputType.getElementType(), input.getLoc());
  }
  builder.setInsertionPointToEnd(body);
  Value scalarResult;
  if (stablehloName == stablehlo::TransposeOp::getOperationName() ||
      stablehloName == stablehlo::BroadcastInDimOp::getOperationName() ||
      stablehloName == stablehlo::ReshapeOp::getOperationName()) {
    scalarResult = body->getArgument(0);
  } else {
    Operation *scalar =
        createScalarOperation(builder, operation, body->getArguments());
    if (!scalar) {
      return operation->emitOpError("has no scalar Map lowering");
    }
    scalarResult = scalar->getResult(0);
  }
  OperationState yieldState(operation->getLoc(), YieldOp::getOperationName());
  yieldState.addOperands(scalarResult);
  builder.create(yieldState);

  operation->getResult(0).replaceAllUsesWith(map.getResult(0));
  operation->erase();
  return success();
}

LogicalResult convertContractOperation(Operation *operation) {
  FailureOr<SmallVector<AffineMap>> maps = dotIndexingMaps(operation);
  if (failed(maps)) {
    return failure();
  }
  SourceRefAttr source = singleSourceRef(operation);
  if (!source) {
    return operation->emitOpError("requires exactly one source result");
  }
  const unsigned domainRank = maps->front().getNumDims();
  const unsigned parallelRank = maps->back().getNumResults();
  OpBuilder builder(operation);
  SmallVector<Attribute> iteratorKinds;
  for (unsigned dimension = 0; dimension < domainRank; ++dimension) {
    iteratorKinds.push_back(builder.getStringAttr(
        dimension < parallelRank ? "parallel" : "reduction"));
  }
  SmallVector<Attribute> precision(operation->getNumOperands(),
                                   builder.getStringAttr("DEFAULT"));
  auto resultType = cast<RankedTensorType>(operation->getResult(0).getType());

  OperationState state(operation->getLoc(), ContractOp::getOperationName());
  state.addOperands(operation->getOperands());
  state.addTypes(operation->getResultTypes());
  state.addAttribute("indexing_maps",
                     affineMapsAttr(operation->getContext(), *maps));
  state.addAttribute("iterator_kinds",
                     ArrayAttr::get(operation->getContext(), iteratorKinds));
  state.addAttribute(
      "accumulator_types",
      ArrayAttr::get(operation->getContext(),
                     {TypeAttr::get(resultType.getElementType())}));
  state.addAttribute("precision",
                     ArrayAttr::get(operation->getContext(), precision));
  state.addAttribute("algorithm", builder.getStringAttr("dot_general"));
  state.addAttribute("source", source);
  Operation *contract = builder.create(state);
  operation->getResult(0).replaceAllUsesWith(contract->getResult(0));
  operation->erase();
  return success();
}

LogicalResult convertReduceOperation(Operation *operation) {
  if (!isSupportedStablehloReduce(operation)) {
    return operation->emitOpError(
        "is outside the first Fold conversion contract");
  }
  SourceRefAttr source = singleSourceRef(operation);
  auto ownerRef =
      operation->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  if (!source || !ownerRef) {
    return operation->emitOpError(
        "requires one result source and one owner operation reference");
  }
  Block &sourceBody = operation->getRegion(0).front();
  Operation &sourceAdd = sourceBody.front();
  Operation &sourceReturn = sourceBody.back();
  ArrayAttr addRefs = sourceRefs(&sourceAdd);
  auto addOwner =
      sourceAdd.getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  auto returnOwner =
      sourceReturn.getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  if (!addRefs || addRefs.size() != 1 || !addOwner || !returnOwner) {
    return operation->emitOpError("has incomplete nested reducer provenance");
  }

  OpBuilder builder(operation);
  OperationState state(operation->getLoc(), FoldOp::getOperationName());
  state.addOperands(operation->getOperands());
  state.addTypes(operation->getResultTypes());
  state.addAttribute("operandSegmentSizes",
                     DenseI32ArrayAttr::get(operation->getContext(), {1, 1}));
  state.addAttribute("reduction_dimensions", operation->getAttr("dimensions"));
  state.addAttribute("accumulator_types",
                     ArrayAttr::get(operation->getContext(),
                                    {TypeAttr::get(builder.getF32Type())}));
  state.addAttribute("order_free", builder.getBoolAttr(true));
  state.addAttribute("source", source);
  state.addAttribute(kOperationRefAttribute, ownerRef);
  state.addRegion();
  auto fold = cast<FoldOp>(builder.create(state));

  Block *body = new Block();
  fold.getCombiner().push_back(body);
  body->addArgument(builder.getF32Type(), sourceBody.getArgument(0).getLoc());
  body->addArgument(builder.getF32Type(), sourceBody.getArgument(1).getLoc());
  builder.setInsertionPointToEnd(body);
  OperationState addState(sourceAdd.getLoc(),
                          arith::AddFOp::getOperationName());
  addState.addOperands(body->getArguments());
  addState.addTypes(builder.getF32Type());
  addState.addAttribute(kSourceRefsAttribute, addRefs);
  addState.addAttribute(kOperationRefAttribute, addOwner);
  Operation *add = builder.create(addState);
  OperationState yieldState(sourceReturn.getLoc(), YieldOp::getOperationName());
  yieldState.addOperands(add->getResult(0));
  yieldState.addAttribute(kOperationRefAttribute, returnOwner);
  builder.create(yieldState);

  operation->getResult(0).replaceAllUsesWith(fold.getResult(0));
  operation->erase();
  return success();
}

struct ConvertStablehloToAlgebraPass
    : impl::ShuttleConvertStablehloToAlgebraPassBase<
          ConvertStablehloToAlgebraPass> {
  void runOnOperation() override {
    if (!getOperation()->hasAttr(kCoverageManifestAttribute)) {
      getOperation().emitError(
          "requires shuttle-form-structural-regions before conversion");
      signalPassFailure();
      return;
    }
    SmallVector<RegionOp> regions;
    getOperation().walk([&](RegionOp region) { regions.push_back(region); });
    for (RegionOp region : regions) {
      SmallVector<Operation *> sourceOperations;
      for (Operation &operation : region.getBody().front()) {
        if (!isa<YieldOp>(operation)) {
          sourceOperations.push_back(&operation);
        }
      }
      for (Operation *operation : sourceOperations) {
        StringRef name = operation->getName().getStringRef();
        LogicalResult result =
            name == stablehlo::ReduceOp::getOperationName()
                ? convertReduceOperation(operation)
            : name == stablehlo::DotGeneralOp::getOperationName()
                ? convertContractOperation(operation)
                : convertMapOperation(operation);
        if (failed(result)) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

LogicalResult verifyRegionLocalCoverage(ModuleOp module) {
  WalkResult result = module.walk([&](RegionOp region) {
    llvm::SmallDenseSet<Attribute> declaredSources;
    llvm::SmallDenseSet<Attribute> representedSources;
    for (Attribute source : region.getSourceRefs()) {
      auto sourceRef = dyn_cast<SourceRefAttr>(source);
      if (!sourceRef || !declaredSources.insert(sourceRef).second) {
        region.emitOpError("requires unique #shuttle.source_ref entries");
        return WalkResult::interrupt();
      }
    }
    WalkResult nestedResult = region.getBody().walk([&](Operation *operation) {
      if (isa<RegionOp>(operation)) {
        return WalkResult::skip();
      }
      Attribute source;
      if (auto map = dyn_cast<MapOp>(operation)) {
        source = map.getSource();
      } else if (auto contract = dyn_cast<ContractOp>(operation)) {
        source = contract.getSource();
      } else if (auto fold = dyn_cast<FoldOp>(operation)) {
        source = fold.getSource();
      } else if (operation->getParentOfType<FoldOp>()) {
        ArrayAttr refs = sourceRefs(operation);
        if (!refs) {
          return WalkResult::advance();
        }
        for (Attribute nestedSource : refs) {
          if (!declaredSources.contains(nestedSource) ||
              !representedSources.insert(nestedSource).second) {
            operation->emitOpError("has invalid nested Fold source coverage");
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      } else {
        return WalkResult::advance();
      }
      if (!declaredSources.contains(source)) {
        operation->emitOpError(
            "source reference is absent from the enclosing shuttle.region");
        return WalkResult::interrupt();
      }
      if (!representedSources.insert(source).second) {
        operation->emitOpError("duplicates region-local source coverage");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (nestedResult.wasInterrupted()) {
      return WalkResult::interrupt();
    }
    for (Attribute declaredSource : declaredSources) {
      if (!representedSources.contains(declaredSource)) {
        region.emitOpError(
            "declared source reference is not represented by algebra");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

llvm::SmallDenseSet<Attribute> attributeSet(ArrayAttr attributes) {
  llvm::SmallDenseSet<Attribute> result;
  if (attributes) {
    result.insert(attributes.begin(), attributes.end());
  }
  return result;
}

bool sameAttributeSet(const llvm::SmallDenseSet<Attribute> &left,
                      const llvm::SmallDenseSet<Attribute> &right) {
  return left.size() == right.size() &&
         llvm::all_of(left, [&](Attribute attribute) {
           return right.contains(attribute);
         });
}

FailureOr<llvm::SmallDenseSet<Attribute>>
selectedManifestSources(DictionaryAttr manifest) {
  auto groups = manifest.getAs<ArrayAttr>(kManifestSelectedRegions);
  if (!groups) {
    return failure();
  }
  llvm::SmallDenseSet<Attribute> selected;
  for (Attribute groupAttribute : groups) {
    auto group = dyn_cast<ArrayAttr>(groupAttribute);
    if (!group) {
      return failure();
    }
    for (Attribute source : group) {
      if (!isa<SourceRefAttr>(source) || !selected.insert(source).second) {
        return failure();
      }
    }
  }
  return selected;
}

FailureOr<llvm::SmallDenseSet<Attribute>>
excludedManifestSources(DictionaryAttr manifest) {
  auto records = manifest.getAs<ArrayAttr>(kManifestExcluded);
  if (!records) {
    return failure();
  }
  llvm::SmallDenseSet<Attribute> excluded;
  for (Attribute recordAttribute : records) {
    auto record = dyn_cast<DictionaryAttr>(recordAttribute);
    auto source =
        record ? record.getAs<SourceRefAttr>("source") : SourceRefAttr{};
    auto reason = record ? record.getAs<StringAttr>("reason") : StringAttr{};
    if (!source || !reason ||
        (reason.getValue() != "unsupported_operation" &&
         reason.getValue() != "enclosing_region_excluded") ||
        !excluded.insert(source).second) {
      return failure();
    }
  }
  return excluded;
}

FailureOr<ArrayAttr> currentExcludedRecords(ModuleOp module,
                                            DictionaryAttr manifest) {
  auto manifestRecords = manifest.getAs<ArrayAttr>(kManifestExcluded);
  if (!manifestRecords) {
    return failure();
  }
  llvm::SmallDenseMap<Attribute, StringAttr> reasons;
  for (Attribute recordAttribute : manifestRecords) {
    auto record = dyn_cast<DictionaryAttr>(recordAttribute);
    auto source =
        record ? record.getAs<SourceRefAttr>("source") : SourceRefAttr{};
    auto reason = record ? record.getAs<StringAttr>("reason") : StringAttr{};
    if (!source || !reason || !reasons.try_emplace(source, reason).second) {
      return failure();
    }
  }

  SmallVector<Attribute> records;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    LogicalResult walk =
        walkSourcePreorder(function.getBody(), [&](Operation *operation) {
          ArrayAttr refs = sourceRefs(operation);
          if (!refs) {
            return success();
          }
          SmallVector<Attribute> operandAnchors;
          for (Value operand : operation->getOperands()) {
            Attribute anchor = valueAnchor(operand);
            if (!anchor) {
              return failure();
            }
            operandAnchors.push_back(anchor);
          }
          for (Attribute source : refs) {
            auto reason = reasons.find(source);
            if (reason == reasons.end()) {
              continue;
            }
            NamedAttribute fields[] = {
                NamedAttribute(
                    StringAttr::get(module.getContext(), "fingerprint"),
                    normalizedOperationFingerprint(operation)),
                NamedAttribute(
                    StringAttr::get(module.getContext(), "operands"),
                    ArrayAttr::get(module.getContext(), operandAnchors)),
                NamedAttribute(StringAttr::get(module.getContext(), "source"),
                               source),
                NamedAttribute(StringAttr::get(module.getContext(), "reason"),
                               reason->second),
            };
            records.push_back(DictionaryAttr::get(module.getContext(), fields));
          }
          return success();
        });
    if (failed(walk)) {
      return failure();
    }
  }
  return ArrayAttr::get(module.getContext(), records);
}

FailureOr<ArrayAttr> currentZeroResultRecords(ModuleOp module) {
  SmallVector<Attribute> records;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    LogicalResult walk =
        walkSourcePreorder(function.getBody(), [&](Operation *operation) {
          if (operation->getNumResults() != 0 ||
              !operation->hasAttr(kOperationRefAttribute)) {
            return success();
          }
          DictionaryAttr record = zeroResultRecord(operation);
          if (!record) {
            return failure();
          }
          records.push_back(record);
          return success();
        });
    if (failed(walk)) {
      return failure();
    }
  }
  return ArrayAttr::get(module.getContext(), records);
}

FailureOr<ArrayAttr> currentFunctionResultRecords(ModuleOp module) {
  SmallVector<Attribute> records;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto functionOrdinal =
        function->getAttrOfType<IntegerAttr>(kFunctionOrdinalAttribute);
    if (!functionOrdinal || !llvm::hasSingleElement(function.getBody())) {
      return failure();
    }
    auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().front().back());
    if (!returnOp) {
      return failure();
    }
    for (auto [ordinal, operand] : llvm::enumerate(returnOp.getOperands())) {
      Attribute anchor = valueAnchor(operand);
      if (!anchor) {
        return failure();
      }
      NamedAttribute fields[] = {
          NamedAttribute(StringAttr::get(module.getContext(), "function"),
                         functionOrdinal),
          NamedAttribute(
              StringAttr::get(module.getContext(), "result"),
              IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                               ordinal)),
          NamedAttribute(StringAttr::get(module.getContext(), "anchor"),
                         anchor),
      };
      records.push_back(DictionaryAttr::get(module.getContext(), fields));
    }
  }
  return ArrayAttr::get(module.getContext(), records);
}

bool zeroResultRecordsMatch(ArrayAttr current, ArrayAttr expected) {
  if (!current || !expected || current.size() != expected.size()) {
    return false;
  }
  llvm::SmallDenseSet<Attribute> matchedOperationRefs;
  for (Attribute currentAttribute : current) {
    auto currentRecord = dyn_cast<DictionaryAttr>(currentAttribute);
    Attribute operationRef =
        currentRecord ? currentRecord.get("operation_ref") : Attribute{};
    auto expectedPosition = llvm::find_if(expected, [&](Attribute attribute) {
      auto record = dyn_cast<DictionaryAttr>(attribute);
      return record && record.get("operation_ref") == operationRef;
    });
    auto expectedRecord = expectedPosition == expected.end()
                              ? DictionaryAttr{}
                              : dyn_cast<DictionaryAttr>(*expectedPosition);
    if (!currentRecord || !operationRef || !expectedRecord ||
        !matchedOperationRefs.insert(operationRef).second ||
        currentRecord.get("classification") !=
            expectedRecord.get("classification") ||
        currentRecord.get("operands") != expectedRecord.get("operands")) {
      return false;
    }
    if (currentRecord.get("fingerprint") == expectedRecord.get("fingerprint")) {
      continue;
    }
    auto currentFingerprint =
        currentRecord.getAs<DictionaryAttr>("fingerprint");
    auto expectedFingerprint =
        expectedRecord.getAs<DictionaryAttr>("fingerprint");
    auto currentName = currentFingerprint
                           ? currentFingerprint.getAs<StringAttr>("name")
                           : StringAttr{};
    auto expectedName = expectedFingerprint
                            ? expectedFingerprint.getAs<StringAttr>("name")
                            : StringAttr{};
    if (!currentName || !expectedName ||
        currentName.getValue() != YieldOp::getOperationName() ||
        expectedName.getValue() != stablehlo::ReturnOp::getOperationName() ||
        currentFingerprint.size() != expectedFingerprint.size() ||
        currentFingerprint.get("attributes") !=
            expectedFingerprint.get("attributes") ||
        currentFingerprint.get("result_types") !=
            expectedFingerprint.get("result_types")) {
      return false;
    }
  }
  return true;
}

LogicalResult verifyManifestCoverage(ModuleOp module, DictionaryAttr manifest) {
  auto version = manifest.getAs<IntegerAttr>("version");
  auto completeArray = manifest.getAs<ArrayAttr>(kManifestComplete);
  auto selectedGroups = manifest.getAs<ArrayAttr>(kManifestSelectedRegions);
  auto policy = manifest.getAs<StringAttr>("policy");
  auto policyDigest = manifest.getAs<StringAttr>("policy_digest");
  auto canonicalOptions = manifest.getAs<StringAttr>("canonical_options");
  auto canonicalTuning = manifest.getAs<StringAttr>("canonical_tuning");
  auto tuningDigest = manifest.getAs<StringAttr>("tuning_digest");
  FailureOr<llvm::SmallDenseSet<Attribute>> selected =
      selectedManifestSources(manifest);
  FailureOr<llvm::SmallDenseSet<Attribute>> excluded =
      excludedManifestSources(manifest);
  if (!version || version.getInt() != 2 || !completeArray || !selectedGroups ||
      !policy || !policyDigest || !canonicalOptions || !canonicalTuning ||
      !tuningDigest || failed(selected) || failed(excluded)) {
    return module.emitError("has a malformed Shuttle coverage manifest");
  }
  std::string roundTripPrefix =
      "{\"execution_mode\":\"stablehlo_round_trip\",\"numerics\":\"";
  roundTripPrefix.append(policy.getValue().data(), policy.getValue().size());
  roundTripPrefix.push_back('"');
  std::string gpuPrefix =
      "{\"execution_mode\":\"gpu_executable_bundle\",\"numerics\":\"";
  gpuPrefix.append(policy.getValue().data(), policy.getValue().size());
  gpuPrefix.push_back('"');
  const bool validOptionsPrefix =
      canonicalOptions.getValue().starts_with(roundTripPrefix) ||
      canonicalOptions.getValue().starts_with(gpuPrefix);
  if ((policy.getValue() != "source_ordered" && policy.getValue() != "fast") ||
      tuningDigest.getValue().size() != 64 ||
      !llvm::all_of(tuningDigest.getValue(), llvm::isHexDigit) ||
      tuningDigest.getValue() != sha256(canonicalTuning.getValue()) ||
      policyDigest.getValue() != sha256(canonicalOptions.getValue()) ||
      !validOptionsPrefix) {
    return module.emitError("has inconsistent Shuttle policy digests");
  }
  llvm::SmallDenseSet<Attribute> complete = attributeSet(completeArray);
  if (complete.size() != completeArray.size()) {
    return module.emitError(
        "coverage manifest contains duplicate source results");
  }
  for (Attribute source : *selected) {
    if (excluded->contains(source)) {
      return module.emitError(
          "coverage manifest selects and excludes one source result");
    }
  }
  llvm::SmallDenseSet<Attribute> partition = *selected;
  partition.insert(excluded->begin(), excluded->end());
  if (!sameAttributeSet(partition, complete)) {
    return module.emitError(
        "coverage manifest is not a total source-result partition");
  }

  llvm::SmallDenseSet<Attribute> operationRefs;
  WalkResult operationRefWalk = module.walk([&](Operation *operation) {
    Attribute operationRef = operation->getAttr(kOperationRefAttribute);
    ArrayAttr refs = sourceRefs(operation);
    if (refs && operation->getParentOfType<FoldOp>() && !operationRef) {
      operation->emitOpError(
          "has nested Fold sources without an operation reference");
      return WalkResult::interrupt();
    }
    if (!operationRef) {
      return WalkResult::advance();
    }
    auto denseOperationRef = dyn_cast<DenseI64ArrayAttr>(operationRef);
    if (!denseOperationRef || denseOperationRef.size() != 3 ||
        !operationRefs.insert(operationRef).second) {
      operation->emitOpError(
          "has a missing-format or duplicate operation reference");
      return WalkResult::interrupt();
    }
    if (refs && llvm::any_of(refs, [&](Attribute ref) {
          auto source = dyn_cast<SourceRefAttr>(ref);
          return !source || operationRefForSource(source) != denseOperationRef;
        })) {
      operation->emitOpError(
          "has an operation reference that differs from its source results");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (operationRefWalk.wasInterrupted()) {
    return failure();
  }

  const bool algebraStage = !module.getOps<func::FuncOp>().empty() && [&]() {
    bool found = false;
    module.walk([&](RegionOp) { found = true; });
    return found;
  }();
  if (algebraStage) {
    SmallVector<Attribute> actualGroups;
    bool policyMismatch = false;
    module.walk([&](RegionOp region) {
      actualGroups.push_back(region.getSourceRefs());
      if (policyName(region.getPolicy()) != policy.getValue()) {
        policyMismatch = true;
      }
    });
    if (policyMismatch) {
      return module.emitError(
          "structural region policy does not equal manifest policy");
    }
    if (ArrayAttr::get(module.getContext(), actualGroups) != selectedGroups) {
      return module.emitError(
          "structural regions do not equal manifest region groups");
    }
  }
  llvm::SmallDenseSet<Attribute> representedSelected;
  llvm::SmallDenseSet<Attribute> representedExcluded;
  auto recordSelected = [&](Operation *operation,
                            Attribute source) -> WalkResult {
    if (!selected->contains(source)) {
      operation->emitOpError(
          "represents a source absent from selected manifest coverage");
      return WalkResult::interrupt();
    }
    if (!representedSelected.insert(source).second) {
      operation->emitOpError("duplicates selected source coverage");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  WalkResult walk = module.walk([&](Operation *operation) {
    if (auto region = dyn_cast<RegionOp>(operation)) {
      for (Attribute source : region.getSourceRefs()) {
        if (!selected->contains(source)) {
          region.emitOpError(
              "declares a source absent from selected manifest coverage");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    }
    if (auto map = dyn_cast<MapOp>(operation)) {
      return recordSelected(operation, map.getSource());
    }
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      return recordSelected(operation, contract.getSource());
    }
    if (auto fold = dyn_cast<FoldOp>(operation)) {
      auto owner =
          fold->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
      if (!owner) {
        fold.emitOpError("requires Reduce owner operation provenance");
        return WalkResult::interrupt();
      }
      if (owner != operationRefForSource(fold.getSource())) {
        fold.emitOpError(
            "has Reduce owner provenance that differs from its source");
        return WalkResult::interrupt();
      }
      for (BlockArgument argument : fold.getCombiner().front().getArguments()) {
        auto anchor = dyn_cast_or_null<DictionaryAttr>(valueAnchor(argument));
        if (!anchor || anchor.get("owner") != owner) {
          fold.emitOpError(
              "has a combiner argument without its Reduce owner anchor");
          return WalkResult::interrupt();
        }
      }
      return recordSelected(operation, fold.getSource());
    }
    ArrayAttr refs = sourceRefs(operation);
    if (!refs) {
      return WalkResult::advance();
    }
    for (Attribute source : refs) {
      if (selected->contains(source)) {
        if (algebraStage) {
          if (!operation->getParentOfType<FoldOp>()) {
            operation->emitOpError(
                "selected source operation survived algebra conversion");
            return WalkResult::interrupt();
          }
          if (!representedSelected.insert(source).second) {
            operation->emitOpError("duplicates selected source coverage");
            return WalkResult::interrupt();
          }
          continue;
        }
        if (!representedSelected.insert(source).second) {
          operation->emitOpError("duplicates selected source coverage");
          return WalkResult::interrupt();
        }
      } else if (excluded->contains(source)) {
        if (!representedExcluded.insert(source).second) {
          operation->emitOpError("duplicates excluded source coverage");
          return WalkResult::interrupt();
        }
      } else {
        operation->emitOpError(
            "carries a source absent from the coverage manifest");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (walk.wasInterrupted()) {
    return failure();
  }
  if (!sameAttributeSet(representedSelected, *selected) ||
      !sameAttributeSet(representedExcluded, *excluded)) {
    return module.emitError(
        "represented source results do not equal manifest coverage");
  }

  FailureOr<ArrayAttr> excludedRecords =
      currentExcludedRecords(module, manifest);
  if (failed(excludedRecords) ||
      *excludedRecords != manifest.getAs<ArrayAttr>(kManifestExcluded)) {
    return module.emitError(
        "excluded operation fingerprint or operand anchors changed");
  }
  FailureOr<ArrayAttr> zeroResultRecords = currentZeroResultRecords(module);
  FailureOr<ArrayAttr> functionResultRecords =
      currentFunctionResultRecords(module);
  if (failed(zeroResultRecords) || failed(functionResultRecords) ||
      !zeroResultRecordsMatch(
          *zeroResultRecords,
          manifest.getAs<ArrayAttr>(kManifestZeroResultOperations)) ||
      *functionResultRecords !=
          manifest.getAs<ArrayAttr>(kManifestFunctionResults)) {
    return module.emitError(
        "zero-result or function-result source anchors changed");
  }
  return success();
}

struct VerifySourceCoveragePass
    : impl::ShuttleVerifySourceCoveragePassBase<VerifySourceCoveragePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto manifest =
        module->getAttrOfType<DictionaryAttr>(kCoverageManifestAttribute);
    LogicalResult result = manifest ? verifyManifestCoverage(module, manifest)
                                    : verifyRegionLocalCoverage(module);
    if (failed(result)) {
      signalPassFailure();
    }
  }
};

struct VerifySemanticErasurePass
    : impl::ShuttleVerifySemanticErasurePassBase<VerifySemanticErasurePass> {
  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (!operation->hasAttr(kSelectedAttribute)) {
        return WalkResult::advance();
      }
      operation->emitOpError(
          "selected source operation survived Shuttle conversion");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

struct ShuttleCanonicalizePass
    : impl::ShuttleCanonicalizePassBase<ShuttleCanonicalizePass> {
  void runOnOperation() override {}
};

FailureOr<unsigned> mapResultPosition(AffineMap map, unsigned domainPosition) {
  for (auto [resultPosition, expression] : llvm::enumerate(map.getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (dimension && dimension.getPosition() == domainPosition) {
      return resultPosition;
    }
  }
  return failure();
}

bool isIdentityMap(AffineMap map) {
  return map ==
         AffineMap::getMultiDimIdentityMap(map.getNumDims(), map.getContext());
}

LogicalResult verifyDefaultScalarSemantics(Operation *operation) {
  if (operation->getNumRegions() != 0 || !isMemoryEffectFree(operation)) {
    return operation->emitOpError(
        "requires one pure region-free scalar operation");
  }
  auto fastMath =
      operation->getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
  if (fastMath && fastMath.getValue() != arith::FastMathFlags::none) {
    return operation->emitOpError(
        "has fast-math semantics with no StableHLO representation");
  }
  for (NamedAttribute attribute : operation->getAttrs()) {
    if (attribute.getName().strref() != "fastmath") {
      return operation->emitOpError(
          "has scalar attributes with no StableHLO representation");
    }
  }
  return success();
}

FailureOr<OperationState> lowerMapState(MapOp map, ValueRange operands) {
  if (map.getInputs().size() != operands.size() ||
      map.getResults().size() != 1 || map.getBody().empty()) {
    map.emitOpError("is outside the first Map lowering contract");
    return failure();
  }
  Block &body = map.getBody().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1) {
    map.emitOpError("requires one scalar yield for StableHLO lowering");
    return failure();
  }
  if (!map->getDiscardableAttrs().empty()) {
    map.emitOpError("has attributes with no StableHLO Map representation");
    return failure();
  }
  if (!yield->getDiscardableAttrs().empty()) {
    yield.emitOpError("has attributes with no stablehlo.return representation");
    return failure();
  }

  SmallVector<Operation *> scalarOperations;
  for (Operation &operation : body.without_terminator()) {
    scalarOperations.push_back(&operation);
  }
  ArrayAttr indexingMaps = map.getIndexingMaps();
  if (indexingMaps.size() != operands.size() + 1) {
    map.emitOpError("has incomplete indexing maps");
    return failure();
  }

  StringRef stablehloName;
  NamedAttrList attributes;
  if (scalarOperations.empty()) {
    if (operands.size() != 1 || yield.getValues()[0] != body.getArgument(0)) {
      map.emitOpError("has no authoritative scalar operation");
      return failure();
    }
    AffineMap inputMap = cast<AffineMapAttr>(indexingMaps[0]).getValue();
    AffineMap resultMap = cast<AffineMapAttr>(indexingMaps[1]).getValue();
    if (inputMap.getNumDims() != resultMap.getNumDims() ||
        !isIdentityMap(resultMap)) {
      map.emitOpError("has an incompatible identity result map");
      return failure();
    }
    if (map.getSemantics() == MapSemantics::BroadcastInDim) {
      auto inputType = dyn_cast<RankedTensorType>(operands[0].getType());
      auto resultType = dyn_cast<RankedTensorType>(map.getResult(0).getType());
      if (!inputType || !resultType ||
          inputMap.getNumResults() != inputType.getRank() ||
          resultMap.getNumResults() != resultType.getRank() ||
          inputMap.getNumResults() > resultMap.getNumResults()) {
        map.emitOpError("has incompatible broadcast indexing maps");
        return failure();
      }
      SmallVector<int64_t> dimensions;
      llvm::SmallDenseSet<int64_t> seenDimensions;
      unsigned expandingSingletons = 0;
      for (auto [inputDimension, expression] :
           llvm::enumerate(inputMap.getResults())) {
        auto dimension = dyn_cast<AffineDimExpr>(expression);
        int64_t position =
            dimension ? static_cast<int64_t>(dimension.getPosition()) : -1;
        bool expandingSingleton = false;
        bool boundedZero = false;
        if (!dimension) {
          auto floorDiv = dyn_cast<AffineBinaryOpExpr>(expression);
          AffineExpr dividendExpression =
              floorDiv && floorDiv.getKind() == AffineExprKind::FloorDiv
                  ? floorDiv.getLHS()
                  : AffineExpr{};
          AffineExpr divisorExpression =
              floorDiv && floorDiv.getKind() == AffineExprKind::FloorDiv
                  ? floorDiv.getRHS()
                  : AffineExpr{};
          auto dividend =
              dyn_cast_if_present<AffineDimExpr>(dividendExpression);
          auto divisor =
              dyn_cast_if_present<AffineConstantExpr>(divisorExpression);
          if (!dividend || !divisor || divisor.getValue() <= 1) {
            map.emitOpError("expanded singleton broadcast dimensions require a "
                            "bounded-zero floordiv expression");
            return failure();
          }
          boundedZero = true;
          position = dividend.getPosition();
          expandingSingleton =
              inputType.getDimSize(inputDimension) == 1 &&
              position < resultType.getRank() &&
              resultType.getDimSize(position) == divisor.getValue();
        }
        if (position < 0 || position >= resultType.getRank() ||
            !seenDimensions.insert(position).second ||
            (boundedZero && !expandingSingleton) ||
            (!expandingSingleton && inputType.getDimSize(inputDimension) !=
                                        resultType.getDimSize(position))) {
          map.emitOpError(
              "broadcast input map dimensions must be unique, in range, and "
              "match direct or bounded singleton extents");
          return failure();
        }
        expandingSingletons += expandingSingleton;
        dimensions.push_back(position);
      }
      const bool supportedRankTwoExpansion =
          inputType.getRank() == 2 && resultType.getRank() == 2 &&
          expandingSingletons == 1;
      const bool supportedRankNExpansion =
          inputType.getRank() == resultType.getRank() &&
          inputType.getRank() >= 3 && inputType.getRank() <= 8 &&
          expandingSingletons > 0;
      if (expandingSingletons != 0 &&
          (dimensions.size() != static_cast<size_t>(inputType.getRank()) ||
           (!supportedRankTwoExpansion && !supportedRankNExpansion))) {
        map.emitOpError(
            "mapped singleton broadcast requires one expanded axis at rank "
            "two or one or more expanded axes at rank three through rank "
            "eight");
        return failure();
      }
      stablehloName = stablehlo::BroadcastInDimOp::getOperationName();
      attributes.append("broadcast_dimensions",
                        DenseI64ArrayAttr::get(map.getContext(), dimensions));
    } else if (map.getSemantics() == MapSemantics::Reshape) {
      auto inputType = dyn_cast<RankedTensorType>(operands[0].getType());
      auto resultType = dyn_cast<RankedTensorType>(map.getResult(0).getType());
      if (!inputType || !resultType || !inputType.hasStaticShape() ||
          !resultType.hasStaticShape()) {
        map.emitOpError("requires static singleton reshape types");
        return failure();
      }
      SmallVector<int64_t> inputNonSingleton;
      SmallVector<int64_t> resultNonSingleton;
      llvm::copy_if(inputType.getShape(), std::back_inserter(inputNonSingleton),
                    [](int64_t extent) { return extent != 1; });
      llvm::copy_if(resultType.getShape(),
                    std::back_inserter(resultNonSingleton),
                    [](int64_t extent) { return extent != 1; });
      SmallVector<unsigned> resultNonSingletonPositions;
      for (auto [position, extent] : llvm::enumerate(resultType.getShape())) {
        if (extent != 1) {
          resultNonSingletonPositions.push_back(position);
        }
      }
      if (inputNonSingleton != resultNonSingleton || inputType == resultType ||
          inputMap.getNumResults() != inputType.getRank()) {
        map.emitOpError("has incompatible singleton reshape types");
        return failure();
      }
      unsigned nextNonSingleton = 0;
      for (auto [inputDimension, expression] :
           llvm::enumerate(inputMap.getResults())) {
        if (inputType.getDimSize(inputDimension) == 1) {
          auto constant = dyn_cast<AffineConstantExpr>(expression);
          if (!constant || constant.getValue() != 0) {
            map.emitOpError(
                "singleton reshape input dimensions must use constant zero");
            return failure();
          }
          continue;
        }
        auto dimension = dyn_cast<AffineDimExpr>(expression);
        if (!dimension ||
            nextNonSingleton >= resultNonSingletonPositions.size() ||
            dimension.getPosition() !=
                resultNonSingletonPositions[nextNonSingleton++]) {
          map.emitOpError("has an ambiguous singleton reshape indexing map");
          return failure();
        }
      }
      if (nextNonSingleton != resultNonSingletonPositions.size()) {
        map.emitOpError("has an incomplete singleton reshape indexing map");
        return failure();
      }
      stablehloName = stablehlo::ReshapeOp::getOperationName();
    } else if (map.getSemantics() == MapSemantics::Transpose) {
      if (inputMap.getNumResults() != resultMap.getNumResults()) {
        map.emitOpError("has incompatible transpose indexing maps");
        return failure();
      }
      SmallVector<int64_t> permutation;
      permutation.reserve(resultMap.getNumResults());
      for (AffineExpr resultExpression : resultMap.getResults()) {
        auto resultDimension = dyn_cast<AffineDimExpr>(resultExpression);
        if (!resultDimension) {
          map.emitOpError("has a non-structural transpose result map");
          return failure();
        }
        FailureOr<unsigned> inputPosition =
            mapResultPosition(inputMap, resultDimension.getPosition());
        if (failed(inputPosition)) {
          map.emitOpError("has an incomplete transpose input map");
          return failure();
        }
        permutation.push_back(*inputPosition);
      }
      stablehloName = stablehlo::TransposeOp::getOperationName();
      attributes.append("permutation",
                        DenseI64ArrayAttr::get(map.getContext(), permutation));
    } else {
      map.emitOpError("has pointwise semantics but no scalar operation");
      return failure();
    }
  } else {
    if (map.getSemantics() != MapSemantics::Pointwise) {
      map.emitOpError("structural Map semantics require an empty scalar body");
      return failure();
    }
    if (scalarOperations.size() != 1) {
      map.emitOpError("requires exactly one representable scalar operation");
      return failure();
    }
    Operation *scalar = scalarOperations.front();
    if (yield.getValues()[0] != scalar->getResult(0) ||
        scalar->getOperands() != body.getArguments()) {
      map.emitOpError(
          "scalar operands and yield must directly match the Map boundary");
      return failure();
    }
    if (!llvm::all_of(indexingMaps, [](Attribute attribute) {
          return isIdentityMap(cast<AffineMapAttr>(attribute).getValue());
        })) {
      map.emitOpError(
          "pointwise StableHLO lowering requires identity indexing maps");
      return failure();
    }
    StringRef scalarName = scalar->getName().getStringRef();
    if (scalarName == math::TanhOp::getOperationName() &&
        operands.size() == 1) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::TanhOp::getOperationName();
    } else if (scalarName == math::ExpOp::getOperationName() &&
               operands.size() == 1) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::ExpOp::getOperationName();
    } else if (scalarName == math::RsqrtOp::getOperationName() &&
               operands.size() == 1) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::RsqrtOp::getOperationName();
    } else if (scalarName == arith::NegFOp::getOperationName() &&
               operands.size() == 1) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::NegOp::getOperationName();
    } else if (scalarName == arith::MulFOp::getOperationName() &&
               operands.size() == 2) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::MulOp::getOperationName();
    } else if (scalarName == arith::AddFOp::getOperationName() &&
               operands.size() == 2) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::AddOp::getOperationName();
    } else if (scalarName == arith::SubFOp::getOperationName() &&
               operands.size() == 2) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::SubtractOp::getOperationName();
    } else if (scalarName == arith::DivFOp::getOperationName() &&
               operands.size() == 2) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::DivOp::getOperationName();
    } else if (auto convert = dyn_cast<ScalarConvertOp>(scalar);
               convert && operands.size() == 1) {
      Type inputType = convert.getInput().getType();
      Type resultType = convert.getResult().getType();
      const bool validSemantics =
          (inputType.isF32() && resultType.isF32() &&
           convert.getSemantics() == ScalarConvertSemantics::Exact) ||
          (inputType.isBF16() && resultType.isF32() &&
           convert.getSemantics() == ScalarConvertSemantics::Exact) ||
          (inputType.isF32() && resultType.isBF16() &&
           convert.getSemantics() == ScalarConvertSemantics::RoundNearestEven);
      if (!validSemantics) {
        convert.emitOpError("has semantics with no StableHLO conversion");
        return failure();
      }
      stablehloName = stablehlo::ConvertOp::getOperationName();
    } else if (auto constant = dyn_cast<arith::ConstantOp>(scalar);
               constant && operands.empty()) {
      auto value = dyn_cast<FloatAttr>(constant.getValue());
      auto resultType = dyn_cast<RankedTensorType>(map.getResult(0).getType());
      if (!value || !resultType || resultType.getRank() != 0 ||
          value.getType() != resultType.getElementType()) {
        constant.emitOpError("has no rank-zero StableHLO constant lowering");
        return failure();
      }
      stablehloName = stablehlo::ConstantOp::getOperationName();
      SmallVector<Attribute> values{value};
      attributes.append("value", DenseElementsAttr::get(resultType, values));
    } else {
      scalar->emitOpError("has no authoritative StableHLO Map lowering");
      return failure();
    }
  }

  OperationState state(map.getLoc(), stablehloName);
  state.addOperands(operands);
  state.addTypes(map.getResultTypes());
  state.addAttributes(attributes);
  return state;
}

FailureOr<stablehlo::Precision> stablehloPrecision(Attribute attribute) {
  auto value = dyn_cast<StringAttr>(attribute);
  if (!value) {
    return failure();
  }
  return llvm::StringSwitch<FailureOr<stablehlo::Precision>>(value.getValue())
      .Case("DEFAULT", stablehlo::Precision::DEFAULT)
      .Case("HIGH", stablehlo::Precision::HIGH)
      .Case("HIGHEST", stablehlo::Precision::HIGHEST)
      .Default(failure());
}

FailureOr<OperationState> lowerContractState(ContractOp contract,
                                             ValueRange operands) {
  if (operands.size() != 2 || contract.getResults().size() != 1 ||
      contract.getAlgorithm() != "dot_general" ||
      contract.getIndexingMaps().size() != 3) {
    contract.emitOpError("is outside the first Contract lowering contract");
    return failure();
  }
  AffineMap lhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[0]).getValue();
  AffineMap rhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[1]).getValue();
  AffineMap resultMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[2]).getValue();
  if (contract.getIteratorKinds().size() != lhsMap.getNumDims()) {
    contract.emitOpError("has incomplete iterator metadata");
    return failure();
  }

  SmallVector<int64_t> lhsBatching;
  SmallVector<int64_t> rhsBatching;
  SmallVector<int64_t> lhsContracting;
  SmallVector<int64_t> rhsContracting;
  SmallVector<unsigned> batchingDomains;
  for (unsigned domain = 0; domain < lhsMap.getNumDims(); ++domain) {
    FailureOr<unsigned> lhsPosition = mapResultPosition(lhsMap, domain);
    FailureOr<unsigned> rhsPosition = mapResultPosition(rhsMap, domain);
    FailureOr<unsigned> resultPosition = mapResultPosition(resultMap, domain);
    auto iterator = dyn_cast<StringAttr>(contract.getIteratorKinds()[domain]);
    if (!iterator) {
      contract.emitOpError("has a non-string iterator kind");
      return failure();
    }
    if (iterator.getValue() == "reduction") {
      if (failed(lhsPosition) || failed(rhsPosition) ||
          succeeded(resultPosition)) {
        contract.emitOpError("has a non-dot reduction domain");
        return failure();
      }
      lhsContracting.push_back(*lhsPosition);
      rhsContracting.push_back(*rhsPosition);
      continue;
    }
    if (iterator.getValue() != "parallel") {
      contract.emitOpError("has an unsupported iterator kind");
      return failure();
    }
    if (succeeded(lhsPosition) && succeeded(rhsPosition) &&
        succeeded(resultPosition)) {
      lhsBatching.push_back(*lhsPosition);
      rhsBatching.push_back(*rhsPosition);
      batchingDomains.push_back(domain);
    }
  }

  SmallVector<unsigned> expectedResultDomains(batchingDomains);
  auto appendFreeDomains = [&](AffineMap operandMap, AffineMap otherMap) {
    for (AffineExpr expression : operandMap.getResults()) {
      unsigned domain = cast<AffineDimExpr>(expression).getPosition();
      if (failed(mapResultPosition(otherMap, domain)) &&
          succeeded(mapResultPosition(resultMap, domain))) {
        expectedResultDomains.push_back(domain);
      }
    }
  };
  appendFreeDomains(lhsMap, rhsMap);
  appendFreeDomains(rhsMap, lhsMap);
  if (expectedResultDomains.size() != resultMap.getNumResults()) {
    contract.emitOpError("has a non-normalized dot result map");
    return failure();
  }
  for (auto [resultPosition, expression] :
       llvm::enumerate(resultMap.getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension ||
        dimension.getPosition() != expectedResultDomains[resultPosition]) {
      contract.emitOpError("has a non-normalized dot result order");
      return failure();
    }
  }

  SmallVector<Attribute> precision;
  for (Attribute attribute : contract.getPrecision()) {
    FailureOr<stablehlo::Precision> value = stablehloPrecision(attribute);
    if (failed(value)) {
      contract.emitOpError("has unsupported precision metadata");
      return failure();
    }
    precision.push_back(
        stablehlo::PrecisionAttr::get(contract.getContext(), *value));
  }
  NamedAttrList attributes;
  attributes.append("dot_dimension_numbers",
                    stablehlo::DotDimensionNumbersAttr::get(
                        contract.getContext(), lhsBatching, rhsBatching,
                        lhsContracting, rhsContracting));
  attributes.append("precision_config",
                    ArrayAttr::get(contract.getContext(), precision));
  OperationState state(contract.getLoc(),
                       stablehlo::DotGeneralOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(contract.getResultTypes());
  state.addAttributes(attributes);
  return state;
}

FailureOr<Operation *> lowerFold(OpBuilder &builder, FoldOp fold,
                                 ValueRange operands) {
  if (!fold.getOrderFree()) {
    fold.emitOpError(
        "order_free=false has no lossless StableHLO Reduce lowering");
    return failure();
  }
  if (llvm::any_of(fold->getDiscardableAttrs(), [](NamedAttribute attribute) {
        return attribute.getName().strref() != kOperationRefAttribute;
      })) {
    fold.emitOpError("has an unsupported discardable Fold attribute");
    return failure();
  }
  if (operands.size() != 2 || fold.getInputs().size() != 1 ||
      fold.getInitializers().size() != 1 || fold.getNumResults() != 1 ||
      fold.getCombiner().empty()) {
    fold.emitOpError("is outside the first Fold lowering contract");
    return failure();
  }
  Block &body = fold.getCombiner().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2) {
    fold.emitOpError("requires exactly one scalar add and one yield");
    return failure();
  }
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  auto fastMath = add ? add->getAttrOfType<arith::FastMathFlagsAttr>("fastmath")
                      : arith::FastMathFlagsAttr{};
  if (!add || !yield || add.getLhs() != body.getArgument(0) ||
      add.getRhs() != body.getArgument(1) || yield.getValues().size() != 1 ||
      yield.getValues()[0] != add.getResult() ||
      !body.getArgument(0).getType().isF32() ||
      !body.getArgument(1).getType().isF32() ||
      (fastMath && fastMath.getValue() != arith::FastMathFlags::none) ||
      llvm::any_of(add->getAttrs(),
                   [](NamedAttribute attribute) {
                     StringRef name = attribute.getName().strref();
                     return name != "fastmath" &&
                            name != kSourceRefsAttribute &&
                            name != kOperationRefAttribute;
                   }) ||
      llvm::any_of(yield->getAttrs(), [](NamedAttribute attribute) {
        return attribute.getName().strref() != kOperationRefAttribute;
      })) {
    fold.emitOpError("requires the closed ordered scalar f32 add combiner");
    return failure();
  }
  ArrayAttr addRefs = sourceRefs(add);
  auto foldOwner =
      fold->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  auto addOwner = add->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  auto yieldOwner =
      yield->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  if (!addRefs || addRefs.size() != 1 || !foldOwner || !addOwner ||
      !yieldOwner) {
    fold.emitOpError("has incomplete Fold source provenance");
    return failure();
  }

  OperationState state(fold.getLoc(), stablehlo::ReduceOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(fold.getResultTypes());
  state.addAttribute("dimensions", fold.getReductionDimensionsAttr());
  state.addAttribute(kSourceRefsAttribute,
                     ArrayAttr::get(fold.getContext(), {fold.getSource()}));
  state.addAttribute(kOperationRefAttribute, foldOwner);
  state.addRegion();
  Operation *reduce = builder.create(state);
  Block *reducer = new Block();
  reduce->getRegion(0).push_back(reducer);
  auto scalarTensor = RankedTensorType::get({}, builder.getF32Type());
  reducer->addArgument(scalarTensor, body.getArgument(0).getLoc());
  reducer->addArgument(scalarTensor, body.getArgument(1).getLoc());
  OpBuilder reducerBuilder = OpBuilder::atBlockEnd(reducer);
  OperationState addState(add.getLoc(), stablehlo::AddOp::getOperationName());
  addState.addOperands(reducer->getArguments());
  addState.addTypes(scalarTensor);
  addState.addAttribute(kSourceRefsAttribute, addRefs);
  addState.addAttribute(kOperationRefAttribute, addOwner);
  Operation *loweredAdd = reducerBuilder.create(addState);
  OperationState returnState(yield.getLoc(),
                             stablehlo::ReturnOp::getOperationName());
  returnState.addOperands(loweredAdd->getResult(0));
  returnState.addAttribute(kOperationRefAttribute, yieldOwner);
  reducerBuilder.create(returnState);
  return reduce;
}

LogicalResult lowerRegion(RegionOp region) {
  OpBuilder builder(region);
  IRMapping mapping;
  for (auto [argument, input] : llvm::zip_equal(
           region.getBody().front().getArguments(), region.getInputs())) {
    mapping.map(argument, input);
  }

  auto yield = cast<YieldOp>(region.getBody().front().getTerminator());
  for (Operation &operation :
       llvm::make_early_inc_range(region.getBody().front())) {
    if (isa<YieldOp>(operation)) {
      continue;
    }
    SmallVector<Value> operands;
    operands.reserve(operation.getNumOperands());
    for (Value operand : operation.getOperands()) {
      Value mapped = mapping.lookupOrNull(operand);
      if (!mapped) {
        return operation.emitOpError("has an unmapped algebra operand");
      }
      operands.push_back(mapped);
    }
    Attribute source;
    FailureOr<OperationState> state = failure();
    if (auto fold = dyn_cast<FoldOp>(operation)) {
      FailureOr<Operation *> lowered = lowerFold(builder, fold, operands);
      if (failed(lowered)) {
        return failure();
      }
      for (auto [result, loweredResult] :
           llvm::zip_equal(operation.getResults(), (*lowered)->getResults())) {
        mapping.map(result, loweredResult);
      }
      continue;
    }
    if (auto map = dyn_cast<MapOp>(operation)) {
      source = map.getSource();
      state = lowerMapState(map, operands);
    } else if (auto contract = dyn_cast<ContractOp>(operation)) {
      source = contract.getSource();
      state = lowerContractState(contract, operands);
    } else {
      return operation.emitOpError("has no source-ordered StableHLO lowering");
    }
    if (failed(state)) {
      return failure();
    }
    state->addAttribute(kSourceRefsAttribute,
                        ArrayAttr::get(region.getContext(), {source}));
    Operation *lowered = builder.create(*state);
    for (auto [result, loweredResult] :
         llvm::zip_equal(operation.getResults(), lowered->getResults())) {
      mapping.map(result, loweredResult);
    }
  }

  if (yield.getValues().size() != region.getNumResults()) {
    return region.emitOpError("yield/result arity changed before lowering");
  }
  for (auto [result, yielded] :
       llvm::zip_equal(region.getResults(), yield.getValues())) {
    Value replacement = mapping.lookupOrNull(yielded);
    if (!replacement) {
      return region.emitOpError("has an unmapped yielded value");
    }
    result.replaceAllUsesWith(replacement);
  }
  region.erase();
  return success();
}

struct LowerAlgebraToStablehloPass
    : impl::ShuttleLowerAlgebraToStablehloPassBase<
          LowerAlgebraToStablehloPass> {
  void runOnOperation() override {
    if (!getOperation()->hasAttr(kCoverageManifestAttribute)) {
      getOperation().emitError(
          "requires structural coverage before StableHLO lowering");
      signalPassFailure();
      return;
    }
    SmallVector<RegionOp> regions;
    getOperation().walk([&](RegionOp region) { regions.push_back(region); });
    for (RegionOp region : regions) {
      if (failed(lowerRegion(region))) {
        signalPassFailure();
        return;
      }
    }
  }
};

struct StripSourceProvenancePass
    : impl::ShuttleStripSourceProvenancePassBase<StripSourceProvenancePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->removeAttr(kCoverageManifestAttribute);
    module.walk([&](Operation *operation) {
      operation->removeAttr(kSourceRefsAttribute);
      operation->removeAttr(kSelectedAttribute);
      operation->removeAttr(kFunctionOrdinalAttribute);
      operation->removeAttr(kOperationRefAttribute);
      operation->removeAttr(kRegionResultSourcesAttribute);
      if (containsShuttleAttribute(operation->getLoc())) {
        operation->setLoc(UnknownLoc::get(operation->getContext()));
      }
    });
  }
};

struct VerifyNoShuttleOpsPass
    : impl::ShuttleVerifyNoShuttleOpsPassBase<VerifyNoShuttleOpsPass> {
  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (operation->getName().getDialectNamespace() ==
          ShuttleDialect::getDialectNamespace()) {
        operation->emitOpError("Shuttle operation remains before HLO export");
        return WalkResult::interrupt();
      }
      if (containsShuttleAttribute(operation->getLoc())) {
        operation->emitOpError(
            "Shuttle attribute remains in a location before HLO export");
        return WalkResult::interrupt();
      }
      for (NamedAttribute namedAttribute : operation->getAttrs()) {
        if (namedAttribute.getName().strref().starts_with("shuttle.") ||
            containsShuttleAttribute(namedAttribute.getValue())) {
          operation->emitOpError()
              << "Shuttle attribute remains before HLO export: "
              << namedAttribute.getName();
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

std::string attributeText(Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return text;
}

std::string normalizedModuleFingerprintImpl(ModuleOp module) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  for (NamedAttribute attribute : module->getAttrs()) {
    if (attribute.getName().strref() == SymbolTable::getSymbolAttrName()) {
      continue;
    }
    stream << "module_attr:" << attribute.getName() << '=';
    attribute.getValue().print(stream);
    stream << '\n';
  }
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    stream << "function:";
    function.getFunctionType().print(stream);
    stream << '\n';
    llvm::SmallDenseMap<Value, uint64_t> values;
    uint64_t nextValue = 0;
    for (BlockArgument argument : function.getArguments()) {
      values.try_emplace(argument, nextValue++);
    }
    std::function<void(Operation &)> printOperation =
        [&](Operation &operation) {
          stream << operation.getName() << '(';
          for (Value operand : operation.getOperands()) {
            auto position = values.find(operand);
            if (position == values.end()) {
              stream << "missing";
            } else {
              stream << position->second;
            }
            stream << ',';
          }
          stream << ")->(";
          for (Type type : operation.getResultTypes()) {
            type.print(stream);
            stream << ',';
          }
          stream << "){";
          for (NamedAttribute attribute : operation.getAttrs()) {
            StringRef name = attribute.getName().strref();
            if (name == SymbolTable::getSymbolAttrName() ||
                name == kSourceRefsAttribute || name == kSelectedAttribute ||
                name == kFunctionOrdinalAttribute ||
                name == kOperationRefAttribute ||
                name == kCoverageManifestAttribute ||
                name == kRegionResultSourcesAttribute) {
              continue;
            }
            stream << name << '=';
            attribute.getValue().print(stream);
            stream << ',';
          }
          stream << "}\n";
          for (Value result : operation.getResults()) {
            values.try_emplace(result, nextValue++);
          }
          for (auto [regionOrdinal, region] :
               llvm::enumerate(operation.getRegions())) {
            stream << "region:" << regionOrdinal << '\n';
            for (auto [blockOrdinal, block] : llvm::enumerate(region)) {
              stream << "block:" << blockOrdinal << '(';
              for (BlockArgument argument : block.getArguments()) {
                argument.getType().print(stream);
                stream << ':' << nextValue << ',';
                values.try_emplace(argument, nextValue++);
              }
              stream << ")\n";
              for (Operation &nested : block) {
                printOperation(nested);
              }
            }
          }
        };
    for (Operation &operation : function.getBody().front()) {
      printOperation(operation);
    }
  }
  stream.flush();
  return sha256(normalized);
}

std::string generatedGpuBundleFingerprintImpl(ModuleOp module) {
  SmallVector<DeviceModuleOp> devices(module.getOps<DeviceModuleOp>());
  SmallVector<InvocationAbiOp> abis(module.getOps<InvocationAbiOp>());
  SmallVector<ExecutableBundleOp> bundles(module.getOps<ExecutableBundleOp>());
  if (bundles.empty() || devices.size() != bundles.size() ||
      abis.size() != bundles.size())
    return {};
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  llvm::SmallDenseSet<Operation *> matchedDevices;
  llvm::SmallDenseSet<Operation *> matchedAbis;
  stream << "bundle_count:" << bundles.size() << '\n';
  for (auto [ordinal, bundle] : llvm::enumerate(bundles)) {
    DeviceModuleOp device;
    for (DeviceModuleOp candidate : devices) {
      if (candidate.getFingerprint() != bundle.getDeviceModuleFingerprint())
        continue;
      if (device)
        return {};
      device = candidate;
    }
    InvocationAbiOp abi;
    for (InvocationAbiOp candidate : abis) {
      if (candidate.getFingerprint() != bundle.getInvocationAbiFingerprint())
        continue;
      if (abi)
        return {};
      abi = candidate;
    }
    if (!device || !abi ||
        !matchedDevices.insert(device.getOperation()).second ||
        !matchedAbis.insert(abi.getOperation()).second)
      return {};
    stream << "tuple:" << ordinal
           << "\ndevice_module:" << deviceModuleFingerprint(device)
           << "\ninvocation_abi:" << invocationAbiFingerprint(abi)
           << "\nexecutable_bundle:" << executableBundleFingerprint(bundle)
           << '\n';
  }
  stream.flush();
  return sha256(normalized);
}

bool isMaterializationFingerprintProvenance(StringRef name) {
  return name == kSourceRefsAttribute || name == kSelectedAttribute ||
         name == kFunctionOrdinalAttribute || name == kOperationRefAttribute ||
         name == kCoverageManifestAttribute ||
         name == kRegionResultSourcesAttribute || name == "source";
}

std::string semanticTaskFingerprint(Operation *root) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  llvm::SmallDenseMap<Value, uint64_t> values;
  uint64_t nextValue = 0;
  for (Value operand : root->getOperands()) {
    values.try_emplace(operand, nextValue++);
  }
  std::function<void(Operation &)> printOperation = [&](Operation &operation) {
    stream << operation.getName() << '(';
    for (Value operand : operation.getOperands()) {
      auto position = values.find(operand);
      stream << (position == values.end() ? UINT64_MAX : position->second)
             << ',';
    }
    stream << ")->(";
    for (Type type : operation.getResultTypes()) {
      type.print(stream);
      stream << ',';
    }
    stream << "){";
    for (NamedAttribute attribute : operation.getAttrs()) {
      if (isMaterializationFingerprintProvenance(
              attribute.getName().strref())) {
        continue;
      }
      stream << attribute.getName() << '=';
      attribute.getValue().print(stream);
      stream << ',';
    }
    stream << "}\n";
    for (Value result : operation.getResults()) {
      values.try_emplace(result, nextValue++);
    }
    for (Region &region : operation.getRegions()) {
      for (Block &block : region) {
        stream << "block(";
        for (BlockArgument argument : block.getArguments()) {
          argument.getType().print(stream);
          stream << ':' << nextValue << ',';
          values.try_emplace(argument, nextValue++);
        }
        stream << ")\n";
        for (Operation &nested : block) {
          printOperation(nested);
        }
      }
    }
  };
  printOperation(*root);
  stream.flush();
  return sha256(normalized);
}

FailureOr<SmallVector<int64_t>> mapDomainShape(MapOp map) {
  if (map.getNumResults() != 1) {
    return failure();
  }
  auto resultType = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() ||
      llvm::any_of(resultType.getShape(),
                   [](int64_t extent) { return extent <= 0; })) {
    return failure();
  }
  AffineMap resultMap =
      cast<AffineMapAttr>(
          map.getIndexingMaps()[map.getIndexingMaps().size() - 1])
          .getValue();
  SmallVector<int64_t> domain(resultMap.getNumDims(), ShapedType::kDynamic);
  for (auto [resultPosition, expression] :
       llvm::enumerate(resultMap.getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension) {
      return failure();
    }
    domain[dimension.getPosition()] = resultType.getDimSize(resultPosition);
  }
  if (llvm::is_contained(domain, ShapedType::kDynamic)) {
    return failure();
  }
  return domain;
}

FailureOr<SmallVector<int64_t>> genericMapDomainShape(MapOp map) {
  if (map.getNumResults() == 0 ||
      map.getIndexingMaps().size() !=
          map.getInputs().size() + map.getNumResults()) {
    return failure();
  }
  std::optional<SmallVector<int64_t>> commonDomain;
  ArrayRef<Attribute> resultMaps =
      map.getIndexingMaps().getValue().drop_front(map.getInputs().size());
  for (auto [result, mapAttribute] :
       llvm::zip_equal(map.getResults(), resultMaps)) {
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    auto resultMap = dyn_cast<AffineMapAttr>(mapAttribute);
    if (!resultType || !resultType.hasStaticShape() || !resultMap ||
        resultMap.getValue().getNumSymbols() != 0 ||
        resultMap.getValue().getNumResults() != resultType.getRank() ||
        llvm::any_of(resultType.getShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      return failure();
    }
    SmallVector<int64_t> domain(resultMap.getValue().getNumDims(),
                                ShapedType::kDynamic);
    for (auto [resultPosition, expression] :
         llvm::enumerate(resultMap.getValue().getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension ||
          domain[dimension.getPosition()] != ShapedType::kDynamic) {
        return failure();
      }
      domain[dimension.getPosition()] = resultType.getDimSize(resultPosition);
    }
    if (llvm::is_contained(domain, ShapedType::kDynamic) ||
        (commonDomain && *commonDomain != domain)) {
      return failure();
    }
    commonDomain = std::move(domain);
  }
  return std::move(*commonDomain);
}

FailureOr<SmallVector<int64_t>> contractDomainShape(ContractOp contract) {
  if (contract.getInputs().size() != 2 || contract.getNumResults() != 1 ||
      contract.getIndexingMaps().size() != 3) {
    return failure();
  }
  AffineMap first =
      cast<AffineMapAttr>(contract.getIndexingMaps()[0]).getValue();
  SmallVector<int64_t> domain(first.getNumDims(), ShapedType::kDynamic);
  SmallVector<Type> indexedTypes;
  llvm::append_range(indexedTypes, contract.getInputs().getTypes());
  llvm::append_range(indexedTypes, contract.getResults().getTypes());
  for (auto [mapAttribute, type] :
       llvm::zip_equal(contract.getIndexingMaps(), indexedTypes)) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    auto map = dyn_cast<AffineMapAttr>(mapAttribute);
    if (!tensor || !tensor.hasStaticShape() || !map ||
        map.getValue().getNumDims() != domain.size() ||
        map.getValue().getNumResults() != tensor.getRank()) {
      return failure();
    }
    for (auto [axis, expression] :
         llvm::enumerate(map.getValue().getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension || tensor.getDimSize(axis) <= 0) {
        return failure();
      }
      int64_t &extent = domain[dimension.getPosition()];
      if (!ShapedType::isDynamic(extent) && extent != tensor.getDimSize(axis)) {
        return failure();
      }
      extent = tensor.getDimSize(axis);
    }
  }
  if (llvm::is_contained(domain, ShapedType::kDynamic)) {
    return failure();
  }
  return domain;
}

bool isClosedProjectedContract(ContractOp contract) {
  auto lhs = contract.getInputs().size() == 2
                 ? dyn_cast<RankedTensorType>(contract.getInputs()[0].getType())
                 : RankedTensorType{};
  auto rhs = contract.getInputs().size() == 2
                 ? dyn_cast<RankedTensorType>(contract.getInputs()[1].getType())
                 : RankedTensorType{};
  auto result =
      contract.getNumResults() == 1
          ? dyn_cast<RankedTensorType>(contract.getResult(0).getType())
          : RankedTensorType{};
  FailureOr<SmallVector<int64_t>> domain = contractDomainShape(contract);
  constexpr int64_t kMaximumContractRank = 8;
  if (!lhs || !rhs || !result || failed(domain) || lhs.getRank() < 2 ||
      rhs.getRank() < 2 || result.getRank() < 2 ||
      lhs.getRank() > kMaximumContractRank ||
      rhs.getRank() > kMaximumContractRank ||
      result.getRank() > kMaximumContractRank ||
      lhs.getElementType() != rhs.getElementType() ||
      (!lhs.getElementType().isBF16() && !lhs.getElementType().isF32()) ||
      !result.getElementType().isF32() ||
      domain->size() != static_cast<size_t>(result.getRank() + 1) ||
      contract.getAlgorithm() != "dot_general" ||
      contract.getAccumulatorTypes().size() != 1 ||
      cast<TypeAttr>(contract.getAccumulatorTypes()[0]).getValue() !=
          Float32Type::get(contract.getContext()) ||
      contract.getPrecision().size() != 2 ||
      llvm::any_of(contract.getPrecision(), [](Attribute precision) {
        auto value = dyn_cast<StringAttr>(precision);
        return !value || value.getValue() != "DEFAULT";
      })) {
    return false;
  }
  int64_t reduction = -1;
  for (auto [position, iterator] :
       llvm::enumerate(contract.getIteratorKinds())) {
    auto value = dyn_cast<StringAttr>(iterator);
    if (!value) {
      return false;
    }
    if (value.getValue() == "reduction") {
      if (reduction >= 0) {
        return false;
      }
      reduction = position;
    } else if (value.getValue() != "parallel") {
      return false;
    }
  }
  if (reduction != static_cast<int64_t>(domain->size()) - 1) {
    return false;
  }
  AffineMap lhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[0]).getValue();
  AffineMap rhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[1]).getValue();
  AffineMap resultMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[2]).getValue();
  MLIRContext *context = contract.getContext();
  SmallVector<AffineExpr> expectedResultExpressions;
  expectedResultExpressions.reserve(result.getRank());
  for (int64_t dimension = 0; dimension < result.getRank(); ++dimension) {
    expectedResultExpressions.push_back(getAffineDimExpr(dimension, context));
  }
  if (resultMap !=
      AffineMap::get(domain->size(), 0, expectedResultExpressions, context)) {
    return false;
  }
  for (unsigned dimension = 0; dimension < domain->size(); ++dimension) {
    const bool inLhs = succeeded(mapResultPosition(lhsMap, dimension));
    const bool inRhs = succeeded(mapResultPosition(rhsMap, dimension));
    const bool inResult = succeeded(mapResultPosition(resultMap, dimension));
    if (dimension == static_cast<unsigned>(reduction)) {
      if (!inLhs || !inRhs || inResult) {
        return false;
      }
    } else if ((!inLhs && !inRhs) || !inResult) {
      return false;
    }
  }
  return true;
}

bool isClosedContractMapRegion(RegionOp region, ArrayRef<Operation *> algebra) {
  auto isExternalType = [](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    return tensor && tensor.hasStaticShape() && tensor.getRank() >= 2 &&
           tensor.getRank() <= 8 && tensor.getElementType().isBF16() &&
           llvm::all_of(tensor.getShape(),
                        [](int64_t extent) { return extent > 0; });
  };
  if (!llvm::all_of(
          region.getInputs(),
          [&](Value input) { return isExternalType(input.getType()); }) ||
      !llvm::all_of(region.getResults(), [&](Value result) {
        return isExternalType(result.getType());
      })) {
    return false;
  }
  return llvm::all_of(algebra, [](Operation *operation) {
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      return isClosedProjectedContract(contract);
    }
    return isa<MapOp>(operation);
  });
}

bool isClosedRankTwoFold(FoldOp fold) {
  auto inputType =
      fold.getInputs().size() == 1
          ? dyn_cast<RankedTensorType>(fold.getInputs()[0].getType())
          : RankedTensorType{};
  auto initType =
      fold.getInitializers().size() == 1
          ? dyn_cast<RankedTensorType>(fold.getInitializers()[0].getType())
          : RankedTensorType{};
  auto resultType =
      fold.getNumResults() == 1
          ? dyn_cast<RankedTensorType>(fold.getResult(0).getType())
          : RankedTensorType{};
  if (!inputType || !initType || !resultType || !inputType.hasStaticShape() ||
      inputType.getRank() != 2 || inputType.getDimSize(0) <= 0 ||
      inputType.getDimSize(1) <= 0 || !inputType.getElementType().isF32() ||
      initType.getRank() != 0 || !initType.getElementType().isF32() ||
      fold.getReductionDimensions().size() != 1 ||
      fold.getReductionDimensions().front() < 0 ||
      fold.getReductionDimensions().front() >= inputType.getRank() ||
      !fold.getOrderFree() || fold.getAccumulatorTypes().size() != 1 ||
      cast<TypeAttr>(fold.getAccumulatorTypes()[0]).getValue() !=
          Float32Type::get(fold.getContext())) {
    return false;
  }
  SmallVector<int64_t> expectedResultShape(inputType.getShape());
  expectedResultShape.erase(expectedResultShape.begin() +
                            fold.getReductionDimensions().front());
  if (resultType.getShape() != ArrayRef<int64_t>(expectedResultShape) ||
      !resultType.getElementType().isF32()) {
    return false;
  }
  Block &body = fold.getCombiner().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2) {
    return false;
  }
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  auto fastMath = add ? add->getAttrOfType<arith::FastMathFlagsAttr>("fastmath")
                      : arith::FastMathFlagsAttr{};
  if (!add || !yield || add.getLhs() != body.getArgument(0) ||
      add.getRhs() != body.getArgument(1) || yield.getValues().size() != 1 ||
      yield.getValues()[0] != add.getResult() ||
      (fastMath && fastMath.getValue() != arith::FastMathFlags::none)) {
    return false;
  }
  return true;
}

bool hasExactPositiveZeroInitializer(FoldOp fold) {
  if (fold.getInitializers().size() != 1) {
    return false;
  }
  auto initializer = fold.getInitializers()[0].getDefiningOp<MapOp>();
  if (!initializer || !initializer.getInputs().empty() ||
      initializer.getBody().empty()) {
    return false;
  }
  Block &body = initializer.getBody().front();
  auto constant = body.getOperations().size() == 2
                      ? dyn_cast<arith::ConstantOp>(body.front())
                      : arith::ConstantOp{};
  auto value =
      constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
  return value && value.getType().isF32() && value.getValue().isZero() &&
         !value.getValue().isNegative();
}

bool isClosedTerminalRankNFold(FoldOp fold) {
  if (!isClosedRankTwoFold(fold)) {
    auto inputType =
        fold.getInputs().size() == 1
            ? dyn_cast<RankedTensorType>(fold.getInputs()[0].getType())
            : RankedTensorType{};
    if (!inputType || !inputType.hasStaticShape() || inputType.getRank() < 3 ||
        inputType.getRank() > 8 ||
        fold.getReductionDimensions() !=
            ArrayRef<int64_t>({inputType.getRank() - 1})) {
      return false;
    }
    // Reuse the closed scalar/type/combiner checks with a shape-only rank-two
    // view would obscure the actual operation. Spell out the rank-independent
    // portion here instead.
    auto initType =
        fold.getInitializers().size() == 1
            ? dyn_cast<RankedTensorType>(fold.getInitializers()[0].getType())
            : RankedTensorType{};
    auto resultType =
        fold.getNumResults() == 1
            ? dyn_cast<RankedTensorType>(fold.getResult(0).getType())
            : RankedTensorType{};
    if (!initType || !resultType || !inputType.getElementType().isF32() ||
        initType.getRank() != 0 || !initType.getElementType().isF32() ||
        !fold.getOrderFree() || fold.getAccumulatorTypes().size() != 1 ||
        cast<TypeAttr>(fold.getAccumulatorTypes()[0]).getValue() !=
            Float32Type::get(fold.getContext()) ||
        resultType.getShape() != inputType.getShape().drop_back() ||
        !resultType.getElementType().isF32()) {
      return false;
    }
    Block &body = fold.getCombiner().front();
    if (body.getNumArguments() != 2 || body.getOperations().size() != 2) {
      return false;
    }
    auto add = dyn_cast<arith::AddFOp>(body.front());
    auto yield = dyn_cast<YieldOp>(body.back());
    auto fastMath =
        add ? add->getAttrOfType<arith::FastMathFlagsAttr>("fastmath")
            : arith::FastMathFlagsAttr{};
    if (!add || !yield || add.getLhs() != body.getArgument(0) ||
        add.getRhs() != body.getArgument(1) || yield.getValues().size() != 1 ||
        yield.getValues()[0] != add.getResult() ||
        (fastMath && fastMath.getValue() != arith::FastMathFlags::none)) {
      return false;
    }
  }
  return hasExactPositiveZeroInitializer(fold);
}

bool isFullPrefixReduction(ArrayRef<int64_t> dimensions, int64_t rank) {
  return rank >= 3 && rank <= 8 &&
         dimensions.size() == static_cast<size_t>(rank - 1) &&
         llvm::all_of(llvm::enumerate(dimensions), [](auto indexedDimension) {
           return indexedDimension.value() ==
                  static_cast<int64_t>(indexedDimension.index());
         });
}

bool isClosedFullPrefixRankNFold(FoldOp fold) {
  auto input =
      fold.getInputs().size() == 1
          ? dyn_cast<RankedTensorType>(fold.getInputs()[0].getType())
          : RankedTensorType{};
  auto initializer =
      fold.getInitializers().size() == 1
          ? dyn_cast<RankedTensorType>(fold.getInitializers()[0].getType())
          : RankedTensorType{};
  auto result = fold.getNumResults() == 1
                    ? dyn_cast<RankedTensorType>(fold.getResult(0).getType())
                    : RankedTensorType{};
  if (!input || !initializer || !result || !input.hasStaticShape() ||
      !result.hasStaticShape() ||
      !isFullPrefixReduction(fold.getReductionDimensions(), input.getRank()) ||
      !input.getElementType().isF32() || initializer.getRank() != 0 ||
      !initializer.getElementType().isF32() || result.getRank() != 1 ||
      !result.getElementType().isF32() ||
      result.getShape() != input.getShape().take_back() ||
      !llvm::all_of(input.getShape(),
                    [](int64_t extent) { return extent > 0; }) ||
      !fold.getOrderFree() || fold.getAccumulatorTypes().size() != 1 ||
      cast<TypeAttr>(fold.getAccumulatorTypes()[0]).getValue() !=
          Float32Type::get(fold.getContext())) {
    return false;
  }
  Block &body = fold.getCombiner().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2) {
    return false;
  }
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  auto fastMath = add ? add->getAttrOfType<arith::FastMathFlagsAttr>("fastmath")
                      : arith::FastMathFlagsAttr{};
  return add && yield && add.getLhs() == body.getArgument(0) &&
         add.getRhs() == body.getArgument(1) &&
         yield.getValues().size() == 1 &&
         yield.getValues()[0] == add.getResult() &&
         (!fastMath || fastMath.getValue() == arith::FastMathFlags::none) &&
         hasExactPositiveZeroInitializer(fold);
}

bool isClosedRankNFold(FoldOp fold) {
  return isClosedTerminalRankNFold(fold) ||
         isClosedFullPrefixRankNFold(fold);
}

bool isClosedAdditivePlanningFold(FoldOp fold) {
  auto input = fold.getInputs().size() == 1
                   ? dyn_cast<RankedTensorType>(fold.getInputs()[0].getType())
                   : RankedTensorType{};
  auto initializer =
      fold.getInitializers().size() == 1
          ? dyn_cast<RankedTensorType>(fold.getInitializers()[0].getType())
          : RankedTensorType{};
  auto result = fold.getNumResults() == 1
                    ? dyn_cast<RankedTensorType>(fold.getResult(0).getType())
                    : RankedTensorType{};
  if (!input || !initializer || !result || !input.hasStaticShape() ||
      !result.hasStaticShape() || !input.getElementType().isF32() ||
      initializer.getRank() != 0 || !initializer.getElementType().isF32() ||
      !result.getElementType().isF32() || !fold.getOrderFree() ||
      fold.getAccumulatorTypes().size() != 1 ||
      cast<TypeAttr>(fold.getAccumulatorTypes()[0]).getValue() !=
          Float32Type::get(fold.getContext()) ||
      llvm::any_of(input.getShape(),
                   [](int64_t extent) { return extent <= 0; })) {
    return false;
  }
  ArrayRef<int64_t> dimensions = fold.getReductionDimensions();
  const bool rankTwo =
      input.getRank() == 2 && dimensions.size() == 1 &&
      dimensions.front() >= 0 && dimensions.front() < input.getRank();
  const bool terminalRankN =
      input.getRank() >= 3 && input.getRank() <= 8 &&
      dimensions == ArrayRef<int64_t>({input.getRank() - 1});
  if (!rankTwo && !terminalRankN &&
      !isFullPrefixReduction(dimensions, input.getRank())) {
    return false;
  }
  SmallVector<int64_t> expectedResult(input.getShape());
  for (int64_t dimension : llvm::reverse(dimensions))
    expectedResult.erase(expectedResult.begin() + dimension);
  if (result.getShape() != ArrayRef<int64_t>(expectedResult))
    return false;
  Block &body = fold.getCombiner().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2)
    return false;
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  auto fastMath = add ? add->getAttrOfType<arith::FastMathFlagsAttr>("fastmath")
                      : arith::FastMathFlagsAttr{};
  return add && yield && add.getLhs() == body.getArgument(0) &&
         add.getRhs() == body.getArgument(1) &&
         yield.getValues().size() == 1 &&
         yield.getValues()[0] == add.getResult() &&
         (!fastMath || fastMath.getValue() == arith::FastMathFlags::none);
}

bool isClosedDirectProjectedMap(MapOp map) {
  if (map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != map.getInputs().size() + 1) {
    return false;
  }
  auto result = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  ArrayRef<Attribute> maps = map.getIndexingMaps().getValue();
  auto resultMap = dyn_cast<AffineMapAttr>(maps.back());
  if (!result || !result.hasStaticShape() || !resultMap ||
      !isIdentityMap(resultMap.getValue())) {
    return false;
  }
  for (auto [input, attribute] :
       llvm::zip_equal(map.getInputs(), maps.drop_back())) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    auto indexing = dyn_cast<AffineMapAttr>(attribute);
    if (!type || !type.hasStaticShape() || !indexing ||
        indexing.getValue().getNumDims() != result.getRank() ||
        indexing.getValue().getNumSymbols() != 0 ||
        indexing.getValue().getNumResults() != type.getRank()) {
      return false;
    }
    llvm::SmallDenseSet<unsigned> dimensions;
    for (auto [axis, expression] :
         llvm::enumerate(indexing.getValue().getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension || dimension.getPosition() >= result.getRank() ||
          !dimensions.insert(dimension.getPosition()).second ||
          type.getDimSize(axis) != result.getDimSize(dimension.getPosition())) {
        return false;
      }
    }
  }
  return true;
}

bool isClosedFullPrefixBf16Result(MapOp map) {
  if (map.getSemantics() != MapSemantics::Pointwise ||
      map.getInputs().size() != 1 || map.getNumResults() != 1 ||
      !isClosedDirectProjectedMap(map) || map.getBody().empty()) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(map.getInputs()[0].getType());
  auto result = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  auto fold = map.getInputs()[0].getDefiningOp<FoldOp>();
  if (!input || !result || !fold || fold->getParentOp() != map->getParentOp() ||
      !isClosedFullPrefixRankNFold(fold) || input.getRank() != 1 ||
      result.getRank() != 1 || input.getShape() != result.getShape() ||
      !input.getElementType().isF32() || !result.getElementType().isBF16() ||
      !llvm::all_of(result.getShape(),
                    [](int64_t extent) { return extent > 0; })) {
    return false;
  }
  Block &body = map.getBody().front();
  if (body.getNumArguments() != 1 || body.getOperations().size() != 2) {
    return false;
  }
  auto convert = dyn_cast<ScalarConvertOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  return convert && yield && convert.getInput() == body.getArgument(0) &&
         convert.getInput().getType().isF32() &&
         convert.getResult().getType().isBF16() &&
         convert.getSemantics() ==
             ScalarConvertSemantics::RoundNearestEven &&
         yield.getValues().size() == 1 &&
         yield.getValues()[0] == convert.getResult();
}

std::optional<unsigned> closedRankNExpandingSingletonMaps(MapOp map) {
  if (map.getSemantics() != MapSemantics::BroadcastInDim ||
      map.getInputs().size() != 1 || map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != 2 || map.getBody().empty()) {
    return std::nullopt;
  }
  auto input = dyn_cast<RankedTensorType>(map.getInputs()[0].getType());
  auto result = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  auto inputMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[0]);
  auto resultMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[1]);
  if (!input || !result || !input.hasStaticShape() ||
      !result.hasStaticShape() || input.getRank() < 3 || input.getRank() > 8 ||
      input.getRank() != result.getRank() || !input.getElementType().isF32() ||
      !result.getElementType().isF32() || !inputMap || !resultMap ||
      inputMap.getValue().getNumSymbols() != 0 ||
      inputMap.getValue().getNumDims() != result.getRank() ||
      inputMap.getValue().getNumResults() != input.getRank() ||
      !isIdentityMap(resultMap.getValue())) {
    return std::nullopt;
  }

  unsigned expandingSingletons = 0;
  for (auto [axis, expression] :
       llvm::enumerate(inputMap.getValue().getResults())) {
    if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
      if (dimension.getPosition() != axis ||
          input.getDimSize(axis) != result.getDimSize(axis)) {
        return std::nullopt;
      }
      continue;
    }
    auto floorDiv = dyn_cast<AffineBinaryOpExpr>(expression);
    AffineExpr dividendExpression =
        floorDiv && floorDiv.getKind() == AffineExprKind::FloorDiv
            ? floorDiv.getLHS()
            : AffineExpr{};
    AffineExpr divisorExpression =
        floorDiv && floorDiv.getKind() == AffineExprKind::FloorDiv
            ? floorDiv.getRHS()
            : AffineExpr{};
    auto dividend = dyn_cast_if_present<AffineDimExpr>(dividendExpression);
    auto divisor = dyn_cast_if_present<AffineConstantExpr>(divisorExpression);
    if (!dividend || !divisor || dividend.getPosition() != axis ||
        input.getDimSize(axis) != 1 || result.getDimSize(axis) <= 1 ||
        divisor.getValue() != result.getDimSize(axis)) {
      return std::nullopt;
    }
    ++expandingSingletons;
  }
  if (expandingSingletons == 0) {
    return std::nullopt;
  }
  Block &body = map.getBody().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (body.getNumArguments() != 1 || body.getOperations().size() != 1 ||
      !yield || yield.getValues().size() != 1 ||
      yield.getValues()[0] != body.getArgument(0)) {
    return std::nullopt;
  }
  return expandingSingletons;
}

bool hasIdentityScalarBody(MapOp map) {
  if (map.getBody().empty()) {
    return false;
  }
  Block &body = map.getBody().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  return body.getNumArguments() == 1 && body.getOperations().size() == 1 &&
         yield && yield.getValues().size() == 1 &&
         yield.getValues()[0] == body.getArgument(0);
}

bool isClosedProjectedParameterInsertionMap(MapOp map) {
  if (map.getSemantics() != MapSemantics::BroadcastInDim ||
      map.getInputs().size() != 1 || map.getNumResults() != 1 ||
      !isClosedDirectProjectedMap(map) || !hasIdentityScalarBody(map)) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(map.getInputs()[0].getType());
  auto result = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  auto inputMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[0]);
  if (!input || !result || !inputMap || !input.getElementType().isF32() ||
      !result.getElementType().isF32() || input.getRank() < 1 ||
      input.getRank() >= result.getRank() || result.getRank() > 8) {
    return false;
  }
  llvm::SmallDenseSet<unsigned> mapped;
  for (AffineExpr expression : inputMap.getValue().getResults()) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || !mapped.insert(dimension.getPosition()).second) {
      return false;
    }
  }
  for (auto [axis, extent] : llvm::enumerate(result.getShape())) {
    if (!mapped.contains(axis) && extent != 1) {
      return false;
    }
  }
  return true;
}

std::optional<Value> projectedBf16ParameterRoot(MapOp conversion,
                                                int64_t expansionRank) {
  if (conversion.getSemantics() != MapSemantics::Pointwise ||
      conversion.getInputs().size() != 1 || conversion.getNumResults() != 1 ||
      conversion.getIndexingMaps().size() != 2 ||
      conversion.getInputs()[0].getDefiningOp() || conversion.getBody().empty()) {
    return std::nullopt;
  }
  auto input = dyn_cast<RankedTensorType>(conversion.getInputs()[0].getType());
  auto result =
      dyn_cast<RankedTensorType>(conversion.getResult(0).getType());
  auto inputMap = dyn_cast<AffineMapAttr>(conversion.getIndexingMaps()[0]);
  auto resultMap = dyn_cast<AffineMapAttr>(conversion.getIndexingMaps()[1]);
  if (!input || !result || !input.hasStaticShape() ||
      !result.hasStaticShape() || input.getRank() < 1 ||
      input.getRank() >= expansionRank || input.getShape() != result.getShape() ||
      !input.getElementType().isBF16() || !result.getElementType().isF32() ||
      !inputMap || !resultMap || !isIdentityMap(inputMap.getValue()) ||
      !isIdentityMap(resultMap.getValue()) ||
      !llvm::all_of(input.getShape(),
                    [](int64_t extent) { return extent > 0; })) {
    return std::nullopt;
  }
  Block &body = conversion.getBody().front();
  if (body.getNumArguments() != 1 || body.getOperations().size() != 2) {
    return std::nullopt;
  }
  auto scalarConvert = dyn_cast<ScalarConvertOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  if (!scalarConvert || !yield ||
      scalarConvert.getInput() != body.getArgument(0) ||
      !scalarConvert.getInput().getType().isBF16() ||
      !scalarConvert.getResult().getType().isF32() ||
      scalarConvert.getSemantics() != ScalarConvertSemantics::Exact ||
      yield.getValues().size() != 1 ||
      yield.getValues()[0] != scalarConvert.getResult()) {
    return std::nullopt;
  }
  return conversion.getInputs()[0];
}

std::optional<Value> projectedParameterRoot(MapOp expansion) {
  std::optional<unsigned> expansions =
      closedRankNExpandingSingletonMaps(expansion);
  if (!expansions || *expansions < 2) {
    return std::nullopt;
  }
  auto expansionType =
      cast<RankedTensorType>(expansion.getResult(0).getType());
  Region *region = expansion->getParentRegion();
  Value producerValue = expansion.getInputs()[0];
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (Operation *producer = producerValue.getDefiningOp()) {
    if (producer->getParentRegion() != region ||
        !visited.insert(producer).second) {
      return std::nullopt;
    }
    auto map = dyn_cast<MapOp>(producer);
    if (!map) {
      return std::nullopt;
    }
    if (std::optional<Value> root =
            projectedBf16ParameterRoot(map, expansionType.getRank())) {
      auto argument = dyn_cast<BlockArgument>(*root);
      auto parentRegion = dyn_cast<RegionOp>(expansion->getParentOp());
      if (!argument || !parentRegion ||
          argument.getOwner() != &parentRegion.getBody().front() ||
          argument.getArgNumber() >= parentRegion.getInputs().size()) {
        return std::nullopt;
      }
      return parentRegion.getInputs()[argument.getArgNumber()];
    }
    if (!isClosedProjectedParameterInsertionMap(map) ||
        map.getInputs().size() != 1) {
      return std::nullopt;
    }
    producerValue = map.getInputs()[0];
  }
  return std::nullopt;
}

bool isExpansionDrivenByTerminalFold(MapOp expansion) {
  Operation *root = expansion.getInputs()[0].getDefiningOp();
  if (!root) {
    return false;
  }
  Operation *parent = expansion->getParentOp();
  SmallVector<Operation *> worklist{root};
  llvm::SmallPtrSet<Operation *, 8> visited;
  FoldOp terminalFold;
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val();
    if (producer->getParentOp() != parent) {
      return false;
    }
    if (!visited.insert(producer).second) {
      continue;
    }
    if (auto fold = dyn_cast<FoldOp>(producer)) {
      if (!isClosedTerminalRankNFold(fold) ||
          (terminalFold && terminalFold != fold)) {
        return false;
      }
      terminalFold = fold;
      continue;
    }
    auto map = dyn_cast<MapOp>(producer);
    if (!map || !isClosedDirectProjectedMap(map) ||
        closedRankNExpandingSingletonMaps(map).has_value()) {
      return false;
    }
    if (map.getInputs().empty()) {
      if (map.getBody().empty()) {
        return false;
      }
      auto result = map.getNumResults() == 1
                        ? dyn_cast<RankedTensorType>(map.getResult(0).getType())
                        : RankedTensorType{};
      Block &body = map.getBody().front();
      auto constant = body.getOperations().size() == 2
                          ? dyn_cast<arith::ConstantOp>(body.front())
                          : arith::ConstantOp{};
      auto yield = dyn_cast<YieldOp>(body.getTerminator());
      auto value = constant ? dyn_cast<FloatAttr>(constant.getValue())
                            : FloatAttr{};
      if (!result || result.getRank() != 0 ||
          !result.getElementType().isF32() || !value ||
          !value.getType().isF32() || !yield ||
          yield.getValues().size() != 1 ||
          yield.getValues()[0] != constant.getResult()) {
        return false;
      }
      continue;
    }
    for (Value input : map.getInputs()) {
      Operation *inputProducer = input.getDefiningOp();
      if (!inputProducer || inputProducer->getParentOp() != parent) {
        return false;
      }
      worklist.push_back(inputProducer);
    }
  }
  return static_cast<bool>(terminalFold);
}

bool isClosedRankNFoldMapRegion(RegionOp region,
                                ArrayRef<Operation *> algebra) {
  llvm::SmallDenseSet<Value, 4> parameterRoots;
  for (Operation *operation : algebra) {
    auto map = dyn_cast<MapOp>(operation);
    if (!map) {
      continue;
    }
    std::optional<unsigned> expansions =
        closedRankNExpandingSingletonMaps(map);
    if (!expansions || *expansions < 2) {
      continue;
    }
    std::optional<Value> root = projectedParameterRoot(map);
    if (!root) {
      return false;
    }
    parameterRoots.insert(*root);
  }
  auto isExternalInput = [&](Value value) {
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    return value.getDefiningOp() == nullptr && tensor &&
           tensor.hasStaticShape() && tensor.getRank() >= 1 &&
           tensor.getRank() <= 8 && tensor.getElementType().isBF16() &&
           (tensor.getRank() >= 3 || parameterRoots.contains(value)) &&
           llvm::all_of(tensor.getShape(),
                        [](int64_t extent) { return extent > 0; });
  };
  auto isExternalResult = [](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    return tensor && tensor.hasStaticShape() && tensor.getRank() >= 1 &&
           tensor.getRank() <= 8 && tensor.getElementType().isBF16() &&
           llvm::all_of(tensor.getShape(),
                        [](int64_t extent) { return extent > 0; });
  };
  if (!llvm::all_of(region.getInputs(), isExternalInput) ||
      !llvm::all_of(region.getResults(), [&](Value result) {
        return isExternalResult(result.getType());
      })) {
    return false;
  }
  auto regionYield = dyn_cast<YieldOp>(region.getBody().front().getTerminator());
  if (!regionYield ||
      regionYield.getValues().size() != region.getResults().size()) {
    return false;
  }
  for (auto [result, yielded] :
       llvm::zip_equal(region.getResults(), regionYield.getValues())) {
    auto type = cast<RankedTensorType>(result.getType());
    if (type.getRank() < 3) {
      auto map = yielded.getDefiningOp<MapOp>();
      if (!map || !isClosedFullPrefixBf16Result(map)) {
        return false;
      }
    }
  }
  return llvm::all_of(algebra, [](Operation *operation) {
    if (auto fold = dyn_cast<FoldOp>(operation)) {
      auto input =
          fold.getInputs().size() == 1
              ? dyn_cast<RankedTensorType>(fold.getInputs()[0].getType())
              : RankedTensorType{};
      return input && input.getRank() >= 3 && input.getRank() <= 8 &&
             isClosedRankNFold(fold);
    }
    auto map = dyn_cast<MapOp>(operation);
    if (!map) {
      return false;
    }
    std::optional<unsigned> expansions =
        closedRankNExpandingSingletonMaps(map);
    if (!expansions) {
      return isClosedDirectProjectedMap(map);
    }
    return *expansions == 1
               ? isExpansionDrivenByTerminalFold(map)
               : projectedParameterRoot(map).has_value();
  });
}

struct PlannedTask {
  Operation *operation;
  MaterializationTaskKind kind;
  SmallVector<int64_t> domain;
  SmallVector<int64_t> reductionDimensions;
  bool orderFree = false;
  SmallVector<int64_t> inputs;
  SmallVector<int64_t> outputs;
  SmallVector<int64_t> dependencies;
  SmallVector<Operation *> fusedOperations;
  SmallVector<AffineMap> composedInputMaps;
  Operation *fusedFold = nullptr;
  SmallVector<Operation *> foldRegionSources;
  SmallVector<Operation *> foldRegionPhases;
  struct BoundaryUse {
    int64_t inputBuffer;
    MapOp source;
    unsigned operand;
    AffineMap indexingMap;
    int64_t phase;
  };
  SmallVector<BoundaryUse> foldRegionBoundaryUses;
  SmallVector<Operation *> foldRegionRecomputedSources;
};

struct PlannedBuffer {
  Value value;
  std::optional<int64_t> producer;
  SmallVector<int64_t> consumers;
  bool liveOut = false;
};

struct DerivedPlan {
  SmallVector<PlannedTask, 0> tasks;
  SmallVector<PlannedBuffer, 0> buffers;
  bool hasContract = false;
  bool hasRankNFold = false;
  bool hasFoldRegion = false;
};

bool hasIdentityResultMap(MapOp map) {
  if (map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != map.getInputs().size() + 1)
    return false;
  auto resultMap =
      dyn_cast<AffineMapAttr>(map.getIndexingMaps().getValue().back());
  return resultMap && resultMap.getValue().isIdentity();
}

bool isEligibleF32Map(MapOp map) {
  if (!hasIdentityResultMap(map))
    return false;
  FailureOr<SmallVector<int64_t>> domain = mapDomainShape(map);
  if (failed(domain))
    return false;
  return llvm::all_of(
      llvm::concat<Type>(map.getOperandTypes(), map.getResultTypes()),
      [](Type type) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        return tensor && tensor.hasStaticShape() &&
               tensor.getElementType().isF32();
      });
}

bool isEligibleFoldMapEdge(FoldOp fold, MapOp map, unsigned operandNumber) {
  if (!isClosedAdditivePlanningFold(fold) ||
      !hasExactPositiveZeroInitializer(fold) ||
      fold.getNumResults() != 1 || !fold.getResult(0).hasOneUse() ||
      !isEligibleF32Map(map) || operandNumber >= map.getInputs().size())
    return false;
  auto folded = dyn_cast<RankedTensorType>(fold.getResult(0).getType());
  auto output = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  auto edge = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[operandNumber]);
  if (!folded || !output || !edge ||
      edge.getValue().getNumDims() != output.getRank() ||
      edge.getValue().getNumSymbols() != 0 ||
      edge.getValue().getNumResults() != folded.getRank())
    return false;
  auto elementCount = [](RankedTensorType type) -> FailureOr<int64_t> {
    int64_t count = 1;
    for (int64_t extent : type.getShape()) {
      if (extent <= 0 || count > std::numeric_limits<int64_t>::max() / extent)
        return failure();
      count *= extent;
    }
    return count;
  };
  FailureOr<int64_t> foldedElements = elementCount(folded);
  FailureOr<int64_t> outputElements = elementCount(output);
  if (failed(foldedElements) || failed(outputElements) ||
      *foldedElements != *outputElements)
    return false;
  llvm::SmallDenseSet<unsigned> mapped;
  std::optional<unsigned> previousDimension;
  for (auto [axis, expression] :
       llvm::enumerate(edge.getValue().getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() >= output.getRank() ||
        !mapped.insert(dimension.getPosition()).second ||
        (previousDimension &&
         dimension.getPosition() <= *previousDimension) ||
        folded.getDimSize(axis) !=
            output.getDimSize(dimension.getPosition()))
      return false;
    previousDimension = dimension.getPosition();
  }
  return llvm::all_of(llvm::enumerate(output.getShape()), [&](auto item) {
    return mapped.contains(item.index()) || item.value() == 1;
  });
}

std::string fusedMapFingerprint(ArrayRef<Operation *> operations) {
  std::string normalized("map-dag-v1\n");
  for (Operation *operation : operations) {
    normalized.append(semanticTaskFingerprint(operation));
    normalized.push_back('\n');
  }
  return sha256(normalized);
}

std::string fusedFoldMapFingerprint(FoldOp fold,
                                    ArrayRef<Operation *> maps) {
  std::string normalized("fold-map-v1\n");
  normalized.append(semanticTaskFingerprint(fold));
  normalized.push_back('\n');
  for (Operation *operation : maps) {
    normalized.append(semanticTaskFingerprint(operation));
    normalized.push_back('\n');
  }
  return sha256(normalized);
}

std::string foldRegionFingerprint(ArrayRef<Operation *> operations) {
  std::string normalized("fold-region-v2\n");
  llvm::SmallDenseMap<Operation *, int64_t> operationOrdinals;
  for (auto [ordinal, operation] : llvm::enumerate(operations))
    operationOrdinals.try_emplace(operation, ordinal);
  for (auto [ordinal, operation] : llvm::enumerate(operations)) {
    normalized.append(semanticTaskFingerprint(operation));
    normalized.push_back('\n');
    for (auto [operandOrdinal, operand] :
         llvm::enumerate(operation->getOperands())) {
      normalized.append((llvm::Twine("edge:") + llvm::Twine(ordinal) + ":" +
                         llvm::Twine(operandOrdinal) + ":")
                            .str());
      if (auto argument = dyn_cast<BlockArgument>(operand)) {
        normalized.append(
            (llvm::Twine("argument:") + llvm::Twine(argument.getArgNumber()))
                .str());
      } else if (auto result = dyn_cast<OpResult>(operand)) {
        Operation *producer = result.getOwner();
        auto producerOrdinal = operationOrdinals.find(producer);
        if (producerOrdinal != operationOrdinals.end()) {
          normalized.append((llvm::Twine("source:") +
                             llvm::Twine(producerOrdinal->second) + ":" +
                             llvm::Twine(result.getResultNumber()))
                                .str());
        } else {
          normalized.append("external:");
          if (Attribute source = producer->getAttr("source")) {
            std::string sourceText;
            llvm::raw_string_ostream sourceStream(sourceText);
            source.print(sourceStream);
            sourceStream.flush();
            normalized.append(sourceText);
          } else {
            normalized.append(producer->getName().getStringRef());
          }
          normalized.append(
              (llvm::Twine(":") + llvm::Twine(result.getResultNumber())).str());
        }
      } else {
        normalized.append("unknown");
      }
      normalized.push_back('\n');
    }
  }
  return sha256(normalized);
}

bool isClosedFoldRegionMap(MapOp map) {
  if (!hasIdentityResultMap(map) || map.getNumResults() != 1)
    return false;
  FailureOr<SmallVector<int64_t>> domain = genericMapDomainShape(map);
  if (failed(domain))
    return false;
  for (Type type :
       llvm::concat<Type>(map.getOperandTypes(), map.getResultTypes())) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() || tensor.getRank() > 8 ||
        (!tensor.getElementType().isF32() && !tensor.getElementType().isBF16()))
      return false;
  }
  return llvm::all_of(map.getIndexingMaps(), [](Attribute attribute) {
    auto affine = dyn_cast<AffineMapAttr>(attribute);
    return affine && affine.getValue().getNumSymbols() == 0;
  });
}

bool isCanonicalFoldResultExpansion(MapOp map, unsigned operandOrdinal) {
  auto input =
      dyn_cast<RankedTensorType>(map.getInputs()[operandOrdinal].getType());
  auto fold =
      dyn_cast_or_null<FoldOp>(map.getInputs()[operandOrdinal].getDefiningOp());
  FailureOr<SmallVector<int64_t>> domain = genericMapDomainShape(map);
  auto indexing =
      dyn_cast<AffineMapAttr>(map.getIndexingMaps()[operandOrdinal]);
  if (!input || !fold || failed(domain) || !indexing ||
      indexing.getValue().getNumSymbols() != 0 ||
      indexing.getValue().getNumResults() != input.getRank())
    return false;
  int64_t previousNonSingleton = -1;
  SmallVector<int64_t> mappedNonSingletonDimensions;
  for (auto [axis, expression] :
       llvm::enumerate(indexing.getValue().getResults())) {
    const int64_t extent = input.getDimSize(axis);
    if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
      if (extent != 1 || constant.getValue() != 0)
        return false;
      continue;
    }
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() >= domain->size() ||
        (*domain)[dimension.getPosition()] != extent)
      return false;
    if (extent != 1) {
      if (static_cast<int64_t>(dimension.getPosition()) <= previousNonSingleton)
        return false;
      previousNonSingleton = dimension.getPosition();
      mappedNonSingletonDimensions.push_back(dimension.getPosition());
    }
  }
  SmallVector<int64_t> domainNonSingletonDimensions;
  for (auto [axis, extent] : llvm::enumerate(*domain)) {
    if (extent != 1)
      domainNonSingletonDimensions.push_back(axis);
  }
  if (mappedNonSingletonDimensions.size() > domainNonSingletonDimensions.size())
    return false;
  auto foldInput = cast<RankedTensorType>(fold.getInputs().front().getType());
  ArrayRef<int64_t> reduction = fold.getReductionDimensions();
  const bool row =
      reduction.size() == 1 && reduction.front() == foldInput.getRank() - 1;
  ArrayRef<int64_t> expected = domainNonSingletonDimensions;
  expected = row ? expected.take_front(mappedNonSingletonDimensions.size())
                 : expected.take_back(mappedNonSingletonDimensions.size());
  return ArrayRef<int64_t>(mappedNonSingletonDimensions) == expected;
}

bool valueDependsOnOperation(Value value, Operation *operation) {
  llvm::SmallDenseSet<Operation *> visited;
  SmallVector<Value> worklist{value};
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (producer == operation)
      return true;
    llvm::append_range(worklist, producer->getOperands());
  }
  return false;
}

FailureOr<DerivedPlan> deriveFoldRegionPlan(RegionOp region) {
  SmallVector<Operation *> algebra;
  SmallVector<FoldOp> folds;
  bool hasReciprocalRoot = false;
  for (Operation &operation : region.getBody().front().without_terminator()) {
    if (auto map = dyn_cast<MapOp>(operation)) {
      if (!isClosedFoldRegionMap(map))
        return failure();
      map.walk([&](math::RsqrtOp) { hasReciprocalRoot = true; });
    } else if (auto fold = dyn_cast<FoldOp>(operation)) {
      if (!isClosedAdditivePlanningFold(fold) ||
          !hasExactPositiveZeroInitializer(fold) || fold.getNumResults() != 1)
        return failure();
      folds.push_back(fold);
    } else {
      return failure();
    }
    algebra.push_back(&operation);
  }
  if (folds.empty() || !hasReciprocalRoot)
    return failure();
  auto regionYield =
      dyn_cast<YieldOp>(region.getBody().front().getTerminator());
  if (!regionYield || regionYield.getValues().empty())
    return failure();

  FoldOp primary;
  for (FoldOp fold : folds) {
    if (llvm::all_of(regionYield.getValues(), [&](Value yielded) {
          return valueDependsOnOperation(yielded, fold);
        })) {
      primary = fold;
      break;
    }
  }
  if (!primary ||
      llvm::is_contained(regionYield.getValues(), primary.getResult(0)))
    return failure();

  auto collectSources = [&](Value root, llvm::SmallDenseSet<Operation *> &set,
                            bool stopAtPrimary, auto &self) -> LogicalResult {
    Operation *producer = root.getDefiningOp();
    if (!producer)
      return success();
    if (producer == primary && stopAtPrimary)
      return success();
    if (!isa<MapOp, FoldOp>(producer) ||
        producer->getParentOfType<RegionOp>() != region)
      return failure();
    if (!set.insert(producer).second)
      return success();
    for (Value operand : producer->getOperands()) {
      if (failed(self(operand, set, stopAtPrimary, self)))
        return failure();
    }
    return success();
  };

  SmallVector<llvm::SmallDenseSet<Operation *>> sourceSets;
  sourceSets.emplace_back();
  for (Value operand : primary->getOperands()) {
    if (failed(
            collectSources(operand, sourceSets.front(), false, collectSources)))
      return failure();
  }
  sourceSets.front().insert(primary);
  for (Value yielded : regionYield.getValues()) {
    llvm::SmallDenseSet<Operation *> branch;
    if (failed(collectSources(yielded, branch, true, collectSources)))
      return failure();
    sourceSets.push_back(std::move(branch));
  }
  if (regionYield.getValues().size() == 1) {
    sourceSets.front().insert(sourceSets.back().begin(),
                              sourceSets.back().end());
    sourceSets.pop_back();
  }

  llvm::SmallDenseSet<Operation *> covered;
  for (const auto &set : sourceSets)
    covered.insert(set.begin(), set.end());
  if (covered.size() != algebra.size())
    return failure();

  SmallVector<SmallVector<Operation *>> orderedSources(sourceSets.size());
  for (auto [taskOrdinal, set] : llvm::enumerate(sourceSets)) {
    for (Operation *operation : algebra) {
      if (set.contains(operation))
        orderedSources[taskOrdinal].push_back(operation);
    }
    SmallVector<FoldOp> phases;
    for (Operation *operation : orderedSources[taskOrdinal]) {
      if (auto fold = dyn_cast<FoldOp>(operation))
        phases.push_back(fold);
    }
    if (phases.empty())
      return failure();
    for (auto [phaseOrdinal, fold] : llvm::enumerate(phases)) {
      auto input = cast<RankedTensorType>(fold.getInputs().front().getType());
      if (llvm::any_of(fold.getInputs(), [&](Value value) {
            return !set.contains(value.getDefiningOp());
          }))
        return failure();
      ArrayRef<int64_t> reduction = fold.getReductionDimensions();
      const bool row =
          reduction.size() == 1 && reduction.front() == input.getRank() - 1;
      const bool column =
          reduction.size() == static_cast<size_t>(input.getRank() - 1) &&
          llvm::all_of(llvm::enumerate(reduction),
                       [](auto axis) { return axis.index() == axis.value(); });
      if (!row && !column)
        return failure();
      if (phaseOrdinal == 0)
        continue;
      int64_t cardinality = 1;
      for (int64_t axis : reduction)
        cardinality *= input.getDimSize(axis);
      FoldOp previous = phases[phaseOrdinal - 1];
      if (cardinality != 1 ||
          fold.getResult(0).getType() != previous.getResult(0).getType() ||
          !valueDependsOnOperation(fold.getInputs().front(), previous))
        return failure();
    }
    for (Operation *operation : orderedSources[taskOrdinal]) {
      auto map = dyn_cast<MapOp>(operation);
      if (!map)
        continue;
      for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
        if (isa_and_nonnull<FoldOp>(operand.getDefiningOp()) &&
            set.contains(operand.getDefiningOp()) &&
            !isCanonicalFoldResultExpansion(map, operandOrdinal))
          return failure();
      }
    }
  }

  DerivedPlan result;
  result.hasRankNFold = true;
  llvm::SmallDenseMap<Value, int64_t> bufferForValue;
  for (BlockArgument argument : region.getBody().front().getArguments()) {
    bufferForValue.try_emplace(argument, result.buffers.size());
    result.buffers.push_back({argument, std::nullopt, {}, false});
  }
  if (sourceSets.size() > 1) {
    bufferForValue.try_emplace(primary.getResult(0), result.buffers.size());
    result.buffers.push_back({primary.getResult(0), int64_t{0}, {}, false});
  }
  for (auto [outputOrdinal, yielded] :
       llvm::enumerate(regionYield.getValues())) {
    int64_t producer = sourceSets.size() == 1 ? 0 : outputOrdinal + 1;
    bufferForValue.try_emplace(yielded, result.buffers.size());
    result.buffers.push_back({yielded, producer, {}, true});
  }

  llvm::SmallDenseMap<Operation *, int64_t> sourceOccurrences;
  for (const auto &set : sourceSets) {
    for (Operation *operation : set)
      ++sourceOccurrences[operation];
  }
  if (llvm::any_of(sourceOccurrences, [](const auto &occurrence) {
        return occurrence.second > 1 && !isa<MapOp>(occurrence.first);
      }))
    return failure();
  for (auto [taskOrdinal, sources] : llvm::enumerate(orderedSources)) {
    PlannedTask task;
    task.foldRegionSources = sources;
    llvm::SmallDenseSet<Operation *> taskSet;
    taskSet.insert(sources.begin(), sources.end());
    for (Operation *operation : sources) {
      if (isa<FoldOp>(operation))
        task.foldRegionPhases.push_back(operation);
      if (sourceOccurrences.lookup(operation) > 1)
        task.foldRegionRecomputedSources.push_back(operation);
    }
    task.operation = task.foldRegionPhases.front();
    task.kind = MaterializationTaskKind::Fold;
    auto firstFold = cast<FoldOp>(task.operation);
    task.domain.assign(
        cast<RankedTensorType>(firstFold.getInputs().front().getType())
            .getShape()
            .begin(),
        cast<RankedTensorType>(firstFold.getInputs().front().getType())
            .getShape()
            .end());
    task.reductionDimensions.assign(firstFold.getReductionDimensions().begin(),
                                    firstFold.getReductionDimensions().end());
    task.orderFree = firstFold.getOrderFree();

    auto addInput = [&](int64_t ordinal) {
      if (!llvm::is_contained(task.inputs, ordinal))
        task.inputs.push_back(ordinal);
    };
    for (Operation *operation : sources) {
      auto map = dyn_cast<MapOp>(operation);
      if (!map)
        continue;
      for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
        if (taskSet.contains(operand.getDefiningOp()))
          continue;
        auto buffer = bufferForValue.find(operand);
        if (buffer == bufferForValue.end())
          return failure();
        addInput(buffer->second);
        int64_t phase = 0;
        for (Operation *phaseOperation : task.foldRegionPhases) {
          auto phaseSource = cast<FoldOp>(phaseOperation).getSource();
          if (std::tuple(phaseSource.getFunctionOrdinal(),
                         phaseSource.getBlockOrdinal(),
                         phaseSource.getOperationOrdinal(),
                         phaseSource.getResultOrdinal()) <
              std::tuple(map.getSource().getFunctionOrdinal(),
                         map.getSource().getBlockOrdinal(),
                         map.getSource().getOperationOrdinal(),
                         map.getSource().getResultOrdinal()))
            ++phase;
        }
        phase = std::min<int64_t>(phase, task.foldRegionPhases.size() - 1);
        task.foldRegionBoundaryUses.push_back(
            {buffer->second, map, static_cast<unsigned>(operandOrdinal),
             cast<AffineMapAttr>(map.getIndexingMaps()[operandOrdinal])
                 .getValue(),
             phase});
      }
    }
    if (task.foldRegionBoundaryUses.empty())
      return failure();
    if (sourceSets.size() == 1) {
      task.outputs.push_back(bufferForValue.lookup(regionYield.getValues()[0]));
    } else if (taskOrdinal == 0) {
      task.outputs.push_back(bufferForValue.lookup(primary.getResult(0)));
    } else {
      task.outputs.push_back(
          bufferForValue.lookup(regionYield.getValues()[taskOrdinal - 1]));
      task.dependencies.push_back(0);
    }
    result.tasks.push_back(std::move(task));
  }
  for (auto [taskOrdinal, task] : llvm::enumerate(result.tasks)) {
    for (int64_t input : task.inputs)
      result.buffers[input].consumers.push_back(taskOrdinal);
  }
  result.hasFoldRegion = true;
  return result;
}

FailureOr<DerivedPlan> fuseEligibleMapTasks(RegionOp region,
                                            DerivedPlan initial) {
  SmallVector<Operation *> algebra;
  for (Operation &operation : region.getBody().front().without_terminator())
    algebra.push_back(&operation);
  llvm::SmallDenseMap<Operation *, unsigned> index;
  for (auto [ordinal, operation] : llvm::enumerate(algebra))
    index.try_emplace(operation, ordinal);
  SmallVector<unsigned> parent(algebra.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](unsigned value) {
    while (parent[value] != value) {
      parent[value] = parent[parent[value]];
      value = parent[value];
    }
    return value;
  };
  auto merge = [&](unsigned left, unsigned right) {
    left = root(left);
    right = root(right);
    if (left != right)
      parent[left] = right;
  };
  for (auto [ordinal, operation] : llvm::enumerate(algebra)) {
    auto producer = dyn_cast<MapOp>(operation);
    if (!producer || !isEligibleF32Map(producer) ||
        !producer.getResult(0).hasOneUse())
      continue;
    OpOperand &use = *producer.getResult(0).getUses().begin();
    auto consumer = dyn_cast<MapOp>(use.getOwner());
    if (!consumer || !isEligibleF32Map(consumer))
      continue;
    auto consumerIndex = index.find(consumer);
    if (consumerIndex == index.end())
      continue;
    AffineMap edge = cast<AffineMapAttr>(
                         consumer.getIndexingMaps()[use.getOperandNumber()])
                         .getValue();
    FailureOr<SmallVector<int64_t>> producerDomain = mapDomainShape(producer);
    FailureOr<SmallVector<int64_t>> consumerDomain = mapDomainShape(consumer);
    if (!edge.isIdentity() || failed(producerDomain) ||
        failed(consumerDomain) || *producerDomain != *consumerDomain)
      continue;
    merge(ordinal, consumerIndex->second);
  }
  for (auto [ordinal, operation] : llvm::enumerate(algebra)) {
    auto fold = dyn_cast<FoldOp>(operation);
    if (!fold || fold.getNumResults() != 1 ||
        !fold.getResult(0).hasOneUse())
      continue;
    OpOperand &use = *fold.getResult(0).getUses().begin();
    auto consumer = dyn_cast<MapOp>(use.getOwner());
    auto consumerIndex = consumer ? index.find(consumer) : index.end();
    if (!consumer || consumerIndex == index.end() ||
        !isEligibleFoldMapEdge(fold, consumer, use.getOperandNumber()))
      continue;
    const unsigned consumerRoot = root(consumerIndex->second);
    const bool anotherFoldBoundary = llvm::any_of(
        llvm::enumerate(algebra), [&](auto member) {
          auto memberMap = dyn_cast<MapOp>(member.value());
          if (!memberMap || root(member.index()) != consumerRoot)
            return false;
          return llvm::any_of(memberMap.getInputs(), [&](Value input) {
            auto other = input.getDefiningOp<FoldOp>();
            return other && other != fold;
          });
        });
    if (!anotherFoldBoundary)
      merge(ordinal, consumerIndex->second);
  }

  SmallVector<SmallVector<unsigned>> members(algebra.size());
  for (unsigned ordinal = 0; ordinal < algebra.size(); ++ordinal)
    members[root(ordinal)].push_back(ordinal);

  llvm::SmallDenseMap<Operation *, PlannedTask *> initialTask;
  for (PlannedTask &task : initial.tasks)
    initialTask.try_emplace(task.operation, &task);
  DerivedPlan fused;
  fused.hasContract = initial.hasContract;
  fused.hasRankNFold = initial.hasRankNFold;
  llvm::SmallDenseMap<Value, int64_t> bufferForValue;
  for (BlockArgument input : region.getBody().front().getArguments()) {
    bufferForValue.try_emplace(input, fused.buffers.size());
    fused.buffers.push_back({input, std::nullopt, {}, false});
  }
  for (auto [ordinal, operation] : llvm::enumerate(algebra)) {
    unsigned groupRoot = root(ordinal);
    ArrayRef<unsigned> group = members[groupRoot];
    if (ordinal != group.back())
      continue;
    PlannedTask task;
    Operation *foldMember = nullptr;
    for (unsigned member : group) {
      if (!isa<FoldOp>(algebra[member]))
        continue;
      if (foldMember)
        return failure();
      foldMember = algebra[member];
    }
    task.operation = foldMember ? foldMember : operation;
    task.kind = initialTask.lookup(task.operation)->kind;
    task.domain = initialTask.lookup(task.operation)->domain;
    task.reductionDimensions =
        initialTask.lookup(task.operation)->reductionDimensions;
    task.orderFree = initialTask.lookup(task.operation)->orderFree;
    llvm::SmallDenseSet<Operation *> groupSet;
    for (unsigned member : group)
      groupSet.insert(algebra[member]);
    if (group.size() > 1) {
      for (unsigned member : group) {
        Operation *memberOperation = algebra[member];
        if (isa<MapOp>(memberOperation))
          task.fusedOperations.push_back(memberOperation);
        else if (!foldMember || memberOperation != foldMember)
          return failure();
      }
      if (foldMember) {
        if (task.fusedOperations.empty())
          return failure();
        task.fusedFold = foldMember;
      } else if (task.fusedOperations.size() != group.size()) {
        return failure();
      }
    }
    unsigned singletonMember = ordinal;
    ArrayRef<unsigned> taskMembers =
        task.fusedOperations.empty() ? ArrayRef<unsigned>(singletonMember)
                                     : group;
    llvm::SmallDenseMap<Value, AffineMap> boundaryMaps;
    const bool fusedGroup = !task.fusedOperations.empty();
    auto addInput = [&](Value operand) -> LogicalResult {
      auto found = bufferForValue.find(operand);
      if (found == bufferForValue.end())
        return failure();
      task.inputs.push_back(found->second);
      PlannedBuffer &buffer = fused.buffers[found->second];
      if (buffer.consumers.empty() ||
          buffer.consumers.back() !=
              static_cast<int64_t>(fused.tasks.size()))
        buffer.consumers.push_back(fused.tasks.size());
      if (buffer.producer &&
          !llvm::is_contained(task.dependencies, *buffer.producer))
        task.dependencies.push_back(*buffer.producer);
      return success();
    };
    if (foldMember) {
      for (Value operand : foldMember->getOperands()) {
        if (failed(addInput(operand)))
          return failure();
      }
    }
    for (unsigned member : llvm::reverse(taskMembers)) {
      Operation *memberOperation = algebra[member];
      auto memberMap = dyn_cast<MapOp>(memberOperation);
      for (auto [operandOrdinal, operand] :
           llvm::enumerate(memberOperation->getOperands())) {
        if (groupSet.contains(operand.getDefiningOp()))
          continue;
        if (foldMember && memberOperation == foldMember)
          continue;
        AffineMap composedMap;
        if (memberMap)
          composedMap = cast<AffineMapAttr>(
                            memberMap.getIndexingMaps()[operandOrdinal])
                            .getValue();
        auto prior = boundaryMaps.find(operand);
        if (fusedGroup && prior != boundaryMaps.end()) {
          if (memberMap && prior->second != composedMap)
            return failure();
          continue;
        }
        if (fusedGroup && memberMap)
          boundaryMaps.try_emplace(operand, composedMap);
        if (failed(addInput(operand)))
          return failure();
        if (fusedGroup && memberMap)
          task.composedInputMaps.push_back(composedMap);
      }
    }
    llvm::sort(task.dependencies);
    for (unsigned member : taskMembers) {
      for (Value result : algebra[member]->getResults()) {
        const bool internal = llvm::all_of(result.getUses(), [&](OpOperand &use) {
          return groupSet.contains(use.getOwner());
        });
        if (internal && !result.use_empty())
          continue;
        int64_t bufferOrdinal = fused.buffers.size();
        bufferForValue.try_emplace(result, bufferOrdinal);
        fused.buffers.push_back(
            {result, static_cast<int64_t>(fused.tasks.size()), {}, false});
        task.outputs.push_back(bufferOrdinal);
      }
    }
    if (task.outputs.empty())
      return failure();
    fused.tasks.push_back(std::move(task));
  }
  auto regionYield = cast<YieldOp>(region.getBody().front().getTerminator());
  for (Value value : regionYield.getValues()) {
    auto found = bufferForValue.find(value);
    if (found == bufferForValue.end())
      return failure();
    fused.buffers[found->second].liveOut = true;
  }
  for (PlannedBuffer &buffer : fused.buffers) {
    if (!buffer.liveOut && buffer.consumers.empty())
      return failure();
  }
  return fused;
}

FailureOr<DerivedPlan> deriveGpuPlan(RegionOp region,
                                     bool preferFusion = false) {
  SmallVector<Operation *> algebra;
  for (Operation &operation : region.getBody().front().without_terminator()) {
    if (!isa<MapOp, FoldOp, ContractOp>(operation)) {
      return failure();
    }
    algebra.push_back(&operation);
  }
  if (algebra.empty()) {
    return failure();
  }
  const bool containsContract = llvm::any_of(
      algebra, [](Operation *operation) { return isa<ContractOp>(operation); });
  if (!preferFusion && containsContract &&
      !isClosedContractMapRegion(region, algebra)) {
    return failure();
  }
  const bool containsRankNFold =
      llvm::any_of(algebra, [](Operation *operation) {
        auto fold = dyn_cast<FoldOp>(operation);
        auto input =
            fold && fold.getInputs().size() == 1
                ? dyn_cast<RankedTensorType>(fold.getInputs()[0].getType())
                : RankedTensorType{};
        return input && input.getRank() > 2;
      });
  if (!preferFusion && containsRankNFold &&
      (containsContract || !isClosedRankNFoldMapRegion(region, algebra))) {
    return failure();
  }
  SmallVector<PlannedBuffer, 0> buffers;
  llvm::SmallDenseMap<Value, int64_t> bufferForValue;
  for (BlockArgument input : region.getBody().front().getArguments()) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    if (!type || !type.hasStaticShape() ||
        llvm::any_of(type.getShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      return failure();
    }
    bufferForValue.try_emplace(input, buffers.size());
    buffers.push_back({input, std::nullopt, {}, false});
  }

  SmallVector<PlannedTask, 0> tasks;
  unsigned foldCount = 0;
  unsigned contractCount = 0;
  for (Operation *operation : algebra) {
    PlannedTask task;
    task.operation = operation;
    task.kind =
        isa<FoldOp>(operation)
            ? MaterializationTaskKind::Fold
            : (isa<ContractOp>(operation) ? MaterializationTaskKind::Contract
                                          : MaterializationTaskKind::Map);
    if (auto fold = dyn_cast<FoldOp>(operation)) {
      const bool closedFold =
          preferFusion ? isClosedAdditivePlanningFold(fold)
                       : (containsRankNFold ? isClosedRankNFold(fold)
                                            : isClosedRankTwoFold(fold));
      if (!closedFold) {
        return failure();
      }
      ++foldCount;
      task.reductionDimensions.assign(fold.getReductionDimensions().begin(),
                                      fold.getReductionDimensions().end());
      task.orderFree = fold.getOrderFree();
      task.domain.assign(cast<RankedTensorType>(fold.getInputs()[0].getType())
                             .getShape()
                             .begin(),
                         cast<RankedTensorType>(fold.getInputs()[0].getType())
                             .getShape()
                             .end());
    } else if (auto contract = dyn_cast<ContractOp>(operation)) {
      if (!isClosedProjectedContract(contract)) {
        return failure();
      }
      ++contractCount;
      FailureOr<SmallVector<int64_t>> domain = contractDomainShape(contract);
      if (failed(domain)) {
        return failure();
      }
      task.domain = std::move(*domain);
      for (auto [position, iterator] :
           llvm::enumerate(contract.getIteratorKinds())) {
        if (cast<StringAttr>(iterator).getValue() == "reduction") {
          task.reductionDimensions.push_back(position);
        }
      }
    } else {
      FailureOr<SmallVector<int64_t>> domain = preferFusion
                                                   ? genericMapDomainShape(
                                                         cast<MapOp>(operation))
                                                   : mapDomainShape(
                                                         cast<MapOp>(operation));
      if (failed(domain)) {
        return failure();
      }
      task.domain = std::move(*domain);
    }
    for (Value operand : operation->getOperands()) {
      auto found = bufferForValue.find(operand);
      if (found == bufferForValue.end()) {
        return failure();
      }
      task.inputs.push_back(found->second);
      if (buffers[found->second].consumers.empty() ||
          buffers[found->second].consumers.back() !=
              static_cast<int64_t>(tasks.size())) {
        buffers[found->second].consumers.push_back(tasks.size());
      }
      if (buffers[found->second].producer &&
          !llvm::is_contained(task.dependencies,
                              *buffers[found->second].producer)) {
        task.dependencies.push_back(*buffers[found->second].producer);
      }
    }
    llvm::sort(task.dependencies);
    for (Value result : operation->getResults()) {
      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type || !type.hasStaticShape() ||
          llvm::any_of(type.getShape(),
                       [](int64_t extent) { return extent <= 0; })) {
        return failure();
      }
      int64_t ordinal = buffers.size();
      bufferForValue.try_emplace(result, ordinal);
      buffers.push_back(
          {result, static_cast<int64_t>(tasks.size()), {}, false});
      task.outputs.push_back(ordinal);
    }
    tasks.push_back(std::move(task));
  }
  if (foldCount == 0 && contractCount == 0) {
    return failure();
  }
  auto regionYield = cast<YieldOp>(region.getBody().front().getTerminator());
  for (Value value : regionYield.getValues()) {
    auto found = bufferForValue.find(value);
    if (found == bufferForValue.end()) {
      return failure();
    }
    buffers[found->second].liveOut = true;
  }
  for (PlannedBuffer &buffer : buffers) {
    if (!buffer.liveOut && buffer.consumers.empty()) {
      return failure();
    }
  }

  llvm::SmallDenseSet<Operation *> connected;
  SmallVector<Operation *> worklist;
  for (Operation *operation : algebra) {
    if (isa<FoldOp, ContractOp>(operation)) {
      connected.insert(operation);
      worklist.push_back(operation);
      break;
    }
  }
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    for (Operation *candidate : algebra) {
      bool adjacent = llvm::any_of(candidate->getOperands(),
                                   [&](Value value) {
                                     return value.getDefiningOp() == operation;
                                   }) ||
                      llvm::any_of(operation->getOperands(), [&](Value value) {
                        return value.getDefiningOp() == candidate;
                      });
      if (adjacent && connected.insert(candidate).second) {
        worklist.push_back(candidate);
      }
    }
  }
  if (connected.size() != algebra.size() && !preferFusion) {
    return failure();
  }
  DerivedPlan result{std::move(tasks), std::move(buffers),
                     contractCount != 0, containsRankNFold};
  if (!preferFusion)
    return result;
  FailureOr<DerivedPlan> foldRegion = deriveFoldRegionPlan(region);
  if (succeeded(foldRegion))
    return foldRegion;
  return fuseEligibleMapTasks(region, std::move(result));
}

Operation *createMaterializationRecord(OpBuilder &builder, Location location,
                                       StringRef name,
                                       ArrayRef<NamedAttribute> attrs) {
  OperationState state(location, name);
  state.addAttributes(attrs);
  return builder.create(state);
}

class PlanRowFoldMaterializationPass
    : public impl::ShuttlePlanRowFoldMaterializationPassBase<
          PlanRowFoldMaterializationPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module.getOps<MaterializationPlanOp>().empty()) {
      module.emitError("already contains a materialization plan");
      return signalPassFailure();
    }
    auto manifest =
        module->getAttrOfType<DictionaryAttr>(kCoverageManifestAttribute);
    auto tuning = manifest ? manifest.getAs<StringAttr>("canonical_tuning")
                           : StringAttr{};
    auto canonicalOptions =
        manifest ? manifest.getAs<StringAttr>("canonical_options")
                 : StringAttr{};
    const bool abi22 =
        !canonicalOptions ||
        canonicalOptions.getValue().contains(
            "\"pipeline_abi_version\":22,\"schema_version\":1");
    const bool validTuning =
        tuning && (tuning.getValue() == kDefaultCanonicalTuning ||
                   tuning.getValue() == kFusionCanonicalTuning ||
                   tuning.getValue() == kMaterializationCanonicalTuning);
    if (!validTuning ||
        (!abi22 && tuning.getValue() != kDefaultCanonicalTuning)) {
      module.emitError(
          "requires an exact ABI21 default or ABI22 materialization tuning");
      return signalPassFailure();
    }
    const bool preferFusion = tuning.getValue() == kFusionCanonicalTuning;
    SmallVector<std::pair<RegionOp, DerivedPlan>> selected;
    bool invalidRegion = false;
    module.walk([&](RegionOp region) {
      FailureOr<DerivedPlan> candidate = deriveGpuPlan(region, preferFusion);
      if (failed(candidate)) {
        invalidRegion = true;
        return WalkResult::interrupt();
      }
      selected.emplace_back(region, std::move(*candidate));
      return WalkResult::advance();
    });
    uint64_t aggregateTasks = 0;
    uint64_t aggregateBuffers = 0;
    for (const auto &[region, derived] : selected) {
      (void)region;
      aggregateTasks += derived.tasks.size();
      aggregateBuffers += derived.buffers.size();
    }
    if (invalidRegion || selected.empty() ||
        selected.size() > kMaximumGpuExecutableRecords ||
        aggregateTasks > kMaximumGpuExecutableRecords ||
        aggregateBuffers > kMaximumGpuExecutableRecords) {
      module.emitError(
          "requires a bounded set of independently connected static GPU "
          "algebra regions");
      return signalPassFailure();
    }
    const bool mixesContractAndRankNFold =
        llvm::any_of(
            selected,
            [](const auto &entry) { return entry.second.hasContract; }) &&
        llvm::any_of(selected, [](const auto &entry) {
          return entry.second.hasRankNFold;
        });
    if (mixesContractAndRankNFold) {
      module.emitError(
          "requires a bounded set of independently connected static GPU "
          "algebra regions");
      return signalPassFailure();
    }
    for (auto &[selectedRegion, derived] : selected) {
      OpBuilder builder(module.getContext());
      builder.setInsertionPointToEnd(module.getBody());
      OperationState planState(module.getLoc(),
                               MaterializationPlanOp::getOperationName());
      planState.addAttribute(
          "schema_version",
          builder.getI64IntegerAttr(
              derived.hasContract || derived.hasFoldRegion ? 2 : 1));
      planState.addAttribute("policy", selectedRegion.getPolicyAttr());
      planState.addAttribute("fingerprint",
                             builder.getStringAttr(std::string(64, '0')));
      planState.addRegion();
      auto plan = cast<MaterializationPlanOp>(builder.create(planState));
      Block *body = new Block();
      plan.getBody().push_back(body);
      builder.setInsertionPointToEnd(body);

      auto &tasks = derived.tasks;
      auto &buffers = derived.buffers;
      for (auto [ordinal, buffer] : llvm::enumerate(buffers)) {
        int64_t end =
            buffer.liveOut
                ? tasks.size()
                : (buffer.consumers.empty() ? 0 : buffer.consumers.back());
        SmallVector<NamedAttribute> attrs{
            builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr("tensor_type",
                                 TypeAttr::get(buffer.value.getType())),
            builder.getNamedAttr("storage",
                                 MaterializationStorageAttr::get(
                                     module.getContext(),
                                     (!buffer.producer || buffer.liveOut)
                                         ? MaterializationStorage::External
                                         : MaterializationStorage::Temporary)),
            builder.getNamedAttr("live_in",
                                 builder.getBoolAttr(!buffer.producer)),
            builder.getNamedAttr("live_out",
                                 builder.getBoolAttr(buffer.liveOut)),
            builder.getNamedAttr(
                "consumers",
                DenseI64ArrayAttr::get(module.getContext(), buffer.consumers)),
            builder.getNamedAttr(
                "lifetime_start",
                builder.getI64IntegerAttr(buffer.producer.value_or(0))),
            builder.getNamedAttr("lifetime_end",
                                 builder.getI64IntegerAttr(end))};
        if (buffer.producer) {
          attrs.push_back(builder.getNamedAttr(
              "producer", builder.getI64IntegerAttr(*buffer.producer)));
        }
        createMaterializationRecord(builder, module.getLoc(),
                                    MaterializationBufferOp::getOperationName(),
                                    attrs);
      }
      for (auto [ordinal, task] : llvm::enumerate(tasks)) {
        const bool fusedFoldMap = task.fusedFold != nullptr;
        const bool fusedMap = !fusedFoldMap && task.fusedOperations.size() > 1;
        const bool foldRegion = !task.foldRegionSources.empty();
        auto source = isa<MapOp>(task.operation)
                          ? cast<MapOp>(task.operation).getSource()
                      : isa<FoldOp>(task.operation)
                          ? cast<FoldOp>(task.operation).getSource()
                          : cast<ContractOp>(task.operation).getSource();
        SmallVector<NamedAttribute> attrs{
            builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr("kind", MaterializationTaskKindAttr::get(
                                             module.getContext(), task.kind)),
            builder.getNamedAttr(
                "domain_shape",
                DenseI64ArrayAttr::get(module.getContext(), task.domain)),
            builder.getNamedAttr(
                "reduction_dimensions",
                DenseI64ArrayAttr::get(module.getContext(),
                                       task.reductionDimensions)),
            builder.getNamedAttr(
                "input_buffers",
                DenseI64ArrayAttr::get(module.getContext(), task.inputs)),
            builder.getNamedAttr(
                "output_buffers",
                DenseI64ArrayAttr::get(module.getContext(), task.outputs)),
            builder.getNamedAttr(
                "dependencies",
                DenseI64ArrayAttr::get(module.getContext(), task.dependencies)),
            builder.getNamedAttr(
                "semantic_fingerprint",
                builder.getStringAttr(
                    foldRegion ? foldRegionFingerprint(task.foldRegionSources)
                    : fusedFoldMap
                        ? fusedFoldMapFingerprint(cast<FoldOp>(task.fusedFold),
                                                  task.fusedOperations)
                    : fusedMap ? fusedMapFingerprint(task.fusedOperations)
                               : semanticTaskFingerprint(task.operation))),
            builder.getNamedAttr("source", source)};
        if (foldRegion) {
          SmallVector<Attribute> sources;
          for (Operation *operation : task.foldRegionSources) {
            if (auto map = dyn_cast<MapOp>(operation))
              sources.push_back(map.getSource());
            else
              sources.push_back(cast<FoldOp>(operation).getSource());
          }
          SmallVector<Attribute> phases;
          for (auto [phaseOrdinal, operation] :
               llvm::enumerate(task.foldRegionPhases)) {
            auto fold = cast<FoldOp>(operation);
            auto domain =
                cast<RankedTensorType>(fold.getInputs().front().getType())
                    .getShape();
            phases.push_back(builder.getDictionaryAttr(
                {builder.getNamedAttr("source", fold.getSource()),
                 builder.getNamedAttr("domain_shape",
                                      builder.getDenseI64ArrayAttr(domain)),
                 builder.getNamedAttr("reduction_dimensions",
                                      builder.getDenseI64ArrayAttr(
                                          fold.getReductionDimensions())),
                 builder.getNamedAttr("order_free",
                                      builder.getBoolAttr(fold.getOrderFree())),
                 builder.getNamedAttr(
                     "transition",
                     builder.getStringAttr(phaseOrdinal == 0 ? "launch_boundary"
                                                             : "cta_local"))}));
          }
          SmallVector<Attribute> boundaryUses;
          for (PlannedTask::BoundaryUse &use : task.foldRegionBoundaryUses) {
            boundaryUses.push_back(builder.getDictionaryAttr(
                {builder.getNamedAttr("input_buffer", builder.getI64IntegerAttr(
                                                          use.inputBuffer)),
                 builder.getNamedAttr("source", use.source.getSource()),
                 builder.getNamedAttr("operand",
                                      builder.getI64IntegerAttr(use.operand)),
                 builder.getNamedAttr("indexing_map",
                                      AffineMapAttr::get(use.indexingMap)),
                 builder.getNamedAttr("phase",
                                      builder.getI64IntegerAttr(use.phase))}));
          }
          attrs.push_back(builder.getNamedAttr("fold_region_sources",
                                               builder.getArrayAttr(sources)));
          attrs.push_back(builder.getNamedAttr("fold_region_phases",
                                               builder.getArrayAttr(phases)));
          attrs.push_back(builder.getNamedAttr(
              "fold_region_boundary_uses", builder.getArrayAttr(boundaryUses)));
          if (!task.foldRegionRecomputedSources.empty()) {
            SmallVector<Attribute> recomputed;
            for (Operation *operation : task.foldRegionRecomputedSources)
              recomputed.push_back(cast<MapOp>(operation).getSource());
            attrs.push_back(
                builder.getNamedAttr("fold_region_recomputed_sources",
                                     builder.getArrayAttr(recomputed)));
          }
        } else if (fusedFoldMap) {
          SmallVector<Attribute> sources;
          for (Operation *operation : task.fusedOperations)
            sources.push_back(cast<MapOp>(operation).getSource());
          SmallVector<Attribute> maps;
          for (AffineMap map : task.composedInputMaps)
            maps.push_back(AffineMapAttr::get(map));
          attrs.push_back(builder.getNamedAttr(
              "fold_epilogue_sources", builder.getArrayAttr(sources)));
          attrs.push_back(builder.getNamedAttr(
              "epilogue_input_maps", builder.getArrayAttr(maps)));
        } else if (fusedMap) {
          SmallVector<Attribute> sources;
          for (Operation *operation : task.fusedOperations)
            sources.push_back(cast<MapOp>(operation).getSource());
          SmallVector<Attribute> maps;
          for (AffineMap map : task.composedInputMaps)
            maps.push_back(AffineMapAttr::get(map));
          attrs.push_back(builder.getNamedAttr(
              "fused_sources", builder.getArrayAttr(sources)));
          attrs.push_back(builder.getNamedAttr(
              "composed_input_maps", builder.getArrayAttr(maps)));
        }
        if (task.kind == MaterializationTaskKind::Fold) {
          attrs.push_back(builder.getNamedAttr(
              "order_free", builder.getBoolAttr(task.orderFree)));
        }
        createMaterializationRecord(builder, task.operation->getLoc(),
                                    MaterializationTaskOp::getOperationName(),
                                    attrs);
      }
      builder.create<MaterializationPlanYieldOp>(module.getLoc());
      plan.setFingerprint(materializationPlanFingerprint(plan));
      if (failed(plan.verifyRegions())) {
        return signalPassFailure();
      }
    }
  }
};

LogicalResult
verifyMaterializationPlanAgainstSource(ModuleOp module,
                                       MaterializationPlanOp plan,
                                       RegionOp *boundRegion = nullptr) {
  if (failed(plan.verifyRegions())) {
    return failure();
  }
  llvm::SmallDenseMap<Attribute, Operation *> sourceOperations;
  llvm::SmallDenseSet<Attribute> duplicateSources;
  module.walk([&](Operation *operation) {
    if (auto map = dyn_cast<MapOp>(operation)) {
      if (!sourceOperations.try_emplace(map.getSource(), operation).second) {
        duplicateSources.insert(map.getSource());
      }
    } else if (auto fold = dyn_cast<FoldOp>(operation)) {
      if (!sourceOperations.try_emplace(fold.getSource(), operation).second) {
        duplicateSources.insert(fold.getSource());
      }
    } else if (auto contract = dyn_cast<ContractOp>(operation)) {
      if (!sourceOperations.try_emplace(contract.getSource(), operation)
               .second) {
        duplicateSources.insert(contract.getSource());
      }
    }
  });
  SmallVector<MaterializationBufferOp> buffers;
  SmallVector<MaterializationTaskOp> tasks;
  for (Operation &operation : plan.getBody().front()) {
    if (auto buffer = dyn_cast<MaterializationBufferOp>(operation)) {
      buffers.push_back(buffer);
    } else if (auto task = dyn_cast<MaterializationTaskOp>(operation)) {
      tasks.push_back(task);
    }
  }
  llvm::SmallDenseSet<Operation *> bound;
  llvm::SmallDenseMap<Operation *, int64_t> foldRegionOccurrences;
  llvm::SmallDenseMap<Operation *, int64_t> recomputationDeclarations;
  bool sawFoldRegion = false;
  for (MaterializationTaskOp task : tasks) {
    auto fusedSources = task->getAttrOfType<ArrayAttr>("fused_sources");
    auto composedMaps =
        task->getAttrOfType<ArrayAttr>("composed_input_maps");
    auto epilogueSources =
        task->getAttrOfType<ArrayAttr>("fold_epilogue_sources");
    auto epilogueMaps =
        task->getAttrOfType<ArrayAttr>("epilogue_input_maps");
    auto foldRegionSources =
        task->getAttrOfType<ArrayAttr>("fold_region_sources");
    if (foldRegionSources) {
      sawFoldRegion = true;
      if (fusedSources || composedMaps || epilogueSources || epilogueMaps ||
          task.getKind() != MaterializationTaskKind::Fold) {
        task.emitOpError("cannot mix Fold-region and legacy fusion metadata");
        return failure();
      }
      SmallVector<Operation *> operations;
      for (Attribute source : foldRegionSources) {
        auto found = sourceOperations.find(source);
        if (!isa<SourceRefAttr>(source) || found == sourceOperations.end() ||
            duplicateSources.contains(source) ||
            !isa<MapOp, FoldOp>(found->second)) {
          task.emitOpError("Fold-region sources must bind unique algebra");
          return failure();
        }
        operations.push_back(found->second);
        bound.insert(found->second);
        ++foldRegionOccurrences[found->second];
      }
      if (task.getSemanticFingerprint() != foldRegionFingerprint(operations)) {
        task.emitOpError("does not fingerprint every Fold-region source");
        return failure();
      }
      if (auto recomputed = task->getAttrOfType<ArrayAttr>(
              "fold_region_recomputed_sources")) {
        for (Attribute source : recomputed) {
          Operation *operation = sourceOperations.lookup(source);
          if (!operation || !isa<MapOp>(operation) ||
              !llvm::is_contained(operations, operation)) {
            task.emitOpError(
                "Fold-region recomputation must bind a task-local Map");
            return failure();
          }
          ++recomputationDeclarations[operation];
        }
      }
      continue;
    }
    if (sawFoldRegion) {
      task.emitOpError("cannot mix Fold-region and legacy tasks");
      return failure();
    }
    if (fusedSources && epilogueSources) {
      task.emitOpError("cannot mix Map fusion and Fold epilogue provenance");
      return failure();
    }
    auto found = sourceOperations.find(task.getSource());
    if (found == sourceOperations.end() ||
        duplicateSources.contains(task.getSource())) {
      task.emitOpError("source must uniquely bind one surviving algebra task");
      return failure();
    }
    Operation *source = found->second;
    SmallVector<Operation *> fusedOperations;
    if (fusedSources) {
      if (task.getKind() != MaterializationTaskKind::Map ||
          fusedSources.size() < 2 ||
          fusedSources.getValue().back() != task.getSource() ||
          !composedMaps ||
          composedMaps.size() != task.getInputBuffers().size()) {
        task.emitOpError(
            "fused Map provenance and composed inputs must be aligned");
        return failure();
      }
      for (Attribute fusedSource : fusedSources) {
        auto found = sourceOperations.find(fusedSource);
        if (!isa<SourceRefAttr>(fusedSource) ||
            found == sourceOperations.end() ||
            duplicateSources.contains(fusedSource) ||
            !isa<MapOp>(found->second) ||
            !bound.insert(found->second).second) {
          task.emitOpError(
              "fused_sources must uniquely bind ordered surviving Maps");
          return failure();
        }
        fusedOperations.push_back(found->second);
      }
      if (task.getSemanticFingerprint() !=
          fusedMapFingerprint(fusedOperations)) {
        task.emitOpError("does not fingerprint every fused Map semantic");
        return failure();
      }
      for (Attribute map : composedMaps) {
        if (!isa<AffineMapAttr>(map)) {
          task.emitOpError("composed_input_maps must contain affine maps");
          return failure();
        }
      }
    } else if (epilogueSources) {
      auto fold = dyn_cast<FoldOp>(source);
      if (!fold || task.getKind() != MaterializationTaskKind::Fold ||
          epilogueSources.empty() || !epilogueMaps ||
          task.getInputBuffers().size() < fold->getNumOperands() ||
          epilogueMaps.size() !=
              task.getInputBuffers().size() - fold->getNumOperands() ||
          !bound.insert(source).second) {
        task.emitOpError(
            "Fold epilogue provenance and composed inputs must be aligned");
        return failure();
      }
      for (Attribute epilogueSource : epilogueSources) {
        auto epilogue = sourceOperations.find(epilogueSource);
        if (!isa<SourceRefAttr>(epilogueSource) ||
            epilogue == sourceOperations.end() ||
            duplicateSources.contains(epilogueSource) ||
            !isa<MapOp>(epilogue->second) ||
            !bound.insert(epilogue->second).second) {
          task.emitOpError(
              "fold_epilogue_sources must uniquely bind ordered Maps");
          return failure();
        }
        fusedOperations.push_back(epilogue->second);
      }
      if (task.getSemanticFingerprint() !=
          fusedFoldMapFingerprint(fold, fusedOperations)) {
        task.emitOpError("does not fingerprint the Fold and every epilogue");
        return failure();
      }
      for (Attribute map : epilogueMaps) {
        if (!isa<AffineMapAttr>(map)) {
          task.emitOpError("epilogue_input_maps must contain affine maps");
          return failure();
        }
      }
    }
    if (!fusedSources && !epilogueSources && !bound.insert(source).second) {
      task.emitOpError("source must uniquely bind one surviving algebra task");
      return failure();
    }
    const bool kindMatches =
        (task.getKind() == MaterializationTaskKind::Map &&
         isa<MapOp>(source)) ||
        (task.getKind() == MaterializationTaskKind::Fold &&
         isa<FoldOp>(source)) ||
        (task.getKind() == MaterializationTaskKind::Contract &&
         isa<ContractOp>(source));
    const std::string expectedFingerprint =
        fusedSources
            ? fusedMapFingerprint(fusedOperations)
        : epilogueSources
            ? fusedFoldMapFingerprint(cast<FoldOp>(source), fusedOperations)
            : semanticTaskFingerprint(source);
    const bool arityMatches =
        fusedSources || epilogueSources ||
        (task.getInputBuffers().size() == source->getNumOperands() &&
         task.getOutputBuffers().size() == source->getNumResults());
    if (!kindMatches ||
        task.getSemanticFingerprint() != expectedFingerprint ||
        !arityMatches) {
      task.emitOpError("does not match its bound algebra semantics");
      return failure();
    }
    if (!fusedSources && !epilogueSources) {
      for (auto [bufferOrdinal, value] :
           llvm::zip_equal(task.getInputBuffers(), source->getOperands())) {
        if (buffers[bufferOrdinal].getTensorType() != value.getType()) {
          task.emitOpError("input buffer type does not match bound algebra");
          return failure();
        }
      }
      for (auto [bufferOrdinal, value] :
           llvm::zip_equal(task.getOutputBuffers(), source->getResults())) {
        if (buffers[bufferOrdinal].getTensorType() != value.getType()) {
          task.emitOpError("output buffer type does not match bound algebra");
          return failure();
        }
      }
    } else if (epilogueSources) {
      auto fold = cast<FoldOp>(source);
      for (auto [bufferOrdinal, value] : llvm::zip_equal(
               task.getInputBuffers().take_front(fold->getNumOperands()),
               source->getOperands())) {
        if (bufferOrdinal < 0 || bufferOrdinal >= buffers.size() ||
            buffers[bufferOrdinal].getTensorType() != value.getType()) {
          task.emitOpError("Fold input buffer type does not match algebra");
          return failure();
        }
      }
      if (task.getOutputBuffers().size() != 1 ||
          task.getOutputBuffers().front() < 0 ||
          task.getOutputBuffers().front() >= buffers.size() ||
          buffers[task.getOutputBuffers().front()].getTensorType() !=
              fusedOperations.back()->getResult(0).getType()) {
        task.emitOpError("Fold epilogue output does not match final Map");
        return failure();
      }
    }
    SmallVector<int64_t> expectedDomain;
    if (auto map = dyn_cast<MapOp>(source)) {
      FailureOr<SmallVector<int64_t>> domain = genericMapDomainShape(map);
      if (failed(domain)) {
        task.emitOpError("bound Map has no closed static domain");
        return failure();
      }
      expectedDomain = std::move(*domain);
    } else if (auto contract = dyn_cast<ContractOp>(source)) {
      FailureOr<SmallVector<int64_t>> domain = contractDomainShape(contract);
      if (failed(domain)) {
        task.emitOpError("bound Contract has no closed static domain");
        return failure();
      }
      expectedDomain = std::move(*domain);
    } else {
      auto fold = cast<FoldOp>(source);
      expectedDomain.assign(
          cast<RankedTensorType>(fold.getInputs()[0].getType())
              .getShape()
              .begin(),
          cast<RankedTensorType>(fold.getInputs()[0].getType())
              .getShape()
              .end());
    }
    if (task.getDomainShape() != ArrayRef<int64_t>(expectedDomain)) {
      task.emitOpError("domain does not match bound algebra indexing");
      return failure();
    }
  }
  if (sawFoldRegion) {
    for (auto [operation, occurrences] : foldRegionOccurrences) {
      const int64_t declarations = recomputationDeclarations.lookup(operation);
      if ((occurrences > 1 &&
           (!isa<MapOp>(operation) || declarations != occurrences)) ||
          (occurrences == 1 && declarations != 0)) {
        plan.emitOpError(
            "Fold-region recomputation must exactly declare repeated Maps");
        return failure();
      }
    }
  }
  RegionOp owner;
  if (!bound.empty()) {
    owner = (*bound.begin())->getParentOfType<RegionOp>();
  }
  if (!owner || owner.getPolicy() != plan.getPolicy() ||
      bound.size() != owner.getBody().front().getOperations().size() - 1) {
    plan.emitOpError("must cover exactly one connected algebra region");
    return failure();
  }
  for (Operation *operation : bound) {
    if (operation->getParentOfType<RegionOp>() != owner) {
      plan.emitOpError("tasks must bind one algebra region");
      return failure();
    }
  }
  auto manifest =
      module->getAttrOfType<DictionaryAttr>(kCoverageManifestAttribute);
  auto tuning = manifest ? manifest.getAs<StringAttr>("canonical_tuning")
                         : StringAttr{};
  const bool preferFusion =
      tuning && tuning.getValue() == kFusionCanonicalTuning;
  FailureOr<DerivedPlan> expected = deriveGpuPlan(owner, preferFusion);
  if (failed(expected) || expected->tasks.size() != tasks.size() ||
      expected->buffers.size() != buffers.size()) {
    plan.emitOpError("does not equal the plan derived from bound algebra");
    return failure();
  }
  for (auto [ordinal, expectedTask] : llvm::enumerate(expected->tasks)) {
    MaterializationTaskOp actual = tasks[ordinal];
    auto actualFoldRegionSources =
        actual->getAttrOfType<ArrayAttr>("fold_region_sources");
    const bool expectedFoldRegion = !expectedTask.foldRegionSources.empty();
    if (expectedFoldRegion) {
      SmallVector<Attribute> expectedRegionSources;
      for (Operation *operation : expectedTask.foldRegionSources) {
        if (auto map = dyn_cast<MapOp>(operation))
          expectedRegionSources.push_back(map.getSource());
        else
          expectedRegionSources.push_back(cast<FoldOp>(operation).getSource());
      }
      auto actualPhases =
          actual->getAttrOfType<ArrayAttr>("fold_region_phases");
      auto actualBoundaryUses =
          actual->getAttrOfType<ArrayAttr>("fold_region_boundary_uses");
      auto actualRecomputed =
          actual->getAttrOfType<ArrayAttr>("fold_region_recomputed_sources");
      SmallVector<Attribute> expectedRecomputed;
      for (Operation *operation : expectedTask.foldRegionRecomputedSources)
        expectedRecomputed.push_back(cast<MapOp>(operation).getSource());
      bool matches =
          actualFoldRegionSources && actualPhases && actualBoundaryUses &&
          actualFoldRegionSources.getValue() ==
              ArrayRef<Attribute>(expectedRegionSources) &&
          actualPhases.size() == expectedTask.foldRegionPhases.size() &&
          actualBoundaryUses.size() ==
              expectedTask.foldRegionBoundaryUses.size() &&
          (expectedRecomputed.empty()
               ? !actualRecomputed
               : actualRecomputed &&
                     actualRecomputed.getValue() ==
                         ArrayRef<Attribute>(expectedRecomputed));
      for (auto [phaseOrdinal, operation] :
           llvm::enumerate(expectedTask.foldRegionPhases)) {
        if (!matches)
          break;
        auto phase = dyn_cast<DictionaryAttr>(actualPhases[phaseOrdinal]);
        auto fold = cast<FoldOp>(operation);
        auto domain =
            phase
                ? dyn_cast_or_null<DenseI64ArrayAttr>(phase.get("domain_shape"))
                : DenseI64ArrayAttr{};
        auto reduction = phase ? dyn_cast_or_null<DenseI64ArrayAttr>(
                                     phase.get("reduction_dimensions"))
                               : DenseI64ArrayAttr{};
        matches = phase && phase.get("source") == fold.getSource() && domain &&
                  reduction &&
                  domain.asArrayRef() ==
                      cast<RankedTensorType>(fold.getInputs().front().getType())
                          .getShape() &&
                  reduction.asArrayRef() == fold.getReductionDimensions() &&
                  cast<BoolAttr>(phase.get("order_free")).getValue() ==
                      fold.getOrderFree() &&
                  cast<StringAttr>(phase.get("transition")).getValue() ==
                      (phaseOrdinal == 0 ? "launch_boundary" : "cta_local");
      }
      for (auto [useOrdinal, expectedUse] :
           llvm::enumerate(expectedTask.foldRegionBoundaryUses)) {
        if (!matches)
          break;
        auto use = dyn_cast<DictionaryAttr>(actualBoundaryUses[useOrdinal]);
        matches =
            use &&
            cast<IntegerAttr>(use.get("input_buffer")).getInt() ==
                expectedUse.inputBuffer &&
            use.get("source") == expectedUse.source.getSource() &&
            cast<IntegerAttr>(use.get("operand")).getInt() ==
                expectedUse.operand &&
            cast<AffineMapAttr>(use.get("indexing_map")).getValue() ==
                expectedUse.indexingMap &&
            cast<IntegerAttr>(use.get("phase")).getInt() == expectedUse.phase;
      }
      if (!matches ||
          expectedTask.operation !=
              sourceOperations.lookup(actual.getSource()) ||
          actual.getReductionDimensions() !=
              ArrayRef<int64_t>(expectedTask.reductionDimensions) ||
          actual.getOrderFree().value_or(false) != expectedTask.orderFree ||
          actual.getInputBuffers() != ArrayRef<int64_t>(expectedTask.inputs) ||
          actual.getOutputBuffers() !=
              ArrayRef<int64_t>(expectedTask.outputs) ||
          actual.getDependencies() !=
              ArrayRef<int64_t>(expectedTask.dependencies)) {
        actual.emitOpError(
            "Fold-region plan must equal structural SSA derivation");
        return failure();
      }
      continue;
    }
    if (actualFoldRegionSources) {
      actual.emitOpError("unexpected Fold-region metadata");
      return failure();
    }
    SmallVector<Attribute> expectedSources;
    for (Operation *operation : expectedTask.fusedOperations)
      expectedSources.push_back(cast<MapOp>(operation).getSource());
    SmallVector<Attribute> expectedMaps;
    for (AffineMap map : expectedTask.composedInputMaps)
      expectedMaps.push_back(AffineMapAttr::get(map));
    auto actualSources = actual->getAttrOfType<ArrayAttr>("fused_sources");
    auto actualMaps =
        actual->getAttrOfType<ArrayAttr>("composed_input_maps");
    auto actualEpilogueSources =
        actual->getAttrOfType<ArrayAttr>("fold_epilogue_sources");
    auto actualEpilogueMaps =
        actual->getAttrOfType<ArrayAttr>("epilogue_input_maps");
    const bool expectedFoldMap = expectedTask.fusedFold != nullptr;
    if (expectedTask.operation != sourceOperations.lookup(actual.getSource()) ||
        (expectedFoldMap && (actualSources || actualMaps)) ||
        (!expectedFoldMap &&
         (expectedSources.empty() != !actualSources)) ||
        (!expectedFoldMap && !expectedSources.empty() &&
         actualSources.getValue() != ArrayRef<Attribute>(expectedSources)) ||
        (!expectedFoldMap && (expectedMaps.empty() != !actualMaps)) ||
        (!expectedFoldMap && !expectedMaps.empty() &&
         actualMaps.getValue() != ArrayRef<Attribute>(expectedMaps)) ||
        (expectedFoldMap != static_cast<bool>(actualEpilogueSources)) ||
        (expectedFoldMap &&
         actualEpilogueSources.getValue() !=
             ArrayRef<Attribute>(expectedSources)) ||
        (expectedFoldMap != static_cast<bool>(actualEpilogueMaps)) ||
        (expectedFoldMap &&
         actualEpilogueMaps.getValue() != ArrayRef<Attribute>(expectedMaps)) ||
        actual.getReductionDimensions() !=
            ArrayRef<int64_t>(expectedTask.reductionDimensions) ||
        actual.getOrderFree().value_or(false) != expectedTask.orderFree ||
        actual.getInputBuffers() != ArrayRef<int64_t>(expectedTask.inputs) ||
        actual.getOutputBuffers() != ArrayRef<int64_t>(expectedTask.outputs) ||
        actual.getDependencies() !=
            ArrayRef<int64_t>(expectedTask.dependencies)) {
      actual.emitOpError(
          "task order and edges must equal bound algebra SSA dependencies");
      return failure();
    }
  }
  for (auto [ordinal, expectedBuffer] : llvm::enumerate(expected->buffers)) {
    MaterializationBufferOp actual = buffers[ordinal];
    auto actualProducer = actual->getAttrOfType<IntegerAttr>("producer");
    std::optional<int64_t> producer =
        actualProducer ? std::optional<int64_t>(actualProducer.getInt())
                       : std::nullopt;
    if (actual.getTensorType() != expectedBuffer.value.getType() ||
        producer != expectedBuffer.producer ||
        actual.getConsumers() != ArrayRef<int64_t>(expectedBuffer.consumers) ||
        actual.getLiveOut() != expectedBuffer.liveOut) {
      actual.emitOpError(
          "buffer ownership and uses must equal bound algebra SSA edges");
      return failure();
    }
  }
  if (boundRegion)
    *boundRegion = owner;
  return success();
}

class VerifyMaterializationPlanPass
    : public impl::ShuttleVerifyMaterializationPlanPassBase<
          VerifyMaterializationPlanPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<MaterializationPlanOp> plans(
        module.getOps<MaterializationPlanOp>());
    SmallVector<RegionOp> regions;
    llvm::SmallDenseSet<Operation *> owners;
    bool invalid = plans.empty();
    for (MaterializationPlanOp plan : plans) {
      RegionOp owner;
      if (failed(
              verifyMaterializationPlanAgainstSource(module, plan, &owner)) ||
          !owners.insert(owner.getOperation()).second) {
        invalid = true;
        break;
      }
      regions.push_back(owner);
    }
    size_t regionCount = 0;
    module.walk([&](RegionOp) { ++regionCount; });
    if (invalid || regions.size() != regionCount) {
      module.emitError(
          "requires one source-bound materialization plan per GPU region");
      signalPassFailure();
    }
  }
};

struct ScheduledGeometry {
  SmallVector<int64_t> grid;
  SmallVector<int64_t> tile;
  int64_t serialTiles;
  int64_t workgroupThreads;
  int64_t subgroupSize;
  int64_t scratchBytes;
  std::optional<int64_t> reductionAxis;
};

FailureOr<int64_t> scheduleElementCount(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t extent : shape) {
    if (extent <= 0 || count > std::numeric_limits<int64_t>::max() / extent) {
      return failure();
    }
    count *= extent;
  }
  return count;
}

int64_t scheduleThreads(int64_t tile) {
  constexpr int64_t kSubgroup = 32;
  return ceilDivPositive(tile, kSubgroup) * kSubgroup;
}

FailureOr<ScheduledGeometry> deriveSimt32Geometry(ScheduleTaskKind kind,
                                                  ArrayRef<int64_t> domain,
                                                  NumericalPolicy policy) {
  constexpr int64_t kSubgroup = 32;
  constexpr int64_t kMaxThreads = 256;
  if (kind == ScheduleTaskKind::Scalar) {
    if (!domain.empty()) {
      return failure();
    }
    return ScheduledGeometry{{1}, {}, 1, 1, kSubgroup, 0, std::nullopt};
  }
  if (kind == ScheduleTaskKind::Elementwise) {
    FailureOr<int64_t> elements = scheduleElementCount(domain);
    if (failed(elements)) {
      return failure();
    }
    int64_t tile = std::min(*elements, kMaxThreads);
    return ScheduledGeometry{{ceilDivPositive(*elements, tile)},
                             {tile},
                             1,
                             scheduleThreads(tile),
                             kSubgroup,
                             0,
                             std::nullopt};
  }
  if (kind == ScheduleTaskKind::Contract) {
    if (domain.size() < 3 || domain.size() > 9) {
      return failure();
    }
    FailureOr<int64_t> outputs = scheduleElementCount(domain.drop_back());
    if (failed(outputs) || *outputs > UINT32_MAX || domain.back() <= 0) {
      return failure();
    }
    const int64_t reductionAxis = domain.size() - 1;
    if (policy == NumericalPolicy::Fast) {
      const int64_t outputsPerBlock = std::min<int64_t>(*outputs, 8);
      return ScheduledGeometry{{ceilDivPositive(*outputs, outputsPerBlock)},
                               {outputsPerBlock},
                               ceilDivPositive(domain.back(), kSubgroup),
                               outputsPerBlock * kSubgroup,
                               kSubgroup,
                               0,
                               reductionAxis};
    }
    const int64_t tile = std::min(*outputs, kMaxThreads);
    return ScheduledGeometry{{ceilDivPositive(*outputs, tile)},
                             {tile},
                             domain.back(),
                             scheduleThreads(tile),
                             kSubgroup,
                             0,
                             reductionAxis};
  }
  if (domain.size() < 2 || domain.size() > 8 ||
      llvm::any_of(domain, [](int64_t extent) { return extent <= 0; })) {
    return failure();
  }
  const bool rowFold = kind == ScheduleTaskKind::RowFold;
  const bool columnFold = kind == ScheduleTaskKind::ColumnFold;
  if (!rowFold && !columnFold) {
    return failure();
  }
  const int64_t reductionAxis = rowFold ? domain.size() - 1 : 0;
  FailureOr<int64_t> reduction =
      rowFold ? FailureOr<int64_t>(domain.back())
              : scheduleElementCount(domain.drop_back());
  FailureOr<int64_t> outputs =
      rowFold ? scheduleElementCount(domain.drop_back())
              : FailureOr<int64_t>(domain.back());
  if (failed(reduction) || failed(outputs) || *outputs > UINT32_MAX) {
    return failure();
  }
  int64_t tile = std::min(*reduction, kMaxThreads);
  int64_t threads = scheduleThreads(tile);
  SmallVector<int64_t> tileShape(domain.size(), 1);
  tileShape[reductionAxis] = tile;
  return ScheduledGeometry{{*outputs},
                           std::move(tileShape),
                           ceilDivPositive(*reduction, tile),
                           threads,
                           kSubgroup,
                           threads * static_cast<int64_t>(sizeof(float)),
                           reductionAxis};
}

void createScheduleRecord(OpBuilder &builder, Location location,
                          StringRef operationName,
                          ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, operationName);
  state.addAttributes(attributes);
  builder.create(state);
}

FailureOr<ScheduleTaskKind>
scheduleKindForMaterializationTask(MaterializationTaskOp task) {
  if (task.getKind() == MaterializationTaskKind::Contract) {
    return ScheduleTaskKind::Contract;
  }
  if (task.getKind() == MaterializationTaskKind::Map) {
    return task.getDomainShape().empty() ? ScheduleTaskKind::Scalar
                                         : ScheduleTaskKind::Elementwise;
  }
  if (task.getKind() != MaterializationTaskKind::Fold) {
    return failure();
  }
  const int64_t rank = task.getDomainShape().size();
  ArrayRef<int64_t> dimensions = task.getReductionDimensions();
  if (dimensions == ArrayRef<int64_t>({rank - 1})) {
    return ScheduleTaskKind::RowFold;
  }
  if ((rank == 2 && dimensions == ArrayRef<int64_t>({0})) ||
      isFullPrefixReduction(dimensions, rank)) {
    return ScheduleTaskKind::ColumnFold;
  }
  return failure();
}

class PlanSimt32RowFoldSchedulePass
    : public impl::ShuttlePlanSimt32RowFoldSchedulePassBase<
          PlanSimt32RowFoldSchedulePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<MaterializationPlanOp> materializationPlans(
        module.getOps<MaterializationPlanOp>());
    if (materializationPlans.empty()) {
      module.emitError("requires source-bound materialization plans");
      return signalPassFailure();
    }
    llvm::SmallDenseSet<Operation *> owners;
    for (MaterializationPlanOp plan : materializationPlans) {
      RegionOp owner;
      if (failed(
              verifyMaterializationPlanAgainstSource(module, plan, &owner)) ||
          !owners.insert(owner.getOperation()).second) {
        module.emitError(
            "requires one source-bound materialization plan per GPU region");
        return signalPassFailure();
      }
    }
    size_t regionCount = 0;
    module.walk([&](RegionOp) { ++regionCount; });
    if (owners.size() != regionCount) {
      module.emitError(
          "requires one source-bound materialization plan per GPU region");
      return signalPassFailure();
    }
    if (!module.getOps<SchedulePlanOp>().empty()) {
      module.emitError("already contains a schedule plan");
      return signalPassFailure();
    }
    for (MaterializationPlanOp source : materializationPlans) {
      SmallVector<MaterializationBufferOp> sourceBuffers(
          source.getBody().front().getOps<MaterializationBufferOp>());
      SmallVector<MaterializationTaskOp> sourceTasks(
          source.getBody().front().getOps<MaterializationTaskOp>());

      OpBuilder builder(module.getContext());
      builder.setInsertionPointToEnd(module.getBody());
      OperationState planState(module.getLoc(),
                               SchedulePlanOp::getOperationName());
      planState.addAttribute("schema_version", builder.getI64IntegerAttr(
                                                   source.getSchemaVersion()));
      planState.addAttribute(
          "target",
          ScheduleTargetAttr::get(module.getContext(), ScheduleTarget::Simt32));
      planState.addAttribute("policy", source.getPolicyAttr());
      planState.addAttribute("source_plan_fingerprint",
                             builder.getStringAttr(source.getFingerprint()));
      planState.addAttribute("fingerprint",
                             builder.getStringAttr(std::string(64, '0')));
      planState.addRegion();
      auto schedule = cast<SchedulePlanOp>(builder.create(planState));
      Block *body = new Block();
      schedule.getBody().push_back(body);
      builder.setInsertionPointToEnd(body);

      for (auto [ordinal, buffer] : llvm::enumerate(sourceBuffers)) {
        auto type = cast<RankedTensorType>(buffer.getTensorType());
        SmallVector<int64_t> order;
        for (int64_t axis = 0; axis < type.getRank(); ++axis) {
          order.push_back(axis);
        }
        createScheduleRecord(
            builder, buffer.getLoc(), ScheduleBufferOp::getOperationName(),
            {builder.getNamedAttr("ordinal",
                                  builder.getI64IntegerAttr(ordinal)),
             builder.getNamedAttr("source_buffer",
                                  builder.getI64IntegerAttr(ordinal)),
             builder.getNamedAttr("tensor_type", TypeAttr::get(type)),
             builder.getNamedAttr(
                 "indexing", ScheduleBufferIndexingAttr::get(
                                 module.getContext(),
                                 type.getRank() == 0
                                     ? ScheduleBufferIndexing::Scalar
                                     : ScheduleBufferIndexing::Lexicographic)),
             builder.getNamedAttr(
                 "iteration_order",
                 DenseI64ArrayAttr::get(module.getContext(), order)),
             builder.getNamedAttr(
                 "lifetime_start",
                 builder.getI64IntegerAttr(buffer.getLifetimeStart())),
             builder.getNamedAttr(
                 "lifetime_end",
                 builder.getI64IntegerAttr(buffer.getLifetimeEnd()))});
      }
      for (auto [ordinal, task] : llvm::enumerate(sourceTasks)) {
        FailureOr<ScheduleTaskKind> derivedKind =
            scheduleKindForMaterializationTask(task);
        if (failed(derivedKind)) {
          task.emitOpError("cannot derive a closed SIMT32 task kind");
          return signalPassFailure();
        }
        ScheduleTaskKind kind = *derivedKind;
        FailureOr<ScheduledGeometry> geometry = deriveSimt32Geometry(
            kind, task.getDomainShape(), source.getPolicy());
        if (failed(geometry)) {
          task.emitOpError("cannot derive a bounded SIMT32 schedule candidate");
          return signalPassFailure();
        }
        SmallVector<NamedAttribute> attrs{
            builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr("source_task",
                                 builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr(
                "kind", ScheduleTaskKindAttr::get(module.getContext(), kind)),
            builder.getNamedAttr("domain_shape", task.getDomainShapeAttr()),
            builder.getNamedAttr(
                "grid_shape",
                DenseI64ArrayAttr::get(module.getContext(), geometry->grid)),
            builder.getNamedAttr(
                "tile_shape",
                DenseI64ArrayAttr::get(module.getContext(), geometry->tile)),
            builder.getNamedAttr("serial_tiles", builder.getI64IntegerAttr(
                                                     geometry->serialTiles)),
            builder.getNamedAttr(
                "workgroup_threads",
                builder.getI64IntegerAttr(geometry->workgroupThreads)),
            builder.getNamedAttr("subgroup_size", builder.getI64IntegerAttr(
                                                      geometry->subgroupSize)),
            builder.getNamedAttr("scratch_bytes", builder.getI64IntegerAttr(
                                                      geometry->scratchBytes)),
            builder.getNamedAttr("input_buffers", task.getInputBuffersAttr()),
            builder.getNamedAttr("output_buffers", task.getOutputBuffersAttr()),
            builder.getNamedAttr("dependencies", task.getDependenciesAttr()),
            builder.getNamedAttr("semantic_fingerprint",
                                 task.getSemanticFingerprintAttr())};
        if (geometry->reductionAxis) {
          attrs.push_back(builder.getNamedAttr(
              "reduction_axis",
              builder.getI64IntegerAttr(*geometry->reductionAxis)));
          attrs.push_back(builder.getNamedAttr(
              "reduction_order",
              ScheduleReductionOrderAttr::get(
                  module.getContext(),
                  kind == ScheduleTaskKind::Contract
                      ? (source.getPolicy() == NumericalPolicy::Fast
                             ? ScheduleReductionOrder::LeafOrderFree
                             : ScheduleReductionOrder::SerialAscendingFmaRnF32)
                      : ScheduleReductionOrder::
                            TreeAssociationFreeLeafOrderFixed)));
        }
        createScheduleRecord(builder, task.getLoc(),
                             ScheduleTaskOp::getOperationName(), attrs);
      }
      builder.create<SchedulePlanYieldOp>(module.getLoc());
      schedule.setFingerprint(schedulePlanFingerprint(schedule));
      if (failed(schedule.verifyRegions())) {
        return signalPassFailure();
      }
    }
  }
};

LogicalResult
verifySchedulePlanAgainstMaterialization(MaterializationPlanOp source,
                                         SchedulePlanOp schedule) {
  if (schedule.getSchemaVersion() != source.getSchemaVersion() ||
      schedule.getTarget() != ScheduleTarget::Simt32 ||
      schedule.getPolicy() != source.getPolicy() ||
      schedule.getSourcePlanFingerprint() != source.getFingerprint()) {
    return schedule.emitOpError(
        "schema, target, policy, and source fingerprint must bind "
        "materialization");
  }
  SmallVector<MaterializationBufferOp> sourceBuffers(
      source.getBody().front().getOps<MaterializationBufferOp>());
  SmallVector<MaterializationTaskOp> sourceTasks(
      source.getBody().front().getOps<MaterializationTaskOp>());
  SmallVector<ScheduleBufferOp> buffers(
      schedule.getBody().front().getOps<ScheduleBufferOp>());
  SmallVector<ScheduleTaskOp> tasks(
      schedule.getBody().front().getOps<ScheduleTaskOp>());
  if (buffers.size() != sourceBuffers.size() ||
      tasks.size() != sourceTasks.size()) {
    return schedule.emitOpError(
        "must cover every materialization buffer and task exactly once");
  }
  for (auto [ordinal, actual, expected] :
       llvm::enumerate(buffers, sourceBuffers)) {
    if (actual.getSourceBuffer() != static_cast<int64_t>(ordinal) ||
        actual.getTensorType() != expected.getTensorType() ||
        actual.getLifetimeStart() != expected.getLifetimeStart() ||
        actual.getLifetimeEnd() != expected.getLifetimeEnd()) {
      return actual.emitOpError(
          "schedule buffer type and lifetime must equal materialization");
    }
  }
  for (auto [ordinal, actual, expected] : llvm::enumerate(tasks, sourceTasks)) {
    FailureOr<ScheduleTaskKind> expectedKind =
        scheduleKindForMaterializationTask(expected);
    if (failed(expectedKind)) {
      return expected.emitOpError("does not have a closed SIMT32 task kind");
    }
    if (actual.getSourceTask() != static_cast<int64_t>(ordinal) ||
        actual.getKind() != *expectedKind ||
        actual.getDomainShape() != expected.getDomainShape() ||
        actual.getInputBuffers() != expected.getInputBuffers() ||
        actual.getOutputBuffers() != expected.getOutputBuffers() ||
        actual.getDependencies() != expected.getDependencies() ||
        actual.getSemanticFingerprint() != expected.getSemanticFingerprint()) {
      return actual.emitOpError(
          "schedule dependencies must equal the materialization task");
    }
  }
  return success();
}

class VerifySimt32RowFoldSchedulePass
    : public impl::ShuttleVerifySimt32RowFoldSchedulePassBase<
          VerifySimt32RowFoldSchedulePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<MaterializationPlanOp> materializationPlans(
        module.getOps<MaterializationPlanOp>());
    SmallVector<SchedulePlanOp> schedules(module.getOps<SchedulePlanOp>());
    if (materializationPlans.empty() ||
        materializationPlans.size() != schedules.size()) {
      module.emitError("requires one valid schedule per materialization plan");
      return signalPassFailure();
    }
    llvm::SmallDenseSet<Operation *> matchedSchedules;
    for (MaterializationPlanOp materialization : materializationPlans) {
      SchedulePlanOp match;
      for (SchedulePlanOp schedule : schedules) {
        if (schedule.getSourcePlanFingerprint() ==
            materialization.getFingerprint()) {
          if (match) {
            module.emitError("materialization plan has duplicate schedules");
            return signalPassFailure();
          }
          match = schedule;
        }
      }
      if (!match || !matchedSchedules.insert(match.getOperation()).second ||
          failed(verifyMaterializationPlanAgainstSource(module,
                                                        materialization)) ||
          failed(match.verifyRegions()) ||
          failed(verifySchedulePlanAgainstMaterialization(materialization,
                                                          match))) {
        module.emitError(
            "requires one valid schedule per materialization plan");
        return signalPassFailure();
      }
    }
  }
};

struct ExecutableSlotSpec {
  Type tensorType;
  int64_t requiredBytes;
  SmallVector<int64_t> strides;
  int64_t alignment;
  ExecutableAccess access;
  MaterializationStorage storage;
};

struct GpuEntrySpec {
  int64_t ordinal;
  int64_t codeOffset;
  int64_t codeLength;
  SmallVector<int64_t> inputs;
  SmallVector<int64_t> outputs;
  SmallVector<int64_t> dependencies;
  ExecutablePredication predication;
  std::optional<ScheduleReductionOrder> reductionOrder;
  std::array<int64_t, 3> grid;
  std::array<int64_t, 3> block;
  int64_t dynamicSharedBytes;
  int64_t kernelArity;
  std::string codeDigest;
};

struct GpuExecutableSpec {
  SmallVector<int8_t> code;
  std::vector<GpuEntrySpec> entries;
  SmallVector<ExecutableSlotSpec> slots;
  SmallVector<int64_t> reuseGroups;
};

FailureOr<int64_t> staticElementCount(RankedTensorType type) {
  int64_t elements = 1;
  for (int64_t extent : type.getShape()) {
    if (extent <= 0 || elements > std::numeric_limits<int64_t>::max() / extent)
      return failure();
    elements *= extent;
  }
  return elements;
}

FailureOr<SmallVector<ExecutableSlotSpec>>
deriveExecutableSlots(MaterializationPlanOp materialization) {
  SmallVector<ExecutableSlotSpec> slots;
  for (MaterializationBufferOp buffer :
       materialization.getBody().front().getOps<MaterializationBufferOp>()) {
    auto type = dyn_cast<RankedTensorType>(buffer.getTensorType());
    if (!type || !type.hasStaticShape() ||
        (!type.getElementType().isBF16() && !type.getElementType().isF32()))
      return failure();
    const int64_t elementBytes = type.getElementType().isBF16() ? 2 : 4;
    FailureOr<int64_t> elements = staticElementCount(type);
    if (failed(elements) ||
        *elements > std::numeric_limits<int64_t>::max() / elementBytes)
      return failure();
    SmallVector<int64_t> strides(type.getRank());
    int64_t stride = elementBytes;
    for (int64_t axis = type.getRank(); axis > 0; --axis) {
      strides[axis - 1] = stride;
      stride *= type.getDimSize(axis - 1);
    }
    ExecutableAccess access = ExecutableAccess::ReadWrite;
    if (buffer.getLiveIn())
      access = ExecutableAccess::Read;
    else if (buffer.getLiveOut())
      access = ExecutableAccess::Write;
    slots.push_back(ExecutableSlotSpec{type, *elements * elementBytes,
                                       std::move(strides), elementBytes,
                                       access, buffer.getStorage()});
  }
  return slots;
}

FailureOr<SmallVector<int64_t>>
deriveGpuReuseGroups(MaterializationPlanOp materialization,
                     ArrayRef<ExecutableSlotSpec> slots) {
  SmallVector<MaterializationBufferOp> buffers(
      materialization.getBody().front().getOps<MaterializationBufferOp>());
  if (buffers.size() != slots.size())
    return failure();
  struct Candidate {
    int64_t ordinal;
    int64_t lifetimeStart;
    int64_t lifetimeEnd;
    int64_t requiredBytes;
  };
  struct ReuseGroup {
    int64_t ordinal;
    SmallVector<std::pair<int64_t, int64_t>> intervals;
  };
  SmallVector<Candidate> candidates;
  SmallVector<int64_t> groups(slots.size(), -1);
  for (auto [ordinal, buffer, slot] : llvm::enumerate(buffers, slots)) {
    if (slot.storage != MaterializationStorage::Temporary) {
      groups[ordinal] = ordinal;
      continue;
    }
    if (slot.requiredBytes <= 0 || buffer.getLifetimeStart() < 0 ||
        buffer.getLifetimeEnd() < buffer.getLifetimeStart())
      return failure();
    candidates.push_back({static_cast<int64_t>(ordinal),
                          static_cast<int64_t>(buffer.getLifetimeStart()),
                          static_cast<int64_t>(buffer.getLifetimeEnd()),
                          slot.requiredBytes});
  }
  llvm::sort(candidates, [](const Candidate &left, const Candidate &right) {
    if (left.requiredBytes != right.requiredBytes)
      return left.requiredBytes > right.requiredBytes;
    if (left.lifetimeStart != right.lifetimeStart)
      return left.lifetimeStart < right.lifetimeStart;
    if (left.lifetimeEnd != right.lifetimeEnd)
      return left.lifetimeEnd < right.lifetimeEnd;
    return left.ordinal < right.ordinal;
  });
  SmallVector<ReuseGroup> available;
  for (const Candidate &candidate : candidates) {
    auto reusable = llvm::find_if(available, [&](const ReuseGroup &group) {
      return llvm::all_of(group.intervals, [&](const auto &interval) {
        return candidate.lifetimeEnd < interval.first ||
               interval.second < candidate.lifetimeStart;
      });
    });
    if (reusable == available.end()) {
      available.push_back(
          {static_cast<int64_t>(slots.size() + available.size()), {}});
      reusable = std::prev(available.end());
    }
    reusable->intervals.push_back(
        {candidate.lifetimeStart, candidate.lifetimeEnd});
    groups[candidate.ordinal] = reusable->ordinal;
  }
  if (llvm::is_contained(groups, -1))
    return failure();
  return groups;
}

struct PtxEmitter {
  explicit PtxEmitter(int64_t arity) : arity(arity), stream(body) {}

  void header(StringRef certificate, bool usesDynamicSharedMemory = false) {
    this->certificate = certificate.str();
    this->usesDynamicSharedMemory = usesDynamicSharedMemory;
  }

  int64_t nextF32() { return f32++; }
  int64_t nextU64() { return u64++; }
  int64_t nextBF16() { return bf16++; }

  void finish() {
    stream.flush();
    llvm::raw_string_ostream output(text);
    output << ".version 8.0\n.target sm_90\n.address_size 64\n";
    if (usesDynamicSharedMemory)
      output << ".extern .shared .align 4 .b8 shuttle_smem[];\n";
    output << "// shuttle structural certificate: " << certificate
           << "\n.visible .entry shuttle_entry(";
    for (int64_t index = 0; index < arity; ++index) {
      if (index)
        output << ", ";
      output << ".param .u64 p" << index;
    }
    output << ")\n{\n"
              "  .reg .pred %p<16>;\n"
           << "  .reg .b16 %h<" << std::max<int64_t>(16, bf16) << ">;\n"
           << "  .reg .b32 %r<64>;\n"
           << "  .reg .b64 %rd<" << std::max<int64_t>(128, u64) << ">;\n"
           << "  .reg .f32 %f<" << std::max<int64_t>(128, f32) << ">;\n"
           << body << "$done:\n  ret;\n}\n";
    output.flush();
  }

  int64_t arity;
  int64_t f32 = 0;
  int64_t u64 = 8;
  int64_t bf16 = 2;
  std::string certificate;
  bool usesDynamicSharedMemory = false;
  std::string body;
  std::string text;
  llvm::raw_string_ostream stream;
};

LogicalResult emitLinearThread(PtxEmitter &ptx, int64_t elements) {
  if (elements <= 0 || elements > UINT32_MAX)
    return failure();
  ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                "  mov.u32 %r1, %ctaid.x;\n"
                "  mov.u32 %r2, %ntid.x;\n"
                "  mad.lo.u32 %r3, %r1, %r2, %r0;\n"
                "  setp.ge.u32 %p0, %r3, "
             << elements << ";\n  @%p0 bra $done;\n"
             << "  cvt.u64.u32 %rd0, %r3;\n";
  return success();
}

FailureOr<SmallVector<int64_t>> emitOutputCoordinates(PtxEmitter &ptx,
                                                      RankedTensorType type) {
  FailureOr<int64_t> elements = staticElementCount(type);
  if (failed(elements) || *elements > UINT32_MAX) {
    return failure();
  }
  SmallVector<int64_t> coordinates;
  int64_t stride = 1;
  SmallVector<int64_t> strides(type.getRank());
  for (int64_t axis = type.getRank(); axis > 0; --axis) {
    strides[axis - 1] = stride;
    stride *= type.getDimSize(axis - 1);
  }
  for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
    int64_t reg = ptx.nextU64();
    ptx.stream << "  div.u64 %rd" << reg << ", %rd0, " << strides[axis]
               << ";\n  rem.u64 %rd" << reg << ", %rd" << reg << ", " << extent
               << ";\n";
    coordinates.push_back(reg);
  }
  return coordinates;
}

FailureOr<int64_t> emitAffineCoordinate(PtxEmitter &ptx, AffineExpr expression,
                                        ArrayRef<int64_t> coordinates) {
  if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
    if (dimension.getPosition() >= coordinates.size())
      return failure();
    return coordinates[dimension.getPosition()];
  }
  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary || binary.getKind() != AffineExprKind::FloorDiv)
    return failure();
  auto dimension = dyn_cast<AffineDimExpr>(binary.getLHS());
  auto divisor = dyn_cast<AffineConstantExpr>(binary.getRHS());
  if (!dimension || !divisor || divisor.getValue() <= 1 ||
      dimension.getPosition() >= coordinates.size())
    return failure();
  int64_t reg = ptx.nextU64();
  ptx.stream << "  div.u64 %rd" << reg << ", %rd"
             << coordinates[dimension.getPosition()] << ", "
             << divisor.getValue() << ";\n";
  return reg;
}

FailureOr<int64_t> emitMapInput(PtxEmitter &ptx, int64_t parameter,
                                RankedTensorType type, AffineMap map,
                                ArrayRef<int64_t> coordinates) {
  if (map.getNumResults() != type.getRank())
    return failure();
  int64_t linear = ptx.nextU64();
  ptx.stream << "  mov.u64 %rd" << linear << ", 0;\n";
  for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
    FailureOr<int64_t> coordinate =
        emitAffineCoordinate(ptx, expression, coordinates);
    if (failed(coordinate))
      return failure();
    ptx.stream << "  mad.lo.u64 %rd" << linear << ", %rd" << linear << ", "
               << type.getDimSize(axis) << ", %rd" << *coordinate << ";\n";
  }
  int64_t pointer = ptx.nextU64();
  int64_t bytes = type.getElementType().isBF16() ? 2 : 4;
  ptx.stream << "  ld.param.u64 %rd" << pointer << ", [p" << parameter
             << "];\n  mad.lo.u64 %rd" << pointer << ", %rd" << linear << ", "
             << bytes << ", %rd" << pointer << ";\n";
  int64_t value = ptx.nextF32();
  if (type.getElementType().isBF16()) {
    ptx.stream << "  ld.global.u16 %h0, [%rd" << pointer
               << "];\n  cvt.f32.bf16 %f" << value << ", %h0;\n";
  } else {
    ptx.stream << "  ld.global.f32 %f" << value << ", [%rd" << pointer
               << "];\n";
  }
  return value;
}

FailureOr<int64_t> emitScalarBody(PtxEmitter &ptx, Block &body,
                                  ArrayRef<int64_t> arguments) {
  if (body.getNumArguments() != arguments.size())
    return failure();
  DenseMap<Value, int64_t> values;
  for (auto [argument, reg] : llvm::zip_equal(body.getArguments(), arguments))
    values[argument] = reg;
  for (Operation &operation : body.without_terminator()) {
    if (operation.getNumResults() != 1 ||
        (!isa<ScalarConvertOp>(operation) &&
         !operation.getResult(0).getType().isF32()))
      return failure();
    int64_t result = ptx.nextF32();
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      auto value = dyn_cast<FloatAttr>(constant.getValue());
      if (!value || !value.getType().isF32())
        return failure();
      uint32_t bits = value.getValue().bitcastToAPInt().getZExtValue();
      ptx.stream << "  mov.b32 %f" << result << ", 0f"
                 << llvm::utohexstr(bits, /*LowerCase=*/false, /*Width=*/8)
                 << ";\n";
    } else if (auto convert = dyn_cast<ScalarConvertOp>(operation)) {
      Type inputType = convert.getInput().getType();
      Type resultType = convert.getResult().getType();
      int64_t operand = values.lookup(convert.getInput());
      if ((inputType.isF32() && resultType.isF32()) ||
          (inputType.isBF16() && resultType.isF32())) {
        result = operand;
      } else if (inputType.isF32() && resultType.isBF16() &&
                 convert.getSemantics() ==
                     ScalarConvertSemantics::RoundNearestEven) {
        int64_t rounded = ptx.nextBF16();
        ptx.stream << "  cvt.rn.bf16.f32 %h" << rounded << ", %f" << operand
                   << ";\n  cvt.f32.bf16 %f" << result << ", %h" << rounded
                   << ";\n";
      } else {
        return failure();
      }
    } else if (auto add = dyn_cast<arith::AddFOp>(operation)) {
      ptx.stream << "  add.rn.f32 %f" << result << ", %f"
                 << values.lookup(add.getLhs()) << ", %f"
                 << values.lookup(add.getRhs()) << ";\n";
    } else if (auto subtract = dyn_cast<arith::SubFOp>(operation)) {
      ptx.stream << "  sub.rn.f32 %f" << result << ", %f"
                 << values.lookup(subtract.getLhs()) << ", %f"
                 << values.lookup(subtract.getRhs()) << ";\n";
    } else if (auto negate = dyn_cast<arith::NegFOp>(operation)) {
      ptx.stream << "  neg.f32 %f" << result << ", %f"
                 << values.lookup(negate.getOperand()) << ";\n";
    } else if (auto multiply = dyn_cast<arith::MulFOp>(operation)) {
      ptx.stream << "  mul.rn.f32 %f" << result << ", %f"
                 << values.lookup(multiply.getLhs()) << ", %f"
                 << values.lookup(multiply.getRhs()) << ";\n";
    } else if (auto divide = dyn_cast<arith::DivFOp>(operation)) {
      ptx.stream << "  div.rn.f32 %f" << result << ", %f"
                 << values.lookup(divide.getLhs()) << ", %f"
                 << values.lookup(divide.getRhs()) << ";\n";
    } else if (auto rsqrt = dyn_cast<math::RsqrtOp>(operation)) {
      int64_t root = ptx.nextF32();
      ptx.stream << "  sqrt.rn.f32 %f" << root << ", %f"
                 << values.lookup(rsqrt.getOperand()) << ";\n  div.rn.f32 %f"
                 << result << ", 0f3F800000, %f" << root << ";\n";
    } else {
      return failure();
    }
    if (operation.getNumResults() != 1)
      return failure();
    values[operation.getResult(0)] = result;
  }
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1 ||
      !values.count(yield.getValues().front()))
    return failure();
  return values.lookup(yield.getValues().front());
}

FailureOr<std::string> emitMapPtx(MapOp map, ScheduleTaskOp task) {
  auto output = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  if (!output || map.getNumResults() != 1)
    return failure();
  ArrayRef<Attribute> maps = map.getIndexingMaps().getValue();
  if (maps.size() != map.getInputs().size() + 1 ||
      !isIdentityMap(cast<AffineMapAttr>(maps.back()).getValue()))
    return failure();
  FailureOr<int64_t> elements = staticElementCount(output);
  if (failed(elements))
    return failure();
  PtxEmitter ptx(map.getInputs().size() + 1);
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, *elements)))
    return failure();
  FailureOr<SmallVector<int64_t>> coordinates =
      emitOutputCoordinates(ptx, output);
  if (failed(coordinates))
    return failure();
  SmallVector<int64_t> arguments;
  for (auto [index, input] : llvm::enumerate(map.getInputs())) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto indexing = dyn_cast<AffineMapAttr>(maps[index]);
    if (!inputType || !indexing)
      return failure();
    FailureOr<int64_t> value =
        emitMapInput(ptx, index, inputType, indexing.getValue(), *coordinates);
    if (failed(value))
      return failure();
    arguments.push_back(*value);
  }
  FailureOr<int64_t> result =
      emitScalarBody(ptx, map.getBody().front(), arguments);
  if (failed(result))
    return failure();
  int64_t outputPointer = ptx.nextU64();
  ptx.stream << "  ld.param.u64 %rd" << outputPointer << ", [p"
             << map.getInputs().size() << "];\n";
  int64_t bytes = output.getElementType().isBF16() ? 2 : 4;
  ptx.stream << "  mad.lo.u64 %rd" << outputPointer << ", %rd0, " << bytes
             << ", %rd" << outputPointer << ";\n";
  if (output.getElementType().isBF16()) {
    ptx.stream << "  cvt.rn.bf16.f32 %h1, %f" << *result
               << ";\n  st.global.u16 [%rd" << outputPointer << "], %h1;\n";
  } else {
    ptx.stream << "  st.global.f32 [%rd" << outputPointer << "], %f" << *result
               << ";\n";
  }
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string>
emitFusedMapPtx(SmallVectorImpl<MapOp> &maps,
                MaterializationTaskOp materializationTask,
                SmallVectorImpl<MaterializationBufferOp> &buffers,
                ScheduleTaskOp scheduleTask) {
  if (maps.size() < 2 || materializationTask.getOutputBuffers().size() != 1 ||
      llvm::any_of(maps, [](MapOp map) { return !hasIdentityResultMap(map); }))
    return failure();
  int64_t outputBuffer = materializationTask.getOutputBuffers().front();
  if (outputBuffer < 0 || outputBuffer >= buffers.size())
    return failure();
  auto output =
      dyn_cast<RankedTensorType>(buffers[outputBuffer].getTensorType());
  auto composedMaps = materializationTask->getAttrOfType<ArrayAttr>(
      "composed_input_maps");
  if (!output || !composedMaps ||
      composedMaps.size() != materializationTask.getInputBuffers().size())
    return failure();
  FailureOr<int64_t> elements = staticElementCount(output);
  if (failed(elements))
    return failure();
  PtxEmitter ptx(materializationTask.getInputBuffers().size() + 1);
  ptx.header(scheduleTask.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, *elements)))
    return failure();
  FailureOr<SmallVector<int64_t>> coordinates =
      emitOutputCoordinates(ptx, output);
  if (failed(coordinates))
    return failure();

  llvm::SmallDenseSet<Operation *> group;
  for (MapOp map : maps)
    group.insert(map.getOperation());
  SmallVector<Value> boundaryInputs;
  llvm::SmallDenseSet<Value> seenBoundary;
  for (MapOp map : llvm::reverse(maps)) {
    for (Value operand : map.getInputs()) {
      if (!group.contains(operand.getDefiningOp()) &&
          seenBoundary.insert(operand).second)
        boundaryInputs.push_back(operand);
    }
  }
  if (boundaryInputs.size() != materializationTask.getInputBuffers().size())
    return failure();
  DenseMap<Value, int64_t> values;
  for (auto [parameter, value, bufferOrdinal, mapAttribute] : llvm::enumerate(
           boundaryInputs, materializationTask.getInputBuffers(),
           composedMaps.getValue())) {
    if (bufferOrdinal < 0 || bufferOrdinal >= buffers.size())
      return failure();
    auto input =
        dyn_cast<RankedTensorType>(buffers[bufferOrdinal].getTensorType());
    auto indexing = dyn_cast<AffineMapAttr>(mapAttribute);
    if (!input || !indexing)
      return failure();
    FailureOr<int64_t> loaded = emitMapInput(
        ptx, parameter, input, indexing.getValue(), *coordinates);
    if (failed(loaded) || input != value.getType())
      return failure();
    values[value] = *loaded;
  }
  for (MapOp map : maps) {
    SmallVector<int64_t> arguments;
    for (Value operand : map.getInputs()) {
      auto found = values.find(operand);
      if (found == values.end())
        return failure();
      arguments.push_back(found->second);
    }
    FailureOr<int64_t> result =
        emitScalarBody(ptx, map.getBody().front(), arguments);
    if (failed(result) || map.getNumResults() != 1)
      return failure();
    values[map.getResult(0)] = *result;
  }
  auto result = values.find(maps.back().getResult(0));
  if (result == values.end())
    return failure();
  int64_t outputPointer = ptx.nextU64();
  ptx.stream << "  ld.param.u64 %rd" << outputPointer << ", [p"
             << materializationTask.getInputBuffers().size() << "];\n";
  ptx.stream << "  mad.lo.u64 %rd" << outputPointer << ", %rd0, 4, %rd"
             << outputPointer << ";\n  st.global.f32 [%rd" << outputPointer
             << "], %f" << result->second << ";\n";
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitContractPtx(ContractOp contract,
                                       ScheduleTaskOp task) {
  const bool fast =
      task.getReductionOrder() == ScheduleReductionOrder::LeafOrderFree;
  FailureOr<SmallVector<int64_t>> contractDomain =
      contractDomainShape(contract);
  if (!isClosedProjectedContract(contract) || failed(contractDomain) ||
      task.getKind() != ScheduleTaskKind::Contract ||
      task.getDomainShape() != ArrayRef<int64_t>(*contractDomain) ||
      task.getReductionAxis() !=
          static_cast<int64_t>(task.getDomainShape().size()) - 1 ||
      (!fast && task.getReductionOrder() !=
                    ScheduleReductionOrder::SerialAscendingFmaRnF32) ||
      task.getInputBuffers().size() != 2 ||
      task.getOutputBuffers().size() != 1 || task.getScratchBytes() != 0) {
    return failure();
  }
  FailureOr<int64_t> outputCount =
      scheduleElementCount(task.getDomainShape().drop_back());
  if (failed(outputCount) || *outputCount > UINT32_MAX) {
    return failure();
  }
  const int64_t outputs = *outputCount;
  const int64_t reduction = task.getDomainShape().back();
  AffineMap lhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[0]).getValue();
  AffineMap rhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[1]).getValue();
  auto lhsType = cast<RankedTensorType>(contract.getInputs()[0].getType());
  auto rhsType = cast<RankedTensorType>(contract.getInputs()[1].getType());
  if (fast) {
    const int64_t outputsPerBlock = std::min<int64_t>(outputs, 8);
    if (task.getGridShape() !=
            ArrayRef<int64_t>({ceilDivPositive(outputs, outputsPerBlock)}) ||
        task.getTileShape() != ArrayRef<int64_t>({outputsPerBlock}) ||
        task.getWorkgroupThreads() != outputsPerBlock * 32 ||
        task.getSubgroupSize() != 32 ||
        task.getSerialTiles() != ceilDivPositive(reduction, 32)) {
      return failure();
    }

    PtxEmitter ptx(/*arity=*/3);
    ptx.header(task.getSemanticFingerprint());
    ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                  "  shr.u32 %r1, %r0, 5;\n"
                  "  and.b32 %r2, %r0, 31;\n"
                  "  mov.u32 %r3, %ctaid.x;\n"
                  "  mad.lo.u32 %r4, %r3, "
               << outputsPerBlock
               << ", %r1;\n"
                  "  setp.ge.u32 %p0, %r4, "
               << outputs
               << ";\n"
                  "  @%p0 bra $done;\n"
                  "  cvt.u64.u32 %rd0, %r4;\n";
    auto result = cast<RankedTensorType>(contract.getResult(0).getType());
    FailureOr<SmallVector<int64_t>> coordinates =
        emitOutputCoordinates(ptx, result);
    if (failed(coordinates) ||
        coordinates->size() + 1 != task.getDomainShape().size()) {
      return failure();
    }
    int64_t accumulator = ptx.nextF32();
    int64_t reductionCoordinate = ptx.nextU64();
    ptx.stream << "  mov.f32 %f" << accumulator
               << ", 0f00000000;\n"
                  "  cvt.u64.u32 %rd"
               << reductionCoordinate
               << ", %r2;\n"
                  "$contract_fast_loop:\n"
                  "  setp.ge.u64 %p1, %rd"
               << reductionCoordinate << ", " << reduction
               << ";\n"
                  "  @%p1 bra $contract_fast_reduce;\n"
                  "";
    SmallVector<int64_t> domainCoordinates(*coordinates);
    domainCoordinates.push_back(reductionCoordinate);
    FailureOr<int64_t> lhs =
        emitMapInput(ptx, 0, lhsType, lhsMap, domainCoordinates);
    FailureOr<int64_t> rhs =
        emitMapInput(ptx, 1, rhsType, rhsMap, domainCoordinates);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }
    ptx.stream << "  fma.rn.f32 %f" << accumulator << ", %f" << *lhs << ", %f"
               << *rhs << ", %f" << accumulator << ";\n  add.u64 %rd"
               << reductionCoordinate << ", %rd" << reductionCoordinate
               << ", 32;\n"
                  "  bra $contract_fast_loop;\n"
                  "$contract_fast_reduce:\n"
                  "  mov.b32 %r5, %f"
               << accumulator << ";\n";
    int64_t shuffled = ptx.nextF32();
    for (int64_t offset : {16, 8, 4, 2, 1}) {
      ptx.stream << "  shfl.sync.down.b32 %r6, %r5, " << offset
                 << ", 31, 0xffffffff;\n"
                    "  mov.b32 %f"
                 << shuffled << ", %r6;\n  add.rn.f32 %f" << accumulator
                 << ", %f" << accumulator << ", %f" << shuffled
                 << ";\n  mov.b32 %r5, %f" << accumulator << ";\n";
    }
    int64_t outputPointer = ptx.nextU64();
    ptx.stream << "  setp.ne.u32 %p2, %r2, 0;\n"
                  "  @%p2 bra $done;\n"
                  "  ld.param.u64 %rd"
               << outputPointer << ", [p2];\n  mad.lo.u64 %rd" << outputPointer
               << ", %rd0, 4, %rd" << outputPointer << ";\n  st.global.f32 [%rd"
               << outputPointer << "], %f" << accumulator << ";\n";
    if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16) {
      return failure();
    }
    ptx.finish();
    return ptx.text;
  }

  const int64_t expectedThreads =
      scheduleThreads(std::min<int64_t>(outputs, 256));
  if (task.getGridShape() != ArrayRef<int64_t>({ceilDivPositive(
                                 outputs, std::min<int64_t>(outputs, 256))}) ||
      task.getTileShape() !=
          ArrayRef<int64_t>({std::min<int64_t>(outputs, 256)}) ||
      task.getWorkgroupThreads() != expectedThreads ||
      task.getSubgroupSize() != 32 || task.getSerialTiles() != reduction) {
    return failure();
  }

  PtxEmitter ptx(/*arity=*/3);
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, outputs))) {
    return failure();
  }
  auto result = cast<RankedTensorType>(contract.getResult(0).getType());
  FailureOr<SmallVector<int64_t>> coordinates =
      emitOutputCoordinates(ptx, result);
  if (failed(coordinates) ||
      coordinates->size() + 1 != task.getDomainShape().size()) {
    return failure();
  }
  int64_t accumulator = ptx.nextF32();
  int64_t reductionCoordinate = ptx.nextU64();
  ptx.stream << "  mov.f32 %f" << accumulator << ", 0f00000000;\n  mov.u64 %rd"
             << reductionCoordinate
             << ", 0;\n"
                "$contract_loop:\n"
                "  setp.ge.u64 %p1, %rd"
             << reductionCoordinate << ", " << reduction
             << ";\n  @%p1 bra $contract_done;\n"
                "";
  SmallVector<int64_t> domainCoordinates(*coordinates);
  domainCoordinates.push_back(reductionCoordinate);
  FailureOr<int64_t> lhs =
      emitMapInput(ptx, 0, lhsType, lhsMap, domainCoordinates);
  FailureOr<int64_t> rhs =
      emitMapInput(ptx, 1, rhsType, rhsMap, domainCoordinates);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  int64_t outputPointer = ptx.nextU64();
  ptx.stream << "  fma.rn.f32 %f" << accumulator << ", %f" << *lhs << ", %f"
             << *rhs << ", %f" << accumulator << ";\n  add.u64 %rd"
             << reductionCoordinate << ", %rd" << reductionCoordinate
             << ", 1;\n"
                "  bra $contract_loop;\n"
                "$contract_done:\n"
                "  ld.param.u64 %rd"
             << outputPointer << ", [p2];\n  mad.lo.u64 %rd" << outputPointer
             << ", %rd0, 4, %rd" << outputPointer << ";\n  st.global.f32 [%rd"
             << outputPointer << "], %f" << accumulator << ";\n";
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16) {
    return failure();
  }
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitFoldMapPtx(
    FoldOp fold, ScheduleTaskOp task, ArrayRef<MapOp> epilogueMaps,
    MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> *buffers) {
  const bool fused = !epilogueMaps.empty();
  if ((!isClosedRankTwoFold(fold) && !isClosedRankNFold(fold)) ||
      fold.getInputs().size() != 1 || fold.getInitializers().size() != 1 ||
      fold.getNumResults() != 1 ||
      task.getDomainShape() !=
          cast<RankedTensorType>(fold.getInputs()[0].getType()).getShape())
    return failure();
  if (!hasExactPositiveZeroInitializer(fold))
    return failure();
  const bool fullPrefix = isClosedFullPrefixRankNFold(fold);
  const int64_t reductionAxis = fold.getReductionDimensions().front();
  FailureOr<int64_t> reductionExtent =
      fullPrefix
          ? scheduleElementCount(task.getDomainShape().drop_back())
          : FailureOr<int64_t>(task.getDomainShape()[reductionAxis]);
  if (failed(reductionExtent))
    return failure();
  auto resultType = cast<RankedTensorType>(fold.getResult(0).getType());
  FailureOr<int64_t> outputElements = staticElementCount(resultType);
  if (failed(outputElements) || *outputElements > UINT32_MAX)
    return failure();
  const int64_t outputExtent = *outputElements;
  const int64_t threads = task.getWorkgroupThreads();
  const int64_t serialTiles = task.getSerialTiles();
  auto nextPowerOfTwo = [](int64_t value) -> FailureOr<int64_t> {
    if (value <= 0)
      return failure();
    int64_t power = 1;
    while (power < value) {
      if (power > 128)
        return failure();
      power *= 2;
    }
    return power;
  };
  FailureOr<int64_t> localWidth = nextPowerOfTwo(serialTiles);
  FailureOr<int64_t> sharedWidth =
      nextPowerOfTwo(std::min(*reductionExtent, threads));
  if (failed(localWidth) || failed(sharedWidth) || threads <= 0 ||
      threads > 256 ||
      task.getGridShape() != ArrayRef<int64_t>({outputExtent}) ||
      task.getScratchBytes() != threads * static_cast<int64_t>(sizeof(float)) ||
      *localWidth > 64 || *sharedWidth > threads)
    return failure();
  if (fused &&
      (!materializationTask || !buffers ||
       materializationTask.getInputBuffers().size() < fold->getNumOperands() ||
       materializationTask.getOutputBuffers().size() != 1))
    return failure();
  PtxEmitter ptx(fused ? materializationTask.getInputBuffers().size() + 1 : 3);
  ptx.header(task.getSemanticFingerprint(), /*usesDynamicSharedMemory=*/true);
  ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                "  mov.u32 %r1, %ctaid.x;\n"
                "  setp.ge.u32 %p0, %r1, "
             << outputExtent
             << ";\n  @%p0 bra $done;\n"
                "  ld.param.u64 %rd0, [p0];\n";
  for (int64_t index = 0; index < *localWidth; ++index)
    ptx.stream << "  mov.f32 %f" << index << ", 0f00000000;\n";
  for (int64_t index = 0; index < serialTiles; ++index) {
    ptx.stream << "  mul.lo.u32 %r4, %r0, " << serialTiles
               << ";\n  add.u32 %r4, %r4, " << index
               << ";\n  setp.ge.u32 %p1, %r4, " << *reductionExtent
               << ";\n  @%p1 bra $load_skip" << index << ";\n"
               << "  cvt.u64.u32 %rd1, %r4;\n"
               << "  cvt.u64.u32 %rd2, %r1;\n";
    if (reductionAxis ==
        static_cast<int64_t>(task.getDomainShape().size()) - 1) {
      ptx.stream << "  mad.lo.u64 %rd3, %rd2, " << *reductionExtent
                 << ", %rd1;\n";
    } else {
      ptx.stream << "  mad.lo.u64 %rd3, %rd1, " << outputExtent << ", %rd2;\n";
    }
    ptx.stream << "  mad.lo.u64 %rd3, %rd3, 4, %rd0;\n"
               << "  ld.global.f32 %f" << index << ", [%rd3];\n$load_skip"
               << index << ":\n";
  }
  int64_t width = *localWidth;
  while (width > 1) {
    for (int64_t index = 0; index < width / 2; ++index)
      ptx.stream << "  add.rn.f32 %f" << index << ", %f" << index * 2 << ", %f"
                 << index * 2 + 1 << ";\n";
    width /= 2;
  }
  ptx.stream << "  cvt.u64.u32 %rd3, %r0;\n"
                "  mul.lo.u64 %rd3, %rd3, 4;\n"
                "  cvta.to.shared.u64 %rd4, shuttle_smem;\n"
                "  add.u64 %rd3, %rd4, %rd3;\n"
                "  st.shared.f32 [%rd3], %f0;\n"
                "  bar.sync 0;\n";
  for (int64_t step = *sharedWidth / 2; step >= 1; step /= 2) {
    ptx.stream << "  setp.ge.u32 %p1, %r0, " << step << ";\n  @%p1 bra $skip"
               << step << ";\n"
               << "  mul.lo.u32 %r4, %r0, 2;\n"
               << "  cvt.u64.u32 %rd5, %r4;\n"
               << "  mul.lo.u64 %rd5, %rd5, 4;\n"
               << "  add.u64 %rd5, %rd4, %rd5;\n"
               << "  ld.shared.f32 %f32, [%rd5];\n"
               << "  ld.shared.f32 %f33, [%rd5+4];\n"
               << "  add.rn.f32 %f34, %f32, %f33;\n"
               << "  cvt.u64.u32 %rd6, %r0;\n"
               << "  mul.lo.u64 %rd6, %rd6, 4;\n"
               << "  add.u64 %rd6, %rd4, %rd6;\n"
               << "  st.shared.f32 [%rd6], %f34;\n$skip" << step
               << ":\n  bar.sync 0;\n";
  }
  ptx.stream << "  setp.ne.u32 %p2, %r0, 0;\n"
                "  @%p2 bra $done;\n"
                "  ld.shared.f32 %f35, [%rd4];\n"
                "  ld.param.u64 %rd7, [p1];\n"
                "  ld.global.f32 %f36, [%rd7];\n"
                "  add.rn.f32 %f37, %f35, %f36;\n";
  if (!fused) {
    ptx.stream << "  ld.param.u64 %rd8, [p2];\n"
                  "  cvt.u64.u32 %rd9, %r1;\n"
                  "  mul.lo.u64 %rd9, %rd9, 4;\n"
                  "  add.u64 %rd8, %rd8, %rd9;\n"
                  "  st.global.f32 [%rd8], %f37;\n";
    ptx.finish();
    return ptx.text;
  }

  unsigned foldConsumers = 0;
  for (MapOp map : epilogueMaps) {
    for (auto [operandNumber, operand] : llvm::enumerate(map.getInputs())) {
      if (operand != fold.getResult(0))
        continue;
      if (!isEligibleFoldMapEdge(fold, map, operandNumber))
        return failure();
      ++foldConsumers;
    }
  }
  if (foldConsumers != 1)
    return failure();
  int64_t outputBuffer = materializationTask.getOutputBuffers().front();
  if (outputBuffer < 0 || outputBuffer >= buffers->size())
    return failure();
  auto output =
      dyn_cast<RankedTensorType>((*buffers)[outputBuffer].getTensorType());
  auto mapAttributes = materializationTask->getAttrOfType<ArrayAttr>(
      "epilogue_input_maps");
  if (!output || !output.getElementType().isF32() || !mapAttributes ||
      mapAttributes.size() != materializationTask.getInputBuffers().size() -
                                  fold->getNumOperands())
    return failure();
  FailureOr<int64_t> finalElements = staticElementCount(output);
  if (failed(finalElements) || *finalElements != outputExtent)
    return failure();
  ptx.stream << "  cvt.u64.u32 %rd0, %r1;\n";
  FailureOr<SmallVector<int64_t>> coordinates =
      emitOutputCoordinates(ptx, output);
  if (failed(coordinates))
    return failure();

  llvm::SmallDenseSet<Operation *> group;
  group.insert(fold.getOperation());
  for (MapOp map : epilogueMaps)
    group.insert(map.getOperation());
  SmallVector<Value> boundaryInputs;
  llvm::SmallDenseSet<Value> seenBoundary;
  for (MapOp map : llvm::reverse(epilogueMaps)) {
    for (Value operand : map.getInputs()) {
      if (!group.contains(operand.getDefiningOp()) &&
          seenBoundary.insert(operand).second)
        boundaryInputs.push_back(operand);
    }
  }
  if (boundaryInputs.size() != mapAttributes.size())
    return failure();
  DenseMap<Value, int64_t> values;
  values[fold.getResult(0)] = 37;
  ptx.f32 = 38;
  ptx.u64 = std::max<int64_t>(ptx.u64, 10);
  ArrayRef<int64_t> mapBuffers =
      materializationTask.getInputBuffers().drop_front(fold->getNumOperands());
  for (auto [ordinal, value, bufferOrdinal, mapAttribute] : llvm::enumerate(
           boundaryInputs, mapBuffers, mapAttributes.getValue())) {
    if (bufferOrdinal < 0 || bufferOrdinal >= buffers->size())
      return failure();
    auto input = dyn_cast<RankedTensorType>(
        (*buffers)[bufferOrdinal].getTensorType());
    auto indexing = dyn_cast<AffineMapAttr>(mapAttribute);
    if (!input || !indexing || input != value.getType())
      return failure();
    FailureOr<int64_t> loaded = emitMapInput(
        ptx, fold->getNumOperands() + ordinal, input, indexing.getValue(),
        *coordinates);
    if (failed(loaded))
      return failure();
    values[value] = *loaded;
  }
  for (MapOp map : epilogueMaps) {
    SmallVector<int64_t> arguments;
    for (Value operand : map.getInputs()) {
      auto found = values.find(operand);
      if (found == values.end())
        return failure();
      arguments.push_back(found->second);
    }
    FailureOr<int64_t> result =
        emitScalarBody(ptx, map.getBody().front(), arguments);
    if (failed(result) || map.getNumResults() != 1)
      return failure();
    values[map.getResult(0)] = *result;
  }
  MapOp finalMap = epilogueMaps.back();
  auto result = values.find(finalMap.getResult(0));
  if (result == values.end())
    return failure();
  int64_t outputPointer = ptx.nextU64();
  int64_t outputParameter = materializationTask.getInputBuffers().size();
  ptx.stream << "  ld.param.u64 %rd" << outputPointer << ", [p"
             << outputParameter << "];\n  mad.lo.u64 %rd" << outputPointer
             << ", %rd0, 4, %rd" << outputPointer
             << ";\n  st.global.f32 [%rd" << outputPointer << "], %f"
             << result->second << ";\n";
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

struct FoldRegionPtxContext {
  PtxEmitter &ptx;
  MaterializationTaskOp task;
  SmallVectorImpl<MaterializationBufferOp> &buffers;
  llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations;
  DenseMap<Operation *, int64_t> foldValues;
};

FailureOr<int64_t> foldRegionBoundaryParameter(FoldRegionPtxContext &context,
                                               MapOp map, unsigned operand) {
  auto uses =
      context.task->getAttrOfType<ArrayAttr>("fold_region_boundary_uses");
  if (!uses)
    return failure();
  for (Attribute attribute : uses) {
    auto use = dyn_cast<DictionaryAttr>(attribute);
    if (!use || use.get("source") != map.getSource() ||
        cast<IntegerAttr>(use.get("operand")).getInt() != operand)
      continue;
    int64_t buffer = cast<IntegerAttr>(use.get("input_buffer")).getInt();
    auto found = llvm::find(context.task.getInputBuffers(), buffer);
    if (found == context.task.getInputBuffers().end())
      return failure();
    return std::distance(context.task.getInputBuffers().begin(), found);
  }
  return failure();
}

FailureOr<SmallVector<int64_t>>
emitMappedCoordinates(PtxEmitter &ptx, AffineMap map,
                      ArrayRef<int64_t> coordinates) {
  if (map.getNumDims() != coordinates.size() || map.getNumSymbols() != 0)
    return failure();
  SmallVector<int64_t> result;
  for (AffineExpr expression : map.getResults()) {
    FailureOr<int64_t> coordinate =
        emitAffineCoordinate(ptx, expression, coordinates);
    if (failed(coordinate))
      return failure();
    result.push_back(*coordinate);
  }
  return result;
}

FailureOr<int64_t> emitFoldRegionValue(FoldRegionPtxContext &context,
                                       Value value,
                                       ArrayRef<int64_t> coordinates) {
  Operation *producer = value.getDefiningOp();
  if (auto fold = dyn_cast_or_null<FoldOp>(producer)) {
    auto found = context.foldValues.find(fold);
    return found == context.foldValues.end() ? FailureOr<int64_t>(failure())
                                             : found->second;
  }
  auto map = dyn_cast_or_null<MapOp>(producer);
  if (!map || map.getNumResults() != 1 || map.getResult(0) != value)
    return failure();
  auto output = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  if (!output || output.getRank() != static_cast<int64_t>(coordinates.size()))
    return failure();
  SmallVector<int64_t> arguments;
  for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
    auto indexing =
        dyn_cast<AffineMapAttr>(map.getIndexingMaps()[operandOrdinal]);
    if (!indexing)
      return failure();
    Operation *operandProducer = operand.getDefiningOp();
    if (isa_and_nonnull<MapOp>(operandProducer)) {
      FailureOr<SmallVector<int64_t>> operandCoordinates =
          emitMappedCoordinates(context.ptx, indexing.getValue(), coordinates);
      if (failed(operandCoordinates))
        return failure();
      FailureOr<int64_t> argument =
          emitFoldRegionValue(context, operand, *operandCoordinates);
      if (failed(argument))
        return failure();
      arguments.push_back(*argument);
      continue;
    }
    if (isa_and_nonnull<FoldOp>(operandProducer)) {
      auto found = context.foldValues.find(operandProducer);
      if (found != context.foldValues.end()) {
        if (!isCanonicalFoldResultExpansion(map, operandOrdinal))
          return failure();
        arguments.push_back(found->second);
        continue;
      }
    }
    FailureOr<int64_t> parameter =
        foldRegionBoundaryParameter(context, map, operandOrdinal);
    if (failed(parameter))
      return failure();
    int64_t bufferOrdinal = context.task.getInputBuffers()[*parameter];
    if (bufferOrdinal < 0 || bufferOrdinal >= context.buffers.size())
      return failure();
    auto input = dyn_cast<RankedTensorType>(
        context.buffers[bufferOrdinal].getTensorType());
    if (!input || input != operand.getType())
      return failure();
    FailureOr<int64_t> argument = emitMapInput(
        context.ptx, *parameter, input, indexing.getValue(), coordinates);
    if (failed(argument))
      return failure();
    arguments.push_back(*argument);
  }
  return emitScalarBody(context.ptx, map.getBody().front(), arguments);
}

FailureOr<std::string> emitFoldRegionPtx(
    MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> &buffers, ScheduleTaskOp task,
    llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations) {
  auto sourceAttributes =
      materializationTask->getAttrOfType<ArrayAttr>("fold_region_sources");
  auto phaseAttributes =
      materializationTask->getAttrOfType<ArrayAttr>("fold_region_phases");
  if (!sourceAttributes || !phaseAttributes || phaseAttributes.empty() ||
      materializationTask.getOutputBuffers().size() != 1)
    return failure();
  SmallVector<Operation *> sources;
  llvm::SmallDenseSet<Operation *> sourceSet;
  for (Attribute source : sourceAttributes) {
    Operation *operation = sourceOperations.lookup(source);
    if (!operation || !sourceSet.insert(operation).second)
      return failure();
    sources.push_back(operation);
  }
  SmallVector<FoldOp> phases;
  for (Attribute attribute : phaseAttributes) {
    auto dictionary = dyn_cast<DictionaryAttr>(attribute);
    auto fold = dictionary ? dyn_cast_or_null<FoldOp>(sourceOperations.lookup(
                                 dictionary.get("source")))
                           : FoldOp{};
    if (!fold)
      return failure();
    phases.push_back(fold);
  }
  FoldOp primary = phases.front();
  auto domain = cast<RankedTensorType>(primary.getInputs().front().getType());
  auto primaryResult = cast<RankedTensorType>(primary.getResult(0).getType());
  FailureOr<int64_t> outputExtent = staticElementCount(primaryResult);
  if (failed(outputExtent) || *outputExtent <= 0 ||
      task.getGridShape() != ArrayRef<int64_t>({*outputExtent}) ||
      task.getWorkgroupThreads() <= 0 || task.getWorkgroupThreads() > 256 ||
      task.getSerialTiles() <= 0)
    return failure();
  ArrayRef<int64_t> primaryReduction = primary.getReductionDimensions();
  const bool row = primaryReduction.size() == 1 &&
                   primaryReduction.front() == domain.getRank() - 1;
  const bool fullPrefix =
      primaryReduction.size() == static_cast<size_t>(domain.getRank() - 1) &&
      llvm::all_of(llvm::enumerate(primaryReduction), [](auto axis) {
        return static_cast<int64_t>(axis.index()) == axis.value();
      });
  if (!row && !fullPrefix)
    return failure();
  FailureOr<int64_t> reductionExtent =
      fullPrefix ? scheduleElementCount(domain.getShape().drop_back())
                 : FailureOr<int64_t>(domain.getShape().back());
  if (failed(reductionExtent))
    return failure();
  if (task.getScratchBytes() !=
      task.getWorkgroupThreads() * static_cast<int64_t>(sizeof(float)))
    return failure();

  PtxEmitter ptx(materializationTask.getInputBuffers().size() + 1);
  ptx.header(task.getSemanticFingerprint(), /*usesDynamicSharedMemory=*/true);
  FoldRegionPtxContext context{ptx, materializationTask, buffers,
                               sourceOperations,
                               DenseMap<Operation *, int64_t>()};
  int64_t accumulator = ptx.nextF32();
  ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                "  mov.u32 %r1, %ctaid.x;\n"
                "  setp.ge.u32 %p0, %r1, "
             << *outputExtent
             << ";\n  @%p0 bra $done;\n"
                "  mul.lo.u32 %r4, %r0, "
             << task.getSerialTiles() << ";\n  add.u32 %r7, %r4, "
             << task.getSerialTiles() << ";\n  min.u32 %r7, %r7, "
             << *reductionExtent << ";\n  mov.f32 %f" << accumulator
             << ", 0f00000000;\n$fold_region_load:\n"
                "  setp.ge.u32 %p1, %r4, %r7;\n"
                "  @%p1 bra $fold_region_loaded;\n"
                "  cvt.u64.u32 %rd0, %r4;\n"
                "  cvt.u64.u32 %rd1, %r1;\n";
  if (fullPrefix)
    ptx.stream << "  mad.lo.u64 %rd0, %rd0, " << *outputExtent << ", %rd1;\n";
  else
    ptx.stream << "  mad.lo.u64 %rd0, %rd1, " << *reductionExtent
               << ", %rd0;\n";
  FailureOr<SmallVector<int64_t>> reductionCoordinates =
      emitOutputCoordinates(ptx, domain);
  if (failed(reductionCoordinates))
    return failure();
  FailureOr<int64_t> leaf = emitFoldRegionValue(
      context, primary.getInputs().front(), *reductionCoordinates);
  if (failed(leaf))
    return failure();
  int64_t combined = ptx.nextF32();
  ptx.stream << "  add.rn.f32 %f" << combined << ", %f" << accumulator << ", %f"
             << *leaf << ";\n  mov.f32 %f" << accumulator << ", %f" << combined
             << ";\n  add.u32 %r4, %r4, 1;\n"
                "  bra $fold_region_load;\n$fold_region_loaded:\n";

  ptx.stream << "  cvt.u64.u32 %rd2, %r0;\n"
                "  mul.lo.u64 %rd2, %rd2, 4;\n"
                "  cvta.to.shared.u64 %rd3, shuttle_smem;\n"
                "  add.u64 %rd2, %rd3, %rd2;\n"
                "  st.shared.f32 [%rd2], %f"
             << accumulator << ";\n  bar.sync 0;\n";
  for (int64_t active = task.getWorkgroupThreads(); active > 1;
       active = (active + 1) / 2) {
    int64_t pairs = active / 2;
    int64_t left = ptx.nextF32();
    int64_t right = ptx.nextF32();
    int64_t sum = ptx.nextF32();
    ptx.stream << "  setp.ge.u32 %p2, %r0, " << pairs
               << ";\n  @%p2 bra $fold_region_skip_pairs" << active
               << ";\n  mul.lo.u32 %r5, %r0, 2;\n"
                  "  cvt.u64.u32 %rd4, %r5;\n"
                  "  mul.lo.u64 %rd4, %rd4, 4;\n"
                  "  add.u64 %rd4, %rd3, %rd4;\n"
                  "  ld.shared.f32 %f"
               << left << ", [%rd4];\n  ld.shared.f32 %f" << right
               << ", [%rd4+4];\n  add.rn.f32 %f" << sum << ", %f" << left
               << ", %f" << right
               << ";\n  cvt.u64.u32 %rd5, %r0;\n"
                  "  mul.lo.u64 %rd5, %rd5, 4;\n"
                  "  add.u64 %rd5, %rd3, %rd5;\n"
                  "  st.shared.f32 [%rd5], %f"
               << sum << ";\n$fold_region_skip_pairs" << active << ":\n";
    if (active % 2 != 0) {
      int64_t carry = ptx.nextF32();
      ptx.stream << "  setp.ne.u32 %p3, %r0, " << pairs
                 << ";\n  @%p3 bra $fold_region_skip_carry" << active
                 << ";\n  ld.shared.f32 %f" << carry << ", [%rd3+"
                 << (active - 1) * static_cast<int64_t>(sizeof(float))
                 << "];\n  st.shared.f32 [%rd3+"
                 << pairs * static_cast<int64_t>(sizeof(float)) << "], %f"
                 << carry << ";\n$fold_region_skip_carry" << active << ":\n";
    }
    ptx.stream << "  bar.sync 0;\n";
  }
  int64_t primaryValue = ptx.nextF32();
  ptx.stream << "  ld.shared.f32 %f" << primaryValue << ", [%rd3];\n";
  context.foldValues[primary] = primaryValue;

  for (FoldOp fold : ArrayRef<FoldOp>(phases).drop_front()) {
    auto input = cast<RankedTensorType>(fold.getInputs().front().getType());
    ptx.stream << "  cvt.u64.u32 %rd0, %r1;\n";
    FailureOr<SmallVector<int64_t>> coordinates =
        emitOutputCoordinates(ptx, input);
    if (failed(coordinates))
      return failure();
    FailureOr<int64_t> value =
        emitFoldRegionValue(context, fold.getInputs().front(), *coordinates);
    if (failed(value))
      return failure();
    int64_t initializer = ptx.nextF32();
    int64_t combined = ptx.nextF32();
    ptx.stream << "  mov.f32 %f" << initializer
               << ", 0f00000000;\n  add.rn.f32 %f" << combined << ", %f"
               << initializer << ", %f" << *value << ";\n";
    context.foldValues[fold] = combined;
  }

  Operation *finalOperation = sources.back();
  int64_t outputBuffer = materializationTask.getOutputBuffers().front();
  if (outputBuffer < 0 || outputBuffer >= buffers.size())
    return failure();
  auto output = cast<RankedTensorType>(buffers[outputBuffer].getTensorType());
  FailureOr<int64_t> outputElements = staticElementCount(output);
  if (failed(outputElements) || *outputElements % *outputExtent != 0)
    return failure();
  int64_t elementsPerBlock = *outputElements / *outputExtent;
  int64_t outputParameter = materializationTask.getInputBuffers().size();
  if (auto finalFold = dyn_cast<FoldOp>(finalOperation)) {
    auto value = context.foldValues.find(finalFold);
    if (value == context.foldValues.end() || elementsPerBlock != 1)
      return failure();
    ptx.stream << "  setp.ne.u32 %p3, %r0, 0;\n"
                  "  @%p3 bra $done;\n"
                  "  ld.param.u64 %rd6, [p"
               << outputParameter
               << "];\n  cvt.u64.u32 %rd7, %r1;\n"
                  "  mad.lo.u64 %rd6, %rd7, 4, %rd6;\n"
                  "  st.global.f32 [%rd6], %f"
               << value->second << ";\n";
  } else {
    auto finalMap = dyn_cast<MapOp>(finalOperation);
    if (!finalMap || finalMap.getResult(0).getType() != output)
      return failure();
    ptx.stream << "  mov.u32 %r6, %r0;\n$fold_region_epilogue:\n"
                  "  setp.ge.u32 %p3, %r6, "
               << elementsPerBlock
               << ";\n  @%p3 bra $done;\n"
                  "  cvt.u64.u32 %rd0, %r1;\n"
                  "  cvt.u64.u32 %rd6, %r6;\n";
    if (row) {
      ptx.stream << "  mad.lo.u64 %rd0, %rd0, " << elementsPerBlock
                 << ", %rd6;\n";
    } else {
      ptx.stream << "  mad.lo.u64 %rd0, %rd6, " << *outputExtent << ", %rd0;\n";
    }
    FailureOr<SmallVector<int64_t>> coordinates =
        emitOutputCoordinates(ptx, output);
    if (failed(coordinates))
      return failure();
    FailureOr<int64_t> value =
        emitFoldRegionValue(context, finalMap.getResult(0), *coordinates);
    if (failed(value))
      return failure();
    int64_t pointer = ptx.nextU64();
    int64_t bytes = output.getElementType().isBF16() ? 2 : 4;
    ptx.stream << "  ld.param.u64 %rd" << pointer << ", [p" << outputParameter
               << "];\n  mad.lo.u64 %rd" << pointer << ", %rd0, " << bytes
               << ", %rd" << pointer << ";\n";
    if (output.getElementType().isBF16()) {
      int64_t rounded = ptx.nextBF16();
      ptx.stream << "  cvt.rn.bf16.f32 %h" << rounded << ", %f" << *value
                 << ";\n  st.global.u16 [%rd" << pointer << "], %h" << rounded
                 << ";\n";
    } else {
      ptx.stream << "  st.global.f32 [%rd" << pointer << "], %f" << *value
                 << ";\n";
    }
    ptx.stream << "  add.u32 %r6, %r6, " << task.getWorkgroupThreads()
               << ";\n  bra $fold_region_epilogue;\n";
  }
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitFoldPtx(FoldOp fold, ScheduleTaskOp task) {
  return emitFoldMapPtx(fold, task, {}, {}, nullptr);
}

FailureOr<GpuExecutableSpec>
deriveGpuExecutable(RegionOp region, MaterializationPlanOp materialization,
                    SchedulePlanOp schedule) {
  SmallVector<Operation *> algebra;
  for (Operation &operation : region.getBody().front().without_terminator())
    algebra.push_back(&operation);
  llvm::SmallDenseMap<Attribute, Operation *> sourceOperations;
  for (Operation *operation : algebra) {
    if (auto map = dyn_cast<MapOp>(operation))
      sourceOperations.try_emplace(map.getSource(), operation);
    else if (auto fold = dyn_cast<FoldOp>(operation))
      sourceOperations.try_emplace(fold.getSource(), operation);
    else if (auto contract = dyn_cast<ContractOp>(operation))
      sourceOperations.try_emplace(contract.getSource(), operation);
  }
  SmallVector<MaterializationBufferOp> materializationBuffers(
      materialization.getBody().front().getOps<MaterializationBufferOp>());
  SmallVector<MaterializationTaskOp> materializationTasks(
      materialization.getBody().front().getOps<MaterializationTaskOp>());
  SmallVector<ScheduleTaskOp> tasks(
      schedule.getBody().front().getOps<ScheduleTaskOp>());
  if (algebra.empty() || materializationTasks.size() != tasks.size() ||
      tasks.size() > kMaximumGpuExecutableRecords)
    return failure();

  GpuExecutableSpec spec;
  FailureOr<SmallVector<ExecutableSlotSpec>> slots =
      deriveExecutableSlots(materialization);
  if (failed(slots) || slots->empty() ||
      slots->size() > kMaximumGpuExecutableRecords)
    return failure();
  spec.slots = std::move(*slots);
  FailureOr<SmallVector<int64_t>> reuseGroups =
      deriveGpuReuseGroups(materialization, spec.slots);
  if (failed(reuseGroups))
    return failure();
  spec.reuseGroups = std::move(*reuseGroups);
  for (auto [ordinal, materializationTask, task] :
       llvm::enumerate(materializationTasks, tasks)) {
    Operation *operation =
        sourceOperations.lookup(materializationTask.getSource());
    if (!operation)
      return failure();
    FailureOr<std::string> ptx = failure();
    if (materializationTask->getAttr("fold_region_sources")) {
      ptx = emitFoldRegionPtx(materializationTask, materializationBuffers, task,
                              sourceOperations);
    } else if (auto epilogueSources =
                   materializationTask->getAttrOfType<ArrayAttr>(
                       "fold_epilogue_sources")) {
      auto fold = dyn_cast<FoldOp>(operation);
      SmallVector<MapOp> maps;
      for (Attribute source : epilogueSources) {
        auto map = dyn_cast_or_null<MapOp>(sourceOperations.lookup(source));
        if (!map)
          return failure();
        maps.push_back(map);
      }
      if (!fold)
        return failure();
      ptx = emitFoldMapPtx(fold, task, maps, materializationTask,
                           &materializationBuffers);
    } else if (auto fusedSources =
                   materializationTask->getAttrOfType<ArrayAttr>(
                       "fused_sources")) {
      SmallVector<MapOp> maps;
      for (Attribute source : fusedSources) {
        auto map = dyn_cast_or_null<MapOp>(sourceOperations.lookup(source));
        if (!map)
          return failure();
        maps.push_back(map);
      }
      ptx = emitFusedMapPtx(maps, materializationTask,
                            materializationBuffers, task);
    } else if (auto map = dyn_cast<MapOp>(operation)) {
      ptx = emitMapPtx(map, task);
    } else if (auto fold = dyn_cast<FoldOp>(operation)) {
      ptx = emitFoldPtx(fold, task);
    } else if (auto contract = dyn_cast<ContractOp>(operation)) {
      ptx = emitContractPtx(contract, task);
    }
    if (failed(ptx)) {
      operation->emitOpError("cannot emit bounded generic SM90 PTX");
      return failure();
    }
    if (ptx->empty() || !StringRef(*ptx).ends_with("\n") ||
        ptx->size() > kMaximumGpuEntryCodeBytes ||
        task.getGridShape().size() != 1 || task.getGridShape()[0] <= 0 ||
        task.getGridShape()[0] > UINT32_MAX ||
        task.getWorkgroupThreads() <= 0 || task.getWorkgroupThreads() > 1024 ||
        task.getScratchBytes() < 0 ||
        task.getScratchBytes() > kMaximumGpuDynamicSharedMemoryBytes) {
      task.emitOpError("has invalid bounded GPU code or launch resources");
      return failure();
    }
    int64_t offset = spec.code.size();
    spec.code.append(ptx->begin(), ptx->end());
    SmallVector<int64_t> inputs(task.getInputBuffers());
    SmallVector<int64_t> outputs(task.getOutputBuffers());
    spec.entries.push_back(
        {static_cast<int64_t>(ordinal),
         offset,
         static_cast<int64_t>(ptx->size()),
         std::move(inputs),
         std::move(outputs),
         SmallVector<int64_t>(task.getDependencies()),
         task.getKind() == ScheduleTaskKind::Scalar
             ? ExecutablePredication::None
             : ExecutablePredication::DomainBounds,
         isa<ContractOp>(operation) ? std::nullopt : task.getReductionOrder(),
         {task.getGridShape()[0], 1, 1},
         {static_cast<int64_t>(task.getWorkgroupThreads()), 1, 1},
         static_cast<int64_t>(task.getScratchBytes()),
         static_cast<int64_t>(task.getInputBuffers().size() +
                              task.getOutputBuffers().size()),
         sha256(*ptx)});
  }
  if (spec.code.size() > kMaximumGpuCodeBytes)
    return failure();
  return spec;
}

ArrayAttr accessArray(MLIRContext *context, ExecutableAccess access,
                      size_t count) {
  SmallVector<Attribute> values(count,
                                ExecutableAccessAttr::get(context, access));
  return ArrayAttr::get(context, values);
}

SchedulePlanOp
uniqueScheduleForMaterialization(ArrayRef<SchedulePlanOp> schedules,
                                 MaterializationPlanOp materialization) {
  SchedulePlanOp match;
  for (SchedulePlanOp schedule : schedules) {
    if (schedule.getSourcePlanFingerprint() != materialization.getFingerprint())
      continue;
    if (match)
      return {};
    match = schedule;
  }
  return match;
}

struct GpuBuildUnit {
  RegionOp region;
  MaterializationPlanOp materialization;
  SchedulePlanOp schedule;
  GpuExecutableSpec spec;
};

bool hasBoundedGpuModuleResources(ArrayRef<GpuBuildUnit> units) {
  if (units.empty() || units.size() > kMaximumGpuExecutableRecords)
    return false;
  uint64_t entries = 0;
  uint64_t slots = 0;
  uint64_t codeBytes = 0;
  uint64_t temporaryBytes = 0;
  uint64_t taskPositions = 0;
  for (const GpuBuildUnit &unit : units) {
    entries += unit.spec.entries.size();
    slots += unit.spec.slots.size();
    codeBytes += unit.spec.code.size();
    if (entries > kMaximumGpuExecutableRecords ||
        slots > kMaximumGpuExecutableRecords ||
        codeBytes > kMaximumGpuCodeBytes)
      return false;
    llvm::DenseMap<int64_t, uint64_t> groupBytes;
    for (auto [slot, reuseGroup] :
         llvm::zip_equal(unit.spec.slots, unit.spec.reuseGroups)) {
      if (slot.storage != MaterializationStorage::Temporary)
        continue;
      uint64_t previous = groupBytes.lookup(reuseGroup);
      uint64_t required = slot.requiredBytes;
      if (required > previous) {
        uint64_t increase = required - previous;
        if (increase > kMaximumGpuTemporaryBytes - temporaryBytes)
          return false;
        temporaryBytes += increase;
        groupBytes[reuseGroup] = required;
      }
    }
    for (const GpuEntrySpec &entry : unit.spec.entries) {
      uint64_t positions = 1;
      auto multiplyExtents = [&](ArrayRef<int64_t> extents) {
        for (int64_t extent : extents) {
          if (extent <= 0 || static_cast<uint64_t>(extent) >
                                 kMaximumGpuAggregateTaskPositions / positions)
            return false;
          positions *= extent;
        }
        return true;
      };
      if (!multiplyExtents(entry.grid) || !multiplyExtents(entry.block))
        return false;
      if (positions > kMaximumGpuAggregateTaskPositions - taskPositions)
        return false;
      taskPositions += positions;
    }
  }
  return true;
}

class BuildGpuExecutableBundlePass
    : public impl::ShuttleBuildGpuExecutableBundlePassBase<
          BuildGpuExecutableBundlePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<MaterializationPlanOp> materializations(
        module.getOps<MaterializationPlanOp>());
    SmallVector<SchedulePlanOp> schedules(module.getOps<SchedulePlanOp>());
    if (materializations.empty() ||
        materializations.size() != schedules.size() ||
        !module.getOps<DeviceModuleOp>().empty() ||
        !module.getOps<InvocationAbiOp>().empty() ||
        !module.getOps<ExecutableBundleOp>().empty()) {
      module.emitError(
          "requires source-bound plans and no GPU executable bundles");
      return signalPassFailure();
    }
    SmallVector<GpuBuildUnit, 0> units;
    llvm::SmallDenseSet<Operation *> regions;
    llvm::SmallDenseSet<Operation *> matchedSchedules;
    for (MaterializationPlanOp materialization : materializations) {
      RegionOp region;
      SchedulePlanOp schedule =
          uniqueScheduleForMaterialization(schedules, materialization);
      if (!schedule ||
          failed(verifyMaterializationPlanAgainstSource(module, materialization,
                                                        &region)) ||
          !regions.insert(region.getOperation()).second ||
          !matchedSchedules.insert(schedule.getOperation()).second ||
          failed(schedule.verifyRegions()) ||
          failed(verifySchedulePlanAgainstMaterialization(materialization,
                                                          schedule))) {
        module.emitError("GPU plans do not form one-to-one source bindings");
        return signalPassFailure();
      }
      FailureOr<GpuExecutableSpec> spec =
          deriveGpuExecutable(region, materialization, schedule);
      if (failed(spec)) {
        module.emitError("failed to build exact GPU executable bundle");
        return signalPassFailure();
      }
      units.push_back({region, materialization, schedule, std::move(*spec)});
    }
    size_t regionCount = 0;
    module.walk([&](RegionOp) { ++regionCount; });
    if (regions.size() != regionCount || !hasBoundedGpuModuleResources(units)) {
      module.emitError("GPU executable bundle aggregate exceeds closed bounds");
      return signalPassFailure();
    }
    for (GpuBuildUnit &unit : units) {
      MaterializationPlanOp materialization = unit.materialization;
      SchedulePlanOp schedule = unit.schedule;
      GpuExecutableSpec &spec = unit.spec;
      OpBuilder builder(module.getContext());
      builder.setInsertionPointToEnd(module.getBody());
      const std::string codeDigest = executableCodeDigest(spec.code);
      OperationState deviceState(module.getLoc(),
                                 DeviceModuleOp::getOperationName());
      deviceState.addAttribute("schema_version", builder.getI64IntegerAttr(3));
      deviceState.addAttribute(
          "code_format",
          ExecutableCodeFormatAttr::get(module.getContext(),
                                        ExecutableCodeFormat::CudaPtxSm90V1));
      deviceState.addAttribute("policy", schedule.getPolicyAttr());
      deviceState.addAttribute("source_schedule_fingerprint",
                               schedule.getFingerprintAttr());
      deviceState.addAttribute(
          "code", DenseI8ArrayAttr::get(module.getContext(), spec.code));
      deviceState.addAttribute("code_digest",
                               builder.getStringAttr(codeDigest));
      deviceState.addAttribute("fingerprint",
                               builder.getStringAttr(std::string(64, '0')));
      deviceState.addRegion();
      auto device = cast<DeviceModuleOp>(builder.create(deviceState));
      device.getBody().push_back(new Block());
      builder.setInsertionPointToEnd(&device.getBody().front());
      for (const GpuEntrySpec &entry : spec.entries) {
        SmallVector<NamedAttribute> attributes{
            builder.getNamedAttr("ordinal",
                                 builder.getI64IntegerAttr(entry.ordinal)),
            builder.getNamedAttr("source_task",
                                 builder.getI64IntegerAttr(entry.ordinal)),
            builder.getNamedAttr("code_offset",
                                 builder.getI64IntegerAttr(entry.codeOffset)),
            builder.getNamedAttr("code_length",
                                 builder.getI64IntegerAttr(entry.codeLength)),
            builder.getNamedAttr(
                "input_buffers",
                DenseI64ArrayAttr::get(module.getContext(), entry.inputs)),
            builder.getNamedAttr(
                "output_buffers",
                DenseI64ArrayAttr::get(module.getContext(), entry.outputs)),
            builder.getNamedAttr("input_accesses",
                                 accessArray(module.getContext(),
                                             ExecutableAccess::Read,
                                             entry.inputs.size())),
            builder.getNamedAttr("output_accesses",
                                 accessArray(module.getContext(),
                                             ExecutableAccess::Write,
                                             entry.outputs.size())),
            builder.getNamedAttr("dependencies",
                                 DenseI64ArrayAttr::get(module.getContext(),
                                                        entry.dependencies)),
            builder.getNamedAttr("predication",
                                 ExecutablePredicationAttr::get(
                                     module.getContext(), entry.predication)),
            builder.getNamedAttr("code_digest",
                                 builder.getStringAttr(entry.codeDigest)),
            builder.getNamedAttr("grid", DenseI64ArrayAttr::get(
                                             module.getContext(), entry.grid)),
            builder.getNamedAttr(
                "block",
                DenseI64ArrayAttr::get(module.getContext(), entry.block)),
            builder.getNamedAttr(
                "dynamic_shared_bytes",
                builder.getI64IntegerAttr(entry.dynamicSharedBytes)),
            builder.getNamedAttr("kernel_arity",
                                 builder.getI64IntegerAttr(entry.kernelArity))};
        if (entry.reductionOrder)
          attributes.push_back(builder.getNamedAttr(
              "reduction_order",
              ScheduleReductionOrderAttr::get(module.getContext(),
                                              *entry.reductionOrder)));
        createScheduleRecord(builder, module.getLoc(),
                             DeviceEntryOp::getOperationName(), attributes);
      }
      builder.create<DeviceModuleYieldOp>(module.getLoc());
      device.setFingerprint(deviceModuleFingerprint(device));

      builder.setInsertionPointToEnd(module.getBody());
      OperationState abiState(module.getLoc(),
                              InvocationAbiOp::getOperationName());
      abiState.addAttribute("schema_version", builder.getI64IntegerAttr(3));
      abiState.addAttribute("source_plan_fingerprint",
                            materialization.getFingerprintAttr());
      abiState.addAttribute("source_schedule_fingerprint",
                            schedule.getFingerprintAttr());
      abiState.addAttribute("fingerprint",
                            builder.getStringAttr(std::string(64, '0')));
      abiState.addRegion();
      auto abi = cast<InvocationAbiOp>(builder.create(abiState));
      abi.getBody().push_back(new Block());
      builder.setInsertionPointToEnd(&abi.getBody().front());
      int64_t operandBinding = 0;
      int64_t resultBinding = 0;
      for (auto [ordinal, slot] : llvm::enumerate(spec.slots)) {
        ExecutableBindingKind binding = ExecutableBindingKind::None;
        std::optional<int64_t> bindingIndex;
        if (slot.storage == MaterializationStorage::External &&
            slot.access == ExecutableAccess::Read) {
          binding = ExecutableBindingKind::Operand;
          bindingIndex = operandBinding++;
        } else if (slot.storage == MaterializationStorage::External &&
                   slot.access == ExecutableAccess::Write) {
          binding = ExecutableBindingKind::Result;
          bindingIndex = resultBinding++;
        }
        SmallVector<NamedAttribute> attributes{
            builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr("source_buffer",
                                 builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr("tensor_type", TypeAttr::get(slot.tensorType)),
            builder.getNamedAttr("required_bytes",
                                 builder.getI64IntegerAttr(slot.requiredBytes)),
            builder.getNamedAttr(
                "strides",
                DenseI64ArrayAttr::get(module.getContext(), slot.strides)),
            builder.getNamedAttr("offset", builder.getI64IntegerAttr(0)),
            builder.getNamedAttr("alignment",
                                 builder.getI64IntegerAttr(slot.alignment)),
            builder.getNamedAttr(
                "address_space",
                ExecutableAddressSpaceAttr::get(
                    module.getContext(), ExecutableAddressSpace::Device)),
            builder.getNamedAttr(
                "access",
                ExecutableAccessAttr::get(module.getContext(), slot.access)),
            builder.getNamedAttr(
                "storage", MaterializationStorageAttr::get(module.getContext(),
                                                           slot.storage)),
            builder.getNamedAttr("alias_group",
                                 builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr("reuse_group", builder.getI64IntegerAttr(
                                                    spec.reuseGroups[ordinal])),
            builder.getNamedAttr("binding", ExecutableBindingKindAttr::get(
                                                module.getContext(), binding))};
        if (bindingIndex)
          attributes.push_back(builder.getNamedAttr(
              "binding_index", builder.getI64IntegerAttr(*bindingIndex)));
        createScheduleRecord(builder, module.getLoc(),
                             InvocationSlotOp::getOperationName(), attributes);
      }
      builder.create<InvocationAbiYieldOp>(module.getLoc());
      abi.setFingerprint(invocationAbiFingerprint(abi));

      builder.setInsertionPointToEnd(module.getBody());
      OperationState bundleState(module.getLoc(),
                                 ExecutableBundleOp::getOperationName());
      bundleState.addAttribute("schema_version", builder.getI64IntegerAttr(2));
      bundleState.addAttribute("source_schedule_fingerprint",
                               schedule.getFingerprintAttr());
      bundleState.addAttribute("device_module_fingerprint",
                               device.getFingerprintAttr());
      bundleState.addAttribute("invocation_abi_fingerprint",
                               abi.getFingerprintAttr());
      bundleState.addAttribute(
          "completion",
          ExecutableCompletionAttr::get(module.getContext(),
                                        ExecutableCompletion::StreamOrdered));
      bundleState.addAttribute("fingerprint",
                               builder.getStringAttr(std::string(64, '0')));
      auto bundle = cast<ExecutableBundleOp>(builder.create(bundleState));
      bundle.setFingerprint(executableBundleFingerprint(bundle));
    }
  }
};

DenseIntElementsAttr rowMajorLayout(RankedTensorType type) {
  SmallVector<int64_t> order;
  for (int64_t axis = type.getRank(); axis > 0; --axis) {
    order.push_back(axis - 1);
  }
  auto layoutType = RankedTensorType::get({type.getRank()},
                                          IndexType::get(type.getContext()));
  return DenseIntElementsAttr::get(layoutType, order);
}

LogicalResult verifyGpuExecutableUnit(ModuleOp module, RegionOp region,
                                      MaterializationPlanOp materialization,
                                      SchedulePlanOp schedule,
                                      DeviceModuleOp device,
                                      InvocationAbiOp abi,
                                      ExecutableBundleOp bundle) {
  if (failed(verifyMaterializationPlanAgainstSource(module, materialization)) ||
      failed(schedule.verifyRegions()) ||
      failed(verifySchedulePlanAgainstMaterialization(materialization,
                                                      schedule)) ||
      failed(device.verifyRegions()) || failed(abi.verifyRegions()) ||
      failed(bundle.verify()))
    return failure();
  FailureOr<GpuExecutableSpec> expected =
      deriveGpuExecutable(region, materialization, schedule);
  if (failed(expected) || device.getSchemaVersion() != 3 ||
      device.getCodeFormat() != ExecutableCodeFormat::CudaPtxSm90V1 ||
      device.getPolicy() != schedule.getPolicy() ||
      device.getSourceScheduleFingerprint() != schedule.getFingerprint() ||
      abi.getSchemaVersion() != 3 ||
      abi.getSourcePlanFingerprint() != materialization.getFingerprint() ||
      abi.getSourceScheduleFingerprint() != schedule.getFingerprint() ||
      bundle.getSchemaVersion() != 2 ||
      bundle.getSourceScheduleFingerprint() != schedule.getFingerprint() ||
      bundle.getDeviceModuleFingerprint() != device.getFingerprint() ||
      bundle.getInvocationAbiFingerprint() != abi.getFingerprint() ||
      bundle.getCompletion() != ExecutableCompletion::StreamOrdered ||
      device.getCode() != ArrayRef<int8_t>(expected->code))
    return failure();
  SmallVector<DeviceEntryOp> entries(
      device.getBody().front().getOps<DeviceEntryOp>());
  SmallVector<InvocationSlotOp> slots(
      abi.getBody().front().getOps<InvocationSlotOp>());
  if (entries.size() != expected->entries.size() ||
      slots.size() != expected->slots.size())
    return failure();
  for (auto [actual, spec] : llvm::zip_equal(entries, expected->entries)) {
    if (actual.getOrdinal() != spec.ordinal ||
        actual.getSourceTask() != spec.ordinal ||
        actual.getCodeOffset() != spec.codeOffset ||
        actual.getCodeLength() != spec.codeLength ||
        actual.getInputBuffers() != ArrayRef<int64_t>(spec.inputs) ||
        actual.getOutputBuffers() != ArrayRef<int64_t>(spec.outputs) ||
        actual.getDependencies() != ArrayRef<int64_t>(spec.dependencies) ||
        actual.getPredication() != spec.predication ||
        actual.getReductionOrder() != spec.reductionOrder ||
        actual.getGrid() != ArrayRef<int64_t>(spec.grid) ||
        actual.getBlock() != ArrayRef<int64_t>(spec.block) ||
        actual.getDynamicSharedBytes() != spec.dynamicSharedBytes ||
        actual.getKernelArity() != spec.kernelArity ||
        actual.getCodeDigest() != spec.codeDigest)
      return failure();
  }
  int64_t operandBinding = 0;
  int64_t resultBinding = 0;
  for (auto [ordinal, actual, spec] : llvm::enumerate(slots, expected->slots)) {
    ExecutableBindingKind binding = ExecutableBindingKind::None;
    std::optional<int64_t> bindingIndex;
    if (spec.storage == MaterializationStorage::External &&
        spec.access == ExecutableAccess::Read) {
      binding = ExecutableBindingKind::Operand;
      bindingIndex = operandBinding++;
    } else if (spec.storage == MaterializationStorage::External &&
               spec.access == ExecutableAccess::Write) {
      binding = ExecutableBindingKind::Result;
      bindingIndex = resultBinding++;
    }
    if (actual.getTensorType() != spec.tensorType ||
        actual.getRequiredBytes() != spec.requiredBytes ||
        actual.getStrides() != ArrayRef<int64_t>(spec.strides) ||
        actual.getOffset() != 0 || actual.getAlignment() != spec.alignment ||
        actual.getAddressSpace() != ExecutableAddressSpace::Device ||
        actual.getAccess() != spec.access ||
        actual.getStorage() != spec.storage ||
        actual.getAliasGroup() != actual.getOrdinal() ||
        actual.getReuseGroup() != expected->reuseGroups[ordinal] ||
        actual.getBinding() != binding ||
        actual.getBindingIndex() != bindingIndex)
      return failure();
  }
  return success();
}

LogicalResult verifyGpuExecutableAgainstSource(ModuleOp module) {
  SmallVector<MaterializationPlanOp> materializations(
      module.getOps<MaterializationPlanOp>());
  SmallVector<SchedulePlanOp> schedules(module.getOps<SchedulePlanOp>());
  SmallVector<DeviceModuleOp> devices(module.getOps<DeviceModuleOp>());
  SmallVector<InvocationAbiOp> abis(module.getOps<InvocationAbiOp>());
  SmallVector<ExecutableBundleOp> bundles(module.getOps<ExecutableBundleOp>());
  if (materializations.empty() || materializations.size() != schedules.size() ||
      materializations.size() != devices.size() ||
      materializations.size() != abis.size() ||
      materializations.size() != bundles.size())
    return failure();

  llvm::SmallDenseSet<Operation *> regions;
  llvm::SmallDenseSet<Operation *> matchedSchedules;
  llvm::SmallDenseSet<Operation *> matchedDevices;
  llvm::SmallDenseSet<Operation *> matchedAbis;
  llvm::SmallDenseSet<Operation *> matchedBundles;
  SmallVector<GpuBuildUnit, 0> units;
  for (MaterializationPlanOp materialization : materializations) {
    RegionOp region;
    if (failed(verifyMaterializationPlanAgainstSource(module, materialization,
                                                      &region)) ||
        !regions.insert(region.getOperation()).second)
      return failure();
    SchedulePlanOp schedule =
        uniqueScheduleForMaterialization(schedules, materialization);
    if (!schedule || !matchedSchedules.insert(schedule.getOperation()).second)
      return failure();

    DeviceModuleOp device;
    for (DeviceModuleOp candidate : devices) {
      if (candidate.getSourceScheduleFingerprint() != schedule.getFingerprint())
        continue;
      if (device)
        return failure();
      device = candidate;
    }
    InvocationAbiOp abi;
    for (InvocationAbiOp candidate : abis) {
      if (candidate.getSourcePlanFingerprint() !=
              materialization.getFingerprint() ||
          candidate.getSourceScheduleFingerprint() != schedule.getFingerprint())
        continue;
      if (abi)
        return failure();
      abi = candidate;
    }
    if (!device || !abi ||
        !matchedDevices.insert(device.getOperation()).second ||
        !matchedAbis.insert(abi.getOperation()).second)
      return failure();

    ExecutableBundleOp bundle;
    for (ExecutableBundleOp candidate : bundles) {
      if (candidate.getSourceScheduleFingerprint() !=
              schedule.getFingerprint() ||
          candidate.getDeviceModuleFingerprint() != device.getFingerprint() ||
          candidate.getInvocationAbiFingerprint() != abi.getFingerprint())
        continue;
      if (bundle)
        return failure();
      bundle = candidate;
    }
    if (!bundle || !matchedBundles.insert(bundle.getOperation()).second ||
        failed(verifyGpuExecutableUnit(module, region, materialization,
                                       schedule, device, abi, bundle)))
      return failure();
    FailureOr<GpuExecutableSpec> spec =
        deriveGpuExecutable(region, materialization, schedule);
    if (failed(spec))
      return failure();
    units.push_back({region, materialization, schedule, std::move(*spec)});
  }
  size_t regionCount = 0;
  module.walk([&](RegionOp) { ++regionCount; });
  if (regions.size() != regionCount || !hasBoundedGpuModuleResources(units))
    return failure();
  return success();
}

class VerifyGpuExecutableBundlePass
    : public impl::ShuttleVerifyGpuExecutableBundlePassBase<
          VerifyGpuExecutableBundlePass> {
public:
  void runOnOperation() override {
    if (failed(verifyGpuExecutableAgainstSource(getOperation()))) {
      getOperation().emitError("GPU executable bundle no longer matches "
                               "Shuttle algebra and schedule");
      signalPassFailure();
    }
  }
};

struct PreparedGpuCall {
  RegionOp region;
  SmallVector<uint8_t> bundleBytes;
  std::string digest;
  SmallVector<Attribute> operandLayouts;
  SmallVector<Attribute> resultLayouts;
};

FailureOr<PreparedGpuCall> prepareGpuCall(ModuleOp module, RegionOp region,
                                          DeviceModuleOp device,
                                          InvocationAbiOp abi,
                                          ExecutableBundleOp bundle) {
  if (device.getPolicy() != region.getPolicy())
    return failure();
  auto closedExternalType = [](Value value) {
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    return tensor && tensor.hasStaticShape() && tensor.getRank() >= 1 &&
           tensor.getRank() <= 8 && tensor.getElementType().isBF16();
  };
  if (!llvm::all_of(region.getInputs(), closedExternalType) ||
      !llvm::all_of(region.getResults(), closedExternalType))
    return failure();
  SmallVector<InvocationSlotOp> operandSlots(region.getInputs().size());
  SmallVector<InvocationSlotOp> resultSlots(region.getResults().size());
  for (InvocationSlotOp slot :
       abi.getBody().front().getOps<InvocationSlotOp>()) {
    if (!slot.getBindingIndex())
      continue;
    SmallVector<InvocationSlotOp> *projection = nullptr;
    if (slot.getBinding() == ExecutableBindingKind::Operand)
      projection = &operandSlots;
    else if (slot.getBinding() == ExecutableBindingKind::Result)
      projection = &resultSlots;
    else
      return failure();
    int64_t index = *slot.getBindingIndex();
    if (index < 0 || index >= static_cast<int64_t>(projection->size()) ||
        (*projection)[index])
      return failure();
    (*projection)[index] = slot;
  }
  for (auto [index, slot] : llvm::enumerate(operandSlots)) {
    if (!slot || slot.getTensorType() != region.getInputs()[index].getType())
      return failure();
  }
  for (auto [index, slot] : llvm::enumerate(resultSlots)) {
    if (!slot || slot.getTensorType() != region.getResults()[index].getType())
      return failure();
  }
  OpBuilder builder(module.getContext());
  OwningOpRef<ModuleOp> stripped = ModuleOp::create(module.getLoc());
  builder.setInsertionPointToEnd(stripped->getBody());
  builder.insert(device->clone());
  builder.insert(abi->clone());
  builder.insert(bundle->clone());
  FailureOr<SmallVector<uint8_t>> bundleBytes =
      serializeGpuExecutableBundle(*stripped);
  if (failed(bundleBytes))
    return failure();
  PreparedGpuCall prepared{region, std::move(*bundleBytes), {}, {}, {}};
  prepared.digest = gpuExecutableBundleDigest(prepared.bundleBytes);
  for (Value input : region.getInputs())
    prepared.operandLayouts.push_back(
        rowMajorLayout(cast<RankedTensorType>(input.getType())));
  for (Value result : region.getResults())
    prepared.resultLayouts.push_back(
        rowMajorLayout(cast<RankedTensorType>(result.getType())));
  return prepared;
}

class ReplaceWithGpuExecutableBundlePass
    : public impl::ShuttleReplaceWithGpuExecutableBundlePassBase<
          ReplaceWithGpuExecutableBundlePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyGpuExecutableAgainstSource(module))) {
      module.emitError("requires source-bound closed GPU executables");
      return signalPassFailure();
    }
    SmallVector<MaterializationPlanOp> materializations(
        module.getOps<MaterializationPlanOp>());
    SmallVector<SchedulePlanOp> schedules(module.getOps<SchedulePlanOp>());
    SmallVector<DeviceModuleOp> devices(module.getOps<DeviceModuleOp>());
    SmallVector<InvocationAbiOp> abis(module.getOps<InvocationAbiOp>());
    SmallVector<ExecutableBundleOp> bundles(
        module.getOps<ExecutableBundleOp>());
    SmallVector<PreparedGpuCall> preparedCalls;
    uint64_t aggregateTransportBytes = 0;
    for (MaterializationPlanOp materialization : materializations) {
      RegionOp region;
      if (failed(verifyMaterializationPlanAgainstSource(module, materialization,
                                                        &region)))
        return signalPassFailure();
      SchedulePlanOp schedule =
          uniqueScheduleForMaterialization(schedules, materialization);
      DeviceModuleOp device;
      for (DeviceModuleOp candidate : devices) {
        if (candidate.getSourceScheduleFingerprint() ==
            schedule.getFingerprint())
          device = candidate;
      }
      InvocationAbiOp abi;
      for (InvocationAbiOp candidate : abis) {
        if (candidate.getSourcePlanFingerprint() ==
                materialization.getFingerprint() &&
            candidate.getSourceScheduleFingerprint() ==
                schedule.getFingerprint())
          abi = candidate;
      }
      ExecutableBundleOp bundle;
      for (ExecutableBundleOp candidate : bundles) {
        if (candidate.getSourceScheduleFingerprint() ==
                schedule.getFingerprint() &&
            candidate.getDeviceModuleFingerprint() == device.getFingerprint() &&
            candidate.getInvocationAbiFingerprint() == abi.getFingerprint())
          bundle = candidate;
      }
      FailureOr<PreparedGpuCall> prepared =
          prepareGpuCall(module, region, device, abi, bundle);
      if (failed(prepared) ||
          prepared->bundleBytes.size() >
              kMaximumGpuTransportBytes - aggregateTransportBytes) {
        module.emitError("failed to prepare bounded GPU custom calls");
        return signalPassFailure();
      }
      aggregateTransportBytes += prepared->bundleBytes.size();
      preparedCalls.push_back(std::move(*prepared));
    }
    OpBuilder builder(module.getContext());
    for (PreparedGpuCall &prepared : preparedCalls) {
      RegionOp region = prepared.region;
      StringRef opaque(
          reinterpret_cast<const char *>(prepared.bundleBytes.data()),
          prepared.bundleBytes.size());
      DictionaryAttr backendConfig = builder.getDictionaryAttr({
          builder.getNamedAttr("bundle_bytes", builder.getStringAttr(opaque)),
          builder.getNamedAttr("bundle_sha256",
                               builder.getStringAttr(prepared.digest)),
          builder.getNamedAttr("bundle_size", builder.getI64IntegerAttr(
                                                  prepared.bundleBytes.size())),
          builder.getNamedAttr("transport_schema_version",
                               builder.getI64IntegerAttr(2)),
          builder.getNamedAttr("device_schema_version",
                               builder.getI64IntegerAttr(3)),
          builder.getNamedAttr("invocation_abi_schema_version",
                               builder.getI64IntegerAttr(3)),
          builder.getNamedAttr("bundle_schema_version",
                               builder.getI64IntegerAttr(2)),
          builder.getNamedAttr("completion",
                               builder.getStringAttr("stream_ordered")),
      });
      builder.setInsertionPoint(region);
      OperationState callState(region.getLoc(), "stablehlo.custom_call");
      callState.addOperands(region.getInputs());
      callState.addTypes(region.getResultTypes());
      callState.addAttribute(
          "call_target_name",
          builder.getStringAttr(kGpuExecutableBundleFfiTarget));
      callState.addAttribute("api_version", builder.getI32IntegerAttr(4));
      callState.addAttribute("backend_config", backendConfig);
      callState.addAttribute("called_computations", builder.getArrayAttr({}));
      callState.addAttribute("has_side_effect", builder.getBoolAttr(false));
      callState.addAttribute("operand_layouts",
                             builder.getArrayAttr(prepared.operandLayouts));
      callState.addAttribute("output_operand_aliases",
                             builder.getArrayAttr({}));
      callState.addAttribute("result_layouts",
                             builder.getArrayAttr(prepared.resultLayouts));
      Operation *call = builder.create(callState);
      region.getResults().replaceAllUsesWith(call->getResults());
      region.erase();
    }
    for (Operation &operation :
         llvm::make_early_inc_range(module.getBody()->without_terminator())) {
      if (isa<MaterializationPlanOp, SchedulePlanOp, DeviceModuleOp,
              InvocationAbiOp, ExecutableBundleOp>(operation))
        operation.erase();
    }
  }
};

void emitObserverSnapshot(
    const std::shared_ptr<detail::ShuttleObserverInvocation> &invocation,
    ModuleOp module, ShuttlePipelinePhase phase,
    llvm::StringRef failurePass = {}) {
  std::string regionMembership;
  std::string coverageManifest;
  std::string unsupportedFingerprint;
  if (auto manifest =
          module->getAttrOfType<DictionaryAttr>(kCoverageManifestAttribute)) {
    coverageManifest = attributeText(manifest);
    if (ArrayAttr regions =
            manifest.getAs<ArrayAttr>(kManifestSelectedRegions)) {
      regionMembership = attributeText(regions);
    }
    if (ArrayAttr excluded = manifest.getAs<ArrayAttr>(kManifestExcluded)) {
      unsupportedFingerprint = sha256(attributeText(excluded));
    }
  }
  invocation->emit(
      phase, std::move(regionMembership), std::move(coverageManifest),
      std::move(unsupportedFingerprint),
      phase == ShuttlePipelinePhase::GeneratedBundle
          ? generatedGpuBundleFingerprintImpl(module)
      : phase == ShuttlePipelinePhase::FinalErasure
          ? normalizedModuleFingerprintImpl(module)
          : std::string{},
      phase == ShuttlePipelinePhase::FinalErasure, failurePass.str());
}

class EmitObserverPass
    : public PassWrapper<EmitObserverPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitObserverPass)

  EmitObserverPass(
      ShuttlePipelinePhase phase,
      std::shared_ptr<detail::ShuttleObserverInvocation> invocation)
      : phase(phase), invocation(std::move(invocation)) {}

  StringRef getArgument() const final { return "shuttle-observer-event"; }
  StringRef getDescription() const final {
    return "Emit one immutable Shuttle pipeline observer snapshot";
  }

  void runOnOperation() override {
    emitObserverSnapshot(invocation, getOperation(), phase);
  }

private:
  ShuttlePipelinePhase phase;
  std::shared_ptr<detail::ShuttleObserverInvocation> invocation;
};

struct FailureCaptureState {
  std::string pass;
};

class FailureCaptureInstrumentation final : public PassInstrumentation {
public:
  explicit FailureCaptureInstrumentation(
      std::shared_ptr<FailureCaptureState> state)
      : state(std::move(state)) {}

  void runAfterPassFailed(Pass *pass, Operation *) final {
    llvm::StringRef argument = pass->getArgument();
    state->pass = argument.empty() ? pass->getName().str() : argument.str();
  }

private:
  std::shared_ptr<FailureCaptureState> state;
};

void buildShuttleStablehloCorePipeline(
    OpPassManager &manager, const ShuttlePipelineOptions &options,
    const std::shared_ptr<detail::ShuttleObserverInvocation> &invocation) {
  manager.addPass(createAnnotateSourcePass());
  manager.addPass(createFormStructuralRegionsPass(
      options.numerics, options.canonicalOptions, options.canonicalTuning));
  manager.addPass(createConvertStablehloToAlgebraPass());
  manager.addPass(createVerifySourceCoveragePass());
  manager.addPass(std::make_unique<EmitObserverPass>(
      ShuttlePipelinePhase::AlgebraCoverage, invocation));
  manager.addPass(createVerifySemanticErasurePass());
  manager.addPass(createShuttleCanonicalizePass());
  manager.addPass(createLowerAlgebraToStablehloPass());
  manager.addPass(createVerifySourceCoveragePass());
  manager.addPass(std::make_unique<EmitObserverPass>(
      ShuttlePipelinePhase::LoweredCoverage, invocation));
  manager.addPass(createStripSourceProvenancePass());
  manager.addPass(createVerifyNoShuttleOpsPass());
  manager.addPass(std::make_unique<EmitObserverPass>(
      ShuttlePipelinePhase::FinalErasure, invocation));
}

void buildShuttleGpuExecutablePipeline(
    OpPassManager &manager, const ShuttlePipelineOptions &options,
    const std::shared_ptr<detail::ShuttleObserverInvocation> &invocation) {
  manager.addPass(createAnnotateSourcePass());
  manager.addPass(createFormStructuralRegionsPass(
      options.numerics, options.canonicalOptions, options.canonicalTuning));
  manager.addPass(createConvertStablehloToAlgebraPass());
  manager.addPass(createVerifySourceCoveragePass());
  manager.addPass(std::make_unique<EmitObserverPass>(
      ShuttlePipelinePhase::AlgebraCoverage, invocation));
  manager.addPass(createVerifySemanticErasurePass());
  manager.addPass(createShuttleCanonicalizePass());
  manager.addPass(createPlanRowFoldMaterializationPass());
  manager.addPass(createVerifyMaterializationPlanPass());
  manager.addPass(createPlanSimt32RowFoldSchedulePass());
  manager.addPass(createVerifySimt32RowFoldSchedulePass());
  manager.addPass(createBuildGpuExecutableBundlePass());
  manager.addPass(createVerifyGpuExecutableBundlePass());
  manager.addPass(createVerifySourceCoveragePass());
  manager.addPass(std::make_unique<EmitObserverPass>(
      ShuttlePipelinePhase::GeneratedBundle, invocation));
  manager.addPass(createReplaceWithGpuExecutableBundlePass());
  manager.addPass(createStripSourceProvenancePass());
  manager.addPass(createVerifyNoShuttleOpsPass());
  manager.addPass(std::make_unique<EmitObserverPass>(
      ShuttlePipelinePhase::FinalErasure, invocation));
}

class RunShuttleStablehloPipelinePass
    : public PassWrapper<RunShuttleStablehloPipelinePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RunShuttleStablehloPipelinePass)
  using Base =
      PassWrapper<RunShuttleStablehloPipelinePass, OperationPass<ModuleOp>>;

  RunShuttleStablehloPipelinePass(
      ShuttlePipelineOptions options,
      std::shared_ptr<const ShuttlePipelineObserver> observer)
      : options(std::move(options)), observer(std::move(observer)) {}
  RunShuttleStablehloPipelinePass(const RunShuttleStablehloPipelinePass &other)
      : Base(other), options(other.options), observer(other.observer) {}

  StringRef getArgument() const final { return "shuttle-stablehlo-pipeline"; }
  StringRef getDescription() const final {
    return "Run one observed fail-closed Shuttle StableHLO invocation";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, ShuttleDialect,
                    stablehlo::StablehloDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto invocation = detail::beginShuttleObserverInvocation(
        shuttlePipelineIdentity(options), observer);
    auto failure = std::make_shared<FailureCaptureState>();
    PassManager manager(module.getContext(), ModuleOp::getOperationName());
    manager.addInstrumentation(
        std::make_unique<FailureCaptureInstrumentation>(failure));
    if (options.executionMode == ExecutionMode::GpuExecutableBundle) {
      buildShuttleGpuExecutablePipeline(manager, options, invocation);
    } else {
      buildShuttleStablehloCorePipeline(manager, options, invocation);
    }
    if (failed(manager.run(module))) {
      emitObserverSnapshot(invocation, module, ShuttlePipelinePhase::Failure,
                           failure->pass);
      signalPassFailure();
    }
  }

private:
  ShuttlePipelineOptions options;
  std::shared_ptr<const ShuttlePipelineObserver> observer;
};

} // namespace

std::string normalizedStablehloFingerprint(ModuleOp module) {
  return normalizedModuleFingerprintImpl(module);
}

std::string generatedGpuBundleFingerprint(ModuleOp module) {
  return generatedGpuBundleFingerprintImpl(module);
}

std::unique_ptr<Pass> createAnnotateSourcePass() {
  return std::make_unique<AnnotateSourcePass>();
}

std::unique_ptr<Pass> createFormStructuralRegionsPass() {
  return std::make_unique<FormStructuralRegionsPass>();
}

std::unique_ptr<Pass>
createFormStructuralRegionsPass(NumericalPolicy numerics,
                                std::string canonicalOptions,
                                std::string canonicalTuning) {
  return std::make_unique<FormStructuralRegionsPass>(
      numerics, std::move(canonicalOptions), std::move(canonicalTuning));
}

std::unique_ptr<Pass> createConvertStablehloToAlgebraPass() {
  return std::make_unique<ConvertStablehloToAlgebraPass>();
}

std::unique_ptr<Pass> createVerifySourceCoveragePass() {
  return std::make_unique<VerifySourceCoveragePass>();
}

std::unique_ptr<Pass> createVerifySemanticErasurePass() {
  return std::make_unique<VerifySemanticErasurePass>();
}

std::unique_ptr<Pass> createShuttleCanonicalizePass() {
  return std::make_unique<ShuttleCanonicalizePass>();
}

std::unique_ptr<Pass> createLowerAlgebraToStablehloPass() {
  return std::make_unique<LowerAlgebraToStablehloPass>();
}

std::unique_ptr<Pass> createStripSourceProvenancePass() {
  return std::make_unique<StripSourceProvenancePass>();
}

std::unique_ptr<Pass> createVerifyNoShuttleOpsPass() {
  return std::make_unique<VerifyNoShuttleOpsPass>();
}

std::unique_ptr<Pass> createPlanRowFoldMaterializationPass() {
  return std::make_unique<PlanRowFoldMaterializationPass>();
}

std::unique_ptr<Pass> createVerifyMaterializationPlanPass() {
  return std::make_unique<VerifyMaterializationPlanPass>();
}

std::unique_ptr<Pass> createPlanSimt32RowFoldSchedulePass() {
  return std::make_unique<PlanSimt32RowFoldSchedulePass>();
}

std::unique_ptr<Pass> createVerifySimt32RowFoldSchedulePass() {
  return std::make_unique<VerifySimt32RowFoldSchedulePass>();
}

std::unique_ptr<Pass> createBuildGpuExecutableBundlePass() {
  return std::make_unique<BuildGpuExecutableBundlePass>();
}

std::unique_ptr<Pass> createVerifyGpuExecutableBundlePass() {
  return std::make_unique<VerifyGpuExecutableBundlePass>();
}

std::unique_ptr<Pass> createReplaceWithGpuExecutableBundlePass() {
  return std::make_unique<ReplaceWithGpuExecutableBundlePass>();
}

void buildShuttleStablehloPipeline(
    OpPassManager &manager, const ShuttlePipelineOptions &options,
    std::shared_ptr<const ShuttlePipelineObserver> observer) {
  manager.addPass(std::make_unique<RunShuttleStablehloPipelinePass>(
      options, std::move(observer)));
}

void registerShuttleStablehloPipelines() {
  PassPipelineRegistration<>(
      "shuttle-stablehlo-source-ordered-pipeline",
      "Run the complete source-ordered Shuttle StableHLO pipeline",
      [](OpPassManager &manager) {
        buildShuttleStablehloPipeline(
            manager,
            commandLinePipelineOptions(NumericalPolicy::SourceOrdered));
      });
  PassPipelineRegistration<>(
      "shuttle-stablehlo-fast-pipeline",
      "Run the complete fast-policy Shuttle StableHLO pipeline",
      [](OpPassManager &manager) {
        buildShuttleStablehloPipeline(
            manager, commandLinePipelineOptions(NumericalPolicy::Fast));
      });
  PassPipelineRegistration<>(
      "shuttle-gpu-executable-bundle-pipeline",
      "Replace a closed source-ordered Map/Fold graph with a CUDA typed FFI "
      "call",
      [](OpPassManager &manager) {
        ShuttlePipelineOptions options =
            commandLinePipelineOptions(NumericalPolicy::SourceOrdered);
        options.executionMode = ExecutionMode::GpuExecutableBundle;
        options.canonicalOptions =
            R"json({"execution_mode":"gpu_executable_bundle","numerics":"source_ordered","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
        buildShuttleStablehloPipeline(manager, options);
      });
  PassPipelineRegistration<>(
      "shuttle-gpu-executable-bundle-fast-pipeline",
      "Replace a closed fast-policy Map/Fold graph with a CUDA typed FFI call",
      [](OpPassManager &manager) {
        ShuttlePipelineOptions options =
            commandLinePipelineOptions(NumericalPolicy::Fast);
        options.executionMode = ExecutionMode::GpuExecutableBundle;
        options.canonicalOptions =
            R"json({"execution_mode":"gpu_executable_bundle","numerics":"fast","pipeline_abi_version":22,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
        buildShuttleStablehloPipeline(manager, options);
      });
}

} // namespace mlir::shuttle
