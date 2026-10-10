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
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
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
#include "llvm/Support/JSON.h"
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
#define GEN_PASS_DEF_SHUTTLEPLANSOURCEEVENTLOWERINGPASS
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
constexpr llvm::StringLiteral kOperationRefGroupsAttribute =
    "shuttle.operation_ref_groups";
constexpr llvm::StringLiteral kCoverageManifestAttribute =
    "shuttle.coverage_manifest";
constexpr llvm::StringLiteral kRegionResultSourcesAttribute =
    "shuttle.result_source_refs";
constexpr llvm::StringLiteral kScanResultSourcesAttribute = "result_sources";
constexpr llvm::StringLiteral kScanErasedTerminatorRecordsAttribute =
    "erased_terminator_records";
constexpr llvm::StringLiteral kDefaultCanonicalTuning =
    R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";

std::string
physicalEventProvenanceFingerprint(StringRef scheduleFingerprint,
                                   StringRef launchRecipeFingerprint,
                                   StringRef canonicalScheduleDigest) {
  llvm::SHA256 digest;
  digest.update(scheduleFingerprint);
  digest.update(launchRecipeFingerprint);
  digest.update(canonicalScheduleDigest);
  return llvm::toHex(digest.final(), true);
}

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
        name == kOperationRefGroupsAttribute ||
        name == kCoverageManifestAttribute ||
        name == kRegionResultSourcesAttribute ||
        name == kScanResultSourcesAttribute ||
        name == kScanErasedTerminatorRecordsAttribute) {
      continue;
    }
    attributes.push_back(attribute);
  }
  return DictionaryAttr::get(operation->getContext(), attributes);
}

bool isCompositeDecompositionFunction(func::FuncOp function, ModuleOp module) {
  bool referenced = false;
  module.walk([&](Operation *operation) {
    if (operation->getName().getStringRef() != "stablehlo.composite")
      return WalkResult::advance();
    auto decomposition =
        operation->getAttrOfType<FlatSymbolRefAttr>("decomposition");
    if (decomposition && SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                             operation, decomposition) == function) {
      referenced = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return referenced;
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
  if (isa<ScanOp>(owner)) {
    if (auto refs = owner->getAttrOfType<ArrayAttr>(
            kScanResultSourcesAttribute);
        refs && result.getResultNumber() < refs.size()) {
      return refs[result.getResultNumber()];
    }
  }
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

bool enclosingFunctionContainsScatter(Operation *operation) {
  auto function = operation->getParentOfType<func::FuncOp>();
  bool found = false;
  if (function)
    function.walk([&](Operation *candidate) {
      found |= candidate->getName().getStringRef() ==
               stablehlo::ScatterOp::getOperationName();
    });
  return found;
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
  if (!input || !result ||
      !(input.getElementType().isF32() || input.getElementType().isBF16()) ||
      input.getElementType() != result.getElementType() ||
      (input.getElementType().isBF16() &&
       !enclosingFunctionContainsScatter(operation)) ||
      !input.hasStaticShape() || !result.hasStaticShape() || !dimensions ||
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

FailureOr<AffineMap> contiguousReshapeInputMap(RankedTensorType input,
                                               RankedTensorType result) {
  if (!input.hasStaticShape() || !result.hasStaticShape() ||
      input.getNumElements() != result.getNumElements() ||
      input.getRank() > 8 || result.getRank() > 8 ||
      llvm::any_of(input.getShape(),
                   [](int64_t extent) { return extent <= 0; }) ||
      llvm::any_of(result.getShape(),
                   [](int64_t extent) { return extent <= 0; })) {
    return failure();
  }
  MLIRContext *context = input.getContext();
  AffineExpr linear = getAffineConstantExpr(0, context);
  for (auto [dimension, extent] : llvm::enumerate(result.getShape())) {
    linear = linear * extent + getAffineDimExpr(dimension, context);
  }
  SmallVector<AffineExpr> inputCoordinates(input.getRank());
  int64_t stride = 1;
  for (int64_t dimension = input.getRank() - 1; dimension >= 0; --dimension) {
    const int64_t extent = input.getDimSize(dimension);
    inputCoordinates[dimension] = linear.floorDiv(stride) % extent;
    stride *= extent;
  }
  return AffineMap::get(result.getRank(), 0, inputCoordinates, context);
}

bool isCopyFreeStaticReshape(Operation *operation) {
  if (operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
      !sourceAttributes(operation).empty()) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!input || !result || input.getElementType() != result.getElementType() ||
      (!input.getElementType().isF32() && !input.getElementType().isBF16()) ||
      input == result) {
    return false;
  }
  return succeeded(contiguousReshapeInputMap(input, result));
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

bool hasSupportedDotPrecision(Operation *operation) {
  auto precision = operation->getAttrOfType<ArrayAttr>("precision_config");
  if (!precision || precision.empty()) {
    return true;
  }
  return precision.size() == operation->getNumOperands() &&
         llvm::all_of(precision, [](Attribute attribute) {
           auto value = dyn_cast<stablehlo::PrecisionAttr>(attribute);
           return value && (value.getValue() == stablehlo::Precision::DEFAULT ||
                            value.getValue() == stablehlo::Precision::HIGH ||
                            value.getValue() == stablehlo::Precision::HIGHEST);
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
  Operation &combiner = body.front();
  Operation &terminator = body.back();
  StringRef combinerName = combiner.getName().getStringRef();
  const bool supportedCombiner =
      combinerName == stablehlo::AddOp::getOperationName() ||
      combinerName == stablehlo::MaxOp::getOperationName();
  return supportedCombiner && sourceAttributes(&combiner).empty() &&
         combiner.getNumOperands() == 2 && combiner.getNumResults() == 1 &&
         combiner.getOperand(0) == body.getArgument(0) &&
         combiner.getOperand(1) == body.getArgument(1) &&
         combiner.getResult(0).getType() == body.getArgument(0).getType() &&
         terminator.getName().getStringRef() ==
             stablehlo::ReturnOp::getOperationName() &&
         sourceAttributes(&terminator).empty() &&
         terminator.getNumOperands() == 1 &&
         terminator.getOperand(0) == combiner.getResult(0);
}

bool isSupportedStablehloSelection(Operation *operation) {
  if (operation->getName().getStringRef() != "stablehlo.composite" ||
      operation->getNumOperands() != 1 || operation->getNumResults() != 2 ||
      operation->getNumRegions() != 0) {
    return false;
  }
  auto name = operation->getAttrOfType<StringAttr>("name");
  auto attributes =
      operation->getAttrOfType<DictionaryAttr>("composite_attributes");
  auto decomposition =
      operation->getAttrOfType<FlatSymbolRefAttr>("decomposition");
  auto version = operation->getAttrOfType<IntegerAttr>("version");
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto values = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  auto indices = dyn_cast<RankedTensorType>(operation->getResult(1).getType());
  auto count = attributes ? attributes.getAs<IntegerAttr>("k") : IntegerAttr{};
  if (!name || name.getValue() != "chlo.top_k" || !attributes ||
      attributes.size() != 1 || !decomposition || !version ||
      version.getInt() != 1 || !count || !count.getType().isInteger(64) ||
      !input || !values || !indices || !input.hasStaticShape() ||
      !values.hasStaticShape() || !indices.hasStaticShape() ||
      input.getRank() == 0 ||
      !(input.getElementType().isF32() || input.getElementType().isBF16()) ||
      values.getElementType() != input.getElementType() ||
      !indices.getElementType().isInteger(32)) {
    return false;
  }
  const int64_t dimension = input.getRank() - 1;
  const int64_t selectedCount = count.getInt();
  if (selectedCount <= 0 || selectedCount > input.getDimSize(dimension)) {
    return false;
  }
  SmallVector<int64_t> expectedShape(input.getShape());
  expectedShape[dimension] = selectedCount;
  if (values.getShape() != ArrayRef<int64_t>(expectedShape) ||
      indices.getShape() != ArrayRef<int64_t>(expectedShape) ||
      sourceAttributes(operation).size() != 4) {
    return false;
  }

  func::FuncOp implementation =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(operation,
                                                         decomposition);
  if (!implementation || !implementation.isPrivate() ||
      implementation.getNumArguments() != 1 ||
      implementation.getNumResults() != 2 ||
      implementation.getArgumentTypes()[0] != input ||
      implementation.getResultTypes()[0] != values ||
      implementation.getResultTypes()[1] != indices ||
      !llvm::hasSingleElement(implementation.getBody())) {
    return false;
  }
  Block &body = implementation.getBody().front();
  SmallVector<Operation *> bodyOperations;
  for (Operation &bodyOperation : body) {
    bodyOperations.push_back(&bodyOperation);
  }
  if (bodyOperations.size() != 5) {
    return false;
  }
  Operation *iota = bodyOperations[0];
  Operation *sort = bodyOperations[1];
  Operation *valuesSlice = bodyOperations[2];
  Operation *indicesSlice = bodyOperations[3];
  Operation *returnOp = bodyOperations[4];

  auto fullIndexType = RankedTensorType::get(
      input.getShape(), IntegerType::get(input.getContext(), 32));
  auto iotaDimension = iota->getAttrOfType<IntegerAttr>("iota_dimension");
  auto sortDimension = sort->getAttrOfType<IntegerAttr>("dimension");
  auto sortStable = sort->getAttrOfType<BoolAttr>("is_stable");
  if (iota->getName().getStringRef() != stablehlo::IotaOp::getOperationName() ||
      iota->getNumOperands() != 0 || iota->getNumResults() != 1 ||
      iota->getResult(0).getType() != fullIndexType || !iotaDimension ||
      iotaDimension.getInt() != dimension ||
      sourceAttributes(iota).size() != 1 ||
      sort->getName().getStringRef() != stablehlo::SortOp::getOperationName() ||
      sort->getNumOperands() != 2 ||
      sort->getOperand(0) != body.getArgument(0) ||
      sort->getOperand(1) != iota->getResult(0) || sort->getNumResults() != 2 ||
      sort->getResult(0).getType() != input ||
      sort->getResult(1).getType() != fullIndexType || !sortDimension ||
      sortDimension.getInt() != dimension || !sortStable ||
      !sortStable.getValue() || sourceAttributes(sort).size() != 2 ||
      sort->getNumRegions() != 1 ||
      !llvm::hasSingleElement(sort->getRegion(0))) {
    return false;
  }

  Block &comparator = sort->getRegion(0).front();
  SmallVector<Operation *> comparatorOperations;
  for (Operation &comparatorOperation : comparator) {
    comparatorOperations.push_back(&comparatorOperation);
  }
  auto scalarValueType = RankedTensorType::get({}, input.getElementType());
  auto scalarIndexType =
      RankedTensorType::get({}, fullIndexType.getElementType());
  if (comparator.getNumArguments() != 4 ||
      comparator.getArgument(0).getType() != scalarValueType ||
      comparator.getArgument(1).getType() != scalarValueType ||
      comparator.getArgument(2).getType() != scalarIndexType ||
      comparator.getArgument(3).getType() != scalarIndexType ||
      comparatorOperations.size() != 2 ||
      !comparator.getArgument(2).use_empty() ||
      !comparator.getArgument(3).use_empty()) {
    return false;
  }
  Operation *compare = comparatorOperations[0];
  Operation *comparatorReturn = comparatorOperations[1];
  auto direction = compare->getAttrOfType<stablehlo::ComparisonDirectionAttr>(
      "comparison_direction");
  auto comparisonType =
      compare->getAttrOfType<stablehlo::ComparisonTypeAttr>("compare_type");
  if (compare->getName().getStringRef() !=
          stablehlo::CompareOp::getOperationName() ||
      compare->getNumOperands() != 2 ||
      compare->getOperand(0) != comparator.getArgument(0) ||
      compare->getOperand(1) != comparator.getArgument(1) ||
      compare->getNumResults() != 1 ||
      compare->getResult(0).getType() !=
          RankedTensorType::get({}, IntegerType::get(input.getContext(), 1)) ||
      !direction || !comparisonType ||
      direction.getValue() != stablehlo::ComparisonDirection::GT ||
      comparisonType.getValue() != stablehlo::ComparisonType::TOTALORDER ||
      sourceAttributes(compare).size() != 2 ||
      comparatorReturn->getName().getStringRef() !=
          stablehlo::ReturnOp::getOperationName() ||
      comparatorReturn->getNumOperands() != 1 ||
      comparatorReturn->getOperand(0) != compare->getResult(0) ||
      comparatorReturn->getNumResults() != 0 ||
      !sourceAttributes(comparatorReturn).empty()) {
    return false;
  }

  auto isLeadingSelectionSlice = [&](Operation *slice, Value sorted,
                                     RankedTensorType resultType) {
    auto starts = slice->getAttrOfType<DenseI64ArrayAttr>("start_indices");
    auto limits = slice->getAttrOfType<DenseI64ArrayAttr>("limit_indices");
    auto strides = slice->getAttrOfType<DenseI64ArrayAttr>("strides");
    SmallVector<int64_t> expectedStarts(input.getRank(), 0);
    SmallVector<int64_t> expectedLimits(input.getShape());
    SmallVector<int64_t> expectedStrides(input.getRank(), 1);
    expectedLimits[dimension] = selectedCount;
    return slice->getName().getStringRef() ==
               stablehlo::SliceOp::getOperationName() &&
           slice->getNumOperands() == 1 && slice->getOperand(0) == sorted &&
           slice->getNumResults() == 1 &&
           slice->getResult(0).getType() == resultType && starts && limits &&
           strides &&
           starts.asArrayRef() == ArrayRef<int64_t>(expectedStarts) &&
           limits.asArrayRef() == ArrayRef<int64_t>(expectedLimits) &&
           strides.asArrayRef() == ArrayRef<int64_t>(expectedStrides) &&
           sourceAttributes(slice).size() == 3;
  };
  return isLeadingSelectionSlice(valuesSlice, sort->getResult(0), values) &&
         isLeadingSelectionSlice(indicesSlice, sort->getResult(1), indices) &&
         returnOp->getName().getStringRef() ==
             func::ReturnOp::getOperationName() &&
         returnOp->getNumOperands() == 2 &&
         returnOp->getOperand(0) == valuesSlice->getResult(0) &&
         returnOp->getOperand(1) == indicesSlice->getResult(0) &&
         returnOp->getNumResults() == 0 && sourceAttributes(returnOp).empty();
}

bool isSupportedStablehloRelation(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (operation->getNumResults() != 1 || operation->getNumRegions() != 0 ||
      !isMemoryEffectFree(operation)) {
    return false;
  }
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!result || !result.hasStaticShape()) {
    return false;
  }
  DictionaryAttr attributes = sourceAttributes(operation);
  if (name == stablehlo::IotaOp::getOperationName()) {
    auto dimension = operation->getAttrOfType<IntegerAttr>("iota_dimension");
    return operation->getNumOperands() == 0 &&
           result.getElementType().isInteger(32) && dimension &&
           dimension.getInt() >= 0 && dimension.getInt() < result.getRank() &&
           attributes.size() == 1;
  }
  if (name == stablehlo::ConstantOp::getOperationName()) {
    auto value = operation->getAttrOfType<DenseElementsAttr>("value");
    return operation->getNumOperands() == 0 &&
           (result.getElementType().isInteger(32) ||
            result.getElementType().isInteger(1)) &&
           value && value.isSplat() && attributes.size() == 1;
  }
  if (operation->getNumOperands() == 1) {
    auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
    if (!input || !input.hasStaticShape()) {
      return false;
    }
    if (name == stablehlo::ReshapeOp::getOperationName()) {
      return attributes.empty() && input.getElementType().isInteger(32) &&
             result.getElementType().isInteger(32) &&
             input.getNumElements() == result.getNumElements();
    }
    if (name == stablehlo::BroadcastInDimOp::getOperationName()) {
      auto dimensions =
          operation->getAttrOfType<DenseI64ArrayAttr>("broadcast_dimensions");
      return dimensions && attributes.size() == 1 &&
             dimensions.size() == static_cast<size_t>(input.getRank()) &&
             input.getElementType() == result.getElementType() &&
             (result.getElementType().isInteger(32) ||
              result.getElementType().isInteger(1)) &&
             hasValidBroadcastDimensions(operation, dimensions);
    }
    if (name == stablehlo::SliceOp::getOperationName()) {
      auto starts =
          operation->getAttrOfType<DenseI64ArrayAttr>("start_indices");
      auto limits =
          operation->getAttrOfType<DenseI64ArrayAttr>("limit_indices");
      auto strides = operation->getAttrOfType<DenseI64ArrayAttr>("strides");
      return input.getElementType().isInteger(32) &&
             result.getElementType().isInteger(32) && starts && limits &&
             strides && starts.size() == static_cast<size_t>(input.getRank()) &&
             limits.size() == starts.size() &&
             strides.size() == starts.size() && attributes.size() == 3;
    }
    if (name == stablehlo::TransposeOp::getOperationName()) {
      auto permutation =
          operation->getAttrOfType<DenseI64ArrayAttr>("permutation");
      return input.getElementType().isInteger(32) &&
             result.getElementType().isInteger(32) && permutation &&
             permutation.size() == static_cast<size_t>(input.getRank()) &&
             attributes.size() == 1;
    }
    return name == stablehlo::ConvertOp::getOperationName() &&
           attributes.empty() && input.getShape() == result.getShape() &&
           input.getElementType().isInteger(1) &&
           result.getElementType().isF32();
  }
  if (name == stablehlo::CompareOp::getOperationName() &&
      operation->getNumOperands() == 2) {
    auto lhs = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
    auto rhs = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
    auto comparisonType =
        operation->getAttrOfType<stablehlo::ComparisonTypeAttr>("compare_type");
    auto direction =
        operation->getAttrOfType<stablehlo::ComparisonDirectionAttr>(
            "comparison_direction");
    return lhs && rhs && comparisonType && direction && lhs == rhs &&
           lhs.getShape() == result.getShape() &&
           lhs.getElementType().isInteger(32) &&
           result.getElementType().isInteger(1) && attributes.size() == 2;
  }
  if ((name == stablehlo::AddOp::getOperationName() ||
       name == stablehlo::MulOp::getOperationName()) &&
      operation->getNumOperands() == 2 && attributes.empty()) {
    return operation->getOperand(0).getType() == result &&
           operation->getOperand(1).getType() == result &&
           result.getElementType().isInteger(32);
  }
  if (name == stablehlo::GatherOp::getOperationName() &&
      operation->getNumOperands() == 2) {
    auto operand =
        dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
    auto indices =
        dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
    auto dimensions =
        operation->getAttrOfType<stablehlo::GatherDimensionNumbersAttr>(
            "dimension_numbers");
    auto sliceSizes =
        operation->getAttrOfType<DenseI64ArrayAttr>("slice_sizes");
    return operand && indices && operand.hasStaticShape() &&
           indices.hasStaticShape() &&
           operand.getElementType() == result.getElementType() &&
           indices.getElementType().isInteger(32) && dimensions && sliceSizes &&
           sliceSizes.size() == static_cast<size_t>(operand.getRank()) &&
           (attributes.size() == 2 || attributes.size() == 3);
  }
  if (name == stablehlo::SelectOp::getOperationName() &&
      operation->getNumOperands() == 3 && attributes.empty()) {
    auto predicate =
        dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
    return predicate && predicate.getElementType().isInteger(1) &&
           predicate.getShape() == result.getShape() &&
           operation->getOperand(1).getType() == result &&
           operation->getOperand(2).getType() == result &&
           (result.getElementType().isF32() ||
            result.getElementType().isInteger(32));
  }
  return false;
}

bool isSupportedStablehloScatter(Operation *operation) {
  if (operation->getName().getStringRef() !=
          stablehlo::ScatterOp::getOperationName() ||
      operation->getNumOperands() != 3 || operation->getNumResults() != 1 ||
      operation->getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation->getRegion(0))) {
    return false;
  }
  auto operand = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto indices = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
  auto updates = dyn_cast<RankedTensorType>(operation->getOperand(2).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  auto dimensions =
      operation->getAttrOfType<stablehlo::ScatterDimensionNumbersAttr>(
          "scatter_dimension_numbers");
  if (!operand || !indices || !updates || !result || !dimensions ||
      !operand.hasStaticShape() || !indices.hasStaticShape() ||
      !updates.hasStaticShape() || result != operand ||
      updates.getElementType() != operand.getElementType() ||
      (!operand.getElementType().isF32() &&
       !operand.getElementType().isBF16()) ||
      !indices.getElementType().isInteger(32) || indices.getRank() == 0 ||
      dimensions.getIndexVectorDim() != indices.getRank() - 1 ||
      indices.getDimSize(indices.getRank() - 1) != 1 ||
      dimensions.getInsertedWindowDims().size() != 1 ||
      dimensions.getScatterDimsToOperandDims().size() != 1 ||
      dimensions.getInsertedWindowDims().front() !=
          dimensions.getScatterDimsToOperandDims().front() ||
      dimensions.getInputBatchingDims().size() !=
          dimensions.getScatterIndicesBatchingDims().size()) {
    return false;
  }
  llvm::SmallDenseSet<int64_t> updateWindowDims(
      dimensions.getUpdateWindowDims().begin(),
      dimensions.getUpdateWindowDims().end());
  llvm::SmallDenseSet<int64_t> batchedUpdateDims;
  for (int64_t indexBatch : dimensions.getScatterIndicesBatchingDims()) {
    if (indexBatch < 0 || indexBatch >= indices.getRank() - 1)
      return false;
    batchedUpdateDims.insert(indexBatch);
  }
  SmallVector<int64_t> routeDims;
  for (int64_t dimension = 0; dimension < updates.getRank(); ++dimension)
    if (!updateWindowDims.contains(dimension) &&
        !batchedUpdateDims.contains(dimension))
      routeDims.push_back(dimension);
  if (routeDims.empty() || routeDims.size() + batchedUpdateDims.size() !=
                               static_cast<size_t>(indices.getRank() - 1))
    return false;
  Block &body = operation->getRegion(0).front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2)
    return false;
  Operation &combiner = body.front();
  Operation &terminator = body.back();
  auto scalar = RankedTensorType::get({}, operand.getElementType());
  return body.getArgument(0).getType() == scalar &&
         body.getArgument(1).getType() == scalar &&
         combiner.getName().getStringRef() ==
             stablehlo::AddOp::getOperationName() &&
         combiner.getNumOperands() == 2 && combiner.getNumResults() == 1 &&
         combiner.getOperand(0) == body.getArgument(0) &&
         combiner.getOperand(1) == body.getArgument(1) &&
         combiner.getResult(0).getType() == scalar &&
         sourceAttributes(&combiner).empty() &&
         terminator.getName().getStringRef() ==
             stablehlo::ReturnOp::getOperationName() &&
         terminator.getNumOperands() == 1 &&
         terminator.getOperand(0) == combiner.getResult(0) &&
         sourceAttributes(&terminator).empty();
}

struct CountedScanDescription {
  int64_t induction = -1;
  int64_t extent = 0;
  ScanDirection direction = ScanDirection::Forward;
  SmallVector<int64_t> state;
  SmallVector<int64_t> invariants;
  SmallVector<int64_t> scanned;
  SmallVector<int64_t> scannedDimensions;
  SmallVector<int64_t> emitted;
  SmallVector<int64_t> emittedDimensions;
  SmallVector<Value> scannedValues;
  SmallVector<Value> emittedValues;
  Operation *step = nullptr;
};

std::optional<int64_t> stablehloSplatInteger(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (auto region = dyn_cast_or_null<RegionOp>(
            argument.getOwner()->getParentOp())) {
      value = region.getInputs()[argument.getArgNumber()];
    }
  }
  Operation *constant = value.getDefiningOp();
  if (auto region = dyn_cast_or_null<RegionOp>(constant)) {
    if (region.getInputs().empty() && region.getResults().size() == 1 &&
        llvm::hasSingleElement(region.getBody())) {
      auto yield = dyn_cast<YieldOp>(region.getBody().front().getTerminator());
      if (yield && yield.getValues().size() == 1)
        return stablehloSplatInteger(yield.getValues().front());
    }
  }
  if (!constant || constant->getName().getStringRef() !=
                       stablehlo::ConstantOp::getOperationName())
    return std::nullopt;
  auto elements = constant->getAttrOfType<DenseIntElementsAttr>("value");
  if (!elements || !elements.isSplat())
    return std::nullopt;
  return elements.getSplatValue<APInt>().getSExtValue();
}

func::FuncOp calledFunction(Operation *operation) {
  if (operation->getName().getStringRef() !=
      func::CallOp::getOperationName())
    return {};
  auto callee = operation->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!callee)
    return {};
  if (func::FuncOp function =
          SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(operation,
                                                             callee))
    return function;
  if (ModuleOp module = operation->getParentOfType<ModuleOp>())
    return module.lookupSymbol<func::FuncOp>(callee.getValue());
  return {};
}

std::optional<int64_t> indexedSliceDimension(Operation *call,
                                             Value source,
                                             Value index) {
  func::FuncOp function = calledFunction(call);
  if (!function || call->getNumOperands() != 2 ||
      call->getOperand(0) != source || call->getOperand(1) != index ||
      function.getNumArguments() != 2 || function.getNumResults() != 1 ||
      !llvm::hasSingleElement(function.getBody()))
    return std::nullopt;
  Block &body = function.getBody().front();
  Operation *slice = nullptr;
  for (Operation &candidate : body)
    if (candidate.getName().getStringRef() ==
        stablehlo::DynamicSliceOp::getOperationName()) {
      if (slice)
        return std::nullopt;
      slice = &candidate;
    }
  if (!slice || slice->getNumOperands() < 2 ||
      slice->getOperand(0) != body.getArgument(0))
    return std::nullopt;
  int64_t indexed = -1;
  for (auto [dimension, start] :
       llvm::enumerate(slice->getOperands().drop_front())) {
    if (start == body.getArgument(1)) {
      if (indexed >= 0)
        return std::nullopt;
      indexed = dimension;
      continue;
    }
    if (stablehloSplatInteger(start) != 0)
      return std::nullopt;
  }
  if (indexed < 0)
    return std::nullopt;
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  auto resultType = dyn_cast<RankedTensorType>(call->getResult(0).getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      sourceType.getRank() != resultType.getRank() + 1)
    return std::nullopt;
  SmallVector<int64_t> expected(sourceType.getShape());
  expected.erase(expected.begin() + indexed);
  if (resultType.getShape() != ArrayRef<int64_t>(expected))
    return std::nullopt;
  return indexed;
}

std::optional<int64_t> indexedUpdateDimension(Operation *call,
                                              Value destination,
                                              Value index,
                                              Value *update) {
  func::FuncOp function = calledFunction(call);
  if (!function || call->getNumOperands() != 3 ||
      call->getOperand(0) != destination || call->getOperand(2) != index ||
      function.getNumArguments() != 3 || function.getNumResults() != 1 ||
      !llvm::hasSingleElement(function.getBody()))
    return std::nullopt;
  Block &body = function.getBody().front();
  Operation *dynamicUpdate = nullptr;
  for (Operation &candidate : body)
    if (candidate.getName().getStringRef() ==
        stablehlo::DynamicUpdateSliceOp::getOperationName()) {
      if (dynamicUpdate)
        return std::nullopt;
      dynamicUpdate = &candidate;
    }
  if (!dynamicUpdate || dynamicUpdate->getNumOperands() < 3 ||
      dynamicUpdate->getOperand(0) != body.getArgument(0))
    return std::nullopt;
  int64_t indexed = -1;
  for (auto [dimension, start] :
       llvm::enumerate(dynamicUpdate->getOperands().drop_front(2))) {
    if (start == body.getArgument(2)) {
      if (indexed >= 0)
        return std::nullopt;
      indexed = dimension;
      continue;
    }
    if (stablehloSplatInteger(start) != 0)
      return std::nullopt;
  }
  if (indexed < 0)
    return std::nullopt;
  auto destinationType = dyn_cast<RankedTensorType>(destination.getType());
  auto updateType = dyn_cast<RankedTensorType>(call->getOperand(1).getType());
  if (!destinationType || !updateType || !destinationType.hasStaticShape() ||
      !updateType.hasStaticShape() ||
      destinationType.getRank() != updateType.getRank() + 1)
    return std::nullopt;
  SmallVector<int64_t> expected(destinationType.getShape());
  expected.erase(expected.begin() + indexed);
  if (updateType.getShape() != ArrayRef<int64_t>(expected))
    return std::nullopt;
  *update = call->getOperand(1);
  return indexed;
}

std::optional<ScanDirection>
scanAccessDirection(Value index, Value induction, int64_t extent,
                    ScanDirection loopDirection, Value loopBound,
                    Value unitStep) {
  if (index == induction)
    return loopDirection;
  Operation *subtractOne = index.getDefiningOp();
  if (!subtractOne || subtractOne->getName().getStringRef() !=
                          stablehlo::SubtractOp::getOperationName() ||
      subtractOne->getNumOperands() != 2 ||
      (subtractOne->getOperand(1) != unitStep &&
       stablehloSplatInteger(subtractOne->getOperand(1)) != 1))
    return std::nullopt;
  Operation *subtractInduction = subtractOne->getOperand(0).getDefiningOp();
  if (!subtractInduction || subtractInduction->getName().getStringRef() !=
                               stablehlo::SubtractOp::getOperationName() ||
      subtractInduction->getNumOperands() != 2 ||
      (subtractInduction->getOperand(0) != loopBound &&
       stablehloSplatInteger(subtractInduction->getOperand(0)) != extent) ||
      subtractInduction->getOperand(1) != induction)
    return std::nullopt;
  return loopDirection == ScanDirection::Forward ? ScanDirection::Reverse
                                                  : ScanDirection::Forward;
}

FailureOr<CountedScanDescription> describeCountedScan(Operation *operation) {
  if (operation->getName().getStringRef() !=
          stablehlo::WhileOp::getOperationName() ||
      operation->getNumRegions() != 2 || operation->getNumOperands() == 0 ||
      operation->getNumOperands() != operation->getNumResults() ||
      !llvm::hasSingleElement(operation->getRegion(0)) ||
      !llvm::hasSingleElement(operation->getRegion(1)))
    return failure();
  Block &condition = operation->getRegion(0).front();
  Block &body = operation->getRegion(1).front();
  if (condition.getNumArguments() != operation->getNumOperands() ||
      body.getNumArguments() != operation->getNumOperands() ||
      condition.getOperations().size() != 2 || body.empty())
    return failure();
  Operation &compare = condition.front();
  Operation &conditionReturn = condition.back();
  auto direction = compare.getAttrOfType<stablehlo::ComparisonDirectionAttr>(
      "comparison_direction");
  if (compare.getName().getStringRef() !=
          stablehlo::CompareOp::getOperationName() ||
      compare.getNumOperands() != 2 || compare.getNumResults() != 1 ||
      conditionReturn.getName().getStringRef() !=
          stablehlo::ReturnOp::getOperationName() ||
      conditionReturn.getNumOperands() != 1 ||
      conditionReturn.getOperand(0) != compare.getResult(0) || !direction)
    return failure();
  auto inductionArgument = dyn_cast<BlockArgument>(compare.getOperand(0));
  if (!inductionArgument || inductionArgument.getOwner() != &condition)
    return failure();
  int64_t induction = inductionArgument.getArgNumber();
  auto bodyReturn = dyn_cast<stablehlo::ReturnOp>(body.getTerminator());
  if (!bodyReturn || bodyReturn.getNumOperands() != operation->getNumOperands())
    return failure();
  CountedScanDescription description;
  description.induction = induction;
  auto cachedExtent =
      operation->getAttrOfType<IntegerAttr>("shuttle.scan_extent");
  if (cachedExtent && cachedExtent.getInt() > 0 &&
      direction.getValue() == stablehlo::ComparisonDirection::LT) {
    description.extent = cachedExtent.getInt();
    description.direction = ScanDirection::Forward;
  } else if (cachedExtent && cachedExtent.getInt() > 0 &&
             direction.getValue() == stablehlo::ComparisonDirection::GE) {
    description.extent = cachedExtent.getInt();
    description.direction = ScanDirection::Reverse;
  } else {
    std::optional<int64_t> bound =
        stablehloSplatInteger(compare.getOperand(1));
    std::optional<int64_t> initial =
        stablehloSplatInteger(operation->getOperand(induction));
    Value nextInduction = bodyReturn.getOperand(induction);
    Operation *increment = nextInduction.getDefiningOp();
    if (!bound || !initial || !increment ||
        increment->getNumOperands() != 2 ||
        increment->getOperand(0) != body.getArgument(induction))
      return failure();
    std::optional<int64_t> step =
        stablehloSplatInteger(increment->getOperand(1));
    if (direction.getValue() == stablehlo::ComparisonDirection::LT &&
        step == 1 && *initial == 0 && *bound > 0) {
      description.extent = *bound;
      description.direction = ScanDirection::Forward;
    } else if (direction.getValue() == stablehlo::ComparisonDirection::GE &&
               step == -1 && *bound == 0 && *initial >= 0) {
      description.extent = *initial + 1;
      description.direction = ScanDirection::Reverse;
    } else {
      return failure();
    }
  }

  llvm::SmallDenseSet<Operation *> framingCalls;
  std::optional<ScanDirection> accessDirection;
  Value loopBound = compare.getOperand(1);
  Operation *increment = bodyReturn.getOperand(induction).getDefiningOp();
  Value unitStep = increment && increment->getNumOperands() == 2
                       ? increment->getOperand(1)
                       : Value{};
  auto acceptAccessIndex = [&](Value index) {
    std::optional<ScanDirection> candidate = scanAccessDirection(
        index, body.getArgument(induction), description.extent,
        description.direction, loopBound, unitStep);
    if (!candidate || (accessDirection && *accessDirection != *candidate))
      return false;
    accessDirection = candidate;
    return true;
  };
  for (int64_t index = 0; index < static_cast<int64_t>(body.getNumArguments());
       ++index) {
    if (index == induction)
      continue;
    Value argument = body.getArgument(index);
    Value yielded = bodyReturn.getOperand(index);
    if (yielded == argument) {
      Operation *sliceCall = nullptr;
      std::optional<int64_t> sliceDimension;
      for (Operation *user : argument.getUsers()) {
        if (!calledFunction(user) || user->getNumOperands() != 2 ||
            !acceptAccessIndex(user->getOperand(1)))
          continue;
        std::optional<int64_t> candidate = indexedSliceDimension(
            user, argument, user->getOperand(1));
        if (!candidate)
          continue;
        if (sliceCall)
          return failure();
        sliceCall = user;
        sliceDimension = candidate;
      }
      if (sliceCall) {
        description.scanned.push_back(index);
        description.scannedDimensions.push_back(*sliceDimension);
        description.scannedValues.push_back(sliceCall->getResult(0));
        framingCalls.insert(sliceCall);
      } else {
        description.invariants.push_back(index);
      }
      continue;
    }
    Value update;
    if (Operation *call = yielded.getDefiningOp()) {
      std::optional<int64_t> updateDimension;
      if (call->getNumOperands() == 3 && acceptAccessIndex(call->getOperand(2)))
        updateDimension = indexedUpdateDimension(call, argument,
                                                 call->getOperand(2), &update);
      if (updateDimension) {
        description.emitted.push_back(index);
        description.emittedDimensions.push_back(*updateDimension);
        description.emittedValues.push_back(update);
        framingCalls.insert(call);
        continue;
      }
    }
    description.state.push_back(index);
  }
  if (description.state.empty() || description.scanned.empty())
    return failure();
  if (accessDirection)
    description.direction = *accessDirection;

  llvm::SmallDenseSet<Operation *> roots;
  for (int64_t index : description.state)
    if (Operation *root = bodyReturn.getOperand(index).getDefiningOp())
      roots.insert(root);
  for (Value value : description.emittedValues)
    if (Operation *root = value.getDefiningOp())
      roots.insert(root);
  for (Operation *root : roots) {
    if (framingCalls.contains(root) || !calledFunction(root))
      continue;
    if (description.step && description.step != root)
      return failure();
    description.step = root;
  }
  if (!description.step)
    return failure();
  return description;
}

bool isSupportedStablehloScan(Operation *operation) {
  return succeeded(describeCountedScan(operation));
}

constexpr llvm::StringLiteral kScanHelperAttribute = "shuttle.scan_helper";

bool isScanHelperFunction(func::FuncOp function) {
  return function->hasAttr(kScanHelperAttribute);
}

void markCountedScanHelpers(ModuleOp module) {
  module.walk([&](Operation *operation) {
    FailureOr<CountedScanDescription> description =
        describeCountedScan(operation);
    if (failed(description))
      return;
    operation->setAttr("shuttle.scan_extent",
                       IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                        description->extent));
    operation->setAttr("shuttle.scan_direction",
                       ScanDirectionAttr::get(module.getContext(),
                                              description->direction));
    Block &body = operation->getRegion(1).front();
    for (Operation &nested : body)
      if (func::FuncOp function = calledFunction(&nested))
        function->setAttr(kScanHelperAttribute,
                          UnitAttr::get(module.getContext()));
  });
}

bool isSupportedStablehlo(Operation *operation) {
  if (isSupportedStablehloScan(operation)) {
    return true;
  }
  if (isSupportedStablehloSelection(operation)) {
    return true;
  }
  if (isSupportedStablehloRelation(operation)) {
    return true;
  }
  if (isSupportedStablehloScatter(operation)) {
    return true;
  }
  if (isSupportedStablehloReduce(operation)) {
    return true;
  }
  if (operation->getName().getStringRef() ==
          stablehlo::CompareOp::getOperationName() &&
      operation->getNumOperands() == 2 && operation->getNumResults() == 1 &&
      operation->getNumRegions() == 0 && isMemoryEffectFree(operation)) {
    auto lhs = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
    auto rhs = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
    auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
    auto comparisonType =
        operation->getAttrOfType<stablehlo::ComparisonTypeAttr>("compare_type");
    auto direction =
        operation->getAttrOfType<stablehlo::ComparisonDirectionAttr>(
            "comparison_direction");
    return lhs && rhs && result && lhs == rhs && lhs.hasStaticShape() &&
           result.hasStaticShape() && lhs.getShape() == result.getShape() &&
           lhs.getElementType().isF32() &&
           result.getElementType().isInteger(1) && comparisonType &&
           comparisonType.getValue() == stablehlo::ComparisonType::FLOAT &&
           direction &&
           direction.getValue() == stablehlo::ComparisonDirection::EQ &&
           sourceAttributes(operation).size() == 2;
  }
  if (operation->getNumRegions() != 0 || !isMemoryEffectFree(operation) ||
      !hasOnlyRankedSupportedFloats(operation->getOperands()) ||
      !hasOnlyRankedSupportedFloats(operation->getResults())) {
    return false;
  }

  StringRef name = operation->getName().getStringRef();
  if (name == stablehlo::DotGeneralOp::getOperationName()) {
    if (operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
        operation->hasAttr("algorithm") ||
        !hasSupportedDotPrecision(operation)) {
      return false;
    }
    auto dimensions =
        operation->getAttrOfType<stablehlo::DotDimensionNumbersAttr>(
            "dot_dimension_numbers");
    // Dots without reduction iterators use a separate Map representation.
    // Keep that representation's existing type and precision boundary.
    if (!dimensions || (dimensions.getLhsContractingDimensions().empty() &&
                        (!hasDefaultDotPrecision(operation) ||
                         !elementType(operation->getResult(0)).isF32()))) {
      return false;
    }
    return elementType(operation->getOperand(0)) ==
               elementType(operation->getOperand(1)) &&
           (elementType(operation->getOperand(0)).isF32() ||
            elementType(operation->getOperand(0)).isBF16()) &&
           (elementType(operation->getResult(0)).isF32() ||
            elementType(operation->getResult(0)).isBF16());
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
           (elementType(operation->getOperand(0)).isF32() ||
            elementType(operation->getOperand(0)).isBF16()) &&
           operation->getOperand(0).getType() ==
               operation->getResult(0).getType();
  }
  if (name == stablehlo::AddOp::getOperationName() ||
      name == stablehlo::MulOp::getOperationName() ||
      name == stablehlo::MaxOp::getOperationName() ||
      name == stablehlo::SubtractOp::getOperationName() ||
      name == stablehlo::DivOp::getOperationName()) {
    return operation->getNumOperands() == 2 &&
           operation->getNumResults() == 1 &&
           (elementType(operation->getOperand(0)).isF32() ||
            elementType(operation->getOperand(0)).isBF16()) &&
           operation->getOperand(0).getType() ==
               operation->getOperand(1).getType() &&
           operation->getOperand(0).getType() ==
               operation->getResult(0).getType();
  }
  if (name == stablehlo::TransposeOp::getOperationName()) {
    auto permutation =
        operation->getAttrOfType<DenseI64ArrayAttr>("permutation");
    auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
    auto result = cast<RankedTensorType>(operation->getResult(0).getType());
    return operation->getNumOperands() == 1 &&
           operation->getNumResults() == 1 && permutation &&
           input.getElementType() == result.getElementType() &&
           (input.getElementType().isF32() ||
            input.getElementType().isBF16()) &&
           permutation.size() == static_cast<size_t>(input.getRank());
  }
  if (name == stablehlo::ConstantOp::getOperationName()) {
    auto value = operation->getAttrOfType<DenseElementsAttr>("value");
    auto result = cast<RankedTensorType>(operation->getResult(0).getType());
    return operation->getNumOperands() == 0 &&
           operation->getNumResults() == 1 && result.getRank() == 0 &&
           (result.getElementType().isF32() ||
            (result.getElementType().isBF16() &&
             enclosingFunctionContainsScatter(operation))) &&
           value && value.isSplat();
  }
  if (name == stablehlo::BroadcastInDimOp::getOperationName()) {
    return isSupportedAffineBroadcast(operation);
  }
  if (name == stablehlo::ReshapeOp::getOperationName()) {
    return isCopyFreeStaticReshape(operation);
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

bool isFullPrefixRankNAdditiveReduce(Operation *operation) {
  if (!isSupportedStablehloReduce(operation)) {
    return false;
  }
  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto dimensions = operation->getAttrOfType<DenseI64ArrayAttr>("dimensions");
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
  auto dimensions = operation->getAttrOfType<DenseI64ArrayAttr>("dimensions");
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
         input && result && input.hasStaticShape() && result.hasStaticShape() &&
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
         name == stablehlo::MaxOp::getOperationName() ||
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
         llvm::all_of(
             input.getShape(), [](int64_t extent) { return extent > 0; });
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
    if (isProjectedBf16ParameterConversion(producer, expansionType.getRank())) {
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

#include "StructuralSelection.inc"

bool hasUnclosedRankOneResult(const CandidateComponent &component) {
  llvm::SmallPtrSet<Operation *, 16> componentOperations(
      component.operations.begin(), component.operations.end());
  return llvm::any_of(component.operations, [&](Operation *operation) {
    return llvm::any_of(operation->getResults(), [&](Value result) {
      auto type = dyn_cast<RankedTensorType>(result.getType());
      SmallVector<Operation *> outsideUsers;
      llvm::copy_if(
          result.getUsers(), std::back_inserter(outsideUsers),
          [&](Operation *user) { return !componentOperations.contains(user); });
      if (!type || type.getRank() != 1 || !type.getElementType().isBF16() ||
          outsideUsers.empty()) {
        return false;
      }
      return !isClosedRankOneBf16Result(operation, componentOperations) ||
             !llvm::all_of(outsideUsers, [](Operation *user) {
               return isa<func::ReturnOp>(user);
             });
    });
  });
}

bool isTrailingConversionOfUnsupportedOperation(
    const CandidateComponent &component) {
  if (component.operations.size() != 1 ||
      component.operations.front()->getName().getStringRef() !=
          stablehlo::ConvertOp::getOperationName() ||
      component.operations.front()->getNumOperands() != 1) {
    return false;
  }
  Operation *producer =
      component.operations.front()->getOperand(0).getDefiningOp();
  return producer && !isSupportedStablehlo(producer);
}

SmallVector<CandidateComponent> candidateComponents(ModuleOp module) {
  SmallVector<CandidateComponent> components;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (isCompositeDecompositionFunction(function, module) ||
        isScanHelperFunction(function)) {
      continue;
    }
    bool hasIndexedAccumulation = false;
    function.walk([&](Operation *operation) {
      hasIndexedAccumulation |= isSupportedStablehloScatter(operation);
    });
    for (Block &block : function.getBody()) {
      SmallVector<Operation *> interval;
      auto flushInterval = [&]() {
        if (interval.empty()) {
          return;
        }
        for (CandidateComponent &component :
             partitionSupportedInterval(interval)) {
          if (!isTrailingConversionOfUnsupportedOperation(component) &&
              (hasIndexedAccumulation ||
              (!hasRankNExpansionWithoutClosedAnchor(component) &&
               !hasUnclosedRankOneResult(component)))) {
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

DictionaryAttr zeroResultRecord(Operation *operation, Attribute operationRef,
                                ArrayRef<int64_t> operandIndices) {
  MLIRContext *context = operation->getContext();
  SmallVector<Attribute> operands;
  operands.reserve(operandIndices.size());
  for (int64_t index : operandIndices) {
    if (index < 0 || index >= static_cast<int64_t>(operation->getNumOperands()))
      return {};
    Value operand = operation->getOperand(index);
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
      NamedAttribute(StringAttr::get(context, "operation_ref"), operationRef),
      NamedAttribute(StringAttr::get(context, "operands"),
                     ArrayAttr::get(context, operands)),
  };
  return DictionaryAttr::get(context, fields);
}

DictionaryAttr zeroResultRecord(Operation *operation) {
  SmallVector<int64_t> operandIndices(operation->getNumOperands());
  std::iota(operandIndices.begin(), operandIndices.end(), 0);
  return zeroResultRecord(operation, operation->getAttr(kOperationRefAttribute),
                          operandIndices);
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
    if (isCompositeDecompositionFunction(function, module) ||
        isScanHelperFunction(function)) {
      continue;
    }
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
      if (isCompositeDecompositionFunction(function, getOperation())) {
        continue;
      }
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
    markCountedScanHelpers(module);
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (isCompositeDecompositionFunction(function, module) ||
          isScanHelperFunction(function)) {
        continue;
      }
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
    FailureOr<AffineMap> inputMap =
        contiguousReshapeInputMap(inputType, resultType);
    if (failed(inputMap)) {
      operation->emitOpError("has incompatible contiguous reshape shapes");
      return failure();
    }
    return SmallVector<AffineMap>{*inputMap, identity};
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
  } else if (stablehloName == stablehlo::MaxOp::getOperationName()) {
    scalarName = arith::MaximumFOp::getOperationName();
  } else if (stablehloName == stablehlo::SubtractOp::getOperationName()) {
    scalarName = arith::SubFOp::getOperationName();
  } else if (stablehloName == stablehlo::DivOp::getOperationName()) {
    scalarName = arith::DivFOp::getOperationName();
  } else if (stablehloName == stablehlo::CompareOp::getOperationName()) {
    auto comparisonType =
        operation->getAttrOfType<stablehlo::ComparisonTypeAttr>("compare_type");
    auto direction =
        operation->getAttrOfType<stablehlo::ComparisonDirectionAttr>(
            "comparison_direction");
    if (!comparisonType ||
        comparisonType.getValue() != stablehlo::ComparisonType::FLOAT ||
        !direction ||
        direction.getValue() != stablehlo::ComparisonDirection::EQ ||
        arguments.size() != 2) {
      return nullptr;
    }
    return builder.create<arith::CmpFOp>(location, arith::CmpFPredicate::OEQ,
                                         arguments[0], arguments[1]);
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
  if (domainRank == parallelRank) {
    if (operation->getNumOperands() != 2)
      return failure();
    OpBuilder builder(operation);
    auto resultType = cast<RankedTensorType>(operation->getResult(0).getType());
    OperationState state(operation->getLoc(), MapOp::getOperationName());
    state.addOperands(operation->getOperands());
    state.addTypes(operation->getResultTypes());
    state.addAttribute("indexing_maps",
                       affineMapsAttr(operation->getContext(), *maps));
    state.addAttribute("semantics",
                       MapSemanticsAttr::get(operation->getContext(),
                                             MapSemantics::Pointwise));
    state.addAttribute("source", source);
    state.addRegion();
    auto map = cast<MapOp>(builder.create(state));
    Block *body = new Block();
    map.getBody().push_back(body);
    for (Value operand : operation->getOperands())
      body->addArgument(elementType(operand), operation->getLoc());
    builder.setInsertionPointToEnd(body);
    SmallVector<Value> scalars;
    for (BlockArgument argument : body->getArguments()) {
      Value scalar = argument;
      if (argument.getType() != resultType.getElementType())
        scalar = builder.create<arith::ExtFOp>(
            operation->getLoc(), resultType.getElementType(), argument);
      scalars.push_back(scalar);
    }
    Value product = builder.create<arith::MulFOp>(operation->getLoc(),
                                                  scalars[0], scalars[1]);
    builder.create<YieldOp>(operation->getLoc(), product);
    operation->getResult(0).replaceAllUsesWith(map.getResult(0));
    operation->erase();
    return success();
  }
  OpBuilder builder(operation);
  SmallVector<Attribute> iteratorKinds;
  for (unsigned dimension = 0; dimension < domainRank; ++dimension) {
    iteratorKinds.push_back(builder.getStringAttr(
        dimension < parallelRank ? "parallel" : "reduction"));
  }
  SmallVector<Attribute> precision(operation->getNumOperands(),
                                   builder.getStringAttr("DEFAULT"));
  auto sourcePrecision =
      operation->getAttrOfType<ArrayAttr>("precision_config");
  if (sourcePrecision && !sourcePrecision.empty()) {
    precision.clear();
    for (Attribute attribute : sourcePrecision) {
      precision.push_back(builder.getStringAttr(stablehlo::stringifyPrecision(
          cast<stablehlo::PrecisionAttr>(attribute).getValue())));
    }
  }
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
  if (!sourcePrecision || sourcePrecision.empty()) {
    state.addAttribute(
        "source_precision_form",
        builder.getStringAttr(sourcePrecision ? "empty" : "omitted"));
  }
  state.addAttribute("algorithm", builder.getStringAttr("dot_general"));
  state.addAttribute("source", source);
  Operation *contract = builder.create(state);
  operation->getResult(0).replaceAllUsesWith(contract->getResult(0));
  operation->erase();
  return success();
}

LogicalResult convertRelationOperation(Operation *operation) {
  SourceRefAttr source = singleSourceRef(operation);
  if (!source) {
    return operation->emitOpError("requires exactly one source result");
  }
  FailureOr<RelationSemantics> semantics =
      llvm::StringSwitch<FailureOr<RelationSemantics>>(
          operation->getName().getStringRef())
          .Case(stablehlo::IotaOp::getOperationName(), RelationSemantics::Iota)
          .Case(stablehlo::ConstantOp::getOperationName(),
                RelationSemantics::Constant)
          .Case(stablehlo::ReshapeOp::getOperationName(),
                RelationSemantics::Reshape)
          .Case(stablehlo::BroadcastInDimOp::getOperationName(),
                RelationSemantics::BroadcastInDim)
          .Case(stablehlo::CompareOp::getOperationName(),
                RelationSemantics::Compare)
          .Case(stablehlo::AddOp::getOperationName(), RelationSemantics::Add)
          .Case(stablehlo::ConvertOp::getOperationName(),
                RelationSemantics::Convert)
          .Case(stablehlo::SelectOp::getOperationName(),
                RelationSemantics::Select)
          .Case(stablehlo::SliceOp::getOperationName(),
                RelationSemantics::Slice)
          .Case(stablehlo::TransposeOp::getOperationName(),
                RelationSemantics::Transpose)
          .Case(stablehlo::MulOp::getOperationName(),
                RelationSemantics::Multiply)
          .Case(stablehlo::GatherOp::getOperationName(),
                RelationSemantics::Gather)
          .Default(failure());
  if (failed(semantics) || !isSupportedStablehloRelation(operation)) {
    return operation->emitOpError("has no lossless Relation conversion");
  }
  OpBuilder builder(operation);
  OperationState state(operation->getLoc(), RelationOp::getOperationName());
  state.addOperands(operation->getOperands());
  state.addTypes(operation->getResultTypes());
  state.addAttribute("semantics", RelationSemanticsAttr::get(
                                      operation->getContext(), *semantics));
  SmallVector<Attribute> parameters;
  if (*semantics == RelationSemantics::Iota) {
    parameters.push_back(operation->getAttr("iota_dimension"));
  } else if (*semantics == RelationSemantics::Constant) {
    parameters.push_back(operation->getAttrOfType<DenseElementsAttr>("value")
                             .getSplatValue<Attribute>());
  } else if (*semantics == RelationSemantics::BroadcastInDim) {
    auto dimensions =
        operation->getAttrOfType<DenseI64ArrayAttr>("broadcast_dimensions");
    for (int64_t dimension : dimensions.asArrayRef()) {
      parameters.push_back(builder.getI64IntegerAttr(dimension));
    }
  } else if (*semantics == RelationSemantics::Compare) {
    auto comparisonType =
        operation->getAttrOfType<stablehlo::ComparisonTypeAttr>("compare_type");
    auto comparisonDirection =
        operation->getAttrOfType<stablehlo::ComparisonDirectionAttr>(
            "comparison_direction");
    parameters.push_back(builder.getStringAttr(
        stringifyComparisonType(comparisonType.getValue())));
    parameters.push_back(builder.getStringAttr(
        stringifyComparisonDirection(comparisonDirection.getValue())));
  } else if (*semantics == RelationSemantics::Slice) {
    parameters.push_back(operation->getAttr("start_indices"));
    parameters.push_back(operation->getAttr("limit_indices"));
    parameters.push_back(operation->getAttr("strides"));
  } else if (*semantics == RelationSemantics::Transpose) {
    parameters.push_back(operation->getAttr("permutation"));
  } else if (*semantics == RelationSemantics::Gather) {
    auto dimensions =
        operation->getAttrOfType<stablehlo::GatherDimensionNumbersAttr>(
            "dimension_numbers");
    NamedAttribute fields[] = {
        builder.getNamedAttr(
            "offset_dims", DenseI64ArrayAttr::get(operation->getContext(),
                                                  dimensions.getOffsetDims())),
        builder.getNamedAttr(
            "collapsed_slice_dims",
            DenseI64ArrayAttr::get(operation->getContext(),
                                   dimensions.getCollapsedSliceDims())),
        builder.getNamedAttr(
            "operand_batching_dims",
            DenseI64ArrayAttr::get(operation->getContext(),
                                   dimensions.getOperandBatchingDims())),
        builder.getNamedAttr(
            "start_indices_batching_dims",
            DenseI64ArrayAttr::get(operation->getContext(),
                                   dimensions.getStartIndicesBatchingDims())),
        builder.getNamedAttr(
            "start_index_map",
            DenseI64ArrayAttr::get(operation->getContext(),
                                   dimensions.getStartIndexMap())),
        builder.getNamedAttr(
            "index_vector_dim",
            builder.getI64IntegerAttr(dimensions.getIndexVectorDim())),
        builder.getNamedAttr("slice_sizes", operation->getAttr("slice_sizes")),
        builder.getNamedAttr("indices_are_sorted",
                             operation->getAttr("indices_are_sorted")
                                 ? operation->getAttr("indices_are_sorted")
                                 : builder.getBoolAttr(false)),
    };
    parameters.push_back(DictionaryAttr::get(operation->getContext(), fields));
  }
  state.addAttribute("parameters",
                     ArrayAttr::get(operation->getContext(), parameters));
  state.addAttribute("source", source);
  Operation *relation = builder.create(state);
  operation->getResult(0).replaceAllUsesWith(relation->getResult(0));
  operation->erase();
  return success();
}

DictionaryAttr algebraSourceGroup(OpBuilder &builder, int64_t member,
                                  int64_t size) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("member", builder.getI64IntegerAttr(member)),
       builder.getNamedAttr("size", builder.getI64IntegerAttr(size))});
}

LogicalResult convertScatterOperation(Operation *operation) {
  if (!isSupportedStablehloScatter(operation))
    return operation->emitOpError(
        "requires a static additive full-window indexed update");
  SourceRefAttr source = singleSourceRef(operation);
  if (!source)
    return operation->emitOpError("requires exactly one source result");
  auto operand = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto indices = cast<RankedTensorType>(operation->getOperand(1).getType());
  auto updates = cast<RankedTensorType>(operation->getOperand(2).getType());
  auto dimensions =
      operation->getAttrOfType<stablehlo::ScatterDimensionNumbersAttr>(
          "scatter_dimension_numbers");
  Block &sourceBody = operation->getRegion(0).front();
  Operation &sourceCombiner = sourceBody.front();
  Operation &sourceReturn = sourceBody.back();
  ArrayAttr combinerRefs = sourceRefs(&sourceCombiner);
  auto combinerOwner = sourceCombiner.getAttr(kOperationRefAttribute);
  auto returnOwner = sourceReturn.getAttr(kOperationRefAttribute);
  auto outerOwner = operation->getAttr(kOperationRefAttribute);
  if (!combinerRefs || combinerRefs.size() != 1 || !combinerOwner ||
      !returnOwner || !outerOwner)
    return operation->emitOpError(
        "has incomplete outer/reducer source provenance");

  llvm::SmallDenseSet<int64_t> updateWindowDims(
      dimensions.getUpdateWindowDims().begin(),
      dimensions.getUpdateWindowDims().end());
  SmallVector<int64_t> updateRouteDims;
  for (int64_t dimension = 0; dimension < updates.getRank(); ++dimension)
    if (!updateWindowDims.contains(dimension))
      updateRouteDims.push_back(dimension);
  llvm::SmallDenseSet<int64_t> batchedIndexDims(
      dimensions.getScatterIndicesBatchingDims().begin(),
      dimensions.getScatterIndicesBatchingDims().end());
  SmallVector<int64_t> reductionUpdateDims;
  for (auto [position, updateDimension] : llvm::enumerate(updateRouteDims))
    if (!batchedIndexDims.contains(position))
      reductionUpdateDims.push_back(updateDimension);
  if (reductionUpdateDims.empty())
    return operation->emitOpError(
        "requires at least one non-batched indexed-update dimension");

  SmallVector<int64_t> contributionShape(operand.getShape());
  for (int64_t dimension : reductionUpdateDims)
    contributionShape.push_back(updates.getDimSize(dimension));
  auto contributionType = RankedTensorType::get(
      contributionShape, operand.getElementType(), operand.getEncoding());
  const int64_t contributionRank = contributionType.getRank();
  const int64_t outputRank = operand.getRank();
  SmallVector<int64_t> reductionDimensions;
  for (int64_t dimension = outputRank; dimension < contributionRank;
       ++dimension)
    reductionDimensions.push_back(dimension);

  OpBuilder builder(operation);
  NamedAttribute metadataFields[] = {
      builder.getNamedAttr("output_shape",
                           builder.getDenseI64ArrayAttr(operand.getShape())),
      builder.getNamedAttr(
          "update_window_dims",
          builder.getDenseI64ArrayAttr(dimensions.getUpdateWindowDims())),
      builder.getNamedAttr(
          "inserted_window_dims",
          builder.getDenseI64ArrayAttr(dimensions.getInsertedWindowDims())),
      builder.getNamedAttr(
          "input_batching_dims",
          builder.getDenseI64ArrayAttr(dimensions.getInputBatchingDims())),
      builder.getNamedAttr("scatter_indices_batching_dims",
                           builder.getDenseI64ArrayAttr(
                               dimensions.getScatterIndicesBatchingDims())),
      builder.getNamedAttr("scatter_dims_to_operand_dims",
                           builder.getDenseI64ArrayAttr(
                               dimensions.getScatterDimsToOperandDims())),
      builder.getNamedAttr(
          "index_vector_dim",
          builder.getI64IntegerAttr(dimensions.getIndexVectorDim())),
      builder.getNamedAttr("indices_are_sorted",
                           operation->getAttr("indices_are_sorted")
                               ? operation->getAttr("indices_are_sorted")
                               : builder.getBoolAttr(false)),
      builder.getNamedAttr("unique_indices",
                           operation->getAttr("unique_indices")
                               ? operation->getAttr("unique_indices")
                               : builder.getBoolAttr(false)),
      builder.getNamedAttr("conflict_resolution",
                           builder.getStringAttr("source_ordered_add"))};
  OperationState relationState(operation->getLoc(),
                               RelationOp::getOperationName());
  relationState.addOperands(
      {operation->getOperand(2), operation->getOperand(1)});
  relationState.addTypes(contributionType);
  relationState.addAttribute(
      "semantics",
      RelationSemanticsAttr::get(operation->getContext(),
                                 RelationSemantics::IndexedContributions));
  relationState.addAttribute(
      "parameters",
      builder.getArrayAttr({builder.getDictionaryAttr(metadataFields)}));
  relationState.addAttribute("source", source);
  relationState.addAttribute("source_group", algebraSourceGroup(builder, 0, 4));
  relationState.addAttribute(kOperationRefAttribute, outerOwner);
  Operation *relation = builder.create(relationState);

  auto scalarTensor = RankedTensorType::get({}, operand.getElementType());
  OperationState initializerState(operation->getLoc(),
                                  MapOp::getOperationName());
  initializerState.addTypes(scalarTensor);
  initializerState.addAttribute(
      "indexing_maps",
      affineMapsAttr(operation->getContext(),
                     {AffineMap::get(0, 0, operation->getContext())}));
  initializerState.addAttribute(
      "semantics",
      MapSemanticsAttr::get(operation->getContext(), MapSemantics::Pointwise));
  initializerState.addAttribute("source", source);
  initializerState.addAttribute("source_group",
                                algebraSourceGroup(builder, 1, 4));
  initializerState.addAttribute(kOperationRefAttribute, outerOwner);
  initializerState.addRegion();
  auto initializer = cast<MapOp>(builder.create(initializerState));
  Block *initializerBody = new Block();
  initializer.getBody().push_back(initializerBody);
  builder.setInsertionPointToEnd(initializerBody);
  OperationState zeroState(operation->getLoc(),
                           arith::ConstantOp::getOperationName());
  zeroState.addTypes(operand.getElementType());
  zeroState.addAttribute("value",
                         builder.getFloatAttr(operand.getElementType(), 0.0));
  Operation *zero = builder.create(zeroState);
  OperationState initializerYield(operation->getLoc(),
                                  YieldOp::getOperationName());
  initializerYield.addOperands(zero->getResult(0));
  builder.create(initializerYield);

  builder.setInsertionPoint(operation);
  SmallVector<AffineExpr> resultExpressions;
  for (int64_t dimension = 0; dimension < outputRank; ++dimension)
    resultExpressions.push_back(
        getAffineDimExpr(dimension, operation->getContext()));
  OperationState foldState(operation->getLoc(), FoldOp::getOperationName());
  foldState.addOperands({relation->getResult(0), initializer.getResult(0)});
  foldState.addAttribute("operand_segment_sizes",
                         builder.getDenseI32ArrayAttr({1, 1}));
  foldState.addTypes(operand);
  foldState.addAttribute(
      "indexing_maps",
      affineMapsAttr(operation->getContext(),
                     {AffineMap::getMultiDimIdentityMap(
                          contributionRank, operation->getContext()),
                      AffineMap::get(contributionRank, 0, resultExpressions,
                                     operation->getContext())}));
  foldState.addAttribute("reduction_dimensions",
                         builder.getDenseI64ArrayAttr(reductionDimensions));
  foldState.addAttribute(
      "accumulator_types",
      builder.getArrayAttr({TypeAttr::get(operand.getElementType())}));
  foldState.addAttribute("order_free", builder.getBoolAttr(false));
  foldState.addAttribute("source", source);
  foldState.addAttribute("source_group", algebraSourceGroup(builder, 2, 4));
  foldState.addAttribute(kOperationRefAttribute, outerOwner);
  foldState.addRegion();
  auto fold = cast<FoldOp>(builder.create(foldState));
  Block *combiner = new Block();
  fold.getCombiner().push_back(combiner);
  combiner->addArgument(operand.getElementType(),
                        sourceBody.getArgument(0).getLoc());
  combiner->addArgument(operand.getElementType(),
                        sourceBody.getArgument(1).getLoc());
  builder.setInsertionPointToEnd(combiner);
  OperationState addState(sourceCombiner.getLoc(),
                          arith::AddFOp::getOperationName());
  addState.addOperands(combiner->getArguments());
  addState.addTypes(operand.getElementType());
  addState.addAttribute(kSourceRefsAttribute, combinerRefs);
  addState.addAttribute(kOperationRefAttribute, combinerOwner);
  Operation *add = builder.create(addState);
  OperationState foldYield(sourceReturn.getLoc(), YieldOp::getOperationName());
  foldYield.addOperands(add->getResult(0));
  foldYield.addAttribute(kOperationRefAttribute, returnOwner);
  builder.create(foldYield);

  builder.setInsertionPoint(operation);
  SmallVector<AffineMap> pointwiseMaps(
      3,
      AffineMap::getMultiDimIdentityMap(outputRank, operation->getContext()));
  OperationState finalizerState(operation->getLoc(), MapOp::getOperationName());
  finalizerState.addOperands({operation->getOperand(0), fold.getResult(0)});
  finalizerState.addTypes(operand);
  finalizerState.addAttribute(
      "indexing_maps", affineMapsAttr(operation->getContext(), pointwiseMaps));
  finalizerState.addAttribute(
      "semantics",
      MapSemanticsAttr::get(operation->getContext(), MapSemantics::Pointwise));
  finalizerState.addAttribute("source", source);
  finalizerState.addAttribute("source_group",
                              algebraSourceGroup(builder, 3, 4));
  finalizerState.addAttribute(kOperationRefAttribute, outerOwner);
  finalizerState.addRegion();
  auto finalizer = cast<MapOp>(builder.create(finalizerState));
  Block *finalizerBody = new Block();
  finalizer.getBody().push_back(finalizerBody);
  finalizerBody->addArgument(operand.getElementType(), operation->getLoc());
  finalizerBody->addArgument(operand.getElementType(), operation->getLoc());
  builder.setInsertionPointToEnd(finalizerBody);
  OperationState finalAddState(operation->getLoc(),
                               arith::AddFOp::getOperationName());
  finalAddState.addOperands(finalizerBody->getArguments());
  finalAddState.addTypes(operand.getElementType());
  Operation *finalAdd = builder.create(finalAddState);
  OperationState finalYield(operation->getLoc(), YieldOp::getOperationName());
  finalYield.addOperands(finalAdd->getResult(0));
  builder.create(finalYield);

  operation->getResult(0).replaceAllUsesWith(finalizer.getResult(0));
  operation->erase();
  return success();
}

LogicalResult convertSelectionOperation(Operation *operation) {
  if (!isSupportedStablehloSelection(operation)) {
    return operation->emitOpError("has no lossless Selection conversion");
  }
  auto decomposition =
      operation->getAttrOfType<FlatSymbolRefAttr>("decomposition");
  func::FuncOp implementation =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(operation,
                                                         decomposition);
  ArrayAttr sources = sourceRefs(operation);
  if (!sources || sources.size() != operation->getNumResults()) {
    return operation->emitOpError(
        "requires one source reference per Selection result");
  }
  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto attributes =
      operation->getAttrOfType<DictionaryAttr>("composite_attributes");
  OpBuilder builder(operation);
  OperationState state(operation->getLoc(), SelectionOp::getOperationName());
  state.addOperands(operation->getOperands());
  state.addTypes(operation->getResultTypes());
  state.addAttribute("dimension",
                     builder.getI64IntegerAttr(input.getRank() - 1));
  state.addAttribute("count", attributes.get("k"));
  state.addAttribute("order",
                     SelectionOrderAttr::get(operation->getContext(),
                                             SelectionOrder::Descending));
  state.addAttribute("comparison", SelectionComparisonAttr::get(
                                       operation->getContext(),
                                       SelectionComparison::TotalOrder));
  state.addAttribute("stable", builder.getBoolAttr(true));
  state.addAttribute("tie_break", SelectionTieBreakAttr::get(
                                      operation->getContext(),
                                      SelectionTieBreak::SourceIndexAscending));
  state.addAttribute("sources", sources);
  Operation *selection = builder.create(state);
  for (auto [sourceResult, selectedResult] :
       llvm::zip_equal(operation->getResults(), selection->getResults())) {
    sourceResult.replaceAllUsesWith(selectedResult);
  }
  operation->erase();
  if (SymbolTable::symbolKnownUseEmpty(implementation,
                                       implementation->getParentOp())) {
    implementation.erase();
  }
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
  Operation &sourceCombiner = sourceBody.front();
  Operation &sourceReturn = sourceBody.back();
  ArrayAttr combinerRefs = sourceRefs(&sourceCombiner);
  auto combinerOwner =
      sourceCombiner.getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  auto returnOwner =
      sourceReturn.getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  if (!combinerRefs || combinerRefs.size() != 1 || !combinerOwner ||
      !returnOwner) {
    return operation->emitOpError("has incomplete nested reducer provenance");
  }

  OpBuilder builder(operation);
  auto inputType = cast<RankedTensorType>(operation->getOperand(0).getType());
  llvm::SmallDenseSet<int64_t> reductionDimensions;
  for (int64_t dimension :
       operation->getAttrOfType<DenseI64ArrayAttr>("dimensions").asArrayRef()) {
    reductionDimensions.insert(dimension);
  }
  SmallVector<AffineExpr> resultIndexing;
  for (int64_t dimension = 0; dimension < inputType.getRank(); ++dimension) {
    if (!reductionDimensions.contains(dimension)) {
      resultIndexing.push_back(
          getAffineDimExpr(dimension, operation->getContext()));
    }
  }
  SmallVector<AffineMap> indexingMaps{
      AffineMap::getMultiDimIdentityMap(inputType.getRank(),
                                        operation->getContext()),
      AffineMap::get(inputType.getRank(), 0, resultIndexing,
                     operation->getContext())};
  OperationState state(operation->getLoc(), FoldOp::getOperationName());
  state.addOperands(operation->getOperands());
  state.addTypes(operation->getResultTypes());
  state.addAttribute("operandSegmentSizes",
                     DenseI32ArrayAttr::get(operation->getContext(), {1, 1}));
  state.addAttribute("indexing_maps",
                     affineMapsAttr(operation->getContext(), indexingMaps));
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
  StringRef scalarName = sourceCombiner.getName().getStringRef() ==
                                 stablehlo::AddOp::getOperationName()
                             ? arith::AddFOp::getOperationName()
                             : arith::MaximumFOp::getOperationName();
  OperationState combinerState(sourceCombiner.getLoc(), scalarName);
  combinerState.addOperands(body->getArguments());
  combinerState.addTypes(builder.getF32Type());
  combinerState.addAttribute(kSourceRefsAttribute, combinerRefs);
  combinerState.addAttribute(kOperationRefAttribute, combinerOwner);
  Operation *combiner = builder.create(combinerState);
  OperationState yieldState(sourceReturn.getLoc(), YieldOp::getOperationName());
  yieldState.addOperands(combiner->getResult(0));
  yieldState.addAttribute(kOperationRefAttribute, returnOwner);
  builder.create(yieldState);

  operation->getResult(0).replaceAllUsesWith(fold.getResult(0));
  operation->erase();
  return success();
}

LogicalResult convertScanOperation(Operation *operation) {
  FailureOr<CountedScanDescription> described = describeCountedScan(operation);
  if (failed(described))
    return operation->emitOpError(
        "is not a statically bounded counted Scan with structural slice/update bindings");
  CountedScanDescription &description = *described;
  Block &sourceBody = operation->getRegion(1).front();
  auto sourceReturn = cast<stablehlo::ReturnOp>(sourceBody.getTerminator());
  func::FuncOp step = calledFunction(description.step);
  if (!step || !llvm::hasSingleElement(step.getBody()))
    return operation->emitOpError("requires one closed generic step function");
  Block &stepBody = step.getBody().front();
  auto stepReturn = dyn_cast<func::ReturnOp>(stepBody.getTerminator());
  if (!stepReturn || stepReturn.getNumOperands() !=
                         description.step->getNumResults())
    return operation->emitOpError("requires a direct step-function return");

  SmallVector<Value> inputs;
  SmallVector<Type> resultTypes;
  for (int64_t index : description.state) {
    inputs.push_back(operation->getOperand(index));
    resultTypes.push_back(operation->getResult(index).getType());
  }
  for (int64_t index : description.invariants)
    inputs.push_back(operation->getOperand(index));
  for (int64_t index : description.scanned)
    inputs.push_back(operation->getOperand(index));
  for (int64_t index : description.emitted)
    resultTypes.push_back(operation->getResult(index).getType());

  SmallVector<Attribute> sources;
  if (failed(walkSourcePreorder(operation, [&](Operation *nested) {
        if (ArrayAttr refs = sourceRefs(nested))
          llvm::append_range(sources, refs);
        return success();
      })))
    return failure();
  if (sources.empty())
    return operation->emitOpError("has no complete source provenance");

  SmallVector<Attribute> resultSources;
  ArrayAttr whileSources = sourceRefs(operation);
  if (!whileSources || whileSources.size() != operation->getNumResults())
    return operation->emitOpError("has incomplete result source provenance");
  for (int64_t index : description.state)
    resultSources.push_back(whileSources[index]);
  for (int64_t index : description.emitted)
    resultSources.push_back(whileSources[index]);

  SmallVector<Attribute> erasedTerminatorRecords;
  for (Region &region : operation->getRegions()) {
    auto sourceTerminator = dyn_cast<stablehlo::ReturnOp>(
        region.front().getTerminator());
    if (!sourceTerminator)
      return operation->emitOpError(
          "requires direct StableHLO return terminators");
    DictionaryAttr record = zeroResultRecord(sourceTerminator);
    if (!record)
      return sourceTerminator.emitOpError(
          "has an operand without a structural source anchor");
    erasedTerminatorRecords.push_back(record);
  }

  OpBuilder builder(operation);
  OperationState state(operation->getLoc(), ScanOp::getOperationName());
  state.addOperands(inputs);
  state.addTypes(resultTypes);
  state.addAttribute("axis", builder.getI64IntegerAttr(0));
  state.addAttribute("extent", builder.getI64IntegerAttr(description.extent));
  state.addAttribute("direction",
                     ScanDirectionAttr::get(operation->getContext(),
                                            description.direction));
  SmallVector<int64_t> stateOperands(description.state.size());
  std::iota(stateOperands.begin(), stateOperands.end(), 0);
  SmallVector<int64_t> invariantOperands(description.invariants.size());
  std::iota(invariantOperands.begin(), invariantOperands.end(),
            description.state.size());
  SmallVector<int64_t> scannedOperands(description.scanned.size());
  std::iota(scannedOperands.begin(), scannedOperands.end(),
            description.state.size() + description.invariants.size());
  SmallVector<int64_t> stateResults(description.state.size());
  std::iota(stateResults.begin(), stateResults.end(), 0);
  SmallVector<int64_t> emittedResults(description.emitted.size());
  std::iota(emittedResults.begin(), emittedResults.end(),
            description.state.size());
  state.addAttribute("state_operands",
                     builder.getDenseI64ArrayAttr(stateOperands));
  state.addAttribute("state_body_arguments",
                     builder.getDenseI64ArrayAttr(stateOperands));
  state.addAttribute("state_body_results",
                     builder.getDenseI64ArrayAttr(stateResults));
  state.addAttribute("state_results",
                     builder.getDenseI64ArrayAttr(stateResults));
  state.addAttribute("invariant_operands",
                     builder.getDenseI64ArrayAttr(invariantOperands));
  state.addAttribute("invariant_body_arguments",
                     builder.getDenseI64ArrayAttr(invariantOperands));
  state.addAttribute("scanned_operands",
                     builder.getDenseI64ArrayAttr(scannedOperands));
  state.addAttribute("scanned_body_arguments",
                     builder.getDenseI64ArrayAttr(scannedOperands));
  state.addAttribute("scanned_dimensions", builder.getDenseI64ArrayAttr(
                                                    description.scannedDimensions));
  state.addAttribute("emitted_body_results",
                     builder.getDenseI64ArrayAttr(emittedResults));
  state.addAttribute("emitted_results",
                     builder.getDenseI64ArrayAttr(emittedResults));
  state.addAttribute("emitted_dimensions", builder.getDenseI64ArrayAttr(
                                                    description.emittedDimensions));
  state.addAttribute("sources", builder.getArrayAttr(sources));
  state.addAttribute(kScanResultSourcesAttribute,
                     builder.getArrayAttr(resultSources));
  state.addAttribute(kScanErasedTerminatorRecordsAttribute,
                     builder.getArrayAttr(erasedTerminatorRecords));
  state.addRegion();
  auto scan = cast<ScanOp>(builder.create(state));
  Block *body = new Block();
  scan.getBody().push_back(body);
  for (auto [inputIndex, input] : llvm::enumerate(inputs)) {
    Type type = input.getType();
    const int64_t scannedStart =
        description.state.size() + description.invariants.size();
    if (static_cast<int64_t>(inputIndex) >= scannedStart) {
      // Scanned body types are supplied by their structurally proven slice.
      int64_t position = inputIndex - scannedStart;
      type = description.scannedValues[position].getType();
    }
    body->addArgument(type, input.getLoc());
  }

  IRMapping mapping;
  for (auto [argument, operand] :
       llvm::zip_equal(stepBody.getArguments(), description.step->getOperands())) {
    Value replacement;
    if (auto bodyArgument = dyn_cast<BlockArgument>(operand);
        bodyArgument && bodyArgument.getOwner() == &sourceBody) {
      int64_t sourceIndex = bodyArgument.getArgNumber();
      auto stateIt = llvm::find(description.state, sourceIndex);
      auto invariantIt = llvm::find(description.invariants, sourceIndex);
      if (stateIt != description.state.end())
        replacement = body->getArgument(stateIt - description.state.begin());
      else if (invariantIt != description.invariants.end())
        replacement = body->getArgument(description.state.size() +
                                        invariantIt - description.invariants.begin());
    }
    if (!replacement) {
      auto scannedIt = llvm::find(description.scannedValues, operand);
      if (scannedIt != description.scannedValues.end())
        replacement = body->getArgument(
            description.state.size() + description.invariants.size() +
            scannedIt - description.scannedValues.begin());
    }
    if (!replacement)
      return operation->emitOpError(
          "step operand is not a carried state, invariant, or scanned slice");
    mapping.map(argument, replacement);
  }
  builder.setInsertionPointToEnd(body);
  for (Operation &nested : stepBody.without_terminator())
    builder.clone(nested, mapping);
  for (auto [result, returned] :
       llvm::zip_equal(description.step->getResults(), stepReturn.getOperands()))
    mapping.map(result, mapping.lookupOrDefault(returned));

  SmallVector<Value> yielded;
  for (int64_t index : description.state)
    yielded.push_back(mapping.lookupOrDefault(sourceReturn.getOperand(index)));
  for (Value emitted : description.emittedValues)
    yielded.push_back(mapping.lookupOrDefault(emitted));
  if (llvm::any_of(yielded, [](Value value) { return !value; }))
    return operation->emitOpError("cannot map a Scan result through its step function");
  OperationState yieldState(operation->getLoc(), YieldOp::getOperationName());
  yieldState.addOperands(yielded);
  builder.create(yieldState);

  SmallVector<Operation *> cloned;
  for (Operation &nested : body->without_terminator())
    cloned.push_back(&nested);
  for (Operation *nested : cloned) {
    StringRef name = nested->getName().getStringRef();
    LogicalResult converted =
        isSupportedStablehloSelection(nested) ? convertSelectionOperation(nested)
        : isSupportedStablehloScatter(nested) ? convertScatterOperation(nested)
        : isSupportedStablehloRelation(nested) ? convertRelationOperation(nested)
        : name == stablehlo::ReduceOp::getOperationName()
            ? convertReduceOperation(nested)
        : name == stablehlo::DotGeneralOp::getOperationName()
            ? convertContractOperation(nested)
            : convertMapOperation(nested);
    if (failed(converted))
      return operation->emitOpError("has an unsupported generic Scan body operation");
  }

  llvm::SmallDenseMap<int64_t, Value> replacementByPosition;
  for (auto [ordinal, position] : llvm::enumerate(description.state))
    replacementByPosition[position] = scan.getResult(ordinal);
  for (auto [ordinal, position] : llvm::enumerate(description.emitted))
    replacementByPosition[position] =
        scan.getResult(description.state.size() + ordinal);
  for (int64_t position : description.invariants)
    replacementByPosition[position] = operation->getOperand(position);
  for (int64_t position : description.scanned)
    replacementByPosition[position] = operation->getOperand(position);
  for (auto [position, result] : llvm::enumerate(operation->getResults())) {
    if (position == static_cast<size_t>(description.induction)) {
      if (!result.use_empty())
        return operation->emitOpError("cannot expose the counted induction result");
      continue;
    }
    auto replacement = replacementByPosition.find(position);
    if (replacement == replacementByPosition.end())
      return operation->emitOpError("has an unclassified loop-carried result");
    result.replaceAllUsesWith(replacement->second);
  }
  operation->erase();

  // Counted-loop induction and destination-buffer initializers are structural
  // StableHLO scaffolding, not Scan inputs. Preserve their source coverage on
  // the aggregate Scan while removing any now-dead pure producer chain.
  bool removed = true;
  while (removed) {
    removed = false;
    SmallVector<Operation *> candidates;
    for (Operation &candidate : scan->getBlock()->without_terminator())
      candidates.push_back(&candidate);
    for (Operation *candidate : llvm::reverse(candidates)) {
      if (candidate == scan.getOperation() ||
          llvm::any_of(candidate->getResults(),
                       [](Value value) { return !value.use_empty(); }) ||
          !isMemoryEffectFree(candidate))
        continue;
      SmallVector<Attribute> aggregate(scan.getSources().begin(),
                                       scan.getSources().end());
      if (ArrayAttr refs = sourceRefs(candidate))
        llvm::append_range(aggregate, refs);
      scan->setAttr("sources", builder.getArrayAttr(aggregate));
      candidate->erase();
      removed = true;
    }
  }
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
      SmallVector<Operation *> scans;
      for (Operation &operation : region.getBody().front()) {
        if (isSupportedStablehloScan(&operation))
          scans.push_back(&operation);
      }
      for (Operation *scan : scans) {
        if (failed(convertScanOperation(scan))) {
          signalPassFailure();
          return;
        }
      }
      SmallVector<Operation *> sourceOperations;
      for (Operation &operation : region.getBody().front())
        if (!isa<YieldOp>(operation) &&
            operation.getDialect() !=
                region->getContext()->getLoadedDialect<ShuttleDialect>())
          sourceOperations.push_back(&operation);
      for (Operation *operation : sourceOperations) {
        StringRef name = operation->getName().getStringRef();
        LogicalResult result =
            name == stablehlo::WhileOp::getOperationName()
                ? operation->emitOpError(
                      "is not a statically bounded counted Scan with structural slice/update bindings")
            : isSupportedStablehloSelection(operation)
                ? convertSelectionOperation(operation)
            : isSupportedStablehloScatter(operation)
                ? convertScatterOperation(operation)
            : isSupportedStablehloRelation(operation)
                ? convertRelationOperation(operation)
            : name == stablehlo::ReduceOp::getOperationName()
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
    for (func::FuncOp function :
         llvm::make_early_inc_range(getOperation().getOps<func::FuncOp>()))
      if (isScanHelperFunction(function) &&
          SymbolTable::symbolKnownUseEmpty(function, getOperation()))
        function.erase();
  }
};

LogicalResult verifyRegionLocalCoverage(ModuleOp module) {
  WalkResult result = module.walk([&](RegionOp region) {
    llvm::SmallDenseSet<Attribute> declaredSources;
    llvm::SmallDenseSet<Attribute> representedSources;
    llvm::SmallDenseMap<Attribute, int64_t> sourceGroupSizes;
    llvm::SmallDenseMap<Attribute, llvm::SmallDenseSet<int64_t, 4>>
        sourceGroupMembers;
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
      if (operation->getParentOfType<ScanOp>() && !isa<ScanOp>(operation)) {
        return WalkResult::advance();
      }
      if (auto scan = dyn_cast<ScanOp>(operation)) {
        for (Attribute source : scan.getSources()) {
          if (!declaredSources.contains(source) ||
              !representedSources.insert(source).second) {
            scan.emitOpError("has invalid ordered Scan source coverage");
            return WalkResult::interrupt();
          }
        }
        return WalkResult::skip();
      }
      if (auto selection = dyn_cast<SelectionOp>(operation)) {
        for (Attribute source : selection.getSources()) {
          if (!declaredSources.contains(source) ||
              !representedSources.insert(source).second) {
            selection.emitOpError(
                "has invalid multi-result Selection source coverage");
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      }
      Attribute source;
      if (auto map = dyn_cast<MapOp>(operation)) {
        source = map.getSource();
      } else if (auto relation = dyn_cast<RelationOp>(operation)) {
        source = relation.getSource();
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
      auto sourceGroup =
          operation->getAttrOfType<DictionaryAttr>("source_group");
      if (sourceGroup) {
        auto member = sourceGroup.getAs<IntegerAttr>("member");
        auto size = sourceGroup.getAs<IntegerAttr>("size");
        if (sourceGroup.size() != 2 || !member || !size ||
            member.getInt() < 0 || size.getInt() <= 1 ||
            member.getInt() >= size.getInt()) {
          operation->emitOpError("has invalid algebra source-group metadata");
          return WalkResult::interrupt();
        }
        auto [sizeIt, insertedSize] =
            sourceGroupSizes.try_emplace(source, size.getInt());
        if ((!insertedSize && sizeIt->second != size.getInt()) ||
            !sourceGroupMembers[source].insert(member.getInt()).second) {
          operation->emitOpError(
              "has duplicate or inconsistent algebra source-group members");
          return WalkResult::interrupt();
        }
        representedSources.insert(source);
      } else if (!representedSources.insert(source).second) {
        operation->emitOpError("duplicates region-local source coverage");
        return WalkResult::interrupt();
      }
      if (auto fold = dyn_cast<FoldOp>(operation)) {
        if (ArrayAttr aggregates = sourceRefs(operation)) {
          for (Attribute aggregate : aggregates) {
            if (aggregate == source)
              continue;
            if (!declaredSources.contains(aggregate) ||
                !representedSources.insert(aggregate).second) {
              operation->emitOpError(
                  "has invalid aggregate Fold source coverage");
              return WalkResult::interrupt();
            }
          }
        }
      }
      return WalkResult::skip();
    });
    if (nestedResult.wasInterrupted()) {
      return WalkResult::interrupt();
    }
    for (const auto &[source, size] : sourceGroupSizes)
      if (sourceGroupMembers.lookup(source).size() !=
          static_cast<size_t>(size)) {
        region.emitOpError("has an incomplete algebra source group");
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
  if (reasons.empty())
    return ArrayAttr::get(module.getContext(), {});

  SmallVector<Attribute> records;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (isCompositeDecompositionFunction(function, module)) {
      continue;
    }
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
    if (isCompositeDecompositionFunction(function, module)) {
      continue;
    }
    LogicalResult walk =
        walkSourcePreorder(function.getBody(), [&](Operation *operation) {
          if (auto scan = dyn_cast<ScanOp>(operation)) {
            if (auto erased = scan.getErasedTerminatorRecordsAttr())
              llvm::append_range(records, erased);
          }
          if (operation->getParentOfType<ScanOp>())
            return success();
          if (operation->getNumResults() != 0) {
            return success();
          }
          if (auto groups = operation->getAttrOfType<ArrayAttr>(
                  kOperationRefGroupsAttribute)) {
            for (Attribute attribute : groups) {
              auto group = dyn_cast<DictionaryAttr>(attribute);
              auto operationRef =
                  group ? group.getAs<DenseI64ArrayAttr>("operation_ref")
                        : DenseI64ArrayAttr{};
              auto operandIndices =
                  group ? group.getAs<DenseI64ArrayAttr>("operand_indices")
                        : DenseI64ArrayAttr{};
              DictionaryAttr record =
                  operationRef && operandIndices
                      ? zeroResultRecord(operation, operationRef,
                                         operandIndices.asArrayRef())
                      : DictionaryAttr{};
              if (!record)
                return failure();
              records.push_back(record);
            }
            return success();
          }
          if (!operation->hasAttr(kOperationRefAttribute))
            return success();
          DictionaryAttr record = zeroResultRecord(operation);
          if (!record)
            return failure();
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
    if (isCompositeDecompositionFunction(function, module)) {
      continue;
    }
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
  std::string sm90GpuPrefix =
      R"json({"execution_mode":"gpu_executable_bundle","gpu_target":"sm_90","numerics":")json";
  sm90GpuPrefix.append(policy.getValue().data(), policy.getValue().size());
  sm90GpuPrefix.push_back('"');
  std::string sm100GpuPrefix =
      R"json({"execution_mode":"gpu_executable_bundle","gpu_target":"sm_100","numerics":")json";
  sm100GpuPrefix.append(policy.getValue().data(), policy.getValue().size());
  sm100GpuPrefix.push_back('"');
  const bool targetedGpuOptions =
      (canonicalOptions.getValue().starts_with(sm90GpuPrefix) ||
       canonicalOptions.getValue().starts_with(sm100GpuPrefix)) &&
      canonicalOptions.getValue().contains(
          "\"pipeline_abi_version\":23,\"schema_version\":1");
  const bool validOptionsPrefix =
      canonicalOptions.getValue().starts_with(roundTripPrefix) ||
      canonicalOptions.getValue().starts_with(gpuPrefix) || targetedGpuOptions;
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
  llvm::SmallDenseMap<Attribute, int64_t> expandedOperationSizes;
  llvm::SmallDenseMap<Attribute, llvm::SmallDenseSet<int64_t, 4>>
      expandedOperationMembers;
  WalkResult operationRefWalk = module.walk([&](Operation *operation) {
    Attribute operationRef = operation->getAttr(kOperationRefAttribute);
    ArrayAttr operationRefGroups =
        operation->getAttrOfType<ArrayAttr>(kOperationRefGroupsAttribute);
    ArrayAttr refs = sourceRefs(operation);
    const bool aggregateProvenance =
        isa<FoldOp>(operation) || isa<stablehlo::ReduceOp>(operation);
    if (refs && operation->getParentOfType<FoldOp>() && !operationRef &&
        !aggregateProvenance) {
      operation->emitOpError(
          "has nested Fold sources without an operation reference");
      return WalkResult::interrupt();
    }
    if (!operationRef) {
      if (!operationRefGroups)
        return WalkResult::advance();
      if (operation->getNumResults() != 0 ||
          !operation->hasTrait<OpTrait::IsTerminator>()) {
        operation->emitOpError(
            "operation reference groups require a zero-result terminator");
        return WalkResult::interrupt();
      }
      for (Attribute attribute : operationRefGroups) {
        auto group = dyn_cast<DictionaryAttr>(attribute);
        auto groupRef = group ? group.getAs<DenseI64ArrayAttr>("operation_ref")
                              : DenseI64ArrayAttr{};
        auto operandIndices =
            group ? group.getAs<DenseI64ArrayAttr>("operand_indices")
                  : DenseI64ArrayAttr{};
        if (!groupRef || !operandIndices || operandIndices.empty() ||
            groupRef.size() != 3 ||
            llvm::any_of(operandIndices.asArrayRef(),
                         [&](int64_t index) {
                           return index < 0 ||
                                  index >= static_cast<int64_t>(
                                               operation->getNumOperands());
                         }) ||
            !operationRefs.insert(groupRef).second) {
          operation->emitOpError(
              "has a malformed or duplicate operation reference group");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    }

    if (operationRefGroups) {
      operation->emitOpError(
          "cannot carry singular and grouped operation references");
      return WalkResult::interrupt();
    }
    auto denseOperationRef = dyn_cast<DenseI64ArrayAttr>(operationRef);
    auto sourceGroup = operation->getAttrOfType<DictionaryAttr>("source_group");
    if (!denseOperationRef || denseOperationRef.size() != 3) {
      operation->emitOpError("has a missing-format operation reference");
      return WalkResult::interrupt();
    }
    if (sourceGroup) {
      auto member = sourceGroup.getAs<IntegerAttr>("member");
      auto size = sourceGroup.getAs<IntegerAttr>("size");
      auto [sizeIt, insertedSize] = expandedOperationSizes.try_emplace(
          operationRef, size ? size.getInt() : 0);
      if (!member || !size || sourceGroup.size() != 2 || member.getInt() < 0 ||
          size.getInt() <= 1 || member.getInt() >= size.getInt() ||
          (!insertedSize && sizeIt->second != size.getInt()) ||
          !expandedOperationMembers[operationRef]
               .insert(member.getInt())
               .second) {
        operation->emitOpError(
            "has malformed or duplicate expanded operation provenance");
        return WalkResult::interrupt();
      }
      operationRefs.insert(operationRef);
    } else if (!operationRefs.insert(operationRef).second) {
      operation->emitOpError("has a duplicate operation reference");
      return WalkResult::interrupt();
    }
    if (!aggregateProvenance && refs && llvm::any_of(refs, [&](Attribute ref) {
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
  for (const auto &[operationRef, size] : expandedOperationSizes)
    if (expandedOperationMembers.lookup(operationRef).size() !=
        static_cast<size_t>(size))
      return module.emitError("has incomplete expanded operation provenance");

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
  llvm::SmallDenseMap<Attribute, int64_t> selectedGroupSizes;
  llvm::SmallDenseMap<Attribute, llvm::SmallDenseSet<int64_t>>
      selectedGroupMembers;
  auto recordSelected = [&](Operation *operation,
                            Attribute source) -> WalkResult {
    if (!selected->contains(source)) {
      operation->emitOpError(
          "represents a source absent from selected manifest coverage");
      return WalkResult::interrupt();
    }
    auto sourceGroup = operation->getAttrOfType<DictionaryAttr>("source_group");
    if (sourceGroup) {
      auto member = sourceGroup.getAs<IntegerAttr>("member");
      auto size = sourceGroup.getAs<IntegerAttr>("size");
      auto [sizeIt, insertedSize] =
          selectedGroupSizes.try_emplace(source, size ? size.getInt() : 0);
      if (!member || !size || sourceGroup.size() != 2 || member.getInt() < 0 ||
          size.getInt() <= 1 || member.getInt() >= size.getInt() ||
          (!insertedSize && sizeIt->second != size.getInt()) ||
          !selectedGroupMembers[source].insert(member.getInt()).second) {
        operation->emitOpError(
            "has malformed or duplicate selected source-group coverage");
        return WalkResult::interrupt();
      }
      representedSelected.insert(source);
      return WalkResult::advance();
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
    if (operation->getParentOfType<ScanOp>() && !isa<ScanOp>(operation)) {
      return WalkResult::advance();
    }
    if (auto scan = dyn_cast<ScanOp>(operation)) {
      for (Attribute source : scan.getSources()) {
        if (recordSelected(operation, source).wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::skip();
    }
    if (auto selection = dyn_cast<SelectionOp>(operation)) {
      for (Attribute source : selection.getSources()) {
        if (recordSelected(operation, source).wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    }
    if (auto map = dyn_cast<MapOp>(operation)) {
      return recordSelected(operation, map.getSource());
    }
    if (auto relation = dyn_cast<RelationOp>(operation)) {
      return recordSelected(operation, relation.getSource());
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
      WalkResult result = recordSelected(operation, fold.getSource());
      if (result.wasInterrupted()) {
        return result;
      }
      ArrayAttr aggregate = sourceRefs(operation);
      if (aggregate) {
        for (Attribute source : aggregate) {
          if (source != fold.getSource() &&
              recordSelected(operation, source).wasInterrupted()) {
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
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
  for (const auto &[source, size] : selectedGroupSizes)
    if (selectedGroupMembers.lookup(source).size() != static_cast<size_t>(size))
      return module.emitError("has incomplete selected source-group coverage");
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

std::optional<std::pair<unsigned, unsigned>>
canonicalBinaryMapArguments(MapOp map, StringRef operationName) {
  if (map.getSemantics() != MapSemantics::Pointwise ||
      map.getInputs().size() != 2 || map.getNumResults() != 1 ||
      map.getBody().empty()) {
    return std::nullopt;
  }
  Block &body = map.getBody().front();
  Operation *operation = body.empty() ? nullptr : &body.front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!operation || operation->getName().getStringRef() != operationName ||
      operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
      body.getNumArguments() != 2 || body.getOperations().size() != 2 ||
      !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != operation->getResult(0) ||
      operation->getOperand(0) != body.getArgument(0) ||
      operation->getOperand(1) != body.getArgument(1)) {
    return std::nullopt;
  }
  return std::pair(body.getArgument(0).getArgNumber(),
                   body.getArgument(1).getArgNumber());
}

std::optional<unsigned> canonicalUnaryMapArgument(MapOp map,
                                                  StringRef operationName) {
  if (map.getSemantics() != MapSemantics::Pointwise ||
      map.getInputs().size() != 1 || map.getNumResults() != 1 ||
      map.getBody().empty()) {
    return std::nullopt;
  }
  Block &body = map.getBody().front();
  Operation *operation = body.empty() ? nullptr : &body.front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!operation || operation->getName().getStringRef() != operationName ||
      operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
      body.getNumArguments() != 1 || body.getOperations().size() != 2 ||
      !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != operation->getResult(0) ||
      operation->getOperand(0) != body.getArgument(0)) {
    return std::nullopt;
  }
  return body.getArgument(0).getArgNumber();
}

bool canonicalValueDependsOn(Value value, Operation *ancestor) {
  SmallVector<Value> worklist{value};
  llvm::SmallDenseSet<Operation *> visited;
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (producer == ancestor)
      return true;
    llvm::append_range(worklist, producer->getOperands());
  }
  return false;
}

bool canonicalFoldWithScalarCombiner(FoldOp fold, StringRef operationName) {
  if (fold.getInputs().size() != 1 || fold.getInitializers().size() != 1 ||
      fold.getNumResults() != 1 || fold.getCombiner().empty() ||
      !fold.getOrderFree() || fold.getAccumulatorTypes().size() != 1 ||
      cast<TypeAttr>(fold.getAccumulatorTypes()[0]).getValue() !=
          Float32Type::get(fold.getContext())) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(fold.getInputs().front().getType());
  auto initializer =
      dyn_cast<RankedTensorType>(fold.getInitializers().front().getType());
  auto result = dyn_cast<RankedTensorType>(fold.getResult(0).getType());
  if (!input || !initializer || !result || !input.hasStaticShape() ||
      initializer.getRank() != 0 || !initializer.getElementType().isF32() ||
      !input.getElementType().isF32() || !result.getElementType().isF32()) {
    return false;
  }
  Block &body = fold.getCombiner().front();
  Operation *operation = body.empty() ? nullptr : &body.front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!operation || operation->getName().getStringRef() != operationName ||
      operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
      body.getNumArguments() != 2 || body.getOperations().size() != 2 ||
      !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != operation->getResult(0) ||
      operation->getOperand(0) != body.getArgument(0) ||
      operation->getOperand(1) != body.getArgument(1)) {
    return false;
  }
  auto fastMath =
      operation->getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
  return !fastMath || fastMath.getValue() == arith::FastMathFlags::none;
}

void appendCanonicalSource(SmallVectorImpl<Attribute> &sources,
                           llvm::SmallDenseSet<Attribute> &seen,
                           Attribute source) {
  if (source && seen.insert(source).second)
    sources.push_back(source);
}

void appendCanonicalOperationSources(Operation *operation,
                                     SmallVectorImpl<Attribute> &sources,
                                     llvm::SmallDenseSet<Attribute> &seen) {
  if (auto map = dyn_cast<MapOp>(operation))
    appendCanonicalSource(sources, seen, map.getSource());
  else if (auto fold = dyn_cast<FoldOp>(operation))
    appendCanonicalSource(sources, seen, fold.getSource());
  if (ArrayAttr refs =
          operation->getAttrOfType<ArrayAttr>(kSourceRefsAttribute))
    for (Attribute ref : refs)
      appendCanonicalSource(sources, seen, ref);
}

FoldOp canonicalFoldAncestor(Value value) {
  llvm::SmallDenseSet<Operation *> visited;
  while (Operation *producer = value.getDefiningOp()) {
    if (!visited.insert(producer).second)
      return {};
    if (auto fold = dyn_cast<FoldOp>(producer))
      return fold;
    auto map = dyn_cast<MapOp>(producer);
    if (!map || map.getInputs().size() != 1 || map.getNumResults() != 1 ||
        map->getParentOfType<RegionOp>() !=
            producer->getParentOfType<RegionOp>())
      return {};
    value = map.getInputs().front();
  }
  return {};
}

bool canonicalMapFeeds(MapOp map, Operation *first, Operation *second) {
  if (map.getInputs().size() != 1)
    return false;
  return llvm::all_of(map.getResult(0).getUsers(), [&](Operation *user) {
    return user == first || user == second;
  });
}

bool canonicalMapChainCanErase(ArrayRef<MapOp> maps, Operation *first,
                               Operation *second) {
  llvm::SmallPtrSet<Operation *, 8> chain;
  chain.insert(first);
  chain.insert(second);
  for (MapOp map : maps)
    chain.insert(map);
  return llvm::all_of(maps, [&](MapOp map) {
    return llvm::all_of(map.getResult(0).getUsers(),
                        [&](Operation *user) { return chain.contains(user); });
  });
}

void copyCanonicalProvenance(Operation *source, Operation *replacement) {
  if (ArrayAttr refs = sourceRefs(source))
    replacement->setAttr(kSourceRefsAttribute, refs);
  if (Attribute owner = source->getAttr(kOperationRefAttribute))
    replacement->setAttr(kOperationRefAttribute, owner);
}

FailureOr<AffineMap> canonicalCoordinatesToAncestor(Value value,
                                                    Value ancestor) {
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  if (!valueType)
    return failure();
  if (value == ancestor)
    return AffineMap::getMultiDimIdentityMap(valueType.getRank(),
                                             value.getContext());
  auto map = value.getDefiningOp<MapOp>();
  if (!map || map.getInputs().size() != 1 || map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != 2)
    return failure();
  auto inputMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[0]);
  auto resultMap = dyn_cast<AffineMapAttr>(
      map.getIndexingMaps()[map.getIndexingMaps().size() - 1]);
  if (!inputMap || !resultMap || !resultMap.getValue().isIdentity())
    return failure();
  FailureOr<AffineMap> upstream =
      canonicalCoordinatesToAncestor(map.getInputs().front(), ancestor);
  if (failed(upstream) ||
      upstream->getNumDims() != inputMap.getValue().getNumResults())
    return failure();
  return upstream->compose(inputMap.getValue());
}

void collectCanonicalMapChain(Value value, Value ancestor,
                              SmallVectorImpl<MapOp> &maps) {
  if (value == ancestor)
    return;
  auto map = value.getDefiningOp<MapOp>();
  if (!map)
    return;
  collectCanonicalMapChain(map.getInputs().front(), ancestor, maps);
  maps.push_back(map);
}

bool rewriteNormalizedExponentialSlice(MapOp normalization) {
  RegionOp region = normalization->getParentOfType<RegionOp>();
  if (!region || normalization->getParentOp() != region.getBody().getParentOp())
    return false;
  auto division = canonicalBinaryMapArguments(
      normalization, arith::DivFOp::getOperationName());
  if (!division)
    return false;

  Value numerator = normalization.getInputs()[division->first];
  Value denominator = normalization.getInputs()[division->second];
  MapOp exponential = numerator.getDefiningOp<MapOp>();
  if (!exponential)
    return false;
  auto exponentialArgument =
      canonicalUnaryMapArgument(exponential, math::ExpOp::getOperationName());
  if (!exponentialArgument)
    return false;

  MapOp centered =
      exponential.getInputs()[*exponentialArgument].getDefiningOp<MapOp>();
  if (!centered || centered.getSemantics() != MapSemantics::Pointwise ||
      centered.getInputs().size() != 2 || centered.getNumResults() != 1 ||
      centered.getBody().empty() || !centered.getResult(0).hasOneUse() ||
      centered.getResult(0).getUses().begin()->getOwner() != exponential)
    return false;
  auto subtraction =
      canonicalBinaryMapArguments(centered, arith::SubFOp::getOperationName());
  if (!subtraction)
    return false;

  FoldOp maximum =
      canonicalFoldAncestor(centered.getInputs()[subtraction->second]);
  FoldOp additive = canonicalFoldAncestor(denominator);
  if (!maximum || !additive || maximum == additive ||
      !canonicalFoldWithScalarCombiner(maximum,
                                       arith::MaximumFOp::getOperationName()) ||
      !canonicalFoldWithScalarCombiner(additive,
                                       arith::AddFOp::getOperationName()) ||
      maximum.getInputs().front() != centered.getInputs()[subtraction->first] ||
      additive.getInputs().front() != exponential.getResult(0) ||
      maximum.getReductionDimensions() != additive.getReductionDimensions() ||
      maximum.getResult(0).getType() != additive.getResult(0).getType() ||
      !canonicalValueDependsOn(centered.getInputs()[subtraction->second],
                               maximum) ||
      !canonicalValueDependsOn(denominator, additive))
    return false;

  SmallVector<MapOp> maximumExpansion;
  SmallVector<MapOp> additiveExpansion;
  collectCanonicalMapChain(centered.getInputs()[subtraction->second],
                           maximum.getResult(0), maximumExpansion);
  collectCanonicalMapChain(denominator, additive.getResult(0),
                           additiveExpansion);
  FailureOr<AffineMap> maximumCoordinates = canonicalCoordinatesToAncestor(
      centered.getInputs()[subtraction->second], maximum.getResult(0));
  FailureOr<AffineMap> additiveCoordinates =
      canonicalCoordinatesToAncestor(denominator, additive.getResult(0));
  if (failed(maximumCoordinates) || failed(additiveCoordinates) ||
      maximumExpansion.empty() || additiveExpansion.empty())
    return false;

  auto maxInputMap = dyn_cast<AffineMapAttr>(maximum.getIndexingMaps()[0]);
  auto maxResultMap = dyn_cast<AffineMapAttr>(maximum.getIndexingMaps()[1]);
  auto addInputMap = dyn_cast<AffineMapAttr>(additive.getIndexingMaps()[0]);
  auto addResultMap = dyn_cast<AffineMapAttr>(additive.getIndexingMaps()[1]);
  auto centeredScoreMap =
      dyn_cast<AffineMapAttr>(centered.getIndexingMaps()[subtraction->first]);
  auto centeredMaximumMap =
      dyn_cast<AffineMapAttr>(centered.getIndexingMaps()[subtraction->second]);
  auto denominatorMap = dyn_cast<AffineMapAttr>(
      normalization.getIndexingMaps()[division->second]);
  auto resultMap = dyn_cast<AffineMapAttr>(
      normalization
          .getIndexingMaps()[normalization.getIndexingMaps().size() - 1]);
  if (!maxInputMap || !maxResultMap || !addInputMap || !addResultMap ||
      !centeredScoreMap || !centeredMaximumMap || !denominatorMap || !resultMap)
    return false;
  if (*maximumCoordinates != maxResultMap.getValue() ||
      *additiveCoordinates != addResultMap.getValue() ||
      maxInputMap.getValue() != addInputMap.getValue() ||
      maxResultMap.getValue() != addResultMap.getValue() ||
      centeredScoreMap.getValue() != maxInputMap.getValue() ||
      centeredMaximumMap.getValue() !=
          AffineMap::getMultiDimIdentityMap(
              cast<RankedTensorType>(centered.getResult(0).getType()).getRank(),
              normalization.getContext()) ||
      denominatorMap.getValue() !=
          AffineMap::getMultiDimIdentityMap(
              cast<RankedTensorType>(normalization.getResult(0).getType())
                  .getRank(),
              normalization.getContext()) ||
      resultMap.getValue() != maxInputMap.getValue())
    return false;

  SmallVector<Attribute> mergedSources;
  llvm::SmallDenseSet<Attribute> seenSources;
  const bool preservesExponential =
      llvm::any_of(exponential.getResult(0).getUsers(), [&](Operation *user) {
        return user != additive && user != normalization;
      });
  const bool preservesCentered = preservesExponential;
  appendCanonicalSource(mergedSources, seenSources, maximum.getSource());
  appendCanonicalSource(mergedSources, seenSources, maximum.getSource());
  if (!preservesCentered)
    appendCanonicalOperationSources(centered, mergedSources, seenSources);
  if (!preservesExponential)
    appendCanonicalOperationSources(additive, mergedSources, seenSources);
  // The normalization source stays on the replacement Map, so it must not be
  // moved to the Fold provenance aliases.

  auto additiveInitializer =
      additive.getInitializers().front().getDefiningOp<MapOp>();
  if (!additiveInitializer || !additiveInitializer.getInputs().empty() ||
      !additiveInitializer.getResult(0).hasOneUse() ||
      *additiveInitializer.getResult(0).getUsers().begin() != additive)
    return false;
  additiveInitializer->moveBefore(maximum);
  OpBuilder builder(maximum);
  auto scoreType =
      cast<RankedTensorType>(maximum.getInputs().front().getType());
  OperationState onesState(exponential.getLoc(), MapOp::getOperationName());
  onesState.addTypes(scoreType);
  onesState.addAttribute(
      "indexing_maps",
      affineMapsAttr(maximum.getContext(), {maxInputMap.getValue()}));
  onesState.addAttribute(
      "semantics",
      MapSemanticsAttr::get(maximum.getContext(), MapSemantics::Pointwise));
  onesState.addAttribute("source", preservesExponential
                                       ? additive.getSource()
                                       : exponential.getSource());
  onesState.addRegion();
  MapOp ones = cast<MapOp>(builder.create(onesState));
  Block *onesBody = new Block();
  ones.getBody().push_back(onesBody);
  OpBuilder onesBuilder = OpBuilder::atBlockEnd(onesBody);
  Value one = onesBuilder.create<arith::ConstantOp>(
      exponential.getLoc(),
      FloatAttr::get(Float32Type::get(maximum.getContext()), 1.0));
  onesBuilder.create<YieldOp>(exponential.getLoc(), one);

  OperationState foldState(maximum.getLoc(), FoldOp::getOperationName());
  foldState.addOperands({maximum.getInputs().front(), ones.getResult(0),
                         maximum.getInitializers().front(),
                         additive.getInitializers().front()});
  foldState.addTypes(
      {maximum.getResult(0).getType(), additive.getResult(0).getType()});
  foldState.addAttribute("operandSegmentSizes",
                         DenseI32ArrayAttr::get(maximum.getContext(), {2, 2}));
  foldState.addAttribute(
      "indexing_maps",
      affineMapsAttr(maximum.getContext(),
                     {maxInputMap.getValue(), addInputMap.getValue(),
                      maxResultMap.getValue(), addResultMap.getValue()}));
  foldState.addAttribute("reduction_dimensions",
                         maximum.getReductionDimensionsAttr());
  foldState.addAttribute(
      "accumulator_types",
      ArrayAttr::get(maximum.getContext(),
                     {TypeAttr::get(Float32Type::get(maximum.getContext())),
                      TypeAttr::get(Float32Type::get(maximum.getContext()))}));
  foldState.addAttribute("order_free", builder.getBoolAttr(true));
  foldState.addAttribute("source", maximum.getSource());
  if (Attribute owner = maximum->getAttr(kOperationRefAttribute))
    foldState.addAttribute(kOperationRefAttribute, owner);
  foldState.addRegion();
  FoldOp mergedFold = cast<FoldOp>(builder.create(foldState));
  if (!mergedSources.empty())
    mergedFold->setAttr(kSourceRefsAttribute,
                        ArrayAttr::get(maximum.getContext(), mergedSources));
  Block *foldBody = new Block();
  mergedFold.getCombiner().push_back(foldBody);
  for (unsigned i = 0; i < 4; ++i)
    foldBody->addArgument(Float32Type::get(maximum.getContext()),
                          maximum.getLoc());
  OpBuilder foldBuilder = OpBuilder::atBlockEnd(foldBody);
  auto maximumCombiner = &maximum.getCombiner().front().front();
  auto additiveCombiner = &additive.getCombiner().front().front();
  auto nextMaximum = foldBuilder.create<arith::MaximumFOp>(
      maximum.getLoc(), foldBody->getArgument(0), foldBody->getArgument(2));
  copyCanonicalProvenance(maximumCombiner, nextMaximum);
  Value lhsDelta = foldBuilder.create<arith::SubFOp>(
      maximum.getLoc(), foldBody->getArgument(0), nextMaximum);
  Value lhsScale = foldBuilder.create<math::ExpOp>(maximum.getLoc(), lhsDelta);
  Value lhsMass = foldBuilder.create<arith::MulFOp>(maximum.getLoc(), lhsScale,
                                                    foldBody->getArgument(1));
  Value rhsDelta = foldBuilder.create<arith::SubFOp>(
      maximum.getLoc(), foldBody->getArgument(2), nextMaximum);
  Value rhsScale = foldBuilder.create<math::ExpOp>(maximum.getLoc(), rhsDelta);
  Value rhsMass = foldBuilder.create<arith::MulFOp>(maximum.getLoc(), rhsScale,
                                                    foldBody->getArgument(3));
  auto nextMass =
      foldBuilder.create<arith::AddFOp>(maximum.getLoc(), lhsMass, rhsMass);
  copyCanonicalProvenance(additiveCombiner, nextMass);
  OperationState yieldState(maximum.getLoc(), YieldOp::getOperationName());
  yieldState.addOperands({nextMaximum, nextMass});
  SmallVector<Attribute> operationRefGroups;
  auto addOperationRefGroup = [&](YieldOp sourceYield, int64_t operandIndex) {
    auto operationRef =
        sourceYield->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
    if (!operationRef)
      return false;
    NamedAttribute fields[] = {
        NamedAttribute(builder.getStringAttr("operation_ref"), operationRef),
        NamedAttribute(
            builder.getStringAttr("operand_indices"),
            DenseI64ArrayAttr::get(maximum.getContext(), {operandIndex})),
    };
    operationRefGroups.push_back(
        DictionaryAttr::get(maximum.getContext(), fields));
    return true;
  };
  if (!addOperationRefGroup(
          cast<YieldOp>(maximum.getCombiner().front().getTerminator()), 0) ||
      !addOperationRefGroup(
          cast<YieldOp>(additive.getCombiner().front().getTerminator()), 1)) {
    mergedFold.erase();
    ones.erase();
    return false;
  }
  yieldState.addAttribute(
      kOperationRefGroupsAttribute,
      ArrayAttr::get(maximum.getContext(), operationRefGroups));
  foldBuilder.create(yieldState);

  maximum.getResult(0).replaceAllUsesWith(mergedFold.getResult(0));
  additive.getResult(0).replaceAllUsesWith(mergedFold.getResult(1));

  builder.setInsertionPoint(normalization);
  OperationState mapState(normalization.getLoc(), MapOp::getOperationName());
  mapState.addOperands({maximum.getInputs().front(),
                        centered.getInputs()[subtraction->second],
                        denominator});
  mapState.addTypes(normalization.getResultTypes());
  mapState.addAttribute(
      "indexing_maps",
      affineMapsAttr(normalization.getContext(),
                     {centeredScoreMap.getValue(),
                      centeredMaximumMap.getValue(), denominatorMap.getValue(),
                      resultMap.getValue()}));
  mapState.addAttribute("semantics",
                        MapSemanticsAttr::get(normalization.getContext(),
                                              MapSemantics::Pointwise));
  mapState.addAttribute("source", normalization.getSource());
  mapState.addRegion();
  MapOp normalized = cast<MapOp>(builder.create(mapState));
  Block *mapBody = new Block();
  normalized.getBody().push_back(mapBody);
  mapBody->addArgument(Float32Type::get(normalization.getContext()),
                       normalization.getLoc());
  mapBody->addArgument(Float32Type::get(normalization.getContext()),
                       normalization.getLoc());
  mapBody->addArgument(Float32Type::get(normalization.getContext()),
                       normalization.getLoc());
  OpBuilder mapBuilder = OpBuilder::atBlockEnd(mapBody);
  Value delta = mapBuilder.create<arith::SubFOp>(
      normalization.getLoc(), mapBody->getArgument(0), mapBody->getArgument(1));
  Value exponentialValue =
      mapBuilder.create<math::ExpOp>(normalization.getLoc(), delta);
  Value normalizedValue = mapBuilder.create<arith::DivFOp>(
      normalization.getLoc(), exponentialValue, mapBody->getArgument(2));
  mapBuilder.create<YieldOp>(normalization.getLoc(), normalizedValue);

  normalization.getResult(0).replaceAllUsesWith(normalized.getResult(0));
  normalization.erase();
  additive.erase();
  maximum.erase();
  if (exponential.getResult(0).use_empty())
    exponential.erase();
  if (centered.getResult(0).use_empty())
    centered.erase();
  return true;
}

struct ShuttleCanonicalizePass
    : impl::ShuttleCanonicalizePassBase<ShuttleCanonicalizePass> {
  void runOnOperation() override {
    SmallVector<RegionOp> regions;
    getOperation().walk([&](RegionOp region) { regions.push_back(region); });
    for (RegionOp region : regions) {
      if (region.getPolicy() != NumericalPolicy::Fast)
        continue;
      // A rewrite can expose another independent local slice in the same
      // region.  Re-scan after every successful replacement; the canonical
      // two-result Fold no longer satisfies the source pattern, so this
      // terminates without requiring a graph/workload recognizer.
      bool changed = true;
      while (changed) {
        changed = false;
        for (MapOp map : llvm::make_early_inc_range(
                 region.getBody().front().getOps<MapOp>())) {
          if (rewriteNormalizedExponentialSlice(map)) {
            changed = true;
            break;
          }
        }
      }
    }
  }

private:
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
    StringRef name = attribute.getName().strref();
    if (name != "fastmath" && name != kSourceRefsAttribute &&
        name != kOperationRefAttribute) {
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
      const bool supportedRankTwoExpansion = inputType.getRank() == 2 &&
                                             resultType.getRank() == 2 &&
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
      FailureOr<AffineMap> expected = failure();
      if (inputType && resultType)
        expected = contiguousReshapeInputMap(inputType, resultType);
      if (failed(expected) || inputMap != *expected) {
        map.emitOpError("has an invalid contiguous reshape indexing map");
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
    } else if (scalarName == arith::MaximumFOp::getOperationName() &&
               operands.size() == 2) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::MaxOp::getOperationName();
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
    } else if (auto compare = dyn_cast<arith::CmpFOp>(scalar);
               compare && operands.size() == 2 &&
               compare.getPredicate() == arith::CmpFPredicate::OEQ) {
      auto fastMath =
          compare->getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
      if ((fastMath && fastMath.getValue() != arith::FastMathFlags::none) ||
          llvm::any_of(compare->getAttrs(), [](NamedAttribute attribute) {
            StringRef name = attribute.getName().strref();
            return name != "predicate" && name != "fastmath";
          })) {
        compare.emitOpError(
            "has scalar attributes with no StableHLO representation");
        return failure();
      }
      stablehloName = stablehlo::CompareOp::getOperationName();
      attributes.append("compare_type", stablehlo::ComparisonTypeAttr::get(
                                            map.getContext(),
                                            stablehlo::ComparisonType::FLOAT));
      attributes.append(
          "comparison_direction",
          stablehlo::ComparisonDirectionAttr::get(
              map.getContext(), stablehlo::ComparisonDirection::EQ));
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
      if (!value || !resultType || !resultType.hasStaticShape() ||
          value.getType() != resultType.getElementType()) {
        constant.emitOpError("has no static splat StableHLO constant lowering");
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

FailureOr<Operation *> lowerPointwiseMapDag(OpBuilder &builder, MapOp map,
                                            ValueRange operands) {
  if (map.getSemantics() != MapSemantics::Pointwise ||
      map.getNumResults() != 1 || map.getBody().empty() || operands.empty() ||
      !map->getDiscardableAttrs().empty())
    return failure();
  auto resultType = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  Block &body = map.getBody().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!resultType || !resultType.hasStaticShape() || !yield ||
      yield.getValues().size() != 1 || !yield->getDiscardableAttrs().empty() ||
      map.getIndexingMaps().size() != operands.size() + 1 ||
      !llvm::all_of(map.getIndexingMaps(), [](Attribute attribute) {
        return isIdentityMap(cast<AffineMapAttr>(attribute).getValue());
      }))
    return failure();
  for (Value operand : operands)
    if (operand.getType() != resultType)
      return failure();

  OperationState state(map.getLoc(), stablehlo::MapOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(map.getResultTypes());
  SmallVector<int64_t> dimensions(resultType.getRank());
  std::iota(dimensions.begin(), dimensions.end(), 0);
  state.addAttribute("dimensions",
                     DenseI64ArrayAttr::get(map.getContext(), dimensions));
  state.addAttribute(kSourceRefsAttribute,
                     ArrayAttr::get(map.getContext(), {map.getSource()}));
  state.addRegion();
  Operation *loweredMap = builder.create(state);
  Block *loweredBody = new Block();
  loweredMap->getRegion(0).push_back(loweredBody);
  for (BlockArgument argument : body.getArguments())
    loweredBody->addArgument(RankedTensorType::get({}, argument.getType()),
                             argument.getLoc());

  OpBuilder nestedBuilder = OpBuilder::atBlockEnd(loweredBody);
  IRMapping mapping;
  for (auto [source, target] :
       llvm::zip_equal(body.getArguments(), loweredBody->getArguments()))
    mapping.map(source, target);

  auto stablehloName = [](Operation *operation) -> StringRef {
    return llvm::StringSwitch<StringRef>(operation->getName().getStringRef())
        .Case(arith::AddFOp::getOperationName(),
              stablehlo::AddOp::getOperationName())
        .Case(arith::SubFOp::getOperationName(),
              stablehlo::SubtractOp::getOperationName())
        .Case(arith::MulFOp::getOperationName(),
              stablehlo::MulOp::getOperationName())
        .Case(arith::DivFOp::getOperationName(),
              stablehlo::DivOp::getOperationName())
        .Case(arith::MaximumFOp::getOperationName(),
              stablehlo::MaxOp::getOperationName())
        .Case(arith::NegFOp::getOperationName(),
              stablehlo::NegOp::getOperationName())
        .Case(math::ExpOp::getOperationName(),
              stablehlo::ExpOp::getOperationName())
        .Case(math::TanhOp::getOperationName(),
              stablehlo::TanhOp::getOperationName())
        .Case(math::RsqrtOp::getOperationName(),
              stablehlo::RsqrtOp::getOperationName())
        .Case(arith::SelectOp::getOperationName(),
              stablehlo::SelectOp::getOperationName())
        .Case(ScalarConvertOp::getOperationName(),
              stablehlo::ConvertOp::getOperationName())
        .Case(arith::ConstantOp::getOperationName(),
              stablehlo::ConstantOp::getOperationName())
        .Default(StringRef{});
  };
  for (Operation &operation : body.without_terminator()) {
    StringRef name = stablehloName(&operation);
    const bool scalarConvert = isa<ScalarConvertOp>(operation);
    if (name.empty() || operation.getNumResults() != 1 ||
        (!scalarConvert && !isa<arith::ConstantOp>(operation) &&
         failed(verifyDefaultScalarSemantics(&operation)))) {
      operation.emitOpError("has no authoritative StableHLO Map DAG lowering");
      loweredMap->erase();
      return failure();
    }
    SmallVector<Value> mappedOperands;
    for (Value operand : operation.getOperands()) {
      Value mapped = mapping.lookupOrNull(operand);
      if (!mapped) {
        operation.emitOpError("has an unmapped scalar DAG operand");
        loweredMap->erase();
        return failure();
      }
      mappedOperands.push_back(mapped);
    }
    Type scalarType = operation.getResult(0).getType();
    OperationState scalarState(operation.getLoc(), name);
    scalarState.addOperands(mappedOperands);
    scalarState.addTypes(RankedTensorType::get({}, scalarType));
    if (auto refs = sourceRefs(&operation))
      scalarState.addAttribute(kSourceRefsAttribute, refs);
    if (Attribute owner = operation.getAttr(kOperationRefAttribute))
      scalarState.addAttribute(kOperationRefAttribute, owner);
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      auto value = dyn_cast<FloatAttr>(constant.getValue());
      if (!value) {
        constant.emitOpError("requires a floating scalar constant");
        loweredMap->erase();
        return failure();
      }
      scalarState.addAttribute(
          "value",
          DenseElementsAttr::get(RankedTensorType::get({}, scalarType), value));
    }
    Operation *loweredScalar = nestedBuilder.create(scalarState);
    mapping.map(operation.getResult(0), loweredScalar->getResult(0));
  }
  Value loweredResult = mapping.lookupOrNull(yield.getValues().front());
  if (!loweredResult) {
    yield.emitOpError("has an unmapped scalar DAG result");
    loweredMap->erase();
    return failure();
  }
  OperationState returnState(yield.getLoc(),
                             stablehlo::ReturnOp::getOperationName());
  returnState.addOperands(loweredResult);
  nestedBuilder.create(returnState);
  return loweredMap;
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

FailureOr<OperationState> lowerRelationState(RelationOp relation,
                                             ValueRange operands) {
  if (llvm::any_of(
          relation->getDiscardableAttrs(), [](NamedAttribute attribute) {
            return attribute.getName().strref() != kOperationRefAttribute;
          })) {
    relation.emitOpError("has an unsupported discardable Relation attribute");
    return failure();
  }
  StringRef stablehloName =
      llvm::StringSwitch<StringRef>(
          stringifyRelationSemantics(relation.getSemantics()))
          .Case("iota", stablehlo::IotaOp::getOperationName())
          .Case("constant", stablehlo::ConstantOp::getOperationName())
          .Case("reshape", stablehlo::ReshapeOp::getOperationName())
          .Case("broadcast_in_dim",
                stablehlo::BroadcastInDimOp::getOperationName())
          .Case("compare", stablehlo::CompareOp::getOperationName())
          .Case("add", stablehlo::AddOp::getOperationName())
          .Case("convert", stablehlo::ConvertOp::getOperationName())
          .Case("select", stablehlo::SelectOp::getOperationName())
          .Case("slice", stablehlo::SliceOp::getOperationName())
          .Case("transpose", stablehlo::TransposeOp::getOperationName())
          .Case("multiply", stablehlo::MulOp::getOperationName())
          .Case("gather", stablehlo::GatherOp::getOperationName())
          .Default({});
  if (stablehloName.empty()) {
    relation.emitOpError("has no authoritative StableHLO lowering");
    return failure();
  }
  OperationState state(relation.getLoc(), stablehloName);
  state.addOperands(operands);
  state.addTypes(relation.getResultTypes());
  if (relation.getSemantics() == RelationSemantics::Iota) {
    state.addAttribute("iota_dimension", relation.getParameters()[0]);
  } else if (relation.getSemantics() == RelationSemantics::Constant) {
    auto result = cast<RankedTensorType>(relation.getResult(0).getType());
    state.addAttribute(
        "value", DenseElementsAttr::get(
                     result, ArrayRef<Attribute>{relation.getParameters()[0]}));
  } else if (relation.getSemantics() == RelationSemantics::BroadcastInDim) {
    SmallVector<int64_t> dimensions;
    for (Attribute parameter : relation.getParameters()) {
      dimensions.push_back(cast<IntegerAttr>(parameter).getInt());
    }
    state.addAttribute(
        "broadcast_dimensions",
        DenseI64ArrayAttr::get(relation.getContext(), dimensions));
  } else if (relation.getSemantics() == RelationSemantics::Compare) {
    auto type = stablehlo::symbolizeComparisonType(
        cast<StringAttr>(relation.getParameters()[0]).getValue());
    auto direction = stablehlo::symbolizeComparisonDirection(
        cast<StringAttr>(relation.getParameters()[1]).getValue());
    if (!type || !direction) {
      relation.emitOpError("has invalid comparison parameters");
      return failure();
    }
    state.addAttribute("compare_type", stablehlo::ComparisonTypeAttr::get(
                                           relation.getContext(), *type));
    state.addAttribute("comparison_direction",
                       stablehlo::ComparisonDirectionAttr::get(
                           relation.getContext(), *direction));
  } else if (relation.getSemantics() == RelationSemantics::Slice) {
    state.addAttribute("start_indices", relation.getParameters()[0]);
    state.addAttribute("limit_indices", relation.getParameters()[1]);
    state.addAttribute("strides", relation.getParameters()[2]);
  } else if (relation.getSemantics() == RelationSemantics::Transpose) {
    state.addAttribute("permutation", relation.getParameters()[0]);
  } else if (relation.getSemantics() == RelationSemantics::Gather) {
    auto metadata = cast<DictionaryAttr>(relation.getParameters()[0]);
    auto dimensions = stablehlo::GatherDimensionNumbersAttr::get(
        relation.getContext(),
        metadata.getAs<DenseI64ArrayAttr>("offset_dims").asArrayRef(),
        metadata.getAs<DenseI64ArrayAttr>("collapsed_slice_dims").asArrayRef(),
        metadata.getAs<DenseI64ArrayAttr>("operand_batching_dims").asArrayRef(),
        metadata.getAs<DenseI64ArrayAttr>("start_indices_batching_dims")
            .asArrayRef(),
        metadata.getAs<DenseI64ArrayAttr>("start_index_map").asArrayRef(),
        metadata.getAs<IntegerAttr>("index_vector_dim").getInt());
    state.addAttribute("dimension_numbers", dimensions);
    state.addAttribute("slice_sizes", metadata.get("slice_sizes"));
    state.addAttribute("indices_are_sorted",
                       metadata.get("indices_are_sorted"));
  }
  return state;
}

FailureOr<SmallVector<Value>>
lowerSelection(OpBuilder &builder, SelectionOp selection, ValueRange operands) {
  auto input = dyn_cast<RankedTensorType>(selection.getInput().getType());
  if (operands.size() != 1 || !input || !selection.getStable() ||
      selection.getComparison() != SelectionComparison::TotalOrder ||
      selection.getTieBreak() != SelectionTieBreak::SourceIndexAscending ||
      !selection->getDiscardableAttrs().empty()) {
    selection.emitOpError("has no authoritative StableHLO Selection lowering");
    return failure();
  }
  const int64_t dimension = selection.getDimension();
  auto fullIndexType =
      RankedTensorType::get(input.getShape(), builder.getI32Type());
  OperationState iotaState(selection.getLoc(),
                           stablehlo::IotaOp::getOperationName());
  iotaState.addTypes(fullIndexType);
  iotaState.addAttribute("iota_dimension",
                         builder.getI64IntegerAttr(dimension));
  Operation *iota = builder.create(iotaState);

  OperationState sortState(selection.getLoc(),
                           stablehlo::SortOp::getOperationName());
  sortState.addOperands({operands[0], iota->getResult(0)});
  sortState.addTypes({input, fullIndexType});
  sortState.addAttribute("dimension", builder.getI64IntegerAttr(dimension));
  sortState.addAttribute("is_stable", builder.getBoolAttr(true));
  sortState.addRegion();
  Operation *sort = builder.create(sortState);
  Region &comparator = sort->getRegion(0);
  Block *comparatorBlock = new Block();
  comparator.push_back(comparatorBlock);
  auto scalarValueType = RankedTensorType::get({}, input.getElementType());
  auto scalarIndexType = RankedTensorType::get({}, builder.getI32Type());
  comparatorBlock->addArgument(scalarValueType, selection.getLoc());
  comparatorBlock->addArgument(scalarValueType, selection.getLoc());
  comparatorBlock->addArgument(scalarIndexType, selection.getLoc());
  comparatorBlock->addArgument(scalarIndexType, selection.getLoc());
  OpBuilder comparatorBuilder = OpBuilder::atBlockEnd(comparatorBlock);
  OperationState compareState(selection.getLoc(),
                              stablehlo::CompareOp::getOperationName());
  compareState.addOperands(
      {comparatorBlock->getArgument(0), comparatorBlock->getArgument(1)});
  compareState.addTypes(
      RankedTensorType::get({}, comparatorBuilder.getI1Type()));
  stablehlo::ComparisonDirection direction =
      selection.getOrder() == SelectionOrder::Descending
          ? stablehlo::ComparisonDirection::GT
          : stablehlo::ComparisonDirection::LT;
  compareState.addAttribute("comparison_direction",
                            stablehlo::ComparisonDirectionAttr::get(
                                selection.getContext(), direction));
  compareState.addAttribute(
      "compare_type",
      stablehlo::ComparisonTypeAttr::get(
          selection.getContext(), stablehlo::ComparisonType::TOTALORDER));
  Operation *compare = comparatorBuilder.create(compareState);
  OperationState returnState(selection.getLoc(),
                             stablehlo::ReturnOp::getOperationName());
  returnState.addOperands(compare->getResult(0));
  comparatorBuilder.create(returnState);

  SmallVector<int64_t> starts(input.getRank(), 0);
  SmallVector<int64_t> limits(input.getShape());
  SmallVector<int64_t> strides(input.getRank(), 1);
  limits[dimension] = selection.getCount();
  SmallVector<Value> results;
  for (auto [resultNumber, resultType] :
       llvm::enumerate(selection.getResultTypes())) {
    OperationState sliceState(selection.getLoc(),
                              stablehlo::SliceOp::getOperationName());
    sliceState.addOperands(sort->getResult(resultNumber));
    sliceState.addTypes(resultType);
    sliceState.addAttribute(
        "start_indices",
        DenseI64ArrayAttr::get(selection.getContext(), starts));
    sliceState.addAttribute(
        "limit_indices",
        DenseI64ArrayAttr::get(selection.getContext(), limits));
    sliceState.addAttribute(
        "strides", DenseI64ArrayAttr::get(selection.getContext(), strides));
    sliceState.addAttribute(
        kSourceRefsAttribute,
        ArrayAttr::get(selection.getContext(),
                       {selection.getSources()[resultNumber]}));
    results.push_back(builder.create(sliceState)->getResult(0));
  }
  return results;
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
  auto sourcePrecisionForm =
      contract->getAttrOfType<StringAttr>("source_precision_form");
  if (!sourcePrecisionForm || sourcePrecisionForm.getValue() != "omitted") {
    if (sourcePrecisionForm && sourcePrecisionForm.getValue() == "empty")
      precision.clear();
    attributes.append("precision_config",
                      ArrayAttr::get(contract.getContext(), precision));
  }
  OperationState state(contract.getLoc(),
                       stablehlo::DotGeneralOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(contract.getResultTypes());
  state.addAttributes(attributes);
  return state;
}

bool hasCanonicalStablehloFoldIndexing(FoldOp fold) {
  const size_t count = fold.getInputs().size();
  if (count == 0 || fold.getNumResults() != count ||
      fold.getIndexingMaps().size() != 2 * count) {
    return false;
  }
  llvm::SmallDenseSet<int64_t> reductionDimensions;
  for (int64_t dimension : fold.getReductionDimensions()) {
    if (dimension < 0)
      return false;
    reductionDimensions.insert(dimension);
  }
  AffineMap first = cast<AffineMapAttr>(fold.getIndexingMaps()[0]).getValue();
  for (size_t index = 0; index < count; ++index) {
    auto input = dyn_cast<RankedTensorType>(fold.getInputs()[index].getType());
    auto inputMap = dyn_cast<AffineMapAttr>(fold.getIndexingMaps()[index]);
    if (!input || !inputMap ||
        inputMap.getValue() != AffineMap::getMultiDimIdentityMap(
                                   input.getRank(), fold.getContext()) ||
        inputMap.getValue().getNumDims() != first.getNumDims()) {
      return false;
    }
  }
  for (size_t index = 0; index < count; ++index) {
    auto result =
        dyn_cast<RankedTensorType>(fold.getResults()[index].getType());
    auto resultMap =
        dyn_cast<AffineMapAttr>(fold.getIndexingMaps()[count + index]);
    if (!result || !resultMap ||
        resultMap.getValue().getNumDims() != first.getNumDims()) {
      return false;
    }
    SmallVector<AffineExpr> expectedResults;
    for (int64_t dimension = 0; dimension < first.getNumDims(); ++dimension)
      if (!reductionDimensions.contains(dimension))
        expectedResults.push_back(
            getAffineDimExpr(dimension, fold.getContext()));
    if (resultMap.getValue() != AffineMap::get(first.getNumDims(), 0,
                                               expectedResults,
                                               fold.getContext())) {
      return false;
    }
    if (resultMap.getValue().getNumResults() != result.getRank())
      return false;
  }
  return true;
}

FailureOr<Operation *> lowerFold(OpBuilder &builder, FoldOp fold,
                                 ValueRange operands) {
  if (llvm::any_of(fold->getDiscardableAttrs(), [](NamedAttribute attribute) {
        StringRef name = attribute.getName().strref();
        return name != kOperationRefAttribute && name != kSourceRefsAttribute;
      })) {
    fold.emitOpError("has an unsupported discardable Fold attribute");
    return failure();
  }
  const size_t count = fold.getInputs().size();
  if (count == 0 || operands.size() != 2 * count ||
      fold.getInitializers().size() != count || fold.getNumResults() != count ||
      fold.getCombiner().empty()) {
    fold.emitOpError("is outside the first Fold lowering contract");
    return failure();
  }
  if (!hasCanonicalStablehloFoldIndexing(fold)) {
    fold.emitOpError(
        "requires canonical StableHLO-compatible Fold indexing maps");
    return failure();
  }
  Block &body = fold.getCombiner().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || body.getNumArguments() != 2 * count ||
      yield.getValues().size() != count) {
    fold.emitOpError("requires two scalar arguments and one yield per result");
    return failure();
  }
  if (!fold.getOrderFree() ||
      llvm::any_of(body.getArguments(),
                   [](Value value) { return !value.getType().isF32(); })) {
    fold.emitOpError("requires an order-free scalar f32 combiner");
    return failure();
  }
  auto foldOwner =
      fold->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  auto yieldOwner =
      yield->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  auto yieldOwnerGroups =
      yield->getAttrOfType<ArrayAttr>(kOperationRefGroupsAttribute);
  if (!foldOwner || (!yieldOwner && !yieldOwnerGroups) ||
      (yieldOwner && yieldOwnerGroups)) {
    fold.emitOpError("has incomplete Fold source provenance");
    return failure();
  }

  OperationState state(fold.getLoc(), stablehlo::ReduceOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(fold.getResultTypes());
  state.addAttribute("dimensions", fold.getReductionDimensionsAttr());
  ArrayAttr foldSources = sourceRefs(fold);
  state.addAttribute(
      kSourceRefsAttribute,
      foldSources ? foldSources
                  : ArrayAttr::get(fold.getContext(), {fold.getSource()}));
  state.addAttribute(kOperationRefAttribute, foldOwner);
  state.addRegion();
  Operation *reduce = builder.create(state);
  Block *reducer = new Block();
  reduce->getRegion(0).push_back(reducer);
  auto scalarTensor = RankedTensorType::get({}, builder.getF32Type());
  for (size_t index = 0; index < 2 * count; ++index)
    reducer->addArgument(scalarTensor, body.getArgument(index).getLoc());
  OpBuilder reducerBuilder = OpBuilder::atBlockEnd(reducer);
  IRMapping mapping;
  for (auto [source, target] :
       llvm::zip_equal(body.getArguments(), reducer->getArguments()))
    mapping.map(source, target);
  auto scalarName = [](Operation *operation) -> StringRef {
    return llvm::StringSwitch<StringRef>(operation->getName().getStringRef())
        .Case(arith::AddFOp::getOperationName(),
              stablehlo::AddOp::getOperationName())
        .Case(arith::SubFOp::getOperationName(),
              stablehlo::SubtractOp::getOperationName())
        .Case(arith::MulFOp::getOperationName(),
              stablehlo::MulOp::getOperationName())
        .Case(arith::MaximumFOp::getOperationName(),
              stablehlo::MaxOp::getOperationName())
        .Case(math::ExpOp::getOperationName(),
              stablehlo::ExpOp::getOperationName())
        .Case(arith::ConstantOp::getOperationName(),
              stablehlo::ConstantOp::getOperationName())
        .Default(StringRef{});
  };
  for (Operation &operation : body.without_terminator()) {
    StringRef name = scalarName(&operation);
    if (name.empty() || operation.getNumResults() != 1 ||
        failed(verifyDefaultScalarSemantics(&operation))) {
      operation.emitOpError("has no authoritative StableHLO Fold lowering");
      return failure();
    }
    OperationState combinerState(operation.getLoc(), name);
    SmallVector<Value> mappedOperands;
    for (Value operand : operation.getOperands()) {
      Value mapped = mapping.lookupOrNull(operand);
      if (!mapped) {
        operation.emitOpError("has an unmapped Fold combiner operand");
        return failure();
      }
      mappedOperands.push_back(mapped);
    }
    combinerState.addOperands(mappedOperands);
    combinerState.addTypes(scalarTensor);
    if (auto refs = sourceRefs(&operation))
      combinerState.addAttribute(kSourceRefsAttribute, refs);
    if (auto owner = operation.getAttr(kOperationRefAttribute))
      combinerState.addAttribute(kOperationRefAttribute, owner);
    if (isa<arith::ConstantOp>(operation)) {
      auto value = operation.getAttrOfType<FloatAttr>("value");
      if (!value) {
        operation.emitOpError("requires a scalar floating constant");
        return failure();
      }
      combinerState.addAttribute("value",
                                 DenseElementsAttr::get(scalarTensor, value));
    }
    Operation *lowered = reducerBuilder.create(combinerState);
    mapping.map(operation.getResult(0), lowered->getResult(0));
  }
  SmallVector<Value> loweredResults;
  for (Value value : yield.getValues()) {
    Value mapped = mapping.lookupOrNull(value);
    if (!mapped) {
      yield.emitOpError("has an unmapped Fold result");
      return failure();
    }
    loweredResults.push_back(mapped);
  }
  OperationState returnState(yield.getLoc(),
                             stablehlo::ReturnOp::getOperationName());
  returnState.addOperands(loweredResults);
  if (yieldOwner)
    returnState.addAttribute(kOperationRefAttribute, yieldOwner);
  else
    returnState.addAttribute(kOperationRefGroupsAttribute, yieldOwnerGroups);
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
    if (auto selection = dyn_cast<SelectionOp>(operation)) {
      FailureOr<SmallVector<Value>> lowered =
          lowerSelection(builder, selection, operands);
      if (failed(lowered)) {
        return failure();
      }
      for (auto [result, loweredResult] :
           llvm::zip_equal(operation.getResults(), *lowered)) {
        mapping.map(result, loweredResult);
      }
      continue;
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
      if (region.getPolicy() == NumericalPolicy::Fast &&
          map.getSemantics() == MapSemantics::Pointwise &&
          llvm::range_size(map.getBody().front().without_terminator()) > 1) {
        FailureOr<Operation *> lowered =
            lowerPointwiseMapDag(builder, map, operands);
        if (failed(lowered))
          return failure();
        mapping.map(map.getResult(0), (*lowered)->getResult(0));
        continue;
      }
      source = map.getSource();
      state = lowerMapState(map, operands);
    } else if (auto relation = dyn_cast<RelationOp>(operation)) {
      source = relation.getSource();
      state = lowerRelationState(relation, operands);
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
      operation->removeAttr(kOperationRefGroupsAttribute);
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

#include "MaterializationPlanning.inc"

#include "ScheduleLowering.inc"

class PlanSourceEventLoweringPass
    : public impl::ShuttlePlanSourceEventLoweringPassBase<
          PlanSourceEventLoweringPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<SchedulePlanOp> schedules(module.getOps<SchedulePlanOp>());
    if (schedules.empty() || !module.getOps<EventLoweringPlanOp>().empty() ||
        !module.getOps<DeviceModuleOp>().empty() ||
        !module.getOps<InvocationAbiOp>().empty() ||
        !module.getOps<ExecutableBundleOp>().empty()) {
      module.emitError(
          "requires source schedules and no Event or executable plans");
      return signalPassFailure();
    }
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    for (SchedulePlanOp schedule : schedules) {
      if (failed(schedule.verifyRegions())) {
        schedule.emitError("cannot lower events from an invalid schedule");
        return signalPassFailure();
      }
      EventLoweringPlanOp eventPlan =
          createSourceBoundImperativeEventLoweringPlan(builder, module.getLoc(),
                                                       schedule);
      if (failed(verifyEventLoweringPlanAgainstSchedule(schedule, eventPlan))) {
        schedule.emitError(
            "cannot derive exact pre-physical Event wait/notify requirements");
        return signalPassFailure();
      }
    }
  }
};

struct GpuBuildUnit {
  RegionOp region;
  MaterializationPlanOp materialization;
  SchedulePlanOp schedule;
  EventLoweringPlanOp eventPlan;
  GpuExecutableSpec spec;
  SmallVector<uint8_t> nativeTransport;
  std::string physicalRequestFingerprint;
  std::string physicalCompileRequestFingerprint;
  std::string physicalLaunchRecipeFingerprint;
  SmallVector<uint8_t> trustedEventCertificates;
  struct TrustedInternalEventCertificate {
    int64_t regionOrdinal;
    int64_t allocationOrdinal;
    std::string allocationId;
    std::string compilerIrDigest;
    std::string deviceImageDigest;
  };
  SmallVector<TrustedInternalEventCertificate> trustedInternalEventCertificates;

  bool isFatbin() const { return !nativeTransport.empty(); }
};

struct PhysicalTaskPosition {
  int64_t launch;
  int64_t position;
};

struct EventRealizationSpec {
  EventRealizationKind kind;
  SmallVector<int64_t> coveringLaunchOrder;
};

FailureOr<SmallVector<uint8_t>> bindTrustedSourceEventLegalizations(
    GpuFatbinExecutableBundle &bundle,
    ArrayRef<GpuSourceEventLegalization> legalizations) {
  SmallVector<uint8_t> trusted(bundle.eventRealizations.size(), 0);
  auto codeIdentity = gpuFatbinCodeIdentity(bundle.images, bundle.launches);
  if (!codeIdentity.ok())
    return failure();
  for (const GpuSourceEventLegalization &legalization : legalizations) {
    if (legalization.eventOrdinal < 0 ||
        legalization.eventOrdinal >=
            static_cast<int64_t>(bundle.eventRealizations.size()) ||
        trusted[legalization.eventOrdinal] || legalization.operations.empty())
      return failure();
    GpuEventRealization &realization =
        bundle.eventRealizations[legalization.eventOrdinal];
    if (realization.eventOrdinal != legalization.eventOrdinal ||
        realization.operationCertificate)
      return failure();
    GpuEventOperationCertificate certificate{
        {}, gpuExecutableBundleDigest(*codeIdentity), legalization.operations};
    certificate.compilerIrDigest =
        gpuEventOperationCertificateDigest(certificate);
    realization.operationCertificate = std::move(certificate);
    trusted[legalization.eventOrdinal] = 1;
  }
  return trusted;
}

SmallVector<GpuInternalEventOperation>
internalEventOperations(const GpuInternalEventAllocation &allocation) {
  SmallVector<GpuInternalEventOperation> operations;
  operations.reserve(allocation.notifyRelations.size() +
                     allocation.waitRelations.size());
  auto coordinateOrdinal = [&](ArrayRef<int64_t> coordinate) {
    auto found =
        llvm::find_if(allocation.mechanismCoordinates,
                      [&](const GpuInternalEventCoordinate &candidate) {
                        return ArrayRef<int64_t>(candidate) == coordinate;
                      });
    assert(found != allocation.mechanismCoordinates.end());
    return static_cast<int64_t>(
        std::distance(allocation.mechanismCoordinates.begin(), found));
  };
  for (auto [relationOrdinal, relation] :
       llvm::enumerate(allocation.notifyRelations)) {
    const int64_t mechanismOrdinal =
        coordinateOrdinal(relation.mechanismCoordinate);
    operations.push_back(
        {static_cast<int64_t>(operations.size()),
         GpuInternalEventOperationKind::Notify,
         static_cast<int64_t>(relationOrdinal), relation.taskFamily,
         relation.taskCoordinate, relation.mechanismCoordinate,
         allocation.legalizationKind, allocation.scope,
         allocation.allocationOwners[mechanismOrdinal],
         allocation.slots[mechanismOrdinal],
         allocation.generations[mechanismOrdinal], allocation.generationCount,
         1, allocation.initialCounts[mechanismOrdinal],
         allocation.releaseOnNotify,
         allocation.transactionBytesByCoordinate
             ? std::optional<int64_t>(
                   (*allocation.transactionBytesByCoordinate)[mechanismOrdinal])
             : std::nullopt});
  }
  for (auto [relationOrdinal, relation] :
       llvm::enumerate(allocation.waitRelations)) {
    const int64_t mechanismOrdinal =
        coordinateOrdinal(relation.mechanismCoordinate);
    operations.push_back(
        {static_cast<int64_t>(operations.size()),
         GpuInternalEventOperationKind::Wait,
         static_cast<int64_t>(relationOrdinal), relation.taskFamily,
         relation.taskCoordinate, relation.mechanismCoordinate,
         allocation.legalizationKind, allocation.scope,
         allocation.allocationOwners[mechanismOrdinal],
         allocation.slots[mechanismOrdinal],
         allocation.generations[mechanismOrdinal], allocation.generationCount,
         allocation.initialCounts[mechanismOrdinal],
         allocation.initialCounts[mechanismOrdinal],
         allocation.acquireBeforeConsumer, std::nullopt});
  }
  return operations;
}

bool equalInternalEventOperation(const GpuInternalEventOperation &left,
                                 const GpuInternalEventOperation &right) {
  return left.ordinal == right.ordinal && left.kind == right.kind &&
         left.relationOrdinal == right.relationOrdinal &&
         left.taskFamily == right.taskFamily &&
         left.taskCoordinate == right.taskCoordinate &&
         left.mechanismCoordinate == right.mechanismCoordinate &&
         left.legalizationKind == right.legalizationKind &&
         left.scope == right.scope &&
         left.allocationOwner == right.allocationOwner &&
         left.slot == right.slot && left.generation == right.generation &&
         left.generationCount == right.generationCount &&
         left.operationCount == right.operationCount &&
         left.initialCount == right.initialCount &&
         left.releaseAcquire == right.releaseAcquire &&
         left.transactionBytes == right.transactionBytes;
}

void updateInternalCertificateDigest(llvm::SHA256 &digest, StringRef value) {
  const std::string size = std::to_string(value.size());
  digest.update(size);
  digest.update(StringRef(":"));
  digest.update(value);
}

void updateInternalCertificateDigest(llvm::SHA256 &digest, int64_t value) {
  updateInternalCertificateDigest(digest, std::to_string(value));
}

std::string
internalEventCertificateDigest(int64_t regionOrdinal, StringRef allocationId,
                               ArrayRef<GpuInternalEventOperation> operations) {
  llvm::SHA256 digest;
  updateInternalCertificateDigest(digest, regionOrdinal);
  updateInternalCertificateDigest(digest, allocationId);
  updateInternalCertificateDigest(digest,
                                  static_cast<int64_t>(operations.size()));
  for (const GpuInternalEventOperation &operation : operations) {
    updateInternalCertificateDigest(digest, operation.ordinal);
    updateInternalCertificateDigest(digest,
                                    static_cast<int64_t>(operation.kind));
    updateInternalCertificateDigest(digest, operation.relationOrdinal);
    updateInternalCertificateDigest(digest, operation.taskFamily);
    updateInternalCertificateDigest(
        digest, static_cast<int64_t>(operation.taskCoordinate.size()));
    for (int64_t coordinate : operation.taskCoordinate)
      updateInternalCertificateDigest(digest, coordinate);
    updateInternalCertificateDigest(
        digest, static_cast<int64_t>(operation.mechanismCoordinate.size()));
    for (int64_t coordinate : operation.mechanismCoordinate)
      updateInternalCertificateDigest(digest, coordinate);
    updateInternalCertificateDigest(
        digest, static_cast<int64_t>(operation.legalizationKind));
    updateInternalCertificateDigest(digest,
                                    static_cast<int64_t>(operation.scope));
    updateInternalCertificateDigest(
        digest, static_cast<int64_t>(operation.allocationOwner.size()));
    for (int64_t coordinate : operation.allocationOwner)
      updateInternalCertificateDigest(digest, coordinate);
    updateInternalCertificateDigest(digest, operation.slot);
    updateInternalCertificateDigest(digest, operation.generation);
    updateInternalCertificateDigest(digest, operation.generationCount);
    updateInternalCertificateDigest(digest, operation.operationCount);
    updateInternalCertificateDigest(digest, operation.initialCount);
    updateInternalCertificateDigest(digest, operation.releaseAcquire ? 1 : 0);
    updateInternalCertificateDigest(
        digest, operation.transactionBytes ? *operation.transactionBytes : -1);
  }
  return llvm::toHex(digest.final(), true);
}

FailureOr<SmallVector<GpuBuildUnit::TrustedInternalEventCertificate>>
bindTrustedInternalEventLegalizations(
    const GpuFatbinExecutableBundle &bundle,
    ArrayRef<GpuInternalEventLegalization> legalizations) {
  SmallVector<GpuBuildUnit::TrustedInternalEventCertificate> certificates;
  if (legalizations.empty())
    return certificates;
  if (bundle.physicalEventProvenance.empty())
    return failure();
  absl::StatusOr<GpuInternalEventSchedule> schedule =
      parseGpuInternalEventSchedule(bundle.physicalEventProvenance);
  if (!schedule.ok())
    return failure();
  llvm::SmallDenseSet<std::pair<int64_t, int64_t>> covered;
  auto codeIdentity = gpuFatbinCodeIdentity(bundle.images, bundle.launches);
  if (!codeIdentity.ok())
    return failure();
  const std::string deviceImageDigest =
      gpuExecutableBundleDigest(*codeIdentity);
  for (const GpuInternalEventLegalization &legalization : legalizations) {
    if (legalization.regionOrdinal < 0 ||
        legalization.regionOrdinal >=
            static_cast<int64_t>(schedule->regions.size()))
      return failure();
    const GpuInternalEventRegion &region =
        schedule->regions[legalization.regionOrdinal];
    auto allocation = llvm::find_if(
        region.allocations, [&](const GpuInternalEventAllocation &candidate) {
          return candidate.allocationId == legalization.allocationId;
        });
    if (allocation == region.allocations.end())
      return failure();
    const int64_t allocationOrdinal =
        std::distance(region.allocations.begin(), allocation);
    if (!covered.insert({legalization.regionOrdinal, allocationOrdinal}).second)
      return failure();
    SmallVector<GpuInternalEventOperation> expected =
        internalEventOperations(*allocation);
    if (expected.empty() || expected.size() != legalization.operations.size() ||
        !llvm::all_of(llvm::zip(expected, legalization.operations),
                      [](auto pair) {
                        return equalInternalEventOperation(std::get<0>(pair),
                                                           std::get<1>(pair));
                      }))
      return failure();
    certificates.push_back(
        {legalization.regionOrdinal, allocationOrdinal,
         legalization.allocationId,
         internalEventCertificateDigest(legalization.regionOrdinal,
                                        legalization.allocationId, expected),
         deviceImageDigest});
  }
  return certificates;
}

FailureOr<AffineExpr> affineExpression(const GpuAffineExpression &expression,
                                       int64_t dimensions,
                                       MLIRContext *context) {
  switch (expression.kind) {
  case GpuAffineExpressionKind::Dimension:
    if (expression.payload < 0 || expression.payload >= dimensions)
      return failure();
    return getAffineDimExpr(expression.payload, context);
  case GpuAffineExpressionKind::Constant:
    return getAffineConstantExpr(expression.payload, context);
  case GpuAffineExpressionKind::Add:
  case GpuAffineExpressionKind::Multiply:
    if (expression.operands.size() != 2)
      return failure();
    break;
  case GpuAffineExpressionKind::Modulo:
  case GpuAffineExpressionKind::FloorDivide:
  case GpuAffineExpressionKind::CeilDivide:
    if (expression.operands.size() != 2 ||
        expression.operands[1].kind != GpuAffineExpressionKind::Constant ||
        expression.operands[1].payload <= 0)
      return failure();
    break;
  }
  FailureOr<AffineExpr> left =
      affineExpression(expression.operands[0], dimensions, context);
  FailureOr<AffineExpr> right =
      affineExpression(expression.operands[1], dimensions, context);
  if (failed(left) || failed(right))
    return failure();
  switch (expression.kind) {
  case GpuAffineExpressionKind::Add:
    return *left + *right;
  case GpuAffineExpressionKind::Multiply:
    return *left * *right;
  case GpuAffineExpressionKind::Modulo:
    return *left % expression.operands[1].payload;
  case GpuAffineExpressionKind::FloorDivide:
    return left->floorDiv(expression.operands[1].payload);
  case GpuAffineExpressionKind::CeilDivide:
    return left->ceilDiv(expression.operands[1].payload);
  case GpuAffineExpressionKind::Dimension:
  case GpuAffineExpressionKind::Constant:
    llvm_unreachable("leaf affine expressions returned above");
  }
  llvm_unreachable("unknown GPU affine expression kind");
}

FailureOr<AffineMap> affineMap(const GpuAffineMap &map, int64_t dimensions,
                               MLIRContext *context) {
  if (map.dimensions != dimensions || map.results.empty())
    return failure();
  SmallVector<AffineExpr> results;
  results.reserve(map.results.size());
  for (const GpuAffineExpression &result : map.results) {
    FailureOr<AffineExpr> expression =
        affineExpression(result, dimensions, context);
    if (failed(expression))
      return failure();
    results.push_back(*expression);
  }
  return AffineMap::get(dimensions, 0, results, context);
}

FailureOr<IntegerSet> integerSet(const GpuIntegerSet &set, int64_t dimensions,
                                 MLIRContext *context) {
  if (set.dimensions != dimensions ||
      set.constraints.size() != set.equalities.size())
    return failure();
  SmallVector<AffineExpr> constraints;
  SmallVector<bool> equalities;
  constraints.reserve(set.constraints.size());
  equalities.reserve(set.equalities.size());
  for (auto [constraint, equality] :
       llvm::zip_equal(set.constraints, set.equalities)) {
    if (equality > 1)
      return failure();
    FailureOr<AffineExpr> expression =
        affineExpression(constraint, dimensions, context);
    if (failed(expression))
      return failure();
    constraints.push_back(*expression);
    equalities.push_back(equality != 0);
  }
  return IntegerSet::get(dimensions, 0, constraints, equalities);
}

FailureOr<presburger::IntegerPolyhedron>
boundedRelationDomain(ArrayRef<int64_t> shape,
                      std::optional<IntegerSet> predicate) {
  if (shape.empty() ||
      llvm::any_of(shape, [](int64_t extent) { return extent <= 0; }))
    return failure();
  presburger::IntegerPolyhedron domain(
      presburger::PresburgerSpace::getSetSpace(shape.size()));
  for (auto [axis, extent] : llvm::enumerate(shape)) {
    domain.addBound(presburger::BoundType::LB, axis, 0);
    domain.addBound(presburger::BoundType::UB, axis, extent - 1);
  }
  if (predicate)
    domain = domain.intersect(FlatLinearValueConstraints(*predicate));
  return domain;
}

FailureOr<bool>
affineMapsEqualOverDomain(AffineMap left, AffineMap right,
                          const presburger::IntegerPolyhedron &domain) {
  if (left.getNumResults() != right.getNumResults())
    return failure();
  SmallVector<AffineExpr> constraints;
  SmallVector<bool> equalities;
  for (auto [leftResult, rightResult] :
       llvm::zip_equal(left.getResults(), right.getResults())) {
    constraints.push_back(leftResult - rightResult);
    equalities.push_back(true);
  }
  IntegerSet equality =
      IntegerSet::get(left.getNumDims(), 0, constraints, equalities);
  presburger::IntegerPolyhedron equalDomain =
      domain.intersect(FlatLinearValueConstraints(equality));
  return domain.isSubsetOf(equalDomain);
}

FailureOr<bool>
affineMapRangeInBounds(AffineMap map,
                       const presburger::IntegerPolyhedron &domain,
                       ArrayRef<uint64_t> shape) {
  if (map.getNumResults() != shape.size())
    return failure();
  presburger::MultiAffineFunction function(
      presburger::PresburgerSpace::getRelationSpace(),
      presburger::IntMatrix(/*rows=*/0, /*columns=*/1));
  if (failed(getMultiAffineFunctionFromMap(map, function)))
    return failure();
  presburger::IntegerRelation graph = function.getAsRelation();
  graph.intersectDomain(domain);
  presburger::IntegerPolyhedron image = graph.getRangeSet();
  presburger::IntegerPolyhedron bounds(
      presburger::PresburgerSpace::getSetSpace(shape.size()));
  for (auto [axis, extent] : llvm::enumerate(shape)) {
    if (extent == 0 ||
        extent > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return failure();
    bounds.addBound(presburger::BoundType::LB, axis, 0);
    bounds.addBound(presburger::BoundType::UB, axis,
                    static_cast<int64_t>(extent) - 1);
  }
  return image.isSubsetOf(bounds);
}

FailureOr<bool> lexicographicallyOrderedOverDomain(
    AffineMap producer, AffineMap consumer,
    const presburger::IntegerPolyhedron &domain) {
  if (producer.getNumResults() != consumer.getNumResults())
    return failure();
  presburger::PresburgerSet ordered = presburger::PresburgerSet::getEmpty(
      presburger::PresburgerSpace::getSetSpace(producer.getNumDims()));
  for (int64_t position = 0;
       position < static_cast<int64_t>(producer.getNumResults()); ++position) {
    SmallVector<AffineExpr> constraints;
    SmallVector<bool> equalities;
    for (int64_t prefix = 0; prefix < position; ++prefix) {
      constraints.push_back(producer.getResult(prefix) -
                            consumer.getResult(prefix));
      equalities.push_back(true);
    }
    constraints.push_back(consumer.getResult(position) -
                          producer.getResult(position) - 1);
    equalities.push_back(false);
    IntegerSet earlier =
        IntegerSet::get(producer.getNumDims(), 0, constraints, equalities);
    ordered.unionInPlace(domain.intersect(FlatLinearValueConstraints(earlier)));
  }
  return presburger::PresburgerSet(domain).isSubsetOf(ordered);
}

LogicalResult
verifyPhysicalEventCoverPieces(EventTensorOp event,
                               const GpuEventRealization &realization,
                               ArrayRef<GpuLaunch> launches) {
  if (realization.coverPieces.empty())
    return success();
  const int64_t dimensions = event.getRelationDomainShape().size();
  FailureOr<presburger::IntegerPolyhedron> required = boundedRelationDomain(
      event.getRelationDomainShape(), event.getRelationPredicate());
  if (failed(required))
    return event.emitOpError("has an unsupported exact relation domain");
  presburger::PresburgerSet covered = presburger::PresburgerSet::getEmpty(
      presburger::PresburgerSpace::getSetSpace(dimensions));
  for (const GpuEventCoverPiece &piece : realization.coverPieces) {
    presburger::IntegerPolyhedron pieceDomain = *required;
    if (piece.relationSubset) {
      FailureOr<IntegerSet> subset =
          integerSet(*piece.relationSubset, dimensions, event.getContext());
      if (failed(subset))
        return event.emitOpError("has an invalid physical cover predicate");
      pieceDomain = pieceDomain.intersect(FlatLinearValueConstraints(*subset));
    }
    const int64_t producerLaunch = piece.producer.launchOrdinal;
    const int64_t consumerLaunch = piece.consumer.launchOrdinal;
    if (producerLaunch < 0 || consumerLaunch < 0 ||
        producerLaunch >= static_cast<int64_t>(launches.size()) ||
        consumerLaunch >= static_cast<int64_t>(launches.size()))
      return event.emitOpError("physical cover references an unknown launch");
    FailureOr<AffineMap> producerOwner =
        affineMap(piece.producer.owner, dimensions, event.getContext());
    FailureOr<AffineMap> consumerOwner =
        affineMap(piece.consumer.owner, dimensions, event.getContext());
    FailureOr<AffineMap> producerTime =
        affineMap(piece.producer.localTime, dimensions, event.getContext());
    FailureOr<AffineMap> consumerTime =
        affineMap(piece.consumer.localTime, dimensions, event.getContext());
    if (failed(producerOwner) || failed(consumerOwner) ||
        failed(producerTime) || failed(consumerTime))
      return event.emitOpError("has invalid physical owner/time maps");
    FailureOr<bool> producerOwnerInBounds = affineMapRangeInBounds(
        *producerOwner, pieceDomain, launches[producerLaunch].grid);
    FailureOr<bool> consumerOwnerInBounds = affineMapRangeInBounds(
        *consumerOwner, pieceDomain, launches[consumerLaunch].grid);
    if (failed(producerOwnerInBounds) || !*producerOwnerInBounds ||
        failed(consumerOwnerInBounds) || !*consumerOwnerInBounds)
      return event.emitOpError("physical owner map escapes its launch grid");
    if (realization.kind == GpuEventRealizationKind::StreamOrder) {
      if (producerLaunch >= consumerLaunch)
        return event.emitOpError(
            "stream-order cover must cross forward in launch order");
    } else {
      if (producerLaunch != consumerLaunch)
        return event.emitOpError(
            "same-launch cover must fuse both Event endpoints");
      FailureOr<bool> sameOwner = affineMapsEqualOverDomain(
          *producerOwner, *consumerOwner, pieceDomain);
      if (failed(sameOwner) || !*sameOwner)
        return event.emitOpError(
            "same-launch cover does not preserve sequential ownership");
      if (realization.kind == GpuEventRealizationKind::ProgramOrder) {
        FailureOr<bool> ordered = lexicographicallyOrderedOverDomain(
            *producerTime, *consumerTime, pieceDomain);
        if (failed(ordered) || !*ordered)
          return event.emitOpError(
              "program-order cover does not order every Event instance");
      }
    }
    covered.unionInPlace(pieceDomain);
  }
  if (!presburger::PresburgerSet(*required).isSubsetOf(covered))
    return event.emitOpError(
        "physical Event cover omits part of the exact relation");
  return success();
}

LogicalResult verifyPhysicalEventMechanism(
    EventTensorOp event, const GpuEventRealization &realization,
    const GpuExecutable &executable, bool trustedInProcessCertificate) {
  const bool erased =
      realization.kind == GpuEventRealizationKind::ProgramOrder ||
      realization.kind == GpuEventRealizationKind::StreamOrder ||
      realization.kind == GpuEventRealizationKind::AlgebraicRefinement;
  if (erased) {
    if (realization.mechanism || realization.operationCertificate)
      return event.emitOpError("erased physical Event cannot allocate a "
                               "synchronization mechanism or certificate");
    return success();
  }
  if (!realization.mechanism)
    return event.emitOpError(
        "physical Event realization requires a bounded mechanism allocation");
  if (!realization.operationCertificate)
    return event.emitOpError(
        "physical Event realization requires an emitted-operation "
        "certificate");

  const GpuEventMechanismAllocation &mechanism = *realization.mechanism;
  const GpuEventOperationCertificate &certificate =
      *realization.operationCertificate;
  if (certificate.compilerIrDigest !=
      gpuEventOperationCertificateDigest(certificate))
    return event.emitOpError(
        "physical Event operation certificate has an invalid compiler-IR "
        "digest");
  auto codeIdentity =
      gpuFatbinCodeIdentity(executable.codeImages(), executable.launches());
  if (!codeIdentity.ok() ||
      certificate.deviceImageDigest != gpuExecutableBundleDigest(*codeIdentity))
    return event.emitOpError(
        "physical Event operation certificate does not bind the loaded "
        "device image");
  if (certificate.operations.size() != realization.coverPieces.size() * 2)
    return event.emitOpError(
        "physical Event operation certificate does not exactly cover every "
        "wait and notify site");

  const int64_t dimensions = event.getRelationDomainShape().size();
  FailureOr<AffineMap> mechanismSlot =
      affineMap(mechanism.slot, dimensions, event.getContext());
  FailureOr<AffineMap> mechanismGeneration =
      affineMap(mechanism.generationPhase, dimensions, event.getContext());
  if (failed(mechanismSlot) || failed(mechanismGeneration))
    return event.emitOpError(
        "physical Event operation certificate has invalid mechanism maps");
  SmallVector<uint8_t> covered(realization.coverPieces.size(), 0);
  for (auto [ordinal, operation] : llvm::enumerate(certificate.operations)) {
    if (operation.ordinal != static_cast<int64_t>(ordinal) ||
        operation.coverPieceOrdinal < 0 ||
        operation.coverPieceOrdinal >=
            static_cast<int64_t>(realization.coverPieces.size()))
      return event.emitOpError(
          "physical Event operation certificate has an invalid operation "
          "ordinal");
    const GpuEventCoverPiece &piece =
        realization.coverPieces[operation.coverPieceOrdinal];
    const bool notify = operation.kind == GpuEventOperationKind::Notify;
    const uint8_t operationBit = notify ? 1 : 2;
    uint8_t &pieceCoverage = covered[operation.coverPieceOrdinal];
    if (pieceCoverage & operationBit)
      return event.emitOpError(
          "physical Event operation certificate duplicates a wait or notify "
          "site");
    pieceCoverage |= operationBit;
    const GpuEventCoverEndpoint &expectedEndpoint =
        notify ? piece.producer : piece.consumer;
    FailureOr<AffineMap> operationOwner = affineMap(
        operation.executionPoint.owner, dimensions, event.getContext());
    FailureOr<AffineMap> expectedOwner =
        affineMap(expectedEndpoint.owner, dimensions, event.getContext());
    FailureOr<AffineMap> operationTime = affineMap(
        operation.executionPoint.localTime, dimensions, event.getContext());
    FailureOr<AffineMap> expectedTime =
        affineMap(expectedEndpoint.localTime, dimensions, event.getContext());
    FailureOr<AffineMap> operationSlot =
        affineMap(operation.slot, dimensions, event.getContext());
    FailureOr<AffineMap> operationGeneration =
        affineMap(operation.generationPhase, dimensions, event.getContext());
    if (failed(operationOwner) || failed(expectedOwner) ||
        failed(operationTime) || failed(expectedTime) ||
        failed(operationSlot) || failed(operationGeneration) ||
        operation.executionPoint.launchOrdinal !=
            expectedEndpoint.launchOrdinal ||
        *operationOwner != *expectedOwner || *operationTime != *expectedTime ||
        *operationSlot != *mechanismSlot ||
        *operationGeneration != *mechanismGeneration ||
        operation.scope != mechanism.scope ||
        operation.count != event.getNotifyCount() ||
        operation.count != mechanism.arrivalCount ||
        operation.releaseAcquire !=
            (notify ? event.getRelease() : event.getAcquire()))
      return event.emitOpError(
          "physical Event operation certificate does not match its exact "
          "source requirement and mechanism");
  }
  if (llvm::any_of(covered, [](uint8_t value) { return value != 3; }))
    return event.emitOpError(
        "physical Event operation certificate omits a wait or notify site");

  if (!trustedInProcessCertificate)
    return event.emitOpError(
        "external device images cannot establish trusted in-process provenance "
        "for a physical Event operation certificate");
  return success();
}

using PhysicalTaskCoverage =
    llvm::SmallDenseMap<int64_t, SmallVector<PhysicalTaskPosition>>;

FailureOr<SmallVector<EventRealizationSpec>>
deriveEventRealizationSpecs(SchedulePlanOp schedule,
                            const PhysicalTaskCoverage &coverage) {
  // These are the exact logical task-dependence events from SchedulePlan. A
  // fused physical kernel may still use independently derived mbarriers for
  // shared-memory pipelines after every event here has erased to phase order.
  SmallVector<EventRealizationSpec> result;
  for (EventTensorOp event :
       schedule.getBody().front().getOps<EventTensorOp>()) {
    auto producers = coverage.find(event.getProducerTask());
    auto consumers = coverage.find(event.getConsumerTask());
    if (producers == coverage.end() || consumers == coverage.end() ||
        producers->second.empty() || consumers->second.empty())
      return event.emitOpError(
          "physical Event erasure is missing source-task coverage");
    const bool localReduction =
        event.getProducerPhase() == "partial" &&
        event.getConsumerPhase() == "merge" &&
        event.getProducerTask() == event.getConsumerTask();
    if (localReduction) {
      if (producers->second.size() < 2)
        return event.emitOpError(
            "partial-to-merge Event erasure requires two ordered physical "
            "launches refining the same Fold task");
      auto first =
          llvm::min_element(producers->second, [](PhysicalTaskPosition left,
                                                  PhysicalTaskPosition right) {
            return std::tie(left.launch, left.position) <
                   std::tie(right.launch, right.position);
          });
      auto last =
          llvm::max_element(producers->second, [](PhysicalTaskPosition left,
                                                  PhysicalTaskPosition right) {
            return std::tie(left.launch, left.position) <
                   std::tie(right.launch, right.position);
          });
      if (first->launch >= last->launch)
        return event.emitOpError(
            "partial-to-merge Event has no exact same-stream launch-order "
            "witness");
      result.push_back(
          {EventRealizationKind::StreamOrderErased,
           {first->launch, first->position, last->launch, last->position}});
      continue;
    }
    EventRealizationSpec spec{EventRealizationKind::ProgramOrderErased, {}};
    for (PhysicalTaskPosition consumer : consumers->second) {
      std::optional<PhysicalTaskPosition> witness;
      for (PhysicalTaskPosition producer : producers->second) {
        if (producer.launch == consumer.launch &&
            producer.position < consumer.position) {
          witness = producer;
          break;
        }
      }
      if (!witness) {
        for (PhysicalTaskPosition producer : producers->second) {
          if (producer.launch >= consumer.launch)
            continue;
          if (!witness || witness->launch < producer.launch)
            witness = producer;
        }
      }
      if (!witness)
        return event.emitOpError(
            "physical Event Tensor edge is unresolved: no ordered producer "
            "covers one consumer launch");
      if (witness->launch < consumer.launch)
        spec.kind = EventRealizationKind::StreamOrderErased;
      llvm::append_range(
          spec.coveringLaunchOrder,
          ArrayRef<int64_t>({witness->launch, witness->position,
                             consumer.launch, consumer.position}));
    }
    result.push_back(std::move(spec));
  }
  return result;
}

FailureOr<PhysicalTaskCoverage> physicalTaskCoverage(const GpuBuildUnit &unit) {
  PhysicalTaskCoverage coverage;
  if (!unit.isFatbin()) {
    for (const GpuEntrySpec &entry : unit.spec.entries)
      coverage[entry.ordinal].push_back({entry.ordinal, 0});
    return coverage;
  }
  absl::StatusOr<std::shared_ptr<const GpuExecutable>> executable =
      GpuExecutable::Load(unit.nativeTransport);
  if (!executable.ok())
    return failure();
  for (const GpuLaunch &launch : (*executable)->launches())
    for (auto [position, sourceTask] :
         llvm::enumerate(launch.sourceTaskOrdinals))
      coverage[sourceTask].push_back(
          {launch.taskOrdinal, static_cast<int64_t>(position)});
  return coverage;
}

FailureOr<SmallVector<EventRealizationSpec>>
physicalEventRealizationSpecs(SchedulePlanOp schedule,
                              const GpuExecutable &executable,
                              ArrayRef<uint8_t> trustedCertificates = {}) {
  SmallVector<EventTensorOp> events(
      schedule.getBody().front().getOps<EventTensorOp>());
  const size_t eventCount = events.size();
  if (executable.eventRealizations().size() != eventCount)
    return failure();
  if (!trustedCertificates.empty() && trustedCertificates.size() != eventCount)
    return failure();
  SmallVector<EventRealizationSpec> result;
  result.reserve(eventCount);
  for (auto [ordinal, realization] :
       llvm::enumerate(executable.eventRealizations())) {
    if (realization.eventOrdinal != static_cast<int64_t>(ordinal))
      return failure();
    EventTensorOp event = events[ordinal];
    if (failed(verifyPhysicalEventCoverPieces(event, realization,
                                              executable.launches())) ||
        failed(verifyPhysicalEventMechanism(event, realization, executable,
                                            !trustedCertificates.empty() &&
                                                trustedCertificates[ordinal])))
      return failure();
    for (size_t index = 0; index < realization.coveringLaunchOrder.size();
         index += 4) {
      const int64_t producerLaunch = realization.coveringLaunchOrder[index];
      const int64_t consumerLaunch = realization.coveringLaunchOrder[index + 2];
      if (producerLaunch < 0 || consumerLaunch < 0 ||
          producerLaunch >=
              static_cast<int64_t>(executable.launches().size()) ||
          consumerLaunch >=
              static_cast<int64_t>(executable.launches().size()) ||
          !llvm::is_contained(
              executable.launches()[producerLaunch].sourceTaskOrdinals,
              event.getProducerTask()) ||
          !llvm::is_contained(
              executable.launches()[consumerLaunch].sourceTaskOrdinals,
              event.getConsumerTask()))
        return failure();
    }
    EventRealizationKind kind;
    switch (realization.kind) {
    case GpuEventRealizationKind::ProgramOrder:
      kind = EventRealizationKind::ProgramOrderErased;
      break;
    case GpuEventRealizationKind::StreamOrder:
      kind = EventRealizationKind::StreamOrderErased;
      break;
    case GpuEventRealizationKind::AlgebraicRefinement:
      kind = EventRealizationKind::AlgebraicRefinementErased;
      break;
    case GpuEventRealizationKind::MBarrier:
      kind = EventRealizationKind::MBarrier;
      break;
    case GpuEventRealizationKind::Pipeline:
      kind = EventRealizationKind::Pipeline;
      break;
    case GpuEventRealizationKind::Queue:
      return event.emitOpError(
          "device queue Event realization is not yet supported");
    }
    result.push_back(
        {kind, SmallVector<int64_t>(realization.coveringLaunchOrder.begin(),
                                    realization.coveringLaunchOrder.end())});
  }
  return result;
}

FailureOr<GpuExecutableSpec>
deriveFatbinBoundary(RegionOp region, MaterializationPlanOp materialization,
                     SchedulePlanOp schedule,
                     const GpuFatbinExecutableBundle &bundle,
                     GpuTarget target) {
  auto codeIdentity = gpuFatbinCodeIdentity(bundle.images, bundle.launches);
  if (!codeIdentity.ok() ||
      bundle.deviceSchemaVersion != fatbinDeviceSchemaForTarget(target) ||
      bundle.codeFormat != fatbinCodeFormatForTarget(target) ||
      bundle.scheduleFingerprint != schedule.getFingerprint() ||
      bundle.planFingerprint != materialization.getFingerprint())
    return failure();
  FailureOr<SmallVector<ExecutableSlotSpec>> expectedSlots =
      deriveExecutableSlots(materialization);
  if (failed(expectedSlots)) {
    materialization.emitError("cannot derive source materialization ABI");
    return failure();
  }
  auto manifest =
      region->getParentOfType<ModuleOp>()->getAttrOfType<DictionaryAttr>(
          kCoverageManifestAttribute);
  auto tuning =
      manifest ? manifest.getAs<StringAttr>("canonical_tuning") : StringAttr{};
  FailureOr<DerivedPlan> derived = deriveGpuPlan(
      region, tuning && tuning.getValue() == kFusionCanonicalTuning);
  if (failed(derived)) {
    materialization.emitError(
        "cannot rederive physical artifact source ownership");
    return failure();
  }
  FailureOr<SmallVector<std::optional<int64_t>>> expectedBindings =
      deriveGpuBindingIndices(region, *derived, *expectedSlots);
  if (failed(expectedBindings))
    return failure();

  GpuExecutableSpec spec;
  spec.code.append(codeIdentity->begin(), codeIdentity->end());
  for (const auto &image : bundle.images)
    spec.rawFatbinCodeBytes += image.size();
  SmallVector<bool> projectedSourceBuffers(expectedSlots->size(), false);
  for (auto [ordinal, slot] : llvm::enumerate(bundle.slots)) {
    if (slot.storage == MaterializationStorage::External) {
      std::optional<size_t> sourceOrdinal;
      for (auto [candidateOrdinal, candidate] :
           llvm::enumerate(*expectedSlots)) {
        if (candidate.storage != MaterializationStorage::External ||
            !(*expectedBindings)[candidateOrdinal])
          continue;
        const ExecutableBindingKind candidateBinding =
            candidate.access == ExecutableAccess::Read
                ? ExecutableBindingKind::Operand
                : ExecutableBindingKind::Result;
        if (slot.bindingKind == candidateBinding &&
            slot.bindingIndex == *(*expectedBindings)[candidateOrdinal]) {
          if (sourceOrdinal) {
            materialization.emitError(
                "source external binding does not identify one buffer");
            return failure();
          }
          sourceOrdinal = candidateOrdinal;
        }
      }
      if (!sourceOrdinal || projectedSourceBuffers[*sourceOrdinal]) {
        materialization.emitError("physical artifact slot ")
            << ordinal << " does not project to one source external buffer";
        return failure();
      }
      const ExecutableSlotSpec &expected = (*expectedSlots)[*sourceOrdinal];
      const bool validAlignment = slot.alignment >= expected.alignment &&
                                  slot.alignment <= 256 &&
                                  (slot.alignment & (slot.alignment - 1)) == 0;
      if (slot.ordinal != static_cast<int64_t>(ordinal) ||
          slot.tensorType != expected.tensorType ||
          slot.requiredBytes != expected.requiredBytes ||
          slot.byteStrides != ArrayRef<int64_t>(expected.strides) ||
          slot.byteOffset != 0 || !validAlignment ||
          slot.addressSpace != ExecutableAddressSpace::Device ||
          slot.access != expected.access ||
          slot.aliasGroup != static_cast<int64_t>(ordinal) ||
          slot.reuseGroup != static_cast<int64_t>(ordinal)) {
        materialization.emitError("physical artifact slot ")
            << ordinal << " does not match projected source buffer "
            << *sourceOrdinal;
        return failure();
      }
      projectedSourceBuffers[*sourceOrdinal] = true;
      spec.slots.push_back({expected.tensorType, expected.requiredBytes,
                            expected.strides, slot.alignment, expected.access,
                            expected.storage});
      spec.bindingIndices.push_back(slot.bindingIndex);
      spec.reuseGroups.push_back(slot.reuseGroup);
      spec.sourceBuffers.push_back(*sourceOrdinal);
      continue;
    }
    auto tensor = dyn_cast<RankedTensorType>(slot.tensorType);
    const bool validAlignment = slot.alignment > 0 && slot.alignment <= 256 &&
                                (slot.alignment & (slot.alignment - 1)) == 0;
    if (slot.ordinal != static_cast<int64_t>(ordinal) || !tensor ||
        !tensor.hasStaticShape() ||
        (!tensor.getElementType().isBF16() &&
         !tensor.getElementType().isF32()) ||
        slot.requiredBytes <= 0 || slot.byteOffset != 0 || !validAlignment ||
        slot.addressSpace != ExecutableAddressSpace::Device ||
        slot.access != ExecutableAccess::ReadWrite ||
        slot.storage != MaterializationStorage::Temporary ||
        slot.aliasGroup != static_cast<int64_t>(ordinal) ||
        slot.reuseGroup < 0 ||
        slot.bindingKind != ExecutableBindingKind::None ||
        slot.bindingIndex != -1) {
      materialization.emitError("physical-private slot ")
          << ordinal << " must be bounded unbound device temporary storage";
      return failure();
    }
    spec.slots.push_back({slot.tensorType, slot.requiredBytes, slot.byteStrides,
                          slot.alignment, slot.access, slot.storage});
    spec.bindingIndices.push_back(std::nullopt);
    spec.reuseGroups.push_back(slot.reuseGroup);
    spec.sourceBuffers.push_back(-1);
  }
  for (auto [sourceOrdinal, expected] : llvm::enumerate(*expectedSlots)) {
    if (expected.storage == MaterializationStorage::External &&
        !projectedSourceBuffers[sourceOrdinal]) {
      materialization.emitError(
          "physical artifact omits source external buffer ")
          << sourceOrdinal;
      return failure();
    }
    if (expected.storage != MaterializationStorage::External &&
        expected.storage != MaterializationStorage::Temporary) {
      materialization.emitError("only source temporaries may be elided");
      return failure();
    }
  }
  for (const GpuLaunch &launch : bundle.launches) {
    ArrayRef<uint8_t> image = bundle.images[launch.imageIndex];
    SmallVector<int64_t> grid(launch.grid.begin(), launch.grid.end());
    SmallVector<int64_t> block(launch.block.begin(), launch.block.end());
    spec.entries.push_back(
        {launch.taskOrdinal,
         0,
         static_cast<int64_t>(image.size()),
         SmallVector<int64_t>(launch.inputSlots.begin(),
                              launch.inputSlots.end()),
         SmallVector<int64_t>(launch.outputSlots.begin(),
                              launch.outputSlots.end()),
         SmallVector<int64_t>(launch.dependencies.begin(),
                              launch.dependencies.end()),
         ExecutablePredication::None,
         std::nullopt,
         {static_cast<int64_t>(launch.grid[0]),
          static_cast<int64_t>(launch.grid[1]),
          static_cast<int64_t>(launch.grid[2])},
         {static_cast<int64_t>(launch.block[0]),
          static_cast<int64_t>(launch.block[1]),
          static_cast<int64_t>(launch.block[2])},
         static_cast<int64_t>(launch.dynamicSharedMemoryBytes),
         static_cast<int64_t>(launch.arguments.size()),
         std::nullopt,
         {},
         executableCodeDigest(ArrayRef<int8_t>(
             reinterpret_cast<const int8_t *>(image.data()), image.size()))});
  }
  return spec;
}

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
    codeBytes +=
        unit.isFatbin() ? unit.spec.rawFatbinCodeBytes : unit.spec.code.size();
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
  BuildGpuExecutableBundlePass() = default;
  explicit BuildGpuExecutableBundlePass(GpuTarget gpuTarget) {
    target = gpuTarget == GpuTarget::Sm90 ? "sm_90" : "sm_100";
  }
  BuildGpuExecutableBundlePass(
      GpuTarget gpuTarget,
      std::shared_ptr<const GpuPhysicalArtifactProvider> provider)
      : provider(std::move(provider)) {
    target = gpuTarget == GpuTarget::Sm90 ? "sm_90" : "sm_100";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<GpuTarget> gpuTarget = parseGpuTarget(target);
    if (failed(gpuTarget)) {
      module.emitError("requires CUDA target sm_90 or sm_100");
      return signalPassFailure();
    }
    SmallVector<MaterializationPlanOp> materializations(
        module.getOps<MaterializationPlanOp>());
    SmallVector<SchedulePlanOp> schedules(module.getOps<SchedulePlanOp>());
    SmallVector<EventLoweringPlanOp> sourceEventPlans(
        module.getOps<EventLoweringPlanOp>());
    if (materializations.empty() ||
        materializations.size() != schedules.size() ||
        sourceEventPlans.size() != materializations.size() ||
        llvm::any_of(sourceEventPlans,
                     [](EventLoweringPlanOp eventPlan) {
                       return eventPlan.getPhysicalLaunchRecipeFingerprint();
                     }) ||
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
      EventLoweringPlanOp sourceEventPlan =
          uniqueSourceEventPlanForSchedule(sourceEventPlans, schedule);
      if (!sourceEventPlan) {
        schedule.emitError(
            "requires one exact source-bound Event lowering plan before "
            "physical binding");
        return signalPassFailure();
      }
      if (failed(verifyEventLoweringPlanAgainstSchedule(schedule,
                                                        sourceEventPlan))) {
        schedule.emitError(
            "cannot derive exact pre-physical Event wait/notify requirements");
        return signalPassFailure();
      }
      const std::string requestFingerprint =
          gpuPhysicalArtifactRequestFingerprint(
              region, materialization, schedule, sourceEventPlan, *gpuTarget);
      std::optional<GpuPhysicalArtifactCompilation> physicalCompilation;
      if (provider) {
        FailureOr<SmallVector<ExecutableSlotSpec>> sourceSlots =
            deriveExecutableSlots(materialization);
        auto manifest =
            module->getAttrOfType<DictionaryAttr>(kCoverageManifestAttribute);
        auto tuning = manifest ? manifest.getAs<StringAttr>("canonical_tuning")
                               : StringAttr{};
        FailureOr<DerivedPlan> sourcePlan = deriveGpuPlan(
            region, tuning && tuning.getValue() == kFusionCanonicalTuning);
        FailureOr<SmallVector<std::optional<int64_t>>> sourceBindings =
            failure();
        FailureOr<SmallVector<std::optional<int64_t>>> sourceRegionResults =
            failure();
        FailureOr<SmallVector<int64_t>> sourceReuse = failure();
        if (succeeded(sourceSlots) && succeeded(sourcePlan)) {
          sourceBindings =
              deriveGpuBindingIndices(region, *sourcePlan, *sourceSlots);
          sourceRegionResults =
              deriveGpuRegionResultIndices(region, *sourcePlan, *sourceSlots);
        }
        if (succeeded(sourceSlots))
          sourceReuse = deriveGpuReuseGroups(materialization, *sourceSlots);
        if (failed(sourceBindings) || failed(sourceRegionResults) ||
            failed(sourceReuse)) {
          module.emitError(
              "cannot derive source bindings or reuse for GPU physical "
              "artifact");
          return signalPassFailure();
        }
        GpuPhysicalArtifactRequest request{region,
                                           materialization,
                                           schedule,
                                           sourceEventPlan,
                                           *gpuTarget,
                                           requestFingerprint,
                                           std::move(*sourceBindings),
                                           std::move(*sourceRegionResults),
                                           std::move(*sourceReuse)};
        absl::StatusOr<std::optional<GpuPhysicalArtifactCompilation>> provided =
            provider->compile(request);
        if (!provided.ok()) {
          module.emitError("GPU physical artifact provider failed: ")
              << provided.status().message();
          return signalPassFailure();
        }
        physicalCompilation = std::move(*provided);
      }
      if (!physicalCompilation &&
          isCertifiedReverseStatePlan(materialization)) {
        materialization.emitError(
            "certified reverse-state Contract/Fold plan has no matching GPU "
            "physical artifact for its structural boundary");
        return signalPassFailure();
      }
      if (physicalCompilation) {
        GpuFatbinExecutableBundle &fatbin = physicalCompilation->bundle;
        const std::string physicalLaunchRecipeFingerprint =
            fatbin.scheduleFingerprint;
        const std::string physicalCompileRequestFingerprint =
            fatbin.planFingerprint;
        fatbin.scheduleFingerprint = schedule.getFingerprint().str();
        fatbin.planFingerprint = materialization.getFingerprint().str();
        FailureOr<SmallVector<uint8_t>> trustedCertificates =
            bindTrustedSourceEventLegalizations(
                fatbin, physicalCompilation->sourceEventLegalizations);
        FailureOr<SmallVector<GpuBuildUnit::TrustedInternalEventCertificate>>
            trustedInternalCertificates = bindTrustedInternalEventLegalizations(
                fatbin, physicalCompilation->internalEventLegalizations);
        FailureOr<GpuExecutableSpec> spec = deriveFatbinBoundary(
            region, materialization, schedule, fatbin, *gpuTarget);
        absl::StatusOr<SmallVector<uint8_t>> transport =
            serializeGpuFatbinExecutableBundle(fatbin);
        if (failed(trustedCertificates) ||
            failed(trustedInternalCertificates) || failed(spec) ||
            !transport.ok()) {
          module.emitError(
              "physical artifact does not match the source-bound GPU plan");
          return signalPassFailure();
        }
        units.push_back({region, materialization, schedule, sourceEventPlan,
                         std::move(*spec), std::move(*transport),
                         requestFingerprint, physicalCompileRequestFingerprint,
                         physicalLaunchRecipeFingerprint,
                         std::move(*trustedCertificates),
                         std::move(*trustedInternalCertificates)});
        continue;
      }
      FailureOr<GpuExecutableSpec> spec =
          deriveGpuExecutable(region, materialization, schedule, *gpuTarget);
      if (failed(spec)) {
        module.emitError("failed to build exact GPU executable bundle");
        return signalPassFailure();
      }
      units.push_back({region,
                       materialization,
                       schedule,
                       sourceEventPlan,
                       std::move(*spec),
                       {},
                       {},
                       {},
                       {},
                       {}});
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
      deviceState.addAttribute("schema_version",
                               builder.getI64IntegerAttr(
                                   unit.isFatbin()
                                       ? fatbinDeviceSchemaForTarget(*gpuTarget)
                                       : deviceSchemaForTarget(*gpuTarget)));
      deviceState.addAttribute(
          "code_format",
          ExecutableCodeFormatAttr::get(
              module.getContext(), unit.isFatbin()
                                       ? fatbinCodeFormatForTarget(*gpuTarget)
                                       : codeFormatForTarget(*gpuTarget)));
      deviceState.addAttribute("policy", schedule.getPolicyAttr());
      deviceState.addAttribute("source_schedule_fingerprint",
                               schedule.getFingerprintAttr());
      deviceState.addAttribute(
          "code", DenseI8ArrayAttr::get(module.getContext(), spec.code));
      deviceState.addAttribute("code_digest",
                               builder.getStringAttr(codeDigest));
      deviceState.addAttribute("fingerprint",
                               builder.getStringAttr(std::string(64, '0')));
      if (unit.isFatbin()) {
        deviceState.addAttribute(
            "native_transport",
            DenseI8ArrayAttr::get(
                module.getContext(),
                ArrayRef<int8_t>(reinterpret_cast<const int8_t *>(
                                     unit.nativeTransport.data()),
                                 unit.nativeTransport.size())));
        deviceState.addAttribute(
            "physical_request_fingerprint",
            builder.getStringAttr(unit.physicalRequestFingerprint));
        deviceState.addAttribute(
            "physical_compile_request_fingerprint",
            builder.getStringAttr(unit.physicalCompileRequestFingerprint));
        deviceState.addAttribute(
            "physical_launch_recipe_fingerprint",
            builder.getStringAttr(unit.physicalLaunchRecipeFingerprint));
        {
          absl::StatusOr<std::shared_ptr<const GpuExecutable>> executable =
              GpuExecutable::Load(unit.nativeTransport);
          if (!executable.ok()) {
            schedule.emitError("cannot read structured device IR snapshots");
            return signalPassFailure();
          }
          if (!(*executable)->deviceIrSnapshots().empty()) {
            SmallVector<Attribute> digests;
            for (const GpuDeviceIrSnapshot &snapshot :
                 (*executable)->deviceIrSnapshots())
              digests.push_back(
                  builder.getStringAttr(snapshot.mlirBytecodeDigest));
            deviceState.addAttribute("device_ir_snapshot_digests",
                                     builder.getArrayAttr(digests));
          }
        }
        if (llvm::is_contained(unit.trustedEventCertificates, uint8_t{1})) {
          absl::StatusOr<std::shared_ptr<const GpuExecutable>> executable =
              GpuExecutable::Load(unit.nativeTransport);
          if (!executable.ok()) {
            schedule.emitError(
                "cannot read trusted source Event certificate provenance");
            return signalPassFailure();
          }
          SmallVector<Attribute> certificates;
          for (auto [eventOrdinal, trusted] :
               llvm::enumerate(unit.trustedEventCertificates)) {
            if (!trusted)
              continue;
            const auto &certificate = (*executable)
                                          ->eventRealizations()[eventOrdinal]
                                          .operationCertificate;
            if (!certificate) {
              schedule.emitError(
                  "trusted source Event legalization has no certificate");
              return signalPassFailure();
            }
            certificates.push_back(builder.getDictionaryAttr(
                {builder.getNamedAttr("event",
                                      builder.getI64IntegerAttr(eventOrdinal)),
                 builder.getNamedAttr(
                     "compiler_ir_digest",
                     builder.getStringAttr(certificate->compilerIrDigest))}));
          }
          deviceState.addAttribute("trusted_source_event_certificates",
                                   builder.getArrayAttr(certificates));
        }
        if (!unit.trustedInternalEventCertificates.empty()) {
          SmallVector<Attribute> certificates;
          for (const auto &certificate : unit.trustedInternalEventCertificates)
            certificates.push_back(builder.getDictionaryAttr(
                {builder.getNamedAttr("region", builder.getI64IntegerAttr(
                                                    certificate.regionOrdinal)),
                 builder.getNamedAttr(
                     "allocation",
                     builder.getI64IntegerAttr(certificate.allocationOrdinal)),
                 builder.getNamedAttr(
                     "allocation_id",
                     builder.getStringAttr(certificate.allocationId)),
                 builder.getNamedAttr(
                     "compiler_ir_digest",
                     builder.getStringAttr(certificate.compilerIrDigest)),
                 builder.getNamedAttr(
                     "device_image_digest",
                     builder.getStringAttr(certificate.deviceImageDigest))}));
          deviceState.addAttribute("trusted_internal_event_certificates",
                                   builder.getArrayAttr(certificates));
        }
      }
      deviceState.addRegion();
      auto device = cast<DeviceModuleOp>(builder.create(deviceState));
      device.getBody().push_back(new Block());
      builder.setInsertionPointToEnd(&device.getBody().front());
      if (!unit.isFatbin())
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
              builder.getNamedAttr(
                  "grid",
                  DenseI64ArrayAttr::get(module.getContext(), entry.grid)),
              builder.getNamedAttr(
                  "block",
                  DenseI64ArrayAttr::get(module.getContext(), entry.block)),
              builder.getNamedAttr(
                  "dynamic_shared_bytes",
                  builder.getI64IntegerAttr(entry.dynamicSharedBytes)),
              builder.getNamedAttr("kernel_arity", builder.getI64IntegerAttr(
                                                       entry.kernelArity))};
          if (entry.reductionOrder)
            attributes.push_back(builder.getNamedAttr(
                "reduction_order",
                ScheduleReductionOrderAttr::get(module.getContext(),
                                                *entry.reductionOrder)));
          if (entry.onlineStateProtocol) {
            SmallVector<Attribute> logicalMaps;
            for (AffineMap map : entry.logicalCoordinateMaps)
              logicalMaps.push_back(AffineMapAttr::get(map));
            attributes.push_back(builder.getNamedAttr(
                "online_state_protocol",
                builder.getStringAttr(*entry.onlineStateProtocol)));
            attributes.push_back(builder.getNamedAttr(
                "logical_coordinate_maps", builder.getArrayAttr(logicalMaps)));
          }
          createScheduleRecord(builder, module.getLoc(),
                               DeviceEntryOp::getOperationName(), attributes);
        }
      builder.create<DeviceModuleYieldOp>(module.getLoc());
      device.setFingerprint(deviceModuleFingerprint(device));

      FailureOr<SmallVector<EventRealizationSpec>> eventRealizations =
          failure();
      if (unit.isFatbin()) {
        absl::StatusOr<std::shared_ptr<const GpuExecutable>> executable =
            GpuExecutable::Load(unit.nativeTransport);
        if (executable.ok())
          eventRealizations = physicalEventRealizationSpecs(
              schedule, **executable, unit.trustedEventCertificates);
      } else {
        FailureOr<PhysicalTaskCoverage> coverage = physicalTaskCoverage(unit);
        if (succeeded(coverage))
          eventRealizations = deriveEventRealizationSpecs(schedule, *coverage);
      }
      if (failed(eventRealizations)) {
        schedule.emitError(
            "cannot lower every exact Event Tensor: the physical compiler did "
            "not provide a source-bound full-relation erasure or physical "
            "mechanism certificate");
        return signalPassFailure();
      }
      builder.setInsertionPointToEnd(module.getBody());
      OperationState eventPlanState(module.getLoc(),
                                    EventLoweringPlanOp::getOperationName());
      eventPlanState.addAttribute("schema_version",
                                  builder.getI64IntegerAttr(1));
      eventPlanState.addAttribute("source_schedule_fingerprint",
                                  schedule.getFingerprintAttr());
      eventPlanState.addAttribute(
          "physical_launch_recipe_fingerprint",
          unit.isFatbin()
              ? builder.getStringAttr(unit.physicalLaunchRecipeFingerprint)
              : device.getFingerprintAttr());
      eventPlanState.addAttribute("fingerprint",
                                  builder.getStringAttr(std::string(64, '0')));
      eventPlanState.addRegion();
      auto eventPlan =
          cast<EventLoweringPlanOp>(builder.create(eventPlanState));
      eventPlan.getBody().push_back(new Block());
      builder.setInsertionPointToEnd(&eventPlan.getBody().front());
      for (auto [eventOrdinal, realization] :
           llvm::enumerate(*eventRealizations))
        createScheduleRecord(
            builder, module.getLoc(), EventRealizationOp::getOperationName(),
            {builder.getNamedAttr("event",
                                  builder.getI64IntegerAttr(eventOrdinal)),
             builder.getNamedAttr(
                 "kind", EventRealizationKindAttr::get(module.getContext(),
                                                       realization.kind)),
             builder.getNamedAttr(
                 "covering_launch_order",
                 DenseI64ArrayAttr::get(module.getContext(),
                                        realization.coveringLaunchOrder))});
      builder.create<EventLoweringPlanYieldOp>(module.getLoc());
      eventPlan.setFingerprint(eventLoweringPlanFingerprint(eventPlan));

      if (unit.isFatbin()) {
        absl::StatusOr<std::shared_ptr<const GpuExecutable>> executable =
            GpuExecutable::Load(unit.nativeTransport);
        if (!executable.ok()) {
          schedule.emitError("cannot read physical Event provenance");
          return signalPassFailure();
        }
        StringRef canonicalSchedule = (*executable)->physicalEventProvenance();
        if (!canonicalSchedule.empty()) {
          StringRef canonicalScheduleDigest =
              (*executable)->physicalEventProvenanceDigest();
          if (!(*executable)->internalEventSchedule()) {
            schedule.emitError(
                "cannot read the typed refined internal Event schedule");
            return signalPassFailure();
          }
          builder.setInsertionPointToEnd(module.getBody());
          if (failed(createInternalEventPlan(
                  builder, module.getLoc(),
                  *(*executable)->internalEventSchedule(),
                  unit.physicalLaunchRecipeFingerprint, canonicalSchedule,
                  canonicalScheduleDigest))) {
            schedule.emitError(
                "cannot materialize the typed refined internal Event plan");
            return signalPassFailure();
          }
        }
      }

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
      if (spec.slots.size() != spec.reuseGroups.size() ||
          spec.slots.size() != spec.bindingIndices.size() ||
          spec.slots.size() != spec.sourceBuffers.size())
        return signalPassFailure();
      for (auto [ordinal, slot] : llvm::enumerate(spec.slots)) {
        ExecutableBindingKind binding = ExecutableBindingKind::None;
        std::optional<int64_t> bindingIndex;
        if (slot.storage == MaterializationStorage::External &&
            slot.access == ExecutableAccess::Read) {
          binding = ExecutableBindingKind::Operand;
          bindingIndex = spec.bindingIndices[ordinal];
        } else if (slot.storage == MaterializationStorage::External &&
                   slot.access == ExecutableAccess::Write) {
          binding = ExecutableBindingKind::Result;
          bindingIndex = spec.bindingIndices[ordinal];
        }
        if ((binding == ExecutableBindingKind::None) != !bindingIndex)
          return signalPassFailure();
        SmallVector<NamedAttribute> attributes{
            builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr(
                "source_buffer",
                builder.getI64IntegerAttr(spec.sourceBuffers[ordinal])),
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
      bundleState.addAttribute("event_lowering_plan_fingerprint",
                               eventPlan.getFingerprintAttr());
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

private:
  std::shared_ptr<const GpuPhysicalArtifactProvider> provider;
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

LogicalResult verifyGpuExecutableUnit(
    ModuleOp module, RegionOp region, MaterializationPlanOp materialization,
    SchedulePlanOp schedule, EventLoweringPlanOp sourceEventPlan,
    EventLoweringPlanOp eventPlan, DeviceModuleOp device, InvocationAbiOp abi,
    ExecutableBundleOp bundle) {
  if (failed(verifyMaterializationPlanAgainstSource(module, materialization)) ||
      failed(schedule.verifyRegions()) ||
      failed(verifySchedulePlanAgainstMaterialization(materialization,
                                                      schedule)) ||
      failed(
          verifyEventLoweringPlanAgainstSchedule(schedule, sourceEventPlan)) ||
      failed(verifyEventLoweringPlanAgainstSchedule(schedule, eventPlan)) ||
      failed(device.verifyRegions()) || failed(abi.verifyRegions()) ||
      failed(bundle.verify()))
    return failure();
  SmallVector<EventTensorOp> events(
      schedule.getBody().front().getOps<EventTensorOp>());
  llvm::SmallDenseMap<int64_t, DeviceEntryOp> entriesBySourceTask;
  for (DeviceEntryOp entry : device.getBody().front().getOps<DeviceEntryOp>())
    if (!entriesBySourceTask.try_emplace(entry.getSourceTask(), entry).second)
      return entry.emitOpError(
          "source task must identify one stream-ordered launch");
  const bool fatbin =
      device.getCodeFormat() == ExecutableCodeFormat::CudaFatbinSm90V1 ||
      device.getCodeFormat() == ExecutableCodeFormat::CudaFatbinSm100V1;
  std::shared_ptr<const GpuExecutable> physicalExecutable;
  PhysicalTaskCoverage physicalCoverage;
  if (fatbin && entriesBySourceTask.empty()) {
    DenseI8ArrayAttr transport = device.getNativeTransportAttr();
    if (!transport)
      return bundle.emitOpError(
          "native artifact requires physical source-task launch provenance");
    ArrayRef<int8_t> signedBytes = transport.asArrayRef();
    ArrayRef<uint8_t> bytes(
        reinterpret_cast<const uint8_t *>(signedBytes.data()),
        signedBytes.size());
    absl::StatusOr<std::shared_ptr<const GpuExecutable>> loaded =
        GpuExecutable::Load(bytes);
    if (!loaded.ok())
      return bundle.emitOpError(
          "native artifact requires valid physical source-task launch "
          "provenance");
    physicalExecutable = std::move(*loaded);
    const int64_t sourceTaskCount =
        llvm::range_size(schedule.getBody().front().getOps<ScheduleTaskOp>());
    for (const GpuLaunch &launch : physicalExecutable->launches()) {
      for (auto [sourceTaskPosition, sourceTask] :
           llvm::enumerate(launch.sourceTaskOrdinals)) {
        if (sourceTask < 0 || sourceTask >= sourceTaskCount)
          return bundle.emitOpError(
              "physical launch references an unknown source task");
        physicalCoverage[sourceTask].push_back(
            {launch.taskOrdinal, static_cast<int64_t>(sourceTaskPosition)});
      }
    }
    for (int64_t sourceTask = 0; sourceTask < sourceTaskCount; ++sourceTask)
      if (!physicalCoverage.count(sourceTask))
        return bundle.emitOpError(
            "physical launch projection does not cover every source task");
  }
  if (!entriesBySourceTask.empty())
    for (auto [sourceTask, entry] : entriesBySourceTask)
      physicalCoverage[sourceTask].push_back(
          {static_cast<int64_t>(entry.getOrdinal()), 0});
  if (fatbin) {
    ArrayAttr snapshotDigests = device.getDeviceIrSnapshotDigestsAttr();
    ArrayRef<GpuDeviceIrSnapshot> snapshots =
        physicalExecutable->deviceIrSnapshots();
    if ((snapshotDigests ? snapshotDigests.size() : 0) != snapshots.size())
      return device.emitOpError(
          "device IR snapshot digests do not equal native transport");
    if (snapshotDigests)
      for (auto [attribute, snapshot] : llvm::zip(snapshotDigests, snapshots)) {
        auto digest = dyn_cast<StringAttr>(attribute);
        if (!digest || digest.getValue() != snapshot.mlirBytecodeDigest)
          return device.emitOpError(
              "device IR snapshot digest does not equal native transport");
      }
  }
  SmallVector<InternalEventPlanOp> internalEventPlans(
      module.getOps<InternalEventPlanOp>());
  if (fatbin) {
    StringRef canonicalSchedule = physicalExecutable->physicalEventProvenance();
    if (canonicalSchedule.empty()) {
      if (!internalEventPlans.empty())
        return internalEventPlans.front().emitOpError(
            "has no matching native physical Event provenance");
    } else {
      if (internalEventPlans.size() != 1)
        return bundle.emitOpError(
            "requires one distinct typed refined internal Event plan");
      InternalEventPlanOp provenance = internalEventPlans.front();
      StringRef canonicalDigest =
          physicalExecutable->physicalEventProvenanceDigest();
      if (!physicalExecutable->internalEventSchedule())
        return provenance.emitOpError(
            "cannot read the native typed refined internal Event schedule");
      StringRef scheduleFingerprint =
          physicalExecutable->internalEventSchedule()
              ->physicalScheduleFingerprint;
      const std::string expectedFingerprint =
          physicalEventProvenanceFingerprint(
              scheduleFingerprint, *device.getPhysicalLaunchRecipeFingerprint(),
              canonicalDigest);
      if (failed(provenance.verify()) ||
          provenance.getPhysicalScheduleFingerprint() != scheduleFingerprint ||
          provenance.getPhysicalLaunchRecipeFingerprint() !=
              device.getPhysicalLaunchRecipeFingerprint() ||
          provenance.getCanonicalSchedule() != canonicalSchedule ||
          provenance.getCanonicalScheduleDigest() != canonicalDigest ||
          provenance.getFingerprint() != expectedFingerprint)
        return provenance.emitOpError(
            "does not equal the native refined internal Event provenance");
      if (failed(verifyInternalEventPlanAgainstNative(
              provenance, *physicalExecutable->internalEventSchedule(),
              *device.getPhysicalLaunchRecipeFingerprint(), canonicalSchedule,
              canonicalDigest)))
        return failure();
    }
  } else if (!internalEventPlans.empty()) {
    return internalEventPlans.front().emitOpError(
        "requires a physically compiled native artifact");
  }
  if (fatbin) {
    ArrayAttr records = device.getTrustedInternalEventCertificatesAttr();
    if (records) {
      if (!physicalExecutable->internalEventSchedule())
        return device.emitOpError(
            "trusted internal Event certificate requires a typed native "
            "internal Event schedule");
      llvm::SmallDenseSet<std::pair<int64_t, int64_t>> covered;
      auto codeIdentity = gpuFatbinCodeIdentity(
          physicalExecutable->codeImages(), physicalExecutable->launches());
      if (!codeIdentity.ok())
        return failure();
      const std::string deviceImageDigest =
          gpuExecutableBundleDigest(*codeIdentity);
      for (Attribute attribute : records) {
        auto record = dyn_cast<DictionaryAttr>(attribute);
        if (!record || record.size() != 5)
          return device.emitOpError(
              "trusted internal Event certificate has invalid fields");
        auto regionAttr = record.getAs<IntegerAttr>("region");
        auto allocationAttr = record.getAs<IntegerAttr>("allocation");
        auto allocationIdAttr = record.getAs<StringAttr>("allocation_id");
        auto compilerDigestAttr =
            record.getAs<StringAttr>("compiler_ir_digest");
        auto deviceDigestAttr = record.getAs<StringAttr>("device_image_digest");
        if (!regionAttr || !allocationAttr || !allocationIdAttr ||
            !compilerDigestAttr || !deviceDigestAttr)
          return device.emitOpError(
              "trusted internal Event certificate has invalid field types");
        const int64_t regionOrdinal = regionAttr.getInt();
        const int64_t allocationOrdinal = allocationAttr.getInt();
        const GpuInternalEventSchedule &schedule =
            *physicalExecutable->internalEventSchedule();
        if (regionOrdinal < 0 ||
            regionOrdinal >= static_cast<int64_t>(schedule.regions.size()))
          return device.emitOpError(
              "trusted internal Event certificate references an unknown "
              "region");
        const GpuInternalEventRegion &region = schedule.regions[regionOrdinal];
        if (allocationOrdinal < 0 ||
            allocationOrdinal >=
                static_cast<int64_t>(region.allocations.size()) ||
            !covered.insert({regionOrdinal, allocationOrdinal}).second)
          return device.emitOpError(
              "trusted internal Event certificate references an unknown or "
              "duplicate allocation");
        const GpuInternalEventAllocation &allocation =
            region.allocations[allocationOrdinal];
        SmallVector<GpuInternalEventOperation> operations =
            internalEventOperations(allocation);
        if (allocationIdAttr.getValue() != allocation.allocationId ||
            operations.empty() ||
            compilerDigestAttr.getValue() !=
                internalEventCertificateDigest(
                    regionOrdinal, allocation.allocationId, operations) ||
            deviceDigestAttr.getValue() != deviceImageDigest)
          return device.emitOpError(
              "trusted internal Event certificate does not equal the native "
              "allocation recipe and device image");
      }
    }
  } else if (device.getTrustedInternalEventCertificatesAttr()) {
    return device.emitOpError(
        "trusted internal Event certificates require a fatbin artifact");
  }
  SmallVector<uint8_t> trustedSourceCertificates;
  if (fatbin) {
    trustedSourceCertificates.assign(
        physicalExecutable->eventRealizations().size(), 0);
    if (ArrayAttr records = device->getAttrOfType<ArrayAttr>(
            "trusted_source_event_certificates")) {
      for (Attribute attribute : records) {
        auto record = cast<DictionaryAttr>(attribute);
        int64_t event = record.getAs<IntegerAttr>("event").getInt();
        StringRef digest =
            record.getAs<StringAttr>("compiler_ir_digest").getValue();
        if (event < 0 ||
            event >= static_cast<int64_t>(
                         physicalExecutable->eventRealizations().size()))
          return device.emitOpError(
              "trusted source Event certificate references an unknown event");
        const auto &certificate =
            physicalExecutable->eventRealizations()[event].operationCertificate;
        if (!certificate || certificate->compilerIrDigest != digest)
          return device.emitOpError("trusted source Event certificate does not "
                                    "equal native transport");
        trustedSourceCertificates[event] = 1;
      }
    }
  }
  FailureOr<SmallVector<EventRealizationSpec>> expectedRealizations =
      fatbin ? physicalEventRealizationSpecs(schedule, *physicalExecutable,
                                             trustedSourceCertificates)
             : deriveEventRealizationSpecs(schedule, physicalCoverage);
  SmallVector<EventRealizationOp> actualRealizations(
      eventPlan.getBody().front().getOps<EventRealizationOp>());
  if (failed(expectedRealizations) ||
      actualRealizations.size() != expectedRealizations->size())
    return eventPlan.emitOpError(
        "must realize every exact Event Tensor against physical launch order");
  for (auto [ordinal, actual, expected] :
       llvm::enumerate(actualRealizations, *expectedRealizations))
    if (actual.getEvent() != static_cast<int64_t>(ordinal) ||
        actual.getKind() != expected.kind ||
        actual.getCoveringLaunchOrder() !=
            ArrayRef<int64_t>(expected.coveringLaunchOrder))
      return actual.emitOpError(
          "does not equal the source-bound physical Event realization");
  std::optional<StringRef> expectedPhysicalFingerprint =
      fatbin ? device.getPhysicalLaunchRecipeFingerprint()
             : std::optional<StringRef>(device.getFingerprint());
  if (eventPlan.getSourceScheduleFingerprint() != schedule.getFingerprint() ||
      eventPlan.getPhysicalLaunchRecipeFingerprint() !=
          *expectedPhysicalFingerprint ||
      bundle.getEventLoweringPlanFingerprint() != eventPlan.getFingerprint())
    return eventPlan.emitOpError(
        "must bind the source schedule, physical launch recipe, and executable "
        "bundle");
  if (fatbin) {
    FailureOr<GpuTarget> target = targetForCodeFormat(device.getCodeFormat());
    if (failed(target) ||
        device.getSchemaVersion() != fatbinDeviceSchemaForTarget(*target) ||
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
        !device.getPhysicalRequestFingerprintAttr() ||
        device.getPhysicalRequestFingerprint() !=
            gpuPhysicalArtifactRequestFingerprint(
                region, materialization, schedule, sourceEventPlan, *target))
      return failure();
    FailureOr<SmallVector<ExecutableSlotSpec>> expectedSlots =
        deriveExecutableSlots(materialization);
    if (failed(expectedSlots))
      return failure();
    auto manifest =
        module->getAttrOfType<DictionaryAttr>(kCoverageManifestAttribute);
    auto tuning = manifest ? manifest.getAs<StringAttr>("canonical_tuning")
                           : StringAttr{};
    FailureOr<DerivedPlan> derived = deriveGpuPlan(
        region, tuning && tuning.getValue() == kFusionCanonicalTuning);
    if (failed(derived))
      return failure();
    FailureOr<SmallVector<std::optional<int64_t>>> expectedBindings =
        deriveGpuBindingIndices(region, *derived, *expectedSlots);
    SmallVector<InvocationSlotOp> actualSlots(
        abi.getBody().front().getOps<InvocationSlotOp>());
    if (failed(expectedBindings))
      return failure();
    SmallVector<bool> projectedSourceBuffers(expectedSlots->size(), false);
    for (InvocationSlotOp actual : actualSlots) {
      const int64_t sourceOrdinal = actual.getSourceBuffer();
      if (sourceOrdinal == -1) {
        if (actual.getStorage() != MaterializationStorage::Temporary ||
            actual.getAccess() != ExecutableAccess::ReadWrite ||
            actual.getAddressSpace() != ExecutableAddressSpace::Device ||
            actual.getBinding() != ExecutableBindingKind::None ||
            actual.getBindingIndex())
          return failure();
        continue;
      }
      if (sourceOrdinal < 0 ||
          sourceOrdinal >= static_cast<int64_t>(expectedSlots->size()) ||
          projectedSourceBuffers[sourceOrdinal])
        return failure();
      const ExecutableSlotSpec &expected = (*expectedSlots)[sourceOrdinal];
      if (expected.storage != MaterializationStorage::External)
        return failure();
      ExecutableBindingKind binding = ExecutableBindingKind::None;
      std::optional<int64_t> bindingIndex;
      if (expected.storage == MaterializationStorage::External &&
          expected.access == ExecutableAccess::Read) {
        binding = ExecutableBindingKind::Operand;
        bindingIndex = (*expectedBindings)[sourceOrdinal];
      } else if (expected.storage == MaterializationStorage::External &&
                 expected.access == ExecutableAccess::Write) {
        binding = ExecutableBindingKind::Result;
        bindingIndex = (*expectedBindings)[sourceOrdinal];
      }
      if (actual.getTensorType() != expected.tensorType ||
          actual.getRequiredBytes() != expected.requiredBytes ||
          actual.getStrides() != ArrayRef<int64_t>(expected.strides) ||
          actual.getOffset() != 0 ||
          actual.getAlignment() < expected.alignment ||
          actual.getAddressSpace() != ExecutableAddressSpace::Device ||
          actual.getAccess() != expected.access ||
          actual.getStorage() != expected.storage ||
          actual.getAliasGroup() != actual.getOrdinal() ||
          actual.getReuseGroup() != actual.getOrdinal() ||
          actual.getBinding() != binding ||
          actual.getBindingIndex() != bindingIndex)
        return failure();
      projectedSourceBuffers[sourceOrdinal] = true;
    }
    for (auto [sourceOrdinal, expected] : llvm::enumerate(*expectedSlots)) {
      if (expected.storage == MaterializationStorage::External &&
          !projectedSourceBuffers[sourceOrdinal])
        return failure();
      if (expected.storage != MaterializationStorage::External &&
          expected.storage != MaterializationStorage::Temporary)
        return failure();
    }
    OpBuilder builder(module.getContext());
    OwningOpRef<ModuleOp> stripped = ModuleOp::create(module.getLoc());
    builder.setInsertionPointToEnd(stripped->getBody());
    builder.insert(eventPlan->clone());
    builder.insert(device->clone());
    builder.insert(abi->clone());
    builder.insert(bundle->clone());
    return succeeded(serializeGpuExecutableBundle(*stripped)) ? success()
                                                              : failure();
  }
  FailureOr<GpuExecutableSpec> expected =
      [&]() -> FailureOr<GpuExecutableSpec> {
    FailureOr<GpuTarget> target = targetForCodeFormat(device.getCodeFormat());
    if (failed(target))
      return failure();
    return deriveGpuExecutable(region, materialization, schedule, *target);
  }();
  FailureOr<GpuTarget> target = targetForCodeFormat(device.getCodeFormat());
  if (failed(expected) || failed(target) ||
      device.getSchemaVersion() != deviceSchemaForTarget(*target) ||
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
  for (auto [ordinal, actual, spec] : llvm::enumerate(slots, expected->slots)) {
    ExecutableBindingKind binding = ExecutableBindingKind::None;
    std::optional<int64_t> bindingIndex;
    if (spec.storage == MaterializationStorage::External &&
        spec.access == ExecutableAccess::Read) {
      binding = ExecutableBindingKind::Operand;
      bindingIndex = expected->bindingIndices[ordinal];
    } else if (spec.storage == MaterializationStorage::External &&
               spec.access == ExecutableAccess::Write) {
      binding = ExecutableBindingKind::Result;
      bindingIndex = expected->bindingIndices[ordinal];
    }
    if (actual.getOrdinal() != static_cast<int64_t>(ordinal) ||
        actual.getSourceBuffer() != expected->sourceBuffers[ordinal] ||
        actual.getTensorType() != spec.tensorType ||
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
  SmallVector<EventLoweringPlanOp> eventPlans(
      module.getOps<EventLoweringPlanOp>());
  SmallVector<DeviceModuleOp> devices(module.getOps<DeviceModuleOp>());
  SmallVector<InvocationAbiOp> abis(module.getOps<InvocationAbiOp>());
  SmallVector<ExecutableBundleOp> bundles(module.getOps<ExecutableBundleOp>());
  if (materializations.empty() || materializations.size() != schedules.size() ||
      materializations.size() != devices.size() ||
      materializations.size() * 2 != eventPlans.size() ||
      materializations.size() != abis.size() ||
      materializations.size() != bundles.size())
    return failure();

  llvm::SmallDenseSet<Operation *> regions;
  llvm::SmallDenseSet<Operation *> matchedSchedules;
  llvm::SmallDenseSet<Operation *> matchedDevices;
  llvm::SmallDenseSet<Operation *> matchedEventPlans;
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

    EventLoweringPlanOp sourceEventPlan;
    EventLoweringPlanOp eventPlan;
    for (EventLoweringPlanOp candidate : eventPlans) {
      if (candidate.getSourceScheduleFingerprint() != schedule.getFingerprint())
        continue;
      if (candidate.getPhysicalLaunchRecipeFingerprint()) {
        if (eventPlan)
          return failure();
        eventPlan = candidate;
      } else {
        if (sourceEventPlan)
          return failure();
        sourceEventPlan = candidate;
      }
    }
    if (!sourceEventPlan || !eventPlan ||
        !matchedEventPlans.insert(sourceEventPlan.getOperation()).second ||
        !matchedEventPlans.insert(eventPlan.getOperation()).second)
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
                                       schedule, sourceEventPlan, eventPlan,
                                       device, abi, bundle)))
      return failure();
    if (device.getCodeFormat() == ExecutableCodeFormat::CudaFatbinSm90V1 ||
        device.getCodeFormat() == ExecutableCodeFormat::CudaFatbinSm100V1) {
      DenseI8ArrayAttr transport = device.getNativeTransportAttr();
      if (!transport)
        return failure();
      ArrayRef<int8_t> bytes = transport.asArrayRef();
      auto fatbin = deserializeGpuFatbinExecutableBundle(
          ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()),
                            bytes.size()),
          module.getContext());
      auto target = targetForCodeFormat(device.getCodeFormat());
      if (!fatbin.ok() || failed(target))
        return failure();
      auto spec = deriveFatbinBoundary(region, materialization, schedule,
                                       *fatbin, *target);
      if (failed(spec))
        return failure();
      units.push_back(
          {region, materialization, schedule, sourceEventPlan, std::move(*spec),
           SmallVector<uint8_t>(
               reinterpret_cast<const uint8_t *>(bytes.data()),
               reinterpret_cast<const uint8_t *>(bytes.data()) + bytes.size()),
           device.getPhysicalRequestFingerprint()->str(),
           device.getPhysicalCompileRequestFingerprint()->str(),
           device.getPhysicalLaunchRecipeFingerprint()->str()});
    } else {
      FailureOr<GpuTarget> target = targetForCodeFormat(device.getCodeFormat());
      if (failed(target))
        return failure();
      FailureOr<GpuExecutableSpec> spec =
          deriveGpuExecutable(region, materialization, schedule, *target);
      if (failed(spec))
        return failure();
      units.push_back({region,
                       materialization,
                       schedule,
                       sourceEventPlan,
                       std::move(*spec),
                       {},
                       {},
                       {},
                       {}});
    }
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
  int64_t transportSchemaVersion;
  int64_t deviceSchemaVersion;
  SmallVector<uint8_t> bundleBytes;
  std::string digest;
  SmallVector<Attribute> operandLayouts;
  SmallVector<Type> resultTypes;
  SmallVector<Attribute> resultLayouts;
  SmallVector<int64_t> regionResultBindings;
};

FailureOr<PreparedGpuCall> prepareGpuCall(ModuleOp module, RegionOp region,
                                          DeviceModuleOp device,
                                          InvocationAbiOp abi,
                                          ExecutableBundleOp bundle) {
  if (device.getPolicy() != region.getPolicy())
    return failure();
  auto closedExternalType = [](Value value) {
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor || !tensor.hasStaticShape() || tensor.getRank() > 8)
      return false;
    return tensor.getElementType().isF32() ||
           (tensor.getElementType().isBF16() && tensor.getRank() >= 1);
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
  SmallVector<int64_t> regionResultBindings =
      publicRegionResultBindings(region);
  for (auto [regionIndex, bindingIndex] :
       llvm::enumerate(regionResultBindings)) {
    InvocationSlotOp slot = resultSlots[bindingIndex];
    if (!slot ||
        slot.getTensorType() != region.getResults()[regionIndex].getType())
      return failure();
  }
  EventLoweringPlanOp eventPlan;
  if (bundle.getEventLoweringPlanFingerprint()) {
    for (EventLoweringPlanOp candidate : module.getOps<EventLoweringPlanOp>()) {
      if (candidate.getFingerprint() !=
          *bundle.getEventLoweringPlanFingerprint())
        continue;
      if (eventPlan)
        return failure();
      eventPlan = candidate;
    }
    if (!eventPlan)
      return failure();
  }
  OpBuilder builder(module.getContext());
  OwningOpRef<ModuleOp> stripped = ModuleOp::create(module.getLoc());
  builder.setInsertionPointToEnd(stripped->getBody());
  if (eventPlan)
    builder.insert(eventPlan->clone());
  builder.insert(device->clone());
  builder.insert(abi->clone());
  builder.insert(bundle->clone());
  FailureOr<SmallVector<uint8_t>> bundleBytes =
      serializeGpuExecutableBundle(*stripped);
  if (failed(bundleBytes))
    return failure();
  PreparedGpuCall prepared{region,
                           device.getSchemaVersion() >= 5 ? 3 : 2,
                           static_cast<int64_t>(device.getSchemaVersion()),
                           std::move(*bundleBytes),
                           {},
                           {},
                           {},
                           {},
                           {}};
  prepared.digest = gpuExecutableBundleDigest(prepared.bundleBytes);
  for (Value input : region.getInputs())
    prepared.operandLayouts.push_back(
        rowMajorLayout(cast<RankedTensorType>(input.getType())));
  prepared.resultTypes.resize(region.getNumResults());
  prepared.resultLayouts.resize(region.getNumResults());
  prepared.regionResultBindings = std::move(regionResultBindings);
  for (auto [regionIndex, bindingIndex] :
       llvm::enumerate(prepared.regionResultBindings)) {
    Type type = region.getResults()[regionIndex].getType();
    prepared.resultTypes[bindingIndex] = type;
    prepared.resultLayouts[bindingIndex] =
        rowMajorLayout(cast<RankedTensorType>(type));
  }
  return prepared;
}

LogicalResult verifyInternalEventPlansCanErase(ModuleOp module) {
  for (InternalEventPlanOp plan : module.getOps<InternalEventPlanOp>()) {
    DeviceModuleOp device;
    for (DeviceModuleOp candidate : module.getOps<DeviceModuleOp>()) {
      if (!candidate.getPhysicalLaunchRecipeFingerprint() ||
          *candidate.getPhysicalLaunchRecipeFingerprint() !=
              plan.getPhysicalLaunchRecipeFingerprint())
        continue;
      if (device)
        return plan.emitOpError(
            "cannot erase a refined internal Event plan bound to multiple "
            "physical artifacts");
      device = candidate;
    }
    if (!device || !device.getNativeTransportAttr())
      return plan.emitOpError(
          "cannot erase a refined internal Event plan without its bound "
          "native physical artifact");

    llvm::SmallDenseSet<std::pair<int64_t, int64_t>> certifiedAllocations;
    if (ArrayAttr certificates =
            device.getTrustedInternalEventCertificatesAttr())
      for (Attribute attribute : certificates) {
        auto certificate = cast<DictionaryAttr>(attribute);
        certifiedAllocations.insert(
            {certificate.getAs<IntegerAttr>("region").getInt(),
             certificate.getAs<IntegerAttr>("allocation").getInt()});
      }
    for (InternalEventAllocationOp allocation :
         plan.getBody().front().getOps<InternalEventAllocationOp>())
      if (!certifiedAllocations.contains(
              {allocation.getRegion(), allocation.getOrdinal()}))
        return allocation.emitOpError(
            "cannot erase a physical internal Event allocation without a "
            "trusted compiler/device legalization certificate");

    for (InternalEventRealizationOp realization :
         plan.getBody().front().getOps<InternalEventRealizationOp>()) {
      StringRef kind = realization.getKind();
      if (kind == "physical") {
        if (realization.getAllocations().empty())
          return realization.emitOpError(
              "cannot erase a physical internal Event realization without a "
              "certified allocation");
        continue;
      }
      if (kind != "erased_program_order" && kind != "erased_stream_order" &&
          kind != "erased_algebraic_refinement")
        return realization.emitOpError(
            "cannot erase an unknown internal Event realization");
      if (!realization.getAllocations().empty())
        return realization.emitOpError(
            "an erased internal Event realization cannot retain a physical "
            "allocation");
    }
  }
  return success();
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
    if (failed(verifyInternalEventPlansCanErase(module)))
      return signalPassFailure();
    SmallVector<MaterializationPlanOp> materializations(
        module.getOps<MaterializationPlanOp>());
    SmallVector<SchedulePlanOp> schedules(module.getOps<SchedulePlanOp>());
    SmallVector<DeviceModuleOp> devices(module.getOps<DeviceModuleOp>());
    SmallVector<InvocationAbiOp> abis(module.getOps<InvocationAbiOp>());
    SmallVector<ExecutableBundleOp> bundles(
        module.getOps<ExecutableBundleOp>());
    SmallVector<PreparedGpuCall, 1> preparedCalls;
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
          builder.getNamedAttr(
              "transport_schema_version",
              builder.getI64IntegerAttr(prepared.transportSchemaVersion)),
          builder.getNamedAttr(
              "device_schema_version",
              builder.getI64IntegerAttr(prepared.deviceSchemaVersion)),
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
      callState.addTypes(prepared.resultTypes);
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
      for (auto [regionIndex, bindingIndex] :
           llvm::enumerate(prepared.regionResultBindings))
        region.getResults()[regionIndex].replaceAllUsesWith(
            call->getResult(bindingIndex));
      region.erase();
    }
    for (Operation &operation :
         llvm::make_early_inc_range(module.getBody()->without_terminator())) {
      if (isa<MaterializationPlanOp, SchedulePlanOp, EventLoweringPlanOp,
              InternalEventPlanOp, DeviceModuleOp, InvocationAbiOp,
              ExecutableBundleOp>(operation))
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
  uint64_t bundleLaunchCount = 0;
  uint64_t bundleTemporaryCount = 0;
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
  if (phase == ShuttlePipelinePhase::GeneratedBundle) {
    for (DeviceModuleOp device : module.getOps<DeviceModuleOp>())
      bundleLaunchCount += static_cast<uint64_t>(
          llvm::range_size(device.getBody().front().getOps<DeviceEntryOp>()));
    for (MaterializationPlanOp plan : module.getOps<MaterializationPlanOp>()) {
      for (MaterializationBufferOp buffer :
           plan.getBody().front().getOps<MaterializationBufferOp>()) {
        if (buffer.getStorage() == MaterializationStorage::Temporary)
          ++bundleTemporaryCount;
      }
    }
  }
  invocation->emit(phase, std::move(regionMembership),
                   std::move(coverageManifest),
                   std::move(unsupportedFingerprint),
                   phase == ShuttlePipelinePhase::GeneratedBundle
                       ? generatedGpuBundleFingerprintImpl(module)
                   : phase == ShuttlePipelinePhase::FinalErasure
                       ? normalizedModuleFingerprintImpl(module)
                       : std::string{},
                   phase == ShuttlePipelinePhase::FinalErasure,
                   failurePass.str(), bundleLaunchCount, bundleTemporaryCount);
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
  manager.addPass(createPlanSourceEventLoweringPass());
  manager.addPass(createBuildGpuExecutableBundlePass(
      options.gpuTarget, gpuPhysicalArtifactProvider()));
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
    const bool canonicalSm100 = StringRef(options.canonicalOptions)
                                    .contains("\"gpu_target\":\"sm_100\"");
    const bool canonicalSm90 = StringRef(options.canonicalOptions)
                                   .contains("\"gpu_target\":\"sm_90\"");
    const bool targetedAbi = StringRef(options.canonicalOptions)
                                 .contains("\"pipeline_abi_version\":23");
    if ((options.gpuTarget == GpuTarget::Sm100) != canonicalSm100 ||
        (canonicalSm90 || canonicalSm100) != targetedAbi) {
      module.emitError(
          "GPU target does not match the canonical compiler identity");
      return signalPassFailure();
    }
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

std::unique_ptr<Pass> createPlanSourceEventLoweringPass() {
  return std::make_unique<PlanSourceEventLoweringPass>();
}

std::unique_ptr<Pass> createBuildGpuExecutableBundlePass() {
  return std::make_unique<BuildGpuExecutableBundlePass>();
}

std::unique_ptr<Pass> createBuildGpuExecutableBundlePass(GpuTarget target) {
  return std::make_unique<BuildGpuExecutableBundlePass>(target);
}

std::unique_ptr<Pass> createBuildGpuExecutableBundlePass(
    GpuTarget target,
    std::shared_ptr<const GpuPhysicalArtifactProvider> provider) {
  return std::make_unique<BuildGpuExecutableBundlePass>(target,
                                                        std::move(provider));
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
