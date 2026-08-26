// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/IR/ShuttleDialect.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include "shuttle/IR/ShuttleEnums.cc.inc"

#define GET_ATTRDEF_CLASSES
#include "shuttle/IR/ShuttleAttrs.cc.inc"

#define GET_OP_CLASSES
#include "shuttle/IR/ShuttleOps.cc.inc"

#include "shuttle/IR/ShuttleDialect.cc.inc"

namespace mlir::shuttle {
namespace {

LogicalResult verifySingleBlockRegion(Operation *owner, Region &region) {
  if (region.empty()) {
    return owner->emitOpError("requires a non-empty body");
  }
  if (!llvm::hasSingleElement(region)) {
    return owner->emitOpError("requires exactly one body block");
  }
  if (!isa<YieldOp>(region.front().getTerminator())) {
    return owner->emitOpError("body must terminate with shuttle.yield");
  }
  return success();
}

bool isLowerHexDigest(StringRef value) {
  return value.size() == 64 && llvm::all_of(value, [](char character) {
           return llvm::isDigit(character) ||
                  (character >= 'a' && character <= 'f');
         });
}

std::optional<int64_t> optionalIntegerAttribute(Operation *operation,
                                                StringRef name) {
  if (auto attribute = operation->getAttrOfType<IntegerAttr>(name)) {
    return attribute.getInt();
  }
  return std::nullopt;
}

SmallVector<int64_t> integerArray(DenseI64ArrayAttr attribute) {
  return SmallVector<int64_t>(attribute.asArrayRef());
}

LogicalResult verifyStringArray(Operation *owner, ArrayAttr values,
                                StringRef name,
                                ArrayRef<StringRef> allowedValues = {}) {
  for (Attribute value : values) {
    auto stringValue = dyn_cast<StringAttr>(value);
    if (!stringValue) {
      return owner->emitOpError() << name << " must contain only strings";
    }
    if (!allowedValues.empty() &&
        !llvm::is_contained(allowedValues, stringValue.getValue())) {
      return owner->emitOpError() << name << " contains unsupported value '"
                                  << stringValue.getValue() << "'";
    }
  }
  return success();
}

Type scalarType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    return shaped.getElementType();
  }
  return type;
}

bool isScalarNumericType(Type type) {
  return isa<IntegerType, FloatType, ComplexType>(type);
}

bool haveCompatibleShapes(RankedTensorType lhs, RankedTensorType rhs) {
  if (lhs.getRank() != rhs.getRank()) {
    return false;
  }
  for (auto [lhsSize, rhsSize] :
       llvm::zip_equal(lhs.getShape(), rhs.getShape())) {
    if (!ShapedType::isDynamic(lhsSize) && !ShapedType::isDynamic(rhsSize) &&
        lhsSize != rhsSize) {
      return false;
    }
  }
  return true;
}

LogicalResult verifyPureScalarComputation(Operation *owner, Region &region) {
  for (Operation &operation : region.front()) {
    if (operation.getNumRegions() != 0) {
      return owner->emitOpError(
          "scalar body operations must not contain nested regions");
    }
    if (llvm::any_of(operation.getOperandTypes(),
                     [](Type type) { return isa<ShapedType>(type); }) ||
        llvm::any_of(operation.getResultTypes(),
                     [](Type type) { return isa<ShapedType>(type); })) {
      return owner->emitOpError(
          "scalar body operations must not use shaped values");
    }
    if (!isMemoryEffectFree(&operation)) {
      return owner->emitOpError(
          "scalar body operations must have proven no memory effects");
    }
  }
  return success();
}

LogicalResult verifyScalarBody(Operation *owner, Region &region,
                               ValueRange inputs, TypeRange results) {
  if (failed(verifySingleBlockRegion(owner, region))) {
    return failure();
  }
  Block &block = region.front();
  if (block.getNumArguments() != inputs.size()) {
    return owner->emitOpError(
        "scalar body argument count must equal input count");
  }
  for (auto [argument, input] : llvm::zip_equal(block.getArguments(), inputs)) {
    if (argument.getType() != scalarType(input.getType())) {
      return owner->emitOpError(
          "scalar body argument types must equal input element types");
    }
  }
  auto yield = cast<YieldOp>(block.getTerminator());
  if (yield.getValues().size() != results.size()) {
    return owner->emitOpError(
        "scalar body yield count must equal result count");
  }
  for (auto [yielded, resultType] :
       llvm::zip_equal(yield.getValues(), results)) {
    if (yielded.getType() != scalarType(resultType)) {
      return owner->emitOpError(
          "scalar body yield types must equal result element types");
    }
  }
  return verifyPureScalarComputation(owner, region);
}

LogicalResult verifyIndexingMaps(Operation *owner, ArrayAttr indexingMaps,
                                 TypeRange indexedTypes) {
  if (indexingMaps.size() != indexedTypes.size()) {
    return owner->emitOpError() << "requires " << indexedTypes.size()
                                << " indexing maps, one per input and result";
  }
  if (indexingMaps.empty()) {
    return success();
  }
  AffineMap first = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  for (auto [mapAttribute, indexedType] :
       llvm::zip_equal(indexingMaps, indexedTypes)) {
    AffineMap map = cast<AffineMapAttr>(mapAttribute).getValue();
    if (map.getNumDims() != first.getNumDims() ||
        map.getNumSymbols() != first.getNumSymbols()) {
      return owner->emitOpError(
          "indexing maps must share one domain and symbol space");
    }
    if (auto shaped = dyn_cast<ShapedType>(indexedType);
        shaped && shaped.hasRank() && map.getNumResults() != shaped.getRank()) {
      return owner->emitOpError(
          "indexing map result count must equal the indexed value rank");
    }
  }
  return success();
}

LogicalResult verifyMapIndexingMaps(MapOp map, TypeRange indexedTypes) {
  ArrayAttr indexingMaps = map.getIndexingMaps();
  if (failed(verifyIndexingMaps(map, indexingMaps, indexedTypes))) {
    return failure();
  }
  if (indexingMaps.empty()) {
    return map.emitOpError("requires at least one indexing map");
  }

  AffineMap domain = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  SmallVector<int64_t> domainExtents(domain.getNumDims(), ShapedType::kDynamic);
  SmallVector<char> boundDimensions(domain.getNumDims(), 0);
  SmallVector<std::pair<unsigned, int64_t>> boundedZeroDivisors;
  bool hasRankedTensor = false;
  bool resultMapProjectsDomain = false;
  size_t mapPosition = 0;
  for (auto [mapAttribute, indexedType] :
       llvm::zip_equal(indexingMaps, indexedTypes)) {
    AffineMap indexingMap = cast<AffineMapAttr>(mapAttribute).getValue();
    const bool isResultMap = mapPosition++ >= map.getInputs().size();
    if (indexingMap.getNumSymbols() != 0) {
      return map.emitOpError(
          "map indexing maps must not contain affine symbols");
    }
    llvm::SmallDenseSet<unsigned> seenDimensions;
    if (isResultMap &&
        indexingMap.getNumResults() != indexingMap.getNumDims()) {
      resultMapProjectsDomain = true;
    }
    auto tensorType = dyn_cast<RankedTensorType>(indexedType);
    if (!tensorType) {
      if (isa<ShapedType>(indexedType)) {
        return map.emitOpError(
            "map inputs and results must be scalars or ranked tensors");
      }
      if (indexingMap.getNumResults() != 0) {
        return map.emitOpError(
            "scalar inputs and results require zero-result indexing maps");
      }
      continue;
    }

    hasRankedTensor = true;
    for (auto [resultPosition, expression] :
         llvm::enumerate(indexingMap.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      bool boundedZero = false;
      if (!dimension) {
        auto constant = dyn_cast<AffineConstantExpr>(expression);
        if (constant) {
          if (isResultMap || map.getSemantics() != MapSemantics::Reshape ||
              constant.getValue() != 0 ||
              tensorType.getDimSize(resultPosition) != 1) {
            return map.emitOpError(
                "constant-zero input indexing is reserved for typed static "
                "singleton reshapes");
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
        auto divisor =
            dyn_cast_if_present<AffineConstantExpr>(divisorExpression);
        if (isResultMap || map.getSemantics() != MapSemantics::BroadcastInDim ||
            !dividend || !divisor || divisor.getValue() <= 1 ||
            tensorType.getDimSize(resultPosition) != 1) {
          return map.emitOpError(
              "bounded-zero floordiv input indexing is reserved for typed "
              "static singleton broadcasts");
        }
        dimension = dividend;
        boundedZero = true;
        boundedZeroDivisors.emplace_back(dimension.getPosition(),
                                         divisor.getValue());
      }
      const unsigned domainPosition = dimension.getPosition();
      if (!seenDimensions.insert(domainPosition).second) {
        return map.emitOpError(
            "map indexing maps must use each direct domain dimension at "
            "most once");
      }
      boundDimensions[domainPosition] = 1;
      if (boundedZero) {
        continue;
      }
      const int64_t extent = tensorType.getDimSize(resultPosition);
      if (ShapedType::isDynamic(extent)) {
        continue;
      }
      int64_t &knownExtent = domainExtents[domainPosition];
      if (!ShapedType::isDynamic(knownExtent) && knownExtent != extent) {
        return map.emitOpError(
            "map indexing maps bind one domain dimension to inconsistent "
            "static extents");
      }
      knownExtent = extent;
    }
  }
  if (!hasRankedTensor && domain.getNumDims() != 0) {
    return map.emitOpError(
        "a scalar-only map requires a zero-dimensional indexing domain");
  }
  if (hasRankedTensor && llvm::is_contained(boundDimensions, 0)) {
    return map.emitOpError(
        "every map domain dimension must be bound by a ranked tensor "
        "dimension");
  }
  for (auto [domainPosition, divisor] : boundedZeroDivisors) {
    if (ShapedType::isDynamic(domainExtents[domainPosition]) ||
        domainExtents[domainPosition] != divisor) {
      return map.emitOpError(
          "bounded-zero broadcast divisor must equal the static domain "
          "extent");
    }
  }
  if (resultMapProjectsDomain) {
    return map.emitOpError(
        "map result indexing maps must cover every domain dimension "
        "exactly once");
  }
  return success();
}

LogicalResult verifyContractMaps(ContractOp contract, TypeRange indexedTypes) {
  ArrayAttr indexingMaps = contract.getIndexingMaps();
  if (failed(verifyIndexingMaps(contract, indexingMaps, indexedTypes))) {
    return failure();
  }
  if (indexingMaps.empty()) {
    return contract.emitOpError("requires non-empty indexing maps");
  }

  AffineMap domain = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  SmallVector<int64_t> domainExtents(domain.getNumDims(), ShapedType::kDynamic);
  SmallVector<char> boundDimensions(domain.getNumDims(), 0);
  for (auto [mapAttribute, indexedTypeValue] :
       llvm::zip_equal(indexingMaps, indexedTypes)) {
    auto indexedType = cast<RankedTensorType>(indexedTypeValue);
    AffineMap map = cast<AffineMapAttr>(mapAttribute).getValue();
    if (map.getNumSymbols() != 0) {
      return contract.emitOpError(
          "contraction indexing maps must not contain affine symbols");
    }
    if (!map.isProjectedPermutation()) {
      return contract.emitOpError(
          "contraction indexing maps must be projected permutations of "
          "direct domain dimensions");
    }
    for (auto [resultPosition, expression] :
         llvm::enumerate(map.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension) {
        return contract.emitOpError(
            "contraction indexing maps must contain only direct dimensions");
      }
      const unsigned domainPosition = dimension.getPosition();
      boundDimensions[domainPosition] = 1;
      const int64_t extent = indexedType.getDimSize(resultPosition);
      if (ShapedType::isDynamic(extent)) {
        continue;
      }
      int64_t &knownExtent = domainExtents[domainPosition];
      if (!ShapedType::isDynamic(knownExtent) && knownExtent != extent) {
        return contract.emitOpError(
            "contraction indexing maps bind one domain dimension to "
            "inconsistent static extents");
      }
      knownExtent = extent;
    }
  }
  if (llvm::is_contained(boundDimensions, 0)) {
    return contract.emitOpError(
        "every contraction domain dimension must be bound by an indexing map");
  }
  return success();
}

bool mapContainsDimension(AffineMap map, unsigned dimension) {
  return llvm::any_of(map.getResults(), [dimension](AffineExpr expression) {
    auto dimExpression = dyn_cast<AffineDimExpr>(expression);
    return dimExpression && dimExpression.getPosition() == dimension;
  });
}

LogicalResult verifyDotGeneralIterators(ContractOp contract) {
  ArrayAttr indexingMaps = contract.getIndexingMaps();
  AffineMap lhsMap = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  AffineMap rhsMap = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  AffineMap resultMap = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  size_t reductionCount = 0;
  for (auto [dimension, iteratorAttribute] :
       llvm::enumerate(contract.getIteratorKinds())) {
    StringRef iterator = cast<StringAttr>(iteratorAttribute).getValue();
    const bool inLhs = mapContainsDimension(lhsMap, dimension);
    const bool inRhs = mapContainsDimension(rhsMap, dimension);
    const bool inResult = mapContainsDimension(resultMap, dimension);
    if (iterator == "reduction") {
      ++reductionCount;
      if (!inLhs || !inRhs || inResult) {
        return contract.emitOpError(
            "each reduction dimension must appear in both input maps and not "
            "in the result map");
      }
      continue;
    }
    if ((!inLhs && !inRhs) || !inResult) {
      return contract.emitOpError(
          "each parallel dimension must appear in an input map and exactly "
          "once in the result map");
    }
  }
  if (reductionCount == 0) {
    return contract.emitOpError(
        "the 'dot_general' algorithm requires at least one reduction iterator");
  }
  return success();
}

LogicalResult verifyDotGeneralElementTypes(ContractOp contract) {
  Type lhsElement = cast<RankedTensorType>(contract.getInputs()[0].getType())
                        .getElementType();
  Type rhsElement = cast<RankedTensorType>(contract.getInputs()[1].getType())
                        .getElementType();
  Type resultElement =
      cast<RankedTensorType>(contract.getResults()[0].getType())
          .getElementType();
  Type accumulator =
      cast<TypeAttr>(contract.getAccumulatorTypes()[0]).getValue();

  if (lhsElement != rhsElement) {
    return contract.emitOpError(
        "dot_general lhs and rhs element types must match");
  }
  if (!lhsElement.isF32() && !lhsElement.isBF16()) {
    return contract.emitOpError(
        "dot_general supports only bf16 or f32 operand elements");
  }
  if (!accumulator.isF32()) {
    return contract.emitOpError("dot_general requires an f32 accumulator");
  }
  if (!resultElement.isF32()) {
    return contract.emitOpError("dot_general requires f32 result elements");
  }
  return success();
}

} // namespace

void ShuttleDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "shuttle/IR/ShuttleAttrs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "shuttle/IR/ShuttleOps.cc.inc"
      >();
}

LogicalResult RegionOp::verifyRegions() {
  if (failed(verifySingleBlockRegion(*this, getBody()))) {
    return failure();
  }
  Block &block = getBody().front();
  if (block.getNumArguments() != getInputs().size()) {
    return emitOpError("body argument count must equal input count");
  }
  for (auto [argument, input] :
       llvm::zip_equal(block.getArguments(), getInputs())) {
    if (argument.getType() != input.getType()) {
      return emitOpError("body argument types must equal input types");
    }
  }
  auto yield = cast<YieldOp>(block.getTerminator());
  if (!llvm::equal(yield.getValues().getTypes(), getResults().getTypes())) {
    return emitOpError("yielded types must equal result types");
  }
  for (Attribute source : getSourceRefs()) {
    if (!isa<SourceRefAttr>(source)) {
      return emitOpError(
          "source_refs must contain only #shuttle.source_ref attributes");
    }
  }
  if (getSourceRefs().empty()) {
    return emitOpError("requires at least one declared source reference");
  }
  return success();
}

LogicalResult MapOp::verifyRegions() {
  if (failed(verifyScalarBody(*this, getBody(), getInputs(),
                              getResults().getTypes()))) {
    return failure();
  }
  if (getSemantics() != MapSemantics::Pointwise) {
    Block &body = getBody().front();
    auto yield = cast<YieldOp>(body.getTerminator());
    if (getInputs().size() != 1 || getResults().size() != 1 ||
        !body.without_terminator().empty() || yield.getValues().size() != 1 ||
        yield.getValues()[0] != body.getArgument(0)) {
      return emitOpError(
          "structural semantics require one input, one result, and a direct "
          "scalar identity body");
    }
  }
  SmallVector<Type> indexedTypes;
  llvm::append_range(indexedTypes, getInputs().getTypes());
  llvm::append_range(indexedTypes, getResults().getTypes());
  return verifyMapIndexingMaps(*this, indexedTypes);
}

LogicalResult ContractOp::verify() {
  if (getInputs().empty() || getResults().empty()) {
    return emitOpError("requires at least one input and one result");
  }
  SmallVector<Type> indexedTypes;
  llvm::append_range(indexedTypes, getInputs().getTypes());
  llvm::append_range(indexedTypes, getResults().getTypes());
  for (Type type : indexedTypes) {
    if (!isa<RankedTensorType>(type)) {
      return emitOpError("requires ranked tensor inputs and results");
    }
  }
  if (failed(verifyContractMaps(*this, indexedTypes))) {
    return failure();
  }
  if (getAccumulatorTypes().size() != getResults().size()) {
    return emitOpError("requires one accumulator type per result");
  }
  for (Attribute accumulator : getAccumulatorTypes()) {
    Type accumulatorType = cast<TypeAttr>(accumulator).getValue();
    if (!isScalarNumericType(accumulatorType)) {
      return emitOpError("accumulator types must be scalar numeric types");
    }
  }
  constexpr StringRef kIteratorKinds[] = {"parallel", "reduction"};
  constexpr StringRef kPrecisionValues[] = {"DEFAULT", "HIGH", "HIGHEST"};
  if (failed(verifyStringArray(*this, getIteratorKinds(), "iterator_kinds",
                               kIteratorKinds)) ||
      failed(verifyStringArray(*this, getPrecision(), "precision",
                               kPrecisionValues))) {
    return failure();
  }
  if (!getIndexingMaps().empty()) {
    AffineMap domain = cast<AffineMapAttr>(getIndexingMaps()[0]).getValue();
    if (getIteratorKinds().size() != domain.getNumDims()) {
      return emitOpError(
          "requires one iterator kind per indexing-map dimension");
    }
  }
  if (getPrecision().size() != getInputs().size()) {
    return emitOpError("requires one precision entry per input");
  }
  if (getAlgorithm() != "dot_general") {
    return emitOpError("supports only the 'dot_general' algorithm");
  }
  if (getInputs().size() != 2 || getResults().size() != 1) {
    return emitOpError(
        "the 'dot_general' algorithm requires two inputs and one result");
  }
  if (failed(verifyDotGeneralElementTypes(*this))) {
    return failure();
  }
  return verifyDotGeneralIterators(*this);
}

LogicalResult RelationOp::verify() {
  if (getResults().size() != 1) {
    return emitOpError("requires exactly one result");
  }
  auto result = dyn_cast<RankedTensorType>(getResult(0).getType());
  if (!result) {
    return emitOpError("requires ranked tensor results");
  }
  SmallVector<RankedTensorType> inputs;
  for (Value input : getInputs()) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return emitOpError("requires ranked tensor inputs");
    }
    inputs.push_back(inputType);
  }
  const size_t expectedInputs =
      (getSemantics() == RelationSemantics::Iota ||
       getSemantics() == RelationSemantics::Constant)
          ? 0
      : (getSemantics() == RelationSemantics::Add ||
         getSemantics() == RelationSemantics::Compare)
          ? 2
      : getSemantics() == RelationSemantics::Select ? 3
                                                    : 1;
  if (getInputs().size() != expectedInputs) {
    return emitOpError("has input arity inconsistent with its semantics");
  }
  const size_t expectedParameters =
      (getSemantics() == RelationSemantics::Iota ||
       getSemantics() == RelationSemantics::Constant)
          ? 1
      : getSemantics() == RelationSemantics::BroadcastInDim
          ? cast<RankedTensorType>(getInputs().front().getType()).getRank()
      : getSemantics() == RelationSemantics::Compare ? 2
                                                     : 0;
  if (getParameters().size() != expectedParameters) {
    return emitOpError("has parameter arity inconsistent with its semantics");
  }
  auto integerParameter = [&](size_t index) {
    return dyn_cast<IntegerAttr>(getParameters()[index]);
  };
  auto stringParameter = [&](size_t index) {
    return dyn_cast<StringAttr>(getParameters()[index]);
  };
  if (getSemantics() == RelationSemantics::Iota) {
    auto dimension = integerParameter(0);
    if (!result.hasStaticShape() || !result.getElementType().isInteger(32) ||
        !dimension || dimension.getInt() < 0 ||
        dimension.getInt() >= result.getRank()) {
      return emitOpError(
          "requires an in-range iota dimension and static i32 result");
    }
  } else if (getSemantics() == RelationSemantics::Constant) {
    auto value = integerParameter(0);
    if (result.getRank() != 0 || !result.getElementType().isInteger(32) ||
        !value || value.getType() != result.getElementType()) {
      return emitOpError("requires one scalar i32 constant value");
    }
  } else if (getSemantics() == RelationSemantics::Reshape) {
    if (!inputs[0].hasStaticShape() || !result.hasStaticShape() ||
        !inputs[0].getElementType().isInteger(32) ||
        !result.getElementType().isInteger(32) ||
        inputs[0].getNumElements() != result.getNumElements()) {
      return emitOpError("requires a copy-free static i32 reshape");
    }
  } else if (getSemantics() == RelationSemantics::BroadcastInDim) {
    if (!inputs[0].hasStaticShape() || !result.hasStaticShape() ||
        inputs[0].getElementType() != result.getElementType() ||
        (!result.getElementType().isInteger(1) &&
         !result.getElementType().isInteger(32))) {
      return emitOpError("requires matching static i1 or i32 element types");
    }
    llvm::SmallDenseSet<int64_t> dimensions;
    for (auto [inputDimension, parameter] :
         llvm::enumerate(getParameters())) {
      auto resultDimension = dyn_cast<IntegerAttr>(parameter);
      if (!resultDimension || resultDimension.getInt() < 0 ||
          resultDimension.getInt() >= result.getRank() ||
          !dimensions.insert(resultDimension.getInt()).second) {
        return emitOpError("requires unique in-range broadcast dimensions");
      }
      int64_t inputExtent = inputs[0].getDimSize(inputDimension);
      int64_t resultExtent = result.getDimSize(resultDimension.getInt());
      if (inputExtent != resultExtent && inputExtent != 1) {
        return emitOpError("has a non-broadcast-compatible dimension");
      }
    }
  } else if (getSemantics() == RelationSemantics::Add) {
    if (inputs[0] != inputs[1] || inputs[0] != result ||
        !result.getElementType().isInteger(32)) {
      return emitOpError("requires same-shaped i32 inputs and result");
    }
  } else if (getSemantics() == RelationSemantics::Compare) {
    auto comparisonType = stringParameter(0);
    auto direction = stringParameter(1);
    if (inputs[0] != inputs[1] ||
        inputs[0].getShape() != result.getShape() ||
        !inputs[0].getElementType().isInteger(32) ||
        !result.getElementType().isInteger(1) || !comparisonType ||
        (comparisonType.getValue() != "SIGNED" &&
         comparisonType.getValue() != "UNSIGNED") ||
        !direction ||
        !llvm::is_contained(ArrayRef<StringRef>{"EQ", "NE", "GE", "GT",
                                                     "LE", "LT"},
                            direction.getValue())) {
      return emitOpError("requires a valid i32 comparison and i1 result");
    }
  } else if (getSemantics() == RelationSemantics::Convert) {
    if (inputs[0].getShape() != result.getShape() ||
        !inputs[0].getElementType().isInteger(1) ||
        !result.getElementType().isF32()) {
      return emitOpError("requires a same-shaped exact i1-to-f32 conversion");
    }
  } else if (getSemantics() == RelationSemantics::Select) {
    if (inputs[0].getShape() != result.getShape() ||
        !inputs[0].getElementType().isInteger(1) || inputs[1] != result ||
        inputs[2] != result || !result.getElementType().isF32()) {
      return emitOpError("requires an i1 predicate and same-shaped f32 values");
    }
  }
  return success();
}

LogicalResult ScalarConvertOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (isa<ShapedType>(inputType) || isa<ShapedType>(resultType)) {
    return emitOpError("requires scalar input and result types");
  }
  if ((!inputType.isBF16() && !inputType.isF32()) ||
      (!resultType.isBF16() && !resultType.isF32())) {
    return emitOpError("supports only bf16 and f32 scalar types");
  }
  if (inputType == resultType) {
    if (!inputType.isF32() || getSemantics() != ScalarConvertSemantics::Exact) {
      return emitOpError(
          "same-type conversion supports only f32 exact semantics");
    }
    return success();
  }
  if (inputType.isBF16()) {
    if (getSemantics() != ScalarConvertSemantics::Exact) {
      return emitOpError("bf16 to f32 requires exact semantics");
    }
    return success();
  }
  if (getSemantics() != ScalarConvertSemantics::RoundNearestEven) {
    return emitOpError("f32 to bf16 requires round_nearest_even semantics");
  }
  return success();
}

LogicalResult FoldOp::verifyRegions() {
  if (failed(verifySingleBlockRegion(*this, getCombiner()))) {
    return failure();
  }
  if (getInputs().empty() || getResults().empty()) {
    return emitOpError("requires at least one input and one result");
  }
  if (getAccumulatorTypes().size() != getResults().size()) {
    return emitOpError("requires one accumulator type per result");
  }
  if (getInputs().size() != getResults().size()) {
    return emitOpError("requires one input per result");
  }
  if (getInitializers().size() != getResults().size()) {
    return emitOpError("requires one explicit initializer per result");
  }
  if (getReductionDimensions().empty()) {
    return emitOpError("requires at least one reduction dimension");
  }
  llvm::SmallDenseSet<int64_t> dimensions;
  for (int64_t dimension : getReductionDimensions()) {
    if (dimension < 0 || !dimensions.insert(dimension).second) {
      return emitOpError(
          "reduction dimensions must be non-negative and unique");
    }
  }
  SmallVector<RankedTensorType> inputTypes;
  SmallVector<RankedTensorType> resultTypes;
  for (auto [input, result] : llvm::zip_equal(getInputs(), getResults())) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!inputType || inputType.getRank() == 0 || !resultType) {
      return emitOpError(
          "requires positive-rank tensor inputs and ranked tensor results");
    }
    inputTypes.push_back(inputType);
    resultTypes.push_back(resultType);
    for (int64_t dimension : dimensions) {
      if (dimension >= inputType.getRank()) {
        return emitOpError("reduction dimension is outside an input rank");
      }
    }
    SmallVector<int64_t> expectedShape;
    for (int64_t dimension = 0; dimension < inputType.getRank(); ++dimension) {
      if (!dimensions.contains(dimension)) {
        expectedShape.push_back(inputType.getDimSize(dimension));
      }
    }
    if (resultType.getShape() != ArrayRef<int64_t>(expectedShape)) {
      return emitOpError("result shape must equal the input shape with reduced "
                         "dimensions removed");
    }
  }
  for (size_t index = 1; index < inputTypes.size(); ++index) {
    if (!haveCompatibleShapes(inputTypes.front(), inputTypes[index]) ||
        !haveCompatibleShapes(resultTypes.front(), resultTypes[index])) {
      return emitOpError(
          "multi-input folds require compatible input and result shapes");
    }
  }
  Block &combiner = getCombiner().front();
  const size_t resultCount = getResults().size();
  if (combiner.getNumArguments() != 2 * resultCount) {
    return emitOpError("combiner requires two block arguments per result");
  }
  auto yield = cast<YieldOp>(combiner.getTerminator());
  if (yield.getValues().size() != resultCount) {
    return emitOpError("combiner requires one yielded value per result");
  }
  for (size_t index = 0; index < resultCount; ++index) {
    Type accumulatorType =
        cast<TypeAttr>(getAccumulatorTypes()[index]).getValue();
    Type inputElementType = scalarType(getInputs()[index].getType());
    auto initializerType =
        dyn_cast<RankedTensorType>(getInitializers()[index].getType());
    if (!isScalarNumericType(accumulatorType)) {
      return emitOpError("accumulator types must be scalar numeric types");
    }
    if (inputElementType != accumulatorType || !initializerType ||
        initializerType.getRank() != 0 ||
        initializerType.getElementType() != accumulatorType ||
        combiner.getArgument(index).getType() != inputElementType ||
        combiner.getArgument(index + resultCount).getType() !=
            accumulatorType ||
        yield.getValues()[index].getType() != accumulatorType ||
        scalarType(getResults()[index].getType()) != accumulatorType) {
      return emitOpError("input elements, rank-zero initializers, combiner "
                         "arguments and yields, and "
                         "result elements must use accumulator types");
    }
  }
  return verifyPureScalarComputation(*this, getCombiner());
}

std::string materializationPlanFingerprint(MaterializationPlanOp plan) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  stream << "schema=" << plan.getSchemaVersion() << ';';
  Attribute(plan.getPolicyAttr()).print(stream);
  stream << '\n';
  for (Operation &operation : plan.getBody().front()) {
    if (isa<MaterializationPlanYieldOp>(operation)) {
      continue;
    }
    stream << operation.getName() << '{';
    for (NamedAttribute attribute : operation.getAttrs()) {
      stream << attribute.getName() << '=';
      attribute.getValue().print(stream);
      stream << ',';
    }
    stream << "}\n";
  }
  stream.flush();
  llvm::SHA256 digest;
  digest.update(normalized);
  return llvm::toHex(digest.final(), true);
}

SmallVector<unsigned> projectedDomainMasks(RankedTensorType type,
                                           ArrayRef<int64_t> domain,
                                           unsigned reductionAxis) {
  SmallVector<unsigned> masks{0};
  for (int64_t extent : type.getShape()) {
    llvm::SmallDenseSet<unsigned> next;
    for (unsigned mask : masks) {
      for (auto [axis, domainExtent] : llvm::enumerate(domain)) {
        const unsigned bit = 1u << axis;
        if ((mask & bit) == 0 && extent == domainExtent) {
          next.insert(mask | bit);
        }
      }
    }
    masks.assign(next.begin(), next.end());
  }
  llvm::erase_if(masks, [reductionAxis](unsigned mask) {
    return (mask & (1u << reductionAxis)) == 0;
  });
  return masks;
}

bool inputShapesBindContractDomain(RankedTensorType lhs, RankedTensorType rhs,
                                   ArrayRef<int64_t> domain) {
  if (domain.size() < 3 || domain.size() > 9 || lhs.getRank() < 2 ||
      rhs.getRank() < 2 || lhs.getRank() > 8 || rhs.getRank() > 8) {
    return false;
  }
  const unsigned reductionAxis = domain.size() - 1;
  SmallVector<unsigned> lhsMasks =
      projectedDomainMasks(lhs, domain, reductionAxis);
  SmallVector<unsigned> rhsMasks =
      projectedDomainMasks(rhs, domain, reductionAxis);
  const unsigned fullMask = (1u << domain.size()) - 1;
  return llvm::any_of(lhsMasks, [&](unsigned lhsMask) {
    return llvm::any_of(rhsMasks, [&](unsigned rhsMask) {
      return (lhsMask | rhsMask) == fullMask;
    });
  });
}

StringRef cardinalName(size_t value) {
  constexpr StringLiteral names[] = {"zero", "one", "two",   "three", "four",
                                     "five", "six", "seven", "eight"};
  return value < 9 ? StringRef(names[value]) : StringRef{};
}

LogicalResult MaterializationPlanOp::verifyRegions() {
  if (!getOperation()->getDiscardableAttrs().empty()) {
    return emitOpError("does not permit discardable attributes");
  }
  Region &body = getBody();
  if (body.empty() || !llvm::hasSingleElement(body)) {
    return emitOpError("requires exactly one non-empty body block");
  }
  Block &block = body.front();
  if (!isa<MaterializationPlanYieldOp>(block.getTerminator())) {
    return emitOpError(
        "body must terminate with shuttle.materialization_plan_yield");
  }
  if (getSchemaVersion() != 1 && getSchemaVersion() != 2) {
    return emitOpError("requires materialization plan schema version 1 or 2");
  }
  if (!isLowerHexDigest(getFingerprint())) {
    return emitOpError("fingerprint must be a lowercase SHA-256 digest");
  }

  SmallVector<MaterializationBufferOp> buffers;
  SmallVector<MaterializationTaskOp> tasks;
  bool sawTask = false;
  for (Operation &operation : block.without_terminator()) {
    if (auto buffer = dyn_cast<MaterializationBufferOp>(operation)) {
      if (!buffer->getDiscardableAttrs().empty()) {
        return buffer.emitOpError("does not permit discardable attributes");
      }
      if (sawTask) {
        return emitOpError(
            "materialization buffers must precede materialization tasks");
      }
      buffers.push_back(buffer);
      continue;
    }
    if (auto task = dyn_cast<MaterializationTaskOp>(operation)) {
      if (!task->getDiscardableAttrs().empty()) {
        return task.emitOpError("does not permit discardable attributes");
      }
      sawTask = true;
      tasks.push_back(task);
      continue;
    }
    return emitOpError(
        "body may contain only materialization buffers and tasks");
  }
  if (buffers.empty() || tasks.empty()) {
    return emitOpError("requires at least one buffer and one task");
  }
  if (!block.getTerminator()->getDiscardableAttrs().empty()) {
    return block.getTerminator()->emitOpError(
        "does not permit discardable attributes");
  }

  unsigned foldCount = 0;
  unsigned contractCount = 0;
  unsigned foldRegionCount = 0;
  unsigned contractRegionCount = 0;
  int64_t contractFoldCompositionTask = -1;
  int64_t mapContractCompositionTask = -1;
  SmallVector<SmallVector<int64_t>> actualConsumers(buffers.size());
  for (auto [position, buffer] : llvm::enumerate(buffers)) {
    if (buffer.getOrdinal() != static_cast<int64_t>(position)) {
      return buffer.emitOpError("buffer ordinals must be contiguous");
    }
    auto tensorType = dyn_cast<RankedTensorType>(buffer.getTensorType());
    if (!tensorType || !tensorType.hasStaticShape() ||
        llvm::any_of(tensorType.getShape(),
                     [](int64_t extent) { return extent <= 0; }))
      return buffer.emitOpError("requires a positive static ranked tensor type");
    std::optional<int64_t> producer =
        optionalIntegerAttribute(buffer, "producer");
    if (buffer.getLiveIn() != !producer.has_value()) {
      return buffer.emitOpError(
          "live-in status must exactly identify producer-free buffers");
    }
    const bool external =
        buffer.getStorage() == MaterializationStorage::External;
    Type elementType = tensorType.getElementType();
    const bool externalElement = elementType.isF32() || elementType.isBF16();
    const bool logicalTemporary = elementType.isInteger(1) ||
                                  elementType.isInteger(32);
    if ((!externalElement && !(logicalTemporary && !external)))
      return buffer.emitOpError(
          "requires bf16/f32 external data or i1/i32 logical temporaries");
    if (external != (buffer.getLiveIn() || buffer.getLiveOut())) {
      return buffer.emitOpError(
          "external storage must exactly identify live-in or live-out buffers");
    }
    if (producer && (*producer < 0 || *producer >= tasks.size())) {
      return buffer.emitOpError("producer is outside the task range");
    }
    if (buffer.getLifetimeStart() != producer.value_or(0)) {
      return buffer.emitOpError("lifetime start must equal producer or zero");
    }
    if (buffer.getLifetimeEnd() < buffer.getLifetimeStart() ||
        buffer.getLifetimeEnd() > static_cast<int64_t>(tasks.size())) {
      return buffer.emitOpError("has an invalid task lifetime interval");
    }
  }

  for (auto [position, task] : llvm::enumerate(tasks)) {
    const int64_t ordinal = static_cast<int64_t>(position);
    if (task.getOrdinal() != ordinal) {
      return task.emitOpError("task ordinals must be contiguous");
    }
    if (!isLowerHexDigest(task.getSemanticFingerprint())) {
      return task.emitOpError(
          "semantic fingerprint must be a lowercase SHA-256 digest");
    }
    if (llvm::any_of(task.getDomainShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      return task.emitOpError("requires a positive static task domain");
    }
    llvm::SmallDenseSet<int64_t> outputSet;
    llvm::SmallDenseSet<int64_t> dependencySet;
    SmallVector<int64_t> expectedDependencies;
    for (int64_t bufferOrdinal : task.getInputBuffers()) {
      if (bufferOrdinal < 0 || bufferOrdinal >= buffers.size()) {
        return task.emitOpError("input buffers must be in range");
      }
      if (actualConsumers[bufferOrdinal].empty() ||
          actualConsumers[bufferOrdinal].back() != ordinal) {
        actualConsumers[bufferOrdinal].push_back(ordinal);
      }
      std::optional<int64_t> producer =
          optionalIntegerAttribute(buffers[bufferOrdinal], "producer");
      if (producer && !llvm::is_contained(expectedDependencies, *producer)) {
        expectedDependencies.push_back(*producer);
      }
    }
    llvm::sort(expectedDependencies);
    for (int64_t bufferOrdinal : task.getOutputBuffers()) {
      if (bufferOrdinal < 0 || bufferOrdinal >= buffers.size() ||
          !outputSet.insert(bufferOrdinal).second ||
          optionalIntegerAttribute(buffers[bufferOrdinal], "producer") !=
              ordinal) {
        return task.emitOpError(
            "output buffers must be unique, in range, and owned by the task");
      }
    }
    SmallVector<int64_t> dependencies =
        integerArray(task.getDependenciesAttr());
    for (int64_t dependency : dependencies) {
      if (dependency < 0 || dependency >= ordinal ||
          !dependencySet.insert(dependency).second) {
        return task.emitOpError(
            "dependencies must be unique earlier task ordinals");
      }
    }
    if (dependencies != expectedDependencies) {
      return task.emitOpError(
          "dependencies must equal input-buffer producer tasks");
    }
    const bool hasFoldRegionSources = task->hasAttr("fold_region_sources");
    const bool hasFoldRegionPhases = task->hasAttr("fold_region_phases");
    const bool hasFoldRegionOutputSources =
        task->hasAttr("fold_region_output_sources");
    const bool hasFoldRegionBoundaryUses =
        task->hasAttr("fold_region_boundary_uses");
    const bool hasFoldRegionRecomputedSources =
        task->hasAttr("fold_region_recomputed_sources");
    const bool hasContractRegionSources =
        task->hasAttr("contract_region_sources");
    const bool hasContractRegionStrategy =
        task->hasAttr("contract_region_strategy");
    const bool hasContractRegionFold =
        task->hasAttr("contract_region_fold_source");
    const bool hasContractRegionContract =
        task->hasAttr("contract_region_contract_source");
    const bool hasContractRegionBoundary =
        task->hasAttr("contract_region_bf16_boundary_source");
    const bool hasContractRegionEpilogue =
        task->hasAttr("contract_region_epilogue_sources");
    const bool hasContractRegionOperand =
        task->hasAttr("contract_region_normalized_operand");
    const bool hasContractRegionCost = task->hasAttr("contract_region_cost");
    const bool hasContractRegionRecomputed =
        task->hasAttr("contract_region_recomputed_sources");
    auto compositionKind = task->getAttrOfType<StringAttr>("composition_kind");
    auto compositionSources =
        task->getAttrOfType<ArrayAttr>("composition_sources");
    auto compositionCost =
        task->getAttrOfType<DictionaryAttr>("composition_cost");
    const bool hasComposition =
        compositionKind || compositionSources || compositionCost;
    if ((hasFoldRegionPhases || hasFoldRegionOutputSources ||
         hasFoldRegionBoundaryUses || hasFoldRegionRecomputedSources) &&
        !hasFoldRegionSources) {
      return task.emitOpError(
          "Fold-region phases, boundary uses, and recomputation require "
          "Fold-region sources");
    }
    if (hasFoldRegionSources &&
        task.getKind() != MaterializationTaskKind::Fold) {
      return task.emitOpError("Fold-region provenance requires a Fold task");
    }
    if ((hasContractRegionStrategy || hasContractRegionFold ||
         hasContractRegionContract || hasContractRegionBoundary ||
         hasContractRegionEpilogue || hasContractRegionOperand ||
         hasContractRegionCost || hasContractRegionRecomputed) &&
        !hasContractRegionSources) {
      return task.emitOpError(
          "Contract-region metadata requires Contract-region sources");
    }
    if (hasContractRegionSources &&
        task.getKind() != MaterializationTaskKind::Contract) {
      return task.emitOpError(
          "Contract-region provenance requires a Contract task");
    }
    if (hasComposition) {
      const bool contractFold =
          compositionKind && compositionKind.getValue() == "contract_fold";
      const bool mapContract =
          compositionKind && compositionKind.getValue() == "map_contract";
      const bool legacyMetadata =
          task->hasAttr("fused_sources") ||
          task->hasAttr("composed_input_maps") ||
          task->hasAttr("fold_epilogue_sources") ||
          task->hasAttr("epilogue_input_maps") || hasFoldRegionSources ||
          hasFoldRegionPhases || hasFoldRegionOutputSources ||
          hasFoldRegionBoundaryUses || hasFoldRegionRecomputedSources ||
          hasContractRegionSources || hasContractRegionStrategy ||
          hasContractRegionFold || hasContractRegionContract ||
          hasContractRegionBoundary || hasContractRegionEpilogue ||
          hasContractRegionOperand || hasContractRegionCost ||
          hasContractRegionRecomputed;
      if (getSchemaVersion() != 2 || !compositionKind || !compositionSources ||
          (contractFold && compositionSources.size() != 2) ||
          (mapContract && compositionSources.size() != 2 &&
           compositionSources.size() != 3) ||
          (!contractFold && !mapContract) || legacyMetadata ||
          llvm::any_of(compositionSources, [](Attribute source) {
            return !isa<SourceRefAttr>(source);
          })) {
        return task.emitOpError(
            "schema-v2 generic composition provenance must be complete and "
            "exclusive");
      }
      if (contractFold) {
        if (contractFoldCompositionTask >= 0 || compositionCost ||
            task.getKind() != MaterializationTaskKind::Fold ||
            task.getInputBuffers().size() != 3 ||
            task.getOutputBuffers().size() != 2 ||
            !task.getOrderFree().value_or(false) ||
            task.getDomainShape().size() < 2 ||
            task.getDomainShape().size() > 8 ||
            task.getReductionDimensions() !=
                ArrayRef<int64_t>(
                    {static_cast<int64_t>(task.getDomainShape().size()) - 1})) {
          return task.emitOpError(
              "Contract-to-Fold composition must bind one final-axis Fold "
              "with three inputs and the ordered full and compact outputs");
        }
        auto lhs = cast<RankedTensorType>(
            buffers[task.getInputBuffers()[0]].getTensorType());
        auto rhs = cast<RankedTensorType>(
            buffers[task.getInputBuffers()[1]].getTensorType());
        auto initializer = cast<RankedTensorType>(
            buffers[task.getInputBuffers()[2]].getTensorType());
        auto full = cast<RankedTensorType>(
            buffers[task.getOutputBuffers()[0]].getTensorType());
        auto compact = cast<RankedTensorType>(
            buffers[task.getOutputBuffers()[1]].getTensorType());
        if (!lhs.getElementType().isF32() || !rhs.getElementType().isF32() ||
            initializer.getRank() != 0 ||
            !initializer.getElementType().isF32() ||
            full.getShape() != task.getDomainShape() ||
            !full.getElementType().isF32() ||
            compact.getShape() != full.getShape().drop_back() ||
            !compact.getElementType().isF32()) {
          return task.emitOpError(
              "Contract-to-Fold buffers must preserve f32 operands, scalar "
              "initializer, full result, and compact statistic");
        }
        contractFoldCompositionTask = ordinal;
        ++foldCount;
        continue;
      }
      if (mapContractCompositionTask >= 0 || !compositionCost ||
          task.getKind() != MaterializationTaskKind::Contract ||
          task.getInputBuffers().size() != 3 ||
          task.getOutputBuffers().size() != 1 || task.getOrderFree() ||
          task.getDomainShape().size() < 3 ||
          task.getDomainShape().size() > 9 ||
          task.getReductionDimensions() !=
              ArrayRef<int64_t>(
                  {static_cast<int64_t>(task.getDomainShape().size()) - 1})) {
        return task.emitOpError(
            "Map-to-Contract composition must bind one final-axis Contract "
            "with the ordered full, compact, and projection inputs");
      }
      auto full = cast<RankedTensorType>(
          buffers[task.getInputBuffers()[0]].getTensorType());
      auto compact = cast<RankedTensorType>(
          buffers[task.getInputBuffers()[1]].getTensorType());
      auto projection = cast<RankedTensorType>(
          buffers[task.getInputBuffers()[2]].getTensorType());
      auto result = cast<RankedTensorType>(
          buffers[task.getOutputBuffers()[0]].getTensorType());
      if (!full.getElementType().isF32() ||
          compact.getShape() != full.getShape().drop_back() ||
          !compact.getElementType().isF32() ||
          !projection.getElementType().isF32() ||
          result.getShape() != task.getDomainShape().drop_back() ||
          !result.getElementType().isF32()) {
        return task.emitOpError(
            "Map-to-Contract buffers must preserve the f32 full and compact "
            "boundary and public result");
      }
      constexpr StringLiteral costFields[] = {
          "baseline_launches",
          "candidate_launches",
          "baseline_temporary_bytes",
          "candidate_temporary_bytes",
          "baseline_temporary_traffic_bytes",
          "candidate_temporary_traffic_bytes",
          "contract_evaluations"};
      if (compositionCost.size() != std::size(costFields)) {
        return task.emitOpError(
            "generic composition cost requires the closed relative fields");
      }
      SmallVector<int64_t> costs;
      for (StringRef field : costFields) {
        auto value = compositionCost.getAs<IntegerAttr>(field);
        if (!value || value.getInt() < 0) {
          return task.emitOpError(
              "generic composition cost fields must be nonnegative i64");
        }
        costs.push_back(value.getInt());
      }
      if (costs[1] > costs[0] || costs[3] > costs[2] || costs[5] > costs[4] ||
          !(costs[1] < costs[0] || costs[3] < costs[2] ||
            costs[5] < costs[4]) ||
          costs[6] != 1) {
        return task.emitOpError(
            "generic composition cost must strictly dominate its baseline");
      }
      mapContractCompositionTask = ordinal;
      ++contractCount;
      continue;
    }
    if (task.getKind() == MaterializationTaskKind::Fold) {
      ++foldCount;
      auto foldRegionSources =
          task->getAttrOfType<ArrayAttr>("fold_region_sources");
      if (foldRegionSources) {
        ++foldRegionCount;
        auto phases = task->getAttrOfType<ArrayAttr>("fold_region_phases");
        auto outputSources =
            task->getAttrOfType<ArrayAttr>("fold_region_output_sources");
        auto boundaryUses =
            task->getAttrOfType<ArrayAttr>("fold_region_boundary_uses");
        auto recomputed =
            task->getAttrOfType<ArrayAttr>("fold_region_recomputed_sources");
        if (getSchemaVersion() != 2 || foldRegionSources.empty() || !phases ||
            phases.empty() || !outputSources || outputSources.empty() ||
            outputSources.size() != task.getOutputBuffers().size() ||
            !boundaryUses || boundaryUses.empty() ||
            task->getAttr("fused_sources") ||
            task->getAttr("composed_input_maps") ||
            task->getAttr("fold_epilogue_sources") ||
            task->getAttr("epilogue_input_maps") ||
            task.getInputBuffers().empty() || task.getOutputBuffers().empty() ||
            !task.getOrderFree().value_or(false)) {
          return task.emitOpError(
              "schema-v2 Fold regions require closed sources, phases, "
              "boundary uses, and ordered outputs");
        }
        llvm::SmallDenseSet<Attribute> sourceSet;
        std::optional<SourceRefAttr> previousSource;
        for (Attribute attribute : foldRegionSources) {
          auto source = dyn_cast<SourceRefAttr>(attribute);
          if (!source || !sourceSet.insert(source).second ||
              (previousSource &&
               std::tuple(previousSource->getFunctionOrdinal(),
                          previousSource->getBlockOrdinal(),
                          previousSource->getOperationOrdinal(),
                          previousSource->getResultOrdinal()) >=
                   std::tuple(source.getFunctionOrdinal(),
                              source.getBlockOrdinal(),
                              source.getOperationOrdinal(),
                              source.getResultOrdinal()))) {
            return task.emitOpError(
                "Fold region sources must be unique and source ordered");
          }
          previousSource = source;
        }
        llvm::SmallDenseSet<Attribute> outputSourceSet;
        for (Attribute source : outputSources) {
          if (!isa<SourceRefAttr>(source) || !sourceSet.contains(source) ||
              !outputSourceSet.insert(source).second) {
            return task.emitOpError(
                "Fold region output sources must uniquely bind task sources");
          }
        }
        for (auto [phaseOrdinal, attribute] : llvm::enumerate(phases)) {
          auto phase = dyn_cast<DictionaryAttr>(attribute);
          auto source =
              phase ? dyn_cast_or_null<SourceRefAttr>(phase.get("source"))
                    : SourceRefAttr{};
          auto domain = phase ? dyn_cast_or_null<DenseI64ArrayAttr>(
                                    phase.get("domain_shape"))
                              : DenseI64ArrayAttr{};
          auto reduction = phase ? dyn_cast_or_null<DenseI64ArrayAttr>(
                                       phase.get("reduction_dimensions"))
                                 : DenseI64ArrayAttr{};
          auto orderFree =
              phase ? dyn_cast_or_null<BoolAttr>(phase.get("order_free"))
                    : BoolAttr{};
          auto transition =
              phase ? dyn_cast_or_null<StringAttr>(phase.get("transition"))
                    : StringAttr{};
          if (!phase || phase.size() != 5 || !source || !domain || !reduction ||
              !orderFree || !transition || !sourceSet.contains(source) ||
              transition.getValue() !=
                  (phaseOrdinal == 0 ? "launch_boundary" : "cta_local") ||
              (phaseOrdinal == 0 &&
               (domain.asArrayRef() != task.getDomainShape() ||
                reduction.asArrayRef() != task.getReductionDimensions() ||
                orderFree.getValue() != task.getOrderFree().value_or(false)))) {
            return task.emitOpError(
                "Fold region phases must close ordered launch-local "
                "reductions");
          }
        }
        llvm::SmallDenseSet<std::pair<Attribute, int64_t>> boundarySet;
        for (Attribute attribute : boundaryUses) {
          auto use = dyn_cast<DictionaryAttr>(attribute);
          auto input =
              use ? dyn_cast_or_null<IntegerAttr>(use.get("input_buffer"))
                  : IntegerAttr{};
          auto source = use ? dyn_cast_or_null<SourceRefAttr>(use.get("source"))
                            : SourceRefAttr{};
          auto operand = use ? dyn_cast_or_null<IntegerAttr>(use.get("operand"))
                             : IntegerAttr{};
          auto indexing =
              use ? dyn_cast_or_null<AffineMapAttr>(use.get("indexing_map"))
                  : AffineMapAttr{};
          auto phase = use ? dyn_cast_or_null<IntegerAttr>(use.get("phase"))
                           : IntegerAttr{};
          if (!use || use.size() != 5 || !input || !source || !operand ||
              !indexing || !phase || !sourceSet.contains(source) ||
              operand.getInt() < 0 || phase.getInt() < 0 ||
              phase.getInt() >= static_cast<int64_t>(phases.size()) ||
              !llvm::is_contained(task.getInputBuffers(), input.getInt()) ||
              !boundarySet.insert({source, operand.getInt()}).second) {
            return task.emitOpError(
                "Fold region boundary uses must uniquely bind task inputs");
          }
        }
        llvm::SmallDenseSet<Attribute> recomputedSet;
        if (recomputed) {
          for (Attribute source : recomputed) {
            if (!isa<SourceRefAttr>(source) || !sourceSet.contains(source) ||
                !recomputedSet.insert(source).second) {
              return task.emitOpError(
                  "Fold region recomputation declarations must be unique "
                  "task sources");
            }
          }
        }
        continue;
      }
      const bool hasEpilogue =
          static_cast<bool>(task->getAttrOfType<ArrayAttr>(
              "fold_epilogue_sources"));
      const int64_t domainRank = task.getDomainShape().size();
      ArrayRef<int64_t> reductionDimensions = task.getReductionDimensions();
      const int64_t reductionAxis = reductionDimensions.size() == 1
                                        ? reductionDimensions.front()
                                        : -1;
      const bool rankTwoFold =
          domainRank == 2 && reductionAxis >= 0 && reductionAxis < domainRank;
      const bool terminalRankNFold =
          domainRank >= 3 && domainRank <= 8 &&
          reductionDimensions == ArrayRef<int64_t>({domainRank - 1});
      const bool fullPrefixRankNFold =
          domainRank >= 3 && domainRank <= 8 &&
          reductionDimensions.size() == static_cast<size_t>(domainRank - 1) &&
          llvm::all_of(llvm::enumerate(reductionDimensions),
                       [](auto indexedAxis) {
                         return indexedAxis.value() ==
                                static_cast<int64_t>(indexedAxis.index());
                       });
      if ((!rankTwoFold && !terminalRankNFold && !fullPrefixRankNFold) ||
          !task.getOrderFree().value_or(false) ||
          (hasEpilogue ? task.getInputBuffers().size() < 2
                       : task.getInputBuffers().size() != 2) ||
          task.getOutputBuffers().size() != 1) {
        return task.emitOpError(
            "Fold tasks require a rank-two in-range reduction or a rank-three "
            "through rank-eight terminal-axis or full-prefix reduction, and "
            "order_free=true");
      }
      auto inputType = cast<RankedTensorType>(
          buffers[task.getInputBuffers()[0]].getTensorType());
      auto initializerType = cast<RankedTensorType>(
          buffers[task.getInputBuffers()[1]].getTensorType());
      auto resultType = cast<RankedTensorType>(
          buffers[task.getOutputBuffers()[0]].getTensorType());
      SmallVector<int64_t> expectedResultShape(task.getDomainShape());
      for (int64_t axis : llvm::reverse(reductionDimensions)) {
        expectedResultShape.erase(expectedResultShape.begin() + axis);
      }
      auto isSingletonInsertion = [](ArrayRef<int64_t> source,
                                     ArrayRef<int64_t> destination) {
        size_t sourceAxis = 0;
        for (int64_t extent : destination) {
          if (sourceAxis < source.size() && extent == source[sourceAxis]) {
            ++sourceAxis;
          } else if (extent != 1) {
            return false;
          }
        }
        return sourceAxis == source.size();
      };
      const bool resultShapeMatches =
          resultType.getShape() == ArrayRef<int64_t>(expectedResultShape) ||
          (hasEpilogue && isSingletonInsertion(
                              expectedResultShape, resultType.getShape()));
      if (inputType.getShape() != task.getDomainShape() ||
          !inputType.getElementType().isF32() ||
          initializerType.getRank() != 0 ||
          !initializerType.getElementType().isF32() ||
          !resultShapeMatches || !resultType.getElementType().isF32() ||
          (hasEpilogue &&
           llvm::any_of(task.getInputBuffers().drop_front(2),
                        [&](int64_t bufferOrdinal) {
                          auto type = cast<RankedTensorType>(
                              buffers[bufferOrdinal].getTensorType());
                          return !type.hasStaticShape() ||
                                 !type.getElementType().isF32();
                        }))) {
        return task.emitOpError(
            "Fold buffer types must match its f32 domain and accumulator");
      }
    } else if (task.getKind() == MaterializationTaskKind::Contract) {
      ++contractCount;
      auto contractSources =
          task->getAttrOfType<ArrayAttr>("contract_region_sources");
      if (contractSources) {
        ++contractRegionCount;
        auto strategy =
            task->getAttrOfType<StringAttr>("contract_region_strategy");
        auto foldSource =
            task->getAttrOfType<SourceRefAttr>("contract_region_fold_source");
        auto contractSource = task->getAttrOfType<SourceRefAttr>(
            "contract_region_contract_source");
        auto boundarySource = task->getAttrOfType<SourceRefAttr>(
            "contract_region_bf16_boundary_source");
        auto epilogueSources =
            task->getAttrOfType<ArrayAttr>("contract_region_epilogue_sources");
        auto normalizedOperand = task->getAttrOfType<IntegerAttr>(
            "contract_region_normalized_operand");
        auto cost = task->getAttrOfType<DictionaryAttr>("contract_region_cost");
        auto recomputed = task->getAttrOfType<ArrayAttr>(
            "contract_region_recomputed_sources");
        ArrayRef<int64_t> outputBuffers = task.getOutputBuffers();
        const bool materializesNormalized = outputBuffers.size() == 2;
        if (getSchemaVersion() != 2 || contractSources.empty() || !strategy ||
            !foldSource || !contractSource || !boundarySource ||
            !epilogueSources || epilogueSources.empty() || !normalizedOperand ||
            normalizedOperand.getInt() != 0 || !cost ||
            task->getAttr("fused_sources") ||
            task->getAttr("composed_input_maps") ||
            task->getAttr("fold_epilogue_sources") ||
            task->getAttr("epilogue_input_maps") ||
            task->getAttr("fold_region_sources") || task.getOrderFree() ||
            (outputBuffers.size() != 1 && !materializesNormalized)) {
          return task.emitOpError(
              "schema-v2 Contract regions require closed provenance, cost, "
              "one result, and at most one normalized boundary output");
        }
        const bool inlineFold = strategy.getValue() == "inline_fold";
        const bool compactStat = strategy.getValue() == "compact_stat";
        if ((!inlineFold && !compactStat) ||
            task.getInputBuffers().size() != (inlineFold ? 3 : 4)) {
          return task.emitOpError(
              "Contract-region strategy must bind inline or compact inputs");
        }
        llvm::SmallDenseSet<Attribute> sourceSet;
        std::optional<SourceRefAttr> previous;
        for (Attribute attribute : contractSources) {
          auto source = dyn_cast<SourceRefAttr>(attribute);
          if (!source || !sourceSet.insert(source).second ||
              (previous && std::tuple(previous->getFunctionOrdinal(),
                                      previous->getBlockOrdinal(),
                                      previous->getOperationOrdinal(),
                                      previous->getResultOrdinal()) >=
                               std::tuple(source.getFunctionOrdinal(),
                                          source.getBlockOrdinal(),
                                          source.getOperationOrdinal(),
                                          source.getResultOrdinal()))) {
            return task.emitOpError(
                "Contract region sources must be unique and source ordered");
          }
          previous = source;
        }
        if (!sourceSet.contains(contractSource) ||
            !sourceSet.contains(boundarySource) ||
            (inlineFold && !sourceSet.contains(foldSource))) {
          return task.emitOpError(
              "Contract-region designated sources must bind its provenance");
        }
        llvm::SmallDenseSet<Attribute> epilogueSet;
        for (Attribute source : epilogueSources) {
          if (!isa<SourceRefAttr>(source) || !sourceSet.contains(source) ||
              !epilogueSet.insert(source).second) {
            return task.emitOpError(
                "Contract-region epilogues must be unique task sources");
          }
        }
        llvm::SmallDenseSet<Attribute> recomputedSet;
        if (recomputed) {
          for (Attribute source : recomputed) {
            if (!isa<SourceRefAttr>(source) || !sourceSet.contains(source) ||
                !recomputedSet.insert(source).second) {
              return task.emitOpError(
                  "Contract-region recomputation must be unique task sources");
            }
          }
        }
        constexpr StringLiteral costFields[] = {"rows",
                                                "reduction",
                                                "columns",
                                                "output_tile",
                                                "output_tiles",
                                                "duplicated_fold_elements",
                                                "normalized_temporary_bytes",
                                                "contract_temporary_bytes",
                                                "compact_stat_bytes"};
        if (cost.size() != std::size(costFields))
          return task.emitOpError(
              "Contract-region cost requires the closed structural fields");
        SmallVector<int64_t> costValues;
        for (StringRef field : costFields) {
          auto value = cost.getAs<IntegerAttr>(field);
          if (!value || value.getInt() < 0)
            return task.emitOpError(
                "Contract-region cost fields must be nonnegative i64");
          costValues.push_back(value.getInt());
        }
        auto [rows, reduction, columns, tile, tiles, duplicated,
              normalizedBytes, contractBytes, compactBytes] =
            std::tuple(costValues[0], costValues[1], costValues[2],
                       costValues[3], costValues[4], costValues[5],
                       costValues[6], costValues[7], costValues[8]);
        auto product = [](std::initializer_list<int64_t> factors)
            -> std::optional<int64_t> {
          int64_t value = 1;
          for (int64_t factor : factors) {
            if (factor < 0 ||
                (factor != 0 &&
                 value > std::numeric_limits<int64_t>::max() / factor))
              return std::nullopt;
            value *= factor;
          }
          return value;
        };
        if (rows <= 0 || reduction <= 0 || columns <= 0 || tile <= 0 ||
            tiles <= 0 || tile > columns ||
            tiles != (columns + tile - 1) / tile ||
            product({rows, reduction, tiles - 1}) != duplicated ||
            product({2, rows, reduction}) != normalizedBytes ||
            product({4, rows, columns}) != contractBytes ||
            (inlineFold ? compactBytes != 0
                        : product({4, rows}) != compactBytes)) {
          return task.emitOpError(
              "Contract-region cost must equal structural work and traffic");
        }
        ArrayRef<int64_t> domain = task.getDomainShape();
        const int64_t reductionAxis = domain.size() - 1;
        auto outputType = cast<RankedTensorType>(
            buffers[outputBuffers.back()].getTensorType());
        int64_t domainRows = 1;
        if (domain.size() >= 3) {
          for (int64_t dimension : domain.drop_back(2)) {
            if (dimension <= 0 ||
                domainRows > std::numeric_limits<int64_t>::max() / dimension) {
              domainRows = -1;
              break;
            }
            domainRows *= dimension;
          }
        }
        if (domain.size() < 3 || domain.size() > 9 || domainRows != rows ||
            task.getReductionDimensions() !=
                ArrayRef<int64_t>({reductionAxis}) ||
            outputType.getShape() != domain.drop_back() ||
            !outputType.getElementType().isBF16()) {
          return task.emitOpError(
              "Contract-region domain and BF16 output must bind its cost");
        }
        if (materializesNormalized) {
          MaterializationBufferOp normalized = buffers[outputBuffers.front()];
          auto normalizedType =
              cast<RankedTensorType>(normalized.getTensorType());
          SmallVector<int64_t> normalizedShape(domain.drop_back(2));
          normalizedShape.push_back(reduction);
          if (normalized.getLiveOut() ||
              normalized.getStorage() != MaterializationStorage::Temporary ||
              normalizedType.getShape() != ArrayRef<int64_t>(normalizedShape) ||
              !normalizedType.getElementType().isBF16()) {
            return task.emitOpError(
                "Contract-region normalized output must be one private BF16 "
                "row boundary");
          }
        }
        for (auto [inputOrdinal, bufferOrdinal] :
             llvm::enumerate(task.getInputBuffers())) {
          auto type =
              cast<RankedTensorType>(buffers[bufferOrdinal].getTensorType());
          const bool compactScale = compactStat && inputOrdinal == 1;
          if ((compactScale && !type.getElementType().isF32()) ||
              (!compactScale && !type.getElementType().isBF16())) {
            return task.emitOpError(
                "Contract-region inputs must preserve BF16 and compact f32 "
                "boundaries");
          }
        }
        continue;
      }
      ArrayRef<int64_t> domain = task.getDomainShape();
      const int64_t reductionAxis = domain.size() - 1;
      if (domain.size() < 3 || domain.size() > 9 ||
          task.getReductionDimensions() != ArrayRef<int64_t>({reductionAxis}) ||
          task.getOrderFree() || task.getInputBuffers().size() != 2 ||
          task.getOutputBuffers().size() != 1) {
        return task.emitOpError()
               << "Contract tasks require rank-three through rank-nine "
                  "domain, final reduction dimension, two inputs, and one "
                  "output";
      }
      auto lhs = cast<RankedTensorType>(
          buffers[task.getInputBuffers()[0]].getTensorType());
      auto rhs = cast<RankedTensorType>(
          buffers[task.getInputBuffers()[1]].getTensorType());
      auto result = cast<RankedTensorType>(
          buffers[task.getOutputBuffers()[0]].getTensorType());
      ArrayRef<int64_t> expectedResultShape = domain.drop_back();
      if (result.getRank() < 2 || result.getRank() > 8 ||
          lhs.getElementType() != rhs.getElementType() ||
          (!lhs.getElementType().isBF16() && !lhs.getElementType().isF32()) ||
          !result.getElementType().isF32() ||
          !inputShapesBindContractDomain(lhs, rhs, domain) ||
          result.getShape() != expectedResultShape) {
        return task.emitOpError()
               << "Contract buffers must bind "
               << cardinalName(domain.size() - 1)
               << " distinct parallel axes and one reduction axis with "
                  "equal bf16 or f32 input types";
      }
    } else {
      if (!task.getReductionDimensions().empty() || task.getOrderFree()) {
        return task.emitOpError(
            "Map tasks must not carry reduction or ordering metadata");
      }
      bool hasTensorResult = false;
      for (int64_t bufferOrdinal : task.getOutputBuffers()) {
        auto resultType =
            cast<RankedTensorType>(buffers[bufferOrdinal].getTensorType());
        hasTensorResult |= resultType.getRank() != 0;
      }
      if (task.getDomainShape().empty() == hasTensorResult) {
        return task.emitOpError(
            "empty Map domains must exactly identify scalar result tasks");
      }
    }
  }
  const bool hasCompositionTasks =
      contractFoldCompositionTask >= 0 || mapContractCompositionTask >= 0;
  if (hasCompositionTasks) {
    if (contractFoldCompositionTask < 0 || mapContractCompositionTask < 0 ||
        contractFoldCompositionTask >= mapContractCompositionTask) {
      return emitOpError(
          "generic composition requires the ordered Contract-to-Fold and "
          "Map-to-Contract semantic groups");
    }
    MaterializationTaskOp producer = tasks[contractFoldCompositionTask];
    MaterializationTaskOp consumer = tasks[mapContractCompositionTask];
    ArrayRef<int64_t> boundary = producer.getOutputBuffers();
    if (consumer.getInputBuffers().take_front(2) != boundary ||
        consumer.getDependencies() !=
            ArrayRef<int64_t>({contractFoldCompositionTask})) {
      return emitOpError(
          "generic composition must preserve the ordered full and compact "
          "boundary edge");
    }
    for (int64_t bufferOrdinal : boundary) {
      MaterializationBufferOp buffer = buffers[bufferOrdinal];
      if (buffer.getStorage() != MaterializationStorage::Temporary ||
          buffer.getLiveIn() || buffer.getLiveOut() ||
          optionalIntegerAttribute(buffer, "producer") !=
              contractFoldCompositionTask ||
          buffer.getConsumers() !=
              ArrayRef<int64_t>({mapContractCompositionTask}) ||
          buffer.getLifetimeStart() != contractFoldCompositionTask ||
          buffer.getLifetimeEnd() != mapContractCompositionTask) {
        return buffer.emitOpError(
            "composition boundary must be private across exactly one edge");
      }
    }
  }
  if (foldCount == 0 && contractCount == 0) {
    if (tasks.size() < 2) {
      return emitOpError("requires at least two Map tasks");
    }
    bool hasTemporary = false;
    for (MaterializationBufferOp buffer : buffers) {
      auto type = cast<RankedTensorType>(buffer.getTensorType());
      if (buffer.getStorage() == MaterializationStorage::External) {
        if (type.getRank() < 1 || type.getRank() > 8 ||
            !type.getElementType().isBF16()) {
          return buffer.emitOpError(
              "Map-only external buffers require rank-one through rank-eight "
              "bf16 tensors");
        }
        continue;
      }
      if (!type.getElementType().isF32()) {
        return buffer.emitOpError("Map-only temporary buffers require f32");
      }
      hasTemporary = true;
    }
    if (!hasTemporary) {
      return emitOpError("Map-only plans require a temporary f32 buffer");
    }
  }
  bool compactContractRegion = false;
  for (MaterializationTaskOp task : tasks) {
    auto strategy = task->getAttrOfType<StringAttr>("contract_region_strategy");
    compactContractRegion |= strategy && strategy.getValue() == "compact_stat";
  }
  if (contractRegionCount > 1 ||
      (contractCount != contractRegionCount && foldRegionCount != 0 &&
       !compactContractRegion) ||
      (foldRegionCount != 0 && !compactContractRegion && contractCount != 0) ||
      getSchemaVersion() !=
          (contractCount != 0 || foldRegionCount != 0 ? 2 : 1)) {
    return emitOpError(
        "materialization schema must equal the admitted task family");
  }

  for (auto [position, buffer] : llvm::enumerate(buffers)) {
    SmallVector<int64_t> consumers = integerArray(buffer.getConsumersAttr());
    if (consumers != actualConsumers[position]) {
      return buffer.emitOpError(
          "consumer list must equal the tasks that read the buffer");
    }
    int64_t expectedEnd = buffer.getLifetimeStart();
    if (!consumers.empty()) {
      expectedEnd = consumers.back();
    }
    if (buffer.getLiveOut()) {
      expectedEnd = tasks.size();
    } else if (consumers.empty() && !buffer.getLiveIn()) {
      return buffer.emitOpError(
          "produced buffers require a consumer or live-out use");
    }
    if (buffer.getLifetimeEnd() != expectedEnd) {
      return buffer.emitOpError(
          "lifetime end must equal the final consumer or plan exit");
    }
  }

  if (getFingerprint() != materializationPlanFingerprint(*this)) {
    return emitOpError(
        "fingerprint does not match the closed materialization plan");
  }
  return success();
}

std::string schedulePlanFingerprint(SchedulePlanOp plan) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  stream << "schema=" << plan.getSchemaVersion() << ';';
  Attribute(plan.getTargetAttr()).print(stream);
  stream << ';';
  Attribute(plan.getPolicyAttr()).print(stream);
  stream << ";source=" << plan.getSourcePlanFingerprint() << '\n';
  for (Operation &operation : plan.getBody().front()) {
    if (isa<SchedulePlanYieldOp>(operation)) {
      continue;
    }
    stream << operation.getName() << '{';
    for (NamedAttribute attribute : operation.getAttrs()) {
      stream << attribute.getName() << '=';
      attribute.getValue().print(stream);
      stream << ',';
    }
    stream << "}\n";
  }
  stream.flush();
  llvm::SHA256 digest;
  digest.update(normalized);
  return llvm::toHex(digest.final(), true);
}

namespace {

FailureOr<int64_t> staticElementCount(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t extent : shape) {
    if (extent <= 0 || count > std::numeric_limits<int64_t>::max() / extent) {
      return failure();
    }
    count *= extent;
  }
  return count;
}

int64_t roundUpToSubgroup(int64_t value) {
  constexpr int64_t kSubgroup = 32;
  return ceilDivPositive(value, kSubgroup) * kSubgroup;
}

struct Simt32Geometry {
  SmallVector<int64_t> grid;
  SmallVector<int64_t> tile;
  int64_t serialTiles;
  int64_t workgroupThreads;
  int64_t subgroupSize;
  int64_t scratchBytes;
  std::optional<int64_t> reductionAxis;
};

FailureOr<Simt32Geometry> simt32Geometry(ScheduleTaskKind kind,
                                         ArrayRef<int64_t> domain,
                                         NumericalPolicy policy) {
  constexpr int64_t kSubgroup = 32;
  constexpr int64_t kMaxThreads = 256;
  if (kind == ScheduleTaskKind::Scalar) {
    if (!domain.empty()) {
      return failure();
    }
    return Simt32Geometry{{1}, {}, 1, 1, kSubgroup, 0, std::nullopt};
  }
  if (kind == ScheduleTaskKind::Elementwise) {
    FailureOr<int64_t> elements = staticElementCount(domain);
    if (failed(elements)) {
      return failure();
    }
    int64_t tile = std::min(*elements, kMaxThreads);
    return Simt32Geometry{{ceilDivPositive(*elements, tile)},
                          {tile},
                          1,
                          roundUpToSubgroup(tile),
                          kSubgroup,
                          0,
                          std::nullopt};
  }
  if (kind == ScheduleTaskKind::Contract) {
    if (domain.size() < 3 || domain.size() > 9) {
      return failure();
    }
    FailureOr<int64_t> outputs = staticElementCount(domain.drop_back());
    if (failed(outputs) || *outputs > UINT32_MAX || domain.back() <= 0) {
      return failure();
    }
    const int64_t reductionAxis = domain.size() - 1;
    if (policy == NumericalPolicy::Fast) {
      const int64_t outputsPerBlock = std::min<int64_t>(*outputs, 8);
      return Simt32Geometry{{ceilDivPositive(*outputs, outputsPerBlock)},
                            {outputsPerBlock},
                            ceilDivPositive(domain.back(), kSubgroup),
                            outputsPerBlock * kSubgroup,
                            kSubgroup,
                            0,
                            reductionAxis};
    }
    const int64_t tile = std::min(*outputs, kMaxThreads);
    return Simt32Geometry{{ceilDivPositive(*outputs, tile)},
                          {tile},
                          domain.back(),
                          roundUpToSubgroup(tile),
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
              : staticElementCount(domain.drop_back());
  FailureOr<int64_t> outputs =
      rowFold ? staticElementCount(domain.drop_back())
              : FailureOr<int64_t>(domain.back());
  if (failed(reduction) || failed(outputs) || *outputs > UINT32_MAX) {
    return failure();
  }
  int64_t tile = std::min(*reduction, kMaxThreads);
  int64_t threads = roundUpToSubgroup(tile);
  SmallVector<int64_t> tileShape(domain.size(), 1);
  tileShape[reductionAxis] = tile;
  return Simt32Geometry{{*outputs},
                        std::move(tileShape),
                        ceilDivPositive(*reduction, tile),
                        threads,
                        kSubgroup,
                        threads * static_cast<int64_t>(sizeof(float)),
                        reductionAxis};
}

FailureOr<Simt32Geometry> contractRegionGeometry(ScheduleTaskOp task,
                                                 NumericalPolicy policy) {
  auto strategy = task->getAttrOfType<StringAttr>("contract_region_strategy");
  auto cost = task->getAttrOfType<DictionaryAttr>("contract_region_cost");
  if (!strategy || !cost || task.getKind() != ScheduleTaskKind::Contract)
    return failure();
  auto rows = cost.getAs<IntegerAttr>("rows");
  auto reduction = cost.getAs<IntegerAttr>("reduction");
  auto tile = cost.getAs<IntegerAttr>("output_tile");
  auto tiles = cost.getAs<IntegerAttr>("output_tiles");
  if (!rows || !reduction || !tile || !tiles || rows.getInt() <= 0 ||
      reduction.getInt() <= 0 || tile.getInt() <= 0 || tiles.getInt() <= 0)
    return failure();
  const bool inlineFold = strategy.getValue() == "inline_fold";
  const bool compactStat = strategy.getValue() == "compact_stat";
  if (!inlineFold && !compactStat)
    return failure();
  const bool fast = policy == NumericalPolicy::Fast;
  const int64_t threads =
      fast ? tile.getInt() * 32 : roundUpToSubgroup(tile.getInt());
  const int64_t scratch = inlineFold
                              ? roundUpToSubgroup(reduction.getInt()) *
                                    static_cast<int64_t>(sizeof(float))
                              : std::min<int64_t>(reduction.getInt(), 256) * 2;
  if (threads <= 0 || threads > 1024 || scratch < 0 || scratch > 16 * 1024)
    return failure();
  return Simt32Geometry{{rows.getInt(), tiles.getInt()},
                        {tile.getInt()},
                        fast ? ceilDivPositive(reduction.getInt(), 32)
                             : reduction.getInt(),
                        threads,
                        32,
                        scratch,
                        static_cast<int64_t>(task.getDomainShape().size()) - 1};
}

} // namespace

LogicalResult SchedulePlanOp::verifyRegions() {
  if (!getOperation()->getDiscardableAttrs().empty()) {
    return emitOpError("does not permit discardable attributes");
  }
  if ((getSchemaVersion() != 1 && getSchemaVersion() != 2) ||
      getTarget() != ScheduleTarget::Simt32) {
    return emitOpError("requires schedule schema 1 or 2 and target simt32");
  }
  if (!isLowerHexDigest(getSourcePlanFingerprint()) ||
      !isLowerHexDigest(getFingerprint())) {
    return emitOpError("fingerprints must be lowercase SHA-256 digests");
  }
  Region &body = getBody();
  if (body.empty() || !llvm::hasSingleElement(body) ||
      !isa<SchedulePlanYieldOp>(body.front().getTerminator())) {
    return emitOpError(
        "requires one block terminated by shuttle.schedule_plan_yield");
  }

  SmallVector<ScheduleBufferOp> buffers;
  SmallVector<ScheduleTaskOp> tasks;
  bool sawTask = false;
  for (Operation &operation : body.front().without_terminator()) {
    if (auto buffer = dyn_cast<ScheduleBufferOp>(operation)) {
      if (sawTask) {
        return buffer.emitOpError("schedule buffers must precede tasks");
      }
      if (!buffer->getDiscardableAttrs().empty()) {
        return buffer.emitOpError("does not permit discardable attributes");
      }
      buffers.push_back(buffer);
      continue;
    }
    if (auto task = dyn_cast<ScheduleTaskOp>(operation)) {
      sawTask = true;
      if (!task->getDiscardableAttrs().empty()) {
        return task.emitOpError("does not permit discardable attributes");
      }
      tasks.push_back(task);
      continue;
    }
    return emitOpError("body may contain only schedule buffers and tasks");
  }
  if (buffers.empty() || tasks.empty()) {
    return emitOpError("requires at least one schedule buffer and task");
  }
  if (!body.front().getTerminator()->getDiscardableAttrs().empty()) {
    return body.front().getTerminator()->emitOpError(
        "does not permit discardable attributes");
  }

  for (auto [ordinal, buffer] : llvm::enumerate(buffers)) {
    if (buffer.getOrdinal() != static_cast<int64_t>(ordinal) ||
        buffer.getSourceBuffer() != static_cast<int64_t>(ordinal)) {
      return buffer.emitOpError(
          "schedule buffer and source ordinals must be contiguous");
    }
    auto type = dyn_cast<RankedTensorType>(buffer.getTensorType());
    if (!type || !type.hasStaticShape() ||
        failed(staticElementCount(type.getShape())) ||
        (!type.getElementType().isBF16() && !type.getElementType().isF32())) {
      return buffer.emitOpError(
          "requires a positive static bf16 or f32 tensor type");
    }
    ScheduleBufferIndexing expectedIndexing =
        type.getRank() == 0 ? ScheduleBufferIndexing::Scalar
                            : ScheduleBufferIndexing::Lexicographic;
    SmallVector<int64_t> expectedOrder;
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      expectedOrder.push_back(axis);
    }
    if (buffer.getIndexing() != expectedIndexing ||
        buffer.getIterationOrder() != ArrayRef<int64_t>(expectedOrder)) {
      return buffer.emitOpError(
          "buffer indexing must equal logical tensor rank");
    }
    if (buffer.getLifetimeStart() < 0 ||
        buffer.getLifetimeEnd() < buffer.getLifetimeStart() ||
        buffer.getLifetimeEnd() > static_cast<int64_t>(tasks.size())) {
      return buffer.emitOpError("has an invalid schedule lifetime interval");
    }
  }

  unsigned foldCount = 0;
  unsigned contractCount = 0;
  for (auto [ordinal, task] : llvm::enumerate(tasks)) {
    if (task.getOrdinal() != static_cast<int64_t>(ordinal) ||
        task.getSourceTask() != static_cast<int64_t>(ordinal)) {
      return task.emitOpError(
          "schedule source tasks must be unique structural ordinals");
    }
    if (!isLowerHexDigest(task.getSemanticFingerprint())) {
      return task.emitOpError(
          "semantic fingerprint must be a lowercase SHA-256 digest");
    }
    for (ArrayRef<int64_t> references :
         {task.getInputBuffers(), task.getOutputBuffers()}) {
      for (int64_t buffer : references) {
        if (buffer < 0 || buffer >= buffers.size()) {
          return task.emitOpError("schedule buffer reference is out of range");
        }
      }
    }
    llvm::SmallDenseSet<int64_t> dependencies;
    for (int64_t dependency : task.getDependencies()) {
      if (dependency < 0 || dependency >= static_cast<int64_t>(ordinal) ||
          !dependencies.insert(dependency).second) {
        return task.emitOpError(
            "dependencies must be unique earlier schedule tasks");
      }
    }
    const bool isFold = task.getKind() == ScheduleTaskKind::RowFold ||
                        task.getKind() == ScheduleTaskKind::ColumnFold;
    const bool isContract = task.getKind() == ScheduleTaskKind::Contract;
    const int64_t expectedFoldAxis =
        task.getKind() == ScheduleTaskKind::RowFold
            ? static_cast<int64_t>(task.getDomainShape().size()) - 1
            : 0;
    if (isFold && task.getReductionAxis() != expectedFoldAxis) {
      return task.emitOpError(
          "Fold schedule kind must match its reduction axis");
    }
    std::optional<ScheduleReductionOrder> expectedOrder;
    if (isFold) {
      expectedOrder = ScheduleReductionOrder::TreeAssociationFreeLeafOrderFixed;
    } else if (isContract) {
      expectedOrder = getPolicy() == NumericalPolicy::Fast
                          ? ScheduleReductionOrder::LeafOrderFree
                          : ScheduleReductionOrder::SerialAscendingFmaRnF32;
    }
    if (task.getReductionOrder() != expectedOrder) {
      return task.emitOpError(
          isContract ? "Contract reduction order must equal the policy-bound "
                       "Contract semantics"
                     : "Fold reduction order must equal bound Fold semantics");
    }
    const bool hasContractStrategy = task->hasAttr("contract_region_strategy");
    const bool hasContractCost = task->hasAttr("contract_region_cost");
    if (hasContractStrategy != hasContractCost ||
        (hasContractStrategy && !isContract)) {
      return task.emitOpError(
          "Contract-region schedule metadata requires a Contract task");
    }
    FailureOr<Simt32Geometry> expected =
        hasContractStrategy
            ? contractRegionGeometry(task, getPolicy())
            : simt32Geometry(task.getKind(), task.getDomainShape(),
                             getPolicy());
    if (failed(expected) ||
        task.getGridShape() != ArrayRef<int64_t>(expected->grid) ||
        task.getTileShape() != ArrayRef<int64_t>(expected->tile) ||
        task.getSerialTiles() != expected->serialTiles ||
        task.getWorkgroupThreads() != expected->workgroupThreads ||
        task.getSubgroupSize() != expected->subgroupSize ||
        task.getScratchBytes() != expected->scratchBytes ||
        task.getReductionAxis() != expected->reductionAxis) {
      return task.emitOpError(
          "schedule geometry must equal the SIMT32 derivation");
    }
    if (isFold) {
      ++foldCount;
    } else if (isContract) {
      ++contractCount;
    }
  }
  if (foldCount == 0 && contractCount == 0 && tasks.size() < 2) {
    return emitOpError("requires at least two Map schedule tasks");
  }
  if ((contractCount != 0 && getSchemaVersion() != 2) ||
      (contractCount == 0 && getSchemaVersion() != 1 &&
       getSchemaVersion() != 2)) {
    return emitOpError("schedule schema must equal the admitted task family");
  }
  if (getFingerprint() != schedulePlanFingerprint(*this)) {
    return emitOpError("fingerprint does not match the closed schedule plan");
  }
  return success();
}

namespace {

std::string digestText(StringRef text) {
  llvm::SHA256 digest;
  digest.update(text);
  return llvm::toHex(digest.final(), true);
}

template <typename PlanOp, typename TerminatorOp>
std::string executablePlanFingerprint(PlanOp plan, StringRef header) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  stream << header << '\n';
  for (NamedAttribute attribute : plan->getAttrs()) {
    if (attribute.getName().strref() == "fingerprint") {
      continue;
    }
    stream << attribute.getName() << '=';
    attribute.getValue().print(stream);
    stream << '\n';
  }
  if constexpr (!std::is_same_v<PlanOp, ExecutableBundleOp>) {
    for (Operation &operation : plan.getBody().front()) {
      if (isa<TerminatorOp>(operation)) {
        continue;
      }
      stream << operation.getName() << '{';
      for (NamedAttribute attribute : operation.getAttrs()) {
        stream << attribute.getName() << '=';
        attribute.getValue().print(stream);
        stream << ',';
      }
      stream << "}\n";
    }
  }
  stream.flush();
  return digestText(normalized);
}

LogicalResult verifyExecutableAccessArray(Operation *owner, ArrayAttr accesses,
                                          size_t expectedSize,
                                          ExecutableAccess expected) {
  if (accesses.size() != expectedSize) {
    return owner->emitOpError("access list must match buffer references");
  }
  for (Attribute access : accesses) {
    auto typed = dyn_cast<ExecutableAccessAttr>(access);
    if (!typed || typed.getValue() != expected) {
      return owner->emitOpError("entrypoint access mode is invalid");
    }
  }
  return success();
}

FailureOr<int64_t> executableElementBytes(Type elementType) {
  if (elementType.isBF16()) {
    return 2;
  }
  if (elementType.isF32()) {
    return 4;
  }
  return failure();
}

} // namespace

std::string executableCodeDigest(ArrayRef<int8_t> code) {
  llvm::SHA256 digest;
  digest.update(ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(code.data()), code.size()));
  return llvm::toHex(digest.final(), true);
}

std::string deviceModuleFingerprint(DeviceModuleOp module) {
  StringRef version = "shuttle.device_module.v1";
  if (module.getCodeFormat() == ExecutableCodeFormat::CpuBytecodeV2) {
    version = "shuttle.device_module.v2";
  } else if (module.getCodeFormat() == ExecutableCodeFormat::CudaPtxSm90V1) {
    version = "shuttle.device_module.v3";
  } else if (module.getCodeFormat() == ExecutableCodeFormat::CudaPtxSm100V1) {
    version = "shuttle.device_module.v4";
  }
  return executablePlanFingerprint<DeviceModuleOp, DeviceModuleYieldOp>(
      module, version);
}

std::string invocationAbiFingerprint(InvocationAbiOp abi) {
  return executablePlanFingerprint<InvocationAbiOp, InvocationAbiYieldOp>(
      abi, abi.getSchemaVersion() == 3 ? "shuttle.invocation_abi.v3"
                                       : "shuttle.invocation_abi.v2");
}

std::string executableBundleFingerprint(ExecutableBundleOp bundle) {
  return executablePlanFingerprint<ExecutableBundleOp, Operation>(
      bundle, bundle.getSchemaVersion() == 2 ? "shuttle.executable_bundle.v2"
                                             : "shuttle.executable_bundle.v1");
}

LogicalResult DeviceModuleOp::verifyRegions() {
  if (!getOperation()->getDiscardableAttrs().empty()) {
    return emitOpError("does not permit discardable attributes");
  }
  const bool validVersionPair =
      (getSchemaVersion() == 1 &&
       getCodeFormat() == ExecutableCodeFormat::CpuBytecodeV1) ||
      (getSchemaVersion() == 2 &&
       getCodeFormat() == ExecutableCodeFormat::CpuBytecodeV2) ||
      (getSchemaVersion() == 3 &&
       getCodeFormat() == ExecutableCodeFormat::CudaPtxSm90V1) ||
      (getSchemaVersion() == 4 &&
       getCodeFormat() == ExecutableCodeFormat::CudaPtxSm100V1);
  if (!validVersionPair) {
    return emitOpError(
        "requires a matching device-module and executable-code version");
  }
  if (getCodeFormat() == ExecutableCodeFormat::CpuBytecodeV2 &&
      getPolicy() != NumericalPolicy::SourceOrdered) {
    return emitOpError("requires source_ordered policy for CPU bytecode v2");
  }
  const bool gpu = getCodeFormat() == ExecutableCodeFormat::CudaPtxSm90V1 ||
                   getCodeFormat() == ExecutableCodeFormat::CudaPtxSm100V1;
  if (!isLowerHexDigest(getSourceScheduleFingerprint()) ||
      !isLowerHexDigest(getCodeDigest()) ||
      !isLowerHexDigest(getFingerprint())) {
    return emitOpError("fingerprints must be lowercase SHA-256 digests");
  }
  if (getCode().empty() || getCodeDigest() != executableCodeDigest(getCode())) {
    return emitOpError("code SHA-256 does not match code bytes");
  }
  Region &body = getBody();
  if (body.empty() || !llvm::hasSingleElement(body) ||
      !isa<DeviceModuleYieldOp>(body.front().getTerminator())) {
    return emitOpError(
        "requires one block terminated by shuttle.device_module_yield");
  }
  int64_t expectedOffset = 0;
  int64_t ordinal = 0;
  llvm::DenseMap<int64_t, int64_t> producers;
  for (Operation &operation : body.front().without_terminator()) {
    auto entry = dyn_cast<DeviceEntryOp>(operation);
    if (!entry) {
      return emitOpError("body may contain only device entries");
    }
    if (!entry->getDiscardableAttrs().empty()) {
      return entry.emitOpError("does not permit discardable attributes");
    }
    if (entry.getOrdinal() != ordinal || entry.getSourceTask() != ordinal++) {
      return entry.emitOpError(
          "entry and source task ordinals must be contiguous");
    }
    if (entry.getCodeOffset() != expectedOffset || entry.getCodeLength() <= 0 ||
        entry.getCodeLength() >
            static_cast<int64_t>(getCode().size()) - expectedOffset) {
      return entry.emitOpError(
          "entry byte range must partition the code object");
    }
    expectedOffset += entry.getCodeLength();
    ArrayRef<int8_t> entryCode =
        getCode().slice(entry.getCodeOffset(), entry.getCodeLength());
    if (gpu && entry.getCodeDigest() != executableCodeDigest(entryCode)) {
      return entry.emitOpError("entrypoint code SHA-256 is invalid");
    }
    if (!gpu && entry.getCodeDigest() != getCodeDigest()) {
      return entry.emitOpError(
          "entrypoint must bind the verified device-module code object");
    }
    if (failed(verifyExecutableAccessArray(entry, entry.getInputAccesses(),
                                           entry.getInputBuffers().size(),
                                           ExecutableAccess::Read)) ||
        failed(verifyExecutableAccessArray(entry, entry.getOutputAccesses(),
                                           entry.getOutputBuffers().size(),
                                           ExecutableAccess::Write))) {
      return failure();
    }
    llvm::SmallDenseSet<int64_t> dependencies;
    SmallVector<int64_t> expectedDependencies;
    for (int64_t buffer : entry.getInputBuffers()) {
      if (buffer < 0) {
        return entry.emitOpError("input buffer ordinal must be nonnegative");
      }
      auto producer = producers.find(buffer);
      if (producer != producers.end() &&
          !llvm::is_contained(expectedDependencies, producer->second)) {
        expectedDependencies.push_back(producer->second);
      }
    }
    llvm::sort(expectedDependencies);
    for (int64_t dependency : entry.getDependencies()) {
      if (dependency < 0 || dependency >= entry.getOrdinal() ||
          !dependencies.insert(dependency).second) {
        return entry.emitOpError(
            "dependencies must be unique earlier entry ordinals");
      }
    }
    if (entry.getDependencies() != ArrayRef<int64_t>(expectedDependencies)) {
      return entry.emitOpError(
          "dependencies must equal entrypoint buffer producers");
    }
    for (int64_t buffer : entry.getOutputBuffers()) {
      if (buffer < 0 ||
          !producers.try_emplace(buffer, entry.getOrdinal()).second) {
        return entry.emitOpError(
            "output buffers must be nonnegative and have one producer");
      }
    }
    if (entry.getPredication() != ExecutablePredication::None &&
        entry.getPredication() != ExecutablePredication::DomainBounds) {
      return entry.emitOpError("uses an unsupported predication policy");
    }
    if (!gpu) {
      if (entry.getGrid() || entry.getBlock() ||
          entry.getDynamicSharedBytes() || entry.getKernelArity()) {
        return entry.emitOpError(
            "CPU entries do not permit CUDA launch attributes");
      }
      continue;
    }
    if (!entry.getGrid() || !entry.getBlock() ||
        !entry.getDynamicSharedBytes() || !entry.getKernelArity() ||
        entry.getGrid()->size() != 3 || entry.getBlock()->size() != 3) {
      return entry.emitOpError("CUDA entry requires complete launch geometry");
    }
    uint64_t threads = 1;
    for (int64_t extent : *entry.getGrid()) {
      if (extent <= 0 || extent > UINT32_MAX) {
        return entry.emitOpError("CUDA grid extent is unsupported");
      }
    }
    for (int64_t extent : *entry.getBlock()) {
      if (extent <= 0 || extent > UINT16_MAX ||
          static_cast<uint64_t>(extent) > 1024 / threads) {
        return entry.emitOpError("CUDA block extent is unsupported");
      }
      threads *= extent;
    }
    if (*entry.getDynamicSharedBytes() < 0 ||
        *entry.getDynamicSharedBytes() > 16 * 1024 ||
        *entry.getKernelArity() !=
            static_cast<int64_t>(entry.getInputBuffers().size() +
                                 entry.getOutputBuffers().size())) {
      return entry.emitOpError("CUDA scratch or kernel arity is invalid");
    }
    StringRef ptx(reinterpret_cast<const char *>(entryCode.data()),
                  entryCode.size());
    const StringRef expectedHeader =
        getCodeFormat() == ExecutableCodeFormat::CudaPtxSm90V1
            ? ".version 8.0\n.target sm_90\n"
            : ".version 8.7\n.target sm_100\n";
    if (ptx.empty() || !ptx.ends_with("\n") ||
        !ptx.starts_with(expectedHeader) ||
        ptx.find(".address_size 64") == StringRef::npos ||
        ptx.find(".visible .entry shuttle_entry(") == StringRef::npos ||
        ptx.find("ret;") == StringRef::npos) {
      return entry.emitOpError(
          "CUDA PTX slice is not the closed SM90 form or closed SM100 form");
    }
    for (char character : ptx) {
      if (character != '\n' && character != '\t' &&
          (character < 0x20 || character > 0x7e)) {
        return entry.emitOpError("CUDA PTX slice must be printable ASCII");
      }
    }
  }
  if (ordinal == 0 ||
      expectedOffset != static_cast<int64_t>(getCode().size())) {
    return emitOpError("device entries must exactly cover the code object");
  }
  if (!body.front().getTerminator()->getDiscardableAttrs().empty()) {
    return body.front().getTerminator()->emitOpError(
        "does not permit discardable attributes");
  }
  if (getFingerprint() != deviceModuleFingerprint(*this)) {
    return emitOpError("fingerprint does not match the closed device module");
  }
  return success();
}

LogicalResult InvocationAbiOp::verifyRegions() {
  if (!getOperation()->getDiscardableAttrs().empty()) {
    return emitOpError("does not permit discardable attributes");
  }
  if ((getSchemaVersion() != 2 && getSchemaVersion() != 3) ||
      !isLowerHexDigest(getSourcePlanFingerprint()) ||
      !isLowerHexDigest(getSourceScheduleFingerprint()) ||
      !isLowerHexDigest(getFingerprint())) {
    return emitOpError(
        "requires ABI schema 2 or 3 and lowercase SHA-256 fingerprints");
  }
  Region &body = getBody();
  if (body.empty() || !llvm::hasSingleElement(body) ||
      !isa<InvocationAbiYieldOp>(body.front().getTerminator())) {
    return emitOpError(
        "requires one block terminated by shuttle.invocation_abi_yield");
  }
  int64_t ordinal = 0;
  llvm::SmallDenseSet<int64_t> aliasGroups;
  llvm::SmallDenseSet<int64_t> operandBindings;
  llvm::SmallDenseSet<int64_t> resultBindings;
  for (Operation &operation : body.front().without_terminator()) {
    auto slot = dyn_cast<InvocationSlotOp>(operation);
    if (!slot) {
      return emitOpError("body may contain only invocation slots");
    }
    if (!slot->getDiscardableAttrs().empty()) {
      return slot.emitOpError("does not permit discardable attributes");
    }
    if (slot.getOrdinal() != ordinal || slot.getSourceBuffer() != ordinal++) {
      return slot.emitOpError(
          "slot and source buffer ordinals must be contiguous");
    }
    auto type = dyn_cast<RankedTensorType>(slot.getTensorType());
    FailureOr<int64_t> elementBytes =
        type ? executableElementBytes(type.getElementType())
             : FailureOr<int64_t>(failure());
    FailureOr<int64_t> elements = type ? staticElementCount(type.getShape())
                                       : FailureOr<int64_t>(failure());
    if (!type || !type.hasStaticShape() || failed(elementBytes) ||
        failed(elements) ||
        *elements > std::numeric_limits<int64_t>::max() / *elementBytes) {
      return slot.emitOpError("requires a bounded static bf16 or f32 tensor");
    }
    const int64_t requiredBytes = *elements * *elementBytes;
    SmallVector<int64_t> strides(type.getRank());
    int64_t stride = *elementBytes;
    for (int64_t axis = type.getRank(); axis > 0; --axis) {
      strides[axis - 1] = stride;
      stride *= type.getDimSize(axis - 1);
    }
    if (slot.getRequiredBytes() != requiredBytes ||
        slot.getStrides() != ArrayRef<int64_t>(strides)) {
      return slot.emitOpError("required bytes and strides must encode "
                              "contiguous row-major storage");
    }
    const ExecutableAddressSpace expectedAddressSpace =
        getSchemaVersion() == 3 ? ExecutableAddressSpace::Device
                                : ExecutableAddressSpace::Host;
    if (slot.getOffset() != 0 || slot.getAlignment() != *elementBytes ||
        slot.getAddressSpace() != expectedAddressSpace) {
      if (getSchemaVersion() == 3)
        return slot.emitOpError(
            "closed GPU ABI requires device storage, zero offset, and "
            "natural alignment");
      return slot.emitOpError(
          "closed CPU ABI requires host storage, zero offset, and natural "
          "alignment");
    }
    if (!aliasGroups.insert(slot.getAliasGroup()).second ||
        slot.getAliasGroup() != slot.getOrdinal())
      return slot.emitOpError("closed executable ABI forbids aliasing");
    if (slot.getReuseGroup() < 0 ||
        (getSchemaVersion() == 2 &&
         slot.getReuseGroup() != slot.getOrdinal()) ||
        (slot.getStorage() == MaterializationStorage::External &&
         slot.getReuseGroup() != slot.getOrdinal()))
      return slot.emitOpError(
          "reuse group is invalid for the slot storage and ABI schema");
    if (slot.getStorage() == MaterializationStorage::External &&
        slot.getAccess() == ExecutableAccess::ReadWrite) {
      return slot.emitOpError("external slots must be read-only or write-only");
    }
    if (slot.getStorage() == MaterializationStorage::Temporary &&
        slot.getAccess() != ExecutableAccess::ReadWrite) {
      return slot.emitOpError("temporary slots must be read-write");
    }
    ExecutableBindingKind expectedBinding = ExecutableBindingKind::None;
    llvm::SmallDenseSet<int64_t> *bindingIndices = nullptr;
    if (slot.getStorage() == MaterializationStorage::External &&
        slot.getAccess() == ExecutableAccess::Read) {
      expectedBinding = ExecutableBindingKind::Operand;
      bindingIndices = &operandBindings;
    } else if (slot.getStorage() == MaterializationStorage::External &&
               slot.getAccess() == ExecutableAccess::Write) {
      expectedBinding = ExecutableBindingKind::Result;
      bindingIndices = &resultBindings;
    }
    std::optional<int64_t> bindingIndex = slot.getBindingIndex();
    if (slot.getBinding() != expectedBinding ||
        ((expectedBinding == ExecutableBindingKind::None) != !bindingIndex) ||
        (bindingIndex &&
         (*bindingIndex < 0 || !bindingIndices->insert(*bindingIndex).second))) {
      return slot.emitOpError(
          "binding kind and index must uniquely identify the external ABI");
    }
  }
  auto isContiguous = [](const llvm::SmallDenseSet<int64_t> &indices) {
    for (int64_t index = 0; index < static_cast<int64_t>(indices.size());
         ++index)
      if (!indices.contains(index))
        return false;
    return true;
  };
  if (!isContiguous(operandBindings) || !isContiguous(resultBindings))
    return emitOpError("external binding indices must be contiguous");
  if (ordinal == 0) {
    return emitOpError("requires at least one invocation slot");
  }
  if (!body.front().getTerminator()->getDiscardableAttrs().empty()) {
    return body.front().getTerminator()->emitOpError(
        "does not permit discardable attributes");
  }
  if (getFingerprint() != invocationAbiFingerprint(*this)) {
    return emitOpError("fingerprint does not match the closed invocation ABI");
  }
  return success();
}

LogicalResult ExecutableBundleOp::verify() {
  if (!getOperation()->getDiscardableAttrs().empty()) {
    return emitOpError("does not permit discardable attributes");
  }
  const bool validVersion =
      (getSchemaVersion() == 1 &&
       getCompletion() == ExecutableCompletion::Synchronous) ||
      (getSchemaVersion() == 2 &&
       getCompletion() == ExecutableCompletion::StreamOrdered);
  if (!validVersion || !isLowerHexDigest(getSourceScheduleFingerprint()) ||
      !isLowerHexDigest(getDeviceModuleFingerprint()) ||
      !isLowerHexDigest(getInvocationAbiFingerprint()) ||
      !isLowerHexDigest(getFingerprint())) {
    return emitOpError(
        "requires a matching bundle schema/completion and SHA-256 bindings");
  }
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) {
    return emitOpError("must be nested in a builtin module");
  }
  if (getSchemaVersion() == 1 &&
      (llvm::range_size(module.getOps<DeviceModuleOp>()) != 1 ||
       llvm::range_size(module.getOps<InvocationAbiOp>()) != 1 ||
       llvm::range_size(module.getOps<ExecutableBundleOp>()) != 1)) {
    return emitOpError(
        "requires exactly one device module, invocation ABI, and bundle root");
  }
  DeviceModuleOp deviceModule;
  for (DeviceModuleOp candidate : module.getOps<DeviceModuleOp>()) {
    if (candidate.getFingerprint() != getDeviceModuleFingerprint())
      continue;
    if (deviceModule)
      return emitOpError("device-module fingerprint is not unique");
    deviceModule = candidate;
  }
  InvocationAbiOp abi;
  for (InvocationAbiOp candidate : module.getOps<InvocationAbiOp>()) {
    if (candidate.getFingerprint() != getInvocationAbiFingerprint())
      continue;
    if (abi)
      return emitOpError("invocation-ABI fingerprint is not unique");
    abi = candidate;
  }
  if (!deviceModule || !abi)
    return emitOpError(getSchemaVersion() == 1
                           ? "invalid closed CPU executable bundle"
                           : "invalid closed GPU executable bundle");
  if (getSourceScheduleFingerprint() !=
          deviceModule.getSourceScheduleFingerprint() ||
      getSourceScheduleFingerprint() != abi.getSourceScheduleFingerprint() ||
      getDeviceModuleFingerprint() != deviceModule.getFingerprint() ||
      getInvocationAbiFingerprint() != abi.getFingerprint() ||
      getFingerprint() != executableBundleFingerprint(*this)) {
    return emitOpError(getSchemaVersion() == 1
                           ? "invalid closed CPU executable bundle"
                           : "invalid closed GPU executable bundle");
  }
  const bool gpuBundle = getSchemaVersion() == 2;
  const bool gpuDevice = deviceModule.getSchemaVersion() == 3 ||
                         deviceModule.getSchemaVersion() == 4;
  if (gpuBundle != gpuDevice || gpuBundle != (abi.getSchemaVersion() == 3)) {
    return emitOpError("bundle schema does not match its device and ABI roots");
  }
  return success();
}

} // namespace mlir::shuttle
