// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/IR/ShuttleDialect.h"

#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
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

FailureOr<presburger::IntegerPolyhedron>
boundedEventDomain(ArrayRef<int64_t> shape,
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
  if (!predicate)
    return domain;
  if (predicate->getNumDims() != shape.size() ||
      predicate->getNumSymbols() != 0)
    return failure();
  std::vector<SmallVector<int64_t, 8>> flattened;
  FlatLinearConstraints localConstraints;
  if (failed(
          getFlattenedAffineExprs(*predicate, &flattened, &localConstraints)))
    return failure();
  FlatLinearValueConstraints predicateConstraints(*predicate);
  return domain.intersect(predicateConstraints);
}

FailureOr<presburger::MultiAffineFunction> multiAffineFunction(AffineMap map) {
  presburger::MultiAffineFunction function(
      presburger::PresburgerSpace::getRelationSpace(),
      presburger::IntMatrix(/*rows=*/0, /*columns=*/1));
  if (failed(getMultiAffineFunctionFromMap(map, function)))
    return failure();
  return function;
}

FailureOr<bool> affineMapIsInBounds(AffineMap map,
                                    const presburger::IntegerPolyhedron &domain,
                                    ArrayRef<int64_t> rangeShape) {
  if (map.getNumDims() != domain.getNumDimVars() || map.getNumSymbols() != 0 ||
      map.getNumResults() != rangeShape.size() || rangeShape.empty() ||
      llvm::any_of(rangeShape, [](int64_t extent) { return extent <= 0; }))
    return failure();
  FailureOr<presburger::MultiAffineFunction> function =
      multiAffineFunction(map);
  if (failed(function))
    return failure();
  presburger::IntegerRelation graph = function->getAsRelation();
  graph.intersectDomain(domain);
  presburger::IntegerPolyhedron image = graph.getRangeSet();
  presburger::IntegerPolyhedron bounds(
      presburger::PresburgerSpace::getSetSpace(rangeShape.size()));
  for (auto [axis, extent] : llvm::enumerate(rangeShape)) {
    bounds.addBound(presburger::BoundType::LB, axis, 0);
    bounds.addBound(presburger::BoundType::UB, axis, extent - 1);
  }
  return image.isSubsetOf(bounds);
}

LogicalResult verifyEventIndexAndSlotRanges(EventTensorOp event,
                                            ScheduleTaskOp producer,
                                            ScheduleTaskOp consumer) {
  FailureOr<presburger::IntegerPolyhedron> domain = boundedEventDomain(
      event.getRelationDomainShape(), event.getRelationPredicate());
  if (failed(domain))
    return event.emitOpError("has an unsupported or unbounded relation domain");
  FailureOr<bool> producerInBounds = affineMapIsInBounds(
      event.getProducerIndices(), *domain, producer.getGridShape());
  if (failed(producerInBounds) || !*producerInBounds)
    return event.emitOpError("producer index map escapes the producer grid");
  FailureOr<bool> consumerInBounds = affineMapIsInBounds(
      event.getConsumerIndices(), *domain, consumer.getGridShape());
  if (failed(consumerInBounds) || !*consumerInBounds)
    return event.emitOpError("consumer index map escapes the consumer grid");
  const uint64_t storageCapacity = event.getStorageCapacity();
  if (storageCapacity == 0 ||
      storageCapacity >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return event.emitOpError("requires positive Event storage capacity");
  std::array<int64_t, 1> storageShape{static_cast<int64_t>(storageCapacity)};
  FailureOr<bool> slotInBounds =
      affineMapIsInBounds(event.getStorageSlot(), *domain, storageShape);
  if (failed(slotInBounds) || !*slotInBounds)
    return event.emitOpError("storage-slot map escapes Event storage");
  return success();
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
  AffineMap expectedReshapeInput;
  if (map.getSemantics() == MapSemantics::Reshape) {
    auto input = dyn_cast<RankedTensorType>(map.getInputs().front().getType());
    auto result =
        dyn_cast<RankedTensorType>(map.getResults().front().getType());
    if (!input || !result || !input.hasStaticShape() ||
        !result.hasStaticShape() ||
        input.getNumElements() != result.getNumElements()) {
      return map.emitOpError("requires equal-size static reshape types");
    }
    MLIRContext *context = map.getContext();
    AffineExpr linear = getAffineConstantExpr(0, context);
    for (auto [dimension, extent] : llvm::enumerate(result.getShape()))
      linear = linear * extent + getAffineDimExpr(dimension, context);
    SmallVector<AffineExpr> coordinates(input.getRank());
    int64_t stride = 1;
    for (int64_t dimension = input.getRank() - 1; dimension >= 0; --dimension) {
      int64_t extent = input.getDimSize(dimension);
      coordinates[dimension] = linear.floorDiv(stride) % extent;
      stride *= extent;
    }
    expectedReshapeInput =
        AffineMap::get(result.getRank(), 0, coordinates, context);
  }
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
    if (!isResultMap && map.getSemantics() == MapSemantics::Reshape) {
      if (indexingMap != expectedReshapeInput)
        return map.emitOpError(
            "reshape input indexing must preserve contiguous linear order");
      continue;
    }
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

LogicalResult verifyFoldIndexingMaps(FoldOp fold, TypeRange indexedTypes) {
  ArrayAttr indexingMaps = fold.getIndexingMaps();
  if (failed(verifyIndexingMaps(fold, indexingMaps, indexedTypes))) {
    return failure();
  }
  if (indexingMaps.empty()) {
    return fold.emitOpError("requires non-empty indexing maps");
  }

  AffineMap domain = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  llvm::SmallDenseSet<int64_t> reductionDimensions;
  for (int64_t dimension : fold.getReductionDimensions()) {
    if (dimension < 0 ||
        dimension >= static_cast<int64_t>(domain.getNumDims()) ||
        !reductionDimensions.insert(dimension).second) {
      return fold.emitOpError(
          "reduction dimensions must be unique dimensions of the indexing "
          "domain");
    }
  }

  SmallVector<int64_t> domainExtents(domain.getNumDims(), ShapedType::kDynamic);
  const size_t inputCount = fold.getInputs().size();
  size_t position = 0;
  for (auto [mapAttribute, indexedTypeValue] :
       llvm::zip_equal(indexingMaps, indexedTypes)) {
    auto indexedType = dyn_cast<RankedTensorType>(indexedTypeValue);
    AffineMap map = cast<AffineMapAttr>(mapAttribute).getValue();
    if (!indexedType) {
      return fold.emitOpError("requires ranked tensor inputs and results");
    }
    if (map.getNumSymbols() != 0 || !map.isProjectedPermutation()) {
      return fold.emitOpError(
          "indexing maps must be projected permutations of direct domain "
          "dimensions without symbols");
    }

    llvm::SmallDenseSet<unsigned> mappedDimensions;
    for (auto [resultPosition, expression] :
         llvm::enumerate(map.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension ||
          !mappedDimensions.insert(dimension.getPosition()).second) {
        return fold.emitOpError(
            "indexing maps must contain each direct domain dimension at "
            "most once");
      }
      const int64_t extent = indexedType.getDimSize(resultPosition);
      if (ShapedType::isDynamic(extent)) {
        continue;
      }
      int64_t &knownExtent = domainExtents[dimension.getPosition()];
      if (!ShapedType::isDynamic(knownExtent) && knownExtent != extent) {
        return fold.emitOpError(
            "indexing maps bind one domain dimension to inconsistent static "
            "extents");
      }
      knownExtent = extent;
    }

    const bool isInput = position < inputCount;
    for (unsigned dimension = 0; dimension < domain.getNumDims(); ++dimension) {
      const bool isReduction = reductionDimensions.contains(dimension);
      const bool isMapped = mappedDimensions.contains(dimension);
      if (isInput && isReduction && !isMapped) {
        return fold.emitOpError(
            "each input indexing map must cover every reduction dimension");
      }
      if (!isInput && isReduction && isMapped) {
        return fold.emitOpError(
            "result indexing maps must project every reduction dimension");
      }
      if (!isInput && !isReduction && !isMapped) {
        return fold.emitOpError(
            "result indexing maps must cover every non-reduction dimension");
      }
    }
    ++position;
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
  if (!resultElement.isF32() && !resultElement.isBF16()) {
    return contract.emitOpError(
        "dot_general supports only bf16 or f32 result elements");
  }
  // This is the source accumulation declaration. A physical implementation's
  // accumulation precision belongs to its separately checked realization.
  if (accumulator != resultElement) {
    return contract.emitOpError(
        "dot_general source accumulator type must match result elements");
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

LogicalResult ScanOp::verifyRegions() {
  if (failed(verifySingleBlockRegion(*this, getBody()))) {
    return failure();
  }
  if (getExtent() <= 0) {
    return emitOpError("requires a positive ordered extent");
  }
  if (getAxis() < 0) {
    return emitOpError("requires a nonnegative logical axis identifier");
  }
  if (getStateOperands().empty()) {
    return emitOpError("requires at least one carried state");
  }
  if (getStateOperands().size() != getStateBodyArguments().size() ||
      getStateOperands().size() != getStateBodyResults().size() ||
      getStateOperands().size() != getStateResults().size()) {
    return emitOpError("state boundary bindings must align");
  }
  if (getInvariantOperands().size() != getInvariantBodyArguments().size()) {
    return emitOpError("invariant boundary bindings must align");
  }
  if (getScannedOperands().size() != getScannedBodyArguments().size() ||
      getScannedOperands().size() != getScannedDimensions().size()) {
    return emitOpError("scanned boundary bindings must align");
  }
  if (getEmittedResults().size() != getEmittedBodyResults().size() ||
      getEmittedResults().size() != getEmittedDimensions().size()) {
    return emitOpError("emitted boundary bindings must align");
  }

  auto verifyPartition = [&](ArrayRef<int64_t> first,
                             ArrayRef<int64_t> second,
                             ArrayRef<int64_t> third, int64_t size,
                             StringRef description) -> LogicalResult {
    SmallVector<bool> seen(size, false);
    for (ArrayRef<int64_t> group : {first, second, third}) {
      for (int64_t index : group) {
        if (index < 0 || index >= size || seen[index]) {
          return emitOpError() << description
                               << " bindings must be a disjoint in-range partition";
        }
        seen[index] = true;
      }
    }
    if (llvm::is_contained(seen, false)) {
      return emitOpError() << description
                           << " bindings must cover every value";
    }
    return success();
  };
  if (failed(verifyPartition(getStateOperands(), getInvariantOperands(),
                             getScannedOperands(), getInputs().size(),
                             "input")) ||
      failed(verifyPartition(getStateResults(), getEmittedResults(), {},
                             getResults().size(), "result"))) {
    return failure();
  }

  for (Attribute source : getSources()) {
    if (!isa<SourceRefAttr>(source)) {
      return emitOpError(
          "sources must contain only #shuttle.source_ref attributes");
    }
  }
  if (getSources().empty()) {
    return emitOpError("requires complete nonempty source provenance");
  }
  if (auto resultSources = getResultSourcesAttr()) {
    if (resultSources.size() != getResults().size()) {
      return emitOpError("result_sources must align with Scan results");
    }
    for (Attribute source : resultSources) {
      if (!isa<SourceRefAttr>(source) ||
          !llvm::is_contained(getSources(), source)) {
        return emitOpError(
            "result_sources must be source references contained in sources");
      }
    }
  }
  if (auto erased = getErasedTerminatorRecordsAttr()) {
    for (Attribute attribute : erased) {
      auto record = dyn_cast<DictionaryAttr>(attribute);
      if (!record || !record.get("operation_ref") ||
          !record.getAs<ArrayAttr>("operands") ||
          !record.getAs<DictionaryAttr>("fingerprint")) {
        return emitOpError(
            "erased_terminator_records must contain structural provenance records");
      }
    }
  }

  Block &body = getBody().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  if (failed(verifyPartition(getStateBodyArguments(),
                             getInvariantBodyArguments(),
                             getScannedBodyArguments(),
                             body.getNumArguments(), "body argument")) ||
      failed(verifyPartition(getStateBodyResults(), getEmittedBodyResults(),
                             {}, yield.getValues().size(), "body result"))) {
    return failure();
  }

  for (auto [operandIndex, bodyArgumentIndex, bodyResultIndex, resultIndex] :
       llvm::zip_equal(getStateOperands(), getStateBodyArguments(),
                       getStateBodyResults(), getStateResults())) {
    Type initialType = getInputs()[operandIndex].getType();
    if (initialType != getResults()[resultIndex].getType() ||
        initialType != body.getArgument(bodyArgumentIndex).getType() ||
        initialType != yield.getValues()[bodyResultIndex].getType()) {
      return emitOpError(
          "carried-state types must agree across the Scan boundary");
    }
  }
  for (auto [operandIndex, bodyArgumentIndex] :
       llvm::zip_equal(getInvariantOperands(), getInvariantBodyArguments())) {
    if (getInputs()[operandIndex].getType() !=
        body.getArgument(bodyArgumentIndex).getType()) {
      return emitOpError("invariant body argument type must equal its input type");
    }
  }

  auto sliceType = [&](Type outerType,
                       int64_t dimension) -> RankedTensorType {
    auto outer = dyn_cast<RankedTensorType>(outerType);
    if (!outer || !outer.hasStaticShape() || dimension < 0 ||
        dimension >= outer.getRank() ||
        outer.getDimSize(dimension) != getExtent()) {
      return {};
    }
    SmallVector<int64_t> shape(outer.getShape());
    shape.erase(shape.begin() + dimension);
    return RankedTensorType::get(shape, outer.getElementType(),
                                 outer.getEncoding());
  };
  for (auto [operandIndex, bodyArgumentIndex, dimension] : llvm::zip_equal(
           getScannedOperands(), getScannedBodyArguments(),
           getScannedDimensions())) {
    RankedTensorType expected =
        sliceType(getInputs()[operandIndex].getType(), dimension);
    if (!expected || expected != body.getArgument(bodyArgumentIndex).getType()) {
      return emitOpError(
          "scanned body argument must remove one ordered input dimension");
    }
  }
  for (auto [bodyResultIndex, resultIndex, dimension] : llvm::zip_equal(
           getEmittedBodyResults(), getEmittedResults(),
           getEmittedDimensions())) {
    RankedTensorType expected =
        sliceType(getResults()[resultIndex].getType(), dimension);
    if (!expected || expected != yield.getValues()[bodyResultIndex].getType()) {
      return emitOpError(
          "emitted body value must remove one ordered result dimension");
    }
  }

  for (Operation &operation : body.without_terminator()) {
    if (!isa<MapOp, FoldOp, ContractOp, RelationOp, SelectionOp>(operation)) {
      return operation.emitOpError(
          "is not a generic algebra operation supported in a Scan body");
    }
    if (!isMemoryEffectFree(&operation)) {
      return operation.emitOpError("must have proven no memory effects");
    }
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
  if (auto form = (*this)->getAttrOfType<StringAttr>("source_precision_form")) {
    if ((form.getValue() != "omitted" && form.getValue() != "empty") ||
        llvm::any_of(getPrecision(), [](Attribute precision) {
          return cast<StringAttr>(precision).getValue() != "DEFAULT";
        })) {
      return emitOpError(
          "source precision form must be omitted or empty with DEFAULT "
          "precision for every input");
    }
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
         getSemantics() == RelationSemantics::Compare ||
         getSemantics() == RelationSemantics::Multiply ||
         getSemantics() == RelationSemantics::Gather ||
         getSemantics() == RelationSemantics::IndexedContributions)
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
      : getSemantics() == RelationSemantics::Slice   ? 3
      : (getSemantics() == RelationSemantics::Transpose ||
         getSemantics() == RelationSemantics::Gather ||
         getSemantics() == RelationSemantics::IndexedContributions)
          ? 1
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
    if ((!result.getElementType().isInteger(32) &&
         !result.getElementType().isInteger(1)) ||
        !value || value.getType() != result.getElementType()) {
      return emitOpError("requires one splat i32 or i1 constant value");
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
    for (auto [inputDimension, parameter] : llvm::enumerate(getParameters())) {
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
  } else if (getSemantics() == RelationSemantics::Add ||
             getSemantics() == RelationSemantics::Multiply) {
    if (inputs[0] != inputs[1] || inputs[0] != result ||
        !result.getElementType().isInteger(32)) {
      return emitOpError("requires same-shaped i32 inputs and result");
    }
  } else if (getSemantics() == RelationSemantics::Compare) {
    auto comparisonType = stringParameter(0);
    auto direction = stringParameter(1);
    if (inputs[0] != inputs[1] || inputs[0].getShape() != result.getShape() ||
        !inputs[0].getElementType().isInteger(32) ||
        !result.getElementType().isInteger(1) || !comparisonType ||
        (comparisonType.getValue() != "SIGNED" &&
         comparisonType.getValue() != "UNSIGNED") ||
        !direction ||
        !llvm::is_contained(
            ArrayRef<StringRef>{"EQ", "NE", "GE", "GT", "LE", "LT"},
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
        inputs[2] != result ||
        (!result.getElementType().isF32() &&
         !result.getElementType().isInteger(32))) {
      return emitOpError(
          "requires an i1 predicate and same-shaped f32 or i32 values");
    }
  } else if (getSemantics() == RelationSemantics::Slice) {
    auto starts = dyn_cast<DenseI64ArrayAttr>(getParameters()[0]);
    auto limits = dyn_cast<DenseI64ArrayAttr>(getParameters()[1]);
    auto strides = dyn_cast<DenseI64ArrayAttr>(getParameters()[2]);
    if (!inputs[0].hasStaticShape() || !result.hasStaticShape() ||
        !inputs[0].getElementType().isInteger(32) ||
        result.getElementType() != inputs[0].getElementType() || !starts ||
        !limits || !strides ||
        starts.size() != static_cast<size_t>(inputs[0].getRank()) ||
        limits.size() != starts.size() || strides.size() != starts.size()) {
      return emitOpError("requires a static i32 slice with full rank metadata");
    }
    for (auto [dimension, start, limit, stride] : llvm::zip_equal(
             llvm::seq<int64_t>(0, inputs[0].getRank()), starts.asArrayRef(),
             limits.asArrayRef(), strides.asArrayRef())) {
      if (start < 0 || limit < start ||
          limit > inputs[0].getDimSize(dimension) || stride <= 0 ||
          result.getDimSize(dimension) !=
              (limit - start + stride - 1) / stride) {
        return emitOpError("has invalid static slice bounds or result shape");
      }
    }
  } else if (getSemantics() == RelationSemantics::Transpose) {
    auto permutation = dyn_cast<DenseI64ArrayAttr>(getParameters()[0]);
    if (!inputs[0].hasStaticShape() || !result.hasStaticShape() ||
        !inputs[0].getElementType().isInteger(32) ||
        result.getElementType() != inputs[0].getElementType() ||
        inputs[0].getRank() != result.getRank() || !permutation ||
        permutation.size() != static_cast<size_t>(inputs[0].getRank())) {
      return emitOpError("requires a static rank-preserving i32 transpose");
    }
    llvm::SmallDenseSet<int64_t> dimensions;
    for (auto [resultDimension, inputDimension] :
         llvm::enumerate(permutation.asArrayRef())) {
      if (inputDimension < 0 || inputDimension >= inputs[0].getRank() ||
          !dimensions.insert(inputDimension).second ||
          result.getDimSize(resultDimension) !=
              inputs[0].getDimSize(inputDimension)) {
        return emitOpError("has an invalid transpose permutation or shape");
      }
    }
  } else if (getSemantics() == RelationSemantics::Gather) {
    auto metadata = dyn_cast<DictionaryAttr>(getParameters()[0]);
    auto offsetDims = metadata
                          ? metadata.getAs<DenseI64ArrayAttr>("offset_dims")
                          : DenseI64ArrayAttr{};
    auto collapsedSliceDims =
        metadata ? metadata.getAs<DenseI64ArrayAttr>("collapsed_slice_dims")
                 : DenseI64ArrayAttr{};
    auto operandBatchingDims =
        metadata ? metadata.getAs<DenseI64ArrayAttr>("operand_batching_dims")
                 : DenseI64ArrayAttr{};
    auto startIndicesBatchingDims =
        metadata
            ? metadata.getAs<DenseI64ArrayAttr>("start_indices_batching_dims")
            : DenseI64ArrayAttr{};
    auto startIndexMap =
        metadata ? metadata.getAs<DenseI64ArrayAttr>("start_index_map")
                 : DenseI64ArrayAttr{};
    auto indexVectorDim = metadata
                              ? metadata.getAs<IntegerAttr>("index_vector_dim")
                              : IntegerAttr{};
    auto sliceSizes = metadata
                          ? metadata.getAs<DenseI64ArrayAttr>("slice_sizes")
                          : DenseI64ArrayAttr{};
    auto indicesAreSorted =
        metadata ? metadata.getAs<BoolAttr>("indices_are_sorted") : BoolAttr{};
    auto validUniqueDimensions = [](DenseI64ArrayAttr values, int64_t rank) {
      llvm::SmallDenseSet<int64_t> unique;
      return values && llvm::all_of(values.asArrayRef(), [&](int64_t value) {
               return value >= 0 && value < rank && unique.insert(value).second;
             });
    };
    if (!metadata || metadata.size() != 8 || !inputs[0].hasStaticShape() ||
        !inputs[1].hasStaticShape() || !result.hasStaticShape() ||
        inputs[0].getElementType() != result.getElementType() ||
        !inputs[1].getElementType().isInteger(32) || !indexVectorDim ||
        indexVectorDim.getInt() < 0 ||
        indexVectorDim.getInt() > inputs[1].getRank() || !indicesAreSorted ||
        !validUniqueDimensions(offsetDims, result.getRank()) ||
        !validUniqueDimensions(collapsedSliceDims, inputs[0].getRank()) ||
        !validUniqueDimensions(operandBatchingDims, inputs[0].getRank()) ||
        !validUniqueDimensions(startIndicesBatchingDims, inputs[1].getRank()) ||
        !validUniqueDimensions(startIndexMap, inputs[0].getRank()) ||
        !sliceSizes ||
        sliceSizes.size() != static_cast<size_t>(inputs[0].getRank()) ||
        operandBatchingDims.size() != startIndicesBatchingDims.size()) {
      return emitOpError(
          "requires complete, in-range static gather relation metadata");
    }
    const int64_t indexVectorSize =
        indexVectorDim.getInt() == inputs[1].getRank()
            ? 1
            : inputs[1].getDimSize(indexVectorDim.getInt());
    if (startIndexMap.size() != static_cast<size_t>(indexVectorSize)) {
      return emitOpError(
          "requires start_index_map to match the index-vector extent");
    }
    if (offsetDims.size() + collapsedSliceDims.size() +
            operandBatchingDims.size() !=
        static_cast<size_t>(inputs[0].getRank())) {
      return emitOpError(
          "requires offset, collapsed-slice, and operand-batching dimensions "
          "to partition the operand rank");
    }
    if (!llvm::is_sorted(offsetDims.asArrayRef()) ||
        !llvm::is_sorted(collapsedSliceDims.asArrayRef()) ||
        !llvm::is_sorted(operandBatchingDims.asArrayRef())) {
      return emitOpError(
          "requires sorted offset, collapsed-slice, and operand-batching "
          "dimensions");
    }
    auto intersects = [](DenseI64ArrayAttr lhs, DenseI64ArrayAttr rhs) {
      return llvm::any_of(lhs.asArrayRef(), [&](int64_t value) {
        return llvm::is_contained(rhs.asArrayRef(), value);
      });
    };
    if (intersects(collapsedSliceDims, operandBatchingDims) ||
        intersects(startIndexMap, operandBatchingDims)) {
      return emitOpError(
          "requires collapsed-slice and start-index dimensions disjoint from "
          "operand-batching dimensions");
    }
    if (llvm::is_contained(startIndicesBatchingDims.asArrayRef(),
                           indexVectorDim.getInt())) {
      return emitOpError(
          "requires start-indices batching dimensions to exclude the "
          "index-vector dimension");
    }
    for (auto [operandDimension, indicesDimension] :
         llvm::zip_equal(operandBatchingDims.asArrayRef(),
                         startIndicesBatchingDims.asArrayRef())) {
      if (inputs[0].getDimSize(operandDimension) !=
          inputs[1].getDimSize(indicesDimension)) {
        return emitOpError("requires paired gather batching extents to match");
      }
    }
    for (auto [dimension, size] : llvm::enumerate(sliceSizes.asArrayRef())) {
      if (size < 0 || size > inputs[0].getDimSize(dimension)) {
        return emitOpError("has an invalid gather slice size");
      }
    }
    for (int64_t dimension : collapsedSliceDims.asArrayRef()) {
      if (sliceSizes.asArrayRef()[dimension] > 1) {
        return emitOpError(
            "requires collapsed gather slice dimensions no larger than one");
      }
    }
    for (int64_t dimension : operandBatchingDims.asArrayRef()) {
      if (sliceSizes.asArrayRef()[dimension] > 1) {
        return emitOpError(
            "requires operand-batching gather slice dimensions no larger "
            "than one");
      }
    }
    const int64_t expandedIndicesRank =
        inputs[1].getRank() +
        (indexVectorDim.getInt() == inputs[1].getRank() ? 1 : 0);
    const int64_t inferredResultRank =
        offsetDims.size() + expandedIndicesRank - 1;
    if (result.getRank() != inferredResultRank) {
      return emitOpError("has a gather result with an incorrect rank");
    }
    SmallVector<int64_t> windowExtents;
    for (int64_t operandDimension = 0; operandDimension < inputs[0].getRank();
         ++operandDimension) {
      if (!llvm::is_contained(collapsedSliceDims.asArrayRef(),
                              operandDimension) &&
          !llvm::is_contained(operandBatchingDims.asArrayRef(),
                              operandDimension)) {
        windowExtents.push_back(sliceSizes.asArrayRef()[operandDimension]);
      }
    }
    SmallVector<int64_t> batchExtents;
    for (int64_t indicesDimension = 0; indicesDimension < expandedIndicesRank;
         ++indicesDimension) {
      if (indicesDimension != indexVectorDim.getInt()) {
        batchExtents.push_back(inputs[1].getDimSize(indicesDimension));
      }
    }
    size_t windowIndex = 0;
    size_t batchIndex = 0;
    for (int64_t resultDimension = 0; resultDimension < result.getRank();
         ++resultDimension) {
      const bool isOffset =
          llvm::is_contained(offsetDims.asArrayRef(), resultDimension);
      const int64_t expectedExtent =
          isOffset ? windowExtents[windowIndex++] : batchExtents[batchIndex++];
      if (result.getDimSize(resultDimension) != expectedExtent) {
        return emitOpError("has a gather result with an incorrect shape");
      }
    }
  } else if (getSemantics() == RelationSemantics::IndexedContributions) {
    auto metadata = dyn_cast<DictionaryAttr>(getParameters()[0]);
    auto outputShape = metadata
                           ? metadata.getAs<DenseI64ArrayAttr>("output_shape")
                           : DenseI64ArrayAttr{};
    auto updateWindowDims =
        metadata ? metadata.getAs<DenseI64ArrayAttr>("update_window_dims")
                 : DenseI64ArrayAttr{};
    auto insertedWindowDims =
        metadata ? metadata.getAs<DenseI64ArrayAttr>("inserted_window_dims")
                 : DenseI64ArrayAttr{};
    auto inputBatchingDims =
        metadata ? metadata.getAs<DenseI64ArrayAttr>("input_batching_dims")
                 : DenseI64ArrayAttr{};
    auto indicesBatchingDims =
        metadata
            ? metadata.getAs<DenseI64ArrayAttr>("scatter_indices_batching_dims")
            : DenseI64ArrayAttr{};
    auto scatterDims =
        metadata
            ? metadata.getAs<DenseI64ArrayAttr>("scatter_dims_to_operand_dims")
            : DenseI64ArrayAttr{};
    auto indexVectorDim = metadata
                              ? metadata.getAs<IntegerAttr>("index_vector_dim")
                              : IntegerAttr{};
    auto sorted =
        metadata ? metadata.getAs<BoolAttr>("indices_are_sorted") : BoolAttr{};
    auto unique =
        metadata ? metadata.getAs<BoolAttr>("unique_indices") : BoolAttr{};
    auto conflict = metadata ? metadata.getAs<StringAttr>("conflict_resolution")
                             : StringAttr{};
    if (!metadata || metadata.size() != 10 || !outputShape ||
        !updateWindowDims || !insertedWindowDims || !inputBatchingDims ||
        !indicesBatchingDims || !scatterDims || !indexVectorDim || !sorted ||
        !unique || !conflict || conflict.getValue() != "source_ordered_add" ||
        !inputs[0].hasStaticShape() || !inputs[1].hasStaticShape() ||
        !result.hasStaticShape() ||
        inputs[0].getElementType() != result.getElementType() ||
        !inputs[1].getElementType().isInteger(32) ||
        result.getRank() <= static_cast<int64_t>(outputShape.size()) ||
        !llvm::equal(result.getShape().take_front(outputShape.size()),
                     outputShape.asArrayRef())) {
      return emitOpError(
          "requires typed additive indexed-contribution metadata and a "
          "result prefixed by the destination shape");
    }
  }
  return success();
}

LogicalResult SelectionOp::verify() {
  auto input = dyn_cast<RankedTensorType>(getInput().getType());
  auto values = dyn_cast<RankedTensorType>(getSelectedValues().getType());
  auto indices = dyn_cast<RankedTensorType>(getSelectedIndices().getType());
  if (!input || !values || !indices || !input.hasStaticShape() ||
      !values.hasStaticShape() || !indices.hasStaticShape()) {
    return emitOpError("requires static ranked tensor input and results");
  }
  if (!isa<FloatType, IntegerType>(input.getElementType()) ||
      values.getElementType() != input.getElementType() ||
      !indices.getElementType().isInteger(32)) {
    return emitOpError(
        "requires selected values to preserve a numeric input element type "
        "and selected indices to use i32");
  }
  if (!getStable() ||
      getTieBreak() != SelectionTieBreak::SourceIndexAscending) {
    return emitOpError("requires stable source-index-ascending tie breaking");
  }
  const int64_t dimension = getDimension();
  const int64_t count = getCount();
  if (dimension < 0 || dimension >= input.getRank() || count <= 0 ||
      count > input.getDimSize(dimension)) {
    return emitOpError(
        "requires an in-range selection dimension and a positive count no "
        "larger than that dimension");
  }
  SmallVector<int64_t> expectedShape(input.getShape());
  expectedShape[dimension] = count;
  if (values.getShape() != ArrayRef<int64_t>(expectedShape) ||
      indices.getShape() != ArrayRef<int64_t>(expectedShape)) {
    return emitOpError(
        "requires result shapes equal to the input shape with the selection "
        "dimension replaced by count");
  }
  if (getSources().size() != 2) {
    return emitOpError("requires one source reference per result");
  }
  SourceRefAttr first;
  for (auto [resultNumber, attribute] : llvm::enumerate(getSources())) {
    auto source = dyn_cast<SourceRefAttr>(attribute);
    if (!source || source.getResultOrdinal() != resultNumber) {
      return emitOpError(
          "requires ordered source references matching result ordinals");
    }
    if (!first) {
      first = source;
      continue;
    }
    if (source.getFunctionOrdinal() != first.getFunctionOrdinal() ||
        source.getBlockOrdinal() != first.getBlockOrdinal() ||
        source.getOperationOrdinal() != first.getOperationOrdinal()) {
      return emitOpError(
          "requires both results to originate at one source operation");
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
  SmallVector<Type> indexedTypes;
  llvm::append_range(indexedTypes, getInputs().getTypes());
  llvm::append_range(indexedTypes, getResults().getTypes());
  if (failed(verifyFoldIndexingMaps(*this, indexedTypes))) {
    return failure();
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
  if (plan.getComponentized()) {
    stream << ";componentized=true";
  }
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
                                           ArrayRef<int64_t> reductionAxes) {
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
  llvm::erase_if(masks, [reductionAxes](unsigned mask) {
    return llvm::any_of(reductionAxes, [mask](int64_t axis) {
      return (mask & (1u << axis)) == 0;
    });
  });
  return masks;
}

bool inputShapesBindContractDomain(RankedTensorType lhs, RankedTensorType rhs,
                                   ArrayRef<int64_t> domain,
                                   ArrayRef<int64_t> reductionAxes) {
  if (domain.size() < 3 || domain.size() > 9 || lhs.getRank() < 2 ||
      rhs.getRank() < 2 || lhs.getRank() > 8 || rhs.getRank() > 8 ||
      reductionAxes.empty()) {
    return false;
  }
  SmallVector<unsigned> lhsMasks =
      projectedDomainMasks(lhs, domain, reductionAxes);
  SmallVector<unsigned> rhsMasks =
      projectedDomainMasks(rhs, domain, reductionAxes);
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
  unsigned statefulContractFoldCount = 0;
  unsigned indexedAccumulationCount = 0;
  unsigned scanCount = 0;
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
      return buffer.emitOpError(
          "requires a positive static ranked tensor type");
    std::optional<int64_t> producer =
        optionalIntegerAttribute(buffer, "producer");
    std::optional<int64_t> sourceArgument =
        optionalIntegerAttribute(buffer, "source_argument");
    if (buffer.getLiveIn() != !producer.has_value()) {
      return buffer.emitOpError(
          "live-in status must exactly identify producer-free buffers");
    }
    if (sourceArgument && (!buffer.getLiveIn() || *sourceArgument < 0)) {
      return buffer.emitOpError(
          "source argument must be a nonnegative live-in binding");
    }
    const bool external =
        buffer.getStorage() == MaterializationStorage::External;
    Type elementType = tensorType.getElementType();
    const bool supportedElement = elementType.isF32() || elementType.isBF16() ||
                                  elementType.isInteger(1) ||
                                  elementType.isInteger(32);
    if (!supportedElement)
      return buffer.emitOpError(
          "requires a bf16/f32/i1/i32 materialization element type");
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
    auto sourceGroup = task->getAttrOfType<ArrayAttr>("source_group");
    if (task.getKind() == MaterializationTaskKind::Selection) {
      auto first = sourceGroup && sourceGroup.size() == 2
                       ? dyn_cast<SourceRefAttr>(sourceGroup[0])
                       : SourceRefAttr{};
      auto second = sourceGroup && sourceGroup.size() == 2
                        ? dyn_cast<SourceRefAttr>(sourceGroup[1])
                        : SourceRefAttr{};
      if (!sourceGroup || sourceGroup.size() != 2 || !first || !second ||
          first != task.getSource() || first.getResultOrdinal() != 0 ||
          second.getResultOrdinal() != 1 ||
          first.getFunctionOrdinal() != second.getFunctionOrdinal() ||
          first.getBlockOrdinal() != second.getBlockOrdinal() ||
          first.getOperationOrdinal() != second.getOperationOrdinal()) {
        return task.emitOpError(
            "Selection tasks require two ordered source bindings");
      }
    } else if (sourceGroup) {
      return task.emitOpError(
          "source groups are reserved for multi-result Selection tasks");
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
    auto sourceExpansionGroup =
        task->getAttrOfType<ArrayAttr>("source_expansion_group");
    auto compositionRecomputedSources =
        task->getAttrOfType<ArrayAttr>("composition_recomputed_sources");
    auto compositionCost =
        task->getAttrOfType<DictionaryAttr>("composition_cost");
    auto compositionOwnerDimensions =
        task->getAttrOfType<DenseI64ArrayAttr>("composition_owner_dimensions");
    auto compositionStreamDimensions =
        task->getAttrOfType<DenseI64ArrayAttr>("composition_stream_dimensions");
    auto compositionStateInputs =
        task->getAttrOfType<DenseI64ArrayAttr>("composition_state_inputs");
    auto compositionStateOutputs =
        task->getAttrOfType<DenseI64ArrayAttr>("composition_state_outputs");
    auto producerContractionDimensions = task->getAttrOfType<DenseI64ArrayAttr>(
        "composition_producer_contraction_dimensions");
    auto consumerContractionDimensions = task->getAttrOfType<DenseI64ArrayAttr>(
        "composition_consumer_contraction_dimensions");
    const bool statefulContractFold =
        !compositionKind && compositionSources && compositionOwnerDimensions &&
        compositionStreamDimensions && producerContractionDimensions &&
        consumerContractionDimensions;
    statefulContractFoldCount += statefulContractFold;
    const bool hasStructuralComposition =
        compositionOwnerDimensions || compositionStreamDimensions ||
        compositionStateInputs || compositionStateOutputs ||
        producerContractionDimensions || consumerContractionDimensions;
    const bool hasComposition = compositionKind || compositionSources ||
                                compositionRecomputedSources ||
                                compositionCost || hasStructuralComposition;
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
      const bool indexedAccumulation =
          compositionKind &&
          compositionKind.getValue() == "indexed_accumulation";
      const bool integerRelation =
          compositionKind && compositionKind.getValue() == "integer_relation";
      const bool relationGather =
          compositionKind && compositionKind.getValue() == "relation_gather";
      const bool scan =
          compositionKind && compositionKind.getValue() == "scan";
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
      if (getSchemaVersion() != 2 || !compositionSources ||
          ((contractFold || mapContract || indexedAccumulation ||
            integerRelation || relationGather || scan) &&
           compositionSources.empty()) ||
          (statefulContractFold && compositionSources.empty()) ||
          (compositionRecomputedSources && !statefulContractFold &&
           !indexedAccumulation && !mapContract) ||
          (!contractFold && !mapContract && !indexedAccumulation &&
           !integerRelation && !relationGather && !scan &&
           !statefulContractFold) ||
          (hasStructuralComposition && !statefulContractFold &&
           !indexedAccumulation && !scan) ||
          (scan && task.getKind() != MaterializationTaskKind::Scan) ||
          (indexedAccumulation &&
           (!sourceExpansionGroup ||
            sourceExpansionGroup.size() != compositionSources.size() ||
            task.getKind() != MaterializationTaskKind::Fold)) ||
          (!indexedAccumulation && sourceExpansionGroup) || legacyMetadata ||
          llvm::any_of(
              compositionSources,
              [](Attribute source) { return !isa<SourceRefAttr>(source); }) ||
          (compositionRecomputedSources &&
           (llvm::any_of(
                compositionRecomputedSources,
                [](Attribute source) { return !isa<SourceRefAttr>(source); }) ||
            llvm::any_of(compositionRecomputedSources, [&](Attribute source) {
              return llvm::is_contained(compositionSources, source);
            })))) {
        return task.emitOpError(
            "schema-v2 generic composition provenance must be complete and "
            "exclusive");
      }
      if (statefulContractFold) {
        auto validAxes = [&](DenseI64ArrayAttr dimensions) {
          llvm::SmallDenseSet<int64_t> seen;
          return dimensions && !dimensions.empty() &&
                 llvm::all_of(dimensions.asArrayRef(), [&](int64_t dimension) {
                   return dimension >= 0 &&
                          dimension < static_cast<int64_t>(
                                          task.getDomainShape().size()) &&
                          seen.insert(dimension).second;
                 });
        };
        auto validBuffers = [&](DenseI64ArrayAttr state,
                                ArrayRef<int64_t> taskBuffers) {
          llvm::SmallDenseSet<int64_t> seen;
          return !state ||
                 llvm::all_of(state.asArrayRef(), [&](int64_t buffer) {
                   return llvm::is_contained(taskBuffers, buffer) &&
                          seen.insert(buffer).second;
                 });
        };
        auto vectorDimensions = task.getOnlineVectorDimensionsAttr();
        llvm::SmallDenseSet<int64_t> partitionedDimensions;
        auto addPartition = [&](DenseI64ArrayAttr dimensions) {
          return dimensions &&
                 llvm::all_of(dimensions.asArrayRef(), [&](int64_t dimension) {
                   return partitionedDimensions.insert(dimension).second;
                 });
        };
        if (task.getKind() != MaterializationTaskKind::Contract ||
            task.getDomainShape().empty() || task.getDomainShape().size() > 9 ||
            !validAxes(compositionOwnerDimensions) ||
            !validAxes(vectorDimensions) ||
            !validAxes(compositionStreamDimensions) ||
            !validAxes(producerContractionDimensions) ||
            !validAxes(consumerContractionDimensions) ||
            !addPartition(compositionOwnerDimensions) ||
            !addPartition(vectorDimensions) ||
            !addPartition(compositionStreamDimensions) ||
            partitionedDimensions.size() != task.getDomainShape().size() ||
            compositionStreamDimensions.asArrayRef() !=
                task.getReductionDimensions() ||
            !validBuffers(compositionStateInputs, task.getInputBuffers()) ||
            !validBuffers(compositionStateOutputs, task.getOutputBuffers()))
          return task.emitOpError(
              "stateful Contract/Fold composition requires explicit valid "
              "owner, vector, stream, state, and producer/consumer "
              "contraction dimensions");
        const bool directOnline =
            !compositionStateInputs && !compositionStateOutputs;
        if (directOnline) {
          constexpr StringLiteral costFields[] = {
              "baseline_launches",
              "candidate_launches",
              "baseline_temporary_bytes",
              "candidate_temporary_bytes",
              "baseline_temporary_traffic_bytes",
              "candidate_temporary_traffic_bytes",
              "contract_evaluations"};
          if (!compositionCost ||
              compositionCost.size() != std::size(costFields) ||
              task.getOutputBuffers().empty() ||
              task.getDomainShape().size() > 9)
            return task.emitOpError(
                "online composition requires abstract Contract state, "
                "outputs, and closed relative cost");
          SmallVector<int64_t> costs;
          for (StringRef field : costFields) {
            auto value = compositionCost.getAs<IntegerAttr>(field);
            if (!value || value.getInt() < 0)
              return task.emitOpError(
                  "online composition cost fields must be nonnegative i64");
            costs.push_back(value.getInt());
          }
          if (costs[1] > costs[0] || costs[3] >= costs[2] ||
              costs[5] >= costs[4] || costs[6] <= 0)
            return task.emitOpError(
                "online composition must improve bytes and traffic without "
                "increasing launch cost");
        } else if (compositionCost ||
                   (!compositionStateInputs && !compositionStateOutputs)) {
          return task.emitOpError(
              "stateful pullback composition requires state flow and no "
              "relative-cost claim");
        }
        ++contractCount;
        continue;
      }
      if (indexedAccumulation) {
        llvm::SmallDenseSet<int64_t> axes;
        auto addAxes = [&](DenseI64ArrayAttr dimensions) {
          return dimensions &&
                 llvm::all_of(dimensions.asArrayRef(), [&](int64_t axis) {
                   return axis >= 0 &&
                          axis < static_cast<int64_t>(
                                     task.getDomainShape().size()) &&
                          axes.insert(axis).second;
                 });
        };
        bool validExpansion = sourceExpansionGroup.size() == 4;
        for (auto [member, attribute] : llvm::enumerate(sourceExpansionGroup)) {
          auto expansion = dyn_cast<DictionaryAttr>(attribute);
          auto actualMember = expansion ? expansion.getAs<IntegerAttr>("member")
                                        : IntegerAttr{};
          auto size =
              expansion ? expansion.getAs<IntegerAttr>("size") : IntegerAttr{};
          validExpansion &=
              actualMember && size &&
              actualMember.getInt() == static_cast<int64_t>(member) &&
              size.getInt() == 4;
        }
        if (!validExpansion || compositionCost ||
            task.getInputBuffers().empty() || task.getOutputBuffers().empty() ||
            task.getDomainShape().empty() || task.getDomainShape().size() > 8 ||
            task.getReductionDimensions().empty() ||
            task.getOrderFree().value_or(true) ||
            !addAxes(compositionOwnerDimensions) ||
            !addAxes(compositionStreamDimensions) ||
            axes.size() != task.getDomainShape().size() ||
            compositionStreamDimensions.asArrayRef() !=
                task.getReductionDimensions()) {
          return task.emitOpError(
              "indexed accumulation requires one source-ordered Fold task "
              "with complete owner/stream axes and four ordered expansion "
              "members");
        }
        ++foldCount;
        ++indexedAccumulationCount;
        continue;
      }
      if (integerRelation) {
        if (compositionCost || hasStructuralComposition ||
            sourceExpansionGroup ||
            task.getKind() != MaterializationTaskKind::Relation ||
            task.getInputBuffers().empty() ||
            task.getOutputBuffers().size() != 1 || task.getOrderFree() ||
            !task.getReductionDimensions().empty() ||
            task.getDomainShape().empty() || task.getDomainShape().size() > 8)
          return task.emitOpError(
              "Relation composition requires one bounded non-reducing "
              "Relation task with explicit inputs and one output");
        continue;
      }
      if (relationGather) {
        if (compositionCost || hasStructuralComposition ||
            sourceExpansionGroup ||
            task.getKind() != MaterializationTaskKind::Relation ||
            task.getInputBuffers().empty() ||
            (task.getOutputBuffers().size() != 1 &&
             task.getOutputBuffers().size() != 2) ||
            task.getOrderFree() ||
            !task.getReductionDimensions().empty() ||
            task.getDomainShape().empty() || task.getDomainShape().size() > 8)
          return task.emitOpError(
              "Relation-to-Gather composition requires one bounded "
              "non-reducing Relation task with explicit inputs and one or "
              "two outputs");
        continue;
      }
      if (scan) {
        auto validStateBuffers = [](DenseI64ArrayAttr state,
                                    ArrayRef<int64_t> taskBuffers) {
          llvm::SmallDenseSet<int64_t> seen;
          return state && !state.empty() &&
                 llvm::all_of(state.asArrayRef(), [&](int64_t buffer) {
                   return llvm::is_contained(taskBuffers, buffer) &&
                          seen.insert(buffer).second;
                 });
        };
        if (compositionCost || sourceExpansionGroup ||
            task.getKind() != MaterializationTaskKind::Scan ||
            task.getInputBuffers().empty() || task.getOutputBuffers().empty() ||
            task.getDomainShape().size() != 1 ||
            !task.getReductionDimensions().empty() || task.getOrderFree() ||
            !compositionOwnerDimensions ||
            !compositionOwnerDimensions.empty() ||
            !compositionStreamDimensions ||
            compositionStreamDimensions.asArrayRef() != ArrayRef<int64_t>({0}) ||
            !producerContractionDimensions ||
            !producerContractionDimensions.empty() ||
            !consumerContractionDimensions ||
            !consumerContractionDimensions.empty() ||
            !validStateBuffers(compositionStateInputs,
                               task.getInputBuffers()) ||
            !validStateBuffers(compositionStateOutputs,
                               task.getOutputBuffers()))
          return task.emitOpError(
              "Scan composition requires one bounded ordered task with explicit state flow");
        ++scanCount;
        continue;
      }
      if (contractFold) {
        if (contractFoldCompositionTask >= 0 || compositionCost ||
            task.getKind() != MaterializationTaskKind::Fold ||
            task.getInputBuffers().empty() || task.getOutputBuffers().empty() ||
            !task.getOrderFree().value_or(false) ||
            task.getDomainShape().empty() || task.getDomainShape().size() > 8 ||
            task.getReductionDimensions().empty()) {
          return task.emitOpError(
              "producer-to-Fold composition requires one Fold task with "
              "nonempty inputs, reductions, and live-across outputs");
        }
        contractFoldCompositionTask = ordinal;
        ++foldCount;
        continue;
      }
      if ((contractFoldCompositionTask >= 0 &&
           mapContractCompositionTask >= 0) ||
          !compositionCost ||
          (task.getKind() != MaterializationTaskKind::Contract &&
           task.getKind() != MaterializationTaskKind::Map) ||
          task.getInputBuffers().empty() || task.getOutputBuffers().empty() ||
          task.getOrderFree() || task.getDomainShape().empty() ||
          task.getDomainShape().size() > 9 ||
          (task.getKind() == MaterializationTaskKind::Contract) !=
              !task.getReductionDimensions().empty()) {
        return task.emitOpError(
            "Fold-dependent composition requires one terminal Map or Contract "
            "task with nonempty inputs and outputs");
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
          costs[6] < 0) {
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
      const bool hasEpilogue = static_cast<bool>(
          task->getAttrOfType<ArrayAttr>("fold_epilogue_sources"));
      const int64_t domainRank = task.getDomainShape().size();
      ArrayRef<int64_t> reductionDimensions = task.getReductionDimensions();
      llvm::SmallDenseSet<int64_t> uniqueReductionDimensions;
      const bool validReductionDimensions =
          domainRank >= 1 && domainRank <= 8 && !reductionDimensions.empty() &&
          reductionDimensions.size() <= static_cast<size_t>(domainRank) &&
          llvm::all_of(reductionDimensions, [&](int64_t axis) {
            return axis >= 0 && axis < domainRank &&
                   uniqueReductionDimensions.insert(axis).second;
          });
      if (!validReductionDimensions || !task.getOrderFree().has_value() ||
          (hasEpilogue ? task.getInputBuffers().size() < 2
                       : task.getInputBuffers().size() != 2) ||
          task.getOutputBuffers().size() != 1) {
        return task.emitOpError(
            "Fold tasks require nonempty unique in-range reduction dimensions "
            "and a declared reduction order");
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
          (hasEpilogue &&
           isSingletonInsertion(expectedResultShape, resultType.getShape()));
      Type elementType = inputType.getElementType();
      if (inputType.getShape() != task.getDomainShape() ||
          !isa<FloatType>(elementType) || initializerType.getRank() != 0 ||
          initializerType.getElementType() != elementType ||
          !resultShapeMatches || resultType.getElementType() != elementType ||
          (hasEpilogue &&
           llvm::any_of(task.getInputBuffers().drop_front(2),
                        [&](int64_t bufferOrdinal) {
                          auto type = cast<RankedTensorType>(
                              buffers[bufferOrdinal].getTensorType());
                          return !type.hasStaticShape() ||
                                 type.getElementType() != elementType;
                        }))) {
        return task.emitOpError(
            "Fold buffer element types must match its floating-point domain "
            "and accumulator");
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
      ArrayRef<int64_t> reductionAxes = task.getReductionDimensions();
      llvm::SmallDenseSet<int64_t> uniqueReductionAxes;
      const bool validReductionAxes =
          !reductionAxes.empty() &&
          llvm::all_of(reductionAxes, [&](int64_t axis) {
            return axis >= 0 && axis < static_cast<int64_t>(domain.size()) &&
                   uniqueReductionAxes.insert(axis).second;
          });
      const bool scalarInnerProduct =
          domain.size() == 1 && reductionAxes.size() == 1 &&
          reductionAxes.front() == 0;
      if ((!scalarInnerProduct && domain.size() < 3) || domain.size() > 9 ||
          !validReductionAxes || task.getOrderFree() ||
          task.getInputBuffers().size() != 2 ||
          task.getOutputBuffers().size() != 1) {
        return task.emitOpError()
               << "Contract tasks require a scalar inner product or a "
                  "rank-three through rank-nine "
                  "domain, explicit reduction dimensions, two inputs, and one "
                  "output";
      }
      auto lhs = cast<RankedTensorType>(
          buffers[task.getInputBuffers()[0]].getTensorType());
      auto rhs = cast<RankedTensorType>(
          buffers[task.getInputBuffers()[1]].getTensorType());
      auto result = cast<RankedTensorType>(
          buffers[task.getOutputBuffers()[0]].getTensorType());
      SmallVector<int64_t> expectedResultShape;
      for (auto [axis, extent] : llvm::enumerate(domain))
        if (!uniqueReductionAxes.contains(axis))
          expectedResultShape.push_back(extent);
      const bool scalarBuffers =
          scalarInnerProduct && lhs.getShape() == domain &&
          rhs.getShape() == domain && result.getRank() == 0;
      if ((!scalarBuffers && (result.getRank() < 2 || result.getRank() > 8)) ||
          lhs.getElementType() != rhs.getElementType() ||
          (!lhs.getElementType().isBF16() && !lhs.getElementType().isF32()) ||
          !result.getElementType().isF32() ||
          (!scalarBuffers &&
           !inputShapesBindContractDomain(lhs, rhs, domain, reductionAxes)) ||
          result.getShape() != ArrayRef<int64_t>(expectedResultShape)) {
        return task.emitOpError() << "Contract buffers must bind "
                                  << cardinalName(expectedResultShape.size())
                                  << " distinct parallel axes and the declared "
                                     "reduction axes with "
                                     "equal bf16 or f32 input types";
      }
    } else if (task.getKind() == MaterializationTaskKind::Selection) {
      ArrayRef<int64_t> domain = task.getDomainShape();
      ArrayRef<int64_t> reductionAxes = task.getReductionDimensions();
      if (domain.empty() || domain.size() > 8 || reductionAxes.size() != 1 ||
          reductionAxes.front() < 0 ||
          reductionAxes.front() >= static_cast<int64_t>(domain.size()) ||
          task.getOrderFree() || task.getInputBuffers().size() != 1 ||
          task.getOutputBuffers().size() != 2) {
        return task.emitOpError(
            "Selection tasks require one bounded reduction axis, one input, "
            "and two outputs");
      }
      auto input = cast<RankedTensorType>(
          buffers[task.getInputBuffers().front()].getTensorType());
      auto values = cast<RankedTensorType>(
          buffers[task.getOutputBuffers().front()].getTensorType());
      auto indices = cast<RankedTensorType>(
          buffers[task.getOutputBuffers().back()].getTensorType());
      SmallVector<int64_t> resultShape(values.getShape());
      const int64_t reductionAxis = reductionAxes.front();
      const bool validResultExtent =
          resultShape[reductionAxis] > 0 &&
          resultShape[reductionAxis] <= domain[reductionAxis];
      resultShape[reductionAxis] = domain[reductionAxis];
      if (input.getShape() != domain || values.getRank() != input.getRank() ||
          indices.getShape() != values.getShape() || !validResultExtent ||
          ArrayRef<int64_t>(resultShape) != domain ||
          values.getElementType() != input.getElementType() ||
          !indices.getElementType().isInteger(32)) {
        return task.emitOpError(
            "Selection buffers must preserve the input domain and replace "
            "the bounded axis equally in value and i32 index outputs");
      }
    } else if (task.getKind() == MaterializationTaskKind::Relation) {
      if (!task.getReductionDimensions().empty() || task.getOrderFree() ||
          task.getOutputBuffers().size() != 1) {
        return task.emitOpError(
            "Relation tasks require one result and no reduction or ordering "
            "metadata");
      }
      auto result = cast<RankedTensorType>(
          buffers[task.getOutputBuffers().front()].getTensorType());
      if (result.getShape() != task.getDomainShape()) {
        return task.emitOpError(
            "Relation result must exactly define its task domain");
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
  if (contractFoldCompositionTask < 0 && mapContractCompositionTask >= 0) {
    MaterializationTaskOp consumer = tasks[mapContractCompositionTask];
    for (auto [ordinal, task] : llvm::enumerate(tasks)) {
      if (!task->hasAttr("fold_region_sources") ||
          !llvm::is_contained(consumer.getDependencies(), ordinal) ||
          consumer.getInputBuffers().take_front(task.getOutputBuffers().size()) !=
              task.getOutputBuffers())
        continue;
      if (contractFoldCompositionTask >= 0)
        return emitOpError(
            "generic composition has multiple Fold-region boundary producers");
      contractFoldCompositionTask = ordinal;
    }
    // `map_contract` is also the generic same-domain Contract/Map component.
    // It participates in this two-task Fold boundary only when an actual Fold
    // producer is connected by the ordered SSA cut.
    if (contractFoldCompositionTask < 0)
      mapContractCompositionTask = -1;
  }
  const bool hasCompositionTasks =
      contractFoldCompositionTask >= 0 || mapContractCompositionTask >= 0;
  if (hasCompositionTasks) {
    if (contractFoldCompositionTask < 0 || mapContractCompositionTask < 0 ||
        contractFoldCompositionTask >= mapContractCompositionTask) {
      return emitOpError(
          "generic composition requires ordered producer-to-Fold and "
          "Fold-dependent task groups");
    }
    MaterializationTaskOp producer = tasks[contractFoldCompositionTask];
    MaterializationTaskOp consumer = tasks[mapContractCompositionTask];
    ArrayRef<int64_t> boundary = producer.getOutputBuffers();
    if (consumer.getInputBuffers().take_front(boundary.size()) != boundary ||
        consumer.getDependencies() !=
            ArrayRef<int64_t>({contractFoldCompositionTask})) {
      return emitOpError(
          "generic composition must preserve its ordered live-across SSA "
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
  const bool containsOnlyMaps =
      llvm::all_of(tasks, [](MaterializationTaskOp task) {
        return task.getKind() == MaterializationTaskKind::Map;
      });
  const bool standaloneMap = tasks.size() == 1;
  if (containsOnlyMaps && !standaloneMap &&
      !getComponentized().value_or(false)) {
    for (MaterializationBufferOp buffer : buffers) {
      auto type = cast<RankedTensorType>(buffer.getTensorType());
      if (type.getRank() > 8 ||
          (type.getRank() == 0 && type.getElementType().isBF16()) ||
          (!type.getElementType().isF32() && !type.getElementType().isBF16())) {
        return buffer.emitOpError(
            "Map-only buffers require rank-zero through rank-eight f32 "
            "or rank-one through rank-eight bf16 tensors");
      }
    }
  }
  bool compactContractRegion = false;
  for (MaterializationTaskOp task : tasks) {
    auto strategy = task->getAttrOfType<StringAttr>("contract_region_strategy");
      compactContractRegion |= strategy && strategy.getValue() == "compact_stat";
  }
  const bool hybridFoldComposition = contractFoldCompositionTask >= 0 &&
                                     mapContractCompositionTask >= 0;
  if (contractRegionCount > 1 ||
      (contractCount != contractRegionCount && foldRegionCount != 0 &&
       !compactContractRegion && !hybridFoldComposition &&
       contractCount != statefulContractFoldCount) ||
      (foldRegionCount != 0 && !compactContractRegion && contractCount != 0 &&
       !hybridFoldComposition &&
       contractCount != statefulContractFoldCount) ||
      getSchemaVersion() != (contractCount != 0 || foldRegionCount != 0 ||
                                     indexedAccumulationCount != 0 ||
                                     scanCount != 0
                                 ? 2
                                 : 1)) {
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
    }
    if (buffer.getLifetimeEnd() != expectedEnd) {
      return buffer.emitOpError(
          "lifetime end must equal the final consumer or plan exit");
    }
  }
  for (MaterializationTaskOp task : tasks) {
    if (llvm::none_of(task.getOutputBuffers(), [&](int64_t output) {
          return buffers[output].getLiveOut() ||
                 !actualConsumers[output].empty();
        })) {
      return task.emitOpError(
          "requires at least one consumed or live-out result");
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
  SmallVector<int64_t> reductionMergeGrid;
  SmallVector<int64_t> reductionMergeTile;
  int64_t reductionMergeSerialTiles = 0;
  SmallVector<int64_t> reductionPartialBufferShape;
};

FailureOr<Simt32Geometry>
multiPhaseGeometry(ScheduleTaskOp task, ArrayAttr records) {
  if (records.empty())
    return failure();
  int64_t nextOffset = 0;
  int64_t maximumTile = 0;
  int64_t maximumThreads = 0;
  int64_t maximumSerialTiles = 1;
  for (auto [phaseOrdinal, attribute] : llvm::enumerate(records)) {
    auto record = dyn_cast<DictionaryAttr>(attribute);
    auto phase = record ? record.getAs<IntegerAttr>("phase") : IntegerAttr{};
    auto offset =
        record ? record.getAs<IntegerAttr>("grid_offset") : IntegerAttr{};
    auto count =
        record ? record.getAs<IntegerAttr>("grid_count") : IntegerAttr{};
    auto elements =
        record ? record.getAs<IntegerAttr>("element_count") : IntegerAttr{};
    auto tile =
        record ? record.getAs<IntegerAttr>("tile_elements") : IntegerAttr{};
    auto serial =
        record ? record.getAs<IntegerAttr>("serial_tiles") : IntegerAttr{};
    auto domain =
        record ? record.getAs<DenseI64ArrayAttr>("domain_shape")
               : DenseI64ArrayAttr{};
    auto reductions =
        record ? record.getAs<DenseI64ArrayAttr>("reduction_dimensions")
               : DenseI64ArrayAttr{};
    if (!record || record.size() != 8 || !phase || !offset || !count ||
        !elements || !tile || !serial || !domain || !reductions ||
        phase.getInt() != static_cast<int64_t>(phaseOrdinal) ||
        offset.getInt() != nextOffset || count.getInt() <= 0 ||
        elements.getInt() <= 0 || tile.getInt() <= 0 ||
        tile.getInt() > elements.getInt() || serial.getInt() <= 0 ||
        count.getInt() !=
            ceilDivPositive(elements.getInt(), tile.getInt()) ||
        llvm::any_of(domain.asArrayRef(),
                     [](int64_t extent) { return extent <= 0; }))
      return failure();
    FailureOr<int64_t> domainElements =
        staticElementCount(domain.asArrayRef());
    int64_t expectedSerial = 1;
    if (!reductions.empty()) {
      if (reductions.size() > domain.size())
        return failure();
      const int64_t reductionStart =
          static_cast<int64_t>(domain.size() - reductions.size());
      SmallVector<int64_t> reductionShape;
      for (auto [ordinal, axis] : llvm::enumerate(reductions.asArrayRef())) {
        if (axis != reductionStart + static_cast<int64_t>(ordinal))
          return failure();
        reductionShape.push_back(domain.asArrayRef()[axis]);
      }
      FailureOr<int64_t> reductionElements =
          staticElementCount(reductionShape);
      if (failed(reductionElements))
        return failure();
      expectedSerial = *reductionElements;
    }
    if (failed(domainElements) ||
        *domainElements != elements.getInt() * serial.getInt() ||
        serial.getInt() != expectedSerial)
      return failure();
    nextOffset += count.getInt();
    maximumTile = std::max(maximumTile, tile.getInt());
    maximumThreads =
        std::max(maximumThreads, roundUpToSubgroup(tile.getInt()));
    maximumSerialTiles = std::max(maximumSerialTiles, serial.getInt());
  }
  return Simt32Geometry{{nextOffset},
                        {maximumTile},
                        maximumSerialTiles,
                        maximumThreads,
                        32,
                        0,
                        task.getReductionAxis()};
}

FailureOr<Simt32Geometry>
simt32Geometry(ScheduleTaskKind kind, ArrayRef<int64_t> domain,
               NumericalPolicy policy,
               std::optional<int64_t> foldReductionAxis = std::nullopt) {
  constexpr int64_t kSubgroup = 32;
  constexpr int64_t kMaxThreads = 256;
  if (kind == ScheduleTaskKind::OrderedState) {
    if (domain.size() != 1 || domain.front() <= 0)
      return failure();
    return Simt32Geometry{{1},
                          {domain.front()},
                          1,
                          128,
                          kSubgroup,
                          0,
                          std::nullopt};
  }
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
    if (domain.size() == 1 && domain.front() > 0) {
      const int64_t tile = std::min(domain.front(), kMaxThreads);
      const int64_t threads = roundUpToSubgroup(tile);
      return Simt32Geometry{
          {1},     {tile},    ceilDivPositive(domain.front(), tile),
          threads, kSubgroup, threads * static_cast<int64_t>(sizeof(float)),
          0};
    }
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
  if (kind == ScheduleTaskKind::AllAxisFold) {
    if (domain.empty() || domain.size() > 8)
      return failure();
    FailureOr<int64_t> reduction = staticElementCount(domain);
    if (failed(reduction))
      return failure();
    constexpr int64_t kItemsPerThread = 16;
    constexpr int64_t kPartialTile = kMaxThreads * kItemsPerThread;
    constexpr int64_t kMaximumMergeTiles = kMaxThreads;
    const int64_t partialCount = ceilDivPositive(*reduction, kPartialTile);
    const int64_t partialSerialTiles =
        ceilDivPositive(*reduction, partialCount * kMaxThreads);
    const int64_t mergeSerialTiles = ceilDivPositive(partialCount, kMaxThreads);
    if (mergeSerialTiles > kMaximumMergeTiles)
      return failure();
    return Simt32Geometry{
        {partialCount},     {kPartialTile},
        partialSerialTiles, kMaxThreads,
        kSubgroup,          kMaxThreads * static_cast<int64_t>(sizeof(float)),
        std::nullopt,       {1},
        {kMaxThreads},      mergeSerialTiles,
        {partialCount}};
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
  const int64_t reductionAxis = rowFold && foldReductionAxis
                                    ? *foldReductionAxis
                                    : (rowFold ? domain.size() - 1 : 0);
  if (reductionAxis < 0 || reductionAxis >= static_cast<int64_t>(domain.size()))
    return failure();
  SmallVector<int64_t> ownerShape;
  for (auto [axis, extent] : llvm::enumerate(domain))
    if (static_cast<int64_t>(axis) != reductionAxis)
      ownerShape.push_back(extent);
  FailureOr<int64_t> reduction = rowFold
                                     ? FailureOr<int64_t>(domain[reductionAxis])
                                     : staticElementCount(domain.drop_back());
  FailureOr<int64_t> outputs = rowFold ? staticElementCount(ownerShape)
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

FailureOr<Simt32Geometry> indexedAccumulationGeometry(ScheduleTaskOp task) {
  ArrayRef<int64_t> domain = task.getDomainShape();
  auto reductions = task.getReductionDimensionsAttr();
  if (domain.empty() || !reductions || reductions.empty())
    return failure();
  llvm::SmallDenseSet<int64_t> reductionSet;
  SmallVector<int64_t> ownerShape;
  SmallVector<int64_t> reductionShape;
  SmallVector<int64_t> tileShape(domain.size(), 1);
  for (int64_t axis : reductions.asArrayRef()) {
    if (axis < 0 || axis >= static_cast<int64_t>(domain.size()) ||
        !reductionSet.insert(axis).second)
      return failure();
    tileShape[axis] = domain[axis];
    reductionShape.push_back(domain[axis]);
  }
  for (auto [axis, extent] : llvm::enumerate(domain))
    if (!reductionSet.contains(axis))
      ownerShape.push_back(extent);
  FailureOr<int64_t> owners = staticElementCount(ownerShape);
  FailureOr<int64_t> reduction = staticElementCount(reductionShape);
  if (failed(owners) || failed(reduction) || *owners > UINT32_MAX)
    return failure();
  constexpr int64_t kThreads = 32;
  return Simt32Geometry{{*owners},
                        std::move(tileShape),
                        ceilDivPositive(*reduction, kThreads),
                        kThreads,
                        kThreads,
                        kThreads * 4,
                        reductions.asArrayRef().back()};
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
  if (threads <= 0 || threads > 1024 || scratch < 0 || scratch > 96 * 1024)
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

FailureOr<Simt32Geometry> onlineStateGeometry(ScheduleTaskOp task) {
  constexpr int64_t kOnlineWarpSize = 32;
  constexpr int64_t kOnlineWarpsPerBlock = 4;
  constexpr int64_t kOnlineMmaRowsPerBlock = 128;
  constexpr int64_t kOnlineMmaSharedMemoryBytes = 64 * 1024;
  constexpr int64_t kMaximumOnlineVectorElements = 128;
  constexpr int64_t kMaximumOnlineReductionElements = 2048;
  auto ownerDimensions = task.getCompositionOwnerDimensionsAttr();
  auto streamDimensions = task.getCompositionStreamDimensionsAttr();
  auto stateInputs = task.getCompositionStateInputsAttr();
  auto stateOutputs = task.getCompositionStateOutputsAttr();
  auto producerContractions =
      task.getCompositionProducerContractionDimensionsAttr();
  auto consumerContractions =
      task.getCompositionConsumerContractionDimensionsAttr();
  auto reductionAxis = task.getReductionAxis();
  auto vectorDimensions = task.getOnlineVectorDimensions();
  ArrayRef<int64_t> domain = task.getDomainShape();
  if (!ownerDimensions || !streamDimensions || !producerContractions ||
      !consumerContractions || ownerDimensions.empty() ||
      streamDimensions.empty() || producerContractions.empty() ||
      consumerContractions.empty())
    return failure();
  if (stateInputs || stateOutputs) {
    constexpr int64_t kThreads = 384;
    constexpr int64_t kOwnerAdjointTile = 64;
    constexpr int64_t kStreamedAdjointTile = 128;
    constexpr int64_t kOwnerStreamTile = 128;
    constexpr int64_t kAdjointStreamTile = 64;
    constexpr int64_t kSupportedVectorElements = 128;
    constexpr int64_t kScratchBytes = 194 * 1024;
    const bool hasStateInputs = stateInputs && !stateInputs.empty();
    const bool hasStateOutputs = stateOutputs && !stateOutputs.empty();
    const bool forward = !hasStateInputs && hasStateOutputs;
    const bool ownerAdjoint = hasStateInputs && hasStateOutputs;
    const bool streamedAdjoint = hasStateInputs && !hasStateOutputs;
    ArrayRef<int64_t> reductions = streamDimensions.asArrayRef();
    if (!vectorDimensions || vectorDimensions->empty() || domain.empty() ||
        domain.size() > 8 ||
        llvm::any_of(domain, [](int64_t extent) { return extent <= 0; }) ||
        (!forward && !ownerAdjoint && !streamedAdjoint) ||
        ((forward || ownerAdjoint) && reductions.size() != 1) ||
        (streamedAdjoint && reductions.size() != 2))
      return failure();

    llvm::SmallDenseSet<int64_t> partitionedDimensions;
    int64_t vectorElements = 1;
    auto addDimensions = [&](ArrayRef<int64_t> dimensions,
                             bool countVectorElements) -> LogicalResult {
      for (int64_t axis : dimensions) {
        if (axis < 0 || axis >= static_cast<int64_t>(domain.size()) ||
            !partitionedDimensions.insert(axis).second)
          return failure();
        if (countVectorElements) {
          if (vectorElements > kSupportedVectorElements / domain[axis])
            return failure();
          vectorElements *= domain[axis];
        }
      }
      return success();
    };
    if (failed(addDimensions(ownerDimensions.asArrayRef(), false)) ||
        failed(addDimensions(*vectorDimensions, true)) ||
        failed(addDimensions(reductions, false)) ||
        partitionedDimensions.size() != domain.size() ||
        vectorElements != kSupportedVectorElements)
      return failure();

    const int64_t ownerAxis = ownerDimensions.asArrayRef().back();
    const int64_t finalStream = reductions.back();
    SmallVector<int64_t> tile(domain.size(), 1);
    for (int64_t axis : *vectorDimensions)
      tile[axis] = domain[axis];
    for (int64_t axis : reductions.drop_back())
      tile[axis] = domain[axis];
    tile[ownerAxis] =
        std::min(domain[ownerAxis],
                 streamedAdjoint ? kStreamedAdjointTile : kOwnerAdjointTile);
    tile[finalStream] =
        std::min(domain[finalStream],
                 streamedAdjoint ? kAdjointStreamTile : kOwnerStreamTile);

    int64_t grid = 1;
    for (int64_t axis : ownerDimensions.asArrayRef().drop_back()) {
      if (grid > std::numeric_limits<int64_t>::max() / domain[axis])
        return failure();
      grid *= domain[axis];
    }
    const int64_t rowTiles =
        ceilDivPositive(domain[ownerAxis], tile[ownerAxis]);
    if (grid > std::numeric_limits<int64_t>::max() / rowTiles)
      return failure();
    grid *= rowTiles;
    const int64_t serialTiles =
        ceilDivPositive(domain[finalStream], tile[finalStream]);
    return Simt32Geometry{{grid}, std::move(tile), serialTiles, kThreads,
                          32,     kScratchBytes,   finalStream};
  }
  if (!reductionAxis || *reductionAxis < 0 ||
      *reductionAxis >= static_cast<int64_t>(domain.size()) ||
      !vectorDimensions || vectorDimensions->empty() || domain.empty() ||
      domain.size() > 8)
    return failure();
  llvm::SmallDenseSet<int64_t> vectorAxes;
  int64_t vectorElements = 1;
  for (int64_t axis : *vectorDimensions) {
    if (axis < 0 || axis >= static_cast<int64_t>(domain.size()) ||
        axis == *reductionAxis || !vectorAxes.insert(axis).second ||
        domain[axis] <= 0 ||
        vectorElements > kMaximumOnlineVectorElements / domain[axis])
      return failure();
    vectorElements *= domain[axis];
  }
  SmallVector<int64_t> outputShape;
  llvm::SmallDenseSet<int64_t> ownerAxes;
  for (int64_t axis : ownerDimensions.asArrayRef()) {
    if (axis < 0 || axis >= static_cast<int64_t>(domain.size()) ||
        axis == *reductionAxis || vectorAxes.contains(axis) ||
        !ownerAxes.insert(axis).second)
      return failure();
    outputShape.push_back(domain[axis]);
  }
  FailureOr<int64_t> outputs = staticElementCount(outputShape);
  const int64_t reduction = domain[*reductionAxis];
  if (failed(outputs) || *outputs > UINT32_MAX || reduction <= 0 ||
      reduction > kMaximumOnlineReductionElements)
    return failure();
  if (outputShape.size() < 2)
    return failure();
  FailureOr<int64_t> sharedRows =
      staticElementCount(ArrayRef<int64_t>(outputShape).drop_back(2));
  FailureOr<int64_t> localRows =
      staticElementCount(ArrayRef<int64_t>(outputShape).take_back(2));
  if (failed(sharedRows) || failed(localRows) || *sharedRows > UINT32_MAX ||
      *localRows > UINT32_MAX ||
      *sharedRows >
          UINT32_MAX / ceilDivPositive(*localRows, kOnlineWarpsPerBlock))
    return failure();
  SmallVector<int64_t> tile(domain.size(), 1);
  tile[*reductionAxis] = reduction;
  for (int64_t axis : vectorAxes)
    tile[axis] = domain[axis];
  const int64_t rowsPerBlock = task.getOnlineMma().value_or(false)
                                   ? kOnlineMmaRowsPerBlock
                                   : kOnlineWarpsPerBlock;
  return Simt32Geometry{
      {*sharedRows * ceilDivPositive(*localRows, rowsPerBlock)},
      std::move(tile),
      1,
      kOnlineWarpsPerBlock * kOnlineWarpSize,
      kOnlineWarpSize,
      task.getOnlineMma().value_or(false) ? kOnlineMmaSharedMemoryBytes
                                          : 96 * 1024,
      *reductionAxis};
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
  SmallVector<EventTensorOp> events;
  enum class ScheduleBodyPhase { Buffers, Tasks, Events };
  ScheduleBodyPhase phase = ScheduleBodyPhase::Buffers;
  for (Operation &operation : body.front().without_terminator()) {
    if (auto buffer = dyn_cast<ScheduleBufferOp>(operation)) {
      if (phase != ScheduleBodyPhase::Buffers) {
        return buffer.emitOpError("schedule buffers must precede tasks");
      }
      if (!buffer->getDiscardableAttrs().empty()) {
        return buffer.emitOpError("does not permit discardable attributes");
      }
      buffers.push_back(buffer);
      continue;
    }
    if (auto task = dyn_cast<ScheduleTaskOp>(operation)) {
      if (phase > ScheduleBodyPhase::Tasks)
        return task.emitOpError("schedule tasks must precede events");
      phase = ScheduleBodyPhase::Tasks;
      if (!task->getDiscardableAttrs().empty()) {
        return task.emitOpError("does not permit discardable attributes");
      }
      tasks.push_back(task);
      continue;
    }
    if (auto event = dyn_cast<EventTensorOp>(operation)) {
      if (phase > ScheduleBodyPhase::Events)
        return event.emitOpError(
            "Event Tensors must precede their realizations");
      phase = ScheduleBodyPhase::Events;
      if (!event->getDiscardableAttrs().empty())
        return event.emitOpError("does not permit discardable attributes");
      events.push_back(event);
      continue;
    }
    return emitOpError(
        "body may contain only schedule buffers, tasks, and Event Tensors");
  }
  if (buffers.empty() || tasks.empty()) {
    return emitOpError("requires at least one schedule buffer and task");
  }
  llvm::SmallDenseSet<std::pair<int64_t, int64_t>> parametricEdges;
  if (ArrayAttr programs = getParametricDependenciesAttr()) {
    for (Attribute attribute : programs) {
      auto program = dyn_cast<DictionaryAttr>(attribute);
      auto dependency = program
                            ? program.getAs<IntegerAttr>("dependency_ordinal")
                            : IntegerAttr{};
      auto producer =
          program ? program.getAs<IntegerAttr>("producer_task") : IntegerAttr{};
      auto consumer =
          program ? program.getAs<IntegerAttr>("consumer_task") : IntegerAttr{};
      auto source = program ? program.getAs<SourceRefAttr>("relation_source")
                            : SourceRefAttr{};
      auto semantics =
          program ? program.getAs<RelationSemanticsAttr>("relation_semantics")
                  : RelationSemanticsAttr{};
      auto shared = program ? program.getAs<DenseI64ArrayAttr>("shared_buffers")
                            : DenseI64ArrayAttr{};
      auto derivation =
          program ? program.getAs<StringAttr>("derivation") : StringAttr{};
      auto capacity = program ? program.getAs<IntegerAttr>("storage_capacity")
                              : IntegerAttr{};
      auto generations =
          program ? program.getAs<IntegerAttr>("generations") : IntegerAttr{};
      if (!program || program.size() != 9 || !dependency ||
          dependency.getInt() < 0 || !producer || !consumer || !source ||
          !semantics || semantics.getValue() != RelationSemantics::Gather ||
          !shared || shared.empty() || !derivation ||
          derivation.getValue() != "exact_consumer" || !capacity ||
          capacity.getInt() <= 0 || !generations || generations.getInt() <= 0 ||
          producer.getInt() < 0 || consumer.getInt() < 0 ||
          producer.getInt() >= static_cast<int64_t>(tasks.size()) ||
          consumer.getInt() >= static_cast<int64_t>(tasks.size()) ||
          !llvm::is_contained(tasks[consumer.getInt()].getDependencies(),
                              producer.getInt()) ||
          !parametricEdges.insert({producer.getInt(), consumer.getInt()})
               .second)
        return emitOpError(
            "runtime-parametric dependencies require unique bounded Gather "
            "program records");
    }
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
        (!type.getElementType().isBF16() && !type.getElementType().isF32() &&
         !type.getElementType().isInteger(1) &&
         !type.getElementType().isInteger(32))) {
      return buffer.emitOpError(
          "requires a positive static bf16, f32, i1, or i32 tensor type");
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
    auto indexDomainSource =
        task->getAttrOfType<SourceRefAttr>("composition_index_domain_source");
    auto indexDomainShape = task->getAttrOfType<DenseI64ArrayAttr>(
        "composition_index_domain_shape");
    auto indexDomainMap =
        task->getAttrOfType<AffineMapAttr>("composition_index_domain_map");
    const bool hasIndexDomain =
        indexDomainSource && indexDomainShape && indexDomainMap;
    if (hasIndexDomain != static_cast<bool>(indexDomainSource) ||
        hasIndexDomain != static_cast<bool>(indexDomainShape) ||
        hasIndexDomain != static_cast<bool>(indexDomainMap)) {
      return task.emitOpError(
          "composition index-domain provenance must be complete");
    }
    if (hasIndexDomain) {
      AffineMap map = indexDomainMap.getValue();
      if (llvm::any_of(indexDomainShape.asArrayRef(),
                       [](int64_t extent) { return extent <= 0; }) ||
          map.getNumSymbols() != 0 ||
          map.getNumDims() != task.getDomainShape().size() ||
          map.getNumResults() != indexDomainShape.size()) {
        return task.emitOpError(
            "composition index-domain map must map the task domain to the "
            "positive index-domain shape");
      }
    }
    auto orderedIndexSource =
        task->getAttrOfType<SourceRefAttr>("composition_ordered_index_source");
    auto orderedIndexDimensions = task->getAttrOfType<DenseI64ArrayAttr>(
        "composition_ordered_index_dimensions");
    auto orderedIndexInclusive =
        task->getAttrOfType<BoolAttr>("composition_ordered_index_inclusive");
    const bool hasOrderedIndex =
        orderedIndexSource && orderedIndexDimensions && orderedIndexInclusive;
    if (hasOrderedIndex != static_cast<bool>(orderedIndexSource) ||
        hasOrderedIndex != static_cast<bool>(orderedIndexDimensions) ||
        hasOrderedIndex != static_cast<bool>(orderedIndexInclusive) ||
        (hasOrderedIndex &&
         (!hasIndexDomain || orderedIndexDimensions.size() != 2 ||
          llvm::any_of(orderedIndexDimensions.asArrayRef(), [&](int64_t axis) {
            return axis < 0 || axis >= indexDomainShape.size();
          })))) {
      return task.emitOpError(
          "ordered index provenance must completely reference two "
          "index-domain dimensions");
    }
    auto stateAccessBuffers = task->getAttrOfType<DenseI64ArrayAttr>(
        "composition_state_access_buffers");
    auto stateAccessMaps =
        task->getAttrOfType<ArrayAttr>("composition_state_access_maps");
    if (static_cast<bool>(stateAccessBuffers) !=
        static_cast<bool>(stateAccessMaps)) {
      return task.emitOpError(
          "composition state buffer accesses require buffers and maps");
    }
    if (stateAccessBuffers) {
      if (!hasIndexDomain ||
          stateAccessBuffers.size() != stateAccessMaps.size())
        return task.emitOpError(
            "composition state buffer accesses require matching index-domain "
            "metadata");
      llvm::SmallDenseSet<int64_t> stateBuffers;
      llvm::SmallDenseSet<int64_t> declaredStateBuffers;
      for (StringRef name :
           {"composition_state_inputs", "composition_state_outputs"}) {
        if (auto attr = task->getAttrOfType<DenseI64ArrayAttr>(name))
          declaredStateBuffers.insert(attr.asArrayRef().begin(),
                                      attr.asArrayRef().end());
      }
      for (auto [bufferOrdinal, mapAttr] :
           llvm::zip_equal(stateAccessBuffers.asArrayRef(), stateAccessMaps)) {
        auto map = dyn_cast<AffineMapAttr>(mapAttr);
        if (bufferOrdinal < 0 ||
            bufferOrdinal >= static_cast<int64_t>(buffers.size()) || !map ||
            !stateBuffers.insert(bufferOrdinal).second ||
            !declaredStateBuffers.contains(bufferOrdinal)) {
          return task.emitOpError(
              "composition state access must uniquely reference a declared "
              "state buffer");
        }
        auto bufferType =
            dyn_cast<RankedTensorType>(buffers[bufferOrdinal].getTensorType());
        if (!bufferType || map.getValue().getNumSymbols() != 0 ||
            map.getValue().getNumDims() != task.getDomainShape().size() ||
            map.getValue().getNumResults() != bufferType.getRank()) {
          return task.emitOpError(
              "composition state access map must map the task domain to the "
              "state buffer rank");
        }
      }
    }
    const bool isFold = task.getKind() == ScheduleTaskKind::RowFold ||
                        task.getKind() == ScheduleTaskKind::ColumnFold ||
                        task.getKind() == ScheduleTaskKind::AllAxisFold;
    const bool isContract = task.getKind() == ScheduleTaskKind::Contract;
    const bool isSelection = task.getSelectionDimension().has_value();
    const bool hasSelectionMetadata =
        task.getSelectionCount().has_value() ||
        task.getSelectionOrder().has_value() ||
        task.getSelectionComparison().has_value() ||
        task.getSelectionStable().has_value() ||
        task.getSelectionTieBreak().has_value();
    const bool completeSelectionMetadata =
        task.getSelectionCount().has_value() &&
        task.getSelectionOrder().has_value() &&
        task.getSelectionComparison().has_value() &&
        task.getSelectionStable().has_value() &&
        task.getSelectionTieBreak().has_value();
    if (hasSelectionMetadata != completeSelectionMetadata ||
        isSelection != completeSelectionMetadata ||
        (isSelection &&
         (!isFold || *task.getSelectionDimension() < 0 ||
          *task.getSelectionDimension() >=
              static_cast<int64_t>(task.getDomainShape().size()) ||
          *task.getSelectionDimension() != *task.getReductionAxis() ||
          *task.getSelectionCount() <= 0 || !*task.getSelectionStable())))
      return task.emitOpError(
          "Selection schedule metadata must completely bind a stable bounded "
          "reduction");
    const bool hasRelationSemantics = task.getRelationSemantics().has_value();
    const bool hasRelationParameters = task.getRelationParameters().has_value();
    if (hasRelationSemantics != hasRelationParameters ||
        (hasRelationSemantics && task.getKind() != ScheduleTaskKind::Scalar &&
         task.getKind() != ScheduleTaskKind::Elementwise))
      return task.emitOpError(
          "Relation schedule metadata requires a scalar or elementwise task");
    const bool isOnline =
        isContract && task.getCompositionOwnerDimensionsAttr() &&
        task.getCompositionStreamDimensionsAttr() &&
        task.getCompositionProducerContractionDimensionsAttr() &&
        task.getCompositionConsumerContractionDimensionsAttr();
    const bool isIndexedAccumulation =
        isFold && task.getCompositionOwnerDimensionsAttr() &&
        task.getCompositionStreamDimensionsAttr() && !isOnline;
    std::optional<int64_t> stateBytes = task.getReductionStateBytesPerThread();
    if (stateBytes && (!isFold || isOnline || *stateBytes <= 0)) {
      return task.emitOpError(
          "reduction state storage requires a positive ordinary Fold value");
    }
    const std::optional<int64_t> foldAxis = task.getReductionAxis();
    auto foldDimensions = task.getReductionDimensions();
    const bool genericFoldComposition = stateBytes.has_value();
    if (task.getKind() == ScheduleTaskKind::AllAxisFold &&
        (foldAxis || !foldDimensions ||
         foldDimensions->size() != task.getDomainShape().size() ||
         llvm::any_of(llvm::enumerate(*foldDimensions), [](auto entry) {
           return entry.value() != static_cast<int64_t>(entry.index());
         }))) {
      return task.emitOpError(
          "all-axis Fold must reduce every domain dimension exactly once");
    }
    if (isFold && task.getKind() != ScheduleTaskKind::AllAxisFold &&
        foldDimensions && !isIndexedAccumulation) {
      return task.emitOpError(
          "single-axis Fold must not carry multi-axis reduction metadata");
    }
    if (isFold && !isOnline && !isIndexedAccumulation &&
        task.getKind() != ScheduleTaskKind::AllAxisFold &&
        (!foldAxis || *foldAxis < 0 ||
         *foldAxis >= static_cast<int64_t>(task.getDomainShape().size()) ||
         (task.getKind() == ScheduleTaskKind::ColumnFold && *foldAxis != 0) ||
         (!genericFoldComposition &&
          task.getKind() == ScheduleTaskKind::RowFold &&
          *foldAxis !=
              static_cast<int64_t>(task.getDomainShape().size()) - 1))) {
      return task.emitOpError(
          "Fold schedule kind must match its reduction axis");
    }
    std::optional<ScheduleReductionOrder> expectedOrder;
    if (isOnline) {
      expectedOrder = ScheduleReductionOrder::LeafOrderFree;
    } else if (isSelection) {
      expectedOrder = ScheduleReductionOrder::StableTotalOrderLeafOrderFixed;
    } else if (isFold) {
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
    auto phaseSchedules =
        task->getAttrOfType<ArrayAttr>("composition_phase_schedules");
    FailureOr<Simt32Geometry> expected =
        phaseSchedules          ? multiPhaseGeometry(task, phaseSchedules)
        : isOnline                ? onlineStateGeometry(task)
        : isIndexedAccumulation ? indexedAccumulationGeometry(task)
        : hasContractStrategy
            ? contractRegionGeometry(task, getPolicy())
            : simt32Geometry(task.getKind(), task.getDomainShape(), getPolicy(),
                             genericFoldComposition ? foldAxis : std::nullopt);
    if (succeeded(expected) && stateBytes) {
      if (expected->workgroupThreads >
          std::numeric_limits<int64_t>::max() / *stateBytes) {
        return task.emitOpError("reduction state scratch size overflows i64");
      }
      expected->scratchBytes = expected->workgroupThreads * *stateBytes;
      if (expected->scratchBytes > 96 * 1024) {
        return task.emitOpError(
            "reduction state scratch exceeds the bounded SM90 allocation");
      }
    }
    if (failed(expected) ||
        task.getGridShape() != ArrayRef<int64_t>(expected->grid) ||
        task.getTileShape() != ArrayRef<int64_t>(expected->tile) ||
        task.getSerialTiles() != expected->serialTiles ||
        task.getWorkgroupThreads() != expected->workgroupThreads ||
        task.getSubgroupSize() != expected->subgroupSize ||
        task.getScratchBytes() != expected->scratchBytes ||
        task.getReductionAxis() != expected->reductionAxis ||
        task.getReductionMergeGridShapeAttr() !=
            (expected->reductionMergeGrid.empty()
                 ? DenseI64ArrayAttr()
                 : DenseI64ArrayAttr::get(task.getContext(),
                                          expected->reductionMergeGrid)) ||
        task.getReductionMergeTileShapeAttr() !=
            (expected->reductionMergeTile.empty()
                 ? DenseI64ArrayAttr()
                 : DenseI64ArrayAttr::get(task.getContext(),
                                          expected->reductionMergeTile)) ||
        task.getReductionMergeSerialTiles() !=
            (expected->reductionMergeSerialTiles == 0
                 ? std::optional<int64_t>()
                 : std::optional<int64_t>(
                       expected->reductionMergeSerialTiles)) ||
        task.getReductionPartialBufferShapeAttr() !=
            (expected->reductionPartialBufferShape.empty()
                 ? DenseI64ArrayAttr()
                 : DenseI64ArrayAttr::get(
                       task.getContext(),
                       expected->reductionPartialBufferShape))) {
      return task.emitOpError(
          "schedule geometry must equal the SIMT32 derivation");
    }
    if (isFold) {
      ++foldCount;
    } else if (isContract) {
      ++contractCount;
    }
  }
  if ((contractCount != 0 && getSchemaVersion() != 2) ||
      (contractCount == 0 && getSchemaVersion() != 1 &&
       getSchemaVersion() != 2)) {
    return emitOpError("schedule schema must equal the admitted task family");
  }
  const bool eventized = getEventized().value_or(false);
  const bool hasTaskDependencies = llvm::any_of(tasks, [](ScheduleTaskOp task) {
    return !task.getDependencies().empty();
  });
  const bool hasLocalReduction = llvm::any_of(tasks, [](ScheduleTaskOp task) {
    return task.getKind() == ScheduleTaskKind::AllAxisFold;
  });
  if (eventized != (hasTaskDependencies || hasLocalReduction))
    return emitOpError(
        "eventized must exactly indicate whether task dependencies exist");
  if (!eventized && !events.empty())
    return emitOpError("Event Tensors require an eventized schedule plan");
  llvm::SmallDenseSet<std::pair<int64_t, int64_t>> eventEdges;
  for (auto [ordinal, event] : llvm::enumerate(events)) {
    const int64_t producer = event.getProducerTask();
    const int64_t consumer = event.getConsumerTask();
    const bool hasPhase = event.getProducerPhase().has_value() ||
                          event.getConsumerPhase().has_value();
    const bool localReduction =
        event.getProducerPhase() == "partial" &&
        event.getConsumerPhase() == "merge" && producer == consumer &&
        producer >= 0 && producer < static_cast<int64_t>(tasks.size()) &&
        tasks[producer].getKind() == ScheduleTaskKind::AllAxisFold;
    if (event.getOrdinal() != static_cast<int64_t>(ordinal) || producer < 0 ||
        hasPhase != localReduction ||
        (!localReduction && consumer <= producer) ||
        consumer >= static_cast<int64_t>(tasks.size()) ||
        (!localReduction &&
         !llvm::is_contained(tasks[consumer].getDependencies(), producer)) ||
        (!localReduction && parametricEdges.contains({producer, consumer})) ||
        !eventEdges.insert({producer, consumer}).second) {
      return event.emitOpError(
          "must uniquely eventize one ordered schedule-task dependency");
    }
    ArrayRef<int64_t> producerGrid = tasks[producer].getGridShape();
    ArrayRef<int64_t> consumerGrid =
        localReduction ? *tasks[consumer].getReductionMergeGridShape()
                       : tasks[consumer].getGridShape();
    ArrayRef<int64_t> relationDomain = event.getRelationDomainShape();
    AffineMap producerIndices = event.getProducerIndices();
    AffineMap consumerIndices = event.getConsumerIndices();
    std::optional<IntegerSet> relationPredicate = event.getRelationPredicate();
    FailureOr<int64_t> expectedStorageCapacity =
        staticElementCount(relationDomain);
    AffineExpr expectedStorageSlot = getAffineConstantExpr(0, getContext());
    for (auto [axis, extent] : llvm::enumerate(relationDomain))
      expectedStorageSlot =
          expectedStorageSlot * extent + getAffineDimExpr(axis, getContext());
    const AffineMap expectedStorageSlotMap = AffineMap::get(
        relationDomain.size(), 0, expectedStorageSlot, getContext());
    const AffineMap expectedGenerationMap =
        AffineMap::get(relationDomain.size(), 0,
                       getAffineConstantExpr(0, getContext()), getContext());
    if (relationDomain.empty() || failed(expectedStorageCapacity) ||
        producerIndices.getNumDims() != relationDomain.size() ||
        producerIndices.getNumSymbols() != 0 ||
        producerIndices.getNumResults() != producerGrid.size() ||
        consumerIndices.getNumDims() != relationDomain.size() ||
        consumerIndices.getNumSymbols() != 0 ||
        consumerIndices.getNumResults() != consumerGrid.size() ||
        (relationPredicate &&
         (relationPredicate->getNumDims() != relationDomain.size() ||
          relationPredicate->getNumSymbols() != 0)) ||
        event.getStorageCapacity() != *expectedStorageCapacity ||
        event.getStorageSlot() != expectedStorageSlotMap ||
        event.getGenerationIndex() != expectedGenerationMap) {
      return event.emitOpError(
          "scheduled relation must have bounded lexicographic "
          "generation-zero storage and maps into both task grids");
    }
    if (failed(verifyEventIndexAndSlotRanges(event, tasks[producer],
                                             tasks[consumer])))
      return failure();
    if (event.getGenerations() != 1 || event.getNotifyCount() != 1 ||
        event.getScope() != EventScope::Device || !event.getRelease() ||
        !event.getAcquire()) {
      return event.emitOpError(
          "scheduled task relation requires one generation, one "
          "notification per relation point, and device release/acquire");
    }
    if (localReduction && (relationDomain.size() != 1 ||
                           relationDomain.front() != producerGrid.front() ||
                           event.getStorageCapacity() != producerGrid.front()))
      return event.emitOpError(
          "partial-to-merge Event must cover every private partial exactly "
          "once");
  }
  if (eventized) {
    for (auto [consumer, task] : llvm::enumerate(tasks)) {
      for (int64_t producer : task.getDependencies())
        if (!eventEdges.contains({producer, static_cast<int64_t>(consumer)}) &&
            !parametricEdges.contains(
                {producer, static_cast<int64_t>(consumer)}))
          return task.emitOpError(
              "dependency is missing its exact Event Tensor");
      if (task.getKind() == ScheduleTaskKind::AllAxisFold &&
          !eventEdges.contains(
              {static_cast<int64_t>(consumer), static_cast<int64_t>(consumer)}))
        return task.emitOpError(
            "hierarchical Fold is missing its partial-to-merge Event Tensor");
    }
  }
  if (getFingerprint() != schedulePlanFingerprint(*this)) {
    return emitOpError("fingerprint does not match the closed schedule plan");
  }
  return success();
}

std::string eventLoweringPlanFingerprint(EventLoweringPlanOp plan) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  stream << "shuttle.event_lowering_plan.v1\n"
         << "schema_version=" << plan.getSchemaVersion() << '\n'
         << "source_schedule_fingerprint="
         << plan.getSourceScheduleFingerprint() << '\n';
  if (std::optional<StringRef> physicalFingerprint =
          plan.getPhysicalLaunchRecipeFingerprint())
    stream << "physical_launch_recipe_fingerprint=" << *physicalFingerprint
           << '\n';
  else
    stream << "source_bound_imperative=true\n";
  for (Operation &operation : plan.getBody().front()) {
    if (isa<EventLoweringPlanYieldOp>(operation))
      continue;
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

LogicalResult EventLoweringPlanOp::verifyRegions() {
  if (!getOperation()->getDiscardableAttrs().empty())
    return emitOpError("does not permit discardable attributes");
  std::optional<StringRef> physicalFingerprint =
      getPhysicalLaunchRecipeFingerprint();
  if (getSchemaVersion() != 1 ||
      !isLowerHexDigest(getSourceScheduleFingerprint()) ||
      (physicalFingerprint && !isLowerHexDigest(*physicalFingerprint)) ||
      !isLowerHexDigest(getFingerprint()))
    return emitOpError(
        "requires schema 1, SHA-256 schedule and plan bindings, and a "
        "SHA-256 launch-recipe binding when physically bound");
  Region &body = getBody();
  if (body.empty() || !llvm::hasSingleElement(body) ||
      !isa<EventLoweringPlanYieldOp>(body.front().getTerminator()))
    return emitOpError(
        "requires one block terminated by shuttle.event_lowering_plan_yield");
  llvm::SmallDenseSet<int64_t> events;
  llvm::SmallDenseMap<int64_t, ImperativeEventNotifyOp> notifications;
  llvm::SmallDenseMap<int64_t, ImperativeEventWaitOp> waits;
  for (Operation &operation : body.front().without_terminator()) {
    auto realization = dyn_cast<EventRealizationOp>(operation);
    if (!realization) {
      if (auto notification = dyn_cast<ImperativeEventNotifyOp>(operation)) {
        if (!notification->getDiscardableAttrs().empty())
          return notification.emitOpError(
              "does not permit discardable attributes");
        if (notification.getEvent() < 0 ||
            !notifications.try_emplace(notification.getEvent(), notification)
                 .second)
          return notification.emitOpError(
              "must uniquely reference one nonnegative Event Tensor ordinal");
        continue;
      }
      if (auto wait = dyn_cast<ImperativeEventWaitOp>(operation)) {
        if (!wait->getDiscardableAttrs().empty())
          return wait.emitOpError("does not permit discardable attributes");
        if (wait.getEvent() < 0 ||
            !waits.try_emplace(wait.getEvent(), wait).second)
          return wait.emitOpError(
              "must uniquely reference one nonnegative Event Tensor ordinal");
        continue;
      }
      return emitOpError(
          "body may contain only event realizations and imperative Event "
          "wait/notify operations");
    }
    if (!realization->getDiscardableAttrs().empty())
      return realization.emitOpError("does not permit discardable attributes");
    if (realization.getEvent() < 0 ||
        !events.insert(realization.getEvent()).second)
      return realization.emitOpError(
          "must uniquely reference one nonnegative Event Tensor ordinal");
    ArrayRef<int64_t> order = realization.getCoveringLaunchOrder();
    const bool imperative =
        realization.getKind() == EventRealizationKind::ImperativeWaitNotify;
    const bool physicalMechanism =
        realization.getKind() == EventRealizationKind::MBarrier ||
        realization.getKind() == EventRealizationKind::Pipeline ||
        realization.getKind() == EventRealizationKind::Queue;
    if (imperative && !order.empty())
      return realization.emitOpError(
          "imperative wait/notify realization cannot claim erasure witnesses");
    if (physicalMechanism && !physicalFingerprint)
      return realization.emitOpError(
          "physical Event mechanism requires a launch-recipe binding");
    if (physicalMechanism && !order.empty())
      return realization.emitOpError(
          "physical Event mechanism cannot claim erasure witnesses");
    if (!imperative && !physicalMechanism &&
        (order.empty() || order.size() % 4 != 0))
      return realization.emitOpError(
          "requires producer-launch, producer owner-local phase, "
          "consumer-launch, and consumer owner-local phase witnesses");
    if (imperative || physicalMechanism)
      continue;
    const bool algebraicRefinement =
        realization.getKind() ==
        EventRealizationKind::AlgebraicRefinementErased;
    bool crossesLaunch = false;
    for (size_t index = 0; index < order.size(); index += 4) {
      const int64_t producerLaunch = order[index];
      const int64_t producerPosition = order[index + 1];
      const int64_t consumerLaunch = order[index + 2];
      const int64_t consumerPosition = order[index + 3];
      if (producerLaunch < 0 || producerPosition < 0 || consumerLaunch < 0 ||
          consumerPosition < 0 || producerLaunch > consumerLaunch)
        return realization.emitOpError(
            "physical witnesses must order each producer before its consumer");
      if (algebraicRefinement) {
        if (producerLaunch != consumerLaunch)
          return realization.emitOpError(
              "algebraic refinement must fuse both Event endpoints into one "
              "physical launch");
      } else if (producerLaunch == consumerLaunch &&
                 producerPosition >= consumerPosition) {
        return realization.emitOpError(
            "physical witnesses must order each producer before its consumer");
      }
      crossesLaunch |= producerLaunch < consumerLaunch;
    }
    if (algebraicRefinement)
      continue;
    if (realization.getKind() == EventRealizationKind::ProgramOrderErased) {
      if (crossesLaunch)
        return realization.emitOpError(
            "program-order erasure requires same-launch witnesses");
      continue;
    }
    if (realization.getKind() == EventRealizationKind::StreamOrderErased) {
      if (!crossesLaunch)
        return realization.emitOpError(
            "stream-order erasure requires at least one cross-launch witness");
      continue;
    }
    return realization.emitOpError(
        "requires an erased Event Tensor or explicit imperative wait/notify "
        "realization");
  }
  for (int64_t event : events) {
    auto notification = notifications.find(event);
    auto wait = waits.find(event);
    EventRealizationOp realization;
    for (EventRealizationOp candidate :
         body.front().getOps<EventRealizationOp>())
      if (candidate.getEvent() == event) {
        realization = candidate;
        break;
      }
    const bool imperative =
        realization &&
        realization.getKind() == EventRealizationKind::ImperativeWaitNotify;
    if (!imperative) {
      if (notification != notifications.end() || wait != waits.end())
        return emitOpError(
            "erased Event Tensor cannot retain imperative wait/notify "
            "operations");
      continue;
    }
    if (notification == notifications.end() || wait == waits.end())
      return realization.emitOpError(
          "requires exactly one matching imperative notify and wait");
    for (NamedAttribute attribute : notification->second->getAttrs()) {
      StringRef name = attribute.getName().strref();
      if (name == "release")
        continue;
      if (wait->second->getAttr(attribute.getName()) != attribute.getValue())
        return realization.emitOpError(
            "requires wait and notify to carry identical exact Event "
            "metadata");
    }
    ArrayRef<int64_t> shape = notification->second.getRelationDomainShape();
    const int64_t dimensions = shape.size();
    FailureOr<presburger::IntegerPolyhedron> domain =
        boundedEventDomain(shape, notification->second.getRelationPredicate());
    if (failed(domain) ||
        notification->second.getProducerIndices().getNumDims() != dimensions ||
        notification->second.getProducerIndices().getNumSymbols() != 0 ||
        notification->second.getConsumerIndices().getNumDims() != dimensions ||
        notification->second.getConsumerIndices().getNumSymbols() != 0 ||
        notification->second.getProducerIndices().getNumResults() == 0 ||
        notification->second.getConsumerIndices().getNumResults() == 0)
      return notification->second.emitOpError(
          "requires symbol-free endpoint maps over a bounded exact relation");
    if (notification->second.getStorageCapacity() == 0 ||
        notification->second.getStorageCapacity() >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        notification->second.getGenerations() == 0 ||
        notification->second.getGenerations() >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        notification->second.getNotifyCount() <= 0)
      return notification->second.emitOpError(
          "requires positive storage capacity, generations, and notify count");
    std::array<int64_t, 1> storageShape{
        static_cast<int64_t>(notification->second.getStorageCapacity())};
    std::array<int64_t, 1> generationShape{
        static_cast<int64_t>(notification->second.getGenerations())};
    FailureOr<bool> slotInBounds = affineMapIsInBounds(
        notification->second.getStorageSlot(), *domain, storageShape);
    FailureOr<bool> generationInBounds = affineMapIsInBounds(
        notification->second.getGenerationIndex(), *domain, generationShape);
    if (failed(slotInBounds) || !*slotInBounds || failed(generationInBounds) ||
        !*generationInBounds)
      return notification->second.emitOpError(
          "storage-slot or generation map escapes its bounded Event range");
  }
  if (notifications.size() != waits.size() ||
      notifications.size() > events.size())
    return emitOpError("contains orphan imperative Event operations");
  for (auto [event, notification] : notifications)
    if (!events.contains(event))
      return notification.emitOpError(
          "references an Event Tensor without a realization");
  if (!physicalFingerprint)
    for (EventRealizationOp realization :
         body.front().getOps<EventRealizationOp>())
      if (realization.getKind() != EventRealizationKind::ImperativeWaitNotify)
        return realization.emitOpError(
            "source-bound Event lowering plan without a physical launch "
            "recipe must retain every Event as exact imperative wait/notify");
  if (!body.front().getTerminator()->getDiscardableAttrs().empty())
    return body.front().getTerminator()->emitOpError(
        "does not permit discardable attributes");
  if (getFingerprint() != eventLoweringPlanFingerprint(*this))
    return emitOpError(
        "fingerprint does not match the closed event lowering plan");
  return success();
}

LogicalResult InternalEventPlanOp::verifyRegions() {
  if (!getOperation()->getDiscardableAttrs().empty())
    return emitOpError("does not permit discardable attributes");
  if (getSchemaVersion() != 1 ||
      !isLowerHexDigest(getPhysicalScheduleFingerprint()) ||
      !isLowerHexDigest(getPhysicalLaunchRecipeFingerprint()) ||
      !isLowerHexDigest(getCanonicalScheduleDigest()) ||
      !isLowerHexDigest(getFingerprint()))
    return emitOpError(
        "requires schema 1 and SHA-256 schedule, launch-recipe, canonical, "
        "and provenance fingerprints");
  llvm::SHA256 scheduleDigest;
  scheduleDigest.update(getCanonicalSchedule());
  if (llvm::toHex(scheduleDigest.final(), true) != getCanonicalScheduleDigest())
    return emitOpError("canonical schedule digest does not match its payload");
  llvm::SHA256 provenanceDigest;
  provenanceDigest.update(getPhysicalScheduleFingerprint());
  provenanceDigest.update(getPhysicalLaunchRecipeFingerprint());
  provenanceDigest.update(getCanonicalScheduleDigest());
  if (llvm::toHex(provenanceDigest.final(), true) != getFingerprint())
    return emitOpError("fingerprint does not bind the internal schedule and "
                       "physical launch recipe");
  if (getBody().empty() || !llvm::hasSingleElement(getBody()) ||
      !isa<InternalEventPlanYieldOp>(getBody().front().getTerminator()))
    return emitOpError("requires one body block terminated by "
                       "shuttle.internal_event_plan_yield");
  for (Operation &operation : getBody().front()) {
    if (!isa<InternalEventTaskOp, InternalEventTensorOp,
             InternalEventAllocationOp, InternalEventRealizationOp,
             InternalEventNotifyOp, InternalEventWaitOp,
             InternalEventPlanYieldOp>(operation))
      return operation.emitOpError(
          "is not a structured refined internal Event plan record");
    if (!operation.getDiscardableAttrs().empty())
      return operation.emitOpError("does not permit discardable attributes");
  }

  using RecordKey = std::pair<int64_t, int64_t>;
  std::map<RecordKey, InternalEventTaskOp> tasks;
  std::map<RecordKey, InternalEventAllocationOp> allocations;
  std::set<std::tuple<int64_t, int64_t, int64_t>> notifications;
  std::set<std::tuple<int64_t, int64_t, int64_t>> waits;
  for (InternalEventTaskOp task :
       getBody().front().getOps<InternalEventTaskOp>())
    if (!tasks.emplace(RecordKey{task.getRegion(), task.getOrdinal()}, task)
             .second)
      return task.emitOpError("duplicates an internal task ordinal");
  for (InternalEventAllocationOp allocation :
       getBody().front().getOps<InternalEventAllocationOp>()) {
    if (!allocations
             .emplace(
                 RecordKey{allocation.getRegion(), allocation.getOrdinal()},
                 allocation)
             .second)
      return allocation.emitOpError(
          "duplicates an internal allocation ordinal");
    const bool hasTransactions =
        static_cast<bool>(allocation.getTransactionBytesByCoordinateAttr());
    const StringRef legalization = allocation.getLegalization();
    const bool valid = (allocation.getScope() == "cta" && !hasTransactions &&
                        (legalization == "cta_mbarrier" ||
                         legalization == "cta_async_completion_queue")) ||
                       (allocation.getScope() == "cta" && hasTransactions &&
                        legalization == "cta_transaction_mbarrier") ||
                       (allocation.getScope() == "device" && !hasTransactions &&
                        legalization == "device_semaphore_queue");
    if (!valid)
      return allocation.emitOpError(
          "legalization does not match scope and transaction mode");
  }

  auto taskMatchesRelation = [&](int64_t region, int64_t taskOrdinal,
                                 DictionaryAttr relation) {
    auto task = tasks.find(RecordKey{region, taskOrdinal});
    StringAttr family = relation.getAs<StringAttr>("task_family");
    return task != tasks.end() && family &&
           task->second.getName() == family.getValue();
  };
  auto verifyMechanismBinding =
      [&](Operation *operation, InternalEventAllocationOp allocation,
          int64_t coordinateOrdinal) -> LogicalResult {
    ArrayAttr coordinates = allocation.getMechanismCoordinates();
    ArrayAttr owners = allocation.getAllocationOwners();
    ArrayRef<int64_t> slots = allocation.getSlots();
    ArrayRef<int64_t> generations = allocation.getGenerations();
    ArrayRef<int64_t> initialCounts = allocation.getInitialCounts();
    if (coordinateOrdinal < 0 ||
        coordinateOrdinal >= static_cast<int64_t>(coordinates.size()) ||
        coordinates.size() != owners.size() ||
        coordinates.size() != slots.size() ||
        coordinates.size() != generations.size() ||
        coordinates.size() != initialCounts.size())
      return operation->emitOpError(
          "references an invalid internal mechanism coordinate");
    const size_t index = coordinateOrdinal;
    if (operation->getAttr("mechanism_coordinate") != coordinates[index] ||
        operation->getAttr("allocation_owner") != owners[index] ||
        operation->getAttrOfType<IntegerAttr>("slot").getInt() !=
            slots[index] ||
        operation->getAttrOfType<IntegerAttr>("generation").getInt() !=
            generations[index] ||
        operation->getAttrOfType<IntegerAttr>("generation_count").getInt() !=
            allocation.getGenerationCount() ||
        operation->getAttr("scope") != allocation.getScopeAttr() ||
        operation->getAttr("legalization") != allocation.getLegalizationAttr())
      return operation->emitOpError(
          "does not equal its bounded internal mechanism allocation");
    return success();
  };

  for (InternalEventNotifyOp notification :
       getBody().front().getOps<InternalEventNotifyOp>()) {
    RecordKey allocationKey{notification.getRegion(),
                            notification.getAllocation()};
    auto allocation = allocations.find(allocationKey);
    if (allocation == allocations.end())
      return notification.emitOpError(
          "references an unknown internal allocation");
    ArrayAttr relations = allocation->second.getNotifyRelations();
    if (notification.getRelation() < 0 ||
        notification.getRelation() >= static_cast<int64_t>(relations.size()))
      return notification.emitOpError(
          "references an unknown notification relation");
    auto relation = cast<DictionaryAttr>(relations[notification.getRelation()]);
    auto taskCoordinate = relation.getAs<DenseI64ArrayAttr>("task_coordinate");
    auto mechanismCoordinate =
        relation.getAs<DenseI64ArrayAttr>("mechanism_coordinate");
    if (failed(verifyMechanismBinding(
            notification, allocation->second,
            notification.getMechanismCoordinateOrdinal())))
      return failure();
    if (!taskMatchesRelation(notification.getRegion(), notification.getTask(),
                             relation) ||
        notification.getTaskCoordinateAttr() != taskCoordinate ||
        notification.getMechanismCoordinateAttr() != mechanismCoordinate ||
        notification.getArrivalCount() != 1 ||
        notification.getInitialCount() !=
            allocation->second.getInitialCounts()
                [notification.getMechanismCoordinateOrdinal()] ||
        notification.getRelease() != allocation->second.getRelease())
      return notification.emitOpError(
          "does not equal its exact internal notification relation");
    DenseI64ArrayAttr transactionBytes =
        allocation->second.getTransactionBytesByCoordinateAttr();
    IntegerAttr operationTransaction = notification.getTransactionBytesAttr();
    if (static_cast<bool>(transactionBytes) !=
            static_cast<bool>(operationTransaction) ||
        (transactionBytes &&
         operationTransaction.getInt() !=
             transactionBytes
                 .asArrayRef()[notification.getMechanismCoordinateOrdinal()]))
      return notification.emitOpError(
          "does not equal its internal notification transaction size");
    if (!notifications
             .insert({notification.getRegion(), notification.getAllocation(),
                      notification.getRelation()})
             .second)
      return notification.emitOpError(
          "duplicates an internal notification relation");
  }
  for (InternalEventWaitOp wait :
       getBody().front().getOps<InternalEventWaitOp>()) {
    RecordKey allocationKey{wait.getRegion(), wait.getAllocation()};
    auto allocation = allocations.find(allocationKey);
    if (allocation == allocations.end())
      return wait.emitOpError("references an unknown internal allocation");
    ArrayAttr relations = allocation->second.getWaitRelations();
    if (wait.getRelation() < 0 ||
        wait.getRelation() >= static_cast<int64_t>(relations.size()))
      return wait.emitOpError("references an unknown wait relation");
    auto relation = cast<DictionaryAttr>(relations[wait.getRelation()]);
    auto taskCoordinate = relation.getAs<DenseI64ArrayAttr>("task_coordinate");
    auto mechanismCoordinate =
        relation.getAs<DenseI64ArrayAttr>("mechanism_coordinate");
    if (failed(verifyMechanismBinding(wait, allocation->second,
                                      wait.getMechanismCoordinateOrdinal())))
      return failure();
    if (!taskMatchesRelation(wait.getRegion(), wait.getTask(), relation) ||
        wait.getTaskCoordinateAttr() != taskCoordinate ||
        wait.getMechanismCoordinateAttr() != mechanismCoordinate ||
        wait.getRequiredCount() !=
            allocation->second
                .getInitialCounts()[wait.getMechanismCoordinateOrdinal()] ||
        wait.getAcquire() != allocation->second.getAcquire())
      return wait.emitOpError(
          "does not equal its exact internal wait relation");
    if (!waits
             .insert(
                 {wait.getRegion(), wait.getAllocation(), wait.getRelation()})
             .second)
      return wait.emitOpError("duplicates an internal wait relation");
  }
  for (auto &[key, allocation] : allocations) {
    for (int64_t relation = 0;
         relation <
         static_cast<int64_t>(allocation.getNotifyRelations().size());
         ++relation)
      if (!notifications.count({key.first, key.second, relation}))
        return allocation.emitOpError(
            "is missing an imperative internal notification relation");
    for (int64_t relation = 0;
         relation < static_cast<int64_t>(allocation.getWaitRelations().size());
         ++relation)
      if (!waits.count({key.first, key.second, relation}))
        return allocation.emitOpError(
            "is missing an imperative internal wait relation");
  }
  return success();
}

namespace {

bool imperativeOperationMatchesEvent(Operation *operation,
                                     EventTensorOp event) {
  auto eventOrdinal = operation->getAttrOfType<IntegerAttr>("event");
  if (!eventOrdinal || eventOrdinal.getInt() != event.getOrdinal())
    return false;
  for (NamedAttribute attribute : event->getAttrs()) {
    if (attribute.getName().strref() == "ordinal")
      continue;
    if (attribute.getName().strref() == "release" &&
        isa<ImperativeEventWaitOp>(operation))
      continue;
    if (attribute.getName().strref() == "acquire" &&
        isa<ImperativeEventNotifyOp>(operation))
      continue;
    if (operation->getAttr(attribute.getName()) != attribute.getValue())
      return false;
  }
  return true;
}

} // namespace

LogicalResult verifyEventLoweringPlanAgainstSchedule(SchedulePlanOp schedule,
                                                     EventLoweringPlanOp plan) {
  if (failed(schedule.verifyRegions()) || failed(plan.verifyRegions()))
    return failure();
  if (plan.getSourceScheduleFingerprint() != schedule.getFingerprint())
    return plan.emitOpError("does not bind its source schedule");
  SmallVector<EventTensorOp> events(
      schedule.getBody().front().getOps<EventTensorOp>());
  SmallVector<EventRealizationOp> realizations(
      plan.getBody().front().getOps<EventRealizationOp>());
  if (events.size() != realizations.size())
    return plan.emitOpError("must realize every exact Event Tensor once");
  for (auto [ordinal, event, realization] :
       llvm::enumerate(events, realizations)) {
    if (event.getOrdinal() != static_cast<int64_t>(ordinal) ||
        realization.getEvent() != static_cast<int64_t>(ordinal))
      return realization.emitOpError(
          "does not correspond to the source Event Tensor ordinal");
    if (realization.getKind() != EventRealizationKind::ImperativeWaitNotify)
      continue;
    ImperativeEventNotifyOp notification;
    ImperativeEventWaitOp wait;
    for (Operation &operation : plan.getBody().front().without_terminator()) {
      if (auto candidate = dyn_cast<ImperativeEventNotifyOp>(operation);
          candidate && candidate.getEvent() == realization.getEvent())
        notification = candidate;
      if (auto candidate = dyn_cast<ImperativeEventWaitOp>(operation);
          candidate && candidate.getEvent() == realization.getEvent())
        wait = candidate;
    }
    if (!notification || !wait ||
        !imperativeOperationMatchesEvent(notification, event) ||
        !imperativeOperationMatchesEvent(wait, event))
      return realization.emitOpError(
          "imperative wait/notify metadata does not equal the exact source "
          "Event Tensor");
  }
  return success();
}

void createImperativeEventLowering(OpBuilder &builder, Location location,
                                   EventTensorOp event) {
  OperationState realizationState(location,
                                  EventRealizationOp::getOperationName());
  realizationState.addAttribute("event",
                                builder.getI64IntegerAttr(event.getOrdinal()));
  realizationState.addAttribute(
      "kind",
      EventRealizationKindAttr::get(
          builder.getContext(), EventRealizationKind::ImperativeWaitNotify));
  realizationState.addAttribute(
      "covering_launch_order",
      DenseI64ArrayAttr::get(builder.getContext(), {}));
  builder.create(realizationState);

  SmallVector<NamedAttribute> notifyAttributes;
  SmallVector<NamedAttribute> waitAttributes;
  notifyAttributes.push_back(builder.getNamedAttr(
      "event", builder.getI64IntegerAttr(event.getOrdinal())));
  waitAttributes.push_back(builder.getNamedAttr(
      "event", builder.getI64IntegerAttr(event.getOrdinal())));
  for (NamedAttribute attribute : event->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name == "ordinal")
      continue;
    if (name != "acquire")
      notifyAttributes.push_back(attribute);
    if (name != "release")
      waitAttributes.push_back(attribute);
  }
  OperationState notificationState(location,
                                   ImperativeEventNotifyOp::getOperationName());
  notificationState.addAttributes(notifyAttributes);
  builder.create(notificationState);
  OperationState waitState(location, ImperativeEventWaitOp::getOperationName());
  waitState.addAttributes(waitAttributes);
  builder.create(waitState);
}

EventLoweringPlanOp createSourceBoundImperativeEventLoweringPlan(
    OpBuilder &builder, Location location, SchedulePlanOp schedule) {
  OperationState planState(location, EventLoweringPlanOp::getOperationName());
  planState.addAttribute("schema_version", builder.getI64IntegerAttr(1));
  planState.addAttribute("source_schedule_fingerprint",
                         schedule.getFingerprintAttr());
  planState.addAttribute("fingerprint",
                         builder.getStringAttr(std::string(64, '0')));
  planState.addRegion();
  auto plan = cast<EventLoweringPlanOp>(builder.create(planState));
  plan.getBody().push_back(new Block());
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&plan.getBody().front());
  for (EventTensorOp event : schedule.getBody().front().getOps<EventTensorOp>())
    createImperativeEventLowering(builder, location, event);
  builder.create<EventLoweringPlanYieldOp>(location);
  plan.setFingerprint(eventLoweringPlanFingerprint(plan));
  return plan;
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
  if (auto integerType = dyn_cast<IntegerType>(elementType);
      integerType && integerType.getWidth() <= 32) {
    return std::max<int64_t>(1, (integerType.getWidth() + 7) / 8);
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
  } else if (module.getCodeFormat() == ExecutableCodeFormat::CudaFatbinSm90V1) {
    version = "shuttle.device_module.v5";
  } else if (module.getCodeFormat() ==
             ExecutableCodeFormat::CudaFatbinSm100V1) {
    version = "shuttle.device_module.v6";
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
       getCodeFormat() == ExecutableCodeFormat::CudaPtxSm100V1) ||
      (getSchemaVersion() == 5 &&
       getCodeFormat() == ExecutableCodeFormat::CudaFatbinSm90V1) ||
      (getSchemaVersion() == 6 &&
       getCodeFormat() == ExecutableCodeFormat::CudaFatbinSm100V1);
  if (!validVersionPair) {
    return emitOpError(
        "requires a matching device-module and executable-code version");
  }
  if (getCodeFormat() == ExecutableCodeFormat::CpuBytecodeV2 &&
      getPolicy() != NumericalPolicy::SourceOrdered) {
    return emitOpError("requires source_ordered policy for CPU bytecode v2");
  }
  const bool ptx = getCodeFormat() == ExecutableCodeFormat::CudaPtxSm90V1 ||
                   getCodeFormat() == ExecutableCodeFormat::CudaPtxSm100V1;
  const bool fatbin =
      getCodeFormat() == ExecutableCodeFormat::CudaFatbinSm90V1 ||
      getCodeFormat() == ExecutableCodeFormat::CudaFatbinSm100V1;
  const bool gpu = ptx || fatbin;
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
  if (fatbin) {
    auto transport = getNativeTransportAttr();
    auto requestFingerprint = getPhysicalRequestFingerprintAttr();
    auto compileRequestFingerprint = getPhysicalCompileRequestFingerprintAttr();
    auto launchRecipeFingerprint = getPhysicalLaunchRecipeFingerprintAttr();
    if (!transport || transport.empty() || !requestFingerprint ||
        !isLowerHexDigest(requestFingerprint.getValue()) ||
        !compileRequestFingerprint ||
        !isLowerHexDigest(compileRequestFingerprint.getValue()) ||
        !launchRecipeFingerprint ||
        !isLowerHexDigest(launchRecipeFingerprint.getValue())) {
      return emitOpError(
          "fatbin device module requires source-bound and physical-request "
          "native transport provenance");
    }
    if (!body.front().without_terminator().empty()) {
      return emitOpError(
          "fatbin launch entries are carried by the native transport");
    }
    if (ArrayAttr certificates = getOperation()->getAttrOfType<ArrayAttr>(
            "trusted_source_event_certificates")) {
      int64_t previousEvent = -1;
      for (Attribute attribute : certificates) {
        auto record = dyn_cast<DictionaryAttr>(attribute);
        auto event =
            record ? record.getAs<IntegerAttr>("event") : IntegerAttr{};
        auto digest = record ? record.getAs<StringAttr>("compiler_ir_digest")
                             : StringAttr{};
        if (!event || event.getInt() <= previousEvent || !digest ||
            !isLowerHexDigest(digest.getValue()))
          return emitOpError("trusted source Event certificates must be "
                             "ordered unique event/digest records");
        previousEvent = event.getInt();
      }
    }
    if (ArrayAttr snapshots = getDeviceIrSnapshotDigestsAttr()) {
      if (snapshots.empty() || llvm::any_of(snapshots, [](Attribute attribute) {
            auto digest = dyn_cast<StringAttr>(attribute);
            return !digest || !isLowerHexDigest(digest.getValue());
          }))
        return emitOpError(
            "device IR snapshots must be nonempty arrays of SHA-256 digests");
    }
    if (getFingerprint() != deviceModuleFingerprint(*this)) {
      return emitOpError("fingerprint does not match the closed device module");
    }
    return success();
  }
  if (getNativeTransportAttr() || getPhysicalRequestFingerprintAttr() ||
      getPhysicalCompileRequestFingerprintAttr() ||
      getPhysicalLaunchRecipeFingerprintAttr() ||
      getOperation()->hasAttr("trusted_source_event_certificates") ||
      getOperation()->hasAttr("trusted_internal_event_certificates") ||
      getDeviceIrSnapshotDigestsAttr()) {
    return emitOpError(
        "native transport is permitted only for CUDA fatbin modules");
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
    if (ptx && entry.getCodeDigest() != executableCodeDigest(entryCode)) {
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
    if (!ptx) {
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
        static_cast<uint64_t>(*entry.getDynamicSharedBytes()) >
            kMaximumGpuDynamicSharedMemoryBytes ||
        *entry.getKernelArity() !=
            static_cast<int64_t>(entry.getInputBuffers().size() +
                                 entry.getOutputBuffers().size())) {
      return entry.emitOpError("CUDA scratch or kernel arity is invalid");
    }
    auto onlineProtocol = entry.getOnlineStateProtocolAttr();
    auto logicalMaps = entry.getLogicalCoordinateMapsAttr();
    if (static_cast<bool>(onlineProtocol) != static_cast<bool>(logicalMaps))
      return entry.emitOpError(
          "online state protocol requires logical coordinate maps");
    if (onlineProtocol) {
      if (getPolicy() != NumericalPolicy::Fast ||
          onlineProtocol.getValue() != "stateful_contract_fold" ||
          entry.getReductionOrder() != ScheduleReductionOrder::LeafOrderFree ||
          logicalMaps.size() > 8)
        return entry.emitOpError(
            "online state entry requires FAST merge-order semantics");
      for (Attribute attribute : logicalMaps) {
        auto map = dyn_cast<AffineMapAttr>(attribute);
        if (!map || map.getValue().getNumDims() != 7 ||
            map.getValue().getNumSymbols() != 0 ||
            map.getValue().getNumResults() != 1)
          return entry.emitOpError(
              "online logical coordinate maps require seven physical axes");
      }
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
  llvm::SmallDenseSet<int64_t> sourceBuffers;
  for (Operation &operation : body.front().without_terminator()) {
    auto slot = dyn_cast<InvocationSlotOp>(operation);
    if (!slot) {
      return emitOpError("body may contain only invocation slots");
    }
    if (!slot->getDiscardableAttrs().empty()) {
      return slot.emitOpError("does not permit discardable attributes");
    }
    if (slot.getOrdinal() != ordinal)
      return slot.emitOpError("slot ordinals must be contiguous");
    const int64_t sourceBuffer = slot.getSourceBuffer();
    const bool physicalPrivate = sourceBuffer == -1;
    if (sourceBuffer < -1)
      return slot.emitOpError("source buffer ")
             << sourceBuffer
             << " must be nonnegative or -1 for physical-private storage";
    if (physicalPrivate &&
        slot.getStorage() != MaterializationStorage::Temporary)
      return slot.emitOpError("physical-private slot must be a temporary");
    if (!physicalPrivate && !sourceBuffers.insert(sourceBuffer).second)
      return slot.emitOpError("source buffer projection must be unique");
    ++ordinal;
    auto type = dyn_cast<RankedTensorType>(slot.getTensorType());
    FailureOr<int64_t> elementBytes =
        type ? executableElementBytes(type.getElementType())
             : FailureOr<int64_t>(failure());
    FailureOr<int64_t> elements = type ? staticElementCount(type.getShape())
                                       : FailureOr<int64_t>(failure());
    if (!type || !type.hasStaticShape() || failed(elementBytes) ||
        failed(elements) ||
        *elements > std::numeric_limits<int64_t>::max() / *elementBytes) {
      return slot.emitOpError("requires a bounded static bf16, f32, or "
                              "up-to-32-bit integer tensor");
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
    const int64_t alignment = slot.getAlignment();
    const bool hasNaturalAlignment = alignment >= *elementBytes &&
                                     alignment <= 256 &&
                                     (alignment & (alignment - 1)) == 0;
    if (slot.getOffset() != 0 || !hasNaturalAlignment ||
        slot.getAddressSpace() != expectedAddressSpace) {
      if (getSchemaVersion() == 3)
        return slot.emitOpError(
            "closed GPU ABI requires device storage, zero offset, and "
            "power-of-two alignment sufficient for the element type");
      return slot.emitOpError(
          "closed CPU ABI requires host storage, zero offset, and "
          "power-of-two alignment sufficient for the element type");
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
        (bindingIndex && (*bindingIndex < 0 ||
                          !bindingIndices->insert(*bindingIndex).second))) {
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
       getCompletion() == ExecutableCompletion::Synchronous &&
       !getEventLoweringPlanFingerprint()) ||
      (getSchemaVersion() == 2 &&
       getCompletion() == ExecutableCompletion::StreamOrdered &&
       getEventLoweringPlanFingerprint() &&
       isLowerHexDigest(*getEventLoweringPlanFingerprint()));
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
  EventLoweringPlanOp eventPlan;
  if (getEventLoweringPlanFingerprint()) {
    for (EventLoweringPlanOp candidate : module.getOps<EventLoweringPlanOp>()) {
      if (candidate.getFingerprint() != *getEventLoweringPlanFingerprint())
        continue;
      if (eventPlan)
        return emitOpError("event-lowering-plan fingerprint is not unique");
      eventPlan = candidate;
    }
    if (!eventPlan)
      return emitOpError("invalid closed GPU executable bundle");
  }
  std::optional<StringRef> expectedPhysicalFingerprint;
  if (eventPlan) {
    expectedPhysicalFingerprint =
        deviceModule.getPhysicalLaunchRecipeFingerprint()
            ? deviceModule.getPhysicalLaunchRecipeFingerprint()
            : std::optional<StringRef>(deviceModule.getFingerprint());
  }
  if (getSourceScheduleFingerprint() !=
          deviceModule.getSourceScheduleFingerprint() ||
      getSourceScheduleFingerprint() != abi.getSourceScheduleFingerprint() ||
      getDeviceModuleFingerprint() != deviceModule.getFingerprint() ||
      getInvocationAbiFingerprint() != abi.getFingerprint() ||
      (eventPlan && getSourceScheduleFingerprint() !=
                        eventPlan.getSourceScheduleFingerprint()) ||
      (eventPlan && (!eventPlan.getPhysicalLaunchRecipeFingerprint() ||
                     eventPlan.getPhysicalLaunchRecipeFingerprint() !=
                         expectedPhysicalFingerprint)) ||
      getFingerprint() != executableBundleFingerprint(*this)) {
    return emitOpError(getSchemaVersion() == 1
                           ? "invalid closed CPU executable bundle"
                           : "invalid closed GPU executable bundle");
  }
  const bool gpuBundle = getSchemaVersion() == 2;
  const bool gpuDevice = deviceModule.getSchemaVersion() == 3 ||
                         deviceModule.getSchemaVersion() == 4 ||
                         deviceModule.getSchemaVersion() == 5 ||
                         deviceModule.getSchemaVersion() == 6;
  if (gpuBundle != gpuDevice || gpuBundle != (abi.getSchemaVersion() == 3)) {
    return emitOpError("bundle schema does not match its device and ABI roots");
  }
  return success();
}

} // namespace mlir::shuttle
