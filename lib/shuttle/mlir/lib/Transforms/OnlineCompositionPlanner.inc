// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

std::string
compositionTaskFingerprint(StringRef kind, ArrayRef<Operation *> operations,
                           ArrayRef<Operation *> recomputedOperations) {
  if (recomputedOperations.empty())
    return compositionFingerprint(kind, operations);
  std::string normalized = compositionFingerprint(kind, operations);
  normalized.append("\nrecomputed\n");
  normalized.append(closedRegionFingerprint("composition-recomputed-v1",
                                            recomputedOperations));
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
  if (map.getSemantics() == MapSemantics::Reshape)
    return operandOrdinal == 0 && isClosedExactContiguousReshapeMap(map);
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

namespace online_planner {

bool foldYields(FoldOp fold, StringRef operationName) {
  if (fold.getCombiner().empty())
    return false;
  Block &body = fold.getCombiner().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  Operation *combiner = yield && yield.getValues().size() == 1
                            ? yield.getValues().front().getDefiningOp()
                            : nullptr;
  return body.getNumArguments() == 2 && combiner &&
         combiner->getName().getStringRef() == operationName &&
         combiner->getNumOperands() == 2 &&
         combiner->getOperand(0) == body.getArgument(0) &&
         combiner->getOperand(1) == body.getArgument(1);
}

FailureOr<AffineMap> coordinatesToAncestor(Value value, Value ancestor) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return failure();
  if (value == ancestor)
    return AffineMap::getMultiDimIdentityMap(type.getRank(),
                                             value.getContext());
  auto map = value.getDefiningOp<MapOp>();
  if (!map || map.getInputs().size() != 1 || map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != 2)
    return failure();
  auto inputMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[0]);
  auto resultMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[1]);
  if (!inputMap || !resultMap || !resultMap.getValue().isIdentity())
    return failure();
  FailureOr<AffineMap> upstream =
      coordinatesToAncestor(map.getInputs().front(), ancestor);
  if (failed(upstream) ||
      upstream->getNumDims() != inputMap.getValue().getNumResults())
    return failure();
  return upstream->compose(inputMap.getValue());
}

FailureOr<AffineMap> reductionProjection(RankedTensorType score,
                                         int64_t reduction) {
  if (reduction < 0 || reduction >= score.getRank())
    return failure();
  SmallVector<AffineExpr> coordinates;
  for (int64_t axis = 0; axis < score.getRank(); ++axis)
    if (axis != reduction)
      coordinates.push_back(getAffineDimExpr(axis, score.getContext()));
  return AffineMap::get(score.getRank(), 0, coordinates, score.getContext());
}

struct NormalizedExponentialFoldState {
  unsigned maximum;
  unsigned normalizer;
  Value score;
};

std::optional<APFloat> scalarMapConstant(Value value) {
  auto map = value.getDefiningOp<MapOp>();
  if (!map || !map.getInputs().empty() || map.getBody().empty())
    return std::nullopt;
  Block &body = map.getBody().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  auto constant = body.getOperations().size() == 2
                      ? dyn_cast<arith::ConstantOp>(body.front())
                      : arith::ConstantOp{};
  auto scalar = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
  if (!scalar || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != constant.getResult())
    return std::nullopt;
  return scalar.getValue();
}

bool isExactFoldInitializer(Value value, bool negativeInfinity) {
  std::optional<APFloat> scalar = scalarMapConstant(value);
  if (!scalar)
    return false;
  if (negativeInfinity)
    return scalar->isInfinity() && scalar->isNegative();
  return scalar->isZero() && !scalar->isNegative();
}

bool isExactPositiveOneMap(Value value) {
  std::optional<APFloat> scalar = scalarMapConstant(value);
  return scalar && scalar->isExactlyValue(1.0);
}

bool matchesNormalizedExponentialScaleTerm(Value value, Value sideMaximum,
                                           Value mergedMaximum,
                                           Value sideNormalizer) {
  auto multiply = value.getDefiningOp<arith::MulFOp>();
  if (!multiply)
    return false;
  Value exponentialValue;
  if (multiply.getLhs() == sideNormalizer)
    exponentialValue = multiply.getRhs();
  else if (multiply.getRhs() == sideNormalizer)
    exponentialValue = multiply.getLhs();
  else
    return false;
  auto exponential = exponentialValue.getDefiningOp<math::ExpOp>();
  auto delta = exponential
                   ? exponential.getOperand().getDefiningOp<arith::SubFOp>()
                   : arith::SubFOp{};
  return delta && delta.getLhs() == sideMaximum &&
         delta.getRhs() == mergedMaximum;
}

/// Classify only the selected Fold's typed associative state. The surrounding
/// score/finalizer Map and Relation programs are deliberately opaque here.
std::optional<NormalizedExponentialFoldState>
normalizedExponentialFoldState(FoldOp fold) {
  if (!isClosedMultiStatePlanningFold(fold) || fold.getInputs().size() != 2 ||
      fold.getInitializers().size() != 2 || fold.getNumResults() != 2)
    return std::nullopt;

  std::optional<unsigned> maximumOrdinal;
  std::optional<unsigned> normalizerOrdinal;
  for (unsigned ordinal = 0; ordinal < 2; ++ordinal) {
    if (isExactFoldInitializer(fold.getInitializers()[ordinal], true))
      maximumOrdinal = ordinal;
    if (isExactFoldInitializer(fold.getInitializers()[ordinal], false))
      normalizerOrdinal = ordinal;
  }
  if (!maximumOrdinal || !normalizerOrdinal ||
      *maximumOrdinal == *normalizerOrdinal ||
      !isExactPositiveOneMap(fold.getInputs()[*normalizerOrdinal]))
    return std::nullopt;

  Block &body = fold.getCombiner().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  const unsigned stateCount = 2;
  Value lhsMaximum = body.getArgument(*maximumOrdinal);
  Value rhsMaximum = body.getArgument(stateCount + *maximumOrdinal);
  Value lhsNormalizer = body.getArgument(*normalizerOrdinal);
  Value rhsNormalizer = body.getArgument(stateCount + *normalizerOrdinal);
  Value mergedMaximum = yield.getValues()[*maximumOrdinal];
  auto maximum = mergedMaximum.getDefiningOp<arith::MaximumFOp>();
  if (!maximum ||
      !((maximum.getLhs() == lhsMaximum && maximum.getRhs() == rhsMaximum) ||
        (maximum.getLhs() == rhsMaximum && maximum.getRhs() == lhsMaximum)))
    return std::nullopt;

  auto sum = yield.getValues()[*normalizerOrdinal].getDefiningOp<arith::AddFOp>();
  if (!sum)
    return std::nullopt;
  const bool direct = matchesNormalizedExponentialScaleTerm(
                          sum.getLhs(), lhsMaximum, mergedMaximum,
                          lhsNormalizer) &&
                      matchesNormalizedExponentialScaleTerm(
                          sum.getRhs(), rhsMaximum, mergedMaximum,
                          rhsNormalizer);
  const bool swapped = matchesNormalizedExponentialScaleTerm(
                           sum.getRhs(), lhsMaximum, mergedMaximum,
                           lhsNormalizer) &&
                       matchesNormalizedExponentialScaleTerm(
                           sum.getLhs(), rhsMaximum, mergedMaximum,
                           rhsNormalizer);
  if (!direct && !swapped)
    return std::nullopt;
  return NormalizedExponentialFoldState{*maximumOrdinal, *normalizerOrdinal,
                                        fold.getInputs()[*maximumOrdinal]};
}

bool collectMapSliceToAncestor(Value value, Value ancestor,
                               llvm::SmallDenseSet<Operation *> &visited,
                               SmallVectorImpl<MapOp> &maps);

std::optional<PlannedNormalizedExpContractRegion>
normalizedExponentialContractCapability(
    RegionOp region, FoldOp fold) {
  std::optional<NormalizedExponentialFoldState> state =
      normalizedExponentialFoldState(fold);
  if (!state)
    return std::nullopt;

  ContractOp producerContract;
  for (ContractOp contract : region.getBody().front().getOps<ContractOp>()) {
    if (!contract || !valueDependsOnOperation(state->score, contract))
      continue;
    if (producerContract)
      return std::nullopt;
    producerContract = contract;
  }
  if (!producerContract)
    return std::nullopt;

  ContractOp consumerContract;
  std::optional<unsigned> normalizedOperand;
  for (ContractOp candidate : region.getBody().front().getOps<ContractOp>()) {
    if (candidate == producerContract)
      continue;
    std::optional<unsigned> candidateOperand;
    for (auto [ordinal, operand] : llvm::enumerate(candidate.getInputs())) {
      if (!valueDependsOnOperation(operand, fold))
        continue;
      if (candidateOperand) {
        candidateOperand.reset();
        break;
      }
      candidateOperand = ordinal;
    }
    if (!candidateOperand)
      continue;
    consumerContract = candidate;
    normalizedOperand = candidateOperand;
    break;
  }
  if (!consumerContract || !normalizedOperand)
    return std::nullopt;
  Value normalizedWeight = consumerContract.getInputs()[*normalizedOperand];
  if (!normalizedWeight.getDefiningOp<MapOp>())
    return std::nullopt;

  auto regionYield = dyn_cast<YieldOp>(region.getBody().front().getTerminator());
  if (!regionYield)
    return std::nullopt;
  std::optional<unsigned> outputResult;
  Value finalizer;
  for (auto [ordinal, yielded] : llvm::enumerate(regionYield.getValues())) {
    llvm::SmallDenseSet<Operation *> visited;
    SmallVector<MapOp> maps;
    if (!collectMapSliceToAncestor(yielded, consumerContract.getResult(0),
                                   visited, maps))
      continue;
    if (finalizer)
      return std::nullopt;
    finalizer = yielded;
    outputResult = ordinal;
  }
  if (!finalizer) {
    auto consumerType =
        dyn_cast<RankedTensorType>(consumerContract.getResult(0).getType());
    for (MapOp map : region.getBody().front().getOps<MapOp>()) {
      auto resultType = map.getNumResults() == 1
                            ? dyn_cast<RankedTensorType>(map.getResult(0).getType())
                            : RankedTensorType{};
      if (!consumerType || !resultType ||
          consumerType.getElementType() == resultType.getElementType())
        continue;
      llvm::SmallDenseSet<Operation *> visited;
      SmallVector<MapOp> maps;
      if (!collectMapSliceToAncestor(map.getResult(0),
                                     consumerContract.getResult(0), visited,
                                     maps))
        continue;
      if (finalizer)
        return std::nullopt;
      finalizer = map.getResult(0);
    }
  }
  if (!finalizer || (finalizer != consumerContract.getResult(0) &&
                     !finalizer.getDefiningOp<MapOp>()))
    return std::nullopt;

  return PlannedNormalizedExpContractRegion{
      producerContract.getOperation(),
      state->score,
      {fold.getOperation()},
      fold.getResult(state->maximum),
      fold.getResult(state->normalizer),
      normalizedWeight,
      consumerContract.getOperation(),
      *normalizedOperand,
      finalizer,
      outputResult};
}

std::optional<PlannedNormalizedExpContractRegion>
normalizedExponentialContractCapability(
    RegionOp region, ArrayRef<Operation *> producerSources,
    ArrayRef<Operation *> consumerSources, FoldOp fold,
    ContractOp consumerContract) {
  std::optional<PlannedNormalizedExpContractRegion> capability =
      normalizedExponentialContractCapability(region, fold);
  if (!capability || capability->consumer != consumerContract ||
      !llvm::is_contained(producerSources, capability->producer) ||
      !llvm::is_contained(consumerSources,
                          capability->normalizedWeight.getDefiningOp()))
    return std::nullopt;
  return capability;
}

bool collectMapSliceToAncestor(Value value, Value ancestor,
                               llvm::SmallDenseSet<Operation *> &visited,
                               SmallVectorImpl<MapOp> &maps) {
  if (value == ancestor)
    return true;
  if (!valueDependsOnOperation(value, ancestor.getDefiningOp()))
    return false;
  auto map = value.getDefiningOp<MapOp>();
  if (!map || map.getNumResults() != 1 || map.getResult(0) != value)
    return false;
  if (!visited.insert(map).second)
    return true;

  bool reachesAncestor = false;
  for (Value input : map.getInputs()) {
    if (!valueDependsOnOperation(input, ancestor.getDefiningOp()))
      continue;
    if (!collectMapSliceToAncestor(input, ancestor, visited, maps))
      return false;
    reachesAncestor = true;
  }
  if (!reachesAncestor)
    return false;
  maps.push_back(map);
  return true;
}

FailureOr<ContractOp> nearestContractAncestor(Value value) {
  SmallVector<Value> worklist{value};
  llvm::SmallDenseSet<Operation *> visited;
  ContractOp found;
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (auto contract = dyn_cast<ContractOp>(producer)) {
      if (found && found != contract)
        return failure();
      found = contract;
      continue;
    }
    llvm::append_range(worklist, producer->getOperands());
  }
  if (!found)
    return failure();
  return found;
}

bool valueDependsOnBlockArgument(Value value, BlockArgument argument) {
  SmallVector<Value> worklist{value};
  llvm::SmallDenseSet<Operation *> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (current == argument)
      return true;
    Operation *producer = current.getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    llvm::append_range(worklist, producer->getOperands());
  }
  return false;
}

FailureOr<BlockArgument> uniqueBlockArgumentRoot(Value value, Block &body) {
  SmallVector<Value> worklist{value};
  llvm::SmallDenseSet<Operation *> visited;
  BlockArgument root;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (auto argument = dyn_cast<BlockArgument>(current)) {
      if (argument.getOwner() != &body || (root && root != argument))
        return failure();
      root = argument;
      continue;
    }
    Operation *producer = current.getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    llvm::append_range(worklist, producer->getOperands());
  }
  if (!root)
    return failure();
  return root;
}

SmallVector<int64_t> contractReductionDimensions(ContractOp contract) {
  SmallVector<int64_t> dimensions;
  for (auto [axis, iterator] : llvm::enumerate(contract.getIteratorKinds()))
    if (cast<StringAttr>(iterator).getValue() == "reduction")
      dimensions.push_back(axis);
  return dimensions;
}

struct NormalizedExponentialContractPullbackDescriptor {
  PlannedNormalizedExpContractRegion forward;
  FoldOp stateFold;
  ContractOp differentialContract;
  BlockArgument outputCotangent;
  FoldOp outputDependentFold;
};

FailureOr<SmallVector<NormalizedExponentialContractPullbackDescriptor>>
describeNormalizedExponentialContractPullbackCandidates(RegionOp region) {
  SmallVector<std::pair<PlannedNormalizedExpContractRegion, FoldOp>> forwards;
  for (FoldOp fold : region.getBody().front().getOps<FoldOp>()) {
    std::optional<PlannedNormalizedExpContractRegion> capability =
        normalizedExponentialContractCapability(region, fold);
    if (!capability || fold.getReductionDimensions().size() != 1)
      continue;
    forwards.push_back({std::move(*capability), fold});
  }
  if (forwards.size() != 1)
    return failure();
  const PlannedNormalizedExpContractRegion &forward = forwards.front().first;
  FoldOp stateFold = forwards.front().second;
  Block &body = region.getBody().front();

  SmallVector<NormalizedExponentialContractPullbackDescriptor> descriptors;
  for (ContractOp candidate : body.getOps<ContractOp>()) {
    if (candidate.getOperation() == forward.producer ||
        candidate.getOperation() == forward.consumer ||
        candidate.getInputs().size() != 2 || candidate.getNumResults() != 1 ||
        candidate.getResult(0).getType() != forward.normalizedWeight.getType())
      continue;
    SmallVector<BlockArgument> outputCotangents;
    for (BlockArgument argument : body.getArguments()) {
      if (!valueDependsOnBlockArgument(candidate.getResult(0), argument) ||
          valueDependsOnBlockArgument(forward.finalizer, argument))
        continue;
      outputCotangents.push_back(argument);
    }
    if (outputCotangents.size() != 1)
      continue;

    SmallVector<FoldOp> outputDependentCandidates;
    for (FoldOp fold : body.getOps<FoldOp>()) {
      if (fold == stateFold ||
          !foldYields(fold, arith::AddFOp::getOperationName()) ||
          fold.getInputs().size() != 1 ||
          fold.getReductionDimensions() !=
              stateFold.getReductionDimensions() ||
          !valueDependsOnOperation(fold.getInputs().front(), candidate) ||
          !valueDependsOnOperation(fold.getInputs().front(), stateFold))
        continue;
      outputDependentCandidates.push_back(fold);
    }
    for (FoldOp fold : outputDependentCandidates) {
      const bool dependsOnCandidate =
          llvm::any_of(outputDependentCandidates, [&](FoldOp other) {
            return other != fold &&
                   valueDependsOnOperation(fold.getInputs().front(), other);
          });
      if (dependsOnCandidate)
        continue;
      descriptors.push_back(
          {forward, stateFold, candidate, outputCotangents.front(), fold});
    }
  }
  if (descriptors.empty())
    return failure();
  return descriptors;
}

struct ContractOutputPartition {
  SmallVector<int64_t> ownerDimensions;
  SmallVector<int64_t> vectorDimensions;
};

FailureOr<ContractOutputPartition>
contractOutputPartition(ContractOp contract) {
  if (contract.getInputs().size() != 2 ||
      contract.getIndexingMaps().size() != 3)
    return failure();
  ArrayRef<Attribute> attributes = contract.getIndexingMaps().getValue();
  auto primary = dyn_cast<AffineMapAttr>(attributes[0]);
  auto secondary = dyn_cast<AffineMapAttr>(attributes[1]);
  auto result = dyn_cast<AffineMapAttr>(attributes[2]);
  if (!primary || !secondary || !result)
    return failure();

  ContractOutputPartition partition;
  llvm::SmallDenseSet<int64_t> seen;
  for (AffineExpr expression : result.getValue().getResults()) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || !seen.insert(dimension.getPosition()).second)
      return failure();
    const int64_t position = dimension.getPosition();
    const bool usedByPrimary = llvm::any_of(
        primary.getValue().getResults(), [&](AffineExpr operandExpression) {
          return operandExpression.isFunctionOfDim(position);
        });
    const bool usedBySecondary = llvm::any_of(
        secondary.getValue().getResults(), [&](AffineExpr operandExpression) {
          return operandExpression.isFunctionOfDim(position);
        });
    if (!usedByPrimary && !usedBySecondary)
      return failure();
    (usedBySecondary && !usedByPrimary ? partition.vectorDimensions
                                       : partition.ownerDimensions)
        .push_back(position);
  }
  if (partition.ownerDimensions.empty() || partition.vectorDimensions.empty())
    return failure();
  return partition;
}

struct ReverseOwnershipPartition {
  SmallVector<int64_t> reverseResults;
  SmallVector<int64_t> forwardResults;
  SmallVector<int64_t> residualResults;
  std::optional<int64_t> normalizedForwardResult;
  llvm::SmallDenseMap<int64_t, ContractOp> terminalContracts;
  SmallVector<int64_t> ownerResults;
  SmallVector<int64_t> streamedResults;
  int64_t streamedAnchorResult = -1;
};

struct CanonicalStreamedContractDomain {
  SmallVector<int64_t> extents;
  int64_t reductionRank;
};

FailureOr<CanonicalStreamedContractDomain>
canonicalStreamedContractDomain(ContractOp contract, Value yielded) {
  FailureOr<SmallVector<int64_t>> domain = contractDomainShape(contract);
  auto yieldedType = dyn_cast<RankedTensorType>(yielded.getType());
  auto resultType = dyn_cast<RankedTensorType>(contract.getResult(0).getType());
  FailureOr<AffineMap> yieldedToContract =
      coordinatesToAncestor(yielded, contract.getResult(0));
  auto resultMap =
      dyn_cast<AffineMapAttr>(contract.getIndexingMaps().getValue().back());
  if (failed(domain) || !yieldedType || !resultType ||
      failed(yieldedToContract) || !resultMap ||
      yieldedToContract->getNumDims() != yieldedType.getRank() ||
      yieldedToContract->getNumResults() != resultType.getRank() ||
      resultMap.getValue().getNumResults() != resultType.getRank())
    return failure();

  SmallVector<int64_t> publicDomainDimensions(yieldedType.getRank(), -1);
  llvm::SmallDenseSet<int64_t> seenDomainDimensions;
  for (auto [resultAxis, publicExpression] :
       llvm::enumerate(yieldedToContract->getResults())) {
    auto publicDimension = dyn_cast<AffineDimExpr>(publicExpression);
    auto domainDimension =
        dyn_cast<AffineDimExpr>(resultMap.getValue().getResult(resultAxis));
    if (!publicDimension || !domainDimension ||
        publicDimension.getPosition() >= publicDomainDimensions.size() ||
        domainDimension.getPosition() >= domain->size() ||
        publicDomainDimensions[publicDimension.getPosition()] != -1 ||
        !seenDomainDimensions.insert(domainDimension.getPosition()).second)
      return failure();
    publicDomainDimensions[publicDimension.getPosition()] =
        domainDimension.getPosition();
  }
  if (llvm::is_contained(publicDomainDimensions, -1))
    return failure();

  SmallVector<int64_t> canonicalDimensions = publicDomainDimensions;
  SmallVector<int64_t> reductions = contractReductionDimensions(contract);
  for (int64_t reduction : reductions) {
    if (reduction < 0 || reduction >= static_cast<int64_t>(domain->size()) ||
        !seenDomainDimensions.insert(reduction).second)
      return failure();
    canonicalDimensions.push_back(reduction);
  }
  if (canonicalDimensions.size() != domain->size() ||
      seenDomainDimensions.size() != domain->size())
    return failure();

  CanonicalStreamedContractDomain canonical;
  canonical.reductionRank = reductions.size();
  for (int64_t dimension : canonicalDimensions)
    canonical.extents.push_back((*domain)[dimension]);
  return canonical;
}

FailureOr<ReverseOwnershipPartition> deriveReverseOwnershipPartition(
    RegionOp region,
    const NormalizedExponentialContractPullbackDescriptor &pullback) {
  auto regionYield =
      dyn_cast<YieldOp>(region.getBody().front().getTerminator());
  if (!regionYield)
    return failure();

  ReverseOwnershipPartition partition;
  for (auto [ordinal, yielded] : llvm::enumerate(regionYield.getValues())) {
    FailureOr<ContractOp> terminal = nearestContractAncestor(yielded);
    if (valueDependsOnBlockArgument(yielded, pullback.outputCotangent) &&
        succeeded(terminal) &&
        terminal->getOperation() != pullback.forward.consumer) {
      partition.reverseResults.push_back(ordinal);
      continue;
    }
    // A public forward value is an index-preserving descendant of the
    // forward region boundary.  Reductions over that value (for example a
    // scalar objective) are residual algebra and must not be mistaken for a
    // differently shaped forward finalizer.
    FailureOr<AffineMap> coordinates =
        coordinatesToAncestor(yielded, pullback.forward.finalizer);
    (succeeded(coordinates) ? partition.forwardResults
                            : partition.residualResults)
        .push_back(ordinal);
  }
  if (partition.reverseResults.empty())
    return failure();
  for (int64_t result : partition.forwardResults) {
    if (!valueDependsOnOperation(regionYield.getValues()[result],
                                 pullback.stateFold))
      continue;
    if (partition.normalizedForwardResult)
      return failure();
    partition.normalizedForwardResult = result;
  }

  for (int64_t ordinal : partition.reverseResults) {
    FailureOr<ContractOp> contract =
        nearestContractAncestor(regionYield.getValues()[ordinal]);
    if (failed(contract) ||
        llvm::any_of(partition.terminalContracts, [&](const auto &entry) {
          return entry.second == *contract;
        }))
      return failure();
    partition.terminalContracts.insert({ordinal, *contract});
  }
  if (partition.normalizedForwardResult) {
    FailureOr<ContractOp> contract = nearestContractAncestor(
        regionYield.getValues()[*partition.normalizedForwardResult]);
    if (failed(contract))
      return failure();
    partition.terminalContracts.insert(
        {*partition.normalizedForwardResult, *contract});
  }

  for (int64_t ordinal : partition.reverseResults) {
    SmallVector<int64_t> reductions = contractReductionDimensions(
        partition.terminalContracts.lookup(ordinal));
    if (reductions.size() == 1)
      partition.ownerResults.push_back(ordinal);
    else if (reductions.size() == 2)
      partition.streamedResults.push_back(ordinal);
    else
      return failure();
  }
  if (partition.ownerResults.size() != 1 ||
      partition.streamedResults.size() != 2)
    return failure();

  llvm::SmallDenseSet<BlockArgument> producerOperandRoots;
  for (Value input : pullback.forward.producer->getOperands()) {
    FailureOr<BlockArgument> root =
        uniqueBlockArgumentRoot(input, region.getBody().front());
    if (failed(root) || !producerOperandRoots.insert(*root).second)
      return failure();
  }
  SmallVector<int64_t> anchors;
  llvm::copy_if(
      partition.streamedResults, std::back_inserter(anchors),
      [&](int64_t result) {
        ContractOp terminal = partition.terminalContracts.lookup(result);
        return llvm::any_of(terminal.getInputs(), [&](Value input) {
          FailureOr<BlockArgument> root =
              uniqueBlockArgumentRoot(input, region.getBody().front());
          return succeeded(root) && producerOperandRoots.contains(*root);
        });
      });
  if (anchors.size() != 1)
    return failure();
  partition.streamedAnchorResult = anchors.front();

  FailureOr<CanonicalStreamedContractDomain> anchorDomain =
      canonicalStreamedContractDomain(
          partition.terminalContracts.lookup(partition.streamedAnchorResult),
          regionYield.getValues()[partition.streamedAnchorResult]);
  if (failed(anchorDomain))
    return failure();
  for (int64_t result : partition.streamedResults) {
    if (result == partition.streamedAnchorResult)
      continue;
    FailureOr<CanonicalStreamedContractDomain> domain =
        canonicalStreamedContractDomain(
            partition.terminalContracts.lookup(result),
            regionYield.getValues()[result]);
    if (failed(domain) || domain->extents != anchorDomain->extents ||
        domain->reductionRank != anchorDomain->reductionRank)
      return failure();
  }
  return partition;
}

FailureOr<DerivedPlan>
deriveNormalizedExponentialReverseFamily(RegionOp region,
                                         const DerivedPlan &materialized) {
  FailureOr<SmallVector<NormalizedExponentialContractPullbackDescriptor>>
      candidates =
          describeNormalizedExponentialContractPullbackCandidates(region);
  auto regionYield =
      dyn_cast<YieldOp>(region.getBody().front().getTerminator());
  if (failed(candidates) || !regionYield) {
    return failure();
  }
  std::optional<NormalizedExponentialContractPullbackDescriptor> pullback;
  std::optional<ReverseOwnershipPartition> ownership;
  for (const NormalizedExponentialContractPullbackDescriptor &candidate :
       *candidates) {
    FailureOr<ReverseOwnershipPartition> candidateOwnership =
        deriveReverseOwnershipPartition(region, candidate);
    if (failed(candidateOwnership))
      continue;
    if (pullback)
      return failure();
    pullback = candidate;
    ownership = std::move(*candidateOwnership);
  }
  if (!pullback || !ownership) {
    return failure();
  }

  Block &body = region.getBody().front();
  ContractOp differentialContract = pullback->differentialContract;
  BlockArgument outputCotangent = pullback->outputCotangent;
  SmallVector<int64_t> &reverseResults = ownership->reverseResults;
  SmallVector<int64_t> &forwardResults = ownership->forwardResults;
  SmallVector<int64_t> &residualResults = ownership->residualResults;
  std::optional<int64_t> normalizedForwardResult =
      ownership->normalizedForwardResult;
  llvm::SmallDenseMap<int64_t, ContractOp> &terminalContracts =
      ownership->terminalContracts;
  SmallVector<int64_t> &ownerResults = ownership->ownerResults;
  SmallVector<int64_t> &streamedResults = ownership->streamedResults;
  const int64_t streamedAnchorResult = ownership->streamedAnchorResult;
  const int64_t queryResult = ownerResults.front();

  SmallVector<Operation *> algebra;
  for (Operation &operation : region.getBody().front().without_terminator())
    algebra.push_back(&operation);
  if (algebra.empty())
    return failure();

  // Residual non-differential results may consume the forward boundary
  // through a generic Map/Fold objective.  Keep that dependency-convex
  // subgraph as its own Fold task family; the reverse family sees values that
  // escape the subgraph as ordinary typed inputs.
  llvm::SmallDenseSet<Operation *> residualSet;
  SmallVector<FoldOp> residualFolds;
  for (int64_t result : residualResults) {
    SmallVector<Value> worklist{regionYield.getValues()[result]};
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      Operation *producer = value.getDefiningOp();
      if (!producer || value == pullback->forward.finalizer)
        continue;
      if (producer->getParentOfType<RegionOp>() != region ||
          !isa<MapOp, FoldOp>(producer))
        return failure();
      if (!residualSet.insert(producer).second)
        continue;
      if (auto fold = dyn_cast<FoldOp>(producer))
        residualFolds.push_back(fold);
      llvm::append_range(worklist, producer->getOperands());
    }
  }
  if ((!residualResults.empty() && residualFolds.size() != 1) ||
      (residualResults.empty() && !residualSet.empty())) {
    return failure();
  }
  SmallVector<Operation *> coreAlgebra;
  SmallVector<Operation *> residualAlgebra;
  for (Operation *operation : algebra) {
    (residualSet.contains(operation) ? residualAlgebra : coreAlgebra)
        .push_back(operation);
  }

  DerivedPlan plan;
  plan.hasContract = true;
  plan.hasRankNFold = materialized.hasRankNFold;
  plan.hasContractFoldComposition = true;
  plan.hasLogicalRelation = materialized.hasLogicalRelation;
  plan.hasStatefulContractFoldPullback = true;
  const bool hasResidualTask = !residualResults.empty();
  const bool hasForwardTask = !forwardResults.empty() || hasResidualTask;
  const int64_t residualTask = hasForwardTask ? 1 : 0;
  const int64_t queryTask = (hasForwardTask ? 1 : 0) +
                            (hasResidualTask ? 1 : 0);
  const int64_t keyValueTask = queryTask + 1;
  for (auto [ordinal, argument] :
       llvm::enumerate(region.getBody().front().getArguments())) {
    SmallVector<int64_t> consumers;
    if (hasForwardTask && argument != outputCotangent)
      consumers.push_back(0);
    if (hasResidualTask && valueDependsOnBlockArgument(
                               regionYield.getValues()[residualResults.front()],
                               argument))
      consumers.push_back(residualTask);
    consumers.push_back(queryTask);
    consumers.push_back(keyValueTask);
    plan.buffers.push_back(
        {argument, std::nullopt, std::move(consumers), false});
  }
  for (auto [ordinal, yielded] : llvm::enumerate(regionYield.getValues())) {
    int64_t producer = hasResidualTask &&
                               llvm::is_contained(residualResults,
                                                  static_cast<int64_t>(ordinal))
                           ? residualTask
                           : keyValueTask;
    SmallVector<int64_t> consumers;
    if (llvm::is_contained(ownerResults, static_cast<int64_t>(ordinal)))
      producer = queryTask;
    else if (hasForwardTask &&
             llvm::is_contained(forwardResults,
                                static_cast<int64_t>(ordinal))) {
      producer = 0;
      consumers.push_back(queryTask);
    }
    plan.buffers.push_back({yielded, producer, std::move(consumers), true});
  }
  SmallVector<int64_t> stateBuffers;
  SmallVector<Value, 3> stateValues{
      pullback->forward.maximum,
      pullback->forward.normalizer,
      pullback->outputDependentFold.getResult(0)};
  for (auto [ordinal, state] : llvm::enumerate(stateValues)) {
    stateBuffers.push_back(plan.buffers.size());
    const bool forwardState = hasForwardTask && ordinal < 2;
    plan.buffers.push_back(
        {state, forwardState ? 0 : queryTask,
         ordinal < 2 ? SmallVector<int64_t>({queryTask, keyValueTask})
                     : SmallVector<int64_t>({keyValueTask}),
         false});
  }

  std::optional<int64_t> forwardBoundaryBuffer;
  if (hasResidualTask) {
    forwardBoundaryBuffer = plan.buffers.size();
    plan.buffers.push_back({pullback->forward.finalizer, int64_t{0},
                            SmallVector<int64_t>({residualTask}), false});
  }
  llvm::SmallDenseMap<Value, int64_t> residualBoundaryBuffers;
  if (hasResidualTask) {
    for (Operation *operation : residualAlgebra) {
      for (Value result : operation->getResults()) {
        const bool escapes = llvm::any_of(result.getUses(), [&](OpOperand &use) {
          return !residualSet.contains(use.getOwner()) &&
                 !isa<YieldOp>(use.getOwner());
        });
        if (!escapes)
          continue;
        int64_t buffer = plan.buffers.size();
        residualBoundaryBuffers.try_emplace(result, buffer);
        plan.buffers.push_back(
            {result, residualTask,
             SmallVector<int64_t>({queryTask, keyValueTask}), false});
      }
    }
  }

  ContractOp producerContract = cast<ContractOp>(pullback->forward.producer);
  SmallVector<int64_t> producerContractionDimensions =
      contractReductionDimensions(producerContract);
  SmallVector<int64_t> consumerContractionDimensions =
      contractReductionDimensions(differentialContract);
  auto indexDomainType =
      dyn_cast<RankedTensorType>(pullback->forward.score.getType());
  Operation *indexDomainOperation =
      pullback->forward.score.getDefiningOp();
  if (producerContractionDimensions.empty() ||
      consumerContractionDimensions.empty() || !indexDomainType ||
      !indexDomainType.hasStaticShape() || !indexDomainOperation)
    return failure();

  RelationOp orderedIndexRelation;
  for (Operation *operation : algebra) {
    auto relation = dyn_cast<RelationOp>(operation);
    if (!relation || relation.getSemantics() != RelationSemantics::Compare)
      continue;
    if (orderedIndexRelation)
      return failure();
    orderedIndexRelation = relation;
  }
  if (!orderedIndexRelation || orderedIndexRelation.getInputs().size() != 2 ||
      orderedIndexRelation.getParameters().size() < 2)
    return failure();
  auto orderedDirection =
      dyn_cast<StringAttr>(orderedIndexRelation.getParameters()[1]);
  std::optional<AffineMap> orderedIndexDomainMap;
  for (Operation *operation : algebra) {
    auto relation = dyn_cast<RelationOp>(operation);
    auto relationType =
        relation && relation.getNumResults() == 1
            ? dyn_cast<RankedTensorType>(relation.getResult(0).getType())
            : RankedTensorType{};
    if (!relationType || !relationType.hasStaticShape() ||
        relationType.getShape() != indexDomainType.getShape() ||
        !valueDependsOnValue(relation.getResult(0),
                             orderedIndexRelation.getResult(0)))
      continue;
    FailureOr<AffineMap> coordinates = relationCoordinatesToAncestor(
        relation.getResult(0), orderedIndexRelation.getResult(0));
    if (failed(coordinates))
      continue;
    if (orderedIndexDomainMap && *orderedIndexDomainMap != *coordinates) {
      relation.emitOpError(
          "ordered comparison has inconsistent index-domain coordinate maps");
      return failure();
    }
    orderedIndexDomainMap = *coordinates;
  }
  if (!orderedIndexDomainMap) {
    orderedIndexRelation.emitOpError(
        "cannot find an index-domain descendant of ordered comparison");
    return failure();
  }
  std::optional<int64_t> lesserIndexDimension = relationVaryingAxis(
      orderedIndexRelation.getInputs()[0], *orderedIndexDomainMap);
  std::optional<int64_t> greaterIndexDimension = relationVaryingAxis(
      orderedIndexRelation.getInputs()[1], *orderedIndexDomainMap);
  if (!orderedDirection ||
      (orderedDirection.getValue() != "LE" &&
       orderedDirection.getValue() != "LT") ||
      !lesserIndexDimension || !greaterIndexDimension ||
      *lesserIndexDimension < 0 || *greaterIndexDimension < 0 ||
      *lesserIndexDimension >= indexDomainType.getRank() ||
      *greaterIndexDimension >= indexDomainType.getRank()) {
    orderedIndexRelation.emitOpError(
        "cannot derive ordered comparison axes in the index domain");
    return failure();
  }

  auto indexDomainMap = [&](ContractOp anchor) -> FailureOr<AffineMap> {
    std::optional<AffineMap> found;
    for (auto [ordinal, input] : llvm::enumerate(anchor.getInputs())) {
      if (!valueDependsOnValue(input, pullback->forward.score))
        continue;
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      auto map = dyn_cast<AffineMapAttr>(anchor.getIndexingMaps()[ordinal]);
      if (!inputType || inputType != indexDomainType || !map ||
          map.getValue().getNumSymbols() != 0 ||
          map.getValue().getNumResults() != indexDomainType.getRank() ||
          (found && *found != map.getValue()))
        return failure();
      found = map.getValue();
    }
    return found ? FailureOr<AffineMap>(*found)
                 : FailureOr<AffineMap>(failure());
  };

  auto makeTask =
      [&](ContractOp anchor, ArrayRef<int64_t> outputs,
          ArrayRef<int64_t> dependencies) -> FailureOr<PlannedTask> {
    FailureOr<SmallVector<int64_t>> domain = contractDomainShape(anchor);
    FailureOr<ContractOutputPartition> outputPartition =
        contractOutputPartition(anchor);
    if (failed(domain) || failed(outputPartition))
      return failure();
    PlannedTask task;
    task.operation = anchor;
    task.kind = MaterializationTaskKind::Contract;
    task.domain = std::move(*domain);
    for (auto [position, iterator] : llvm::enumerate(anchor.getIteratorKinds()))
      if (cast<StringAttr>(iterator).getValue() == "reduction")
        task.reductionDimensions.push_back(position);
    for (auto [ordinal, argument] : llvm::enumerate(body.getArguments())) {
      (void)argument;
      task.inputs.push_back(ordinal);
    }
    task.outputs.assign(outputs.begin(), outputs.end());
    task.dependencies.assign(dependencies.begin(), dependencies.end());
    task.statefulContractFoldComposition = true;
    task.compositionSources = coreAlgebra;
    task.compositionOwnerDimensions = outputPartition->ownerDimensions;
    task.onlineVectorDimensions = outputPartition->vectorDimensions;
    task.compositionStreamDimensions = task.reductionDimensions;
    task.compositionProducerContractionDimensions =
        producerContractionDimensions;
    task.compositionConsumerContractionDimensions =
        consumerContractionDimensions;
    FailureOr<AffineMap> taskIndexDomainMap = indexDomainMap(anchor);
    if (failed(taskIndexDomainMap))
      return failure();
    task.compositionIndexDomainSource =
        algebraSource(indexDomainOperation);
    task.compositionIndexDomainShape.assign(indexDomainType.getShape().begin(),
                                            indexDomainType.getShape().end());
    task.compositionIndexDomainMap = *taskIndexDomainMap;
    task.compositionOrderedIndexSource =
        algebraSource(orderedIndexRelation.getOperation());
    task.compositionOrderedIndexDimensions = {*lesserIndexDimension,
                                               *greaterIndexDimension};
    task.compositionOrderedIndexInclusive =
        orderedDirection.getValue() == "LE";
    task.onlineMma = true;
    return task;
  };

  const int64_t indexReduction =
      pullback->stateFold.getReductionDimensions().front();
  auto addStateAccesses = [&](PlannedTask &task,
                              ArrayRef<int64_t> stateBuffersToAccess)
      -> LogicalResult {
    if (!task.compositionIndexDomainMap || indexReduction < 0 ||
        indexReduction >=
            static_cast<int64_t>(task.compositionIndexDomainShape.size()))
      return failure();
    SmallVector<AffineExpr> stateCoordinates;
    for (auto [axis, coordinate] :
         llvm::enumerate(task.compositionIndexDomainMap.getResults()))
      if (static_cast<int64_t>(axis) != indexReduction)
        stateCoordinates.push_back(coordinate);
    AffineMap stateMap = AffineMap::get(
        task.compositionIndexDomainMap.getNumDims(), 0, stateCoordinates,
        task.compositionIndexDomainMap.getContext());
    for (int64_t buffer : stateBuffersToAccess) {
      if (buffer < 0 || buffer >= static_cast<int64_t>(plan.buffers.size()))
        return failure();
      auto stateType =
          dyn_cast<RankedTensorType>(plan.buffers[buffer].value.getType());
      if (!stateType || !stateType.hasStaticShape() ||
          stateType.getRank() != stateMap.getNumResults())
        return failure();
      task.compositionStateAccessBuffers.push_back(buffer);
      task.compositionStateAccessMaps.push_back(stateMap);
    }
    return success();
  };
  const int64_t resultBufferBase = body.getNumArguments();
  if (hasForwardTask) {
    ContractOp forwardAnchor =
        normalizedForwardResult
            ? terminalContracts.lookup(*normalizedForwardResult)
            : cast<ContractOp>(pullback->forward.consumer);
    SmallVector<int64_t> forwardOutputs;
    for (int64_t result : forwardResults)
      forwardOutputs.push_back(resultBufferBase + result);
    if (forwardBoundaryBuffer)
      forwardOutputs.push_back(*forwardBoundaryBuffer);
    forwardOutputs.append({stateBuffers[0], stateBuffers[1]});
    FailureOr<PlannedTask> forward =
        makeTask(forwardAnchor, forwardOutputs, ArrayRef<int64_t>{});
    if (failed(forward))
      return failure();
    forward->inputs.clear();
    for (auto [ordinal, argument] : llvm::enumerate(body.getArguments()))
      if (argument != outputCotangent)
        forward->inputs.push_back(ordinal);
    forward->compositionStateOutputs = {stateBuffers[0], stateBuffers[1]};
    if (failed(addStateAccesses(*forward,
                                forward->compositionStateOutputs)))
      return failure();
    plan.tasks.push_back(std::move(*forward));
  }
  if (hasResidualTask) {
    FoldOp fold = residualFolds.front();
    PlannedTask task;
    task.operation = fold;
    task.kind = MaterializationTaskKind::Fold;
    auto inputType = dyn_cast<RankedTensorType>(fold.getInputs().front().getType());
    if (!inputType || !inputType.hasStaticShape())
      return failure();
    task.domain.assign(inputType.getShape().begin(), inputType.getShape().end());
    task.reductionDimensions.assign(fold.getReductionDimensions().begin(),
                                    fold.getReductionDimensions().end());
    task.orderFree = fold.getOrderFree();
    task.dependencies.push_back(0);
    task.foldRegionSources = residualAlgebra;
    task.foldRegionPhases.push_back(fold);
    llvm::SmallDenseSet<int64_t> seenInputs;
    auto addBoundary = [&](Value value, MapOp map,
                           unsigned operand) -> LogicalResult {
      int64_t buffer = -1;
      if (value == pullback->forward.finalizer && forwardBoundaryBuffer)
        buffer = *forwardBoundaryBuffer;
      else if (auto argument = dyn_cast<BlockArgument>(value))
        buffer = argument.getArgNumber();
      else
        return failure();
      if (seenInputs.insert(buffer).second)
        task.inputs.push_back(buffer);
      task.foldRegionBoundaryUses.push_back(
          {buffer, map, operand,
           cast<AffineMapAttr>(map.getIndexingMaps()[operand]).getValue(), 0});
      return success();
    };
    for (Operation *operation : residualAlgebra) {
      auto map = dyn_cast<MapOp>(operation);
      if (!map)
        continue;
      for (auto [operand, value] : llvm::enumerate(map.getInputs()))
        if (!residualSet.contains(value.getDefiningOp()) &&
            failed(addBoundary(value, map, operand)))
          return failure();
    }
    if (task.foldRegionBoundaryUses.empty())
      return failure();
    for (int64_t result : residualResults) {
      task.outputs.push_back(resultBufferBase + result);
      task.foldRegionOutputs.push_back(
          regionYield.getValues()[result].getDefiningOp());
    }
    for (auto [value, buffer] : residualBoundaryBuffers) {
      task.outputs.push_back(buffer);
      task.foldRegionOutputs.push_back(value.getDefiningOp());
    }
    plan.tasks.push_back(std::move(task));
    plan.hasFoldRegion = true;
  }
  SmallVector<int64_t> queryOutputs;
  for (int64_t result : ownerResults)
    queryOutputs.push_back(resultBufferBase + result);
  queryOutputs.push_back(stateBuffers[2]);
  if (!hasForwardTask)
    queryOutputs.append({stateBuffers[0], stateBuffers[1]});
  FailureOr<PlannedTask> query =
      makeTask(terminalContracts.lookup(queryResult), queryOutputs,
               hasForwardTask ? ArrayRef<int64_t>({0}) : ArrayRef<int64_t>{});
  SmallVector<int64_t> keyValueOutputs;
  for (int64_t result : streamedResults)
    keyValueOutputs.push_back(resultBufferBase + result);
  SmallVector<int64_t> keyValueDependencies;
  if (hasForwardTask)
    keyValueDependencies.push_back(0);
  keyValueDependencies.push_back(queryTask);
  FailureOr<PlannedTask> keyValue =
      makeTask(terminalContracts.lookup(streamedAnchorResult),
               keyValueOutputs, keyValueDependencies);
  if (failed(query) || failed(keyValue))
    return failure();
  if (hasForwardTask) {
    for (int64_t result : forwardResults)
      query->inputs.push_back(resultBufferBase + result);
    query->inputs.push_back(stateBuffers[0]);
    query->inputs.push_back(stateBuffers[1]);
    query->compositionStateInputs = {stateBuffers[0], stateBuffers[1]};
  }
  for (auto [value, buffer] : residualBoundaryBuffers) {
    query->inputs.push_back(buffer);
    keyValue->inputs.push_back(buffer);
  }
  if (hasResidualTask) {
    query->dependencies.push_back(residualTask);
    keyValue->dependencies.push_back(residualTask);
  }
  llvm::sort(query->dependencies);
  llvm::sort(keyValue->dependencies);
  query->compositionStateOutputs = hasForwardTask
                                       ? SmallVector<int64_t>({stateBuffers[2]})
                                       : SmallVector<int64_t>(stateBuffers);
  SmallVector<int64_t> queryStateAccesses(query->compositionStateInputs);
  llvm::append_range(queryStateAccesses, query->compositionStateOutputs);
  if (failed(addStateAccesses(*query, queryStateAccesses)))
    return failure();
  keyValue->inputs.append({stateBuffers[0], stateBuffers[1], stateBuffers[2]});
  keyValue->compositionStateInputs = stateBuffers;
  if (failed(addStateAccesses(*keyValue,
                              keyValue->compositionStateInputs)))
    return failure();
  plan.tasks.push_back(std::move(*query));
  plan.tasks.push_back(std::move(*keyValue));

  SmallVector<Value> sharedScoreCotangents;
  ContractOp queryTerminal = terminalContracts.lookup(queryResult);
  ContractOp streamedTerminal = terminalContracts.lookup(streamedAnchorResult);
  for (Value queryInput : queryTerminal.getInputs())
    if (llvm::is_contained(streamedTerminal.getInputs(), queryInput))
      sharedScoreCotangents.push_back(queryInput);
  if (sharedScoreCotangents.size() != 1)
    return failure();
  auto scoreMapVjp = sharedScoreCotangents.front().getDefiningOp<MapOp>();
  auto outputProduct =
      pullback->outputDependentFold.getInputs().front().getDefiningOp<MapOp>();
  SmallVector<MapOp> postNormalizationVjps;
  for (Value input : differentialContract.getInputs()) {
    if (!valueDependsOnBlockArgument(input, outputCotangent))
      continue;
    if (auto map = input.getDefiningOp<MapOp>())
      postNormalizationVjps.push_back(map);
  }
  if (!scoreMapVjp || !outputProduct || postNormalizationVjps.size() != 1)
    return failure();

  SmallVector<MapOp> normalizedExpPullbackCandidates;
  for (MapOp map : body.getOps<MapOp>()) {
    if (map == scoreMapVjp ||
        !valueDependsOnOperation(scoreMapVjp.getResult(0), map) ||
        !valueDependsOnOperation(map.getResult(0), differentialContract) ||
        !valueDependsOnOperation(map.getResult(0),
                                 pullback->outputDependentFold) ||
        !valueDependsOnOperation(map.getResult(0), pullback->stateFold))
      continue;
    normalizedExpPullbackCandidates.push_back(map);
  }
  SmallVector<MapOp> maximalNormalizedExpPullbacks;
  for (MapOp candidate : normalizedExpPullbackCandidates) {
    const bool hasLaterCandidate =
        llvm::any_of(normalizedExpPullbackCandidates, [&](MapOp other) {
          return other != candidate &&
                 valueDependsOnOperation(other.getResult(0), candidate);
        });
    if (!hasLaterCandidate)
      maximalNormalizedExpPullbacks.push_back(candidate);
  }
  if (maximalNormalizedExpPullbacks.size() != 1)
    return failure();

  SmallVector<Operation *> terminalGradientContracts{
      queryTerminal.getOperation(), streamedTerminal.getOperation()};
  for (int64_t result : streamedResults)
    if (result != streamedAnchorResult)
      terminalGradientContracts.push_back(
          terminalContracts.lookup(result).getOperation());
  if (terminalGradientContracts.size() != 3)
    return failure();

  SmallVector<int64_t> terminalCotangentBuffers{
      resultBufferBase + queryResult,
      resultBufferBase + streamedAnchorResult};
  for (int64_t result : streamedResults)
    if (result != streamedAnchorResult)
      terminalCotangentBuffers.push_back(resultBufferBase + result);
  if (terminalCotangentBuffers.size() != terminalGradientContracts.size())
    return failure();
  std::optional<int64_t> forwardResultBuffer = forwardBoundaryBuffer;
  if (!forwardResultBuffer && normalizedForwardResult)
    forwardResultBuffer = resultBufferBase + *normalizedForwardResult;

  plan.normalizedExpContractRegions.push_back(pullback->forward);
  plan.normalizedExpContractPullbacks.push_back(
      {0,
       0,
       hasResidualTask ? std::optional<int64_t>(residualTask) : std::nullopt,
       queryTask,
       keyValueTask,
       stateBuffers[0],
       stateBuffers[1],
       stateBuffers[2],
       forwardResultBuffer,
       outputCotangent.getArgNumber(),
       std::move(terminalCotangentBuffers),
       pullback->forward.finalizer.getDefiningOp(),
       postNormalizationVjps.front().getOperation(),
       outputProduct.getOperation(),
       pullback->outputDependentFold.getOperation(),
       differentialContract.getOperation(),
       pullback->forward.normalizedWeight.getDefiningOp(),
       maximalNormalizedExpPullbacks.front().getOperation(),
       scoreMapVjp.getOperation(),
       std::move(terminalGradientContracts)});

  for (PlannedBuffer &buffer : plan.buffers)
    buffer.consumers.clear();
  for (auto [taskOrdinal, task] : llvm::enumerate(plan.tasks))
    for (int64_t input : task.inputs)
      if (plan.buffers[input].consumers.empty() ||
          plan.buffers[input].consumers.back() !=
              static_cast<int64_t>(taskOrdinal))
        plan.buffers[input].consumers.push_back(taskOrdinal);
  return plan;
}

} // namespace online_planner
