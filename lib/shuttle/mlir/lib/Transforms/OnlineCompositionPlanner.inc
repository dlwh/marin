// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

namespace online_planner {

bool foldYields(FoldOp fold, StringRef operationName) {
  if (fold.getCombiner().empty())
    return false;
  Block &body = fold.getCombiner().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  Operation *combiner =
      yield && yield.getValues().size() == 1
          ? yield.getValues().front().getDefiningOp()
          : nullptr;
  return body.getNumArguments() == 2 && combiner &&
         combiner->getName().getStringRef() == operationName &&
         combiner->getNumOperands() == 2 &&
         combiner->getOperand(0) == body.getArgument(0) &&
         combiner->getOperand(1) == body.getArgument(1);
}

Operation *mapYieldedOperation(MapOp map, StringRef operationName) {
  if (map.getBody().empty())
    return nullptr;
  auto yield = dyn_cast<YieldOp>(map.getBody().front().getTerminator());
  Operation *operation =
      yield && yield.getValues().size() == 1
          ? yield.getValues().front().getDefiningOp()
          : nullptr;
  return operation && operation->getName().getStringRef() == operationName
             ? operation
             : nullptr;
}

std::optional<unsigned> unaryMapArgument(MapOp map, StringRef operationName) {
  Operation *operation = mapYieldedOperation(map, operationName);
  auto argument = operation && operation->getNumOperands() == 1
                      ? dyn_cast<BlockArgument>(operation->getOperand(0))
                      : BlockArgument{};
  if (!argument || argument.getOwner() != &map.getBody().front() ||
      argument.getArgNumber() >= map.getInputs().size())
    return std::nullopt;
  return argument.getArgNumber();
}

std::optional<std::pair<unsigned, unsigned>>
binaryMapArguments(MapOp map, StringRef operationName) {
  Operation *operation = mapYieldedOperation(map, operationName);
  if (!operation || operation->getNumOperands() != 2)
    return std::nullopt;
  auto lhs = dyn_cast<BlockArgument>(operation->getOperand(0));
  auto rhs = dyn_cast<BlockArgument>(operation->getOperand(1));
  if (!lhs || !rhs || lhs.getOwner() != &map.getBody().front() ||
      rhs.getOwner() != &map.getBody().front() ||
      lhs.getArgNumber() == rhs.getArgNumber() ||
      lhs.getArgNumber() >= map.getInputs().size() ||
      rhs.getArgNumber() >= map.getInputs().size())
    return std::nullopt;
  return std::pair(lhs.getArgNumber(), rhs.getArgNumber());
}

FailureOr<FoldOp> uniqueFoldAncestor(Value value, StringRef operationName) {
  FoldOp found;
  llvm::SmallDenseSet<Operation *> visited;
  SmallVector<Value> worklist{value};
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (auto fold = dyn_cast<FoldOp>(producer);
        fold && foldYields(fold, operationName)) {
      if (found && found != fold)
        return failure();
      found = fold;
      continue;
    }
    llvm::append_range(worklist, producer->getOperands());
  }
  if (!found)
    return failure();
  return found;
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

FailureOr<int64_t> logicalTensorBytes(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return failure();
  int64_t elementBytes = 0;
  if (tensor.getElementType().isF32() ||
      tensor.getElementType().isInteger(32))
    elementBytes = 4;
  else if (tensor.getElementType().isBF16())
    elementBytes = 2;
  else if (tensor.getElementType().isInteger(1))
    elementBytes = 1;
  else
    return failure();
  FailureOr<int64_t> elements = checkedShapeElements(tensor.getShape());
  if (failed(elements) ||
      *elements > std::numeric_limits<int64_t>::max() / elementBytes)
    return failure();
  return *elements * elementBytes;
}

LogicalResult accumulate(int64_t value, int64_t &total) {
  if (value < 0 || total > std::numeric_limits<int64_t>::max() - value)
    return failure();
  total += value;
  return success();
}

FailureOr<DerivedPlan>
derive(RegionOp region, const DerivedPlan &materialized) {
  if (region.getPolicy() != NumericalPolicy::Fast)
    return failure();
  Block &body = region.getBody().front();
  auto regionYield = dyn_cast<YieldOp>(body.getTerminator());
  if (!regionYield || regionYield.getValues().size() != 1)
    return failure();
  SmallVector<Operation *> algebra;
  ContractOp weightedContract;
  for (Operation &operation : body.without_terminator())
    algebra.push_back(&operation);
  for (Operation *operation : algebra) {
    auto contract = dyn_cast<ContractOp>(operation);
    if (!contract ||
        !valueDependsOnOperation(regionYield.getValues().front(), contract))
      continue;
    const bool feedsContract = llvm::any_of(algebra, [&](Operation *candidate) {
      auto other = dyn_cast<ContractOp>(candidate);
      return other && other != contract &&
             llvm::any_of(other.getInputs(), [&](Value input) {
               return valueDependsOnOperation(input, contract);
             });
    });
    if (!feedsContract) {
      if (weightedContract)
        return failure();
      weightedContract = contract;
    }
  }
  if (!weightedContract || weightedContract.getNumResults() != 1)
    return failure();

  FoldOp maximumFold;
  FoldOp additiveFold;
  MapOp exponentialMap;
  Value score;
  Value maximumExpansion;
  Value denominatorExpansion;
  std::optional<unsigned> normalizedOperand;
  for (auto [operand, input] : llvm::enumerate(weightedContract.getInputs())) {
    auto normalization = input.getDefiningOp<MapOp>();
    auto division = normalization
                        ? binaryMapArguments(normalization,
                                             arith::DivFOp::getOperationName())
                        : std::nullopt;
    if (!division)
      continue;
    auto exponential = normalization.getInputs()[division->first]
                           .getDefiningOp<MapOp>();
    auto exponentialArgument =
        exponential
            ? unaryMapArgument(exponential, math::ExpOp::getOperationName())
            : std::nullopt;
    FailureOr<FoldOp> additive = uniqueFoldAncestor(
        normalization.getInputs()[division->second],
        arith::AddFOp::getOperationName());
    if (!exponentialArgument || failed(additive) ||
        (*additive).getInputs().front() != exponential.getResult(0))
      continue;
    auto subtraction = exponential.getInputs()[*exponentialArgument]
                           .getDefiningOp<MapOp>();
    auto subtractionArguments =
        subtraction ? binaryMapArguments(subtraction,
                                         arith::SubFOp::getOperationName())
                    : std::nullopt;
    if (!subtractionArguments)
      continue;
    FailureOr<FoldOp> maximum = uniqueFoldAncestor(
        subtraction.getInputs()[subtractionArguments->second],
        arith::MaximumFOp::getOperationName());
    if (failed(maximum))
      continue;
    if (maximumFold)
      return failure();
    maximumFold = *maximum;
    additiveFold = *additive;
    exponentialMap = exponential;
    score = subtraction.getInputs()[subtractionArguments->first];
    maximumExpansion =
        subtraction.getInputs()[subtractionArguments->second];
    denominatorExpansion = normalization.getInputs()[division->second];
    normalizedOperand = operand;
  }
  if (!maximumFold || !additiveFold || !exponentialMap || !score ||
      !normalizedOperand ||
      !isClosedScalarPlanningFold(maximumFold) ||
      !isClosedScalarPlanningFold(additiveFold) ||
      maximumFold.getReductionDimensions() !=
          additiveFold.getReductionDimensions() ||
      maximumFold.getInputs().front().getType() !=
          additiveFold.getInputs().front().getType() ||
      maximumFold.getResult(0).getType() != additiveFold.getResult(0).getType())
    return failure();

  if (maximumFold.getInputs().front() != score)
    return failure();
  auto scoreType = dyn_cast<RankedTensorType>(score.getType());
  const int64_t reduction = maximumFold.getReductionDimensions().front();
  bool relationFeedsScore = false;
  bool contractFeedsScore = false;
  for (Operation *operation : algebra) {
    relationFeedsScore |=
        isa<RelationOp>(operation) && valueDependsOnOperation(score, operation);
    contractFeedsScore |= isa<ContractOp>(operation) &&
                          operation != weightedContract &&
                          valueDependsOnOperation(score, operation);
  }
  if (!relationFeedsScore || !contractFeedsScore ||
      !valueDependsOnOperation(maximumExpansion, maximumFold) ||
      additiveFold.getInputs().front() != exponentialMap.getResult(0))
    return failure();
  if (!scoreType || !normalizedOperand || !denominatorExpansion)
    return failure();
  FailureOr<AffineMap> expected = reductionProjection(scoreType, reduction);
  FailureOr<AffineMap> maximumCoordinates =
      coordinatesToAncestor(maximumExpansion, maximumFold.getResult(0));
  FailureOr<AffineMap> denominatorCoordinates =
      coordinatesToAncestor(denominatorExpansion, additiveFold.getResult(0));
  if (failed(expected) || failed(maximumCoordinates) ||
      failed(denominatorCoordinates) || *maximumCoordinates != *expected ||
      *denominatorCoordinates != *expected ||
      *normalizedOperand >= weightedContract.getIndexingMaps().size())
    return failure();
  auto weightMap = dyn_cast<AffineMapAttr>(
      weightedContract.getIndexingMaps()[*normalizedOperand]);
  if (!weightMap || reduction < 0 ||
      reduction >= weightMap.getValue().getNumResults())
    return failure();
  auto reductionCoordinate =
      dyn_cast<AffineDimExpr>(weightMap.getValue().getResult(reduction));
  if (!reductionCoordinate ||
      reductionCoordinate.getPosition() >=
          weightedContract.getIteratorKinds().size())
    return failure();
  auto iterator = dyn_cast<StringAttr>(weightedContract.getIteratorKinds()[
      reductionCoordinate.getPosition()]);
  if (!iterator || iterator.getValue() != "reduction")
    return failure();

  int64_t baselineBytes = 0;
  int64_t baselineTraffic = 0;
  for (Operation *operation : algebra) {
    for (Value result : operation->getResults()) {
      const bool liveOut = llvm::any_of(result.getUses(), [&](OpOperand &use) {
        return use.getOwner() == regionYield;
      });
      llvm::SmallDenseSet<Operation *> consumers;
      for (OpOperand &use : result.getUses())
        consumers.insert(use.getOwner());
      if (liveOut || consumers.empty())
        continue;
      FailureOr<int64_t> bytes = logicalTensorBytes(result.getType());
      const int64_t accesses = static_cast<int64_t>(consumers.size()) + 1;
      if (failed(bytes) || failed(accumulate(*bytes, baselineBytes)) ||
          (*bytes != 0 &&
           accesses > std::numeric_limits<int64_t>::max() / *bytes) ||
          failed(accumulate(*bytes * accesses, baselineTraffic)))
        return failure();
    }
  }
  FailureOr<int64_t> candidateBytes =
      logicalTensorBytes(maximumFold.getResult(0).getType());
  FailureOr<int64_t> scoreBytes = logicalTensorBytes(scoreType);
  if (failed(candidateBytes) || failed(scoreBytes) ||
      *candidateBytes > std::numeric_limits<int64_t>::max() / 2 ||
      *candidateBytes >= baselineBytes || 2 * *candidateBytes >= baselineTraffic ||
      *candidateBytes >= *scoreBytes || materialized.tasks.size() < 2 ||
      materialized.tasks.size() >
          static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    return failure();

  llvm::SmallDenseSet<Operation *> firstSources;
  SmallVector<Value> worklist(maximumFold->getOperands());
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val().getDefiningOp();
    if (!producer || producer->getParentOp() != region ||
        !firstSources.insert(producer).second)
      continue;
    llvm::append_range(worklist, producer->getOperands());
  }
  firstSources.insert(maximumFold);
  SmallVector<Operation *> firstOrdered;
  SmallVector<Operation *> secondOrdered;
  for (Operation *operation : algebra)
    (firstSources.contains(operation) ? firstOrdered : secondOrdered)
        .push_back(operation);
  if (firstOrdered.empty() || secondOrdered.empty() ||
      !llvm::is_contained(secondOrdered, weightedContract))
    return failure();
  auto usedArguments = [&](ArrayRef<Operation *> sources) {
    llvm::SmallDenseSet<unsigned> used;
    for (Operation *operation : sources)
      for (Value operand : operation->getOperands())
        if (auto argument = dyn_cast<BlockArgument>(operand);
            argument && argument.getOwner() == &body)
          used.insert(argument.getArgNumber());
    return used;
  };
  llvm::SmallDenseSet<unsigned> firstArguments = usedArguments(firstOrdered);
  llvm::SmallDenseSet<unsigned> secondArguments = usedArguments(algebra);

  DerivedPlan plan;
  plan.hasContract = true;
  plan.hasRankNFold = scoreType.getRank() > 2;
  plan.hasContractFoldComposition = true;
  plan.hasLogicalRelation = true;
  for (BlockArgument argument : body.getArguments()) {
    SmallVector<int64_t> consumers;
    if (firstArguments.contains(argument.getArgNumber()))
      consumers.push_back(0);
    if (secondArguments.contains(argument.getArgNumber()))
      consumers.push_back(1);
    plan.buffers.push_back({argument, std::nullopt, consumers, false});
  }
  const int64_t maximumBuffer = plan.buffers.size();
  plan.buffers.push_back({maximumFold.getResult(0), 0, {1}, false});
  const int64_t outputBuffer = plan.buffers.size();
  plan.buffers.push_back(
      {regionYield.getValues().front(), 1, {}, true});

  PlannedTask maximumTask;
  maximumTask.operation = maximumFold;
  maximumTask.kind = MaterializationTaskKind::Fold;
  maximumTask.domain.assign(scoreType.getShape().begin(),
                            scoreType.getShape().end());
  maximumTask.reductionDimensions.push_back(reduction);
  maximumTask.orderFree = true;
  for (unsigned input = 0; input < body.getNumArguments(); ++input)
    if (firstArguments.contains(input))
      maximumTask.inputs.push_back(input);
  maximumTask.outputs.push_back(maximumBuffer);
  maximumTask.compositionKind =
      "online_normalized_exponential_weighted_contract";
  maximumTask.compositionSources = std::move(firstOrdered);

  PlannedTask contractionTask;
  contractionTask.operation = weightedContract;
  contractionTask.kind = MaterializationTaskKind::Contract;
  FailureOr<SmallVector<int64_t>> contractDomain =
      contractDomainShape(weightedContract);
  if (failed(contractDomain))
    return failure();
  contractionTask.domain = std::move(*contractDomain);
  for (auto [position, kind] :
       llvm::enumerate(weightedContract.getIteratorKinds()))
    if (cast<StringAttr>(kind).getValue() == "reduction")
      contractionTask.reductionDimensions.push_back(position);
  for (unsigned input = 0; input < body.getNumArguments(); ++input)
    if (secondArguments.contains(input))
      contractionTask.inputs.push_back(input);
  contractionTask.inputs.push_back(maximumBuffer);
  contractionTask.outputs.push_back(outputBuffer);
  contractionTask.dependencies.push_back(0);
  contractionTask.compositionKind = maximumTask.compositionKind;
  contractionTask.compositionSources = std::move(secondOrdered);

  PlannedTask::CompositionCost cost{
      static_cast<int64_t>(materialized.tasks.size()),
      2,
      baselineBytes,
      *candidateBytes,
      baselineTraffic,
      2 * *candidateBytes,
      2};
  maximumTask.compositionCost = cost;
  contractionTask.compositionCost = cost;
  plan.tasks.push_back(std::move(maximumTask));
  plan.tasks.push_back(std::move(contractionTask));
  return plan;
}

} // namespace online_planner
