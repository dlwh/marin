// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

bool isExpansionDrivenByScalarFold(
    Operation *expansion,
    const llvm::SmallPtrSetImpl<Operation *> &componentOperations) {
  Operation *root = expansion->getOperand(0).getDefiningOp();
  if (!root)
    return false;
  SmallVector<Operation *> worklist{root};
  llvm::SmallPtrSet<Operation *, 8> visited;
  Operation *scalarFold = nullptr;
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val();
    if (!componentOperations.contains(producer))
      return false;
    if (!visited.insert(producer).second)
      continue;
    if (isSupportedStablehloReduce(producer)) {
      if (scalarFold && scalarFold != producer)
        return false;
      scalarFold = producer;
      continue;
    }
    if (!isClosedDirectStablehloMap(producer))
      return false;
    if (producer->getNumOperands() == 0) {
      if (producer->getName().getStringRef() !=
          stablehlo::ConstantOp::getOperationName())
        return false;
      continue;
    }
    for (Value operand : producer->getOperands()) {
      Operation *operandProducer = operand.getDefiningOp();
      if (!operandProducer || !componentOperations.contains(operandProducer))
        return false;
      worklist.push_back(operandProducer);
    }
  }
  return scalarFold != nullptr;
}

bool hasRankNExpansionWithoutClosedAnchor(const CandidateComponent &component) {
  llvm::SmallPtrSet<Operation *, 16> componentOperations(
      component.operations.begin(), component.operations.end());
  const bool hasScalarFold =
      llvm::any_of(component.operations, isSupportedStablehloReduce);
  return llvm::any_of(component.operations, [&](Operation *operation) {
    unsigned expansions = rankNExpandingSingletonBroadcasts(operation);
    if (expansions == 0)
      return false;
    if (expansions == 1)
      return !isExpansionDrivenByScalarFold(operation, componentOperations);
    return !hasScalarFold || !isAxisAlignedRankNExpansion(operation) ||
           !isExpansionDrivenByProjectedParameter(operation,
                                                  componentOperations);
  });
}
