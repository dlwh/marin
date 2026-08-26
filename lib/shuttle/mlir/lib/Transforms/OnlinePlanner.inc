// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Included by Passes.cc inside the Shuttle transforms implementation namespace.

bool hasExactNegativeInfinityInitializer(FoldOp fold) {
  if (fold.getInitializers().size() != 1)
    return false;
  auto initializer = fold.getInitializers()[0].getDefiningOp<MapOp>();
  if (!initializer || !initializer.getInputs().empty() ||
      initializer.getBody().empty())
    return false;
  Block &body = initializer.getBody().front();
  auto constant = body.getOperations().size() == 2
                      ? dyn_cast<arith::ConstantOp>(body.front())
                      : arith::ConstantOp{};
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  auto value = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
  return value && value.getType().isF32() && value.getValue().isInfinity() &&
         value.getValue().isNegative() && yield &&
         yield.getValues().size() == 1 &&
         yield.getValues().front() == constant.getResult();
}

bool isClosedScalarPlanningFold(FoldOp fold) {
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
      input.getRank() < 1 || input.getRank() > 8 ||
      !result.hasStaticShape() || !input.getElementType().isF32() ||
      initializer.getRank() != 0 || !initializer.getElementType().isF32() ||
      !result.getElementType().isF32() || !fold.getOrderFree() ||
      fold.getAccumulatorTypes().size() != 1 ||
      cast<TypeAttr>(fold.getAccumulatorTypes()[0]).getValue() !=
          Float32Type::get(fold.getContext()) ||
      llvm::any_of(input.getShape(),
                   [](int64_t extent) { return extent <= 0; }) ||
      fold.getReductionDimensions().size() != 1)
    return false;
  const int64_t reduction = fold.getReductionDimensions().front();
  if (reduction < 0 || reduction >= input.getRank())
    return false;
  SmallVector<int64_t> expectedResult(input.getShape());
  expectedResult.erase(expectedResult.begin() + reduction);
  if (result.getShape() != ArrayRef<int64_t>(expectedResult))
    return false;
  Block &body = fold.getCombiner().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2)
    return false;
  Operation &combiner = body.front();
  auto yield = dyn_cast<YieldOp>(body.back());
  const bool additive = isa<arith::AddFOp>(combiner);
  const bool maximum = isa<arith::MaximumFOp>(combiner);
  auto fastMath =
      combiner.getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
  if ((!additive && !maximum) || !yield || combiner.getNumOperands() != 2 ||
      combiner.getOperand(0) != body.getArgument(0) ||
      combiner.getOperand(1) != body.getArgument(1) ||
      combiner.getNumResults() != 1 || yield.getValues().size() != 1 ||
      yield.getValues().front() != combiner.getResult(0) ||
      (fastMath && fastMath.getValue() != arith::FastMathFlags::none))
    return false;
  return additive ? hasExactPositiveZeroInitializer(fold)
                  : hasExactNegativeInfinityInitializer(fold);
}
