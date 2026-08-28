// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

struct OnlineEmissionMatch {
  ContractOp weightedContract;
  ContractOp scoreContract;
  FoldOp maximumFold;
  FoldOp additiveFold;
  Value score;
  unsigned normalizedOperand;
  SmallVector<RelationOp> relations;
};

FailureOr<OnlineEmissionMatch> matchOnlineEmission(RegionOp region) {
  Block &body = region.getBody().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return failure();
  SmallVector<Operation *> algebra;
  for (Operation &operation : body.without_terminator())
    algebra.push_back(&operation);
  ContractOp weighted;
  for (Operation *operation : algebra) {
    auto contract = dyn_cast<ContractOp>(operation);
    if (!contract ||
        !valueDependsOnOperation(yield.getValues().front(), contract))
      continue;
    const bool feedsContract = llvm::any_of(algebra, [&](Operation *candidate) {
      auto other = dyn_cast<ContractOp>(candidate);
      return other && other != contract &&
             llvm::any_of(other.getInputs(), [&](Value input) {
               return valueDependsOnOperation(input, contract);
             });
    });
    if (!feedsContract) {
      if (weighted)
        return failure();
      weighted = contract;
    }
  }
  if (!weighted || weighted.getNumResults() != 1)
    return failure();
  FoldOp maximum;
  FoldOp additive;
  Value score;
  std::optional<unsigned> normalizedOperand;
  for (auto [operand, input] : llvm::enumerate(weighted.getInputs())) {
    auto normalization = input.getDefiningOp<MapOp>();
    auto division = normalization
                        ? online_planner::binaryMapArguments(
                              normalization, arith::DivFOp::getOperationName())
                        : std::nullopt;
    if (!division)
      continue;
    auto exponential =
        normalization.getInputs()[division->first].getDefiningOp<MapOp>();
    auto exponentialArgument =
        exponential ? online_planner::unaryMapArgument(
                          exponential, math::ExpOp::getOperationName())
                    : std::nullopt;
    FailureOr<FoldOp> sum = online_planner::uniqueFoldAncestor(
        normalization.getInputs()[division->second],
        arith::AddFOp::getOperationName());
    if (!exponentialArgument || failed(sum) ||
        (*sum).getInputs().front() != exponential.getResult(0))
      continue;
    auto subtraction =
        exponential.getInputs()[*exponentialArgument].getDefiningOp<MapOp>();
    auto subtractionArguments =
        subtraction ? online_planner::binaryMapArguments(
                          subtraction, arith::SubFOp::getOperationName())
                    : std::nullopt;
    if (!subtractionArguments)
      continue;
    FailureOr<FoldOp> max = online_planner::uniqueFoldAncestor(
        subtraction.getInputs()[subtractionArguments->second],
        arith::MaximumFOp::getOperationName());
    if (failed(max) || maximum)
      return failure();
    maximum = *max;
    additive = *sum;
    score = subtraction.getInputs()[subtractionArguments->first];
    normalizedOperand = operand;
  }
  if (!maximum || !additive || !score || !normalizedOperand ||
      maximum.getInputs().front() != score)
    return failure();
  ContractOp scoreContract;
  SmallVector<RelationOp> relations;
  for (Operation *operation : algebra) {
    if (auto relation = dyn_cast<RelationOp>(operation);
        relation && valueDependsOnOperation(score, relation))
      relations.push_back(relation);
    if (auto contract = dyn_cast<ContractOp>(operation);
        contract && contract != weighted &&
        valueDependsOnOperation(score, contract)) {
      if (scoreContract)
        return failure();
      scoreContract = contract;
    }
  }
  if (!scoreContract || relations.empty())
    return failure();
  return OnlineEmissionMatch{
      weighted,           scoreContract,       maximum, additive, score,
      *normalizedOperand, std::move(relations)};
}

enum class OnlineScalarKind { F32, U32, Predicate };

struct OnlineScalarValue {
  OnlineScalarKind kind;
  int64_t reg;
};

struct OnlineEvalContext {
  PtxEmitter &ptx;
  RegionOp region;
  MaterializationTaskOp task;
  ScheduleTaskOp schedule;
  OnlineEmissionMatch &match;
  llvm::SmallDenseMap<unsigned, int64_t> argumentParameters;
  std::optional<int64_t> maximumParameter;
  DenseMap<Value, OnlineScalarValue> relationValues;
  DenseMap<Value, OnlineScalarValue> valueOverrides;
  DenseMap<Value, std::string> relationRoles;
  std::optional<int64_t> cooperativeLane;
  int64_t label = 0;

  OnlineEvalContext(PtxEmitter &ptx, RegionOp region,
                    MaterializationTaskOp task, ScheduleTaskOp schedule,
                    OnlineEmissionMatch &match)
      : ptx(ptx), region(region), task(task), schedule(schedule), match(match) {
  }
};

FailureOr<SmallVector<AffineMap>>
onlineLogicalCoordinateMaps(ScheduleTaskOp task);

void emitOnlineRole(PtxEmitter &ptx, StringRef role) {
  ptx.stream << "// shuttle.online.role " << role << "\n";
}

void emitOnlineSource(PtxEmitter &ptx, RelationOp relation) {
  SourceRefAttr source = relation.getSource();
  ptx.stream << "// shuttle.online.source_ref " << source.getFunctionOrdinal()
             << ':' << source.getBlockOrdinal() << ':'
             << source.getOperationOrdinal() << ':' << source.getResultOrdinal()
             << "\n";
}

void emitOnlineSource(OnlineEvalContext &context, RelationOp relation) {
  emitOnlineSource(context.ptx, relation);
  auto role = context.relationRoles.find(relation.getResult(0));
  if (role != context.relationRoles.end())
    emitOnlineRole(context.ptx, role->second);
}

FailureOr<int64_t> emitOnlineAffineU32(PtxEmitter &ptx, AffineExpr expression,
                                       ArrayRef<int64_t> coordinates) {
  if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
    if (dimension.getPosition() >= coordinates.size())
      return failure();
    return coordinates[dimension.getPosition()];
  }
  if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
    if (constant.getValue() < 0 || constant.getValue() > UINT32_MAX)
      return failure();
    int64_t result = ptx.nextU32();
    ptx.stream << "  mov.u32 %r" << result << ", " << constant.getValue()
               << ";\n";
    return result;
  }
  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return failure();
  FailureOr<int64_t> lhs =
      emitOnlineAffineU32(ptx, binary.getLHS(), coordinates);
  FailureOr<int64_t> rhs =
      emitOnlineAffineU32(ptx, binary.getRHS(), coordinates);
  auto rhsConstant = dyn_cast<AffineConstantExpr>(binary.getRHS());
  if (failed(lhs) || failed(rhs))
    return failure();
  int64_t result = ptx.nextU32();
  switch (binary.getKind()) {
  case AffineExprKind::Add:
    ptx.stream << "  add.u32 %r" << result << ", %r" << *lhs << ", %r" << *rhs
               << ";\n";
    break;
  case AffineExprKind::Mul:
    ptx.stream << "  mul.lo.u32 %r" << result << ", %r" << *lhs << ", %r"
               << *rhs << ";\n";
    break;
  case AffineExprKind::FloorDiv:
    if (!rhsConstant || rhsConstant.getValue() <= 0)
      return failure();
    ptx.stream << "  div.u32 %r" << result << ", %r" << *lhs << ", "
               << rhsConstant.getValue() << ";\n";
    break;
  case AffineExprKind::Mod:
    if (!rhsConstant || rhsConstant.getValue() <= 0)
      return failure();
    ptx.stream << "  rem.u32 %r" << result << ", %r" << *lhs << ", "
               << rhsConstant.getValue() << ";\n";
    break;
  default:
    return failure();
  }
  return result;
}

FailureOr<SmallVector<int64_t>>
emitOnlineMappedU32(PtxEmitter &ptx, AffineMap map,
                    ArrayRef<int64_t> coordinates) {
  if (map.getNumDims() != coordinates.size() || map.getNumSymbols() != 0)
    return failure();
  SmallVector<int64_t> result;
  for (AffineExpr expression : map.getResults()) {
    FailureOr<int64_t> coordinate =
        emitOnlineAffineU32(ptx, expression, coordinates);
    if (failed(coordinate))
      return failure();
    result.push_back(*coordinate);
  }
  return result;
}

FailureOr<SmallVector<int64_t>>
emitOnlineLogicalCoordinates(PtxEmitter &ptx, ScheduleTaskOp task) {
  FailureOr<SmallVector<AffineMap>> maps = onlineLogicalCoordinateMaps(task);
  if (failed(maps))
    return failure();
  int64_t block = ptx.nextU32();
  int64_t thread = ptx.nextU32();
  int64_t zero = ptx.nextU32();
  ptx.stream << "  mov.u32 %r" << block << ", %ctaid.x;\n"
             << "  mov.u32 %r" << thread << ", %tid.x;\n"
             << "  mov.u32 %r" << zero << ", 0;\n";
  SmallVector<int64_t> physical{block, zero, zero, thread, zero, zero, zero};
  SmallVector<int64_t> coordinates;
  for (AffineMap map : *maps) {
    FailureOr<int64_t> coordinate =
        emitOnlineAffineU32(ptx, map.getResult(0), physical);
    if (failed(coordinate))
      return failure();
    coordinates.push_back(*coordinate);
  }
  return coordinates;
}

FailureOr<int64_t> emitOnlineTensorPointer(PtxEmitter &ptx, int64_t parameter,
                                           RankedTensorType type,
                                           ArrayRef<int64_t> coordinates) {
  if (coordinates.size() != static_cast<size_t>(type.getRank()))
    return failure();
  int64_t linear = ptx.nextU64();
  ptx.stream << "  mov.u64 %rd" << linear << ", 0;\n";
  for (auto [axis, coordinate] : llvm::enumerate(coordinates)) {
    int64_t wide = ptx.nextU64();
    ptx.stream << "  cvt.u64.u32 %rd" << wide << ", %r" << coordinate
               << ";\n  mad.lo.u64 %rd" << linear << ", %rd" << linear << ", "
               << type.getDimSize(axis) << ", %rd" << wide << ";\n";
  }
  int64_t pointer = ptx.nextU64();
  const int64_t bytes = type.getElementType().isBF16() ? 2 : 4;
  ptx.stream << "  ld.param.u64 %rd" << pointer << ", [p" << parameter
             << "];\n  mad.lo.u64 %rd" << pointer << ", %rd" << linear << ", "
             << bytes << ", %rd" << pointer << ";\n";
  return pointer;
}

FailureOr<int64_t> emitOnlineTensorLoad(PtxEmitter &ptx, int64_t parameter,
                                        RankedTensorType type,
                                        ArrayRef<int64_t> coordinates) {
  FailureOr<int64_t> pointer =
      emitOnlineTensorPointer(ptx, parameter, type, coordinates);
  if (failed(pointer))
    return failure();
  int64_t value = ptx.nextF32();
  if (type.getElementType().isBF16()) {
    int64_t bits = ptx.nextBF16();
    ptx.stream << "  ld.global.u16 %h" << bits << ", [%rd" << *pointer
               << "];\n  cvt.f32.bf16 %f" << value << ", %h" << bits << ";\n";
  } else if (type.getElementType().isF32()) {
    ptx.stream << "  ld.global.f32 %f" << value << ", [%rd" << *pointer
               << "];\n";
  } else {
    return failure();
  }
  return value;
}

FailureOr<OnlineScalarValue> emitOnlineValue(OnlineEvalContext &context,
                                             Value value,
                                             ArrayRef<int64_t> coordinates);

struct OnlineScoreGeometry {
  SmallVector<int64_t> domainShape;
  AffineMap outputMap;
  AffineMap queryMap;
  AffineMap keyMap;
  unsigned queryOperand;
  unsigned keyOperand;
  int64_t reductionAxis;
};

bool onlineDependsOn(Value value, Value ancestor) {
  if (value == ancestor)
    return true;
  Operation *operation = value.getDefiningOp();
  return operation && llvm::any_of(operation->getOperands(), [&](Value input) {
           return onlineDependsOn(input, ancestor);
         });
}

FailureOr<AffineMap> onlineAncestorCoordinateMap(Value value, Value ancestor) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return failure();
  if (value == ancestor)
    return AffineMap::getMultiDimIdentityMap(type.getRank(),
                                             value.getContext());
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return failure();
  std::optional<unsigned> selected;
  for (auto [operand, input] : llvm::enumerate(operation->getOperands())) {
    if (!onlineDependsOn(input, ancestor))
      continue;
    if (selected)
      return failure();
    selected = operand;
  }
  if (!selected)
    return failure();
  AffineMap inputMap;
  if (auto map = dyn_cast<MapOp>(operation)) {
    auto indexing = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[*selected]);
    if (!indexing)
      return failure();
    inputMap = indexing.getValue();
  } else if (auto relation = dyn_cast<RelationOp>(operation)) {
    if (relation.getSemantics() != RelationSemantics::Select || *selected != 1)
      return failure();
    auto inputType =
        dyn_cast<RankedTensorType>(relation.getInputs()[1].getType());
    if (!inputType || inputType.getShape() != type.getShape())
      return failure();
    inputMap =
        AffineMap::getMultiDimIdentityMap(type.getRank(), value.getContext());
  } else {
    return failure();
  }
  FailureOr<AffineMap> parent =
      onlineAncestorCoordinateMap(operation->getOperand(*selected), ancestor);
  if (failed(parent) || parent->getNumDims() != inputMap.getNumResults())
    return failure();
  return parent->compose(inputMap);
}

FailureOr<OnlineScoreGeometry>
onlineScoreGeometry(OnlineEmissionMatch &match, AffineMap normalizedMap,
                    int64_t weightedReductionAxis) {
  ContractOp contract = match.scoreContract;
  FailureOr<SmallVector<int64_t>> shape = contractDomainShape(contract);
  if (failed(shape) || contract.getInputs().size() != 2 ||
      contract.getIndexingMaps().size() != 3)
    return failure();
  auto output = dyn_cast<AffineMapAttr>(contract.getIndexingMaps()[2]);
  auto lhs = dyn_cast<AffineMapAttr>(contract.getIndexingMaps()[0]);
  auto rhs = dyn_cast<AffineMapAttr>(contract.getIndexingMaps()[1]);
  if (!output || !lhs || !rhs)
    return failure();
  std::optional<unsigned> keyResult;
  for (auto [position, expression] :
       llvm::enumerate(normalizedMap.getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (dimension && dimension.getPosition() ==
                         static_cast<unsigned>(weightedReductionAxis)) {
      if (keyResult)
        return failure();
      keyResult = position;
    }
  }
  if (!keyResult || *keyResult >= output.getValue().getNumResults())
    return failure();
  auto keyDomain =
      dyn_cast<AffineDimExpr>(output.getValue().getResult(*keyResult));
  if (!keyDomain)
    return failure();
  auto containsKey = [&](AffineMap map) {
    return llvm::any_of(map.getResults(), [&](AffineExpr expression) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      return dimension && dimension.getPosition() == keyDomain.getPosition();
    });
  };
  const bool lhsKey = containsKey(lhs.getValue());
  const bool rhsKey = containsKey(rhs.getValue());
  if (lhsKey == rhsKey)
    return failure();
  std::optional<int64_t> reduction;
  for (auto [axis, iterator] : llvm::enumerate(contract.getIteratorKinds())) {
    if (cast<StringAttr>(iterator).getValue() != "reduction")
      continue;
    if (reduction)
      return failure();
    reduction = axis;
  }
  if (!reduction)
    return failure();
  return OnlineScoreGeometry{std::move(*shape),
                             output.getValue(),
                             lhsKey ? rhs.getValue() : lhs.getValue(),
                             lhsKey ? lhs.getValue() : rhs.getValue(),
                             lhsKey ? 1u : 0u,
                             lhsKey ? 0u : 1u,
                             *reduction};
}

FailureOr<SmallVector<int64_t>> onlineScoreDomainCoordinates(
    PtxEmitter &ptx, const OnlineScoreGeometry &geometry,
    ArrayRef<int64_t> scoreCoordinates, int64_t reductionCoordinate) {
  if (scoreCoordinates.size() != geometry.outputMap.getNumResults())
    return failure();
  SmallVector<int64_t> domain(geometry.domainShape.size(), -1);
  for (auto [result, expression] :
       llvm::enumerate(geometry.outputMap.getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() >= domain.size() ||
        domain[dimension.getPosition()] >= 0)
      return failure();
    domain[dimension.getPosition()] = scoreCoordinates[result];
  }
  domain[geometry.reductionAxis] = reductionCoordinate;
  for (int64_t &coordinate : domain) {
    if (coordinate >= 0)
      continue;
    coordinate = ptx.nextU32();
    ptx.stream << "  mov.u32 %r" << coordinate << ", 0;\n";
  }
  return domain;
}

int64_t emitOnlineSharedBf16Load(PtxEmitter &ptx, int64_t sharedBase,
                                 int64_t element) {
  int64_t wide = ptx.nextU64();
  int64_t pointer = ptx.nextU64();
  int64_t bits = ptx.nextBF16();
  int64_t value = ptx.nextF32();
  ptx.stream << "  cvt.u64.u32 %rd" << wide << ", %r" << element
             << ";\n  mul.lo.u64 %rd" << wide << ", %rd" << wide
             << ", 2;\n  add.u64 %rd" << pointer << ", %rd" << sharedBase
             << ", %rd" << wide << ";\n  ld.shared.u16 %h" << bits << ", [%rd"
             << pointer << "];\n  cvt.f32.bf16 %f" << value << ", %h" << bits
             << ";\n";
  return value;
}

int64_t emitOnlineSharedBf16Bits(PtxEmitter &ptx, int64_t sharedBase,
                                 int64_t element) {
  int64_t wide = ptx.nextU64();
  int64_t pointer = ptx.nextU64();
  int64_t bits = ptx.nextBF16();
  ptx.stream << "  cvt.u64.u32 %rd" << wide << ", %r" << element
             << ";\n  mul.lo.u64 %rd" << wide << ", %rd" << wide
             << ", 2;\n  add.u64 %rd" << pointer << ", %rd" << sharedBase
             << ", %rd" << wide << ";\n  ld.shared.u16 %h" << bits << ", [%rd"
             << pointer << "];\n";
  return bits;
}

int64_t emitOnlinePackBf16(PtxEmitter &ptx, int64_t low, int64_t high) {
  int64_t packed = ptx.nextU32();
  ptx.stream << "  mov.b32 %r" << packed << ", {%h" << low << ", %h" << high
             << "};\n";
  return packed;
}

void emitOnlineSharedBf16Store(PtxEmitter &ptx, int64_t sharedBase,
                               int64_t element, int64_t value) {
  int64_t wide = ptx.nextU64();
  int64_t pointer = ptx.nextU64();
  int64_t bits = ptx.nextBF16();
  ptx.stream << "  cvt.u64.u32 %rd" << wide << ", %r" << element
             << ";\n  mul.lo.u64 %rd" << wide << ", %rd" << wide
             << ", 2;\n  add.u64 %rd" << pointer << ", %rd" << sharedBase
             << ", %rd" << wide << ";\n  cvt.rn.bf16.f32 %h" << bits << ", %f"
             << value << ";\n  st.shared.u16 [%rd" << pointer << "], %h" << bits
             << ";\n";
}

FailureOr<SmallVector<int64_t>>
emitOnlineReshapeCoordinates(PtxEmitter &ptx, RankedTensorType input,
                             RankedTensorType output,
                             ArrayRef<int64_t> outputCoordinates) {
  if (outputCoordinates.size() != static_cast<size_t>(output.getRank()))
    return failure();
  SmallVector<int64_t> inputNonUnit;
  SmallVector<int64_t> outputNonUnit;
  for (auto [axis, extent] : llvm::enumerate(input.getShape()))
    if (extent != 1)
      inputNonUnit.push_back(axis);
  for (auto [axis, extent] : llvm::enumerate(output.getShape()))
    if (extent != 1)
      outputNonUnit.push_back(axis);
  bool unitDimensionReshape = inputNonUnit.size() == outputNonUnit.size();
  for (auto [inputAxis, outputAxis] : llvm::zip(inputNonUnit, outputNonUnit))
    unitDimensionReshape &=
        input.getDimSize(inputAxis) == output.getDimSize(outputAxis);
  if (unitDimensionReshape) {
    SmallVector<int64_t> result;
    size_t nonUnit = 0;
    for (int64_t axis = 0; axis < input.getRank(); ++axis) {
      if (input.getDimSize(axis) != 1) {
        result.push_back(outputCoordinates[outputNonUnit[nonUnit++]]);
        continue;
      }
      int64_t zero = ptx.nextU32();
      ptx.stream << "  mov.u32 %r" << zero << ", 0;\n";
      result.push_back(zero);
    }
    return result;
  }
  int64_t linear = ptx.nextU32();
  ptx.stream << "  mov.u32 %r" << linear << ", 0;\n";
  for (auto [axis, coordinate] : llvm::enumerate(outputCoordinates)) {
    int64_t next = ptx.nextU32();
    ptx.stream << "  mul.lo.u32 %r" << next << ", %r" << linear << ", "
               << output.getDimSize(axis) << ";\n  add.u32 %r" << next << ", %r"
               << next << ", %r" << coordinate << ";\n";
    linear = next;
  }
  SmallVector<int64_t> result;
  int64_t stride = 1;
  SmallVector<int64_t> strides(input.getRank());
  for (int64_t axis = input.getRank(); axis > 0; --axis) {
    strides[axis - 1] = stride;
    stride *= input.getDimSize(axis - 1);
  }
  for (auto [axis, extent] : llvm::enumerate(input.getShape())) {
    int64_t divided = ptx.nextU32();
    int64_t coordinate = ptx.nextU32();
    ptx.stream << "  div.u32 %r" << divided << ", %r" << linear << ", "
               << strides[axis] << ";\n  rem.u32 %r" << coordinate << ", %r"
               << divided << ", " << extent << ";\n";
    result.push_back(coordinate);
  }
  return result;
}

FailureOr<OnlineScalarValue>
emitOnlineContractValue(OnlineEvalContext &context, ContractOp contract,
                        ArrayRef<int64_t> resultCoordinates) {
  FailureOr<SmallVector<int64_t>> domainShape = contractDomainShape(contract);
  auto resultType = dyn_cast<RankedTensorType>(contract.getResult(0).getType());
  if (failed(domainShape) || contract.getIndexingMaps().size() != 3 ||
      !resultType ||
      resultCoordinates.size() != static_cast<size_t>(resultType.getRank()))
    return failure();
  auto maps = contract.getIndexingMaps();
  auto outputMap = dyn_cast<AffineMapAttr>(maps[maps.size() - 1]);
  if (!outputMap ||
      outputMap.getValue().getNumResults() != resultCoordinates.size())
    return failure();
  SmallVector<int64_t> domainCoordinates(domainShape->size(), -1);
  for (auto [result, expression] :
       llvm::enumerate(outputMap.getValue().getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() >= domainCoordinates.size() ||
        domainCoordinates[dimension.getPosition()] >= 0)
      return failure();
    domainCoordinates[dimension.getPosition()] = resultCoordinates[result];
  }
  std::optional<int64_t> reductionAxis;
  for (auto [axis, iterator] : llvm::enumerate(contract.getIteratorKinds())) {
    if (cast<StringAttr>(iterator).getValue() != "reduction")
      continue;
    if (reductionAxis)
      return failure();
    reductionAxis = axis;
  }
  if (!reductionAxis || domainCoordinates[*reductionAxis] >= 0)
    return failure();
  const bool cooperativeScore =
      contract == context.match.scoreContract && context.cooperativeLane;
  int64_t reduction = context.ptx.nextU32();
  domainCoordinates[*reductionAxis] = reduction;
  int64_t accumulator = context.ptx.nextF32();
  int64_t predicate = context.ptx.nextPred();
  const int64_t label = context.label++;
  context.ptx.stream << "  mov.f32 %f" << accumulator
                     << ", 0f00000000;\n  mov.u32 %r" << reduction << ", ";
  if (cooperativeScore)
    context.ptx.stream << "%r" << *context.cooperativeLane;
  else
    context.ptx.stream << "0";
  context.ptx.stream << ";\n$online_contract_" << label
                     << "_loop:\n  setp.ge.u32 %p" << predicate << ", %r"
                     << reduction << ", " << (*domainShape)[*reductionAxis]
                     << ";\n  @%p" << predicate << " bra $online_contract_"
                     << label << "_done;\n";
  SmallVector<OnlineScalarValue> operands;
  for (auto [operand, input] : llvm::enumerate(contract.getInputs())) {
    auto map = dyn_cast<AffineMapAttr>(contract.getIndexingMaps()[operand]);
    if (!map)
      return failure();
    FailureOr<SmallVector<int64_t>> inputCoordinates =
        emitOnlineMappedU32(context.ptx, map.getValue(), domainCoordinates);
    if (failed(inputCoordinates))
      return failure();
    FailureOr<OnlineScalarValue> scalar =
        emitOnlineValue(context, input, *inputCoordinates);
    if (failed(scalar) || scalar->kind != OnlineScalarKind::F32)
      return failure();
    operands.push_back(*scalar);
  }
  if (operands.size() != 2)
    return failure();
  int64_t next = context.ptx.nextF32();
  context.ptx.stream << "  fma.rn.f32 %f" << next << ", %f" << operands[0].reg
                     << ", %f" << operands[1].reg << ", %f" << accumulator
                     << ";\n  mov.f32 %f" << accumulator << ", %f" << next
                     << ";\n  add.u32 %r" << reduction << ", %r" << reduction
                     << ", " << (cooperativeScore ? 32 : 1)
                     << ";\n  bra $online_contract_" << label
                     << "_loop;\n$online_contract_" << label << "_done:\n";
  if (cooperativeScore) {
    for (int offset : {16, 8, 4, 2, 1}) {
      int64_t shuffled = context.ptx.nextF32();
      int64_t reduced = context.ptx.nextF32();
      context.ptx.stream << "  shfl.sync.down.b32 %f" << shuffled << ", %f"
                         << accumulator << ", " << offset
                         << ", 0x1f, 0xffffffff;\n  add.rn.f32 %f" << reduced
                         << ", %f" << accumulator << ", %f" << shuffled
                         << ";\n";
      accumulator = reduced;
    }
    int64_t broadcast = context.ptx.nextF32();
    context.ptx.stream << "  shfl.sync.idx.b32 %f" << broadcast << ", %f"
                       << accumulator << ", 0, 0x1f, 0xffffffff;\n";
    accumulator = broadcast;
  }
  return OnlineScalarValue{OnlineScalarKind::F32, accumulator};
}

FailureOr<OnlineScalarValue>
emitOnlineRelationValue(OnlineEvalContext &context, RelationOp relation,
                        ArrayRef<int64_t> coordinates) {
  if (auto found = context.relationValues.find(relation.getResult(0));
      found != context.relationValues.end())
    return found->second;
  auto resultType = dyn_cast<RankedTensorType>(relation.getResult(0).getType());
  if (!resultType || resultType.getRank() != coordinates.size())
    return failure();
  auto remember = [&](OnlineScalarValue value) {
    context.relationValues[relation.getResult(0)] = value;
    return FailureOr<OnlineScalarValue>(value);
  };
  switch (relation.getSemantics()) {
  case RelationSemantics::Iota: {
    auto axis = dyn_cast<IntegerAttr>(relation.getParameters()[0]);
    if (!axis || axis.getInt() < 0 || axis.getInt() >= coordinates.size())
      return failure();
    int64_t result = context.ptx.nextU32();
    emitOnlineSource(context, relation);
    context.ptx.stream << "  mov.u32 %r" << result << ", %r"
                       << coordinates[axis.getInt()] << ";\n";
    return remember({OnlineScalarKind::U32, result});
  }
  case RelationSemantics::Constant: {
    auto value = dyn_cast<IntegerAttr>(relation.getParameters()[0]);
    if (!value)
      return failure();
    int64_t result = context.ptx.nextU32();
    emitOnlineSource(context, relation);
    context.ptx.stream << "  mov.s32 %r" << result << ", " << value.getInt()
                       << ";\n";
    return remember({OnlineScalarKind::U32, result});
  }
  case RelationSemantics::Reshape:
  case RelationSemantics::BroadcastInDim: {
    if (relation.getInputs().size() != 1)
      return failure();
    auto inputType =
        dyn_cast<RankedTensorType>(relation.getInputs().front().getType());
    if (!inputType)
      return failure();
    FailureOr<SmallVector<int64_t>> inputCoordinates = failure();
    if (relation.getSemantics() == RelationSemantics::Reshape) {
      inputCoordinates = emitOnlineReshapeCoordinates(context.ptx, inputType,
                                                      resultType, coordinates);
    } else {
      if (relation.getParameters().size() !=
          static_cast<size_t>(inputType.getRank()))
        return failure();
      SmallVector<int64_t> mapped;
      for (auto [inputAxis, attribute] :
           llvm::enumerate(relation.getParameters())) {
        auto dimension = dyn_cast<IntegerAttr>(attribute);
        if (!dimension || dimension.getInt() < 0 ||
            dimension.getInt() >= coordinates.size())
          return failure();
        if (inputType.getDimSize(inputAxis) == 1 &&
            resultType.getDimSize(dimension.getInt()) != 1) {
          int64_t zero = context.ptx.nextU32();
          context.ptx.stream << "  mov.u32 %r" << zero << ", 0;\n";
          mapped.push_back(zero);
        } else {
          mapped.push_back(coordinates[dimension.getInt()]);
        }
      }
      inputCoordinates = std::move(mapped);
    }
    if (failed(inputCoordinates))
      return failure();
    FailureOr<OnlineScalarValue> input = emitOnlineValue(
        context, relation.getInputs().front(), *inputCoordinates);
    if (failed(input))
      return failure();
    int64_t result =
        input->kind == OnlineScalarKind::F32         ? context.ptx.nextF32()
        : input->kind == OnlineScalarKind::Predicate ? context.ptx.nextPred()
                                                     : context.ptx.nextU32();
    emitOnlineSource(context, relation);
    context.ptx.stream << (input->kind == OnlineScalarKind::F32 ? "  mov.f32 %f"
                           : input->kind == OnlineScalarKind::Predicate
                               ? "  mov.pred %p"
                               : "  mov.u32 %r")
                       << result << ", "
                       << (input->kind == OnlineScalarKind::F32         ? "%f"
                           : input->kind == OnlineScalarKind::Predicate ? "%p"
                                                                        : "%r")
                       << input->reg << ";\n";
    return remember({input->kind, result});
  }
  case RelationSemantics::Add: {
    if (relation.getInputs().size() != 2)
      return failure();
    SmallVector<OnlineScalarValue> inputs;
    for (Value input : relation.getInputs()) {
      FailureOr<OnlineScalarValue> scalar =
          emitOnlineValue(context, input, coordinates);
      if (failed(scalar) || scalar->kind != OnlineScalarKind::U32)
        return failure();
      inputs.push_back(*scalar);
    }
    int64_t result = context.ptx.nextU32();
    emitOnlineSource(context, relation);
    context.ptx.stream << "  add.s32 %r" << result << ", %r" << inputs[0].reg
                       << ", %r" << inputs[1].reg << ";\n";
    return remember({OnlineScalarKind::U32, result});
  }
  case RelationSemantics::Compare: {
    if (relation.getInputs().size() != 2 ||
        relation.getParameters().size() != 2)
      return failure();
    SmallVector<OnlineScalarValue> inputs;
    for (auto [operand, input] : llvm::enumerate(relation.getInputs())) {
      context.relationRoles.try_emplace(input, operand == 0 ? "relation_lhs"
                                                            : "relation_rhs");
      FailureOr<OnlineScalarValue> scalar =
          emitOnlineValue(context, input, coordinates);
      if (failed(scalar) || scalar->kind != OnlineScalarKind::U32)
        return failure();
      inputs.push_back(*scalar);
    }
    auto type = dyn_cast<StringAttr>(relation.getParameters()[0]);
    auto direction = dyn_cast<StringAttr>(relation.getParameters()[1]);
    if (!type || !direction)
      return failure();
    StringRef suffix = type.getValue() == "SIGNED" ? "s32" : "u32";
    StringRef comparison = llvm::StringSwitch<StringRef>(direction.getValue())
                               .Case("EQ", "eq")
                               .Case("NE", "ne")
                               .Case("GE", "ge")
                               .Case("GT", "gt")
                               .Case("LE", "le")
                               .Case("LT", "lt")
                               .Default("");
    if (comparison.empty())
      return failure();
    int64_t result = context.ptx.nextPred();
    emitOnlineSource(context, relation);
    emitOnlineRole(context.ptx, "relation_compare");
    context.ptx.stream << "  setp." << comparison << '.' << suffix << " %p"
                       << result << ", %r" << inputs[0].reg << ", %r"
                       << inputs[1].reg << ";\n";
    return remember({OnlineScalarKind::Predicate, result});
  }
  case RelationSemantics::Convert: {
    FailureOr<OnlineScalarValue> input =
        emitOnlineValue(context, relation.getInputs().front(), coordinates);
    if (failed(input) || input->kind != OnlineScalarKind::Predicate)
      return failure();
    int64_t result = context.ptx.nextF32();
    emitOnlineSource(context, relation);
    context.ptx.stream << "  selp.f32 %f" << result
                       << ", 0f3F800000, 0f00000000, %p" << input->reg << ";\n";
    return remember({OnlineScalarKind::F32, result});
  }
  case RelationSemantics::Select: {
    if (relation.getInputs().size() != 3)
      return failure();
    FailureOr<OnlineScalarValue> whenTrue =
        emitOnlineValue(context, relation.getInputs()[1], coordinates);
    FailureOr<OnlineScalarValue> whenFalse =
        emitOnlineValue(context, relation.getInputs()[2], coordinates);
    // The predicate and selected score are evaluated at the same score-domain
    // point. In the online path this includes the active reduction coordinate;
    // the projected launch coordinates alone deliberately omit that dimension.
    FailureOr<OnlineScalarValue> predicate =
        emitOnlineValue(context, relation.getInputs()[0], coordinates);
    if (failed(predicate) || failed(whenTrue) || failed(whenFalse) ||
        predicate->kind != OnlineScalarKind::Predicate ||
        whenTrue->kind != OnlineScalarKind::F32 ||
        whenFalse->kind != OnlineScalarKind::F32)
      return failure();
    int64_t score = context.ptx.nextF32();
    int64_t negativeInfinity = context.ptx.nextF32();
    emitOnlineRole(context.ptx, "score_value");
    context.ptx.stream << "  mov.f32 %f" << score << ", %f" << whenTrue->reg
                       << ";\n";
    emitOnlineRole(context.ptx, "negative_infinity");
    context.ptx.stream << "  mov.f32 %f" << negativeInfinity << ", %f"
                       << whenFalse->reg << ";\n";
    int64_t result = context.ptx.nextF32();
    emitOnlineSource(context, relation);
    emitOnlineRole(context.ptx, "mask_select");
    context.ptx.stream << "  selp.f32 %f" << result << ", %f" << score << ", %f"
                       << negativeInfinity << ", %p" << predicate->reg << ";\n";
    return remember({OnlineScalarKind::F32, result});
  }
  }
  return failure();
}

int64_t emitOnlineExp(PtxEmitter &ptx, int64_t argument) {
  int64_t scaled = ptx.nextF32();
  int64_t result = ptx.nextF32();
  ptx.stream << "  mul.rn.f32 %f" << scaled << ", %f" << argument
             << ", 0f3FB8AA3B;\n  ex2.approx.f32 %f" << result << ", %f"
             << scaled << ";\n";
  return result;
}

FailureOr<int64_t> emitOnlineGlobalStorePointer(PtxEmitter &ptx,
                                                int64_t parameter,
                                                RankedTensorType type,
                                                ArrayRef<int64_t> coordinates) {
  if (coordinates.size() != static_cast<size_t>(type.getRank()))
    return failure();
  int64_t linear = ptx.nextU64();
  ptx.stream << "  mov.u64 %rd" << linear << ", 0;\n";
  for (auto [axis, coordinate] : llvm::enumerate(coordinates)) {
    int64_t wide = ptx.nextU64();
    ptx.stream << "  cvt.u64.u32 %rd" << wide << ", %r" << coordinate
               << ";\n  mad.lo.u64 %rd" << linear << ", %rd" << linear << ", "
               << type.getDimSize(axis) << ", %rd" << wide << ";\n";
  }
  int64_t pointer = ptx.nextU64();
  const int64_t bytes = type.getElementType().isBF16() ? 2 : 4;
  ptx.stream << "  ld.param.u64 %rd" << pointer << ", [p" << parameter
             << "];\n  mad.lo.u64 %rd" << pointer << ", %rd" << linear << ", "
             << bytes << ", %rd" << pointer << ";\n";
  return pointer;
}

FailureOr<SmallVector<int64_t>>
onlineOutputCoordinates(Value ancestor, Value output,
                        ArrayRef<int64_t> ancestorCoordinates) {
  SmallVector<MapOp> reverseChain;
  Value current = output;
  while (current != ancestor) {
    auto map = current.getDefiningOp<MapOp>();
    if (!map || map.getNumResults() != 1)
      return failure();
    auto input = llvm::find_if(map.getInputs(), [&](Value candidate) {
      return candidate == ancestor ||
             valueDependsOnOperation(candidate, ancestor.getDefiningOp());
    });
    if (input == map.getInputs().end())
      return failure();
    reverseChain.push_back(map);
    current = *input;
  }
  SmallVector<int64_t> coordinates(ancestorCoordinates);
  for (MapOp map : llvm::reverse(reverseChain)) {
    auto input = llvm::find_if(map.getInputs(), [&](Value candidate) {
      return candidate == current ||
             valueDependsOnOperation(current, candidate.getDefiningOp());
    });
    if (input == map.getInputs().end())
      return failure();
    unsigned operand = std::distance(map.getInputs().begin(), input);
    auto indexing = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[operand]);
    auto resultType = dyn_cast<RankedTensorType>(map.getResult(0).getType());
    if (!indexing || !resultType ||
        indexing.getValue().getNumResults() != coordinates.size())
      return failure();
    SmallVector<int64_t> next(resultType.getRank(), -1);
    for (auto [axis, expression] :
         llvm::enumerate(indexing.getValue().getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension || dimension.getPosition() >= next.size() ||
          next[dimension.getPosition()] >= 0)
        return failure();
      next[dimension.getPosition()] = coordinates[axis];
    }
    if (llvm::is_contained(next, -1))
      return failure();
    coordinates = std::move(next);
    current = map.getResult(0);
  }
  return coordinates;
}

void emitOnlinePartialStateLoop(PtxEmitter &ptx, int64_t threads,
                                int64_t sharedBase, int64_t &partialM,
                                int64_t &partialL, int64_t &partialO,
                                int64_t &index, int64_t label) {
  int64_t predicate = ptx.nextPred();
  ptx.stream << "  mov.f32 %f" << partialM << ", 0fFF800000;\n  mov.f32 %f"
             << partialL << ", 0f00000000;\n  mov.f32 %f" << partialO
             << ", 0f00000000;\n  mov.u32 %r" << index
             << ", 1;\n$online_partial_" << label << "_loop:\n  setp.ge.u32 %p"
             << predicate << ", %r" << index << ", " << threads << ";\n  @%p"
             << predicate << " bra $online_partial_" << label << "_done;\n";
  int64_t offset = ptx.nextU64();
  int64_t pointerM = ptx.nextU64();
  int64_t pointerL = ptx.nextU64();
  int64_t pointerO = ptx.nextU64();
  int64_t rightM = ptx.nextF32();
  int64_t rightL = ptx.nextF32();
  int64_t rightO = ptx.nextF32();
  ptx.stream << "  cvt.u64.u32 %rd" << offset << ", %r" << index
             << ";\n  mul.lo.u64 %rd" << offset << ", %rd" << offset
             << ", 4;\n  add.u64 %rd" << pointerM << ", %rd" << sharedBase
             << ", %rd" << offset << ";\n  add.u64 %rd" << pointerL << ", %rd"
             << pointerM << ", " << threads * 4 << ";\n  add.u64 %rd"
             << pointerO << ", %rd" << pointerM << ", " << threads * 8
             << ";\n  ld.shared.f32 %f" << rightM << ", [%rd" << pointerM
             << "];\n  ld.shared.f32 %f" << rightL << ", [%rd" << pointerL
             << "];\n  ld.shared.f32 %f" << rightO << ", [%rd" << pointerO
             << "];\n";
  int64_t mergedM = ptx.nextF32();
  int64_t leftDelta = ptx.nextF32();
  int64_t rightDelta = ptx.nextF32();
  int64_t leftScale = ptx.nextF32();
  int64_t rightScale = ptx.nextF32();
  int64_t scaledLeftL = ptx.nextF32();
  int64_t scaledRightL = ptx.nextF32();
  int64_t mergedL = ptx.nextF32();
  int64_t scaledLeftO = ptx.nextF32();
  int64_t scaledRightO = ptx.nextF32();
  int64_t mergedO = ptx.nextF32();
  ptx.stream << "  max.f32 %f" << mergedM << ", %f" << partialM << ", %f"
             << rightM << ";\n  sub.rn.f32 %f" << leftDelta << ", %f"
             << partialM << ", %f" << mergedM << ";\n  sub.rn.f32 %f"
             << rightDelta << ", %f" << rightM << ", %f" << mergedM << ";\n";
  leftScale = emitOnlineExp(ptx, leftDelta);
  rightScale = emitOnlineExp(ptx, rightDelta);
  ptx.stream << "  mul.rn.f32 %f" << scaledLeftL << ", %f" << leftScale
             << ", %f" << partialL << ";\n  mul.rn.f32 %f" << scaledRightL
             << ", %f" << rightScale << ", %f" << rightL << ";\n  add.rn.f32 %f"
             << mergedL << ", %f" << scaledLeftL << ", %f" << scaledRightL
             << ";\n  mul.rn.f32 %f" << scaledLeftO << ", %f" << leftScale
             << ", %f" << partialO << ";\n  mul.rn.f32 %f" << scaledRightO
             << ", %f" << rightScale << ", %f" << rightO << ";\n  add.rn.f32 %f"
             << mergedO << ", %f" << scaledLeftO << ", %f" << scaledRightO
             << ";\n  mov.f32 %f" << partialM << ", %f" << mergedM
             << ";\n  mov.f32 %f" << partialL << ", %f" << mergedL
             << ";\n  mov.f32 %f" << partialO << ", %f" << mergedO
             << ";\n  add.u32 %r" << index << ", %r" << index
             << ", 1;\n  bra $online_partial_" << label
             << "_loop;\n$online_partial_" << label << "_done:\n";
}

FailureOr<OnlineEvalContext> onlineEvalContext(PtxEmitter &ptx, RegionOp region,
                                               MaterializationTaskOp task,
                                               ScheduleTaskOp schedule,
                                               OnlineEmissionMatch &match) {
  OnlineEvalContext context{ptx, region, task, schedule, match};
  const unsigned arguments = region.getBody().front().getNumArguments();
  for (auto [parameter, buffer] : llvm::enumerate(task.getInputBuffers())) {
    if (buffer >= 0 && buffer < arguments)
      context.argumentParameters.try_emplace(buffer, parameter);
    else
      context.maximumParameter = parameter;
  }
  if (context.argumentParameters.empty())
    return failure();
  return context;
}

FailureOr<std::string>
emitOnlineMaximumPtx(RegionOp region, MaterializationTaskOp materializationTask,
                     SmallVectorImpl<MaterializationBufferOp> &buffers,
                     ScheduleTaskOp task, OnlineEmissionMatch &match) {
  if (materializationTask.getKind() != MaterializationTaskKind::Fold ||
      materializationTask.getOutputBuffers().size() != 1 ||
      task.getReductionAxis() !=
          std::optional<int64_t>(
              match.maximumFold.getReductionDimensions().front()))
    return failure();
  PtxEmitter ptx(materializationTask.getInputBuffers().size() + 1);
  ptx.header(task.getSemanticFingerprint(), /*usesDynamicSharedMemory=*/true);
  FailureOr<OnlineEvalContext> context =
      onlineEvalContext(ptx, region, materializationTask, task, match);
  FailureOr<SmallVector<int64_t>> scoreCoordinates =
      emitOnlineLogicalCoordinates(ptx, task);
  if (failed(context) || failed(scoreCoordinates))
    return failure();
  FailureOr<OnlineScalarValue> score =
      emitOnlineValue(*context, match.score, *scoreCoordinates);
  if (failed(score) || score->kind != OnlineScalarKind::F32)
    return failure();
  int64_t maxInit = ptx.nextF32();
  emitOnlineRole(ptx, "max_init");
  ptx.stream << "  mov.f32 %f" << maxInit << ", 0fFF800000;\n";
  const int64_t threads = task.getWorkgroupThreads();
  int64_t sharedBase = ptx.nextU64();
  int64_t offset = ptx.nextU64();
  int64_t leafPointer = ptx.nextU64();
  ptx.stream << "  mov.u64 %rd" << sharedBase
             << ", shuttle_smem;\n  cvta.to.shared.u64 %rd" << sharedBase
             << ", %rd" << sharedBase << ";\n  mov.u32 %r0, %tid.x;\n"
             << "  cvt.u64.u32 %rd" << offset << ", %r0;\n  mul.lo.u64 %rd"
             << offset << ", %rd" << offset << ", 4;\n  add.u64 %rd"
             << leafPointer << ", %rd" << sharedBase << ", %rd" << offset
             << ";\n  st.shared.f32 [%rd" << leafPointer << "], %f"
             << score->reg << ";\n  bar.sync 0;\n";
  int64_t notLeader = ptx.nextPred();
  ptx.stream << "  setp.ne.u32 %p" << notLeader << ", %r0, 0;\n  @%p"
             << notLeader << " bra $done;\n";
  int64_t partial = ptx.nextF32();
  int64_t index = ptx.nextU32();
  int64_t loopPredicate = ptx.nextPred();
  const int64_t label = context->label++;
  ptx.stream << "  mov.f32 %f" << partial << ", %f" << maxInit
             << ";\n  mov.u32 %r" << index << ", 1;\n$online_max_" << label
             << "_loop:\n  setp.ge.u32 %p" << loopPredicate << ", %r" << index
             << ", " << threads << ";\n  @%p" << loopPredicate
             << " bra $online_max_" << label << "_done;\n";
  int64_t loopOffset = ptx.nextU64();
  int64_t loopPointer = ptx.nextU64();
  int64_t loaded = ptx.nextF32();
  int64_t combined = ptx.nextF32();
  ptx.stream << "  cvt.u64.u32 %rd" << loopOffset << ", %r" << index
             << ";\n  mul.lo.u64 %rd" << loopOffset << ", %rd" << loopOffset
             << ", 4;\n  add.u64 %rd" << loopPointer << ", %rd" << sharedBase
             << ", %rd" << loopOffset << ";\n  ld.shared.f32 %f" << loaded
             << ", [%rd" << loopPointer << "];\n  max.f32 %f" << combined
             << ", %f" << partial << ", %f" << loaded << ";\n  mov.f32 %f"
             << partial << ", %f" << combined << ";\n  add.u32 %r" << index
             << ", %r" << index << ", 1;\n  bra $online_max_" << label
             << "_loop;\n$online_max_" << label << "_done:\n";
  int64_t maximum = ptx.nextF32();
  emitOnlineRole(ptx, "max_update");
  ptx.stream << "  max.f32 %f" << maximum << ", %f" << partial << ", %f"
             << score->reg << ";\n";
  auto compact = compositionBufferType(materializationTask, buffers, true, 0);
  if (!compact || !compact.getElementType().isF32())
    return failure();
  SmallVector<int64_t> compactCoordinates;
  for (auto [axis, coordinate] : llvm::enumerate(*scoreCoordinates))
    if (static_cast<int64_t>(axis) != *task.getReductionAxis())
      compactCoordinates.push_back(coordinate);
  FailureOr<int64_t> pointer = emitOnlineGlobalStorePointer(
      ptx, materializationTask.getInputBuffers().size(), compact,
      compactCoordinates);
  if (failed(pointer))
    return failure();
  emitOnlineRole(ptx, "compact_store");
  ptx.stream << "  st.global.f32 [%rd" << *pointer << "], %f" << maximum
             << ";\n";
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string>
emitOnlineStatePtx(RegionOp region, MaterializationTaskOp materializationTask,
                   SmallVectorImpl<MaterializationBufferOp> &buffers,
                   ScheduleTaskOp task, OnlineEmissionMatch &match) {
  if (materializationTask.getKind() != MaterializationTaskKind::Contract ||
      materializationTask.getOutputBuffers().size() != 1 ||
      match.normalizedOperand >=
          match.weightedContract.getIndexingMaps().size())
    return failure();
  PtxEmitter ptx(materializationTask.getInputBuffers().size() + 1);
  ptx.header(task.getSemanticFingerprint(), /*usesDynamicSharedMemory=*/true);
  FailureOr<OnlineEvalContext> context =
      onlineEvalContext(ptx, region, materializationTask, task, match);
  FailureOr<SmallVector<int64_t>> contractCoordinates =
      emitOnlineLogicalCoordinates(ptx, task);
  if (failed(context) || failed(contractCoordinates))
    return failure();
  auto normalizedMap = dyn_cast<AffineMapAttr>(
      match.weightedContract.getIndexingMaps()[match.normalizedOperand]);
  const unsigned valueOperand = 1 - match.normalizedOperand;
  auto valueMap = dyn_cast<AffineMapAttr>(
      match.weightedContract.getIndexingMaps()[valueOperand]);
  if (!normalizedMap || !valueMap)
    return failure();
  FailureOr<SmallVector<int64_t>> scoreCoordinates =
      emitOnlineMappedU32(ptx, normalizedMap.getValue(), *contractCoordinates);
  FailureOr<SmallVector<int64_t>> valueCoordinates =
      emitOnlineMappedU32(ptx, valueMap.getValue(), *contractCoordinates);
  if (failed(scoreCoordinates) || failed(valueCoordinates))
    return failure();
  FailureOr<OnlineScalarValue> score =
      emitOnlineValue(*context, match.score, *scoreCoordinates);
  FailureOr<OnlineScalarValue> value = emitOnlineValue(
      *context, match.weightedContract.getInputs()[valueOperand],
      *valueCoordinates);
  if (failed(score) || failed(value) || score->kind != OnlineScalarKind::F32 ||
      value->kind != OnlineScalarKind::F32)
    return failure();

  int64_t maxInit = ptx.nextF32();
  int64_t zeroInit = ptx.nextF32();
  emitOnlineRole(ptx, "max_init");
  ptx.stream << "  mov.f32 %f" << maxInit << ", 0fFF800000;\n";
  emitOnlineRole(ptx, "zero_init");
  ptx.stream << "  mov.f32 %f" << zeroInit << ", 0f00000000;\n";
  int64_t stateM = ptx.nextF32();
  int64_t stateL = ptx.nextF32();
  int64_t stateO = ptx.nextF32();
  ptx.stream << "  mov.f32 %f" << stateM << ", %f" << maxInit << ";\n";
  ptx.stream << "  mov.f32 %f" << stateL << ", %f" << zeroInit << ";\n";
  ptx.stream << "  mov.f32 %f" << stateO << ", %f" << zeroInit << ";\n";
  int64_t maxUpdate = ptx.nextF32();
  ptx.stream << "  max.f32 %f" << maxUpdate << ", %f" << stateM << ", %f"
             << score->reg << ";\n";
  int64_t oldDelta = ptx.nextF32();
  ptx.stream << "  sub.rn.f32 %f" << oldDelta << ", %f" << stateM << ", %f"
             << maxUpdate << ";\n";
  int64_t oldExpArgument = ptx.nextF32();
  ptx.stream << "  mul.rn.f32 %f" << oldExpArgument << ", %f" << oldDelta
             << ", 0f3FB8AA3B;\n";
  int64_t oldRescale = ptx.nextF32();
  ptx.stream << "  ex2.approx.f32 %f" << oldRescale << ", %f" << oldExpArgument
             << ";\n";
  int64_t scoreDelta = ptx.nextF32();
  ptx.stream << "  sub.rn.f32 %f" << scoreDelta << ", %f" << score->reg
             << ", %f" << maxUpdate << ";\n";
  int64_t scoreExpArgument = ptx.nextF32();
  ptx.stream << "  mul.rn.f32 %f" << scoreExpArgument << ", %f" << scoreDelta
             << ", 0f3FB8AA3B;\n";
  int64_t scoreWeight = ptx.nextF32();
  ptx.stream << "  ex2.approx.f32 %f" << scoreWeight << ", %f"
             << scoreExpArgument << ";\n";
  int64_t valueLoad = ptx.nextF32();
  emitOnlineRole(ptx, "value_load");
  ptx.stream << "  mov.f32 %f" << valueLoad << ", %f" << value->reg << ";\n";
  int64_t valueWeight = ptx.nextF32();
  ptx.stream << "  mul.rn.f32 %f" << valueWeight << ", %f" << scoreWeight
             << ", %f" << valueLoad << ";\n";
  int64_t lUpdate = ptx.nextF32();
  int64_t oUpdate = ptx.nextF32();
  ptx.stream << "  fma.rn.f32 %f" << lUpdate << ", %f" << oldRescale << ", %f"
             << stateL << ", %f" << scoreWeight << ";\n";
  ptx.stream << "  fma.rn.f32 %f" << oUpdate << ", %f" << oldRescale << ", %f"
             << stateO << ", %f" << valueWeight << ";\n";

  const int64_t threads = task.getWorkgroupThreads();
  int64_t sharedBase = ptx.nextU64();
  int64_t offset = ptx.nextU64();
  int64_t pointerM = ptx.nextU64();
  int64_t pointerL = ptx.nextU64();
  int64_t pointerO = ptx.nextU64();
  ptx.stream << "  mov.u64 %rd" << sharedBase
             << ", shuttle_smem;\n  cvta.to.shared.u64 %rd" << sharedBase
             << ", %rd" << sharedBase << ";\n  mov.u32 %r0, %tid.x;\n"
             << "  cvt.u64.u32 %rd" << offset << ", %r0;\n  mul.lo.u64 %rd"
             << offset << ", %rd" << offset << ", 4;\n  add.u64 %rd" << pointerM
             << ", %rd" << sharedBase << ", %rd" << offset << ";\n  add.u64 %rd"
             << pointerL << ", %rd" << pointerM << ", " << threads * 4
             << ";\n  add.u64 %rd" << pointerO << ", %rd" << pointerM << ", "
             << threads * 8 << ";\n  st.shared.f32 [%rd" << pointerM << "], %f"
             << maxUpdate << ";\n  st.shared.f32 [%rd" << pointerL << "], %f"
             << lUpdate << ";\n  st.shared.f32 [%rd" << pointerO << "], %f"
             << oUpdate << ";\n  bar.sync 0;\n";
  int64_t notLeader = ptx.nextPred();
  ptx.stream << "  setp.ne.u32 %p" << notLeader << ", %r0, 0;\n  @%p"
             << notLeader << " bra $done;\n";
  int64_t partialM = ptx.nextF32();
  int64_t partialL = ptx.nextF32();
  int64_t partialO = ptx.nextF32();
  int64_t index = ptx.nextU32();
  emitOnlinePartialStateLoop(ptx, threads, sharedBase, partialM, partialL,
                             partialO, index, context->label++);
  int64_t leftM = ptx.nextF32();
  int64_t leftL = ptx.nextF32();
  int64_t leftO = ptx.nextF32();
  ptx.stream << "  max.f32 %f" << leftM << ", %f" << partialM << ", %f"
             << maxInit << ";\n  add.rn.f32 %f" << leftL << ", %f" << partialL
             << ", %f" << zeroInit << ";\n  add.rn.f32 %f" << leftO << ", %f"
             << partialO << ", %f" << zeroInit << ";\n";
  // Rebind the semantic left state to the partial reduction while preserving
  // the role-specific recurrence proof.
  emitOnlineRole(ptx, "state_m");
  ptx.stream << "  mov.f32 %f" << stateM << ", %f" << leftM << ";\n";
  emitOnlineRole(ptx, "state_l");
  ptx.stream << "  mov.f32 %f" << stateL << ", %f" << leftL << ";\n";
  emitOnlineRole(ptx, "state_o");
  ptx.stream << "  mov.f32 %f" << stateO << ", %f" << leftO << ";\n";
  emitOnlineRole(ptx, "max_update");
  ptx.stream << "  max.f32 %f" << maxUpdate << ", %f" << stateM << ", %f"
             << score->reg << ";\n";
  emitOnlineRole(ptx, "old_delta");
  ptx.stream << "  sub.rn.f32 %f" << oldDelta << ", %f" << stateM << ", %f"
             << maxUpdate << ";\n";
  emitOnlineRole(ptx, "old_exp_argument");
  ptx.stream << "  mul.rn.f32 %f" << oldExpArgument << ", %f" << oldDelta
             << ", 0f3FB8AA3B;\n";
  emitOnlineRole(ptx, "old_rescale");
  ptx.stream << "  ex2.approx.f32 %f" << oldRescale << ", %f" << oldExpArgument
             << ";\n";
  emitOnlineRole(ptx, "score_delta");
  ptx.stream << "  sub.rn.f32 %f" << scoreDelta << ", %f" << score->reg
             << ", %f" << maxUpdate << ";\n";
  emitOnlineRole(ptx, "score_exp_argument");
  ptx.stream << "  mul.rn.f32 %f" << scoreExpArgument << ", %f" << scoreDelta
             << ", 0f3FB8AA3B;\n";
  emitOnlineRole(ptx, "score_weight");
  ptx.stream << "  ex2.approx.f32 %f" << scoreWeight << ", %f"
             << scoreExpArgument << ";\n";
  emitOnlineRole(ptx, "value_weight");
  ptx.stream << "  mul.rn.f32 %f" << valueWeight << ", %f" << scoreWeight
             << ", %f" << valueLoad << ";\n";
  emitOnlineRole(ptx, "l_update");
  ptx.stream << "  fma.rn.f32 %f" << lUpdate << ", %f" << oldRescale << ", %f"
             << stateL << ", %f" << scoreWeight << ";\n";
  emitOnlineRole(ptx, "o_update");
  ptx.stream << "  fma.rn.f32 %f" << oUpdate << ", %f" << oldRescale << ", %f"
             << stateO << ", %f" << valueWeight << ";\n";

  int64_t finalMPtr = sharedBase;
  int64_t finalLPtr = ptx.nextU64();
  int64_t finalOPtr = ptx.nextU64();
  ptx.stream << "  add.u64 %rd" << finalLPtr << ", %rd" << sharedBase << ", "
             << threads * 4 << ";\n  add.u64 %rd" << finalOPtr << ", %rd"
             << sharedBase << ", " << threads * 8 << ";\n";
  emitOnlineRole(ptx, "shared_m_store");
  ptx.stream << "  st.shared.f32 [%rd" << finalMPtr << "], %f" << maxUpdate
             << ";\n";
  emitOnlineRole(ptx, "shared_l_store");
  ptx.stream << "  st.shared.f32 [%rd" << finalLPtr << "], %f" << lUpdate
             << ";\n";
  emitOnlineRole(ptx, "shared_o_store");
  ptx.stream << "  st.shared.f32 [%rd" << finalOPtr << "], %f" << oUpdate
             << ";\n";
  emitOnlineRole(ptx, "merge_barrier");
  ptx.stream << "  bar.sync 0;\n";
  int64_t sharedM = ptx.nextF32();
  int64_t sharedL = ptx.nextF32();
  int64_t sharedO = ptx.nextF32();
  emitOnlineRole(ptx, "shared_m_load");
  ptx.stream << "  ld.shared.f32 %f" << sharedM << ", [%rd" << finalMPtr
             << "];\n";
  emitOnlineRole(ptx, "shared_l_load");
  ptx.stream << "  ld.shared.f32 %f" << sharedL << ", [%rd" << finalLPtr
             << "];\n";
  emitOnlineRole(ptx, "shared_o_load");
  ptx.stream << "  ld.shared.f32 %f" << sharedO << ", [%rd" << finalOPtr
             << "];\n";
  int64_t mergeMax = ptx.nextF32();
  emitOnlineRole(ptx, "merge_max");
  ptx.stream << "  max.f32 %f" << mergeMax << ", %f" << maxUpdate << ", %f"
             << sharedM << ";\n";
  int64_t mergeLeftDelta = ptx.nextF32();
  int64_t mergeRightDelta = ptx.nextF32();
  emitOnlineRole(ptx, "merge_left_delta");
  ptx.stream << "  sub.rn.f32 %f" << mergeLeftDelta << ", %f" << maxUpdate
             << ", %f" << mergeMax << ";\n";
  int64_t mergeLeftArg = ptx.nextF32();
  emitOnlineRole(ptx, "merge_left_exp_argument");
  ptx.stream << "  mul.rn.f32 %f" << mergeLeftArg << ", %f" << mergeLeftDelta
             << ", 0f3FB8AA3B;\n";
  int64_t mergeLeftScale = ptx.nextF32();
  emitOnlineRole(ptx, "merge_left_rescale");
  ptx.stream << "  ex2.approx.f32 %f" << mergeLeftScale << ", %f"
             << mergeLeftArg << ";\n";
  emitOnlineRole(ptx, "merge_right_delta");
  ptx.stream << "  sub.rn.f32 %f" << mergeRightDelta << ", %f" << sharedM
             << ", %f" << mergeMax << ";\n";
  int64_t mergeRightArg = ptx.nextF32();
  emitOnlineRole(ptx, "merge_right_exp_argument");
  ptx.stream << "  mul.rn.f32 %f" << mergeRightArg << ", %f" << mergeRightDelta
             << ", 0f3FB8AA3B;\n";
  int64_t mergeRightScale = ptx.nextF32();
  emitOnlineRole(ptx, "merge_right_rescale");
  ptx.stream << "  ex2.approx.f32 %f" << mergeRightScale << ", %f"
             << mergeRightArg << ";\n";
  int64_t mergeLeftL = ptx.nextF32();
  int64_t mergeRightL = ptx.nextF32();
  int64_t mergeL = ptx.nextF32();
  emitOnlineRole(ptx, "merge_left_l");
  ptx.stream << "  mul.rn.f32 %f" << mergeLeftL << ", %f" << mergeLeftScale
             << ", %f" << lUpdate << ";\n";
  emitOnlineRole(ptx, "merge_right_l");
  ptx.stream << "  mul.rn.f32 %f" << mergeRightL << ", %f" << mergeRightScale
             << ", %f" << sharedL << ";\n";
  emitOnlineRole(ptx, "merge_l");
  ptx.stream << "  add.rn.f32 %f" << mergeL << ", %f" << mergeLeftL << ", %f"
             << mergeRightL << ";\n";
  int64_t mergeLeftO = ptx.nextF32();
  int64_t mergeRightO = ptx.nextF32();
  int64_t mergeO = ptx.nextF32();
  emitOnlineRole(ptx, "merge_left_o");
  ptx.stream << "  mul.rn.f32 %f" << mergeLeftO << ", %f" << mergeLeftScale
             << ", %f" << oUpdate << ";\n";
  emitOnlineRole(ptx, "merge_right_o");
  ptx.stream << "  mul.rn.f32 %f" << mergeRightO << ", %f" << mergeRightScale
             << ", %f" << sharedO << ";\n";
  emitOnlineRole(ptx, "merge_o");
  ptx.stream << "  add.rn.f32 %f" << mergeO << ", %f" << mergeLeftO << ", %f"
             << mergeRightO << ";\n";
  int64_t finalized = ptx.nextF32();
  emitOnlineRole(ptx, "finalize");
  ptx.stream << "  div.rn.f32 %f" << finalized << ", %f" << mergeO << ", %f"
             << mergeL << ";\n";

  auto weightedMaps = match.weightedContract.getIndexingMaps();
  auto outputMap =
      dyn_cast<AffineMapAttr>(weightedMaps[weightedMaps.size() - 1]);
  if (!outputMap)
    return failure();
  FailureOr<SmallVector<int64_t>> contractResultCoordinates =
      emitOnlineMappedU32(ptx, outputMap.getValue(), *contractCoordinates);
  auto regionYield = cast<YieldOp>(region.getBody().front().getTerminator());
  FailureOr<SmallVector<int64_t>> outputCoordinates = onlineOutputCoordinates(
      match.weightedContract.getResult(0), regionYield.getValues().front(),
      *contractResultCoordinates);
  auto outputType =
      dyn_cast<RankedTensorType>(regionYield.getValues().front().getType());
  if (failed(outputCoordinates) || !outputType ||
      !outputType.getElementType().isF32())
    return failure();
  FailureOr<int64_t> outputPointer = emitOnlineGlobalStorePointer(
      ptx, materializationTask.getInputBuffers().size(), outputType,
      *outputCoordinates);
  if (failed(outputPointer))
    return failure();
  emitOnlineRole(ptx, "public_store");
  ptx.stream << "  st.global.f32 [%rd" << *outputPointer << "], %f" << finalized
             << ";\n";
  ptx.finish();
  return ptx.text;
}

FailureOr<OnlineScalarValue>
emitOnlineEpilogue(OnlineEvalContext &context, Value value,
                   Value weightedResult, OnlineScalarValue weightedScalar,
                   ArrayRef<int64_t> coordinates) {
  if (value == weightedResult)
    return weightedScalar;
  auto map = value.getDefiningOp<MapOp>();
  if (!map || map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != map.getInputs().size() + 1)
    return failure();
  SmallVector<int64_t> arguments;
  for (auto [operand, input] : llvm::enumerate(map.getInputs())) {
    auto indexing = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[operand]);
    if (!indexing)
      return failure();
    FailureOr<SmallVector<int64_t>> inputCoordinates =
        emitOnlineMappedU32(context.ptx, indexing.getValue(), coordinates);
    if (failed(inputCoordinates))
      return failure();
    FailureOr<OnlineScalarValue> scalar =
        input == weightedResult ||
                valueDependsOnOperation(input, weightedResult.getDefiningOp())
            ? emitOnlineEpilogue(context, input, weightedResult, weightedScalar,
                                 *inputCoordinates)
            : emitOnlineValue(context, input, *inputCoordinates);
    if (failed(scalar) || scalar->kind != OnlineScalarKind::F32)
      return failure();
    arguments.push_back(scalar->reg);
  }
  FailureOr<int64_t> result =
      emitScalarBody(context.ptx, map.getBody().front(), arguments);
  if (failed(result))
    return failure();
  return OnlineScalarValue{OnlineScalarKind::F32, *result};
}

FailureOr<std::string>
emitOnlineVectorStatePtx(RegionOp region,
                         MaterializationTaskOp materializationTask,
                         SmallVectorImpl<MaterializationBufferOp> &buffers,
                         ScheduleTaskOp task, OnlineEmissionMatch &match) {
  (void)buffers;
  auto vectorDimensions = materializationTask.getOnlineVectorDimensions();
  if (materializationTask.getKind() != MaterializationTaskKind::Contract ||
      materializationTask.getOutputBuffers().size() != 1 || !vectorDimensions ||
      vectorDimensions->size() != 1 || !task.getReductionAxis() ||
      task.getWorkgroupThreads() <= 0 || task.getWorkgroupThreads() % 32 != 0 ||
      task.getSubgroupSize() != 32)
    return failure();
  const int64_t vectorAxis = vectorDimensions->front();
  const int64_t reductionAxis = *task.getReductionAxis();
  ArrayRef<int64_t> domain = materializationTask.getDomainShape();
  if (vectorAxis < 0 || vectorAxis >= static_cast<int64_t>(domain.size()) ||
      reductionAxis < 0 ||
      reductionAxis >= static_cast<int64_t>(domain.size()) ||
      vectorAxis == reductionAxis || domain[vectorAxis] <= 0 ||
      domain[vectorAxis] > 128 || domain[reductionAxis] <= 0 ||
      domain[reductionAxis] > 2048)
    return failure();
  auto normalizedMap = dyn_cast<AffineMapAttr>(
      match.weightedContract.getIndexingMaps()[match.normalizedOperand]);
  const unsigned valueOperand = 1 - match.normalizedOperand;
  auto valueMap = dyn_cast<AffineMapAttr>(
      match.weightedContract.getIndexingMaps()[valueOperand]);
  auto weightedOutputMap =
      dyn_cast<AffineMapAttr>(match.weightedContract.getIndexingMaps()[2]);
  if (!normalizedMap || !valueMap || !weightedOutputMap)
    return failure();
  FailureOr<AffineMap> rawScoreMap = onlineAncestorCoordinateMap(
      match.score, match.scoreContract.getResult(0));
  if (failed(rawScoreMap) ||
      rawScoreMap->getNumDims() != normalizedMap.getValue().getNumResults())
    return failure();
  AffineMap scoreInputMap = rawScoreMap->compose(normalizedMap.getValue());
  FailureOr<OnlineScoreGeometry> scoreGeometry =
      onlineScoreGeometry(match, scoreInputMap, reductionAxis);
  if (failed(scoreGeometry) || scoreGeometry->reductionAxis < 0 ||
      scoreGeometry->reductionAxis >=
          static_cast<int64_t>(scoreGeometry->domainShape.size()))
    return failure();
  const int64_t scoreElements =
      scoreGeometry->domainShape[scoreGeometry->reductionAxis];
  constexpr int64_t kSharedBytes = 16 * 1024;
  const int64_t tileElements = std::min<int64_t>(
      domain[reductionAxis],
      kSharedBytes / (2 * (scoreElements + domain[vectorAxis])));
  if (scoreElements <= 0 || scoreElements > 256 || tileElements <= 0 ||
      task.getScratchBytes() < kSharedBytes)
    return failure();
  std::optional<int64_t> causalQueryAxis;
  const bool hasOffsetRelation =
      llvm::any_of(match.relations, [](RelationOp op) {
        return op.getSemantics() == RelationSemantics::Add;
      });
  if (!hasOffsetRelation) {
    for (auto [axis, extent] : llvm::enumerate(domain)) {
      if (static_cast<int64_t>(axis) == reductionAxis ||
          static_cast<int64_t>(axis) == vectorAxis ||
          extent != domain[reductionAxis] || extent % 4 != 0)
        continue;
      if (causalQueryAxis)
        return failure();
      causalQueryAxis = axis;
    }
  }
  PtxEmitter ptx(materializationTask.getInputBuffers().size() + 1);
  ptx.header(task.getSemanticFingerprint(), /*usesDynamicSharedMemory=*/true);
  FailureOr<OnlineEvalContext> context =
      onlineEvalContext(ptx, region, materializationTask, task, match);
  FailureOr<SmallVector<int64_t>> contractCoordinates =
      emitOnlineLogicalCoordinates(ptx, task);
  if (failed(context) || failed(contractCoordinates))
    return failure();
  int64_t thread = ptx.nextU32();
  int64_t lane = ptx.nextU32();
  int64_t warp = ptx.nextU32();
  int64_t block = ptx.nextU32();
  int64_t sharedRow = ptx.nextU32();
  int64_t localBlock = ptx.nextU32();
  int64_t localRow = ptx.nextU32();
  int64_t row = ptx.nextU32();
  int64_t rowOutOfBounds = ptx.nextPred();
  SmallVector<int64_t> rowShape;
  for (auto [axis, extent] : llvm::enumerate(domain))
    if (static_cast<int64_t>(axis) != reductionAxis &&
        static_cast<int64_t>(axis) != vectorAxis)
      rowShape.push_back(extent);
  FailureOr<int64_t> rowCount = scheduleElementCount(rowShape);
  if (failed(rowCount) || rowShape.size() < 2)
    return failure();
  FailureOr<int64_t> sharedRows =
      scheduleElementCount(ArrayRef<int64_t>(rowShape).drop_back(2));
  FailureOr<int64_t> localRows =
      scheduleElementCount(ArrayRef<int64_t>(rowShape).take_back(2));
  if (failed(sharedRows) || failed(localRows))
    return failure();
  const int64_t blocksPerShared = ceilDivPositive(*localRows, 4);
  ptx.stream << "  mov.u32 %r" << thread << ", %tid.x;\n"
             << "  and.b32 %r" << lane << ", %r" << thread << ", 31;\n"
             << "  shr.u32 %r" << warp << ", %r" << thread << ", 5;\n"
             << "  mov.u32 %r" << block << ", %ctaid.x;\n"
             << "  div.u32 %r" << sharedRow << ", %r" << block << ", "
             << blocksPerShared << ";\n  rem.u32 %r" << localBlock << ", %r"
             << block << ", " << blocksPerShared << ";\n  mad.lo.u32 %r"
             << localRow << ", %r" << localBlock << ", 4, %r" << warp
             << ";\n  mad.lo.u32 %r" << row << ", %r" << sharedRow << ", "
             << *localRows << ", %r" << localRow << ";\n  setp.ge.u32 %p"
             << rowOutOfBounds << ", %r" << localRow << ", " << *localRows
             << ";\n";

  int64_t sharedBase = ptx.nextU64();
  ptx.stream << "  mov.u64 %rd" << sharedBase
             << ", shuttle_smem;\n  cvta.to.shared.u64 %rd" << sharedBase
             << ", %rd" << sharedBase << ";\n";
  int64_t reductionLimit = ptx.nextU32();
  ptx.stream << "  mov.u32 %r" << reductionLimit << ", "
             << domain[reductionAxis] << ";\n";
  if (causalQueryAxis) {
    int64_t blockFirstRow = ptx.nextU32();
    int64_t queryMaximum = ptx.nextU32();
    ptx.stream << "  mul.lo.u32 %r" << blockFirstRow << ", %r" << localBlock
               << ", 4;\n  rem.u32 %r" << queryMaximum << ", %r"
               << blockFirstRow << ", " << domain[*causalQueryAxis] << ";\n";
    for (int64_t warpIndex = 1; warpIndex < 4; ++warpIndex) {
      int64_t candidateRow = ptx.nextU32();
      int64_t candidateQuery = ptx.nextU32();
      int64_t nextMaximum = ptx.nextU32();
      int64_t greater = ptx.nextPred();
      ptx.stream << "  add.u32 %r" << candidateRow << ", %r" << blockFirstRow
                 << ", " << warpIndex << ";\n  rem.u32 %r" << candidateQuery
                 << ", %r" << candidateRow << ", " << domain[*causalQueryAxis]
                 << ";\n  setp.gt.u32 %p" << greater << ", %r" << candidateQuery
                 << ", %r" << queryMaximum << ";\n  selp.u32 %r" << nextMaximum
                 << ", %r" << candidateQuery << ", %r" << queryMaximum << ", %p"
                 << greater << ";\n  mov.u32 %r" << queryMaximum << ", %r"
                 << nextMaximum << ";\n";
    }
    ptx.stream << "  add.u32 %r" << reductionLimit << ", %r" << queryMaximum
               << ", 1;\n";
  }

  // Q is invariant across the streamed key tiles. Each lane retains the
  // feature components it owns for the lifetime of the row.
  const int64_t scoreComponents = ceilDivPositive(scoreElements, 32);
  SmallVector<int64_t> queryValues;
  (*contractCoordinates)[reductionAxis] = contractCoordinates->front();
  FailureOr<SmallVector<int64_t>> queryScoreCoordinates =
      emitOnlineMappedU32(ptx, scoreInputMap, *contractCoordinates);
  if (failed(queryScoreCoordinates))
    return failure();
  for (int64_t component = 0; component < scoreComponents; ++component) {
    int64_t coordinate = ptx.nextU32();
    int64_t valid = ptx.nextPred();
    int64_t value = ptx.nextF32();
    const int64_t label = context->label++;
    ptx.stream << "  add.u32 %r" << coordinate << ", %r" << lane << ", "
               << component * 32 << ";\n  mov.f32 %f" << value
               << ", 0f00000000;\n  setp.ge.u32 %p" << valid << ", %r"
               << coordinate << ", " << scoreElements << ";\n  @%p" << valid
               << " bra $online_query_" << label << "_done;\n";
    FailureOr<SmallVector<int64_t>> scoreDomain = onlineScoreDomainCoordinates(
        ptx, *scoreGeometry, *queryScoreCoordinates, coordinate);
    FailureOr<SmallVector<int64_t>> inputCoordinates =
        failed(scoreDomain)
            ? FailureOr<SmallVector<int64_t>>(failure())
            : emitOnlineMappedU32(ptx, scoreGeometry->queryMap, *scoreDomain);
    FailureOr<OnlineScalarValue> loaded =
        failed(inputCoordinates)
            ? FailureOr<OnlineScalarValue>(failure())
            : emitOnlineValue(
                  *context,
                  match.scoreContract.getInputs()[scoreGeometry->queryOperand],
                  *inputCoordinates);
    if (failed(loaded) || loaded->kind != OnlineScalarKind::F32)
      return failure();
    ptx.stream << "  mov.f32 %f" << value << ", %f" << loaded->reg
               << ";\n$online_query_" << label << "_done:\n";
    queryValues.push_back(value);
  }

  int64_t maxState = ptx.nextF32();
  int64_t sumState = ptx.nextF32();
  ptx.stream << "  mov.f32 %f" << maxState << ", 0fFF800000;\n  mov.f32 %f"
             << sumState << ", 0f00000000;\n";
  const int64_t vectorComponents = ceilDivPositive(domain[vectorAxis], 32);
  SmallVector<int64_t> outputState;
  for (int64_t component = 0; component < vectorComponents; ++component) {
    int64_t state = ptx.nextF32();
    ptx.stream << "  mov.f32 %f" << state << ", 0f00000000;\n";
    outputState.push_back(state);
  }

  int64_t tileStart = ptx.nextU32();
  int64_t tileDone = ptx.nextPred();
  const int64_t tileLabel = context->label++;
  ptx.stream << "  mov.u32 %r" << tileStart << ", 0;\n$online_tile_"
             << tileLabel << "_loop:\n  setp.ge.u32 %p" << tileDone << ", %r"
             << tileStart << ", %r" << reductionLimit << ";\n  @%p" << tileDone
             << " bra $online_tile_" << tileLabel << "_done;\n";
  int64_t remaining = ptx.nextU32();
  int64_t tileCount = ptx.nextU32();
  int64_t shortTile = ptx.nextPred();
  int64_t keyElements = ptx.nextU32();
  int64_t stageElements = ptx.nextU32();
  ptx.stream << "  sub.u32 %r" << remaining << ", %r" << reductionLimit
             << ", %r" << tileStart << ";\n  setp.lt.u32 %p" << shortTile
             << ", %r" << remaining << ", " << tileElements
             << ";\n  selp.u32 %r" << tileCount << ", %r" << remaining << ", "
             << tileElements << ", %p" << shortTile << ";\n  mul.lo.u32 %r"
             << keyElements << ", %r" << tileCount << ", " << scoreElements
             << ";\n  mul.lo.u32 %r" << stageElements << ", %r" << tileCount
             << ", " << scoreElements + domain[vectorAxis] << ";\n";

  int64_t stage = ptx.nextU32();
  int64_t stageDone = ptx.nextPred();
  int64_t stageIsValue = ptx.nextPred();
  const int64_t stageLabel = context->label++;
  const int64_t valueLabel = context->label++;
  const int64_t stageNextLabel = context->label++;
  ptx.stream << "  mov.u32 %r" << stage << ", %r" << thread
             << ";\n$online_stage_" << stageLabel << "_loop:\n  setp.ge.u32 %p"
             << stageDone << ", %r" << stage << ", %r" << stageElements
             << ";\n  @%p" << stageDone << " bra $online_stage_" << stageLabel
             << "_done;\n  setp.ge.u32 %p" << stageIsValue << ", %r" << stage
             << ", %r" << keyElements << ";\n  @%p" << stageIsValue
             << " bra $online_stage_" << valueLabel << "_value;\n";

  int64_t keyOffset = ptx.nextU32();
  int64_t scoreCoordinate = ptx.nextU32();
  int64_t stagedKey = ptx.nextU32();
  ptx.stream << "  div.u32 %r" << keyOffset << ", %r" << stage << ", "
             << scoreElements << ";\n  rem.u32 %r" << scoreCoordinate << ", %r"
             << stage << ", " << scoreElements << ";\n  add.u32 %r" << stagedKey
             << ", %r" << tileStart << ", %r" << keyOffset << ";\n";
  (*contractCoordinates)[reductionAxis] = stagedKey;
  FailureOr<SmallVector<int64_t>> stagedScoreCoordinates =
      emitOnlineMappedU32(ptx, scoreInputMap, *contractCoordinates);
  FailureOr<SmallVector<int64_t>> stagedScoreDomain =
      failed(stagedScoreCoordinates)
          ? FailureOr<SmallVector<int64_t>>(failure())
          : onlineScoreDomainCoordinates(
                ptx, *scoreGeometry, *stagedScoreCoordinates, scoreCoordinate);
  FailureOr<SmallVector<int64_t>> stagedKeyCoordinates =
      failed(stagedScoreDomain)
          ? FailureOr<SmallVector<int64_t>>(failure())
          : emitOnlineMappedU32(ptx, scoreGeometry->keyMap, *stagedScoreDomain);
  FailureOr<OnlineScalarValue> stagedKeyValue =
      failed(stagedKeyCoordinates)
          ? FailureOr<OnlineScalarValue>(failure())
          : emitOnlineValue(
                *context,
                match.scoreContract.getInputs()[scoreGeometry->keyOperand],
                *stagedKeyCoordinates);
  if (failed(stagedKeyValue) || stagedKeyValue->kind != OnlineScalarKind::F32)
    return failure();
  emitOnlineSharedBf16Store(ptx, sharedBase, stage, stagedKeyValue->reg);
  ptx.stream << "  bra $online_stage_" << stageNextLabel
             << "_next;\n$online_stage_" << valueLabel << "_value:\n";

  int64_t valueFlat = ptx.nextU32();
  int64_t valueKeyOffset = ptx.nextU32();
  int64_t valueCoordinate = ptx.nextU32();
  int64_t stagedValueKey = ptx.nextU32();
  ptx.stream << "  sub.u32 %r" << valueFlat << ", %r" << stage << ", %r"
             << keyElements << ";\n  div.u32 %r" << valueKeyOffset << ", %r"
             << valueFlat << ", " << domain[vectorAxis] << ";\n  rem.u32 %r"
             << valueCoordinate << ", %r" << valueFlat << ", "
             << domain[vectorAxis] << ";\n  add.u32 %r" << stagedValueKey
             << ", %r" << tileStart << ", %r" << valueKeyOffset << ";\n";
  (*contractCoordinates)[reductionAxis] = stagedValueKey;
  (*contractCoordinates)[vectorAxis] = valueCoordinate;
  FailureOr<SmallVector<int64_t>> stagedValueCoordinates =
      emitOnlineMappedU32(ptx, valueMap.getValue(), *contractCoordinates);
  FailureOr<OnlineScalarValue> stagedValue =
      failed(stagedValueCoordinates)
          ? FailureOr<OnlineScalarValue>(failure())
          : emitOnlineValue(*context,
                            match.weightedContract.getInputs()[valueOperand],
                            *stagedValueCoordinates);
  if (failed(stagedValue) || stagedValue->kind != OnlineScalarKind::F32)
    return failure();
  emitOnlineSharedBf16Store(ptx, sharedBase, stage, stagedValue->reg);
  ptx.stream << "$online_stage_" << stageNextLabel << "_next:\n  add.u32 %r"
             << stage << ", %r" << stage << ", " << task.getWorkgroupThreads()
             << ";\n  bra $online_stage_" << stageLabel
             << "_loop;\n$online_stage_" << stageLabel
             << "_done:\n  bar.sync 0;\n";

  int64_t key = ptx.nextU32();
  int64_t tileEnd = ptx.nextU32();
  int64_t keyDone = ptx.nextPred();
  const int64_t loopLabel = context->label++;
  ptx.stream << "  mov.u32 %r" << key << ", %r" << tileStart
             << ";\n  add.u32 %r" << tileEnd << ", %r" << tileStart << ", %r"
             << tileCount << ";\n$online_vector_" << loopLabel
             << "_loop:\n  setp.ge.u32 %p" << keyDone << ", %r" << key << ", %r"
             << tileEnd << ";\n  @%p" << keyDone << " bra $online_vector_"
             << loopLabel << "_done;\n";
  (*contractCoordinates)[reductionAxis] = key;
  FailureOr<SmallVector<int64_t>> scoreCoordinates =
      emitOnlineMappedU32(ptx, normalizedMap.getValue(), *contractCoordinates);
  if (failed(scoreCoordinates))
    return failure();
  int64_t activeKeyOffset = ptx.nextU32();
  ptx.stream << "  sub.u32 %r" << activeKeyOffset << ", %r" << key << ", %r"
             << tileStart << ";\n";
  int64_t rawScore = ptx.nextF32();
  ptx.stream << "  mov.f32 %f" << rawScore << ", 0f00000000;\n";
  for (int64_t component = 0; component < scoreComponents; ++component) {
    int64_t feature = ptx.nextU32();
    int64_t valid = ptx.nextPred();
    int64_t sharedElement = ptx.nextU32();
    int64_t term = ptx.nextF32();
    const int64_t label = context->label++;
    ptx.stream << "  add.u32 %r" << feature << ", %r" << lane << ", "
               << component * 32 << ";\n  setp.ge.u32 %p" << valid << ", %r"
               << feature << ", " << scoreElements << ";\n  @%p" << valid
               << " bra $online_score_" << label << "_done;\n  mad.lo.u32 %r"
               << sharedElement << ", %r" << activeKeyOffset << ", "
               << scoreElements << ", %r" << feature << ";\n";
    int64_t keyValue = emitOnlineSharedBf16Load(ptx, sharedBase, sharedElement);
    ptx.stream << "  fma.rn.f32 %f" << term << ", %f" << queryValues[component]
               << ", %f" << keyValue << ", %f" << rawScore << ";\n  mov.f32 %f"
               << rawScore << ", %f" << term << ";\n$online_score_" << label
               << "_done:\n";
  }
  for (int offset : {16, 8, 4, 2, 1}) {
    int64_t shuffled = ptx.nextF32();
    int64_t reduced = ptx.nextF32();
    ptx.stream << "  shfl.sync.down.b32 %f" << shuffled << ", %f" << rawScore
               << ", " << offset << ", 0x1f, 0xffffffff;\n  add.rn.f32 %f"
               << reduced << ", %f" << rawScore << ", %f" << shuffled << ";\n";
    rawScore = reduced;
  }
  int64_t broadcastScore = ptx.nextF32();
  ptx.stream << "  shfl.sync.idx.b32 %f" << broadcastScore << ", %f" << rawScore
             << ", 0, 0x1f, 0xffffffff;\n";
  context->valueOverrides[match.scoreContract.getResult(0)] = {
      OnlineScalarKind::F32, broadcastScore};
  FailureOr<OnlineScalarValue> score =
      emitOnlineValue(*context, match.score, *scoreCoordinates);
  context->valueOverrides.erase(match.scoreContract.getResult(0));
  auto select = match.score.getDefiningOp<RelationOp>();
  if (failed(score) || score->kind != OnlineScalarKind::F32 || !select ||
      select.getSemantics() != RelationSemantics::Select)
    return failure();
  auto predicate = context->relationValues.find(select.getInputs().front());
  if (predicate == context->relationValues.end() ||
      predicate->second.kind != OnlineScalarKind::Predicate)
    return failure();
  const int64_t nextLabel = context->label++;
  ptx.stream << "  @!%p" << predicate->second.reg << " bra $online_vector_"
             << nextLabel << "_next;\n";
  int64_t empty = ptx.nextPred();
  const int64_t firstLabel = context->label++;
  const int64_t updatedLabel = context->label++;
  ptx.stream << "  setp.eq.f32 %p" << empty << ", %f" << sumState
             << ", 0f00000000;\n  @%p" << empty << " bra $online_vector_"
             << firstLabel << "_first;\n";

  int64_t nextMaximum = ptx.nextF32();
  int64_t oldDelta = ptx.nextF32();
  int64_t scoreDelta = ptx.nextF32();
  ptx.stream << "  max.f32 %f" << nextMaximum << ", %f" << maxState << ", %f"
             << score->reg << ";\n  sub.rn.f32 %f" << oldDelta << ", %f"
             << maxState << ", %f" << nextMaximum << ";\n  sub.rn.f32 %f"
             << scoreDelta << ", %f" << score->reg << ", %f" << nextMaximum
             << ";\n";
  int64_t oldScale = emitOnlineExp(ptx, oldDelta);
  int64_t scoreScale = emitOnlineExp(ptx, scoreDelta);
  int64_t nextSum = ptx.nextF32();
  ptx.stream << "  fma.rn.f32 %f" << nextSum << ", %f" << oldScale << ", %f"
             << sumState << ", %f" << scoreScale << ";\n";
  SmallVector<int64_t> nextOutput;
  for (int64_t component = 0; component < vectorComponents; ++component) {
    int64_t vectorCoordinate = ptx.nextU32();
    int64_t vectorValid = ptx.nextPred();
    int64_t nextState = ptx.nextF32();
    const int64_t componentLabel = context->label++;
    ptx.stream << "  add.u32 %r" << vectorCoordinate << ", %r" << lane << ", "
               << component * 32 << ";\n  setp.lt.u32 %p" << vectorValid
               << ", %r" << vectorCoordinate << ", " << domain[vectorAxis]
               << ";\n  mov.f32 %f" << nextState << ", %f"
               << outputState[component] << ";\n  @!%p" << vectorValid
               << " bra $online_vector_" << componentLabel << "_skip;\n";
    int64_t valueElement = ptx.nextU32();
    ptx.stream << "  mad.lo.u32 %r" << valueElement << ", %r" << activeKeyOffset
               << ", " << domain[vectorAxis] << ", %r" << vectorCoordinate
               << ";\n  add.u32 %r" << valueElement << ", %r" << valueElement
               << ", %r" << keyElements << ";\n";
    int64_t value = emitOnlineSharedBf16Load(ptx, sharedBase, valueElement);
    int64_t weighted = ptx.nextF32();
    int64_t next = ptx.nextF32();
    ptx.stream << "  mul.rn.f32 %f" << weighted << ", %f" << scoreScale
               << ", %f" << value << ";\n  fma.rn.f32 %f" << next << ", %f"
               << oldScale << ", %f" << outputState[component] << ", %f"
               << weighted << ";\n  mov.f32 %f" << nextState << ", %f" << next
               << ";\n$online_vector_" << componentLabel << "_skip:\n";
    nextOutput.push_back(nextState);
  }
  ptx.stream << "  mov.f32 %f" << maxState << ", %f" << nextMaximum
             << ";\n  mov.f32 %f" << sumState << ", %f" << nextSum << ";\n";
  for (auto [state, next] : llvm::zip(outputState, nextOutput))
    ptx.stream << "  mov.f32 %f" << state << ", %f" << next << ";\n";
  ptx.stream << "  bra $online_vector_" << updatedLabel
             << "_updated;\n$online_vector_" << firstLabel << "_first:\n";

  ptx.stream << "  mov.f32 %f" << maxState << ", %f" << score->reg
             << ";\n  mov.f32 %f" << sumState << ", 0f3F800000;\n";
  for (int64_t component = 0; component < vectorComponents; ++component) {
    int64_t vectorCoordinate = ptx.nextU32();
    int64_t vectorValid = ptx.nextPred();
    const int64_t componentLabel = context->label++;
    ptx.stream << "  add.u32 %r" << vectorCoordinate << ", %r" << lane << ", "
               << component * 32 << ";\n  setp.lt.u32 %p" << vectorValid
               << ", %r" << vectorCoordinate << ", " << domain[vectorAxis]
               << ";\n  @!%p" << vectorValid << " bra $online_vector_"
               << componentLabel << "_skip;\n";
    int64_t valueElement = ptx.nextU32();
    ptx.stream << "  mad.lo.u32 %r" << valueElement << ", %r" << activeKeyOffset
               << ", " << domain[vectorAxis] << ", %r" << vectorCoordinate
               << ";\n  add.u32 %r" << valueElement << ", %r" << valueElement
               << ", %r" << keyElements << ";\n";
    int64_t value = emitOnlineSharedBf16Load(ptx, sharedBase, valueElement);
    ptx.stream << "  mov.f32 %f" << outputState[component] << ", %f" << value
               << ";\n$online_vector_" << componentLabel << "_skip:\n";
  }
  ptx.stream << "$online_vector_" << updatedLabel
             << "_updated:\n$online_vector_" << nextLabel
             << "_next:\n  add.u32 %r" << key << ", %r" << key
             << ", 1;\n  bra $online_vector_" << loopLabel
             << "_loop;\n$online_vector_" << loopLabel
             << "_done:\n  bar.sync 0;\n  add.u32 %r" << tileStart << ", %r"
             << tileStart << ", " << tileElements << ";\n  bra $online_tile_"
             << tileLabel << "_loop;\n$online_tile_" << tileLabel << "_done:\n";

  auto regionYield = cast<YieldOp>(region.getBody().front().getTerminator());
  auto publicType =
      dyn_cast<RankedTensorType>(regionYield.getValues().front().getType());
  if (!publicType || (!publicType.getElementType().isF32() &&
                      !publicType.getElementType().isBF16()))
    return failure();
  ptx.stream << "  @%p" << rowOutOfBounds << " bra $done;\n";
  for (int64_t component = 0; component < vectorComponents; ++component) {
    int64_t vectorCoordinate = ptx.nextU32();
    int64_t vectorValid = ptx.nextPred();
    const int64_t componentLabel = context->label++;
    ptx.stream << "  add.u32 %r" << vectorCoordinate << ", %r" << lane << ", "
               << component * 32 << ";\n  setp.lt.u32 %p" << vectorValid
               << ", %r" << vectorCoordinate << ", " << domain[vectorAxis]
               << ";\n  @!%p" << vectorValid << " bra $online_vector_"
               << componentLabel << "_skip;\n";
    (*contractCoordinates)[vectorAxis] = vectorCoordinate;
    FailureOr<SmallVector<int64_t>> weightedCoordinates = emitOnlineMappedU32(
        ptx, weightedOutputMap.getValue(), *contractCoordinates);
    FailureOr<SmallVector<int64_t>> publicCoordinates =
        failed(weightedCoordinates)
            ? FailureOr<SmallVector<int64_t>>(failure())
            : onlineOutputCoordinates(match.weightedContract.getResult(0),
                                      regionYield.getValues().front(),
                                      *weightedCoordinates);
    if (failed(publicCoordinates))
      return failure();
    int64_t normalized = ptx.nextF32();
    ptx.stream << "  div.rn.f32 %f" << normalized << ", %f"
               << outputState[component] << ", %f" << sumState << ";\n";
    FailureOr<OnlineScalarValue> epilogue = emitOnlineEpilogue(
        *context, regionYield.getValues().front(),
        match.weightedContract.getResult(0),
        {OnlineScalarKind::F32, normalized}, *publicCoordinates);
    FailureOr<int64_t> outputPointer =
        failed(epilogue)
            ? FailureOr<int64_t>(failure())
            : emitOnlineGlobalStorePointer(
                  ptx, materializationTask.getInputBuffers().size(), publicType,
                  *publicCoordinates);
    if (failed(epilogue) || failed(outputPointer))
      return failure();
    if (publicType.getElementType().isBF16()) {
      int64_t rounded = ptx.nextBF16();
      ptx.stream << "  cvt.rn.bf16.f32 %h" << rounded << ", %f" << epilogue->reg
                 << ";\n  st.global.u16 [%rd" << *outputPointer << "], %h"
                 << rounded << ";\n";
    } else {
      ptx.stream << "  st.global.f32 [%rd" << *outputPointer << "], %f"
                 << epilogue->reg << ";\n";
    }
    ptx.stream << "$online_vector_" << componentLabel << "_skip:\n";
  }
  ptx.finish();
  return ptx.text;
}

#include "OnlineAttentionMmaEmitter.inc"

FailureOr<std::string> emitOnlineSufficientStatePtx(
    RegionOp region, MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> &buffers, ScheduleTaskOp task) {
  if (!isOnlineStateTask(materializationTask) ||
      materializationTask.getCompositionKind() != kOnlineStateProtocol ||
      task.getOnlineStateProtocol() != kOnlineStateProtocol ||
      task.getReductionOrder() != ScheduleReductionOrder::LeafOrderFree)
    return failure();
  FailureOr<OnlineEmissionMatch> match = matchOnlineEmission(region);
  if (failed(match))
    return failure();
  if (materializationTask.getOnlineMma().value_or(false))
    return emitOnlineMmaVectorStatePtx(region, materializationTask, task,
                                       *match);
  return materializationTask.getKind() == MaterializationTaskKind::Fold
             ? emitOnlineMaximumPtx(region, materializationTask, buffers, task,
                                    *match)
             : emitOnlineVectorStatePtx(region, materializationTask, buffers,
                                        task, *match);
}

FailureOr<OnlineScalarValue> emitOnlineValue(OnlineEvalContext &context,
                                             Value value,
                                             ArrayRef<int64_t> coordinates) {
  if (auto override = context.valueOverrides.find(value);
      override != context.valueOverrides.end())
    return override->second;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto parameter = context.argumentParameters.find(argument.getArgNumber());
    auto type = dyn_cast<RankedTensorType>(argument.getType());
    if (parameter == context.argumentParameters.end() || !type)
      return failure();
    FailureOr<int64_t> loaded =
        emitOnlineTensorLoad(context.ptx, parameter->second, type, coordinates);
    if (failed(loaded))
      return failure();
    return OnlineScalarValue{OnlineScalarKind::F32, *loaded};
  }
  if (value == context.match.maximumFold.getResult(0) &&
      context.maximumParameter) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      return failure();
    FailureOr<int64_t> loaded = emitOnlineTensorLoad(
        context.ptx, *context.maximumParameter, type, coordinates);
    if (failed(loaded))
      return failure();
    return OnlineScalarValue{OnlineScalarKind::F32, *loaded};
  }
  if (auto relation = value.getDefiningOp<RelationOp>())
    return emitOnlineRelationValue(context, relation, coordinates);
  if (auto map = value.getDefiningOp<MapOp>()) {
    if (map.getNumResults() != 1 ||
        map.getIndexingMaps().size() != map.getInputs().size() + 1)
      return failure();
    SmallVector<int64_t> arguments;
    for (auto [operand, input] : llvm::enumerate(map.getInputs())) {
      auto indexing = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[operand]);
      if (!indexing)
        return failure();
      FailureOr<SmallVector<int64_t>> inputCoordinates =
          emitOnlineMappedU32(context.ptx, indexing.getValue(), coordinates);
      if (failed(inputCoordinates))
        return failure();
      FailureOr<OnlineScalarValue> scalar =
          emitOnlineValue(context, input, *inputCoordinates);
      if (failed(scalar) || scalar->kind != OnlineScalarKind::F32)
        return failure();
      arguments.push_back(scalar->reg);
    }
    FailureOr<int64_t> result =
        emitScalarBody(context.ptx, map.getBody().front(), arguments);
    if (failed(result))
      return failure();
    return OnlineScalarValue{OnlineScalarKind::F32, *result};
  }
  if (auto contract = value.getDefiningOp<ContractOp>())
    return emitOnlineContractValue(context, contract, coordinates);
  return failure();
}

FailureOr<SmallVector<AffineMap>>
onlineLogicalCoordinateMaps(ScheduleTaskOp task) {
  if (!task.getOnlineStateProtocolAttr() || !task.getReductionAxis() ||
      !task.getOnlineVectorDimensions())
    return failure();
  ArrayRef<int64_t> domain = task.getDomainShape();
  const int64_t reductionAxis = *task.getReductionAxis();
  llvm::SmallDenseSet<int64_t> vectorAxes(
      task.getOnlineVectorDimensions()->begin(),
      task.getOnlineVectorDimensions()->end());
  if (reductionAxis < 0 || reductionAxis >= static_cast<int64_t>(domain.size()))
    return failure();
  MLIRContext *context = task.getContext();
  AffineExpr block = getAffineDimExpr(0, context);
  AffineExpr thread = getAffineDimExpr(3, context);
  SmallVector<int64_t> rowAxes;
  for (int64_t axis = 0; axis < static_cast<int64_t>(domain.size()); ++axis)
    if (axis != reductionAxis && !vectorAxes.contains(axis))
      rowAxes.push_back(axis);
  if (rowAxes.size() < 2)
    return failure();
  const int64_t sharedAxisCount = rowAxes.size() - 2;
  int64_t localRows = 1;
  for (int64_t axis : ArrayRef<int64_t>(rowAxes).drop_front(sharedAxisCount)) {
    if (domain[axis] <= 0 ||
        localRows > std::numeric_limits<int64_t>::max() / domain[axis])
      return failure();
    localRows *= domain[axis];
  }
  const int64_t rowsPerBlock = task.getOnlineMma().value_or(false) ? 128 : 4;
  const int64_t blocksPerShared = ceilDivPositive(localRows, rowsPerBlock);
  AffineExpr sharedRow = block.floorDiv(blocksPerShared);
  AffineExpr localRow =
      (block % blocksPerShared) * rowsPerBlock +
      (task.getOnlineMma().value_or(false) ? thread : thread.floorDiv(32));
  SmallVector<int64_t> sharedStrides(domain.size(), 1);
  SmallVector<int64_t> localStrides(domain.size(), 1);
  int64_t stride = 1;
  for (int64_t position = sharedAxisCount; position > 0; --position) {
    int64_t axis = rowAxes[position - 1];
    sharedStrides[axis] = stride;
    if (stride > std::numeric_limits<int64_t>::max() / domain[axis])
      return failure();
    stride *= domain[axis];
  }
  stride = 1;
  for (int64_t position = rowAxes.size(); position > sharedAxisCount;
       --position) {
    int64_t axis = rowAxes[position - 1];
    localStrides[axis] = stride;
    if (stride > std::numeric_limits<int64_t>::max() / domain[axis])
      return failure();
    stride *= domain[axis];
  }
  SmallVector<AffineMap> maps;
  for (auto [axis, extent] : llvm::enumerate(domain)) {
    AffineExpr coordinate = getAffineConstantExpr(0, context);
    if (static_cast<int64_t>(axis) != reductionAxis &&
        !vectorAxes.contains(axis)) {
      auto position = llvm::find(rowAxes, static_cast<int64_t>(axis));
      if (position == rowAxes.end())
        return failure();
      const int64_t ordinal = std::distance(rowAxes.begin(), position);
      AffineExpr row = ordinal < sharedAxisCount ? sharedRow : localRow;
      int64_t rowStride =
          ordinal < sharedAxisCount ? sharedStrides[axis] : localStrides[axis];
      coordinate = row.floorDiv(rowStride) % extent;
    }
    maps.push_back(AffineMap::get(7, 0, coordinate, context));
  }
  return maps;
}
