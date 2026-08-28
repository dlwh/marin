// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

constexpr int64_t onlineMmaAccumulatorRowOffset(int64_t fragment) {
  return fragment < 2 ? 0 : 8;
}

constexpr int64_t onlineMmaAccumulatorColumnOffset(int64_t fragment) {
  return fragment & 1;
}

constexpr int64_t onlineMmaAFragmentRowOffset(int64_t fragment) {
  return (fragment & 1) == 0 ? 0 : 8;
}

constexpr int64_t onlineMmaAFragmentColumnOffset(int64_t fragment) {
  return fragment < 2 ? 0 : 8;
}

constexpr int64_t onlineMmaSwizzledColumn(int64_t row, int64_t column) {
  return column ^ ((row & 7) << 3);
}

constexpr bool onlineMmaSwizzleHasExactAtomCoverage() {
  for (int64_t row = 0; row < 8; ++row) {
    bool covered[64] = {};
    for (int64_t column = 0; column < 64; ++column) {
      const int64_t physical = onlineMmaSwizzledColumn(row, column);
      if (physical < 0 || physical >= 64 || covered[physical])
        return false;
      covered[physical] = true;
    }
    for (bool element : covered)
      if (!element)
        return false;
  }
  return true;
}

// ldmatrix.x2 returns one packed pair from each of two 8x8 matrices.  For a
// row-major K tile, the address-providing lane selects the key row and matrix
// number selects the low/high feature octet.  This is exactly the B-fragment
// layout of mma.m16n8k16: feature varies within the packed pair and key is the
// fragment column.
constexpr int64_t onlineMmaKAddressProvider(int64_t lane, int64_t pair) {
  return pair * 8 + lane / 4;
}

constexpr int64_t onlineMmaKLoadedKey(int64_t lane, int64_t pair) {
  return onlineMmaKAddressProvider(lane, pair) & 7;
}

constexpr int64_t onlineMmaKLoadedFeature(int64_t lane, int64_t pair,
                                          int64_t element) {
  const int64_t provider = onlineMmaKAddressProvider(lane, pair);
  return (provider / 8) * 8 + (lane % 4) * 2 + element;
}

constexpr bool onlineMmaKLoadMatchesBFragment() {
  for (int64_t lane = 0; lane < 32; ++lane)
    for (int64_t pair = 0; pair < 2; ++pair)
      for (int64_t element = 0; element < 2; ++element) {
        const int64_t bElement = pair * 2 + element;
        const int64_t expectedFeature =
            (lane % 4) * 2 + (bElement & 1) + (bElement >= 2 ? 8 : 0);
        const int64_t expectedKey = lane / 4;
        if (onlineMmaKLoadedFeature(lane, pair, element) != expectedFeature ||
            onlineMmaKLoadedKey(lane, pair) != expectedKey)
          return false;
      }
  return true;
}

constexpr int64_t onlineMmaProbabilityStoreRow(int64_t lane, int64_t rowHalf) {
  return lane / 4 + rowHalf * 8;
}

constexpr int64_t onlineMmaProbabilityStoreColumn(int64_t lane,
                                                  int64_t keyChunk,
                                                  int64_t element) {
  return (lane % 4) * 2 + keyChunk * 8 + element;
}

constexpr int64_t onlineMmaProbabilityLoadRow(int64_t lane, int64_t fragment) {
  return lane / 4 + onlineMmaAFragmentRowOffset(fragment);
}

constexpr int64_t onlineMmaProbabilityLoadColumn(int64_t lane, int64_t fragment,
                                                 int64_t element) {
  return (lane % 4) * 2 + onlineMmaAFragmentColumnOffset(fragment) + element;
}

constexpr bool onlineMmaAccumulatorGroupHasExactCoverage() {
  bool covered[2][8] = {};
  for (int64_t thread = 0; thread < 4; ++thread) {
    for (int64_t fragment = 0; fragment < 4; ++fragment) {
      const int64_t row = onlineMmaAccumulatorRowOffset(fragment) / 8;
      const int64_t column =
          thread * 2 + onlineMmaAccumulatorColumnOffset(fragment);
      if (covered[row][column])
        return false;
      covered[row][column] = true;
    }
  }
  for (const auto &row : covered)
    for (bool element : row)
      if (!element)
        return false;
  return true;
}

constexpr bool onlineMmaProbabilityStoreAndLoadHaveExactCoverage() {
  bool stored[16][16] = {};
  bool loaded[16][16] = {};
  for (int64_t lane = 0; lane < 32; ++lane) {
    for (int64_t rowHalf = 0; rowHalf < 2; ++rowHalf)
      for (int64_t keyChunk = 0; keyChunk < 2; ++keyChunk)
        for (int64_t element = 0; element < 2; ++element) {
          const int64_t row = onlineMmaProbabilityStoreRow(lane, rowHalf);
          const int64_t column =
              onlineMmaProbabilityStoreColumn(lane, keyChunk, element);
          if (stored[row][column])
            return false;
          stored[row][column] = true;
        }
    for (int64_t fragment = 0; fragment < 4; ++fragment)
      for (int64_t element = 0; element < 2; ++element) {
        const int64_t row = onlineMmaProbabilityLoadRow(lane, fragment);
        const int64_t column =
            onlineMmaProbabilityLoadColumn(lane, fragment, element);
        if (loaded[row][column])
          return false;
        loaded[row][column] = true;
      }
  }
  for (int64_t row = 0; row < 16; ++row)
    for (int64_t column = 0; column < 16; ++column)
      if (!stored[row][column] || !loaded[row][column] ||
          stored[row][column] != loaded[row][column])
        return false;
  return true;
}

struct OnlineMmaCausalAxes {
  int64_t query;
  int64_t key;
};

struct OnlineMmaBf16Source {
  int64_t parameter;
  RankedTensorType type;
  AffineMap coordinateMap;
};

FailureOr<OnlineMmaBf16Source> onlineMmaBf16Source(OnlineEvalContext &context,
                                                   Value value) {
  auto conversion = value.getDefiningOp<MapOp>();
  if (!conversion || conversion.getSemantics() != MapSemantics::Pointwise ||
      conversion.getInputs().size() != 1 || conversion.getNumResults() != 1 ||
      conversion.getIndexingMaps().size() != 2 || conversion.getBody().empty())
    return failure();
  auto argument = dyn_cast<BlockArgument>(conversion.getInputs().front());
  auto inputType =
      dyn_cast<RankedTensorType>(conversion.getInputs().front().getType());
  auto resultType = dyn_cast<RankedTensorType>(value.getType());
  auto inputMap = dyn_cast<AffineMapAttr>(conversion.getIndexingMaps()[0]);
  auto resultMap = dyn_cast<AffineMapAttr>(conversion.getIndexingMaps()[1]);
  if (!argument || !inputType || !resultType || !inputMap || !resultMap ||
      !inputType.hasStaticShape() ||
      inputType.getShape() != resultType.getShape() ||
      !inputType.getElementType().isBF16() ||
      !resultType.getElementType().isF32() ||
      !isIdentityMap(resultMap.getValue()))
    return failure();
  Block &body = conversion.getBody().front();
  auto scalarConvert = body.empty() ? ScalarConvertOp{}
                                    : dyn_cast<ScalarConvertOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (body.getNumArguments() != 1 || body.getOperations().size() != 2 ||
      !scalarConvert || !yield ||
      scalarConvert.getInput() != body.getArgument(0) ||
      scalarConvert.getSemantics() != ScalarConvertSemantics::Exact ||
      !scalarConvert.getInput().getType().isBF16() ||
      !scalarConvert.getResult().getType().isF32() ||
      yield.getValues().size() != 1 ||
      yield.getValues().front() != scalarConvert.getResult())
    return failure();
  auto parameter = context.argumentParameters.find(argument.getArgNumber());
  if (parameter == context.argumentParameters.end())
    return failure();
  return OnlineMmaBf16Source{parameter->second, inputType, inputMap.getValue()};
}

void emitOnlineMmaLoadGlobal128(PtxEmitter &ptx, int64_t pointer,
                                const std::array<int64_t, 4> &words) {
  ptx.stream << "  ld.global.v4.u32 {%r" << words[0] << ", %r" << words[1]
             << ", %r" << words[2] << ", %r" << words[3] << "}, [%rd" << pointer
             << "];\n";
}

int64_t emitOnlineMmaSwizzledElement(PtxEmitter &ptx, int64_t element,
                                     int64_t rowStride) {
  int64_t shift = 0;
  while ((int64_t{1} << shift) < rowStride)
    ++shift;
  if ((int64_t{1} << shift) != rowStride)
    return element;
  int64_t row = ptx.nextU32();
  int64_t mask = ptx.nextU32();
  int64_t swizzled = ptx.nextU32();
  ptx.stream << "  shr.u32 %r" << row << ", %r" << element << ", " << shift
             << ";\n  and.b32 %r" << mask << ", %r" << row
             << ", 7;\n  shl.b32 %r" << mask << ", %r" << mask
             << ", 3;\n  xor.b32 %r" << swizzled << ", %r" << element << ", %r"
             << mask << ";\n";
  return swizzled;
}

void emitOnlineMmaStoreShared128(
    PtxEmitter &ptx, int64_t sharedBase, int64_t element,
    const std::array<int64_t, 4> &words,
    std::optional<int64_t> swizzleStride = std::nullopt) {
  if (swizzleStride)
    element = emitOnlineMmaSwizzledElement(ptx, element, *swizzleStride);
  int64_t address = ptx.nextU32();
  ptx.stream << "  mad.lo.u32 %r" << address << ", %r" << element << ", 2, %r"
             << sharedBase << ";\n  st.shared.v4.u32 [%r" << address << "], {%r"
             << words[0] << ", %r" << words[1] << ", %r" << words[2] << ", %r"
             << words[3] << "};\n";
}

std::array<int64_t, 4> emitOnlineMmaLoadShared128(
    PtxEmitter &ptx, int64_t sharedBase, int64_t element,
    std::optional<int64_t> swizzleStride = std::nullopt) {
  if (swizzleStride)
    element = emitOnlineMmaSwizzledElement(ptx, element, *swizzleStride);
  int64_t address = ptx.nextU32();
  std::array<int64_t, 4> words;
  for (int64_t &word : words)
    word = ptx.nextU32();
  ptx.stream << "  mad.lo.u32 %r" << address << ", %r" << element << ", 2, %r"
             << sharedBase << ";\n  ld.shared.v4.u32 {%r" << words[0] << ", %r"
             << words[1] << ", %r" << words[2] << ", %r" << words[3] << "}, [%r"
             << address << "];\n";
  return words;
}

void emitOnlineMmaStoreGlobal128(PtxEmitter &ptx, int64_t pointer,
                                 const std::array<int64_t, 4> &words) {
  ptx.stream << "  st.global.v4.u32 [%rd" << pointer << "], {%r" << words[0]
             << ", %r" << words[1] << ", %r" << words[2] << ", %r" << words[3]
             << "};\n";
}

void emitOnlineMmaAsyncCopy128(
    PtxEmitter &ptx, int64_t sharedBase, int64_t element, int64_t pointer,
    std::optional<int64_t> predicate = std::nullopt,
    std::optional<int64_t> swizzleStride = std::nullopt) {
  if (swizzleStride)
    element = emitOnlineMmaSwizzledElement(ptx, element, *swizzleStride);
  int64_t address = ptx.nextU32();
  ptx.stream << "  mad.lo.u32 %r" << address << ", %r" << element << ", 2, %r"
             << sharedBase << ";\n  ";
  if (predicate)
    ptx.stream << "@%p" << *predicate << " ";
  ptx.stream << "cp.async.ca.shared.global [%r" << address << "], [%rd"
             << pointer << "], 16;\n";
}

std::optional<int64_t> onlineMmaRelationVaryingAxis(Value value) {
  auto relation = value.getDefiningOp<RelationOp>();
  if (!relation)
    return std::nullopt;
  auto resultType = dyn_cast<RankedTensorType>(value.getType());
  if (!resultType)
    return std::nullopt;
  if (relation.getSemantics() == RelationSemantics::Iota) {
    auto dimension = dyn_cast<IntegerAttr>(relation.getParameters()[0]);
    return dimension ? std::optional<int64_t>(dimension.getInt())
                     : std::nullopt;
  }
  if (relation.getSemantics() == RelationSemantics::BroadcastInDim) {
    std::optional<int64_t> inputAxis =
        onlineMmaRelationVaryingAxis(relation.getInputs().front());
    if (!inputAxis || *inputAxis < 0 ||
        *inputAxis >= static_cast<int64_t>(relation.getParameters().size()))
      return std::nullopt;
    auto resultAxis =
        dyn_cast<IntegerAttr>(relation.getParameters()[*inputAxis]);
    return resultAxis ? std::optional<int64_t>(resultAxis.getInt())
                      : std::nullopt;
  }
  if (relation.getSemantics() == RelationSemantics::Reshape) {
    auto inputType =
        dyn_cast<RankedTensorType>(relation.getInputs().front().getType());
    std::optional<int64_t> inputAxis =
        onlineMmaRelationVaryingAxis(relation.getInputs().front());
    if (!inputType || !inputAxis)
      return std::nullopt;
    const int64_t varyingExtent = inputType.getDimSize(*inputAxis);
    std::optional<int64_t> resultAxis;
    for (int64_t axis = 0; axis < resultType.getRank(); ++axis) {
      if (resultType.getDimSize(axis) == varyingExtent && varyingExtent != 1) {
        if (resultAxis)
          return std::nullopt;
        resultAxis = axis;
      } else if (resultType.getDimSize(axis) != 1) {
        return std::nullopt;
      }
    }
    return resultAxis;
  }
  return std::nullopt;
}

std::optional<OnlineMmaCausalAxes> onlineMmaCausalAxes(Value maskedScore) {
  auto select = maskedScore.getDefiningOp<RelationOp>();
  if (!select || select.getSemantics() != RelationSemantics::Select)
    return std::nullopt;
  Value predicate = select.getInputs().front();
  auto scoreType = dyn_cast<RankedTensorType>(maskedScore.getType());
  if (!scoreType)
    return std::nullopt;
  SmallVector<int64_t> resultAxes(scoreType.getRank());
  std::iota(resultAxes.begin(), resultAxes.end(), 0);
  while (auto relation = predicate.getDefiningOp<RelationOp>()) {
    if (relation.getSemantics() == RelationSemantics::Compare) {
      auto direction = dyn_cast<StringAttr>(relation.getParameters()[1]);
      if (!direction || direction.getValue() != "LE")
        return std::nullopt;
      std::optional<int64_t> lhs =
          onlineMmaRelationVaryingAxis(relation.getInputs()[0]);
      std::optional<int64_t> rhs =
          onlineMmaRelationVaryingAxis(relation.getInputs()[1]);
      if (!lhs || !rhs || *lhs < 0 || *rhs < 0 ||
          *lhs >= static_cast<int64_t>(resultAxes.size()) ||
          *rhs >= static_cast<int64_t>(resultAxes.size()) || *lhs == *rhs)
        return std::nullopt;
      return OnlineMmaCausalAxes{resultAxes[*rhs], resultAxes[*lhs]};
    }
    if (relation.getSemantics() != RelationSemantics::BroadcastInDim)
      return std::nullopt;
    SmallVector<int64_t> inputAxes;
    for (Attribute parameter : relation.getParameters()) {
      auto outputAxis = dyn_cast<IntegerAttr>(parameter);
      if (!outputAxis || outputAxis.getInt() < 0 ||
          outputAxis.getInt() >= static_cast<int64_t>(resultAxes.size()))
        return std::nullopt;
      inputAxes.push_back(resultAxes[outputAxis.getInt()]);
    }
    resultAxes = std::move(inputAxes);
    predicate = relation.getInputs().front();
  }
  return std::nullopt;
}

static_assert(onlineMmaAccumulatorGroupHasExactCoverage());
static_assert(onlineMmaProbabilityStoreAndLoadHaveExactCoverage());
static_assert(onlineMmaKLoadMatchesBFragment());
static_assert(onlineMmaSwizzleHasExactAtomCoverage());
static_assert(onlineMmaAFragmentRowOffset(0) == 0 &&
              onlineMmaAFragmentRowOffset(1) == 8 &&
              onlineMmaAFragmentRowOffset(2) == 0 &&
              onlineMmaAFragmentRowOffset(3) == 8);
static_assert(onlineMmaAFragmentColumnOffset(0) == 0 &&
              onlineMmaAFragmentColumnOffset(1) == 0 &&
              onlineMmaAFragmentColumnOffset(2) == 8 &&
              onlineMmaAFragmentColumnOffset(3) == 8);

FailureOr<SmallVector<int64_t>>
emitOnlineMmaCoordinates(PtxEmitter &ptx, ScheduleTaskOp task, int64_t block,
                         int64_t rowWithinTile) {
  FailureOr<SmallVector<AffineMap>> maps = onlineLogicalCoordinateMaps(task);
  if (failed(maps))
    return failure();
  int64_t zero = ptx.nextU32();
  ptx.stream << "  mov.u32 %r" << zero << ", 0;\n";
  SmallVector<int64_t> physical{block, zero, zero, rowWithinTile,
                                zero,  zero, zero};
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

int64_t emitOnlineMmaSharedElement(PtxEmitter &ptx, int64_t base, int64_t row,
                                   int64_t stride, int64_t column,
                                   bool swizzled) {
  if (swizzled) {
    int64_t mask = ptx.nextU32();
    int64_t physicalColumn = ptx.nextU32();
    ptx.stream << "  and.b32 %r" << mask << ", %r" << row
               << ", 7;\n  shl.b32 %r" << mask << ", %r" << mask
               << ", 3;\n  xor.b32 %r" << physicalColumn << ", %r" << column
               << ", %r" << mask << ";\n";
    column = physicalColumn;
  }
  int64_t element = ptx.nextU32();
  ptx.stream << "  mad.lo.u32 %r" << element << ", %r" << row << ", " << stride
             << ", %r" << column << ";\n";
  if (base != 0)
    ptx.stream << "  add.u32 %r" << element << ", %r" << element << ", " << base
               << ";\n";
  return element;
}

int64_t emitOnlineMmaConstant(PtxEmitter &ptx, int64_t value) {
  int64_t result = ptx.nextU32();
  ptx.stream << "  mov.u32 %r" << result << ", " << value << ";\n";
  return result;
}

std::array<int64_t, 4> emitOnlineMmaLdmatrixX4(PtxEmitter &ptx,
                                               int64_t sharedBase32,
                                               int64_t element) {
  int64_t address = ptx.nextU32();
  std::array<int64_t, 4> result;
  for (int64_t &reg : result)
    reg = ptx.nextU32();
  ptx.stream << "  mad.lo.u32 %r" << address << ", %r" << element << ", 2, %r"
             << sharedBase32
             << ";\n  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r" << result[0]
             << ", %r" << result[1] << ", %r" << result[2] << ", %r"
             << result[3] << "}, [%r" << address << "];\n";
  return result;
}

std::array<int64_t, 2> emitOnlineMmaLdmatrixX2Trans(PtxEmitter &ptx,
                                                    int64_t sharedBase32,
                                                    int64_t element) {
  int64_t address = ptx.nextU32();
  std::array<int64_t, 2> result;
  for (int64_t &reg : result)
    reg = ptx.nextU32();
  ptx.stream << "  mad.lo.u32 %r" << address << ", %r" << element << ", 2, %r"
             << sharedBase32
             << ";\n  ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r"
             << result[0] << ", %r" << result[1] << "}, [%r" << address
             << "];\n";
  return result;
}

std::array<int64_t, 2> emitOnlineMmaLdmatrixX2(PtxEmitter &ptx,
                                               int64_t sharedBase32,
                                               int64_t element) {
  int64_t address = ptx.nextU32();
  std::array<int64_t, 2> result;
  for (int64_t &reg : result)
    reg = ptx.nextU32();
  ptx.stream << "  mad.lo.u32 %r" << address << ", %r" << element << ", 2, %r"
             << sharedBase32
             << ";\n  ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r" << result[0]
             << ", %r" << result[1] << "}, [%r" << address << "];\n";
  return result;
}

FailureOr<std::string>
emitOnlineMmaVectorStatePtx(RegionOp region,
                            MaterializationTaskOp materializationTask,
                            ScheduleTaskOp task, OnlineEmissionMatch &match) {
  constexpr int64_t kRows = 128;
  constexpr int64_t kRowsPerPass = 64;
  constexpr int64_t kKeys = 64;
  constexpr int64_t kMmaKeys = 16;
  constexpr int64_t kThreads = 128;
  auto vectorDimensions = materializationTask.getOnlineVectorDimensions();
  if (!materializationTask.getOnlineMma().value_or(false) ||
      materializationTask.getKind() != MaterializationTaskKind::Contract ||
      materializationTask.getOutputBuffers().size() != 1 || !vectorDimensions ||
      vectorDimensions->size() != 1 || !task.getReductionAxis() ||
      task.getWorkgroupThreads() != kThreads || task.getSubgroupSize() != 32)
    return failure();
  ArrayRef<int64_t> domain = materializationTask.getDomainShape();
  const int64_t vectorAxis = vectorDimensions->front();
  const int64_t reductionAxis = *task.getReductionAxis();
  if (vectorAxis < 0 || vectorAxis >= static_cast<int64_t>(domain.size()) ||
      reductionAxis < 0 ||
      reductionAxis >= static_cast<int64_t>(domain.size()) ||
      vectorAxis == reductionAxis || domain[vectorAxis] <= 0 ||
      domain[vectorAxis] > 128 || domain[vectorAxis] % 8 != 0 ||
      domain[reductionAxis] <= 0 || domain[reductionAxis] > 2048 ||
      domain[reductionAxis] % kKeys != 0)
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
  std::optional<OnlineMmaCausalAxes> causalAxes =
      onlineMmaCausalAxes(match.score);
  if (!causalAxes || causalAxes->query < 0 || causalAxes->key < 0 ||
      causalAxes->query >=
          static_cast<int64_t>(normalizedMap.getValue().getNumResults()) ||
      causalAxes->key >=
          static_cast<int64_t>(normalizedMap.getValue().getNumResults()) ||
      normalizedMap.getValue().getResult(causalAxes->key) !=
          getAffineDimExpr(reductionAxis, region.getContext()))
    return failure();
  FailureOr<AffineMap> rawScoreMap = onlineAncestorCoordinateMap(
      match.score, match.scoreContract.getResult(0));
  if (failed(rawScoreMap) ||
      rawScoreMap->getNumDims() != normalizedMap.getValue().getNumResults())
    return failure();
  AffineMap scoreInputMap = rawScoreMap->compose(normalizedMap.getValue());
  FailureOr<OnlineScoreGeometry> scoreGeometry =
      onlineScoreGeometry(match, scoreInputMap, reductionAxis);
  if (failed(scoreGeometry))
    return failure();
  const int64_t scoreElements =
      scoreGeometry->domainShape[scoreGeometry->reductionAxis];
  if (scoreElements <= 0 || scoreElements > 256 || scoreElements % 16 != 0)
    return failure();
  auto regionYield = cast<YieldOp>(region.getBody().front().getTerminator());
  auto publicType =
      dyn_cast<RankedTensorType>(regionYield.getValues().front().getType());
  if (!publicType || !publicType.getElementType().isBF16())
    return failure();
  const int64_t qBase = 0;
  const int64_t kBase = kRows * scoreElements;
  const int64_t vBase = kBase + kKeys * scoreElements;
  constexpr int64_t kSharedBytes = 64 * 1024;
  // SM90 schedules use cp.async; the synchronous vector code remains the
  // fail-closed route for a schedule that does not grant the full tile budget.
  const bool asynchronousCopies = task.getScratchBytes() >= kSharedBytes;
  const int64_t sharedBytes = kSharedBytes;
  if (sharedBytes > task.getScratchBytes())
    return failure();

  PtxEmitter ptx(materializationTask.getInputBuffers().size() + 1);
  ptx.header(task.getSemanticFingerprint(), /*usesDynamicSharedMemory=*/true);
  FailureOr<OnlineEvalContext> context =
      onlineEvalContext(ptx, region, materializationTask, task, match);
  if (failed(context))
    return failure();
  FailureOr<OnlineMmaBf16Source> querySource = onlineMmaBf16Source(
      *context, match.scoreContract.getInputs()[scoreGeometry->queryOperand]);
  FailureOr<OnlineMmaBf16Source> keySource = onlineMmaBf16Source(
      *context, match.scoreContract.getInputs()[scoreGeometry->keyOperand]);
  FailureOr<OnlineMmaBf16Source> valueSource = onlineMmaBf16Source(
      *context, match.weightedContract.getInputs()[valueOperand]);
  if (failed(querySource) || failed(keySource) || failed(valueSource))
    return failure();
  auto isTrailingDimension = [](AffineMap map, int64_t domainAxis) {
    if (map.getNumResults() == 0)
      return false;
    auto dimension = dyn_cast<AffineDimExpr>(map.getResults().back());
    return dimension &&
           dimension.getPosition() == static_cast<unsigned>(domainAxis);
  };
  if (!isIdentityMap(querySource->coordinateMap) ||
      !isIdentityMap(keySource->coordinateMap) ||
      !isIdentityMap(valueSource->coordinateMap) ||
      querySource->type.getDimSize(querySource->type.getRank() - 1) !=
          scoreElements ||
      keySource->type.getDimSize(keySource->type.getRank() - 1) !=
          scoreElements ||
      valueSource->type.getDimSize(valueSource->type.getRank() - 1) !=
          domain[vectorAxis] ||
      !isTrailingDimension(scoreGeometry->queryMap,
                           scoreGeometry->reductionAxis) ||
      !isTrailingDimension(scoreGeometry->keyMap,
                           scoreGeometry->reductionAxis) ||
      !isTrailingDimension(valueMap.getValue(), vectorAxis))
    return failure();
  // CuTe's D128 atom is Swizzle<3,3,3>: the three row-within-octet bits
  // XOR the three 8-element column-sector bits. Select it only when each
  // source-proven row stride and shared base preserve the 8x64 atom.
  const bool swizzledShared =
      scoreElements == domain[vectorAxis] && scoreElements % 64 == 0 &&
      kBase % (scoreElements * 8) == 0 && vBase % (domain[vectorAxis] * 8) == 0;
  int64_t thread = ptx.nextU32();
  int64_t lane = ptx.nextU32();
  int64_t warp = ptx.nextU32();
  int64_t group = ptx.nextU32();
  int64_t threadInGroup = ptx.nextU32();
  int64_t block = ptx.nextU32();
  ptx.stream << "  mov.u32 %r" << thread << ", %tid.x;\n"
             << "  and.b32 %r" << lane << ", %r" << thread << ", 31;\n"
             << "  shr.u32 %r" << warp << ", %r" << thread << ", 5;\n"
             << "  shr.u32 %r" << group << ", %r" << lane << ", 2;\n"
             << "  and.b32 %r" << threadInGroup << ", %r" << lane
             << ", 3;\n  mov.u32 %r" << block << ", %ctaid.x;\n";
  int64_t sharedBase = ptx.nextU64();
  int64_t sharedBase32 = ptx.nextU32();
  ptx.stream << "  mov.u64 %rd" << sharedBase
             << ", shuttle_smem;\n  cvta.to.shared.u64 %rd" << sharedBase
             << ", %rd" << sharedBase << ";\n  cvt.u32.u64 %r" << sharedBase32
             << ", %rd" << sharedBase << ";\n";

  // Stage the 16-row Q tile once. All four warps participate in the copy;
  // rows beyond the source domain are explicitly zero padded.
  constexpr int64_t kVectorElements = 8;
  int64_t qStage = ptx.nextU32();
  int64_t qStageDone = ptx.nextPred();
  const int64_t qStageLabel = context->label++;
  int64_t initialQElement = ptx.nextU32();
  int64_t initialQRow = ptx.nextU32();
  int64_t initialQFeature = ptx.nextU32();
  ptx.stream << "  mul.lo.u32 %r" << initialQElement << ", %r" << thread << ", "
             << kVectorElements << ";\n  div.u32 %r" << initialQRow << ", %r"
             << initialQElement << ", " << scoreElements << ";\n  rem.u32 %r"
             << initialQFeature << ", %r" << initialQElement << ", "
             << scoreElements << ";\n";
  FailureOr<SmallVector<int64_t>> initialQContract =
      emitOnlineMmaCoordinates(ptx, task, block, initialQRow);
  if (failed(initialQContract))
    return failure();
  (*initialQContract)[reductionAxis] = initialQContract->front();
  FailureOr<SmallVector<int64_t>> initialQScore =
      emitOnlineMappedU32(ptx, scoreInputMap, *initialQContract);
  FailureOr<SmallVector<int64_t>> initialQDomain =
      failed(initialQScore)
          ? FailureOr<SmallVector<int64_t>>(failure())
          : onlineScoreDomainCoordinates(ptx, *scoreGeometry, *initialQScore,
                                         initialQFeature);
  FailureOr<SmallVector<int64_t>> initialQInput =
      failed(initialQDomain)
          ? FailureOr<SmallVector<int64_t>>(failure())
          : emitOnlineMappedU32(ptx, scoreGeometry->queryMap, *initialQDomain);
  FailureOr<SmallVector<int64_t>> initialQSource =
      failed(initialQInput)
          ? FailureOr<SmallVector<int64_t>>(failure())
          : emitOnlineMappedU32(ptx, querySource->coordinateMap,
                                *initialQInput);
  FailureOr<int64_t> recurrentQPointer =
      failed(initialQSource)
          ? FailureOr<int64_t>(failure())
          : emitOnlineTensorPointer(ptx, querySource->parameter,
                                    querySource->type, *initialQSource);
  if (failed(recurrentQPointer))
    return failure();
  ptx.stream << "  mov.u32 %r" << qStage << ", %r" << thread
             << ";\n$online_mma_q_" << qStageLabel << "_loop:\n  setp.ge.u32 %p"
             << qStageDone << ", %r" << qStage << ", "
             << kRows * scoreElements / kVectorElements << ";\n  @%p"
             << qStageDone << " bra $online_mma_q_" << qStageLabel
             << "_done;\n";
  int64_t qElement = initialQElement;
  int64_t qRow = initialQRow;
  std::array<int64_t, 4> qWords;
  for (int64_t &word : qWords) {
    word = ptx.nextU32();
    ptx.stream << "  mov.u32 %r" << word << ", 0;\n";
  }
  int64_t qLocalRow = ptx.nextU32();
  int64_t qRowValid = ptx.nextPred();
  SmallVector<int64_t> rowShape;
  for (auto [axis, extent] : llvm::enumerate(domain))
    if (static_cast<int64_t>(axis) != reductionAxis &&
        static_cast<int64_t>(axis) != vectorAxis)
      rowShape.push_back(extent);
  if (rowShape.size() < 2)
    return failure();
  FailureOr<int64_t> localRows =
      scheduleElementCount(ArrayRef<int64_t>(rowShape).take_back(2));
  if (failed(localRows))
    return failure();
  const int64_t blocksPerShared = ceilDivPositive(*localRows, kRows);
  ptx.stream << "  rem.u32 %r" << qLocalRow << ", %r" << block << ", "
             << blocksPerShared << ";\n  mad.lo.u32 %r" << qLocalRow << ", %r"
             << qLocalRow << ", " << kRows << ", %r" << qRow
             << ";\n  setp.lt.u32 %p" << qRowValid << ", %r" << qLocalRow
             << ", " << *localRows << ";\n";
  const int64_t qLoadLabel = context->label++;
  ptx.stream << "  @!%p" << qRowValid << " bra $online_mma_q_" << qLoadLabel
             << "_zero;\n";
  FailureOr<int64_t> qPointer = recurrentQPointer;
  emitOnlineMmaLoadGlobal128(ptx, *qPointer, qWords);
  ptx.stream << "$online_mma_q_" << qLoadLabel << "_zero:\n";
  emitOnlineMmaStoreShared128(
      ptx, sharedBase32, qElement, qWords,
      swizzledShared ? std::optional<int64_t>(scoreElements) : std::nullopt);
  ptx.stream << "  add.u64 %rd" << *recurrentQPointer << ", %rd"
             << *recurrentQPointer << ", " << kThreads * 16 << ";\n  add.u32 %r"
             << qElement << ", %r" << qElement << ", "
             << kThreads * kVectorElements << ";\n  add.u32 %r" << qRow
             << ", %r" << qRow << ", "
             << kThreads * kVectorElements / scoreElements << ";\n  add.u32 %r"
             << qStage << ", %r" << qStage << ", " << kThreads
             << ";\n  bra $online_mma_q_" << qStageLabel
             << "_loop;\n$online_mma_q_" << qStageLabel
             << "_done:\n  bar.sync 0;\n";

  std::array<SmallVector<int64_t>, 2> maxStateByPass;
  std::array<SmallVector<int64_t>, 2> sumStateByPass;
  for (int64_t pass = 0; pass < 2; ++pass) {
    maxStateByPass[pass].resize(2);
    sumStateByPass[pass].resize(2);
    for (int64_t rowHalf = 0; rowHalf < 2; ++rowHalf) {
      maxStateByPass[pass][rowHalf] = ptx.nextF32();
      sumStateByPass[pass][rowHalf] = ptx.nextF32();
      ptx.stream << "  mov.f32 %f" << maxStateByPass[pass][rowHalf]
                 << ", 0fFF800000;\n  mov.f32 %f"
                 << sumStateByPass[pass][rowHalf] << ", 0f00000000;\n";
    }
  }
  const int64_t valueChunks = domain[vectorAxis] / 8;
  const int64_t keyChunks = kKeys / 8;
  const int64_t ownedValueChunks = valueChunks;
  std::array<SmallVector<std::array<int64_t, 4>>, 2> outputStateByPass;
  for (auto &passState : outputStateByPass) {
    passState.resize(ownedValueChunks);
    for (auto &fragment : passState)
      for (int64_t &value : fragment) {
        value = ptx.nextF32();
        ptx.stream << "  mov.f32 %f" << value << ", 0f00000000;\n";
      }
  }

  // The selected source maps prove that K and V are row-major in their
  // trailing key/value dimensions. Decode B/Hkv and the lane's vector offset
  // once; each causal N64 tile is then a byte offset from these bases.
  int64_t baseElement = ptx.nextU32();
  int64_t baseKRow = ptx.nextU32();
  int64_t baseKFeature = ptx.nextU32();
  ptx.stream << "  mul.lo.u32 %r" << baseElement << ", %r" << thread << ", "
             << kVectorElements << ";\n  div.u32 %r" << baseKRow << ", %r"
             << baseElement << ", " << scoreElements << ";\n  rem.u32 %r"
             << baseKFeature << ", %r" << baseElement << ", " << scoreElements
             << ";\n";
  int64_t baseRowZero = emitOnlineMmaConstant(ptx, 0);
  FailureOr<SmallVector<int64_t>> baseContract =
      emitOnlineMmaCoordinates(ptx, task, block, baseRowZero);
  if (failed(baseContract))
    return failure();
  (*baseContract)[reductionAxis] = baseKRow;
  FailureOr<SmallVector<int64_t>> baseKScore =
      emitOnlineMappedU32(ptx, scoreInputMap, *baseContract);
  FailureOr<SmallVector<int64_t>> baseKDomain =
      failed(baseKScore) ? FailureOr<SmallVector<int64_t>>(failure())
                         : onlineScoreDomainCoordinates(
                               ptx, *scoreGeometry, *baseKScore, baseKFeature);
  FailureOr<SmallVector<int64_t>> baseKInput =
      failed(baseKDomain)
          ? FailureOr<SmallVector<int64_t>>(failure())
          : emitOnlineMappedU32(ptx, scoreGeometry->keyMap, *baseKDomain);
  FailureOr<SmallVector<int64_t>> baseKSource =
      failed(baseKInput)
          ? FailureOr<SmallVector<int64_t>>(failure())
          : emitOnlineMappedU32(ptx, keySource->coordinateMap, *baseKInput);
  FailureOr<int64_t> keyCtaBase =
      failed(baseKSource)
          ? FailureOr<int64_t>(failure())
          : emitOnlineTensorPointer(ptx, keySource->parameter, keySource->type,
                                    *baseKSource);

  int64_t baseVRow = ptx.nextU32();
  int64_t baseVColumn = ptx.nextU32();
  ptx.stream << "  div.u32 %r" << baseVRow << ", %r" << baseElement << ", "
             << domain[vectorAxis] << ";\n  rem.u32 %r" << baseVColumn << ", %r"
             << baseElement << ", " << domain[vectorAxis] << ";\n";
  (*baseContract)[reductionAxis] = baseVRow;
  (*baseContract)[vectorAxis] = baseVColumn;
  FailureOr<SmallVector<int64_t>> baseVInput =
      emitOnlineMappedU32(ptx, valueMap.getValue(), *baseContract);
  FailureOr<SmallVector<int64_t>> baseVSource =
      failed(baseVInput)
          ? FailureOr<SmallVector<int64_t>>(failure())
          : emitOnlineMappedU32(ptx, valueSource->coordinateMap, *baseVInput);
  FailureOr<int64_t> valueCtaBase =
      failed(baseVSource)
          ? FailureOr<int64_t>(failure())
          : emitOnlineTensorPointer(ptx, valueSource->parameter,
                                    valueSource->type, *baseVSource);
  if (failed(keyCtaBase) || failed(valueCtaBase))
    return failure();

  auto emitAsyncTileCopy = [&](int64_t globalBase, int64_t tileCoordinate,
                               int64_t rowElements, int64_t sharedElementBase,
                               int64_t vectorCount,
                               std::optional<int64_t> enabled = std::nullopt) {
    const int64_t copyLabel = context->label++;
    int64_t pointer = ptx.nextU64();
    int64_t stage = ptx.nextU32();
    int64_t element = ptx.nextU32();
    int64_t done = ptx.nextPred();
    if (enabled)
      ptx.stream << "  @!%p" << *enabled << " bra $online_mma_async_"
                 << copyLabel << "_done;\n";
    ptx.stream << "  mad.wide.u32 %rd" << pointer << ", %r" << tileCoordinate
               << ", " << rowElements * 2 << ", %rd" << globalBase
               << ";\n  mov.u32 %r" << stage << ", %r" << thread
               << ";\n  mad.lo.u32 %r" << element << ", %r" << thread << ", "
               << kVectorElements << ", " << sharedElementBase
               << ";\n$online_mma_async_" << copyLabel
               << "_loop:\n  setp.ge.u32 %p" << done << ", %r" << stage << ", "
               << vectorCount << ";\n  @%p" << done << " bra $online_mma_async_"
               << copyLabel << "_done;\n";
    emitOnlineMmaAsyncCopy128(
        ptx, sharedBase32, element, pointer, std::nullopt,
        swizzledShared ? std::optional<int64_t>(rowElements) : std::nullopt);
    ptx.stream << "  add.u64 %rd" << pointer << ", %rd" << pointer << ", "
               << kThreads * 16 << ";\n  add.u32 %r" << element << ", %r"
               << element << ", " << kThreads * kVectorElements
               << ";\n  add.u32 %r" << stage << ", %r" << stage << ", "
               << kThreads << ";\n  bra $online_mma_async_" << copyLabel
               << "_loop;\n$online_mma_async_" << copyLabel
               << "_done:\n  cp.async.commit_group;\n";
  };

  int64_t tileStart = ptx.nextU32();
  int64_t tileDone = ptx.nextPred();
  const int64_t tileLabel = context->label++;
  int64_t highestRow = emitOnlineMmaConstant(ptx, kRows - 1);
  FailureOr<SmallVector<int64_t>> highestContract =
      emitOnlineMmaCoordinates(ptx, task, block, highestRow);
  FailureOr<SmallVector<int64_t>> highestScore =
      failed(highestContract)
          ? FailureOr<SmallVector<int64_t>>(failure())
          : emitOnlineMappedU32(ptx, normalizedMap.getValue(),
                                *highestContract);
  if (failed(highestScore))
    return failure();
  ptx.stream << "  div.u32 %r" << tileStart << ", %r"
             << (*highestScore)[causalAxes->query] << ", " << kKeys
             << ";\n  mul.lo.u32 %r" << tileStart << ", %r" << tileStart << ", "
             << kKeys << ";\n";

  const int64_t keyElements = kKeys * scoreElements;
  const int64_t valueElements = kKeys * domain[vectorAxis];
  const int64_t keyVectors = keyElements / kVectorElements;
  const int64_t valueVectors = valueElements / kVectorElements;
  if (asynchronousCopies) {
    emitAsyncTileCopy(*keyCtaBase, tileStart, scoreElements, kBase, keyVectors);
    ptx.stream << "  cp.async.wait_group 0;\n  bar.sync 0;\n";
  }
  ptx.stream << "$online_mma_tile_" << tileLabel << "_loop:\n";
  if (asynchronousCopies)
    emitAsyncTileCopy(*valueCtaBase, tileStart, domain[vectorAxis], vBase,
                      valueVectors);

  // Stage one K/V tile. The key tile is exactly 16 because capability
  // checking requires a multiple-of-16 source reduction extent.
  if (!asynchronousCopies) {
    int64_t kvStage = ptx.nextU32();
    int64_t kvStageDone = ptx.nextPred();
    int64_t kvIsValue = ptx.nextPred();
    const int64_t kvLabel = context->label++;
    const int64_t valueLabel = context->label++;
    const int64_t nextLabel = context->label++;

    int64_t recurrentKPointer = ptx.nextU64();
    int64_t recurrentVPointer = ptx.nextU64();
    ptx.stream << "  mad.wide.u32 %rd" << recurrentKPointer << ", %r"
               << tileStart << ", " << scoreElements * 2 << ", %rd"
               << *keyCtaBase << ";\n  mad.wide.u32 %rd" << recurrentVPointer
               << ", %r" << tileStart << ", " << domain[vectorAxis] * 2
               << ", %rd" << *valueCtaBase << ";\n";

    ptx.stream << "  mov.u32 %r" << kvStage << ", %r" << thread
               << ";\n$online_mma_kv_" << kvLabel << "_loop:\n  setp.ge.u32 %p"
               << kvStageDone << ", %r" << kvStage << ", "
               << keyVectors + valueVectors << ";\n  @%p" << kvStageDone
               << " bra $online_mma_kv_" << kvLabel
               << "_done;\n  setp.ge.u32 %p" << kvIsValue << ", %r" << kvStage
               << ", " << keyVectors << ";\n";
    ptx.stream << "  @%p" << kvIsValue << " bra $online_mma_kv_" << valueLabel
               << "_value;\n";
    int64_t kvElement = ptx.nextU32();
    ptx.stream << "  mul.lo.u32 %r" << kvElement << ", %r" << kvStage << ", "
               << kVectorElements << ";\n";
    std::array<int64_t, 4> kWords;
    for (int64_t &word : kWords)
      word = ptx.nextU32();
    emitOnlineMmaLoadGlobal128(ptx, recurrentKPointer, kWords);
    int64_t kShared = ptx.nextU32();
    ptx.stream << "  add.u32 %r" << kShared << ", %r" << kvElement << ", "
               << kBase << ";\n";
    emitOnlineMmaStoreShared128(
        ptx, sharedBase32, kShared, kWords,
        swizzledShared ? std::optional<int64_t>(scoreElements) : std::nullopt);
    ptx.stream << "  bra $online_mma_kv_" << nextLabel
               << "_next;\n$online_mma_kv_" << valueLabel << "_value:\n";
    int64_t vFlat = ptx.nextU32();
    ptx.stream << "  sub.u32 %r" << vFlat << ", %r" << kvStage << ", "
               << keyVectors << ";\n  mul.lo.u32 %r" << vFlat << ", %r" << vFlat
               << ", " << kVectorElements << ";\n";
    std::array<int64_t, 4> vWords;
    for (int64_t &word : vWords)
      word = ptx.nextU32();
    emitOnlineMmaLoadGlobal128(ptx, recurrentVPointer, vWords);
    int64_t vShared = ptx.nextU32();
    ptx.stream << "  add.u32 %r" << vShared << ", %r" << vFlat << ", " << vBase
               << ";\n";
    emitOnlineMmaStoreShared128(ptx, sharedBase32, vShared, vWords,
                                swizzledShared
                                    ? std::optional<int64_t>(domain[vectorAxis])
                                    : std::nullopt);
    ptx.stream << "$online_mma_kv_" << nextLabel << "_next:\n  @!%p"
               << kvIsValue << " add.u64 %rd" << recurrentKPointer << ", %rd"
               << recurrentKPointer << ", " << kThreads * 16 << ";\n  @%p"
               << kvIsValue << " add.u64 %rd" << recurrentVPointer << ", %rd"
               << recurrentVPointer << ", " << kThreads * 16
               << ";\n  add.u32 %r" << kvStage << ", %r" << kvStage << ", "
               << kThreads << ";\n  bra $online_mma_kv_" << kvLabel
               << "_loop;\n$online_mma_kv_" << kvLabel
               << "_done:\n  bar.sync 0;\n";
  }

  std::array<SmallVector<std::array<int64_t, 4>>, 2> scoreFragmentsByPass;
  for (int64_t pass = 0; pass < 2; ++pass) {
    SmallVector<std::array<int64_t, 4>> &scoreFragments =
        scoreFragmentsByPass[pass];
    scoreFragments.resize(keyChunks);
    for (auto &fragment : scoreFragments)
      for (int64_t &score : fragment) {
        score = ptx.nextF32();
        ptx.stream << "  mov.f32 %f" << score << ", 0f00000000;\n";
      }
  }
  auto sharedElement = [&](int64_t base, int64_t row, int64_t stride,
                           int64_t column) {
    return emitOnlineMmaSharedElement(ptx, base, row, stride, column,
                                      swizzledShared);
  };
  for (int64_t featureBlock = 0; featureBlock < scoreElements;
       featureBlock += 16) {
    SmallVector<std::array<int64_t, 2>> bFragments(keyChunks);
    for (int64_t keyChunk = 0; keyChunk < keyChunks; ++keyChunk) {
      int64_t addressLane = ptx.nextU32();
      int64_t bMatrix = ptx.nextU32();
      int64_t bRow = ptx.nextU32();
      int64_t bColumn = ptx.nextU32();
      ptx.stream << "  and.b32 %r" << addressLane << ", %r" << lane
                 << ", 15;\n  shr.u32 %r" << bMatrix << ", %r" << addressLane
                 << ", 3;\n  and.b32 %r" << bRow << ", %r" << addressLane
                 << ", 7;\n  add.u32 %r" << bRow << ", %r" << bRow << ", "
                 << keyChunk * 8 << ";\n  mad.lo.u32 %r" << bColumn << ", %r"
                 << bMatrix << ", 8, " << featureBlock << ";\n";
      int64_t bElement = sharedElement(kBase, bRow, scoreElements, bColumn);
      bFragments[keyChunk] =
          emitOnlineMmaLdmatrixX2(ptx, sharedBase32, bElement);
    }
    for (int64_t pass = 0; pass < 2; ++pass) {
      int64_t aMatrix = ptx.nextU32();
      int64_t aRowOffset = ptx.nextU32();
      int64_t aColumnOffset = ptx.nextU32();
      int64_t aRow = ptx.nextU32();
      int64_t aColumn = ptx.nextU32();
      ptx.stream << "  shr.u32 %r" << aMatrix << ", %r" << lane
                 << ", 3;\n  and.b32 %r" << aRowOffset << ", %r" << aMatrix
                 << ", 1;\n  shl.b32 %r" << aRowOffset << ", %r" << aRowOffset
                 << ", 3;\n  shr.u32 %r" << aColumnOffset << ", %r" << aMatrix
                 << ", 1;\n  shl.b32 %r" << aColumnOffset << ", %r"
                 << aColumnOffset << ", 3;\n  and.b32 %r" << aRow << ", %r"
                 << lane << ", 7;\n  add.u32 %r" << aRow << ", %r" << aRow
                 << ", %r" << aRowOffset << ";\n  add.u32 %r" << aRow << ", %r"
                 << aRow << ", " << pass * kRowsPerPass << ";\n  mad.lo.u32 %r"
                 << aRow << ", %r" << warp << ", 16, %r" << aRow
                 << ";\n  add.u32 %r" << aColumn << ", %r" << aColumnOffset
                 << ", " << featureBlock << ";\n";
      int64_t aElement = sharedElement(qBase, aRow, scoreElements, aColumn);
      std::array<int64_t, 4> a =
          emitOnlineMmaLdmatrixX4(ptx, sharedBase32, aElement);
      for (int64_t keyChunk = 0; keyChunk < keyChunks; ++keyChunk) {
        auto &b = bFragments[keyChunk];
        auto &c = scoreFragmentsByPass[pass][keyChunk];
        ptx.stream << "  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                      "{%f"
                   << c[0] << ", %f" << c[1] << ", %f" << c[2] << ", %f" << c[3]
                   << "}, {%r" << a[0] << ", %r" << a[1] << ", %r" << a[2]
                   << ", %r" << a[3] << "}, {%r" << b[0] << ", %r" << b[1]
                   << "}, {%f" << c[0] << ", %f" << c[1] << ", %f" << c[2]
                   << ", %f" << c[3] << "};\n";
      }
    }
  }

  if (asynchronousCopies) {
    // Current V arrives while QK consumes K.  Once all warps are through the
    // QK phase, K is dead and the next reverse tile can start arriving while
    // softmax and PV consume the separate P/V buffers.
    ptx.stream << "  cp.async.wait_group 0;\n  bar.sync 0;\n";
    int64_t nextTile = ptx.nextU32();
    int64_t hasNextTile = ptx.nextPred();
    ptx.stream << "  setp.ge.u32 %p" << hasNextTile << ", %r" << tileStart
               << ", " << kKeys << ";\n  sub.u32 %r" << nextTile << ", %r"
               << tileStart << ", " << kKeys << ";\n";
    emitAsyncTileCopy(*keyCtaBase, nextTile, scoreElements, kBase, keyVectors,
                      hasNextTile);
  }

  std::array<std::array<int64_t, 2>, 2> nextMaximumByPass;
  std::array<std::array<int64_t, 2>, 2> oldScaleByPass;
  std::array<std::array<int64_t, 2>, 2> nextSumByPass;
  std::array<SmallVector<std::array<int64_t, 2>>, 2> probabilityPairsByPass;
  for (int64_t pass = 0; pass < 2; ++pass) {
    SmallVector<int64_t> &maxState = maxStateByPass[pass];
    SmallVector<int64_t> &sumState = sumStateByPass[pass];
    SmallVector<std::array<int64_t, 4>> &outputState = outputStateByPass[pass];
    SmallVector<std::array<int64_t, 4>> &scoreFragments =
        scoreFragmentsByPass[pass];
    probabilityPairsByPass[pass].resize(keyChunks);
    int64_t rowLow = ptx.nextU32();
    int64_t rowHigh = ptx.nextU32();
    ptx.stream << "  mad.lo.u32 %r" << rowLow << ", %r" << warp << ", 16, "
               << pass * kRowsPerPass << ";\n  add.u32 %r" << rowLow << ", %r"
               << rowLow << ", %r" << group << ";\n  add.u32 %r" << rowHigh
               << ", %r" << rowLow << ", 8;\n";
    auto sharedElement = [&](int64_t base, int64_t row, int64_t stride,
                             int64_t column) {
      return emitOnlineMmaSharedElement(ptx, base, row, stride, column,
                                        swizzledShared);
    };

    SmallVector<std::array<int64_t, 4>> transformedScores(keyChunks);
    SmallVector<std::array<int64_t, 4>> scorePredicates(keyChunks);
    for (int64_t keyChunk = 0; keyChunk < keyChunks; ++keyChunk) {
      for (int64_t fragment = 0; fragment < 4; ++fragment) {
        const bool highRow = onlineMmaAccumulatorRowOffset(fragment) != 0;
        int64_t keyColumn = ptx.nextU32();
        ptx.stream << "  mad.lo.u32 %r" << keyColumn << ", %r" << threadInGroup
                   << ", 2, "
                   << keyChunk * 8 + onlineMmaAccumulatorColumnOffset(fragment)
                   << ";\n";
        FailureOr<SmallVector<int64_t>> scoreContractCoordinates =
            emitOnlineMmaCoordinates(ptx, task, block,
                                     highRow ? rowHigh : rowLow);
        if (failed(scoreContractCoordinates))
          return failure();
        int64_t globalKey = ptx.nextU32();
        ptx.stream << "  add.u32 %r" << globalKey << ", %r" << tileStart
                   << ", %r" << keyColumn << ";\n";
        (*scoreContractCoordinates)[reductionAxis] = globalKey;
        FailureOr<SmallVector<int64_t>> scoreCoordinates = emitOnlineMappedU32(
            ptx, normalizedMap.getValue(), *scoreContractCoordinates);
        if (failed(scoreCoordinates))
          return failure();
        context->relationValues.clear();
        context->valueOverrides[match.scoreContract.getResult(0)] = {
            OnlineScalarKind::F32, scoreFragments[keyChunk][fragment]};
        FailureOr<OnlineScalarValue> score =
            emitOnlineValue(*context, match.score, *scoreCoordinates);
        context->valueOverrides.erase(match.scoreContract.getResult(0));
        auto select = match.score.getDefiningOp<RelationOp>();
        auto predicate =
            select ? context->relationValues.find(select.getInputs().front())
                   : context->relationValues.end();
        if (failed(score) || score->kind != OnlineScalarKind::F32 || !select ||
            select.getSemantics() != RelationSemantics::Select ||
            predicate == context->relationValues.end() ||
            predicate->second.kind != OnlineScalarKind::Predicate)
          return failure();
        transformedScores[keyChunk][fragment] = score->reg;
        scorePredicates[keyChunk][fragment] = predicate->second.reg;
      }
    }

    std::array<int64_t, 2> &nextMaximum = nextMaximumByPass[pass];
    std::array<int64_t, 2> &oldScale = oldScaleByPass[pass];
    std::array<int64_t, 2> &nextSum = nextSumByPass[pass];
    for (int64_t rowHalf = 0; rowHalf < 2; ++rowHalf) {
      int64_t localMaximum = ptx.nextF32();
      ptx.stream << "  max.f32 %f" << localMaximum << ", %f"
                 << transformedScores[0][rowHalf * 2] << ", %f"
                 << transformedScores[0][rowHalf * 2 + 1] << ";\n";
      for (int64_t keyChunk = 1; keyChunk < keyChunks; ++keyChunk) {
        int64_t pairMaximum = ptx.nextF32();
        int64_t combined = ptx.nextF32();
        ptx.stream << "  max.f32 %f" << pairMaximum << ", %f"
                   << transformedScores[keyChunk][rowHalf * 2] << ", %f"
                   << transformedScores[keyChunk][rowHalf * 2 + 1]
                   << ";\n  max.f32 %f" << combined << ", %f" << localMaximum
                   << ", %f" << pairMaximum << ";\n";
        localMaximum = combined;
      }
      for (int offset : {2, 1}) {
        int64_t shuffled = ptx.nextF32();
        int64_t combined = ptx.nextF32();
        ptx.stream << "  shfl.sync.down.b32 %f" << shuffled << ", %f"
                   << localMaximum << ", " << offset
                   << ", 0x1c03, 0xffffffff;\n  max.f32 %f" << combined
                   << ", %f" << localMaximum << ", %f" << shuffled << ";\n";
        localMaximum = combined;
      }
      int64_t groupLeader = ptx.nextU32();
      int64_t tileMaximum = ptx.nextF32();
      ptx.stream << "  shl.b32 %r" << groupLeader << ", %r" << group
                 << ", 2;\n  shfl.sync.idx.b32 %f" << tileMaximum << ", %f"
                 << localMaximum << ", %r" << groupLeader
                 << ", 0x1f, 0xffffffff;\n";
      nextMaximum[rowHalf] = ptx.nextF32();
      int64_t oldDelta = ptx.nextF32();
      ptx.stream << "  max.f32 %f" << nextMaximum[rowHalf] << ", %f"
                 << maxState[rowHalf] << ", %f" << tileMaximum
                 << ";\n  sub.rn.f32 %f" << oldDelta << ", %f"
                 << maxState[rowHalf] << ", %f" << nextMaximum[rowHalf]
                 << ";\n";
      int64_t rawOldScale = emitOnlineExp(ptx, oldDelta);
      int64_t stateEmpty = ptx.nextPred();
      oldScale[rowHalf] = ptx.nextF32();
      ptx.stream << "  setp.eq.f32 %p" << stateEmpty << ", %f"
                 << sumState[rowHalf] << ", 0f00000000;\n  selp.f32 %f"
                 << oldScale[rowHalf] << ", 0f00000000, %f" << rawOldScale
                 << ", %p" << stateEmpty << ";\n";
    }

    for (int64_t rowHalf = 0; rowHalf < 2; ++rowHalf) {
      int64_t localSum = ptx.nextF32();
      ptx.stream << "  mov.f32 %f" << localSum << ", 0f00000000;\n";
      for (int64_t keyChunk = 0; keyChunk < keyChunks; ++keyChunk) {
        std::array<int64_t, 2> probabilities;
        for (int64_t element = 0; element < 2; ++element) {
          const int64_t fragment = rowHalf * 2 + element;
          int64_t delta = ptx.nextF32();
          int64_t exponential = 0;
          int64_t probability = ptx.nextF32();
          ptx.stream << "  sub.rn.f32 %f" << delta << ", %f"
                     << transformedScores[keyChunk][fragment] << ", %f"
                     << nextMaximum[rowHalf] << ";\n";
          exponential = emitOnlineExp(ptx, delta);
          ptx.stream << "  selp.f32 %f" << probability << ", %f" << exponential
                     << ", 0f00000000, %p"
                     << scorePredicates[keyChunk][fragment]
                     << ";\n  add.rn.f32 %f" << localSum << ", %f" << localSum
                     << ", %f" << probability << ";\n";
          probabilities[element] = probability;
        }
        int64_t low = ptx.nextBF16();
        int64_t high = ptx.nextBF16();
        ptx.stream << "  cvt.rn.bf16.f32 %h" << low << ", %f"
                   << probabilities[0] << ";\n  cvt.rn.bf16.f32 %h" << high
                   << ", %f" << probabilities[1] << ";\n";
        probabilityPairsByPass[pass][keyChunk][rowHalf] =
            emitOnlinePackBf16(ptx, low, high);
      }
      for (int offset : {2, 1}) {
        int64_t shuffled = ptx.nextF32();
        int64_t combined = ptx.nextF32();
        ptx.stream << "  shfl.sync.down.b32 %f" << shuffled << ", %f"
                   << localSum << ", " << offset
                   << ", 0x1c03, 0xffffffff;\n  add.rn.f32 %f" << combined
                   << ", %f" << localSum << ", %f" << shuffled << ";\n";
        localSum = combined;
      }
      int64_t leader = ptx.nextU32();
      int64_t tileSum = ptx.nextF32();
      nextSum[rowHalf] = ptx.nextF32();
      ptx.stream << "  shl.b32 %r" << leader << ", %r" << group
                 << ", 2;\n  shfl.sync.idx.b32 %f" << tileSum << ", %f"
                 << localSum << ", %r" << leader
                 << ", 0x1f, 0xffffffff;\n  fma.rn.f32 %f" << nextSum[rowHalf]
                 << ", %f" << oldScale[rowHalf] << ", %f" << sumState[rowHalf]
                 << ", %f" << tileSum << ";\n";
    }
    for (int64_t ownedChunk = 0; ownedChunk < ownedValueChunks; ++ownedChunk) {
      auto &c = outputState[ownedChunk];
      for (int64_t fragment = 0; fragment < 4; ++fragment) {
        ptx.stream << "  mul.rn.f32 %f" << c[fragment] << ", %f" << c[fragment]
                   << ", %f" << oldScale[fragment < 2 ? 0 : 1] << ";\n";
      }
    }
  }

  // Both M64 halves now own live P and online state fragments.  Load each V
  // fragment once and feed both row halves before advancing the K fragment.
  for (int64_t ownedChunk = 0; ownedChunk < ownedValueChunks; ++ownedChunk) {
    int64_t valueChunk = ptx.nextU32();
    ptx.stream << "  mov.u32 %r" << valueChunk << ", " << ownedChunk << ";\n";
    for (int64_t keyBlock = 0; keyBlock < kKeys; keyBlock += kMmaKeys) {
      int64_t bMatrix = ptx.nextU32();
      int64_t bRow = ptx.nextU32();
      int64_t bColumn = ptx.nextU32();
      ptx.stream << "  shr.u32 %r" << bMatrix << ", %r" << lane
                 << ", 3;\n  and.b32 %r" << bMatrix << ", %r" << bMatrix
                 << ", 1;\n  and.b32 %r" << bRow << ", %r" << lane
                 << ", 7;\n  mad.lo.u32 %r" << bRow << ", %r" << bMatrix
                 << ", 8, %r" << bRow << ";\n  add.u32 %r" << bRow << ", %r"
                 << bRow << ", " << keyBlock << ";\n  mul.lo.u32 %r" << bColumn
                 << ", %r" << valueChunk << ", 8;\n";
      int64_t bElement =
          sharedElement(vBase, bRow, domain[vectorAxis], bColumn);
      std::array<int64_t, 2> b =
          emitOnlineMmaLdmatrixX2Trans(ptx, sharedBase32, bElement);
      for (int64_t pass = 0; pass < 2; ++pass) {
        auto &c = outputStateByPass[pass][ownedChunk];
        const int64_t lowChunk = keyBlock / 8;
        std::array<int64_t, 4> a{probabilityPairsByPass[pass][lowChunk][0],
                                 probabilityPairsByPass[pass][lowChunk][1],
                                 probabilityPairsByPass[pass][lowChunk + 1][0],
                                 probabilityPairsByPass[pass][lowChunk + 1][1]};
        ptx.stream << "  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                      "{%f"
                   << c[0] << ", %f" << c[1] << ", %f" << c[2] << ", %f" << c[3]
                   << "}, {%r" << a[0] << ", %r" << a[1] << ", %r" << a[2]
                   << ", %r" << a[3] << "}, {%r" << b[0] << ", %r" << b[1]
                   << "}, {%f" << c[0] << ", %f" << c[1] << ", %f" << c[2]
                   << ", %f" << c[3] << "};\n";
      }
    }
  }
  for (int64_t pass = 0; pass < 2; ++pass) {
    for (int64_t rowHalf = 0; rowHalf < 2; ++rowHalf)
      ptx.stream << "  mov.f32 %f" << maxStateByPass[pass][rowHalf] << ", %f"
                 << nextMaximumByPass[pass][rowHalf] << ";\n  mov.f32 %f"
                 << sumStateByPass[pass][rowHalf] << ", %f"
                 << nextSumByPass[pass][rowHalf] << ";\n";
  }
  if (asynchronousCopies)
    ptx.stream << "  cp.async.wait_group 0;\n  bar.sync 0;\n";
  else
    ptx.stream << "  bar.sync 0;\n";
  ptx.stream << "  setp.eq.u32 %p" << tileDone << ", %r" << tileStart
             << ", 0;\n  @%p" << tileDone << " bra $online_mma_tile_"
             << tileLabel << "_done;\n  sub.u32 %r" << tileStart << ", %r"
             << tileStart << ", " << kKeys << ";\n  bra $online_mma_tile_"
             << tileLabel << "_loop;\n$online_mma_tile_" << tileLabel
             << "_done:\n";

  // Q is dead. Reuse its swizzled shared tile for the finalized BF16 output,
  // then drain it with coalesced 16-byte global transactions.
  for (int64_t pass = 0; pass < 2; ++pass) {
    SmallVector<int64_t> &sumState = sumStateByPass[pass];
    SmallVector<std::array<int64_t, 4>> &outputState = outputStateByPass[pass];
    int64_t rowLow = ptx.nextU32();
    int64_t rowHigh = ptx.nextU32();
    ptx.stream << "  mad.lo.u32 %r" << rowLow << ", %r" << warp << ", 16, "
               << pass * kRowsPerPass << ";\n  add.u32 %r" << rowLow << ", %r"
               << rowLow << ", %r" << group << ";\n  add.u32 %r" << rowHigh
               << ", %r" << rowLow << ", 8;\n";

    for (int64_t ownedChunk = 0; ownedChunk < ownedValueChunks; ++ownedChunk) {
      int64_t valueBase = ptx.nextU32();
      ptx.stream << "  mad.lo.u32 %r" << valueBase << ", %r" << threadInGroup
                 << ", 2, " << ownedChunk * 8 << ";\n";
      for (int64_t rowHalf = 0; rowHalf < 2; ++rowHalf) {
        int64_t row = rowHalf == 0 ? rowLow : rowHigh;
        int64_t localRow = ptx.nextU32();
        int64_t rowValid = ptx.nextPred();
        ptx.stream << "  rem.u32 %r" << localRow << ", %r" << block << ", "
                   << blocksPerShared << ";\n  mad.lo.u32 %r" << localRow
                   << ", %r" << localRow << ", " << kRows << ", %r" << row
                   << ";\n  setp.lt.u32 %p" << rowValid << ", %r" << localRow
                   << ", " << *localRows << ";\n";
        const int64_t stageLabel = context->label++;
        ptx.stream << "  @!%p" << rowValid << " bra $online_mma_ostage_"
                   << stageLabel << "_skip;\n";
        std::array<int64_t, 2> rounded;
        for (int64_t element = 0; element < 2; ++element) {
          int64_t valueCoordinate = ptx.nextU32();
          ptx.stream << "  add.u32 %r" << valueCoordinate << ", %r" << valueBase
                     << ", " << element << ";\n";
          FailureOr<SmallVector<int64_t>> outputContractCoordinates =
              emitOnlineMmaCoordinates(ptx, task, block, row);
          if (failed(outputContractCoordinates))
            return failure();
          (*outputContractCoordinates)[vectorAxis] = valueCoordinate;
          FailureOr<SmallVector<int64_t>> weightedCoordinates =
              emitOnlineMappedU32(ptx, weightedOutputMap.getValue(),
                                  *outputContractCoordinates);
          FailureOr<SmallVector<int64_t>> publicCoordinates =
              failed(weightedCoordinates)
                  ? FailureOr<SmallVector<int64_t>>(failure())
                  : onlineOutputCoordinates(match.weightedContract.getResult(0),
                                            regionYield.getValues().front(),
                                            *weightedCoordinates);
          if (failed(publicCoordinates))
            return failure();
          int64_t normalized = ptx.nextF32();
          const int64_t fragment = rowHalf * 2 + element;
          ptx.stream << "  div.rn.f32 %f" << normalized << ", %f"
                     << outputState[ownedChunk][fragment] << ", %f"
                     << sumState[rowHalf] << ";\n";
          FailureOr<OnlineScalarValue> epilogue = emitOnlineEpilogue(
              *context, regionYield.getValues().front(),
              match.weightedContract.getResult(0),
              {OnlineScalarKind::F32, normalized}, *publicCoordinates);
          if (failed(epilogue))
            return failure();
          rounded[element] = ptx.nextBF16();
          ptx.stream << "  cvt.rn.bf16.f32 %h" << rounded[element] << ", %f"
                     << epilogue->reg << ";\n";
        }
        int64_t packed = emitOnlinePackBf16(ptx, rounded[0], rounded[1]);
        int64_t sharedElement = emitOnlineMmaSharedElement(
            ptx, qBase, row, domain[vectorAxis], valueBase, swizzledShared);
        int64_t sharedAddress = ptx.nextU32();
        ptx.stream << "  mad.lo.u32 %r" << sharedAddress << ", %r"
                   << sharedElement << ", 2, %r" << sharedBase32
                   << ";\n  st.shared.u32 [%r" << sharedAddress << "], %r"
                   << packed << ";\n$online_mma_ostage_" << stageLabel
                   << "_skip:\n";
      }
    }
  }
  ptx.stream << "  bar.sync 0;\n";

  int64_t outputStage = ptx.nextU32();
  int64_t outputStageDone = ptx.nextPred();
  const int64_t outputStageLabel = context->label++;
  ptx.stream << "  mov.u32 %r" << outputStage << ", %r" << thread
             << ";\n$online_mma_odrain_" << outputStageLabel
             << "_loop:\n  setp.ge.u32 %p" << outputStageDone << ", %r"
             << outputStage << ", "
             << kRows * domain[vectorAxis] / kVectorElements << ";\n  @%p"
             << outputStageDone << " bra $online_mma_odrain_"
             << outputStageLabel << "_done;\n";
  int64_t outputElement = ptx.nextU32();
  int64_t outputRow = ptx.nextU32();
  int64_t outputColumn = ptx.nextU32();
  ptx.stream << "  mul.lo.u32 %r" << outputElement << ", %r" << outputStage
             << ", " << kVectorElements << ";\n  div.u32 %r" << outputRow
             << ", %r" << outputElement << ", " << domain[vectorAxis]
             << ";\n  rem.u32 %r" << outputColumn << ", %r" << outputElement
             << ", " << domain[vectorAxis] << ";\n";
  FailureOr<SmallVector<int64_t>> outputContractCoordinates =
      emitOnlineMmaCoordinates(ptx, task, block, outputRow);
  if (failed(outputContractCoordinates))
    return failure();
  (*outputContractCoordinates)[vectorAxis] = outputColumn;
  FailureOr<SmallVector<int64_t>> weightedCoordinates = emitOnlineMappedU32(
      ptx, weightedOutputMap.getValue(), *outputContractCoordinates);
  FailureOr<SmallVector<int64_t>> publicCoordinates =
      failed(weightedCoordinates)
          ? FailureOr<SmallVector<int64_t>>(failure())
          : onlineOutputCoordinates(match.weightedContract.getResult(0),
                                    regionYield.getValues().front(),
                                    *weightedCoordinates);
  if (failed(publicCoordinates))
    return failure();
  int64_t outputLocalRow = ptx.nextU32();
  int64_t outputRowValid = ptx.nextPred();
  ptx.stream << "  rem.u32 %r" << outputLocalRow << ", %r" << block << ", "
             << blocksPerShared << ";\n  mad.lo.u32 %r" << outputLocalRow
             << ", %r" << outputLocalRow << ", " << kRows << ", %r" << outputRow
             << ";\n  setp.lt.u32 %p" << outputRowValid << ", %r"
             << outputLocalRow << ", " << *localRows << ";\n";
  const int64_t drainLabel = context->label++;
  ptx.stream << "  @!%p" << outputRowValid << " bra $online_mma_odrain_"
             << drainLabel << "_skip;\n";
  FailureOr<int64_t> outputPointer = emitOnlineGlobalStorePointer(
      ptx, materializationTask.getInputBuffers().size(), publicType,
      *publicCoordinates);
  if (failed(outputPointer))
    return failure();
  std::array<int64_t, 4> outputWords = emitOnlineMmaLoadShared128(
      ptx, sharedBase32, outputElement,
      swizzledShared ? std::optional<int64_t>(domain[vectorAxis])
                     : std::nullopt);
  emitOnlineMmaStoreGlobal128(ptx, *outputPointer, outputWords);
  ptx.stream << "$online_mma_odrain_" << drainLabel << "_skip:\n  add.u32 %r"
             << outputStage << ", %r" << outputStage << ", " << kThreads
             << ";\n  bra $online_mma_odrain_" << outputStageLabel
             << "_loop;\n$online_mma_odrain_" << outputStageLabel << "_done:\n";
  ptx.finish();
  return ptx.text;
}
