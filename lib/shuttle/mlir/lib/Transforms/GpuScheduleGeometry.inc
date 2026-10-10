// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

struct ScheduledGeometry {
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

constexpr int64_t kGenericSimt32MaximumScratchBytes = 96 * 1024;

DenseI64ArrayAttr structuralDimensions(MaterializationTaskOp task,
                                       StringRef name) {
  return task->getAttrOfType<DenseI64ArrayAttr>(name);
}

bool isStatefulContractFoldTask(MaterializationTaskOp task) {
  return !task->hasAttr("composition_kind") &&
         task->hasAttr("composition_sources") &&
         structuralDimensions(task, "composition_owner_dimensions") &&
         structuralDimensions(task, "composition_stream_dimensions") &&
         structuralDimensions(task,
                              "composition_producer_contraction_dimensions") &&
         structuralDimensions(task,
                              "composition_consumer_contraction_dimensions");
}

bool isOnlineStateTask(MaterializationTaskOp task) {
  return isStatefulContractFoldTask(task) &&
         task.getOnlineVectorDimensionsAttr() &&
         !task->hasAttr("composition_state_inputs") &&
         !task->hasAttr("composition_state_outputs");
}

bool isReverseStateTask(MaterializationTaskOp task) {
  return isStatefulContractFoldTask(task) &&
         task.getOnlineVectorDimensionsAttr() &&
         (task->hasAttr("composition_state_inputs") ||
          task->hasAttr("composition_state_outputs"));
}

bool isCertifiedReverseStatePlan(MaterializationPlanOp plan) {
  if (plan.getPolicy() != NumericalPolicy::Fast ||
      !plan.getStatefulContractFoldPullback().value_or(false))
    return false;
  unsigned forwardTasks = 0;
  unsigned ownerAdjointTasks = 0;
  unsigned streamedAdjointTasks = 0;
  for (MaterializationTaskOp task :
       plan.getBody().front().getOps<MaterializationTaskOp>()) {
    if (!isReverseStateTask(task))
      return false;
    auto owner = structuralDimensions(task, "composition_owner_dimensions");
    auto stream = structuralDimensions(task, "composition_stream_dimensions");
    auto vector = task.getOnlineVectorDimensionsAttr();
    if (!owner || owner.empty() || !stream || !vector || vector.empty() ||
        stream.asArrayRef() != task.getReductionDimensions())
      return false;
    const bool hasStateInputs = task->hasAttr("composition_state_inputs");
    const bool hasStateOutputs = task->hasAttr("composition_state_outputs");
    if (!hasStateInputs && hasStateOutputs && stream.size() == 1)
      ++forwardTasks;
    else if (hasStateInputs && hasStateOutputs && stream.size() == 1)
      ++ownerAdjointTasks;
    else if (hasStateInputs && !hasStateOutputs && stream.size() == 2)
      ++streamedAdjointTasks;
    else
      return false;
  }
  return forwardTasks <= 1 && ownerAdjointTasks == 1 &&
         streamedAdjointTasks == 1;
}

bool isCertifiedOnlineStatePlan(MaterializationPlanOp plan) {
  if (plan.getPolicy() != NumericalPolicy::Fast)
    return false;
  bool found = false;
  for (MaterializationTaskOp task :
       plan.getBody().front().getOps<MaterializationTaskOp>()) {
    if (!isOnlineStateTask(task))
      return false;
    found = true;
  }
  return found;
}

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
                                                  NumericalPolicy policy,
                                                  std::optional<int64_t>
                                                      foldReductionAxis =
                                                          std::nullopt) {
  constexpr int64_t kSubgroup = 32;
  constexpr int64_t kMaxThreads = 256;
  if (kind == ScheduleTaskKind::OrderedState) {
    if (domain.size() != 1 || domain.front() <= 0)
      return failure();
    return ScheduledGeometry{{1},
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
    if (domain.size() == 1 && domain.front() > 0) {
      const int64_t tile = std::min(domain.front(), kMaxThreads);
      const int64_t threads = scheduleThreads(tile);
      return ScheduledGeometry{{1},
                               {tile},
                               ceilDivPositive(domain.front(), tile),
                               threads,
                               kSubgroup,
                               threads * static_cast<int64_t>(sizeof(float)),
                               0};
    }
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
  if (kind == ScheduleTaskKind::AllAxisFold) {
    if (domain.empty() || domain.size() > 8)
      return failure();
    FailureOr<int64_t> reduction = scheduleElementCount(domain);
    if (failed(reduction))
      return failure();
    constexpr int64_t kItemsPerThread = 16;
    constexpr int64_t kPartialTile = kMaxThreads * kItemsPerThread;
    constexpr int64_t kMaximumMergeTiles = kMaxThreads;
    const int64_t partialCount = ceilDivPositive(*reduction, kPartialTile);
    const int64_t partialSerialTiles =
        ceilDivPositive(*reduction, partialCount * kMaxThreads);
    const int64_t mergeSerialTiles =
        ceilDivPositive(partialCount, kMaxThreads);
    if (mergeSerialTiles > kMaximumMergeTiles)
      return failure();
    return ScheduledGeometry{{partialCount},
                             {kPartialTile},
                             partialSerialTiles,
                             kMaxThreads,
                             kSubgroup,
                             kMaxThreads * static_cast<int64_t>(sizeof(float)),
                             std::nullopt,
                             {1},
                             {kMaxThreads},
                             mergeSerialTiles,
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
  const int64_t reductionAxis =
      rowFold && foldReductionAxis ? *foldReductionAxis
                                   : (rowFold ? domain.size() - 1 : 0);
  if (reductionAxis < 0 || reductionAxis >= static_cast<int64_t>(domain.size()))
    return failure();
  SmallVector<int64_t> ownerShape;
  for (auto [axis, extent] : llvm::enumerate(domain))
    if (static_cast<int64_t>(axis) != reductionAxis)
      ownerShape.push_back(extent);
  FailureOr<int64_t> reduction =
      rowFold ? FailureOr<int64_t>(domain[reductionAxis])
              : scheduleElementCount(domain.drop_back());
  FailureOr<int64_t> outputs =
      rowFold ? scheduleElementCount(ownerShape)
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

FailureOr<ScheduledGeometry>
deriveOnlineStateGeometry(MaterializationTaskOp task) {
  constexpr int64_t kOnlineWarpSize = 32;
  constexpr int64_t kOnlineWarpsPerBlock = 4;
  // The current SM90 MMA lowering assigns one m16 row fragment to each warp.
  // This is the resource-safe half-size form of the M128/N64 tuning target;
  // the emitter may grow it to two row fragments per warp when its register
  // budget admits the full target.
  constexpr int64_t kOnlineMmaRowsPerBlock = 128;
  // Persistent Q plus one K and one V tile. Probabilities remain in registers.
  constexpr int64_t kOnlineMmaSharedMemoryBytes = 64 * 1024;
  constexpr int64_t kMaximumOnlineVectorElements = 128;
  constexpr int64_t kMaximumOnlineReductionElements = 2048;
  ArrayRef<int64_t> domain = task.getDomainShape();
  ArrayRef<int64_t> reductions = task.getReductionDimensions();
  auto vectorDimensions = task.getOnlineVectorDimensions();
  auto ownerDimensions =
      structuralDimensions(task, "composition_owner_dimensions");
  auto streamDimensions =
      structuralDimensions(task, "composition_stream_dimensions");
  if (!isOnlineStateTask(task) || domain.empty() || domain.size() > 8 ||
      reductions.size() != 1 || reductions.front() < 0 ||
      reductions.front() >= static_cast<int64_t>(domain.size()) ||
      !vectorDimensions || vectorDimensions->empty() || !ownerDimensions ||
      ownerDimensions.empty() || !streamDimensions ||
      streamDimensions.asArrayRef() != reductions)
    return failure();
  const int64_t reductionAxis = reductions.front();
  const int64_t reduction = domain[reductionAxis];
  llvm::SmallDenseSet<int64_t> vectorAxes;
  int64_t vectorElements = 1;
  for (int64_t axis : *vectorDimensions) {
    if (axis < 0 || axis >= static_cast<int64_t>(domain.size()) ||
        axis == reductionAxis || !vectorAxes.insert(axis).second ||
        domain[axis] <= 0 ||
        vectorElements > kMaximumOnlineVectorElements / domain[axis])
      return failure();
    vectorElements *= domain[axis];
  }
  llvm::SmallDenseSet<int64_t> ownerAxes;
  SmallVector<int64_t> outputShape;
  for (int64_t axis : ownerDimensions.asArrayRef()) {
    if (axis < 0 || axis >= static_cast<int64_t>(domain.size()) ||
        axis == reductionAxis || vectorAxes.contains(axis) ||
        !ownerAxes.insert(axis).second)
      return failure();
    outputShape.push_back(domain[axis]);
  }
  FailureOr<int64_t> outputs = scheduleElementCount(outputShape);
  if (failed(outputs) || *outputs > UINT32_MAX || reduction <= 0 ||
      reduction > kMaximumOnlineReductionElements)
    return failure();
  if (outputShape.size() < 2)
    return failure();
  FailureOr<int64_t> sharedRows =
      scheduleElementCount(ArrayRef<int64_t>(outputShape).drop_back(2));
  FailureOr<int64_t> localRows =
      scheduleElementCount(ArrayRef<int64_t>(outputShape).take_back(2));
  if (failed(sharedRows) || failed(localRows) || *sharedRows > UINT32_MAX ||
      *localRows > UINT32_MAX ||
      *sharedRows >
          UINT32_MAX / ceilDivPositive(*localRows, kOnlineWarpsPerBlock))
    return failure();
  SmallVector<int64_t> tile(domain.size(), 1);
  tile[reductionAxis] = reduction;
  for (int64_t axis : vectorAxes)
    tile[axis] = domain[axis];
  const bool onlineMma = task.getOnlineMma().value_or(false);
  const int64_t rowsPerBlock =
      onlineMma ? kOnlineMmaRowsPerBlock : kOnlineWarpsPerBlock;
  return ScheduledGeometry{
      {*sharedRows * ceilDivPositive(*localRows, rowsPerBlock)},
      std::move(tile),
      1,
      kOnlineWarpsPerBlock * kOnlineWarpSize,
      kOnlineWarpSize,
      onlineMma ? kOnlineMmaSharedMemoryBytes
                : kGenericSimt32MaximumScratchBytes,
      reductionAxis};
}

FailureOr<ScheduledGeometry>
deriveReverseStateGeometry(MaterializationTaskOp task) {
  constexpr int64_t kThreads = 384;
  constexpr int64_t kOwnerAdjointTile = 64;
  constexpr int64_t kStreamedAdjointTile = 128;
  constexpr int64_t kOwnerStreamTile = 128;
  constexpr int64_t kAdjointStreamTile = 64;
  constexpr int64_t kSupportedVectorElements = 128;
  // The SM90 D=128 reverse schedule keeps two streamed Q/dO stages, K/V,
  // two dS stages, the query-gradient exchange, and row state in shared
  // memory.  Keep this in sync with the independently verified schedule
  // geometry in ShuttleDialect.cc.
  constexpr int64_t kScratchBytes = 194 * 1024;
  ArrayRef<int64_t> domain = task.getDomainShape();
  ArrayRef<int64_t> reductions = task.getReductionDimensions();
  auto ownerDimensions =
      structuralDimensions(task, "composition_owner_dimensions");
  auto streamDimensions =
      structuralDimensions(task, "composition_stream_dimensions");
  auto vectorDimensions = task.getOnlineVectorDimensionsAttr();
  if (!isReverseStateTask(task) || domain.empty() || domain.size() > 8 ||
      llvm::any_of(domain, [](int64_t extent) { return extent <= 0; }) ||
      reductions.empty() || !ownerDimensions || ownerDimensions.empty() ||
      !vectorDimensions || vectorDimensions.empty() || !streamDimensions ||
      streamDimensions.asArrayRef() != reductions)
    return failure();
  const bool hasStateInputs = task->hasAttr("composition_state_inputs");
  const bool hasStateOutputs = task->hasAttr("composition_state_outputs");
  const bool forward = !hasStateInputs && hasStateOutputs;
  const bool ownerAdjoint = hasStateInputs && hasStateOutputs;
  const bool streamedAdjoint = hasStateInputs && !hasStateOutputs;
  if ((!forward && !ownerAdjoint && !streamedAdjoint) ||
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
      failed(addDimensions(vectorDimensions.asArrayRef(), true)) ||
      failed(addDimensions(reductions, false)) ||
      partitionedDimensions.size() != domain.size() ||
      vectorElements != kSupportedVectorElements)
    return failure();

  const int64_t owner = ownerDimensions.asArrayRef().back();
  const int64_t finalStream = reductions.back();
  SmallVector<int64_t> tile(domain.size(), 1);
  for (int64_t axis : vectorDimensions.asArrayRef())
    tile[axis] = domain[axis];
  for (int64_t axis : ArrayRef<int64_t>(reductions).drop_back())
    tile[axis] = domain[axis];
  tile[owner] = std::min(domain[owner], streamedAdjoint ? kStreamedAdjointTile
                                                        : kOwnerAdjointTile);
  tile[finalStream] =
      std::min(domain[finalStream],
               streamedAdjoint ? kAdjointStreamTile : kOwnerStreamTile);

  int64_t grid = 1;
  for (int64_t axis : ownerDimensions.asArrayRef().drop_back()) {
    if (grid > std::numeric_limits<int64_t>::max() / domain[axis])
      return failure();
    grid *= domain[axis];
  }
  const int64_t rowTiles = ceilDivPositive(domain[owner], tile[owner]);
  if (grid > std::numeric_limits<int64_t>::max() / rowTiles)
    return failure();
  grid *= rowTiles;
  const int64_t serialTiles =
      ceilDivPositive(domain[finalStream], tile[finalStream]);
  return ScheduledGeometry{{grid}, std::move(tile), serialTiles, kThreads,
                           32,     kScratchBytes,   finalStream};
}

FailureOr<ScheduledGeometry>
deriveContractRegionGeometry(MaterializationTaskOp task,
                             NumericalPolicy policy) {
  auto strategy = task->getAttrOfType<StringAttr>("contract_region_strategy");
  auto cost = task->getAttrOfType<DictionaryAttr>("contract_region_cost");
  if (!strategy || !cost || task.getKind() != MaterializationTaskKind::Contract)
    return failure();
  auto rows = cost.getAs<IntegerAttr>("rows");
  auto reduction = cost.getAs<IntegerAttr>("reduction");
  auto tile = cost.getAs<IntegerAttr>("output_tile");
  auto tiles = cost.getAs<IntegerAttr>("output_tiles");
  if (!rows || !reduction || !tile || !tiles || rows.getInt() <= 0 ||
      reduction.getInt() <= 0 || tile.getInt() <= 0 || tiles.getInt() <= 0)
    return failure();
  const bool fast = policy == NumericalPolicy::Fast;
  const bool inlineFold = strategy.getValue() == "inline_fold";
  const bool compactStat = strategy.getValue() == "compact_stat";
  if (!inlineFold && !compactStat)
    return failure();
  const int64_t threads =
      fast ? tile.getInt() * 32 : scheduleThreads(tile.getInt());
  const int64_t scratch = inlineFold
                              ? scheduleThreads(reduction.getInt()) *
                                    static_cast<int64_t>(sizeof(float))
                              : std::min<int64_t>(reduction.getInt(), 256) * 2;
  if (threads <= 0 || threads > 1024 || scratch < 0 ||
      scratch > kGenericSimt32MaximumScratchBytes)
    return failure();
  return ScheduledGeometry{
      {rows.getInt(), tiles.getInt()},
      {tile.getInt()},
      fast ? ceilDivPositive(reduction.getInt(), 32) : reduction.getInt(),
      threads,
      32,
      scratch,
      static_cast<int64_t>(task.getDomainShape().size()) - 1};
}
