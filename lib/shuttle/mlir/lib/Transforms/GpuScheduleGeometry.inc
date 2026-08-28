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
};

constexpr llvm::StringLiteral kOnlineStateProtocol =
    "online_normalized_exponential_weighted_contract";

bool isOnlineStateTask(MaterializationTaskOp task) {
  auto kind = task->getAttrOfType<StringAttr>("composition_kind");
  return kind && kind.getValue() == kOnlineStateProtocol;
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
                                                  NumericalPolicy policy) {
  constexpr int64_t kSubgroup = 32;
  constexpr int64_t kMaxThreads = 256;
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
  FailureOr<int64_t> reduction = rowFold
                                     ? FailureOr<int64_t>(domain.back())
                                     : scheduleElementCount(domain.drop_back());
  FailureOr<int64_t> outputs = rowFold
                                   ? scheduleElementCount(domain.drop_back())
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
  if (!isOnlineStateTask(task) || domain.empty() || domain.size() > 8 ||
      reductions.size() != 1 || reductions.front() < 0 ||
      reductions.front() >= static_cast<int64_t>(domain.size()) ||
      !vectorDimensions || vectorDimensions->empty())
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
  SmallVector<int64_t> outputShape;
  for (auto [axis, extent] : llvm::enumerate(domain))
    if (static_cast<int64_t>(axis) != reductionAxis &&
        !vectorAxes.contains(axis))
      outputShape.push_back(extent);
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
                : static_cast<int64_t>(kMaximumGpuDynamicSharedMemoryBytes),
      reductionAxis};
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
      scratch > kMaximumGpuDynamicSharedMemoryBytes)
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
