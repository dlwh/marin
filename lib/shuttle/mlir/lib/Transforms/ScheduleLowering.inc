// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "GpuScheduleGeometry.inc"

FailureOr<SmallVector<Operation *>> compositionSourceOperations(
    MaterializationTaskOp task,
    const llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations) {
  auto sources = task->getAttrOfType<ArrayAttr>("composition_sources");
  if (!sources)
    return failure();
  SmallVector<Operation *> result;
  for (Attribute source : sources) {
    Operation *operation = sourceOperations.lookup(source);
    if (!operation)
      return failure();
    result.push_back(operation);
  }
  return result;
}

FailureOr<SmallVector<Operation *>>
compositionSourceOperations(MaterializationTaskOp task) {
  llvm::SmallDenseMap<Attribute, Operation *> sourceOperations;
  llvm::SmallDenseSet<Attribute> duplicateSources;
  task->getParentOfType<ModuleOp>().walk([&](Operation *operation) {
    Attribute source;
    if (auto map = dyn_cast<MapOp>(operation))
      source = map.getSource();
    else if (auto relation = dyn_cast<RelationOp>(operation))
      source = relation.getSource();
    else if (auto fold = dyn_cast<FoldOp>(operation))
      source = fold.getSource();
    else if (auto contract = dyn_cast<ContractOp>(operation))
      source = contract.getSource();
    if (source && !sourceOperations.try_emplace(source, operation).second)
      duplicateSources.insert(source);
  });
  if (!duplicateSources.empty())
    return failure();
  return compositionSourceOperations(task, sourceOperations);
}

bool relationExpressionReadsEntireValue(
    Value value, Value ancestor,
    const llvm::SmallDenseSet<Operation *> &composition) {
  if (value == ancestor)
    return true;
  auto relation = value.getDefiningOp<RelationOp>();
  if (!relation || !composition.contains(relation) ||
      relation.getNumResults() != 1)
    return false;
  if (relation.getSemantics() == RelationSemantics::BroadcastInDim) {
    if (relation.getInputs().size() != 1)
      return false;
    auto input = dyn_cast<RankedTensorType>(relation.getInputs()[0].getType());
    auto output = dyn_cast<RankedTensorType>(relation.getResult(0).getType());
    if (!input || !output || relation.getParameters().size() != input.getRank())
      return false;
    for (auto [inputAxis, parameter] :
         llvm::enumerate(relation.getParameters())) {
      auto outputAxis = dyn_cast<IntegerAttr>(parameter);
      if (!outputAxis || outputAxis.getInt() < 0 ||
          outputAxis.getInt() >= output.getRank() ||
          input.getDimSize(inputAxis) != output.getDimSize(outputAxis.getInt()))
        return false;
    }
  }
  return llvm::any_of(relation.getInputs(), [&](Value input) {
    return relationExpressionReadsEntireValue(input, ancestor, composition);
  });
}

FailureOr<AffineMap>
mapCoordinatesToAncestor(Value value, Value ancestor,
                         const llvm::SmallDenseSet<Operation *> &sources) {
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  if (!valueType)
    return failure();
  if (value == ancestor)
    return AffineMap::getMultiDimIdentityMap(valueType.getRank(),
                                             value.getContext());
  auto map = value.getDefiningOp<MapOp>();
  if (!map || !sources.contains(map) || map.getNumResults() != 1 ||
      map.getResult(0) != value ||
      map.getIndexingMaps().size() != map.getInputs().size() + 1)
    return failure();
  auto resultMap =
      dyn_cast<AffineMapAttr>(map.getIndexingMaps().getValue().back());
  if (!resultMap || !resultMap.getValue().isIdentity())
    return failure();

  std::optional<AffineMap> coordinates;
  for (auto [ordinal, input] : llvm::enumerate(map.getInputs())) {
    if (!valueDependsOnValue(input, ancestor))
      continue;
    auto inputMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[ordinal]);
    FailureOr<AffineMap> upstream =
        inputMap ? mapCoordinatesToAncestor(input, ancestor, sources)
                 : FailureOr<AffineMap>(failure());
    if (failed(upstream) ||
        upstream->getNumDims() != inputMap.getValue().getNumResults())
      return failure();
    AffineMap candidate = upstream->compose(inputMap.getValue());
    if (coordinates && *coordinates != candidate)
      return failure();
    coordinates = candidate;
  }
  if (!coordinates)
    return failure();
  return *coordinates;
}

struct ScheduledTaskRelation {
  SmallVector<int64_t> domain;
  std::optional<IntegerSet> predicate;
  AffineMap producerIndices;
  AffineMap consumerIndices;
  int64_t storageCapacity;
  AffineMap storageSlot;
  AffineMap generationIndex;
};

struct OrderedIndexConstraint {
  int64_t lesserAxis;
  int64_t greaterAxis;
  bool inclusive;
};

struct ScheduledTaskBufferAccess {
  SmallVector<int64_t> buffers;
  SmallVector<int64_t> domain;
  std::optional<IntegerSet> predicate;
  AffineMap taskIndices;
  AffineMap bufferIndices;
};

struct ScheduledTaskBufferAccessPair {
  ScheduledTaskBufferAccess producer;
  ScheduledTaskBufferAccess consumer;
};

FailureOr<Operation *> boundAlgebraOperation(MaterializationTaskOp task);

FailureOr<std::optional<ScheduledTaskRelation>>
deriveAffineMappedLogicalTaskRelation(
    ScheduleTaskOp producer, ScheduleTaskOp consumer,
    MaterializationTaskOp producerSource, MaterializationTaskOp consumerSource,
    SmallVectorImpl<MaterializationBufferOp> &buffers,
    ArrayRef<int64_t> sharedBuffers) {
  FailureOr<Operation *> bound = boundAlgebraOperation(consumerSource);
  auto map = succeeded(bound) ? dyn_cast<MapOp>(*bound) : MapOp{};
  if (!map || producer.getGridShape().size() != 1 ||
      consumer.getGridShape().size() != 1 ||
      map.getIndexingMaps().size() != map.getNumOperands() + 1)
    return std::optional<ScheduledTaskRelation>{};

  ArrayRef<int64_t> domain = consumerSource.getDomainShape();
  FailureOr<int64_t> domainElements = scheduleElementCount(domain);
  constexpr int64_t kMaximumLogicalEventCoordinates = 1 << 20;
  if (failed(domainElements) ||
      *domainElements > kMaximumLogicalEventCoordinates)
    return failure();
  FailureOr<int64_t> producerInstances =
      scheduleElementCount(producer.getGridShape());
  FailureOr<int64_t> consumerInstances =
      scheduleElementCount(consumer.getGridShape());
  if (failed(producerInstances) || failed(consumerInstances))
    return failure();

  MLIRContext *context = producer.getContext();
  AffineExpr consumerLinear = getAffineConstantExpr(0, context);
  for (auto [axis, extent] : llvm::enumerate(domain))
    consumerLinear = consumerLinear * extent + getAffineDimExpr(axis, context);
  const int64_t consumerChunk =
      ceilDivPositive(*domainElements, *consumerInstances);
  AffineExpr consumerIndex = consumerLinear.floorDiv(consumerChunk);

  std::optional<AffineExpr> producerIndex;
  for (int64_t sharedBuffer : sharedBuffers) {
    auto inputPosition =
        llvm::find(consumerSource.getInputBuffers(), sharedBuffer);
    if (inputPosition == consumerSource.getInputBuffers().end() ||
        sharedBuffer < 0 ||
        sharedBuffer >= static_cast<int64_t>(buffers.size()))
      return failure();
    const int64_t operand =
        std::distance(consumerSource.getInputBuffers().begin(), inputPosition);
    auto indexing = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[operand]);
    auto bufferType =
        dyn_cast<RankedTensorType>(buffers[sharedBuffer].getTensorType());
    if (!indexing || !bufferType || !bufferType.hasStaticShape() ||
        indexing.getValue().getNumDims() != domain.size() ||
        indexing.getValue().getNumSymbols() != 0 ||
        indexing.getValue().getNumResults() != bufferType.getRank())
      return failure();
    FailureOr<int64_t> bufferElements =
        scheduleElementCount(bufferType.getShape());
    if (failed(bufferElements))
      return failure();
    const int64_t producerChunk =
        ceilDivPositive(*bufferElements, *producerInstances);
    AffineExpr inputLinear = getAffineConstantExpr(0, context);
    for (auto [axis, extent] : llvm::enumerate(bufferType.getShape()))
      inputLinear = inputLinear * extent + indexing.getValue().getResult(axis);
    AffineExpr candidate = inputLinear.floorDiv(producerChunk);
    if (producerIndex && *producerIndex != candidate)
      return failure();
    producerIndex = candidate;
  }
  if (!producerIndex)
    return failure();

  AffineExpr storageSlot = getAffineConstantExpr(0, context);
  for (auto [axis, extent] : llvm::enumerate(domain))
    storageSlot = storageSlot * extent + getAffineDimExpr(axis, context);
  return std::optional<ScheduledTaskRelation>(ScheduledTaskRelation{
      SmallVector<int64_t>(domain.begin(), domain.end()), std::nullopt,
      AffineMap::get(domain.size(), 0, *producerIndex, context),
      AffineMap::get(domain.size(), 0, consumerIndex, context), *domainElements,
      AffineMap::get(domain.size(), 0, storageSlot, context),
      AffineMap::get(domain.size(), 0, getAffineConstantExpr(0, context),
                     context)});
}

FailureOr<std::optional<ScheduledTaskRelation>> deriveClosedLogicalTaskRelation(
    ScheduleTaskOp producer, ScheduleTaskOp consumer,
    MaterializationTaskOp producerSource, MaterializationTaskOp consumerSource,
    SmallVectorImpl<MaterializationBufferOp> &buffers) {
  const bool logicalEdge =
      producerSource.getKind() == MaterializationTaskKind::Relation ||
      producerSource.getKind() == MaterializationTaskKind::Selection ||
      consumerSource.getKind() == MaterializationTaskKind::Relation ||
      consumerSource.getKind() == MaterializationTaskKind::Selection;
  if (!logicalEdge)
    return std::optional<ScheduledTaskRelation>{};

  SmallVector<int64_t> sharedBuffers;
  for (int64_t output : producerSource.getOutputBuffers())
    if (llvm::is_contained(consumerSource.getInputBuffers(), output))
      sharedBuffers.push_back(output);
  if (sharedBuffers.empty())
    return failure();

  bool composedRelationReadsWholeSharedBuffers = false;
  bool recomputedRelationReadsWholeSharedBuffers = false;
  auto compositionKind =
      consumerSource->getAttrOfType<StringAttr>("composition_kind");
  if (compositionKind && compositionKind.getValue() == "relation_gather" &&
      consumer.getGridShape() == ArrayRef<int64_t>({1})) {
    SmallVector<Operation *> operations;
    bool uniqueSources = true;
    auto sources =
        consumerSource->getAttrOfType<ArrayAttr>("composition_sources");
    if (sources) {
      for (Attribute source : sources) {
        RelationOp found;
        consumerSource->getParentOfType<ModuleOp>().walk([&](RelationOp op) {
          if (op.getSource() != source)
            return;
          if (found)
            uniqueSources = false;
          else
            found = op;
        });
        if (!found) {
          uniqueSources = false;
          break;
        }
        operations.push_back(found);
      }
    }
    if (uniqueSources && !operations.empty()) {
      llvm::SmallDenseSet<Operation *> composition(operations.begin(),
                                                   operations.end());
      auto gather = dyn_cast<RelationOp>(operations.back());
      SmallVector<Value> boundaryValues;
      llvm::SmallDenseSet<Value> seen;
      for (Operation *operation : operations) {
        for (Value operand : operation->getOperands()) {
          if (composition.contains(operand.getDefiningOp()) ||
              !seen.insert(operand).second)
            continue;
          boundaryValues.push_back(operand);
        }
      }
      composedRelationReadsWholeSharedBuffers =
          gather && gather.getSemantics() == RelationSemantics::Gather &&
          boundaryValues.size() == consumerSource.getInputBuffers().size() &&
          llvm::all_of(sharedBuffers, [&](int64_t sharedBuffer) {
            auto position =
                llvm::find(consumerSource.getInputBuffers(), sharedBuffer);
            if (position == consumerSource.getInputBuffers().end())
              return false;
            const size_t input = std::distance(
                consumerSource.getInputBuffers().begin(), position);
            return relationExpressionReadsEntireValue(
                gather.getInputs()[1], boundaryValues[input], composition);
          });
    }
  }

  if (compositionKind &&
      compositionKind.getValue() == "indexed_accumulation") {
    auto recomputed = consumerSource->getAttrOfType<ArrayAttr>(
        "composition_recomputed_sources");
    SmallVector<Operation *> relations;
    bool uniqueSources = recomputed != nullptr;
    if (uniqueSources) {
      for (Attribute source : recomputed) {
        RelationOp found;
        consumerSource->getParentOfType<ModuleOp>().walk([&](RelationOp op) {
          if (op.getSource() != source)
            return;
          if (found)
            uniqueSources = false;
          else
            found = op;
        });
        if (!found) {
          uniqueSources = false;
          break;
        }
        relations.push_back(found);
      }
    }
    RelationOp contributions;
    if (uniqueSources) {
      consumerSource->getParentOfType<ModuleOp>().walk([&](RelationOp op) {
        if (op.getSource() != consumerSource.getSource())
          return;
        auto member = sourceExpansionMember(op, "member");
        auto size = sourceExpansionMember(op, "size");
        if (member == std::optional<int64_t>(0) &&
            size == std::optional<int64_t>(4))
          contributions = op;
      });
    }
    if (contributions && !relations.empty() &&
        contributions.getSemantics() ==
            RelationSemantics::IndexedContributions &&
        contributions.getInputs().size() == 2 &&
        relations.back()->getResult(0) == contributions.getInputs()[1]) {
      llvm::SmallDenseSet<Operation *> relationSet(relations.begin(),
                                                   relations.end());
      SmallVector<Value> boundaries;
      llvm::SmallDenseSet<Value> seen;
      for (Operation *operation : relations)
        for (Value operand : operation->getOperands())
          if (!relationSet.contains(operand.getDefiningOp()) &&
              seen.insert(operand).second)
            boundaries.push_back(operand);
      recomputedRelationReadsWholeSharedBuffers =
          consumerSource.getInputBuffers().size() == boundaries.size() + 2 &&
          llvm::all_of(sharedBuffers, [&](int64_t sharedBuffer) {
            auto position =
                llvm::find(consumerSource.getInputBuffers(), sharedBuffer);
            if (position == consumerSource.getInputBuffers().end())
              return false;
            size_t input = std::distance(
                consumerSource.getInputBuffers().begin(), position);
            return input >= 2 &&
                   relationExpressionReadsEntireValue(
                       relations.back()->getResult(0), boundaries[input - 2],
                       relationSet);
          });
    }
  }

  auto producerComposition =
      producerSource->getAttrOfType<StringAttr>("composition_kind");
  if (producerComposition &&
      producerComposition.getValue() == "map_contract" &&
      sharedBuffers.size() == 1 && producer.getGridShape().size() == 1 &&
      consumer.getGridShape().size() == 1 &&
      producer.getTileShape().size() == 1) {
    auto shared = dyn_cast<RankedTensorType>(
        buffers[sharedBuffers.front()].getTensorType());
    FailureOr<int64_t> elements =
        shared ? scheduleElementCount(shared.getShape())
               : FailureOr<int64_t>(failure());
    FailureOr<int64_t> consumerTile =
        scheduleElementCount(consumer.getTileShape());
    const int64_t producerTile = producer.getTileShape().front();
    if (succeeded(elements) && succeeded(consumerTile) && shared &&
        shared.hasStaticShape() && producerTile > 0 && *consumerTile > 0 &&
        consumerSource.getDomainShape() == shared.getShape() &&
        producer.getGridShape().front() ==
            ceilDivPositive(*elements, producerTile) &&
        consumer.getGridShape().front() ==
            ceilDivPositive(*elements, *consumerTile)) {
      MLIRContext *context = producer.getContext();
      AffineExpr linear = getAffineConstantExpr(0, context);
      for (auto [axis, extent] : llvm::enumerate(shared.getShape()))
        linear = linear * extent + getAffineDimExpr(axis, context);
      return std::optional<ScheduledTaskRelation>(ScheduledTaskRelation{
          SmallVector<int64_t>(shared.getShape().begin(),
                               shared.getShape().end()),
          std::nullopt,
          AffineMap::get(shared.getRank(), 0, linear.floorDiv(producerTile),
                         context),
          AffineMap::get(shared.getRank(), 0, linear.floorDiv(*consumerTile),
                         context),
          *elements,
          AffineMap::get(shared.getRank(), 0, linear, context),
          AffineMap::get(shared.getRank(), 0,
                         getAffineConstantExpr(0, context), context)});
    }
  }

  if (producerSource.getKind() == MaterializationTaskKind::Relation &&
      consumerSource.getKind() == MaterializationTaskKind::Contract &&
      producer.getGridShape().size() == 1 &&
      consumer.getGridShape().size() == 1 &&
      consumer.getTileShape().size() == 1 &&
      consumerSource.getReductionDimensions().size() == 1) {
    auto shared = dyn_cast<RankedTensorType>(
        buffers[sharedBuffers.front()].getTensorType());
    FailureOr<int64_t> elements = shared
                                      ? scheduleElementCount(shared.getShape())
                                      : FailureOr<int64_t>(failure());
    const int64_t reductionAxis =
        consumerSource.getReductionDimensions().front();
    if (succeeded(elements) && shared &&
        shared.getShape() == consumerSource.getDomainShape() &&
        reductionAxis == static_cast<int64_t>(shared.getRank()) - 1 &&
        *elements % producer.getGridShape().front() == 0) {
      const int64_t producerChunk = *elements / producer.getGridShape().front();
      const int64_t consumerChunk =
          consumer.getTileShape().front() * shared.getShape().back();
      if (consumerChunk > 0 && producerChunk % consumerChunk == 0) {
        MLIRContext *context = producer.getContext();
        AffineExpr consumerIndex = getAffineDimExpr(0, context);
        AffineExpr producerIndex =
            (consumerIndex * consumerChunk).floorDiv(producerChunk);
        return std::optional<ScheduledTaskRelation>(ScheduledTaskRelation{
            SmallVector<int64_t>{consumer.getGridShape().front()}, std::nullopt,
            AffineMap::get(1, 0, producerIndex, context),
            AffineMap::get(1, 0, consumerIndex, context),
            consumer.getGridShape().front(),
            AffineMap::get(1, 0, consumerIndex, context),
            AffineMap::get(1, 0, getAffineConstantExpr(0, context), context)});
      }
    }
  }

  FailureOr<std::optional<ScheduledTaskRelation>> mapped =
      deriveAffineMappedLogicalTaskRelation(producer, consumer, producerSource,
                                            consumerSource, buffers,
                                            sharedBuffers);
  if (failed(mapped))
    return failure();
  if (*mapped)
    return *mapped;

  FailureOr<int64_t> producerInstances =
      scheduleElementCount(producer.getGridShape());
  FailureOr<int64_t> consumerInstances =
      scheduleElementCount(consumer.getGridShape());
  if (failed(producerInstances) || failed(consumerInstances))
    return failure();

  enum class ClosedRelationKind {
    OneProducer,
    OneConsumer,
    Identity,
    Cartesian
  };
  std::optional<ClosedRelationKind> kind;
  if (*producerInstances == 1) {
    kind = ClosedRelationKind::OneProducer;
  } else {
    FailureOr<int64_t> consumerElements =
        scheduleElementCount(consumerSource.getDomainShape());
    const bool consumerReadsWholeSharedBuffers =
        succeeded(consumerElements) &&
        llvm::all_of(sharedBuffers, [&](int64_t buffer) {
          if (buffer < 0 || buffer >= static_cast<int64_t>(buffers.size()))
            return false;
          auto type =
              dyn_cast<RankedTensorType>(buffers[buffer].getTensorType());
          FailureOr<int64_t> elements =
              type ? scheduleElementCount(type.getShape())
                   : FailureOr<int64_t>(failure());
          return succeeded(elements) && *elements == *consumerElements;
        });
    if (recomputedRelationReadsWholeSharedBuffers) {
      kind = ClosedRelationKind::Cartesian;
    } else if (*consumerInstances == 1 &&
               (consumerReadsWholeSharedBuffers ||
                composedRelationReadsWholeSharedBuffers)) {
      kind = ClosedRelationKind::OneConsumer;
    } else if (producer.getGridShape() == consumer.getGridShape() &&
               producerSource.getDomainShape() ==
                   consumerSource.getDomainShape() &&
               llvm::all_of(sharedBuffers, [&](int64_t buffer) {
                 if (buffer < 0 ||
                     buffer >= static_cast<int64_t>(buffers.size()))
                   return false;
                 auto type = dyn_cast<RankedTensorType>(
                     buffers[buffer].getTensorType());
                 return type &&
                        type.getShape() == producerSource.getDomainShape();
               })) {
      kind = ClosedRelationKind::Identity;
    }
  }
  if (!kind)
    return failure();

  MLIRContext *context = producer.getContext();
  SmallVector<int64_t> domain;
  SmallVector<AffineExpr> producerIndices;
  SmallVector<AffineExpr> consumerIndices;
  if (*kind == ClosedRelationKind::OneProducer) {
    llvm::append_range(domain, consumer.getGridShape());
    producerIndices.assign(producer.getGridShape().size(),
                           getAffineConstantExpr(0, context));
    for (int64_t axis = 0;
         axis < static_cast<int64_t>(consumer.getGridShape().size()); ++axis)
      consumerIndices.push_back(getAffineDimExpr(axis, context));
  } else if (*kind == ClosedRelationKind::OneConsumer) {
    llvm::append_range(domain, producer.getGridShape());
    for (int64_t axis = 0;
         axis < static_cast<int64_t>(producer.getGridShape().size()); ++axis)
      producerIndices.push_back(getAffineDimExpr(axis, context));
    consumerIndices.assign(consumer.getGridShape().size(),
                           getAffineConstantExpr(0, context));
  } else if (*kind == ClosedRelationKind::Identity) {
    llvm::append_range(domain, producer.getGridShape());
    for (int64_t axis = 0; axis < static_cast<int64_t>(domain.size()); ++axis) {
      producerIndices.push_back(getAffineDimExpr(axis, context));
      consumerIndices.push_back(getAffineDimExpr(axis, context));
    }
  } else {
    llvm::append_range(domain, producer.getGridShape());
    llvm::append_range(domain, consumer.getGridShape());
    for (int64_t axis = 0;
         axis < static_cast<int64_t>(producer.getGridShape().size()); ++axis)
      producerIndices.push_back(getAffineDimExpr(axis, context));
    for (int64_t axis = 0;
         axis < static_cast<int64_t>(consumer.getGridShape().size()); ++axis)
      consumerIndices.push_back(getAffineDimExpr(
          producer.getGridShape().size() + axis, context));
  }
  FailureOr<int64_t> storageCapacity = scheduleElementCount(domain);
  if (failed(storageCapacity))
    return failure();
  AffineExpr storageSlot = getAffineConstantExpr(0, context);
  for (auto [axis, extent] : llvm::enumerate(domain))
    storageSlot = storageSlot * extent + getAffineDimExpr(axis, context);
  return std::optional<ScheduledTaskRelation>(ScheduledTaskRelation{
      domain, std::nullopt,
      AffineMap::get(domain.size(), 0, producerIndices, context),
      AffineMap::get(domain.size(), 0, consumerIndices, context),
      *storageCapacity, AffineMap::get(domain.size(), 0, storageSlot, context),
      AffineMap::get(domain.size(), 0, getAffineConstantExpr(0, context),
                     context)});
}

FailureOr<std::optional<ScheduledTaskBufferAccessPair>>
deriveFoldStateBufferAccesses(
    ScheduleTaskOp producer, ScheduleTaskOp consumer,
    MaterializationTaskOp producerSource, MaterializationTaskOp consumerSource,
    SmallVectorImpl<MaterializationBufferOp> &buffers) {
  if (!producerSource->hasAttr("composition_sources") ||
      !consumerSource->hasAttr("composition_sources"))
    return std::optional<ScheduledTaskBufferAccessPair>{};

  // Only a Fold task can define the compact row state recognized below.
  // Other composed tasks can still have ordinary SSA dependencies; those use
  // the generic bounded task relation rather than this Fold-state refinement.
  if (producerSource.getKind() != MaterializationTaskKind::Fold)
    return std::optional<ScheduledTaskBufferAccessPair>{};

  FailureOr<SmallVector<Operation *>> producerSources =
      compositionSourceOperations(producerSource);
  FailureOr<SmallVector<Operation *>> consumerSources =
      compositionSourceOperations(consumerSource);
  if (failed(producerSources) || failed(consumerSources) ||
      producerSources->empty() || consumerSources->empty())
    return failure();
  auto fold = dyn_cast<FoldOp>(producerSources->back());
  if (!fold || !isClosedMultiStatePlanningFold(fold) ||
      fold.getNumResults() == 0)
    return failure();

  llvm::SmallDenseSet<Operation *> consumerSet(consumerSources->begin(),
                                               consumerSources->end());
  SmallVector<Value> cutValues;
  for (Operation *operation : *producerSources)
    for (Value result : operation->getResults())
      if (llvm::any_of(result.getUsers(), [&](Operation *user) {
            return consumerSet.contains(user);
          }))
        cutValues.push_back(result);
  if (cutValues.size() != producerSource.getOutputBuffers().size())
    return failure();
  DenseMap<Value, int64_t> cutBuffer;
  for (auto [value, buffer] :
       llvm::zip_equal(cutValues, producerSource.getOutputBuffers()))
    cutBuffer[value] = buffer;

  RankedTensorType compactType;
  for (Value result : fold.getResults()) {
    auto type = dyn_cast<RankedTensorType>(result.getType());
    auto found = cutBuffer.find(result);
    if (!type || !type.hasStaticShape() || found == cutBuffer.end() ||
        !llvm::is_contained(consumerSource.getInputBuffers(), found->second) ||
        (compactType && compactType.getShape() != type.getShape()))
      return failure();
    compactType = type;
  }
  FailureOr<int64_t> producerRows =
      compactType ? scheduleElementCount(compactType.getShape())
                  : FailureOr<int64_t>(failure());
  if (failed(producerRows) ||
      producer.getGridShape() != ArrayRef<int64_t>({*producerRows}))
    return failure();

  ContractOp contract;
  for (Operation *operation : *consumerSources) {
    auto candidate = dyn_cast<ContractOp>(operation);
    if (!candidate)
      continue;
    if (contract)
      return failure();
    contract = candidate;
  }
  Operation *terminal = consumerSources->back();
  if (!contract || terminal->getNumResults() != 1 ||
      consumerSource.getOutputBuffers().size() != 1 ||
      !valueDependsOnOperation(terminal->getResult(0), contract))
    return failure();
  const int64_t outputBuffer = consumerSource.getOutputBuffers().front();
  if (outputBuffer < 0 || outputBuffer >= static_cast<int64_t>(buffers.size()))
    return failure();
  Value publicValue = terminal->getResult(0);
  auto outputType = dyn_cast<RankedTensorType>(publicValue.getType());
  if (!outputType || !outputType.hasStaticShape() ||
      buffers[outputBuffer].getTensorType() != outputType ||
      outputType.getRank() <= compactType.getRank() ||
      outputType.getShape().take_front(compactType.getRank()) !=
          compactType.getShape())
    return failure();

  std::optional<AffineMap> foldCoordinates;
  for (auto [operandOrdinal, operand] : llvm::enumerate(contract.getInputs())) {
    auto operandMap =
        dyn_cast<AffineMapAttr>(contract.getIndexingMaps()[operandOrdinal]);
    if (!operandMap)
      return failure();
    for (Value result : fold.getResults()) {
      FailureOr<AffineMap> operandCoordinates =
          mapCoordinatesToAncestor(operand, result, consumerSet);
      if (failed(operandCoordinates))
        continue;
      if (operandCoordinates->getNumDims() !=
          operandMap.getValue().getNumResults())
        return failure();
      AffineMap candidate = operandCoordinates->compose(operandMap.getValue());
      if (foldCoordinates && *foldCoordinates != candidate)
        return failure();
      foldCoordinates = candidate;
    }
  }
  FailureOr<AffineMap> contractOutputCoordinates =
      mapCoordinatesToAncestor(publicValue, contract.getResult(0), consumerSet);
  FailureOr<SmallVector<int64_t>> contractDomain =
      contractDomainShape(contract);
  if (!foldCoordinates || failed(contractOutputCoordinates) ||
      failed(contractDomain) ||
      foldCoordinates->getNumDims() != contractDomain->size() ||
      contractOutputCoordinates->getNumResults() + 1 != contractDomain->size())
    return failure();

  MLIRContext *context = producer.getContext();
  SmallVector<AffineExpr> publicToContractDomain(
      contractOutputCoordinates->getResults());
  publicToContractDomain.push_back(
      getAffineDimExpr(outputType.getRank(), context));
  AffineMap foldFromOutputAndReduction =
      foldCoordinates->compose(AffineMap::get(outputType.getRank() + 1, 0,
                                              publicToContractDomain, context));
  SmallVector<AffineExpr> expectedFoldCoordinates;
  for (int64_t axis = 0; axis < compactType.getRank(); ++axis)
    expectedFoldCoordinates.push_back(getAffineDimExpr(axis, context));
  if (foldFromOutputAndReduction != AffineMap::get(outputType.getRank() + 1, 0,
                                                   expectedFoldCoordinates,
                                                   context))
    return failure();

  FailureOr<int64_t> outputElements =
      scheduleElementCount(outputType.getShape());
  if (failed(outputElements) || *outputElements % *producerRows != 0 ||
      consumer.getGridShape().size() != 1 ||
      consumer.getTileShape().size() != 1 ||
      consumer.getTileShape().front() <= 0)
    return failure();
  const int64_t outputsPerProducerRow = *outputElements / *producerRows;
  const int64_t outputsPerConsumerBlock = consumer.getTileShape().front();
  const int64_t consumerBlocks =
      ceilDivPositive(*outputElements, outputsPerConsumerBlock);
  if (consumer.getGridShape().front() != consumerBlocks)
    return failure();

  SmallVector<int64_t> sharedBuffers;
  for (int64_t buffer : producerSource.getOutputBuffers())
    if (llvm::is_contained(consumerSource.getInputBuffers(), buffer))
      sharedBuffers.push_back(buffer);
  if (sharedBuffers.empty())
    return failure();

  AffineExpr consumerBlock = getAffineDimExpr(0, context);
  AffineExpr localProducerRow = getAffineDimExpr(1, context);
  AffineExpr producerRow = (consumerBlock * outputsPerConsumerBlock)
                               .floorDiv(outputsPerProducerRow) +
                           localProducerRow;
  const int64_t alignmentPeriod =
      std::gcd(outputsPerProducerRow, outputsPerConsumerBlock);
  const int64_t unalignedConsumerSpan =
      outputsPerConsumerBlock - alignmentPeriod;
  const int64_t maximumProducerRowsPerConsumer =
      unalignedConsumerSpan == 0
          ? 1
          : ceilDivPositive(unalignedConsumerSpan, outputsPerProducerRow) + 1;
  IntegerSet consumerPredicate = IntegerSet::get(
      2, 0,
      {producerRow * outputsPerProducerRow + outputsPerProducerRow - 1 -
           consumerBlock * outputsPerConsumerBlock,
       consumerBlock * outputsPerConsumerBlock + outputsPerConsumerBlock - 1 -
           producerRow * outputsPerProducerRow,
       getAffineConstantExpr(*producerRows - 1, context) - producerRow},
      {false, false, false});
  return std::optional<ScheduledTaskBufferAccessPair>(
      ScheduledTaskBufferAccessPair{
          ScheduledTaskBufferAccess{
              sharedBuffers,
              {*producerRows},
              std::nullopt,
              AffineMap::get(1, 0, getAffineDimExpr(0, context), context),
              AffineMap::get(1, 0, getAffineDimExpr(0, context), context)},
          ScheduledTaskBufferAccess{
              sharedBuffers,
              {consumerBlocks, maximumProducerRowsPerConsumer},
              consumerPredicate,
              AffineMap::get(2, 0, consumerBlock, context),
              AffineMap::get(2, 0, producerRow, context)}});
}

FailureOr<ScheduledTaskRelation>
deriveRelationFromBufferAccesses(const ScheduledTaskBufferAccessPair &accesses,
                                 ScheduleTaskOp producer,
                                 ScheduleTaskOp consumer) {
  MLIRContext *context = producer.getContext();
  if (accesses.producer.buffers.empty() ||
      accesses.producer.buffers != accesses.consumer.buffers ||
      accesses.producer.domain != producer.getGridShape() ||
      accesses.producer.predicate ||
      accesses.producer.taskIndices != accesses.producer.bufferIndices ||
      accesses.producer.taskIndices !=
          AffineMap::get(accesses.producer.domain.size(), 0,
                         getAffineDimExpr(0, context), context) ||
      accesses.consumer.taskIndices.getNumResults() !=
          consumer.getGridShape().size() ||
      accesses.consumer.bufferIndices.getNumResults() !=
          producer.getGridShape().size())
    return failure();

  FailureOr<int64_t> storageCapacity =
      scheduleElementCount(accesses.consumer.domain);
  if (failed(storageCapacity))
    return failure();
  AffineExpr storageSlot = getAffineConstantExpr(0, context);
  for (auto [axis, extent] : llvm::enumerate(accesses.consumer.domain))
    storageSlot = storageSlot * extent + getAffineDimExpr(axis, context);
  const int64_t rank = accesses.consumer.domain.size();
  return ScheduledTaskRelation{
      accesses.consumer.domain,
      accesses.consumer.predicate,
      accesses.consumer.bufferIndices,
      accesses.consumer.taskIndices,
      *storageCapacity,
      AffineMap::get(rank, 0, storageSlot, context),
      AffineMap::get(rank, 0, getAffineConstantExpr(0, context), context)};
}

FailureOr<int64_t> ownerPositionForIndexDomainAxis(ScheduleTaskOp task,
                                                   int64_t indexAxis,
                                                   int64_t commonRank) {
  auto shape = task.getCompositionIndexDomainShapeAttr();
  auto map = task.getCompositionIndexDomainMapAttr();
  auto owners = task.getCompositionOwnerDimensionsAttr();
  if (!shape || !map || !owners || indexAxis < 0 ||
      indexAxis >= static_cast<int64_t>(shape.size()) ||
      map.getValue().getNumSymbols() != 0 ||
      map.getValue().getNumDims() != task.getDomainShape().size() ||
      map.getValue().getNumResults() != shape.size())
    return failure();
  auto taskAxis = dyn_cast<AffineDimExpr>(map.getValue().getResult(indexAxis));
  if (!taskAxis)
    return failure();
  auto found = llvm::find(owners.asArrayRef(), taskAxis.getPosition());
  if (found == owners.asArrayRef().end())
    return failure();
  const int64_t ownerPosition =
      std::distance(owners.asArrayRef().begin(), found);
  if (ownerPosition < commonRank)
    return failure();
  return ownerPosition - commonRank;
}

bool hasSharedStateFlow(ScheduleTaskOp producer, ScheduleTaskOp consumer) {
  auto outputs =
      producer->getAttrOfType<DenseI64ArrayAttr>("composition_state_outputs");
  auto inputs =
      consumer->getAttrOfType<DenseI64ArrayAttr>("composition_state_inputs");
  if (!outputs || !inputs)
    return false;
  return llvm::any_of(outputs.asArrayRef(), [&](int64_t output) {
    return llvm::is_contained(inputs.asArrayRef(), output);
  });
}

FailureOr<SmallVector<int64_t>> ownerGridFactors(ScheduleTaskOp task) {
  auto ownerDimensions =
      task->getAttrOfType<DenseI64ArrayAttr>("composition_owner_dimensions");
  if (!ownerDimensions)
    return failure();
  ArrayRef<int64_t> domain = task.getDomainShape();
  ArrayRef<int64_t> tile = task.getTileShape();
  SmallVector<int64_t> factors;
  for (int64_t axis : ownerDimensions.asArrayRef()) {
    if (axis < 0 || axis >= static_cast<int64_t>(domain.size()) ||
        axis >= static_cast<int64_t>(tile.size()) || domain[axis] <= 0 ||
        tile[axis] <= 0)
      return failure();
    factors.push_back(ceilDivPositive(domain[axis], tile[axis]));
  }
  FailureOr<int64_t> factorCount = scheduleElementCount(factors);
  FailureOr<int64_t> gridCount = scheduleElementCount(task.getGridShape());
  if (failed(factorCount) || failed(gridCount) || *factorCount != *gridCount)
    return failure();
  return factors;
}

FailureOr<ScheduledTaskRelation>
deriveScheduledTaskRelation(ScheduleTaskOp producer, ScheduleTaskOp consumer,
                            MaterializationTaskOp producerSource,
                            MaterializationTaskOp consumerSource,
                            SmallVectorImpl<MaterializationBufferOp> &buffers,
                            StringRef *derivationStage = nullptr) {
  MLIRContext *context = producer.getContext();
  SmallVector<int64_t> relationDomain;
  std::optional<IntegerSet> relationPredicate;
  SmallVector<AffineExpr> producerIndices;
  SmallVector<AffineExpr> consumerIndices;

  if (hasSharedStateFlow(producer, consumer)) {
    if (derivationStage)
      *derivationStage = "shared-state";
    FailureOr<SmallVector<int64_t>> producerFactors =
        ownerGridFactors(producer);
    FailureOr<SmallVector<int64_t>> consumerFactors =
        ownerGridFactors(consumer);
    if (failed(producerFactors) || failed(consumerFactors) ||
        producer.getGridShape().size() != 1 ||
        consumer.getGridShape().size() != 1)
      return failure();

    auto producerOwners = producer->getAttrOfType<DenseI64ArrayAttr>(
        "composition_owner_dimensions");
    auto consumerOwners = consumer->getAttrOfType<DenseI64ArrayAttr>(
        "composition_owner_dimensions");
    ArrayRef<int64_t> producerDomain = producer.getDomainShape();
    ArrayRef<int64_t> consumerDomain = consumer.getDomainShape();
    ArrayRef<int64_t> producerTile = producer.getTileShape();
    ArrayRef<int64_t> consumerTile = consumer.getTileShape();
    int64_t commonRank = 0;
    while (commonRank < static_cast<int64_t>(producerFactors->size()) &&
           commonRank < static_cast<int64_t>(consumerFactors->size())) {
      const int64_t producerAxis = producerOwners.asArrayRef()[commonRank];
      const int64_t consumerAxis = consumerOwners.asArrayRef()[commonRank];
      if (producerAxis != consumerAxis ||
          producerDomain[producerAxis] != consumerDomain[consumerAxis] ||
          producerTile[producerAxis] != consumerTile[consumerAxis])
        break;
      ++commonRank;
    }
    FailureOr<int64_t> common = scheduleElementCount(
        ArrayRef<int64_t>(*producerFactors).take_front(commonRank));
    if (failed(common))
      return failure();

    if (commonRank == static_cast<int64_t>(producerFactors->size()) &&
        commonRank == static_cast<int64_t>(consumerFactors->size())) {
      relationDomain = {*common};
      producerIndices.push_back(getAffineDimExpr(0, context));
      consumerIndices.push_back(getAffineDimExpr(0, context));
    } else {
      ArrayRef<int64_t> producerSuffix =
          ArrayRef<int64_t>(*producerFactors).drop_front(commonRank);
      ArrayRef<int64_t> consumerSuffix =
          ArrayRef<int64_t>(*consumerFactors).drop_front(commonRank);
      relationDomain.push_back(*common);
      llvm::append_range(relationDomain, producerSuffix);
      llvm::append_range(relationDomain, consumerSuffix);
      AffineExpr producerIndex = getAffineDimExpr(0, context);
      for (auto [axis, extent] : llvm::enumerate(producerSuffix))
        producerIndex =
            producerIndex * extent + getAffineDimExpr(axis + 1, context);
      AffineExpr consumerIndex = getAffineDimExpr(0, context);
      for (auto [axis, extent] : llvm::enumerate(consumerSuffix))
        consumerIndex =
            consumerIndex * extent +
            getAffineDimExpr(1 + producerSuffix.size() + axis, context);
      producerIndices.push_back(producerIndex);
      consumerIndices.push_back(consumerIndex);

      auto producerOrderedDimensions =
          producer.getCompositionOrderedIndexDimensionsAttr();
      auto consumerOrderedDimensions =
          consumer.getCompositionOrderedIndexDimensionsAttr();
      auto producerOrderedSource =
          producer.getCompositionOrderedIndexSourceAttr();
      auto consumerOrderedSource =
          consumer.getCompositionOrderedIndexSourceAttr();
      auto producerOrderedInclusive =
          producer.getCompositionOrderedIndexInclusiveAttr();
      auto consumerOrderedInclusive =
          consumer.getCompositionOrderedIndexInclusiveAttr();
      const bool hasOrderedConstraint = producerOrderedDimensions ||
                                        producerOrderedSource ||
                                        producerOrderedInclusive;
      if (hasOrderedConstraint &&
          (!producerOrderedDimensions || !consumerOrderedDimensions ||
           producerOrderedDimensions != consumerOrderedDimensions ||
           !producerOrderedSource ||
           producerOrderedSource != consumerOrderedSource ||
           !producerOrderedInclusive ||
           producerOrderedInclusive != consumerOrderedInclusive ||
           producerOrderedDimensions.size() != 2))
        return failure();
      if (hasOrderedConstraint) {
        OrderedIndexConstraint constraint{
            producerOrderedDimensions.asArrayRef()[0],
            producerOrderedDimensions.asArrayRef()[1],
            producerOrderedInclusive.getValue()};
        auto producerIndexShape = producer.getCompositionIndexDomainShapeAttr();
        auto consumerIndexShape = consumer.getCompositionIndexDomainShapeAttr();
        if (!producerIndexShape || !consumerIndexShape ||
            producerIndexShape != consumerIndexShape ||
            constraint.lesserAxis < 0 || constraint.greaterAxis < 0 ||
            constraint.lesserAxis >= producerIndexShape.size() ||
            constraint.greaterAxis >= producerIndexShape.size())
          return failure();
        FailureOr<int64_t> producerGreater = ownerPositionForIndexDomainAxis(
            producer, constraint.greaterAxis, commonRank);
        FailureOr<int64_t> consumerLesser = ownerPositionForIndexDomainAxis(
            consumer, constraint.lesserAxis, commonRank);
        if (failed(producerGreater) || failed(consumerLesser))
          return failure();
        const int64_t producerOwner =
            producerOwners.asArrayRef()[commonRank + *producerGreater];
        const int64_t consumerOwner =
            consumerOwners.asArrayRef()[commonRank + *consumerLesser];
        const int64_t greaterTile = producerTile[producerOwner];
        const int64_t lesserTile = consumerTile[consumerOwner];
        const int64_t greaterPosition = 1 + *producerGreater;
        const int64_t lesserPosition =
            1 + producerSuffix.size() + *consumerLesser;
        const int64_t strictOffset = constraint.inclusive ? 1 : 2;
        AffineExpr lesserFirst =
            getAffineDimExpr(lesserPosition, context) * lesserTile;
        AffineExpr greaterLast =
            getAffineDimExpr(greaterPosition, context) * greaterTile +
            greaterTile - strictOffset;
        relationPredicate = IntegerSet::get(
            relationDomain.size(), 0,
            {greaterLast - lesserFirst,
             getAffineConstantExpr(
                 producerIndexShape.asArrayRef()[constraint.greaterAxis] -
                     strictOffset,
                 context) -
                 lesserFirst},
            {false, false});
      }
    }
  } else if (FailureOr<std::optional<ScheduledTaskRelation>> logical =
                 deriveClosedLogicalTaskRelation(producer, consumer,
                                                 producerSource, consumerSource,
                                                 buffers);
             failed(logical)) {
    if (derivationStage)
      *derivationStage = "logical-task";
    return failure();
  } else if (*logical) {
    if (derivationStage)
      *derivationStage = "logical-task";
    return **logical;
  } else if (FailureOr<std::optional<ScheduledTaskBufferAccessPair>> accesses =
                 deriveFoldStateBufferAccesses(producer, consumer,
                                               producerSource, consumerSource,
                                               buffers);
             failed(accesses)) {
    if (derivationStage)
      *derivationStage = "fold-state";
    return failure();
  } else if (*accesses) {
    if (derivationStage)
      *derivationStage = "fold-state";
    return deriveRelationFromBufferAccesses(**accesses, producer, consumer);
  } else {
    if (derivationStage)
      *derivationStage = "generic";
    ArrayRef<int64_t> producerGrid = producer.getGridShape();
    ArrayRef<int64_t> consumerGrid = consumer.getGridShape();
    llvm::append_range(relationDomain, producerGrid);
    llvm::append_range(relationDomain, consumerGrid);
    for (int64_t axis = 0; axis < static_cast<int64_t>(producerGrid.size());
         ++axis)
      producerIndices.push_back(getAffineDimExpr(axis, context));
    for (int64_t axis = 0; axis < static_cast<int64_t>(consumerGrid.size());
         ++axis)
      consumerIndices.push_back(
          getAffineDimExpr(producerGrid.size() + axis, context));
  }

  FailureOr<int64_t> storageCapacity = scheduleElementCount(relationDomain);
  if (failed(storageCapacity)) {
    if (derivationStage)
      *derivationStage = "storage";
    return failure();
  }
  AffineExpr storageSlot = getAffineConstantExpr(0, context);
  for (auto [axis, extent] : llvm::enumerate(relationDomain))
    storageSlot = storageSlot * extent + getAffineDimExpr(axis, context);
  const int64_t relationRank = relationDomain.size();
  return ScheduledTaskRelation{
      std::move(relationDomain),
      relationPredicate,
      AffineMap::get(relationRank, 0, producerIndices, context),
      AffineMap::get(relationRank, 0, consumerIndices, context),
      *storageCapacity,
      AffineMap::get(relationRank, 0, storageSlot, context),
      AffineMap::get(relationRank, 0, getAffineConstantExpr(0, context),
                     context)};
}

FailureOr<FoldOp> contractFoldCompositionSource(MaterializationTaskOp task) {
  auto kind = task->getAttrOfType<StringAttr>("composition_kind");
  if (!kind || kind.getValue() != "contract_fold")
    return failure();
  FoldOp found;
  bool duplicate = false;
  task->getParentOfType<ModuleOp>().walk([&](FoldOp fold) {
    if (fold.getSource() != task.getSource())
      return;
    if (found)
      duplicate = true;
    else
      found = fold;
  });
  if (!found || duplicate || !isClosedMultiStatePlanningFold(found))
    return failure();
  return found;
}

FailureOr<int64_t> foldAccumulatorBytes(FoldOp fold) {
  int64_t accumulatorBytes = 0;
  for (Attribute attribute : fold.getAccumulatorTypes()) {
    auto type = dyn_cast<TypeAttr>(attribute);
    int64_t bytes = type && type.getValue().isF32()
                        ? 4
                        : (type && type.getValue().isBF16() ? 2 : 0);
    if (bytes == 0 ||
        accumulatorBytes > std::numeric_limits<int64_t>::max() - bytes)
      return failure();
    accumulatorBytes += bytes;
  }
  return accumulatorBytes;
}

FailureOr<ScheduledGeometry>
deriveContractFoldCompositionGeometry(MaterializationTaskOp task,
                                      NumericalPolicy policy) {
  FailureOr<FoldOp> fold = contractFoldCompositionSource(task);
  ArrayRef<int64_t> reductions = task.getReductionDimensions();
  if (reductions.size() != 1)
    return failure();
  FailureOr<ScheduledGeometry> geometry =
      deriveSimt32Geometry(ScheduleTaskKind::RowFold, task.getDomainShape(),
                           policy, reductions.front());
  if (failed(fold) || failed(geometry))
    return failure();
  FailureOr<int64_t> accumulatorBytes = foldAccumulatorBytes(*fold);
  if (failed(accumulatorBytes))
    return failure();
  if (geometry->workgroupThreads >
      std::numeric_limits<int64_t>::max() / *accumulatorBytes)
    return failure();
  constexpr int64_t kMaximumCompositionScratchBytes = 96 * 1024;
  geometry->scratchBytes = geometry->workgroupThreads * *accumulatorBytes;
  if (geometry->scratchBytes > kMaximumCompositionScratchBytes)
    return failure();
  return geometry;
}

FailureOr<ScheduledGeometry>
deriveIndexedAccumulationGeometry(MaterializationTaskOp task) {
  ArrayRef<int64_t> domain = task.getDomainShape();
  ArrayRef<int64_t> reductions = task.getReductionDimensions();
  if (domain.empty() || reductions.empty())
    return failure();
  llvm::SmallDenseSet<int64_t> reductionSet;
  SmallVector<int64_t> ownerShape;
  SmallVector<int64_t> reductionShape;
  SmallVector<int64_t> tileShape(domain.size(), 1);
  for (int64_t axis : reductions) {
    if (axis < 0 || axis >= static_cast<int64_t>(domain.size()) ||
        !reductionSet.insert(axis).second)
      return failure();
    tileShape[axis] = domain[axis];
    reductionShape.push_back(domain[axis]);
  }
  for (auto [axis, extent] : llvm::enumerate(domain))
    if (!reductionSet.contains(axis))
      ownerShape.push_back(extent);
  FailureOr<int64_t> owners = scheduleElementCount(ownerShape);
  FailureOr<int64_t> reduction = scheduleElementCount(reductionShape);
  if (failed(owners) || failed(reduction) || *owners > UINT32_MAX)
    return failure();
  constexpr int64_t kThreads = 32;
  return ScheduledGeometry{{*owners},
                           std::move(tileShape),
                           ceilDivPositive(*reduction, kThreads),
                           kThreads,
                           kThreads,
                           kThreads * 4,
                           reductions.back()};
}

FailureOr<ScheduledGeometry>
deriveMultiPhaseCompositionGeometry(MaterializationTaskOp task) {
  auto phases = task->getAttrOfType<ArrayAttr>("composition_phases");
  auto bindings =
      task->getAttrOfType<ArrayAttr>("composition_result_bindings");
  if (!phases || phases.empty() || !bindings ||
      bindings.size() != task.getOutputBuffers().size())
    return failure();
  int64_t totalBlocks = 0;
  int64_t maximumTile = 0;
  int64_t maximumThreads = 0;
  int64_t maximumSerialTiles = 1;
  for (auto [phaseOrdinal, attribute] : llvm::enumerate(phases)) {
    auto phase = dyn_cast<DictionaryAttr>(attribute);
    auto domain = phase ? phase.getAs<DenseI64ArrayAttr>("domain_shape")
                        : DenseI64ArrayAttr{};
    auto reductions =
        phase ? phase.getAs<DenseI64ArrayAttr>("reduction_dimensions")
              : DenseI64ArrayAttr{};
    if (!phase || !domain || !reductions)
      return failure();
    FailureOr<int64_t> elements = scheduleElementCount(domain.asArrayRef());
    if (failed(elements))
      return failure();
    int64_t serialTiles = 1;
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
          scheduleElementCount(reductionShape);
      if (failed(reductionElements))
        return failure();
      serialTiles = *reductionElements;
      *elements /= serialTiles;
    }
    if (*elements <= 0 || *elements > UINT32_MAX)
      return failure();
    const int64_t tile = std::min<int64_t>(*elements, 256);
    const int64_t blocks = ceilDivPositive(*elements, tile);
    if (totalBlocks > UINT32_MAX - blocks)
      return failure();
    totalBlocks += blocks;
    maximumTile = std::max(maximumTile, tile);
    maximumThreads = std::max(maximumThreads, scheduleThreads(tile));
    maximumSerialTiles = std::max(maximumSerialTiles, serialTiles);
  }
  return ScheduledGeometry{{totalBlocks},
                           {maximumTile},
                           maximumSerialTiles,
                           maximumThreads,
                           32,
                           0,
                           task.getReductionDimensions().empty()
                               ? std::optional<int64_t>()
                               : std::optional<int64_t>(
                                     task.getReductionDimensions().back())};
}

FailureOr<ArrayAttr> multiPhaseCompositionScheduleRecords(
    OpBuilder &builder, MaterializationTaskOp task) {
  auto phases = task->getAttrOfType<ArrayAttr>("composition_phases");
  if (!phases || phases.empty())
    return failure();
  SmallVector<Attribute> records;
  int64_t gridOffset = 0;
  for (auto [phaseOrdinal, attribute] : llvm::enumerate(phases)) {
    auto phase = dyn_cast<DictionaryAttr>(attribute);
    auto domain = phase ? phase.getAs<DenseI64ArrayAttr>("domain_shape")
                        : DenseI64ArrayAttr{};
    auto reductions =
        phase ? phase.getAs<DenseI64ArrayAttr>("reduction_dimensions")
              : DenseI64ArrayAttr{};
    if (!phase || !domain || !reductions)
      return failure();
    FailureOr<int64_t> elements = scheduleElementCount(domain.asArrayRef());
    if (failed(elements))
      return failure();
    int64_t serialTiles = 1;
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
          scheduleElementCount(reductionShape);
      if (failed(reductionElements))
        return failure();
      serialTiles = *reductionElements;
      *elements /= serialTiles;
    }
    const int64_t tile = std::min<int64_t>(*elements, 256);
    const int64_t blocks = ceilDivPositive(*elements, tile);
    records.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("phase", builder.getI64IntegerAttr(phaseOrdinal)),
         builder.getNamedAttr("grid_offset",
                              builder.getI64IntegerAttr(gridOffset)),
         builder.getNamedAttr("grid_count", builder.getI64IntegerAttr(blocks)),
         builder.getNamedAttr("element_count",
                              builder.getI64IntegerAttr(*elements)),
         builder.getNamedAttr("tile_elements", builder.getI64IntegerAttr(tile)),
         builder.getNamedAttr("serial_tiles",
                              builder.getI64IntegerAttr(serialTiles)),
         builder.getNamedAttr("domain_shape", domain),
         builder.getNamedAttr("reduction_dimensions", reductions)}));
    gridOffset += blocks;
  }
  return builder.getArrayAttr(records);
}

bool isGenericContractFoldCompositionPlan(MaterializationPlanOp plan) {
  bool sawContractFold = false;
  bool sawMapContract = false;
  for (MaterializationTaskOp task :
       plan.getBody().front().getOps<MaterializationTaskOp>()) {
    auto kind = task->getAttrOfType<StringAttr>("composition_kind");
    if (!kind)
      return false;
    if (kind.getValue() == "contract_fold")
      sawContractFold = true;
    else if (kind.getValue() == "map_contract")
      sawMapContract = true;
    else
      return false;
  }
  return sawContractFold && sawMapContract;
}

void createScheduleRecord(OpBuilder &builder, Location location,
                          StringRef operationName,
                          ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, operationName);
  state.addAttributes(attributes);
  builder.create(state);
}

ArrayAttr internalCoordinateArray(OpBuilder &builder,
                                  ArrayRef<GpuInternalEventCoordinate> values) {
  SmallVector<Attribute> coordinates;
  coordinates.reserve(values.size());
  for (const GpuInternalEventCoordinate &coordinate : values)
    coordinates.push_back(
        DenseI64ArrayAttr::get(builder.getContext(), coordinate));
  return builder.getArrayAttr(coordinates);
}

ArrayAttr internalAxisArray(OpBuilder &builder,
                            ArrayRef<GpuInternalEventAxis> axes) {
  SmallVector<Attribute> attributes;
  attributes.reserve(axes.size());
  for (const GpuInternalEventAxis &axis : axes)
    attributes.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("name", builder.getStringAttr(axis.name)),
         builder.getNamedAttr("extent",
                              builder.getI64IntegerAttr(axis.extent))}));
  return builder.getArrayAttr(attributes);
}

ArrayAttr
internalRelationPairArray(OpBuilder &builder,
                          ArrayRef<GpuInternalEventRelationPair> pairs) {
  SmallVector<Attribute> attributes;
  attributes.reserve(pairs.size());
  for (const GpuInternalEventRelationPair &pair : pairs)
    attributes.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("source", DenseI64ArrayAttr::get(
                                            builder.getContext(), pair.source)),
         builder.getNamedAttr(
             "target",
             DenseI64ArrayAttr::get(builder.getContext(), pair.target))}));
  return builder.getArrayAttr(attributes);
}

StringRef internalScopeName(GpuEventPhysicalScope scope) {
  switch (scope) {
  case GpuEventPhysicalScope::ThreadBlock:
    return "cta";
  case GpuEventPhysicalScope::Cluster:
    return "cluster";
  case GpuEventPhysicalScope::Device:
    return "device";
  case GpuEventPhysicalScope::System:
    return "system";
  }
  llvm_unreachable("unknown internal Event scope");
}

StringRef internalRealizationKindName(GpuInternalEventRealizationKind kind) {
  switch (kind) {
  case GpuInternalEventRealizationKind::ErasedProgramOrder:
    return "erased_program_order";
  case GpuInternalEventRealizationKind::ErasedStreamOrder:
    return "erased_stream_order";
  case GpuInternalEventRealizationKind::ErasedAlgebraicRefinement:
    return "erased_algebraic_refinement";
  case GpuInternalEventRealizationKind::Physical:
    return "physical";
  }
  llvm_unreachable("unknown internal Event realization kind");
}

StringRef internalLegalizationKindName(
    GpuInternalEventAllocation::LegalizationKind kind) {
  switch (kind) {
  case GpuInternalEventAllocation::LegalizationKind::CtaMbarrier:
    return "cta_mbarrier";
  case GpuInternalEventAllocation::LegalizationKind::CtaTransactionMbarrier:
    return "cta_transaction_mbarrier";
  case GpuInternalEventAllocation::LegalizationKind::CtaAsyncCompletionQueue:
    return "cta_async_completion_queue";
  case GpuInternalEventAllocation::LegalizationKind::DeviceSemaphoreQueue:
    return "device_semaphore_queue";
  }
  llvm_unreachable("unknown internal Event legalization kind");
}

FailureOr<InternalEventPlanOp> createInternalEventPlan(
    OpBuilder &builder, Location location,
    const GpuInternalEventSchedule &schedule, StringRef launchRecipeFingerprint,
    StringRef canonicalSchedule, StringRef canonicalScheduleDigest) {
  const std::string fingerprint = physicalEventProvenanceFingerprint(
      schedule.physicalScheduleFingerprint, launchRecipeFingerprint,
      canonicalScheduleDigest);
  OperationState state(location, InternalEventPlanOp::getOperationName());
  state.addAttribute("schema_version", builder.getI64IntegerAttr(1));
  state.addAttribute(
      "physical_schedule_fingerprint",
      builder.getStringAttr(schedule.physicalScheduleFingerprint));
  state.addAttribute("physical_launch_recipe_fingerprint",
                     builder.getStringAttr(launchRecipeFingerprint));
  state.addAttribute("canonical_schedule",
                     builder.getStringAttr(canonicalSchedule));
  state.addAttribute("canonical_schedule_digest",
                     builder.getStringAttr(canonicalScheduleDigest));
  state.addAttribute("fingerprint", builder.getStringAttr(fingerprint));
  state.addRegion();
  auto plan = cast<InternalEventPlanOp>(builder.create(state));
  plan.getBody().push_back(new Block());
  builder.setInsertionPointToEnd(&plan.getBody().front());

  for (auto [regionOrdinal, region] : llvm::enumerate(schedule.regions)) {
    DenseMap<StringRef, int64_t> tasks;
    DenseMap<StringRef, int64_t> events;
    DenseMap<StringRef, int64_t> allocations;
    for (auto [ordinal, task] : llvm::enumerate(region.taskFamilies)) {
      tasks.insert({task.name, ordinal});
      SmallVector<NamedAttribute> attributes{
          builder.getNamedAttr("region",
                               builder.getI64IntegerAttr(regionOrdinal)),
          builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
          builder.getNamedAttr("name", builder.getStringAttr(task.name)),
          builder.getNamedAttr("axes", internalAxisArray(builder, task.axes))};
      if (task.placement)
        attributes.push_back(builder.getNamedAttr(
            "placement", builder.getStringAttr(*task.placement)));
      createScheduleRecord(builder, location,
                           InternalEventTaskOp::getOperationName(), attributes);
    }
    for (auto [ordinal, event] : llvm::enumerate(region.events)) {
      events.insert({event.name, ordinal});
      auto producer = tasks.find(event.requiredDependence.source.name);
      auto consumer = tasks.find(event.requiredDependence.target.name);
      if (producer == tasks.end() || consumer == tasks.end())
        return failure();
      SmallVector<Attribute> initialCounts;
      for (const GpuInternalEventInitialCount &count : event.initialCounts)
        initialCounts.push_back(builder.getDictionaryAttr(
            {builder.getNamedAttr("coordinate",
                                  DenseI64ArrayAttr::get(builder.getContext(),
                                                         count.coordinate)),
             builder.getNamedAttr("value",
                                  builder.getI64IntegerAttr(count.value))}));
      SmallVector<NamedAttribute> attributes{
          builder.getNamedAttr("region",
                               builder.getI64IntegerAttr(regionOrdinal)),
          builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
          builder.getNamedAttr("name", builder.getStringAttr(event.name)),
          builder.getNamedAttr("producer_task",
                               builder.getI64IntegerAttr(producer->second)),
          builder.getNamedAttr("consumer_task",
                               builder.getI64IntegerAttr(consumer->second)),
          builder.getNamedAttr("required_pairs",
                               internalRelationPairArray(
                                   builder, event.requiredDependence.pairs)),
          builder.getNamedAttr(
              "event_axes", internalAxisArray(builder, event.eventDomain.axes)),
          builder.getNamedAttr(
              "notify_pairs",
              internalRelationPairArray(builder, event.notifyRelation.pairs)),
          builder.getNamedAttr(
              "trigger_pairs",
              internalRelationPairArray(builder, event.triggerRelation.pairs)),
          builder.getNamedAttr("initial_counts",
                               builder.getArrayAttr(initialCounts)),
          builder.getNamedAttr(
              "initial_count_provenance",
              builder.getStringAttr(event.initialCountProvenance)),
          builder.getNamedAttr(
              "memory_scope",
              builder.getStringAttr(internalScopeName(event.memoryScope))),
          builder.getNamedAttr(
              "generation_policy",
              builder.getStringAttr(
                  event.generationPolicy ==
                          GpuInternalEventGenerationPolicy::PerInvocation
                      ? "per_invocation"
                      : "phased")),
          builder.getNamedAttr(
              "visibility_scope",
              builder.getStringAttr(internalScopeName(event.visibility.scope))),
          builder.getNamedAttr(
              "release", builder.getBoolAttr(event.visibility.releaseOnNotify)),
          builder.getNamedAttr(
              "acquire",
              builder.getBoolAttr(event.visibility.acquireBeforeConsumer))};
      if (event.schedulingMode)
        attributes.push_back(builder.getNamedAttr(
            "scheduling_mode", builder.getStringAttr(*event.schedulingMode)));
      createScheduleRecord(builder, location,
                           InternalEventTensorOp::getOperationName(),
                           attributes);
    }
    for (auto [ordinal, allocation] : llvm::enumerate(region.allocations)) {
      allocations.insert({allocation.allocationId, ordinal});
      SmallVector<int64_t> coveredEvents;
      for (const std::string &covered : allocation.coveredPlans) {
        auto event = events.find(covered);
        if (event == events.end())
          return failure();
        coveredEvents.push_back(event->second);
      }
      SmallVector<Attribute> notifyRelations;
      for (const GpuInternalEventAllocation::NotifyRelation &relation :
           allocation.notifyRelations)
        notifyRelations.push_back(builder.getDictionaryAttr(
            {builder.getNamedAttr("task_family",
                                  builder.getStringAttr(relation.taskFamily)),
             builder.getNamedAttr(
                 "task_coordinate",
                 DenseI64ArrayAttr::get(builder.getContext(),
                                        relation.taskCoordinate)),
             builder.getNamedAttr(
                 "mechanism_coordinate",
                 DenseI64ArrayAttr::get(builder.getContext(),
                                        relation.mechanismCoordinate))}));
      SmallVector<Attribute> waitRelations;
      for (const GpuInternalEventAllocation::WaitRelation &relation :
           allocation.waitRelations)
        waitRelations.push_back(builder.getDictionaryAttr(
            {builder.getNamedAttr(
                 "mechanism_coordinate",
                 DenseI64ArrayAttr::get(builder.getContext(),
                                        relation.mechanismCoordinate)),
             builder.getNamedAttr("task_family",
                                  builder.getStringAttr(relation.taskFamily)),
             builder.getNamedAttr(
                 "task_coordinate",
                 DenseI64ArrayAttr::get(builder.getContext(),
                                        relation.taskCoordinate))}));
      SmallVector<NamedAttribute> attributes{
          builder.getNamedAttr("region",
                               builder.getI64IntegerAttr(regionOrdinal)),
          builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
          builder.getNamedAttr("allocation_id",
                               builder.getStringAttr(allocation.allocationId)),
          builder.getNamedAttr(
              "legalization",
              builder.getStringAttr(
                  internalLegalizationKindName(allocation.legalizationKind))),
          builder.getNamedAttr(
              "covered_events",
              DenseI64ArrayAttr::get(builder.getContext(), coveredEvents)),
          builder.getNamedAttr("scope", builder.getStringAttr(internalScopeName(
                                            allocation.scope))),
          builder.getNamedAttr("release",
                               builder.getBoolAttr(allocation.releaseOnNotify)),
          builder.getNamedAttr(
              "acquire", builder.getBoolAttr(allocation.acquireBeforeConsumer)),
          builder.getNamedAttr("capacity",
                               builder.getI64IntegerAttr(allocation.capacity)),
          builder.getNamedAttr("mechanism_coordinates",
                               internalCoordinateArray(
                                   builder, allocation.mechanismCoordinates)),
          builder.getNamedAttr(
              "allocation_owners",
              internalCoordinateArray(builder, allocation.allocationOwners)),
          builder.getNamedAttr(
              "slots",
              DenseI64ArrayAttr::get(builder.getContext(), allocation.slots)),
          builder.getNamedAttr("generations",
                               DenseI64ArrayAttr::get(builder.getContext(),
                                                      allocation.generations)),
          builder.getNamedAttr(
              "generation_count",
              builder.getI64IntegerAttr(allocation.generationCount)),
          builder.getNamedAttr("notify_relations",
                               builder.getArrayAttr(notifyRelations)),
          builder.getNamedAttr("wait_relations",
                               builder.getArrayAttr(waitRelations)),
          builder.getNamedAttr(
              "initial_counts",
              DenseI64ArrayAttr::get(builder.getContext(),
                                     allocation.initialCounts))};
      if (allocation.transactionBytesByCoordinate)
        attributes.push_back(builder.getNamedAttr(
            "transaction_bytes_by_coordinate",
            DenseI64ArrayAttr::get(builder.getContext(),
                                   *allocation.transactionBytesByCoordinate)));
      createScheduleRecord(builder, location,
                           InternalEventAllocationOp::getOperationName(),
                           attributes);

      auto mechanismOrdinal =
          [&](ArrayRef<int64_t> coordinate) -> std::optional<int64_t> {
        for (auto [mechanismOrdinal, mechanismCoordinate] :
             llvm::enumerate(allocation.mechanismCoordinates))
          if (ArrayRef<int64_t>(mechanismCoordinate) == coordinate)
            return mechanismOrdinal;
        return std::nullopt;
      };
      auto imperativeAttributes = [&](int64_t relationOrdinal,
                                      StringRef taskFamily,
                                      ArrayRef<int64_t> taskCoordinate,
                                      ArrayRef<int64_t> mechanismCoordinate)
          -> std::optional<SmallVector<NamedAttribute>> {
        auto task = tasks.find(taskFamily);
        std::optional<int64_t> coordinateOrdinal =
            mechanismOrdinal(mechanismCoordinate);
        if (task == tasks.end() || !coordinateOrdinal)
          return std::nullopt;
        SmallVector<NamedAttribute> result{
            builder.getNamedAttr("region",
                                 builder.getI64IntegerAttr(regionOrdinal)),
            builder.getNamedAttr("allocation",
                                 builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr("relation",
                                 builder.getI64IntegerAttr(relationOrdinal)),
            builder.getNamedAttr("task",
                                 builder.getI64IntegerAttr(task->second)),
            builder.getNamedAttr(
                "task_coordinate",
                DenseI64ArrayAttr::get(builder.getContext(), taskCoordinate)),
            builder.getNamedAttr("mechanism_coordinate_ordinal",
                                 builder.getI64IntegerAttr(*coordinateOrdinal)),
            builder.getNamedAttr("mechanism_coordinate",
                                 DenseI64ArrayAttr::get(builder.getContext(),
                                                        mechanismCoordinate)),
            builder.getNamedAttr(
                "allocation_owner",
                DenseI64ArrayAttr::get(
                    builder.getContext(),
                    allocation.allocationOwners[*coordinateOrdinal])),
            builder.getNamedAttr("slot",
                                 builder.getI64IntegerAttr(
                                     allocation.slots[*coordinateOrdinal])),
            builder.getNamedAttr(
                "generation", builder.getI64IntegerAttr(
                                  allocation.generations[*coordinateOrdinal])),
            builder.getNamedAttr(
                "generation_count",
                builder.getI64IntegerAttr(allocation.generationCount)),
            builder.getNamedAttr(
                "scope",
                builder.getStringAttr(internalScopeName(allocation.scope))),
            builder.getNamedAttr(
                "legalization",
                builder.getStringAttr(internalLegalizationKindName(
                    allocation.legalizationKind)))};
        return result;
      };
      for (auto [relationOrdinal, relation] :
           llvm::enumerate(allocation.notifyRelations)) {
        std::optional<SmallVector<NamedAttribute>> imperative =
            imperativeAttributes(relationOrdinal, relation.taskFamily,
                                 relation.taskCoordinate,
                                 relation.mechanismCoordinate);
        if (!imperative)
          return failure();
        const int64_t coordinateOrdinal =
            *mechanismOrdinal(relation.mechanismCoordinate);
        imperative->push_back(builder.getNamedAttr(
            "arrival_count", builder.getI64IntegerAttr(1)));
        imperative->push_back(builder.getNamedAttr(
            "initial_count", builder.getI64IntegerAttr(
                                 allocation.initialCounts[coordinateOrdinal])));
        imperative->push_back(builder.getNamedAttr(
            "release", builder.getBoolAttr(allocation.releaseOnNotify)));
        if (allocation.transactionBytesByCoordinate)
          imperative->push_back(builder.getNamedAttr(
              "transaction_bytes",
              builder.getI64IntegerAttr(
                  (*allocation
                        .transactionBytesByCoordinate)[coordinateOrdinal])));
        createScheduleRecord(builder, location,
                             InternalEventNotifyOp::getOperationName(),
                             *imperative);
      }
      for (auto [relationOrdinal, relation] :
           llvm::enumerate(allocation.waitRelations)) {
        std::optional<SmallVector<NamedAttribute>> imperative =
            imperativeAttributes(relationOrdinal, relation.taskFamily,
                                 relation.taskCoordinate,
                                 relation.mechanismCoordinate);
        if (!imperative)
          return failure();
        const int64_t coordinateOrdinal =
            *mechanismOrdinal(relation.mechanismCoordinate);
        imperative->push_back(builder.getNamedAttr(
            "required_count",
            builder.getI64IntegerAttr(
                allocation.initialCounts[coordinateOrdinal])));
        imperative->push_back(builder.getNamedAttr(
            "acquire", builder.getBoolAttr(allocation.acquireBeforeConsumer)));
        createScheduleRecord(builder, location,
                             InternalEventWaitOp::getOperationName(),
                             *imperative);
      }
    }
    for (const GpuInternalEventRealization &realization : region.realizations) {
      auto event = events.find(realization.plan);
      if (event == events.end())
        return failure();
      SmallVector<int64_t> allocationOrdinals;
      for (const std::string &allocation : realization.allocationIds) {
        auto found = allocations.find(allocation);
        if (found == allocations.end())
          return failure();
        allocationOrdinals.push_back(found->second);
      }
      createScheduleRecord(
          builder, location, InternalEventRealizationOp::getOperationName(),
          {builder.getNamedAttr("region",
                                builder.getI64IntegerAttr(regionOrdinal)),
           builder.getNamedAttr("event",
                                builder.getI64IntegerAttr(event->second)),
           builder.getNamedAttr(
               "kind", builder.getStringAttr(
                           internalRealizationKindName(realization.kind))),
           builder.getNamedAttr("mechanism",
                                builder.getStringAttr(realization.mechanism)),
           builder.getNamedAttr("allocations",
                                DenseI64ArrayAttr::get(builder.getContext(),
                                                       allocationOrdinals))});
    }
  }
  builder.create<InternalEventPlanYieldOp>(location);
  if (failed(plan.verify()))
    return failure();
  return plan;
}

LogicalResult verifyInternalEventPlanAgainstNative(
    InternalEventPlanOp actual, const GpuInternalEventSchedule &schedule,
    StringRef launchRecipeFingerprint, StringRef canonicalSchedule,
    StringRef canonicalScheduleDigest) {
  OwningOpRef<ModuleOp> expectedModule =
      ModuleOp::create(UnknownLoc::get(actual.getContext()));
  OpBuilder builder(actual.getContext());
  builder.setInsertionPointToEnd(expectedModule->getBody());
  FailureOr<InternalEventPlanOp> expected = createInternalEventPlan(
      builder, expectedModule->getLoc(), schedule, launchRecipeFingerprint,
      canonicalSchedule, canonicalScheduleDigest);
  if (failed(expected))
    return actual.emitOpError(
        "cannot reconstruct the native refined internal Event plan");
  SmallVector<Operation *> actualRecords;
  SmallVector<Operation *> expectedRecords;
  for (Operation &operation : actual.getBody().front())
    actualRecords.push_back(&operation);
  for (Operation &operation : (*expected).getBody().front())
    expectedRecords.push_back(&operation);
  if (actual->getAttrDictionary() != (*expected)->getAttrDictionary() ||
      actualRecords.size() != expectedRecords.size())
    return actual.emitOpError(
        "does not equal the reloaded native refined internal Event plan");
  for (auto [actualRecord, expectedRecord] :
       llvm::zip(actualRecords, expectedRecords))
    if (actualRecord->getName() != expectedRecord->getName() ||
        actualRecord->getAttrDictionary() !=
            expectedRecord->getAttrDictionary())
      return actualRecord->emitOpError(
          "does not equal its reloaded native refined internal Event record");
  return success();
}

FailureOr<ScheduleTaskKind>
scheduleKindForMaterializationTask(MaterializationTaskOp task) {
  if (task.getKind() == MaterializationTaskKind::Scan)
    return ScheduleTaskKind::OrderedState;
  if (isOnlineStateTask(task) &&
      task.getKind() == MaterializationTaskKind::Fold)
    return ScheduleTaskKind::RowFold;
  if (task.getKind() == MaterializationTaskKind::Contract) {
    return ScheduleTaskKind::Contract;
  }
  if (task.getKind() == MaterializationTaskKind::Map) {
    return task.getDomainShape().empty() ? ScheduleTaskKind::Scalar
                                         : ScheduleTaskKind::Elementwise;
  }
  if (task.getKind() == MaterializationTaskKind::Relation) {
    return task.getDomainShape().empty() ? ScheduleTaskKind::Scalar
                                         : ScheduleTaskKind::Elementwise;
  }
  if (task.getKind() != MaterializationTaskKind::Fold &&
      task.getKind() != MaterializationTaskKind::Selection) {
    return failure();
  }
  const int64_t rank = task.getDomainShape().size();
  ArrayRef<int64_t> dimensions = task.getReductionDimensions();
  SmallVector<int64_t> allDimensions;
  for (int64_t axis = 0; axis < rank; ++axis)
    allDimensions.push_back(axis);
  if (rank > 0 && dimensions == ArrayRef<int64_t>(allDimensions))
    return ScheduleTaskKind::AllAxisFold;
  if (auto composition = task->getAttrOfType<StringAttr>("composition_kind");
      composition && composition.getValue() == "indexed_accumulation")
    return ScheduleTaskKind::RowFold;
  if (task->getAttrOfType<StringAttr>("composition_kind") &&
      task->getAttrOfType<StringAttr>("composition_kind").getValue() ==
          "contract_fold" &&
      dimensions.size() == 1 && dimensions.front() >= 0 &&
      dimensions.front() < rank)
    return ScheduleTaskKind::RowFold;
  if (dimensions == ArrayRef<int64_t>({rank - 1})) {
    return ScheduleTaskKind::RowFold;
  }
  if ((rank == 2 && dimensions == ArrayRef<int64_t>({0})) ||
      isFullPrefixReduction(dimensions, rank)) {
    return ScheduleTaskKind::ColumnFold;
  }
  if (dimensions.size() == 1 && dimensions.front() >= 0 &&
      dimensions.front() < rank)
    return ScheduleTaskKind::RowFold;
  return failure();
}

FailureOr<Operation *> boundAlgebraOperation(MaterializationTaskOp task) {
  Operation *found = nullptr;
  bool duplicate = false;
  auto expansionGroup = task.getSourceExpansionGroupAttr();
  auto sourceExpansion = task.getSourceExpansionAttr();
  auto matchesTaskKind = [&](Operation *operation) {
    switch (task.getKind()) {
    case MaterializationTaskKind::Map:
      return isa<MapOp>(operation);
    case MaterializationTaskKind::Relation:
      return isa<RelationOp>(operation);
    case MaterializationTaskKind::Selection:
      return isa<SelectionOp>(operation);
    case MaterializationTaskKind::Fold:
      return isa<FoldOp>(operation);
    case MaterializationTaskKind::Contract:
      return isa<ContractOp>(operation);
    case MaterializationTaskKind::Scan:
      return isa<ScanOp>(operation);
    }
    llvm_unreachable("unknown materialization task kind");
  };
  task->getParentOfType<ModuleOp>().walk([&](Operation *operation) {
    Attribute source;
    if (auto map = dyn_cast<MapOp>(operation))
      source = map.getSource();
    else if (auto relation = dyn_cast<RelationOp>(operation))
      source = relation.getSource();
    else if (auto selection = dyn_cast<SelectionOp>(operation))
      source = selection.getSources().getValue().front();
    else if (auto fold = dyn_cast<FoldOp>(operation))
      source = fold.getSource();
    else if (auto contract = dyn_cast<ContractOp>(operation))
      source = contract.getSource();
    else if (auto scan = dyn_cast<ScanOp>(operation))
      source = scan.getSources().getValue().front();
    if (!source || source != task.getSource() ||
        (sourceExpansion &&
         algebraSourceExpansion(operation) != sourceExpansion) ||
        (expansionGroup && !matchesTaskKind(operation)))
      return;
    if (found)
      duplicate = true;
    else
      found = operation;
  });
  if (!found || duplicate)
    return failure();
  return found;
}

ScheduleReductionOrder scheduleReductionOrderForTask(MaterializationTaskOp task,
                                                     ScheduleTaskKind kind,
                                                     NumericalPolicy policy) {
  if (isOnlineStateTask(task))
    return ScheduleReductionOrder::LeafOrderFree;
  if (task.getKind() == MaterializationTaskKind::Selection)
    return ScheduleReductionOrder::StableTotalOrderLeafOrderFixed;
  if (kind == ScheduleTaskKind::Contract)
    return policy == NumericalPolicy::Fast
               ? ScheduleReductionOrder::LeafOrderFree
               : ScheduleReductionOrder::SerialAscendingFmaRnF32;
  return ScheduleReductionOrder::TreeAssociationFreeLeafOrderFixed;
}

class PlanSimt32RowFoldSchedulePass
    : public impl::ShuttlePlanSimt32RowFoldSchedulePassBase<
          PlanSimt32RowFoldSchedulePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<MaterializationPlanOp> materializationPlans(
        module.getOps<MaterializationPlanOp>());
    if (materializationPlans.empty()) {
      module.emitError("requires source-bound materialization plans");
      return signalPassFailure();
    }
    llvm::SmallDenseSet<Operation *> owners;
    for (MaterializationPlanOp plan : materializationPlans) {
      RegionOp owner;
      if (failed(
              verifyMaterializationPlanAgainstSource(module, plan, &owner)) ||
          !owner || !owners.insert(owner.getOperation()).second) {
        module.emitError(
            "requires one source-bound executable materialization plan per "
            "GPU region");
        return signalPassFailure();
      }
    }
    size_t regionCount = 0;
    module.walk([&](RegionOp) { ++regionCount; });
    if (owners.size() != regionCount) {
      module.emitError(
          "requires one source-bound materialization plan per GPU region");
      return signalPassFailure();
    }
    if (!module.getOps<SchedulePlanOp>().empty()) {
      module.emitError("already contains a schedule plan");
      return signalPassFailure();
    }
    for (MaterializationPlanOp source : materializationPlans) {
      SmallVector<MaterializationBufferOp> sourceBuffers(
          source.getBody().front().getOps<MaterializationBufferOp>());
      SmallVector<MaterializationTaskOp> sourceTasks(
          source.getBody().front().getOps<MaterializationTaskOp>());
      const bool hasTaskDependencies =
          llvm::any_of(sourceTasks, [](MaterializationTaskOp task) {
            if (!task.getDependencies().empty())
              return true;
            FailureOr<ScheduleTaskKind> kind =
                scheduleKindForMaterializationTask(task);
            return succeeded(kind) && *kind == ScheduleTaskKind::AllAxisFold;
          });

      OpBuilder builder(module.getContext());
      builder.setInsertionPointToEnd(module.getBody());
      OperationState planState(module.getLoc(),
                               SchedulePlanOp::getOperationName());
      planState.addAttribute("schema_version", builder.getI64IntegerAttr(
                                                   source.getSchemaVersion()));
      planState.addAttribute(
          "target",
          ScheduleTargetAttr::get(module.getContext(), ScheduleTarget::Simt32));
      planState.addAttribute("policy", source.getPolicyAttr());
      planState.addAttribute("source_plan_fingerprint",
                             builder.getStringAttr(source.getFingerprint()));
      planState.addAttribute("fingerprint",
                             builder.getStringAttr(std::string(64, '0')));
      if (hasTaskDependencies)
        planState.addAttribute("eventized", builder.getBoolAttr(true));
      planState.addRegion();
      auto schedule = cast<SchedulePlanOp>(builder.create(planState));
      Block *body = new Block();
      schedule.getBody().push_back(body);
      builder.setInsertionPointToEnd(body);

      for (auto [ordinal, buffer] : llvm::enumerate(sourceBuffers)) {
        auto type = cast<RankedTensorType>(buffer.getTensorType());
        SmallVector<int64_t> order;
        for (int64_t axis = 0; axis < type.getRank(); ++axis) {
          order.push_back(axis);
        }
        createScheduleRecord(
            builder, buffer.getLoc(), ScheduleBufferOp::getOperationName(),
            {builder.getNamedAttr("ordinal",
                                  builder.getI64IntegerAttr(ordinal)),
             builder.getNamedAttr("source_buffer",
                                  builder.getI64IntegerAttr(ordinal)),
             builder.getNamedAttr("tensor_type", TypeAttr::get(type)),
             builder.getNamedAttr(
                 "indexing", ScheduleBufferIndexingAttr::get(
                                 module.getContext(),
                                 type.getRank() == 0
                                     ? ScheduleBufferIndexing::Scalar
                                     : ScheduleBufferIndexing::Lexicographic)),
             builder.getNamedAttr(
                 "iteration_order",
                 DenseI64ArrayAttr::get(module.getContext(), order)),
             builder.getNamedAttr(
                 "lifetime_start",
                 builder.getI64IntegerAttr(buffer.getLifetimeStart())),
             builder.getNamedAttr(
                 "lifetime_end",
                 builder.getI64IntegerAttr(buffer.getLifetimeEnd()))});
      }
      for (auto [ordinal, task] : llvm::enumerate(sourceTasks)) {
        FailureOr<ScheduleTaskKind> derivedKind =
            scheduleKindForMaterializationTask(task);
        if (failed(derivedKind)) {
          task.emitOpError("cannot derive a closed SIMT32 task kind");
          return signalPassFailure();
        }
        ScheduleTaskKind kind = *derivedKind;
        const bool multiPhaseComposition = static_cast<bool>(
            task->getAttrOfType<ArrayAttr>("composition_phases"));
        FailureOr<ScheduledGeometry> geometry =
            multiPhaseComposition
                ? deriveMultiPhaseCompositionGeometry(task)
            : isOnlineStateTask(task)    ? deriveOnlineStateGeometry(task)
            : isReverseStateTask(task) ? deriveReverseStateGeometry(task)
            : task->getAttrOfType<StringAttr>("composition_kind") &&
                    task->getAttrOfType<StringAttr>("composition_kind")
                            .getValue() == "indexed_accumulation"
                ? deriveIndexedAccumulationGeometry(task)
            : task->getAttrOfType<StringAttr>("composition_kind") &&
                    task->getAttrOfType<StringAttr>("composition_kind")
                            .getValue() == "contract_fold"
                ? deriveContractFoldCompositionGeometry(task,
                                                        source.getPolicy())
            : task->hasAttr("contract_region_sources")
                ? deriveContractRegionGeometry(task, source.getPolicy())
                : deriveSimt32Geometry(
                      kind, task.getDomainShape(), source.getPolicy(),
                      task.getKind() == MaterializationTaskKind::Fold &&
                              task.getReductionDimensions().size() == 1
                          ? std::optional<int64_t>(
                                task.getReductionDimensions().front())
                          : std::nullopt);
        if (failed(geometry)) {
          task.emitOpError("cannot derive a bounded SIMT32 schedule candidate");
          return signalPassFailure();
        }
        SmallVector<NamedAttribute> attrs{
            builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr("source_task",
                                 builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr(
                "kind", ScheduleTaskKindAttr::get(module.getContext(), kind)),
            builder.getNamedAttr("domain_shape", task.getDomainShapeAttr()),
            builder.getNamedAttr(
                "grid_shape",
                DenseI64ArrayAttr::get(module.getContext(), geometry->grid)),
            builder.getNamedAttr(
                "tile_shape",
                DenseI64ArrayAttr::get(module.getContext(), geometry->tile)),
            builder.getNamedAttr("serial_tiles", builder.getI64IntegerAttr(
                                                     geometry->serialTiles)),
            builder.getNamedAttr(
                "workgroup_threads",
                builder.getI64IntegerAttr(geometry->workgroupThreads)),
            builder.getNamedAttr("subgroup_size", builder.getI64IntegerAttr(
                                                      geometry->subgroupSize)),
            builder.getNamedAttr("scratch_bytes", builder.getI64IntegerAttr(
                                                      geometry->scratchBytes)),
            builder.getNamedAttr("input_buffers", task.getInputBuffersAttr()),
            builder.getNamedAttr("output_buffers", task.getOutputBuffersAttr()),
            builder.getNamedAttr("dependencies", task.getDependenciesAttr()),
            builder.getNamedAttr("semantic_fingerprint",
                                 task.getSemanticFingerprintAttr())};
        if (multiPhaseComposition) {
          FailureOr<ArrayAttr> phaseSchedules =
              multiPhaseCompositionScheduleRecords(builder, task);
          if (failed(phaseSchedules)) {
            task.emitOpError("cannot derive typed multi-phase CTA ranges");
            return signalPassFailure();
          }
          attrs.push_back(builder.getNamedAttr("composition_phase_schedules",
                                               *phaseSchedules));
        }
        FailureOr<Operation *> bound = boundAlgebraOperation(task);
        if (failed(bound)) {
          task.emitOpError("cannot bind its generic algebra operation");
          return signalPassFailure();
        }
        if (auto relation = dyn_cast<RelationOp>(*bound)) {
          attrs.push_back(builder.getNamedAttr("relation_semantics",
                                               relation.getSemanticsAttr()));
          attrs.push_back(builder.getNamedAttr("relation_parameters",
                                               relation.getParametersAttr()));
        } else if (auto selection = dyn_cast<SelectionOp>(*bound)) {
          attrs.push_back(builder.getNamedAttr("selection_dimension",
                                               selection.getDimensionAttr()));
          attrs.push_back(builder.getNamedAttr("selection_count",
                                               selection.getCountAttr()));
          attrs.push_back(builder.getNamedAttr("selection_order",
                                               selection.getOrderAttr()));
          attrs.push_back(builder.getNamedAttr("selection_comparison",
                                               selection.getComparisonAttr()));
          attrs.push_back(builder.getNamedAttr("selection_stable",
                                               selection.getStableAttr()));
          attrs.push_back(builder.getNamedAttr("selection_tie_break",
                                               selection.getTieBreakAttr()));
        }
        for (StringRef name :
             {"composition_owner_dimensions", "composition_stream_dimensions",
              "composition_state_inputs", "composition_state_outputs",
              "composition_producer_contraction_dimensions",
              "composition_consumer_contraction_dimensions",
              "composition_index_domain_source",
              "composition_index_domain_shape", "composition_index_domain_map",
              "composition_ordered_index_source",
              "composition_ordered_index_dimensions",
              "composition_ordered_index_inclusive",
              "composition_state_access_buffers",
              "composition_state_access_maps"})
          if (Attribute attribute = task->getAttr(name))
            attrs.push_back(builder.getNamedAttr(name, attribute));
        if (auto vectorDimensions = task.getOnlineVectorDimensionsAttr())
          attrs.push_back(builder.getNamedAttr("online_vector_dimensions",
                                               vectorDimensions));
        if (task.getOnlineMma().value_or(false))
          attrs.push_back(
              builder.getNamedAttr("online_mma", builder.getBoolAttr(true)));
        if (geometry->reductionAxis || kind == ScheduleTaskKind::AllAxisFold) {
          auto compositionKind =
              task->getAttrOfType<StringAttr>("composition_kind");
          if (kind == ScheduleTaskKind::AllAxisFold ||
              (compositionKind &&
               compositionKind.getValue() == "indexed_accumulation"))
            attrs.push_back(builder.getNamedAttr(
                "reduction_dimensions", task.getReductionDimensionsAttr()));
        }
        if (kind == ScheduleTaskKind::AllAxisFold) {
          attrs.push_back(builder.getNamedAttr(
              "reduction_merge_grid_shape",
              DenseI64ArrayAttr::get(module.getContext(),
                                     geometry->reductionMergeGrid)));
          attrs.push_back(builder.getNamedAttr(
              "reduction_merge_tile_shape",
              DenseI64ArrayAttr::get(module.getContext(),
                                     geometry->reductionMergeTile)));
          attrs.push_back(builder.getNamedAttr(
              "reduction_merge_serial_tiles",
              builder.getI64IntegerAttr(geometry->reductionMergeSerialTiles)));
          attrs.push_back(builder.getNamedAttr(
              "reduction_partial_buffer_shape",
              DenseI64ArrayAttr::get(module.getContext(),
                                     geometry->reductionPartialBufferShape)));
        }
        if (geometry->reductionAxis) {
          attrs.push_back(builder.getNamedAttr(
              "reduction_axis",
              builder.getI64IntegerAttr(*geometry->reductionAxis)));
        }
        if (geometry->reductionAxis || kind == ScheduleTaskKind::AllAxisFold) {
          attrs.push_back(builder.getNamedAttr(
              "reduction_order",
              ScheduleReductionOrderAttr::get(
                  module.getContext(), scheduleReductionOrderForTask(
                                           task, kind, source.getPolicy()))));
        }
        if (FailureOr<FoldOp> fold = contractFoldCompositionSource(task);
            succeeded(fold)) {
          FailureOr<int64_t> stateBytes = foldAccumulatorBytes(*fold);
          if (failed(stateBytes)) {
            task.emitOpError("cannot derive bounded Fold accumulator storage");
            return signalPassFailure();
          }
          attrs.push_back(
              builder.getNamedAttr("reduction_state_bytes_per_thread",
                                   builder.getI64IntegerAttr(*stateBytes)));
        } else if (auto fold = dyn_cast<FoldOp>(*bound);
                   fold && kind == ScheduleTaskKind::RowFold &&
                   geometry->reductionAxis &&
                   *geometry->reductionAxis !=
                       static_cast<int64_t>(task.getDomainShape().size()) - 1) {
          FailureOr<int64_t> stateBytes = foldAccumulatorBytes(fold);
          if (failed(stateBytes)) {
            task.emitOpError("cannot derive bounded Fold accumulator storage");
            return signalPassFailure();
          }
          attrs.push_back(
              builder.getNamedAttr("reduction_state_bytes_per_thread",
                                   builder.getI64IntegerAttr(*stateBytes)));
        }
        if (auto strategy =
                task->getAttrOfType<StringAttr>("contract_region_strategy")) {
          attrs.push_back(
              builder.getNamedAttr("contract_region_strategy", strategy));
          attrs.push_back(builder.getNamedAttr(
              "contract_region_cost",
              task->getAttrOfType<DictionaryAttr>("contract_region_cost")));
        }
        createScheduleRecord(builder, task.getLoc(),
                             ScheduleTaskOp::getOperationName(), attrs);
      }
      SmallVector<std::pair<int64_t, int64_t>> eventEdges;
      for (auto [consumerOrdinal, task] : llvm::enumerate(sourceTasks)) {
        for (int64_t producerOrdinal : task.getDependencies())
          eventEdges.push_back(
              {producerOrdinal, static_cast<int64_t>(consumerOrdinal)});
      }
      SmallVector<ScheduleTaskOp> scheduledTasks(
          body->getOps<ScheduleTaskOp>());
      SmallVector<Attribute> parametricDependencies;
      int64_t eventOrdinal = 0;
      for (auto [dependencyOrdinal, edge] : llvm::enumerate(eventEdges)) {
        const auto [producerOrdinal, consumerOrdinal] = edge;
        MaterializationTaskOp task = sourceTasks[consumerOrdinal];
        StringRef derivationStage = "unclassified";
        FailureOr<ScheduledTaskRelation> relation = deriveScheduledTaskRelation(
            scheduledTasks[producerOrdinal], scheduledTasks[consumerOrdinal],
            sourceTasks[producerOrdinal], sourceTasks[consumerOrdinal],
            sourceBuffers, &derivationStage);
        if (failed(relation)) {
          FailureOr<Operation *> producerOperation =
              boundAlgebraOperation(sourceTasks[producerOrdinal]);
          auto gather = succeeded(producerOperation)
                            ? dyn_cast<RelationOp>(*producerOperation)
                            : RelationOp{};
          if (gather && gather.getSemantics() == RelationSemantics::Gather) {
            SmallVector<int64_t> sharedBuffers;
            for (int64_t output :
                 sourceTasks[producerOrdinal].getOutputBuffers())
              if (llvm::is_contained(
                      sourceTasks[consumerOrdinal].getInputBuffers(), output))
                sharedBuffers.push_back(output);
            FailureOr<int64_t> capacity = scheduleElementCount(
                scheduledTasks[consumerOrdinal].getGridShape());
            if (sharedBuffers.empty() || failed(capacity)) {
              task.emitOpError(
                  "cannot bound runtime-parametric Gather dependence");
              return signalPassFailure();
            }
            parametricDependencies.push_back(builder.getDictionaryAttr({
                builder.getNamedAttr(
                    "dependency_ordinal",
                    builder.getI64IntegerAttr(dependencyOrdinal)),
                builder.getNamedAttr("producer_task", builder.getI64IntegerAttr(
                                                          producerOrdinal)),
                builder.getNamedAttr("consumer_task", builder.getI64IntegerAttr(
                                                          consumerOrdinal)),
                builder.getNamedAttr("relation_source", gather.getSourceAttr()),
                builder.getNamedAttr("relation_semantics",
                                     gather.getSemanticsAttr()),
                builder.getNamedAttr(
                    "shared_buffers",
                    builder.getDenseI64ArrayAttr(sharedBuffers)),
                builder.getNamedAttr("derivation",
                                     builder.getStringAttr("exact_consumer")),
                builder.getNamedAttr("storage_capacity",
                                     builder.getI64IntegerAttr(*capacity)),
                builder.getNamedAttr("generations",
                                     builder.getI64IntegerAttr(1)),
            }));
            continue;
          }
          auto diagnostic = task.emitOpError(
              "cannot derive an exact bounded Event Tensor relation");
          auto appendTask = [&](StringRef role, int64_t ordinal,
                                MaterializationTaskOp sourceTask,
                                ScheduleTaskOp scheduledTask) {
            diagnostic << "; " << role << "_task=" << ordinal
                       << " kind=" << sourceTask->getAttr("kind")
                       << " composition_kind=";
            if (Attribute kind = sourceTask->getAttr("composition_kind"))
              diagnostic << kind;
            else
              diagnostic << "<none>";
            diagnostic << " grid=" << scheduledTask.getGridShapeAttr();
            for (StringRef name :
                 {"composition_owner_dimensions", "composition_state_inputs",
                  "composition_state_outputs",
                  "composition_state_access_buffers",
                  "composition_state_access_maps",
                  "composition_index_domain_shape",
                  "composition_index_domain_map",
                  "composition_ordered_index_dimensions",
                  "composition_ordered_index_source",
                  "composition_ordered_index_inclusive"}) {
              if (Attribute attribute = scheduledTask->getAttr(name))
                diagnostic << " " << name << "=" << attribute;
            }
            diagnostic << " domain=" << scheduledTask.getDomainShapeAttr()
                       << " tile=" << scheduledTask.getTileShapeAttr();
          };
          diagnostic << "; derivation_stage=" << derivationStage;
          appendTask("producer", producerOrdinal, sourceTasks[producerOrdinal],
                     scheduledTasks[producerOrdinal]);
          appendTask("consumer", consumerOrdinal, sourceTasks[consumerOrdinal],
                     scheduledTasks[consumerOrdinal]);
          return signalPassFailure();
        }
        SmallVector<NamedAttribute> eventAttributes{
            builder.getNamedAttr("ordinal",
                                 builder.getI64IntegerAttr(eventOrdinal)),
            builder.getNamedAttr("producer_task",
                                 builder.getI64IntegerAttr(producerOrdinal)),
            builder.getNamedAttr("consumer_task",
                                 builder.getI64IntegerAttr(consumerOrdinal)),
            builder.getNamedAttr(
                "relation_domain_shape",
                DenseI64ArrayAttr::get(module.getContext(), relation->domain)),
            builder.getNamedAttr("producer_indices",
                                 AffineMapAttr::get(relation->producerIndices)),
            builder.getNamedAttr("consumer_indices",
                                 AffineMapAttr::get(relation->consumerIndices)),
            builder.getNamedAttr(
                "storage_capacity",
                builder.getI64IntegerAttr(relation->storageCapacity)),
            builder.getNamedAttr("storage_slot",
                                 AffineMapAttr::get(relation->storageSlot)),
            builder.getNamedAttr("generation_index",
                                 AffineMapAttr::get(relation->generationIndex)),
            builder.getNamedAttr("generations", builder.getI64IntegerAttr(1)),
            builder.getNamedAttr("notify_count", builder.getI64IntegerAttr(1)),
            builder.getNamedAttr(
                "scope",
                EventScopeAttr::get(module.getContext(), EventScope::Device)),
            builder.getNamedAttr("release", builder.getBoolAttr(true)),
            builder.getNamedAttr("acquire", builder.getBoolAttr(true))};
        if (relation->predicate)
          eventAttributes.push_back(builder.getNamedAttr(
              "relation_predicate", IntegerSetAttr::get(*relation->predicate)));
        createScheduleRecord(builder, task.getLoc(),
                             EventTensorOp::getOperationName(),
                             eventAttributes);
        ++eventOrdinal;
      }
      if (!parametricDependencies.empty())
        schedule->setAttr("parametric_dependencies",
                          builder.getArrayAttr(parametricDependencies));
      int64_t nextEventOrdinal = eventOrdinal;
      for (ScheduleTaskOp task : scheduledTasks) {
        if (task.getKind() != ScheduleTaskKind::AllAxisFold)
          continue;
        const int64_t partialCount = task.getGridShape().front();
        AffineExpr point = getAffineDimExpr(0, module.getContext());
        AffineExpr zero = getAffineConstantExpr(0, module.getContext());
        createScheduleRecord(
            builder, task.getLoc(), EventTensorOp::getOperationName(),
            {builder.getNamedAttr(
                 "ordinal", builder.getI64IntegerAttr(nextEventOrdinal++)),
             builder.getNamedAttr("producer_task",
                                  builder.getI64IntegerAttr(task.getOrdinal())),
             builder.getNamedAttr("consumer_task",
                                  builder.getI64IntegerAttr(task.getOrdinal())),
             builder.getNamedAttr("relation_domain_shape",
                                  builder.getDenseI64ArrayAttr({partialCount})),
             builder.getNamedAttr("producer_indices",
                                  AffineMapAttr::get(AffineMap::get(
                                      1, 0, point, module.getContext()))),
             builder.getNamedAttr("consumer_indices",
                                  AffineMapAttr::get(AffineMap::get(
                                      1, 0, zero, module.getContext()))),
             builder.getNamedAttr("storage_capacity",
                                  builder.getI64IntegerAttr(partialCount)),
             builder.getNamedAttr("storage_slot",
                                  AffineMapAttr::get(AffineMap::get(
                                      1, 0, point, module.getContext()))),
             builder.getNamedAttr("generation_index",
                                  AffineMapAttr::get(AffineMap::get(
                                      1, 0, zero, module.getContext()))),
             builder.getNamedAttr("generations", builder.getI64IntegerAttr(1)),
             builder.getNamedAttr("notify_count", builder.getI64IntegerAttr(1)),
             builder.getNamedAttr(
                 "scope",
                 EventScopeAttr::get(module.getContext(), EventScope::Device)),
             builder.getNamedAttr("release", builder.getBoolAttr(true)),
             builder.getNamedAttr("acquire", builder.getBoolAttr(true)),
             builder.getNamedAttr("producer_phase",
                                  builder.getStringAttr("partial")),
             builder.getNamedAttr("consumer_phase",
                                  builder.getStringAttr("merge"))});
      }
      builder.create<SchedulePlanYieldOp>(module.getLoc());
      schedule.setFingerprint(schedulePlanFingerprint(schedule));
      if (failed(schedule.verifyRegions())) {
        return signalPassFailure();
      }
    }
  }
};

LogicalResult
verifySchedulePlanAgainstMaterialization(MaterializationPlanOp source,
                                         SchedulePlanOp schedule) {
  if (schedule.getSchemaVersion() != source.getSchemaVersion() ||
      schedule.getTarget() != ScheduleTarget::Simt32 ||
      schedule.getPolicy() != source.getPolicy() ||
      schedule.getSourcePlanFingerprint() != source.getFingerprint()) {
    return schedule.emitOpError(
        "schema, target, policy, and source fingerprint must bind "
        "materialization");
  }
  SmallVector<MaterializationBufferOp> sourceBuffers(
      source.getBody().front().getOps<MaterializationBufferOp>());
  SmallVector<MaterializationTaskOp> sourceTasks(
      source.getBody().front().getOps<MaterializationTaskOp>());
  SmallVector<ScheduleBufferOp> buffers(
      schedule.getBody().front().getOps<ScheduleBufferOp>());
  SmallVector<ScheduleTaskOp> tasks(
      schedule.getBody().front().getOps<ScheduleTaskOp>());
  SmallVector<EventTensorOp> events(
      schedule.getBody().front().getOps<EventTensorOp>());
  if (buffers.size() != sourceBuffers.size() ||
      tasks.size() != sourceTasks.size()) {
    return schedule.emitOpError(
        "must cover every materialization buffer and task exactly once");
  }
  for (auto [ordinal, actual, expected] :
       llvm::enumerate(buffers, sourceBuffers)) {
    if (actual.getSourceBuffer() != static_cast<int64_t>(ordinal) ||
        actual.getTensorType() != expected.getTensorType() ||
        actual.getLifetimeStart() != expected.getLifetimeStart() ||
        actual.getLifetimeEnd() != expected.getLifetimeEnd()) {
      return actual.emitOpError(
          "schedule buffer type and lifetime must equal materialization");
    }
  }
  for (auto [ordinal, actual, expected] : llvm::enumerate(tasks, sourceTasks)) {
    FailureOr<ScheduleTaskKind> expectedKind =
        scheduleKindForMaterializationTask(expected);
    if (failed(expectedKind)) {
      return expected.emitOpError("does not have a closed SIMT32 task kind");
    }
    auto expectedContractStrategy =
        expected->getAttr("contract_region_strategy");
    auto expectedContractCost = expected->getAttr("contract_region_cost");
    auto expectedCompositionKind =
        expected->getAttrOfType<StringAttr>("composition_kind");
    const bool indexedAccumulation =
        expectedCompositionKind &&
        expectedCompositionKind.getValue() == "indexed_accumulation";
    if (actual.getSourceTask() != static_cast<int64_t>(ordinal) ||
        actual.getKind() != *expectedKind ||
        actual.getDomainShape() != expected.getDomainShape() ||
        actual.getInputBuffers() != expected.getInputBuffers() ||
        actual.getOutputBuffers() != expected.getOutputBuffers() ||
        actual.getDependencies() != expected.getDependencies() ||
        actual.getSemanticFingerprint() != expected.getSemanticFingerprint() ||
        actual.getReductionDimensionsAttr() !=
            (*expectedKind == ScheduleTaskKind::AllAxisFold ||
                     indexedAccumulation
                 ? expected.getReductionDimensionsAttr()
                 : DenseI64ArrayAttr()) ||
        actual->getAttr("contract_region_strategy") !=
            expectedContractStrategy ||
        actual->getAttr("contract_region_cost") != expectedContractCost) {
      return actual.emitOpError(
          "schedule dependencies must equal the materialization task");
    }
    for (StringRef name :
         {"composition_owner_dimensions", "composition_stream_dimensions",
          "composition_state_inputs", "composition_state_outputs",
          "composition_producer_contraction_dimensions",
          "composition_consumer_contraction_dimensions",
          "composition_index_domain_source", "composition_index_domain_shape",
          "composition_index_domain_map", "composition_ordered_index_source",
          "composition_ordered_index_dimensions",
          "composition_ordered_index_inclusive",
          "composition_state_access_buffers", "composition_state_access_maps"})
      if (actual->getAttr(name) != expected->getAttr(name))
        return actual.emitOpError(
            "structural composition metadata must equal materialization");
    FailureOr<Operation *> bound = boundAlgebraOperation(expected);
    if (failed(bound))
      return expected.emitOpError("cannot bind its generic algebra operation");
    if (auto relation = dyn_cast<RelationOp>(*bound)) {
      if (actual->getAttr("relation_semantics") !=
              relation.getSemanticsAttr() ||
          actual->getAttr("relation_parameters") !=
              relation.getParametersAttr())
        return actual.emitOpError(
            "Relation schedule metadata must equal bound algebra semantics");
    } else if (auto selection = dyn_cast<SelectionOp>(*bound)) {
      if (actual->getAttr("selection_dimension") !=
              selection.getDimensionAttr() ||
          actual->getAttr("selection_count") != selection.getCountAttr() ||
          actual->getAttr("selection_order") != selection.getOrderAttr() ||
          actual->getAttr("selection_comparison") !=
              selection.getComparisonAttr() ||
          actual->getAttr("selection_stable") != selection.getStableAttr() ||
          actual->getAttr("selection_tie_break") != selection.getTieBreakAttr())
        return actual.emitOpError(
            "Selection schedule metadata must equal bound algebra semantics");
    } else if (actual->hasAttr("relation_semantics") ||
               actual->hasAttr("relation_parameters") ||
               actual->hasAttr("selection_dimension") ||
               actual->hasAttr("selection_count") ||
               actual->hasAttr("selection_order") ||
               actual->hasAttr("selection_comparison") ||
               actual->hasAttr("selection_stable") ||
               actual->hasAttr("selection_tie_break")) {
      return actual.emitOpError(
          "logical task metadata requires matching bound algebra semantics");
    }
    if (actual.getOnlineVectorDimensionsAttr() !=
        expected.getOnlineVectorDimensionsAttr())
      return actual.emitOpError(
          "online vector dimensions must equal materialization semantics");
    if (actual.getOnlineMma().value_or(false) !=
        expected.getOnlineMma().value_or(false))
      return actual.emitOpError(
          "online MMA capability must equal materialization semantics");
    std::optional<int64_t> expectedStateBytes;
    if (FailureOr<FoldOp> fold = contractFoldCompositionSource(expected);
        succeeded(fold)) {
      FailureOr<int64_t> bytes = foldAccumulatorBytes(*fold);
      if (failed(bytes))
        return expected.emitOpError(
            "does not have bounded Fold accumulator storage");
      expectedStateBytes = *bytes;
    } else if (auto fold = dyn_cast<FoldOp>(*bound);
               fold && *expectedKind == ScheduleTaskKind::RowFold &&
               expected.getReductionDimensions().size() == 1 &&
               expected.getReductionDimensions().front() !=
                   static_cast<int64_t>(expected.getDomainShape().size()) - 1) {
      FailureOr<int64_t> bytes = foldAccumulatorBytes(fold);
      if (failed(bytes))
        return expected.emitOpError(
            "does not have bounded Fold accumulator storage");
      expectedStateBytes = *bytes;
    }
    if (actual.getReductionStateBytesPerThread() != expectedStateBytes)
      return actual.emitOpError(
          "Fold accumulator storage must equal materialization semantics");
    FailureOr<ScheduledGeometry> expectedGeometry =
        expected->getAttrOfType<ArrayAttr>("composition_phases")
            ? deriveMultiPhaseCompositionGeometry(expected)
        : isOnlineStateTask(expected)    ? deriveOnlineStateGeometry(expected)
        : isReverseStateTask(expected) ? deriveReverseStateGeometry(expected)
        : expected->getAttrOfType<StringAttr>("composition_kind") &&
                expected->getAttrOfType<StringAttr>("composition_kind")
                        .getValue() == "indexed_accumulation"
            ? deriveIndexedAccumulationGeometry(expected)
        : expected->getAttrOfType<StringAttr>("composition_kind") &&
                expected->getAttrOfType<StringAttr>("composition_kind")
                        .getValue() == "contract_fold"
            ? deriveContractFoldCompositionGeometry(expected,
                                                    source.getPolicy())
        : expected->hasAttr("contract_region_sources")
            ? deriveContractRegionGeometry(expected, source.getPolicy())
            : deriveSimt32Geometry(
                  *expectedKind, expected.getDomainShape(), source.getPolicy(),
                  expected.getKind() == MaterializationTaskKind::Fold &&
                          expected.getReductionDimensions().size() == 1
                      ? std::optional<int64_t>(
                            expected.getReductionDimensions().front())
                      : std::nullopt);
    if (failed(expectedGeometry) ||
        actual.getGridShape() != ArrayRef<int64_t>(expectedGeometry->grid) ||
        actual.getTileShape() != ArrayRef<int64_t>(expectedGeometry->tile) ||
        actual.getSerialTiles() != expectedGeometry->serialTiles ||
        actual.getWorkgroupThreads() != expectedGeometry->workgroupThreads ||
        actual.getScratchBytes() != expectedGeometry->scratchBytes ||
        actual.getReductionAxis() != expectedGeometry->reductionAxis ||
        actual.getReductionMergeGridShapeAttr() !=
            (expectedGeometry->reductionMergeGrid.empty()
                 ? DenseI64ArrayAttr()
                 : DenseI64ArrayAttr::get(
                       actual.getContext(),
                       expectedGeometry->reductionMergeGrid)) ||
        actual.getReductionMergeTileShapeAttr() !=
            (expectedGeometry->reductionMergeTile.empty()
                 ? DenseI64ArrayAttr()
                 : DenseI64ArrayAttr::get(
                       actual.getContext(),
                       expectedGeometry->reductionMergeTile)) ||
        actual.getReductionMergeSerialTilesAttr() !=
            (expectedGeometry->reductionMergeSerialTiles == 0
                 ? IntegerAttr()
                 : IntegerAttr::get(
                       IntegerType::get(actual.getContext(), 64),
                       expectedGeometry->reductionMergeSerialTiles)) ||
        actual.getReductionPartialBufferShapeAttr() !=
            (expectedGeometry->reductionPartialBufferShape.empty()
                 ? DenseI64ArrayAttr()
                 : DenseI64ArrayAttr::get(
                       actual.getContext(),
                       expectedGeometry->reductionPartialBufferShape)) ||
        actual.getReductionOrder() !=
            (expectedGeometry->reductionAxis ||
                     *expectedKind == ScheduleTaskKind::AllAxisFold
                 ? std::optional<ScheduleReductionOrder>(
                       scheduleReductionOrderForTask(expected, *expectedKind,
                                                     source.getPolicy()))
            : std::nullopt))
      return actual.emitOpError(
                 "schedule geometry must equal the certified derivation: ")
             << "grid " << actual.getGridShape() << " versus "
             << ArrayRef<int64_t>(expectedGeometry->grid) << ", tile "
             << actual.getTileShape() << " versus "
             << ArrayRef<int64_t>(expectedGeometry->tile) << ", serial tiles "
             << actual.getSerialTiles() << " versus "
             << expectedGeometry->serialTiles << ", workgroup threads "
             << actual.getWorkgroupThreads() << " versus "
             << expectedGeometry->workgroupThreads << ", scratch bytes "
             << actual.getScratchBytes() << " versus "
             << expectedGeometry->scratchBytes;
    if (auto phases = expected->getAttrOfType<ArrayAttr>("composition_phases");
        phases) {
      OpBuilder builder(expected.getContext());
      FailureOr<ArrayAttr> expectedPhaseSchedules =
          multiPhaseCompositionScheduleRecords(builder, expected);
      if (failed(expectedPhaseSchedules) ||
          actual->getAttr("composition_phase_schedules") !=
              *expectedPhaseSchedules)
        return actual.emitOpError(
            "multi-phase CTA ranges must equal the typed phase derivation");
    } else if (actual->hasAttr("composition_phase_schedules")) {
      return actual.emitOpError(
          "single-phase task must not carry multi-phase CTA ranges");
    }
  }
  SmallVector<std::pair<int64_t, int64_t>> expectedEdges;
  for (auto [consumerOrdinal, task] : llvm::enumerate(sourceTasks))
    for (int64_t producerOrdinal : task.getDependencies())
      expectedEdges.push_back(
          {producerOrdinal, static_cast<int64_t>(consumerOrdinal)});
  const int64_t localReductionEvents =
      llvm::count_if(tasks, [](ScheduleTaskOp task) {
        return task.getKind() == ScheduleTaskKind::AllAxisFold;
      });
  ArrayAttr parametricDependencies = schedule.getParametricDependenciesAttr();
  const size_t parametricDependencyCount =
      parametricDependencies ? parametricDependencies.size() : 0;
  if (events.size() + parametricDependencyCount !=
      expectedEdges.size() + localReductionEvents)
    return schedule.emitOpError(
        "must represent every structural task dependence exactly once");
  llvm::SmallDenseMap<int64_t, DictionaryAttr> parametricByDependency;
  if (parametricDependencies)
    for (Attribute attribute : parametricDependencies) {
      auto program = cast<DictionaryAttr>(attribute);
      parametricByDependency.try_emplace(
          cast<IntegerAttr>(program.get("dependency_ordinal")).getInt(),
          program);
    }
  int64_t eventOrdinal = 0;
  for (auto [dependencyOrdinal, edge] : llvm::enumerate(expectedEdges)) {
    const auto [producerOrdinal, consumerOrdinal] = edge;
    FailureOr<ScheduledTaskRelation> expected = deriveScheduledTaskRelation(
        tasks[producerOrdinal], tasks[consumerOrdinal],
        sourceTasks[producerOrdinal], sourceTasks[consumerOrdinal],
        sourceBuffers);
    if (failed(expected)) {
      auto found = parametricByDependency.find(dependencyOrdinal);
      FailureOr<Operation *> producer =
          boundAlgebraOperation(sourceTasks[producerOrdinal]);
      auto gather =
          succeeded(producer) ? dyn_cast<RelationOp>(*producer) : RelationOp{};
      DictionaryAttr program = found == parametricByDependency.end()
                                   ? DictionaryAttr{}
                                   : found->second;
      FailureOr<int64_t> capacity =
          scheduleElementCount(tasks[consumerOrdinal].getGridShape());
      if (!program || !gather ||
          gather.getSemantics() != RelationSemantics::Gather ||
          program.get("relation_source") != gather.getSourceAttr() ||
          cast<IntegerAttr>(program.get("producer_task")).getInt() !=
              producerOrdinal ||
          cast<IntegerAttr>(program.get("consumer_task")).getInt() !=
              consumerOrdinal ||
          failed(capacity) ||
          cast<IntegerAttr>(program.get("storage_capacity")).getInt() !=
              *capacity)
        return schedule.emitOpError(
            "cannot derive the structural tiled task dependence");
      parametricByDependency.erase(found);
      continue;
    }
    if (parametricByDependency.contains(dependencyOrdinal))
      return schedule.emitOpError(
          "affine task dependence must not use a parametric program");
    if (eventOrdinal >= static_cast<int64_t>(events.size()))
      return schedule.emitOpError(
          "static task dependence is missing its exact Event Tensor");
    EventTensorOp event = events[eventOrdinal];
    if (event.getOrdinal() != static_cast<int64_t>(eventOrdinal) ||
        event.getProducerTask() != producerOrdinal ||
        event.getConsumerTask() != consumerOrdinal ||
        event.getRelationDomainShape() != ArrayRef<int64_t>(expected->domain) ||
        event.getRelationPredicate() != expected->predicate ||
        event.getProducerIndices() != expected->producerIndices ||
        event.getConsumerIndices() != expected->consumerIndices ||
        event.getStorageCapacity() != expected->storageCapacity ||
        event.getStorageSlot() != expected->storageSlot ||
        event.getGenerationIndex() != expected->generationIndex)
      return event.emitOpError(
          "must equal the structural tiled task dependence");
    ++eventOrdinal;
  }
  if (!parametricByDependency.empty())
    return schedule.emitOpError(
        "parametric dependency does not bind a structural task edge");
  int64_t localOrdinal = eventOrdinal;
  for (ScheduleTaskOp task : tasks) {
    if (task.getKind() != ScheduleTaskKind::AllAxisFold)
      continue;
    if (localOrdinal >= static_cast<int64_t>(events.size()))
      return schedule.emitOpError(
          "all-axis Fold is missing its partial-to-merge Event Tensor");
    EventTensorOp event = events[localOrdinal];
    if (event.getOrdinal() != localOrdinal ||
        event.getProducerTask() != task.getOrdinal() ||
        event.getConsumerTask() != task.getOrdinal() ||
        event.getProducerPhase() != "partial" ||
        event.getConsumerPhase() != "merge" ||
        event.getRelationDomainShape() != task.getGridShape())
      return event.emitOpError(
          "must equal the hierarchical Fold partial-to-merge dependence");
    ++localOrdinal;
  }
  return success();
}

class VerifySimt32RowFoldSchedulePass
    : public impl::ShuttleVerifySimt32RowFoldSchedulePassBase<
          VerifySimt32RowFoldSchedulePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<MaterializationPlanOp> materializationPlans(
        module.getOps<MaterializationPlanOp>());
    SmallVector<SchedulePlanOp> schedules(module.getOps<SchedulePlanOp>());
    if (materializationPlans.empty() ||
        materializationPlans.size() != schedules.size()) {
      module.emitError("requires one valid schedule per materialization plan");
      return signalPassFailure();
    }
    llvm::SmallDenseSet<Operation *> matchedSchedules;
    for (MaterializationPlanOp materialization : materializationPlans) {
      SchedulePlanOp match;
      for (SchedulePlanOp schedule : schedules) {
        if (schedule.getSourcePlanFingerprint() ==
            materialization.getFingerprint()) {
          if (match) {
            module.emitError("materialization plan has duplicate schedules");
            return signalPassFailure();
          }
          match = schedule;
        }
      }
      if (!match || !matchedSchedules.insert(match.getOperation()).second ||
          failed(verifyMaterializationPlanAgainstSource(module,
                                                        materialization)) ||
          failed(match.verifyRegions()) ||
          failed(verifySchedulePlanAgainstMaterialization(materialization,
                                                          match))) {
        module.emitError(
            "requires one valid schedule per materialization plan");
        return signalPassFailure();
      }
    }
  }
};

struct ExecutableSlotSpec {
  Type tensorType;
  int64_t requiredBytes;
  SmallVector<int64_t> strides;
  int64_t alignment;
  ExecutableAccess access;
  MaterializationStorage storage;
};

struct GpuEntrySpec {
  int64_t ordinal;
  int64_t codeOffset;
  int64_t codeLength;
  SmallVector<int64_t> inputs;
  SmallVector<int64_t> outputs;
  SmallVector<int64_t> dependencies;
  ExecutablePredication predication;
  std::optional<ScheduleReductionOrder> reductionOrder;
  std::array<int64_t, 3> grid;
  std::array<int64_t, 3> block;
  int64_t dynamicSharedBytes;
  int64_t kernelArity;
  std::optional<std::string> onlineStateProtocol;
  SmallVector<AffineMap> logicalCoordinateMaps;
  std::string codeDigest;
};

struct GpuExecutableSpec {
  SmallVector<int8_t> code;
  uint64_t rawFatbinCodeBytes = 0;
  std::vector<GpuEntrySpec> entries;
  SmallVector<ExecutableSlotSpec> slots;
  SmallVector<int64_t> reuseGroups;
  SmallVector<std::optional<int64_t>> bindingIndices;
  SmallVector<int64_t> sourceBuffers;
};

SmallVector<int64_t> publicRegionResultBindings(RegionOp region) {
  SmallVector<int64_t> bindings(region.getNumResults(), -1);
  int64_t nextBinding = 0;
  if (auto function = region->getParentOfType<func::FuncOp>()) {
    if (auto returnOp =
            dyn_cast<func::ReturnOp>(function.getBody().front().back())) {
      for (Value value : returnOp.getOperands()) {
        auto result = dyn_cast<OpResult>(value);
        if (!result || result.getOwner() != region ||
            bindings[result.getResultNumber()] >= 0)
          continue;
        bindings[result.getResultNumber()] = nextBinding++;
      }
    }
  }
  for (int64_t &binding : bindings)
    if (binding < 0)
      binding = nextBinding++;
  return bindings;
}

FailureOr<int64_t> staticElementCount(RankedTensorType type) {
  int64_t elements = 1;
  for (int64_t extent : type.getShape()) {
    if (extent <= 0 || elements > std::numeric_limits<int64_t>::max() / extent)
      return failure();
    elements *= extent;
  }
  return elements;
}

FailureOr<SmallVector<ExecutableSlotSpec>>
deriveExecutableSlots(MaterializationPlanOp materialization) {
  SmallVector<ExecutableSlotSpec> slots;
  for (MaterializationBufferOp buffer :
       materialization.getBody().front().getOps<MaterializationBufferOp>()) {
    auto type = dyn_cast<RankedTensorType>(buffer.getTensorType());
    if (!type || !type.hasStaticShape())
      return failure();
    Type elementType = type.getElementType();
    int64_t elementBytes = 0;
    if (elementType.isBF16())
      elementBytes = 2;
    else if (elementType.isF32())
      elementBytes = 4;
    else if (auto integerType = dyn_cast<IntegerType>(elementType);
             integerType && integerType.getWidth() <= 32)
      elementBytes = std::max<int64_t>(1, (integerType.getWidth() + 7) / 8);
    else {
      buffer.emitError("GPU executable ABI does not support element type ")
          << elementType;
      return failure();
    }
    FailureOr<int64_t> elements = staticElementCount(type);
    if (failed(elements) ||
        *elements > std::numeric_limits<int64_t>::max() / elementBytes)
      return failure();
    SmallVector<int64_t> strides(type.getRank());
    int64_t stride = elementBytes;
    for (int64_t axis = type.getRank(); axis > 0; --axis) {
      strides[axis - 1] = stride;
      stride *= type.getDimSize(axis - 1);
    }
    ExecutableAccess access = ExecutableAccess::ReadWrite;
    if (buffer.getLiveIn())
      access = ExecutableAccess::Read;
    else if (buffer.getLiveOut())
      access = ExecutableAccess::Write;
    slots.push_back(ExecutableSlotSpec{type, *elements * elementBytes,
                                       std::move(strides), elementBytes, access,
                                       buffer.getStorage()});
  }
  return slots;
}

FailureOr<SmallVector<int64_t>>
deriveGpuReuseGroups(MaterializationPlanOp materialization,
                     ArrayRef<ExecutableSlotSpec> slots) {
  SmallVector<MaterializationBufferOp> buffers(
      materialization.getBody().front().getOps<MaterializationBufferOp>());
  if (buffers.size() != slots.size())
    return failure();
  struct Candidate {
    int64_t ordinal;
    int64_t lifetimeStart;
    int64_t lifetimeEnd;
    int64_t requiredBytes;
  };
  struct ReuseGroup {
    int64_t ordinal;
    SmallVector<std::pair<int64_t, int64_t>> intervals;
  };
  SmallVector<Candidate> candidates;
  SmallVector<int64_t> groups(slots.size(), -1);
  for (auto [ordinal, buffer, slot] : llvm::enumerate(buffers, slots)) {
    if (slot.storage != MaterializationStorage::Temporary) {
      groups[ordinal] = ordinal;
      continue;
    }
    if (slot.requiredBytes <= 0 || buffer.getLifetimeStart() < 0 ||
        buffer.getLifetimeEnd() < buffer.getLifetimeStart())
      return failure();
    candidates.push_back({static_cast<int64_t>(ordinal),
                          static_cast<int64_t>(buffer.getLifetimeStart()),
                          static_cast<int64_t>(buffer.getLifetimeEnd()),
                          slot.requiredBytes});
  }
  llvm::sort(candidates, [](const Candidate &left, const Candidate &right) {
    if (left.requiredBytes != right.requiredBytes)
      return left.requiredBytes > right.requiredBytes;
    if (left.lifetimeStart != right.lifetimeStart)
      return left.lifetimeStart < right.lifetimeStart;
    if (left.lifetimeEnd != right.lifetimeEnd)
      return left.lifetimeEnd < right.lifetimeEnd;
    return left.ordinal < right.ordinal;
  });
  SmallVector<ReuseGroup> available;
  for (const Candidate &candidate : candidates) {
    auto reusable = llvm::find_if(available, [&](const ReuseGroup &group) {
      return llvm::all_of(group.intervals, [&](const auto &interval) {
        return candidate.lifetimeEnd < interval.first ||
               interval.second < candidate.lifetimeStart;
      });
    });
    if (reusable == available.end()) {
      available.push_back(
          {static_cast<int64_t>(slots.size() + available.size()), {}});
      reusable = std::prev(available.end());
    }
    reusable->intervals.push_back(
        {candidate.lifetimeStart, candidate.lifetimeEnd});
    groups[candidate.ordinal] = reusable->ordinal;
  }
  if (llvm::is_contained(groups, -1))
    return failure();
  return groups;
}

FailureOr<SmallVector<std::optional<int64_t>>>
deriveGpuBindingIndices(RegionOp region, const DerivedPlan &derived,
                        ArrayRef<ExecutableSlotSpec> slots) {
  auto regionYield = cast<YieldOp>(region.getBody().front().getTerminator());
  SmallVector<int64_t> resultBindings = publicRegionResultBindings(region);
  if (derived.buffers.size() != slots.size())
    return failure();
  SmallVector<std::optional<int64_t>> bindingIndices(slots.size());
  llvm::SmallDenseSet<int64_t> seenOperandBindings;
  llvm::SmallDenseSet<int64_t> seenResultBindings;
  for (auto [ordinal, buffer, slot] : llvm::enumerate(derived.buffers, slots)) {
    if (slot.storage != MaterializationStorage::External)
      continue;
    if (!buffer.producer) {
      auto argument = dyn_cast<BlockArgument>(buffer.value);
      if (!argument || argument.getOwner() != &region.getBody().front() ||
          !seenOperandBindings.insert(argument.getArgNumber()).second)
        return failure();
      bindingIndices[ordinal] = argument.getArgNumber();
      continue;
    }
    if (!buffer.liveOut)
      return failure();
    auto yielded = llvm::find(regionYield.getValues(), buffer.value);
    if (yielded == regionYield.getValues().end())
      return failure();
    int64_t resultIndex =
        std::distance(regionYield.getValues().begin(), yielded);
    const int64_t binding = resultBindings[resultIndex];
    if (!seenResultBindings.insert(binding).second)
      return failure();
    bindingIndices[ordinal] = binding;
  }
  if (seenOperandBindings.size() !=
          region.getBody().front().getNumArguments() ||
      seenResultBindings.size() != regionYield.getValues().size())
    return failure();
  return bindingIndices;
}

FailureOr<SmallVector<std::optional<int64_t>>>
deriveGpuRegionResultIndices(RegionOp region, const DerivedPlan &derived,
                             ArrayRef<ExecutableSlotSpec> slots) {
  auto regionYield = cast<YieldOp>(region.getBody().front().getTerminator());
  if (derived.buffers.size() != slots.size())
    return failure();
  SmallVector<std::optional<int64_t>> resultIndices(slots.size());
  for (auto [ordinal, buffer, slot] : llvm::enumerate(derived.buffers, slots)) {
    if (slot.storage != MaterializationStorage::External || !buffer.producer)
      continue;
    if (!buffer.liveOut)
      return failure();
    auto yielded = llvm::find(regionYield.getValues(), buffer.value);
    if (yielded == regionYield.getValues().end())
      return failure();
    resultIndices[ordinal] =
        std::distance(regionYield.getValues().begin(), yielded);
  }
  return resultIndices;
}

struct PtxEmitter {
  explicit PtxEmitter(int64_t arity) : arity(arity), stream(body) {}

  void header(StringRef certificate, bool usesDynamicSharedMemory = false) {
    this->certificate = certificate.str();
    this->usesDynamicSharedMemory = usesDynamicSharedMemory;
  }

  int64_t nextF32() { return f32++; }
  int64_t nextU64() { return u64++; }
  int64_t nextU32() { return u32++; }
  int64_t nextPred() { return pred++; }
  int64_t nextBF16() { return bf16++; }

  void finish() {
    stream.flush();
    llvm::raw_string_ostream output(text);
    output << ".version 8.0\n.target sm_90\n.address_size 64\n";
    if (usesDynamicSharedMemory)
      output << ".extern .shared .align 4 .b8 shuttle_smem[];\n";
    output << "// shuttle structural certificate: " << certificate
           << "\n.visible .entry shuttle_entry(";
    for (int64_t index = 0; index < arity; ++index) {
      if (index)
        output << ", ";
      output << ".param .u64 p" << index;
    }
    output << ")\n{\n"
           << "  .reg .pred %p<" << std::max<int64_t>(64, pred) << ">;\n"
           << "  .reg .b16 %h<" << std::max<int64_t>(16, bf16) << ">;\n"
           << "  .reg .b32 %r<" << std::max<int64_t>(128, u32) << ">;\n"
           << "  .reg .b64 %rd<" << std::max<int64_t>(128, u64) << ">;\n"
           << "  .reg .f32 %f<" << std::max<int64_t>(128, f32) << ">;\n"
           << body << "$done:\n  ret;\n}\n";
    output.flush();
  }

  int64_t arity;
  int64_t f32 = 0;
  int64_t u64 = 8;
  int64_t u32 = 8;
  int64_t pred = 4;
  int64_t bf16 = 2;
  std::string certificate;
  bool usesDynamicSharedMemory = false;
  std::string body;
  std::string text;
  llvm::raw_string_ostream stream;
};

FailureOr<GpuTarget> parseGpuTarget(StringRef target) {
  if (target == "sm_90")
    return GpuTarget::Sm90;
  if (target == "sm_100")
    return GpuTarget::Sm100;
  return failure();
}

ExecutableCodeFormat codeFormatForTarget(GpuTarget target) {
  return target == GpuTarget::Sm90 ? ExecutableCodeFormat::CudaPtxSm90V1
                                   : ExecutableCodeFormat::CudaPtxSm100V1;
}

ExecutableCodeFormat fatbinCodeFormatForTarget(GpuTarget target) {
  return target == GpuTarget::Sm90 ? ExecutableCodeFormat::CudaFatbinSm90V1
                                   : ExecutableCodeFormat::CudaFatbinSm100V1;
}

int64_t deviceSchemaForTarget(GpuTarget target) {
  return target == GpuTarget::Sm90 ? 3 : 4;
}

int64_t fatbinDeviceSchemaForTarget(GpuTarget target) {
  return target == GpuTarget::Sm90 ? 5 : 6;
}

FailureOr<GpuTarget> targetForCodeFormat(ExecutableCodeFormat format) {
  if (format == ExecutableCodeFormat::CudaPtxSm90V1)
    return GpuTarget::Sm90;
  if (format == ExecutableCodeFormat::CudaPtxSm100V1)
    return GpuTarget::Sm100;
  if (format == ExecutableCodeFormat::CudaFatbinSm90V1)
    return GpuTarget::Sm90;
  if (format == ExecutableCodeFormat::CudaFatbinSm100V1)
    return GpuTarget::Sm100;
  return failure();
}

LogicalResult retargetPtx(std::string &ptx, GpuTarget target) {
  constexpr StringLiteral sm90Header = ".version 8.0\n.target sm_90\n";
  if (!StringRef(ptx).starts_with(sm90Header))
    return failure();
  if (target == GpuTarget::Sm100)
    ptx.replace(0, sm90Header.size(), ".version 8.7\n.target sm_100\n");
  return success();
}

LogicalResult emitLinearThread(PtxEmitter &ptx, int64_t elements) {
  if (elements <= 0 || elements > UINT32_MAX)
    return failure();
  ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                "  mov.u32 %r1, %ctaid.x;\n"
                "  mov.u32 %r2, %ntid.x;\n"
                "  mad.lo.u32 %r3, %r1, %r2, %r0;\n"
                "  setp.ge.u32 %p0, %r3, "
             << elements << ";\n  @%p0 bra $done;\n"
             << "  cvt.u64.u32 %rd0, %r3;\n";
  return success();
}

FailureOr<SmallVector<int64_t>> emitOutputCoordinates(PtxEmitter &ptx,
                                                      RankedTensorType type) {
  FailureOr<int64_t> elements = staticElementCount(type);
  if (failed(elements) || *elements > UINT32_MAX) {
    return failure();
  }
  SmallVector<int64_t> coordinates;
  int64_t stride = 1;
  SmallVector<int64_t> strides(type.getRank());
  for (int64_t axis = type.getRank(); axis > 0; --axis) {
    strides[axis - 1] = stride;
    stride *= type.getDimSize(axis - 1);
  }
  for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
    int64_t reg = ptx.nextU64();
    ptx.stream << "  div.u64 %rd" << reg << ", %rd0, " << strides[axis]
               << ";\n  rem.u64 %rd" << reg << ", %rd" << reg << ", " << extent
               << ";\n";
    coordinates.push_back(reg);
  }
  return coordinates;
}

FailureOr<int64_t> emitRowMajorLinearIndex(PtxEmitter &ptx,
                                           ArrayRef<int64_t> coordinates,
                                           ArrayRef<int64_t> shape) {
  if (coordinates.size() != shape.size() ||
      llvm::any_of(shape, [](int64_t extent) { return extent <= 0; }))
    return failure();
  int64_t linear = ptx.nextU64();
  ptx.stream << "  mov.u64 %rd" << linear << ", 0;\n";
  for (auto [coordinate, extent] : llvm::zip_equal(coordinates, shape))
    ptx.stream << "  mad.lo.u64 %rd" << linear << ", %rd" << linear << ", "
               << extent << ", %rd" << coordinate << ";\n";
  return linear;
}

FailureOr<int64_t> emitAffineCoordinate(PtxEmitter &ptx, AffineExpr expression,
                                        ArrayRef<int64_t> coordinates) {
  if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
    if (dimension.getPosition() >= coordinates.size())
      return failure();
    return coordinates[dimension.getPosition()];
  }
  if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
    if (constant.getValue() < 0)
      return failure();
    int64_t reg = ptx.nextU64();
    ptx.stream << "  mov.u64 %rd" << reg << ", " << constant.getValue()
               << ";\n";
    return reg;
  }
  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return failure();
  FailureOr<int64_t> lhs =
      emitAffineCoordinate(ptx, binary.getLHS(), coordinates);
  if (failed(lhs))
    return failure();
  StringRef instruction;
  FailureOr<int64_t> rhs = failure();
  if (binary.getKind() == AffineExprKind::Add) {
    instruction = "add";
    rhs = emitAffineCoordinate(ptx, binary.getRHS(), coordinates);
  } else if (binary.getKind() == AffineExprKind::Mul) {
    instruction = "mul.lo";
    rhs = emitAffineCoordinate(ptx, binary.getRHS(), coordinates);
  } else if (binary.getKind() == AffineExprKind::FloorDiv ||
             binary.getKind() == AffineExprKind::Mod) {
    auto divisor = dyn_cast<AffineConstantExpr>(binary.getRHS());
    if (!divisor || divisor.getValue() <= 0)
      return failure();
    instruction = binary.getKind() == AffineExprKind::FloorDiv ? "div" : "rem";
    rhs = emitAffineCoordinate(ptx, binary.getRHS(), coordinates);
  } else {
    return failure();
  }
  if (failed(rhs))
    return failure();
  int64_t reg = ptx.nextU64();
  ptx.stream << "  " << instruction << ".u64 %rd" << reg << ", %rd" << *lhs
             << ", %rd" << *rhs << ";\n";
  return reg;
}

FailureOr<int64_t> emitMapInput(PtxEmitter &ptx, int64_t parameter,
                                RankedTensorType type, AffineMap map,
                                ArrayRef<int64_t> coordinates) {
  if (!isSupportedGenericGpuAffineMap(map, coordinates.size(), type.getRank()))
    return failure();
  int64_t linear = ptx.nextU64();
  ptx.stream << "  mov.u64 %rd" << linear << ", 0;\n";
  for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
    FailureOr<int64_t> coordinate =
        emitAffineCoordinate(ptx, expression, coordinates);
    if (failed(coordinate))
      return failure();
    ptx.stream << "  mad.lo.u64 %rd" << linear << ", %rd" << linear << ", "
               << type.getDimSize(axis) << ", %rd" << *coordinate << ";\n";
  }
  int64_t pointer = ptx.nextU64();
  int64_t bytes = type.getElementType().isBF16() ? 2 : 4;
  ptx.stream << "  ld.param.u64 %rd" << pointer << ", [p" << parameter
             << "];\n  mad.lo.u64 %rd" << pointer << ", %rd" << linear << ", "
             << bytes << ", %rd" << pointer << ";\n";
  int64_t value = ptx.nextF32();
  if (type.getElementType().isBF16()) {
    ptx.stream << "  ld.global.u16 %h0, [%rd" << pointer
               << "];\n  cvt.f32.bf16 %f" << value << ", %h0;\n";
  } else {
    ptx.stream << "  ld.global.f32 %f" << value << ", [%rd" << pointer
               << "];\n";
  }
  return value;
}

FailureOr<SmallVector<int64_t>>
emitScalarResults(PtxEmitter &ptx, Block &body, ArrayRef<int64_t> arguments,
                  bool allowBf16Arithmetic = false) {
  if (body.getNumArguments() != arguments.size())
    return failure();
  DenseMap<Value, int64_t> values;
  for (auto [argument, reg] : llvm::zip_equal(body.getArguments(), arguments))
    values[argument] = reg;
  for (Operation &operation : body.without_terminator()) {
    if (operation.getNumResults() != 1)
      return failure();
    Type resultType = operation.getResult(0).getType();
    if (!isa<ScalarConvertOp>(operation) && !resultType.isF32() &&
        !(allowBf16Arithmetic && resultType.isBF16()) &&
        !(isa<arith::ConstantOp>(operation) && resultType.isBF16()))
      return failure();
    int64_t result = ptx.nextF32();
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      auto value = dyn_cast<FloatAttr>(constant.getValue());
      if (!value || (!value.getType().isF32() && !value.getType().isBF16()))
        return failure();
      llvm::APFloat floatValue = value.getValue();
      bool losesInformation = false;
      floatValue.convert(llvm::APFloat::IEEEsingle(),
                         llvm::APFloat::rmNearestTiesToEven, &losesInformation);
      uint32_t bits = floatValue.bitcastToAPInt().getZExtValue();
      ptx.stream << "  mov.b32 %f" << result << ", 0f"
                 << llvm::utohexstr(bits, /*LowerCase=*/false, /*Width=*/8)
                 << ";\n";
    } else if (auto convert = dyn_cast<ScalarConvertOp>(operation)) {
      Type inputType = convert.getInput().getType();
      Type resultType = convert.getResult().getType();
      int64_t operand = values.lookup(convert.getInput());
      if ((inputType.isF32() && resultType.isF32()) ||
          (inputType.isBF16() && resultType.isF32())) {
        result = operand;
      } else if (inputType.isF32() && resultType.isBF16() &&
                 convert.getSemantics() ==
                     ScalarConvertSemantics::RoundNearestEven) {
        int64_t rounded = ptx.nextBF16();
        ptx.stream << "  cvt.rn.bf16.f32 %h" << rounded << ", %f" << operand
                   << ";\n  cvt.f32.bf16 %f" << result << ", %h" << rounded
                   << ";\n";
      } else {
        return failure();
      }
    } else if (auto add = dyn_cast<arith::AddFOp>(operation)) {
      ptx.stream << "  add.rn.f32 %f" << result << ", %f"
                 << values.lookup(add.getLhs()) << ", %f"
                 << values.lookup(add.getRhs()) << ";\n";
    } else if (auto subtract = dyn_cast<arith::SubFOp>(operation)) {
      ptx.stream << "  sub.rn.f32 %f" << result << ", %f"
                 << values.lookup(subtract.getLhs()) << ", %f"
                 << values.lookup(subtract.getRhs()) << ";\n";
    } else if (auto negate = dyn_cast<arith::NegFOp>(operation)) {
      ptx.stream << "  neg.f32 %f" << result << ", %f"
                 << values.lookup(negate.getOperand()) << ";\n";
    } else if (auto multiply = dyn_cast<arith::MulFOp>(operation)) {
      ptx.stream << "  mul.rn.f32 %f" << result << ", %f"
                 << values.lookup(multiply.getLhs()) << ", %f"
                 << values.lookup(multiply.getRhs()) << ";\n";
    } else if (auto divide = dyn_cast<arith::DivFOp>(operation)) {
      ptx.stream << "  div.rn.f32 %f" << result << ", %f"
                 << values.lookup(divide.getLhs()) << ", %f"
                 << values.lookup(divide.getRhs()) << ";\n";
    } else if (auto maximum = dyn_cast<arith::MaximumFOp>(operation)) {
      ptx.stream << "  max.f32 %f" << result << ", %f"
                 << values.lookup(maximum.getLhs()) << ", %f"
                 << values.lookup(maximum.getRhs()) << ";\n";
    } else if (auto exponential = dyn_cast<math::ExpOp>(operation)) {
      int64_t scaled = ptx.nextF32();
      ptx.stream << "  mul.rn.f32 %f" << scaled << ", %f"
                 << values.lookup(exponential.getOperand())
                 << ", 0f3FB8AA3B;\n  ex2.approx.f32 %f" << result << ", %f"
                 << scaled << ";\n";
    } else if (auto hyperbolic = dyn_cast<math::TanhOp>(operation)) {
      ptx.stream << "  tanh.approx.f32 %f" << result << ", %f"
                 << values.lookup(hyperbolic.getOperand()) << ";\n";
    } else if (auto rsqrt = dyn_cast<math::RsqrtOp>(operation)) {
      int64_t root = ptx.nextF32();
      ptx.stream << "  sqrt.rn.f32 %f" << root << ", %f"
                 << values.lookup(rsqrt.getOperand()) << ";\n  div.rn.f32 %f"
                 << result << ", 0f3F800000, %f" << root << ";\n";
    } else {
      return failure();
    }
    if (resultType.isBF16() && !isa<arith::ConstantOp>(operation)) {
      int64_t rounded = ptx.nextBF16();
      int64_t widened = ptx.nextF32();
      ptx.stream << "  cvt.rn.bf16.f32 %h" << rounded << ", %f" << result
                 << ";\n  cvt.f32.bf16 %f" << widened << ", %h" << rounded
                 << ";\n";
      result = widened;
    }
    if (operation.getNumResults() != 1)
      return failure();
    values[operation.getResult(0)] = result;
  }
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().empty())
    return failure();
  SmallVector<int64_t> results;
  for (Value value : yield.getValues()) {
    auto found = values.find(value);
    if (found == values.end())
      return failure();
    results.push_back(found->second);
  }
  return results;
}

FailureOr<int64_t> emitScalarBody(PtxEmitter &ptx, Block &body,
                                  ArrayRef<int64_t> arguments,
                                  bool allowBf16Arithmetic = false) {
  FailureOr<SmallVector<int64_t>> results =
      emitScalarResults(ptx, body, arguments, allowBf16Arithmetic);
  if (failed(results) || results->size() != 1)
    return failure();
  return results->front();
}

FailureOr<std::string> emitMapPtx(MapOp map, ScheduleTaskOp task) {
  auto output = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  if (!output || map.getNumResults() != 1)
    return failure();
  ArrayRef<Attribute> maps = map.getIndexingMaps().getValue();
  if (maps.size() != map.getInputs().size() + 1 ||
      !isIdentityMap(cast<AffineMapAttr>(maps.back()).getValue()))
    return failure();
  FailureOr<int64_t> elements = staticElementCount(output);
  if (failed(elements))
    return failure();
  if (output.getElementType().isInteger(1)) {
    Block &body = map.getBody().front();
    if (map.getInputs().size() != 2 || body.getNumArguments() != 2 ||
        body.getOperations().size() != 2)
      return failure();
    auto compare = dyn_cast<arith::CmpFOp>(body.front());
    auto yield = dyn_cast<YieldOp>(body.back());
    if (!compare || compare.getPredicate() != arith::CmpFPredicate::OEQ ||
        compare.getLhs() != body.getArgument(0) ||
        compare.getRhs() != body.getArgument(1) || !yield ||
        yield.getValues().size() != 1 ||
        yield.getValues().front() != compare.getResult())
      return failure();
    PtxEmitter ptx(/*arity=*/3);
    ptx.header(task.getSemanticFingerprint());
    if (failed(emitLinearThread(ptx, *elements)))
      return failure();
    FailureOr<SmallVector<int64_t>> coordinates =
        emitOutputCoordinates(ptx, output);
    if (failed(coordinates))
      return failure();
    SmallVector<int64_t> arguments;
    for (auto [index, input] : llvm::enumerate(map.getInputs())) {
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      auto indexing = dyn_cast<AffineMapAttr>(maps[index]);
      if (!inputType || !inputType.getElementType().isF32() || !indexing)
        return failure();
      FailureOr<int64_t> value = emitMapInput(
          ptx, index, inputType, indexing.getValue(), *coordinates);
      if (failed(value))
        return failure();
      arguments.push_back(*value);
    }
    int64_t predicate = ptx.nextPred();
    int64_t encoded = ptx.nextU32();
    int64_t outputPointer = ptx.nextU64();
    ptx.stream << "  setp.eq.f32 %p" << predicate << ", %f" << arguments[0]
               << ", %f" << arguments[1] << ";\n  selp.u32 %r" << encoded
               << ", 1, 0, %p" << predicate << ";\n  ld.param.u64 %rd"
               << outputPointer << ", [p2];\n  add.u64 %rd" << outputPointer
               << ", %rd" << outputPointer << ", %rd0;\n  st.global.u8 [%rd"
               << outputPointer << "], %r" << encoded << ";\n";
    ptx.finish();
    return ptx.text;
  }
  PtxEmitter ptx(map.getInputs().size() + 1);
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, *elements)))
    return failure();
  FailureOr<SmallVector<int64_t>> coordinates =
      emitOutputCoordinates(ptx, output);
  if (failed(coordinates))
    return failure();
  SmallVector<int64_t> arguments;
  for (auto [index, input] : llvm::enumerate(map.getInputs())) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto indexing = dyn_cast<AffineMapAttr>(maps[index]);
    if (!inputType || !indexing)
      return failure();
    FailureOr<int64_t> value =
        emitMapInput(ptx, index, inputType, indexing.getValue(), *coordinates);
    if (failed(value))
      return failure();
    arguments.push_back(*value);
  }
  FailureOr<int64_t> result =
      emitScalarBody(ptx, map.getBody().front(), arguments,
                     /*allowBf16Arithmetic=*/true);
  if (failed(result))
    return failure();
  int64_t outputPointer = ptx.nextU64();
  ptx.stream << "  ld.param.u64 %rd" << outputPointer << ", [p"
             << map.getInputs().size() << "];\n";
  int64_t bytes = output.getElementType().isBF16() ? 2 : 4;
  ptx.stream << "  mad.lo.u64 %rd" << outputPointer << ", %rd0, " << bytes
             << ", %rd" << outputPointer << ";\n";
  if (output.getElementType().isBF16()) {
    ptx.stream << "  cvt.rn.bf16.f32 %h1, %f" << *result
               << ";\n  st.global.u16 [%rd" << outputPointer << "], %h1;\n";
  } else {
    ptx.stream << "  st.global.f32 [%rd" << outputPointer << "], %f" << *result
               << ";\n";
  }
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitGeneratedRelationPtx(RelationOp relation,
                                                ScheduleTaskOp task) {
  if (!relation.getInputs().empty() || relation.getNumResults() != 1 ||
      task.getInputBuffers().size() != 0 ||
      task.getOutputBuffers().size() != 1 || task.getScratchBytes() != 0)
    return failure();
  auto output = dyn_cast<RankedTensorType>(relation.getResult(0).getType());
  if (!output || !output.hasStaticShape())
    return failure();
  auto integerType = dyn_cast<IntegerType>(output.getElementType());
  if (!integerType ||
      (integerType.getWidth() != 1 && integerType.getWidth() != 32))
    return failure();
  FailureOr<int64_t> elements = staticElementCount(output);
  if (failed(elements))
    return failure();

  PtxEmitter ptx(/*arity=*/1);
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, *elements)))
    return failure();
  int64_t value = ptx.nextU32();
  if (relation.getSemantics() == RelationSemantics::Constant) {
    if (relation.getParameters().size() != 1)
      return failure();
    auto constant = dyn_cast<IntegerAttr>(relation.getParameters()[0]);
    if (!constant || constant.getType() != integerType)
      return failure();
    ptx.stream << "  mov.u32 %r" << value << ", "
               << constant.getValue().getZExtValue() << ";\n";
  } else if (relation.getSemantics() == RelationSemantics::Iota) {
    if (integerType.getWidth() != 32 || relation.getParameters().size() != 1)
      return failure();
    auto dimension = dyn_cast<IntegerAttr>(relation.getParameters()[0]);
    if (!dimension || dimension.getInt() < 0 ||
        dimension.getInt() >= output.getRank())
      return failure();
    FailureOr<SmallVector<int64_t>> coordinates =
        emitOutputCoordinates(ptx, output);
    if (failed(coordinates))
      return failure();
    ptx.stream << "  cvt.u32.u64 %r" << value << ", %rd"
               << (*coordinates)[dimension.getInt()] << ";\n";
  } else {
    relation.emitOpError(
        "standalone GPU Relation fallback supports only nullary constant and "
        "iota semantics; tensor-indexed Relations require a generic physical "
        "artifact provider");
    return failure();
  }
  int64_t outputPointer = ptx.nextU64();
  const int64_t bytes = integerType.getWidth() == 1 ? 1 : 4;
  ptx.stream << "  ld.param.u64 %rd" << outputPointer
             << ", [p0];\n  mad.lo.u64 %rd" << outputPointer << ", %rd0, "
             << bytes << ", %rd" << outputPointer << ";\n  st.global.u"
             << bytes * 8 << " [%rd" << outputPointer << "], %r" << value
             << ";\n";
  if (ptx.u32 > 128 || ptx.u64 > 128)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitElementwiseIntegerRelationPtx(RelationOp relation,
                                                         ScheduleTaskOp task) {
  if (relation.getInputs().size() != 2 || relation.getNumResults() != 1 ||
      task.getInputBuffers().size() != 2 ||
      task.getOutputBuffers().size() != 1 || task.getScratchBytes() != 0)
    return failure();
  auto lhs = dyn_cast<RankedTensorType>(relation.getInputs()[0].getType());
  auto rhs = dyn_cast<RankedTensorType>(relation.getInputs()[1].getType());
  auto output = dyn_cast<RankedTensorType>(relation.getResult(0).getType());
  if (!lhs || !rhs || !output || !lhs.hasStaticShape() ||
      !rhs.hasStaticShape() || !output.hasStaticShape() ||
      lhs.getShape() != rhs.getShape() || lhs.getShape() != output.getShape() ||
      !lhs.getElementType().isInteger(32) ||
      rhs.getElementType() != lhs.getElementType())
    return failure();
  const bool compare = relation.getSemantics() == RelationSemantics::Compare;
  const bool add = relation.getSemantics() == RelationSemantics::Add;
  if ((!compare && !add) ||
      (compare && (!output.getElementType().isInteger(1) ||
                   relation.getParameters().size() != 2)) ||
      (add && (output.getElementType() != lhs.getElementType() ||
               !relation.getParameters().empty())))
    return failure();
  StringRef instruction;
  StringRef type;
  if (compare) {
    auto comparisonType = dyn_cast<StringAttr>(relation.getParameters()[0]);
    auto direction = dyn_cast<StringAttr>(relation.getParameters()[1]);
    if (!comparisonType || !direction)
      return failure();
    instruction = llvm::StringSwitch<StringRef>(direction.getValue())
                      .Case("EQ", "eq")
                      .Case("NE", "ne")
                      .Case("LT", "lt")
                      .Case("LE", "le")
                      .Case("GT", "gt")
                      .Case("GE", "ge")
                      .Default({});
    type = comparisonType.getValue() == "SIGNED"     ? "s32"
           : comparisonType.getValue() == "UNSIGNED" ? "u32"
                                                     : StringRef{};
  }
  FailureOr<int64_t> elements = staticElementCount(output);
  if ((compare && (instruction.empty() || type.empty())) || failed(elements))
    return failure();

  PtxEmitter ptx(/*arity=*/3);
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, *elements)))
    return failure();
  int64_t lhsPointer = ptx.nextU64();
  int64_t rhsPointer = ptx.nextU64();
  int64_t outputPointer = ptx.nextU64();
  int64_t lhsValue = ptx.nextU32();
  int64_t rhsValue = ptx.nextU32();
  int64_t result = ptx.nextU32();
  ptx.stream << "  ld.param.u64 %rd" << lhsPointer
             << ", [p0];\n  mad.lo.u64 %rd" << lhsPointer << ", %rd0, 4, %rd"
             << lhsPointer << ";\n  ld.global.u32 %r" << lhsValue << ", [%rd"
             << lhsPointer << "];\n  ld.param.u64 %rd" << rhsPointer
             << ", [p1];\n  mad.lo.u64 %rd" << rhsPointer << ", %rd0, 4, %rd"
             << rhsPointer << ";\n  ld.global.u32 %r" << rhsValue << ", [%rd"
             << rhsPointer << "];\n";
  if (compare) {
    int64_t predicate = ptx.nextPred();
    ptx.stream << "  setp." << instruction << "." << type << " %p" << predicate
               << ", %r" << lhsValue << ", %r" << rhsValue << ";\n  selp.u32 %r"
               << result << ", 1, 0, %p" << predicate << ";\n";
  } else {
    ptx.stream << "  add.u32 %r" << result << ", %r" << lhsValue << ", %r"
               << rhsValue << ";\n";
  }
  const int64_t outputBytes = compare ? 1 : 4;
  ptx.stream << "  ld.param.u64 %rd" << outputPointer
             << ", [p2];\n  mad.lo.u64 %rd" << outputPointer << ", %rd0, "
             << outputBytes << ", %rd" << outputPointer << ";\n  st.global.u"
             << outputBytes * 8 << " [%rd" << outputPointer << "], %r" << result
             << ";\n";
  if (ptx.u32 > 128 || ptx.u64 > 128 || ptx.pred > 64)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitIntegerSelectRelationPtx(RelationOp relation,
                                                    ScheduleTaskOp task) {
  if (relation.getSemantics() != RelationSemantics::Select ||
      relation.getInputs().size() != 3 || relation.getNumResults() != 1 ||
      task.getInputBuffers().size() != 3 ||
      task.getOutputBuffers().size() != 1 || task.getScratchBytes() != 0 ||
      !relation.getParameters().empty())
    return failure();
  auto predicate =
      dyn_cast<RankedTensorType>(relation.getInputs()[0].getType());
  auto onTrue = dyn_cast<RankedTensorType>(relation.getInputs()[1].getType());
  auto onFalse = dyn_cast<RankedTensorType>(relation.getInputs()[2].getType());
  auto output = dyn_cast<RankedTensorType>(relation.getResult(0).getType());
  if (!predicate || !onTrue || !onFalse || !output ||
      !predicate.hasStaticShape() || !onTrue.hasStaticShape() ||
      !onFalse.hasStaticShape() || !output.hasStaticShape() ||
      predicate.getShape() != output.getShape() ||
      onTrue.getShape() != output.getShape() ||
      onFalse.getShape() != output.getShape() ||
      !predicate.getElementType().isInteger(1) ||
      !onTrue.getElementType().isInteger(32) ||
      onFalse.getElementType() != onTrue.getElementType() ||
      output.getElementType() != onTrue.getElementType())
    return failure();
  FailureOr<int64_t> elements = staticElementCount(output);
  if (failed(elements))
    return failure();

  PtxEmitter ptx(/*arity=*/4);
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, *elements)))
    return failure();
  int64_t predicatePointer = ptx.nextU64();
  int64_t truePointer = ptx.nextU64();
  int64_t falsePointer = ptx.nextU64();
  int64_t outputPointer = ptx.nextU64();
  int64_t predicateValue = ptx.nextU32();
  int64_t trueValue = ptx.nextU32();
  int64_t falseValue = ptx.nextU32();
  int64_t nonzero = ptx.nextPred();
  int64_t result = ptx.nextU32();
  ptx.stream << "  ld.param.u64 %rd" << predicatePointer
             << ", [p0];\n  add.u64 %rd" << predicatePointer << ", %rd"
             << predicatePointer << ", %rd0;\n  ld.global.u8 %r"
             << predicateValue << ", [%rd" << predicatePointer
             << "];\n  ld.param.u64 %rd" << truePointer
             << ", [p1];\n  mad.lo.u64 %rd" << truePointer << ", %rd0, 4, %rd"
             << truePointer << ";\n  ld.global.u32 %r" << trueValue << ", [%rd"
             << truePointer << "];\n  ld.param.u64 %rd" << falsePointer
             << ", [p2];\n  mad.lo.u64 %rd" << falsePointer << ", %rd0, 4, %rd"
             << falsePointer << ";\n  ld.global.u32 %r" << falseValue
             << ", [%rd" << falsePointer << "];\n  setp.ne.u32 %p" << nonzero
             << ", %r" << predicateValue << ", 0;\n  selp.u32 %r" << result
             << ", %r" << trueValue << ", %r" << falseValue << ", %p" << nonzero
             << ";\n  ld.param.u64 %rd" << outputPointer
             << ", [p3];\n  mad.lo.u64 %rd" << outputPointer << ", %rd0, 4, %rd"
             << outputPointer << ";\n  st.global.u32 [%rd" << outputPointer
             << "], %r" << result << ";\n";
  if (ptx.u32 > 128 || ptx.u64 > 128 || ptx.pred > 64)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<int64_t> emitIntegerRelationValueAt(
    PtxEmitter &ptx, Value value, ArrayRef<int64_t> coordinates,
    const llvm::SmallDenseSet<Operation *> &composition,
    const DenseMap<Value, unsigned> &boundaryParameters) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() ||
      (!type.getElementType().isInteger(1) &&
       !type.getElementType().isInteger(32)) ||
      coordinates.size() != static_cast<size_t>(type.getRank()))
    return failure();
  Operation *definition = value.getDefiningOp();
  if (!definition || !composition.contains(definition)) {
    auto parameter = boundaryParameters.find(value);
    if (parameter == boundaryParameters.end())
      return failure();
    FailureOr<int64_t> linear =
        emitRowMajorLinearIndex(ptx, coordinates, type.getShape());
    if (failed(linear))
      return failure();
    int64_t pointer = ptx.nextU64();
    int64_t result = ptx.nextU32();
    const int64_t bytes = type.getElementType().isInteger(1) ? 1 : 4;
    ptx.stream << "  ld.param.u64 %rd" << pointer << ", [p" << parameter->second
               << "];\n  mad.lo.u64 %rd" << pointer << ", %rd" << *linear
               << ", " << bytes << ", %rd" << pointer << ";\n  ld.global.u"
               << bytes * 8 << " %r" << result << ", [%rd" << pointer << "];\n";
    return result;
  }
  RelationOp relation = cast<RelationOp>(definition);
  if (relation.getNumResults() != 1)
    return failure();
  if (relation.getSemantics() == RelationSemantics::BroadcastInDim) {
    if (relation.getInputs().size() != 1 ||
        relation.getParameters().size() !=
            cast<RankedTensorType>(relation.getInputs()[0].getType()).getRank())
      return failure();
    SmallVector<int64_t> inputCoordinates;
    for (Attribute parameter : relation.getParameters()) {
      auto dimension = dyn_cast<IntegerAttr>(parameter);
      if (!dimension || dimension.getInt() < 0 ||
          dimension.getInt() >= coordinates.size())
        return failure();
      inputCoordinates.push_back(coordinates[dimension.getInt()]);
    }
    return emitIntegerRelationValueAt(ptx, relation.getInputs()[0],
                                      inputCoordinates, composition,
                                      boundaryParameters);
  }
  if (relation.getSemantics() == RelationSemantics::Add ||
      relation.getSemantics() == RelationSemantics::Compare) {
    if (relation.getInputs().size() != 2)
      return failure();
    FailureOr<int64_t> lhs =
        emitIntegerRelationValueAt(ptx, relation.getInputs()[0], coordinates,
                                   composition, boundaryParameters);
    FailureOr<int64_t> rhs =
        emitIntegerRelationValueAt(ptx, relation.getInputs()[1], coordinates,
                                   composition, boundaryParameters);
    if (failed(lhs) || failed(rhs))
      return failure();
    int64_t result = ptx.nextU32();
    if (relation.getSemantics() == RelationSemantics::Add) {
      if (!relation.getParameters().empty())
        return failure();
      ptx.stream << "  add.u32 %r" << result << ", %r" << *lhs << ", %r" << *rhs
                 << ";\n";
      return result;
    }
    if (relation.getParameters().size() != 2)
      return failure();
    auto comparisonType = dyn_cast<StringAttr>(relation.getParameters()[0]);
    auto direction = dyn_cast<StringAttr>(relation.getParameters()[1]);
    if (!comparisonType || !direction)
      return failure();
    StringRef instruction = llvm::StringSwitch<StringRef>(direction.getValue())
                                .Case("EQ", "eq")
                                .Case("NE", "ne")
                                .Case("LT", "lt")
                                .Case("LE", "le")
                                .Case("GT", "gt")
                                .Case("GE", "ge")
                                .Default({});
    StringRef integerType = comparisonType.getValue() == "SIGNED"     ? "s32"
                            : comparisonType.getValue() == "UNSIGNED" ? "u32"
                                                                      : "";
    if (instruction.empty() || integerType.empty())
      return failure();
    int64_t predicate = ptx.nextPred();
    ptx.stream << "  setp." << instruction << "." << integerType << " %p"
               << predicate << ", %r" << *lhs << ", %r" << *rhs
               << ";\n  selp.u32 %r" << result << ", 1, 0, %p" << predicate
               << ";\n";
    return result;
  }
  if (relation.getSemantics() == RelationSemantics::Select) {
    if (relation.getInputs().size() != 3 || !relation.getParameters().empty())
      return failure();
    FailureOr<int64_t> predicate =
        emitIntegerRelationValueAt(ptx, relation.getInputs()[0], coordinates,
                                   composition, boundaryParameters);
    FailureOr<int64_t> onTrue =
        emitIntegerRelationValueAt(ptx, relation.getInputs()[1], coordinates,
                                   composition, boundaryParameters);
    FailureOr<int64_t> onFalse =
        emitIntegerRelationValueAt(ptx, relation.getInputs()[2], coordinates,
                                   composition, boundaryParameters);
    if (failed(predicate) || failed(onTrue) || failed(onFalse))
      return failure();
    int64_t nonzero = ptx.nextPred();
    int64_t result = ptx.nextU32();
    ptx.stream << "  setp.ne.u32 %p" << nonzero << ", %r" << *predicate
               << ", 0;\n  selp.u32 %r" << result << ", %r" << *onTrue << ", %r"
               << *onFalse << ", %p" << nonzero << ";\n";
    return result;
  }
  return failure();
}

FailureOr<std::string>
emitIntegerRelationCompositionPtx(ArrayRef<RelationOp> relations,
                                  ScheduleTaskOp task) {
  if (relations.size() < 2 || task.getOutputBuffers().size() != 1 ||
      task.getScratchBytes() != 0)
    return failure();
  RelationOp terminal = relations.back();
  if (terminal.getNumResults() != 1)
    return failure();
  auto output = dyn_cast<RankedTensorType>(terminal.getResult(0).getType());
  if (!output || !output.hasStaticShape() ||
      (!output.getElementType().isInteger(1) &&
       !output.getElementType().isInteger(32)) ||
      task.getDomainShape() != output.getShape())
    return failure();
  llvm::SmallDenseSet<Operation *> composition;
  for (RelationOp relation : relations)
    composition.insert(relation);
  SmallVector<Value> boundaryValues;
  llvm::SmallDenseSet<Value> seen;
  for (RelationOp relation : relations) {
    for (Value operand : relation.getInputs()) {
      if (composition.contains(operand.getDefiningOp()) ||
          !seen.insert(operand).second)
        continue;
      boundaryValues.push_back(operand);
    }
  }
  if (boundaryValues.size() != task.getInputBuffers().size())
    return failure();
  DenseMap<Value, unsigned> boundaryParameters;
  for (auto [parameter, value] : llvm::enumerate(boundaryValues))
    boundaryParameters[value] = parameter;
  FailureOr<int64_t> elements = staticElementCount(output);
  if (failed(elements))
    return failure();
  PtxEmitter ptx(task.getInputBuffers().size() + 1);
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, *elements)))
    return failure();
  FailureOr<SmallVector<int64_t>> coordinates =
      emitOutputCoordinates(ptx, output);
  if (failed(coordinates))
    return failure();
  FailureOr<int64_t> value =
      emitIntegerRelationValueAt(ptx, terminal.getResult(0), *coordinates,
                                 composition, boundaryParameters);
  if (failed(value))
    return failure();
  const int64_t bytes = output.getElementType().isInteger(1) ? 1 : 4;
  int64_t outputPointer = ptx.nextU64();
  ptx.stream << "  ld.param.u64 %rd" << outputPointer << ", [p"
             << task.getInputBuffers().size() << "];\n  mad.lo.u64 %rd"
             << outputPointer << ", %rd0, " << bytes << ", %rd" << outputPointer
             << ";\n  st.global.u" << bytes * 8 << " [%rd" << outputPointer
             << "], %r" << *value << ";\n";
  if (ptx.u32 > 128 || ptx.u64 > 128 || ptx.pred > 64)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitGatherRelationPtx(RelationOp relation,
                                             ScheduleTaskOp task) {
  if (relation.getSemantics() != RelationSemantics::Gather ||
      relation.getInputs().size() != 2 || relation.getNumResults() != 1 ||
      task.getInputBuffers().size() != 2 ||
      task.getOutputBuffers().size() != 1 || task.getScratchBytes() != 0 ||
      relation.getParameters().size() != 1)
    return failure();
  auto operand = dyn_cast<RankedTensorType>(relation.getInputs()[0].getType());
  auto indices = dyn_cast<RankedTensorType>(relation.getInputs()[1].getType());
  auto output = dyn_cast<RankedTensorType>(relation.getResult(0).getType());
  auto metadata = dyn_cast<DictionaryAttr>(relation.getParameters()[0]);
  if (!operand || !indices || !output || !metadata ||
      !operand.hasStaticShape() || !indices.hasStaticShape() ||
      !output.hasStaticShape() || !indices.getElementType().isInteger(32) ||
      output.getElementType() != operand.getElementType())
    return failure();
  auto offsetDims = metadata.getAs<DenseI64ArrayAttr>("offset_dims");
  auto collapsedDims =
      metadata.getAs<DenseI64ArrayAttr>("collapsed_slice_dims");
  auto operandBatchingDims =
      metadata.getAs<DenseI64ArrayAttr>("operand_batching_dims");
  auto indicesBatchingDims =
      metadata.getAs<DenseI64ArrayAttr>("start_indices_batching_dims");
  auto startIndexMap = metadata.getAs<DenseI64ArrayAttr>("start_index_map");
  auto indexVectorDim = metadata.getAs<IntegerAttr>("index_vector_dim");
  auto sliceSizes = metadata.getAs<DenseI64ArrayAttr>("slice_sizes");
  if (!offsetDims || !collapsedDims || !operandBatchingDims ||
      !indicesBatchingDims || !startIndexMap || !indexVectorDim ||
      !sliceSizes || !operandBatchingDims.empty() ||
      !indicesBatchingDims.empty() ||
      sliceSizes.size() != static_cast<size_t>(operand.getRank()) ||
      indexVectorDim.getInt() < 0 ||
      indexVectorDim.getInt() >= indices.getRank() ||
      indices.getDimSize(indexVectorDim.getInt()) !=
          static_cast<int64_t>(startIndexMap.size()))
    return failure();
  SmallVector<int64_t> batchOutputDims;
  for (int64_t dimension = 0; dimension < output.getRank(); ++dimension) {
    if (!llvm::is_contained(offsetDims.asArrayRef(), dimension))
      batchOutputDims.push_back(dimension);
  }
  if (batchOutputDims.size() + 1 != static_cast<size_t>(indices.getRank()) ||
      offsetDims.size() + collapsedDims.size() !=
          static_cast<size_t>(operand.getRank()))
    return failure();
  const int64_t bytes = operand.getElementType().isBF16()        ? 2
                        : operand.getElementType().isF32()       ? 4
                        : operand.getElementType().isInteger(32) ? 4
                                                                 : 0;
  FailureOr<int64_t> elements = staticElementCount(output);
  if (bytes == 0 || failed(elements))
    return failure();

  PtxEmitter ptx(/*arity=*/3);
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, *elements)))
    return failure();
  FailureOr<SmallVector<int64_t>> outputCoordinates =
      emitOutputCoordinates(ptx, output);
  if (failed(outputCoordinates))
    return failure();
  int64_t operandBase = ptx.nextU64();
  int64_t indicesBase = ptx.nextU64();
  ptx.stream << "  ld.param.u64 %rd" << operandBase
             << ", [p0];\n  ld.param.u64 %rd" << indicesBase << ", [p1];\n";
  SmallVector<int64_t> starts(operand.getRank(), -1);
  for (auto [component, operandDimension] :
       llvm::enumerate(startIndexMap.asArrayRef())) {
    if (operandDimension < 0 || operandDimension >= operand.getRank())
      return failure();
    SmallVector<int64_t> indexCoordinates;
    size_t batch = 0;
    for (int64_t dimension = 0; dimension < indices.getRank(); ++dimension) {
      if (dimension == indexVectorDim.getInt()) {
        int64_t coordinate = ptx.nextU64();
        ptx.stream << "  mov.u64 %rd" << coordinate << ", " << component
                   << ";\n";
        indexCoordinates.push_back(coordinate);
      } else {
        indexCoordinates.push_back(
            (*outputCoordinates)[batchOutputDims[batch++]]);
      }
    }
    FailureOr<int64_t> indexLinear =
        emitRowMajorLinearIndex(ptx, indexCoordinates, indices.getShape());
    if (failed(indexLinear))
      return failure();
    int64_t pointer = ptx.nextU64();
    int64_t raw = ptx.nextU32();
    ptx.stream << "  mad.lo.u64 %rd" << pointer << ", %rd" << *indexLinear
               << ", 4, %rd" << indicesBase << ";\n  ld.global.u32 %r" << raw
               << ", [%rd" << pointer << "];\n";
    int64_t nonnegative = ptx.nextU32();
    int64_t bounded = ptx.nextU32();
    int64_t start = ptx.nextU64();
    const int64_t maximum =
        operand.getDimSize(operandDimension) - sliceSizes[operandDimension];
    if (maximum < 0)
      return failure();
    ptx.stream << "  max.s32 %r" << nonnegative << ", %r" << raw
               << ", 0;\n  min.s32 %r" << bounded << ", %r" << nonnegative
               << ", " << maximum << ";\n  cvt.s64.s32 %rd" << start << ", %r"
               << bounded << ";\n";
    starts[operandDimension] = start;
  }
  SmallVector<int64_t> operandCoordinates;
  size_t offset = 0;
  for (int64_t dimension = 0; dimension < operand.getRank(); ++dimension) {
    int64_t coordinate = starts[dimension];
    if (coordinate < 0) {
      coordinate = ptx.nextU64();
      ptx.stream << "  mov.u64 %rd" << coordinate << ", 0;\n";
    }
    if (!llvm::is_contained(collapsedDims.asArrayRef(), dimension)) {
      if (offset >= offsetDims.size() || offsetDims[offset] < 0 ||
          offsetDims[offset] >= output.getRank())
        return failure();
      int64_t withOffset = ptx.nextU64();
      ptx.stream << "  add.u64 %rd" << withOffset << ", %rd" << coordinate
                 << ", %rd" << (*outputCoordinates)[offsetDims[offset++]]
                 << ";\n";
      coordinate = withOffset;
    }
    operandCoordinates.push_back(coordinate);
  }
  FailureOr<int64_t> operandLinear =
      emitRowMajorLinearIndex(ptx, operandCoordinates, operand.getShape());
  if (failed(operandLinear))
    return failure();
  int64_t inputPointer = ptx.nextU64();
  int64_t outputPointer = ptx.nextU64();
  int64_t value = ptx.nextU32();
  ptx.stream << "  mad.lo.u64 %rd" << inputPointer << ", %rd" << *operandLinear
             << ", " << bytes << ", %rd" << operandBase << ";\n  ld.global.u"
             << bytes * 8 << " %r" << value << ", [%rd" << inputPointer
             << "];\n  ld.param.u64 %rd" << outputPointer
             << ", [p2];\n  mad.lo.u64 %rd" << outputPointer << ", %rd0, "
             << bytes << ", %rd" << outputPointer << ";\n  st.global.u"
             << bytes * 8 << " [%rd" << outputPointer << "], %r" << value
             << ";\n";
  if (ptx.u32 > 128 || ptx.u64 > 128 || ptx.pred > 64)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitRelationGatherCompositionPtx(
    ArrayRef<RelationOp> relations, MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> &buffers, ScheduleTaskOp task) {
  if (relations.size() < 2 ||
      (task.getOutputBuffers().size() != 1 &&
       task.getOutputBuffers().size() != 2) ||
      task.getScratchBytes() != 0)
    return failure();
  RelationOp gather = relations.back();
  if (gather.getSemantics() != RelationSemantics::Gather ||
      gather.getInputs().size() != 2 || gather.getNumResults() != 1 ||
      gather.getParameters().size() != 1)
    return failure();
  auto operand = dyn_cast<RankedTensorType>(gather.getInputs()[0].getType());
  auto indices = dyn_cast<RankedTensorType>(gather.getInputs()[1].getType());
  auto output = dyn_cast<RankedTensorType>(gather.getResult(0).getType());
  auto metadata = dyn_cast<DictionaryAttr>(gather.getParameters()[0]);
  if (!operand || !indices || !output || !metadata ||
      !operand.hasStaticShape() || !indices.hasStaticShape() ||
      !output.hasStaticShape() || !indices.getElementType().isInteger(32) ||
      output.getElementType() != operand.getElementType() ||
      task.getDomainShape() != output.getShape())
    return failure();
  auto outputBuffers = materializationTask.getOutputBuffers();
  const bool writeIndices = outputBuffers.size() == 2;
  const int64_t indexOutput = writeIndices ? outputBuffers[0] : -1;
  const int64_t valueOutput = outputBuffers[writeIndices ? 1 : 0];
  if ((writeIndices &&
       (indexOutput < 0 || indexOutput >= buffers.size() ||
        buffers[indexOutput].getTensorType() != indices)) ||
      valueOutput < 0 || valueOutput >= buffers.size() ||
      buffers[valueOutput].getTensorType() != output)
    return failure();
  auto offsetDims = metadata.getAs<DenseI64ArrayAttr>("offset_dims");
  auto collapsedDims =
      metadata.getAs<DenseI64ArrayAttr>("collapsed_slice_dims");
  auto operandBatchingDims =
      metadata.getAs<DenseI64ArrayAttr>("operand_batching_dims");
  auto indicesBatchingDims =
      metadata.getAs<DenseI64ArrayAttr>("start_indices_batching_dims");
  auto startIndexMap = metadata.getAs<DenseI64ArrayAttr>("start_index_map");
  auto indexVectorDim = metadata.getAs<IntegerAttr>("index_vector_dim");
  auto sliceSizes = metadata.getAs<DenseI64ArrayAttr>("slice_sizes");
  if (!offsetDims || !collapsedDims || !operandBatchingDims ||
      !indicesBatchingDims || !startIndexMap || !indexVectorDim ||
      !sliceSizes || !operandBatchingDims.empty() ||
      !indicesBatchingDims.empty() ||
      sliceSizes.size() != static_cast<size_t>(operand.getRank()) ||
      indexVectorDim.getInt() < 0 ||
      indexVectorDim.getInt() >= indices.getRank() ||
      indices.getDimSize(indexVectorDim.getInt()) !=
          static_cast<int64_t>(startIndexMap.size()))
    return failure();
  SmallVector<int64_t> batchOutputDims;
  for (int64_t dimension = 0; dimension < output.getRank(); ++dimension)
    if (!llvm::is_contained(offsetDims.asArrayRef(), dimension))
      batchOutputDims.push_back(dimension);
  if (batchOutputDims.size() + 1 != static_cast<size_t>(indices.getRank()) ||
      offsetDims.size() + collapsedDims.size() !=
          static_cast<size_t>(operand.getRank()))
    return failure();
  const int64_t bytes = operand.getElementType().isBF16()        ? 2
                        : operand.getElementType().isF32()       ? 4
                        : operand.getElementType().isInteger(32) ? 4
                                                                 : 0;
  FailureOr<int64_t> elements = staticElementCount(output);
  if (bytes == 0 || failed(elements))
    return failure();

  llvm::SmallDenseSet<Operation *> composition;
  for (RelationOp relation : relations)
    composition.insert(relation);
  SmallVector<Value> boundaryValues;
  llvm::SmallDenseSet<Value> seen;
  for (RelationOp relation : relations) {
    for (Value operandValue : relation.getInputs()) {
      if (composition.contains(operandValue.getDefiningOp()) ||
          !seen.insert(operandValue).second)
        continue;
      boundaryValues.push_back(operandValue);
    }
  }
  if (boundaryValues.size() != task.getInputBuffers().size())
    return failure();
  DenseMap<Value, unsigned> boundaryParameters;
  for (auto [parameter, value] : llvm::enumerate(boundaryValues))
    boundaryParameters[value] = parameter;
  auto operandParameter = boundaryParameters.find(gather.getInputs()[0]);
  if (operandParameter == boundaryParameters.end())
    return failure();

  PtxEmitter ptx(task.getInputBuffers().size() + outputBuffers.size());
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, *elements)))
    return failure();
  FailureOr<SmallVector<int64_t>> outputCoordinates =
      emitOutputCoordinates(ptx, output);
  if (failed(outputCoordinates))
    return failure();
  int64_t operandBase = ptx.nextU64();
  ptx.stream << "  ld.param.u64 %rd" << operandBase << ", [p"
             << operandParameter->second << "];\n";
  SmallVector<int64_t> starts(operand.getRank(), -1);
  std::optional<int64_t> indicesOutputBase;
  if (writeIndices) {
    indicesOutputBase = ptx.nextU64();
    ptx.stream << "  ld.param.u64 %rd" << *indicesOutputBase << ", [p"
               << task.getInputBuffers().size() << "];\n";
  }
  for (auto [component, operandDimension] :
       llvm::enumerate(startIndexMap.asArrayRef())) {
    if (operandDimension < 0 || operandDimension >= operand.getRank())
      return failure();
    SmallVector<int64_t> indexCoordinates;
    size_t batch = 0;
    for (int64_t dimension = 0; dimension < indices.getRank(); ++dimension) {
      if (dimension == indexVectorDim.getInt()) {
        int64_t coordinate = ptx.nextU64();
        ptx.stream << "  mov.u64 %rd" << coordinate << ", " << component
                   << ";\n";
        indexCoordinates.push_back(coordinate);
      } else {
        indexCoordinates.push_back(
            (*outputCoordinates)[batchOutputDims[batch++]]);
      }
    }
    FailureOr<int64_t> raw =
        emitIntegerRelationValueAt(ptx, gather.getInputs()[1], indexCoordinates,
                                   composition, boundaryParameters);
    if (failed(raw))
      return failure();
    if (writeIndices) {
      FailureOr<int64_t> indexLinear =
          emitRowMajorLinearIndex(ptx, indexCoordinates, indices.getShape());
      if (failed(indexLinear))
        return failure();
      int64_t indexPointer = ptx.nextU64();
      ptx.stream << "  mad.lo.u64 %rd" << indexPointer << ", %rd"
                 << *indexLinear << ", 4, %rd" << *indicesOutputBase << ";\n";
      std::optional<int64_t> owner;
      for (int64_t offsetDimension : offsetDims.asArrayRef()) {
        if (offsetDimension < 0 || offsetDimension >= output.getRank())
          return failure();
        int64_t zero = ptx.nextPred();
        ptx.stream << "  setp.eq.u64 %p" << zero << ", %rd"
                   << (*outputCoordinates)[offsetDimension] << ", 0;\n";
        if (!owner) {
          owner = zero;
        } else {
          int64_t both = ptx.nextPred();
          ptx.stream << "  and.pred %p" << both << ", %p" << *owner << ", %p"
                     << zero << ";\n";
          owner = both;
        }
      }
      if (owner) {
        ptx.stream << "  @%p" << *owner << " st.global.u32 [%rd"
                   << indexPointer << "], %r" << *raw << ";\n";
      } else {
        ptx.stream << "  st.global.u32 [%rd" << indexPointer << "], %r"
                   << *raw << ";\n";
      }
    }
    int64_t nonnegative = ptx.nextU32();
    int64_t bounded = ptx.nextU32();
    int64_t start = ptx.nextU64();
    const int64_t maximum =
        operand.getDimSize(operandDimension) - sliceSizes[operandDimension];
    if (maximum < 0)
      return failure();
    ptx.stream << "  max.s32 %r" << nonnegative << ", %r" << *raw
               << ", 0;\n  min.s32 %r" << bounded << ", %r" << nonnegative
               << ", " << maximum << ";\n  cvt.s64.s32 %rd" << start << ", %r"
               << bounded << ";\n";
    starts[operandDimension] = start;
  }
  SmallVector<int64_t> operandCoordinates;
  size_t offset = 0;
  for (int64_t dimension = 0; dimension < operand.getRank(); ++dimension) {
    int64_t coordinate = starts[dimension];
    if (coordinate < 0) {
      coordinate = ptx.nextU64();
      ptx.stream << "  mov.u64 %rd" << coordinate << ", 0;\n";
    }
    if (!llvm::is_contained(collapsedDims.asArrayRef(), dimension)) {
      if (offset >= offsetDims.size() || offsetDims[offset] < 0 ||
          offsetDims[offset] >= output.getRank())
        return failure();
      int64_t withOffset = ptx.nextU64();
      ptx.stream << "  add.u64 %rd" << withOffset << ", %rd" << coordinate
                 << ", %rd" << (*outputCoordinates)[offsetDims[offset++]]
                 << ";\n";
      coordinate = withOffset;
    }
    operandCoordinates.push_back(coordinate);
  }
  FailureOr<int64_t> operandLinear =
      emitRowMajorLinearIndex(ptx, operandCoordinates, operand.getShape());
  if (failed(operandLinear))
    return failure();
  int64_t inputPointer = ptx.nextU64();
  int64_t outputPointer = ptx.nextU64();
  int64_t value = ptx.nextU32();
  ptx.stream << "  mad.lo.u64 %rd" << inputPointer << ", %rd" << *operandLinear
             << ", " << bytes << ", %rd" << operandBase << ";\n  ld.global.u"
             << bytes * 8 << " %r" << value << ", [%rd" << inputPointer
             << "];\n  ld.param.u64 %rd" << outputPointer << ", [p"
             << task.getInputBuffers().size() + (writeIndices ? 1 : 0)
             << "];\n  mad.lo.u64 %rd"
             << outputPointer << ", %rd0, " << bytes << ", %rd" << outputPointer
             << ";\n  st.global.u" << bytes * 8 << " [%rd" << outputPointer
             << "], %r" << value << ";\n";
  if (ptx.u32 > 128 || ptx.u64 > 128 || ptx.pred > 64)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitSourceRelationPtx(RelationOp relation,
                                             ScheduleTaskOp task) {
  if (relation.getInputs().size() != 1 || relation.getNumResults() != 1 ||
      task.getInputBuffers().size() != 1 ||
      task.getOutputBuffers().size() != 1 || task.getScratchBytes() != 0)
    return failure();
  auto input = dyn_cast<RankedTensorType>(relation.getInputs()[0].getType());
  auto output = dyn_cast<RankedTensorType>(relation.getResult(0).getType());
  auto integerType =
      output ? dyn_cast<IntegerType>(output.getElementType()) : IntegerType{};
  const bool predicateToFloat =
      input && output &&
      relation.getSemantics() == RelationSemantics::Convert &&
      input.getElementType().isInteger(1) && output.getElementType().isF32() &&
      input.getShape() == output.getShape();
  const bool copyInteger =
      input && output && input.getElementType() == output.getElementType() &&
      integerType &&
      (integerType.getWidth() == 1 || integerType.getWidth() == 32);
  if (!input || !output || !input.hasStaticShape() ||
      !output.hasStaticShape() || (!copyInteger && !predicateToFloat))
    return failure();
  FailureOr<int64_t> outputElements = staticElementCount(output);
  FailureOr<int64_t> inputElements = staticElementCount(input);
  if (failed(outputElements) || failed(inputElements))
    return failure();
  PtxEmitter ptx(/*arity=*/2);
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, *outputElements)))
    return failure();
  int64_t inputLinear = ptx.nextU64();
  if (relation.getSemantics() == RelationSemantics::Reshape) {
    if (!copyInteger || *inputElements != *outputElements ||
        !relation.getParameters().empty())
      return failure();
    ptx.stream << "  mov.u64 %rd" << inputLinear << ", %rd0;\n";
  } else if (relation.getSemantics() == RelationSemantics::BroadcastInDim) {
    if (!copyInteger ||
        relation.getParameters().size() != static_cast<size_t>(input.getRank()))
      return failure();
    FailureOr<SmallVector<int64_t>> outputCoordinates =
        emitOutputCoordinates(ptx, output);
    if (failed(outputCoordinates))
      return failure();
    SmallVector<int64_t> inputCoordinates;
    for (Attribute parameter : relation.getParameters()) {
      auto dimension = dyn_cast<IntegerAttr>(parameter);
      if (!dimension || dimension.getInt() < 0 ||
          dimension.getInt() >= output.getRank())
        return failure();
      inputCoordinates.push_back((*outputCoordinates)[dimension.getInt()]);
    }
    FailureOr<int64_t> mapped =
        emitRowMajorLinearIndex(ptx, inputCoordinates, input.getShape());
    if (failed(mapped))
      return failure();
    ptx.stream << "  mov.u64 %rd" << inputLinear << ", %rd" << *mapped << ";\n";
  } else if (predicateToFloat) {
    ptx.stream << "  mov.u64 %rd" << inputLinear << ", %rd0;\n";
  } else {
    relation.emitOpError(
        "standalone GPU source Relation fallback currently supports only "
        "copy-free integer reshape/broadcast and i1-to-f32 conversion; "
        "tensor-indexed Gather and other index "
        "relations require a generic physical artifact provider");
    return failure();
  }
  const int64_t inputBytes =
      predicateToFloat ? 1 : (integerType.getWidth() == 1 ? 1 : 4);
  const int64_t outputBytes = predicateToFloat ? 4 : inputBytes;
  int64_t inputPointer = ptx.nextU64();
  int64_t outputPointer = ptx.nextU64();
  int64_t value = ptx.nextU32();
  ptx.stream << "  ld.param.u64 %rd" << inputPointer
             << ", [p0];\n  mad.lo.u64 %rd" << inputPointer << ", %rd"
             << inputLinear << ", " << inputBytes << ", %rd" << inputPointer
             << ";\n  ld.global.u" << inputBytes * 8 << " %r" << value
             << ", [%rd" << inputPointer << "];\n";
  int64_t converted = -1;
  if (predicateToFloat) {
    converted = ptx.nextF32();
    ptx.stream << "  cvt.rn.f32.u32 %f" << converted << ", %r" << value
               << ";\n";
  }
  ptx.stream << "  ld.param.u64 %rd" << outputPointer
             << ", [p1];\n  mad.lo.u64 %rd" << outputPointer << ", %rd0, "
             << outputBytes << ", %rd" << outputPointer << ";\n";
  if (predicateToFloat)
    ptx.stream << "  st.global.f32 [%rd" << outputPointer << "], %f"
               << converted << ";\n";
  else
    ptx.stream << "  st.global.u" << outputBytes * 8 << " [%rd" << outputPointer
               << "], %r" << value << ";\n";
  if (ptx.u32 > 128 || ptx.u64 > 128)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitSelectionPtx(SelectionOp selection,
                                        ScheduleTaskOp task) {
  auto input = dyn_cast<RankedTensorType>(selection.getInput().getType());
  auto values =
      dyn_cast<RankedTensorType>(selection.getSelectedValues().getType());
  auto indices =
      dyn_cast<RankedTensorType>(selection.getSelectedIndices().getType());
  const int64_t dimension = selection.getDimension();
  if (!input || !values || !indices || !input.hasStaticShape() ||
      !values.hasStaticShape() || !indices.hasStaticShape() ||
      !input.getElementType().isBF16() ||
      values.getElementType() != input.getElementType() ||
      !indices.getElementType().isInteger(32) || dimension < 0 ||
      dimension >= input.getRank() || selection.getCount() <= 0 ||
      selection.getCount() > input.getDimSize(dimension) ||
      input.getDimSize(dimension) > 4 || !selection.getStable() ||
      selection.getComparison() != SelectionComparison::TotalOrder ||
      selection.getTieBreak() != SelectionTieBreak::SourceIndexAscending ||
      task.getInputBuffers().size() != 1 ||
      task.getOutputBuffers().size() != 2 || task.getGridShape().size() != 1 ||
      task.getGridShape()[0] !=
          input.getNumElements() / input.getDimSize(dimension) ||
      task.getWorkgroupThreads() < selection.getCount()) {
    selection.emitOpError(
        "standalone GPU Selection requires static BF16 input, BF16/I32 "
        "outputs, total-order stable source-index-ascending semantics, and a "
        "selection extent no larger than 4");
    return failure();
  }
  FailureOr<int64_t> elements = staticElementCount(values);
  if (failed(elements))
    return failure();
  PtxEmitter ptx(/*arity=*/3);
  ptx.header(task.getSemanticFingerprint());
  ptx.stream << "  mov.u32 %r0, %tid.x;\n  mov.u32 %r1, %ctaid.x;\n"
                "  setp.ge.u32 %p0, %r0, "
             << selection.getCount()
             << ";\n  @%p0 bra $done;\n  cvt.u64.u32 %rd0, %r0;\n"
                "  mad.wide.u32 %rd0, %r1, "
             << selection.getCount() << ", %rd0;\n";
  FailureOr<SmallVector<int64_t>> outputCoordinates =
      emitOutputCoordinates(ptx, values);
  if (failed(outputCoordinates))
    return failure();
  int64_t inputBase = ptx.nextU64();
  ptx.stream << "  ld.param.u64 %rd" << inputBase << ", [p0];\n";
  int64_t selectedBits = ptx.nextU32();
  int64_t selectedIndex = ptx.nextU32();
  ptx.stream << "  mov.u32 %r" << selectedBits << ", 0;\n  mov.u32 %r"
             << selectedIndex << ", 0;\n";

  auto emitKey = [&](int64_t bits) {
    int64_t sign = ptx.nextU32();
    int64_t positiveKey = ptx.nextU32();
    int64_t negativeKey = ptx.nextU32();
    int64_t key = ptx.nextU32();
    int64_t negative = ptx.nextPred();
    ptx.stream << "  shr.u32 %r" << sign << ", %r" << bits
               << ", 15;\n  setp.ne.u32 %p" << negative << ", %r" << sign
               << ", 0;\n  xor.b32 %r" << positiveKey << ", %r" << bits
               << ", 32768;\n  xor.b32 %r" << negativeKey << ", %r" << bits
               << ", 65535;\n  selp.u32 %r" << key << ", %r" << negativeKey
               << ", %r" << positiveKey << ", %p" << negative << ";\n";
    return key;
  };
  const int64_t selectionExtent = input.getDimSize(dimension);
  for (int64_t candidate = 0; candidate < selectionExtent; ++candidate) {
    SmallVector<int64_t> candidateCoordinates(*outputCoordinates);
    int64_t candidateCoordinate = ptx.nextU64();
    ptx.stream << "  mov.u64 %rd" << candidateCoordinate << ", " << candidate
               << ";\n";
    candidateCoordinates[dimension] = candidateCoordinate;
    FailureOr<int64_t> candidateLinear =
        emitRowMajorLinearIndex(ptx, candidateCoordinates, input.getShape());
    if (failed(candidateLinear))
      return failure();
    int64_t candidatePointer = ptx.nextU64();
    int64_t candidateBits = ptx.nextU32();
    ptx.stream << "  mad.lo.u64 %rd" << candidatePointer << ", %rd"
               << *candidateLinear << ", 2, %rd" << inputBase
               << ";\n  ld.global.u16 %r" << candidateBits << ", [%rd"
               << candidatePointer << "];\n";
    int64_t candidateKey = emitKey(candidateBits);
    int64_t rank = ptx.nextU32();
    ptx.stream << "  mov.u32 %r" << rank << ", 0;\n";
    for (int64_t competitor = 0; competitor < selectionExtent; ++competitor) {
      SmallVector<int64_t> competitorCoordinates(candidateCoordinates);
      int64_t competitorCoordinate = ptx.nextU64();
      ptx.stream << "  mov.u64 %rd" << competitorCoordinate << ", "
                 << competitor << ";\n";
      competitorCoordinates[dimension] = competitorCoordinate;
      FailureOr<int64_t> competitorLinear =
          emitRowMajorLinearIndex(ptx, competitorCoordinates, input.getShape());
      if (failed(competitorLinear))
        return failure();
      int64_t competitorPointer = ptx.nextU64();
      int64_t competitorBits = ptx.nextU32();
      ptx.stream << "  mad.lo.u64 %rd" << competitorPointer << ", %rd"
                 << *competitorLinear << ", 2, %rd" << inputBase
                 << ";\n  ld.global.u16 %r" << competitorBits << ", [%rd"
                 << competitorPointer << "];\n";
      int64_t competitorKey = emitKey(competitorBits);
      int64_t better = ptx.nextPred();
      const bool includeEqual = competitor < candidate;
      StringRef comparison;
      if (selection.getOrder() == SelectionOrder::Descending)
        comparison = includeEqual ? "ge" : "gt";
      else
        comparison = includeEqual ? "le" : "lt";
      ptx.stream << "  setp." << comparison << ".u32 %p" << better << ", %r"
                 << competitorKey << ", %r" << candidateKey << ";\n  @%p"
                 << better << " add.u32 %r" << rank << ", %r" << rank
                 << ", 1;\n";
    }
    int64_t desiredRank = ptx.nextU32();
    int64_t chosen = ptx.nextPred();
    ptx.stream << "  cvt.u32.u64 %r" << desiredRank << ", %rd"
               << (*outputCoordinates)[dimension] << ";\n  setp.eq.u32 %p"
               << chosen << ", %r" << rank << ", %r" << desiredRank
               << ";\n  @%p" << chosen << " mov.u32 %r" << selectedBits
               << ", %r" << candidateBits << ";\n  @%p" << chosen
               << " mov.u32 %r" << selectedIndex << ", " << candidate << ";\n";
  }
  int64_t valuesPointer = ptx.nextU64();
  int64_t indicesPointer = ptx.nextU64();
  int64_t rounded = ptx.nextBF16();
  ptx.stream << "  ld.param.u64 %rd" << valuesPointer
             << ", [p1];\n  mad.lo.u64 %rd" << valuesPointer << ", %rd0, 2, %rd"
             << valuesPointer << ";\n  cvt.u16.u32 %h" << rounded << ", %r"
             << selectedBits << ";\n  st.global.u16 [%rd" << valuesPointer
             << "], %h" << rounded << ";\n  ld.param.u64 %rd" << indicesPointer
             << ", [p2];\n  mad.lo.u64 %rd" << indicesPointer
             << ", %rd0, 4, %rd" << indicesPointer << ";\n  st.global.u32 [%rd"
             << indicesPointer << "], %r" << selectedIndex << ";\n";
  if (ptx.u32 > 128 || ptx.u64 > 128 || ptx.pred > 64 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string>
emitFusedMapPtx(SmallVectorImpl<MapOp> &maps,
                MaterializationTaskOp materializationTask,
                SmallVectorImpl<MaterializationBufferOp> &buffers,
                ScheduleTaskOp scheduleTask) {
  if (maps.size() < 2 || materializationTask.getOutputBuffers().empty() ||
      llvm::any_of(maps, [](MapOp map) { return !hasIdentityResultMap(map); })) {
    materializationTask.emitOpError(
        "grouped Map codegen requires identity-result scalar Maps");
    return failure();
  }
  SmallVector<MapOp> outputMaps;
  auto outputSources =
      materializationTask->getAttrOfType<ArrayAttr>("fused_output_sources");
  if (outputSources) {
    if (outputSources.size() != materializationTask.getOutputBuffers().size()) {
      materializationTask.emitOpError(
          "grouped Map output provenance does not match physical outputs");
      return failure();
    }
    llvm::SmallDenseMap<Attribute, MapOp> mapsBySource;
    for (MapOp map : maps)
      mapsBySource.try_emplace(map.getSource(), map);
    for (Attribute source : outputSources) {
      MapOp output = mapsBySource.lookup(source);
      if (!output) {
        materializationTask.emitOpError(
            "grouped Map output source is not in the fused source set");
        return failure();
      }
      outputMaps.push_back(output);
    }
  } else {
    if (materializationTask.getOutputBuffers().size() != 1) {
      materializationTask.emitOpError(
          "multi-output grouped Map requires explicit output provenance");
      return failure();
    }
    outputMaps.push_back(maps.back());
  }
  SmallVector<RankedTensorType> outputs;
  for (auto [outputBuffer, outputMap] :
       llvm::zip_equal(materializationTask.getOutputBuffers(), outputMaps)) {
    if (outputBuffer < 0 || outputBuffer >= buffers.size()) {
      materializationTask.emitOpError(
          "grouped Map output references an invalid physical buffer");
      return failure();
    }
    auto output =
        dyn_cast<RankedTensorType>(buffers[outputBuffer].getTensorType());
    if (!output || output != outputMap.getResult(0).getType() ||
        (!output.getElementType().isBF16() &&
         !output.getElementType().isF32()) ||
        (!outputs.empty() && output.getShape() != outputs.front().getShape())) {
      materializationTask.emitOpError(
          "grouped Map outputs must share one typed logical domain");
      return failure();
    }
    outputs.push_back(output);
  }
  RankedTensorType output = outputs.front();
  auto composedMaps =
      materializationTask->getAttrOfType<ArrayAttr>("composed_input_maps");
  if (!composedMaps ||
      composedMaps.size() != materializationTask.getInputBuffers().size()) {
    materializationTask.emitOpError(
        "grouped Map boundary maps do not match physical inputs");
    return failure();
  }
  FailureOr<int64_t> elements = staticElementCount(output);
  if (failed(elements)) {
    materializationTask.emitOpError(
        "grouped Map output domain has no bounded element count");
    return failure();
  }
  PtxEmitter ptx(materializationTask.getInputBuffers().size() +
                 materializationTask.getOutputBuffers().size());
  ptx.header(scheduleTask.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, *elements))) {
    materializationTask.emitOpError(
        "grouped Map owner domain cannot be linearized by the CUDA emitter");
    return failure();
  }
  FailureOr<SmallVector<int64_t>> coordinates =
      emitOutputCoordinates(ptx, output);
  if (failed(coordinates)) {
    materializationTask.emitOpError(
        "grouped Map owner coordinates cannot be reconstructed");
    return failure();
  }

  llvm::SmallDenseSet<Operation *> group;
  for (MapOp map : maps)
    group.insert(map.getOperation());
  SmallVector<Value> boundaryInputs;
  llvm::SmallDenseSet<Value> seenBoundary;
  for (MapOp map : llvm::reverse(maps)) {
    for (Value operand : map.getInputs()) {
      if (!group.contains(operand.getDefiningOp()) &&
          seenBoundary.insert(operand).second)
        boundaryInputs.push_back(operand);
    }
  }
  if (boundaryInputs.size() != materializationTask.getInputBuffers().size()) {
    materializationTask.emitOpError(
        "grouped Map SSA boundary does not match physical inputs");
    return failure();
  }
  DenseMap<Value, int64_t> values;
  for (auto [parameter, value, bufferOrdinal, mapAttribute] :
       llvm::enumerate(boundaryInputs, materializationTask.getInputBuffers(),
                       composedMaps.getValue())) {
    if (bufferOrdinal < 0 || bufferOrdinal >= buffers.size()) {
      materializationTask.emitOpError(
          "grouped Map input references an invalid physical buffer");
      return failure();
    }
    auto input =
        dyn_cast<RankedTensorType>(buffers[bufferOrdinal].getTensorType());
    auto indexing = dyn_cast<AffineMapAttr>(mapAttribute);
    if (!input || !indexing) {
      materializationTask.emitOpError(
          "grouped Map input lacks typed buffer or affine indexing provenance");
      return failure();
    }
    FailureOr<int64_t> loaded =
        emitMapInput(ptx, parameter, input, indexing.getValue(), *coordinates);
    if (failed(loaded) || input != value.getType()) {
      materializationTask.emitOpError(
          "grouped Map input cannot be reconstructed at its owner element");
      return failure();
    }
    values[value] = *loaded;
  }
  for (MapOp map : maps) {
    SmallVector<int64_t> arguments;
    for (Value operand : map.getInputs()) {
      auto found = values.find(operand);
      if (found == values.end()) {
        map.emitOpError(
            "grouped Map operand is neither a boundary input nor an earlier "
            "fused result");
        return failure();
      }
      arguments.push_back(found->second);
    }
    FailureOr<int64_t> result =
        emitScalarBody(ptx, map.getBody().front(), arguments,
                       /*allowBf16Arithmetic=*/true);
    if (failed(result) || map.getNumResults() != 1) {
      map.emitOpError(
          "scalar body is unsupported in a fused floating Map component");
      return failure();
    }
    values[map.getResult(0)] = *result;
  }
  for (auto [outputOrdinal, outputMap, outputType] :
       llvm::enumerate(outputMaps, outputs)) {
    auto result = values.find(outputMap.getResult(0));
    if (result == values.end()) {
      outputMap.emitOpError(
          "grouped Map output was not materialized by the fused scalar DAG");
      return failure();
    }
    int64_t outputPointer = ptx.nextU64();
    ptx.stream << "  ld.param.u64 %rd" << outputPointer << ", [p"
               << materializationTask.getInputBuffers().size() + outputOrdinal
               << "];\n";
    if (outputType.getElementType().isBF16()) {
      int64_t rounded = ptx.nextBF16();
      ptx.stream << "  mad.lo.u64 %rd" << outputPointer << ", %rd0, 2, %rd"
                 << outputPointer << ";\n  cvt.rn.bf16.f32 %h" << rounded
                 << ", %f" << result->second << ";\n  st.global.u16 [%rd"
                 << outputPointer << "], %h" << rounded << ";\n";
    } else {
      ptx.stream << "  mad.lo.u64 %rd" << outputPointer << ", %rd0, 4, %rd"
                 << outputPointer << ";\n  st.global.f32 [%rd" << outputPointer
                 << "], %f" << result->second << ";\n";
    }
  }
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 64) {
    materializationTask.emitOpError(
        "fused floating Map component exceeds the bounded PTX register "
        "estimate");
    return failure();
  }
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitContractPtx(ContractOp contract,
                                       ScheduleTaskOp task) {
  const bool fast =
      task.getReductionOrder() == ScheduleReductionOrder::LeafOrderFree;
  FailureOr<SmallVector<int64_t>> contractDomain =
      contractDomainShape(contract);
  if (failed(contractDomain) || task.getKind() != ScheduleTaskKind::Contract ||
      task.getDomainShape() != ArrayRef<int64_t>(*contractDomain) ||
      task.getReductionAxis() !=
          static_cast<int64_t>(task.getDomainShape().size()) - 1 ||
      (!fast && task.getReductionOrder() !=
                    ScheduleReductionOrder::SerialAscendingFmaRnF32) ||
      task.getInputBuffers().size() != 2 ||
      task.getOutputBuffers().size() != 1) {
    return failure();
  }
  FailureOr<int64_t> outputCount =
      scheduleElementCount(task.getDomainShape().drop_back());
  if (failed(outputCount) || *outputCount > UINT32_MAX) {
    return failure();
  }
  const int64_t outputs = *outputCount;
  const int64_t reduction = task.getDomainShape().back();
  AffineMap lhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[0]).getValue();
  AffineMap rhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[1]).getValue();
  AffineMap resultMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[2]).getValue();
  auto lhsType = cast<RankedTensorType>(contract.getInputs()[0].getType());
  auto rhsType = cast<RankedTensorType>(contract.getInputs()[1].getType());
  auto resultType = cast<RankedTensorType>(contract.getResult(0).getType());
  const bool scalarDot =
      lhsType.getRank() == 1 && rhsType.getRank() == 1 &&
      resultType.getRank() == 0 && contractDomain->size() == 1 &&
      lhsMap == AffineMap::getMultiDimIdentityMap(1, contract.getContext()) &&
      rhsMap == lhsMap && resultMap.getNumDims() == 1 &&
      resultMap.getNumResults() == 0 &&
      contract.getIteratorKinds().size() == 1 &&
      cast<StringAttr>(contract.getIteratorKinds()[0]).getValue() ==
          "reduction" &&
      contract.getAlgorithm() == "dot_general" &&
      contract.getAccumulatorTypes().size() == 1 &&
      cast<TypeAttr>(contract.getAccumulatorTypes()[0]).getValue().isF32() &&
      contract.getPrecision().size() == 2 &&
      llvm::all_of(contract.getPrecision(), [](Attribute precision) {
        auto value = dyn_cast<StringAttr>(precision);
        return value && value.getValue() == "DEFAULT";
      });
  const bool closedProjected = isClosedProjectedContract(contract);
  const bool closedMultipleReductions =
      isClosedProjectedContract(contract, /*allowMultipleReductions=*/true);
  if (!scalarDot && !closedMultipleReductions)
    return failure();
  if (outputs == 1 && task.getGridShape() == ArrayRef<int64_t>({1}) &&
      task.getWorkgroupThreads() > 0 && task.getWorkgroupThreads() <= 256 &&
      lhsType.getElementType().isF32() && rhsType.getElementType().isF32()) {
    PtxEmitter ptx(/*arity=*/3);
    ptx.header(task.getSemanticFingerprint());
    ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                  "  setp.ne.u32 %p0, %r0, 0;\n"
                  "  @%p0 bra $done;\n";
    int64_t accumulator = ptx.nextF32();
    int64_t reductionCoordinate = ptx.nextU64();
    ptx.stream << "  mov.f32 %f" << accumulator
               << ", 0f00000000;\n  mov.u64 %rd" << reductionCoordinate
               << ", 0;\n$scalar_contract_loop:\n  setp.ge.u64 %p1, %rd"
               << reductionCoordinate << ", " << reduction
               << ";\n  @%p1 bra $scalar_contract_done;\n";
    SmallVector<int64_t> domainCoordinates{reductionCoordinate};
    FailureOr<int64_t> lhs =
        emitMapInput(ptx, 0, lhsType, lhsMap, domainCoordinates);
    FailureOr<int64_t> rhs =
        emitMapInput(ptx, 1, rhsType, rhsMap, domainCoordinates);
    if (failed(lhs) || failed(rhs))
      return failure();
    int64_t outputPointer = ptx.nextU64();
    ptx.stream << "  fma.rn.f32 %f" << accumulator << ", %f" << *lhs << ", %f"
               << *rhs << ", %f" << accumulator << ";\n  add.u64 %rd"
               << reductionCoordinate << ", %rd" << reductionCoordinate
               << ", 1;\n  bra $scalar_contract_loop;\n$scalar_contract_done:\n"
                  "  ld.param.u64 %rd"
               << outputPointer << ", [p2];\n  st.global.f32 [%rd"
               << outputPointer << "], %f" << accumulator << ";\n";
    if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
      return failure();
    ptx.finish();
    return ptx.text;
  }
  if (!closedProjected) {
    const int64_t parallelRank = resultType.getRank();
    FailureOr<int64_t> reductionElements = scheduleElementCount(
        ArrayRef<int64_t>(*contractDomain).drop_front(parallelRank));
    if (failed(reductionElements) || task.getGridShape().size() != 1 ||
        task.getGridShape()[0] * task.getWorkgroupThreads() < outputs ||
        task.getScratchBytes() != 0)
      return failure();
    PtxEmitter ptx(/*arity=*/3);
    ptx.header(task.getSemanticFingerprint());
    if (failed(emitLinearThread(ptx, outputs)))
      return failure();
    FailureOr<SmallVector<int64_t>> outputCoordinates =
        emitOutputCoordinates(ptx, resultType);
    if (failed(outputCoordinates))
      return failure();
    int64_t accumulator = ptx.nextF32();
    int64_t reductionLinear = ptx.nextU64();
    ptx.stream << "  mov.f32 %f" << accumulator
               << ", 0f00000000;\n  mov.u64 %rd" << reductionLinear
               << ", 0;\n$projected_contract_loop:\n  setp.ge.u64 %p1, %rd"
               << reductionLinear << ", " << *reductionElements
               << ";\n  @%p1 bra $projected_contract_done;\n";
    SmallVector<int64_t> domainCoordinates(*outputCoordinates);
    int64_t stride = *reductionElements;
    for (int64_t dimension = parallelRank;
         dimension < static_cast<int64_t>(contractDomain->size());
         ++dimension) {
      stride /= (*contractDomain)[dimension];
      int64_t coordinate = ptx.nextU64();
      ptx.stream << "  div.u64 %rd" << coordinate << ", %rd" << reductionLinear
                 << ", " << stride << ";\n  rem.u64 %rd" << coordinate
                 << ", %rd" << coordinate << ", "
                 << (*contractDomain)[dimension] << ";\n";
      domainCoordinates.push_back(coordinate);
    }
    FailureOr<int64_t> lhs =
        emitMapInput(ptx, 0, lhsType, lhsMap, domainCoordinates);
    FailureOr<int64_t> rhs =
        emitMapInput(ptx, 1, rhsType, rhsMap, domainCoordinates);
    if (failed(lhs) || failed(rhs))
      return failure();
    int64_t outputPointer = ptx.nextU64();
    ptx.stream << "  fma.rn.f32 %f" << accumulator << ", %f" << *lhs << ", %f"
               << *rhs << ", %f" << accumulator << ";\n  add.u64 %rd"
               << reductionLinear << ", %rd" << reductionLinear
               << ", 1;\n  bra $projected_contract_loop;\n"
                  "$projected_contract_done:\n  ld.param.u64 %rd"
               << outputPointer << ", [p2];\n  mad.lo.u64 %rd" << outputPointer
               << ", %rd0, 4, %rd" << outputPointer << ";\n  st.global.f32 [%rd"
               << outputPointer << "], %f" << accumulator << ";\n";
    if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
      return failure();
    ptx.finish();
    return ptx.text;
  }
  if (task.getScratchBytes() != 0)
    return failure();
  if (fast) {
    const int64_t outputsPerBlock = std::min<int64_t>(outputs, 8);
    if (task.getGridShape() !=
            ArrayRef<int64_t>({ceilDivPositive(outputs, outputsPerBlock)}) ||
        task.getTileShape() != ArrayRef<int64_t>({outputsPerBlock}) ||
        task.getWorkgroupThreads() != outputsPerBlock * 32 ||
        task.getSubgroupSize() != 32 ||
        task.getSerialTiles() != ceilDivPositive(reduction, 32)) {
      return failure();
    }

    PtxEmitter ptx(/*arity=*/3);
    ptx.header(task.getSemanticFingerprint());
    ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                  "  shr.u32 %r1, %r0, 5;\n"
                  "  and.b32 %r2, %r0, 31;\n"
                  "  mov.u32 %r3, %ctaid.x;\n"
                  "  mad.lo.u32 %r4, %r3, "
               << outputsPerBlock
               << ", %r1;\n"
                  "  setp.ge.u32 %p0, %r4, "
               << outputs
               << ";\n"
                  "  @%p0 bra $done;\n"
                  "  cvt.u64.u32 %rd0, %r4;\n";
    auto result = cast<RankedTensorType>(contract.getResult(0).getType());
    FailureOr<SmallVector<int64_t>> coordinates =
        emitOutputCoordinates(ptx, result);
    if (failed(coordinates) ||
        coordinates->size() + 1 != task.getDomainShape().size()) {
      return failure();
    }
    int64_t accumulator = ptx.nextF32();
    int64_t reductionCoordinate = ptx.nextU64();
    ptx.stream << "  mov.f32 %f" << accumulator
               << ", 0f00000000;\n"
                  "  cvt.u64.u32 %rd"
               << reductionCoordinate
               << ", %r2;\n"
                  "$contract_fast_loop:\n"
                  "  setp.ge.u64 %p1, %rd"
               << reductionCoordinate << ", " << reduction
               << ";\n"
                  "  @%p1 bra $contract_fast_reduce;\n"
                  "";
    SmallVector<int64_t> domainCoordinates(*coordinates);
    domainCoordinates.push_back(reductionCoordinate);
    FailureOr<int64_t> lhs =
        emitMapInput(ptx, 0, lhsType, lhsMap, domainCoordinates);
    FailureOr<int64_t> rhs =
        emitMapInput(ptx, 1, rhsType, rhsMap, domainCoordinates);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }
    ptx.stream << "  fma.rn.f32 %f" << accumulator << ", %f" << *lhs << ", %f"
               << *rhs << ", %f" << accumulator << ";\n  add.u64 %rd"
               << reductionCoordinate << ", %rd" << reductionCoordinate
               << ", 32;\n"
                  "  bra $contract_fast_loop;\n"
                  "$contract_fast_reduce:\n"
                  "  mov.b32 %r5, %f"
               << accumulator << ";\n";
    int64_t shuffled = ptx.nextF32();
    for (int64_t offset : {16, 8, 4, 2, 1}) {
      ptx.stream << "  shfl.sync.down.b32 %r6, %r5, " << offset
                 << ", 31, 0xffffffff;\n"
                    "  mov.b32 %f"
                 << shuffled << ", %r6;\n  add.rn.f32 %f" << accumulator
                 << ", %f" << accumulator << ", %f" << shuffled
                 << ";\n  mov.b32 %r5, %f" << accumulator << ";\n";
    }
    int64_t outputPointer = ptx.nextU64();
    ptx.stream << "  setp.ne.u32 %p2, %r2, 0;\n"
                  "  @%p2 bra $done;\n"
                  "  ld.param.u64 %rd"
               << outputPointer << ", [p2];\n  mad.lo.u64 %rd" << outputPointer
               << ", %rd0, 4, %rd" << outputPointer << ";\n  st.global.f32 [%rd"
               << outputPointer << "], %f" << accumulator << ";\n";
    if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16) {
      return failure();
    }
    ptx.finish();
    return ptx.text;
  }

  const int64_t expectedThreads =
      scheduleThreads(std::min<int64_t>(outputs, 256));
  if (task.getGridShape() != ArrayRef<int64_t>({ceilDivPositive(
                                 outputs, std::min<int64_t>(outputs, 256))}) ||
      task.getTileShape() !=
          ArrayRef<int64_t>({std::min<int64_t>(outputs, 256)}) ||
      task.getWorkgroupThreads() != expectedThreads ||
      task.getSubgroupSize() != 32 || task.getSerialTiles() != reduction) {
    return failure();
  }

  PtxEmitter ptx(/*arity=*/3);
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, outputs))) {
    return failure();
  }
  auto result = cast<RankedTensorType>(contract.getResult(0).getType());
  FailureOr<SmallVector<int64_t>> coordinates =
      emitOutputCoordinates(ptx, result);
  if (failed(coordinates) ||
      coordinates->size() + 1 != task.getDomainShape().size()) {
    return failure();
  }
  int64_t accumulator = ptx.nextF32();
  int64_t reductionCoordinate = ptx.nextU64();
  ptx.stream << "  mov.f32 %f" << accumulator << ", 0f00000000;\n  mov.u64 %rd"
             << reductionCoordinate
             << ", 0;\n"
                "$contract_loop:\n"
                "  setp.ge.u64 %p1, %rd"
             << reductionCoordinate << ", " << reduction
             << ";\n  @%p1 bra $contract_done;\n"
                "";
  SmallVector<int64_t> domainCoordinates(*coordinates);
  domainCoordinates.push_back(reductionCoordinate);
  FailureOr<int64_t> lhs =
      emitMapInput(ptx, 0, lhsType, lhsMap, domainCoordinates);
  FailureOr<int64_t> rhs =
      emitMapInput(ptx, 1, rhsType, rhsMap, domainCoordinates);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  int64_t outputPointer = ptx.nextU64();
  ptx.stream << "  fma.rn.f32 %f" << accumulator << ", %f" << *lhs << ", %f"
             << *rhs << ", %f" << accumulator << ";\n  add.u64 %rd"
             << reductionCoordinate << ", %rd" << reductionCoordinate
             << ", 1;\n"
                "  bra $contract_loop;\n"
                "$contract_done:\n"
                "  ld.param.u64 %rd"
             << outputPointer << ", [p2];\n  mad.lo.u64 %rd" << outputPointer
             << ", %rd0, 4, %rd" << outputPointer << ";\n  st.global.f32 [%rd"
             << outputPointer << "], %f" << accumulator << ";\n";
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16) {
    return failure();
  }
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string>
emitFoldMapPtx(FoldOp fold, ScheduleTaskOp task, ArrayRef<MapOp> epilogueMaps,
               MaterializationTaskOp materializationTask,
               SmallVectorImpl<MaterializationBufferOp> *buffers) {
  const bool fused = !epilogueMaps.empty();
  if ((!isClosedRankTwoFold(fold) && !isClosedRankNFold(fold)) ||
      fold.getInputs().size() != 1 || fold.getInitializers().size() != 1 ||
      fold.getNumResults() != 1 ||
      task.getDomainShape() !=
          cast<RankedTensorType>(fold.getInputs()[0].getType()).getShape())
    return failure();
  if (!hasExactPositiveZeroInitializer(fold))
    return failure();
  const bool fullPrefix = isClosedFullPrefixRankNFold(fold);
  const int64_t reductionAxis = fold.getReductionDimensions().front();
  FailureOr<int64_t> reductionExtent =
      fullPrefix ? scheduleElementCount(task.getDomainShape().drop_back())
                 : FailureOr<int64_t>(task.getDomainShape()[reductionAxis]);
  if (failed(reductionExtent))
    return failure();
  auto resultType = cast<RankedTensorType>(fold.getResult(0).getType());
  FailureOr<int64_t> outputElements = staticElementCount(resultType);
  if (failed(outputElements) || *outputElements > UINT32_MAX)
    return failure();
  const int64_t outputExtent = *outputElements;
  const int64_t threads = task.getWorkgroupThreads();
  const int64_t serialTiles = task.getSerialTiles();
  auto nextPowerOfTwo = [](int64_t value) -> FailureOr<int64_t> {
    if (value <= 0)
      return failure();
    int64_t power = 1;
    while (power < value) {
      if (power > 128)
        return failure();
      power *= 2;
    }
    return power;
  };
  FailureOr<int64_t> localWidth = nextPowerOfTwo(serialTiles);
  FailureOr<int64_t> sharedWidth =
      nextPowerOfTwo(std::min(*reductionExtent, threads));
  if (failed(localWidth) || failed(sharedWidth) || threads <= 0 ||
      threads > 256 ||
      task.getGridShape() != ArrayRef<int64_t>({outputExtent}) ||
      task.getScratchBytes() != threads * static_cast<int64_t>(sizeof(float)) ||
      *localWidth > 64 || *sharedWidth > threads)
    return failure();
  if (fused &&
      (!materializationTask || !buffers ||
       materializationTask.getInputBuffers().size() < fold->getNumOperands() ||
       materializationTask.getOutputBuffers().size() != 1))
    return failure();
  PtxEmitter ptx(fused ? materializationTask.getInputBuffers().size() + 1 : 3);
  ptx.header(task.getSemanticFingerprint(), /*usesDynamicSharedMemory=*/true);
  ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                "  mov.u32 %r1, %ctaid.x;\n"
                "  setp.ge.u32 %p0, %r1, "
             << outputExtent
             << ";\n  @%p0 bra $done;\n"
                "  ld.param.u64 %rd0, [p0];\n";
  for (int64_t index = 0; index < *localWidth; ++index)
    ptx.stream << "  mov.f32 %f" << index << ", 0f00000000;\n";
  for (int64_t index = 0; index < serialTiles; ++index) {
    ptx.stream << "  mul.lo.u32 %r4, %r0, " << serialTiles
               << ";\n  add.u32 %r4, %r4, " << index
               << ";\n  setp.ge.u32 %p1, %r4, " << *reductionExtent
               << ";\n  @%p1 bra $load_skip" << index << ";\n"
               << "  cvt.u64.u32 %rd1, %r4;\n"
               << "  cvt.u64.u32 %rd2, %r1;\n";
    if (reductionAxis ==
        static_cast<int64_t>(task.getDomainShape().size()) - 1) {
      ptx.stream << "  mad.lo.u64 %rd3, %rd2, " << *reductionExtent
                 << ", %rd1;\n";
    } else {
      ptx.stream << "  mad.lo.u64 %rd3, %rd1, " << outputExtent << ", %rd2;\n";
    }
    ptx.stream << "  mad.lo.u64 %rd3, %rd3, 4, %rd0;\n"
               << "  ld.global.f32 %f" << index << ", [%rd3];\n$load_skip"
               << index << ":\n";
  }
  int64_t width = *localWidth;
  while (width > 1) {
    for (int64_t index = 0; index < width / 2; ++index)
      ptx.stream << "  add.rn.f32 %f" << index << ", %f" << index * 2 << ", %f"
                 << index * 2 + 1 << ";\n";
    width /= 2;
  }
  ptx.stream << "  cvt.u64.u32 %rd3, %r0;\n"
                "  mul.lo.u64 %rd3, %rd3, 4;\n"
                "  mov.u64 %rd4, shuttle_smem;\n"
                "  cvta.to.shared.u64 %rd4, %rd4;\n"
                "  add.u64 %rd3, %rd4, %rd3;\n"
                "  st.shared.f32 [%rd3], %f0;\n"
                "  bar.sync 0;\n";
  for (int64_t step = *sharedWidth / 2; step >= 1; step /= 2) {
    ptx.stream << "  setp.ge.u32 %p1, %r0, " << step << ";\n  @%p1 bra $skip"
               << step << ";\n"
               << "  mul.lo.u32 %r4, %r0, 2;\n"
               << "  cvt.u64.u32 %rd5, %r4;\n"
               << "  mul.lo.u64 %rd5, %rd5, 4;\n"
               << "  add.u64 %rd5, %rd4, %rd5;\n"
               << "  ld.shared.f32 %f32, [%rd5];\n"
               << "  ld.shared.f32 %f33, [%rd5+4];\n"
               << "  add.rn.f32 %f34, %f32, %f33;\n"
               << "  cvt.u64.u32 %rd6, %r0;\n"
               << "  mul.lo.u64 %rd6, %rd6, 4;\n"
               << "  add.u64 %rd6, %rd4, %rd6;\n"
               << "  st.shared.f32 [%rd6], %f34;\n$skip" << step
               << ":\n  bar.sync 0;\n";
  }
  ptx.stream << "  setp.ne.u32 %p2, %r0, 0;\n"
                "  @%p2 bra $done;\n"
                "  ld.shared.f32 %f35, [%rd4];\n"
                "  ld.param.u64 %rd7, [p1];\n"
                "  ld.global.f32 %f36, [%rd7];\n"
                "  add.rn.f32 %f37, %f35, %f36;\n";
  if (!fused) {
    ptx.stream << "  ld.param.u64 %rd8, [p2];\n"
                  "  cvt.u64.u32 %rd9, %r1;\n"
                  "  mul.lo.u64 %rd9, %rd9, 4;\n"
                  "  add.u64 %rd8, %rd8, %rd9;\n"
                  "  st.global.f32 [%rd8], %f37;\n";
    ptx.finish();
    return ptx.text;
  }

  unsigned foldConsumers = 0;
  for (MapOp map : epilogueMaps) {
    for (auto [operandNumber, operand] : llvm::enumerate(map.getInputs())) {
      if (operand != fold.getResult(0))
        continue;
      if (!isEligibleFoldMapEdge(fold, map, operandNumber))
        return failure();
      ++foldConsumers;
    }
  }
  if (foldConsumers != 1)
    return failure();
  int64_t outputBuffer = materializationTask.getOutputBuffers().front();
  if (outputBuffer < 0 || outputBuffer >= buffers->size())
    return failure();
  auto output =
      dyn_cast<RankedTensorType>((*buffers)[outputBuffer].getTensorType());
  auto mapAttributes =
      materializationTask->getAttrOfType<ArrayAttr>("epilogue_input_maps");
  if (!output || !output.getElementType().isF32() || !mapAttributes ||
      mapAttributes.size() !=
          materializationTask.getInputBuffers().size() - fold->getNumOperands())
    return failure();
  FailureOr<int64_t> finalElements = staticElementCount(output);
  if (failed(finalElements) || *finalElements != outputExtent)
    return failure();
  ptx.stream << "  cvt.u64.u32 %rd0, %r1;\n";
  FailureOr<SmallVector<int64_t>> coordinates =
      emitOutputCoordinates(ptx, output);
  if (failed(coordinates))
    return failure();

  llvm::SmallDenseSet<Operation *> group;
  group.insert(fold.getOperation());
  for (MapOp map : epilogueMaps)
    group.insert(map.getOperation());
  SmallVector<Value> boundaryInputs;
  llvm::SmallDenseSet<Value> seenBoundary;
  for (MapOp map : llvm::reverse(epilogueMaps)) {
    for (Value operand : map.getInputs()) {
      if (!group.contains(operand.getDefiningOp()) &&
          seenBoundary.insert(operand).second)
        boundaryInputs.push_back(operand);
    }
  }
  if (boundaryInputs.size() != mapAttributes.size())
    return failure();
  DenseMap<Value, int64_t> values;
  values[fold.getResult(0)] = 37;
  ptx.f32 = 38;
  ptx.u64 = std::max<int64_t>(ptx.u64, 10);
  ArrayRef<int64_t> mapBuffers =
      materializationTask.getInputBuffers().drop_front(fold->getNumOperands());
  for (auto [ordinal, value, bufferOrdinal, mapAttribute] :
       llvm::enumerate(boundaryInputs, mapBuffers, mapAttributes.getValue())) {
    if (bufferOrdinal < 0 || bufferOrdinal >= buffers->size())
      return failure();
    auto input =
        dyn_cast<RankedTensorType>((*buffers)[bufferOrdinal].getTensorType());
    auto indexing = dyn_cast<AffineMapAttr>(mapAttribute);
    if (!input || !indexing || input != value.getType())
      return failure();
    FailureOr<int64_t> loaded =
        emitMapInput(ptx, fold->getNumOperands() + ordinal, input,
                     indexing.getValue(), *coordinates);
    if (failed(loaded))
      return failure();
    values[value] = *loaded;
  }
  for (MapOp map : epilogueMaps) {
    SmallVector<int64_t> arguments;
    for (Value operand : map.getInputs()) {
      auto found = values.find(operand);
      if (found == values.end())
        return failure();
      arguments.push_back(found->second);
    }
    FailureOr<int64_t> result =
        emitScalarBody(ptx, map.getBody().front(), arguments);
    if (failed(result) || map.getNumResults() != 1)
      return failure();
    values[map.getResult(0)] = *result;
  }
  MapOp finalMap = epilogueMaps.back();
  auto result = values.find(finalMap.getResult(0));
  if (result == values.end())
    return failure();
  int64_t outputPointer = ptx.nextU64();
  int64_t outputParameter = materializationTask.getInputBuffers().size();
  ptx.stream << "  ld.param.u64 %rd" << outputPointer << ", [p"
             << outputParameter << "];\n  mad.lo.u64 %rd" << outputPointer
             << ", %rd0, 4, %rd" << outputPointer << ";\n  st.global.f32 [%rd"
             << outputPointer << "], %f" << result->second << ";\n";
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

struct FoldRegionPtxContext {
  PtxEmitter &ptx;
  MaterializationTaskOp task;
  SmallVectorImpl<MaterializationBufferOp> &buffers;
  llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations;
  DenseMap<Operation *, int64_t> foldValues;
  llvm::SmallDenseSet<Operation *> sourceSet;
  DenseMap<Value, int64_t> boundaryParameters;
  DenseMap<Value, int64_t> materializedValues;
};

FailureOr<int64_t> foldRegionBoundaryParameter(FoldRegionPtxContext &context,
                                               MapOp map, unsigned operand) {
  auto explicitParameter =
      context.boundaryParameters.find(map.getInputs()[operand]);
  if (explicitParameter != context.boundaryParameters.end())
    return explicitParameter->second;
  auto uses =
      context.task->getAttrOfType<ArrayAttr>("fold_region_boundary_uses");
  if (!uses)
    return failure();
  for (Attribute attribute : uses) {
    auto use = dyn_cast<DictionaryAttr>(attribute);
    if (!use || use.get("source") != map.getSource() ||
        cast<IntegerAttr>(use.get("operand")).getInt() != operand)
      continue;
    int64_t buffer = cast<IntegerAttr>(use.get("input_buffer")).getInt();
    auto found = llvm::find(context.task.getInputBuffers(), buffer);
    if (found == context.task.getInputBuffers().end())
      return failure();
    return std::distance(context.task.getInputBuffers().begin(), found);
  }
  return failure();
}

FailureOr<SmallVector<int64_t>>
emitMappedCoordinates(PtxEmitter &ptx, AffineMap map,
                      ArrayRef<int64_t> coordinates) {
  if (map.getNumDims() != coordinates.size() || map.getNumSymbols() != 0)
    return failure();
  SmallVector<int64_t> result;
  for (AffineExpr expression : map.getResults()) {
    FailureOr<int64_t> coordinate =
        emitAffineCoordinate(ptx, expression, coordinates);
    if (failed(coordinate))
      return failure();
    result.push_back(*coordinate);
  }
  return result;
}

FailureOr<std::pair<Operation *, Operation *>> compositionSourcePair(
    MaterializationTaskOp task,
    const llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations) {
  FailureOr<SmallVector<Operation *>> sources =
      compositionSourceOperations(task, sourceOperations);
  if (failed(sources) || sources->size() != 2)
    return failure();
  return std::pair<Operation *, Operation *>{sources->front(), sources->back()};
}

RankedTensorType
compositionBufferType(MaterializationTaskOp task,
                      SmallVectorImpl<MaterializationBufferOp> &buffers,
                      bool output, unsigned position) {
  ArrayRef<int64_t> ordinals =
      output ? task.getOutputBuffers() : task.getInputBuffers();
  if (position >= ordinals.size() || ordinals[position] < 0 ||
      ordinals[position] >= static_cast<int64_t>(buffers.size()))
    return {};
  return dyn_cast<RankedTensorType>(
      buffers[ordinals[position]].getTensorType());
}

FailureOr<int64_t>
emitSerialContractElement(PtxEmitter &ptx, ContractOp contract,
                          int64_t lhsParameter, int64_t rhsParameter,
                          ArrayRef<int64_t> outputCoordinates,
                          StringRef label) {
  FailureOr<SmallVector<int64_t>> domain = contractDomainShape(contract);
  if (!isClosedProjectedContract(contract, /*allowMultipleReductions=*/true) ||
      failed(domain) || outputCoordinates.size() >= domain->size())
    return failure();
  auto lhsMap = dyn_cast<AffineMapAttr>(contract.getIndexingMaps()[0]);
  auto rhsMap = dyn_cast<AffineMapAttr>(contract.getIndexingMaps()[1]);
  auto lhsType = dyn_cast<RankedTensorType>(contract.getInputs()[0].getType());
  auto rhsType = dyn_cast<RankedTensorType>(contract.getInputs()[1].getType());
  if (!lhsMap || !rhsMap || !lhsType || !rhsType)
    return failure();
  FailureOr<int64_t> reductionElements = scheduleElementCount(
      ArrayRef<int64_t>(*domain).drop_front(outputCoordinates.size()));
  if (failed(reductionElements))
    return failure();
  int64_t accumulator = ptx.nextF32();
  int64_t reductionCoordinate = ptx.nextU64();
  ptx.stream << "  mov.f32 %f" << accumulator << ", 0f00000000;\n  mov.u64 %rd"
             << reductionCoordinate << ", 0;\n$" << label
             << "_loop:\n  setp.ge.u64 %p4, %rd" << reductionCoordinate << ", "
             << *reductionElements << ";\n  @%p4 bra $" << label
             << "_done;\n";
  SmallVector<int64_t> coordinates(outputCoordinates);
  int64_t stride = *reductionElements;
  for (int64_t dimension = outputCoordinates.size();
       dimension < static_cast<int64_t>(domain->size()); ++dimension) {
    stride /= (*domain)[dimension];
    int64_t coordinate = ptx.nextU64();
    ptx.stream << "  div.u64 %rd" << coordinate << ", %rd"
               << reductionCoordinate << ", " << stride << ";\n  rem.u64 %rd"
               << coordinate << ", %rd" << coordinate << ", "
               << (*domain)[dimension] << ";\n";
    coordinates.push_back(coordinate);
  }
  FailureOr<int64_t> lhs =
      emitMapInput(ptx, lhsParameter, lhsType, lhsMap.getValue(), coordinates);
  FailureOr<int64_t> rhs =
      emitMapInput(ptx, rhsParameter, rhsType, rhsMap.getValue(), coordinates);
  if (failed(lhs) || failed(rhs))
    return failure();
  ptx.stream << "  fma.rn.f32 %f" << accumulator << ", %f" << *lhs << ", %f"
             << *rhs << ", %f" << accumulator << ";\n  add.u64 %rd"
             << reductionCoordinate << ", %rd" << reductionCoordinate
             << ", 1;\n  bra $" << label << "_loop;\n$" << label << "_done:\n";
  return accumulator;
}

enum class CompositionScalarKind { F32, U64, Predicate };

struct CompositionScalar {
  CompositionScalarKind kind;
  int64_t reg;
};

struct ContractFoldPtxContext {
  PtxEmitter &ptx;
  llvm::SmallDenseSet<Operation *> sources;
  DenseMap<Value, int64_t> boundaryParameters;
  int64_t contractLabel = 0;
};

FailureOr<SmallVector<int64_t>>
emitContiguousReshapeCoordinates(PtxEmitter &ptx, RankedTensorType input,
                                 RankedTensorType output,
                                 ArrayRef<int64_t> outputCoordinates) {
  FailureOr<int64_t> inputElements = staticElementCount(input);
  FailureOr<int64_t> outputElements = staticElementCount(output);
  if (failed(inputElements) || failed(outputElements) ||
      *inputElements != *outputElements ||
      output.getRank() != static_cast<int64_t>(outputCoordinates.size()))
    return failure();
  int64_t linear = ptx.nextU64();
  ptx.stream << "  mov.u64 %rd" << linear << ", 0;\n";
  for (auto [axis, coordinate] : llvm::enumerate(outputCoordinates))
    ptx.stream << "  mad.lo.u64 %rd" << linear << ", %rd" << linear << ", "
               << output.getDimSize(axis) << ", %rd" << coordinate << ";\n";
  SmallVector<int64_t> coordinates;
  int64_t stride = 1;
  SmallVector<int64_t> strides(input.getRank());
  for (int64_t axis = input.getRank(); axis > 0; --axis) {
    strides[axis - 1] = stride;
    stride *= input.getDimSize(axis - 1);
  }
  for (auto [axis, extent] : llvm::enumerate(input.getShape())) {
    int64_t coordinate = ptx.nextU64();
    ptx.stream << "  div.u64 %rd" << coordinate << ", %rd" << linear << ", "
               << strides[axis] << ";\n  rem.u64 %rd" << coordinate << ", %rd"
               << coordinate << ", " << extent << ";\n";
    coordinates.push_back(coordinate);
  }
  return coordinates;
}

FailureOr<CompositionScalar>
emitContractFoldValue(ContractFoldPtxContext &context, Value value,
                      ArrayRef<int64_t> coordinates);

FailureOr<CompositionScalar>
emitContractFoldRelation(ContractFoldPtxContext &context, RelationOp relation,
                         ArrayRef<int64_t> coordinates) {
  if (relation.getNumResults() != 1)
    return failure();
  auto resultType = dyn_cast<RankedTensorType>(relation.getResult(0).getType());
  if (!resultType ||
      resultType.getRank() != static_cast<int64_t>(coordinates.size()))
    return failure();
  auto operandValue = [&](unsigned ordinal,
                          ArrayRef<int64_t> operandCoordinates)
      -> FailureOr<CompositionScalar> {
    if (ordinal >= relation.getInputs().size())
      return failure();
    return emitContractFoldValue(context, relation.getInputs()[ordinal],
                                 operandCoordinates);
  };
  if (relation.getSemantics() == RelationSemantics::Iota) {
    if (relation.getParameters().size() != 1)
      return failure();
    auto dimension = dyn_cast<IntegerAttr>(relation.getParameters()[0]);
    if (!dimension || dimension.getInt() < 0 ||
        dimension.getInt() >= static_cast<int64_t>(coordinates.size()))
      return failure();
    return CompositionScalar{CompositionScalarKind::U64,
                             coordinates[dimension.getInt()]};
  }
  if (relation.getSemantics() == RelationSemantics::Constant) {
    if (relation.getParameters().size() != 1)
      return failure();
    if (auto integer = dyn_cast<IntegerAttr>(relation.getParameters()[0])) {
      int64_t reg = context.ptx.nextU64();
      context.ptx.stream << "  mov.u64 %rd" << reg << ", "
                         << integer.getValue().getZExtValue() << ";\n";
      return CompositionScalar{CompositionScalarKind::U64, reg};
    }
    return failure();
  }
  if (relation.getSemantics() == RelationSemantics::Reshape) {
    if (relation.getInputs().size() != 1)
      return failure();
    auto input = dyn_cast<RankedTensorType>(relation.getInputs()[0].getType());
    FailureOr<SmallVector<int64_t>> inputCoordinates =
        input ? emitContiguousReshapeCoordinates(context.ptx, input, resultType,
                                                 coordinates)
              : FailureOr<SmallVector<int64_t>>(failure());
    return failed(inputCoordinates) ? FailureOr<CompositionScalar>(failure())
                                    : operandValue(0, *inputCoordinates);
  }
  if (relation.getSemantics() == RelationSemantics::BroadcastInDim) {
    if (relation.getInputs().size() != 1)
      return failure();
    auto input = dyn_cast<RankedTensorType>(relation.getInputs()[0].getType());
    if (!input ||
        relation.getParameters().size() != static_cast<size_t>(input.getRank()))
      return failure();
    SmallVector<int64_t> inputCoordinates;
    for (Attribute parameter : relation.getParameters()) {
      auto dimension = dyn_cast<IntegerAttr>(parameter);
      if (!dimension || dimension.getInt() < 0 ||
          dimension.getInt() >= static_cast<int64_t>(coordinates.size()))
        return failure();
      inputCoordinates.push_back(coordinates[dimension.getInt()]);
    }
    return operandValue(0, inputCoordinates);
  }
  if (relation.getSemantics() == RelationSemantics::Add) {
    FailureOr<CompositionScalar> lhs = operandValue(0, coordinates);
    FailureOr<CompositionScalar> rhs = operandValue(1, coordinates);
    if (failed(lhs) || failed(rhs) || lhs->kind != CompositionScalarKind::U64 ||
        rhs->kind != CompositionScalarKind::U64)
      return failure();
    int64_t result = context.ptx.nextU64();
    context.ptx.stream << "  add.u64 %rd" << result << ", %rd" << lhs->reg
                       << ", %rd" << rhs->reg << ";\n";
    return CompositionScalar{CompositionScalarKind::U64, result};
  }
  if (relation.getSemantics() == RelationSemantics::Convert) {
    FailureOr<CompositionScalar> input = operandValue(0, coordinates);
    if (failed(input) || input->kind != CompositionScalarKind::U64)
      return failure();
    return *input;
  }
  if (relation.getSemantics() == RelationSemantics::Compare) {
    FailureOr<CompositionScalar> lhs = operandValue(0, coordinates);
    FailureOr<CompositionScalar> rhs = operandValue(1, coordinates);
    if (failed(lhs) || failed(rhs) || lhs->kind != CompositionScalarKind::U64 ||
        rhs->kind != CompositionScalarKind::U64 ||
        relation.getParameters().size() != 2)
      return failure();
    auto comparisonType = dyn_cast<StringAttr>(relation.getParameters()[0]);
    auto direction = dyn_cast<StringAttr>(relation.getParameters()[1]);
    if (!comparisonType || !direction)
      return failure();
    StringRef instruction = llvm::StringSwitch<StringRef>(direction.getValue())
                                .Case("EQ", "eq")
                                .Case("NE", "ne")
                                .Case("LT", "lt")
                                .Case("LE", "le")
                                .Case("GT", "gt")
                                .Case("GE", "ge")
                                .Default({});
    StringRef type = comparisonType.getValue() == "SIGNED"     ? "s64"
                     : comparisonType.getValue() == "UNSIGNED" ? "u64"
                                                               : StringRef{};
    if (instruction.empty() || type.empty())
      return failure();
    int64_t result = context.ptx.nextPred();
    context.ptx.stream << "  setp." << instruction << "." << type << " %p"
                       << result << ", %rd" << lhs->reg << ", %rd" << rhs->reg
                       << ";\n";
    return CompositionScalar{CompositionScalarKind::Predicate, result};
  }
  if (relation.getSemantics() == RelationSemantics::Select) {
    FailureOr<CompositionScalar> predicate = operandValue(0, coordinates);
    FailureOr<CompositionScalar> onTrue = operandValue(1, coordinates);
    FailureOr<CompositionScalar> onFalse = operandValue(2, coordinates);
    if (failed(predicate) || failed(onTrue) || failed(onFalse) ||
        predicate->kind != CompositionScalarKind::Predicate ||
        onTrue->kind != CompositionScalarKind::F32 ||
        onFalse->kind != CompositionScalarKind::F32)
      return failure();
    int64_t result = context.ptx.nextF32();
    context.ptx.stream << "  selp.f32 %f" << result << ", %f" << onTrue->reg
                       << ", %f" << onFalse->reg << ", %p" << predicate->reg
                       << ";\n";
    return CompositionScalar{CompositionScalarKind::F32, result};
  }
  return failure();
}

FailureOr<CompositionScalar>
emitContractFoldValue(ContractFoldPtxContext &context, Value value,
                      ArrayRef<int64_t> coordinates) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != static_cast<int64_t>(coordinates.size()))
    return failure();
  if (auto parameter = context.boundaryParameters.find(value);
      parameter != context.boundaryParameters.end()) {
    if (!type.getElementType().isF32() && !type.getElementType().isBF16())
      return failure();
    AffineMap identity =
        AffineMap::getMultiDimIdentityMap(type.getRank(), value.getContext());
    FailureOr<int64_t> loaded = emitMapInput(context.ptx, parameter->second,
                                             type, identity, coordinates);
    return failed(loaded)
               ? FailureOr<CompositionScalar>(failure())
               : CompositionScalar{CompositionScalarKind::F32, *loaded};
  }
  Operation *producer = value.getDefiningOp();
  if (!producer || !context.sources.contains(producer))
    return failure();
  if (auto relation = dyn_cast<RelationOp>(producer))
    return emitContractFoldRelation(context, relation, coordinates);
  if (auto contract = dyn_cast<ContractOp>(producer)) {
    if (contract.getNumResults() != 1 || contract.getResult(0) != value)
      return failure();
    SmallVector<int64_t> parameters;
    for (Value input : contract.getInputs()) {
      auto parameter = context.boundaryParameters.find(input);
      if (parameter == context.boundaryParameters.end())
        return failure();
      parameters.push_back(parameter->second);
    }
    if (parameters.size() != 2)
      return failure();
    std::string label = (llvm::Twine("composition_producer_contract_") +
                         llvm::Twine(context.contractLabel++))
                            .str();
    FailureOr<int64_t> emitted =
        emitSerialContractElement(context.ptx, contract, parameters[0],
                                  parameters[1], coordinates, label);
    return failed(emitted)
               ? FailureOr<CompositionScalar>(failure())
               : CompositionScalar{CompositionScalarKind::F32, *emitted};
  }
  auto map = dyn_cast<MapOp>(producer);
  if (!map || map.getNumResults() != 1 || map.getResult(0) != value ||
      map.getIndexingMaps().size() != map.getInputs().size() + 1)
    return failure();
  auto resultMap =
      dyn_cast<AffineMapAttr>(map.getIndexingMaps().getValue().back());
  if (!resultMap || !isIdentityMap(resultMap.getValue()))
    return failure();
  SmallVector<int64_t> arguments;
  for (auto [ordinal, input] : llvm::enumerate(map.getInputs())) {
    auto indexing = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[ordinal]);
    if (!indexing)
      return failure();
    FailureOr<SmallVector<int64_t>> inputCoordinates =
        emitMappedCoordinates(context.ptx, indexing.getValue(), coordinates);
    if (failed(inputCoordinates))
      return failure();
    FailureOr<CompositionScalar> argument =
        emitContractFoldValue(context, input, *inputCoordinates);
    if (failed(argument) || argument->kind != CompositionScalarKind::F32)
      return failure();
    arguments.push_back(argument->reg);
  }
  FailureOr<int64_t> emitted =
      emitScalarBody(context.ptx, map.getBody().front(), arguments);
  return failed(emitted)
             ? FailureOr<CompositionScalar>(failure())
             : CompositionScalar{CompositionScalarKind::F32, *emitted};
}

FailureOr<std::string> emitLegacyContractFoldCompositionPtx(
    MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> &buffers, ScheduleTaskOp task,
    const llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations) {
  auto kind =
      materializationTask->getAttrOfType<StringAttr>("composition_kind");
  FailureOr<std::pair<Operation *, Operation *>> sources =
      compositionSourcePair(materializationTask, sourceOperations);
  if (!kind || kind.getValue() != "contract_fold" || failed(sources))
    return failure();
  auto contract = dyn_cast<ContractOp>(sources->first);
  auto fold = dyn_cast<FoldOp>(sources->second);
  if (!contract || !fold || fold.getInputs().size() != 1 ||
      fold.getInitializers().size() != 1 || fold.getNumResults() != 1 ||
      fold.getInputs().front() != contract.getResult(0) ||
      materializationTask.getInputBuffers().size() != 3 ||
      materializationTask.getOutputBuffers().size() != 2)
    return failure();
  auto lhs = compositionBufferType(materializationTask, buffers, false, 0);
  auto rhs = compositionBufferType(materializationTask, buffers, false, 1);
  auto initializer =
      compositionBufferType(materializationTask, buffers, false, 2);
  auto full = compositionBufferType(materializationTask, buffers, true, 0);
  auto compact = compositionBufferType(materializationTask, buffers, true, 1);
  if (!lhs || !rhs || !initializer || !full || !compact ||
      lhs != contract.getInputs()[0].getType() ||
      rhs != contract.getInputs()[1].getType() ||
      initializer != fold.getInitializers().front().getType() ||
      full != contract.getResult(0).getType() ||
      compact != fold.getResult(0).getType() || full.getRank() < 2 ||
      full.getRank() > 8 || !full.getElementType().isF32() ||
      compact.getShape() != full.getShape().drop_back() ||
      !compact.getElementType().isF32())
    return failure();
  FailureOr<ScheduledGeometry> geometry =
      deriveSimt32Geometry(ScheduleTaskKind::RowFold, full.getShape(),
                           NumericalPolicy::SourceOrdered);
  FailureOr<int64_t> outputElements = staticElementCount(compact);
  if (failed(geometry) || failed(outputElements) ||
      task.getKind() != ScheduleTaskKind::RowFold ||
      task.getDomainShape() != full.getShape() ||
      task.getGridShape() != ArrayRef<int64_t>(geometry->grid) ||
      task.getTileShape() != ArrayRef<int64_t>(geometry->tile) ||
      task.getSerialTiles() != geometry->serialTiles ||
      task.getWorkgroupThreads() != geometry->workgroupThreads ||
      task.getSubgroupSize() != geometry->subgroupSize ||
      task.getScratchBytes() != geometry->scratchBytes ||
      task.getReductionAxis() != geometry->reductionAxis ||
      task.getReductionOrder() !=
          ScheduleReductionOrder::TreeAssociationFreeLeafOrderFixed)
    return failure();

  const int64_t reduction = full.getShape().back();
  const int64_t serialTiles = task.getSerialTiles();
  if (serialTiles <= 0)
    return failure();
  const int64_t activeThreads = ceilDivPositive(reduction, serialTiles);
  if (activeThreads <= 0 || activeThreads > task.getWorkgroupThreads())
    return failure();
  PtxEmitter ptx(materializationTask.getInputBuffers().size() +
                 materializationTask.getOutputBuffers().size());
  ptx.header(task.getSemanticFingerprint(), /*usesDynamicSharedMemory=*/true);
  int64_t local = ptx.nextF32();
  int64_t leaf = ptx.nextU64();
  int64_t leafEnd = ptx.nextU64();
  int64_t row = ptx.nextU64();
  ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                "  mov.u32 %r1, %ctaid.x;\n"
                "  mov.u32 %r2, %ntid.x;\n"
                "  setp.ge.u32 %p0, %r1, "
             << *outputElements << ";\n  @%p0 bra $done;\n  cvt.u64.u32 %rd"
             << row << ", %r1;\n  cvt.u64.u32 %rd" << leaf
             << ", %r0;\n  mul.lo.u64 %rd" << leaf << ", %rd" << leaf << ", "
             << serialTiles << ";\n  add.u64 %rd" << leafEnd << ", %rd" << leaf
             << ", " << serialTiles
             << ";\n  mov.u32 %r4, 1;\n$composition_fold_load:\n"
                "  setp.ge.u64 %p1, %rd"
             << leaf << ", " << reduction
             << ";\n  @%p1 bra $composition_fold_loaded;\n"
                "  setp.ge.u64 %p2, %rd"
             << leaf << ", %rd" << leafEnd
             << ";\n  @%p2 bra $composition_fold_loaded;\n"
                "  mad.lo.u64 %rd0, %rd"
             << row << ", " << reduction << ", %rd" << leaf << ";\n";
  FailureOr<SmallVector<int64_t>> fullCoordinates =
      emitOutputCoordinates(ptx, full);
  if (failed(fullCoordinates))
    return failure();
  FailureOr<int64_t> value = emitSerialContractElement(
      ptx, contract, 0, 1, *fullCoordinates, "composition_inner_contract");
  if (failed(value))
    return failure();
  int64_t fullPointer = ptx.nextU64();
  int64_t combined = ptx.nextF32();
  ptx.stream << "  ld.param.u64 %rd" << fullPointer
             << ", [p3];\n  mad.lo.u64 %rd" << fullPointer << ", %rd0, 4, %rd"
             << fullPointer << ";\n  st.global.f32 [%rd" << fullPointer
             << "], %f" << *value
             << ";\n  setp.ne.u32 %p3, %r4, 0;\n  @%p3 bra "
                "$composition_fold_first_leaf;\n  add.rn.f32 %f"
             << combined << ", %f" << local << ", %f" << *value
             << ";\n  mov.f32 %f" << local << ", %f" << combined
             << ";\n  bra $composition_fold_leaf_combined;\n"
                "$composition_fold_first_leaf:\n  mov.f32 %f"
             << local << ", %f" << *value
             << ";\n  mov.u32 %r4, 0;\n$composition_fold_leaf_combined:\n"
                "  add.u64 %rd"
             << leaf << ", %rd" << leaf
             << ", 1;\n"
                "  bra $composition_fold_load;\n"
                "$composition_fold_loaded:\n";
  int64_t threadOffset = ptx.nextU64();
  int64_t sharedBase = ptx.nextU64();
  int64_t threadPointer = ptx.nextU64();
  ptx.stream << "  setp.ge.u32 %p1, %r0, " << activeThreads
             << ";\n  @%p1 bra $composition_fold_skip_initial_store;\n"
                "  cvt.u64.u32 %rd"
             << threadOffset << ", %r0;\n  mul.lo.u64 %rd" << threadOffset
             << ", %rd" << threadOffset << ", 4;\n  mov.u64 %rd" << sharedBase
             << ", shuttle_smem;\n  cvta.to.shared.u64 %rd" << sharedBase
             << ", %rd" << sharedBase << ";\n  add.u64 %rd" << threadPointer
             << ", %rd" << sharedBase << ", %rd" << threadOffset
             << ";\n  st.shared.f32 [%rd" << threadPointer << "], %f" << local
             << ";\n$composition_fold_skip_initial_store:\n  bar.sync 0;\n";
  for (int64_t active = activeThreads; active > 1; active = (active + 1) / 2) {
    const int64_t pairs = active / 2;
    int64_t pairIndex = ptx.nextU64();
    int64_t pairPointer = ptx.nextU64();
    int64_t left = ptx.nextF32();
    int64_t right = ptx.nextF32();
    int64_t sum = ptx.nextF32();
    ptx.stream << "  setp.ge.u32 %p2, %r0, " << pairs
               << ";\n  @%p2 bra $composition_fold_skip_pair_read_" << active
               << ";\n  mul.lo.u32 %r6, %r0, 2;\n  cvt.u64.u32 %rd" << pairIndex
               << ", %r6;\n  mul.lo.u64 %rd" << pairIndex << ", %rd"
               << pairIndex << ", 4;\n  add.u64 %rd" << pairPointer << ", %rd"
               << sharedBase << ", %rd" << pairIndex << ";\n  ld.shared.f32 %f"
               << left << ", [%rd" << pairPointer << "];\n  ld.shared.f32 %f"
               << right << ", [%rd" << pairPointer << "+4];\n  add.rn.f32 %f"
               << sum << ", %f" << left << ", %f" << right
               << ";\n$composition_fold_skip_pair_read_" << active << ":\n";
    std::optional<int64_t> carry;
    if (active % 2 != 0) {
      carry = ptx.nextF32();
      ptx.stream << "  setp.ne.u32 %p3, %r0, " << pairs
                 << ";\n  @%p3 bra $composition_fold_skip_carry_read_" << active
                 << ";\n  ld.shared.f32 %f" << *carry << ", [%rd" << sharedBase
                 << "+" << (active - 1) * 4
                 << "];\n$composition_fold_skip_carry_read_" << active << ":\n";
    }
    ptx.stream << "  bar.sync 0;\n  @%p2 bra "
                  "$composition_fold_skip_pair_write_"
               << active << ";\n  cvt.u64.u32 %rd" << threadOffset
               << ", %r0;\n  mul.lo.u64 %rd" << threadOffset << ", %rd"
               << threadOffset << ", 4;\n  add.u64 %rd" << threadPointer
               << ", %rd" << sharedBase << ", %rd" << threadOffset
               << ";\n  st.shared.f32 [%rd" << threadPointer << "], %f" << sum
               << ";\n$composition_fold_skip_pair_write_" << active << ":\n";
    if (carry) {
      ptx.stream << "  @%p3 bra $composition_fold_skip_carry_write_" << active
                 << ";\n  st.shared.f32 [%rd" << sharedBase << "+" << pairs * 4
                 << "], %f" << *carry
                 << ";\n$composition_fold_skip_carry_write_" << active << ":\n";
    }
    ptx.stream << "  bar.sync 0;\n";
  }
  int64_t reduced = ptx.nextF32();
  int64_t initializerPointer = ptx.nextU64();
  int64_t initializerValue = ptx.nextF32();
  int64_t finalValue = ptx.nextF32();
  int64_t compactPointer = ptx.nextU64();
  ptx.stream << "  setp.ne.u32 %p2, %r0, 0;\n"
                "  @%p2 bra $done;\n"
                "  ld.shared.f32 %f"
             << reduced << ", [%rd" << sharedBase << "];\n  ld.param.u64 %rd"
             << initializerPointer << ", [p2];\n  ld.global.f32 %f"
             << initializerValue << ", [%rd" << initializerPointer
             << "];\n  add.rn.f32 %f" << finalValue << ", %f" << reduced
             << ", %f" << initializerValue << ";\n  ld.param.u64 %rd"
             << compactPointer
             << ", [p4];\n  cvt.u64.u32 %rd0, %r1;\n  mad.lo.u64 %rd"
             << compactPointer << ", %rd0, 4, %rd" << compactPointer
             << ";\n  st.global.f32 [%rd" << compactPointer << "], %f"
             << finalValue << ";\n";
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitGenericContractFoldCompositionPtx(
    MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> &buffers, ScheduleTaskOp task,
    const llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations) {
  auto kind =
      materializationTask->getAttrOfType<StringAttr>("composition_kind");
  FailureOr<SmallVector<Operation *>> sources =
      compositionSourceOperations(materializationTask, sourceOperations);
  if (!kind || kind.getValue() != "contract_fold" || failed(sources) ||
      sources->empty())
    return failure();
  auto fold = dyn_cast<FoldOp>(sources->back());
  if (!fold || !isClosedMultiStatePlanningFold(fold) ||
      fold.getInputs().size() != fold.getNumResults())
    return failure();
  const int64_t stateCount = fold.getNumResults();
  FailureOr<SmallVector<int64_t>> domain = planningFoldDomainShape(fold);
  ArrayRef<int64_t> reductions = fold.getReductionDimensions();
  if (failed(domain) || domain->size() < 2 || reductions.size() != 1 ||
      reductions.front() < 0 ||
      reductions.front() >= static_cast<int64_t>(domain->size()))
    return failure();
  const int64_t reductionAxis = reductions.front();
  SmallVector<int64_t> ownerShape;
  for (auto [axis, extent] : llvm::enumerate(*domain))
    if (static_cast<int64_t>(axis) != reductionAxis)
      ownerShape.push_back(extent);
  auto schedulePlan = task->getParentOfType<SchedulePlanOp>();
  FailureOr<ScheduledGeometry> geometry = deriveContractFoldCompositionGeometry(
      materializationTask, schedulePlan.getPolicy());
  if (failed(geometry) || task.getKind() != ScheduleTaskKind::RowFold ||
      task.getDomainShape() != ArrayRef<int64_t>(*domain) ||
      task.getGridShape() != ArrayRef<int64_t>(geometry->grid) ||
      task.getTileShape() != ArrayRef<int64_t>(geometry->tile) ||
      task.getSerialTiles() != geometry->serialTiles ||
      task.getWorkgroupThreads() != geometry->workgroupThreads ||
      task.getSubgroupSize() != geometry->subgroupSize ||
      task.getScratchBytes() != geometry->scratchBytes ||
      task.getReductionAxis() != geometry->reductionAxis ||
      task.getReductionStateBytesPerThread() != stateCount * 4 ||
      task.getReductionOrder() !=
          ScheduleReductionOrder::TreeAssociationFreeLeafOrderFixed)
    return failure();

  llvm::SmallDenseSet<Operation *> sourceSet(sources->begin(), sources->end());
  SmallVector<Value> cutValues;
  for (Operation *operation : *sources) {
    for (Value result : operation->getResults()) {
      if (llvm::any_of(result.getUsers(), [&](Operation *user) {
            return !sourceSet.contains(user);
          }))
        cutValues.push_back(result);
    }
  }
  if (cutValues.size() != materializationTask.getOutputBuffers().size())
    return failure();
  DenseMap<Value, int64_t> outputParameters;
  for (auto [position, value] : llvm::enumerate(cutValues))
    outputParameters[value] =
        materializationTask.getInputBuffers().size() + position;
  for (Value result : fold.getResults())
    if (!outputParameters.count(result))
      return failure();
  for (Value cut : cutValues) {
    if (llvm::is_contained(fold.getResults(), cut))
      continue;
    auto type = dyn_cast<RankedTensorType>(cut.getType());
    if (!type || !type.getElementType().isF32() ||
        type.getShape() != ArrayRef<int64_t>(*domain))
      return failure();
  }

  RegionOp region = fold->getParentOfType<RegionOp>();
  DenseMap<Value, int64_t> boundaryParameters;
  for (auto [parameter, bufferOrdinal] :
       llvm::enumerate(materializationTask.getInputBuffers())) {
    if (bufferOrdinal < 0 ||
        bufferOrdinal >= static_cast<int64_t>(buffers.size()))
      return failure();
    auto sourceArgument =
        buffers[bufferOrdinal]->getAttrOfType<IntegerAttr>("source_argument");
    if (!sourceArgument || sourceArgument.getInt() < 0 ||
        sourceArgument.getInt() >=
            static_cast<int64_t>(region.getBody().front().getNumArguments()))
      return failure();
    boundaryParameters[region.getBody().front().getArgument(
        sourceArgument.getInt())] = parameter;
  }

  SmallVector<RankedTensorType> foldResults;
  std::optional<int64_t> outputElements;
  for (Value result : fold.getResults()) {
    auto type = dyn_cast<RankedTensorType>(result.getType());
    FailureOr<int64_t> elements =
        type ? staticElementCount(type) : FailureOr<int64_t>(failure());
    if (!type || !type.getElementType().isF32() || failed(elements) ||
        (outputElements && *outputElements != *elements))
      return failure();
    outputElements = *elements;
    foldResults.push_back(type);
  }
  if (!outputElements || *outputElements > UINT32_MAX)
    return failure();
  FailureOr<int64_t> ownerElements = scheduleElementCount(ownerShape);
  if (failed(ownerElements) || *outputElements != *ownerElements)
    return failure();
  const int64_t reduction = (*domain)[reductionAxis];
  const int64_t serialTiles = task.getSerialTiles();
  const int64_t activeThreads = ceilDivPositive(reduction, serialTiles);
  if (serialTiles <= 0 || activeThreads <= 0 ||
      activeThreads > task.getWorkgroupThreads())
    return failure();

  PtxEmitter ptx(materializationTask.getInputBuffers().size() +
                 materializationTask.getOutputBuffers().size());
  ptx.header(task.getSemanticFingerprint(), /*usesDynamicSharedMemory=*/true);
  ContractFoldPtxContext context{ptx, sourceSet, std::move(boundaryParameters)};
  SmallVector<int64_t> localStates;
  for (int64_t state = 0; state < stateCount; ++state)
    localStates.push_back(ptx.nextF32());
  int64_t leaf = ptx.nextU64();
  int64_t leafEnd = ptx.nextU64();
  int64_t row = ptx.nextU64();
  ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                "  mov.u32 %r1, %ctaid.x;\n"
                "  mov.u32 %r2, %ntid.x;\n"
                "  setp.ge.u32 %p0, %r1, "
             << *outputElements << ";\n  @%p0 bra $done;\n  cvt.u64.u32 %rd"
             << row << ", %r1;\n  mov.u64 %rd0, %rd" << row << ";\n";
  auto ownerType = RankedTensorType::get(
      ownerShape, Float32Type::get(materializationTask.getContext()));
  FailureOr<SmallVector<int64_t>> ownerCoordinates =
      emitOutputCoordinates(ptx, ownerType);
  if (failed(ownerCoordinates))
    return failure();
  ptx.stream << "  cvt.u64.u32 %rd" << leaf << ", %r0;\n  mul.lo.u64 %rd"
             << leaf << ", %rd" << leaf << ", " << serialTiles
             << ";\n  add.u64 %rd" << leafEnd << ", %rd" << leaf << ", "
             << serialTiles
             << ";\n  mov.u32 %r4, 1;\n$composition_fold_load:\n"
                "  setp.ge.u64 %p1, %rd"
             << leaf << ", " << reduction
             << ";\n  @%p1 bra $composition_fold_loaded;\n"
                "  setp.ge.u64 %p2, %rd"
             << leaf << ", %rd" << leafEnd
             << ";\n  @%p2 bra $composition_fold_loaded;\n";
  SmallVector<int64_t> fullCoordinates;
  int64_t ownerAxis = 0;
  for (int64_t axis = 0; axis < static_cast<int64_t>(domain->size()); ++axis)
    fullCoordinates.push_back(
        axis == reductionAxis ? leaf : (*ownerCoordinates)[ownerAxis++]);
  FailureOr<int64_t> fullLinear =
      emitRowMajorLinearIndex(ptx, fullCoordinates, *domain);
  if (failed(fullLinear))
    return failure();

  DenseMap<Value, int64_t> leafValues;
  for (Value cut : cutValues) {
    auto type = dyn_cast<RankedTensorType>(cut.getType());
    if (!type || type.getShape() != ArrayRef<int64_t>(*domain))
      continue;
    FailureOr<CompositionScalar> emitted =
        emitContractFoldValue(context, cut, fullCoordinates);
    if (failed(emitted) || emitted->kind != CompositionScalarKind::F32)
      return failure();
    leafValues[cut] = emitted->reg;
    int64_t pointer = ptx.nextU64();
    ptx.stream << "  ld.param.u64 %rd" << pointer << ", [p"
               << outputParameters.lookup(cut) << "];\n  mad.lo.u64 %rd"
               << pointer << ", %rd" << *fullLinear << ", 4, %rd" << pointer
               << ";\n  st.global.f32 [%rd" << pointer << "], %f"
               << emitted->reg << ";\n";
  }

  SmallVector<int64_t> leafStates;
  ArrayRef<Attribute> foldMaps = fold.getIndexingMaps().getValue();
  for (auto [state, input] : llvm::enumerate(fold.getInputs())) {
    auto cached = leafValues.find(input);
    if (cached != leafValues.end()) {
      leafStates.push_back(cached->second);
      continue;
    }
    auto indexing = dyn_cast<AffineMapAttr>(foldMaps[state]);
    FailureOr<SmallVector<int64_t>> inputCoordinates =
        indexing
            ? emitMappedCoordinates(ptx, indexing.getValue(), fullCoordinates)
            : FailureOr<SmallVector<int64_t>>(failure());
    FailureOr<CompositionScalar> emitted =
        failed(inputCoordinates)
            ? FailureOr<CompositionScalar>(failure())
            : emitContractFoldValue(context, input, *inputCoordinates);
    if (failed(emitted) || emitted->kind != CompositionScalarKind::F32)
      return failure();
    leafStates.push_back(emitted->reg);
  }
  SmallVector<int64_t> combineArguments(localStates);
  llvm::append_range(combineArguments, leafStates);
  FailureOr<SmallVector<int64_t>> combined =
      emitScalarResults(ptx, fold.getCombiner().front(), combineArguments);
  if (failed(combined) || combined->size() != static_cast<size_t>(stateCount))
    return failure();
  ptx.stream << "  setp.ne.u32 %p3, %r4, 0;\n  @%p3 bra "
                "$composition_fold_first_leaf;\n";
  for (auto [local, value] : llvm::zip_equal(localStates, *combined))
    ptx.stream << "  mov.f32 %f" << local << ", %f" << value << ";\n";
  ptx.stream << "  bra $composition_fold_leaf_combined;\n"
                "$composition_fold_first_leaf:\n";
  for (auto [local, value] : llvm::zip_equal(localStates, leafStates))
    ptx.stream << "  mov.f32 %f" << local << ", %f" << value << ";\n";
  ptx.stream << "  mov.u32 %r4, 0;\n"
                "$composition_fold_leaf_combined:\n  add.u64 %rd"
             << leaf << ", %rd" << leaf
             << ", 1;\n  bra $composition_fold_load;\n"
                "$composition_fold_loaded:\n";

  const int64_t stateBytes = stateCount * 4;
  int64_t threadOffset = ptx.nextU64();
  int64_t sharedBase = ptx.nextU64();
  int64_t threadPointer = ptx.nextU64();
  ptx.stream << "  setp.ge.u32 %p1, %r0, " << activeThreads
             << ";\n  @%p1 bra $composition_fold_skip_initial_store;\n"
                "  cvt.u64.u32 %rd"
             << threadOffset << ", %r0;\n  mul.lo.u64 %rd" << threadOffset
             << ", %rd" << threadOffset << ", " << stateBytes
             << ";\n  mov.u64 %rd" << sharedBase
             << ", shuttle_smem;\n  cvta.to.shared.u64 %rd" << sharedBase
             << ", %rd" << sharedBase << ";\n  add.u64 %rd" << threadPointer
             << ", %rd" << sharedBase << ", %rd" << threadOffset << ";\n";
  for (auto [state, local] : llvm::enumerate(localStates))
    ptx.stream << "  st.shared.f32 [%rd" << threadPointer << "+" << state * 4
               << "], %f" << local << ";\n";
  ptx.stream << "$composition_fold_skip_initial_store:\n  bar.sync 0;\n";

  for (int64_t active = activeThreads; active > 1; active = (active + 1) / 2) {
    const int64_t pairs = active / 2;
    int64_t pairOffset = ptx.nextU64();
    int64_t pairPointer = ptx.nextU64();
    ptx.stream << "  setp.ge.u32 %p2, %r0, " << pairs
               << ";\n  @%p2 bra $composition_fold_skip_pair_read_" << active
               << ";\n  mul.lo.u32 %r6, %r0, 2;\n  cvt.u64.u32 %rd"
               << pairOffset << ", %r6;\n  mul.lo.u64 %rd" << pairOffset
               << ", %rd" << pairOffset << ", " << stateBytes
               << ";\n  add.u64 %rd" << pairPointer << ", %rd" << sharedBase
               << ", %rd" << pairOffset << ";\n";
    SmallVector<int64_t> leftStates;
    SmallVector<int64_t> rightStates;
    for (int64_t state = 0; state < stateCount; ++state) {
      int64_t left = ptx.nextF32();
      int64_t right = ptx.nextF32();
      ptx.stream << "  ld.shared.f32 %f" << left << ", [%rd" << pairPointer
                 << "+" << state * 4 << "];\n  ld.shared.f32 %f" << right
                 << ", [%rd" << pairPointer << "+" << stateBytes + state * 4
                 << "];\n";
      leftStates.push_back(left);
      rightStates.push_back(right);
    }
    ptx.stream << "$composition_fold_skip_pair_read_" << active << ":\n";
    SmallVector<int64_t> pairArguments(leftStates);
    llvm::append_range(pairArguments, rightStates);
    FailureOr<SmallVector<int64_t>> pairResults =
        emitScalarResults(ptx, fold.getCombiner().front(), pairArguments);
    if (failed(pairResults) ||
        pairResults->size() != static_cast<size_t>(stateCount))
      return failure();
    SmallVector<int64_t> carryStates;
    if (active % 2 != 0) {
      ptx.stream << "  setp.ne.u32 %p3, %r0, " << pairs
                 << ";\n  @%p3 bra $composition_fold_skip_carry_read_" << active
                 << ";\n";
      for (int64_t state = 0; state < stateCount; ++state) {
        int64_t carry = ptx.nextF32();
        ptx.stream << "  ld.shared.f32 %f" << carry << ", [%rd" << sharedBase
                   << "+" << (active - 1) * stateBytes + state * 4 << "];\n";
        carryStates.push_back(carry);
      }
      ptx.stream << "$composition_fold_skip_carry_read_" << active << ":\n";
    }
    ptx.stream << "  bar.sync 0;\n  @%p2 bra "
                  "$composition_fold_skip_pair_write_"
               << active << ";\n  cvt.u64.u32 %rd" << threadOffset
               << ", %r0;\n  mul.lo.u64 %rd" << threadOffset << ", %rd"
               << threadOffset << ", " << stateBytes << ";\n  add.u64 %rd"
               << threadPointer << ", %rd" << sharedBase << ", %rd"
               << threadOffset << ";\n";
    for (auto [state, value] : llvm::enumerate(*pairResults))
      ptx.stream << "  st.shared.f32 [%rd" << threadPointer << "+" << state * 4
                 << "], %f" << value << ";\n";
    ptx.stream << "$composition_fold_skip_pair_write_" << active << ":\n";
    if (!carryStates.empty()) {
      ptx.stream << "  @%p3 bra $composition_fold_skip_carry_write_" << active
                 << ";\n";
      for (auto [state, value] : llvm::enumerate(carryStates))
        ptx.stream << "  st.shared.f32 [%rd" << sharedBase << "+"
                   << pairs * stateBytes + state * 4 << "], %f" << value
                   << ";\n";
      ptx.stream << "$composition_fold_skip_carry_write_" << active << ":\n";
    }
    ptx.stream << "  bar.sync 0;\n";
  }

  SmallVector<int64_t> reducedStates;
  for (int64_t state = 0; state < stateCount; ++state) {
    int64_t reduced = ptx.nextF32();
    ptx.stream << "  ld.shared.f32 %f" << reduced << ", [%rd" << sharedBase
               << "+" << state * 4 << "];\n";
    reducedStates.push_back(reduced);
  }
  SmallVector<int64_t> initializerStates;
  for (Value initializer : fold.getInitializers()) {
    FailureOr<CompositionScalar> emitted =
        emitContractFoldValue(context, initializer, {});
    if (failed(emitted) || emitted->kind != CompositionScalarKind::F32)
      return failure();
    initializerStates.push_back(emitted->reg);
  }
  SmallVector<int64_t> finalArguments(initializerStates);
  llvm::append_range(finalArguments, reducedStates);
  FailureOr<SmallVector<int64_t>> finalStates =
      emitScalarResults(ptx, fold.getCombiner().front(), finalArguments);
  if (failed(finalStates) ||
      finalStates->size() != static_cast<size_t>(stateCount))
    return failure();
  ptx.stream << "  setp.ne.u32 %p2, %r0, 0;\n  @%p2 bra $done;\n";
  for (auto [state, value] : llvm::enumerate(*finalStates)) {
    auto resultMap =
        dyn_cast<AffineMapAttr>(foldMaps[fold.getInputs().size() + state]);
    FailureOr<SmallVector<int64_t>> resultCoordinates =
        resultMap
            ? emitMappedCoordinates(ptx, resultMap.getValue(), fullCoordinates)
            : FailureOr<SmallVector<int64_t>>(failure());
    FailureOr<int64_t> resultLinear =
        failed(resultCoordinates)
            ? FailureOr<int64_t>(failure())
            : emitRowMajorLinearIndex(ptx, *resultCoordinates,
                                      foldResults[state].getShape());
    if (failed(resultLinear))
      return failure();
    int64_t pointer = ptx.nextU64();
    ptx.stream << "  ld.param.u64 %rd" << pointer << ", [p"
               << outputParameters.lookup(fold.getResult(state))
               << "];\n  mad.lo.u64 %rd" << pointer << ", %rd" << *resultLinear
               << ", 4, %rd" << pointer << ";\n  st.global.f32 [%rd" << pointer
               << "], %f" << value << ";\n";
  }
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitIndexedAccumulationPtx(
    RegionOp region, MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> &buffers, ScheduleTaskOp task) {
  auto kind =
      materializationTask->getAttrOfType<StringAttr>("composition_kind");
  if (!kind || kind.getValue() != "indexed_accumulation")
    return failure();
  SmallVector<Operation *, 4> members(4, nullptr);
  for (Operation &operation : region.getBody().front()) {
    if (!isa<MapOp, RelationOp, FoldOp, ContractOp, SelectionOp>(operation))
      continue;
    if (algebraSource(&operation) != materializationTask.getSource())
      continue;
    auto member = sourceExpansionMember(&operation, "member");
    auto size = sourceExpansionMember(&operation, "size");
    if (!member || !size || *size != 4 || *member < 0 || *member >= 4 ||
        members[*member])
      return failure();
    members[*member] = &operation;
  }
  auto relation = dyn_cast_or_null<RelationOp>(members[0]);
  auto initializer = dyn_cast_or_null<MapOp>(members[1]);
  auto fold = dyn_cast_or_null<FoldOp>(members[2]);
  auto finalizer = dyn_cast_or_null<MapOp>(members[3]);
  auto recomputedSources = materializationTask->getAttrOfType<ArrayAttr>(
      "composition_recomputed_sources");
  SmallVector<RelationOp> recomputedRelations;
  if (recomputedSources) {
    for (Attribute source : recomputedSources) {
      RelationOp found;
      bool duplicate = false;
      region.getBody().walk([&](RelationOp candidate) {
        if (candidate.getSource() != source)
          return;
        if (found)
          duplicate = true;
        else
          found = candidate;
      });
      if (!found || duplicate)
        return failure();
      recomputedRelations.push_back(found);
    }
  }
  const bool recomputeIndices = !recomputedRelations.empty();
  if (!relation || !initializer || !fold || !finalizer ||
      relation.getSemantics() != RelationSemantics::IndexedContributions ||
      relation.getInputs().size() != 2 ||
      relation.getParameters().size() != 1 ||
      ((!recomputeIndices && materializationTask.getInputBuffers().size() != 3) ||
       (recomputeIndices && materializationTask.getInputBuffers().size() < 2)) ||
      materializationTask.getOutputBuffers().size() != 1)
    return failure();
  const unsigned updatesParameter = recomputeIndices ? 0 : 1;
  const unsigned baselineParameter = recomputeIndices ? 1 : 2;
  auto indices = dyn_cast<RankedTensorType>(relation.getInputs()[1].getType());
  auto updates = compositionBufferType(materializationTask, buffers, false,
                                       updatesParameter);
  auto baseline = compositionBufferType(materializationTask, buffers, false,
                                        baselineParameter);
  auto output = compositionBufferType(materializationTask, buffers, true, 0);
  auto metadata = dyn_cast<DictionaryAttr>(relation.getParameters()[0]);
  if (!updates || !indices || !baseline || !output || !metadata ||
      updates != relation.getInputs()[0].getType() ||
      baseline != output ||
      !updates.hasStaticShape() || !indices.hasStaticShape() ||
      !output.hasStaticShape() || !updates.getElementType().isBF16() ||
      !indices.getElementType().isInteger(32) ||
      !output.getElementType().isBF16())
    return failure();
  auto updateWindowDims =
      metadata.getAs<DenseI64ArrayAttr>("update_window_dims");
  auto insertedWindowDims =
      metadata.getAs<DenseI64ArrayAttr>("inserted_window_dims");
  auto scatterMap =
      metadata.getAs<DenseI64ArrayAttr>("scatter_dims_to_operand_dims");
  auto inputBatchingDims =
      metadata.getAs<DenseI64ArrayAttr>("input_batching_dims");
  auto indicesBatchingDims =
      metadata.getAs<DenseI64ArrayAttr>("scatter_indices_batching_dims");
  auto indexVectorDim = metadata.getAs<IntegerAttr>("index_vector_dim");
  auto conflict = metadata.getAs<StringAttr>("conflict_resolution");
  if (!updateWindowDims || !insertedWindowDims || !scatterMap ||
      !inputBatchingDims || !indicesBatchingDims || !indexVectorDim ||
      !conflict || conflict.getValue() != "source_ordered_add" ||
      inputBatchingDims.size() != indicesBatchingDims.size() ||
      scatterMap.size() != 1 || insertedWindowDims.size() != 1 ||
      updateWindowDims.size() + insertedWindowDims.size() +
              inputBatchingDims.size() !=
          static_cast<size_t>(output.getRank()) ||
      indexVectorDim.getInt() < 0 ||
      indexVectorDim.getInt() >= indices.getRank() ||
      indices.getDimSize(indexVectorDim.getInt()) != 1)
    return failure();
  SmallVector<int64_t> updateScatterDims;
  for (int64_t dimension = 0; dimension < updates.getRank(); ++dimension)
    if (!llvm::is_contained(updateWindowDims.asArrayRef(), dimension))
      updateScatterDims.push_back(dimension);
  SmallVector<int64_t> indicesScatterDims;
  for (int64_t dimension = 0; dimension < indices.getRank(); ++dimension)
    if (dimension != indexVectorDim.getInt())
      indicesScatterDims.push_back(dimension);
  if (updateScatterDims.size() != indicesScatterDims.size())
    return failure();
  SmallVector<int64_t> streamUpdateDims;
  for (auto [position, indicesDimension] : llvm::enumerate(indicesScatterDims))
    if (!llvm::is_contained(indicesBatchingDims.asArrayRef(), indicesDimension))
      streamUpdateDims.push_back(updateScatterDims[position]);
  FailureOr<int64_t> owners = staticElementCount(output);
  SmallVector<int64_t> streamShape;
  for (int64_t dimension : streamUpdateDims)
    streamShape.push_back(updates.getDimSize(dimension));
  FailureOr<int64_t> streams = scheduleElementCount(streamShape);
  if (failed(owners) || failed(streams) ||
      task.getGridShape() != ArrayRef<int64_t>({*owners}) ||
      task.getWorkgroupThreads() != 32 || task.getSubgroupSize() != 32)
    return failure();

  llvm::SmallDenseSet<Operation *> recomputedSet;
  SmallVector<Value> recomputedBoundaries;
  DenseMap<Value, unsigned> boundaryParameters;
  if (recomputeIndices) {
    if (recomputedRelations.size() > 8 ||
        recomputedRelations.back().getResult(0) != relation.getInputs()[1])
      return failure();
    for (RelationOp recomputed : recomputedRelations)
      recomputedSet.insert(recomputed);
    llvm::SmallDenseSet<Value> seen;
    for (RelationOp recomputed : recomputedRelations) {
      for (Value operand : recomputed.getInputs()) {
        if (recomputedSet.contains(operand.getDefiningOp()) ||
            !seen.insert(operand).second)
          continue;
        recomputedBoundaries.push_back(operand);
      }
    }
    if (materializationTask.getInputBuffers().size() !=
        recomputedBoundaries.size() + 2)
      return failure();
    for (auto [ordinal, boundary] : llvm::enumerate(recomputedBoundaries)) {
      auto bufferType = compositionBufferType(materializationTask, buffers,
                                              false, ordinal + 2);
      if (!bufferType || bufferType != boundary.getType())
        return failure();
      boundaryParameters[boundary] = ordinal + 2;
    }
  }

  PtxEmitter ptx(materializationTask.getInputBuffers().size() + 1);
  ptx.header(task.getSemanticFingerprint());
  ptx.stream << "  mov.u32 %r0, %tid.x;\n  setp.ne.u32 %p0, %r0, 0;\n"
                "  @%p0 bra $done;\n  mov.u32 %r1, %ctaid.x;\n"
                "  cvt.u64.u32 %rd0, %r1;\n";
  FailureOr<SmallVector<int64_t>> ownerCoordinates =
      emitOutputCoordinates(ptx, output);
  if (failed(ownerCoordinates))
    return failure();
  int64_t accumulator = ptx.nextF32();
  int64_t streamLinear = ptx.nextU64();
  ptx.stream << "  mov.f32 %f" << accumulator << ", 0f00000000;\n  mov.u64 %rd"
             << streamLinear
             << ", 0;\n$indexed_accumulation_loop:\n  setp.ge.u64 %p1, %rd"
             << streamLinear << ", " << *streams
             << ";\n  @%p1 bra $indexed_accumulation_finalizer;\n";
  SmallVector<int64_t> streamCoordinates;
  int64_t stride = *streams;
  for (int64_t extent : streamShape) {
    stride /= extent;
    int64_t coordinate = ptx.nextU64();
    ptx.stream << "  div.u64 %rd" << coordinate << ", %rd" << streamLinear
               << ", " << stride << ";\n  rem.u64 %rd" << coordinate << ", %rd"
               << coordinate << ", " << extent << ";\n";
    streamCoordinates.push_back(coordinate);
  }
  SmallVector<int64_t> indexCoordinates;
  size_t streamPosition = 0;
  for (int64_t dimension = 0; dimension < indices.getRank(); ++dimension) {
    if (dimension == indexVectorDim.getInt()) {
      int64_t zero = ptx.nextU64();
      ptx.stream << "  mov.u64 %rd" << zero << ", 0;\n";
      indexCoordinates.push_back(zero);
    } else if (auto found =
                   llvm::find(indicesBatchingDims.asArrayRef(), dimension);
               found != indicesBatchingDims.asArrayRef().end()) {
      size_t batchingPosition =
          std::distance(indicesBatchingDims.asArrayRef().begin(), found);
      int64_t outputDimension = inputBatchingDims[batchingPosition];
      if (outputDimension < 0 || outputDimension >= output.getRank())
        return failure();
      indexCoordinates.push_back((*ownerCoordinates)[outputDimension]);
    } else {
      indexCoordinates.push_back(streamCoordinates[streamPosition++]);
    }
  }
  int64_t target;
  if (recomputeIndices) {
    FailureOr<int64_t> recomputedTarget = emitIntegerRelationValueAt(
        ptx, recomputedRelations.back().getResult(0), indexCoordinates,
        recomputedSet, boundaryParameters);
    if (failed(recomputedTarget))
      return failure();
    target = *recomputedTarget;
  } else {
    FailureOr<int64_t> indexLinear =
        emitRowMajorLinearIndex(ptx, indexCoordinates, indices.getShape());
    if (failed(indexLinear))
      return failure();
    int64_t indicesBase = ptx.nextU64();
    int64_t indexPointer = ptx.nextU64();
    target = ptx.nextU32();
    ptx.stream << "  ld.param.u64 %rd" << indicesBase
               << ", [p0];\n  mad.lo.u64 %rd" << indexPointer << ", %rd"
               << *indexLinear << ", 4, %rd" << indicesBase
               << ";\n  ld.global.u32 %r" << target << ", [%rd" << indexPointer
               << "];\n";
  }
  int64_t matches = ptx.nextPred();
  const int64_t scatterOutputDim = scatterMap[0];
  if (scatterOutputDim < 0 || scatterOutputDim >= output.getRank())
    return failure();
  ptx.stream << "  cvt.u64.u32 %rd7, %r" << target
             << ";\n  setp.eq.u64 %p"
             << matches << ", %rd7, %rd"
             << (*ownerCoordinates)[scatterOutputDim] << ";\n  @!%p" << matches
             << " bra $indexed_accumulation_next;\n";
  SmallVector<int64_t> updateCoordinates(updates.getRank(), -1);
  streamPosition = 0;
  for (auto [position, indicesDimension] :
       llvm::enumerate(indicesScatterDims)) {
    int64_t updateDimension = updateScatterDims[position];
    if (auto found =
            llvm::find(indicesBatchingDims.asArrayRef(), indicesDimension);
        found != indicesBatchingDims.asArrayRef().end()) {
      size_t batchingPosition =
          std::distance(indicesBatchingDims.asArrayRef().begin(), found);
      int64_t outputDimension = inputBatchingDims[batchingPosition];
      if (outputDimension < 0 || outputDimension >= output.getRank())
        return failure();
      updateCoordinates[updateDimension] = (*ownerCoordinates)[outputDimension];
    } else {
      updateCoordinates[updateDimension] = streamCoordinates[streamPosition++];
    }
  }
  SmallVector<int64_t> windowOutputDims;
  for (int64_t dimension = 0; dimension < output.getRank(); ++dimension)
    if (!llvm::is_contained(insertedWindowDims.asArrayRef(), dimension) &&
        !llvm::is_contained(inputBatchingDims.asArrayRef(), dimension))
      windowOutputDims.push_back(dimension);
  if (windowOutputDims.size() != updateWindowDims.size())
    return failure();
  for (auto [position, dimension] :
       llvm::enumerate(updateWindowDims.asArrayRef())) {
    if (dimension < 0 || dimension >= updates.getRank())
      return failure();
    updateCoordinates[dimension] =
        (*ownerCoordinates)[windowOutputDims[position]];
  }
  if (llvm::is_contained(updateCoordinates, -1))
    return failure();
  FailureOr<int64_t> updateLinear =
      emitRowMajorLinearIndex(ptx, updateCoordinates, updates.getShape());
  if (failed(updateLinear))
    return failure();
  int64_t updatesBase = ptx.nextU64();
  int64_t updatePointer = ptx.nextU64();
  int64_t updateValue = ptx.nextF32();
  ptx.stream << "  ld.param.u64 %rd" << updatesBase
             << ", [p" << updatesParameter << "];\n  mad.lo.u64 %rd"
             << updatePointer << ", %rd"
             << *updateLinear << ", 2, %rd" << updatesBase
             << ";\n  ld.global.u16 %h0, [%rd" << updatePointer
             << "];\n  cvt.f32.bf16 %f" << updateValue
             << ", %h0;\n  add.rn.f32 %f" << accumulator << ", %f"
             << accumulator << ", %f" << updateValue
             << ";\n  cvt.rn.bf16.f32 %h0, %f" << accumulator
             << ";\n  cvt.f32.bf16 %f" << accumulator
             << ", %h0;\n$indexed_accumulation_next:\n  add.u64 %rd"
             << streamLinear << ", %rd" << streamLinear
             << ", 1;\n  bra $indexed_accumulation_loop;\n"
                "$indexed_accumulation_finalizer:\n";
  int64_t baselineBase = ptx.nextU64();
  int64_t baselinePointer = ptx.nextU64();
  int64_t baselineValue = ptx.nextF32();
  int64_t outputPointer = ptx.nextU64();
  ptx.stream << "  ld.param.u64 %rd" << baselineBase
             << ", [p" << baselineParameter << "];\n  mad.lo.u64 %rd"
             << baselinePointer
             << ", %rd0, 2, %rd" << baselineBase
             << ";\n  ld.global.u16 %h1, [%rd" << baselinePointer
             << "];\n  cvt.f32.bf16 %f" << baselineValue
             << ", %h1;\n  add.rn.f32 %f" << accumulator << ", %f"
             << baselineValue << ", %f" << accumulator
             << ";\n  cvt.rn.bf16.f32 %h0, %f" << accumulator
             << ";\n  ld.param.u64 %rd" << outputPointer
             << ", [p" << materializationTask.getInputBuffers().size()
             << "];\n  mad.lo.u64 %rd" << outputPointer << ", %rd0, 2, %rd"
             << outputPointer << ";\n  st.global.u16 [%rd" << outputPointer
             << "], %h0;\n";
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16 || ptx.pred > 64)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitContractFoldCompositionPtx(
    MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> &buffers, ScheduleTaskOp task,
    const llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations) {
  FailureOr<SmallVector<Operation *>> sources =
      compositionSourceOperations(materializationTask, sourceOperations);
  if (succeeded(sources) && sources->size() == 2) {
    if (auto fold = dyn_cast<FoldOp>(sources->back());
        fold && fold.getNumResults() == 1)
      return emitLegacyContractFoldCompositionPtx(materializationTask, buffers,
                                                  task, sourceOperations);
  }
  return emitGenericContractFoldCompositionPtx(materializationTask, buffers,
                                               task, sourceOperations);
}

struct MapContractLeaf {
  int64_t lhs;
  int64_t rhs;
};

FailureOr<DenseMap<Value, int64_t>> compositionBoundaryParameters(
    MaterializationTaskOp task,
    SmallVectorImpl<MaterializationBufferOp> &buffers,
    const llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations,
    ArrayRef<Operation *> sources) {
  llvm::SmallDenseSet<Operation *> consumerSet(sources.begin(), sources.end());
  DenseMap<int64_t, Value> valueForBuffer;
  RegionOp region = sources.empty()
                        ? RegionOp{}
                        : sources.front()->getParentOfType<RegionOp>();
  if (!region)
    return failure();
  for (int64_t bufferOrdinal : task.getInputBuffers()) {
    if (bufferOrdinal < 0 ||
        bufferOrdinal >= static_cast<int64_t>(buffers.size()))
      return failure();
    auto sourceArgument =
        buffers[bufferOrdinal]->getAttrOfType<IntegerAttr>("source_argument");
    if (!sourceArgument)
      continue;
    int64_t argument = sourceArgument.getInt();
    if (argument < 0 ||
        argument >=
            static_cast<int64_t>(region.getBody().front().getNumArguments()))
      return failure();
    valueForBuffer[bufferOrdinal] =
        region.getBody().front().getArgument(argument);
  }

  auto plan = task->getParentOfType<MaterializationPlanOp>();
  SmallVector<MaterializationTaskOp> planTasks(
      plan.getBody().front().getOps<MaterializationTaskOp>());
  for (MaterializationTaskOp producerTask : planTasks) {
    if (producerTask.getOrdinal() >= task.getOrdinal())
      break;
    Operation *producer =
        sourceOperations.lookup(producerTask.getSource());
    auto fusedOutputSources =
        producerTask->getAttrOfType<ArrayAttr>("fused_output_sources");
    auto foldRegionOutputSources =
        producerTask->getAttrOfType<ArrayAttr>("fold_region_output_sources");
    auto outputSources =
        fusedOutputSources ? fusedOutputSources : foldRegionOutputSources;
    if (outputSources &&
        outputSources.size() == producerTask.getOutputBuffers().size()) {
      SmallVector<Value> outputValues;
      for (Attribute source : outputSources) {
        Operation *output = sourceOperations.lookup(source);
        if (!output || output->getNumResults() != 1)
          return failure();
        outputValues.push_back(output->getResult(0));
      }
      for (auto [buffer, value] : llvm::zip_equal(
               producerTask.getOutputBuffers(), outputValues))
        valueForBuffer.try_emplace(buffer, value);
    } else if (producer && producer->getNumResults() ==
                               producerTask.getOutputBuffers().size()) {
      for (auto [buffer, value] : llvm::zip_equal(
               producerTask.getOutputBuffers(), producer->getResults()))
        valueForBuffer.try_emplace(buffer, value);
    }
    auto producerKind =
        producerTask->getAttrOfType<StringAttr>("composition_kind");
    if (!producerKind || producerKind.getValue() != "contract_fold")
      continue;
    FailureOr<SmallVector<Operation *>> producerSources =
        compositionSourceOperations(producerTask, sourceOperations);
    if (failed(producerSources))
      return failure();
    SmallVector<Value> cutValues;
    for (Operation *operation : *producerSources) {
      for (Value result : operation->getResults()) {
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              return consumerSet.contains(user);
            }))
          cutValues.push_back(result);
      }
    }
    if (cutValues.size() != producerTask.getOutputBuffers().size())
      return failure();
    for (auto [buffer, value] :
         llvm::zip_equal(producerTask.getOutputBuffers(), cutValues))
      valueForBuffer[buffer] = value;
  }

  DenseMap<Value, int64_t> parameters;
  for (auto [parameter, buffer] : llvm::enumerate(task.getInputBuffers())) {
    auto found = valueForBuffer.find(buffer);
    if (found == valueForBuffer.end() ||
        !parameters.try_emplace(found->second, parameter).second)
      return failure();
  }
  return parameters;
}

struct MapContractPtxContext {
  PtxEmitter &ptx;
  llvm::SmallDenseSet<Operation *> sources;
  DenseMap<Value, int64_t> boundaryParameters;
  DenseMap<Value, int64_t> computedValues;
  std::string labelPrefix;
  int64_t contractLabel = 0;
};

FailureOr<int64_t> emitSerialMapContractElement(
    MapContractPtxContext &context, ContractOp contract,
    ArrayRef<int64_t> outputCoordinates, StringRef label);

FailureOr<int64_t> emitMapContractValue(MapContractPtxContext &context,
                                        Value value,
                                        ArrayRef<int64_t> coordinates) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != static_cast<int64_t>(coordinates.size()))
    return failure();
  if (auto computed = context.computedValues.find(value);
      computed != context.computedValues.end())
    return computed->second;
  if (auto parameter = context.boundaryParameters.find(value);
      parameter != context.boundaryParameters.end()) {
    AffineMap identity =
        AffineMap::getMultiDimIdentityMap(type.getRank(), value.getContext());
    return emitMapInput(context.ptx, parameter->second, type, identity,
                        coordinates);
  }
  if (auto contract = value.getDefiningOp<ContractOp>();
      contract && context.sources.contains(contract) &&
      contract.getNumResults() == 1 && contract.getResult(0) == value) {
    const std::string label =
        context.labelPrefix + "_contract_" +
        std::to_string(context.contractLabel++);
    return emitSerialMapContractElement(context, contract, coordinates, label);
  }
  auto map = value.getDefiningOp<MapOp>();
  if (!map || !context.sources.contains(map) || map.getNumResults() != 1 ||
      map.getResult(0) != value ||
      map.getIndexingMaps().size() != map.getInputs().size() + 1)
    return failure();
  auto resultMap =
      dyn_cast<AffineMapAttr>(map.getIndexingMaps().getValue().back());
  if (!resultMap || !isIdentityMap(resultMap.getValue()))
    return failure();
  SmallVector<int64_t> arguments;
  for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
    auto indexing =
        dyn_cast<AffineMapAttr>(map.getIndexingMaps()[operandOrdinal]);
    if (!indexing)
      return failure();
    FailureOr<SmallVector<int64_t>> operandCoordinates =
        emitMappedCoordinates(context.ptx, indexing.getValue(), coordinates);
    if (failed(operandCoordinates))
      return failure();
    FailureOr<int64_t> argument =
        emitMapContractValue(context, operand, *operandCoordinates);
    if (failed(argument))
      return failure();
    arguments.push_back(*argument);
  }
  return emitScalarBody(context.ptx, map.getBody().front(), arguments,
                        /*allowBf16Arithmetic=*/type.getElementType().isBF16());
}

FailureOr<MapContractLeaf>
emitMapContractLeaf(MapContractPtxContext &context, ContractOp contract,
                    ArrayRef<int64_t> contractCoordinates) {
  SmallVector<int64_t> operands;
  for (auto [operandOrdinal, operand] : llvm::enumerate(contract.getInputs())) {
    auto indexing =
        dyn_cast<AffineMapAttr>(contract.getIndexingMaps()[operandOrdinal]);
    if (!indexing)
      return failure();
    FailureOr<SmallVector<int64_t>> operandCoordinates = emitMappedCoordinates(
        context.ptx, indexing.getValue(), contractCoordinates);
    if (failed(operandCoordinates))
      return failure();
    FailureOr<int64_t> emitted =
        emitMapContractValue(context, operand, *operandCoordinates);
    if (failed(emitted))
      return failure();
    operands.push_back(*emitted);
  }
  if (operands.size() != 2)
    return failure();
  return MapContractLeaf{operands[0], operands[1]};
}

FailureOr<int64_t> emitSerialMapContractElement(
    MapContractPtxContext &context, ContractOp contract,
    ArrayRef<int64_t> outputCoordinates, StringRef label) {
  FailureOr<SmallVector<int64_t>> domain = contractDomainShape(contract);
  if (!isClosedProjectedContract(contract, /*allowMultipleReductions=*/true) ||
      failed(domain) || outputCoordinates.size() >= domain->size())
    return failure();
  FailureOr<int64_t> reductionElements = scheduleElementCount(
      ArrayRef<int64_t>(*domain).drop_front(outputCoordinates.size()));
  if (failed(reductionElements))
    return failure();
  int64_t accumulator = context.ptx.nextF32();
  int64_t reductionCoordinate = context.ptx.nextU64();
  context.ptx.stream << "  mov.f32 %f" << accumulator
                     << ", 0f00000000;\n  mov.u64 %rd" << reductionCoordinate
                     << ", 0;\n$" << label << "_loop:\n  setp.ge.u64 %p3, %rd"
                     << reductionCoordinate << ", " << *reductionElements
                     << ";\n  @%p3 bra $" << label << "_done;\n";
  SmallVector<int64_t> coordinates(outputCoordinates);
  int64_t stride = *reductionElements;
  for (int64_t dimension = outputCoordinates.size();
       dimension < static_cast<int64_t>(domain->size()); ++dimension) {
    stride /= (*domain)[dimension];
    int64_t coordinate = context.ptx.nextU64();
    context.ptx.stream << "  div.u64 %rd" << coordinate << ", %rd"
                       << reductionCoordinate << ", " << stride
                       << ";\n  rem.u64 %rd" << coordinate << ", %rd"
                       << coordinate << ", " << (*domain)[dimension] << ";\n";
    coordinates.push_back(coordinate);
  }
  FailureOr<MapContractLeaf> leaf =
      emitMapContractLeaf(context, contract, coordinates);
  if (failed(leaf))
    return failure();
  context.ptx.stream << "  fma.rn.f32 %f" << accumulator << ", %f" << leaf->lhs
                     << ", %f" << leaf->rhs << ", %f" << accumulator
                     << ";\n  add.u64 %rd" << reductionCoordinate << ", %rd"
                     << reductionCoordinate << ", 1;\n  bra $" << label
                     << "_loop;\n$" << label << "_done:\n";
  return accumulator;
}

LogicalResult emitMapContractOutputStore(PtxEmitter &ptx, int64_t parameter,
                                         RankedTensorType type, int64_t value) {
  int64_t pointer = ptx.nextU64();
  const int64_t bytes = type.getElementType().isBF16() ? 2 : 4;
  ptx.stream << "  ld.param.u64 %rd" << pointer << ", [p" << parameter
             << "];\n  mad.lo.u64 %rd" << pointer << ", %rd0, " << bytes
             << ", %rd" << pointer << ";\n";
  if (type.getElementType().isBF16()) {
    int64_t rounded = ptx.nextBF16();
    ptx.stream << "  cvt.rn.bf16.f32 %h" << rounded << ", %f" << value
               << ";\n  st.global.u16 [%rd" << pointer << "], %h" << rounded
               << ";\n";
  } else if (type.getElementType().isF32()) {
    ptx.stream << "  st.global.f32 [%rd" << pointer << "], %f" << value
               << ";\n";
  } else {
    return failure();
  }
  return success();
}

FailureOr<std::string> emitTypedMultiPhaseMapContractPtx(
    MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> &buffers, ScheduleTaskOp task,
    ArrayRef<Operation *> sources,
    const llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations,
    const DenseMap<Value, int64_t> &boundaryParameters) {
  auto phaseSchedules =
      task->getAttrOfType<ArrayAttr>("composition_phase_schedules");
  auto phases =
      materializationTask->getAttrOfType<ArrayAttr>("composition_phases");
  auto bindings = materializationTask->getAttrOfType<ArrayAttr>(
      "composition_result_bindings");
  if (!phaseSchedules || !phases || !bindings ||
      phaseSchedules.size() != sources.size() ||
      phases.size() != sources.size() ||
      bindings.size() != materializationTask.getOutputBuffers().size() ||
      task.getOutputBuffers().size() !=
          materializationTask.getOutputBuffers().size() ||
      task.getScratchBytes() != 0)
    return failure();
  llvm::SmallDenseSet<Operation *> sourceSet(sources.begin(), sources.end());
  SmallVector<std::optional<unsigned>> outputForPhase(sources.size());
  for (auto [outputOrdinal, attribute] : llvm::enumerate(bindings)) {
    auto binding = dyn_cast<DictionaryAttr>(attribute);
    auto phase = binding ? binding.getAs<IntegerAttr>("phase") : IntegerAttr{};
    auto source =
        binding ? binding.getAs<SourceRefAttr>("source") : SourceRefAttr{};
    auto result =
        binding ? binding.getAs<IntegerAttr>("result") : IntegerAttr{};
    auto buffer =
        binding ? binding.getAs<IntegerAttr>("buffer") : IntegerAttr{};
    if (!binding || !phase || !source || !result || !buffer ||
        phase.getInt() < 0 ||
        phase.getInt() >= static_cast<int64_t>(sources.size()) ||
        sourceOperations.lookup(source) != sources[phase.getInt()] ||
        result.getInt() < 0 ||
        result.getInt() >= sources[phase.getInt()]->getNumResults() ||
        buffer.getInt() !=
            materializationTask.getOutputBuffers()[outputOrdinal] ||
        outputForPhase[phase.getInt()])
      return failure();
    outputForPhase[phase.getInt()] = outputOrdinal;
  }
  SmallVector<ContractOp> contracts;
  for (Operation *source : sources)
    if (auto contract = dyn_cast<ContractOp>(source))
      contracts.push_back(contract);
  if (contracts.empty())
    return failure();
  PtxEmitter ptx(materializationTask.getInputBuffers().size() +
                 materializationTask.getOutputBuffers().size());
  ptx.header(task.getSemanticFingerprint());
  ptx.stream << "  mov.u32 %r0, %tid.x;\n  mov.u32 %r1, %ctaid.x;\n"
                "  mov.u32 %r2, %ntid.x;\n";
  int64_t maximumF32 = ptx.f32;
  int64_t maximumU64 = ptx.u64;
  int64_t maximumU32 = ptx.u32;
  int64_t maximumPred = ptx.pred;
  int64_t maximumBF16 = ptx.bf16;
  for (auto [phaseOrdinal, attribute] : llvm::enumerate(phaseSchedules)) {
    auto record = cast<DictionaryAttr>(attribute);
    const int64_t offset =
        cast<IntegerAttr>(record.get("grid_offset")).getInt();
    const int64_t count = cast<IntegerAttr>(record.get("grid_count")).getInt();
    const int64_t elements =
        cast<IntegerAttr>(record.get("element_count")).getInt();
    const int64_t tileElements =
        cast<IntegerAttr>(record.get("tile_elements")).getInt();
    const std::string nextLabel =
        "composition_phase_" + std::to_string(phaseOrdinal) + "_next";
    ptx.stream << "  setp.lt.u32 %p0, %r1, " << offset << ";\n  @%p0 bra $"
               << nextLabel << ";\n  setp.ge.u32 %p1, %r1, " << offset + count
               << ";\n  @%p1 bra $" << nextLabel << ";\n  sub.u32 %r3, %r1, "
               << offset << ";\n  mad.lo.u32 %r4, %r3, " << tileElements
               << ", %r0;\n"
                  "  setp.ge.u32 %p2, %r4, "
               << elements
               << ";\n  @%p2 bra $done;\n  cvt.u64.u32 %rd0, %r4;\n";
    Operation *operation = sources[phaseOrdinal];
    Value value = operation->getResult(0);
    auto valueType = dyn_cast<RankedTensorType>(value.getType());
    if (!valueType)
      return failure();
    FailureOr<SmallVector<int64_t>> coordinates =
        emitOutputCoordinates(ptx, valueType);
    if (failed(coordinates))
      return failure();
    MapContractPtxContext context{
        ptx, sourceSet, boundaryParameters, DenseMap<Value, int64_t>(),
        "composition_phase_" + std::to_string(phaseOrdinal)};
    FailureOr<int64_t> emitted = failure();
    if (auto phaseContract = dyn_cast<ContractOp>(operation)) {
      emitted = emitSerialMapContractElement(
          context, phaseContract, *coordinates,
          "composition_phase_" + std::to_string(phaseOrdinal) + "_contract");
    } else if (isa<MapOp>(operation)) {
      emitted = emitMapContractValue(context, value, *coordinates);
    }
    if (failed(emitted))
      return failure();
    if (outputForPhase[phaseOrdinal]) {
      const unsigned outputOrdinal = *outputForPhase[phaseOrdinal];
      auto outputType = compositionBufferType(materializationTask, buffers,
                                              true, outputOrdinal);
      if (!outputType || outputType != valueType ||
          failed(emitMapContractOutputStore(
              ptx, materializationTask.getInputBuffers().size() + outputOrdinal,
              outputType, *emitted)))
        return failure();
    }
    ptx.stream << "  bra $done;\n$" << nextLabel << ":\n";
    maximumF32 = std::max(maximumF32, ptx.f32);
    maximumU64 = std::max(maximumU64, ptx.u64);
    maximumU32 = std::max(maximumU32, ptx.u32);
    maximumPred = std::max(maximumPred, ptx.pred);
    maximumBF16 = std::max(maximumBF16, ptx.bf16);
    ptx.f32 = 0;
    ptx.u64 = 8;
    ptx.u32 = 8;
    ptx.pred = 4;
    ptx.bf16 = 2;
  }
  ptx.f32 = maximumF32;
  ptx.u64 = maximumU64;
  ptx.u32 = maximumU32;
  ptx.pred = maximumPred;
  ptx.bf16 = maximumBF16;
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 32 || ptx.pred > 64)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitMapContractCompositionPtx(
    MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> &buffers, ScheduleTaskOp task,
    const llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations) {
  auto kind =
      materializationTask->getAttrOfType<StringAttr>("composition_kind");
  FailureOr<SmallVector<Operation *>> sources =
      compositionSourceOperations(materializationTask, sourceOperations);
  if (!kind || kind.getValue() != "map_contract" || failed(sources) ||
      sources->empty()) {
    materializationTask.emitOpError(
        "Map/Contract composition has invalid source provenance");
    return failure();
  }
  SmallVector<ContractOp> contracts;
  for (Operation *operation : *sources) {
    if (auto candidate = dyn_cast<ContractOp>(operation)) {
      contracts.push_back(candidate);
    } else if (!isa<MapOp>(operation)) {
      materializationTask.emitOpError(
          "Map/Contract composition contains a non-Map, non-Contract source");
      return failure();
    }
  }
  Operation *terminal = sources->back();
  if (contracts.empty() || terminal->getNumResults() != 1) {
    materializationTask.emitOpError(
        "Map/Contract composition requires Contract roots and one-result "
        "sources");
    return failure();
  }
  SmallVector<Operation *> executionSources(*sources);
  auto recomputedAttributes = materializationTask->getAttrOfType<ArrayAttr>(
      "composition_recomputed_sources");
  if (recomputedAttributes) {
    for (Attribute source : recomputedAttributes) {
      Operation *operation = sourceOperations.lookup(source);
      if (!operation)
        return failure();
      executionSources.push_back(operation);
    }
  }
  FailureOr<DenseMap<Value, int64_t>> boundaryParameters =
      compositionBoundaryParameters(materializationTask, buffers,
                                    sourceOperations, executionSources);
  if (failed(boundaryParameters) ||
      boundaryParameters->size() !=
          materializationTask.getInputBuffers().size()) {
    materializationTask.emitOpError(
        "Map/Contract physical boundary does not match its algebra values");
    return failure();
  }
  if (task->hasAttr("composition_phase_schedules"))
    return emitTypedMultiPhaseMapContractPtx(
        materializationTask, buffers, task, *sources, sourceOperations,
        *boundaryParameters);
  auto result =
      compositionBufferType(materializationTask, buffers, true, 0);
  if (!result || result != terminal->getResult(0).getType() ||
      (!result.getElementType().isF32() &&
       !result.getElementType().isBF16()) ||
      materializationTask.getOutputBuffers().size() != 1)
    return failure();
  Value publicValue = terminal->getResult(0);
  llvm::SmallDenseSet<Operation *> sourceSet(executionSources.begin(),
                                              executionSources.end());
  if (contracts.size() > 1) {
    auto terminalMap = dyn_cast<MapOp>(terminal);
    FailureOr<int64_t> elements = staticElementCount(result);
    if (!terminalMap || failed(elements) || *elements <= 0 ||
        *elements > UINT32_MAX || task.getKind() != ScheduleTaskKind::Elementwise ||
        task.getDomainShape() != result.getShape() ||
        task.getInputBuffers().size() !=
            materializationTask.getInputBuffers().size() ||
        task.getOutputBuffers().size() != 1 ||
        task.getScratchBytes() != 0)
      return failure();
    const int64_t tile = std::min<int64_t>(*elements, 256);
    if (task.getGridShape() !=
            ArrayRef<int64_t>({ceilDivPositive(*elements, tile)}) ||
        task.getTileShape() != ArrayRef<int64_t>({tile}) ||
        task.getWorkgroupThreads() != scheduleThreads(tile) ||
        task.getSubgroupSize() != 32 || task.getSerialTiles() != 1)
      return failure();

    PtxEmitter ptx(materializationTask.getInputBuffers().size() + 1);
    MapContractPtxContext context{ptx, sourceSet, *boundaryParameters,
                                  DenseMap<Value, int64_t>(),
                                  "composition_root"};
    ptx.header(task.getSemanticFingerprint());
    if (failed(emitLinearThread(ptx, *elements)))
      return failure();
    FailureOr<SmallVector<int64_t>> outputCoordinates =
        emitOutputCoordinates(ptx, result);
    if (failed(outputCoordinates))
      return failure();
    for (auto [contractOrdinal, root] : llvm::enumerate(contracts)) {
      if (!isClosedProjectedContract(root,
                                     /*allowMultipleReductions=*/true) ||
          root.getInputs().size() != 2)
        return failure();
      FailureOr<AffineMap> rootCoordinates = mapCoordinatesToAncestor(
          publicValue, root.getResult(0), sourceSet);
      FailureOr<SmallVector<int64_t>> mappedCoordinates =
          failed(rootCoordinates)
              ? FailureOr<SmallVector<int64_t>>(failure())
              : emitMappedCoordinates(ptx, *rootCoordinates,
                                      *outputCoordinates);
      auto lhs = context.boundaryParameters.find(root.getInputs()[0]);
      auto rhs = context.boundaryParameters.find(root.getInputs()[1]);
      if (failed(mappedCoordinates) ||
          lhs == context.boundaryParameters.end() ||
          rhs == context.boundaryParameters.end())
        return failure();
      const std::string label =
          "composition_contract_root_" + std::to_string(contractOrdinal);
      FailureOr<int64_t> accumulator = emitSerialContractElement(
          ptx, root, lhs->second, rhs->second, *mappedCoordinates, label);
      if (failed(accumulator))
        return failure();
      context.computedValues[root.getResult(0)] = *accumulator;
    }
    FailureOr<int64_t> finalized =
        emitMapContractValue(context, publicValue, *outputCoordinates);
    if (failed(finalized))
      return failure();
    int64_t outputPointer = ptx.nextU64();
    const int64_t resultBytes = result.getElementType().isBF16() ? 2 : 4;
    ptx.stream << "  ld.param.u64 %rd" << outputPointer << ", [p"
               << materializationTask.getInputBuffers().size()
               << "];\n  mad.lo.u64 %rd" << outputPointer << ", %rd0, "
               << resultBytes << ", %rd" << outputPointer << ";\n";
    if (result.getElementType().isBF16()) {
      int64_t rounded = ptx.nextBF16();
      ptx.stream << "  cvt.rn.bf16.f32 %h" << rounded << ", %f"
                 << *finalized << ";\n  st.global.u16 [%rd" << outputPointer
                 << "], %h" << rounded << ";\n";
    } else {
      ptx.stream << "  st.global.f32 [%rd" << outputPointer << "], %f"
                 << *finalized << ";\n";
    }
    if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16 || ptx.pred > 64)
      return failure();
    ptx.finish();
    return ptx.text;
  }

  ContractOp contract = contracts.front();
  FailureOr<AffineMap> contractCoordinates =
      mapCoordinatesToAncestor(publicValue, contract.getResult(0), sourceSet);
  FailureOr<SmallVector<int64_t>> domain = contractDomainShape(contract);
  if (!isClosedProjectedContract(contract) || failed(contractCoordinates) ||
      failed(domain) || task.getKind() != ScheduleTaskKind::Contract ||
      task.getDomainShape() != ArrayRef<int64_t>(*domain) ||
      task.getReductionAxis() != static_cast<int64_t>(domain->size()) - 1 ||
      task.getInputBuffers().size() !=
          materializationTask.getInputBuffers().size() ||
      task.getOutputBuffers().size() != 1 || task.getScratchBytes() != 0) {
    materializationTask.emitOpError(
        "Map/Contract composition does not match its scheduled contraction");
    return failure();
  }
  const bool fast =
      task.getReductionOrder() == ScheduleReductionOrder::LeafOrderFree;
  if (!fast && task.getReductionOrder() !=
                   ScheduleReductionOrder::SerialAscendingFmaRnF32) {
    materializationTask.emitOpError(
        "Map/Contract composition has unsupported reduction order");
    return failure();
  }
  FailureOr<int64_t> outputCount =
      scheduleElementCount(task.getDomainShape().drop_back());
  if (failed(outputCount) || *outputCount > UINT32_MAX) {
    materializationTask.emitOpError(
        "Map/Contract output domain exceeds bounded CUDA indexing");
    return failure();
  }
  const int64_t outputs = *outputCount;
  const int64_t reduction = task.getDomainShape().back();
  if (fast) {
    const int64_t outputsPerBlock = std::min<int64_t>(outputs, 8);
    if (task.getGridShape() !=
            ArrayRef<int64_t>({ceilDivPositive(outputs, outputsPerBlock)}) ||
        task.getTileShape() != ArrayRef<int64_t>({outputsPerBlock}) ||
        task.getWorkgroupThreads() != outputsPerBlock * 32 ||
        task.getSubgroupSize() != 32 ||
        task.getSerialTiles() != ceilDivPositive(reduction, 32)) {
      materializationTask.emitOpError(
          "Map/Contract composition fast schedule geometry is inconsistent");
      return failure();
    }
    PtxEmitter ptx(materializationTask.getInputBuffers().size() + 1);
    MapContractPtxContext context{ptx, sourceSet, *boundaryParameters,
                                  DenseMap<Value, int64_t>(),
                                  "composition_contract"};
    ptx.header(task.getSemanticFingerprint());
    ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                  "  shr.u32 %r1, %r0, 5;\n"
                  "  and.b32 %r2, %r0, 31;\n"
                  "  mov.u32 %r3, %ctaid.x;\n"
                  "  mad.lo.u32 %r4, %r3, "
               << outputsPerBlock << ", %r1;\n  setp.ge.u32 %p0, %r4, "
               << outputs << ";\n  @%p0 bra $done;\n  cvt.u64.u32 %rd0, %r4;\n";
    FailureOr<SmallVector<int64_t>> outputCoordinates =
        emitOutputCoordinates(ptx, result);
    FailureOr<SmallVector<int64_t>> contractOutputCoordinates =
        failed(outputCoordinates)
            ? FailureOr<SmallVector<int64_t>>(failure())
            : emitMappedCoordinates(ptx, *contractCoordinates,
                                    *outputCoordinates);
    if (failed(contractOutputCoordinates) ||
        contractOutputCoordinates->size() + 1 != domain->size()) {
      materializationTask.emitOpError(
          "Map/Contract output relation cannot reconstruct Contract owners");
      return failure();
    }
    int64_t accumulator = ptx.nextF32();
    int64_t reductionCoordinate = ptx.nextU64();
    ptx.stream << "  mov.f32 %f" << accumulator
               << ", 0f00000000;\n  cvt.u64.u32 %rd" << reductionCoordinate
               << ", %r2;\n$composition_contract_fast_loop:\n"
                  "  setp.ge.u64 %p1, %rd"
               << reductionCoordinate << ", " << reduction
               << ";\n  @%p1 bra $composition_contract_fast_reduce;\n";
    SmallVector<int64_t> leafCoordinates(*contractOutputCoordinates);
    leafCoordinates.push_back(reductionCoordinate);
    FailureOr<MapContractLeaf> leaf =
        emitMapContractLeaf(context, contract, leafCoordinates);
    if (failed(leaf)) {
      materializationTask.emitOpError(
          "Map/Contract leaf operands cannot be reconstructed");
      return failure();
    }
    ptx.stream << "  fma.rn.f32 %f" << accumulator << ", %f" << leaf->lhs
               << ", %f" << leaf->rhs << ", %f" << accumulator
               << ";\n  add.u64 %rd" << reductionCoordinate << ", %rd"
               << reductionCoordinate
               << ", 32;\n  bra $composition_contract_fast_loop;\n"
                  "$composition_contract_fast_reduce:\n  mov.b32 %r5, %f"
               << accumulator << ";\n";
    int64_t shuffled = ptx.nextF32();
    for (int64_t offset : {16, 8, 4, 2, 1}) {
      ptx.stream << "  shfl.sync.down.b32 %r6, %r5, " << offset
                 << ", 31, 0xffffffff;\n  mov.b32 %f" << shuffled
                 << ", %r6;\n  add.rn.f32 %f" << accumulator << ", %f"
                 << accumulator << ", %f" << shuffled << ";\n  mov.b32 %r5, %f"
                 << accumulator << ";\n";
    }
    context.computedValues[contract.getResult(0)] = accumulator;
    FailureOr<int64_t> finalized =
        emitMapContractValue(context, publicValue, *outputCoordinates);
    if (failed(finalized)) {
      materializationTask.emitOpError(
          "Map/Contract finalizer cannot be evaluated from its Contract root");
      return failure();
    }
    int64_t outputPointer = ptx.nextU64();
    const int64_t resultBytes = result.getElementType().isBF16() ? 2 : 4;
    ptx.stream << "  setp.ne.u32 %p2, %r2, 0;\n  @%p2 bra $done;\n"
                  "  ld.param.u64 %rd"
               << outputPointer << ", [p"
               << materializationTask.getInputBuffers().size()
               << "];\n  mad.lo.u64 %rd" << outputPointer << ", %rd0, "
               << resultBytes << ", %rd" << outputPointer << ";\n";
    if (result.getElementType().isBF16())
      ptx.stream << "  cvt.rn.bf16.f32 %h0, %f" << *finalized
                 << ";\n  st.global.u16 [%rd" << outputPointer << "], %h0;\n";
    else
      ptx.stream << "  st.global.f32 [%rd" << outputPointer << "], %f"
                 << *finalized << ";\n";
    if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16) {
      materializationTask.emitOpError(
          "Map/Contract composition exceeds bounded PTX register estimate");
      return failure();
    }
    ptx.finish();
    return ptx.text;
  }

  const int64_t tile = std::min<int64_t>(outputs, 256);
  if (task.getGridShape() !=
          ArrayRef<int64_t>({ceilDivPositive(outputs, tile)}) ||
      task.getTileShape() != ArrayRef<int64_t>({tile}) ||
      task.getWorkgroupThreads() != scheduleThreads(tile) ||
      task.getSubgroupSize() != 32 || task.getSerialTiles() != reduction)
    return failure();
  PtxEmitter ptx(materializationTask.getInputBuffers().size() + 1);
  MapContractPtxContext context{ptx, sourceSet, *boundaryParameters,
                                DenseMap<Value, int64_t>()};
  ptx.header(task.getSemanticFingerprint());
  if (failed(emitLinearThread(ptx, outputs)))
    return failure();
  FailureOr<SmallVector<int64_t>> outputCoordinates =
      emitOutputCoordinates(ptx, result);
  FailureOr<SmallVector<int64_t>> contractOutputCoordinates =
      failed(outputCoordinates)
          ? FailureOr<SmallVector<int64_t>>(failure())
          : emitMappedCoordinates(ptx, *contractCoordinates,
                                  *outputCoordinates);
  if (failed(contractOutputCoordinates) ||
      contractOutputCoordinates->size() + 1 != domain->size())
    return failure();
  int64_t accumulator = ptx.nextF32();
  int64_t reductionCoordinate = ptx.nextU64();
  ptx.stream << "  mov.f32 %f" << accumulator << ", 0f00000000;\n  mov.u64 %rd"
             << reductionCoordinate
             << ", 0;\n$composition_contract_loop:\n"
                "  setp.ge.u64 %p1, %rd"
             << reductionCoordinate << ", " << reduction
             << ";\n  @%p1 bra $composition_contract_done;\n";
  SmallVector<int64_t> leafCoordinates(*contractOutputCoordinates);
  leafCoordinates.push_back(reductionCoordinate);
  FailureOr<MapContractLeaf> leaf =
      emitMapContractLeaf(context, contract, leafCoordinates);
  if (failed(leaf))
    return failure();
  int64_t outputPointer = ptx.nextU64();
  ptx.stream << "  fma.rn.f32 %f" << accumulator << ", %f" << leaf->lhs
             << ", %f" << leaf->rhs << ", %f" << accumulator
             << ";\n  add.u64 %rd" << reductionCoordinate << ", %rd"
             << reductionCoordinate
             << ", 1;\n  bra $composition_contract_loop;\n"
                "$composition_contract_done:\n";
  context.computedValues[contract.getResult(0)] = accumulator;
  FailureOr<int64_t> finalized =
      emitMapContractValue(context, publicValue, *outputCoordinates);
  if (failed(finalized))
    return failure();
  const int64_t resultBytes = result.getElementType().isBF16() ? 2 : 4;
  ptx.stream << "  ld.param.u64 %rd" << outputPointer << ", [p"
             << materializationTask.getInputBuffers().size()
             << "];\n  mad.lo.u64 %rd" << outputPointer << ", %rd0, "
             << resultBytes << ", %rd" << outputPointer << ";\n";
  if (result.getElementType().isBF16())
    ptx.stream << "  cvt.rn.bf16.f32 %h0, %f" << *finalized
               << ";\n  st.global.u16 [%rd" << outputPointer << "], %h0;\n";
  else
    ptx.stream << "  st.global.f32 [%rd" << outputPointer << "], %f"
               << *finalized << ";\n";
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<int64_t> emitFoldRegionValue(FoldRegionPtxContext &context,
                                       Value value,
                                       ArrayRef<int64_t> coordinates) {
  auto materialized = context.materializedValues.find(value);
  if (materialized != context.materializedValues.end())
    return materialized->second;
  Operation *producer = value.getDefiningOp();
  if (auto fold = dyn_cast_or_null<FoldOp>(producer)) {
    auto found = context.foldValues.find(fold);
    return found == context.foldValues.end() ? FailureOr<int64_t>(failure())
                                             : found->second;
  }
  auto map = dyn_cast_or_null<MapOp>(producer);
  if (!map || map.getNumResults() != 1 || map.getResult(0) != value)
    return failure();
  auto output = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  if (!output || output.getRank() != static_cast<int64_t>(coordinates.size()))
    return failure();
  SmallVector<int64_t> arguments;
  for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
    auto indexing =
        dyn_cast<AffineMapAttr>(map.getIndexingMaps()[operandOrdinal]);
    if (!indexing)
      return failure();
    Operation *operandProducer = operand.getDefiningOp();
    if (isa_and_nonnull<MapOp>(operandProducer) &&
        context.sourceSet.contains(operandProducer)) {
      FailureOr<SmallVector<int64_t>> operandCoordinates =
          emitMappedCoordinates(context.ptx, indexing.getValue(), coordinates);
      if (failed(operandCoordinates))
        return failure();
      FailureOr<int64_t> argument =
          emitFoldRegionValue(context, operand, *operandCoordinates);
      if (failed(argument))
        return failure();
      arguments.push_back(*argument);
      continue;
    }
    if (isa_and_nonnull<FoldOp>(operandProducer)) {
      auto found = context.foldValues.find(operandProducer);
      if (found != context.foldValues.end()) {
        if (!isCanonicalFoldResultExpansion(map, operandOrdinal))
          return failure();
        arguments.push_back(found->second);
        continue;
      }
    }
    FailureOr<int64_t> parameter =
        foldRegionBoundaryParameter(context, map, operandOrdinal);
    if (failed(parameter))
      return failure();
    int64_t bufferOrdinal = context.task.getInputBuffers()[*parameter];
    if (bufferOrdinal < 0 || bufferOrdinal >= context.buffers.size())
      return failure();
    auto input = dyn_cast<RankedTensorType>(
        context.buffers[bufferOrdinal].getTensorType());
    if (!input || input != operand.getType())
      return failure();
    FailureOr<int64_t> argument = emitMapInput(
        context.ptx, *parameter, input, indexing.getValue(), coordinates);
    if (failed(argument))
      return failure();
    arguments.push_back(*argument);
  }
  return emitScalarBody(context.ptx, map.getBody().front(), arguments);
}

FailureOr<std::string> emitFoldRegionPtx(
    MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> &buffers, ScheduleTaskOp task,
    llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations) {
  auto sourceAttributes =
      materializationTask->getAttrOfType<ArrayAttr>("fold_region_sources");
  auto phaseAttributes =
      materializationTask->getAttrOfType<ArrayAttr>("fold_region_phases");
  auto outputAttributes = materializationTask->getAttrOfType<ArrayAttr>(
      "fold_region_output_sources");
  if (!sourceAttributes || !phaseAttributes || phaseAttributes.empty() ||
      !outputAttributes || outputAttributes.empty() ||
      outputAttributes.size() != materializationTask.getOutputBuffers().size())
    return failure();
  SmallVector<Operation *> sources;
  llvm::SmallDenseSet<Operation *> sourceSet;
  for (Attribute source : sourceAttributes) {
    Operation *operation = sourceOperations.lookup(source);
    if (!operation || !sourceSet.insert(operation).second)
      return failure();
    sources.push_back(operation);
  }
  SmallVector<Operation *> outputOperations;
  for (Attribute source : outputAttributes) {
    Operation *operation = sourceOperations.lookup(source);
    if (!operation || !sourceSet.contains(operation) ||
        operation->getNumResults() != 1)
      return failure();
    outputOperations.push_back(operation);
  }
  SmallVector<FoldOp> phases;
  for (Attribute attribute : phaseAttributes) {
    auto dictionary = dyn_cast<DictionaryAttr>(attribute);
    auto fold = dictionary ? dyn_cast_or_null<FoldOp>(sourceOperations.lookup(
                                 dictionary.get("source")))
                           : FoldOp{};
    if (!fold)
      return failure();
    phases.push_back(fold);
  }
  FoldOp primary = phases.front();
  auto domain = cast<RankedTensorType>(primary.getInputs().front().getType());
  auto primaryResult = cast<RankedTensorType>(primary.getResult(0).getType());
  FailureOr<int64_t> outputExtent = staticElementCount(primaryResult);
  if (failed(outputExtent) || *outputExtent <= 0 ||
      task.getGridShape() != ArrayRef<int64_t>({*outputExtent}) ||
      task.getWorkgroupThreads() <= 0 || task.getWorkgroupThreads() > 256 ||
      task.getSerialTiles() <= 0)
    return failure();
  ArrayRef<int64_t> primaryReduction = primary.getReductionDimensions();
  const bool row = primaryReduction.size() == 1 &&
                   primaryReduction.front() == domain.getRank() - 1;
  const bool fullPrefix =
      primaryReduction.size() == static_cast<size_t>(domain.getRank() - 1) &&
      llvm::all_of(llvm::enumerate(primaryReduction), [](auto axis) {
        return static_cast<int64_t>(axis.index()) == axis.value();
      });
  if (!row && !fullPrefix)
    return failure();
  FailureOr<int64_t> reductionExtent =
      fullPrefix ? scheduleElementCount(domain.getShape().drop_back())
                 : FailureOr<int64_t>(domain.getShape().back());
  if (failed(reductionExtent))
    return failure();
  if (task.getScratchBytes() !=
      task.getWorkgroupThreads() * static_cast<int64_t>(sizeof(float)))
    return failure();

  PtxEmitter ptx(materializationTask.getInputBuffers().size() +
                 materializationTask.getOutputBuffers().size());
  ptx.header(task.getSemanticFingerprint(), /*usesDynamicSharedMemory=*/true);
  FoldRegionPtxContext context{ptx,
                               materializationTask,
                               buffers,
                               sourceOperations,
                               DenseMap<Operation *, int64_t>(),
                               sourceSet,
                               DenseMap<Value, int64_t>(),
                               DenseMap<Value, int64_t>()};
  auto boundaryUses = materializationTask->getAttrOfType<ArrayAttr>(
      "fold_region_boundary_uses");
  if (!boundaryUses)
    return failure();
  for (Attribute attribute : boundaryUses) {
    auto use = dyn_cast<DictionaryAttr>(attribute);
    auto map = use ? dyn_cast_or_null<MapOp>(
                         sourceOperations.lookup(use.get("source")))
                   : MapOp{};
    auto operand =
        use ? dyn_cast_or_null<IntegerAttr>(use.get("operand")) : IntegerAttr{};
    auto inputBuffer =
        use ? dyn_cast_or_null<IntegerAttr>(use.get("input_buffer"))
            : IntegerAttr{};
    if (!map || !operand || !inputBuffer || operand.getInt() < 0 ||
        operand.getInt() >= static_cast<int64_t>(map.getInputs().size()))
      return failure();
    auto found =
        llvm::find(materializationTask.getInputBuffers(), inputBuffer.getInt());
    if (found == materializationTask.getInputBuffers().end())
      return failure();
    int64_t parameter =
        std::distance(materializationTask.getInputBuffers().begin(), found);
    Value value = map.getInputs()[operand.getInt()];
    auto [position, inserted] =
        context.boundaryParameters.try_emplace(value, parameter);
    if (!inserted && position->second != parameter)
      return failure();
  }
  auto emitLeaf = [&](Value value,
                      ArrayRef<int64_t> coordinates) -> FailureOr<int64_t> {
    auto boundary = context.boundaryParameters.find(value);
    if (boundary == context.boundaryParameters.end())
      return emitFoldRegionValue(context, value, coordinates);
    int64_t bufferOrdinal =
        materializationTask.getInputBuffers()[boundary->second];
    if (bufferOrdinal < 0 || bufferOrdinal >= buffers.size())
      return failure();
    auto input =
        dyn_cast<RankedTensorType>(buffers[bufferOrdinal].getTensorType());
    if (!input || input.getRank() != static_cast<int64_t>(coordinates.size()))
      return failure();
    return emitMapInput(ptx, boundary->second, input,
                        AffineMap::getMultiDimIdentityMap(
                            input.getRank(), materializationTask.getContext()),
                        coordinates);
  };
  SmallVector<int64_t> phaseResultRegisters;
  for (size_t phase = 0; phase < phases.size(); ++phase)
    phaseResultRegisters.push_back(ptx.nextF32());
  const int64_t phaseScratchBase = ptx.f32;
  int64_t f32RegisterHighWater = ptx.f32;
  int64_t accumulator = ptx.nextF32();
  ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                "  mov.u32 %r1, %ctaid.x;\n"
                "  setp.ge.u32 %p0, %r1, "
             << *outputExtent
             << ";\n  @%p0 bra $done;\n"
                "  mul.lo.u32 %r4, %r0, "
             << task.getSerialTiles() << ";\n  add.u32 %r7, %r4, "
             << task.getSerialTiles() << ";\n  min.u32 %r7, %r7, "
             << *reductionExtent << ";\n  mov.f32 %f" << accumulator
             << ", 0f00000000;\n$fold_region_load:\n"
                "  setp.ge.u32 %p1, %r4, %r7;\n"
                "  @%p1 bra $fold_region_loaded;\n"
                "  cvt.u64.u32 %rd0, %r4;\n"
                "  cvt.u64.u32 %rd1, %r1;\n";
  if (fullPrefix)
    ptx.stream << "  mad.lo.u64 %rd0, %rd0, " << *outputExtent << ", %rd1;\n";
  else
    ptx.stream << "  mad.lo.u64 %rd0, %rd1, " << *reductionExtent
               << ", %rd0;\n";
  FailureOr<SmallVector<int64_t>> reductionCoordinates =
      emitOutputCoordinates(ptx, domain);
  if (failed(reductionCoordinates))
    return failure();
  FailureOr<int64_t> leaf =
      emitLeaf(primary.getInputs().front(), *reductionCoordinates);
  if (failed(leaf))
    return failure();
  int64_t combined = ptx.nextF32();
  ptx.stream << "  add.rn.f32 %f" << combined << ", %f" << accumulator << ", %f"
             << *leaf << ";\n  mov.f32 %f" << accumulator << ", %f" << combined
             << ";\n  add.u32 %r4, %r4, 1;\n"
                "  bra $fold_region_load;\n$fold_region_loaded:\n";

  ptx.stream << "  cvt.u64.u32 %rd2, %r0;\n"
                "  mul.lo.u64 %rd2, %rd2, 4;\n"
                "  mov.u64 %rd3, shuttle_smem;\n"
                "  cvta.to.shared.u64 %rd3, %rd3;\n"
                "  add.u64 %rd2, %rd3, %rd2;\n"
                "  st.shared.f32 [%rd2], %f"
             << accumulator << ";\n  bar.sync 0;\n";
  for (int64_t active = task.getWorkgroupThreads(); active > 1;
       active = (active + 1) / 2) {
    int64_t pairs = active / 2;
    int64_t left = ptx.nextF32();
    int64_t right = ptx.nextF32();
    int64_t sum = ptx.nextF32();
    ptx.stream << "  setp.ge.u32 %p2, %r0, " << pairs
               << ";\n  @%p2 bra $fold_region_skip_pair_read" << active
               << ";\n  mul.lo.u32 %r5, %r0, 2;\n"
                  "  cvt.u64.u32 %rd4, %r5;\n"
                  "  mul.lo.u64 %rd4, %rd4, 4;\n"
                  "  add.u64 %rd4, %rd3, %rd4;\n"
                  "  ld.shared.f32 %f"
               << left << ", [%rd4];\n  ld.shared.f32 %f" << right
               << ", [%rd4+4];\n  add.rn.f32 %f" << sum << ", %f" << left
               << ", %f" << right << ";\n$fold_region_skip_pair_read" << active
               << ":\n";
    std::optional<int64_t> carry;
    if (active % 2 != 0) {
      carry = ptx.nextF32();
      ptx.stream << "  setp.ne.u32 %p3, %r0, " << pairs
                 << ";\n  @%p3 bra $fold_region_skip_carry_read" << active
                 << ";\n  ld.shared.f32 %f" << *carry << ", [%rd3+"
                 << (active - 1) * static_cast<int64_t>(sizeof(float))
                 << "];\n$fold_region_skip_carry_read" << active << ":\n";
    }
    ptx.stream << "  bar.sync 0;\n"
                  "  @%p2 bra $fold_region_skip_pair_write"
               << active
               << ";\n  cvt.u64.u32 %rd5, %r0;\n"
                  "  mul.lo.u64 %rd5, %rd5, 4;\n"
                  "  add.u64 %rd5, %rd3, %rd5;\n"
                  "  st.shared.f32 [%rd5], %f"
               << sum << ";\n$fold_region_skip_pair_write" << active << ":\n";
    if (carry) {
      ptx.stream << "  @%p3 bra $fold_region_skip_carry_write" << active
                 << ";\n  st.shared.f32 [%rd3+"
                 << pairs * static_cast<int64_t>(sizeof(float)) << "], %f"
                 << *carry << ";\n$fold_region_skip_carry_write" << active
                 << ":\n";
    }
    ptx.stream << "  bar.sync 0;\n";
  }
  int64_t primaryValue = phaseResultRegisters.front();
  ptx.stream << "  ld.shared.f32 %f" << primaryValue << ", [%rd3];\n";
  context.foldValues[primary] = primaryValue;
  f32RegisterHighWater = std::max(f32RegisterHighWater, ptx.f32);
  ptx.f32 = phaseScratchBase;

  for (auto indexedFold :
       llvm::enumerate(ArrayRef<FoldOp>(phases).drop_front())) {
    const size_t phaseOrdinal = indexedFold.index();
    FoldOp fold = indexedFold.value();
    auto input = cast<RankedTensorType>(fold.getInputs().front().getType());
    int64_t cardinality = 1;
    for (int64_t axis : fold.getReductionDimensions())
      cardinality *= input.getDimSize(axis);
    if (cardinality != 1) {
      if (input != domain ||
          fold.getReductionDimensions() != primary.getReductionDimensions())
        return failure();
      const int64_t phase = phaseOrdinal + 1;
      int64_t phaseAccumulator = ptx.nextF32();
      ptx.stream << "  bar.sync 0;\n  mul.lo.u32 %r4, %r0, "
                 << task.getSerialTiles() << ";\n  add.u32 %r7, %r4, "
                 << task.getSerialTiles() << ";\n  min.u32 %r7, %r7, "
                 << *reductionExtent << ";\n  mov.f32 %f" << phaseAccumulator
                 << ", 0f00000000;\n$fold_region_phase_" << phase
                 << "_load:\n"
                    "  setp.ge.u32 %p1, %r4, %r7;\n"
                    "  @%p1 bra $fold_region_phase_"
                 << phase
                 << "_loaded;\n"
                    "  cvt.u64.u32 %rd0, %r4;\n"
                    "  cvt.u64.u32 %rd1, %r1;\n";
      if (fullPrefix)
        ptx.stream << "  mad.lo.u64 %rd0, %rd0, " << *outputExtent
                   << ", %rd1;\n";
      else
        ptx.stream << "  mad.lo.u64 %rd0, %rd1, " << *reductionExtent
                   << ", %rd0;\n";
      FailureOr<SmallVector<int64_t>> phaseCoordinates =
          emitOutputCoordinates(ptx, input);
      if (failed(phaseCoordinates))
        return failure();
      FailureOr<int64_t> phaseLeaf =
          emitLeaf(fold.getInputs().front(), *phaseCoordinates);
      if (failed(phaseLeaf))
        return failure();
      int64_t phaseCombined = ptx.nextF32();
      ptx.stream << "  add.rn.f32 %f" << phaseCombined << ", %f"
                 << phaseAccumulator << ", %f" << *phaseLeaf
                 << ";\n  mov.f32 %f" << phaseAccumulator << ", %f"
                 << phaseCombined
                 << ";\n  add.u32 %r4, %r4, 1;\n"
                    "  bra $fold_region_phase_"
                 << phase << "_load;\n$fold_region_phase_" << phase
                 << "_loaded:\n"
                    "  cvt.u64.u32 %rd2, %r0;\n"
                    "  mul.lo.u64 %rd2, %rd2, 4;\n"
                    "  add.u64 %rd2, %rd3, %rd2;\n"
                    "  st.shared.f32 [%rd2], %f"
                 << phaseAccumulator << ";\n  bar.sync 0;\n";
      for (int64_t active = task.getWorkgroupThreads(); active > 1;
           active = (active + 1) / 2) {
        int64_t pairs = active / 2;
        int64_t left = ptx.nextF32();
        int64_t right = ptx.nextF32();
        int64_t sum = ptx.nextF32();
        ptx.stream << "  setp.ge.u32 %p2, %r0, " << pairs
                   << ";\n  @%p2 bra $fold_region_phase_" << phase
                   << "_skip_pair_read" << active
                   << ";\n  mul.lo.u32 %r5, %r0, 2;\n"
                      "  cvt.u64.u32 %rd4, %r5;\n"
                      "  mul.lo.u64 %rd4, %rd4, 4;\n"
                      "  add.u64 %rd4, %rd3, %rd4;\n"
                      "  ld.shared.f32 %f"
                   << left << ", [%rd4];\n  ld.shared.f32 %f" << right
                   << ", [%rd4+4];\n  add.rn.f32 %f" << sum << ", %f" << left
                   << ", %f" << right << ";\n$fold_region_phase_" << phase
                   << "_skip_pair_read" << active << ":\n";
        std::optional<int64_t> carry;
        if (active % 2 != 0) {
          carry = ptx.nextF32();
          ptx.stream << "  setp.ne.u32 %p3, %r0, " << pairs
                     << ";\n  @%p3 bra $fold_region_phase_" << phase
                     << "_skip_carry_read" << active << ";\n  ld.shared.f32 %f"
                     << *carry << ", [%rd3+"
                     << (active - 1) * static_cast<int64_t>(sizeof(float))
                     << "];\n$fold_region_phase_" << phase << "_skip_carry_read"
                     << active << ":\n";
        }
        ptx.stream << "  bar.sync 0;\n"
                      "  @%p2 bra $fold_region_phase_"
                   << phase << "_skip_pair_write" << active
                   << ";\n  cvt.u64.u32 %rd5, %r0;\n"
                      "  mul.lo.u64 %rd5, %rd5, 4;\n"
                      "  add.u64 %rd5, %rd3, %rd5;\n"
                      "  st.shared.f32 [%rd5], %f"
                   << sum << ";\n$fold_region_phase_" << phase
                   << "_skip_pair_write" << active << ":\n";
        if (carry) {
          ptx.stream << "  @%p3 bra $fold_region_phase_" << phase
                     << "_skip_carry_write" << active
                     << ";\n  st.shared.f32 [%rd3+"
                     << pairs * static_cast<int64_t>(sizeof(float)) << "], %f"
                     << *carry << ";\n$fold_region_phase_" << phase
                     << "_skip_carry_write" << active << ":\n";
        }
        ptx.stream << "  bar.sync 0;\n";
      }
      int64_t phaseValue = phaseResultRegisters[phase];
      ptx.stream << "  ld.shared.f32 %f" << phaseValue << ", [%rd3];\n";
      context.foldValues[fold] = phaseValue;
      f32RegisterHighWater = std::max(f32RegisterHighWater, ptx.f32);
      ptx.f32 = phaseScratchBase;
      continue;
    }
    ptx.stream << "  cvt.u64.u32 %rd0, %r1;\n";
    FailureOr<SmallVector<int64_t>> coordinates =
        emitOutputCoordinates(ptx, input);
    if (failed(coordinates))
      return failure();
    FailureOr<int64_t> value = emitLeaf(fold.getInputs().front(), *coordinates);
    if (failed(value))
      return failure();
    int64_t initializer = ptx.nextF32();
    int64_t combined = ptx.nextF32();
    ptx.stream << "  mov.f32 %f" << initializer
               << ", 0f00000000;\n  add.rn.f32 %f" << combined << ", %f"
               << initializer << ", %f" << *value << ";\n  mov.f32 %f"
               << phaseResultRegisters[phaseOrdinal + 1] << ", %f" << combined
               << ";\n";
    context.foldValues[fold] = phaseResultRegisters[phaseOrdinal + 1];
    f32RegisterHighWater = std::max(f32RegisterHighWater, ptx.f32);
    ptx.f32 = phaseScratchBase;
  }

  const int64_t epilogueF32Base = phaseScratchBase;
  for (auto [outputOrdinal, outputOperation, outputBuffer] : llvm::enumerate(
           outputOperations, materializationTask.getOutputBuffers())) {
    ptx.f32 = epilogueF32Base;
    if (outputBuffer < 0 || outputBuffer >= buffers.size())
      return failure();
    auto output = cast<RankedTensorType>(buffers[outputBuffer].getTensorType());
    FailureOr<int64_t> outputElements = staticElementCount(output);
    if (failed(outputElements) || *outputElements % *outputExtent != 0)
      return failure();
    int64_t elementsPerBlock = *outputElements / *outputExtent;
    int64_t outputParameter = materializationTask.getInputBuffers().size() +
                              static_cast<int64_t>(outputOrdinal);
    if (auto outputFold = dyn_cast<FoldOp>(outputOperation)) {
      auto value = context.foldValues.find(outputFold);
      if (value == context.foldValues.end() || elementsPerBlock != 1 ||
          !output.getElementType().isF32())
        return failure();
      ptx.stream << "  setp.ne.u32 %p3, %r0, 0;\n"
                    "  @%p3 bra $fold_region_output_"
                 << outputOrdinal
                 << "_done;\n"
                    "  ld.param.u64 %rd6, [p"
                 << outputParameter
                 << "];\n  cvt.u64.u32 %rd7, %r1;\n"
                    "  mad.lo.u64 %rd6, %rd7, 4, %rd6;\n"
                    "  st.global.f32 [%rd6], %f"
                 << value->second << ";\n$fold_region_output_" << outputOrdinal
                 << "_done:\n";
      f32RegisterHighWater = std::max(f32RegisterHighWater, ptx.f32);
      continue;
    }
    auto outputMap = dyn_cast<MapOp>(outputOperation);
    if (!outputMap || outputMap.getResult(0).getType() != output)
      return failure();
    ptx.stream << "  mov.u32 %r6, %r0;\n$fold_region_output_" << outputOrdinal
               << "_epilogue:\n"
                  "  setp.ge.u32 %p3, %r6, "
               << elementsPerBlock << ";\n  @%p3 bra $fold_region_output_"
               << outputOrdinal
               << "_done;\n"
                  "  cvt.u64.u32 %rd0, %r1;\n"
                  "  cvt.u64.u32 %rd6, %r6;\n";
    if (row) {
      ptx.stream << "  mad.lo.u64 %rd0, %rd0, " << elementsPerBlock
                 << ", %rd6;\n";
    } else {
      ptx.stream << "  mad.lo.u64 %rd0, %rd6, " << *outputExtent << ", %rd0;\n";
    }
    FailureOr<SmallVector<int64_t>> coordinates =
        emitOutputCoordinates(ptx, output);
    if (failed(coordinates))
      return failure();
    FailureOr<int64_t> value =
        emitFoldRegionValue(context, outputMap.getResult(0), *coordinates);
    if (failed(value))
      return failure();
    int64_t pointer = ptx.nextU64();
    int64_t bytes = output.getElementType().isBF16() ? 2 : 4;
    ptx.stream << "  ld.param.u64 %rd" << pointer << ", [p" << outputParameter
               << "];\n  mad.lo.u64 %rd" << pointer << ", %rd0, " << bytes
               << ", %rd" << pointer << ";\n";
    if (output.getElementType().isBF16()) {
      int64_t rounded = ptx.nextBF16();
      ptx.stream << "  cvt.rn.bf16.f32 %h" << rounded << ", %f" << *value
                 << ";\n  st.global.u16 [%rd" << pointer << "], %h" << rounded
                 << ";\n";
    } else {
      ptx.stream << "  st.global.f32 [%rd" << pointer << "], %f" << *value
                 << ";\n";
    }
    ptx.stream << "  add.u32 %r6, %r6, " << task.getWorkgroupThreads()
               << ";\n  bra $fold_region_output_" << outputOrdinal
               << "_epilogue;\n$fold_region_output_" << outputOrdinal
               << "_done:\n";
    f32RegisterHighWater = std::max(f32RegisterHighWater, ptx.f32);
  }
  ptx.f32 = f32RegisterHighWater;
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitContractRegionPtx(
    MaterializationTaskOp materializationTask,
    SmallVectorImpl<MaterializationBufferOp> &buffers, ScheduleTaskOp task,
    llvm::SmallDenseMap<Attribute, Operation *> &sourceOperations) {
  auto sourceAttributes =
      materializationTask->getAttrOfType<ArrayAttr>("contract_region_sources");
  auto strategy = materializationTask->getAttrOfType<StringAttr>(
      "contract_region_strategy");
  auto cost = materializationTask->getAttrOfType<DictionaryAttr>(
      "contract_region_cost");
  auto foldSource = materializationTask->getAttrOfType<SourceRefAttr>(
      "contract_region_fold_source");
  auto contractSource = materializationTask->getAttrOfType<SourceRefAttr>(
      "contract_region_contract_source");
  auto boundarySource = materializationTask->getAttrOfType<SourceRefAttr>(
      "contract_region_bf16_boundary_source");
  auto epilogueSources = materializationTask->getAttrOfType<ArrayAttr>(
      "contract_region_epilogue_sources");
  ArrayRef<int64_t> outputBuffers = materializationTask.getOutputBuffers();
  const bool materializesNormalized = outputBuffers.size() == 2;
  if (!sourceAttributes || !strategy || !cost || !foldSource ||
      !contractSource || !boundarySource || !epilogueSources ||
      epilogueSources.size() != 1 ||
      (outputBuffers.size() != 1 && !materializesNormalized) ||
      task.getGridShape().size() != 2)
    return failure();
  auto rows = cost.getAs<IntegerAttr>("rows");
  auto reductionAttribute = cost.getAs<IntegerAttr>("reduction");
  auto columnsAttribute = cost.getAs<IntegerAttr>("columns");
  auto outputTileAttribute = cost.getAs<IntegerAttr>("output_tile");
  auto outputTilesAttribute = cost.getAs<IntegerAttr>("output_tiles");
  if (!rows || !reductionAttribute || !columnsAttribute ||
      !outputTileAttribute || !outputTilesAttribute)
    return failure();
  const int64_t reduction = reductionAttribute.getInt();
  const int64_t columns = columnsAttribute.getInt();
  const int64_t outputTile = outputTileAttribute.getInt();
  const int64_t outputTiles = outputTilesAttribute.getInt();
  const bool inlineFold = strategy.getValue() == "inline_fold";
  const bool compactStat = strategy.getValue() == "compact_stat";
  const bool fast =
      task.getReductionOrder() == ScheduleReductionOrder::LeafOrderFree;
  if ((!inlineFold && !compactStat) ||
      task.getGridShape() != ArrayRef<int64_t>({rows.getInt(), outputTiles}) ||
      task.getTileShape() != ArrayRef<int64_t>({outputTile}) ||
      task.getWorkgroupThreads() !=
          (fast ? outputTile * 32 : scheduleThreads(outputTile)) ||
      task.getSerialTiles() !=
          (fast ? ceilDivPositive(reduction, 32) : reduction))
    return failure();

  FoldOp fold = dyn_cast_or_null<FoldOp>(sourceOperations.lookup(foldSource));
  ContractOp contract =
      dyn_cast_or_null<ContractOp>(sourceOperations.lookup(contractSource));
  MapOp boundary =
      dyn_cast_or_null<MapOp>(sourceOperations.lookup(boundarySource));
  MapOp epilogue = dyn_cast_or_null<MapOp>(
      sourceOperations.lookup(epilogueSources.getValue().front()));
  if (!fold || !contract || !boundary || !epilogue ||
      !isRneF32ToBf16Map(boundary) || !isRneF32ToBf16Map(epilogue))
    return failure();
  MapOp reciprocal;
  for (const auto &[source, operation] : sourceOperations) {
    (void)source;
    auto map = dyn_cast<MapOp>(operation);
    bool hasReciprocal = false;
    if (map)
      map.walk([&](math::RsqrtOp) { hasReciprocal = true; });
    if (hasReciprocal && valueDependsOnOperation(map.getResult(0), fold)) {
      if (reciprocal)
        return failure();
      reciprocal = map;
    }
  }
  if (!reciprocal)
    return failure();
  SmallVector<Operation *> sources;
  llvm::SmallDenseSet<Operation *> sourceSet;
  for (Attribute source : sourceAttributes) {
    Operation *operation = sourceOperations.lookup(source);
    if (!operation || !sourceSet.insert(operation).second)
      return failure();
    sources.push_back(operation);
  }
  auto lhsType = cast<RankedTensorType>(boundary.getResult(0).getType());
  auto rhsType = cast<RankedTensorType>(contract.getInputs()[1].getType());
  auto resultType = cast<RankedTensorType>(contract.getResult(0).getType());
  if (llvm::any_of(outputBuffers, [&](int64_t ordinal) {
        return ordinal < 0 || ordinal >= static_cast<int64_t>(buffers.size());
      }))
    return failure();
  if (materializesNormalized &&
      buffers[outputBuffers.front()].getTensorType() != lhsType)
    return failure();
  if (buffers[outputBuffers.back()].getTensorType() !=
      cast<RankedTensorType>(epilogue.getResult(0).getType()))
    return failure();
  if (lhsType.getShape().back() != reduction ||
      rhsType.getShape() != ArrayRef<int64_t>({reduction, columns}))
    return failure();

  PtxEmitter ptx(materializationTask.getInputBuffers().size() +
                 outputBuffers.size());
  ptx.header(task.getSemanticFingerprint(), /*usesDynamicSharedMemory=*/true);
  FoldRegionPtxContext context{ptx,
                               materializationTask,
                               buffers,
                               sourceOperations,
                               DenseMap<Operation *, int64_t>(),
                               sourceSet,
                               DenseMap<Value, int64_t>(),
                               DenseMap<Value, int64_t>()};
  RegionOp owner = contract->getParentOfType<RegionOp>();
  if (!owner)
    return failure();
  for (BlockArgument argument : owner.getBody().front().getArguments()) {
    int64_t buffer = argument.getArgNumber();
    auto found = llvm::find(materializationTask.getInputBuffers(), buffer);
    if (found != materializationTask.getInputBuffers().end())
      context.boundaryParameters[argument] =
          std::distance(materializationTask.getInputBuffers().begin(), found);
  }
  if (compactStat)
    context.boundaryParameters[reciprocal.getResult(0)] = 1;

  ptx.stream << "  mov.u32 %r0, %tid.x;\n"
                "  mov.u32 %r1, %ctaid.x;\n"
                "  mov.u32 %r2, %ctaid.y;\n"
                "  mov.u64 %rd3, shuttle_smem;\n"
                "  cvta.to.shared.u64 %rd3, %rd3;\n";
  if (inlineFold) {
    const int64_t paddedReduction = scheduleThreads(reduction);
    if (task.getScratchBytes() != paddedReduction * 4)
      return failure();
    int64_t foldIndex = ptx.nextU64();
    ptx.stream << "  mov.u32 %r4, %r0;\n"
                  "$contract_region_fold_fill:\n"
                  "  setp.ge.u32 %p0, %r4, "
               << paddedReduction
               << ";\n  @%p0 bra $contract_region_fold_filled;\n"
                  "  setp.ge.u32 %p1, %r4, "
               << reduction
               << ";\n  @%p1 bra $contract_region_fold_zero;\n"
                  "  cvt.u64.u32 %rd0, %r1;\n"
                  "  cvt.u64.u32 %rd"
               << foldIndex << ", %r4;\n  mad.lo.u64 %rd0, %rd0, " << reduction
               << ", %rd" << foldIndex << ";\n";
    FailureOr<SmallVector<int64_t>> leafCoordinates = emitOutputCoordinates(
        ptx, cast<RankedTensorType>(fold.getInputs().front().getType()));
    if (failed(leafCoordinates))
      return failure();
    FailureOr<int64_t> leaf = emitFoldRegionValue(
        context, fold.getInputs().front(), *leafCoordinates);
    if (failed(leaf))
      return failure();
    ptx.stream << "  mov.f32 %f0, %f" << *leaf
               << ";\n  bra $contract_region_fold_store;\n"
                  "$contract_region_fold_zero:\n"
                  "  mov.f32 %f0, 0f00000000;\n"
                  "$contract_region_fold_store:\n"
                  "  cvt.u64.u32 %rd4, %r4;\n"
                  "  mad.lo.u64 %rd4, %rd4, 4, %rd3;\n"
                  "  st.shared.f32 [%rd4], %f0;\n"
                  "  add.u32 %r4, %r4, "
               << task.getWorkgroupThreads()
               << ";\n  bra $contract_region_fold_fill;\n"
                  "$contract_region_fold_filled:\n"
                  "  bar.sync 0;\n"
                  "  setp.ne.u32 %p0, %r0, 0;\n"
                  "  @%p0 bra $contract_region_scale_ready;\n";
    int64_t treeLeft = ptx.nextF32();
    int64_t treeRight = ptx.nextF32();
    int64_t treeSum = ptx.nextF32();
    int64_t treeCarry = ptx.nextF32();
    int64_t active = paddedReduction;
    while (active > 1) {
      const int64_t pairs = active / 2;
      for (int64_t pair = 0; pair < pairs; ++pair) {
        ptx.stream << "  ld.shared.f32 %f" << treeLeft << ", [%rd3+" << pair * 8
                   << "];\n  ld.shared.f32 %f" << treeRight << ", [%rd3+"
                   << pair * 8 + 4 << "];\n  add.rn.f32 %f" << treeSum << ", %f"
                   << treeLeft << ", %f" << treeRight
                   << ";\n  st.shared.f32 [%rd3+" << pair * 4 << "], %f"
                   << treeSum << ";\n";
      }
      if (active % 2 != 0) {
        ptx.stream << "  ld.shared.f32 %f" << treeCarry << ", [%rd3+"
                   << (active - 1) * 4 << "];\n  st.shared.f32 [%rd3+"
                   << pairs * 4 << "], %f" << treeCarry << ";\n";
      }
      active = (active + 1) / 2;
    }
    int64_t sum = ptx.nextF32();
    ptx.stream << "  ld.shared.f32 %f" << sum << ", [%rd3];\n";
    context.foldValues[fold] = sum;
    ptx.stream << "  cvt.u64.u32 %rd0, %r1;\n";
    FailureOr<SmallVector<int64_t>> scaleCoordinates = emitOutputCoordinates(
        ptx, cast<RankedTensorType>(reciprocal.getResult(0).getType()));
    if (failed(scaleCoordinates))
      return failure();
    FailureOr<int64_t> scale = emitFoldRegionValue(
        context, reciprocal.getResult(0), *scaleCoordinates);
    if (failed(scale))
      return failure();
    ptx.stream << "  st.shared.f32 [%rd3], %f" << *scale
               << ";\n$contract_region_scale_ready:\n"
                  "  bar.sync 0;\n";
    int64_t sharedScale = ptx.nextF32();
    ptx.stream << "  ld.shared.f32 %f" << sharedScale << ", [%rd3];\n";
    context.materializedValues[reciprocal.getResult(0)] = sharedScale;
  } else if (task.getScratchBytes() != std::min<int64_t>(reduction, 256) * 2) {
    return failure();
  }

  int64_t column = 0;
  int64_t lane = -1;
  if (fast) {
    ptx.stream << "  shr.u32 %r5, %r0, 5;\n"
                  "  and.b32 %r6, %r0, 31;\n"
                  "  mad.lo.u32 %r5, %r2, "
               << outputTile
               << ", %r5;\n"
                  "  setp.ge.u32 %p0, %r5, "
               << columns << ";\n  @%p0 bra $done;\n";
    column = 5;
    lane = 6;
  } else {
    ptx.stream << "  mad.lo.u32 %r5, %r2, " << outputTile
               << ", %r0;\n  setp.ge.u32 %p0, %r5, " << columns
               << ";\n  @%p0 bra $done;\n";
    column = 5;
  }
  int64_t outputLinear = ptx.nextU64();
  int64_t columnIndex = ptx.nextU64();
  ptx.stream << "  cvt.u64.u32 %rd" << outputLinear
             << ", %r1;\n  cvt.u64.u32 %rd" << columnIndex << ", %r" << column
             << ";\n  mad.lo.u64 %rd" << outputLinear << ", %rd" << outputLinear
             << ", " << columns << ", %rd" << columnIndex
             << ";\n  mov.u64 %rd0, %rd" << outputLinear << ";\n";
  FailureOr<SmallVector<int64_t>> outputCoordinates =
      emitOutputCoordinates(ptx, resultType);
  if (failed(outputCoordinates))
    return failure();
  int64_t accumulator = ptx.nextF32();
  int64_t reductionCoordinate = ptx.nextU64();
  ptx.stream << "  mov.f32 %f" << accumulator << ", 0f00000000;\n";
  if (fast)
    ptx.stream << "  cvt.u64.u32 %rd" << reductionCoordinate << ", %r" << lane
               << ";\n$contract_region_fast_loop:\n";
  else
    ptx.stream << "  mov.u64 %rd" << reductionCoordinate
               << ", 0;\n$contract_region_source_ordered_loop:\n";
  ptx.stream << "  setp.ge.u64 %p1, %rd" << reductionCoordinate << ", "
             << reduction << ";\n  @%p1 bra "
             << (fast ? "$contract_region_fast_reduce"
                      : "$contract_region_source_ordered_done")
             << ";\n"
                "  cvt.u64.u32 %rd0, %r1;\n"
                "  mad.lo.u64 %rd0, %rd0, "
             << reduction << ", %rd" << reductionCoordinate << ";\n";
  FailureOr<SmallVector<int64_t>> activationCoordinates =
      emitOutputCoordinates(ptx, lhsType);
  if (failed(activationCoordinates))
    return failure();
  FailureOr<int64_t> lhs = emitFoldRegionValue(context, boundary.getResult(0),
                                               *activationCoordinates);
  SmallVector<int64_t> contractCoordinates(*outputCoordinates);
  contractCoordinates.push_back(reductionCoordinate);
  auto rhsMap = cast<AffineMapAttr>(contract.getIndexingMaps()[1]).getValue();
  auto projectionParameter =
      context.boundaryParameters.find(contract.getInputs()[1]);
  if (failed(lhs) || projectionParameter == context.boundaryParameters.end())
    return failure();
  if (materializesNormalized) {
    const int64_t normalizedParameter =
        materializationTask.getInputBuffers().size();
    int64_t normalizedLinear = ptx.nextU64();
    int64_t normalizedPointer = ptx.nextU64();
    int64_t normalizedRounded = ptx.nextBF16();
    ptx.stream << "  setp.ne.u32 %p3, %r2, 0;\n"
                  "  @%p3 bra $contract_region_normalized_done;\n"
                  "  setp.ne.u32 %p3, %r5, 0;\n"
                  "  @%p3 bra $contract_region_normalized_done;\n"
                  "  cvt.u64.u32 %rd"
               << normalizedLinear << ", %r1;\n  mad.lo.u64 %rd"
               << normalizedLinear << ", %rd" << normalizedLinear << ", "
               << reduction << ", %rd" << reductionCoordinate
               << ";\n  ld.param.u64 %rd" << normalizedPointer << ", [p"
               << normalizedParameter << "];\n  mad.lo.u64 %rd"
               << normalizedPointer << ", %rd" << normalizedLinear << ", 2, %rd"
               << normalizedPointer << ";\n  cvt.rn.bf16.f32 %h"
               << normalizedRounded << ", %f" << *lhs
               << ";\n  st.global.u16 [%rd" << normalizedPointer << "], %h"
               << normalizedRounded << ";\n$contract_region_normalized_done:\n";
  }
  FailureOr<int64_t> rhs = emitMapInput(ptx, projectionParameter->second,
                                        rhsType, rhsMap, contractCoordinates);
  if (failed(rhs))
    return failure();
  ptx.stream << "  fma.rn.f32 %f" << accumulator << ", %f" << *lhs << ", %f"
             << *rhs << ", %f" << accumulator << ";\n  add.u64 %rd"
             << reductionCoordinate << ", %rd" << reductionCoordinate << ", "
             << (fast ? 32 : 1) << ";\n  bra "
             << (fast ? "$contract_region_fast_loop"
                      : "$contract_region_source_ordered_loop")
             << ";\n";
  if (fast) {
    ptx.stream << "$contract_region_fast_reduce:\n"
                  "  mov.b32 %r7, %f"
               << accumulator << ";\n";
    int64_t shuffled = ptx.nextF32();
    for (int64_t offset : {16, 8, 4, 2, 1}) {
      ptx.stream << "  shfl.sync.down.b32 %r8, %r7, " << offset
                 << ", 31, 0xffffffff;\n  mov.b32 %f" << shuffled
                 << ", %r8;\n  add.rn.f32 %f" << accumulator << ", %f"
                 << accumulator << ", %f" << shuffled << ";\n  mov.b32 %r7, %f"
                 << accumulator << ";\n";
    }
    ptx.stream << "  setp.ne.u32 %p2, %r6, 0;\n"
                  "  @%p2 bra $done;\n";
  } else {
    ptx.stream << "$contract_region_source_ordered_done:\n";
  }
  int64_t outputPointer = ptx.nextU64();
  int64_t rounded = ptx.nextBF16();
  const int64_t outputParameter = materializationTask.getInputBuffers().size() +
                                  (materializesNormalized ? 1 : 0);
  ptx.stream << "  ld.param.u64 %rd" << outputPointer << ", [p"
             << outputParameter << "];\n  mad.lo.u64 %rd" << outputPointer
             << ", %rd" << outputLinear << ", 2, %rd" << outputPointer
             << ";\n  cvt.rn.bf16.f32 %h" << rounded << ", %f" << accumulator
             << ";\n  st.global.u16 [%rd" << outputPointer << "], %h" << rounded
             << ";\n";
  if (ptx.f32 > 128 || ptx.u64 > 128 || ptx.bf16 > 16)
    return failure();
  ptx.finish();
  return ptx.text;
}

FailureOr<std::string> emitFoldPtx(FoldOp fold, ScheduleTaskOp task) {
  FailureOr<std::string> parallel = emitFoldMapPtx(fold, task, {}, {}, nullptr);
  if (succeeded(parallel))
    return parallel;
  if (fold.getInputs().size() != 1 || fold.getInitializers().size() != 1 ||
      fold.getNumResults() != 1 || fold.getReductionDimensions().size() != 1 ||
      task.getInputBuffers().size() != 2 || task.getOutputBuffers().size() != 1)
    return failure();
  const bool additive = llvm::any_of(
      fold.getCombiner().front().without_terminator(),
      [](Operation &operation) { return isa<arith::AddFOp>(operation); });
  if (additive && !hasExactPositiveZeroInitializer(fold))
    return failure();
  auto input = dyn_cast<RankedTensorType>(fold.getInputs()[0].getType());
  auto initializer =
      dyn_cast<RankedTensorType>(fold.getInitializers()[0].getType());
  auto output = dyn_cast<RankedTensorType>(fold.getResult(0).getType());
  const int64_t reductionAxis = fold.getReductionDimensions().front();
  if (!input || !initializer || !output || !input.hasStaticShape() ||
      !output.hasStaticShape() || !input.getElementType().isF32() ||
      !initializer.getElementType().isF32() || initializer.getRank() != 0 ||
      !output.getElementType().isF32() || reductionAxis < 0 ||
      reductionAxis >= input.getRank())
    return failure();
  FailureOr<int64_t> outputElements = staticElementCount(output);
  if (failed(outputElements) ||
      task.getGridShape() != ArrayRef<int64_t>({*outputElements}))
    return failure();
  PtxEmitter ptx(/*arity=*/3);
  ptx.header(task.getSemanticFingerprint());
  ptx.stream << "  mov.u32 %r0, %tid.x;\n  setp.ne.u32 %p0, %r0, 0;\n"
                "  @%p0 bra $done;\n  mov.u32 %r1, %ctaid.x;\n"
                "  cvt.u64.u32 %rd0, %r1;\n";
  FailureOr<SmallVector<int64_t>> outputCoordinates =
      emitOutputCoordinates(ptx, output);
  if (failed(outputCoordinates))
    return failure();
  int64_t inputBase = ptx.nextU64();
  int64_t initializerBase = ptx.nextU64();
  int64_t accumulator = ptx.nextF32();
  ptx.stream << "  ld.param.u64 %rd" << inputBase
             << ", [p0];\n  ld.param.u64 %rd" << initializerBase
             << ", [p1];\n  ld.global.f32 %f" << accumulator << ", [%rd"
             << initializerBase << "];\n";
  int64_t reduction = ptx.nextU64();
  ptx.stream << "  mov.u64 %rd" << reduction
             << ", 0;\n$generic_fold_loop:\n  setp.ge.u64 %p1, %rd" << reduction
             << ", " << input.getDimSize(reductionAxis)
             << ";\n  @%p1 bra $generic_fold_done;\n";
  SmallVector<int64_t> inputCoordinates;
  int64_t outputAxis = 0;
  for (int64_t axis = 0; axis < input.getRank(); ++axis)
    inputCoordinates.push_back(
        axis == reductionAxis ? reduction : (*outputCoordinates)[outputAxis++]);
  FailureOr<int64_t> inputLinear =
      emitRowMajorLinearIndex(ptx, inputCoordinates, input.getShape());
  if (failed(inputLinear))
    return failure();
  int64_t inputPointer = ptx.nextU64();
  int64_t value = ptx.nextF32();
  ptx.stream << "  mad.lo.u64 %rd" << inputPointer << ", %rd" << *inputLinear
             << ", 4, %rd" << inputBase << ";\n  ld.global.f32 %f" << value
             << ", [%rd" << inputPointer << "];\n";
  FailureOr<int64_t> combined =
      emitScalarBody(ptx, fold.getCombiner().front(), {accumulator, value});
  if (failed(combined))
    return failure();
  ptx.stream << "  mov.f32 %f" << accumulator << ", %f" << *combined
             << ";\n  add.u64 %rd" << reduction << ", %rd" << reduction
             << ", 1;\n  bra $generic_fold_loop;\n$generic_fold_done:\n";
  int64_t outputPointer = ptx.nextU64();
  ptx.stream << "  ld.param.u64 %rd" << outputPointer
             << ", [p2];\n  mad.lo.u64 %rd" << outputPointer << ", %rd0, 4, %rd"
             << outputPointer << ";\n  st.global.f32 [%rd" << outputPointer
             << "], %f" << accumulator << ";\n";
  if (ptx.f32 > 128 || ptx.u64 > 128)
    return failure();
  ptx.finish();
  return ptx.text;
}

#include "OnlineAttentionEmitter.inc"
FailureOr<GpuExecutableSpec>
deriveGpuExecutable(RegionOp region, MaterializationPlanOp materialization,
                    SchedulePlanOp schedule, GpuTarget target) {
  SmallVector<Operation *> algebra;
  for (Operation &operation : region.getBody().front().without_terminator())
    algebra.push_back(&operation);
  llvm::SmallDenseMap<Attribute, Operation *> sourceOperations;
  for (Operation *operation : algebra) {
    if (auto map = dyn_cast<MapOp>(operation))
      sourceOperations.try_emplace(map.getSource(), operation);
    else if (auto fold = dyn_cast<FoldOp>(operation))
      sourceOperations.try_emplace(fold.getSource(), operation);
    else if (auto contract = dyn_cast<ContractOp>(operation))
      sourceOperations.try_emplace(contract.getSource(), operation);
    else if (auto relation = dyn_cast<RelationOp>(operation))
      sourceOperations.try_emplace(relation.getSource(), operation);
    else if (auto selection = dyn_cast<SelectionOp>(operation))
      for (Attribute source : selection.getSources())
        sourceOperations.try_emplace(source, operation);
  }
  SmallVector<MaterializationBufferOp> materializationBuffers(
      materialization.getBody().front().getOps<MaterializationBufferOp>());
  SmallVector<MaterializationTaskOp> materializationTasks(
      materialization.getBody().front().getOps<MaterializationTaskOp>());
  SmallVector<ScheduleTaskOp> tasks(
      schedule.getBody().front().getOps<ScheduleTaskOp>());
  if (algebra.empty()) {
    region.emitError("GPU executable region has no algebra operations");
    return failure();
  }
  if (materializationTasks.size() != tasks.size()) {
    schedule.emitError(
        "GPU executable task count does not match materialization: ")
        << tasks.size() << " schedules versus " << materializationTasks.size()
        << " materialization tasks";
    return failure();
  }
  if (tasks.size() > kMaximumGpuExecutableRecords) {
    schedule.emitError(
        "GPU executable task count exceeds bounded record capacity: ")
        << tasks.size() << " > " << kMaximumGpuExecutableRecords;
    return failure();
  }

  GpuExecutableSpec spec;
  FailureOr<SmallVector<ExecutableSlotSpec>> slots =
      deriveExecutableSlots(materialization);
  if (failed(slots)) {
    materialization.emitError("cannot derive GPU executable buffer slots");
    return failure();
  }
  if (slots->empty() || slots->size() > kMaximumGpuExecutableRecords) {
    materialization.emitError(
        "GPU executable buffer-slot count is outside bounded capacity: ")
        << slots->size() << " (maximum " << kMaximumGpuExecutableRecords << ")";
    return failure();
  }
  spec.slots = std::move(*slots);
  spec.sourceBuffers.resize(spec.slots.size());
  std::iota(spec.sourceBuffers.begin(), spec.sourceBuffers.end(), 0);
  FailureOr<SmallVector<int64_t>> reuseGroups =
      deriveGpuReuseGroups(materialization, spec.slots);
  if (failed(reuseGroups)) {
    materialization.emitError("cannot derive GPU temporary reuse groups");
    return failure();
  }
  spec.reuseGroups = std::move(*reuseGroups);
  auto manifest =
      region->getParentOfType<ModuleOp>()->getAttrOfType<DictionaryAttr>(
          kCoverageManifestAttribute);
  auto tuning =
      manifest ? manifest.getAs<StringAttr>("canonical_tuning") : StringAttr{};
  FailureOr<DerivedPlan> derived = deriveGpuPlan(
      region, tuning && tuning.getValue() == kFusionCanonicalTuning);
  if (failed(derived)) {
    region.emitError("cannot rederive the generic GPU materialization plan");
    return failure();
  }
  if (derived->buffers.size() != spec.slots.size()) {
    materialization.emitError(
        "GPU source-plan buffer count does not match physical slots: ")
        << derived->buffers.size() << " source buffers versus "
        << spec.slots.size() << " slots";
    return failure();
  }
  FailureOr<SmallVector<std::optional<int64_t>>> bindingIndices =
      deriveGpuBindingIndices(region, *derived, spec.slots);
  if (failed(bindingIndices)) {
    materialization.emitError("cannot derive GPU public ABI binding indices");
    return failure();
  }
  spec.bindingIndices = std::move(*bindingIndices);
  for (auto [ordinal, materializationTask, task] :
       llvm::enumerate(materializationTasks, tasks)) {
    Operation *operation =
        sourceOperations.lookup(materializationTask.getSource());
    if (!operation) {
      materializationTask.emitOpError(
          "has no source algebra operation for GPU emission");
      return failure();
    }
    FailureOr<std::string> ptx = failure();
    auto compositionKind =
        materializationTask->getAttrOfType<StringAttr>("composition_kind");
    if (compositionKind && compositionKind.getValue() == "contract_fold") {
      ptx = emitContractFoldCompositionPtx(
          materializationTask, materializationBuffers, task, sourceOperations);
    } else if (compositionKind &&
               compositionKind.getValue() == "map_contract") {
      ptx = emitMapContractCompositionPtx(
          materializationTask, materializationBuffers, task, sourceOperations);
    } else if (compositionKind &&
               compositionKind.getValue() == "indexed_accumulation") {
      ptx = emitIndexedAccumulationPtx(region, materializationTask,
                                       materializationBuffers, task);
    } else if (compositionKind &&
               compositionKind.getValue() == "integer_relation") {
      SmallVector<RelationOp> relations;
      auto sources =
          materializationTask->getAttrOfType<ArrayAttr>("composition_sources");
      if (!sources)
        return failure();
      for (Attribute source : sources) {
        auto relation =
            dyn_cast_or_null<RelationOp>(sourceOperations.lookup(source));
        if (!relation)
          return failure();
        relations.push_back(relation);
      }
      ptx = emitIntegerRelationCompositionPtx(relations, task);
    } else if (compositionKind &&
               compositionKind.getValue() == "relation_gather") {
      SmallVector<RelationOp> relations;
      auto sources =
          materializationTask->getAttrOfType<ArrayAttr>("composition_sources");
      if (!sources)
        return failure();
      for (Attribute source : sources) {
        auto relation =
            dyn_cast_or_null<RelationOp>(sourceOperations.lookup(source));
        if (!relation)
          return failure();
        relations.push_back(relation);
      }
      ptx = emitRelationGatherCompositionPtx(relations, materializationTask,
                                             materializationBuffers, task);
    } else if (isOnlineStateTask(materializationTask)) {
      ptx = emitOnlineSufficientStatePtx(region, materializationTask,
                                         materializationBuffers, task);
    } else if (compositionKind) {
      materializationTask.emitOpError(
          "GPU fallback does not support generic composition kind '")
          << compositionKind.getValue() << "'";
      return failure();
    } else if (materializationTask->getAttr("contract_region_sources")) {
      ptx = emitContractRegionPtx(materializationTask, materializationBuffers,
                                  task, sourceOperations);
    } else if (materializationTask->getAttr("fold_region_sources")) {
      ptx = emitFoldRegionPtx(materializationTask, materializationBuffers, task,
                              sourceOperations);
    } else if (auto epilogueSources =
                   materializationTask->getAttrOfType<ArrayAttr>(
                       "fold_epilogue_sources")) {
      auto fold = dyn_cast<FoldOp>(operation);
      SmallVector<MapOp> maps;
      for (Attribute source : epilogueSources) {
        auto map = dyn_cast_or_null<MapOp>(sourceOperations.lookup(source));
        if (!map) {
          materializationTask.emitOpError(
              "Fold epilogue source is not a scalar Map");
          return failure();
        }
        maps.push_back(map);
      }
      if (!fold) {
        materializationTask.emitOpError(
            "Fold epilogue task source is not a Fold");
        return failure();
      }
      ptx = emitFoldMapPtx(fold, task, maps, materializationTask,
                           &materializationBuffers);
    } else if (auto fusedSources =
                   materializationTask->getAttrOfType<ArrayAttr>(
                       "fused_sources")) {
      SmallVector<MapOp> maps;
      for (Attribute source : fusedSources) {
        auto map = dyn_cast_or_null<MapOp>(sourceOperations.lookup(source));
        if (!map) {
          materializationTask.emitOpError("fused source is not a scalar Map");
          return failure();
        }
        maps.push_back(map);
      }
      ptx = emitFusedMapPtx(maps, materializationTask, materializationBuffers,
                            task);
    } else if (auto map = dyn_cast<MapOp>(operation)) {
      ptx = emitMapPtx(map, task);
    } else if (auto fold = dyn_cast<FoldOp>(operation)) {
      ptx = emitFoldPtx(fold, task);
    } else if (auto contract = dyn_cast<ContractOp>(operation)) {
      ptx = emitContractPtx(contract, task);
    } else if (auto relation = dyn_cast<RelationOp>(operation)) {
      if (relation.getInputs().empty())
        ptx = emitGeneratedRelationPtx(relation, task);
      else if (relation.getInputs().size() == 1)
        ptx = emitSourceRelationPtx(relation, task);
      else if (relation.getSemantics() == RelationSemantics::Gather)
        ptx = emitGatherRelationPtx(relation, task);
      else if (relation.getSemantics() == RelationSemantics::Select)
        ptx = emitIntegerSelectRelationPtx(relation, task);
      else
        ptx = emitElementwiseIntegerRelationPtx(relation, task);
    } else if (auto selection = dyn_cast<SelectionOp>(operation)) {
      ptx = emitSelectionPtx(selection, task);
    }
    if (failed(ptx)) {
      operation->emitOpError("cannot emit bounded generic CUDA PTX");
      return failure();
    }
    if (failed(retargetPtx(*ptx, target))) {
      operation->emitOpError("cannot bind generic CUDA PTX to its target");
      return failure();
    }
    const bool contractRegion =
        materializationTask->hasAttr("contract_region_sources");
    const size_t expectedGridRank = contractRegion ? 2 : 1;
    if (ptx->empty() || !StringRef(*ptx).ends_with("\n") ||
        ptx->size() > kMaximumGpuEntryCodeBytes ||
        task.getGridShape().size() != expectedGridRank ||
        llvm::any_of(task.getGridShape(),
                     [](int64_t extent) {
                       return extent <= 0 || extent > UINT32_MAX;
                     }) ||
        task.getWorkgroupThreads() <= 0 || task.getWorkgroupThreads() > 1024 ||
        task.getScratchBytes() < 0 ||
        task.getScratchBytes() > kMaximumGpuDynamicSharedMemoryBytes) {
      task.emitOpError("has invalid bounded GPU code or launch resources");
      return failure();
    }
    int64_t offset = spec.code.size();
    spec.code.append(ptx->begin(), ptx->end());
    SmallVector<int64_t> inputs(task.getInputBuffers());
    SmallVector<int64_t> outputs(task.getOutputBuffers());
    SmallVector<int64_t> grid(task.getGridShape());
    while (grid.size() < 3)
      grid.push_back(1);
    std::array<int64_t, 3> launchGrid = {grid[0], grid[1], grid[2]};
    FailureOr<SmallVector<AffineMap>> logicalMaps =
        isOnlineStateTask(materializationTask)
            ? onlineLogicalCoordinateMaps(task)
            : FailureOr<SmallVector<AffineMap>>(SmallVector<AffineMap>{});
    if (failed(logicalMaps)) {
      task.emitOpError("cannot derive logical coordinate maps for GPU entry");
      return failure();
    }
    spec.entries.push_back(
        {static_cast<int64_t>(ordinal),
         offset,
         static_cast<int64_t>(ptx->size()),
         std::move(inputs),
         std::move(outputs),
         SmallVector<int64_t>(task.getDependencies()),
         task.getKind() == ScheduleTaskKind::Scalar
             ? ExecutablePredication::None
             : ExecutablePredication::DomainBounds,
         (isOnlineStateTask(materializationTask) ||
          isReverseStateTask(materializationTask))
             ? task.getReductionOrder()
         : isa<ContractOp>(operation) ? std::nullopt
                                      : task.getReductionOrder(),
         launchGrid,
         {static_cast<int64_t>(task.getWorkgroupThreads()), 1, 1},
         static_cast<int64_t>(task.getScratchBytes()),
         static_cast<int64_t>(task.getInputBuffers().size() +
                              task.getOutputBuffers().size()),
         (isOnlineStateTask(materializationTask) ||
          isReverseStateTask(materializationTask))
             ? std::optional<std::string>("stateful_contract_fold")
             : std::nullopt,
         std::move(*logicalMaps),
         sha256(*ptx)});
  }
  if (spec.code.size() > kMaximumGpuCodeBytes) {
    schedule.emitError("aggregate GPU code exceeds bounded byte capacity: ")
        << spec.code.size() << " > " << kMaximumGpuCodeBytes;
    return failure();
  }
  return spec;
}

ArrayAttr accessArray(MLIRContext *context, ExecutableAccess access,
                      size_t count) {
  SmallVector<Attribute> values(count,
                                ExecutableAccessAttr::get(context, access));
  return ArrayAttr::get(context, values);
}

SchedulePlanOp
uniqueScheduleForMaterialization(ArrayRef<SchedulePlanOp> schedules,
                                 MaterializationPlanOp materialization) {
  SchedulePlanOp match;
  for (SchedulePlanOp schedule : schedules) {
    if (schedule.getSourcePlanFingerprint() != materialization.getFingerprint())
      continue;
    if (match)
      return {};
    match = schedule;
  }
  return match;
}

EventLoweringPlanOp
uniqueSourceEventPlanForSchedule(ArrayRef<EventLoweringPlanOp> eventPlans,
                                 SchedulePlanOp schedule) {
  EventLoweringPlanOp match;
  for (EventLoweringPlanOp eventPlan : eventPlans) {
    if (eventPlan.getSourceScheduleFingerprint() != schedule.getFingerprint() ||
        eventPlan.getPhysicalLaunchRecipeFingerprint())
      continue;
    if (match)
      return {};
    match = eventPlan;
  }
  return match;
}
