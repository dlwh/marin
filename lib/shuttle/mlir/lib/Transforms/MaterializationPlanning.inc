// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

std::string attributeText(Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return text;
}

std::string normalizedModuleFingerprintImpl(ModuleOp module) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  for (NamedAttribute attribute : module->getAttrs()) {
    if (attribute.getName().strref() == SymbolTable::getSymbolAttrName()) {
      continue;
    }
    stream << "module_attr:" << attribute.getName() << '=';
    attribute.getValue().print(stream);
    stream << '\n';
  }
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    stream << "function:";
    function.getFunctionType().print(stream);
    stream << '\n';
    llvm::SmallDenseMap<Value, uint64_t> values;
    uint64_t nextValue = 0;
    for (BlockArgument argument : function.getArguments()) {
      values.try_emplace(argument, nextValue++);
    }
    std::function<void(Operation &)> printOperation =
        [&](Operation &operation) {
          stream << operation.getName() << '(';
          for (Value operand : operation.getOperands()) {
            auto position = values.find(operand);
            if (position == values.end()) {
              stream << "missing";
            } else {
              stream << position->second;
            }
            stream << ',';
          }
          stream << ")->(";
          for (Type type : operation.getResultTypes()) {
            type.print(stream);
            stream << ',';
          }
          stream << "){";
          for (NamedAttribute attribute : operation.getAttrs()) {
            StringRef name = attribute.getName().strref();
            if (name == SymbolTable::getSymbolAttrName() ||
                name == kSourceRefsAttribute || name == kSelectedAttribute ||
                name == kFunctionOrdinalAttribute ||
                name == kOperationRefAttribute ||
                name == kCoverageManifestAttribute ||
                name == kRegionResultSourcesAttribute) {
              continue;
            }
            stream << name << '=';
            attribute.getValue().print(stream);
            stream << ',';
          }
          stream << "}\n";
          for (Value result : operation.getResults()) {
            values.try_emplace(result, nextValue++);
          }
          for (auto [regionOrdinal, region] :
               llvm::enumerate(operation.getRegions())) {
            stream << "region:" << regionOrdinal << '\n';
            for (auto [blockOrdinal, block] : llvm::enumerate(region)) {
              stream << "block:" << blockOrdinal << '(';
              for (BlockArgument argument : block.getArguments()) {
                argument.getType().print(stream);
                stream << ':' << nextValue << ',';
                values.try_emplace(argument, nextValue++);
              }
              stream << ")\n";
              for (Operation &nested : block) {
                printOperation(nested);
              }
            }
          }
        };
    for (Operation &operation : function.getBody().front()) {
      printOperation(operation);
    }
  }
  stream.flush();
  return sha256(normalized);
}

std::string generatedGpuBundleFingerprintImpl(ModuleOp module) {
  SmallVector<DeviceModuleOp> devices(module.getOps<DeviceModuleOp>());
  SmallVector<InvocationAbiOp> abis(module.getOps<InvocationAbiOp>());
  SmallVector<ExecutableBundleOp> bundles(module.getOps<ExecutableBundleOp>());
  if (bundles.empty() || devices.size() != bundles.size() ||
      abis.size() != bundles.size())
    return {};
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  llvm::SmallDenseSet<Operation *> matchedDevices;
  llvm::SmallDenseSet<Operation *> matchedAbis;
  stream << "bundle_count:" << bundles.size() << '\n';
  for (auto [ordinal, bundle] : llvm::enumerate(bundles)) {
    DeviceModuleOp device;
    for (DeviceModuleOp candidate : devices) {
      if (candidate.getFingerprint() != bundle.getDeviceModuleFingerprint())
        continue;
      if (device)
        return {};
      device = candidate;
    }
    InvocationAbiOp abi;
    for (InvocationAbiOp candidate : abis) {
      if (candidate.getFingerprint() != bundle.getInvocationAbiFingerprint())
        continue;
      if (abi)
        return {};
      abi = candidate;
    }
    if (!device || !abi ||
        !matchedDevices.insert(device.getOperation()).second ||
        !matchedAbis.insert(abi.getOperation()).second)
      return {};
    stream << "tuple:" << ordinal
           << "\ndevice_module:" << deviceModuleFingerprint(device)
           << "\ninvocation_abi:" << invocationAbiFingerprint(abi)
           << "\nexecutable_bundle:" << executableBundleFingerprint(bundle)
           << '\n';
  }
  stream.flush();
  return sha256(normalized);
}

bool isMaterializationFingerprintProvenance(StringRef name) {
  return name == kSourceRefsAttribute || name == kSelectedAttribute ||
         name == kFunctionOrdinalAttribute || name == kOperationRefAttribute ||
         name == kCoverageManifestAttribute ||
         name == kRegionResultSourcesAttribute ||
         name == kScanResultSourcesAttribute ||
         name == kScanErasedTerminatorRecordsAttribute || name == "source" ||
         name == "sources" || name == "source_group";
}

std::string semanticTaskFingerprint(Operation *root) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  llvm::SmallDenseMap<Value, uint64_t> values;
  uint64_t nextValue = 0;
  for (Value operand : root->getOperands()) {
    values.try_emplace(operand, nextValue++);
  }
  std::function<void(Operation &)> printOperation = [&](Operation &operation) {
    stream << operation.getName() << '(';
    for (Value operand : operation.getOperands()) {
      auto position = values.find(operand);
      stream << (position == values.end() ? UINT64_MAX : position->second)
             << ',';
    }
    stream << ")->(";
    for (Type type : operation.getResultTypes()) {
      type.print(stream);
      stream << ',';
    }
    stream << "){";
    for (NamedAttribute attribute : operation.getAttrs()) {
      if (isMaterializationFingerprintProvenance(
              attribute.getName().strref())) {
        continue;
      }
      stream << attribute.getName() << '=';
      attribute.getValue().print(stream);
      stream << ',';
    }
    stream << "}\n";
    for (Value result : operation.getResults()) {
      values.try_emplace(result, nextValue++);
    }
    for (Region &region : operation.getRegions()) {
      for (Block &block : region) {
        stream << "block(";
        for (BlockArgument argument : block.getArguments()) {
          argument.getType().print(stream);
          stream << ':' << nextValue << ',';
          values.try_emplace(argument, nextValue++);
        }
        stream << ")\n";
        for (Operation &nested : block) {
          printOperation(nested);
        }
      }
    }
  };
  printOperation(*root);
  stream.flush();
  return sha256(normalized);
}

FailureOr<SmallVector<int64_t>> mapDomainShape(MapOp map) {
  if (map.getNumResults() != 1) {
    return failure();
  }
  auto resultType = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() ||
      llvm::any_of(resultType.getShape(),
                   [](int64_t extent) { return extent <= 0; })) {
    return failure();
  }
  AffineMap resultMap =
      cast<AffineMapAttr>(
          map.getIndexingMaps()[map.getIndexingMaps().size() - 1])
          .getValue();
  SmallVector<int64_t> domain(resultMap.getNumDims(), ShapedType::kDynamic);
  for (auto [resultPosition, expression] :
       llvm::enumerate(resultMap.getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension) {
      return failure();
    }
    domain[dimension.getPosition()] = resultType.getDimSize(resultPosition);
  }
  if (llvm::is_contained(domain, ShapedType::kDynamic)) {
    return failure();
  }
  return domain;
}

FailureOr<SmallVector<int64_t>> genericMapDomainShape(MapOp map) {
  if (map.getNumResults() == 0 ||
      map.getIndexingMaps().size() !=
          map.getInputs().size() + map.getNumResults()) {
    return failure();
  }
  std::optional<SmallVector<int64_t>> commonDomain;
  ArrayRef<Attribute> resultMaps =
      map.getIndexingMaps().getValue().drop_front(map.getInputs().size());
  for (auto [result, mapAttribute] :
       llvm::zip_equal(map.getResults(), resultMaps)) {
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    auto resultMap = dyn_cast<AffineMapAttr>(mapAttribute);
    if (!resultType || !resultType.hasStaticShape() || !resultMap ||
        resultMap.getValue().getNumSymbols() != 0 ||
        resultMap.getValue().getNumResults() != resultType.getRank() ||
        llvm::any_of(resultType.getShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      return failure();
    }
    SmallVector<int64_t> domain(resultMap.getValue().getNumDims(),
                                ShapedType::kDynamic);
    for (auto [resultPosition, expression] :
         llvm::enumerate(resultMap.getValue().getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension ||
          domain[dimension.getPosition()] != ShapedType::kDynamic) {
        return failure();
      }
      domain[dimension.getPosition()] = resultType.getDimSize(resultPosition);
    }
    if (llvm::is_contained(domain, ShapedType::kDynamic) ||
        (commonDomain && *commonDomain != domain)) {
      return failure();
    }
    commonDomain = std::move(domain);
  }
  return std::move(*commonDomain);
}

FailureOr<SmallVector<int64_t>> contractDomainShape(ContractOp contract) {
  if (contract.getInputs().size() != 2 || contract.getNumResults() != 1 ||
      contract.getIndexingMaps().size() != 3) {
    return failure();
  }
  AffineMap first =
      cast<AffineMapAttr>(contract.getIndexingMaps()[0]).getValue();
  SmallVector<int64_t> domain(first.getNumDims(), ShapedType::kDynamic);
  SmallVector<Type> indexedTypes;
  llvm::append_range(indexedTypes, contract.getInputs().getTypes());
  llvm::append_range(indexedTypes, contract.getResults().getTypes());
  for (auto [mapAttribute, type] :
       llvm::zip_equal(contract.getIndexingMaps(), indexedTypes)) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    auto map = dyn_cast<AffineMapAttr>(mapAttribute);
    if (!tensor || !tensor.hasStaticShape() || !map ||
        map.getValue().getNumDims() != domain.size() ||
        map.getValue().getNumResults() != tensor.getRank()) {
      return failure();
    }
    for (auto [axis, expression] :
         llvm::enumerate(map.getValue().getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension || tensor.getDimSize(axis) <= 0) {
        return failure();
      }
      int64_t &extent = domain[dimension.getPosition()];
      if (!ShapedType::isDynamic(extent) && extent != tensor.getDimSize(axis)) {
        return failure();
      }
      extent = tensor.getDimSize(axis);
    }
  }
  if (llvm::is_contained(domain, ShapedType::kDynamic)) {
    return failure();
  }
  return domain;
}

bool isClosedProjectedContract(ContractOp contract,
                               bool allowMultipleReductions = false) {
  auto lhs = contract.getInputs().size() == 2
                 ? dyn_cast<RankedTensorType>(contract.getInputs()[0].getType())
                 : RankedTensorType{};
  auto rhs = contract.getInputs().size() == 2
                 ? dyn_cast<RankedTensorType>(contract.getInputs()[1].getType())
                 : RankedTensorType{};
  auto result =
      contract.getNumResults() == 1
          ? dyn_cast<RankedTensorType>(contract.getResult(0).getType())
          : RankedTensorType{};
  FailureOr<SmallVector<int64_t>> domain = contractDomainShape(contract);
  constexpr int64_t kMaximumContractRank = 8;
  if (!lhs || !rhs || !result || failed(domain) || lhs.getRank() < 2 ||
      rhs.getRank() < 2 || result.getRank() < 2 ||
      lhs.getRank() > kMaximumContractRank ||
      rhs.getRank() > kMaximumContractRank ||
      result.getRank() > kMaximumContractRank ||
      lhs.getElementType() != rhs.getElementType() ||
      (!lhs.getElementType().isBF16() && !lhs.getElementType().isF32()) ||
      !result.getElementType().isF32() ||
      contract.getAlgorithm() != "dot_general" ||
      contract.getAccumulatorTypes().size() != 1 ||
      cast<TypeAttr>(contract.getAccumulatorTypes()[0]).getValue() !=
          Float32Type::get(contract.getContext()) ||
      contract.getPrecision().size() != 2 ||
      llvm::any_of(contract.getPrecision(), [](Attribute precision) {
        auto value = dyn_cast<StringAttr>(precision);
        return !value || value.getValue() != "DEFAULT";
      })) {
    return false;
  }
  SmallVector<int64_t> reductions;
  for (auto [position, iterator] :
       llvm::enumerate(contract.getIteratorKinds())) {
    auto value = dyn_cast<StringAttr>(iterator);
    if (!value) {
      return false;
    }
    if (value.getValue() == "reduction") {
      if (!allowMultipleReductions && !reductions.empty()) {
        return false;
      }
      reductions.push_back(position);
    } else if (value.getValue() != "parallel") {
      return false;
    }
  }
  if (reductions.empty() ||
      domain->size() != result.getRank() + reductions.size() ||
      reductions.back() != static_cast<int64_t>(domain->size()) - 1 ||
      !llvm::all_of(llvm::enumerate(reductions), [&](auto item) {
        return item.value() ==
               static_cast<int64_t>(domain->size() - reductions.size() +
                                    item.index());
      })) {
    return false;
  }
  AffineMap lhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[0]).getValue();
  AffineMap rhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[1]).getValue();
  AffineMap resultMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[2]).getValue();
  MLIRContext *context = contract.getContext();
  SmallVector<AffineExpr> expectedResultExpressions;
  expectedResultExpressions.reserve(result.getRank());
  for (int64_t dimension = 0; dimension < result.getRank(); ++dimension) {
    expectedResultExpressions.push_back(getAffineDimExpr(dimension, context));
  }
  if (resultMap !=
      AffineMap::get(domain->size(), 0, expectedResultExpressions, context)) {
    return false;
  }
  for (unsigned dimension = 0; dimension < domain->size(); ++dimension) {
    const bool inLhs = succeeded(mapResultPosition(lhsMap, dimension));
    const bool inRhs = succeeded(mapResultPosition(rhsMap, dimension));
    const bool inResult = succeeded(mapResultPosition(resultMap, dimension));
    if (llvm::is_contained(reductions, dimension)) {
      if (!inLhs || !inRhs || inResult) {
        return false;
      }
    } else if ((!inLhs && !inRhs) || !inResult) {
      return false;
    }
  }
  return true;
}

bool isClosedContractMapRegion(RegionOp region, ArrayRef<Operation *> algebra) {
  auto isExternalType = [](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    return tensor && tensor.hasStaticShape() && tensor.getRank() >= 2 &&
           tensor.getRank() <= 8 && tensor.getElementType().isBF16() &&
           llvm::all_of(tensor.getShape(),
                        [](int64_t extent) { return extent > 0; });
  };
  if (!llvm::all_of(
          region.getInputs(),
          [&](Value input) { return isExternalType(input.getType()); }) ||
      !llvm::all_of(region.getResults(), [&](Value result) {
        return isExternalType(result.getType());
      })) {
    return false;
  }
  return llvm::all_of(algebra, [](Operation *operation) {
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      return isClosedProjectedContract(contract);
    }
    return isa<MapOp>(operation);
  });
}

bool isClosedRankTwoFold(FoldOp fold) {
  auto inputType =
      fold.getInputs().size() == 1
          ? dyn_cast<RankedTensorType>(fold.getInputs()[0].getType())
          : RankedTensorType{};
  auto initType =
      fold.getInitializers().size() == 1
          ? dyn_cast<RankedTensorType>(fold.getInitializers()[0].getType())
          : RankedTensorType{};
  auto resultType =
      fold.getNumResults() == 1
          ? dyn_cast<RankedTensorType>(fold.getResult(0).getType())
          : RankedTensorType{};
  if (!inputType || !initType || !resultType || !inputType.hasStaticShape() ||
      inputType.getRank() != 2 || inputType.getDimSize(0) <= 0 ||
      inputType.getDimSize(1) <= 0 || !inputType.getElementType().isF32() ||
      initType.getRank() != 0 || !initType.getElementType().isF32() ||
      fold.getReductionDimensions().size() != 1 ||
      fold.getReductionDimensions().front() < 0 ||
      fold.getReductionDimensions().front() >= inputType.getRank() ||
      !fold.getOrderFree() || fold.getAccumulatorTypes().size() != 1 ||
      cast<TypeAttr>(fold.getAccumulatorTypes()[0]).getValue() !=
          Float32Type::get(fold.getContext())) {
    return false;
  }
  SmallVector<int64_t> expectedResultShape(inputType.getShape());
  expectedResultShape.erase(expectedResultShape.begin() +
                            fold.getReductionDimensions().front());
  if (resultType.getShape() != ArrayRef<int64_t>(expectedResultShape) ||
      !resultType.getElementType().isF32()) {
    return false;
  }
  Block &body = fold.getCombiner().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2) {
    return false;
  }
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto maximum = dyn_cast<arith::MaximumFOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  Operation *reducer = add ? add.getOperation() : maximum.getOperation();
  auto fastMath =
      reducer ? reducer->getAttrOfType<arith::FastMathFlagsAttr>("fastmath")
              : arith::FastMathFlagsAttr{};
  if (!reducer || !yield || reducer->getOperand(0) != body.getArgument(0) ||
      reducer->getOperand(1) != body.getArgument(1) ||
      yield.getValues().size() != 1 ||
      yield.getValues()[0] != reducer->getResult(0) ||
      (fastMath && fastMath.getValue() != arith::FastMathFlags::none)) {
    return false;
  }
  return true;
}

bool hasExactPositiveZeroInitializer(FoldOp fold) {
  if (fold.getInitializers().size() != 1) {
    return false;
  }
  auto initializer = fold.getInitializers()[0].getDefiningOp<MapOp>();
  if (!initializer || !initializer.getInputs().empty() ||
      initializer.getBody().empty()) {
    return false;
  }
  Block &body = initializer.getBody().front();
  auto constant = body.getOperations().size() == 2
                      ? dyn_cast<arith::ConstantOp>(body.front())
                      : arith::ConstantOp{};
  auto value =
      constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
  return value && value.getType().isF32() && value.getValue().isZero() &&
         !value.getValue().isNegative();
}

bool isClosedTerminalRankNFold(FoldOp fold) {
  if (!isClosedRankTwoFold(fold)) {
    auto inputType =
        fold.getInputs().size() == 1
            ? dyn_cast<RankedTensorType>(fold.getInputs()[0].getType())
            : RankedTensorType{};
    if (!inputType || !inputType.hasStaticShape() || inputType.getRank() < 3 ||
        inputType.getRank() > 8 ||
        fold.getReductionDimensions() !=
            ArrayRef<int64_t>({inputType.getRank() - 1})) {
      return false;
    }
    // Reuse the closed scalar/type/combiner checks with a shape-only rank-two
    // view would obscure the actual operation. Spell out the rank-independent
    // portion here instead.
    auto initType =
        fold.getInitializers().size() == 1
            ? dyn_cast<RankedTensorType>(fold.getInitializers()[0].getType())
            : RankedTensorType{};
    auto resultType =
        fold.getNumResults() == 1
            ? dyn_cast<RankedTensorType>(fold.getResult(0).getType())
            : RankedTensorType{};
    if (!initType || !resultType || !inputType.getElementType().isF32() ||
        initType.getRank() != 0 || !initType.getElementType().isF32() ||
        !fold.getOrderFree() || fold.getAccumulatorTypes().size() != 1 ||
        cast<TypeAttr>(fold.getAccumulatorTypes()[0]).getValue() !=
            Float32Type::get(fold.getContext()) ||
        resultType.getShape() != inputType.getShape().drop_back() ||
        !resultType.getElementType().isF32()) {
      return false;
    }
    Block &body = fold.getCombiner().front();
    if (body.getNumArguments() != 2 || body.getOperations().size() != 2) {
      return false;
    }
    auto add = dyn_cast<arith::AddFOp>(body.front());
    auto yield = dyn_cast<YieldOp>(body.back());
    auto fastMath =
        add ? add->getAttrOfType<arith::FastMathFlagsAttr>("fastmath")
            : arith::FastMathFlagsAttr{};
    if (!add || !yield || add.getLhs() != body.getArgument(0) ||
        add.getRhs() != body.getArgument(1) || yield.getValues().size() != 1 ||
        yield.getValues()[0] != add.getResult() ||
        (fastMath && fastMath.getValue() != arith::FastMathFlags::none)) {
      return false;
    }
  }
  return hasExactPositiveZeroInitializer(fold);
}

bool isFullPrefixReduction(ArrayRef<int64_t> dimensions, int64_t rank) {
  return rank >= 3 && rank <= 8 &&
         dimensions.size() == static_cast<size_t>(rank - 1) &&
         llvm::all_of(llvm::enumerate(dimensions), [](auto indexedDimension) {
           return indexedDimension.value() ==
                  static_cast<int64_t>(indexedDimension.index());
         });
}

bool isClosedFullPrefixRankNFold(FoldOp fold) {
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
      !result.hasStaticShape() ||
      !isFullPrefixReduction(fold.getReductionDimensions(), input.getRank()) ||
      !input.getElementType().isF32() || initializer.getRank() != 0 ||
      !initializer.getElementType().isF32() || result.getRank() != 1 ||
      !result.getElementType().isF32() ||
      result.getShape() != input.getShape().take_back() ||
      !llvm::all_of(input.getShape(),
                    [](int64_t extent) { return extent > 0; }) ||
      !fold.getOrderFree() || fold.getAccumulatorTypes().size() != 1 ||
      cast<TypeAttr>(fold.getAccumulatorTypes()[0]).getValue() !=
          Float32Type::get(fold.getContext())) {
    return false;
  }
  Block &body = fold.getCombiner().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2) {
    return false;
  }
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  auto fastMath = add ? add->getAttrOfType<arith::FastMathFlagsAttr>("fastmath")
                      : arith::FastMathFlagsAttr{};
  return add && yield && add.getLhs() == body.getArgument(0) &&
         add.getRhs() == body.getArgument(1) && yield.getValues().size() == 1 &&
         yield.getValues()[0] == add.getResult() &&
         (!fastMath || fastMath.getValue() == arith::FastMathFlags::none) &&
         hasExactPositiveZeroInitializer(fold);
}

bool isClosedRankNFold(FoldOp fold) {
  return isClosedTerminalRankNFold(fold) || isClosedFullPrefixRankNFold(fold);
}

bool isClosedAdditivePlanningFold(FoldOp fold) {
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
      !result.hasStaticShape() || !input.getElementType().isF32() ||
      initializer.getRank() != 0 || !initializer.getElementType().isF32() ||
      !result.getElementType().isF32() || !fold.getOrderFree() ||
      fold.getAccumulatorTypes().size() != 1 ||
      cast<TypeAttr>(fold.getAccumulatorTypes()[0]).getValue() !=
          Float32Type::get(fold.getContext()) ||
      llvm::any_of(input.getShape(),
                   [](int64_t extent) { return extent <= 0; })) {
    return false;
  }
  ArrayRef<int64_t> dimensions = fold.getReductionDimensions();
  const bool rankTwo = input.getRank() == 2 && dimensions.size() == 1 &&
                       dimensions.front() >= 0 &&
                       dimensions.front() < input.getRank();
  const bool terminalRankN =
      input.getRank() >= 3 && input.getRank() <= 8 &&
      dimensions == ArrayRef<int64_t>({input.getRank() - 1});
  if (!rankTwo && !terminalRankN &&
      !isFullPrefixReduction(dimensions, input.getRank())) {
    return false;
  }
  SmallVector<int64_t> expectedResult(input.getShape());
  for (int64_t dimension : llvm::reverse(dimensions))
    expectedResult.erase(expectedResult.begin() + dimension);
  if (result.getShape() != ArrayRef<int64_t>(expectedResult))
    return false;
  Block &body = fold.getCombiner().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2)
    return false;
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  auto fastMath = add ? add->getAttrOfType<arith::FastMathFlagsAttr>("fastmath")
                      : arith::FastMathFlagsAttr{};
  return add && yield && add.getLhs() == body.getArgument(0) &&
         add.getRhs() == body.getArgument(1) && yield.getValues().size() == 1 &&
         yield.getValues()[0] == add.getResult() &&
         (!fastMath || fastMath.getValue() == arith::FastMathFlags::none);
}

bool isClosedSourceOrderedFloatingAddFold(FoldOp fold) {
  auto input =
      fold.getInputs().size() == 1
          ? dyn_cast<RankedTensorType>(fold.getInputs().front().getType())
          : RankedTensorType{};
  auto initializer =
      fold.getInitializers().size() == 1
          ? dyn_cast<RankedTensorType>(fold.getInitializers().front().getType())
          : RankedTensorType{};
  auto result = fold.getNumResults() == 1
                    ? dyn_cast<RankedTensorType>(fold.getResult(0).getType())
                    : RankedTensorType{};
  if (!input || !initializer || !result || !input.hasStaticShape() ||
      !result.hasStaticShape() || initializer.getRank() != 0 ||
      fold.getOrderFree() || fold.getReductionDimensions().empty() ||
      fold.getAccumulatorTypes().size() != 1 ||
      input.getElementType() != initializer.getElementType() ||
      input.getElementType() != result.getElementType() ||
      !isa<FloatType>(input.getElementType()) ||
      cast<TypeAttr>(fold.getAccumulatorTypes().getValue().front())
              .getValue() != input.getElementType())
    return false;
  Block &body = fold.getCombiner().front();
  auto add = body.getOperations().size() == 2
                 ? dyn_cast<arith::AddFOp>(body.front())
                 : arith::AddFOp{};
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  return body.getNumArguments() == 2 && add && yield &&
         add.getLhs() == body.getArgument(0) &&
         add.getRhs() == body.getArgument(1) &&
         yield.getValues() == ValueRange(add.getResult());
}

FailureOr<SmallVector<int64_t>> planningFoldDomainShape(FoldOp fold) {
  const size_t inputCount = fold.getInputs().size();
  if (inputCount == 0 || fold.getIndexingMaps().size() < inputCount)
    return failure();
  auto firstMap =
      dyn_cast<AffineMapAttr>(fold.getIndexingMaps().getValue().front());
  if (!firstMap || firstMap.getValue().getNumSymbols() != 0)
    return failure();
  SmallVector<int64_t> domain(firstMap.getValue().getNumDims(),
                              ShapedType::kDynamic);
  for (auto [input, mapAttr] : llvm::zip_equal(
           fold.getInputs(),
           fold.getIndexingMaps().getValue().take_front(inputCount))) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    auto map = dyn_cast<AffineMapAttr>(mapAttr);
    if (!type || !type.hasStaticShape() || !map ||
        map.getValue().getNumDims() != domain.size() ||
        map.getValue().getNumSymbols() != 0 ||
        map.getValue().getNumResults() != type.getRank()) {
      return failure();
    }
    for (auto [axis, expression] :
         llvm::enumerate(map.getValue().getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension || type.getDimSize(axis) <= 0)
        return failure();
      int64_t &extent = domain[dimension.getPosition()];
      if (!ShapedType::isDynamic(extent) && extent != type.getDimSize(axis))
        return failure();
      extent = type.getDimSize(axis);
    }
  }
  if (llvm::is_contained(domain, ShapedType::kDynamic))
    return failure();
  return domain;
}

bool isClosedMultiStatePlanningFold(FoldOp fold) {
  const size_t stateCount = fold.getInputs().size();
  if (stateCount == 0 || fold.getInitializers().size() != stateCount ||
      fold.getNumResults() != stateCount ||
      fold.getAccumulatorTypes().size() != stateCount ||
      fold.getIndexingMaps().size() != stateCount * 2 ||
      fold.getReductionDimensions().empty() || !fold.getOrderFree() ||
      fold.getCombiner().empty()) {
    return false;
  }

  FailureOr<SmallVector<int64_t>> domain = planningFoldDomainShape(fold);
  if (failed(domain))
    return false;
  const unsigned domainRank = domain->size();
  for (auto [input, initializer, result, inputMapAttr, resultMapAttr,
             accumulatorTypeAttr] :
       llvm::zip_equal(fold.getInputs(), fold.getInitializers(),
                       fold.getResults(),
                       fold.getIndexingMaps().getValue().take_front(stateCount),
                       fold.getIndexingMaps().getValue().drop_front(stateCount),
                       fold.getAccumulatorTypes())) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto initializerType = dyn_cast<RankedTensorType>(initializer.getType());
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    auto inputMap = dyn_cast<AffineMapAttr>(inputMapAttr);
    auto resultMap = dyn_cast<AffineMapAttr>(resultMapAttr);
    auto accumulatorType = dyn_cast<TypeAttr>(accumulatorTypeAttr);
    if (!inputType || !initializerType || !resultType || !inputMap ||
        !resultMap || !accumulatorType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || inputType.getRank() < 1 ||
        inputType.getRank() > 8 || !inputType.getElementType().isF32() ||
        initializerType.getRank() != 0 ||
        !initializerType.getElementType().isF32() ||
        !resultType.getElementType().isF32() ||
        accumulatorType.getValue() != Float32Type::get(fold.getContext()) ||
        inputMap.getValue().getNumSymbols() != 0 ||
        resultMap.getValue().getNumSymbols() != 0 ||
        inputMap.getValue().getNumResults() != inputType.getRank() ||
        resultMap.getValue().getNumResults() != resultType.getRank() ||
        llvm::any_of(inputType.getShape(),
                     [](int64_t extent) { return extent <= 0; }) ||
        llvm::any_of(resultType.getShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      return false;
    }
    if (inputMap.getValue().getNumDims() != domainRank ||
        resultMap.getValue().getNumDims() != domainRank) {
      return false;
    }
  }
  llvm::SmallDenseSet<int64_t> reductionDimensions;
  for (int64_t dimension : fold.getReductionDimensions()) {
    if (dimension < 0 || dimension >= domainRank ||
        !reductionDimensions.insert(dimension).second) {
      return false;
    }
  }

  Block &body = fold.getCombiner().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (body.getNumArguments() != stateCount * 2 || !yield ||
      yield.getValues().size() != stateCount ||
      llvm::any_of(
          body.getArguments(),
          [](BlockArgument argument) { return !argument.getType().isF32(); }) ||
      llvm::any_of(yield.getValues(),
                   [](Value value) { return !value.getType().isF32(); })) {
    return false;
  }
  for (Operation &operation : body.without_terminator()) {
    if (!isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::MaximumFOp, arith::NegFOp, math::ExpOp, math::TanhOp,
             math::RsqrtOp, arith::ConstantOp, arith::CmpFOp, arith::SelectOp,
             ScalarConvertOp>(operation)) {
      return false;
    }
    if (!llvm::all_of(operation.getOperands(), [&](Value operand) {
          return operand.getParentBlock() == &body;
        })) {
      return false;
    }
  }
  return true;
}

#include "OnlinePlanner.inc"

bool isSupportedGenericGpuAffineExpression(AffineExpr expression,
                                           unsigned domainRank) {
  if (auto dimension = dyn_cast<AffineDimExpr>(expression))
    return dimension.getPosition() < domainRank;
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return constant.getValue() >= 0;
  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return false;
  if (binary.getKind() == AffineExprKind::FloorDiv ||
      binary.getKind() == AffineExprKind::Mod) {
    auto divisor = dyn_cast<AffineConstantExpr>(binary.getRHS());
    return divisor && divisor.getValue() > 0 &&
           isSupportedGenericGpuAffineExpression(binary.getLHS(), domainRank);
  }
  return (binary.getKind() == AffineExprKind::Add ||
          binary.getKind() == AffineExprKind::Mul) &&
         isSupportedGenericGpuAffineExpression(binary.getLHS(), domainRank) &&
         isSupportedGenericGpuAffineExpression(binary.getRHS(), domainRank);
}

bool isSupportedGenericGpuAffineMap(AffineMap map, unsigned domainRank,
                                    unsigned resultRank) {
  return map.getNumDims() == domainRank && map.getNumSymbols() == 0 &&
         map.getNumResults() == resultRank &&
         llvm::all_of(map.getResults(), [&](AffineExpr expression) {
           return isSupportedGenericGpuAffineExpression(expression, domainRank);
         });
}

bool isClosedDirectProjectedMap(MapOp map) {
  if (map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != map.getInputs().size() + 1) {
    return false;
  }
  auto result = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  ArrayRef<Attribute> maps = map.getIndexingMaps().getValue();
  auto resultMap = dyn_cast<AffineMapAttr>(maps.back());
  if (!result || !result.hasStaticShape() || !resultMap ||
      !isIdentityMap(resultMap.getValue())) {
    return false;
  }
  for (auto [input, attribute] :
       llvm::zip_equal(map.getInputs(), maps.drop_back())) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    auto indexing = dyn_cast<AffineMapAttr>(attribute);
    if (!type || !type.hasStaticShape() || !indexing ||
        indexing.getValue().getNumDims() != result.getRank() ||
        indexing.getValue().getNumSymbols() != 0 ||
        indexing.getValue().getNumResults() != type.getRank()) {
      return false;
    }
    llvm::SmallDenseSet<unsigned> dimensions;
    for (auto [axis, expression] :
         llvm::enumerate(indexing.getValue().getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension || dimension.getPosition() >= result.getRank() ||
          !dimensions.insert(dimension.getPosition()).second ||
          type.getDimSize(axis) != result.getDimSize(dimension.getPosition())) {
        return false;
      }
    }
  }
  return true;
}

bool isClosedExactContiguousReshapeMap(MapOp map) {
  if (map.getSemantics() != MapSemantics::Reshape ||
      map.getInputs().size() != 1 || map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != 2 || map.getBody().empty()) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(map.getInputs().front().getType());
  auto result = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  auto inputMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[0]);
  auto resultMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[1]);
  if (!input || !result || !input.hasStaticShape() ||
      !result.hasStaticShape() || !input.getElementType().isF32() ||
      !result.getElementType().isF32() || !inputMap || !resultMap ||
      !isIdentityMap(resultMap.getValue()) ||
      !isSupportedGenericGpuAffineMap(inputMap.getValue(), result.getRank(),
                                      input.getRank())) {
    return false;
  }
  FailureOr<AffineMap> expected = contiguousReshapeInputMap(input, result);
  Block &body = map.getBody().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  return succeeded(expected) && inputMap.getValue() == *expected &&
         body.getNumArguments() == 1 && body.getOperations().size() == 1 &&
         yield && yield.getValues().size() == 1 &&
         yield.getValues().front() == body.getArgument(0);
}

bool isClosedFullPrefixBf16Result(MapOp map) {
  if (map.getSemantics() != MapSemantics::Pointwise ||
      map.getInputs().size() != 1 || map.getNumResults() != 1 ||
      !isClosedDirectProjectedMap(map) || map.getBody().empty()) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(map.getInputs()[0].getType());
  auto result = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  auto fold = map.getInputs()[0].getDefiningOp<FoldOp>();
  if (!input || !result || !fold || fold->getParentOp() != map->getParentOp() ||
      !isClosedFullPrefixRankNFold(fold) || input.getRank() != 1 ||
      result.getRank() != 1 || input.getShape() != result.getShape() ||
      !input.getElementType().isF32() || !result.getElementType().isBF16() ||
      !llvm::all_of(result.getShape(),
                    [](int64_t extent) { return extent > 0; })) {
    return false;
  }
  Block &body = map.getBody().front();
  if (body.getNumArguments() != 1 || body.getOperations().size() != 2) {
    return false;
  }
  auto convert = dyn_cast<ScalarConvertOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  return convert && yield && convert.getInput() == body.getArgument(0) &&
         convert.getInput().getType().isF32() &&
         convert.getResult().getType().isBF16() &&
         convert.getSemantics() == ScalarConvertSemantics::RoundNearestEven &&
         yield.getValues().size() == 1 &&
         yield.getValues()[0] == convert.getResult();
}

std::optional<unsigned> closedRankNExpandingSingletonMaps(MapOp map) {
  if (map.getSemantics() != MapSemantics::BroadcastInDim ||
      map.getInputs().size() != 1 || map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != 2 || map.getBody().empty()) {
    return std::nullopt;
  }
  auto input = dyn_cast<RankedTensorType>(map.getInputs()[0].getType());
  auto result = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  auto inputMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[0]);
  auto resultMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[1]);
  if (!input || !result || !input.hasStaticShape() ||
      !result.hasStaticShape() || input.getRank() < 3 || input.getRank() > 8 ||
      input.getRank() != result.getRank() || !input.getElementType().isF32() ||
      !result.getElementType().isF32() || !inputMap || !resultMap ||
      inputMap.getValue().getNumSymbols() != 0 ||
      inputMap.getValue().getNumDims() != result.getRank() ||
      inputMap.getValue().getNumResults() != input.getRank() ||
      !isIdentityMap(resultMap.getValue())) {
    return std::nullopt;
  }

  unsigned expandingSingletons = 0;
  for (auto [axis, expression] :
       llvm::enumerate(inputMap.getValue().getResults())) {
    if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
      if (dimension.getPosition() != axis ||
          input.getDimSize(axis) != result.getDimSize(axis)) {
        return std::nullopt;
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
    auto divisor = dyn_cast_if_present<AffineConstantExpr>(divisorExpression);
    if (!dividend || !divisor || dividend.getPosition() != axis ||
        input.getDimSize(axis) != 1 || result.getDimSize(axis) <= 1 ||
        divisor.getValue() != result.getDimSize(axis)) {
      return std::nullopt;
    }
    ++expandingSingletons;
  }
  if (expandingSingletons == 0) {
    return std::nullopt;
  }
  Block &body = map.getBody().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (body.getNumArguments() != 1 || body.getOperations().size() != 1 ||
      !yield || yield.getValues().size() != 1 ||
      yield.getValues()[0] != body.getArgument(0)) {
    return std::nullopt;
  }
  return expandingSingletons;
}

bool hasIdentityScalarBody(MapOp map) {
  if (map.getBody().empty()) {
    return false;
  }
  Block &body = map.getBody().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  return body.getNumArguments() == 1 && body.getOperations().size() == 1 &&
         yield && yield.getValues().size() == 1 &&
         yield.getValues()[0] == body.getArgument(0);
}

bool isClosedProjectedParameterInsertionMap(MapOp map) {
  if (map.getSemantics() != MapSemantics::BroadcastInDim ||
      map.getInputs().size() != 1 || map.getNumResults() != 1 ||
      !isClosedDirectProjectedMap(map) || !hasIdentityScalarBody(map)) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(map.getInputs()[0].getType());
  auto result = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  auto inputMap = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[0]);
  if (!input || !result || !inputMap || !input.getElementType().isF32() ||
      !result.getElementType().isF32() || input.getRank() < 1 ||
      input.getRank() >= result.getRank() || result.getRank() > 8) {
    return false;
  }
  llvm::SmallDenseSet<unsigned> mapped;
  for (AffineExpr expression : inputMap.getValue().getResults()) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || !mapped.insert(dimension.getPosition()).second) {
      return false;
    }
  }
  for (auto [axis, extent] : llvm::enumerate(result.getShape())) {
    if (!mapped.contains(axis) && extent != 1) {
      return false;
    }
  }
  return true;
}

std::optional<Value> projectedBf16ParameterRoot(MapOp conversion,
                                                int64_t expansionRank) {
  if (conversion.getSemantics() != MapSemantics::Pointwise ||
      conversion.getInputs().size() != 1 || conversion.getNumResults() != 1 ||
      conversion.getIndexingMaps().size() != 2 ||
      conversion.getInputs()[0].getDefiningOp() ||
      conversion.getBody().empty()) {
    return std::nullopt;
  }
  auto input = dyn_cast<RankedTensorType>(conversion.getInputs()[0].getType());
  auto result = dyn_cast<RankedTensorType>(conversion.getResult(0).getType());
  auto inputMap = dyn_cast<AffineMapAttr>(conversion.getIndexingMaps()[0]);
  auto resultMap = dyn_cast<AffineMapAttr>(conversion.getIndexingMaps()[1]);
  if (!input || !result || !input.hasStaticShape() ||
      !result.hasStaticShape() || input.getRank() < 1 ||
      input.getRank() >= expansionRank ||
      input.getShape() != result.getShape() ||
      !input.getElementType().isBF16() || !result.getElementType().isF32() ||
      !inputMap || !resultMap || !isIdentityMap(inputMap.getValue()) ||
      !isIdentityMap(resultMap.getValue()) ||
      !llvm::all_of(input.getShape(),
                    [](int64_t extent) { return extent > 0; })) {
    return std::nullopt;
  }
  Block &body = conversion.getBody().front();
  if (body.getNumArguments() != 1 || body.getOperations().size() != 2) {
    return std::nullopt;
  }
  auto scalarConvert = dyn_cast<ScalarConvertOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  if (!scalarConvert || !yield ||
      scalarConvert.getInput() != body.getArgument(0) ||
      !scalarConvert.getInput().getType().isBF16() ||
      !scalarConvert.getResult().getType().isF32() ||
      scalarConvert.getSemantics() != ScalarConvertSemantics::Exact ||
      yield.getValues().size() != 1 ||
      yield.getValues()[0] != scalarConvert.getResult()) {
    return std::nullopt;
  }
  return conversion.getInputs()[0];
}

std::optional<Value> projectedParameterRoot(MapOp expansion) {
  std::optional<unsigned> expansions =
      closedRankNExpandingSingletonMaps(expansion);
  if (!expansions || *expansions < 2) {
    return std::nullopt;
  }
  auto expansionType = cast<RankedTensorType>(expansion.getResult(0).getType());
  Region *region = expansion->getParentRegion();
  Value producerValue = expansion.getInputs()[0];
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (Operation *producer = producerValue.getDefiningOp()) {
    if (producer->getParentRegion() != region ||
        !visited.insert(producer).second) {
      return std::nullopt;
    }
    auto map = dyn_cast<MapOp>(producer);
    if (!map) {
      return std::nullopt;
    }
    if (std::optional<Value> root =
            projectedBf16ParameterRoot(map, expansionType.getRank())) {
      auto argument = dyn_cast<BlockArgument>(*root);
      auto parentRegion = dyn_cast<RegionOp>(expansion->getParentOp());
      if (!argument || !parentRegion ||
          argument.getOwner() != &parentRegion.getBody().front() ||
          argument.getArgNumber() >= parentRegion.getInputs().size()) {
        return std::nullopt;
      }
      return parentRegion.getInputs()[argument.getArgNumber()];
    }
    if (!isClosedProjectedParameterInsertionMap(map) ||
        map.getInputs().size() != 1) {
      return std::nullopt;
    }
    producerValue = map.getInputs()[0];
  }
  return std::nullopt;
}

bool isExpansionDrivenByTerminalFold(MapOp expansion) {
  Operation *root = expansion.getInputs()[0].getDefiningOp();
  if (!root) {
    return false;
  }
  Operation *parent = expansion->getParentOp();
  SmallVector<Operation *> worklist{root};
  llvm::SmallPtrSet<Operation *, 8> visited;
  FoldOp terminalFold;
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val();
    if (producer->getParentOp() != parent) {
      return false;
    }
    if (!visited.insert(producer).second) {
      continue;
    }
    if (auto fold = dyn_cast<FoldOp>(producer)) {
      if (!isClosedTerminalRankNFold(fold) ||
          (terminalFold && terminalFold != fold)) {
        return false;
      }
      terminalFold = fold;
      continue;
    }
    auto map = dyn_cast<MapOp>(producer);
    if (!map || !isClosedDirectProjectedMap(map) ||
        closedRankNExpandingSingletonMaps(map).has_value()) {
      return false;
    }
    if (map.getInputs().empty()) {
      if (map.getBody().empty()) {
        return false;
      }
      auto result = map.getNumResults() == 1
                        ? dyn_cast<RankedTensorType>(map.getResult(0).getType())
                        : RankedTensorType{};
      Block &body = map.getBody().front();
      auto constant = body.getOperations().size() == 2
                          ? dyn_cast<arith::ConstantOp>(body.front())
                          : arith::ConstantOp{};
      auto yield = dyn_cast<YieldOp>(body.getTerminator());
      auto value =
          constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
      if (!result || result.getRank() != 0 ||
          !result.getElementType().isF32() || !value ||
          !value.getType().isF32() || !yield || yield.getValues().size() != 1 ||
          yield.getValues()[0] != constant.getResult()) {
        return false;
      }
      continue;
    }
    for (Value input : map.getInputs()) {
      Operation *inputProducer = input.getDefiningOp();
      if (!inputProducer || inputProducer->getParentOp() != parent) {
        return false;
      }
      worklist.push_back(inputProducer);
    }
  }
  return static_cast<bool>(terminalFold);
}

bool isClosedRankNFoldMapRegion(RegionOp region,
                                ArrayRef<Operation *> algebra) {
  llvm::SmallDenseSet<Value, 4> parameterRoots;
  for (Operation *operation : algebra) {
    auto map = dyn_cast<MapOp>(operation);
    if (!map) {
      continue;
    }
    std::optional<unsigned> expansions = closedRankNExpandingSingletonMaps(map);
    if (!expansions || *expansions < 2) {
      continue;
    }
    std::optional<Value> root = projectedParameterRoot(map);
    if (!root) {
      return false;
    }
    parameterRoots.insert(*root);
  }
  auto isExternalInput = [&](Value value) {
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    return value.getDefiningOp() == nullptr && tensor &&
           tensor.hasStaticShape() && tensor.getRank() >= 1 &&
           tensor.getRank() <= 8 && tensor.getElementType().isBF16() &&
           (tensor.getRank() >= 3 || parameterRoots.contains(value)) &&
           llvm::all_of(tensor.getShape(),
                        [](int64_t extent) { return extent > 0; });
  };
  auto isExternalResult = [](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    return tensor && tensor.hasStaticShape() && tensor.getRank() >= 1 &&
           tensor.getRank() <= 8 && tensor.getElementType().isBF16() &&
           llvm::all_of(tensor.getShape(),
                        [](int64_t extent) { return extent > 0; });
  };
  if (!llvm::all_of(region.getInputs(), isExternalInput) ||
      !llvm::all_of(region.getResults(), [&](Value result) {
        return isExternalResult(result.getType());
      })) {
    return false;
  }
  auto regionYield =
      dyn_cast<YieldOp>(region.getBody().front().getTerminator());
  if (!regionYield ||
      regionYield.getValues().size() != region.getResults().size()) {
    return false;
  }
  for (auto [result, yielded] :
       llvm::zip_equal(region.getResults(), regionYield.getValues())) {
    auto type = cast<RankedTensorType>(result.getType());
    if (type.getRank() < 3) {
      auto map = yielded.getDefiningOp<MapOp>();
      if (!map || !isClosedFullPrefixBf16Result(map)) {
        return false;
      }
    }
  }
  return llvm::all_of(algebra, [](Operation *operation) {
    if (auto fold = dyn_cast<FoldOp>(operation)) {
      auto input =
          fold.getInputs().size() == 1
              ? dyn_cast<RankedTensorType>(fold.getInputs()[0].getType())
              : RankedTensorType{};
      return input && input.getRank() >= 3 && input.getRank() <= 8 &&
             isClosedRankNFold(fold);
    }
    auto map = dyn_cast<MapOp>(operation);
    if (!map) {
      return false;
    }
    std::optional<unsigned> expansions = closedRankNExpandingSingletonMaps(map);
    if (!expansions)
      return isClosedDirectProjectedMap(map) ||
             isClosedExactContiguousReshapeMap(map);
    return *expansions == 1 ? isExpansionDrivenByTerminalFold(map)
                            : projectedParameterRoot(map).has_value();
  });
}

FailureOr<AffineMap> relationIotaCoordinateMap(Value value) {
  auto relation = value.getDefiningOp<RelationOp>();
  if (!relation || relation.getNumResults() != 1 ||
      relation.getResult(0) != value)
    return failure();
  auto resultType = dyn_cast<RankedTensorType>(value.getType());
  if (!resultType)
    return failure();
  if (relation.getSemantics() == RelationSemantics::Iota) {
    if (!relation.getInputs().empty() || relation.getParameters().size() != 1)
      return failure();
    auto dimension = dyn_cast<IntegerAttr>(relation.getParameters()[0]);
    if (!dimension || dimension.getInt() < 0 ||
        dimension.getInt() >= resultType.getRank())
      return failure();
    return AffineMap::get(
        resultType.getRank(), 0,
        getAffineDimExpr(dimension.getInt(), value.getContext()),
        value.getContext());
  }
  if (relation.getInputs().size() != 1)
    return failure();
  Value input = relation.getInputs().front();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  FailureOr<AffineMap> inputIota = relationIotaCoordinateMap(input);
  if (!inputType || failed(inputIota))
    return failure();
  if (relation.getSemantics() == RelationSemantics::BroadcastInDim) {
    if (relation.getParameters().size() != inputType.getRank())
      return failure();
    SmallVector<AffineExpr> inputCoordinates;
    for (Attribute parameter : relation.getParameters()) {
      auto resultDimension = dyn_cast<IntegerAttr>(parameter);
      if (!resultDimension || resultDimension.getInt() < 0 ||
          resultDimension.getInt() >= resultType.getRank())
        return failure();
      inputCoordinates.push_back(
          getAffineDimExpr(resultDimension.getInt(), value.getContext()));
    }
    return simplifyAffineMap(inputIota->compose(AffineMap::get(
        resultType.getRank(), 0, inputCoordinates, value.getContext())));
  }
  if (relation.getSemantics() != RelationSemantics::Reshape)
    return failure();
  FailureOr<AffineMap> reshapeInput =
      contiguousReshapeInputMap(inputType, resultType);
  if (failed(reshapeInput))
    return failure();
  AffineMap composed = simplifyAffineMap(inputIota->compose(*reshapeInput));
  if (isa<AffineDimExpr>(composed.getResult(0)))
    return composed;

  // The generic affine simplifier has no static dimension bounds, so it
  // cannot reduce `(d mod extent)` to `d`.  Prove that reduction from the
  // exact row-major digit: both its radix and trailing stride must match.
  auto inputDimension = dyn_cast<AffineDimExpr>(inputIota->getResult(0));
  if (!inputDimension ||
      inputType.getDimSize(inputDimension.getPosition()) <= 1)
    return composed;
  int64_t inputStride = 1;
  for (int64_t axis = inputDimension.getPosition() + 1;
       axis < inputType.getRank(); ++axis) {
    const int64_t extent = inputType.getDimSize(axis);
    if (extent <= 0 ||
        inputStride > std::numeric_limits<int64_t>::max() / extent)
      return failure();
    inputStride *= extent;
  }
  int64_t resultStride = 1;
  std::optional<int64_t> exactResultDimension;
  for (int64_t axis = resultType.getRank() - 1; axis >= 0; --axis) {
    if (resultStride == inputStride &&
        resultType.getDimSize(axis) ==
            inputType.getDimSize(inputDimension.getPosition())) {
      if (exactResultDimension)
        return composed;
      exactResultDimension = axis;
    }
    const int64_t extent = resultType.getDimSize(axis);
    if (extent <= 0 ||
        resultStride > std::numeric_limits<int64_t>::max() / extent)
      return failure();
    resultStride *= extent;
  }
  if (!exactResultDimension)
    return composed;
  return AffineMap::get(
      resultType.getRank(), 0,
      getAffineDimExpr(*exactResultDimension, value.getContext()),
      value.getContext());
}

FailureOr<AffineMap> relationCoordinatesToAncestor(Value value,
                                                   Value ancestor) {
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  auto ancestorType = dyn_cast<RankedTensorType>(ancestor.getType());
  if (!valueType || !ancestorType)
    return failure();
  if (value == ancestor)
    return AffineMap::getMultiDimIdentityMap(valueType.getRank(),
                                             value.getContext());
  auto relation = value.getDefiningOp<RelationOp>();
  if (!relation || relation.getNumResults() != 1 ||
      relation.getResult(0) != value || relation.getInputs().size() != 1)
    return failure();
  Value input = relation.getInputs().front();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  FailureOr<AffineMap> upstream =
      relationCoordinatesToAncestor(input, ancestor);
  if (!inputType || failed(upstream) ||
      upstream->getNumDims() != inputType.getRank())
    return failure();

  AffineMap inputCoordinates;
  if (relation.getSemantics() == RelationSemantics::BroadcastInDim) {
    if (relation.getParameters().size() != inputType.getRank())
      return failure();
    SmallVector<AffineExpr> coordinates;
    for (Attribute parameter : relation.getParameters()) {
      auto resultDimension = dyn_cast<IntegerAttr>(parameter);
      if (!resultDimension || resultDimension.getInt() < 0 ||
          resultDimension.getInt() >= valueType.getRank())
        return failure();
      coordinates.push_back(
          getAffineDimExpr(resultDimension.getInt(), value.getContext()));
    }
    inputCoordinates =
        AffineMap::get(valueType.getRank(), 0, coordinates, value.getContext());
  } else if (relation.getSemantics() == RelationSemantics::Reshape) {
    FailureOr<AffineMap> reshape =
        contiguousReshapeInputMap(inputType, valueType);
    if (failed(reshape))
      return failure();
    inputCoordinates = *reshape;
  } else {
    return failure();
  }
  return simplifyAffineMap(upstream->compose(inputCoordinates));
}

std::optional<int64_t> relationVaryingAxis(Value value, AffineMap domainMap) {
  FailureOr<AffineMap> coordinateMap = relationIotaCoordinateMap(value);
  if (failed(coordinateMap) || coordinateMap->getNumResults() != 1 ||
      coordinateMap->getNumDims() != domainMap.getNumResults())
    return std::nullopt;
  AffineMap lifted = simplifyAffineMap(coordinateMap->compose(domainMap));
  auto dimension = dyn_cast<AffineDimExpr>(lifted.getResult(0));
  return dimension ? std::optional<int64_t>(dimension.getPosition())
                   : std::nullopt;
}

struct PlannedTask {
  Operation *operation;
  MaterializationTaskKind kind;
  SmallVector<int64_t> domain;
  SmallVector<int64_t> reductionDimensions;
  bool orderFree = false;
  SmallVector<int64_t> inputs;
  SmallVector<int64_t> outputs;
  SmallVector<int64_t> dependencies;
  SmallVector<Operation *> fusedOperations;
  SmallVector<AffineMap> composedInputMaps;
  SmallVector<Operation *> fusedMapOutputs;
  Operation *fusedFold = nullptr;
  SmallVector<Operation *> foldRegionSources;
  SmallVector<Operation *> foldRegionPhases;
  SmallVector<Operation *> foldRegionOutputs;
  struct BoundaryUse {
    int64_t inputBuffer;
    MapOp source;
    unsigned operand;
    AffineMap indexingMap;
    int64_t phase;
  };
  SmallVector<BoundaryUse> foldRegionBoundaryUses;
  SmallVector<Operation *> foldRegionRecomputedSources;
  SmallVector<Operation *> contractRegionSources;
  std::string contractRegionStrategy;
  Operation *contractRegionFold = nullptr;
  Operation *contractRegionContract = nullptr;
  Operation *contractRegionBf16Boundary = nullptr;
  SmallVector<Operation *> contractRegionEpilogueSources;
  int64_t contractRegionNormalizedOperand = -1;
  struct ContractRegionCost {
    int64_t rows;
    int64_t reduction;
    int64_t columns;
    int64_t outputTile;
    int64_t outputTiles;
    int64_t duplicatedFoldElements;
    int64_t normalizedTemporaryBytes;
    int64_t contractTemporaryBytes;
    int64_t compactStatBytes;
  } contractRegionCost{};
  SmallVector<Operation *> contractRegionRecomputedSources;
  std::string compositionKind;
  bool statefulContractFoldComposition = false;
  SmallVector<Operation *> compositionSources;
  SmallVector<Operation *> compositionRecomputedSources;
  SmallVector<Operation *> compositionPhases;
  SmallVector<int64_t> compositionOwnerDimensions;
  SmallVector<int64_t> compositionStreamDimensions;
  SmallVector<int64_t> compositionStateInputs;
  SmallVector<int64_t> compositionStateOutputs;
  SmallVector<int64_t> compositionProducerContractionDimensions;
  SmallVector<int64_t> compositionConsumerContractionDimensions;
  SourceRefAttr compositionIndexDomainSource;
  SmallVector<int64_t> compositionIndexDomainShape;
  AffineMap compositionIndexDomainMap;
  SourceRefAttr compositionOrderedIndexSource;
  SmallVector<int64_t> compositionOrderedIndexDimensions;
  std::optional<bool> compositionOrderedIndexInclusive;
  SmallVector<int64_t> compositionStateAccessBuffers;
  SmallVector<AffineMap> compositionStateAccessMaps;
  SmallVector<int64_t> onlineVectorDimensions;
  bool onlineMma = false;
  struct CompositionCost {
    int64_t baselineLaunches;
    int64_t candidateLaunches;
    int64_t baselineTemporaryBytes;
    int64_t candidateTemporaryBytes;
    int64_t baselineTemporaryTrafficBytes;
    int64_t candidateTemporaryTrafficBytes;
    int64_t contractEvaluations;
  } compositionCost{};
};

struct PlannedBuffer {
  Value value;
  std::optional<int64_t> producer;
  SmallVector<int64_t> consumers;
  bool liveOut = false;
};

struct PlannedNormalizedExpContractRegion {
  Operation *producer;
  Value score;
  SmallVector<Operation *> foldOperations;
  Value maximum;
  Value normalizer;
  Value normalizedWeight;
  Operation *consumer;
  unsigned normalizedOperand;
  Value finalizer;
  std::optional<unsigned> outputResult;
};

struct PlannedNormalizedExpContractPullback {
  unsigned forwardRegionOrdinal;
  int64_t forwardTask;
  std::optional<int64_t> objectiveTask;
  int64_t ownerPullbackTask;
  int64_t streamedPullbackTask;
  int64_t maximumStateBuffer;
  int64_t normalizerStateBuffer;
  int64_t outputDependentStateBuffer;
  std::optional<int64_t> forwardResultBuffer;
  int64_t outputCotangentBuffer;
  SmallVector<int64_t> terminalCotangentBuffers;
  Operation *normalizedOutput;
  Operation *postNormalizationVjp;
  Operation *outputProduct;
  Operation *outputDependentFold;
  Operation *consumerWeightCotangent;
  Operation *normalizedWeight;
  Operation *normalizedExpPullback;
  Operation *scoreMapVjp;
  SmallVector<Operation *> terminalContracts;
};

struct DerivedPlan {
  SmallVector<PlannedTask, 0> tasks;
  SmallVector<PlannedBuffer, 0> buffers;
  bool hasContract = false;
  bool hasRankNFold = false;
  bool hasFoldRegion = false;
  bool hasContractRegion = false;
  bool hasContractFoldComposition = false;
  bool hasLogicalRelation = false;
  bool hasStatefulContractFoldPullback = false;
  SmallVector<PlannedNormalizedExpContractRegion> normalizedExpContractRegions;
  SmallVector<PlannedNormalizedExpContractPullback, 0>
      normalizedExpContractPullbacks;
};

SourceRefAttr algebraSource(Operation *operation);
DictionaryAttr algebraSourceExpansion(Operation *operation);

struct IndexedAccumulationExpansion {
  unsigned begin;
  unsigned end;
  RelationOp contributions;
  MapOp initializer;
  FoldOp fold;
  MapOp finalizer;
};

std::optional<int64_t> sourceExpansionMember(Operation *operation,
                                             StringRef field) {
  DictionaryAttr expansion = algebraSourceExpansion(operation);
  auto value = expansion ? expansion.getAs<IntegerAttr>(field) : IntegerAttr{};
  return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
}

std::optional<IndexedAccumulationExpansion>
indexedAccumulationExpansion(ArrayRef<Operation *> algebra, unsigned begin) {
  if (begin + 4 > algebra.size())
    return std::nullopt;
  auto relation = dyn_cast<RelationOp>(algebra[begin]);
  auto initializer = dyn_cast<MapOp>(algebra[begin + 1]);
  auto fold = dyn_cast<FoldOp>(algebra[begin + 2]);
  auto finalizer = dyn_cast<MapOp>(algebra[begin + 3]);
  if (!relation || !initializer || !fold || !finalizer ||
      relation.getSemantics() != RelationSemantics::IndexedContributions ||
      fold.getInputs().size() != 1 || fold.getInitializers().size() != 1 ||
      fold.getNumResults() != 1 || fold.getOrderFree() ||
      fold.getReductionDimensions().empty() ||
      fold.getInputs().front() != relation.getResult(0) ||
      fold.getInitializers().front() != initializer.getResult(0) ||
      !fold.getResult(0).hasOneUse() ||
      *fold.getResult(0).getUsers().begin() != finalizer.getOperation() ||
      finalizer.getSemantics() != MapSemantics::Pointwise)
    return std::nullopt;
  SourceRefAttr source = algebraSource(relation);
  for (auto [member, operation] : llvm::enumerate(algebra.slice(begin, 4))) {
    if (algebraSource(operation) != source ||
        sourceExpansionMember(operation, "member") !=
            std::optional<int64_t>(member) ||
        sourceExpansionMember(operation, "size") != std::optional<int64_t>(4))
      return std::nullopt;
  }
  llvm::SmallDenseSet<Operation *> members(algebra.begin() + begin,
                                           algebra.begin() + begin + 4);
  for (Operation *operation : algebra.slice(begin, 3))
    for (Value result : operation->getResults())
      if (llvm::any_of(result.getUsers(), [&](Operation *user) {
            return !members.contains(user);
          }))
        return std::nullopt;
  return IndexedAccumulationExpansion{begin,       begin + 4, relation,
                                      initializer, fold,      finalizer};
}

FailureOr<DerivedPlan>
groupIndexedAccumulationExpansions(ArrayRef<Operation *> algebra,
                                   DerivedPlan base) {
  if (base.tasks.size() != algebra.size())
    return failure();
  SmallVector<SmallVector<unsigned>> components;
  SmallVector<std::optional<IndexedAccumulationExpansion>> expansions;
  for (unsigned position = 0; position < algebra.size();) {
    auto expansion = indexedAccumulationExpansion(algebra, position);
    SmallVector<unsigned> members;
    if (expansion) {
      for (unsigned member = expansion->begin; member < expansion->end;
           ++member)
        members.push_back(member);
      position = expansion->end;
    } else {
      members.push_back(position++);
    }
    components.push_back(std::move(members));
    expansions.push_back(expansion);
  }
  if (llvm::none_of(expansions,
                    [](const auto &value) { return value.has_value(); }))
    return failure();

  SmallVector<unsigned> componentForTask(base.tasks.size());
  for (auto [component, members] : llvm::enumerate(components))
    for (unsigned member : members)
      componentForTask[member] = component;

  DerivedPlan result;
  result.hasContract = base.hasContract;
  result.hasRankNFold = base.hasRankNFold;
  result.hasLogicalRelation = base.hasLogicalRelation;
  llvm::SmallDenseMap<unsigned, unsigned> newBufferForOld;
  for (auto [oldOrdinal, buffer] : llvm::enumerate(base.buffers)) {
    std::optional<unsigned> producer =
        buffer.producer
            ? std::optional<unsigned>(componentForTask[*buffer.producer])
            : std::nullopt;
    bool crossesBoundary = !producer || buffer.liveOut;
    for (int64_t consumer : buffer.consumers)
      crossesBoundary |= !producer || componentForTask[consumer] != *producer;
    if (!crossesBoundary)
      continue;
    unsigned newOrdinal = result.buffers.size();
    newBufferForOld.try_emplace(oldOrdinal, newOrdinal);
    PlannedBuffer planned{buffer.value,
                          producer ? std::optional<int64_t>(*producer)
                                   : std::nullopt,
                          {},
                          buffer.liveOut};
    for (int64_t consumer : buffer.consumers) {
      int64_t mapped = componentForTask[consumer];
      if (planned.consumers.empty() || planned.consumers.back() != mapped)
        planned.consumers.push_back(mapped);
    }
    result.buffers.push_back(std::move(planned));
  }

  for (auto [component, members] : llvm::enumerate(components)) {
    PlannedTask task = base.tasks[members.front()];
    if (expansions[component]) {
      IndexedAccumulationExpansion expansion = *expansions[component];
      task = PlannedTask{};
      task.operation = expansion.fold;
      task.kind = MaterializationTaskKind::Fold;
      FailureOr<SmallVector<int64_t>> domain =
          planningFoldDomainShape(expansion.fold);
      if (failed(domain))
        return failure();
      task.domain = std::move(*domain);
      task.reductionDimensions.assign(
          expansion.fold.getReductionDimensions().begin(),
          expansion.fold.getReductionDimensions().end());
      task.orderFree = false;
      task.compositionKind = "indexed_accumulation";
      task.compositionSources.assign(algebra.begin() + expansion.begin,
                                     algebra.begin() + expansion.end);
      llvm::SmallDenseSet<int64_t> reductions(task.reductionDimensions.begin(),
                                              task.reductionDimensions.end());
      for (int64_t axis = 0; axis < static_cast<int64_t>(task.domain.size());
           ++axis) {
        if (reductions.contains(axis))
          task.compositionStreamDimensions.push_back(axis);
        else
          task.compositionOwnerDimensions.push_back(axis);
      }
    }
    task.inputs.clear();
    task.outputs.clear();
    task.dependencies.clear();
    for (auto [oldOrdinal, buffer] : llvm::enumerate(base.buffers)) {
      auto mapped = newBufferForOld.find(oldOrdinal);
      if (mapped == newBufferForOld.end())
        continue;
      const bool consumes = llvm::any_of(buffer.consumers, [&](int64_t value) {
        return componentForTask[value] == component;
      });
      const bool produced =
          buffer.producer && componentForTask[*buffer.producer] == component;
      if (consumes && (!buffer.producer || !produced)) {
        task.inputs.push_back(mapped->second);
        if (buffer.producer) {
          int64_t dependency = componentForTask[*buffer.producer];
          if (!llvm::is_contained(task.dependencies, dependency))
            task.dependencies.push_back(dependency);
        }
      }
      if (produced)
        task.outputs.push_back(mapped->second);
    }
    llvm::sort(task.dependencies);
    result.tasks.push_back(std::move(task));
  }
  return result;
}

SourceRefAttr algebraSource(Operation *operation) {
  if (auto scan = dyn_cast<ScanOp>(operation))
    return cast<SourceRefAttr>(scan.getSources().getValue().front());
  if (auto selection = dyn_cast<SelectionOp>(operation))
    return cast<SourceRefAttr>(selection.getSources().getValue().front());
  if (auto map = dyn_cast<MapOp>(operation))
    return map.getSource();
  if (auto relation = dyn_cast<RelationOp>(operation))
    return relation.getSource();
  if (auto fold = dyn_cast<FoldOp>(operation))
    return fold.getSource();
  return cast<ContractOp>(operation).getSource();
}

DictionaryAttr algebraSourceExpansion(Operation *operation) {
  return operation->getAttrOfType<DictionaryAttr>("source_group");
}

bool hasIdentityResultMap(MapOp map) {
  if (map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != map.getInputs().size() + 1)
    return false;
  auto resultMap =
      dyn_cast<AffineMapAttr>(map.getIndexingMaps().getValue().back());
  return resultMap && resultMap.getValue().isIdentity();
}

bool isEligibleF32Map(MapOp map) {
  if (!hasIdentityResultMap(map))
    return false;
  FailureOr<SmallVector<int64_t>> domain = mapDomainShape(map);
  if (failed(domain))
    return false;
  return llvm::all_of(
      llvm::concat<Type>(map.getOperandTypes(), map.getResultTypes()),
      [](Type type) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        return tensor && tensor.hasStaticShape() &&
               tensor.getElementType().isF32();
      });
}

bool isEligibleFloatingMap(MapOp map) {
  if (!hasIdentityResultMap(map))
    return false;
  FailureOr<SmallVector<int64_t>> domain = genericMapDomainShape(map);
  if (failed(domain))
    return false;
  return llvm::all_of(
      llvm::concat<Type>(map.getOperandTypes(), map.getResultTypes()),
      [](Type type) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        return tensor && tensor.hasStaticShape() &&
               (tensor.getElementType().isBF16() ||
                tensor.getElementType().isF32());
      });
}

bool isEligibleFoldMapEdge(FoldOp fold, MapOp map, unsigned operandNumber) {
  if (!isClosedAdditivePlanningFold(fold) ||
      !hasExactPositiveZeroInitializer(fold) || fold.getNumResults() != 1 ||
      !fold.getResult(0).hasOneUse() || !isEligibleF32Map(map) ||
      operandNumber >= map.getInputs().size())
    return false;
  auto folded = dyn_cast<RankedTensorType>(fold.getResult(0).getType());
  auto output = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  auto edge = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[operandNumber]);
  if (!folded || !output || !edge ||
      edge.getValue().getNumDims() != output.getRank() ||
      edge.getValue().getNumSymbols() != 0 ||
      edge.getValue().getNumResults() != folded.getRank())
    return false;
  auto elementCount = [](RankedTensorType type) -> FailureOr<int64_t> {
    int64_t count = 1;
    for (int64_t extent : type.getShape()) {
      if (extent <= 0 || count > std::numeric_limits<int64_t>::max() / extent)
        return failure();
      count *= extent;
    }
    return count;
  };
  FailureOr<int64_t> foldedElements = elementCount(folded);
  FailureOr<int64_t> outputElements = elementCount(output);
  if (failed(foldedElements) || failed(outputElements) ||
      *foldedElements != *outputElements)
    return false;
  llvm::SmallDenseSet<unsigned> mapped;
  std::optional<unsigned> previousDimension;
  for (auto [axis, expression] :
       llvm::enumerate(edge.getValue().getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() >= output.getRank() ||
        !mapped.insert(dimension.getPosition()).second ||
        (previousDimension && dimension.getPosition() <= *previousDimension) ||
        folded.getDimSize(axis) != output.getDimSize(dimension.getPosition()))
      return false;
    previousDimension = dimension.getPosition();
  }
  return llvm::all_of(llvm::enumerate(output.getShape()), [&](auto item) {
    return mapped.contains(item.index()) || item.value() == 1;
  });
}

std::string fusedMapFingerprint(ArrayRef<Operation *> operations) {
  std::string normalized("map-dag-v1\n");
  for (Operation *operation : operations) {
    normalized.append(semanticTaskFingerprint(operation));
    normalized.push_back('\n');
  }
  return sha256(normalized);
}

std::string fusedFoldMapFingerprint(FoldOp fold, ArrayRef<Operation *> maps) {
  std::string normalized("fold-map-v1\n");
  normalized.append(semanticTaskFingerprint(fold));
  normalized.push_back('\n');
  for (Operation *operation : maps) {
    normalized.append(semanticTaskFingerprint(operation));
    normalized.push_back('\n');
  }
  return sha256(normalized);
}

std::string closedRegionFingerprint(StringRef family,
                                    ArrayRef<Operation *> operations) {
  std::string normalized = (llvm::Twine(family) + "\n").str();
  llvm::SmallDenseMap<Operation *, int64_t> operationOrdinals;
  for (auto [ordinal, operation] : llvm::enumerate(operations))
    operationOrdinals.try_emplace(operation, ordinal);
  for (auto [ordinal, operation] : llvm::enumerate(operations)) {
    normalized.append(semanticTaskFingerprint(operation));
    normalized.push_back('\n');
    for (auto [operandOrdinal, operand] :
         llvm::enumerate(operation->getOperands())) {
      normalized.append((llvm::Twine("edge:") + llvm::Twine(ordinal) + ":" +
                         llvm::Twine(operandOrdinal) + ":")
                            .str());
      if (auto argument = dyn_cast<BlockArgument>(operand)) {
        normalized.append(
            (llvm::Twine("argument:") + llvm::Twine(argument.getArgNumber()))
                .str());
      } else if (auto result = dyn_cast<OpResult>(operand)) {
        Operation *producer = result.getOwner();
        auto producerOrdinal = operationOrdinals.find(producer);
        if (producerOrdinal != operationOrdinals.end()) {
          normalized.append((llvm::Twine("source:") +
                             llvm::Twine(producerOrdinal->second) + ":" +
                             llvm::Twine(result.getResultNumber()))
                                .str());
        } else {
          normalized.append("external:");
          if (Attribute source = producer->getAttr("source")) {
            std::string sourceText;
            llvm::raw_string_ostream sourceStream(sourceText);
            source.print(sourceStream);
            sourceStream.flush();
            normalized.append(sourceText);
          } else {
            normalized.append(producer->getName().getStringRef());
          }
          normalized.append(
              (llvm::Twine(":") + llvm::Twine(result.getResultNumber())).str());
        }
      } else {
        normalized.append("unknown");
      }
      normalized.push_back('\n');
    }
  }
  return sha256(normalized);
}

std::string foldRegionFingerprint(ArrayRef<Operation *> operations) {
  return closedRegionFingerprint("fold-region-v2", operations);
}

std::string contractRegionFingerprint(ArrayRef<Operation *> operations) {
  return closedRegionFingerprint("contract-region-v2", operations);
}

std::string compositionFingerprint(StringRef kind,
                                   ArrayRef<Operation *> operations) {
  return closedRegionFingerprint((llvm::Twine("composition-") + kind).str(),
                                 operations);
}

bool valueDependsOnOperation(Value value, Operation *operation) {
  llvm::SmallDenseSet<Operation *> visited;
  SmallVector<Value> worklist{value};
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (producer == operation)
      return true;
    llvm::append_range(worklist, producer->getOperands());
  }
  return false;
}

bool valueDependsOnValue(Value value, Value ancestor) {
  llvm::SmallDenseSet<Value> visited;
  SmallVector<Value> worklist{value};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (current == ancestor)
      return true;
    if (!visited.insert(current).second)
      continue;
    Operation *producer = current.getDefiningOp();
    if (producer)
      llvm::append_range(worklist, producer->getOperands());
  }
  return false;
}

bool isRneF32ToBf16Map(MapOp map) {
  if (map.getSemantics() != MapSemantics::Pointwise ||
      map.getInputs().size() != 1 || map.getNumResults() != 1 ||
      !isClosedDirectProjectedMap(map))
    return false;
  auto input = dyn_cast<RankedTensorType>(map.getInputs().front().getType());
  auto result = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  if (!input || !result || input.getShape() != result.getShape() ||
      !input.getElementType().isF32() || !result.getElementType().isBF16())
    return false;
  Block &body = map.getBody().front();
  auto conversion = body.empty() ? ScalarConvertOp{}
                                 : dyn_cast<ScalarConvertOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  return body.getNumArguments() == 1 && body.getOperations().size() == 2 &&
         conversion && yield && conversion.getInput() == body.getArgument(0) &&
         conversion.getSemantics() ==
             ScalarConvertSemantics::RoundNearestEven &&
         yield.getValues().size() == 1 &&
         yield.getValues().front() == conversion.getResult();
}

FailureOr<int64_t> checkedShapeElements(ArrayRef<int64_t> shape) {
  int64_t elements = 1;
  for (int64_t extent : shape) {
    if (extent <= 0 || elements > std::numeric_limits<int64_t>::max() / extent)
      return failure();
    elements *= extent;
  }
  return elements;
}

FailureOr<int64_t> checkedTensorBytes(RankedTensorType type) {
  FailureOr<int64_t> elements = checkedShapeElements(type.getShape());
  if (!type.getElementType().isBF16() && !type.getElementType().isF32())
    return failure();
  const int64_t elementBytes = type.getElementType().isBF16() ? 2 : 4;
  if (failed(elements) ||
      *elements > std::numeric_limits<int64_t>::max() / elementBytes)
    return failure();
  return *elements * elementBytes;
}

FailureOr<int64_t> checkedPlanningTensorBytes(RankedTensorType type) {
  FailureOr<int64_t> elements = checkedShapeElements(type.getShape());
  Type elementType = type.getElementType();
  if (failed(elements) || !elementType.isIntOrFloat())
    return failure();
  const unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  const int64_t elementBytes = std::max<int64_t>(1, (bitWidth + 7) / 8);
  if (*elements > std::numeric_limits<int64_t>::max() / elementBytes)
    return failure();
  return *elements * elementBytes;
}

#include "OnlineCompositionPlanner.inc"

bool isProjectedFoldStatistic(MapOp map, unsigned operand,
                              RankedTensorType full, RankedTensorType compact) {
  if (operand >= map.getInputs().size() ||
      map.getInputs()[operand].getType() != compact ||
      map.getResult(0).getType() != full ||
      compact.getShape() != full.getShape().drop_back())
    return false;
  auto indexing = dyn_cast<AffineMapAttr>(map.getIndexingMaps()[operand]);
  if (!indexing || indexing.getValue().getNumDims() != full.getRank() ||
      indexing.getValue().getNumSymbols() != 0 ||
      indexing.getValue().getNumResults() != compact.getRank())
    return false;
  for (auto [axis, expression] :
       llvm::enumerate(indexing.getValue().getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() != axis)
      return false;
  }
  return true;
}

FailureOr<DerivedPlan>
deriveGenericContractFoldCompositionPlan(RegionOp region,
                                         const DerivedPlan &materialized) {
  SmallVector<Operation *> algebra;
  for (Operation &operation : region.getBody().front().without_terminator())
    algebra.push_back(&operation);
  auto regionYield =
      dyn_cast<YieldOp>(region.getBody().front().getTerminator());
  if (!regionYield || regionYield.getValues().empty()) {
    return failure();
  }

  FoldOp fold;
  SmallVector<FoldOp> folds;
  for (Operation *operation : algebra) {
    if (auto candidate = dyn_cast<FoldOp>(operation)) {
      fold = candidate;
      folds.push_back(candidate);
      continue;
    }
    if (!isa<MapOp, RelationOp, ContractOp>(operation)) {
      return failure();
    }
  }
  if (!fold || !isClosedMultiStatePlanningFold(fold)) {
    return failure();
  }
  const bool sequentialFoldRegion = folds.size() > 1;
  if (sequentialFoldRegion) {
    for (auto [ordinal, candidate] : llvm::enumerate(folds)) {
      if (!isClosedAdditivePlanningFold(candidate) ||
          !hasExactPositiveZeroInitializer(candidate) ||
          candidate.getNumResults() != 1)
        return failure();
      if (ordinal == 0)
        continue;
      FoldOp previous = folds[ordinal - 1];
      if (candidate.getInputs().size() != 1 ||
          candidate.getInputs().front().getType() !=
              previous.getInputs().front().getType() ||
          candidate.getReductionDimensions() !=
              previous.getReductionDimensions() ||
          candidate.getResult(0).getType() != previous.getResult(0).getType() ||
          !valueDependsOnOperation(candidate.getInputs().front(), previous))
        return failure();
    }
  }

  llvm::SmallDenseSet<Operation *> producerSet;
  SmallVector<Value> producerWorklist(fold.getInputs().begin(),
                                      fold.getInputs().end());
  llvm::append_range(producerWorklist, fold.getInitializers());
  while (!producerWorklist.empty()) {
    Operation *producer = producerWorklist.pop_back_val().getDefiningOp();
    if (!producer)
      continue;
    if (producer->getParentOfType<RegionOp>() != region ||
        !(isa<MapOp, RelationOp, ContractOp>(producer) ||
          (sequentialFoldRegion && isa<FoldOp>(producer)))) {
      return failure();
    }
    if (!producerSet.insert(producer).second)
      continue;
    llvm::append_range(producerWorklist, producer->getOperands());
  }
  producerSet.insert(fold);

  SmallVector<Operation *> producerSources;
  SmallVector<Operation *> consumerSources;
  for (Operation *operation : algebra) {
    if (producerSet.contains(operation))
      producerSources.push_back(operation);
    else
      consumerSources.push_back(operation);
  }
  if (producerSources.empty() || producerSources.back() != fold ||
      consumerSources.empty()) {
    return failure();
  }

  llvm::SmallDenseSet<Operation *> consumerSet;
  consumerSet.insert(consumerSources.begin(), consumerSources.end());
  unsigned consumerContracts = 0;
  ContractOp consumerContract;
  for (Operation *operation : consumerSources) {
    if (auto map = dyn_cast<MapOp>(operation)) {
      if (!isClosedFoldRegionMap(map)) {
        return failure();
      }
    } else if (auto contract = dyn_cast<ContractOp>(operation)) {
      if (!isClosedProjectedContract(contract) ||
          failed(contractDomainShape(contract)) || ++consumerContracts > 1) {
        return failure();
      }
      consumerContract = contract;
    } else if (!isa<RelationOp>(operation)) {
      return failure();
    }
    SmallVector<Value> taskOperands;
    if (auto fold = dyn_cast<FoldOp>(operation)) {
      llvm::append_range(taskOperands, fold.getInputs());
      llvm::append_range(taskOperands, fold.getInitializers());
    } else {
      llvm::append_range(taskOperands, operation->getOperands());
    }
    for (Value operand : taskOperands) {
      Operation *producer = operand.getDefiningOp();
      if (producer && producer->getParentOfType<RegionOp>() == region &&
          !producerSet.contains(producer) && !consumerSet.contains(producer)) {
        return failure();
      }
    }
  }
  for (Operation *operation : producerSources) {
    if (isa<FoldOp>(operation))
      continue;
    if (auto map = dyn_cast<MapOp>(operation)) {
      if (!isClosedFoldRegionMap(map)) {
        return failure();
      }
    } else if (auto contract = dyn_cast<ContractOp>(operation)) {
      if (!isClosedProjectedContract(contract) ||
          failed(contractDomainShape(contract))) {
        return failure();
      }
    } else if (!isa<RelationOp>(operation)) {
      return failure();
    }
  }

  Operation *terminal = nullptr;
  for (Value yielded : regionYield.getValues()) {
    Operation *producer = yielded.getDefiningOp();
    if (!producer || !consumerSet.contains(producer) ||
        !valueDependsOnOperation(yielded, fold) ||
        (terminal && terminal != producer)) {
      return failure();
    }
    terminal = producer;
  }
  if (!terminal || terminal != consumerSources.back() ||
      consumerContracts != 1 || !consumerContract ||
      !valueDependsOnOperation(regionYield.getValues().front(),
                               consumerContract)) {
    return failure();
  }

  SmallVector<Value> cutValues;
  llvm::SmallDenseSet<Operation *> recomputedSet;
  for (Operation *operation : producerSources) {
    for (Value result : operation->getResults()) {
      if (!llvm::any_of(result.getUsers(), [&](Operation *user) {
            return consumerSet.contains(user);
          }))
        continue;
      if (!sequentialFoldRegion || isa<FoldOp>(operation)) {
        cutValues.push_back(result);
        continue;
      }
      if (!isa<MapOp>(operation))
        return failure();
      SmallVector<Operation *> worklist{operation};
      while (!worklist.empty()) {
        Operation *recomputed = worklist.pop_back_val();
        if (!recomputedSet.insert(recomputed).second)
          continue;
        if (!isa<MapOp>(recomputed))
          return failure();
        for (Value operand : recomputed->getOperands()) {
          Operation *dependency = operand.getDefiningOp();
          if (dependency && producerSet.contains(dependency) &&
              !isa<FoldOp>(dependency))
            worklist.push_back(dependency);
        }
      }
    }
  }
  if (sequentialFoldRegion) {
    for (FoldOp candidate : folds) {
      for (Value result : candidate.getResults()) {
        if (!llvm::is_contained(cutValues, result) &&
            llvm::any_of(result.getUsers(), [&](Operation *user) {
              return consumerSet.contains(user) || recomputedSet.contains(user);
            }))
          cutValues.push_back(result);
      }
    }
  }
  if (llvm::any_of(folds, [&](FoldOp candidate) {
        return llvm::any_of(candidate.getResults(), [&](Value result) {
          return !llvm::is_contained(cutValues, result);
        });
      })) {
    return failure();
  }
  llvm::SmallDenseSet<Value> cutSet(cutValues.begin(), cutValues.end());
  cutValues.clear();
  for (Operation *operation : producerSources)
    for (Value result : operation->getResults())
      if (cutSet.contains(result))
        cutValues.push_back(result);
  SmallVector<Operation *> recomputedSources;
  for (Operation *operation : producerSources)
    if (recomputedSet.contains(operation))
      recomputedSources.push_back(operation);
  int64_t baselineTemporaryBytes = 0;
  int64_t baselineTemporaryTrafficBytes = 0;
  for (const PlannedBuffer &buffer : materialized.buffers) {
    if (!buffer.producer || buffer.liveOut)
      continue;
    auto type = dyn_cast<RankedTensorType>(buffer.value.getType());
    if (!type)
      return failure();
    FailureOr<int64_t> bytes = checkedPlanningTensorBytes(type);
    const int64_t uses = static_cast<int64_t>(buffer.consumers.size()) + 1;
    if (failed(bytes) ||
        baselineTemporaryBytes > std::numeric_limits<int64_t>::max() - *bytes ||
        uses > std::numeric_limits<int64_t>::max() / *bytes ||
        baselineTemporaryTrafficBytes >
            std::numeric_limits<int64_t>::max() - *bytes * uses)
      return failure();
    baselineTemporaryBytes += *bytes;
    baselineTemporaryTrafficBytes += *bytes * uses;
  }
  int64_t candidateTemporaryBytes = 0;
  for (Value value : cutValues) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    FailureOr<int64_t> bytes =
        type ? checkedPlanningTensorBytes(type) : FailureOr<int64_t>(failure());
    if (failed(bytes) ||
        candidateTemporaryBytes > std::numeric_limits<int64_t>::max() - *bytes)
      return failure();
    candidateTemporaryBytes += *bytes;
  }
  if (candidateTemporaryBytes > std::numeric_limits<int64_t>::max() / 2)
    return failure();
  const int64_t candidateTemporaryTrafficBytes = 2 * candidateTemporaryBytes;
  if (materialized.tasks.size() >
      static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    return failure();
  const int64_t baselineLaunches =
      static_cast<int64_t>(materialized.tasks.size());
  constexpr int64_t candidateLaunches = 2;
  if (candidateLaunches > baselineLaunches ||
      candidateTemporaryBytes > baselineTemporaryBytes ||
      candidateTemporaryTrafficBytes > baselineTemporaryTrafficBytes ||
      !(candidateLaunches < baselineLaunches ||
        candidateTemporaryBytes < baselineTemporaryBytes ||
        candidateTemporaryTrafficBytes < baselineTemporaryTrafficBytes))
    return failure();

  DerivedPlan result;
  result.hasContract = llvm::any_of(
      algebra, [](Operation *operation) { return isa<ContractOp>(operation); });
  FailureOr<SmallVector<int64_t>> foldDomain = planningFoldDomainShape(fold);
  if (failed(foldDomain))
    return failure();
  result.hasRankNFold = foldDomain->size() > 2;
  result.hasContractFoldComposition = true;
  llvm::SmallDenseMap<Value, int64_t> bufferForValue;
  for (BlockArgument argument : region.getBody().front().getArguments()) {
    bufferForValue.try_emplace(argument, result.buffers.size());
    result.buffers.push_back({argument, std::nullopt, {}, false});
  }
  auto addBuffer = [&](Value value, int64_t producer, bool liveOut) {
    int64_t ordinal = result.buffers.size();
    bufferForValue.try_emplace(value, ordinal);
    result.buffers.push_back({value, producer, {}, liveOut});
    return ordinal;
  };
  SmallVector<int64_t> cutBuffers;
  for (Value value : cutValues)
    cutBuffers.push_back(addBuffer(value, 0, false));
  SmallVector<int64_t> resultBuffers;
  for (Value value : regionYield.getValues())
    resultBuffers.push_back(addBuffer(value, 1, true));

  auto taskBlockInputs = [&](ArrayRef<Operation *> sources) {
    llvm::SmallDenseSet<Value> used;
    for (Operation *operation : sources)
      for (Value operand : operation->getOperands())
        if (isa<BlockArgument>(operand))
          used.insert(operand);
    SmallVector<int64_t> inputs;
    for (BlockArgument argument : region.getBody().front().getArguments())
      if (used.contains(argument))
        inputs.push_back(bufferForValue.lookup(argument));
    return inputs;
  };

  PlannedTask firstTask;
  firstTask.operation = sequentialFoldRegion ? folds.front() : fold;
  firstTask.kind = MaterializationTaskKind::Fold;
  FailureOr<SmallVector<int64_t>> firstFoldDomain =
      planningFoldDomainShape(cast<FoldOp>(firstTask.operation));
  if (failed(firstFoldDomain))
    return failure();
  firstTask.domain = *firstFoldDomain;
  firstTask.reductionDimensions.assign(
      cast<FoldOp>(firstTask.operation).getReductionDimensions().begin(),
      cast<FoldOp>(firstTask.operation).getReductionDimensions().end());
  firstTask.orderFree = cast<FoldOp>(firstTask.operation).getOrderFree();
  firstTask.inputs = taskBlockInputs(producerSources);
  firstTask.outputs = cutBuffers;
  if (sequentialFoldRegion) {
    llvm::SmallDenseSet<Operation *> firstTaskSet(producerSources.begin(),
                                                  producerSources.end());
    firstTask.foldRegionSources = producerSources;
    for (FoldOp candidate : folds) {
      firstTask.foldRegionPhases.push_back(candidate);
      firstTask.foldRegionOutputs.push_back(candidate);
    }
    for (Operation *operation : producerSources) {
      auto map = dyn_cast<MapOp>(operation);
      if (!map)
        continue;
      for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
        if (firstTaskSet.contains(operand.getDefiningOp()))
          continue;
        auto buffer = bufferForValue.find(operand);
        if (buffer == bufferForValue.end())
          return failure();
        int64_t phase = llvm::count_if(folds, [&](FoldOp candidate) {
          return candidate->isBeforeInBlock(map);
        });
        phase = std::min<int64_t>(phase, folds.size() - 1);
        firstTask.foldRegionBoundaryUses.push_back(
            {buffer->second, map, static_cast<unsigned>(operandOrdinal),
             cast<AffineMapAttr>(map.getIndexingMaps()[operandOrdinal])
                 .getValue(),
             phase});
      }
    }
    if (firstTask.foldRegionBoundaryUses.empty())
      return failure();
  } else {
    firstTask.compositionKind = "contract_fold";
    firstTask.compositionSources = producerSources;
  }

  PlannedTask secondTask;
  secondTask.operation = terminal;
  secondTask.kind = MaterializationTaskKind::Contract;
  secondTask.domain = *contractDomainShape(consumerContract);
  secondTask.reductionDimensions = {
      static_cast<int64_t>(secondTask.domain.size()) - 1};
  secondTask.inputs = cutBuffers;
  SmallVector<Operation *> consumerExecutionSources(consumerSources);
  llvm::append_range(consumerExecutionSources, recomputedSources);
  llvm::append_range(secondTask.inputs,
                     taskBlockInputs(consumerExecutionSources));
  secondTask.outputs = resultBuffers;
  secondTask.dependencies = {0};
  secondTask.compositionKind = "map_contract";
  secondTask.compositionSources = consumerSources;
  secondTask.compositionRecomputedSources = recomputedSources;
  const int64_t producerContractEvaluations =
      llvm::count_if(producerSources, [](Operation *operation) {
        return isa<ContractOp>(operation);
      });
  secondTask.compositionCost = {baselineLaunches,
                                candidateLaunches,
                                baselineTemporaryBytes,
                                candidateTemporaryBytes,
                                baselineTemporaryTrafficBytes,
                                candidateTemporaryTrafficBytes,
                                producerContractEvaluations};
  result.tasks = {std::move(firstTask), std::move(secondTask)};
  for (auto [taskOrdinal, task] : llvm::enumerate(result.tasks))
    for (int64_t input : task.inputs)
      result.buffers[input].consumers.push_back(taskOrdinal);
  if (std::optional<PlannedNormalizedExpContractRegion> capability =
          online_planner::normalizedExponentialContractCapability(
              region, producerSources, consumerSources, fold,
              consumerContract)) {
    result.normalizedExpContractRegions.push_back(std::move(*capability));
  }
  return result;
}

FailureOr<DerivedPlan> deriveNormContractSubplan(RegionOp region,
                                                 ArrayRef<Operation *> algebra,
                                                 Value yieldedValue) {
  SmallVector<FoldOp> folds;
  SmallVector<ContractOp> contracts;
  SmallVector<MapOp> reciprocalMaps;
  for (Operation *candidate : algebra) {
    Operation &operation = *candidate;
    if (auto map = dyn_cast<MapOp>(operation)) {
      if (!isClosedFoldRegionMap(map))
        return failure();
      int64_t reciprocalCount = 0;
      map.walk([&](math::RsqrtOp) { ++reciprocalCount; });
      if (reciprocalCount > 1)
        return failure();
      if (reciprocalCount == 1)
        reciprocalMaps.push_back(map);
    } else if (auto fold = dyn_cast<FoldOp>(operation)) {
      folds.push_back(fold);
    } else if (auto contract = dyn_cast<ContractOp>(operation)) {
      contracts.push_back(contract);
    } else {
      return failure();
    }
  }
  if (folds.size() != 1 || contracts.size() != 1 || reciprocalMaps.size() != 1)
    return failure();
  FoldOp fold = folds.front();
  ContractOp contract = contracts.front();
  MapOp reciprocal = reciprocalMaps.front();
  MapOp boundary = contract.getInputs().front().getDefiningOp<MapOp>();
  MapOp epilogue = yieldedValue.getDefiningOp<MapOp>();
  if (!isClosedAdditivePlanningFold(fold) ||
      !hasExactPositiveZeroInitializer(fold) || fold.getNumResults() != 1 ||
      !isClosedProjectedContract(contract) || !boundary || !epilogue ||
      !isRneF32ToBf16Map(boundary) || !isRneF32ToBf16Map(epilogue) ||
      epilogue.getInputs().front() != contract.getResult(0) ||
      !valueDependsOnOperation(reciprocal.getResult(0), fold) ||
      !valueDependsOnOperation(boundary.getResult(0), reciprocal) ||
      !valueDependsOnOperation(boundary.getResult(0), fold))
    return failure();
  auto activation = dyn_cast<RankedTensorType>(boundary.getResult(0).getType());
  auto foldInput =
      dyn_cast<RankedTensorType>(fold.getInputs().front().getType());
  auto foldResult = dyn_cast<RankedTensorType>(fold.getResult(0).getType());
  auto scale = dyn_cast<RankedTensorType>(reciprocal.getResult(0).getType());
  auto projection =
      dyn_cast<RankedTensorType>(contract.getInputs()[1].getType());
  auto contractResult =
      dyn_cast<RankedTensorType>(contract.getResult(0).getType());
  auto result = dyn_cast<RankedTensorType>(epilogue.getResult(0).getType());
  if (!activation || !foldInput || !foldResult || !scale || !projection ||
      !contractResult || !result || activation.getRank() < 2 ||
      activation.getRank() > 8 ||
      activation.getShape() != foldInput.getShape() ||
      !activation.getElementType().isBF16() ||
      !foldInput.getElementType().isF32() ||
      fold.getReductionDimensions() !=
          ArrayRef<int64_t>({foldInput.getRank() - 1}) ||
      foldResult.getShape() != activation.getShape().drop_back() ||
      scale.getShape().drop_back() != foldResult.getShape() ||
      scale.getShape().back() != 1 || !scale.getElementType().isF32() ||
      projection.getRank() != 2 ||
      projection.getDimSize(0) !=
          activation.getDimSize(activation.getRank() - 1) ||
      !projection.getElementType().isBF16() ||
      contractResult.getShape() != result.getShape() ||
      result.getShape().drop_back() != activation.getShape().drop_back() ||
      !contractResult.getElementType().isF32() ||
      !result.getElementType().isBF16())
    return failure();
  const unsigned activationRank = activation.getRank();
  const unsigned contractRank = activationRank + 1;
  MLIRContext *context = region.getContext();
  SmallVector<AffineExpr> lhsExpressions;
  for (unsigned dimension = 0; dimension + 1 < activationRank; ++dimension)
    lhsExpressions.push_back(getAffineDimExpr(dimension, context));
  lhsExpressions.push_back(getAffineDimExpr(contractRank - 1, context));
  AffineMap expectedLhs =
      AffineMap::get(contractRank, 0, lhsExpressions, context);
  AffineMap expectedRhs =
      AffineMap::get(contractRank, 0,
                     {getAffineDimExpr(contractRank - 1, context),
                      getAffineDimExpr(contractRank - 2, context)},
                     context);
  SmallVector<AffineExpr> resultExpressions;
  for (unsigned dimension = 0; dimension + 1 < contractRank; ++dimension)
    resultExpressions.push_back(getAffineDimExpr(dimension, context));
  AffineMap expectedResult =
      AffineMap::get(contractRank, 0, resultExpressions, context);
  auto epilogueInputMap =
      dyn_cast<AffineMapAttr>(epilogue.getIndexingMaps()[0]);
  if (cast<AffineMapAttr>(contract.getIndexingMaps()[0]).getValue() !=
          expectedLhs ||
      cast<AffineMapAttr>(contract.getIndexingMaps()[1]).getValue() !=
          expectedRhs ||
      cast<AffineMapAttr>(contract.getIndexingMaps()[2]).getValue() !=
          expectedResult ||
      !epilogueInputMap || !isIdentityMap(epilogueInputMap.getValue()))
    return failure();

  auto collect = [&](Value value, Operation *stop,
                     llvm::SmallDenseSet<Operation *> &operations,
                     auto &self) -> LogicalResult {
    Operation *producer = value.getDefiningOp();
    if (!producer || producer == stop)
      return success();
    if (producer->getParentOfType<RegionOp>() != region ||
        !isa<MapOp, FoldOp, ContractOp>(producer))
      return failure();
    if (!operations.insert(producer).second)
      return success();
    for (Value operand : producer->getOperands())
      if (failed(self(operand, stop, operations, self)))
        return failure();
    return success();
  };
  llvm::SmallDenseSet<Operation *> allSources;
  if (failed(collect(yieldedValue, nullptr, allSources, collect)) ||
      allSources.size() != algebra.size())
    return failure();
  llvm::SmallDenseSet<Operation *> foldSources;
  if (failed(collect(reciprocal.getResult(0), nullptr, foldSources, collect)))
    return failure();
  llvm::SmallDenseSet<Operation *> contractSources;
  if (failed(collect(yieldedValue, reciprocal, contractSources, collect)))
    return failure();
  llvm::SmallDenseSet<Operation *> covered(foldSources);
  covered.insert(contractSources.begin(), contractSources.end());
  const bool invalidOverlap =
      llvm::any_of(foldSources, [&](Operation *operation) {
        return contractSources.contains(operation) && !isa<MapOp>(operation);
      });
  if (covered.size() != algebra.size() || invalidOverlap)
    return failure();

  llvm::SmallDenseSet<Value> foldExternal;
  llvm::SmallDenseSet<Value> boundaryExternal;
  auto collectExternal = [&](Value root, llvm::SmallDenseSet<Value> &external) {
    llvm::SmallDenseSet<Operation *> visited;
    SmallVector<Value> worklist{root};
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      Operation *producer = value.getDefiningOp();
      if (!producer) {
        external.insert(value);
        continue;
      }
      if (!visited.insert(producer).second)
        continue;
      llvm::append_range(worklist, producer->getOperands());
    }
  };
  collectExternal(fold.getInputs().front(), foldExternal);
  collectExternal(boundary.getResult(0), boundaryExternal);
  Value projectionValue = contract.getInputs()[1];
  if (projectionValue.getDefiningOp() || foldExternal.size() != 1 ||
      boundaryExternal.size() != 2 ||
      boundaryExternal.contains(projectionValue))
    return failure();
  Value activationValue = *foldExternal.begin();
  if (!boundaryExternal.contains(activationValue))
    return failure();
  Value weightValue = *llvm::find_if(
      boundaryExternal, [&](Value value) { return value != activationValue; });
  auto activationArgument = dyn_cast<BlockArgument>(activationValue);
  auto weightArgument = dyn_cast<BlockArgument>(weightValue);
  auto projectionArgument = dyn_cast<BlockArgument>(projectionValue);
  auto activationBoundary =
      dyn_cast<RankedTensorType>(activationValue.getType());
  auto weightBoundary = dyn_cast<RankedTensorType>(weightValue.getType());
  if (!activationArgument || !weightArgument || !projectionArgument ||
      activationArgument.getOwner() != &region.getBody().front() ||
      weightArgument.getOwner() != &region.getBody().front() ||
      projectionArgument.getOwner() != &region.getBody().front() ||
      activationArgument == weightArgument ||
      activationArgument == projectionArgument ||
      weightArgument == projectionArgument || !activationBoundary ||
      !weightBoundary || activationBoundary != activation ||
      weightBoundary.getRank() != 1 ||
      weightBoundary.getDimSize(0) != activation.getShape().back() ||
      !weightBoundary.getElementType().isBF16())
    return failure();

  FailureOr<int64_t> rows =
      checkedShapeElements(activation.getShape().drop_back());
  if (failed(rows))
    return failure();
  const int64_t reduction = activation.getShape().back();
  const int64_t columns = result.getShape().back();
  const NumericalPolicy policy = region.getPolicy();
  const int64_t capacity = policy == NumericalPolicy::Fast ? 8 : 256;
  const int64_t outputTile = std::min(columns, capacity);
  const int64_t outputTiles = ceilDivPositive(columns, outputTile);
  const bool inlineFold = outputTiles == 1 && reduction <= 256;
  auto checkedProduct =
      [](std::initializer_list<int64_t> factors) -> FailureOr<int64_t> {
    int64_t product = 1;
    for (int64_t factor : factors) {
      if (factor < 0 ||
          (factor != 0 &&
           product > std::numeric_limits<int64_t>::max() / factor))
        return failure();
      product *= factor;
    }
    return product;
  };
  FailureOr<int64_t> duplicated =
      checkedProduct({*rows, reduction, outputTiles - 1});
  FailureOr<int64_t> normalizedBytes = checkedProduct({2, *rows, reduction});
  FailureOr<int64_t> contractBytes = checkedProduct({4, *rows, columns});
  FailureOr<int64_t> compactBytes = checkedProduct({4, *rows});
  if (failed(duplicated) || failed(normalizedBytes) || failed(contractBytes) ||
      failed(compactBytes))
    return failure();

  SmallVector<Operation *> orderedFoldSources;
  SmallVector<Operation *> orderedContractSources;
  for (Operation *operation : algebra) {
    if (foldSources.contains(operation))
      orderedFoldSources.push_back(operation);
    if (contractSources.contains(operation))
      orderedContractSources.push_back(operation);
  }
  DerivedPlan plan;
  plan.hasContract = true;
  plan.hasContractRegion = true;
  llvm::SmallDenseMap<Value, int64_t> bufferForValue;
  for (Value argument : {activationValue, weightValue, projectionValue}) {
    bufferForValue.try_emplace(argument, plan.buffers.size());
    plan.buffers.push_back({argument, std::nullopt, {}, false});
  }
  int64_t compactBuffer = -1;
  if (!inlineFold) {
    compactBuffer = plan.buffers.size();
    bufferForValue.try_emplace(reciprocal.getResult(0), compactBuffer);
    plan.buffers.push_back({reciprocal.getResult(0), int64_t{0}, {}, false});
  }
  int64_t outputBuffer = plan.buffers.size();
  bufferForValue.try_emplace(yieldedValue, outputBuffer);
  plan.buffers.push_back(
      {yieldedValue, inlineFold ? int64_t{0} : int64_t{1}, {}, true});

  auto boundaryValues = [&](ArrayRef<Operation *> sources,
                            Operation *materializedStop) {
    llvm::SmallDenseSet<Operation *> sourceSet(sources.begin(), sources.end());
    SmallVector<Value> values;
    llvm::SmallDenseSet<Value> seen;
    for (Operation *operation : sources) {
      for (Value operand : operation->getOperands()) {
        if (operand.getDefiningOp() == materializedStop ||
            !sourceSet.contains(operand.getDefiningOp())) {
          if (seen.insert(operand).second)
            values.push_back(operand);
        }
      }
    }
    return values;
  };
  auto fillContractMetadata = [&](PlannedTask &task,
                                  ArrayRef<Operation *> sources) {
    task.contractRegionSources.assign(sources.begin(), sources.end());
    task.contractRegionStrategy = inlineFold ? "inline_fold" : "compact_stat";
    task.contractRegionFold = fold;
    task.contractRegionContract = contract;
    task.contractRegionBf16Boundary = boundary;
    task.contractRegionEpilogueSources.push_back(epilogue);
    task.contractRegionNormalizedOperand = 0;
    task.contractRegionCost = {*rows,
                               reduction,
                               columns,
                               outputTile,
                               outputTiles,
                               *duplicated,
                               *normalizedBytes,
                               *contractBytes,
                               inlineFold ? 0 : *compactBytes};
  };
  if (!inlineFold) {
    PlannedTask foldTask;
    foldTask.operation = fold;
    foldTask.kind = MaterializationTaskKind::Fold;
    foldTask.domain.assign(foldInput.getShape().begin(),
                           foldInput.getShape().end());
    foldTask.reductionDimensions.assign(fold.getReductionDimensions().begin(),
                                        fold.getReductionDimensions().end());
    foldTask.orderFree = fold.getOrderFree();
    foldTask.foldRegionSources = orderedFoldSources;
    foldTask.foldRegionPhases.push_back(fold);
    for (Value value : boundaryValues(orderedFoldSources, nullptr)) {
      auto found = bufferForValue.find(value);
      if (found == bufferForValue.end())
        return failure();
      if (!llvm::is_contained(foldTask.inputs, found->second))
        foldTask.inputs.push_back(found->second);
    }
    for (Operation *operation : orderedFoldSources) {
      auto map = dyn_cast<MapOp>(operation);
      if (!map)
        continue;
      for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
        if (foldSources.contains(operand.getDefiningOp()))
          continue;
        auto found = bufferForValue.find(operand);
        if (found == bufferForValue.end())
          return failure();
        foldTask.foldRegionBoundaryUses.push_back(
            {found->second, map, static_cast<unsigned>(operandOrdinal),
             cast<AffineMapAttr>(map.getIndexingMaps()[operandOrdinal])
                 .getValue(),
             0});
      }
    }
    foldTask.outputs.push_back(compactBuffer);
    foldTask.foldRegionOutputs.push_back(reciprocal);
    plan.tasks.push_back(std::move(foldTask));
  }

  PlannedTask contractTask;
  contractTask.operation = contract;
  contractTask.kind = MaterializationTaskKind::Contract;
  FailureOr<SmallVector<int64_t>> contractDomain =
      contractDomainShape(contract);
  if (failed(contractDomain))
    return failure();
  contractTask.domain = std::move(*contractDomain);
  contractTask.reductionDimensions.push_back(contractTask.domain.size() - 1);
  ArrayRef<Operation *> contractTaskSources =
      inlineFold ? ArrayRef<Operation *>(algebra)
                 : ArrayRef<Operation *>(orderedContractSources);
  fillContractMetadata(contractTask, contractTaskSources);
  SmallVector<Value> actualContractBoundaries = boundaryValues(
      contractTaskSources, inlineFold ? nullptr : reciprocal.getOperation());
  SmallVector<Value> canonicalContractBoundaries{activationValue};
  if (!inlineFold)
    canonicalContractBoundaries.push_back(reciprocal.getResult(0));
  canonicalContractBoundaries.push_back(weightValue);
  canonicalContractBoundaries.push_back(projectionValue);
  llvm::SmallDenseSet<Value> actualContractBoundarySet(
      actualContractBoundaries.begin(), actualContractBoundaries.end());
  if (actualContractBoundarySet.size() != canonicalContractBoundaries.size() ||
      llvm::any_of(canonicalContractBoundaries, [&](Value value) {
        return !actualContractBoundarySet.contains(value);
      }))
    return failure();
  for (Value value : canonicalContractBoundaries) {
    auto found = bufferForValue.find(value);
    if (found == bufferForValue.end())
      return failure();
    if (!llvm::is_contained(contractTask.inputs, found->second))
      contractTask.inputs.push_back(found->second);
  }
  contractTask.outputs.push_back(outputBuffer);
  if (!inlineFold)
    contractTask.dependencies.push_back(0);
  plan.tasks.push_back(std::move(contractTask));

  llvm::SmallDenseMap<Operation *, int64_t> occurrences;
  for (PlannedTask &task : plan.tasks) {
    for (Operation *operation : task.foldRegionSources)
      ++occurrences[operation];
    for (Operation *operation : task.contractRegionSources)
      ++occurrences[operation];
  }
  for (PlannedTask &task : plan.tasks) {
    for (Operation *operation : task.foldRegionSources)
      if (occurrences.lookup(operation) > 1)
        task.foldRegionRecomputedSources.push_back(operation);
    for (Operation *operation : task.contractRegionSources)
      if (occurrences.lookup(operation) > 1)
        task.contractRegionRecomputedSources.push_back(operation);
  }
  for (auto [taskOrdinal, task] : llvm::enumerate(plan.tasks)) {
    for (int64_t input : task.inputs)
      plan.buffers[input].consumers.push_back(taskOrdinal);
  }
  return plan;
}

FailureOr<DerivedPlan> deriveNormContractPlan(RegionOp region) {
  SmallVector<Operation *> algebra;
  for (Operation &operation : region.getBody().front().without_terminator())
    algebra.push_back(&operation);
  auto yield = dyn_cast<YieldOp>(region.getBody().front().getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return failure();
  return deriveNormContractSubplan(region, algebra, yield.getValues().front());
}

FailureOr<DerivedPlan> deriveFoldRegionPlan(RegionOp region) {
  SmallVector<Operation *> algebra;
  SmallVector<FoldOp> folds;
  bool hasReciprocalRoot = false;
  for (Operation &operation : region.getBody().front().without_terminator()) {
    if (auto map = dyn_cast<MapOp>(operation)) {
      if (!isClosedFoldRegionMap(map))
        return failure();
      map.walk([&](math::RsqrtOp) { hasReciprocalRoot = true; });
    } else if (auto fold = dyn_cast<FoldOp>(operation)) {
      if (!isClosedAdditivePlanningFold(fold) ||
          !hasExactPositiveZeroInitializer(fold) || fold.getNumResults() != 1)
        return failure();
      folds.push_back(fold);
    } else {
      return failure();
    }
    algebra.push_back(&operation);
  }
  if (folds.empty() || !hasReciprocalRoot)
    return failure();
  auto regionYield =
      dyn_cast<YieldOp>(region.getBody().front().getTerminator());
  if (!regionYield || regionYield.getValues().empty())
    return failure();

  FoldOp primary;
  for (FoldOp fold : folds) {
    if (llvm::all_of(regionYield.getValues(), [&](Value yielded) {
          return valueDependsOnOperation(yielded, fold);
        })) {
      primary = fold;
      break;
    }
  }
  if (!primary ||
      llvm::is_contained(regionYield.getValues(), primary.getResult(0)))
    return failure();

  auto collectSources = [&](Value root, llvm::SmallDenseSet<Operation *> &set,
                            bool stopAtPrimary, auto &self) -> LogicalResult {
    Operation *producer = root.getDefiningOp();
    if (!producer)
      return success();
    if (producer == primary && stopAtPrimary)
      return success();
    if (!isa<MapOp, FoldOp>(producer) ||
        producer->getParentOfType<RegionOp>() != region)
      return failure();
    if (!set.insert(producer).second)
      return success();
    for (Value operand : producer->getOperands()) {
      if (failed(self(operand, set, stopAtPrimary, self)))
        return failure();
    }
    return success();
  };

  SmallVector<llvm::SmallDenseSet<Operation *>> sourceSets;
  sourceSets.emplace_back();
  for (Value operand : primary->getOperands()) {
    if (failed(
            collectSources(operand, sourceSets.front(), false, collectSources)))
      return failure();
  }
  sourceSets.front().insert(primary);
  SmallVector<SmallVector<Value>> publicOutputs(1);
  for (Value yielded : regionYield.getValues()) {
    llvm::SmallDenseSet<Operation *> branch;
    if (failed(collectSources(yielded, branch, true, collectSources)))
      return failure();
    if (regionYield.getValues().size() == 1) {
      sourceSets.front().insert(branch.begin(), branch.end());
      publicOutputs.front().push_back(yielded);
      continue;
    }
    if (llvm::none_of(branch, [](Operation *operation) {
          return isa<FoldOp>(operation);
        })) {
      llvm::SmallDenseSet<Operation *> complete = sourceSets.front();
      complete.insert(branch.begin(), branch.end());
      sourceSets.push_back(std::move(complete));
      publicOutputs.push_back({yielded});
      continue;
    }
    sourceSets.push_back(std::move(branch));
    publicOutputs.push_back({yielded});
  }

  llvm::SmallDenseSet<Operation *> covered;
  for (const auto &set : sourceSets)
    covered.insert(set.begin(), set.end());
  if (covered.size() != algebra.size())
    return failure();

  SmallVector<SmallVector<Operation *>> orderedSources(sourceSets.size());
  for (auto [taskOrdinal, set] : llvm::enumerate(sourceSets)) {
    for (Operation *operation : algebra) {
      if (set.contains(operation))
        orderedSources[taskOrdinal].push_back(operation);
    }
    SmallVector<FoldOp> phases;
    for (Operation *operation : orderedSources[taskOrdinal]) {
      if (auto fold = dyn_cast<FoldOp>(operation))
        phases.push_back(fold);
    }
    if (phases.empty()) {
      return failure();
    }
    for (auto [phaseOrdinal, fold] : llvm::enumerate(phases)) {
      auto input = cast<RankedTensorType>(fold.getInputs().front().getType());
      if (llvm::any_of(fold.getInputs(), [&](Value value) {
            return !set.contains(value.getDefiningOp());
          }))
        return failure();
      ArrayRef<int64_t> reduction = fold.getReductionDimensions();
      const bool row =
          reduction.size() == 1 && reduction.front() == input.getRank() - 1;
      const bool column =
          reduction.size() == static_cast<size_t>(input.getRank() - 1) &&
          llvm::all_of(llvm::enumerate(reduction),
                       [](auto axis) { return axis.index() == axis.value(); });
      if (!row && !column)
        return failure();
      if (phaseOrdinal == 0)
        continue;
      int64_t cardinality = 1;
      for (int64_t axis : reduction)
        cardinality *= input.getDimSize(axis);
      FoldOp previous = phases[phaseOrdinal - 1];
      auto previousInput =
          cast<RankedTensorType>(previous.getInputs().front().getType());
      const bool degenerateTransition = cardinality == 1;
      const bool sameDomainTransition =
          input == previousInput &&
          fold.getReductionDimensions() == previous.getReductionDimensions();
      if ((!degenerateTransition && !sameDomainTransition) ||
          fold.getResult(0).getType() != previous.getResult(0).getType() ||
          !valueDependsOnOperation(fold.getInputs().front(), previous))
        return failure();
    }
    for (Operation *operation : orderedSources[taskOrdinal]) {
      auto map = dyn_cast<MapOp>(operation);
      if (!map)
        continue;
      for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
        if (isa_and_nonnull<FoldOp>(operand.getDefiningOp()) &&
            set.contains(operand.getDefiningOp()) &&
            !isCanonicalFoldResultExpansion(map, operandOrdinal))
          return failure();
      }
    }
  }

  DerivedPlan result;
  result.hasRankNFold = true;
  llvm::SmallDenseMap<Value, int64_t> bufferForValue;
  for (BlockArgument argument : region.getBody().front().getArguments()) {
    bufferForValue.try_emplace(argument, result.buffers.size());
    result.buffers.push_back({argument, std::nullopt, {}, false});
  }
  if (sourceSets.size() > 1) {
    bufferForValue.try_emplace(primary.getResult(0), result.buffers.size());
    result.buffers.push_back({primary.getResult(0), int64_t{0}, {}, false});
  }
  for (auto [taskOrdinal, outputs] : llvm::enumerate(publicOutputs)) {
    for (Value output : outputs) {
      bufferForValue.try_emplace(output, result.buffers.size());
      result.buffers.push_back(
          {output, static_cast<int64_t>(taskOrdinal), {}, true});
    }
  }

  llvm::SmallDenseMap<Operation *, int64_t> sourceOccurrences;
  for (const auto &set : sourceSets) {
    for (Operation *operation : set)
      ++sourceOccurrences[operation];
  }
  if (llvm::any_of(sourceOccurrences, [](const auto &occurrence) {
        return occurrence.second > 1 && !isa<MapOp, FoldOp>(occurrence.first);
      }))
    return failure();
  for (auto [taskOrdinal, sources] : llvm::enumerate(orderedSources)) {
    PlannedTask task;
    task.foldRegionSources = sources;
    llvm::SmallDenseSet<Operation *> taskSet;
    taskSet.insert(sources.begin(), sources.end());
    for (Operation *operation : sources) {
      if (isa<FoldOp>(operation))
        task.foldRegionPhases.push_back(operation);
      if (sourceOccurrences.lookup(operation) > 1)
        task.foldRegionRecomputedSources.push_back(operation);
    }
    task.operation = task.foldRegionPhases.front();
    task.kind = MaterializationTaskKind::Fold;
    auto firstFold = cast<FoldOp>(task.operation);
    task.domain.assign(
        cast<RankedTensorType>(firstFold.getInputs().front().getType())
            .getShape()
            .begin(),
        cast<RankedTensorType>(firstFold.getInputs().front().getType())
            .getShape()
            .end());
    task.reductionDimensions.assign(firstFold.getReductionDimensions().begin(),
                                    firstFold.getReductionDimensions().end());
    task.orderFree = firstFold.getOrderFree();

    auto addInput = [&](int64_t ordinal) {
      if (!llvm::is_contained(task.inputs, ordinal))
        task.inputs.push_back(ordinal);
    };
    for (Operation *operation : sources) {
      auto map = dyn_cast<MapOp>(operation);
      if (!map)
        continue;
      for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
        if (taskSet.contains(operand.getDefiningOp()))
          continue;
        auto buffer = bufferForValue.find(operand);
        if (buffer == bufferForValue.end())
          return failure();
        addInput(buffer->second);
        int64_t phase = 0;
        for (Operation *phaseOperation : task.foldRegionPhases) {
          auto phaseSource = cast<FoldOp>(phaseOperation).getSource();
          if (std::tuple(phaseSource.getFunctionOrdinal(),
                         phaseSource.getBlockOrdinal(),
                         phaseSource.getOperationOrdinal(),
                         phaseSource.getResultOrdinal()) <
              std::tuple(map.getSource().getFunctionOrdinal(),
                         map.getSource().getBlockOrdinal(),
                         map.getSource().getOperationOrdinal(),
                         map.getSource().getResultOrdinal()))
            ++phase;
        }
        phase = std::min<int64_t>(phase, task.foldRegionPhases.size() - 1);
        task.foldRegionBoundaryUses.push_back(
            {buffer->second, map, static_cast<unsigned>(operandOrdinal),
             cast<AffineMapAttr>(map.getIndexingMaps()[operandOrdinal])
                 .getValue(),
             phase});
      }
    }
    if (task.foldRegionBoundaryUses.empty())
      return failure();
    if (taskOrdinal == 0 && sourceSets.size() > 1) {
      task.outputs.push_back(bufferForValue.lookup(primary.getResult(0)));
      task.foldRegionOutputs.push_back(primary);
    }
    for (Value output : publicOutputs[taskOrdinal]) {
      task.outputs.push_back(bufferForValue.lookup(output));
      task.foldRegionOutputs.push_back(output.getDefiningOp());
    }
    for (int64_t input : task.inputs) {
      if (std::optional<int64_t> producer = result.buffers[input].producer;
          producer && !llvm::is_contained(task.dependencies, *producer))
        task.dependencies.push_back(*producer);
    }
    llvm::sort(task.dependencies);
    result.tasks.push_back(std::move(task));
  }
  for (auto [taskOrdinal, task] : llvm::enumerate(result.tasks)) {
    for (int64_t input : task.inputs)
      result.buffers[input].consumers.push_back(taskOrdinal);
  }
  result.hasFoldRegion = true;
  return result;
}

FailureOr<int64_t> scheduleElementCount(ArrayRef<int64_t> shape);
FailureOr<int64_t> staticElementCount(RankedTensorType type);

FailureOr<DerivedPlan> deriveSharedStatFoldRegionPlan(RegionOp region) {
  SmallVector<Operation *> algebra;
  SmallVector<FoldOp> folds;
  for (Operation &operation : region.getBody().front().without_terminator()) {
    if (auto map = dyn_cast<MapOp>(operation)) {
      if (!isClosedFoldRegionMap(map))
        return failure();
    } else if (auto fold = dyn_cast<FoldOp>(operation)) {
      if (!isClosedAdditivePlanningFold(fold) ||
          !hasExactPositiveZeroInitializer(fold) || fold.getNumResults() != 1)
        return failure();
      folds.push_back(fold);
    } else {
      return failure();
    }
    algebra.push_back(&operation);
  }
  if (folds.empty())
    return failure();
  auto regionYield =
      dyn_cast<YieldOp>(region.getBody().front().getTerminator());
  if (!regionYield || regionYield.getValues().size() < 3)
    return failure();

  RankedTensorType domain;
  for (FoldOp fold : folds) {
    auto input = cast<RankedTensorType>(fold.getInputs().front().getType());
    if (fold.getReductionDimensions() ==
        ArrayRef<int64_t>({input.getRank() - 1})) {
      domain = input;
      break;
    }
  }
  if (!domain || domain.getRank() < 2)
    return failure();
  SmallVector<int64_t> rowShape(domain.getShape().drop_back());
  SmallVector<int64_t> featureShape(domain.getShape().take_back());
  SmallVector<Value> rowOutputs;
  SmallVector<Value> featureOutputs;
  llvm::SmallDenseMap<Value, int64_t> outputTask;
  for (Value output : regionYield.getValues()) {
    auto type = dyn_cast<RankedTensorType>(output.getType());
    if (!type)
      return failure();
    if (type.getShape() == domain.getShape()) {
      outputTask.try_emplace(output, 0);
      rowOutputs.push_back(output);
    } else if (type.getShape() == ArrayRef<int64_t>(featureShape)) {
      outputTask.try_emplace(output, 1);
      featureOutputs.push_back(output);
    } else {
      return failure();
    }
  }
  if (rowOutputs.empty() || featureOutputs.size() < 2)
    return failure();

  auto collectSources = [&](Value root, const llvm::SmallDenseSet<Value> &stops,
                            llvm::SmallDenseSet<Operation *> &set,
                            auto &self) -> LogicalResult {
    if (stops.contains(root))
      return success();
    Operation *producer = root.getDefiningOp();
    if (!producer)
      return success();
    if (!isa<MapOp, FoldOp>(producer) ||
        producer->getParentOfType<RegionOp>() != region)
      return failure();
    if (!set.insert(producer).second)
      return success();
    for (Value operand : producer->getOperands()) {
      if (failed(self(operand, stops, set, self)))
        return failure();
    }
    return success();
  };
  const llvm::SmallDenseSet<Value> noStops;
  llvm::SmallDenseSet<Operation *> rowSet;
  llvm::SmallDenseSet<Operation *> featureRawSet;
  for (Value output : rowOutputs)
    if (failed(collectSources(output, noStops, rowSet, collectSources)))
      return failure();
  for (Value output : featureOutputs)
    if (failed(collectSources(output, noStops, featureRawSet, collectSources)))
      return failure();

  SmallVector<Value> sharedStats;
  llvm::SmallDenseSet<Value> sharedStatSet;
  FailureOr<int64_t> rowElements = scheduleElementCount(rowShape);
  if (failed(rowElements))
    return failure();
  for (Operation *operation : algebra) {
    auto map = dyn_cast<MapOp>(operation);
    if (!map || !rowSet.contains(operation) ||
        !featureRawSet.contains(operation))
      continue;
    auto type = cast<RankedTensorType>(map.getResult(0).getType());
    FailureOr<int64_t> elements = staticElementCount(type);
    if (failed(elements) || *elements != *rowElements ||
        !type.getElementType().isF32() ||
        llvm::none_of(map.getResult(0).getUsers(), [&](Operation *user) {
          auto resultType =
              user->getNumResults() == 1
                  ? dyn_cast<RankedTensorType>(user->getResult(0).getType())
                  : RankedTensorType{};
          if (!resultType)
            return false;
          FailureOr<int64_t> resultElements = staticElementCount(resultType);
          return succeeded(resultElements) && *resultElements > *rowElements;
        }))
      continue;
    sharedStats.push_back(map.getResult(0));
    sharedStatSet.insert(map.getResult(0));
  }
  if (sharedStats.size() != 2)
    return failure();

  llvm::SmallDenseSet<Operation *> featureSet;
  for (Value output : featureOutputs)
    if (failed(
            collectSources(output, sharedStatSet, featureSet, collectSources)))
      return failure();
  llvm::SmallDenseSet<Operation *> covered(rowSet.begin(), rowSet.end());
  covered.insert(featureSet.begin(), featureSet.end());
  if (covered.size() != algebra.size())
    return failure();

  std::array<llvm::SmallDenseSet<Operation *> *, 2> sourceSets = {&rowSet,
                                                                  &featureSet};
  std::array<SmallVector<Operation *>, 2> orderedSources;
  for (auto [taskOrdinal, set] : llvm::enumerate(sourceSets)) {
    for (Operation *operation : algebra)
      if (set->contains(operation))
        orderedSources[taskOrdinal].push_back(operation);
    SmallVector<FoldOp> phases;
    for (Operation *operation : orderedSources[taskOrdinal])
      if (auto fold = dyn_cast<FoldOp>(operation))
        phases.push_back(fold);
    if (phases.empty())
      return failure();
    auto firstInput =
        cast<RankedTensorType>(phases.front().getInputs().front().getType());
    ArrayRef<int64_t> firstReduction = phases.front().getReductionDimensions();
    for (auto [phaseOrdinal, fold] : llvm::enumerate(phases)) {
      auto input = cast<RankedTensorType>(fold.getInputs().front().getType());
      ArrayRef<int64_t> reduction = fold.getReductionDimensions();
      const bool row = reduction == ArrayRef<int64_t>({input.getRank() - 1});
      const bool column =
          (input.getRank() == 2 && reduction == ArrayRef<int64_t>({0})) ||
          isFullPrefixReduction(reduction, input.getRank());
      int64_t cardinality = 1;
      for (int64_t axis : reduction)
        cardinality *= input.getDimSize(axis);
      const bool degenerate = cardinality == 1;
      if ((!row && !column) || (phaseOrdinal == 0 && input != domain) ||
          (phaseOrdinal != 0 &&
           ((!degenerate &&
             (input != firstInput || reduction != firstReduction)) ||
            fold.getResult(0).getType() !=
                phases.front().getResult(0).getType() ||
            (degenerate &&
             !valueDependsOnOperation(fold.getInputs().front(),
                                      phases[phaseOrdinal - 1]))))) {
        return failure();
      }
      Value initializer = fold.getInitializers().front();
      if (!set->contains(initializer.getDefiningOp())) {
        return failure();
      }
    }
    for (Operation *operation : orderedSources[taskOrdinal]) {
      auto map = dyn_cast<MapOp>(operation);
      if (!map)
        continue;
      for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
        if (isa_and_nonnull<FoldOp>(operand.getDefiningOp()) &&
            set->contains(operand.getDefiningOp()) &&
            !isCanonicalFoldResultExpansion(map, operandOrdinal)) {
          return failure();
        }
      }
    }
  }

  DerivedPlan result;
  result.hasRankNFold = domain.getRank() > 2;
  result.hasFoldRegion = true;
  llvm::SmallDenseMap<Value, int64_t> bufferForValue;
  for (BlockArgument argument : region.getBody().front().getArguments()) {
    bufferForValue.try_emplace(argument, result.buffers.size());
    result.buffers.push_back({argument, std::nullopt, {}, false});
  }
  for (Value stat : sharedStats) {
    bufferForValue.try_emplace(stat, result.buffers.size());
    result.buffers.push_back({stat, int64_t{0}, {}, false});
  }
  for (Value output : regionYield.getValues()) {
    int64_t producer = outputTask.lookup(output);
    bufferForValue.try_emplace(output, result.buffers.size());
    result.buffers.push_back({output, producer, {}, true});
  }

  llvm::SmallDenseMap<Operation *, int64_t> sourceOccurrences;
  for (auto *set : sourceSets)
    for (Operation *operation : *set)
      ++sourceOccurrences[operation];
  for (auto [operation, occurrences] : sourceOccurrences)
    if (occurrences > 1 && !isa<MapOp>(operation))
      return failure();
  for (auto [taskOrdinal, sources] : llvm::enumerate(orderedSources)) {
    PlannedTask task;
    task.foldRegionSources = sources;
    llvm::SmallDenseSet<Operation *> taskSet(sources.begin(), sources.end());
    for (Operation *operation : sources) {
      if (isa<FoldOp>(operation))
        task.foldRegionPhases.push_back(operation);
      if (sourceOccurrences.lookup(operation) > 1)
        task.foldRegionRecomputedSources.push_back(operation);
    }
    task.operation = task.foldRegionPhases.front();
    task.kind = MaterializationTaskKind::Fold;
    auto firstFold = cast<FoldOp>(task.operation);
    task.domain.assign(domain.getShape().begin(), domain.getShape().end());
    task.reductionDimensions.assign(firstFold.getReductionDimensions().begin(),
                                    firstFold.getReductionDimensions().end());
    task.orderFree = firstFold.getOrderFree();
    auto addInput = [&](Value value) -> LogicalResult {
      auto found = bufferForValue.find(value);
      if (found == bufferForValue.end())
        return failure();
      if (!llvm::is_contained(task.inputs, found->second))
        task.inputs.push_back(found->second);
      return success();
    };
    for (Operation *operation : sources) {
      auto map = dyn_cast<MapOp>(operation);
      if (!map)
        continue;
      for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
        if (taskSet.contains(operand.getDefiningOp()))
          continue;
        if (failed(addInput(operand)))
          return failure();
        int64_t phase = 0;
        for (Operation *phaseOperation : task.foldRegionPhases) {
          if (cast<FoldOp>(phaseOperation).getSource().getOperationOrdinal() <
              map.getSource().getOperationOrdinal())
            ++phase;
        }
        phase = std::min<int64_t>(phase, task.foldRegionPhases.size() - 1);
        task.foldRegionBoundaryUses.push_back(
            {bufferForValue.lookup(operand), map,
             static_cast<unsigned>(operandOrdinal),
             cast<AffineMapAttr>(map.getIndexingMaps()[operandOrdinal])
                 .getValue(),
             phase});
      }
    }
    llvm::sort(task.inputs);
    for (Operation *phase : task.foldRegionPhases) {
      Value input = cast<FoldOp>(phase).getInputs().front();
      if (!taskSet.contains(input.getDefiningOp()) &&
          (!bufferForValue.count(input) ||
           !llvm::is_contained(task.inputs, bufferForValue.lookup(input))))
        return failure();
    }
    if (task.foldRegionBoundaryUses.empty())
      return failure();
    if (taskOrdinal == 0) {
      for (Value stat : sharedStats) {
        task.outputs.push_back(bufferForValue.lookup(stat));
        task.foldRegionOutputs.push_back(stat.getDefiningOp());
      }
      for (Value output : rowOutputs) {
        task.outputs.push_back(bufferForValue.lookup(output));
        task.foldRegionOutputs.push_back(output.getDefiningOp());
      }
    } else {
      for (Value output : featureOutputs) {
        task.outputs.push_back(bufferForValue.lookup(output));
        task.foldRegionOutputs.push_back(output.getDefiningOp());
      }
    }
    for (int64_t input : task.inputs) {
      if (std::optional<int64_t> producer = result.buffers[input].producer;
          producer && !llvm::is_contained(task.dependencies, *producer))
        task.dependencies.push_back(*producer);
    }
    llvm::sort(task.dependencies);
    result.tasks.push_back(std::move(task));
  }
  for (auto [taskOrdinal, task] : llvm::enumerate(result.tasks))
    for (int64_t input : task.inputs)
      result.buffers[input].consumers.push_back(taskOrdinal);
  return result;
}

FailureOr<DerivedPlan> fuseEligibleMapTasks(RegionOp region,
                                            DerivedPlan initial) {
  SmallVector<Operation *> algebra;
  for (Operation &operation : region.getBody().front().without_terminator())
    algebra.push_back(&operation);
  llvm::SmallDenseMap<Operation *, unsigned> index;
  for (auto [ordinal, operation] : llvm::enumerate(algebra))
    index.try_emplace(operation, ordinal);
  SmallVector<unsigned> parent(algebra.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](unsigned value) {
    while (parent[value] != value) {
      parent[value] = parent[parent[value]];
      value = parent[value];
    }
    return value;
  };
  auto merge = [&](unsigned left, unsigned right) {
    left = root(left);
    right = root(right);
    if (left != right)
      parent[left] = right;
  };
  for (auto [ordinal, operation] : llvm::enumerate(algebra)) {
    auto producer = dyn_cast<MapOp>(operation);
    if (!producer || !isEligibleF32Map(producer) ||
        !producer.getResult(0).hasOneUse())
      continue;
    OpOperand &use = *producer.getResult(0).getUses().begin();
    auto consumer = dyn_cast<MapOp>(use.getOwner());
    if (!consumer || !isEligibleF32Map(consumer))
      continue;
    auto consumerIndex = index.find(consumer);
    if (consumerIndex == index.end())
      continue;
    AffineMap edge =
        cast<AffineMapAttr>(consumer.getIndexingMaps()[use.getOperandNumber()])
            .getValue();
    FailureOr<SmallVector<int64_t>> producerDomain = mapDomainShape(producer);
    FailureOr<SmallVector<int64_t>> consumerDomain = mapDomainShape(consumer);
    if (!edge.isIdentity() || failed(producerDomain) ||
        failed(consumerDomain) || *producerDomain != *consumerDomain)
      continue;
    merge(ordinal, consumerIndex->second);
  }
  for (auto [ordinal, operation] : llvm::enumerate(algebra)) {
    auto fold = dyn_cast<FoldOp>(operation);
    if (!fold || fold.getNumResults() != 1 || !fold.getResult(0).hasOneUse())
      continue;
    OpOperand &use = *fold.getResult(0).getUses().begin();
    auto consumer = dyn_cast<MapOp>(use.getOwner());
    auto consumerIndex = consumer ? index.find(consumer) : index.end();
    if (!consumer || consumerIndex == index.end() ||
        !isEligibleFoldMapEdge(fold, consumer, use.getOperandNumber()))
      continue;
    const unsigned consumerRoot = root(consumerIndex->second);
    const bool anotherFoldBoundary =
        llvm::any_of(llvm::enumerate(algebra), [&](auto member) {
          auto memberMap = dyn_cast<MapOp>(member.value());
          if (!memberMap || root(member.index()) != consumerRoot)
            return false;
          return llvm::any_of(memberMap.getInputs(), [&](Value input) {
            auto other = input.getDefiningOp<FoldOp>();
            return other && other != fold;
          });
        });
    if (!anotherFoldBoundary)
      merge(ordinal, consumerIndex->second);
  }

  SmallVector<SmallVector<unsigned>> members(algebra.size());
  for (unsigned ordinal = 0; ordinal < algebra.size(); ++ordinal)
    members[root(ordinal)].push_back(ordinal);

  llvm::SmallDenseMap<Operation *, PlannedTask *> initialTask;
  for (PlannedTask &task : initial.tasks)
    initialTask.try_emplace(task.operation, &task);
  DerivedPlan fused;
  fused.hasContract = initial.hasContract;
  fused.hasRankNFold = initial.hasRankNFold;
  llvm::SmallDenseMap<Value, int64_t> bufferForValue;
  for (BlockArgument input : region.getBody().front().getArguments()) {
    bufferForValue.try_emplace(input, fused.buffers.size());
    fused.buffers.push_back({input, std::nullopt, {}, false});
  }
  for (auto [ordinal, operation] : llvm::enumerate(algebra)) {
    unsigned groupRoot = root(ordinal);
    ArrayRef<unsigned> group = members[groupRoot];
    if (ordinal != group.back())
      continue;
    PlannedTask task;
    Operation *foldMember = nullptr;
    for (unsigned member : group) {
      if (!isa<FoldOp>(algebra[member]))
        continue;
      if (foldMember)
        return failure();
      foldMember = algebra[member];
    }
    task.operation = foldMember ? foldMember : operation;
    task.kind = initialTask.lookup(task.operation)->kind;
    task.domain = initialTask.lookup(task.operation)->domain;
    task.reductionDimensions =
        initialTask.lookup(task.operation)->reductionDimensions;
    task.orderFree = initialTask.lookup(task.operation)->orderFree;
    llvm::SmallDenseSet<Operation *> groupSet;
    for (unsigned member : group)
      groupSet.insert(algebra[member]);
    if (group.size() > 1) {
      for (unsigned member : group) {
        Operation *memberOperation = algebra[member];
        if (isa<MapOp>(memberOperation))
          task.fusedOperations.push_back(memberOperation);
        else if (!foldMember || memberOperation != foldMember)
          return failure();
      }
      if (foldMember) {
        if (task.fusedOperations.empty())
          return failure();
        task.fusedFold = foldMember;
      } else if (task.fusedOperations.size() != group.size()) {
        return failure();
      }
    }
    unsigned singletonMember = ordinal;
    ArrayRef<unsigned> taskMembers = task.fusedOperations.empty()
                                         ? ArrayRef<unsigned>(singletonMember)
                                         : group;
    llvm::SmallDenseMap<Value, AffineMap> boundaryMaps;
    const bool fusedGroup = !task.fusedOperations.empty();
    auto addInput = [&](Value operand) -> LogicalResult {
      auto found = bufferForValue.find(operand);
      if (found == bufferForValue.end())
        return failure();
      task.inputs.push_back(found->second);
      PlannedBuffer &buffer = fused.buffers[found->second];
      if (buffer.consumers.empty() ||
          buffer.consumers.back() != static_cast<int64_t>(fused.tasks.size()))
        buffer.consumers.push_back(fused.tasks.size());
      if (buffer.producer &&
          !llvm::is_contained(task.dependencies, *buffer.producer))
        task.dependencies.push_back(*buffer.producer);
      return success();
    };
    if (foldMember) {
      auto fold = cast<FoldOp>(foldMember);
      SmallVector<Value> foldOperands;
      llvm::append_range(foldOperands, fold.getInputs());
      llvm::append_range(foldOperands, fold.getInitializers());
      for (Value operand : foldOperands) {
        if (failed(addInput(operand)))
          return failure();
      }
    }
    for (unsigned member : llvm::reverse(taskMembers)) {
      Operation *memberOperation = algebra[member];
      auto memberMap = dyn_cast<MapOp>(memberOperation);
      for (auto [operandOrdinal, operand] :
           llvm::enumerate(memberOperation->getOperands())) {
        if (groupSet.contains(operand.getDefiningOp()))
          continue;
        if (foldMember && memberOperation == foldMember)
          continue;
        AffineMap composedMap;
        if (memberMap)
          composedMap =
              cast<AffineMapAttr>(memberMap.getIndexingMaps()[operandOrdinal])
                  .getValue();
        auto prior = boundaryMaps.find(operand);
        if (fusedGroup && prior != boundaryMaps.end()) {
          if (memberMap && prior->second != composedMap)
            return failure();
          continue;
        }
        if (fusedGroup && memberMap)
          boundaryMaps.try_emplace(operand, composedMap);
        if (failed(addInput(operand)))
          return failure();
        if (fusedGroup && memberMap)
          task.composedInputMaps.push_back(composedMap);
      }
    }
    llvm::sort(task.dependencies);
    for (unsigned member : taskMembers) {
      for (Value result : algebra[member]->getResults()) {
        const bool internal =
            llvm::all_of(result.getUses(), [&](OpOperand &use) {
              return groupSet.contains(use.getOwner());
            });
        if (internal && !result.use_empty())
          continue;
        int64_t bufferOrdinal = fused.buffers.size();
        bufferForValue.try_emplace(result, bufferOrdinal);
        fused.buffers.push_back(
            {result, static_cast<int64_t>(fused.tasks.size()), {}, false});
        task.outputs.push_back(bufferOrdinal);
      }
    }
    if (task.outputs.empty())
      return failure();
    fused.tasks.push_back(std::move(task));
  }
  auto regionYield = cast<YieldOp>(region.getBody().front().getTerminator());
  for (Value value : regionYield.getValues()) {
    auto found = bufferForValue.find(value);
    if (found == bufferForValue.end())
      return failure();
    fused.buffers[found->second].liveOut = true;
  }
  for (PlannedBuffer &buffer : fused.buffers) {
    if (!buffer.liveOut && buffer.consumers.empty())
      return failure();
  }
  return fused;
}

FailureOr<DerivedPlan>
fuseEligibleFloatingMapTasks(DerivedPlan initial,
                             bool requireMapContractInput = false) {
  if (initial.tasks.size() < 2)
    return failure();
  auto plainMap = [&](unsigned ordinal) -> MapOp {
    const PlannedTask &task = initial.tasks[ordinal];
    if (task.kind != MaterializationTaskKind::Map ||
        !task.fusedOperations.empty() || task.fusedFold ||
        !task.foldRegionSources.empty() ||
        !task.contractRegionSources.empty() || !task.compositionKind.empty() ||
        task.statefulContractFoldComposition)
      return {};
    return dyn_cast_or_null<MapOp>(task.operation);
  };
  auto plainFold = [&](unsigned ordinal) -> FoldOp {
    const PlannedTask &task = initial.tasks[ordinal];
    if (task.kind != MaterializationTaskKind::Fold ||
        !task.fusedOperations.empty() || task.fusedFold ||
        !task.foldRegionSources.empty() ||
        !task.contractRegionSources.empty() || !task.compositionKind.empty() ||
        task.statefulContractFoldComposition)
      return {};
    return dyn_cast_or_null<FoldOp>(task.operation);
  };
  auto plainContract = [&](unsigned ordinal) -> ContractOp {
    const PlannedTask &task = initial.tasks[ordinal];
    if (task.kind != MaterializationTaskKind::Contract ||
        !task.fusedOperations.empty() || task.fusedFold ||
        !task.foldRegionSources.empty() ||
        !task.contractRegionSources.empty() || !task.compositionKind.empty() ||
        task.statefulContractFoldComposition)
      return {};
    return dyn_cast_or_null<ContractOp>(task.operation);
  };

  llvm::SmallDenseMap<Value, unsigned> bufferForValue;
  for (auto [ordinal, buffer] : llvm::enumerate(initial.buffers))
    bufferForValue.try_emplace(buffer.value, ordinal);
  SmallVector<unsigned> parent(initial.tasks.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](unsigned value) {
    while (parent[value] != value) {
      parent[value] = parent[parent[value]];
      value = parent[value];
    }
    return value;
  };
  auto merge = [&](unsigned producer, unsigned consumer) {
    producer = root(producer);
    consumer = root(consumer);
    if (producer != consumer)
      parent[producer] = consumer;
  };

  bool changed = false;
  bool fusedMapContractInput = false;
  for (unsigned producerOrdinal = 0; producerOrdinal < initial.tasks.size();
       ++producerOrdinal) {
    MapOp producer = plainMap(producerOrdinal);
    if (!producer || !isEligibleFloatingMap(producer) ||
        producer.getNumResults() != 1)
      continue;
    auto buffer = bufferForValue.find(producer.getResult(0));
    if (buffer == bufferForValue.end() ||
        initial.buffers[buffer->second].consumers.size() != 1)
      continue;
    const unsigned consumerOrdinal =
        initial.buffers[buffer->second].consumers.front();
    if (consumerOrdinal <= producerOrdinal ||
        consumerOrdinal >= initial.tasks.size())
      continue;
    ContractOp consumer = plainContract(consumerOrdinal);
    if (!consumer || !isClosedProjectedContract(
                         consumer, /*allowMultipleReductions=*/true))
      continue;
    auto use = llvm::find_if(producer.getResult(0).getUses(),
                             [&](OpOperand &operand) {
                               return operand.getOwner() ==
                                      consumer.getOperation();
                             });
    if (use == producer.getResult(0).getUses().end())
      continue;
    auto producerType =
        dyn_cast<RankedTensorType>(producer.getResult(0).getType());
    auto consumerType =
        dyn_cast<RankedTensorType>(consumer.getInputs()[use->getOperandNumber()]
                                       .getType());
    if (!producerType || producerType != consumerType)
      continue;
    merge(producerOrdinal, consumerOrdinal);
    fusedMapContractInput = true;
    changed = true;
  }
  for (unsigned producerOrdinal = 0; producerOrdinal < initial.tasks.size();
       ++producerOrdinal) {
    MapOp producer = plainMap(producerOrdinal);
    if (!producer || !isEligibleFloatingMap(producer) ||
        producer.getNumResults() != 1 || !producer.getResult(0).hasOneUse())
      continue;
    auto buffer = bufferForValue.find(producer.getResult(0));
    if (buffer == bufferForValue.end() ||
        initial.buffers[buffer->second].liveOut ||
        initial.buffers[buffer->second].consumers.size() != 1)
      continue;
    const unsigned consumerOrdinal =
        initial.buffers[buffer->second].consumers.front();
    if (consumerOrdinal <= producerOrdinal ||
        consumerOrdinal >= initial.tasks.size())
      continue;
    MapOp consumer = plainMap(consumerOrdinal);
    if (!consumer || !isEligibleFloatingMap(consumer))
      continue;
    OpOperand &use = *producer.getResult(0).getUses().begin();
    if (use.getOwner() != consumer.getOperation())
      continue;
    auto indexing = dyn_cast<AffineMapAttr>(
        consumer.getIndexingMaps()[use.getOperandNumber()]);
    FailureOr<SmallVector<int64_t>> producerDomain =
        genericMapDomainShape(producer);
    FailureOr<SmallVector<int64_t>> consumerDomain =
        genericMapDomainShape(consumer);
    if (!indexing || !indexing.getValue().isIdentity() ||
        failed(producerDomain) || failed(consumerDomain) ||
        *producerDomain != *consumerDomain)
      continue;
    merge(producerOrdinal, consumerOrdinal);
    changed = true;
  }
  for (unsigned consumerOrdinal = 0; consumerOrdinal < initial.tasks.size();
       ++consumerOrdinal) {
    MapOp consumer = plainMap(consumerOrdinal);
    if (!consumer || consumer.getInputs().size() < 2 ||
        !isEligibleFloatingMap(consumer))
      continue;
    FailureOr<SmallVector<int64_t>> consumerDomain =
        genericMapDomainShape(consumer);
    if (failed(consumerDomain))
      continue;
    SmallVector<unsigned> contractProducers;
    bool compatible = true;
    for (auto [operandOrdinal, operand] : llvm::enumerate(consumer.getInputs())) {
      auto contract = operand.getDefiningOp<ContractOp>();
      auto buffer = bufferForValue.find(operand);
      auto indexing = dyn_cast<AffineMapAttr>(
          consumer.getIndexingMaps()[operandOrdinal]);
      if (!contract || buffer == bufferForValue.end() || !indexing ||
          !indexing.getValue().isIdentity() || !operand.hasOneUse() ||
          initial.buffers[buffer->second].liveOut ||
          initial.buffers[buffer->second].consumers.size() != 1 ||
          !initial.buffers[buffer->second].producer ||
          !plainContract(*initial.buffers[buffer->second].producer) ||
          !isClosedProjectedContract(contract,
                                     /*allowMultipleReductions=*/true)) {
        compatible = false;
        break;
      }
      auto result = dyn_cast<RankedTensorType>(operand.getType());
      if (!result || result.getShape() != ArrayRef<int64_t>(*consumerDomain)) {
        compatible = false;
        break;
      }
      contractProducers.push_back(*initial.buffers[buffer->second].producer);
    }
    if (!compatible || contractProducers.size() < 2)
      continue;
    for (unsigned producerOrdinal : contractProducers)
      merge(producerOrdinal, consumerOrdinal);
    changed = true;
  }
  for (unsigned consumerOrdinal = 0; consumerOrdinal < initial.tasks.size();
       ++consumerOrdinal) {
    if (!plainMap(consumerOrdinal) || root(consumerOrdinal) != consumerOrdinal)
      continue;
    const unsigned consumerRoot = root(consumerOrdinal);
    SmallVector<unsigned> mapMembers;
    for (unsigned member = 0; member < initial.tasks.size(); ++member)
      if (root(member) == consumerRoot)
        mapMembers.push_back(member);
    if (mapMembers.size() < 2 ||
        !llvm::all_of(mapMembers, [&](unsigned member) {
          MapOp map = plainMap(member);
          return map && isEligibleFloatingMap(map);
        }))
      continue;
    MapOp terminal = plainMap(mapMembers.back());
    FailureOr<SmallVector<int64_t>> terminalDomain =
        terminal ? genericMapDomainShape(terminal)
                 : FailureOr<SmallVector<int64_t>>(failure());
    if (failed(terminalDomain))
      continue;
    llvm::SmallDenseSet<unsigned> contractProducerSet;
    bool compatible = true;
    for (unsigned member : mapMembers) {
      MapOp map = plainMap(member);
      for (auto [operandOrdinal, operand] : llvm::enumerate(map.getInputs())) {
        Operation *producer = operand.getDefiningOp();
        auto buffer = bufferForValue.find(operand);
        if (buffer != bufferForValue.end() &&
            initial.buffers[buffer->second].producer &&
            root(*initial.buffers[buffer->second].producer) == consumerRoot)
          continue;
        auto contract = dyn_cast_or_null<ContractOp>(producer);
        auto indexing = dyn_cast<AffineMapAttr>(
            map.getIndexingMaps()[operandOrdinal]);
        if (!contract || buffer == bufferForValue.end() || !indexing ||
            !indexing.getValue().isIdentity() || !operand.hasOneUse() ||
            initial.buffers[buffer->second].liveOut ||
            initial.buffers[buffer->second].consumers.size() != 1 ||
            !initial.buffers[buffer->second].producer ||
            !plainContract(*initial.buffers[buffer->second].producer) ||
            !isClosedProjectedContract(
                contract, /*allowMultipleReductions=*/true)) {
          compatible = false;
          break;
        }
        auto type = dyn_cast<RankedTensorType>(operand.getType());
        if (!type || type.getShape() != ArrayRef<int64_t>(*terminalDomain)) {
          compatible = false;
          break;
        }
        contractProducerSet.insert(*initial.buffers[buffer->second].producer);
      }
      if (!compatible)
        break;
    }
    if (!compatible || contractProducerSet.size() < 2)
      continue;
    for (unsigned producerOrdinal : contractProducerSet)
      merge(producerOrdinal, consumerRoot);
    changed = true;
  }
  for (unsigned producerOrdinal = 0; producerOrdinal < initial.tasks.size();
       ++producerOrdinal) {
    ContractOp contract = plainContract(producerOrdinal);
    if (!contract || !isClosedProjectedContract(contract) ||
        contract.getNumResults() != 1 ||
        !contract.getResult(0).hasOneUse())
      continue;
    auto buffer = bufferForValue.find(contract.getResult(0));
    if (buffer == bufferForValue.end() ||
        initial.buffers[buffer->second].liveOut ||
        initial.buffers[buffer->second].consumers.size() != 1)
      continue;
    const unsigned consumerOrdinal =
        initial.buffers[buffer->second].consumers.front();
    if (consumerOrdinal <= producerOrdinal ||
        consumerOrdinal >= initial.tasks.size())
      continue;
    MapOp consumer = plainMap(consumerOrdinal);
    if (!consumer || consumer.getInputs().size() != 1 ||
        !isEligibleFloatingMap(consumer))
      continue;
    OpOperand &use = *contract.getResult(0).getUses().begin();
    if (use.getOwner() != consumer.getOperation())
      continue;
    const unsigned consumerRoot = root(consumerOrdinal);
    const bool allUnaryMaps = llvm::all_of(
        llvm::seq<unsigned>(0, initial.tasks.size()), [&](unsigned member) {
          MapOp map = plainMap(member);
          return root(member) != consumerRoot || !map ||
                 map.getInputs().size() == 1;
        });
    const bool alreadyHasContract = llvm::any_of(
        llvm::seq<unsigned>(0, initial.tasks.size()), [&](unsigned member) {
          return root(member) == consumerRoot && plainContract(member);
        });
    if (!allUnaryMaps || alreadyHasContract)
      continue;
    merge(producerOrdinal, consumerOrdinal);
    changed = true;
  }
  for (unsigned producerOrdinal = 0; producerOrdinal < initial.tasks.size();
       ++producerOrdinal) {
    FoldOp fold = plainFold(producerOrdinal);
    if (!fold || fold.getNumResults() != 1 ||
        !fold.getResult(0).hasOneUse())
      continue;
    auto buffer = bufferForValue.find(fold.getResult(0));
    if (buffer == bufferForValue.end() ||
        initial.buffers[buffer->second].liveOut ||
        initial.buffers[buffer->second].consumers.size() != 1)
      continue;
    const unsigned consumerOrdinal =
        initial.buffers[buffer->second].consumers.front();
    if (consumerOrdinal <= producerOrdinal ||
        consumerOrdinal >= initial.tasks.size())
      continue;
    MapOp consumer = plainMap(consumerOrdinal);
    if (!consumer)
      continue;
    OpOperand &use = *fold.getResult(0).getUses().begin();
    if (use.getOwner() != consumer.getOperation() ||
        !isEligibleFoldMapEdge(fold, consumer, use.getOperandNumber()))
      continue;
    const unsigned consumerRoot = root(consumerOrdinal);
    const bool hasContractBoundary = llvm::any_of(
        llvm::seq<unsigned>(0, initial.tasks.size()), [&](unsigned member) {
          return root(member) == consumerRoot && plainContract(member);
        });
    const bool hasNonF32Result = llvm::any_of(
        llvm::seq<unsigned>(0, initial.tasks.size()), [&](unsigned member) {
          MapOp memberMap = plainMap(member);
          if (!memberMap || root(member) != consumerRoot)
            return false;
          return llvm::any_of(memberMap->getResultTypes(), [](Type type) {
            auto tensor = dyn_cast<RankedTensorType>(type);
            return !tensor || !tensor.getElementType().isF32();
          });
        });
    const bool anotherFoldBoundary = llvm::any_of(
        llvm::seq<unsigned>(0, initial.tasks.size()), [&](unsigned member) {
          MapOp memberMap = plainMap(member);
          if (!memberMap || root(member) != consumerRoot)
            return false;
          return llvm::any_of(memberMap.getInputs(), [&](Value input) {
            auto other = input.getDefiningOp<FoldOp>();
            return other && other != fold;
          });
        });
    if (hasContractBoundary || hasNonF32Result || anotherFoldBoundary)
      continue;
    merge(producerOrdinal, consumerOrdinal);
    changed = true;
  }

  SmallVector<llvm::SmallDenseSet<unsigned>> ancestors(initial.tasks.size());
  for (unsigned ordinal = 0; ordinal < initial.tasks.size(); ++ordinal) {
    for (int64_t dependency : initial.tasks[ordinal].dependencies) {
      if (dependency < 0 || dependency >= ordinal)
        return failure();
      ancestors[ordinal].insert(dependency);
      ancestors[ordinal].insert(ancestors[dependency].begin(),
                                ancestors[dependency].end());
    }
  }
  struct MapCoLaunchCandidate {
    unsigned root;
    SmallVector<unsigned> members;
    SmallVector<int64_t> domain;
    int64_t inputs;
    int64_t outputs;
    int64_t earliestConsumer;
  };
  SmallVector<MapCoLaunchCandidate> candidates;
  for (unsigned candidateRoot = 0; candidateRoot < initial.tasks.size();
       ++candidateRoot) {
    if (root(candidateRoot) != candidateRoot)
      continue;
    SmallVector<unsigned> members;
    for (unsigned member = 0; member < initial.tasks.size(); ++member)
      if (root(member) == candidateRoot)
        members.push_back(member);
    if (members.empty() || members.size() > 16 ||
        !llvm::all_of(members, [&](unsigned member) {
          MapOp map = plainMap(member);
          return map && isEligibleFloatingMap(map);
        }))
      continue;
    FailureOr<SmallVector<int64_t>> domain =
        genericMapDomainShape(plainMap(members.back()));
    if (failed(domain) || domain->empty())
      continue;
    llvm::SmallDenseSet<int64_t> boundaryInputs;
    int64_t outputs = 0;
    int64_t earliestConsumer = initial.tasks.size();
    bool compatible = true;
    for (unsigned member : members) {
      for (int64_t input : initial.tasks[member].inputs) {
        if (input < 0 || input >= initial.buffers.size()) {
          compatible = false;
          break;
        }
        std::optional<int64_t> producer = initial.buffers[input].producer;
        if (!producer || root(*producer) != candidateRoot)
          boundaryInputs.insert(input);
      }
      for (int64_t output : initial.tasks[member].outputs) {
        if (output < 0 || output >= initial.buffers.size()) {
          compatible = false;
          break;
        }
        const PlannedBuffer &buffer = initial.buffers[output];
        const bool crossesBoundary =
            buffer.liveOut || llvm::any_of(buffer.consumers, [&](int64_t use) {
              return use < 0 || use >= initial.tasks.size() ||
                     root(use) != candidateRoot;
            });
        if (!crossesBoundary)
          continue;
        auto type = dyn_cast<RankedTensorType>(buffer.value.getType());
        if (!type || type.getShape() != ArrayRef<int64_t>(*domain) ||
            (!type.getElementType().isF32() &&
             !type.getElementType().isBF16())) {
          compatible = false;
          break;
        }
        ++outputs;
        for (int64_t consumer : buffer.consumers)
          if (consumer >= 0 && consumer < initial.tasks.size() &&
              root(consumer) != candidateRoot)
            earliestConsumer = std::min(earliestConsumer, consumer);
      }
      if (!compatible)
        break;
    }
    if (compatible && outputs > 0 && outputs <= 8 &&
        boundaryInputs.size() <= 32)
      candidates.push_back({candidateRoot, std::move(members),
                            std::move(*domain),
                            static_cast<int64_t>(boundaryInputs.size()),
                            outputs, earliestConsumer});
  }
  SmallVector<SmallVector<unsigned>> coLaunchGroups;
  for (unsigned candidate = 0; candidate < candidates.size(); ++candidate) {
    const MapCoLaunchCandidate &current = candidates[candidate];
    bool placed = false;
    for (SmallVector<unsigned> &group : coLaunchGroups) {
      int64_t sources = current.members.size();
      int64_t inputs = current.inputs;
      int64_t outputs = current.outputs;
      bool incomparable = true;
      for (unsigned grouped : group) {
        const MapCoLaunchCandidate &other = candidates[grouped];
        if (other.domain != current.domain) {
          incomparable = false;
          break;
        }
        sources += other.members.size();
        inputs += other.inputs;
        outputs += other.outputs;
        incomparable &= other.earliestConsumer > current.root;
        for (unsigned left : current.members)
          for (unsigned right : other.members)
            incomparable &= !ancestors[left].contains(right) &&
                            !ancestors[right].contains(left);
      }
      if (!incomparable || sources > 16 || inputs > 32 || outputs > 8)
        continue;
      group.push_back(candidate);
      placed = true;
      break;
    }
    if (!placed)
      coLaunchGroups.push_back({candidate});
  }
  for (ArrayRef<unsigned> group : coLaunchGroups) {
    if (group.size() < 2)
      continue;
    const unsigned owner = candidates[group.back()].root;
    for (unsigned candidate : group.drop_back())
      merge(candidates[candidate].root, owner);
    changed = true;
  }
  if (!changed)
    return failure();
  if (requireMapContractInput && !fusedMapContractInput)
    return failure();

  llvm::SmallDenseMap<unsigned, SmallVector<unsigned>> membersForRoot;
  for (unsigned ordinal = 0; ordinal < initial.tasks.size(); ++ordinal)
    membersForRoot[root(ordinal)].push_back(ordinal);
  SmallVector<SmallVector<unsigned>> components;
  components.reserve(membersForRoot.size());
  for (auto &entry : membersForRoot)
    components.push_back(std::move(entry.second));
  llvm::sort(components, [](ArrayRef<unsigned> left, ArrayRef<unsigned> right) {
    return left.back() < right.back();
  });
  SmallVector<unsigned> componentForTask(initial.tasks.size());
  for (auto [component, members] : llvm::enumerate(components))
    for (unsigned member : members)
      componentForTask[member] = component;

  DerivedPlan fused;
  fused.hasContract = initial.hasContract;
  fused.hasRankNFold = initial.hasRankNFold;
  fused.hasFoldRegion = initial.hasFoldRegion;
  fused.hasContractRegion = initial.hasContractRegion;
  fused.hasContractFoldComposition = initial.hasContractFoldComposition;
  fused.hasLogicalRelation = initial.hasLogicalRelation;
  fused.hasStatefulContractFoldPullback =
      initial.hasStatefulContractFoldPullback;
  fused.normalizedExpContractRegions = initial.normalizedExpContractRegions;
  fused.normalizedExpContractPullbacks = initial.normalizedExpContractPullbacks;

  SmallVector<int64_t> newBufferForOld(initial.buffers.size(), -1);
  for (auto [oldOrdinal, buffer] : llvm::enumerate(initial.buffers)) {
    std::optional<unsigned> producer =
        buffer.producer
            ? std::optional<unsigned>(componentForTask[*buffer.producer])
            : std::nullopt;
    bool crossesBoundary = !producer || buffer.liveOut;
    for (int64_t consumer : buffer.consumers)
      crossesBoundary |= !producer || componentForTask[consumer] != *producer;
    if (!crossesBoundary)
      continue;
    newBufferForOld[oldOrdinal] = fused.buffers.size();
    PlannedBuffer planned{buffer.value,
                          producer ? std::optional<int64_t>(*producer)
                                   : std::nullopt,
                          {},
                          buffer.liveOut};
    for (int64_t consumer : buffer.consumers) {
      const int64_t mapped = componentForTask[consumer];
      if (producer && mapped == static_cast<int64_t>(*producer))
        continue;
      planned.consumers.push_back(mapped);
    }
    llvm::sort(planned.consumers);
    planned.consumers.erase(
        std::unique(planned.consumers.begin(), planned.consumers.end()),
        planned.consumers.end());
    fused.buffers.push_back(std::move(planned));
  }

  for (auto [component, members] : llvm::enumerate(components)) {
    std::optional<unsigned> foldMember;
    SmallVector<unsigned> contractMembers;
    for (unsigned member : members) {
      if (!plainFold(member))
        continue;
      if (foldMember)
        return failure();
      foldMember = member;
      continue;
    }
    for (unsigned member : members) {
      if (!plainContract(member))
        continue;
      contractMembers.push_back(member);
    }
    if (foldMember && !contractMembers.empty())
      return failure();
    PlannedTask task =
        initial.tasks[foldMember.value_or(
            contractMembers.empty() ? members.back()
                                    : contractMembers.front())];
    task.inputs.clear();
    task.outputs.clear();
    task.dependencies.clear();
    if (members.size() > 1) {
      task.fusedOperations.clear();
      task.composedInputMaps.clear();
      llvm::SmallDenseSet<Operation *> group;
      for (unsigned member : members) {
        MapOp map = plainMap(member);
        if (!map && (!foldMember || member != *foldMember) &&
            !llvm::is_contained(contractMembers, member))
          return failure();
        Operation *operation = initial.tasks[member].operation;
        group.insert(operation);
        if (map && contractMembers.empty())
          task.fusedOperations.push_back(map);
      }
      if (!contractMembers.empty()) {
        task.operation = initial.tasks[members.back()].operation;
        task.compositionKind = "map_contract";
        for (unsigned member : members)
          task.compositionSources.push_back(initial.tasks[member].operation);
        if (llvm::any_of(members, [&](unsigned member) {
              return member < contractMembers.front() && plainMap(member);
            }))
          task.compositionPhases = task.compositionSources;
        if (contractMembers.size() > 1) {
          auto terminal = plainMap(members.back());
          FailureOr<SmallVector<int64_t>> domain =
              terminal ? genericMapDomainShape(terminal)
                       : FailureOr<SmallVector<int64_t>>(failure());
          if (failed(domain))
            return failure();
          task.kind = MaterializationTaskKind::Map;
          task.domain = std::move(*domain);
          task.reductionDimensions.clear();
          task.orderFree = false;
        }
        llvm::SmallDenseSet<int64_t> seenInputs;
        for (unsigned member : members) {
          for (int64_t input : initial.tasks[member].inputs) {
            if (input < 0 || input >= newBufferForOld.size())
              return failure();
            std::optional<int64_t> producer = initial.buffers[input].producer;
            if (producer && componentForTask[*producer] == component)
              continue;
            if (newBufferForOld[input] < 0 || !seenInputs.insert(input).second)
              continue;
            task.inputs.push_back(newBufferForOld[input]);
          }
        }
        int64_t temporaryBytes = 0;
        for (unsigned contractMember : contractMembers) {
          auto temporary = dyn_cast<RankedTensorType>(
              plainContract(contractMember).getResult(0).getType());
          FailureOr<int64_t> bytes =
              temporary ? checkedPlanningTensorBytes(temporary)
                        : FailureOr<int64_t>(failure());
          if (failed(bytes) ||
              temporaryBytes >
                  std::numeric_limits<int64_t>::max() - *bytes)
            return failure();
          temporaryBytes += *bytes;
        }
        task.compositionCost = {
            static_cast<int64_t>(members.size()), 1, temporaryBytes, 0,
            2 * temporaryBytes, 0,
            static_cast<int64_t>(contractMembers.size())};
      } else if (foldMember) {
        if (task.fusedOperations.empty())
          return failure();
        task.fusedFold = initial.tasks[*foldMember].operation;
        FoldOp fold = plainFold(*foldMember);
        SmallVector<Value> foldOperands;
        llvm::append_range(foldOperands, fold.getInputs());
        llvm::append_range(foldOperands, fold.getInitializers());
        for (Value operand : foldOperands) {
          auto oldBuffer = bufferForValue.find(operand);
          if (oldBuffer == bufferForValue.end() ||
              newBufferForOld[oldBuffer->second] < 0)
            return failure();
          task.inputs.push_back(newBufferForOld[oldBuffer->second]);
        }
      }
      llvm::SmallDenseSet<Value> seenBoundary;
      for (Operation *operation : llvm::reverse(task.fusedOperations)) {
        auto map = cast<MapOp>(operation);
        for (auto [operandOrdinal, operand] :
             llvm::enumerate(map.getInputs())) {
          if (group.contains(operand.getDefiningOp()) ||
              !seenBoundary.insert(operand).second)
            continue;
          auto oldBuffer = bufferForValue.find(operand);
          if (oldBuffer == bufferForValue.end() ||
              newBufferForOld[oldBuffer->second] < 0)
            return failure();
          task.inputs.push_back(newBufferForOld[oldBuffer->second]);
          task.composedInputMaps.push_back(
              cast<AffineMapAttr>(map.getIndexingMaps()[operandOrdinal])
                  .getValue());
        }
      }
    } else {
      const PlannedTask &original = initial.tasks[members.front()];
      for (int64_t input : original.inputs) {
        if (input < 0 || input >= newBufferForOld.size() ||
            newBufferForOld[input] < 0)
          return failure();
        task.inputs.push_back(newBufferForOld[input]);
      }
    }
    for (unsigned member : members) {
      for (int64_t output : initial.tasks[member].outputs) {
        if (output >= 0 && output < newBufferForOld.size() &&
            newBufferForOld[output] >= 0) {
          task.outputs.push_back(newBufferForOld[output]);
          if (members.size() > 1 && contractMembers.empty() && !foldMember) {
            Operation *producer = initial.buffers[output].value.getDefiningOp();
            if (!isa_and_nonnull<MapOp>(producer) ||
                llvm::none_of(members, [&](unsigned candidate) {
                  return initial.tasks[candidate].operation == producer;
                }))
              return failure();
            task.fusedMapOutputs.push_back(producer);
          }
        }
      }
    }
    if (task.compositionKind == "map_contract" && task.outputs.size() > 1) {
      task.fusedMapOutputs.clear();
      for (int64_t output : task.outputs) {
        auto producer = dyn_cast_or_null<MapOp>(
            fused.buffers[output].value.getDefiningOp());
        if (!producer ||
            !llvm::is_contained(task.compositionSources,
                                producer.getOperation())) {
          task.fusedMapOutputs.clear();
          break;
        }
        task.fusedMapOutputs.push_back(producer);
      }
    }
    for (int64_t input : task.inputs) {
      std::optional<int64_t> producer = fused.buffers[input].producer;
      if (producer && *producer != static_cast<int64_t>(component) &&
          !llvm::is_contained(task.dependencies, *producer))
        task.dependencies.push_back(*producer);
    }
    llvm::sort(task.dependencies);
    if (task.outputs.empty())
      return failure();
    fused.tasks.push_back(std::move(task));
  }
  return fused;
}

FailureOr<DerivedPlan> groupIntegerRelationCompositions(DerivedPlan initial) {
  llvm::SmallDenseMap<Value, unsigned> bufferForValue;
  for (auto [ordinal, buffer] : llvm::enumerate(initial.buffers))
    bufferForValue.try_emplace(buffer.value, ordinal);
  auto plainRelation = [&](unsigned ordinal) -> RelationOp {
    const PlannedTask &task = initial.tasks[ordinal];
    if (task.kind != MaterializationTaskKind::Relation ||
        !task.fusedOperations.empty() || !task.compositionKind.empty() ||
        task.statefulContractFoldComposition)
      return {};
    return dyn_cast_or_null<RelationOp>(task.operation);
  };
  SmallVector<unsigned> parent(initial.tasks.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](unsigned value) {
    while (parent[value] != value) {
      parent[value] = parent[parent[value]];
      value = parent[value];
    }
    return value;
  };
  bool changed = false;
  // A Gather may consume an index expression that is also live across another
  // domain.  Keep that index as a component output while evaluating its
  // Relation DAG in the Gather's ownership domain.
  for (unsigned anchorOrdinal = 0; anchorOrdinal < initial.tasks.size();
       ++anchorOrdinal) {
    RelationOp gather = plainRelation(anchorOrdinal);
    if (!gather || gather.getSemantics() != RelationSemantics::Gather ||
        gather.getInputs().size() != 2 || gather.getNumResults() != 1)
      continue;
    auto found = bufferForValue.find(gather.getInputs()[1]);
    if (found == bufferForValue.end())
      continue;
    const PlannedBuffer &indices = initial.buffers[found->second];
    if (!indices.producer || *indices.producer >= anchorOrdinal)
      continue;
    const unsigned indicesProducer = *indices.producer;
    RelationOp terminalIndices = plainRelation(indicesProducer);
    auto indicesType =
        dyn_cast<RankedTensorType>(gather.getInputs()[1].getType());
    if (!terminalIndices || !indicesType ||
        !indicesType.getElementType().isInteger(32))
      continue;
    llvm::SmallDenseSet<unsigned> members{anchorOrdinal, indicesProducer};
    SmallVector<Value> worklist(terminalIndices.getInputs().begin(),
                                terminalIndices.getInputs().end());
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      auto producerBuffer = bufferForValue.find(value);
      if (producerBuffer == bufferForValue.end())
        continue;
      const PlannedBuffer &buffer = initial.buffers[producerBuffer->second];
      if (!buffer.producer || buffer.liveOut || buffer.consumers.size() != 1 ||
          buffer.consumers.front() >= initial.tasks.size())
        continue;
      unsigned producerOrdinal = *buffer.producer;
      RelationOp producer = plainRelation(producerOrdinal);
      if (!producer || producerOrdinal >= indicesProducer ||
          !members.insert(producerOrdinal).second)
        continue;
      llvm::append_range(worklist, producer.getInputs());
    }
    if (members.size() < 3)
      continue;
    for (unsigned member : members)
      if (member != anchorOrdinal)
        parent[root(member)] = root(anchorOrdinal);
    changed = true;
  }
  for (unsigned anchorOrdinal = 0; anchorOrdinal < initial.tasks.size();
       ++anchorOrdinal) {
    if (root(anchorOrdinal) != anchorOrdinal)
      continue;
    RelationOp anchor = plainRelation(anchorOrdinal);
    if (!anchor || anchor.getNumResults() != 1)
      continue;
    auto output = dyn_cast<RankedTensorType>(anchor.getResult(0).getType());
    auto outputBuffer = bufferForValue.find(anchor.getResult(0));
    if (!output ||
        (!output.getElementType().isInteger(1) &&
         !output.getElementType().isInteger(32)) ||
        outputBuffer == bufferForValue.end())
      continue;
    const PlannedBuffer &terminalBuffer = initial.buffers[outputBuffer->second];
    const bool terminal = terminalBuffer.liveOut ||
                          terminalBuffer.consumers.size() != 1 ||
                          !plainRelation(terminalBuffer.consumers.front());
    if (!terminal)
      continue;
    SmallVector<Value> worklist(anchor.getInputs().begin(),
                                anchor.getInputs().end());
    llvm::SmallDenseSet<unsigned> members{anchorOrdinal};
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      auto found = bufferForValue.find(value);
      if (found == bufferForValue.end())
        continue;
      const PlannedBuffer &buffer = initial.buffers[found->second];
      if (!buffer.producer || buffer.liveOut || buffer.consumers.size() != 1 ||
          buffer.consumers.front() >= initial.tasks.size())
        continue;
      unsigned producerOrdinal = *buffer.producer;
      RelationOp producer = plainRelation(producerOrdinal);
      if (!producer || producerOrdinal >= anchorOrdinal ||
          !members.insert(producerOrdinal).second)
        continue;
      llvm::append_range(worklist, producer.getInputs());
    }
    if (members.size() < 2)
      continue;
    for (unsigned member : members)
      if (member != anchorOrdinal)
        parent[root(member)] = root(anchorOrdinal);
    changed = true;
  }
  if (!changed)
    return failure();

  llvm::SmallDenseMap<unsigned, SmallVector<unsigned>> membersForRoot;
  for (unsigned ordinal = 0; ordinal < initial.tasks.size(); ++ordinal)
    membersForRoot[root(ordinal)].push_back(ordinal);
  SmallVector<SmallVector<unsigned>> components;
  for (auto &entry : membersForRoot)
    components.push_back(std::move(entry.second));
  llvm::sort(components, [](ArrayRef<unsigned> left, ArrayRef<unsigned> right) {
    return left.back() < right.back();
  });
  SmallVector<unsigned> componentForTask(initial.tasks.size());
  for (auto [component, members] : llvm::enumerate(components))
    for (unsigned member : members)
      componentForTask[member] = component;

  DerivedPlan grouped;
  grouped.hasContract = initial.hasContract;
  grouped.hasRankNFold = initial.hasRankNFold;
  grouped.hasFoldRegion = initial.hasFoldRegion;
  grouped.hasContractRegion = initial.hasContractRegion;
  grouped.hasContractFoldComposition = initial.hasContractFoldComposition;
  grouped.hasLogicalRelation = initial.hasLogicalRelation;
  grouped.hasStatefulContractFoldPullback =
      initial.hasStatefulContractFoldPullback;
  grouped.normalizedExpContractRegions = initial.normalizedExpContractRegions;
  grouped.normalizedExpContractPullbacks =
      initial.normalizedExpContractPullbacks;

  SmallVector<int64_t> newBufferForOld(initial.buffers.size(), -1);
  for (auto [oldOrdinal, buffer] : llvm::enumerate(initial.buffers)) {
    std::optional<unsigned> producer =
        buffer.producer
            ? std::optional<unsigned>(componentForTask[*buffer.producer])
            : std::nullopt;
    bool crossesBoundary = !producer || buffer.liveOut;
    for (int64_t consumer : buffer.consumers)
      crossesBoundary |= !producer || componentForTask[consumer] != *producer;
    if (!crossesBoundary)
      continue;
    newBufferForOld[oldOrdinal] = grouped.buffers.size();
    PlannedBuffer planned{buffer.value,
                          producer ? std::optional<int64_t>(*producer)
                                   : std::nullopt,
                          {},
                          buffer.liveOut};
    for (int64_t consumer : buffer.consumers) {
      const int64_t mappedConsumer = componentForTask[consumer];
      if (!producer || mappedConsumer != *producer)
        planned.consumers.push_back(mappedConsumer);
    }
    llvm::sort(planned.consumers);
    planned.consumers.erase(
        std::unique(planned.consumers.begin(), planned.consumers.end()),
        planned.consumers.end());
    grouped.buffers.push_back(std::move(planned));
  }

  for (auto [component, members] : llvm::enumerate(components)) {
    PlannedTask task = initial.tasks[members.back()];
    task.inputs.clear();
    task.outputs.clear();
    task.dependencies.clear();
    if (members.size() > 1) {
      RelationOp terminal = plainRelation(members.back());
      task.compositionKind =
          terminal && terminal.getSemantics() == RelationSemantics::Gather
              ? "relation_gather"
              : "integer_relation";
      task.compositionSources.clear();
      llvm::SmallDenseSet<Operation *> operations;
      for (unsigned member : members) {
        RelationOp relation = plainRelation(member);
        if (!relation)
          return failure();
        task.compositionSources.push_back(relation);
        operations.insert(relation);
      }
      llvm::SmallDenseSet<Value> seen;
      for (Operation *operation : task.compositionSources) {
        for (Value operand : operation->getOperands()) {
          if (operations.contains(operand.getDefiningOp()) ||
              !seen.insert(operand).second)
            continue;
          auto oldBuffer = bufferForValue.find(operand);
          if (oldBuffer == bufferForValue.end() ||
              newBufferForOld[oldBuffer->second] < 0)
            return failure();
          task.inputs.push_back(newBufferForOld[oldBuffer->second]);
        }
      }
    } else {
      const PlannedTask &original = initial.tasks[members.front()];
      for (int64_t input : original.inputs) {
        if (input < 0 || input >= newBufferForOld.size() ||
            newBufferForOld[input] < 0)
          return failure();
        task.inputs.push_back(newBufferForOld[input]);
      }
    }
    for (unsigned member : members) {
      for (int64_t output : initial.tasks[member].outputs) {
        if (output >= 0 && output < newBufferForOld.size() &&
            newBufferForOld[output] >= 0)
          task.outputs.push_back(newBufferForOld[output]);
      }
    }
    for (int64_t input : task.inputs) {
      std::optional<int64_t> producer = grouped.buffers[input].producer;
      if (producer && *producer != static_cast<int64_t>(component) &&
          !llvm::is_contained(task.dependencies, *producer))
        task.dependencies.push_back(*producer);
    }
    llvm::sort(task.dependencies);
    if (task.outputs.empty())
      return failure();
    grouped.tasks.push_back(std::move(task));
  }
  return grouped;
}

// Duplicate a bounded, pure integer Relation expression at a later indexed
// accumulation rather than transporting its tiny result across unrelated CTA
// ownership domains.  The forward Gather remains an independent consumer of
// the same expression.  This is legal only when every external user of the
// materialized index is an indexed accumulation, so pruning the temporary does
// not change any other consumer's storage or ordering contract.
FailureOr<DerivedPlan>
recomputeIntegerRelationsAtIndexedAccumulations(DerivedPlan initial) {
  llvm::SmallDenseMap<Value, int64_t> bufferForValue;
  for (auto [ordinal, buffer] : llvm::enumerate(initial.buffers))
    bufferForValue.try_emplace(buffer.value, ordinal);

  struct Candidate {
    int64_t indexBuffer;
    int64_t producerTask;
    SmallVector<int64_t> consumers;
    SmallVector<Operation *> relations;
    SmallVector<Value> boundaries;
  };
  SmallVector<Candidate> candidates;
  llvm::SmallDenseSet<int64_t> claimedConsumers;
  constexpr size_t kMaximumRecomputedRelations = 8;
  for (auto [bufferOrdinal, buffer] : llvm::enumerate(initial.buffers)) {
    auto indexType = dyn_cast<RankedTensorType>(buffer.value.getType());
    if (!buffer.producer || buffer.liveOut || buffer.consumers.empty() ||
        !indexType || !indexType.hasStaticShape() ||
        !indexType.getElementType().isInteger(32))
      continue;
    const int64_t producerOrdinal = *buffer.producer;
    if (producerOrdinal < 0 || producerOrdinal >= initial.tasks.size())
      continue;
    PlannedTask &producer = initial.tasks[producerOrdinal];
    if (producer.compositionKind != "relation_gather" ||
        producer.compositionSources.size() < 2 ||
        producer.compositionSources.size() - 1 >
            kMaximumRecomputedRelations ||
        !llvm::is_contained(producer.outputs,
                            static_cast<int64_t>(bufferOrdinal)))
      continue;
    auto gather = dyn_cast<RelationOp>(producer.compositionSources.back());
    if (!gather || gather.getSemantics() != RelationSemantics::Gather ||
        gather.getInputs().size() != 2 || gather.getInputs()[1] != buffer.value)
      continue;
    SmallVector<Operation *> relations(
        producer.compositionSources.begin(),
        std::prev(producer.compositionSources.end()));
    if (relations.empty() ||
        llvm::any_of(relations, [](Operation *operation) {
          auto relation = dyn_cast<RelationOp>(operation);
          return !relation || relation.getNumResults() != 1 ||
                 (relation.getSemantics() != RelationSemantics::BroadcastInDim &&
                  relation.getSemantics() != RelationSemantics::Add &&
                  relation.getSemantics() != RelationSemantics::Compare &&
                  relation.getSemantics() != RelationSemantics::Select);
        }) ||
        relations.back()->getResult(0) != buffer.value)
      continue;
    llvm::SmallDenseSet<Operation *> relationSet(relations.begin(),
                                                 relations.end());
    SmallVector<Value> boundaries;
    llvm::SmallDenseSet<Value> seen;
    bool boundaryMissing = false;
    for (Operation *operation : relations) {
      for (Value operand : operation->getOperands()) {
        if (relationSet.contains(operand.getDefiningOp()) ||
            !seen.insert(operand).second)
          continue;
        if (!bufferForValue.contains(operand)) {
          boundaryMissing = true;
          break;
        }
        boundaries.push_back(operand);
      }
    }
    if (boundaryMissing)
      continue;
    SmallVector<int64_t> consumers;
    bool validConsumers = true;
    for (int64_t consumerOrdinal : buffer.consumers) {
      if (consumerOrdinal < 0 || consumerOrdinal >= initial.tasks.size()) {
        validConsumers = false;
        break;
      }
      PlannedTask &consumer = initial.tasks[consumerOrdinal];
      if (consumer.compositionKind != "indexed_accumulation" ||
          consumer.compositionSources.size() != 4) {
        validConsumers = false;
        break;
      }
      auto contributions =
          dyn_cast<RelationOp>(consumer.compositionSources.front());
      if (!contributions ||
          contributions.getSemantics() !=
              RelationSemantics::IndexedContributions ||
          contributions.getInputs().size() != 2 ||
          contributions.getInputs()[1] != buffer.value ||
          !claimedConsumers.insert(consumerOrdinal).second) {
        validConsumers = false;
        break;
      }
      consumers.push_back(consumerOrdinal);
    }
    if (validConsumers)
      candidates.push_back(Candidate{static_cast<int64_t>(bufferOrdinal),
                                     producerOrdinal, std::move(consumers),
                                     std::move(relations),
                                     std::move(boundaries)});
  }
  if (candidates.empty())
    return failure();

  llvm::SmallDenseSet<int64_t> removedBuffers;
  for (Candidate &candidate : candidates) {
    removedBuffers.insert(candidate.indexBuffer);
    PlannedTask &producer = initial.tasks[candidate.producerTask];
    llvm::erase(producer.outputs, candidate.indexBuffer);
    for (int64_t consumerOrdinal : candidate.consumers) {
      PlannedTask &consumer = initial.tasks[consumerOrdinal];
      llvm::erase(consumer.inputs, candidate.indexBuffer);
      consumer.compositionRecomputedSources = candidate.relations;
      for (Value boundary : candidate.boundaries) {
        int64_t boundaryBuffer = bufferForValue.lookup(boundary);
        if (!llvm::is_contained(consumer.inputs, boundaryBuffer))
          consumer.inputs.push_back(boundaryBuffer);
      }
    }
  }

  SmallVector<int64_t> newBufferForOld(initial.buffers.size(), -1);
  DerivedPlan result;
  result.hasContract = initial.hasContract;
  result.hasRankNFold = initial.hasRankNFold;
  result.hasFoldRegion = initial.hasFoldRegion;
  result.hasContractRegion = initial.hasContractRegion;
  result.hasContractFoldComposition = initial.hasContractFoldComposition;
  result.hasLogicalRelation = initial.hasLogicalRelation;
  result.hasStatefulContractFoldPullback =
      initial.hasStatefulContractFoldPullback;
  result.normalizedExpContractRegions = initial.normalizedExpContractRegions;
  result.normalizedExpContractPullbacks =
      initial.normalizedExpContractPullbacks;
  for (auto [oldOrdinal, buffer] : llvm::enumerate(initial.buffers)) {
    if (removedBuffers.contains(oldOrdinal))
      continue;
    newBufferForOld[oldOrdinal] = result.buffers.size();
    PlannedBuffer rebuilt = buffer;
    rebuilt.consumers.clear();
    result.buffers.push_back(std::move(rebuilt));
  }
  for (auto [taskOrdinal, task] : llvm::enumerate(initial.tasks)) {
    for (int64_t &input : task.inputs) {
      if (input < 0 || input >= newBufferForOld.size() ||
          newBufferForOld[input] < 0)
        return failure();
      input = newBufferForOld[input];
      result.buffers[input].consumers.push_back(taskOrdinal);
    }
    for (int64_t &output : task.outputs) {
      if (output < 0 || output >= newBufferForOld.size() ||
          newBufferForOld[output] < 0)
        return failure();
      output = newBufferForOld[output];
    }
    task.dependencies.clear();
    for (int64_t input : task.inputs) {
      std::optional<int64_t> dependency = result.buffers[input].producer;
      if (dependency && *dependency != static_cast<int64_t>(taskOrdinal) &&
          !llvm::is_contained(task.dependencies, *dependency))
        task.dependencies.push_back(*dependency);
    }
    llvm::sort(task.dependencies);
    result.tasks.push_back(std::move(task));
  }
  for (PlannedBuffer &buffer : result.buffers) {
    llvm::sort(buffer.consumers);
    buffer.consumers.erase(
        std::unique(buffer.consumers.begin(), buffer.consumers.end()),
        buffer.consumers.end());
  }
  return result;
}

FailureOr<DerivedPlan> deriveHybridNormContractPlan(RegionOp region,
                                                    DerivedPlan generic) {
  SmallVector<Operation *> algebra;
  for (Operation &operation : region.getBody().front().without_terminator())
    algebra.push_back(&operation);
  auto regionYield =
      dyn_cast<YieldOp>(region.getBody().front().getTerminator());
  if (!regionYield || regionYield.getValues().size() < 2)
    return failure();

  std::optional<DerivedPlan> specialized;
  SmallVector<Operation *> specializedSources;
  Value specializedOutput;
  for (Value yielded : regionYield.getValues()) {
    llvm::SmallDenseSet<Operation *> closure;
    SmallVector<Value> worklist{yielded};
    bool closed = true;
    while (!worklist.empty() && closed) {
      Operation *producer = worklist.pop_back_val().getDefiningOp();
      if (!producer)
        continue;
      if (producer->getParentOfType<RegionOp>() != region ||
          !isa<MapOp, FoldOp, ContractOp>(producer)) {
        closed = false;
        break;
      }
      if (closure.insert(producer).second)
        llvm::append_range(worklist, producer->getOperands());
    }
    if (!closed)
      continue;
    SmallVector<Operation *> ordered;
    llvm::copy_if(
        algebra, std::back_inserter(ordered),
        [&](Operation *operation) { return closure.contains(operation); });
    if (ordered.size() != closure.size())
      continue;
    FailureOr<DerivedPlan> candidate =
        deriveNormContractSubplan(region, ordered, yielded);
    if (failed(candidate))
      continue;
    if (specialized)
      return failure();
    specialized = std::move(*candidate);
    specializedSources = std::move(ordered);
    specializedOutput = yielded;
  }
  if (!specialized)
    return failure();

  auto specializedContractIt =
      llvm::find_if(specialized->tasks, [](PlannedTask &task) {
        return !task.contractRegionSources.empty();
      });
  if (specializedContractIt == specialized->tasks.end())
    return failure();
  PlannedTask *specializedContract = &*specializedContractIt;
  if (!specializedContract->contractRegionBf16Boundary ||
      !specializedContract->contractRegionContract ||
      specializedContract->contractRegionEpilogueSources.size() != 1)
    return failure();
  Value normalized =
      cast<MapOp>(specializedContract->contractRegionBf16Boundary).getResult(0);
  Value contractIntermediate =
      cast<ContractOp>(specializedContract->contractRegionContract)
          .getResult(0);
  Operation *epilogue = specializedContract->contractRegionEpilogueSources[0];

  auto taskSources = [](const PlannedTask &task) {
    SmallVector<Operation *> sources;
    if (!task.contractRegionSources.empty()) {
      llvm::append_range(sources, task.contractRegionSources);
    } else if (!task.foldRegionSources.empty()) {
      llvm::append_range(sources, task.foldRegionSources);
    } else if (task.fusedFold) {
      sources.push_back(task.fusedFold);
      llvm::append_range(sources, task.fusedOperations);
    } else if (!task.fusedOperations.empty()) {
      llvm::append_range(sources, task.fusedOperations);
    } else {
      sources.push_back(task.operation);
    }
    return sources;
  };
  auto singletonTask = [&](Operation *operation) -> std::optional<int64_t> {
    std::optional<int64_t> found;
    for (auto [ordinal, task] : llvm::enumerate(generic.tasks)) {
      SmallVector<Operation *> sources = taskSources(task);
      if (sources.size() != 1 || sources.front() != operation)
        continue;
      if (found)
        return std::nullopt;
      found = ordinal;
    }
    return found;
  };
  std::optional<int64_t> boundaryTask =
      singletonTask(specializedContract->contractRegionBf16Boundary);
  std::optional<int64_t> contractTask =
      singletonTask(specializedContract->contractRegionContract);
  std::optional<int64_t> epilogueTask = singletonTask(epilogue);
  if (!boundaryTask || !contractTask || !epilogueTask ||
      !(*boundaryTask < *contractTask && *contractTask < *epilogueTask))
    return failure();

  DerivedPlan hybrid;
  hybrid.hasContract = generic.hasContract;
  hybrid.hasRankNFold = generic.hasRankNFold;
  hybrid.hasContractRegion = true;
  hybrid.hasFoldRegion =
      llvm::any_of(specialized->tasks, [](const PlannedTask &task) {
        return !task.foldRegionSources.empty();
      });
  SmallVector<int64_t> oldBufferToNew(generic.buffers.size(), -1);
  llvm::SmallDenseMap<Value, int64_t> combinedBufferForValue;
  for (auto [ordinal, buffer] : llvm::enumerate(generic.buffers)) {
    if (buffer.value == contractIntermediate)
      continue;
    int64_t combined = hybrid.buffers.size();
    oldBufferToNew[ordinal] = combined;
    combinedBufferForValue.try_emplace(buffer.value, combined);
    hybrid.buffers.push_back({buffer.value, std::nullopt, {}, buffer.liveOut});
  }
  auto normalizedBuffer = combinedBufferForValue.find(normalized);
  auto outputBuffer = combinedBufferForValue.find(specializedOutput);
  if (normalizedBuffer == combinedBufferForValue.end() ||
      outputBuffer == combinedBufferForValue.end() ||
      hybrid.buffers[normalizedBuffer->second].liveOut)
    return failure();

  SmallVector<int64_t> specializedBufferToCombined(specialized->buffers.size(),
                                                   -1);
  for (auto [ordinal, buffer] : llvm::enumerate(specialized->buffers)) {
    if (!buffer.producer || buffer.liveOut) {
      auto found = combinedBufferForValue.find(buffer.value);
      if (found == combinedBufferForValue.end())
        return failure();
      specializedBufferToCombined[ordinal] = found->second;
      continue;
    }
    specializedBufferToCombined[ordinal] = hybrid.buffers.size();
    hybrid.buffers.push_back({buffer.value, std::nullopt, {}, false});
  }

  auto remap = [](SmallVectorImpl<int64_t> &ordinals,
                  ArrayRef<int64_t> mapping) -> LogicalResult {
    for (int64_t &ordinal : ordinals) {
      if (ordinal < 0 || ordinal >= static_cast<int64_t>(mapping.size()) ||
          mapping[ordinal] < 0)
        return failure();
      ordinal = mapping[ordinal];
    }
    return success();
  };
  for (auto [ordinal, original] : llvm::enumerate(generic.tasks)) {
    if (ordinal == *boundaryTask) {
      for (PlannedTask task : specialized->tasks) {
        if (failed(remap(task.inputs, specializedBufferToCombined)) ||
            failed(remap(task.outputs, specializedBufferToCombined)))
          return failure();
        if (!task.contractRegionSources.empty())
          task.outputs.insert(task.outputs.begin(), normalizedBuffer->second);
        task.dependencies.clear();
        hybrid.tasks.push_back(std::move(task));
      }
      continue;
    }
    if (ordinal == *contractTask || ordinal == *epilogueTask)
      continue;
    PlannedTask task = original;
    if (failed(remap(task.inputs, oldBufferToNew)) ||
        failed(remap(task.outputs, oldBufferToNew)))
      return failure();
    task.dependencies.clear();
    hybrid.tasks.push_back(std::move(task));
  }

  for (PlannedBuffer &buffer : hybrid.buffers) {
    buffer.producer.reset();
    buffer.consumers.clear();
  }
  for (auto [ordinal, task] : llvm::enumerate(hybrid.tasks)) {
    for (int64_t output : task.outputs) {
      if (output < 0 || output >= static_cast<int64_t>(hybrid.buffers.size()) ||
          hybrid.buffers[output].producer)
        return failure();
      hybrid.buffers[output].producer = ordinal;
    }
  }

  SmallVector<bool> requiredTasks(hybrid.tasks.size(), false);
  SmallVector<bool> requiredBuffers(hybrid.buffers.size(), false);
  SmallVector<int64_t> worklist;
  for (auto [ordinal, buffer] : llvm::enumerate(hybrid.buffers)) {
    if (!buffer.liveOut)
      continue;
    requiredBuffers[ordinal] = true;
    worklist.push_back(ordinal);
  }
  while (!worklist.empty()) {
    int64_t bufferOrdinal = worklist.pop_back_val();
    std::optional<int64_t> producer = hybrid.buffers[bufferOrdinal].producer;
    if (!producer || requiredTasks[*producer])
      continue;
    requiredTasks[*producer] = true;
    PlannedTask &task = hybrid.tasks[*producer];
    for (int64_t output : task.outputs) {
      if (output < 0 || output >= static_cast<int64_t>(hybrid.buffers.size()))
        return failure();
      requiredBuffers[output] = true;
    }
    for (int64_t input : task.inputs) {
      if (input < 0 || input >= static_cast<int64_t>(hybrid.buffers.size()))
        return failure();
      if (requiredBuffers[input])
        continue;
      requiredBuffers[input] = true;
      worklist.push_back(input);
    }
  }

  SmallVector<int64_t> liveBufferMapping(hybrid.buffers.size(), -1);
  DerivedPlan live;
  live.hasContract = hybrid.hasContract;
  live.hasRankNFold = false;
  live.hasContractRegion = hybrid.hasContractRegion;
  live.hasFoldRegion = hybrid.hasFoldRegion;
  for (auto [ordinal, buffer] : llvm::enumerate(hybrid.buffers)) {
    if (!requiredBuffers[ordinal])
      continue;
    liveBufferMapping[ordinal] = live.buffers.size();
    live.buffers.push_back({buffer.value, std::nullopt, {}, buffer.liveOut});
  }
  for (auto [ordinal, task] : llvm::enumerate(hybrid.tasks)) {
    if (!requiredTasks[ordinal])
      continue;
    if (failed(remap(task.inputs, liveBufferMapping)) ||
        failed(remap(task.outputs, liveBufferMapping)))
      return failure();
    task.dependencies.clear();
    if (task.kind == MaterializationTaskKind::Fold &&
        task.foldRegionSources.empty() && task.domain.size() > 2)
      live.hasRankNFold = true;
    live.tasks.push_back(std::move(task));
  }
  hybrid = std::move(live);

  llvm::SmallDenseMap<Operation *, int64_t> sourceOccurrences;
  for (PlannedTask &task : hybrid.tasks) {
    for (Operation *operation : taskSources(task))
      ++sourceOccurrences[operation];
  }
  for (PlannedTask &task : hybrid.tasks) {
    const bool regionTask =
        !task.foldRegionSources.empty() || !task.contractRegionSources.empty();
    task.foldRegionRecomputedSources.clear();
    task.contractRegionRecomputedSources.clear();
    SmallVector<Operation *> sources = taskSources(task);
    for (Operation *operation : sources) {
      if (sourceOccurrences.lookup(operation) == 1 || !regionTask)
        continue;
      if (!isa<MapOp, FoldOp>(operation))
        return failure();
      if (!task.foldRegionSources.empty())
        task.foldRegionRecomputedSources.push_back(operation);
      else
        task.contractRegionRecomputedSources.push_back(operation);
    }
  }

  for (auto [ordinal, task] : llvm::enumerate(hybrid.tasks)) {
    for (int64_t output : task.outputs) {
      if (output < 0 || output >= static_cast<int64_t>(hybrid.buffers.size()) ||
          hybrid.buffers[output].producer)
        return failure();
      hybrid.buffers[output].producer = ordinal;
    }
  }
  for (auto [ordinal, task] : llvm::enumerate(hybrid.tasks)) {
    for (int64_t input : task.inputs) {
      if (input < 0 || input >= static_cast<int64_t>(hybrid.buffers.size()))
        return failure();
      PlannedBuffer &buffer = hybrid.buffers[input];
      if (buffer.consumers.empty() || buffer.consumers.back() != ordinal)
        buffer.consumers.push_back(ordinal);
      if (buffer.producer &&
          !llvm::is_contained(task.dependencies, *buffer.producer))
        task.dependencies.push_back(*buffer.producer);
    }
    llvm::sort(task.dependencies);
  }
  for (const PlannedTask &task : hybrid.tasks) {
    if (llvm::none_of(task.outputs, [&](int64_t output) {
          const PlannedBuffer &buffer = hybrid.buffers[output];
          return buffer.liveOut || !buffer.consumers.empty();
        }))
      return failure();
  }
  return hybrid;
}

FailureOr<DerivedPlan> deriveGpuPlan(RegionOp region,
                                     bool preferFusion = false) {
  SmallVector<Operation *> algebra;
  for (Operation &operation : region.getBody().front().without_terminator()) {
    if (!isa<MapOp, RelationOp, SelectionOp, FoldOp, ContractOp, ScanOp>(
            operation)) {
      return failure();
    }
    algebra.push_back(&operation);
  }
  if (algebra.empty()) {
    return failure();
  }
  if (preferFusion) {
    FailureOr<DerivedPlan> normContract = deriveNormContractPlan(region);
    if (succeeded(normContract))
      return normContract;
  }
  const bool containsContract = llvm::any_of(
      algebra, [](Operation *operation) { return isa<ContractOp>(operation); });
  const bool containsFold = llvm::any_of(
      algebra, [](Operation *operation) { return isa<FoldOp>(operation); });
  // A Fold-to-Contract composition needs the generic domains preserved while
  // its materialized baseline is assembled.  Whether it wins is decided later
  // by deriveGenericContractFoldCompositionPlan's resource comparison.
  const bool genericFoldContractCandidate = containsFold && containsContract;
  const bool containsRelation = llvm::any_of(
      algebra, [](Operation *operation) { return isa<RelationOp>(operation); });
  const bool containsIndexedContributions =
      llvm::any_of(algebra, [](Operation *operation) {
        auto relation = dyn_cast<RelationOp>(operation);
        return relation && relation.getSemantics() ==
                               RelationSemantics::IndexedContributions;
      });
  bool componentizedProgram = false;
  region->getParentOfType<ModuleOp>().walk([&](RelationOp relation) {
    componentizedProgram |=
        relation.getSemantics() == RelationSemantics::IndexedContributions;
  });
  const bool containsSelection =
      llvm::any_of(algebra, [](Operation *operation) {
        return isa<SelectionOp>(operation);
      });
  const bool containsScan = llvm::any_of(
      algebra, [](Operation *operation) { return isa<ScanOp>(operation); });
  const bool hasNormalizedExponentialPullback = succeeded(
      online_planner::describeNormalizedExponentialContractPullbackCandidates(
          region));
  if (!componentizedProgram && !preferFusion && !containsScan &&
      !genericFoldContractCandidate &&
      containsContract &&
      !isClosedContractMapRegion(region, algebra)) {
    return failure();
  }
  const bool containsRankNFold =
      llvm::any_of(algebra, [&](Operation *operation) {
        auto fold = dyn_cast<FoldOp>(operation);
        if (!fold)
          return false;
        if (preferFusion) {
          FailureOr<SmallVector<int64_t>> domain =
              planningFoldDomainShape(fold);
          return succeeded(domain) && domain->size() > 2;
        }
        auto input =
            fold.getInputs().size() == 1
                ? dyn_cast<RankedTensorType>(fold.getInputs().front().getType())
                : RankedTensorType{};
        return input && input.getRank() > 2;
      });
  if (!componentizedProgram && !preferFusion && !containsScan &&
      !genericFoldContractCandidate &&
      containsRankNFold &&
      (containsContract || !isClosedRankNFoldMapRegion(region, algebra))) {
    return failure();
  }
  SmallVector<PlannedBuffer, 0> buffers;
  llvm::SmallDenseMap<Value, int64_t> bufferForValue;
  for (BlockArgument input : region.getBody().front().getArguments()) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    if (!type || !type.hasStaticShape() ||
        llvm::any_of(type.getShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      return failure();
    }
    bufferForValue.try_emplace(input, buffers.size());
    buffers.push_back({input, std::nullopt, {}, false});
  }

  SmallVector<PlannedTask, 0> tasks;
  unsigned foldCount = 0;
  unsigned contractCount = 0;
  for (Operation *operation : algebra) {
    PlannedTask task;
    task.operation = operation;
    task.kind =
        isa<FoldOp>(operation)
            ? MaterializationTaskKind::Fold
            : (isa<ContractOp>(operation)   ? MaterializationTaskKind::Contract
               : isa<RelationOp>(operation) ? MaterializationTaskKind::Relation
               : isa<SelectionOp>(operation)
                   ? MaterializationTaskKind::Selection
               : isa<ScanOp>(operation) ? MaterializationTaskKind::Scan
                   : MaterializationTaskKind::Map);
    if (auto fold = dyn_cast<FoldOp>(operation)) {
      const bool closedFold = algebraSourceExpansion(fold)
                                  ? isClosedSourceOrderedFloatingAddFold(fold)
                              : (componentizedProgram || preferFusion || containsScan ||
                                 genericFoldContractCandidate)
                                  ? isClosedMultiStatePlanningFold(fold)
                              : containsRankNFold ? isClosedRankNFold(fold)
                                                  : isClosedRankTwoFold(fold);
      if (!closedFold) {
        return failure();
      }
      ++foldCount;
      task.reductionDimensions.assign(fold.getReductionDimensions().begin(),
                                      fold.getReductionDimensions().end());
      task.orderFree = fold.getOrderFree();
      if (preferFusion || genericFoldContractCandidate) {
        FailureOr<SmallVector<int64_t>> domain = planningFoldDomainShape(fold);
        if (failed(domain))
          return failure();
        task.domain = *domain;
      } else {
        auto input =
            fold.getInputs().size() == 1
                ? dyn_cast<RankedTensorType>(fold.getInputs().front().getType())
                : RankedTensorType{};
        if (!input)
          return failure();
        task.domain.assign(input.getShape().begin(), input.getShape().end());
      }
    } else if (auto contract = dyn_cast<ContractOp>(operation)) {
      if (!componentizedProgram && !preferFusion && !containsScan &&
          !genericFoldContractCandidate &&
          !isClosedProjectedContract(contract)) {
        return failure();
      }
      ++contractCount;
      FailureOr<SmallVector<int64_t>> domain = contractDomainShape(contract);
      if (failed(domain)) {
        return failure();
      }
      task.domain = std::move(*domain);
      for (auto [position, iterator] :
           llvm::enumerate(contract.getIteratorKinds())) {
        if (cast<StringAttr>(iterator).getValue() == "reduction") {
          task.reductionDimensions.push_back(position);
        }
      }
    } else if (auto scan = dyn_cast<ScanOp>(operation)) {
      // This is the coarse ordered task. Richer ownership and contraction
      // tiling are refinements of the typed Scan body, not recognition here.
      task.domain = {static_cast<int64_t>(scan.getExtent())};
      task.compositionKind = "scan";
      task.compositionSources = {operation};
      task.compositionStreamDimensions = {0};
    } else if (auto map = dyn_cast<MapOp>(operation)) {
      FailureOr<SmallVector<int64_t>> domain =
          (componentizedProgram || preferFusion || containsScan ||
           genericFoldContractCandidate)
              ? genericMapDomainShape(map)
                                                 : mapDomainShape(map);
      if (failed(domain)) {
        return failure();
      }
      task.domain = std::move(*domain);
    } else if (auto selection = dyn_cast<SelectionOp>(operation)) {
      auto input = dyn_cast<RankedTensorType>(selection.getInput().getType());
      const int64_t dimension = selection.getDimension();
      if (!input || !input.hasStaticShape() || input.getRank() == 0 ||
          input.getRank() > 8 || dimension < 0 ||
          dimension >= input.getRank() ||
          llvm::any_of(
              input.getShape(), [](int64_t extent) { return extent <= 0; }))
        return failure();
      task.domain.assign(input.getShape().begin(), input.getShape().end());
      task.reductionDimensions.push_back(dimension);
    } else {
      auto relation = cast<RelationOp>(operation);
      if (relation.getNumResults() != 1)
        return failure();
      auto result = dyn_cast<RankedTensorType>(relation.getResult(0).getType());
      if (!result || !result.hasStaticShape() || result.getRank() > 8 ||
          llvm::any_of(result.getShape(),
                       [](int64_t extent) { return extent <= 0; }))
        return failure();
      task.domain.assign(result.getShape().begin(), result.getShape().end());
    }
    for (Value operand : operation->getOperands()) {
      auto found = bufferForValue.find(operand);
      if (found == bufferForValue.end()) {
        return failure();
      }
      task.inputs.push_back(found->second);
      if (buffers[found->second].consumers.empty() ||
          buffers[found->second].consumers.back() !=
              static_cast<int64_t>(tasks.size())) {
        buffers[found->second].consumers.push_back(tasks.size());
      }
      if (buffers[found->second].producer &&
          !llvm::is_contained(task.dependencies,
                              *buffers[found->second].producer)) {
        task.dependencies.push_back(*buffers[found->second].producer);
      }
    }
    llvm::sort(task.dependencies);
    for (Value result : operation->getResults()) {
      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type || !type.hasStaticShape() ||
          llvm::any_of(type.getShape(),
                       [](int64_t extent) { return extent <= 0; })) {
        return failure();
      }
      int64_t ordinal = buffers.size();
      bufferForValue.try_emplace(result, ordinal);
      buffers.push_back(
          {result, static_cast<int64_t>(tasks.size()), {}, false});
      task.outputs.push_back(ordinal);
    }
    if (auto scan = dyn_cast<ScanOp>(operation)) {
      for (int64_t position : scan.getStateOperands())
        task.compositionStateInputs.push_back(task.inputs[position]);
      for (int64_t position : scan.getStateResults())
        task.compositionStateOutputs.push_back(task.outputs[position]);
    }
    tasks.push_back(std::move(task));
  }
  auto regionYield = cast<YieldOp>(region.getBody().front().getTerminator());
  for (Value value : regionYield.getValues()) {
    auto found = bufferForValue.find(value);
    if (found == bufferForValue.end()) {
      return failure();
    }
    buffers[found->second].liveOut = true;
  }
  if (!componentizedProgram && !preferFusion && !containsSelection &&
      !containsScan) {
    for (const PlannedBuffer &buffer : buffers) {
      if (!buffer.liveOut && buffer.consumers.empty()) {
        return failure();
      }
    }
  }

  const bool containsOnlyMaps = foldCount == 0 && contractCount == 0 &&
                                !containsScan &&
                                !containsRelation && !containsSelection;
  const bool standaloneMap =
      containsOnlyMaps && algebra.size() == 1 &&
      (cast<MapOp>(algebra.front()).getSemantics() != MapSemantics::Pointwise ||
       (algebra.front()->getNumOperands() == 0 &&
        llvm::all_of(algebra.front()->getResultTypes(), [](Type type) {
          auto tensor = dyn_cast<RankedTensorType>(type);
          return tensor && tensor.getRank() == 0;
        })));
  if (!componentizedProgram && containsOnlyMaps && !standaloneMap) {
    for (Operation *operation : algebra) {
      auto map = cast<MapOp>(operation);
      if (map.getSemantics() != MapSemantics::Pointwise ||
          !llvm::all_of(
              map.getIndexingMaps().getValue(), [](Attribute indexing) {
                return isIdentityMap(cast<AffineMapAttr>(indexing).getValue());
              })) {
        return failure();
      }
    }
    // Map tasks preserve their typed results in separate materialized
    // buffers. BF16 scalar emission rounds every operation before storage,
    // including live outputs that subsequent tasks read through their result
    // binding. Neither F32 temporaries nor a dead-at-yield output is required.
    for (const PlannedBuffer &buffer : buffers) {
      auto type = cast<RankedTensorType>(buffer.value.getType());
      if (type.getRank() > 8 ||
          (type.getRank() == 0 && type.getElementType().isBF16()) ||
          (!type.getElementType().isF32() && !type.getElementType().isBF16())) {
        return failure();
      }
    }
  }
  for (const PlannedTask &task : tasks) {
    if (llvm::none_of(task.outputs, [&](int64_t output) {
          const PlannedBuffer &buffer = buffers[output];
          return buffer.liveOut || !buffer.consumers.empty();
        })) {
      return failure();
    }
  }

  // A typed Scan region can retain loop-invariant constants and layout values
  // as independent live-outs for a later reverse Scan.  Exact task
  // dependencies already represent those roots; requiring the whole region
  // to be one undirected component would reject ordinary JAX-owned AD graphs.
  if (!componentizedProgram && !preferFusion && !containsScan) {
    llvm::SmallDenseSet<Operation *> connected;
    SmallVector<Operation *> worklist{algebra.front()};
    connected.insert(algebra.front());
    while (!worklist.empty()) {
      Operation *operation = worklist.pop_back_val();
      for (Operation *candidate : algebra) {
        const bool adjacent =
            llvm::any_of(candidate->getOperands(),
                         [&](Value value) {
                           return value.getDefiningOp() == operation;
                         }) ||
            llvm::any_of(operation->getOperands(), [&](Value value) {
              return value.getDefiningOp() == candidate;
            });
        if (adjacent && connected.insert(candidate).second) {
          worklist.push_back(candidate);
        }
      }
    }
    if (connected.size() != algebra.size()) {
      return failure();
    }
  }
  DerivedPlan result{std::move(tasks), std::move(buffers), contractCount != 0,
                     containsRankNFold};
  result.hasLogicalRelation = containsRelation || containsSelection;
  result.hasStatefulContractFoldPullback = hasNormalizedExponentialPullback;
  if (containsIndexedContributions) {
    FailureOr<DerivedPlan> grouped =
        groupIndexedAccumulationExpansions(algebra, result);
    if (succeeded(grouped))
      result = std::move(*grouped);
  }
  if (!preferFusion) {
    FailureOr<DerivedPlan> composition =
        deriveGenericContractFoldCompositionPlan(region, result);
    return succeeded(composition) ? std::move(*composition) : std::move(result);
  }
  if (hasNormalizedExponentialPullback) {
    FailureOr<DerivedPlan> reverse =
        online_planner::deriveNormalizedExponentialReverseFamily(region,
                                                                 result);
    if (succeeded(reverse))
      return reverse;
    return failure();
  }
  FailureOr<DerivedPlan> composition =
      deriveGenericContractFoldCompositionPlan(region, result);
  if (succeeded(composition))
    return composition;
  if (containsRelation || containsSelection) {
    FailureOr<DerivedPlan> related = groupIntegerRelationCompositions(result);
    if (succeeded(related)) {
      result = std::move(*related);
      FailureOr<DerivedPlan> recomputed =
          recomputeIntegerRelationsAtIndexedAccumulations(result);
      if (succeeded(recomputed))
        result = std::move(*recomputed);
    }
    FailureOr<DerivedPlan> fusedMaps = fuseEligibleFloatingMapTasks(result);
    return succeeded(fusedMaps) ? std::move(*fusedMaps) : std::move(result);
  }
  FailureOr<DerivedPlan> foldRegion = deriveFoldRegionPlan(region);
  if (succeeded(foldRegion))
    return foldRegion;
  FailureOr<DerivedPlan> sharedStatFoldRegion =
      deriveSharedStatFoldRegionPlan(region);
  if (succeeded(sharedStatFoldRegion))
    return sharedStatFoldRegion;
  if (!containsFold) {
    FailureOr<DerivedPlan> phasedMapContract =
        fuseEligibleFloatingMapTasks(result,
                                     /*requireMapContractInput=*/true);
    if (succeeded(phasedMapContract))
      return phasedMapContract;
  }
  FailureOr<DerivedPlan> fused =
      fuseEligibleMapTasks(region, std::move(result));
  if (failed(fused))
    return failure();
  FailureOr<DerivedPlan> hybrid = deriveHybridNormContractPlan(region, *fused);
  if (succeeded(hybrid))
    return hybrid;
  return fused;
}

LogicalResult normalizePlainTaskInputs(DerivedPlan &plan) {
  for (PlannedTask &task : plan.tasks) {
    auto fold = dyn_cast_or_null<FoldOp>(task.operation);
    if (!task.fusedOperations.empty() || !task.foldRegionSources.empty() ||
        !task.contractRegionSources.empty() || !task.compositionKind.empty() ||
        task.statefulContractFoldComposition)
      continue;
    SmallVector<Value> sourceInputs;
    if (fold) {
      llvm::append_range(sourceInputs, fold.getInputs());
      llvm::append_range(sourceInputs, fold.getInitializers());
    } else {
      llvm::append_range(sourceInputs, task.operation->getOperands());
    }
    SmallVector<int64_t> orderedInputs;
    for (Value value : sourceInputs) {
      auto found =
          llvm::find_if(plan.buffers, [&](const PlannedBuffer &buffer) {
            return buffer.value == value;
          });
      if (found == plan.buffers.end())
        return failure();
      orderedInputs.push_back(std::distance(plan.buffers.begin(), found));
    }
    task.inputs = std::move(orderedInputs);
  }
  return success();
}

Operation *createMaterializationRecord(OpBuilder &builder, Location location,
                                       StringRef name,
                                       ArrayRef<NamedAttribute> attrs) {
  OperationState state(location, name);
  state.addAttributes(attrs);
  return builder.create(state);
}

class PlanRowFoldMaterializationPass
    : public impl::ShuttlePlanRowFoldMaterializationPassBase<
          PlanRowFoldMaterializationPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module.getOps<MaterializationPlanOp>().empty()) {
      module.emitError("already contains a materialization plan");
      return signalPassFailure();
    }
    auto manifest =
        module->getAttrOfType<DictionaryAttr>(kCoverageManifestAttribute);
    auto tuning = manifest ? manifest.getAs<StringAttr>("canonical_tuning")
                           : StringAttr{};
    auto canonicalOptions =
        manifest ? manifest.getAs<StringAttr>("canonical_options")
                 : StringAttr{};
    const bool materializationTuningAbi =
        !canonicalOptions ||
        canonicalOptions.getValue().contains(
            "\"pipeline_abi_version\":22,\"schema_version\":1") ||
        canonicalOptions.getValue().contains(
            "\"pipeline_abi_version\":23,\"schema_version\":1");
    const bool validTuning =
        tuning && (tuning.getValue() == kDefaultCanonicalTuning ||
                   tuning.getValue() == kFusionCanonicalTuning ||
                   tuning.getValue() == kMaterializationCanonicalTuning);
    if (!validTuning || (!materializationTuningAbi &&
                         tuning.getValue() != kDefaultCanonicalTuning)) {
      module.emitError(
          "requires an exact ABI21 default or ABI22/ABI23 materialization "
          "tuning");
      return signalPassFailure();
    }
    const bool preferFusion = tuning.getValue() == kFusionCanonicalTuning;
    if (preferFusion) {
      // Physical planning consumes the canonical associative Fold state. Keep
      // direct users of this pass on the same algebra path as the registered
      // GPU pipeline instead of falling back to a second physical recognizer.
      SmallVector<RegionOp> regions;
      module.walk([&](RegionOp region) { regions.push_back(region); });
      for (RegionOp region : regions) {
        bool changed = true;
        while (changed) {
          changed = false;
          for (MapOp map : llvm::make_early_inc_range(
                   region.getBody().front().getOps<MapOp>())) {
            if (rewriteNormalizedExponentialSlice(map)) {
              changed = true;
              break;
            }
          }
        }
      }
    }
    SmallVector<std::pair<RegionOp, DerivedPlan>, 0> selected;
    bool invalidRegion = false;
    module.walk([&](RegionOp region) {
      FailureOr<DerivedPlan> candidate = deriveGpuPlan(region, preferFusion);
      if (failed(candidate)) {
        region.emitError(
            "cannot derive a bounded generic task plan for this algebra "
            "component");
        invalidRegion = true;
        return WalkResult::interrupt();
      }
      if (failed(normalizePlainTaskInputs(*candidate))) {
        region.emitError(
            "cannot bind algebra operands to materialized buffers");
        invalidRegion = true;
        return WalkResult::interrupt();
      }
      selected.emplace_back(region, std::move(*candidate));
      return WalkResult::advance();
    });
    uint64_t aggregateTasks = 0;
    uint64_t aggregateBuffers = 0;
    for (const auto &[region, derived] : selected) {
      (void)region;
      aggregateTasks += derived.tasks.size();
      aggregateBuffers += derived.buffers.size();
    }
    bool componentizedProgram = false;
    module.walk([&](RelationOp relation) {
      componentizedProgram |=
          relation.getSemantics() == RelationSemantics::IndexedContributions;
    });
    if (invalidRegion || selected.empty() ||
        selected.size() > kMaximumGpuExecutableRecords ||
        (!componentizedProgram &&
         (aggregateTasks > kMaximumGpuExecutableRecords ||
          aggregateBuffers > kMaximumGpuExecutableRecords))) {
      module.emitError(
          "requires a bounded set of independently connected static GPU "
          "algebra regions");
      return signalPassFailure();
    }
    const bool mixesContractAndRankNFold =
        llvm::any_of(selected,
                     [](const auto &entry) {
                       return entry.second.hasContract &&
                              !entry.second.hasContractFoldComposition &&
                              !entry.second.hasLogicalRelation;
                     }) &&
        llvm::any_of(selected, [](const auto &entry) {
          return entry.second.hasRankNFold &&
                 !entry.second.hasContractFoldComposition &&
                 !entry.second.hasLogicalRelation;
        });
    if (!componentizedProgram && mixesContractAndRankNFold) {
      module.emitError(
          "requires a bounded set of independently connected static GPU "
          "algebra regions");
      return signalPassFailure();
    }
    for (auto &[selectedRegion, derived] : selected) {
      OpBuilder builder(module.getContext());
      builder.setInsertionPointToEnd(module.getBody());
      OperationState planState(module.getLoc(),
                               MaterializationPlanOp::getOperationName());
      planState.addAttribute(
          "schema_version",
          builder.getI64IntegerAttr(
              derived.hasContract || derived.hasFoldRegion ||
                      llvm::any_of(
                          derived.tasks,
                          [](const PlannedTask &task) {
                            return !task.compositionKind.empty() ||
                                   task.statefulContractFoldComposition;
                          })
                  ? 2
                  : 1));
      planState.addAttribute("policy", selectedRegion.getPolicyAttr());
      if (componentizedProgram) {
        planState.addAttribute("componentized", builder.getBoolAttr(true));
      }
      if (derived.hasStatefulContractFoldPullback) {
        planState.addAttribute("stateful_contract_fold_pullback",
                               builder.getBoolAttr(true));
      }
      if (!derived.normalizedExpContractRegions.empty() ||
          !derived.normalizedExpContractPullbacks.empty()) {
        llvm::SmallDenseMap<Operation *, int64_t> operationOrdinals;
        int64_t operationOrdinal = 0;
        for (Operation &operation :
             selectedRegion.getBody().front().without_terminator()) {
          if (operation.getNumResults() == 0)
            continue;
          operationOrdinals.try_emplace(&operation, operationOrdinal++);
        }
        auto operationIndex = [&](Operation *operation) -> int64_t {
          auto found = operationOrdinals.find(operation);
          assert(found != operationOrdinals.end() &&
                 "planned normalized-exp operation must belong to its region");
          return found->second;
        };
        auto resultIndex = [](Value value) -> int64_t {
          return cast<OpResult>(value).getResultNumber();
        };
        SmallVector<Attribute> regionRecords;
        for (auto [ordinal, capability] :
             llvm::enumerate(derived.normalizedExpContractRegions)) {
          SmallVector<int64_t> foldOperations;
          for (Operation *operation : capability.foldOperations)
            foldOperations.push_back(operationIndex(operation));
          SmallVector<NamedAttribute> fields{
              builder.getNamedAttr("ordinal",
                                   builder.getI64IntegerAttr(ordinal)),
              builder.getNamedAttr("producer_operation",
                                   builder.getI64IntegerAttr(
                                       operationIndex(capability.producer))),
              builder.getNamedAttr("score_operation",
                                   builder.getI64IntegerAttr(operationIndex(
                                       capability.score.getDefiningOp()))),
              builder.getNamedAttr(
                  "score_result",
                  builder.getI64IntegerAttr(resultIndex(capability.score))),
              builder.getNamedAttr(
                  "fold_operations",
                  builder.getDenseI64ArrayAttr(foldOperations)),
              builder.getNamedAttr("maximum_operation",
                                   builder.getI64IntegerAttr(operationIndex(
                                       capability.maximum.getDefiningOp()))),
              builder.getNamedAttr(
                  "maximum_result",
                  builder.getI64IntegerAttr(resultIndex(capability.maximum))),
              builder.getNamedAttr("normalizer_operation",
                                   builder.getI64IntegerAttr(operationIndex(
                                       capability.normalizer.getDefiningOp()))),
              builder.getNamedAttr("normalizer_result",
                                   builder.getI64IntegerAttr(
                                       resultIndex(capability.normalizer))),
              builder.getNamedAttr(
                  "normalized_weight_operation",
                  builder.getI64IntegerAttr(operationIndex(
                      capability.normalizedWeight.getDefiningOp()))),
              builder.getNamedAttr("normalized_weight_result",
                                   builder.getI64IntegerAttr(resultIndex(
                                       capability.normalizedWeight))),
              builder.getNamedAttr("consumer_operation",
                                   builder.getI64IntegerAttr(
                                       operationIndex(capability.consumer))),
              builder.getNamedAttr(
                  "normalized_operand",
                  builder.getI64IntegerAttr(capability.normalizedOperand)),
              builder.getNamedAttr("finalizer_operation",
                                   builder.getI64IntegerAttr(operationIndex(
                                       capability.finalizer.getDefiningOp()))),
              builder.getNamedAttr("finalizer_result",
                                   builder.getI64IntegerAttr(
                                       resultIndex(capability.finalizer)))};
          if (capability.outputResult)
            fields.push_back(builder.getNamedAttr(
                "output_result",
                builder.getI64IntegerAttr(*capability.outputResult)));
          regionRecords.push_back(builder.getDictionaryAttr(fields));
        }
        if (!regionRecords.empty())
          planState.addAttribute("normalized_exp_contract_regions",
                                 builder.getArrayAttr(regionRecords));

        SmallVector<Attribute> pullbackRecords;
        for (auto [ordinal, capability] :
             llvm::enumerate(derived.normalizedExpContractPullbacks)) {
          SmallVector<int64_t> terminalContracts;
          for (Operation *operation : capability.terminalContracts)
            terminalContracts.push_back(operationIndex(operation));
          SmallVector<NamedAttribute> fields{
              builder.getNamedAttr("ordinal",
                                   builder.getI64IntegerAttr(ordinal)),
              builder.getNamedAttr(
                  "forward_region_ordinal",
                  builder.getI64IntegerAttr(capability.forwardRegionOrdinal)),
              builder.getNamedAttr("forward_task", builder.getI64IntegerAttr(
                                                       capability.forwardTask)),
              builder.getNamedAttr(
                  "owner_pullback_task",
                  builder.getI64IntegerAttr(capability.ownerPullbackTask)),
              builder.getNamedAttr(
                  "streamed_pullback_task",
                  builder.getI64IntegerAttr(capability.streamedPullbackTask)),
              builder.getNamedAttr(
                  "maximum_state_buffer",
                  builder.getI64IntegerAttr(capability.maximumStateBuffer)),
              builder.getNamedAttr(
                  "normalizer_state_buffer",
                  builder.getI64IntegerAttr(capability.normalizerStateBuffer)),
              builder.getNamedAttr("output_dependent_state_buffer",
                                   builder.getI64IntegerAttr(
                                       capability.outputDependentStateBuffer)),
              builder.getNamedAttr(
                  "output_cotangent_buffer",
                  builder.getI64IntegerAttr(capability.outputCotangentBuffer)),
              builder.getNamedAttr("terminal_cotangent_buffers",
                                   builder.getDenseI64ArrayAttr(
                                       capability.terminalCotangentBuffers)),
              builder.getNamedAttr("normalized_output_operation",
                                   builder.getI64IntegerAttr(operationIndex(
                                       capability.normalizedOutput))),
              builder.getNamedAttr("post_normalization_vjp_operation",
                                   builder.getI64IntegerAttr(operationIndex(
                                       capability.postNormalizationVjp))),
              builder.getNamedAttr("output_product_operation",
                                   builder.getI64IntegerAttr(operationIndex(
                                       capability.outputProduct))),
              builder.getNamedAttr("output_dependent_fold_operation",
                                   builder.getI64IntegerAttr(operationIndex(
                                       capability.outputDependentFold))),
              builder.getNamedAttr("consumer_weight_cotangent_operation",
                                   builder.getI64IntegerAttr(operationIndex(
                                       capability.consumerWeightCotangent))),
              builder.getNamedAttr("normalized_weight_operation",
                                   builder.getI64IntegerAttr(operationIndex(
                                       capability.normalizedWeight))),
              builder.getNamedAttr("normalized_exp_pullback_operation",
                                   builder.getI64IntegerAttr(operationIndex(
                                       capability.normalizedExpPullback))),
              builder.getNamedAttr("score_map_vjp_operation",
                                   builder.getI64IntegerAttr(
                                       operationIndex(capability.scoreMapVjp))),
              builder.getNamedAttr(
                  "terminal_contract_operations",
                  builder.getDenseI64ArrayAttr(terminalContracts))};
          if (capability.objectiveTask)
            fields.push_back(builder.getNamedAttr(
                "objective_task",
                builder.getI64IntegerAttr(*capability.objectiveTask)));
          if (capability.forwardResultBuffer)
            fields.push_back(builder.getNamedAttr(
                "forward_result_buffer",
                builder.getI64IntegerAttr(*capability.forwardResultBuffer)));
          pullbackRecords.push_back(builder.getDictionaryAttr(fields));
        }
        if (!pullbackRecords.empty())
          planState.addAttribute("normalized_exp_contract_pullbacks",
                                 builder.getArrayAttr(pullbackRecords));
      }
      planState.addAttribute("fingerprint",
                             builder.getStringAttr(std::string(64, '0')));
      planState.addRegion();
      auto plan = cast<MaterializationPlanOp>(builder.create(planState));
      Block *body = new Block();
      plan.getBody().push_back(body);
      builder.setInsertionPointToEnd(body);

      auto &tasks = derived.tasks;
      auto &buffers = derived.buffers;
      for (auto [ordinal, buffer] : llvm::enumerate(buffers)) {
        int64_t end = buffer.liveOut ? tasks.size()
                                     : (buffer.consumers.empty()
                                            ? buffer.producer.value_or(0)
                                            : buffer.consumers.back());
        SmallVector<NamedAttribute> attrs{
            builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr("tensor_type",
                                 TypeAttr::get(buffer.value.getType())),
            builder.getNamedAttr("storage",
                                 MaterializationStorageAttr::get(
                                     module.getContext(),
                                     (!buffer.producer || buffer.liveOut)
                                         ? MaterializationStorage::External
                                         : MaterializationStorage::Temporary)),
            builder.getNamedAttr("live_in",
                                 builder.getBoolAttr(!buffer.producer)),
            builder.getNamedAttr("live_out",
                                 builder.getBoolAttr(buffer.liveOut)),
            builder.getNamedAttr(
                "consumers",
                DenseI64ArrayAttr::get(module.getContext(), buffer.consumers)),
            builder.getNamedAttr(
                "lifetime_start",
                builder.getI64IntegerAttr(buffer.producer.value_or(0))),
            builder.getNamedAttr("lifetime_end",
                                 builder.getI64IntegerAttr(end))};
        if (buffer.producer) {
          attrs.push_back(builder.getNamedAttr(
              "producer", builder.getI64IntegerAttr(*buffer.producer)));
        }
        if (auto argument = dyn_cast<BlockArgument>(buffer.value)) {
          attrs.push_back(builder.getNamedAttr(
              "source_argument",
              builder.getI64IntegerAttr(argument.getArgNumber())));
        }
        createMaterializationRecord(builder, module.getLoc(),
                                    MaterializationBufferOp::getOperationName(),
                                    attrs);
      }
      for (auto [ordinal, task] : llvm::enumerate(tasks)) {
        const bool fusedFoldMap = task.fusedFold != nullptr;
        const bool fusedMap = !fusedFoldMap && task.fusedOperations.size() > 1;
        const bool foldRegion = !task.foldRegionSources.empty();
        const bool contractRegion = !task.contractRegionSources.empty();
        const bool composition = !task.compositionKind.empty() ||
                                 task.statefulContractFoldComposition;
        SourceRefAttr source = algebraSource(task.operation);
        SmallVector<NamedAttribute> attrs{
            builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
            builder.getNamedAttr("kind", MaterializationTaskKindAttr::get(
                                             module.getContext(), task.kind)),
            builder.getNamedAttr(
                "domain_shape",
                DenseI64ArrayAttr::get(module.getContext(), task.domain)),
            builder.getNamedAttr(
                "reduction_dimensions",
                DenseI64ArrayAttr::get(module.getContext(),
                                       task.reductionDimensions)),
            builder.getNamedAttr(
                "input_buffers",
                DenseI64ArrayAttr::get(module.getContext(), task.inputs)),
            builder.getNamedAttr(
                "output_buffers",
                DenseI64ArrayAttr::get(module.getContext(), task.outputs)),
            builder.getNamedAttr(
                "dependencies",
                DenseI64ArrayAttr::get(module.getContext(), task.dependencies)),
            builder.getNamedAttr(
                "semantic_fingerprint",
                builder.getStringAttr(
                    composition ? compositionTaskFingerprint(
                                      task.statefulContractFoldComposition
                                          ? "stateful_contract_fold"
                                          : StringRef(task.compositionKind),
                                      task.compositionSources,
                                      task.compositionRecomputedSources)
                    : contractRegion
                        ? contractRegionFingerprint(task.contractRegionSources)
                    : foldRegion ? foldRegionFingerprint(task.foldRegionSources)
                    : fusedFoldMap
                        ? fusedFoldMapFingerprint(cast<FoldOp>(task.fusedFold),
                                                  task.fusedOperations)
                    : fusedMap ? fusedMapFingerprint(task.fusedOperations)
                               : semanticTaskFingerprint(task.operation))),
            builder.getNamedAttr("source", source)};
        if (auto selection = dyn_cast<SelectionOp>(task.operation))
          attrs.push_back(
              builder.getNamedAttr("source_group", selection.getSourcesAttr()));
        if (!composition) {
          if (DictionaryAttr expansion =
                  algebraSourceExpansion(task.operation)) {
            attrs.push_back(
                builder.getNamedAttr("source_expansion", expansion));
          }
        }
        if (composition) {
          SmallVector<Attribute> sources;
          SmallVector<Attribute> sourceExpansions;
          for (Operation *operation : task.compositionSources)
            sources.push_back(algebraSource(operation));
          for (Operation *operation : task.compositionSources) {
            if (DictionaryAttr expansion = algebraSourceExpansion(operation))
              sourceExpansions.push_back(expansion);
          }
          if (!task.compositionKind.empty())
            attrs.push_back(builder.getNamedAttr(
                "composition_kind",
                builder.getStringAttr(task.compositionKind)));
          attrs.push_back(builder.getNamedAttr("composition_sources",
                                               builder.getArrayAttr(sources)));
          if (!task.compositionPhases.empty()) {
            SmallVector<Attribute> phases;
            for (Operation *operation : task.compositionPhases) {
              SmallVector<int64_t> domain;
              SmallVector<int64_t> reductions;
              MaterializationTaskKind kind;
              if (auto map = dyn_cast<MapOp>(operation)) {
                FailureOr<SmallVector<int64_t>> mapDomain =
                    genericMapDomainShape(map);
                if (failed(mapDomain))
                  return signalPassFailure();
                domain = std::move(*mapDomain);
                kind = MaterializationTaskKind::Map;
              } else if (auto contract = dyn_cast<ContractOp>(operation)) {
                FailureOr<SmallVector<int64_t>> contractDomain =
                    contractDomainShape(contract);
                if (failed(contractDomain))
                  return signalPassFailure();
                domain = std::move(*contractDomain);
                for (auto [dimension, iterator] :
                     llvm::enumerate(contract.getIteratorKinds()))
                  if (cast<StringAttr>(iterator).getValue() == "reduction")
                    reductions.push_back(dimension);
                kind = MaterializationTaskKind::Contract;
              } else {
                return signalPassFailure();
              }
              phases.push_back(builder.getDictionaryAttr(
                  {builder.getNamedAttr("source", algebraSource(operation)),
                   builder.getNamedAttr(
                       "kind", MaterializationTaskKindAttr::get(
                                   module.getContext(), kind)),
                   builder.getNamedAttr(
                       "domain_shape",
                       builder.getDenseI64ArrayAttr(domain)),
                   builder.getNamedAttr(
                       "reduction_dimensions",
                       builder.getDenseI64ArrayAttr(reductions))}));
            }
            llvm::SmallDenseMap<Operation *, int64_t> phaseOrdinals;
            for (auto [ordinal, operation] :
                 llvm::enumerate(task.compositionPhases))
              phaseOrdinals.try_emplace(operation, ordinal);
            SmallVector<Attribute> phaseEdges;
            for (auto [targetPhase, operation] :
                 llvm::enumerate(task.compositionPhases)) {
              ArrayAttr indexingMaps;
              if (auto map = dyn_cast<MapOp>(operation))
                indexingMaps = map.getIndexingMaps();
              else if (auto contract = dyn_cast<ContractOp>(operation))
                indexingMaps = contract.getIndexingMaps();
              else
                return signalPassFailure();
              for (auto [operandOrdinal, operand] :
                   llvm::enumerate(operation->getOperands())) {
                auto result = dyn_cast<OpResult>(operand);
                if (!result)
                  continue;
                auto sourcePhase = phaseOrdinals.find(result.getOwner());
                if (sourcePhase == phaseOrdinals.end())
                  continue;
                auto accessMap = dyn_cast<AffineMapAttr>(
                    indexingMaps.getValue()[operandOrdinal]);
                if (!accessMap)
                  return signalPassFailure();
                phaseEdges.push_back(builder.getDictionaryAttr(
                    {builder.getNamedAttr(
                         "source_phase",
                         builder.getI64IntegerAttr(sourcePhase->second)),
                     builder.getNamedAttr(
                         "source_result",
                         builder.getI64IntegerAttr(result.getResultNumber())),
                     builder.getNamedAttr(
                         "target_phase",
                         builder.getI64IntegerAttr(targetPhase)),
                     builder.getNamedAttr(
                         "target_operand",
                         builder.getI64IntegerAttr(operandOrdinal)),
                     builder.getNamedAttr("access_map", accessMap)}));
              }
            }
            SmallVector<Attribute> resultBindings;
            for (int64_t output : task.outputs) {
              if (output < 0 ||
                  output >= static_cast<int64_t>(derived.buffers.size()))
                return signalPassFailure();
              auto result = dyn_cast<OpResult>(derived.buffers[output].value);
              if (!result)
                return signalPassFailure();
              auto phase = llvm::find(task.compositionPhases,
                                      result.getOwner());
              if (phase == task.compositionPhases.end())
                return signalPassFailure();
              resultBindings.push_back(builder.getDictionaryAttr(
                  {builder.getNamedAttr(
                       "phase", builder.getI64IntegerAttr(std::distance(
                                    task.compositionPhases.begin(), phase))),
                   builder.getNamedAttr("source",
                                        algebraSource(result.getOwner())),
                   builder.getNamedAttr(
                       "result", builder.getI64IntegerAttr(
                                     result.getResultNumber())),
                   builder.getNamedAttr("buffer",
                                        builder.getI64IntegerAttr(output))}));
            }
            attrs.push_back(builder.getNamedAttr(
                "composition_phases", builder.getArrayAttr(phases)));
            attrs.push_back(builder.getNamedAttr(
                "composition_phase_edges", builder.getArrayAttr(phaseEdges)));
            attrs.push_back(builder.getNamedAttr(
                "composition_result_bindings",
                builder.getArrayAttr(resultBindings)));
          }
          if (!sourceExpansions.empty()) {
            if (sourceExpansions.size() != task.compositionSources.size())
              return signalPassFailure();
            attrs.push_back(
                builder.getNamedAttr("source_expansion_group",
                                     builder.getArrayAttr(sourceExpansions)));
          }
          auto addDimensions = [&](StringRef name,
                                   ArrayRef<int64_t> dimensions) {
            if (!dimensions.empty() || isa<ScanOp>(task.operation))
              attrs.push_back(builder.getNamedAttr(
                  name, builder.getDenseI64ArrayAttr(dimensions)));
          };
          addDimensions("composition_owner_dimensions",
                        task.compositionOwnerDimensions);
          addDimensions("composition_stream_dimensions",
                        task.compositionStreamDimensions);
          addDimensions("composition_state_inputs",
                        task.compositionStateInputs);
          addDimensions("composition_state_outputs",
                        task.compositionStateOutputs);
          addDimensions("composition_producer_contraction_dimensions",
                        task.compositionProducerContractionDimensions);
          addDimensions("composition_consumer_contraction_dimensions",
                        task.compositionConsumerContractionDimensions);
          if (task.compositionIndexDomainSource) {
            attrs.push_back(
                builder.getNamedAttr("composition_index_domain_source",
                                     task.compositionIndexDomainSource));
            attrs.push_back(
                builder.getNamedAttr("composition_index_domain_shape",
                                     builder.getDenseI64ArrayAttr(
                                         task.compositionIndexDomainShape)));
            attrs.push_back(builder.getNamedAttr(
                "composition_index_domain_map",
                AffineMapAttr::get(task.compositionIndexDomainMap)));
          }
          if (task.compositionOrderedIndexSource) {
            attrs.push_back(
                builder.getNamedAttr("composition_ordered_index_source",
                                     task.compositionOrderedIndexSource));
            attrs.push_back(builder.getNamedAttr(
                "composition_ordered_index_dimensions",
                builder.getDenseI64ArrayAttr(
                    task.compositionOrderedIndexDimensions)));
            attrs.push_back(builder.getNamedAttr(
                "composition_ordered_index_inclusive",
                builder.getBoolAttr(*task.compositionOrderedIndexInclusive)));
          }
          if (!task.compositionStateAccessBuffers.empty()) {
            SmallVector<Attribute> accessMaps;
            for (AffineMap map : task.compositionStateAccessMaps)
              accessMaps.push_back(AffineMapAttr::get(map));
            attrs.push_back(
                builder.getNamedAttr("composition_state_access_buffers",
                                     builder.getDenseI64ArrayAttr(
                                         task.compositionStateAccessBuffers)));
            attrs.push_back(
                builder.getNamedAttr("composition_state_access_maps",
                                     builder.getArrayAttr(accessMaps)));
          }
          if (!task.onlineVectorDimensions.empty())
            attrs.push_back(builder.getNamedAttr(
                "online_vector_dimensions",
                builder.getDenseI64ArrayAttr(task.onlineVectorDimensions)));
          if (task.onlineMma)
            attrs.push_back(
                builder.getNamedAttr("online_mma", builder.getBoolAttr(true)));
          if (!task.compositionRecomputedSources.empty()) {
            SmallVector<Attribute> recomputed;
            for (Operation *operation : task.compositionRecomputedSources)
              recomputed.push_back(algebraSource(operation));
            attrs.push_back(
                builder.getNamedAttr("composition_recomputed_sources",
                                     builder.getArrayAttr(recomputed)));
          }
          if (task.compositionKind == "map_contract" ||
              (task.statefulContractFoldComposition &&
               task.compositionCost.baselineLaunches > 0)) {
            auto cost = task.compositionCost;
            attrs.push_back(builder.getNamedAttr(
                "composition_cost",
                builder.getDictionaryAttr(
                    {builder.getNamedAttr(
                         "baseline_launches",
                         builder.getI64IntegerAttr(cost.baselineLaunches)),
                     builder.getNamedAttr(
                         "candidate_launches",
                         builder.getI64IntegerAttr(cost.candidateLaunches)),
                     builder.getNamedAttr("baseline_temporary_bytes",
                                          builder.getI64IntegerAttr(
                                              cost.baselineTemporaryBytes)),
                     builder.getNamedAttr("candidate_temporary_bytes",
                                          builder.getI64IntegerAttr(
                                              cost.candidateTemporaryBytes)),
                     builder.getNamedAttr(
                         "baseline_temporary_traffic_bytes",
                         builder.getI64IntegerAttr(
                             cost.baselineTemporaryTrafficBytes)),
                     builder.getNamedAttr(
                         "candidate_temporary_traffic_bytes",
                         builder.getI64IntegerAttr(
                             cost.candidateTemporaryTrafficBytes)),
                     builder.getNamedAttr("contract_evaluations",
                                          builder.getI64IntegerAttr(
                                              cost.contractEvaluations))})));
          }
        } else if (contractRegion) {
          SmallVector<Attribute> sources;
          for (Operation *operation : task.contractRegionSources) {
            if (auto map = dyn_cast<MapOp>(operation))
              sources.push_back(map.getSource());
            else if (auto fold = dyn_cast<FoldOp>(operation))
              sources.push_back(fold.getSource());
            else
              sources.push_back(cast<ContractOp>(operation).getSource());
          }
          SmallVector<Attribute> epilogues;
          for (Operation *operation : task.contractRegionEpilogueSources)
            epilogues.push_back(cast<MapOp>(operation).getSource());
          auto cost = task.contractRegionCost;
          attrs.push_back(builder.getNamedAttr("contract_region_sources",
                                               builder.getArrayAttr(sources)));
          attrs.push_back(builder.getNamedAttr(
              "contract_region_strategy",
              builder.getStringAttr(task.contractRegionStrategy)));
          attrs.push_back(builder.getNamedAttr(
              "contract_region_fold_source",
              cast<FoldOp>(task.contractRegionFold).getSource()));
          attrs.push_back(builder.getNamedAttr(
              "contract_region_contract_source",
              cast<ContractOp>(task.contractRegionContract).getSource()));
          attrs.push_back(builder.getNamedAttr(
              "contract_region_bf16_boundary_source",
              cast<MapOp>(task.contractRegionBf16Boundary).getSource()));
          attrs.push_back(
              builder.getNamedAttr("contract_region_epilogue_sources",
                                   builder.getArrayAttr(epilogues)));
          attrs.push_back(builder.getNamedAttr(
              "contract_region_normalized_operand",
              builder.getI64IntegerAttr(task.contractRegionNormalizedOperand)));
          attrs.push_back(builder.getNamedAttr(
              "contract_region_cost",
              builder.getDictionaryAttr(
                  {builder.getNamedAttr("rows",
                                        builder.getI64IntegerAttr(cost.rows)),
                   builder.getNamedAttr(
                       "reduction", builder.getI64IntegerAttr(cost.reduction)),
                   builder.getNamedAttr(
                       "columns", builder.getI64IntegerAttr(cost.columns)),
                   builder.getNamedAttr(
                       "output_tile",
                       builder.getI64IntegerAttr(cost.outputTile)),
                   builder.getNamedAttr(
                       "output_tiles",
                       builder.getI64IntegerAttr(cost.outputTiles)),
                   builder.getNamedAttr(
                       "duplicated_fold_elements",
                       builder.getI64IntegerAttr(cost.duplicatedFoldElements)),
                   builder.getNamedAttr("normalized_temporary_bytes",
                                        builder.getI64IntegerAttr(
                                            cost.normalizedTemporaryBytes)),
                   builder.getNamedAttr(
                       "contract_temporary_bytes",
                       builder.getI64IntegerAttr(cost.contractTemporaryBytes)),
                   builder.getNamedAttr(
                       "compact_stat_bytes",
                       builder.getI64IntegerAttr(cost.compactStatBytes))})));
          if (!task.contractRegionRecomputedSources.empty()) {
            SmallVector<Attribute> recomputed;
            for (Operation *operation : task.contractRegionRecomputedSources) {
              if (auto map = dyn_cast<MapOp>(operation))
                recomputed.push_back(map.getSource());
              else
                recomputed.push_back(cast<FoldOp>(operation).getSource());
            }
            attrs.push_back(
                builder.getNamedAttr("contract_region_recomputed_sources",
                                     builder.getArrayAttr(recomputed)));
          }
        } else if (foldRegion) {
          SmallVector<Attribute> sources;
          for (Operation *operation : task.foldRegionSources) {
            if (auto map = dyn_cast<MapOp>(operation))
              sources.push_back(map.getSource());
            else
              sources.push_back(cast<FoldOp>(operation).getSource());
          }
          SmallVector<Attribute> phases;
          for (auto [phaseOrdinal, operation] :
               llvm::enumerate(task.foldRegionPhases)) {
            auto fold = cast<FoldOp>(operation);
            auto domain =
                cast<RankedTensorType>(fold.getInputs().front().getType())
                    .getShape();
            phases.push_back(builder.getDictionaryAttr(
                {builder.getNamedAttr("source", fold.getSource()),
                 builder.getNamedAttr("domain_shape",
                                      builder.getDenseI64ArrayAttr(domain)),
                 builder.getNamedAttr("reduction_dimensions",
                                      builder.getDenseI64ArrayAttr(
                                          fold.getReductionDimensions())),
                 builder.getNamedAttr("order_free",
                                      builder.getBoolAttr(fold.getOrderFree())),
                 builder.getNamedAttr(
                     "transition",
                     builder.getStringAttr(phaseOrdinal == 0 ? "launch_boundary"
                                                             : "cta_local"))}));
          }
          SmallVector<Attribute> boundaryUses;
          for (PlannedTask::BoundaryUse &use : task.foldRegionBoundaryUses) {
            boundaryUses.push_back(builder.getDictionaryAttr(
                {builder.getNamedAttr("input_buffer", builder.getI64IntegerAttr(
                                                          use.inputBuffer)),
                 builder.getNamedAttr("source", use.source.getSource()),
                 builder.getNamedAttr("operand",
                                      builder.getI64IntegerAttr(use.operand)),
                 builder.getNamedAttr("indexing_map",
                                      AffineMapAttr::get(use.indexingMap)),
                 builder.getNamedAttr("phase",
                                      builder.getI64IntegerAttr(use.phase))}));
          }
          attrs.push_back(builder.getNamedAttr("fold_region_sources",
                                               builder.getArrayAttr(sources)));
          SmallVector<Attribute> outputs;
          for (Operation *operation : task.foldRegionOutputs) {
            if (auto map = dyn_cast<MapOp>(operation))
              outputs.push_back(map.getSource());
            else
              outputs.push_back(cast<FoldOp>(operation).getSource());
          }
          attrs.push_back(builder.getNamedAttr("fold_region_output_sources",
                                               builder.getArrayAttr(outputs)));
          attrs.push_back(builder.getNamedAttr("fold_region_phases",
                                               builder.getArrayAttr(phases)));
          attrs.push_back(builder.getNamedAttr(
              "fold_region_boundary_uses", builder.getArrayAttr(boundaryUses)));
          if (!task.foldRegionRecomputedSources.empty()) {
            SmallVector<Attribute> recomputed;
            for (Operation *operation : task.foldRegionRecomputedSources) {
              if (auto map = dyn_cast<MapOp>(operation))
                recomputed.push_back(map.getSource());
              else
                recomputed.push_back(cast<FoldOp>(operation).getSource());
            }
            attrs.push_back(
                builder.getNamedAttr("fold_region_recomputed_sources",
                                     builder.getArrayAttr(recomputed)));
          }
        } else if (fusedFoldMap) {
          SmallVector<Attribute> sources;
          for (Operation *operation : task.fusedOperations)
            sources.push_back(cast<MapOp>(operation).getSource());
          SmallVector<Attribute> maps;
          for (AffineMap map : task.composedInputMaps)
            maps.push_back(AffineMapAttr::get(map));
          attrs.push_back(builder.getNamedAttr("fold_epilogue_sources",
                                               builder.getArrayAttr(sources)));
          attrs.push_back(builder.getNamedAttr("epilogue_input_maps",
                                               builder.getArrayAttr(maps)));
        } else if (fusedMap) {
          SmallVector<Attribute> sources;
          for (Operation *operation : task.fusedOperations)
            sources.push_back(cast<MapOp>(operation).getSource());
          SmallVector<Attribute> maps;
          for (AffineMap map : task.composedInputMaps)
            maps.push_back(AffineMapAttr::get(map));
          attrs.push_back(builder.getNamedAttr("fused_sources",
                                               builder.getArrayAttr(sources)));
          attrs.push_back(builder.getNamedAttr("composed_input_maps",
                                               builder.getArrayAttr(maps)));
          if (task.fusedMapOutputs.size() > 1) {
            SmallVector<Attribute> outputs;
            for (Operation *operation : task.fusedMapOutputs)
              outputs.push_back(cast<MapOp>(operation).getSource());
            attrs.push_back(builder.getNamedAttr(
                "fused_output_sources", builder.getArrayAttr(outputs)));
          }
        }
        if (task.kind == MaterializationTaskKind::Fold) {
          attrs.push_back(builder.getNamedAttr(
              "order_free", builder.getBoolAttr(task.orderFree)));
        }
        createMaterializationRecord(builder, task.operation->getLoc(),
                                    MaterializationTaskOp::getOperationName(),
                                    attrs);
      }
      builder.create<MaterializationPlanYieldOp>(module.getLoc());
      plan.setFingerprint(materializationPlanFingerprint(plan));
      if (failed(plan.verifyRegions())) {
        return signalPassFailure();
      }
    }
  }
};

LogicalResult
verifyMaterializationPlanAgainstSource(ModuleOp module,
                                       MaterializationPlanOp plan,
                                       RegionOp *boundRegion = nullptr) {
  if (failed(plan.verifyRegions())) {
    return failure();
  }
  llvm::SmallDenseMap<Attribute, Operation *> sourceOperations;
  llvm::SmallDenseMap<Attribute, Operation *> expandedSourceOperations;
  llvm::SmallDenseSet<Attribute> duplicateSources;
  auto recordSourceOperation = [&](Attribute source, Operation *operation) {
    if (DictionaryAttr expansion = algebraSourceExpansion(operation)) {
      Attribute key = ArrayAttr::get(module.getContext(), {source, expansion});
      if (!expandedSourceOperations.try_emplace(key, operation).second)
        duplicateSources.insert(key);
      sourceOperations.try_emplace(source, operation);
      return;
    }
    if (!sourceOperations.try_emplace(source, operation).second)
      duplicateSources.insert(source);
  };
  module.walk([&](Operation *operation) -> WalkResult {
    if (auto scan = dyn_cast<ScanOp>(operation)) {
      for (Attribute source : scan.getSources())
        recordSourceOperation(source, operation);
      return WalkResult::skip();
    }
    if (auto selection = dyn_cast<SelectionOp>(operation)) {
      for (Attribute source : selection.getSources()) {
        recordSourceOperation(source, operation);
      }
    } else if (auto map = dyn_cast<MapOp>(operation)) {
      recordSourceOperation(map.getSource(), operation);
    } else if (auto relation = dyn_cast<RelationOp>(operation)) {
      recordSourceOperation(relation.getSource(), operation);
    } else if (auto fold = dyn_cast<FoldOp>(operation)) {
      recordSourceOperation(fold.getSource(), operation);
    } else if (auto contract = dyn_cast<ContractOp>(operation)) {
      recordSourceOperation(contract.getSource(), operation);
    }
    return WalkResult::advance();
  });
  SmallVector<MaterializationBufferOp> buffers;
  SmallVector<MaterializationTaskOp> tasks;
  for (Operation &operation : plan.getBody().front()) {
    if (auto buffer = dyn_cast<MaterializationBufferOp>(operation)) {
      buffers.push_back(buffer);
    } else if (auto task = dyn_cast<MaterializationTaskOp>(operation)) {
      tasks.push_back(task);
    }
  }
  llvm::SmallDenseSet<Operation *> bound;
  llvm::SmallDenseMap<Operation *, int64_t> sourceOccurrences;
  llvm::SmallDenseMap<Operation *, int64_t> regionOccurrences;
  llvm::SmallDenseMap<Operation *, int64_t> recomputationDeclarations;
  bool sawFoldRegion = false;
  bool sawContractRegion = false;
  auto taskSourceOperation = [&](MaterializationTaskOp task) -> Operation * {
    auto expansion = task->getAttrOfType<DictionaryAttr>("source_expansion");
    if (!expansion)
      return sourceOperations.lookup(task.getSource());
    Attribute key =
        ArrayAttr::get(module.getContext(), {task.getSource(), expansion});
    return expandedSourceOperations.lookup(key);
  };
  for (MaterializationTaskOp task : tasks) {
    auto fusedSources = task->getAttrOfType<ArrayAttr>("fused_sources");
    auto fusedOutputSources =
        task->getAttrOfType<ArrayAttr>("fused_output_sources");
    auto sourceGroup = task->getAttrOfType<ArrayAttr>("source_group");
    auto composedMaps = task->getAttrOfType<ArrayAttr>("composed_input_maps");
    auto epilogueSources =
        task->getAttrOfType<ArrayAttr>("fold_epilogue_sources");
    auto epilogueMaps = task->getAttrOfType<ArrayAttr>("epilogue_input_maps");
    auto foldRegionSources =
        task->getAttrOfType<ArrayAttr>("fold_region_sources");
    auto foldRegionOutputSources =
        task->getAttrOfType<ArrayAttr>("fold_region_output_sources");
    auto contractRegionSources =
        task->getAttrOfType<ArrayAttr>("contract_region_sources");
    auto compositionKind = task->getAttrOfType<StringAttr>("composition_kind");
    auto compositionSources =
        task->getAttrOfType<ArrayAttr>("composition_sources");
    auto compositionSourceExpansions =
        task->getAttrOfType<ArrayAttr>("source_expansion_group");
    auto compositionRecomputedSources =
        task->getAttrOfType<ArrayAttr>("composition_recomputed_sources");
    auto compositionPhases =
        task->getAttrOfType<ArrayAttr>("composition_phases");
    auto compositionPhaseEdges =
        task->getAttrOfType<ArrayAttr>("composition_phase_edges");
    auto compositionResultBindings =
        task->getAttrOfType<ArrayAttr>("composition_result_bindings");
    auto compositionCost =
        task->getAttrOfType<DictionaryAttr>("composition_cost");
    auto ownerDimensions =
        task->getAttrOfType<DenseI64ArrayAttr>("composition_owner_dimensions");
    auto streamDimensions =
        task->getAttrOfType<DenseI64ArrayAttr>("composition_stream_dimensions");
    auto stateInputs =
        task->getAttrOfType<DenseI64ArrayAttr>("composition_state_inputs");
    auto stateOutputs =
        task->getAttrOfType<DenseI64ArrayAttr>("composition_state_outputs");
    auto producerContractions = task->getAttrOfType<DenseI64ArrayAttr>(
        "composition_producer_contraction_dimensions");
    auto consumerContractions = task->getAttrOfType<DenseI64ArrayAttr>(
        "composition_consumer_contraction_dimensions");
    auto indexDomainSource =
        task->getAttrOfType<SourceRefAttr>("composition_index_domain_source");
    auto indexDomainShape = task->getAttrOfType<DenseI64ArrayAttr>(
        "composition_index_domain_shape");
    auto indexDomainMap =
        task->getAttrOfType<AffineMapAttr>("composition_index_domain_map");
    auto orderedIndexSource =
        task->getAttrOfType<SourceRefAttr>("composition_ordered_index_source");
    auto orderedIndexDimensions = task->getAttrOfType<DenseI64ArrayAttr>(
        "composition_ordered_index_dimensions");
    auto orderedIndexInclusive =
        task->getAttrOfType<BoolAttr>("composition_ordered_index_inclusive");
    auto stateAccessBuffers = task->getAttrOfType<DenseI64ArrayAttr>(
        "composition_state_access_buffers");
    auto stateAccessMaps =
        task->getAttrOfType<ArrayAttr>("composition_state_access_maps");
    const bool statefulContractFold =
        !compositionKind && compositionSources && ownerDimensions &&
        streamDimensions && producerContractions && consumerContractions;
    const bool hasStructuralComposition =
        ownerDimensions || streamDimensions || stateInputs || stateOutputs ||
        producerContractions || consumerContractions || indexDomainSource ||
        indexDomainShape || indexDomainMap || orderedIndexSource ||
        orderedIndexDimensions || orderedIndexInclusive || stateAccessBuffers ||
        stateAccessMaps || compositionPhases || compositionPhaseEdges ||
        compositionResultBindings;
    if (compositionKind || compositionSources || compositionSourceExpansions ||
        compositionRecomputedSources || compositionPhases ||
        compositionPhaseEdges || compositionResultBindings || compositionCost ||
        hasStructuralComposition) {
      if (!compositionSources || fusedSources || composedMaps ||
          epilogueSources || epilogueMaps || foldRegionSources ||
          foldRegionOutputSources || contractRegionSources)
        return task.emitOpError(
            "generic composition metadata must be complete and exclusive");
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
      const bool hasStateFlow = stateInputs || stateOutputs;
      if ((!contractFold && !mapContract && !indexedAccumulation &&
           !integerRelation && !relationGather && !scan &&
           !statefulContractFold) ||
          (hasStructuralComposition && !statefulContractFold &&
           !indexedAccumulation && !mapContract && !scan) ||
          ((contractFold || mapContract || indexedAccumulation ||
            integerRelation || relationGather || scan) &&
           compositionSources.empty()) ||
          (indexedAccumulation &&
           (!compositionSourceExpansions ||
            compositionSourceExpansions.size() != compositionSources.size())) ||
          (!indexedAccumulation && compositionSourceExpansions) ||
          (contractFold && task.getKind() != MaterializationTaskKind::Fold) ||
          (indexedAccumulation &&
           task.getKind() != MaterializationTaskKind::Fold) ||
          ((integerRelation || relationGather) &&
           task.getKind() != MaterializationTaskKind::Relation) ||
          (scan && task.getKind() != MaterializationTaskKind::Scan) ||
          (statefulContractFold &&
           task.getKind() != MaterializationTaskKind::Contract) ||
          (contractFold && compositionCost) ||
          (mapContract && !compositionCost) ||
          (statefulContractFold && !task.getOnlineVectorDimensionsAttr()) ||
          (statefulContractFold && !hasStateFlow && !compositionCost) ||
          (statefulContractFold && hasStateFlow && compositionCost))
        return task.emitOpError(
            "generic composition kind must bind its task family and cost");
      SmallVector<Operation *> operations;
      for (auto [ordinal, source] : llvm::enumerate(compositionSources)) {
        Attribute key = source;
        if (compositionSourceExpansions)
          key = ArrayAttr::get(module.getContext(),
                               {source, compositionSourceExpansions[ordinal]});
        auto &operationMap = compositionSourceExpansions
                                 ? expandedSourceOperations
                                 : sourceOperations;
        auto found = operationMap.find(key);
        if (!isa<SourceRefAttr>(source) || found == operationMap.end() ||
            duplicateSources.contains(key) ||
            (!statefulContractFold && !bound.insert(found->second).second))
          return task.emitOpError(
              "composition sources must uniquely partition algebra");
        if (statefulContractFold)
          bound.insert(found->second);
        operations.push_back(found->second);
        ++sourceOccurrences[found->second];
      }
      SmallVector<Operation *> recomputedOperations;
      llvm::SmallDenseSet<Operation *> recomputedSet;
      if (compositionRecomputedSources) {
        if (!statefulContractFold && !indexedAccumulation && !mapContract)
          return task.emitOpError(
              "only online, Map/Contract, or indexed accumulation "
              "composition may declare recomputation");
        for (Attribute source : compositionRecomputedSources) {
          auto found = sourceOperations.find(source);
          if (!isa<SourceRefAttr>(source) || found == sourceOperations.end() ||
              duplicateSources.contains(source) ||
              !bound.contains(found->second) ||
              !recomputedSet.insert(found->second).second)
            return task.emitOpError(
                "recomputation must bind unique earlier composition sources");
          recomputedOperations.push_back(found->second);
        }
      }
      Operation *terminal = operations.empty() ? nullptr : operations.back();
      const bool validContractFold =
          contractFold && isa_and_nonnull<FoldOp>(terminal) &&
          sourceOperations.lookup(task.getSource()) == terminal &&
          llvm::all_of(ArrayRef<Operation *>(operations).drop_back(),
                       [](Operation *operation) {
                         return isa<MapOp, RelationOp, ContractOp>(operation);
                       });
      SmallVector<ContractOp> mapContractRoots;
      for (Operation *operation : operations) {
        auto candidate = dyn_cast<ContractOp>(operation);
        if (candidate)
          mapContractRoots.push_back(candidate);
      }
      bool validCompositionPhases =
          !compositionPhases && !compositionPhaseEdges &&
          !compositionResultBindings;
      if (compositionPhases || compositionPhaseEdges ||
          compositionResultBindings) {
        validCompositionPhases =
            compositionPhases && compositionPhaseEdges &&
            compositionResultBindings &&
            compositionPhases.size() == operations.size() &&
            compositionResultBindings.size() ==
                task.getOutputBuffers().size();
        for (auto [phaseOrdinal, operation] : llvm::enumerate(operations)) {
          if (!validCompositionPhases)
            break;
          auto phase = dyn_cast<DictionaryAttr>(compositionPhases[phaseOrdinal]);
          auto source = phase ? phase.getAs<SourceRefAttr>("source")
                              : SourceRefAttr{};
          auto kind = phase ? phase.getAs<MaterializationTaskKindAttr>("kind")
                            : MaterializationTaskKindAttr{};
          auto domain = phase
                            ? phase.getAs<DenseI64ArrayAttr>("domain_shape")
                            : DenseI64ArrayAttr{};
          auto reductions =
              phase ? phase.getAs<DenseI64ArrayAttr>("reduction_dimensions")
                    : DenseI64ArrayAttr{};
          SmallVector<int64_t> expectedDomain;
          SmallVector<int64_t> expectedReductions;
          MaterializationTaskKind expectedKind;
          if (auto map = dyn_cast<MapOp>(operation)) {
            FailureOr<SmallVector<int64_t>> mapDomain =
                genericMapDomainShape(map);
            if (failed(mapDomain)) {
              validCompositionPhases = false;
              break;
            }
            expectedDomain = std::move(*mapDomain);
            expectedKind = MaterializationTaskKind::Map;
          } else if (auto contract = dyn_cast<ContractOp>(operation)) {
            FailureOr<SmallVector<int64_t>> contractDomain =
                contractDomainShape(contract);
            if (failed(contractDomain)) {
              validCompositionPhases = false;
              break;
            }
            expectedDomain = std::move(*contractDomain);
            for (auto [dimension, iterator] :
                 llvm::enumerate(contract.getIteratorKinds()))
              if (cast<StringAttr>(iterator).getValue() == "reduction")
                expectedReductions.push_back(dimension);
            expectedKind = MaterializationTaskKind::Contract;
          } else {
            validCompositionPhases = false;
            break;
          }
          validCompositionPhases =
              source == algebraSource(operation) && kind && domain &&
              reductions && kind.getValue() == expectedKind &&
              domain.asArrayRef() == ArrayRef<int64_t>(expectedDomain) &&
              reductions.asArrayRef() ==
                  ArrayRef<int64_t>(expectedReductions);
        }
        SmallVector<std::tuple<int64_t, int64_t, int64_t, int64_t,
                               AffineMap>> expectedEdges;
        llvm::SmallDenseMap<Operation *, int64_t> phaseOrdinals;
        for (auto [ordinal, operation] : llvm::enumerate(operations))
          phaseOrdinals.try_emplace(operation, ordinal);
        for (auto [targetPhase, operation] : llvm::enumerate(operations)) {
          ArrayAttr indexingMaps;
          if (auto map = dyn_cast<MapOp>(operation))
            indexingMaps = map.getIndexingMaps();
          else if (auto contract = dyn_cast<ContractOp>(operation))
            indexingMaps = contract.getIndexingMaps();
          else {
            validCompositionPhases = false;
            break;
          }
          for (auto [targetOperand, operand] :
               llvm::enumerate(operation->getOperands())) {
            auto result = dyn_cast<OpResult>(operand);
            if (!result)
              continue;
            auto sourcePhase = phaseOrdinals.find(result.getOwner());
            if (sourcePhase == phaseOrdinals.end())
              continue;
            auto accessMap = dyn_cast<AffineMapAttr>(
                indexingMaps.getValue()[targetOperand]);
            if (!accessMap) {
              validCompositionPhases = false;
              break;
            }
            expectedEdges.emplace_back(
                sourcePhase->second, result.getResultNumber(), targetPhase,
                targetOperand, accessMap.getValue());
          }
        }
        validCompositionPhases &=
            compositionPhaseEdges.size() == expectedEdges.size();
        for (auto [edgeAttribute, expectedEdge] :
             llvm::zip(compositionPhaseEdges, expectedEdges)) {
          if (!validCompositionPhases)
            break;
          auto edge = dyn_cast<DictionaryAttr>(edgeAttribute);
          auto sourcePhase = edge ? edge.getAs<IntegerAttr>("source_phase")
                                  : IntegerAttr{};
          auto sourceResult = edge ? edge.getAs<IntegerAttr>("source_result")
                                   : IntegerAttr{};
          auto targetPhase = edge ? edge.getAs<IntegerAttr>("target_phase")
                                  : IntegerAttr{};
          auto targetOperand = edge ? edge.getAs<IntegerAttr>("target_operand")
                                    : IntegerAttr{};
          auto accessMap = edge ? edge.getAs<AffineMapAttr>("access_map")
                                : AffineMapAttr{};
          auto [expectedSourcePhase, expectedSourceResult,
                expectedTargetPhase, expectedTargetOperand,
                expectedAccessMap] = expectedEdge;
          validCompositionPhases =
              sourcePhase && sourceResult && targetPhase && targetOperand &&
              accessMap && sourcePhase.getInt() == expectedSourcePhase &&
              sourceResult.getInt() == expectedSourceResult &&
              targetPhase.getInt() == expectedTargetPhase &&
              targetOperand.getInt() == expectedTargetOperand &&
              accessMap.getValue() == expectedAccessMap;
        }
        for (auto [bindingOrdinal, outputBuffer] :
             llvm::enumerate(task.getOutputBuffers())) {
          if (!validCompositionPhases)
            break;
          auto binding = dyn_cast<DictionaryAttr>(
              compositionResultBindings[bindingOrdinal]);
          auto phase = binding ? binding.getAs<IntegerAttr>("phase")
                               : IntegerAttr{};
          auto source = binding ? binding.getAs<SourceRefAttr>("source")
                                : SourceRefAttr{};
          auto result = binding ? binding.getAs<IntegerAttr>("result")
                                : IntegerAttr{};
          auto buffer = binding ? binding.getAs<IntegerAttr>("buffer")
                                : IntegerAttr{};
          if (!phase || !source || !result || !buffer || phase.getInt() < 0 ||
              phase.getInt() >= static_cast<int64_t>(operations.size()) ||
              buffer.getInt() != outputBuffer || outputBuffer < 0 ||
              outputBuffer >= static_cast<int64_t>(buffers.size())) {
            validCompositionPhases = false;
            break;
          }
          Operation *producer = operations[phase.getInt()];
          validCompositionPhases =
              source == algebraSource(producer) && result.getInt() >= 0 &&
              result.getInt() < static_cast<int64_t>(producer->getNumResults()) &&
              producer->getResult(result.getInt()).getType() ==
                  buffers[outputBuffer].getTensorType();
        }
      }
      const bool validMapContract =
          mapContract && isa_and_nonnull<MapOp, ContractOp>(terminal) &&
          !mapContractRoots.empty() &&
          validCompositionPhases &&
          ((!compositionPhases && task.getOutputBuffers().size() == 1) ||
           (compositionPhases && !task.getOutputBuffers().empty())) &&
          !fusedOutputSources &&
          terminal->getNumResults() == 1 &&
          llvm::all_of(mapContractRoots, [&](ContractOp root) {
            return valueDependsOnOperation(terminal->getResult(0), root);
          }) &&
          sourceOperations.lookup(task.getSource()) == terminal &&
          task.getKind() == (mapContractRoots.size() == 1
                                 ? MaterializationTaskKind::Contract
                                 : MaterializationTaskKind::Map) &&
          llvm::all_of(operations, [](Operation *operation) {
            return isa<MapOp, RelationOp, ContractOp>(operation);
          }) &&
          llvm::all_of(recomputedOperations, [&](Operation *operation) {
            auto map = dyn_cast<MapOp>(operation);
            if (!map || map.getNumResults() != 1)
              return false;
            return llvm::all_of(map.getInputs(), [&](Value input) {
              Operation *dependency = input.getDefiningOp();
              return !dependency || !bound.contains(dependency) ||
                     recomputedSet.contains(dependency) ||
                     isa<FoldOp>(dependency);
            });
          });
      const bool validIntegerRelation =
          integerRelation && operations.size() >= 2 &&
          isa_and_nonnull<RelationOp>(terminal) &&
          sourceOperations.lookup(task.getSource()) == terminal &&
          llvm::all_of(operations, [](Operation *operation) {
            return isa<RelationOp>(operation);
          });
      const bool validRelationGather =
          relationGather && operations.size() >= 2 &&
          isa_and_nonnull<RelationOp>(terminal) &&
          cast<RelationOp>(terminal).getSemantics() ==
              RelationSemantics::Gather &&
          sourceOperations.lookup(task.getSource()) == terminal &&
          llvm::all_of(operations, [](Operation *operation) {
            return isa<RelationOp>(operation);
          });
      Operation *principal = taskSourceOperation(task);
      if (indexedAccumulation && operations.size() == 4)
        principal = operations[2];
      bool validIndexedRecomputation = true;
      if (indexedAccumulation && !recomputedOperations.empty()) {
        auto contributions = dyn_cast<RelationOp>(operations.front());
        validIndexedRecomputation =
            contributions && contributions.getInputs().size() == 2 &&
            recomputedOperations.size() <= 8 &&
            llvm::all_of(recomputedOperations, [](Operation *operation) {
              auto relation = dyn_cast<RelationOp>(operation);
              return relation && relation.getNumResults() == 1 &&
                     (relation.getSemantics() ==
                          RelationSemantics::BroadcastInDim ||
                      relation.getSemantics() == RelationSemantics::Add ||
                      relation.getSemantics() == RelationSemantics::Compare ||
                      relation.getSemantics() == RelationSemantics::Select);
            }) &&
            recomputedOperations.back()->getResult(0) ==
                contributions.getInputs()[1];
      }
      if (((contractFold || mapContract) && !validContractFold &&
           !validMapContract) ||
          (integerRelation && !validIntegerRelation) ||
          (relationGather && !validRelationGather) ||
          (scan &&
           (operations.size() != 1 || !isa<ScanOp>(operations.front()) ||
            sourceOperations.lookup(task.getSource()) != operations.front())) ||
          (indexedAccumulation &&
           (!indexedAccumulationExpansion(operations, 0) ||
            !validIndexedRecomputation ||
            principal != operations[2] ||
            ownerDimensions.size() + streamDimensions.size() !=
                task.getDomainShape().size() ||
            streamDimensions.asArrayRef() != task.getReductionDimensions())) ||
          (statefulContractFold &&
           !llvm::is_contained(operations,
                               sourceOperations.lookup(task.getSource()))) ||
          task.getSemanticFingerprint() !=
              compositionTaskFingerprint(statefulContractFold
                                             ? "stateful_contract_fold"
                                             : compositionKind.getValue(),
                                         operations, recomputedOperations))
        return task.emitOpError(
            "composition sources must bind their source-ordered algebra");
      const bool hasIndexDomain =
          indexDomainSource || indexDomainShape || indexDomainMap;
      if (hasIndexDomain &&
          (!indexDomainSource || !indexDomainShape || !indexDomainMap ||
           !statefulContractFold ||
           !llvm::is_contained(compositionSources.getValue(),
                               Attribute(indexDomainSource)) ||
           indexDomainShape.empty() ||
           llvm::any_of(indexDomainShape.asArrayRef(),
                        [](int64_t extent) { return extent <= 0; }) ||
           indexDomainMap.getValue().getNumSymbols() != 0 ||
           indexDomainMap.getValue().getNumDims() !=
               task.getDomainShape().size() ||
           indexDomainMap.getValue().getNumResults() !=
               indexDomainShape.size()))
        return task.emitOpError(
            "index-domain provenance must bind one composition source and "
            "map the complete task domain");
      const bool hasOrderedIndex =
          orderedIndexSource || orderedIndexDimensions || orderedIndexInclusive;
      if (hasOrderedIndex &&
          (!hasIndexDomain || !orderedIndexSource || !orderedIndexDimensions ||
           !orderedIndexInclusive ||
           !llvm::is_contained(compositionSources.getValue(),
                               Attribute(orderedIndexSource)) ||
           orderedIndexDimensions.size() != 2 ||
           llvm::any_of(orderedIndexDimensions.asArrayRef(), [&](int64_t axis) {
             return axis < 0 || axis >= indexDomainShape.size();
           })))
        return task.emitOpError(
            "ordered index provenance must bind one comparison source and "
            "two index-domain dimensions");
      if (stateAccessBuffers || stateAccessMaps) {
        if (!stateAccessBuffers || !stateAccessMaps || !hasIndexDomain ||
            stateAccessBuffers.size() != stateAccessMaps.size() ||
            stateAccessBuffers.empty())
          return task.emitOpError(
              "state-buffer accesses require paired buffers and affine maps");
        llvm::SmallDenseSet<int64_t> uniqueAccesses;
        for (auto [buffer, mapAttribute] : llvm::zip_equal(
                 stateAccessBuffers.asArrayRef(), stateAccessMaps)) {
          auto map = dyn_cast<AffineMapAttr>(mapAttribute);
          const bool declaredState =
              (stateInputs &&
               llvm::is_contained(stateInputs.asArrayRef(), buffer)) ||
              (stateOutputs &&
               llvm::is_contained(stateOutputs.asArrayRef(), buffer));
          auto bufferType =
              buffer >= 0 && buffer < static_cast<int64_t>(buffers.size())
                  ? dyn_cast<RankedTensorType>(buffers[buffer].getTensorType())
                  : RankedTensorType{};
          if (!uniqueAccesses.insert(buffer).second || !declaredState || !map ||
              map.getValue().getNumSymbols() != 0 ||
              map.getValue().getNumDims() != task.getDomainShape().size() ||
              !bufferType ||
              map.getValue().getNumResults() != bufferType.getRank())
            return task.emitOpError(
                "state-buffer access must uniquely map one declared state "
                "buffer from the task domain");
        }
      }
      if (mapContract || (statefulContractFold && !hasStateFlow &&
                          task.getOnlineVectorDimensionsAttr())) {
        constexpr StringLiteral fields[] = {"baseline_launches",
                                            "candidate_launches",
                                            "baseline_temporary_bytes",
                                            "candidate_temporary_bytes",
                                            "baseline_temporary_traffic_bytes",
                                            "candidate_temporary_traffic_bytes",
                                            "contract_evaluations"};
        if (compositionCost.size() != std::size(fields))
          return task.emitOpError(
              "composition cost requires the closed relative fields");
        SmallVector<int64_t> values;
        for (StringRef field : fields) {
          auto value = compositionCost.getAs<IntegerAttr>(field);
          if (!value || value.getInt() < 0)
            return task.emitOpError(
                "composition cost fields must be nonnegative i64");
          values.push_back(value.getInt());
        }
        if (values[1] > values[0] || values[3] > values[2] ||
            values[5] > values[4] ||
            !(values[1] < values[0] || values[3] < values[2] ||
              values[5] < values[4]) ||
            (statefulContractFold && values[6] <= 0))
          return task.emitOpError(
              "composition cost must dominate the materialized baseline");
      }
      continue;
    }
    if (contractRegionSources) {
      sawContractRegion = true;
      if (fusedSources || composedMaps || epilogueSources || epilogueMaps ||
          foldRegionSources || foldRegionOutputSources ||
          task.getKind() != MaterializationTaskKind::Contract) {
        task.emitOpError(
            "cannot mix Contract-region and legacy fusion metadata");
        return failure();
      }
      SmallVector<Operation *> operations;
      for (Attribute source : contractRegionSources) {
        auto found = sourceOperations.find(source);
        if (!isa<SourceRefAttr>(source) || found == sourceOperations.end() ||
            duplicateSources.contains(source) ||
            !isa<MapOp, FoldOp, ContractOp>(found->second)) {
          task.emitOpError("Contract-region sources must bind unique algebra");
          return failure();
        }
        operations.push_back(found->second);
        bound.insert(found->second);
        ++sourceOccurrences[found->second];
        ++regionOccurrences[found->second];
      }
      if (task.getSemanticFingerprint() !=
          contractRegionFingerprint(operations)) {
        task.emitOpError("does not fingerprint every Contract-region source");
        return failure();
      }
      auto foldSource =
          task->getAttrOfType<SourceRefAttr>("contract_region_fold_source");
      auto contractSource =
          task->getAttrOfType<SourceRefAttr>("contract_region_contract_source");
      auto boundarySource = task->getAttrOfType<SourceRefAttr>(
          "contract_region_bf16_boundary_source");
      auto epilogues =
          task->getAttrOfType<ArrayAttr>("contract_region_epilogue_sources");
      if (!foldSource ||
          !isa_and_nonnull<FoldOp>(sourceOperations.lookup(foldSource)) ||
          !contractSource ||
          !isa_and_nonnull<ContractOp>(
              sourceOperations.lookup(contractSource)) ||
          sourceOperations.lookup(contractSource) !=
              sourceOperations.lookup(task.getSource()) ||
          !boundarySource ||
          !isa_and_nonnull<MapOp>(sourceOperations.lookup(boundarySource)) ||
          !epilogues || llvm::any_of(epilogues, [&](Attribute source) {
            return !isa_and_nonnull<MapOp>(sourceOperations.lookup(source));
          })) {
        task.emitOpError("Contract-region roles must bind exact algebra kinds");
        return failure();
      }
      if (auto recomputed = task->getAttrOfType<ArrayAttr>(
              "contract_region_recomputed_sources")) {
        for (Attribute source : recomputed) {
          Operation *operation = sourceOperations.lookup(source);
          if (!operation || !isa<MapOp, FoldOp>(operation) ||
              !llvm::is_contained(operations, operation)) {
            task.emitOpError(
                "Contract-region recomputation must bind task-local algebra");
            return failure();
          }
          ++recomputationDeclarations[operation];
        }
      }
      continue;
    }
    if (foldRegionSources) {
      sawFoldRegion = true;
      if (fusedSources || composedMaps || epilogueSources || epilogueMaps ||
          task.getKind() != MaterializationTaskKind::Fold) {
        task.emitOpError("cannot mix Fold-region and legacy fusion metadata");
        return failure();
      }
      SmallVector<Operation *> operations;
      for (Attribute source : foldRegionSources) {
        auto found = sourceOperations.find(source);
        if (!isa<SourceRefAttr>(source) || found == sourceOperations.end() ||
            duplicateSources.contains(source) ||
            !isa<MapOp, FoldOp>(found->second)) {
          task.emitOpError("Fold-region sources must bind unique algebra");
          return failure();
        }
        operations.push_back(found->second);
        bound.insert(found->second);
        ++sourceOccurrences[found->second];
        ++regionOccurrences[found->second];
      }
      if (task.getSemanticFingerprint() != foldRegionFingerprint(operations)) {
        task.emitOpError("does not fingerprint every Fold-region source");
        return failure();
      }
      auto outputSources =
          task->getAttrOfType<ArrayAttr>("fold_region_output_sources");
      if (!outputSources ||
          outputSources.size() != task.getOutputBuffers().size()) {
        task.emitOpError("Fold-region outputs must bind ordered sources");
        return failure();
      }
      for (auto [source, bufferOrdinal] :
           llvm::zip_equal(outputSources, task.getOutputBuffers())) {
        Operation *operation = sourceOperations.lookup(source);
        if (!operation || !llvm::is_contained(operations, operation) ||
            operation->getNumResults() != 1 || bufferOrdinal < 0 ||
            bufferOrdinal >= static_cast<int64_t>(buffers.size()) ||
            buffers[bufferOrdinal].getTensorType() !=
                operation->getResult(0).getType()) {
          task.emitOpError(
              "Fold-region output buffers must match their source results");
          return failure();
        }
      }
      if (auto recomputed = task->getAttrOfType<ArrayAttr>(
              "fold_region_recomputed_sources")) {
        for (Attribute source : recomputed) {
          Operation *operation = sourceOperations.lookup(source);
          if (!operation || !isa<MapOp, FoldOp>(operation) ||
              !llvm::is_contained(operations, operation)) {
            task.emitOpError(
                "Fold-region recomputation must bind task-local algebra");
            return failure();
          }
          ++recomputationDeclarations[operation];
        }
      }
      continue;
    }
    if (foldRegionOutputSources) {
      task.emitOpError("Fold-region outputs require Fold-region sources");
      return failure();
    }
    if (sawFoldRegion && !sawContractRegion) {
      task.emitOpError("cannot mix Fold-region and legacy tasks");
      return failure();
    }
    if (fusedSources && epilogueSources) {
      task.emitOpError("cannot mix Map fusion and Fold epilogue provenance");
      return failure();
    }
    if (fusedOutputSources && !fusedSources) {
      task.emitOpError("fused Map outputs require fused Map provenance");
      return failure();
    }
    Operation *source = taskSourceOperation(task);
    if (!source) {
      task.emitOpError("source must uniquely bind one surviving algebra task");
      return failure();
    }
    auto selection = dyn_cast<SelectionOp>(source);
    if (selection) {
      if (!sourceGroup || sourceGroup != selection.getSourcesAttr() ||
          sourceGroup.empty() ||
          sourceGroup.getValue().front() != task.getSource()) {
        task.emitOpError(
            "Selection source group must bind every ordered source");
        return failure();
      }
    } else if (sourceGroup) {
      task.emitOpError("source group is only valid for Selection tasks");
      return failure();
    }
    SmallVector<Operation *> fusedOperations;
    if (fusedSources) {
      if (task.getKind() != MaterializationTaskKind::Map ||
          fusedSources.size() < 2 ||
          fusedSources.getValue().back() != task.getSource() || !composedMaps ||
          composedMaps.size() != task.getInputBuffers().size()) {
        task.emitOpError(
            "fused Map provenance and composed inputs must be aligned");
        return failure();
      }
      for (Attribute fusedSource : fusedSources) {
        auto found = sourceOperations.find(fusedSource);
        if (!isa<SourceRefAttr>(fusedSource) ||
            found == sourceOperations.end() ||
            duplicateSources.contains(fusedSource) ||
            !isa<MapOp>(found->second) || !bound.insert(found->second).second) {
          task.emitOpError(
              "fused_sources must uniquely bind ordered surviving Maps");
          return failure();
        }
        fusedOperations.push_back(found->second);
      }
      if (task.getSemanticFingerprint() !=
          fusedMapFingerprint(fusedOperations)) {
        task.emitOpError("does not fingerprint every fused Map semantic");
        return failure();
      }
      for (Attribute map : composedMaps) {
        if (!isa<AffineMapAttr>(map)) {
          task.emitOpError("composed_input_maps must contain affine maps");
          return failure();
        }
      }
      if (fusedOutputSources) {
        if (fusedOutputSources.size() != task.getOutputBuffers().size() ||
            fusedOutputSources.size() < 2)
          return task.emitOpError(
              "fused Map outputs must bind every physical output");
        for (auto [outputSource, outputBuffer] :
             llvm::zip_equal(fusedOutputSources, task.getOutputBuffers())) {
          Operation *output = sourceOperations.lookup(outputSource);
          if (!output || !llvm::is_contained(fusedOperations, output) ||
              output->getNumResults() != 1 || outputBuffer < 0 ||
              outputBuffer >= buffers.size() ||
              buffers[outputBuffer].getTensorType() !=
                  output->getResult(0).getType())
            return task.emitOpError(
                "fused Map output buffers must match their source results");
        }
      } else if (task.getOutputBuffers().size() != 1 ||
                 task.getOutputBuffers().front() < 0 ||
                 task.getOutputBuffers().front() >= buffers.size() ||
                 buffers[task.getOutputBuffers().front()].getTensorType() !=
                     fusedOperations.back()->getResult(0).getType()) {
        return task.emitOpError(
            "fused Map output must match its terminal source result");
      }
    } else if (epilogueSources) {
      auto fold = dyn_cast<FoldOp>(source);
      if (!fold || task.getKind() != MaterializationTaskKind::Fold ||
          epilogueSources.empty() || !epilogueMaps ||
          task.getInputBuffers().size() < fold->getNumOperands() ||
          epilogueMaps.size() !=
              task.getInputBuffers().size() - fold->getNumOperands() ||
          !bound.insert(source).second) {
        task.emitOpError(
            "Fold epilogue provenance and composed inputs must be aligned");
        return failure();
      }
      for (Attribute epilogueSource : epilogueSources) {
        auto epilogue = sourceOperations.find(epilogueSource);
        if (!isa<SourceRefAttr>(epilogueSource) ||
            epilogue == sourceOperations.end() ||
            duplicateSources.contains(epilogueSource) ||
            !isa<MapOp>(epilogue->second) ||
            !bound.insert(epilogue->second).second) {
          task.emitOpError(
              "fold_epilogue_sources must uniquely bind ordered Maps");
          return failure();
        }
        fusedOperations.push_back(epilogue->second);
      }
      if (task.getSemanticFingerprint() !=
          fusedFoldMapFingerprint(fold, fusedOperations)) {
        task.emitOpError("does not fingerprint the Fold and every epilogue");
        return failure();
      }
      for (Attribute map : epilogueMaps) {
        if (!isa<AffineMapAttr>(map)) {
          task.emitOpError("epilogue_input_maps must contain affine maps");
          return failure();
        }
      }
    }
    if (!fusedSources && !epilogueSources && !bound.insert(source).second) {
      task.emitOpError("source must uniquely bind one surviving algebra task");
      return failure();
    }
    const bool kindMatches =
        (task.getKind() == MaterializationTaskKind::Map &&
         isa<MapOp>(source)) ||
        (task.getKind() == MaterializationTaskKind::Relation &&
         isa<RelationOp>(source)) ||
        (task.getKind() == MaterializationTaskKind::Selection && selection) ||
        (task.getKind() == MaterializationTaskKind::Fold &&
         isa<FoldOp>(source)) ||
        (task.getKind() == MaterializationTaskKind::Contract &&
         isa<ContractOp>(source));
    const std::string expectedFingerprint =
        fusedSources ? fusedMapFingerprint(fusedOperations)
        : epilogueSources
            ? fusedFoldMapFingerprint(cast<FoldOp>(source), fusedOperations)
            : semanticTaskFingerprint(source);
    const bool arityMatches =
        fusedSources || epilogueSources ||
        (task.getInputBuffers().size() == source->getNumOperands() &&
         task.getOutputBuffers().size() == source->getNumResults());
    if (!kindMatches || task.getSemanticFingerprint() != expectedFingerprint ||
        !arityMatches) {
      task.emitOpError("does not match its bound algebra semantics");
      return failure();
    }
    if (!fusedSources && !epilogueSources) {
      for (auto [bufferOrdinal, value] :
           llvm::zip_equal(task.getInputBuffers(), source->getOperands())) {
        if (buffers[bufferOrdinal].getTensorType() != value.getType()) {
          task.emitOpError("input buffer type does not match bound algebra");
          return failure();
        }
      }
      for (auto [bufferOrdinal, value] :
           llvm::zip_equal(task.getOutputBuffers(), source->getResults())) {
        if (buffers[bufferOrdinal].getTensorType() != value.getType()) {
          task.emitOpError("output buffer type does not match bound algebra");
          return failure();
        }
      }
    } else if (epilogueSources) {
      auto fold = cast<FoldOp>(source);
      for (auto [bufferOrdinal, value] : llvm::zip_equal(
               task.getInputBuffers().take_front(fold->getNumOperands()),
               source->getOperands())) {
        if (bufferOrdinal < 0 || bufferOrdinal >= buffers.size() ||
            buffers[bufferOrdinal].getTensorType() != value.getType()) {
          task.emitOpError("Fold input buffer type does not match algebra");
          return failure();
        }
      }
      if (task.getOutputBuffers().size() != 1 ||
          task.getOutputBuffers().front() < 0 ||
          task.getOutputBuffers().front() >= buffers.size() ||
          buffers[task.getOutputBuffers().front()].getTensorType() !=
              fusedOperations.back()->getResult(0).getType()) {
        task.emitOpError("Fold epilogue output does not match final Map");
        return failure();
      }
    }
    SmallVector<int64_t> expectedDomain;
    if (auto map = dyn_cast<MapOp>(source)) {
      FailureOr<SmallVector<int64_t>> domain = genericMapDomainShape(map);
      if (failed(domain)) {
        task.emitOpError("bound Map has no closed static domain");
        return failure();
      }
      expectedDomain = std::move(*domain);
    } else if (auto relation = dyn_cast<RelationOp>(source)) {
      auto result =
          relation.getNumResults() == 1
              ? dyn_cast<RankedTensorType>(relation.getResult(0).getType())
              : RankedTensorType{};
      if (!result || !result.hasStaticShape()) {
        task.emitOpError("bound Relation has no closed static domain");
        return failure();
      }
      expectedDomain.assign(result.getShape().begin(), result.getShape().end());
    } else if (selection) {
      auto input = dyn_cast<RankedTensorType>(selection.getInput().getType());
      if (!input || !input.hasStaticShape()) {
        task.emitOpError("bound Selection has no closed static domain");
        return failure();
      }
      expectedDomain.assign(input.getShape().begin(), input.getShape().end());
    } else if (auto contract = dyn_cast<ContractOp>(source)) {
      FailureOr<SmallVector<int64_t>> domain = contractDomainShape(contract);
      if (failed(domain)) {
        task.emitOpError("bound Contract has no closed static domain");
        return failure();
      }
      expectedDomain = std::move(*domain);
    } else {
      auto fold = cast<FoldOp>(source);
      expectedDomain.assign(
          cast<RankedTensorType>(fold.getInputs()[0].getType())
              .getShape()
              .begin(),
          cast<RankedTensorType>(fold.getInputs()[0].getType())
              .getShape()
              .end());
    }
    if (task.getDomainShape() != ArrayRef<int64_t>(expectedDomain)) {
      task.emitOpError("domain does not match bound algebra indexing");
      return failure();
    }
    if (fusedSources) {
      for (Operation *operation : fusedOperations)
        ++sourceOccurrences[operation];
    } else {
      ++sourceOccurrences[source];
      for (Operation *operation : fusedOperations)
        ++sourceOccurrences[operation];
    }
  }
  if (sawFoldRegion || sawContractRegion) {
    for (auto [operation, occurrences] : regionOccurrences) {
      const int64_t declarations = recomputationDeclarations.lookup(operation);
      const int64_t expectedDeclarations =
          sourceOccurrences.lookup(operation) > 1 ? occurrences : 0;
      if (declarations != expectedDeclarations) {
        plan.emitOpError(
            "Fold-region recomputation must exactly declare repeated sources");
        return failure();
      }
    }
  }
  RegionOp owner;
  if (!bound.empty()) {
    owner = (*bound.begin())->getParentOfType<RegionOp>();
  }
  if (!owner || owner.getPolicy() != plan.getPolicy() ||
      bound.size() != owner.getBody().front().getOperations().size() - 1) {
    plan.emitOpError("must cover exactly one connected algebra region");
    return failure();
  }
  for (Operation *operation : bound) {
    if (operation->getParentOfType<RegionOp>() != owner) {
      plan.emitOpError("tasks must bind one algebra region");
      return failure();
    }
  }
  auto manifest =
      module->getAttrOfType<DictionaryAttr>(kCoverageManifestAttribute);
  auto tuning =
      manifest ? manifest.getAs<StringAttr>("canonical_tuning") : StringAttr{};
  const bool preferFusion =
      tuning && tuning.getValue() == kFusionCanonicalTuning;
  FailureOr<DerivedPlan> expected = deriveGpuPlan(owner, preferFusion);
  if (succeeded(expected) && failed(normalizePlainTaskInputs(*expected)))
    expected = failure();
  if (failed(expected) || expected->tasks.size() != tasks.size() ||
      expected->buffers.size() != buffers.size()) {
    plan.emitOpError("does not equal the plan derived from bound algebra");
    return failure();
  }
  for (auto [ordinal, expectedTask] : llvm::enumerate(expected->tasks)) {
    MaterializationTaskOp actual = tasks[ordinal];
    auto actualCompositionKind =
        actual->getAttrOfType<StringAttr>("composition_kind");
    auto actualCompositionSources =
        actual->getAttrOfType<ArrayAttr>("composition_sources");
    auto actualCompositionSourceExpansions =
        actual->getAttrOfType<ArrayAttr>("source_expansion_group");
    auto actualCompositionRecomputedSources =
        actual->getAttrOfType<ArrayAttr>("composition_recomputed_sources");
    auto actualCompositionPhases =
        actual->getAttrOfType<ArrayAttr>("composition_phases");
    auto actualCompositionPhaseEdges =
        actual->getAttrOfType<ArrayAttr>("composition_phase_edges");
    auto actualCompositionResultBindings =
        actual->getAttrOfType<ArrayAttr>("composition_result_bindings");
    auto actualCompositionCost =
        actual->getAttrOfType<DictionaryAttr>("composition_cost");
    auto actualFoldRegionSources =
        actual->getAttrOfType<ArrayAttr>("fold_region_sources");
    auto actualContractRegionSources =
        actual->getAttrOfType<ArrayAttr>("contract_region_sources");
    const bool expectedComposition =
        !expectedTask.compositionKind.empty() ||
        expectedTask.statefulContractFoldComposition;
    if (expectedComposition) {
      SmallVector<Attribute> expectedSources;
      SmallVector<Attribute> expectedSourceExpansions;
      for (Operation *operation : expectedTask.compositionSources)
        expectedSources.push_back(algebraSource(operation));
      for (Operation *operation : expectedTask.compositionSources)
        if (DictionaryAttr expansion = algebraSourceExpansion(operation))
          expectedSourceExpansions.push_back(expansion);
      SmallVector<Attribute> expectedRecomputedSources;
      for (Operation *operation : expectedTask.compositionRecomputedSources)
        expectedRecomputedSources.push_back(algebraSource(operation));
      auto costMatches = [&](StringRef field, int64_t expectedValue) {
        auto value = actualCompositionCost
                         ? actualCompositionCost.getAs<IntegerAttr>(field)
                         : IntegerAttr{};
        return value && value.getInt() == expectedValue;
      };
      const bool expectsCost = expectedTask.compositionKind == "map_contract" ||
                               (expectedTask.statefulContractFoldComposition &&
                                expectedTask.compositionStateInputs.empty() &&
                                expectedTask.compositionStateOutputs.empty());
      auto dimensionsMatch = [&](StringRef name, ArrayRef<int64_t> expected) {
        auto actualDimensions = actual->getAttrOfType<DenseI64ArrayAttr>(name);
        return expected.empty() ? (!actualDimensions ||
                                   (isa<ScanOp>(expectedTask.operation) &&
                                    actualDimensions.empty()))
                                : actualDimensions &&
                                      actualDimensions.asArrayRef() == expected;
      };
      auto affineMapsMatch = [&](StringRef name,
                                 ArrayRef<AffineMap> expectedMaps) {
        auto actualMaps = actual->getAttrOfType<ArrayAttr>(name);
        if (expectedMaps.empty())
          return !actualMaps;
        if (!actualMaps || actualMaps.size() != expectedMaps.size())
          return false;
        return llvm::all_of(llvm::zip_equal(actualMaps, expectedMaps),
                            [](auto pair) {
                              auto [actualMap, expectedMap] = pair;
                              auto affine = dyn_cast<AffineMapAttr>(actualMap);
                              return affine && affine.getValue() == expectedMap;
                            });
      };
      const auto cost = expectedTask.compositionCost;
      const bool matches =
          actualCompositionSources &&
          (expectedTask.compositionKind.empty()
               ? !actualCompositionKind
               : actualCompositionKind && actualCompositionKind.getValue() ==
                                              expectedTask.compositionKind) &&
          actualCompositionSources.getValue() ==
              ArrayRef<Attribute>(expectedSources) &&
          (expectedSourceExpansions.empty()
               ? !actualCompositionSourceExpansions
               : actualCompositionSourceExpansions &&
                     actualCompositionSourceExpansions.getValue() ==
                         ArrayRef<Attribute>(expectedSourceExpansions)) &&
          (expectedRecomputedSources.empty()
               ? !actualCompositionRecomputedSources
               : actualCompositionRecomputedSources &&
                     actualCompositionRecomputedSources.getValue() ==
                         ArrayRef<Attribute>(expectedRecomputedSources)) &&
          (expectedTask.compositionPhases.empty()
               ? !actualCompositionPhases && !actualCompositionPhaseEdges &&
                     !actualCompositionResultBindings
               : actualCompositionPhases &&
                     actualCompositionPhaseEdges &&
                     actualCompositionResultBindings &&
                     actualCompositionPhases.size() ==
                         expectedTask.compositionPhases.size() &&
                     actualCompositionResultBindings.size() ==
                         expectedTask.outputs.size()) &&
          (expectsCost == static_cast<bool>(actualCompositionCost)) &&
          (!expectsCost ||
           (costMatches("baseline_launches", cost.baselineLaunches) &&
            costMatches("candidate_launches", cost.candidateLaunches) &&
            costMatches("baseline_temporary_bytes",
                        cost.baselineTemporaryBytes) &&
            costMatches("candidate_temporary_bytes",
                        cost.candidateTemporaryBytes) &&
            costMatches("baseline_temporary_traffic_bytes",
                        cost.baselineTemporaryTrafficBytes) &&
            costMatches("candidate_temporary_traffic_bytes",
                        cost.candidateTemporaryTrafficBytes) &&
            costMatches("contract_evaluations", cost.contractEvaluations))) &&
          dimensionsMatch("composition_owner_dimensions",
                          expectedTask.compositionOwnerDimensions) &&
          dimensionsMatch("composition_stream_dimensions",
                          expectedTask.compositionStreamDimensions) &&
          dimensionsMatch("composition_state_inputs",
                          expectedTask.compositionStateInputs) &&
          dimensionsMatch("composition_state_outputs",
                          expectedTask.compositionStateOutputs) &&
          dimensionsMatch(
              "composition_producer_contraction_dimensions",
              expectedTask.compositionProducerContractionDimensions) &&
          dimensionsMatch(
              "composition_consumer_contraction_dimensions",
              expectedTask.compositionConsumerContractionDimensions) &&
          actual->getAttr("composition_index_domain_source") ==
              Attribute(expectedTask.compositionIndexDomainSource) &&
          dimensionsMatch("composition_index_domain_shape",
                          expectedTask.compositionIndexDomainShape) &&
          actual->getAttrOfType<AffineMapAttr>(
              "composition_index_domain_map") ==
              (expectedTask.compositionIndexDomainMap
                   ? AffineMapAttr::get(expectedTask.compositionIndexDomainMap)
                   : AffineMapAttr{}) &&
          actual->getAttr("composition_ordered_index_source") ==
              Attribute(expectedTask.compositionOrderedIndexSource) &&
          dimensionsMatch("composition_ordered_index_dimensions",
                          expectedTask.compositionOrderedIndexDimensions) &&
          actual->getAttrOfType<BoolAttr>(
              "composition_ordered_index_inclusive") ==
              (expectedTask.compositionOrderedIndexInclusive
                   ? BoolAttr::get(
                         module.getContext(),
                         *expectedTask.compositionOrderedIndexInclusive)
                   : BoolAttr{}) &&
          dimensionsMatch("composition_state_access_buffers",
                          expectedTask.compositionStateAccessBuffers) &&
          affineMapsMatch("composition_state_access_maps",
                          expectedTask.compositionStateAccessMaps) &&
          dimensionsMatch("online_vector_dimensions",
                          expectedTask.onlineVectorDimensions) &&
          expectedTask.operation ==
              (algebraSourceExpansion(expectedTask.operation)
                   ? expandedSourceOperations.lookup(ArrayAttr::get(
                         module.getContext(),
                         {actual.getSource(),
                          algebraSourceExpansion(expectedTask.operation)}))
                   : sourceOperations.lookup(actual.getSource())) &&
          actual.getDomainShape() == ArrayRef<int64_t>(expectedTask.domain) &&
          actual.getReductionDimensions() ==
              ArrayRef<int64_t>(expectedTask.reductionDimensions) &&
          actual.getOrderFree().value_or(false) == expectedTask.orderFree &&
          actual.getInputBuffers() == ArrayRef<int64_t>(expectedTask.inputs) &&
          actual.getOutputBuffers() ==
              ArrayRef<int64_t>(expectedTask.outputs) &&
          actual.getDependencies() ==
              ArrayRef<int64_t>(expectedTask.dependencies);
      if (!matches) {
        actual.emitOpError(
            "generic composition must equal structural SSA derivation");
        return failure();
      }
      continue;
    }
    if (actualCompositionKind || actualCompositionSources ||
        actualCompositionSourceExpansions ||
        actualCompositionRecomputedSources || actualCompositionPhases ||
        actualCompositionPhaseEdges || actualCompositionResultBindings ||
        actualCompositionCost ||
        actual->hasAttr("composition_owner_dimensions") ||
        actual->hasAttr("composition_stream_dimensions") ||
        actual->hasAttr("composition_state_inputs") ||
        actual->hasAttr("composition_state_outputs") ||
        actual->hasAttr("composition_producer_contraction_dimensions") ||
        actual->hasAttr("composition_consumer_contraction_dimensions") ||
        actual->hasAttr("composition_index_domain_source") ||
        actual->hasAttr("composition_index_domain_shape") ||
        actual->hasAttr("composition_index_domain_map") ||
        actual->hasAttr("composition_ordered_index_source") ||
        actual->hasAttr("composition_ordered_index_dimensions") ||
        actual->hasAttr("composition_ordered_index_inclusive") ||
        actual->hasAttr("composition_state_access_buffers") ||
        actual->hasAttr("composition_state_access_maps")) {
      actual.emitOpError("unexpected generic composition metadata");
      return failure();
    }
    const bool expectedContractRegion =
        !expectedTask.contractRegionSources.empty();
    if (expectedContractRegion) {
      SmallVector<Attribute> expectedSources;
      for (Operation *operation : expectedTask.contractRegionSources) {
        if (auto map = dyn_cast<MapOp>(operation))
          expectedSources.push_back(map.getSource());
        else if (auto fold = dyn_cast<FoldOp>(operation))
          expectedSources.push_back(fold.getSource());
        else
          expectedSources.push_back(cast<ContractOp>(operation).getSource());
      }
      SmallVector<Attribute> expectedEpilogues;
      for (Operation *operation : expectedTask.contractRegionEpilogueSources)
        expectedEpilogues.push_back(cast<MapOp>(operation).getSource());
      SmallVector<Attribute> expectedRecomputed;
      for (Operation *operation :
           expectedTask.contractRegionRecomputedSources) {
        if (auto map = dyn_cast<MapOp>(operation))
          expectedRecomputed.push_back(map.getSource());
        else
          expectedRecomputed.push_back(cast<FoldOp>(operation).getSource());
      }
      auto actualStrategy =
          actual->getAttrOfType<StringAttr>("contract_region_strategy");
      auto actualEpilogues =
          actual->getAttrOfType<ArrayAttr>("contract_region_epilogue_sources");
      auto actualOperand = actual->getAttrOfType<IntegerAttr>(
          "contract_region_normalized_operand");
      auto actualCost =
          actual->getAttrOfType<DictionaryAttr>("contract_region_cost");
      auto actualRecomputed = actual->getAttrOfType<ArrayAttr>(
          "contract_region_recomputed_sources");
      auto cost = expectedTask.contractRegionCost;
      auto costMatches = [&](StringRef field, int64_t expectedValue) {
        auto value =
            actualCost ? actualCost.getAs<IntegerAttr>(field) : IntegerAttr{};
        return value && value.getInt() == expectedValue;
      };
      const bool matches =
          actualContractRegionSources && actualStrategy && actualEpilogues &&
          actualOperand && actualCost &&
          actualContractRegionSources.getValue() ==
              ArrayRef<Attribute>(expectedSources) &&
          actualStrategy.getValue() == expectedTask.contractRegionStrategy &&
          actual->getAttr("contract_region_fold_source") ==
              cast<FoldOp>(expectedTask.contractRegionFold).getSource() &&
          actual->getAttr("contract_region_contract_source") ==
              cast<ContractOp>(expectedTask.contractRegionContract)
                  .getSource() &&
          actual->getAttr("contract_region_bf16_boundary_source") ==
              cast<MapOp>(expectedTask.contractRegionBf16Boundary)
                  .getSource() &&
          actualEpilogues.getValue() ==
              ArrayRef<Attribute>(expectedEpilogues) &&
          actualOperand.getInt() ==
              expectedTask.contractRegionNormalizedOperand &&
          (expectedRecomputed.empty()
               ? !actualRecomputed
               : actualRecomputed &&
                     actualRecomputed.getValue() ==
                         ArrayRef<Attribute>(expectedRecomputed)) &&
          costMatches("rows", cost.rows) &&
          costMatches("reduction", cost.reduction) &&
          costMatches("columns", cost.columns) &&
          costMatches("output_tile", cost.outputTile) &&
          costMatches("output_tiles", cost.outputTiles) &&
          costMatches("duplicated_fold_elements",
                      cost.duplicatedFoldElements) &&
          costMatches("normalized_temporary_bytes",
                      cost.normalizedTemporaryBytes) &&
          costMatches("contract_temporary_bytes",
                      cost.contractTemporaryBytes) &&
          costMatches("compact_stat_bytes", cost.compactStatBytes) &&
          expectedTask.operation ==
              sourceOperations.lookup(actual.getSource()) &&
          actual.getDomainShape() == ArrayRef<int64_t>(expectedTask.domain) &&
          actual.getReductionDimensions() ==
              ArrayRef<int64_t>(expectedTask.reductionDimensions) &&
          actual.getInputBuffers() == ArrayRef<int64_t>(expectedTask.inputs) &&
          actual.getOutputBuffers() ==
              ArrayRef<int64_t>(expectedTask.outputs) &&
          actual.getDependencies() ==
              ArrayRef<int64_t>(expectedTask.dependencies);
      if (!matches) {
        actual.emitOpError(
            "Contract-region plan must equal structural SSA derivation");
        return failure();
      }
      continue;
    }
    if (actualContractRegionSources) {
      actual.emitOpError("unexpected Contract-region metadata");
      return failure();
    }
    const bool expectedFoldRegion = !expectedTask.foldRegionSources.empty();
    if (expectedFoldRegion) {
      SmallVector<Attribute> expectedRegionSources;
      for (Operation *operation : expectedTask.foldRegionSources) {
        if (auto map = dyn_cast<MapOp>(operation))
          expectedRegionSources.push_back(map.getSource());
        else
          expectedRegionSources.push_back(cast<FoldOp>(operation).getSource());
      }
      auto actualPhases =
          actual->getAttrOfType<ArrayAttr>("fold_region_phases");
      auto actualOutputs =
          actual->getAttrOfType<ArrayAttr>("fold_region_output_sources");
      auto actualBoundaryUses =
          actual->getAttrOfType<ArrayAttr>("fold_region_boundary_uses");
      auto actualRecomputed =
          actual->getAttrOfType<ArrayAttr>("fold_region_recomputed_sources");
      SmallVector<Attribute> expectedRecomputed;
      for (Operation *operation : expectedTask.foldRegionRecomputedSources) {
        if (auto map = dyn_cast<MapOp>(operation))
          expectedRecomputed.push_back(map.getSource());
        else
          expectedRecomputed.push_back(cast<FoldOp>(operation).getSource());
      }
      SmallVector<Attribute> expectedOutputs;
      for (Operation *operation : expectedTask.foldRegionOutputs) {
        if (auto map = dyn_cast<MapOp>(operation))
          expectedOutputs.push_back(map.getSource());
        else
          expectedOutputs.push_back(cast<FoldOp>(operation).getSource());
      }
      bool matches =
          actualFoldRegionSources && actualPhases && actualOutputs &&
          actualBoundaryUses &&
          actualFoldRegionSources.getValue() ==
              ArrayRef<Attribute>(expectedRegionSources) &&
          actualOutputs.getValue() == ArrayRef<Attribute>(expectedOutputs) &&
          actualPhases.size() == expectedTask.foldRegionPhases.size() &&
          actualBoundaryUses.size() ==
              expectedTask.foldRegionBoundaryUses.size() &&
          (expectedRecomputed.empty()
               ? !actualRecomputed
               : actualRecomputed &&
                     actualRecomputed.getValue() ==
                         ArrayRef<Attribute>(expectedRecomputed));
      for (auto [phaseOrdinal, operation] :
           llvm::enumerate(expectedTask.foldRegionPhases)) {
        if (!matches)
          break;
        auto phase = dyn_cast<DictionaryAttr>(actualPhases[phaseOrdinal]);
        auto fold = cast<FoldOp>(operation);
        auto domain =
            phase
                ? dyn_cast_or_null<DenseI64ArrayAttr>(phase.get("domain_shape"))
                : DenseI64ArrayAttr{};
        auto reduction = phase ? dyn_cast_or_null<DenseI64ArrayAttr>(
                                     phase.get("reduction_dimensions"))
                               : DenseI64ArrayAttr{};
        matches = phase && phase.get("source") == fold.getSource() && domain &&
                  reduction &&
                  domain.asArrayRef() ==
                      cast<RankedTensorType>(fold.getInputs().front().getType())
                          .getShape() &&
                  reduction.asArrayRef() == fold.getReductionDimensions() &&
                  cast<BoolAttr>(phase.get("order_free")).getValue() ==
                      fold.getOrderFree() &&
                  cast<StringAttr>(phase.get("transition")).getValue() ==
                      (phaseOrdinal == 0 ? "launch_boundary" : "cta_local");
      }
      for (auto [useOrdinal, expectedUse] :
           llvm::enumerate(expectedTask.foldRegionBoundaryUses)) {
        if (!matches)
          break;
        auto use = dyn_cast<DictionaryAttr>(actualBoundaryUses[useOrdinal]);
        matches =
            use &&
            cast<IntegerAttr>(use.get("input_buffer")).getInt() ==
                expectedUse.inputBuffer &&
            use.get("source") == expectedUse.source.getSource() &&
            cast<IntegerAttr>(use.get("operand")).getInt() ==
                expectedUse.operand &&
            cast<AffineMapAttr>(use.get("indexing_map")).getValue() ==
                expectedUse.indexingMap &&
            cast<IntegerAttr>(use.get("phase")).getInt() == expectedUse.phase;
      }
      if (!matches || expectedTask.operation != taskSourceOperation(actual) ||
          actual.getReductionDimensions() !=
              ArrayRef<int64_t>(expectedTask.reductionDimensions) ||
          actual.getOrderFree().value_or(false) != expectedTask.orderFree ||
          actual.getInputBuffers() != ArrayRef<int64_t>(expectedTask.inputs) ||
          actual.getOutputBuffers() !=
              ArrayRef<int64_t>(expectedTask.outputs) ||
          actual.getDependencies() !=
              ArrayRef<int64_t>(expectedTask.dependencies)) {
        actual.emitOpError(
            "Fold-region plan must equal structural SSA derivation");
        return failure();
      }
      continue;
    }
    if (actualFoldRegionSources) {
      actual.emitOpError("unexpected Fold-region metadata");
      return failure();
    }
    SmallVector<Attribute> expectedSources;
    for (Operation *operation : expectedTask.fusedOperations)
      expectedSources.push_back(cast<MapOp>(operation).getSource());
    SmallVector<Attribute> expectedMaps;
    for (AffineMap map : expectedTask.composedInputMaps)
      expectedMaps.push_back(AffineMapAttr::get(map));
    auto actualSources = actual->getAttrOfType<ArrayAttr>("fused_sources");
    auto actualMaps = actual->getAttrOfType<ArrayAttr>("composed_input_maps");
    auto actualOutputs =
        actual->getAttrOfType<ArrayAttr>("fused_output_sources");
    auto actualEpilogueSources =
        actual->getAttrOfType<ArrayAttr>("fold_epilogue_sources");
    auto actualEpilogueMaps =
        actual->getAttrOfType<ArrayAttr>("epilogue_input_maps");
    const bool expectedFoldMap = expectedTask.fusedFold != nullptr;
    SmallVector<Attribute> expectedOutputs;
    if (expectedTask.fusedMapOutputs.size() > 1)
      for (Operation *operation : expectedTask.fusedMapOutputs)
        expectedOutputs.push_back(cast<MapOp>(operation).getSource());
    if (expectedTask.operation != taskSourceOperation(actual) ||
        (expectedFoldMap && (actualSources || actualMaps)) ||
        (!expectedFoldMap && (expectedSources.empty() != !actualSources)) ||
        (!expectedFoldMap && !expectedSources.empty() &&
         actualSources.getValue() != ArrayRef<Attribute>(expectedSources)) ||
        (!expectedFoldMap && (expectedMaps.empty() != !actualMaps)) ||
        (!expectedFoldMap && !expectedMaps.empty() &&
         actualMaps.getValue() != ArrayRef<Attribute>(expectedMaps)) ||
        (expectedOutputs.empty() != !actualOutputs) ||
        (!expectedOutputs.empty() &&
         actualOutputs.getValue() != ArrayRef<Attribute>(expectedOutputs)) ||
        (expectedFoldMap != static_cast<bool>(actualEpilogueSources)) ||
        (expectedFoldMap && actualEpilogueSources.getValue() !=
                                ArrayRef<Attribute>(expectedSources)) ||
        (expectedFoldMap != static_cast<bool>(actualEpilogueMaps)) ||
        (expectedFoldMap &&
         actualEpilogueMaps.getValue() != ArrayRef<Attribute>(expectedMaps)) ||
        actual.getReductionDimensions() !=
            ArrayRef<int64_t>(expectedTask.reductionDimensions) ||
        actual.getOrderFree().value_or(false) != expectedTask.orderFree ||
        actual.getInputBuffers() != ArrayRef<int64_t>(expectedTask.inputs) ||
        actual.getOutputBuffers() != ArrayRef<int64_t>(expectedTask.outputs) ||
        actual.getDependencies() !=
            ArrayRef<int64_t>(expectedTask.dependencies)) {
      actual.emitOpError(
          "task order and edges must equal bound algebra SSA dependencies");
      return failure();
    }
  }
  for (auto [ordinal, expectedBuffer] : llvm::enumerate(expected->buffers)) {
    MaterializationBufferOp actual = buffers[ordinal];
    auto actualProducer = actual->getAttrOfType<IntegerAttr>("producer");
    auto actualSourceArgument =
        actual->getAttrOfType<IntegerAttr>("source_argument");
    std::optional<int64_t> producer =
        actualProducer ? std::optional<int64_t>(actualProducer.getInt())
                       : std::nullopt;
    auto expectedArgument = dyn_cast<BlockArgument>(expectedBuffer.value);
    if (actual.getTensorType() != expectedBuffer.value.getType() ||
        producer != expectedBuffer.producer ||
        (expectedArgument
             ? (!actualSourceArgument || actualSourceArgument.getInt() !=
                                             expectedArgument.getArgNumber())
             : static_cast<bool>(actualSourceArgument)) ||
        actual.getConsumers() != ArrayRef<int64_t>(expectedBuffer.consumers) ||
        actual.getLiveOut() != expectedBuffer.liveOut) {
      actual.emitOpError(
          "buffer ownership and uses must equal bound algebra SSA edges");
      return failure();
    }
  }
  if (boundRegion)
    *boundRegion = owner;
  return success();
}

class VerifyMaterializationPlanPass
    : public impl::ShuttleVerifyMaterializationPlanPassBase<
          VerifyMaterializationPlanPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<MaterializationPlanOp> plans(
        module.getOps<MaterializationPlanOp>());
    SmallVector<RegionOp> regions;
    llvm::SmallDenseSet<Operation *> owners;
    bool invalid = plans.empty();
    for (MaterializationPlanOp plan : plans) {
      RegionOp owner;
      if (failed(
              verifyMaterializationPlanAgainstSource(module, plan, &owner)) ||
          !owners.insert(owner.getOperation()).second) {
        invalid = true;
        break;
      }
      regions.push_back(owner);
    }
    size_t regionCount = 0;
    module.walk([&](RegionOp) { ++regionCount; });
    if (invalid || regions.size() != regionCount) {
      module.emitError(
          "requires one source-bound materialization plan per GPU region");
      signalPassFailure();
    }
  }
};
