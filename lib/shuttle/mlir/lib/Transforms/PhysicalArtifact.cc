// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/PhysicalArtifact.h"

#include <mutex>
#include <utility>

#include "absl/status/status.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::shuttle {
namespace {

std::string printedAttribute(Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return text;
}

std::string printedType(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

llvm::json::Value encodeAffineExpr(AffineExpr expression) {
  llvm::json::Object record;
  if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
    record["kind"] = "dimension";
    record["position"] = static_cast<int64_t>(dimension.getPosition());
    return record;
  }
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expression)) {
    record["kind"] = "symbol";
    record["position"] = static_cast<int64_t>(symbol.getPosition());
    return record;
  }
  if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
    record["kind"] = "constant";
    record["value"] = constant.getValue();
    return record;
  }
  auto binary = cast<AffineBinaryOpExpr>(expression);
  switch (expression.getKind()) {
  case AffineExprKind::Add:
    record["kind"] = "add";
    break;
  case AffineExprKind::Mul:
    record["kind"] = "multiply";
    break;
  case AffineExprKind::Mod:
    record["kind"] = "modulo";
    break;
  case AffineExprKind::FloorDiv:
    record["kind"] = "floor_divide";
    break;
  case AffineExprKind::CeilDiv:
    record["kind"] = "ceil_divide";
    break;
  default:
    llvm_unreachable("unknown affine expression kind");
  }
  record["left"] = encodeAffineExpr(binary.getLHS());
  record["right"] = encodeAffineExpr(binary.getRHS());
  return record;
}

llvm::json::Value encodeType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    llvm::json::Array shape;
    for (int64_t extent : tensor.getShape())
      shape.push_back(extent);
    llvm::json::Object record;
    record["kind"] = "ranked_tensor";
    record["shape"] = std::move(shape);
    record["element_type"] = printedType(tensor.getElementType());
    return record;
  }
  llvm::json::Object record;
  record["kind"] = "scalar";
  record["element_type"] = printedType(type);
  return record;
}

llvm::json::Value encodeAttribute(Attribute attribute) {
  if (auto value = dyn_cast<BoolAttr>(attribute))
    return value.getValue();
  if (auto value = dyn_cast<IntegerAttr>(attribute))
    return value.getInt();
  if (auto value = dyn_cast<FloatAttr>(attribute)) {
    llvm::SmallString<32> bits;
    value.getValue().bitcastToAPInt().toString(bits, 16, false);
    llvm::json::Object record;
    record["kind"] = "float";
    record["element_type"] = printedType(value.getType());
    record["bits"] = bits.str().str();
    return record;
  }
  if (auto value = dyn_cast<StringAttr>(attribute))
    return value.getValue().str();
  if (auto value = dyn_cast<TypeAttr>(attribute))
    return encodeType(value.getValue());
  if (auto value = dyn_cast<DenseI64ArrayAttr>(attribute)) {
    llvm::json::Array result;
    for (int64_t element : value.asArrayRef())
      result.push_back(element);
    return result;
  }
  if (auto value = dyn_cast<DenseI32ArrayAttr>(attribute)) {
    llvm::json::Array result;
    for (int32_t element : value.asArrayRef())
      result.push_back(element);
    return result;
  }
  if (auto value = dyn_cast<ArrayAttr>(attribute)) {
    llvm::json::Array result;
    for (Attribute element : value)
      result.push_back(encodeAttribute(element));
    return result;
  }
  if (auto value = dyn_cast<DictionaryAttr>(attribute)) {
    llvm::json::Object result;
    for (NamedAttribute element : value)
      result[element.getName().strref()] = encodeAttribute(element.getValue());
    return result;
  }
  if (auto value = dyn_cast<AffineMapAttr>(attribute)) {
    AffineMap map = value.getValue();
    llvm::json::Array results;
    for (AffineExpr expression : map.getResults())
      results.push_back(encodeAffineExpr(expression));
    llvm::json::Object record;
    record["kind"] = "affine_map";
    record["dimensions"] = static_cast<int64_t>(map.getNumDims());
    record["symbols"] = static_cast<int64_t>(map.getNumSymbols());
    record["results"] = std::move(results);
    return record;
  }
  if (auto value = dyn_cast<IntegerSetAttr>(attribute)) {
    IntegerSet set = value.getValue();
    llvm::json::Array constraints;
    llvm::json::Array equalities;
    for (auto [constraint, equality] :
         llvm::zip_equal(set.getConstraints(), set.getEqFlags())) {
      constraints.push_back(encodeAffineExpr(constraint));
      equalities.push_back(equality);
    }
    llvm::json::Object record;
    record["kind"] = "integer_set";
    record["dimensions"] = static_cast<int64_t>(set.getNumDims());
    record["symbols"] = static_cast<int64_t>(set.getNumSymbols());
    record["constraints"] = std::move(constraints);
    record["equalities"] = std::move(equalities);
    return record;
  }
  llvm::json::Object record;
  record["kind"] = "mlir_attribute";
  record["value"] = printedAttribute(attribute);
  return record;
}

class StructuredOperationEncoder {
public:
  llvm::json::Object encode(Operation *operation) {
    llvm::json::Object record;
    record["name"] = operation->getName().getStringRef();
    llvm::json::Object attributes;
    for (NamedAttribute attribute : operation->getAttrs())
      attributes[attribute.getName().strref()] =
          encodeAttribute(attribute.getValue());
    record["attributes"] = std::move(attributes);
    llvm::json::Array operands;
    for (Value operand : operation->getOperands())
      operands.push_back(valueId(operand));
    record["operands"] = std::move(operands);
    llvm::json::Array results;
    for (Value result : operation->getResults()) {
      llvm::json::Object value;
      value["id"] = valueId(result);
      value["type"] = encodeType(result.getType());
      results.push_back(std::move(value));
    }
    record["results"] = std::move(results);
    llvm::json::Array regions;
    for (Region &region : operation->getRegions()) {
      llvm::json::Array blocks;
      for (Block &block : region) {
        llvm::json::Object blockRecord;
        llvm::json::Array arguments;
        for (BlockArgument argument : block.getArguments()) {
          llvm::json::Object value;
          value["id"] = valueId(argument);
          value["type"] = encodeType(argument.getType());
          arguments.push_back(std::move(value));
        }
        blockRecord["arguments"] = std::move(arguments);
        llvm::json::Array operations;
        for (Operation &nested : block)
          operations.push_back(encode(&nested));
        blockRecord["operations"] = std::move(operations);
        blocks.push_back(std::move(blockRecord));
      }
      regions.push_back(std::move(blocks));
    }
    record["regions"] = std::move(regions);
    return record;
  }

private:
  int64_t valueId(Value value) {
    auto [position, inserted] = values.try_emplace(value, nextValue);
    if (inserted)
      ++nextValue;
    return position->second;
  }
  llvm::DenseMap<Value, int64_t> values;
  int64_t nextValue = 0;
};

void writeCanonicalJson(const llvm::json::Value &value,
                        llvm::raw_ostream &output) {
  if (const auto *object = value.getAsObject()) {
    llvm::SmallVector<llvm::StringRef> keys;
    for (const auto &field : *object)
      keys.push_back(field.first);
    llvm::sort(keys);
    output << '{';
    for (auto [index, key] : llvm::enumerate(keys)) {
      if (index)
        output << ',';
      output << llvm::json::Value(key) << ':';
      writeCanonicalJson(*object->get(key), output);
    }
    output << '}';
    return;
  }
  if (const auto *array = value.getAsArray()) {
    output << '[';
    for (auto [index, element] : llvm::enumerate(*array)) {
      if (index)
        output << ',';
      writeCanonicalJson(element, output);
    }
    output << ']';
    return;
  }
  output << value;
}

llvm::json::Object
structuralRequestRecord(RegionOp region, MaterializationPlanOp materialization,
                        SchedulePlanOp schedule, EventLoweringPlanOp eventPlan,
                        GpuTarget target) {
  StructuredOperationEncoder encoder;
  llvm::json::Object record;
  record["schema_version"] = 3;
  record["target"] = target == GpuTarget::Sm90 ? "sm_90" : "sm_100";
  record["region"] = encoder.encode(region.getOperation());
  record["materialization"] = encoder.encode(materialization.getOperation());
  record["schedule"] = encoder.encode(schedule.getOperation());
  record["event_lowering_plan"] = encoder.encode(eventPlan.getOperation());
  return record;
}

std::mutex &providerMutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::shared_ptr<const GpuPhysicalArtifactProvider> &providerStorage() {
  static auto *provider =
      new std::shared_ptr<const GpuPhysicalArtifactProvider>();
  return *provider;
}

} // namespace

std::string gpuPhysicalArtifactRequestFingerprint(
    RegionOp region, MaterializationPlanOp materialization,
    SchedulePlanOp schedule, EventLoweringPlanOp eventPlan, GpuTarget target) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  writeCanonicalJson(llvm::json::Value(structuralRequestRecord(
                         region, materialization, schedule, eventPlan, target)),
                     stream);
  stream.flush();
  llvm::SHA256 digest;
  digest.update(canonical);
  return llvm::toHex(digest.final(), true);
}

std::string
gpuPhysicalArtifactRequestJson(const GpuPhysicalArtifactRequest &request) {
  llvm::json::Object record =
      structuralRequestRecord(request.region, request.materialization,
                              request.schedule, request.eventPlan,
                              request.target);
  record["fingerprint"] = request.fingerprint;
  llvm::json::Array bindingIndices;
  for (std::optional<int64_t> binding : request.bindingIndices) {
    if (binding)
      bindingIndices.push_back(*binding);
    else
      bindingIndices.push_back(nullptr);
  }
  record["binding_indices"] = std::move(bindingIndices);
  llvm::json::Array regionResultIndices;
  for (std::optional<int64_t> result : request.regionResultIndices) {
    if (result)
      regionResultIndices.push_back(*result);
    else
      regionResultIndices.push_back(nullptr);
  }
  record["region_result_indices"] = std::move(regionResultIndices);
  llvm::json::Array reuseGroups;
  for (int64_t group : request.reuseGroups)
    reuseGroups.push_back(group);
  record["reuse_groups"] = std::move(reuseGroups);
  std::string json;
  llvm::raw_string_ostream stream(json);
  writeCanonicalJson(llvm::json::Value(std::move(record)), stream);
  stream.flush();
  return json;
}

absl::Status registerGpuPhysicalArtifactProvider(
    std::shared_ptr<const GpuPhysicalArtifactProvider> provider) {
  if (!provider)
    return absl::InvalidArgumentError(
        "Shuttle GPU physical artifact provider must be non-null");
  std::lock_guard<std::mutex> lock(providerMutex());
  if (providerStorage())
    return absl::AlreadyExistsError(
        "Shuttle GPU physical artifact provider is already registered");
  providerStorage() = std::move(provider);
  return absl::OkStatus();
}

std::shared_ptr<const GpuPhysicalArtifactProvider>
gpuPhysicalArtifactProvider() {
  std::lock_guard<std::mutex> lock(providerMutex());
  return providerStorage();
}

} // namespace mlir::shuttle
