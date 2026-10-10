// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Runtime/GpuExecutable.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::shuttle {
namespace {

absl::Status invalidInternalEvent(llvm::Twine detail) {
  return absl::InvalidArgumentError(
      (llvm::Twine("invalid refined internal Event schedule: ") + detail)
          .str());
}

bool isDigest(llvm::StringRef value) {
  return value.size() == 64 && llvm::all_of(value, [](char character) {
           return llvm::isDigit(character) ||
                  (character >= 'a' && character <= 'f');
         });
}

bool hasOnlyFields(const llvm::json::Object &object,
                   llvm::ArrayRef<llvm::StringRef> allowed) {
  return llvm::all_of(object, [&](const auto &field) {
    return llvm::is_contained(allowed, field.first.str());
  });
}

std::string coordinateKey(llvm::ArrayRef<int64_t> coordinate) {
  std::string key;
  llvm::raw_string_ostream stream(key);
  for (int64_t value : coordinate)
    stream << value << ',';
  return key;
}

std::string pairKey(const GpuInternalEventRelationPair &pair) {
  return coordinateKey(pair.source) + ">" + coordinateKey(pair.target);
}

bool sameAxes(llvm::ArrayRef<GpuInternalEventAxis> lhs,
              llvm::ArrayRef<GpuInternalEventAxis> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
           return std::get<0>(pair).name == std::get<1>(pair).name &&
                  std::get<0>(pair).extent == std::get<1>(pair).extent;
         });
}

bool sameDomain(const GpuInternalEventDomain &lhs,
                const GpuInternalEventDomain &rhs) {
  return lhs.kind == rhs.kind && lhs.name == rhs.name &&
         lhs.placement == rhs.placement && sameAxes(lhs.axes, rhs.axes);
}

bool coordinateInDomain(llvm::ArrayRef<int64_t> coordinate,
                        const GpuInternalEventDomain &domain) {
  if (coordinate.size() != domain.axes.size())
    return false;
  return llvm::all_of(llvm::zip(coordinate, domain.axes), [](auto pair) {
    const int64_t value = std::get<0>(pair);
    return value >= 0 && value < std::get<1>(pair).extent;
  });
}

void enumerateCoordinates(const GpuInternalEventDomain &domain, size_t axis,
                          GpuInternalEventCoordinate &coordinate,
                          std::vector<GpuInternalEventCoordinate> &result) {
  if (axis == domain.axes.size()) {
    result.push_back(coordinate);
    return;
  }
  for (int64_t value = 0; value < domain.axes[axis].extent; ++value) {
    coordinate.push_back(value);
    enumerateCoordinates(domain, axis + 1, coordinate, result);
    coordinate.pop_back();
  }
}

std::vector<GpuInternalEventCoordinate>
domainCoordinates(const GpuInternalEventDomain &domain) {
  std::vector<GpuInternalEventCoordinate> result;
  GpuInternalEventCoordinate coordinate;
  enumerateCoordinates(domain, 0, coordinate, result);
  return result;
}

absl::StatusOr<GpuInternalEventCoordinate>
parseCoordinate(const llvm::json::Value &value, llvm::StringRef context) {
  const llvm::json::Array *array = value.getAsArray();
  if (!array)
    return invalidInternalEvent(llvm::Twine(context) +
                                " coordinate must be an array");
  GpuInternalEventCoordinate coordinate;
  coordinate.reserve(array->size());
  for (const llvm::json::Value &component : *array) {
    std::optional<int64_t> integer = component.getAsInteger();
    if (!integer || *integer < 0)
      return invalidInternalEvent(
          llvm::Twine(context) +
          " coordinate must contain nonnegative integers");
    coordinate.push_back(*integer);
  }
  return coordinate;
}

absl::StatusOr<std::vector<GpuInternalEventCoordinate>>
parseCoordinates(const llvm::json::Array *array, llvm::StringRef context) {
  if (!array)
    return invalidInternalEvent(llvm::Twine(context) + " must be an array");
  std::vector<GpuInternalEventCoordinate> coordinates;
  coordinates.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    auto coordinate = parseCoordinate(value, context);
    if (!coordinate.ok())
      return coordinate.status();
    coordinates.push_back(std::move(*coordinate));
  }
  return coordinates;
}

absl::StatusOr<std::vector<int64_t>>
parseIntegers(const llvm::json::Array *array, llvm::StringRef context,
              bool nonnegative = true) {
  if (!array)
    return invalidInternalEvent(llvm::Twine(context) + " must be an array");
  std::vector<int64_t> integers;
  integers.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    std::optional<int64_t> integer = value.getAsInteger();
    if (!integer || (nonnegative && *integer < 0))
      return invalidInternalEvent(llvm::Twine(context) +
                                  " must contain valid integers");
    integers.push_back(*integer);
  }
  return integers;
}

absl::StatusOr<GpuEventPhysicalScope> parseScope(llvm::StringRef scope,
                                                 llvm::StringRef context) {
  if (scope == "cta")
    return GpuEventPhysicalScope::ThreadBlock;
  if (scope == "cluster")
    return GpuEventPhysicalScope::Cluster;
  if (scope == "device")
    return GpuEventPhysicalScope::Device;
  if (scope == "system")
    return GpuEventPhysicalScope::System;
  return invalidInternalEvent(llvm::Twine(context) +
                              " has unsupported scope '" + scope + "'");
}

absl::StatusOr<GpuInternalEventDomain>
parseDomain(const llvm::json::Object *object, llvm::StringRef context) {
  if (!object)
    return invalidInternalEvent(llvm::Twine(context) + " must be an object");
  if (!hasOnlyFields(*object, {"kind", "name", "axes", "placement"}))
    return invalidInternalEvent(llvm::Twine(context) +
                                " has unsupported domain fields");
  std::optional<llvm::StringRef> kind = object->getString("kind");
  std::optional<llvm::StringRef> name = object->getString("name");
  const llvm::json::Array *axes = object->getArray("axes");
  if (!kind || !name || name->empty() || !axes)
    return invalidInternalEvent(llvm::Twine(context) +
                                " requires kind, name, and axes");
  GpuInternalEventDomain domain;
  if (*kind == "task_family")
    domain.kind = GpuInternalEventDomainKind::TaskFamily;
  else if (*kind == "event_domain")
    domain.kind = GpuInternalEventDomainKind::EventDomain;
  else
    return invalidInternalEvent(llvm::Twine(context) +
                                " has unsupported domain kind");
  domain.name = name->str();
  std::set<std::string> axisNames;
  for (const llvm::json::Value &value : *axes) {
    const llvm::json::Object *axis = value.getAsObject();
    if (!axis)
      return invalidInternalEvent(llvm::Twine(context) +
                                  " axis must be an object");
    if (!hasOnlyFields(*axis, {"name", "extent"}))
      return invalidInternalEvent(llvm::Twine(context) +
                                  " axis has unsupported fields");
    std::optional<llvm::StringRef> axisName = axis->getString("name");
    std::optional<int64_t> extent = axis->getInteger("extent");
    if (!axisName || axisName->empty() || !extent || *extent <= 0 ||
        !axisNames.insert(axisName->str()).second)
      return invalidInternalEvent(llvm::Twine(context) +
                                  " requires unique positive axes");
    domain.axes.push_back({axisName->str(), *extent});
  }
  std::optional<llvm::StringRef> placement = object->getString("placement");
  if (domain.kind == GpuInternalEventDomainKind::TaskFamily) {
    if (!placement || placement->empty())
      return invalidInternalEvent(llvm::Twine(context) +
                                  " task family requires placement");
    domain.placement = placement->str();
  } else if (placement) {
    return invalidInternalEvent(llvm::Twine(context) +
                                " event domain cannot have placement");
  }
  return domain;
}

absl::StatusOr<GpuInternalEventRelation>
parseRelation(const llvm::json::Object *object, llvm::StringRef context) {
  if (!object)
    return invalidInternalEvent(llvm::Twine(context) + " must be an object");
  if (!hasOnlyFields(*object, {"source_domain", "target_domain", "pairs"}))
    return invalidInternalEvent(llvm::Twine(context) +
                                " has unsupported relation fields");
  auto source = parseDomain(object->getObject("source_domain"),
                            (llvm::Twine(context) + ".source_domain").str());
  auto target = parseDomain(object->getObject("target_domain"),
                            (llvm::Twine(context) + ".target_domain").str());
  const llvm::json::Array *pairs = object->getArray("pairs");
  if (!source.ok())
    return source.status();
  if (!target.ok())
    return target.status();
  if (!pairs)
    return invalidInternalEvent(llvm::Twine(context) +
                                ".pairs must be an array");
  GpuInternalEventRelation relation{std::move(*source), std::move(*target), {}};
  std::set<std::string> pairKeys;
  for (const llvm::json::Value &value : *pairs) {
    const llvm::json::Object *pair = value.getAsObject();
    if (!pair)
      return invalidInternalEvent(llvm::Twine(context) +
                                  " relation pair must be an object");
    if (!hasOnlyFields(*pair, {"source", "target"}))
      return invalidInternalEvent(llvm::Twine(context) +
                                  " relation pair has unsupported fields");
    const llvm::json::Value *sourceValue = pair->get("source");
    const llvm::json::Value *targetValue = pair->get("target");
    if (!sourceValue || !targetValue)
      return invalidInternalEvent(llvm::Twine(context) +
                                  " relation pair requires source and target");
    auto sourceCoordinate = parseCoordinate(*sourceValue, context);
    auto targetCoordinate = parseCoordinate(*targetValue, context);
    if (!sourceCoordinate.ok())
      return sourceCoordinate.status();
    if (!targetCoordinate.ok())
      return targetCoordinate.status();
    GpuInternalEventRelationPair parsedPair{std::move(*sourceCoordinate),
                                            std::move(*targetCoordinate)};
    if (!coordinateInDomain(parsedPair.source, relation.source) ||
        !coordinateInDomain(parsedPair.target, relation.target))
      return invalidInternalEvent(
          llvm::Twine(context) +
          " relation pair is outside its declared domains");
    if (!pairKeys.insert(pairKey(parsedPair)).second)
      return invalidInternalEvent(llvm::Twine(context) +
                                  " contains a duplicate relation pair");
    relation.pairs.push_back(std::move(parsedPair));
  }
  return relation;
}

absl::StatusOr<GpuInternalEventVisibility>
parseVisibility(const llvm::json::Object *object, llvm::StringRef context) {
  if (!object)
    return invalidInternalEvent(llvm::Twine(context) + " must be an object");
  if (!hasOnlyFields(*object,
                     {"scope", "release_on_notify", "acquire_before_consumer"}))
    return invalidInternalEvent(llvm::Twine(context) +
                                " has unsupported visibility fields");
  std::optional<llvm::StringRef> scope = object->getString("scope");
  std::optional<bool> release = object->getBoolean("release_on_notify");
  std::optional<bool> acquire = object->getBoolean("acquire_before_consumer");
  if (!scope || !release || !acquire)
    return invalidInternalEvent(llvm::Twine(context) +
                                " requires scope, release, and acquire");
  auto parsedScope = parseScope(*scope, context);
  if (!parsedScope.ok())
    return parsedScope.status();
  return GpuInternalEventVisibility{*parsedScope, *release, *acquire};
}

absl::StatusOr<GpuInternalEventTensorPlan>
parseEvent(const llvm::json::Object *object, llvm::StringRef context) {
  if (!object)
    return invalidInternalEvent(llvm::Twine(context) + " must be an object");
  if (!hasOnlyFields(*object,
                     {"name", "required_dependence", "event_domain",
                      "notify_relation", "trigger_relation", "initial_count",
                      "memory_scope", "generation_policy", "visibility",
                      "scheduling_mode"}))
    return invalidInternalEvent(llvm::Twine(context) +
                                " has unsupported EventTensor fields");
  std::optional<llvm::StringRef> name = object->getString("name");
  const llvm::json::Object *required = object->getObject("required_dependence");
  const llvm::json::Object *initial = object->getObject("initial_count");
  std::optional<llvm::StringRef> memoryScope =
      object->getString("memory_scope");
  std::optional<llvm::StringRef> generationPolicy =
      object->getString("generation_policy");
  if (!name || name->empty() || !required || !initial || !memoryScope ||
      !generationPolicy)
    return invalidInternalEvent(llvm::Twine(context) +
                                " is missing required EventTensor fields");
  if (!hasOnlyFields(*required, {"relation", "visibility"}) ||
      !hasOnlyFields(*initial, {"counts", "provenance"}))
    return invalidInternalEvent(llvm::Twine(context) +
                                " has unsupported dependence/count fields");
  auto requiredRelation = parseRelation(
      required->getObject("relation"),
      (llvm::Twine(context) + ".required_dependence.relation").str());
  auto requiredVisibility = parseVisibility(
      required->getObject("visibility"),
      (llvm::Twine(context) + ".required_dependence.visibility").str());
  auto eventDomain =
      parseDomain(object->getObject("event_domain"),
                  (llvm::Twine(context) + ".event_domain").str());
  auto notifyRelation =
      parseRelation(object->getObject("notify_relation"),
                    (llvm::Twine(context) + ".notify_relation").str());
  auto triggerRelation =
      parseRelation(object->getObject("trigger_relation"),
                    (llvm::Twine(context) + ".trigger_relation").str());
  auto visibility =
      parseVisibility(object->getObject("visibility"),
                      (llvm::Twine(context) + ".visibility").str());
  auto parsedScope = parseScope(*memoryScope, context);
  if (!requiredRelation.ok())
    return requiredRelation.status();
  if (!requiredVisibility.ok())
    return requiredVisibility.status();
  if (!eventDomain.ok())
    return eventDomain.status();
  if (!notifyRelation.ok())
    return notifyRelation.status();
  if (!triggerRelation.ok())
    return triggerRelation.status();
  if (!visibility.ok())
    return visibility.status();
  if (!parsedScope.ok())
    return parsedScope.status();
  if (requiredRelation->source.kind != GpuInternalEventDomainKind::TaskFamily ||
      requiredRelation->target.kind != GpuInternalEventDomainKind::TaskFamily ||
      eventDomain->kind != GpuInternalEventDomainKind::EventDomain ||
      !sameDomain(notifyRelation->source, requiredRelation->source) ||
      !sameDomain(notifyRelation->target, *eventDomain) ||
      !sameDomain(triggerRelation->source, *eventDomain) ||
      !sameDomain(triggerRelation->target, requiredRelation->target))
    return invalidInternalEvent(llvm::Twine(context) +
                                " has inconsistent task and Event domains");
  if (requiredVisibility->scope != *parsedScope ||
      visibility->scope != *parsedScope ||
      requiredVisibility->releaseOnNotify != visibility->releaseOnNotify ||
      requiredVisibility->acquireBeforeConsumer !=
          visibility->acquireBeforeConsumer)
    return invalidInternalEvent(llvm::Twine(context) +
                                " has inconsistent visibility contracts");

  GpuInternalEventTensorPlan event;
  event.name = name->str();
  event.requiredDependence = std::move(*requiredRelation);
  event.requiredVisibility = *requiredVisibility;
  event.eventDomain = std::move(*eventDomain);
  event.notifyRelation = std::move(*notifyRelation);
  event.triggerRelation = std::move(*triggerRelation);
  event.memoryScope = *parsedScope;
  event.visibility = *visibility;
  if (*generationPolicy == "per_invocation")
    event.generationPolicy = GpuInternalEventGenerationPolicy::PerInvocation;
  else if (*generationPolicy == "phased")
    event.generationPolicy = GpuInternalEventGenerationPolicy::Phased;
  else
    return invalidInternalEvent(llvm::Twine(context) +
                                " has unsupported generation policy");
  if (std::optional<llvm::StringRef> schedulingMode =
          object->getString("scheduling_mode")) {
    if (*schedulingMode != "static" && *schedulingMode != "dynamic")
      return invalidInternalEvent(llvm::Twine(context) +
                                  " has unsupported scheduling mode");
    event.schedulingMode = schedulingMode->str();
  } else if (const llvm::json::Value *mode = object->get("scheduling_mode");
             mode && mode->kind() != llvm::json::Value::Null) {
    return invalidInternalEvent(llvm::Twine(context) +
                                " scheduling_mode must be null or a string");
  }

  const llvm::json::Array *counts = initial->getArray("counts");
  std::optional<llvm::StringRef> provenance = initial->getString("provenance");
  if (!counts || !provenance || provenance->empty())
    return invalidInternalEvent(
        llvm::Twine(context) + " initial_count requires counts and provenance");
  std::map<std::string, int64_t> initialByCoordinate;
  for (const llvm::json::Value &value : *counts) {
    const llvm::json::Object *count = value.getAsObject();
    const llvm::json::Value *coordinateValue =
        count ? count->get("coordinate") : nullptr;
    std::optional<int64_t> countValue =
        count ? count->getInteger("value") : std::nullopt;
    if (!coordinateValue || !countValue || *countValue < 0)
      return invalidInternalEvent(llvm::Twine(context) +
                                  " has an invalid initial count");
    if (!hasOnlyFields(*count, {"coordinate", "value"}))
      return invalidInternalEvent(llvm::Twine(context) +
                                  " initial count has unsupported fields");
    auto coordinate = parseCoordinate(*coordinateValue, context);
    if (!coordinate.ok())
      return coordinate.status();
    if (!coordinateInDomain(*coordinate, event.eventDomain) ||
        !initialByCoordinate.emplace(coordinateKey(*coordinate), *countValue)
             .second)
      return invalidInternalEvent(
          llvm::Twine(context) +
          " initial counts must cover unique Event coordinates");
    event.initialCounts.push_back({std::move(*coordinate), *countValue});
  }
  event.initialCountProvenance = provenance->str();

  std::map<std::string, int64_t> notifyIndegrees;
  for (const GpuInternalEventCoordinate &coordinate :
       domainCoordinates(event.eventDomain))
    notifyIndegrees.emplace(coordinateKey(coordinate), 0);
  for (const GpuInternalEventRelationPair &pair : event.notifyRelation.pairs)
    ++notifyIndegrees[coordinateKey(pair.target)];
  if (initialByCoordinate != notifyIndegrees)
    return invalidInternalEvent(
        llvm::Twine(context) +
        " initial counts do not equal notify-relation indegrees");
  std::map<std::string, std::vector<GpuInternalEventCoordinate>>
      consumersByEvent;
  for (const GpuInternalEventRelationPair &pair : event.triggerRelation.pairs)
    consumersByEvent[coordinateKey(pair.source)].push_back(pair.target);
  std::set<std::string> scheduledDependence;
  for (const GpuInternalEventRelationPair &notify : event.notifyRelation.pairs)
    for (const GpuInternalEventCoordinate &consumer :
         consumersByEvent[coordinateKey(notify.target)])
      scheduledDependence.insert(coordinateKey(notify.source) + ">" +
                                 coordinateKey(consumer));
  for (const GpuInternalEventRelationPair &required :
       event.requiredDependence.pairs)
    if (!scheduledDependence.count(pairKey(required)))
      return invalidInternalEvent(
          llvm::Twine(context) +
          " notify/trigger composition omits a required dependence");
  return event;
}

absl::Status
rememberTask(const GpuInternalEventDomain &domain,
             std::map<std::string, GpuInternalEventDomain> &tasks) {
  if (domain.kind != GpuInternalEventDomainKind::TaskFamily)
    return absl::OkStatus();
  auto [iterator, inserted] = tasks.emplace(domain.name, domain);
  if (!inserted && !sameDomain(iterator->second, domain))
    return invalidInternalEvent(llvm::Twine("task family '") + domain.name +
                                "' has inconsistent definitions");
  return absl::OkStatus();
}

absl::StatusOr<GpuInternalEventAllocation>
parseAllocation(const llvm::json::Object *object, llvm::StringRef context,
                bool requireMechanismKind) {
  if (!object)
    return invalidInternalEvent(llvm::Twine(context) + " must be an object");
  if (!hasOnlyFields(*object,
                     {"allocation_id", "covered_plans", "scope",
                      "release_on_notify", "acquire_before_consumer",
                      "capacity", "allocation_owners", "mechanism_coordinates",
                      "slots", "generations", "generation_count",
                      "notify_relations", "wait_relations", "initial_counts",
                      "transaction_bytes_by_coordinate", "mechanism_kind"}))
    return invalidInternalEvent(llvm::Twine(context) +
                                " has unsupported allocation fields");
  std::optional<llvm::StringRef> id = object->getString("allocation_id");
  const llvm::json::Array *covered = object->getArray("covered_plans");
  std::optional<llvm::StringRef> scope = object->getString("scope");
  std::optional<bool> release = object->getBoolean("release_on_notify");
  std::optional<bool> acquire = object->getBoolean("acquire_before_consumer");
  std::optional<int64_t> capacity = object->getInteger("capacity");
  std::optional<int64_t> generationCount =
      object->getInteger("generation_count");
  const llvm::json::Array *notifyRelations =
      object->getArray("notify_relations");
  const llvm::json::Array *waitRelations = object->getArray("wait_relations");
  if (!id || id->empty() || !covered || !scope || !capacity || *capacity <= 0 ||
      !generationCount || *generationCount <= 0 || !release || !acquire ||
      !notifyRelations || !waitRelations)
    return invalidInternalEvent(
        llvm::Twine(context) +
        " is missing positive bounded-allocation fields");
  GpuInternalEventAllocation allocation;
  allocation.allocationId = id->str();
  auto parsedScope = parseScope(*scope, context);
  auto coordinates =
      parseCoordinates(object->getArray("mechanism_coordinates"), context);
  auto owners =
      parseCoordinates(object->getArray("allocation_owners"), context);
  auto slots = parseIntegers(object->getArray("slots"), context);
  auto generations = parseIntegers(object->getArray("generations"), context);
  if (!parsedScope.ok())
    return parsedScope.status();
  if (!coordinates.ok())
    return coordinates.status();
  if (!owners.ok())
    return owners.status();
  if (!slots.ok())
    return slots.status();
  if (!generations.ok())
    return generations.status();
  allocation.scope = *parsedScope;
  allocation.releaseOnNotify = *release;
  allocation.acquireBeforeConsumer = *acquire;
  allocation.capacity = *capacity;
  allocation.mechanismCoordinates = std::move(*coordinates);
  allocation.allocationOwners = std::move(*owners);
  allocation.slots = std::move(*slots);
  allocation.generations = std::move(*generations);
  allocation.generationCount = *generationCount;
  for (const llvm::json::Value &value : *covered) {
    std::optional<llvm::StringRef> plan = value.getAsString();
    if (!plan || plan->empty())
      return invalidInternalEvent(llvm::Twine(context) +
                                  " covered_plans must contain names");
    allocation.coveredPlans.push_back(plan->str());
  }
  if (allocation.coveredPlans.empty() ||
      std::set<std::string>(allocation.coveredPlans.begin(),
                            allocation.coveredPlans.end())
              .size() != allocation.coveredPlans.size())
    return invalidInternalEvent(llvm::Twine(context) +
                                " must cover unique Event plans");
  for (const llvm::json::Value &value : *notifyRelations) {
    const llvm::json::Object *relation = value.getAsObject();
    std::optional<llvm::StringRef> task =
        relation ? relation->getString("task_family") : std::nullopt;
    const llvm::json::Value *taskCoordinateValue =
        relation ? relation->get("task_coordinate") : nullptr;
    const llvm::json::Value *mechanismCoordinateValue =
        relation ? relation->get("mechanism_coordinate") : nullptr;
    if (!task || task->empty() || !taskCoordinateValue ||
        !mechanismCoordinateValue)
      return invalidInternalEvent(llvm::Twine(context) +
                                  " has a malformed notify relation");
    if (!hasOnlyFields(*relation, {"task_family", "task_coordinate",
                                   "mechanism_coordinate"}))
      return invalidInternalEvent(llvm::Twine(context) +
                                  " notify relation has unsupported fields");
    auto taskCoordinate = parseCoordinate(*taskCoordinateValue, context);
    auto mechanismCoordinate =
        parseCoordinate(*mechanismCoordinateValue, context);
    if (!taskCoordinate.ok())
      return taskCoordinate.status();
    if (!mechanismCoordinate.ok())
      return mechanismCoordinate.status();
    allocation.notifyRelations.push_back({task->str(),
                                          std::move(*taskCoordinate),
                                          std::move(*mechanismCoordinate)});
  }
  for (const llvm::json::Value &value : *waitRelations) {
    const llvm::json::Object *relation = value.getAsObject();
    std::optional<llvm::StringRef> task =
        relation ? relation->getString("task_family") : std::nullopt;
    const llvm::json::Value *taskCoordinateValue =
        relation ? relation->get("task_coordinate") : nullptr;
    const llvm::json::Value *mechanismCoordinateValue =
        relation ? relation->get("mechanism_coordinate") : nullptr;
    if (!task || task->empty() || !taskCoordinateValue ||
        !mechanismCoordinateValue)
      return invalidInternalEvent(llvm::Twine(context) +
                                  " has a malformed wait relation");
    if (!hasOnlyFields(*relation, {"mechanism_coordinate", "task_family",
                                   "task_coordinate"}))
      return invalidInternalEvent(llvm::Twine(context) +
                                  " wait relation has unsupported fields");
    auto taskCoordinate = parseCoordinate(*taskCoordinateValue, context);
    auto mechanismCoordinate =
        parseCoordinate(*mechanismCoordinateValue, context);
    if (!taskCoordinate.ok())
      return taskCoordinate.status();
    if (!mechanismCoordinate.ok())
      return mechanismCoordinate.status();
    allocation.waitRelations.push_back({std::move(*mechanismCoordinate),
                                        task->str(),
                                        std::move(*taskCoordinate)});
  }
  auto initialCounts =
      parseIntegers(object->getArray("initial_counts"), context);
  if (!initialCounts.ok())
    return initialCounts.status();
  allocation.initialCounts = std::move(*initialCounts);
  const size_t size = allocation.mechanismCoordinates.size();
  if (size == 0 || allocation.allocationOwners.size() != size ||
      allocation.slots.size() != size ||
      allocation.generations.size() != size ||
      allocation.initialCounts.size() != size)
    return invalidInternalEvent(
        llvm::Twine(context) +
        " coordinate, owner, slot, and generation arrays must align");
  std::set<std::string> coordinateKeys;
  std::set<std::string> allocationKeys;
  const size_t mechanismRank = allocation.mechanismCoordinates.front().size();
  const size_t ownerRank = allocation.allocationOwners.front().size();
  for (size_t index = 0; index < size; ++index) {
    if (allocation.mechanismCoordinates[index].size() != mechanismRank ||
        allocation.allocationOwners[index].size() != ownerRank ||
        !coordinateKeys
             .insert(coordinateKey(allocation.mechanismCoordinates[index]))
             .second ||
        allocation.slots[index] >= allocation.capacity ||
        allocation.generations[index] >= allocation.generationCount)
      return invalidInternalEvent(
          llvm::Twine(context) +
          " has duplicate coordinates or out-of-bounds slots/generations");
    std::string allocationKey =
        coordinateKey(allocation.allocationOwners[index]) + ":" +
        std::to_string(allocation.slots[index]) + ":" +
        std::to_string(allocation.generations[index]);
    if (!allocationKeys.insert(std::move(allocationKey)).second)
      return invalidInternalEvent(llvm::Twine(context) +
                                  " reuses an owner/slot/generation tuple");
  }
  std::set<std::string> notifyKeys;
  for (const GpuInternalEventAllocation::NotifyRelation &relation :
       allocation.notifyRelations) {
    if (!coordinateKeys.count(coordinateKey(relation.mechanismCoordinate)) ||
        !notifyKeys
             .insert(relation.taskFamily + ":" +
                     coordinateKey(relation.taskCoordinate) + ">" +
                     coordinateKey(relation.mechanismCoordinate))
             .second)
      return invalidInternalEvent(
          llvm::Twine(context) +
          " notify relation is duplicate or outside the mechanism domain");
  }
  std::set<std::string> waitKeys;
  for (const GpuInternalEventAllocation::WaitRelation &relation :
       allocation.waitRelations) {
    if (!coordinateKeys.count(coordinateKey(relation.mechanismCoordinate)) ||
        !waitKeys
             .insert(coordinateKey(relation.mechanismCoordinate) + ">" +
                     relation.taskFamily + ":" +
                     coordinateKey(relation.taskCoordinate))
             .second)
      return invalidInternalEvent(
          llvm::Twine(context) +
          " wait relation is duplicate or outside the mechanism domain");
  }
  const llvm::json::Value *transactions =
      object->get("transaction_bytes_by_coordinate");
  if (transactions && transactions->kind() != llvm::json::Value::Null) {
    auto parsed = parseIntegers(transactions->getAsArray(), context);
    if (!parsed.ok() || parsed->size() != size ||
        llvm::any_of(*parsed, [](int64_t value) { return value <= 0; }))
      return invalidInternalEvent(
          llvm::Twine(context) +
          " transaction bytes must align and be positive");
    allocation.transactionBytesByCoordinate = std::move(*parsed);
  }
  std::optional<llvm::StringRef> mechanismKind =
      object->getString("mechanism_kind");
  if (requireMechanismKind && !mechanismKind)
    return invalidInternalEvent(
        llvm::Twine(context) + " requires a typed mechanism_kind");
  if (!mechanismKind) {
    // Legacy records are accepted only when their structural facts select one
    // primitive uniquely. A CTA wait/notify without transaction completion
    // could be either an mbarrier or an async-completion queue and must be
    // regenerated with an explicit mechanism kind.
    if (allocation.scope == GpuEventPhysicalScope::ThreadBlock &&
        allocation.transactionBytesByCoordinate)
      allocation.legalizationKind = GpuInternalEventAllocation::
          LegalizationKind::CtaTransactionMbarrier;
    else if (allocation.scope == GpuEventPhysicalScope::Device &&
             !allocation.transactionBytesByCoordinate)
      allocation.legalizationKind =
          GpuInternalEventAllocation::LegalizationKind::DeviceSemaphoreQueue;
    else
      return invalidInternalEvent(
          llvm::Twine(context) +
          " requires an explicit mechanism_kind because its structural facts "
          "do not select one internal Event primitive");
    return allocation;
  }
  if (*mechanismKind == "cta_mbarrier")
    allocation.legalizationKind =
        GpuInternalEventAllocation::LegalizationKind::CtaMbarrier;
  else if (*mechanismKind == "cta_transaction_mbarrier")
    allocation.legalizationKind = GpuInternalEventAllocation::
        LegalizationKind::CtaTransactionMbarrier;
  else if (*mechanismKind == "cta_async_completion_queue")
    allocation.legalizationKind = GpuInternalEventAllocation::
        LegalizationKind::CtaAsyncCompletionQueue;
  else if (*mechanismKind == "device_semaphore_queue")
    allocation.legalizationKind =
        GpuInternalEventAllocation::LegalizationKind::DeviceSemaphoreQueue;
  else
    return invalidInternalEvent(llvm::Twine(context) +
                                " has an unsupported mechanism_kind");
  const bool cta = allocation.scope == GpuEventPhysicalScope::ThreadBlock;
  const bool device = allocation.scope == GpuEventPhysicalScope::Device;
  const bool transaction = allocation.transactionBytesByCoordinate.has_value();
  const bool valid =
      (allocation.legalizationKind ==
           GpuInternalEventAllocation::LegalizationKind::CtaMbarrier &&
       cta && !transaction) ||
      (allocation.legalizationKind == GpuInternalEventAllocation::
                                           LegalizationKind::
                                               CtaTransactionMbarrier &&
       cta && transaction) ||
      (allocation.legalizationKind == GpuInternalEventAllocation::
                                           LegalizationKind::
                                               CtaAsyncCompletionQueue &&
       cta && !transaction) ||
      (allocation.legalizationKind == GpuInternalEventAllocation::
                                           LegalizationKind::
                                               DeviceSemaphoreQueue &&
       device && !transaction);
  if (!valid)
    return invalidInternalEvent(
        llvm::Twine(context) +
        " mechanism_kind does not match scope and transaction mode");
  return allocation;
}

absl::StatusOr<GpuInternalEventRegion>
parseRegion(const llvm::json::Object *object, int64_t regionOrdinal) {
  const std::string context = "region[" + std::to_string(regionOrdinal) + "]";
  if (!object)
    return invalidInternalEvent(context + " must be an object");
  std::optional<int64_t> schemaVersion = object->getInteger("schema_version");
  std::optional<llvm::StringRef> fingerprint = object->getString("fingerprint");
  const llvm::json::Array *realizations = object->getArray("realizations");
  const llvm::json::Array *allocations = object->getArray("allocations");
  llvm::SmallVector<llvm::StringRef> allowedFields{
      "schema_version", "fingerprint", "realizations", "allocations"};
  if (schemaVersion && (*schemaVersion == 2 || *schemaVersion == 7))
    allowedFields.append({"name", "task_axes", "predecessor_pairs"});
  if (schemaVersion && (*schemaVersion == 4 || *schemaVersion == 6))
    allowedFields.push_back("partition_count");
  if (!hasOnlyFields(*object, allowedFields))
    return invalidInternalEvent(context + " has unsupported schedule fields");
  if (!schemaVersion ||
      (*schemaVersion != 2 && *schemaVersion != 3 && *schemaVersion != 4 &&
       *schemaVersion != 5 && *schemaVersion != 6 && *schemaVersion != 7) ||
      (fingerprint && !isDigest(*fingerprint)) ||
      (*schemaVersion != 2 && *schemaVersion != 7 && !fingerprint) ||
      !realizations || !allocations)
    return invalidInternalEvent(
        context +
        " requires a supported schema, digest, realizations, and allocations");
  GpuInternalEventRegion region;
  region.schemaVersion = *schemaVersion;
  if (fingerprint)
    region.fingerprint = fingerprint->str();
  std::map<std::string, GpuInternalEventDomain> tasks;
  std::map<std::string, size_t> events;
  for (const llvm::json::Value &value : *realizations) {
    const llvm::json::Object *realizationObject = value.getAsObject();
    std::optional<llvm::StringRef> plan =
        realizationObject ? realizationObject->getString("plan") : std::nullopt;
    std::optional<llvm::StringRef> kind =
        realizationObject ? realizationObject->getString("kind") : std::nullopt;
    std::optional<llvm::StringRef> mechanism =
        realizationObject ? realizationObject->getString("mechanism")
                          : std::nullopt;
    if (realizationObject &&
        !hasOnlyFields(*realizationObject, {"plan", "event", "kind",
                                            "mechanism", "allocation_ids"}))
      return invalidInternalEvent(context +
                                  " realization has unsupported fields");
    auto event = parseEvent(
        realizationObject ? realizationObject->getObject("event") : nullptr,
        context + ".realizations.event");
    if (!plan || !kind || !mechanism || mechanism->empty())
      return invalidInternalEvent(
          context + " realization requires plan, event, kind, and mechanism");
    if (!event.ok())
      return event.status();
    if (*plan != event->name ||
        !events.emplace(event->name, events.size()).second)
      return invalidInternalEvent(
          context + " realization plan must uniquely equal event.name");
    GpuInternalEventRealization realization;
    realization.plan = plan->str();
    realization.event = *event;
    realization.mechanism = mechanism->str();
    if (*kind == "erased_program_order") {
      realization.kind = GpuInternalEventRealizationKind::ErasedProgramOrder;
    } else if (*kind == "erased_stream_order") {
      realization.kind = GpuInternalEventRealizationKind::ErasedStreamOrder;
    } else if (*kind == "erased_algebraic_refinement") {
      realization.kind =
          GpuInternalEventRealizationKind::ErasedAlgebraicRefinement;
    } else if (*kind == "physical") {
      realization.kind = GpuInternalEventRealizationKind::Physical;
      const llvm::json::Array *allocations =
          realizationObject->getArray("allocation_ids");
      if (!allocations || allocations->empty())
        return invalidInternalEvent(
            context + " physical realization requires allocation_ids");
      std::set<std::string> uniqueAllocations;
      for (const llvm::json::Value &value : *allocations) {
        std::optional<llvm::StringRef> allocation = value.getAsString();
        if (!allocation || allocation->empty() ||
            !uniqueAllocations.insert(allocation->str()).second)
          return invalidInternalEvent(
              context + " realization allocation_ids must be unique names");
        realization.allocationIds.push_back(allocation->str());
      }
    } else {
      return invalidInternalEvent(context +
                                  " has unsupported realization kind");
    }
    if (realization.kind != GpuInternalEventRealizationKind::Physical &&
        (!realizationObject->getArray("allocation_ids") ||
         !realizationObject->getArray("allocation_ids")->empty()))
      return invalidInternalEvent(
          context + " erased realization cannot have an allocation");
    for (const GpuInternalEventDomain *domain :
         {&event->requiredDependence.source,
          &event->requiredDependence.target}) {
      absl::Status status = rememberTask(*domain, tasks);
      if (!status.ok())
        return status;
    }
    region.events.push_back(*event);
    region.realizations.push_back(std::move(realization));
  }

  std::map<std::string, size_t> allocationIds;
  std::map<std::string, std::set<std::string>> allocationsForPlan;
  for (const llvm::json::Value &value : *allocations) {
    auto allocation =
        parseAllocation(value.getAsObject(), context + ".allocations",
                        *schemaVersion >= 5);
    if (!allocation.ok())
      return allocation.status();
    if (!allocationIds.emplace(allocation->allocationId, allocationIds.size())
             .second)
      return invalidInternalEvent(context + " has duplicate allocation IDs");
    for (const std::string &plan : allocation->coveredPlans) {
      auto event = events.find(plan);
      if (event == events.end() ||
          region.realizations[event->second].kind !=
              GpuInternalEventRealizationKind::Physical ||
          !llvm::is_contained(region.realizations[event->second].allocationIds,
                              allocation->allocationId) ||
          region.events[event->second].memoryScope != allocation->scope ||
          region.events[event->second].visibility.releaseOnNotify !=
              allocation->releaseOnNotify ||
          region.events[event->second].visibility.acquireBeforeConsumer !=
              allocation->acquireBeforeConsumer ||
          !allocationsForPlan[plan].insert(allocation->allocationId).second)
        return invalidInternalEvent(
            context + " allocation join does not match physical realizations");
      const GpuInternalEventTensorPlan &coveredEvent =
          region.events[event->second];
      std::set<std::string> composed;
      for (const GpuInternalEventAllocation::NotifyRelation &notify :
           allocation->notifyRelations) {
        if (notify.taskFamily != coveredEvent.requiredDependence.source.name)
          continue;
        for (const GpuInternalEventAllocation::WaitRelation &wait :
             allocation->waitRelations)
          if (wait.taskFamily == coveredEvent.requiredDependence.target.name &&
              wait.mechanismCoordinate == notify.mechanismCoordinate)
            composed.insert(coordinateKey(notify.taskCoordinate) + ">" +
                            coordinateKey(wait.taskCoordinate));
      }
      for (const GpuInternalEventRelationPair &pair :
           coveredEvent.requiredDependence.pairs)
        if (!composed.count(pairKey(pair)))
          return invalidInternalEvent(context +
                                      " mechanism notify/wait composition does "
                                      "not cover a required dependence");
    }
    std::map<std::string, int64_t> notifyIndegrees;
    for (const GpuInternalEventCoordinate &coordinate :
         allocation->mechanismCoordinates)
      notifyIndegrees.emplace(coordinateKey(coordinate), 0);
    for (const GpuInternalEventAllocation::NotifyRelation &notify :
         allocation->notifyRelations) {
      auto task = tasks.find(notify.taskFamily);
      if (task == tasks.end() ||
          !coordinateInDomain(notify.taskCoordinate, task->second))
        return invalidInternalEvent(
            context +
            " mechanism notify references an unknown task coordinate");
      ++notifyIndegrees[coordinateKey(notify.mechanismCoordinate)];
    }
    for (const GpuInternalEventAllocation::WaitRelation &wait :
         allocation->waitRelations) {
      auto task = tasks.find(wait.taskFamily);
      if (task == tasks.end() ||
          !coordinateInDomain(wait.taskCoordinate, task->second))
        return invalidInternalEvent(
            context + " mechanism wait references an unknown task coordinate");
    }
    for (auto [index, coordinate] :
         llvm::enumerate(allocation->mechanismCoordinates))
      if (allocation->initialCounts[index] !=
          notifyIndegrees[coordinateKey(coordinate)])
        return invalidInternalEvent(
            context +
            " mechanism initial counts do not equal notify-relation indegrees");
    region.allocations.push_back(std::move(*allocation));
  }
  for (const GpuInternalEventRealization &realization : region.realizations) {
    if (realization.kind == GpuInternalEventRealizationKind::Physical &&
        allocationsForPlan[realization.plan] !=
            std::set<std::string>(realization.allocationIds.begin(),
                                  realization.allocationIds.end()))
      return invalidInternalEvent(
          context + " physical realization has no exact allocation join");
  }
  for (auto &[name, task] : tasks)
    region.taskFamilies.push_back(std::move(task));
  return region;
}

absl::Status flattenRegions(const llvm::json::Object *schedule,
                            std::vector<GpuInternalEventRegion> &regions) {
  if (!schedule)
    return invalidInternalEvent("schedule must be an object");
  if (const llvm::json::Array *nested = schedule->getArray("regions")) {
    if (!hasOnlyFields(*schedule, {"schema_version", "fingerprint", "regions"}))
      return invalidInternalEvent("combined schedule has unsupported fields");
    std::optional<int64_t> schemaVersion =
        schedule->getInteger("schema_version");
    std::optional<llvm::StringRef> fingerprint =
        schedule->getString("fingerprint");
    if (!schemaVersion || *schemaVersion != 1 || !fingerprint ||
        !isDigest(*fingerprint) || nested->empty())
      return invalidInternalEvent(
          "combined schedule requires schema 1, digest, and regions");
    for (const llvm::json::Value &value : *nested) {
      absl::Status status = flattenRegions(value.getAsObject(), regions);
      if (!status.ok())
        return status;
    }
    return absl::OkStatus();
  }
  auto region = parseRegion(schedule, regions.size());
  if (!region.ok())
    return region.status();
  regions.push_back(std::move(*region));
  return absl::OkStatus();
}

} // namespace

absl::StatusOr<GpuInternalEventSchedule>
parseGpuInternalEventSchedule(llvm::StringRef canonicalProvenance) {
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse(canonicalProvenance);
  if (!parsed)
    return invalidInternalEvent("canonical provenance is not JSON");
  llvm::json::Object *envelope = parsed->getAsObject();
  if (!envelope)
    return invalidInternalEvent("canonical provenance must be an object");
  if (!hasOnlyFields(*envelope,
                     {"schema_version", "physical_schedule_fingerprint",
                      "schedule", "fingerprint"}))
    return invalidInternalEvent("canonical provenance has unsupported fields");
  std::optional<int64_t> schemaVersion = envelope->getInteger("schema_version");
  std::optional<llvm::StringRef> physicalFingerprint =
      envelope->getString("physical_schedule_fingerprint");
  std::optional<llvm::StringRef> fingerprint =
      envelope->getString("fingerprint");
  llvm::json::Object *schedule = envelope->getObject("schedule");
  if (!schemaVersion || *schemaVersion != 1 || !physicalFingerprint ||
      !isDigest(*physicalFingerprint) || !fingerprint ||
      !isDigest(*fingerprint) || !schedule)
    return invalidInternalEvent(
        "envelope requires schema 1, schedule fingerprints, and schedule");
  std::optional<llvm::StringRef> nestedFingerprint =
      schedule->getString("fingerprint");
  if (!nestedFingerprint || *nestedFingerprint != *physicalFingerprint)
    return invalidInternalEvent("envelope physical_schedule_fingerprint must "
                                "equal schedule fingerprint");
  GpuInternalEventSchedule result;
  result.schemaVersion = *schemaVersion;
  result.physicalScheduleFingerprint = physicalFingerprint->str();
  result.fingerprint = fingerprint->str();
  absl::Status status = flattenRegions(schedule, result.regions);
  if (!status.ok())
    return status;
  return result;
}

} // namespace mlir::shuttle
