// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/PythonPhysicalArtifactBridge.h"

#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Python.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "shuttle/Transforms/PhysicalArtifact.h"

namespace nb = nanobind;

namespace mlir::shuttle {
namespace {

constexpr Py_ssize_t kMaximumCoordinateRank = 16;
constexpr Py_ssize_t kMaximumNameBytes = 1024;

absl::Status invalidLegalization(const std::string &detail) {
  return absl::InvalidArgumentError(
      "invalid Shuttle internal Event legalization: " + detail);
}

bool hasExactKeys(PyObject *mapping, std::initializer_list<const char *> keys) {
  if (!PyDict_CheckExact(mapping) ||
      PyDict_Size(mapping) != static_cast<Py_ssize_t>(keys.size()))
    return false;
  for (const char *key : keys)
    if (PyDict_GetItemString(mapping, key) == nullptr)
      return false;
  return true;
}

PyObject *exactSequenceItem(PyObject *sequence, Py_ssize_t index) {
  if (PyList_CheckExact(sequence))
    return PyList_GET_ITEM(sequence, index);
  return PyTuple_GET_ITEM(sequence, index);
}

std::optional<int64_t> exactInteger(PyObject *value, bool positive) {
  if (!PyLong_CheckExact(value))
    return std::nullopt;
  long long parsed = PyLong_AsLongLong(value);
  if (parsed == -1 && PyErr_Occurred()) {
    PyErr_Clear();
    return std::nullopt;
  }
  if (positive ? parsed <= 0 : parsed < 0)
    return std::nullopt;
  return static_cast<int64_t>(parsed);
}

std::optional<std::string> exactName(PyObject *value) {
  if (!PyUnicode_CheckExact(value))
    return std::nullopt;
  Py_ssize_t size = 0;
  const char *data = PyUnicode_AsUTF8AndSize(value, &size);
  if (data == nullptr) {
    PyErr_Clear();
    return std::nullopt;
  }
  if (size <= 0 || size > kMaximumNameBytes ||
      std::string_view(data, size).find('\0') != std::string_view::npos)
    return std::nullopt;
  return std::string(data, size);
}

std::optional<GpuInternalEventCoordinate> exactCoordinate(PyObject *value) {
  if (!PyList_CheckExact(value) && !PyTuple_CheckExact(value))
    return std::nullopt;
  const Py_ssize_t size = PySequence_Size(value);
  if (size < 0 || size > kMaximumCoordinateRank)
    return std::nullopt;
  GpuInternalEventCoordinate coordinate;
  coordinate.reserve(size);
  for (Py_ssize_t index = 0; index < size; ++index) {
    PyObject *item = exactSequenceItem(value, index);
    std::optional<int64_t> parsed = exactInteger(item, false);
    if (!parsed)
      return std::nullopt;
    coordinate.push_back(*parsed);
  }
  return coordinate;
}

std::optional<GpuInternalEventOperationKind> operationKind(PyObject *value) {
  std::optional<std::string> name = exactName(value);
  if (!name)
    return std::nullopt;
  if (*name == "notify")
    return GpuInternalEventOperationKind::Notify;
  if (*name == "wait")
    return GpuInternalEventOperationKind::Wait;
  return std::nullopt;
}

std::optional<GpuInternalEventAllocation::LegalizationKind>
mechanismKind(PyObject *value) {
  std::optional<std::string> name = exactName(value);
  if (!name)
    return std::nullopt;
  if (*name == "cta_mbarrier")
    return GpuInternalEventAllocation::LegalizationKind::CtaMbarrier;
  if (*name == "cta_transaction_mbarrier")
    return GpuInternalEventAllocation::LegalizationKind::CtaTransactionMbarrier;
  if (*name == "cta_async_completion_queue")
    return GpuInternalEventAllocation::LegalizationKind::
        CtaAsyncCompletionQueue;
  if (*name == "device_semaphore_queue")
    return GpuInternalEventAllocation::LegalizationKind::DeviceSemaphoreQueue;
  return std::nullopt;
}

std::optional<GpuEventPhysicalScope> physicalScope(PyObject *value) {
  std::optional<std::string> name = exactName(value);
  if (!name)
    return std::nullopt;
  if (*name == "cta")
    return GpuEventPhysicalScope::ThreadBlock;
  if (*name == "cluster")
    return GpuEventPhysicalScope::Cluster;
  if (*name == "device")
    return GpuEventPhysicalScope::Device;
  if (*name == "system")
    return GpuEventPhysicalScope::System;
  return std::nullopt;
}

absl::StatusOr<GpuInternalEventOperation>
parseInternalEventOperation(PyObject *mapping, int64_t expectedOrdinal) {
  const std::initializer_list<const char *> requiredKeys = {
      "ordinal",          "kind",
      "relation_ordinal", "task_family",
      "task_coordinate",  "mechanism_coordinate",
      "mechanism_kind",   "scope",
      "allocation_owner", "slot",
      "generation",       "generation_count",
      "operation_count",  "initial_count",
      "release_acquire"};
  if (!PyDict_CheckExact(mapping))
    return invalidLegalization("operation fields");
  const Py_ssize_t fieldCount = PyDict_Size(mapping);
  if (fieldCount != static_cast<Py_ssize_t>(requiredKeys.size()) &&
      fieldCount != static_cast<Py_ssize_t>(requiredKeys.size() + 1))
    return invalidLegalization("operation fields");
  for (const char *key : requiredKeys)
    if (PyDict_GetItemString(mapping, key) == nullptr)
      return invalidLegalization("operation fields");
  PyObject *transaction = PyDict_GetItemString(mapping, "transaction_bytes");
  if (fieldCount == static_cast<Py_ssize_t>(requiredKeys.size() + 1) &&
      transaction == nullptr)
    return invalidLegalization("operation fields");
  auto field = [&](const char *name) {
    return PyDict_GetItemString(mapping, name);
  };
  std::optional<int64_t> ordinal = exactInteger(field("ordinal"), false);
  std::optional<GpuInternalEventOperationKind> kind =
      operationKind(field("kind"));
  std::optional<int64_t> relationOrdinal =
      exactInteger(field("relation_ordinal"), false);
  std::optional<std::string> taskFamily = exactName(field("task_family"));
  std::optional<GpuInternalEventCoordinate> taskCoordinate =
      exactCoordinate(field("task_coordinate"));
  std::optional<GpuInternalEventCoordinate> mechanismCoordinate =
      exactCoordinate(field("mechanism_coordinate"));
  std::optional<GpuInternalEventAllocation::LegalizationKind> mechanism =
      mechanismKind(field("mechanism_kind"));
  std::optional<GpuEventPhysicalScope> scope = physicalScope(field("scope"));
  std::optional<GpuInternalEventCoordinate> allocationOwner =
      exactCoordinate(field("allocation_owner"));
  std::optional<int64_t> slot = exactInteger(field("slot"), false);
  std::optional<int64_t> generation = exactInteger(field("generation"), false);
  std::optional<int64_t> generationCount =
      exactInteger(field("generation_count"), true);
  std::optional<int64_t> operationCount =
      exactInteger(field("operation_count"), false);
  std::optional<int64_t> initialCount =
      exactInteger(field("initial_count"), false);
  PyObject *releaseAcquire = field("release_acquire");
  std::optional<int64_t> transactionBytes;
  if (transaction != nullptr && transaction != Py_None) {
    transactionBytes = exactInteger(transaction, true);
    if (!transactionBytes)
      return invalidLegalization("transaction_bytes");
  }
  if (!ordinal || *ordinal != expectedOrdinal || !kind || !relationOrdinal ||
      !taskFamily || !taskCoordinate || !mechanismCoordinate || !mechanism ||
      !scope || !allocationOwner || !slot || !generation || !generationCount ||
      !operationCount || !initialCount ||
      (releaseAcquire != Py_True && releaseAcquire != Py_False))
    return invalidLegalization("operation value");
  GpuInternalEventOperation operation;
  operation.ordinal = *ordinal;
  operation.kind = *kind;
  operation.relationOrdinal = *relationOrdinal;
  operation.taskFamily = std::move(*taskFamily);
  operation.taskCoordinate = std::move(*taskCoordinate);
  operation.mechanismCoordinate = std::move(*mechanismCoordinate);
  operation.legalizationKind = *mechanism;
  operation.scope = *scope;
  operation.allocationOwner = std::move(*allocationOwner);
  operation.slot = *slot;
  operation.generation = *generation;
  operation.generationCount = *generationCount;
  operation.operationCount = *operationCount;
  operation.initialCount = *initialCount;
  operation.releaseAcquire = releaseAcquire == Py_True;
  operation.transactionBytes = transactionBytes;
  return operation;
}

absl::StatusOr<std::vector<GpuInternalEventLegalization>>
parseInternalEventLegalizations(PyObject *value) {
  if (!PyList_CheckExact(value) && !PyTuple_CheckExact(value))
    return invalidLegalization("top-level sequence");
  const Py_ssize_t count = PySequence_Size(value);
  if (count < 0 || count > kMaximumGpuExecutableRecords)
    return invalidLegalization("top-level count");
  std::vector<GpuInternalEventLegalization> legalizations;
  legalizations.reserve(count);
  for (Py_ssize_t index = 0; index < count; ++index) {
    PyObject *mapping = exactSequenceItem(value, index);
    if (!hasExactKeys(mapping,
                      {"region_ordinal", "allocation_id", "operations"}))
      return invalidLegalization("allocation fields");
    std::optional<int64_t> regionOrdinal =
        exactInteger(PyDict_GetItemString(mapping, "region_ordinal"), false);
    std::optional<std::string> allocationId =
        exactName(PyDict_GetItemString(mapping, "allocation_id"));
    PyObject *operations = PyDict_GetItemString(mapping, "operations");
    if (!regionOrdinal || !allocationId ||
        (!PyList_CheckExact(operations) && !PyTuple_CheckExact(operations)))
      return invalidLegalization("allocation value");
    const Py_ssize_t operationCount = PySequence_Size(operations);
    if (operationCount <= 0 || operationCount > kMaximumGpuExecutableRecords)
      return invalidLegalization("operation count");
    GpuInternalEventLegalization legalization;
    legalization.regionOrdinal = *regionOrdinal;
    legalization.allocationId = std::move(*allocationId);
    legalization.operations.reserve(operationCount);
    for (Py_ssize_t operationOrdinal = 0; operationOrdinal < operationCount;
         ++operationOrdinal) {
      absl::StatusOr<GpuInternalEventOperation> operation =
          parseInternalEventOperation(
              exactSequenceItem(operations, operationOrdinal),
              operationOrdinal);
      if (!operation.ok())
        return operation.status();
      legalization.operations.push_back(std::move(*operation));
    }
    legalizations.push_back(std::move(legalization));
  }
  return legalizations;
}

class PythonPhysicalArtifactProvider final
    : public GpuPhysicalArtifactProvider {
public:
  explicit PythonPhysicalArtifactProvider(nb::callable compiler)
      // The provider is process-lifetime state, matching nanobind's module.
      // Intentionally retain the callable so its destructor never runs after
      // Python finalization without the GIL.
      : compiler(new nb::callable(std::move(compiler))) {}

  absl::StatusOr<std::optional<GpuPhysicalArtifactCompilation>>
  compile(const GpuPhysicalArtifactRequest &request) const final {
    try {
      nb::gil_scoped_acquire acquire;
      std::string requestJson = gpuPhysicalArtifactRequestJson(request);
      nb::object result =
          (*compiler)(nb::str(requestJson.data(), requestJson.size()));
      if (result.is_none())
        return std::nullopt;
      PyObject *transportObject = result.ptr();
      std::vector<GpuInternalEventLegalization> internalEventLegalizations;
      if (PyTuple_CheckExact(result.ptr()) &&
          PyTuple_GET_SIZE(result.ptr()) == 2) {
        transportObject = PyTuple_GET_ITEM(result.ptr(), 0);
        absl::StatusOr<std::vector<GpuInternalEventLegalization>> parsed =
            parseInternalEventLegalizations(PyTuple_GET_ITEM(result.ptr(), 1));
        if (!parsed.ok())
          return parsed.status();
        internalEventLegalizations = std::move(*parsed);
      }
      if (!PyBytes_CheckExact(transportObject))
        return absl::InvalidArgumentError(
            "Shuttle physical compiler must return bytes, "
            "(bytes, internal_event_legalizations), or None");
      char *data = nullptr;
      Py_ssize_t size = 0;
      if (PyBytes_AsStringAndSize(transportObject, &data, &size) != 0)
        throw nb::python_error();
      std::vector<uint8_t> transport(reinterpret_cast<const uint8_t *>(data),
                                     reinterpret_cast<const uint8_t *>(data) +
                                         size);
      RegionOp region = request.region;
      absl::StatusOr<GpuFatbinExecutableBundle> bundle =
          deserializeGpuFatbinExecutableBundle(transport, region.getContext());
      if (!bundle.ok())
        return bundle.status();
      const int64_t expectedSchema = request.target == GpuTarget::Sm90 ? 5 : 6;
      if (bundle->deviceSchemaVersion != expectedSchema)
        return absl::InvalidArgumentError(
            "Shuttle physical compiler returned an artifact for the wrong GPU "
            "target");
      // Opaque bytes cannot establish compiler-owned Event legalizations. The
      // optional typed tuple channel is deliberately outside transport and is
      // accepted only from this in-process provider boundary.
      return std::optional<GpuPhysicalArtifactCompilation>(
          GpuPhysicalArtifactCompilation{
              std::move(*bundle), {}, std::move(internalEventLegalizations)});
    } catch (const nb::python_error &error) {
      return absl::InternalError(
          std::string(
              "Shuttle Python physical compiler raised an exception: ") +
          error.what());
    }
  }

private:
  nb::callable *compiler;
};

void registerCompiler(nb::callable compiler) {
  absl::Status status = registerGpuPhysicalArtifactProvider(
      std::make_shared<PythonPhysicalArtifactProvider>(std::move(compiler)));
  if (!status.ok()) {
    std::string message(status.message());
    throw nb::value_error(message.c_str());
  }
}

} // namespace

void registerShuttlePhysicalArtifactBindings(nb::module_ &module) {
  nb::module_ bridge = module.def_submodule(
      "_shuttle_physical_artifact",
      "Compile-time bridge for source-bound Shuttle GPU artifacts.");
  bridge.def("register", &registerCompiler, nb::arg("compiler"));
}

} // namespace mlir::shuttle
