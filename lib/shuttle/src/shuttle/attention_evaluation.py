# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Closed evidence contract for the representative attention evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

SCHEMA_VERSION = 1
ATTENTION_SHAPE_ID = "dot_product_attention_bf16_b1_s2048_hq32_hkv8_dqk128_dv128_causal"
ATTENTION_SHAPE_SPECIFICATION = (
    "q:bf16[1,8,4,2048,128], k:bf16[1,8,2048,128], v:bf16[1,8,2048,128], "
    "do:bf16[1,8,4,2048,128], o:bf16[1,8,4,2048,128], "
    "dq:bf16[1,8,4,2048,128], dk:bf16[1,8,2048,128], dv:bf16[1,8,2048,128], "
    "scale:f32=1/sqrt(128), causal=true, layout:row_major_grouped_gqa"
)

_IMPLEMENTATIONS = ("ordinary_xla", "fa4_oracle", "shuttle_source_ordered", "shuttle_fast")
_POLICIES = (None, None, "source_ordered", "fast")
_BOUNDARY_OUTPUTS = {
    "forward": ("output",),
    "backward": ("query_cotangent", "key_cotangent", "value_cotangent"),
    "composed_forward_backward": ("output", "query_cotangent", "key_cotangent", "value_cotangent"),
}
_METRICS = (
    "maximum_absolute_error",
    "mean_absolute_error",
    "maximum_relative_error",
    "maximum_ulp_error",
)
_COMPILE_TIMINGS = (
    "trace_ns",
    "lowering_ns",
    "compilation_ns",
    "tuning_ns",
    "warmup_ns",
    "cold_cache_compile_ns",
    "warm_cache_compile_ns",
)
_PROVENANCE_FIELDS = (
    "source_commit",
    "source_tree",
    "source_dirty",
    "source_diff_sha256",
    "shuttle_source_sha256",
    "python_version",
    "jax_version",
    "jaxlib_version",
    "xla_commit",
    "cutlass_dsl_version",
    "tvm_ffi_version",
    "cuda_toolkit_version",
    "cuda_driver_version",
    "nvcc_version",
    "ptxas_version",
    "ncu_version",
    "nsys_version",
    "device_name",
    "device_uuid",
    "compute_capability",
    "oracle_package",
    "oracle_version",
    "oracle_wheel_sha256",
    "oracle_sdist_sha256",
    "oracle_backend_source_sha256",
    "oracle_config_source_sha256",
)
_PAYLOADS = (
    "stablehlo",
    "transformed_mlir",
    "physical_request",
    "generated_source",
    "fatbin",
    "ptx",
    "sass",
    "resource_report",
    "profiler_report",
)
_PINNED_ORACLE = {
    "oracle_package": "flash-attn-4",
    "oracle_version": "4.0.0b19",
    "oracle_wheel_sha256": "bc3856e018fa32e2b833726566641e51d02d45f2d444849937276a46c01653a2",
    "oracle_sdist_sha256": "c904c9da0387c2ac0420cffdf61d6201d94cfeb354b6768fde794b04e9fc4e89",
    "oracle_backend_source_sha256": "66a956ea9505f384105845140837824a8062724547b87c0a9bb2c92d663fd37c",
    "oracle_config_source_sha256": "d99bb01e74ac5f68a3551f70f8f17ffa92f2b84fed6061bc423c0e1e25ec52f2",
}


class AttentionEvaluationError(ValueError):
    """The attention evidence violates its closed evaluation contract."""


@dataclass(frozen=True)
class AttentionEvaluationRecord:
    """Validated attention evidence and its scorecard coordinate."""

    record_id: str
    status: str
    boundary: str
    hardware: str
    missing_fields: tuple[str, ...]
    source_commit: str

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def load_attention_evaluation(path: Path) -> AttentionEvaluationRecord:
    """Load and validate one content-addressed attention evidence record."""
    document = json.loads(path.read_text(), object_pairs_hook=_unique_json_object)
    return validate_attention_evaluation(document, artifact_root=path.parent)


def write_attention_evaluation(
    document: Mapping[str, object],
    path: Path,
    *,
    payloads: Mapping[str, bytes],
) -> AttentionEvaluationRecord:
    """Content-address payloads, validate the record, and write canonical JSON."""
    if set(payloads) != set(_PAYLOADS):
        raise AttentionEvaluationError(f"payload contents must cover exactly {list(_PAYLOADS)}")
    encoded = deepcopy(dict(document))
    payload_records: dict[str, dict[str, object]] = {}
    payload_directory = path.parent / f"{path.stem}.payloads"
    payload_directory.mkdir(parents=True, exist_ok=True)
    for kind in _PAYLOADS:
        content = payloads[kind]
        if not isinstance(content, bytes):
            raise AttentionEvaluationError(f"payload {kind!r} must be bytes")
        digest = hashlib.sha256(content).hexdigest()
        payload_path = payload_directory / f"{digest}.{kind}"
        payload_path.write_bytes(content)
        payload_records[kind] = {
            "path": payload_path.relative_to(path.parent).as_posix(),
            "sha256": digest,
            "bytes": len(content),
        }
    encoded["payloads"] = payload_records
    record = validate_attention_evaluation(encoded, artifact_root=path.parent)
    path.write_text(json.dumps(encoded, indent=2, sort_keys=True) + "\n")
    return record


def validate_attention_evaluation(document: object, *, artifact_root: Path) -> AttentionEvaluationRecord:
    """Validate the complete or explicitly partial attention evidence schema."""
    root = _closed_mapping(
        document,
        "record",
        {
            "schema_version",
            "record_id",
            "status",
            "missing_fields",
            "coordinate",
            "boundary_contract",
            "numerical_contract",
            "cells",
            "timing",
            "accuracy",
            "repeatability",
            "provenance",
            "payloads",
        },
    )
    if _integer(root["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise AttentionEvaluationError(f"attention evaluation schema_version must be {SCHEMA_VERSION}")
    record_id = _identifier(root["record_id"], "record_id")
    status = _choice(root["status"], "status", {"partial", "accepted"})
    declared_missing = tuple(_unique_strings(root["missing_fields"], "missing_fields"))
    if tuple(sorted(declared_missing)) != declared_missing:
        raise AttentionEvaluationError("missing_fields must be sorted")

    coordinate = _coordinate(root["coordinate"])
    outputs = _BOUNDARY_OUTPUTS[coordinate["boundary"]]
    _boundary_contract(root["boundary_contract"])
    contracts = _numerical_contract(root["numerical_contract"], outputs)
    cells, missing = _cells(root["cells"])
    timing_missing = _timing(root["timing"], cells)
    accuracy, accuracy_missing = _accuracy(root["accuracy"], outputs)
    repeat_missing = _repeatability(root["repeatability"], outputs, contracts)
    provenance_missing = _provenance(root["provenance"])
    payload_missing = _payloads(root["payloads"], artifact_root=artifact_root)
    derived_missing = tuple(
        sorted((*missing, *timing_missing, *accuracy_missing, *repeat_missing, *provenance_missing, *payload_missing))
    )
    if declared_missing != derived_missing:
        raise AttentionEvaluationError(
            "missing_fields do not match absent evidence: "
            f"declared={list(declared_missing)}, derived={list(derived_missing)}"
        )
    if status == "accepted" and derived_missing:
        raise AttentionEvaluationError("accepted attention evidence must not have missing fields")
    if status == "partial" and not derived_missing:
        raise AttentionEvaluationError("partial attention evidence must identify at least one missing field")
    if not derived_missing:
        _validate_numerical_acceptance(accuracy, outputs, contracts)
        _validate_performance_acceptance(cells)

    provenance = cast(dict[str, object], root["provenance"])
    return AttentionEvaluationRecord(
        record_id=record_id,
        status=status,
        boundary=coordinate["boundary"],
        hardware=coordinate["hardware"],
        missing_fields=derived_missing,
        source_commit=cast(str, provenance["source_commit"]),
    )


def _coordinate(value: object) -> dict[str, str]:
    mapping = _closed_mapping(value, "coordinate", {"target_id", "boundary", "shape", "hardware"})
    if _integer(mapping["target_id"], "coordinate.target_id") != 3:
        raise AttentionEvaluationError("attention evidence must target scorecard target 3")
    boundary = _choice(mapping["boundary"], "coordinate.boundary", set(_BOUNDARY_OUTPUTS))
    shape = _string(mapping["shape"], "coordinate.shape")
    if shape != ATTENTION_SHAPE_ID:
        raise AttentionEvaluationError(f"coordinate.shape must be {ATTENTION_SHAPE_ID!r}")
    hardware = _choice(mapping["hardware"], "coordinate.hardware", {"h100", "gb200_or_b200"})
    return {"boundary": boundary, "hardware": hardware}


def _boundary_contract(value: object) -> None:
    mapping = _closed_mapping(value, "boundary_contract", {"shape_specification", "public_tensors", "adapters"})
    if _string(mapping["shape_specification"], "boundary_contract.shape_specification") != ATTENTION_SHAPE_SPECIFICATION:
        raise AttentionEvaluationError("attention boundary shape specification differs from schema version 1")
    tensors = _list(mapping["public_tensors"], "boundary_contract.public_tensors")
    expected_shapes = {
        "query": (1, 8, 4, 2048, 128),
        "key": (1, 8, 2048, 128),
        "value": (1, 8, 2048, 128),
        "output_cotangent": (1, 8, 4, 2048, 128),
        "output": (1, 8, 4, 2048, 128),
        "query_cotangent": (1, 8, 4, 2048, 128),
        "key_cotangent": (1, 8, 2048, 128),
        "value_cotangent": (1, 8, 2048, 128),
    }
    if len(tensors) != len(expected_shapes):
        raise AttentionEvaluationError("boundary_contract.public_tensors must describe all eight public tensors")
    for index, (name, shape) in enumerate(expected_shapes.items()):
        tensor = _closed_mapping(
            tensors[index],
            f"boundary_contract.public_tensors[{index}]",
            {"name", "dtype", "shape", "byte_strides", "layout"},
        )
        if _string(tensor["name"], f"public_tensors[{index}].name") != name:
            raise AttentionEvaluationError("public tensors must use the schema-defined order")
        if _string(tensor["dtype"], f"public_tensors[{index}].dtype") != "bf16":
            raise AttentionEvaluationError("attention public tensors must be BF16")
        actual_shape = tuple(_positive_integers(tensor["shape"], f"public_tensors[{index}].shape"))
        if actual_shape != shape:
            raise AttentionEvaluationError(f"public tensor {name!r} shape differs from the representative boundary")
        expected_strides = _row_major_byte_strides(shape, element_bytes=2)
        actual_strides = tuple(_positive_integers(tensor["byte_strides"], f"public_tensors[{index}].byte_strides"))
        if actual_strides != expected_strides:
            raise AttentionEvaluationError(f"public tensor {name!r} byte strides are not row-major")
        if _string(tensor["layout"], f"public_tensors[{index}].layout") != "row_major":
            raise AttentionEvaluationError(f"public tensor {name!r} must declare row_major layout")

    adapters = _list(mapping["adapters"], "boundary_contract.adapters")
    if len(adapters) != len(_IMPLEMENTATIONS):
        raise AttentionEvaluationError("boundary_contract.adapters must cover all four implementations")
    for index, implementation in enumerate(_IMPLEMENTATIONS):
        adapter = _closed_mapping(
            adapters[index],
            f"boundary_contract.adapters[{index}]",
            {"implementation", "included_in_timing", "operations"},
        )
        if _string(adapter["implementation"], f"adapters[{index}].implementation") != implementation:
            raise AttentionEvaluationError("adapter declarations must follow implementation order")
        if adapter["included_in_timing"] is not True:
            raise AttentionEvaluationError("all public-boundary adapters must be included in timing")
        operations = _unique_strings(adapter["operations"], f"adapters[{index}].operations")
        if implementation == "fa4_oracle" and not operations:
            raise AttentionEvaluationError("FA4 must declare its grouped-GQA boundary adapters")
        if implementation != "fa4_oracle" and operations:
            raise AttentionEvaluationError(f"{implementation} must not declare a public layout adapter")


def _numerical_contract(value: object, outputs: tuple[str, ...]) -> dict[str, dict[str, dict[str, float]]]:
    mapping = _closed_mapping(value, "numerical_contract", {"declared_before_timing", "outputs"})
    if mapping["declared_before_timing"] is not True:
        raise AttentionEvaluationError("numerical thresholds and floors must be declared before timing")
    records = _list(mapping["outputs"], "numerical_contract.outputs")
    if len(records) != len(outputs):
        raise AttentionEvaluationError("numerical_contract.outputs differs from the selected boundary")
    result: dict[str, dict[str, dict[str, float]]] = {}
    for index, output in enumerate(outputs):
        record = _closed_mapping(
            records[index], f"numerical_contract.outputs[{index}]", {"output", "floor", "limit", "repeat_bound"}
        )
        if _string(record["output"], f"numerical_contract.outputs[{index}].output") != output:
            raise AttentionEvaluationError("numerical output contracts must follow public result order")
        floor = _metric_mapping(record["floor"], f"numerical_contract.outputs[{index}].floor")
        limit = _metric_mapping(record["limit"], f"numerical_contract.outputs[{index}].limit")
        repeat = _closed_mapping(
            record["repeat_bound"],
            f"numerical_contract.outputs[{index}].repeat_bound",
            {"maximum_absolute_error", "maximum_ulp_error"},
        )
        repeat_bound = {
            "maximum_absolute_error": _nonnegative_number(
                repeat["maximum_absolute_error"], "repeat_bound.maximum_absolute_error"
            ),
            "maximum_ulp_error": _nonnegative_number(repeat["maximum_ulp_error"], "repeat_bound.maximum_ulp_error"),
        }
        result[output] = {"floor": floor, "limit": limit, "repeat_bound": repeat_bound}
    return result


def _cells(value: object) -> tuple[dict[str, dict[str, object]], list[str]]:
    records = _list(value, "cells")
    if len(records) != len(_IMPLEMENTATIONS):
        raise AttentionEvaluationError("cells must contain ordinary XLA, oracle, SOURCE_ORDERED, and FAST")
    cells: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for index, (implementation, policy) in enumerate(zip(_IMPLEMENTATIONS, _POLICIES, strict=True)):
        record = _closed_mapping(
            records[index], f"cells[{index}]", {"implementation", "policy", "median_ns", "compile_timings_ns"}
        )
        if _string(record["implementation"], f"cells[{index}].implementation") != implementation:
            raise AttentionEvaluationError("cells must follow the schema-defined implementation order")
        if record["policy"] != policy:
            raise AttentionEvaluationError(f"cells[{index}].policy does not match {implementation}")
        median = _optional_positive_integer(record["median_ns"], f"cells[{index}].median_ns", missing)
        timings = _closed_mapping(
            record["compile_timings_ns"], f"cells[{index}].compile_timings_ns", set(_COMPILE_TIMINGS)
        )
        parsed_timings = {
            name: _optional_nonnegative_integer(timings[name], f"cells[{index}].compile_timings_ns.{name}", missing)
            for name in _COMPILE_TIMINGS
        }
        cells[implementation] = {"median_ns": median, "compile_timings_ns": parsed_timings}
    return cells, missing


def _timing(value: object, cells: dict[str, dict[str, object]]) -> list[str]:
    mapping = _closed_mapping(
        value,
        "timing",
        {"warmup_iterations", "measured_rounds", "calls_per_sample", "counterbalance", "raw_samples"},
    )
    _positive_integer(mapping["warmup_iterations"], "timing.warmup_iterations")
    rounds = _positive_integer(mapping["measured_rounds"], "timing.measured_rounds")
    if rounds < 3 or rounds % 2 == 0:
        raise AttentionEvaluationError("timing.measured_rounds must be odd and at least three")
    _positive_integer(mapping["calls_per_sample"], "timing.calls_per_sample")
    if _string(mapping["counterbalance"], "timing.counterbalance") != "alternating_forward_reverse":
        raise AttentionEvaluationError("timing.counterbalance must be alternating_forward_reverse")
    if mapping["raw_samples"] is None:
        return ["timing.raw_samples"]
    samples = _list(mapping["raw_samples"], "timing.raw_samples")
    if len(samples) != rounds * len(_IMPLEMENTATIONS):
        raise AttentionEvaluationError("timing.raw_samples must contain every implementation in every round")
    grouped: dict[str, list[int]] = {name: [] for name in _IMPLEMENTATIONS}
    cursor = 0
    for round_index in range(rounds):
        order = _IMPLEMENTATIONS if round_index % 2 == 0 else tuple(reversed(_IMPLEMENTATIONS))
        for position, implementation in enumerate(order):
            sample = _closed_mapping(
                samples[cursor],
                f"timing.raw_samples[{cursor}]",
                {"round", "position", "implementation", "elapsed_ns"},
            )
            if _integer(sample["round"], f"raw_samples[{cursor}].round") != round_index:
                raise AttentionEvaluationError("raw sample rounds must be contiguous")
            if _integer(sample["position"], f"raw_samples[{cursor}].position") != position:
                raise AttentionEvaluationError("raw sample positions must be contiguous within each round")
            if _string(sample["implementation"], f"raw_samples[{cursor}].implementation") != implementation:
                raise AttentionEvaluationError("raw samples do not follow the declared counterbalance")
            grouped[implementation].append(_positive_integer(sample["elapsed_ns"], f"raw_samples[{cursor}].elapsed_ns"))
            cursor += 1
    for implementation, values in grouped.items():
        if cells[implementation]["median_ns"] != statistics.median(values):
            raise AttentionEvaluationError(f"{implementation} median_ns does not match raw samples")
    return []


def _accuracy(value: object, outputs: tuple[str, ...]) -> tuple[dict[tuple[str, str], dict[str, float]], list[str]]:
    if value is None:
        return {}, ["accuracy"]
    records = _list(value, "accuracy")
    expected = tuple((implementation, output) for implementation in _IMPLEMENTATIONS for output in outputs)
    if len(records) != len(expected):
        raise AttentionEvaluationError("accuracy must cover every implementation and public output")
    result: dict[tuple[str, str], dict[str, float]] = {}
    missing: list[str] = []
    for index, (implementation, output) in enumerate(expected):
        record = _closed_mapping(records[index], f"accuracy[{index}]", {"implementation", "output", *_METRICS})
        if record["implementation"] != implementation or record["output"] != output:
            raise AttentionEvaluationError("accuracy records must follow implementation and output order")
        metrics: dict[str, float] = {}
        for metric in _METRICS:
            parsed = _optional_nonnegative_number(record[metric], f"accuracy[{index}].{metric}", missing)
            if parsed is not None:
                metrics[metric] = parsed
        result[(implementation, output)] = metrics
    return result, missing


def _repeatability(
    value: object,
    outputs: tuple[str, ...],
    contracts: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    if value is None:
        return ["repeatability"]
    records = _list(value, "repeatability")
    expected = tuple((implementation, output) for implementation in _IMPLEMENTATIONS for output in outputs)
    if len(records) != len(expected):
        raise AttentionEvaluationError("repeatability must cover every implementation and public output")
    missing: list[str] = []
    for index, (implementation, output) in enumerate(expected):
        record = _closed_mapping(
            records[index],
            f"repeatability[{index}]",
            {"implementation", "output", "hashes", "maximum_pairwise_absolute_error", "maximum_pairwise_ulp_error"},
        )
        if record["implementation"] != implementation or record["output"] != output:
            raise AttentionEvaluationError("repeatability records must follow implementation and output order")
        hashes = _list(record["hashes"], f"repeatability[{index}].hashes")
        if not hashes:
            missing.append(f"repeatability[{index}].hashes")
        elif len(hashes) < 3 or any(not _is_sha256(item) for item in hashes):
            raise AttentionEvaluationError("repeatability hashes must contain at least three SHA-256 values")
        maximum_absolute = _optional_nonnegative_number(
            record["maximum_pairwise_absolute_error"],
            f"repeatability[{index}].maximum_pairwise_absolute_error",
            missing,
        )
        maximum_ulp = _optional_nonnegative_number(
            record["maximum_pairwise_ulp_error"],
            f"repeatability[{index}].maximum_pairwise_ulp_error",
            missing,
        )
        bound = contracts[output]["repeat_bound"]
        if maximum_absolute is not None and maximum_absolute > bound["maximum_absolute_error"]:
            raise AttentionEvaluationError(f"repeatability[{index}] exceeds its absolute-error bound")
        if maximum_ulp is not None and maximum_ulp > bound["maximum_ulp_error"]:
            raise AttentionEvaluationError(f"repeatability[{index}] exceeds its ULP bound")
    return missing


def _provenance(value: object) -> list[str]:
    mapping = _closed_mapping(value, "provenance", set(_PROVENANCE_FIELDS))
    missing: list[str] = []
    for field in _PROVENANCE_FIELDS:
        item = mapping[field]
        path = f"provenance.{field}"
        if item is None:
            missing.append(path)
            continue
        if field == "source_dirty":
            if type(item) is not bool:
                raise AttentionEvaluationError(f"{path} must be a boolean")
            continue
        text = _string(item, path)
        if field in {"source_commit", "source_tree", "xla_commit"} and not _is_sha1(text):
            raise AttentionEvaluationError(f"{path} must be a SHA-1")
        if field.endswith("sha256") and not _is_sha256(text):
            raise AttentionEvaluationError(f"{path} must be a SHA-256")
    for field, expected in _PINNED_ORACLE.items():
        actual = mapping[field]
        if actual is not None and actual != expected:
            raise AttentionEvaluationError(f"provenance.{field} differs from the pinned FA4 oracle")
    if mapping["source_dirty"] is False and mapping["source_diff_sha256"] not in {None, hashlib.sha256(b"").hexdigest()}:
        raise AttentionEvaluationError("a clean source identity must use the empty-diff SHA-256")
    return missing


def _payloads(value: object, *, artifact_root: Path) -> list[str]:
    mapping = _closed_mapping(value, "payloads", set(_PAYLOADS))
    missing: list[str] = []
    for kind in _PAYLOADS:
        item = mapping[kind]
        if item is None:
            missing.append(f"payloads.{kind}")
            continue
        record = _closed_mapping(item, f"payloads.{kind}", {"path", "sha256", "bytes"})
        path_text = _string(record["path"], f"payloads.{kind}.path")
        path = PurePosixPath(path_text)
        if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
            raise AttentionEvaluationError(f"payloads.{kind}.path must be a relative artifact path")
        sha256 = _string(record["sha256"], f"payloads.{kind}.sha256")
        if not _is_sha256(sha256):
            raise AttentionEvaluationError(f"payloads.{kind}.sha256 must be a SHA-256")
        expected_bytes = _positive_integer(record["bytes"], f"payloads.{kind}.bytes")
        payload_path = artifact_root.joinpath(*path.parts)
        if not payload_path.is_file():
            raise AttentionEvaluationError(f"payloads.{kind} file does not exist: {path_text}")
        payload = payload_path.read_bytes()
        if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != sha256:
            raise AttentionEvaluationError(f"payloads.{kind} content does not match its size and SHA-256")
    return missing


def _validate_numerical_acceptance(
    accuracy: dict[tuple[str, str], dict[str, float]],
    outputs: tuple[str, ...],
    contracts: dict[str, dict[str, dict[str, float]]],
) -> None:
    for output in outputs:
        oracle = accuracy[("fa4_oracle", output)]
        contract = contracts[output]
        for implementation in ("shuttle_source_ordered", "shuttle_fast"):
            metrics = accuracy[(implementation, output)]
            for metric in _METRICS:
                if metrics[metric] > contract["limit"][metric]:
                    raise AttentionEvaluationError(f"{implementation} {output} {metric} exceeds the declared limit")
                if metrics[metric] > max(oracle[metric], contract["floor"][metric]):
                    raise AttentionEvaluationError(
                        f"{implementation} {output} {metric} exceeds both oracle error and dtype-resolution floor"
                    )


def _validate_performance_acceptance(cells: dict[str, dict[str, object]]) -> None:
    oracle = cast(int, cells["fa4_oracle"]["median_ns"])
    for implementation in ("shuttle_source_ordered", "shuttle_fast"):
        median = cast(int, cells[implementation]["median_ns"])
        if median > 1.20 * oracle:
            raise AttentionEvaluationError(f"{implementation} median exceeds the 1.20x oracle policy")


def _metric_mapping(value: object, name: str) -> dict[str, float]:
    mapping = _closed_mapping(value, name, set(_METRICS))
    return {metric: _nonnegative_number(mapping[metric], f"{name}.{metric}") for metric in _METRICS}


def _row_major_byte_strides(shape: tuple[int, ...], *, element_bytes: int) -> tuple[int, ...]:
    stride = element_bytes
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _closed_mapping(value: object, name: str, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict:
        raise AttentionEvaluationError(f"{name} must be an object")
    mapping = cast(dict[object, object], value)
    if any(type(key) is not str for key in mapping):
        raise AttentionEvaluationError(f"{name} keys must be strings")
    typed = cast(dict[str, object], mapping)
    missing = fields - typed.keys()
    unknown = typed.keys() - fields
    if missing or unknown:
        raise AttentionEvaluationError(
            f"{name} fields differ from schema: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return typed


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AttentionEvaluationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise AttentionEvaluationError(f"{name} must be a list")
    return cast(list[object], value)


def _string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise AttentionEvaluationError(f"{name} must be a nonempty string")
    return value


def _identifier(value: object, name: str) -> str:
    text = _string(value, name)
    if not text.replace("_", "a").isalnum() or not text[0].isalnum():
        raise AttentionEvaluationError(f"{name} must be a lowercase identifier")
    return text


def _choice(value: object, name: str, choices: set[str]) -> str:
    text = _string(value, name)
    if text not in choices:
        raise AttentionEvaluationError(f"{name} has unknown value {text!r}")
    return text


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise AttentionEvaluationError(f"{name} must be an integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise AttentionEvaluationError(f"{name} must be positive")
    return result


def _optional_positive_integer(value: object, name: str, missing: list[str]) -> int | None:
    if value is None:
        missing.append(name)
        return None
    return _positive_integer(value, name)


def _optional_nonnegative_integer(value: object, name: str, missing: list[str]) -> int | None:
    if value is None:
        missing.append(name)
        return None
    result = _integer(value, name)
    if result < 0:
        raise AttentionEvaluationError(f"{name} must be nonnegative")
    return result


def _nonnegative_number(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise AttentionEvaluationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise AttentionEvaluationError(f"{name} must be finite and nonnegative")
    return result


def _optional_nonnegative_number(value: object, name: str, missing: list[str]) -> float | None:
    if value is None:
        missing.append(name)
        return None
    return _nonnegative_number(value, name)


def _positive_integers(value: object, name: str) -> list[int]:
    return [_positive_integer(item, f"{name}[{index}]") for index, item in enumerate(_list(value, name))]


def _unique_strings(value: object, name: str) -> list[str]:
    result = [_string(item, f"{name}[{index}]") for index, item in enumerate(_list(value, name))]
    if len(result) != len(set(result)):
        raise AttentionEvaluationError(f"{name} must not contain duplicates")
    return result


def _is_sha1(value: object) -> bool:
    return type(value) is str and len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
