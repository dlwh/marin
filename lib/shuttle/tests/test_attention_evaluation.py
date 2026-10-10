# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from shuttle.attention_evaluation import (
    ATTENTION_SHAPE_ID,
    ATTENTION_SHAPE_SPECIFICATION,
    AttentionEvaluationError,
    load_attention_evaluation,
    validate_attention_evaluation,
    write_attention_evaluation,
)
from shuttle.evaluation_manifest import (
    ArchitectureConformance,
    Boundary,
    EvidenceFormat,
    Gate,
    Hardware,
    Status,
    _artifact_claims,
)
from shuttle.options import Numerics

IMPLEMENTATIONS = ("ordinary_xla", "fa4_oracle", "shuttle_source_ordered", "shuttle_fast")
OUTPUTS = ("output", "query_cotangent", "key_cotangent", "value_cotangent")
METRICS = (
    "maximum_absolute_error",
    "mean_absolute_error",
    "maximum_relative_error",
    "maximum_ulp_error",
)
PAYLOADS = (
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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strides(shape: tuple[int, ...]) -> list[int]:
    stride = 2
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return list(reversed(result))


def _record(tmp_path: Path) -> dict[str, object]:
    tensor_shapes = {
        "query": (1, 8, 4, 2048, 128),
        "key": (1, 8, 2048, 128),
        "value": (1, 8, 2048, 128),
        "output_cotangent": (1, 8, 4, 2048, 128),
        "output": (1, 8, 4, 2048, 128),
        "query_cotangent": (1, 8, 4, 2048, 128),
        "key_cotangent": (1, 8, 2048, 128),
        "value_cotangent": (1, 8, 2048, 128),
    }
    payloads = {}
    for name in PAYLOADS:
        data = f"representative {name} payload\n".encode()
        path = tmp_path / f"{name}.artifact"
        path.write_bytes(data)
        payloads[name] = {"path": path.name, "sha256": _sha256(data), "bytes": len(data)}

    medians = {"ordinary_xla": 101, "fa4_oracle": 81, "shuttle_source_ordered": 91, "shuttle_fast": 71}
    values = {
        "ordinary_xla": (100, 101, 102),
        "fa4_oracle": (80, 81, 82),
        "shuttle_source_ordered": (90, 91, 92),
        "shuttle_fast": (70, 71, 72),
    }
    samples = []
    for round_index in range(3):
        order = IMPLEMENTATIONS if round_index % 2 == 0 else tuple(reversed(IMPLEMENTATIONS))
        for position, implementation in enumerate(order):
            samples.append(
                {
                    "round": round_index,
                    "position": position,
                    "implementation": implementation,
                    "elapsed_ns": values[implementation][round_index],
                }
            )

    accuracy = []
    repeatability = []
    metric_values = {
        "ordinary_xla": (0.0, 0.0, 0.0, 0),
        "fa4_oracle": (0.015625, 0.0008, 0.008, 2),
        "shuttle_source_ordered": (0.015625, 0.0007, 0.007, 2),
        "shuttle_fast": (0.015625, 0.00075, 0.0075, 2),
    }
    for implementation in IMPLEMENTATIONS:
        for output in OUTPUTS:
            accuracy.append(
                {
                    "implementation": implementation,
                    "output": output,
                    **dict(zip(METRICS, metric_values[implementation], strict=True)),
                }
            )
            repeatability.append(
                {
                    "implementation": implementation,
                    "output": output,
                    "hashes": [_sha256(f"{implementation}:{output}:{repeat}".encode()) for repeat in range(3)],
                    "maximum_pairwise_absolute_error": 0.0,
                    "maximum_pairwise_ulp_error": 0,
                }
            )

    compile_timings = {
        "trace_ns": 10,
        "lowering_ns": 20,
        "compilation_ns": 30,
        "tuning_ns": 40,
        "warmup_ns": 50,
        "cold_cache_compile_ns": 60,
        "warm_cache_compile_ns": 15,
    }
    floor = dict(zip(METRICS, (0.015625, 0.001, 0.01, 2), strict=True))
    limit = dict(zip(METRICS, (0.03125, 0.002, 0.02, 4), strict=True))
    empty_sha = _sha256(b"")
    return {
        "schema_version": 1,
        "record_id": "attention_composed_h100_run_1",
        "status": "accepted",
        "missing_fields": [],
        "coordinate": {
            "target_id": 3,
            "boundary": "composed_forward_backward",
            "shape": ATTENTION_SHAPE_ID,
            "hardware": "h100",
        },
        "boundary_contract": {
            "shape_specification": ATTENTION_SHAPE_SPECIFICATION,
            "public_tensors": [
                {
                    "name": name,
                    "dtype": "bf16",
                    "shape": list(shape),
                    "byte_strides": _strides(shape),
                    "layout": "row_major",
                }
                for name, shape in tensor_shapes.items()
            ],
            "adapters": [
                {"implementation": "ordinary_xla", "included_in_timing": True, "operations": []},
                {
                    "implementation": "fa4_oracle",
                    "included_in_timing": True,
                    "operations": [
                        "q/do: transpose(0,3,1,2,4), reshape BSHD",
                        "k/v: transpose(0,2,1,3)",
                        "o/dq: reshape grouped GQA, transpose(0,2,3,1,4)",
                        "dk/dv: transpose(0,2,1,3)",
                    ],
                },
                {"implementation": "shuttle_source_ordered", "included_in_timing": True, "operations": []},
                {"implementation": "shuttle_fast", "included_in_timing": True, "operations": []},
            ],
        },
        "numerical_contract": {
            "declared_before_timing": True,
            "outputs": [
                {
                    "output": output,
                    "floor": floor,
                    "limit": limit,
                    "repeat_bound": {"maximum_absolute_error": 0.001, "maximum_ulp_error": 1},
                }
                for output in OUTPUTS
            ],
        },
        "cells": [
            {
                "implementation": implementation,
                "policy": policy,
                "median_ns": medians[implementation],
                "compile_timings_ns": compile_timings,
            }
            for implementation, policy in zip(IMPLEMENTATIONS, (None, None, "source_ordered", "fast"), strict=True)
        ],
        "timing": {
            "warmup_iterations": 10,
            "measured_rounds": 3,
            "calls_per_sample": 5,
            "counterbalance": "alternating_forward_reverse",
            "raw_samples": samples,
        },
        "accuracy": accuracy,
        "repeatability": repeatability,
        "provenance": {
            "source_commit": "1" * 40,
            "source_tree": "2" * 40,
            "source_dirty": False,
            "source_diff_sha256": empty_sha,
            "shuttle_source_sha256": "3" * 64,
            "python_version": "3.12.11",
            "jax_version": "0.10.1",
            "jaxlib_version": "0.10.1+shuttle",
            "xla_commit": "4" * 40,
            "cutlass_dsl_version": "4.5.3",
            "tvm_ffi_version": "0.1.13.post2",
            "cuda_toolkit_version": "13.0",
            "cuda_driver_version": "580.82.07",
            "nvcc_version": "13.0.88",
            "ptxas_version": "13.0.88",
            "ncu_version": "2026.1.1",
            "nsys_version": "2026.1.3",
            "device_name": "NVIDIA H100 80GB HBM3",
            "device_uuid": "GPU-00000000-0000-0000-0000-000000000001",
            "compute_capability": "9.0",
            "oracle_package": "flash-attn-4",
            "oracle_version": "4.0.0b19",
            "oracle_wheel_sha256": "bc3856e018fa32e2b833726566641e51d02d45f2d444849937276a46c01653a2",
            "oracle_sdist_sha256": "c904c9da0387c2ac0420cffdf61d6201d94cfeb354b6768fde794b04e9fc4e89",
            "oracle_backend_source_sha256": "66a956ea9505f384105845140837824a8062724547b87c0a9bb2c92d663fd37c",
            "oracle_config_source_sha256": "d99bb01e74ac5f68a3551f70f8f17ffa92f2b84fed6061bc423c0e1e25ec52f2",
        },
        "payloads": payloads,
    }


def test_accepted_attention_evidence_round_trips_complete_contract(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_text(json.dumps(_record(tmp_path)))

    record = load_attention_evaluation(path)

    assert record.accepted
    assert record.missing_fields == ()
    assert record.boundary == "composed_forward_backward"


def test_attention_evidence_writer_content_addresses_payloads_and_round_trips(tmp_path: Path) -> None:
    document = _record(tmp_path)
    payloads = {name: f"live run {name}\n".encode() for name in PAYLOADS}
    path = tmp_path / "attention.json"

    record = write_attention_evaluation(document, path, payloads=payloads)
    persisted = json.loads(path.read_text())

    assert record.accepted
    assert load_attention_evaluation(path) == record
    for name, content in payloads.items():
        payload = persisted["payloads"][name]
        payload_path = tmp_path / payload["path"]
        assert payload_path.read_bytes() == content
        assert payload_path.name == f"{_sha256(content)}.{name}"


def test_attention_evidence_writer_rejects_incomplete_payload_set(tmp_path: Path) -> None:
    document = _record(tmp_path)
    payloads = {name: name.encode() for name in PAYLOADS if name != "sass"}

    with pytest.raises(AttentionEvaluationError, match="cover exactly"):
        write_attention_evaluation(document, tmp_path / "attention.json", payloads=payloads)


def test_partial_attention_evidence_requires_exact_derived_missing_fields(tmp_path: Path) -> None:
    document = _record(tmp_path)
    document["status"] = "partial"
    payloads = document["payloads"]
    assert isinstance(payloads, dict)
    payloads["sass"] = None
    document["missing_fields"] = ["payloads.sass"]

    record = validate_attention_evaluation(document, artifact_root=tmp_path)

    assert not record.accepted
    assert record.missing_fields == ("payloads.sass",)


def test_accepted_attention_evidence_claims_only_its_exact_boundary_and_hardware(tmp_path: Path) -> None:
    document = _record(tmp_path)

    claims = _artifact_claims(
        EvidenceFormat.ATTENTION_EVALUATION_V1,
        json.dumps(document).encode(),
        artifact_root=tmp_path,
        source_commit="1" * 40,
        architecture_conformance=ArchitectureConformance.CONFORMING,
        name="attention",
    )

    assert {claim.identity.policy for claim in claims} == {Numerics.SOURCE_ORDERED, Numerics.FAST}
    assert {claim.identity.boundary for claim in claims} == {Boundary.COMPOSED_FORWARD_BACKWARD}
    assert {claim.identity.hardware for claim in claims} == {Hardware.H100}
    assert all({gate.gate for gate in claim.gates} == set(Gate) for claim in claims)
    assert all({gate.status for gate in claim.gates} == {Status.ACCEPTED} for claim in claims)


def test_partial_attention_evidence_cannot_claim_acceptance_gates(tmp_path: Path) -> None:
    document = _record(tmp_path)
    document["status"] = "partial"
    document["payloads"]["sass"] = None
    document["missing_fields"] = ["payloads.sass"]

    claims = _artifact_claims(
        EvidenceFormat.ATTENTION_EVALUATION_V1,
        json.dumps(document).encode(),
        artifact_root=tmp_path,
        source_commit="1" * 40,
        architecture_conformance=ArchitectureConformance.CONFORMING,
        name="attention",
    )

    assert claims == ()


def test_partial_attention_evidence_rejects_an_unreported_missing_field(tmp_path: Path) -> None:
    document = _record(tmp_path)
    document["status"] = "partial"
    payloads = document["payloads"]
    assert isinstance(payloads, dict)
    payloads["sass"] = None

    with pytest.raises(AttentionEvaluationError, match="missing_fields do not match"):
        validate_attention_evaluation(document, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("layout", "byte strides"),
        ("threshold", "declared before timing"),
        ("cell_order", "implementation order"),
        ("sample_order", "counterbalance"),
        ("accuracy", "missing_fields do not match"),
        ("repeat", "at least three"),
        ("compile", "missing_fields do not match"),
        ("provenance", "missing_fields do not match"),
        ("payload", "missing_fields do not match"),
        ("payload_hash", "content does not match"),
    ],
)
def test_accepted_attention_evidence_rejects_missing_or_unsealed_requirements(
    tmp_path: Path, mutation: str, message: str
) -> None:
    document = _record(tmp_path)
    if mutation == "layout":
        document["boundary_contract"]["public_tensors"][0]["byte_strides"][0] += 2
    elif mutation == "threshold":
        document["numerical_contract"]["declared_before_timing"] = False
    elif mutation == "cell_order":
        document["cells"][0], document["cells"][1] = document["cells"][1], document["cells"][0]
    elif mutation == "sample_order":
        document["timing"]["raw_samples"][0]["implementation"] = "fa4_oracle"
    elif mutation == "accuracy":
        document["accuracy"][0]["maximum_ulp_error"] = None
    elif mutation == "repeat":
        document["repeatability"][0]["hashes"] = document["repeatability"][0]["hashes"][:2]
    elif mutation == "compile":
        document["cells"][0]["compile_timings_ns"]["cold_cache_compile_ns"] = None
    elif mutation == "provenance":
        document["provenance"]["cuda_driver_version"] = None
    elif mutation == "payload":
        document["payloads"]["ptx"] = None
    elif mutation == "payload_hash":
        document["payloads"]["fatbin"]["sha256"] = "0" * 64
    else:
        raise AssertionError(f"unhandled mutation {mutation}")

    with pytest.raises(AttentionEvaluationError, match=message):
        validate_attention_evaluation(document, artifact_root=tmp_path)


def test_accepted_attention_evidence_enforces_oracle_relative_numerics_and_performance(tmp_path: Path) -> None:
    bad_numerics = _record(tmp_path)
    fast_output = next(
        item
        for item in bad_numerics["accuracy"]
        if item["implementation"] == "shuttle_fast" and item["output"] == "output"
    )
    fast_output["maximum_absolute_error"] = 0.02
    with pytest.raises(AttentionEvaluationError, match="dtype-resolution floor"):
        validate_attention_evaluation(bad_numerics, artifact_root=tmp_path)

    bad_performance = _record(tmp_path)
    fast_cell = next(item for item in bad_performance["cells"] if item["implementation"] == "shuttle_fast")
    fast_cell["median_ns"] = 100
    fast_samples = [
        item for item in bad_performance["timing"]["raw_samples"] if item["implementation"] == "shuttle_fast"
    ]
    for sample, elapsed in zip(fast_samples, (99, 100, 101), strict=True):
        sample["elapsed_ns"] = elapsed
    with pytest.raises(AttentionEvaluationError, match=r"1\.20x oracle"):
        validate_attention_evaluation(bad_performance, artifact_root=tmp_path)


def test_attention_evidence_rejects_oracle_identity_not_pinned_by_lock_and_adapter_sources(tmp_path: Path) -> None:
    document = _record(tmp_path)
    document["provenance"]["oracle_version"] = "4.0.0b20"

    with pytest.raises(AttentionEvaluationError, match="pinned FA4 oracle"):
        validate_attention_evaluation(document, artifact_root=tmp_path)


def test_attention_evidence_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    encoded = json.dumps(_record(tmp_path)).replace(
        '{"schema_version": 1,', '{"schema_version": 1, "schema_version": 1,', 1
    )
    path.write_text(encoded)

    with pytest.raises(AttentionEvaluationError, match="duplicate JSON object key"):
        load_attention_evaluation(path)
