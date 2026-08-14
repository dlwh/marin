# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independently verify one raw, untimed ABI15 H100 subject artifact."""

import hashlib
import json
import struct
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from abi15_h100_composed_contract import load_contract
from abi15_h100_evidence import (
    ProcessReport,
    RawArtifact,
    expected_role_paths,
    load_raw_artifact,
    load_role_manifest,
    require_sealed_hardware_authority,
    validate_process_reports,
)
from abi15_h100_gpu_build import validate_build_candidate
from abi15_h100_gpu_transport import (
    CallRole,
    NumericalPolicy,
    TransportCoordinate,
    gpu_transport_descriptor,
    load_gpu_transport,
)
from abi15_h100_numerics import (
    COLUMNS,
    REDUCTION,
    ROWS,
    ReductionPolicy,
    dense_inputs,
    reference,
    sparse_policy_inputs,
)
from acceptance_contract import EVENT_FIELDS

OUTPUT_ROLES = ("primal", "lhs_cotangent", "rhs_cotangent")
DATASETS = {"dense": dense_inputs, "sparse_policy_discriminator": sparse_policy_inputs}
SHAPES = {
    "primal": ROWS * COLUMNS,
    "lhs_cotangent": ROWS * REDUCTION,
    "rhs_cotangent": REDUCTION * COLUMNS,
}
RUNS = ("first", "sequential", "thread_0", "thread_1", "reuse")


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != _canonical_json(value):
        raise ValueError(f"{name} must be one canonical JSON object")
    return value


def _bits(path: Path, count: int, name: str) -> tuple[int, ...]:
    raw = path.read_bytes()
    if len(raw) != count * 2:
        raise ValueError(f"{name} has the wrong BF16 byte size")
    return struct.unpack(f"<{count}H", raw)


def validate_numerical_files(root: Path, roles: Mapping[str, str]) -> dict[str, object]:
    """Recompute all independent input/reference joins and repeated GPU outputs."""
    digests = {}
    for dataset, factory in DATASETS.items():
        inputs = factory()
        for input_role, expected in (
            ("lhs", inputs.lhs),
            ("rhs", inputs.rhs),
            ("output_cotangent", inputs.output_cotangent),
        ):
            observed = _bits(
                root / roles[f"input_{dataset}_{input_role}"],
                len(expected),
                f"{dataset} {input_role}",
            )
            if observed != expected:
                raise ValueError(f"{dataset} {input_role} differs from the independent fixture")
        for output_role in OUTPUT_ROLES:
            _bits(
                root / roles[f"baseline_{dataset}_{output_role}"],
                SHAPES[output_role],
                f"baseline {dataset} {output_role}",
            )
        for policy in ReductionPolicy:
            expected_outputs = reference(inputs, policy)
            values = (
                expected_outputs.primal,
                expected_outputs.lhs_cotangent,
                expected_outputs.rhs_cotangent,
            )
            for output_role, expected in zip(OUTPUT_ROLES, values, strict=True):
                reference_path = root / roles[f"reference_{policy.value}_{dataset}_{output_role}"]
                if (
                    _bits(
                        reference_path,
                        len(expected),
                        f"reference {policy.value} {dataset} {output_role}",
                    )
                    != expected
                ):
                    raise ValueError("stored ABI15 reference differs from the independent oracle")
                for run in RUNS:
                    output_path = root / roles[f"output_{policy.value}_{dataset}_{run}_{output_role}"]
                    observed = _bits(
                        output_path,
                        len(expected),
                        f"output {policy.value} {dataset} {run} {output_role}",
                    )
                    if observed != expected:
                        raise ValueError(f"ABI15 {policy.value} {dataset} {run} {output_role} differs bitwise")
                digests[f"{policy.value}/{dataset}/{output_role}"] = _sha256(reference_path.read_bytes())
    return {
        "bitwise_repeatability": True,
        "independent_reference_sha256": dict(sorted(digests.items())),
    }


def _process_report(path: Path, expected_process: str) -> ProcessReport:
    value = _load_json(path, f"{expected_process} report")
    if set(value) != {
        "cache_after",
        "cache_before",
        "cache_hits",
        "observer_events",
        "pid",
        "process",
    }:
        raise ValueError(f"{expected_process} report uses an open schema")
    if (
        value["process"] != expected_process
        or type(value["cache_before"]) is not list
        or type(value["cache_after"]) is not list
    ):
        raise ValueError(f"{expected_process} report identity or cache schema drift")
    return ProcessReport(
        process=value["process"],
        pid=value["pid"],
        cache_hits=value["cache_hits"],
        observer_events=value["observer_events"],
        cache_before=tuple(tuple(item) for item in value["cache_before"]),
        cache_after=tuple(tuple(item) for item in value["cache_after"]),
    )


def validate_observer_and_process_files(root: Path, roles: Mapping[str, str]) -> dict[str, object]:
    """Join fresh-process cache reports and policy-specific observer fingerprints."""
    reports = tuple(
        _process_report(root / roles[f"{process}_report"], process) for process in ("baseline", "populate", "reuse")
    )
    validate_process_reports(reports)
    contract = load_contract()
    for policy in ("source_ordered", "fast"):
        populate_path = root / roles[f"observer_populate_{policy}"]
        records = tuple(json.loads(line) for line in populate_path.read_text().splitlines())
        if any(type(record) is not list or len(record) != len(EVENT_FIELDS) for record in records):
            raise ValueError(f"{policy} observer record schema drift")
        events = tuple(dict(zip(EVENT_FIELDS, record, strict=True)) for record in records)
        if tuple(event["phase"] for event in events) != (
            "algebra_coverage",
            "generated_bundle",
            "final_erasure",
        ):
            raise ValueError(f"{policy} observer phase order drift")
        identity = contract.source_ordered if policy == "source_ordered" else contract.fast
        if any(event["policy"] != policy or event["policy_digest"] != identity.options_sha256 for event in events):
            raise ValueError(f"{policy} observer policy identity drift")
        if len({event["invocation_id"] for event in events}) != 1 or any(event["failure_pass"] for event in events):
            raise ValueError(f"{policy} observer invocation or failure state drift")
        if events[1]["normalized_module_fingerprint"] != identity.generated_fingerprint:
            raise ValueError(f"{policy} generated fingerprint drift")
        if (
            events[2]["normalized_module_fingerprint"] != identity.final_fingerprint
            or events[2]["no_shuttle_semantics"] is not True
        ):
            raise ValueError(f"{policy} final fingerprint or erasure drift")
        reuse_path = root / roles[f"observer_reuse_{policy}"]
        if reuse_path.read_bytes():
            raise ValueError(f"{policy} cache reuse unexpectedly invoked the pipeline")
    return {"observer_events": 6, "populate_hits": 0, "reuse_hits": 2}


def _validate_wheel_member(wheel: Path, suffix: str, extracted: Path, name: str) -> None:
    if wheel.stat().st_size > 512 * 1024 * 1024 or extracted.stat().st_size > 512 * 1024 * 1024:
        raise ValueError(f"{name} exceeds the bounded wheel evidence size")
    with zipfile.ZipFile(wheel) as archive:
        members = [member for member in archive.namelist() if member.endswith(suffix)]
        if len(members) != 1 or archive.read(members[0]) != extracted.read_bytes():
            raise ValueError(f"{name} differs from its exact wheel member")


def validate_build_and_transport_files(root: Path, roles: Mapping[str, str]) -> dict[str, object]:
    """Join build manifest, v2 registration, decoded transports, PTX, and cubins."""
    build = validate_build_candidate(_load_json(root / roles["build_manifest"], "build manifest"))
    for wheel_name, role in (
        ("jaxlib", "jaxlib_wheel"),
        ("cuda_plugin", "cuda_plugin_wheel"),
        ("cuda_pjrt", "cuda_pjrt_wheel"),
    ):
        raw = (root / roles[role]).read_bytes()
        if build["wheels"][wheel_name] != {"sha256": _sha256(raw), "size": len(raw)}:
            raise ValueError(f"{wheel_name} wheel differs from the build manifest")
    registration = _load_json(root / roles["registration_transcript"], "registration transcript")
    if build["registration_transcript_sha256"] != _sha256(_canonical_json(registration)):
        raise ValueError("registration transcript differs from the build manifest")
    expected_plugin = _sha256((root / roles["cuda_plugin_extension"]).read_bytes())
    if registration.get("pipeline_abi_version") != 15 or registration.get("plugin_extension_sha256") != expected_plugin:
        raise ValueError("registration transcript pipeline or plugin identity drift")
    targets = registration.get("exported_targets")
    if (
        type(targets) is not list
        or targets.count("shuttle.gpu.executable_bundle.v2") != 1
        or len(targets) != len(set(targets))
        or registration.get("upstream_target_count") != len(targets) - 1
    ):
        raise ValueError("registration transcript lacks one ABI15 v2 target")
    registered = registration.get("registration")
    if (
        type(registered) is not dict
        or set(registered) != {"api_version", "stage_pointers", "target", "traits"}
        or registered.get("target") != "shuttle.gpu.executable_bundle.v2"
        or registered.get("api_version") != 1
        or registered.get("traits") != 0
    ):
        raise ValueError("registration target drift")
    pointers = registered.get("stage_pointers")
    if type(pointers) is not dict or set(pointers) != {
        "instantiate",
        "prepare",
        "initialize",
        "execute",
    }:
        raise ValueError("registration stage schema drift")
    pointer_values = list(pointers.values())
    if any(type(value) is not int or value <= 0 for value in pointer_values) or len(set(pointer_values)) != 4:
        raise ValueError("registration stages must be distinct nonzero pointers")
    _validate_wheel_member(
        root / roles["jaxlib_wheel"],
        "/_jax.so",
        root / roles["jax_extension"],
        "JAX extension",
    )
    _validate_wheel_member(
        root / roles["jaxlib_wheel"],
        "/libjax_common.so",
        root / roles["jax_common"],
        "JAX common library",
    )
    _validate_wheel_member(
        root / roles["cuda_plugin_wheel"],
        "/cuda_plugin_extension.so",
        root / roles["cuda_plugin_extension"],
        "CUDA plugin extension",
    )
    _validate_wheel_member(
        root / roles["cuda_plugin_wheel"],
        ".dist-info/RECORD",
        root / roles["cuda_plugin_record"],
        "CUDA plugin RECORD",
    )
    _validate_wheel_member(
        root / roles["cuda_pjrt_wheel"],
        "/xla_cuda_plugin.so",
        root / roles["cuda_pjrt_plugin"],
        "CUDA PJRT plugin",
    )
    _validate_wheel_member(
        root / roles["cuda_pjrt_wheel"],
        ".dist-info/RECORD",
        root / roles["cuda_pjrt_record"],
        "CUDA PJRT RECORD",
    )
    for transport_record in build["transports"]:
        policy = NumericalPolicy(transport_record["policy"])
        call = CallRole(transport_record["call"])
        coordinate = TransportCoordinate(policy, call)
        prefix = f"{policy.value}_{call.value}"
        transport = load_gpu_transport(root / roles[f"transport_{prefix}"], coordinate)
        if _load_json(root / roles[f"descriptor_{prefix}"], f"{prefix} descriptor") != gpu_transport_descriptor(
            transport
        ):
            raise ValueError(f"{prefix} descriptor differs from the independent decoder")
        for field, observed in (
            ("transport_sha256", transport.transport_root),
            ("device_root", transport.device_root),
            ("invocation_root", transport.invocation_root),
            ("bundle_root", transport.bundle_root),
            ("code_sha256", transport.code_digest),
        ):
            if transport_record[field] != observed:
                raise ValueError(f"{prefix} {field} differs from decoded transport")
        for launch, slice_record in zip(transport.launches, transport_record["ptx_slices"], strict=True):
            ptx = (root / roles[f"ptx_{prefix}_{launch.ordinal:02d}"]).read_bytes()
            cubin = (root / roles[f"cubin_{prefix}_{launch.ordinal:02d}"]).read_bytes()
            if (
                ptx != launch.code
                or slice_record["ptx_sha256"] != _sha256(ptx)
                or slice_record["cubin_sha256"] != _sha256(cubin)
            ):
                raise ValueError(f"{prefix} PTX/cubin ordinal {launch.ordinal} join drift")
    return {
        "ptx_slices": 24,
        "registration_target": "shuttle.gpu.executable_bundle.v2",
        "status": build["status"],
    }


def verify_raw_artifact(root: Path, *, expected_harness_commit: str, expected_harness_tree: str) -> dict[str, object]:
    """Return a controller-side report without authoring an accepted envelope."""
    artifact: RawArtifact = load_raw_artifact(
        root,
        expected_harness_commit=expected_harness_commit,
        expected_harness_tree=expected_harness_tree,
    )
    roles = expected_role_paths(load_role_manifest())
    report = {
        "artifact_manifest_sha256": artifact.manifest_sha256,
        "build": validate_build_and_transport_files(artifact.root, roles),
        "numerical": validate_numerical_files(artifact.root, roles),
        "observer_cache": validate_observer_and_process_files(artifact.root, roles),
        "performance_status": "not_started",
        "schema_version": 1,
        "status": "controller_verified_raw_artifact_untimed",
    }
    if (root / "accepted_bundle.json").exists():
        raise ValueError("raw verifier must not consume a task-authored acceptance envelope")
    return report


def main() -> int:
    require_sealed_hardware_authority()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
