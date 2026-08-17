# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed loader for the immutable ABI15 composed H100 pre-run contract."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from abi15_h100_numerics import (
    ReductionPolicy,
    bf16_sha256,
    dense_inputs,
    reference,
    sparse_policy_inputs,
)

from shuttle.options import (
    ExecutionMode,
    Materialization,
    Numerics,
    Tuning,
)

CONTRACT_SHA256 = "7b681b7a74eef86a3ce257f396dab442150a5cca41580f953f745add741849ba"
CONTRACT_ID = "shuttle_abi15_h100_composed_matmul_vjp_prerun_v1"
STATUS = "local_prerun_fixture_oracle_sealed_build_and_hardware_unexecuted"
PRODUCTION_COMMIT = "d31a88cb9a93e591469fa8edfcf9dacada90c7b4"
PRODUCTION_TREE = "66175cbd89dcac0857ee2fbc456f7715170d239c"
FIXTURE_SHA256 = "f82300541a6009beb1d3785d42cf613eb338c8c1d91b331c162e0a4e2c3c7e4e"
EXPECTED_GENERATED_FINGERPRINTS = {
    "source_ordered": "d5f91111b5e05c7cdf4af807702287b1dc55233e697269c04769af438b4f8540",
    "fast": "96e8a120c055843446523819aa1e085f1391c7c790cb6f607c5c4339bd6cb571",
}
EXPECTED_FINAL_FINGERPRINTS = {
    "source_ordered": "036085a583909874d0c4e6c134926880af03d5b54ceffb3084d838362fe13fcc",
    "fast": "9b8de0ccb413b410c0176acd9bd6b7ac1bbca1759c1fab8e3bee60fd18e4beaa",
}


def abi15_options_json(*, numerics: Numerics, tuning: Tuning) -> str:
    """Return the exact canonical options text sealed by the ABI15 contract."""
    payload = {
        "execution_mode": ExecutionMode.GPU_EXECUTABLE_BUNDLE.value,
        "numerics": numerics.value,
        "pipeline_abi_version": 15,
        "schema_version": 1,
        "tuning": {
            "cluster_shape": list(tuning.cluster_shape),
            "materialization": tuning.materialization.value,
            "maximum_candidates": tuning.maximum_candidates,
            "pipeline_stages": tuning.pipeline_stages,
            "tile_sizes": list(tuning.tile_sizes),
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _historical_options_digest(*, numerics: Numerics, tuning: Tuning) -> str:
    return hashlib.sha256(abi15_options_json(numerics=numerics, tuning=tuning).encode()).hexdigest()


@dataclass(frozen=True)
class PolicyIdentity:
    options_sha256: str
    generated_fingerprint: str
    final_fingerprint: str


@dataclass(frozen=True)
class Contract:
    sha256: str
    contract_id: str
    status: str
    production_commit: str
    production_tree: str
    fixture_sha256: str
    source_ordered: PolicyIdentity
    fast: PolicyIdentity


def contract_path() -> Path:
    """Return the checked-in contract path."""
    return (
        Path(__file__).parents[4]
        / ".agents"
        / "projects"
        / "tile_lifetime_compiler"
        / "shuttle_abi15_h100_composed_prerun_v1.json"
    )


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an object")
    return value


def _require_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if type(value) is not str:
        raise ValueError(f"{key} must be a string")
    return value


def _policy(document: dict[str, object], name: str) -> PolicyIdentity:
    policies = _require_mapping(document.get("policies"), "policies")
    policy = _require_mapping(policies.get(name), name)
    return PolicyIdentity(
        options_sha256=_require_string(policy, "options_sha256"),
        generated_fingerprint=_require_string(policy, "generated_fingerprint"),
        final_fingerprint=_require_string(policy, "final_fingerprint"),
    )


def _validate_oracle_digests(document: dict[str, object]) -> None:
    datasets = _require_mapping(document.get("datasets"), "datasets")
    for name, inputs in (
        ("dense", dense_inputs()),
        ("sparse_policy_discriminator", sparse_policy_inputs()),
    ):
        dataset = _require_mapping(datasets.get(name), name)
        input_digests = _require_mapping(dataset.get("inputs"), f"{name}.inputs")
        expected_inputs = {
            "lhs_sha256": bf16_sha256(inputs.lhs),
            "rhs_sha256": bf16_sha256(inputs.rhs),
            "output_cotangent_sha256": bf16_sha256(inputs.output_cotangent),
        }
        if input_digests != expected_inputs:
            raise ValueError(f"{name} input digest drift")
        output_digests = _require_mapping(dataset.get("outputs"), f"{name}.outputs")
        for policy in ReductionPolicy:
            outputs = reference(inputs, policy)
            expected_outputs = {
                "primal_sha256": bf16_sha256(outputs.primal),
                "lhs_cotangent_sha256": bf16_sha256(outputs.lhs_cotangent),
                "rhs_cotangent_sha256": bf16_sha256(outputs.rhs_cotangent),
            }
            if output_digests.get(policy.value) != expected_outputs:
                raise ValueError(f"{name} {policy.value} output digest drift")

    sparse = reference(sparse_policy_inputs(), ReductionPolicy.SOURCE_ORDERED)
    sparse_fast = reference(sparse_policy_inputs(), ReductionPolicy.FAST)
    selected = _require_mapping(
        _require_mapping(
            datasets.get("sparse_policy_discriminator"),
            "sparse_policy_discriminator",
        ).get("selected_bits"),
        "selected_bits",
    )
    expected_selected = {
        "source_ordered": {
            "lhs_cotangent_0_0": f"0x{sparse.lhs_cotangent[0]:04x}",
            "rhs_cotangent_0_0": f"0x{sparse.rhs_cotangent[0]:04x}",
        },
        "fast": {
            "lhs_cotangent_0_0": f"0x{sparse_fast.lhs_cotangent[0]:04x}",
            "rhs_cotangent_0_0": f"0x{sparse_fast.rhs_cotangent[0]:04x}",
        },
    }
    if selected != expected_selected:
        raise ValueError("sparse policy selected-bit drift")


def validate_contract_document(document: dict[str, object]) -> None:
    """Validate semantics without establishing the document as an authority root."""
    if document.get("boundary") != "ordinary_jax_composed_matmul_vjp":
        raise ValueError("ABI15 evidence boundary drift")
    if document.get("target") != "shuttle.gpu.executable_bundle.v2":
        raise ValueError("ABI15 evidence target drift")
    if document.get("status") != STATUS:
        raise ValueError("ABI15 evidence status drift")
    if document.get("schemas") != {
        "bundle": 2,
        "device": 3,
        "invocation": 3,
        "materialization": 2,
        "schedule": 2,
        "transport": 2,
    }:
        raise ValueError("ABI15 evidence schema drift")
    if document.get("topology") != {
        "calls_per_policy": 2,
        "forward": {"external_bindings": 3, "launches": 2, "slots": 4},
        "backward": {
            "contract_ordinals": [3, 8],
            "external_bindings": 5,
            "launches": 10,
            "slots": 13,
            "transpose_ordinal": 4,
        },
        "ptx_slices_across_policies": 24,
    }:
        raise ValueError("ABI15 evidence topology drift")
    if document.get("process_contract") != {
        "cache_entries_after_populate": 2,
        "phases_per_populate_policy": [
            "algebra_coverage",
            "generated_bundle",
            "final_erasure",
        ],
        "populate_hits": 0,
        "processes": ["baseline", "populate", "reuse"],
        "reuse_hits": 2,
        "reuse_observer_events": 0,
    }:
        raise ValueError("ABI15 evidence process contract drift")
    fixture = _require_mapping(document.get("fixture"), "fixture")
    if fixture != {
        "columns": 34,
        "filename": "jax-0.10.1-bf16-matmul-vjp-33x2x34-composed.mlir",
        "jax_version": "0.10.1",
        "jaxlib_version": "0.10.1",
        "reduction": 2,
        "rows": 33,
        "sha256": FIXTURE_SHA256,
    }:
        raise ValueError("ordinary-JAX ABI15 fixture contract drift")
    tuning = Tuning(
        tile_sizes=(),
        cluster_shape=(),
        pipeline_stages=1,
        materialization=Materialization.AUTOMATIC,
        maximum_candidates=1,
    )
    for policy, numerics in (
        ("source_ordered", Numerics.SOURCE_ORDERED),
        ("fast", Numerics.FAST),
    ):
        identity = _policy(document, policy)
        expected_options = _historical_options_digest(numerics=numerics, tuning=tuning)
        if identity.options_sha256 != expected_options:
            raise ValueError(f"{policy} options identity drift")
        if identity.generated_fingerprint != EXPECTED_GENERATED_FINGERPRINTS[policy]:
            raise ValueError(f"{policy} generated fingerprint drift")
        if identity.final_fingerprint != EXPECTED_FINAL_FINGERPRINTS[policy]:
            raise ValueError(f"{policy} final fingerprint drift")
    _validate_oracle_digests(document)


def load_contract(path: Path | None = None) -> Contract:
    """Load the exact reviewed contract and recompute its local fixture/oracle joins."""
    selected_path = contract_path() if path is None else path
    payload = selected_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CONTRACT_SHA256:
        raise ValueError("ABI15 H100 pre-run contract byte drift")
    document = _require_mapping(json.loads(payload), "contract")
    validate_contract_document(document)
    compiler = _require_mapping(document.get("compiler"), "compiler")
    fixture = _require_mapping(document.get("fixture"), "fixture")
    authority = _require_mapping(document.get("hardware_authority"), "hardware_authority")
    if authority != {
        "image_digest": None,
        "launch_authorization_sha256": None,
        "source_capsule_sha256": None,
        "status": "not_authorized",
    }:
        raise ValueError("hardware authority must remain unsealed")
    fixture_path = Path(__file__).parents[1] / "test" / "Inputs" / _require_string(fixture, "filename")
    fixture_digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    if fixture_digest != FIXTURE_SHA256 or fixture_digest != _require_string(fixture, "sha256"):
        raise ValueError("ordinary-JAX ABI15 fixture drift")
    contract = Contract(
        sha256=digest,
        contract_id=_require_string(document, "contract_id"),
        status=_require_string(document, "status"),
        production_commit=_require_string(compiler, "production_commit"),
        production_tree=_require_string(compiler, "production_tree"),
        fixture_sha256=fixture_digest,
        source_ordered=_policy(document, "source_ordered"),
        fast=_policy(document, "fast"),
    )
    if contract.contract_id != CONTRACT_ID or contract.status != STATUS:
        raise ValueError("ABI15 H100 pre-run contract identity drift")
    if contract.production_commit != PRODUCTION_COMMIT or contract.production_tree != PRODUCTION_TREE:
        raise ValueError("ABI15 compiler production identity drift")
    return contract
