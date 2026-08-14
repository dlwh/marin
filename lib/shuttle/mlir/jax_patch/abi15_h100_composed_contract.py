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

CONTRACT_SHA256 = "7b681b7a74eef86a3ce257f396dab442150a5cca41580f953f745add741849ba"
CONTRACT_ID = "shuttle_abi15_h100_composed_matmul_vjp_prerun_v1"
STATUS = "local_prerun_fixture_oracle_sealed_build_and_hardware_unexecuted"
PRODUCTION_COMMIT = "d31a88cb9a93e591469fa8edfcf9dacada90c7b4"
PRODUCTION_TREE = "66175cbd89dcac0857ee2fbc456f7715170d239c"
FIXTURE_SHA256 = "f82300541a6009beb1d3785d42cf613eb338c8c1d91b331c162e0a4e2c3c7e4e"


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


def load_contract(path: Path | None = None) -> Contract:
    """Load the exact reviewed contract and recompute its local fixture/oracle joins."""
    selected_path = contract_path() if path is None else path
    payload = selected_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CONTRACT_SHA256:
        raise ValueError("ABI15 H100 pre-run contract byte drift")
    document = _require_mapping(json.loads(payload), "contract")
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
    _validate_oracle_digests(document)
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
