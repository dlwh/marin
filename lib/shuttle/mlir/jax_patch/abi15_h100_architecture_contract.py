# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Closed local ABI15 architecture/build design with no launch authority."""

import json
from pathlib import Path
from typing import Any

from abi15_h100_composed_contract import FIXTURE_SHA256, PRODUCTION_COMMIT, PRODUCTION_TREE, load_contract

FILENAME = "abi15-h100-architecture-build-contract-v1.json"
CONTRACT_SHA256 = "b93db1880eba19e41322780d29f14bb9b200cc7418165e8201264788f953ce72"
STATUS = "local_unsealed_non_authoritative_architecture_build_design"
JAX_COMMIT = "619764c15117fbefc4ba13ab941871cb514c23f6"
JAX_TREE = "56d186bccb48075374f32e46d2e090731b66e1ea"
XLA_COMMIT = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
TASK_ENTRYPOINT = "lib/shuttle/mlir/jax_patch/abi15_h100_evidence_launcher.py"
PATCHES = (
    (
        "lib/shuttle/mlir/jax_patch/0001-link-shuttle-xla-registry-adapter.patch",
        "1e8b1400ee05bf8c8037277046113e1f1b1fae5ef56077899b7e853855e3424e",
    ),
    (
        "lib/shuttle/mlir/jax_patch/0002-add-acceptance-observer-bridge.patch",
        "c4d5bc4aaa4b72ee7e2ecb44d28719be99da8defcf01ebd107308df41cb71942",
    ),
    (
        "lib/shuttle/mlir/jax_patch/0003-link-shuttle-cuda-ffi.patch",
        "a98f04ac7afc6e5d8df63f2679cf89bc7cd0b29509a393da86f352cb6ba2ec9f",
    ),
)


def _slot(
    ordinal: int,
    element_type: str,
    dimensions: tuple[int, int],
    storage: str,
    access: str,
    binding_kind: str,
    binding_index: int | None,
) -> dict[str, object]:
    return {
        "access": access,
        "binding_index": binding_index,
        "binding_kind": binding_kind,
        "dimensions": list(dimensions),
        "element_type": element_type,
        "ordinal": ordinal,
        "storage": storage,
    }


def _launch(
    ordinal: int,
    inputs: tuple[int, ...],
    output: int,
    dependencies: tuple[int, ...],
    kind: str,
) -> dict[str, object]:
    return {
        "dependencies": list(dependencies),
        "input_slots": list(inputs),
        "kind": kind,
        "ordinal": ordinal,
        "output_slots": [output],
    }


def _expected_coordinates() -> dict[str, object]:
    rows, reduction, columns = 33, 2, 34
    external_read = ("external", "read", "operand")
    temporary = ("temporary", "read_write", "none")
    external_write = ("external", "write", "result")
    return {
        "forward": {
            "launches": [
                _launch(0, (0, 1), 2, (), "contract_lhs_d0_d2_rhs_d2_d1"),
                _launch(1, (2,), 3, (0,), "map_f32_to_bf16_identity"),
            ],
            "slots": [
                _slot(0, "bf16", (rows, reduction), *external_read, 0),
                _slot(1, "bf16", (reduction, columns), *external_read, 1),
                _slot(2, "f32", (rows, columns), *temporary, None),
                _slot(3, "bf16", (rows, columns), *external_write, 0),
            ],
        },
        "backward": {
            "launches": [
                _launch(0, (0,), 3, (), "map_bf16_to_f32_identity"),
                _launch(1, (3,), 4, (0,), "map_f32_identity"),
                _launch(2, (1,), 5, (), "map_bf16_to_f32_identity"),
                _launch(3, (4, 5), 6, (1, 2), "contract_lhs_d2_d0_rhs_d2_d1"),
                _launch(4, (6,), 7, (3,), "map_f32_transpose_d1_d0"),
                _launch(5, (7,), 8, (4,), "map_f32_to_bf16_identity"),
                _launch(6, (3,), 9, (0,), "map_f32_identity"),
                _launch(7, (2,), 10, (), "map_bf16_to_f32_identity"),
                _launch(8, (9, 10), 11, (6, 7), "contract_lhs_d0_d2_rhs_d1_d2"),
                _launch(9, (11,), 12, (8,), "map_f32_to_bf16_identity"),
            ],
            "slots": [
                _slot(0, "bf16", (rows, columns), *external_read, 0),
                _slot(1, "bf16", (rows, reduction), *external_read, 1),
                _slot(2, "bf16", (reduction, columns), *external_read, 2),
                _slot(3, "f32", (rows, columns), *temporary, None),
                _slot(4, "f32", (rows, columns), *temporary, None),
                _slot(5, "f32", (rows, reduction), *temporary, None),
                _slot(6, "f32", (columns, reduction), *temporary, None),
                _slot(7, "f32", (reduction, columns), *temporary, None),
                _slot(8, "bf16", (reduction, columns), *external_write, 1),
                _slot(9, "f32", (rows, columns), *temporary, None),
                _slot(10, "f32", (reduction, columns), *temporary, None),
                _slot(11, "f32", (rows, reduction), *temporary, None),
                _slot(12, "bf16", (rows, reduction), *external_write, 0),
            ],
        },
    }


def _closed(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} must use the closed ABI15 design schema")
    return value


def _validate_slots(value: object, expected_count: int, name: str) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != expected_count:
        raise ValueError(f"{name} slot count drift")
    slots = []
    for ordinal, raw in enumerate(value):
        slot = _closed(
            raw,
            {"access", "binding_index", "binding_kind", "dimensions", "element_type", "ordinal", "storage"},
            f"{name}.slots[{ordinal}]",
        )
        if slot["ordinal"] != ordinal or slot["element_type"] not in {"bf16", "f32"}:
            raise ValueError(f"{name} slot ordinal or element type drift")
        if (
            type(slot["dimensions"]) is not list
            or not slot["dimensions"]
            or any(type(extent) is not int or extent <= 0 for extent in slot["dimensions"])
        ):
            raise ValueError(f"{name} slot dimensions drift")
        external = slot["storage"] == "external"
        expected_binding = "operand" if slot["access"] == "read" else "result"
        if external:
            if slot["binding_kind"] != expected_binding or type(slot["binding_index"]) is not int:
                raise ValueError(f"{name} external slot binding drift")
        elif (
            slot["storage"] != "temporary"
            or slot["access"] != "read_write"
            or slot["binding_kind"] != "none"
            or slot["binding_index"] is not None
        ):
            raise ValueError(f"{name} temporary slot contract drift")
        slots.append(slot)
    return slots


def _validate_launches(value: object, slots: list[dict[str, Any]], expected_kinds: tuple[str, ...], name: str) -> None:
    if type(value) is not list or len(value) != len(expected_kinds):
        raise ValueError(f"{name} launch count drift")
    initialized = {slot["ordinal"] for slot in slots if slot["binding_kind"] == "operand"}
    producer = {}
    for ordinal, raw in enumerate(value):
        launch = _closed(
            raw,
            {"dependencies", "input_slots", "kind", "ordinal", "output_slots"},
            f"{name}.launches[{ordinal}]",
        )
        if launch["ordinal"] != ordinal or launch["kind"] != expected_kinds[ordinal]:
            raise ValueError(f"{name} launch ordinal or semantic kind drift")
        inputs = launch["input_slots"]
        outputs = launch["output_slots"]
        if (
            type(inputs) is not list
            or type(outputs) is not list
            or not inputs
            or len(outputs) != 1
            or any(type(slot) is not int or not 0 <= slot < len(slots) for slot in inputs + outputs)
        ):
            raise ValueError(f"{name} launch slot roster drift")
        if not set(inputs) <= initialized or set(outputs) & initialized:
            raise ValueError(f"{name} launch reads uninitialized or overwrites initialized storage")
        if any(slots[slot]["access"] == "write" for slot in inputs) or any(
            slots[slot]["access"] == "read" for slot in outputs
        ):
            raise ValueError(f"{name} launch access drift")
        dependencies = sorted({producer[slot] for slot in inputs if slot in producer})
        if launch["dependencies"] != dependencies:
            raise ValueError(f"{name} launch dependencies drift")
        for slot in outputs:
            initialized.add(slot)
            producer[slot] = ordinal
    results = {slot["ordinal"] for slot in slots if slot["binding_kind"] == "result"}
    if not results <= initialized:
        raise ValueError(f"{name} leaves an external result uninitialized")


def load_architecture_build_contract(path: Path | None = None) -> dict[str, Any]:
    """Load and semantically recompute the unsealed local architecture design."""
    load_contract()
    selected = Path(__file__).with_name(FILENAME) if path is None else path
    raw = selected.read_bytes()
    document = json.loads(raw)
    if raw != (json.dumps(document, indent=2, sort_keys=True) + "\n").encode():
        raise ValueError("ABI15 architecture/build design must use canonical JSON")
    root = _closed(
        document,
        {"build_policy", "coordinates", "derivation", "design_stops", "schema_version", "status"},
        "design",
    )
    if root["schema_version"] != 1 or root["status"] != STATUS:
        raise ValueError("ABI15 architecture/build design status drift")
    derivation = _closed(root["derivation"], {"fixture_sha256", "generic_semantics"}, "derivation")
    if derivation["fixture_sha256"] != FIXTURE_SHA256 or type(derivation["generic_semantics"]) is not list:
        raise ValueError("ABI15 architecture derivation drift")
    coordinates = _closed(root["coordinates"], {"backward", "forward"}, "coordinates")
    if coordinates != _expected_coordinates():
        raise ValueError("ABI15 four-coordinate graph differs from the independent derivation")
    expected_kinds = {
        "forward": ("contract_lhs_d0_d2_rhs_d2_d1", "map_f32_to_bf16_identity"),
        "backward": (
            "map_bf16_to_f32_identity",
            "map_f32_identity",
            "map_bf16_to_f32_identity",
            "contract_lhs_d2_d0_rhs_d2_d1",
            "map_f32_transpose_d1_d0",
            "map_f32_to_bf16_identity",
            "map_f32_identity",
            "map_bf16_to_f32_identity",
            "contract_lhs_d0_d2_rhs_d1_d2",
            "map_f32_to_bf16_identity",
        ),
    }
    for name, slot_count in (("forward", 4), ("backward", 13)):
        coordinate = _closed(coordinates[name], {"launches", "slots"}, name)
        slots = _validate_slots(coordinate["slots"], slot_count, name)
        _validate_launches(coordinate["launches"], slots, expected_kinds[name], name)
    policy = _closed(root["build_policy"], {"commands", "future_outputs", "source", "tool_constraints"}, "build policy")
    source = _closed(
        policy["source"],
        {
            "harness_commit",
            "harness_tree",
            "jax_commit",
            "jax_patches",
            "jax_tree",
            "production_commit",
            "production_tree",
            "task_entrypoint",
            "xla_commit",
        },
        "build source",
    )
    if source["production_commit"] != PRODUCTION_COMMIT or source["production_tree"] != PRODUCTION_TREE:
        raise ValueError("ABI15 build source production identity drift")
    if source["harness_commit"] is not None or source["harness_tree"] is not None:
        raise ValueError("ABI15 build source must remain locally unsealed")
    if (
        source["jax_commit"] != JAX_COMMIT
        or source["jax_tree"] != JAX_TREE
        or source["xla_commit"] != XLA_COMMIT
        or source["task_entrypoint"] != TASK_ENTRYPOINT
        or source["jax_patches"] != [{"path": path, "sha256": digest} for path, digest in PATCHES]
    ):
        raise ValueError("ABI15 build source or ordered patch identity drift")
    outputs = _closed(
        policy["future_outputs"],
        {
            "ack_trust_root",
            "authorization_sha256",
            "bazel_path",
            "bazel_sha256",
            "cubin_roots",
            "image_digest",
            "ptxas_path",
            "ptxas_sha256",
            "ptx_roots",
            "wheel_sha256",
        },
        "future outputs",
    )
    if any(value is not None for value in outputs.values()):
        raise ValueError("ABI15 learned build outputs must remain null")
    stops = _closed(
        root["design_stops"],
        {
            "ack_trust_root",
            "ack_verification",
            "controller_accepted_envelope",
            "controller_process_receipts",
            "launch_readiness",
            "task_success_exit",
            "terminal_receipt",
        },
        "design stops",
    )
    if any(value is not None for value in stops.values()):
        raise ValueError("ABI15 authority and controller design stops must remain null")
    commands = policy["commands"]
    if (
        type(commands) is not list
        or len(commands) != 2
        or any(
            type(command) is not list or not command or any(type(item) is not str or not item for item in command)
            for command in commands
        )
    ):
        raise ValueError("ABI15 build command templates drift")
    if policy["tool_constraints"] != {"bazel_version": "7.4.1", "ptx_target": "sm_90", "ptxas_version": "13.2.78"}:
        raise ValueError("ABI15 build tool constraints drift")
    return root
