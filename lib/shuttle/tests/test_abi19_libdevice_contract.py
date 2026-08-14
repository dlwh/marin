# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import re
from pathlib import Path

import pytest

_CONTRACT_PATH = Path(__file__).with_name("abi19-libdevice-acquisition-v1.json")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_EXPECTED_WHEELS = {
    (
        "manylinux2010_x86_64.manylinux_2_12_x86_64",
        "f5aa433631109bbdec81802c5b5f319bf10bc891fe2f212e4e445845211d6f77",
        64279462,
        "https://files.pythonhosted.org/packages/e8/1f/930d63ccc8adcdf27bfc051a24e3e4da2cf6ef987848d6d1d642e29d704b/"
        "nvidia_nvvm-13.2.78-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
    ),
    (
        "manylinux2014_aarch64.manylinux_2_17_aarch64",
        "88075f87a361a1dce95c799cabc028f7093af616a5702dcfb74eba4045dbbd5f",
        61886055,
        "https://files.pythonhosted.org/packages/8b/fd/db44b7a662a6af75a9a0683ca4580c855a3f5fcfdf1261b0ddb9fce0ee26/"
        "nvidia_nvvm-13.2.78-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
    ),
}


def _contract() -> dict[str, object]:
    return json.loads(_CONTRACT_PATH.read_text())


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be an object with string keys")
    return value


def _closed(value: object, path: str, keys: set[str]) -> dict[str, object]:
    mapping = _mapping(value, path)
    if set(mapping) != keys:
        raise ValueError(f"{path} fields do not match the closed contract")
    return mapping


def _exact(value: object, expected: object, path: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"{path} does not match the pinned value")


def _validate_source_contract(payload: dict[str, object]) -> None:
    root = _closed(payload, "contract", {"artifact", "compile_contract", "license", "schema_version", "source"})
    _exact(root["schema_version"], 1, "schema_version")

    source = _closed(root["source"], "source", {"lockfile_sha256", "package", "registry", "version", "wheels"})
    _exact(
        source["lockfile_sha256"],
        "e03633c1f7f8cd01fff82db197206d1ddeb0fad7ba0f4968b3b21e4831f20bc9",
        "source.lockfile_sha256",
    )
    _exact(source["package"], "nvidia-nvvm", "source.package")
    _exact(source["registry"], "https://pypi.org/simple", "source.registry")
    _exact(source["version"], "13.2.78", "source.version")
    wheels = source["wheels"]
    if not isinstance(wheels, list) or len(wheels) != 2:
        raise ValueError("source.wheels must contain the two pinned Linux artifacts")
    decoded_wheels: set[tuple[object, object, object, object]] = set()
    for index, value in enumerate(wheels):
        wheel = _closed(value, f"source.wheels[{index}]", {"platform", "sha256", "size", "url"})
        if not isinstance(wheel["sha256"], str) or _SHA256.fullmatch(wheel["sha256"]) is None:
            raise ValueError(f"source.wheels[{index}].sha256 must be lowercase SHA-256")
        decoded_wheels.add((wheel["platform"], wheel["sha256"], wheel["size"], wheel["url"]))
    if decoded_wheels != _EXPECTED_WHEELS:
        raise ValueError("source.wheels do not match the locked NVIDIA artifacts")

    license_contract = _closed(
        root["license"],
        "license",
        {"classification", "evidence", "source_path", "source_revision", "source_sha256"},
    )
    _exact(license_contract["classification"], "restricted", "license.classification")
    _exact(
        license_contract["source_path"],
        "third_party/gpus/cuda/hermetic/cuda_nvcc.BUILD.tpl",
        "license.source_path",
    )
    _exact(
        license_contract["source_revision"],
        "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69",
        "license.source_revision",
    )
    _exact(
        license_contract["source_sha256"],
        "8d967dc789bb28df23df42b51003300c3b3d67b4bbacad59aaedfb8a6747c86d",
        "license.source_sha256",
    )
    _exact(license_contract["evidence"], None, "license.evidence")

    compile_contract = _closed(
        root["compile_contract"],
        "compile_contract",
        {
            "accuracy",
            "address_size",
            "fast_function_forbidden",
            "ftz",
            "function",
            "linkage",
            "ptx_version",
            "target",
            "target_triple",
            "unresolved_calls",
        },
    )
    expected_compile = {
        "accuracy": "DEFAULT",
        "address_size": 64,
        "fast_function_forbidden": "__nv_fast_expf",
        "ftz": False,
        "function": "__nv_expf",
        "linkage": "compile_time",
        "ptx_version": "8.0",
        "target": "sm_90",
        "target_triple": "nvptx64-nvidia-cuda",
        "unresolved_calls": 0,
    }
    for key, expected in expected_compile.items():
        _exact(compile_contract[key], expected, f"compile_contract.{key}")


def _validate_unsealed_artifact(payload: dict[str, object]) -> None:
    artifact = _closed(
        payload["artifact"],
        "artifact",
        {
            "libdevice_path",
            "libdevice_sha256",
            "libdevice_size",
            "license_path",
            "license_sha256",
            "selected_platform",
        },
    )
    if any(value is not None for value in artifact.values()):
        raise ValueError("artifact must remain entirely null before authorized acquisition")


def _set_path(payload: dict[str, object], path: tuple[str | int, ...], value: object) -> None:
    current: object = payload
    for component in path[:-1]:
        if isinstance(component, int):
            if not isinstance(current, list):
                raise ValueError("integer path component requires a list")
            current = current[component]
        else:
            current = _mapping(current, "mutation path")[component]
    final = path[-1]
    if isinstance(final, int):
        if not isinstance(current, list):
            raise ValueError("integer path component requires a list")
        current[final] = value
    else:
        _mapping(current, "mutation path")[final] = value


def test_abi19_source_contract_pins_official_nvvm_and_default_exp() -> None:
    payload = _contract()
    _validate_source_contract(payload)
    _validate_unsealed_artifact(payload)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), 2, "schema_version"),
        (("source", "package"), "nvidia-cuda-nvcc", "source.package"),
        (("source", "lockfile_sha256"), "0" * 64, "source.lockfile_sha256"),
        (("source", "version"), "13.2.79", "source.version"),
        (("source", "wheels", 0, "sha256"), "0" * 64, "source.wheels"),
        (("source", "wheels", 0, "size"), 64279463, "source.wheels"),
        (("source", "wheels", 1, "url"), "https://example.invalid/nvvm.whl", "source.wheels"),
        (("license", "classification"), "notice", "license.classification"),
        (("license", "source_sha256"), "0" * 64, "license.source_sha256"),
        (("license", "source_revision"), "0" * 40, "license.source_revision"),
        (("compile_contract", "accuracy"), "APPROXIMATE", "compile_contract.accuracy"),
        (("compile_contract", "function"), "__nv_fast_expf", "compile_contract.function"),
        (("compile_contract", "fast_function_forbidden"), "__nv_expf", "fast_function_forbidden"),
        (("compile_contract", "ftz"), True, "compile_contract.ftz"),
        (("compile_contract", "ptx_version"), "8.1", "compile_contract.ptx_version"),
        (("compile_contract", "target"), "sm_100", "compile_contract.target"),
        (("compile_contract", "target_triple"), "nvptx-nvidia-cuda", "target_triple"),
        (("compile_contract", "unresolved_calls"), 1, "unresolved_calls"),
    ],
)
def test_abi19_source_contract_rejects_metadata_and_codegen_mutations(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    payload = copy.deepcopy(_contract())
    _set_path(payload, path, value)
    with pytest.raises(ValueError, match=message):
        _validate_source_contract(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_platform", "manylinux2010_x86_64.manylinux_2_12_x86_64"),
        ("libdevice_path", "nvvm/libdevice/libdevice.10.bc"),
        ("libdevice_sha256", "0" * 64),
        ("libdevice_size", 1),
        ("license_path", "LICENSE"),
        ("license_sha256", "0" * 64),
    ],
)
def test_abi19_unsealed_contract_rejects_partial_artifact_fields(field: str, value: object) -> None:
    payload = copy.deepcopy(_contract())
    artifact = _mapping(payload["artifact"], "artifact")
    artifact[field] = value
    with pytest.raises(ValueError, match="entirely null"):
        _validate_unsealed_artifact(payload)


def test_abi19_unsealed_contract_rejects_unexpected_artifact_fields() -> None:
    payload = copy.deepcopy(_contract())
    artifact = _mapping(payload["artifact"], "artifact")
    artifact["download_command"] = "curl"
    with pytest.raises(ValueError, match="closed contract"):
        _validate_unsealed_artifact(payload)
