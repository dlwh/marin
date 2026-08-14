# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the sealed Transformer Engine 2.17 LayerNorm oracle contract."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

CONTRACT_SHA256 = "5a3391f00bf5453d735cfe4359d8e8dff73fb70ccdf7908a0d19b592c413d005"
PLAN_SHA256 = "f42f251961131e5b1c870161f48da14e018001e4aa762752d82d0a7a8c1a6965"
ADAPTER = Path(__file__).with_name("abi11_layernorm_te_oracle_adapter.cpp")
ADAPTER_SHA256 = "4743cbd5398c5be9a3a5f73b8ebaa1c6ac01d245a1178f3b3766633682e948c4"
MAX_CONTRACT_BYTES = 1024 * 1024
SOURCE_FILES = {
    "transformer_engine/common/include/transformer_engine/normalization.h": (
        "9533ef14eb870236c3200c9f143973e870ef52c4d07ddeb74fc2c4acd948fbc2"
    ),
    "transformer_engine/common/include/transformer_engine/transformer_engine.h": (
        "29963fa4df150956b7775b0e80e88e657def80b22397eff7710a33c40b9a4c33"
    ),
    "transformer_engine/common/normalization/common.cpp": (
        "f7a24af63ca0110a888ebbf2a3ab5c209035be5973e708186a376d5990bf37a7"
    ),
    "transformer_engine/common/normalization/layernorm/ln_api.cpp": (
        "ad1abb6e76d0f2bdaea28877ab1eca741c797dc088278f2f760204cf222b9f53"
    ),
    "transformer_engine/common/normalization/layernorm/ln_fwd_cuda_kernel.cu": (
        "863a89caf4d930ef346facd08a3b3ac5cff3f71118c1199eaf21f30aaf7f8879"
    ),
    "transformer_engine/common/normalization/layernorm/ln_bwd_semi_cuda_kernel.cu": (
        "8c4549440568ca34bbe4e0f00b22f38e82bde0f68a732b0f1260ccd324128187"
    ),
}
SUBMODULES = {
    "3rdparty/cudnn-frontend": "e46d7082450264ce05cf898f8740011c4896f817",
    "3rdparty/cutlass": "57e3cfb47a2d9e0d46eb6335c3dc411498efa198",
    "3rdparty/googletest": "f8d7d77c06936315286eb55f8de22cd23c188571",
    "3rdparty/nccl": "a6b5de08b6af4f938cef541ae6e4d405632f89a4",
}
API_DECLARATIONS = {
    "nvte_layernorm_fwd": (
        "void nvte_layernorm_fwd(const NVTETensor x, const NVTETensor gamma, const NVTETensor beta, "
        "const float epsilon, NVTETensor z, NVTETensor mu, NVTETensor rsigma, NVTETensor workspace, "
        "const int multiprocessorCount, const bool zero_centered_gamma, cudaStream_t stream);"
    ),
    "nvte_layernorm_bwd": (
        "void nvte_layernorm_bwd(const NVTETensor dz, const NVTETensor x, const NVTETensor mu, "
        "const NVTETensor rsigma, const NVTETensor gamma, NVTETensor dx, NVTETensor dgamma, "
        "NVTETensor dbeta, NVTETensor workspace, const int multiprocessorCount, "
        "const bool zero_centered_gamma, cudaStream_t stream);"
    ),
    "nvte_enable_cudnn_norm_fwd": "void nvte_enable_cudnn_norm_fwd(bool enable);",
    "nvte_enable_cudnn_norm_bwd": "void nvte_enable_cudnn_norm_bwd(bool enable);",
}


@dataclass(frozen=True)
class ExecutionConfig:
    source_root: Path
    adapter_binary: Path
    adapter_binary_sha256: str
    transformer_engine_library: Path
    transformer_engine_library_sha256: str
    output: Path
    device: int


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _lowercase_sha256(value: str, name: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def verify_file_identity(path: Path, expected_sha256: str, name: str, *, executable: bool = False) -> None:
    """Verify one caller-supplied file identity without treating it as build provenance."""
    expected = _lowercase_sha256(expected_sha256, f"{name} identity")
    if not path.is_file() or (executable and not os.access(path, os.X_OK)):
        requirement = "an executable file" if executable else "a file"
        raise ValueError(f"{name} must be {requirement}")
    if _sha256(path) != expected:
        raise ValueError(f"{name} identity drifted")


def load_contract(path: Path) -> dict[str, Any]:
    """Load the byte-pinned closed contract and validate its independent identities."""
    if not path.is_file():
        raise ValueError(f"contract is not a file: {path}")
    if path.stat().st_size > MAX_CONTRACT_BYTES:
        raise ValueError("contract exceeds the 1 MiB validation limit")
    if _sha256(path) != CONTRACT_SHA256:
        raise ValueError("contract byte identity drifted")
    contract = json.loads(path.read_bytes(), object_pairs_hook=_strict_object, parse_constant=_reject_constant)
    if not isinstance(contract, dict):
        raise ValueError("contract root must be an object")
    if contract.get("contract_id") != "abi11_layernorm_bf16_te_2_17_oracle_red_v1":
        raise ValueError("contract identity drifted")
    if contract.get("provider", {}).get("source_files") != SOURCE_FILES:
        raise ValueError("pinned Transformer Engine source identities drifted")
    signatures = contract.get("api", {}).get("signatures")
    if not isinstance(signatures, dict) or set(signatures) != set(API_DECLARATIONS):
        raise ValueError("pinned Transformer Engine API set drifted")
    for name, declaration in API_DECLARATIONS.items():
        if signatures[name] != {
            "declaration": declaration,
            "sha256": hashlib.sha256(declaration.encode()).hexdigest(),
        }:
            raise ValueError(f"pinned Transformer Engine API drifted: {name}")
    records = contract.get("runner", {}).get("canonical_plan")
    if not isinstance(records, list) or len(records) != 20 or _canonical_sha256(records) != PLAN_SHA256:
        raise ValueError("canonical 20-record plan drifted")
    if contract["runner"].get("canonical_plan_sha256") != PLAN_SHA256:
        raise ValueError("canonical plan identity drifted")
    if contract.get("claims") != {
        "hardware_executed": False,
        "timing_collected": False,
        "authority_sealed": False,
        "scorecard_status_changed": False,
    }:
        raise ValueError("pre-run claims drifted")
    if contract.get("objective", {}).get("timing") != "not_authorized_not_part_of_this_contract":
        raise ValueError("timing policy drifted")
    return contract


def dry_run_plan(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical CPU-only plan without inspecting CUDA or provider files."""
    return {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "hardware_required": False,
        "records": contract["runner"]["canonical_plan"],
    }


def validate_checkout_identity(
    head: str, branch: str, status: str, submodule_status: list[str], contract: dict[str, Any]
) -> None:
    """Validate the claimed detached, clean, exact-commit checkout state."""
    if head != contract["provider"]["commit"]:
        raise ValueError("Transformer Engine checkout HEAD drifted")
    if branch:
        raise ValueError("Transformer Engine checkout must be detached")
    if status:
        raise ValueError("Transformer Engine checkout must be clean")
    observed_submodules: dict[str, str] = {}
    for line in submodule_status:
        if not line.startswith("-"):
            raise ValueError("Transformer Engine submodules must remain uninitialized")
        revision, path, *_description = line[1:].split()
        observed_submodules[path] = revision
    if observed_submodules != SUBMODULES:
        raise ValueError("Transformer Engine uninitialized submodule identities drifted")


def verify_source_checkout(source_root: Path, contract: dict[str, Any]) -> None:
    """Verify every source file pinned by the contract before loading an executable."""
    if contract["provider"]["commit"] != "2e559f062497bef768dfbe9d7e45548fadeca80a":
        raise ValueError("Transformer Engine commit drifted")
    try:
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(source_root), "branch", "--show-current"], check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        submodule_status = subprocess.run(
            ["git", "-C", str(source_root), "submodule", "status"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
    except subprocess.CalledProcessError as error:
        raise ValueError("Transformer Engine source root is not a readable Git checkout") from error
    validate_checkout_identity(head, branch, status, submodule_status, contract)
    for relative, expected in SOURCE_FILES.items():
        source = source_root / relative
        if not source.is_file():
            raise ValueError(f"pinned Transformer Engine source is missing: {relative}")
        if _sha256(source) != expected:
            raise ValueError(f"pinned Transformer Engine source drifted: {relative}")


def _fixed_inputs(rows: int, columns: int) -> dict[str, np.ndarray]:
    shapes = {
        "x": (rows, columns),
        "gamma": (columns,),
        "beta": (columns,),
        "dy": (rows, columns),
    }
    ranges = {
        "x": (-0.75, 0.875),
        "gamma": (-0.625, 1.0),
        "beta": (-0.25, 0.375),
        "dy": (-0.5, 0.75),
    }
    return {
        role: np.linspace(*ranges[role], math.prod(shape), dtype=np.float32).reshape(shape).astype(ml_dtypes.bfloat16)
        for role, shape in shapes.items()
    }


def _write_inputs(directory: Path, record: dict[str, Any], contract: dict[str, Any]) -> dict[str, dict[str, int | str]]:
    rows, columns = record["shape"]
    inputs = _fixed_inputs(rows, columns)
    expected = contract["inputs"]["digests"][f"{rows}x{columns}"]
    directory.mkdir()
    artifacts: dict[str, dict[str, int | str]] = {}
    for role in contract["boundaries"][record["boundary"]]["public_inputs"]:
        value = inputs[role]
        payload = value.tobytes(order="C")
        if hashlib.sha256(payload).hexdigest() != expected[role]["raw_sha256"]:
            raise ValueError(f"canonical input generation drifted: {role}")
        (directory / f"{role}.bf16").write_bytes(payload)
        artifacts[role] = {"bytes": len(payload), "sha256": expected[role]["raw_sha256"]}
    return artifacts


def _validate_execution_inputs(config: ExecutionConfig, contract: dict[str, Any], ordinal: int) -> dict[str, Any]:
    if not 0 <= ordinal < len(contract["runner"]["canonical_plan"]):
        raise ValueError("record ordinal is outside the closed 20-record plan")
    verify_source_checkout(config.source_root, contract)
    if _sha256(ADAPTER) != ADAPTER_SHA256:
        raise ValueError("LayerNorm adapter source drifted")
    verify_file_identity(config.adapter_binary, config.adapter_binary_sha256, "adapter binary", executable=True)
    verify_file_identity(
        config.transformer_engine_library,
        config.transformer_engine_library_sha256,
        "Transformer Engine library",
    )
    if config.output.exists():
        raise ValueError("execution output must not exist")
    if config.device < 0:
        raise ValueError("CUDA device ordinal must be nonnegative")
    return contract["runner"]["canonical_plan"][ordinal]


def _validate_adapter_metadata(
    metadata: object, record: dict[str, Any], ordinal: int, outputs: Path, contract: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, int | str]], dict[str, int | str]]:
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema_version",
        "status",
        "coordinate",
        "epsilon",
        "scaling_mode",
        "layout",
        "physical_sm_count",
        "workspace_queries",
        "timing_collected",
        "authority_sealed",
    }:
        raise ValueError("adapter metadata fields drifted")
    expected_coordinate = {
        "ordinal": ordinal,
        "shape": record["shape"],
        "boundary": record["boundary"],
        "forward_backend": record["forward_backend"],
        "backward_backend": record["backward_backend"],
    }
    if metadata["schema_version"] != 1 or metadata["status"] != "unsealed_local_hardware_observation":
        raise ValueError("adapter metadata status drifted")
    if metadata["coordinate"] != expected_coordinate:
        raise ValueError("adapter output coordinate drifted")
    if (
        metadata["epsilon"] != contract["execution"]["epsilon"]
        or metadata["scaling_mode"] != contract["execution"]["tensor_scaling_mode"]
        or metadata["layout"] != contract["execution"]["tensor_layout"]
        or metadata["timing_collected"] is not False
        or metadata["authority_sealed"] is not False
    ):
        raise ValueError("adapter execution semantics drifted")
    if type(metadata["physical_sm_count"]) is not int or metadata["physical_sm_count"] <= 0:
        raise ValueError("adapter physical SM count drifted")
    workspaces = metadata["workspace_queries"]
    if not isinstance(workspaces, dict) or set(workspaces) != {"forward", "backward"}:
        raise ValueError("adapter workspace metadata drifted")
    for name, workspace in workspaces.items():
        if name == "backward" and record["boundary"] == "forward":
            if workspace is not None:
                raise ValueError("forward-only record emitted a backward workspace")
            continue
        if (
            not isinstance(workspace, dict)
            or set(workspace) != {"shape", "dtype"}
            or workspace["dtype"] != "kNVTEByte"
            or not isinstance(workspace["shape"], list)
            or any(type(extent) is not int or extent < 0 for extent in workspace["shape"])
        ):
            raise ValueError(f"adapter {name} workspace metadata drifted")
    output_artifacts: dict[str, dict[str, int | str]] = {}
    rows, columns = record["shape"]
    sizes = {"y": rows * columns * 2, "dx": rows * columns * 2, "dgamma": columns * 2, "dbeta": columns * 2}
    expected_roles = contract["metrics"]["public_output_roles"][record["boundary"]]
    expected_entries = {"metadata.json", *(f"{role}.bf16" for role in expected_roles)}
    observed_entries = {path.name for path in outputs.iterdir()}
    if observed_entries != expected_entries:
        raise ValueError("adapter output directory entry set drifted")
    for path in outputs.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("adapter outputs must be regular files")
    for role in expected_roles:
        path = outputs / f"{role}.bf16"
        if not path.is_file() or path.stat().st_size != sizes[role]:
            raise ValueError(f"adapter public output drifted: {role}")
        output_artifacts[role] = {"bytes": sizes[role], "sha256": _sha256(path)}
    metadata_path = outputs / "metadata.json"
    metadata_artifact = {"bytes": metadata_path.stat().st_size, "sha256": _sha256(metadata_path)}
    return metadata, output_artifacts, metadata_artifact


def execute_record(contract: dict[str, Any], ordinal: int, config: ExecutionConfig) -> dict[str, Any]:
    """Execute one record through the pinned adapter in this fresh process."""
    record = _validate_execution_inputs(config, contract, ordinal)
    config.output.mkdir()
    inputs = config.output / "inputs"
    outputs = config.output / "outputs"
    outputs.mkdir()
    input_artifacts = _write_inputs(inputs, record, contract)
    argv = [
        str(config.adapter_binary),
        "--ordinal",
        str(ordinal),
        "--boundary",
        record["boundary"],
        "--forward-backend",
        record["forward_backend"],
        "--backward-backend",
        record["backward_backend"] or "none",
        "--rows",
        str(record["shape"][0]),
        "--columns",
        str(record["shape"][1]),
        "--device",
        str(config.device),
        "--inputs",
        str(inputs),
        "--outputs",
        str(outputs),
        "--te-library",
        str(config.transformer_engine_library),
    ]
    subprocess.run(argv, check=True)
    metadata_path = outputs / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("adapter did not emit metadata.json")
    raw_metadata = json.loads(
        metadata_path.read_bytes(), object_pairs_hook=_strict_object, parse_constant=_reject_constant
    )
    metadata, output_artifacts, metadata_artifact = _validate_adapter_metadata(
        raw_metadata, record, ordinal, outputs, contract
    )
    result = {
        "schema_version": 1,
        "status": "unsealed_local_hardware_observation",
        "hardware_executed": True,
        "timing_collected": False,
        "authority_sealed": False,
        "scorecard_status_changed": False,
        "contract_identity": {
            "contract_id": contract["contract_id"],
            "sha256": CONTRACT_SHA256,
        },
        "plan_identity": {
            "sha256": PLAN_SHA256,
            "ordinal": ordinal,
        },
        "source_identity": {
            "commit": contract["provider"]["commit"],
            "checkout_state": contract["provider"]["audit_checkout_state"],
        },
        "build_identity": {
            "runner_source_sha256": _sha256(Path(__file__)),
            "adapter_source_sha256": ADAPTER_SHA256,
            "adapter_binary_sha256": config.adapter_binary_sha256,
            "transformer_engine_library_sha256": config.transformer_engine_library_sha256,
            "authority": "caller_supplied_digests_not_source_derived_build_proof",
        },
        "execution_metadata": metadata,
        "execution_metadata_artifact": metadata_artifact,
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
    }
    (config.output / "result.json").write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--print-plan", action="store_true")
    modes.add_argument("--execute-record", type=int)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--adapter-binary", type=Path)
    parser.add_argument("--adapter-binary-sha256")
    parser.add_argument("--transformer-engine-library", type=Path)
    parser.add_argument("--transformer-engine-library-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    return parser


def _execution_config(arguments: argparse.Namespace) -> ExecutionConfig:
    names = (
        "source_root",
        "adapter_binary",
        "adapter_binary_sha256",
        "transformer_engine_library",
        "transformer_engine_library_sha256",
        "output",
    )
    missing = [name.replace("_", "-") for name in names if getattr(arguments, name) is None]
    if missing:
        raise ValueError(f"execution mode requires: {', '.join('--' + name for name in missing)}")
    return ExecutionConfig(
        source_root=arguments.source_root,
        adapter_binary=arguments.adapter_binary,
        adapter_binary_sha256=arguments.adapter_binary_sha256,
        transformer_engine_library=arguments.transformer_engine_library,
        transformer_engine_library_sha256=arguments.transformer_engine_library_sha256,
        output=arguments.output,
        device=arguments.device,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        contract = load_contract(arguments.contract)
        if arguments.print_plan:
            result = dry_run_plan(contract)
        else:
            result = execute_record(contract, arguments.execute_record, _execution_config(arguments))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"abi11_layernorm_te_oracle_runner: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
