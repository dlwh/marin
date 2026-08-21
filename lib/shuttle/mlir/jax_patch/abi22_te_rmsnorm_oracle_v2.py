# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned matched Transformer Engine/XLA RMSNorm oracle protocol."""

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import ml_dtypes
import numpy as np

ARCHITECTURE_SHAPE = (7, 13)
REPRESENTATIVE_SHAPE = (2048, 4096)
SHAPES = (ARCHITECTURE_SHAPE, REPRESENTATIVE_SHAPE)
BOUNDARIES = ("forward", "backward_recompute", "composed")
EPSILON = 1.0e-6
MEASURED_INVOCATIONS = 20
WARMUP_INVOCATIONS = 10
NORM_CONTRACT_BLOCKER = "te_2_17_rmsnorm_oracle_v2_does_not_cover_norm_contract"
QUALIFICATION_PATTERN = "qualification_linspace_v2"
HELDOUT_PATTERN = "heldout_reversed_linspace_v1"
ACCEPTANCE_LIMITS = {
    "y": {"max_absolute_error": 0.0078125, "max_bfloat16_ulp": 2},
    "dx": {"max_absolute_error": 0.0625, "max_bfloat16_ulp": 64},
    "dgamma": {"max_absolute_error": 0.015625, "max_bfloat16_ulp": 8},
}
NUMERICAL_POLICY = {
    "schema": "shuttle_abi22_rmsnorm_role_policy_v1",
    "patterns": [QUALIFICATION_PATTERN, HELDOUT_PATTERN],
    "repeat_count": 2,
    "limits_by_role": ACCEPTANCE_LIMITS,
}
TE_SOURCE_COMMIT = "2e559f062497bef768dfbe9d7e45548fadeca80a"
JAX_REVISION = "619764c15117fbefc4ba13ab941871cb514c23f6"
XLA_REVISION = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
PRODUCERS = ("transformer_engine_2_17", "ordinary_jax_xla")
OUTPUT_ROLES = {
    "forward": ("y",),
    "backward_recompute": ("dx", "dgamma"),
    "composed": ("y", "dx", "dgamma"),
}
CALL_SEQUENCES = {
    "forward": ("rmsnorm_forward",),
    "backward_recompute": ("rmsnorm_forward_to_private_rsigma", "rmsnorm_backward"),
    "composed": ("rmsnorm_forward_to_public_y_and_private_rsigma", "rmsnorm_backward"),
}
INPUT_DIGESTS = {
    ARCHITECTURE_SHAPE: {
        "x": "bf45599484c0439c23fcd951f8f67645c1812c709002b7cbe3a7f7c67c69a9bb",
        "gamma": "909f58ed4a0ea4d2a436730afc3cb5c0960eb11019dfa2836d7827019ae87771",
        "dy": "7f1673f57926522d586d66896ff64949d88a02aa0bd43b246b519004ffa6f89f",
    },
    REPRESENTATIVE_SHAPE: {
        "x": "0f3b7cfa2054631d63038001deb298b086d0c2baffeaddd6082c4518df91e13c",
        "gamma": "e90137a7f830b75f00ee6dae7c89bbd7fba30dfb31ff7951fec129441c170821",
        "dy": "275b6bb7bf115e9110a4403637863cd0a3f0fb2cbb7b18ac9de05560db5a154c",
    },
}
REFERENCE_DIGESTS = {
    ARCHITECTURE_SHAPE: {
        "y": "5a78b29a63f3163c7012bcb40581b7290340f189c0e2f6b14036bb7ec1a99ab3",
        "dx": "79678506dafd3e75dc57813e76f46f9fcaa3f0b5b96d4243ebda9c9cb2a35302",
        "dgamma": "0b1c5df0409d00bf82a6d90748424f03019c6439d599003f28f9a71c79e7c7e9",
    },
    REPRESENTATIVE_SHAPE: {
        "y": "85924a7a28e8edb0a21aff788016315a0b804465ae04cac79c69517ff7c71284",
        "dx": "b832dffb3e31d5e5639c2e272bbd9e74cc5ec3a664298b02604448628c0fbdab",
        "dgamma": "7e8e78fc4dad00d99b614018ae47cfe67162a859cc10bab79c2c2b0ca73746b3",
    },
}
HELDOUT_INPUT_DIGESTS = {
    "x": "595fc9c13633ba99278591d4288b9150b1d7c71eb17bd9c5f0d4720e1762ea8b",
    "gamma": "bc3f6002b8b4c4718db1aeee0cf2b459c18ec80aa69958d2c8945be915a409d6",
    "dy": "237d26f9c6c048e5d3979da9e283aff46f062518d9a91ea5b74d2f8eaf80526d",
}
HELDOUT_REFERENCE_DIGESTS = {
    "y": "15a4e1660f2dc93b91841a0ac8c54c90372548a4b1269a09cc9c428cdc3e98f5",
    "dx": "d5e6be3d8cf52e4139235b9dde69a84db489a705580a376f08a613eb7beeab9a",
    "dgamma": "06498bae527509f636aae0baa0b247e657e5f6dc1f7453cf3279e53db90664e0",
}
INPUT_RANGES = ((-0.75, 0.875), (-0.625, 1.0), (-0.5, 1.125))


_RUNTIME_IDENTITY_SEAL = object()


class ValidatedTeRuntimeIdentity(Mapping[str, str]):
    """Immutable identity proven against resolved runner and TE library bytes."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, str], seal: object) -> None:
        if seal is not _RUNTIME_IDENTITY_SEAL:
            raise TypeError("validated runtime identities are created only by manifest validation")
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("validated runtime identities are immutable")

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and self._values == dict(other)


class ValidatedExecutionPlan(dict[str, Any]):
    """Exact command matrix returned by the canonical plan derivation."""


class ValidatedQualificationPlan(dict[str, Any]):
    """Exact repeated qualification matrix for the frozen role policy."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bf16_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    if array.dtype != ml_dtypes.bfloat16:
        raise ValueError("array is not bfloat16")
    return array.view(np.uint16).tobytes(order="C")


def _fixed_inputs(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if shape not in SHAPES:
        raise ValueError("shape is outside the RMSNorm v2 contract")
    rows, features = shape
    dimensions = ((rows, features), (features,), (rows, features))
    return tuple(
        np.linspace(start, stop, math.prod(dimensions[index]), dtype=np.float32)
        .reshape(dimensions[index])
        .astype(ml_dtypes.bfloat16)
        for index, (start, stop) in enumerate(INPUT_RANGES)
    )  # type: ignore[return-value]


def _reference(
    x: np.ndarray, gamma: np.ndarray, dy: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x64 = x.astype(np.float64)
    gamma64 = gamma.astype(np.float64)
    dy64 = dy.astype(np.float64)
    features = x.shape[1]
    inverse = np.reciprocal(np.sqrt(np.sum(x64 * x64, axis=1, keepdims=True, dtype=np.float64) / features + epsilon))
    y = (x64 * inverse * gamma64).astype(ml_dtypes.bfloat16)
    correlation = np.sum(dy64 * x64 * gamma64, axis=1, keepdims=True, dtype=np.float64)
    dx = (dy64 * gamma64 * inverse - x64 * inverse**3 * correlation / features).astype(ml_dtypes.bfloat16)
    dgamma = np.sum(dy64 * x64 * inverse, axis=0, dtype=np.float64).astype(ml_dtypes.bfloat16)
    return y, dx, dgamma


def _case_files(shape: tuple[int, int]) -> dict[str, tuple[int, str]]:
    rows, features = shape
    return {
        "x.bf16": (rows * features * 2, INPUT_DIGESTS[shape]["x"]),
        "gamma.bf16": (features * 2, INPUT_DIGESTS[shape]["gamma"]),
        "dy.bf16": (rows * features * 2, INPUT_DIGESTS[shape]["dy"]),
        "reference_y.bf16": (rows * features * 2, REFERENCE_DIGESTS[shape]["y"]),
        "reference_dx.bf16": (rows * features * 2, REFERENCE_DIGESTS[shape]["dx"]),
        "reference_dgamma.bf16": (features * 2, REFERENCE_DIGESTS[shape]["dgamma"]),
    }


def _verify_case_directory(case_root: Path) -> None:
    for shape in SHAPES:
        rows, features = shape
        directory = case_root / f"{rows}x{features}"
        for name, (size, digest) in _case_files(shape).items():
            path = directory / name
            if not path.is_file():
                raise ValueError(f"case file missing: {path}")
            payload = path.read_bytes()
            if len(payload) != size:
                raise ValueError(f"case file size drifted: {path}")
            if _sha256_bytes(payload) != digest:
                raise ValueError(f"case file digest drifted: {path}")


def prepare_cases(case_root: Path) -> dict[str, Any]:
    """Write the canonical BF16 operands and binary64-derived references."""
    for shape in SHAPES:
        rows, features = shape
        directory = case_root / f"{rows}x{features}"
        directory.mkdir(parents=True, exist_ok=True)
        x, gamma, dy = _fixed_inputs(shape)
        y, dx, dgamma = _reference(x, gamma, dy, EPSILON)
        values = {
            "x.bf16": x,
            "gamma.bf16": gamma,
            "dy.bf16": dy,
            "reference_y.bf16": y,
            "reference_dx.bf16": dx,
            "reference_dgamma.bf16": dgamma,
        }
        for name, value in values.items():
            (directory / name).write_bytes(_bf16_bytes(value))
    _verify_case_directory(case_root)
    return {"input_digests": INPUT_DIGESTS, "reference_digests": REFERENCE_DIGESTS}


def _heldout_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, features = REPRESENTATIVE_SHAPE
    specifications = (
        (-1.125, 0.5, (rows, features)),
        (0.375, 1.25, (features,)),
        (1.0, -0.75, (rows, features)),
    )
    return tuple(
        np.linspace(start, stop, math.prod(shape), dtype=np.float32).reshape(shape).astype(ml_dtypes.bfloat16)
        for start, stop, shape in specifications
    )  # type: ignore[return-value]


def _heldout_case_files() -> dict[str, tuple[int, str]]:
    rows, features = REPRESENTATIVE_SHAPE
    return {
        "x.bf16": (rows * features * 2, HELDOUT_INPUT_DIGESTS["x"]),
        "gamma.bf16": (features * 2, HELDOUT_INPUT_DIGESTS["gamma"]),
        "dy.bf16": (rows * features * 2, HELDOUT_INPUT_DIGESTS["dy"]),
        "reference_y.bf16": (rows * features * 2, HELDOUT_REFERENCE_DIGESTS["y"]),
        "reference_dx.bf16": (rows * features * 2, HELDOUT_REFERENCE_DIGESTS["dx"]),
        "reference_dgamma.bf16": (features * 2, HELDOUT_REFERENCE_DIGESTS["dgamma"]),
    }


def _verify_heldout_case(case_root: Path) -> None:
    for name, (size, digest) in _heldout_case_files().items():
        path = case_root / HELDOUT_PATTERN / name
        if not path.is_file():
            raise ValueError(f"case file missing: {path}")
        payload = path.read_bytes()
        if len(payload) != size:
            raise ValueError(f"case file size drifted: {path}")
        if _sha256_bytes(payload) != digest:
            raise ValueError(f"case file digest drifted: {path}")


def prepare_heldout_case(case_root: Path) -> dict[str, dict[str, str]]:
    """Write the predeclared representative input that was not used to tune the role limits."""
    directory = case_root / HELDOUT_PATTERN
    directory.mkdir(parents=True, exist_ok=True)
    x, gamma, dy = _heldout_inputs()
    y, dx, dgamma = _reference(x, gamma, dy, EPSILON)
    for name, value in {
        "x.bf16": x,
        "gamma.bf16": gamma,
        "dy.bf16": dy,
        "reference_y.bf16": y,
        "reference_dx.bf16": dx,
        "reference_dgamma.bf16": dgamma,
    }.items():
        (directory / name).write_bytes(_bf16_bytes(value))
    _verify_heldout_case(case_root)
    return {"input_digests": HELDOUT_INPUT_DIGESTS, "reference_digests": HELDOUT_REFERENCE_DIGESTS}


def _ordered_bf16(value: np.ndarray) -> np.ndarray:
    bits = value.view(np.uint16).astype(np.int32)
    return np.where((bits & 0x8000) != 0, 0x7FFF - (bits & 0x7FFF), 0x8000 + bits)


def epsilon_sensitivity(case_root: Path, shape: tuple[int, int]) -> dict[str, dict[str, float | int]]:
    """Measure the pinned epsilon reference against the same equations at epsilon zero."""
    _verify_case_directory(case_root)
    x, gamma, dy = _fixed_inputs(shape)
    zero_outputs = _reference(x, gamma, dy, 0.0)
    rows, features = shape
    shapes = ((rows, features), (rows, features), (features,))
    result = {}
    for role, output_shape, zero in zip(("y", "dx", "dgamma"), shapes, zero_outputs, strict=True):
        pinned = np.fromfile(case_root / f"{rows}x{features}" / f"reference_{role}.bf16", dtype=np.uint16)
        pinned = pinned.view(ml_dtypes.bfloat16).reshape(output_shape)
        result[role] = {
            "max_absolute_error": float(
                np.max(np.abs(pinned.astype(np.float32) - zero.astype(np.float32)), initial=0.0)
            ),
            "max_bfloat16_ulp": int(np.max(np.abs(_ordered_bf16(pinned) - _ordered_bf16(zero)), initial=0)),
        }
    return result


def execution_plan(
    te_runner: str,
    xla_runner: str,
    python: str,
    case_root: Path,
) -> ValidatedExecutionPlan:
    """Return the exact closed producer command matrix after case preflight."""
    _verify_case_directory(case_root)
    runs = []
    for shape in SHAPES:
        rows, features = shape
        scope = "architecture_fixture" if shape == ARCHITECTURE_SHAPE else "representative_acceptance"
        for boundary in BOUNDARIES:
            for producer in PRODUCERS:
                prefix = [te_runner] if producer == "transformer_engine_2_17" else [python, xla_runner]
                runs.append(
                    {
                        "shape": list(shape),
                        "scope": scope,
                        "boundary": boundary,
                        "producer": producer,
                        "argv": [
                            *prefix,
                            "--boundary",
                            boundary,
                            "--rows",
                            str(rows),
                            "--features",
                            str(features),
                            "--epsilon",
                            "1e-06",
                            "--case-directory",
                            str(case_root / f"{rows}x{features}"),
                            "--output",
                            str(case_root.parent / "raw" / f"{producer}-{rows}x{features}-{boundary}.json"),
                        ],
                    }
                )
    return ValidatedExecutionPlan(
        schema="shuttle_abi22_te_xla_rmsnorm_plan_v2",
        epsilon=EPSILON,
        excluded_capability=NORM_CONTRACT_BLOCKER,
        case_root=str(case_root),
        executables={"te": te_runner, "xla": xla_runner, "python": python},
        runs=runs,
    )


def execute_plan(plan: Mapping[str, Any], invoke: Callable[[list[str]], int]) -> None:
    """Execute every reviewed command once and stop on the first failed producer."""
    if not isinstance(plan, ValidatedExecutionPlan):
        raise ValueError("execution plan was not returned by canonical derivation")
    executables = plan.get("executables")
    if not isinstance(executables, Mapping) or set(executables) != {"te", "xla", "python"}:
        raise ValueError("execution plan executable identities drifted")
    case_root = Path(_string(plan.get("case_root"), "case_root"))
    expected = execution_plan(
        _string(executables["te"], "TE executable"),
        _string(executables["xla"], "XLA executable"),
        _string(executables["python"], "Python executable"),
        case_root,
    )
    if plan != expected:
        raise ValueError("execution plan differs from canonical 12-run derivation")
    for run in plan["runs"]:
        status = invoke(list(run["argv"]))
        if status != 0:
            raise RuntimeError(f"{run['producer']} producer exited with status {status}")


def qualification_plan(
    te_runner: str,
    xla_runner: str,
    python: str,
    case_root: Path,
) -> ValidatedQualificationPlan:
    """Return the repeated qualification and held-out matrix for the role policy."""
    _verify_case_directory(case_root)
    _verify_heldout_case(case_root)
    rows, features = REPRESENTATIVE_SHAPE
    runs = []
    for pattern in (QUALIFICATION_PATTERN, HELDOUT_PATTERN):
        directory = case_root / (f"{rows}x{features}" if pattern == QUALIFICATION_PATTERN else HELDOUT_PATTERN)
        for boundary in BOUNDARIES:
            for producer in PRODUCERS:
                prefix = [te_runner] if producer == "transformer_engine_2_17" else [python, xla_runner]
                for repeat in range(NUMERICAL_POLICY["repeat_count"]):
                    runs.append(
                        {
                            "shape": [rows, features],
                            "scope": "representative_policy_qualification",
                            "input_pattern": pattern,
                            "boundary": boundary,
                            "producer": producer,
                            "repeat": repeat,
                            "argv": [
                                *prefix,
                                "--boundary",
                                boundary,
                                "--rows",
                                str(rows),
                                "--features",
                                str(features),
                                "--epsilon",
                                "1e-06",
                                "--case-directory",
                                str(directory),
                                "--output",
                                str(
                                    case_root.parent
                                    / "raw"
                                    / f"policy-{producer}-{pattern}-{boundary}-repeat{repeat}.json"
                                ),
                            ],
                        }
                    )
    return ValidatedQualificationPlan(
        schema="shuttle_abi22_rmsnorm_policy_plan_v1",
        policy=NUMERICAL_POLICY,
        case_root=str(case_root),
        executables={"te": te_runner, "xla": xla_runner, "python": python},
        runs=runs,
    )


def execute_qualification_plan(plan: Mapping[str, Any], invoke: Callable[[list[str]], int]) -> None:
    """Execute only the exact canonical repeated qualification plan."""
    if not isinstance(plan, ValidatedQualificationPlan):
        raise ValueError("qualification plan was not returned by canonical derivation")
    executables = plan.get("executables")
    if not isinstance(executables, Mapping) or set(executables) != {"te", "xla", "python"}:
        raise ValueError("qualification plan executable identities drifted")
    expected = qualification_plan(
        _string(executables["te"], "TE executable"),
        _string(executables["xla"], "XLA executable"),
        _string(executables["python"], "Python executable"),
        Path(_string(plan.get("case_root"), "case_root")),
    )
    if plan != expected:
        raise ValueError("qualification plan differs from canonical repeated derivation")
    for run in plan["runs"]:
        status = invoke(list(run["argv"]))
        if status != 0:
            raise RuntimeError(f"{run['producer']} qualification producer exited with status {status}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value: object, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _load_document(value: Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Path):
        return json.loads(value.read_text())
    return dict(value)


def validate_build_manifest(path: Path | Mapping[str, Any], runner: Path) -> dict[str, Any]:
    """Validate the versioned external TE build and runner-source closure."""
    document = _load_document(path)
    if set(document) != {"schema", "runner", "provider", "build", "runtime_identity"}:
        raise ValueError("build manifest fields drifted")
    if document["schema"] != "shuttle_abi22_te_rmsnorm_runner_build_v2":
        raise ValueError("build manifest schema drifted")
    runner_record = document["runner"]
    if set(runner_record) != {"path", "sha256"}:
        raise ValueError("runner build record drifted")
    if _sha(runner_record["sha256"], "runner digest") != _sha256_bytes(runner.read_bytes()):
        raise ValueError("runner source digest drifted")
    if document["provider"] != {"distribution": "transformer_engine", "version": "2.17.0"}:
        raise ValueError("build provider drifted")
    build = document["build"]
    if build != {
        "language": "c++20",
        "cuda_architecture": "sm_90a",
        "libraries": ["transformer_engine", "cudart"],
    }:
        raise ValueError("build contract drifted")
    if document["runtime_identity"] != "shuttle_abi22_te_rmsnorm_runtime_v2_required":
        raise ValueError("runtime identity requirement drifted")
    return document


def validate_runtime_manifest(document: Mapping[str, Any], runner: Path) -> ValidatedTeRuntimeIdentity:
    """Bind the report identity to the runner and resolved TE library bytes."""
    if set(document) != {"schema", "runner_sha256", "transformer_engine_library"}:
        raise ValueError("runtime manifest fields drifted")
    if document["schema"] != "shuttle_abi22_te_rmsnorm_runtime_v2":
        raise ValueError("runtime manifest schema drifted")
    runner_sha = _sha(document["runner_sha256"], "runner_sha256")
    if runner_sha != _sha256_bytes(runner.read_bytes()):
        raise ValueError("runner digest differs from resolved bytes")
    library_record = document["transformer_engine_library"]
    if not isinstance(library_record, Mapping) or set(library_record) != {"path", "sha256"}:
        raise ValueError("runtime library record drifted")
    library = Path(_string(library_record["path"], "runtime library path"))
    if not library.is_file():
        raise ValueError("runtime library path is not a regular file")
    library_sha = _sha(library_record["sha256"], "library digest")
    if library_sha != _sha256_bytes(library.read_bytes()):
        raise ValueError("library digest differs from resolved bytes")
    return ValidatedTeRuntimeIdentity(
        {
            "distribution": "transformer_engine",
            "version": "2.17.0",
            "source_commit": TE_SOURCE_COMMIT,
            "runner_sha256": runner_sha,
            "library_sha256": library_sha,
        },
        _RUNTIME_IDENTITY_SEAL,
    )


def _expected_inputs(shape: tuple[int, int], boundary: str) -> dict[str, str]:
    roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
    return {role: INPUT_DIGESTS[shape][role] for role in roles}


def _expected_execution(boundary: str) -> dict[str, Any]:
    return {"call_sequence": list(CALL_SEQUENCES[boundary]), "rsigma_state": "invocation_local_not_public"}


def _output_shape(shape: tuple[int, int], role: str) -> tuple[int, ...]:
    return (shape[1],) if role == "dgamma" else shape


def _read_bf16_file(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"raw output is missing: {path}")
    value = np.fromfile(path, dtype=np.uint16)
    if value.size != math.prod(shape):
        raise ValueError(f"raw output size drifted: {path}")
    return value.view(ml_dtypes.bfloat16).reshape(shape)


def _error_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    return {
        "max_absolute_error": float(
            np.max(np.abs(actual.astype(np.float32) - reference.astype(np.float32)), initial=0.0)
        ),
        "max_bfloat16_ulp": int(np.max(np.abs(_ordered_bf16(actual) - _ordered_bf16(reference)), initial=0)),
    }


def load_te_raw_cell(
    path: Path,
    case_directory: Path,
    *,
    expected_input_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Convert one TE runner observation into a validated-report cell."""
    raw = json.loads(path.read_text())
    required = {
        "schema",
        "shape",
        "boundary",
        "epsilon",
        "warmup_invocations",
        "raw_host_blocking_ns",
        "raw_cuda_event_ns",
        "output_files",
        "workspace_bytes",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("raw TE cell fields drifted")
    if raw["schema"] != "shuttle_abi22_te_rmsnorm_raw_cell_v2" or raw["epsilon"] != EPSILON:
        raise ValueError("raw TE schema or epsilon drifted")
    shape = tuple(raw["shape"])
    boundary = raw["boundary"]
    if shape not in SHAPES or boundary not in BOUNDARIES:
        raise ValueError("raw TE coordinate is outside the matrix")
    expected_directory = f"{shape[0]}x{shape[1]}" if expected_input_digests is None else HELDOUT_PATTERN
    if case_directory.name != expected_directory:
        raise ValueError("raw TE case directory does not match its shape")
    _verify_case_directory(case_directory.parent)
    roles = OUTPUT_ROLES[boundary]
    output_files = raw["output_files"]
    if not isinstance(output_files, Mapping) or tuple(sorted(output_files)) != tuple(sorted(roles)):
        raise ValueError("raw TE output roles drifted")
    outputs = []
    for role in roles:
        output_shape = _output_shape(shape, role)
        actual = _read_bf16_file(Path(_string(output_files[role], f"raw {role} path")), output_shape)
        reference = _read_bf16_file(case_directory / f"reference_{role}.bf16", output_shape)
        outputs.append(
            {
                "role": role,
                "digest": _sha256_bytes(_bf16_bytes(actual)),
                "reference_digest": _sha256_bytes(_bf16_bytes(reference)),
                **_error_metrics(actual, reference),
            }
        )
    host = _positive_samples(raw["raw_host_blocking_ns"], "host", permitted_none=False)
    cuda = _positive_samples(raw["raw_cuda_event_ns"], "CUDA-event", permitted_none=False)
    assert host is not None and cuda is not None
    if raw["warmup_invocations"] != WARMUP_INVOCATIONS:
        raise ValueError("raw TE warmup drifted")
    workspace = raw["workspace_bytes"]
    if (
        not isinstance(workspace, Mapping)
        or set(workspace) != {"forward", "backward"}
        or any(type(value) is not int or value < 0 for value in workspace.values())
    ):
        raise ValueError("raw TE workspace metadata drifted")
    return {
        "shape": list(shape),
        "scope": "architecture_fixture" if shape == ARCHITECTURE_SHAPE else "representative_acceptance",
        "boundary": boundary,
        "input_digests": (
            _expected_inputs(shape, boundary)
            if expected_input_digests is None
            else {
                role: expected_input_digests[role]
                for role in (("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy"))
            }
        ),
        "execution": _expected_execution(boundary),
        "outputs": outputs,
        "timing": {
            "warmup_invocations": WARMUP_INVOCATIONS,
            "raw_host_blocking_ns": host,
            "median_host_blocking_ns": statistics.median(host),
            "raw_cuda_event_ns": cuda,
        },
    }


def assemble_report(
    producer: str,
    cells: list[dict[str, Any]],
    provider_identity: Mapping[str, str],
    device: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the six cell observations into the versioned report envelope."""
    report = {
        "schema": "shuttle_abi22_te_xla_rmsnorm_oracle_v2",
        "protocol_mode": "matched_host_blocking_v2",
        "producer": producer,
        "epsilon": EPSILON,
        "reference": {
            "implementation": "independent_numpy_binary64_closed_form_then_bfloat16",
            "input_generator": "numpy_linspace_float32_then_bfloat16_v2",
            "epsilon": EPSILON,
        },
        "provider_identity": dict(provider_identity),
        "device": dict(device),
        "capabilities": {
            "rmsnorm": list(BOUNDARIES),
            "composed_semantics": "forward_then_backward_with_invocation_local_rsigma",
            "norm_contract": {"status": "unavailable", "blocker": NORM_CONTRACT_BLOCKER},
        },
        "cells": cells,
    }
    return validate_report(
        report,
        expected_producer=producer,
        expected_te_identity=(
            provider_identity
            if producer == "transformer_engine_2_17" and isinstance(provider_identity, ValidatedTeRuntimeIdentity)
            else None
        ),
    )


def _validate_identity(
    identity: object,
    producer: str,
    expected_te_identity: ValidatedTeRuntimeIdentity | None,
) -> None:
    if not isinstance(identity, Mapping):
        raise ValueError("provider_identity must be an object")
    if producer == "transformer_engine_2_17":
        fields = {"distribution", "version", "source_commit", "runner_sha256", "library_sha256"}
        if set(identity) != fields:
            raise ValueError("provider identity fields drifted")
        if identity.get("distribution") != "transformer_engine":
            raise ValueError("provider identity distribution drifted")
        if identity.get("version") != "2.17.0":
            raise ValueError("provider identity version drifted")
        if identity.get("source_commit") != TE_SOURCE_COMMIT:
            raise ValueError("provider identity source_commit drifted")
        _sha(identity.get("runner_sha256"), "provider identity runner_sha256")
        _sha(identity.get("library_sha256"), "provider identity library_sha256")
        if not isinstance(expected_te_identity, ValidatedTeRuntimeIdentity):
            raise ValueError("provider identity lacks a validated runtime manifest")
        if dict(identity) != dict(expected_te_identity):
            for field in ("version", "source_commit", "runner_sha256", "library_sha256"):
                if identity.get(field) != (expected_te_identity or {}).get(field):
                    raise ValueError(f"provider identity {field} drifted")
            raise ValueError("provider identity drifted")
        return
    expected = {"jax_version": "0.10.1", "jaxlib_version": "0.10.1", "xla_revision": XLA_REVISION}
    if dict(identity) != expected:
        for field, value in expected.items():
            if identity.get(field) != value:
                raise ValueError(f"provider identity {field} drifted")
        raise ValueError("provider identity drifted")


def _positive_samples(value: object, name: str, *, permitted_none: bool) -> list[int] | None:
    if value is None and permitted_none:
        return None
    if not isinstance(value, list) or len(value) != MEASURED_INVOCATIONS:
        raise ValueError(f"{name} samples must contain exactly 20 values")
    if any(type(sample) is not int or sample <= 0 for sample in value):
        raise ValueError(f"{name} samples must be positive integer nanoseconds")
    return value


def validate_report(
    document: Mapping[str, Any],
    *,
    expected_producer: str,
    expected_te_identity: ValidatedTeRuntimeIdentity | None = None,
) -> dict[str, Any]:
    """Validate one complete six-cell producer report."""
    if expected_producer not in PRODUCERS:
        raise ValueError("expected producer is outside the protocol")
    required = {
        "schema",
        "protocol_mode",
        "producer",
        "epsilon",
        "reference",
        "provider_identity",
        "device",
        "capabilities",
        "cells",
    }
    if set(document) != required:
        raise ValueError("report fields drifted")
    if document["schema"] != "shuttle_abi22_te_xla_rmsnorm_oracle_v2":
        raise ValueError("report schema drifted")
    if document["protocol_mode"] != "matched_host_blocking_v2":
        raise ValueError("protocol mode drifted")
    if document["producer"] != expected_producer:
        raise ValueError("producer drifted")
    if type(document["epsilon"]) is not float or document["epsilon"] != EPSILON:
        raise ValueError("epsilon drifted")
    if document["reference"] != {
        "implementation": "independent_numpy_binary64_closed_form_then_bfloat16",
        "input_generator": "numpy_linspace_float32_then_bfloat16_v2",
        "epsilon": EPSILON,
    }:
        raise ValueError("reference contract drifted")
    _validate_identity(document["provider_identity"], expected_producer, expected_te_identity)
    device = document["device"]
    if not isinstance(device, Mapping) or set(device) != {"model", "uuid", "compute_capability"}:
        raise ValueError("device identity drifted")
    for field in device:
        _string(device[field], f"device {field}")
    if document["capabilities"] != {
        "rmsnorm": list(BOUNDARIES),
        "composed_semantics": "forward_then_backward_with_invocation_local_rsigma",
        "norm_contract": {"status": "unavailable", "blocker": NORM_CONTRACT_BLOCKER},
    }:
        raise ValueError("norm_contract capability or RMSNorm boundary contract drifted")
    cells = document["cells"]
    if not isinstance(cells, list) or len(cells) != len(SHAPES) * len(BOUNDARIES):
        raise ValueError("cell matrix drifted")
    expected_coordinates = [(shape, boundary) for shape in SHAPES for boundary in BOUNDARIES]
    for cell, (shape, boundary) in zip(cells, expected_coordinates, strict=True):
        if not isinstance(cell, Mapping):
            raise ValueError("cell matrix contains a non-object")
        if tuple(cell.get("shape", ())) != shape or cell.get("boundary") != boundary:
            raise ValueError("cell matrix coordinate drifted")
        scope = "architecture_fixture" if shape == ARCHITECTURE_SHAPE else "representative_acceptance"
        if cell.get("scope") != scope:
            raise ValueError("cell scope drifted")
        if cell.get("input_digests") != _expected_inputs(shape, boundary):
            raise ValueError("cell input digest mismatch in input provenance")
        if cell.get("execution") != _expected_execution(boundary):
            raise ValueError("cell call sequence or rsigma state drifted")
        outputs = cell.get("outputs")
        roles = OUTPUT_ROLES[boundary]
        if not isinstance(outputs, list) or [output.get("role") for output in outputs] != list(roles):
            raise ValueError("cell output roles drifted")
        for output, role in zip(outputs, roles, strict=True):
            _sha(output.get("digest"), "output digest")
            reference_digest = _sha(output.get("reference_digest"), "reference digest")
            if reference_digest != REFERENCE_DIGESTS[shape][role]:
                raise ValueError("composed reference provenance drifted")
            absolute = output.get("max_absolute_error")
            ulp = output.get("max_bfloat16_ulp")
            if type(absolute) not in (int, float) or not math.isfinite(float(absolute)) or absolute < 0:
                raise ValueError("max_absolute_error must be nonnegative and finite")
            limits = ACCEPTANCE_LIMITS[role]
            if shape == REPRESENTATIVE_SHAPE and float(absolute) > limits["max_absolute_error"]:
                raise ValueError("max_absolute_error exceeds acceptance limit")
            if type(ulp) is not int or ulp < 0:
                raise ValueError("max_bfloat16_ulp must be a nonnegative integer")
            if shape == REPRESENTATIVE_SHAPE and ulp > limits["max_bfloat16_ulp"]:
                raise ValueError("max_bfloat16_ulp exceeds acceptance limit")
        timing = cell.get("timing")
        if not isinstance(timing, Mapping) or timing.get("warmup_invocations") != WARMUP_INVOCATIONS:
            raise ValueError("timing warmup contract drifted")
        host = _positive_samples(timing.get("raw_host_blocking_ns"), "host", permitted_none=False)
        assert host is not None
        if timing.get("median_host_blocking_ns") != statistics.median(host):
            raise ValueError("median_host_blocking_ns does not match raw samples")
        cuda = _positive_samples(
            timing.get("raw_cuda_event_ns"),
            "CUDA-event",
            permitted_none=expected_producer == "ordinary_jax_xla",
        )
        if expected_producer == "ordinary_jax_xla" and cuda is not None:
            raise ValueError("XLA report must not contain CUDA-event samples")
    return dict(document)


def validate_qualification_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate both repeated representative patterns against the frozen role policy."""
    document = dict(value)
    if set(document) not in ({"schema", "cells"}, {"schema", "policy", "cells"}):
        raise ValueError("qualification evidence fields drifted")
    if document.get("schema") != "shuttle_abi22_rmsnorm_policy_evidence_v1":
        raise ValueError("qualification evidence schema drifted")
    if "policy" in document and document["policy"] != NUMERICAL_POLICY:
        raise ValueError("qualification numerical policy drifted")
    cells = document.get("cells")
    expected_coordinates = [
        (producer, pattern, boundary, repeat)
        for producer in PRODUCERS
        for pattern in (QUALIFICATION_PATTERN, HELDOUT_PATTERN)
        for boundary in BOUNDARIES
        for repeat in range(NUMERICAL_POLICY["repeat_count"])
    ]
    if not isinstance(cells, list) or len(cells) != len(expected_coordinates):
        raise ValueError("qualification cell matrix drifted")
    digests: dict[tuple[str, str, str, int, str], str] = {}
    for cell, (producer, pattern, boundary, repeat) in zip(cells, expected_coordinates, strict=True):
        if not isinstance(cell, Mapping):
            raise ValueError("qualification cell is not an object")
        if (
            cell.get("producer") != producer
            or cell.get("input_pattern") != pattern
            or cell.get("boundary") != boundary
            or cell.get("repeat") != repeat
            or tuple(cell.get("shape", ())) != REPRESENTATIVE_SHAPE
            or cell.get("scope") != "representative_policy_qualification"
        ):
            raise ValueError("qualification coordinate drifted")
        if pattern == QUALIFICATION_PATTERN:
            input_digests = INPUT_DIGESTS[REPRESENTATIVE_SHAPE]
            reference_digests = REFERENCE_DIGESTS[REPRESENTATIVE_SHAPE]
        else:
            input_digests = HELDOUT_INPUT_DIGESTS
            reference_digests = HELDOUT_REFERENCE_DIGESTS
        if cell.get("input_digests") != {
            role: input_digests[role] for role in (("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy"))
        }:
            raise ValueError("qualification input provenance drifted")
        if cell.get("execution") != _expected_execution(boundary):
            raise ValueError("qualification call sequence drifted")
        outputs = cell.get("outputs")
        roles = OUTPUT_ROLES[boundary]
        if not isinstance(outputs, list) or [output.get("role") for output in outputs] != list(roles):
            raise ValueError("qualification output role drifted")
        for output, role in zip(outputs, roles, strict=True):
            digest = _sha(output.get("digest"), "qualification output digest")
            if _sha(output.get("reference_digest"), "qualification reference digest") != reference_digests[role]:
                raise ValueError("qualification reference digest drifted")
            absolute = output.get("max_absolute_error")
            ulp = output.get("max_bfloat16_ulp")
            if type(absolute) not in (int, float) or not math.isfinite(float(absolute)) or absolute < 0:
                raise ValueError("qualification max_absolute_error must be finite and nonnegative")
            if type(ulp) is not int or ulp < 0:
                raise ValueError("qualification max_bfloat16_ulp must be nonnegative")
            limits = ACCEPTANCE_LIMITS[role]
            if float(absolute) > limits["max_absolute_error"]:
                raise ValueError("qualification max_absolute_error exceeds role limit")
            if ulp > limits["max_bfloat16_ulp"]:
                raise ValueError("qualification max_bfloat16_ulp exceeds role limit")
            digests[(producer, pattern, boundary, repeat, role)] = digest
        timing = cell.get("timing")
        if not isinstance(timing, Mapping) or timing.get("warmup_invocations") != WARMUP_INVOCATIONS:
            raise ValueError("qualification timing drifted")
        host = _positive_samples(timing.get("raw_host_blocking_ns"), "qualification host", permitted_none=False)
        assert host is not None
        if timing.get("median_host_blocking_ns") != statistics.median(host):
            raise ValueError("qualification host median drifted")
        cuda = _positive_samples(
            timing.get("raw_cuda_event_ns"),
            "qualification CUDA-event",
            permitted_none=producer == "ordinary_jax_xla",
        )
        if (producer == "transformer_engine_2_17") != (cuda is not None):
            raise ValueError("qualification CUDA-event provider contract drifted")
    for producer in PRODUCERS:
        for pattern in (QUALIFICATION_PATTERN, HELDOUT_PATTERN):
            for boundary in BOUNDARIES:
                for role in OUTPUT_ROLES[boundary]:
                    if (
                        digests[(producer, pattern, boundary, 0, role)]
                        != digests[(producer, pattern, boundary, 1, role)]
                    ):
                        raise ValueError("qualification repeat output digest drifted")
    return {
        "schema": "shuttle_abi22_rmsnorm_policy_evidence_v1",
        "policy": NUMERICAL_POLICY,
        "cells": [dict(cell) for cell in cells],
    }


def build_matched_acceptance(
    te_report: Mapping[str, Any],
    xla_report: Mapping[str, Any],
    *,
    qualification: Mapping[str, Any],
    expected_te_identity: ValidatedTeRuntimeIdentity,
) -> dict[str, Any]:
    """Build the representative numerical acceptance and clock-matched timing summary."""
    for report in (te_report, xla_report):
        cells = report.get("cells", [])
        if not any(tuple(cell.get("shape", ())) == REPRESENTATIVE_SHAPE for cell in cells):
            raise ValueError("representative cells are missing")
    te = validate_report(
        te_report,
        expected_producer="transformer_engine_2_17",
        expected_te_identity=expected_te_identity,
    )
    xla = validate_report(xla_report, expected_producer="ordinary_jax_xla")
    qualified = validate_qualification_evidence(qualification)
    if te["device"] != xla["device"]:
        raise ValueError("producer device identities differ")
    ratios = {}
    for boundary in BOUNDARIES:
        te_cell = next(
            cell for cell in te["cells"] if tuple(cell["shape"]) == REPRESENTATIVE_SHAPE and cell["boundary"] == boundary
        )
        xla_cell = next(
            cell
            for cell in xla["cells"]
            if tuple(cell["shape"]) == REPRESENTATIVE_SHAPE and cell["boundary"] == boundary
        )
        if te_cell["input_digests"] != xla_cell["input_digests"]:
            raise ValueError("producer input digests differ")
        ratios[boundary] = te_cell["timing"]["median_host_blocking_ns"] / xla_cell["timing"]["median_host_blocking_ns"]
    return {
        "schema": "shuttle_abi22_te_xla_rmsnorm_acceptance_v2",
        "status": "representative_numerics_accepted",
        "accepted_shape": list(REPRESENTATIVE_SHAPE),
        "accepted_boundaries": list(BOUNDARIES),
        "excluded_architecture_fixture_shape": list(ARCHITECTURE_SHAPE),
        "numerical_policy": NUMERICAL_POLICY,
        "qualification": qualified,
        "performance": {
            "status": "reported_without_threshold",
            "clock": "host_blocking_ns",
            "median_ratio_te_over_xla": ratios,
        },
        "norm_contract": {"status": "unavailable", "blocker": NORM_CONTRACT_BLOCKER},
    }
