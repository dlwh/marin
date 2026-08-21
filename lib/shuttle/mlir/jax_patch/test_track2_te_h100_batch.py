# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contract for the sealed Track 2 TE/XLA H100 batch."""

import hashlib
import json
import statistics
import zipfile
from pathlib import Path

import abi22_xla_rmsnorm_oracle_v2 as xla_oracle
import pytest
from abi22_te_rmsnorm_oracle_v2 import (
    BOUNDARIES,
    CALL_SEQUENCES,
    INPUT_DIGESTS,
    OUTPUT_ROLES,
    REFERENCE_DIGESTS,
    SHAPES,
)
from track2_te_h100_batch import (
    BASE_IMAGE,
    NVTE_CUDA_INCLUDE_DIR,
    RUNTIME_LIBRARY_PATHS,
    SHUTTLE_BLOCKER,
    TE_SOURCE_COMMIT,
    TE_SOURCE_TREE,
    TE_WHEEL_SHA256,
    TE_WHEEL_URL,
    BatchPaths,
    _invoke_oracle,
    _validate_link_closure,
    _validated_cuda_include,
    _validated_runtime_libraries,
    _validated_xla_raw_cell,
    batch_plan,
    canonical_json,
    execute_batch,
    execute_oracle,
    runtime_environment,
    validate_source_manifest,
    validate_te_hydration,
)

XLA_OBSERVED_RUNTIME = {
    "backend": "gpu",
    "device_count": 1,
    "device_id": 0,
    "device_kind": "NVIDIA H100 80GB HBM3",
    "jax_git_commit": "619764c15117fbefc4ba13ab941871cb514c23f6",
    "jax_version": "0.10.1",
    "jaxlib_version": "0.10.1",
    "platform_version": "CUDA 13.2.0",
}
RUNTIME_LIBRARY_IDENTITIES = {
    role: {
        "requested_path": str(path),
        "resolved_path": str(path),
        "sha256": hashlib.sha256(role.encode()).hexdigest(),
        "soname": path.name,
    }
    for role, path in RUNTIME_LIBRARY_PATHS.items()
}
CUDA_HEADER_IDENTITY = {
    "include_directory": str(NVTE_CUDA_INCLUDE_DIR),
    "header_path": str(NVTE_CUDA_INCLUDE_DIR / "cuda_runtime_api.h"),
    "header_sha256": hashlib.sha256(b"cuda-runtime-header").hexdigest(),
}


def _manifest(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    manifest = root / ".shuttle-track2-te-source-manifest.json"
    manifest.write_bytes(
        canonical_json(
            {
                "commit": "1" * 40,
                "entries": [
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                ],
                "schema": "shuttle_track2_te_h100_source_manifest_v1",
                "tree": "2" * 40,
            }
        )
    )
    return manifest


def test_batch_plan_pins_te_wheel_source_build_and_exact_v2_matrix(tmp_path: Path) -> None:
    workspace = tmp_path / "work"

    plan = batch_plan(workspace=workspace, runner_repository=Path("/app"))
    phases = plan["phases"]

    assert plan["schema"] == "shuttle_track2_te_h100_batch_plan_v1"
    assert plan["base_image"] == BASE_IMAGE
    assert plan["hardware"] == {"gpu_count": 1, "gpu_model": "H100"}
    assert plan["oracle"] == {
        "schema": "shuttle_abi22_te_xla_rmsnorm_plan_v2",
        "epsilon": 1e-6,
        "shapes": [[7, 13], [2048, 4096]],
        "representative_shape": [2048, 4096],
        "boundaries": ["forward", "backward_recompute", "composed"],
        "producer_order": ["transformer_engine_2_17", "ordinary_jax_xla"],
        "run_count": 12,
    }
    assert SHUTTLE_BLOCKER == (
        "reviewed_dbff57e3_rmsnorm_canary_uses_epsilon_1e-6_but_does_not_admit_"
        "representative_2048x4096_without_new_harness_authorization"
    )
    assert plan["shuttle"] == {"status": "pending", "blocker": SHUTTLE_BLOCKER}
    assert [phase["name"] for phase in phases] == [
        "device",
        "te_init",
        "te_remote",
        "te_fetch",
        "te_checkout",
        "te_wheel_fetch",
        "te_wheel_extract",
        "te_runner_build",
        "te_runner_link_check",
    ]
    assert phases[3]["argv"] == ["git", "fetch", "--depth=1", "origin", TE_SOURCE_COMMIT]
    assert phases[4]["argv"] == ["git", "checkout", "--detach", "FETCH_HEAD"]
    assert phases[5]["argv"] == [
        "curl",
        "--fail",
        "--location",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--output",
        str(BatchPaths.under(workspace).te_wheel),
        TE_WHEEL_URL,
    ]
    compile_argv = phases[7]["argv"]
    assert compile_argv[:4] == ["/usr/bin/g++", "-std=c++20", "-O3", "-Wall"]
    assert "-Werror" in compile_argv
    assert "-Wl,--allow-shlib-undefined" in compile_argv
    assert set(RUNTIME_LIBRARY_PATHS) == {"cublas", "cublas_lt", "cudnn", "nccl", "cudart", "nvrtc"}
    assert RUNTIME_LIBRARY_PATHS["nvrtc"] == (
        Path("/opt/h100-evidence-runtime/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.13")
    )
    libraries = [str(path) for path in RUNTIME_LIBRARY_PATHS.values()]
    no_as_needed = compile_argv.index("-Wl,--no-as-needed")
    assert compile_argv[no_as_needed : no_as_needed + len(libraries) + 2] == [
        "-Wl,--no-as-needed",
        *libraries,
        "-Wl,--as-needed",
    ]
    te_include = str(BatchPaths.under(workspace).te_source / "transformer_engine/common/include")
    assert compile_argv[compile_argv.index("-isystem") : compile_argv.index("-isystem") + 2] == [
        "-isystem",
        te_include,
    ]
    assert f"-I{te_include}" not in compile_argv
    assert f"-L{BatchPaths.under(workspace).te_library.parent}" in compile_argv
    assert f"-Wl,-rpath,{BatchPaths.under(workspace).te_library.parent}" in compile_argv
    assert "-ltransformer_engine" in compile_argv
    assert "-lcudart" in compile_argv
    assert str(BatchPaths.under(workspace).te_runner) == compile_argv[-1]
    assert phases[8]["argv"] == ["ldd", str(BatchPaths.under(workspace).te_runner)]
    assert TE_WHEEL_SHA256 == "1c8e6627cf02358201f513ab902c7b4f82f179214bd775edc97ce0f2001d82da"
    assert TE_SOURCE_COMMIT == "2e559f062497bef768dfbe9d7e45548fadeca80a"
    assert TE_SOURCE_TREE == "c722c6cd46cc59a0dbabadd846af3f45b15c1a95"
    assert all("shuttle" not in phase["name"] for phase in phases)


def test_hydration_binds_source_tree_wheel_and_resolved_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = BatchPaths.under(tmp_path)
    paths.te_source.mkdir(parents=True)
    (paths.te_source / "build_tools").mkdir()
    (paths.te_source / "build_tools/VERSION.txt").write_text("2.17.0\n")
    paths.te_wheel.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(paths.te_wheel, "w") as archive:
        archive.writestr("transformer_engine/wheel_lib/libtransformer_engine.so", b"library")
    paths.te_library.parent.mkdir(parents=True)
    paths.te_library.write_bytes(b"library")

    observed = {
        "commit": TE_SOURCE_COMMIT,
        "tree": TE_SOURCE_TREE,
        "status": "",
    }
    monkeypatch.setattr("track2_te_h100_batch._observed_te_source", lambda _: observed)
    monkeypatch.setattr(
        "track2_te_h100_batch._file_sha256",
        lambda path: TE_WHEEL_SHA256 if path == paths.te_wheel else hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    identity = validate_te_hydration(paths)

    assert identity == {
        "source_commit": TE_SOURCE_COMMIT,
        "source_tree": TE_SOURCE_TREE,
        "version": "2.17.0",
        "wheel_sha256": TE_WHEEL_SHA256,
        "library_path": str(paths.te_library),
        "library_sha256": hashlib.sha256(b"library").hexdigest(),
    }
    monkeypatch.setenv("LD_BIND_NOW", "1")
    environment = runtime_environment(paths)
    assert environment["LD_LIBRARY_PATH"].split(":")[0] == str(paths.te_library.parent)
    assert "LD_BIND_NOW" not in environment
    assert environment["NVTE_CUDA_INCLUDE_DIR"] == (
        "/opt/h100-evidence-runtime/lib/python3.12/site-packages/nvidia/cuda_runtime/include"
    )
    for field, value in (("commit", "0" * 40), ("tree", "0" * 40), ("status", "dirty")):
        mutated = dict(observed)
        mutated[field] = value
        monkeypatch.setattr("track2_te_h100_batch._observed_te_source", lambda _, value=mutated: value)
        with pytest.raises(RuntimeError, match="Transformer Engine source"):
            validate_te_hydration(paths)
    monkeypatch.setattr("track2_te_h100_batch._observed_te_source", lambda _: observed)
    monkeypatch.setattr("track2_te_h100_batch._file_sha256", lambda _: "0" * 64)
    with pytest.raises(RuntimeError, match="wheel SHA-256"):
        validate_te_hydration(paths)
    monkeypatch.setattr(
        "track2_te_h100_batch._file_sha256",
        lambda path: TE_WHEEL_SHA256 if path == paths.te_wheel else hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    paths.te_library.write_bytes(b"ambient-library")
    with pytest.raises(RuntimeError, match="extracted Transformer Engine library"):
        validate_te_hydration(paths)


def test_runtime_libraries_bind_resolved_paths_sonames_and_bytes(tmp_path: Path) -> None:
    root = tmp_path / "approved"
    root.mkdir()
    library_paths = {}
    sonames = {}
    for index, role in enumerate(RUNTIME_LIBRARY_PATHS):
        target = root / f"{role}.so.{index}"
        target.write_bytes(f"library-{role}".encode())
        link = root / f"{role}.so"
        link.symlink_to(target.name)
        library_paths[role] = link
        sonames[target] = f"lib{role}.so.{index}"

    identities = _validated_runtime_libraries(
        library_paths,
        approved_roots=(root,),
        soname_reader=lambda path: sonames[path],
    )

    assert set(identities) == set(RUNTIME_LIBRARY_PATHS)
    for role, identity in identities.items():
        target = library_paths[role].resolve()
        assert identity == {
            "requested_path": str(library_paths[role]),
            "resolved_path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "soname": sonames[target],
        }

    outside = tmp_path / "outside.so"
    outside.write_bytes(b"outside")
    library_paths["nvrtc"].unlink()
    library_paths["nvrtc"].symlink_to(outside)
    with pytest.raises(RuntimeError, match="approved runtime root"):
        _validated_runtime_libraries(
            library_paths,
            approved_roots=(root,),
            soname_reader=lambda path: path.name,
        )


def test_cuda_runtime_headers_are_bound_to_the_nvrtc_package_generation(tmp_path: Path) -> None:
    include = tmp_path / "nvidia/cuda_runtime/include"
    include.mkdir(parents=True)
    header = include / "cuda_runtime_api.h"
    header.write_bytes(b"sealed cuda runtime header")

    identity = _validated_cuda_include(include, approved_root=tmp_path / "nvidia")

    assert identity == {
        "include_directory": str(include),
        "header_path": str(header),
        "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
    }
    assert NVTE_CUDA_INCLUDE_DIR.name == "include"

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "cuda_runtime_api.h").write_bytes(b"outside")
    with pytest.raises(RuntimeError, match="approved runtime root"):
        _validated_cuda_include(outside, approved_root=tmp_path / "nvidia")


def test_link_closure_rejects_any_unresolved_or_substituted_dependency(tmp_path: Path) -> None:
    libraries = {}
    lines = ["libtransformer_engine.so => /tmp/runtime/libtransformer_engine.so (0x1)"]
    for role in RUNTIME_LIBRARY_PATHS:
        path = tmp_path / f"lib{role}.so"
        path.write_bytes(f"library-{role}".encode())
        soname = f"lib{role}.so.1"
        libraries[role] = {
            "requested_path": str(path),
            "resolved_path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "soname": soname,
        }
        lines.append(f"{soname} => {path} (0x1)")
    log = tmp_path / "ldd.log"
    log.write_text("\n".join(lines) + "\n")

    _validate_link_closure(log, libraries)

    log.write_text("libnccl.so.2 => not found\n")
    with pytest.raises(RuntimeError, match="runtime dependency closure"):
        _validate_link_closure(log, libraries)

    substituted = tmp_path / "substituted.so"
    substituted.write_bytes(b"substituted")
    log.write_text("\n".join([*lines[:-1], f"{libraries[role]['soname']} => {substituted} (0x1)"]) + "\n")
    with pytest.raises(RuntimeError, match="runtime dependency closure"):
        _validate_link_closure(log, libraries)


def _valid_cell(shape: tuple[int, int], boundary: str, *, te: bool) -> dict:
    roles = OUTPUT_ROLES[boundary]
    input_roles = ("x", "gamma") if boundary == "forward" else ("x", "gamma", "dy")
    scale = 1 if te else 2
    samples = [scale * (100_000 + index * 1_000) for index in range(20)]
    cell = {
        "shape": list(shape),
        "scope": "architecture_fixture" if shape == SHAPES[0] else "representative_acceptance",
        "boundary": boundary,
        "input_digests": {role: INPUT_DIGESTS[shape][role] for role in input_roles},
        "execution": {
            "call_sequence": list(CALL_SEQUENCES[boundary]),
            "rsigma_state": "invocation_local_not_public",
        },
        "outputs": [
            {
                "role": role,
                "digest": REFERENCE_DIGESTS[shape][role],
                "reference_digest": REFERENCE_DIGESTS[shape][role],
                "max_absolute_error": 0.0,
                "max_bfloat16_ulp": 0,
            }
            for role in roles
        ],
        "timing": {
            "warmup_invocations": 10,
            "raw_host_blocking_ns": samples,
            "median_host_blocking_ns": statistics.median(samples),
            "raw_cuda_event_ns": [50_000 + index * 100 for index in range(20)] if te else None,
        },
    }
    if not te:
        cell["observed_runtime"] = dict(XLA_OBSERVED_RUNTIME)
    return cell


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "cpu"),
        ("device_count", 2),
        ("device_id", 1),
        ("device_kind", "NVIDIA L4"),
        ("jax_git_commit", "0" * 40),
        ("jax_version", "0.10.0"),
        ("jaxlib_version", "0.10.0"),
        ("platform_version", ""),
    ],
)
def test_xla_raw_cell_requires_observed_h100_runtime(field: str, value: object) -> None:
    cell = _valid_cell(SHAPES[1], "forward", te=False)
    cell["observed_runtime"][field] = value

    with pytest.raises(ValueError, match=r"ordinary-JAX.*runtime|H100"):
        _validated_xla_raw_cell(
            cell,
            expected_device={"model": "NVIDIA H100 80GB HBM3", "uuid": "GPU-test", "compute_capability": "9.0"},
        )


def test_xla_raw_cell_returns_claimed_identity_from_observed_runtime() -> None:
    cell = _valid_cell(SHAPES[1], "forward", te=False)

    validated, runtime = _validated_xla_raw_cell(
        cell,
        expected_device={"model": "NVIDIA H100 80GB HBM3", "uuid": "GPU-test", "compute_capability": "9.0"},
    )

    assert "observed_runtime" not in validated
    assert runtime == XLA_OBSERVED_RUNTIME


def test_xla_producer_observes_one_h100_and_its_loaded_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        platform_version = "CUDA 13.2.0"

    class Device:
        client = Client()
        device_kind = "NVIDIA H100 80GB HBM3"
        id = 0

    monkeypatch.setattr(xla_oracle.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(xla_oracle.jax, "devices", lambda _: [Device()])

    assert xla_oracle._observed_runtime() == XLA_OBSERVED_RUNTIME

    monkeypatch.setattr(xla_oracle.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="GPU backend"):
        xla_oracle._observed_runtime()


def _fake_oracle_invoke(paths: BatchPaths, calls: list[list[str]], argv: list[str]) -> int:
    calls.append(argv)
    argument = {value[2:]: argv[index + 1] for index, value in enumerate(argv[:-1]) if value.startswith("--")}
    shape = (int(argument["rows"]), int(argument["features"]))
    boundary = argument["boundary"]
    output = paths.workspace / argument["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if argv[0] == str(paths.te_runner):
        roles = OUTPUT_ROLES[boundary]
        case = Path(argument["case-directory"])
        output.write_text(
            json.dumps(
                {
                    "schema": "shuttle_abi22_te_rmsnorm_raw_cell_v2",
                    "shape": list(shape),
                    "boundary": boundary,
                    "epsilon": 1e-6,
                    "warmup_invocations": 10,
                    "raw_host_blocking_ns": [100_000 + index * 1_000 for index in range(20)],
                    "raw_cuda_event_ns": [50_000 + index * 100 for index in range(20)],
                    "output_files": {role: str(case / f"reference_{role}.bf16") for role in roles},
                    "workspace_bytes": {"forward": 0, "backward": 0},
                },
                sort_keys=True,
            )
        )
    else:
        output.write_text(json.dumps(_valid_cell(shape, boundary, te=False), sort_keys=True))
    return 0


def test_execute_oracle_runs_exact_matrix_and_builds_matched_acceptance(tmp_path: Path) -> None:
    paths = BatchPaths.under(tmp_path / "workspace")
    paths.workspace.mkdir()
    paths.reports.mkdir()
    paths.raw.mkdir()
    paths.te_runner.write_bytes(b"runner-binary")
    paths.te_library.parent.mkdir(parents=True)
    paths.te_library.write_bytes(b"resolved-te-library")
    calls: list[list[str]] = []
    device = {"model": "NVIDIA H100 80GB HBM3", "uuid": "GPU-test", "compute_capability": "9.0"}

    result = execute_oracle(
        paths=paths,
        device=device,
        invoke=lambda argv: _fake_oracle_invoke(paths, calls, argv),
    )

    assert len(calls) == 12
    assert [
        (call[call.index("--rows") + 1], call[call.index("--features") + 1], call[call.index("--boundary") + 1])
        for call in calls[::2]
    ] == [(str(shape[0]), str(shape[1]), boundary) for shape in SHAPES for boundary in BOUNDARIES]
    assert all(call[call.index("--epsilon") + 1] == "1e-06" for call in calls)
    assert result["acceptance"]["status"] == "representative_numerics_accepted"
    assert result["acceptance"]["accepted_shape"] == [2048, 4096]
    assert result["acceptance"]["performance"]["median_ratio_te_over_xla"] == {boundary: 0.5 for boundary in BOUNDARIES}
    assert result["observed_xla_runtime"] == XLA_OBSERVED_RUNTIME
    assert result["shuttle"] == {"status": "pending", "blocker": SHUTTLE_BLOCKER}
    assert set(result["reports"]) == {"transformer_engine_2_17", "ordinary_jax_xla"}
    assert all(path.is_file() for path in (paths.te_report, paths.xla_report, paths.acceptance))


def test_execute_oracle_stops_on_first_failed_producer(tmp_path: Path) -> None:
    paths = BatchPaths.under(tmp_path / "workspace")
    paths.workspace.mkdir()
    paths.reports.mkdir()
    paths.raw.mkdir()
    paths.te_runner.write_bytes(b"runner-binary")
    paths.te_library.parent.mkdir(parents=True)
    paths.te_library.write_bytes(b"resolved-te-library")
    calls: list[list[str]] = []

    def fail_sixth(argv: list[str]) -> int:
        if len(calls) == 5:
            calls.append(argv)
            return 7
        return _fake_oracle_invoke(paths, calls, argv)

    with pytest.raises(RuntimeError, match="exited with status 7"):
        execute_oracle(
            paths=paths,
            device={"model": "H100", "uuid": "GPU-test", "compute_capability": "9.0"},
            invoke=fail_sixth,
        )
    assert len(calls) == 6


def test_execute_batch_drives_validated_oracle_and_writes_closed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest = _manifest(repository, "lib/shuttle/mlir/probe.txt", b"reviewed bytes")
    workspace = tmp_path / "workspace"
    calls: list[list[str]] = []

    def fake_hydration(paths: BatchPaths, plan: dict) -> dict:
        assert all("shuttle" not in phase["name"] for phase in plan["phases"])
        paths.te_runner.write_bytes(b"runner-binary")
        paths.te_library.parent.mkdir(parents=True)
        paths.te_library.write_bytes(b"resolved-te-library")
        return {
            "device": {"model": "NVIDIA H100 80GB HBM3", "uuid": "GPU-test", "compute_capability": "9.0"},
            "driver": "580.00",
            "transformer_engine": {
                "source_commit": TE_SOURCE_COMMIT,
                "source_tree": TE_SOURCE_TREE,
                "version": "2.17.0",
                "wheel_sha256": TE_WHEEL_SHA256,
                "library_path": str(paths.te_library),
                "library_sha256": hashlib.sha256(b"resolved-te-library").hexdigest(),
            },
            "runtime_libraries": RUNTIME_LIBRARY_IDENTITIES,
            "cuda_runtime_headers": CUDA_HEADER_IDENTITY,
        }

    monkeypatch.setattr("track2_te_h100_batch.platform.system", lambda: "Linux")
    monkeypatch.setattr("track2_te_h100_batch._execute_hydration", fake_hydration)
    monkeypatch.setattr(
        "track2_te_h100_batch._invoke_oracle",
        lambda paths, argv: _fake_oracle_invoke(paths, calls, argv),
    )

    result = execute_batch(
        workspace=workspace,
        runner_repository=repository,
        manifest_path=manifest,
        observed_base_image=BASE_IMAGE,
    )

    assert result["schema"] == "shuttle_track2_te_h100_batch_result_v1"
    assert result["source"]["commit"] == "1" * 40
    assert result["tools"]["transformer_engine"] == {
        "source_commit": TE_SOURCE_COMMIT,
        "source_tree": TE_SOURCE_TREE,
        "version": "2.17.0",
        "wheel_sha256": TE_WHEEL_SHA256,
        "library_path": str(BatchPaths.under(workspace).te_library),
        "library_sha256": hashlib.sha256(b"resolved-te-library").hexdigest(),
    }
    assert result["tools"]["runtime_libraries"] == RUNTIME_LIBRARY_IDENTITIES
    assert result["tools"]["cuda_runtime_headers"] == CUDA_HEADER_IDENTITY
    assert result["evidence"]["acceptance"]["accepted_shape"] == [2048, 4096]
    assert result["evidence"]["shuttle"] == {"status": "pending", "blocker": SHUTTLE_BLOCKER}
    assert len(calls) == 12
    assert json.loads(BatchPaths.under(workspace).result.read_bytes()) == result
    assert not any("shuttle" in path.name for path in BatchPaths.under(workspace).reports.iterdir())


def test_oracle_subprocess_uses_resolved_te_library_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = BatchPaths.under(tmp_path / "workspace")
    paths.workspace.mkdir()
    paths.te_library.parent.mkdir(parents=True)
    paths.te_library.write_bytes(b"resolved-te-library")
    observed: dict[str, object] = {}

    class Result:
        returncode = 0

    def fake_run(argv: list[str], **kwargs: object) -> Result:
        observed["argv"] = argv
        observed.update(kwargs)
        return Result()

    monkeypatch.setattr("track2_te_h100_batch.subprocess.run", fake_run)

    assert _invoke_oracle(paths, [str(paths.te_runner), "--probe", "1"]) == 0
    assert observed["argv"] == [str(paths.te_runner), "--probe", "1"]
    assert observed["cwd"] == paths.workspace
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert environment["LD_LIBRARY_PATH"].split(":")[0] == str(paths.te_library.parent)


def test_execute_batch_rejects_zero_exit_with_missing_oracle_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest = _manifest(repository, "lib/shuttle/mlir/probe.txt", b"reviewed bytes")
    workspace = tmp_path / "workspace"

    def fake_hydration(paths: BatchPaths, plan: dict) -> dict:
        paths.te_runner.write_bytes(b"runner-binary")
        paths.te_library.parent.mkdir(parents=True)
        paths.te_library.write_bytes(b"resolved-te-library")
        return {
            "device": {"model": "H100", "uuid": "GPU-test", "compute_capability": "9.0"},
            "driver": "580.00",
            "transformer_engine": {},
        }

    monkeypatch.setattr("track2_te_h100_batch.platform.system", lambda: "Linux")
    monkeypatch.setattr("track2_te_h100_batch._execute_hydration", fake_hydration)
    monkeypatch.setattr("track2_te_h100_batch._invoke_oracle", lambda paths, argv: 0)
    with pytest.raises((FileNotFoundError, ValueError), match=r"raw|No such file"):
        execute_batch(
            workspace=workspace,
            runner_repository=repository,
            manifest_path=manifest,
            observed_base_image=BASE_IMAGE,
        )


def test_source_manifest_binds_every_declared_byte(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, "lib/shuttle/mlir/probe.txt", b"reviewed bytes")

    identity = validate_source_manifest(tmp_path, manifest)

    assert identity == {
        "commit": "1" * 40,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "tree": "2" * 40,
    }
    (tmp_path / "lib/shuttle/mlir/probe.txt").write_bytes(b"drifted bytes")
    with pytest.raises(RuntimeError, match="differs from its manifest"):
        validate_source_manifest(tmp_path, manifest)


def test_batch_paths_keep_raw_reports_and_runtime_identity_separate(tmp_path: Path) -> None:
    paths = BatchPaths.under(tmp_path)

    assert (
        len(
            {
                paths.te_report,
                paths.xla_report,
                paths.acceptance,
                paths.runtime_manifest,
                paths.result,
            }
        )
        == 5
    )
    assert paths.te_library.name == "libtransformer_engine.so"
    assert paths.raw.name == "raw"
