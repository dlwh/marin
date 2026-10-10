# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the pinned standalone CUDA runner on a CPU-only Linux task."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from track1_map_h100_build import XLA_COMMIT

TARGET = "@shuttle_mlir//:gpu_fatbin_h100_test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bazel-url", required=True)
    parser.add_argument("--bazel-sha256", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--ram-mb", type=int, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[4]
    args.workspace.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=True)
    xla = args.workspace / "xla"
    xla.mkdir()
    bazel = args.workspace / "bazel"
    report = {"xla_commit": XLA_COMMIT, "target": TARGET, "cuda_version": args.cuda_version, "commands": []}

    def run(command, *, cwd=xla, name):
        report["commands"].append({"name": name, "argv": [str(value) for value in command]})
        (args.output / "build-plan.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"native build phase: {name}", flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            print((args.output / f"{name}.log").read_text()[-12000:], flush=True)
            completed.check_returncode()

    run(["git", "init", "--quiet"], name="xla-init")
    run(["git", "remote", "add", "origin", "https://github.com/openxla/xla.git"], name="xla-remote")
    run(["git", "fetch", "--depth=1", "origin", XLA_COMMIT], name="xla-fetch")
    run(["git", "checkout", "--detach", "FETCH_HEAD"], name="xla-checkout")
    patch = repository / "lib/shuttle/mlir/jax_patch/track2_xla_workspace.patch"
    run(["git", "apply", "--check", patch], name="workspace-check")
    run(["git", "apply", patch], name="workspace-apply")
    run(
        [
            "curl",
            "--fail",
            "--location",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--output",
            bazel,
            args.bazel_url,
        ],
        name="bazel-fetch",
    )
    digest = hashlib.sha256(bazel.read_bytes()).hexdigest()
    if digest != args.bazel_sha256:
        raise ValueError("Bazel digest does not match the requested build tool")
    bazel.chmod(0o755)
    common = [str(bazel), f"--output_user_root={args.workspace / 'bazel-output'}"]
    options = [
        f"--override_repository=shuttle_mlir={repository / 'lib/shuttle/mlir'}",
        "--config=cuda",
        f"--repo_env=HERMETIC_CUDA_VERSION={args.cuda_version}",
        "--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=9.0",
        "--repo_env=HERMETIC_PYTHON_VERSION=3.12",
        f"--jobs={args.jobs}",
        f"--local_ram_resources={args.ram_mb}",
    ]
    run([*common, "build", *options, "--nobuild", TARGET], name="toolchain-analysis")
    fixture_source = repository / "lib/shuttle/mlir/test/gpu-fatbin-duplicate-symbol.cu"
    shutil.copy2(fixture_source, args.output)
    nvcc_paths = tuple((args.workspace / "bazel-output").glob("*/external/cuda_nvcc/bin/nvcc"))
    if len(nvcc_paths) != 1:
        raise ValueError("Expected one hermetic CUDA compiler from the pinned build")
    nvcc = nvcc_paths[0]
    external = nvcc.parents[2]
    sysroot_alias = external / "sysroot_linux_x86_64/BUILD"
    sysroot_repository = re.search(r'actual = "@@([^/]+)//:sysroot"', sysroot_alias.read_text())
    if sysroot_repository is None:
        raise ValueError("Pinned XLA sysroot alias does not identify its header repository")
    sysroot = external / sysroot_repository.group(1)
    if not (sysroot / "usr/include/features.h").is_file():
        raise ValueError("Pinned XLA sysroot is missing C library headers")
    report["fixture_host_sysroot"] = str(sysroot)
    report["fixture_host_sysroot_alias_sha256"] = hashlib.sha256(sysroot_alias.read_bytes()).hexdigest()
    for image, factor, bias in ((0, 1, 1), (1, 2, 0)):
        run(
            [
                nvcc,
                "--fatbin",
                "--gpu-architecture=sm_90",
                f"--compiler-options=--sysroot={sysroot}",
                f"--include-path={external / 'cuda_cudart/include'}",
                f"-DSHUTTLE_IMAGE_FACTOR={factor}",
                f"-DSHUTTLE_IMAGE_BIAS={bias}",
                fixture_source,
                "--output-file",
                args.output / f"image{image}.fatbin",
            ],
            name=f"fixture-image{image}",
        )
    report["nvcc_sha256"] = hashlib.sha256(nvcc.read_bytes()).hexdigest()
    report["fixture_images"] = {
        f"image{image}.fatbin": hashlib.sha256((args.output / f"image{image}.fatbin").read_bytes()).hexdigest()
        for image in (0, 1)
    }
    run([*common, "build", *options, TARGET], name="runner-build")
    binary = xla / "bazel-bin/external/shuttle_mlir/gpu_fatbin_h100_test"
    if not binary.is_file():
        raise ValueError("Bazel did not produce the standalone native runner")
    shutil.copy2(binary, args.output / "gpu_fatbin_h100_test")
    dependencies = subprocess.run(["ldd", binary], check=True, capture_output=True, text=True).stdout
    (args.output / "ldd.txt").write_text(dependencies)
    libraries = args.output / "lib"
    libraries.mkdir()
    for line in dependencies.splitlines():
        if line.strip() == "libcuda.so.1 => not found":
            report["gpu_driver_dependency"] = "libcuda.so.1 (provided by GPU task)"
            continue
        if "=> not found" in line:
            raise ValueError(f"Native runner has an unresolved runtime dependency: {line}")
        fields = line.split()
        if len(fields) >= 3 and fields[1] == "=>" and fields[2].startswith(str(args.workspace)):
            library = Path(fields[2])
            shutil.copy2(library, libraries / library.name)
    report["source_digests"] = {
        str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((repository / "lib/shuttle/mlir").rglob("*"))
        if path.is_file()
    }
    report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    report["bazel_sha256"] = digest
    report["status"] = "built"
    (args.output / "build-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"status": "built", "binary_bytes": binary.stat().st_size, "output": os.fspath(args.output)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
