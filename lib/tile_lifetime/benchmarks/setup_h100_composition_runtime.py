# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the pinned Linux runtime used by the H100 composition experiments.

Run with Python 3.12 on Linux, passing a new --directory and --manifest path.
The external lock is generated from h100_composition_runtime.in. QuACK uses an
immutable source revision with the repository's explicit FP32 row-scale patch.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

QUACK_REVISION = "84ef91df9bec87c7e4938517234fafb07ef844dd"
QUACK_URL = f"https://codeload.github.com/Dao-AILab/quack/tar.gz/{QUACK_REVISION}"
QUACK_SHA256 = "a1936556314bd7998088f7c9083389b62615c302214c7c67dad40542e1cca72a"
PATCH_SHA256 = "8813b2fb67690545e554119dc69d53a39ceb5239c41d6f7375435c2d032bd954"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("This runtime is pinned to Linux x86_64 CPython 3.12")
    root = Path(__file__).resolve().parents[3]
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    archive = directory / "quack.tar.gz"
    urllib.request.urlretrieve(QUACK_URL, archive)
    if digest(archive) != QUACK_SHA256:
        raise ValueError("QuACK archive digest mismatch")
    with tarfile.open(archive) as source:
        source.extractall(directory, filter="data")
    quack = directory / f"quack-{QUACK_REVISION}"
    patch = root / "lib/tile_lifetime/backends/h100/quack_fp32_row_scale.patch"
    if digest(patch) != PATCH_SHA256:
        raise ValueError("QuACK row-scale patch digest mismatch")
    run("git", "-C", str(quack), "apply", "--check", str(patch))
    run("git", "-C", str(quack), "apply", str(patch))
    lock = Path(__file__).with_name("h100_composition_runtime.txt")
    environment = directory / "venv"
    python = environment / "bin/python"
    run("uv", "--no-config", "venv", "--python", sys.executable, str(environment))
    run("uv", "--no-config", "pip", "sync", "--python", str(python), "--require-hashes", str(lock))
    # The lock pins build backends too; local packages retain dependency checking.
    run(
        "uv",
        "--no-config",
        "pip",
        "install",
        "--python",
        str(python),
        "--no-build-isolation",
        "--constraint",
        str(lock),
        str(quack),
        str(root / "lib/shuttle"),
        str(root / "lib/tile_lifetime"),
    )
    run("uv", "--no-config", "pip", "check", "--python", str(python))
    preflight = subprocess.check_output(
        [
            str(python),
            "-c",
            """
import importlib.metadata, inspect, json
import torch
import jax
import quack.gemm_runtime.host
assert 'recompile_limit' in inspect.signature(torch.compile).parameters
assert torch.__version__ == '2.12.1+cu130'
assert jax.__version__ == '0.10.1'
assert importlib.metadata.version('nvidia-cutlass-dsl') == '4.6.1'
compiled = torch.compile(lambda x: x + 1, backend='eager', recompile_limit=64)
assert compiled(torch.tensor(1)).item() == 2
print(json.dumps({'torch_compile_signature': str(inspect.signature(torch.compile)),
                  'torch_cuda': torch.version.cuda,
                  'devices': [str(x) for x in jax.devices()],
                  'packages': {x.metadata['Name']: x.version for x in importlib.metadata.distributions()}}))
""",
        ],
        text=True,
    )
    result = json.loads(preflight)
    result.update(
        {
            "quack_revision": QUACK_REVISION,
            "quack_archive_url": QUACK_URL,
            "quack_archive_sha256": QUACK_SHA256,
            "quack_patch_sha256": PATCH_SHA256,
            "quack_patched_source_sha256": {
                str(path.relative_to(quack)): digest(path) for path in sorted((quack / "quack").rglob("*.py"))
            },
            "external_lock_sha256": digest(lock),
            "bootstrap_sha256": digest(Path(__file__)),
            "python": str(python),
            "python_version": sys.version,
        }
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Runtime ready: {python}", flush=True)


if __name__ == "__main__":
    main()
