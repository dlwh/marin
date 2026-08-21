# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed launcher for the authorization-bound Track 1 prototype."""

import argparse
from pathlib import Path

from track1_map_h100_prototype import canonical_json, execute_prototype, load_authorization, prototype_plan


def require_launch_ready() -> None:
    """Stop before importing JAX, starting a process, or discovering a device."""
    raise RuntimeError("Track 1 Map H100 launch authority is explicitly unsealed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-launch-ready", action="store_true")
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--runner-repository", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--base-image")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.require_launch_ready or arguments.authorization_file is None:
        require_launch_ready()
    required = {
        "authorization SHA-256": arguments.authorization_sha256,
        "workspace": arguments.workspace,
        "runner repository": arguments.runner_repository,
        "Python": arguments.python,
        "CUDA root": arguments.cuda_root,
        "base image": arguments.base_image,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing or not (arguments.dry_run or arguments.execute):
        parser.error(f"authorization-bound launch is incomplete: {', '.join(missing)}")

    authorization = load_authorization(arguments.authorization_file, arguments.authorization_sha256)
    keywords = {
        "workspace": arguments.workspace,
        "runner_repository": arguments.runner_repository,
        "python": arguments.python,
        "cuda_root": arguments.cuda_root,
        "observed_base_image": arguments.base_image,
    }
    if arguments.dry_run:
        result = prototype_plan(authorization, **keywords)
    else:
        result = execute_prototype(authorization, **keywords)
    print(canonical_json(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
