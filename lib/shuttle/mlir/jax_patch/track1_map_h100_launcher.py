# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed launcher for the unsealed Track 1 Map H100 subject."""

import argparse


def require_launch_ready() -> None:
    """Stop before importing JAX, starting a process, or discovering a device."""
    raise RuntimeError("Track 1 Map H100 launch authority is explicitly unsealed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-launch-ready", action="store_true", required=True)
    parser.parse_args(argv)
    require_launch_ready()
    raise AssertionError("unreachable until a separately reviewed launch authority is supplied")


if __name__ == "__main__":
    raise SystemExit(main())
