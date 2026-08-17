# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed entry point for the unsealed ABI21 RMSNorm evidence plan."""

import argparse

from abi21_rmsnorm_prehardware_contract import load_contract


def require_launch_ready() -> None:
    """Reject external execution until every predeclared blocker is sealed."""
    contract = load_contract()
    blockers = ", ".join(contract.readiness_blockers)
    raise RuntimeError(f"ABI21 RMSNorm evidence is explicitly unsealed: {blockers}")


def main(argv: list[str] | None = None) -> int:
    """Validate the local contract and reject the only external action."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-launch-ready", action="store_true", required=True)
    parser.parse_args(argv)
    require_launch_ready()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
