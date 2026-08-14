# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only ABI15 task launcher that stops before importing JAX."""

import hashlib
import json
from pathlib import Path

CONTRACT_SHA256 = "7b681b7a74eef86a3ce257f396dab442150a5cca41580f953f745add741849ba"
UNSEALED_AUTHORITY = {
    "image_digest": None,
    "launch_authorization_sha256": None,
    "source_capsule_sha256": None,
    "status": "not_authorized",
}


def require_launch_authority_before_import() -> None:
    """Fail before imports, writes, subprocesses, or backend discovery."""
    path = (
        Path(__file__).parents[4] / ".agents/projects/tile_lifetime_compiler/shuttle_abi15_h100_composed_prerun_v1.json"
    )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONTRACT_SHA256:
        raise RuntimeError("ABI15 launcher contract bytes drifted")
    document = json.loads(raw)
    if document.get("hardware_authority") != UNSEALED_AUTHORITY:
        raise RuntimeError("ABI15 launcher has no reviewed external authority loader")
    raise RuntimeError("ABI15 H100 launch authority is explicitly unsealed")


def main() -> int:
    require_launch_authority_before_import()
    raise AssertionError("unreachable until a separately reviewed authority and ACK trust root exist")


if __name__ == "__main__":
    raise SystemExit(main())
