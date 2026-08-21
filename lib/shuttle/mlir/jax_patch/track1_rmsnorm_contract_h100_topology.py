# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reviewed bundle topology for the ordinary-JAX RMSNorm-to-Contract canary."""

PRIMAL_BUNDLE_SHAPES = {
    (5, 19, 7): {"launches": 1, "temporary_bytes": 0, "temporaries": 0},
    (5, 19, 257): {"launches": 2, "temporary_bytes": 20, "temporaries": 1},
}

COMPOSED_BUNDLE_SHAPES = {
    (5, 19, 7): {"launches": 39, "temporary_bytes": 6996, "temporaries": 36},
    (5, 19, 257): {"launches": 40, "temporary_bytes": 79016, "temporaries": 37},
}

PRIMAL_TOPOLOGY_STATUS = "normative_primal"
COMPOSED_TOPOLOGY_STATUS = "reviewed_hybrid_exact"


def authorization_bundle_shapes(shapes: dict[tuple[int, ...], dict[str, int]]) -> dict[str, dict[str, int]]:
    """Return shape-keyed topology for the canonical authorization document."""
    return {"x".join(map(str, shape)): dict(resources) for shape, resources in shapes.items()}
