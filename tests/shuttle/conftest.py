# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

JAX_PATCH = Path(__file__).parents[2] / "lib/shuttle/mlir/jax_patch"
sys.path.insert(0, str(JAX_PATCH))
