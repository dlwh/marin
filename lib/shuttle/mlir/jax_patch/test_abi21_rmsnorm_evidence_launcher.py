# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from abi21_rmsnorm_evidence_launcher import main, require_launch_ready
from abi21_rmsnorm_prehardware_contract import readiness_blockers


def test_launcher_rejects_before_external_or_hardware_action() -> None:
    blockers = readiness_blockers()

    assert blockers == (
        "representative_numerical_evidence_missing",
        "sealed_build_identity_missing",
        "sealed_image_identity_missing",
        "controller_authority_missing",
        "sm100_codegen_missing",
        "h100_execution_missing",
        "b200_execution_missing",
        "performance_contract_pending",
    )
    with pytest.raises(RuntimeError, match="explicitly unsealed"):
        require_launch_ready()
    with pytest.raises(RuntimeError, match="explicitly unsealed"):
        main(["--require-launch-ready"])
