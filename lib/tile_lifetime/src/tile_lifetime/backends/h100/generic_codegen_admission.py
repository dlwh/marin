# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission for generic H100 code generation after physical provider selection."""

from collections.abc import Iterable

from tile_lifetime.tensor_program import ContractPrimitive, TensorPrimitive


class UnsupportedContractLoweringError(ValueError):
    """A Contract has no admitted physical lowering; scalar expansion is forbidden."""

    def __init__(
        self,
        contracts: tuple[tuple[int, ContractPrimitive], ...],
        *,
        component_id: int | None,
        phase_id: int | None,
    ):
        self.contracts = contracts
        self.component_id = component_id
        self.phase_id = phase_id
        location = "program" if component_id is None else f"component {component_id}"
        if phase_id is not None:
            location += f" phase {phase_id}"
        details = "; ".join(
            f"operation {ordinal} {operation.name!r}: "
            f"inputs={tuple(value.shape for value in operation.inputs)}, "
            f"output={operation.output.shape}, "
            f"reduction={tuple(axis.extent for axis in operation.reduction_axes)}"
            for ordinal, operation in contracts
        )
        super().__init__(
            f"No physical Contract lowering admitted for {location}: {details}. "
            "Scalar Contract code generation is disabled; provide a physical lowering "
            "that preserves the source precision and rounding."
        )


def require_generic_codegen_admission(
    operations: Iterable[tuple[int, TensorPrimitive]],
    *,
    component_id: int | None = None,
    phase_id: int | None = None,
) -> None:
    """Reject remaining Contracts before generic allocation or code generation."""
    contracts = tuple(
        (ordinal, operation) for ordinal, operation in operations if isinstance(operation, ContractPrimitive)
    )
    if contracts:
        raise UnsupportedContractLoweringError(contracts, component_id=component_id, phase_id=phase_id)
