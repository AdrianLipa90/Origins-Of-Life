"""Canonical abiogenesis repository-semantics aliases over the orbital substrate."""

from ..orbital.repository_assignment import (
    assign_orbital_state_to_entity,
    build_repository_system_state,
)

assign_emergence_state_to_entity = assign_orbital_state_to_entity
build_origin_repository_state = build_repository_system_state

__all__ = ["assign_emergence_state_to_entity", "build_origin_repository_state"]
