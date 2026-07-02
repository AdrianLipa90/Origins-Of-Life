"""Repo-native repository semantics facade over internal orbital assignment."""
from ..orbital.repository_assignment import assign_orbital_state_to_entity as assign_emergence_state_to_entity
from ..orbital.repository_assignment import build_repository_system_state as build_origin_repository_state

__all__ = ["assign_emergence_state_to_entity", "build_origin_repository_state"]
