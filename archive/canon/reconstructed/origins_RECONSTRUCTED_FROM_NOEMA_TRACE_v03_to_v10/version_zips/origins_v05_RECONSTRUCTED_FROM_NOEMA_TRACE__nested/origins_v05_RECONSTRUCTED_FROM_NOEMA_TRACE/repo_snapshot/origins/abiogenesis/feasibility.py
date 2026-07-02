"""Repo-native feasibility facade over internal orbital potentials.

This file intentionally does not duplicate orbital logic. It exposes the
abiogenesis-facing vocabulary expected by the public package surface.
"""
from ..orbital.potentials import PotentialTerms as FeasibilityTerms
from ..orbital.potentials import compute_potential_terms as compute_feasibility_terms

__all__ = ["FeasibilityTerms", "compute_feasibility_terms"]
