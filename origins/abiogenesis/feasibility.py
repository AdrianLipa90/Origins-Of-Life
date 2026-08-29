"""Canonical abiogenesis feasibility aliases over the orbital substrate."""

from ..orbital.potentials import PotentialTerms, compute_potential_terms

FeasibilityTerms = PotentialTerms
compute_feasibility_terms = compute_potential_terms

__all__ = ["FeasibilityTerms", "compute_feasibility_terms"]
