"""Canonical abiogenesis local-clock alias over the orbital substrate."""

from ..orbital.subjective_time import compute_local_subjective_time

compute_emergence_clock = compute_local_subjective_time

__all__ = ["compute_emergence_clock"]
