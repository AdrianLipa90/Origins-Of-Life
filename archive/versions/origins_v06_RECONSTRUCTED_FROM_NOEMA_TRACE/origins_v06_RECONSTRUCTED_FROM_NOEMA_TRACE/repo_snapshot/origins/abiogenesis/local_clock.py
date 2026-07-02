"""Repo-native emergence clock facade over internal orbital subjective time."""
from ..orbital.subjective_time import compute_local_subjective_time as compute_emergence_clock

__all__ = ["compute_emergence_clock"]
