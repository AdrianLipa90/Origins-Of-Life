"""Repo-native emergence transition facade over internal OORP pipeline."""
from ..orbital.oorp import OORPTrace as EmergenceTrace
from ..orbital.oorp import run_oorp_pipeline as run_emergence_transition

__all__ = ["EmergenceTrace", "run_emergence_transition"]
