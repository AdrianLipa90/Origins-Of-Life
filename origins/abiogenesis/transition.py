"""Canonical abiogenesis transition aliases over the orbital OORP substrate."""

from ..orbital.oorp import OORPTrace, run_oorp_pipeline

EmergenceTrace = OORPTrace
run_emergence_transition = run_oorp_pipeline

__all__ = ["EmergenceTrace", "run_emergence_transition"]
