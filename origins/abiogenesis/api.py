"""Canonical high-level API for repo-facing use.

Prefer this module over direct public use of `origins.orbital` names.
"""

from ..analysis.sweep_orbital import (
    export_orbital_analysis_snapshot,
    run_all_scenarios_orbital,
    run_sweep_v3_orbital,
    run_topo_sweep_orbital,
)
from ..simulator.factory import available_simulator_modes, create_simulator
from ..simulator.universal_orbital import OrbitalUniversalOriginSimulator
from . import AbiogenesisRuntimeAdapter, AbiogenesisRunBundle


def create_abiogenesis_runtime(config, mode: str = "orbital", **kwargs):
    return create_simulator(config=config, mode=mode, **kwargs)


def run_habitat_scan(**kwargs):
    return run_sweep_v3_orbital(**kwargs)


def run_feasibility_scan(**kwargs):
    return run_topo_sweep_orbital(**kwargs)


def run_origin_comparison(**kwargs):
    return run_all_scenarios_orbital(**kwargs)


def export_historical_state(**kwargs):
    return export_orbital_analysis_snapshot(**kwargs)


__all__ = [
    "AbiogenesisRunBundle",
    "AbiogenesisRuntimeAdapter",
    "OrbitalUniversalOriginSimulator",
    "available_simulator_modes",
    "create_abiogenesis_runtime",
    "run_habitat_scan",
    "run_feasibility_scan",
    "run_origin_comparison",
    "export_historical_state",
]
