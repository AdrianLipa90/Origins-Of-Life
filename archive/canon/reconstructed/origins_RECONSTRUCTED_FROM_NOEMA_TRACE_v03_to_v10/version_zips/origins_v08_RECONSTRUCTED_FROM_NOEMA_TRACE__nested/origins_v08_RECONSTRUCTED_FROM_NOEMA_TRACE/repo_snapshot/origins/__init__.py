"""
Origins-Of-Life Holonomic — Python package.

Sub-packages
------------
constants   – centralised physical constants
scenarios   – 5 scenario configurations (A–E)
chemistry   – field utilities and clay catalysis
biology     – RNA molecules and protocell detection
topology    – Kähler-Berry-Euler fields and Zeta-Riemann constraints
simulator   – UniversalOriginSimulator (main simulation engine)
analysis    – parameter sweeps and multi-scenario comparison
cosmology   – CIEL/0 operators and CMB / PBH analysis
"""

__version__ = "1.0.0"
__author__  = "Adrian Lipa, CIEL-Omega Research"

from .scenarios import (
    ScenarioConfig,
    ALL_SCENARIOS,
    SCENARIOS_BY_CODE,
    SCENARIO_A,
    SCENARIO_B,
    SCENARIO_C,
    SCENARIO_D,
    SCENARIO_E,
)
from .simulator import UniversalOriginSimulator
