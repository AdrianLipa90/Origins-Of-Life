"""Canonical public semantic surface for Origins-Of-Life.

This package is the repository-facing semantic layer.
The lower-level `origins.orbital` package remains the implementation substrate.
"""

from ..orbital.sphere import OrbitalSphere as OriginHabitatShell
from ..orbital.sphere import SphereEmbedding as HabitatEmbedding
from ..orbital.state import OrbitalCoordinate as EmergenceCoordinate
from ..orbital.state import OrbitalSystemState as EmergenceSystemState
from ..orbital.potentials import PotentialTerms as FeasibilityTerms
from ..orbital.potentials import compute_potential_terms as compute_feasibility_terms
from ..orbital.subjective_time import compute_local_subjective_time as compute_emergence_clock
from ..orbital.winding import WindingComponents as RecurrenceComponents
from ..orbital.winding import compute_winding_components as compute_recurrence_components
from ..orbital.memory import ReductionResidue as HistoricalResidue
from ..orbital.memory import MemoryState as HistoricalMemory
from ..orbital.memory import apply_memory_update as apply_residue_update
from ..orbital.oorp import OORPTrace as EmergenceTrace
from ..orbital.oorp import run_oorp_pipeline as run_emergence_transition
from ..orbital.repository_assignment import assign_orbital_state_to_entity as assign_emergence_state_to_entity
from ..orbital.repository_assignment import build_repository_system_state as build_origin_repository_state
from ..orbital.runtime_bridge import OrbitalRunBundle as AbiogenesisRunBundle
from ..orbital.runtime_bridge import OrbitalRuntimeBridge as AbiogenesisRuntimeAdapter

__all__ = [
    "OriginHabitatShell",
    "HabitatEmbedding",
    "EmergenceCoordinate",
    "EmergenceSystemState",
    "FeasibilityTerms",
    "compute_feasibility_terms",
    "compute_emergence_clock",
    "RecurrenceComponents",
    "compute_recurrence_components",
    "HistoricalResidue",
    "HistoricalMemory",
    "apply_residue_update",
    "EmergenceTrace",
    "run_emergence_transition",
    "assign_emergence_state_to_entity",
    "build_origin_repository_state",
    "AbiogenesisRunBundle",
    "AbiogenesisRuntimeAdapter",
]
