from origins.abiogenesis import (
    AbiogenesisRuntimeAdapter,
    EmergenceCoordinate,
    FeasibilityTerms,
    HistoricalMemory,
    OriginHabitatShell,
)
from origins.abiogenesis.api import (
    create_abiogenesis_runtime,
    run_feasibility_scan,
    run_habitat_scan,
    run_origin_comparison,
)
from origins.orbital.runtime_bridge import OrbitalRuntimeBridge
from origins.orbital.state import OrbitalCoordinate
from origins.orbital.potentials import PotentialTerms
from origins.orbital.memory import MemoryState
from origins.orbital.sphere import OrbitalSphere


def test_canonical_semantic_surface_points_to_internal_substrate():
    assert AbiogenesisRuntimeAdapter is OrbitalRuntimeBridge
    assert EmergenceCoordinate is OrbitalCoordinate
    assert FeasibilityTerms is PotentialTerms
    assert HistoricalMemory is MemoryState
    assert OriginHabitatShell is OrbitalSphere


def test_canonical_api_exports_expected_callables():
    assert callable(create_abiogenesis_runtime)
    assert callable(run_habitat_scan)
    assert callable(run_feasibility_scan)
    assert callable(run_origin_comparison)
