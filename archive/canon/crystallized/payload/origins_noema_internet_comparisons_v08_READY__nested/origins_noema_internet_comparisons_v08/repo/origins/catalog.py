from __future__ import annotations

import math
from typing import Dict

from .definitions import (
    EpistemicStatus,
    ObjectType,
    ScenarioDefinition,
    Sector,
    SemanticLayer,
)
from .scenarios import ALL_SCENARIOS


def _scenario_phase(cfg) -> float:
    """Derive orbital phase from physical parameters of the scenario.

    Phase encodes how far from ideal prebiotic conditions the scenario sits.
    Uses Euler phase coherence (config param) and thermal decoherence:
      phi = (1 - euler_phase_coherence) * pi + thermal_decoherence
    where thermal_decoherence = arctan(|temp_C - 65| / 200) * topo_strength
    Result: earth-like shallow ocean (A) ≈ 0.1 rad, exotic (D, Titan) ≈ 1.4 rad.
    """
    coherence_term = (1.0 - cfg.euler_phase_coherence) * math.pi
    temp_deviation = abs(cfg.temp_C - 65.0)  # 65°C = Hadean reference
    thermal_term = math.atan(temp_deviation / 200.0) * float(cfg.topo_strength)
    return coherence_term + thermal_term


CANONICAL_SCENARIO_DEFINITIONS: Dict[str, ScenarioDefinition] = {
    cfg.code: ScenarioDefinition(
        canonical_id=f"OOL-SCENARIO-{cfg.code}",
        name=cfg.name,
        code=cfg.code,
        object_type=ObjectType.SCENARIO,
        sector=Sector.CHEMISTRY,
        semantic_layer=SemanticLayer.PROCESS,
        epistemic_status=EpistemicStatus.WORKING,
        orbit_index=0,
        phase=_scenario_phase(cfg),
        winding_number=0,
        relation_depth=1,
        semantic_mass=max(1.0, float(cfg.expected_protocells) / 100.0),
        subjective_time_scale=1.0,
        provenance_links=["origins/scenarios.py"],
        dependency_links=[
            "origins/simulator/universal.py",
            "origins/analysis/sweep.py",
            "scripts/run_simulation.py",
            "scripts/run_sweep.py",
        ],
    )
    for cfg in ALL_SCENARIOS
}


CANONICAL_RUNTIME_OBJECTS = {
    "package": {
        "canonical_id": "OOL-PACKAGE-ORIGINS",
        "object_type": ObjectType.PACKAGE,
        "sector": Sector.CORE,
        "semantic_layer": SemanticLayer.RELATION,
        "epistemic_status": EpistemicStatus.CANONICAL,
        "source_path": "origins/__init__.py",
    },
    "scenarios": {
        "canonical_id": "OOL-MODULE-SCENARIOS",
        "object_type": ObjectType.MODULE,
        "sector": Sector.CHEMISTRY,
        "semantic_layer": SemanticLayer.IDENTITY,
        "epistemic_status": EpistemicStatus.CANONICAL,
        "source_path": "origins/scenarios.py",
    },
    "simulator": {
        "canonical_id": "OOL-MODULE-SIMULATOR",
        "object_type": ObjectType.SIMULATOR,
        "sector": Sector.CORE,
        "semantic_layer": SemanticLayer.PROCESS,
        "epistemic_status": EpistemicStatus.CANONICAL,
        "source_path": "origins/simulator/universal.py",
    },
    "analysis": {
        "canonical_id": "OOL-MODULE-ANALYSIS",
        "object_type": ObjectType.ANALYSIS,
        "sector": Sector.ANALYSIS,
        "semantic_layer": SemanticLayer.PROCESS,
        "epistemic_status": EpistemicStatus.CANONICAL,
        "source_path": "origins/analysis/sweep.py",
    },
    "registry": {
        "canonical_id": "OOL-MODULE-REGISTRY",
        "object_type": ObjectType.REGISTRY,
        "sector": Sector.CORE,
        "semantic_layer": SemanticLayer.IDENTITY,
        "epistemic_status": EpistemicStatus.CANONICAL,
        "source_path": "origins/registry.py",
    },
    "definitions": {
        "canonical_id": "OOL-MODULE-DEFINITIONS",
        "object_type": ObjectType.DOCUMENT,
        "sector": Sector.DOCS,
        "semantic_layer": SemanticLayer.RELATION,
        "epistemic_status": EpistemicStatus.CANONICAL,
        "source_path": "origins/definitions.py",
    },
}
