from typing import Dict

from .definitions import (
    EpistemicStatus,
    ObjectType,
    ScenarioDefinition,
    Sector,
    SemanticLayer,
)
from .scenarios import ALL_SCENARIOS


# Conservative catalog: phase/orbit values remain minimal until a dedicated
# orbital classification pass assigns nontrivial geometry from measured data.
CANONICAL_SCENARIO_DEFINITIONS: Dict[str, ScenarioDefinition] = {
    cfg.code: ScenarioDefinition(
        canonical_id=f"OOL-SCENARIO-{cfg.code}",
        name=cfg.name,
        object_type=ObjectType.SCENARIO,
        sector=Sector.CHEMISTRY,
        semantic_layer=SemanticLayer.PROCESS,
        epistemic_status=EpistemicStatus.WORKING,
        orbit_index=0,
        phase=0.0,
        winding_number=0,
        relation_depth=1,
        semantic_mass=1.0,
        subjective_time_scale=1.0,
        provenance_links=["src/origin_of_life/scenarios.py"],
        dependency_links=[
            "src/origin_of_life/simulator.py",
            "src/origin_of_life/runners.py",
            "src/origin_of_life/sweeps.py",
        ],
    )
    for cfg in ALL_SCENARIOS
}


CANONICAL_RUNTIME_OBJECTS = {
    "simulator": {
        "canonical_id": "OOL-RUNTIME-SIMULATOR",
        "object_type": ObjectType.SIMULATOR,
        "sector": Sector.CORE,
        "semantic_layer": SemanticLayer.PROCESS,
        "epistemic_status": EpistemicStatus.CANONICAL,
        "source_path": "src/origin_of_life/simulator.py",
    },
    "sweep_v3": {
        "canonical_id": "OOL-RUNTIME-SWEEP-V3",
        "object_type": ObjectType.SWEEP,
        "sector": Sector.CHEMISTRY,
        "semantic_layer": SemanticLayer.PROCESS,
        "epistemic_status": EpistemicStatus.LEGACY,
        "source_path": "src/origin_of_life/sweep_v3.py",
    },
    "registry": {
        "canonical_id": "OOL-RUNTIME-REGISTRY",
        "object_type": ObjectType.REGISTRY,
        "sector": Sector.CORE,
        "semantic_layer": SemanticLayer.IDENTITY,
        "epistemic_status": EpistemicStatus.CANONICAL,
        "source_path": "src/origin_of_life/registry.py",
    },
    "definitions": {
        "canonical_id": "OOL-RUNTIME-DEFINITIONS",
        "object_type": ObjectType.DOCUMENT,
        "sector": Sector.DOCS,
        "semantic_layer": SemanticLayer.RELATION,
        "epistemic_status": EpistemicStatus.CANONICAL,
        "source_path": "src/origin_of_life/definitions.py",
    },
}
