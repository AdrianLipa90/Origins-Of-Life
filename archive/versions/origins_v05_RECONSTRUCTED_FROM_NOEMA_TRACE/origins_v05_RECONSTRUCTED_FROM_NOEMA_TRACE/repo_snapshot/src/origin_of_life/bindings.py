from typing import Dict, Iterable, Optional

from .catalog import CANONICAL_RUNTIME_OBJECTS, CANONICAL_SCENARIO_DEFINITIONS
from .definitions import ObjectType, ScenarioDefinition, Sector, EpistemicStatus
from .registry import EntityRecord, build_scenario_entity_record, default_attractor_weights


def scenario_config_to_definition(cfg) -> ScenarioDefinition:
    return CANONICAL_SCENARIO_DEFINITIONS.get(
        cfg.code,
        ScenarioDefinition(
            canonical_id=f"OOL-SCENARIO-{cfg.code}",
            name=cfg.name,
            provenance_links=["src/origin_of_life/scenarios.py"],
            dependency_links=["src/origin_of_life/simulator.py"],
        ),
    )


def scenario_config_to_entity_record(cfg, source_path: str = "src/origin_of_life/scenarios.py") -> EntityRecord:
    definition = scenario_config_to_definition(cfg)
    record = build_scenario_entity_record(
        canonical_id=definition.canonical_id,
        source_path=source_path,
        orbit_index=definition.orbit_index,
        phase=definition.phase,
    )
    record.provenance_links = list(definition.provenance_links)
    record.dependency_links = list(definition.dependency_links)
    record.semantic_mass = definition.semantic_mass
    record.subjective_time_scale = definition.subjective_time_scale
    record.sphere_id = definition.sphere_id
    record.parent_sphere_id = definition.parent_sphere_id
    record.leak_mode = definition.leak_mode
    if definition.attractor_weights is not None:
        record.attractor_weights = definition.attractor_weights
    return record


def runtime_report_to_entity_record(report_name: str, source_path: str, dependencies: Optional[Iterable[str]] = None) -> EntityRecord:
    return EntityRecord(
        canonical_id=f"OOL-REPORT-{report_name.upper().replace(' ', '-')}",
        object_type=ObjectType.REPORT,
        source_path=source_path,
        sector=Sector.REPORTING,
        orbit_index=0,
        phase=0.0,
        winding_number=0,
        relation_depth=1,
        epistemic_status=EpistemicStatus.DERIVED,
        provenance_links=[source_path],
        dependency_links=list(dependencies or []),
        semantic_mass=1.0,
        subjective_time_scale=1.0,
        attractor_weights=default_attractor_weights(),
    )


def build_runtime_catalog_snapshot(configs) -> Dict[str, Dict[str, object]]:
    return {cfg.code: scenario_config_to_entity_record(cfg).to_dict() for cfg in configs}


def build_core_runtime_snapshot() -> Dict[str, Dict[str, object]]:
    snapshot: Dict[str, Dict[str, object]] = {}
    for key, value in CANONICAL_RUNTIME_OBJECTS.items():
        snapshot[key] = {
            "canonical_id": value["canonical_id"],
            "object_type": value["object_type"].value,
            "sector": value["sector"].value,
            "semantic_layer": value["semantic_layer"].value,
            "epistemic_status": value["epistemic_status"].value,
            "source_path": value["source_path"],
        }
    return snapshot
