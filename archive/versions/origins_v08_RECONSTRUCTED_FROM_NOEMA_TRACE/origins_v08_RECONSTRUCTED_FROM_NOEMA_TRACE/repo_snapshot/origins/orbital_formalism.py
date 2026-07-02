from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

from .bindings import build_core_runtime_snapshot, scenario_config_to_entity_record
from .definitions import EpistemicStatus, ObjectType, Sector
from .registry import EntityRecord, default_attractor_weights
from .scenarios import ALL_SCENARIOS


@dataclass
class OrbitalState:
    canonical_id: str
    orbit_index: int
    phase: float
    winding_number: int
    relation_depth: int
    semantic_mass: float
    subjective_time_scale: float
    epistemic_status: str
    sector: str
    source_path: str
    provenance_links: List[str]
    dependency_links: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class OrbitalEdge:
    source_id: str
    target_id: str
    relation_type: str
    coupling_weight: float = 1.0
    phase_offset: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class OrbitalRepositorySnapshot:
    hierarchy: List[str]
    objects: Dict[str, Dict[str, object]]
    edges: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "hierarchy": list(self.hierarchy),
            "objects": dict(self.objects),
            "edges": list(self.edges),
        }


def entity_record_to_orbital_state(record: EntityRecord) -> OrbitalState:
    return OrbitalState(
        canonical_id=record.canonical_id,
        orbit_index=record.orbit_index,
        phase=record.phase,
        winding_number=record.winding_number,
        relation_depth=record.relation_depth,
        semantic_mass=record.semantic_mass,
        subjective_time_scale=record.subjective_time_scale,
        epistemic_status=record.epistemic_status.value,
        sector=record.sector.value,
        source_path=record.source_path,
        provenance_links=list(record.provenance_links),
        dependency_links=list(record.dependency_links),
    )


def build_repo_module_record(canonical_id: str, source_path: str, sector: Sector, object_type: ObjectType, epistemic_status: EpistemicStatus = EpistemicStatus.CANONICAL) -> EntityRecord:
    return EntityRecord(
        canonical_id=canonical_id,
        object_type=object_type,
        source_path=source_path,
        sector=sector,
        orbit_index=0,
        phase=0.0,
        winding_number=0,
        relation_depth=1,
        epistemic_status=epistemic_status,
        provenance_links=[source_path],
        dependency_links=[],
        semantic_mass=1.0,
        subjective_time_scale=1.0,
        attractor_weights=default_attractor_weights(),
    )


def build_scenario_orbital_states(configs: Iterable = ALL_SCENARIOS) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for cfg in configs:
        rec = scenario_config_to_entity_record(cfg, source_path="origins/scenarios.py")
        out[cfg.code] = entity_record_to_orbital_state(rec).to_dict()
    return out


def build_repo_module_states() -> Dict[str, Dict[str, object]]:
    module_records = [
        build_repo_module_record("OOL-MODULE-SCENARIOS", "origins/scenarios.py", Sector.CHEMISTRY, ObjectType.MODULE),
        build_repo_module_record("OOL-MODULE-SIMULATOR", "origins/simulator/universal.py", Sector.CORE, ObjectType.SIMULATOR),
        build_repo_module_record("OOL-MODULE-ANALYSIS", "origins/analysis/sweep.py", Sector.ANALYSIS, ObjectType.ANALYSIS),
        build_repo_module_record("OOL-MODULE-REGISTRY", "origins/registry.py", Sector.CORE, ObjectType.REGISTRY),
        build_repo_module_record("OOL-MODULE-DEFINITIONS", "origins/definitions.py", Sector.DOCS, ObjectType.DOCUMENT),
        build_repo_module_record("OOL-SCRIPT-RUN-SIMULATION", "scripts/run_simulation.py", Sector.SCRIPTS, ObjectType.SCRIPT, EpistemicStatus.WORKING),
        build_repo_module_record("OOL-SCRIPT-RUN-SWEEP", "scripts/run_sweep.py", Sector.SCRIPTS, ObjectType.SCRIPT, EpistemicStatus.WORKING),
    ]
    return {rec.canonical_id: entity_record_to_orbital_state(rec).to_dict() for rec in module_records}


def build_orbital_edges(configs: Iterable = ALL_SCENARIOS) -> List[Dict[str, object]]:
    edges: List[OrbitalEdge] = []
    for cfg in configs:
        rec = scenario_config_to_entity_record(cfg, source_path="origins/scenarios.py")
        edges.append(OrbitalEdge(source_id=rec.canonical_id, target_id="OOL-MODULE-SCENARIOS", relation_type="declared_in").to_dict())
        edges.append(OrbitalEdge(source_id=rec.canonical_id, target_id="OOL-MODULE-SIMULATOR", relation_type="executed_by").to_dict())
        edges.append(OrbitalEdge(source_id=rec.canonical_id, target_id="OOL-MODULE-ANALYSIS", relation_type="analyzed_by", coupling_weight=0.5).to_dict())
    edges.extend([
        OrbitalEdge(source_id="OOL-MODULE-SIMULATOR", target_id="OOL-MODULE-REGISTRY", relation_type="indexed_by").to_dict(),
        OrbitalEdge(source_id="OOL-MODULE-SIMULATOR", target_id="OOL-MODULE-DEFINITIONS", relation_type="constrained_by").to_dict(),
        OrbitalEdge(source_id="OOL-SCRIPT-RUN-SIMULATION", target_id="OOL-MODULE-SIMULATOR", relation_type="launches").to_dict(),
        OrbitalEdge(source_id="OOL-SCRIPT-RUN-SWEEP", target_id="OOL-MODULE-ANALYSIS", relation_type="launches").to_dict(),
    ])
    return edges


def build_orbital_repository_snapshot(configs: Iterable = ALL_SCENARIOS) -> OrbitalRepositorySnapshot:
    objects: Dict[str, Dict[str, object]] = {}
    objects.update(build_scenario_orbital_states(configs))
    objects.update(build_repo_module_states())
    core_runtime = build_core_runtime_snapshot()
    for key, value in core_runtime.items():
        objects[f"core::{key}"] = value
    hierarchy = [
        "relation",
        "identity",
        "memory",
        "process",
        "artifact",
    ]
    return OrbitalRepositorySnapshot(
        hierarchy=hierarchy,
        objects=objects,
        edges=build_orbital_edges(configs),
    )
