from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .definitions import AttractorId, AttractorWeights, EpistemicStatus, ObjectType, Sector


@dataclass
class EntityRecord:
    canonical_id: str
    object_type: ObjectType
    source_path: str
    sector: Sector
    orbit_index: int
    phase: float
    winding_number: int
    relation_depth: int
    epistemic_status: EpistemicStatus
    provenance_links: List[str] = field(default_factory=list)
    dependency_links: List[str] = field(default_factory=list)
    semantic_mass: float = 1.0
    subjective_time_scale: float = 1.0
    sphere_id: Optional[str] = None
    parent_sphere_id: Optional[str] = None
    attractor_weights: Optional[AttractorWeights] = None
    leak_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "canonical_id": self.canonical_id,
            "object_type": self.object_type.value,
            "source_path": self.source_path,
            "sector": self.sector.value,
            "orbit_index": self.orbit_index,
            "phase": self.phase,
            "winding_number": self.winding_number,
            "relation_depth": self.relation_depth,
            "epistemic_status": self.epistemic_status.value,
            "provenance_links": list(self.provenance_links),
            "dependency_links": list(self.dependency_links),
            "semantic_mass": self.semantic_mass,
            "subjective_time_scale": self.subjective_time_scale,
            "sphere_id": self.sphere_id,
            "parent_sphere_id": self.parent_sphere_id,
            "attractor_weights": {k.value: v for k, v in self.attractor_weights.weights.items()} if self.attractor_weights else None,
            "leak_mode": self.leak_mode,
        }


def default_attractor_weights() -> AttractorWeights:
    return AttractorWeights(
        weights={
            AttractorId.ATTRACTOR_EC: 0.5,
            AttractorId.ATTRACTOR_ZS: 0.5,
            AttractorId.ATTRACTOR_LLM_TEMP: 0.0,
        }
    ).normalized()


def build_scenario_entity_record(canonical_id: str, source_path: str, orbit_index: int = 2, phase: float = 0.0) -> EntityRecord:
    return EntityRecord(
        canonical_id=canonical_id,
        object_type=ObjectType.SCENARIO,
        source_path=source_path,
        sector=Sector.CHEMISTRY,
        orbit_index=orbit_index,
        phase=phase,
        winding_number=0,
        relation_depth=1,
        epistemic_status=EpistemicStatus.WORKING,
        attractor_weights=default_attractor_weights(),
    )
