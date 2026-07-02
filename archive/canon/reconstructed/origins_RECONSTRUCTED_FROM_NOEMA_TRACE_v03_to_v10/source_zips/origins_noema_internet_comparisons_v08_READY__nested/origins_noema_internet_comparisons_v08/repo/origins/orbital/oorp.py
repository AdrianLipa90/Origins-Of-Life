from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from .memory import MemoryState, ReductionResidue, apply_memory_update
from .potentials import compute_potential_terms
from .state import OrbitalCoordinate


@dataclass
class OORPTrace:
    relation_score: float
    orchestration_score: float
    reduction_score: float
    memory_updated: bool
    residue: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_oorp_pipeline(coordinate: OrbitalCoordinate, memory_state: MemoryState, external_load: float = 0.0) -> OORPTrace:
    relation_score = max(0.0, 1.0 - coordinate.defect) * (1.0 + coordinate.coherence)
    potentials = compute_potential_terms(
        coherence=coordinate.coherence,
        defect=coordinate.defect,
        relation_depth=coordinate.relation_depth,
        semantic_mass=coordinate.semantic_mass,
        memory_affinity=memory_state.affinity.get(coordinate.canonical_id, 0.0),
        external_load=external_load,
    )
    orchestration_score = max(0.0, relation_score - potentials.V_tot)
    reduction_score = max(0.0, orchestration_score / (1.0 + coordinate.radius))
    residue = ReductionResidue(
        source_id=coordinate.canonical_id,
        closure_delta=coordinate.defect,
        truth_delta=coordinate.coherence,
        memory_delta=reduction_score,
        winding_delta=coordinate.omega,
    )
    apply_memory_update(memory_state, residue)
    return OORPTrace(
        relation_score=relation_score,
        orchestration_score=orchestration_score,
        reduction_score=reduction_score,
        memory_updated=True,
        residue=residue.to_dict(),
    )
