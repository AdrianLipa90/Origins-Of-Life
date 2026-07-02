from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List


@dataclass
class ReductionResidue:
    source_id: str
    closure_delta: float
    truth_delta: float
    memory_delta: float
    winding_delta: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class MemoryState:
    residues: List[ReductionResidue] = field(default_factory=list)
    affinity: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "residues": [r.to_dict() for r in self.residues],
            "affinity": dict(self.affinity),
        }


def apply_memory_update(memory_state: MemoryState, residue: ReductionResidue) -> MemoryState:
    memory_state.residues.append(residue)
    memory_state.affinity[residue.source_id] = memory_state.affinity.get(residue.source_id, 0.0) + residue.memory_delta
    return memory_state
