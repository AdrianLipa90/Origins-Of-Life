from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class PotentialTerms:
    V_EC: float
    V_ZS: float
    V_rel: float
    V_mem: float
    V_def: float
    V_ext: float

    @property
    def V_tot(self) -> float:
        return self.V_EC + self.V_ZS + self.V_rel + self.V_mem + self.V_def + self.V_ext

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data["V_tot"] = self.V_tot
        return data


def compute_potential_terms(coherence: float, defect: float, relation_depth: int, semantic_mass: float, memory_affinity: float = 0.0, external_load: float = 0.0) -> PotentialTerms:
    c = max(0.0, min(1.0, coherence))
    d = max(0.0, defect)
    rel = max(1, relation_depth)
    m = max(0.0, semantic_mass)
    return PotentialTerms(
        V_EC=(1.0 - c) * rel,
        V_ZS=d * (1.0 + m),
        V_rel=rel / (1.0 + c),
        V_mem=max(0.0, memory_affinity) * (1.0 + m),
        V_def=d,
        V_ext=max(0.0, external_load),
    )
