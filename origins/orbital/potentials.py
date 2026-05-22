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
    import math
    c = max(0.0, min(1.0, coherence))
    d = max(0.0, min(1.0, defect))
    rel = max(1, relation_depth)
    m = max(0.0, semantic_mass)

    # Bloch sphere geometry: coherence = cos²(θ/2), defect = sin²(θ/2)
    # Reconstruct polar angle θ ∈ [0, π]
    theta = 2.0 * math.acos(math.sqrt(c))

    # V_EC: emergence cost — distance from |0⟩ (north pole = full coherence)
    # On S²: geodesic distance from north pole = θ, so V_EC ∝ theta * rel
    V_EC = (theta / math.pi) * rel

    # V_ZS: Zeta-Schrödinger constraint — penalizes defect weighted by mass
    # sin²(θ/2) = defect is the Bloch-native decoherence measure
    V_ZS = d * (1.0 + m)

    # V_rel: relational potential — Fubini-Study metric: 1/cos²(θ/2) = 1/c
    # Capped at rel*20 to avoid divergence at θ→π (south pole = total decoherence)
    V_rel = min(rel / max(c, 0.05), rel * 20.0)

    # V_mem: memory affinity weighted by mass
    V_mem = max(0.0, memory_affinity) * (1.0 + m)

    # V_def: raw defect (sin²) — independent decoherence penalty
    V_def = d

    # V_ext: external load (protocell count normalised)
    V_ext = max(0.0, external_load)

    return PotentialTerms(
        V_EC=V_EC,
        V_ZS=V_ZS,
        V_rel=V_rel,
        V_mem=V_mem,
        V_def=V_def,
        V_ext=V_ext,
    )
