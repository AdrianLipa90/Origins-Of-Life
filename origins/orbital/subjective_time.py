from __future__ import annotations


def compute_local_subjective_time(delta_t: float, radius: float, semantic_mass: float, coherence: float, defect: float) -> float:
    r = max(0.0, radius)
    m = max(0.0, semantic_mass)
    c = max(0.0, min(1.0, coherence))
    d = max(0.0, defect)
    g = (1.0 + m + c) / (1.0 + r + d)
    return max(1e-9, delta_t * g)
