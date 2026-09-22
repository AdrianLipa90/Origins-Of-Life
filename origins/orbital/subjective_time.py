from __future__ import annotations

import math


def compute_local_subjective_time(delta_t: float, radius: float, semantic_mass: float, coherence: float, defect: float) -> float:
    """Candidate local-clock scaling for the semantic/orbital layer.

    The expression uses coherence as a bounded model coordinate:
      g = (1 + m) * cos²(theta/2) / (1 + r)

    It is not a derivation of relativistic proper time or gravitational
    redshift.  Higher coherence produces a larger retained local interval.
    """
    r = max(0.0, radius)
    m = max(0.0, semantic_mass)
    c = max(1e-9, min(1.0, coherence))
    # Dimensionless candidate clock factor; physical binding remains open.
    g = (1.0 + m) * c / (1.0 + r)
    return max(1e-9, delta_t * g)
