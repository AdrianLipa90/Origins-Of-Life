from __future__ import annotations

import math


def compute_local_subjective_time(delta_t: float, radius: float, semantic_mass: float, coherence: float, defect: float) -> float:
    """Local subjective time with Bloch sphere metric correction.

    On S² with Fubini-Study metric, proper time is dilated by curvature:
      g = (1 + m) * cos²(θ/2) / (1 + r)
    where cos²(θ/2) = coherence and θ is the Bloch polar angle.
    High coherence (near north pole) → time flows slower (more stable).
    High defect (near south pole) → time flows faster (rapid decoherence).
    """
    r = max(0.0, radius)
    m = max(0.0, semantic_mass)
    c = max(1e-9, min(1.0, coherence))
    # Fubini-Study time dilation: coherence acts as gravitational redshift
    g = (1.0 + m) * c / (1.0 + r)
    return max(1e-9, delta_t * g)
