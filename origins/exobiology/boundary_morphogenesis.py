from __future__ import annotations

import numpy as np


COLOCATED_BASELINE = "COLOCATED_BASELINE"
EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE = "EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE"

BOUNDARY_MORPHOGENESIS_MODES = (
    COLOCATED_BASELINE,
    EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
)


def validate_boundary_morphogenesis_mode(mode: str) -> str:
    value = str(mode)
    if value not in BOUNDARY_MORPHOGENESIS_MODES:
        raise ValueError(
            f"unknown boundary morphogenesis mode {value!r}; "
            f"expected one of {BOUNDARY_MORPHOGENESIS_MODES!r}"
        )
    return value


def exterior_information_interface_gate(information: np.ndarray) -> np.ndarray:
    """Dimensionless outer-interface gate on a periodic 2-D lattice.

    The gate is positive where a site has less information mass than its
    nearest-neighbour mean:

        G = max(<I>_N - I, 0) / (<I>_N + I + eps)

    It therefore suppresses boundary production inside information-rich
    interiors and activates it immediately outside information gradients.

    This is a computational morphogenesis candidate. It is not a molecular
    membrane law and contains no tuned threshold.
    """
    field = np.asarray(information, dtype=float)
    if field.ndim != 2 or field.size == 0:
        raise ValueError("information field must be a non-empty 2D array")
    if not np.isfinite(field).all():
        raise FloatingPointError("information field contains NaN/Inf")
    if float(np.min(field)) < -1e-12:
        raise ValueError("information field must be non-negative")

    neighbors = 0.25 * (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
    )
    exterior = np.maximum(neighbors - field, 0.0)
    denominator = neighbors + field + 1e-15
    gate = np.divide(
        exterior,
        denominator,
        out=np.zeros_like(field),
        where=denominator > 0.0,
    )
    return np.clip(gate, 0.0, 1.0)
