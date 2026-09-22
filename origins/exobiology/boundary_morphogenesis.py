from __future__ import annotations

import numpy as np


COLOCATED_BASELINE = "COLOCATED_BASELINE"
EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE = "EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE"
EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE = "EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE"

BOUNDARY_MORPHOGENESIS_MODES = (
    COLOCATED_BASELINE,
    EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
    EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
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



def redistribute_boundary_source_to_information_exterior(
    baseline_source: np.ndarray,
    information: np.ndarray,
) -> np.ndarray:
    """Redistribute a boundary source onto the exterior information interface.

    Whenever a nonzero exterior gradient exists, the instantaneous total source
    budget is preserved exactly up to floating-point roundoff:

        sum(S_B^redistributed) = sum(S_B^baseline)

    Only spatial geometry changes. If no information gradient exists, there is
    no defined exterior interface and the returned source is zero.

    This is a computational intervention, not a molecular law.
    """
    source = np.asarray(baseline_source, dtype=float)
    field = np.asarray(information, dtype=float)
    if source.shape != field.shape or source.ndim != 2:
        raise ValueError("baseline source and information must be same-shape 2D arrays")
    if source.size == 0:
        raise ValueError("source fields must be non-empty")
    if not np.isfinite(source).all():
        raise FloatingPointError("baseline source contains NaN/Inf")
    if float(np.min(source)) < -1e-12:
        raise ValueError("baseline source must be non-negative")

    source = np.maximum(source, 0.0)
    total = float(np.sum(source))
    if total <= 0.0:
        return np.zeros_like(source)

    gate = exterior_information_interface_gate(field)
    weighted = source * gate
    weighted_total = float(np.sum(weighted))
    if weighted_total <= 1e-15:
        return np.zeros_like(source)

    redistributed = weighted * (total / weighted_total)
    if not np.isfinite(redistributed).all():
        raise FloatingPointError("redistributed boundary source contains NaN/Inf")
    return redistributed
