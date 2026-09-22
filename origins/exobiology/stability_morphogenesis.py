from __future__ import annotations

import numpy as np

from .boundary_morphogenesis import exterior_information_interface_gate


SHELL_MAINTENANCE_OFF = "SHELL_MAINTENANCE_OFF"
LOW_BOUNDARY_EXTERIOR_MAINTENANCE_CANDIDATE = (
    "LOW_BOUNDARY_EXTERIOR_MAINTENANCE_CANDIDATE"
)

ISLAND_PRESERVATION_OFF = "ISLAND_PRESERVATION_OFF"
STATE_PEAK_ISLAND_PRESERVATION_CANDIDATE = (
    "STATE_PEAK_ISLAND_PRESERVATION_CANDIDATE"
)

SHELL_MAINTENANCE_MODES = (
    SHELL_MAINTENANCE_OFF,
    LOW_BOUNDARY_EXTERIOR_MAINTENANCE_CANDIDATE,
)
ISLAND_PRESERVATION_MODES = (
    ISLAND_PRESERVATION_OFF,
    STATE_PEAK_ISLAND_PRESERVATION_CANDIDATE,
)


def validate_shell_maintenance_mode(mode: str) -> str:
    value = str(mode)
    if value not in SHELL_MAINTENANCE_MODES:
        raise ValueError(
            f"unknown shell maintenance mode {value!r}; "
            f"expected one of {SHELL_MAINTENANCE_MODES!r}"
        )
    return value


def validate_island_preservation_mode(mode: str) -> str:
    value = str(mode)
    if value not in ISLAND_PRESERVATION_MODES:
        raise ValueError(
            f"unknown island preservation mode {value!r}; "
            f"expected one of {ISLAND_PRESERVATION_MODES!r}"
        )
    return value


def _validate_same_shape_nonnegative(
    source: np.ndarray,
    state: np.ndarray,
    *,
    source_name: str,
    state_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(source, dtype=float)
    field = np.asarray(state, dtype=float)
    if src.shape != field.shape or src.ndim != 2:
        raise ValueError(
            f"{source_name} and {state_name} must be same-shape 2D arrays"
        )
    if src.size == 0:
        raise ValueError("stability fields must be non-empty")
    if not np.isfinite(src).all() or not np.isfinite(field).all():
        raise FloatingPointError("stability field contains NaN/Inf")
    if float(np.min(src)) < -1e-12 or float(np.min(field)) < -1e-12:
        raise ValueError("stability fields must be non-negative")
    return np.maximum(src, 0.0), np.maximum(field, 0.0)


def redistribute_boundary_source_to_shell_deficits(
    boundary_source: np.ndarray,
    information: np.ndarray,
    boundary: np.ndarray,
) -> np.ndarray:
    """Conservatively favor weaker points on an existing information exterior.

    No detector threshold is used. The operator derives a continuous exterior
    gate from I and biases the current source budget toward lower-B sites on
    that exterior. When no exterior exists, the source is returned unchanged.

    The total instantaneous source budget is preserved whenever it is nonzero.
    """
    source, info = _validate_same_shape_nonnegative(
        boundary_source,
        information,
        source_name="boundary_source",
        state_name="information",
    )
    _, bound = _validate_same_shape_nonnegative(
        source,
        boundary,
        source_name="boundary_source",
        state_name="boundary",
    )

    total = float(np.sum(source))
    if total <= 0.0:
        return np.zeros_like(source)

    exterior = exterior_information_interface_gate(info)
    active = exterior > 0.0
    if not np.any(active):
        return source.copy()

    mean_boundary = float(np.mean(bound[active]))
    scale = mean_boundary + 1e-15
    deficit_weight = 1.0 / (1.0 + bound / scale)
    weighted = source * exterior * deficit_weight
    weighted_total = float(np.sum(weighted))
    if weighted_total <= 1e-15:
        return source.copy()

    redistributed = weighted * (total / weighted_total)
    if not np.isfinite(redistributed).all():
        raise FloatingPointError("shell-maintenance redistribution produced NaN/Inf")
    return redistributed


def state_peak_gate(information: np.ndarray) -> np.ndarray:
    """Continuous gate favoring current local information maxima."""
    field = np.asarray(information, dtype=float)
    if field.ndim != 2 or field.size == 0:
        raise ValueError("information must be a non-empty 2D array")
    if not np.isfinite(field).all():
        raise FloatingPointError("information contains NaN/Inf")
    if float(np.min(field)) < -1e-12:
        raise ValueError("information must be non-negative")

    field = np.maximum(field, 0.0)
    neighbors = 0.25 * (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
    )
    excess = np.maximum(field - neighbors, 0.0)
    denominator = field + neighbors + 1e-15
    return np.divide(
        excess,
        denominator,
        out=np.zeros_like(field),
        where=denominator > 0.0,
    )


def redistribute_information_source_to_state_peaks(
    information_source: np.ndarray,
    information: np.ndarray,
) -> np.ndarray:
    """Conservatively reinforce existing localized information peaks.

    This operator does not know about connected-component labels or detection
    thresholds. It only redistributes the already available source budget away
    from flat bridges/background toward current local maxima. If the state has
    no local peak, the source is returned unchanged.
    """
    source, info = _validate_same_shape_nonnegative(
        information_source,
        information,
        source_name="information_source",
        state_name="information",
    )
    total = float(np.sum(source))
    if total <= 0.0:
        return np.zeros_like(source)

    gate = state_peak_gate(info)
    weighted = source * gate
    weighted_total = float(np.sum(weighted))
    if weighted_total <= 1e-15:
        return source.copy()

    redistributed = weighted * (total / weighted_total)
    if not np.isfinite(redistributed).all():
        raise FloatingPointError("island-preservation redistribution produced NaN/Inf")
    return redistributed
