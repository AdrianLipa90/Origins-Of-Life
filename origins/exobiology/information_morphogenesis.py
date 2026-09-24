from __future__ import annotations

import numpy as np


DISTRIBUTED_INFORMATION_BASELINE = "DISTRIBUTED_INFORMATION_BASELINE"
PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE = "PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE"

INFORMATION_MORPHOGENESIS_MODES = (
    DISTRIBUTED_INFORMATION_BASELINE,
    PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
)


def validate_information_morphogenesis_mode(mode: str) -> str:
    value = str(mode)
    if value not in INFORMATION_MORPHOGENESIS_MODES:
        raise ValueError(
            f"unknown information morphogenesis mode {value!r}; "
            f"expected one of {INFORMATION_MORPHOGENESIS_MODES!r}"
        )
    return value


def local_source_peak_gate(source: np.ndarray) -> np.ndarray:
    """Dimensionless gate selecting local source peaks on a periodic lattice."""
    field = np.asarray(source, dtype=float)
    if field.ndim != 2 or field.size == 0:
        raise ValueError("information source must be a non-empty 2D array")
    if not np.isfinite(field).all():
        raise FloatingPointError("information source contains NaN/Inf")
    if float(np.min(field)) < -1e-12:
        raise ValueError("information source must be non-negative")

    field = np.maximum(field, 0.0)
    neighbors = 0.25 * (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
    )
    peaks = np.maximum(field - neighbors, 0.0)
    denominator = field + neighbors + 1e-15
    gate = np.divide(
        peaks,
        denominator,
        out=np.zeros_like(field),
        where=denominator > 0.0,
    )
    return np.clip(gate, 0.0, 1.0)


def redistribute_information_source_to_local_peaks(
    baseline_source: np.ndarray,
) -> np.ndarray:
    """Preserve total information-source budget while concentrating local peaks.

    For a heterogeneous source field, only spatial allocation changes:

        sum(S_I^redistributed) = sum(S_I^baseline).

    If no local peak exists, the baseline source is returned unchanged.  No
    detection threshold or tuned parameter is introduced.
    """
    source = np.asarray(baseline_source, dtype=float)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("baseline information source must be a non-empty 2D array")
    if not np.isfinite(source).all():
        raise FloatingPointError("baseline information source contains NaN/Inf")
    if float(np.min(source)) < -1e-12:
        raise ValueError("baseline information source must be non-negative")

    source = np.maximum(source, 0.0)
    total = float(np.sum(source))
    if total <= 0.0:
        return np.zeros_like(source)

    gate = local_source_peak_gate(source)
    weighted = source * gate
    weighted_total = float(np.sum(weighted))
    if weighted_total <= 1e-15:
        return source.copy()

    redistributed = weighted * (total / weighted_total)
    if not np.isfinite(redistributed).all():
        raise FloatingPointError("redistributed information source contains NaN/Inf")
    return redistributed
