from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def bounded_compartment_observation(mask: np.ndarray) -> dict[str, object]:
    """Classify threshold-positive regions without mistaking global saturation for a compartment.

    The transport operators use periodic rolls, so the natural diagnostic
    domain is a discrete torus. A uniformly TRUE mask has no internal/external
    interface on that torus and therefore cannot satisfy the bounded-system
    observable merely because a connected-component routine returns one label.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("compartment mask must be a 2D array")
    if array.size == 0:
        raise ValueError("compartment mask must be non-empty")

    labels, raw_count = label(array)
    area_pixels = int(array.sum())
    total_pixels = int(array.size)
    occupancy_fraction = float(area_pixels / total_pixels)

    # Count each periodic nearest-neighbor interface edge once.
    interface_edge_count = int(
        np.count_nonzero(array != np.roll(array, -1, axis=0))
        + np.count_nonzero(array != np.roll(array, -1, axis=1))
    )

    global_saturation = bool(
        area_pixels == total_pixels and interface_edge_count == 0
    )
    bounded_count = 0 if global_saturation else int(raw_count)

    if global_saturation:
        status = "GLOBAL_SATURATION"
    elif bounded_count > 0:
        status = "LOCALIZED_CANDIDATE"
    else:
        status = "NONE"

    return {
        "count": bounded_count,
        "raw_component_count": int(raw_count),
        "area_pixels": area_pixels,
        "occupancy_fraction": occupancy_fraction,
        "interface_edge_count": interface_edge_count,
        "global_saturation": global_saturation,
        "bounded_system_status": status,
        "labels": labels,
    }
