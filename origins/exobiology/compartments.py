from __future__ import annotations

import numpy as np


def _periodic_label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-neighbour connected components on a 2D periodic lattice."""
    array = np.asarray(mask, dtype=bool)
    nx, ny = array.shape
    labels = np.zeros((nx, ny), dtype=np.int32)
    component = 0

    for i in range(nx):
        for j in range(ny):
            if not array[i, j] or labels[i, j] != 0:
                continue
            component += 1
            labels[i, j] = component
            stack = [(i, j)]
            while stack:
                x, y = stack.pop()
                for xx, yy in (
                    ((x - 1) % nx, y),
                    ((x + 1) % nx, y),
                    (x, (y - 1) % ny),
                    (x, (y + 1) % ny),
                ):
                    if array[xx, yy] and labels[xx, yy] == 0:
                        labels[xx, yy] = component
                        stack.append((xx, yy))

    return labels, component


def bounded_compartment_observation(mask: np.ndarray) -> dict[str, object]:
    """Classify threshold-positive regions without mistaking global saturation for a compartment.

    The transport operators use periodic rolls, so both component connectivity
    and boundary detection are evaluated on a discrete torus. A uniformly TRUE
    mask has no internal/external
    interface on that torus and therefore cannot satisfy the bounded-system
    observable merely because a connected-component routine returns one label.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("compartment mask must be a 2D array")
    if array.size == 0:
        raise ValueError("compartment mask must be non-empty")

    labels, raw_count = _periodic_label(array)
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
