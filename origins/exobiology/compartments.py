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


def _periodic_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """4-neighbor connected components on a periodic 2-D grid."""
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("component mask must be a 2D array")
    nx, ny = array.shape
    labels = np.zeros((nx, ny), dtype=int)
    components: list[list[tuple[int, int]]] = []
    label_id = 0

    for i in range(nx):
        for j in range(ny):
            if not array[i, j] or labels[i, j] != 0:
                continue
            label_id += 1
            stack = [(i, j)]
            labels[i, j] = label_id
            component: list[tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                component.append((x, y))
                for xx, yy in (
                    ((x - 1) % nx, y),
                    ((x + 1) % nx, y),
                    (x, (y - 1) % ny),
                    (x, (y + 1) % ny),
                ):
                    if array[xx, yy] and labels[xx, yy] == 0:
                        labels[xx, yy] = label_id
                        stack.append((xx, yy))
            components.append(component)
    return labels, components


def closed_boundary_compartment_observation(
    information: np.ndarray,
    boundary: np.ndarray,
    information_threshold: float,
    boundary_threshold: float,
) -> dict[str, object]:
    """Detect information-rich interiors enclosed by a closed boundary shell.

    A candidate bounded system requires a connected information-rich interior
    whose immediate periodic 4-neighbor exterior shell is entirely above the
    declared boundary threshold.  This is intentionally stricter than simple
    co-threshold colocalization.

    Threshold values are supplied by the runtime and are not tuned here.
    """
    info = np.asarray(information, dtype=float)
    bound = np.asarray(boundary, dtype=float)
    if info.shape != bound.shape or info.ndim != 2:
        raise ValueError("information and boundary fields must be same-shape 2D arrays")
    if info.size == 0:
        raise ValueError("fields must be non-empty")
    if not np.isfinite(info).all() or not np.isfinite(bound).all():
        raise FloatingPointError("compartment fields contain NaN/Inf")
    if information_threshold <= 0.0 or boundary_threshold <= 0.0:
        raise ValueError("compartment thresholds must be positive")

    info_mask = info >= float(information_threshold)
    boundary_mask = bound >= float(boundary_threshold)
    labels, components = _periodic_components(info_mask)
    nx, ny = info.shape
    total_pixels = int(info.size)

    accepted_labels: list[int] = []
    accepted_area = 0
    accepted_shell_pixels = 0
    shell_coverages: list[float] = []
    rejected_global_saturation = False

    for label_id, component in enumerate(components, start=1):
        comp_set = set(component)
        if len(comp_set) == total_pixels:
            rejected_global_saturation = True
            continue

        shell: set[tuple[int, int]] = set()
        for x, y in component:
            for xx, yy in (
                ((x - 1) % nx, y),
                ((x + 1) % nx, y),
                (x, (y - 1) % ny),
                (x, (y + 1) % ny),
            ):
                if (xx, yy) not in comp_set:
                    shell.add((xx, yy))

        if not shell:
            continue

        covered = sum(1 for x, y in shell if boundary_mask[x, y])
        coverage = float(covered / len(shell))
        shell_coverages.append(coverage)
        if covered == len(shell):
            accepted_labels.append(label_id)
            accepted_area += len(component)
            accepted_shell_pixels += len(shell)

    accepted_label_set = set(accepted_labels)
    accepted_mask = np.isin(labels, list(accepted_label_set)) if accepted_label_set else np.zeros_like(info_mask)
    accepted_component_count = len(accepted_labels)

    if accepted_component_count:
        status = "CLOSED_BOUNDARY_CANDIDATE"
    elif rejected_global_saturation:
        status = "GLOBAL_SATURATION_REJECTED"
    elif components:
        status = "INTERIOR_NOT_CLOSED"
    else:
        status = "NONE"

    return {
        "count": int(accepted_component_count),
        "raw_information_component_count": int(len(components)),
        "area_pixels": int(accepted_area),
        "shell_pixels": int(accepted_shell_pixels),
        "max_shell_coverage": float(max(shell_coverages) if shell_coverages else 0.0),
        "global_saturation": bool(rejected_global_saturation),
        "bounded_system_status": status,
        "labels": labels,
        "accepted_mask": accepted_mask,
    }
