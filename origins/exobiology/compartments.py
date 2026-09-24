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


def periodic_component_topology(mask: np.ndarray) -> dict[str, object]:
    """Characterize 4-neighbor components and non-contractible winding on a torus.

    Each connected component is lifted from the periodic Nx x Ny lattice to
    integer coordinates in Z^2. Reaching the same periodic site with a
    different lifted coordinate reveals a non-contractible cycle. The
    difference must be an integer multiple of the domain size and gives the
    component winding in each periodic direction.

    This is a discrete topological diagnostic; it does not change the model.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("topology mask must be a non-empty 2D array")

    nx, ny = array.shape
    visited = np.zeros((nx, ny), dtype=bool)
    records: list[dict[str, object]] = []

    for i in range(nx):
        for j in range(ny):
            if not array[i, j] or visited[i, j]:
                continue

            start = (i, j)
            stack = [start]
            visited[start] = True
            lift: dict[tuple[int, int], tuple[int, int]] = {start: (i, j)}
            cells: list[tuple[int, int]] = []
            winding_vectors: set[tuple[int, int]] = set()

            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                ux, uy = lift[(x, y)]

                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    xx = (x + dx) % nx
                    yy = (y + dy) % ny
                    if not array[xx, yy]:
                        continue

                    proposed = (ux + dx, uy + dy)
                    key = (xx, yy)
                    if key not in lift:
                        lift[key] = proposed
                        visited[key] = True
                        stack.append(key)
                    else:
                        prior = lift[key]
                        delta_x = proposed[0] - prior[0]
                        delta_y = proposed[1] - prior[1]
                        if delta_x % nx != 0 or delta_y % ny != 0:
                            raise RuntimeError("inconsistent periodic lift")
                        wx = delta_x // nx
                        wy = delta_y // ny
                        if wx != 0 or wy != 0:
                            winding_vectors.add((int(wx), int(wy)))

            wraps_x = any(wx != 0 for wx, _ in winding_vectors)
            wraps_y = any(wy != 0 for _, wy in winding_vectors)
            area = len(cells)
            records.append(
                {
                    "label_id": int(len(records) + 1),
                    "area_pixels": int(area),
                    "area_fraction": float(area / array.size),
                    "wraps_x": bool(wraps_x),
                    "wraps_y": bool(wraps_y),
                    "noncontractible": bool(wraps_x or wraps_y),
                    "winding_vectors": sorted(winding_vectors),
                }
            )

    largest = max((r["area_pixels"] for r in records), default=0)
    return {
        "component_count": int(len(records)),
        "largest_component_area_pixels": int(largest),
        "largest_component_fraction": float(largest / array.size),
        "noncontractible_component_count": int(
            sum(1 for r in records if r["noncontractible"])
        ),
        "wraps_x_component_count": int(sum(1 for r in records if r["wraps_x"])),
        "wraps_y_component_count": int(sum(1 for r in records if r["wraps_y"])),
        "any_noncontractible": bool(any(r["noncontractible"] for r in records)),
        "components": records,
    }


def contractible_component_geometries(
    information: np.ndarray,
    boundary: np.ndarray,
    information_threshold: float,
    boundary_threshold: float,
) -> list[dict[str, object]]:
    """Return event-level geometry for each contractible information component.

    The diagnostic is read-only. It reports threshold margins, shell coverage,
    and the minimum toroidal Manhattan separation from any non-contractible
    information component. A separation of 2 means one below-threshold lattice
    site lies between the contractible island and the percolating background.
    """
    info = np.asarray(information, dtype=float)
    bound = np.asarray(boundary, dtype=float)
    if info.shape != bound.shape or info.ndim != 2 or info.size == 0:
        raise ValueError("information and boundary must be same-shape non-empty 2D arrays")
    if not np.isfinite(info).all() or not np.isfinite(bound).all():
        raise FloatingPointError("component geometry fields contain NaN/Inf")
    if information_threshold <= 0.0 or boundary_threshold <= 0.0:
        raise ValueError("component geometry thresholds must be positive")

    info_mask = info >= float(information_threshold)
    boundary_mask = bound >= float(boundary_threshold)
    labels, components = _periodic_components(info_mask)
    topology = periodic_component_topology(info_mask)
    topology_by_label = {
        int(record["label_id"]): record
        for record in topology["components"]
    }
    nx, ny = info.shape

    noncontractible_cells: list[tuple[int, int]] = []
    for label_id, component in enumerate(components, start=1):
        if bool(topology_by_label[label_id]["noncontractible"]):
            noncontractible_cells.extend(component)

    def toroidal_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return int(min(dx, nx - dx) + min(dy, ny - dy))

    records: list[dict[str, object]] = []
    for label_id, component in enumerate(components, start=1):
        topo = topology_by_label[label_id]
        if bool(topo["noncontractible"]):
            continue

        comp_set = set(component)
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

        covered = sum(1 for x, y in shell if boundary_mask[x, y])
        shell_pixels = len(shell)
        shell_coverage = float(covered / shell_pixels) if shell_pixels else 0.0

        values = np.asarray([info[x, y] for x, y in component], dtype=float)
        margins = values - float(information_threshold)

        nearest_noncontractible_distance: int | None = None
        if noncontractible_cells:
            nearest_noncontractible_distance = min(
                toroidal_distance(cell, other)
                for cell in component
                for other in noncontractible_cells
            )

        records.append(
            {
                "label_id": int(label_id),
                "area_pixels": int(len(component)),
                "shell_pixels": int(shell_pixels),
                "covered_shell_pixels": int(covered),
                "missing_shell_pixels": int(shell_pixels - covered),
                "shell_coverage": float(shell_coverage),
                "closed_shell": bool(shell_pixels > 0 and covered == shell_pixels),
                "information_min": float(np.min(values)),
                "information_mean": float(np.mean(values)),
                "information_max": float(np.max(values)),
                "information_margin_min": float(np.min(margins)),
                "information_margin_mean": float(np.mean(margins)),
                "information_margin_max": float(np.max(margins)),
                "nearest_noncontractible_distance": nearest_noncontractible_distance,
                "nearest_noncontractible_gap_cells": (
                    None
                    if nearest_noncontractible_distance is None
                    else int(max(0, nearest_noncontractible_distance - 1))
                ),
            }
        )

    records.sort(
        key=lambda record: (
            bool(record["closed_shell"]),
            float(record["shell_coverage"]),
            int(record["area_pixels"]),
        ),
        reverse=True,
    )
    return records


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
    topology = periodic_component_topology(info_mask)
    topology_by_label = {
        int(record["label_id"]): record
        for record in topology["components"]
    }
    nx, ny = info.shape
    total_pixels = int(info.size)

    accepted_labels: list[int] = []
    accepted_area = 0
    accepted_shell_pixels = 0
    accepted_interface_edges = 0
    shell_coverages: list[float] = []
    rejected_global_saturation = False
    rejected_noncontractible = 0
    best_contractible_shell_coverage = 0.0
    best_contractible_area_pixels = 0
    best_contractible_shell_pixels = 0
    best_contractible_missing_shell_pixels = 0

    for label_id, component in enumerate(components, start=1):
        comp_set = set(component)
        if len(comp_set) == total_pixels:
            rejected_global_saturation = True
            continue

        component_topology = topology_by_label[label_id]
        if bool(component_topology["noncontractible"]):
            rejected_noncontractible += 1
            continue

        shell: set[tuple[int, int]] = set()
        interface_edges = 0
        for x, y in component:
            for xx, yy in (
                ((x - 1) % nx, y),
                ((x + 1) % nx, y),
                (x, (y - 1) % ny),
                (x, (y + 1) % ny),
            ):
                if (xx, yy) not in comp_set:
                    shell.add((xx, yy))
                    interface_edges += 1

        if not shell:
            continue

        covered = sum(1 for x, y in shell if boundary_mask[x, y])
        coverage = float(covered / len(shell))
        shell_coverages.append(coverage)

        if (
            coverage > best_contractible_shell_coverage
            or (
                coverage == best_contractible_shell_coverage
                and len(component) > best_contractible_area_pixels
            )
        ):
            best_contractible_shell_coverage = coverage
            best_contractible_area_pixels = len(component)
            best_contractible_shell_pixels = len(shell)
            best_contractible_missing_shell_pixels = len(shell) - covered

        if covered == len(shell):
            accepted_labels.append(label_id)
            accepted_area += len(component)
            accepted_shell_pixels += len(shell)
            accepted_interface_edges += interface_edges

    accepted_label_set = set(accepted_labels)
    accepted_mask = np.isin(labels, list(accepted_label_set)) if accepted_label_set else np.zeros_like(info_mask)
    accepted_component_count = len(accepted_labels)

    if accepted_component_count:
        status = "CLOSED_BOUNDARY_CANDIDATE"
    elif rejected_global_saturation:
        status = "GLOBAL_SATURATION_REJECTED"
    elif components and rejected_noncontractible == len(components):
        status = "NONCONTRACTIBLE_INTERIOR_ONLY"
    elif components:
        status = "INTERIOR_NOT_CLOSED"
    else:
        status = "NONE"

    occupancy_fraction = float(accepted_area / total_pixels)

    return {
        "count": int(accepted_component_count),
        "raw_information_component_count": int(len(components)),
        "contractible_information_component_count": int(
            len(components) - int(topology["noncontractible_component_count"])
        ),
        "noncontractible_information_component_count": int(
            topology["noncontractible_component_count"]
        ),
        "rejected_noncontractible_component_count": int(rejected_noncontractible),
        # Backward-compatible shared-observation alias.  It refers to the
        # information-rich interior components before the closed-shell gate.
        "raw_component_count": int(len(components)),
        "area_pixels": int(accepted_area),
        "occupancy_fraction": occupancy_fraction,
        "shell_pixels": int(accepted_shell_pixels),
        "interface_edge_count": int(accepted_interface_edges),
        "max_shell_coverage": float(max(shell_coverages) if shell_coverages else 0.0),
        "best_contractible_shell_coverage": float(best_contractible_shell_coverage),
        "best_contractible_area_pixels": int(best_contractible_area_pixels),
        "best_contractible_shell_pixels": int(best_contractible_shell_pixels),
        "best_contractible_missing_shell_pixels": int(
            best_contractible_missing_shell_pixels
        ),
        "global_saturation": bool(rejected_global_saturation),
        "bounded_system_status": status,
        "labels": labels,
        "accepted_mask": accepted_mask,
    }
