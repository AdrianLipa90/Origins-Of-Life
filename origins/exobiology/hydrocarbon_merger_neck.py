from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .boundary_morphogenesis import EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE
from .compartments import _periodic_components, periodic_component_topology
from .factory import create_exotic_candidate_simulator
from .information_morphogenesis import PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE


SCHEMA = "ORIGINS_HYDROCARBON_MERGER_NECK_V0_1"

EVENT_SINGLE_PIXEL_NECK = "SINGLE_PIXEL_NECK"
EVENT_MULTI_PIXEL_NECK = "MULTI_PIXEL_NECK"
EVENT_NO_MERGER_WITHIN_HORIZON = "NO_MERGER_WITHIN_HORIZON"
EVENT_MERGER_WITHOUT_RESOLVED_NECK = "MERGER_WITHOUT_RESOLVED_NECK"


@dataclass(frozen=True)
class MergerNeckResult:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    last_closed_step: int | None
    first_loss_step: int | None
    event: str
    prior_accepted_component_area_pixels: int | None
    prior_noncontractible_component_area_pixels: int | None
    pre_transition_min_periodic_manhattan_distance: int | None
    pre_transition_inactive_gap_pixels: int | None
    current_descendant_area_pixels: int | None
    current_descendant_new_active_pixels: int | None
    minimum_new_active_bridge_pixels: int | None
    current_descendant_noncontractible: bool | None
    physical_binding: str
    model_mutation_allowed: bool = False
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _periodic_manhattan(
    left: tuple[int, int],
    right: tuple[int, int],
    shape: tuple[int, int],
) -> int:
    nx, ny = shape
    dx = abs(int(left[0]) - int(right[0]))
    dy = abs(int(left[1]) - int(right[1]))
    dx = min(dx, nx - dx)
    dy = min(dy, ny - dy)
    return int(dx + dy)


def minimum_periodic_set_distance(
    left_cells: Iterable[tuple[int, int]],
    right_cells: Iterable[tuple[int, int]],
    shape: tuple[int, int],
) -> int:
    left = tuple((int(x), int(y)) for x, y in left_cells)
    right = tuple((int(x), int(y)) for x, y in right_cells)
    if not left or not right:
        raise ValueError("cell sets must be non-empty")
    return min(
        _periodic_manhattan(a, b, shape)
        for a in left
        for b in right
    )


def minimum_new_active_bridge_cost(
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    source_cells: Iterable[tuple[int, int]],
    target_cells: Iterable[tuple[int, int]],
) -> int | None:
    """Minimum number of newly-active cells needed to connect source to target.

    The path is restricted to the current active information mask. Entering a
    cell that was already active at the previous step costs 0; entering a newly
    active cell costs 1. Because source and target were disconnected previously,
    a true merger must have strictly positive cost.
    """
    previous = np.asarray(previous_mask, dtype=bool)
    current = np.asarray(current_mask, dtype=bool)
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("previous and current masks must be same-shape 2D arrays")
    if previous.size == 0:
        raise ValueError("masks must be non-empty")

    nx, ny = current.shape
    sources = tuple((int(x), int(y)) for x, y in source_cells)
    targets = {(int(x), int(y)) for x, y in target_cells}
    if not sources or not targets:
        raise ValueError("source and target sets must be non-empty")

    inf = np.iinfo(np.int32).max
    distance = np.full((nx, ny), inf, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()

    for x, y in sources:
        if not current[x, y]:
            continue
        distance[x, y] = 0
        queue.appendleft((x, y))

    while queue:
        x, y = queue.popleft()
        base = int(distance[x, y])
        if (x, y) in targets:
            return base

        for xx, yy in (
            ((x - 1) % nx, y),
            ((x + 1) % nx, y),
            (x, (y - 1) % ny),
            (x, (y + 1) % ny),
        ):
            if not current[xx, yy]:
                continue
            cost = 0 if previous[xx, yy] else 1
            candidate = base + cost
            if candidate < int(distance[xx, yy]):
                distance[xx, yy] = candidate
                if cost == 0:
                    queue.appendleft((xx, yy))
                else:
                    queue.append((xx, yy))

    return None


def diagnose_merger_neck(
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    island_cells: Iterable[tuple[int, int]],
    background_cells: Iterable[tuple[int, int]],
) -> dict[str, object]:
    previous = np.asarray(previous_mask, dtype=bool)
    current = np.asarray(current_mask, dtype=bool)
    island = frozenset((int(x), int(y)) for x, y in island_cells)
    background = frozenset((int(x), int(y)) for x, y in background_cells)
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("previous and current masks must be same-shape 2D arrays")
    if not island or not background:
        raise ValueError("island and background cell sets must be non-empty")

    gap_distance = minimum_periodic_set_distance(
        island,
        background,
        previous.shape,
    )
    bridge_cost = minimum_new_active_bridge_cost(
        previous,
        current,
        island,
        background,
    )

    labels, components = _periodic_components(current)
    descendant_labels = {
        int(labels[x, y])
        for x, y in island
        if int(labels[x, y]) > 0
    }
    descendant_cells: set[tuple[int, int]] = set()
    for label_id in descendant_labels:
        descendant_cells.update(components[label_id - 1])

    new_active = current & ~previous
    descendant_new = sum(
        1
        for x, y in descendant_cells
        if bool(new_active[x, y])
    )

    topology = periodic_component_topology(current)
    topo_by_label = {
        int(record["label_id"]): record
        for record in topology["components"]
    }
    descendant_noncontractible = any(
        bool(topo_by_label[label_id]["noncontractible"])
        for label_id in descendant_labels
    )

    if bridge_cost is None:
        event = EVENT_MERGER_WITHOUT_RESOLVED_NECK
    elif bridge_cost == 1:
        event = EVENT_SINGLE_PIXEL_NECK
    else:
        event = EVENT_MULTI_PIXEL_NECK

    return {
        "event": event,
        "pre_transition_min_periodic_manhattan_distance": int(gap_distance),
        "pre_transition_inactive_gap_pixels": int(max(0, gap_distance - 1)),
        "current_descendant_area_pixels": int(len(descendant_cells)),
        "current_descendant_new_active_pixels": int(descendant_new),
        "minimum_new_active_bridge_pixels": (
            None if bridge_cost is None else int(bridge_cost)
        ),
        "current_descendant_noncontractible": bool(descendant_noncontractible),
    }


def _component_records(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    labels, components = _periodic_components(mask)
    topology = periodic_component_topology(mask)
    topo_by_label = {
        int(record["label_id"]): record
        for record in topology["components"]
    }
    records = []
    for label_id, cells in enumerate(components, start=1):
        records.append(
            {
                "label_id": int(label_id),
                "cells": frozenset(cells),
                "area_pixels": int(len(cells)),
                "noncontractible": bool(
                    topo_by_label[label_id]["noncontractible"]
                ),
            }
        )
    return labels, records


def run_hydrocarbon_merger_neck_case(
    scenario,
    *,
    seed: int,
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> MergerNeckResult:
    if getattr(scenario, "biochemistry_profile", None) != "HYDROCARBON_CANDIDATE":
        raise ValueError("merger-neck diagnostic requires HYDROCARBON_CANDIDATE")
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be positive")

    sim = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=int(seed),
        information_morphogenesis=PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        boundary_morphogenesis=EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
    )
    sim.initialize()

    last_closed_step: int | None = None
    first_loss_step: int | None = None
    previous_mask: np.ndarray | None = None
    previous_observation: dict[str, object] | None = None
    result_fields: dict[str, object] | None = None
    island_area: int | None = None
    background_area: int | None = None
    was_closed = False

    for step in range(1, int(horizon_steps) + 1):
        sim.step()
        observation = sim.candidate_compartments()
        closed = int(observation["count"]) > 0
        current_mask = np.asarray(sim.I) >= float(
            sim.parameters.information_threshold
        )

        if closed:
            last_closed_step = step
            previous_mask = current_mask.copy()
            previous_observation = dict(observation)
        elif was_closed:
            first_loss_step = step
            if previous_mask is None or previous_observation is None:
                raise RuntimeError("missing previous closed state")

            prev_labels, prev_records = _component_records(previous_mask)
            accepted_mask = np.asarray(
                previous_observation["accepted_mask"],
                dtype=bool,
            )
            accepted_labels = sorted(
                int(value)
                for value in np.unique(prev_labels[accepted_mask])
                if int(value) > 0
            )
            noncontractible = [
                record
                for record in prev_records
                if bool(record["noncontractible"])
            ]

            candidates: list[tuple[int, dict[str, object], dict[str, object]]] = []
            for label_id in accepted_labels:
                island = prev_records[label_id - 1]
                for background in noncontractible:
                    diagnosis = diagnose_merger_neck(
                        previous_mask,
                        current_mask,
                        island["cells"],
                        background["cells"],
                    )
                    cost = diagnosis["minimum_new_active_bridge_pixels"]
                    if (
                        diagnosis["current_descendant_noncontractible"]
                        and cost is not None
                    ):
                        candidates.append((int(cost), island, background))

            if candidates:
                _, island, background = min(
                    candidates,
                    key=lambda item: (
                        item[0],
                        item[1]["area_pixels"],
                        -item[2]["area_pixels"],
                    ),
                )
                result_fields = diagnose_merger_neck(
                    previous_mask,
                    current_mask,
                    island["cells"],
                    background["cells"],
                )
                island_area = int(island["area_pixels"])
                background_area = int(background["area_pixels"])
            break

        was_closed = closed

    claim = sim.claim_status()

    if first_loss_step is None or result_fields is None:
        return MergerNeckResult(
            schema=SCHEMA,
            profile=str(claim["profile"]),
            runtime=str(claim["runtime"]),
            seed=int(seed),
            horizon_steps=int(horizon_steps),
            last_closed_step=last_closed_step,
            first_loss_step=first_loss_step,
            event=EVENT_NO_MERGER_WITHIN_HORIZON,
            prior_accepted_component_area_pixels=None,
            prior_noncontractible_component_area_pixels=None,
            pre_transition_min_periodic_manhattan_distance=None,
            pre_transition_inactive_gap_pixels=None,
            current_descendant_area_pixels=None,
            current_descendant_new_active_pixels=None,
            minimum_new_active_bridge_pixels=None,
            current_descendant_noncontractible=None,
            physical_binding=str(claim["physical_binding"]),
        )

    return MergerNeckResult(
        schema=SCHEMA,
        profile=str(claim["profile"]),
        runtime=str(claim["runtime"]),
        seed=int(seed),
        horizon_steps=int(horizon_steps),
        last_closed_step=last_closed_step,
        first_loss_step=first_loss_step,
        event=str(result_fields["event"]),
        prior_accepted_component_area_pixels=island_area,
        prior_noncontractible_component_area_pixels=background_area,
        pre_transition_min_periodic_manhattan_distance=int(
            result_fields["pre_transition_min_periodic_manhattan_distance"]
        ),
        pre_transition_inactive_gap_pixels=int(
            result_fields["pre_transition_inactive_gap_pixels"]
        ),
        current_descendant_area_pixels=int(
            result_fields["current_descendant_area_pixels"]
        ),
        current_descendant_new_active_pixels=int(
            result_fields["current_descendant_new_active_pixels"]
        ),
        minimum_new_active_bridge_pixels=int(
            result_fields["minimum_new_active_bridge_pixels"]
        ),
        current_descendant_noncontractible=bool(
            result_fields["current_descendant_noncontractible"]
        ),
        physical_binding=str(claim["physical_binding"]),
    )


def run_matched_hydrocarbon_merger_neck(
    scenario,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> list[MergerNeckResult]:
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    return [
        run_hydrocarbon_merger_neck_case(
            scenario,
            seed=seed,
            horizon_steps=horizon_steps,
            Nx=Nx,
            Ny=Ny,
        )
        for seed in seed_list
    ]


def diagnostic_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "profile": "HYDROCARBON_CANDIDATE",
        "information_mode": PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        "boundary_mode": EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
        "diagnostic_only": True,
        "model_mutation_allowed": False,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
        "events": [
            EVENT_SINGLE_PIXEL_NECK,
            EVENT_MULTI_PIXEL_NECK,
            EVENT_NO_MERGER_WITHIN_HORIZON,
            EVENT_MERGER_WITHOUT_RESOLVED_NECK,
        ],
    }
