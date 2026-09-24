from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .boundary_morphogenesis import EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE
from .compartments import _periodic_components, periodic_component_topology
from .factory import create_exotic_candidate_simulator
from .information_morphogenesis import PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE


SCHEMA = "ORIGINS_CONTRACTIBLE_ISLAND_ANCESTRY_TRACE_V0_1"

LINEAGE_CONTRACTIBLE_SURVIVAL = "CONTRACTIBLE_INTERIOR_SURVIVES"
LINEAGE_MERGER_WITH_NONCONTRACTIBLE = "MERGER_WITH_NONCONTRACTIBLE_BACKGROUND"
LINEAGE_BECAME_NONCONTRACTIBLE = "BECAME_NONCONTRACTIBLE"
LINEAGE_SPLIT = "SPLIT"
LINEAGE_THRESHOLD_EROSION = "THRESHOLD_EROSION_OR_DISPLACEMENT"

EVENT_PERSISTS = "PERSISTS_WITHIN_HORIZON"
EVENT_INTERIOR_SURVIVES_CLOSURE_LOST = "INTERIOR_SURVIVES_CLOSURE_LOST"
EVENT_MERGER_WITH_NONCONTRACTIBLE = "MERGER_WITH_NONCONTRACTIBLE_BACKGROUND"
EVENT_BECAME_NONCONTRACTIBLE = "BECAME_NONCONTRACTIBLE"
EVENT_SPLIT = "SPLIT"
EVENT_THRESHOLD_EROSION = "THRESHOLD_EROSION_OR_DISPLACEMENT"
EVENT_MIXED = "MIXED_TRANSITION"


@dataclass(frozen=True)
class IslandLineageTransition:
    previous_label: int
    previous_area_pixels: int
    classification: str
    descendant_labels: tuple[int, ...]
    overlap_pixels: tuple[int, ...]
    descendant_noncontractible: tuple[bool, ...]
    descendant_overlap_with_previous_noncontractible_pixels: tuple[int, ...]

    def as_record(self) -> dict[str, object]:
        data = asdict(self)
        data["descendant_labels"] = list(self.descendant_labels)
        data["overlap_pixels"] = list(self.overlap_pixels)
        data["descendant_noncontractible"] = list(self.descendant_noncontractible)
        data["descendant_overlap_with_previous_noncontractible_pixels"] = list(
            self.descendant_overlap_with_previous_noncontractible_pixels
        )
        return data


@dataclass(frozen=True)
class ContractibleIslandAncestryResult:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    first_closed_step: int | None
    last_closed_step: int | None
    first_loss_after_closure_step: int | None
    ancestry_event: str
    previous_accepted_component_count: int
    current_contractible_component_count: int
    current_noncontractible_component_count: int
    current_best_contractible_shell_coverage: float
    current_best_contractible_missing_shell_pixels: int
    transitions: tuple[IslandLineageTransition, ...]
    physical_binding: str
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        data = asdict(self)
        data["transitions"] = [transition.as_record() for transition in self.transitions]
        return data


def _component_records(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    labels, components = _periodic_components(mask)
    topology = periodic_component_topology(mask)
    topo_by_label = {
        int(record["label_id"]): record
        for record in topology["components"]
    }

    records: list[dict[str, object]] = []
    for label_id, cells in enumerate(components, start=1):
        topo = topo_by_label[label_id]
        if int(topo["area_pixels"]) != len(cells):
            raise RuntimeError("component/topology label ordering drift")
        records.append(
            {
                "label_id": int(label_id),
                "cells": frozenset(cells),
                "area_pixels": int(len(cells)),
                "noncontractible": bool(topo["noncontractible"]),
            }
        )
    return labels, records


def accepted_component_labels(observation: dict[str, object]) -> tuple[int, ...]:
    labels = np.asarray(observation["labels"], dtype=int)
    accepted = np.asarray(observation["accepted_mask"], dtype=bool)
    if labels.shape != accepted.shape:
        raise ValueError("observation labels and accepted mask shape mismatch")
    values = sorted(int(value) for value in np.unique(labels[accepted]) if int(value) > 0)
    return tuple(values)


def classify_component_transition(
    previous_mask: np.ndarray,
    previous_accepted_labels: Iterable[int],
    current_mask: np.ndarray,
) -> tuple[IslandLineageTransition, ...]:
    """Classify the one-step ancestry of previously accepted contractible interiors."""
    prev_labels, prev_records = _component_records(previous_mask)
    _, curr_records = _component_records(current_mask)

    prev_by_label = {int(record["label_id"]): record for record in prev_records}
    accepted_labels = tuple(int(value) for value in previous_accepted_labels)
    if any(label not in prev_by_label for label in accepted_labels):
        raise ValueError("accepted label missing from previous component state")

    prev_noncontractible_cells = [
        record["cells"]
        for record in prev_records
        if bool(record["noncontractible"])
    ]

    transitions: list[IslandLineageTransition] = []
    for label in accepted_labels:
        previous = prev_by_label[label]
        previous_cells = previous["cells"]
        descendants: list[tuple[dict[str, object], int, int]] = []

        for current in curr_records:
            overlap = len(previous_cells & current["cells"])
            if overlap <= 0:
                continue
            background_overlap = max(
                (
                    len(current["cells"] & background)
                    for background in prev_noncontractible_cells
                ),
                default=0,
            )
            descendants.append((current, int(overlap), int(background_overlap)))

        if not descendants:
            classification = LINEAGE_THRESHOLD_EROSION
        elif len(descendants) > 1:
            classification = LINEAGE_SPLIT
        else:
            current, _, background_overlap = descendants[0]
            if bool(current["noncontractible"]):
                if background_overlap > 0:
                    classification = LINEAGE_MERGER_WITH_NONCONTRACTIBLE
                else:
                    classification = LINEAGE_BECAME_NONCONTRACTIBLE
            else:
                classification = LINEAGE_CONTRACTIBLE_SURVIVAL

        transitions.append(
            IslandLineageTransition(
                previous_label=int(label),
                previous_area_pixels=int(previous["area_pixels"]),
                classification=classification,
                descendant_labels=tuple(
                    int(current["label_id"]) for current, _, _ in descendants
                ),
                overlap_pixels=tuple(overlap for _, overlap, _ in descendants),
                descendant_noncontractible=tuple(
                    bool(current["noncontractible"]) for current, _, _ in descendants
                ),
                descendant_overlap_with_previous_noncontractible_pixels=tuple(
                    background_overlap for _, _, background_overlap in descendants
                ),
            )
        )

    return tuple(transitions)


def summarize_ancestry_event(
    transitions: Iterable[IslandLineageTransition],
) -> str:
    transitions = tuple(transitions)
    if not transitions:
        return EVENT_THRESHOLD_EROSION

    classes = {transition.classification for transition in transitions}
    if classes == {LINEAGE_CONTRACTIBLE_SURVIVAL}:
        return EVENT_INTERIOR_SURVIVES_CLOSURE_LOST
    if LINEAGE_MERGER_WITH_NONCONTRACTIBLE in classes:
        return EVENT_MERGER_WITH_NONCONTRACTIBLE
    if LINEAGE_BECAME_NONCONTRACTIBLE in classes:
        return EVENT_BECAME_NONCONTRACTIBLE
    if LINEAGE_SPLIT in classes:
        return EVENT_SPLIT
    if classes == {LINEAGE_THRESHOLD_EROSION}:
        return EVENT_THRESHOLD_EROSION
    return EVENT_MIXED


def run_contractible_island_ancestry_case(
    scenario,
    *,
    seed: int,
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> ContractibleIslandAncestryResult:
    if getattr(scenario, "biochemistry_profile", None) != "HYDROCARBON_CANDIDATE":
        raise ValueError("ancestry trace requires HYDROCARBON_CANDIDATE")
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

    first_closed: int | None = None
    last_closed: int | None = None
    first_loss: int | None = None

    previous_closed_mask: np.ndarray | None = None
    previous_accepted_labels: tuple[int, ...] = ()
    transitions: tuple[IslandLineageTransition, ...] = ()
    loss_observation: dict[str, object] | None = None

    was_closed = False

    for step in range(1, int(horizon_steps) + 1):
        sim.step()
        observation = sim.candidate_compartments()
        closed = int(observation["count"]) > 0

        if closed:
            if first_closed is None:
                first_closed = step
            last_closed = step
            information_mask = np.asarray(sim.I) >= float(
                sim.parameters.information_threshold
            )
            previous_closed_mask = information_mask.copy()
            previous_accepted_labels = accepted_component_labels(observation)
        elif was_closed and first_loss is None:
            first_loss = step
            loss_observation = dict(observation)
            current_mask = np.asarray(sim.I) >= float(
                sim.parameters.information_threshold
            )
            if previous_closed_mask is None or not previous_accepted_labels:
                raise RuntimeError("missing previous closed-state ancestry context")
            transitions = classify_component_transition(
                previous_closed_mask,
                previous_accepted_labels,
                current_mask,
            )
            break

        was_closed = closed

    claim = sim.claim_status()

    if first_loss is None:
        event = EVENT_PERSISTS
        observation = sim.candidate_compartments()
    else:
        event = summarize_ancestry_event(transitions)
        assert loss_observation is not None
        observation = loss_observation

    return ContractibleIslandAncestryResult(
        schema=SCHEMA,
        profile=str(claim["profile"]),
        runtime=str(claim["runtime"]),
        seed=int(seed),
        horizon_steps=int(horizon_steps),
        first_closed_step=first_closed,
        last_closed_step=last_closed,
        first_loss_after_closure_step=first_loss,
        ancestry_event=event,
        previous_accepted_component_count=int(len(previous_accepted_labels)),
        current_contractible_component_count=int(
            observation["contractible_information_component_count"]
        ),
        current_noncontractible_component_count=int(
            observation["noncontractible_information_component_count"]
        ),
        current_best_contractible_shell_coverage=float(
            observation["best_contractible_shell_coverage"]
        ),
        current_best_contractible_missing_shell_pixels=int(
            observation["best_contractible_missing_shell_pixels"]
        ),
        transitions=transitions,
        physical_binding=str(claim["physical_binding"]),
        parameter_tuning_allowed=False,
        threshold_tuning_allowed=False,
        ranking_allowed=False,
    )


def run_matched_contractible_island_ancestry(
    scenario,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> list[ContractibleIslandAncestryResult]:
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    return [
        run_contractible_island_ancestry_case(
            scenario,
            seed=seed,
            horizon_steps=horizon_steps,
            Nx=Nx,
            Ny=Ny,
        )
        for seed in seed_list
    ]


def ancestry_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "profile": "HYDROCARBON_CANDIDATE",
        "information_mode": PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        "boundary_mode": EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
        "model_mutation_allowed": False,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
        "lineage_classes": [
            LINEAGE_CONTRACTIBLE_SURVIVAL,
            LINEAGE_MERGER_WITH_NONCONTRACTIBLE,
            LINEAGE_BECAME_NONCONTRACTIBLE,
            LINEAGE_SPLIT,
            LINEAGE_THRESHOLD_EROSION,
        ],
    }
