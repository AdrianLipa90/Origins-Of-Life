from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .boundary_morphogenesis import EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE
from .compartments import periodic_component_topology
from .factory import create_exotic_candidate_simulator
from .information_morphogenesis import PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE


SCHEMA = "ORIGINS_HYDROCARBON_ISLAND_ANCESTRY_V0_1"

EVENT_MERGER = "MERGER_WITH_NONCONTRACTIBLE_BACKGROUND"
EVENT_EROSION = "THRESHOLD_EROSION"
EVENT_SPLIT = "SPLIT"
EVENT_SURVIVES = "CONTRACTIBLE_SURVIVES_SHELL_FAILURE"
EVENT_MIXED = "MIXED_TRANSITION"


@dataclass(frozen=True)
class IslandTransition:
    prior_label_id: int
    prior_area_pixels: int
    retained_pixels: int
    retained_fraction: float
    descendant_label_ids: tuple[int, ...]
    descendant_count: int
    noncontractible_descendant_count: int
    event: str

    def as_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HydrocarbonIslandAncestryResult:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    last_closed_step: int | None
    first_loss_step: int | None
    overall_transition: str | None
    prior_closed_component_count: int
    current_information_component_count: int
    current_contractible_component_count: int
    current_noncontractible_component_count: int
    transitions: tuple[IslandTransition, ...]
    physical_binding: str
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        payload = asdict(self)
        payload["transitions"] = [t.as_record() for t in self.transitions]
        return payload


def classify_contractible_ancestry_transition(
    previous_labels: np.ndarray,
    previous_accepted_mask: np.ndarray,
    current_information_mask: np.ndarray,
) -> tuple[str | None, tuple[IslandTransition, ...], dict[str, int]]:
    """Classify descendants of previously accepted contractible interiors.

    Matching uses exact periodic lattice-site overlap from t to t+1. This is an
    observational diagnostic only; it does not alter fields, parameters,
    thresholds, or source budgets.
    """
    prev_labels = np.asarray(previous_labels, dtype=int)
    prev_accepted = np.asarray(previous_accepted_mask, dtype=bool)
    current_mask = np.asarray(current_information_mask, dtype=bool)

    if (
        prev_labels.shape != prev_accepted.shape
        or prev_labels.shape != current_mask.shape
        or prev_labels.ndim != 2
    ):
        raise ValueError("ancestry fields must be same-shape 2D arrays")

    current_topology = periodic_component_topology(current_mask)

    # Build current periodic labels with the same deterministic scan order as
    # periodic_component_topology.
    current_labels = np.zeros_like(prev_labels, dtype=int)
    nx, ny = current_mask.shape
    label_id = 0
    for i in range(nx):
        for j in range(ny):
            if not current_mask[i, j] or current_labels[i, j] != 0:
                continue
            label_id += 1
            current_labels[i, j] = label_id
            stack = [(i, j)]
            while stack:
                x, y = stack.pop()
                for xx, yy in (
                    ((x - 1) % nx, y),
                    ((x + 1) % nx, y),
                    (x, (y - 1) % ny),
                    (x, (y + 1) % ny),
                ):
                    if current_mask[xx, yy] and current_labels[xx, yy] == 0:
                        current_labels[xx, yy] = label_id
                        stack.append((xx, yy))

    topology_by_label = {
        int(record["label_id"]): record
        for record in current_topology["components"]
    }

    prior_ids = sorted(
        int(x)
        for x in np.unique(prev_labels[prev_accepted])
        if int(x) > 0
    )

    transitions: list[IslandTransition] = []
    for prior_id in prior_ids:
        prior_mask = (prev_labels == prior_id) & prev_accepted
        prior_area = int(np.count_nonzero(prior_mask))
        retained_mask = prior_mask & current_mask
        retained_pixels = int(np.count_nonzero(retained_mask))
        descendants = tuple(
            sorted(
                int(x)
                for x in np.unique(current_labels[retained_mask])
                if int(x) > 0
            )
        )
        noncontractible = sum(
            1
            for descendant in descendants
            if bool(topology_by_label[descendant]["noncontractible"])
        )

        if not descendants:
            event = EVENT_EROSION
        elif noncontractible > 0:
            event = EVENT_MERGER
        elif len(descendants) > 1:
            event = EVENT_SPLIT
        else:
            event = EVENT_SURVIVES

        transitions.append(
            IslandTransition(
                prior_label_id=prior_id,
                prior_area_pixels=prior_area,
                retained_pixels=retained_pixels,
                retained_fraction=(
                    float(retained_pixels / prior_area) if prior_area > 0 else 0.0
                ),
                descendant_label_ids=descendants,
                descendant_count=len(descendants),
                noncontractible_descendant_count=int(noncontractible),
                event=event,
            )
        )

    events = {transition.event for transition in transitions}
    if not events:
        overall: str | None = None
    elif len(events) == 1:
        overall = next(iter(events))
    elif EVENT_MERGER in events:
        overall = EVENT_MERGER
    elif EVENT_SPLIT in events:
        overall = EVENT_SPLIT
    elif EVENT_SURVIVES in events:
        overall = EVENT_SURVIVES
    else:
        overall = EVENT_MIXED

    counts = {
        "component_count": int(current_topology["component_count"]),
        "contractible_component_count": int(
            current_topology["component_count"]
            - current_topology["noncontractible_component_count"]
        ),
        "noncontractible_component_count": int(
            current_topology["noncontractible_component_count"]
        ),
    }
    return overall, tuple(transitions), counts


def run_hydrocarbon_island_ancestry_case(
    scenario,
    *,
    seed: int,
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> HydrocarbonIslandAncestryResult:
    if getattr(scenario, "biochemistry_profile", None) != "HYDROCARBON_CANDIDATE":
        raise ValueError("island ancestry requires HYDROCARBON_CANDIDATE")
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

    was_closed = False
    last_closed_step: int | None = None
    first_loss_step: int | None = None
    previous_labels: np.ndarray | None = None
    previous_accepted: np.ndarray | None = None
    overall: str | None = None
    transitions: tuple[IslandTransition, ...] = ()
    current_counts = {
        "component_count": 0,
        "contractible_component_count": 0,
        "noncontractible_component_count": 0,
    }

    for step in range(1, int(horizon_steps) + 1):
        sim.step()
        observation = sim.candidate_compartments()
        closed = int(observation["count"]) > 0

        if closed:
            last_closed_step = step
            previous_labels = np.asarray(observation["labels"], dtype=int).copy()
            previous_accepted = np.asarray(
                observation["accepted_mask"],
                dtype=bool,
            ).copy()
        elif was_closed and first_loss_step is None:
            first_loss_step = step
            if previous_labels is None or previous_accepted is None:
                raise RuntimeError("missing previous closed-state ancestry snapshot")
            current_mask = np.asarray(sim.I) >= float(
                sim.parameters.information_threshold
            )
            overall, transitions, current_counts = (
                classify_contractible_ancestry_transition(
                    previous_labels,
                    previous_accepted,
                    current_mask,
                )
            )
            break

        was_closed = closed

    claim = sim.claim_status()
    return HydrocarbonIslandAncestryResult(
        schema=SCHEMA,
        profile=str(claim["profile"]),
        runtime=str(claim["runtime"]),
        seed=int(seed),
        horizon_steps=int(horizon_steps),
        last_closed_step=last_closed_step,
        first_loss_step=first_loss_step,
        overall_transition=overall,
        prior_closed_component_count=len(transitions),
        current_information_component_count=int(
            current_counts["component_count"]
        ),
        current_contractible_component_count=int(
            current_counts["contractible_component_count"]
        ),
        current_noncontractible_component_count=int(
            current_counts["noncontractible_component_count"]
        ),
        transitions=transitions,
        physical_binding=str(claim["physical_binding"]),
        parameter_tuning_allowed=False,
        threshold_tuning_allowed=False,
        ranking_allowed=False,
    )


def ancestry_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "profile": "HYDROCARBON_CANDIDATE",
        "information_mode": PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        "boundary_mode": EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
        "diagnostic_only": True,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
        "event_taxonomy": [
            EVENT_MERGER,
            EVENT_EROSION,
            EVENT_SPLIT,
            EVENT_SURVIVES,
            EVENT_MIXED,
        ],
    }
