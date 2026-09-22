from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np

from .factory import create_exotic_candidate_simulator


DIAGNOSTIC_SCHEMA = "ORIGINS_EXOTIC_THRESHOLD_REACHABILITY_V0_1"
STATUS_REACHED = "REACHED"
STATUS_NOT_REACHED = "NOT_REACHED_WITHIN_HORIZON"


@dataclass(frozen=True)
class ThresholdReachabilityResult:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    information_threshold: float
    boundary_threshold: float
    max_information: float
    max_boundary: float
    information_threshold_ratio: float
    boundary_threshold_ratio: float
    candidate_compartment_count: int
    raw_component_count: int
    candidate_compartment_area_pixels: int
    compartment_occupancy_fraction: float
    interface_edge_count: int
    global_saturation: bool
    bounded_system_status: str
    first_compartment_step: int | None
    reachability_status: str
    baseline_information_mass: float
    no_selection_information_mass: float
    selection_information_effect_fraction: float
    physical_binding: str
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _new_simulator(scenario, *, seed: int, Nx: int, Ny: int, no_selection: bool):
    simulator = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=seed,
    )
    if no_selection:
        simulator.parameters = replace(simulator.parameters, selection_strength=0.0)
        simulator.parameters.validate()
    simulator.initialize()
    return simulator


def run_threshold_reachability_case(
    scenario,
    *,
    seed: int,
    horizons: Iterable[int] = (300, 1000, 3000),
    Nx: int = 24,
    Ny: int = 24,
) -> list[ThresholdReachabilityResult]:
    """Probe declared thresholds without changing them.

    Baseline and NO_SELECTION use matched initial seed/state.  A missing
    threshold crossing means only NOT_REACHED_WITHIN_HORIZON.
    """
    horizon_list = tuple(sorted(set(int(h) for h in horizons)))
    if not horizon_list or horizon_list[0] <= 0:
        raise ValueError("horizons must contain positive integers")

    baseline = _new_simulator(scenario, seed=seed, Nx=Nx, Ny=Ny, no_selection=False)
    no_selection = _new_simulator(scenario, seed=seed, Nx=Nx, Ny=Ny, no_selection=True)

    p = baseline.parameters
    information_threshold = float(p.information_threshold)
    boundary_threshold = float(p.boundary_threshold)
    if information_threshold <= 0.0 or boundary_threshold <= 0.0:
        raise ValueError("declared thresholds must be positive")

    first_compartment_step: int | None = None
    results: list[ThresholdReachabilityResult] = []
    previous_horizon = 0

    for horizon in horizon_list:
        for step in range(previous_horizon + 1, horizon + 1):
            baseline.step()
            no_selection.step()
            if first_compartment_step is None:
                observation = baseline.candidate_compartments()
                if int(observation["count"]) > 0:
                    first_compartment_step = step

        observation = baseline.candidate_compartments()
        max_information = float(np.max(baseline.I))
        max_boundary = float(np.max(baseline.B))
        baseline_information_mass = float(np.sum(baseline.I))
        no_selection_information_mass = float(np.sum(no_selection.I))
        selection_effect = (
            (baseline_information_mass - no_selection_information_mass)
            / baseline_information_mass
            if baseline_information_mass > 0.0
            else 0.0
        )
        claim = baseline.claim_status()

        reached = first_compartment_step is not None and first_compartment_step <= horizon
        results.append(
            ThresholdReachabilityResult(
                schema=DIAGNOSTIC_SCHEMA,
                profile=str(claim["profile"]),
                runtime=str(claim["runtime"]),
                seed=int(seed),
                horizon_steps=int(horizon),
                information_threshold=information_threshold,
                boundary_threshold=boundary_threshold,
                max_information=max_information,
                max_boundary=max_boundary,
                information_threshold_ratio=max_information / information_threshold,
                boundary_threshold_ratio=max_boundary / boundary_threshold,
                candidate_compartment_count=int(observation["count"]),
                raw_component_count=int(observation["raw_component_count"]),
                candidate_compartment_area_pixels=int(observation["area_pixels"]),
                compartment_occupancy_fraction=float(observation["occupancy_fraction"]),
                interface_edge_count=int(observation["interface_edge_count"]),
                global_saturation=bool(observation["global_saturation"]),
                bounded_system_status=str(observation["bounded_system_status"]),
                first_compartment_step=first_compartment_step,
                reachability_status=(
                    STATUS_REACHED if reached else STATUS_NOT_REACHED
                ),
                baseline_information_mass=baseline_information_mass,
                no_selection_information_mass=no_selection_information_mass,
                selection_information_effect_fraction=float(selection_effect),
                physical_binding=str(claim["physical_binding"]),
                ranking_allowed=False,
            )
        )
        previous_horizon = horizon

    return results


def run_matched_threshold_reachability(
    scenarios: Iterable,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizons: Iterable[int] = (300, 1000, 3000),
    Nx: int = 24,
    Ny: int = 24,
) -> list[ThresholdReachabilityResult]:
    scenario_list = list(scenarios)
    seed_list = tuple(int(seed) for seed in seeds)
    if not scenario_list:
        raise ValueError("at least one scenario is required")
    if not seed_list:
        raise ValueError("at least one seed is required")

    out: list[ThresholdReachabilityResult] = []
    for scenario in scenario_list:
        for seed in seed_list:
            out.extend(
                run_threshold_reachability_case(
                    scenario,
                    seed=seed,
                    horizons=horizons,
                    Nx=Nx,
                    Ny=Ny,
                )
            )
    return out


def diagnostic_manifest() -> dict[str, object]:
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "ranking_allowed": False,
        "threshold_tuning_allowed": False,
        "missing_crossing_semantics": STATUS_NOT_REACHED,
        "matched_seed_selection_control": True,
        "physical_binding": "OPEN",
    }
