from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .boundary_morphogenesis import (
    COLOCATED_BASELINE,
    EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
)
from .factory import create_exotic_candidate_simulator
from .source_geometry import diagnose_source_geometry


SCHEMA = "ORIGINS_EXOTIC_BOUNDARY_MORPHOGENESIS_TRIAL_V0_1"
MODES = (
    COLOCATED_BASELINE,
    EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
)


@dataclass(frozen=True)
class BoundaryMorphogenesisTrialPoint:
    schema: str
    profile: str
    runtime: str
    mode: str
    seed: int
    horizon_steps: int
    material_residual_abs: float
    information_source_cosine_to_boundary_source: float
    information_source_pearson_to_boundary_source: float
    information_mass: float
    boundary_mass: float
    candidate_compartment_count: int
    raw_information_component_count: int
    candidate_compartment_area_pixels: int
    candidate_compartment_shell_pixels: int
    max_shell_coverage: float
    global_saturation: bool
    bounded_system_status: str
    first_closed_boundary_step: int | None
    closed_boundary_steps: int
    longest_consecutive_closed_boundary_steps: int
    localized_at_horizon: bool
    physical_binding: str
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _new_simulator(scenario, *, mode: str, seed: int, Nx: int, Ny: int):
    return create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=seed,
        boundary_morphogenesis=mode,
    )


def run_boundary_morphogenesis_trial_case(
    scenario,
    *,
    mode: str,
    seed: int,
    horizons: Iterable[int] = (300, 1000, 3000),
    Nx: int = 24,
    Ny: int = 24,
) -> list[BoundaryMorphogenesisTrialPoint]:
    if mode not in MODES:
        raise ValueError(f"unknown morphogenesis mode: {mode!r}")
    horizon_list = tuple(sorted(set(int(h) for h in horizons)))
    if not horizon_list or horizon_list[0] <= 0:
        raise ValueError("horizons must contain positive integers")

    sim = _new_simulator(
        scenario,
        mode=mode,
        seed=int(seed),
        Nx=Nx,
        Ny=Ny,
    )
    sim.initialize()
    material_initial = sim.material_total()

    first_closed: int | None = None
    closed_steps = 0
    current_run = 0
    longest_run = 0
    previous = 0
    rows: list[BoundaryMorphogenesisTrialPoint] = []

    for horizon in horizon_list:
        for step in range(previous + 1, horizon + 1):
            sim.step()
            observation = sim.candidate_compartments()
            if int(observation["count"]) > 0:
                if first_closed is None:
                    first_closed = step
                closed_steps += 1
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0

        previous = horizon
        observation = sim.candidate_compartments()
        source = diagnose_source_geometry(sim)
        claim = sim.claim_status()
        material_final = sim.material_total()

        rows.append(
            BoundaryMorphogenesisTrialPoint(
                schema=SCHEMA,
                profile=str(claim["profile"]),
                runtime=str(claim["runtime"]),
                mode=str(mode),
                seed=int(seed),
                horizon_steps=int(horizon),
                material_residual_abs=float(abs(material_final - material_initial)),
                information_source_cosine_to_boundary_source=float(source.cosine_overlap),
                information_source_pearson_to_boundary_source=float(source.pearson_correlation),
                information_mass=float(np.sum(sim.I)),
                boundary_mass=float(np.sum(sim.B)),
                candidate_compartment_count=int(observation["count"]),
                raw_information_component_count=int(observation["raw_information_component_count"]),
                candidate_compartment_area_pixels=int(observation["area_pixels"]),
                candidate_compartment_shell_pixels=int(observation["shell_pixels"]),
                max_shell_coverage=float(observation["max_shell_coverage"]),
                global_saturation=bool(observation["global_saturation"]),
                bounded_system_status=str(observation["bounded_system_status"]),
                first_closed_boundary_step=first_closed,
                closed_boundary_steps=int(closed_steps),
                longest_consecutive_closed_boundary_steps=int(longest_run),
                localized_at_horizon=bool(int(observation["count"]) > 0),
                physical_binding=str(claim["physical_binding"]),
                parameter_tuning_allowed=False,
                threshold_tuning_allowed=False,
                ranking_allowed=False,
            )
        )
    return rows


def run_matched_boundary_morphogenesis_trial(
    scenarios: Iterable,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizons: Iterable[int] = (300, 1000, 3000),
    Nx: int = 24,
    Ny: int = 24,
) -> list[BoundaryMorphogenesisTrialPoint]:
    scenario_list = list(scenarios)
    seed_list = tuple(int(s) for s in seeds)
    if not scenario_list:
        raise ValueError("at least one scenario is required")
    if not seed_list:
        raise ValueError("at least one seed is required")

    return [
        row
        for scenario in scenario_list
        for mode in MODES
        for seed in seed_list
        for row in run_boundary_morphogenesis_trial_case(
            scenario,
            mode=mode,
            seed=seed,
            horizons=horizons,
            Nx=Nx,
            Ny=Ny,
        )
    ]


def trial_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "modes": list(MODES),
        "matched_seed_required": True,
        "parameters_frozen": True,
        "thresholds_frozen": True,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
        "only_varied_mechanism": "boundary_morphogenesis_mode",
    }
