from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .boundary_morphogenesis import (
    COLOCATED_BASELINE,
    EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
)
from .factory import create_exotic_candidate_simulator
from .source_geometry import diagnose_source_geometry


SCHEMA = "ORIGINS_EXOTIC_BOUNDARY_REDISTRIBUTION_TRIAL_V0_2"


@dataclass(frozen=True)
class BoundaryRedistributionTrialPair:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    baseline_mode: str
    intervention_mode: str
    baseline_boundary_source_total: float
    intervention_boundary_source_total: float
    intervention_same_state_baseline_source_total: float
    boundary_source_budget_relative_error: float
    baseline_source_cosine: float
    intervention_source_cosine: float
    baseline_information_mass: float
    intervention_information_mass: float
    baseline_boundary_mass: float
    intervention_boundary_mass: float
    baseline_compartment_count: int
    intervention_compartment_count: int
    baseline_max_shell_coverage: float
    intervention_max_shell_coverage: float
    baseline_status: str
    intervention_status: str
    baseline_first_closed_step: int | None
    intervention_first_closed_step: int | None
    baseline_closed_steps: int
    intervention_closed_steps: int
    baseline_longest_closed_run: int
    intervention_longest_closed_run: int
    baseline_material_residual_abs: float
    intervention_material_residual_abs: float
    physical_binding: str
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _new_pair(scenario, *, seed: int, Nx: int, Ny: int):
    baseline = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=seed,
        boundary_morphogenesis=COLOCATED_BASELINE,
    )
    intervention = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=seed,
        boundary_morphogenesis=EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
    )
    baseline.initialize()
    intervention.initialize()
    return baseline, intervention


def run_boundary_redistribution_trial_case(
    scenario,
    *,
    seed: int,
    horizons: Iterable[int] = (300, 1000, 3000),
    Nx: int = 24,
    Ny: int = 24,
) -> list[BoundaryRedistributionTrialPair]:
    horizon_list = tuple(sorted(set(int(h) for h in horizons)))
    if not horizon_list or horizon_list[0] <= 0:
        raise ValueError("horizons must contain positive integers")

    baseline, intervention = _new_pair(
        scenario,
        seed=int(seed),
        Nx=Nx,
        Ny=Ny,
    )
    m0_base = baseline.material_total()
    m0_int = intervention.material_total()

    first_base: int | None = None
    first_int: int | None = None
    closed_base = 0
    closed_int = 0
    run_base = 0
    run_int = 0
    longest_base = 0
    longest_int = 0
    previous = 0
    out: list[BoundaryRedistributionTrialPair] = []

    for horizon in horizon_list:
        for step in range(previous + 1, horizon + 1):
            baseline.step()
            intervention.step()

            ob = baseline.candidate_compartments()
            oi = intervention.candidate_compartments()

            if int(ob["count"]) > 0:
                if first_base is None:
                    first_base = step
                closed_base += 1
                run_base += 1
                longest_base = max(longest_base, run_base)
            else:
                run_base = 0

            if int(oi["count"]) > 0:
                if first_int is None:
                    first_int = step
                closed_int += 1
                run_int += 1
                longest_int = max(longest_int, run_int)
            else:
                run_int = 0

        previous = horizon
        ob = baseline.candidate_compartments()
        oi = intervention.candidate_compartments()
        baseline_sources = baseline.candidate_assembly_sources()
        intervention_sources = intervention.candidate_assembly_sources()
        total_b = float(np.sum(baseline_sources["boundary"]))
        total_i = float(np.sum(intervention_sources["boundary"]))
        intervention_reference = float(
            np.sum(intervention_sources["boundary_baseline"])
        )
        budget_error = (
            abs(total_i - intervention_reference) / intervention_reference
            if intervention_reference > 1e-15
            else (0.0 if total_i <= 1e-15 else float("inf"))
        )
        gb = diagnose_source_geometry(baseline)
        gi = diagnose_source_geometry(intervention)
        claim = baseline.claim_status()

        out.append(
            BoundaryRedistributionTrialPair(
                schema=SCHEMA,
                profile=str(claim["profile"]),
                runtime=str(claim["runtime"]),
                seed=int(seed),
                horizon_steps=int(horizon),
                baseline_mode=COLOCATED_BASELINE,
                intervention_mode=EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
                baseline_boundary_source_total=total_b,
                intervention_boundary_source_total=total_i,
                intervention_same_state_baseline_source_total=intervention_reference,
                boundary_source_budget_relative_error=float(budget_error),
                baseline_source_cosine=float(gb.cosine_overlap),
                intervention_source_cosine=float(gi.cosine_overlap),
                baseline_information_mass=float(np.sum(baseline.I)),
                intervention_information_mass=float(np.sum(intervention.I)),
                baseline_boundary_mass=float(np.sum(baseline.B)),
                intervention_boundary_mass=float(np.sum(intervention.B)),
                baseline_compartment_count=int(ob["count"]),
                intervention_compartment_count=int(oi["count"]),
                baseline_max_shell_coverage=float(ob["max_shell_coverage"]),
                intervention_max_shell_coverage=float(oi["max_shell_coverage"]),
                baseline_status=str(ob["bounded_system_status"]),
                intervention_status=str(oi["bounded_system_status"]),
                baseline_first_closed_step=first_base,
                intervention_first_closed_step=first_int,
                baseline_closed_steps=int(closed_base),
                intervention_closed_steps=int(closed_int),
                baseline_longest_closed_run=int(longest_base),
                intervention_longest_closed_run=int(longest_int),
                baseline_material_residual_abs=float(abs(baseline.material_total() - m0_base)),
                intervention_material_residual_abs=float(abs(intervention.material_total() - m0_int)),
                physical_binding=str(claim["physical_binding"]),
                parameter_tuning_allowed=False,
                threshold_tuning_allowed=False,
                ranking_allowed=False,
            )
        )

    return out


def run_matched_boundary_redistribution_trial(
    scenarios: Iterable,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizons: Iterable[int] = (300, 1000, 3000),
    Nx: int = 24,
    Ny: int = 24,
) -> list[BoundaryRedistributionTrialPair]:
    scenario_list = list(scenarios)
    seed_list = tuple(int(seed) for seed in seeds)
    if not scenario_list:
        raise ValueError("at least one scenario is required")
    if not seed_list:
        raise ValueError("at least one seed is required")
    return [
        row
        for scenario in scenario_list
        for seed in seed_list
        for row in run_boundary_redistribution_trial_case(
            scenario,
            seed=seed,
            horizons=horizons,
            Nx=Nx,
            Ny=Ny,
        )
    ]


def trial_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "baseline_mode": COLOCATED_BASELINE,
        "intervention_mode": EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
        "matched_seed_required": True,
        "same_parameters_required": True,
        "same_thresholds_required": True,
        "source_budget_preserved_when_interface_exists": True,
        "only_varied_mechanism": "boundary_source_spatial_distribution",
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
    }
