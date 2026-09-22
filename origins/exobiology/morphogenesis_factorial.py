from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .boundary_morphogenesis import (
    COLOCATED_BASELINE,
    EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
)
from .factory import create_exotic_candidate_simulator
from .information_morphogenesis import (
    DISTRIBUTED_INFORMATION_BASELINE,
    PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
)


SCHEMA = "ORIGINS_EXOTIC_MORPHOGENESIS_FACTORIAL_V0_3"

INFORMATION_MODES = (
    DISTRIBUTED_INFORMATION_BASELINE,
    PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
)
BOUNDARY_MODES = (
    COLOCATED_BASELINE,
    EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
)


@dataclass(frozen=True)
class MorphogenesisFactorialResult:
    schema: str
    profile: str
    runtime: str
    information_mode: str
    boundary_mode: str
    seed: int
    horizon_steps: int
    information_source_budget_relative_error: float
    boundary_source_budget_relative_error: float
    material_residual_abs: float
    first_information_threshold_step: int | None
    first_boundary_threshold_step: int | None
    first_closed_boundary_step: int | None
    last_closed_boundary_step: int | None
    closed_boundary_steps: int
    longest_closed_boundary_run: int
    first_global_information_saturation_step: int | None
    final_information_threshold_fraction: float
    final_boundary_threshold_fraction: float
    final_information_mass: float
    final_boundary_mass: float
    final_information_cv: float
    final_boundary_cv: float
    final_compartment_count: int
    final_bounded_system_status: str
    lifecycle_status: str
    physical_binding: str
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _relative_budget_error(actual: np.ndarray, reference: np.ndarray) -> float:
    a = float(np.sum(actual))
    r = float(np.sum(reference))
    if r <= 1e-15:
        return 0.0 if a <= 1e-15 else float("inf")
    return float(abs(a - r) / r)


def _cv(field: np.ndarray) -> float:
    mean = float(np.mean(field))
    return float(np.std(field) / mean) if mean > 1e-15 else 0.0


def _lifecycle_status(
    *,
    first_closed: int | None,
    final_count: int,
    global_info_step: int | None,
) -> str:
    if first_closed is None:
        return "NO_CLOSED_BOUNDARY_WITHIN_HORIZON"
    if final_count > 0:
        return "CLOSED_BOUNDARY_AT_HORIZON"
    if global_info_step is not None:
        return "TRANSIENT_CLOSURE_THEN_INFORMATION_SATURATION"
    return "TRANSIENT_CLOSURE_LOST_WITHOUT_INFORMATION_SATURATION"


def run_morphogenesis_factorial_case(
    scenario,
    *,
    information_mode: str,
    boundary_mode: str,
    seed: int,
    horizon_steps: int,
    Nx: int = 24,
    Ny: int = 24,
) -> MorphogenesisFactorialResult:
    if information_mode not in INFORMATION_MODES:
        raise ValueError(f"unknown information mode: {information_mode!r}")
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(f"unknown boundary mode: {boundary_mode!r}")
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be positive")

    sim = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=int(seed),
        information_morphogenesis=information_mode,
        boundary_morphogenesis=boundary_mode,
    )
    sim.initialize()
    initial_material = sim.material_total()

    p = sim.parameters
    info_threshold = float(p.information_threshold)
    boundary_threshold = float(p.boundary_threshold)

    first_info: int | None = None
    first_boundary: int | None = None
    first_closed: int | None = None
    last_closed: int | None = None
    first_global_info: int | None = None

    closed_steps = 0
    current_run = 0
    longest_run = 0

    for step in range(1, int(horizon_steps) + 1):
        sim.step()
        info_mask = np.asarray(sim.I) >= info_threshold
        boundary_mask = np.asarray(sim.B) >= boundary_threshold

        if first_info is None and bool(np.any(info_mask)):
            first_info = step
        if first_boundary is None and bool(np.any(boundary_mask)):
            first_boundary = step
        if first_global_info is None and bool(np.all(info_mask)):
            first_global_info = step

        observation = sim.candidate_compartments()
        if int(observation["count"]) > 0:
            if first_closed is None:
                first_closed = step
            last_closed = step
            closed_steps += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    observation = sim.candidate_compartments()
    sources = sim.candidate_assembly_sources()
    info = np.asarray(sim.I, dtype=float)
    boundary = np.asarray(sim.B, dtype=float)
    claim = sim.claim_status()

    return MorphogenesisFactorialResult(
        schema=SCHEMA,
        profile=str(claim["profile"]),
        runtime=str(claim["runtime"]),
        information_mode=str(information_mode),
        boundary_mode=str(boundary_mode),
        seed=int(seed),
        horizon_steps=int(horizon_steps),
        information_source_budget_relative_error=_relative_budget_error(
            sources["information"],
            sources["information_baseline"],
        ),
        boundary_source_budget_relative_error=_relative_budget_error(
            sources["boundary"],
            sources["boundary_baseline"],
        ),
        material_residual_abs=float(abs(sim.material_total() - initial_material)),
        first_information_threshold_step=first_info,
        first_boundary_threshold_step=first_boundary,
        first_closed_boundary_step=first_closed,
        last_closed_boundary_step=last_closed,
        closed_boundary_steps=int(closed_steps),
        longest_closed_boundary_run=int(longest_run),
        first_global_information_saturation_step=first_global_info,
        final_information_threshold_fraction=float(np.mean(info >= info_threshold)),
        final_boundary_threshold_fraction=float(np.mean(boundary >= boundary_threshold)),
        final_information_mass=float(np.sum(info)),
        final_boundary_mass=float(np.sum(boundary)),
        final_information_cv=_cv(info),
        final_boundary_cv=_cv(boundary),
        final_compartment_count=int(observation["count"]),
        final_bounded_system_status=str(observation["bounded_system_status"]),
        lifecycle_status=_lifecycle_status(
            first_closed=first_closed,
            final_count=int(observation["count"]),
            global_info_step=first_global_info,
        ),
        physical_binding=str(claim["physical_binding"]),
        parameter_tuning_allowed=False,
        threshold_tuning_allowed=False,
        ranking_allowed=False,
    )


def run_profile_factorial(
    scenario,
    *,
    seeds: Iterable[int],
    horizon_steps: int,
    Nx: int = 24,
    Ny: int = 24,
) -> list[MorphogenesisFactorialResult]:
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    return [
        run_morphogenesis_factorial_case(
            scenario,
            information_mode=information_mode,
            boundary_mode=boundary_mode,
            seed=seed,
            horizon_steps=horizon_steps,
            Nx=Nx,
            Ny=Ny,
        )
        for information_mode in INFORMATION_MODES
        for boundary_mode in BOUNDARY_MODES
        for seed in seed_list
    ]


def factorial_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "information_modes": list(INFORMATION_MODES),
        "boundary_modes": list(BOUNDARY_MODES),
        "design": "2x2_MATCHED_SEED_FACTORIAL",
        "same_parameters_required": True,
        "same_thresholds_required": True,
        "source_budgets_preserved": True,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
    }
