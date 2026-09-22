from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .boundary_morphogenesis import (
    COLOCATED_BASELINE,
    EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
)
from .factory import create_exotic_candidate_simulator


SCHEMA = "ORIGINS_EXOTIC_LOCALIZATION_LIFECYCLE_V0_1"
MODES = (
    COLOCATED_BASELINE,
    EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
)


@dataclass(frozen=True)
class LocalizationLifecycleResult:
    schema: str
    profile: str
    runtime: str
    mode: str
    seed: int
    horizon_steps: int
    first_information_threshold_step: int | None
    first_boundary_threshold_step: int | None
    first_closed_boundary_step: int | None
    last_closed_boundary_step: int | None
    closed_boundary_steps: int
    longest_closed_boundary_run: int
    first_global_information_saturation_step: int | None
    first_global_boundary_saturation_step: int | None
    max_information_threshold_fraction: float
    max_boundary_threshold_fraction: float
    final_information_threshold_fraction: float
    final_boundary_threshold_fraction: float
    final_information_min: float
    final_information_mean: float
    final_information_max: float
    final_information_cv: float
    final_boundary_min: float
    final_boundary_mean: float
    final_boundary_max: float
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


def _cv(field: np.ndarray) -> float:
    mean = float(np.mean(field))
    return float(np.std(field) / mean) if mean > 1e-15 else 0.0


def _status(
    *,
    first_closed: int | None,
    final_count: int,
    info_saturation: int | None,
) -> str:
    if first_closed is None:
        return "NO_CLOSED_BOUNDARY_WITHIN_HORIZON"
    if final_count > 0:
        return "CLOSED_BOUNDARY_AT_HORIZON"
    if info_saturation is not None:
        return "TRANSIENT_CLOSURE_THEN_INFORMATION_SATURATION"
    return "TRANSIENT_CLOSURE_LOST_WITHOUT_INFORMATION_SATURATION"


def run_localization_lifecycle_case(
    scenario,
    *,
    mode: str,
    seed: int,
    horizon_steps: int = 3000,
    Nx: int = 24,
    Ny: int = 24,
) -> LocalizationLifecycleResult:
    if mode not in MODES:
        raise ValueError(f"unknown lifecycle mode: {mode!r}")
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be positive")

    sim = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=int(seed),
        boundary_morphogenesis=mode,
    )
    sim.initialize()

    p = sim.parameters
    info_threshold = float(p.information_threshold)
    boundary_threshold = float(p.boundary_threshold)

    first_info: int | None = None
    first_boundary: int | None = None
    first_closed: int | None = None
    last_closed: int | None = None
    first_info_saturation: int | None = None
    first_boundary_saturation: int | None = None

    closed_steps = 0
    current_run = 0
    longest_run = 0
    max_info_fraction = 0.0
    max_boundary_fraction = 0.0

    for step in range(1, int(horizon_steps) + 1):
        sim.step()

        info_mask = np.asarray(sim.I) >= info_threshold
        boundary_mask = np.asarray(sim.B) >= boundary_threshold
        info_fraction = float(np.mean(info_mask))
        boundary_fraction = float(np.mean(boundary_mask))
        max_info_fraction = max(max_info_fraction, info_fraction)
        max_boundary_fraction = max(max_boundary_fraction, boundary_fraction)

        if first_info is None and bool(np.any(info_mask)):
            first_info = step
        if first_boundary is None and bool(np.any(boundary_mask)):
            first_boundary = step
        if first_info_saturation is None and bool(np.all(info_mask)):
            first_info_saturation = step
        if first_boundary_saturation is None and bool(np.all(boundary_mask)):
            first_boundary_saturation = step

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

    final_observation = sim.candidate_compartments()
    info = np.asarray(sim.I, dtype=float)
    boundary = np.asarray(sim.B, dtype=float)
    final_info_fraction = float(np.mean(info >= info_threshold))
    final_boundary_fraction = float(np.mean(boundary >= boundary_threshold))
    claim = sim.claim_status()

    return LocalizationLifecycleResult(
        schema=SCHEMA,
        profile=str(claim["profile"]),
        runtime=str(claim["runtime"]),
        mode=str(mode),
        seed=int(seed),
        horizon_steps=int(horizon_steps),
        first_information_threshold_step=first_info,
        first_boundary_threshold_step=first_boundary,
        first_closed_boundary_step=first_closed,
        last_closed_boundary_step=last_closed,
        closed_boundary_steps=int(closed_steps),
        longest_closed_boundary_run=int(longest_run),
        first_global_information_saturation_step=first_info_saturation,
        first_global_boundary_saturation_step=first_boundary_saturation,
        max_information_threshold_fraction=float(max_info_fraction),
        max_boundary_threshold_fraction=float(max_boundary_fraction),
        final_information_threshold_fraction=final_info_fraction,
        final_boundary_threshold_fraction=final_boundary_fraction,
        final_information_min=float(np.min(info)),
        final_information_mean=float(np.mean(info)),
        final_information_max=float(np.max(info)),
        final_information_cv=_cv(info),
        final_boundary_min=float(np.min(boundary)),
        final_boundary_mean=float(np.mean(boundary)),
        final_boundary_max=float(np.max(boundary)),
        final_boundary_cv=_cv(boundary),
        final_compartment_count=int(final_observation["count"]),
        final_bounded_system_status=str(final_observation["bounded_system_status"]),
        lifecycle_status=_status(
            first_closed=first_closed,
            final_count=int(final_observation["count"]),
            info_saturation=first_info_saturation,
        ),
        physical_binding=str(claim["physical_binding"]),
        parameter_tuning_allowed=False,
        threshold_tuning_allowed=False,
        ranking_allowed=False,
    )


def run_matched_localization_lifecycle(
    scenarios: Iterable,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizon_steps: int = 3000,
    Nx: int = 24,
    Ny: int = 24,
    modes: Iterable[str] = MODES,
) -> list[LocalizationLifecycleResult]:
    scenario_list = list(scenarios)
    seed_list = tuple(int(seed) for seed in seeds)
    mode_list = tuple(str(mode) for mode in modes)
    if not scenario_list:
        raise ValueError("at least one scenario is required")
    if not seed_list:
        raise ValueError("at least one seed is required")
    if not mode_list:
        raise ValueError("at least one mode is required")
    if any(mode not in MODES for mode in mode_list):
        raise ValueError("unknown lifecycle mode")

    return [
        run_localization_lifecycle_case(
            scenario,
            mode=mode,
            seed=seed,
            horizon_steps=horizon_steps,
            Nx=Nx,
            Ny=Ny,
        )
        for scenario in scenario_list
        for mode in mode_list
        for seed in seed_list
    ]


def lifecycle_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "modes": list(MODES),
        "matched_seed_required": True,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
        "purpose": "separate reachability, transient closure, persistence, and global saturation",
    }
