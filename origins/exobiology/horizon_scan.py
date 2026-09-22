from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .factory import create_exotic_candidate_simulator


HORIZON_SCAN_SCHEMA = "ORIGINS_EXOTIC_HORIZON_SCAN_V0_1"
DEFAULT_HORIZONS: tuple[int, ...] = (300, 1000, 3000)


@dataclass(frozen=True)
class ExoticHorizonScanPoint:
    schema: str
    runtime: str
    profile: str
    seed: int
    horizon_steps: int
    material_residual_abs: float
    energy_throughput_integral: float
    information_mass: float
    boundary_mass: float
    information_max: float
    boundary_max: float
    joint_threshold_score_max: float
    candidate_compartment_count: int
    raw_information_component_count: int
    candidate_compartment_area_pixels: int
    candidate_compartment_shell_pixels: int
    max_shell_coverage: float
    global_saturation: bool
    bounded_system_status: str
    trait_mean: float
    trait_variance: float
    information_threshold: float
    boundary_threshold: float
    physical_binding: str
    parameter_status: str
    thresholds_frozen: bool = True
    parameters_frozen: bool = True
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _joint_threshold_score(simulator) -> float:
    info_threshold = float(simulator.parameters.information_threshold)
    boundary_threshold = float(simulator.parameters.boundary_threshold)
    if info_threshold <= 0.0 or boundary_threshold <= 0.0:
        raise ValueError("horizon scan requires positive fixed observation thresholds")
    score = np.minimum(
        simulator.I / info_threshold,
        simulator.B / boundary_threshold,
    )
    return float(np.max(score))


def run_exotic_horizon_scan_case(
    scenario,
    *,
    seed: int,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    Nx: int = 24,
    Ny: int = 24,
) -> list[ExoticHorizonScanPoint]:
    """Run one uninterrupted trajectory and sample fixed horizons.

    Parameters and observation thresholds remain frozen.  The scan therefore
    isolates time-horizon sensitivity instead of tuning the model after seeing
    the 300-step zero-compartment baseline.
    """
    horizon_list = tuple(int(h) for h in horizons)
    if not horizon_list or any(h <= 0 for h in horizon_list):
        raise ValueError("horizons must be positive")
    if tuple(sorted(set(horizon_list))) != horizon_list:
        raise ValueError("horizons must be strictly increasing and unique")

    simulator = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=int(seed),
    )
    simulator.initialize()
    material_initial = simulator.material_total()
    claim = simulator.claim_status()

    rows: list[ExoticHorizonScanPoint] = []
    previous = 0
    for horizon in horizon_list:
        simulator.run(horizon - previous)
        previous = horizon

        compartments = simulator.candidate_compartments()
        material_final = simulator.material_total()

        rows.append(
            ExoticHorizonScanPoint(
                schema=HORIZON_SCAN_SCHEMA,
                runtime=str(claim["runtime"]),
                profile=str(claim["profile"]),
                seed=int(seed),
                horizon_steps=horizon,
                material_residual_abs=float(abs(material_final - material_initial)),
                energy_throughput_integral=float(simulator.energy_throughput_integral),
                information_mass=float(np.sum(simulator.I)),
                boundary_mass=float(np.sum(simulator.B)),
                information_max=float(np.max(simulator.I)),
                boundary_max=float(np.max(simulator.B)),
                joint_threshold_score_max=_joint_threshold_score(simulator),
                candidate_compartment_count=int(compartments["count"]),
                raw_information_component_count=int(compartments["raw_information_component_count"]),
                candidate_compartment_area_pixels=int(compartments["area_pixels"]),
                candidate_compartment_shell_pixels=int(compartments["shell_pixels"]),
                max_shell_coverage=float(compartments["max_shell_coverage"]),
                global_saturation=bool(compartments["global_saturation"]),
                bounded_system_status=str(compartments["bounded_system_status"]),
                trait_mean=float(np.mean(simulator.Q)),
                trait_variance=float(np.var(simulator.Q)),
                information_threshold=float(simulator.parameters.information_threshold),
                boundary_threshold=float(simulator.parameters.boundary_threshold),
                physical_binding=str(claim["physical_binding"]),
                parameter_status=str(claim["parameter_status"]),
            )
        )
    return rows


def run_matched_exotic_horizon_scan(
    scenarios: Iterable,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    Nx: int = 24,
    Ny: int = 24,
) -> list[ExoticHorizonScanPoint]:
    scenario_list = list(scenarios)
    seed_list = tuple(int(seed) for seed in seeds)
    horizon_list = tuple(int(h) for h in horizons)
    if not scenario_list:
        raise ValueError("at least one scenario is required")
    if not seed_list:
        raise ValueError("at least one seed is required")

    out: list[ExoticHorizonScanPoint] = []
    for scenario in scenario_list:
        for seed in seed_list:
            out.extend(
                run_exotic_horizon_scan_case(
                    scenario,
                    seed=seed,
                    horizons=horizon_list,
                    Nx=Nx,
                    Ny=Ny,
                )
            )
    return out


def horizon_scan_manifest() -> dict[str, object]:
    return {
        "schema": HORIZON_SCAN_SCHEMA,
        "frozen_dimensions": ["parameters", "observation_thresholds", "grid", "seeds"],
        "varied_dimension": "horizon_steps_only",
        "default_horizons": list(DEFAULT_HORIZONS),
        "ranking_allowed": False,
        "primary_diagnostic": "joint_threshold_score_max",
        "interpretation": {
            "score_below_1": "no grid point jointly satisfies I and B thresholds",
            "score_at_least_1": "at least one grid point jointly reaches both scalar thresholds; this is not sufficient for a bounded compartment",
            "global_saturation": "uniform threshold crossing over the whole periodic domain is not a bounded compartment",
        },
    }
