from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .boundary_morphogenesis import COLOCATED_BASELINE
from .factory import create_exotic_candidate_simulator


PERSISTENCE_SCHEMA = "ORIGINS_EXOTIC_COMPARTMENT_PERSISTENCE_V0_1"

STATUS_NEVER_LOCALIZED = "NEVER_LOCALIZED"
STATUS_LOCALIZED_AT_HORIZON = "LOCALIZED_AT_HORIZON"
STATUS_LOCALIZED_THEN_LOST = "LOCALIZED_THEN_LOST"


@dataclass(frozen=True)
class CompartmentPersistenceResult:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    boundary_morphogenesis: str
    first_localized_step: int | None
    last_localized_step: int | None
    localized_step_count: int
    localized_fraction: float
    longest_consecutive_localized_steps: int
    localized_at_horizon: bool
    first_global_saturation_step: int | None
    global_saturation_step_count: int
    final_global_saturation: bool
    max_localized_component_count: int
    max_localized_area_pixels: int
    max_shell_pixels: int
    persistence_status: str
    physical_binding: str
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def run_compartment_persistence_case(
    scenario,
    *,
    seed: int,
    horizon_steps: int = 3000,
    Nx: int = 24,
    Ny: int = 24,
    boundary_morphogenesis: str = COLOCATED_BASELINE,
) -> CompartmentPersistenceResult:
    """Track localization persistence without changing thresholds or parameters."""
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be positive")

    simulator = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=int(seed),
        boundary_morphogenesis=boundary_morphogenesis,
    )
    simulator.initialize()

    first_localized: int | None = None
    last_localized: int | None = None
    first_saturation: int | None = None

    localized_steps = 0
    saturation_steps = 0
    longest_run = 0
    current_run = 0
    max_count = 0
    max_area = 0
    max_shell_pixels = 0
    final_observation: dict[str, object] | None = None

    for step in range(1, int(horizon_steps) + 1):
        simulator.step()
        observation = simulator.candidate_compartments()
        final_observation = observation
        status = str(observation["bounded_system_status"])

        if status == "CLOSED_BOUNDARY_CANDIDATE":
            localized_steps += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
            first_localized = step if first_localized is None else first_localized
            last_localized = step
            max_count = max(max_count, int(observation["count"]))
            max_area = max(max_area, int(observation["area_pixels"]))
            max_shell_pixels = max(
                max_shell_pixels,
                int(observation["shell_pixels"]),
            )
        else:
            current_run = 0

        if status == "GLOBAL_SATURATION_REJECTED":
            saturation_steps += 1
            if first_saturation is None:
                first_saturation = step

    assert final_observation is not None
    localized_at_horizon = (
        str(final_observation["bounded_system_status"]) == "CLOSED_BOUNDARY_CANDIDATE"
    )
    final_global_saturation = bool(final_observation["global_saturation"])

    if first_localized is None:
        persistence_status = STATUS_NEVER_LOCALIZED
    elif localized_at_horizon:
        persistence_status = STATUS_LOCALIZED_AT_HORIZON
    else:
        persistence_status = STATUS_LOCALIZED_THEN_LOST

    claim = simulator.claim_status()
    return CompartmentPersistenceResult(
        schema=PERSISTENCE_SCHEMA,
        profile=str(claim["profile"]),
        runtime=str(claim["runtime"]),
        seed=int(seed),
        horizon_steps=int(horizon_steps),
        boundary_morphogenesis=str(claim["boundary_morphogenesis"]),
        first_localized_step=first_localized,
        last_localized_step=last_localized,
        localized_step_count=int(localized_steps),
        localized_fraction=float(localized_steps / int(horizon_steps)),
        longest_consecutive_localized_steps=int(longest_run),
        localized_at_horizon=localized_at_horizon,
        first_global_saturation_step=first_saturation,
        global_saturation_step_count=int(saturation_steps),
        final_global_saturation=final_global_saturation,
        max_localized_component_count=int(max_count),
        max_localized_area_pixels=int(max_area),
        max_shell_pixels=int(max_shell_pixels),
        persistence_status=persistence_status,
        physical_binding=str(claim["physical_binding"]),
        ranking_allowed=False,
    )


def run_matched_compartment_persistence(
    scenarios: Iterable,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizon_steps: int = 3000,
    Nx: int = 24,
    Ny: int = 24,
    boundary_morphogenesis: str = COLOCATED_BASELINE,
) -> list[CompartmentPersistenceResult]:
    scenario_list = list(scenarios)
    seed_list = tuple(int(seed) for seed in seeds)
    if not scenario_list:
        raise ValueError("at least one scenario is required")
    if not seed_list:
        raise ValueError("at least one seed is required")

    return [
        run_compartment_persistence_case(
            scenario,
            seed=seed,
            horizon_steps=horizon_steps,
            Nx=Nx,
            Ny=Ny,
            boundary_morphogenesis=boundary_morphogenesis,
        )
        for scenario in scenario_list
        for seed in seed_list
    ]


def persistence_manifest() -> dict[str, object]:
    return {
        "schema": PERSISTENCE_SCHEMA,
        "threshold_tuning_allowed": False,
        "parameter_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
        "statuses": [
            STATUS_NEVER_LOCALIZED,
            STATUS_LOCALIZED_AT_HORIZON,
            STATUS_LOCALIZED_THEN_LOST,
        ],
    }
