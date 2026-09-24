from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .boundary_morphogenesis import EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE
from .compartments import periodic_component_topology
from .factory import create_exotic_candidate_simulator
from .information_morphogenesis import PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE
from .stability_morphogenesis import (
    ISLAND_PRESERVATION_OFF,
    LOW_BOUNDARY_EXTERIOR_MAINTENANCE_CANDIDATE,
    SHELL_MAINTENANCE_OFF,
    STATE_PEAK_ISLAND_PRESERVATION_CANDIDATE,
)


SCHEMA = "ORIGINS_HYDROCARBON_STABILITY_FACTORIAL_V0_5"

SHELL_MODES = (
    SHELL_MAINTENANCE_OFF,
    LOW_BOUNDARY_EXTERIOR_MAINTENANCE_CANDIDATE,
)
ISLAND_MODES = (
    ISLAND_PRESERVATION_OFF,
    STATE_PEAK_ISLAND_PRESERVATION_CANDIDATE,
)

OUTCOME_PERSISTS = "PERSISTS_WITHIN_HORIZON"
OUTCOME_NEVER_CLOSED = "NEVER_CLOSED_WITHIN_HORIZON"
OUTCOME_INTERIOR_LOSS = "CONTRACTIBLE_INTERIOR_LOSS"
OUTCOME_SHELL_GAP = "CONTRACTIBLE_SHELL_GAP"
OUTCOME_OTHER = "OTHER_OPEN"


@dataclass(frozen=True)
class HydrocarbonStabilityFactorialResult:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    shell_maintenance: str
    island_preservation: str
    first_closed_step: int | None
    last_closed_step: int | None
    first_loss_after_closure_step: int | None
    closed_step_count: int
    longest_closed_run: int
    final_compartment_count: int
    final_contractible_information_components: int
    final_noncontractible_information_components: int
    final_largest_information_component_fraction: float
    final_best_contractible_shell_coverage: float
    final_best_contractible_missing_shell_pixels: int
    dropout_mechanism: str
    max_information_stability_budget_relative_error: float
    max_boundary_stability_budget_relative_error: float
    material_residual_abs: float
    physical_binding: str
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _relative_error(actual: float, reference: float) -> float:
    if reference <= 1e-15:
        return 0.0 if actual <= 1e-15 else float("inf")
    return float(abs(actual - reference) / reference)


def _classify_dropout(
    *,
    ever_closed: bool,
    closed_at_horizon: bool,
    first_loss_observation: dict[str, object] | None,
) -> str:
    if closed_at_horizon:
        return OUTCOME_PERSISTS
    if not ever_closed:
        return OUTCOME_NEVER_CLOSED
    if first_loss_observation is None:
        return OUTCOME_OTHER

    contractible = int(
        first_loss_observation["contractible_information_component_count"]
    )
    if contractible == 0:
        return OUTCOME_INTERIOR_LOSS

    coverage = float(
        first_loss_observation["best_contractible_shell_coverage"]
    )
    if coverage < 1.0:
        return OUTCOME_SHELL_GAP
    return OUTCOME_OTHER


def run_hydrocarbon_stability_factorial_case(
    scenario,
    *,
    seed: int,
    shell_maintenance: str,
    island_preservation: str,
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> HydrocarbonStabilityFactorialResult:
    if getattr(scenario, "biochemistry_profile", None) != "HYDROCARBON_CANDIDATE":
        raise ValueError("stability factorial requires HYDROCARBON_CANDIDATE")
    if shell_maintenance not in SHELL_MODES:
        raise ValueError(f"unknown shell maintenance mode: {shell_maintenance!r}")
    if island_preservation not in ISLAND_MODES:
        raise ValueError(f"unknown island preservation mode: {island_preservation!r}")
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be positive")

    sim = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=int(seed),
        information_morphogenesis=PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        boundary_morphogenesis=EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
        shell_maintenance=shell_maintenance,
        island_preservation=island_preservation,
    )
    sim.initialize()
    initial_material = sim.material_total()

    first_closed: int | None = None
    last_closed: int | None = None
    first_loss: int | None = None
    first_loss_observation: dict[str, object] | None = None
    ever_closed = False
    was_closed = False
    closed_steps = 0
    current_run = 0
    longest_run = 0
    max_info_budget_error = 0.0
    max_boundary_budget_error = 0.0

    for step in range(1, int(horizon_steps) + 1):
        sim.step()

        sources = sim.candidate_assembly_sources()
        info_pre = float(np.sum(sources["information_pre_stability"]))
        info_post = float(np.sum(sources["information"]))
        bound_pre = float(np.sum(sources["boundary_pre_stability"]))
        bound_post = float(np.sum(sources["boundary"]))
        max_info_budget_error = max(
            max_info_budget_error,
            _relative_error(info_post, info_pre),
        )
        max_boundary_budget_error = max(
            max_boundary_budget_error,
            _relative_error(bound_post, bound_pre),
        )

        observation = sim.candidate_compartments()
        closed = int(observation["count"]) > 0
        if closed:
            if first_closed is None:
                first_closed = step
            last_closed = step
            closed_steps += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
            ever_closed = True
        else:
            if was_closed and first_loss is None:
                first_loss = step
                first_loss_observation = dict(observation)
            current_run = 0
        was_closed = closed

    observation = sim.candidate_compartments()
    info_mask = np.asarray(sim.I) >= float(sim.parameters.information_threshold)
    topology = periodic_component_topology(info_mask)
    claim = sim.claim_status()

    dropout = _classify_dropout(
        ever_closed=ever_closed,
        closed_at_horizon=int(observation["count"]) > 0,
        first_loss_observation=first_loss_observation,
    )

    return HydrocarbonStabilityFactorialResult(
        schema=SCHEMA,
        profile=str(claim["profile"]),
        runtime=str(claim["runtime"]),
        seed=int(seed),
        horizon_steps=int(horizon_steps),
        shell_maintenance=str(shell_maintenance),
        island_preservation=str(island_preservation),
        first_closed_step=first_closed,
        last_closed_step=last_closed,
        first_loss_after_closure_step=first_loss,
        closed_step_count=int(closed_steps),
        longest_closed_run=int(longest_run),
        final_compartment_count=int(observation["count"]),
        final_contractible_information_components=int(
            observation["contractible_information_component_count"]
        ),
        final_noncontractible_information_components=int(
            observation["noncontractible_information_component_count"]
        ),
        final_largest_information_component_fraction=float(
            topology["largest_component_fraction"]
        ),
        final_best_contractible_shell_coverage=float(
            observation["best_contractible_shell_coverage"]
        ),
        final_best_contractible_missing_shell_pixels=int(
            observation["best_contractible_missing_shell_pixels"]
        ),
        dropout_mechanism=dropout,
        max_information_stability_budget_relative_error=float(
            max_info_budget_error
        ),
        max_boundary_stability_budget_relative_error=float(
            max_boundary_budget_error
        ),
        material_residual_abs=float(abs(sim.material_total() - initial_material)),
        physical_binding=str(claim["physical_binding"]),
        parameter_tuning_allowed=False,
        threshold_tuning_allowed=False,
        ranking_allowed=False,
    )


def run_hydrocarbon_stability_factorial(
    scenario,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> list[HydrocarbonStabilityFactorialResult]:
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    return [
        run_hydrocarbon_stability_factorial_case(
            scenario,
            seed=seed,
            shell_maintenance=shell_mode,
            island_preservation=island_mode,
            horizon_steps=horizon_steps,
            Nx=Nx,
            Ny=Ny,
        )
        for shell_mode in SHELL_MODES
        for island_mode in ISLAND_MODES
        for seed in seed_list
    ]


def factorial_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "profile": "HYDROCARBON_CANDIDATE",
        "fixed_information_morphogenesis": PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        "fixed_boundary_morphogenesis": EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
        "shell_modes": list(SHELL_MODES),
        "island_modes": list(ISLAND_MODES),
        "design": "2x2_MATCHED_SEED_CAUSAL_SPLIT",
        "same_parameters_required": True,
        "same_thresholds_required": True,
        "stability_source_budgets_preserved": True,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
        "outcome_taxonomy": [
            OUTCOME_PERSISTS,
            OUTCOME_NEVER_CLOSED,
            OUTCOME_INTERIOR_LOSS,
            OUTCOME_SHELL_GAP,
            OUTCOME_OTHER,
        ],
    }
