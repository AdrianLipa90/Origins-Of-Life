from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np

from .boundary_morphogenesis import EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE
from .compartments import periodic_component_topology
from .factory import create_exotic_candidate_simulator
from .information_morphogenesis import PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE


SCHEMA = "ORIGINS_HYDROCARBON_INFORMATION_DIFFUSION_ABLATION_V0_1"

DIFFUSION_BASELINE = "INFORMATION_DIFFUSION_BASELINE"
DIFFUSION_OFF = "INFORMATION_DIFFUSION_OFF"


@dataclass(frozen=True)
class InformationDiffusionAblationResult:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    diffusion_condition: str
    information_diffusion: float
    first_closed_step: int | None
    last_closed_step: int | None
    first_loss_after_closure_step: int | None
    closed_step_count: int
    longest_closed_run: int
    closed_at_horizon: bool
    final_compartment_count: int
    final_contractible_information_components: int
    final_noncontractible_information_components: int
    final_largest_information_component_fraction: float
    final_information_mass: float
    final_boundary_mass: float
    material_residual_abs: float
    physical_binding: str
    causal_ablation: bool = True
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def run_information_diffusion_condition(
    scenario,
    *,
    seed: int,
    diffusion_condition: str,
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> InformationDiffusionAblationResult:
    if getattr(scenario, "biochemistry_profile", None) != "HYDROCARBON_CANDIDATE":
        raise ValueError("diffusion ablation requires HYDROCARBON_CANDIDATE")
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be positive")
    if diffusion_condition not in (DIFFUSION_BASELINE, DIFFUSION_OFF):
        raise ValueError(f"unknown diffusion condition: {diffusion_condition!r}")

    sim = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=int(seed),
        information_morphogenesis=PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        boundary_morphogenesis=EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
    )
    if diffusion_condition == DIFFUSION_OFF:
        sim.parameters = replace(sim.parameters, information_diffusion=0.0)
        sim.parameters.validate()

    sim.initialize()
    initial_material = sim.material_total()

    first_closed: int | None = None
    last_closed: int | None = None
    first_loss: int | None = None
    closed_steps = 0
    current_run = 0
    longest_run = 0
    was_closed = False

    for step in range(1, int(horizon_steps) + 1):
        sim.step()
        observation = sim.candidate_compartments()
        closed = int(observation["count"]) > 0

        if closed:
            if first_closed is None:
                first_closed = step
            last_closed = step
            closed_steps += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            if was_closed and first_loss is None:
                first_loss = step
            current_run = 0
        was_closed = closed

    observation = sim.candidate_compartments()
    info_mask = np.asarray(sim.I) >= float(sim.parameters.information_threshold)
    topology = periodic_component_topology(info_mask)
    claim = sim.claim_status()

    return InformationDiffusionAblationResult(
        schema=SCHEMA,
        profile=str(claim["profile"]),
        runtime=str(claim["runtime"]),
        seed=int(seed),
        horizon_steps=int(horizon_steps),
        diffusion_condition=str(diffusion_condition),
        information_diffusion=float(sim.parameters.information_diffusion),
        first_closed_step=first_closed,
        last_closed_step=last_closed,
        first_loss_after_closure_step=first_loss,
        closed_step_count=int(closed_steps),
        longest_closed_run=int(longest_run),
        closed_at_horizon=int(observation["count"]) > 0,
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
        final_information_mass=float(np.sum(sim.I)),
        final_boundary_mass=float(np.sum(sim.B)),
        material_residual_abs=float(abs(sim.material_total() - initial_material)),
        physical_binding=str(claim["physical_binding"]),
        causal_ablation=True,
        parameter_tuning_allowed=False,
        threshold_tuning_allowed=False,
        ranking_allowed=False,
    )


def run_information_diffusion_ablation(
    scenario,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> list[InformationDiffusionAblationResult]:
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    return [
        run_information_diffusion_condition(
            scenario,
            seed=seed,
            diffusion_condition=condition,
            horizon_steps=horizon_steps,
            Nx=Nx,
            Ny=Ny,
        )
        for condition in (DIFFUSION_BASELINE, DIFFUSION_OFF)
        for seed in seed_list
    ]


def ablation_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "profile": "HYDROCARBON_CANDIDATE",
        "fixed_information_morphogenesis": PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        "fixed_boundary_morphogenesis": EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
        "conditions": [DIFFUSION_BASELINE, DIFFUSION_OFF],
        "only_varied_operator": "information_diffusion",
        "same_seed_required": True,
        "same_thresholds_required": True,
        "same_other_parameters_required": True,
        "causal_ablation": True,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
    }
