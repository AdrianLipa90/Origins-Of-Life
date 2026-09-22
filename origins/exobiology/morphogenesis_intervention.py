from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .boundary_morphogenesis import (
    COLOCATED_BASELINE,
    EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
)
from .persistence import run_compartment_persistence_case


SCHEMA = "ORIGINS_EXOTIC_BOUNDARY_MORPHOGENESIS_INTERVENTION_V0_1"


@dataclass(frozen=True)
class BoundaryMorphogenesisInterventionResult:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    baseline_mode: str
    intervention_mode: str
    baseline_persistence_status: str
    intervention_persistence_status: str
    baseline_first_localized_step: int | None
    intervention_first_localized_step: int | None
    baseline_localized_fraction: float
    intervention_localized_fraction: float
    delta_localized_fraction: float
    baseline_longest_localized_run: int
    intervention_longest_localized_run: int
    delta_longest_localized_run: int
    baseline_final_global_saturation: bool
    intervention_final_global_saturation: bool
    baseline_first_global_saturation_step: int | None
    intervention_first_global_saturation_step: int | None
    physical_binding: str
    interpretation: str
    ranking_allowed: bool = False
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def run_boundary_morphogenesis_intervention_case(
    scenario,
    *,
    seed: int,
    horizon_steps: int = 3000,
    Nx: int = 24,
    Ny: int = 24,
) -> BoundaryMorphogenesisInterventionResult:
    """Matched structural intervention: only the boundary operator changes."""
    baseline = run_compartment_persistence_case(
        scenario,
        seed=seed,
        horizon_steps=horizon_steps,
        Nx=Nx,
        Ny=Ny,
        boundary_morphogenesis=COLOCATED_BASELINE,
    )
    intervention = run_compartment_persistence_case(
        scenario,
        seed=seed,
        horizon_steps=horizon_steps,
        Nx=Nx,
        Ny=Ny,
        boundary_morphogenesis=EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
    )

    if baseline.profile != intervention.profile or baseline.runtime != intervention.runtime:
        raise RuntimeError("matched morphogenesis intervention changed runtime identity")

    return BoundaryMorphogenesisInterventionResult(
        schema=SCHEMA,
        profile=baseline.profile,
        runtime=baseline.runtime,
        seed=int(seed),
        horizon_steps=int(horizon_steps),
        baseline_mode=baseline.boundary_morphogenesis,
        intervention_mode=intervention.boundary_morphogenesis,
        baseline_persistence_status=baseline.persistence_status,
        intervention_persistence_status=intervention.persistence_status,
        baseline_first_localized_step=baseline.first_localized_step,
        intervention_first_localized_step=intervention.first_localized_step,
        baseline_localized_fraction=baseline.localized_fraction,
        intervention_localized_fraction=intervention.localized_fraction,
        delta_localized_fraction=(
            intervention.localized_fraction - baseline.localized_fraction
        ),
        baseline_longest_localized_run=baseline.longest_consecutive_localized_steps,
        intervention_longest_localized_run=intervention.longest_consecutive_localized_steps,
        delta_longest_localized_run=(
            intervention.longest_consecutive_localized_steps
            - baseline.longest_consecutive_localized_steps
        ),
        baseline_final_global_saturation=baseline.final_global_saturation,
        intervention_final_global_saturation=intervention.final_global_saturation,
        baseline_first_global_saturation_step=baseline.first_global_saturation_step,
        intervention_first_global_saturation_step=intervention.first_global_saturation_step,
        physical_binding=baseline.physical_binding,
        interpretation="MODEL_OPERATOR_EFFECT_ONLY",
        ranking_allowed=False,
        parameter_tuning_allowed=False,
        threshold_tuning_allowed=False,
    )


def run_matched_boundary_morphogenesis_intervention(
    scenarios: Iterable,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizon_steps: int = 3000,
    Nx: int = 24,
    Ny: int = 24,
) -> list[BoundaryMorphogenesisInterventionResult]:
    scenario_list = list(scenarios)
    seed_list = tuple(int(seed) for seed in seeds)
    if not scenario_list:
        raise ValueError("at least one scenario is required")
    if not seed_list:
        raise ValueError("at least one seed is required")

    return [
        run_boundary_morphogenesis_intervention_case(
            scenario,
            seed=seed,
            horizon_steps=horizon_steps,
            Nx=Nx,
            Ny=Ny,
        )
        for scenario in scenario_list
        for seed in seed_list
    ]


def intervention_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "baseline_mode": COLOCATED_BASELINE,
        "intervention_mode": EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
        "matched_seed_required": True,
        "same_parameters_required": True,
        "same_thresholds_required": True,
        "ranking_allowed": False,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "physical_binding": "OPEN",
        "interpretation": "MODEL_OPERATOR_EFFECT_ONLY",
    }
