from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np

from .ammonia_runtime import (
    AmmoniaCandidateParameters,
    AmmoniaCandidateSimulator,
)
from .factory import create_exotic_candidate_simulator
from .hydrocarbon_runtime import (
    HydrocarbonCandidateParameters,
    HydrocarbonCandidateSimulator,
)


BENCHMARK_SCHEMA = "ORIGINS_EXOTIC_RELATIONAL_BENCHMARK_V0_1"
COMPARISON_SCOPE = "SHARED_RELATIONAL_OBSERVABLES_ONLY"

ABLATIONS: tuple[str, ...] = (
    "BASELINE",
    "NO_SELECTION",
    "NO_INHERITANCE",
    "NO_BOUNDARY_ASSEMBLY",
    "NO_EXTERNAL_ENERGY_INPUT",
)


@dataclass(frozen=True)
class ExoticRelationalBenchmarkResult:
    schema: str
    comparison_scope: str
    runtime: str
    profile: str
    ablation: str
    seed: int
    steps: int
    material_initial: float
    material_final: float
    material_residual_abs: float
    energy_throughput_integral: float
    information_mass: float
    boundary_mass: float
    candidate_compartment_count: int
    candidate_compartment_area_pixels: int
    trait_mean: float
    trait_variance: float
    physical_binding: str
    parameter_status: str
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _parameters_for_ablation(simulator, ablation: str):
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {ablation!r}")

    params = simulator.parameters
    changes: dict[str, float] = {}
    if ablation == "NO_SELECTION":
        changes["selection_strength"] = 0.0
    elif ablation == "NO_INHERITANCE":
        changes["inheritance_rate"] = 0.0
    elif ablation == "NO_BOUNDARY_ASSEMBLY":
        changes["boundary_assembly_rate"] = 0.0
    elif ablation == "NO_EXTERNAL_ENERGY_INPUT":
        changes["energy_input"] = 0.0

    return replace(params, **changes)


def _common_snapshot(simulator) -> dict[str, object]:
    compartments = simulator.candidate_compartments()
    claim = simulator.claim_status()
    return {
        "runtime": claim["runtime"],
        "profile": claim["profile"],
        "energy_throughput_integral": float(simulator.energy_throughput_integral),
        "information_mass": float(np.sum(simulator.I)),
        "boundary_mass": float(np.sum(simulator.B)),
        "candidate_compartment_count": int(compartments["count"]),
        "candidate_compartment_area_pixels": int(compartments["area_pixels"]),
        "trait_mean": float(np.mean(simulator.Q)),
        "trait_variance": float(np.var(simulator.Q)),
        "physical_binding": str(claim["physical_binding"]),
        "parameter_status": str(claim["parameter_status"]),
    }


def run_exotic_relational_case(
    scenario,
    *,
    ablation: str = "BASELINE",
    seed: int = 0,
    steps: int = 200,
    Nx: int = 24,
    Ny: int = 24,
) -> ExoticRelationalBenchmarkResult:
    """Run one substrate-neutral benchmark case.

    Only observables that exist in both dedicated exotic runtimes are returned.
    The result explicitly forbids ranking the candidate chemistries.
    """
    if int(steps) < 0:
        raise ValueError("steps must be non-negative")

    simulator = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=int(seed),
    )
    simulator.parameters = _parameters_for_ablation(simulator, ablation)
    simulator.parameters.validate()
    simulator.initialize()

    material_initial = simulator.material_total()
    simulator.run(int(steps))
    material_final = simulator.material_total()
    snapshot = _common_snapshot(simulator)

    return ExoticRelationalBenchmarkResult(
        schema=BENCHMARK_SCHEMA,
        comparison_scope=COMPARISON_SCOPE,
        runtime=str(snapshot["runtime"]),
        profile=str(snapshot["profile"]),
        ablation=ablation,
        seed=int(seed),
        steps=int(steps),
        material_initial=float(material_initial),
        material_final=float(material_final),
        material_residual_abs=float(abs(material_final - material_initial)),
        energy_throughput_integral=float(snapshot["energy_throughput_integral"]),
        information_mass=float(snapshot["information_mass"]),
        boundary_mass=float(snapshot["boundary_mass"]),
        candidate_compartment_count=int(snapshot["candidate_compartment_count"]),
        candidate_compartment_area_pixels=int(snapshot["candidate_compartment_area_pixels"]),
        trait_mean=float(snapshot["trait_mean"]),
        trait_variance=float(snapshot["trait_variance"]),
        physical_binding=str(snapshot["physical_binding"]),
        parameter_status=str(snapshot["parameter_status"]),
        ranking_allowed=False,
    )


def run_matched_exotic_relational_benchmark(
    scenarios: Iterable,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    steps: int = 200,
    Nx: int = 24,
    Ny: int = 24,
    ablations: Iterable[str] = ABLATIONS,
) -> list[ExoticRelationalBenchmarkResult]:
    """Run matched seeds/ablations across dedicated exotic candidate runtimes."""
    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("at least one scenario is required")

    ablation_list = tuple(ablations)
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")

    results: list[ExoticRelationalBenchmarkResult] = []
    for scenario in scenario_list:
        for ablation in ablation_list:
            if ablation not in ABLATIONS:
                raise ValueError(f"unknown ablation: {ablation!r}")
            for seed in seed_list:
                results.append(
                    run_exotic_relational_case(
                        scenario,
                        ablation=ablation,
                        seed=seed,
                        steps=steps,
                        Nx=Nx,
                        Ny=Ny,
                    )
                )
    return results


def benchmark_manifest() -> dict[str, object]:
    return {
        "schema": BENCHMARK_SCHEMA,
        "comparison_scope": COMPARISON_SCOPE,
        "ablations": list(ABLATIONS),
        "matched_seed_required": True,
        "ranking_allowed": False,
        "shared_observables": [
            "material_residual_abs",
            "energy_throughput_integral",
            "information_mass",
            "boundary_mass",
            "candidate_compartment_count",
            "candidate_compartment_area_pixels",
            "trait_mean",
            "trait_variance",
        ],
        "non_claims": [
            "raw kinetic coefficients are not compared across candidate chemistries",
            "no runtime is ranked as a more plausible chemistry",
            "physical molecular binding remains open",
        ],
    }
