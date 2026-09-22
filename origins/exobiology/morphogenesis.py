from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .factory import create_exotic_candidate_simulator


DIAGNOSTIC_SCHEMA = "ORIGINS_EXOTIC_MORPHOGENESIS_SOURCE_OVERLAP_V0_1"


@dataclass(frozen=True)
class MorphogenesisSourceOverlapResult:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    source_distribution_overlap: float
    source_cosine_similarity: float
    state_distribution_overlap: float
    state_cosine_similarity: float
    boundary_source_information_gradient_cosine: float
    information_source_total: float
    boundary_source_total: float
    candidate_compartment_count: int
    bounded_system_status: str
    global_saturation: bool
    physical_binding: str
    ranking_allowed: bool = False
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _nonnegative_flat(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if not np.isfinite(arr).all():
        raise FloatingPointError("diagnostic input contains NaN/Inf")
    if float(np.min(arr)) < -1e-12:
        raise ValueError("diagnostic expects non-negative fields")
    return np.maximum(arr, 0.0).ravel()


def _distribution_overlap(left: np.ndarray, right: np.ndarray) -> float:
    a = _nonnegative_flat(left)
    b = _nonnegative_flat(right)
    sa = float(np.sum(a))
    sb = float(np.sum(b))
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    pa = a / sa
    pb = b / sb
    return float(np.sum(np.minimum(pa, pb)))


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = _nonnegative_flat(left)
    b = _nonnegative_flat(right)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _periodic_gradient_magnitude(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    gx = 0.5 * (np.roll(arr, -1, axis=0) - np.roll(arr, 1, axis=0))
    gy = 0.5 * (np.roll(arr, -1, axis=1) - np.roll(arr, 1, axis=1))
    return np.sqrt(gx * gx + gy * gy)


def run_morphogenesis_source_overlap_case(
    scenario,
    *,
    seed: int,
    horizons: Iterable[int] = (300, 1000, 3000),
    Nx: int = 24,
    Ny: int = 24,
) -> list[MorphogenesisSourceOverlapResult]:
    """Measure source/state colocalization without changing model parameters."""
    horizon_list = tuple(sorted(set(int(h) for h in horizons)))
    if not horizon_list or horizon_list[0] <= 0:
        raise ValueError("horizons must contain positive integers")

    simulator = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=int(seed),
    )
    simulator.initialize()

    out: list[MorphogenesisSourceOverlapResult] = []
    previous = 0

    for horizon in horizon_list:
        simulator.run(horizon - previous)
        previous = horizon

        sources = simulator.candidate_assembly_sources()
        information_source = np.asarray(sources["information"], dtype=float)
        boundary_source = np.asarray(sources["boundary"], dtype=float)

        information_field = np.asarray(simulator.I, dtype=float)
        boundary_field = np.asarray(simulator.B, dtype=float)
        information_gradient = _periodic_gradient_magnitude(information_field)

        observation = simulator.candidate_compartments()
        claim = simulator.claim_status()

        out.append(
            MorphogenesisSourceOverlapResult(
                schema=DIAGNOSTIC_SCHEMA,
                profile=str(claim["profile"]),
                runtime=str(claim["runtime"]),
                seed=int(seed),
                horizon_steps=int(horizon),
                source_distribution_overlap=_distribution_overlap(
                    information_source,
                    boundary_source,
                ),
                source_cosine_similarity=_cosine_similarity(
                    information_source,
                    boundary_source,
                ),
                state_distribution_overlap=_distribution_overlap(
                    information_field,
                    boundary_field,
                ),
                state_cosine_similarity=_cosine_similarity(
                    information_field,
                    boundary_field,
                ),
                boundary_source_information_gradient_cosine=_cosine_similarity(
                    boundary_source,
                    information_gradient,
                ),
                information_source_total=float(np.sum(information_source)),
                boundary_source_total=float(np.sum(boundary_source)),
                candidate_compartment_count=int(observation["count"]),
                bounded_system_status=str(observation["bounded_system_status"]),
                global_saturation=bool(observation["global_saturation"]),
                physical_binding=str(claim["physical_binding"]),
                ranking_allowed=False,
                parameter_tuning_allowed=False,
                threshold_tuning_allowed=False,
            )
        )

    return out


def run_matched_morphogenesis_source_overlap(
    scenarios: Iterable,
    *,
    seeds: Iterable[int] = (11, 23, 47),
    horizons: Iterable[int] = (300, 1000, 3000),
    Nx: int = 24,
    Ny: int = 24,
) -> list[MorphogenesisSourceOverlapResult]:
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
        for row in run_morphogenesis_source_overlap_case(
            scenario,
            seed=seed,
            horizons=horizons,
            Nx=Nx,
            Ny=Ny,
        )
    ]


def diagnostic_manifest() -> dict[str, object]:
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "ranking_allowed": False,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "physical_binding": "OPEN",
        "metrics": [
            "source_distribution_overlap",
            "source_cosine_similarity",
            "state_distribution_overlap",
            "state_cosine_similarity",
            "boundary_source_information_gradient_cosine",
        ],
    }
