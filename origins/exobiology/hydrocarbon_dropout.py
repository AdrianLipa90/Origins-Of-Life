from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .boundary_morphogenesis import EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE
from .compartments import periodic_component_topology
from .factory import create_exotic_candidate_simulator
from .information_morphogenesis import PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE
from .source_geometry import diagnose_source_geometry


SCHEMA = "ORIGINS_HYDROCARBON_CLOSURE_DROPOUT_V0_1"


def _cv(field: np.ndarray) -> float:
    mean = float(np.mean(field))
    return float(np.std(field) / mean) if mean > 1e-15 else 0.0


def _snapshot(simulator, step: int) -> dict[str, object]:
    p = simulator.parameters
    info = np.asarray(simulator.I, dtype=float)
    boundary = np.asarray(simulator.B, dtype=float)
    observation = simulator.candidate_compartments()
    source = diagnose_source_geometry(simulator)
    info_mask = info >= float(p.information_threshold)
    topology = periodic_component_topology(info_mask)

    return {
        "step": int(step),
        "compartment_count": int(observation["count"]),
        "raw_information_component_count": int(observation["raw_information_component_count"]),
        "largest_information_component_fraction": float(
            topology["largest_component_fraction"]
        ),
        "information_noncontractible_component_count": int(
            topology["noncontractible_component_count"]
        ),
        "information_wraps_x_component_count": int(
            topology["wraps_x_component_count"]
        ),
        "information_wraps_y_component_count": int(
            topology["wraps_y_component_count"]
        ),
        "information_any_noncontractible": bool(
            topology["any_noncontractible"]
        ),
        "candidate_area_pixels": int(observation["area_pixels"]),
        "shell_pixels": int(observation["shell_pixels"]),
        "max_shell_coverage": float(observation["max_shell_coverage"]),
        "global_saturation": bool(observation["global_saturation"]),
        "bounded_system_status": str(observation["bounded_system_status"]),
        "information_threshold_fraction": float(
            np.mean(info >= float(p.information_threshold))
        ),
        "boundary_threshold_fraction": float(
            np.mean(boundary >= float(p.boundary_threshold))
        ),
        "information_min": float(np.min(info)),
        "information_mean": float(np.mean(info)),
        "information_max": float(np.max(info)),
        "information_cv": _cv(info),
        "boundary_min": float(np.min(boundary)),
        "boundary_mean": float(np.mean(boundary)),
        "boundary_max": float(np.max(boundary)),
        "boundary_cv": _cv(boundary),
        "source_cosine_overlap": float(source.cosine_overlap),
        "source_pearson_correlation": float(source.pearson_correlation),
    }


@dataclass(frozen=True)
class HydrocarbonClosureDropoutResult:
    schema: str
    profile: str
    runtime: str
    seed: int
    horizon_steps: int
    first_closed_step: int | None
    last_closed_step: int | None
    first_loss_after_closure_step: int | None
    closure_run_count: int
    reclosure_count: int
    closed_step_count: int
    longest_closed_run: int
    first_closed_snapshot: dict[str, object] | None
    last_closed_snapshot: dict[str, object] | None
    first_loss_snapshot: dict[str, object] | None
    final_snapshot: dict[str, object]
    physical_binding: str
    parameter_tuning_allowed: bool = False
    threshold_tuning_allowed: bool = False
    ranking_allowed: bool = False

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def run_hydrocarbon_closure_dropout_case(
    scenario,
    *,
    seed: int,
    horizon_steps: int = 10000,
    Nx: int = 24,
    Ny: int = 24,
) -> HydrocarbonClosureDropoutResult:
    if getattr(scenario, "biochemistry_profile", None) != "HYDROCARBON_CANDIDATE":
        raise ValueError("closure-dropout diagnostic requires HYDROCARBON_CANDIDATE")
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be positive")

    sim = create_exotic_candidate_simulator(
        scenario,
        Nx=Nx,
        Ny=Ny,
        seed=int(seed),
        information_morphogenesis=PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        boundary_morphogenesis=EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
    )
    sim.initialize()

    first_closed_step: int | None = None
    last_closed_step: int | None = None
    first_loss_step: int | None = None
    first_closed_snapshot: dict[str, object] | None = None
    last_closed_snapshot: dict[str, object] | None = None
    first_loss_snapshot: dict[str, object] | None = None

    ever_closed = False
    was_closed = False
    closure_runs = 0
    reclosures = 0
    closed_steps = 0
    current_run = 0
    longest_run = 0

    for step in range(1, int(horizon_steps) + 1):
        sim.step()
        observation = sim.candidate_compartments()
        closed = int(observation["count"]) > 0

        if closed:
            closed_steps += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
            if not was_closed:
                closure_runs += 1
                if ever_closed:
                    reclosures += 1
            if first_closed_step is None:
                first_closed_step = step
                first_closed_snapshot = _snapshot(sim, step)
            ever_closed = True
            last_closed_step = step
            last_closed_snapshot = _snapshot(sim, step)
        else:
            if was_closed and first_loss_step is None:
                first_loss_step = step
                first_loss_snapshot = _snapshot(sim, step)
            current_run = 0

        was_closed = closed

    claim = sim.claim_status()
    return HydrocarbonClosureDropoutResult(
        schema=SCHEMA,
        profile=str(claim["profile"]),
        runtime=str(claim["runtime"]),
        seed=int(seed),
        horizon_steps=int(horizon_steps),
        first_closed_step=first_closed_step,
        last_closed_step=last_closed_step,
        first_loss_after_closure_step=first_loss_step,
        closure_run_count=int(closure_runs),
        reclosure_count=int(reclosures),
        closed_step_count=int(closed_steps),
        longest_closed_run=int(longest_run),
        first_closed_snapshot=first_closed_snapshot,
        last_closed_snapshot=last_closed_snapshot,
        first_loss_snapshot=first_loss_snapshot,
        final_snapshot=_snapshot(sim, int(horizon_steps)),
        physical_binding=str(claim["physical_binding"]),
        parameter_tuning_allowed=False,
        threshold_tuning_allowed=False,
        ranking_allowed=False,
    )


def dropout_manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "profile": "HYDROCARBON_CANDIDATE",
        "information_mode": PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        "boundary_mode": EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
        "parameter_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "ranking_allowed": False,
        "physical_binding": "OPEN",
        "purpose": "diagnose loss of closed-shell persistence and toroidal information percolation without changing the model",
    }
