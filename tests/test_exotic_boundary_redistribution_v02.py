from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology import (
    AmmoniaCandidateSimulator,
    HydrocarbonCandidateSimulator,
)
from origins.exobiology.boundary_morphogenesis import (
    COLOCATED_BASELINE,
    EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
    redistribute_boundary_source_to_information_exterior,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_redistribution_preserves_total_source_budget_when_gradient_exists() -> None:
    source = np.ones((9, 9), dtype=float)
    info = np.zeros((9, 9), dtype=float)
    info[3:6, 3:6] = 1.0

    redistributed = redistribute_boundary_source_to_information_exterior(source, info)
    assert float(np.sum(redistributed)) == pytest.approx(float(np.sum(source)), rel=1e-12, abs=1e-12)
    assert redistributed[4, 4] == pytest.approx(0.0)
    assert float(np.sum(redistributed)) > 0.0


def test_redistribution_returns_zero_without_information_interface() -> None:
    source = np.ones((7, 7), dtype=float)
    info = np.zeros((7, 7), dtype=float)
    redistributed = redistribute_boundary_source_to_information_exterior(source, info)
    assert np.count_nonzero(redistributed) == 0


@pytest.mark.parametrize(
    "scenario, simulator_cls",
    [
        (SCENARIO_C, AmmoniaCandidateSimulator),
        (SCENARIO_D, HydrocarbonCandidateSimulator),
    ],
)
def test_redistributed_mode_conserves_material(scenario, simulator_cls) -> None:
    sim = simulator_cls(
        deepcopy(scenario),
        Nx=16,
        Ny=16,
        seed=31,
        boundary_morphogenesis=EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
    )
    sim.initialize()
    before = sim.material_total()
    sim.run(steps=300)
    after = sim.material_total()
    assert after == pytest.approx(before, rel=2e-10, abs=2e-10)


@pytest.mark.parametrize(
    "scenario, simulator_cls",
    [
        (SCENARIO_C, AmmoniaCandidateSimulator),
        (SCENARIO_D, HydrocarbonCandidateSimulator),
    ],
)
def test_redistributed_mode_preserves_instantaneous_budget_after_gradient_exists(
    scenario,
    simulator_cls,
) -> None:
    base = simulator_cls(
        deepcopy(scenario),
        Nx=12,
        Ny=12,
        seed=37,
        boundary_morphogenesis=COLOCATED_BASELINE,
    )
    red = simulator_cls(
        deepcopy(scenario),
        Nx=12,
        Ny=12,
        seed=37,
        boundary_morphogenesis=EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
    )
    base.initialize()
    red.initialize()

    # Let identical information dynamics generate a nonzero gradient first.
    base.step_energy_throughput()
    red.step_energy_throughput()
    if hasattr(base, "step_interface_exchange"):
        base.step_interface_exchange()
        red.step_interface_exchange()
    if hasattr(base, "step_information_and_inheritance"):
        base.step_information_and_inheritance()
        red.step_information_and_inheritance()
    else:
        base.step_information_and_selection()
        red.step_information_and_selection()

    s0 = base.candidate_assembly_sources()["boundary"]
    s1 = red.candidate_assembly_sources()["boundary"]
    assert float(np.sum(s1)) == pytest.approx(float(np.sum(s0)), rel=1e-10, abs=1e-12)
