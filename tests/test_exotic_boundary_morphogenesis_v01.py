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
    EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
    exterior_information_interface_gate,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_exterior_gradient_gate_is_zero_in_interior_and_positive_outside_edge() -> None:
    info = np.zeros((7, 7), dtype=float)
    info[2:5, 2:5] = 1.0
    gate = exterior_information_interface_gate(info)

    assert gate[3, 3] == pytest.approx(0.0)
    assert gate[0, 0] == pytest.approx(0.0)
    assert gate[1, 3] > 0.0
    assert gate[5, 3] > 0.0
    assert gate[3, 1] > 0.0
    assert gate[3, 5] > 0.0
    assert np.min(gate) >= 0.0
    assert np.max(gate) <= 1.0


@pytest.mark.parametrize(
    "scenario, simulator_cls",
    [
        (SCENARIO_C, AmmoniaCandidateSimulator),
        (SCENARIO_D, HydrocarbonCandidateSimulator),
    ],
)
def test_default_boundary_morphogenesis_is_explicit_colocated_baseline(
    scenario,
    simulator_cls,
) -> None:
    implicit = simulator_cls(deepcopy(scenario), Nx=12, Ny=12, seed=17)
    explicit = simulator_cls(
        deepcopy(scenario),
        Nx=12,
        Ny=12,
        seed=17,
        boundary_morphogenesis=COLOCATED_BASELINE,
    )
    implicit.initialize()
    explicit.initialize()
    implicit.run(steps=80)
    explicit.run(steps=80)

    assert implicit.boundary_morphogenesis == COLOCATED_BASELINE
    for name in ("I", "B", "E", "Q"):
        assert np.array_equal(getattr(implicit, name), getattr(explicit, name))


@pytest.mark.parametrize(
    "scenario, simulator_cls",
    [
        (SCENARIO_C, AmmoniaCandidateSimulator),
        (SCENARIO_D, HydrocarbonCandidateSimulator),
    ],
)
def test_exterior_gradient_mode_conserves_material(
    scenario,
    simulator_cls,
) -> None:
    sim = simulator_cls(
        deepcopy(scenario),
        Nx=16,
        Ny=16,
        seed=19,
        boundary_morphogenesis=EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
    )
    sim.initialize()
    before = sim.material_total()
    sim.run(steps=300)
    after = sim.material_total()
    assert after == pytest.approx(before, rel=2e-10, abs=2e-10)
    assert sim.claim_status()["boundary_morphogenesis"] == (
        EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE
    )
    assert sim.claim_status()["morphogenesis_physical_binding"] == "OPEN"


def test_ammonia_gradient_mode_moves_boundary_source_to_exterior_signal() -> None:
    sim = AmmoniaCandidateSimulator(
        deepcopy(SCENARIO_C),
        Nx=7,
        Ny=7,
        seed=23,
        boundary_morphogenesis=EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
    )
    sim.initialize()
    sim.I[:] = 0.0
    sim.I[2:5, 2:5] = 1.0

    boundary = sim.candidate_assembly_sources()["boundary"]
    assert boundary[3, 3] == pytest.approx(0.0)
    assert boundary[1, 3] > 0.0
    assert boundary[5, 3] > 0.0


def test_hydrocarbon_gradient_mode_uses_information_exterior_on_existing_interface() -> None:
    sim = HydrocarbonCandidateSimulator(
        deepcopy(SCENARIO_D),
        Nx=7,
        Ny=7,
        seed=29,
        boundary_morphogenesis=EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
    )
    sim.initialize()
    sim.A[:] = 0.4
    sim.I[:] = 0.0
    sim.I[2:5, 2:5] = 1.0

    boundary = sim.candidate_assembly_sources()["boundary"]
    assert boundary[3, 3] == pytest.approx(0.0)
    assert float(np.sum(boundary)) > 0.0


@pytest.mark.parametrize(
    "scenario, simulator_cls",
    [
        (SCENARIO_C, AmmoniaCandidateSimulator),
        (SCENARIO_D, HydrocarbonCandidateSimulator),
    ],
)
def test_unknown_boundary_morphogenesis_mode_fails_closed(
    scenario,
    simulator_cls,
) -> None:
    with pytest.raises(ValueError):
        simulator_cls(
            deepcopy(scenario),
            Nx=8,
            Ny=8,
            boundary_morphogenesis="MAGIC_SHELL",
        )
