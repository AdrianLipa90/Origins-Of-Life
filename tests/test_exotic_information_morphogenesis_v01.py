from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology import (
    AmmoniaCandidateSimulator,
    HydrocarbonCandidateSimulator,
)
from origins.exobiology.information_morphogenesis import (
    DISTRIBUTED_INFORMATION_BASELINE,
    PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
    local_source_peak_gate,
    redistribute_information_source_to_local_peaks,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_peak_gate_selects_local_source_maximum() -> None:
    source = np.ones((7, 7), dtype=float)
    source[3, 3] = 4.0
    gate = local_source_peak_gate(source)
    assert gate[3, 3] > 0.0
    assert gate[0, 0] == pytest.approx(0.0)
    assert np.min(gate) >= 0.0
    assert np.max(gate) <= 1.0


def test_information_redistribution_preserves_total_budget() -> None:
    source = np.ones((9, 9), dtype=float)
    source[2, 2] = 3.0
    source[6, 6] = 2.0
    redistributed = redistribute_information_source_to_local_peaks(source)
    assert float(np.sum(redistributed)) == pytest.approx(
        float(np.sum(source)),
        rel=1e-12,
        abs=1e-12,
    )
    assert np.count_nonzero(redistributed) < np.count_nonzero(source)


def test_uniform_information_source_falls_back_to_baseline() -> None:
    source = np.ones((8, 8), dtype=float)
    redistributed = redistribute_information_source_to_local_peaks(source)
    assert np.array_equal(redistributed, source)


@pytest.mark.parametrize(
    "scenario, simulator_cls",
    [
        (SCENARIO_C, AmmoniaCandidateSimulator),
        (SCENARIO_D, HydrocarbonCandidateSimulator),
    ],
)
def test_default_information_mode_is_explicit_distributed_baseline(
    scenario,
    simulator_cls,
) -> None:
    implicit = simulator_cls(deepcopy(scenario), Nx=12, Ny=12, seed=43)
    explicit = simulator_cls(
        deepcopy(scenario),
        Nx=12,
        Ny=12,
        seed=43,
        information_morphogenesis=DISTRIBUTED_INFORMATION_BASELINE,
    )
    implicit.initialize()
    explicit.initialize()
    implicit.run(steps=80)
    explicit.run(steps=80)
    assert implicit.information_morphogenesis == DISTRIBUTED_INFORMATION_BASELINE
    for name in ("I", "B", "E", "Q"):
        assert np.array_equal(getattr(implicit, name), getattr(explicit, name))


@pytest.mark.parametrize(
    "scenario, simulator_cls",
    [
        (SCENARIO_C, AmmoniaCandidateSimulator),
        (SCENARIO_D, HydrocarbonCandidateSimulator),
    ],
)
def test_peak_information_mode_preserves_material(scenario, simulator_cls) -> None:
    sim = simulator_cls(
        deepcopy(scenario),
        Nx=16,
        Ny=16,
        seed=47,
        information_morphogenesis=PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
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
def test_peak_information_mode_preserves_same_state_source_budget(
    scenario,
    simulator_cls,
) -> None:
    sim = simulator_cls(
        deepcopy(scenario),
        Nx=12,
        Ny=12,
        seed=53,
        information_morphogenesis=PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
    )
    sim.initialize()
    sim.step_energy_throughput()
    if hasattr(sim, "step_interface_exchange"):
        sim.step_interface_exchange()
    sources = sim.candidate_assembly_sources()
    assert float(np.sum(sources["information"])) == pytest.approx(
        float(np.sum(sources["information_baseline"])),
        rel=1e-10,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "scenario, simulator_cls",
    [
        (SCENARIO_C, AmmoniaCandidateSimulator),
        (SCENARIO_D, HydrocarbonCandidateSimulator),
    ],
)
def test_unknown_information_morphogenesis_mode_fails_closed(
    scenario,
    simulator_cls,
) -> None:
    with pytest.raises(ValueError):
        simulator_cls(
            deepcopy(scenario),
            Nx=8,
            Ny=8,
            information_morphogenesis="MAGIC_INFORMATION",
        )
