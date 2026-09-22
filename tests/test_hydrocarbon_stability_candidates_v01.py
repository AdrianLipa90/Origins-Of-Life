from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology.hydrocarbon_runtime import HydrocarbonCandidateSimulator
from origins.exobiology.stability_morphogenesis import (
    ISLAND_PRESERVATION_OFF,
    LOW_BOUNDARY_EXTERIOR_MAINTENANCE_CANDIDATE,
    SHELL_MAINTENANCE_OFF,
    STATE_PEAK_ISLAND_PRESERVATION_CANDIDATE,
    redistribute_boundary_source_to_shell_deficits,
    redistribute_information_source_to_state_peaks,
    state_peak_gate,
)
from origins.scenarios import SCENARIO_D


def test_shell_maintenance_preserves_source_budget_on_existing_exterior() -> None:
    source = np.ones((9, 9), dtype=float)
    information = np.zeros((9, 9), dtype=float)
    information[3:6, 3:6] = 1.0
    boundary = np.ones((9, 9), dtype=float)
    boundary[2, 4] = 0.01

    redistributed = redistribute_boundary_source_to_shell_deficits(
        source,
        information,
        boundary,
    )
    assert float(np.sum(redistributed)) == pytest.approx(
        float(np.sum(source)),
        rel=1e-12,
        abs=1e-12,
    )
    assert redistributed[2, 4] > source[2, 4]


def test_shell_maintenance_is_identity_when_no_information_exterior_exists() -> None:
    source = np.arange(1, 50, dtype=float).reshape(7, 7)
    information = np.zeros((7, 7), dtype=float)
    boundary = np.ones((7, 7), dtype=float)
    redistributed = redistribute_boundary_source_to_shell_deficits(
        source,
        information,
        boundary,
    )
    assert np.array_equal(redistributed, source)


def test_state_peak_gate_selects_local_information_peak() -> None:
    information = np.zeros((7, 7), dtype=float)
    information[3, 3] = 1.0
    gate = state_peak_gate(information)
    assert gate[3, 3] > 0.0
    assert gate[0, 0] == pytest.approx(0.0)


def test_island_preservation_preserves_information_source_budget() -> None:
    source = np.ones((7, 7), dtype=float)
    information = np.zeros((7, 7), dtype=float)
    information[3, 3] = 1.0
    redistributed = redistribute_information_source_to_state_peaks(
        source,
        information,
    )
    assert float(np.sum(redistributed)) == pytest.approx(
        float(np.sum(source)),
        rel=1e-12,
        abs=1e-12,
    )
    assert redistributed[3, 3] > source[3, 3]


def test_island_preservation_is_identity_without_state_peak() -> None:
    source = np.arange(1, 50, dtype=float).reshape(7, 7)
    information = np.zeros((7, 7), dtype=float)
    redistributed = redistribute_information_source_to_state_peaks(
        source,
        information,
    )
    assert np.array_equal(redistributed, source)


def test_explicit_off_modes_reproduce_default_runtime() -> None:
    default = HydrocarbonCandidateSimulator(
        deepcopy(SCENARIO_D),
        Nx=12,
        Ny=12,
        seed=17,
    )
    explicit = HydrocarbonCandidateSimulator(
        deepcopy(SCENARIO_D),
        Nx=12,
        Ny=12,
        seed=17,
        shell_maintenance=SHELL_MAINTENANCE_OFF,
        island_preservation=ISLAND_PRESERVATION_OFF,
    )
    default.initialize()
    explicit.initialize()
    default.run(steps=120)
    explicit.run(steps=120)

    for name in ("S", "A", "I", "B", "E", "Q", "interface_template"):
        assert np.array_equal(getattr(default, name), getattr(explicit, name))


@pytest.mark.parametrize(
    "shell_mode,island_mode",
    [
        (LOW_BOUNDARY_EXTERIOR_MAINTENANCE_CANDIDATE, ISLAND_PRESERVATION_OFF),
        (SHELL_MAINTENANCE_OFF, STATE_PEAK_ISLAND_PRESERVATION_CANDIDATE),
        (
            LOW_BOUNDARY_EXTERIOR_MAINTENANCE_CANDIDATE,
            STATE_PEAK_ISLAND_PRESERVATION_CANDIDATE,
        ),
    ],
)
def test_stability_modes_conserve_material_and_source_budgets(
    shell_mode,
    island_mode,
) -> None:
    sim = HydrocarbonCandidateSimulator(
        deepcopy(SCENARIO_D),
        Nx=16,
        Ny=16,
        seed=23,
        shell_maintenance=shell_mode,
        island_preservation=island_mode,
    )
    sim.initialize()
    initial_material = sim.material_total()

    for _ in range(300):
        sim.step()
        sources = sim.candidate_assembly_sources()

        info_pre = float(np.sum(sources["information_pre_stability"]))
        info_post = float(np.sum(sources["information"]))
        boundary_pre = float(np.sum(sources["boundary_pre_stability"]))
        boundary_post = float(np.sum(sources["boundary"]))

        assert info_post == pytest.approx(info_pre, rel=1e-10, abs=1e-12)
        assert boundary_post == pytest.approx(boundary_pre, rel=1e-10, abs=1e-12)

    assert sim.material_total() == pytest.approx(
        initial_material,
        rel=2e-10,
        abs=2e-10,
    )
    status = sim.claim_status()
    assert status["shell_maintenance"] == shell_mode
    assert status["island_preservation"] == island_mode
    assert status["morphogenesis_physical_binding"] == "OPEN"


def test_unknown_stability_modes_fail_closed() -> None:
    with pytest.raises(ValueError):
        HydrocarbonCandidateSimulator(
            deepcopy(SCENARIO_D),
            Nx=8,
            Ny=8,
            shell_maintenance="MAGIC_SHELL",
        )
    with pytest.raises(ValueError):
        HydrocarbonCandidateSimulator(
            deepcopy(SCENARIO_D),
            Nx=8,
            Ny=8,
            island_preservation="MAGIC_ISLAND",
        )
