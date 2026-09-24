from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology.compartments import (
    bounded_compartment_observation,
    closed_boundary_compartment_observation,
)
from origins.exobiology import (
    AmmoniaCandidateSimulator,
    HydrocarbonCandidateSimulator,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_full_periodic_domain_is_global_saturation_not_a_bounded_compartment() -> None:
    obs = bounded_compartment_observation(np.ones((8, 8), dtype=bool))
    assert obs["raw_component_count"] == 1
    assert obs["count"] == 0
    assert obs["area_pixels"] == 64
    assert obs["occupancy_fraction"] == pytest.approx(1.0)
    assert obs["interface_edge_count"] == 0
    assert obs["global_saturation"] is True
    assert obs["bounded_system_status"] == "GLOBAL_SATURATION"


def test_localized_island_has_nonzero_interface_and_counts_as_candidate() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 3:6] = True
    obs = bounded_compartment_observation(mask)
    assert obs["raw_component_count"] == 1
    assert obs["count"] == 1
    assert obs["area_pixels"] == 9
    assert 0.0 < obs["occupancy_fraction"] < 1.0
    assert obs["interface_edge_count"] > 0
    assert obs["global_saturation"] is False
    assert obs["bounded_system_status"] == "LOCALIZED_CANDIDATE"


def test_empty_domain_has_no_compartment() -> None:
    obs = bounded_compartment_observation(np.zeros((8, 8), dtype=bool))
    assert obs["raw_component_count"] == 0
    assert obs["count"] == 0
    assert obs["area_pixels"] == 0
    assert obs["interface_edge_count"] == 0
    assert obs["global_saturation"] is False
    assert obs["bounded_system_status"] == "NONE"


@pytest.mark.parametrize(
    "simulator_cls,scenario",
    [
        (AmmoniaCandidateSimulator, SCENARIO_C),
        (HydrocarbonCandidateSimulator, SCENARIO_D),
    ],
)
def test_runtime_cannot_report_full_grid_saturation_as_bounded_system(
    simulator_cls,
    scenario,
) -> None:
    sim = simulator_cls(deepcopy(scenario), Nx=8, Ny=8, seed=17)
    sim.initialize()
    sim.I.fill(sim.parameters.information_threshold * 2.0)
    sim.B.fill(sim.parameters.boundary_threshold * 2.0)

    obs = sim.candidate_compartments()
    assert obs["raw_information_component_count"] == 1
    assert obs["count"] == 0
    assert obs["global_saturation"] is True
    assert obs["shell_pixels"] == 0
    assert obs["max_shell_coverage"] == pytest.approx(0.0)
    assert obs["bounded_system_status"] == "GLOBAL_SATURATION_REJECTED"

    invariant = sim.life_invariant_status()["BOUNDED_SYSTEM"]
    assert invariant["detected_count"] == 0
    assert invariant["global_saturation"] is True


def test_closed_boundary_operator_accepts_information_interior_with_complete_shell() -> None:
    information = np.zeros((7, 7), dtype=float)
    boundary = np.zeros((7, 7), dtype=float)
    information[3, 3] = 1.0
    for i, j in ((2, 3), (4, 3), (3, 2), (3, 4)):
        boundary[i, j] = 1.0

    obs = closed_boundary_compartment_observation(
        information,
        boundary,
        information_threshold=0.5,
        boundary_threshold=0.5,
    )
    assert obs["raw_information_component_count"] == 1
    assert obs["count"] == 1
    assert obs["area_pixels"] == 1
    assert obs["shell_pixels"] == 4
    assert obs["max_shell_coverage"] == pytest.approx(1.0)
    assert obs["global_saturation"] is False
    assert obs["bounded_system_status"] == "CLOSED_BOUNDARY_CANDIDATE"


def test_closed_boundary_operator_rejects_open_shell() -> None:
    information = np.zeros((7, 7), dtype=float)
    boundary = np.zeros((7, 7), dtype=float)
    information[3, 3] = 1.0
    boundary[2, 3] = 1.0
    boundary[4, 3] = 1.0
    boundary[3, 2] = 1.0

    obs = closed_boundary_compartment_observation(
        information,
        boundary,
        information_threshold=0.5,
        boundary_threshold=0.5,
    )
    assert obs["raw_information_component_count"] == 1
    assert obs["count"] == 0
    assert obs["max_shell_coverage"] == pytest.approx(0.75)
    assert obs["bounded_system_status"] == "INTERIOR_NOT_CLOSED"
