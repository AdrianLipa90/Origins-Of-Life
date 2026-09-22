from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology.compartments import (
    bounded_compartment_observation,
    closed_boundary_compartment_observation,
)
from origins.exobiology.factory import create_exotic_candidate_simulator
from origins.exobiology.persistence import (
    PERSISTENCE_SCHEMA,
    STATUS_LOCALIZED_AT_HORIZON,
    STATUS_LOCALIZED_THEN_LOST,
    STATUS_NEVER_LOCALIZED,
    persistence_manifest,
    run_compartment_persistence_case,
    run_matched_compartment_persistence,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_periodic_component_connectivity_wraps_opposite_edges() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 2] = True
    mask[4, 2] = True
    observation = bounded_compartment_observation(mask)
    assert observation["raw_component_count"] == 1
    assert observation["count"] == 1
    assert observation["bounded_system_status"] == "LOCALIZED_CANDIDATE"
    assert observation["interface_edge_count"] > 0


def test_periodic_component_counter_keeps_separate_regions_separate() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 2] = True
    mask[4, 2] = True
    mask[2, 4] = True
    observation = bounded_compartment_observation(mask)
    assert observation["raw_component_count"] == 2
    assert observation["count"] == 2


def test_global_saturation_is_not_a_bounded_compartment() -> None:
    observation = bounded_compartment_observation(np.ones((6, 6), dtype=bool))
    assert observation["raw_component_count"] == 1
    assert observation["count"] == 0
    assert observation["global_saturation"] is True
    assert observation["interface_edge_count"] == 0
    assert observation["bounded_system_status"] == "GLOBAL_SATURATION"


def test_persistence_manifest_forbids_tuning_and_ranking() -> None:
    manifest = persistence_manifest()
    assert manifest["schema"] == PERSISTENCE_SCHEMA
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False
    assert manifest["physical_binding"] == "OPEN"


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_persistence_result_is_descriptive_not_ranked(scenario) -> None:
    result = run_compartment_persistence_case(
        deepcopy(scenario),
        seed=3,
        horizon_steps=40,
        Nx=10,
        Ny=10,
    )
    assert result.ranking_allowed is False
    assert result.physical_binding == "OPEN"
    assert result.persistence_status in {
        STATUS_NEVER_LOCALIZED,
        STATUS_LOCALIZED_AT_HORIZON,
        STATUS_LOCALIZED_THEN_LOST,
    }
    assert 0.0 <= result.localized_fraction <= 1.0
    assert result.longest_consecutive_localized_steps <= result.localized_step_count


def test_matched_persistence_returns_profile_times_seed_rows() -> None:
    rows = run_matched_compartment_persistence(
        [deepcopy(SCENARIO_C), deepcopy(SCENARIO_D)],
        seeds=(5, 7),
        horizon_steps=20,
        Nx=8,
        Ny=8,
    )
    assert len(rows) == 4
    assert {row.profile for row in rows} == {
        "AMMONIA_CANDIDATE",
        "HYDROCARBON_CANDIDATE",
    }


def test_invalid_persistence_horizon_fails_closed() -> None:
    with pytest.raises(ValueError):
        run_compartment_persistence_case(
            deepcopy(SCENARIO_C),
            seed=1,
            horizon_steps=0,
            Nx=8,
            Ny=8,
        )


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_candidate_compartment_observation_has_stable_shared_contract(scenario) -> None:
    sim = create_exotic_candidate_simulator(
        deepcopy(scenario),
        Nx=8,
        Ny=8,
        seed=17,
    )
    sim.initialize()
    observation = sim.candidate_compartments()
    required = {
        "count",
        "raw_component_count",
        "area_pixels",
        "occupancy_fraction",
        "interface_edge_count",
        "global_saturation",
        "bounded_system_status",
    }
    assert required <= set(observation)
    assert 0.0 <= float(observation["occupancy_fraction"]) <= 1.0
    assert int(observation["count"]) >= 0
    assert int(observation["raw_component_count"]) >= int(observation["count"])



def test_noncontractible_information_stripe_is_rejected_even_with_complete_shell() -> None:
    info = np.zeros((7, 7), dtype=float)
    boundary = np.zeros((7, 7), dtype=float)

    # Information-rich stripe wraps around axis 0 on the torus.
    info[:, 3] = 1.0
    # Perfect two-sided shell around the stripe.
    boundary[:, 2] = 1.0
    boundary[:, 4] = 1.0

    observation = closed_boundary_compartment_observation(
        info,
        boundary,
        information_threshold=0.5,
        boundary_threshold=0.5,
    )
    assert observation["raw_information_component_count"] == 1
    assert observation["noncontractible_information_component_count"] == 1
    assert observation["contractible_information_component_count"] == 0
    assert observation["rejected_noncontractible_component_count"] == 1
    assert observation["count"] == 0
    assert observation["bounded_system_status"] == "NONCONTRACTIBLE_INTERIOR_ONLY"


def test_contractible_information_patch_with_complete_shell_is_accepted() -> None:
    info = np.zeros((7, 7), dtype=float)
    boundary = np.zeros((7, 7), dtype=float)

    info[3, 3] = 1.0
    boundary[2, 3] = 1.0
    boundary[4, 3] = 1.0
    boundary[3, 2] = 1.0
    boundary[3, 4] = 1.0

    observation = closed_boundary_compartment_observation(
        info,
        boundary,
        information_threshold=0.5,
        boundary_threshold=0.5,
    )
    assert observation["raw_information_component_count"] == 1
    assert observation["noncontractible_information_component_count"] == 0
    assert observation["contractible_information_component_count"] == 1
    assert observation["count"] == 1
    assert observation["bounded_system_status"] == "CLOSED_BOUNDARY_CANDIDATE"
