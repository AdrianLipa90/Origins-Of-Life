from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology.compartments import bounded_compartment_observation
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
