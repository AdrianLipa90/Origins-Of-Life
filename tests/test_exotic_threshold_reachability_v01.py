from __future__ import annotations

from copy import deepcopy

import pytest

from origins.exobiology.reachability import (
    DIAGNOSTIC_SCHEMA,
    STATUS_NOT_REACHED,
    STATUS_REACHED,
    diagnostic_manifest,
    run_matched_threshold_reachability,
    run_threshold_reachability_case,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_reachability_manifest_forbids_threshold_tuning_and_ranking() -> None:
    manifest = diagnostic_manifest()
    assert manifest["schema"] == DIAGNOSTIC_SCHEMA
    assert manifest["ranking_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["missing_crossing_semantics"] == STATUS_NOT_REACHED
    assert manifest["physical_binding"] == "OPEN"


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_reachability_preserves_declared_thresholds_across_horizons(scenario) -> None:
    rows = run_threshold_reachability_case(
        deepcopy(scenario),
        seed=5,
        horizons=(20, 40, 80),
        Nx=10,
        Ny=10,
    )
    assert [row.horizon_steps for row in rows] == [20, 40, 80]
    assert len({row.information_threshold for row in rows}) == 1
    assert len({row.boundary_threshold for row in rows}) == 1
    assert all(row.ranking_allowed is False for row in rows)
    assert all(row.physical_binding == "OPEN" for row in rows)
    assert all(
        row.reachability_status in {STATUS_REACHED, STATUS_NOT_REACHED}
        for row in rows
    )


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_no_crossing_means_not_reached_within_horizon_not_impossible(scenario) -> None:
    rows = run_threshold_reachability_case(
        deepcopy(scenario),
        seed=7,
        horizons=(1,),
        Nx=8,
        Ny=8,
    )
    row = rows[0]
    if row.first_compartment_step is None:
        assert row.reachability_status == STATUS_NOT_REACHED


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_selection_effect_uses_matched_seed_control(scenario) -> None:
    rows = run_threshold_reachability_case(
        deepcopy(scenario),
        seed=13,
        horizons=(30, 60),
        Nx=10,
        Ny=10,
    )
    for row in rows:
        assert row.baseline_information_mass >= 0.0
        assert row.no_selection_information_mass >= 0.0
        assert row.selection_information_effect_fraction == pytest.approx(
            (
                row.baseline_information_mass - row.no_selection_information_mass
            ) / row.baseline_information_mass
            if row.baseline_information_mass > 0.0
            else 0.0
        )


def test_matched_reachability_row_count_is_profiles_times_seeds_times_horizons() -> None:
    rows = run_matched_threshold_reachability(
        [deepcopy(SCENARIO_C), deepcopy(SCENARIO_D)],
        seeds=(2, 3),
        horizons=(10, 20, 30),
        Nx=8,
        Ny=8,
    )
    assert len(rows) == 2 * 2 * 3
    assert {row.profile for row in rows} == {
        "AMMONIA_CANDIDATE",
        "HYDROCARBON_CANDIDATE",
    }


def test_invalid_horizons_fail_closed() -> None:
    with pytest.raises(ValueError):
        run_threshold_reachability_case(
            deepcopy(SCENARIO_C),
            seed=1,
            horizons=(0, 10),
            Nx=8,
            Ny=8,
        )


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_persistence_counters_are_internally_consistent(scenario) -> None:
    rows = run_threshold_reachability_case(
        deepcopy(scenario),
        seed=37,
        horizons=(25, 50, 100),
        Nx=10,
        Ny=10,
    )
    for row in rows:
        assert row.localized_steps >= 0
        assert row.max_consecutive_localized_steps >= 0
        assert row.localized_steps >= row.max_consecutive_localized_steps
        assert row.global_saturation_steps >= 0

        if row.first_compartment_step is None:
            assert row.last_compartment_step is None
            assert row.localized_steps == 0
            assert row.max_consecutive_localized_steps == 0
            assert row.lost_to_global_saturation is False
        else:
            assert row.last_compartment_step is not None
            assert row.last_compartment_step >= row.first_compartment_step
            assert row.localized_steps > 0

        if row.lost_to_global_saturation:
            assert row.first_compartment_step is not None
            assert row.first_global_saturation_step is not None
            assert row.first_global_saturation_step > row.first_compartment_step


def test_reachability_manifest_does_not_invent_a_persistence_cutoff() -> None:
    manifest = diagnostic_manifest()
    assert manifest["reachability_semantics"] == "EVER_LOCALIZED_BOUNDED_CANDIDATE"
    assert manifest["final_bounded_state_reported_separately"] is True
    assert manifest["persistence_threshold"] is None
