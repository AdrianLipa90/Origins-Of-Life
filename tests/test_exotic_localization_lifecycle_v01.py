from __future__ import annotations

from copy import deepcopy

import pytest

from origins.exobiology.boundary_morphogenesis import (
    COLOCATED_BASELINE,
    EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
)
from origins.exobiology.localization_lifecycle import (
    SCHEMA,
    lifecycle_manifest,
    run_localization_lifecycle_case,
    run_matched_localization_lifecycle,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_lifecycle_manifest_forbids_tuning_and_ranking() -> None:
    manifest = lifecycle_manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False
    assert manifest["physical_binding"] == "OPEN"


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
@pytest.mark.parametrize(
    "mode",
    [COLOCATED_BASELINE, EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE],
)
def test_lifecycle_fractions_and_event_order_are_well_formed(scenario, mode) -> None:
    row = run_localization_lifecycle_case(
        deepcopy(scenario),
        mode=mode,
        seed=7,
        horizon_steps=60,
        Nx=10,
        Ny=10,
    )
    for value in (
        row.max_information_threshold_fraction,
        row.max_boundary_threshold_fraction,
        row.final_information_threshold_fraction,
        row.final_boundary_threshold_fraction,
    ):
        assert 0.0 <= value <= 1.0
    assert row.longest_closed_boundary_run <= row.closed_boundary_steps
    if row.first_closed_boundary_step is not None:
        assert row.last_closed_boundary_step is not None
        assert row.first_closed_boundary_step <= row.last_closed_boundary_step
    assert row.parameter_tuning_allowed is False
    assert row.threshold_tuning_allowed is False
    assert row.ranking_allowed is False


def test_matched_lifecycle_row_count() -> None:
    rows = run_matched_localization_lifecycle(
        [deepcopy(SCENARIO_C), deepcopy(SCENARIO_D)],
        seeds=(3, 5),
        horizon_steps=20,
        Nx=8,
        Ny=8,
    )
    assert len(rows) == 2 * 2 * 2


def test_invalid_lifecycle_horizon_fails_closed() -> None:
    with pytest.raises(ValueError):
        run_localization_lifecycle_case(
            deepcopy(SCENARIO_C),
            mode=COLOCATED_BASELINE,
            seed=1,
            horizon_steps=0,
            Nx=8,
            Ny=8,
        )
