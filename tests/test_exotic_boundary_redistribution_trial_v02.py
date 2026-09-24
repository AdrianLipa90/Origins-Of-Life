from __future__ import annotations

from copy import deepcopy

import pytest

from origins.exobiology.morphogenesis_redistribution_trial import (
    SCHEMA,
    run_boundary_redistribution_trial_case,
    run_matched_boundary_redistribution_trial,
    trial_manifest,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_redistribution_trial_manifest_freezes_parameters_thresholds_and_budget() -> None:
    manifest = trial_manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["matched_seed_required"] is True
    assert manifest["same_parameters_required"] is True
    assert manifest["same_thresholds_required"] is True
    assert manifest["source_budget_preserved_when_interface_exists"] is True
    assert manifest["only_varied_mechanism"] == "boundary_source_spatial_distribution"
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_redistribution_trial_preserves_material_and_source_budget(scenario) -> None:
    rows = run_boundary_redistribution_trial_case(
        deepcopy(scenario),
        seed=41,
        horizons=(100, 200),
        Nx=12,
        Ny=12,
    )
    for row in rows:
        assert row.baseline_material_residual_abs < 1e-8
        assert row.intervention_material_residual_abs < 1e-8
        assert row.boundary_source_budget_relative_error < 1e-9
        assert row.ranking_allowed is False


def test_matched_redistribution_trial_row_count() -> None:
    rows = run_matched_boundary_redistribution_trial(
        [deepcopy(SCENARIO_C), deepcopy(SCENARIO_D)],
        seeds=(3, 5),
        horizons=(20, 40),
        Nx=8,
        Ny=8,
    )
    assert len(rows) == 2 * 2 * 2
