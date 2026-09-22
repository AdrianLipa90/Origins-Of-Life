from __future__ import annotations

from copy import deepcopy

import pytest

from origins.exobiology.boundary_morphogenesis import (
    COLOCATED_BASELINE,
    EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
)
from origins.exobiology.morphogenesis_intervention import (
    SCHEMA,
    intervention_manifest,
    run_boundary_morphogenesis_intervention_case,
    run_matched_boundary_morphogenesis_intervention,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_intervention_manifest_freezes_only_structural_operator_change() -> None:
    manifest = intervention_manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["baseline_mode"] == COLOCATED_BASELINE
    assert manifest["intervention_mode"] == EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE
    assert manifest["matched_seed_required"] is True
    assert manifest["same_parameters_required"] is True
    assert manifest["same_thresholds_required"] is True
    assert manifest["ranking_allowed"] is False
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_matched_intervention_reports_effect_without_requiring_improvement(scenario) -> None:
    result = run_boundary_morphogenesis_intervention_case(
        deepcopy(scenario),
        seed=7,
        horizon_steps=40,
        Nx=10,
        Ny=10,
    )
    assert result.baseline_mode == COLOCATED_BASELINE
    assert result.intervention_mode == EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE
    assert result.delta_localized_fraction == pytest.approx(
        result.intervention_localized_fraction - result.baseline_localized_fraction
    )
    assert result.delta_longest_localized_run == (
        result.intervention_longest_localized_run
        - result.baseline_longest_localized_run
    )
    assert result.interpretation == "MODEL_OPERATOR_EFFECT_ONLY"
    assert result.ranking_allowed is False
    assert result.physical_binding == "OPEN"


def test_matched_intervention_row_count_is_profile_times_seed() -> None:
    rows = run_matched_boundary_morphogenesis_intervention(
        [deepcopy(SCENARIO_C), deepcopy(SCENARIO_D)],
        seeds=(3, 5),
        horizon_steps=20,
        Nx=8,
        Ny=8,
    )
    assert len(rows) == 4
    assert {row.profile for row in rows} == {
        "AMMONIA_CANDIDATE",
        "HYDROCARBON_CANDIDATE",
    }


def test_invalid_intervention_horizon_fails_loud_through_persistence_gate() -> None:
    with pytest.raises(ValueError):
        run_boundary_morphogenesis_intervention_case(
            deepcopy(SCENARIO_C),
            seed=1,
            horizon_steps=0,
            Nx=8,
            Ny=8,
        )
