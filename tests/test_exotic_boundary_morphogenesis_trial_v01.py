from __future__ import annotations

from copy import deepcopy
import inspect

import pytest

from origins.exobiology.boundary_morphogenesis import (
    COLOCATED_BASELINE,
    EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
    exterior_information_interface_gate,
)
from origins.exobiology.morphogenesis_trial import (
    MODES,
    SCHEMA,
    run_boundary_morphogenesis_trial_case,
    run_matched_boundary_morphogenesis_trial,
    trial_manifest,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_trial_manifest_freezes_everything_except_morphogenesis_mode() -> None:
    manifest = trial_manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["modes"] == list(MODES)
    assert manifest["parameters_frozen"] is True
    assert manifest["thresholds_frozen"] is True
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False
    assert manifest["only_varied_mechanism"] == "boundary_morphogenesis_mode"


def test_gradient_gate_does_not_receive_detection_thresholds() -> None:
    signature = inspect.signature(exterior_information_interface_gate)
    assert tuple(signature.parameters) == ("information",)


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_baseline_and_candidate_start_from_same_material_under_matched_seed(scenario) -> None:
    base = run_boundary_morphogenesis_trial_case(
        deepcopy(scenario),
        mode=COLOCATED_BASELINE,
        seed=17,
        horizons=(1,),
        Nx=10,
        Ny=10,
    )[0]
    candidate = run_boundary_morphogenesis_trial_case(
        deepcopy(scenario),
        mode=EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
        seed=17,
        horizons=(1,),
        Nx=10,
        Ny=10,
    )[0]
    # No parameter/threshold adjustment is allowed; only the boundary source
    # geometry differs. Both must still conserve their declared material.
    assert base.material_residual_abs < 1e-8
    assert candidate.material_residual_abs < 1e-8


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_both_modes_conserve_material_across_trial(scenario) -> None:
    for mode in MODES:
        rows = run_boundary_morphogenesis_trial_case(
            deepcopy(scenario),
            mode=mode,
            seed=23,
            horizons=(40, 100),
            Nx=12,
            Ny=12,
        )
        assert all(row.material_residual_abs < 1e-8 for row in rows)
        assert all(row.parameter_tuning_allowed is False for row in rows)
        assert all(row.threshold_tuning_allowed is False for row in rows)
        assert all(row.ranking_allowed is False for row in rows)


def test_matched_trial_row_count_is_profiles_modes_seeds_horizons() -> None:
    rows = run_matched_boundary_morphogenesis_trial(
        [deepcopy(SCENARIO_C), deepcopy(SCENARIO_D)],
        seeds=(3, 5),
        horizons=(20, 40),
        Nx=8,
        Ny=8,
    )
    assert len(rows) == 2 * 2 * 2 * 2
    assert {row.mode for row in rows} == {
        COLOCATED_BASELINE,
        EXTERIOR_GRADIENT_BOUNDARY_CANDIDATE,
    }


def test_unknown_trial_mode_fails_closed() -> None:
    with pytest.raises(ValueError):
        run_boundary_morphogenesis_trial_case(
            deepcopy(SCENARIO_C),
            mode="MAGIC",
            seed=1,
            horizons=(10,),
            Nx=8,
            Ny=8,
        )
