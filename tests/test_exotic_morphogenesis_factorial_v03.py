from __future__ import annotations

from copy import deepcopy

import pytest

from origins.exobiology.boundary_morphogenesis import (
    COLOCATED_BASELINE,
    EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
)
from origins.exobiology.information_morphogenesis import (
    DISTRIBUTED_INFORMATION_BASELINE,
    PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
)
from origins.exobiology.morphogenesis_factorial import (
    SCHEMA,
    factorial_manifest,
    run_morphogenesis_factorial_case,
    run_profile_factorial,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_factorial_manifest_freezes_parameters_thresholds_and_budgets() -> None:
    manifest = factorial_manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["design"] == "2x2_MATCHED_SEED_FACTORIAL"
    assert manifest["same_parameters_required"] is True
    assert manifest["same_thresholds_required"] is True
    assert manifest["source_budgets_preserved"] is True
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
@pytest.mark.parametrize(
    "information_mode",
    [DISTRIBUTED_INFORMATION_BASELINE, PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE],
)
@pytest.mark.parametrize(
    "boundary_mode",
    [COLOCATED_BASELINE, EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE],
)
def test_factorial_conditions_conserve_material_and_source_budgets(
    scenario,
    information_mode,
    boundary_mode,
) -> None:
    row = run_morphogenesis_factorial_case(
        deepcopy(scenario),
        information_mode=information_mode,
        boundary_mode=boundary_mode,
        seed=59,
        horizon_steps=120,
        Nx=10,
        Ny=10,
    )
    assert row.material_residual_abs < 1e-8
    assert row.information_source_budget_relative_error < 1e-9
    # Exterior boundary redistribution is budget-preserving once an interface
    # exists; short test horizons may still have no information gradient.
    if row.boundary_source_budget_relative_error != 1.0:
        assert row.boundary_source_budget_relative_error < 1e-9
    assert row.ranking_allowed is False


def test_profile_factorial_has_four_conditions_per_seed() -> None:
    rows = run_profile_factorial(
        deepcopy(SCENARIO_C),
        seeds=(3, 7),
        horizon_steps=20,
        Nx=8,
        Ny=8,
    )
    assert len(rows) == 4 * 2
    assert len({(r.information_mode, r.boundary_mode) for r in rows}) == 4
