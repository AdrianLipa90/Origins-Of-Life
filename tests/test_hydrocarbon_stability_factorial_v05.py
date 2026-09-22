from __future__ import annotations

from copy import deepcopy

import pytest

from origins.exobiology.hydrocarbon_stability_factorial import (
    ISLAND_MODES,
    SCHEMA,
    SHELL_MODES,
    factorial_manifest,
    run_hydrocarbon_stability_factorial,
    run_hydrocarbon_stability_factorial_case,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_stability_factorial_manifest_is_causal_split_only() -> None:
    manifest = factorial_manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["profile"] == "HYDROCARBON_CANDIDATE"
    assert manifest["design"] == "2x2_MATCHED_SEED_CAUSAL_SPLIT"
    assert manifest["same_parameters_required"] is True
    assert manifest["same_thresholds_required"] is True
    assert manifest["stability_source_budgets_preserved"] is True
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False


def test_stability_factorial_small_grid_has_four_conditions_per_seed() -> None:
    rows = run_hydrocarbon_stability_factorial(
        deepcopy(SCENARIO_D),
        seeds=(3, 5),
        horizon_steps=40,
        Nx=8,
        Ny=8,
    )
    assert len(rows) == len(SHELL_MODES) * len(ISLAND_MODES) * 2
    assert {row.shell_maintenance for row in rows} == set(SHELL_MODES)
    assert {row.island_preservation for row in rows} == set(ISLAND_MODES)


@pytest.mark.parametrize("shell_mode", SHELL_MODES)
@pytest.mark.parametrize("island_mode", ISLAND_MODES)
def test_stability_factorial_preserves_material_and_source_budgets(
    shell_mode,
    island_mode,
) -> None:
    row = run_hydrocarbon_stability_factorial_case(
        deepcopy(SCENARIO_D),
        seed=7,
        shell_maintenance=shell_mode,
        island_preservation=island_mode,
        horizon_steps=80,
        Nx=10,
        Ny=10,
    )
    assert row.material_residual_abs < 1e-8
    assert row.max_information_stability_budget_relative_error < 1e-9
    assert row.max_boundary_stability_budget_relative_error < 1e-9
    assert row.parameter_tuning_allowed is False
    assert row.threshold_tuning_allowed is False
    assert row.ranking_allowed is False


def test_stability_factorial_rejects_non_hydrocarbon_profile() -> None:
    with pytest.raises(ValueError):
        run_hydrocarbon_stability_factorial_case(
            deepcopy(SCENARIO_C),
            seed=1,
            shell_maintenance=SHELL_MODES[0],
            island_preservation=ISLAND_MODES[0],
            horizon_steps=10,
            Nx=8,
            Ny=8,
        )
