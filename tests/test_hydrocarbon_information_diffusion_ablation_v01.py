from __future__ import annotations

from copy import deepcopy

import pytest

from origins.exobiology.hydrocarbon_information_diffusion_ablation import (
    DIFFUSION_BASELINE,
    DIFFUSION_OFF,
    ablation_manifest,
    run_information_diffusion_ablation,
    run_information_diffusion_condition,
)
from origins.scenarios import SCENARIO_D


def test_diffusion_ablation_manifest_is_single_operator_causal_split() -> None:
    manifest = ablation_manifest()
    assert manifest["only_varied_operator"] == "information_diffusion"
    assert manifest["same_seed_required"] is True
    assert manifest["same_thresholds_required"] is True
    assert manifest["same_other_parameters_required"] is True
    assert manifest["causal_ablation"] is True
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False


@pytest.mark.parametrize("condition", [DIFFUSION_BASELINE, DIFFUSION_OFF])
def test_diffusion_ablation_conserves_material(condition) -> None:
    row = run_information_diffusion_condition(
        deepcopy(SCENARIO_D),
        seed=17,
        diffusion_condition=condition,
        horizon_steps=120,
        Nx=12,
        Ny=12,
    )
    assert row.material_residual_abs < 1e-8
    assert row.causal_ablation is True
    assert row.ranking_allowed is False
    assert row.physical_binding == "OPEN"


def test_diffusion_off_is_exactly_zero_and_baseline_is_positive() -> None:
    baseline = run_information_diffusion_condition(
        deepcopy(SCENARIO_D),
        seed=19,
        diffusion_condition=DIFFUSION_BASELINE,
        horizon_steps=20,
        Nx=8,
        Ny=8,
    )
    off = run_information_diffusion_condition(
        deepcopy(SCENARIO_D),
        seed=19,
        diffusion_condition=DIFFUSION_OFF,
        horizon_steps=20,
        Nx=8,
        Ny=8,
    )
    assert baseline.information_diffusion > 0.0
    assert off.information_diffusion == pytest.approx(0.0, abs=0.0)


def test_matched_ablation_returns_two_conditions_per_seed() -> None:
    rows = run_information_diffusion_ablation(
        deepcopy(SCENARIO_D),
        seeds=(3, 5),
        horizon_steps=30,
        Nx=8,
        Ny=8,
    )
    assert len(rows) == 4
    assert {row.diffusion_condition for row in rows} == {
        DIFFUSION_BASELINE,
        DIFFUSION_OFF,
    }
    assert {row.seed for row in rows} == {3, 5}


def test_unknown_diffusion_condition_fails_closed() -> None:
    with pytest.raises(ValueError):
        run_information_diffusion_condition(
            deepcopy(SCENARIO_D),
            seed=1,
            diffusion_condition="MAGIC_DIFFUSION",
            horizon_steps=10,
            Nx=8,
            Ny=8,
        )
