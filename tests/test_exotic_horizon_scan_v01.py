from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology.horizon_scan import (
    DEFAULT_HORIZONS,
    HORIZON_SCAN_SCHEMA,
    horizon_scan_manifest,
    run_exotic_horizon_scan_case,
    run_matched_exotic_horizon_scan,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_horizon_manifest_freezes_everything_except_time() -> None:
    manifest = horizon_scan_manifest()
    assert manifest["schema"] == HORIZON_SCAN_SCHEMA
    assert manifest["varied_dimension"] == "horizon_steps_only"
    assert manifest["default_horizons"] == list(DEFAULT_HORIZONS)
    assert manifest["ranking_allowed"] is False


def test_matched_horizon_scan_has_two_profiles_three_seeds_three_horizons() -> None:
    rows = run_matched_exotic_horizon_scan(
        [deepcopy(SCENARIO_C), deepcopy(SCENARIO_D)],
        seeds=(11, 23, 47),
        horizons=(30, 60, 90),
        Nx=10,
        Ny=10,
    )
    assert len(rows) == 18
    assert {r.profile for r in rows} == {
        "AMMONIA_CANDIDATE",
        "HYDROCARBON_CANDIDATE",
    }
    assert all(r.thresholds_frozen for r in rows)
    assert all(r.parameters_frozen for r in rows)
    assert all(r.ranking_allowed is False for r in rows)


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_horizon_scan_preserves_material_and_threshold_logic(scenario) -> None:
    rows = run_exotic_horizon_scan_case(
        deepcopy(scenario),
        seed=13,
        horizons=(40, 100, 180),
        Nx=12,
        Ny=12,
    )
    assert [r.horizon_steps for r in rows] == [40, 100, 180]
    for row in rows:
        assert row.material_residual_abs <= 2e-9 * max(
            1.0,
            row.information_mass + row.boundary_mass,
        )
        assert np.isfinite(row.joint_threshold_score_max)
        assert row.joint_threshold_score_max >= 0.0
        if row.joint_threshold_score_max < 1.0:
            assert row.candidate_compartment_count == 0


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_horizon_scan_is_deterministic_for_fixed_seed(scenario) -> None:
    a = run_exotic_horizon_scan_case(
        deepcopy(scenario),
        seed=29,
        horizons=(50, 120),
        Nx=10,
        Ny=10,
    )
    b = run_exotic_horizon_scan_case(
        deepcopy(scenario),
        seed=29,
        horizons=(50, 120),
        Nx=10,
        Ny=10,
    )
    assert a == b


def test_horizons_must_be_strictly_increasing_and_unique() -> None:
    with pytest.raises(ValueError):
        run_exotic_horizon_scan_case(
            deepcopy(SCENARIO_C),
            seed=1,
            horizons=(100, 50),
            Nx=8,
            Ny=8,
        )
    with pytest.raises(ValueError):
        run_exotic_horizon_scan_case(
            deepcopy(SCENARIO_C),
            seed=1,
            horizons=(100, 100),
            Nx=8,
            Ny=8,
        )
