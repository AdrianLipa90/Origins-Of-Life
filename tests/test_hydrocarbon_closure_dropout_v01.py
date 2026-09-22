from __future__ import annotations

from copy import deepcopy

import pytest

from origins.exobiology.hydrocarbon_dropout import (
    SCHEMA,
    dropout_manifest,
    run_hydrocarbon_closure_dropout_case,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_dropout_manifest_is_diagnostic_only() -> None:
    manifest = dropout_manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["profile"] == "HYDROCARBON_CANDIDATE"
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False
    assert manifest["physical_binding"] == "OPEN"


def test_dropout_result_has_consistent_event_snapshots() -> None:
    row = run_hydrocarbon_closure_dropout_case(
        deepcopy(SCENARIO_D),
        seed=5,
        horizon_steps=80,
        Nx=10,
        Ny=10,
    )
    assert row.longest_closed_run <= row.closed_step_count
    assert row.reclosure_count <= row.closure_run_count
    assert row.final_snapshot["step"] == 80
    assert 0.0 <= row.final_snapshot["information_threshold_fraction"] <= 1.0
    assert 0.0 <= row.final_snapshot["boundary_threshold_fraction"] <= 1.0
    if row.first_closed_step is None:
        assert row.first_closed_snapshot is None
        assert row.last_closed_step is None


def test_dropout_rejects_non_hydrocarbon_profile() -> None:
    with pytest.raises(ValueError):
        run_hydrocarbon_closure_dropout_case(
            deepcopy(SCENARIO_C),
            seed=1,
            horizon_steps=10,
            Nx=8,
            Ny=8,
        )
