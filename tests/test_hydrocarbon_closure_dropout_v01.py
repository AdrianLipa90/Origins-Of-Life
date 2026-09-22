from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology.compartments import periodic_component_topology

from origins.exobiology.hydrocarbon_dropout import (
    SCHEMA,
    DROPOUT_CONTRACTIBLE_INTERIOR_LOSS,
    DROPOUT_CONTRACTIBLE_SHELL_GAP,
    DROPOUT_OTHER,
    DROPOUT_PERSISTS,
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
    assert 0.0 <= row.final_snapshot["largest_information_component_fraction"] <= 1.0
    assert row.final_snapshot["information_noncontractible_component_count"] >= 0
    assert row.final_snapshot["contractible_information_component_count"] >= 0
    assert row.final_snapshot["best_contractible_missing_shell_pixels"] >= 0
    assert row.final_snapshot["best_contractible_shell_pixels"] >= 0
    assert 0.0 <= row.final_snapshot["best_contractible_shell_coverage"] <= 1.0
    assert row.dropout_mechanism in {
        DROPOUT_PERSISTS,
        DROPOUT_CONTRACTIBLE_INTERIOR_LOSS,
        DROPOUT_CONTRACTIBLE_SHELL_GAP,
        DROPOUT_OTHER,
    }
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



def test_periodic_topology_detects_contractible_patch() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    topology = periodic_component_topology(mask)
    assert topology["component_count"] == 1
    assert topology["any_noncontractible"] is False
    assert topology["largest_component_fraction"] == pytest.approx(9 / 49)


def test_periodic_topology_detects_noncontractible_x_winding() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[:, 3] = True
    topology = periodic_component_topology(mask)
    assert topology["component_count"] == 1
    assert topology["any_noncontractible"] is True
    assert topology["wraps_x_component_count"] == 1
    assert topology["wraps_y_component_count"] == 0


def test_periodic_topology_detects_noncontractible_y_winding() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, :] = True
    topology = periodic_component_topology(mask)
    assert topology["component_count"] == 1
    assert topology["any_noncontractible"] is True
    assert topology["wraps_x_component_count"] == 0
    assert topology["wraps_y_component_count"] == 1



def test_dropout_manifest_declares_mechanism_taxonomy() -> None:
    manifest = dropout_manifest()
    assert set(manifest["dropout_mechanisms"]) == {
        DROPOUT_PERSISTS,
        DROPOUT_CONTRACTIBLE_INTERIOR_LOSS,
        DROPOUT_CONTRACTIBLE_SHELL_GAP,
        DROPOUT_OTHER,
    }
