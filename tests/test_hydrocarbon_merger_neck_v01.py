from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology.hydrocarbon_merger_neck import (
    EVENT_MULTI_PIXEL_NECK,
    EVENT_NO_MERGER_WITHIN_HORIZON,
    EVENT_SINGLE_PIXEL_NECK,
    diagnose_merger_neck,
    diagnostic_manifest,
    minimum_new_active_bridge_cost,
    minimum_periodic_set_distance,
    run_hydrocarbon_merger_neck_case,
)
from origins.scenarios import SCENARIO_D


def _single_pixel_neck_masks():
    previous = np.zeros((7, 7), dtype=bool)
    # non-contractible vertical background stripe
    previous[:, 4] = True
    # one-pixel contractible island
    previous[3, 2] = True

    current = previous.copy()
    current[3, 3] = True
    return previous, current, {(3, 2)}, {(x, 4) for x in range(7)}


def test_single_pixel_neck_cost_is_one() -> None:
    previous, current, island, background = _single_pixel_neck_masks()
    diagnosis = diagnose_merger_neck(
        previous,
        current,
        island,
        background,
    )
    assert diagnosis["event"] == EVENT_SINGLE_PIXEL_NECK
    assert diagnosis["pre_transition_min_periodic_manhattan_distance"] == 2
    assert diagnosis["pre_transition_inactive_gap_pixels"] == 1
    assert diagnosis["minimum_new_active_bridge_pixels"] == 1
    assert diagnosis["current_descendant_noncontractible"] is True


def test_two_pixel_neck_cost_is_two() -> None:
    previous = np.zeros((9, 9), dtype=bool)
    previous[:, 6] = True
    previous[4, 3] = True

    current = previous.copy()
    current[4, 4] = True
    current[4, 5] = True

    diagnosis = diagnose_merger_neck(
        previous,
        current,
        {(4, 3)},
        {(x, 6) for x in range(9)},
    )
    assert diagnosis["event"] == EVENT_MULTI_PIXEL_NECK
    assert diagnosis["pre_transition_min_periodic_manhattan_distance"] == 3
    assert diagnosis["pre_transition_inactive_gap_pixels"] == 2
    assert diagnosis["minimum_new_active_bridge_pixels"] == 2


def test_no_current_path_returns_none() -> None:
    previous, _, island, background = _single_pixel_neck_masks()
    cost = minimum_new_active_bridge_cost(
        previous,
        previous,
        island,
        background,
    )
    assert cost is None


def test_periodic_set_distance_uses_torus_geometry() -> None:
    distance = minimum_periodic_set_distance(
        {(0, 0)},
        {(6, 0)},
        (7, 7),
    )
    assert distance == 1


def test_merger_neck_manifest_is_diagnostic_only() -> None:
    manifest = diagnostic_manifest()
    assert manifest["diagnostic_only"] is True
    assert manifest["model_mutation_allowed"] is False
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False
    assert manifest["physical_binding"] == "OPEN"


def test_short_horizon_without_dropout_is_reported_without_invention() -> None:
    row = run_hydrocarbon_merger_neck_case(
        deepcopy(SCENARIO_D),
        seed=47,
        horizon_steps=20,
        Nx=8,
        Ny=8,
    )
    assert row.event == EVENT_NO_MERGER_WITHIN_HORIZON
    assert row.minimum_new_active_bridge_pixels is None
    assert row.model_mutation_allowed is False
    assert row.parameter_tuning_allowed is False
    assert row.threshold_tuning_allowed is False
    assert row.ranking_allowed is False
