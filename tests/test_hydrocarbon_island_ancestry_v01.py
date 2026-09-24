from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology.hydrocarbon_island_ancestry import (
    EVENT_EROSION,
    EVENT_MERGER,
    EVENT_SPLIT,
    EVENT_SURVIVES,
    SCHEMA,
    ancestry_manifest,
    classify_contractible_ancestry_transition,
    run_hydrocarbon_island_ancestry_case,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def _prior_singleton(shape=(7, 7), site=(3, 3)):
    labels = np.zeros(shape, dtype=int)
    accepted = np.zeros(shape, dtype=bool)
    labels[site] = 1
    accepted[site] = True
    return labels, accepted


def test_ancestry_classifies_threshold_erosion() -> None:
    labels, accepted = _prior_singleton()
    current = np.zeros((7, 7), dtype=bool)
    event, transitions, _ = classify_contractible_ancestry_transition(
        labels, accepted, current
    )
    assert event == EVENT_EROSION
    assert transitions[0].retained_pixels == 0


def test_ancestry_classifies_contractible_survival() -> None:
    labels, accepted = _prior_singleton()
    current = np.zeros((7, 7), dtype=bool)
    current[3, 3] = True
    event, transitions, counts = classify_contractible_ancestry_transition(
        labels, accepted, current
    )
    assert event == EVENT_SURVIVES
    assert transitions[0].retained_fraction == pytest.approx(1.0)
    assert counts["contractible_component_count"] == 1


def test_ancestry_classifies_merger_into_noncontractible_background() -> None:
    labels, accepted = _prior_singleton(site=(0, 3))
    current = np.zeros((7, 7), dtype=bool)
    current[:, 3] = True
    event, transitions, counts = classify_contractible_ancestry_transition(
        labels, accepted, current
    )
    assert event == EVENT_MERGER
    assert transitions[0].noncontractible_descendant_count == 1
    assert counts["noncontractible_component_count"] == 1


def test_ancestry_classifies_split_for_multi_pixel_prior_component() -> None:
    labels = np.zeros((7, 7), dtype=int)
    accepted = np.zeros((7, 7), dtype=bool)
    labels[3, 2:5] = 1
    accepted[3, 2:5] = True

    current = np.zeros((7, 7), dtype=bool)
    current[3, 2] = True
    current[3, 4] = True

    event, transitions, counts = classify_contractible_ancestry_transition(
        labels, accepted, current
    )
    assert event == EVENT_SPLIT
    assert transitions[0].descendant_count == 2
    assert counts["contractible_component_count"] == 2


def test_ancestry_manifest_is_diagnostic_only() -> None:
    manifest = ancestry_manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["diagnostic_only"] is True
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False


def test_ancestry_rejects_non_hydrocarbon_profile() -> None:
    with pytest.raises(ValueError):
        run_hydrocarbon_island_ancestry_case(
            deepcopy(SCENARIO_C),
            seed=1,
            horizon_steps=10,
            Nx=8,
            Ny=8,
        )


def test_short_horizon_returns_no_transition_if_no_closure_loss_seen() -> None:
    row = run_hydrocarbon_island_ancestry_case(
        deepcopy(SCENARIO_D),
        seed=11,
        horizon_steps=40,
        Nx=10,
        Ny=10,
    )
    assert row.overall_transition is None
    assert row.first_loss_step is None
