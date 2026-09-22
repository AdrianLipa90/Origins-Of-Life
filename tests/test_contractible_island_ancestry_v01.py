from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology.contractible_island_ancestry import (
    EVENT_INTERIOR_SURVIVES_CLOSURE_LOST,
    EVENT_MERGER_WITH_NONCONTRACTIBLE,
    LINEAGE_CONTRACTIBLE_SURVIVAL,
    LINEAGE_MERGER_WITH_NONCONTRACTIBLE,
    LINEAGE_SPLIT,
    LINEAGE_THRESHOLD_EROSION,
    accepted_component_labels,
    ancestry_manifest,
    classify_component_transition,
    run_contractible_island_ancestry_case,
    summarize_ancestry_event,
)
from origins.exobiology.compartments import closed_boundary_compartment_observation
from origins.scenarios import SCENARIO_C, SCENARIO_D


def _label_at(mask: np.ndarray, position: tuple[int, int]) -> int:
    obs = closed_boundary_compartment_observation(
        mask.astype(float),
        np.ones_like(mask, dtype=float),
        information_threshold=0.5,
        boundary_threshold=0.5,
    )
    return int(obs["labels"][position])


def test_contractible_component_survival_is_detected() -> None:
    previous = np.zeros((7, 7), dtype=bool)
    previous[3, 3] = True
    current = previous.copy()
    label = _label_at(previous, (3, 3))

    transitions = classify_component_transition(previous, (label,), current)
    assert len(transitions) == 1
    assert transitions[0].classification == LINEAGE_CONTRACTIBLE_SURVIVAL
    assert summarize_ancestry_event(transitions) == EVENT_INTERIOR_SURVIVES_CLOSURE_LOST


def test_merger_with_existing_noncontractible_background_is_detected() -> None:
    previous = np.zeros((7, 7), dtype=bool)
    previous[:, 0] = True
    previous[3, 3] = True
    island_label = _label_at(previous, (3, 3))

    current = previous.copy()
    current[3, 1:3] = True

    transitions = classify_component_transition(
        previous,
        (island_label,),
        current,
    )
    assert transitions[0].classification == LINEAGE_MERGER_WITH_NONCONTRACTIBLE
    assert summarize_ancestry_event(transitions) == EVENT_MERGER_WITH_NONCONTRACTIBLE


def test_component_split_is_detected() -> None:
    previous = np.zeros((7, 7), dtype=bool)
    previous[3, 2:5] = True
    label = _label_at(previous, (3, 3))
    current = np.zeros((7, 7), dtype=bool)
    current[3, 2] = True
    current[3, 4] = True

    transitions = classify_component_transition(previous, (label,), current)
    assert transitions[0].classification == LINEAGE_SPLIT


def test_threshold_erosion_or_displacement_is_detected() -> None:
    previous = np.zeros((7, 7), dtype=bool)
    previous[3, 3] = True
    label = _label_at(previous, (3, 3))
    current = np.zeros((7, 7), dtype=bool)

    transitions = classify_component_transition(previous, (label,), current)
    assert transitions[0].classification == LINEAGE_THRESHOLD_EROSION


def test_accepted_component_labels_reads_only_accepted_closed_interiors() -> None:
    information = np.zeros((7, 7), dtype=float)
    boundary = np.zeros((7, 7), dtype=float)
    information[3, 3] = 1.0
    boundary[2, 3] = 1.0
    boundary[4, 3] = 1.0
    boundary[3, 2] = 1.0
    boundary[3, 4] = 1.0

    observation = closed_boundary_compartment_observation(
        information,
        boundary,
        information_threshold=0.5,
        boundary_threshold=0.5,
    )
    labels = accepted_component_labels(observation)
    assert len(labels) == 1
    assert observation["count"] == 1


def test_ancestry_manifest_is_diagnostic_only() -> None:
    manifest = ancestry_manifest()
    assert manifest["model_mutation_allowed"] is False
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False
    assert manifest["physical_binding"] == "OPEN"


def test_ancestry_trace_rejects_non_hydrocarbon_profile() -> None:
    with pytest.raises(ValueError):
        run_contractible_island_ancestry_case(
            deepcopy(SCENARIO_C),
            seed=1,
            horizon_steps=10,
            Nx=8,
            Ny=8,
        )
