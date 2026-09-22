from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology import (
    AmmoniaCandidateSimulator,
    HydrocarbonCandidateSimulator,
)
from origins.exobiology.morphogenesis import (
    DIAGNOSTIC_SCHEMA,
    diagnostic_manifest,
    run_matched_morphogenesis_source_overlap,
    run_morphogenesis_source_overlap_case,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_ammonia_initial_assembly_sources_are_spatially_proportional() -> None:
    sim = AmmoniaCandidateSimulator(deepcopy(SCENARIO_C), Nx=12, Ny=12, seed=3)
    sim.initialize()
    sources = sim.candidate_assembly_sources()
    ratio = (
        sim.parameters.information_assembly_rate
        / sim.parameters.boundary_assembly_rate
    )
    assert np.allclose(sources["information"], ratio * sources["boundary"])


def test_hydrocarbon_interface_breaks_exact_initial_source_proportionality_after_capture() -> None:
    sim = HydrocarbonCandidateSimulator(deepcopy(SCENARIO_D), Nx=12, Ny=12, seed=3)
    sim.initialize()
    sim.step_interface_exchange()
    sources = sim.candidate_assembly_sources()
    info = sources["information"]
    boundary = sources["boundary"]
    assert float(np.sum(info)) > 0.0
    assert float(np.sum(boundary)) > 0.0
    ratio = np.divide(
        boundary,
        info,
        out=np.zeros_like(boundary),
        where=info > 0.0,
    )
    active = info > 0.0
    assert float(np.std(ratio[active])) > 0.0


def test_morphogenesis_manifest_forbids_tuning_and_ranking() -> None:
    manifest = diagnostic_manifest()
    assert manifest["schema"] == DIAGNOSTIC_SCHEMA
    assert manifest["ranking_allowed"] is False
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["physical_binding"] == "OPEN"


@pytest.mark.parametrize("scenario", [SCENARIO_C, SCENARIO_D])
def test_overlap_metrics_are_bounded(scenario) -> None:
    rows = run_morphogenesis_source_overlap_case(
        deepcopy(scenario),
        seed=5,
        horizons=(20, 40),
        Nx=10,
        Ny=10,
    )
    assert len(rows) == 2
    for row in rows:
        for value in (
            row.source_distribution_overlap,
            row.source_cosine_similarity,
            row.state_distribution_overlap,
            row.state_cosine_similarity,
            row.boundary_source_information_gradient_cosine,
        ):
            assert 0.0 <= value <= 1.0 + 1e-12
        assert row.ranking_allowed is False
        assert row.parameter_tuning_allowed is False
        assert row.threshold_tuning_allowed is False


def test_matched_source_overlap_has_profile_seed_horizon_product() -> None:
    rows = run_matched_morphogenesis_source_overlap(
        [deepcopy(SCENARIO_C), deepcopy(SCENARIO_D)],
        seeds=(2, 7),
        horizons=(10, 20, 30),
        Nx=8,
        Ny=8,
    )
    assert len(rows) == 2 * 2 * 3
    assert {row.profile for row in rows} == {
        "AMMONIA_CANDIDATE",
        "HYDROCARBON_CANDIDATE",
    }


def test_invalid_overlap_horizon_fails_closed() -> None:
    with pytest.raises(ValueError):
        run_morphogenesis_source_overlap_case(
            deepcopy(SCENARIO_C),
            seed=1,
            horizons=(0, 10),
            Nx=8,
            Ny=8,
        )
