from __future__ import annotations

from copy import deepcopy

import pytest

from origins.exobiology import AmmoniaCandidateSimulator, HydrocarbonCandidateSimulator
from origins.exobiology.source_geometry import (
    SCHEMA,
    diagnose_source_geometry,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D


def test_ammonia_initial_sources_are_exactly_colocated_up_to_floating_point() -> None:
    sim = AmmoniaCandidateSimulator(deepcopy(SCENARIO_C), Nx=16, Ny=16, seed=5)
    sim.initialize()
    result = diagnose_source_geometry(sim)

    assert result.schema == SCHEMA
    assert result.exact_initial_proportionality_expected is True
    assert result.proportionality_residual_relative is not None
    assert result.proportionality_residual_relative < 1e-14
    assert result.cosine_overlap == pytest.approx(1.0, abs=1e-14)
    assert result.pearson_correlation == pytest.approx(1.0, abs=1e-14)
    assert result.interpretation == "INITIAL_SOURCE_COLOCATION_EXACT_UP_TO_FLOATING_POINT"


def test_hydrocarbon_source_geometry_is_measured_without_assuming_proportionality() -> None:
    sim = HydrocarbonCandidateSimulator(deepcopy(SCENARIO_D), Nx=16, Ny=16, seed=7)
    sim.initialize()
    # Populate A explicitly so the instantaneous source geometry is nonzero.
    sim.step_interface_exchange()
    result = diagnose_source_geometry(sim)

    assert result.schema == SCHEMA
    assert result.exact_initial_proportionality_expected is False
    assert result.proportionality_residual_relative is None
    assert -1.0 <= result.pearson_correlation <= 1.0
    assert 0.0 <= result.cosine_overlap <= 1.0
    assert result.physical_binding == "OPEN"


def test_ammonia_colocation_breaks_only_after_state_departs_from_zero_trait() -> None:
    sim = AmmoniaCandidateSimulator(deepcopy(SCENARIO_C), Nx=12, Ny=12, seed=11)
    sim.initialize()
    sim.run(steps=100)
    result = diagnose_source_geometry(sim)
    assert result.exact_initial_proportionality_expected is False
    assert result.proportionality_residual_relative is None
