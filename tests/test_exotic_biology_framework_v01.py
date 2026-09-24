from __future__ import annotations

from copy import deepcopy

import pytest

from origins.exobiology import (
    AMMONIA_CANDIDATE,
    HYDROCARBON_CANDIDATE,
    LIFE_RELATIONAL_INVARIANTS,
    WATER_REFERENCE,
    EpistemicStatus,
    RuntimeStatus,
    WorldEnvironment,
    available_biochemistry_profiles,
    get_biochemistry_profile,
)
from origins.scenarios import SCENARIO_A, SCENARIO_C, SCENARIO_D, SCENARIO_E
from origins.simulator.universal import UniversalOriginSimulator


def test_profiles_share_substrate_agnostic_life_invariants() -> None:
    assert LIFE_RELATIONAL_INVARIANTS == (
        "BOUNDED_SYSTEM",
        "ENERGY_THROUGHPUT",
        "PERSISTENT_INFORMATION_STATE",
        "HERITABLE_STATE_TRANSFORMATION",
        "SELECTION_OR_DIFFERENTIAL_PERSISTENCE",
    )
    for code in available_biochemistry_profiles():
        profile = get_biochemistry_profile(code)
        assert profile.invariant_requirements == LIFE_RELATIONAL_INVARIANTS


def test_world_environment_is_separate_from_biology_profile() -> None:
    env = WorldEnvironment.from_scenario(SCENARIO_C)
    assert env.solvent == "NH3"
    assert SCENARIO_C.biochemistry_profile == "AMMONIA_CANDIDATE"
    assert not hasattr(env, "information_carrier")


def test_reference_and_exotic_profiles_have_explicit_epistemic_status() -> None:
    assert WATER_REFERENCE.epistemic_status == EpistemicStatus.REFERENCE_MODEL
    assert WATER_REFERENCE.runtime_status == RuntimeStatus.REFERENCE_IMPLEMENTED

    assert AMMONIA_CANDIDATE.epistemic_status == EpistemicStatus.CANDIDATE
    assert AMMONIA_CANDIDATE.runtime_status == RuntimeStatus.TERRACENTRIC_CONTROL_ONLY
    assert AMMONIA_CANDIDATE.runtime_implemented is False

    assert HYDROCARBON_CANDIDATE.epistemic_status == EpistemicStatus.CANDIDATE
    assert HYDROCARBON_CANDIDATE.runtime_status == RuntimeStatus.TERRACENTRIC_CONTROL_ONLY
    assert HYDROCARBON_CANDIDATE.runtime_implemented is False


def test_scenarios_bind_expected_profiles_without_conflating_world_and_biology() -> None:
    assert SCENARIO_A.biochemistry_profile == "WATER_REFERENCE"
    assert SCENARIO_E.biochemistry_profile == "WATER_REFERENCE"
    assert SCENARIO_C.biochemistry_profile == "AMMONIA_CANDIDATE"
    assert SCENARIO_D.biochemistry_profile == "HYDROCARBON_CANDIDATE"


def test_profile_environment_mismatch_fails_loud() -> None:
    cfg = deepcopy(SCENARIO_C)
    cfg.biochemistry_profile = "WATER_REFERENCE"
    with pytest.raises(ValueError):
        UniversalOriginSimulator(cfg, Nx=8, Ny=8, include_clay=False)


def test_candidate_exotic_runtime_is_reported_as_control_only() -> None:
    sim = UniversalOriginSimulator(
        deepcopy(SCENARIO_C),
        Nx=8,
        Ny=8,
        include_clay=False,
        preseed_rna=False,
    )
    status = sim.biology_claim_status()
    assert status["profile"] == "AMMONIA_CANDIDATE"
    assert status["terracentric_control_only"] is True
    assert status["exotic_biology_simulated"] is False
    assert status["interpretation_allowed"] == "CONTROL_ONLY_NOT_EXOTIC_BIOLOGY"


def test_exotic_summary_does_not_score_against_terracentric_expected_protocells() -> None:
    sim = UniversalOriginSimulator(
        deepcopy(SCENARIO_D),
        Nx=8,
        Ny=8,
        include_clay=False,
        preseed_rna=False,
    )
    sim.initialize()
    record = sim.summary_record()
    assert record["Biochemistry_Profile"] == "HYDROCARBON_CANDIDATE"
    assert record["Biology_Runtime_Status"] == "TERRACENTRIC_CONTROL_ONLY"
    assert record["Exotic_Biology_Simulated"] is False
    assert record["Expected_ProtoC"] is None
    assert record["Success_Rate_pct"] is None


def test_water_reference_summary_retains_reference_comparison() -> None:
    sim = UniversalOriginSimulator(
        deepcopy(SCENARIO_A),
        Nx=8,
        Ny=8,
        include_clay=False,
        preseed_rna=False,
    )
    sim.initialize()
    record = sim.summary_record()
    assert record["Biochemistry_Profile"] == "WATER_REFERENCE"
    assert record["Biology_Runtime_Status"] == "REFERENCE_IMPLEMENTED"
    assert record["Expected_ProtoC"] == SCENARIO_A.expected_protocells
    assert record["Success_Rate_pct"] == 0.0
