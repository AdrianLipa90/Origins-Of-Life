from __future__ import annotations

from copy import deepcopy
import inspect

import numpy as np
import pytest

from origins.exobiology import (
    AMMONIA_CANDIDATE,
    AMMONIA_RUNTIME_CODE,
    AmmoniaCandidateParameters,
    AmmoniaCandidateSimulator,
    available_exotic_candidate_runtimes,
    create_exotic_candidate_simulator,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D
from origins.simulator.universal import UniversalOriginSimulator


def test_ammonia_profile_registers_dedicated_candidate_runtime() -> None:
    status = AMMONIA_CANDIDATE.claim_status()
    assert status["dedicated_runtime"] == AMMONIA_RUNTIME_CODE
    assert status["dedicated_runtime_status"] == "COMPUTATIONAL_CANDIDATE_IMPLEMENTED"
    assert status["runtime_status"] == "TERRACENTRIC_CONTROL_ONLY"


def test_ammonia_runtime_is_non_rna_and_non_lipid_by_construction() -> None:
    source = inspect.getsource(AmmoniaCandidateSimulator)
    assert "RNAPopulation" not in source
    assert "RNASequence" not in source
    assert "K_MEMBRANE" not in source
    assert "lipid" not in source.lower()


def test_ammonia_factory_returns_dedicated_runtime_and_hydrocarbon_fails_closed() -> None:
    ammonia = create_exotic_candidate_simulator(
        deepcopy(SCENARIO_C),
        Nx=12,
        Ny=12,
    )
    assert isinstance(ammonia, AmmoniaCandidateSimulator)
    assert AMMONIA_RUNTIME_CODE in available_exotic_candidate_runtimes()

    with pytest.raises(NotImplementedError):
        create_exotic_candidate_simulator(deepcopy(SCENARIO_D), Nx=12, Ny=12)


def test_ammonia_candidate_conserves_material_over_many_steps() -> None:
    sim = AmmoniaCandidateSimulator(
        deepcopy(SCENARIO_C),
        Nx=20,
        Ny=20,
        seed=1234,
    )
    sim.initialize()
    before = sim.material_total()
    sim.run(steps=250)
    after = sim.material_total()
    assert after == pytest.approx(before, rel=2e-10, abs=2e-10)


def test_ammonia_candidate_is_reproducible_for_fixed_seed() -> None:
    a = AmmoniaCandidateSimulator(deepcopy(SCENARIO_C), Nx=16, Ny=16, seed=7)
    b = AmmoniaCandidateSimulator(deepcopy(SCENARIO_C), Nx=16, Ny=16, seed=7)
    a.initialize()
    b.initialize()
    a.run(steps=100)
    b.run(steps=100)
    for name in ("P", "I", "B", "E", "Q"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_ammonia_candidate_operationalizes_all_relational_invariants_but_keeps_binding_open() -> None:
    sim = AmmoniaCandidateSimulator(deepcopy(SCENARIO_C), Nx=16, Ny=16, seed=9)
    sim.initialize()
    sim.run(steps=80)
    status = sim.life_invariant_status()
    assert set(status) == {
        "BOUNDED_SYSTEM",
        "ENERGY_THROUGHPUT",
        "PERSISTENT_INFORMATION_STATE",
        "HERITABLE_STATE_TRANSFORMATION",
        "SELECTION_OR_DIFFERENTIAL_PERSISTENCE",
    }
    assert all(item["operationalized"] is True for item in status.values())
    assert all(item["physical_binding"] == "OPEN" for item in status.values())


def test_ammonia_claim_is_candidate_only_not_established_exotic_life() -> None:
    sim = AmmoniaCandidateSimulator(deepcopy(SCENARIO_C), Nx=10, Ny=10)
    sim.initialize()
    status = sim.claim_status()
    assert status["runtime"] == "AMMONIA_CANDIDATE_V0_1"
    assert status["dedicated_non_rna_runtime"] is True
    assert status["dedicated_non_lipid_runtime"] is True
    assert status["physical_binding"] == "OPEN"
    assert status["exotic_biology_established"] is False
    assert status["interpretation_allowed"] == "COMPUTATIONAL_CANDIDATE_ONLY"


def test_universal_ammonia_runtime_remains_terracentric_control_despite_dedicated_candidate() -> None:
    sim = UniversalOriginSimulator(
        deepcopy(SCENARIO_C),
        Nx=8,
        Ny=8,
        include_clay=False,
        preseed_rna=False,
    )
    assert sim.biology_claim_status()["interpretation_allowed"] == "CONTROL_ONLY_NOT_EXOTIC_BIOLOGY"


def test_ammonia_parameter_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError):
        AmmoniaCandidateParameters(dt=0.0).validate()
    with pytest.raises(ValueError):
        AmmoniaCandidateParameters(selection_strength=1.1).validate()
