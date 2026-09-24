from __future__ import annotations

from copy import deepcopy
import inspect

import numpy as np
import pytest

from origins.exobiology import (
    HYDROCARBON_CANDIDATE,
    HYDROCARBON_RUNTIME_CODE,
    HydrocarbonCandidateParameters,
    HydrocarbonCandidateSimulator,
    available_exotic_candidate_runtimes,
    create_exotic_candidate_simulator,
)
from origins.scenarios import SCENARIO_C, SCENARIO_D
from origins.simulator.universal import UniversalOriginSimulator


def test_hydrocarbon_profile_registers_dedicated_candidate_runtime() -> None:
    status = HYDROCARBON_CANDIDATE.claim_status()
    assert status["dedicated_runtime"] == HYDROCARBON_RUNTIME_CODE
    assert status["dedicated_runtime_status"] == "COMPUTATIONAL_CANDIDATE_IMPLEMENTED"
    assert status["runtime_status"] == "TERRACENTRIC_CONTROL_ONLY"


def test_hydrocarbon_runtime_is_non_rna_non_lipid_and_has_explicit_interface_reservoir() -> None:
    source = inspect.getsource(HydrocarbonCandidateSimulator)
    assert "RNAPopulation" not in source
    assert "RNASequence" not in source
    assert "K_MEMBRANE" not in source
    assert "biology.rna" not in source
    assert "self.A" in source
    assert "interface_template" in source


def test_exotic_factory_routes_both_dedicated_candidate_runtimes() -> None:
    hydro = create_exotic_candidate_simulator(
        deepcopy(SCENARIO_D),
        Nx=12,
        Ny=12,
    )
    assert isinstance(hydro, HydrocarbonCandidateSimulator)
    assert "HYDROCARBON_CANDIDATE_V0_1" in available_exotic_candidate_runtimes()

    ammonia = create_exotic_candidate_simulator(
        deepcopy(SCENARIO_C),
        Nx=12,
        Ny=12,
    )
    assert ammonia.__class__.__name__ == "AmmoniaCandidateSimulator"


def test_hydrocarbon_candidate_conserves_material_over_many_steps() -> None:
    sim = HydrocarbonCandidateSimulator(
        deepcopy(SCENARIO_D),
        Nx=20,
        Ny=20,
        seed=1234,
    )
    sim.initialize()
    before = sim.material_total()
    sim.run(steps=250)
    after = sim.material_total()
    assert after == pytest.approx(before, rel=2e-10, abs=2e-10)


def test_hydrocarbon_candidate_is_reproducible_for_fixed_seed() -> None:
    a = HydrocarbonCandidateSimulator(deepcopy(SCENARIO_D), Nx=16, Ny=16, seed=5)
    b = HydrocarbonCandidateSimulator(deepcopy(SCENARIO_D), Nx=16, Ny=16, seed=5)
    a.initialize()
    b.initialize()
    a.run(steps=120)
    b.run(steps=120)
    for name in ("S", "A", "I", "B", "E", "Q", "interface_template"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_hydrocarbon_candidate_uses_interfacial_exchange_not_membrane_assumption() -> None:
    sim = HydrocarbonCandidateSimulator(deepcopy(SCENARIO_D), Nx=18, Ny=18, seed=11)
    sim.initialize()
    dissolved_before = float(np.sum(sim.S))
    aggregate_before = float(np.sum(sim.A))
    sim.step_interface_exchange()
    assert float(np.sum(sim.S)) < dissolved_before
    assert float(np.sum(sim.A)) > aggregate_before
    assert float(np.sum(sim.S) + np.sum(sim.A)) == pytest.approx(
        dissolved_before + aggregate_before,
        rel=1e-12,
        abs=1e-12,
    )


def test_hydrocarbon_candidate_operationalizes_same_relational_invariants_with_open_binding() -> None:
    sim = HydrocarbonCandidateSimulator(deepcopy(SCENARIO_D), Nx=16, Ny=16, seed=13)
    sim.initialize()
    sim.run(steps=100)
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


def test_hydrocarbon_claim_does_not_promote_specific_azotosome_or_established_life() -> None:
    sim = HydrocarbonCandidateSimulator(deepcopy(SCENARIO_D), Nx=10, Ny=10)
    sim.initialize()
    status = sim.claim_status()
    assert status["runtime"] == "HYDROCARBON_CANDIDATE_V0_1"
    assert status["explicit_interface_reservoir"] is True
    assert status["specific_azotosome_claim"] is False
    assert status["physical_binding"] == "OPEN"
    assert status["exotic_biology_established"] is False
    assert status["interpretation_allowed"] == "COMPUTATIONAL_CANDIDATE_ONLY"


def test_universal_hydrocarbon_runtime_remains_terracentric_control() -> None:
    sim = UniversalOriginSimulator(
        deepcopy(SCENARIO_D),
        Nx=8,
        Ny=8,
        include_clay=False,
        preseed_rna=False,
    )
    assert sim.biology_claim_status()["interpretation_allowed"] == "CONTROL_ONLY_NOT_EXOTIC_BIOLOGY"


def test_hydrocarbon_parameter_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError):
        HydrocarbonCandidateParameters(dt=0.0).validate()
    with pytest.raises(ValueError):
        HydrocarbonCandidateParameters(inheritance_rate=1.1).validate()
