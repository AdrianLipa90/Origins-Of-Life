from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology import (
    JOVIAN_TIME_VARYING_FIELD_CANDIDATE,
    MAGNETO_INDUCTION_OFF,
    AmmoniaCandidateSimulator,
    MagnetoInductionParameters,
    magneto_induction_observation,
)
from origins.scenarios import SCENARIO_C


def test_static_nonzero_field_does_not_create_induction_source() -> None:
    p = MagnetoInductionParameters(
        field_bias=2.0,
        primary_amplitude=0.0,
        secondary_amplitude=0.0,
        conductivity_proxy=1.0,
        energy_coupling_gain=1.0,
    )
    obs = magneto_induction_observation(
        3.0,
        mode=JOVIAN_TIME_VARYING_FIELD_CANDIDATE,
        parameters=p,
    )
    assert obs.field_proxy == pytest.approx(2.0)
    assert obs.field_derivative_proxy == pytest.approx(0.0)
    assert obs.induced_electric_proxy == pytest.approx(0.0)
    assert obs.current_density_proxy == pytest.approx(0.0)
    assert obs.ohmic_power_density_proxy == pytest.approx(0.0)
    assert obs.energy_source_rate_proxy == pytest.approx(0.0)


def test_zero_conductivity_blocks_current_and_ohmic_source() -> None:
    p = MagnetoInductionParameters(
        conductivity_proxy=0.0,
        primary_amplitude=0.5,
        primary_angular_frequency=0.7,
        secondary_amplitude=0.0,
    )
    obs = magneto_induction_observation(
        0.0,
        mode=JOVIAN_TIME_VARYING_FIELD_CANDIDATE,
        parameters=p,
    )
    assert abs(obs.field_derivative_proxy) > 0.0
    assert abs(obs.induced_electric_proxy) > 0.0
    assert obs.current_density_proxy == pytest.approx(0.0)
    assert obs.ohmic_power_density_proxy == pytest.approx(0.0)
    assert obs.energy_source_rate_proxy == pytest.approx(0.0)


def test_magneto_induction_off_is_exactly_zero() -> None:
    obs = magneto_induction_observation(
        123.0,
        mode=MAGNETO_INDUCTION_OFF,
    )
    assert obs.field_proxy == 0.0
    assert obs.field_derivative_proxy == 0.0
    assert obs.induced_electric_proxy == 0.0
    assert obs.current_density_proxy == 0.0
    assert obs.ohmic_power_density_proxy == 0.0
    assert obs.energy_source_rate_proxy == 0.0


def test_default_ammonia_runtime_replays_explicit_magneto_off_bitwise() -> None:
    implicit = AmmoniaCandidateSimulator(
        deepcopy(SCENARIO_C),
        Nx=14,
        Ny=14,
        seed=17,
    )
    explicit = AmmoniaCandidateSimulator(
        deepcopy(SCENARIO_C),
        Nx=14,
        Ny=14,
        seed=17,
        magneto_induction=MAGNETO_INDUCTION_OFF,
    )
    implicit.initialize()
    explicit.initialize()
    implicit.run(steps=120)
    explicit.run(steps=120)

    for name in ("P", "I", "B", "E", "Q"):
        assert np.array_equal(getattr(implicit, name), getattr(explicit, name))
    assert implicit.magnetic_energy_throughput_integral == 0.0
    assert explicit.magnetic_energy_throughput_integral == 0.0


def test_jovian_candidate_adds_energy_without_breaking_material_conservation() -> None:
    off = AmmoniaCandidateSimulator(
        deepcopy(SCENARIO_C),
        Nx=16,
        Ny=16,
        seed=23,
        magneto_induction=MAGNETO_INDUCTION_OFF,
    )
    on = AmmoniaCandidateSimulator(
        deepcopy(SCENARIO_C),
        Nx=16,
        Ny=16,
        seed=23,
        magneto_induction=JOVIAN_TIME_VARYING_FIELD_CANDIDATE,
    )
    off.initialize()
    on.initialize()

    m0 = on.material_total()
    off.run(steps=300)
    on.run(steps=300)

    assert on.material_total() == pytest.approx(m0, rel=2e-10, abs=2e-10)
    assert on.magnetic_energy_throughput_integral > 0.0
    assert float(np.sum(on.E)) > float(np.sum(off.E))
    assert on.last_magneto_induction is not None
    assert on.last_magneto_induction.physical_binding == "OPEN"


def test_magneto_claim_remains_candidate_only() -> None:
    sim = AmmoniaCandidateSimulator(
        deepcopy(SCENARIO_C),
        Nx=8,
        Ny=8,
        magneto_induction=JOVIAN_TIME_VARYING_FIELD_CANDIDATE,
    )
    sim.initialize()
    status = sim.claim_status()
    assert status["magneto_induction"] == JOVIAN_TIME_VARYING_FIELD_CANDIDATE
    assert status["magneto_induction_physical_binding"] == "OPEN"
    assert status["magneto_induction_parameter_status"] == (
        "UNVALIDATED_DIMENSIONLESS_CANDIDATE"
    )
    assert status["exotic_biology_established"] is False


def test_unknown_magneto_mode_fails_closed() -> None:
    with pytest.raises(ValueError):
        AmmoniaCandidateSimulator(
            deepcopy(SCENARIO_C),
            Nx=8,
            Ny=8,
            magneto_induction="MAGIC_FIELD",
        )
