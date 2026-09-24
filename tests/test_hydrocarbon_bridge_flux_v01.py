from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from origins.exobiology import HydrocarbonCandidateSimulator
from origins.exobiology.boundary_morphogenesis import (
    EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
)
from origins.exobiology.hydrocarbon_bridge_flux import (
    EVENT_NO_MERGER_WITHIN_HORIZON,
    audited_hydrocarbon_step,
    diagnostic_manifest,
    run_hydrocarbon_bridge_flux_case,
)
from origins.exobiology.information_morphogenesis import (
    PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
)
from origins.scenarios import SCENARIO_D


def _sim(seed: int) -> HydrocarbonCandidateSimulator:
    return HydrocarbonCandidateSimulator(
        deepcopy(SCENARIO_D),
        Nx=12,
        Ny=12,
        seed=seed,
        information_morphogenesis=PEAK_REDISTRIBUTED_INFORMATION_CANDIDATE,
        boundary_morphogenesis=EXTERIOR_REDISTRIBUTED_BOUNDARY_CANDIDATE,
    )


def test_audited_step_matches_native_step_exactly() -> None:
    audited = _sim(17)
    native = _sim(17)
    audited.initialize()
    native.initialize()

    for _ in range(25):
        audited.step()
        native.step()

    audit = audited_hydrocarbon_step(audited)
    native.step()

    for name in ("S", "A", "I", "B", "E", "Q", "interface_template"):
        assert np.array_equal(getattr(audited, name), getattr(native, name))
    assert audited.time == pytest.approx(native.time, abs=0.0)
    assert audited.energy_throughput_integral == pytest.approx(
        native.energy_throughput_integral,
        abs=0.0,
    )
    assert audited.material_total() == pytest.approx(native.material_total(), abs=0.0)

    reconstructed = (
        audit["information_start"]
        + audit["reaction_net_delta"]
        + audit["diffusion_delta"]
    )
    assert np.allclose(
        audit["information_final"],
        reconstructed,
        rtol=0.0,
        atol=2e-15,
    )
    assert float(np.max(np.abs(audit["closure_error"]))) <= 2e-15


def test_audited_reaction_split_is_assembly_minus_decay() -> None:
    sim = _sim(23)
    sim.initialize()
    sim.run(steps=30)
    audit = audited_hydrocarbon_step(sim)

    expected = audit["assembly_growth"] - audit["decay_loss"]
    assert np.allclose(
        audit["reaction_net_delta"],
        expected,
        rtol=0.0,
        atol=2e-15,
    )
    assert float(np.min(audit["assembly_growth"])) >= 0.0
    assert float(np.min(audit["decay_loss"])) >= 0.0


def test_short_horizon_does_not_invent_bridge_flux() -> None:
    result = run_hydrocarbon_bridge_flux_case(
        deepcopy(SCENARIO_D),
        seed=47,
        horizon_steps=20,
        Nx=8,
        Ny=8,
    )
    assert result.event == EVENT_NO_MERGER_WITHIN_HORIZON
    assert result.bridge_x is None
    assert result.bridge_y is None
    assert result.reaction_net_delta is None
    assert result.diffusion_delta is None


def test_bridge_flux_manifest_is_diagnostic_only() -> None:
    manifest = diagnostic_manifest()
    assert manifest["decomposition"] == "DELTA_I = ASSEMBLY - DECAY + DIFFUSION"
    assert manifest["diagnostic_only"] is True
    assert manifest["model_mutation_allowed"] is False
    assert manifest["parameter_tuning_allowed"] is False
    assert manifest["threshold_tuning_allowed"] is False
    assert manifest["ranking_allowed"] is False
    assert manifest["physical_binding"] == "OPEN"
