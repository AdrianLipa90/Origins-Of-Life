import numpy as np

from origins.analysis.first_rna_reactive_state_v03 import (
    ACTIVATED_CLAY_PROFILE,
    CYCLIC_WET_DRY_PROFILE,
    ReactiveProfile,
    ReactiveState,
    run_reactive_state_calibration,
    simulate_reactive_profile,
    step_reactive_state,
)


def test_reactive_state_initialization_is_exactly_conservative():
    state = ReactiveState.initialize(CYCLIC_WET_DRY_PROFILE)
    assert state.total_units() == 10_000.0
    assert state.unreactive_free == 0.0
    assert state.reactive_free == 10_000.0
    assert np.count_nonzero(state.chains) == 0


def test_extreme_competing_fluxes_remain_nonnegative_and_conservative():
    profile = ReactiveProfile(
        name="stress",
        hours=1.0,
        dt_h=1.0,
        initial_units=100.0,
        k_deactivation=10.0,
        k_nucleation=10.0,
        k_extension=10.0,
        k_hydrolysis=10.0,
        cycle_h=None,
        dry_fraction=0.0,
        dry_extension_multiplier=1.0,
        dry_hydrolysis_multiplier=1.0,
    )
    state = ReactiveState.initialize(profile)
    before = state.total_units()
    step_reactive_state(state, profile, t_h=0.0)
    after = state.total_units()

    assert state.unreactive_free >= 0.0
    assert state.reactive_free >= 0.0
    assert np.all(state.chains >= -1e-12)
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-10)


def test_frozen_cyclic_profile_hits_calibration_envelope():
    row = simulate_reactive_profile(CYCLIC_WET_DRY_PROFILE)
    assert 3e-4 <= row["mass_fraction_ge_10"] <= 3e-3
    assert row["max_relative_conservation_error"] <= 1e-10
    assert row["geometry_used"] is False
    assert row["zeta_used"] is False
    assert row["replication_used"] is False


def test_frozen_activated_clay_profile_has_sparse_long_tail():
    row = simulate_reactive_profile(ACTIVATED_CLAY_PROFILE)
    assert 1e-9 <= row["mass_fraction_ge_40"] <= 1e-6
    assert 40 <= row["max_length_mass_fraction_ge_1e_10"] <= 55
    assert row["max_relative_conservation_error"] <= 1e-10


def test_v03_calibration_gate_is_explicitly_not_predictive_validation():
    rows, summary = run_reactive_state_calibration()
    assert len(rows) == 2
    assert summary["calibration_not_prediction"] is True
    assert summary["geometry_used"] is False
    assert summary["zeta_used"] is False
    assert summary["replication_used"] is False
