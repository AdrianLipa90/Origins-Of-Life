from origins.analysis.first_rna_base_holdout_v04 import (
    C_EMPIRICAL_TAIL_MAX_NT,
    C_EMPIRICAL_TAIL_MIN_NT,
    C_HOLDOUT_HOURS,
    run_base_specific_holdout_v04,
)
from origins.analysis.first_rna_reactive_state_v03 import (
    ACTIVATED_CLAY_PROFILE,
)


def test_holdout_changes_only_duration_from_v03_activated_clay_fixture():
    assert C_HOLDOUT_HOURS == 216.0
    assert ACTIVATED_CLAY_PROFILE.hours == 24.0
    assert ACTIVATED_CLAY_PROFILE.k_extension == 1e-4
    assert ACTIVATED_CLAY_PROFILE.k_nucleation == 1e-10
    assert ACTIVATED_CLAY_PROFILE.k_hydrolysis == 1e-5
    assert ACTIVATED_CLAY_PROFILE.k_deactivation == 1e-4


def test_holdout_result_preserves_causal_boundary_and_conservation():
    result = run_base_specific_holdout_v04()
    assert result["parameters_retuned"] is False
    assert result["base_identity_present_in_model"] is False
    assert result["geometry_used"] is False
    assert result["zeta_used"] is False
    assert result["replication_used"] is False
    assert result["conservation_pass"] is True


def test_holdout_gate_uses_frozen_20_to_25_nt_empirical_window():
    assert C_EMPIRICAL_TAIL_MIN_NT == 20
    assert C_EMPIRICAL_TAIL_MAX_NT == 25
