from origins.analysis.first_rna_base_resolved_v05 import (
    BASE_EXTENSION_MULTIPLIER,
    BASE_HOURS,
    BASE_TAIL_WINDOW,
    run_base_resolved_calibration_v05,
    simulate_base,
)


def test_base_parameters_are_explicit_and_g_is_not_guessed():
    assert BASE_EXTENSION_MULTIPLIER["A"] == 1.0
    assert BASE_EXTENSION_MULTIPLIER["U"] == 1.0
    assert BASE_EXTENSION_MULTIPLIER["C"] == 0.025
    assert BASE_EXTENSION_MULTIPLIER["G"] is None
    assert BASE_HOURS["G"] is None
    assert BASE_TAIL_WINDOW["G"] is None


def test_a_u_retain_v03_calibrated_tail_window():
    for base in ("A", "U"):
        row = simulate_base(base)
        assert row["tail_gate_pass"] is True
        assert 40 <= row["predicted_tail_nt"] <= 55
        assert row["extension_multiplier"] == 1.0
        assert row["conservation_pass"] is True


def test_c_calibrated_profile_resolves_parent_holdout_failure():
    row = simulate_base("C")
    assert row["tail_gate_pass"] is True
    assert 20 <= row["predicted_tail_nt"] <= 25
    assert row["extension_multiplier"] == 0.025
    assert row["conservation_pass"] is True


def test_g_remains_explicitly_unresolved():
    row = simulate_base("G")
    assert row["status"] == "UNRESOLVED_EXPERIMENTAL_LENGTH"
    assert row["predicted_tail_nt"] is None
    assert row["extension_multiplier"] is None


def test_base_resolved_gate_preserves_causal_boundary():
    rows, summary = run_base_resolved_calibration_v05()
    assert len(rows) == 4
    assert summary["calibration_not_prediction"] is True
    assert summary["global_rates_retuned"] is False
    assert summary["geometry_used"] is False
    assert summary["zeta_used"] is False
    assert summary["replication_used"] is False
