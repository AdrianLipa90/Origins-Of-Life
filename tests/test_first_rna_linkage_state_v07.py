import math

from origins.analysis.first_rna_linkage_state_v07 import (
    DIMER_3P5_FRACTION,
    TRIMER_3P5_FRACTION_OBSERVED,
    LinkageState,
    intervals_overlap,
    predict_length_invariant_linkage,
    rounded_percent_interval,
    run_linkage_state_v07,
)


def test_linkage_state_is_explicit_and_normalized():
    state = LinkageState.from_3p5_fraction(0.51)
    assert math.isclose(state.total_bond_weight, 1.0)
    assert math.isclose(state.fraction_3p5, 0.51)
    assert math.isclose(state.bond_weight_2p5, 0.49)


def test_length_invariant_prediction_is_frozen_from_dimer_only():
    assert DIMER_3P5_FRACTION == 0.51
    assert TRIMER_3P5_FRACTION_OBSERVED == 0.61
    assert predict_length_invariant_linkage(2).fraction_3p5 == 0.51
    assert predict_length_invariant_linkage(3).fraction_3p5 == 0.51


def test_reported_integer_percent_intervals_are_disjoint():
    dimer = rounded_percent_interval(51)
    trimer = rounded_percent_interval(61)
    assert dimer == (0.505, 0.515)
    assert trimer == (0.605, 0.615)
    assert intervals_overlap(dimer, trimer) is False


def test_v07_fails_without_retuning_or_sequence_terms():
    rows, summary = run_linkage_state_v07()
    assert len(rows) == 2
    assert summary["status"] == "FAIL_LENGTH_INVARIANT_LINKAGE_POINT_ESTIMATE"
    assert math.isclose(summary["trimer_predicted_3p5_fraction"], 0.51)
    assert math.isclose(summary["trimer_observed_3p5_fraction"], 0.61)
    assert math.isclose(summary["trimer_delta"], -0.10)
    assert summary["rounding_intervals_overlap"] is False
    assert summary["statistical_significance_claim"] is False
    assert summary["global_rates_retuned"] is False
    assert summary["base_extension_multipliers_retuned"] is False
    assert summary["sequence_pair_term_added"] is False
    assert summary["chain_length_linkage_term_added"] is False
    assert summary["geometry_used"] is False
    assert summary["zeta_used"] is False
    assert summary["replication_used"] is False
