import math

from origins.analysis.first_rna_pair_context_holdout_v08 import (
    AC_TRIMER_3P5_OBSERVED,
    interval_difference,
    interval_sum,
    intervals_overlap,
    rounded_percent_envelope,
    run_pair_context_holdout_v08,
)


def test_rounding_envelope_arithmetic_is_explicit():
    shift = interval_difference(
        rounded_percent_envelope(61),
        rounded_percent_envelope(51),
    )
    prediction = interval_sum(rounded_percent_envelope(49), shift)
    observed = rounded_percent_envelope(56)

    assert math.isclose(shift[0], 0.09)
    assert math.isclose(shift[1], 0.11)
    assert math.isclose(prediction[0], 0.575)
    assert math.isclose(prediction[1], 0.605)
    assert math.isclose(observed[0], 0.555)
    assert math.isclose(observed[1], 0.565)
    assert intervals_overlap(prediction, observed) is False


def test_pair_held_out_prediction_is_not_observation():
    rows, summary = run_pair_context_holdout_v08()
    ac = rows.loc[rows["condition"] == "A/C"].iloc[0]

    assert math.isclose(ac["trimer_3p5_predicted"], 0.59)
    assert math.isclose(ac["trimer_3p5_observed"], AC_TRIMER_3P5_OBSERVED)
    assert not math.isclose(
        ac["trimer_3p5_predicted"],
        ac["trimer_3p5_observed"],
    )
    assert math.isclose(summary["ac_prediction_error"], 0.03)


def test_universal_stage_shift_fails_pair_holdout():
    _, summary = run_pair_context_holdout_v08()

    assert summary["status"] == "FAIL_UNIVERSAL_STAGE_SHIFT"
    assert summary["rounding_envelopes_overlap"] is False
    assert summary["statistical_significance_claim"] is False
    assert summary["pair_specific_parameter_added"] is False


def test_causal_boundary_is_preserved():
    _, summary = run_pair_context_holdout_v08()
    assert summary["global_rates_retuned"] is False
    assert summary["base_extension_multipliers_retuned"] is False
    assert summary["geometry_used"] is False
    assert summary["zeta_used"] is False
    assert summary["replication_used"] is False
