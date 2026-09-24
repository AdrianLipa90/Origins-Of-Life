"""Cross-condition transfer test for first-RNA linkage selectivity v0.8.

The A/U dimer-to-trimer linkage shift is calibrated once and transferred
unchanged to the A/C binary system. The A/C trimer point estimate is held out
from prediction.

This is a rounding-compatibility test, not a statistical-significance test.
"""

from __future__ import annotations

import pandas as pd


AU_DIMER_3P5 = 0.51
AU_TRIMER_3P5 = 0.61
AC_DIMER_3P5 = 0.49
AC_TRIMER_3P5_OBSERVED = 0.56

AU_SOURCE_DOI = "10.1089/ast.2007.0138"
AC_SOURCE_DOI = "10.1017/S147355040700393X"


def rounded_percent_envelope(percent: int) -> tuple[float, float]:
    value = int(percent)
    if not 0 <= value <= 100:
        raise ValueError("percent must lie in [0,100]")
    return ((value - 0.5) / 100.0, (value + 0.5) / 100.0)


def interval_sum(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    return (left[0] + right[0], left[1] + right[1])


def interval_difference(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    return (left[0] - right[1], left[1] - right[0])


def intervals_overlap(
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    return bool(max(left[0], right[0]) <= min(left[1], right[1]))


def run_pair_context_holdout_v08() -> tuple[pd.DataFrame, dict[str, object]]:
    stage_shift = AU_TRIMER_3P5 - AU_DIMER_3P5
    ac_prediction = AC_DIMER_3P5 + stage_shift

    au_dimer_env = rounded_percent_envelope(51)
    au_trimer_env = rounded_percent_envelope(61)
    shift_env = interval_difference(au_trimer_env, au_dimer_env)

    ac_dimer_env = rounded_percent_envelope(49)
    ac_prediction_env = interval_sum(ac_dimer_env, shift_env)
    ac_observed_env = rounded_percent_envelope(56)

    overlap = intervals_overlap(ac_prediction_env, ac_observed_env)

    rows = pd.DataFrame(
        [
            {
                "condition": "A/U",
                "role": "CALIBRATION",
                "dimer_3p5": AU_DIMER_3P5,
                "trimer_3p5_observed": AU_TRIMER_3P5,
                "stage_shift": stage_shift,
                "trimer_3p5_predicted": AU_TRIMER_3P5,
            },
            {
                "condition": "A/C",
                "role": "PAIR_HELD_OUT",
                "dimer_3p5": AC_DIMER_3P5,
                "trimer_3p5_observed": AC_TRIMER_3P5_OBSERVED,
                "stage_shift": stage_shift,
                "trimer_3p5_predicted": ac_prediction,
            },
        ]
    )

    verdict = (
        "PASS_UNIVERSAL_STAGE_SHIFT_ROUNDING_COMPATIBILITY"
        if overlap
        else "FAIL_UNIVERSAL_STAGE_SHIFT"
    )

    summary = {
        "schema": "ORIGINS_FIRST_RNA_PAIR_CONTEXT_HOLDOUT_V0_8",
        "status": verdict,
        "au_source_doi": AU_SOURCE_DOI,
        "ac_source_doi": AC_SOURCE_DOI,
        "au_stage_shift": stage_shift,
        "ac_trimer_predicted": ac_prediction,
        "ac_trimer_observed": AC_TRIMER_3P5_OBSERVED,
        "ac_prediction_error": ac_prediction - AC_TRIMER_3P5_OBSERVED,
        "au_dimer_rounding_envelope": au_dimer_env,
        "au_trimer_rounding_envelope": au_trimer_env,
        "stage_shift_rounding_envelope": shift_env,
        "ac_dimer_rounding_envelope": ac_dimer_env,
        "ac_prediction_rounding_envelope": ac_prediction_env,
        "ac_observed_rounding_envelope": ac_observed_env,
        "rounding_envelopes_overlap": overlap,
        "statistical_significance_claim": False,
        "pair_specific_parameter_added": False,
        "global_rates_retuned": False,
        "base_extension_multipliers_retuned": False,
        "geometry_used": False,
        "zeta_used": False,
        "replication_used": False,
    }
    return rows, summary
