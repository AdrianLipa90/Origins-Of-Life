"""Held-out base-specific validation v0.4.

No v0.3 kinetic parameter is changed. The base-blind activated-clay profile is
extended from 24 h to the 216 h C-observation horizon.
"""

from __future__ import annotations

from dataclasses import replace

from .first_rna_reactive_state_v03 import (
    ACTIVATED_CLAY_PROFILE,
    simulate_reactive_profile,
)


C_HOLDOUT_HOURS = 216.0
C_EMPIRICAL_TAIL_MIN_NT = 20
C_EMPIRICAL_TAIL_MAX_NT = 25


def run_base_specific_holdout_v04() -> dict[str, object]:
    profile = replace(
        ACTIVATED_CLAY_PROFILE,
        name="activated_clay_C_holdout_base_blind",
        hours=C_HOLDOUT_HOURS,
    )
    result = simulate_reactive_profile(profile)
    predicted_tail = int(result["max_length_mass_fraction_ge_1e_10"])
    conservation_ok = bool(
        float(result["max_relative_conservation_error"]) <= 1e-10
    )
    tail_ok = bool(
        C_EMPIRICAL_TAIL_MIN_NT
        <= predicted_tail
        <= C_EMPIRICAL_TAIL_MAX_NT
    )
    passed = bool(conservation_ok and tail_ok)
    return {
        "schema": "ORIGINS_FIRST_RNA_BASE_HOLDOUT_V0_4",
        "status": (
            "PASS_BASE_SPECIFIC_HOLDOUT"
            if passed
            else "FAIL_BASE_SPECIFIC_HOLDOUT"
        ),
        "held_out_base": "C",
        "hours": C_HOLDOUT_HOURS,
        "predicted_tail_length_nt": predicted_tail,
        "empirical_tail_min_nt": C_EMPIRICAL_TAIL_MIN_NT,
        "empirical_tail_max_nt": C_EMPIRICAL_TAIL_MAX_NT,
        "tail_gate_pass": tail_ok,
        "mass_fraction_ge_40": float(result["mass_fraction_ge_40"]),
        "mass_fraction_ge_30": float(result["mass_fraction_ge_30"]),
        "mass_fraction_ge_10": float(result["mass_fraction_ge_10"]),
        "max_relative_conservation_error": float(
            result["max_relative_conservation_error"]
        ),
        "conservation_pass": conservation_ok,
        "parameters_retuned": False,
        "base_identity_present_in_model": False,
        "geometry_used": False,
        "zeta_used": False,
        "replication_used": False,
        "physical_rate_claim": False,
    }
