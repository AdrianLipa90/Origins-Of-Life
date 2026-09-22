"""Base-resolved reactive-state calibration v0.5.

This layer retains the v0.3 conservative chemistry and adds an explicit base
identity at the profile level. Only the effective extension multiplier is
base-specific in v0.5.

The C multiplier is a calibration parameter, not a measured kinetic constant.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .first_rna_reactive_state_v03 import (
    ACTIVATED_CLAY_PROFILE,
    simulate_reactive_profile,
)


BASE_EXTENSION_MULTIPLIER: dict[str, float | None] = {
    "A": 1.0,
    "U": 1.0,
    "C": 0.025,
    "G": None,
}

BASE_HOURS: dict[str, float | None] = {
    "A": 24.0,
    "U": 24.0,
    "C": 216.0,
    "G": None,
}

BASE_TAIL_WINDOW: dict[str, tuple[int, int] | None] = {
    "A": (40, 55),
    "U": (40, 55),
    "C": (20, 25),
    "G": None,
}


def simulate_base(base: str) -> dict[str, object]:
    key = str(base).upper()
    if key not in BASE_EXTENSION_MULTIPLIER:
        raise ValueError(f"unsupported base: {base!r}")

    multiplier = BASE_EXTENSION_MULTIPLIER[key]
    hours = BASE_HOURS[key]
    window = BASE_TAIL_WINDOW[key]

    if multiplier is None or hours is None or window is None:
        return {
            "base": key,
            "status": "UNRESOLVED_EXPERIMENTAL_LENGTH",
            "extension_multiplier": None,
            "hours": None,
            "tail_min_nt": None,
            "tail_max_nt": None,
            "predicted_tail_nt": None,
            "tail_gate_pass": None,
            "max_relative_conservation_error": None,
            "conservation_pass": None,
            "geometry_used": False,
            "zeta_used": False,
            "replication_used": False,
        }

    profile = replace(
        ACTIVATED_CLAY_PROFILE,
        name=f"activated_clay_base_{key}",
        hours=float(hours),
        k_extension=(
            ACTIVATED_CLAY_PROFILE.k_extension * float(multiplier)
        ),
    )
    result = simulate_reactive_profile(profile)
    predicted = int(result["max_length_mass_fraction_ge_1e_10"])
    lo, hi = window
    tail_ok = bool(lo <= predicted <= hi)
    conservation_ok = bool(
        float(result["max_relative_conservation_error"]) <= 1e-10
    )

    return {
        "base": key,
        "status": "CALIBRATED",
        "extension_multiplier": float(multiplier),
        "hours": float(hours),
        "tail_min_nt": int(lo),
        "tail_max_nt": int(hi),
        "predicted_tail_nt": predicted,
        "tail_gate_pass": tail_ok,
        "mass_fraction_ge_10": float(result["mass_fraction_ge_10"]),
        "mass_fraction_ge_30": float(result["mass_fraction_ge_30"]),
        "mass_fraction_ge_40": float(result["mass_fraction_ge_40"]),
        "max_relative_conservation_error": float(
            result["max_relative_conservation_error"]
        ),
        "conservation_pass": conservation_ok,
        "geometry_used": False,
        "zeta_used": False,
        "replication_used": False,
    }


def run_base_resolved_calibration_v05() -> tuple[pd.DataFrame, dict[str, object]]:
    rows = pd.DataFrame([simulate_base(base) for base in ("A", "U", "C", "G")])
    resolved = rows[rows["status"] == "CALIBRATED"].copy()

    tails_ok = bool(resolved["tail_gate_pass"].all())
    conservation_ok = bool(resolved["conservation_pass"].all())
    g_unresolved = bool(
        rows.loc[rows["base"] == "G", "status"].iloc[0]
        == "UNRESOLVED_EXPERIMENTAL_LENGTH"
    )
    passed = bool(tails_ok and conservation_ok and g_unresolved)

    summary = {
        "schema": "ORIGINS_FIRST_RNA_BASE_RESOLVED_V0_5",
        "status": (
            "PASS_BASE_RESOLVED_CALIBRATION"
            if passed
            else "FAIL_BASE_RESOLVED_CALIBRATION"
        ),
        "calibration_not_prediction": True,
        "a_tail_nt": int(rows.loc[rows["base"] == "A", "predicted_tail_nt"].iloc[0]),
        "u_tail_nt": int(rows.loc[rows["base"] == "U", "predicted_tail_nt"].iloc[0]),
        "c_tail_nt": int(rows.loc[rows["base"] == "C", "predicted_tail_nt"].iloc[0]),
        "g_status": str(rows.loc[rows["base"] == "G", "status"].iloc[0]),
        "tail_gates_pass": tails_ok,
        "conservation_pass": conservation_ok,
        "g_unresolved_preserved": g_unresolved,
        "global_rates_retuned": False,
        "geometry_used": False,
        "zeta_used": False,
        "replication_used": False,
    }
    return rows, summary
