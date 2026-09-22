"""Held-out phosphodiester-linkage capability audit for first-RNA v0.6.

v0.6 does not add linkage chemistry. It asks whether the frozen v0.5 model
already contains enough state to predict 2',5' versus 3',5' regioselectivity.

The expected honest outcome for the current architecture is
FAIL_MISSING_LINKAGE_STATE.
"""

from __future__ import annotations

import pandas as pd

from . import first_rna_base_resolved_v05 as v05
from . import first_rna_reactive_state_v03 as v03


EMPIRICAL_3P5_FRACTION: dict[str, float] = {
    "A": 0.74,
    "U": 0.61,
}

SOURCE_DOI = "10.1021/ja061782k"
HOLDOUT_BASES = ("A", "U")

_LINKAGE_TOKENS = (
    "linkage",
    "phosphodiester",
    "3p5",
    "2p5",
    "3_5",
    "2_5",
    "3prime5",
    "2prime5",
)


def _has_linkage_name(names: list[str] | tuple[str, ...]) -> bool:
    normalized = [str(name).lower().replace("′", "").replace("'", "") for name in names]
    return any(
        any(token in name for token in _LINKAGE_TOKENS)
        for name in normalized
    )


def frozen_model_linkage_capability() -> dict[str, object]:
    """Audit whether the frozen v0.5/v0.3 state can represent linkage identity."""
    profile_fields = tuple(v03.ReactiveProfile.__dataclass_fields__.keys())
    state_fields = tuple(v03.ReactiveState.__dataclass_fields__.keys())

    probe_outputs: dict[str, tuple[str, ...]] = {}
    for base in HOLDOUT_BASES:
        probe_outputs[base] = tuple(v05.simulate_base(base).keys())

    module_symbols = tuple(name for name in dir(v05) if not name.startswith("__"))

    profile_has_linkage = _has_linkage_name(profile_fields)
    state_has_linkage = _has_linkage_name(state_fields)
    output_has_linkage = any(
        _has_linkage_name(keys) for keys in probe_outputs.values()
    )
    module_has_linkage = _has_linkage_name(module_symbols)

    explicit_linkage_state = bool(
        profile_has_linkage or state_has_linkage or output_has_linkage or module_has_linkage
    )

    return {
        "explicit_linkage_state": explicit_linkage_state,
        "profile_has_linkage": profile_has_linkage,
        "state_has_linkage": state_has_linkage,
        "output_has_linkage": output_has_linkage,
        "module_has_linkage": module_has_linkage,
        "profile_fields": profile_fields,
        "state_fields": state_fields,
        "probe_output_keys": probe_outputs,
    }


def run_linkage_holdout_v06() -> tuple[pd.DataFrame, dict[str, object]]:
    capability = frozen_model_linkage_capability()

    rows = []
    for base in HOLDOUT_BASES:
        rows.append(
            {
                "base": base,
                "empirical_3p5_fraction": EMPIRICAL_3P5_FRACTION[base],
                "model_3p5_fraction": None,
                "numerical_comparison": "NOT_RUN",
                "status": (
                    "UNEXPECTED_LINKAGE_STATE_PRESENT"
                    if capability["explicit_linkage_state"]
                    else "MISSING_LINKAGE_STATE"
                ),
            }
        )

    frame = pd.DataFrame(rows)

    if capability["explicit_linkage_state"]:
        verdict = "ESCALATE_UNEXPECTED_LINKAGE_STATE"
    else:
        verdict = "FAIL_MISSING_LINKAGE_STATE"

    summary = {
        "schema": "ORIGINS_FIRST_RNA_LINKAGE_HOLDOUT_V0_6",
        "status": verdict,
        "held_out_observable": "3p5_phosphodiester_fraction",
        "source_doi": SOURCE_DOI,
        "empirical_a_3p5_fraction": EMPIRICAL_3P5_FRACTION["A"],
        "empirical_u_3p5_fraction": EMPIRICAL_3P5_FRACTION["U"],
        "explicit_linkage_state": bool(capability["explicit_linkage_state"]),
        "numerical_comparison_run": False,
        "global_rates_retuned": False,
        "base_extension_multipliers_retuned": False,
        "linkage_parameter_added": False,
        "empirical_fraction_copied_into_prediction": False,
        "geometry_used": False,
        "zeta_used": False,
        "replication_used": False,
        "capability": capability,
    }
    return frame, summary
