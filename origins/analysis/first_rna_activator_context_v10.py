"""Activator-context capability audit for first-RNA v0.10.

The frozen v0.9 symbolic local-context model is inspected for an explicit
activation-chemistry state. v0.10 adds no activator parameter.

Expected honest result: FAIL_MISSING_ACTIVATOR_CONTEXT.
"""

from __future__ import annotations

from dataclasses import fields

import pandas as pd

from .first_rna_local_context_v09 import DimerContext, GrowthContext


SOURCE_DOI = "10.1021/ja061782k"

EMPIRICAL_3P5_BY_ACTIVATOR: dict[str, dict[str, float]] = {
    "A": {
        "imidazole": 0.67,
        "1-methyladenine": 0.74,
    },
    "U": {
        "imidazole": 0.20,
        "1-methyladenine": 0.61,
    },
}

_ACTIVATOR_TOKENS = ("activator", "activation", "imidazole", "methyladenine")


def _has_activator_name(names: tuple[str, ...] | list[str]) -> bool:
    normalized = [str(name).lower().replace("_", "-") for name in names]
    return any(
        any(token in name for token in _ACTIVATOR_TOKENS)
        for name in normalized
    )


def frozen_v09_activator_capability() -> dict[str, object]:
    dimer_fields = tuple(field.name for field in fields(DimerContext))
    growth_fields = tuple(field.name for field in fields(GrowthContext))

    dimer_has_activator = _has_activator_name(dimer_fields)
    growth_has_activator = _has_activator_name(growth_fields)
    explicit_activator_context = bool(
        dimer_has_activator or growth_has_activator
    )

    return {
        "explicit_activator_context": explicit_activator_context,
        "dimer_fields": dimer_fields,
        "growth_fields": growth_fields,
        "dimer_has_activator": dimer_has_activator,
        "growth_has_activator": growth_has_activator,
    }


def run_activator_context_audit_v10() -> tuple[pd.DataFrame, dict[str, object]]:
    capability = frozen_v09_activator_capability()

    rows = pd.DataFrame(
        [
            {
                "base": base,
                "imidazole_3p5_fraction": values["imidazole"],
                "one_methyladenine_3p5_fraction": values["1-methyladenine"],
                "model_condition_comparison": "NOT_RUN",
                "status": (
                    "UNEXPECTED_ACTIVATOR_STATE_PRESENT"
                    if capability["explicit_activator_context"]
                    else "MISSING_ACTIVATOR_CONTEXT"
                ),
            }
            for base, values in EMPIRICAL_3P5_BY_ACTIVATOR.items()
        ]
    )

    verdict = (
        "ESCALATE_UNEXPECTED_ACTIVATOR_STATE"
        if capability["explicit_activator_context"]
        else "FAIL_MISSING_ACTIVATOR_CONTEXT"
    )

    summary = {
        "schema": "ORIGINS_FIRST_RNA_ACTIVATOR_CONTEXT_V0_10",
        "status": verdict,
        "source_doi": SOURCE_DOI,
        "explicit_activator_context": bool(
            capability["explicit_activator_context"]
        ),
        "numerical_comparison_run": False,
        "activator_parameter_added": False,
        "capability": capability,
        "global_rates_retuned": False,
        "base_extension_multipliers_retuned": False,
        "local_context_relation_retuned": False,
        "geometry_used": False,
        "zeta_used": False,
        "replication_used": False,
        "predictive_validation_claim": False,
    }
    return rows, summary
