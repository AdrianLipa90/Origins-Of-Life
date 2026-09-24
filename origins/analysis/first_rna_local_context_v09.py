"""Symbolic local linkage context for first-RNA v0.9.

This layer repairs the representation gap exposed by v0.8. It records a
published categorical sequence/regioselectivity ordering without assigning
arbitrary numerical rates or probabilities.

v0.9 is representation calibration, not predictive validation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import pandas as pd


BASE_CLASSES = ("Pu", "Py")
LINKAGES = ("3p5", "2p5")
SOURCE_DOI = "10.1021/ja034328e"

# Explicitly empty: v0.9 introduces no fitted numeric local-context parameters.
NUMERIC_CONTEXT_PARAMETERS: dict[str, float] = {}


def _validate_base_class(value: str) -> None:
    if value not in BASE_CLASSES:
        raise ValueError(f"unsupported base class: {value!r}")


def _validate_linkage(value: str) -> None:
    if value not in LINKAGES:
        raise ValueError(f"unsupported linkage: {value!r}")


@dataclass(frozen=True)
class DimerContext:
    left_class: str
    linkage: str
    right_class: str

    def __post_init__(self) -> None:
        _validate_base_class(self.left_class)
        _validate_linkage(self.linkage)
        _validate_base_class(self.right_class)

    @property
    def label(self) -> str:
        return f"{self.left_class}-{self.linkage}-{self.right_class}"


@dataclass(frozen=True)
class GrowthContext:
    terminal_class: str
    previous_linkage: str
    incoming_class: str

    def __post_init__(self) -> None:
        _validate_base_class(self.terminal_class)
        _validate_linkage(self.previous_linkage)
        _validate_base_class(self.incoming_class)


PU_3P5_PY = DimerContext("Pu", "3p5", "Py")
PU_3P5_PU = DimerContext("Pu", "3p5", "Pu")
PU_2P5_PY = DimerContext("Pu", "2p5", "Py")
PU_2P5_PU = DimerContext("Pu", "2p5", "Pu")

# High to low. Tuple membership expresses equality within a level.
SOURCE_ORDER_LEVELS: tuple[tuple[DimerContext, ...], ...] = (
    (PU_3P5_PY,),
    (PU_3P5_PU, PU_2P5_PY),
    (PU_2P5_PU,),
)


def ordinal_relation(left: DimerContext, right: DimerContext) -> str:
    level_of: dict[DimerContext, int] = {}
    for level_index, level in enumerate(SOURCE_ORDER_LEVELS):
        for context in level:
            level_of[context] = level_index

    if left not in level_of or right not in level_of:
        return "UNCONSTRAINED"

    if level_of[left] < level_of[right]:
        return "GREATER"
    if level_of[left] > level_of[right]:
        return "LESS"
    return "EQUAL"


def run_local_context_calibration_v09() -> tuple[pd.DataFrame, dict[str, object]]:
    contexts = [context for level in SOURCE_ORDER_LEVELS for context in level]

    rows = pd.DataFrame(
        [
            {
                "level": level_index,
                "context": context.label,
                "left_class": context.left_class,
                "linkage": context.linkage,
                "right_class": context.right_class,
            }
            for level_index, level in enumerate(SOURCE_ORDER_LEVELS)
            for context in level
        ]
    )

    distinct_ok = len(contexts) == 4 and len(set(contexts)) == 4
    equality_ok = ordinal_relation(PU_3P5_PU, PU_2P5_PY) == "EQUAL"
    strict_top_ok = ordinal_relation(PU_3P5_PY, PU_3P5_PU) == "GREATER"
    strict_bottom_ok = ordinal_relation(PU_2P5_PY, PU_2P5_PU) == "GREATER"

    growth_fields = tuple(field.name for field in fields(GrowthContext))
    growth_context_ok = growth_fields == (
        "terminal_class",
        "previous_linkage",
        "incoming_class",
    )
    chain_length_absent = "chain_length" not in growth_fields
    no_numeric_parameters = NUMERIC_CONTEXT_PARAMETERS == {}

    passed = bool(
        distinct_ok
        and equality_ok
        and strict_top_ok
        and strict_bottom_ok
        and growth_context_ok
        and chain_length_absent
        and no_numeric_parameters
    )

    summary = {
        "schema": "ORIGINS_FIRST_RNA_LOCAL_CONTEXT_V0_9",
        "status": (
            "PASS_LOCAL_CONTEXT_REPRESENTATION_CALIBRATION"
            if passed
            else "FAIL_LOCAL_CONTEXT_REPRESENTATION_CALIBRATION"
        ),
        "source_doi": SOURCE_DOI,
        "representation_calibration_not_prediction": True,
        "distinct_source_contexts": distinct_ok,
        "source_equality_preserved": equality_ok,
        "source_strict_inequalities_preserved": bool(
            strict_top_ok and strict_bottom_ok
        ),
        "growth_context_fields": growth_fields,
        "growth_context_complete": growth_context_ok,
        "chain_length_term_present": not chain_length_absent,
        "numeric_context_parameters": dict(NUMERIC_CONTEXT_PARAMETERS),
        "numeric_context_parameter_added": not no_numeric_parameters,
        "global_rates_retuned": False,
        "base_extension_multipliers_retuned": False,
        "geometry_used": False,
        "zeta_used": False,
        "replication_used": False,
        "predictive_validation_claim": False,
    }
    return rows, summary
