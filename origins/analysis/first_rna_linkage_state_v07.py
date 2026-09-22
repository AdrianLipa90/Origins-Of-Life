"""Minimal explicit phosphodiester-linkage state for first-RNA v0.7.

The model is intentionally zero-order: it calibrates a 3',5' bond fraction on
the reported binary A/U dimer point estimate and carries that fraction forward
unchanged to the trimer. The trimer is then checked without retuning.

This is a point-estimate/rounding compatibility test, not a statistical
significance test.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SOURCE_DOI = "10.1089/ast.2007.0138"
DIMER_3P5_FRACTION = 0.51
TRIMER_3P5_FRACTION_OBSERVED = 0.61


@dataclass(frozen=True)
class LinkageState:
    bond_weight_3p5: float
    bond_weight_2p5: float

    @classmethod
    def from_3p5_fraction(cls, fraction: float) -> "LinkageState":
        p = float(fraction)
        if not 0.0 <= p <= 1.0:
            raise ValueError("3p5 fraction must lie in [0,1]")
        return cls(bond_weight_3p5=p, bond_weight_2p5=1.0 - p)

    @property
    def total_bond_weight(self) -> float:
        return float(self.bond_weight_3p5 + self.bond_weight_2p5)

    @property
    def fraction_3p5(self) -> float:
        total = self.total_bond_weight
        if total <= 0.0:
            raise ZeroDivisionError("total bond weight must be positive")
        return float(self.bond_weight_3p5 / total)


def rounded_percent_interval(percent: int) -> tuple[float, float]:
    value = int(percent)
    if not 0 <= value <= 100:
        raise ValueError("percent must lie in [0,100]")
    return ((value - 0.5) / 100.0, (value + 0.5) / 100.0)


def intervals_overlap(
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    return bool(max(left[0], right[0]) < min(left[1], right[1]))


def predict_length_invariant_linkage(chain_length: int) -> LinkageState:
    if int(chain_length) < 2:
        raise ValueError("chain_length must be >=2")
    return LinkageState.from_3p5_fraction(DIMER_3P5_FRACTION)


def run_linkage_state_v07() -> tuple[pd.DataFrame, dict[str, object]]:
    dimer = predict_length_invariant_linkage(2)
    trimer = predict_length_invariant_linkage(3)

    dimer_interval = rounded_percent_interval(51)
    trimer_interval = rounded_percent_interval(61)
    overlap = intervals_overlap(dimer_interval, trimer_interval)

    rows = pd.DataFrame(
        [
            {
                "chain_length": 2,
                "role": "CALIBRATION",
                "observed_3p5_fraction": DIMER_3P5_FRACTION,
                "predicted_3p5_fraction": dimer.fraction_3p5,
                "delta": dimer.fraction_3p5 - DIMER_3P5_FRACTION,
            },
            {
                "chain_length": 3,
                "role": "PARAMETER_HELD_OUT",
                "observed_3p5_fraction": TRIMER_3P5_FRACTION_OBSERVED,
                "predicted_3p5_fraction": trimer.fraction_3p5,
                "delta": trimer.fraction_3p5 - TRIMER_3P5_FRACTION_OBSERVED,
            },
        ]
    )

    verdict = (
        "PASS_LENGTH_INVARIANT_LINKAGE_POINT_ESTIMATE"
        if overlap
        else "FAIL_LENGTH_INVARIANT_LINKAGE_POINT_ESTIMATE"
    )

    summary = {
        "schema": "ORIGINS_FIRST_RNA_LINKAGE_STATE_V0_7",
        "status": verdict,
        "source_doi": SOURCE_DOI,
        "explicit_linkage_state": True,
        "calibration_3p5_fraction": DIMER_3P5_FRACTION,
        "trimer_predicted_3p5_fraction": trimer.fraction_3p5,
        "trimer_observed_3p5_fraction": TRIMER_3P5_FRACTION_OBSERVED,
        "trimer_delta": trimer.fraction_3p5 - TRIMER_3P5_FRACTION_OBSERVED,
        "dimer_rounding_interval": dimer_interval,
        "trimer_rounding_interval": trimer_interval,
        "rounding_intervals_overlap": overlap,
        "statistical_significance_claim": False,
        "global_rates_retuned": False,
        "base_extension_multipliers_retuned": False,
        "sequence_pair_term_added": False,
        "chain_length_linkage_term_added": False,
        "geometry_used": False,
        "zeta_used": False,
        "replication_used": False,
    }
    return rows, summary
