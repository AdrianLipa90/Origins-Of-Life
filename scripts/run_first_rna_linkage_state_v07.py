#!/usr/bin/env python3
"""Run explicit first-RNA linkage-state test v0.7."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.first_rna_linkage_state_v07 import (
    DIMER_3P5_FRACTION,
    SOURCE_DOI,
    TRIMER_3P5_FRACTION_OBSERVED,
    run_linkage_state_v07,
)


def main() -> int:
    rows, summary = run_linkage_state_v07()

    out = Path("first_rna_linkage_state_v07")
    out.mkdir(parents=True, exist_ok=True)

    rows_path = out / "first_rna_linkage_state_v07_rows.csv"
    receipt_path = out / "first_rna_linkage_state_v07_receipt.json"

    rows.to_csv(rows_path, index=False)

    receipt = {
        **summary,
        "rows_csv": str(rows_path),
        "source_doi": SOURCE_DOI,
        "dimer_calibration_3p5_fraction": DIMER_3P5_FRACTION,
        "trimer_holdout_3p5_fraction": TRIMER_3P5_FRACTION_OBSERVED,
        "validation_type": "CALIBRATION_PLUS_PARAMETER_HELD_OUT_POINT_ESTIMATE",
        "analyst_blind": False,
        "physical_rate_claim": False,
        "statistical_significance_claim": False,
        "parent_result": "FAIL_MISSING_LINKAGE_STATE",
    }

    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(rows.to_string(index=False))
    print()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
