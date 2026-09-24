#!/usr/bin/env python3
"""Run the first-RNA linkage/regioselectivity holdout v0.6."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.first_rna_linkage_holdout_v06 import (
    EMPIRICAL_3P5_FRACTION,
    SOURCE_DOI,
    run_linkage_holdout_v06,
)


def main() -> int:
    rows, summary = run_linkage_holdout_v06()

    out = Path("first_rna_linkage_holdout_v06")
    out.mkdir(parents=True, exist_ok=True)

    rows_path = out / "first_rna_linkage_holdout_v06_rows.csv"
    receipt_path = out / "first_rna_linkage_holdout_v06_receipt.json"

    rows.to_csv(rows_path, index=False)

    receipt = {
        **summary,
        "rows_csv": str(rows_path),
        "empirical_3p5_fraction": EMPIRICAL_3P5_FRACTION,
        "source_doi": SOURCE_DOI,
        "validation_type": "PARAMETER_HELD_OUT_STRUCTURAL_CAPABILITY",
        "analyst_blind": False,
        "physical_rate_claim": False,
        "predictive_numerical_claim": False,
        "parent_result": "PASS_BASE_RESOLVED_CALIBRATION",
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
