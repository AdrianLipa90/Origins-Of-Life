#!/usr/bin/env python3
"""Run first-RNA pair-context transfer holdout v0.8."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.first_rna_pair_context_holdout_v08 import (
    AC_SOURCE_DOI,
    AU_SOURCE_DOI,
    run_pair_context_holdout_v08,
)


def main() -> int:
    rows, summary = run_pair_context_holdout_v08()

    out = Path("first_rna_pair_context_holdout_v08")
    out.mkdir(parents=True, exist_ok=True)

    rows_path = out / "first_rna_pair_context_holdout_v08_rows.csv"
    receipt_path = out / "first_rna_pair_context_holdout_v08_receipt.json"

    rows.to_csv(rows_path, index=False)

    receipt = {
        **summary,
        "rows_csv": str(rows_path),
        "sources": {
            "calibration_AU": AU_SOURCE_DOI,
            "held_out_AC": AC_SOURCE_DOI,
        },
        "validation_type": "CROSS_CONDITION_PAIR_HELD_OUT_TRANSFER",
        "analyst_blind": False,
        "physical_rate_claim": False,
        "statistical_significance_claim": False,
        "parent_result": "FAIL_LENGTH_INVARIANT_LINKAGE_POINT_ESTIMATE",
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
