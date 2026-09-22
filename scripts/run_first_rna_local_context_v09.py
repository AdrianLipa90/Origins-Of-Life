#!/usr/bin/env python3
"""Run symbolic first-RNA local linkage-context calibration v0.9."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.first_rna_local_context_v09 import (
    SOURCE_DOI,
    run_local_context_calibration_v09,
)


def main() -> int:
    rows, summary = run_local_context_calibration_v09()

    out = Path("first_rna_local_context_v09")
    out.mkdir(parents=True, exist_ok=True)

    rows_path = out / "first_rna_local_context_v09_rows.csv"
    receipt_path = out / "first_rna_local_context_v09_receipt.json"

    rows.to_csv(rows_path, index=False)

    receipt = {
        **summary,
        "rows_csv": str(rows_path),
        "source_doi": SOURCE_DOI,
        "validation_type": "REPRESENTATION_CALIBRATION",
        "analyst_blind": False,
        "physical_rate_claim": False,
        "parent_result": "FAIL_UNIVERSAL_STAGE_SHIFT",
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
