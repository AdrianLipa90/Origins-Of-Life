#!/usr/bin/env python3
"""Run base-resolved first-RNA calibration v0.5."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.first_rna_base_resolved_v05 import (
    BASE_EXTENSION_MULTIPLIER,
    BASE_HOURS,
    BASE_TAIL_WINDOW,
    run_base_resolved_calibration_v05,
)


def main() -> int:
    rows, summary = run_base_resolved_calibration_v05()
    out = Path("first_rna_base_resolved_v05")
    out.mkdir(parents=True, exist_ok=True)

    rows_path = out / "first_rna_base_resolved_v05_rows.csv"
    receipt_path = out / "first_rna_base_resolved_v05_receipt.json"
    rows.to_csv(rows_path, index=False)

    receipt = {
        **summary,
        "rows_csv": str(rows_path),
        "base_extension_multiplier": BASE_EXTENSION_MULTIPLIER,
        "base_hours": BASE_HOURS,
        "base_tail_window": {
            key: list(value) if value is not None else None
            for key, value in BASE_TAIL_WINDOW.items()
        },
        "physical_rate_claim": False,
        "predictive_validation_claim": False,
        "parent_holdout_failure": "FAIL_BASE_SPECIFIC_HOLDOUT",
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
