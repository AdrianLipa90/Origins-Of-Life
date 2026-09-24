#!/usr/bin/env python3
"""Run first-RNA activator-context capability audit v0.10."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.first_rna_activator_context_v10 import (
    EMPIRICAL_3P5_BY_ACTIVATOR,
    SOURCE_DOI,
    run_activator_context_audit_v10,
)


def main() -> int:
    rows, summary = run_activator_context_audit_v10()

    out = Path("first_rna_activator_context_v10")
    out.mkdir(parents=True, exist_ok=True)

    rows_path = out / "first_rna_activator_context_v10_rows.csv"
    receipt_path = out / "first_rna_activator_context_v10_receipt.json"

    rows.to_csv(rows_path, index=False)

    receipt = {
        **summary,
        "rows_csv": str(rows_path),
        "empirical_3p5_by_activator": EMPIRICAL_3P5_BY_ACTIVATOR,
        "validation_type": "STRUCTURAL_CAPABILITY_AUDIT",
        "analyst_blind": False,
        "physical_rate_claim": False,
        "parent_result": "PASS_LOCAL_CONTEXT_REPRESENTATION_CALIBRATION",
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
