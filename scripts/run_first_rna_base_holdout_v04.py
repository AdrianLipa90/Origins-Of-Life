#!/usr/bin/env python3
"""Run held-out base-specific validation v0.4."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.first_rna_base_holdout_v04 import (
    run_base_specific_holdout_v04,
)


def main() -> int:
    result = run_base_specific_holdout_v04()
    out = Path("first_rna_base_holdout_v04")
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "first_rna_base_holdout_v04_receipt.json"
    receipt_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
