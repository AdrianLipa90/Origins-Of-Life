#!/usr/bin/env python3
"""Run the preregistered first-RNA candidate-length causal gate v0.2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.first_rna_candidate_v02 import (
    CandidateRunConfig,
    run_candidate_gate,
)


PREREG_PATH = "docs/FIRST_RNA_CANDIDATE_PREREG_V0_2.md"


def main() -> int:
    seeds = list(range(301, 321))
    config = CandidateRunConfig()
    runs, summary = run_candidate_gate(seeds=seeds, config=config)

    out = Path("first_rna_candidate_v02")
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "first_rna_candidate_v02_runs.csv"
    receipt_path = out / "first_rna_candidate_v02_receipt.json"
    runs.to_csv(runs_path, index=False)

    receipt = {
        **summary,
        "preregistration": PREREG_PATH,
        "hours": config.hours,
        "dt_h": config.dt_h,
        "temp_C": config.temp_C,
        "pH": config.pH,
        "k_catalysis": config.k_catalysis,
        "concentration_boost": config.concentration_boost,
        "drying_cycle_h": config.drying_cycle_h,
        "drying_fraction": config.drying_fraction,
        "initial_nucleotide_units": config.initial_nucleotide_units,
        "runs_csv": str(runs_path),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(runs.to_string(index=False))
    print()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
