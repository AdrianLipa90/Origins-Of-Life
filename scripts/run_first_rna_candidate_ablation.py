#!/usr/bin/env python3
"""Run first-RNA polymerase-size candidate ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.first_rna_candidate import run_first_rna_candidate_ablation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="101,102,103,104,105,106,107,108,109,110")
    ap.add_argument("--hours", type=float, default=500.0)
    ap.add_argument("--dt-h", type=float, default=0.5)
    ap.add_argument("--outdir", default="first_rna_candidate_ablation")
    ns = ap.parse_args()
    seeds = [int(x.strip()) for x in ns.seeds.split(",") if x.strip()]

    df = run_first_rna_candidate_ablation(
        seeds=seeds,
        hours=ns.hours,
        dt_h=ns.dt_h,
    )

    root = Path(ns.outdir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "first_rna_candidate_runs.csv"
    receipt_path = root / "first_rna_candidate_receipt.json"
    df.to_csv(csv_path, index=False)

    receipt = {
        "schema": "ORIGINS_FIRST_RNA_CANDIDATE_ABLATION_V0_1",
        "seeds": seeds,
        "hours": ns.hours,
        "dt_h": ns.dt_h,
        "replicator_mode": "candidate_only",
        "modes": ["chemistry_only", "legacy_geometry_candidate"],
        "functional_replication_claim": False,
        "candidate_reference_nt": 45,
        "csv": str(csv_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    print(df.to_string(index=False))
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
