#!/usr/bin/env python3
"""Run the matched 2x2 geometry/zeta ablation and write tabular receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.ablation import run_matched_factorial_ablation
from origins.scenarios import SCENARIOS_BY_CODE


def _parse_seeds(raw: str | None, default: int) -> list[int]:
    if raw is None:
        return [int(default)]
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Matched 2x2 ablation: chemistry / zeta / geometry / full"
    )
    ap.add_argument("--scenario", choices=sorted(SCENARIOS_BY_CODE), default="A")
    ap.add_argument("--seeds", default=None, help="comma-separated integer seeds")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--dt-h", type=float, default=0.05)
    ap.add_argument("--outdir", default="outputs_ablation")
    ap.add_argument("--polymer-free", action="store_true")
    ap.add_argument("--no-clay", action="store_true")
    ns = ap.parse_args()

    cfg = SCENARIOS_BY_CODE[ns.scenario]
    seeds = _parse_seeds(ns.seeds, cfg.seed)

    runs, effects = run_matched_factorial_ablation(
        cfg,
        seeds=seeds,
        Nx=ns.nx,
        Ny=ns.ny,
        dt_h=ns.dt_h,
        hours=ns.hours,
        include_clay=not ns.no_clay,
        preseed_rna=not ns.polymer_free,
        outdir=ns.outdir,
    )

    root = Path(ns.outdir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"scenario_{cfg.code}_matched_factorial"
    runs_path = root / f"{stem}_runs.csv"
    effects_path = root / f"{stem}_effects.csv"
    receipt_path = root / f"{stem}_receipt.json"

    runs.to_csv(runs_path, index=False)
    effects.to_csv(effects_path, index=False)

    receipt = {
        "schema": "ORIGINS_MATCHED_FACTORIAL_ABLATION_V0_1",
        "scenario": cfg.code,
        "seeds": seeds,
        "grid": [ns.nx, ns.ny],
        "dt_h": ns.dt_h,
        "hours": ns.hours,
        "preseed_rna": not ns.polymer_free,
        "include_clay": not ns.no_clay,
        "arms": ["chemistry_only", "zeta_only", "geometry_only", "full"],
        "initial_state_digests": sorted(runs["initial_state_digest"].unique()),
        "physical_binding_claim": False,
        "runs_csv": str(runs_path),
        "effects_csv": str(effects_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    print(runs.to_string(index=False))
    print()
    print(effects.to_string(index=False))
    print()
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
