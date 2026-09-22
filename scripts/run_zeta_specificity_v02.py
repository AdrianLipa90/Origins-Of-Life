#!/usr/bin/env python3
"""Run preregistered zeta specificity v0.2 blocks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.zeta_specificity_v02 import (
    V02_MODES,
    ZetaSpecificityV02Modulator,
    run_zeta_specificity_v02,
)
from origins.scenarios import SCENARIOS_BY_CODE


PREREG_COMMIT = "7999001817d4fe461fda79e100ed803a4d340bd9"


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Preregistered zeta spectral-specificity v0.2"
    )
    ap.add_argument("--scenario", choices=["A", "C"], required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--outdir", required=True)
    ns = ap.parse_args()

    cfg = SCENARIOS_BY_CODE[ns.scenario]
    seeds = _parse_seeds(ns.seeds)
    runs, diagnostics = run_zeta_specificity_v02(
        cfg,
        seeds=seeds,
        Nx=32,
        Ny=32,
        dt_h=0.05,
        hours=120.0,
        outdir=ns.outdir,
    )

    root = Path(ns.outdir)
    root.mkdir(parents=True, exist_ok=True)
    seed_tag = f"{min(seeds)}-{max(seeds)}"
    stem = f"scenario_{cfg.code}_zeta_specificity_v02_{seed_tag}"
    runs_path = root / f"{stem}_runs.csv"
    diagnostics_path = root / f"{stem}_mask_diagnostics.csv"
    receipt_path = root / f"{stem}_receipt.json"

    runs.to_csv(runs_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    receipt = {
        "schema": "ORIGINS_ZETA_SPECTRAL_SPECIFICITY_V0_2_BLOCK",
        "prereg_commit_sha": PREREG_COMMIT,
        "scenario": cfg.code,
        "seeds": seeds,
        "hours": 120.0,
        "grid": [32, 32],
        "dt_h": 0.05,
        "modes": list(V02_MODES),
        "initial_state_digests": sorted(runs["initial_state_digest"].unique()),
        "physical_binding_claim": False,
        "zero_specificity_claim": False,
        "runs_csv": str(runs_path),
        "mask_diagnostics_csv": str(diagnostics_path),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(runs.to_string(index=False))
    print()
    print(diagnostics.to_string(index=False))
    print()
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
