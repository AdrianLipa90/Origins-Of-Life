#!/usr/bin/env python3
"""Run matched zeta-specificity controls and write CSV/JSON receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.zeta_specificity import (
    ControlledZetaModulator,
    MODES,
    run_zeta_specificity_controls,
)
from origins.scenarios import SCENARIOS_BY_CODE


def _parse_seeds(raw: str | None, default: int) -> list[int]:
    if raw is None:
        return [int(default)]
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser(description="Matched zeta spectral-specificity controls")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS_BY_CODE), default="A")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--hours", type=float, default=120.0)
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--dt-h", type=float, default=0.05)
    ap.add_argument("--outdir", default="outputs_zeta_specificity")
    ap.add_argument("--geometry-off", action="store_true")
    ns = ap.parse_args()

    cfg = SCENARIOS_BY_CODE[ns.scenario]
    seeds = _parse_seeds(ns.seeds, cfg.seed)
    runs, contrasts = run_zeta_specificity_controls(
        cfg,
        seeds=seeds,
        geometry_enabled=not ns.geometry_off,
        Nx=ns.nx,
        Ny=ns.ny,
        dt_h=ns.dt_h,
        hours=ns.hours,
        outdir=ns.outdir,
    )

    root = Path(ns.outdir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"scenario_{cfg.code}_zeta_specificity"
    runs_path = root / f"{stem}_runs.csv"
    contrasts_path = root / f"{stem}_contrasts.csv"
    receipt_path = root / f"{stem}_receipt.json"
    runs.to_csv(runs_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)

    controls = {}
    for mode in MODES:
        if mode == "none":
            controls[mode] = {"spectral_operator": False, "noise": False}
        else:
            controls[mode] = ControlledZetaModulator(
                mode=mode,
                lambda_soft=cfg.zeta_lambda_soft,
                sigma_heis=cfg.zeta_sigma_heis,
            ).control_metadata((ns.nx, ns.ny))

    receipt = {
        "schema": "ORIGINS_ZETA_SPECTRAL_SPECIFICITY_V0_1",
        "scenario": cfg.code,
        "seeds": seeds,
        "hours": ns.hours,
        "grid": [ns.nx, ns.ny],
        "dt_h": ns.dt_h,
        "geometry_enabled": not ns.geometry_off,
        "modes": list(MODES),
        "initial_state_digests": sorted(runs["initial_state_digest"].unique()),
        "controls": controls,
        "physical_binding_claim": False,
        "zero_specificity_claim": False,
        "runs_csv": str(runs_path),
        "contrasts_csv": str(contrasts_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    print(runs.to_string(index=False))
    print()
    print(contrasts.to_string(index=False))
    print()
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
