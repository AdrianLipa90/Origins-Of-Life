#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.api import (
    run_all_scenarios_orbital,
    run_sweep_v3_orbital,
    run_topo_sweep_orbital,
)
from origins.scenarios import ALL_SCENARIOS, SCENARIOS_BY_CODE


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified native orbital analysis runner")
    parser.add_argument(
        "--mode",
        choices=["sweep_v3", "topo_sweep", "all_scenarios"],
        default="topo_sweep",
    )
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--hours", type=float, default=60.0)
    parser.add_argument("--outdir", type=str, default="outputs_native_orbital_analysis")
    parser.add_argument("--enable-multiproc", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.scenarios:
        scenarios = [SCENARIOS_BY_CODE[c] for c in args.scenarios]
    else:
        scenarios = ALL_SCENARIOS

    if args.mode == "sweep_v3":
        df = run_sweep_v3_orbital(outdir=args.outdir, Nx=args.nx, Ny=args.ny, dt_h=args.dt)
    elif args.mode == "all_scenarios":
        df = run_all_scenarios_orbital(
            scenarios=scenarios,
            Nx=max(args.nx, 96),
            Ny=max(args.ny, 96),
            dt_h=args.dt,
            base_hours=args.hours,
            outdir=args.outdir,
        )
    else:
        df = run_topo_sweep_orbital(
            scenarios=scenarios,
            Nx=args.nx,
            Ny=args.ny,
            dt_h=args.dt,
            hours=args.hours,
            outdir=args.outdir,
            enable_multiproc=args.enable_multiproc,
            workers=args.workers,
        )

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
