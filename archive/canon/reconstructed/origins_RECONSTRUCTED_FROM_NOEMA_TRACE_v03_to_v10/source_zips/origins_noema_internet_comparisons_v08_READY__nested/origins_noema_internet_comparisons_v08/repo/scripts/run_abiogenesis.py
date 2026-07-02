#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.abiogenesis.api import (
    create_abiogenesis_runtime,
    run_feasibility_scan,
    run_habitat_scan,
    run_origin_comparison,
)
from origins.scenarios import ALL_SCENARIOS, SCENARIOS_BY_CODE


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical repo-facing abiogenesis runner")
    parser.add_argument(
        "--mode",
        choices=["single", "habitat_scan", "feasibility_scan", "origin_comparison"],
        default="single",
    )
    parser.add_argument("--scenario", type=str, default="A")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--hours", type=float, default=120.0)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--outdir", type=str, default="outputs_abiogenesis")
    parser.add_argument("--enable-multiproc", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.mode == "single":
        cfg = SCENARIOS_BY_CODE[args.scenario]
        sim = create_abiogenesis_runtime(
            config=cfg,
            mode="orbital",
            Nx=args.nx,
            Ny=args.ny,
            dt_h=args.dt,
            outdir=args.outdir,
        )
        sim.initialize()
        sim.run(hours=args.hours, record_interval=2.0, verbose=True, orbital_export=False)
        sim.save_outputs(prefix="final", export_orbital=True)
        print(sim.build_orbital_bundle(delta_t=max(args.hours, sim.t_h), prefix="final").to_dict())
        return

    if args.scenarios:
        scenarios = [SCENARIOS_BY_CODE[c] for c in args.scenarios]
    else:
        scenarios = ALL_SCENARIOS

    if args.mode == "habitat_scan":
        df = run_habitat_scan(outdir=args.outdir, Nx=args.nx, Ny=args.ny, dt_h=args.dt)
    elif args.mode == "origin_comparison":
        df = run_origin_comparison(
            scenarios=scenarios,
            Nx=max(96, args.nx),
            Ny=max(96, args.ny),
            dt_h=args.dt,
            base_hours=args.hours,
            outdir=args.outdir,
        )
    else:
        df = run_feasibility_scan(
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
