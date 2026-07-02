#!/usr/bin/env python3
"""
Run a single scenario simulation.

Examples
--------
    python scripts/run_simulation.py --scenario A --hours 120 --outdir outputs/
    python scripts/run_simulation.py --scenario E --hours 200 --nx 64 --ny 64
    python scripts/run_simulation.py --all --hours 120
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins import UniversalOriginSimulator, SCENARIOS_BY_CODE, ALL_SCENARIOS


def main():
    parser = argparse.ArgumentParser(description="Origin-of-Life Holonomic simulator")
    parser.add_argument("--scenario", choices=list(SCENARIOS_BY_CODE) + ["all"],
                        default="A", help="Scenario code or 'all'")
    parser.add_argument("--all", action="store_true", help="Run all 5 scenarios")
    parser.add_argument("--hours",  type=float, default=120.0)
    parser.add_argument("--nx",     type=int,   default=96)
    parser.add_argument("--ny",     type=int,   default=96)
    parser.add_argument("--dt",     type=float, default=0.05)
    parser.add_argument("--outdir", type=str,   default="outputs")
    parser.add_argument("--no-clay", action="store_true",
                        help="Disable explicit clay mineral objects")
    args = parser.parse_args()

    configs = ALL_SCENARIOS if args.all else [SCENARIOS_BY_CODE[args.scenario]]

    for cfg in configs:
        sim = UniversalOriginSimulator(
            cfg,
            Nx=args.nx,
            Ny=args.ny,
            dt_h=args.dt,
            outdir=args.outdir,
            include_clay=not args.no_clay,
        )
        sim.initialize()
        hours = args.hours if cfg.code != "D" else max(args.hours, 500.0)
        sim.run(hours=hours)
        sim.save_outputs()


if __name__ == "__main__":
    main()
