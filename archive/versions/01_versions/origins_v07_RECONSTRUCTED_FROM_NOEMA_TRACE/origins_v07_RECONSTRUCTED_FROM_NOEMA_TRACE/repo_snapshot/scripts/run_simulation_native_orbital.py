#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.scenarios import SCENARIOS_BY_CODE
from origins.simulator.universal_orbital import OrbitalUniversalOriginSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native orbital simulator subclass")
    parser.add_argument("--scenario", type=str, default="A")
    parser.add_argument("--hours", type=float, default=120.0)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--outdir", type=str, default="outputs_native_orbital")
    parser.add_argument("--prefix", type=str, default="final")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = SCENARIOS_BY_CODE[args.scenario]
    sim = OrbitalUniversalOriginSimulator(cfg, Nx=args.nx, Ny=args.ny, dt_h=args.dt, outdir=args.outdir)
    sim.initialize()
    sim.run(hours=args.hours, record_interval=2.0, verbose=not args.quiet, orbital_export=False, orbital_prefix=args.prefix)
    sim.save_outputs(prefix=args.prefix, export_orbital=True)
    print(sim.build_orbital_bundle(delta_t=max(args.hours, sim.t_h), prefix=args.prefix).to_dict())


if __name__ == "__main__":
    main()
