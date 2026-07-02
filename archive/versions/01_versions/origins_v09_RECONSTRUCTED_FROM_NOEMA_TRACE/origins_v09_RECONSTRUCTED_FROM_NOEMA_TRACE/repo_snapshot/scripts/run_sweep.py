#!/usr/bin/env python3
"""
Run parameter sweeps.

Examples
--------
    python scripts/run_sweep.py --mode v3 --outdir sweep_v3/
    python scripts/run_sweep.py --mode topo --scenarios A B --workers 4
    python scripts/run_sweep.py --mode topo --enable-multiproc
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from origins import SCENARIOS_BY_CODE, ALL_SCENARIOS
from origins.analysis import run_sweep_v3, run_topo_sweep


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep runner")
    parser.add_argument("--mode", choices=["v3", "topo"], default="v3")
    parser.add_argument("--outdir", type=str, default="sweep_outputs")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Scenario codes to include (default: all)")
    parser.add_argument("--nx",    type=int,   default=64)
    parser.add_argument("--ny",    type=int,   default=64)
    parser.add_argument("--dt",    type=float, default=0.05)
    parser.add_argument("--hours", type=float, default=60.0)
    parser.add_argument("--topo-min",   type=float, default=0.0)
    parser.add_argument("--topo-max",   type=float, default=0.5)
    parser.add_argument("--topo-steps", type=int,   default=6)
    parser.add_argument("--synth-min",  type=float, default=0.001)
    parser.add_argument("--synth-max",  type=float, default=0.20)
    parser.add_argument("--synth-steps",type=int,   default=6)
    parser.add_argument("--enable-multiproc", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.scenarios:
        configs = [SCENARIOS_BY_CODE[c] for c in args.scenarios]
    else:
        configs = ALL_SCENARIOS

    if args.mode == "v3":
        run_sweep_v3(outdir=args.outdir, Nx=args.nx, Ny=args.ny, dt_h=args.dt)
    else:
        topo_list  = list(np.linspace(args.topo_min,  args.topo_max,  args.topo_steps))
        synth_list = list(np.linspace(args.synth_min, args.synth_max, args.synth_steps))
        run_topo_sweep(
            scenarios=configs,
            topo_list=topo_list, synth_list=synth_list,
            Nx=args.nx, Ny=args.ny, dt_h=args.dt, hours=args.hours,
            outdir=args.outdir,
            enable_multiproc=args.enable_multiproc,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
