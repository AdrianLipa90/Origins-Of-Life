import argparse
import os
import time

import numpy as np

from .runners import quick_test_run, run_all_scenarios
from .scenarios import ALL_SCENARIOS
from .sweep_v3 import run_sweep_v3_and_save
from .sweeps import run_topo_param_sweep
from .utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Systematized origin-of-life runtime")
    parser.add_argument("--mode", choices=["sweep_v3", "scenarios", "both", "topo_sweep", "quick_test"], default="both")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--hours", type=float, default=120.0)
    parser.add_argument("--synth_min", type=float, default=0.001)
    parser.add_argument("--synth_max", type=float, default=0.20)
    parser.add_argument("--synth_steps", type=int, default=6)
    parser.add_argument("--topo_min", type=float, default=0.0)
    parser.add_argument("--topo_max", type=float, default=0.5)
    parser.add_argument("--topo_steps", type=int, default=6)
    parser.add_argument("--enable-multiproc", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    if args.mode in ("sweep_v3", "both"):
        run_sweep_v3_and_save(outdir=os.path.join(args.outdir, "sweep_v3"), Nx=args.nx, Ny=args.ny, dt_h=args.dt)
    if args.mode in ("scenarios", "both"):
        run_all_scenarios(ALL_SCENARIOS, Nx=args.nx, Ny=args.ny, dt_h=args.dt, base_hours=args.hours, outdir=os.path.join(args.outdir, "scenarios"))
    if args.mode == "topo_sweep":
        synth_list = list(np.linspace(args.synth_min, args.synth_max, args.synth_steps))
        topo_list = list(np.linspace(args.topo_min, args.topo_max, args.topo_steps))
        run_topo_param_sweep(ALL_SCENARIOS, topo_list=topo_list, synth_list=synth_list, Nx=args.nx, Ny=args.ny, dt_h=args.dt, hours=args.hours, outdir=os.path.join(args.outdir, "topo_sweep"), enable_multiproc=args.enable_multiproc, workers=args.workers)
    if args.mode == "quick_test":
        quick_test_run(outdir=os.path.join(args.outdir, "quick_test"), hours=args.hours, Nx=min(64, args.nx), Ny=min(64, args.ny))


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nSystematized runtime finished in {end - start:.1f}s")
