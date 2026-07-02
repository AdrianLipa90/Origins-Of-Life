#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.analysis.sweep import run_topo_sweep
from origins.orbital.repository_assignment import build_repository_system_state
from origins.orbital_formalism import build_orbital_repository_snapshot
from origins.scenarios import ALL_SCENARIOS, SCENARIOS_BY_CODE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run topo sweep with orbital exports")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Scenario codes, default: all")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--hours", type=float, default=60.0)
    parser.add_argument("--outdir", type=str, default="outputs_orbital_sweep")
    parser.add_argument("--enable-multiproc", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.scenarios:
        scenarios = [SCENARIOS_BY_CODE[c] for c in args.scenarios]
    else:
        scenarios = ALL_SCENARIOS

    os.makedirs(args.outdir, exist_ok=True)
    df = run_topo_sweep(
        scenarios=scenarios,
        Nx=args.nx,
        Ny=args.ny,
        dt_h=args.dt,
        hours=args.hours,
        outdir=args.outdir,
        enable_multiproc=args.enable_multiproc,
        workers=args.workers,
    )

    system_state = build_repository_system_state(scenarios)
    with open(os.path.join(args.outdir, "orbital_system_state.json"), "w", encoding="utf-8") as f:
        json.dump(system_state.to_dict(), f, indent=2, ensure_ascii=False)

    snapshot = build_orbital_repository_snapshot(scenarios)
    with open(os.path.join(args.outdir, "orbital_repository_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
