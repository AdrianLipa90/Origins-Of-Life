#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from origins.orbital.runtime_bridge import OrbitalRuntimeBridge
from origins.scenarios import ALL_SCENARIOS, SCENARIOS_BY_CODE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a scenario with orbital runtime bridge")
    parser.add_argument("--scenario", type=str, default="A", help="Scenario code, e.g. A B C D E")
    parser.add_argument("--hours", type=float, default=120.0)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--outdir", type=str, default="outputs_orbital")
    parser.add_argument("--prefix", type=str, default="final")
    parser.add_argument("--no-clay", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = SCENARIOS_BY_CODE[args.scenario]
    bridge = OrbitalRuntimeBridge.from_config(
        cfg,
        Nx=args.nx,
        Ny=args.ny,
        dt_h=args.dt,
        outdir=args.outdir,
        include_clay=not args.no_clay,
    )
    bundle = bridge.run(
        hours=args.hours,
        record_interval=2.0,
        verbose=not args.quiet,
        save_outputs=True,
        prefix=args.prefix,
        export_orbital=True,
    )
    print(json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
