from __future__ import annotations

import argparse
import json
from pathlib import Path

from origins import SCENARIOS_BY_CODE
from origins.simulator.universal import UniversalOriginSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Planetary first-biology side-app")
    parser.add_argument("--scenario", choices=list(SCENARIOS_BY_CODE.keys()), default="A")
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.10)
    parser.add_argument("--record-interval", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="results/generated")
    args = parser.parse_args()

    cfg = SCENARIOS_BY_CODE[args.scenario]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sim = UniversalOriginSimulator(cfg, Nx=args.nx, Ny=args.ny, dt_h=args.dt, outdir=str(outdir))
    sim.initialize()
    df = sim.run(hours=args.hours, record_interval=args.record_interval, verbose=False)

    summary = {
        "scenario": cfg.code,
        "scenario_name": cfg.name,
        "hours": args.hours,
        "nx": args.nx,
        "ny": args.ny,
        "dt": args.dt,
        "final_time_h": float(df["time_h"].iloc[-1]) if not df.empty else 0.0,
        "final_mean_R": float(df["mean_R"].iloc[-1]) if not df.empty else 0.0,
        "final_mean_M": float(df["mean_M"].iloc[-1]) if not df.empty else 0.0,
        "final_n_polymers": int(df["n_polymers"].iloc[-1]) if not df.empty else 0,
        "final_n_protocells": int(df["n_protocells"].iloc[-1]) if not df.empty else 0,
    }

    with open(outdir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
