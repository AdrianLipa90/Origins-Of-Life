from __future__ import annotations

import json
from pathlib import Path

from origins import SCENARIOS_BY_CODE
from origins.simulator.universal import UniversalOriginSimulator


def run_smoke(code: str, hours: float = 4.0, nx: int = 32, ny: int = 32, dt: float = 0.10) -> dict:
    cfg = SCENARIOS_BY_CODE[code]
    sim = UniversalOriginSimulator(cfg, Nx=nx, Ny=ny, dt_h=dt, outdir="results/generated")
    sim.initialize()
    df = sim.run(hours=hours, record_interval=1.0, verbose=False)
    return {
        "scenario": cfg.code,
        "scenario_name": cfg.name,
        "final_time_h": float(df["time_h"].iloc[-1]) if not df.empty else 0.0,
        "final_mean_R": float(df["mean_R"].iloc[-1]) if not df.empty else 0.0,
        "final_mean_M": float(df["mean_M"].iloc[-1]) if not df.empty else 0.0,
        "final_n_polymers": int(df["n_polymers"].iloc[-1]) if not df.empty else 0,
        "final_n_protocells": int(df["n_protocells"].iloc[-1]) if not df.empty else 0,
    }


def main() -> None:
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    summaries = [run_smoke("A"), run_smoke("E")]
    with open(out / "smoke_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
