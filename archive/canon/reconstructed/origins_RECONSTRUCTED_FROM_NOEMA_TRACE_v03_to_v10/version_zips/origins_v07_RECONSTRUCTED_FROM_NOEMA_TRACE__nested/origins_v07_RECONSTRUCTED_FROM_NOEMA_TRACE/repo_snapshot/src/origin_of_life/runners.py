import os

import pandas as pd

from .scenarios import ALL_SCENARIOS
from .simulator import UniversalOriginSimulator
from .utils import ensure_dir


def run_all_scenarios(all_scenarios=ALL_SCENARIOS, Nx=96, Ny=96, dt_h=0.05, base_hours=120.0, outdir="sim_outputs"):
    ensure_dir(outdir)
    overall = []
    for cfg in all_scenarios:
        sim = UniversalOriginSimulator(cfg, Nx=Nx, Ny=Ny, dt_h=dt_h, outdir=outdir)
        sim.initialize()
        hours = base_hours if cfg.code != "D" else max(base_hours, 500.0)
        sim.run(hours=hours, record_interval=2.0, verbose=True)
        sim.plot_heatmaps(prefix="final")
        summary_df = sim.save_summary()
        overall.append(summary_df)
    combined = pd.concat(overall, ignore_index=True)
    combined.to_csv(os.path.join(outdir, "ALL_5_SCENARIOS_SUMMARY.csv"), index=False)
    return combined


def quick_test_run(outdir="quick_test", hours=20.0, Nx=64, Ny=64):
    ensure_dir(outdir)
    for cfg in ALL_SCENARIOS:
        subdir = os.path.join(outdir, f"scenario_{cfg.code}")
        ensure_dir(subdir)
        sim = UniversalOriginSimulator(cfg, Nx=Nx, Ny=Ny, dt_h=0.05, outdir=subdir)
        sim.initialize()
        sim.run(hours=hours, record_interval=2.0, verbose=True)
        sim.plot_heatmaps(prefix="quick")
        sim.save_summary()
