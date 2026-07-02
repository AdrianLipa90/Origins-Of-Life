from __future__ import annotations

import json
import os
from typing import List, Optional

import pandas as pd

from ..orbital.repository_assignment import build_repository_system_state
from ..orbital_formalism import build_orbital_repository_snapshot
from ..scenarios import ALL_SCENARIOS, ScenarioConfig
from ..simulator.universal_orbital import OrbitalUniversalOriginSimulator
from .sweep import run_sweep_v3, run_topo_sweep


def _ensure(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def export_orbital_analysis_snapshot(
    scenarios: List[ScenarioConfig],
    outdir: str,
    filename_prefix: str = "orbital",
) -> dict[str, str]:
    _ensure(outdir)
    system_state = build_repository_system_state(scenarios)
    repo_snapshot = build_orbital_repository_snapshot(scenarios)

    system_state_path = os.path.join(outdir, f"{filename_prefix}_system_state.json")
    repo_snapshot_path = os.path.join(outdir, f"{filename_prefix}_repository_snapshot.json")

    with open(system_state_path, "w", encoding="utf-8") as f:
        json.dump(system_state.to_dict(), f, indent=2, ensure_ascii=False)
    with open(repo_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(repo_snapshot.to_dict(), f, indent=2, ensure_ascii=False)

    return {
        "system_state": system_state_path,
        "repository_snapshot": repo_snapshot_path,
    }


def run_sweep_v3_orbital(
    outdir: str = "sweep_v3_orbital_outputs",
    Nx: int = 96,
    Ny: int = 96,
    dt_h: float = 0.05,
) -> pd.DataFrame:
    _ensure(outdir)
    df = run_sweep_v3(outdir=outdir, Nx=Nx, Ny=Ny, dt_h=dt_h)
    export_orbital_analysis_snapshot(ALL_SCENARIOS, outdir=outdir, filename_prefix="sweep_v3_orbital")
    return df


def run_topo_sweep_orbital(
    scenarios: Optional[List[ScenarioConfig]] = None,
    Nx: int = 64,
    Ny: int = 64,
    dt_h: float = 0.05,
    hours: float = 60.0,
    outdir: str = "topo_sweep_orbital_outputs",
    enable_multiproc: bool = False,
    workers: int = 4,
) -> pd.DataFrame:
    if scenarios is None:
        scenarios = ALL_SCENARIOS
    _ensure(outdir)
    df = run_topo_sweep(
        scenarios=scenarios,
        Nx=Nx,
        Ny=Ny,
        dt_h=dt_h,
        hours=hours,
        outdir=outdir,
        enable_multiproc=enable_multiproc,
        workers=workers,
    )
    export_orbital_analysis_snapshot(scenarios, outdir=outdir, filename_prefix="topo_sweep_orbital")
    return df


def run_all_scenarios_orbital(
    scenarios: Optional[List[ScenarioConfig]] = None,
    Nx: int = 96,
    Ny: int = 96,
    dt_h: float = 0.05,
    base_hours: float = 120.0,
    outdir: str = "outputs_orbital_native_analysis",
) -> pd.DataFrame:
    if scenarios is None:
        scenarios = ALL_SCENARIOS
    _ensure(outdir)

    summaries = []
    bundles = []
    for cfg in scenarios:
        sim = OrbitalUniversalOriginSimulator(cfg, Nx=Nx, Ny=Ny, dt_h=dt_h, outdir=outdir)
        sim.initialize()
        hours = base_hours if cfg.code != "D" else max(base_hours, 500.0)
        sim.run(hours=hours, record_interval=2.0, verbose=True, orbital_export=False)
        sim.save_outputs(prefix="final", export_orbital=True)
        summaries.append({
            "Scenario": cfg.code,
            "Name": cfg.name,
            "Final_Polymers": sim.rna_population.size,
            "Final_ProtoC": sim.protocell_count,
            "Expected_ProtoC": cfg.expected_protocells,
        })
        bundles.append(sim.build_orbital_bundle(delta_t=max(hours, sim.t_h), prefix="final").to_dict())

    combined = pd.DataFrame(summaries)
    combined.to_csv(os.path.join(outdir, "ALL_SCENARIOS_ORBITAL_SUMMARY.csv"), index=False)
    with open(os.path.join(outdir, "ALL_SCENARIOS_ORBITAL_BUNDLES.json"), "w", encoding="utf-8") as f:
        json.dump(bundles, f, indent=2, ensure_ascii=False)
    export_orbital_analysis_snapshot(scenarios, outdir=outdir, filename_prefix="all_scenarios_orbital")
    return combined
