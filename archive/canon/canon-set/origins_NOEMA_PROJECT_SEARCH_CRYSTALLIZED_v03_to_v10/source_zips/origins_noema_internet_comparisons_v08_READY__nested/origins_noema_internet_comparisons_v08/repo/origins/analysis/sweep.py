"""
Parameter sweep utilities.

sweep_v3          – sweep EM amplitude × zenith factor × day-fraction
run_topo_sweep    – sweep topo_strength × k_synthesis across all scenarios
run_all_scenarios – run and compare all 5 scenarios
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import List, Optional
import multiprocessing
import concurrent.futures

import numpy as np
import pandas as pd

from ..scenarios import ScenarioConfig, ALL_SCENARIOS
from ..simulator.universal import UniversalOriginSimulator
from ..chemistry.fields import sun_envelope_window, tide_semidiurnal, laplacian


# ============================================================================
# HELPERS
# ============================================================================

def _ensure(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================================
# SWEEP V3 — EM amplitude × zenith × day-fraction
# ============================================================================

def _run_once_sweep_v3(
    Nx: int = 96,
    Ny: int = 96,
    hours: float = 30.0,
    dt_h: float = 0.05,
    seed: int = 42,
    EM_amp0: float = 0.20,
    mu: float = 1.0,
    day_fraction: float = 0.50,
) -> dict:
    """Run a single sweep_v3 parameter combination."""
    rng = np.random.default_rng(seed)

    # Constants
    D_U0, D_P0, D_L0, D_M0, D_C0 = 0.35, 0.12, 0.08, 0.015, 0.25
    k_photo0 = 0.35; k_poly0 = 0.08; k_hyd = 0.04; k_lip0 = 0.05
    k_degL = 0.03; k_agg0 = 0.35; k_break0 = 0.010
    k_red_gen0 = 0.10; k_red_loss = 0.06
    alpha_c = 0.474812; beta_s = 0.856234; PhiI0 = 0.02

    x = np.linspace(-1, 1, Nx); y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    grad = 0.6 * (X + 1.0) / 2.0
    speckle = 0.15 * rng.standard_normal((Nx, Ny))
    hotspots = (
        0.7 * np.exp(-((X - 0.35)**2 + (Y + 0.10)**2) / 0.03)
        + 0.5 * np.exp(-((X + 0.45)**2 + (Y - 0.25)**2) / 0.02)
    )
    light_field = np.clip(0.6 + grad + speckle + hotspots, 0.0, 2.0)
    EM_pattern = np.sin(2 * np.pi * 2 * X) * np.sin(2 * np.pi * 3 * Y)

    shore = (Y > -0.2) & (Y < -0.1)
    patches = rng.random((Nx, Ny)) < 0.06
    catalyst = (shore | patches).astype(float)
    catalyst = np.clip(
        catalyst + 0.5 * np.exp(-((X + 0.2)**2 + (Y + 0.5)**2) / 0.02), 0.0, 1.0
    )

    U = 0.9 + 0.1 * rng.random((Nx, Ny))
    P = 0.05 * rng.random((Nx, Ny))
    L = 0.25 * rng.random((Nx, Ny))
    M = 0.04 * rng.random((Nx, Ny))
    C = 0.05 * rng.random((Nx, Ny))

    steps = int(hours / dt_h)
    area_series = []
    for n in range(steps):
        t_h = n * dt_h
        S = sun_envelope_window(t_h, mu=mu, day_fraction=day_fraction)
        Nflag = 0.0 if S > 0 else 1.0
        Tide = tide_semidiurnal(t_h); TideAbs = abs(Tide)

        k_photo = k_photo0 * (1 + 1.2 * S)
        k_lip   = k_lip0   * (1 + 0.2 * S)
        k_agg   = k_agg0   * (1 + 0.10 * TideAbs)
        k_break = k_break0 * (1 + 0.60 * TideAbs)
        k_poly  = k_poly0  * (1 + 0.10 * S + 0.05 * Tide)
        shore_factor   = 1 + 0.50 * max(0.0, -Tide)
        k_poly_map     = k_poly * (1 + shore_factor * shore.astype(float))
        k_red_gen      = k_red_gen0 * (1 + 0.3 * Nflag + 0.1 * S)

        D_U = D_U0 * (1 + 0.15 * TideAbs)
        D_P = D_P0 * (1 + 0.10 * TideAbs)
        D_L = D_L0 * (1 + 0.10 * TideAbs)
        D_M = D_M0 * (1 + 0.20 * TideAbs)
        D_C = D_C0 * (1 + 0.10 * TideAbs)

        EM_env     = EM_amp0 * (1 + 0.35 * Nflag + 0.05 * S) * EM_pattern
        photofactor = 1.0 + 0.8 * light_field * (0.2 + 0.8 * S) + 0.25 * EM_env

        r_photo = k_photo * photofactor * U
        r_poly  = k_poly_map * (1.0 + 0.25 * beta_s * C) * U * (P + 0.05)
        r_lip   = k_lip * (1.0 + 0.4 * catalyst) * U

        dU = -r_photo - r_poly - r_lip + k_hyd * P
        dP =  r_photo + r_poly - k_hyd * P
        dL =  r_lip - k_degL * L - k_agg * (L * L) + k_break * M
        curvature = 1.0 + 0.6 * np.tanh(EM_env)
        dM =  k_agg * (L * L) * curvature * (1.0 + PhiI0) - k_break * M
        dC =  k_red_gen * (0.5 * (0.2 + 0.8 * S) * light_field + 0.4 * np.abs(EM_env)) - k_red_loss * C

        U += dt_h * (D_U * laplacian(U) + dU)
        P += dt_h * (D_P * laplacian(P) + dP)
        L += dt_h * (D_L * laplacian(L) + dL)
        M += dt_h * (D_M * laplacian(M) + dM)
        C += dt_h * (D_C * laplacian(C) + dC)

        for Z in (U, P, L, M, C):
            Z += alpha_c * 1e-3 * rng.standard_normal(Z.shape)
            np.clip(Z, 0.0, None, out=Z)

        if n % max(1, int(1.0 / dt_h)) == 0 or n == steps - 1:
            thresh = max(0.08, float(np.percentile(M, 90)))
            area_series.append(float((M > thresh).mean()))

    return {
        "avgP_final":        float(P.mean()),
        "avgM_final":        float(M.mean()),
        "vesicle_area_mean": float(np.mean(area_series)),
        "vesicle_area_max":  float(np.max(area_series)),
    }


def run_sweep_v3(
    outdir: str = "sweep_v3_outputs",
    Nx: int = 96,
    Ny: int = 96,
    dt_h: float = 0.05,
) -> pd.DataFrame:
    """
    Grid sweep over EM amplitude × zenith factor × day-fraction.

    Saves CSV + heatmap PNGs to *outdir*.
    """
    _ensure(outdir)
    EM_list   = [0.10, 0.20, 0.30, 0.40]
    mu_list   = [0.5, 0.75, 1.0]
    dayf_list = [0.33, 0.50, 0.67]

    rows = []
    combo = 0
    for EM in EM_list:
        for mu in mu_list:
            for dayf in dayf_list:
                combo += 1
                metrics = _run_once_sweep_v3(
                    Nx=Nx, Ny=Ny, dt_h=dt_h, seed=4748 + combo,
                    EM_amp0=EM, mu=mu, day_fraction=dayf,
                )
                rows.append({"EM_amp0": EM, "mu": mu, "day_fraction": dayf, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "sweep_v3_results.csv"), index=False)

    # Heatmaps
    try:
        import matplotlib.pyplot as plt
        for dayf in dayf_list:
            sub = df[df["day_fraction"] == dayf]
            pivot = sub.pivot(index="EM_amp0", columns="mu", values="vesicle_area_mean")
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(pivot.values, origin="lower", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
            fig.colorbar(im, ax=ax, label="mean high-M area")
            ax.set_xlabel("mu = cos(zenith)")
            ax.set_ylabel("EM_amp0")
            ax.set_title(f"Vesicle yield — day_fraction={dayf:.2f}")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"heatmap_dayf_{int(dayf*100)}.png"), dpi=120)
            plt.close(fig)
    except ImportError:
        pass

    print(f"Sweep v3 results saved to: {outdir}")
    return df


# ============================================================================
# TOPOLOGY SWEEP — topo_strength × k_synthesis
# ============================================================================

def _topo_worker(args: tuple) -> tuple:
    cfg, s_val, ks, Nx, Ny, dt_h, hours, outdir = args
    cfg = deepcopy(cfg)
    cfg.topo_strength = float(s_val)
    cfg.k_synthesis   = float(ks)
    sim = UniversalOriginSimulator(cfg, Nx=Nx, Ny=Ny, dt_h=dt_h, outdir=outdir)
    sim.initialize()
    sim.run(hours=hours, record_interval=max(1.0, hours / 20.0), verbose=False)
    return (s_val, ks, int(sim.protocell_count))


def run_topo_sweep(
    scenarios: Optional[List[ScenarioConfig]] = None,
    topo_list: Optional[List[float]] = None,
    synth_list: Optional[List[float]] = None,
    Nx: int = 64,
    Ny: int = 64,
    dt_h: float = 0.05,
    hours: float = 60.0,
    outdir: str = "topo_sweep_outputs",
    enable_multiproc: bool = False,
    workers: int = 4,
) -> pd.DataFrame:
    """
    Sweep topo_strength × k_synthesis for each scenario.

    Returns a summary DataFrame; also saves per-scenario NPZ and heatmaps.
    """
    if scenarios is None:
        scenarios = ALL_SCENARIOS
    if topo_list is None:
        topo_list = list(np.linspace(0.0, 0.5, 6))
    if synth_list is None:
        synth_list = list(np.linspace(0.001, 0.20, 6))

    _ensure(outdir)
    results = []

    for cfg in scenarios:
        print(f"\n=== Topo sweep: scenario {cfg.code} ===")
        outdir_cfg = _ensure(os.path.join(outdir, f"scenario_{cfg.code}"))
        metric_matrix = np.zeros((len(topo_list), len(synth_list)))
        arglist = [
            (deepcopy(cfg), s, ks, Nx, Ny, dt_h, hours, outdir_cfg)
            for s in topo_list
            for ks in synth_list
        ]

        if enable_multiproc:
            max_w = min(workers, multiprocessing.cpu_count())
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_w) as exe:
                for s, ks, metric in exe.map(_topo_worker, arglist):
                    i = topo_list.index(s); j = synth_list.index(ks)
                    metric_matrix[i, j] = metric
        else:
            for args in arglist:
                s, ks, metric = _topo_worker(args)
                i = topo_list.index(s); j = synth_list.index(ks)
                metric_matrix[i, j] = metric
                print(f"  topo={s:.3f}, k_synth={ks:.4f} → proto={metric}")

        np.savez_compressed(
            os.path.join(outdir_cfg, "metric_matrix.npz"),
            topo_list=np.array(topo_list),
            synth_list=np.array(synth_list),
            metric=metric_matrix,
        )
        results.append({
            'scenario': cfg.code,
            'matrix_file': os.path.join(outdir_cfg, 'metric_matrix.npz'),
        })

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(metric_matrix, origin='lower', aspect='auto')
            ax.set_xticks(range(len(synth_list)))
            ax.set_xticklabels([f"{v:.4f}" for v in synth_list], rotation=45)
            ax.set_yticks(range(len(topo_list)))
            ax.set_yticklabels([f"{v:.3f}" for v in topo_list])
            fig.colorbar(im, ax=ax, label='final protocell count')
            ax.set_xlabel('k_synthesis')
            ax.set_ylabel('topo_strength')
            ax.set_title(f"Topo × k_synthesis — scenario {cfg.code}")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir_cfg, "heatmap.png"), dpi=120)
            plt.close(fig)
        except ImportError:
            pass

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(outdir, 'topo_sweep_summary.csv'), index=False)
    return df


# ============================================================================
# RUN ALL 5 SCENARIOS
# ============================================================================

def run_all_scenarios(
    scenarios: Optional[List[ScenarioConfig]] = None,
    Nx: int = 96,
    Ny: int = 96,
    dt_h: float = 0.05,
    base_hours: float = 120.0,
    outdir: str = "outputs",
) -> pd.DataFrame:
    """
    Run all scenarios and return a combined summary DataFrame.

    Titan (D) runs for 500 h by default due to its slow kinetics.
    """
    if scenarios is None:
        scenarios = ALL_SCENARIOS
    _ensure(outdir)
    summaries = []

    for cfg in scenarios:
        sim = UniversalOriginSimulator(cfg, Nx=Nx, Ny=Ny, dt_h=dt_h, outdir=outdir)
        sim.initialize()
        hours = base_hours if cfg.code != 'D' else max(base_hours, 500.0)
        sim.run(hours=hours, record_interval=2.0, verbose=True)
        sim.save_outputs(prefix='final')
        summaries.append({
            'Scenario':        cfg.code,
            'Name':            cfg.name,
            'Temp_C':          cfg.temp_C,
            'Solvent':         cfg.solvent.value,
            'Final_Polymers':  sim.rna_population.size,
            'Final_ProtoC':    sim.protocell_count,
            'Expected_ProtoC': cfg.expected_protocells,
        })

    combined = pd.DataFrame(summaries)
    combined.to_csv(os.path.join(outdir, 'ALL_SCENARIOS_SUMMARY.csv'), index=False)
    print("\n=== ALL SCENARIOS COMPLETE ===")
    print(combined.to_string(index=False))
    return combined
