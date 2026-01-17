#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
origin_monolith_v2.py
Monolithic origin-of-life simulator + topology + RNA replication + multiprocessing sweep
- Merged sweep_v3 + 5-scenario topo simulator (Berry–Euler analog)
- Replication / fragmentation / mutation / selection for RNA population (vectorized arrays)
- Optional multiprocessing for parameter sweeps (--enable-multiproc)
- Optional numba acceleration if numba is installed (graceful fallback)
Usage:
    python origin_monolith_v2.py --mode scenarios --hours 20
    python origin_monolith_v2.py --mode topo_sweep --enable-multiproc --workers 6
"""
import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
from copy import deepcopy
import multiprocessing
import concurrent.futures
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try optional numba
try:
    import numba as _numba
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(d: str):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_figure(fig, path, dpi=160):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

# Laplacian: if numba available provide a jitted fallback (loops). Otherwise use numpy.roll.
if NUMBA_AVAILABLE:
    @njit
    def laplacian_numba(Z):
        nx, ny = Z.shape
        out = np.empty_like(Z)
        for i in range(nx):
            for j in range(ny):
                up = Z[i-1 if i-1>=0 else nx-1, j]
                down = Z[i+1 if i+1<nx else 0, j]
                left = Z[i, j-1 if j-1>=0 else ny-1]
                right = Z[i, j+1 if j+1<ny else 0]
                out[i,j] = -4.0 * Z[i,j] + up + down + left + right
        return out
    def laplacian(Z):
        return laplacian_numba(Z)
else:
    def laplacian(Z):
        return (-4 * Z + np.roll(Z, 1, 0) + np.roll(Z, -1, 0) +
                np.roll(Z, 1, 1) + np.roll(Z, -1, 1))

# ---------------------------
# Part A: sweep_v3 reproduction (original run_once)
# ---------------------------
def sun_envelope_window(hour, mu=1.0, day_fraction=0.5):
    h = hour % 24.0
    half = 12.0 * day_fraction
    d = abs(h - 12.0)
    if d > half or half <= 0:
        return 0.0
    return float(mu * 0.5 * (1.0 + np.cos(np.pi * d / half)))

def tide_semidiurnal(hour):
    return np.sin(2 * np.pi * hour / 12.42)

def run_once_sweep_v3(Nx=96, Ny=96, hours=30.0, dt_h=0.05, seed=42,
                      EM_amp0=0.20, mu=1.0, day_fraction=0.50):
    rng = np.random.default_rng(seed)
    # base diffs
    D_U0, D_P0, D_L0, D_M0, D_C0 = 0.35, 0.12, 0.08, 0.015, 0.25
    # kinetics
    k_photo0 = 0.35; k_poly0 = 0.08; k_hyd = 0.04; k_lip0 = 0.05
    k_degL = 0.03; k_agg0 = 0.35; k_break0 = 0.010; k_red_gen0 = 0.10; k_red_loss = 0.06
    # CIEL-style constants
    alpha_c = 0.474812; beta_s = 0.856234; PhiI0 = 0.02

    x = np.linspace(-1, 1, Nx); y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    grad = 0.6*(X + 1.0)/2.0
    speckle = 0.15 * rng.standard_normal((Nx, Ny))
    hotspots = 0.7 * np.exp(-((X-0.35)**2 + (Y+0.10)**2)/0.03) + 0.5 * np.exp(-((X+0.45)**2 + (Y-0.25)**2)/0.02)
    light_field = np.clip(0.6 + grad + speckle + hotspots, 0.0, 2.0)
    EM_pattern = np.sin(2*np.pi*2*X) * np.sin(2*np.pi*3*Y)

    shore = (Y > -0.2) & (Y < -0.1)
    patches = rng.random((Nx, Ny)) < 0.06
    catalyst = (shore | patches).astype(float)
    catalyst = np.clip(catalyst + 0.5 * np.exp(-((X+0.2)**2 + (Y+0.5)**2)/0.02), 0, 1)

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
        Nflag = 1.0 - (1.0 if S > 0 else 0.0)
        Tide = tide_semidiurnal(t_h); TideAbs = abs(Tide)

        k_photo = k_photo0 * (1 + 1.2*S)
        k_lip   = k_lip0   * (1 + 0.2*S)
        k_agg   = k_agg0   * (1 + 0.10*TideAbs)
        k_break = k_break0 * (1 + 0.60*TideAbs)
        k_poly  = k_poly0  * (1 + 0.10*S + 0.05*Tide)
        shore_factor = 1 + 0.50*max(0.0, -Tide)
        k_poly_map = k_poly * (1 + shore_factor*shore.astype(float))
        k_red_gen = k_red_gen0 * (1 + 0.3*Nflag + 0.1*S)

        D_U = D_U0*(1 + 0.15*TideAbs); D_P = D_P0*(1 + 0.10*TideAbs)
        D_L = D_L0*(1 + 0.10*TideAbs); D_M = D_M0*(1 + 0.20*TideAbs); D_C = D_C0*(1 + 0.10*TideAbs)

        EM_env = EM_amp0*(1 + 0.35*Nflag + 0.05*S) * EM_pattern
        photofactor = (1.0 + 0.8*light_field*(0.2 + 0.8*S) + 0.25*EM_env)

        r_photo = k_photo * photofactor * U
        r_poly  = k_poly_map * (1.0 + 0.25*beta_s*C) * U * (P + 0.05)
        r_lip   = k_lip * (1.0 + 0.4*catalyst) * U

        dU = -r_photo - r_poly - r_lip + k_hyd*P
        dP = r_photo + r_poly - k_hyd*P
        dL = r_lip - k_degL*L - k_agg*(L*L) + k_break*M
        curvature = 1.0 + 0.6*np.tanh(EM_env)
        dM = k_agg*(L*L)*curvature*(1.0 + PhiI0) - k_break*M
        dC = k_red_gen*(0.5*(0.2 + 0.8*S)*light_field + 0.4*np.abs(EM_env)) - k_red_loss*C

        U += dt_h*(D_U*laplacian(U) + dU)
        P += dt_h*(D_P*laplacian(P) + dP)
        L += dt_h*(D_L*laplacian(L) + dL)
        M += dt_h*(D_M*laplacian(M) + dM)
        C += dt_h*(D_C*laplacian(C) + dC)

        for Z in (U, P, L, M, C):
            Z += (0.474812*1e-3) * rng.standard_normal(Z.shape)
            np.clip(Z, 0.0, None, out=Z)

        if n % int(1.0 / dt_h) == 0 or n == steps - 1:
            thresh = max(0.08, np.percentile(M, 90))
            binM = (M > thresh)
            area_series.append(float(binM.mean()))

    return {
        "avgP_final": float(P.mean()),
        "avgM_final": float(M.mean()),
        "vesicle_area_mean": float(np.mean(area_series)),
        "vesicle_area_max": float(np.max(area_series)),
    }

def run_sweep_v3_and_save(outdir='sweep_v3_outputs', Nx=96, Ny=96, dt_h=0.05):
    ensure_dir(outdir)
    EM_list = [0.10, 0.20, 0.30, 0.40]
    mu_list = [0.5, 0.75, 1.0]
    dayf_list = [0.33, 0.50, 0.67]

    rows = []
    combo_id = 0
    for EM_amp0 in EM_list:
        for mu in mu_list:
            for dayf in dayf_list:
                combo_id += 1
                metrics = run_once_sweep_v3(
                    EM_amp0=EM_amp0, mu=mu, day_fraction=dayf,
                    hours=30.0, dt_h=dt_h, seed=4748+combo_id, Nx=Nx, Ny=Ny
                )
                rows.append({"EM_amp0": EM_amp0, "mu": mu, "day_fraction": dayf, **metrics})
    df = pd.DataFrame(rows)
    csv_fn = os.path.join(outdir, "prebiotic_sweep_v3_results.csv")
    df.to_csv(csv_fn, index=False)

    # heatmaps per day_fraction
    for dayf in dayf_list:
        sub = df[df["day_fraction"] == dayf]
        pivot_area = sub.pivot(index="EM_amp0", columns="mu", values="vesicle_area_mean")
        pivot_M = sub.pivot(index="EM_amp0", columns="mu", values="avgM_final")

        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(pivot_area.values, origin="lower", aspect="auto")
        ax.set_xticks(range(len(pivot_area.columns))); ax.set_xticklabels([f"{c:.2f}" for c in pivot_area.columns])
        ax.set_yticks(range(len(pivot_area.index))); ax.set_yticklabels([f"{r:.2f}" for r in pivot_area.index])
        fig.colorbar(im, ax=ax, label="mean high-M area")
        ax.set_xlabel("mu = cos(zenith)"); ax.set_ylabel("EM_amp0")
        ax.set_title(f"Vesicle yield — day_fraction={dayf:.2f}")
        save_figure(fig, os.path.join(outdir, f"heatmap_area_dayf_{int(dayf*100)}.png"))

        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(pivot_M.values, origin="lower", aspect="auto")
        ax.set_xticks(range(len(pivot_M.columns))); ax.set_xticklabels([f"{c:.2f}" for c in pivot_M.columns])
        ax.set_yticks(range(len(pivot_M.index))); ax.set_yticklabels([f"{r:.2f}" for r in pivot_M.index])
        fig.colorbar(im, ax=ax, label="avgM_final")
        ax.set_xlabel("mu = cos(zenith)"); ax.set_ylabel("EM_amp0")
        ax.set_title(f"Final membrane fraction — day_fraction={dayf:.2f}")
        save_figure(fig, os.path.join(outdir, f"heatmap_M_dayf_{int(dayf*100)}.png"))

    print(f"Saved sweep results to: {csv_fn} and heatmaps in {outdir}")
    return df

# ---------------------------
# Part B: Universal 5-scenario simulator with Kähler-Berry-Euler topology
# + RNA replication/fragmentation/selection
# ---------------------------
@dataclass
class ScenarioConfig:
    name: str
    code: str
    location: str
    temp_C: float
    pressure_atm: float
    UV_flux: float
    solvent: str
    pH: float
    redox: str
    energy_source: str
    k_energy: float
    catalyst: str
    k_catalysis: float
    concentration_boost: float
    k_synthesis: float
    k_degradation: float
    expected_protocells: int
    timescale_description: str
    topo_strength: float
    topo_pattern: str
    topo_time_dependence: str
    topo_pulse_freq: float = 0.0
    seed: int = 42

# Scenarios definitions (same as before)
SCENARIO_A = ScenarioConfig(
    name="Shallow Ocean + UV (Clay Hypothesis)", code="A",
    location="Earth - Coastal tidal zones, 10m depth",
    temp_C=65.0, pressure_atm=1.0, UV_flux=30.0, solvent="H2O", pH=7.5,
    redox="Mildly oxidizing", energy_source="UV photochemistry", k_energy=0.35,
    catalyst="Montmorillonite clay", k_catalysis=7.5, concentration_boost=1000.0,
    k_synthesis=0.15, k_degradation=0.0042, expected_protocells=600,
    timescale_description="Hours", topo_strength=0.25, topo_pattern='sin',
    topo_time_dependence='pulsing', topo_pulse_freq=0.05, seed=101
)

SCENARIO_B = ScenarioConfig(
    name="Deep-Sea Hydrothermal Vents (Iron-Sulfur World)", code="B",
    location="Earth - Mid-ocean ridges", temp_C=90.0, pressure_atm=200.0, UV_flux=0.0,
    solvent="H2O", pH=9.0, redox="Strongly reducing", energy_source="Chemosynthesis (H2 + CO2)",
    k_energy=0.08, catalyst="Fe-S clusters", k_catalysis=15.0, concentration_boost=500.0,
    k_synthesis=0.05, k_degradation=0.02, expected_protocells=400, timescale_description="Days",
    topo_strength=0.35, topo_pattern='vortex', topo_time_dependence='static', seed=202
)

SCENARIO_C = ScenarioConfig(
    name="Ammonia-Based Biochemistry", code="C", location="Cold NH3 worlds", temp_C=-55.0,
    pressure_atm=1.0, UV_flux=10.0, solvent="NH3", pH=11.0, redox="Variable",
    energy_source="UV + Chemosynthesis", k_energy=0.02, catalyst="NH3-ice minerals",
    k_catalysis=5.0, concentration_boost=300.0, k_synthesis=0.01, k_degradation=0.001,
    expected_protocells=100, timescale_description="Weeks", topo_strength=0.15,
    topo_pattern='gauss', topo_time_dependence='drift', topo_pulse_freq=0.005, seed=303
)

SCENARIO_D = ScenarioConfig(
    name="Titan Methane Lakes (Hydrocarbon Biochemistry)", code="D",
    location="Titan - Kraken Mare", temp_C=-179.0, pressure_atm=1.45, UV_flux=2.0,
    solvent="CH4/C2H6", pH=7.0, redox="Non-oxidizing",
    energy_source="Atmospheric photochemistry", k_energy=0.001, catalyst="Tholins",
    k_catalysis=2.0, concentration_boost=100.0, k_synthesis=0.001, k_degradation=0.0001,
    expected_protocells=30, timescale_description="1000s hours", topo_strength=0.05,
    topo_pattern='random', topo_time_dependence='static', seed=404
)

SCENARIO_E = ScenarioConfig(
    name="Enceladus Subsurface Ocean", code="E", location="Enceladus - Sub-ice ocean",
    temp_C=4.0, pressure_atm=800.0, UV_flux=0.0, solvent="H2O", pH=9.5, redox="Reducing (H2-rich)",
    energy_source="Hydrothermal (tidal heating)", k_energy=0.08, catalyst="Fe-S + Mg-silicates",
    k_catalysis=12.0, concentration_boost=600.0, k_synthesis=0.08, k_degradation=0.01,
    expected_protocells=450, timescale_description="Days", topo_strength=0.28, topo_pattern='cos',
    topo_time_dependence='pulsing', topo_pulse_freq=0.02, seed=505
)

ALL_SCENARIOS = [SCENARIO_A, SCENARIO_B, SCENARIO_C, SCENARIO_D, SCENARIO_E]

class UniversalOriginSimulator:
    def __init__(self, config: ScenarioConfig, Nx=96, Ny=96, dt_h=0.05, outdir='sim_outputs'):
        self.config = config
        self.Nx, self.Ny = Nx, Ny
        self.dt_h = dt_h
        self.t_h = 0.0
        self.outdir = os.path.join(outdir, f"scenario_{config.code}")
        ensure_dir(self.outdir)
        self._rng = np.random.default_rng(self.config.seed)

        # chemical fields
        self.E = None; self.O = None; self.N = None
        self.R = None; self.M = None; self.L = None
        self.Cat = None

        # RNA population arrays
        self.rna_active = None

        # tracking
        self.protocell_count = 0
        self.history = {'time_h': [], 'mean_R': [], 'mean_M': [],
                        'n_polymers': [], 'n_protocells': [], 'mean_fitness': []}

        # topology
        self.topo_field = None
        self.topo_curvature = None
        self._initialize_topology()

    def _initialize_topology(self):
        x = np.linspace(-1, 1, self.Nx)
        y = np.linspace(-1, 1, self.Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        s = float(self.config.topo_strength)
        pattern = self.config.topo_pattern.lower()
        if pattern == 'sin':
            base = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        elif pattern == 'cos':
            base = np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        elif pattern == 'vortex':
            theta = np.arctan2(Y, X)
            r = np.sqrt(X**2 + Y**2) + 1e-9
            base = np.sin(4 * theta) * np.exp(-3 * r**2)
        elif pattern == 'gauss':
            base = np.exp(-((X-0.2)**2 + (Y+0.1)**2) / 0.02) - 0.5 * np.exp(-((X+0.3)**2 + (Y-0.3)**2) / 0.05)
        elif pattern == 'random':
            base = self._rng.standard_normal((self.Nx, self.Ny))
            base = (np.roll(base, 1, 0) + base + np.roll(base, -1, 0) + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5.0
        else:
            base = np.zeros((self.Nx, self.Ny))
        if np.std(base) > 0:
            base = (base - np.mean(base)) / (np.std(base) + 1e-12)
        self.topo_field = s * base
        self._update_topo_curvature()

    def _update_topo_curvature(self):
        Z = self.topo_field
        lap = (-4 * Z + np.roll(Z, 1, 0) + np.roll(Z, -1, 0) + np.roll(Z, 1, 1) + np.roll(Z, -1, 1))
        if np.std(lap) > 0:
            lap = (lap - lap.mean()) / (np.std(lap) + 1e-12)
        self.topo_curvature = lap

    def _advance_topo_in_time(self):
        mode = self.config.topo_time_dependence
        if mode == 'static':
            return
        elif mode == 'pulsing':
            f = max(1e-9, self.config.topo_pulse_freq)
            factor = 1.0 + 0.5 * math.sin(2 * math.pi * f * self.t_h)
            self.topo_field = self.topo_field * factor
            self._update_topo_curvature()
        elif mode == 'drift':
            shift = int((self.t_h * 0.02) % self.Nx)
            self.topo_field = np.roll(self.topo_field, shift, axis=0)
            self._update_topo_curvature()

    def initialize(self):
        temp_factor = math.exp((self.config.temp_C - 25) / 100.0)
        rng = self._rng
        self.E = rng.uniform(0.1, 0.3, (self.Nx, self.Ny)) * temp_factor
        self.O = rng.uniform(0.05, 0.15, (self.Nx, self.Ny))
        self.N = rng.uniform(0.01, 0.05, (self.Nx, self.Ny))
        self.R = np.zeros((self.Nx, self.Ny))
        self.M = np.zeros((self.Nx, self.Ny))
        self.L = rng.uniform(0.005, 0.01, (self.Nx, self.Ny))
        self.Cat = rng.uniform(0.8, 1.2, (self.Nx, self.Ny))

        n_seed = max(5, int(20 * temp_factor))
        pos_x = rng.integers(0, self.Nx, size=n_seed)
        pos_y = rng.integers(0, self.Ny, size=n_seed)
        lengths = rng.integers(20, 50, size=n_seed)
        fitness = rng.uniform(0.3, 0.6, size=n_seed)
        age = np.zeros(n_seed, dtype=float)
        self.rna_active = {
            'pos_x': pos_x.astype(np.int32),
            'pos_y': pos_y.astype(np.int32),
            'length': lengths.astype(np.int32),
            'fitness': fitness.astype(float),
            'age': age.astype(float)
        }
        for i in range(len(pos_x)):
            self.R[pos_x[i], pos_y[i]] += 0.1

    # ---------------------------
    # Replication & selection mechanics
    # ---------------------------
    def step_replication_and_selection(self):
        """
        Vectorized replication:
         - Each RNA has replication probability proportional to local R and fitness.
         - Successful replication creates a new RNA nearby (with mutation on fitness and length).
         - Fragmentation: long RNAs have small chance to split into two.
         - Selection implemented via degradation step and probabilistic death.
        """
        if self.rna_active is None:
            return
        rng = self._rng
        posx = self.rna_active['pos_x']; posy = self.rna_active['pos_y']
        fitness = self.rna_active['fitness']
        lengths = self.rna_active['length']

        # local field influence
        local_R = self.R[posx, posy]
        # base replication rate scaled by local R and fitness
        base_rep_rate = 0.02  # tunable baseline
        rep_probs = np.clip(base_rep_rate * (1.0 + fitness) * (local_R + 1e-6) * self.dt_h, 0.0, 0.8)

        draws = rng.random(len(rep_probs))
        reproduce_mask = draws < rep_probs

        n_new = int(np.sum(reproduce_mask))
        if n_new > 0:
            # create offspring near parents (random +/-1 cell)
            parents_idx = np.nonzero(reproduce_mask)[0]
            offs_pos_x = (posx[parents_idx] + rng.integers(-1, 2, size=n_new)) % self.Nx
            offs_pos_y = (posy[parents_idx] + rng.integers(-1, 2, size=n_new)) % self.Ny
            offs_length = np.clip(lengths[parents_idx] + rng.integers(-2, 3, size=n_new), 5, 200)
            # mutation in fitness (small gaussian)
            offs_fitness = np.clip(fitness[parents_idx] + rng.normal(0.0, 0.02, size=n_new), 0.0, 1.0)
            offs_age = np.zeros(n_new, dtype=float)

            # append offspring to arrays
            for key, arr in [('pos_x', offs_pos_x), ('pos_y', offs_pos_y),
                             ('length', offs_length), ('fitness', offs_fitness), ('age', offs_age)]:
                self.rna_active[key] = np.concatenate([self.rna_active[key], arr])

            # seed modest amount into R field where new RNA placed
            for i in range(n_new):
                self.R[int(offs_pos_x[i]), int(offs_pos_y[i])] += 0.05

        # Fragmentation: long RNAs may split
        frag_mask = lengths > 80
        if np.any(frag_mask):
            frag_prob = 0.005 * self.dt_h
            draws2 = rng.random(len(lengths))
            do_frag = (frag_mask) & (draws2 < frag_prob)
            idx_frag = np.nonzero(do_frag)[0]
            for idx in idx_frag:
                L0 = int(self.rna_active['length'][idx])
                if L0 <= 10:
                    continue
                cut = rng.integers(5, max(6, L0-4))
                L1, L2 = cut, max(5, L0 - cut)
                # parent becomes fragment1, add fragment2 as new RNA near
                self.rna_active['length'][idx] = L1
                new_pos_x = (self.rna_active['pos_x'][idx] + rng.integers(-1,2)) % self.Nx
                new_pos_y = (self.rna_active['pos_y'][idx] + rng.integers(-1,2)) % self.Ny
                new_len = np.int32(L2)
                new_fit = np.clip(self.rna_active['fitness'][idx] + rng.normal(0.0, 0.01), 0.0, 1.0)
                self.rna_active['pos_x'] = np.concatenate([self.rna_active['pos_x'], np.array([new_pos_x], dtype=np.int32)])
                self.rna_active['pos_y'] = np.concatenate([self.rna_active['pos_y'], np.array([new_pos_y], dtype=np.int32)])
                self.rna_active['length'] = np.concatenate([self.rna_active['length'], np.array([new_len], dtype=np.int32)])
                self.rna_active['fitness'] = np.concatenate([self.rna_active['fitness'], np.array([new_fit], dtype=float)])
                self.rna_active['age'] = np.concatenate([self.rna_active['age'], np.array([0.0], dtype=float)])
                # seed R
                self.R[int(new_pos_x), int(new_pos_y)] += 0.03

    # ---------------------------
    # Vectorized chemical steps
    # ---------------------------
    def step_energy_conversion(self):
        k = self.config.k_energy
        efficiency = 0.8 if self.config.UV_flux > 0 else 0.6
        mod = 1.0 + 0.6 * self.topo_field + 0.4 * self.topo_curvature
        mod = np.clip(mod, 0.2, 3.0)
        dE = -k * self.E * self.dt_h
        dO = k * self.E * efficiency * self.dt_h * mod
        self.E = np.clip(self.E + dE, 0.0, 1.0)
        self.O = np.clip(self.O + dO, 0.0, 1.0)

    def step_catalysis(self):
        k_cat = self.config.k_catalysis * 0.1
        catalyst_effect = self.Cat / (self.Cat + 1.0)
        mod = 1.0 + 0.5 * self.topo_field + 0.6 * self.topo_curvature
        mod = np.clip(mod, 0.2, 4.0)
        dO = -k_cat * self.O * catalyst_effect * self.dt_h * mod
        dN =  k_cat * self.O * catalyst_effect * self.dt_h * mod
        self.O = np.clip(self.O + dO, 0.0, 1.0)
        self.N = np.clip(self.N + dN, 0.0, 1.0)

    def step_polymerization(self):
        k_syn = self.config.k_synthesis
        boost = max(1e-6, self.config.concentration_boost / 1000.0)
        mod = 1.0 + 0.8 * self.topo_field + 0.2 * self.topo_curvature + 0.3 * (self.Cat - 1.0)
        mod = np.clip(mod, 0.1, 5.0)
        dN = -k_syn * self.N * boost * self.dt_h * mod
        dR =  k_syn * self.N * boost * self.dt_h * mod
        self.N = np.clip(self.N + dN, 0.0, 1.0)
        self.R = np.clip(self.R + dR, 0.0, 1.0)

    def step_degradation(self):
        k_deg = self.config.k_degradation
        temp_factor = math.exp(0.05 * (self.config.temp_C - 25.0))
        mod = 1.0 + 0.5 * (-self.topo_field) + 0.3 * (self.topo_curvature)
        mod = np.clip(mod, 0.05, 4.0)
        dR = -k_deg * self.R * temp_factor * self.dt_h * mod
        self.R = np.clip(self.R + dR, 0.0, 1.0)

        if self.rna_active is None:
            return
        ages = self.rna_active['age'] + self.dt_h
        posx = self.rna_active['pos_x']; posy = self.rna_active['pos_y']
        local_topo = self.topo_field[posx, posy]
        local_curv = self.topo_curvature[posx, posy]
        local_mod = np.clip(1.0 + 0.5 * (-local_topo) + 0.3 * local_curv, 0.05, 4.0)
        probs = self.config.k_degradation * temp_factor * self.dt_h * local_mod
        rdraws = self._rng.random(len(probs))
        keep_mask = rdraws >= probs
        for key in list(self.rna_active.keys()):
            self.rna_active[key] = self.rna_active[key][keep_mask]
        if self.rna_active['age'].size > 0:
            self.rna_active['age'] = ages[keep_mask]
        else:
            self.rna_active['age'] = np.zeros(0, dtype=float)

    def step_membrane_formation(self):
        k_mem = 0.4
        T = self.config.temp_C
        if T < -100:
            temp_factor = 0.1
        elif T < 0:
            temp_factor = 0.5
        elif T < 70:
            temp_factor = 1.0
        elif T < 100:
            temp_factor = 0.7
        else:
            temp_factor = 0.3
        mod = 1.0 + 0.6 * self.topo_field + 0.5 * self.topo_curvature
        mod = np.clip(mod, 0.05, 6.0)
        dL = -k_mem * self.L * temp_factor * self.dt_h * mod
        dM =  k_mem * self.L * temp_factor * self.dt_h * mod
        self.L = np.clip(self.L + dL, 0.0, 1.0)
        self.M = np.clip(self.M + dM, 0.0, 1.0)

    def step_protocell_detection(self):
        threshold_M = 0.05
        threshold_R = 0.03
        protocells = np.where((self.M > threshold_M) & (self.R > threshold_R))
        self.protocell_count = int(len(protocells[0]))

    # ---------------------------
    # Recording
    # ---------------------------
    def record_state(self):
        self.history['time_h'].append(self.t_h)
        self.history['mean_R'].append(float(np.mean(self.R)))
        self.history['mean_M'].append(float(np.mean(self.M)))
        self.history['n_polymers'].append(int(self.rna_active['pos_x'].size if self.rna_active else 0))
        self.history['n_protocells'].append(int(self.protocell_count))
        if self.rna_active is not None and self.rna_active['fitness'].size > 0:
            self.history['mean_fitness'].append(float(np.mean(self.rna_active['fitness'])))
        else:
            self.history['mean_fitness'].append(0.0)

    # ---------------------------
    # Single-step & run
    # ---------------------------
    def step(self):
        self._advance_topo_in_time()
        # chemical dynamics
        self.step_energy_conversion()
        self.step_catalysis()
        self.step_polymerization()
        # replication/selection happens after polymerization, before degradation
        self.step_replication_and_selection()
        self.step_degradation()
        self.step_membrane_formation()
        self.step_protocell_detection()
        self.t_h += self.dt_h

    def run(self, hours=120.0, record_interval=2.0, verbose=True):
        n_steps = int(hours / self.dt_h)
        record_steps = max(1, int(record_interval / self.dt_h))
        if verbose:
            print(f"\nRunning SCENARIO {self.config.code}: {self.config.name}")
            print(f"  Grid: {self.Nx}x{self.Ny} | dt_h={self.dt_h}h | total hours={hours} ({n_steps:,} steps)")
        for step in range(n_steps):
            self.step()
            if step % record_steps == 0 or step == n_steps - 1:
                self.record_state()
            if verbose and (step % (max(1, record_steps * 10)) == 0):
                n_pol = int(self.rna_active['pos_x'].size if self.rna_active else 0)
                print(f"  t={self.t_h:6.1f}h | Polymers={n_pol:4d} | Proto-cells={self.protocell_count:4d}")
        if verbose:
            print(f"\n✅ SCENARIO {self.config.code} complete! Polymers={self.rna_active['pos_x'].size if self.rna_active else 0}, Proto-cells={self.protocell_count}")
        df = pd.DataFrame(self.history)
        csv_fn = os.path.join(self.outdir, f"scenario_{self.config.code}_history.csv")
        df.to_csv(csv_fn, index=False)
        np.savez_compressed(os.path.join(self.outdir, f"scenario_{self.config.code}_final_fields.npz"),
                            E=self.E, O=self.O, N=self.N, R=self.R, M=self.M, L=self.L,
                            topo=self.topo_field, curvature=self.topo_curvature)
        return df

    def plot_heatmaps(self, prefix="final"):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        im0 = axs[0,0].imshow(self.R, origin='lower', aspect='auto'); axs[0,0].set_title('R (genetic polymer)')
        fig.colorbar(im0, ax=axs[0,0])
        im1 = axs[0,1].imshow(self.M, origin='lower', aspect='auto'); axs[0,1].set_title('M (membrane)')
        fig.colorbar(im1, ax=axs[0,1])
        im2 = axs[1,0].imshow(self.topo_field, origin='lower', aspect='auto'); axs[1,0].set_title('topo_field (Berry analog)')
        fig.colorbar(im2, ax=axs[1,0])
        im3 = axs[1,1].imshow(self.topo_curvature, origin='lower', aspect='auto'); axs[1,1].set_title('topo_curvature (laplacian proxy)')
        fig.colorbar(im3, ax=axs[1,1])
        plt.suptitle(f"Scenario {self.config.code}: {self.config.name}")
        save_figure(fig, os.path.join(self.outdir, f"{prefix}_heatmaps.png"))

    def save_summary(self):
        summary = {
            'Scenario': self.config.code,
            'Name': self.config.name,
            'Location': self.config.location,
            'Temp_C': self.config.temp_C,
            'Solvent': self.config.solvent,
            'Energy_Source': self.config.energy_source,
            'Catalyst': self.config.catalyst,
            'Final_Polymers': int(self.rna_active['pos_x'].size if self.rna_active else 0),
            'Final_ProtoC': int(self.protocell_count),
            'Expected_ProtoC': int(self.config.expected_protocells),
            'Success_Rate_pct': float(100.0 * self.protocell_count / max(1, self.config.expected_protocells)),
            'Timescale': self.config.timescale_description
        }
        df = pd.DataFrame([summary])
        df.to_csv(os.path.join(self.outdir, 'scenario_summary.csv'), index=False)
        return df

# ---------------------------
# Topology sweep: vary topo_strength x k_synthesis
#   Multiprocessing-enabled version (optional)
# ---------------------------
def _topo_sweep_worker(args):
    """
    Worker for parallel sweep. Receives a tuple:
    (cfg_serialized, s_val, ks, Nx, Ny, dt_h, hours, outdir_cfg)
    Returns metric (final protocell count).
    """
    cfg_copy, s_val, ks, Nx, Ny, dt_h, hours, outdir_cfg = args
    # cfg_copy is a pure dataclass; we can mutate it
    cfg_copy = deepcopy(cfg_copy)
    cfg_copy.topo_strength = float(s_val)
    cfg_copy.k_synthesis = float(ks)
    sim = UniversalOriginSimulator(cfg_copy, Nx=Nx, Ny=Ny, dt_h=dt_h, outdir=outdir_cfg)
    sim.initialize()
    sim.run(hours=hours, record_interval=max(1.0, hours/20.0), verbose=False)
    metric = int(sim.protocell_count)
    return (s_val, ks, metric)

def run_topo_param_sweep(all_scenarios: List[ScenarioConfig],
                         topo_list: List[float], synth_list: List[float],
                         Nx=64, Ny=64, dt_h=0.05, hours=60.0, outdir='topo_sweep_outputs',
                         enable_multiproc: bool = False, workers: int = 4):
    ensure_dir(outdir)
    results = []
    for cfg in all_scenarios:
        print(f"\n=== Sweeping scenario {cfg.code}: {cfg.name} ===")
        outdir_cfg = os.path.join(outdir, f"scenario_{cfg.code}")
        ensure_dir(outdir_cfg)
        metric_matrix = np.zeros((len(topo_list), len(synth_list)), dtype=float)

        # Prepare args list
        arglist = []
        for i, s_val in enumerate(topo_list):
            for j, ks in enumerate(synth_list):
                arglist.append((deepcopy(cfg), s_val, ks, Nx, Ny, dt_h, hours, outdir_cfg))

        if enable_multiproc:
            max_workers = min(workers, max(1, multiprocessing.cpu_count()))
            print(f"Launching parallel sweep with {max_workers} workers...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
                for res in exe.map(_topo_sweep_worker, arglist):
                    s_val, ks, metric = res
                    i = topo_list.index(s_val); j = synth_list.index(ks)
                    metric_matrix[i, j] = metric
                    print(f"  topo={s_val:.3f}, k_synth={ks:.4f} -> proto={metric}")
        else:
            for (cfg_copy, s_val, ks, Nx_, Ny_, dt_h_, hours_, outdir_cfg_) in arglist:
                res = _topo_sweep_worker((cfg_copy, s_val, ks, Nx_, Ny_, dt_h_, hours_, outdir_cfg_))
                s_val, ks, metric = res
                i = topo_list.index(s_val); j = synth_list.index(ks)
                metric_matrix[i, j] = metric
                print(f"  topo={s_val:.3f}, k_synth={ks:.4f} -> proto={metric}")

        # save matrix and heatmap
        np.savez_compressed(os.path.join(outdir_cfg, "metric_matrix.npz"),
                            topo_list=np.array(topo_list), synth_list=np.array(synth_list), metric=metric_matrix)
        # plot heatmap: y topo, x k_synthesis
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(metric_matrix, origin='lower', aspect='auto')
        ax.set_xticks(range(len(synth_list))); ax.set_xticklabels([f"{v:.4f}" for v in synth_list], rotation=45)
        ax.set_yticks(range(len(topo_list))); ax.set_yticklabels([f"{v:.3f}" for v in topo_list])
        fig.colorbar(im, ax=ax, label='final protocell count')
        ax.set_xlabel('k_synthesis'); ax.set_ylabel('topo_strength')
        ax.set_title(f"Topo x k_synthesis — scenario {cfg.code}")
        save_figure(fig, os.path.join(outdir_cfg, "topo_vs_synth_heatmap.png"))
        results.append({'scenario': cfg.code, 'metric_matrix_file': os.path.join(outdir_cfg, 'metric_matrix.npz'),
                        'heatmap': os.path.join(outdir_cfg, 'topo_vs_synth_heatmap.png')})
    # combined summary CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(outdir, 'topo_sweep_summary.csv'), index=False)
    print(f"\nSaved topo sweep outputs to {outdir}")
    return df_res

# ---------------------------
# Run-all helpers and quick test-run generator
# ---------------------------
def run_all_scenarios(all_scenarios: List[ScenarioConfig], Nx=96, Ny=96, dt_h=0.05, base_hours=120.0, outdir='sim_outputs'):
    ensure_dir(outdir)
    overall = []
    for cfg in all_scenarios:
        sim = UniversalOriginSimulator(cfg, Nx=Nx, Ny=Ny, dt_h=dt_h, outdir=outdir)
        sim.initialize()
        hours = base_hours if cfg.code != 'D' else max(base_hours, 500.0)
        df_hist = sim.run(hours=hours, record_interval=2.0, verbose=True)
        sim.plot_heatmaps(prefix='final')
        summary_df = sim.save_summary()
        overall.append(summary_df)
    combined = pd.concat(overall, ignore_index=True)
    combined.to_csv(os.path.join(outdir, 'ALL_5_SCENARIOS_SUMMARY.csv'), index=False)
    print("\n=== ALL SCENARIOS COMPLETE ===")
    print(combined.to_string(index=False))
    return combined

def quick_test_run(outdir='quick_test', hours=20.0, Nx=64, Ny=64):
    """
    Run a short, low-res test for all scenarios to ensure pipeline & produce example heatmaps.
    """
    ensure_dir(outdir)
    for cfg in ALL_SCENARIOS:
        subdir = os.path.join(outdir, f"scenario_{cfg.code}")
        ensure_dir(subdir)
        sim = UniversalOriginSimulator(cfg, Nx=Nx, Ny=Ny, dt_h=0.05, outdir=subdir)
        sim.initialize()
        sim.run(hours=hours, record_interval=2.0, verbose=True)
        sim.plot_heatmaps(prefix='quick')
        sim.save_summary()
    print("Quick test runs complete. Heatmaps and summaries are in:", outdir)

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Monolithic origin-of-life simulator + replication + multiprocessing sweeps")
    parser.add_argument('--mode', choices=['sweep_v3', 'scenarios', 'both', 'topo_sweep', 'quick_test'], default='both',
                        help='Which module to run')
    parser.add_argument('--outdir', type=str, default='outputs', help='Base output directory')
    parser.add_argument('--nx', type=int, default=96, help='Grid Nx for scenarios')
    parser.add_argument('--ny', type=int, default=96, help='Grid Ny for scenarios')
    parser.add_argument('--dt', type=float, default=0.05, help='Time step hours')
    parser.add_argument('--hours', type=float, default=120.0, help='Base runtime hours for scenarios')
    # topo_sweep parameters
    parser.add_argument('--synth_min', type=float, default=0.001, help='k_synthesis min for topo sweep')
    parser.add_argument('--synth_max', type=float, default=0.20, help='k_synthesis max for topo sweep')
    parser.add_argument('--synth_steps', type=int, default=6, help='k_synthesis steps for topo sweep')
    parser.add_argument('--topo_min', type=float, default=0.0, help='topo_strength min for topo sweep')
    parser.add_argument('--topo_max', type=float, default=0.5, help='topo_strength max for topo sweep')
    parser.add_argument('--topo_steps', type=int, default=6, help='topo_strength steps for topo sweep')
    parser.add_argument('--enable-multiproc', action='store_true', help='Enable multiprocessing for topo sweep')
    parser.add_argument('--workers', type=int, default=4, help='Max workers for multiprocessing')
    # Modify this line to parse only known arguments or no arguments
    args = parser.parse_args(args=[]) # Passing an empty list to parse_args()

    ensure_dir(args.outdir)

    if args.mode in ('sweep_v3', 'both'):
        print("\n=== RUNNING sweep_v3 ===")
        sweep_outdir = os.path.join(args.outdir, 'sweep_v3')
        ensure_dir(sweep_outdir)
        run_sweep_v3_and_save(outdir=sweep_outdir, Nx=args.nx, Ny=args.ny, dt_h=args.dt)

    if args.mode in ('scenarios', 'both'):
        print("\n=== RUNNING 5-SCENARIO TOPO SIMULATOR ===")
        sim_outdir = os.path.join(args.outdir, 'scenarios')
        ensure_dir(sim_outdir)
        run_all_scenarios(ALL_SCENARIOS, Nx=args.nx, Ny=args.ny, dt_h=args.dt, base_hours=args.hours, outdir=sim_outdir)

    if args.mode == 'topo_sweep':
        print("\n=== RUNNING TOPOLOGY PARAM SWEEP ===")
        topo_outdir = os.path.join(args.outdir, 'topo_sweep')
        ensure_dir(topo_outdir)
        synth_list = list(np.linspace(args.synth_min, args.synth_max, args.synth_steps))
        topo_list = list(np.linspace(args.topo_min, args.topo_max, args.topo_steps))
        run_topo_param_sweep(ALL_SCENARIOS, topo_list=topo_list, synth_list=synth_list,
                             Nx=args.nx, Ny=args.ny, dt_h=args.dt, hours=args.hours, outdir=topo_outdir,
                             enable_multiproc=args.enable_multiproc, workers=args.workers)

    if args.mode == 'quick_test':
        quick_out = os.path.join(args.outdir, 'quick_test')
        quick_test_run(outdir=quick_out, hours=args.hours, Nx=min(64, args.nx), Ny=min(64, args.ny))

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nScript finished in {end - start:.1f}s")