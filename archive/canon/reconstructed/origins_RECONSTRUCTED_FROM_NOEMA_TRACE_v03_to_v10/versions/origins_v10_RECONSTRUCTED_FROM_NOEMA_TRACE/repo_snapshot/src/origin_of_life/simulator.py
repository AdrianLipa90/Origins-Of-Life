import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import save_figure
from .scenarios import ScenarioConfig
from .utils import ensure_dir


class UniversalOriginSimulator:
    def __init__(self, config: ScenarioConfig, Nx=96, Ny=96, dt_h=0.05, outdir="sim_outputs"):
        self.config = config
        self.Nx, self.Ny = Nx, Ny
        self.dt_h = dt_h
        self.t_h = 0.0
        self.outdir = os.path.join(outdir, f"scenario_{config.code}")
        ensure_dir(self.outdir)
        self._rng = np.random.default_rng(self.config.seed)

        self.E = None
        self.O = None
        self.N = None
        self.R = None
        self.M = None
        self.L = None
        self.Cat = None

        self.rna_active = None
        self.protocell_count = 0
        self.history = {
            "time_h": [],
            "mean_R": [],
            "mean_M": [],
            "n_polymers": [],
            "n_protocells": [],
            "mean_fitness": [],
        }
        self.topo_field = None
        self.topo_curvature = None
        self._initialize_topology()

    def _initialize_topology(self):
        x = np.linspace(-1, 1, self.Nx)
        y = np.linspace(-1, 1, self.Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        s = float(self.config.topo_strength)
        pattern = self.config.topo_pattern.lower()
        if pattern == "sin":
            base = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        elif pattern == "cos":
            base = np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        elif pattern == "vortex":
            theta = np.arctan2(Y, X)
            r = np.sqrt(X**2 + Y**2) + 1e-9
            base = np.sin(4 * theta) * np.exp(-3 * r**2)
        elif pattern == "gauss":
            base = np.exp(-((X - 0.2) ** 2 + (Y + 0.1) ** 2) / 0.02) - 0.5 * np.exp(-((X + 0.3) ** 2 + (Y - 0.3) ** 2) / 0.05)
        elif pattern == "random":
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
        lap = -4 * Z + np.roll(Z, 1, 0) + np.roll(Z, -1, 0) + np.roll(Z, 1, 1) + np.roll(Z, -1, 1)
        if np.std(lap) > 0:
            lap = (lap - lap.mean()) / (np.std(lap) + 1e-12)
        self.topo_curvature = lap

    def _advance_topo_in_time(self):
        mode = self.config.topo_time_dependence
        if mode == "static":
            return
        if mode == "pulsing":
            f = max(1e-9, self.config.topo_pulse_freq)
            factor = 1.0 + 0.5 * math.sin(2 * math.pi * f * self.t_h)
            self.topo_field = self.topo_field * factor
            self._update_topo_curvature()
        elif mode == "drift":
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
            "pos_x": pos_x.astype(np.int32),
            "pos_y": pos_y.astype(np.int32),
            "length": lengths.astype(np.int32),
            "fitness": fitness.astype(float),
            "age": age.astype(float),
        }
        for i in range(len(pos_x)):
            self.R[pos_x[i], pos_y[i]] += 0.1

    def step_replication_and_selection(self):
        if self.rna_active is None:
            return
        rng = self._rng
        posx = self.rna_active["pos_x"]
        posy = self.rna_active["pos_y"]
        fitness = self.rna_active["fitness"]
        lengths = self.rna_active["length"]
        local_R = self.R[posx, posy]
        rep_probs = np.clip(0.02 * (1.0 + fitness) * (local_R + 1e-6) * self.dt_h, 0.0, 0.8)
        reproduce_mask = rng.random(len(rep_probs)) < rep_probs
        n_new = int(np.sum(reproduce_mask))
        if n_new > 0:
            parents_idx = np.nonzero(reproduce_mask)[0]
            offs_pos_x = (posx[parents_idx] + rng.integers(-1, 2, size=n_new)) % self.Nx
            offs_pos_y = (posy[parents_idx] + rng.integers(-1, 2, size=n_new)) % self.Ny
            offs_length = np.clip(lengths[parents_idx] + rng.integers(-2, 3, size=n_new), 5, 200)
            offs_fitness = np.clip(fitness[parents_idx] + rng.normal(0.0, 0.02, size=n_new), 0.0, 1.0)
            offs_age = np.zeros(n_new, dtype=float)
            for key, arr in [("pos_x", offs_pos_x), ("pos_y", offs_pos_y), ("length", offs_length), ("fitness", offs_fitness), ("age", offs_age)]:
                self.rna_active[key] = np.concatenate([self.rna_active[key], arr])
            for i in range(n_new):
                self.R[int(offs_pos_x[i]), int(offs_pos_y[i])] += 0.05

        frag_mask = lengths > 80
        if np.any(frag_mask):
            do_frag = frag_mask & (rng.random(len(lengths)) < 0.005 * self.dt_h)
            for idx in np.nonzero(do_frag)[0]:
                L0 = int(self.rna_active["length"][idx])
                if L0 <= 10:
                    continue
                cut = rng.integers(5, max(6, L0 - 4))
                L1, L2 = cut, max(5, L0 - cut)
                self.rna_active["length"][idx] = L1
                new_pos_x = (self.rna_active["pos_x"][idx] + rng.integers(-1, 2)) % self.Nx
                new_pos_y = (self.rna_active["pos_y"][idx] + rng.integers(-1, 2)) % self.Ny
                new_len = np.int32(L2)
                new_fit = np.clip(self.rna_active["fitness"][idx] + rng.normal(0.0, 0.01), 0.0, 1.0)
                self.rna_active["pos_x"] = np.concatenate([self.rna_active["pos_x"], np.array([new_pos_x], dtype=np.int32)])
                self.rna_active["pos_y"] = np.concatenate([self.rna_active["pos_y"], np.array([new_pos_y], dtype=np.int32)])
                self.rna_active["length"] = np.concatenate([self.rna_active["length"], np.array([new_len], dtype=np.int32)])
                self.rna_active["fitness"] = np.concatenate([self.rna_active["fitness"], np.array([new_fit], dtype=float)])
                self.rna_active["age"] = np.concatenate([self.rna_active["age"], np.array([0.0], dtype=float)])
                self.R[int(new_pos_x), int(new_pos_y)] += 0.03

    def step_energy_conversion(self):
        mod = np.clip(1.0 + 0.6 * self.topo_field + 0.4 * self.topo_curvature, 0.2, 3.0)
        efficiency = 0.8 if self.config.UV_flux > 0 else 0.6
        dE = -self.config.k_energy * self.E * self.dt_h
        dO = self.config.k_energy * self.E * efficiency * self.dt_h * mod
        self.E = np.clip(self.E + dE, 0.0, 1.0)
        self.O = np.clip(self.O + dO, 0.0, 1.0)

    def step_catalysis(self):
        k_cat = self.config.k_catalysis * 0.1
        catalyst_effect = self.Cat / (self.Cat + 1.0)
        mod = np.clip(1.0 + 0.5 * self.topo_field + 0.6 * self.topo_curvature, 0.2, 4.0)
        dO = -k_cat * self.O * catalyst_effect * self.dt_h * mod
        dN = k_cat * self.O * catalyst_effect * self.dt_h * mod
        self.O = np.clip(self.O + dO, 0.0, 1.0)
        self.N = np.clip(self.N + dN, 0.0, 1.0)

    def step_polymerization(self):
        boost = max(1e-6, self.config.concentration_boost / 1000.0)
        mod = np.clip(1.0 + 0.8 * self.topo_field + 0.2 * self.topo_curvature + 0.3 * (self.Cat - 1.0), 0.1, 5.0)
        dN = -self.config.k_synthesis * self.N * boost * self.dt_h * mod
        dR = self.config.k_synthesis * self.N * boost * self.dt_h * mod
        self.N = np.clip(self.N + dN, 0.0, 1.0)
        self.R = np.clip(self.R + dR, 0.0, 1.0)

    def step_degradation(self):
        temp_factor = math.exp(0.05 * (self.config.temp_C - 25.0))
        mod = np.clip(1.0 + 0.5 * (-self.topo_field) + 0.3 * self.topo_curvature, 0.05, 4.0)
        dR = -self.config.k_degradation * self.R * temp_factor * self.dt_h * mod
        self.R = np.clip(self.R + dR, 0.0, 1.0)
        if self.rna_active is None:
            return
        ages = self.rna_active["age"] + self.dt_h
        posx = self.rna_active["pos_x"]
        posy = self.rna_active["pos_y"]
        local_topo = self.topo_field[posx, posy]
        local_curv = self.topo_curvature[posx, posy]
        local_mod = np.clip(1.0 + 0.5 * (-local_topo) + 0.3 * local_curv, 0.05, 4.0)
        probs = self.config.k_degradation * temp_factor * self.dt_h * local_mod
        keep_mask = self._rng.random(len(probs)) >= probs
        for key in list(self.rna_active.keys()):
            self.rna_active[key] = self.rna_active[key][keep_mask]
        self.rna_active["age"] = ages[keep_mask] if ages[keep_mask].size > 0 else np.zeros(0, dtype=float)

    def step_membrane_formation(self):
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
        mod = np.clip(1.0 + 0.6 * self.topo_field + 0.5 * self.topo_curvature, 0.05, 6.0)
        dL = -0.4 * self.L * temp_factor * self.dt_h * mod
        dM = 0.4 * self.L * temp_factor * self.dt_h * mod
        self.L = np.clip(self.L + dL, 0.0, 1.0)
        self.M = np.clip(self.M + dM, 0.0, 1.0)

    def step_protocell_detection(self):
        protocells = np.where((self.M > 0.05) & (self.R > 0.03))
        self.protocell_count = int(len(protocells[0]))

    def record_state(self):
        self.history["time_h"].append(self.t_h)
        self.history["mean_R"].append(float(np.mean(self.R)))
        self.history["mean_M"].append(float(np.mean(self.M)))
        self.history["n_polymers"].append(int(self.rna_active["pos_x"].size if self.rna_active else 0))
        self.history["n_protocells"].append(int(self.protocell_count))
        self.history["mean_fitness"].append(float(np.mean(self.rna_active["fitness"])) if self.rna_active is not None and self.rna_active["fitness"].size > 0 else 0.0)

    def step(self):
        self._advance_topo_in_time()
        self.step_energy_conversion()
        self.step_catalysis()
        self.step_polymerization()
        self.step_replication_and_selection()
        self.step_degradation()
        self.step_membrane_formation()
        self.step_protocell_detection()
        self.t_h += self.dt_h

    def run(self, hours=120.0, record_interval=2.0, verbose=True):
        n_steps = int(hours / self.dt_h)
        record_steps = max(1, int(record_interval / self.dt_h))
        for step in range(n_steps):
            self.step()
            if step % record_steps == 0 or step == n_steps - 1:
                self.record_state()
            if verbose and (step % (max(1, record_steps * 10)) == 0):
                n_pol = int(self.rna_active["pos_x"].size if self.rna_active else 0)
                print(f"  t={self.t_h:6.1f}h | Polymers={n_pol:4d} | Proto-cells={self.protocell_count:4d}")
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.outdir, f"scenario_{self.config.code}_history.csv"), index=False)
        np.savez_compressed(os.path.join(self.outdir, f"scenario_{self.config.code}_final_fields.npz"), E=self.E, O=self.O, N=self.N, R=self.R, M=self.M, L=self.L, topo=self.topo_field, curvature=self.topo_curvature)
        return df

    def plot_heatmaps(self, prefix="final"):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        im0 = axs[0, 0].imshow(self.R, origin="lower", aspect="auto")
        axs[0, 0].set_title("R (genetic polymer)")
        fig.colorbar(im0, ax=axs[0, 0])
        im1 = axs[0, 1].imshow(self.M, origin="lower", aspect="auto")
        axs[0, 1].set_title("M (membrane)")
        fig.colorbar(im1, ax=axs[0, 1])
        im2 = axs[1, 0].imshow(self.topo_field, origin="lower", aspect="auto")
        axs[1, 0].set_title("topo_field (Berry analog)")
        fig.colorbar(im2, ax=axs[1, 0])
        im3 = axs[1, 1].imshow(self.topo_curvature, origin="lower", aspect="auto")
        axs[1, 1].set_title("topo_curvature (laplacian proxy)")
        fig.colorbar(im3, ax=axs[1, 1])
        plt.suptitle(f"Scenario {self.config.code}: {self.config.name}")
        save_figure(fig, os.path.join(self.outdir, f"{prefix}_heatmaps.png"))

    def save_summary(self):
        summary = {
            "Scenario": self.config.code,
            "Name": self.config.name,
            "Location": self.config.location,
            "Temp_C": self.config.temp_C,
            "Solvent": self.config.solvent,
            "Energy_Source": self.config.energy_source,
            "Catalyst": self.config.catalyst,
            "Final_Polymers": int(self.rna_active["pos_x"].size if self.rna_active else 0),
            "Final_ProtoC": int(self.protocell_count),
            "Expected_ProtoC": int(self.config.expected_protocells),
            "Success_Rate_pct": float(100.0 * self.protocell_count / max(1, self.config.expected_protocells)),
            "Timescale": self.config.timescale_description,
        }
        df = pd.DataFrame([summary])
        df.to_csv(os.path.join(self.outdir, "scenario_summary.csv"), index=False)
        return df
