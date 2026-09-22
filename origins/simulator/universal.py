"""
UniversalOriginSimulator — the main simulation engine.

Merges the best features from all draft versions:
  – 6-step chemical dynamics (v2.0 / v4.0)
  – Vectorised RNA population with replication, fragmentation, selection (holonomy v2)
  – Kähler-Berry-Euler topology field with time evolution (originsHolo2)
  – Zeta-Riemann soft constraints + Heisenberg noise (origin_merged_complete)
  – Full output: CSV history, compressed NPZ field snapshots, heatmap figures

Usage
-----
    from origins.scenarios import SCENARIO_A
    from origins.simulator import UniversalOriginSimulator

    sim = UniversalOriginSimulator(SCENARIO_A, Nx=96, Ny=96)
    sim.initialize()
    df = sim.run(hours=120)
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import pandas as pd

from ..scenarios import ScenarioConfig, TimeDependence
from ..chemistry.fields import (
    diffuse_field,
    solar_envelope,
    tide_semidiurnal,
)
from ..chemistry.clay import ClayMineral
from ..biology.rna import RNASequence, RNAPopulation
from ..biology.protocell import ProtocellDetector
from ..topology.fields import TopologyField
from ..topology.constraints import ZetaRiemannModulator
from ..constants import (
    K_MEMBRANE,
    K_PHOTO_BASE,
    CLAY_NUCLEOTIDE_EFFICIENCY,
    CLAY_CONCENTRATION_FACTOR,
    K_RNA_REPLICATE_BASE,
    RNA_FIDELITY,
    K_DEGRADATION,
    K_RNA_DEGRADE_CLAY,
    D_SMALL_MOL_GRID,
    D_LIPID_GRID,
    D_RNA_GRID,
    CLAY_CONC_G_L,
    TEMP_C_DEFAULT,
)


class UniversalOriginSimulator:
    """
    Universal 2-D origin-of-life simulator.

    Works for all 5 scenarios by adapting kinetic parameters from the
    ScenarioConfig.  Incorporates topological modulation, Zeta-Riemann
    constraints, vectorised RNA population dynamics, and protocell detection.

    Parameters
    ----------
    config        : ScenarioConfig defining physical/chemical environment
    Nx, Ny        : grid dimensions (default 96×96)
    dt_h          : time step in hours (default 0.05 h)
    outdir        : root directory for output files
    include_clay  : whether to instantiate explicit ClayMineral objects (v2.0 mode)
    """

    def __init__(
        self,
        config: ScenarioConfig,
        Nx: int = 96,
        Ny: int = 96,
        dt_h: float = 0.05,
        outdir: str = "outputs",
        include_clay: bool = True,
        preseed_rna: bool = True,
    ):
        self.config = config
        self.Nx, self.Ny = Nx, Ny
        self.dt_h  = dt_h
        self.t_h   = 0.0
        self.outdir = os.path.join(outdir, f"scenario_{config.code}")
        self._rng  = np.random.default_rng(config.seed)
        # Keep experimental zeta noise on an independent deterministic stream.
        # Otherwise enabling zeta consumes the same RNG used by RNA replication
        # and fragmentation, confounding matched-control comparisons.
        self._zeta_rng = np.random.default_rng(
            np.random.SeedSequence([int(config.seed), 0x5A455441])
        )

        # ----------------------------------------------------------
        # Chemical fields  (all shape (Nx, Ny), float32)
        # ----------------------------------------------------------
        self.E:   Optional[np.ndarray] = None  # energy carriers
        self.O:   Optional[np.ndarray] = None  # organic precursors
        self.N:   Optional[np.ndarray] = None  # activated nucleotides / bulk monomers
        self.N_surface: Optional[np.ndarray] = None  # clay-bound monomer reservoir
        self.R:   Optional[np.ndarray] = None  # genetic polymer (RNA-like)
        self.M:   Optional[np.ndarray] = None  # membrane
        self.L:   Optional[np.ndarray] = None  # lipids / amphiphiles
        self.Cat: Optional[np.ndarray] = None  # catalyst distribution

        # Environmental fields
        self.light_field: Optional[np.ndarray] = None
        self.temp_field:  Optional[np.ndarray] = None

        # ----------------------------------------------------------
        # Objects
        # ----------------------------------------------------------
        self.clay_particles: list[ClayMineral] = []
        self.rna_sequences:  list[RNASequence]  = []   # explicit-sequence pop (v2 mode)
        self.rna_population: RNAPopulation = RNAPopulation()  # vectorised pop

        # ----------------------------------------------------------
        # Subsystems
        # ----------------------------------------------------------
        self.topo = TopologyField(config, Nx, Ny)
        self.protocell_detector = ProtocellDetector()

        self.zeta_modulator: Optional[ZetaRiemannModulator] = None
        if config.use_zeta_constraints:
            self.zeta_modulator = ZetaRiemannModulator(
                lambda_soft=config.zeta_lambda_soft,
                sigma_heis=config.zeta_sigma_heis,
            )

        # ----------------------------------------------------------
        # History
        # ----------------------------------------------------------
        self.protocell_count = 0
        self.protocell_area_pixels = 0
        self.history: dict[str, list] = {
            'time_h':       [],
            'mean_R':       [],
            'mean_M':       [],
            'n_polymers':   [],
            'n_protocells': [],
            'protocell_area_pixels': [],
            'surface_monomer_total': [],
            'nucleotide_material_total': [],
            'mean_fitness':      [],
            'berry_accumulated': [],
            'bloch_coherence':   [],
        }

        self._include_clay = include_clay
        self._preseed_rna = bool(preseed_rna)

    # ==================================================================
    # INITIALISATION
    # ==================================================================

    def initialize(self) -> None:
        """Initialise all fields and populations."""
        cfg  = self.config
        rng  = self._rng
        Nx, Ny = self.Nx, self.Ny

        temp_factor = math.exp((cfg.temp_C - 25.0) / 100.0)

        self.E   = np.clip(rng.uniform(0.1, 0.3, (Nx, Ny)) * temp_factor, 0.0, 1.0).astype(np.float32)
        self.O   = rng.uniform(0.05, 0.15, (Nx, Ny)).astype(np.float32)
        self.N   = rng.uniform(0.01, 0.05, (Nx, Ny)).astype(np.float32)
        self.N_surface = np.zeros((Nx, Ny), dtype=np.float32)
        self.R   = np.zeros((Nx, Ny), dtype=np.float32)
        self.M   = np.zeros((Nx, Ny), dtype=np.float32)
        self.L   = rng.uniform(0.005, 0.01, (Nx, Ny)).astype(np.float32)
        self.Cat = rng.uniform(0.8,  1.2,  (Nx, Ny)).astype(np.float32)

        self.light_field = np.ones((Nx, Ny), dtype=np.float32)
        self.temp_field  = np.full((Nx, Ny), cfg.temp_C, dtype=np.float32)

        # Clay minerals
        if self._include_clay:
            for _ in range(50):
                i = int(rng.integers(0, Nx))
                j = int(rng.integers(0, Ny))
                self.clay_particles.append(ClayMineral((i, j), conc_g_L=CLAY_CONC_G_L))

        # Vectorised RNA population.  This is an explicit initial-condition
        # choice, not abiogenesis from monomers.  Set preseed_rna=False for a
        # genuinely polymer-free starting state.
        if self._preseed_rna:
            n_seed = max(5, int(20 * temp_factor))
            self.rna_population = RNAPopulation.seed(n_seed, Nx, Ny, rng)
            for xi, yi in zip(self.rna_population.pos_x, self.rna_population.pos_y):
                self.R[xi, yi] += 0.1
        else:
            self.rna_population = RNAPopulation()

    # ==================================================================
    # CHEMICAL STEPS
    # ==================================================================

    def step_energy_conversion(self) -> None:
        """STEP 1 – Energy source → organic precursors."""
        k = self.config.k_energy
        efficiency = 0.8 if self.config.UV_flux > 0 else 0.6
        mod = self.topo.energy_mod()
        dE = -k * self.E * self.dt_h
        dO =  k * self.E * efficiency * self.dt_h * mod
        self.E = np.clip(self.E + dE, 0.0, 1.0)
        self.O = np.clip(self.O + dO, 0.0, 1.0)

    def step_catalysis(self) -> None:
        """STEP 2 – Mineral-catalysed activation of monomers."""
        k_cat = self.config.k_catalysis * 0.1
        cat_eff = self.Cat / (self.Cat + 1.0)
        mod = self.topo.catalysis_mod()
        dO = -k_cat * self.O * cat_eff * self.dt_h * mod
        dN =  k_cat * self.O * cat_eff * self.dt_h * mod
        self.O = np.clip(self.O + dO, 0.0, 1.0)
        self.N = np.clip(self.N + dN, 0.0, 1.0)

        # Explicit clay-particle catalysis (v2.0 mode)
        if self._include_clay:
            self._clay_catalysis_explicit()

    def _clay_catalysis_explicit(self) -> None:
        """Move bulk monomers into an explicit clay-bound reservoir.

        Concentration enhancement changes local reaction kinetics; it must not
        multiply material.  Adsorption therefore transfers N_bulk -> N_surface
        one-for-one.
        """
        if self.N_surface is None:
            raise RuntimeError("surface monomer reservoir is not initialized")
        for clay in self.clay_particles:
            i, j = clay.position
            available = float(self.N[i, j])
            transfer = min(
                available * CLAY_NUCLEOTIDE_EFFICIENCY * 0.01,
                available,
            )
            if transfer <= 0.0:
                continue
            self.N[i, j] -= transfer
            self.N_surface[i, j] += transfer
            clay.nucleotides_adsorbed += transfer

        if not np.isfinite(self.N_surface).all():
            raise FloatingPointError("clay adsorption produced NaN/Inf")

    def step_polymerization(self) -> None:
        """STEP 3 – Polymer synthesis with explicit substrate conservation."""
        if self.N_surface is None:
            raise RuntimeError("surface monomer reservoir is not initialized")

        k_syn = self.config.k_synthesis
        bulk_boost = max(1e-6, self.config.concentration_boost / 1000.0)
        surface_boost = math.sqrt(max(1.0, CLAY_CONCENTRATION_FACTOR))
        mod = self.topo.synthesis_mod()

        bulk_request = np.minimum(
            k_syn * self.N * bulk_boost * self.dt_h * mod,
            self.N,
        )
        surface_request = np.minimum(
            k_syn * self.N_surface * surface_boost * self.dt_h * mod,
            self.N_surface,
        )
        requested = bulk_request + surface_request

        # R is a normalized concentration field capped at 1.  If a cell is
        # saturated, only consume the amount that can actually become polymer.
        capacity = np.maximum(0.0, 1.0 - self.R)
        scale = np.ones_like(requested, dtype=float)
        active = requested > 0.0
        scale[active] = np.minimum(1.0, capacity[active] / requested[active])

        bulk_flux = bulk_request * scale
        surface_flux = surface_request * scale
        polymer_flux = bulk_flux + surface_flux

        self.N = np.maximum(self.N - bulk_flux, 0.0)
        self.N_surface = np.maximum(self.N_surface - surface_flux, 0.0)
        self.R = np.minimum(self.R + polymer_flux, 1.0)

    def step_replication_and_selection(self) -> None:
        """STEP 4a – Vectorised RNA replication + fragmentation."""
        rng = self._rng

        # Population events do not create concentration-field material.
        # R is changed only by explicit polymerization/degradation/diffusion.
        self.rna_population.replicate_and_select(
            self.R, self.topo.field, self.topo.curvature,
            self.dt_h, rng,
        )

        self.rna_population.fragment(
            rng, Nx=self.Nx, Ny=self.Ny, dt=self.dt_h,
        )

    def step_degradation(self) -> None:
        """STEP 4b – Temperature-dependent polymer degradation."""
        k_deg  = self.config.k_degradation
        temp_f = math.exp(0.05 * (self.config.temp_C - 25.0))
        mod    = self.topo.degradation_mod()
        dR = -k_deg * self.R * temp_f * self.dt_h * mod
        self.R = np.clip(self.R + dR, 0.0, 1.0)

        self.rna_population.degrade(
            k_deg, temp_f,
            self.topo.field, self.topo.curvature,
            self.dt_h, self._rng,
        )

    def step_diffusion(self) -> None:
        """STEP 5 – Spatial diffusion of all mobile species."""
        dt = self.dt_h
        self.E = diffuse_field(self.E, D_SMALL_MOL_GRID, dt).astype(np.float32)
        self.O = diffuse_field(self.O, D_SMALL_MOL_GRID, dt).astype(np.float32)
        self.N = diffuse_field(self.N, D_SMALL_MOL_GRID, dt).astype(np.float32)
        self.L = diffuse_field(self.L, D_LIPID_GRID,     dt).astype(np.float32)
        self.R = diffuse_field(self.R, D_RNA_GRID,        dt).astype(np.float32)

        for field in (self.E, self.O, self.N, self.L, self.R):
            np.clip(field, 0.0, 1.0, out=field)

    def step_membrane_formation(self) -> None:
        """STEP 6 – Lipid vesicle / membrane formation."""
        T = self.config.temp_C
        if T < -100:
            # Raised from 0.1: amphiphile vesicle formation via mist-droplet mechanism
            # confirmed for Titan-like conditions (Mayer & Nixon, Int.J.Astrobiol. 2025)
            temp_factor = 0.15
        elif T < 0:
            temp_factor = 0.5
        elif T < 70:
            temp_factor = 1.0
        elif T < 100:
            temp_factor = 0.7
        else:
            temp_factor = 0.3

        mod = self.topo.membrane_mod()
        dL = -K_MEMBRANE * self.L * temp_factor * self.dt_h * mod
        dM =  K_MEMBRANE * self.L * temp_factor * self.dt_h * mod
        self.L = np.clip(self.L + dL, 0.0, 1.0)
        self.M = np.clip(self.M + dM, 0.0, 1.0)

    def step_protocell_detection(self) -> None:
        """Detect connected membrane+polymer structures and update observables."""
        observation = self.protocell_detector.detect_components(self.M, self.R)
        self.protocell_count = int(observation['count'])
        self.protocell_area_pixels = int(observation['area_pixels'])

    def nucleotide_material_total(self) -> float:
        """Coarse field-level nucleotide material N_bulk + N_surface + R."""
        if self.N is None or self.N_surface is None or self.R is None:
            raise RuntimeError("simulator is not initialized")
        return float(np.sum(self.N) + np.sum(self.N_surface) + np.sum(self.R))

    def protocell_coverage_fraction(self) -> float:
        """Fraction of grid area occupied by threshold-positive protocell regions."""
        return float(self.protocell_area_pixels) / float(max(1, self.Nx * self.Ny))

    def _validate_finite_fields(self) -> None:
        """Fail closed on non-finite or materially negative state."""
        for name in ('E', 'O', 'N', 'N_surface', 'R', 'M', 'L', 'Cat'):
            field = getattr(self, name)
            if field is None:
                raise RuntimeError(f"{name} is not initialized")
            if not np.isfinite(field).all():
                raise FloatingPointError(f"{name} contains NaN/Inf")
            if float(np.min(field)) < -1e-8:
                raise FloatingPointError(f"{name} contains negative state")

    # ==================================================================
    # ZETA CONSTRAINT APPLICATION
    # ==================================================================

    def _apply_zeta_constraints(self) -> None:
        """Apply Zeta-Riemann spectral constraints to key fields."""
        if self.zeta_modulator is None:
            return
        pc = self.config.euler_phase_coherence
        rng = self._zeta_rng
        self.R = self.zeta_modulator.apply(self.R, rng, pc)
        self.N = self.zeta_modulator.apply(self.N, rng, pc)

    # ==================================================================
    # MAIN LOOP
    # ==================================================================

    def step(self) -> None:
        """Execute one complete simulation time step."""
        self.topo.advance(self.t_h)
        self.step_energy_conversion()
        self.step_catalysis()
        self.step_polymerization()
        self.step_replication_and_selection()
        self.step_degradation()
        self.step_diffusion()
        self.step_membrane_formation()
        if self.config.use_zeta_constraints:
            self._apply_zeta_constraints()
        # Detection must observe the same post-operator state that is recorded.
        self.step_protocell_detection()
        self._validate_finite_fields()
        self.t_h += self.dt_h

    def record_state(self) -> None:
        """Append current metrics to history."""
        self.history['time_h'].append(self.t_h)
        self.history['mean_R'].append(float(np.mean(self.R)))
        self.history['mean_M'].append(float(np.mean(self.M)))
        self.history['n_polymers'].append(self.rna_population.size)
        self.history['n_protocells'].append(self.protocell_count)
        self.history['protocell_area_pixels'].append(self.protocell_area_pixels)
        self.history['surface_monomer_total'].append(float(np.sum(self.N_surface)))
        self.history['nucleotide_material_total'].append(self.nucleotide_material_total())
        self.history['mean_fitness'].append(self.rna_population.mean_fitness())
        self.history['berry_accumulated'].append(self.topo.berry_accumulated)
        self.history['bloch_coherence'].append(self.topo.bloch_coherence())

    def run(
        self,
        hours: float = 120.0,
        record_interval: float = 2.0,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run the simulation and return history as a DataFrame.

        Parameters
        ----------
        hours           : total simulation time (hours)
        record_interval : how often to save metrics (hours)
        verbose         : print progress to stdout
        """
        n_steps      = int(hours / self.dt_h)
        record_every = max(1, int(record_interval / self.dt_h))

        if verbose:
            print(f"\n{'='*70}")
            print(f"SCENARIO {self.config.code}: {self.config.name}")
            print(f"  Location : {self.config.location}")
            print(f"  Temp     : {self.config.temp_C} °C")
            print(f"  Solvent  : {self.config.solvent.value}")
            print(f"  Grid     : {self.Nx}×{self.Ny}  dt={self.dt_h} h  total={hours} h ({n_steps:,} steps)")
            print(f"{'='*70}")

        for step_idx in range(n_steps):
            self.step()
            if step_idx % record_every == 0 or step_idx == n_steps - 1:
                self.record_state()
            if verbose and step_idx % (record_every * 10) == 0:
                print(
                    f"  t={self.t_h:7.1f} h | "
                    f"Polymers={self.rna_population.size:4d} | "
                    f"Proto-cells={self.protocell_count:4d}"
                )

        if verbose:
            print(f"\n✓ Scenario {self.config.code} complete.")
            print(f"  Polymers={self.rna_population.size}  "
                  f"Proto-cells={self.protocell_count}  "
                  f"(expected {self.config.expected_protocells})")

        return pd.DataFrame(self.history)

    # ==================================================================
    # OUTPUT
    # ==================================================================

    def save_outputs(self, prefix: str = "final") -> None:
        """Save CSV history, NPZ field snapshot, and heatmap figure."""
        os.makedirs(self.outdir, exist_ok=True)

        # CSV
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.outdir, f"{prefix}_history.csv"), index=False)

        # NPZ field snapshot
        np.savez_compressed(
            os.path.join(self.outdir, f"{prefix}_fields.npz"),
            E=self.E, O=self.O, N=self.N, N_surface=self.N_surface,
            R=self.R, M=self.M, L=self.L,
            topo=self.topo.field, curvature=self.topo.curvature,
        )

        # Heatmap
        self._save_heatmap(prefix)

        # Summary CSV
        self._save_summary()

    def _save_heatmap(self, prefix: str) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        pairs = [
            (self.R,              'R – genetic polymer'),
            (self.M,              'M – membrane'),
            (self.L,              'L – lipids'),
            (self.topo.field,     'Topology field (Berry analog)'),
            (self.topo.curvature, 'Topo curvature (Laplacian)'),
            (self.E,              'E – energy carriers'),
        ]
        for ax, (data, title) in zip(axs.flat, pairs):
            im = ax.imshow(data, origin='lower', aspect='auto')
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax)

        fig.suptitle(f"Scenario {self.config.code}: {self.config.name}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, f"{prefix}_heatmaps.png"), dpi=120)
        plt.close(fig)

    def _save_summary(self) -> None:
        summary = {
            'Scenario':        self.config.code,
            'Name':            self.config.name,
            'Location':        self.config.location,
            'Temp_C':          self.config.temp_C,
            'Solvent':         self.config.solvent.value,
            'Energy_Source':   self.config.energy_source,
            'Catalyst':        self.config.catalyst,
            'Final_Polymers':  self.rna_population.size,
            'Final_ProtoC':    self.protocell_count,
            'Final_ProtoC_Area_Pixels': self.protocell_area_pixels,
            'Initial_Condition': 'PRESEEDED_RNA' if self._preseed_rna else 'POLYMER_FREE',
            'Expected_ProtoC': self.config.expected_protocells,
            'Success_Rate_pct': round(
                100.0 * self.protocell_count / max(1, self.config.expected_protocells), 1
            ),
            'Timescale':       self.config.timescale_description,
        }
        pd.DataFrame([summary]).to_csv(
            os.path.join(self.outdir, 'summary.csv'), index=False
        )
