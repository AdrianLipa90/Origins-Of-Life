"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║                 DUAL-SCENARIO ORIGIN OF LIFE MODEL V3.0                        ║
║                                                                                ║
║              Complete simulation with TWO independent scenarios                ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

SCENARIO A: Shallow Ocean + UV (Clay Hypothesis)
SCENARIO B: Deep-Sea Hydrothermal Vents (Iron-Sulfur World)

Both scenarios simulate life's emergence independently to compare pathways.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ============================================================================
# SCENARIO-SPECIFIC PARAMETERS
# ============================================================================

@dataclass
class ScenarioParameters:
    """Parameters for each environmental scenario."""

    # Identification
    name: str
    location: str

    # Physical
    depth_m: float
    UV_flux_W_m2: float
    temp_C: float
    pH: float

    # Chemical
    energy_source: str
    primary_minerals: str
    mineral_conc_g_L: float
    redox_state: str

    # Kinetic modifiers
    k_energy_conversion: float  # Rate of energy → organics
    k_mineral_catalysis: float  # Mineral catalytic enhancement
    k_rna_synthesis: float      # RNA polymerization rate
    k_rna_degradation: float    # RNA degradation rate
    catalyst_boost: float       # Overall catalytic boost factor


# SCENARIO A: SHALLOW UV
SHALLOW_UV = ScenarioParameters(
    name="Shallow Ocean + UV",
    location="Tidal zones, coastal areas, depth 10m",
    depth_m=10.0,
    UV_flux_W_m2=30.0,
    temp_C=65.0,
    pH=7.5,
    energy_source="Photochemistry (UV → organics)",
    primary_minerals="Montmorillonite clay (Al-Si)",
    mineral_conc_g_L=5.0,
    redox_state="Mildly oxidizing",
    k_energy_conversion=0.35,   # UV photochemistry
    k_mineral_catalysis=7.5,    # Clay catalysis (from Ferris)
    k_rna_synthesis=0.15,       # Clay-catalyzed RNA synthesis
    k_rna_degradation=0.0042,   # Protected by clay
    catalyst_boost=1000.0       # Concentration factor on clay
)

# SCENARIO B: DEEP HYDROTHERMAL
DEEP_HYDROTHERMAL = ScenarioParameters(
    name="Deep-Sea Hydrothermal Vents",
    location="Mid-ocean ridges, volcanic vents, depth 2000m",
    depth_m=2000.0,
    UV_flux_W_m2=0.0,           # NO UV!
    temp_C=90.0,
    pH=9.0,                     # Alkaline vents
    energy_source="Chemosynthesis (H2 + CO2 → organics)",
    primary_minerals="Fe-S, Ni-S clusters (pyrite, greigite)",
    mineral_conc_g_L=10.0,      # Higher mineral content
    redox_state="Strongly reducing",
    k_energy_conversion=0.08,   # Chemosynthesis (slower than UV)
    k_mineral_catalysis=15.0,   # Fe-S catalysis (very strong!)
    k_rna_synthesis=0.05,       # Slower at high temp
    k_rna_degradation=0.02,     # Faster at 90°C (thermophilic adapted)
    catalyst_boost=500.0        # Concentration in mineral pores
)


# ============================================================================
# DUAL-SCENARIO SIMULATOR
# ============================================================================

class DualScenarioSimulator:
    """Simulate life's origin in TWO independent scenarios."""

    def __init__(self, scenario: ScenarioParameters, 
                 Nx: int = 64, Ny: int = 64, dt_h: float = 0.02):
        self.scenario = scenario
        self.Nx, self.Ny = Nx, Ny
        self.dt_h = dt_h
        self.t_h = 0.0

        # Chemical fields
        self.E = None  # Energy carriers (UV products or H2)
        self.O = None  # Organic precursors
        self.N = None  # Activated nucleotides
        self.R = None  # RNA concentration
        self.M = None  # Membrane concentration
        self.L = None  # Lipid concentration

        # Mineral field
        self.Min = None  # Mineral catalyst distribution

        # Biological objects
        self.rna_population: List = []
        self.protocell_count = 0

        # History
        self.history = {
            'time_h': [],
            'mean_R': [],
            'mean_M': [],
            'n_rna': [],
            'n_protocells': [],
            'mean_rna_fitness': []
        }

    def initialize(self):
        """Initialize chemical fields and minerals."""

        # Initialize fields
        self.E = np.random.uniform(0.1, 0.3, (self.Nx, self.Ny))
        self.O = np.random.uniform(0.05, 0.15, (self.Nx, self.Ny))
        self.N = np.random.uniform(0.01, 0.05, (self.Nx, self.Ny))
        self.R = np.zeros((self.Nx, self.Ny))
        self.M = np.zeros((self.Nx, self.Ny))
        self.L = np.random.uniform(0.005, 0.01, (self.Nx, self.Ny))

        # Mineral distribution
        self.Min = np.random.uniform(
            self.scenario.mineral_conc_g_L * 0.8,
            self.scenario.mineral_conc_g_L * 1.2,
            (self.Nx, self.Ny)
        )

        # Seed RNA population
        for _ in range(10):
            seq_len = np.random.randint(30, 60)
            sequence = ''.join(np.random.choice(['A', 'U', 'G', 'C'], seq_len))
            pos = (np.random.randint(0, self.Nx), np.random.randint(0, self.Ny))
            fitness = np.random.uniform(0.3, 0.6)
            self.rna_population.append({
                'sequence': sequence,
                'position': pos,
                'fitness': fitness,
                'age': 0.0
            })

    def step_energy_conversion(self):
        """STEP 1: Energy source → organic precursors."""

        if self.scenario.UV_flux_W_m2 > 0:
            # SCENARIO A: UV photochemistry
            k = self.scenario.k_energy_conversion
            dE_dt = -k * self.E
            dO_dt = +k * self.E * 0.8  # 80% efficiency
        else:
            # SCENARIO B: Chemosynthesis (H2 + CO2)
            k = self.scenario.k_energy_conversion
            # H2 availability modulates rate
            h2_factor = 1.0 + 0.5 * np.sin(2 * np.pi * self.t_h / 48.0)
            dE_dt = -k * self.E * h2_factor
            dO_dt = +k * self.E * h2_factor * 0.6  # 60% efficiency

        self.E += dE_dt * self.dt_h
        self.O += dO_dt * self.dt_h

        self.E = np.clip(self.E, 0, 1)
        self.O = np.clip(self.O, 0, 1)

    def step_mineral_catalysis(self):
        """STEP 2: Mineral-catalyzed nucleotide activation."""

        k_cat = self.scenario.k_mineral_catalysis

        # Mineral catalysis: O + Min → N
        catalyst_effect = self.Min / (self.Min + 5.0)  # Saturation

        dO_dt = -0.1 * k_cat * self.O * catalyst_effect
        dN_dt = +0.1 * k_cat * self.O * catalyst_effect

        self.O += dO_dt * self.dt_h
        self.N += dN_dt * self.dt_h

        self.O = np.clip(self.O, 0, 1)
        self.N = np.clip(self.N, 0, 1)

    def step_rna_synthesis(self):
        """STEP 3: RNA polymerization."""

        k_synth = self.scenario.k_rna_synthesis

        # N → R (mineral-catalyzed)
        catalyst_boost = self.scenario.catalyst_boost / 1000.0

        dN_dt = -k_synth * self.N * catalyst_boost
        dR_dt = +k_synth * self.N * catalyst_boost

        self.N += dN_dt * self.dt_h
        self.R += dR_dt * self.dt_h

        self.N = np.clip(self.N, 0, 1)
        self.R = np.clip(self.R, 0, 1)

    def step_rna_degradation(self):
        """STEP 4: RNA degradation (temperature-dependent)."""

        k_deg = self.scenario.k_rna_degradation

        # Temperature effect
        T = self.scenario.temp_C
        T_ref = 65.0
        temp_factor = np.exp(0.05 * (T - T_ref))  # Arrhenius-like

        dR_dt = -k_deg * self.R * temp_factor

        self.R += dR_dt * self.dt_h
        self.R = np.clip(self.R, 0, 1)

        # Update RNA population
        for rna in self.rna_population:
            rna['age'] += self.dt_h
            if np.random.random() < k_deg * temp_factor * self.dt_h:
                self.rna_population.remove(rna)

    def step_lipid_aggregation(self):
        """STEP 5: Lipid membrane formation."""

        k_agg = 0.4

        # Temperature-dependent aggregation
        T = self.scenario.temp_C
        if T < 70:
            temp_factor = 1.0
        elif T < 90:
            temp_factor = 0.7  # Less stable at high temp
        else:
            temp_factor = 0.3  # Very unstable >90°C

        dL_dt = -k_agg * self.L * temp_factor
        dM_dt = +k_agg * self.L * temp_factor

        self.L += dL_dt * self.dt_h
        self.M += dM_dt * self.dt_h

        self.L = np.clip(self.L, 0, 1)
        self.M = np.clip(self.M, 0, 1)

    def step_protocell_detection(self):
        """STEP 6: Identify proto-cells."""

        threshold_M = 0.05
        threshold_R = 0.03

        protocells = np.where((self.M > threshold_M) & (self.R > threshold_R))
        self.protocell_count = len(protocells[0])

    def record_state(self):
        """Record current state."""

        self.history['time_h'].append(self.t_h)
        self.history['mean_R'].append(np.mean(self.R))
        self.history['mean_M'].append(np.mean(self.M))
        self.history['n_rna'].append(len(self.rna_population))
        self.history['n_protocells'].append(self.protocell_count)

        if len(self.rna_population) > 0:
            fitnesses = [rna['fitness'] for rna in self.rna_population]
            self.history['mean_rna_fitness'].append(np.mean(fitnesses))
        else:
            self.history['mean_rna_fitness'].append(0)

    def step(self):
        """Execute one time step."""

        self.step_energy_conversion()
        self.step_mineral_catalysis()
        self.step_rna_synthesis()
        self.step_rna_degradation()
        self.step_lipid_aggregation()
        self.step_protocell_detection()

        self.t_h += self.dt_h

    def run(self, hours: float = 120, record_interval: float = 2.0):
        """Run simulation."""

        n_steps = int(hours / self.dt_h)
        record_steps = int(record_interval / self.dt_h)

        print(f"\nRunning {self.scenario.name}...")
        print(f"  Duration: {hours}h, Steps: {n_steps:,}")

        for step in range(n_steps):
            self.step()

            if step % record_steps == 0:
                self.record_state()

                if step % (record_steps * 10) == 0:
                    print(f"  t={self.t_h:6.1f}h | " +
                          f"RNA={len(self.rna_population):3d} | " +
                          f"Proto-cells={self.protocell_count:3d}")

        print(f"\n✅ {self.scenario.name} complete!")
        print(f"   Final: {len(self.rna_population)} RNA, " +
              f"{self.protocell_count} proto-cells")


# ============================================================================
# MAIN EXECUTION - RUN BOTH SCENARIOS
# ============================================================================

if __name__ == "__main__":

    print("="*80)
    print("DUAL-SCENARIO ORIGIN OF LIFE SIMULATION V3.0")
    print("="*80)

    # Run SCENARIO A: Shallow UV
    print("\n" + "="*80)
    print("SCENARIO A: SHALLOW OCEAN + UV")
    print("="*80)

    sim_A = DualScenarioSimulator(SHALLOW_UV, Nx=64, Ny=64, dt_h=0.02)
    sim_A.initialize()
    sim_A.run(hours=120, record_interval=2.0)

    df_A = pd.DataFrame(sim_A.history)
    df_A.to_csv('scenario_A_shallow_UV_results.csv', index=False)

    # Run SCENARIO B: Deep Hydrothermal
    print("\n" + "="*80)
    print("SCENARIO B: DEEP-SEA HYDROTHERMAL VENTS")
    print("="*80)

    sim_B = DualScenarioSimulator(DEEP_HYDROTHERMAL, Nx=64, Ny=64, dt_h=0.02)
    sim_B.initialize()
    sim_B.run(hours=120, record_interval=2.0)

    df_B = pd.DataFrame(sim_B.history)
    df_B.to_csv('scenario_B_deep_hydrothermal_results.csv', index=False)

    # COMPARISON
    print("\n" + "="*80)
    print("SCENARIO COMPARISON")
    print("="*80)

    comparison = pd.DataFrame({
        'Scenario': [sim_A.scenario.name, sim_B.scenario.name],
        'Final_RNA': [len(sim_A.rna_population), len(sim_B.rna_population)],
        'Final_ProtoC': [sim_A.protocell_count, sim_B.protocell_count],
        'Mean_R': [np.mean(df_A['mean_R']), np.mean(df_B['mean_R'])],
        'Mean_M': [np.mean(df_A['mean_M']), np.mean(df_B['mean_M'])],
        'UV_Flux': [sim_A.scenario.UV_flux_W_m2, sim_B.scenario.UV_flux_W_m2],
        'Temp_C': [sim_A.scenario.temp_C, sim_B.scenario.temp_C],
        'Minerals': [sim_A.scenario.primary_minerals, sim_B.scenario.primary_minerals]
    })

    print(comparison.to_string(index=False))

    comparison.to_csv('dual_scenario_comparison.csv', index=False)

    print("\n" + "="*80)
    print("FILES SAVED:")
    print("  • scenario_A_shallow_UV_results.csv")
    print("  • scenario_B_deep_hydrothermal_results.csv")
    print("  • dual_scenario_comparison.csv")
    print("="*80)
