import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

# ============================================================================
# SCENARIO CONFIGURATION
# ============================================================================

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
    topo_strength: float  # new: magnitude of topological effect
    topo_pattern: str     # new: pattern type (sin/cos/random etc)

# ============================================================================
# EXAMPLE SCENARIOS WITH TOPOLOGY PARAMS
# ============================================================================

SCENARIO_A = ScenarioConfig(
    name="Shallow Ocean + UV (Clay Hypothesis)",
    code="A",
    location="Earth - Coastal tidal zones",
    temp_C=65.0,
    pressure_atm=1.0,
    UV_flux=30.0,
    solvent="H2O",
    pH=7.5,
    redox="Mildly oxidizing",
    energy_source="UV photochemistry",
    k_energy=0.35,
    catalyst="Montmorillonite clay",
    k_catalysis=7.5,
    concentration_boost=1000.0,
    k_synthesis=0.15,
    k_degradation=0.0042,
    expected_protocells=600,
    timescale_description="Hours",
    topo_strength=0.25,
    topo_pattern='sin'
)

# (Analogicznie można zdefiniować B–E)
ALL_SCENARIOS = [SCENARIO_A]  # dla przykładu, reszta do dodania

# ============================================================================
# UNIVERSAL ORIGIN SIMULATOR WITH KÄHLER-BERRY-EULER TOPOLOGY
# ============================================================================

class UniversalOriginSimulator:
    def __init__(self, config: ScenarioConfig, Nx=64, Ny=64, dt_h=0.02):
        self.config = config
        self.Nx, self.Ny = Nx, Ny
        self.dt_h = dt_h
        self.t_h = 0.0

        # Chemical fields
        self.E = None
        self.O = None
        self.N = None
        self.R = None
        self.M = None
        self.L = None
        self.Cat = None

        # Tracking
        self.rna_population = []
        self.protocell_count = 0
        self.history = {'time_h': [], 'mean_R': [], 'mean_M': [], 
                        'n_polymers': [], 'n_protocells': [], 'mean_fitness': []}

        # Topological Kähler field (Berry–Euler)
        self.topo_field = None
        self._initialize_topology()

    def _initialize_topology(self):
        """Create global Kähler-like topological field."""
        x = np.linspace(-1,1,self.Nx)
        y = np.linspace(-1,1,self.Ny)
        X,Y = np.meshgrid(x,y,indexing='ij')
        s = self.config.topo_strength
        if self.config.topo_pattern == 'sin':
            self.topo_field = s*np.sin(2*np.pi*X)*np.cos(2*np.pi*Y)
        elif self.config.topo_pattern == 'cos':
            self.topo_field = s*np.cos(2*np.pi*X)*np.sin(2*np.pi*Y)
        elif self.config.topo_pattern == 'random':
            rng = np.random.default_rng(42)
            self.topo_field = s * rng.standard_normal((self.Nx,self.Ny))
        else:
            self.topo_field = s * np.ones((self.Nx,self.Ny))  # flat

    def initialize(self):
        """Initialize chemical fields and RNA population."""
        temp_factor = np.exp((self.config.temp_C - 25)/100.0)
        self.E = np.random.uniform(0.1,0.3,(self.Nx,self.Ny))*temp_factor
        self.O = np.random.uniform(0.05,0.15,(self.Nx,self.Ny))
        self.N = np.random.uniform(0.01,0.05,(self.Nx,self.Ny))
        self.R = np.zeros((self.Nx,self.Ny))
        self.M = np.zeros((self.Nx,self.Ny))
        self.L = np.random.uniform(0.005,0.01,(self.Nx,self.Ny))
        self.Cat = np.random.uniform(0.8,1.2,(self.Nx,self.Ny))

        # Seed RNA population
        n_seed = max(5,int(20*temp_factor))
        for _ in range(n_seed):
            self.rna_population.append({
                'length': np.random.randint(20,50),
                'position': (np.random.randint(0,self.Nx), np.random.randint(0,self.Ny)),
                'fitness': np.random.uniform(0.3,0.6),
                'age': 0.0
            })

    # --------------------------
    # VECTORISED CHEMICAL STEPS
    # --------------------------

    def step_energy_conversion(self):
        k = self.config.k_energy
        efficiency = 0.8 if self.config.UV_flux>0 else 0.6
        dE = -k*self.E*self.dt_h
        dO = k*self.E*efficiency*self.dt_h * (1 + self.topo_field)  # topological modulation
        self.E = np.clip(self.E + dE,0,1)
        self.O = np.clip(self.O + dO,0,1)

    def step_catalysis(self):
        k_cat = self.config.k_catalysis * 0.1
        catalyst_effect = self.Cat/(self.Cat+1.0)
        dO = -k_cat*self.O*catalyst_effect*self.dt_h*(1 + self.topo_field)
        dN = k_cat*self.O*catalyst_effect*self.dt_h*(1 + self.topo_field)
        self.O = np.clip(self.O + dO,0,1)
        self.N = np.clip(self.N + dN,0,1)

    def step_polymerization(self):
        k_syn = self.config.k_synthesis
        boost = self.config.concentration_boost/1000.0
        dN = -k_syn*self.N*boost*self.dt_h*(1 + self.topo_field)
        dR = k_syn*self.N*boost*self.dt_h*(1 + self.topo_field)
        self.N = np.clip(self.N + dN,0,1)
        self.R = np.clip(self.R + dR,0,1)

    def step_degradation(self):
        k_deg = self.config.k_degradation
        temp_factor = np.exp(0.05*(self.config.temp_C-25))
        dR = -k_deg*self.R*temp_factor*self.dt_h*(1 + self.topo_field)
        self.R = np.clip(self.R + dR,0,1)
        # RNA population degradation
        for poly in list(self.rna_population):
            poly['age'] += self.dt_h
            if np.random.random() < k_deg*temp_factor*self.dt_h:
                self.rna_population.remove(poly)

    def step_membrane_formation(self):
        k_mem = 0.4
        T = self.config.temp_C
        temp_factor = 1.0 if 0<=T<70 else 0.3
        dL = -k_mem*self.L*temp_factor*self.dt_h*(1 + self.topo_field)
        dM = k_mem*self.L*temp_factor*self.dt_h*(1 + self.topo_field)
        self.L = np.clip(self.L + dL,0,1)
        self.M = np.clip(self.M + dM,0,1)

    def step_protocell_detection(self):
        threshold_M, threshold_R = 0.05, 0.03
        protocells = np.where((self.M>threshold_M)&(self.R>threshold_R))
        self.protocell_count = len(protocells[0])

    def record_state(self):
        self.history['time_h'].append(self.t_h)
        self.history['mean_R'].append(np.mean(self.R))
        self.history['mean_M'].append(np.mean(self.M))
        self.history['n_polymers'].append(len(self.rna_population))
        self.history['n_protocells'].append(self.protocell_count)
        if len(self.rna_population)>0:
            self.history['mean_fitness'].append(np.mean([p['fitness'] for p in self.rna_population]))
        else:
            self.history['mean_fitness'].append(0)

    def step(self):
        self.step_energy_conversion()
        self.step_catalysis()
        self.step_polymerization()
        self.step_degradation()
        self.step_membrane_formation()
        self.step_protocell_detection()
        self.t_h += self.dt_h

    def run(self, hours=120, record_interval=2.0):
        n_steps = int(hours/self.dt_h)
        record_steps = int(record_interval/self.dt_h)
        for step in range(n_steps):
            self.step()
            if step%record_steps==0:
                self.record_state()
        return pd.DataFrame(self.history)