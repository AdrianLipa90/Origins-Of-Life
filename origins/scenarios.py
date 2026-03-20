"""
All 5 origin-of-life scenario configurations.

Scenarios:
  A – Shallow UV + Clay (Hadean Earth)
  B – Deep-Sea Hydrothermal Vents (Iron-Sulfur World)
  C – Ammonia-Based Biochemistry (Cold Worlds)
  D – Titan Methane Lakes (Hydrocarbon Biochemistry)
  E – Enceladus Subsurface Ocean (Under-Ice Life)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TopologyPattern(Enum):
    SINUSOIDAL   = "sin"
    COSINUSOIDAL = "cos"
    VORTEX       = "vortex"
    GAUSSIAN     = "gauss"
    RANDOM       = "random"
    STATIC       = "static"


class TimeDependence(Enum):
    STATIC  = "static"
    PULSING = "pulsing"
    DRIFT   = "drift"


class SolventType(Enum):
    WATER          = "H2O"
    AMMONIA        = "NH3"
    METHANE_ETHANE = "CH4/C2H6"


@dataclass
class ScenarioConfig:
    """Complete configuration for any origin-of-life scenario."""

    # Identification
    name:     str
    code:     str
    location: str

    # Physical environment
    temp_C:       float
    pressure_atm: float
    UV_flux:      float
    solvent:      SolventType
    pH:           float
    redox:        str

    # Energy & catalysis
    energy_source:      str
    k_energy:           float
    catalyst:           str
    k_catalysis:        float
    concentration_boost: float

    # Chemical kinetics
    k_synthesis:   float
    k_degradation: float

    # Outcomes
    expected_protocells:   int
    timescale_description: str

    # Topology (Kähler-Berry-Euler)
    topo_strength:         float
    topo_pattern:          TopologyPattern
    topo_time_dependence:  TimeDependence
    topo_pulse_freq:       float = 0.0

    # Zeta-Riemann / Heisenberg constraints
    use_zeta_constraints:  bool  = True
    zeta_lambda_soft:      float = 5.0
    zeta_sigma_heis:       float = 0.001
    euler_phase_coherence: float = 0.8
    quantum_fluctuations:  bool  = True

    seed: int = 42

    def __post_init__(self):
        if isinstance(self.solvent, str):
            self.solvent = SolventType(self.solvent)
        if isinstance(self.topo_pattern, str):
            self.topo_pattern = TopologyPattern(self.topo_pattern)
        if isinstance(self.topo_time_dependence, str):
            self.topo_time_dependence = TimeDependence(self.topo_time_dependence)


# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

SCENARIO_A = ScenarioConfig(
    name="Shallow Ocean + UV (Clay Hypothesis)",
    code="A",
    location="Earth – Coastal tidal zones, 10 m depth",
    temp_C=65.0, pressure_atm=1.0, UV_flux=30.0,
    solvent=SolventType.WATER, pH=7.5, redox="Mildly oxidizing",
    energy_source="UV photochemistry", k_energy=0.35,
    catalyst="Montmorillonite clay (Al-Si)", k_catalysis=7.5, concentration_boost=1000.0,
    k_synthesis=0.15, k_degradation=0.0042,
    expected_protocells=600, timescale_description="Hours (rapid emergence)",
    topo_strength=0.25, topo_pattern=TopologyPattern.SINUSOIDAL,
    topo_time_dependence=TimeDependence.PULSING, topo_pulse_freq=0.05,
    use_zeta_constraints=True, zeta_lambda_soft=6.0, zeta_sigma_heis=0.0005,
    euler_phase_coherence=0.9, quantum_fluctuations=True, seed=101,
)

SCENARIO_B = ScenarioConfig(
    name="Deep-Sea Hydrothermal Vents (Iron-Sulfur World)",
    code="B",
    location="Earth – Mid-ocean ridges, 2000 m depth",
    temp_C=90.0, pressure_atm=200.0, UV_flux=0.0,
    solvent=SolventType.WATER, pH=9.0, redox="Strongly reducing",
    energy_source="Chemosynthesis (H₂ + CO₂)", k_energy=0.08,
    catalyst="Fe-S clusters (pyrite, greigite)", k_catalysis=15.0, concentration_boost=500.0,
    k_synthesis=0.05, k_degradation=0.02,
    expected_protocells=400, timescale_description="Days (moderate pace)",
    topo_strength=0.35, topo_pattern=TopologyPattern.VORTEX,
    topo_time_dependence=TimeDependence.STATIC,
    use_zeta_constraints=True, zeta_lambda_soft=4.0, zeta_sigma_heis=0.001,
    euler_phase_coherence=0.7, quantum_fluctuations=True, seed=202,
)

SCENARIO_C = ScenarioConfig(
    name="Ammonia-Based Biochemistry (Exotic Life)",
    code="C",
    location="Cold moons/planets with liquid NH₃",
    temp_C=-55.0, pressure_atm=1.0, UV_flux=10.0,
    solvent=SolventType.AMMONIA, pH=11.0, redox="Variable",
    energy_source="UV + Chemosynthesis", k_energy=0.02,
    catalyst="NH₃-ice minerals", k_catalysis=5.0, concentration_boost=300.0,
    k_synthesis=0.01, k_degradation=0.001,
    expected_protocells=100, timescale_description="Weeks (slow but stable)",
    topo_strength=0.15, topo_pattern=TopologyPattern.GAUSSIAN,
    topo_time_dependence=TimeDependence.DRIFT, topo_pulse_freq=0.005,
    use_zeta_constraints=True, zeta_lambda_soft=3.0, zeta_sigma_heis=0.002,
    euler_phase_coherence=0.6, quantum_fluctuations=False, seed=303,
)

SCENARIO_D = ScenarioConfig(
    name="Titan Methane Lakes (Hydrocarbon Biochemistry)",
    code="D",
    location="Titan (Saturn) – Kraken Mare, −179 °C",
    # pressure: 1.5 atm (ESA Cassini-Huygens); UV: 2.5 W/m² (tholins confirmed)
    temp_C=-179.0, pressure_atm=1.5, UV_flux=2.5,
    solvent=SolventType.METHANE_ETHANE, pH=7.0, redox="Non-oxidizing",
    # k_energy +50%: richer tholin photochemistry (NASA 2025)
    energy_source="Atmospheric photochemistry + tholins", k_energy=0.0015,
    # amphiphile vesicle mechanism via mist droplets (Mayer & Nixon, Int.J.Astrobiol. 2025)
    catalyst="Tholins + HCN + amphiphiles (Mayer & Nixon 2025)", k_catalysis=2.5,
    # concentration_boost up: mist-droplet focusing mechanism (2025)
    concentration_boost=120.0,
    k_synthesis=0.0012, k_degradation=0.0001,
    # expected_protocells revised up: new vesicle-formation pathway (2025)
    expected_protocells=45, timescale_description="1000s of hours (glacial pace, new pathway 2025)",
    topo_strength=0.05, topo_pattern=TopologyPattern.RANDOM,
    topo_time_dependence=TimeDependence.STATIC,
    use_zeta_constraints=False, quantum_fluctuations=False, seed=404,
)

SCENARIO_E = ScenarioConfig(
    name="Enceladus Subsurface Ocean (Under-Ice Life)",
    code="E",
    location="Enceladus (Saturn) – 40 km under ice, serpentinization",
    temp_C=4.0, pressure_atm=800.0, UV_flux=0.0,
    solvent=SolventType.WATER,
    # pH 8.8: phosphates confirm alkaline ocean (Geochimica et Cosmochimica Acta 2025)
    pH=8.8, redox="Strongly reducing (H₂ from serpentinization)",
    # k_energy +25%: serpentinization H₂ confirmed more potent energy source (NASA 2024)
    energy_source="Hydrothermal (serpentinization + tidal heating)", k_energy=0.10,
    # k_catalysis -12%: Co/Cu depletion constrains methanogenesis (JAMSTEC 2025)
    # HCN + complex organics detected in fresh ice grains (Nature Astronomy, Mar 2025)
    catalyst="Fe-S + Mg-silicates + HCN (Nature Astronomy 2025)", k_catalysis=10.5,
    concentration_boost=650.0,
    # k_synthesis +12%: aryl/aliphatic organics detected directly (Nature Astronomy 2025)
    k_synthesis=0.09, k_degradation=0.008,
    expected_protocells=480, timescale_description="Days (highly favourable, NASA 2025)",
    topo_strength=0.28, topo_pattern=TopologyPattern.COSINUSOIDAL,
    topo_time_dependence=TimeDependence.PULSING, topo_pulse_freq=0.02,
    use_zeta_constraints=True, zeta_lambda_soft=5.0, zeta_sigma_heis=0.001,
    euler_phase_coherence=0.75, quantum_fluctuations=True, seed=505,
)

ALL_SCENARIOS = [SCENARIO_A, SCENARIO_B, SCENARIO_C, SCENARIO_D, SCENARIO_E]
SCENARIOS_BY_CODE = {s.code: s for s in ALL_SCENARIOS}
