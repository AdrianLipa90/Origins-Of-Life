from dataclasses import dataclass


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


SCENARIO_A = ScenarioConfig(
    name="Shallow Ocean + UV (Clay Hypothesis)",
    code="A",
    location="Earth - Coastal tidal zones, 10m depth",
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
    topo_pattern="sin",
    topo_time_dependence="pulsing",
    topo_pulse_freq=0.05,
    seed=101,
)

SCENARIO_B = ScenarioConfig(
    name="Deep-Sea Hydrothermal Vents (Iron-Sulfur World)",
    code="B",
    location="Earth - Mid-ocean ridges",
    temp_C=90.0,
    pressure_atm=200.0,
    UV_flux=0.0,
    solvent="H2O",
    pH=9.0,
    redox="Strongly reducing",
    energy_source="Chemosynthesis (H2 + CO2)",
    k_energy=0.08,
    catalyst="Fe-S clusters",
    k_catalysis=15.0,
    concentration_boost=500.0,
    k_synthesis=0.05,
    k_degradation=0.02,
    expected_protocells=400,
    timescale_description="Days",
    topo_strength=0.35,
    topo_pattern="vortex",
    topo_time_dependence="static",
    seed=202,
)

SCENARIO_C = ScenarioConfig(
    name="Ammonia-Based Biochemistry",
    code="C",
    location="Cold NH3 worlds",
    temp_C=-55.0,
    pressure_atm=1.0,
    UV_flux=10.0,
    solvent="NH3",
    pH=11.0,
    redox="Variable",
    energy_source="UV + Chemosynthesis",
    k_energy=0.02,
    catalyst="NH3-ice minerals",
    k_catalysis=5.0,
    concentration_boost=300.0,
    k_synthesis=0.01,
    k_degradation=0.001,
    expected_protocells=100,
    timescale_description="Weeks",
    topo_strength=0.15,
    topo_pattern="gauss",
    topo_time_dependence="drift",
    topo_pulse_freq=0.005,
    seed=303,
)

SCENARIO_D = ScenarioConfig(
    name="Titan Methane Lakes (Hydrocarbon Biochemistry)",
    code="D",
    location="Titan - Kraken Mare",
    temp_C=-179.0,
    pressure_atm=1.45,
    UV_flux=2.0,
    solvent="CH4/C2H6",
    pH=7.0,
    redox="Non-oxidizing",
    energy_source="Atmospheric photochemistry",
    k_energy=0.001,
    catalyst="Tholins",
    k_catalysis=2.0,
    concentration_boost=100.0,
    k_synthesis=0.001,
    k_degradation=0.0001,
    expected_protocells=30,
    timescale_description="1000s hours",
    topo_strength=0.05,
    topo_pattern="random",
    topo_time_dependence="static",
    seed=404,
)

SCENARIO_E = ScenarioConfig(
    name="Enceladus Subsurface Ocean",
    code="E",
    location="Enceladus - Sub-ice ocean",
    temp_C=4.0,
    pressure_atm=800.0,
    UV_flux=0.0,
    solvent="H2O",
    pH=9.5,
    redox="Reducing (H2-rich)",
    energy_source="Hydrothermal (tidal heating)",
    k_energy=0.08,
    catalyst="Fe-S + Mg-silicates",
    k_catalysis=12.0,
    concentration_boost=600.0,
    k_synthesis=0.08,
    k_degradation=0.01,
    expected_protocells=450,
    timescale_description="Days",
    topo_strength=0.28,
    topo_pattern="cos",
    topo_time_dependence="pulsing",
    topo_pulse_freq=0.02,
    seed=505,
)

ALL_SCENARIOS = [SCENARIO_A, SCENARIO_B, SCENARIO_C, SCENARIO_D, SCENARIO_E]
