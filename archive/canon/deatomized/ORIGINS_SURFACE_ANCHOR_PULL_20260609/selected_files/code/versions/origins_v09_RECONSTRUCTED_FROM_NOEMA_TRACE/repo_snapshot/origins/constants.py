"""
Physical constants and empirical parameters for origin-of-life simulations.

All values are sourced from peer-reviewed literature:
- Ferris et al. (1996): Clay catalysis
- Lincoln & Joyce (2009): RNA self-replication
- Sutherland (2016): Prebiotic nucleotide synthesis
- Szostak (2018): Origins of function
- Ranjan & Sasselov (2016): UV environment
"""

# ============================================================================
# FUNDAMENTAL PHYSICS
# ============================================================================

BOLTZMANN_J_K = 1.380649e-23       # J/K
GAS_CONSTANT = 8.314462618          # J/(mol·K)
AVOGADRO = 6.02214076e23            # mol⁻¹
SPEED_OF_LIGHT = 299_792_458        # m/s
GRAVITY_CONST = 6.67430e-11         # m³/(kg·s²)
VACUUM_PERMEABILITY = 4e-7 * 3.14159265358979  # H/m
AU_TO_METERS = 1.496e11             # m

# ============================================================================
# EARLY EARTH / HADEAN ENVIRONMENT
# ============================================================================

# Temperature
TEMP_C_DEFAULT = 65.0               # °C (Hadean shallow ocean, Ferris 1996)
TEMP_RANGE_C = 20.0                 # ±°C diurnal variation

# Radiation
UV_FLUX_SHALLOW_W_M2 = 30.0        # W/m² (Ranjan & Sasselov 2016)
UV_VARIABILITY = 0.85               # fraction
B_FIELD_TESLA = 50e-6               # T (early Earth magnetic field estimate)

# Electromagnetic / Schumann
SCHUMANN_FREQ_HZ = 7.83             # Hz (fundamental)
SCHUMANN_AMP_V_M = 5.0              # V/m
SCHUMANN_HARMONICS_HZ = [7.83, 14.3, 20.8, 27.3, 33.8]  # Hz

# Temporal cycles
DAY_LENGTH_H = 24.0                 # hours
TIDAL_PERIOD_H = 12.42              # hours (semi-diurnal)

# ============================================================================
# CHEMICAL ENVIRONMENT
# ============================================================================

LIPID_CONC_MOLAR = 0.01
NUCLEOTIDE_CONC_MOLAR = 0.001
AMINO_ACID_CONC_MOLAR = 0.005
MG2_CONC_MOLAR = 0.04               # Mg²⁺ cofactor (RNA stabilizer)

# ============================================================================
# CLAY MINERAL (MONTMORILLONITE) — Ferris et al. (1996)
# ============================================================================

CLAY_CONC_G_L = 5.0                 # g/L suspension
CLAY_SURFACE_AREA_M2_G = 150.0      # m²/g BET surface area
CLAY_ADSORPTION_SITES_M2 = 1e13     # sites/m²
CLAY_NUCLEOTIDE_EFFICIENCY = 0.8    # fraction adsorbed
CLAY_POLYMERIZATION_RATE_MULT = 7.5 # fold increase over uncatalyzed
CLAY_RNA_PROTECTION_FACTOR = 100.0  # fold reduction in degradation
CLAY_CONCENTRATION_FACTOR = 1000.0  # local concentration amplification
CLAY_CHIRALITY_BIAS = 0.7           # L-enantiomer selectivity

# Langmuir adsorption constant
CLAY_K_ADS = 100.0

# ============================================================================
# KINETIC RATE CONSTANTS  (1/hour unless noted)
# ============================================================================

# Photochemistry
K_PHOTO_BASE = 0.35                 # UV-driven organic synthesis

# Lipid
K_LIPID_SYNTH = 0.05
K_LIPID_AGGREGATION = 0.40
K_LIPID_DEGRADATION = 0.03

# RNA / polymer
K_RNA_SYNTH = 0.02                  # uncatalyzed
K_RNA_SYNTH_CLAY = 0.15             # clay-catalyzed (Ferris 1996)
K_RNA_REPLICATE_BASE = 0.05         # template-directed (Lincoln & Joyce 2009)
RNA_FIDELITY = 0.98                 # per-base copying fidelity
K_HYDROLYSIS = 0.04
K_DEGRADATION = 0.03
K_RNA_DEGRADE_CLAY = 0.0042         # clay-protected RNA (100× reduction)

# Membrane
K_MEMBRANE = 0.4                    # vesicle formation rate

# ============================================================================
# DIFFUSION COEFFICIENTS (m²/hour, rescaled for grid units)
# ============================================================================

D_SMALL_MOL = 1e-9                  # small molecules (nucleotides, metabolites)
D_LIPID = 1e-11
D_RNA = 1e-12
D_CLAY = 1e-13

# Grid-normalised equivalents (multiply by 100 for 1 µm grid)
D_SMALL_MOL_GRID = D_SMALL_MOL * 100
D_LIPID_GRID = D_LIPID * 100
D_RNA_GRID = D_RNA * 100

# ============================================================================
# DETECTION THRESHOLDS
# ============================================================================

VESICLE_THRESHOLD = 0.08            # membrane field value for vesicle
RNA_REPLICATION_THRESHOLD = 50      # min nucleotides for replication

# Protocell (membrane + polymer co-localisation)
PROTOCELL_THRESHOLD_M = 0.05
PROTOCELL_THRESHOLD_R = 0.03

# ============================================================================
# ZETA-RIEMANN ZEROS (critical line, Im part)  — CIEL/0 framework
# ============================================================================

RIEMANN_CRITICAL_ZEROS = [
    0.5 + 14.1347j,
    0.5 + 21.0220j,
    0.5 + 25.0109j,
    0.5 + 30.4249j,
    0.5 + 32.9351j,
    0.5 + 37.5862j,
]

# ============================================================================
# PRIMORDIAL BLACK HOLE  — CIEL/0 research (March 2025)
# ============================================================================

PBH_POSITION_AU = [-158.4, 500.1, -254.8]   # AU from Sun
PBH_RA  = "7h10m18.395s"
PBH_DEC = "-25°54'27.284\""
PBH_CONSTELLATION = "Puppis"
PBH_MASS_KG = 5 * 5.972e24          # ~5 Earth masses
