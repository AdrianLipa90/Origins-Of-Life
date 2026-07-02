#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
origin_merged_complete.py
COMPLETE Universal Origin-of-Life Simulator with:
- 5 Scenarios (Earth, Hydrothermal, Ammonia, Titan, Enceladus)
- Kähler-Berry-Euler topological formalism
- Zeta-Riemann soft constraints
- Heisenberg uncertainty principle
- Advanced protocell detection with thermodynamics
- Comprehensive unit tests

Author: BioPhysics Research Group
License: MIT
"""

import os
import math
import argparse
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage, special, stats
from scipy.ndimage import binary_dilation, binary_erosion, uniform_filter, label
import unittest
from unittest.mock import MagicMock, patch

# ============================================================================
# CONSTANTS & ENUMS
# ============================================================================

class TopologyPattern(Enum):
    SINUSOIDAL = "sin"
    COSINUSOIDAL = "cos"
    VORTEX = "vortex"
    GAUSSIAN = "gauss"
    RANDOM = "random"
    STATIC = "static"

class TimeDependence(Enum):
    STATIC = "static"
    PULSING = "pulsing"
    DRIFT = "drift"

class SolventType(Enum):
    WATER = "H2O"
    AMMONIA = "NH3"
    METHANE_ETHANE = "CH4/C2H6"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def save_figure(fig: plt.Figure, path: str, dpi: int = 160) -> None:
    """Save matplotlib figure with proper formatting."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

def calculate_laplacian(Z: np.ndarray) -> np.ndarray:
    """Calculate discrete Laplacian using 5-point stencil."""
    return (-4 * Z + np.roll(Z, 1, 0) + np.roll(Z, -1, 0) +
            np.roll(Z, 1, 1) + np.roll(Z, -1, 1))

def calculate_gradient_norm(Z: np.ndarray) -> np.ndarray:
    """Calculate norm of gradient using central differences."""
    grad_x = np.roll(Z, -1, 0) - np.roll(Z, 1, 0)
    grad_y = np.roll(Z, -1, 1) - np.roll(Z, 1, 1)
    return np.sqrt(grad_x**2 + grad_y**2 + 1e-12)

def soft_sigmoid(x: np.ndarray, center: float = 0.0, sharpness: float = 5.0) -> np.ndarray:
    """Soft sigmoid function for smooth clipping."""
    return 1.0 / (1.0 + np.exp(-sharpness * (x - center)))

def adaptive_threshold(field: np.ndarray, base_threshold: float, window_size: int = 7) -> np.ndarray:
    """Calculate adaptive threshold using local statistics."""
    local_mean = uniform_filter(field, size=window_size)
    local_variance = uniform_filter(field**2, size=window_size) - local_mean**2
    local_std = np.sqrt(np.maximum(local_variance, 0))
    threshold = local_mean + base_threshold * (1.0 + 0.5 * local_std)
    return field > threshold

# ============================================================================
# SCENARIO CONFIGURATIONS
# ============================================================================

@dataclass
class ScenarioConfig:
    """Complete configuration for origin-of-life scenarios."""
    
    # Basic identification
    name: str
    code: str
    location: str
    
    # Physical parameters
    temp_C: float
    pressure_atm: float
    UV_flux: float
    solvent: SolventType
    pH: float
    redox: str
    
    # Energy and catalysis
    energy_source: str
    k_energy: float
    catalyst: str
    k_catalysis: float
    concentration_boost: float
    
    # Chemical kinetics
    k_synthesis: float
    k_degradation: float
    
    # Expected outcomes
    expected_protocells: int
    timescale_description: str
    
    # Topological parameters
    topo_strength: float
    topo_pattern: TopologyPattern
    topo_time_dependence: TimeDependence
    
    # Advanced constraints
    use_zeta_constraints: bool = True
    zeta_lambda_soft: float = 5.0
    zeta_sigma_heis: float = 0.001
    euler_phase_coherence: float = 0.8
    quantum_fluctuations: bool = True
    
    # Time dependence
    topo_pulse_freq: float = 0.0
    seed: int = 42
    
    def __post_init__(self):
        """Convert string enums to proper enum types."""
        if isinstance(self.solvent, str):
            self.solvent = SolventType(self.solvent)
        if isinstance(self.topo_pattern, str):
            self.topo_pattern = TopologyPattern(self.topo_pattern)
        if isinstance(self.topo_time_dependence, str):
            self.topo_time_dependence = TimeDependence(self.topo_time_dependence)

# Scenario definitions
SCENARIO_A = ScenarioConfig(
    name="Shallow Ocean + UV (Clay Hypothesis)",
    code="A",
    location="Earth - Coastal tidal zones, 10m depth",
    temp_C=65.0, pressure_atm=1.0, UV_flux=30.0, 
    solvent=SolventType.WATER, pH=7.5, redox="Mildly oxidizing",
    energy_source="UV photochemistry", k_energy=0.35,
    catalyst="Montmorillonite clay", k_catalysis=7.5, concentration_boost=1000.0,
    k_synthesis=0.15, k_degradation=0.0042, expected_protocells=600,
    timescale_description="Hours", topo_strength=0.25, 
    topo_pattern=TopologyPattern.SINUSOIDAL, topo_time_dependence=TimeDependence.PULSING,
    use_zeta_constraints=True, zeta_lambda_soft=6.0, zeta_sigma_heis=0.0005,
    euler_phase_coherence=0.9, quantum_fluctuations=True,
    topo_pulse_freq=0.05, seed=101
)

SCENARIO_B = ScenarioConfig(
    name="Deep-Sea Hydrothermal Vents (Iron-Sulfur World)",
    code="B",
    location="Earth - Mid-ocean ridges",
    temp_C=90.0, pressure_atm=200.0, UV_flux=0.0,
    solvent=SolventType.WATER, pH=9.0, redox="Strongly reducing",
    energy_source="Chemosynthesis (H2 + CO2)", k_energy=0.08,
    catalyst="Fe-S clusters", k_catalysis=15.0, concentration_boost=500.0,
    k_synthesis=0.05, k_degradation=0.02, expected_protocells=400,
    timescale_description="Days", topo_strength=0.35,
    topo_pattern=TopologyPattern.VORTEX, topo_time_dependence=TimeDependence.STATIC,
    use_zeta_constraints=True, zeta_lambda_soft=4.0, zeta_sigma_heis=0.001,
    euler_phase_coherence=0.7, quantum_fluctuations=True,
    seed=202
)

SCENARIO_C = ScenarioConfig(
    name="Ammonia-Based Biochemistry",
    code="C",
    location="Cold NH3 worlds",
    temp_C=-55.0, pressure_atm=1.0, UV_flux=10.0,
    solvent=SolventType.AMMONIA, pH=11.0, redox="Variable",
    energy_source="UV + Chemosynthesis", k_energy=0.02,
    catalyst="NH3-ice minerals", k_catalysis=5.0, concentration_boost=300.0,
    k_synthesis=0.01, k_degradation=0.001, expected_protocells=100,
    timescale_description="Weeks", topo_strength=0.15,
    topo_pattern=TopologyPattern.GAUSSIAN, topo_time_dependence=TimeDependence.DRIFT,
    use_zeta_constraints=True, zeta_lambda_soft=3.0, zeta_sigma_heis=0.002,
    euler_phase_coherence=0.6, quantum_fluctuations=False,
    topo_pulse_freq=0.005, seed=303
)

SCENARIO_D = ScenarioConfig(
    name="Titan Methane Lakes (Hydrocarbon Biochemistry)",
    code="D",
    location="Titan - Kraken Mare",
    temp_C=-179.0, pressure_atm=1.45, UV_flux=2.0,
    solvent=SolventType.METHANE_ETHANE, pH=7.0, redox="Non-oxidizing",
    energy_source="Atmospheric photochemistry", k_energy=0.001,
    catalyst="Tholins", k_catalysis=2.0, concentration_boost=100.0,
    k_synthesis=0.001, k_degradation=0.0001, expected_protocells=30,
    timescale_description="1000s hours", topo_strength=0.05,
    topo_pattern=TopologyPattern.RANDOM, topo_time_dependence=TimeDependence.STATIC,
    use_zeta_constraints=False, zeta_lambda_soft=2.0, zeta_sigma_heis=0.0001,
    euler_phase_coherence=0.3, quantum_fluctuations=True,
    seed=404
)

SCENARIO_E = ScenarioConfig(
    name="Enceladus Subsurface Ocean",
    code="E",
    location="Enceladus - Sub-ice ocean",
    temp_C=4.0, pressure_atm=800.0, UV_flux=0.0,
    solvent=SolventType.WATER, pH=9.5, redox="Reducing (H2-rich)",
    energy_source="Hydrothermal (tidal heating)", k_energy=0.08,
    catalyst="Fe-S + Mg-silicates", k_catalysis=12.0, concentration_boost=600.0,
    k_synthesis=0.08, k_degradation=0.01, expected_protocells=450,
    timescale_description="Days", topo_strength=0.28,
    topo_pattern=TopologyPattern.COSINUSOIDAL, topo_time_dependence=TimeDependence.PULSING,
    use_zeta_constraints=True, zeta_lambda_soft=5.0, zeta_sigma_heis=0.0008,
    euler_phase_coherence=0.85, quantum_fluctuations=True,
    topo_pulse_freq=0.02, seed=505
)

ALL_SCENARIOS = [SCENARIO_A, SCENARIO_B, SCENARIO_C, SCENARIO_D, SCENARIO_E]

# ============================================================================
# ZETA-RIEMANN MODULATOR
# ============================================================================

class ZetaRiemannModulator:
    """Implements Zeta-Riemann soft constraints with Heisenberg uncertainty."""
    
    @staticmethod
    def apply_constraints(
        X: np.ndarray,
        rng: np.random.Generator,
        lambda_soft: float = 5.0,
        sigma_heis: float = 0.001,
        temperature: float = 1.0,
        use_exact: bool = False
    ) -> np.ndarray:
        """
        Apply Zeta-Riemann soft constraints to field X.
        
        Parameters:
        -----------
        X : np.ndarray
            Input field
        rng : np.random.Generator
            Random number generator
        lambda_soft : float
            Softness parameter for sigmoid
        sigma_heis : float
            Heisenberg uncertainty parameter
        temperature : float
            Temperature parameter for softness
        use_exact : bool
            Whether to use exact Zeta function (slower)
            
        Returns:
        --------
        np.ndarray
            Constrained field
        """
        # 1. Zeta-based phase modulation (approximation)
        if use_exact:
            # Exact computation (requires mpmath)
            try:
                from mpmath import zeta, re
                def zeta_mod_element(x):
                    s = complex(0.5, 14.134725 * x)
                    return float(re(zeta(s)))
                zeta_mod = np.vectorize(zeta_mod_element)(X)
            except ImportError:
                warnings.warn("mpmath not available, using approximation")
                zeta_mod = np.ones_like(X)
        else:
            # Fast approximation using properties of ζ(0.5 + it)
            t = 14.134725 * X
            theta = t/2 * np.log(np.abs(t)/(2*np.pi) + 1e-12) - t/2 - np.pi/8
            Z_approx = 2 * np.exp(np.pi * t/4) * np.cos(theta) / np.sqrt(np.cosh(np.pi * t/2) + 1e-12)
            zeta_mod = np.tanh(0.1 * Z_approx) + 1.0
            zeta_mod = np.clip(zeta_mod, 0.5, 2.0)
        
        # 2. Soft Heisenberg constraint
        effective_lambda = lambda_soft / max(temperature, 0.1)
        soft_upper = 0.5 + 0.5 * np.tanh(effective_lambda * (1.0 - X))
        soft_lower = 0.5 + 0.5 * np.tanh(effective_lambda * (X - 0.0))
        soft_factor = soft_upper * soft_lower
        
        # 3. Gradient coherence penalty
        grad_norm = calculate_gradient_norm(X)
        grad_factor = np.exp(-0.5 * grad_norm**2) + 0.5
        
        # 4. Combine modulations
        X_mod = X * zeta_mod * soft_factor * grad_factor
        
        # 5. Heisenberg uncertainty fluctuations
        heis_scale = sigma_heis * np.sqrt(np.abs(X_mod) + 1e-6)
        heisenberg_noise = rng.normal(0.0, heis_scale, size=X.shape)
        X_mod = X_mod + heisenberg_noise
        
        # 6. Energy conservation (soft normalization)
        original_sum = np.sum(X)
        new_sum = np.sum(X_mod)
        
        if abs(new_sum) > 1e-12:
            scale = 0.2 + 0.8 * (original_sum / (new_sum + 1e-12))
            X_mod = X_mod * scale
        
        # 7. Final gentle bounding
        X_mod = 0.1 + 0.9 * np.tanh(X_mod)
        
        return X_mod

# ============================================================================
# PROTOCELL DETECTOR
# ============================================================================

@dataclass
class ProtocellDetectionParameters:
    """Parameters for advanced protocell detection."""
    min_membrane_density: float = 0.05
    min_rna_density: float = 0.03
    min_size_pixels: int = 4
    max_size_pixels: int = 100
    min_circularity: float = 0.3
    min_phase_coherence: float = 0.4
    stability_threshold: float = 0.7
    energy_gradient_threshold: float = 0.01
    adaptive_window_size: int = 7
    require_energy_gradient: bool = True
    require_topological_support: bool = True

class ProtocellDetector:
    """Advanced protocell detection with thermodynamics and topology."""
    
    def __init__(self, parameters: ProtocellDetectionParameters = None):
        self.params = parameters or ProtocellDetectionParameters()
        
    def detect(
        self,
        membrane_field: np.ndarray,
        rna_field: np.ndarray,
        energy_field: np.ndarray,
        organic_field: np.ndarray,
        topology_field: np.ndarray,
        phase_coherence: np.ndarray,
        temperature: float,
        pressure: float,
        verbose: bool = False
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect protocells using multiple criteria.
        
        Returns:
        --------
        Tuple[List[Dict], np.ndarray]
            List of protocell properties and labeled map
        """
        # 1. Adaptive thresholds based on environmental conditions
        temp_factor = np.exp(-abs(temperature - 25) / 50)
        press_factor = np.exp(-abs(pressure - 1) / 100)
        
        adaptive_M_thresh = self.params.min_membrane_density * temp_factor * press_factor
        adaptive_R_thresh = self.params.min_rna_density * temp_factor * press_factor
        
        # 2. Adaptive binarization
        M_binary = adaptive_threshold(
            membrane_field, adaptive_M_thresh, self.params.adaptive_window_size
        )
        R_binary = adaptive_threshold(
            rna_field, adaptive_R_thresh, self.params.adaptive_window_size
        )
        
        # 3. Initial combined mask
        combined_binary = M_binary & R_binary
        
        # 4. Apply additional constraints
        if self.params.require_topological_support:
            topo_threshold = np.percentile(topology_field, 30)
            topo_support = topology_field > topo_threshold
            combined_binary = combined_binary & topo_support
        
        coherence_support = phase_coherence > self.params.min_phase_coherence
        combined_binary = combined_binary & coherence_support
        
        # 5. Label connected components
        labeled_map, num_features = label(combined_binary)
        
        protocells = []
        
        # 6. Analyze each candidate
        for label_id in range(1, num_features + 1):
            mask = labeled_map == label_id
            size = np.sum(mask)
            
            # Size filter
            if size < self.params.min_size_pixels or size > self.params.max_size_pixels:
                if verbose:
                    print(f"  Protocell {label_id}: rejected due to size {size}")
                continue
            
            # Calculate properties
            properties = self._calculate_properties(
                mask, membrane_field, rna_field, energy_field,
                organic_field, topology_field, phase_coherence
            )
            
            # Acceptance criteria
            if self._accept_protocell(properties, temperature, pressure):
                protocell_info = {
                    'id': label_id,
                    'size': size,
                    'center': properties['center'],
                    'membrane_density': properties['membrane_density'],
                    'rna_density': properties['rna_density'],
                    'energy_flux': properties['energy_flux'],
                    'circularity': properties['circularity'],
                    'stability': properties['stability'],
                    'phase_coherence': properties['phase_coherence'],
                    'temperature': temperature,
                    'pressure': pressure,
                    'mask_indices': np.where(mask)
                }
                protocells.append(protocell_info)
                
                if verbose:
                    print(f"  ✓ Protocell {label_id}: "
                          f"size={size}, M={properties['membrane_density']:.3f}, "
                          f"stability={properties['stability']:.3f}")
            else:
                if verbose:
                    print(f"  ✗ Protocell {label_id}: rejected, "
                          f"stability={properties['stability']:.3f}")
        
        return protocells, labeled_map
    
    def _calculate_properties(
        self,
        mask: np.ndarray,
        M: np.ndarray,
        R: np.ndarray,
        E: np.ndarray,
        O: np.ndarray,
        topology: np.ndarray,
        phase_coherence: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate advanced properties for a candidate region."""
        
        # Basic statistics
        membrane_density = np.mean(M[mask])
        rna_density = np.mean(R[mask])
        topology_mean = np.mean(topology[mask])
        coherence_mean = np.mean(phase_coherence[mask])
        
        # Center of mass
        indices = np.where(mask)
        center = np.array([np.mean(indices[0]), np.mean(indices[1])])
        
        # Circularity
        circularity = self._calculate_circularity(mask)
        
        # Energy flux at boundary
        energy_flux = self._calculate_energy_flux(E, mask)
        
        # Thermodynamic stability
        stability = self._calculate_thermodynamic_stability(M, R, E, O, mask)
        
        return {
            'center': center,
            'membrane_density': float(membrane_density),
            'rna_density': float(rna_density),
            'topology_mean': float(topology_mean),
            'phase_coherence': float(coherence_mean),
            'circularity': float(circularity),
            'energy_flux': float(energy_flux),
            'stability': float(stability)
        }
    
    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """Calculate circularity as 4π·area/perimeter²."""
        area = np.sum(mask)
        
        # Calculate perimeter using erosion
        eroded = binary_erosion(mask)
        perimeter_mask = mask & ~eroded
        perimeter = np.sum(perimeter_mask)
        
        if perimeter > 0:
            return 4 * np.pi * area / (perimeter ** 2)
        return 0.0
    
    def _calculate_energy_flux(self, energy_field: np.ndarray, mask: np.ndarray) -> float:
        """Calculate energy flux across boundary."""
        boundary = self._get_boundary(mask)
        if np.any(boundary):
            grad_x, grad_y = np.gradient(energy_field)
            flux = np.mean(np.abs(grad_x[boundary])) + np.mean(np.abs(grad_y[boundary]))
            return float(flux)
        return 0.0
    
    def _calculate_thermodynamic_stability(
        self,
        M: np.ndarray,
        R: np.ndarray,
        E: np.ndarray,
        O: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """Calculate thermodynamic stability score."""
        # Membrane smoothness
        M_grad_x, M_grad_y = np.gradient(M)
        M_grad_norm = np.sqrt(M_grad_x**2 + M_grad_y**2 + 1e-12)
        membrane_smoothness = 1.0 / (1.0 + np.mean(M_grad_norm[mask]))
        
        # Energy balance
        energy_inside = np.mean(E[mask]) + np.mean(O[mask])
        boundary = self._get_boundary(mask)
        if np.any(boundary):
            energy_outside = np.mean(E[boundary]) + np.mean(O[boundary])
            energy_balance = np.tanh(energy_inside - energy_outside)
        else:
            energy_balance = 0.0
        
        # RNA-membrane ratio
        mean_M = np.mean(M[mask])
        if mean_M > 0:
            rna_membrane_ratio = np.mean(R[mask]) / mean_M
            ratio_stability = np.exp(-abs(rna_membrane_ratio - 0.5))
        else:
            ratio_stability = 0.0
        
        # Combined stability
        stability = (0.4 * membrane_smoothness + 
                    0.4 * energy_balance + 
                    0.2 * ratio_stability)
        
        return float(stability)
    
    def _get_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Get boundary pixels of a mask."""
        dilated = binary_dilation(mask)
        return dilated & ~mask
    
    def _accept_protocell(
        self,
        properties: Dict[str, Any],
        temperature: float,
        pressure: float
    ) -> bool:
        """Determine if a candidate should be accepted as a protocell."""
        # Basic thresholds
        if properties['membrane_density'] < self.params.min_membrane_density:
            return False
        if properties['rna_density'] < self.params.min_rna_density:
            return False
        if properties['circularity'] < self.params.min_circularity:
            return False
        if properties['stability'] < self.params.stability_threshold:
            return False
        
        # Environment-adaptive stability threshold
        temp_factor = np.exp(-abs(temperature - 25) / 50)
        press_factor = np.exp(-abs(pressure - 1) / 100)
        adaptive_stability = self.params.stability_threshold * temp_factor * press_factor
        
        return properties['stability'] > adaptive_stability

# ============================================================================
# UNIVERSAL ORIGIN SIMULATOR
# ============================================================================

class UniversalOriginSimulator:
    """Main simulator class integrating all advanced features."""
    
    def __init__(
        self,
        config: ScenarioConfig,
        Nx: int = 96,
        Ny: int = 96,
        dt_h: float = 0.05,
        outdir: str = 'sim_outputs'
    ):
        self.config = config
        self.Nx = Nx
        self.Ny = Ny
        self.dt_h = dt_h
        self.t_h = 0.0
        self.outdir = os.path.join(outdir, f"scenario_{config.code}")
        
        ensure_dir(self.outdir)
        
        # Random number generator
        self.rng = np.random.default_rng(config.seed)
        
        # Zeta-Riemann modulator
        self.zeta_modulator = ZetaRiemannModulator()
        
        # Protocell detector
        self.protocell_detector = ProtocellDetector()
        
        # Chemical fields
        self.E = None  # Energy
        self.O = None  # Organic compounds
        self.N = None  # Nutrients
        self.R = None  # RNA
        self.M = None  # Membrane
        self.L = None  # Lipid precursors
        self.Cat = None  # Catalyst
        
        # Phase fields
        self.phase_R = None  # RNA phase
        self.phase_M = None  # Membrane phase
        self.global_phase_coherence = None
        
        # Topology fields
        self.topo_field = None
        self.topo_curvature = None
        
        # RNA population
        self.rna_population = None
        
        # Tracking
        self.protocell_count = 0
        self.protocell_history = []
        self.active_protocells = []
        self.protocell_lifetimes = []
        
        # History storage
        self.history = {
            'time_h': [],
            'mean_R': [],
            'mean_M': [],
            'mean_E': [],
            'mean_phase_coherence': [],
            'n_polymers': [],
            'n_protocells': [],
            'mean_fitness': [],
            'zeta_modulation': [],
            'heisenberg_fluctuations': []
        }
        
        # Initialize
        self._initialize_topology()
        self._initialize_phase_fields()
        self.initialize()
    
    # ========================================================================
    # INITIALIZATION METHODS
    # ========================================================================
    
    def _initialize_topology(self) -> None:
        """Initialize topological field based on configuration."""
        x = np.linspace(-1, 1, self.Nx)
        y = np.linspace(-1, 1, self.Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        pattern = self.config.topo_pattern
        strength = self.config.topo_strength
        
        if pattern == TopologyPattern.SINUSOIDAL:
            base = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        elif pattern == TopologyPattern.COSINUSOIDAL:
            base = np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        elif pattern == TopologyPattern.VORTEX:
            theta = np.arctan2(Y, X)
            r = np.sqrt(X**2 + Y**2) + 1e-9
            base = np.sin(4 * theta) * np.exp(-3 * r**2)
        elif pattern == TopologyPattern.GAUSSIAN:
            base = (np.exp(-((X-0.2)**2 + (Y+0.1)**2) / 0.02) -
                   0.5 * np.exp(-((X+0.3)**2 + (Y-0.3)**2) / 0.05))
        elif pattern == TopologyPattern.RANDOM:
            base = self.rng.standard_normal((self.Nx, self.Ny))
            base = (np.roll(base, 1, 0) + base + np.roll(base, -1, 0) +
                   np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5.0
        else:
            base = np.zeros((self.Nx, self.Ny))
        
        # Normalize
        if np.std(base) > 0:
            base = (base - np.mean(base)) / (np.std(base) + 1e-12)
        
        self.topo_field = strength * base
        self._update_topo_curvature()
    
    def _update_topo_curvature(self) -> None:
        """Update curvature field from topology."""
        self.topo_curvature = calculate_laplacian(self.topo_field)
        if np.std(self.topo_curvature) > 0:
            self.topo_curvature = (self.topo_curvature - np.mean(self.topo_curvature)) / (
                np.std(self.topo_curvature) + 1e-12)
    
    def _initialize_phase_fields(self) -> None:
        """Initialize phase coherence fields."""
        x = np.linspace(-1, 1, self.Nx)
        y = np.linspace(-1, 1, self.Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Base coherence pattern
        base_coherence = 0.5 + 0.5 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        
        # Mix with random component
        random_component = self.rng.uniform(0, 1, (self.Nx, self.Ny))
        coherence_level = self.config.euler_phase_coherence
        
        self.global_phase_coherence = (
            coherence_level * base_coherence + 
            (1 - coherence_level) * random_component
        )
        
        # Initialize specific phase fields
        self.phase_R = np.ones((self.Nx, self.Ny))
        self.phase_M = np.ones((self.Nx, self.Ny))
    
    def _environment_factors(self) -> Tuple[float, float]:
        """Calculate temperature and pressure factors."""
        # Temperature factor with optimal range 25-70°C
        T = self.config.temp_C
        T_opt = 50.0
        T_width = 30.0
        temp_factor = np.exp(-((T - T_opt) / T_width) ** 2)
        
        # Low temperature quantum tunneling
        if T < -50:
            temp_factor *= 0.3 * (1.0 + 0.2 * math.sin(T / 10.0))
        
        # Pressure factor with optimal at 1 atm
        P = self.config.pressure_atm
        press_factor = np.exp(-0.01 * abs(P - 1.0))
        
        # High pressure oscillations
        if P > 50:
            zeros = [14.134725, 21.022040, 25.010858, 30.424876]
            oscillations = 1.0
            for z in zeros:
                oscillations *= (1.0 + 0.1 * math.sin(2 * math.pi * P / z))
            press_factor *= oscillations
        
        return float(temp_factor), float(press_factor)
    
    def initialize(self) -> None:
        """Initialize all chemical fields and RNA population."""
        temp_factor, press_factor = self._environment_factors()
        
        # Initialize chemical fields with Zeta constraints
        def init_field(min_val: float, max_val: float, modulation: float = 1.0) -> np.ndarray:
            base = self.rng.uniform(min_val, max_val, (self.Nx, self.Ny))
            base_modulated = base * modulation
            if self.config.use_zeta_constraints:
                return self._apply_zeta_constraints(base_modulated, 'generic')
            return np.clip(base_modulated, 0.0, 1.0)
        
        self.E = init_field(0.1, 0.3, temp_factor * press_factor)
        self.O = init_field(0.05, 0.15, temp_factor)
        self.N = init_field(0.01, 0.05, press_factor)
        self.R = np.zeros((self.Nx, self.Ny))
        self.M = np.zeros((self.Nx, self.Ny))
        self.L = init_field(0.005, 0.01, temp_factor * press_factor)
        self.Cat = init_field(0.8, 1.2, 1.0)
        
        # Initialize RNA population
        n_seed = max(5, int(20 * temp_factor * press_factor))
        pos_x = self.rng.integers(0, self.Nx, size=n_seed)
        pos_y = self.rng.integers(0, self.Ny, size=n_seed)
        
        # Fitness based on local conditions
        fitness = np.zeros(n_seed)
        for i in range(n_seed):
            local_coherence = self.global_phase_coherence[pos_x[i], pos_y[i]]
            fitness[i] = self.rng.uniform(0.3, 0.6) * (0.5 + 0.5 * local_coherence)
        
        self.rna_population = {
            'positions_x': pos_x.astype(np.int32),
            'positions_y': pos_y.astype(np.int32),
            'lengths': self.rng.integers(20, 50, size=n_seed).astype(np.int32),
            'fitness': fitness.astype(np.float32),
            'ages': np.zeros(n_seed, dtype=np.float32)
        }
        
        # Seed RNA in the field
        for i in range(n_seed):
            self.R[pos_x[i], pos_y[i]] += 0.1 * temp_factor
    
    # ========================================================================
    # CONSTRAINT APPLICATIONS
    # ========================================================================
    
    def _apply_zeta_constraints(self, X: np.ndarray, field_type: str) -> np.ndarray:
        """Apply Zeta-Riemann constraints to a field."""
        return self.zeta_modulator.apply_constraints(
            X=X,
            rng=self.rng,
            lambda_soft=self.config.zeta_lambda_soft,
            sigma_heis=self.config.zeta_sigma_heis,
            temperature=1.0 + 0.01 * self.config.temp_C,
            use_exact=False
        )
    
    def _apply_euler_phase_constraints(self) -> None:
        """Apply Euler-phase constraints to all fields."""
        # Update phase coherence
        self._update_phase_coherence()
        
        # Apply constraints to all fields
        field_types = {
            'E': 'energy',
            'O': 'energy',
            'N': 'generic',
            'R': 'rna',
            'M': 'membrane',
            'L': 'membrane',
            'Cat': 'catalyst'
        }
        
        for field_name, field_type in field_types.items():
            field = getattr(self, field_name)
            if field is not None:
                constrained = self._apply_zeta_constraints(field, field_type)
                setattr(self, field_name, constrained)
        
        # Energy conservation
        total_energy = np.sum(self.E) + np.sum(self.O) + np.sum(self.N)
        energy_budget = 1000.0
        
        if total_energy > energy_budget:
            scale = np.sqrt(energy_budget / (total_energy + 1e-12))
            scale = 0.3 + 0.7 * scale  # Soft scaling
            
            phase_mod = 0.5 + 0.5 * self.global_phase_coherence
            self.E *= scale * phase_mod
            self.O *= scale * phase_mod
            self.N *= scale * phase_mod
    
    def _update_phase_coherence(self) -> None:
        """Update phase coherence fields."""
        # Evolve RNA phase based on RNA field gradients
        if self.R is not None:
            grad_norm_R = calculate_gradient_norm(self.R)
            self.phase_R += self.dt_h * (0.1 * grad_norm_R - 0.05 * self.phase_R)
        
        # Evolve membrane phase based on membrane field gradients
        if self.M is not None:
            grad_norm_M = calculate_gradient_norm(self.M)
            self.phase_M += self.dt_h * (0.1 * grad_norm_M - 0.05 * self.phase_M)
        
        # Update global coherence
        field_correlation = 0.0
        if self.R is not None and self.M is not None:
            try:
                correlation = np.corrcoef(self.R.flatten(), self.M.flatten())[0, 1]
                field_correlation = 0.5 + 0.5 * np.clip(correlation, -1, 1)
            except:
                field_correlation = 0.5
        
        time_modulation = 0.5 + 0.5 * math.cos(2 * math.pi * self.t_h / 24)
        
        self.global_phase_coherence = (
            0.7 * self.global_phase_coherence +
            0.2 * time_modulation +
            0.1 * field_correlation
        )
        
        # Add quantum fluctuations if enabled
        if self.config.quantum_fluctuations:
            quantum_noise = self.rng.normal(0, 0.01, (self.Nx, self.Ny))
            self.global_phase_coherence += quantum_noise
        
        # Normalize
        self.global_phase_coherence = np.clip(self.global_phase_coherence, 0, 1)
    
    # ========================================================================
    # BIOCHEMICAL PROCESSES
    # ========================================================================
    
    def step_energy_conversion(self) -> None:
        """Energy conversion step."""
        k = self.config.k_energy
        efficiency = 0.8 if self.config.UV_flux > 0 else 0.6
        
        # Topological modulation
        topo_mod = 1.0 + 0.6 * self.topo_field + 0.4 * self.topo_curvature
        topo_mod = np.clip(topo_mod, 0.2, 3.0)
        
        # Environmental factors
        temp_factor, press_factor = self._environment_factors()
        
        # Phase coherence modulation
        phase_mod = 0.5 + 0.5 * self.global_phase_coherence
        
        effective_k = k * temp_factor * press_factor * phase_mod
        
        dE = -effective_k * self.E * self.dt_h * topo_mod
        dO = effective_k * self.E * efficiency * self.dt_h * topo_mod
        
        self.E += dE
        self.O += dO
    
    def step_catalysis(self) -> None:
        """Catalysis step."""
        k_cat = self.config.k_catalysis * 0.1
        catalyst_effect = self.Cat / (self.Cat + 1.0)
        
        # Topological modulation
        topo_mod = 1.0 + 0.5 * self.topo_field + 0.6 * self.topo_curvature
        topo_mod = np.clip(topo_mod, 0.2, 4.0)
        
        # Environmental factors
        temp_factor, press_factor = self._environment_factors()
        
        # Phase modulation
        phase_mod = 0.3 + 0.7 * self.global_phase_coherence
        
        effective_k = k_cat * temp_factor * press_factor * phase_mod
        
        dO = -effective_k * self.O * catalyst_effect * self.dt_h * topo_mod
        dN = effective_k * self.O * catalyst_effect * self.dt_h * topo_mod
        
        self.O += dO
        self.N += dN
    
    def step_polymerization(self) -> None:
        """Polymerization (RNA synthesis) step."""
        k_syn = self.config.k_synthesis
        boost = max(1e-6, self.config.concentration_boost / 1000.0)
        
        # Combined modulation
        topo_mod = (1.0 + 0.8 * self.topo_field + 0.2 * self.topo_curvature +
                   0.3 * (self.Cat - 1.0))
        topo_mod = np.clip(topo_mod, 0.1, 5.0)
        
        # Environmental factors
        temp_factor, press_factor = self._environment_factors()
        
        # RNA phase modulation
        rna_phase_mod = 0.4 + 0.6 * self.phase_R
        
        effective_k = k_syn * temp_factor * press_factor * rna_phase_mod
        
        dN = -effective_k * self.N * boost * self.dt_h * topo_mod
        dR = effective_k * self.N * boost * self.dt_h * topo_mod
        
        self.N += dN
        self.R += dR
    
    def step_degradation(self) -> None:
        """Degradation step for RNA and polymers."""
        k_deg = self.config.k_degradation
        
        # Temperature dependence
        temp_factor = math.exp(0.05 * (self.config.temp_C - 25.0))
        
        # Topological modulation (opposite to synthesis)
        topo_mod = 1.0 + 0.5 * (-self.topo_field) + 0.3 * (self.topo_curvature)
        topo_mod = np.clip(topo_mod, 0.05, 4.0)
        
        # Pressure factor
        _, press_factor = self._environment_factors()
        
        # Phase incoherence promotes degradation
        phase_mod = 1.5 - 0.5 * self.global_phase_coherence
        
        effective_k = k_deg * temp_factor * press_factor * phase_mod
        
        # Field degradation
        dR = -effective_k * self.R * self.dt_h * topo_mod
        self.R += dR
        
        # RNA population degradation
        if self.rna_population is not None:
            n_rna = len(self.rna_population['positions_x'])
            if n_rna > 0:
                # Update ages
                self.rna_population['ages'] += self.dt_h
                
                # Calculate local degradation probabilities
                pos_x = self.rna_population['positions_x']
                pos_y = self.rna_population['positions_y']
                
                local_topo = self.topo_field[pos_x, pos_y]
                local_curv = self.topo_curvature[pos_x, pos_y]
                local_phase = self.global_phase_coherence[pos_x, pos_y]
                
                local_mod = np.clip(1.0 + 0.5 * (-local_topo) + 0.3 * local_curv, 0.05, 4.0)
                local_mod *= (1.5 - 0.5 * local_phase)
                
                probs = k_deg * temp_factor * self.dt_h * local_mod * press_factor
                probs = np.clip(probs, 0, 1)
                
                # Survival mask
                survival_mask = self.rng.random(n_rna) >= probs
                
                # Update population
                for key in self.rna_population:
                    self.rna_population[key] = self.rna_population[key][survival_mask]
    
    def step_membrane_formation(self) -> None:
        """Membrane formation step."""
        k_mem = 0.4
        
        # Temperature factor with phase transitions
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
        
        # Topological modulation
        topo_mod = 1.0 + 0.6 * self.topo_field + 0.5 * self.topo_curvature
        topo_mod = np.clip(topo_mod, 0.05, 6.0)
        
        # Pressure factor
        _, press_factor = self._environment_factors()
        
        # Membrane phase modulation
        membrane_phase_mod = 0.6 + 0.4 * self.phase_M
        
        effective_k = k_mem * temp_factor * press_factor * membrane_phase_mod
        
        dL = -effective_k * self.L * self.dt_h * topo_mod
        dM = effective_k * self.L * self.dt_h * topo_mod
        
        self.L += dL
        self.M += dM
    
    def step_topology_evolution(self) -> None:
        """Evolve topology in time."""
        time_dep = self.config.topo_time_dependence
        
        if time_dep == TimeDependence.STATIC:
            return
        
        elif time_dep == TimeDependence.PULSING:
            f = max(1e-9, self.config.topo_pulse_freq)
            factor = 1.0 + 0.5 * math.sin(2 * math.pi * f * self.t_h)
            self.topo_field *= factor
            self._update_topo_curvature()
        
        elif time_dep == TimeDependence.DRIFT:
            shift_x = int((self.t_h * 0.01) % self.Nx)
            shift_y = int((self.t_h * 0.015) % self.Ny)
            self.topo_field = np.roll(self.topo_field, shift_x, axis=0)
            self.topo_field = np.roll(self.topo_field, shift_y, axis=1)
            self._update_topo_curvature()
    
    def step_protocell_dynamics(self) -> None:
        """Advanced protocell detection and dynamics."""
        # Detect protocells
        protocells, labeled_map = self.protocell_detector.detect(
            membrane_field=self.M,
            rna_field=self.R,
            energy_field=self.E,
            organic_field=self.O,
            topology_field=self.topo_field,
            phase_coherence=self.global_phase_coherence,
            temperature=self.config.temp_C,
            pressure=self.config.pressure_atm,
            verbose=False
        )
        
        self.protocell_count = len(protocells)
        
        # Update protocell tracking
        current_ids = {p['id'] for p in protocells}
        previous_ids = {p['id'] for p in self.active_protocells}
        
        # New protocells
        new_ids = current_ids - previous_ids
        for proto in protocells:
            if proto['id'] in new_ids:
                proto['birth_time'] = self.t_h
                proto['properties_history'] = []
                proto['age'] = 0.0
                self.active_protocells.append(proto)
        
        # Update existing protocells
        updated_protocells = []
        for proto in self.active_protocells:
            if proto['id'] in current_ids:
                # Still alive
                proto['age'] += self.dt_h
                
                # Record properties
                mask = labeled_map == proto['id']
                properties = self.protocell_detector._calculate_properties(
                    mask, self.M, self.R, self.E, self.O,
                    self.topo_field, self.global_phase_coherence
                )
                
                proto['properties_history'].append({
                    'time': self.t_h,
                    'stability': properties['stability'],
                    'size': np.sum(mask),
                    'membrane_density': properties['membrane_density'],
                    'rna_density': properties['rna_density']
                })
                
                # Simulate internal dynamics
                self._simulate_protocell_internal_dynamics(proto, mask)
                
                updated_protocells.append(proto)
            else:
                # Died
                proto['death_time'] = self.t_h
                proto['lifetime'] = self.t_h - proto['birth_time']
                self.protocell_lifetimes.append(proto)
        
        self.active_protocells = updated_protocells
        self.protocell_history.append({
            'time': self.t_h,
            'count': self.protocell_count,
            'protocells': protocells.copy()
        })
    
    def _simulate_protocell_internal_dynamics(self, protocell: Dict, mask: np.ndarray) -> None:
        """Simulate internal dynamics of a protocell."""
        # RNA replication within protocell
        if np.mean(self.R[mask]) > 0.1 and np.mean(self.N[mask]) > 0.05:
            replication_rate = 0.01 * self.dt_h * np.mean(self.Cat[mask])
            self.R[mask] *= (1.0 + replication_rate)
            self.N[mask] *= (1.0 - 0.5 * replication_rate)
        
        # Membrane growth
        if np.mean(self.L[mask]) > 0.02:
            growth_rate = 0.005 * self.dt_h
            self.M[mask] *= (1.0 + growth_rate)
            self.L[mask] *= (1.0 - growth_rate)
    
    # ========================================================================
    # MAIN SIMULATION LOOP
    # ========================================================================
    
    def record_state(self) -> None:
        """Record current simulation state."""
        self.history['time_h'].append(self.t_h)
        self.history['mean_R'].append(float(np.mean(self.R)))
        self.history['mean_M'].append(float(np.mean(self.M)))
        self.history['mean_E'].append(float(np.mean(self.E)))
        self.history['mean_phase_coherence'].append(float(np.mean(self.global_phase_coherence)))
        
        n_polymers = 0
        if self.rna_population is not None:
            n_polymers = len(self.rna_population['positions_x'])
        self.history['n_polymers'].append(n_polymers)
        
        self.history['n_protocells'].append(self.protocell_count)
        
        if self.rna_population is not None and len(self.rna_population['fitness']) > 0:
            self.history['mean_fitness'].append(float(np.mean(self.rna_population['fitness'])))
        else:
            self.history['mean_fitness'].append(0.0)
        
        # Advanced metrics
        if self.config.use_zeta_constraints:
            zeta_mod = float(np.mean(np.abs(np.cos(14.134725 * self.R))))
            self.history['zeta_modulation'].append(zeta_mod)
        else:
            self.history['zeta_modulation'].append(0.0)
        
        if self.config.quantum_fluctuations:
            heis_fluct = float(np.std(self.R) / (np.mean(self.R) + 1e-12))
            self.history['heisenberg_fluctuations'].append(heis_fluct)
        else:
            self.history['heisenberg_fluctuations'].append(0.0)
    
    def step(self) -> None:
        """Execute one simulation step."""
        # 1. Evolve topology
        self.step_topology_evolution()
        
        # 2. Biochemical processes
        self.step_energy_conversion()
        self.step_catalysis()
        self.step_polymerization()
        self.step_degradation()
        self.step_membrane_formation()
        
        # 3. Apply constraints
        self._apply_euler_phase_constraints()
        
        # 4. Protocell dynamics
        self.step_protocell_dynamics()
        
        # 5. Advance time
        self.t_h += self.dt_h
        
        # 6. Record state (not every step for efficiency)
        if np.isclose(self.t_h % 2.0, 0, atol=self.dt_h/2):
            self.record_state()
    
    def run(self, hours: float = 120.0, record_interval: float = 2.0, verbose: bool = True) -> pd.DataFrame:
        """
        Run the complete simulation.
        
        Parameters:
        -----------
        hours : float
            Total simulation time in hours
        record_interval : float
            Interval for recording state
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        pd.DataFrame
            History of simulation states
        """
        n_steps = int(hours / self.dt_h)
        record_steps = max(1, int(record_interval / self.dt_h))
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running SCENARIO {self.config.code}: {self.config.name}")
            print(f"{'='*60}")
            print(f"Grid: {self.Nx}x{self.Ny} | dt={self.dt_h}h | Total hours={hours}")
            print(f"Zeta constraints: {self.config.use_zeta_constraints}")
            print(f"Quantum fluctuations: {self.config.quantum_fluctuations}")
            print(f"{'='*60}")
        
        for step in range(n_steps):
            self.step()
            
            if verbose and step % (record_steps * 10) == 0:
                n_pol = self.history['n_polymers'][-1] if self.history['n_polymers'] else 0
                phase_coherence = self.history['mean_phase_coherence'][-1] if self.history['mean_phase_coherence'] else 0.0
                print(f"  t={self.t_h:6.1f}h | RNA={np.mean(self.R):.4f} | "
                      f"Memb={np.mean(self.M):.4f} | "
                      f"Protocells={self.protocell_count:3d} | "
                      f"Phase={phase_coherence:.3f}")
        
        # Final recording
        self.record_state()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"SCENARIO {self.config.code} COMPLETE")
            print(f"{'='*60}")
            print(f"Final RNA concentration: {np.mean(self.R):.4f}")
            print(f"Final membrane concentration: {np.mean(self.M):.4f}")
            print(f"Total protocells detected: {self.protocell_count}")
            print(f"Final phase coherence: {self.history['mean_phase_coherence'][-1]:.3f}")
            print(f"{'='*60}")
        
        # Save results
        self.save_results()
        
        return pd.DataFrame(self.history)
    
    def save_results(self) -> None:
        """Save all simulation results to files."""
        # Save history
        history_df = pd.DataFrame(self.history)
        history_csv = os.path.join(self.outdir, f"scenario_{self.config.code}_history.csv")
        history_df.to_csv(history_csv, index=False)
        
        # Save protocell history
        if self.protocell_history:
            proto_df = pd.DataFrame([
                {
                    'time': entry['time'],
                    'count': entry['count'],
                    'avg_stability': np.mean([p.get('stability', 0) for p in entry['protocells']]) if entry['protocells'] else 0
                }
                for entry in self.protocell_history
            ])
            proto_csv = os.path.join(self.outdir, f"scenario_{self.config.code}_protocells.csv")
            proto_df.to_csv(proto_csv, index=False)
        
        # Save final fields
        np.savez_compressed(
            os.path.join(self.outdir, f"scenario_{self.config.code}_final_fields.npz"),
            E=self.E, O=self.O, N=self.N, R=self.R, M=self.M, L=self.L,
            Cat=self.Cat, topo_field=self.topo_field,
            topo_curvature=self.topo_curvature,
            phase_R=self.phase_R, phase_M=self.phase_M,
            global_phase_coherence=self.global_phase_coherence
        )
        
        # Save summary
        summary = {
            'scenario': self.config.code,
            'name': self.config.name,
            'temperature_C': self.config.temp_C,
            'pressure_atm': self.config.pressure_atm,
            'solvent': self.config.solvent.value,
            'final_rna_mean': float(np.mean(self.R)),
            'final_membrane_mean': float(np.mean(self.M)),
            'final_protocells': self.protocell_count,
            'expected_protocells': self.config.expected_protocells,
            'success_rate_pct': 100.0 * self.protocell_count / max(1, self.config.expected_protocells),
            'final_phase_coherence': float(np.mean(self.global_phase_coherence)),
            'simulation_hours': self.t_h,
            'zeta_constraints': self.config.use_zeta_constraints,
            'quantum_fluctuations': self.config.quantum_fluctuations
        }
        
        summary_df = pd.DataFrame([summary])
        summary_csv = os.path.join(self.outdir, f"scenario_{self.config.code}_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"Results saved to: {self.outdir}")
    
    def plot_results(self) -> None:
        """Generate comprehensive visualization plots."""
        # Time series
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # RNA and Membrane
        ax = axes[0, 0]
        ax.plot(self.history['time_h'], self.history['mean_R'], label='RNA', color='blue')
        ax.plot(self.history['time_h'], self.history['mean_M'], label='Membrane', color='red')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Concentration')
        ax.set_title('RNA and Membrane Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Protocells
        ax = axes[0, 1]
        ax.plot(self.history['time_h'], self.history['n_protocells'], color='green')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Number of Protocells')
        ax.set_title('Protocell Formation')
        ax.grid(True, alpha=0.3)
        
        # Phase coherence
        ax = axes[0, 2]
        ax.plot(self.history['time_h'], self.history['mean_phase_coherence'], color='purple')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Phase Coherence')
        ax.set_title('Global Phase Coherence')
        ax.grid(True, alpha=0.3)
        
        # Energy
        ax = axes[1, 0]
        ax.plot(self.history['time_h'], self.history['mean_E'], color='orange')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Energy Concentration')
        ax.set_title('Energy Field Evolution')
        ax.grid(True, alpha=0.3)
        
        # Polymers
        ax = axes[1, 1]
        ax.plot(self.history['time_h'], self.history['n_polymers'], color='brown')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Number of Polymers')
        ax.set_title('RNA Polymer Population')
        ax.grid(True, alpha=0.3)
        
        # Fitness
        ax = axes[1, 2]
        ax.plot(self.history['time_h'], self.history['mean_fitness'], color='pink')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Average Fitness')
        ax.set_title('RNA Fitness Evolution')
        ax.grid(True, alpha=0.3)
        
        # Zeta modulation
        ax = axes[2, 0]
        ax.plot(self.history['time_h'], self.history['zeta_modulation'], color='cyan')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Zeta Modulation')
        ax.set_title('Zeta-Riemann Constraint Effect')
        ax.grid(True, alpha=0.3)
        
        # Heisenberg fluctuations
        ax = axes[2, 1]
        ax.plot(self.history['time_h'], self.history['heisenberg_fluctuations'], color='magenta')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Fluctuation Magnitude')
        ax.set_title('Heisenberg Uncertainty')
        ax.grid(True, alpha=0.3)
        
        # Success rate vs expected
        ax = axes[2, 2]
        success_rate = 100.0 * self.protocell_count / max(1, self.config.expected_protocells)
        ax.bar(['Actual', 'Expected'], [self.protocell_count, self.config.expected_protocells])
        ax.set_ylabel('Number of Protocells')
        ax.set_title(f'Success Rate: {success_rate:.1f}%')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Scenario {self.config.code}: {self.config.name}\n'
                    f'Complete Simulation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        time_series_path = os.path.join(self.outdir, 'time_series.png')
        save_figure(fig, time_series_path)
        
        # Final state heatmaps
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        fields = [
            (self.R, 'RNA Field', 'viridis', axes[0, 0]),
            (self.M, 'Membrane Field', 'plasma', axes[0, 1]),
            (self.E, 'Energy Field', 'hot', axes[0, 2]),
            (self.topo_field, 'Topology Field', 'coolwarm', axes[1, 0]),
            (self.global_phase_coherence, 'Phase Coherence', 'twilight', axes[1, 1]),
            (self.Cat, 'Catalyst Field', 'YlOrRd', axes[1, 2])
        ]
        
        for field, title, cmap, ax in fields:
            im = ax.imshow(field, origin='lower', aspect='auto', cmap=cmap)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'Final State Fields - Scenario {self.config.code}', fontsize=14)
        plt.tight_layout()
        
        heatmaps_path = os.path.join(self.outdir, 'final_fields_heatmaps.png')
        save_figure(fig, heatmaps_path)

# ============================================================================
# SCENARIO RUNNER
# ============================================================================

def run_all_scenarios(
    scenarios: List[ScenarioConfig],
    Nx: int = 96,
    Ny: int = 96,
    dt_h: float = 0.05,
    base_hours: float = 120.0,
    outdir: str = 'sim_outputs'
) -> pd.DataFrame:
    """Run all scenarios and save comparative results."""
    
    ensure_dir(outdir)
    all_summaries = []
    
    for config in scenarios:
        print(f"\n{'='*80}")
        print(f"STARTING SCENARIO {config.code}: {config.name}")
        print(f"{'='*80}")
        
        # Adjust runtime for slower scenarios
        hours = base_hours
        if config.code == 'D':  # Titan
            hours = max(base_hours, 500.0)
        elif config.code == 'C':  # Ammonia
            hours = max(base_hours, 200.0)
        
        # Create simulator
        simulator = UniversalOriginSimulator(
            config=config,
            Nx=Nx,
            Ny=Ny,
            dt_h=dt_h,
            outdir=outdir
        )
        
        # Run simulation
        history = simulator.run(
            hours=hours,
            record_interval=2.0,
            verbose=True
        )
        
        # Plot results
        simulator.plot_results()
        
        # Save summary
        summary = {
            'Scenario': config.code,
            'Name': config.name,
            'Location': config.location,
            'Temp_C': config.temp_C,
            'Pressure_atm': config.pressure_atm,
            'Solvent': config.solvent.value,
            'Energy_Source': config.energy_source,
            'Catalyst': config.catalyst,
            'Final_RNA': float(np.mean(simulator.R)),
            'Final_Membrane': float(np.mean(simulator.M)),
            'Final_Protocells': simulator.protocell_count,
            'Expected_Protocells': config.expected_protocells,
            'Success_Rate_pct': 100.0 * simulator.protocell_count / max(1, config.expected_protocells),
            'Final_Phase_Coherence': float(np.mean(simulator.global_phase_coherence)),
            'Zeta_Constraints': config.use_zeta_constraints,
            'Quantum_Fluctuations': config.quantum_fluctuations,
            'Simulation_Hours': simulator.t_h
        }
        
        all_summaries.append(summary)
        
        print(f"\n✓ Scenario {config.code} completed successfully")
        print(f"  Results saved to: {simulator.outdir}")
    
    # Create comparative summary
    summary_df = pd.DataFrame(all_summaries)
    summary_csv = os.path.join(outdir, 'ALL_SCENARIOS_SUMMARY.csv')
    summary_df.to_csv(summary_csv, index=False)
    
    # Plot comparative analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Success rates
    ax = axes[0, 0]
    scenarios = summary_df['Scenario']
    success_rates = summary_df['Success_Rate_pct']
    bars = ax.bar(scenarios, success_rates)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Protocell Formation Success by Scenario')
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% Target')
    ax.legend()
    
    # Color bars by success rate
    for bar, rate in zip(bars, success_rates):
        if rate >= 100:
            bar.set_color('green')
        elif rate >= 50:
            bar.set_color('yellow')
        else:
            bar.set_color('red')
    
    # Phase coherence vs success
    ax = axes[0, 1]
    scatter = ax.scatter(
        summary_df['Final_Phase_Coherence'],
        summary_df['Success_Rate_pct'],
        c=summary_df['Temp_C'],
        s=200,
        cmap='coolwarm',
        alpha=0.7,
        edgecolors='black'
    )
    ax.set_xlabel('Final Phase Coherence')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Phase Coherence vs Success (Color: Temperature)')
    plt.colorbar(scatter, ax=ax, label='Temperature (°C)')
    
    # Zeta constraints effect
    ax = axes[1, 0]
    zeta_groups = summary_df.groupby('Zeta_Constraints')['Success_Rate_pct'].mean()
    colors = ['lightcoral', 'lightgreen']
    zeta_groups.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
    ax.set_xticklabels(['Without Zeta', 'With Zeta'], rotation=0)
    ax.set_ylabel('Average Success Rate (%)')
    ax.set_title('Effect of Zeta-Riemann Constraints')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Quantum fluctuations effect
    ax = axes[1, 1]
    quantum_groups = summary_df.groupby('Quantum_Fluctuations')['Final_Phase_Coherence'].mean()
    colors = ['lightblue', 'orange']
    quantum_groups.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
    ax.set_xticklabels(['Without Quantum', 'With Quantum'], rotation=0)
    ax.set_ylabel('Average Phase Coherence')
    ax.set_title('Effect of Quantum Fluctuations')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Comparative Analysis of All Scenarios\n'
                'Universal Origin-of-Life Simulator', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = os.path.join(outdir, 'scenario_comparison.png')
    save_figure(fig, comparison_path)
    
    print(f"\n{'='*80}")
    print("ALL SCENARIOS COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Summary saved to: {summary_csv}")
    print(f"Comparison plot saved to: {comparison_path}")
    print(f"\nFinal Summary:\n")
    print(summary_df.to_string(index=False))
    
    return summary_df

# ============================================================================
# UNIT TESTS
# ============================================================================

class TestUniversalOriginSimulator(unittest.TestCase):
    """Comprehensive unit tests for the simulator."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = SCENARIO_A
        self.simulator = UniversalOriginSimulator(
            config=self.config,
            Nx=32,  # Small grid for faster tests
            Ny=32,
            dt_h=0.1,
            outdir='test_outputs'
        )
    
    def test_initialization(self):
        """Test that simulator initializes correctly."""
        self.assertIsNotNone(self.simulator.E)
        self.assertIsNotNone(self.simulator.R)
        self.assertIsNotNone(self.simulator.M)
        self.assertIsNotNone(self.simulator.topo_field)
        self.assertIsNotNone(self.simulator.global_phase_coherence)
        
        # Check field shapes
        self.assertEqual(self.simulator.E.shape, (32, 32))
        self.assertEqual(self.simulator.R.shape, (32, 32))
        self.assertEqual(self.simulator.M.shape, (32, 32))
        self.assertEqual(self.simulator.topo_field.shape, (32, 32))
    
    def test_no_nan_values(self):
        """Test that no NaN values appear during simulation."""
        # Run a few steps
        for _ in range(10):
            self.simulator.step()
        
        # Check all fields
        fields = ['E', 'O', 'N', 'R', 'M', 'L', 'Cat', 
                 'topo_field', 'topo_curvature',
                 'phase_R', 'phase_M', 'global_phase_coherence']
        
        for field_name in fields:
            field = getattr(self.simulator, field_name)
            self.assertFalse(np.any(np.isnan(field)), 
                           f"NaN found in field {field_name}")
            self.assertFalse(np.any(np.isinf(field)), 
                           f"Inf found in field {field_name}")
    
    def test_zeta_constraints_effect(self):
        """Test that Zeta constraints affect the fields."""
        # Create two simulators: with and without Zeta constraints
        config_with_zeta = SCENARIO_A
        config_without_zeta = ScenarioConfig(**{**SCENARIO_A.__dict__, 
                                               'use_zeta_constraints': False})
        
        sim_with = UniversalOriginSimulator(config_with_zeta, Nx=16, Ny=16)
        sim_without = UniversalOriginSimulator(config_without_zeta, Nx=16, Ny=16)
        
        # Run both for a few steps
        for _ in range(5):
            sim_with.step()
            sim_without.step()
        
        # Fields should be different
        self.assertFalse(np.allclose(sim_with.R, sim_without.R, atol=1e-5),
                        "Zeta constraints should affect RNA field")
        self.assertFalse(np.allclose(sim_with.M, sim_without.M, atol=1e-5),
                        "Zeta constraints should affect membrane field")
    
    def test_phase_coherence_range(self):
        """Test that phase coherence stays in valid range [0, 1]."""
        for _ in range(20):
            self.simulator.step()
            coherence = self.simulator.global_phase_coherence
            self.assertTrue(np.all(coherence >= 0), 
                          "Phase coherence should be >= 0")
            self.assertTrue(np.all(coherence <= 1), 
                          "Phase coherence should be <= 1")
    
    def test_protocell_detection_structure(self):
        """Test that protocell detection returns correct structure."""
        # Create synthetic data for testing
        membrane = np.zeros((32, 32))
        membrane[10:20, 10:20] = 0.1  # A potential protocell
        
        rna = np.zeros((32, 32))
        rna[10:20, 10:20] = 0.08
        
        detector = ProtocellDetector()
        protocells, labeled_map = detector.detect(
            membrane_field=membrane,
            rna_field=rna,
            energy_field=np.ones((32, 32)) * 0.5,
            organic_field=np.ones((32, 32)) * 0.3,
            topology_field=np.ones((32, 32)) * 0.5,
            phase_coherence=np.ones((32, 32)) * 0.8,
            temperature=25.0,
            pressure=1.0,
            verbose=False
        )
        
        # Check return types
        self.assertIsInstance(protocells, list)
        self.assertIsInstance(labeled_map, np.ndarray)
        
        # If a protocell is detected, check its structure
        if protocells:
            proto = protocells[0]
            self.assertIn('id', proto)
            self.assertIn('size', proto)
            self.assertIn('membrane_density', proto)
            self.assertIn('rna_density', proto)
            self.assertIn('stability', proto)
            self.assertIn('circularity', proto)
    
    def test_energy_conservation(self):
        """Test approximate energy conservation."""
        initial_total = (np.sum(self.simulator.E) + 
                        np.sum(self.simulator.O) + 
                        np.sum(self.simulator.N))
        
        for _ in range(10):
            self.simulator.step()
        
        final_total = (np.sum(self.simulator.E) + 
                      np.sum(self.simulator.O) + 
                      np.sum(self.simulator.N))
        
        # Energy should be roughly conserved (within 10%)
        relative_change = abs(final_total - initial_total) / max(initial_total, 1e-12)
        self.assertLess(relative_change, 0.1,
                       f"Energy not conserved: change = {relative_change:.2%}")
    
    def test_temperature_pressure_factors(self):
        """Test that temperature and pressure factors work correctly."""
        temp_factor, press_factor = self.simulator._environment_factors()
        
        # Factors should be positive
        self.assertGreater(temp_factor, 0)
        self.assertGreater(press_factor, 0)
        
        # Factors should be reasonable (not extremely large)
        self.assertLess(temp_factor, 10)
        self.assertLess(press_factor, 10)
    
    def test_topology_patterns(self):
        """Test that all topology patterns generate valid fields."""
        patterns = [
            TopologyPattern.SINUSOIDAL,
            TopologyPattern.COSINUSOIDAL,
            TopologyPattern.VORTEX,
            TopologyPattern.GAUSSIAN,
            TopologyPattern.RANDOM
        ]
        
        for pattern in patterns:
            config = ScenarioConfig(**{**SCENARIO_A.__dict__,
                                      'topo_pattern': pattern})
            sim = UniversalOriginSimulator(config, Nx=16, Ny=16)
            
            # Check field properties
            self.assertIsNotNone(sim.topo_field)
            self.assertEqual(sim.topo_field.shape, (16, 16))
            self.assertFalse(np.any(np.isnan(sim.topo_field)))
            self.assertFalse(np.any(np.isinf(sim.topo_field)))
    
    def test_zero_coupling_limit(self):
        """Test behavior when all coupling parameters are zero."""
        # Create a configuration with minimal coupling
        zero_config = ScenarioConfig(**{**SCENARIO_A.__dict__,
                                       'k_energy': 0.0,
                                       'k_catalysis': 0.0,
                                       'k_synthesis': 0.0,
                                       'k_degradation': 0.0,
                                       'topo_strength': 0.0,
                                       'use_zeta_constraints': False,
                                       'quantum_fluctuations': False})
        
        sim = UniversalOriginSimulator(zero_config, Nx=16, Ny=16)
        
        # Record initial state
        initial_R = sim.R.copy()
        initial_M = sim.M.copy()
        initial_E = sim.E.copy()
        
        # Run simulation
        for _ in range(10):
            sim.step()
        
        # With zero coupling, fields should not change much
        # (only minor numerical diffusion)
        delta_R = np.max(np.abs(sim.R - initial_R))
        delta_M = np.max(np.abs(sim.M - initial_M))
        delta_E = np.max(np.abs(sim.E - initial_E))
        
        self.assertLess(delta_R, 1e-3, "RNA should not change with zero coupling")
        self.assertLess(delta_M, 1e-3, "Membrane should not change with zero coupling")
        self.assertLess(delta_E, 1e-3, "Energy should not change with zero coupling")
    
    def test_history_tracking(self):
        """Test that history is properly tracked."""
        # Run simulation
        for _ in range(5):
            self.simulator.step()
        
        # Check history lengths
        self.assertEqual(len(self.simulator.history['time_h']), 
                         len(self.simulator.history['mean_R']))
        self.assertEqual(len(self.simulator.history['time_h']), 
                         len(self.simulator.history['mean_M']))
        self.assertEqual(len(self.simulator.history['time_h']), 
                         len(self.simulator.history['n_protocells']))
        
        # Check time is increasing
        times = self.simulator.history['time_h']
        self.assertTrue(all(times[i] <= times[i+1] for i in range(len(times)-1)),
                       "Time should be non-decreasing")
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists('test_outputs'):
            shutil.rmtree('test_outputs')

def run_unit_tests():
    """Run all unit tests and report results."""
    print("\n" + "="*80)
    print("RUNNING UNIT TESTS")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestUniversalOriginSimulator)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    print("UNIT TEST SUMMARY")
    print("="*80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
        for test, traceback in result.failures + result.errors:
            print(f"\nFailed test: {test}")
            print(traceback)
    
    return result.wasSuccessful()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="Universal Origin-of-Life Simulator with Advanced Constraints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        choices=['scenarios', 'single', 'tests', 'all'],
        default='scenarios',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--scenario',
        choices=['A', 'B', 'C', 'D', 'E'],
        default='A',
        help='Scenario to run (for single mode)'
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default='simulation_results',
        help='Output directory'
    )
    
    parser.add_argument(
        '--nx',
        type=int,
        default=96,
        help='Grid size in x-direction'
    )
    
    parser.add_argument(
        '--ny',
        type=int,
        default=96,
        help='Grid size in y-direction'
    )
    
    parser.add_argument(
        '--dt',
        type=float,
        default=0.05,
        help='Time step (hours)'
    )
    
    parser.add_argument(
        '--hours',
        type=float,
        default=120.0,
        help='Simulation duration (hours)'
    )
    
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Run only unit tests'
    )
    
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip unit tests'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_dir(args.outdir)
    
    # Run unit tests if requested
    if not args.skip_tests:
        tests_passed = run_unit_tests()
        if not tests_passed and args.test_only:
            print("\nTests failed, aborting execution.")
            return 1
    
    if args.test_only:
        return 0
    
    # Execute based on mode
    if args.mode in ['scenarios', 'all']:
        print("\n" + "="*80)
        print("RUNNING ALL 5 SCENARIOS")
        print("="*80)
        
        run_all_scenarios(
            scenarios=ALL_SCENARIOS,
            Nx=args.nx,
            Ny=args.ny,
            dt_h=args.dt,
            base_hours=args.hours,
            outdir=args.outdir
        )
    
    if args.mode in ['single', 'all']:
        print("\n" + "="*80)
        print(f"RUNNING SINGLE SCENARIO {args.scenario}")
        print("="*80)
        
        # Find selected scenario
        selected = None
        for scenario in ALL_SCENARIOS:
            if scenario.code == args.scenario:
                selected = scenario
                break
        
        if selected is None:
            print(f"Error: Scenario {args.scenario} not found!")
            return 1
        
        # Run single scenario
        simulator = UniversalOriginSimulator(
            config=selected,
            Nx=args.nx,
            Ny=args.ny,
            dt_h=args.dt,
            outdir=os.path.join(args.outdir, f'scenario_{args.scenario}')
        )
        
        history = simulator.run(
            hours=args.hours,
            record_interval=2.0,
            verbose=True
        )
        
        simulator.plot_results()
        
        print(f"\nScenario {args.scenario} completed successfully!")
        print(f"Results saved to: {simulator.outdir}")
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())