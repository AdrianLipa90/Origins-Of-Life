"""
CIEL/0 invariant operators for CMB modulation analysis.

Mathematical framework (CIEL-Omega Research):

  Σ̂ = exp(i∮_C A_φ dl)          — Soul invariant (topological quantisation)
  ζ̂                              — Zeta-Riemann operator (spectral modulation)
  τ(x,t) = exp(-c_s²∇²t/2)      — Time-fluid invariant (hydrodynamic time)
  Λ_plasma = B²/(μ₀ρc²L²)×res  — Dynamic plasma cosmological constant
  I(x) = |I|e^(iφ)               — Intention field resonance
"""

from __future__ import annotations

import numpy as np

from ..constants import (
    RIEMANN_CRITICAL_ZEROS,
    VACUUM_PERMEABILITY,
    SPEED_OF_LIGHT,
    SCHUMANN_HARMONICS_HZ,
)


# ============================================================================
class SoulInvariantOperator:
    """
    Soul invariant  Σ̂ = exp(i ∮_C A_φ dl).

    Computes the holonomy of a U(1) gauge field along a closed loop C,
    acting as a topological quantisation signature.

    Parameters
    ----------
    gauge_connection : 1-D complex array — A_φ values along the loop
    loop             : (N, 3) array of loop points in parameter space
    """

    def __init__(
        self,
        gauge_connection: np.ndarray,
        loop: np.ndarray,
    ):
        self.A_phi = gauge_connection
        self.loop  = loop

    def compute(self) -> complex:
        """Return Σ̂ as a complex number on the unit circle."""
        deltas = np.diff(self.loop, axis=0)
        A_trim = self.A_phi[:deltas.shape[0]]
        integrand = np.sum(np.sum(A_trim[:, None] * deltas, axis=1))
        return complex(np.exp(1j * integrand))

    def is_integer_quantized(self, threshold: float = 1e-6) -> bool:
        """
        Check whether the winding number is close to an integer.

        True ↔ topologically quantised (consistent with advanced structure).
        """
        phase   = np.angle(self.compute())
        winding = phase / (2 * np.pi)
        return bool(np.min(np.abs(winding - np.round(winding))) < threshold)

    @classmethod
    def default(cls, n_points: int = 100) -> "SoulInvariantOperator":
        """Create a unit-circle loop with flat gauge connection."""
        theta = np.linspace(0, 2 * np.pi, n_points)
        loop  = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n_points)])
        gauge = np.ones(n_points - 1, dtype=complex) * 0.5
        return cls(gauge, loop)


# ============================================================================
class ZetaRiemannOperator:
    """
    Riemann-zeta spectrum operator for CMB modulation.

    Modulates an observational spectrum by inserting resonance nodes at
    the non-trivial Riemann zeros, analogous to a notch-filter bank.

    Parameters
    ----------
    zeros : complex array of Riemann critical-line zeros
    """

    def __init__(self, zeros: np.ndarray | None = None):
        if zeros is None:
            zeros = np.array(RIEMANN_CRITICAL_ZEROS)
        self.zeros = zeros

    def apply(
        self,
        psi_spectrum: np.ndarray,
        s_values: np.ndarray,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """
        Multiply spectrum by the zeta-modulation kernel.

        Parameters
        ----------
        psi_spectrum : input spectrum (real or complex)
        s_values     : spectral abscissa (same shape as psi_spectrum)
        """
        zeta_vals = np.ones_like(s_values, dtype=complex)
        for z in self.zeros:
            zeta_vals *= s_values - (z + epsilon)
        return zeta_vals * psi_spectrum


# ============================================================================
class TimeFluidInvariant:
    """
    Time-fluid invariant  τ(x,t) = exp(-c_s² ∇²t / 2).

    Treats time as a compressible fluid with sound speed c_s, producing
    a spatially-varying temporal dilation field.
    """

    def __init__(self, flow_velocity: float = 1.0):
        self.c_s = flow_velocity

    def temporal_modulation(self, coords: np.ndarray, t: float) -> np.ndarray:
        lap = np.gradient(np.gradient(coords))
        return np.exp(-self.c_s**2 * lap * t / 2.0)


# ============================================================================
class LambdaPlasmaOperator:
    """
    Dynamic plasma cosmological constant.

    Λ_plasma = B² / (μ₀ ρ c²) × (1/L²) × resonance
    """

    def __init__(self):
        self.mu0 = VACUUM_PERMEABILITY
        self.c   = SPEED_OF_LIGHT

    def compute(
        self,
        B: float,
        rho: float,
        L: float,
        resonance: float,
    ) -> float:
        """
        Parameters
        ----------
        B         : magnetic field strength (T)
        rho       : mass density (kg/m³)
        L         : characteristic length scale (m)
        resonance : dimensionless resonance factor (e.g. Schumann coupling)
        """
        return (B**2 / (self.mu0 * rho * self.c**2)) * (1.0 / L**2) * resonance


# ============================================================================
class IntentionInvariant:
    """
    Intention field  I(x) = |I| e^(iφ).

    Represents a coherent phase-locked field whose overlap with an
    observational state vector constitutes an 'intention signature'.
    """

    def __init__(self, amplitude: float = 1.0, phase: float = 0.0):
        self.amplitude = amplitude
        self.phase     = phase

    def compute_resonance(self, symbolic_state: np.ndarray) -> float:
        """
        Squared overlap of the intention field with the symbolic state.
        """
        intention_vec = self.amplitude * np.exp(1j * self.phase)
        return float(abs(np.vdot(symbolic_state, intention_vec))**2)

    def schumann_coupling(self, t_h: float) -> float:
        """Schumann-harmonic coupling strength at simulation time t_h."""
        omega = 2 * np.pi * np.array(SCHUMANN_HARMONICS_HZ) / 3600.0
        return float(np.sum(np.cos(omega * t_h)) / len(omega))
