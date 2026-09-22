"""
Candidate geometric field generation for abiogenesis experiments.

The topology field T(x,y) is a synthetic spatial modulator on chemical
reaction rates. The Bloch/CP1 mapping and geometric-phase readout are
experimental model components; they are not asserted to be a derived
space-time or quantum-chemical mechanism.

Three time-evolution modes are supported:
  static  – field is fixed throughout the simulation
  pulsing – field amplitude oscillates with a given frequency
  drift   – field pattern translates spatially over time

Bloch sphere geometry (2026 extension):
  Each point on the 2-D grid maps to a point on S² via:
    bloch_theta = pi * (field_norm + 1) / 2   ∈ [0, π]
    bloch_phi   = 2*pi * curvature_norm        ∈ [0, 2π]
  Geometric-phase candidate = discrete integral of A_phi dphi along the
  modeled CP1 path. Pure positive amplitude rescaling must not create an
  azimuthal phase change by normalization artifact.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..scenarios import TopologyPattern, TimeDependence, ScenarioConfig
from ..chemistry.fields import laplacian


class TopologyField:
    """
    2-D Kähler-Berry-Euler topological field with Bloch sphere geometry.

    Parameters
    ----------
    config : ScenarioConfig
    Nx, Ny : grid dimensions

    Attributes
    ----------
    field          : main topology modulator T(x,y)
    curvature      : Berry-phase curvature (Laplacian of field)
    bloch_theta    : polar angle on S² per grid point ∈ [0, π]
    bloch_phi      : azimuthal angle on S² per grid point ∈ [0, 2π]
    berry_accumulated : scalar accumulated Berry phase (holonomy)
    """

    def __init__(self, config: ScenarioConfig, Nx: int, Ny: int):
        self.config = config
        self.Nx = Nx
        self.Ny = Ny
        self.field: np.ndarray = np.zeros((Nx, Ny))
        self.curvature: np.ndarray = np.zeros((Nx, Ny))
        self.curvature_raw: np.ndarray = np.zeros((Nx, Ny))
        self._base: np.ndarray = np.zeros((Nx, Ny))
        self._base_laplacian_scale: float = 1.0
        self.bloch_theta: np.ndarray = np.full((Nx, Ny), math.pi / 2)
        self.bloch_phi: np.ndarray = np.zeros((Nx, Ny))
        self.berry_accumulated: float = 0.0
        self._prev_bloch_phi: Optional[np.ndarray] = None

        rng = np.random.default_rng(config.seed)
        self._rng = rng
        self._build(rng)

    # ------------------------------------------------------------------
    def _build(self, rng: np.random.Generator) -> None:
        """Construct the base pattern and compute initial curvature."""
        x = np.linspace(-1.0, 1.0, self.Nx)
        y = np.linspace(-1.0, 1.0, self.Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        s = float(self.config.topo_strength)
        pattern = self.config.topo_pattern

        if pattern == TopologyPattern.SINUSOIDAL:
            base = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        elif pattern == TopologyPattern.COSINUSOIDAL:
            base = np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        elif pattern == TopologyPattern.VORTEX:
            theta = np.arctan2(Y, X)
            r = np.sqrt(X**2 + Y**2) + 1e-9
            base = np.sin(4 * theta) * np.exp(-3 * r**2)
        elif pattern == TopologyPattern.GAUSSIAN:
            base = (
                np.exp(-((X - 0.2)**2 + (Y + 0.1)**2) / 0.02)
                - 0.5 * np.exp(-((X + 0.3)**2 + (Y - 0.3)**2) / 0.05)
            )
        elif pattern == TopologyPattern.RANDOM:
            noise = rng.standard_normal((self.Nx, self.Ny))
            # Smooth with a single Laplacian pass
            base = (
                np.roll(noise, 1, 0) + noise + np.roll(noise, -1, 0)
                + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)
            ) / 5.0
        else:  # STATIC / flat
            base = np.zeros((self.Nx, self.Ny))

        # Normalise to zero-mean unit-variance, then scale
        std = float(np.std(base))
        if std > 0:
            base = (base - float(np.mean(base))) / (std + 1e-12)
        self._base = base
        base_lap = laplacian(self._base)
        self._base_laplacian_scale = max(float(np.std(base_lap)), 1e-12)
        self.field = s * base
        self._update_curvature()

    def _update_curvature(self) -> None:
        """Update the Laplacian-derived candidate curvature and Bloch coordinates.

        The normalization scale is frozen from the base pattern. Re-normalizing
        every time step erases topo_strength and can turn pure amplitude
        modulation into a spurious azimuthal rotation.
        """
        lap = laplacian(self.field)
        self.curvature_raw = lap
        self.curvature = lap / self._base_laplacian_scale
        self._update_bloch()

    def _update_bloch(self) -> None:
        """Map field → Bloch sphere coordinates and accumulate Berry phase.

        Fubini-Study metric on CP¹ ≃ S²:
          bloch_theta ∈ [0,π]: polar angle, driven by field amplitude
          bloch_phi   ∈ [0,2π]: azimuthal angle, driven by curvature

        Berry connection A = (1-cos θ)/2 · dφ (magnetic monopole gauge).
        Accumulated holonomy ≈ mean(A · Δφ) over all grid points.

        Pure positive amplitude rescaling changes theta but, when field,
        curvature and gradients scale consistently, does not by itself rotate
        phi. Therefore amplitude-only pulsing is not forced to accumulate a
        geometric phase.
        """
        # Polar angle from absolute dimensionless amplitude. A fixed reference
        # keeps topo_strength identifiable instead of cancelling it out.
        field_reference = 1.0
        self.bloch_theta = 2.0 * np.arctan(np.abs(self.field) / field_reference)

        # Azimuthal angle: atan2(curvature, field) + spatial gradient phase
        # For DRIFT mode: field shifts spatially → gradient changes sign → phi rotates
        raw_phi = np.arctan2(self.curvature, self.field)  # [-π, π]
        # Add gradient-induced winding: grad_x field contributes to azimuthal rotation
        grad_x = np.roll(self.field, -1, axis=0) - np.roll(self.field, 1, axis=0)
        grad_y = np.roll(self.field, -1, axis=1) - np.roll(self.field, 1, axis=1)
        grad_phi = np.arctan2(grad_y, grad_x + 1e-12) * 0.1  # small spatial contribution
        new_phi = (raw_phi + grad_phi) % (2.0 * math.pi)  # [0, 2π]

        # Accumulate Berry phase: A·Δφ where A = (1-cos θ)/2
        if self._prev_bloch_phi is not None:
            delta_phi = new_phi - self._prev_bloch_phi
            # Wrap delta to (-π, π) to avoid 2π jumps
            delta_phi = (delta_phi + math.pi) % (2.0 * math.pi) - math.pi
            berry_connection = (1.0 - np.cos(self.bloch_theta)) / 2.0
            self.berry_accumulated += float(np.mean(berry_connection * delta_phi))

        self._prev_bloch_phi = new_phi.copy()
        self.bloch_phi = new_phi

    def bloch_coherence(self) -> float:
        """Mean cos²(theta/2) of the candidate CP1 mapping."""
        return float(np.mean(np.cos(self.bloch_theta / 2.0) ** 2))

    def geometry_status(self) -> dict[str, object]:
        """Explicit epistemic status of the synthetic geometry layer."""
        return {
            "status": "GEOMETRIC_CANDIDATE",
            "physical_binding": "OPEN",
            "curvature_operator": "LAPLACIAN_DERIVED_FIXED_REFERENCE",
            "phase_readout": "DISCRETE_CP1_CONNECTION_CANDIDATE",
        }

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------

    def advance(self, t_h: float) -> None:
        """
        Update the topology field for the current simulation time.

        Only modifies the field if `topo_time_dependence` is not STATIC.
        """
        mode = self.config.topo_time_dependence
        if mode == TimeDependence.STATIC:
            return

        s = float(self.config.topo_strength)

        if mode == TimeDependence.PULSING:
            freq = max(1e-9, self.config.topo_pulse_freq)
            factor = 1.0 + 0.5 * math.sin(2.0 * math.pi * freq * t_h)
            self.field = s * self._base * factor

        elif mode == TimeDependence.DRIFT:
            shift = int((t_h * 0.02) % self.Nx)
            self.field = np.roll(s * self._base, shift, axis=0)

        self._update_curvature()

    # ------------------------------------------------------------------
    # Modulation helpers
    # ------------------------------------------------------------------

    def synthesis_mod(self) -> np.ndarray:
        """Modulation factor for polymer synthesis step."""
        return np.clip(
            1.0 + 0.8 * self.field + 0.2 * self.curvature,
            0.1, 5.0,
        )

    def catalysis_mod(self) -> np.ndarray:
        """Modulation factor for catalysis step."""
        return np.clip(
            1.0 + 0.5 * self.field + 0.6 * self.curvature,
            0.2, 4.0,
        )

    def energy_mod(self) -> np.ndarray:
        """Modulation factor for energy conversion step."""
        return np.clip(
            1.0 + 0.6 * self.field + 0.4 * self.curvature,
            0.2, 3.0,
        )

    def degradation_mod(self) -> np.ndarray:
        """Modulation factor for degradation (negative topology stabilises)."""
        return np.clip(
            1.0 + 0.5 * (-self.field) + 0.3 * self.curvature,
            0.05, 4.0,
        )

    def membrane_mod(self) -> np.ndarray:
        """Modulation factor for membrane formation."""
        return np.clip(
            1.0 + 0.6 * self.field + 0.5 * self.curvature,
            0.05, 6.0,
        )
