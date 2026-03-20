"""
Topological field generation — Kähler-Berry-Euler formalism.

The topology field T(x,y) acts as a spatially-varying modulator on all
chemical reaction rates, representing the influence of curved space-time
geometry (Kähler manifold curvature) on prebiotic chemistry.

Three time-evolution modes are supported:
  static  – field is fixed throughout the simulation
  pulsing – field amplitude oscillates with a given frequency
  drift   – field pattern translates spatially over time
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..scenarios import TopologyPattern, TimeDependence, ScenarioConfig
from ..chemistry.fields import laplacian


class TopologyField:
    """
    2-D Kähler-Berry-Euler topological field.

    Parameters
    ----------
    config : ScenarioConfig
    Nx, Ny : grid dimensions
    """

    def __init__(self, config: ScenarioConfig, Nx: int, Ny: int):
        self.config = config
        self.Nx = Nx
        self.Ny = Ny
        self.field: np.ndarray = np.zeros((Nx, Ny))
        self.curvature: np.ndarray = np.zeros((Nx, Ny))
        self._base: np.ndarray = np.zeros((Nx, Ny))

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
        self.field = s * base
        self._update_curvature()

    def _update_curvature(self) -> None:
        """Berry-phase curvature ≈ discrete Laplacian of the field."""
        lap = laplacian(self.field)
        std = float(np.std(lap))
        if std > 0:
            lap = (lap - float(lap.mean())) / (std + 1e-12)
        self.curvature = lap

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
