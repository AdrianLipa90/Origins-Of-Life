"""
Primordial Black Hole analyser and CMB modulation engine (CIEL/0).

The TelescopeDataModulator combines all CIEL/0 invariants to compute
a unified modulation field for Cosmic Microwave Background spectra:

  M(x,y) = Σ̂ × ζ̂ × τ × Λ × I × Ψ_life

This can be used to search for organised signatures in observational data
from Planck, WMAP, or future CMB experiments (FITS format).

References
----------
CIEL-Omega Research, March 2025 — internal preprint.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..constants import GRAVITY_CONST, PBH_POSITION_AU, PBH_MASS_KG, AU_TO_METERS
from .operators import (
    SoulInvariantOperator,
    ZetaRiemannOperator,
    TimeFluidInvariant,
    LambdaPlasmaOperator,
    IntentionInvariant,
)


# ============================================================================
class PrimordialBlackHoleAnalyzer:
    """
    Analyse the gravitational influence of a proposed primordial black hole
    located at RA 7h10m18.395s / Dec −25°54′27″ (Puppis constellation).

    Position data from CIEL-Omega Research (March 2025 preprint).
    """

    def __init__(self):
        self.pbh_pos_au = np.array(PBH_POSITION_AU, dtype=float)
        self.pbh_mass   = PBH_MASS_KG
        self.celestial  = {
            'ra':            '7h10m18.395s',
            'dec':           "-25°54'27.284\"",
            'constellation': 'Puppis',
        }

    def gravitational_influence(self, distance_au: float) -> float:
        """
        Newtonian gravitational acceleration at distance *distance_au*.

        Returns acceleration in m/s².
        """
        distance_m = distance_au * AU_TO_METERS
        return GRAVITY_CONST * self.pbh_mass / distance_m**2

    def tno_perturbation_strength(
        self,
        tno_positions_au: np.ndarray,
    ) -> np.ndarray:
        """
        Gravitational perturbation strength on Trans-Neptunian Object positions.

        Parameters
        ----------
        tno_positions_au : (N, 3) array of TNO positions in AU

        Returns
        -------
        (N,) array of accelerations (m/s²)
        """
        perturbations = []
        for pos in tno_positions_au:
            dist = float(np.linalg.norm(pos - self.pbh_pos_au))
            perturbations.append(self.gravitational_influence(dist))
        return np.array(perturbations)

    def correlation_with_anomalies(
        self,
        anomaly_positions_au: np.ndarray,
        threshold_m_s2: float = 1e-12,
    ) -> Dict:
        """
        Compute positional correlation between the PBH and observed anomalies.

        Returns a summary dict with distances, influence values, and
        fraction of anomalies above the influence threshold.
        """
        distances = np.linalg.norm(
            anomaly_positions_au - self.pbh_pos_au, axis=1
        )
        influences = np.array([
            self.gravitational_influence(d) for d in distances
        ])
        above = float((influences > threshold_m_s2).mean())
        return {
            'mean_distance_au':     float(distances.mean()),
            'min_distance_au':      float(distances.min()),
            'mean_influence_m_s2':  float(influences.mean()),
            'fraction_above_thresh': above,
        }


# ============================================================================
class TelescopeDataModulator:
    """
    CMB spectrum modulator using the full CIEL/0 invariant set.

    The unified modulation field is computed as:

        M(x,y) = Σ̂ × ζ̂(f) × τ(x,t) × Λ_plasma × I_res × Ψ_life

    Each factor is evaluated across a 2-D frequency/angle grid.

    Usage
    -----
        mod = TelescopeDataModulator()
        result = mod.compute_unified_modulation(data_shape=(512, 512))
        sigs   = mod.detect_intelligence_signatures(result)
    """

    def __init__(self):
        self.soul_operator   = SoulInvariantOperator.default()
        self.zeta_operator   = ZetaRiemannOperator()
        self.time_fluid      = TimeFluidInvariant()
        self.lambda_plasma   = LambdaPlasmaOperator()
        self.intention_field = IntentionInvariant()
        self.pbh_analyzer    = PrimordialBlackHoleAnalyzer()

    # ------------------------------------------------------------------
    def compute_unified_modulation(
        self,
        data_shape: Tuple[int, int],
        observation_params: Optional[Dict] = None,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Compute M(x,y) for a grid of shape *data_shape*.

        Parameters
        ----------
        data_shape        : (Nx, Ny) output grid
        observation_params: optional dict with keys B, rho, L, resonance
        t                 : observation epoch (arbitrary units)
        """
        if observation_params is None:
            observation_params = {
                'B': 1e-10, 'rho': 1e-27, 'L': 3e22, 'resonance': 1.0
            }

        Nx, Ny = data_shape
        x = np.linspace(-np.pi, np.pi, Nx)
        y = np.linspace(-np.pi, np.pi, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # 1. Soul invariant — scalar applied uniformly
        soul_val   = abs(self.soul_operator.compute())
        sigma_field = np.full(data_shape, soul_val)

        # 2. Zeta-Riemann — frequency modulation along x-axis
        freq_1d = np.linspace(1.0, 40.0, Nx)
        spectrum_1d = np.ones(Nx, dtype=complex)
        zeta_1d = self.zeta_operator.apply(spectrum_1d, freq_1d)
        zeta_field = np.outer(np.abs(zeta_1d) / (np.abs(zeta_1d).max() + 1e-12),
                              np.ones(Ny))

        # 3. Temporal modulation
        coords = np.linspace(0, 1, Nx)
        tau_1d = self.time_fluid.temporal_modulation(coords, t)
        tau_field = np.outer(tau_1d / (np.abs(tau_1d).max() + 1e-12), np.ones(Ny))

        # 4. Lambda-plasma — scalar
        lp = self.lambda_plasma.compute(
            observation_params['B'],
            observation_params['rho'],
            observation_params['L'],
            observation_params['resonance'],
        )
        lambda_field = np.full(data_shape, min(abs(lp), 10.0))

        # 5. Intention resonance — angular pattern
        flat = (X + Y).ravel()
        intention_1d = np.array([
            self.intention_field.compute_resonance(np.array([v]))
            for v in flat[:min(100, len(flat))]
        ])
        intention_scalar = float(np.mean(intention_1d)) if len(intention_1d) > 0 else 1.0
        intention_field_arr = np.full(data_shape, min(intention_scalar, 5.0))

        # 6. Ψ_life — PBH gravitational imprint on the sky plane
        r = np.sqrt(X**2 + Y**2) + 1e-9
        pbh_distance_au = float(np.linalg.norm(self.pbh_analyzer.pbh_pos_au))
        psi_life = np.exp(-r / (pbh_distance_au / 1e3 + 1e-9))

        # Combined modulation (normalised)
        M = sigma_field * zeta_field * np.abs(tau_field) * lambda_field * intention_field_arr * psi_life
        M_max = M.max()
        return M / (M_max + 1e-12)

    # ------------------------------------------------------------------
    def detect_intelligence_signatures(
        self, modulation_field: np.ndarray, threshold: float = 0.7
    ) -> Dict:
        """
        Search for organised signatures above *threshold*.

        Returns a summary dict of candidate statistics.
        """
        from scipy.ndimage import label
        binary = modulation_field > threshold
        labelled, n_structures = label(binary)
        sizes = [int((labelled == i).sum()) for i in range(1, n_structures + 1)]
        return {
            'n_structures':   n_structures,
            'mean_size':      float(np.mean(sizes)) if sizes else 0.0,
            'max_size':       float(max(sizes)) if sizes else 0.0,
            'coverage_frac':  float(binary.mean()),
            'soul_quantized': self.soul_operator.is_integer_quantized(),
            'pbh_position':   self.pbh_analyzer.celestial,
        }
