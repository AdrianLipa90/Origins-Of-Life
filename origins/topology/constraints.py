"""
Experimental zeta-indexed spectral regularizer.

The imaginary parts of selected non-trivial zeta zeros are used only as
indices for a deterministic spectral-notch family.  This module does not claim
that prebiotic chemistry is physically constrained by the Riemann zeta
function, and the stochastic term is an explicit numerical/model noise source,
not a Heisenberg-uncertainty derivation.
"""

from __future__ import annotations

import numpy as np

from ..constants import RIEMANN_CRITICAL_ZEROS


class ZetaRiemannModulator:
    """
    Apply an experimental zeta-indexed spectral regularizer.

    The modulation introduces spectral notches at deterministic normalized
    frequencies derived from the imaginary parts of the selected zeros.
    The mapping is resolution-independent and explicitly model-level.

    Parameters
    ----------
    zeros      : list of complex Riemann zeros (default: first 6)
    lambda_soft: softness parameter — larger = weaker constraint
    sigma_heis : legacy parameter name for Gaussian model-noise amplitude
    """

    def __init__(
        self,
        zeros: list | None = None,
        lambda_soft: float = 5.0,
        sigma_heis: float = 0.001,
    ):
        self.zeros = zeros if zeros is not None else RIEMANN_CRITICAL_ZEROS
        self.lambda_soft = lambda_soft
        self.sigma_heis  = sigma_heis

    # ------------------------------------------------------------------

    @staticmethod
    def _normalized_target(imag_part: float) -> float:
        """Map a positive zeta-zero ordinate monotonically into [0, 0.5).

        FFT frequencies are already normalized cycles/sample.  The historical
        implementation divided by Nx, changing the operator when grid
        resolution changed.  This bounded mapping keeps the candidate operator
        dimensionless and resolution-independent.
        """
        gamma = abs(float(imag_part))
        return 0.5 * gamma / (1.0 + gamma)

    def spectral_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Return the deterministic real-valued zeta-indexed notch mask."""
        Nx, Ny = shape
        if Nx <= 0 or Ny <= 0:
            raise ValueError("shape dimensions must be positive")
        kx = np.fft.fftfreq(Nx)
        ky = np.fft.fftfreq(Ny)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        k_mag = np.sqrt(KX**2 + KY**2)

        mask = np.ones((Nx, Ny), dtype=float)
        for z in self.zeros:
            target = self._normalized_target(abs(z.imag))
            mask *= 1.0 - np.exp(
                -((k_mag - target) ** 2) * self.lambda_soft
            )
        return np.clip(mask, 0.0, 1.0)

    # ------------------------------------------------------------------

    def apply(
        self,
        field: np.ndarray,
        rng: np.random.Generator,
        phase_coherence: float = 0.8,
    ) -> np.ndarray:
        """
        Apply zeta modulation + Heisenberg noise to a field.

        Parameters
        ----------
        field           : 2-D chemical field (modified in-place copy)
        rng             : reproducible RNG
        phase_coherence : fraction of field energy preserved (Euler phase term)
        """
        out = field.copy()

        # 1. Experimental zeta-indexed spectral notch regularization.
        freqs = np.fft.fft2(out)
        zeta_mod = self.spectral_mask(out.shape)
        freqs *= zeta_mod
        modulated = np.real(np.fft.ifft2(freqs))

        # 2. Candidate blend coefficient retained for compatibility.
        out = phase_coherence * out + (1.0 - phase_coherence) * modulated

        # 3. Explicit Gaussian model noise (legacy parameter: sigma_heis).
        if self.sigma_heis > 0:
            out += rng.normal(0.0, self.sigma_heis, out.shape)

        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------

    def apply_to_fields(
        self,
        fields: dict[str, np.ndarray],
        rng: np.random.Generator,
        phase_coherence: float = 0.8,
    ) -> dict[str, np.ndarray]:
        """Convenience wrapper: apply to a dict of named fields."""
        return {
            name: self.apply(f, rng, phase_coherence)
            for name, f in fields.items()
        }
