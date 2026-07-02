"""
Zeta-Riemann soft constraints and Heisenberg uncertainty noise injection.

The ZetaRiemannModulator applies soft modulation derived from the
non-trivial zeros of the Riemann zeta function ζ(s) to chemical fields,
enforcing quantum-inspired field coherence (CIEL/0 framework).
"""

from __future__ import annotations

import numpy as np

from ..constants import RIEMANN_CRITICAL_ZEROS


class ZetaRiemannModulator:
    """
    Apply Zeta-Riemann spectral constraints to a chemical field.

    The modulation introduces resonance peaks at spatial frequencies
    corresponding to the imaginary parts of the Riemann zeros,
    suppressing divergence while preserving structured patterns.

    Parameters
    ----------
    zeros      : list of complex Riemann zeros (default: first 6)
    lambda_soft: softness parameter — larger = weaker constraint
    sigma_heis : Heisenberg noise amplitude (standard deviation)
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

        # 1. Zeta soft constraint via spectral smoothing
        freqs = np.fft.fft2(out)
        Nx, Ny = out.shape
        kx = np.fft.fftfreq(Nx)
        ky = np.fft.fftfreq(Ny)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        k_mag = np.sqrt(KX**2 + KY**2) + 1e-9

        # Build modulation from Riemann zeros
        zeta_mod = np.ones((Nx, Ny), dtype=complex)
        for z in self.zeros:
            imag_part = abs(z.imag)
            # soft Lorentzian resonance
            zeta_mod *= 1.0 - np.exp(
                -((k_mag - imag_part / (2 * np.pi * Nx))**2)
                * self.lambda_soft
            )

        freqs *= zeta_mod
        modulated = np.real(np.fft.ifft2(freqs))

        # 2. Phase coherence (Euler term): blend original and modulated
        out = phase_coherence * out + (1.0 - phase_coherence) * modulated

        # 3. Heisenberg noise
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
