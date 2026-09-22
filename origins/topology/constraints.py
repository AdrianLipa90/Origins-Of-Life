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

        mask = np.clip(mask, 0.0, 1.0)
        # A spatial regularizer must not silently attenuate the DC component:
        # doing so makes repeated applications an unmodelled material sink.
        mask[0, 0] = 1.0
        return mask

    # ------------------------------------------------------------------

    @staticmethod
    def _project_box_sum(values: np.ndarray, target_sum: float) -> np.ndarray:
        """Project onto 0<=x<=1 while preserving a requested global sum.

        The projection is the Euclidean projection onto the capped simplex:
        x = clip(values + shift, 0, 1), with shift found by bisection.
        This makes the experimental operator a redistribution step rather than
        an implicit chemical source or sink.
        """
        arr = np.asarray(values, dtype=float)
        if not np.isfinite(arr).all():
            raise FloatingPointError("zeta regularizer received non-finite values")

        n = arr.size
        target = float(target_sum)
        if target < -1e-10 or target > float(n) + 1e-10:
            raise ValueError("target_sum is outside the feasible [0, field.size] range")
        target = min(max(target, 0.0), float(n))

        if target == 0.0:
            return np.zeros_like(arr, dtype=float)
        if target == float(n):
            return np.ones_like(arr, dtype=float)

        lo = -float(np.max(arr)) - 1.0
        hi = 1.0 - float(np.min(arr)) + 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            total = float(np.sum(np.clip(arr + mid, 0.0, 1.0)))
            if total < target:
                lo = mid
            else:
                hi = mid

        projected = np.clip(arr + 0.5 * (lo + hi), 0.0, 1.0)
        residual = target - float(np.sum(projected))
        if abs(residual) > 1e-10:
            free = (projected > 1e-12) & (projected < 1.0 - 1e-12)
            n_free = int(np.count_nonzero(free))
            if n_free:
                projected[free] += residual / n_free
                projected = np.clip(projected, 0.0, 1.0)

        if abs(float(np.sum(projected)) - target) > 1e-8:
            raise FloatingPointError("capped-simplex projection failed to conserve field sum")
        return projected

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
        out = np.asarray(field, dtype=float).copy()
        if out.ndim != 2:
            raise ValueError("zeta regularizer expects a 2-D field")
        if not np.isfinite(out).all():
            raise FloatingPointError("zeta regularizer received NaN/Inf")
        if float(np.min(out)) < -1e-10 or float(np.max(out)) > 1.0 + 1e-10:
            raise ValueError("zeta regularizer expects field values in [0, 1]")

        target_sum = float(np.sum(out))

        # 1. Experimental zeta-indexed spectral notch regularization.
        # spectral_mask() explicitly leaves DC untouched.
        freqs = np.fft.fft2(out)
        zeta_mod = self.spectral_mask(out.shape)
        freqs *= zeta_mod
        modulated = np.real(np.fft.ifft2(freqs))

        # 2. Candidate blend coefficient retained for compatibility.
        out = phase_coherence * out + (1.0 - phase_coherence) * modulated

        # 3. Explicit Gaussian model noise (legacy parameter: sigma_heis).
        # Remove its finite-sample mean so noise does not inject/remove material.
        if self.sigma_heis > 0:
            noise = rng.normal(0.0, self.sigma_heis, out.shape)
            noise -= float(np.mean(noise))
            out += noise

        # 4. Enforce the chemical no-source/no-sink boundary exactly under
        # the [0,1] field bounds. Any later material change must come from
        # explicit chemistry/dynamics, not from this spectral regularizer.
        return self._project_box_sum(out, target_sum)

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
