"""
Field utilities: initialisation, diffusion, Laplacian, environmental drivers.

All simulation grids are 2-D NumPy arrays of shape (Nx, Ny) with
periodic boundary conditions.
"""

import math
import numpy as np

# Optional Numba JIT for Laplacian
try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    _NUMBA = False


# ============================================================================
# LAPLACIAN
# ============================================================================

if _NUMBA:
    @_njit
    def _laplacian_nb(Z):
        nx, ny = Z.shape
        out = np.empty_like(Z)
        for i in range(nx):
            for j in range(ny):
                up    = Z[i - 1 if i > 0 else nx - 1, j]
                down  = Z[i + 1 if i < nx - 1 else 0, j]
                left  = Z[i, j - 1 if j > 0 else ny - 1]
                right = Z[i, j + 1 if j < ny - 1 else 0]
                out[i, j] = -4.0 * Z[i, j] + up + down + left + right
        return out

    def laplacian(Z: np.ndarray) -> np.ndarray:
        """5-point discrete Laplacian (Numba-accelerated)."""
        return _laplacian_nb(Z)
else:
    def laplacian(Z: np.ndarray) -> np.ndarray:
        """5-point discrete Laplacian with periodic boundaries."""
        return (
            -4.0 * Z
            + np.roll(Z,  1, axis=0)
            + np.roll(Z, -1, axis=0)
            + np.roll(Z,  1, axis=1)
            + np.roll(Z, -1, axis=1)
        )


# ============================================================================
# FIELD INITIALISATION
# ============================================================================

def init_field(
    Nx: int,
    Ny: int,
    mean: float,
    noise_std: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Initialise a 2-D chemical field.

    Parameters
    ----------
    Nx, Ny : grid dimensions
    mean   : base concentration value
    noise_std : standard deviation of additive Gaussian noise
    rng    : optional NumPy random generator (reproducibility)
    """
    if rng is None:
        rng = np.random.default_rng()
    field = np.full((Nx, Ny), mean, dtype=np.float32)
    field += rng.normal(0.0, noise_std, (Nx, Ny)).astype(np.float32)
    return np.clip(field, 0.0, 1.0)


# ============================================================================
# DIFFUSION
# ============================================================================

def diffuse_field(field: np.ndarray, D: float, dt: float) -> np.ndarray:
    """
    Apply one explicit Euler diffusion step.

    Parameters
    ----------
    field : 2-D concentration array
    D     : diffusion coefficient (grid units²/hour)
    dt    : time step (hours)
    """
    return np.clip(field + D * laplacian(field) * dt, 0.0, None)


# ============================================================================
# ENVIRONMENTAL DRIVERS
# ============================================================================

def solar_envelope(t_h: float, day_length_h: float = 24.0) -> float:
    """
    Smooth day/night UV envelope (cosine bell, 0→1).

    Returns 0 during night, peaks at solar noon.
    """
    phase = (t_h % day_length_h) / day_length_h
    value = (math.sin(2.0 * math.pi * (phase - 0.25)) + 1.0) / 2.0
    return max(0.0, min(1.0, value))


def sun_envelope_window(
    t_h: float,
    mu: float = 1.0,
    day_fraction: float = 0.5,
    day_length_h: float = 24.0,
) -> float:
    """
    Cosine-bell day envelope with controllable day fraction and zenith factor.

    Parameters
    ----------
    mu           : cos(zenith) scaling factor
    day_fraction : fraction of 24 h that is daylight
    """
    h = t_h % day_length_h
    half = 12.0 * day_fraction
    d = abs(h - 12.0)
    if d > half or half <= 0:
        return 0.0
    return float(mu * 0.5 * (1.0 + math.cos(math.pi * d / half)))


def tide_semidiurnal(t_h: float, period_h: float = 12.42) -> float:
    """Semi-diurnal tidal forcing (dimensionless, −1 … +1)."""
    return math.sin(2.0 * math.pi * t_h / period_h)


# ============================================================================
# GRADIENT UTILITIES
# ============================================================================

def gradient_norm(Z: np.ndarray) -> np.ndarray:
    """L2 norm of the discrete gradient (central differences)."""
    gx = np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)
    gy = np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)
    return np.sqrt(gx**2 + gy**2 + 1e-12)
