import numpy as np

try:
    import numba as _numba
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit
    def _laplacian_numba(Z):
        nx, ny = Z.shape
        out = np.empty_like(Z)
        for i in range(nx):
            for j in range(ny):
                up = Z[i - 1 if i - 1 >= 0 else nx - 1, j]
                down = Z[i + 1 if i + 1 < nx else 0, j]
                left = Z[i, j - 1 if j - 1 >= 0 else ny - 1]
                right = Z[i, j + 1 if j + 1 < ny else 0]
                out[i, j] = -4.0 * Z[i, j] + up + down + left + right
        return out

    def laplacian(Z):
        return _laplacian_numba(Z)
else:
    def laplacian(Z):
        return (-4 * Z + np.roll(Z, 1, 0) + np.roll(Z, -1, 0) + np.roll(Z, 1, 1) + np.roll(Z, -1, 1))
