"""
Protocell detection utilities.

A proto-cell is identified as a grid location where both the membrane
field (M) and the genetic-polymer field (R) exceed defined thresholds,
indicating co-localisation of a lipid membrane with information-carrying
polymers.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, uniform_filter, label

from ..constants import PROTOCELL_THRESHOLD_M, PROTOCELL_THRESHOLD_R


class ProtocellDetector:
    """
    Detect proto-cells in a simulation snapshot using either a simple
    threshold method or an advanced morphology-aware method.
    """

    def __init__(
        self,
        threshold_M: float = PROTOCELL_THRESHOLD_M,
        threshold_R: float = PROTOCELL_THRESHOLD_R,
        window_size: int = 7,
    ):
        self.threshold_M = threshold_M
        self.threshold_R = threshold_R
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Simple detection
    # ------------------------------------------------------------------

    def detect(self, M: np.ndarray, R: np.ndarray) -> int:
        """
        Count grid cells that pass the basic membrane + polymer threshold.

        This is the fast, per-step method.
        """
        mask = (M > self.threshold_M) & (R > self.threshold_R)
        return int(mask.sum())

    # ------------------------------------------------------------------
    # Advanced detection (morphological)
    # ------------------------------------------------------------------

    def detect_advanced(
        self, M: np.ndarray, R: np.ndarray
    ) -> dict:
        """
        Morphological protocell detection with thermodynamic stability scoring.

        Returns a dict with:
          count          – number of detected structures
          stability_mean – mean thermodynamic stability score (0–1)
          labels         – labelled array (same shape as M)
        """
        # Adaptive threshold based on local statistics
        mem_mask = self._adaptive_threshold(M, self.threshold_M)
        rna_mask = self._adaptive_threshold(R, self.threshold_R)

        # Morphological cleaning: remove noise, fill small holes
        struct = np.ones((3, 3), dtype=bool)
        mem_clean = binary_erosion(binary_dilation(mem_mask, struct), struct)
        combined  = mem_clean & rna_mask

        labelled, n_structures = label(combined)
        if n_structures == 0:
            return {'count': 0, 'stability_mean': 0.0, 'labels': labelled}

        # Thermodynamic stability: based on local M gradient and R density
        stabilities = []
        for lbl in range(1, n_structures + 1):
            region = labelled == lbl
            m_vals = M[region]
            r_vals = R[region]
            # High M and high R → more stable
            score = float(np.mean(m_vals) * 0.6 + np.mean(r_vals) * 0.4)
            stabilities.append(min(score, 1.0))

        return {
            'count':          n_structures,
            'stability_mean': float(np.mean(stabilities)),
            'labels':         labelled,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _adaptive_threshold(
        field: np.ndarray,
        base_threshold: float,
        window_size: int = 7,
    ) -> np.ndarray:
        local_mean = uniform_filter(field, size=window_size)
        local_var  = uniform_filter(field**2, size=window_size) - local_mean**2
        local_std  = np.sqrt(np.maximum(local_var, 0.0))
        threshold  = local_mean + base_threshold * (1.0 + 0.5 * local_std)
        return field > threshold
