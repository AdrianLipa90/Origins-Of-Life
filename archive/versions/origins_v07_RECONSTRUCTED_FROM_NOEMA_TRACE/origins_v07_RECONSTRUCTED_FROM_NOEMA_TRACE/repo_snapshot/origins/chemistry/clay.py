"""
Montmorillonite clay mineral — catalysis and adsorption model.

Based on Ferris et al. (1996):
 – Clay surfaces concentrate nucleotides ~1000-fold.
 – Catalytic enhancement of RNA polymerisation: ~7.5×.
 – Clay-bound RNA degrades 100× more slowly than free RNA.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from ..constants import (
    CLAY_SURFACE_AREA_M2_G,
    CLAY_ADSORPTION_SITES_M2,
    CLAY_CONCENTRATION_FACTOR,
    CLAY_K_ADS,
    K_RNA_SYNTH_CLAY,
)

if TYPE_CHECKING:
    from ..biology.rna import RNASequence


class ClayMineral:
    """
    Montmorillonite clay particle with explicit surface chemistry.

    Each clay object represents a local patch in the 2-D simulation grid.
    """

    def __init__(
        self,
        position: Tuple[int, int],
        conc_g_L: float = 5.0,
    ):
        self.position = position
        self.conc_g_L = conc_g_L

        # Derived surface properties
        total_surface_m2 = (conc_g_L / 1000.0) * CLAY_SURFACE_AREA_M2_G
        self.total_sites = total_surface_m2 * CLAY_ADSORPTION_SITES_M2
        self._max_capacity = self.total_sites * 1e-9

        # State
        self.nucleotides_adsorbed: float = 0.0
        self.rna_molecules_bound: float  = 0.0

    # ------------------------------------------------------------------
    # Surface chemistry
    # ------------------------------------------------------------------

    def adsorb_nucleotides(self, n_free: float) -> float:
        """
        Langmuir adsorption isotherm.

        Returns the amount adsorbed (≤ n_free).
        """
        adsorbed = (
            (CLAY_K_ADS * n_free * self._max_capacity)
            / (1.0 + CLAY_K_ADS * n_free)
        )
        adsorbed = min(adsorbed, n_free)
        self.nucleotides_adsorbed += adsorbed
        return adsorbed

    def catalyze_polymerization(
        self,
        n_adsorbed: float,
        template_rna: Optional["RNASequence"] = None,
    ) -> float:
        """
        Clay-catalysed RNA polymerisation.

        A template RNA boosts synthesis by a fitness-dependent factor.
        Returns amount of new RNA synthesised (field units).
        """
        rna_synthesised = K_RNA_SYNTH_CLAY * n_adsorbed * 0.02
        if template_rna is not None:
            rna_synthesised *= 1.5 * template_rna.fitness
        return rna_synthesised

    def get_concentration_boost(self) -> float:
        """Local nucleotide concentration amplification factor."""
        return CLAY_CONCENTRATION_FACTOR

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ClayMineral(pos={self.position}, "
            f"conc={self.conc_g_L} g/L, "
            f"adsorbed={self.nucleotides_adsorbed:.3f})"
        )
