"""Empirical RNA-polymerization reaction-regime anchors.

This module provides representation/provenance only. It intentionally contains
no fitted kinetic constants and no topology/zeta/replication terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


class MonomerReactiveState(str, Enum):
    CYCLIC_2_3_PHOSPHATE = "CYCLIC_2_3_PHOSPHATE"
    ORDINARY_5_MONOPHOSPHATE_FREE_ACID = "ORDINARY_5_MONOPHOSPHATE_FREE_ACID"
    IMIDAZOLIDE_ACTIVATED = "IMIDAZOLIDE_ACTIVATED"
    UNSPECIFIED_LEGACY = "UNSPECIFIED_LEGACY"


class InterfaceState(str, Enum):
    SLOW_DRY_FILM = "SLOW_DRY_FILM"
    HOT_WATER_SOLID_THIN_FILM = "HOT_WATER_SOLID_THIN_FILM"
    MINERAL_SURFACE = "MINERAL_SURFACE"
    GENERIC_LEGACY = "GENERIC_LEGACY"


class RegimeId(str, Enum):
    CAIMI_2025_AUGC_PH11 = "CAIMI_2025_AUGC_PH11"
    SONG_2024_UMP_HOT_ACID = "SONG_2024_UMP_HOT_ACID"
    UNBOUND = "UNBOUND"


@dataclass(frozen=True)
class RegimeSignature:
    monomer_state: MonomerReactiveState
    interface: InterfaceState
    temperature_C: float
    pH: float
    total_concentration_mM: float | None
    wet_dry_cycles: int | None


@dataclass(frozen=True)
class PolymerizationAnchor:
    regime_id: RegimeId
    doi: str
    monomer_state: MonomerReactiveState
    interface: InterfaceState
    temperature_C: float
    temperature_tolerance_C: float
    pH_min: float
    pH_max: float
    concentration_mM_min: float | None
    concentration_mM_max: float | None
    cycles_min: int | None
    cycles_max: int | None
    observables: Mapping[str, float | int | tuple[int, ...] | str]
    limitations: tuple[str, ...]

    def matches(self, signature: RegimeSignature) -> bool:
        if signature.monomer_state is not self.monomer_state:
            return False
        if signature.interface is not self.interface:
            return False
        if abs(signature.temperature_C - self.temperature_C) > self.temperature_tolerance_C:
            return False
        if not (self.pH_min <= signature.pH <= self.pH_max):
            return False

        if (
            self.concentration_mM_min is not None
            and self.concentration_mM_max is not None
        ):
            if signature.total_concentration_mM is None:
                return False
            if not (
                self.concentration_mM_min
                <= signature.total_concentration_mM
                <= self.concentration_mM_max
            ):
                return False

        if self.cycles_min is not None and self.cycles_max is not None:
            if signature.wet_dry_cycles is None:
                return False
            if not (
                self.cycles_min
                <= signature.wet_dry_cycles
                <= self.cycles_max
            ):
                return False

        return True


CAIMI_2025_AUGC_PH11 = PolymerizationAnchor(
    regime_id=RegimeId.CAIMI_2025_AUGC_PH11,
    doi="10.1021/acscentsci.5c00488",
    monomer_state=MonomerReactiveState.CYCLIC_2_3_PHOSPHATE,
    interface=InterfaceState.SLOW_DRY_FILM,
    temperature_C=23.0,
    temperature_tolerance_C=2.0,
    pH_min=10.5,
    pH_max=11.5,
    concentration_mM_min=45.0,
    concentration_mM_max=55.0,
    cycles_min=10,
    cycles_max=10,
    observables={
        "polymerization_yield_fraction": 0.36,
        "max_detected_nt": 8,
        "eight_mer_concentration_uM": 30.0,
        "eight_mer_mass_fraction": 0.005,
        "cycle_period_h": 24.0,
    },
    limitations=(
        "The 8-mer/36% values belong to the pH-11 AUGC condition; "
        "pH-10 10-mer results are not merged into this anchor.",
        "Reported product concentrations are analytical observables, "
        "not direct kinetic rate constants.",
    ),
)


SONG_2024_UMP_HOT_ACID = PolymerizationAnchor(
    regime_id=RegimeId.SONG_2024_UMP_HOT_ACID,
    doi="10.1073/pnas.2412784121",
    monomer_state=MonomerReactiveState.ORDINARY_5_MONOPHOSPHATE_FREE_ACID,
    interface=InterfaceState.HOT_WATER_SOLID_THIN_FILM,
    temperature_C=85.0,
    temperature_tolerance_C=3.0,
    # The free-acid protocols are strongly acidic and base-dependent.
    # The broad binding window is deliberate; it is not a pH-rate law.
    pH_min=1.0,
    pH_max=3.0,
    concentration_mM_min=8.0,
    concentration_mM_max=12.0,
    cycles_min=2,
    cycles_max=2,
    observables={
        "ump_mean_detected_nt": 16.3,
        "ump_sd_detected_nt": 10.5,
        "ump_max_detected_nt": 53,
        "explicitly_identified_lengths_nt": (35, 43),
    },
    limitations=(
        "MS intensity and ion-cluster frequency are approximate abundance proxies; "
        "ionization efficiency can depend on chain length.",
        "This anchor describes the two-cycle UMP distribution and must not be "
        "silently mixed with third-cycle degradation behavior.",
    ),
)


POLYMERIZATION_ANCHORS: tuple[PolymerizationAnchor, ...] = (
    CAIMI_2025_AUGC_PH11,
    SONG_2024_UMP_HOT_ACID,
)


@dataclass(frozen=True)
class FunctionalAnchor:
    name: str
    doi: str
    length_nt: int
    observables: Mapping[str, float | str]
    warning: str


QT45_FUNCTIONAL_ANCHOR = FunctionalAnchor(
    name="QT45",
    doi="10.1126/science.adt2760",
    length_nt=45,
    observables={
        "complement_fidelity_per_nt": 0.941,
        "self_copy_yield_fraction": 0.002,
        "self_copy_duration_days": 72.0,
        "substrate_class": "activated trinucleotide triphosphates",
        "environment": "mildly alkaline eutectic ice",
    },
    warning=(
        "Specific selected 45-nt sequence and reaction system; "
        "not evidence that an arbitrary 45-mer is a replicator."
    ),
)


FUNCTIONAL_ANCHORS: tuple[FunctionalAnchor, ...] = (
    QT45_FUNCTIONAL_ANCHOR,
)


def classify_regime(signature: RegimeSignature) -> RegimeId:
    """Return one unique empirical binding or UNBOUND.

    Multiple matches are treated as a representation error instead of being
    resolved by arbitrary priority.
    """
    matches = [a.regime_id for a in POLYMERIZATION_ANCHORS if a.matches(signature)]
    if not matches:
        return RegimeId.UNBOUND
    if len(matches) != 1:
        raise RuntimeError(f"ambiguous empirical regime binding: {matches}")
    return matches[0]


def require_empirical_binding(signature: RegimeSignature) -> PolymerizationAnchor:
    regime = classify_regime(signature)
    if regime is RegimeId.UNBOUND:
        raise ValueError("reaction signature is not bound to an empirical anchor")
    for anchor in POLYMERIZATION_ANCHORS:
        if anchor.regime_id is regime:
            return anchor
    raise RuntimeError("classified regime missing from registry")


def chain_mass_fractions(counts: Sequence[float]) -> np.ndarray:
    """Convert arbitrary chain counts to nucleotide-mass fractions by length."""
    arr = np.asarray(counts, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("counts must be a non-empty 1-D sequence")
    if not np.isfinite(arr).all():
        raise FloatingPointError("counts contain NaN/Inf")
    if np.any(arr < 0.0):
        raise ValueError("counts must be non-negative")

    lengths = np.arange(1, arr.size + 1, dtype=float)
    mass = arr * lengths
    total = float(np.sum(mass))
    if total <= 0.0:
        raise ValueError("positive nucleotide mass is required")
    return mass / total


def polymerized_mass_fraction(counts: Sequence[float]) -> float:
    """Mass fraction carried by chains of length >=2."""
    fractions = chain_mass_fractions(counts)
    return float(np.sum(fractions[1:]))


def tail_mass_fraction(counts: Sequence[float], min_length_nt: int) -> float:
    if min_length_nt < 1:
        raise ValueError("min_length_nt must be >=1")
    fractions = chain_mass_fractions(counts)
    if min_length_nt > fractions.size:
        return 0.0
    return float(np.sum(fractions[min_length_nt - 1 :]))


def caimi_signature() -> RegimeSignature:
    return RegimeSignature(
        monomer_state=MonomerReactiveState.CYCLIC_2_3_PHOSPHATE,
        interface=InterfaceState.SLOW_DRY_FILM,
        temperature_C=23.0,
        pH=11.0,
        total_concentration_mM=50.0,
        wet_dry_cycles=10,
    )


def song_ump_signature() -> RegimeSignature:
    return RegimeSignature(
        monomer_state=MonomerReactiveState.ORDINARY_5_MONOPHOSPHATE_FREE_ACID,
        interface=InterfaceState.HOT_WATER_SOLID_THIN_FILM,
        temperature_C=85.0,
        pH=2.0,
        total_concentration_mM=10.0,
        wet_dry_cycles=2,
    )


def legacy_first_rna_v02_signature() -> RegimeSignature:
    """Signature of the legacy candidate gate; intentionally empirically unbound."""
    return RegimeSignature(
        monomer_state=MonomerReactiveState.UNSPECIFIED_LEGACY,
        interface=InterfaceState.GENERIC_LEGACY,
        temperature_C=65.0,
        pH=7.5,
        total_concentration_mM=None,
        wet_dry_cycles=None,
    )


def representation_gate_receipt() -> dict[str, object]:
    caimi = classify_regime(caimi_signature())
    song = classify_regime(song_ump_signature())
    legacy = classify_regime(legacy_first_rna_v02_signature())

    # Explicit scale-invariance check on a nontrivial synthetic distribution.
    counts = np.array([100.0, 20.0, 5.0, 1.0, 0.25], dtype=float)
    f1 = chain_mass_fractions(counts)
    f2 = chain_mass_fractions(counts * 1.0e9)
    scale_invariant = bool(np.array_equal(f1, f2))

    qt45_separate = all(
        not hasattr(anchor, "regime_id")
        for anchor in FUNCTIONAL_ANCHORS
    )

    passed = bool(
        caimi is RegimeId.CAIMI_2025_AUGC_PH11
        and song is RegimeId.SONG_2024_UMP_HOT_ACID
        and legacy is RegimeId.UNBOUND
        and scale_invariant
        and qt45_separate
    )

    return {
        "schema": "ORIGINS_FIRST_RNA_REACTION_REGIMES_V0_3",
        "caimi_binding": caimi.value,
        "song_binding": song.value,
        "legacy_v02_binding": legacy.value,
        "mass_fraction_scale_invariant": scale_invariant,
        "functional_anchor_registry_separate": qt45_separate,
        "kinetic_parameters_fitted": False,
        "topology_used": False,
        "zeta_used": False,
        "replication_used": False,
        "verdict": (
            "PASS_REACTION_REGIME_REPRESENTATION"
            if passed
            else "FAIL_REACTION_REGIME_REPRESENTATION"
        ),
    }
