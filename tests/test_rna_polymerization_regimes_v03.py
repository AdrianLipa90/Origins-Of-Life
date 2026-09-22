import numpy as np
import pytest

from origins.chemistry.rna_polymerization_regimes import (
    CAIMI_2025_AUGC_PH11,
    FUNCTIONAL_ANCHORS,
    QT45_FUNCTIONAL_ANCHOR,
    RegimeId,
    caimi_signature,
    chain_mass_fractions,
    classify_regime,
    legacy_first_rna_v02_signature,
    polymerized_mass_fraction,
    representation_gate_receipt,
    song_ump_signature,
    tail_mass_fraction,
)


def test_empirical_signatures_bind_uniquely():
    assert classify_regime(caimi_signature()) is RegimeId.CAIMI_2025_AUGC_PH11
    assert classify_regime(song_ump_signature()) is RegimeId.SONG_2024_UMP_HOT_ACID


def test_legacy_candidate_signature_fails_closed_as_unbound():
    assert classify_regime(legacy_first_rna_v02_signature()) is RegimeId.UNBOUND


def test_mass_fraction_is_exactly_scale_invariant():
    counts = np.array([100.0, 20.0, 5.0, 1.0, 0.25])
    f1 = chain_mass_fractions(counts)
    f2 = chain_mass_fractions(counts * 1e12)
    np.testing.assert_array_equal(f1, f2)
    assert np.sum(f1) == pytest.approx(1.0)


def test_polymerized_fraction_is_mass_weighted_not_chain_weighted():
    # 10 monomers + one 10-mer => 50% of nucleotide mass is polymerized,
    # although only 1/11 of the chain count is polymeric.
    counts = np.zeros(10)
    counts[0] = 10.0
    counts[9] = 1.0
    assert polymerized_mass_fraction(counts) == pytest.approx(0.5)


def test_tail_mass_fraction_uses_nucleotide_mass():
    counts = np.zeros(10)
    counts[0] = 10.0
    counts[9] = 1.0
    assert tail_mass_fraction(counts, 10) == pytest.approx(0.5)
    assert tail_mass_fraction(counts, 11) == 0.0


def test_caimi_anchor_does_not_silently_mix_ph10_tail():
    assert CAIMI_2025_AUGC_PH11.observables["max_detected_nt"] == 8
    assert CAIMI_2025_AUGC_PH11.observables["eight_mer_mass_fraction"] == pytest.approx(0.005)
    assert any("pH-10" in note for note in CAIMI_2025_AUGC_PH11.limitations)


def test_qt45_is_downstream_function_reference_only():
    assert QT45_FUNCTIONAL_ANCHOR.length_nt == 45
    assert QT45_FUNCTIONAL_ANCHOR in FUNCTIONAL_ANCHORS
    assert "arbitrary 45-mer" in QT45_FUNCTIONAL_ANCHOR.warning


def test_representation_gate_passes_without_fitting_kinetics():
    receipt = representation_gate_receipt()
    assert receipt["verdict"] == "PASS_REACTION_REGIME_REPRESENTATION"
    assert receipt["kinetic_parameters_fitted"] is False
    assert receipt["topology_used"] is False
    assert receipt["zeta_used"] is False
    assert receipt["replication_used"] is False
