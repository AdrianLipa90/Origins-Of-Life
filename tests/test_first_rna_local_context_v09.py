import pytest

from origins.analysis.first_rna_local_context_v09 import (
    NUMERIC_CONTEXT_PARAMETERS,
    PU_2P5_PU,
    PU_2P5_PY,
    PU_3P5_PU,
    PU_3P5_PY,
    DimerContext,
    GrowthContext,
    ordinal_relation,
    run_local_context_calibration_v09,
)


def test_source_partial_order_is_preserved_without_scores():
    assert ordinal_relation(PU_3P5_PY, PU_3P5_PU) == "GREATER"
    assert ordinal_relation(PU_3P5_PU, PU_2P5_PY) == "EQUAL"
    assert ordinal_relation(PU_2P5_PY, PU_2P5_PU) == "GREATER"
    assert NUMERIC_CONTEXT_PARAMETERS == {}


def test_growth_context_tracks_terminal_linkage_and_incoming_identity():
    context = GrowthContext(
        terminal_class="Pu",
        previous_linkage="3p5",
        incoming_class="Py",
    )
    assert context.terminal_class == "Pu"
    assert context.previous_linkage == "3p5"
    assert context.incoming_class == "Py"
    assert not hasattr(context, "chain_length")


def test_invalid_symbolic_context_fails_closed():
    with pytest.raises(ValueError):
        DimerContext("invalid", "3p5", "Py")
    with pytest.raises(ValueError):
        GrowthContext("Pu", "invalid", "Py")


def test_v09_is_representation_calibration_not_prediction():
    rows, summary = run_local_context_calibration_v09()

    assert len(rows) == 4
    assert summary["status"] == "PASS_LOCAL_CONTEXT_REPRESENTATION_CALIBRATION"
    assert summary["representation_calibration_not_prediction"] is True
    assert summary["predictive_validation_claim"] is False
    assert summary["source_equality_preserved"] is True
    assert summary["source_strict_inequalities_preserved"] is True
    assert summary["growth_context_complete"] is True
    assert summary["chain_length_term_present"] is False
    assert summary["numeric_context_parameter_added"] is False
    assert summary["global_rates_retuned"] is False
    assert summary["base_extension_multipliers_retuned"] is False
    assert summary["geometry_used"] is False
    assert summary["zeta_used"] is False
    assert summary["replication_used"] is False
