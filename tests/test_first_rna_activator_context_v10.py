from origins.analysis.first_rna_activator_context_v10 import (
    EMPIRICAL_3P5_BY_ACTIVATOR,
    frozen_v09_activator_capability,
    run_activator_context_audit_v10,
)


def test_external_activator_observations_are_frozen():
    assert EMPIRICAL_3P5_BY_ACTIVATOR["A"] == {
        "imidazole": 0.67,
        "1-methyladenine": 0.74,
    }
    assert EMPIRICAL_3P5_BY_ACTIVATOR["U"] == {
        "imidazole": 0.20,
        "1-methyladenine": 0.61,
    }


def test_frozen_v09_has_no_activator_context():
    capability = frozen_v09_activator_capability()
    assert capability["dimer_has_activator"] is False
    assert capability["growth_has_activator"] is False
    assert capability["explicit_activator_context"] is False


def test_v10_fails_structurally_without_fake_prediction():
    rows, summary = run_activator_context_audit_v10()

    assert len(rows) == 2
    assert (rows["model_condition_comparison"] == "NOT_RUN").all()
    assert (rows["status"] == "MISSING_ACTIVATOR_CONTEXT").all()
    assert summary["status"] == "FAIL_MISSING_ACTIVATOR_CONTEXT"
    assert summary["numerical_comparison_run"] is False
    assert summary["activator_parameter_added"] is False
    assert summary["predictive_validation_claim"] is False


def test_v10_preserves_causal_boundary():
    _, summary = run_activator_context_audit_v10()
    assert summary["global_rates_retuned"] is False
    assert summary["base_extension_multipliers_retuned"] is False
    assert summary["local_context_relation_retuned"] is False
    assert summary["geometry_used"] is False
    assert summary["zeta_used"] is False
    assert summary["replication_used"] is False
