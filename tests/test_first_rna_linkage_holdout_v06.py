from origins.analysis.first_rna_linkage_holdout_v06 import (
    EMPIRICAL_3P5_FRACTION,
    frozen_model_linkage_capability,
    run_linkage_holdout_v06,
)


def test_empirical_linkage_targets_are_frozen():
    assert EMPIRICAL_3P5_FRACTION == {"A": 0.74, "U": 0.61}


def test_frozen_parent_has_no_explicit_linkage_state():
    capability = frozen_model_linkage_capability()
    assert capability["profile_has_linkage"] is False
    assert capability["state_has_linkage"] is False
    assert capability["output_has_linkage"] is False
    assert capability["module_has_linkage"] is False
    assert capability["explicit_linkage_state"] is False


def test_linkage_holdout_fails_structurally_without_fake_prediction():
    rows, summary = run_linkage_holdout_v06()

    assert len(rows) == 2
    assert set(rows["base"]) == {"A", "U"}
    assert rows["model_3p5_fraction"].isna().all()
    assert (rows["numerical_comparison"] == "NOT_RUN").all()
    assert (rows["status"] == "MISSING_LINKAGE_STATE").all()

    assert summary["status"] == "FAIL_MISSING_LINKAGE_STATE"
    assert summary["numerical_comparison_run"] is False
    assert summary["linkage_parameter_added"] is False
    assert summary["global_rates_retuned"] is False
    assert summary["base_extension_multipliers_retuned"] is False
    assert summary["empirical_fraction_copied_into_prediction"] is False


def test_linkage_holdout_preserves_causal_boundary():
    _, summary = run_linkage_holdout_v06()
    assert summary["geometry_used"] is False
    assert summary["zeta_used"] is False
    assert summary["replication_used"] is False
