import pytest

from origins.analysis.first_rna_candidate import run_first_rna_candidate_ablation


def test_candidate_ablation_never_promotes_function_in_safe_mode():
    df = run_first_rna_candidate_ablation(
        seeds=[77],
        hours=2.0,
        dt_h=0.5,
        topo_strength=0.25,
    )

    assert set(df["mode"]) == {
        "chemistry_only",
        "legacy_geometry_candidate",
    }
    assert set(df["functional_replication_status"]) == {
        "UNRESOLVED_SEQUENCE_AND_ACTIVITY"
    }
    assert df["functional_replicator_t"].isna().all()
    assert (df["final_functional_replicators"] == 0.0).all()

    # Both arms have identical declared external nucleotide input, so kinetic
    # redistribution must not alter total nucleotide-equivalent material.
    totals = df.set_index("mode")["final_total_nucleotide_units"]
    assert totals["chemistry_only"] == pytest.approx(
        totals["legacy_geometry_candidate"],
        rel=1e-10,
        abs=1e-10,
    )
